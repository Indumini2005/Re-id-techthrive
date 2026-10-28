"""Scene analytics: who is where, who they are, and where they are going.

This sits on top of detection, recognition and cross-view fusion, and adds
what an operator - or an exhibition audience - actually watches:

* a stable ID per person per camera, with Kalman prediction of where they
  are heading (pipeline/tracker.py);
* re-identification of people who were never enrolled, as "Visitor #N"
  (pipeline/visitors.py);
* each person's journey across the cameras, the learned walking time
  between cameras, and a scored prediction of where they will appear next
  (pipeline/journey.py);
* dwell time, and a loitering alert after LOITER_SECONDS;
* confidence percentages, only when calibrate.py has measured them
  (pipeline/calibration.py).

Used identically by the Nano's main_pipeline.py (two cameras, one process)
and by hub/server.py (two capture laptops), so both drive the same
dashboard. The call sequence per cycle is:

    tracks = analytics.track(camera, t, boxes, confs, frame_size)   # per camera
    for each track: if analytics.should_check_face(trk, t): ... face_result(...)
    fused = CrossViewFusion.process(observations)                   # existing
    analytics.resolve(t, fused)
    state = analytics.state(t)

Times are wall-clock seconds. No OpenCV: thumbnails arrive already encoded.
"""

import time
from collections import Counter, deque

import numpy as np

import config
from .calibration import ScoreCalibrator
from .fusion import fuse_embeddings
from .journey import JourneyBook
from .tracker import MultiTracker
from .vectors import l2_normalize
from .visitors import VisitorGallery


def _r(v, nd=1):
    return None if v is None else round(float(v), nd)


def _clock(ts):
    return time.strftime("%H:%M:%S", time.localtime(ts))


class SceneAnalytics:
    def __init__(self, calibrator=None, embed_refresh_s=None, face_retry_s=None,
                 visitor_threshold=None, loiter_s=None, visit_gap=None,
                 visitor_ttl_s=None):
        self.trackers = {}
        self.frames = {}
        self.calibrator = calibrator if calibrator is not None else ScoreCalibrator()
        self.embed_refresh_s = float(embed_refresh_s if embed_refresh_s is not None
                                     else getattr(config, "TRACK_EMBED_REFRESH_S", 2.0))
        self.face_retry_s = float(face_retry_s if face_retry_s is not None
                                  else getattr(config, "TRACK_FACE_RETRY_S", 1.0))
        self.visitors = VisitorGallery(threshold=visitor_threshold, ttl_s=visitor_ttl_s)
        self.journeys = JourneyBook(visit_gap=visit_gap, loiter_s=loiter_s, keep_s=visitor_ttl_s)
        self.events = deque(maxlen=60)
        self.thumbs = {}            # identity key -> latest face thumbnail (base64)
        self.last_identity = {}     # identity key -> last decision dict
        self.counters = {"faces_embedded": 0, "faces_reused": 0,
                         "face_checks": 0, "face_checks_skipped": 0}
        self.last_t = 0.0

    # -- settings ----------------------------------------------------------------
    @property
    def loiter_s(self):
        return self.journeys.loiter_s

    def apply_settings(self, settings):
        if "visitor_threshold" in settings:
            self.visitors.threshold = float(settings["visitor_threshold"])
        if "loiter_seconds" in settings:
            self.journeys.loiter_s = float(settings["loiter_seconds"])

    # -- phase 1: per camera -------------------------------------------------
    def tracker(self, camera):
        trk = self.trackers.get(camera)
        if trk is None:
            trk = self.trackers[camera] = MultiTracker(camera)
        return trk

    def track(self, camera, t, boxes, confs=None, frame_size=None):
        """Associate this camera's detections with its tracks."""
        tracks = self.tracker(camera).update(boxes, t, confs)
        info = self.frames.setdefault(camera, {"w": 0, "h": 0, "t": 0.0})
        if frame_size and frame_size[0] and frame_size[1]:
            info["w"], info["h"] = int(frame_size[0]), int(frame_size[1])
        info["t"] = float(t)
        return tracks

    def _wants_face(self, trk, t):
        """Is it worth running face detection + ArcFace on this track now?

        This is track-then-recognise, the main speed-up on the Nano: a person
        whose identity is settled keeps it while tracked, and is re-embedded
        only every ``embed_refresh_s`` - or immediately if they overlapped
        someone, since that is where a tracker can swap two people. After a
        crop yields no usable face, the next attempt waits ``face_retry_s``.
        """
        t = float(t)
        if trk.ambiguous:
            return True
        if t < trk.next_face_try:
            return False
        if trk.embedding is None or trk.n_embeddings < 2 or trk.identity is None:
            return True
        if self.embed_refresh_s <= 0:
            return True
        return (t - trk.embedding_t) >= self.embed_refresh_s

    def _count(self, trk, checked):
        if checked:
            self.counters["face_checks"] += 1
        elif trk.embedding is not None:
            self.counters["faces_reused"] += 1
        else:
            self.counters["face_checks_skipped"] += 1

    def should_check_face(self, trk, t):
        """Single-track form of select_face_checks, with no budget."""
        decision = self._wants_face(trk, t)
        self._count(trk, decision)
        return decision

    def select_face_checks(self, tracks, t, budget=None):
        """Which of this frame's tracked people get face recognition now.

        Everyone is tracked - that is cheap - but recognition is rationed to
        ``budget`` people per camera per cycle, because it is the expensive
        part. Among those who want it: people never recognised first (largest,
        i.e. nearest, first), then people who just crossed someone, then the
        longest since their last check. The rest wait for a later cycle.
        Returns the set of indices into ``tracks`` to check.
        """
        t = float(t)
        wanted = [(i, trk) for i, trk in enumerate(tracks) if self._wants_face(trk, t)]

        def priority(item):
            trk = item[1]
            area = max(0.0, trk.box[2] - trk.box[0]) * max(0.0, trk.box[3] - trk.box[1])
            age = (t - trk.embedding_t) if trk.embedding_t is not None else 0.0
            return (trk.embedding is None, trk.ambiguous, age, area)

        wanted.sort(key=priority, reverse=True)
        if budget is not None and budget > 0:
            wanted = wanted[:int(budget)]
        chosen = {i for i, _ in wanted}
        for i, trk in enumerate(tracks):
            self._count(trk, i in chosen)
        return chosen

    def face_result(self, trk, t, embedding=None, thumb=None):
        """Record the outcome of a face check on ``trk``."""
        t = float(t)
        if embedding is None:
            trk.next_face_try = t + self.face_retry_s
            return
        trk.embedding = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(-1))
        trk.embedding_t = t
        trk.n_embeddings += 1
        trk.next_face_try = t
        if thumb:
            trk.face_thumb = thumb
        self.counters["faces_embedded"] += 1

    @staticmethod
    def embedding_for(trk):
        """(embedding, cached) to put in a ViewObservation for this track."""
        if trk is None or trk.embedding is None:
            return None, False
        return trk.embedding, trk.embedding_t != trk.last_update

    # -- phase 2: identities, journeys ------------------------------------------
    def resolve(self, t, fused_tracks):
        """Decide identities for the fused tracks, then update journeys,
        dwell, loitering and predictions. Returns the new events."""
        t = float(t)
        self.last_t = t
        events = []

        for key in self.visitors.expire(t):
            self.thumbs.pop(key, None)

        for ft in fused_tracks or []:
            members = [o.track for o in ft.observations if getattr(o, "track", None) is not None]
            fresh = [m for m in members if m.embedding is not None
                     and (m.decided_t is None or m.embedding_t > m.decided_t)]
            if not fresh:
                continue                     # nothing new: identities carry over
            events.extend(self._decide(ft, members, fresh, t))

        for camera, tracker in self.trackers.items():
            tracker.prune(t)
            for trk in tracker.tracks:
                if trk.last_update != tracker.last_t:
                    continue                 # coasting: not actually seen now
                if trk.identity:
                    ident = trk.identity
                    events.extend(self.journeys.sight(
                        ident["key"], ident["name"], ident["kind"], camera,
                        trk.last_update, since=trk.first_seen, loitered=trk.loiter_flagged))
                elif not trk.loiter_flagged and trk.dwell(trk.last_update) >= self.loiter_s:
                    trk.loiter_flagged = True
                    events.append({"kind": "loiter", "key": trk.id, "camera": camera,
                                   "text": "Person %s has been at %s for %d s"
                                           % (trk.label, camera, int(trk.dwell(trk.last_update)))})

        events.extend(self.journeys.tick(t))
        for ev in events:
            ev.setdefault("ts", t)
            ev["at"] = _clock(ev["ts"])
            self.events.append(ev)
        return events

    def _decide(self, ft, members, fresh, t):
        events = []
        embs = [o.embedding for o in ft.observations if o.embedding is not None]
        emb = ft.fused_embedding if ft.fused_embedding is not None else fuse_embeddings(embs, "mean")
        thumb = next((m.face_thumb for m in fresh if m.face_thumb), None)
        cams = "+".join(sorted({m.id.split(":")[0] for m in members}))

        if ft.match is not None and ft.match.is_match:
            name = ft.match.identity
            decision = {"key": "id:" + name, "name": name, "kind": "enrolled",
                        "score": round(float(ft.match.score), 4),
                        "confidence": self.calibrator.probability(ft.match.score)}
            for m in members:
                prev = m.identity
                if prev and prev["kind"] == "visitor":
                    # Someone the system had only known as a visitor turns out
                    # to be enrolled: keep their journey, under their name.
                    self.journeys.merge(decision["key"], prev["key"], name, "enrolled")
                    self.visitors.remove(prev["key"])
                    self.thumbs.pop(prev["key"], None)
                    events.append({"kind": "identified", "key": decision["key"], "camera": cams,
                                   "text": "%s identified as %s" % (prev["name"], name)})
        else:
            held = [m.identity["key"] for m in members
                    if m.identity and m.identity["kind"] == "visitor"
                    and self.visitors.get(m.identity["key"]) is not None]
            current = Counter(held).most_common(1)[0][0] if held else None
            # Fusion decided these views show one person. If they carry
            # different visitor names, those names are the same person.
            for other in sorted(set(held) - {current}):
                keep_v, drop_v = self.visitors.get(current), self.visitors.get(other)
                if keep_v is not None and drop_v is not None:
                    self.visitors.merge(keep_v, drop_v)
                    self.journeys.merge(current, other)
                    self.thumbs.pop(other, None)
                    events.append({"kind": "merged", "key": current, "camera": cams,
                                   "text": "%s and %s are the same person (seen together on %s)"
                                           % (keep_v.name, drop_v.name, cams)})

            seen_before = self.journeys.journeys.get(current) if current else None
            visitor, score, outcome, merged = self.visitors.assign(emb, t, current, thumb)
            for mk in merged:
                self.journeys.merge(visitor.key, mk)
                self.thumbs.pop(mk, None)
                events.append({"kind": "merged", "key": visitor.key, "camera": cams,
                               "text": "Visitor #%s is the same person as %s"
                                       % (mk.split(":")[1], visitor.name)})
            decision = {"key": visitor.key, "name": visitor.name, "kind": "visitor",
                        "score": None if score is None else round(float(score), 4),
                        "confidence": self.calibrator.probability(score)}

            if outcome == "new":
                events.append({"kind": "visitor_new", "key": visitor.key, "camera": cams,
                               "text": "New visitor on %s: %s" % (cams, visitor.name)})
            elif outcome == "returned":
                j = self.journeys.journeys.get(visitor.key) or seen_before
                where = ""
                if j and j.visits:
                    last = max(j.visits, key=lambda v: v.end)
                    where = " - last seen on %s %d s ago" % (last.camera, int(t - last.end))
                events.append({"kind": "visitor_return", "key": visitor.key, "camera": cams,
                               "text": "%s recognised again on %s%s" % (visitor.name, cams, where)})

        for m in members:
            m.identity = decision
            m.decided_t = m.embedding_t if m.embedding_t is not None else t
        self.last_identity[decision["key"]] = decision
        if thumb:
            self.thumbs[decision["key"]] = thumb
        return events

    # -- read-out ------------------------------------------------------------------
    def view_state(self, camera, t, at=None):
        """One camera's people, for drawing.

        With ``at`` set, every box is moved to where the Kalman filter puts
        that person at that moment - used to keep boxes on people in frames
        shown between detections (``--detect_every 2`` and above).
        """
        tracker = self.trackers.get(camera)
        info = self.frames.get(camera, {})
        rows = []
        for trk in (tracker.tracks if tracker else []):
            ident = trk.identity
            if ident:
                dwell, loitering = self.journeys.current_dwell(ident["key"], t)
                if dwell is None:
                    dwell, loitering = trk.dwell(trk.last_update), trk.loiter_flagged
            else:
                dwell, loitering = trk.dwell(trk.last_update), trk.loiter_flagged
            coasting = trk.last_update != tracker.last_t
            ref = float(at) if at is not None else (tracker.last_t or t)
            if at is not None or coasting:
                box = trk.box_at(ref)
            else:
                box = trk.box
            vx, vy = trk.velocity
            rows.append({
                "id": trk.label,
                "box": [_r(v) for v in box],
                "pred_box": [_r(v) for v in trk.box_at(ref + 1.0)],
                "velocity": [_r(vx), _r(vy)],
                "trail": [[_r(x), _r(y)] for _, x, y in list(trk.trail)[-12:]],
                "dwell_s": _r(dwell),
                "loitering": bool(loitering),
                "coasting": bool(coasting),
                "confirmed": bool(trk.confirmed),
                "has_face": trk.embedding is not None,
                "identity": dict(ident) if ident else None,
            })
        return {"frame_w": info.get("w", 0), "frame_h": info.get("h", 0), "tracks": rows}

    def state(self, t=None):
        """Everything the dashboard and HUD draw. JSON-safe."""
        t = float(t if t is not None else self.last_t)
        views = {camera: self.view_state(camera, t) for camera in self.trackers}

        people = []
        gap = self.journeys.visit_gap
        for key, j in self.journeys.journeys.items():
            if not j.visits:
                continue
            active = j.active_visits(t, gap)
            dwell, loitering = self.journeys.current_dwell(key, t)
            last = max(j.visits, key=lambda v: v.end)
            ident = self.last_identity.get(key, {})
            pred = None
            if j.prediction:
                p = j.prediction
                pred = {k: (_r(v) if isinstance(v, float) else v) for k, v in p.items()}
                if "departed" in p:
                    pred["elapsed_s"] = _r(t - p["departed"])
                    if "eta_s" in p:
                        pred["remaining_s"] = _r(p["departed"] + p["eta_s"] - t)
            people.append({
                "key": key,
                "name": j.name,
                "kind": j.kind,
                "thumb": self.thumbs.get(key),
                "confidence": ident.get("confidence"),
                "score": ident.get("score"),
                "in_view": bool(active),
                "cameras_now": sorted({v.camera for v in active}),
                "dwell_s": _r(dwell),
                "loitering": bool(loitering),
                "first_seen": _r(j.first_seen, 2),
                "last_seen": _r(j.last_seen, 2),
                "last_seen_ago": _r(t - j.last_seen),
                "last_camera": last.camera,
                "visits": [{"camera": v.camera, "start": _r(v.start, 2), "end": _r(v.end, 2)}
                           for v in j.visits[-20:]],
                "transitions": [dict(tr, at=_r(tr["at"], 2)) for tr in list(j.transitions)[-5:]],
                "prediction": pred,
            })
        people.sort(key=lambda p: (not p["in_view"], -(p["last_seen"] or 0)))

        return {
            "ts": round(t, 3),
            "views": views,
            "people": people[:12],
            "transit": self.journeys.model.summary(),
            "predictions": self.journeys.prediction_stats(),
            "events": list(self.events)[-40:],
            "calibration": self.calibrator.summary(),
            "visitors": {"active": len(self.visitors), "created": self.visitors.created,
                         "ttl_s": self.visitors.ttl_s, "threshold": self.visitors.threshold},
            "efficiency": self.efficiency(),
            "loiter_seconds": self.loiter_s,
        }

    def efficiency(self):
        """How much face work tracking saved: each face check is face
        detection + alignment + ArcFace, the most expensive stages."""
        c = self.counters
        checks = c["face_checks"] + c["faces_reused"] + c["face_checks_skipped"]
        return {
            "faces_embedded": c["faces_embedded"],
            "faces_reused": c["faces_reused"],
            "face_checks_skipped": c["face_checks_skipped"],
            "skipped_pct": round(100.0 * (c["faces_reused"] + c["face_checks_skipped"])
                                 / checks, 1) if checks else None,
        }


def fused_track_payload(track, calibrator=None):
    """One FusedTrack as the dashboard's track row. Shared by the hub and the
    standalone pipeline so both produce exactly the same shape."""
    views = []
    for obs in track.observations:
        trk = getattr(obs, "track", None)
        views.append({
            "camera": obs.camera,
            "box": [int(v) for v in obs.box],
            "conf": round(float(obs.conf), 3),
            "escalated": bool(obs.escalated),
            "cached": bool(getattr(obs, "cached", False)),
            "track_id": trk.label if trk is not None else None,
            "thumb": getattr(obs, "_thumb", None) or (trk.face_thumb if trk is not None else None),
            "solo": {
                "identity": obs.solo_match.identity,
                "score": round(obs.solo_match.score, 4),
                "is_match": obs.solo_match.is_match,
            } if obs.solo_match else None,
        })

    solo_scores = [v["solo"]["score"] for v in views if v["solo"]]
    fused_score = round(track.match.score, 4) if track.match else None
    gain = None
    if fused_score is not None and solo_scores:
        gain = round(fused_score - max(solo_scores), 4)

    member = next((getattr(o, "track", None) for o in track.observations
                   if getattr(o, "track", None) is not None and o.track.identity), None)
    ident = member.identity if member is not None else None
    prob = calibrator.probability if calibrator is not None else (lambda s: None)

    return {
        "cameras": track.cameras,
        "n_views": track.n_views,
        "multiview": track.is_multiview,
        "cross_view_score": round(track.cross_view_score, 4),
        "cross_view_confidence": prob(track.cross_view_score) if track.is_multiview else None,
        "identity": track.match.identity if track.match else None,
        "score": fused_score,
        "confidence": prob(track.match.score) if track.match else None,
        "is_match": bool(track.match and track.match.is_match),
        "agreement": track.agreement,
        "fusion_gain": gain,
        "notes": list(track.notes),
        "views": views,
        "person": ident["name"] if ident else None,
        "person_kind": ident["kind"] if ident else None,
        "person_confidence": ident.get("confidence") if ident else None,
    }
