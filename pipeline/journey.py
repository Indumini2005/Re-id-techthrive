"""Where each person has been, where they will go next, and for how long.

Three things are built from one stream of sightings (identity, camera, time):

* **Journeys.** A sighting extends that person's current visit to a camera,
  or opens a new one after a gap of ``visit_gap`` seconds. The visit list is
  the journey timeline the dashboard draws.
* **Next-camera prediction.** Every completed walk from camera X to camera Y
  adds its walking time to a route model. When someone leaves X, the model
  predicts where they will appear next and in what time window; when they
  do appear, the prediction is scored as a hit or a miss. The running hit
  rate is reported, so the claim stays honest - it is a learned average of
  real walks, not a trained model, and it says so when it has not learned a
  route yet.
* **Dwell and loitering.** How long someone has been continuously in view
  at a camera; past ``loiter_s`` seconds a loitering alert fires once per
  visit.

Two cameras that overlap (both see the same spot) produce simultaneous
visits rather than walks between them. That is recognised and not counted
as a transit - otherwise every multi-angle sighting would teach the model a
walking time of zero.

Pure Python + numpy. Timestamps are wall-clock seconds.
"""

from collections import deque

import numpy as np

import config


class Visit:
    __slots__ = ("camera", "start", "end", "loitered")

    def __init__(self, camera, t):
        self.camera = camera
        self.start = float(t)
        self.end = float(t)
        self.loitered = False

    def duration(self):
        return self.end - self.start


class TransitModel:
    """Learned walking times between cameras."""

    def __init__(self, max_samples=50):
        self.samples = {}             # (from, to) -> deque of seconds
        self.max_samples = max_samples

    def add(self, frm, to, seconds):
        key = (frm, to)
        if key not in self.samples:
            self.samples[key] = deque(maxlen=self.max_samples)
        self.samples[key].append(float(seconds))

    @staticmethod
    def window(samples):
        """(eta, lo, hi) for a list of walking times.

        Few walks: a wide margin around what was seen. Five or more: the
        10th-90th percentile band, widened by a quarter of its width. A
        heuristic, stated as such; the hit rate shows whether it works.
        """
        s = np.asarray(samples, dtype=np.float64)
        eta = float(np.median(s))
        if len(s) >= 5:
            lo, hi = float(np.percentile(s, 10)), float(np.percentile(s, 90))
            spread = max(hi - lo, 2.0)
            lo, hi = lo - 0.25 * spread, hi + 0.25 * spread
        else:
            lo, hi = float(s.min()) * 0.6 - 1.0, float(s.max()) * 1.5 + 2.0
        return eta, max(0.0, lo), hi

    def predict(self, frm):
        """Most likely next camera after leaving ``frm``, or None."""
        routes = [(to, s) for (f, to), s in self.samples.items() if f == frm and s]
        if not routes:
            return None
        to, samples = max(routes, key=lambda r: len(r[1]))
        eta, lo, hi = self.window(list(samples))
        return {"to": to, "eta_s": eta, "lo_s": lo, "hi_s": hi, "n": len(samples)}

    def summary(self):
        out = []
        for (frm, to), s in sorted(self.samples.items()):
            if not s:
                continue
            eta, lo, hi = self.window(list(s))
            out.append({"from": frm, "to": to, "n": len(s), "median_s": round(eta, 1),
                        "lo_s": round(lo, 1), "hi_s": round(hi, 1)})
        return out


class Journey:
    def __init__(self, key, name, kind):
        self.key = key
        self.name = name
        self.kind = kind              # "enrolled" | "visitor"
        self.visits = []
        self.transitions = deque(maxlen=20)
        self.prediction = None        # dict, see JourneyBook.tick
        self.departure_handled = None # start time of the visit last predicted from

    @property
    def last_seen(self):
        return max(v.end for v in self.visits) if self.visits else 0.0

    @property
    def first_seen(self):
        return min(v.start for v in self.visits) if self.visits else 0.0

    def active_visits(self, t, gap):
        return [v for v in self.visits if float(t) - v.end <= gap]


class JourneyBook:
    def __init__(self, visit_gap=None, max_transit=None, loiter_s=None, keep_s=None):
        self.visit_gap = float(visit_gap if visit_gap is not None
                               else getattr(config, "VISIT_GAP_S", 3.0))
        self.max_transit = float(max_transit if max_transit is not None
                                 else getattr(config, "TRANSIT_MAX_S", 180.0))
        self.loiter_s = float(loiter_s if loiter_s is not None
                              else getattr(config, "LOITER_SECONDS", 30.0))
        self.keep_s = float(keep_s if keep_s is not None
                            else getattr(config, "VISITOR_TTL_S", 600.0))
        self.journeys = {}
        self.model = TransitModel()
        self.stats = {"made": 0, "hits": 0, "misses": 0}

    # -- sightings -----------------------------------------------------------
    def sight(self, key, name, kind, camera, t, since=None, loitered=False):
        """Record that ``key`` was seen on ``camera`` at ``t``.

        ``since`` is when the tracker first saw this person on this camera.
        Identity is often settled a cycle or two after someone appears, and
        the visit - and so any walking time measured into it - should start
        when they appeared, not when they were recognised. ``loitered``
        carries over a loitering alert already raised on the anonymous
        track, so it is not raised twice.

        Returns a list of event dicts (arrival, prediction outcome,
        loitering). Idempotent for repeated calls with the same time.
        """
        t = float(t)
        events = []
        j = self.journeys.get(key)
        if j is None:
            j = self.journeys[key] = Journey(key, name, kind)
        j.name, j.kind = name, kind

        start = t if since is None else min(t, float(since))
        same_cam = [v for v in j.visits if v.camera == camera]
        current = same_cam[-1] if same_cam else None
        if current is not None and (t - current.end <= self.visit_gap
                                    or start <= current.end + self.visit_gap):
            current.end = max(current.end, t)
        else:
            events.extend(self._arrive(j, camera, start, t))
            current = Visit(camera, start)
            current.end = t
            current.loitered = bool(loitered)
            j.visits.append(current)
            if len(j.visits) > 60:
                j.visits = j.visits[-60:]

        if not current.loitered and current.duration() >= self.loiter_s:
            current.loitered = True
            events.append({"kind": "loiter", "key": key, "camera": camera,
                           "text": "%s has been at %s for %d s" % (name, camera, int(current.duration()))})
        return events

    def _arrive(self, j, camera, start, t):
        """A new visit is starting at ``start`` (seen until ``t``). Work out
        whether it completes a walk from another camera."""
        events = []
        elsewhere_now = [v for v in j.visits if v.camera != camera and t - v.end <= self.visit_gap]
        if elsewhere_now:
            # Seen on another camera right now: overlapping views, not a walk.
            return events

        previous = max(j.visits, key=lambda v: v.end) if j.visits else None
        pred = j.prediction
        if previous is None:
            return events

        transit = start - previous.end
        if transit < 0:
            return events                                 # overlapping visits
        if previous.camera != camera and transit <= self.max_transit:
            if pred and pred.get("status") in ("pending", "overdue") and pred.get("from") == previous.camera:
                hit = pred["to"] == camera and pred["lo_s"] <= transit <= pred["hi_s"]
                pred["status"] = "hit" if hit else "miss"
                pred["actual_s"] = round(transit, 1)
                pred["actual_to"] = camera
                self.stats["hits" if hit else "misses"] += 1
                if hit:
                    text = "%s reached %s after %.1f s - predicted %.0f s (%.0f-%.0f s)" % (
                        j.name, camera, transit, pred["eta_s"], pred["lo_s"], pred["hi_s"])
                elif pred["to"] != camera:
                    text = "%s went to %s, not the predicted %s" % (j.name, camera, pred["to"])
                else:
                    text = "%s reached %s after %.1f s - outside the predicted %.0f-%.0f s" % (
                        j.name, camera, transit, pred["lo_s"], pred["hi_s"])
                events.append({"kind": "prediction_hit" if hit else "prediction_miss",
                               "key": j.key, "camera": camera, "text": text})
            # Learn from this walk only after scoring the prediction, so a
            # walk never helps predict itself.
            self.model.add(previous.camera, camera, transit)
            j.transitions.append({"from": previous.camera, "to": camera,
                                  "transit_s": round(transit, 1), "at": t})
            events.append({"kind": "arrival", "key": j.key, "camera": camera,
                           "text": "%s moved %s -> %s in %.1f s" % (j.name, previous.camera, camera, transit)})
        elif pred and pred.get("status") in ("pending", "overdue") and previous.camera == camera:
            if transit < pred["lo_s"]:
                # Back before they could have reached anywhere else: they
                # never really left (stepped behind something). Not a test of
                # the prediction either way, so it is withdrawn, not scored.
                pred["status"] = "cancelled"
                self.stats["made"] -= 1
            else:
                pred["status"] = "miss"
                pred["actual_to"] = camera
                self.stats["misses"] += 1
                events.append({"kind": "prediction_miss", "key": j.key, "camera": camera,
                               "text": "%s came back to %s instead of going to %s"
                                       % (j.name, camera, pred.get("to"))})
        return events

    # -- time passing ----------------------------------------------------------
    def tick(self, t):
        """Open, age and close predictions; forget stale journeys."""
        t = float(t)
        events = []
        for key in list(self.journeys):
            j = self.journeys[key]
            if not j.visits:
                continue
            if t - j.last_seen > self.keep_s:
                del self.journeys[key]
                continue

            if j.active_visits(t, self.visit_gap):
                continue                                  # still in view

            last = max(j.visits, key=lambda v: v.end)
            if j.departure_handled != last.start:
                j.departure_handled = last.start
                guess = self.model.predict(last.camera)
                if guess is None:
                    j.prediction = {"status": "learning", "from": last.camera,
                                    "departed": last.end}
                else:
                    guess.update({"status": "pending", "from": last.camera,
                                  "departed": last.end})
                    j.prediction = guess
                    self.stats["made"] += 1
                    events.append({"kind": "prediction", "key": key, "camera": guess["to"],
                                   "text": "%s left %s - expected at %s in ~%.0f s (%.0f-%.0f s)" % (
                                       j.name, last.camera, guess["to"], guess["eta_s"],
                                       guess["lo_s"], guess["hi_s"])})

            p = j.prediction
            if p and p.get("status") in ("pending", "overdue"):
                late_by = t - (p["departed"] + p["hi_s"])
                if late_by > 0 and p["status"] == "pending":
                    p["status"] = "overdue"
                    events.append({"kind": "overdue", "key": key, "camera": p["to"],
                                   "text": "%s is overdue at %s" % (j.name, p["to"])})
                if late_by > max(30.0, p["hi_s"]):
                    p["status"] = "lost"
                    self.stats["misses"] += 1
        return events

    # -- identity changes -----------------------------------------------------
    def merge(self, keep_key, drop_key, name=None, kind=None):
        """Fold one journey into another: two names turned out to be one
        person, or a visitor was identified as someone enrolled."""
        drop = self.journeys.pop(drop_key, None)
        if drop is None:
            return
        keep = self.journeys.get(keep_key)
        if keep is None:
            drop.key = keep_key
            if name:
                drop.name = name
            if kind:
                drop.kind = kind
            self.journeys[keep_key] = drop
            return
        keep.visits = sorted(keep.visits + drop.visits, key=lambda v: v.start)[-60:]
        for tr in drop.transitions:
            keep.transitions.append(tr)
        if keep.prediction is None:
            keep.prediction = drop.prediction

    # -- read-outs -------------------------------------------------------------
    def current_dwell(self, key, t):
        j = self.journeys.get(key)
        if j is None:
            return None, False
        active = j.active_visits(t, self.visit_gap)
        if not active:
            return None, False
        longest = max(active, key=lambda v: v.duration())
        return longest.duration(), any(v.loitered for v in active)

    def prediction_stats(self):
        pending = sum(1 for j in self.journeys.values()
                      if j.prediction and j.prediction.get("status") in ("pending", "overdue"))
        decided = self.stats["hits"] + self.stats["misses"]
        return {
            "made": self.stats["made"],
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "pending": pending,
            "accuracy": round(self.stats["hits"] / float(decided), 3) if decided else None,
        }
