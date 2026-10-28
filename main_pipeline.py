"""Real-time dual-camera edge inference loop.

    CAM_A ─┐
           ├─ pair in time ─► denoise ─► crop ─► restore ─► embed ─┐
    CAM_B ─┘                                                        │
                                            cross-view association ─┤
                                                 fuse both angles ──┤
                                                   FAISS gallery ◄──┘
                                                          │
                                                  operator alert

Both cameras are read at their native 30 fps on separate threads. Each
inference cycle takes the newest frame from the lead camera, pairs it with
the temporally closest frame from the other, and runs the stack on both. The
two resulting embeddings are compared against *each other* to decide they
show the same person, then fused into a single gallery query.

    python main_pipeline.py                                   # config.CAMERAS
    python main_pipeline.py --source rtsp://.../a --source rtsp://.../b
    python main_pipeline.py --source CAM_A=a.mp4 --source CAM_B=b.mp4 --no_gstreamer
    python main_pipeline.py --dashboard 8765 --headless       # serve the web UI

On top of identification, pipeline/analytics.py tracks every person with a
Kalman filter, re-identifies strangers as "Visitor #N", records each
person's journey between the cameras, predicts where they will appear next,
and measures dwell time and loitering. With --dashboard the board also
serves the web UI, so any laptop or phone on the same network can watch it.

Everything runs locally; nothing is transmitted off the local network.
"""

import argparse
import threading
import time
from collections import deque
from datetime import datetime
from typing import Dict, List

import cv2
import numpy as np

import config
from pipeline.analytics import SceneAnalytics, fused_track_payload
from pipeline.fusion import CrossViewFusion, ViewObservation
from pipeline.sources import SyncedCapture, cameras_from_args, gstreamer_pipeline  # noqa: F401
from pipeline.utils import StageTimer, downscale, draw_box, draw_hud, draw_scene_view

# One colour per track, reused across both views so the operator can see at a
# glance which box in CAM_A corresponds to which box in CAM_B.
TRACK_COLORS = [
    (0, 220, 255), (120, 255, 120), (255, 180, 80),
    (200, 120, 255), (120, 220, 255), (255, 255, 120),
]
ALERT_COLOR = (0, 0, 255)


class Stages:
    """Loads each model once, tolerating missing weights."""

    def __init__(self, enable: dict, args=None):
        self.denoiser = self.detector = self.restorer = None
        self.embedder = self.index = None
        self.enabled = dict(enable)

        if self.enabled.get("denoise"):
            try:
                from pipeline.stage1_denoise import get_denoiser
                self.denoiser = get_denoiser(
                    profile=getattr(args, "denoise_profile", None))
                print("[stage1] restorer ready: %s" % self.denoiser.describe())
            except Exception as exc:
                print(f"[stage1] disabled: {exc}")
                self.enabled["denoise"] = False

        if self.enabled.get("detect"):
            try:
                from pipeline.stage2_detect import get_detector
                self.detector = get_detector()
                print("[stage2] YOLOv8 detector ready")
            except Exception as exc:
                print(f"[stage2] disabled: {exc}")
                self.enabled["detect"] = False

        if self.enabled.get("restore"):
            try:
                from pipeline.stage3_restore import get_restorer
                self.restorer = get_restorer()
                print("[stage3] GFPGAN restorer ready")
            except Exception as exc:
                print(f"[stage3] disabled: {exc}")
                self.enabled["restore"] = False

        if self.enabled.get("match"):
            try:
                from pipeline.stage4_embed import get_embedder
                from pipeline.stage5_match import get_index
                self.embedder = get_embedder()
                self.index = get_index()
                print(f"[stage4/5] ArcFace + FAISS ready: {self.index.stats()}")
            except Exception as exc:
                print(f"[stage4/5] disabled: {exc}")
                self.enabled["match"] = False


def wall_time(monotonic_ts: float) -> float:
    """Frame timestamps are time.monotonic(); analytics and the UI work in
    wall-clock seconds."""
    return time.time() - (time.monotonic() - monotonic_ts)


def process_view(
    frame_bgr: np.ndarray,
    camera: str,
    stages: Stages,
    timer: StageTimer,
    args,
    analytics: SceneAnalytics = None,
    t_wall: float = None,
    face_thumbs: bool = False,
) -> tuple:
    """Runs stages 1-4 on one camera's frame.

    Returns ``(observations, display_frame)``. The display frame is the
    denoised one when ``--show_denoised`` is set, otherwise the raw capture.

    With ``analytics``, every detection is matched to a tracked person
    first, and the expensive part - crop denoise, face detection, ArcFace -
    runs only when that person's identity needs (re)checking. Otherwise the
    track's recent embedding is reused (``obs.cached``).
    """
    frame_bgr = downscale(frame_bgr, args.process_width)
    working = frame_bgr
    display = frame_bgr

    # --- Stage 1 (whole-frame mode) ---------------------------------------
    if stages.denoiser and args.denoise_scope == "frame":
        with timer(f"denoise:{camera}"):
            working = stages.denoiser.denoise(frame_bgr)
        if args.show_denoised:
            display = working.copy()

    # --- Stage 2 ----------------------------------------------------------
    # With tracking, detect everyone (cheap) and ration only the face work
    # below; without it, the old largest-N cap applies to both.
    subjects = []
    if stages.detector:
        with timer(f"detect:{camera}"):
            limit = config.TRACK_MAX_PEOPLE if analytics is not None else args.max_subjects
            subjects = stages.detector.detect(working, max_subjects=limit)

    tracks = [None] * len(subjects)
    check = set(range(len(subjects)))
    if analytics is not None:
        h, w = frame_bgr.shape[:2]
        tracks = analytics.track(camera, t_wall, [s.box for s in subjects],
                                 [s.conf for s in subjects], (w, h))
        check = analytics.select_face_checks(tracks, t_wall, budget=args.max_subjects)

    observations: List[ViewObservation] = []
    for i, (subject, trk) in enumerate(zip(subjects, tracks)):
        obs = ViewObservation(
            camera=camera, box=subject.box, conf=subject.conf, crop=subject.crop, track=trk
        )

        if i in check:
            # --- Stage 1 (crop mode) -------------------------------------
            # Detection ran on the noisy frame; only the crop gets cleaned
            # up, which is where recognition actually needs the detail.
            if stages.denoiser and args.denoise_scope == "crop":
                with timer(f"denoise:{camera}"):
                    obs.crop = stages.denoiser.denoise(obs.crop)

            # --- Stages 3 + 4 ---------------------------------------------
            _restore_and_embed(obs, camera, stages, timer, args)
            if trk is not None:
                thumb = None
                if face_thumbs and obs.embedding is not None and obs.face is not None:
                    from hub.protocol import encode_jpeg
                    thumb = encode_jpeg(obs.face, quality=72, max_width=96)
                analytics.face_result(trk, t_wall, embedding=obs.embedding, thumb=thumb)

        if trk is not None and obs.embedding is None:
            obs.embedding, obs.cached = SceneAnalytics.embedding_for(trk)
        observations.append(obs)

    return observations, display


def _restore_and_embed(obs, camera: str, stages: Stages, timer: StageTimer, args) -> None:
    """Stage 3 and stage 4, under the selected restoration policy.

    ``escalate`` is what makes this tractable on weak hardware: GFPGAN is a
    StyleGAN2 decoder costing seconds per face on Maxwell, while ArcFace on
    the raw crop is cheap. Almost every frame contains nobody enrolled, so
    rejecting those on the cheap embedding alone avoids the expensive path
    entirely; only plausible candidates are restored and re-embedded.
    """
    if stages.embedder is None:
        return

    from pipeline.face_align import prepare_for_embedding

    def embed(image, tag="embed"):
        if image is None or image.size == 0:
            return None
        with timer("%s:%s" % (tag, camera)):
            return stages.embedder.embed(image)

    def restore(image):
        if stages.restorer is None:
            return None
        with timer("restore:%s" % camera):
            return stages.restorer.restore(image)

    # Detect and align the face before anything else touches it. Stage 4 is
    # trained on faces warped onto fixed landmark positions; a person crop
    # resized to 112x112 is not a weaker version of that, it is a different
    # kind of image, and the embedding would describe clothing and pose.
    with timer("align:%s" % camera):
        aligned = prepare_for_embedding(obs.crop, size=config.ARCFACE_INPUT)

    if aligned is None:
        return                      # no face in this crop; nothing to embed
    obs.face = aligned

    mode = args.restore_mode
    if stages.restorer is None:
        mode = "never"

    if mode == "never":
        obs.embedding = embed(aligned)
        return

    if mode == "always":
        restored = restore(obs.crop)
        if restored is not None:
            obs.face = restored
        obs.embedding = embed(obs.face)
        return

    # --- escalate ---------------------------------------------------------
    preliminary = embed(aligned, tag="embed_fast")
    obs.embedding = preliminary
    if preliminary is None:
        return

    # With an empty gallery there is nothing to escalate towards, and the
    # cheap embedding is all the live loop can act on anyway.
    if stages.index is None or stages.index.index.ntotal == 0:
        return

    with timer("prefilter:%s" % camera):
        prelim = stages.index.query(preliminary, args.threshold)

    if prelim.score < args.escalate_above:
        return

    restored = restore(obs.crop)
    if restored is None:
        return                      # keep the aligned face and its embedding
    refined = embed(restored)
    if refined is not None:
        obs.face = restored
        obs.embedding = refined
        obs.escalated = True


def compose_display(views: Dict[str, np.ndarray], height: int = 540) -> np.ndarray:
    """Side-by-side canvas of every camera, scaled to a common height."""
    panels = []
    for name, frame in views.items():
        h, w = frame.shape[:2]
        scale = height / h
        panel = cv2.resize(frame, (int(w * scale), height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), (70, 70, 70), 1)
        cv2.putText(panel, name, (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    return np.hstack(panels) if panels else np.zeros((height, height, 3), np.uint8)


def _node_rows(pair, by_camera, cap_stats, timer, pair_count):
    """Per-camera status rows in the hub's 'nodes' shape, for the dashboard."""
    summary = timer.summary()
    shared = {k: v for k, v in summary.items() if ":" not in k}
    rows = []
    for i, name in enumerate(sorted(pair.frames)):
        stages_ms = {k.split(":")[0]: v for k, v in summary.items() if k.endswith(":" + name)}
        if i == 0:
            stages_ms.update(shared)
        rows.append({
            "name": name,
            "online": True,
            "fps": round(cap_stats["capture_fps"].get(name, 0.0), 1),
            "frames": pair_count,
            "stages_ms": stages_ms,
            "clock_offset_ms": 0.0,
            "last_seen_ago": 0.0,
            "subjects": len(by_camera.get(name, [])),
        })
    return rows


def run(args):
    cameras = cameras_from_args(args.source, codec=args.codec)
    capture = SyncedCapture(
        cameras,
        sync_window_ms=args.sync_window,
        use_gstreamer=not args.no_gstreamer,
        target_fps=args.capture_fps,
        require_all=not args.allow_single_camera,
        frame_select=args.frame_select,
    ).start()

    print(f"\ncameras: {', '.join(c['name'] + ' -> ' + c['url'] for c in cameras)}")
    print(f"waiting for all {len(cameras)} stream(s) ...")
    if not capture.wait_until_ready(timeout=args.startup_timeout):
        print("error: not every camera delivered a frame; check the URLs and network")
        capture.release()
        return
    print("all streams live")

    stages = Stages(
        {
            "denoise": not args.no_denoise,
            "detect": not args.no_detect,
            "restore": not args.no_restore,
            "match": not args.no_match,
        },
        args=args,
    )

    fusion = CrossViewFusion(
        index=stages.index,
        cross_view_threshold=args.cross_view_threshold,
        gallery_threshold=args.threshold,
        fusion_mode=args.fusion,
        require_agreement=args.require_agreement,
        require_both_views=args.require_both_views,
    )

    # Tracking, visitors, journeys, predictions, loitering. The dashboard
    # is built on it, so it cannot be switched off while serving one.
    analytics = None
    if not args.no_analytics or args.dashboard:
        analytics = SceneAnalytics(
            embed_refresh_s=args.embed_refresh,
            visitor_threshold=args.visitor_threshold,
            loiter_s=args.loiter_seconds,
        )
        cal = analytics.calibrator
        print("[analytics] tracking on | visitor threshold %.2f | loitering after %.0f s | "
              "re-embed every %.1f s | confidence: %s"
              % (args.visitor_threshold, args.loiter_seconds, args.embed_refresh,
                 "calibrated (%s pairs)" % (cal.data.get("n_genuine", 0) + cal.data.get("n_impostor", 0))
                 if cal.calibrated else "uncalibrated - run calibrate.py; raw scores shown"))

    hub = None
    alerts = deque(maxlen=100)
    if args.dashboard:
        from hub.server import Handler, Hub, ThreadingHTTPServer, compose_result, local_ip
        hub = Hub(fusion=fusion, analytics=analytics, index=stages.index)
        alerts = hub.alerts
        Handler.hub = hub
        server = ThreadingHTTPServer(("0.0.0.0", args.dashboard), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, name="dashboard", daemon=True).start()
        print("[dashboard] open http://%s:%d/ on any device on this network"
              % (local_ip(), args.dashboard))
    started = time.time()

    timer = StageTimer(window=20)
    pair_count = alert_count = 0
    last_alert_time: dict = {}
    infer_fps_ema = 0.0
    last_tracks = []
    t_now = time.time()

    # Report the mode that will actually be used, not the one requested.
    # _restore_and_embed degrades to "never" when no restorer loaded - via
    # --no_restore, or because gfpgan cannot install on Python 3.6 - and a
    # banner claiming "always" there would misdescribe every later timing.
    effective_restore = args.restore_mode if stages.restorer else "never"
    restore_note = "" if stages.restorer else " (no restorer loaded)"
    print(f"\nprofile: {config.PROFILE} | denoise scope: {args.denoise_scope} "
          f"| process width: {args.process_width or 'native'} "
          f"| restore: {effective_restore}{restore_note}"
          + (f" (>{args.escalate_above})" if effective_restore == "escalate" else ""))
    print(f"fusion: {args.fusion} | cross-view >= {args.cross_view_threshold} "
          f"| gallery >= {args.threshold}")
    print("pipeline ready - 'q' quit, 's' snapshot, 'd' toggle denoised view\n")

    try:
        while True:
            loop_start = time.perf_counter()

            pair = capture.read(timeout=args.timeout)
            if pair is None:
                if capture.exhausted:
                    print("[loop] all sources exhausted")
                    break
                print("[loop] no synchronized pair within timeout; still waiting")
                continue
            pair_count += 1

            displays = {name: downscale(f.image, args.process_width)
                        for name, f in pair.frames.items()}
            run_inference = pair_count % args.detect_every == 0

            if run_inference:
                by_camera: Dict[str, List[ViewObservation]] = {}
                t_now = max(wall_time(f.timestamp) for f in pair.frames.values())
                for name, frame in pair.frames.items():
                    t_cam = wall_time(frame.timestamp)
                    observations, display = process_view(
                        frame.image, name, stages, timer, args,
                        analytics=analytics, t_wall=t_cam, face_thumbs=hub is not None)
                    by_camera[name] = observations
                    displays[name] = display

                # --- cross-view association + fused identification --------
                with timer("fuse"):
                    tracks = fusion.process(by_camera)
                last_tracks = tracks

                events = []
                if analytics is not None:
                    with timer("analytics"):
                        events = analytics.resolve(t_now, tracks)
                    for ev in events:
                        print(f"[{ev['at']}] {ev['text']}")
                        if ev.get("kind") == "loiter":
                            alerts.append({
                                "at": ev["at"], "kind": "loiter",
                                "identity": ev["text"].split(" has been")[0],
                                "camera": ev.get("camera"), "text": ev["text"],
                                "score": None, "cross_view": None,
                                "cameras": [ev.get("camera")], "multiview": False,
                                "agreement": False})

                # The dashboard draws its own overlay, so it gets the frames
                # before anything is drawn on them.
                if hub is not None:
                    from hub.protocol import encode_jpeg
                    frames_b64 = {n: encode_jpeg(img, quality=60, max_width=args.thumb_width)
                                  for n, img in displays.items()}

                for t_idx, track in enumerate(tracks):
                    color = TRACK_COLORS[t_idx % len(TRACK_COLORS)]
                    matched = track.match is not None and track.match.is_match
                    if matched:
                        color = ALERT_COLOR

                    if analytics is None:
                        for obs in track.observations:
                            if obs.camera in displays:
                                draw_box(displays[obs.camera], obs.box,
                                         f"T{t_idx} {track.label()}", color)

                    if not matched:
                        continue

                    identity = track.match.identity
                    now = time.time()
                    if now - last_alert_time.get(identity, 0) < args.alert_cooldown:
                        continue
                    last_alert_time[identity] = now
                    alert_count += 1

                    stamp = datetime.now().strftime("%H:%M:%S")
                    views = "+".join(track.cameras)
                    conf = analytics.calibrator.probability(track.match.score) if analytics else None
                    alerts.append({
                        "at": stamp, "kind": "match", "identity": identity,
                        "score": round(track.match.score, 4), "confidence": conf,
                        "cross_view": round(track.cross_view_score, 4),
                        "cameras": track.cameras, "multiview": track.is_multiview,
                        "agreement": track.agreement})
                    print(f"\n[!] {stamp}  THREAT MATCH: {identity} "
                          f"(fused cosine {track.match.score:.3f}"
                          + (f", {100 * conf:.0f}% confidence" if conf is not None else "")
                          + f", views {views})")
                    if track.is_multiview:
                        print(f"    cross-view similarity {track.cross_view_score:.3f} "
                              f"| views agree: {track.agreement}")
                        for obs in track.observations:
                            if obs.solo_match:
                                print(f"      {obs.camera}: {obs.solo_match.identity} "
                                      f"{obs.solo_match.score:.3f}"
                                      + ("  (GFPGAN escalated)" if obs.escalated else ""))
                    else:
                        print("    single-view detection (the other camera did not see them)")
                    for note in track.notes:
                        print(f"    note: {note}")
                    if not pair.in_sync:
                        print(f"    warning: frames were {pair.skew_ms:.0f}ms apart")

                    if args.save_matches:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        stem = config.CAPTURE_DIR / f"{ts}_{identity}"
                        for obs in track.observations:
                            cv2.imwrite(f"{stem}_{obs.camera}_frame.jpg",
                                        pair.frames[obs.camera].image)
                            if obs.face is not None:
                                cv2.imwrite(f"{stem}_{obs.camera}_face.jpg", obs.face)
                        print(f"    saved {stem}_*.jpg")

                    if not args.headless:
                        face = track.best_face
                        if face is not None:
                            cv2.imshow("Human Verification Required", face)

                if analytics is not None:
                    for name in displays:
                        draw_scene_view(displays[name], analytics.view_state(name, t_now))

                if hub is not None:
                    cap_stats = capture.stats()
                    multiview = [t for t in tracks if t.is_multiview]
                    similarity = max((t.cross_view_score for t in multiview), default=None)
                    hub.publish(compose_result(
                        started=started,
                        cameras=pair.frames.keys(),
                        skew_ms=pair.skew_ms,
                        in_sync=pair.in_sync,
                        similarity=similarity,
                        track_payloads=[fused_track_payload(t, analytics.calibrator) for t in tracks],
                        nodes=_node_rows(pair, by_camera, cap_stats, timer, pair_count),
                        frames=frames_b64,
                        settings=hub.settings,
                        gallery=stages.index.stats() if stages.index else None,
                        stats={"pairs": pair_count, "skewed": cap_stats["skewed"],
                               "alerts": len(alerts)},
                        alerts=alerts,
                        scene=analytics.state(t_now),
                    ))
            else:
                # Between detections: move every box to where the Kalman
                # filter says that person is now, instead of freezing it.
                for name, frame in pair.frames.items():
                    if analytics is not None:
                        t_frame = wall_time(frame.timestamp)
                        draw_scene_view(displays[name],
                                        analytics.view_state(name, t_frame, at=t_frame))
                    else:
                        for t_idx, track in enumerate(last_tracks):
                            matched = track.match is not None and track.match.is_match
                            color = ALERT_COLOR if matched else TRACK_COLORS[t_idx % len(TRACK_COLORS)]
                            for obs in track.observations:
                                if obs.camera == name:
                                    draw_box(displays[name], obs.box,
                                             f"T{t_idx} {track.label()}", color)

            # --- HUD ----------------------------------------------------
            loop_ms = (time.perf_counter() - loop_start) * 1000
            fps = 1000.0 / max(loop_ms, 1e-3)
            infer_fps_ema = fps if infer_fps_ema == 0 else 0.9 * infer_fps_ema + 0.1 * fps

            cap_stats = capture.stats()
            cap_fps = "  ".join(f"{n} {v:.0f}" for n, v in cap_stats["capture_fps"].items())
            multiview = sum(1 for t in last_tracks if t.is_multiview)
            sync_tag = f"{pair.skew_ms:4.0f}ms" + ("" if pair.in_sync else " SKEW")

            hud = [
                f"capture {cap_fps} fps   inference {infer_fps_ema:5.1f} fps   sync {sync_tag}",
                "  ".join(f"{k} {v:.0f}ms" for k, v in timer.summary().items()) or "warming up",
                f"tracks {len(last_tracks)} ({multiview} multi-view)   "
                f"alerts {alert_count}   skipped {cap_stats['skipped_frames']}",
            ]
            if analytics is not None:
                ps = analytics.journeys.prediction_stats()
                eff = analytics.efficiency()
                decided = ps["hits"] + ps["misses"]
                hud.append(
                    f"visitors {len(analytics.visitors)} (seen {analytics.visitors.created})   "
                    f"predictions {ps['hits']}/{decided} correct, {ps['pending']} pending   "
                    + (f"face work skipped {eff['skipped_pct']:.0f}%"
                       if eff and eff["skipped_pct"] is not None else ""))

            canvas = compose_display(displays, height=args.display_height)
            draw_hud(canvas, hud)

            if not args.headless:
                cv2.imshow("Privacy-First Edge Pipeline - Multi-Angle", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("d"):
                    args.show_denoised = not args.show_denoised
                    print(f"[view] denoised display: {args.show_denoised}")
                if key == ord("s"):
                    path = config.CAPTURE_DIR / f"snap_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    cv2.imwrite(str(path), canvas)
                    print(f"[snap] {path}")
            elif pair_count % 60 == 0:
                print(f"[loop] pair {pair_count} | {infer_fps_ema:.1f} fps | {timer.summary()}")

            if args.max_pairs and pair_count >= args.max_pairs:
                break

    except KeyboardInterrupt:
        print("\nstopped by operator")
    finally:
        capture.release()
        cv2.destroyAllWindows()
        stats = capture.stats()

        print("\n--- session summary ---")
        print(f"frame pairs processed : {pair_count}")
        print(f"capture fps           : {stats['capture_fps']}")
        print(f"inference fps         : {infer_fps_ema:.2f}")
        print(f"frames skipped        : {stats['skipped_frames']} "
              f"(inference could not keep up with capture)")
        print(f"frame selection       : {stats['frame_select']} "
              f"(a sharper, older frame won {stats['sharper_older_chosen']} time(s))")
        from pipeline.face_align import QUALITY_STATS
        q = QUALITY_STATS
        print(f"faces                 : {q['accepted']} embedded, "
              f"{q['too_small']} too small, {q['turned_away']} turned away, "
              f"{q['no_face']} crops with no face")
        if analytics is not None:
            st = analytics.state(t_now)
            eff, ps = st["efficiency"], st["predictions"]
            print(f"face work (tracking)  : {eff['faces_embedded']} faces embedded, "
                  f"{eff['faces_reused']} reused from a track, "
                  f"{eff['face_checks_skipped']} retries deferred"
                  + (f" -> {eff['skipped_pct']:.0f}% of face checks skipped"
                     if eff["skipped_pct"] is not None else ""))
            print(f"visitors              : {analytics.visitors.created} seen, "
                  f"{len(analytics.visitors)} still remembered "
                  f"(forgotten after {analytics.visitors.ttl_s:.0f} s)")
            print(f"predictions           : {ps['made']} made, {ps['hits']} correct, "
                  f"{ps['misses']} wrong" + (f" ({100 * ps['accuracy']:.0f}%)"
                                             if ps["accuracy"] is not None else ""))
            print(f"loitering alerts      : "
                  f"{sum(1 for e in analytics.events if e['kind'] == 'loiter')}")
        print(f"pairs out of sync     : {stats['skewed']}")
        print(f"reconnects            : {stats['reconnects']}")
        print(f"alerts raised         : {alert_count}")
        print(f"mean stage ms         : {timer.summary()}")
        print("resources freed. air-gapped compliance maintained.")


def parse_args():
    p = argparse.ArgumentParser(description="Dual-camera edge surveillance pipeline")

    # --- sources
    p.add_argument("--source", action="append", default=None,
                   help="repeat once per camera. RTSP URL, file, webcam index, "
                        "or NAME=url. Defaults to config.CAMERAS.")
    p.add_argument("--codec", default="h264", choices=["h264", "h265"])
    p.add_argument("--capture_fps", type=float, default=config.CAPTURE_FPS,
                   help="camera frame rate; also paces file replay")
    p.add_argument("--sync_window", type=float, default=config.SYNC_WINDOW_MS,
                   help="max ms between paired frames before they count as skewed")
    p.add_argument("--no_gstreamer", action="store_true",
                   help="skip the Jetson GStreamer path (use off-device)")
    p.add_argument("--allow_single_camera", action="store_true",
                   help="emit pairs even when one camera is down")
    p.add_argument("--startup_timeout", type=float, default=15.0)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--max_pairs", type=int, default=0)

    # --- matching / fusion
    p.add_argument("--threshold", type=float, default=config.MATCH_THRESHOLD,
                   help="gallery cosine threshold for a positive ID")
    p.add_argument("--cross_view_threshold", type=float, default=config.CROSS_VIEW_THRESHOLD,
                   help="cosine between two live views to call them the same person")
    p.add_argument("--fusion", default=config.FUSION_MODE, choices=["mean", "max", "first"])
    p.add_argument("--require_agreement", action="store_true",
                   default=config.REQUIRE_VIEW_AGREEMENT,
                   help="both views must independently name the same identity")
    p.add_argument("--require_both_views", action="store_true",
                   default=config.REQUIRE_BOTH_VIEWS,
                   help="suppress alerts for subjects only one camera can see")

    # --- stages
    p.add_argument("--denoise_scope", default=config.DENOISE_SCOPE, choices=["frame", "crop"],
                   help="crop is the designed path; frame is kept only for an A/B")
    p.add_argument("--denoise_profile", default=config.DENOISE_PROFILE,
                   choices=["off", "fast", "balanced", "quality"],
                   help="stage-1 strength; see pipeline/stage1_denoise.py")
    p.add_argument("--process_width", type=int, default=config.PROCESS_WIDTH,
                   help="downscale frames to this width before inference; 0 disables")
    p.add_argument("--restore_mode", default=config.RESTORE_MODE,
                   choices=["always", "escalate", "never"],
                   help="when to spend time on GFPGAN (escalate = only on candidates)")
    p.add_argument("--escalate_above", type=float, default=config.ESCALATE_ABOVE,
                   help="preliminary cosine above which escalate mode restores the face")
    p.add_argument("--max_subjects", type=int, default=config.YOLO_MAX_SUBJECTS,
                   help="people given face recognition per camera per cycle; everyone "
                        "up to config.TRACK_MAX_PEOPLE is still tracked")
    p.add_argument("--frame_select", default=config.FRAME_SELECT,
                   choices=["sharpest", "newest"],
                   help="which of the frames that arrived during the last "
                        "inference cycle to process")
    p.add_argument("--detect_every", type=int, default=1,
                   help="run inference every Nth frame pair")
    p.add_argument("--alert_cooldown", type=float, default=10.0, help="seconds per identity")
    p.add_argument("--no_denoise", action="store_true")
    p.add_argument("--no_detect", action="store_true")
    p.add_argument("--no_restore", action="store_true")
    p.add_argument("--no_match", action="store_true")

    # --- tracking and scene analytics (pipeline/analytics.py)
    p.add_argument("--no_analytics", action="store_true",
                   help="no tracking, visitors, journeys, predictions or loitering")
    p.add_argument("--embed_refresh", type=float, default=config.TRACK_EMBED_REFRESH_S,
                   help="seconds before a tracked, identified person's face is "
                        "re-embedded; 0 = every cycle (the old behaviour)")
    p.add_argument("--visitor_threshold", type=float, default=config.VISITOR_THRESHOLD,
                   help="similarity to recognise an unenrolled visitor again")
    p.add_argument("--loiter_seconds", type=float, default=config.LOITER_SECONDS,
                   help="continuous presence that raises a loitering alert")

    # --- display
    p.add_argument("--dashboard", type=int, default=0, metavar="PORT",
                   help="serve the web dashboard on this port (e.g. 8765); "
                        "open http://<this-device-ip>:PORT/ from any device on the network")
    p.add_argument("--thumb_width", type=int, default=480,
                   help="width of the frames sent to the dashboard")
    p.add_argument("--display_height", type=int, default=540)
    p.add_argument("--show_denoised", action="store_true")
    p.add_argument("--headless", action="store_true", default=config.HEADLESS)
    p.add_argument("--save_matches", action="store_true", default=config.SAVE_MATCHES)
    p.add_argument("--no_save_matches", dest="save_matches", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
