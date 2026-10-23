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

Everything runs locally; nothing is transmitted off the node.
"""

import argparse
import time
from datetime import datetime
from typing import Dict, List

import cv2
import numpy as np

import config
from pipeline.fusion import CrossViewFusion, ViewObservation
from pipeline.sources import SyncedCapture, cameras_from_args, gstreamer_pipeline  # noqa: F401
from pipeline.utils import StageTimer, downscale, draw_box, draw_hud

# One colour per track, reused across both views so the operator can see at a
# glance which box in CAM_A corresponds to which box in CAM_B.
TRACK_COLORS = [
    (0, 220, 255), (120, 255, 120), (255, 180, 80),
    (200, 120, 255), (120, 220, 255), (255, 255, 120),
]
ALERT_COLOR = (0, 0, 255)


class Stages:
    """Loads each model once, tolerating missing weights."""

    def __init__(self, enable: dict):
        self.denoiser = self.detector = self.restorer = None
        self.embedder = self.index = None
        self.enabled = dict(enable)

        if self.enabled.get("denoise"):
            try:
                from pipeline.stage1_denoise import get_denoiser
                self.denoiser = get_denoiser()
                print("[stage1] NAFNet denoiser ready")
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


def process_view(
    frame_bgr: np.ndarray,
    camera: str,
    stages: Stages,
    timer: StageTimer,
    args,
) -> tuple:
    """Runs stages 1-4 on one camera's frame.

    Returns ``(observations, display_frame)``. The display frame is the
    denoised one when ``--show_denoised`` is set, otherwise the raw capture.
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
    subjects = []
    if stages.detector:
        with timer(f"detect:{camera}"):
            subjects = stages.detector.detect(working, max_subjects=args.max_subjects)

    observations: List[ViewObservation] = []
    for subject in subjects:
        crop = subject.crop

        # --- Stage 1 (crop mode) -----------------------------------------
        # Detection ran on the noisy frame; only the crop gets cleaned up,
        # which is where recognition actually needs the detail.
        if stages.denoiser and args.denoise_scope == "crop":
            with timer(f"denoise:{camera}"):
                crop = stages.denoiser.denoise(crop)

        obs = ViewObservation(
            camera=camera, box=subject.box, conf=subject.conf, crop=crop
        )

        # --- Stages 3 + 4 -------------------------------------------------
        _restore_and_embed(obs, camera, stages, timer, args)
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

    mode = args.restore_mode
    if stages.restorer is None:
        mode = "never"

    if mode == "never":
        obs.embedding = embed(obs.crop)
        return

    if mode == "always":
        obs.face = restore(obs.crop)
        obs.embedding = embed(obs.face if obs.face is not None else obs.crop)
        return

    # --- escalate ---------------------------------------------------------
    preliminary = embed(obs.crop, tag="embed_fast")
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

    obs.face = restore(obs.crop)
    if obs.face is None:
        return
    refined = embed(obs.face)
    if refined is not None:
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


def run(args):
    cameras = cameras_from_args(args.source, codec=args.codec)
    capture = SyncedCapture(
        cameras,
        sync_window_ms=args.sync_window,
        use_gstreamer=not args.no_gstreamer,
        target_fps=args.capture_fps,
        require_all=not args.allow_single_camera,
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
        }
    )

    fusion = CrossViewFusion(
        index=stages.index,
        cross_view_threshold=args.cross_view_threshold,
        gallery_threshold=args.threshold,
        fusion_mode=args.fusion,
        require_agreement=args.require_agreement,
        require_both_views=args.require_both_views,
    )

    timer = StageTimer(window=20)
    pair_count = alerts = 0
    last_alert_time: dict = {}
    infer_fps_ema = 0.0
    last_tracks = []

    print(f"\nprofile: {config.PROFILE} | denoise scope: {args.denoise_scope} "
          f"| process width: {args.process_width or 'native'} "
          f"| restore: {args.restore_mode}"
          + (f" (>{args.escalate_above})" if args.restore_mode == "escalate" else ""))
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

            displays = {name: f.image for name, f in pair.frames.items()}
            run_inference = pair_count % args.detect_every == 0

            if run_inference:
                by_camera: Dict[str, List[ViewObservation]] = {}
                for name, frame in pair.frames.items():
                    observations, display = process_view(frame.image, name, stages, timer, args)
                    by_camera[name] = observations
                    displays[name] = display

                # --- cross-view association + fused identification --------
                with timer("fuse"):
                    tracks = fusion.process(by_camera)
                last_tracks = tracks

                for t_idx, track in enumerate(tracks):
                    color = TRACK_COLORS[t_idx % len(TRACK_COLORS)]
                    matched = track.match is not None and track.match.is_match
                    if matched:
                        color = ALERT_COLOR

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
                    alerts += 1

                    stamp = datetime.now().strftime("%H:%M:%S")
                    views = "+".join(track.cameras)
                    print(f"\n[!] {stamp}  THREAT MATCH: {identity} "
                          f"(fused cosine {track.match.score:.3f}, views {views})")
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
            else:
                # Carry the previous boxes forward so the display does not flicker.
                for t_idx, track in enumerate(last_tracks):
                    matched = track.match is not None and track.match.is_match
                    color = ALERT_COLOR if matched else TRACK_COLORS[t_idx % len(TRACK_COLORS)]
                    for obs in track.observations:
                        if obs.camera in displays:
                            draw_box(displays[obs.camera], obs.box,
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
                f"alerts {alerts}   skipped {cap_stats['skipped_frames']}",
            ]

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
        print(f"pairs out of sync     : {stats['skewed']}")
        print(f"reconnects            : {stats['reconnects']}")
        print(f"alerts raised         : {alerts}")
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
                   help="denoise the whole frame, or detect first and denoise the crop")
    p.add_argument("--process_width", type=int, default=config.PROCESS_WIDTH,
                   help="downscale frames to this width before inference; 0 disables")
    p.add_argument("--restore_mode", default=config.RESTORE_MODE,
                   choices=["always", "escalate", "never"],
                   help="when to spend time on GFPGAN (escalate = only on candidates)")
    p.add_argument("--escalate_above", type=float, default=config.ESCALATE_ABOVE,
                   help="preliminary cosine above which escalate mode restores the face")
    p.add_argument("--max_subjects", type=int, default=config.YOLO_MAX_SUBJECTS)
    p.add_argument("--detect_every", type=int, default=1,
                   help="run inference every Nth frame pair")
    p.add_argument("--alert_cooldown", type=float, default=10.0, help="seconds per identity")
    p.add_argument("--no_denoise", action="store_true")
    p.add_argument("--no_detect", action="store_true")
    p.add_argument("--no_restore", action="store_true")
    p.add_argument("--no_match", action="store_true")

    # --- display
    p.add_argument("--display_height", type=int, default=540)
    p.add_argument("--show_denoised", action="store_true")
    p.add_argument("--headless", action="store_true", default=config.HEADLESS)
    p.add_argument("--save_matches", action="store_true", default=config.SAVE_MATCHES)
    p.add_argument("--no_save_matches", dest="save_matches", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
