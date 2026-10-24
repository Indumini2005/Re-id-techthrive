"""Capture node - runs on each laptop that has a camera.

Grabs frames, runs whatever pipeline stages are available locally, and POSTs
the resulting embedding to the hub. Only vectors and small thumbnails leave
this machine; the source frames never do.

    # Laptop A (built-in webcam)
    python node_agent.py --name CAM_A --source 0 --hub http://192.168.1.50:8765

    # Laptop B
    python node_agent.py --name CAM_B --source 0 --hub http://192.168.1.50:8765

    # Against an IP camera instead
    python node_agent.py --name CAM_A --source rtsp://... --hub http://...

Stages degrade gracefully: a laptop with no models still streams thumbnails
and detections, so you can bring the demo up one piece at a time. Check what
a node actually loaded from its startup banner.
"""

import argparse
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import config  # noqa: E402
from hub.protocol import HUB_DEFAULT_PORT, build_observation, dumps, encode_jpeg  # noqa: E402
from pipeline.utils import StageTimer, downscale, draw_box  # noqa: E402


class Sender(threading.Thread):
    """POSTs observations on a background thread.

    Capture must never block on the network. If the hub is slow or down we
    drop the backlog and keep the newest - the same always-newest rule the
    frame pipeline uses, for the same reason.
    """

    def __init__(self, hub_url, timeout=2.0):
        super().__init__(daemon=True)
        self.url = hub_url.rstrip("/") + "/api/observe"
        self.timeout = timeout
        self.queue = deque(maxlen=2)
        self.event = threading.Event()
        self.stop_flag = threading.Event()
        self.sent = 0
        self.dropped = 0
        self.failures = 0
        self.connected = False

    def submit(self, payload):
        if len(self.queue) == self.queue.maxlen:
            self.dropped += 1
        self.queue.append(payload)
        self.event.set()

    def run(self):
        while not self.stop_flag.is_set():
            self.event.wait(timeout=0.5)
            self.event.clear()
            while self.queue:
                payload = self.queue.popleft()
                try:
                    req = urllib.request.Request(
                        self.url, data=dumps(payload),
                        headers={"Content-Type": "application/json"},
                    )
                    urllib.request.urlopen(req, timeout=self.timeout).read()
                    self.sent += 1
                    if not self.connected:
                        print("[net] connected to hub")
                    self.connected = True
                    self.failures = 0
                except (urllib.error.URLError, OSError) as exc:
                    self.failures += 1
                    if self.connected or self.failures == 1:
                        print("[net] hub unreachable (%s); retrying" % exc)
                    self.connected = False
                    time.sleep(0.5)

    def shutdown(self):
        self.stop_flag.set()
        self.event.set()


class Stages:
    """Loads what this laptop has, reports what it does not."""

    def __init__(self, args):
        self.denoiser = self.detector = self.restorer = self.embedder = self.index = None

        if not args.no_denoise:
            try:
                from pipeline.stage1_denoise import get_denoiser
                self.denoiser = get_denoiser(
                    profile=getattr(args, "denoise_profile", None))
                print("  stage1 denoise  : ready (%s)" % self.denoiser.describe())
            except Exception as exc:
                print("  stage1 denoise  : off (%s)" % exc)

        if not args.no_detect:
            try:
                from pipeline.stage2_detect import get_detector
                self.detector = get_detector()
                print("  stage2 detect   : ready (%s)" % self.detector.backend_name)
            except Exception as exc:
                print("  stage2 detect   : off (%s)" % exc)

        if not args.no_restore:
            try:
                from pipeline.stage3_restore import get_restorer
                self.restorer = get_restorer()
                print("  stage3 restore  : ready")
            except Exception as exc:
                print("  stage3 restore  : off (%s)" % exc)

        if not args.no_embed:
            try:
                from pipeline.stage4_embed import get_embedder
                self.embedder = get_embedder()
                print("  stage4 embed    : ready")
            except Exception as exc:
                print("  stage4 embed    : off (%s)" % exc)
                print("                    without this the hub has no vectors "
                      "to compare - fix this one first.")

        if args.restore_mode == "escalate" and self.restorer is not None:
            try:
                from pipeline.stage5_match import get_index
                self.index = get_index()
            except Exception:
                self.index = None


def open_capture(source, width, height):
    src = int(source) if str(source).isdigit() else source
    if str(source).startswith("rtsp://"):
        cap = cv2.VideoCapture(source)
    elif sys.platform == "win32" and isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)   # avoids MSMF startup lag
    else:
        cap = cv2.VideoCapture(src)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def process_frame(frame, stages, timer, args):
    """Returns (subjects payload, annotated frame for the operator view)."""
    frame = downscale(frame, args.process_width)
    working = frame
    annotated = frame.copy()

    if stages.denoiser and args.denoise_scope == "frame":
        with timer("denoise"):
            working = stages.denoiser.denoise(frame)

    subjects = []
    if stages.detector:
        with timer("detect"):
            subjects = stages.detector.detect(working, max_subjects=args.max_subjects)
    elif stages.embedder:
        # No detector: treat the whole frame as one subject so the demo still
        # produces a comparable vector.
        from pipeline.stage2_detect import Subject
        h, w = frame.shape[:2]
        subjects = [Subject(box=(0, 0, w, h), conf=1.0, crop=frame)]

    out = []
    for subject in subjects:
        crop = subject.crop
        if stages.denoiser and args.denoise_scope == "crop":
            with timer("denoise"):
                crop = stages.denoiser.denoise(crop)

        face = None
        escalated = False
        embedding = None

        if stages.embedder:
            mode = args.restore_mode if stages.restorer else "never"

            if mode == "always":
                with timer("restore"):
                    face = stages.restorer.restore(crop)
                with timer("embed"):
                    embedding = stages.embedder.embed(face if face is not None else crop)

            elif mode == "never":
                with timer("embed"):
                    embedding = stages.embedder.embed(crop)

            else:   # escalate
                with timer("embed"):
                    embedding = stages.embedder.embed(crop)
                if embedding is not None and stages.index is not None \
                        and stages.index.index.ntotal > 0:
                    prelim = stages.index.query(embedding, args.threshold)
                    if prelim.score >= args.escalate_above:
                        with timer("restore"):
                            face = stages.restorer.restore(crop)
                        if face is not None:
                            with timer("embed"):
                                better = stages.embedder.embed(face)
                            if better is not None:
                                embedding, escalated = better, True

        draw_box(annotated, subject.box, "person %.2f" % subject.conf, (53, 224, 212))

        out.append({
            "box": subject.box,
            "conf": subject.conf,
            "embedding": embedding,
            "escalated": escalated,
            "face_thumb": encode_jpeg(face, quality=72, max_width=96)
            if face is not None else encode_jpeg(crop, quality=65, max_width=96),
        })

    return out, annotated


def main():
    ap = argparse.ArgumentParser(description="Capture node for the multi-camera hub")
    ap.add_argument("--name", required=True, help="camera label, e.g. CAM_A")
    ap.add_argument("--source", default="0", help="webcam index, file, or rtsp:// URL")
    ap.add_argument("--hub", default="http://127.0.0.1:%d" % HUB_DEFAULT_PORT)
    ap.add_argument("--fps", type=float, default=6.0,
                    help="observations sent per second; the hub does not need 30")
    ap.add_argument("--cap_width", type=int, default=1280)
    ap.add_argument("--cap_height", type=int, default=720)
    ap.add_argument("--process_width", type=int, default=config.PROCESS_WIDTH or 640)
    ap.add_argument("--thumb_width", type=int, default=480, help="preview sent to the hub")
    ap.add_argument("--thumb_quality", type=int, default=65)

    ap.add_argument("--denoise_scope", default=config.DENOISE_SCOPE, choices=["frame", "crop"])
    ap.add_argument("--denoise_profile", default=config.DENOISE_PROFILE,
                    choices=["off", "fast", "balanced", "quality"])
    ap.add_argument("--restore_mode", default=config.RESTORE_MODE,
                    choices=["always", "escalate", "never"])
    ap.add_argument("--escalate_above", type=float, default=config.ESCALATE_ABOVE)
    ap.add_argument("--threshold", type=float, default=config.MATCH_THRESHOLD)
    ap.add_argument("--max_subjects", type=int, default=2)

    ap.add_argument("--no_denoise", action="store_true")
    ap.add_argument("--no_detect", action="store_true")
    ap.add_argument("--no_restore", action="store_true")
    ap.add_argument("--no_embed", action="store_true")
    ap.add_argument("--preview", action="store_true", help="show a local window too")
    args = ap.parse_args()

    print("=" * 58)
    print("node %s  ->  %s" % (args.name, args.hub))
    print("=" * 58)
    stages = Stages(args)
    print("-" * 58)

    cap = open_capture(args.source, args.cap_width, args.cap_height)
    if not cap.isOpened():
        raise SystemExit("could not open camera source: %s" % args.source)

    sender = Sender(args.hub)
    sender.start()

    timer = StageTimer(window=15)
    interval = 1.0 / max(args.fps, 0.1)
    seq = 0
    fps_ema = 0.0
    next_due = time.monotonic()

    print("streaming at up to %.1f obs/s - Ctrl+C to stop\n" % args.fps)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[cam] read failed; reopening")
                cap.release()
                time.sleep(1.0)
                cap = open_capture(args.source, args.cap_width, args.cap_height)
                continue

            now = time.monotonic()
            if now < next_due:
                time.sleep(min(0.005, next_due - now))
                continue
            next_due = now + interval

            t0 = time.perf_counter()
            subjects, annotated = process_frame(frame, stages, timer, args)
            elapsed = time.perf_counter() - t0
            inst = 1.0 / max(elapsed, 1e-6)
            fps_ema = inst if fps_ema == 0 else 0.85 * fps_ema + 0.15 * inst

            seq += 1
            sender.submit(build_observation(
                node=args.name,
                seq=seq,
                subjects=subjects,
                frame_shape=annotated.shape,
                fps=min(fps_ema, args.fps),
                stages_ms=timer.summary(),
                frame_thumb=encode_jpeg(annotated, args.thumb_quality, args.thumb_width),
            ))

            if seq % 25 == 0:
                print("  seq %-6d sent %-6d dropped %-4d %s  %s"
                      % (seq, sender.sent, sender.dropped,
                         "online" if sender.connected else "OFFLINE",
                         timer.summary()))

            if args.preview:
                cv2.imshow("node %s" % args.name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        sender.shutdown()
        cap.release()
        cv2.destroyAllWindows()
        print("sent %d observations, dropped %d" % (sender.sent, sender.dropped))


if __name__ == "__main__":
    main()
