"""Per-stage latency, throughput and power benchmark.

    python benchmark.py --resolution 1280x720 --iters 30
    python benchmark.py --source clip.mp4 --iters 100 --power

Produces the numbers the competition slides need: ms per stage, end-to-end
FPS, and (on a Jetson, via tegrastats) the power draw against the 10-15W
budget.
"""

import argparse
import json
import platform
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

import config


class PowerMonitor:
    """Samples ``tegrastats`` in the background. No-ops off the Jetson."""

    def __init__(self, interval_ms: int = 200):
        self.interval_ms = interval_ms
        self.samples: List[float] = []
        self._proc = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.available = Path("/usr/bin/tegrastats").exists()

    def _reader(self):
        pattern = re.compile(r"(?:VDD_IN|POM_5V_IN)\s+(\d+)mW")
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            m = pattern.search(line)
            if m:
                self.samples.append(int(m.group(1)) / 1000.0)

    def start(self):
        if not self.available:
            return self
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            # universal_newlines, not text=: the latter is 3.7+, and JetPack
            # 4.6 on the Nano ships Python 3.6.
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True,
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        if not self.available or self._proc is None:
            return {}
        self._stop.set()
        self._proc.terminate()
        if self._thread:
            self._thread.join(timeout=2.0)
        if not self.samples:
            return {}
        return {
            "watts_mean": round(float(np.mean(self.samples)), 2),
            "watts_peak": round(float(np.max(self.samples)), 2),
            "samples": len(self.samples),
        }


def timed(fn, *fn_args, warmup: int = 3, iters: int = 20):
    """Returns ``(result, {mean, p50, p95, min, max} in ms)``."""
    result = None
    for _ in range(warmup):
        result = fn(*fn_args)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = fn(*fn_args)
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    return result, {
        "mean": round(float(arr.mean()), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "min": round(float(arr.min()), 2),
        "max": round(float(arr.max()), 2),
    }


def get_frame(args) -> np.ndarray:
    if args.source:
        if Path(args.source).exists():
            cap = cv2.VideoCapture(args.source)
            ok, frame = cap.read()
            cap.release()
            if ok:
                return frame
        print(f"[bench] could not read {args.source}; using a synthetic frame")

    w, h = (int(v) for v in args.resolution.lower().split("x"))
    rng = np.random.default_rng(0)
    # Structured content plus noise, so the detector has something to find
    # and the denoiser is not just fitting flat gradients.
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = (40, 45, 55)
    cv2.rectangle(frame, (w // 3, h // 4), (2 * w // 3, h), (120, 110, 100), -1)
    cv2.circle(frame, (w // 2, h // 4), min(w, h) // 10, (190, 170, 160), -1)
    noise = rng.normal(0, 14, frame.shape).astype(np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Benchmark the edge pipeline")
    ap.add_argument("--source", default=None, help="video file to pull a real frame from")
    ap.add_argument("--resolution", default="1280x720")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--power", action="store_true", help="sample tegrastats (Jetson only)")
    ap.add_argument("--cameras", type=int, default=len(config.CAMERAS),
                    help="per-camera stages are multiplied by this")
    ap.add_argument("--target_fps", type=float, default=config.CAPTURE_FPS)
    ap.add_argument("--denoise_scope", default=config.DENOISE_SCOPE, choices=["frame", "crop"])
    ap.add_argument("--out", type=Path, default=config.RUNS_DIR / "benchmark.json")
    args = ap.parse_args()

    frame = get_frame(args)
    h, w = frame.shape[:2]

    print("=" * 62)
    print(f"platform  : {platform.platform()}")
    print(f"provider  : {config.EXEC_PROVIDER}")
    print(f"frame     : {w}x{h}   iters {args.iters} (warmup {args.warmup})")
    print(f"cameras   : {args.cameras}   denoise scope: {args.denoise_scope}   "
          f"target {args.target_fps:.0f} fps")
    print("=" * 62)
    print("Stages 1-4 are timed once and multiplied by the camera count; "
          "stage 5 runs once per track.\n")

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "provider": config.EXEC_PROVIDER,
        "resolution": f"{w}x{h}",
        "iters": args.iters,
        "cameras": args.cameras,
        "denoise_scope": args.denoise_scope,
        "target_fps": args.target_fps,
        "stages": {},
    }

    monitor = PowerMonitor().start() if args.power else None

    # --- Stage 1 -----------------------------------------------------------
    # In "crop" scope the denoiser never sees the full frame, only a person
    # box, so timing it on the whole frame would overstate the cost badly.
    denoise_input = frame
    if args.denoise_scope == "crop":
        denoise_input = frame[h // 4 :, w // 3 : 2 * w // 3]

    denoised = frame
    try:
        from pipeline.stage1_denoise import get_denoiser

        denoiser = get_denoiser()
        out, stats = timed(denoiser.denoise, denoise_input, warmup=args.warmup, iters=args.iters)
        if args.denoise_scope == "frame":
            denoised = out
        stats["scope"] = args.denoise_scope
        stats["profile"] = denoiser.describe()
        stats["input"] = f"{denoise_input.shape[1]}x{denoise_input.shape[0]}"
        report["stages"]["1_denoise"] = stats
        print(f"1 denoise  : {stats['mean']:7.2f} ms  (p95 {stats['p95']:.2f})  "
              f"{stats['input']} {args.denoise_scope}, {stats['profile']}")
    except Exception as exc:
        print(f"1 denoise  : skipped ({exc})")

    # --- Stage 2 -----------------------------------------------------------
    subjects = []
    try:
        from pipeline.stage2_detect import get_detector

        detector = get_detector()
        subjects, stats = timed(detector.detect, denoised, warmup=args.warmup, iters=args.iters)
        stats["subjects"] = len(subjects)
        report["stages"]["2_detect"] = stats
        print(f"2 detect   : {stats['mean']:7.2f} ms  (p95 {stats['p95']:.2f})  "
              f"{len(subjects)} subject(s)")
    except Exception as exc:
        print(f"2 detect   : skipped ({exc})")

    crop = subjects[0].crop if subjects else frame[h // 4 : h, w // 3 : 2 * w // 3]

    # --- Stage 3 -----------------------------------------------------------
    # Unlike stage 1, this cost does NOT scale with the crop. GFPGAN aligns
    # the face and works at a fixed 512x512 internally, so a 40px face and a
    # 400px one cost the same. The per-face figure is the whole story.
    face = None
    try:
        from pipeline.stage3_restore import get_restorer

        restorer = get_restorer()
        face, stats = timed(restorer.restore, crop, warmup=1, iters=max(3, args.iters // 2))
        stats["per_face"] = True
        report["stages"]["3_restore"] = stats
        print(f"3 restore  : {stats['mean']:7.2f} ms  (p95 {stats['p95']:.2f})  "
              f"{'face found' if face is not None else 'no face in the crop'}")

        # What that figure means for the loop, which is the decision it drives.
        faces = args.cameras * 2
        always = stats["mean"] * faces
        print(f"             always : {always:8.0f} ms/cycle for {faces} faces "
              f"-> {1000.0 / always:.2f} pairs/s" if always else "")
        print(f"             escalate: pays it only on candidates; most cycles "
              f"skip stage 3 entirely")
        if stats["mean"] > 400:
            print("             ^ too slow for a live loop. Use "
                  "--restore_mode never, or escalate if the gallery is small.")
    except Exception as exc:
        print(f"3 restore  : skipped ({exc})")

    probe = face if face is not None else cv2.resize(crop, (112, 112))

    # --- Stage 4 -----------------------------------------------------------
    embedding = None
    try:
        from pipeline.stage4_embed import get_embedder

        embedder = get_embedder()
        embedding, stats = timed(embedder.embed, probe, warmup=args.warmup, iters=args.iters)
        report["stages"]["4_embed"] = stats
        print(f"4 embed    : {stats['mean']:7.2f} ms  (p95 {stats['p95']:.2f})")
    except Exception as exc:
        print(f"4 embed    : skipped ({exc})")

    # --- Stage 5 -----------------------------------------------------------
    try:
        from pipeline.stage5_match import get_index

        index = get_index()
        if embedding is None:
            embedding = np.random.rand(config.EMBEDDING_DIM).astype(np.float32)
        if index.index.ntotal == 0:
            # Synthetic population, so the search cost is representative.
            rng = np.random.default_rng(0)
            for i in range(1000):
                index.add(rng.normal(size=config.EMBEDDING_DIM).astype(np.float32),
                          f"_bench_{i}", persist=False)
            print("           (populated 1000 synthetic vectors for timing)")

        _, stats = timed(index.query, embedding, warmup=args.warmup, iters=max(50, args.iters))
        stats["vectors"] = int(index.index.ntotal)
        stats["backend"] = index.stats()["backend"]
        report["stages"]["5_match"] = stats
        print(f"5 match    : {stats['mean']:7.3f} ms  over {stats['vectors']} vectors "
              f"({stats['backend']})")
    except Exception as exc:
        print(f"5 match    : skipped ({exc})")

    # --- Cross-view fusion --------------------------------------------------
    if args.cameras > 1:
        try:
            from pipeline.fusion import CrossViewFusion, ViewObservation

            rng = np.random.default_rng(1)
            views = {}
            base = rng.normal(size=config.EMBEDDING_DIM).astype(np.float32)
            for c in range(args.cameras):
                obs = [
                    ViewObservation(
                        camera=f"CAM_{c}", box=(0, 0, 10, 10), conf=0.9,
                        crop=np.zeros((2, 2, 3), np.uint8),
                        embedding=(base + 0.1 * rng.normal(size=config.EMBEDDING_DIM)).astype(np.float32),
                    )
                    for _ in range(2)
                ]
                views[f"CAM_{c}"] = obs

            from pipeline.stage5_match import get_index as _gi
            fusion = CrossViewFusion(index=_gi())
            _, stats = timed(fusion.process, views, warmup=args.warmup, iters=max(50, args.iters))
            report["stages"]["6_fuse"] = stats
            print(f"6 fuse     : {stats['mean']:7.3f} ms  "
                  f"(associate {args.cameras} views + fused query)")
        except Exception as exc:
            print(f"6 fuse     : skipped ({exc})")

    # --- Totals ------------------------------------------------------------
    # Stages 1-4 run once per camera; stages 5-6 run once per frame pair.
    per_camera_keys = {"1_denoise", "2_detect", "3_restore", "4_embed"}
    per_camera_ms = sum(
        s["mean"] for k, s in report["stages"].items() if k in per_camera_keys
    )
    shared_ms = sum(
        s["mean"] for k, s in report["stages"].items() if k not in per_camera_keys
    )
    total_ms = per_camera_ms * args.cameras + shared_ms

    report["per_camera_ms"] = round(per_camera_ms, 2)
    report["shared_ms"] = round(shared_ms, 2)
    report["total_ms"] = round(total_ms, 2)
    report["fps"] = round(1000.0 / total_ms, 2) if total_ms else 0.0
    report["single_camera_fps"] = (
        round(1000.0 / (per_camera_ms + shared_ms), 2) if per_camera_ms + shared_ms else 0.0
    )

    print("-" * 62)
    print(f"per camera : {per_camera_ms:7.2f} ms  x {args.cameras} cameras")
    print(f"shared     : {shared_ms:7.2f} ms  (match + fuse, once per pair)")
    print(f"end-to-end : {total_ms:7.2f} ms  ->  {report['fps']:.2f} pairs/s")

    # The honest comparison against the capture rate.
    budget_ms = 1000.0 / args.target_fps
    if total_ms > budget_ms:
        stride = int(np.ceil(total_ms / budget_ms))
        report["frames_dropped_pct"] = round(100.0 * (1 - 1 / stride), 1)
        report["effective_stride"] = stride
        print(f"\n{args.target_fps:.0f} fps needs {budget_ms:.1f} ms/pair; this is "
              f"{total_ms / budget_ms:.1f}x over budget.")
        print(f"The loop will process roughly every {stride}th frame pair "
              f"(~{report['frames_dropped_pct']:.0f}% of frames dropped).")
        if "1_denoise" in report["stages"]:
            share = 100.0 * report["stages"]["1_denoise"]["mean"] / per_camera_ms
            print(f"Denoising is {share:.0f}% of the per-camera cost - "
                  "try --denoise_scope crop, INT8, or a narrower model.")
    else:
        print(f"\nKeeps up with {args.target_fps:.0f} fps "
              f"({budget_ms:.1f} ms budget, {total_ms:.1f} ms used).")

    if monitor:
        power = monitor.stop()
        report["power"] = power
        if power:
            print(f"power      : {power['watts_mean']:.2f} W mean, "
                  f"{power['watts_peak']:.2f} W peak "
                  f"({'within' if power['watts_peak'] <= 15 else 'OVER'} the 15 W budget)")
        else:
            print("power      : tegrastats unavailable (not a Jetson?)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"report     : {args.out}")


if __name__ == "__main__":
    main()
