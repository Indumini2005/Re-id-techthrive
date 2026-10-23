"""Build a paired noisy/clean dataset on disk.

Two sources:

1. **Images** - degrade a folder of sharp stills.

       python -m training.make_dataset images --src data/raw --out data/train

2. **Video** - pull sharp frames out of your own CCTV/phone footage, then
   degrade them. Frames below a blur threshold are dropped, so you do not
   train against already-soft references.

       python -m training.make_dataset video --src clips/ --out data/train --fps 2

Output layout:

    out/
      clean/0000001.png
      noisy/0000001.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.dataset import list_images  # noqa: E402
from training.degradations import PRESETS, degrade  # noqa: E402

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian - a cheap, reliable blur score."""
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def write_pair(clean: np.ndarray, out_dir: Path, index: int, preset: str) -> None:
    (out_dir / "clean").mkdir(parents=True, exist_ok=True)
    (out_dir / "noisy").mkdir(parents=True, exist_ok=True)
    noisy = degrade(clean, PRESETS[preset])
    cv2.imwrite(str(out_dir / "clean" / f"{index:07d}.png"), clean)
    cv2.imwrite(str(out_dir / "noisy" / f"{index:07d}.png"), noisy)


def from_images(args) -> int:
    paths = list_images(args.src)
    print(f"found {len(paths)} images under {args.src}")
    written = 0
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        if min(h, w) < args.min_size:
            continue
        if args.max_side and max(h, w) > args.max_side:
            s = args.max_side / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        if args.min_sharpness and sharpness(img) < args.min_sharpness:
            continue
        write_pair(img, args.out, written, args.preset)
        written += 1
        if written % 200 == 0:
            print(f"  {written} pairs...")
    return written


def from_video(args) -> int:
    src = Path(args.src)
    videos = (
        [src] if src.is_file()
        else sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    )
    print(f"found {len(videos)} video(s)")

    written = 0
    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  skip (unreadable): {video_path.name}")
            continue

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, int(round(src_fps / max(args.fps, 0.1))))
        frame_i = 0
        taken = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_i % stride == 0:
                h, w = frame.shape[:2]
                if args.max_side and max(h, w) > args.max_side:
                    s = args.max_side / max(h, w)
                    frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                if not args.min_sharpness or sharpness(frame) >= args.min_sharpness:
                    write_pair(frame, args.out, written, args.preset)
                    written += 1
                    taken += 1
                    if args.per_video and taken >= args.per_video:
                        break
            frame_i += 1

        cap.release()
        print(f"  {video_path.name}: {taken} frames kept")
    return written


def main():
    ap = argparse.ArgumentParser(description="Generate paired denoising data")
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--src", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--preset", default="cctv", choices=list(PRESETS))
        p.add_argument("--max_side", type=int, default=1280, help="0 disables downscaling")
        p.add_argument("--min_sharpness", type=float, default=60.0,
                       help="drop frames blurrier than this; 0 disables")

    p_img = sub.add_parser("images", help="degrade a folder of stills")
    common(p_img)
    p_img.add_argument("--min_size", type=int, default=256)

    p_vid = sub.add_parser("video", help="extract sharp frames from video, then degrade")
    common(p_vid)
    p_vid.add_argument("--fps", type=float, default=2.0, help="frames sampled per second")
    p_vid.add_argument("--per_video", type=int, default=0, help="cap per clip; 0 = no cap")

    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    written = from_images(args) if args.mode == "images" else from_video(args)

    print("-" * 60)
    print(f"wrote {written} pairs to {args.out}")
    print(f"train with:\n  python -m training.train_denoiser "
          f"--train_dir {args.out / 'clean'} --noisy_train_dir {args.out / 'noisy'}")


if __name__ == "__main__":
    main()
