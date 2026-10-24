"""Batch crop restoration - run stage 1 over a folder of images.

    python denoise.py --input_dir crops --output_dir out --compare

For the live stream use ``main_pipeline.py``; this is for offline sweeps and
for producing before/after strips for a report.

``--compare`` writes ``noisy | restored`` strips, which is the only test
that really counts: judge it by eye on real frames, because the metric that
matters downstream is whether ArcFace can embed the face, not PSNR.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import config
from pipeline.stage1_denoise import ClassicalDenoiser, estimate_noise_sigma

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
NOISE_SUFFIX = "_noise"


def strip_noise_suffix(stem):
    return stem[: -len(NOISE_SUFFIX)] if stem.lower().endswith(NOISE_SUFFIX) else stem


def main():
    parser = argparse.ArgumentParser(description="Batch classical crop restoration")
    parser.add_argument("--input_dir", "--noise_dir", dest="input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", "--denoised_dir", dest="output_dir", required=True, type=Path)
    parser.add_argument("--profile", default=config.DENOISE_PROFILE,
                        choices=["off", "fast", "balanced", "quality"])
    parser.add_argument("--clahe", action="store_true",
                        help="illumination normalisation; off by default because "
                             "it measurably reduces fidelity (see the README)")
    parser.add_argument("--sharpen", type=float, default=None,
                        help="unsharp amount, e.g. 0.6; off by default")
    parser.add_argument("--fixed_strength", action="store_true",
                        help="disable adaptive strength and treat every crop alike")
    parser.add_argument("--ext", default="png", choices=["png", "jpg"])
    parser.add_argument("--compare", action="store_true",
                        help="also write noisy|restored side-by-side strips")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    it = args.input_dir.rglob("*") if args.recursive else args.input_dir.glob("*")
    image_paths = sorted(p for p in it if p.suffix.lower() in VALID_EXTENSIONS)
    if not image_paths:
        print("Error: no images found in %s" % args.input_dir)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    compare_dir = args.output_dir / "compare"
    if args.compare:
        compare_dir.mkdir(parents=True, exist_ok=True)

    denoiser = ClassicalDenoiser(
        profile=args.profile,
        adaptive=not args.fixed_strength,
        clahe=True if args.clahe else None,
        sharpen=args.sharpen,
    )

    print("restorer : %s" % denoiser.describe())
    print("images   : %d" % len(image_paths))
    print("-" * 62)

    processed, skipped, t0 = 0, 0, time.time()
    for i, src in enumerate(image_paths, 1):
        dst = args.output_dir / ("%s.%s" % (strip_noise_suffix(src.stem), args.ext))
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            print("  skip (unreadable): %s" % src.name)
            skipped += 1
            continue

        sigma_in = estimate_noise_sigma(img)
        t_img = time.perf_counter()
        out = denoiser.denoise(img)
        elapsed = (time.perf_counter() - t_img) * 1000

        cv2.imwrite(str(dst), out)
        if args.compare:
            cv2.imwrite(str(compare_dir / ("%s_cmp.png" % dst.stem)), np.hstack([img, out]))

        processed += 1
        print("  %5d/%d  %-34s %4dx%-4d  sigma %5.2f -> %5.2f  %6.1f ms"
              % (i, len(image_paths), src.name[:34], img.shape[1], img.shape[0],
                 sigma_in, estimate_noise_sigma(out), elapsed))

    total = time.time() - t0
    print("-" * 62)
    print("processed %d image(s), skipped %d, in %.2fs%s"
          % (processed, skipped, total,
             " (%.1f ms/image)" % (total / processed * 1000) if processed else ""))
    print("output: %s" % args.output_dir)


if __name__ == "__main__":
    main()
