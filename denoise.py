"""Batch denoising CLI - run the trained NAFNet over a folder of images.

    python denoise.py --input_dir data/noisy --output_dir data/denoised
    python denoise.py --input_dir frames --output_dir out --tile 0 --compare

For the live stream use ``main_pipeline.py``; this is for offline sweeps,
dataset preparation and before/after comparisons for the report.

Images are processed one at a time but their tiles are batched, which is what
actually keeps the accelerator busy. Spawning one process per image (the usual
first instinct) reloads the model in every worker and is slower on any machine
with a GPU.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import config
from pipeline.stage1_denoise import Denoiser

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
NOISE_SUFFIX = "_noise"


def strip_noise_suffix(stem: str) -> str:
    return stem[: -len(NOISE_SUFFIX)] if stem.lower().endswith(NOISE_SUFFIX) else stem


def main():
    parser = argparse.ArgumentParser(description="Batch NAFNet denoising")
    parser.add_argument("--input_dir", "--noise_dir", dest="input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", "--denoised_dir", dest="output_dir", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=config.NAFNET_ONNX)
    parser.add_argument("--tile", type=int, default=config.TILE_SIZE,
                        help="tile size; 0 runs the whole image in one pass")
    parser.add_argument("--overlap", type=int, default=config.TILE_OVERLAP)
    parser.add_argument("--batch_tiles", type=int, default=4)
    parser.add_argument("--provider", default=config.EXEC_PROVIDER,
                        choices=["tensorrt", "cuda", "cpu"])
    parser.add_argument("--threads", type=int, default=0,
                        help="ONNX Runtime intra-op threads (CPU only; 0 = auto)")
    parser.add_argument("--ext", default="png", choices=["png", "jpg"])
    parser.add_argument("--compare", action="store_true",
                        help="also write noisy|denoised side-by-side strips")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="redo images that already exist in the output folder")
    args = parser.parse_args()

    it = args.input_dir.rglob("*") if args.recursive else args.input_dir.glob("*")
    image_paths = sorted(p for p in it if p.suffix.lower() in VALID_EXTENSIONS)
    if not image_paths:
        print(f"Error: no images found in {args.input_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    compare_dir = args.output_dir / "compare"
    if args.compare:
        compare_dir.mkdir(parents=True, exist_ok=True)

    cv2.setNumThreads(0 if args.provider != "cpu" else cv2.getNumThreads())

    denoiser = Denoiser(
        model_path=args.model,
        tile_size=args.tile,
        overlap=args.overlap,
        batch_size=args.batch_tiles,
        provider=args.provider,
        intra_op_threads=args.threads,
    )
    print(f"model    : {args.model.name}  ({denoiser.session.get_providers()[0]})")
    print(f"tiling   : {'off (full frame)' if not args.tile else f'{args.tile}px / {args.overlap}px overlap'}")
    print(f"images   : {len(image_paths)}")
    print("-" * 60)

    processed, skipped, t0 = 0, 0, time.time()
    for i, src in enumerate(image_paths, 1):
        dst = args.output_dir / f"{strip_noise_suffix(src.stem)}.{args.ext}"
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  skip (unreadable): {src.name}")
            skipped += 1
            continue

        t_img = time.perf_counter()
        out = denoiser.denoise(img)
        elapsed = (time.perf_counter() - t_img) * 1000

        cv2.imwrite(str(dst), out)
        if args.compare:
            cv2.imwrite(str(compare_dir / f"{dst.stem}_cmp.png"), np.hstack([img, out]))

        processed += 1
        print(f"  {i:5d}/{len(image_paths)}  {src.name[:40]:40s} "
              f"{img.shape[1]}x{img.shape[0]}  {elapsed:8.1f} ms")

    total = time.time() - t0
    print("-" * 60)
    print(f"processed {processed} image(s), skipped {skipped}, in {total:.2f}s"
          + (f" ({total / processed * 1000:.1f} ms/image)" if processed else ""))
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()
