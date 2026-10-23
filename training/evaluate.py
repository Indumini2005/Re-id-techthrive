"""Evaluate a trained denoiser: PSNR / SSIM against the noisy baseline.

    python -m training.evaluate --ckpt runs/nafnet/best.pth --data_dir data/val

Add ``--onnx models/nafnet.onnx`` instead of ``--ckpt`` to score the exported
graph, which is what actually runs on the Jetson - worth checking that export
and quantization did not cost you anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from training.checkpoint import build_from_checkpoint, load_checkpoint  # noqa: E402
from training.dataset import FullImageDataset, to_image  # noqa: E402
from training.losses import psnr, ssim  # noqa: E402


def load_torch_model(ckpt_path: Path, device: torch.device, use_ema: bool = True):
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model, _ = build_from_checkpoint(ckpt, use_ema=use_ema)
    return model.to(device).eval()


class OnnxRunner:
    """Scores the exported graph through the real deployment path.

    It routes through ``pipeline.stage1_denoise.Denoiser`` rather than
    calling the session directly, for two reasons. The graph is exported
    with a fixed tile size, so a whole frame cannot be fed to it at all.
    And tiling plus Hann blending is what actually runs on the device, so
    measuring anything else would report a number the Jetson never achieves.
    """

    def __init__(self, path: Path, provider: str = "cpu", tile: int = None,
                 overlap: int = None):
        from pipeline.stage1_denoise import Denoiser

        self.denoiser = Denoiser(
            model_path=path,
            tile_size=tile if tile is not None else config.TILE_SIZE,
            overlap=overlap if overlap is not None else config.TILE_OVERLAP,
            provider=provider,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        import cv2

        out = []
        for sample in x:                                   # N,3,H,W RGB [0,1]
            rgb = (sample.cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            denoised = self.denoiser.denoise(bgr)
            back = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            out.append(back.transpose(2, 0, 1))
        return torch.from_numpy(np.stack(out))


def main():
    ap = argparse.ArgumentParser(description="Evaluate the stage-1 denoiser")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--onnx", type=Path, default=None)
    ap.add_argument("--data_dir", type=Path, required=True, help="CLEAN reference images")
    ap.add_argument("--noisy_dir", type=Path, default=None, help="paired noisy images (optional)")
    ap.add_argument("--preset", default="cctv", choices=["light", "cctv", "harsh"])
    ap.add_argument("--max_side", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save_dir", type=Path, default=None, help="write side-by-side comparisons")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--tile", type=int, default=None,
                    help="tile size for --onnx; must match what was exported")
    ap.add_argument("--overlap", type=int, default=None)
    args = ap.parse_args()

    if not (args.ckpt or args.onnx):
        raise SystemExit("pass --ckpt or --onnx")

    device = torch.device("cpu" if args.onnx else args.device)
    model = (
        OnnxRunner(args.onnx, provider=config.EXEC_PROVIDER,
                   tile=args.tile, overlap=args.overlap)
        if args.onnx
        else load_torch_model(args.ckpt, device, use_ema=not args.no_ema)
    )

    dataset = FullImageDataset(
        args.data_dir, args.noisy_dir, preset=args.preset,
        max_side=args.max_side, max_images=args.limit or None,
    )
    if len(dataset) == 0:
        raise SystemExit(f"no images under {args.data_dir}")

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t_total = 0.0
    for i in range(len(dataset)):
        noisy, clean, name = dataset[i]
        noisy_b, clean_b = noisy.unsqueeze(0).to(device), clean.unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model(noisy_b)
        elapsed = (time.perf_counter() - t0) * 1000
        t_total += elapsed

        pred = pred.float().clamp(0, 1).to(clean_b.device)
        row = {
            "image": name,
            "psnr_in": round(psnr(noisy_b, clean_b), 3),
            "psnr_out": round(psnr(pred, clean_b), 3),
            "ssim_in": round(ssim(noisy_b, clean_b), 4),
            "ssim_out": round(ssim(pred, clean_b), 4),
            "ms": round(elapsed, 1),
        }
        row["psnr_gain"] = round(row["psnr_out"] - row["psnr_in"], 3)
        rows.append(row)

        print(
            f"{i+1:4d}/{len(dataset)} {name[:34]:34s} "
            f"PSNR {row['psnr_in']:6.2f} -> {row['psnr_out']:6.2f} "
            f"({row['psnr_gain']:+.2f} dB)  {row['ms']:7.1f} ms"
        )

        if args.save_dir:
            strip = np.hstack([to_image(noisy[0] if noisy.ndim == 4 else noisy),
                               to_image(pred[0]),
                               to_image(clean[0] if clean.ndim == 4 else clean)])
            cv2.imwrite(str(args.save_dir / f"{Path(name).stem}_cmp.png"), strip)

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    print("-" * 70)
    print(f"images        : {len(rows)}")
    print(f"PSNR  noisy   : {mean('psnr_in'):.3f} dB")
    print(f"PSNR  denoised: {mean('psnr_out'):.3f} dB   ({mean('psnr_gain'):+.3f} dB)")
    print(f"SSIM  noisy   : {mean('ssim_in'):.4f}")
    print(f"SSIM  denoised: {mean('ssim_out'):.4f}")
    print(f"latency       : {t_total / len(rows):.1f} ms/image")

    report = {
        "model": str(args.onnx or args.ckpt),
        "images": len(rows),
        "psnr_in": round(mean("psnr_in"), 3),
        "psnr_out": round(mean("psnr_out"), 3),
        "psnr_gain": round(mean("psnr_gain"), 3),
        "ssim_in": round(mean("ssim_in"), 4),
        "ssim_out": round(mean("ssim_out"), 4),
        "ms_per_image": round(t_total / len(rows), 2),
        "per_image": rows,
    }
    out_json = config.RUNS_DIR / "eval_report.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"report        : {out_json}")


if __name__ == "__main__":
    main()
