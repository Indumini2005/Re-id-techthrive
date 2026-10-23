"""Export a trained NAFNet checkpoint to ONNX.

    python export/export_onnx.py --ckpt runs/nafnet/best.pth

By default the graph takes a dynamic batch and a fixed 256x256 tile, which is
what ``pipeline/stage1_denoise.py`` feeds it. Pass ``--dynamic_hw`` for a
fully dynamic graph (handy for whole-frame inference, slower under TensorRT
because it cannot specialize the kernels).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from training.checkpoint import build_from_checkpoint, load_checkpoint  # noqa: E402
from training.nafnet import count_parameters  # noqa: E402


def load_model(ckpt_path: Path, use_ema: bool = True):
    ckpt = load_checkpoint(ckpt_path)
    model, arch = build_from_checkpoint(ckpt, use_ema=use_ema)
    print(f"loaded {ckpt_path.name} "
          f"({'EMA' if use_ema and ckpt.get('ema') else 'raw'} weights, "
          f"arch {arch.get('name') or 'custom'} w{arch.get('width')}, "
          f"{count_parameters(model)/1e6:.2f}M params, "
          f"best PSNR {ckpt.get('best_psnr', float('nan')):.3f} dB)")
    return model


def main():
    ap = argparse.ArgumentParser(description="NAFNet -> ONNX")
    ap.add_argument("--ckpt", type=Path, default=config.RUNS_DIR / "nafnet" / "best.pth")
    ap.add_argument("--out", type=Path, default=config.NAFNET_ONNX)
    ap.add_argument("--tile", type=int, default=config.TILE_SIZE)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamic_batch", action="store_true", default=True)
    ap.add_argument("--static_batch", dest="dynamic_batch", action="store_false")
    ap.add_argument("--dynamic_hw", action="store_true", help="also make H/W dynamic")
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--fp16", action="store_true", help="store weights in fp16")
    ap.add_argument("--simplify", action="store_true", help="run onnx-simplifier if installed")
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt}")

    model = load_model(args.ckpt, use_ema=not args.no_ema)
    dummy = torch.randn(args.batch, 3, args.tile, args.tile)

    dynamic_axes = {}
    if args.dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
    if args.dynamic_hw:
        dynamic_axes.setdefault("input", {}).update({2: "height", 3: "width"})
        dynamic_axes.setdefault("output", {}).update({2: "height", 3: "width"})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = dict(
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes or None,
    )

    # torch 2.6+ defaults to the dynamo exporter, which needs onnxscript and
    # emits a more elaborate graph. The TorchScript path is the right choice
    # here: it is available without extra packages and produces the plain
    # conv/mul/pixel-shuffle graph that JetPack 4.6's ONNX Runtime 1.11
    # understands. Older torch has no `dynamo` argument at all.
    try:
        torch.onnx.export(model, dummy, str(args.out), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(args.out), **export_kwargs)
    print(f"wrote {args.out}")

    # -- optional graph simplification
    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify

            model_onnx = onnx.load(str(args.out))
            simplified, ok = simplify(model_onnx)
            if ok:
                onnx.save(simplified, str(args.out))
                print("simplified graph")
            else:
                print("onnxsim could not validate the simplified graph; kept the original")
        except ImportError:
            print("onnx-simplifier not installed; skipping (pip install onnxsim)")

    if args.fp16:
        try:
            import onnx
            from onnxconverter_common import float16

            model_onnx = onnx.load(str(args.out))
            onnx.save(float16.convert_float_to_float16(model_onnx), str(args.out))
            print("converted weights to fp16")
        except ImportError:
            print("onnxconverter-common not installed; skipping fp16 conversion")

    # -- verify parity against PyTorch
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
        test = torch.rand(1, 3, args.tile, args.tile)
        with torch.no_grad():
            torch_out = model(test).numpy()
        onnx_out = session.run(None, {session.get_inputs()[0].name: test.numpy()})[0]

        diff = float(np.abs(torch_out - onnx_out).max())
        print(f"max |torch - onnx| = {diff:.3e}  {'OK' if diff < 1e-3 else 'CHECK THIS'}")
    except ImportError:
        print("onnxruntime not installed; skipped the parity check")

    size_mb = args.out.stat().st_size / 1e6
    print(f"size: {size_mb:.1f} MB")
    print(f"\nnext:  python export/export_tensorrt.py --onnx {args.out} --precision fp16")


if __name__ == "__main__":
    main()
