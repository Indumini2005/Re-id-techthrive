"""Load a pretrained NAFNet checkpoint from elsewhere and judge whether it
can actually run on this project's target hardware.

    python -m training.pretrained inspect --ckpt NAFNet-SIDD-width32.pth
    python -m training.pretrained to_onnx --ckpt NAFNet-SIDD-width32.pth --tile 128

The architecture is **inferred from the checkpoint's own tensors** rather
than read from a config file or hardcoded here. Released NAFNet variants
differ in width and block counts, the published YAMLs are easy to mismatch,
and a wrong guess produces either a load error or - far worse - a model that
loads partially and quietly performs badly. Counting the tensors cannot
drift out of date.

Where to get weights:

* **NAFNet** - github.com/megvii-research/NAFNet, "Results and Pre-trained
  Models" in the README. The SIDD variants are trained on real smartphone
  sensor noise, which is the closest public match to CCTV sensor noise.
* **SCUNet** - github.com/cszn/SCUNet. Its training degradation (blur,
  resize, noise, JPEG) is close to this project's CCTV chain, but it uses
  Swin transformer blocks and is not a NAFNet, so it will not load here.
* **FFDNet / DnCNN** - github.com/cszn/KAIR. Small enough for a Nano, but
  trained on synthetic Gaussian noise only.

Verify any architecture detail against the source repository before quoting
it; the inference below is what this code actually relies on.
"""

import argparse
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402
from training.nafnet import NAFNet, count_parameters  # noqa: E402

# Jetson Nano, 128 Maxwell cores: ~472 GFLOPS FP16 theoretical. Sustained
# throughput on a real graph is a fraction of peak; 30% is a fair planning
# figure for a conv-heavy network with no tensor cores.
NANO_GFLOPS_PEAK = 472.0
NANO_EFFICIENCY = 0.30


def unwrap_state_dict(obj):
    """Pull the tensor dict out of whatever wrapper a release used."""
    for key in ("params", "params_ema", "state_dict", "model", "ema"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
    if isinstance(obj, dict) and any(isinstance(v, torch.Tensor) for v in obj.values()):
        # Strip a DataParallel prefix if present.
        if all(k.startswith("module.") for k in obj):
            obj = {k[len("module."):]: v for k, v in obj.items()}
        return obj
    raise ValueError("could not find a tensor state dict in this checkpoint")


def infer_nafnet_arch(state):
    """Recover width and block counts from the tensors themselves."""
    if "intro.weight" not in state:
        raise ValueError(
            "no 'intro.weight' tensor - this does not look like a NAFNet "
            "checkpoint. SCUNet, Restormer and MPRNet are different "
            "architectures and cannot be loaded into this model."
        )

    width = int(state["intro.weight"].shape[0])

    def count_stage(prefix):
        """How many blocks in each stage of encoders./decoders."""
        stages = {}
        pattern = re.compile(r"^%s\.(\d+)\.(\d+)\." % prefix)
        for key in state:
            m = pattern.match(key)
            if m:
                stage, block = int(m.group(1)), int(m.group(2))
                stages[stage] = max(stages.get(stage, -1), block)
        return [stages[i] + 1 for i in sorted(stages)]

    enc = count_stage("encoders")
    dec = count_stage("decoders")

    middle = 0
    pattern = re.compile(r"^middle_blks\.(\d+)\.")
    for key in state:
        m = pattern.match(key)
        if m:
            middle = max(middle, int(m.group(1)) + 1)

    if not enc or not dec:
        raise ValueError("could not determine encoder/decoder depth")

    return {"width": width, "enc": enc, "middle": middle, "dec": dec}


def estimate_gmacs(model, height, width):
    """Multiply-accumulates for one forward pass, by hooking conv layers.

    Convolutions dominate NAFNet; the pointwise multiplies in SimpleGate and
    SCA are negligible by comparison. Good enough to decide whether a model
    is in the right order of magnitude for a given board.
    """
    total = [0]
    handles = []

    def hook(module, inputs, output):
        out_elems = output.numel()
        kernel = module.kernel_size[0] * module.kernel_size[1]
        in_ch = module.in_channels // module.groups
        total[0] += out_elems * kernel * in_ch

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(hook))

    with torch.no_grad():
        model(torch.zeros(1, 3, height, width))

    for h in handles:
        h.remove()
    return total[0] / 1e9


def build_and_load(ckpt_path, verbose=True):
    """Returns (model, arch, report)."""
    raw = load_checkpoint(ckpt_path)
    state = unwrap_state_dict(raw)
    arch = infer_nafnet_arch(state)

    model = NAFNet(
        img_channel=3,
        width=arch["width"],
        enc_blk_nums=arch["enc"],
        middle_blk_num=arch["middle"],
        dec_blk_nums=arch["dec"],
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()

    report = {
        "arch": arch,
        "params_m": count_parameters(model) / 1e6,
        "missing": len(missing),
        "unexpected": len(unexpected),
        "total_tensors": len(model.state_dict()),
    }

    if verbose:
        print("checkpoint : %s" % Path(ckpt_path).name)
        print("inferred   : width %d, enc %s, middle %d, dec %s"
              % (arch["width"], arch["enc"], arch["middle"], arch["dec"]))
        print("params     : %.2fM" % report["params_m"])
        print("loaded     : %d/%d tensors (%d missing, %d unexpected)"
              % (report["total_tensors"] - report["missing"], report["total_tensors"],
                 report["missing"], report["unexpected"]))
        if report["missing"] or report["unexpected"]:
            print("             ^ a partial load means the weights do not belong to")
            print("               this graph. Do not trust the result.")

    return model, arch, report


def feasibility(model, tile, label="tile"):
    """Print an honest verdict about running this on the target board."""
    gmacs = estimate_gmacs(model, tile, tile)
    gflops = gmacs * 2
    budget = NANO_GFLOPS_PEAK * NANO_EFFICIENCY
    seconds = gflops / budget

    print("\n%s %dx%d" % (label, tile, tile))
    print("  compute   : %.2f GMACs (%.2f GFLOPs)" % (gmacs, gflops))
    print("  Jetson Nano estimate: %.0f ms per %s at ~%.0f GFLOPS sustained"
          % (seconds * 1000, label, budget))

    # 1080p needs 112 tiles at 256px, or ~450 at 128px.
    per_frame = {128: 450, 256: 112}.get(tile)
    if per_frame:
        frame_s = seconds * per_frame
        print("  whole 1080p frame (%d tiles): %.1f s  -> %.2f fps"
              % (per_frame, frame_s, 1.0 / frame_s if frame_s else 0))
    crop_s = seconds * 2
    print("  one person crop (~2 tiles): %.0f ms, x2 cameras = %.0f ms"
          % (crop_s * 1000, crop_s * 2000))

    verdict = "usable on crops" if crop_s * 2 < 0.5 else "too slow for live use on a Nano"
    print("  verdict   : %s" % verdict)
    return {"gmacs": gmacs, "ms_per_tile": seconds * 1000}


def cmd_inspect(args):
    model, arch, report = build_and_load(args.ckpt)
    for tile in (128, 256):
        feasibility(model, tile)

    print("\nFor comparison, this project's own architectures:")
    for name, spec in sorted(config.NAFNET_ARCHS.items()):
        ref = NAFNet(width=spec["width"], enc_blk_nums=spec["enc"],
                     middle_blk_num=spec["middle"], dec_blk_nums=spec["dec"])
        print("  %-6s width %2d  %.2fM params  %.2f GMACs @128px"
              % (name, spec["width"], count_parameters(ref) / 1e6,
                 estimate_gmacs(ref, 128, 128)))

    if report["missing"] == 0 and report["unexpected"] == 0:
        print("\nExport it for the pipeline with:")
        print("  python -m training.pretrained to_onnx --ckpt %s --tile 128" % args.ckpt)
        print("\nOr fine-tune it on your own CCTV degradation (keeps its real-noise")
        print("priors, adapts to compression and low light):")
        print("  python -m training.train_denoiser --train_dir <dirs> \\")
        print("      --width %d --enc_blks %s --middle_blks %d --dec_blks %s \\"
              % (arch["width"], " ".join(map(str, arch["enc"])),
                 arch["middle"], " ".join(map(str, arch["dec"]))))
        print("      --init_from %s --lr 2e-4 --epochs 15" % args.ckpt)


def cmd_to_onnx(args):
    model, arch, report = build_and_load(args.ckpt)
    if report["missing"] or report["unexpected"]:
        raise SystemExit(
            "refusing to export a partially-loaded model; the checkpoint does "
            "not match the architecture inferred from it"
        )

    feasibility(model, args.tile)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, args.tile, args.tile)
    export_kwargs = dict(
        export_params=True, opset_version=args.opset, do_constant_folding=True,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    try:
        torch.onnx.export(model, dummy, str(out), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(out), **export_kwargs)

    print("\nwrote %s (%.1f MB)" % (out, out.stat().st_size / 1e6))

    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        test = torch.rand(1, 3, args.tile, args.tile)
        with torch.no_grad():
            expected = model(test).numpy()
        got = sess.run(None, {sess.get_inputs()[0].name: test.numpy()})[0]
        diff = float(np.abs(expected - got).max())
        print("max |torch - onnx| = %.3e  %s" % (diff, "OK" if diff < 1e-3 else "CHECK THIS"))
    except ImportError:
        print("onnxruntime not installed; skipped the parity check")

    print("\nPoint the pipeline at it:")
    print("  config.NAFNET_ONNX = MODELS_DIR / \"%s\"" % out.name)
    print("  config.TILE_SIZE   = %d" % args.tile)


def main():
    ap = argparse.ArgumentParser(description="Use a pretrained NAFNet checkpoint")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_i = sub.add_parser("inspect", help="identify the architecture and judge feasibility")
    p_i.add_argument("--ckpt", type=Path, required=True)
    p_i.set_defaults(func=cmd_inspect)

    p_o = sub.add_parser("to_onnx", help="export it for this pipeline")
    p_o.add_argument("--ckpt", type=Path, required=True)
    p_o.add_argument("--out", type=Path, default=config.MODELS_DIR / "nafnet_pretrained.onnx")
    p_o.add_argument("--tile", type=int, default=128)
    p_o.add_argument("--opset", type=int, default=12)
    p_o.set_defaults(func=cmd_to_onnx)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
