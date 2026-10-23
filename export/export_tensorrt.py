"""Build a TensorRT engine from the exported ONNX graph. Run this ON the Jetson.

    python export/export_tensorrt.py --onnx models/nafnet.onnx --precision fp16
    python export/export_tensorrt.py --onnx models/nafnet.onnx --precision int8 \
        --calib_dir data/calib

FP16 is the safe default on Orin: roughly 2x the FP32 throughput at a PSNR
cost in the hundredths of a dB. INT8 roughly doubles it again but needs a
calibration set - a few hundred *representative* tiles from your own cameras,
not random internet images.

A TensorRT engine is tied to the exact GPU, TensorRT version and driver it was
built on. Build it on the device you deploy to; never copy an engine between
machines.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def cuda_compute_capability():
    """(major, minor) of device 0, or None if it cannot be determined."""
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda

        return cuda.Device(0).compute_capability()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(0)
    except Exception:
        pass
    return None


def _refuse_int8_on_maxwell():
    """INT8 inference needs the DP4A integer dot-product instruction, which
    arrived with compute capability 6.1. The original Jetson Nano is Maxwell
    (5.3), so TensorRT cannot quantize there however the flags are set - it
    either errors out or silently runs everything in FP16 while you believe
    you are measuring INT8."""
    capability = cuda_compute_capability()
    if capability is None:
        print("warning: could not read the compute capability; if this is a "
              "Jetson Nano (Maxwell 5.3), INT8 is not supported - use fp16.")
        return
    value = capability[0] + capability[1] / 10.0
    if value < config.INT8_MIN_COMPUTE:
        raise SystemExit(
            "INT8 is not available on this GPU.\n"
            "  compute capability : %.1f (needs %.1f+ for DP4A)\n"
            "  detected profile   : %s\n"
            "The original Jetson Nano is Maxwell and has no integer dot-product\n"
            "instruction. Rebuild with --precision fp16; on this board FP16 is\n"
            "the fastest path there is, not a fallback."
            % (value, config.INT8_MIN_COMPUTE, config.PROFILE)
        )


def load_calibration_tiles(calib_dir: Path, tile: int, limit: int) -> np.ndarray:
    """Crops tiles from the calibration images, preprocessed exactly as
    stage 1 does at runtime: RGB, CHW, float32 [0,1]."""
    import cv2

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    paths = sorted(p for p in Path(calib_dir).rglob("*") if p.suffix.lower() in exts)
    if not paths:
        raise SystemExit(f"no calibration images under {calib_dir}")

    rng = np.random.default_rng(0)
    tiles: List[np.ndarray] = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        if h < tile or w < tile:
            img = cv2.resize(img, (max(tile, w), max(tile, h)))
            h, w = img.shape[:2]
        for _ in range(2):
            y, x = rng.integers(0, h - tile + 1), rng.integers(0, w - tile + 1)
            patch = img[y : y + tile, x : x + tile]
            rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tiles.append(rgb.transpose(2, 0, 1))
            if len(tiles) >= limit:
                break
        if len(tiles) >= limit:
            break

    print(f"calibration: {len(tiles)} tiles of {tile}x{tile} from {len(paths)} images")
    return np.ascontiguousarray(np.stack(tiles), dtype=np.float32)


def make_calibrator(trt, tiles: np.ndarray, batch_size: int, cache_path: Path):
    """Entropy calibrator feeding the tiles to TensorRT in batches."""
    import pycuda.driver as cuda

    class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.tiles = tiles
            self.batch_size = batch_size
            self.index = 0
            self.cache_path = cache_path
            self.device_input = cuda.mem_alloc(
                batch_size * tiles.shape[1] * tiles.shape[2] * tiles.shape[3] * 4
            )

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            if self.index + self.batch_size > len(self.tiles):
                return None
            batch = np.ascontiguousarray(self.tiles[self.index : self.index + self.batch_size])
            cuda.memcpy_htod(self.device_input, batch)
            self.index += self.batch_size
            return [int(self.device_input)]

        def read_calibration_cache(self):
            if self.cache_path.exists():
                return self.cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache):
            self.cache_path.write_bytes(cache)

    return EntropyCalibrator()


def build_engine(args) -> Path:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO if args.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = Path(args.onnx).read_bytes()
    if not parser.parse(onnx_bytes):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit("ONNX parse failed")

    cfg = builder.create_config()
    try:
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace << 30)
    except AttributeError:                         # TensorRT < 8.4
        cfg.max_workspace_size = args.workspace << 30

    input_tensor = network.get_input(0)
    print(f"input '{input_tensor.name}' shape {input_tensor.shape}")

    # Optimization profile for the dynamic batch dimension.
    if any(d == -1 for d in input_tensor.shape):
        profile = builder.create_optimization_profile()
        c, h, w = input_tensor.shape[1:]
        h = args.tile if h == -1 else h
        w = args.tile if w == -1 else w
        profile.set_shape(
            input_tensor.name,
            (1, c, h, w),
            (args.opt_batch, c, h, w),
            (args.max_batch, c, h, w),
        )
        cfg.add_optimization_profile(profile)
        print(f"profile: min 1 / opt {args.opt_batch} / max {args.max_batch} @ {h}x{w}")

    if args.precision in ("fp16", "int8"):
        if not builder.platform_has_fast_fp16:
            print("warning: no fast FP16 on this platform")
        cfg.set_flag(trt.BuilderFlag.FP16)

    calibrator = None
    if args.precision == "int8":
        _refuse_int8_on_maxwell()
        if not builder.platform_has_fast_int8:
            raise SystemExit(
                "this platform has no fast INT8 support; rebuild with "
                "--precision fp16"
            )
        if not args.calib_dir:
            raise SystemExit("--calib_dir is required for INT8")
        cfg.set_flag(trt.BuilderFlag.INT8)
        tiles = load_calibration_tiles(Path(args.calib_dir), args.tile, args.calib_images)
        calibrator = make_calibrator(
            trt, tiles, args.calib_batch, Path(args.out).with_suffix(".calib")
        )
        cfg.int8_calibrator = calibrator

    print(f"building {args.precision.upper()} engine (this takes several minutes)...")
    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, cfg)
        if serialized is None:
            raise SystemExit("engine build failed")
        Path(args.out).write_bytes(serialized)
    else:
        engine = builder.build_engine(network, cfg)
        if engine is None:
            raise SystemExit("engine build failed")
        Path(args.out).write_bytes(engine.serialize())

    del calibrator
    return Path(args.out)


def build_with_trtexec(args) -> Path:
    """Fallback path: shell out to trtexec, which ships with every JetPack."""
    import subprocess

    cmd = [
        "trtexec",
        f"--onnx={args.onnx}",
        f"--saveEngine={args.out}",
        f"--workspace={args.workspace * 1024}",
    ]
    if args.precision == "fp16":
        cmd.append("--fp16")
    elif args.precision == "int8":
        _refuse_int8_on_maxwell()
        cmd += ["--int8", "--fp16"]
        print("note: trtexec INT8 without a calibrator uses dummy dynamic ranges; "
              "prefer the TensorRT Python path for a real calibration.")
    cmd += [
        f"--minShapes=input:1x3x{args.tile}x{args.tile}",
        f"--optShapes=input:{args.opt_batch}x3x{args.tile}x{args.tile}",
        f"--maxShapes=input:{args.max_batch}x3x{args.tile}x{args.tile}",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return Path(args.out)


def benchmark_engine(engine_path: Path, tile: int, batch: int, iters: int = 50) -> None:
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        import tensorrt as trt
    except ImportError:
        print("pycuda/tensorrt unavailable; skipping the benchmark")
        return

    import time

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    ctx = engine.create_execution_context()

    shape = (batch, 3, tile, tile)
    ctx.set_input_shape(engine.get_tensor_name(0), shape)

    host_in = np.random.rand(*shape).astype(np.float32)
    d_in = cuda.mem_alloc(host_in.nbytes)
    host_out = np.empty(shape, dtype=np.float32)
    d_out = cuda.mem_alloc(host_out.nbytes)
    cuda.memcpy_htod(d_in, host_in)

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    ctx.set_tensor_address(names[0], int(d_in))
    ctx.set_tensor_address(names[1], int(d_out))
    stream = cuda.Stream()

    for _ in range(10):
        ctx.execute_async_v3(stream.handle)
    stream.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        ctx.execute_async_v3(stream.handle)
    stream.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / iters

    print(f"engine latency: {ms:.2f} ms / batch of {batch} "
          f"({ms / batch:.2f} ms per {tile}x{tile} tile)")


def main():
    ap = argparse.ArgumentParser(description="ONNX -> TensorRT engine")
    ap.add_argument("--onnx", type=Path, default=config.NAFNET_ONNX)
    ap.add_argument("--out", type=Path, default=config.NAFNET_ENGINE)
    ap.add_argument("--precision", default=config.TRT_PRECISION,
                    choices=["fp32", "fp16", "int8"],
                    help="defaults to the hardware profile (fp16 on a Jetson Nano)")
    ap.add_argument("--tile", type=int, default=config.TILE_SIZE)
    ap.add_argument("--opt_batch", type=int, default=1)
    ap.add_argument("--max_batch", type=int, default=4)
    # 4 GB shared memory on a Nano; a 2 GiB workspace will fail to allocate.
    ap.add_argument("--workspace", type=int, default=1 if config.PROFILE == "nano" else 2,
                    help="GiB")
    ap.add_argument("--calib_dir", type=Path, default=None, help="INT8 calibration images")
    ap.add_argument("--calib_images", type=int, default=512)
    ap.add_argument("--calib_batch", type=int, default=8)
    ap.add_argument("--trtexec", action="store_true", help="use the trtexec CLI instead")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not Path(args.onnx).exists():
        raise SystemExit(f"ONNX model not found: {args.onnx}. Run export/export_onnx.py first.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    engine_path = build_with_trtexec(args) if args.trtexec else build_engine(args)

    print(f"wrote {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")
    if args.benchmark:
        benchmark_engine(engine_path, args.tile, args.opt_batch)

    print(
        "\nTo use it, run the pipeline with the TensorRT execution provider:\n"
        "  $env:EDGE_PROVIDER='tensorrt'   # PowerShell\n"
        "  export EDGE_PROVIDER=tensorrt   # Jetson\n"
        "ONNX Runtime will pick up the engine cache under models/trt_cache."
    )


if __name__ == "__main__":
    main()
