"""Export YOLOv8 for the edge node.

**For a Jetson Nano, export ONNX on your workstation and copy it over.**
JetPack 4.6 pins the Nano to Python 3.6 and ultralytics requires 3.8+, so
it cannot be installed on the device - which also means the device cannot
build its own TensorRT engine through ultralytics. The ONNX graph runs under
ONNX Runtime (or TensorRT's own ONNX parser) with no ultralytics dependency;
``pipeline/stage2_detect.py`` implements the letterbox and NMS itself.

    # On the workstation (Jetson Nano target)
    python export/export_yolo.py --format onnx --imgsz 416

    # On a Xavier/Orin running JetPack 5+ (ultralytics available on-device)
    python export/export_yolo.py --format engine --precision fp16

A TensorRT engine is tied to the exact GPU, TensorRT version and driver that
built it, so an engine never travels between machines. ONNX does.
"""

import shutil
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="YOLOv8 -> ONNX or TensorRT")
    ap.add_argument("--weights", default="yolov8n.pt", help=".pt checkpoint (downloaded if absent)")
    ap.add_argument("--format", default="onnx", choices=["onnx", "engine"],
                    help="onnx travels between machines; engine does not")
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"],
                    help="engine format only")
    ap.add_argument("--imgsz", type=int, default=config.YOLO_IMGSZ,
                    help="416 is a good speed/accuracy point on a Nano; 640 elsewhere")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--opset", type=int, default=12,
                    help="12 keeps JetPack 4.6's ONNX Runtime 1.11 happy")
    ap.add_argument("--simplify", action="store_true", default=True)
    ap.add_argument("--no_simplify", dest="simplify", action="store_false")
    ap.add_argument("--workspace", type=int, default=2, help="GiB, engine format only")
    ap.add_argument("--data", default="coco128.yaml", help="INT8 calibration set")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    out = args.out or config.MODELS_DIR / ("yolov8n.onnx" if args.format == "onnx"
                                           else "yolov8n.engine")

    model = YOLO(args.weights)
    print("exporting %s -> %s (imgsz=%d)" % (args.weights, args.format, args.imgsz))

    kwargs = dict(format=args.format, imgsz=args.imgsz, batch=args.batch)

    if args.format == "onnx":
        kwargs.update(opset=args.opset, simplify=args.simplify, dynamic=False)
        # Keep the export on CPU so the graph carries no device-specific ops.
        kwargs["device"] = "cpu"
    else:
        if args.precision == "int8" and not config.INT8_AVAILABLE:
            raise SystemExit(
                "INT8 is unavailable on the '%s' profile (Maxwell has no DP4A "
                "instruction). Use --precision fp16." % config.PROFILE
            )
        kwargs.update(
            workspace=args.workspace,
            device=0,
            half=args.precision == "fp16",
            int8=args.precision == "int8",
        )
        if args.precision == "int8":
            kwargs["data"] = args.data

    exported = Path(model.export(**kwargs))
    print("ultralytics wrote %s" % exported)

    out.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != out.resolve():
        shutil.copy2(exported, out)
        print("copied to %s" % out)

    print("\nsize: %.1f MB" % (out.stat().st_size / 1e6))
    if args.format == "onnx":
        print("\nCopy this file to the Jetson's models/ folder. Set:")
        print("  YOLO_WEIGHTS = MODELS_DIR / \"%s\"   (already the default)" % out.name)
        print("  YOLO_IMGSZ   = %d" % args.imgsz)
        print("No ultralytics install is needed on the device.")


if __name__ == "__main__":
    main()
