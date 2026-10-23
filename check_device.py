"""Identify the Jetson you are actually on, and say what it can run.

    python3 check_device.py

Reports the board, L4T/JetPack version, GPU compute capability, memory,
power mode, and which runtime features are available. Several things this
project might want - INT8 quantization, DLA offload - exist only on newer
Tegra parts, and failing loudly here is better than discovering it halfway
through a TensorRT build.
"""

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Compute capability 6.1 is where TensorRT gained the DP4A instruction that
# INT8 inference relies on. Maxwell (5.3) has no integer dot product, so
# TensorRT silently refuses INT8 on it.
INT8_MIN_COMPUTE = 6.1

# board substring -> (friendly name, compute capability, GPU arch, cores,
#                     fp16 GFLOPS (approx), typical power modes)
KNOWN_BOARDS = [
    ("orin nano",  ("Jetson Orin Nano",  8.7, "Ampere",  1024, 20000.0, "7W / 15W / 25W")),
    ("orin nx",    ("Jetson Orin NX",    8.7, "Ampere",  1024, 35000.0, "10W / 15W / 25W")),
    ("agx orin",   ("Jetson AGX Orin",   8.7, "Ampere",  2048, 85000.0, "15W - 60W")),
    ("xavier nx",  ("Jetson Xavier NX",  7.2, "Volta",    384,  6000.0, "10W / 15W / 20W")),
    ("agx xavier", ("Jetson AGX Xavier", 7.2, "Volta",    512, 11000.0, "10W - 30W")),
    ("tx2",        ("Jetson TX2",        6.2, "Pascal",   256,  1300.0, "7.5W / 15W")),
    ("nano",       ("Jetson Nano",       5.3, "Maxwell",  128,   472.0, "5W / 10W (MAXN)")),
]

L4T_TO_JETPACK = {
    "32.7": "JetPack 4.6.x  (final release for Jetson Nano / TX2)",
    "32.6": "JetPack 4.6",
    "32.5": "JetPack 4.5",
    "35.": "JetPack 5.x",
    "36.": "JetPack 6.x",
}


def read_text(path):
    try:
        return Path(path).read_text(errors="ignore").strip("\x00 \n")
    except Exception:
        return ""


def run(cmd):
    try:
        out = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, timeout=10,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def detect_board():
    model = read_text("/proc/device-tree/model")
    if not model:
        return None, ""
    lowered = model.lower()
    for needle, info in KNOWN_BOARDS:
        if needle in lowered:
            return info, model
    return None, model


def detect_l4t():
    raw = read_text("/etc/nv_tegra_release")
    if not raw:
        return "", ""
    m = re.search(r"R(\d+).*?REVISION:\s*([\d.]+)", raw)
    if not m:
        return raw, ""
    version = "%s.%s" % (m.group(1), m.group(2))
    jetpack = ""
    for prefix, name in L4T_TO_JETPACK.items():
        if version.startswith(prefix):
            jetpack = name
            break
    return version, jetpack


def memory_mb():
    info = read_text("/proc/meminfo")
    total = available = 0
    for line in info.splitlines():
        if line.startswith("MemTotal:"):
            total = int(line.split()[1]) // 1024
        elif line.startswith("MemAvailable:"):
            available = int(line.split()[1]) // 1024
    return total, available


def swap_mb():
    info = read_text("/proc/meminfo")
    for line in info.splitlines():
        if line.startswith("SwapTotal:"):
            return int(line.split()[1]) // 1024
    return 0


def power_mode():
    if not shutil.which("nvpmodel"):
        return ""
    out = run(["nvpmodel", "-q"])
    return " / ".join(l.strip() for l in out.splitlines() if l.strip())[:120]


def check_python():
    major, minor = sys.version_info[:2]
    notes = []
    if (major, minor) < (3, 6):
        notes.append("unsupported; this project needs 3.6+")
    elif (major, minor) == (3, 6):
        notes.append("JetPack 4.6 default. The runtime path targets 3.6, but")
        notes.append("ultralytics and gfpgan both require 3.8+ and will NOT install.")
        notes.append("Run YOLOv8 and ArcFace from ONNX instead (see README).")
    return "%d.%d.%d" % sys.version_info[:3], notes


def check_packages():
    results = []
    for name, label in [
        ("numpy", "numpy"),
        ("cv2", "OpenCV"),
        ("onnxruntime", "ONNX Runtime"),
        ("tensorrt", "TensorRT"),
        ("pycuda", "pycuda"),
        ("faiss", "FAISS"),
        ("torch", "PyTorch"),
        ("ultralytics", "ultralytics"),
        ("gfpgan", "GFPGAN"),
    ]:
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", "?")
            extra = ""
            if name == "onnxruntime":
                try:
                    extra = "  providers: %s" % ", ".join(mod.get_available_providers())
                except Exception:
                    pass
            if name == "torch":
                try:
                    extra = "  cuda: %s" % mod.cuda.is_available()
                except Exception:
                    pass
            results.append((label, version, extra, True))
        except Exception:
            results.append((label, "not installed", "", False))
    return results


def main():
    print("=" * 70)
    print("DEVICE REPORT")
    print("=" * 70)

    info, model = detect_board()
    is_jetson = bool(model)

    if not is_jetson:
        print("board          : not a Jetson (%s %s)" % (platform.system(), platform.machine()))
        print("                 This report is most useful run on the device itself.")
    else:
        print("board          : %s" % model)

    compute = None
    if info:
        name, compute, arch, cores, gflops, power = info
        print("identified as  : %s" % name)
        print("GPU            : %s, %d CUDA cores, compute capability %.1f" % (arch, cores, compute))
        print("FP16 throughput: ~%.0f GFLOPS theoretical" % gflops)
        print("power modes    : %s" % power)
    elif is_jetson:
        print("identified as  : unrecognized Tegra board")

    l4t, jetpack = detect_l4t()
    if l4t:
        print("L4T            : %s%s" % (l4t, ("   (%s)" % jetpack) if jetpack else ""))

    if is_jetson:
        total, avail = memory_mb()
        if total:
            print("memory         : %d MB total, %d MB available (shared CPU+GPU)" % (total, avail))
        sw = swap_mb()
        print("swap           : %d MB%s" % (sw, "" if sw else "   <- consider adding some"))
        mode = power_mode()
        if mode:
            print("power mode     : %s" % mode)

    version, notes = check_python()
    print("python         : %s" % version)
    for note in notes:
        print("                 %s" % note)

    print("\n" + "-" * 70)
    print("PACKAGES")
    print("-" * 70)
    for label, ver, extra, ok in check_packages():
        print("  %-14s %-16s%s" % (label, ver, extra))

    print("\n" + "-" * 70)
    print("CAPABILITIES FOR THIS PIPELINE")
    print("-" * 70)

    if compute is None:
        print("  Board not identified - cannot judge INT8 / DLA support.")
    else:
        fp16 = compute >= 5.3
        int8 = compute >= INT8_MIN_COMPUTE
        dla = compute >= 7.2
        print("  FP16 inference    : %s" % ("YES" if fp16 else "no"))
        print("  INT8 quantization : %s" % (
            "YES" if int8 else
            "NO - needs compute %.1f+, this GPU is %.1f (no DP4A instruction)."
            % (INT8_MIN_COMPUTE, compute)))
        print("  DLA offload       : %s" % ("YES" if dla else "no - Xavier/Orin only"))

        if not int8:
            print("\n  => Build TensorRT engines with --precision fp16.")
            print("     An INT8 build will either fail or silently fall back.")

    if info and info[1] < 6.0:
        total, _ = memory_mb()
        print("\n  This is a first-generation Jetson. Realistic expectations:")
        print("   - Whole-frame NAFNet at 1080p is not viable. Use")
        print("     --denoise_scope crop and a 'nano' width-16 model.")
        print("   - GFPGAN (StyleGAN2) costs seconds per face here. Use")
        print("     --restore_mode escalate so it runs only on candidates.")
        if total and total <= 4500:
            print("   - 4 GB is shared between CPU and GPU. All four models")
            print("     resident at once will not fit; keep GFPGAN out of the")
            print("     hot path and run headless to free the desktop's ~500 MB.")

    print("\n" + "=" * 70)
    print("Suggested settings for this device:")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import config

        profile = config.detect_profile()
        for key, value in sorted(config.PROFILES[profile].items()):
            print("  %-20s %s" % (key, value))
        print("\n  export EDGE_PROFILE=%s" % profile)
    except Exception as exc:
        print("  (could not load config profiles: %s)" % exc)
    print("=" * 70)


if __name__ == "__main__":
    main()
