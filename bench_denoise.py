"""Benchmark classical crop restoration: quality gained per millisecond.

    python bench_denoise.py                 # the shipped profiles
    python bench_denoise.py --sweep         # candidate filters, ranked
    python bench_denoise.py --crop 128x256

Run it on the Jetson itself for the numbers that matter. On any other
machine it prints a scaled estimate, clearly labelled as such, because the
Nano's 4-core Cortex-A57 is roughly 6x slower per core than a current
laptop CPU and stock JetPack OpenCV has no CUDA - these filters are CPU
work there.

Every default in pipeline/denoise_classical.py came out of this script.
"""

import argparse
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from pipeline.denoise_classical import PROFILES, ClassicalDenoiser  # noqa: E402

NANO_SCALE = 6.0


def is_jetson():
    try:
        with open("/proc/device-tree/model", "rb") as fh:
            return b"jetson" in fh.read().lower()
    except Exception:
        return False


def make_crop(w, h, seed):
    """A person-shaped crop carrying texture at several scales.

    A flat fill would make every filter look good; the fine lines are what
    separates a filter that preserves detail from one that smears it.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (58, 64, 72)
    cv2.ellipse(img, (w // 2, h // 5), (w // 4, h // 7), 0, 0, 360, (150, 142, 135), -1)
    cv2.rectangle(img, (w // 4, h // 3), (3 * w // 4, h), (92, 78, 70), -1)
    for _ in range(14):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(img, p1, p2, tuple(int(c) for c in rng.integers(40, 190, 3)), 1)
    return img


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)


def degrade(clean, rng):
    """Approximate what a CCTV sensor and codec do to a frame.

    Optical softness, then sensor noise amplified by gain, then compression.
    The order matters: JPEG runs last, as it does in a real camera, which is
    why its block artifacts are the dominant structure left for stage 1 to
    remove.
    """
    x = cv2.GaussianBlur(clean, (0, 0), rng.uniform(0.4, 1.2)).astype(np.float32)

    if rng.random() < 0.4:                       # low light: darken, gain back up
        gamma = rng.uniform(1.3, 2.2)
        x = np.clip(x / 255.0, 0, 1) ** gamma * 255.0
        x += rng.normal(0, rng.uniform(6, 16), x.shape)
        x = np.clip(np.clip(x / 255.0, 0, 1) ** (1.0 / gamma) * 255.0, 0, 255)
    else:
        x += rng.normal(0, rng.uniform(6, 22), x.shape)

    x = np.clip(x, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", x, [int(cv2.IMWRITE_JPEG_QUALITY),
                                       int(rng.integers(35, 80))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else x


def build_pairs(w, h, count):
    """(clean, degraded) pairs."""
    rng = np.random.default_rng(0)
    return [(c, degrade(c, rng)) for c in (make_crop(w, h, s) for s in range(count))]


def time_fn(fn, sample, max_seconds=1.0):
    fn(sample)
    iters, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < max_seconds and iters < 200:
        fn(sample)
        iters += 1
    return (time.perf_counter() - t0) * 1000 / max(iters, 1)


def report(rows, baseline, native, scale, subjects):
    label = "ms" if native else "ms (est. Nano)"
    print("%-26s %8s %8s %10s %12s" % ("", "PSNR", "gain", "ms/crop", label))
    print("-" * 70)
    for name, gain, ms in rows:
        print("%-26s %8.2f %+8.2f %10.2f %12.1f"
              % (name, baseline + gain, gain, ms, ms * scale))

    print("\nPer inference cycle (%d crops: 2 cameras x %d subjects)"
          % (subjects * 2, subjects))
    print("%-26s %12s %14s" % ("", "ms/cycle", "max pairs/s"))
    print("-" * 56)
    for name, gain, ms in rows:
        cycle = ms * scale * subjects * 2
        print("%-26s %12.1f %14s"
              % (name, cycle, "%.1f" % (1000.0 / cycle) if cycle > 0.01 else "-"))


def main():
    ap = argparse.ArgumentParser(description="Classical denoise benchmark")
    ap.add_argument("--crop", default="96x192", help="WxH of the test crop")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--subjects", type=int, default=2, help="subjects per camera")
    ap.add_argument("--sweep", action="store_true",
                    help="rank candidate filters instead of the shipped profiles")
    args = ap.parse_args()

    w, h = (int(v) for v in args.crop.lower().split("x"))
    native = is_jetson()
    scale = 1.0 if native else NANO_SCALE

    print("=" * 70)
    print("platform : %s" % platform.platform())
    print("crop     : %dx%d, %d samples" % (w, h, args.samples))
    print("timing   : %s" % ("measured on this Jetson" if native else
                             "measured here, x%.0f for a Nano estimate" % NANO_SCALE))
    print("=" * 70)

    pairs = build_pairs(w, h, args.samples)
    baseline = float(np.mean([psnr(n, c) for c, n in pairs]))
    print("degraded input: %.2f dB\n" % baseline)

    if args.sweep:
        candidates = {
            "median3": lambda x: cv2.medianBlur(x, 3),
            "median5": lambda x: cv2.medianBlur(x, 5),
            "bilateral d5 sc75": lambda x: cv2.bilateralFilter(x, 5, 75, 5),
            "bilateral d7 sc100": lambda x: cv2.bilateralFilter(x, 7, 100, 7),
            "bilateral d9 sc150": lambda x: cv2.bilateralFilter(x, 9, 150, 9),
            "median3+bil d7 sc75": lambda x: cv2.bilateralFilter(cv2.medianBlur(x, 3), 7, 75, 7),
            "median3+bil d9 sc150": lambda x: cv2.bilateralFilter(cv2.medianBlur(x, 3), 9, 150, 9),
            "edgePreserving": lambda x: cv2.edgePreservingFilter(
                x, flags=cv2.RECURS_FILTER, sigma_s=30, sigma_r=0.3),
            "NLM h18 t5 s11": lambda x: cv2.fastNlMeansDenoisingColored(x, None, 18, 18, 5, 11),
        }
        rows = []
        for name, fn in candidates.items():
            gain = float(np.mean([psnr(fn(n), c) for c, n in pairs])) - baseline
            rows.append((name, gain, time_fn(fn, pairs[0][1])))
        report(rows, baseline, native, scale, args.subjects)

        print("\nRanked by gain per millisecond on the target board:")
        print("%-26s %10s %12s" % ("", "gain", "dB per ms"))
        print("-" * 52)
        for name, gain, ms in sorted(rows, key=lambda r: -(r[1] / max(r[2] * scale, 1e-9))):
            print("%-26s %+10.2f %12.3f" % (name, gain, gain / max(ms * scale, 1e-9)))
        return

    rows = []
    for name in ["off", "fast", "balanced", "quality"]:
        den = ClassicalDenoiser(profile=name)
        gain = float(np.mean([psnr(den.denoise(n), c) for c, n in pairs])) - baseline
        rows.append((den.describe(), gain, time_fn(den.denoise, pairs[0][1])))

    # The two stages the measurements rejected, shown so the choice is visible.
    for label, kwargs in [("balanced +clahe", {"clahe": True}),
                          ("balanced +clahe +sharpen", {"clahe": True, "sharpen": 0.6})]:
        den = ClassicalDenoiser(profile="balanced", **kwargs)
        gain = float(np.mean([psnr(den.denoise(n), c) for c, n in pairs])) - baseline
        rows.append((label, gain, time_fn(den.denoise, pairs[0][1])))

    report(rows, baseline, native, scale, args.subjects)
    print("\nCLAHE and sharpening are off by default: both reduce fidelity "
          "against\nthe clean reference. They are exposed because illumination "
          "normalisation\nis standard for face embedding - verify with real "
          "match rates before\nenabling, not with PSNR.")


if __name__ == "__main__":
    main()
