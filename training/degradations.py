"""Synthetic CCTV degradation pipeline.

Real surveillance footage is not corrupted by clean additive Gaussian noise.
It is corrupted by a chain: optical blur, sensor downsampling, shot noise
amplified by a high ISO in poor light, and then H.264/JPEG compression. A
denoiser trained only on AWGN falls apart on real cameras, so this module
reproduces the whole chain.

Use it either to generate a paired dataset on disk
(``make_dataset.py``) or on the fly inside the Dataset.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class DegradationConfig:
    """Ranges are sampled uniformly per image. Set a probability to 0 to
    switch that stage off."""

    # Optical / motion blur
    blur_prob: float = 0.6
    gaussian_sigma: Tuple[float, float] = (0.2, 1.6)
    motion_blur_prob: float = 0.3        # share of blurs that are motion blur
    motion_kernel: Tuple[int, int] = (5, 15)

    # Sensor resolution loss (down then back up)
    resize_prob: float = 0.5
    resize_range: Tuple[float, float] = (0.5, 1.0)

    # Read + shot noise
    gaussian_noise_prob: float = 0.9
    noise_sigma: Tuple[float, float] = (2.0, 30.0)     # on a 0-255 scale
    gray_noise_prob: float = 0.3                       # luminance-only noise
    poisson_prob: float = 0.4
    poisson_scale: Tuple[float, float] = (0.5, 3.0)

    # Low-light: darken, add noise at that level, then lift back up
    lowlight_prob: float = 0.35
    gamma_range: Tuple[float, float] = (1.3, 2.6)

    # Codec artifacts
    jpeg_prob: float = 0.75
    jpeg_quality: Tuple[int, int] = (35, 92)

    # Fixed-pattern sensor banding
    banding_prob: float = 0.15
    banding_strength: Tuple[float, float] = (1.0, 5.0)

    seed: Optional[int] = None
    _rng: random.Random = field(default=None, repr=False, compare=False)

    def rng(self) -> random.Random:
        if self._rng is None:
            self._rng = random.Random(self.seed)
        return self._rng


# --- individual operators --------------------------------------------------
def _apply_gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    ksize = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def _apply_motion_blur(img: np.ndarray, ksize: int, angle: float) -> np.ndarray:
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (ksize, ksize))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def _apply_resize(img: np.ndarray, scale: float, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    down = cv2.resize(
        img, (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC]),
    )
    return cv2.resize(down, (w, h), interpolation=rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))


def _add_gaussian_noise(img: np.ndarray, sigma: float, gray: bool, rng: random.Random) -> np.ndarray:
    np_rng = np.random.default_rng(rng.randrange(2**31))
    if gray:
        noise = np_rng.normal(0, sigma / 255.0, img.shape[:2]).astype(np.float32)[..., None]
    else:
        noise = np_rng.normal(0, sigma / 255.0, img.shape).astype(np.float32)
    return img + noise


def _add_poisson_noise(img: np.ndarray, scale: float, rng: random.Random) -> np.ndarray:
    """Shot noise: variance proportional to signal, as on a real sensor."""
    np_rng = np.random.default_rng(rng.randrange(2**31))
    clipped = np.clip(img, 0, 1)
    vals = 2 ** np_rng.uniform(4, 8)          # effective bit depth
    noisy = np_rng.poisson(clipped * vals) / vals
    return clipped + (noisy - clipped).astype(np.float32) * scale


def _apply_lowlight(img: np.ndarray, gamma: float, sigma: float, rng: random.Random) -> np.ndarray:
    dark = np.clip(img, 0, 1) ** gamma
    dark = _add_gaussian_noise(dark, sigma, gray=False, rng=rng)
    return np.clip(dark, 0, 1) ** (1.0 / gamma)   # gain brings the noise up with it


def _apply_jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", u8, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img
    return cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32) / 255.0


def _apply_banding(img: np.ndarray, strength: float, rng: random.Random) -> np.ndarray:
    np_rng = np.random.default_rng(rng.randrange(2**31))
    h = img.shape[0]
    rows = np_rng.normal(0, strength / 255.0, (h, 1, 1)).astype(np.float32)
    return img + rows


# --- full chain ------------------------------------------------------------
def degrade(img_bgr: np.ndarray, cfg: Optional[DegradationConfig] = None) -> np.ndarray:
    """Clean BGR uint8 -> synthetically degraded BGR uint8 of the same size."""
    cfg = cfg or DegradationConfig()
    rng = cfg.rng()
    img = img_bgr.astype(np.float32) / 255.0

    if rng.random() < cfg.blur_prob:
        if rng.random() < cfg.motion_blur_prob:
            k = rng.randrange(cfg.motion_kernel[0], cfg.motion_kernel[1] + 1, 2)
            img = _apply_motion_blur(img, k, rng.uniform(0, 180))
        else:
            img = _apply_gaussian_blur(img, rng.uniform(*cfg.gaussian_sigma))

    if rng.random() < cfg.resize_prob:
        img = _apply_resize(img, rng.uniform(*cfg.resize_range), rng)

    if rng.random() < cfg.lowlight_prob:
        img = _apply_lowlight(
            img, rng.uniform(*cfg.gamma_range), rng.uniform(*cfg.noise_sigma) * 0.5, rng
        )

    if rng.random() < cfg.gaussian_noise_prob:
        img = _add_gaussian_noise(
            img,
            rng.uniform(*cfg.noise_sigma),
            gray=rng.random() < cfg.gray_noise_prob,
            rng=rng,
        )

    if rng.random() < cfg.poisson_prob:
        img = _add_poisson_noise(img, rng.uniform(*cfg.poisson_scale), rng)

    if rng.random() < cfg.banding_prob:
        img = _apply_banding(img, rng.uniform(*cfg.banding_strength), rng)

    img = np.clip(img, 0, 1)

    if rng.random() < cfg.jpeg_prob:
        img = _apply_jpeg(img, rng.randint(*cfg.jpeg_quality))

    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


PRESETS = {
    "light": DegradationConfig(
        noise_sigma=(1.0, 12.0), jpeg_quality=(70, 95), lowlight_prob=0.15, blur_prob=0.3
    ),
    "cctv": DegradationConfig(),                    # the default chain above
    "harsh": DegradationConfig(
        noise_sigma=(10.0, 50.0), jpeg_quality=(20, 70),
        lowlight_prob=0.6, blur_prob=0.8, resize_prob=0.8, resize_range=(0.33, 0.85),
    ),
}


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Preview the degradation chain on one image")
    ap.add_argument("image", type=Path)
    ap.add_argument("--preset", default="cctv", choices=list(PRESETS))
    ap.add_argument("--n", type=int, default=4, help="variants to generate")
    ap.add_argument("--out", type=Path, default=Path("degradation_preview.png"))
    args = ap.parse_args()

    clean = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if clean is None:
        raise SystemExit(f"could not read {args.image}")

    variants = [degrade(clean, PRESETS[args.preset]) for _ in range(args.n)]
    cv2.imwrite(str(args.out), np.hstack([clean] + variants))
    print(f"wrote {args.out} (clean + {args.n} degraded variants)")
