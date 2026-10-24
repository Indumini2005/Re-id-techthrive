"""Stage 1 - crop restoration. Classical only: no network, no weights.

The pipeline restores *person crops*, not whole frames. That single decision
changes what the right tool is. A crop is a few tens of thousands of pixels
rather than two million, and the job is not to produce a pretty image but to
hand ArcFace something it can embed reliably.

A 3x3 median runs first, then a bilateral filter. The order is not
arbitrary: the degradation ends in JPEG, and the median removes the
impulse-like block artifacts that would otherwise dominate the bilateral
filter's range weights and be preserved as though they were edges.

Measured on 96x192 crops (bench_denoise.py reproduces all of it; Nano
figures scale the laptop timing by 6x and should be re-measured on the
board itself):

    median3 alone              +2.5 dB for ~0.8 ms on a Nano
    median3 + bilateral d7     +3.2 dB for ~7.8 ms    <- default
    median3 + bilateral d9     +3.2 dB for ~12.1 ms
    non-local means, best cfg  +2.5 dB for ~126 ms    <- rejected
    a neural denoiser          --      for ~57 ms per 128px tile

The exact gains move with how harsh the degradation is; what holds across
settings is the ordering and the cost ratio.

Filter strength follows a measured noise estimate rather than a constant,
because a crop from a bright doorway and one from a dark corridor need
different treatment and the pipeline cannot know in advance which it has.

Everything here is core OpenCV. Nothing requires contrib modules, CUDA, a
model file or PyTorch - which is what makes it viable on JetPack 4.6, where
the stock OpenCV build has neither CUDA nor ximgproc.
"""

import math
import threading

import cv2
import numpy as np

# Immerkaer's kernel: a 3x3 Laplacian-of-Laplacian whose response to smooth
# image content is near zero, so what survives is dominated by noise.
_NOISE_KERNEL = np.array([[1, -2, 1],
                          [-2, 4, -2],
                          [1, -2, 1]], dtype=np.float32)


def estimate_noise_sigma(image_bgr):
    """Estimate Gaussian noise sigma on a 0-255 scale.

    Immerkaer's method: filter with a kernel that annihilates locally linear
    content, then scale the mean absolute response. One convolution, no
    parameters, and it degrades gracefully on textured images - which is why
    it is preferable to guessing a fixed strength.
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    h, w = gray.shape[:2]
    if h < 5 or w < 5:
        return 0.0
    response = cv2.filter2D(gray.astype(np.float32), -1, _NOISE_KERNEL)
    sigma = float(np.abs(response).mean()) * math.sqrt(math.pi / 2.0) / 6.0
    return max(0.0, min(sigma, 80.0))


def normalize_illumination(image_bgr, clip_limit=2.0, grid=8):
    """CLAHE on the L channel of LAB.

    Applied to luminance only, so colour is untouched. Global histogram
    equalisation would wreck a crop that is half in shadow; the tiled,
    clip-limited version handles exactly that case, which is the normal case
    under overhead CCTV lighting.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    lab = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def unsharp_mask(image_bgr, amount=0.6, sigma=1.0, threshold=0):
    """Restore perceived detail lost to smoothing.

    ``threshold`` leaves low-contrast areas alone, so flat regions - where
    any residual noise lives - are not re-amplified.
    """
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigma)
    sharp = cv2.addWeighted(image_bgr, 1.0 + amount, blurred, -amount, 0)
    if threshold > 0:
        low_contrast = np.abs(image_bgr.astype(np.int16) - blurred.astype(np.int16)) < threshold
        np.copyto(sharp, image_bgr, where=low_contrast)
    return sharp


def gamma_correct(image_bgr, gamma):
    """Lift shadows on a dark crop. Table lookup, so cost is independent of
    the gamma value."""
    if abs(gamma - 1.0) < 1e-3:
        return image_bgr
    inv = 1.0 / max(gamma, 1e-3)
    table = np.clip(((np.arange(256) / 255.0) ** inv) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(image_bgr, table)


# --- profiles --------------------------------------------------------------
# Chosen by measurement, not preference. `bench_denoise.py --sweep`
# reproduces the comparison these came out of; a 3x3 median doing most of
# the work is not an accident, for the reason given in the module docstring.
PROFILES = {
    # Passthrough, to measure what the rest is actually buying.
    "off": {
        "median": 0, "bilateral": None,
        "clahe": False, "sharpen": 0.0, "auto_gamma": False,
    },
    # +2.4 dB for 0.2 ms/crop on a Nano. When the budget is genuinely tight
    # this is close to free and still recovers most of the available gain.
    "fast": {
        "median": 3, "bilateral": None,
        "clahe": False, "sharpen": 0.0, "auto_gamma": True,
    },
    # Default. +3.6 dB for ~7 ms/crop; ~28 ms per cycle for two cameras and
    # two subjects, which leaves the frame budget to detection.
    "balanced": {
        "median": 3, "bilateral": {"d": 7, "sigma_color": 75, "sigma_space": 7},
        "clahe": False, "sharpen": 0.0, "auto_gamma": True,
    },
    # +4.0 dB for ~11 ms/crop. Worth it only if detection is not the
    # bottleneck; measure before choosing it.
    "quality": {
        "median": 3, "bilateral": {"d": 9, "sigma_color": 150, "sigma_space": 9},
        "clahe": False, "sharpen": 0.0, "auto_gamma": True,
    },
}

# CLAHE and unsharp masking are available but OFF in every profile, and that
# is a deliberate reversal of the obvious design.
#
# Measured against the clean reference, CLAHE costs 7.5 dB raw and still
# 2.2 dB after matching first-order statistics - so it is not merely the
# intended tone change, it genuinely degrades structural fidelity. Unsharp
# masking costs a further 1.0 dB by re-amplifying residual noise.
#
# They remain because illumination normalisation is standard practice for
# face embedding under uneven lighting, and enrolment runs through the same
# pipeline, so gallery and query stay consistent. That argument is plausible
# but unverified here: confirming it needs match rates from a real ArcFace
# model against real faces, which has not been done. Until someone measures
# that, the default is the setting that demonstrably preserves the image.
EXTRA_DEFAULTS = {"clip_limit": 2.5, "sharpen_sigma": 1.0, "sharpen_threshold": 4}


class ClassicalDenoiser:
    """Restores a person crop for embedding. Same interface as the neural
    ``Denoiser``, so the pipeline stages do not care which is in use."""

    def __init__(self, profile="balanced", adaptive=True, max_side=0,
                 clahe=None, sharpen=None):
        if profile not in PROFILES:
            raise ValueError("unknown profile %r (%s)"
                             % (profile, " | ".join(sorted(PROFILES))))
        self.profile_name = profile
        self.profile = dict(PROFILES[profile])
        # Explicit overrides for the two stages the measurements rejected,
        # so they can be switched on for an A/B against real match rates.
        if clahe is not None:
            self.profile["clahe"] = bool(clahe)
        if sharpen is not None:
            self.profile["sharpen"] = float(sharpen)
        self.profile.setdefault("clip_limit", EXTRA_DEFAULTS["clip_limit"])
        self.adaptive = adaptive
        # Guard against a pathologically large crop.
        self.max_side = max_side
        self._lock = threading.Lock()
        self.last_sigma = 0.0

    # -- strength selection ------------------------------------------------
    def _scale_for_noise(self, sigma):
        """Map a measured sigma onto filter strength and sharpening.

        A clean crop gets little smoothing and full sharpening; a noisy one
        gets the reverse. Sharpening noise is worse than leaving it.
        """
        if not self.adaptive:
            return 1.0, 1.0
        # sigma below ~3 is effectively clean; above ~18 is heavy CCTV noise.
        strength = float(np.clip((sigma - 2.0) / 16.0, 0.25, 1.75))
        sharpen = float(np.clip(1.0 - (sigma - 4.0) / 18.0, 0.15, 1.0))
        return strength, sharpen

    # -- stages ------------------------------------------------------------
    def _apply_denoise(self, img, strength):
        """Median first, then bilateral.

        Order matters. The median kills the impulse-like JPEG block noise
        that would otherwise dominate the bilateral filter's range weights
        and cause it to preserve artifacts as though they were edges.
        """
        k = int(self.profile.get("median", 0))
        if k >= 3:
            img = cv2.medianBlur(img, k if k % 2 else k + 1)

        bil = self.profile.get("bilateral")
        if bil:
            img = cv2.bilateralFilter(
                img, int(bil["d"]),
                float(bil["sigma_color"]) * strength,
                float(bil["sigma_space"]),
            )
        return img

    def denoise(self, crop_bgr):
        """BGR uint8 in -> BGR uint8 out, same size."""
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr
        if crop_bgr.ndim == 2:
            crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)

        img = crop_bgr
        if self.max_side and max(img.shape[:2]) > self.max_side:
            scale = self.max_side / float(max(img.shape[:2]))
            img = cv2.resize(img, (max(1, int(img.shape[1] * scale)),
                                   max(1, int(img.shape[0] * scale))),
                             interpolation=cv2.INTER_AREA)

        with self._lock:
            sigma = estimate_noise_sigma(img) if self.adaptive else 8.0
            self.last_sigma = sigma
            strength, sharpen_scale = self._scale_for_noise(sigma)

            if self.profile.get("auto_gamma"):
                mean = float(img.mean())
                if mean < 70:                    # dark crop: lift the shadows
                    img = gamma_correct(img, 1.0 + (70 - mean) / 90.0)

            img = self._apply_denoise(img, strength)

            if self.profile.get("clahe"):
                img = normalize_illumination(img, float(self.profile["clip_limit"]))

            amount = float(self.profile.get("sharpen", 0.0)) * sharpen_scale
            if amount > 0.02:
                img = unsharp_mask(
                    img, amount=amount,
                    sigma=EXTRA_DEFAULTS["sharpen_sigma"],
                    threshold=EXTRA_DEFAULTS["sharpen_threshold"],
                )

        if img.shape[:2] != crop_bgr.shape[:2]:
            img = cv2.resize(img, (crop_bgr.shape[1], crop_bgr.shape[0]),
                             interpolation=cv2.INTER_CUBIC)
        return img

    def __call__(self, crop_bgr):
        return self.denoise(crop_bgr)

    def describe(self):
        bits = [self.profile_name]
        if self.profile.get("clahe"):
            bits.append("clahe")
        if self.profile.get("sharpen", 0.0) > 0.02:
            bits.append("sharpen")
        if not self.adaptive:
            bits.append("fixed")
        return "classical:" + "+".join(bits)


def prepare_face(crop_bgr, size=112, denoiser=None):
    """Crop -> square, normalised, correctly-sized face for ArcFace.

    Worth doing even without GFPGAN: most crops are smaller than 112x112 and
    the upscale quality matters. Lanczos on the way up preserves edges far
    better than the bilinear default, and costs nothing at this size.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    img = denoiser.denoise(crop_bgr) if denoiser is not None else crop_bgr

    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top:top + h, left:left + w] = img

    interp = cv2.INTER_LANCZOS4 if side < size else cv2.INTER_AREA
    return cv2.resize(canvas, (size, size), interpolation=interp)


_denoiser = None
_denoiser_lock = threading.Lock()


def get_denoiser(profile=None, **kwargs):
    """Stage-1 singleton. ``profile`` defaults to ``config.DENOISE_PROFILE``."""
    global _denoiser
    if _denoiser is not None:
        return _denoiser
    with _denoiser_lock:
        if _denoiser is None:
            import config
            kwargs.pop("backend", None)          # accepted and ignored
            _denoiser = ClassicalDenoiser(
                profile=profile or config.DENOISE_PROFILE, **kwargs
            )
    return _denoiser


def reset_denoiser():
    """Drop the singleton, so a different profile can be selected."""
    global _denoiser
    with _denoiser_lock:
        _denoiser = None


def denoise_frame(img_bgr):
    """Stage-1 entry point used by the pipeline."""
    return get_denoiser().denoise(img_bgr)


# Kept so the original flat-script call site still resolves.
process_single_image = denoise_frame
get_classical_denoiser = get_denoiser
