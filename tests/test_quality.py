"""Frame sharpness and the face quality gate.

Both are cheap heuristics standing in front of expensive stages, so the
thing worth asserting is that they rank inputs the right way round - and,
for sharpness, that sensor noise does not pass for detail.
"""

import sys
import types
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.modules.setdefault("onnxruntime", types.ModuleType("onnxruntime"))

import config                                                        # noqa: E402
from pipeline import face_align                                      # noqa: E402
from pipeline.face_align import (ARCFACE_REFERENCE, face_quality,    # noqa: E402
                                 quality_gate)
from pipeline.sources import frame_sharpness                         # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("OK   " if cond else "FAIL ") + name + ("   " + detail if detail else ""))
    if not cond:
        failures.append(name)


rng = np.random.default_rng(3)

# ===========================================================================
# Sharpness
# ===========================================================================
# A 640x360 scene with real edges: a few rectangles and a person-sized block.
scene = np.full((360, 640, 3), 90, np.uint8)
for (x, y, w, h, c) in [(40, 60, 120, 200, 200), (250, 100, 60, 220, 30),
                        (380, 40, 200, 90, 160), (420, 200, 150, 120, 60)]:
    cv2.rectangle(scene, (x, y), (x + w, y + h), (c, c, c), -1)
cv2.putText(scene, "CAM", (60, 330), cv2.FONT_HERSHEY_SIMPLEX, 2, (240, 240, 240), 4)

sharp = frame_sharpness(scene)
soft = frame_sharpness(cv2.GaussianBlur(scene, (0, 0), 2.0))
smeared = frame_sharpness(cv2.GaussianBlur(scene, (0, 0), 5.0))
check("blur lowers sharpness, monotonically", sharp > soft > smeared,
      "sharp %.0f > sigma2 %.0f > sigma5 %.0f" % (sharp, soft, smeared))

# Horizontal motion blur, the case that matters for a walking subject.
kernel = np.zeros((1, 15), np.float32)
kernel[0, :] = 1.0 / 15
moving = frame_sharpness(cv2.filter2D(scene, -1, kernel))
check("motion blur lowers sharpness", moving < sharp, "%.0f < %.0f" % (moving, sharp))

# The trap: noise has high Laplacian energy. A noisy frame of the same scene
# must not outrank the clean one by much, and a noisy BLURRED frame must not
# outrank a clean sharp one.
def noisy(img, sigma):
    out = img.astype(np.int16) + rng.normal(0, sigma, img.shape).astype(np.int16)
    return np.clip(out, 0, 255).astype(np.uint8)

noisy_soft = frame_sharpness(noisy(cv2.GaussianBlur(scene, (0, 0), 2.0), 10))
check("noise does not make a blurred frame look sharp", noisy_soft < sharp,
      "noisy blurred %.0f < clean sharp %.0f" % (noisy_soft, sharp))

# Same comparison WITHOUT the downscale, to show the downscale is load-bearing.
def raw_lap(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())

raw_noisy_soft = raw_lap(noisy(cv2.GaussianBlur(scene, (0, 0), 2.0), 10))
raw_sharp = raw_lap(scene)
print("     (at full resolution the same pair scores noisy-blurred %.0f vs sharp %.0f%s)"
      % (raw_noisy_soft, raw_sharp,
         " - noise WINS there, which is why it downscales first"
         if raw_noisy_soft > raw_sharp else ""))

check("tiny and grey inputs are handled",
      frame_sharpness(np.zeros((20, 30), np.uint8)) == 0.0
      and frame_sharpness(np.zeros((90, 120, 3), np.uint8)) == 0.0)

# Cost - it runs on every frame on the camera threads.
import time
big = noisy(cv2.resize(scene, (1920, 1080)), 6)
frame_sharpness(big)
t0 = time.perf_counter()
for _ in range(50):
    frame_sharpness(big)
ms = (time.perf_counter() - t0) / 50 * 1000
check("cheap enough for every frame", ms < 5.0, "%.2f ms per 1080p frame" % ms)

# ===========================================================================
# Face quality
# ===========================================================================
eye, yaw = face_quality(ARCFACE_REFERENCE)
check("reference face: eyes 35 px apart", abs(eye - 35.2) < 0.5, "%.2f px" % eye)
check("reference face: frontal (yaw ~0)", yaw < 0.02, "%.3f" % yaw)


def rotate(points, deg, scale=1.0, shift=(0.0, 0.0)):
    t = np.radians(deg)
    r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], np.float32) * scale
    return points @ r.T + np.array(shift, np.float32)


# A tilted head is not a turned head.
for deg in (-30, 15, 45):
    _, y = face_quality(rotate(ARCFACE_REFERENCE, deg))
    check("tilted %+d deg still reads frontal" % deg, y < 0.02, "yaw %.3f" % y)

# Simulate turning: the nose slides towards one eye along the eye line.
def turned(frac):
    pts = ARCFACE_REFERENCE.copy()
    eye_dist = pts[1, 0] - pts[0, 0]
    pts[2, 0] += frac * eye_dist
    return pts

for frac, expect in [(0.10, None), (0.30, None), (0.45, "turned_away")]:
    reason = quality_gate(turned(frac))
    check("nose offset %.2f -> %s" % (frac, expect or "accepted"), reason == expect,
          "got %s" % reason)
check("turning either way counts", quality_gate(turned(-0.45)) == "turned_away")

# Size: the same face scaled down until the eyes are too close.
for scale, expect in [(1.0, None), (0.4, None), (0.25, "too_small")]:
    pts = ARCFACE_REFERENCE * scale
    eye, _ = face_quality(pts)
    reason = quality_gate(pts)
    check("eyes %.1f px -> %s" % (eye, expect or "accepted"), reason == expect,
          "got %s (min %.0f)" % (reason, config.FACE_MIN_EYE_DIST))

check("degenerate landmarks rejected, not divided by zero",
      quality_gate(np.zeros((5, 2), np.float32)) is not None)

# ===========================================================================
# The gate inside prepare_for_embedding
# ===========================================================================
class FakeDetector:
    def __init__(self, faces):
        self.faces = faces

    def detect(self, image):
        return self.faces


crop = rng.integers(0, 255, (192, 96, 3), dtype=np.uint8)
good = ARCFACE_REFERENCE * 0.6 + np.array([10, 10], np.float32)
profile = turned(0.5) * 0.6 + np.array([10, 10], np.float32)
tiny = ARCFACE_REFERENCE * 0.2 + np.array([40, 20], np.float32)

saved_detector, saved_failed = face_align._detector, face_align._detector_failed
face_align._detector_failed = False
for k in face_align.QUALITY_STATS:
    face_align.QUALITY_STATS[k] = 0

try:
    face_align._detector = FakeDetector([((0, 0, 50, 50), 0.9, good)])
    out = face_align.prepare_for_embedding(crop, 112)
    check("good face is aligned", out is not None and out.shape == (112, 112, 3))
    check("  and no reject reason", face_align.last_reject_reason is None)

    face_align._detector = FakeDetector([((0, 0, 50, 50), 0.9, profile)])
    check("profile face is refused", face_align.prepare_for_embedding(crop, 112) is None)
    check("  reason: turned_away", face_align.last_reject_reason == "turned_away",
          str(face_align.last_reject_reason))

    face_align._detector = FakeDetector([((0, 0, 10, 10), 0.9, tiny)])
    check("tiny face is refused", face_align.prepare_for_embedding(crop, 112) is None)
    check("  reason: too_small", face_align.last_reject_reason == "too_small")

    face_align._detector = FakeDetector([])
    check("no face -> None", face_align.prepare_for_embedding(crop, 112) is None)
    check("  reason: no_face", face_align.last_reject_reason == "no_face")

    face_align._detector = FakeDetector([((0, 0, 50, 50), 0.9, profile)])
    out = face_align.prepare_for_embedding(crop, 112, quality_check=False)
    check("quality_check=False lets it through", out is not None)

    q = face_align.QUALITY_STATS
    check("counters tally each outcome",
          q == {"accepted": 2, "too_small": 1, "turned_away": 1, "no_face": 1}, str(q))
finally:
    face_align._detector, face_align._detector_failed = saved_detector, saved_failed

print("\n" + ("ALL PASS" if not failures
              else "%d FAILURE(S): %s" % (len(failures), failures)))
sys.exit(1 if failures else 0)
