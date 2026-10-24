"""Face alignment geometry and SCRFD decoding.

A warp that is subtly wrong produces a plausible-looking face and a useless
embedding, so the test asserts where landmarks actually land."""

import sys
import threading
import types
from pathlib import Path

import cv2
import numpy as np

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)
sys.modules.setdefault("onnxruntime", types.ModuleType("onnxruntime"))

from pipeline.face_align import (  # noqa: E402
    ARCFACE_REFERENCE, SCRFDetector, align_face, centre_crop_face,
    distance2bbox, distance2kps,
)

failures = []


def check(name, cond, detail=""):
    print(("OK   " if cond else "FAIL ") + name + ("   " + detail if detail else ""))
    if not cond:
        failures.append(name)


def warp_points(matrix, pts):
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.transform(pts, matrix).reshape(-1, 2)


# --- 1. identity case: landmarks already at the reference ------------------
img = np.zeros((112, 112, 3), np.uint8)
out = align_face(img, ARCFACE_REFERENCE, 112)
check("aligns a face already in position", out is not None and out.shape == (112, 112, 3),
      str(None if out is None else out.shape))

# --- 2. landmarks must LAND on the reference after the warp ----------------
# Place landmarks in a scaled, rotated, translated pose and verify the warp
# brings them back to the canonical positions.
for angle, scale, tx, ty in [(0, 1.0, 0, 0), (15, 0.6, 30, 10),
                             (-25, 1.8, -40, 25), (40, 0.35, 80, 60)]:
    theta = np.radians(angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], np.float32) * scale
    src = (ARCFACE_REFERENCE @ rot.T) + np.array([tx, ty], np.float32)

    matrix, _ = cv2.estimateAffinePartial2D(src.astype(np.float32),
                                            ARCFACE_REFERENCE, method=cv2.LMEDS)
    landed = warp_points(matrix, src)
    err = float(np.abs(landed - ARCFACE_REFERENCE).max())
    check("warp lands landmarks (rot %+d, scale %.2f)" % (angle, scale), err < 0.5,
          "max error %.3f px" % err)

# --- 3. the transform is a similarity, not a full affine -------------------
# Shear would change identity-bearing proportions, so the matrix must have
# equal scale on both axes and stay orthogonal.
theta = np.radians(20)
rot = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]], np.float32) * 1.3
src = (ARCFACE_REFERENCE @ rot.T) + np.array([12, -7], np.float32)
m, _ = cv2.estimateAffinePartial2D(src.astype(np.float32), ARCFACE_REFERENCE,
                                   method=cv2.LMEDS)
a, b, c, d = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
check("transform is a similarity (no shear)",
      abs(a - d) < 1e-3 and abs(b + c) < 1e-3,
      "a=%.4f d=%.4f b=%.4f c=%.4f" % (a, d, b, c))
sx = np.hypot(a, c)
check("uniform scale recovered", abs(sx - 1.0 / 1.3) < 1e-2, "scale %.4f vs %.4f" % (sx, 1 / 1.3))

# --- 4. output size scaling -------------------------------------------------
for size in (112, 128, 224):
    out = align_face(np.zeros((200, 200, 3), np.uint8), ARCFACE_REFERENCE * (size / 112.0), size)
    check("size %d output" % size, out is not None and out.shape == (size, size, 3))

# --- 5. bad input -----------------------------------------------------------
check("rejects None landmarks", align_face(img, None) is None)
check("rejects wrong landmark count", align_face(img, ARCFACE_REFERENCE[:3]) is None)

# --- 6. centre-crop fallback preserves aspect and size ----------------------
for (w, h) in [(96, 192), (40, 80), (200, 200)]:
    crop = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    face = centre_crop_face(crop, 112)
    check("fallback %dx%d -> 112" % (w, h),
          face is not None and face.shape == (112, 112, 3))
check("fallback rejects empty", centre_crop_face(None) is None)

# --- 7. SCRFD distance decoding --------------------------------------------
centers = np.array([[100.0, 100.0], [50.0, 60.0]], np.float32)
dist = np.array([[10, 20, 30, 40], [5, 5, 5, 5]], np.float32)
boxes = distance2bbox(centers, dist)
check("distance2bbox decodes l,t,r,b offsets",
      np.allclose(boxes[0], [90, 80, 130, 140]) and np.allclose(boxes[1], [45, 55, 55, 65]),
      str(boxes[0].tolist()))

kps_d = np.arange(10, dtype=np.float32).reshape(1, 10)
kps = distance2kps(np.array([[10.0, 20.0]], np.float32), kps_d)
check("distance2kps decodes x,y pairs",
      np.allclose(kps.reshape(5, 2)[:, 0], [10, 12, 14, 16, 18])
      and np.allclose(kps.reshape(5, 2)[:, 1], [21, 23, 25, 27, 29]),
      str(kps.reshape(5, 2).tolist()))

# --- 8. SCRFD end-to-end decode with a fake session ------------------------
SIZE, STRIDES, NA = 640, [8, 16, 32], 2
TRUE_BOX = (200.0, 150.0, 360.0, 370.0)
TRUE_KPS = np.array([[250, 220], [310, 220], [280, 265], [255, 310], [305, 310]], np.float32)


class FakeSession:
    def get_inputs(self):
        return [types.SimpleNamespace(name="input.1", shape=[1, 3, SIZE, SIZE])]

    def get_outputs(self):
        return [types.SimpleNamespace(name="o%d" % i) for i in range(9)]

    def run(self, names, feed):
        scores, bboxes, kpss = [], [], []
        for stride in STRIDES:
            n = (SIZE // stride) ** 2 * NA
            s = np.zeros((n, 1), np.float32)
            b = np.zeros((n, 4), np.float32)
            k = np.zeros((n, 10), np.float32)
            if stride == 16:                       # plant one detection here
                gw = SIZE // stride
                cx, cy = 280.0, 260.0              # near the box centre
                gx, gy = int(cx // stride), int(cy // stride)
                idx = (gy * gw + gx) * NA
                ax, ay = gx * stride, gy * stride
                s[idx] = 0.95
                b[idx] = [(ax - TRUE_BOX[0]) / stride, (ay - TRUE_BOX[1]) / stride,
                          (TRUE_BOX[2] - ax) / stride, (TRUE_BOX[3] - ay) / stride]
                k[idx] = ((TRUE_KPS - np.array([ax, ay], np.float32)) / stride).reshape(-1)
            scores.append(s)
            bboxes.append(b)
            kpss.append(k)
        return scores + bboxes + kpss


det = SCRFDetector.__new__(SCRFDetector)
det.session = FakeSession()
det.input_name = "input.1"
det.input_size = SIZE
det.min_size = 128
det.dynamic = False        # FakeSession plants its detection at 640 only
det.conf, det.iou = 0.4, 0.4
det.fmc, det.use_kps, det.num_anchors = 3, True, NA
det.strides = STRIDES
det._lock = threading.Lock()

frame = np.zeros((SIZE, SIZE, 3), np.uint8)       # scale 1.0, no letterbox error
faces = det.detect(frame)
check("SCRFD returns one detection", len(faces) == 1, "got %d" % len(faces))
if faces:
    box, score, kps = faces[0]
    berr = float(np.abs(np.array(box) - np.array(TRUE_BOX)).max())
    kerr = float(np.abs(kps - TRUE_KPS).max())
    check("SCRFD box decoded", berr < 1.0, "max err %.2f px  %s" % (berr, np.round(box, 1)))
    check("SCRFD landmarks decoded", kerr < 1.0, "max err %.2f px" % kerr)
    check("SCRFD score kept", abs(score - 0.95) < 1e-5, "%.3f" % score)

    aligned = align_face(frame, kps, 112)
    check("aligned output from detected landmarks",
          aligned is not None and aligned.shape == (112, 112, 3))

# --- 9. scaling: a non-square frame must map back correctly ----------------
tall = np.zeros((1280, 720, 3), np.uint8)
faces = det.detect(tall)
if faces:
    box, _, kps = faces[0]
    scale = min(SIZE / 1280.0, SIZE / 720.0)
    expected = np.array(TRUE_BOX) / scale
    err = float(np.abs(np.array(box) - expected).max())
    check("non-square frame rescaled back", err < 2.0, "max err %.2f px" % err)
else:
    check("non-square frame rescaled back", False, "no detection")

# --- 10. detector input size is chosen, not fixed at 640 -------------------
# Cost goes as the square of this: 640 measured 56.0 ms against 5.5 ms at
# 192 on one laptop CPU. A person crop must not be upscaled into 640.
sizer = SCRFDetector.__new__(SCRFDetector)
sizer.input_size, sizer.min_size, sizer.dynamic = 640, 128, True

check("96x192 crop detected at 192, not 640", sizer._choose_size(192, 96) == 192,
      "chose %d" % sizer._choose_size(192, 96))
check("size is a multiple of 32 (stride-32 level divides evenly)",
      all(sizer._choose_size(h, w) % 32 == 0
          for h, w in [(192, 96), (200, 100), (57, 23), (333, 777)]))
check("full frame clamped to the 640 ceiling",
      sizer._choose_size(1080, 1920) == 640, "chose %d" % sizer._choose_size(1080, 1920))
check("tiny crop raised to the 128 floor",
      sizer._choose_size(40, 18) == 128, "chose %d" % sizer._choose_size(40, 18))
check("never smaller than the image until the ceiling",
      all(sizer._choose_size(h, w) >= min(640, max(h, w))
          for h, w in [(192, 96), (300, 150), (639, 100), (641, 100)]))
check("static export ignores the choice",
      SCRFDetector._choose_size(
          types.SimpleNamespace(dynamic=False, input_size=640, min_size=128),
          192, 96) == 640)

# The chosen size must still leave the head big enough for SCRFD's finest
# level (stride 8, so ~16 px minimum).
for h, w in [(192, 96), (120, 60), (40, 18)]:
    s = sizer._choose_size(h, w)
    scale = min(s / float(h), s / float(w))
    head_px = h * 0.28 * scale
    check("head stays detectable for a %dx%d crop" % (w, h), head_px >= 16.0,
          "size %d -> head %.0f px" % (s, head_px))

print("\n" + ("ALL PASS" if not failures else "%d FAILURE(S): %s" % (len(failures), failures)))
sys.exit(1 if failures else 0)
