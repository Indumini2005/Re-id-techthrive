"""YOLOv8 ONNX letterbox + box decode + NMS, with cv2's array ops stubbed in
numpy so the GEOMETRY is what gets tested, not OpenCV."""

import sys
import types
from pathlib import Path

import numpy as np

# --- minimal cv2 -----------------------------------------------------------
cv2 = types.ModuleType("cv2")
cv2.INTER_LINEAR = 1
cv2.INTER_AREA = 3
cv2.BORDER_CONSTANT = 0
cv2.COLOR_BGR2RGB = 4


def _resize(img, dsize, interpolation=None):
    w, h = dsize
    ys = (np.arange(h) * img.shape[0] / float(h)).astype(int).clip(0, img.shape[0] - 1)
    xs = (np.arange(w) * img.shape[1] / float(w)).astype(int).clip(0, img.shape[1] - 1)
    return img[np.ix_(ys, xs)]


def _copy_make_border(img, top, bottom, left, right, borderType, value=0):
    return np.pad(img, ((top, bottom), (left, right), (0, 0)),
                  mode="constant", constant_values=0)


def _nms_boxes(boxes, scores, score_thr, nms_thr):
    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        x1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        y1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        x2 = np.minimum(boxes[i, 0] + boxes[i, 2], boxes[order[1:], 0] + boxes[order[1:], 2])
        y2 = np.minimum(boxes[i, 1] + boxes[i, 3], boxes[order[1:], 1] + boxes[order[1:], 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        union = boxes[i, 2] * boxes[i, 3] + boxes[order[1:], 2] * boxes[order[1:], 3] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = order[1:][iou <= nms_thr]
    return np.array(keep).reshape(-1, 1)


cv2.resize = _resize
cv2.copyMakeBorder = _copy_make_border
cv2.cvtColor = lambda img, code: img[:, :, ::-1]
cv2.dnn = types.SimpleNamespace(NMSBoxes=_nms_boxes)
sys.modules["cv2"] = cv2
sys.modules.setdefault("onnxruntime", types.ModuleType("onnxruntime"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from pipeline.stage2_detect import PersonDetector, letterbox  # noqa: E402

IMGSZ = 416
NUM_CLASSES = 80


class FakeSession:
    """Stands in for an ONNX Runtime session, emitting boxes we placed."""

    def __init__(self, boxes_xywh, scores, imgsz=IMGSZ):
        self.boxes_xywh, self.scores, self.imgsz = boxes_xywh, scores, imgsz

    def get_inputs(self):
        return [types.SimpleNamespace(name="images", shape=[1, 3, self.imgsz, self.imgsz])]

    def run(self, outputs, feed):
        # A real YOLOv8 export emits one column per anchor: 3549 at 416px,
        # 8400 at 640px. Emitting only the detections would misrepresent the
        # output shape the decoder has to interpret.
        anchors = (self.imgsz // 8) ** 2 + (self.imgsz // 16) ** 2 + (self.imgsz // 32) ** 2
        pred = np.zeros((4 + NUM_CLASSES, anchors), dtype=np.float32)
        for i, (box, score) in enumerate(zip(self.boxes_xywh, self.scores)):
            pred[:4, i] = box
            pred[4 + 0, i] = score           # class 0 = person
        return [pred[None]]


def make_detector(boxes_xywh, scores, imgsz=IMGSZ):
    det = PersonDetector.__new__(PersonDetector)
    backend = object.__new__(
        __import__("pipeline.stage2_detect", fromlist=["x"])._OnnxBackend
    )
    backend.session = FakeSession(boxes_xywh, scores, imgsz)
    backend.input_name = "images"
    backend.imgsz = imgsz
    det.backend = backend
    det.backend_name = "onnxruntime"
    det.conf, det.iou, det.margin = 0.5, 0.45, 0.0
    det.imgsz = imgsz
    import threading
    det._lock = threading.Lock()
    return det


def to_letterbox_xywh(box_xyxy, frame_shape, imgsz):
    """Forward-map a frame box into letterboxed cx,cy,w,h - the inverse of
    what the detector does, so a round-trip must return the original."""
    frame = np.zeros(frame_shape, np.uint8)
    _, r, (dw, dh) = letterbox(frame, imgsz)
    x1, y1, x2, y2 = box_xyxy
    lx1, ly1 = x1 * r + dw, y1 * r + dh
    lx2, ly2 = x2 * r + dw, y2 * r + dh
    return [(lx1 + lx2) / 2, (ly1 + ly2) / 2, lx2 - lx1, ly2 - ly1]


failures = 0


def check(name, cond, detail=""):
    global failures
    print(("OK   " if cond else "FAIL ") + name + ("   " + detail if detail else ""))
    if not cond:
        failures += 1


# --- 1. letterbox geometry -------------------------------------------------
for shape in [(1080, 1920, 3), (360, 640, 3), (480, 480, 3), (720, 405, 3)]:
    padded, r, (dw, dh) = letterbox(np.zeros(shape, np.uint8), IMGSZ)
    ok = padded.shape[:2] == (IMGSZ, IMGSZ)
    ok &= abs(r - min(IMGSZ / shape[0], IMGSZ / shape[1])) < 1e-6
    check("letterbox %dx%d -> %dx%d" % (shape[1], shape[0], padded.shape[1], padded.shape[0]),
          ok, "ratio %.4f pad (%d,%d)" % (r, dw, dh))

# --- 2. a box round-trips through letterbox and back -----------------------
FRAME = (360, 640, 3)
truth = (100.0, 60.0, 260.0, 340.0)
lb = to_letterbox_xywh(truth, FRAME, IMGSZ)
det = make_detector([lb], [0.91])
subjects = det.detect(np.zeros(FRAME, np.uint8))

check("one detection returned", len(subjects) == 1, "got %d" % len(subjects))
if subjects:
    got = subjects[0].box
    err = max(abs(a - b) for a, b in zip(got, truth))
    check("box recovered", err <= 1.0, "%s vs truth %s (max err %.2f px)" % (got, truth, err))
    check("confidence kept", abs(subjects[0].conf - 0.91) < 1e-5)
    crop_h, crop_w = subjects[0].crop.shape[:2]
    check("crop matches box", crop_w == got[2] - got[0] and crop_h == got[3] - got[1],
          "%dx%d" % (crop_w, crop_h))

# --- 3. wide frame (letterbox pads top/bottom) -----------------------------
FRAME_W = (1080, 1920, 3)
truth_w = (800.0, 300.0, 1100.0, 900.0)
det = make_detector([to_letterbox_xywh(truth_w, FRAME_W, IMGSZ)], [0.8])
subjects = det.detect(np.zeros(FRAME_W, np.uint8))
err = max(abs(a - b) for a, b in zip(subjects[0].box, truth_w)) if subjects else 999
check("1080p box recovered", err <= 3.0, "max err %.2f px" % err)

# --- 4. sorting largest-first and max_subjects -----------------------------
small = to_letterbox_xywh((10.0, 10.0, 60.0, 110.0), FRAME, IMGSZ)
big = to_letterbox_xywh((200.0, 50.0, 420.0, 350.0), FRAME, IMGSZ)
det = make_detector([small, big], [0.7, 0.75])
subjects = det.detect(np.zeros(FRAME, np.uint8))
check("largest subject first", len(subjects) == 2 and subjects[0].area > subjects[1].area,
      "areas %s" % [s.area for s in subjects])
subjects = det.detect(np.zeros(FRAME, np.uint8), max_subjects=1)
check("max_subjects honoured", len(subjects) == 1)

# --- 5. sub-threshold detections are dropped -------------------------------
det = make_detector([lb], [0.2])
check("low confidence filtered", det.detect(np.zeros(FRAME, np.uint8)) == [])

# --- 6. duplicate boxes collapse under NMS ---------------------------------
dup = to_letterbox_xywh((102.0, 62.0, 258.0, 338.0), FRAME, IMGSZ)
det = make_detector([lb, dup], [0.9, 0.85])
subjects = det.detect(np.zeros(FRAME, np.uint8))
check("NMS removes the duplicate", len(subjects) == 1, "got %d" % len(subjects))

# --- 7. tiny boxes rejected as unusable for faces --------------------------
tiny = to_letterbox_xywh((10.0, 10.0, 20.0, 25.0), FRAME, IMGSZ)
det = make_detector([tiny], [0.99])
check("tiny box rejected", det.detect(np.zeros(FRAME, np.uint8)) == [])

# --- 8. empty frame --------------------------------------------------------
det = make_detector([], [])
check("no detections -> []", det.detect(np.zeros(FRAME, np.uint8)) == [])

print("\n" + ("ALL PASS" if not failures else "%d FAILURE(S)" % failures))
sys.exit(1 if failures else 0)
