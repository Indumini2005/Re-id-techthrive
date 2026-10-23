"""Stage 2 - YOLOv8 subject detection and cropping.

Two backends, selected by the weights file extension:

* ``.onnx`` - ONNX Runtime with the letterbox and NMS implemented here.
  This is the one that runs on the original Jetson Nano: JetPack 4.6 pins
  the device to Python 3.6, and ultralytics requires 3.8+, so it cannot be
  installed there at all. Export the graph on your workstation
  (``export/export_yolo.py --format onnx``) and copy it over.
* ``.pt`` / ``.engine`` - ultralytics, for development machines and for
  Xavier/Orin boards running JetPack 5 or 6.
"""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config


@dataclass
class Subject:
    """One detected person."""
    box: Tuple[int, int, int, int]   # x1, y1, x2, y2 in frame coordinates
    conf: float
    crop: np.ndarray                 # BGR pixels, margin included

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """Resize preserving aspect ratio and pad to a square, the way
    ultralytics does it. Returns ``(padded, ratio, (dw, dh))``."""
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new_shape[1] - nw) / 2.0, (new_shape[0] - nh) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, r, (left, top)


class _OnnxBackend:
    """YOLOv8 under ONNX Runtime, with postprocessing written out in full.

    The exported graph emits ``(1, 4 + num_classes, num_anchors)`` with boxes
    as cx/cy/w/h in letterboxed pixels and per-class scores already activated
    - there is no objectness column, unlike YOLOv5.
    """

    def __init__(self, weights: Path, imgsz: int, provider: Optional[str] = None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(weights), sess_options=opts, providers=config.provider_list(provider)
        )
        self.input_name = self.session.get_inputs()[0].name

        # A statically-shaped export dictates the letterbox size.
        shape = self.session.get_inputs()[0].shape
        static_h = shape[2] if isinstance(shape[2], int) else None
        self.imgsz = static_h or imgsz

    def infer(self, frame_bgr, conf, iou, person_class):
        padded, ratio, (dw, dh) = letterbox(frame_bgr, self.imgsz)
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])

        output = self.session.run(None, {self.input_name: blob})[0]
        pred = np.squeeze(output, axis=0)            # (4 + nc, anchors)

        # Orientation: ultralytics emits (4 + nc, anchors), some exporters
        # transpose it. Any real export has far more anchors (3549 at 416,
        # 8400 at 640) than channels (4 + 80), so the smaller axis is the
        # channel axis. Exports with NMS baked in have a different output
        # signature entirely and are not supported here - export the raw
        # graph, which is what export/export_yolo.py produces.
        if pred.ndim != 2:
            return [], []
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T                             # -> (anchors, 4 + nc)

        if pred.shape[1] <= 4 + person_class:
            return [], []
        scores = pred[:, 4 + person_class]
        keep = scores >= conf
        if not np.any(keep):
            return [], []

        boxes_xywh = pred[keep, :4]
        scores = scores[keep]

        # cx,cy,w,h (letterboxed) -> x1,y1,x2,y2 (original frame)
        cx, cy, bw, bh = boxes_xywh.T
        x1 = (cx - bw / 2.0 - dw) / ratio
        y1 = (cy - bh / 2.0 - dh) / ratio
        x2 = (cx + bw / 2.0 - dw) / ratio
        y2 = (cy + bh / 2.0 - dh) / ratio

        h, w = frame_bgr.shape[:2]
        x1 = np.clip(x1, 0, w - 1)
        y1 = np.clip(y1, 0, h - 1)
        x2 = np.clip(x2, 0, w - 1)
        y2 = np.clip(y2, 0, h - 1)

        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        indices = cv2.dnn.NMSBoxes(
            nms_boxes.tolist(), scores.astype(float).tolist(), float(conf), float(iou)
        )
        if len(indices) == 0:
            return [], []
        indices = np.array(indices).reshape(-1)

        boxes = np.stack([x1, y1, x2, y2], axis=1)[indices]
        return boxes, scores[indices]


class _UltralyticsBackend:
    """PyTorch or TensorRT through the ultralytics package."""

    def __init__(self, weights: Path, imgsz: int):
        from ultralytics import YOLO

        self.model = YOLO(str(weights), task="detect")
        self.imgsz = imgsz

    def infer(self, frame_bgr, conf, iou, person_class):
        results = self.model.predict(
            source=frame_bgr, classes=[person_class], conf=conf,
            iou=iou, imgsz=self.imgsz, verbose=False,
        )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return [], []
        return result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()


class PersonDetector:
    def __init__(
        self,
        weights: Optional[Path] = None,
        conf: float = config.YOLO_CONF,
        iou: float = config.YOLO_IOU,
        margin: float = config.YOLO_CROP_MARGIN,
        imgsz: int = None,
        provider: Optional[str] = None,
    ):
        weights = Path(weights or config.YOLO_WEIGHTS)
        imgsz = imgsz or config.YOLO_IMGSZ

        if not weights.exists():
            # Try the sibling formats before giving up.
            for suffix in (".onnx", ".engine", ".pt"):
                candidate = weights.with_suffix(suffix)
                if candidate.exists():
                    weights = candidate
                    break
            else:
                raise FileNotFoundError(
                    "YOLO weights not found at %s. Export one with "
                    "export/export_yolo.py (use --format onnx for a Jetson "
                    "Nano, whose Python 3.6 cannot run ultralytics)." % weights
                )

        self.weights = weights
        if weights.suffix.lower() == ".onnx":
            self.backend = _OnnxBackend(weights, imgsz, provider)
            self.backend_name = "onnxruntime"
        else:
            self.backend = _UltralyticsBackend(weights, imgsz)
            self.backend_name = "ultralytics"

        self.conf, self.iou, self.margin = conf, iou, margin
        self.imgsz = getattr(self.backend, "imgsz", imgsz)
        self._lock = threading.Lock()

    def _pad_box(self, box, img_h: int, img_w: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = (int(v) for v in box)
        mx = int((x2 - x1) * self.margin)
        my = int((y2 - y1) * self.margin)
        return (
            max(0, x1 - mx),
            max(0, y1 - my),
            min(img_w, x2 + mx),
            min(img_h, y2 + my),
        )

    def detect(self, frame_bgr: np.ndarray, max_subjects: int = None) -> List[Subject]:
        """Returns subjects sorted largest-first (closest to the camera)."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        if max_subjects is None:
            max_subjects = config.YOLO_MAX_SUBJECTS

        with self._lock:
            boxes, confs = self.backend.infer(
                frame_bgr, self.conf, self.iou, config.YOLO_PERSON_CLASS
            )

        if len(boxes) == 0:
            return []

        img_h, img_w = frame_bgr.shape[:2]
        subjects = []
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = self._pad_box(box, img_h, img_w)
            if x2 - x1 < 16 or y2 - y1 < 32:     # too small to hold a usable face
                continue
            subjects.append(
                Subject(box=(x1, y1, x2, y2), conf=float(conf),
                        crop=frame_bgr[y1:y2, x1:x2].copy())
            )

        subjects.sort(key=lambda s: s.area, reverse=True)
        return subjects[:max_subjects]


_detector = None
_detector_lock = threading.Lock()


def get_detector(**kwargs) -> PersonDetector:
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = PersonDetector(**kwargs)
    return _detector


def get_person_crops(denoised_frame: np.ndarray) -> List[Subject]:
    return get_detector().detect(denoised_frame)


def get_person_crop(denoised_frame: np.ndarray) -> Optional[np.ndarray]:
    """Largest person in the frame, or None. Matches the original API."""
    subjects = get_person_crops(denoised_frame)
    return subjects[0].crop if subjects else None
