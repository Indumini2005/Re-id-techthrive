"""Face detection and alignment - the step between a person crop and ArcFace.

ArcFace is trained on faces cropped tightly and warped so that the eyes,
nose and mouth corners sit at fixed pixel positions. Feeding it anything
else does not degrade gracefully: a whole-body crop produces an embedding
dominated by clothing and pose, and comparing two of those says little
about whether the faces match.

GFPGAN used to supply this incidentally - it runs facexlib internally to
find and align the face before restoring it. Dropping GFPGAN for being too
slow therefore removed alignment as well, which is a separate and more
important function. This module puts it back on its own.

Two backends:

* **SCRFD** (default) - insightface's detector, an ONNX graph in the same
  buffalo_l pack that supplies the ArcFace model. It returns five landmarks
  per face, which is exactly what the alignment warp needs, and it runs
  under the same ONNX Runtime as the rest of the pipeline. ~2.5 MB.
* **centre-crop fallback** - when no detector model is present. It assumes
  the face occupies the upper portion of a standing person and crops
  there, preserving aspect ratio. Crude, unaligned, and much worse than
  SCRFD; it exists so the pipeline runs rather than because it is adequate.
"""

import math
import threading

import cv2
import numpy as np

import config

# Where ArcFace expects the five landmarks to land in a 112x112 image:
# left eye, right eye, nose tip, left mouth corner, right mouth corner.
# These are insightface's reference points and must not be changed - the
# published recognition weights were trained against exactly this geometry.
ARCFACE_REFERENCE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(image_bgr, landmarks, size=112):
    """Warp a face so its landmarks sit on the ArcFace reference positions.

    A similarity transform (rotation, uniform scale, translation) rather
    than a full affine: it corrects head tilt and distance without shearing
    the face, which would change its identity-bearing proportions.
    """
    if landmarks is None or len(landmarks) != 5:
        return None

    source = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    reference = ARCFACE_REFERENCE * (size / 112.0)

    matrix, _ = cv2.estimateAffinePartial2D(
        source, reference, method=cv2.LMEDS
    )
    if matrix is None:
        return None
    return cv2.warpAffine(image_bgr, matrix, (size, size),
                          borderValue=(0, 0, 0), flags=cv2.INTER_LINEAR)


def distance2bbox(points, distance):
    """SCRFD predicts distances from an anchor centre to each box edge."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance):
    """Same idea for the five landmarks: offsets from the anchor centre."""
    out = []
    for i in range(0, distance.shape[1], 2):
        out.append(points[:, 0] + distance[:, i])
        out.append(points[:, 1] + distance[:, i + 1])
    return np.stack(out, axis=-1)


class SCRFDetector:
    """insightface's SCRFD face detector under ONNX Runtime."""

    def __init__(self, model_path=None, input_size=None, conf=0.4, iou=0.4,
                 provider=None, min_size=None):
        import onnxruntime as ort
        from pathlib import Path

        self.model_path = Path(model_path or config.FACE_DETECT_ONNX)
        if input_size is None:
            input_size = getattr(config, "FACE_DETECT_SIZE", 640)
        if min_size is None:
            min_size = getattr(config, "FACE_DETECT_MIN", 128)
        if not self.model_path.exists():
            raise FileNotFoundError(
                "Face detector not found at %s. It ships in the insightface "
                "buffalo_l pack as det_10g.onnx, alongside the ArcFace model "
                "this pipeline already needs. Without it, stage 4 receives an "
                "unaligned body crop and recognition will not work properly."
                % self.model_path
            )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # det_10g ships with its output shapes pinned to a 640 input even
        # though the input itself is dynamic, so running at any other size
        # logs a stale shape-mismatch warning for each of the nine outputs.
        # The values are correct; only the metadata is wrong.
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=opts,
            providers=config.provider_list(provider),
        )
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        static = shape[2] if isinstance(shape[2], int) else None
        self.input_size = static or input_size
        # A statically-shaped export has no choice of size; a dynamic one
        # should match the image rather than upscale it to 640.
        self.dynamic = static is None
        self.min_size = min(min_size, self.input_size)

        self.conf, self.iou = conf, iou
        self._lock = threading.Lock()

        # SCRFD emits 3 feature levels. Six outputs means scores+boxes only;
        # nine means landmarks too, which is what alignment requires.
        n_out = len(self.session.get_outputs())
        self.fmc = 3
        self.use_kps = n_out >= 9
        self.num_anchors = 2
        self.strides = [8, 16, 32]
        if not self.use_kps:
            raise ValueError(
                "%s has %d outputs and emits no landmarks. Alignment needs a "
                "SCRFD variant with keypoints, e.g. det_10g.onnx or "
                "det_2.5g.onnx." % (self.model_path.name, n_out)
            )

    def _choose_size(self, h, w):
        """Smallest multiple of 32 that holds the image without upscaling.

        Cost scales with the square of this, and upscaling an already-cropped
        subject adds no detail - so a 96x192 crop is detected at 192, not at
        the 640 the model is distributed for. Clamped below because SCRFD
        needs a face to be roughly 16 px at the network input to find it.
        """
        if not self.dynamic:
            return self.input_size
        longest = max(int(h), int(w))
        size = ((longest + 31) // 32) * 32
        return max(self.min_size, min(self.input_size, size))

    def detect(self, image_bgr, size=None):
        """-> list of (box, score, landmarks) in input-image coordinates."""
        if image_bgr is None or image_bgr.size == 0:
            return []

        h, w = image_bgr.shape[:2]
        if size is None:
            size = self._choose_size(h, w)
        scale = min(size / float(h), size / float(w))
        nh, nw = int(round(h * scale)), int(round(w * scale))

        resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = resized                 # SCRFD pads bottom-right

        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 128.0, (size, size), (127.5, 127.5, 127.5), swapRB=True
        )
        with self._lock:
            outputs = self.session.run(None, {self.input_name: blob})

        boxes, scores, kpss = [], [], []
        for idx, stride in enumerate(self.strides):
            score = outputs[idx].reshape(-1)
            bbox = outputs[idx + self.fmc].reshape(-1, 4) * stride
            kps = outputs[idx + self.fmc * 2].reshape(-1, 10) * stride

            height, width = size // stride, size // stride
            cx, cy = np.meshgrid(np.arange(width), np.arange(height))
            centers = np.stack([cx, cy], axis=-1).astype(np.float32) * stride
            centers = centers.reshape(-1, 2)
            if self.num_anchors > 1:
                centers = np.repeat(centers, self.num_anchors, axis=0)

            keep = score >= self.conf
            if not np.any(keep):
                continue
            boxes.append(distance2bbox(centers[keep], bbox[keep]))
            kpss.append(distance2kps(centers[keep], kps[keep]).reshape(-1, 5, 2))
            scores.append(score[keep])

        if not boxes:
            return []

        boxes = np.vstack(boxes)
        kpss = np.vstack(kpss)
        scores = np.concatenate(scores)

        nms_boxes = np.stack(
            [boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]],
            axis=1,
        )
        indices = cv2.dnn.NMSBoxes(
            nms_boxes.tolist(), scores.astype(float).tolist(),
            float(self.conf), float(self.iou),
        )
        if len(indices) == 0:
            return []
        indices = np.array(indices).reshape(-1)

        results = []
        for i in indices:
            box = boxes[i] / scale
            kps = kpss[i] / scale
            results.append((box, float(scores[i]), kps))
        results.sort(key=lambda r: -r[1])
        return results

    def align_largest(self, image_bgr, size=112):
        """Detect, take the most confident face, return it aligned."""
        faces = self.detect(image_bgr)
        if not faces:
            return None
        _, _, landmarks = faces[0]
        return align_face(image_bgr, landmarks, size)


def centre_crop_face(crop_bgr, size=112):
    """Fallback when no detector model is available. Off by default.

    Assumes a standing person fills the crop and the head occupies the top
    ~28% of it. Preserves aspect ratio, which the plain resize in stage 4
    does not.

    Read config.FACE_ALLOW_FALLBACK before relying on this. It is not a
    lower-accuracy version of alignment: ArcFace maps inputs that are not
    properly aligned faces into one region of its embedding space, where
    they resemble each other far more than the match thresholds allow. An
    unaligned crop therefore yields a confident wrong identification rather
    than no identification.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    h, w = crop_bgr.shape[:2]
    head_h = max(8, int(h * 0.28))
    face = crop_bgr[:head_h, :]

    fh, fw = face.shape[:2]
    side = max(fh, fw)
    canvas = np.zeros((side, side, 3), dtype=crop_bgr.dtype)
    top, left = (side - fh) // 2, (side - fw) // 2
    canvas[top:top + fh, left:left + fw] = face

    interp = cv2.INTER_LANCZOS4 if side < size else cv2.INTER_AREA
    return cv2.resize(canvas, (size, size), interpolation=interp)


_detector = None
_detector_lock = threading.Lock()
_detector_failed = False


def get_face_detector(**kwargs):
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = SCRFDetector(**kwargs)
    return _detector


def prepare_for_embedding(crop_bgr, size=112, denoiser=None, allow_fallback=None):
    """Person crop -> aligned face ready for ArcFace, or None.

    Restores first, then detects: the detector sees the cleaned crop, which
    is the point of stage 1 running before this.
    """
    global _detector_failed

    if crop_bgr is None or crop_bgr.size == 0:
        return None
    if allow_fallback is None:
        allow_fallback = getattr(config, "FACE_ALLOW_FALLBACK", False)

    working = denoiser.denoise(crop_bgr) if denoiser is not None else crop_bgr

    if not _detector_failed:
        try:
            aligned = get_face_detector().align_largest(working, size)
            if aligned is not None:
                return aligned
            return None            # a detector that found nothing means no face
        except Exception as exc:
            _detector_failed = True
            print("[face] detector unavailable (%s)" % exc)
            if allow_fallback:
                print("[face] FACE_ALLOW_FALLBACK is on: embedding unaligned "
                      "centre crops. Unaligned faces land close together in "
                      "ArcFace space, so expect confident WRONG matches, not "
                      "just misses. Identifications from here are not "
                      "trustworthy.")
            else:
                print("[face] refusing to embed without alignment; no "
                      "identifications will be made. Install det_10g.onnx, "
                      "or set FACE_ALLOW_FALLBACK = True to run degraded.")

    return centre_crop_face(working, size) if allow_fallback else None
