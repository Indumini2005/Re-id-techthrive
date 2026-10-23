"""Wire format between capture nodes and the hub.

Deliberately plain JSON over HTTP POST. It debugs with curl, needs no broker,
and survives the flaky links this project keeps running into. An observation
carries a 512-D float32 embedding (2 KB base64) plus optional JPEG
thumbnails for the operator display - never the source frame at full rate.

Clock handling is the part worth reading. Two laptops do not share a clock,
and NTP is often not running on a demo network, so the hub cannot pair
frames on the nodes' wall-clock timestamps directly. Each node stamps its
own ``captured_at``; the hub estimates each node's offset from the minimum
observed (arrival - captured) and corrects before pairing. Minimum, because
the lowest transit delay seen is the closest thing to pure clock offset.
"""

import base64
import json
import time

import numpy as np

PROTOCOL_VERSION = 1
HUB_DEFAULT_PORT = 8765

# A node that has not reported within this long is shown as offline.
NODE_TIMEOUT_S = 6.0


# --- vectors ---------------------------------------------------------------
def encode_vector(vector):
    """float32 ndarray -> base64 str."""
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def decode_vector(blob):
    """base64 str -> float32 ndarray."""
    if not blob:
        return None
    raw = base64.b64decode(blob)
    return np.frombuffer(raw, dtype=np.float32).copy()


# --- images ----------------------------------------------------------------
def encode_jpeg(image_bgr, quality=70, max_width=0):
    """BGR ndarray -> base64 JPEG str. Requires OpenCV (node side only)."""
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return None
    import cv2

    img = image_bgr
    if max_width and img.shape[1] > max_width:
        scale = max_width / float(img.shape[1])
        img = cv2.resize(
            img, (max_width, max(1, int(round(img.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_jpeg(blob):
    """base64 JPEG str -> BGR ndarray. Requires OpenCV (relay mode only)."""
    if not blob:
        return None
    import cv2

    raw = np.frombuffer(base64.b64decode(blob), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


# --- payloads --------------------------------------------------------------
def build_observation(
    node,
    seq,
    subjects,
    frame_shape,
    fps=0.0,
    stages_ms=None,
    frame_thumb=None,
    mode="edge",
):
    """Assemble the JSON body a node POSTs to /api/observe.

    ``subjects`` is a list of dicts with keys: box, conf, embedding (ndarray
    or None), face_thumb (base64 or None), escalated (bool), crop_jpeg
    (base64, relay mode only).
    """
    payload_subjects = []
    for s in subjects:
        payload_subjects.append({
            "box": [int(v) for v in s["box"]],
            "conf": round(float(s.get("conf", 0.0)), 4),
            "embedding": encode_vector(s.get("embedding")),
            "face_thumb": s.get("face_thumb"),
            "crop_jpeg": s.get("crop_jpeg"),
            "escalated": bool(s.get("escalated", False)),
        })

    return {
        "v": PROTOCOL_VERSION,
        "node": node,
        "seq": int(seq),
        "mode": mode,
        "captured_at": time.time(),
        "frame_w": int(frame_shape[1]),
        "frame_h": int(frame_shape[0]),
        "fps": round(float(fps), 2),
        "stages_ms": {k: round(float(v), 2) for k, v in (stages_ms or {}).items()},
        "frame_thumb": frame_thumb,
        "subjects": payload_subjects,
    }


def dumps(payload):
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def loads(raw):
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


class ClockOffset:
    """Rolling estimate of one node's clock offset relative to the hub.

    offset ~= min(arrival - captured). Transit delay is always >= 0, so the
    smallest difference observed is dominated by the clock difference rather
    than by network jitter.
    """

    def __init__(self, window=200):
        self.window = window
        self._samples = []
        self.offset = 0.0

    def update(self, captured_at, arrived_at):
        self._samples.append(arrived_at - captured_at)
        if len(self._samples) > self.window:
            self._samples = self._samples[-self.window:]
        self.offset = min(self._samples)
        return self.offset

    def to_hub_time(self, captured_at):
        return captured_at + self.offset
