"""Shared helpers: timing, drawing, frame scaling."""

import time
from collections import defaultdict, deque
from typing import List

import cv2
import numpy as np


class StageTimer:
    """Rolling per-stage latency, for the on-screen HUD and benchmarks."""

    def __init__(self, window: int = 30):
        self._samples = defaultdict(lambda: deque(maxlen=window))
        self._starts = {}

    def start(self, stage: str) -> None:
        self._starts[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        elapsed_ms = (time.perf_counter() - self._starts.pop(stage, time.perf_counter())) * 1000.0
        self._samples[stage].append(elapsed_ms)
        return elapsed_ms

    def mean(self, stage: str) -> float:
        vals = self._samples.get(stage)
        return float(np.mean(vals)) if vals else 0.0

    def summary(self) -> dict:
        return {k: round(float(np.mean(v)), 2) for k, v in self._samples.items() if v}

    def __call__(self, stage: str):
        return _TimerContext(self, stage)


class _TimerContext:
    def __init__(self, timer: StageTimer, stage: str):
        self.timer, self.stage = timer, stage

    def __enter__(self):
        self.timer.start(self.stage)
        return self

    def __exit__(self, *exc):
        self.timer.stop(self.stage)
        return False


def draw_hud(frame: np.ndarray, lines: List[str], color=(0, 255, 160)) -> np.ndarray:
    """Draws a translucent stat panel in the top-left corner."""
    if not lines:
        return frame
    pad, lh = 10, 22
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for t in lines) + 2 * pad
    h = lh * len(lines) + pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + w, 10 + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, text in enumerate(lines):
        cv2.putText(
            frame, text, (10 + pad, 10 + pad + lh * i + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
        )
    return frame


def draw_box(frame: np.ndarray, box, label: str, color=(0, 220, 255)) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 4, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return frame


# One colour per visitor, stable for as long as they are remembered, so the
# same person is the same colour on both cameras. Enrolled (watchlist) people
# are always red; people with no face yet are grey.
_PALETTE = [(255, 180, 80), (120, 255, 120), (200, 120, 255), (0, 220, 255),
            (255, 120, 200), (120, 220, 255), (80, 200, 160), (220, 220, 120)]
WATCHLIST_COLOR = (60, 60, 255)
UNKNOWN_COLOR = (170, 170, 170)
LOITER_COLOR = (0, 0, 255)


def identity_color(identity):
    if not identity:
        return UNKNOWN_COLOR
    if identity.get("kind") == "enrolled":
        return WATCHLIST_COLOR
    try:
        n = int(str(identity.get("key", "")).split(":")[1])
    except (IndexError, ValueError):
        n = 0
    return _PALETTE[n % len(_PALETTE)]


def scene_label(row):
    """'A3 Visitor #2 91% 12s' - track ID, who, how sure, how long here."""
    parts = [row["id"]]
    ident = row.get("identity")
    if ident:
        parts.append(ident["name"])
        if ident.get("confidence") is not None:
            parts.append("%d%%" % round(100 * ident["confidence"]))
        elif ident.get("score") is not None:
            parts.append("s%.2f" % ident["score"])
    if row.get("dwell_s") is not None:
        parts.append("%ds" % int(row["dwell_s"]))
    return " ".join(parts)


def draw_scene_view(frame: np.ndarray, view: dict) -> np.ndarray:
    """Draw SceneAnalytics.state()['views'][camera] onto that camera's frame:
    box and label per person, their recent path, and an arrow to where the
    Kalman filter predicts they will be in one second."""
    labels = []
    for row in view.get("tracks", []):
        loiter = row.get("loitering")
        color = LOITER_COLOR if loiter else identity_color(row.get("identity"))
        x1, y1, x2, y2 = [int(round(v)) for v in row["box"]]

        trail = row.get("trail") or []
        if len(trail) >= 2:
            pts = np.array([[int(x), int(y)] for x, y in trail], np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)

        thickness = 3 if loiter else (1 if row.get("coasting") else 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        pb = row.get("pred_box")
        if pb:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            px, py = int((pb[0] + pb[2]) / 2), int((pb[1] + pb[3]) / 2)
            if abs(px - cx) + abs(py - cy) > 6:
                cv2.arrowedLine(frame, (cx, cy), (px, py), color, 2, cv2.LINE_AA, tipLength=0.25)

        label = scene_label(row)
        labels.append(("LOITERING " + label if loiter else label, color, x1, y1, y2))

    # Labels after all boxes, placed so people standing close together do
    # not get their names drawn over each other.
    fh, fw = frame.shape[:2]
    placed = []
    for label, color, x1, y1, y2 in labels:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bw, bh = tw + 8, th + 8
        x = max(0, min(x1, fw - bw))
        y = None
        for top in (y1 - bh, y1, y2, y1 - 2 * bh, y1 + bh):
            if top < 0 or top + bh > fh:
                continue
            if not any(x < px + pw and x + bw > px and top < py + ph and top + bh > py
                       for px, py, pw, ph in placed):
                y = top
                break
        if y is None:
            y = max(0, min(fh - bh, y1 - bh))
        placed.append((x, y, bw, bh))
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, -1)
        cv2.putText(frame, label, (x + 4, y + bh - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def downscale(frame: np.ndarray, target_width: int) -> np.ndarray:
    """Shrink a frame before inference, never enlarge it.

    The largest single lever on weak hardware: cost scales with pixel count,
    and a 1080p frame is nine times the work of 640x360 for no recognition
    benefit once the face crop is upscaled downstream anyway.
    """
    if not target_width or frame.shape[1] <= target_width:
        return frame
    scale = target_width / float(frame.shape[1])
    return cv2.resize(
        frame, (target_width, max(1, int(round(frame.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def letterbox_square(img: np.ndarray, size: int) -> np.ndarray:
    """Resize preserving aspect ratio, pad to a square `size`x`size`."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas
