"""Multi-camera RTSP capture with temporal pairing.

Each camera gets its own reader thread holding a short ring buffer. The
pairer then hands back one frame per camera, captured close enough together
in time to be treated as the same instant.

Two rules drive the design:

* **Never queue frames.** If inference falls behind a 30 fps camera, the
  right answer is to drop the backlog. A surveillance system that drifts
  five seconds into the past is worse than one that skips frames.
* **Pair before you infer.** Comparing a subject across two views only means
  something if both frames show the same moment. Frames further apart than
  ``sync_window_ms`` are reported as skewed rather than silently fused.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

import config


@dataclass
class Frame:
    """One captured frame plus the metadata the pairer needs."""
    image: np.ndarray
    timestamp: float          # time.monotonic() at capture
    index: int                # per-camera frame counter
    camera: str

    @property
    def shape(self):
        return self.image.shape


@dataclass
class FramePair:
    """One frame per camera, captured at approximately the same instant."""
    frames: Dict[str, Frame]
    skew_ms: float            # spread between the earliest and latest frame
    in_sync: bool
    index: int = 0
    _captured: float = field(default_factory=time.monotonic)

    @property
    def cameras(self) -> List[str]:
        return list(self.frames.keys())

    @property
    def age_ms(self) -> float:
        """How long this pair sat waiting before inference picked it up."""
        return (time.monotonic() - self._captured) * 1000.0

    def __len__(self) -> int:
        return len(self.frames)


def gstreamer_pipeline(
    rtsp_url: str,
    latency: int = config.RTSP_LATENCY_MS,
    codec: str = "h264",
) -> str:
    """Hardware-accelerated decode on the Jetson (nvv4l2decoder), so the CPU
    stays free for the numpy work between stages. With two 1080p30 streams,
    software decode alone would eat most of a core per camera."""
    depay = "rtph265depay ! h265parse" if codec == "h265" else "rtph264depay ! h264parse"
    return (
        f"rtspsrc location={rtsp_url} latency={latency} ! "
        f"{depay} ! nvv4l2decoder ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=1 max-buffers=1 sync=false"
    )


class CameraStream:
    """One RTSP source, read on its own thread into a short ring buffer."""

    def __init__(
        self,
        url: str,
        name: str,
        use_gstreamer: bool = True,
        codec: str = "h264",
        buffer_size: int = 8,
        reconnect_delay: float = 2.0,
        target_fps: Optional[float] = None,
    ):
        self.url = url
        self.name = name
        self.codec = codec
        self.use_gstreamer = use_gstreamer and str(url).startswith("rtsp://")
        self.reconnect_delay = reconnect_delay
        # Pacing only applies to file sources; a live camera sets its own rate.
        self.target_fps = target_fps
        self.is_file = not str(url).startswith("rtsp://") and not str(url).isdigit()

        self._buffer: deque = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap = None

        self.frames_read = 0
        self.connected = False
        self.reconnects = 0
        self._fps_window: deque = deque(maxlen=30)
        self.exhausted = False          # file sources only

    # -- connection --------------------------------------------------------
    def _open(self):
        if self.use_gstreamer:
            cap = cv2.VideoCapture(gstreamer_pipeline(self.url, codec=self.codec), cv2.CAP_GSTREAMER)
            if cap.isOpened():
                return cap
            print(f"[{self.name}] GStreamer failed; falling back to the FFmpeg backend")

        src = int(self.url) if str(self.url).isdigit() else self.url
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _reader(self):
        next_due = 0.0
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                print(f"[{self.name}] connecting to {self.url} ...")
                self._cap = self._open()
                if not self._cap.isOpened():
                    self.connected = False
                    time.sleep(self.reconnect_delay)
                    continue
                self.connected = True
                print(f"[{self.name}] connected")

            ok, image = self._cap.read()
            if not ok:
                if self.is_file:
                    print(f"[{self.name}] end of file")
                    self.exhausted = True
                    self.connected = False
                    return
                print(f"[{self.name}] stream dropped; reconnecting")
                self.connected = False
                self.reconnects += 1
                self._cap.release()
                self._cap = None
                time.sleep(self.reconnect_delay)
                continue

            now = time.monotonic()
            self.frames_read += 1
            self._fps_window.append(now)

            with self._lock:
                self._buffer.append(Frame(image, now, self.frames_read, self.name))

            # Replay files at their real rate so sync behaves like it will live.
            if self.is_file and self.target_fps:
                period = 1.0 / self.target_fps
                next_due = max(next_due + period, now)
                sleep_for = next_due - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)

    # -- control -----------------------------------------------------------
    def start(self) -> "CameraStream":
        self._thread = threading.Thread(target=self._reader, name=f"cam-{self.name}", daemon=True)
        self._thread.start()
        return self

    def release(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()

    # -- access ------------------------------------------------------------
    def latest(self) -> Optional[Frame]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def snapshot(self) -> List[Frame]:
        with self._lock:
            return list(self._buffer)

    def closest_to(self, timestamp: float) -> Optional[Frame]:
        """Buffered frame nearest in time to ``timestamp``."""
        with self._lock:
            if not self._buffer:
                return None
            return min(self._buffer, key=lambda f: abs(f.timestamp - timestamp))

    @property
    def measured_fps(self) -> float:
        if len(self._fps_window) < 2:
            return 0.0
        span = self._fps_window[-1] - self._fps_window[0]
        return (len(self._fps_window) - 1) / span if span > 0 else 0.0


class SyncedCapture:
    """N cameras read in parallel, handed out as temporally paired frames."""

    def __init__(
        self,
        cameras: List[dict],
        sync_window_ms: float = config.SYNC_WINDOW_MS,
        use_gstreamer: bool = True,
        buffer_size: int = 8,
        target_fps: Optional[float] = config.CAPTURE_FPS,
        require_all: bool = True,
    ):
        if not cameras:
            raise ValueError("no cameras configured")

        self.streams: Dict[str, CameraStream] = {}
        for i, cam in enumerate(cameras):
            name = cam.get("name") or f"CAM_{i}"
            self.streams[name] = CameraStream(
                url=cam["url"],
                name=name,
                use_gstreamer=use_gstreamer,
                codec=cam.get("codec", "h264"),
                buffer_size=buffer_size,
                target_fps=target_fps,
            )

        self.sync_window = sync_window_ms / 1000.0
        # The lead camera anchors each pair; the others are matched against it.
        self.reference = next(iter(self.streams))
        self.require_all = require_all

        self.pairs_emitted = 0
        self.pairs_skewed = 0
        self.frames_skipped = 0
        self._last_reference_index = -1

    def start(self) -> "SyncedCapture":
        for stream in self.streams.values():
            stream.start()
        return self

    def release(self):
        for stream in self.streams.values():
            stream.release()

    @property
    def exhausted(self) -> bool:
        """True once every file source has run out (never true for RTSP)."""
        return all(s.exhausted for s in self.streams.values())

    def wait_until_ready(self, timeout: float = 15.0) -> bool:
        """Block until every camera has delivered at least one frame."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(s.latest() is not None for s in self.streams.values()):
                return True
            if self.exhausted:
                return False
            time.sleep(0.05)
        return False

    def read(self, timeout: float = 5.0) -> Optional[FramePair]:
        """Newest pair of frames, one per camera.

        Blocks until the reference camera produces a frame it has not already
        handed out, then matches the other cameras against its timestamp.
        Returns None on timeout.
        """
        deadline = time.monotonic() + timeout
        reference_stream = self.streams[self.reference]

        while time.monotonic() < deadline:
            anchor = reference_stream.latest()
            if anchor is None or anchor.index == self._last_reference_index:
                if self.exhausted:
                    return None
                time.sleep(0.002)
                continue

            frames = {self.reference: anchor}
            for name, stream in self.streams.items():
                if name == self.reference:
                    continue
                match = stream.closest_to(anchor.timestamp)
                if match is not None:
                    frames[name] = match

            # Keep retrying against this same anchor until the other cameras
            # catch up. Consuming it here instead would throw away the lead
            # camera's frames whenever the second stream starts a beat late.
            if self.require_all and len(frames) < len(self.streams):
                time.sleep(0.002)
                continue

            # Only now is the anchor spent. Everything between the previously
            # emitted frame and this one was dropped on purpose: inference
            # could not keep up with the camera.
            skipped = anchor.index - self._last_reference_index - 1
            if self._last_reference_index >= 0 and skipped > 0:
                self.frames_skipped += skipped
            self._last_reference_index = anchor.index

            timestamps = [f.timestamp for f in frames.values()]
            skew_ms = (max(timestamps) - min(timestamps)) * 1000.0
            in_sync = skew_ms <= self.sync_window * 1000.0
            if not in_sync:
                self.pairs_skewed += 1

            self.pairs_emitted += 1
            return FramePair(
                frames=frames, skew_ms=skew_ms, in_sync=in_sync, index=self.pairs_emitted
            )

        return None

    def stats(self) -> dict:
        return {
            "cameras": len(self.streams),
            "pairs": self.pairs_emitted,
            "skewed": self.pairs_skewed,
            "skipped_frames": self.frames_skipped,
            "capture_fps": {n: round(s.measured_fps, 1) for n, s in self.streams.items()},
            "reconnects": {n: s.reconnects for n, s in self.streams.items()},
        }


def cameras_from_args(sources: Optional[List[str]], codec: str = "h264") -> List[dict]:
    """Builds the camera list from ``--source`` arguments, falling back to
    ``config.CAMERAS``. Accepts ``NAME=url`` to label a stream."""
    if not sources:
        return list(config.CAMERAS)

    cameras = []
    for i, raw in enumerate(sources):
        if "=" in raw and not raw.split("=", 1)[0].startswith(("rtsp", "http", "/")):
            name, url = raw.split("=", 1)
        else:
            name, url = f"CAM_{chr(ord('A') + i)}", raw
        cameras.append({"name": name, "url": url, "codec": codec})
    return cameras
