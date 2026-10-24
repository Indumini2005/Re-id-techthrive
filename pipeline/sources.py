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

Frame selection
---------------
Inference is slower than the cameras, so every cycle several frames arrive
and only one can be processed. Taking the newest is arbitrary: it may be the
one blurred by motion or by the encoder. Instead each reader thread scores
every frame's sharpness as it arrives - off the main loop, a fraction of a
millisecond on a downscaled grey copy - and ``read()`` picks the sharpest of
the frames that arrived since its last decision.

Only those frames are candidates, and none older than ``max_age_ms``, so the
choice never reaches further back than one inference cycle: it trades a
little recency for a frame worth processing, and cannot drift into the past.
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
    sharpness: Optional[float] = None   # see frame_sharpness(); None = unscored

    @property
    def shape(self):
        return self.image.shape


SHARPNESS_WIDTH = 160


def frame_sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian on a small grey copy. Higher = sharper.

    The downscale is not only for speed. Sensor noise also has high Laplacian
    energy and would make a grainy low-light frame look 'sharp'; area
    averaging down to 160 px wide removes most of that noise while keeping
    the edges of anything large enough to matter.

    The value depends on scene content, so it is only meaningful compared
    with other frames from the same camera - which is how read() uses it.
    """
    h, w = image.shape[:2]
    if w > SHARPNESS_WIDTH:
        small = cv2.resize(image, (SHARPNESS_WIDTH, max(1, int(h * SHARPNESS_WIDTH / w))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = image
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    return float(cv2.Laplacian(grey, cv2.CV_32F).var())


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
        score_frames: bool = True,
    ):
        self.url = url
        self.name = name
        self.score_frames = score_frames
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

            # Scored here, on the camera's own thread, so frame selection
            # costs the inference loop nothing.
            sharpness = None
            if self.score_frames:
                try:
                    sharpness = frame_sharpness(image)
                except cv2.error:
                    sharpness = None

            with self._lock:
                self._buffer.append(
                    Frame(image, now, self.frames_read, self.name, sharpness))

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

    @staticmethod
    def closest_in(frames: List[Frame], timestamp: float) -> Optional[Frame]:
        """Same as closest_to, over a snapshot already taken."""
        if not frames:
            return None
        return min(frames, key=lambda f: abs(f.timestamp - timestamp))

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
        frame_select: str = getattr(config, "FRAME_SELECT", "sharpest"),
        max_age_ms: float = getattr(config, "FRAME_SELECT_MAX_AGE_MS", 250.0),
    ):
        if not cameras:
            raise ValueError("no cameras configured")
        if frame_select not in ("sharpest", "newest"):
            raise ValueError("frame_select must be 'sharpest' or 'newest', got %r"
                             % (frame_select,))

        self.frame_select = frame_select
        self.max_age = max_age_ms / 1000.0

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
                score_frames=(frame_select == "sharpest"),
            )

        self.sync_window = sync_window_ms / 1000.0
        # The lead camera anchors each pair; the others are matched against it.
        self.reference = next(iter(self.streams))
        self.require_all = require_all

        self.pairs_emitted = 0
        self.pairs_skewed = 0
        self.frames_skipped = 0
        self.frames_selected_older = 0   # times a sharper, older frame won
        # Newest reference frame seen at the last decision. Every frame after
        # it is a fresh candidate; nothing at or before it is ever reused.
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

    def _choose(self, candidates: List[Frame], others: Dict[str, List[Frame]]):
        """Pick the anchor frame and its partners from this cycle's candidates.

        Returns (anchor, {camera: frame}) with the anchor included.

        'newest' keeps the old behaviour. 'sharpest' ranks each candidate by
        the sharpness of its WEAKER view, because a pair is only as useful as
        the worse of its two frames. Sharpness is normalised within each
        camera first: raw values depend on what the camera is pointed at, so
        a textured scene would otherwise outrank a plain one regardless of
        blur. Pairs inside the sync window always beat pairs outside it.
        """
        def partners(anchor):
            frames = {self.reference: anchor}
            for name, snap in others.items():
                match = CameraStream.closest_in(snap, anchor.timestamp)
                if match is not None:
                    frames[name] = match
            return frames

        newest = candidates[-1]
        scored = [f for f in candidates if f.sharpness is not None]
        if self.frame_select == "newest" or len(candidates) == 1 \
                or len(scored) != len(candidates):
            return newest, partners(newest)

        def scale(frames):
            values = [f.sharpness for f in frames if f.sharpness is not None]
            top = max(values) if values else 0.0
            return top if top > 0 else 1.0

        ref_scale = scale(candidates)
        other_scale = {name: scale(snap) for name, snap in others.items()}

        best = None
        for anchor in candidates:
            frames = partners(anchor)
            stamps = [f.timestamp for f in frames.values()]
            in_sync = (max(stamps) - min(stamps)) <= self.sync_window
            rel = [anchor.sharpness / ref_scale]
            for name, f in frames.items():
                if name == self.reference:
                    continue
                if f.sharpness is None:
                    rel.append(0.0)
                else:
                    rel.append(f.sharpness / other_scale[name])
            # Ties go to the newer frame: candidates are in arrival order.
            key = (in_sync, min(rel), anchor.index)
            if best is None or key > best[0]:
                best = (key, anchor, frames)

        return best[1], best[2]

    def read(self, timeout: float = 5.0) -> Optional[FramePair]:
        """The best pair of frames, one per camera, from this cycle.

        Blocks until the reference camera has produced a frame newer than any
        seen at the last decision. The candidates are every reference frame
        that arrived since then, no older than max_age_ms; with
        frame_select='sharpest' the sharpest pair among them wins, otherwise
        the newest. Returns None on timeout.
        """
        deadline = time.monotonic() + timeout
        reference_stream = self.streams[self.reference]

        while time.monotonic() < deadline:
            latest = reference_stream.latest()
            if latest is None or latest.index <= self._last_reference_index:
                if self.exhausted:
                    return None
                time.sleep(0.002)
                continue

            snapshot = reference_stream.snapshot()
            candidates = [f for f in snapshot
                          if f.index > self._last_reference_index
                          and latest.timestamp - f.timestamp <= self.max_age]
            if not candidates:                  # cannot happen; latest qualifies
                candidates = [latest]

            others = {name: s.snapshot() for name, s in self.streams.items()
                      if name != self.reference}
            anchor, frames = self._choose(candidates, others)

            # Keep retrying against this same set until the other cameras
            # catch up. Consuming it here instead would throw away the lead
            # camera's frames whenever the second stream starts a beat late.
            if self.require_all and len(frames) < len(self.streams):
                time.sleep(0.002)
                continue

            # Only now are the candidates spent. Everything that arrived since
            # the last decision, except the one frame used, was dropped on
            # purpose: inference could not keep up with the camera.
            newest_index = candidates[-1].index
            skipped = newest_index - self._last_reference_index - 1
            if self._last_reference_index >= 0 and skipped > 0:
                self.frames_skipped += skipped
            if anchor.index != newest_index:
                self.frames_selected_older += 1
            self._last_reference_index = newest_index

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
            "frame_select": self.frame_select,
            "sharper_older_chosen": self.frames_selected_older,
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
