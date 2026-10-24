"""Frame pairing logic in SyncedCapture, driven by injected frames so no
camera or cv2 backend is needed (reader threads are never started)."""

import sys
import types
from pathlib import Path

import numpy as np

sys.modules.setdefault("cv2", types.ModuleType("cv2"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.sources import Frame, SyncedCapture, cameras_from_args  # noqa: E402

CAMS = [{"name": "CAM_A", "url": "rtsp://a"}, {"name": "CAM_B", "url": "rtsp://b"}]
IMG = np.zeros((4, 4, 3), np.uint8)


def push(cap, name, ts, index):
    s = cap.streams[name]
    with s._lock:
        s._buffer.append(Frame(IMG, ts, index, name))
    s.frames_read = index


def fresh(window_ms=50.0):
    return SyncedCapture(CAMS, sync_window_ms=window_ms, use_gstreamer=False)


# --- 1. frames 10ms apart pair cleanly -------------------------------------
cap = fresh()
push(cap, "CAM_A", 100.000, 1)
push(cap, "CAM_B", 100.010, 1)
pair = cap.read(timeout=0.5)
assert pair is not None and len(pair) == 2, pair
assert abs(pair.skew_ms - 10.0) < 0.001, pair.skew_ms
assert pair.in_sync
print(f"10ms apart  : paired, skew {pair.skew_ms:.1f}ms, in_sync={pair.in_sync}")

# --- 2. the same reference frame is never handed out twice -----------------
assert cap.read(timeout=0.2) is None, "re-emitted a pair with no new reference frame"
print("no new frame: read() correctly returns None")

# --- 3. 120ms apart pairs but is flagged as skewed -------------------------
push(cap, "CAM_A", 101.000, 2)
push(cap, "CAM_B", 101.120, 2)
pair = cap.read(timeout=0.5)
assert pair is not None and not pair.in_sync, pair
assert abs(pair.skew_ms - 120.0) < 0.001, pair.skew_ms
assert cap.pairs_skewed == 1, cap.pairs_skewed
print(f"120ms apart : paired but flagged, skew {pair.skew_ms:.1f}ms")

# --- 4. the temporally CLOSEST frame is chosen, not just the newest --------
cap = fresh()
for i, ts in enumerate([200.00, 200.03, 200.06, 200.09], start=1):
    push(cap, "CAM_B", ts, i)
push(cap, "CAM_A", 200.031, 1)          # anchor sits next to CAM_B frame #2
pair = cap.read(timeout=0.5)
assert pair.frames["CAM_B"].index == 2, pair.frames["CAM_B"].index
assert pair.skew_ms < 2.0, pair.skew_ms
print(f"closest pick: chose CAM_B frame #{pair.frames['CAM_B'].index} "
      f"of 4 buffered (skew {pair.skew_ms:.1f}ms)")

# --- 5. skipped frames are counted when inference falls behind -------------
cap = fresh()
push(cap, "CAM_A", 300.0, 1)
push(cap, "CAM_B", 300.0, 1)
cap.read(timeout=0.5)
assert cap.frames_skipped == 0, cap.frames_skipped
# Camera delivered frames 2..10 while inference was busy; only 10 is used.
push(cap, "CAM_A", 300.3, 10)
push(cap, "CAM_B", 300.3, 10)
pair = cap.read(timeout=0.5)
assert pair.frames["CAM_A"].index == 10, pair.frames["CAM_A"].index
assert cap.frames_skipped == 8, f"expected 8 skipped, got {cap.frames_skipped}"
print(f"backlog     : used frame #10, counted {cap.frames_skipped} skipped")

# --- 6. require_all waits for the second camera ----------------------------
cap = fresh()
push(cap, "CAM_A", 400.0, 1)
assert cap.read(timeout=0.3) is None, "emitted a pair with only one camera"
push(cap, "CAM_B", 400.0, 1)
assert cap.read(timeout=0.5) is not None
print("require_all : waits for the second camera, then pairs")

# --- 7. allow_single_camera emits anyway ----------------------------------
cap = SyncedCapture(CAMS, use_gstreamer=False, require_all=False)
push(cap, "CAM_A", 500.0, 1)
pair = cap.read(timeout=0.5)
assert pair is not None and len(pair) == 1 and pair.cameras == ["CAM_A"], pair
assert pair.skew_ms == 0.0 and pair.in_sync
print("single cam  : emitted with 1 view when require_all=False")

# --- 8. --source parsing ---------------------------------------------------
cams = cameras_from_args(["rtsp://x/1", "rtsp://y/2"])
assert [c["name"] for c in cams] == ["CAM_A", "CAM_B"], cams
cams = cameras_from_args(["LOBBY=rtsp://x/1", "GATE=b.mp4"])
assert [c["name"] for c in cams] == ["LOBBY", "GATE"], cams
assert cams[0]["url"] == "rtsp://x/1" and cams[1]["url"] == "b.mp4"
cams = cameras_from_args(["rtsp://user:pw@h/s"])      # '=' absent, ':' present
assert cams[0]["name"] == "CAM_A" and cams[0]["url"] == "rtsp://user:pw@h/s", cams
print("arg parsing : positional, NAME=url, and credential URLs all OK")

print("\nALL PASS")
