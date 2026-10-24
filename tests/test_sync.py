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


def push(cap, name, ts, index, sharpness=None):
    s = cap.streams[name]
    with s._lock:
        s._buffer.append(Frame(IMG, ts, index, name, sharpness))
    s.frames_read = index


def fresh(window_ms=50.0, select="sharpest"):
    return SyncedCapture(CAMS, sync_window_ms=window_ms, use_gstreamer=False,
                         frame_select=select)


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


# ===========================================================================
# Frame selection
# ===========================================================================
def burst(cap, start_ts, start_idx, a_sharp, b_sharp, step=0.033):
    """Push matching A/B frames, as if they arrived during one busy cycle."""
    for i, (sa, sb) in enumerate(zip(a_sharp, b_sharp)):
        ts = start_ts + i * step
        push(cap, "CAM_A", ts, start_idx + i, sa)
        push(cap, "CAM_B", ts + 0.002, start_idx + i, sb)


def primed(select="sharpest", window_ms=50.0):
    """A capture that has already made one decision at index 1."""
    cap = fresh(window_ms, select)
    push(cap, "CAM_A", 600.0, 1, 50.0)
    push(cap, "CAM_B", 600.0, 1, 50.0)
    cap.read(timeout=0.5)
    return cap


# --- 9. the sharpest frame of the cycle wins, not the newest ---------------
cap = primed()
burst(cap, 600.1, 2, [40, 90, 55, 30], [40, 90, 55, 30])     # #3 is sharpest
pair = cap.read(timeout=0.5)
assert pair.frames["CAM_A"].index == 3, pair.frames["CAM_A"].index
assert pair.frames["CAM_B"].index == 3, pair.frames["CAM_B"].index
assert cap.frames_skipped == 3, cap.frames_skipped          # 4 arrived, 1 used
assert cap.frames_selected_older == 1
print("sharpest    : chose frame #3 of 2-5 (sharpness 90), 3 counted skipped")

# --- 10. a pair is ranked by its WEAKER view -------------------------------
# #2 has the sharpest A frame but a smeared B frame; #4 is good in both.
cap = primed()
burst(cap, 600.1, 2, [100, 60, 85], [20, 60, 80])
pair = cap.read(timeout=0.5)
assert pair.frames["CAM_A"].index == 4, pair.frames["CAM_A"].index
print("weaker view : chose #4 (85/80) over #2 (100/20) - a pair is only as good "
      "as its worse frame")

# --- 11. sharpness is compared within a camera, not across ----------------
# B is pointed at a busier scene, so its raw numbers are 100x larger. Without
# per-camera normalisation B would decide everything and A's blur would be
# invisible. #3 is sharpest in A and only slightly weaker in B.
cap = primed()
burst(cap, 600.1, 2, [30, 95, 50], [10000, 9500, 9900])
pair = cap.read(timeout=0.5)
assert pair.frames["CAM_A"].index == 3, pair.frames["CAM_A"].index
print("per-camera  : B's 100x larger raw values did not swamp A's blur")

# --- 12. a frame passed over is never reconsidered -------------------------
# After choosing #3 from 2-5, frames 4 and 5 are spent: the next decision
# must look only at what arrives afterwards, or the stream drifts backwards.
cap = primed()
burst(cap, 600.1, 2, [40, 90, 55, 30], [40, 90, 55, 30])
cap.read(timeout=0.5)                                        # takes #3
assert cap.read(timeout=0.2) is None, "reused a frame from the spent cycle"
push(cap, "CAM_A", 600.4, 6, 20.0)
push(cap, "CAM_B", 600.4, 6, 20.0)
pair = cap.read(timeout=0.5)
assert pair.frames["CAM_A"].index == 6, pair.frames["CAM_A"].index
print("no reuse    : frames 4-5 passed over stay passed over; next pick is #6")

# --- 13. nothing older than max_age is a candidate -------------------------
cap = primed()
push(cap, "CAM_A", 601.0, 2, 999.0)       # very sharp, but 400 ms before #3
push(cap, "CAM_B", 601.0, 2, 999.0)
push(cap, "CAM_A", 601.4, 3, 10.0)
push(cap, "CAM_B", 601.4, 3, 10.0)
pair = cap.read(timeout=0.5)
assert pair.frames["CAM_A"].index == 3, pair.frames["CAM_A"].index
print("max age     : a sharp frame 400 ms old lost to a fresh blurry one "
      "(limit %.0f ms)" % (cap.max_age * 1000))

# --- 14. an in-sync pair beats a sharper out-of-sync one -------------------
cap = primed(window_ms=50.0)
push(cap, "CAM_A", 602.00, 2, 99.0)       # B's nearest frame is 120 ms away
push(cap, "CAM_A", 602.12, 3, 40.0)
push(cap, "CAM_B", 602.12, 3, 40.0)
pair = cap.read(timeout=0.5)
assert pair.frames["CAM_A"].index == 3 and pair.in_sync, (pair.frames, pair.skew_ms)
print("sync first  : sharper frame #2 rejected, its partner was 120 ms away")

# --- 15. 'newest' mode ignores sharpness -----------------------------------
cap = primed(select="newest")
burst(cap, 600.1, 2, [40, 90, 55, 30], [40, 90, 55, 30])
pair = cap.read(timeout=0.5)
assert pair.frames["CAM_A"].index == 5, pair.frames["CAM_A"].index
assert cap.frames_selected_older == 0
print("newest mode : took #5 regardless of sharpness")

# --- 16. unscored frames fall back to newest -------------------------------
cap = primed()
burst(cap, 600.1, 2, [None, None, None], [None, None, None])
pair = cap.read(timeout=0.5)
assert pair.frames["CAM_A"].index == 4, pair.frames["CAM_A"].index
print("unscored    : no sharpness available -> newest, as before")

print("\nALL PASS")
