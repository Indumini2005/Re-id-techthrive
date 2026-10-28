"""Kalman person tracker: prediction, stable IDs, gaps, crossings.

The claims worth testing are the ones the dashboard shows: that the
1-second-ahead prediction lands where the person actually goes, and that a
person keeps one ID through what the Nano really produces - about three
detections a second, jittery boxes, and the odd missed frame.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.tracker import MultiTracker, iou  # noqa: E402

failures = []
rng = np.random.default_rng(11)


def check(name, cond, detail=""):
    print(("OK   " if cond else "FAIL ") + name + ("   " + detail if detail else ""))
    if not cond:
        failures.append(name)


def person(cx, cy, h=200.0, jitter=0.0):
    w = 0.4 * h
    j = rng.normal(0, jitter, 4) if jitter else np.zeros(4)
    return (cx - w / 2 + j[0], cy - h / 2 + j[1], cx + w / 2 + j[2], cy + h / 2 + j[3])


DT = 1.0 / 3.0          # the Nano's measured ~3 frame pairs per second

# --- 1. constant walk: velocity learned, 1 s prediction accurate -----------
trk = MultiTracker("CAM_A")
speed = 90.0            # px/s, about half a body height per second
ids = set()
for k in range(10):
    t = k * DT
    out = trk.update([person(100 + speed * t, 300, jitter=3.0)], t)
    ids.add(out[0].label)
track = out[0]
vx, vy = track.velocity
check("one ID for one walking person", ids == {"A1"}, str(sorted(ids)))
check("velocity learned", abs(vx - speed) < 20 and abs(vy) < 20, "vx %.1f vy %.1f" % (vx, vy))

t_now = 9 * DT
pred = track.box_at(t_now + 1.0)
truth = person(100 + speed * (t_now + 1.0), 300)
err = abs((pred[0] + pred[2]) / 2 - (truth[0] + truth[2]) / 2)
check("1 s-ahead prediction lands on the person", err < 25, "error %.1f px" % err)
check("box_at does not move the track", abs(track.current_box[0] - track.box_at(t_now)[0]) < 1e-6)

# --- 2. fast mover at low fps: no overlap between frames, same ID -----------
trk = MultiTracker("CAM_B")
labels = []
for k in range(8):
    t = k * DT
    # 300 px/s with an 80 px-wide box: consecutive boxes barely overlap.
    labels.append(trk.update([person(80 + 300 * t, 300)], t)[0].label)
check("fast walker keeps one ID", len(set(labels)) == 1, str(labels))
check("label prefix follows the camera", labels[0].startswith("B"), labels[0])
check("first IoU between frames is small",
      iou(person(80, 300), person(80 + 300 * DT, 300)) < 0.35)

# --- 3. missed detections: coast, then re-acquire ---------------------------
trk = MultiTracker("CAM_A", max_age=2.0)
for k in range(6):
    first = trk.update([person(100 + 60 * k * DT, 300)], k * DT)[0]
gap_t = 5 * DT + 1.0                       # one second with no detection
trk.update([], 5 * DT + 0.5)               # an empty frame in between
back = trk.update([person(100 + 60 * gap_t, 300)], gap_t)[0]
check("re-acquired after a 1 s gap with the same ID", back is first,
      "%s vs %s" % (back.label, first.label))

later = gap_t + 3.0                        # past max_age
trk.update([], later)
fresh = trk.update([person(100 + 60 * gap_t, 300)], later + 0.1)[0]
check("after max_age a new ID is issued", fresh is not first, fresh.label)

# --- 4. two people walking past each other ----------------------------------
trk = MultiTracker("CAM_A")
a_ids, b_ids, ambiguous_seen = set(), set(), False
for k in range(12):
    t = k * DT
    a = person(100 + 120 * t, 300)          # left to right
    b = person(600 - 120 * t, 330)          # right to left, slightly lower
    out = trk.update([a, b], t)
    a_ids.add(out[0].label)
    b_ids.add(out[1].label)
    ambiguous_seen = ambiguous_seen or out[0].ambiguous
check("crossing walkers keep their own IDs", len(a_ids) == 1 and len(b_ids) == 1
      and a_ids != b_ids, "A %s  B %s" % (sorted(a_ids), sorted(b_ids)))
check("overlap flagged as ambiguous while they pass", ambiguous_seen)

# --- 5. dwell and confirmation ------------------------------------------------
trk = MultiTracker("CAM_A")
for k in range(4):
    out = trk.update([person(300, 300)], 10.0 + k)
check("dwell measured from first sighting", abs(out[0].dwell(13.0) - 3.0) < 1e-9)
check("confirmed after repeated hits", out[0].confirmed)
check("a single sighting is tentative",
      not MultiTracker("X").update([person(1, 1)], 0)[0].confirmed)

# --- 6. time never runs backwards ---------------------------------------------
trk = MultiTracker("CAM_A")
o = trk.update([person(100, 300)], 5.0)[0]
before = o.current_box
o.predict(4.0)
check("predicting into the past is a no-op", o.current_box == before)

print("\n" + ("ALL PASS" if not failures else "%d FAILURE(S): %s" % (len(failures), failures)))
sys.exit(1 if failures else 0)
