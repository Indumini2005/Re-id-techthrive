"""Per-camera person tracking with a constant-velocity Kalman filter.

Detection alone says "there are three people in this frame". Tracking says
"this is the same person as 0.3 s ago", which everything else needs:

* **Prediction between frames.** The Nano processes about three frame pairs
  a second, so a walking person moves a long way between detections. Each
  track carries a Kalman estimate of position *and velocity*, so the system
  knows where someone will be before the next detection arrives - for
  matching them to it, and for drawing where they are heading.
* **Stable IDs.** A person keeps the label ``A3`` while they stay in view,
  which is what dwell time and loitering are measured on.
* **Track-then-recognise.** Once a track's identity is settled, the face
  does not need re-embedding every cycle. ArcFace is the most expensive
  stage on the Nano, so skipping it for people already known is the
  largest single speed-up available (see SceneAnalytics.wants_face).

State per track: centre x/y, width, height, and the rate of change of each,
in pixels and pixels per second. Time steps are real seconds, not frame
counts, because cycles are irregular on a loaded board.

Association is greedy best-first on a score mixing overlap (IoU) with
centre distance relative to body height. Pure IoU fails at low frame rates:
a person who moved half a body-width since the last detection no longer
overlaps their own predicted box. With a handful of people per scene,
greedy matching gives the same answer as the Hungarian algorithm without
pulling in scipy.

Pure numpy, no OpenCV: the hub runs this too.
"""

import math
from collections import deque

import numpy as np

import config


def box_to_state(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0,
                     max(x2 - x1, 1.0), max(y2 - y1, 1.0)])


def state_to_box(z):
    cx, cy, w, h = [float(v) for v in z[:4]]
    w, h = max(w, 1.0), max(h, 1.0)
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def _transition(dt):
    F = np.eye(8)
    F[0:4, 4:8] = np.eye(4) * dt
    return F


class KalmanTrack:
    """One person in one camera."""

    # Noise, as fractions of the person's height in pixels. Scaling by
    # height makes the filter behave the same for a person 80 px tall at the
    # back of the room and one 300 px tall at the front.
    ACCEL_POS = 0.6      # px/s^2 per px of height: walking starts and stops
    ACCEL_SIZE = 0.15    # boxes change size slowly
    MEAS_POS = 0.05      # detector jitter on the centre
    MEAS_SIZE = 0.08     # and on the box size
    INIT_VEL = 1.0       # unknown speed at birth: up to ~1 body height / s
    MAX_SPEED = 3.0      # clamp: nobody crosses three body heights a second

    def __init__(self, track_id, label, box, t, conf=0.0):
        self.id = track_id
        self.label = label
        z = box_to_state(box)
        self.x = np.concatenate([z, np.zeros(4)])
        h = z[3]
        meas = np.array([self.MEAS_POS, self.MEAS_POS, self.MEAS_SIZE, self.MEAS_SIZE]) * h
        vel = np.full(4, self.INIT_VEL * h)
        self.P = np.diag(np.concatenate([meas, vel]) ** 2)
        self.t = float(t)

        self.box = tuple(float(v) for v in box)     # last measured box
        self.conf = float(conf)
        self.first_seen = float(t)
        self.last_update = float(t)
        self.hits = 1
        self.trail = deque(maxlen=24)
        self.trail.append((float(t), float(z[0]), float(z[1])))
        self.ambiguous = False

        # Recognition state, written by SceneAnalytics. Kept on the track
        # because it lives and dies with it.
        self.embedding = None
        self.embedding_t = None
        self.n_embeddings = 0
        self.face_thumb = None
        self.next_face_try = float(t)
        self.decided_t = None
        self.identity = None          # dict: key, name, kind, score, confidence
        self.loiter_flagged = False

    # -- Kalman ------------------------------------------------------------
    def _height(self):
        return max(float(self.x[3]), 10.0)

    def _process_noise(self, dt):
        h = self._height()
        q = np.zeros((8, 8))
        for i, accel in enumerate((self.ACCEL_POS, self.ACCEL_POS,
                                   self.ACCEL_SIZE, self.ACCEL_SIZE)):
            s2 = (accel * h) ** 2
            q[i, i] = s2 * dt ** 3 / 3.0
            q[i, i + 4] = q[i + 4, i] = s2 * dt ** 2 / 2.0
            q[i + 4, i + 4] = s2 * dt
        return q

    def predict(self, t):
        """Advance the state to time ``t``. Never runs backwards."""
        dt = float(t) - self.t
        if dt <= 0:
            return
        F = _transition(dt)
        self.x = F.dot(self.x)
        self.P = F.dot(self.P).dot(F.T) + self._process_noise(dt)
        self.t = float(t)

    def update(self, box, t, conf=0.0):
        self.predict(t)
        z = box_to_state(box)
        h = self._height()
        H = np.zeros((4, 8))
        H[0:4, 0:4] = np.eye(4)
        R = np.diag((np.array([self.MEAS_POS, self.MEAS_POS,
                               self.MEAS_SIZE, self.MEAS_SIZE]) * h) ** 2)
        y = z - H.dot(self.x)
        S = H.dot(self.P).dot(H.T) + R
        K = np.linalg.solve(S, H.dot(self.P)).T
        self.x = self.x + K.dot(y)
        self.P = (np.eye(8) - K.dot(H)).dot(self.P)

        # Keep velocities physically plausible; one bad detection should not
        # fling the prediction across the frame.
        limit = self.MAX_SPEED * self._height()
        self.x[4:8] = np.clip(self.x[4:8], -limit, limit)

        self.box = tuple(float(v) for v in box)
        self.conf = float(conf)
        self.last_update = float(t)
        self.hits += 1
        self.trail.append((float(t), float(self.x[0]), float(self.x[1])))

    # -- read-outs ---------------------------------------------------------
    def box_at(self, t):
        """Predicted box at time ``t`` without changing the track."""
        dt = max(0.0, float(t) - self.t)
        return state_to_box(_transition(dt).dot(self.x))

    @property
    def current_box(self):
        return state_to_box(self.x)

    @property
    def velocity(self):
        return float(self.x[4]), float(self.x[5])

    @property
    def confirmed(self):
        return self.hits >= getattr(config, "TRACK_MIN_HITS", 2)

    def coasting(self, t):
        return float(t) - self.last_update > 1e-6

    def dwell(self, t):
        return max(0.0, float(t) - self.first_seen)


def _prefix(camera):
    """CAM_A -> A, LOBBY -> L. Short labels for the screen."""
    tail = str(camera).replace("-", "_").split("_")[-1] or str(camera)
    return tail[0].upper() if tail else "T"


class MultiTracker:
    """All tracks for one camera."""

    def __init__(self, camera, max_age=None, iou_min=0.1, dist_max=0.7,
                 ambiguity_iou=0.3):
        self.camera = camera
        self.prefix = _prefix(camera)
        self.max_age = float(max_age if max_age is not None
                             else getattr(config, "TRACK_MAX_AGE_S", 2.0))
        self.iou_min = iou_min
        self.dist_max = dist_max
        self.ambiguity_iou = ambiguity_iou
        self.tracks = []
        self._next = 1
        self.last_t = None

    def update(self, boxes, t, confs=None):
        """Match this frame's detections to tracks. Returns one track per box,
        in the same order."""
        t = float(t)
        boxes = [tuple(float(v) for v in b) for b in boxes]
        confs = list(confs) if confs is not None else [0.0] * len(boxes)
        self.last_t = t

        for trk in self.tracks:
            trk.predict(t)
            trk.ambiguous = False

        # Score every plausible (track, detection) pair.
        pairs = []
        for ti, trk in enumerate(self.tracks):
            pb = trk.current_box
            ph = max(pb[3] - pb[1], 1.0)
            pcx, pcy = (pb[0] + pb[2]) / 2.0, (pb[1] + pb[3]) / 2.0
            for di, db in enumerate(boxes):
                overlap = iou(pb, db)
                dcx, dcy = (db[0] + db[2]) / 2.0, (db[1] + db[3]) / 2.0
                dist = math.hypot(dcx - pcx, dcy - pcy) / ph
                if overlap >= self.iou_min or dist <= self.dist_max:
                    pairs.append((overlap - 0.5 * dist, ti, di))
        pairs.sort(key=lambda p: p[0], reverse=True)

        assigned = [None] * len(boxes)
        used = set()
        for _, ti, di in pairs:
            if ti in used or assigned[di] is not None:
                continue
            used.add(ti)
            trk = self.tracks[ti]
            trk.update(boxes[di], t, confs[di])
            assigned[di] = trk

        for di, box in enumerate(boxes):
            if assigned[di] is None:
                trk = KalmanTrack("%s:%d" % (self.camera, self._next),
                                  "%s%d" % (self.prefix, self._next), box, t, confs[di])
                self._next += 1
                self.tracks.append(trk)
                assigned[di] = trk

        # Two people overlapping: their identities could swap here, so both
        # tracks must be re-recognised rather than trusting the cache.
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if iou(boxes[i], boxes[j]) >= self.ambiguity_iou:
                    assigned[i].ambiguous = True
                    assigned[j].ambiguous = True

        self.tracks = [trk for trk in self.tracks if t - trk.last_update <= self.max_age]
        return assigned

    def prune(self, t):
        """Drop tracks that have coasted too long, without new detections."""
        self.tracks = [trk for trk in self.tracks if float(t) - trk.last_update <= self.max_age]

    def get(self, track_id):
        for trk in self.tracks:
            if trk.id == track_id:
                return trk
        return None
