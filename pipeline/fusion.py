"""Cross-view association and multi-angle identification.

With two cameras watching the same area, each frame pair gives two
independent looks at a subject. This module does two separate things with
that, and it is worth keeping them distinct:

1. **Association** - decide whether the person camera A sees *is* the person
   camera B sees, by comparing their live embeddings to each other. This
   needs no gallery and no enrollment; it is pure cross-view comparison.

2. **Identification** - having decided they are the same person, fuse the two
   embeddings into one query vector and search the gallery once. Averaging
   two viewing angles cancels a good deal of the pose and lighting error that
   sinks single-view CCTV re-ID.

Thresholds differ between the two and must not be shared. Association
compares two *degraded* live faces, so it sits lower (~0.45). Identification
compares a fused live vector against a clean gallery vector, so it keeps the
usual ~0.65.
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np

import config
from .stage5_match import Match
from .vectors import l2_normalize


@dataclass
class ViewObservation:
    """One subject as seen by one camera in one frame."""
    camera: str
    box: tuple
    conf: float
    crop: np.ndarray
    face: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    # True when GFPGAN was invoked because the cheap pass looked promising
    # (``--restore_mode escalate``). Worth logging: it is the expensive path.
    escalated: bool = False
    # Filled in by identify(); the per-view gallery hit, for diagnostics.
    solo_match: Optional[Match] = None

    @property
    def usable(self) -> bool:
        return self.embedding is not None


@dataclass
class FusedTrack:
    """One physical person, as reconstructed from one or more views."""
    observations: List[ViewObservation]
    cross_view_score: float = 0.0      # mean pairwise similarity between views
    fused_embedding: Optional[np.ndarray] = None
    match: Optional[Match] = None      # gallery hit for the fused vector
    agreement: bool = False            # every view independently agrees
    notes: List[str] = field(default_factory=list)

    @property
    def cameras(self) -> List[str]:
        return [o.camera for o in self.observations]

    @property
    def n_views(self) -> int:
        return len(self.observations)

    @property
    def is_multiview(self) -> bool:
        return self.n_views > 1

    def observation_for(self, camera: str) -> Optional[ViewObservation]:
        for o in self.observations:
            if o.camera == camera:
                return o
        return None

    @property
    def best_face(self) -> Optional[np.ndarray]:
        """The restored face from the highest-confidence view, for the
        operator verification window."""
        with_faces = [o for o in self.observations if o.face is not None]
        if not with_faces:
            return None
        return max(with_faces, key=lambda o: o.conf).face

    def label(self) -> str:
        if self.match and self.match.is_match:
            tag = f"{self.match.identity} {self.match.score:.2f}"
        elif self.match:
            tag = f"unknown {self.match.score:.2f}"
        else:
            tag = "no embedding"
        if self.is_multiview:
            tag += f" [{self.n_views}v {self.cross_view_score:.2f}]"
        return tag


def fuse_embeddings(embeddings: List[np.ndarray], mode: str = "mean") -> np.ndarray:
    """Combine several unit vectors into one query vector.

    ``mean``   - average then renormalize. The right default: it is the
                 maximum-likelihood direction for vectors scattered around a
                 shared true identity, so extra angles strictly help.
    ``max``    - element-wise max. Keeps the strongest response per dimension;
                 occasionally better when one view is badly occluded, but
                 noisier in general.
    ``first``  - ignore fusion, use the leading view (for ablation runs).
    """
    stack = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
    if stack.shape[0] == 1 or mode == "first":
        return l2_normalize(stack[0])
    if mode == "max":
        return l2_normalize(stack.max(axis=0))
    if mode == "mean":
        return l2_normalize(stack.mean(axis=0))
    raise ValueError(f"unknown fusion mode '{mode}' (mean | max | first)")


class CrossViewFusion:
    """Associates subjects across cameras, then identifies the fused track."""

    def __init__(
        self,
        index=None,
        cross_view_threshold: float = config.CROSS_VIEW_THRESHOLD,
        gallery_threshold: float = config.MATCH_THRESHOLD,
        fusion_mode: str = config.FUSION_MODE,
        require_agreement: bool = config.REQUIRE_VIEW_AGREEMENT,
        require_both_views: bool = config.REQUIRE_BOTH_VIEWS,
    ):
        self.index = index
        self.cross_view_threshold = cross_view_threshold
        self.gallery_threshold = gallery_threshold
        self.fusion_mode = fusion_mode
        self.require_agreement = require_agreement
        self.require_both_views = require_both_views

    # -- 1. association ----------------------------------------------------
    def associate(self, by_camera: Dict[str, List[ViewObservation]]) -> List[FusedTrack]:
        """Group observations that appear to be the same person.

        Greedy highest-similarity-first matching. With the handful of subjects
        a single scene holds, this lands on the same answer as the Hungarian
        algorithm without pulling in scipy.
        """
        usable = {cam: [o for o in obs if o.usable] for cam, obs in by_camera.items()}
        usable = {cam: obs for cam, obs in usable.items() if obs}

        if not usable:
            return []
        if len(usable) == 1:
            return [FusedTrack(observations=[o]) for o in next(iter(usable.values()))]

        # Score every cross-camera pair.
        candidates = []
        for cam_a, cam_b in combinations(usable.keys(), 2):
            for i, obs_a in enumerate(usable[cam_a]):
                for j, obs_b in enumerate(usable[cam_b]):
                    score = float(np.dot(obs_a.embedding, obs_b.embedding))
                    if score >= self.cross_view_threshold:
                        candidates.append((score, (cam_a, i), (cam_b, j)))
        candidates.sort(key=lambda c: c[0], reverse=True)

        # Union-find over (camera, index) keys, refusing to merge two
        # observations from the same camera into one track.
        parent: Dict[tuple, tuple] = {}
        members: Dict[tuple, List[tuple]] = {}
        scores: Dict[tuple, List[float]] = {}

        def find(key):
            parent.setdefault(key, key)
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        for cam, obs_list in usable.items():
            for i in range(len(obs_list)):
                key = (cam, i)
                parent[key] = key
                members[key] = [key]
                scores[key] = []

        for score, key_a, key_b in candidates:
            root_a, root_b = find(key_a), find(key_b)
            if root_a == root_b:
                continue
            cams_a = {k[0] for k in members[root_a]}
            cams_b = {k[0] for k in members[root_b]}
            if cams_a & cams_b:                    # same camera twice -> not one person
                continue
            parent[root_b] = root_a
            members[root_a].extend(members[root_b])
            scores[root_a] = scores[root_a] + scores[root_b] + [score]
            del members[root_b]

        tracks = []
        for root, keys in members.items():
            observations = [usable[cam][i] for cam, i in keys]
            observations.sort(key=lambda o: o.camera)
            pair_scores = scores.get(root, [])
            tracks.append(
                FusedTrack(
                    observations=observations,
                    cross_view_score=float(np.mean(pair_scores)) if pair_scores else 0.0,
                )
            )

        # Largest tracks first, so multi-view hits are handled before singles.
        tracks.sort(key=lambda t: (t.n_views, t.cross_view_score), reverse=True)
        return tracks

    # -- 2. identification -------------------------------------------------
    def identify(self, tracks: List[FusedTrack]) -> List[FusedTrack]:
        """Fill in ``fused_embedding``, ``match`` and ``agreement``."""
        if self.index is None:
            return tracks

        for track in tracks:
            embeddings = [o.embedding for o in track.observations if o.usable]
            if not embeddings:
                continue

            if self.require_both_views and not track.is_multiview:
                track.notes.append("single view; suppressed by --require_both_views")
                track.match = Match(False, "Unknown", 0.0)
                continue

            track.fused_embedding = fuse_embeddings(embeddings, self.fusion_mode)
            track.match = self.index.query(track.fused_embedding, self.gallery_threshold)

            # Per-view hits: cheap, and they are what tells you whether fusion
            # actually helped or whether one view is carrying the result.
            for obs in track.observations:
                if obs.usable:
                    obs.solo_match = self.index.query(obs.embedding, self.gallery_threshold)

            solo = [o.solo_match for o in track.observations if o.solo_match]
            track.agreement = bool(
                track.match
                and track.match.is_match
                and solo
                and all(m.is_match and m.identity == track.match.identity for m in solo)
            )

            if self.require_agreement and track.is_multiview and not track.agreement:
                disagreeing = {m.identity for m in solo}
                track.notes.append(
                    f"views disagree ({', '.join(sorted(disagreeing))}); suppressed"
                )
                track.match = Match(False, "Unknown", track.match.score if track.match else 0.0)

            if track.is_multiview and track.match and solo:
                best_solo = max(m.score for m in solo)
                gain = track.match.score - best_solo
                track.notes.append(f"fusion {gain:+.3f} vs best single view")

        return tracks

    def process(self, by_camera: Dict[str, List[ViewObservation]]) -> List[FusedTrack]:
        """Association followed by identification."""
        return self.identify(self.associate(by_camera))
