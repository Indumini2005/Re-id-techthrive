"""Cross-view association + fused identification, on the numpy backend.
cv2/onnxruntime are stubbed; neither is used by the code under test."""

import sys
import types
from pathlib import Path

import numpy as np

for name in ("cv2", "onnxruntime"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TMP = Path(__file__).parent / "fuse_test"
TMP.mkdir(exist_ok=True)

from pipeline.fusion import CrossViewFusion, ViewObservation, fuse_embeddings  # noqa: E402
from pipeline.stage4_embed import l2_normalize  # noqa: E402
from pipeline.stage5_match import ThreatIndex  # noqa: E402

rng = np.random.default_rng(11)
D = 512


def unit(v):
    return l2_normalize(np.asarray(v, dtype=np.float32))


def view(cam, emb, conf=0.9):
    return ViewObservation(camera=cam, box=(0, 0, 10, 20), conf=conf,
                           crop=np.zeros((2, 2, 3), np.uint8), embedding=unit(emb))


# Two true identities. Each camera sees a noisy, angle-shifted version.
true_a = rng.normal(size=D)
true_b = rng.normal(size=D)
VIEW_NOISE = 0.55      # how much the two camera angles differ
GALLERY_NOISE = 0.45   # how much the live view differs from the enrolled shot

idx_path, lbl_path = TMP / "f.index", TMP / "f.json"
for p in (idx_path, lbl_path, idx_path.with_suffix(".npy")):
    if p.exists():                 # unlink(missing_ok=) is 3.8+; Nano is 3.6
        p.unlink()

db = ThreatIndex(index_path=idx_path, labels_path=lbl_path, threshold=0.65)
for centre, name in ((true_a, "Subject_A"), (true_b, "Subject_B")):
    for _ in range(3):
        db.add(unit(centre + GALLERY_NOISE * rng.normal(size=D)), name)

fusion = CrossViewFusion(index=db, cross_view_threshold=0.45,
                         gallery_threshold=0.65, fusion_mode="mean")

# --- 1. two cameras see the SAME person -> one fused track -----------------
obs_a = view("CAM_A", true_a + VIEW_NOISE * rng.normal(size=D))
obs_b = view("CAM_B", true_a + VIEW_NOISE * rng.normal(size=D))
tracks = fusion.process({"CAM_A": [obs_a], "CAM_B": [obs_b]})

assert len(tracks) == 1, f"expected 1 fused track, got {len(tracks)}"
t = tracks[0]
assert t.is_multiview and sorted(t.cameras) == ["CAM_A", "CAM_B"], t.cameras
assert t.match.identity == "Subject_A", t.match.identity
solo_best = max(o.solo_match.score for o in t.observations)
print(f"same person : cross-view {t.cross_view_score:.3f} | "
      f"best single {solo_best:.3f} -> fused {t.match.score:.3f} "
      f"({t.match.score - solo_best:+.3f})")
assert t.match.score > solo_best, "fusion should beat the best single view"
assert t.label().endswith("[2v %.2f]" % t.cross_view_score), t.label()

# --- 2. two cameras see DIFFERENT people -> two separate tracks -------------
obs_a2 = view("CAM_A", true_a + VIEW_NOISE * rng.normal(size=D))
obs_b2 = view("CAM_B", true_b + VIEW_NOISE * rng.normal(size=D))
tracks = fusion.process({"CAM_A": [obs_a2], "CAM_B": [obs_b2]})

assert len(tracks) == 2, f"expected 2 separate tracks, got {len(tracks)}"
assert all(not t.is_multiview for t in tracks), "distinct people must not fuse"
ids = sorted(t.match.identity for t in tracks)
assert ids == ["Subject_A", "Subject_B"], ids
print(f"two people  : {len(tracks)} separate tracks -> {ids}")

# --- 3. two subjects per camera -> correct pairwise association ------------
pool = {
    "CAM_A": [view("CAM_A", true_a + VIEW_NOISE * rng.normal(size=D)),
              view("CAM_A", true_b + VIEW_NOISE * rng.normal(size=D))],
    "CAM_B": [view("CAM_B", true_b + VIEW_NOISE * rng.normal(size=D)),
              view("CAM_B", true_a + VIEW_NOISE * rng.normal(size=D))],
}
tracks = fusion.process(pool)
assert len(tracks) == 2, f"expected 2 fused tracks, got {len(tracks)}"
for t in tracks:
    assert t.n_views == 2, t.n_views
    assert len(set(t.cameras)) == 2, "a track must not hold two boxes from one camera"
ids = sorted(t.match.identity for t in tracks)
assert ids == ["Subject_A", "Subject_B"], ids
print(f"2x2 subjects: {len(tracks)} fused tracks, both 2-view -> {ids}")

# --- 4. single camera still works -----------------------------------------
tracks = fusion.process({"CAM_A": [view("CAM_A", true_a + VIEW_NOISE * rng.normal(size=D))]})
assert len(tracks) == 1 and not tracks[0].is_multiview
assert tracks[0].match.identity == "Subject_A"
print(f"single view : {tracks[0].match.identity} {tracks[0].match.score:.3f}")

# --- 5. require_both_views suppresses a lone sighting ----------------------
strict = CrossViewFusion(index=db, cross_view_threshold=0.45,
                         gallery_threshold=0.65, require_both_views=True)
tracks = strict.process({"CAM_A": [view("CAM_A", true_a)]})
assert not tracks[0].match.is_match, "single view should be suppressed"
assert tracks[0].notes, tracks[0].notes
print(f"strict mode : suppressed -> {tracks[0].notes[0]}")

# --- 6. fusion modes -------------------------------------------------------
e1, e2 = unit(rng.normal(size=D)), unit(rng.normal(size=D))
for mode in ("mean", "max", "first"):
    v = fuse_embeddings([e1, e2], mode)
    assert v.shape == (D,), v.shape
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5, f"{mode} not unit-norm"
assert np.allclose(fuse_embeddings([e1, e2], "first"), e1)
assert np.allclose(fuse_embeddings([e1], "mean"), e1)
print("fusion modes: mean/max/first all unit-norm")

# --- 7. empty input --------------------------------------------------------
assert fusion.process({}) == []
assert fusion.process({"CAM_A": [], "CAM_B": []}) == []
print("empty input : handled")

print("\nALL PASS")
