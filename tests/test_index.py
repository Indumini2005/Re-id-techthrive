"""Exercise ThreatIndex end to end on the numpy fallback backend.
cv2/onnxruntime are stubbed; neither is used by the code under test."""

import sys
import types
from pathlib import Path

import numpy as np

for name in ("cv2", "onnxruntime"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TMP = Path(__file__).parent / "db_test"
TMP.mkdir(exist_ok=True)

from pipeline.stage5_match import ThreatIndex  # noqa: E402

idx_path, lbl_path = TMP / "t.index", TMP / "t.json"
for p in (idx_path, lbl_path, idx_path.with_suffix(".npy")):
    if p.exists():                 # unlink(missing_ok=) is 3.8+; Nano is 3.6
        p.unlink()

rng = np.random.default_rng(7)
db = ThreatIndex(index_path=idx_path, labels_path=lbl_path, threshold=0.65)
print("backend:", db.stats()["backend"])

# Two identities, three vectors each, clustered around a per-identity centre.
centres = {"Subject_A": rng.normal(size=512), "Subject_B": rng.normal(size=512)}
for name, centre in centres.items():
    for _ in range(3):
        db.add(centre + 0.05 * rng.normal(size=512), name)

assert db.index.ntotal == 6, db.index.ntotal
assert db.identities == ["Subject_A", "Subject_B"], db.identities
print("added  :", db.stats())

# A probe near A's centre must match A; an unrelated probe must not match.
hit = db.query(centres["Subject_A"] + 0.05 * rng.normal(size=512))
print("probe A:", hit.identity, round(hit.score, 4), "match =", hit.is_match)
assert hit.is_match and hit.identity == "Subject_A"

miss = db.query(rng.normal(size=512))
print("probe ?:", miss.identity, round(miss.score, 4), "match =", miss.is_match)
assert not miss.is_match and miss.identity == "Unknown"

# top-k ordering must be descending
hits = db.search(centres["Subject_B"], top_k=4)
scores = [h.score for h in hits]
assert scores == sorted(scores, reverse=True), scores
assert hits[0].identity == "Subject_B"
print("top-k  :", [(h.identity, round(h.score, 3)) for h in hits])

# persistence round-trip
reloaded = ThreatIndex(index_path=idx_path, labels_path=lbl_path, threshold=0.65)
assert reloaded.index.ntotal == 6, reloaded.index.ntotal
assert reloaded.labels == db.labels
re_hit = reloaded.query(centres["Subject_A"])
assert re_hit.is_match and re_hit.identity == "Subject_A"
print("reload :", reloaded.stats())

# removal must drop exactly that identity's rows and keep the rest searchable
removed = reloaded.remove_identity("Subject_A")
assert removed == 3, removed
assert reloaded.index.ntotal == 3, reloaded.index.ntotal
assert reloaded.identities == ["Subject_B"], reloaded.identities
still = reloaded.query(centres["Subject_B"])
assert still.is_match and still.identity == "Subject_B"
gone = reloaded.query(centres["Subject_A"])
assert not gone.is_match
print("remove :", reloaded.stats())

# empty index must not crash
reloaded.clear()
assert reloaded.query(centres["Subject_A"]).identity == "Unknown"
assert reloaded.search(centres["Subject_A"]) == []
print("clear  :", reloaded.stats())

print("\nALL PASS")
