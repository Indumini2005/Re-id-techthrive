"""--restore_mode policy in main_pipeline._restore_and_embed.

GFPGAN costs seconds per face on Maxwell, so the question "was the expensive
model invoked?" is the thing worth asserting."""

import sys
import types
from pathlib import Path

import numpy as np

for name in ("onnxruntime", "ultralytics", "gfpgan", "faiss"):
    sys.modules.setdefault(name, types.ModuleType(name))
cv2 = types.ModuleType("cv2")
cv2.setNumThreads = lambda n: None
cv2.getNumThreads = lambda: 1
cv2.resize = lambda img, dsize, interpolation=None: np.zeros(
    (dsize[1], dsize[0]) + img.shape[2:], img.dtype)
cv2.INTER_AREA = 3
cv2.getTextSize = lambda *a: ((10, 10), 1)
sys.modules["cv2"] = cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main_pipeline  # noqa: E402
import pipeline.face_align as face_align  # noqa: E402
from pipeline.fusion import ViewObservation  # noqa: E402
from pipeline.stage5_match import Match  # noqa: E402
from pipeline.utils import StageTimer  # noqa: E402

CROP = np.full((80, 40, 3), 7, np.uint8)
FACE = np.full((512, 512, 3), 9, np.uint8)
ALIGNED = np.full((112, 112, 3), 5, np.uint8)

# This file tests the restore POLICY - when GFPGAN is invoked - not face
# alignment, which test_align.py covers. Stub the aligner so the two
# concerns stay separable and a stub cv2 is enough.
ALIGN_CALLS = []


def fake_align(crop, size=112, denoiser=None, allow_fallback=True):
    ALIGN_CALLS.append(crop.shape)
    return ALIGNED.copy()


face_align.prepare_for_embedding = fake_align
RAW_VEC = np.array([1.0] + [0.0] * 511, np.float32)
FINE_VEC = np.array([0.0, 1.0] + [0.0] * 510, np.float32)


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, image):
        self.calls.append(image.shape)
        # 512 = a GFPGAN-restored face; 112 = the plain aligned face
        return FINE_VEC if image.shape[0] == 512 else RAW_VEC


class FakeRestorer:
    def __init__(self, returns=FACE):
        self.calls = 0
        self.returns = returns

    def restore(self, image):
        self.calls += 1
        return self.returns


class FakeIndex:
    def __init__(self, prelim_score, ntotal=10):
        self.index = types.SimpleNamespace(ntotal=ntotal)
        self.prelim_score = prelim_score
        self.queries = 0

    def query(self, embedding, threshold=None):
        self.queries += 1
        return Match(self.prelim_score >= 0.65, "Subject_A", self.prelim_score)


class Args:
    def __init__(self, mode, escalate_above=0.40, threshold=0.65):
        self.restore_mode = mode
        self.escalate_above = escalate_above
        self.threshold = threshold


def build(mode, prelim=0.0, restorer=True, ntotal=10, escalate_above=0.40):
    stages = types.SimpleNamespace(
        embedder=FakeEmbedder(),
        restorer=FakeRestorer() if restorer else None,
        index=FakeIndex(prelim, ntotal),
    )
    obs = ViewObservation(camera="CAM_A", box=(0, 0, 40, 80), conf=0.9, crop=CROP)
    main_pipeline._restore_and_embed(
        obs, "CAM_A", stages, StageTimer(), Args(mode, escalate_above)
    )
    return obs, stages


failures = 0


def check(name, cond, detail=""):
    global failures
    print(("OK   " if cond else "FAIL ") + name + ("   " + detail if detail else ""))
    if not cond:
        failures += 1


# --- never: GFPGAN must not be touched -------------------------------------
obs, st = build("never", prelim=0.99)
check("never: no restore", st.restorer.calls == 0)
check("never: embeds the crop", np.allclose(obs.embedding, RAW_VEC))
check("never: not flagged escalated", obs.escalated is False)

# --- always: GFPGAN on every subject ---------------------------------------
obs, st = build("always", prelim=0.0)
check("always: restore called once", st.restorer.calls == 1)
check("always: embeds the restored face", np.allclose(obs.embedding, FINE_VEC))
check("always: no prefilter query", st.index.queries == 0)

# --- escalate, clearly nobody: cheap path only ------------------------------
obs, st = build("escalate", prelim=0.11)
check("escalate/low: no restore", st.restorer.calls == 0, "prelim 0.11 < 0.40")
check("escalate/low: keeps the cheap embedding", np.allclose(obs.embedding, RAW_VEC))
check("escalate/low: one prefilter query", st.index.queries == 1)
check("escalate/low: not escalated", obs.escalated is False)

# --- escalate, plausible candidate: pay for GFPGAN --------------------------
obs, st = build("escalate", prelim=0.52)
check("escalate/high: restore called", st.restorer.calls == 1, "prelim 0.52 >= 0.40")
check("escalate/high: embedding upgraded", np.allclose(obs.embedding, FINE_VEC))
check("escalate/high: flagged escalated", obs.escalated is True)
check("escalate/high: embedded twice", len(st.embedder.calls) == 2,
      "aligned then restored: %s" % st.embedder.calls)

# --- alignment runs before embedding, always --------------------------------
ALIGN_CALLS.clear()
obs, st = build("never", prelim=0.0)
check("alignment runs even in 'never' mode", len(ALIGN_CALLS) == 1, str(ALIGN_CALLS))
check("embedder receives the 112x112 aligned face",
      st.embedder.calls and st.embedder.calls[0] == (112, 112, 3), str(st.embedder.calls))

# --- no face detected: nothing is embedded ----------------------------------
face_align.prepare_for_embedding = lambda *a, **k: None
obs, st = build("escalate", prelim=0.99)
check("no face -> no embedding", obs.embedding is None)
check("no face -> no restore spent", st.restorer.calls == 0)
face_align.prepare_for_embedding = fake_align

# --- threshold boundary is inclusive ---------------------------------------
obs, st = build("escalate", prelim=0.40)
check("escalate: boundary escalates", st.restorer.calls == 1, "prelim == escalate_above")

# --- empty gallery: nothing to escalate towards -----------------------------
obs, st = build("escalate", prelim=0.99, ntotal=0)
check("escalate/empty DB: no restore", st.restorer.calls == 0)
check("escalate/empty DB: cheap embedding kept", np.allclose(obs.embedding, RAW_VEC))

# --- restorer unavailable: degrade to 'never', do not crash -----------------
obs, st = build("escalate", prelim=0.99, restorer=False)
check("no restorer: still embeds", np.allclose(obs.embedding, RAW_VEC))
check("no restorer: no prefilter spent", st.index.queries == 0)

# --- GFPGAN finds no face: keep the cheap embedding -------------------------
stages = types.SimpleNamespace(
    embedder=FakeEmbedder(), restorer=FakeRestorer(returns=None), index=FakeIndex(0.60)
)
obs = ViewObservation(camera="CAM_A", box=(0, 0, 40, 80), conf=0.9, crop=CROP)
main_pipeline._restore_and_embed(obs, "CAM_A", stages, StageTimer(), Args("escalate"))
check("restore returns None: embedding retained", np.allclose(obs.embedding, RAW_VEC))
check("restore returns None: not escalated", obs.escalated is False)

# --- downscale helper -------------------------------------------------------
big = np.zeros((1080, 1920, 3), np.uint8)
check("downscale 1920 -> 640", main_pipeline.downscale(big, 640).shape[:2] == (360, 640),
      str(main_pipeline.downscale(big, 640).shape))
check("downscale 0 = passthrough", main_pipeline.downscale(big, 0) is big)
small = np.zeros((360, 640, 3), np.uint8)
check("no upscaling", main_pipeline.downscale(small, 1280) is small)

print("\n" + ("ALL PASS" if not failures else "%d FAILURE(S)" % failures))
sys.exit(1 if failures else 0)
