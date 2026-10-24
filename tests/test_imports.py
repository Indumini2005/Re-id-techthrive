"""Import every device-side module with PEP 563 removed.

Without `from __future__ import annotations`, every signature and every
module/class-level annotation is evaluated at import time, so a forward
reference that used to be harmless is now a NameError. py_compile cannot
catch that - only an actual import can.
"""

import importlib
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Stub the heavy third-party modules; we are testing our own annotations.
for name in ("cv2", "onnxruntime", "ultralytics", "gfpgan", "faiss", "torch"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["cv2"].setNumThreads = lambda n: None
sys.modules["cv2"].getNumThreads = lambda: 1
sys.modules["ultralytics"].YOLO = object
sys.modules["gfpgan"].GFPGANer = object
sys.modules["torch"].cuda = types.SimpleNamespace(is_available=lambda: False)

MODULES = [
    "config",
    "pipeline.utils",
    "pipeline.sources",
    "pipeline.fusion",
    "pipeline.stage1_denoise",
    "pipeline.stage2_detect",
    "pipeline.stage3_restore",
    "pipeline.stage4_embed",
    "pipeline.stage5_match",
    "pipeline",
    "arcface_faiss",
    "gpfgan",
    "yolocropping",
]

failures = []
for name in MODULES:
    try:
        importlib.import_module(name)
        print("OK   %s" % name)
    except Exception as exc:
        failures.append((name, exc))
        print("FAIL %s -> %s: %s" % (name, type(exc).__name__, exc))

# Instantiating the dataclasses exercises the evaluated field annotations.
try:
    from pipeline.fusion import FusedTrack, ViewObservation
    from pipeline.sources import Frame, FramePair

    obs = ViewObservation(camera="CAM_A", box=(0, 0, 1, 1), conf=0.5,
                          crop=np.zeros((2, 2, 3), np.uint8))
    track = FusedTrack(observations=[obs])
    frame = Frame(np.zeros((2, 2, 3), np.uint8), 1.0, 1, "CAM_A")
    pair = FramePair(frames={"CAM_A": frame}, skew_ms=0.0, in_sync=True)
    assert track.cameras == ["CAM_A"] and pair.cameras == ["CAM_A"]
    print("OK   dataclass construction")
except Exception as exc:
    failures.append(("dataclasses", exc))
    print("FAIL dataclasses -> %s: %s" % (type(exc).__name__, exc))

print("\n%s" % ("ALL PASS" if not failures else "%d FAILURE(S)" % len(failures)))
sys.exit(1 if failures else 0)
