"""Stage 4 - ArcFace embedding extraction.

Runs the ONNX ArcFace backbone directly (no insightface runtime) so the same
file works under ONNX Runtime's TensorRT provider on the Jetson.

Input contract: an *aligned* face, which is what GFPGAN hands back. Images are
resized to 112x112, converted to RGB and scaled to [-1, 1] - the standard
ArcFace / insightface preprocessing.

Output: a 512-D L2-normalized vector, so a dot product is cosine similarity.
"""

import threading
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

import config

# Re-exported for backwards compatibility; the implementation lives in
# pipeline/vectors.py so OpenCV-free consumers can use it.
from .vectors import cosine, l2_normalize  # noqa: F401


class ArcFaceEmbedder:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        provider: Optional[str] = None,
        input_size: int = config.ARCFACE_INPUT,
    ):
        import onnxruntime as ort

        self.model_path = Path(model_path or config.ARCFACE_ONNX)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ArcFace model not found at {self.model_path}. Place an ONNX "
                "recognition backbone (e.g. w600k_r50.onnx from the "
                "insightface buffalo_l pack) there."
            )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=opts,
            providers=config.provider_list(provider),
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = input_size
        self._lock = threading.Lock()

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """BGR HWC uint8 -> CHW float32 in [-1, 1], RGB channel order."""
        if face_bgr.shape[:2] != (self.input_size, self.input_size):
            face_bgr = cv2.resize(
                face_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA
            )
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 127.5
        return np.transpose(rgb, (2, 0, 1))

    def embed(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Single aligned face -> (512,) float32 unit vector."""
        if face_bgr is None or face_bgr.size == 0:
            return None
        return self.embed_batch([face_bgr])[0]

    def embed_batch(self, faces_bgr: List[np.ndarray]) -> np.ndarray:
        """-> (N, 512) float32, rows L2-normalized."""
        if not faces_bgr:
            return np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)
        batch = np.stack([self.preprocess(f) for f in faces_bgr]).astype(np.float32)
        with self._lock:
            out = self.session.run([self.output_name], {self.input_name: batch})[0]
        return l2_normalize(np.asarray(out, dtype=np.float32))

    def embed_flipped(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Flip-averaged embedding. Slightly more robust on off-angle CCTV
        faces at the cost of a second forward pass."""
        if face_bgr is None or face_bgr.size == 0:
            return None
        both = self.embed_batch([face_bgr, cv2.flip(face_bgr, 1)])
        return l2_normalize(both.mean(axis=0))


_embedder: Optional[ArcFaceEmbedder] = None
_embedder_lock = threading.Lock()


def get_embedder(**kwargs) -> ArcFaceEmbedder:
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = ArcFaceEmbedder(**kwargs)
    return _embedder


def extract_embedding(face_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Stage-4 entry point used by ``main_pipeline``."""
    return get_embedder().embed(face_bgr)
