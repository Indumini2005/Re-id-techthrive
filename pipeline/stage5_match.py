"""Stage 5 - FAISS vector DB and cosine threat matching.

Vectors are L2-normalized by stage 4, so an inner-product index
(``IndexFlatIP``) gives exact cosine similarity. A subject may hold several
vectors (multiple angles); a query matches an identity if *any* of its
vectors clears the threshold.

Everything lives on local disk under ``vector_db/`` - nothing leaves the node.
"""

import json
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import config
from .vectors import l2_normalize

try:
    import faiss
    _HAS_FAISS = True
except ImportError:                                    # dev machines without faiss
    faiss = None
    _HAS_FAISS = False


@dataclass
class Match:
    is_match: bool
    identity: str
    score: float
    rank: int = 0

    def as_tuple(self) -> Tuple[bool, str, float]:
        return self.is_match, self.identity, self.score

    def to_dict(self) -> dict:
        return asdict(self)


class _NumpyIndex:
    """Exact inner-product fallback so the pipeline runs without faiss."""

    def __init__(self, dim: int):
        self.d = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return self._vectors.shape[0]

    def add(self, vectors: np.ndarray) -> None:
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])

    def search(self, queries: np.ndarray, k: int):
        if self.ntotal == 0:
            return (np.zeros((len(queries), k), np.float32),
                    -np.ones((len(queries), k), np.int64))
        sims = queries.astype(np.float32) @ self._vectors.T
        k = min(k, self.ntotal)
        idx = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, idx, axis=1), idx.astype(np.int64)

    def reconstruct_n(self, start: int, count: int) -> np.ndarray:
        return self._vectors[start : start + count].copy()


class ThreatIndex:
    """Persistent identity store. Index rows and ``labels`` stay parallel."""

    def __init__(
        self,
        index_path: Optional[Path] = None,
        labels_path: Optional[Path] = None,
        dim: int = config.EMBEDDING_DIM,
        threshold: float = config.MATCH_THRESHOLD,
    ):
        self.index_path = Path(index_path or config.FAISS_INDEX)
        self.labels_path = Path(labels_path or config.FAISS_LABELS)
        self.dim = dim
        self.threshold = threshold
        self.labels: List[str] = []
        self.meta: List[dict] = []
        self._lock = threading.Lock()

        self.index = self._new_index()
        self.load()

    # -- persistence -------------------------------------------------------
    def _new_index(self):
        return faiss.IndexFlatIP(self.dim) if _HAS_FAISS else _NumpyIndex(self.dim)

    def load(self) -> bool:
        if not (self.index_path.exists() and self.labels_path.exists()):
            return False
        try:
            with open(self.labels_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.labels = payload.get("labels", [])
            self.meta = payload.get("meta", [{} for _ in self.labels])

            if _HAS_FAISS:
                self.index = faiss.read_index(str(self.index_path))
            else:
                self.index = self._new_index()
                self.index.add(np.load(self.index_path.with_suffix(".npy")))

            if self.index.ntotal != len(self.labels):
                print(
                    f"[vector-db] index/label mismatch "
                    f"({self.index.ntotal} vs {len(self.labels)}); rebuilding empty."
                )
                self.clear(persist=False)
                return False
            return True
        except Exception as exc:                        # corrupt store -> start clean
            print(f"[vector-db] could not load index: {exc}")
            self.clear(persist=False)
            return False

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if _HAS_FAISS:
            faiss.write_index(self.index, str(self.index_path))
        else:
            np.save(self.index_path.with_suffix(".npy"), self.index.reconstruct_n(0, self.index.ntotal))
            self.index_path.touch()
        with open(self.labels_path, "w", encoding="utf-8") as fh:
            json.dump({"labels": self.labels, "meta": self.meta}, fh, indent=2)

    def clear(self, persist: bool = True) -> None:
        self.index = self._new_index()
        self.labels, self.meta = [], []
        if persist:
            self.save()

    # -- writes ------------------------------------------------------------
    def add(self, embedding: np.ndarray, identity: str, note: str = "", persist: bool = True) -> int:
        """Register one vector for ``identity``. Returns its row id."""
        if embedding is None:
            raise ValueError("embedding is None")
        vec = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(1, -1))
        if vec.shape[1] != self.dim:
            raise ValueError(f"expected {self.dim}-D embedding, got {vec.shape[1]}")

        with self._lock:
            self.index.add(vec)
            self.labels.append(identity)
            self.meta.append({"note": note, "added": datetime.now().isoformat(timespec="seconds")})
            row = len(self.labels) - 1
            if persist:
                self.save()
        return row

    def add_many(self, embeddings: np.ndarray, identity: str, note: str = "") -> List[int]:
        rows = [self.add(e, identity, note, persist=False) for e in np.atleast_2d(embeddings)]
        self.save()
        return rows

    def remove_identity(self, identity: str) -> int:
        """Drop every vector for an identity. Rebuilds the index."""
        with self._lock:
            keep = [i for i, lab in enumerate(self.labels) if lab != identity]
            removed = len(self.labels) - len(keep)
            if removed == 0:
                return 0
            # IndexFlat supports exact reconstruction, so a rebuild is lossless.
            vectors = np.asarray(
                self.index.reconstruct_n(0, self.index.ntotal), dtype=np.float32
            ).reshape(self.index.ntotal, self.dim)
            kept_vectors = np.ascontiguousarray(vectors[keep], dtype=np.float32)
            self.labels = [self.labels[i] for i in keep]
            self.meta = [self.meta[i] for i in keep]
            self.index = self._new_index()
            if len(kept_vectors):
                self.index.add(kept_vectors)
            self.save()
        return removed

    # -- reads -------------------------------------------------------------
    def search(self, embedding: np.ndarray, top_k: int = config.MATCH_TOP_K) -> List[Match]:
        if embedding is None or self.index.ntotal == 0:
            return []
        query = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(1, -1))
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results: List[Match] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            results.append(
                Match(
                    is_match=bool(score >= self.threshold),
                    identity=self.labels[int(idx)],
                    score=float(score),
                    rank=rank,
                )
            )
        return results

    def query(self, embedding: np.ndarray, threshold: Optional[float] = None) -> Match:
        """Best single hit. Always returns a Match; check ``.is_match``."""
        thr = self.threshold if threshold is None else threshold
        hits = self.search(embedding, top_k=1)
        if not hits:
            return Match(False, "Unknown", 0.0)
        best = hits[0]
        best.is_match = best.score >= thr
        if not best.is_match:
            return Match(False, "Unknown", best.score)
        return best

    @property
    def identities(self) -> List[str]:
        return sorted(set(self.labels))

    def stats(self) -> dict:
        return {
            "backend": "faiss" if _HAS_FAISS else "numpy",
            "vectors": int(self.index.ntotal),
            "identities": len(self.identities),
            "threshold": self.threshold,
            "path": str(self.index_path),
        }


_index: Optional[ThreatIndex] = None
_index_lock = threading.Lock()


def get_index(**kwargs) -> ThreatIndex:
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = ThreatIndex(**kwargs)
    return _index


def register_known_subject(face_bgr: np.ndarray, identity: str, note: str = "") -> bool:
    """Enroll a face image (already restored/aligned) under ``identity``."""
    from .stage4_embed import get_embedder

    embedding = get_embedder().embed_flipped(face_bgr)
    if embedding is None:
        return False
    get_index().add(embedding, identity, note)
    return True


def query_cctv_crop(
    face_bgr: np.ndarray, threshold: float = config.MATCH_THRESHOLD
) -> Tuple[bool, str, float]:
    """Stage-5 entry point. Returns ``(is_match, identity, score)``."""
    from .stage4_embed import get_embedder

    embedding = get_embedder().embed(face_bgr)
    if embedding is None:
        return False, "Unknown", 0.0
    return get_index().query(embedding, threshold).as_tuple()
