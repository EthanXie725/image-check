from __future__ import annotations

import threading
from pathlib import Path

import faiss
import numpy as np

from app.config import CLIP_EMBED_DIM


class FaissStore:
    """FAISS IndexIDMap2 over IndexFlatIP for cosine similarity on L2-normalized vectors."""

    def __init__(self, dim: int = CLIP_EMBED_DIM) -> None:
        self.dim = dim
        self._lock = threading.Lock()
        base = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap2(base)

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors shape (N, {self.dim}), got {vectors.shape}")
        if ids.ndim != 1 or ids.shape[0] != vectors.shape[0]:
            raise ValueError("ids shape must be (N,) matching vectors")

        ids64 = np.ascontiguousarray(ids.astype(np.int64))
        vecs = np.ascontiguousarray(vectors.astype(np.float32))
        with self._lock:
            self.index.add_with_ids(vecs, ids64)

    def search(
        self, query: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, ids) each shape (1, top_k). Missing slots have id=-1."""
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[1] != self.dim:
            raise ValueError(f"Expected query dim {self.dim}, got {query.shape[1]}")

        q = np.ascontiguousarray(query.astype(np.float32))
        k = max(1, min(top_k, max(self.ntotal, 1)))
        with self._lock:
            if self.ntotal == 0:
                return (
                    np.full((1, top_k), -1.0, dtype=np.float32),
                    np.full((1, top_k), -1, dtype=np.int64),
                )
            scores, ids = self.index.search(q, k)
        # Pad if ntotal < top_k requested
        if scores.shape[1] < top_k:
            pad = top_k - scores.shape[1]
            scores = np.pad(scores, ((0, 0), (0, pad)), constant_values=-1.0)
            ids = np.pad(ids, ((0, 0), (0, pad)), constant_values=-1)
        return scores, ids

    def remove_ids(self, ids: list[int]) -> int:
        if not ids:
            return 0
        id_sel = faiss.IDSelectorBatch(np.asarray(ids, dtype=np.int64))
        with self._lock:
            return int(self.index.remove_ids(id_sel))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            faiss.write_index(self.index, str(path))

    def load(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found: {path}")
        with self._lock:
            self.index = faiss.read_index(str(path))
            # Infer dim from loaded index when possible
            if hasattr(self.index, "d"):
                self.dim = int(self.index.d)

    @classmethod
    def load_or_create(cls, path: Path, dim: int = CLIP_EMBED_DIM) -> "FaissStore":
        store = cls(dim=dim)
        path = Path(path)
        if path.exists():
            store.load(path)
        return store
