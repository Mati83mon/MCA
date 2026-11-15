"""Dense embedding backends and a cosine vector index (FAISS when available)."""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol, Sequence

import numpy as np

from mca.config import MCAConfig
from mca.utils.text import STOPWORDS, strip_accents, tokenize

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalised float32 embeddings of shape (n, dim)."""
        ...


class SentenceTransformerEmbedder:
    """``sentence-transformers`` bi-encoder (default all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        from sentence_transformers import SentenceTransformer

        kwargs = {} if device == "auto" else {"device": device}
        self.model = SentenceTransformer(model_name, **kwargs)
        self.name = f"st:{model_name}"
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype(np.float32)


class HashingEmbedder:
    """Signed feature hashing of word stems + accent-folded character n-grams.

    A zero-download dense representation that is robust to Polish inflection;
    used offline and as fallback when no sentence-transformer can be loaded.
    """

    def __init__(self, dim: int = 4096, ngram_range: tuple[int, int] = (3, 5)) -> None:
        self.dim = dim
        self.ngram_range = ngram_range
        self.name = f"hashing:{dim}"

    def _features(self, text: str) -> list[tuple[str, float]]:
        feats: list[tuple[str, float]] = []
        for tok in tokenize(text):
            if tok in STOPWORDS:
                continue
            folded = strip_accents(tok)
            feats.append((f"w:{folded[:6]}", 2.0))
            padded = f"<{folded}>"
            lo, hi = self.ngram_range
            for n in range(lo, hi + 1):
                for i in range(max(0, len(padded) - n + 1)):
                    feats.append((f"c:{padded[i:i + n]}", 1.0))
        return feats

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, weight in self._features(text):
                digest = hashlib.blake2b(feat.encode(), digest_size=8).digest()
                h = int.from_bytes(digest, "little")
                out[row, h % self.dim] += weight if (h >> 63) & 1 else -weight
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


class VectorIndex:
    """Inner-product index over normalised vectors (FAISS ``IndexFlatIP`` or NumPy)."""

    def __init__(self, dim: int, use_faiss: bool = True) -> None:
        self.dim = dim
        self._faiss = None
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        if use_faiss:
            try:
                import faiss

                self._faiss = faiss.IndexFlatIP(dim)
            except ImportError:
                logger.info("faiss not installed; using NumPy brute-force index")
        self.backend = "faiss" if self._faiss is not None else "numpy"

    def __len__(self) -> int:
        return self._faiss.ntotal if self._faiss is not None else self._matrix.shape[0]

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self._faiss is not None:
            self._faiss.add(vectors)
        else:
            self._matrix = np.vstack([self._matrix, vectors])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        k = min(k, len(self))
        if k == 0:
            return np.zeros((len(queries), 0), np.float32), np.zeros((len(queries), 0), np.int64)
        if self._faiss is not None:
            return self._faiss.search(queries, k)
        sims = queries @ self._matrix.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, idx, axis=1), idx


def load_embedder(config: MCAConfig) -> Embedder:
    """Instantiate the configured embedder; ``auto`` falls back to hashing when offline."""
    if config.embedding_backend in {"auto", "sentence-transformers"}:
        try:
            return SentenceTransformerEmbedder(config.embedding_model, device=config.device)
        except Exception as exc:  # noqa: BLE001 - offline / missing extras -> fallback
            if config.embedding_backend == "sentence-transformers":
                raise
            logger.warning("embedding model %s unavailable (%s); using hashing embedder", config.embedding_model, exc)
    return HashingEmbedder()
