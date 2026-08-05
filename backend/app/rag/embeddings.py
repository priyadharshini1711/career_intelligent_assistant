"""Embedding models.

Choice: `all-MiniLM-L6-v2`, run locally.

Reasoning -- resumes and JDs are short documents and a session holds maybe a
few hundred chunks. That workload does not need a 1024-dimension hosted model;
it needs something accurate enough to tell "built REST APIs in Django" apart
from "consumed third-party REST APIs", which MiniLM does well. Running it in
process also means no per-query embedding cost, no rate limit, no network hop
on the upload path, and no user document leaving the machine -- which matters
when the document is someone's resume.

The `Embedder` protocol exists so the hosted-API option stays open, and so the
test suite can use a deterministic hashing embedder that needs no model
download and no network.
"""

import hashlib
import threading
from typing import List, Protocol

import numpy as np

from app.observability import get_logger

logger = get_logger(__name__)


class Embedder(Protocol):
    dimension: int
    name: str
    # Cosine score below which a match is treated as "no real evidence".
    # It belongs to the embedder, not to the retriever: what counts as a weak
    # similarity is a property of the vector space, and hard-coding one number
    # for every model made the retriever silently report "no evidence" when the
    # model was swapped.
    min_relevance: float

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalised embeddings, shape (len(texts), dimension)."""
        ...


class HashingEmbedder:
    """Deterministic bag-of-hashed-ngrams vectoriser.

    Not competitive with a real model, and not meant to be. It exists so tests
    and CI runs are fast, offline, and reproducible, and so the app can still
    boot if the model download fails. Selected with EMBEDDING_MODEL=hashing.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.name = "hashing"
        # Sparse hashed vectors score near zero even for genuinely related
        # text, so any positive similarity is as much signal as this space
        # offers. The lexical channel carries the weak-evidence check here.
        self.min_relevance = 0.0

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = text.lower().split()
        # Unigrams plus bigrams gives a little word-order sensitivity.
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._vector(text) for text in texts])


class SentenceTransformerEmbedder:
    """Wraps sentence-transformers with lazy loading.

    Loading is deferred to the first encode so that importing the app (and
    running the parts of the test suite that never embed anything) does not pay
    for a model load.
    """

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.name = model_name
        self.batch_size = batch_size
        self.dimension = 384  # corrected once the model reports its real size
        self.min_relevance = 0.05
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("loading embedding model", extra={"model": self.name})
                    self._model = SentenceTransformer(self.name)
                    self.dimension = self._model.get_sentence_embedding_dimension()
                    logger.info(
                        "embedding model ready",
                        extra={"model": self.name, "dimension": self.dimension},
                    )
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        model = self._ensure_model()
        return model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)


def build_embedder(model_name: str, batch_size: int = 32) -> Embedder:
    if model_name.lower() in {"hashing", "hash", "test"}:
        return HashingEmbedder()
    return SentenceTransformerEmbedder(model_name, batch_size=batch_size)
