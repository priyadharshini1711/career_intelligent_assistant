"""Chunk index: dense vectors + a lexical (BM25) index over the same chunks.

Choice: an in-process NumPy index, not a vector database.

Reasoning -- a session holds one resume and up to ten job descriptions, which
is a few hundred chunks. Exact cosine over a (300, 384) matrix is a single
matrix multiply: tens of microseconds. A dedicated vector DB would add a
container, a network hop, and an index-tuning conversation to buy approximate
search we do not need at this size. What a vector DB *would* buy is
persistence and multi-tenancy, and this take-home is explicitly session-scoped
and ephemeral -- so the honest trade is to skip it and keep the seam.

The seam is the `ChunkStore` protocol. Everything above it (retriever, RAG
pipeline) only knows these five methods, so swapping in pgvector or Qdrant is
one new class, not a refactor. The README covers what that migration looks
like.

Both an exact-cosine index and BM25 live here because they index the same
chunks and must stay in lockstep; splitting them would mean two structures to
keep consistent on every add and delete.
"""

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from app.schemas import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")

# Words that carry no signal in this domain. Kept short on purpose: aggressive
# stoplists strip things like "C" or "R" that are real skills here.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "our", "that", "the", "to", "we",
    "will", "with", "you", "your",
}

# BM25 parameters. k1 controls term-frequency saturation, b controls
# length normalisation. These are the standard defaults and there is no
# tuning signal at this corpus size to justify moving them.
_BM25_K1 = 1.4
_BM25_B = 0.75


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens, keeping `c++`, `c#`, `node.js` intact."""
    return [
        token.strip(".")
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS and len(token.strip(".")) > 1
    ]


class ChunkStore(Protocol):
    """The seam between retrieval and storage."""

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...

    def remove_document(self, document_id: str) -> int: ...

    def chunks(self) -> List[Chunk]: ...

    def dense_search(
        self, query_vector: np.ndarray, allowed_ids: Optional[Iterable[str]], top_k: int
    ) -> List[Tuple[Chunk, float]]: ...

    def lexical_search(
        self, query: str, allowed_ids: Optional[Iterable[str]], top_k: int
    ) -> List[Tuple[Chunk, float]]: ...


class InMemoryChunkStore:
    """Exact cosine + BM25 over chunks held in memory.

    Invariant: `self._chunks[i]` is described by `self._vectors[i]` and
    `self._term_freqs[i]`. Every mutation rebuilds all three together. The v0
    of this project kept metadata in a parallel list that got re-sorted
    independently of the scores, so citations pointed at the wrong document --
    keeping the arrays index-aligned and rebuilt as a unit is what prevents
    that class of bug.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._chunks: List[Chunk] = []
        self._vectors = np.zeros((0, dimension), dtype=np.float32)
        self._term_freqs: List[Counter] = []
        self._lengths: List[int] = []
        self._doc_freqs: Counter = Counter()
        self._avg_length: float = 0.0

    # -- mutation ---------------------------------------------------------

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
            )
        if not chunks:
            return

        self._chunks.extend(chunks)
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            self._term_freqs.append(Counter(tokens))
            self._lengths.append(len(tokens))
        self._reindex_lexical()

    def remove_document(self, document_id: str) -> int:
        keep = [i for i, chunk in enumerate(self._chunks) if chunk.document_id != document_id]
        removed = len(self._chunks) - len(keep)
        if not removed:
            return 0

        self._chunks = [self._chunks[i] for i in keep]
        self._vectors = self._vectors[keep] if keep else np.zeros((0, self.dimension), np.float32)
        self._term_freqs = [self._term_freqs[i] for i in keep]
        self._lengths = [self._lengths[i] for i in keep]
        self._reindex_lexical()
        return removed

    def _reindex_lexical(self) -> None:
        self._doc_freqs = Counter()
        for freqs in self._term_freqs:
            self._doc_freqs.update(freqs.keys())
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

    # -- reads ------------------------------------------------------------

    def chunks(self) -> List[Chunk]:
        return list(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def _allowed_mask(self, allowed_ids: Optional[Iterable[str]]) -> Optional[np.ndarray]:
        if allowed_ids is None:
            return None
        allowed = set(allowed_ids)
        return np.array([chunk.document_id in allowed for chunk in self._chunks], dtype=bool)

    def dense_search(
        self, query_vector: np.ndarray, allowed_ids: Optional[Iterable[str]], top_k: int
    ) -> List[Tuple[Chunk, float]]:
        if not self._chunks or top_k <= 0:
            return []

        # Vectors are stored L2-normalised, so the dot product is cosine.
        scores = self._vectors @ query_vector.astype(np.float32)

        mask = self._allowed_mask(allowed_ids)
        if mask is not None:
            if not mask.any():
                return []
            scores = np.where(mask, scores, -np.inf)

        count = min(top_k, int(np.isfinite(scores).sum()))
        if count <= 0:
            return []
        top = np.argpartition(-scores, count - 1)[:count]
        top = top[np.argsort(-scores[top])]
        return [(self._chunks[i], float(scores[i])) for i in top]

    def lexical_search(
        self, query: str, allowed_ids: Optional[Iterable[str]], top_k: int
    ) -> List[Tuple[Chunk, float]]:
        """BM25.

        Dense retrieval is weak on exact tokens -- version numbers, acronyms,
        and tools that look alike in embedding space ("Kafka" vs "Kinesis").
        Those are exactly the tokens a skill-gap question turns on, so a
        lexical channel earns its place next to the vectors.
        """
        if not self._chunks or top_k <= 0:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        total_docs = len(self._chunks)
        mask = self._allowed_mask(allowed_ids)

        scored: List[Tuple[int, float]] = []
        for i, freqs in enumerate(self._term_freqs):
            if mask is not None and not mask[i]:
                continue
            length = self._lengths[i] or 1
            score = 0.0
            for term in query_terms:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                df = self._doc_freqs.get(term, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denominator = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * length / (self._avg_length or 1))
                score += idf * (tf * (_BM25_K1 + 1)) / denominator
            if score > 0:
                scored.append((i, score))

        scored.sort(key=lambda item: -item[1])
        return [(self._chunks[i], score) for i, score in scored[:top_k]]

    def stats(self) -> Dict[str, float]:
        return {
            "chunk_count": len(self._chunks),
            "vocabulary_size": len(self._doc_freqs),
            "avg_chunk_tokens": round(self._avg_length, 1),
        }
