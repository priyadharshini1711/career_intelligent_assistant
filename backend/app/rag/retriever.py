"""Hybrid retrieval.

Three decisions worth stating, because they are the ones that actually changed
answer quality:

1. **Hybrid, not pure vector.** Dense retrieval understands paraphrase
   ("led a team" ~ "managed engineers") but is unreliable on the exact tokens
   this domain turns on: `Kafka` vs `Kinesis`, `PyTorch` vs `TensorFlow`,
   `Python 3.11`. BM25 nails those and misses the paraphrase. Fusing the two
   with Reciprocal Rank Fusion covers both failure modes.

2. **RRF rather than a weighted sum of scores.** Cosine lives in [-1, 1] and
   BM25 is unbounded and corpus-dependent, so summing them requires
   normalising against a candidate set whose scale shifts with every query.
   RRF only uses ranks, so it is stable without tuning. The weight knob is
   preserved by weighting each channel's RRF contribution.

3. **A retrieval budget split across documents, not one pooled ranked list.**
   Every question here is comparative -- it needs JD requirements *and* resume
   evidence in the same context. A single ranked list frequently returns six
   chunks from whichever document happens to phrase things closest to the
   question, and the model then hallucinates the other half. Retrieving each
   side under its own quota makes that structurally impossible.
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from app.observability import get_logger
from app.rag.embeddings import Embedder
from app.rag.intent import RESUME_SHARE, SECTION_PRIORS, Intent
from app.rag.store import ChunkStore, tokenize
from app.schemas import Chunk, DocumentKind, RetrievedChunk

logger = get_logger(__name__)

# RRF constant. 60 is the value from the original paper and the de-facto
# default; it damps the influence of the very top rank enough that a single
# channel cannot dominate the fusion.
_RRF_K = 60.0

# Maximum multiplier a section prior can apply. Small on purpose: the prior
# encodes "requirements sections usually matter for gap questions", which is a
# tie-breaker, not evidence.
_SECTION_BOOST = 0.15


def _rrf(rank: int) -> float:
    return 1.0 / (_RRF_K + rank)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    return intersection / (len(a) + len(b) - intersection)


def _fuse(
    dense: List[Tuple[Chunk, float]],
    lexical: List[Tuple[Chunk, float]],
    dense_weight: float,
    section_prior: Dict[str, float],
) -> List[Tuple[Chunk, float, float, float]]:
    """Weighted RRF over the two channels. Returns (chunk, fused, dense, lexical)."""
    fused: Dict[str, float] = {}
    dense_scores: Dict[str, float] = {}
    lexical_scores: Dict[str, float] = {}
    by_id: Dict[str, Chunk] = {}

    for rank, (chunk, score) in enumerate(dense):
        by_id[chunk.id] = chunk
        dense_scores[chunk.id] = score
        fused[chunk.id] = fused.get(chunk.id, 0.0) + dense_weight * _rrf(rank)

    for rank, (chunk, score) in enumerate(lexical):
        by_id[chunk.id] = chunk
        lexical_scores[chunk.id] = score
        fused[chunk.id] = fused.get(chunk.id, 0.0) + (1.0 - dense_weight) * _rrf(rank)

    results = []
    for chunk_id, score in fused.items():
        chunk = by_id[chunk_id]
        boost = 1.0 + _SECTION_BOOST * section_prior.get(chunk.section, 0.0)
        results.append(
            (
                chunk,
                score * boost,
                dense_scores.get(chunk_id, 0.0),
                lexical_scores.get(chunk_id, 0.0),
            )
        )

    results.sort(key=lambda item: -item[1])
    return results


def _mmr_select(
    candidates: List[Tuple[Chunk, float, float, float]],
    limit: int,
    lambda_: float,
) -> List[Tuple[Chunk, float, float, float]]:
    """Maximal Marginal Relevance over token overlap.

    Chunks overlap by design (the chunker carries 30 words across boundaries),
    so the raw top-k often contains three near-copies of the same requirement
    list. That wastes the context budget on redundancy. Similarity is measured
    with token Jaccard rather than cosine because the thing being suppressed
    here is literal near-duplication, which token overlap detects directly --
    and it avoids having to hold the vectors alongside the results.
    """
    if limit <= 0 or not candidates:
        return []

    token_sets = {chunk.id: set(tokenize(chunk.text)) for chunk, _, _, _ in candidates}
    remaining = list(candidates)
    selected: List[Tuple[Chunk, float, float, float]] = []

    while remaining and len(selected) < limit:
        best_index = 0
        best_value = -np.inf
        for index, item in enumerate(remaining):
            redundancy = max(
                (_jaccard(token_sets[item[0].id], token_sets[chosen[0].id]) for chosen in selected),
                default=0.0,
            )
            value = lambda_ * item[1] - (1 - lambda_) * redundancy * item[1]
            if value > best_value:
                best_value, best_index = value, index
        selected.append(remaining.pop(best_index))

    return selected


class Retriever:
    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder,
        top_k: int = 6,
        candidate_k: int = 20,
        dense_weight: float = 0.65,
        mmr_lambda: float = 0.6,
        min_score: float = 0.05,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.dense_weight = dense_weight
        self.mmr_lambda = mmr_lambda
        self.min_score = min_score

    def _retrieve_side(
        self,
        query: str,
        query_vector: np.ndarray,
        document_ids: Sequence[str],
        limit: int,
        section_prior: Dict[str, float],
    ) -> List[RetrievedChunk]:
        if not document_ids or limit <= 0:
            return []

        dense = self.store.dense_search(query_vector, document_ids, self.candidate_k)
        lexical = self.store.lexical_search(query, document_ids, self.candidate_k)
        fused = _fuse(dense, lexical, self.dense_weight, section_prior)
        chosen = _mmr_select(fused, limit, self.mmr_lambda)

        return [
            RetrievedChunk(
                chunk=chunk,
                score=round(float(fused_score), 6),
                dense_score=round(float(dense_score), 4),
                lexical_score=round(float(lexical_score), 4),
            )
            for chunk, fused_score, dense_score, lexical_score in chosen
        ]

    def retrieve(
        self,
        query: str,
        intent: Intent,
        resume_id: Optional[str],
        job_ids: Sequence[str],
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve a budget-balanced context for one question."""
        budget = top_k or self.top_k
        if budget <= 0 or (not resume_id and not job_ids):
            return []

        query_vector = self.embedder.encode([query])[0]
        section_prior = SECTION_PRIORS.get(intent, {})

        resume_budget = round(budget * RESUME_SHARE.get(intent, 0.5)) if resume_id else 0
        job_budget = budget - resume_budget
        if not job_ids:
            resume_budget, job_budget = budget, 0

        # With several jobs in play (a "which role suits me best?" question)
        # give each one its own quota so one verbose posting cannot crowd the
        # others out of the comparison.
        job_hits: List[RetrievedChunk] = []
        if job_budget > 0 and job_ids:
            per_job = max(1, job_budget // len(job_ids))
            for job_id in job_ids:
                job_hits.extend(
                    self._retrieve_side(query, query_vector, [job_id], per_job, section_prior)
                )
            job_hits.sort(key=lambda hit: -hit.score)
            job_hits = job_hits[: max(job_budget, len(job_ids))]

        resume_hits = self._retrieve_side(
            query, query_vector, [resume_id] if resume_id else [], resume_budget, section_prior
        )

        results = resume_hits + job_hits

        # A weak best-match means the question probably is not answerable from
        # these documents. Report it rather than silently returning noise --
        # the pipeline turns this into an honest "not in your documents".
        results.sort(key=lambda hit: (hit.chunk.document_kind != DocumentKind.RESUME, -hit.score))
        return results

    def max_dense_score(self, hits: Sequence[RetrievedChunk]) -> float:
        return max((hit.dense_score for hit in hits), default=0.0)

    def is_weak(self, hits: Sequence[RetrievedChunk]) -> bool:
        """True when nothing retrieved is plausibly on-topic.

        Both channels have to fail. Gating on the dense score alone discards
        questions that BM25 answered perfectly well -- "do they want Kafka?"
        is a strong lexical hit and a mediocre semantic one, and treating that
        as "no evidence" was a real bug caught in testing.
        """
        if not hits:
            return True
        weak_dense = self.max_dense_score(hits) < self.min_score
        weak_lexical = max((hit.lexical_score for hit in hits), default=0.0) <= 0.0
        return weak_dense and weak_lexical
