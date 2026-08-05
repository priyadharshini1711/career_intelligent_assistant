"""The index and the retriever.

The index tests exist mostly to pin down one invariant: chunks, vectors and
lexical stats stay index-aligned through every mutation. Breaking that produces
citations pointing at the wrong document -- an answer that looks sourced and is
not, which is the worst failure this system can have.
"""

import numpy as np
import pytest

from app.ingestion.chunking import chunk_document
from app.ingestion.extract import normalize
from app.rag.embeddings import HashingEmbedder, build_embedder
from app.rag.intent import Intent, classify
from app.rag.retriever import Retriever
from app.rag.store import InMemoryChunkStore, tokenize
from app.schemas import DocumentKind
from tests.conftest import JOB_BACKEND_TEXT, JOB_ML_TEXT, RESUME_TEXT


@pytest.fixture
def populated_store(embedder):
    store = InMemoryChunkStore(dimension=embedder.dimension)
    for doc_id, kind, title, text in [
        ("res1", DocumentKind.RESUME, "Resume", RESUME_TEXT),
        ("job1", DocumentKind.JOB, "Backend Engineer", JOB_BACKEND_TEXT),
        ("job2", DocumentKind.JOB, "ML Engineer", JOB_ML_TEXT),
    ]:
        chunks = chunk_document(doc_id, kind, title, normalize(text))
        store.add(chunks, embedder.encode([chunk.text for chunk in chunks]))
    return store


class TestTokenizer:
    def test_keeps_symbol_bearing_tokens(self):
        tokens = tokenize("Experience with C++, C#, Node.js and Go")
        assert "c++" in tokens
        assert "c#" in tokens
        assert "node.js" in tokens

    def test_drops_stopwords(self):
        assert "the" not in tokenize("the python developer")


class TestInMemoryChunkStore:
    def test_rejects_mismatched_chunk_and_vector_counts(self, embedder):
        store = InMemoryChunkStore(dimension=embedder.dimension)
        chunks = chunk_document("d", DocumentKind.RESUME, "R", normalize(RESUME_TEXT))
        with pytest.raises(ValueError):
            store.add(chunks, embedder.encode([chunks[0].text]))

    def test_dense_search_respects_document_filter(self, populated_store, embedder):
        query = embedder.encode(["payments and postgresql"])[0]
        results = populated_store.dense_search(query, ["job1"], top_k=5)
        assert results
        assert all(chunk.document_id == "job1" for chunk, _ in results)

    def test_lexical_search_finds_exact_tokens(self, populated_store):
        """The case dense retrieval is bad at: a specific named tool."""
        results = populated_store.lexical_search("pinecone weaviate qdrant", None, top_k=3)
        assert results
        assert results[0][0].document_id == "job2"

    def test_removal_keeps_arrays_aligned(self, populated_store, embedder):
        before = len(populated_store)
        removed = populated_store.remove_document("job1")
        assert removed > 0
        assert len(populated_store) == before - removed
        assert all(chunk.document_id != "job1" for chunk in populated_store.chunks())

        # After a rebuild, every chunk must still be described by its own
        # vector. Cross-checking each chunk against itself catches a shifted
        # index that a length check would miss.
        query_chunks = populated_store.chunks()
        vectors = embedder.encode([chunk.text for chunk in query_chunks])
        for chunk, vector in zip(query_chunks, vectors):
            top = populated_store.dense_search(vector, None, top_k=1)
            assert top[0][0].id == chunk.id

    def test_removing_unknown_document_is_a_noop(self, populated_store):
        assert populated_store.remove_document("nope") == 0

    def test_empty_store_returns_nothing(self, embedder):
        store = InMemoryChunkStore(dimension=embedder.dimension)
        assert store.dense_search(np.zeros(embedder.dimension, np.float32), None, 5) == []
        assert store.lexical_search("python", None, 5) == []


class TestIntentClassification:
    @pytest.mark.parametrize(
        "question,expected",
        [
            ("What skills am I missing for this role?", Intent.SKILL_GAP),
            ("What gaps do I have?", Intent.SKILL_GAP),
            ("How does my experience align with Job 2?", Intent.ALIGNMENT),
            ("Am I a good fit for this position?", Intent.ALIGNMENT),
            ("What should I prepare for the interview?", Intent.INTERVIEW_PREP),
            ("How should I rewrite my resume for this job?", Intent.RESUME_IMPROVEMENT),
            ("Which of these roles fits me best?", Intent.COMPARISON),
            ("Compare these two postings", Intent.COMPARISON),
            ("What does the company do?", Intent.GENERAL),
        ],
    )
    def test_classifies_representative_questions(self, question, expected):
        assert classify(question) == expected


class TestRetriever:
    def _retriever(self, store, embedder):
        return Retriever(store=store, embedder=embedder, top_k=6, min_score=embedder.min_relevance)

    def test_returns_both_resume_and_job_context(self, populated_store, embedder):
        """The central retrieval guarantee: never one side of the comparison."""
        hits = self._retriever(populated_store, embedder).retrieve(
            query="What skills am I missing for this role?",
            intent=Intent.SKILL_GAP,
            resume_id="res1",
            job_ids=["job1"],
        )
        kinds = {hit.chunk.document_kind for hit in hits}
        assert kinds == {DocumentKind.RESUME, DocumentKind.JOB}

    def test_scopes_to_the_requested_job(self, populated_store, embedder):
        hits = self._retriever(populated_store, embedder).retrieve(
            query="what does this role require",
            intent=Intent.SKILL_GAP,
            resume_id="res1",
            job_ids=["job1"],
        )
        job_docs = {
            hit.chunk.document_id for hit in hits if hit.chunk.document_kind == DocumentKind.JOB
        }
        assert job_docs == {"job1"}

    def test_gives_every_job_a_share_when_comparing(self, populated_store, embedder):
        hits = self._retriever(populated_store, embedder).retrieve(
            query="which role suits me best",
            intent=Intent.COMPARISON,
            resume_id="res1",
            job_ids=["job1", "job2"],
        )
        job_docs = {
            hit.chunk.document_id for hit in hits if hit.chunk.document_kind == DocumentKind.JOB
        }
        # A verbose posting must not crowd the other out of the comparison.
        assert job_docs == {"job1", "job2"}

    def test_deduplicates_overlapping_chunks(self, populated_store, embedder):
        hits = self._retriever(populated_store, embedder).retrieve(
            query="python postgresql experience",
            intent=Intent.ALIGNMENT,
            resume_id="res1",
            job_ids=["job1"],
        )
        assert len({hit.chunk.id for hit in hits}) == len(hits)

    def test_reports_weak_evidence_when_nothing_matches(self, embedder):
        store = InMemoryChunkStore(dimension=embedder.dimension)
        retriever = self._retriever(store, embedder)
        assert retriever.is_weak([])

    def test_lexical_hit_alone_is_not_weak_evidence(self, populated_store, embedder):
        """Regression: gating on the dense score alone discarded good BM25 hits."""
        retriever = self._retriever(populated_store, embedder)
        hits = retriever.retrieve(
            query="Do they want Kafka or Pinecone experience?",
            intent=Intent.GENERAL,
            resume_id="res1",
            job_ids=["job1", "job2"],
        )
        assert hits
        assert max(hit.lexical_score for hit in hits) > 0
        assert not retriever.is_weak(hits)


class TestHashingEmbedder:
    def test_is_deterministic(self):
        first = HashingEmbedder().encode(["python developer"])
        second = HashingEmbedder().encode(["python developer"])
        np.testing.assert_allclose(first, second)

    def test_vectors_are_normalised(self):
        vectors = HashingEmbedder().encode(["python", "postgresql and redis"])
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)

    def test_similar_text_scores_higher_than_unrelated(self):
        embedder = HashingEmbedder()
        vectors = embedder.encode(
            ["python backend engineer", "python backend developer", "pastry chef patisserie"]
        )
        assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])

    def test_build_embedder_selects_by_name(self):
        assert build_embedder("hashing").name == "hashing"
