"""Shared test fixtures.

The whole suite runs offline and deterministically: the hashing embedder needs
no model download, and the stub LLM needs no API key. That is the point of
having both -- a test suite that reaches the network is a test suite that fails
on someone else's laptop.
"""

import os
import pathlib
import sys

import pytest

# Import the app package from `backend/` regardless of where pytest was invoked.
BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

# Must be set before app.config is first imported: Settings is cached.
os.environ.setdefault("EMBEDDING_MODEL", "hashing")
os.environ.setdefault("LLM_PROVIDER", "stub")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from app.config import get_settings  # noqa: E402
from app.ingestion.chunking import chunk_document  # noqa: E402
from app.rag.embeddings import build_embedder  # noqa: E402
from app.schemas import Document, DocumentKind  # noqa: E402

SAMPLES = BACKEND_ROOT.parent / "samples"

RESUME_TEXT = (SAMPLES / "resume_priya_backend.txt").read_text(encoding="utf-8")
JOB_BACKEND_TEXT = (SAMPLES / "job_backend_engineer.txt").read_text(encoding="utf-8")
JOB_PLATFORM_TEXT = (SAMPLES / "job_platform_engineer.txt").read_text(encoding="utf-8")
JOB_ML_TEXT = (SAMPLES / "job_ml_engineer.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def settings():
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture(scope="session")
def embedder():
    return build_embedder("hashing")


def make_document(doc_id: str, kind: DocumentKind, title: str, text: str) -> Document:
    chunks = chunk_document(document_id=doc_id, kind=kind, title=title, text=text)
    return Document(id=doc_id, kind=kind, title=title, filename=f"{doc_id}.txt", text=text, chunks=chunks)


@pytest.fixture
def resume_doc() -> Document:
    return make_document("res1", DocumentKind.RESUME, "Resume", RESUME_TEXT)


@pytest.fixture
def backend_job_doc() -> Document:
    return make_document("job1", DocumentKind.JOB, "Senior Backend Engineer, Payments", JOB_BACKEND_TEXT)


@pytest.fixture
def platform_job_doc() -> Document:
    return make_document("job2", DocumentKind.JOB, "Senior Platform Engineer", JOB_PLATFORM_TEXT)


@pytest.fixture
def ml_job_doc() -> Document:
    return make_document("job3", DocumentKind.JOB, "Machine Learning Engineer", JOB_ML_TEXT)


@pytest.fixture
def client():
    """A TestClient with a fresh session store per test."""
    from fastapi.testclient import TestClient

    from app.api.deps import get_embedder, get_session_store
    from app.main import create_app

    get_embedder.cache_clear()
    get_session_store.cache_clear()
    return TestClient(create_app())


@pytest.fixture
def uploaded(client):
    """A session with the sample resume and all three sample job descriptions."""
    response = client.post(
        "/api/documents/resume",
        files={"file": ("resume.txt", RESUME_TEXT.encode(), "text/plain")},
    )
    assert response.status_code == 200
    session_id = response.json()["session_id"]
    headers = {"X-Session-Id": session_id}

    files = [
        ("files", ("job_backend.txt", JOB_BACKEND_TEXT.encode(), "text/plain")),
        ("files", ("job_platform.txt", JOB_PLATFORM_TEXT.encode(), "text/plain")),
        ("files", ("job_ml.txt", JOB_ML_TEXT.encode(), "text/plain")),
    ]
    response = client.post("/api/documents/jobs", files=files, headers=headers)
    assert response.status_code == 200
    jobs = response.json()["uploaded"]

    return {"client": client, "headers": headers, "session_id": session_id, "jobs": jobs}
