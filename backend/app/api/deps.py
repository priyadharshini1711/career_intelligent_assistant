"""Shared dependencies.

The expensive singletons -- the embedding model and the session registry --
are built once per process and handed to routes through FastAPI's dependency
system, which keeps them out of module-level import side effects and makes
them overridable in tests via `app.dependency_overrides`.
"""

from functools import lru_cache
from typing import Optional

from fastapi import Depends, Header

from app.config import Settings, get_settings
from app.llm.base import LLMProvider
from app.llm.factory import get_llm
from app.rag.embeddings import Embedder, build_embedder
from app.rag.pipeline import RagPipeline
from app.session import Session, SessionStore

SESSION_HEADER = "X-Session-Id"


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    settings = get_settings()
    return build_embedder(settings.embedding_model, settings.embedding_batch_size)


def warm_embedder() -> None:
    """Load the embedding model at startup rather than on the first upload.

    The model loads lazily, which is right for tests and imports but wrong for
    a running server: it made the first upload take 13 seconds while the user
    stared at a spinner, and pushed the cost onto whoever happened to arrive
    first. Paying it during startup is what readiness probes are for.
    """
    get_embedder().encode(["warmup"])


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    settings = get_settings()
    return SessionStore(
        embedding_dimension=get_embedder().dimension,
        ttl_seconds=settings.session_ttl_seconds,
        max_sessions=settings.max_sessions,
    )


def get_provider(settings: Settings = Depends(get_settings)) -> LLMProvider:
    return get_llm(settings)


def get_pipeline(
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    llm: LLMProvider = Depends(get_provider),
) -> RagPipeline:
    return RagPipeline(settings=settings, embedder=embedder, llm=llm)


def current_session(
    x_session_id: Optional[str] = Header(default=None, alias=SESSION_HEADER),
    store: SessionStore = Depends(get_session_store),
) -> Session:
    """Resolve the caller's session, creating one if the header is absent.

    Creating on absence keeps the client simple: the first upload gets a
    session id back in the response and echoes it on every later call.
    """
    return store.get_or_create(x_session_id)


def existing_session(
    x_session_id: Optional[str] = Header(default=None, alias=SESSION_HEADER),
    store: SessionStore = Depends(get_session_store),
) -> Session:
    """Like `current_session`, but refuses to invent one.

    Used by read and chat endpoints, where a missing session means the user's
    documents are gone and they need to be told, not handed a blank slate.
    """
    from app.errors import SessionNotFound

    if not x_session_id:
        raise SessionNotFound("No session provided. Upload your documents first.")
    return store.get(x_session_id)
