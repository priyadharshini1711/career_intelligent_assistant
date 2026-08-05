"""Session state.

Choice: in-process, TTL'd, evicted LRU. No database.

Reasoning -- the product is "upload some documents, ask about them, leave".
There is no account, nothing worth persisting past the visit, and a resume is
data you should not keep without a reason. Ephemeral state is the privacy-
respecting default here, not a shortcut.

What it costs, stated plainly: state dies on restart, and the app cannot run as
more than one replica because a user's second request could land on a process
that has never seen their upload. Both are disqualifying for production, and
both are fixed the same way -- move `Session` behind Redis or Postgres and the
chunk index into a shared vector store. The class is deliberately a plain data
holder so that move is mechanical. The README covers it.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.errors import LimitExceeded, SessionNotFound
from app.observability import get_logger
from app.rag.store import InMemoryChunkStore
from app.schemas import Document, DocumentSummary, FitReport

logger = get_logger(__name__)


def _summarise(document: Document) -> DocumentSummary:
    return DocumentSummary(
        id=document.id,
        kind=document.kind,
        title=document.title,
        filename=document.filename,
        word_count=len(document.text.split()),
        chunk_count=len(document.chunks),
        created_at=document.created_at,
    )


@dataclass
class Session:
    id: str
    store: InMemoryChunkStore
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    resume: Optional[Document] = None
    jobs: Dict[str, Document] = field(default_factory=dict)
    history: List[Tuple[str, str]] = field(default_factory=list)
    # Fit reports are pure functions of (resume, job) and cost a few hundred
    # embeddings, so they are cached until one of the two documents changes.
    fit_cache: Dict[str, FitReport] = field(default_factory=dict)
    injection_warnings: Dict[str, List[str]] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_seen = time.time()

    @property
    def ready(self) -> bool:
        return self.resume is not None and bool(self.jobs)

    def job_list(self) -> List[Document]:
        return sorted(self.jobs.values(), key=lambda job: job.created_at)

    def set_resume(self, document: Document) -> None:
        if self.resume is not None:
            self.store.remove_document(self.resume.id)
        self.resume = document
        # Every cached report compared against the old resume.
        self.fit_cache.clear()

    def add_job(self, document: Document, limit: int) -> None:
        if len(self.jobs) >= limit:
            raise LimitExceeded(
                f"You can compare up to {limit} job descriptions at a time. "
                "Remove one before adding another.",
                {"limit": limit},
            )
        self.jobs[document.id] = document

    def remove_document(self, document_id: str) -> bool:
        if self.resume is not None and self.resume.id == document_id:
            self.store.remove_document(document_id)
            self.resume = None
            self.fit_cache.clear()
            return True
        if document_id in self.jobs:
            self.store.remove_document(document_id)
            del self.jobs[document_id]
            self.fit_cache.pop(document_id, None)
            return True
        return False

    def record_turn(self, question: str, answer: str, max_turns: int) -> None:
        self.history.append((question, answer))
        if len(self.history) > max_turns:
            self.history = self.history[-max_turns:]

    def resume_summary(self) -> Optional[DocumentSummary]:
        return _summarise(self.resume) if self.resume else None

    def job_summaries(self) -> List[DocumentSummary]:
        return [_summarise(job) for job in self.job_list()]


class SessionStore:
    """Thread-safe session registry with TTL and a hard cap."""

    def __init__(self, embedding_dimension: int, ttl_seconds: int, max_sessions: int) -> None:
        self.embedding_dimension = embedding_dimension
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def _new_session(self) -> Session:
        return Session(
            id=uuid.uuid4().hex,
            store=InMemoryChunkStore(dimension=self.embedding_dimension),
        )

    def _evict_locked(self) -> None:
        now = time.time()
        expired = [
            key
            for key, session in self._sessions.items()
            if now - session.last_seen > self.ttl_seconds
        ]
        for key in expired:
            del self._sessions[key]
        if expired:
            logger.info("evicted expired sessions", extra={"count": len(expired)})

        # Hard cap as a memory backstop: documents live in RAM, so an unbounded
        # session table is an unbounded heap.
        overflow = len(self._sessions) - self.max_sessions
        if overflow > 0:
            oldest = sorted(self._sessions.items(), key=lambda item: item[1].last_seen)
            for key, _ in oldest[:overflow]:
                del self._sessions[key]
            logger.warning("evicted sessions over capacity", extra={"count": overflow})

    def get_or_create(self, session_id: Optional[str]) -> Session:
        with self._lock:
            self._evict_locked()
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.touch()
                return session

            session = self._new_session()
            self._sessions[session.id] = session
            logger.info("session created", extra={"session_id": session.id})
            return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            self._evict_locked()
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFound(
                    "That session has expired or does not exist. Upload your documents again.",
                    {"session_id": session_id},
                )
            session.touch()
            return session

    def drop(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "documents": sum(
                    len(session.jobs) + (1 if session.resume else 0)
                    for session in self._sessions.values()
                ),
            }
