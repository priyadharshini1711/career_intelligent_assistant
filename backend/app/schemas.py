"""Pydantic models: the domain objects and the API contract.

Domain objects (`Chunk`, `Document`) are deliberately plain and serialisable so
they can move to a database or a managed vector store later without a rewrite.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DocumentKind(str, Enum):
    RESUME = "resume"
    JOB = "job"


# --------------------------------------------------------------------------
# Domain
# --------------------------------------------------------------------------


class Chunk(BaseModel):
    id: str
    document_id: str
    document_kind: DocumentKind
    document_title: str
    # Resume/JD headings ("Experience", "Requirements", ...) when we can infer
    # them. Carried into the prompt so the model knows what it is reading.
    section: str = "General"
    index: int
    text: str
    word_count: int


class Document(BaseModel):
    id: str
    kind: DocumentKind
    title: str
    filename: str
    text: str
    chunks: List[Chunk] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    meta: Dict[str, Any] = Field(default_factory=dict)


class DocumentSummary(BaseModel):
    """What the UI needs to list a document without shipping its full text."""

    id: str
    kind: DocumentKind
    title: str
    filename: str
    word_count: int
    chunk_count: int
    created_at: datetime


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    dense_score: float
    lexical_score: float


class Citation(BaseModel):
    """A pointer the UI can render and the user can verify against the source."""

    marker: str  # e.g. "R1", "J2" -- what appears inline in the answer
    chunk_id: str
    document_id: str
    document_kind: DocumentKind
    document_title: str
    section: str
    snippet: str
    score: float


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


class SkillEvidence(BaseModel):
    skill: str
    # Where in the resume we found support for this skill, if anywhere.
    resume_snippet: Optional[str] = None
    similarity: float = 0.0


class SkillGap(BaseModel):
    skill: str
    importance: str = "required"  # "required" | "preferred"
    reason: str = ""


class FitComponent(BaseModel):
    name: str
    score: float  # 0-100
    weight: float
    explanation: str


class FitReport(BaseModel):
    job_id: str
    job_title: str
    overall_score: float
    verdict: str
    components: List[FitComponent]
    matched_skills: List[SkillEvidence]
    missing_skills: List[SkillGap]
    partial_skills: List[SkillEvidence]
    generated_at: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------
# API requests / responses
# --------------------------------------------------------------------------


class SessionState(BaseModel):
    session_id: str
    resume: Optional[DocumentSummary] = None
    jobs: List[DocumentSummary] = Field(default_factory=list)
    llm_provider: str
    llm_model: str
    ready: bool = False


class UploadResponse(BaseModel):
    session_id: str
    uploaded: List[DocumentSummary]
    skipped: List[Dict[str, str]] = Field(default_factory=list)
    state: SessionState


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # None = consider every uploaded job (useful for "which role fits me best?")
    job_id: Optional[str] = None
    include_trace: bool = True

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be blank")
        return cleaned


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    grounded: bool = True
    refused: bool = False
    trace: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
