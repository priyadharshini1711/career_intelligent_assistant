"""Document ingestion: bytes in, indexed `Document` out.

Sits between the HTTP layer and the RAG internals so the routes stay thin and
the whole upload path is testable without FastAPI.
"""

import re
import uuid

from app.config import Settings
from app.guardrails import scan_document_for_injection
from app.ingestion.chunking import chunk_document, section_stats
from app.ingestion.extract import extract_text, validate_upload
from app.observability import METRICS, get_logger
from app.rag.embeddings import Embedder
from app.schemas import Document, DocumentKind
from app.session import Session

logger = get_logger(__name__)

# "Job Title: Senior Backend Engineer" / "Position - Data Analyst"
_TITLE_LABEL = re.compile(
    r"^\s*(?:job\s*title|position|role|vacancy)\s*[:\-]\s*(.+)$", re.IGNORECASE | re.MULTILINE
)
_SENIORITY_HINT = re.compile(
    r"\b(engineer|developer|analyst|scientist|manager|designer|architect|consultant|"
    r"lead|specialist|administrator|director|intern|associate)\b",
    re.IGNORECASE,
)


def infer_job_title(text: str, filename: str) -> str:
    """Best-effort job title.

    Order of preference: an explicit "Job Title:" label, then the first early
    line that reads like a role name, then the filename. The filename is the
    fallback rather than the default because "jd_final_v3.pdf" tells the user
    nothing when they are looking at three postings side by side.
    """
    labelled = _TITLE_LABEL.search(text)
    if labelled:
        return labelled.group(1).strip()[:80]

    for line in text.split("\n")[:12]:
        stripped = line.strip()
        if 2 <= len(stripped.split()) <= 10 and _SENIORITY_HINT.search(stripped):
            return stripped.strip(":-").strip()[:80]

    stem = re.sub(r"[_-]+", " ", filename.rsplit(".", 1)[0]).strip()
    return (stem[:80] or "Job description").title()


def ingest_document(
    session: Session,
    filename: str,
    data: bytes,
    kind: DocumentKind,
    settings: Settings,
    embedder: Embedder,
) -> Document:
    """Validate, parse, chunk, embed, and index one uploaded file."""
    suffix = validate_upload(
        filename=filename,
        size_bytes=len(data),
        allowed=settings.allowed_extensions,
        max_bytes=settings.max_upload_bytes,
    )

    extracted = extract_text(filename=filename, data=data, suffix=suffix)
    document_id = uuid.uuid4().hex[:12]

    title = (
        "Resume"
        if kind == DocumentKind.RESUME
        else infer_job_title(extracted.text, filename)
    )

    chunks = chunk_document(
        document_id=document_id,
        kind=kind,
        title=title,
        text=extracted.text,
        target_words=settings.chunk_target_words,
        overlap_words=settings.chunk_overlap_words,
        min_words=settings.chunk_min_words,
    )

    document = Document(
        id=document_id,
        kind=kind,
        title=title,
        filename=filename,
        text=extracted.text,
        chunks=chunks,
        meta={**extracted.meta, "sections": section_stats(chunks)},
    )

    # Uploaded files are untrusted input. We index them either way -- refusing
    # someone's real resume over a false positive would be far worse than the
    # risk, which the prompt fencing already handles -- but we record it.
    warnings = scan_document_for_injection(extracted.text)
    if warnings:
        session.injection_warnings[document_id] = warnings
        METRICS.increment("injection_warnings")
        logger.warning(
            "uploaded document contains instruction-like text",
            extra={"document_id": document_id, "filename": filename, "count": len(warnings)},
        )

    if kind == DocumentKind.RESUME:
        session.set_resume(document)
    else:
        session.add_job(document, limit=settings.max_jobs_per_session)

    # Embed the section-prefixed text, not the raw chunk: the structural label
    # is part of what makes a chunk retrievable.
    from app.ingestion.chunking import embedding_text

    vectors = embedder.encode([embedding_text(chunk) for chunk in chunks])
    session.store.add(chunks, vectors)

    METRICS.increment(f"documents_ingested_{kind.value}")
    logger.info(
        "document ingested",
        extra={
            "document_id": document_id,
            "kind": kind.value,
            "title": title,
            "words": extracted.meta.get("word_count"),
            "chunks": len(chunks),
            "sections": list(section_stats(chunks)),
        },
    )
    return document
