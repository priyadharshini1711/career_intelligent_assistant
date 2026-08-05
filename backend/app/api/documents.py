"""Upload, list, and delete documents."""

import pathlib
from typing import Dict, List, Tuple

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import current_session, existing_session, get_embedder, get_provider
from app.config import Settings, get_settings
from app.errors import AppError, ResourceNotFound
from app.llm.base import LLMProvider
from app.rag.embeddings import Embedder
from app.schemas import DocumentKind, DocumentSummary, SessionState, UploadResponse
from app.services import ingest_document
from app.session import Session

router = APIRouter(prefix="/api/documents", tags=["documents"])

#: `samples/` sits next to `backend/` in the repo; the container mounts it at
#: /samples. SAMPLES_DIR in the environment overrides both.
_DEFAULT_SAMPLES_DIR = pathlib.Path(__file__).resolve().parents[3] / "samples"


def samples_dir(settings: Settings) -> pathlib.Path:
    return pathlib.Path(settings.samples_dir) if settings.samples_dir else _DEFAULT_SAMPLES_DIR


def _sample_files(directory: pathlib.Path) -> List[Tuple[pathlib.Path, DocumentKind]]:
    """Bundled samples, resume first so the session is valid as jobs land."""
    resumes = sorted(directory.glob("resume*.txt"))
    jobs = sorted(directory.glob("job*.txt"))
    return [(path, DocumentKind.RESUME) for path in resumes[:1]] + [
        (path, DocumentKind.JOB) for path in jobs
    ]


def _state(session: Session, llm: LLMProvider) -> SessionState:
    return SessionState(
        session_id=session.id,
        resume=session.resume_summary(),
        jobs=session.job_summaries(),
        llm_provider=llm.name,
        llm_model=llm.model,
        ready=session.ready,
    )


def _ingest_many(
    session: Session,
    files: List[UploadFile],
    kind: DocumentKind,
    settings: Settings,
    embedder: Embedder,
    collect_errors: bool = True,
) -> tuple:
    """Ingest a batch, collecting per-file failures instead of failing the lot.

    Someone uploading five job descriptions should not lose four good ones
    because the fifth was a scanned PDF. Failures come back alongside the
    successes with a reason the UI can show against the specific file.

    `collect_errors=False` restores fail-fast behaviour for single-file
    endpoints, where a 200 response listing the one file you sent as "skipped"
    is a worse contract than an error status.
    """
    uploaded: List[DocumentSummary] = []
    skipped: List[Dict[str, str]] = []

    for upload in files:
        data = upload.file.read()
        try:
            document = ingest_document(
                session=session,
                filename=upload.filename or "upload",
                data=data,
                kind=kind,
                settings=settings,
                embedder=embedder,
            )
        except AppError as exc:
            if not collect_errors:
                raise
            skipped.append(
                {"filename": upload.filename or "upload", "code": exc.code, "reason": exc.message}
            )
            continue
        finally:
            upload.file.close()

        uploaded.append(
            DocumentSummary(
                id=document.id,
                kind=document.kind,
                title=document.title,
                filename=document.filename,
                word_count=len(document.text.split()),
                chunk_count=len(document.chunks),
                created_at=document.created_at,
            )
        )

    return uploaded, skipped


@router.post("/resume", response_model=UploadResponse)
async def upload_resume(
    file: UploadFile = File(...),
    session: Session = Depends(current_session),
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    llm: LLMProvider = Depends(get_provider),
) -> UploadResponse:
    """Upload or replace the resume. A session holds exactly one."""
    uploaded, skipped = _ingest_many(
        session, [file], DocumentKind.RESUME, settings, embedder, collect_errors=False
    )
    return UploadResponse(
        session_id=session.id, uploaded=uploaded, skipped=skipped, state=_state(session, llm)
    )


@router.post("/jobs", response_model=UploadResponse)
async def upload_jobs(
    files: List[UploadFile] = File(...),
    session: Session = Depends(current_session),
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    llm: LLMProvider = Depends(get_provider),
) -> UploadResponse:
    """Upload one or more job descriptions."""
    uploaded, skipped = _ingest_many(session, files, DocumentKind.JOB, settings, embedder)
    return UploadResponse(
        session_id=session.id, uploaded=uploaded, skipped=skipped, state=_state(session, llm)
    )


@router.post("/samples", response_model=UploadResponse)
async def load_samples(
    session: Session = Depends(current_session),
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    llm: LLMProvider = Depends(get_provider),
) -> UploadResponse:
    """Load the bundled sample resume and job descriptions.

    Exists so the app can be evaluated in one click. Asking a reviewer to find
    a resume and three job postings before they can see anything work is a
    great way to have the app never get evaluated at all.
    """
    directory = samples_dir(settings)
    if not directory.exists():
        raise ResourceNotFound(
            "Sample documents are not bundled with this deployment.",
            {"expected_path": str(directory)},
        )

    uploaded: List[DocumentSummary] = []
    skipped: List[Dict[str, str]] = []

    for path, kind in _sample_files(directory):
        try:
            document = ingest_document(
                session=session,
                filename=path.name,
                data=path.read_bytes(),
                kind=kind,
                settings=settings,
                embedder=embedder,
            )
        except AppError as exc:
            skipped.append({"filename": path.name, "code": exc.code, "reason": exc.message})
            continue

        uploaded.append(
            DocumentSummary(
                id=document.id,
                kind=document.kind,
                title=document.title,
                filename=document.filename,
                word_count=len(document.text.split()),
                chunk_count=len(document.chunks),
                created_at=document.created_at,
            )
        )

    return UploadResponse(
        session_id=session.id, uploaded=uploaded, skipped=skipped, state=_state(session, llm)
    )


@router.get("", response_model=SessionState)
async def list_documents(
    session: Session = Depends(existing_session),
    llm: LLMProvider = Depends(get_provider),
) -> SessionState:
    return _state(session, llm)


@router.get("/{document_id}/text")
async def document_text(
    document_id: str, session: Session = Depends(existing_session)
) -> Dict[str, object]:
    """Full text of one document, so the UI can show the source behind a citation."""
    document = session.jobs.get(document_id)
    if document is None and session.resume is not None and session.resume.id == document_id:
        document = session.resume
    if document is None:
        raise ResourceNotFound("No such document in this session.", {"document_id": document_id})

    return {
        "id": document.id,
        "title": document.title,
        "kind": document.kind.value,
        "text": document.text,
        "sections": document.meta.get("sections", {}),
        "warnings": session.injection_warnings.get(document_id, []),
    }


@router.delete("/{document_id}", response_model=SessionState)
async def delete_document(
    document_id: str,
    session: Session = Depends(existing_session),
    llm: LLMProvider = Depends(get_provider),
) -> SessionState:
    if not session.remove_document(document_id):
        raise ResourceNotFound("No such document in this session.", {"document_id": document_id})
    return _state(session, llm)
