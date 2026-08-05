"""Deterministic analysis endpoints: fit reports and skill gaps.

These do not call the LLM. That is the point -- the numbers a user is going to
screenshot and act on are computed from their documents, reproducibly, and can
be traced back to a specific line. Chat handles the parts that genuinely need
language; this handles the parts that need arithmetic.
"""

from typing import Dict, List

from fastapi import APIRouter, Depends

from app.api.deps import existing_session, get_pipeline
from app.rag.pipeline import RagPipeline
from app.schemas import FitReport
from app.session import Session

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


@router.get("/fit/{job_id}", response_model=FitReport)
async def fit_for_job(
    job_id: str,
    session: Session = Depends(existing_session),
    pipeline: RagPipeline = Depends(get_pipeline),
) -> FitReport:
    return pipeline.fit_report(session, job_id)


@router.get("/fit", response_model=List[FitReport])
async def fit_for_all_jobs(
    session: Session = Depends(existing_session),
    pipeline: RagPipeline = Depends(get_pipeline),
) -> List[FitReport]:
    """Every uploaded job, ranked best-fit first."""
    return pipeline.all_fit_reports(session)


@router.get("/gaps/{job_id}")
async def gaps_for_job(
    job_id: str,
    session: Session = Depends(existing_session),
    pipeline: RagPipeline = Depends(get_pipeline),
) -> Dict[str, object]:
    """Skill gaps only -- a lighter payload for the gap panel."""
    report = pipeline.fit_report(session, job_id)
    return {
        "job_id": report.job_id,
        "job_title": report.job_title,
        "missing": [gap.model_dump() for gap in report.missing_skills],
        "partial": [evidence.model_dump() for evidence in report.partial_skills],
        "matched": [evidence.model_dump() for evidence in report.matched_skills],
    }
