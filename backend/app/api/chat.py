"""Ask questions about the uploaded documents."""

from typing import Dict, List

from fastapi import APIRouter, Depends

from app.api.deps import existing_session, get_pipeline
from app.config import Settings, get_settings
from app.rag.pipeline import RagPipeline
from app.schemas import ChatRequest, ChatResponse
from app.session import Session

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def ask(
    request: ChatRequest,
    session: Session = Depends(existing_session),
    pipeline: RagPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    # The trace carries chunk text and scores. Useful in the demo UI, but it is
    # debug detail rather than product surface, so production suppresses it
    # regardless of what the client asks for.
    include_trace = request.include_trace and settings.environment != "production"
    return pipeline.answer(
        session=session,
        question=request.question,
        job_id=request.job_id,
        include_trace=include_trace,
    )


@router.get("/history")
async def history(session: Session = Depends(existing_session)) -> Dict[str, List[Dict[str, str]]]:
    return {
        "turns": [
            {"question": question, "answer": answer} for question, answer in session.history
        ]
    }


@router.delete("/history")
async def clear_history(session: Session = Depends(existing_session)) -> Dict[str, str]:
    session.history.clear()
    return {"status": "cleared"}
