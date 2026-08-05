"""Health, readiness, and metrics.

Split into liveness and readiness because they answer different questions and
an orchestrator needs both: `/health` says the process is up (never fails on a
degraded dependency, or Kubernetes restarts a pod that is serving fine), while
`/ready` says the dependencies it needs are actually usable.
"""

from typing import Any, Dict

from fastapi import APIRouter, Depends, Response

from app.api.deps import get_embedder, get_provider, get_session_store
from app.config import Settings, get_settings
from app.llm.base import LLMProvider
from app.observability import METRICS
from app.rag.embeddings import Embedder
from app.session import SessionStore

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/health")
async def health(settings: Settings = Depends(get_settings)) -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.environment,
    }


@router.get("/ready")
async def ready(
    response: Response,
    llm: LLMProvider = Depends(get_provider),
    embedder: Embedder = Depends(get_embedder),
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    llm_health = llm.health()

    # The stub is "ready" in the sense that the app works, but it is not a real
    # model, so we report degraded rather than ok. Silent degradation is how a
    # demo ends up presenting templated text as model output.
    degraded = llm.name == "stub" and settings.llm_provider != "stub"
    configured = bool(llm_health.get("configured"))

    payload: Dict[str, Any] = {
        "status": "degraded" if (degraded or not configured) else "ok",
        "llm": llm_health,
        "configured_provider": settings.llm_provider,
        "embedding_model": embedder.name,
        "embedding_dimension": embedder.dimension,
    }
    if degraded:
        payload["note"] = (
            f"'{settings.llm_provider}' was requested but could not be initialised; "
            "falling back to the offline stub. Check the API key."
        )
    if payload["status"] != "ok":
        response.status_code = 503
    return payload


@router.get("/metrics")
async def metrics(store: SessionStore = Depends(get_session_store)) -> Dict[str, Any]:
    """In-process counters.

    Deliberately not Prometheus. At this scale a JSON snapshot is enough to
    demonstrate what is worth measuring -- answer latency, grounding ratio,
    guardrail blocks, invalid citations -- without adding a metrics backend to
    the compose file. The production path is in the README.
    """
    return {"counters": METRICS.snapshot(), "sessions": store.stats()}
