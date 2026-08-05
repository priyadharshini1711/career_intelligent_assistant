"""Provider selection.

One place that turns configuration into an `LLMProvider`. If a configured
provider cannot be constructed -- missing key, typo in the name -- we fall back
to the stub and log loudly rather than refusing to start. A career assistant
that boots and tells you the model is not configured is more useful than one
that crashes on import, and the `/api/system/health` endpoint reports the
degraded state so it is never silent.
"""

from typing import Optional

from app.config import Settings
from app.errors import LLMUnavailable
from app.llm.base import LLMProvider
from app.llm.gemini import GeminiProvider
from app.llm.groq import GroqProvider
from app.llm.ollama import OllamaProvider
from app.llm.stub import StubProvider
from app.observability import get_logger

logger = get_logger(__name__)


def build_llm(settings: Settings) -> LLMProvider:
    provider_name = settings.llm_provider.lower()

    try:
        if provider_name == "gemini":
            return GeminiProvider(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
            )
        if provider_name == "groq":
            return GroqProvider(
                api_key=settings.groq_api_key,
                model=settings.groq_model,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
            )
        if provider_name == "ollama":
            return OllamaProvider(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                timeout=max(settings.llm_timeout_seconds, 120.0),
                max_retries=settings.llm_max_retries,
            )
        if provider_name == "stub":
            return StubProvider()
    except LLMUnavailable as exc:
        logger.error(
            "llm provider unavailable, falling back to stub",
            extra={"provider": provider_name, "reason": exc.message},
        )
        return StubProvider()

    logger.error(
        "unknown llm provider, falling back to stub", extra={"provider": provider_name}
    )
    return StubProvider()


_cached: Optional[LLMProvider] = None


def get_llm(settings: Settings) -> LLMProvider:
    global _cached
    if _cached is None:
        _cached = build_llm(settings)
        logger.info(
            "llm provider ready",
            extra={"provider": _cached.name, "model": _cached.model},
        )
    return _cached


def reset_llm_cache() -> None:
    """Test hook -- lets a test swap providers without a fresh process."""
    global _cached
    _cached = None
