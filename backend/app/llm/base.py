"""The LLM seam.

Every provider is reached through `LLMProvider.complete()`. Nothing above this
layer knows which model answered, which is what makes the choice reversible --
and the choice deserves to stay reversible, because "which model" is the part
of a RAG system most likely to change after it ships.

Providers talk raw HTTP through `httpx` rather than each vendor's SDK. Three
SDKs would mean three dependency trees, three auth conventions, and three sets
of breaking changes, to wrap what is one POST with a JSON body in each case.
The cost of that decision is that we hand-roll request shaping per provider,
which is ~40 lines each and visible in one place.
"""

import abc
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from app.errors import LLMUnavailable
from app.observability import get_logger

logger = get_logger(__name__)

# Status codes worth another attempt: rate limits and transient server errors.
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    finish_reason: str = "stop"

    def usage(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 1),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "finish_reason": self.finish_reason,
        }


class LLMProvider(abc.ABC):
    """A text-in / text-out chat model."""

    name: str = "base"

    def __init__(self, model: str, timeout: float = 60.0, max_retries: int = 2) -> None:
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    @abc.abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> LLMResponse:
        ...

    def health(self) -> Dict[str, Any]:
        """Cheap, non-billable readiness check used by /api/system/health."""
        return {"provider": self.name, "model": self.model, "configured": True}

    # -- shared HTTP plumbing --------------------------------------------

    def _post_with_retries(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """POST JSON with bounded exponential backoff and jitter.

        Free-tier endpoints rate-limit aggressively, so 429 has to be a retry
        rather than a failure or the demo falls over on the third question.
        Jitter matters because without it concurrent requests retry in lockstep
        and hit the same limit again.
        """
        last_error: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload, headers=headers, params=params)

                if response.status_code < 400:
                    return response.json()

                last_error = f"HTTP {response.status_code}: {response.text[:400]}"
                if response.status_code not in _RETRYABLE_STATUS:
                    raise LLMUnavailable(
                        f"{self.name} rejected the request.",
                        {"status": response.status_code, "detail": response.text[:400]},
                    )

            except httpx.TimeoutException as exc:
                last_error = f"timeout after {self.timeout}s"
                if attempt == self.max_retries:
                    raise LLMUnavailable(
                        f"{self.name} timed out. Try again, or switch provider.",
                        {"timeout_seconds": self.timeout},
                    ) from exc
            except httpx.HTTPError as exc:
                last_error = str(exc)
                if attempt == self.max_retries:
                    raise LLMUnavailable(
                        f"Could not reach {self.name}.", {"reason": str(exc)}
                    ) from exc

            if attempt < self.max_retries:
                delay = (2**attempt) * 0.75 + random.uniform(0, 0.4)
                logger.warning(
                    "llm call failed, retrying",
                    extra={
                        "provider": self.name,
                        "attempt": attempt + 1,
                        "delay_s": round(delay, 2),
                        "error": last_error,
                    },
                )
                time.sleep(delay)

        raise LLMUnavailable(
            f"{self.name} failed after {self.max_retries + 1} attempts.",
            {"last_error": last_error},
        )
