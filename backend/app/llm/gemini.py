"""Google Gemini provider -- the default.

Chosen because it is the only frontier-quality model with a *permanent* free
tier that needs no credit card: roughly 1,500 requests/day on the Flash models
at the time of writing. For a project that has to be cloneable and runnable by
a reviewer who will not enter payment details, that constraint dominates.

Flash rather than Pro: the reasoning here is comparison and extraction over a
context we assembled ourselves, not open-ended reasoning. Flash handles that at
a fraction of the latency, and low latency is what makes the chat feel usable.
The 1M-token window is far more than we need -- we deliberately send ~2k words
-- but it removes any risk of the truncation bug that made the first version of
this project produce nonsense.
"""

import time
from typing import Any, Dict, Optional

from app.errors import LLMUnavailable
from app.llm.base import LLMProvider, LLMResponse

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        timeout: float = 60.0,
        max_retries: int = 2,
        thinking_budget: Optional[int] = 0,
    ) -> None:
        super().__init__(model=model, timeout=timeout, max_retries=max_retries)
        if not api_key:
            raise LLMUnavailable(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey, or set LLM_PROVIDER=stub."
            )
        self.api_key = api_key
        self.thinking_budget = thinking_budget

    def _supports_thinking(self) -> bool:
        # The 2.5+ Flash line reasons before answering. We disable it: our
        # prompts do the structuring, and thinking tokens are billed against
        # the same output budget, which can leave the visible answer empty.
        return any(marker in self.model for marker in ("2.5", "3.0", "3.1", "3.5"))

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> LLMResponse:
        generation_config: Dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if self.thinking_budget is not None and self._supports_thinking():
            generation_config["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": generation_config,
        }

        started = time.perf_counter()
        data = self._post_with_retries(
            _ENDPOINT.format(model=self.model),
            payload=payload,
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
        )
        latency_ms = (time.perf_counter() - started) * 1000

        candidates = data.get("candidates") or []
        if not candidates:
            # Usually a safety filter on the prompt. Surface the actual reason
            # rather than an empty answer the user cannot act on.
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates returned")
            raise LLMUnavailable(
                "Gemini returned no answer.", {"reason": reason, "response": str(data)[:400]}
            )

        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts).strip()

        usage = data.get("usageMetadata") or {}
        finish_reason = candidate.get("finishReason", "STOP")

        if not text:
            raise LLMUnavailable(
                "Gemini returned an empty answer.",
                {"finish_reason": finish_reason, "usage": usage},
            )

        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
            finish_reason=str(finish_reason).lower(),
        )

    def health(self) -> Dict[str, Any]:
        return {"provider": self.name, "model": self.model, "configured": bool(self.api_key)}
