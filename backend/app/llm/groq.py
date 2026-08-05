"""Groq provider -- the alternate.

Kept as a first-class option for two reasons. It is a second free tier with no
card required, so a reviewer who cannot get a Gemini key is not blocked. And it
serves open-weight Llama models on custom silicon at genuinely unusual speed,
which is the interesting counterpoint to Gemini: same interface, open weights,
different latency profile.

The API is OpenAI-compatible, so this class also works unchanged against
OpenAI, Together, Fireworks, or a vLLM server by changing the base URL.
"""

import time
from typing import Any, Dict

from app.errors import LLMUnavailable
from app.llm.base import LLMProvider, LLMResponse

_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(model=model, timeout=timeout, max_retries=max_retries)
        if not api_key:
            raise LLMUnavailable(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys, "
                "or set LLM_PROVIDER=stub."
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        started = time.perf_counter()
        data = self._post_with_retries(
            f"{self.base_url}/chat/completions",
            payload=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        latency_ms = (time.perf_counter() - started) * 1000

        choices = data.get("choices") or []
        if not choices:
            raise LLMUnavailable("Groq returned no answer.", {"response": str(data)[:400]})

        message = choices[0].get("message") or {}
        text = (message.get("content") or "").strip()
        if not text:
            raise LLMUnavailable(
                "Groq returned an empty answer.",
                {"finish_reason": choices[0].get("finish_reason")},
            )

        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            finish_reason=choices[0].get("finish_reason", "stop"),
        )

    def health(self) -> Dict[str, Any]:
        return {"provider": self.name, "model": self.model, "configured": bool(self.api_key)}
