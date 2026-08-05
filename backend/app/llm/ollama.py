"""Ollama provider -- the fully local option.

Present because "your resume never leaves your laptop" is a real requirement
for this particular product, not a hypothetical one. People are reluctant to
paste a CV into a hosted model, and with Ollama the whole pipeline -- parsing,
embedding, generation -- runs offline. Quality on an 8B model is a step below
the hosted options, which is the trade being made explicitly.
"""

import time
from typing import Any, Dict

import httpx

from app.errors import LLMUnavailable
from app.llm.base import LLMProvider, LLMResponse


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: float = 120.0,
        max_retries: int = 1,
    ) -> None:
        # Local generation on CPU is slow; a short timeout here would fail
        # requests that were about to succeed.
        super().__init__(model=model, timeout=timeout, max_retries=max_retries)
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
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        started = time.perf_counter()
        data = self._post_with_retries(f"{self.base_url}/api/chat", payload=payload)
        latency_ms = (time.perf_counter() - started) * 1000

        text = ((data.get("message") or {}).get("content") or "").strip()
        if not text:
            raise LLMUnavailable("Ollama returned an empty answer.", {"response": str(data)[:400]})

        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            finish_reason=data.get("done_reason", "stop"),
        )

    def health(self) -> Dict[str, Any]:
        """Ask the daemon whether the configured model is actually pulled.

        A missing `ollama pull` is by far the most common setup failure, and
        it is much friendlier to catch it at /health than on the first question.
        """
        info: Dict[str, Any] = {"provider": self.name, "model": self.model, "configured": False}
        try:
            with httpx.Client(timeout=3.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            available = [m.get("name", "") for m in response.json().get("models", [])]
            info["configured"] = any(
                name == self.model or name.split(":")[0] == self.model.split(":")[0]
                for name in available
            )
            info["available_models"] = available
            if not info["configured"]:
                info["hint"] = f"run: ollama pull {self.model}"
        except Exception as exc:
            info["error"] = f"Ollama not reachable at {self.base_url}: {exc}"
        return info
