"""Deterministic stub provider.

This is not a toy. It does two jobs that matter:

1. **Tests run offline and deterministically.** Asserting on a real model's
   prose is either flaky or vacuous. The stub makes the *pipeline* testable --
   retrieval feeding the prompt, citations surviving the round trip, guardrails
   firing -- without asserting on generated text at all.

2. **The app boots with no API key.** `git clone && uvicorn` gives a reviewer a
   working UI immediately: uploads parse, retrieval runs, the trace inspector
   is populated, the fit report is real (it is computed, not generated). Only
   the prose is templated, and it says so.

It answers by extracting and reorganising the retrieved context, so its output
is grounded by construction -- which also makes it a useful control when
judging whether a real model is adding value over "just show me the chunks".
"""

import re
import time
from typing import Dict, List, Tuple

from app.llm.base import LLMProvider, LLMResponse

# Matches the context block header emitted by prompts.render_context:
#   [J1] (Senior Backend Engineer | Requirements)
_BLOCK_RE = re.compile(r"^\[([RJ]\d+)\]\s+\((.+?)\)\s*$", re.MULTILINE)


class StubProvider(LLMProvider):
    name = "stub"

    def __init__(self, model: str = "deterministic-stub", **_: object) -> None:
        super().__init__(model=model, timeout=1.0, max_retries=0)

    @staticmethod
    def _parse_context(user_prompt: str) -> List[Tuple[str, str, str]]:
        """Recover (marker, source label, text) triples from the rendered prompt."""
        blocks: List[Tuple[str, str, str]] = []
        matches = list(_BLOCK_RE.finditer(user_prompt))
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(user_prompt)
            body = user_prompt[start:end].strip()
            # Stop at the next prompt section header rather than swallowing it.
            body = re.split(r"\n#{2,3}\s", body)[0].strip()
            blocks.append((match.group(1), match.group(2), body))
        return blocks

    @staticmethod
    def _extract_question(user_prompt: str) -> str:
        match = re.search(r"(?:^|\n)#{2,3}\s*Question\s*\n+(.+)", user_prompt)
        return match.group(1).strip() if match else "your question"

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> LLMResponse:
        started = time.perf_counter()

        blocks = self._parse_context(user_prompt)
        question = self._extract_question(user_prompt)

        if not blocks:
            text = (
                "I could not find anything in your uploaded documents that relates to "
                f"\"{question}\". Try uploading a resume and a job description first, or "
                "rephrase the question around your experience and the role."
            )
            return LLMResponse(
                text=text,
                provider=self.name,
                model=self.model,
                latency_ms=(time.perf_counter() - started) * 1000,
                finish_reason="stop",
            )

        resume_blocks = [b for b in blocks if b[0].startswith("R")]
        job_blocks = [b for b in blocks if b[0].startswith("J")]

        lines: List[str] = [
            # Asterisks, not underscores: the frontend renderer does not treat
            # `_` as emphasis, because this text is full of snake_case names.
            "*This answer comes from the offline stub provider. Set LLM_PROVIDER=gemini "
            "and a GEMINI_API_KEY in backend/.env for a real generated answer -- the "
            "retrieval, citations and fit scoring below are real either way.*",
            "",
            f"**Question:** {question}",
            "",
        ]

        if job_blocks:
            lines.append("**What the role asks for**")
            for marker, _source, body in job_blocks:
                lines.append(f"- {self._summarise(body)} [{marker}]")
            lines.append("")

        if resume_blocks:
            lines.append("**What your resume shows**")
            for marker, _source, body in resume_blocks:
                lines.append(f"- {self._summarise(body)} [{marker}]")
            lines.append("")

        lines.append(
            "**Reading the two together:** the strongest overlap is between the resume "
            "evidence and the requirements cited above; anything in the role list without "
            "a matching resume line is where the gap sits. The Fit tab quantifies this."
        )

        return LLMResponse(
            text="\n".join(lines),
            provider=self.name,
            model=self.model,
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=len(user_prompt.split()),
            output_tokens=sum(len(line.split()) for line in lines),
            finish_reason="stop",
        )

    @staticmethod
    def _summarise(body: str, max_words: int = 32) -> str:
        """First meaningful line of a chunk, truncated on a word boundary."""
        candidate = ""
        for line in body.split("\n"):
            stripped = line.strip().lstrip("- ").strip()
            if len(stripped.split()) >= 4:
                candidate = stripped
                break
        candidate = candidate or body.strip().replace("\n", " ")

        words = candidate.split()
        if len(words) > max_words:
            return " ".join(words[:max_words]) + "..."
        return candidate

    def health(self) -> Dict[str, object]:
        return {
            "provider": self.name,
            "model": self.model,
            "configured": True,
            "note": "Deterministic offline provider - answers are templated, not generated.",
        }
