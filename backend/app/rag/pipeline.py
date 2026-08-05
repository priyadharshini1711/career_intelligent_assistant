"""The RAG pipeline: one question in, one grounded answer out.

    guardrail -> classify -> retrieve -> budget -> render -> generate -> verify

Everything is instrumented into a `QueryTrace`, which is logged and (in local
and staging environments) returned to the UI. Being able to see *which chunks
were retrieved, with what scores, and how much of the answer they supported* is
the difference between "the answer looks wrong" and "the retriever ranked the
benefits section above the requirements section". That inspector caught more
real bugs during development than the test suite did.
"""

from typing import List, Optional

from app.analysis.fit import build_fit_report
from app.config import Settings
from app.errors import LLMUnavailable, PreconditionFailed, ResourceNotFound
from app.guardrails import check_input, check_output
from app.llm.base import LLMProvider
from app.observability import METRICS, QueryTrace, get_logger
from app.rag import prompts
from app.rag.embeddings import Embedder
from app.rag.intent import Intent, classify
from app.rag.retriever import Retriever
from app.schemas import ChatResponse, Document, FitReport
from app.session import Session

logger = get_logger(__name__)

_NO_EVIDENCE_ANSWER = (
    "I couldn't find anything in your uploaded documents that speaks to that.\n\n"
    "That usually means one of three things: the detail genuinely isn't in your "
    "resume or the job description, the question is about something outside those "
    "documents, or the wording is far enough from the documents that retrieval "
    "missed it. Try naming the specific skill, role, or section you're asking about."
)


class RagPipeline:
    def __init__(self, settings: Settings, embedder: Embedder, llm: LLMProvider) -> None:
        self.settings = settings
        self.embedder = embedder
        self.llm = llm

    def _retriever(self, session: Session) -> Retriever:
        return Retriever(
            store=session.store,
            embedder=self.embedder,
            top_k=self.settings.retrieval_top_k,
            candidate_k=self.settings.retrieval_candidate_k,
            dense_weight=self.settings.retrieval_dense_weight,
            mmr_lambda=self.settings.retrieval_mmr_lambda,
            min_score=(
                self.settings.retrieval_min_score
                if self.settings.retrieval_min_score is not None
                else getattr(self.embedder, "min_relevance", 0.05)
            ),
        )

    def _target_jobs(self, session: Session, job_id: Optional[str]) -> List[Document]:
        if job_id is None:
            return session.job_list()
        job = session.jobs.get(job_id)
        if job is None:
            raise ResourceNotFound(
                "That job description is not in this session.", {"job_id": job_id}
            )
        return [job]

    def answer(
        self,
        session: Session,
        question: str,
        job_id: Optional[str] = None,
        include_trace: bool = True,
    ) -> ChatResponse:
        trace = QueryTrace()
        trace.set("session_id", session.id)
        trace.set("question_words", len(question.split()))

        if session.resume is None:
            raise PreconditionFailed("Upload your resume before asking questions.")
        if not session.jobs:
            raise PreconditionFailed("Upload at least one job description before asking questions.")

        # -- 1. input guardrails ------------------------------------------
        with trace.stage("guardrail_input") as attrs:
            verdict = check_input(question)
            attrs["allowed"] = verdict.allowed
            attrs["reason"] = verdict.reason

        if not verdict.allowed:
            METRICS.increment(f"guardrail_blocked_{verdict.reason}")
            trace.set("outcome", f"blocked:{verdict.reason}")
            trace.log(logger, "query blocked by input guardrail")
            return ChatResponse(
                answer=verdict.response or "",
                citations=[],
                suggestions=verdict.suggestions,
                grounded=True,
                refused=True,
                trace=trace.to_dict() if include_trace else None,
            )

        # -- 2. intent ------------------------------------------------------
        with trace.stage("classify") as attrs:
            intent = classify(question)
            attrs["intent"] = intent.value
        trace.set("intent", intent.value)

        jobs = self._target_jobs(session, job_id)
        trace.set("job_count", len(jobs))

        # -- 3. retrieval ---------------------------------------------------
        retriever = self._retriever(session)
        with trace.stage("retrieve") as attrs:
            hits = retriever.retrieve(
                query=question,
                intent=intent,
                resume_id=session.resume.id,
                job_ids=[job.id for job in jobs],
            )
            attrs["retrieved"] = len(hits)
            attrs["max_dense_score"] = round(retriever.max_dense_score(hits), 4)
            attrs["sources"] = [
                {
                    "chunk_id": hit.chunk.id,
                    "kind": hit.chunk.document_kind.value,
                    "document": hit.chunk.document_title,
                    "section": hit.chunk.section,
                    "score": hit.score,
                    "dense": hit.dense_score,
                    "lexical": hit.lexical_score,
                }
                for hit in hits
            ]

        if retriever.is_weak(hits):
            METRICS.increment("no_evidence")
            trace.set("outcome", "no_evidence")
            trace.log(logger, "query had no usable evidence")
            return ChatResponse(
                answer=_NO_EVIDENCE_ANSWER,
                citations=[],
                suggestions=prompts.FOLLOW_UPS[Intent.GENERAL],
                grounded=True,
                refused=False,
                trace=trace.to_dict() if include_trace else None,
            )

        # -- 4. context assembly --------------------------------------------
        with trace.stage("build_context") as attrs:
            budgeted, used_words = prompts.apply_context_budget(
                hits, self.settings.context_max_words
            )
            citations, _ = prompts.build_citations(budgeted)
            context_block = prompts.render_context(budgeted, citations)
            history_block = prompts.render_history(session.history)

            user_prompt = prompts.build_user_prompt(
                question=question,
                intent=intent,
                context_block=context_block,
                history_block=history_block,
                job_titles=[job.title for job in jobs] if len(jobs) > 1 else None,
            )
            attrs["context_words"] = used_words
            attrs["chunks_in_prompt"] = len(budgeted)
            attrs["dropped_chunks"] = len(hits) - len(budgeted)
            attrs["prompt_words"] = len(user_prompt.split())

        # -- 5. generation ---------------------------------------------------
        with trace.stage("generate") as attrs:
            try:
                response = self.llm.complete(
                    system_prompt=prompts.SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=self.settings.llm_temperature,
                    max_tokens=self.settings.llm_max_output_tokens,
                )
            except LLMUnavailable:
                METRICS.increment("llm_error")
                trace.set("outcome", "llm_error")
                trace.log(logger, "llm call failed")
                raise
            attrs.update(response.usage())

        # -- 6. output guardrails ---------------------------------------------
        with trace.stage("guardrail_output") as attrs:
            check = check_output(response.text, [citation.marker for citation in citations])
            attrs["grounding_ratio"] = check.grounding
            attrs["invalid_markers"] = check.invalid_markers
            attrs["used_markers"] = check.used_markers

        if check.invalid_markers:
            # The model cited something we never gave it. Worth an explicit
            # warning-level log: a rising rate here means the prompt contract
            # is degrading, which no aggregate latency metric would reveal.
            METRICS.increment("invalid_citations")
            logger.warning(
                "model produced citations that were not supplied",
                extra={"invalid": check.invalid_markers, "session_id": session.id},
            )

        # Only return the citations the answer actually leaned on, so the UI
        # does not imply support that was never used.
        used = set(check.used_markers)
        surfaced = [citation for citation in citations if citation.marker in used] or citations

        session.record_turn(question, check.answer, self.settings.max_chat_history_turns)

        METRICS.increment("questions_answered")
        METRICS.observe("answer_latency_ms", trace.total_ms)
        METRICS.observe("grounding_ratio", check.grounding)
        trace.set("outcome", "answered")
        trace.set("grounded", check.grounded)
        trace.log(logger)

        return ChatResponse(
            answer=check.answer,
            citations=surfaced,
            suggestions=prompts.FOLLOW_UPS.get(intent, prompts.FOLLOW_UPS[Intent.GENERAL]),
            grounded=check.grounded,
            refused=False,
            trace=trace.to_dict() if include_trace else None,
        )

    # -- analysis ------------------------------------------------------------

    def fit_report(self, session: Session, job_id: str) -> FitReport:
        """Fit report for one job, cached per (session, job)."""
        if session.resume is None:
            raise PreconditionFailed("Upload your resume before requesting a fit report.")
        job = session.jobs.get(job_id)
        if job is None:
            raise ResourceNotFound(
                "That job description is not in this session.", {"job_id": job_id}
            )

        cached = session.fit_cache.get(job_id)
        if cached is not None:
            return cached

        trace = QueryTrace()
        with trace.stage("fit_report") as attrs:
            report = build_fit_report(job=job, resume=session.resume, embedder=self.embedder)
            attrs["overall_score"] = report.overall_score
            attrs["missing"] = len(report.missing_skills)
            attrs["matched"] = len(report.matched_skills)

        session.fit_cache[job_id] = report
        trace.set("session_id", session.id)
        trace.set("job_id", job_id)
        trace.log(logger, "fit report generated")
        METRICS.increment("fit_reports")
        return report

    def all_fit_reports(self, session: Session) -> List[FitReport]:
        reports = [self.fit_report(session, job.id) for job in session.job_list()]
        return sorted(reports, key=lambda report: -report.overall_score)
