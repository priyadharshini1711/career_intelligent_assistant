"""Fit scoring.

The score is computed, not generated. That is the central decision here.

Asking an LLM "score this resume against this JD out of 100" produces a number
that moves between runs, cannot be explained, and quietly encodes whatever bias
the model has about job titles. Users treat a number as objective, so it has to
actually be objective. Every component below is arithmetic over evidence we can
point at, and the UI shows the breakdown rather than the total alone.

Weights are a judgement call, not a fitted model -- there is no labelled
dataset of "good hires" here, and pretending otherwise would be dishonest.
They encode a simple prior: what the role explicitly requires matters most,
how the work actually overlaps matters next, and seniority is a weak signal
because it is the least reliably extractable. They live in one dict so they
are easy to argue with and easy to change.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

from app.analysis.skills import (
    SkillMatchResult,
    estimate_resume_years,
    match_skills,
    required_years,
)
from app.ingestion.chunking import JD_REQUIREMENT_SECTIONS
from app.rag.embeddings import Embedder
from app.schemas import Chunk, Document, FitComponent, FitReport

# Weights were rebalanced after measuring how much each component actually
# separates a good match from a bad one on the sample set. Skill coverage
# scored 80 / 46 / 13 across three jobs of decreasing relevance; semantic
# alignment scored 88 / 88 / 81 for the same three. So coverage does the
# discriminating and alignment mostly confirms the resume is from the right
# industry. The weights now reflect that rather than my initial guess.
WEIGHTS = {
    "required_skills": 0.50,
    "semantic_alignment": 0.20,
    "preferred_skills": 0.20,
    "seniority": 0.10,
}

# Cosine similarity between a JD requirement and the closest resume chunk,
# mapped linearly onto 0-100.
#
# Calibrated against a control: an unrelated resume (a pastry chef's) scores
# 0.00-0.09 against all three sample engineering roles, while a relevant
# software resume scores 0.50-0.55. So the floor sits just above the noise
# level and the ceiling at the top of the realistic range for this model.
#
# The honest caveat, which the UI repeats: the gap between a *relevant* resume
# and the *right* resume is small in this space (0.54 for the matching role vs
# 0.50 for a poorly matched one). This component reliably detects "wrong
# field"; it does not reliably rank two roles in the same field. That is why
# it carries 20% and not 40%.
_SIMILARITY_FLOOR = 0.10
_SIMILARITY_CEILING = 0.60


def _scale(value: float, floor: float, ceiling: float) -> float:
    if ceiling <= floor:
        return 0.0
    return float(np.clip((value - floor) / (ceiling - floor), 0.0, 1.0)) * 100


def _semantic_alignment(
    job_chunks: Sequence[Chunk], resume_chunks: Sequence[Chunk], embedder: Embedder
) -> Tuple[float, str]:
    """How well the resume covers the job's requirement/responsibility text."""
    targets = [chunk for chunk in job_chunks if chunk.section in JD_REQUIREMENT_SECTIONS]
    if not targets:
        targets = list(job_chunks)
    if not targets or not resume_chunks:
        return 0.0, "Not enough text to compare."

    job_vectors = embedder.encode([chunk.text for chunk in targets])
    resume_vectors = embedder.encode([chunk.text for chunk in resume_chunks])

    # Best resume chunk per requirement chunk -- "is this covered anywhere?"
    best_per_requirement = (job_vectors @ resume_vectors.T).max(axis=1)

    # Mean over *every* requirement passage.
    #
    # An earlier version averaged only the strongest two-thirds, on the theory
    # that every JD contains a clause nobody matches and including it would
    # drag all candidates toward the same score. Measuring it showed the
    # opposite: trimming raised 0.54 to 0.66 and compressed the spread between
    # a strong and a weak candidate, because the clauses being discarded were
    # exactly the ones that told them apart. Keeping all of them is both more
    # honest and more discriminative.
    mean_similarity = float(best_per_requirement.mean())
    covered = int((best_per_requirement >= 0.45).sum())

    return (
        _scale(mean_similarity, _SIMILARITY_FLOOR, _SIMILARITY_CEILING),
        f"Across {len(targets)} requirement passages, your closest resume evidence "
        f"averaged {mean_similarity:.2f} similarity ({covered} passages matched strongly). "
        "An unrelated resume scores near 0.05 here.",
    )


def _seniority(job_text: str, resume_text: str) -> Tuple[Optional[float], str]:
    """Compare stated experience requirement against an estimate from the resume."""
    wanted = required_years(job_text)
    have = estimate_resume_years(resume_text)

    if wanted is None:
        return None, "The job description does not state a years-of-experience requirement."
    if have is None:
        return None, f"The role asks for {wanted}+ years; your resume does not state a total."

    if have >= wanted:
        score = 100.0
        verdict = f"meets the {wanted}+ year requirement"
    else:
        # Partial credit rather than a cliff: 4 years against a "5+ years"
        # posting is not a disqualification in practice.
        score = float(np.clip(have / wanted, 0, 1) * 100)
        verdict = f"is short of the {wanted}+ year requirement"

    return score, (
        f"Estimated {have:.0f} years of experience from your resume, which {verdict}. "
        "This is an approximation from dates and explicit claims."
    )


def _verdict(score: float) -> str:
    if score >= 78:
        return "Strong match"
    if score >= 62:
        return "Good match"
    if score >= 45:
        return "Partial match"
    return "Weak match"


def build_fit_report(
    job: Document,
    resume: Document,
    embedder: Embedder,
    precomputed_match: Optional[SkillMatchResult] = None,
) -> FitReport:
    match = precomputed_match or match_skills(job.chunks, resume.chunks, embedder)

    components: List[FitComponent] = []
    weights: List[float] = []

    # -- required skills --------------------------------------------------
    if match.required_total:
        components.append(
            FitComponent(
                name="Required skills",
                score=round(match.required_covered * 100, 1),
                weight=WEIGHTS["required_skills"],
                explanation=(
                    f"{match.required_matched} of {match.required_total} required skills "
                    f"matched directly, {match.required_partial} partially (counted as half)."
                ),
            )
        )
        weights.append(WEIGHTS["required_skills"])

    # -- preferred skills -------------------------------------------------
    if match.preferred_total:
        components.append(
            FitComponent(
                name="Preferred skills",
                score=round(match.preferred_covered * 100, 1),
                weight=WEIGHTS["preferred_skills"],
                explanation=(
                    f"{match.preferred_matched} of {match.preferred_total} nice-to-have "
                    f"skills matched, {match.preferred_partial} partially."
                ),
            )
        )
        weights.append(WEIGHTS["preferred_skills"])

    # -- semantic alignment ------------------------------------------------
    alignment_score, alignment_note = _semantic_alignment(job.chunks, resume.chunks, embedder)
    components.append(
        FitComponent(
            name="Experience alignment",
            score=round(alignment_score, 1),
            weight=WEIGHTS["semantic_alignment"],
            explanation=alignment_note,
        )
    )
    weights.append(WEIGHTS["semantic_alignment"])

    # -- seniority ---------------------------------------------------------
    seniority_score, seniority_note = _seniority(job.text, resume.text)
    if seniority_score is not None:
        components.append(
            FitComponent(
                name="Seniority",
                score=round(seniority_score, 1),
                weight=WEIGHTS["seniority"],
                explanation=seniority_note,
            )
        )
        weights.append(WEIGHTS["seniority"])
    else:
        # Surfaced with zero weight so the UI can explain the absence rather
        # than silently dropping a component the user expected to see.
        components.append(
            FitComponent(
                name="Seniority",
                score=0.0,
                weight=0.0,
                explanation=seniority_note + " Excluded from the score.",
            )
        )

    # Renormalise over the components that actually applied, so a JD with no
    # "preferred" section is not penalised for the section it does not have.
    total_weight = sum(weights) or 1.0
    overall = sum(
        component.score * component.weight
        for component in components
        if component.weight > 0
    ) / total_weight

    return FitReport(
        job_id=job.id,
        job_title=job.title,
        overall_score=round(overall, 1),
        verdict=_verdict(overall),
        components=components,
        matched_skills=match.matched,
        missing_skills=match.missing,
        partial_skills=match.partial,
    )
