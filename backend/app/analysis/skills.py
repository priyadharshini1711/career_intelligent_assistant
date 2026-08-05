"""Skill extraction and resume matching.

The pipeline is deliberately two-tier:

* **Tier 1 -- dictionary match.** High precision. If the JD says "Kubernetes"
  and the resume says "Kubernetes", that is a match, full stop, and we can
  point at the exact line. This drives the headline numbers.

* **Tier 2 -- semantic match.** Catches the case the dictionary cannot:
  the JD asks for "container orchestration", the resume says "deployed
  services on EKS". Reported as a *partial* match rather than a full one,
  because the evidence is inferred rather than stated, and the user deserves
  to know which is which before they walk into an interview claiming it.

Anything the JD requires that neither tier finds is a gap. Being explicit
about the partial tier is the honesty guardrail on this feature -- a tool that
tells someone they match a requirement they do not actually have is worse than
useless.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from app.analysis.taxonomy import ALIAS_TO_CANONICAL, AMBIGUOUS, category_of
from app.ingestion.chunking import JD_REQUIREMENT_SECTIONS
from app.rag.embeddings import Embedder
from app.schemas import Chunk, SkillEvidence, SkillGap

# A dictionary hit is definitive. These thresholds only gate the semantic tier.
PARTIAL_MATCH_THRESHOLD = 0.42
STRONG_SEMANTIC_THRESHOLD = 0.62

_DELIMITERS = re.compile(r"[,|/•·;]")
_YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:-|to)?\s*(?:\d{1,2})?\s*\+?\s*years?", re.IGNORECASE)
_YEAR_TOKEN = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
# Words that, near an "N years" phrase, mark it as a claim about the
# candidate's own tenure rather than an incidental number. Resume summaries
# rarely say "years of experience" outright -- "engineer with 5 years building
# Python services" is the common phrasing -- so matching only on "experience"
# missed the explicit claim and fell through to the crude date-span estimate.
_EXPERIENCE_CONTEXT = re.compile(
    r"\b(experience|engineer|developer|analyst|scientist|manager|designer|architect|"
    r"consultant|specialist|building|working|professional|background|career)\b",
    re.IGNORECASE,
)


def _alias_pattern(alias: str) -> re.Pattern:
    """Word-boundary matcher that survives `C++`, `C#`, `.NET` and `Node.js`.

    Python's `\\b` is useless here because `+`, `#` and `.` are non-word
    characters, so `\\bC\\b` happily matches the "C" inside "C++". These
    explicit look-arounds treat those three characters as part of a token.
    """
    return re.compile(
        r"(?<![\w+#.])" + re.escape(alias) + r"(?![\w+#])",
        re.IGNORECASE,
    )


_ALIAS_PATTERNS: List[Tuple[str, str, re.Pattern]] = [
    (canonical, alias, _alias_pattern(alias)) for alias, canonical in ALIAS_TO_CANONICAL.items()
]
# Longer aliases first so "react native" wins over "react" on the same span.
_ALIAS_PATTERNS.sort(key=lambda item: -len(item[1]))


def _looks_like_skill_list(line: str) -> bool:
    """A comma/pipe separated run of short items, i.e. a skills line."""
    return len(_DELIMITERS.findall(line)) >= 2


def _ambiguous_ok(canonical: str, alias: str, line: str, section: str) -> bool:
    """Gate for one-or-two-character skill names.

    "Go", "R" and "C" are ordinary English words. We accept them only when the
    evidence is strong: an unambiguous alias was used ("golang"), or the match
    sits in a Skills section, or on a line that is clearly a delimited list.
    Without this, every "go to market" in a JD scores as Golang.
    """
    if canonical not in AMBIGUOUS:
        return True
    if alias.lower() != canonical.lower():
        return True
    return section == "Skills" or _looks_like_skill_list(line)


@dataclass
class SkillHit:
    canonical: str
    category: str
    line: str
    section: str


def find_skills(text: str, section: str = "General") -> Dict[str, SkillHit]:
    """Dictionary-match skills in a block of text. First hit per skill wins."""
    hits: Dict[str, SkillHit] = {}

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        for canonical, alias, pattern in _ALIAS_PATTERNS:
            if canonical in hits:
                continue
            if pattern.search(stripped) and _ambiguous_ok(canonical, alias, stripped, section):
                hits[canonical] = SkillHit(
                    canonical=canonical,
                    category=category_of(canonical),
                    line=stripped,
                    section=section,
                )
    return hits


def find_skills_in_chunks(chunks: Sequence[Chunk]) -> Dict[str, SkillHit]:
    hits: Dict[str, SkillHit] = {}
    for chunk in chunks:
        for canonical, hit in find_skills(chunk.text, chunk.section).items():
            hits.setdefault(canonical, hit)
    return hits


@dataclass
class JobRequirement:
    """One demand made by the job description."""

    text: str
    section: str
    importance: str  # "required" | "preferred"
    skills: List[str] = field(default_factory=list)


def _importance_for(section: str) -> str:
    return "preferred" if section == "Preferred" else "required"


def extract_job_requirements(chunks: Sequence[Chunk]) -> List[JobRequirement]:
    """Pull individual requirement statements out of a job description.

    Works line by line rather than chunk by chunk because JD requirements are
    almost always one bullet each, and a bullet is the unit a user reasons
    about ("I don't have that one").
    """
    requirements: List[JobRequirement] = []
    seen: Set[str] = set()

    for chunk in chunks:
        if chunk.section not in JD_REQUIREMENT_SECTIONS:
            continue
        for raw_line in chunk.text.split("\n"):
            line = raw_line.strip().lstrip("- ").strip()
            # Skip fragments and whole paragraphs; requirements sit in between.
            if len(line.split()) < 3 or len(line.split()) > 60:
                continue
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)

            requirements.append(
                JobRequirement(
                    text=line,
                    section=chunk.section,
                    importance=_importance_for(chunk.section),
                    skills=sorted(find_skills(line, chunk.section).keys()),
                )
            )

    return requirements


def _best_semantic_match(
    phrase: str, resume_chunks: Sequence[Chunk], resume_vectors: np.ndarray, embedder: Embedder
) -> Tuple[float, Optional[Chunk]]:
    if not len(resume_chunks) or resume_vectors.size == 0:
        return 0.0, None
    query = embedder.encode([f"experience with {phrase}"])[0]
    scores = resume_vectors @ query
    best = int(np.argmax(scores))
    return float(scores[best]), resume_chunks[best]


def _snippet(text: str, max_words: int = 28) -> str:
    words = text.replace("\n", " ").split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")


@dataclass
class SkillMatchResult:
    matched: List[SkillEvidence]
    partial: List[SkillEvidence]
    missing: List[SkillGap]
    required_total: int
    required_covered: float
    preferred_total: int
    preferred_covered: float
    # Counts restricted to required skills. The `matched`/`partial` lists above
    # span both importance levels, so reporting len(matched) against
    # required_total produced nonsense like "11 of 10 matched".
    required_matched: int = 0
    required_partial: int = 0
    preferred_matched: int = 0
    preferred_partial: int = 0


def match_skills(
    job_chunks: Sequence[Chunk],
    resume_chunks: Sequence[Chunk],
    embedder: Embedder,
) -> SkillMatchResult:
    """Compare what the job asks for against what the resume evidences."""
    requirements = extract_job_requirements(job_chunks)
    resume_hits = find_skills_in_chunks(resume_chunks)
    resume_vectors = (
        embedder.encode([chunk.text for chunk in resume_chunks])
        if resume_chunks
        else np.zeros((0, 384), dtype=np.float32)
    )

    # Importance per skill: required wins if a skill appears in both sections.
    skill_importance: Dict[str, str] = {}
    skill_context: Dict[str, str] = {}
    for requirement in requirements:
        for skill in requirement.skills:
            if skill_importance.get(skill) != "required":
                skill_importance[skill] = requirement.importance
            skill_context.setdefault(skill, requirement.text)

    matched: List[SkillEvidence] = []
    partial: List[SkillEvidence] = []
    missing: List[SkillGap] = []

    for skill, importance in sorted(skill_importance.items()):
        if skill in resume_hits:
            matched.append(
                SkillEvidence(
                    skill=skill,
                    resume_snippet=_snippet(resume_hits[skill].line),
                    similarity=1.0,
                )
            )
            continue

        score, chunk = _best_semantic_match(skill, resume_chunks, resume_vectors, embedder)
        if score >= PARTIAL_MATCH_THRESHOLD and chunk is not None:
            partial.append(
                SkillEvidence(
                    skill=skill,
                    resume_snippet=_snippet(chunk.text),
                    similarity=round(score, 3),
                )
            )
        else:
            missing.append(
                SkillGap(
                    skill=skill,
                    importance=importance,
                    reason=_snippet(skill_context.get(skill, ""), 20)
                    or "Listed in the job description with no matching resume evidence.",
                )
            )

    matched_names = {evidence.skill for evidence in matched}
    partial_names = {evidence.skill for evidence in partial}

    # A partial counts as half when scoring coverage: real but unproven.
    def coverage(importance: str) -> Tuple[int, float, int, int]:
        skills = [s for s, imp in skill_importance.items() if imp == importance]
        if not skills:
            return 0, 0.0, 0, 0
        full = sum(1 for s in skills if s in matched_names)
        half = sum(1 for s in skills if s in partial_names)
        return len(skills), (full + 0.5 * half) / len(skills), full, half

    required_total, required_covered, required_matched, required_partial = coverage("required")
    preferred_total, preferred_covered, preferred_matched, preferred_partial = coverage("preferred")

    return SkillMatchResult(
        matched=matched,
        partial=partial,
        missing=missing,
        required_total=required_total,
        required_covered=required_covered,
        preferred_total=preferred_total,
        preferred_covered=preferred_covered,
        required_matched=required_matched,
        required_partial=required_partial,
        preferred_matched=preferred_matched,
        preferred_partial=preferred_partial,
    )


def required_years(text: str) -> Optional[int]:
    """Smallest "N years" figure stated in a job description.

    Smallest, not largest, because JDs mix "5+ years overall" with "2+ years
    with Kubernetes" and the lower bar is the one that gates an application.
    """
    values = [int(match.group(1)) for match in _YEARS_RE.finditer(text)]
    plausible = [value for value in values if 0 < value <= 25]
    return min(plausible) if plausible else None


def estimate_resume_years(text: str) -> Optional[float]:
    """Approximate years of experience from a resume.

    Two strategies, in order of trustworthiness:

    1. An explicit claim -- "6 years of experience" -- is taken at face value.
    2. Otherwise, span from the earliest four-digit year mentioned to the
       latest. This over-counts (education years get swept in) and cannot see
       gaps, which is exactly why the fit score treats seniority as its
       lowest-weighted component and labels it as an estimate in the UI.
    """
    explicit = [
        int(match.group(1))
        for match in _YEARS_RE.finditer(text)
        if _EXPERIENCE_CONTEXT.search(text[max(0, match.start() - 70) : match.end() + 70])
    ]
    plausible = [value for value in explicit if 0 < value <= 45]
    if plausible:
        return float(max(plausible))

    years = sorted({int(match.group(1)) for match in _YEAR_TOKEN.finditer(text)})
    if len(years) >= 2:
        span = years[-1] - years[0]
        return float(span) if 0 < span <= 45 else None
    return None
