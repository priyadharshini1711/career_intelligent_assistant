"""Lightweight query-intent classification.

Rules, not a model. The intent taxonomy here is small and closed -- a career
assistant gets asked about five or six things -- and keyword rules over that
taxonomy are transparent, instant, free, and trivially testable. A classifier
would add a model call to every question in exchange for accuracy we cannot
measure at this corpus size. When the rules do not match we fall back to
GENERAL, which uses the neutral prompt and unbiased retrieval, so a
misclassification degrades gracefully instead of failing.

Intent drives two things: which prompt template is used, and how retrieval
splits its budget between the resume and the job description.
"""

import re
from enum import Enum
from typing import Dict, List, Tuple


class Intent(str, Enum):
    SKILL_GAP = "skill_gap"
    ALIGNMENT = "alignment"
    INTERVIEW_PREP = "interview_prep"
    RESUME_IMPROVEMENT = "resume_improvement"
    COMPARISON = "comparison"
    GENERAL = "general"


# Ordered by specificity: the first intent with a match wins, so narrower
# intents are listed before broader ones.
_PATTERNS: List[Tuple[Intent, List[str]]] = [
    (
        Intent.SKILL_GAP,
        [
            r"\b(missing|lacks?|lacking|gaps?|don'?t have|do not have|short(fall)?|weak(ness(es)?)?)\b",
            r"\bwhat.*(need|require).*(learn|develop|improve)\b",
            r"\bnot qualified\b",
        ],
    ),
    (
        Intent.INTERVIEW_PREP,
        [
            r"\binterview\b",
            r"\b(prepare|prep)\b.*\b(role|position|job)\b",
            r"\b(talking points?|elevator pitch|tell me about myself)\b",
            r"\bquestions? (they|I) (might|could|will) (ask|be asked)\b",
            r"\bstar (method|format|answer)\b",
        ],
    ),
    (
        Intent.RESUME_IMPROVEMENT,
        [
            r"\b(improve|rewrite|reword|tailor|optimi[sz]e|strengthen)\b.*\bresume\b",
            r"\bresume\b.*\b(better|stronger|tailor)\b",
            r"\bhow (should|can|do) I (phrase|word|present)\b",
            r"\bbullet points?\b",
        ],
    ),
    (
        Intent.COMPARISON,
        [
            # "which of these roles fits me best" needs to match as readily as
            # "which role is best" -- the earlier, tighter pattern required the
            # noun to follow "which" immediately and missed the common phrasing.
            r"\bwhich\b.*\b(jobs?|roles?|positions?|postings?|openings?|one)\b.*\b(best|better|suit|fit|strong)",
            r"\bcompare\b",
            r"\b(rank|order|prioriti[sz]e)\b.*\b(jobs?|roles?|positions?|application)",
            r"\bbetween (the )?(jobs?|roles?|positions?)\b",
            r"\b(best|strongest|closest) (match|fit)\b",
            r"\ball (of )?(these|the) (jobs?|roles?|positions?)\b",
        ],
    ),
    (
        Intent.ALIGNMENT,
        [
            r"\b(align(s|ed|ment)?|match(es|ed)?|fits?|suited|qualified|relevant)\b",
            r"\bhow (does|do|well)\b.*\b(experience|background|resume)\b",
            r"\bam I a good\b",
            r"\bstrengths?\b",
        ],
    ),
]

_COMPILED: List[Tuple[Intent, List[re.Pattern]]] = [
    (intent, [re.compile(p, re.IGNORECASE) for p in patterns]) for intent, patterns in _PATTERNS
]


def classify(question: str) -> Intent:
    for intent, patterns in _COMPILED:
        if any(pattern.search(question) for pattern in patterns):
            return intent
    return Intent.GENERAL


# How much of the retrieval budget goes to resume chunks. The rest goes to the
# job description. A skill-gap question is mostly "what does the JD demand",
# while a resume-rewrite question is mostly "what does the candidate already
# say". Retrieving both sides every time is the point -- a single ranked list
# over the pooled corpus will happily return six JD chunks and no resume, and
# then the model invents the candidate's background.
RESUME_SHARE: Dict[Intent, float] = {
    Intent.SKILL_GAP: 0.45,
    Intent.ALIGNMENT: 0.5,
    Intent.INTERVIEW_PREP: 0.5,
    Intent.RESUME_IMPROVEMENT: 0.6,
    Intent.COMPARISON: 0.35,
    Intent.GENERAL: 0.5,
}

# Sections worth a small ranking nudge for a given intent, applied after
# fusion. Kept deliberately mild (see SECTION_BOOST in retriever) so it breaks
# ties rather than overriding relevance.
SECTION_PRIORS: Dict[Intent, Dict[str, float]] = {
    Intent.SKILL_GAP: {"Requirements": 1.0, "Preferred": 0.8, "Skills": 0.8, "Responsibilities": 0.5},
    Intent.ALIGNMENT: {"Experience": 0.8, "Requirements": 0.8, "Responsibilities": 0.6, "Skills": 0.5},
    Intent.INTERVIEW_PREP: {"Experience": 1.0, "Projects": 0.8, "Responsibilities": 0.8, "Requirements": 0.5},
    Intent.RESUME_IMPROVEMENT: {"Experience": 1.0, "Summary": 0.8, "Skills": 0.8, "Requirements": 0.5},
    Intent.COMPARISON: {"Requirements": 0.8, "Responsibilities": 0.8, "Skills": 0.5},
    Intent.GENERAL: {},
}
