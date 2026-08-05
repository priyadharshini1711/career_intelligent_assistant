"""Section-aware chunking for resumes and job descriptions.

Why not a plain fixed-size splitter? Because these two document types have
strong, predictable structure and the questions users ask map directly onto it.
"What am I missing?" is a question about the JD's *Requirements* section versus
the resume's *Skills* and *Experience* sections. A blind 500-character window
cuts a requirements list in half and staples the tail of one bullet to the head
of another, which then retrieves as a chunk that means neither thing.

So the pipeline is:

    text -> sections (heading detection)
         -> blocks   (a bullet, or a paragraph, stays whole)
         -> chunks   (blocks packed to a word budget, with overlap)

Chunks stay small (~120 words). Resumes and JDs are dense -- a single bullet
carries a whole competency -- so small chunks give sharper retrieval, and the
documents are short enough that we can afford more of them.
"""

import re
from typing import Dict, List, Optional, Tuple

from app.ingestion.extract import is_bullet, normalize_bullet
from app.schemas import Chunk, DocumentKind

# Canonical section names. Mapping the many ways a heading can be phrased onto
# one label means downstream code (retrieval boosts, skill extraction) can rely
# on a small fixed vocabulary instead of matching raw heading text.
_SECTION_PATTERNS: List[Tuple[str, str]] = [
    ("Summary", r"^(professional\s+)?(summary|profile|objective|about\s+me|overview)\b"),
    ("Skills", r"^(technical\s+|core\s+|key\s+)?(skills|competencies|technologies|tech\s+stack|expertise|proficiencies)\b"),
    ("Experience", r"^(work\s+|professional\s+|employment\s+|relevant\s+)?(experience|history|background)\b"),
    ("Projects", r"^(personal\s+|key\s+|selected\s+)?projects?\b"),
    ("Education", r"^(education|academics?|qualifications?\s+&\s+education)\b"),
    ("Certifications", r"^(certifications?|licen[cs]es?|courses?|training)\b"),
    ("Achievements", r"^(achievements?|awards?|honou?rs?|publications?)\b"),
    # --- job-description side ---
    ("Responsibilities", r"^(responsibilities|what\s+you'?ll\s+do|the\s+role|role\s+overview|duties|key\s+deliverables|day\s+to\s+day)\b"),
    ("Requirements", r"^(requirements?|qualifications?|what\s+we'?re\s+looking\s+for|must\s+have|minimum\s+qualifications?|who\s+you\s+are|basic\s+qualifications?)\b"),
    ("Preferred", r"^(preferred|nice\s+to\s+have|bonus|good\s+to\s+have|desirable|plus(es)?|preferred\s+qualifications?)\b"),
    ("Benefits", r"^(benefits?|perks|what\s+we\s+offer|compensation|salary|why\s+join)\b"),
    ("Company", r"^(about\s+(us|the\s+company|the\s+team)|who\s+we\s+are|company\s+overview)\b"),
]
_COMPILED_SECTIONS = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in _SECTION_PATTERNS]

# Sections that carry the requirements a candidate is judged against. Used to
# weight retrieval and to drive skill extraction.
JD_REQUIREMENT_SECTIONS = {"Requirements", "Preferred", "Responsibilities", "Skills"}
RESUME_EVIDENCE_SECTIONS = {"Skills", "Experience", "Projects", "Summary", "Certifications"}

_HEADING_MAX_WORDS = 8
_ENDS_WITH_SENTENCE = re.compile(r"[.!?,;:]$")
# Contact lines at the top of a resume are noise for retrieval.
_CONTACT_HINT = re.compile(
    r"(@|https?://|linkedin\.com|github\.com|\+\d[\d\s().-]{7,}|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b)",
    re.IGNORECASE,
)


def canonical_section(line: str) -> Optional[str]:
    """Return the canonical section name if `line` looks like a heading."""
    stripped = line.strip().strip(":").strip()
    if not stripped or len(stripped.split()) > _HEADING_MAX_WORDS:
        return None

    for name, pattern in _COMPILED_SECTIONS:
        if pattern.match(stripped):
            return name
    return None


def _looks_like_heading(line: str) -> bool:
    """Structural heading test for headings we don't have vocabulary for.

    Deliberately conservative: a false positive here fragments a section, which
    is worse than missing a heading and keeping text together.
    """
    stripped = line.strip().strip(":").strip()
    if not stripped or len(stripped.split()) > _HEADING_MAX_WORDS:
        return False
    if _ENDS_WITH_SENTENCE.search(stripped) or is_bullet(line):
        return False

    letters = [c for c in stripped if c.isalpha()]
    if len(letters) < 3:
        return False

    # ALL CAPS, or Title Case with no lowercase-only sentence feel.
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper_ratio > 0.85


def split_sections(text: str) -> List[Tuple[str, List[str]]]:
    """Split text into `(section_name, lines)` pairs."""
    sections: List[Tuple[str, List[str]]] = []
    current_name = "General"
    current_lines: List[str] = []

    for raw_line in text.split("\n"):
        line = raw_line.rstrip()
        if not line.strip():
            current_lines.append("")
            continue

        detected = canonical_section(line)
        if detected is None and _looks_like_heading(line):
            detected = line.strip().strip(":").strip().title()

        if detected is not None:
            if any(item.strip() for item in current_lines):
                sections.append((current_name, current_lines))
            current_name = detected
            current_lines = []
        else:
            current_lines.append(line)

    if any(item.strip() for item in current_lines):
        sections.append((current_name, current_lines))

    return sections or [("General", text.split("\n"))]


def _to_blocks(lines: List[str]) -> List[str]:
    """Group lines into atomic blocks: one bullet, or one paragraph."""
    blocks: List[str] = []
    paragraph: List[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(" ".join(paragraph).strip())
            paragraph.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if is_bullet(line):
            flush()
            blocks.append("- " + normalize_bullet(line))
        else:
            paragraph.append(stripped)

    flush()
    return [block for block in blocks if block]


def _pack(blocks: List[str], target_words: int, overlap_words: int) -> List[str]:
    """Pack blocks into chunks near `target_words`, overlapping by tail words.

    Overlap is carried as whole trailing words of the previous chunk. It exists
    so a requirement that straddles a boundary is still fully present in one of
    the two chunks.
    """
    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for block in blocks:
        block_words = len(block.split())

        # A single oversized block (a wall-of-text paragraph) gets split on its
        # own rather than being allowed to blow the budget.
        if block_words > target_words * 1.8:
            if current:
                chunks.append("\n".join(current))
                current, current_words = [], 0
            words = block.split()
            step = max(1, target_words - overlap_words)
            for start in range(0, len(words), step):
                piece = " ".join(words[start : start + target_words])
                if piece.strip():
                    chunks.append(piece)
            continue

        if current_words + block_words > target_words and current:
            chunks.append("\n".join(current))
            tail = " ".join("\n".join(current).split()[-overlap_words:]) if overlap_words else ""
            current = [tail] if tail else []
            current_words = len(tail.split())

        current.append(block)
        current_words += block_words

    if current and any(item.strip() for item in current):
        chunks.append("\n".join(current))

    return chunks


def chunk_document(
    document_id: str,
    kind: DocumentKind,
    title: str,
    text: str,
    target_words: int = 120,
    overlap_words: int = 30,
    min_words: int = 15,
) -> List[Chunk]:
    """Chunk a document, preserving section labels on every chunk."""
    chunks: List[Chunk] = []
    index = 0

    for section_name, lines in split_sections(text):
        blocks = _to_blocks(lines)
        if not blocks:
            continue

        # Contact details retrieve badly and leak PII into prompts for no gain.
        # Applied to every resume section, not just the leading one: a resume's
        # name line usually trips the heading detector, so the contact block
        # ends up in a section named after the candidate rather than in
        # "General". Scoping this filter to "General" meant it never ran on the
        # block it was written for.
        if kind == DocumentKind.RESUME:
            blocks = [block for block in blocks if not _CONTACT_HINT.search(block)]
            if not blocks:
                continue

        section_started = len(chunks)

        for piece in _pack(blocks, target_words, overlap_words):
            words = piece.split()
            # A runt is usually the tail of a longer section, and folding it
            # into its predecessor beats emitting a fragment that will never
            # retrieve on its own.
            #
            # But a short section is not a runt. "CERTIFICATIONS / AWS
            # Solutions Architect" is seven words and is exactly the kind of
            # thing a JD asks about. Dropping it because it fell below the
            # minimum silently lost real content -- so a section that has
            # produced nothing yet always emits, however short.
            if len(words) < min_words and len(chunks) > section_started:
                merged = chunks[-1].text + "\n" + piece
                chunks[-1].text = merged
                chunks[-1].word_count = len(merged.split())
                continue
            if not words:
                continue

            chunks.append(
                Chunk(
                    id=f"{document_id}:{index}",
                    document_id=document_id,
                    document_kind=kind,
                    document_title=title,
                    section=section_name,
                    index=index,
                    # The section name is prepended to the embedded text so the
                    # vector carries structural context, not just wording. This
                    # measurably helps "what are the requirements" style queries.
                    text=piece,
                    word_count=len(words),
                )
            )
            index += 1

    return chunks


def embedding_text(chunk: Chunk) -> str:
    """The string we actually embed -- section and document title included."""
    return f"[{chunk.document_kind.value} | {chunk.document_title} | {chunk.section}]\n{chunk.text}"


def section_stats(chunks: List[Chunk]) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for chunk in chunks:
        stats[chunk.section] = stats.get(chunk.section, 0) + 1
    return stats
