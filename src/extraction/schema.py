"""Shared vocabulary for the entity-extraction comparison.

Every arm — spaCy, LLM, hybrid — emits :class:`Mention` records over the same
chunks, with the same three types, so the systems can be pooled and scored
against each other rather than against their own conventions.

The type set is deliberately small. spaCy ships 18 labels; the knowledge graph
needs people, organisations and places. Comparing on the full label set would
measure taxonomy alignment as much as extraction quality, and the extra
relation-bearing categories in the project outline (Event, Flight, Document)
have no spaCy equivalent at all — those are handled by rules in the pipeline
proper, not by this comparison.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

#: The frozen type set for the comparison.
TYPES = ("PERSON", "ORG", "GPE")

#: spaCy's labels folded onto the frozen set. Anything absent is discarded.
SPACY_TYPE_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "GPE",
    "LOC": "GPE",
    "FAC": "GPE",
}

#: Salutations and discourse openers that spaCy types as part of a person's
#: name — "Dear Jeffrey", "Hi Reid". The name inside is real, so these are not
#: discarded; they are just never allowed to represent the entity.
SALUTATION_RE = re.compile(
    r"^(dear|hi|hello|hey|thanks|thank|dearest|attn|re|fw|fwd|i)\b", re.IGNORECASE
)

TITLE_RE = re.compile(
    r"^(mr|mrs|ms|miss|dr|prof|professor|judge|justice|senator|sen|rep|hon|sir|dame|"
    r"president|amb|ambassador|gov|governor|atty|attorney|det|officer)\.?\s+",
    re.IGNORECASE,
)
POSSESSIVE_RE = re.compile(r"['’]s$")
PUNCT_RE = re.compile(r"[^\w\s]")
WS_RE = re.compile(r"\s+")

# Spans that are artefacts of normalisation, scanning or markup — never entities.
# Kept deliberately narrow: this removes things no adjudicator would call an
# entity under any label, not things a system merely got *wrong*. A person tagged
# ORG is a scoring error and must survive the filter; a URL fragment is noise.
JUNK_RE = re.compile(
    r"redacted|non-responsive|house\s*oversight|https?|www\.|\.com\b|@",
    re.IGNORECASE,
)

# "CA 94111" — a postcode fragment, not an organisation.
MOSTLY_DIGITS_RE = re.compile(r"^\W*\w*\d{3,}\w*\W*$")


@dataclass(frozen=True)
class Mention:
    """One entity mention located in a chunk.

    ``start`` / ``end`` are offsets into the chunk text; ``doc_start`` / ``doc_end``
    are the same span in normalised-document coordinates, which is what makes
    mentions from different systems comparable and what ties a mention back to a
    page citation.
    """

    system: str
    doc_id: str
    chunk_id: str
    start: int
    end: int
    doc_start: int
    doc_end: int
    surface: str
    type: str
    confidence: float | None = None
    note: str | None = None

    def as_row(self) -> dict:
        row = asdict(self)
        row["norm"] = normalise_surface(self.surface)
        return row


def normalise_surface(surface: str) -> str:
    """Fold a surface form for matching across systems.

    Systems disagree constantly on trivia — casing, a leading title, a trailing
    possessive, an OCR'd curly quote. Matching on the raw string would score those
    as disagreements when they are the same entity.
    """
    text = unicodedata.normalize("NFKC", surface)
    text = TITLE_RE.sub("", text.strip())
    text = POSSESSIVE_RE.sub("", text)
    text = PUNCT_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip().lower()


def chunk_offsets(row) -> tuple[int, int]:
    """``(document offset of the chunk's body, length of any prepended header)``.

    Chunks carry their message's ``From:/Sent:/To:`` block prepended to the text,
    but ``char_start``/``char_end`` delimit only the *body*. So a position in
    ``chunk.text`` maps into the document as
    ``char_start + (position - prefix_len)`` — and a position inside the prefix
    does not correspond to this span at all, because that header text physically
    lives elsewhere in the document.
    """
    span = int(row["char_end"]) - int(row["char_start"])
    return int(row["char_start"]), max(len(row["text"]) - span, 0)


def document_span(
    char_start: int, prefix_len: int, start: int, end: int
) -> tuple[int, int, str | None]:
    """Map a chunk-relative span to document coordinates.

    Mentions found inside a prepended header get no document offsets: the header
    is a copy, so the position is ambiguous. They are flagged instead — and the
    header layer is covered deterministically by the email parser anyway.
    """
    if start < prefix_len:
        return -1, -1, "in_prepended_header"
    return char_start + (start - prefix_len), char_start + (end - prefix_len), None


def is_plausible(surface: str, type_: str) -> bool:
    """Cheap filter for spans no adjudicator should have to look at.

    Applied identically to every arm, and part of each arm's reported pipeline —
    it is not a scoring adjustment. The bar is "nobody would call this an entity
    under any label", so genuine typing mistakes are left in to be scored.
    """
    if type_ not in TYPES:
        return False
    stripped = surface.strip()
    if len(stripped) < 2 or not any(c.isalpha() for c in stripped):
        return False
    if JUNK_RE.search(stripped) or MOSTLY_DIGITS_RE.match(stripped):
        return False
    return bool(normalise_surface(stripped))


__all__ = [
    "Mention",
    "TYPES",
    "chunk_offsets",
    "document_span",
    "SPACY_TYPE_MAP",
    "normalise_surface",
    "is_plausible",
]
