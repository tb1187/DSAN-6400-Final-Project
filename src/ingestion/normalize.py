"""Turn raw OCR text into text that is safe to chunk, embed and run NER over.

Four things in the raw text actively damage downstream extraction:

* **Burned-in Bates stamps** (``HOUSE OVERSIGHT 026677``) sit in the middle of the
  running text and get tagged as organisations. They are removed — but their
  positions are kept, because each stamp marks a *page boundary* and carries that
  page's own Bates number. That is how a chunk earns a real page citation.
* **Redaction runs** (``____``, ``Non-Responsive - Redacted``) look like content to
  a retriever and get absorbed into adjacent entity spans. They collapse to a
  single ``[REDACTED]`` sentinel.
* **Hard-wrapped lines**, where a sentence is split across many short lines and
  words are hyphenated at the line edge.
* **Email header blocks**, which are parsed separately and precisely by
  :mod:`src.ingestion.email_headers`. They stay in the text but are marked, so a
  chunk knows whether it is header or body.

Every one of those rewrites shifts character positions. Rather than run them as
sequential passes and try to reconcile the drift afterwards, all of them are
collected as :class:`Edit` records over the *raw* text, applied in one pass, and
the resulting :class:`NormalisedDoc` keeps a segment table that maps any position
in the clean text back to its original offset exactly.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field

from .email_headers import HEADER_RE
from .textstats import BATES_STAMP_RE, REDACTION_RE

REDACTION_SENTINEL = "[REDACTED]"

# "word-\nnext" -> "wordnext". Only when the next line starts lower-case, so a
# hyphenated proper noun split across lines is left alone. The match deliberately
# covers only the hyphen and newline, never the letters either side — anything
# inside an edit loses its character-level source mapping, and the letters are
# real content we want to keep addressable.
DEHYPHENATE_RE = re.compile(r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[a-z])")

# A line that does not end a sentence, followed by a line that starts mid-sentence.
SOFT_WRAP_RE = re.compile(r"(?<=[^\n.!?:;,\"'\)\]\-])\n(?=[a-z(])")

# "Non-Responsive - Redacted" is two matches separated by punctuation; runs like
# that collapse to one sentinel rather than several.
REDACTION_JOIN = 4

TRAILING_WS_RE = re.compile(r"[ \t]+(?=\n)")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
MULTI_BLANK_RE = re.compile(r"\n{3,}")

BATES_NUMBER_RE = re.compile(r"(\d{4,})")

# Header lines separated by less than this many characters belong to one block.
HEADER_GAP = 200


def _canonical_bates(stamp: str) -> str:
    """``HOUSE OVERSIGHT 026677`` -> ``HOUSE_OVERSIGHT_026677``."""
    match = BATES_NUMBER_RE.search(stamp)
    return f"HOUSE_OVERSIGHT_{match.group(1)}" if match else stamp.strip()


@dataclass(frozen=True)
class Edit:
    """One rewrite of the raw text, in raw coordinates."""

    start: int
    end: int
    replacement: str
    kind: str
    bates: str | None = None


@dataclass(frozen=True)
class Segment:
    """A run of the clean text and where it came from.

    ``is_replacement`` segments have no character-for-character source, so every
    position inside one maps back to where the replaced span began.
    """

    out_start: int
    src_start: int
    length: int
    is_replacement: bool


@dataclass
class PageMark:
    """A page boundary observed in the text, with that page's Bates number."""

    src_offset: int
    bates: str


@dataclass
class NormalisedDoc:
    doc_id: str
    text: str
    segments: list[Segment] = field(default_factory=list)
    page_marks: list[PageMark] = field(default_factory=list)
    header_spans: list[tuple[int, int]] = field(default_factory=list)
    n_redactions: int = 0

    def __post_init__(self) -> None:
        self._out_starts = [s.out_start for s in self.segments]
        self._mark_offsets = [m.src_offset for m in self.page_marks]

    # -- offsets ---------------------------------------------------------
    def _segment_at(self, out_pos: int) -> Segment | None:
        if not self.segments:
            return None
        i = max(bisect_right(self._out_starts, out_pos) - 1, 0)
        return self.segments[i]

    def to_source(self, out_pos: int) -> int:
        """Map a position in the clean text back to the raw text.

        Characters the normaliser kept map to their exact original offset.
        Characters it *introduced* — the redaction sentinel, a space standing in
        for a line break — have no original of their own, and map to where the
        span they replaced began. Use :meth:`is_synthetic` to tell them apart.
        """
        seg = self._segment_at(out_pos)
        if seg is None:
            return out_pos
        if seg.is_replacement:
            return seg.src_start
        return seg.src_start + (out_pos - seg.out_start)

    def is_synthetic(self, out_pos: int) -> bool:
        """Whether the character at ``out_pos`` was introduced by normalisation."""
        seg = self._segment_at(out_pos)
        return bool(seg and seg.is_replacement)

    # -- pages -----------------------------------------------------------
    def page_at(self, out_pos: int) -> str | None:
        """Bates number of the page containing ``out_pos``.

        A stamp is printed at the foot of the page it belongs to, so a position
        belongs to the first stamp at or after it.
        """
        if not self.page_marks:
            return None
        src = self.to_source(out_pos)
        i = bisect_right(self._mark_offsets, src - 1)
        if i >= len(self.page_marks):
            return self.page_marks[-1].bates
        return self.page_marks[i].bates

    def page_range(self, start: int, end: int) -> tuple[str | None, str | None]:
        return self.page_at(start), self.page_at(max(end - 1, start))

    @property
    def has_pages(self) -> bool:
        return bool(self.page_marks)

    def in_header(self, start: int, end: int) -> bool:
        """Whether a span overlaps any email header block."""
        return any(hs < end and start < he for hs, he in self.header_spans)


def page_marks_from_raw(raw: str) -> list[PageMark]:
    """Page boundaries in raw coordinates, without normalising the text.

    For callers that work on the raw text directly — the email-header parser, for
    instance — and only need page attribution.
    """
    return [
        PageMark(m.start(), _canonical_bates(m.group(0)))
        for m in BATES_STAMP_RE.finditer(raw)
    ]


def page_at_offset(marks: list[PageMark], src_offset: int) -> str | None:
    """Bates number of the page containing a raw-text offset.

    A stamp is printed at the foot of its page, so an offset belongs to the first
    stamp at or after it.
    """
    if not marks:
        return None
    i = bisect_right([m.src_offset for m in marks], src_offset - 1)
    return marks[min(i, len(marks) - 1)].bates


def _merge_runs(spans: list[tuple[int, int]], raw: str, join: int) -> list[tuple[int, int]]:
    """Merge spans separated by at most ``join`` non-alphanumeric characters."""
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged:
            prev_start, prev_end = merged[-1]
            gap = raw[prev_end:start]
            if len(gap) <= join and not any(c.isalnum() for c in gap):
                merged[-1] = (prev_start, end)
                continue
        merged.append((start, end))
    return merged


def _collect_edits(raw: str) -> list[Edit]:
    """Find every rewrite site in the raw text, highest priority first.

    A lower-priority pattern is skipped where it would overlap an edit already
    claimed by a higher-priority one, so the rewrites can be applied in a single
    left-to-right pass without interfering with each other.
    """
    claimed = bytearray(len(raw))
    edits: list[Edit] = []

    def claim(start: int, end: int) -> bool:
        if claimed.find(b"\x01", start, end) != -1:
            return False
        claimed[start:end] = b"\x01" * (end - start)
        return True

    for match in BATES_STAMP_RE.finditer(raw):
        if claim(match.start(), match.end()):
            edits.append(
                Edit(
                    match.start(),
                    match.end(),
                    "",
                    "bates",
                    _canonical_bates(match.group(0)),
                )
            )

    for start, end in _merge_runs(
        [(m.start(), m.end()) for m in REDACTION_RE.finditer(raw)], raw, REDACTION_JOIN
    ):
        if claim(start, end):
            edits.append(Edit(start, end, REDACTION_SENTINEL, "redaction"))

    for match in DEHYPHENATE_RE.finditer(raw):
        if claim(match.start(), match.end()):
            edits.append(Edit(match.start(), match.end(), "", "dehyphenate"))

    for match in SOFT_WRAP_RE.finditer(raw):
        if claim(match.start(), match.end()):
            edits.append(Edit(match.start(), match.end(), " ", "softwrap"))

    for pattern, replacement, kind in (
        (TRAILING_WS_RE, "", "trailing_ws"),
        (MULTI_SPACE_RE, " ", "multi_space"),
        (MULTI_BLANK_RE, "\n\n", "multi_blank"),
    ):
        for match in pattern.finditer(raw):
            if claim(match.start(), match.end()):
                edits.append(Edit(match.start(), match.end(), replacement, kind))

    edits.sort(key=lambda e: e.start)
    return edits


def _find_header_spans(text: str) -> list[tuple[int, int]]:
    """Group runs of email header lines into blocks."""
    spans: list[tuple[int, int]] = []
    for match in HEADER_RE.finditer(text):
        if spans and match.start() - spans[-1][1] < HEADER_GAP:
            spans[-1] = (spans[-1][0], match.end())
        else:
            spans.append((match.start(), match.end()))
    return spans


def normalise(doc_id: str, raw: str) -> NormalisedDoc:
    """Normalise one document's text, keeping an exact map back to the raw text."""
    edits = _collect_edits(raw)

    pieces: list[str] = []
    segments: list[Segment] = []
    page_marks: list[PageMark] = []
    out_len = 0
    cursor = 0
    n_redactions = 0

    for edit in edits:
        if edit.start > cursor:
            kept = raw[cursor : edit.start]
            pieces.append(kept)
            segments.append(Segment(out_len, cursor, len(kept), False))
            out_len += len(kept)

        if edit.kind == "bates":
            page_marks.append(PageMark(edit.start, edit.bates or ""))
        elif edit.kind == "redaction":
            n_redactions += 1

        if edit.replacement:
            pieces.append(edit.replacement)
            segments.append(Segment(out_len, edit.start, len(edit.replacement), True))
            out_len += len(edit.replacement)
        cursor = edit.end

    if cursor < len(raw):
        tail = raw[cursor:]
        pieces.append(tail)
        segments.append(Segment(out_len, cursor, len(tail), False))

    text = "".join(pieces)
    return NormalisedDoc(
        doc_id=doc_id,
        text=text,
        segments=segments,
        page_marks=page_marks,
        header_spans=_find_header_spans(text),
        n_redactions=n_redactions,
    )


__all__ = [
    "NormalisedDoc",
    "PageMark",
    "Segment",
    "Edit",
    "normalise",
    "page_marks_from_raw",
    "page_at_offset",
    "REDACTION_SENTINEL",
]
