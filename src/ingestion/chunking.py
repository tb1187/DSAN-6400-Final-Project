"""Split normalised documents into retrieval- and extraction-sized chunks.

Chunks are contiguous character spans of a :class:`~src.ingestion.normalize.NormalisedDoc`,
so every chunk keeps an exact offset back into the raw text and therefore a real
page citation.

Three rules are specific to this production:

1. **Header text is metadata, not body content.** An email's ``From:/Sent:/To:``
   block is stripped out before body text is divided into chunks — it is never
   itself a chunk, and it is never packed together with body words toward the
   word-count target. Instead, the header belonging to a chunk's message is
   prepended onto every one of that message's body chunks (all of them, not
   just the first), so a chunk retrieved in isolation always says who wrote it.
   The one exception is a message with no body at all (a bare forwarded header,
   say) — that gets emitted as a chunk of just the header, so it still gets a
   citation instead of vanishing from the corpus.
2. **Paragraphs first, sentences only as a fallback.** The scanned material is
   already fragmented; slicing mid-sentence at a fixed word count makes it worse.
3. **A body chunk never crosses a message boundary, and overlap never bridges
   two unrelated messages.** Body text is packed toward the ~350-word target,
   but packing always breaks at a header boundary regardless of word count, so
   one chunk's body is never a splice of two different senders' text. The
   ~60-word overlap carried across that break is only kept when the new
   message's ``Subject:`` (stripped of ``Re:``/``Fwd:`` prefixes) matches the
   previous message's — i.e. they are actually the same thread.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize import NormalisedDoc

TARGET_WORDS = 350
OVERLAP_WORDS = 60

PARAGRAPH_RE = re.compile(r"\n[ \t]*\n")
WORD_RE = re.compile(r"\S+")
SENTENCE_END_RE = re.compile(r"(?<=[.!?])[ \t]+(?=[A-Z\"'(])|\n")

SUBJECT_RE = re.compile(r"(?im)^[ \t]*Subject[ \t]*:[ \t]*(?P<value>.*)$")
SUBJECT_PREFIX_RE = re.compile(r"(?i)^(re|fw|fwd)\s*:\s*")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_index: int
    char_start: int
    char_end: int
    src_start: int
    src_end: int
    page_bates_start: str | None
    page_bates_end: str | None
    page_is_exact: bool
    has_header: bool
    n_words: int
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}#{self.chunk_index:04d}"


@dataclass(frozen=True)
class _Unit:
    """A block of body text that is never split across chunks."""

    start: int
    end: int
    n_words: int


def _count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Split on blank lines, keeping only non-empty spans."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in PARAGRAPH_RE.finditer(text):
        if text[cursor : match.start()].strip():
            spans.append((cursor, match.start()))
        cursor = match.end()
    if text[cursor:].strip():
        spans.append((cursor, len(text)))
    return spans


def _strip_header_spans(
    spans: list[tuple[int, int]], header_spans: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Remove any portion of ``spans`` that overlaps a header block.

    Headers are metadata, not body content — see rule 1 — so they never
    become part of a body unit in the first place, rather than being
    protected from splitting the way body text is.
    """
    if not header_spans:
        return list(spans)

    stripped: list[tuple[int, int]] = []
    for start, end in spans:
        cursor = start
        for hs, he in header_spans:
            if he <= cursor:
                continue
            if hs >= end:
                break
            if hs > cursor:
                stripped.append((cursor, hs))
            cursor = max(cursor, he)
        if cursor < end:
            stripped.append((cursor, end))
    return stripped


def _hard_split(text: str, start: int, end: int, target: int) -> list[_Unit]:
    """Last-resort split at word boundaries.

    Sentence splitting needs punctuation followed by a capital, which a lot of
    this corpus does not have — plenty of the email bodies are written entirely
    in lower case. Without this fallback such a document becomes one enormous
    chunk.
    """
    positions = [m.start() for m in WORD_RE.finditer(text, start, end)]
    if len(positions) <= target:
        return [_Unit(start, end, len(positions))]

    units: list[_Unit] = []
    unit_start = start
    for i in range(target, len(positions), target):
        cut = positions[i]
        units.append(_Unit(unit_start, cut, target))
        unit_start = cut
    if end > unit_start:
        units.append(_Unit(unit_start, end, _count_words(text[unit_start:end])))
    return units


def _sentence_units(text: str, start: int, end: int, target: int) -> list[_Unit]:
    """Break an oversized block into sentence-aligned units under ``target`` words."""
    boundaries = {start, end}
    for match in SENTENCE_END_RE.finditer(text, start, end):
        boundaries.add(match.end())
    ordered = sorted(b for b in boundaries if start <= b <= end)

    units: list[_Unit] = []
    unit_start = start
    words = 0
    for left, right in zip(ordered, ordered[1:]):
        sentence_words = _count_words(text[left:right])
        if words and words + sentence_words > target:
            units.append(_Unit(unit_start, left, words))
            unit_start = left
            words = 0
        words += sentence_words
    if end > unit_start:
        units.append(_Unit(unit_start, end, words))

    # A single "sentence" can still blow past the target — fall back to words.
    bounded: list[_Unit] = []
    for unit in units:
        if unit.n_words > target:
            bounded.extend(_hard_split(text, unit.start, unit.end, target))
        else:
            bounded.append(unit)
    return bounded


def _build_units(nd: NormalisedDoc, target: int) -> list[_Unit]:
    body_spans = _strip_header_spans(_paragraph_spans(nd.text), nd.header_spans)
    units: list[_Unit] = []
    for start, end in body_spans:
        n_words = _count_words(nd.text[start:end])
        if n_words > target:
            units.extend(_sentence_units(nd.text, start, end, target))
        else:
            units.append(_Unit(start, end, n_words))
    return units


def _header_subject(text: str, start: int, end: int) -> str | None:
    """Normalised ``Subject:`` line within a header span, or ``None`` if absent.

    ``Re:``/``Fwd:``/``Fw:`` prefixes are stripped (repeatedly, for chains like
    ``Re: Fwd: Re:``) so replies and forwards of the same thread compare equal.
    """
    match = SUBJECT_RE.search(text, start, end)
    if not match:
        return None
    value = match.group("value").strip()
    while True:
        rest = SUBJECT_PREFIX_RE.sub("", value)
        if rest == value:
            break
        value = rest
    value = value.strip().lower()
    return value or None


def _governing_header(header_spans: list[tuple[int, int]], pos: int) -> int:
    """Index into ``header_spans`` of the message that owns position ``pos``.

    That is the last header block ending at or before ``pos`` — ``-1`` if no
    header precedes it (body text can appear before any header, in a document
    that isn't an email, or in the doc's preamble before the first message).
    """
    idx = -1
    for i, (_, he) in enumerate(header_spans):
        if he <= pos:
            idx = i
        else:
            break
    return idx


def _same_thread(nd: NormalisedDoc, prev_idx: int, next_idx: int) -> bool:
    if prev_idx < 0 or next_idx < 0:
        return False
    prev_hs, prev_he = nd.header_spans[prev_idx]
    next_hs, next_he = nd.header_spans[next_idx]
    prev_subject = _header_subject(nd.text, prev_hs, prev_he)
    next_subject = _header_subject(nd.text, next_hs, next_he)
    return prev_subject is not None and prev_subject == next_subject


def _overlap_start(text: str, end: int, overlap: int, floor: int) -> int:
    """Character offset ``overlap`` words back from ``end``, never below ``floor``."""
    if overlap <= 0:
        return end
    positions = [m.start() for m in WORD_RE.finditer(text, floor, end)]
    if len(positions) <= overlap:
        return floor
    return positions[-overlap]


@dataclass(frozen=True)
class _SpanMeta:
    start: int
    end: int
    # Overlap text carried across a same-thread message boundary. It is real
    # text from the *previous* message, so it is spliced into ``text`` as an
    # extra piece rather than by extending this span's own char range — doing
    # that used to make the contiguous range silently swallow the header sitting
    # between the two messages.
    carry: tuple[int, int] | None = None


def _pack_body_spans(nd: NormalisedDoc, units: list[_Unit], target: int, overlap: int) -> list[_SpanMeta]:
    spans: list[_SpanMeta] = []
    cur_start: int | None = None
    cur_end = 0
    cur_words = 0
    cur_header = -2  # sentinel: "no chunk open yet"
    pending_carry: tuple[int, int] | None = None

    for unit in units:
        header = _governing_header(nd.header_spans, unit.start)
        crosses_message = cur_start is not None and header != cur_header
        must_break = cur_start is not None and (
            crosses_message or cur_words + unit.n_words > target
        )
        if must_break:
            spans.append(_SpanMeta(cur_start, cur_end, pending_carry))
            pending_carry = None
            if crosses_message:
                if _same_thread(nd, cur_header, header):
                    back = _overlap_start(nd.text, cur_end, overlap, spans[-1].start + 1)
                    pending_carry = (back, cur_end)
                cur_start = unit.start
            else:
                back = _overlap_start(nd.text, cur_end, overlap, cur_start + 1)
                cur_start = min(back, unit.start)
            cur_words = _count_words(nd.text[cur_start : unit.start])
        cur_header = header
        if cur_start is None:
            cur_start = unit.start
        cur_end = unit.end
        cur_words += unit.n_words

    if cur_start is not None and cur_end > cur_start:
        spans.append(_SpanMeta(cur_start, cur_end, pending_carry))
    return spans


def chunk_document(
    nd: NormalisedDoc,
    target: int = TARGET_WORDS,
    overlap: int = OVERLAP_WORDS,
) -> list[Chunk]:
    """Pack a normalised document's body text into overlapping chunks."""
    units = _build_units(nd, target)
    spans = _pack_body_spans(nd, units, target, overlap)

    # A message with no body at all still needs a citable chunk of its own.
    covered = {_governing_header(nd.header_spans, s.start) for s in spans}
    for i, (hs, he) in enumerate(nd.header_spans):
        if i not in covered:
            spans.append(_SpanMeta(hs, he))
    spans.sort(key=lambda s: s.start)

    chunks: list[Chunk] = []
    for i, span in enumerate(spans):
        start, end = span.start, span.end
        header_idx = _governing_header(nd.header_spans, start)
        is_bare_header = header_idx >= 0 and nd.header_spans[header_idx] == (start, end)

        body_words = _count_words(nd.text[start:end])
        pieces: list[str] = []
        if span.carry is not None:
            pieces.append(nd.text[span.carry[0] : span.carry[1]])
        if not is_bare_header and header_idx >= 0:
            hs, he = nd.header_spans[header_idx]
            pieces.append(nd.text[hs:he])
        pieces.append(nd.text[start:end])
        text = "\n\n".join(pieces)

        page_start, page_end = nd.page_range(start, end)
        chunks.append(
            Chunk(
                doc_id=nd.doc_id,
                chunk_index=i,
                char_start=start,
                char_end=end,
                src_start=nd.to_source(start),
                src_end=nd.to_source(max(end - 1, start)) + 1,
                page_bates_start=page_start,
                page_bates_end=page_end,
                page_is_exact=nd.has_pages,
                has_header=header_idx >= 0,
                n_words=body_words,
                text=text,
            )
        )
    return chunks


__all__ = ["Chunk", "chunk_document", "TARGET_WORDS", "OVERLAP_WORDS"]
