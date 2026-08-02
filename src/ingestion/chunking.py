"""Split normalised documents into retrieval- and extraction-sized chunks.

Chunks are contiguous character spans of a :class:`~src.ingestion.normalize.NormalisedDoc`,
so every chunk keeps an exact offset back into the raw text and therefore a real
page citation.

Three rules are specific to this production:

1. **Email header blocks are atomic and lead their message.** A ``From:/Sent:/To:``
   block is never split, and it starts a new chunk rather than trailing at the end
   of the previous one — so the first body chunk of a message carries its own
   header. That is what lets a name in the body be attributed to the right
   conversation, and what makes a retrieved chunk interpretable on its own. A long
   message that spills into further chunks would otherwise leave those chunks
   headerless, so its header is also copied onto the front of every later chunk
   of that same message (not counted toward that chunk's own char span, since the
   text is a repeat — only its own body content is real, citable source text).
2. **Paragraphs first, sentences only as a fallback.** The scanned material is
   already fragmented; slicing mid-sentence at a fixed word count makes it worse.
3. **Overlap is a span, not a copy.** Each chunk begins a fixed number of words
   before the previous one ended, so chunks stay single contiguous spans and the
   offset bookkeeping stays trivial.
4. **Overlap never bridges two unrelated messages.** Rule 1 guarantees a header
   opens its own chunk, but the overlap in rule 3 would otherwise still pull the
   previous message's trailing words into it — splicing two different senders'
   text together just because they happen to be adjacent in the document. That
   bleed is only kept when the new header's ``Subject:`` (stripped of
   ``Re:``/``Fwd:`` prefixes) matches the previous message's, i.e. they are
   actually the same thread. Otherwise the new chunk starts exactly at the header.
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
    is_header: bool
    n_words: int
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}#{self.chunk_index:04d}"


@dataclass(frozen=True)
class _Unit:
    """A block of text that is never split across chunks."""

    start: int
    end: int
    n_words: int
    is_header: bool


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


def _merge_header_spans(
    spans: list[tuple[int, int]], header_spans: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Merge paragraph spans that a single header block straddles.

    Header lines are usually contiguous, but a blank line inside the block would
    otherwise let the paragraph splitter cut it in half.
    """
    if not header_spans:
        return spans

    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged:
            prev_start, prev_end = merged[-1]
            straddles = any(hs < end and prev_start < he for hs, he in header_spans)
            if straddles and any(hs < prev_end and start < he for hs, he in header_spans):
                merged[-1] = (prev_start, end)
                continue
        merged.append((start, end))
    return merged


def _hard_split(nd: NormalisedDoc, start: int, end: int, target: int) -> list[_Unit]:
    """Last-resort split at word boundaries.

    Sentence splitting needs punctuation followed by a capital, which a lot of
    this corpus does not have — plenty of the email bodies are written entirely
    in lower case. Without this fallback such a document becomes one enormous
    chunk. Boundaries inside a header block are still avoided.
    """
    positions = [m.start() for m in WORD_RE.finditer(nd.text, start, end)]
    if len(positions) <= target:
        return [_Unit(start, end, len(positions), nd.in_header(start, end))]

    units: list[_Unit] = []
    unit_start = start
    for i in range(target, len(positions), target):
        cut = positions[i]
        if any(hs < cut < he for hs, he in nd.header_spans):
            continue
        units.append(_Unit(unit_start, cut, target, nd.in_header(unit_start, cut)))
        unit_start = cut
    if end > unit_start:
        units.append(
            _Unit(unit_start, end, _count_words(nd.text[unit_start:end]), nd.in_header(unit_start, end))
        )
    return units


def _sentence_units(nd: NormalisedDoc, start: int, end: int, target: int) -> list[_Unit]:
    """Break an oversized block into sentence-aligned units under ``target`` words.

    Candidate boundaries that fall *inside* an email header block are dropped, so
    a header is never cut in half — but a long block that merely contains one is
    still split, rather than being exempted wholesale.
    """
    text = nd.text
    boundaries = {start, end}
    for match in SENTENCE_END_RE.finditer(text, start, end):
        pos = match.end()
        if not any(hs < pos < he for hs, he in nd.header_spans):
            boundaries.add(pos)
    ordered = sorted(b for b in boundaries if start <= b <= end)

    units: list[_Unit] = []
    unit_start = start
    words = 0
    for left, right in zip(ordered, ordered[1:]):
        sentence_words = _count_words(text[left:right])
        if words and words + sentence_words > target:
            units.append(_Unit(unit_start, left, words, nd.in_header(unit_start, left)))
            unit_start = left
            words = 0
        words += sentence_words
    if end > unit_start:
        units.append(_Unit(unit_start, end, words, nd.in_header(unit_start, end)))

    # A single "sentence" can still blow past the target — fall back to words.
    bounded: list[_Unit] = []
    for unit in units:
        if unit.n_words > target:
            bounded.extend(_hard_split(nd, unit.start, unit.end, target))
        else:
            bounded.append(unit)
    return bounded


def _build_units(nd: NormalisedDoc, target: int) -> list[_Unit]:
    spans = _merge_header_spans(_paragraph_spans(nd.text), nd.header_spans)
    units: list[_Unit] = []
    for start, end in spans:
        n_words = _count_words(nd.text[start:end])
        if n_words > target:
            units.extend(_sentence_units(nd, start, end, target))
        else:
            units.append(_Unit(start, end, n_words, nd.in_header(start, end)))
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
        stripped = SUBJECT_PREFIX_RE.sub("", value)
        if stripped == value:
            break
        value = stripped
    value = value.strip().lower()
    return value or None


def _governing_header(header_spans: list[tuple[int, int]], pos: int) -> tuple[int, int] | None:
    """The most recent header block ending at or before ``pos``, if any."""
    governing = None
    for hs, he in header_spans:
        if he <= pos:
            governing = (hs, he)
        else:
            break
    return governing


def _overlap_start(text: str, end: int, overlap: int, floor: int) -> int:
    """Character offset ``overlap`` words back from ``end``, never below ``floor``."""
    if overlap <= 0:
        return end
    positions = [m.start() for m in WORD_RE.finditer(text, floor, end)]
    if len(positions) <= overlap:
        return floor
    return positions[-overlap]


def chunk_document(
    nd: NormalisedDoc,
    target: int = TARGET_WORDS,
    overlap: int = OVERLAP_WORDS,
) -> list[Chunk]:
    """Pack a normalised document into overlapping chunks."""
    units = _build_units(nd, target)
    if not units:
        return []

    spans: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end = 0
    cur_words = 0
    last_subject: str | None = None

    for unit in units:
        # A header block always starts a chunk so it leads its own message.
        must_break = cur_words and (
            unit.is_header or cur_words + unit.n_words > target
        )
        subject = _header_subject(nd.text, unit.start, unit.end) if unit.is_header else None
        if must_break:
            spans.append((cur_start, cur_end))
            # Overlap would otherwise splice the previous message's tail onto an
            # unrelated header — only keep it within the same thread.
            same_thread = subject is not None and subject == last_subject
            if unit.is_header and not same_thread:
                cur_start = unit.start
            else:
                back = _overlap_start(nd.text, cur_end, overlap, cur_start + 1)
                cur_start = min(back, unit.start)
            cur_words = _count_words(nd.text[cur_start : unit.start])
        if subject is not None:
            last_subject = subject
        if cur_start is None:
            cur_start = unit.start
        cur_end = unit.end
        cur_words += unit.n_words

    if cur_start is not None and cur_end > cur_start:
        spans.append((cur_start, cur_end))

    chunks: list[Chunk] = []
    for i, (start, end) in enumerate(spans):
        text = nd.text[start:end]
        is_header = nd.in_header(start, end)
        if not is_header:
            governing = _governing_header(nd.header_spans, start)
            if governing is not None:
                text = nd.text[governing[0] : governing[1]] + "\n\n" + text
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
                is_header=is_header,
                n_words=_count_words(text),
                text=text,
            )
        )
    return chunks


__all__ = ["Chunk", "chunk_document", "TARGET_WORDS", "OVERLAP_WORDS"]
