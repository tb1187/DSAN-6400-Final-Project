"""Offset-integrity tests for normalisation and chunking.

Offset drift is the failure mode that silently ruins every page citation
downstream and shows up in no output anyone eyeballs, so it gets the tests.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ingestion.chunking import chunk_document  # noqa: E402
from src.ingestion.loadfile import load_text  # noqa: E402
from src.ingestion.normalize import REDACTION_SENTINEL, normalise  # noqa: E402

MANIFEST = REPO_ROOT / "data" / "processed" / "manifest.parquet"

RAW = """From: Richard Kahn
Sent: 11/14/2016 3:04:29 PM
To: jeffrey E. [jee@example.com]
Subject: aapl

The quarterly report describes a settle-
ment between the parties, which was
signed in March.

HOUSE OVERSIGHT 026677

Non-Responsive - Redacted

A second paragraph follows the redaction.
"""


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# -- normalisation ------------------------------------------------------


def test_bates_stamp_removed_and_recorded():
    nd = normalise("DOC_1", RAW)
    assert "HOUSE OVERSIGHT 026677" not in nd.text
    assert [m.bates for m in nd.page_marks] == ["HOUSE_OVERSIGHT_026677"]


def test_redaction_collapsed_to_sentinel():
    nd = normalise("DOC_1", RAW)
    assert REDACTION_SENTINEL in nd.text
    assert "Non-Responsive" not in nd.text
    assert nd.n_redactions == 1


def test_line_wrap_repaired():
    nd = normalise("DOC_1", RAW)
    assert "settlement" in nd.text
    assert "settle-\nment" not in nd.text


def test_header_span_detected():
    nd = normalise("DOC_1", RAW)
    assert nd.header_spans
    start, end = nd.header_spans[0]
    assert "From:" in nd.text[start:end]
    assert nd.in_header(start, end)


def test_offsets_are_monotonic():
    nd = normalise("DOC_1", RAW)
    mapped = [nd.to_source(i) for i in range(len(nd.text))]
    assert all(a <= b for a, b in zip(mapped, mapped[1:]))


def test_offsets_stay_in_range():
    nd = normalise("DOC_1", RAW)
    assert all(0 <= nd.to_source(i) <= len(RAW) for i in range(len(nd.text)))


def test_kept_text_maps_back_to_itself():
    """A character the normaliser kept must map to the same character in the raw text."""
    nd = normalise("DOC_1", RAW)
    checked = 0
    for i, char in enumerate(nd.text):
        if not char.isalpha() or nd.is_synthetic(i):
            continue
        assert RAW[nd.to_source(i)] == char, f"position {i} mapped to {nd.to_source(i)}"
        checked += 1
    assert checked > 100


def test_only_introduced_characters_are_synthetic():
    nd = normalise("DOC_1", RAW)
    synthetic = "".join(c for i, c in enumerate(nd.text) if nd.is_synthetic(i))
    assert set(synthetic) <= set(REDACTION_SENTINEL + " \n")


def test_page_lookup_uses_the_stamp_at_or_after_the_position():
    """A stamp sits at the foot of its page, so text before it belongs to it."""
    nd = normalise("DOC_1", RAW)
    early = nd.text.index("quarterly")
    assert nd.page_at(early) == "HOUSE_OVERSIGHT_026677"


def test_document_without_stamps_has_no_pages():
    nd = normalise("DOC_2", "Just some text with no stamp anywhere.\n")
    assert not nd.has_pages
    assert nd.page_at(0) is None


def test_empty_document():
    nd = normalise("DOC_3", "")
    assert nd.text == ""
    assert chunk_document(nd) == []


# -- chunking -----------------------------------------------------------


def test_chunks_are_contiguous_spans_of_the_clean_text():
    nd = normalise("DOC_1", RAW)
    for chunk in chunk_document(nd, target=40, overlap=5):
        assert nd.text[chunk.char_start : chunk.char_end] == chunk.text


def test_chunks_advance():
    nd = normalise("DOC_1", RAW)
    chunks = chunk_document(nd, target=30, overlap=5)
    starts = [c.char_start for c in chunks]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


def test_header_block_is_not_split():
    nd = normalise("DOC_1", RAW)
    chunks = chunk_document(nd, target=20, overlap=0)
    header_chunks = [c for c in chunks if "From: Richard Kahn" in c.text]
    assert header_chunks
    assert "Subject: aapl" in header_chunks[0].text


def test_long_block_containing_a_header_is_still_split():
    """Regression: a block that merely *touches* a header must not skip splitting.

    Treating any header-overlapping block as unsplittable produced a single
    67,000-word chunk on the real corpus.
    """
    body = " ".join(f"Sentence number {i} here." for i in range(400))
    raw = f"Preamble paragraph.\n{body}\nSubject: an inline header line\n{body}\n"
    nd = normalise("DOC_4", raw)
    assert nd.header_spans, "test needs a real header block to be meaningful"
    chunks = chunk_document(nd, target=350, overlap=60)
    assert max(c.n_words for c in chunks) < 900


def test_lowercase_block_without_sentence_punctuation_is_still_split():
    """Regression: sentence splitting needs a capital letter; much of this corpus
    has none, so a word-count fallback has to bound the chunk."""
    raw = "hey " + " ".join(f"word{i}" for i in range(3000)) + "\n"
    nd = normalise("DOC_5", raw)
    chunks = chunk_document(nd, target=350, overlap=60)
    assert max(c.n_words for c in chunks) < 900


def test_chunk_ids_are_unique_and_ordered():
    nd = normalise("DOC_1", RAW)
    chunks = chunk_document(nd, target=30, overlap=5)
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)


# -- against the real corpus -------------------------------------------


@pytest.mark.skipif(not MANIFEST.exists(), reason="manifest not built")
def test_real_documents_round_trip():
    """Sample real documents: every chunk's words must survive the source mapping."""
    import pandas as pd

    manifest = pd.read_parquet(MANIFEST)
    sample = manifest[manifest["text_path"].notna()].sample(25, random_state=0)

    for _, row in sample.iterrows():
        raw = load_text(row["text_path"])
        nd = normalise(row["doc_id"], raw)
        chunks = chunk_document(nd)
        assert chunks or not nd.text.strip()

        for chunk in chunks:
            # A continuation chunk of a multi-chunk message has its governing
            # header copied onto the front, so the real span is a suffix of
            # chunk.text rather than an exact match — see chunking.py rule 1.
            body_text = nd.text[chunk.char_start : chunk.char_end]
            assert chunk.text.endswith(body_text)
            assert 0 <= chunk.src_start <= chunk.src_end <= len(raw)
            # The chunk's own body words (excluding any prepended header, which
            # isn't drawn from this span) should appear in the raw span it claims,
            # allowing for the words normalisation repaired or removed.
            source_words = set(_words(raw[chunk.src_start : chunk.src_end]))
            chunk_words = _words(body_text)
            if len(chunk_words) >= 20:
                overlap = sum(1 for w in chunk_words if w in source_words)
                assert overlap / len(chunk_words) > 0.8, chunk.chunk_id


@pytest.mark.skipif(not MANIFEST.exists(), reason="manifest not built")
def test_page_citations_land_inside_the_document_range():
    import pandas as pd

    manifest = pd.read_parquet(MANIFEST).set_index("doc_id")
    sample = manifest[manifest["text_path"].notna()].sample(25, random_state=1)

    for doc_id, row in sample.iterrows():
        nd = normalise(doc_id, load_text(row["text_path"]))
        if not nd.has_pages:
            continue
        low, high = int(doc_id.rsplit("_", 1)[-1]), int(row["bates_end"].rsplit("_", 1)[-1])
        for chunk in chunk_document(nd):
            for page in (chunk.page_bates_start, chunk.page_bates_end):
                if page is None:
                    continue
                assert low <= int(page.rsplit("_", 1)[-1]) <= high, f"{chunk.chunk_id} -> {page}"
