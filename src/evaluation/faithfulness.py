"""Automatic, non-LLM faithfulness proxy: lexical overlap with cited sources.

Extracts proper nouns and multi-digit numbers from a generated answer and
checks how many of them actually appear in the text of the chunks that were
retrieved for it. An answer full of names/dates absent from every cited
excerpt is a concrete, checkable red flag — this doesn't require an LLM judge
or a gold answer, just the same :class:`~src.retrieval.rag.RAGAnswer` the
pipeline already returns.

This is a coarse proxy, not a precise one: the proper-noun regex is a plain
"capitalized word" heuristic (no real NER), so it both over-flags sentence-
initial common words and misses lowercase names. Good enough to catch gross
hallucination, not a substitute for a human read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CITATION_RE = re.compile(r"\[\d+\]")
NUMBER_RE = re.compile(r"\b\d{2,}\b")
PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b")


@dataclass(frozen=True)
class FaithfulnessResult:
    total_terms: int
    supported_terms: int
    unsupported: list[str]

    @property
    def coverage(self) -> float:
        return self.supported_terms / self.total_terms if self.total_terms else float("nan")


def _checkable_terms(text: str) -> set[str]:
    text = CITATION_RE.sub("", text)
    return set(NUMBER_RE.findall(text)) | set(PROPER_NOUN_RE.findall(text))


def check_faithfulness(rag_answer) -> FaithfulnessResult:
    terms = _checkable_terms(rag_answer.answer)
    source_text = " ".join(rag_answer.sources["text"].tolist())
    supported = [t for t in terms if t in source_text]
    unsupported = sorted(t for t in terms if t not in source_text)
    return FaithfulnessResult(len(terms), len(supported), unsupported)


__all__ = ["FaithfulnessResult", "check_faithfulness"]
