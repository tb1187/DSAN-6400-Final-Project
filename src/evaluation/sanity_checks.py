"""Label-free retrieval health checks.

Neither of these needs a gold set or human judgment — they catch structural
failures in the embedding/index pipeline on their own:

* :func:`self_retrieval_check` — a chunk is definitionally relevant to itself,
  so querying with a chunk's own text should retrieve that chunk. If it
  doesn't, the embedding space is broken for that kind of content (garbled
  OCR, for instance).
* :func:`paraphrase_stability_check` — rewording the same question shouldn't
  change which chunks come back. Low overlap across paraphrases means
  retrieval is keying off surface wording rather than meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pandas as pd


@dataclass(frozen=True)
class SelfRetrievalResult:
    n: int
    top1_rate: float
    topk_rate: float


def self_retrieval_check(
    retriever, chunks: pd.DataFrame, n: int = 50, top_k: int = 5, random_state: int = 0
) -> SelfRetrievalResult:
    sample = chunks.sample(min(n, len(chunks)), random_state=random_state)
    hits_at_1 = 0
    hits_at_k = 0
    for _, row in sample.iterrows():
        ranked = retriever.retrieve(row["text"], top_k=top_k)["chunk_id"].tolist()
        if ranked and ranked[0] == row["chunk_id"]:
            hits_at_1 += 1
        if row["chunk_id"] in ranked:
            hits_at_k += 1
    return SelfRetrievalResult(len(sample), hits_at_1 / len(sample), hits_at_k / len(sample))


def paraphrase_stability_check(
    retriever, paraphrase_groups: list[list[str]], top_k: int = 8
) -> float:
    """Mean pairwise Jaccard overlap of retrieved chunk_ids within each group."""
    group_scores = []
    for group in paraphrase_groups:
        if len(group) < 2:
            continue
        sets = [set(retriever.retrieve(q, top_k=top_k)["chunk_id"]) for q in group]
        jaccards = [
            len(a & b) / len(a | b) if (a | b) else 0.0 for a, b in combinations(sets, 2)
        ]
        group_scores.append(sum(jaccards) / len(jaccards))
    return sum(group_scores) / len(group_scores) if group_scores else float("nan")


__all__ = ["SelfRetrievalResult", "self_retrieval_check", "paraphrase_stability_check"]
