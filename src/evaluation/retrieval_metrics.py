"""Recall@k and MRR for the retriever, scored against the gold question set.

Only ``machine_checkable`` gold questions (single_hop, today) are scored here.
two_hop/community questions don't have a reliable expected chunk set yet — see
``GoldQuestion.machine_checkable`` — so they're graded by manual read instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from .gold_set import GoldQuestion


@dataclass(frozen=True)
class QuestionResult:
    question: GoldQuestion
    ranked_chunk_ids: list[str]
    # 1-indexed ranks at which an expected chunk_id was found, in the order
    # retrieved. Empty if none of the expected chunks were retrieved at all.
    hit_ranks: list[int]


def evaluate_retrieval(retriever, questions: list[GoldQuestion], max_k: int = 8) -> list[QuestionResult]:
    results = []
    for q in questions:
        if not q.machine_checkable:
            continue
        retrieved = retriever.retrieve(q.question, top_k=max_k)
        ranked_ids = retrieved["chunk_id"].tolist()
        expected = set(q.expected_chunk_ids)
        hit_ranks = [i + 1 for i, cid in enumerate(ranked_ids) if cid in expected]
        results.append(QuestionResult(q, ranked_ids, hit_ranks))
    return results


def recall_at_k(results: list[QuestionResult], k: int) -> float:
    """Fraction of questions where an expected chunk appeared in the top-k."""
    if not results:
        return float("nan")
    hits = sum(1 for r in results if any(rank <= k for rank in r.hit_ranks))
    return hits / len(results)


def mean_reciprocal_rank(results: list[QuestionResult]) -> float:
    """Average of 1/rank of the first expected chunk found (0 if never found)."""
    if not results:
        return float("nan")
    total = sum(1.0 / min(r.hit_ranks) for r in results if r.hit_ranks)
    return total / len(results)


__all__ = ["QuestionResult", "evaluate_retrieval", "recall_at_k", "mean_reciprocal_rank"]
