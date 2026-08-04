"""Load the hand-curated eval question set.

See ``config/eval_questions.yaml`` for the question format and the rationale
behind the three categories (single_hop / two_hop / community).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GoldQuestion:
    id: str
    category: str
    question: str
    expected_chunk_ids: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def machine_checkable(self) -> bool:
        """Whether retrieval metrics can be scored against this question today.

        two_hop/community questions are placeholders until the knowledge graph
        exists to generate real expected relationships — they're graded by
        manual read for now, not included in Recall@k/MRR.
        """
        return bool(self.expected_chunk_ids)


def load_gold_set(path: Path | str) -> list[GoldQuestion]:
    raw = yaml.safe_load(Path(path).read_text()) or []
    return [
        GoldQuestion(
            id=item["id"],
            category=item["category"],
            question=item["question"],
            expected_chunk_ids=item.get("expected_chunk_ids") or [],
            notes=item.get("notes", ""),
        )
        for item in raw
    ]


__all__ = ["GoldQuestion", "load_gold_set"]
