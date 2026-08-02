"""Answer a query by retrieving chunks and asking Claude to synthesise them.

Kept separate from :class:`~.retriever.Retriever` so retrieval can be tested
and used (e.g. for graph construction) without an API key or network call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anthropic
import pandas as pd
import yaml

from .retriever import Retriever

REPO_ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = (
    "You are answering questions about a corpus of legal/business documents "
    "using only the numbered excerpts provided. Cite the excerpt number(s) "
    "supporting each claim, like [1] or [1][3]. If the excerpts do not contain "
    "the answer, say so plainly instead of guessing."
)


@dataclass
class RAGAnswer:
    query: str
    answer: str
    sources: pd.DataFrame


def _format_context(chunks: pd.DataFrame) -> str:
    blocks = []
    for i, row in enumerate(chunks.itertuples(), start=1):
        citation = row.doc_id
        if getattr(row, "page_bates_start", None):
            citation += f" (p. {row.page_bates_start}-{row.page_bates_end})"
        blocks.append(f"[{i}] ({citation})\n{row.text}")
    return "\n\n".join(blocks)


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever | None = None,
        config_path: Path | str = REPO_ROOT / "config" / "config.yaml",
        api_key: str | None = None,
    ) -> None:
        config = yaml.safe_load(Path(config_path).read_text())
        self.retriever = retriever or Retriever(config_path)
        self.model = config["retrieval"]["anthropic_model"]
        self.client = anthropic.Anthropic(api_key=api_key)

    def answer(self, query: str, top_k: int | None = None) -> RAGAnswer:
        sources = self.retriever.retrieve(query, top_k=top_k)
        context = _format_context(sources)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Excerpts:\n\n{context}\n\nQuestion: {query}",
                }
            ],
        )
        answer_text = "".join(block.text for block in message.content if block.type == "text")
        return RAGAnswer(query=query, answer=answer_text, sources=sources)


__all__ = ["RAGPipeline", "RAGAnswer"]
