"""RAGPipeline + a graph-derived relationship block appended to the context.

Add-on, not a replacement: the normally-retrieved vector chunks are used
exactly as in :class:`~.rag.RAGPipeline` — this only adds a second block of
context ahead of them, built by entity-linking the query against the real
communication graph (:mod:`src.knowledge_graph`) and rendering whatever direct
edge or 2-hop path connects the named entities to plain text. See
``src/knowledge_graph/render.py`` for why that's real, citable text rather
than a raw graph object — the LLM only ever sees text either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.knowledge_graph.graph_store import GraphStore, load_graph_store
from src.knowledge_graph.render import render_connection, render_exploration
from src.knowledge_graph.traversal import EdgeFact, TwoHopFact, explore, find_connection

from .rag import REPO_ROOT, RAGPipeline, _format_context

GRAPH_SYSTEM_PROMPT = (
    "You are answering questions about a corpus of legal/business documents. "
    "You are given two kinds of context: (1) known relationships from a "
    "communication graph, each already citing the message(s) that evidence it, "
    "and (2) numbered document excerpts. Use both. Cite excerpt numbers like "
    "[1] or [1][3] for claims drawn from the excerpts; for claims drawn from "
    "the relationship graph, refer to it as 'the communication graph' rather "
    "than inventing a number. If neither source supports the answer, say so "
    "plainly instead of guessing — in particular, if the graph explicitly "
    "states no connection was found, do not speculate about one anyway."
)


@dataclass
class GraphRAGAnswer:
    query: str
    answer: str
    sources: pd.DataFrame
    graph_context: str  # empty string if no entities were matched in the query


class GraphAugmentedRAGPipeline(RAGPipeline):
    def __init__(
        self,
        retriever=None,
        graph_store: GraphStore | None = None,
        config_path: Path | str = REPO_ROOT / "config" / "config.yaml",
        api_key: str | None = None,
    ) -> None:
        super().__init__(retriever=retriever, config_path=config_path, api_key=api_key)
        self.graph_store = graph_store or load_graph_store(REPO_ROOT / "data" / "processed")

    def _graph_context(self, query: str) -> tuple[str, list[EdgeFact]]:
        """Rendered relationship text, plus the edges it's built from.

        The edges are returned too so their real evidence documents can be
        pulled into the excerpt list alongside the normally-retrieved chunks
        — the rendered sentence cites a doc_id, but doesn't put that
        document's actual text in front of the model on its own.
        """
        entity_ids = self.graph_store.match_entities(query)
        if len(entity_ids) >= 2:
            a, b = entity_ids[0], entity_ids[1]
            result = find_connection(self.graph_store, a, b)
            text = render_connection(self.graph_store, a, b, result)
            if isinstance(result, EdgeFact):
                return text, [result]
            if isinstance(result, TwoHopFact):
                return text, [result.edge_a_via, result.edge_via_b]
            return text, []
        if len(entity_ids) == 1:
            direct, two_hop = explore(self.graph_store, entity_ids[0])
            text = render_exploration(self.graph_store, entity_ids[0], direct, two_hop)
            edges = list(direct) + [f.edge_a_via for f in two_hop] + [f.edge_via_b for f in two_hop]
            return text, edges
        return "", []

    def _evidence_chunks(self, edge_facts: list[EdgeFact], per_edge: int = 2) -> pd.DataFrame:
        """Real chunk text for the documents that evidence each graph edge.

        Prefers each doc's header chunk (``#0000``-style lead chunk, has_header
        True) since that's the one carrying sender/recipient/subject context;
        falls back to the doc's first chunk if it has no detected header.
        """
        all_chunks = self.retriever._chunks
        pieces = []
        seen_docs: set[str] = set()
        for fact in edge_facts:
            for doc_id in fact.doc_ids:
                if doc_id in seen_docs:
                    continue
                seen_docs.add(doc_id)
                doc_chunks = all_chunks[all_chunks["doc_id"] == doc_id]
                headered = doc_chunks[doc_chunks["has_header"]]
                pieces.append((headered if not headered.empty else doc_chunks).head(per_edge))
        if not pieces:
            empty = all_chunks.iloc[0:0].reset_index()
            empty.insert(0, "score", pd.Series(dtype="float64"))
            return empty
        combined = pd.concat(pieces)
        combined = combined[~combined.index.duplicated()]
        result = combined.reset_index()
        result.insert(0, "score", float("nan"))
        return result

    def answer(self, query: str, top_k: int | None = None) -> GraphRAGAnswer:
        sources = self.retriever.retrieve(query, top_k=top_k)
        graph_context, edge_facts = self._graph_context(query)
        evidence_chunks = self._evidence_chunks(edge_facts)

        frames = [df for df in (sources, evidence_chunks) if not df.empty]
        combined_sources = (
            pd.concat(frames, ignore_index=True).drop_duplicates(subset="chunk_id", keep="first").reset_index(drop=True)
            if frames
            else sources
        )
        chunk_context = _format_context(combined_sources)

        parts = [graph_context] if graph_context else []
        parts.append(f"Excerpts:\n\n{chunk_context}")
        full_context = "\n\n".join(parts)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=GRAPH_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"{full_context}\n\nQuestion: {query}"}],
        )
        answer_text = "".join(block.text for block in message.content if block.type == "text")
        return GraphRAGAnswer(query=query, answer=answer_text, sources=combined_sources, graph_context=graph_context)


__all__ = ["GraphAugmentedRAGPipeline", "GraphRAGAnswer"]
