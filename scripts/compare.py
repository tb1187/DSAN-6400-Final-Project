"""Interactive side-by-side comparison: plain RAG vs graph-augmented RAG.

Usage:
    python scripts/compare.py

Loads both pipelines once (they share one Retriever, so retrieval only runs
once per question and the two only diverge at prompt-assembly), then answers
questions typed at the prompt until you exit (Ctrl-D, Ctrl-C, or "quit").
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.graph_rag import GraphAugmentedRAGPipeline  # noqa: E402
from src.retrieval.rag import RAGPipeline  # noqa: E402


def main() -> None:
    print("Loading index, embedding model, and communication graph...")
    plain = RAGPipeline()
    graph = GraphAugmentedRAGPipeline(retriever=plain.retriever)
    print("Ready. Type a question (or 'quit' to exit).\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in {"quit", "exit"}:
            break

        plain_result = plain.answer(query)
        graph_result = graph.answer(query)

        print("\n--- plain vector RAG ---")
        print(plain_result.answer)

        if graph_result.graph_context:
            print("\n--- graph context used ---")
            print(graph_result.graph_context)

        print("\n--- graph-augmented RAG ---")
        print(graph_result.answer)
        print()


if __name__ == "__main__":
    main()
