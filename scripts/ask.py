"""Interactive RAG query loop.

Usage:
    python scripts/ask.py

Loads the retriever and Claude client once, then answers questions typed at
the prompt until you exit (Ctrl-D, Ctrl-C, or "quit").
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.rag import RAGPipeline  # noqa: E402


def main() -> None:
    print("Loading index and embedding model...")
    pipe = RAGPipeline()
    print("Ready. Type a question (or 'quit' to exit).\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in {"quit", "exit"}:
            break

        result = pipe.answer(query)
        print()
        print(result.answer)
        print()


if __name__ == "__main__":
    main()
