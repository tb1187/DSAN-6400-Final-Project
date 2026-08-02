"""Query-time retrieval: embed a query, search the FAISS index, return chunks.

Keeps chunk metadata (text, doc_id, page citation) in a DataFrame indexed by
``chunk_id`` rather than in the FAISS index itself, since FAISS only stores
vectors — see :mod:`.index` for the row <-> chunk_id mapping this joins against.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .embeddings import Embedder
from .index import load_index

REPO_ROOT = Path(__file__).resolve().parents[2]


class Retriever:
    def __init__(self, config_path: Path | str = REPO_ROOT / "config" / "config.yaml") -> None:
        config = yaml.safe_load(Path(config_path).read_text())
        retrieval_cfg = config["retrieval"]
        processed = REPO_ROOT / config["paths"]["processed_data"]

        self.top_k = retrieval_cfg["top_k"]
        self.embedder = Embedder(retrieval_cfg["embedding_model"])
        self.index, chunk_ids = load_index(
            REPO_ROOT / retrieval_cfg["index_path"],
            REPO_ROOT / retrieval_cfg["index_ids_path"],
        )
        self._row_to_chunk_id = chunk_ids

        chunks = pd.read_parquet(processed / "chunks.parquet").set_index("chunk_id")
        self._chunks = chunks.loc[chunk_ids]

    def retrieve(self, query: str, top_k: int | None = None) -> pd.DataFrame:
        top_k = top_k or self.top_k
        query_vec = self.embedder.encode_query(query).reshape(1, -1)
        scores, rows = self.index.search(query_vec, top_k)

        results = self._chunks.iloc[rows[0]].copy()
        results.insert(0, "score", scores[0])
        return results.reset_index()


__all__ = ["Retriever"]
