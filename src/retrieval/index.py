"""Build, save, and load a FAISS similarity index over chunk embeddings.

Embeddings are L2-normalised (see :mod:`.embeddings`), so a flat inner-product
index gives exact cosine similarity search. ~28k chunks is small enough that
a flat index is fast without needing an approximate structure.

FAISS only stores vectors, not metadata, so the index is paired with a small
parquet file recording which ``chunk_id`` sits at each row — position ``i`` in
the index corresponds to row ``i`` of that file.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd


def build_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, chunk_ids: list[str], index_path: Path, ids_path: Path) -> None:
    faiss.write_index(index, str(index_path))
    pd.DataFrame({"chunk_id": chunk_ids}).to_parquet(ids_path, index=False)


def load_index(index_path: Path, ids_path: Path) -> tuple[faiss.Index, list[str]]:
    index = faiss.read_index(str(index_path))
    chunk_ids = pd.read_parquet(ids_path)["chunk_id"].tolist()
    return index, chunk_ids


__all__ = ["build_index", "save_index", "load_index"]
