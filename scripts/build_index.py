"""Embed usable chunks and build the FAISS retrieval index.

Usage:
    python scripts/build_index.py [--config config/config.yaml]

Reads ``chunks.parquet``, embeds every chunk flagged ``usable``, and writes a
FAISS index plus a parquet file mapping index row -> ``chunk_id`` (see
``src/retrieval/index.py`` for why the mapping is a separate file).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.embeddings import Embedder  # noqa: E402
from src.retrieval.index import build_index, save_index  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=REPO_ROOT / "config" / "config.yaml")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    retrieval_cfg = config["retrieval"]
    processed = REPO_ROOT / config["paths"]["processed_data"]

    chunks = pd.read_parquet(processed / "chunks.parquet")
    chunks = chunks[chunks["usable"]].reset_index(drop=True)

    embedder = Embedder(retrieval_cfg["embedding_model"])
    t0 = time.time()
    embeddings = embedder.encode_passages(chunks["text"].tolist())
    elapsed = time.time() - t0

    index = build_index(embeddings)
    index_path = REPO_ROOT / retrieval_cfg["index_path"]
    ids_path = REPO_ROOT / retrieval_cfg["index_ids_path"]
    save_index(index, chunks["chunk_id"].tolist(), index_path, ids_path)

    print(f"model               {retrieval_cfg['embedding_model']}")
    print(f"chunks embedded     {len(chunks):>10,}")
    print(f"embedding dim       {embeddings.shape[1]:>10,}")
    print(f"encode time (s)     {elapsed:>10.1f}")
    print(f"wrote {index_path}")
    print(f"wrote {ids_path}")


if __name__ == "__main__":
    main()
