"""Normalise and chunk the corpus.

Usage:
    python scripts/build_chunks.py [--out data/processed] [--target 350] [--overlap 60]

Reads ``manifest.parquet`` + ``text_stats.parquet`` and writes ``chunks.parquet``:
one row per chunk, carrying the document's quality flags so retrieval and
extraction can filter without re-joining.

Every document with extracted text is chunked, including the ones flagged
unusable — the ``usable`` column rides along so that decision stays reversible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ingestion.chunking import OVERLAP_WORDS, TARGET_WORDS, chunk_document  # noqa: E402
from src.ingestion.loadfile import load_text  # noqa: E402
from src.ingestion.normalize import normalise  # noqa: E402

CARRY_COLUMNS = ["ocr_quality", "text_style", "usable", "doc_types"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=REPO_ROOT / "data" / "processed")
    ap.add_argument("--target", type=int, default=TARGET_WORDS)
    ap.add_argument("--overlap", type=int, default=OVERLAP_WORDS)
    args = ap.parse_args()

    out = Path(args.out)
    manifest = pd.read_parquet(out / "manifest.parquet")
    stats = pd.read_parquet(out / "text_stats.parquet").set_index("doc_id")
    docs = manifest.set_index("doc_id").join(stats[CARRY_COLUMNS])

    rows = []
    n_docs = 0
    for doc_id, row in docs.iterrows():
        path = row["text_path"]
        if not isinstance(path, str):
            continue
        nd = normalise(doc_id, load_text(path))
        n_docs += 1
        for chunk in chunk_document(nd, target=args.target, overlap=args.overlap):
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "src_start": chunk.src_start,
                    "src_end": chunk.src_end,
                    "page_bates_start": chunk.page_bates_start,
                    "page_bates_end": chunk.page_bates_end,
                    "page_is_exact": chunk.page_is_exact,
                    "has_header": chunk.has_header,
                    "n_words": chunk.n_words,
                    "text": chunk.text,
                    **{col: row[col] for col in CARRY_COLUMNS},
                }
            )

    chunks = pd.DataFrame(rows)
    chunks.to_parquet(out / "chunks.parquet", index=False)

    print(f"documents chunked   {n_docs:>10,}")
    print(f"chunks              {len(chunks):>10,}")
    print(f"mean words / chunk  {chunks['n_words'].mean():>10.0f}")
    print(f"chunks with a page  {chunks['page_is_exact'].sum():>10,}")
    print(f"chunks with header  {int(chunks['has_header'].sum()):>10,}")
    print(f"wrote {out / 'chunks.parquet'}")


if __name__ == "__main__":
    main()
