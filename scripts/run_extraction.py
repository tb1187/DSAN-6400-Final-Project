"""Run one extraction arm over the evaluation sample.

Usage:
    python scripts/run_extraction.py --system spacy
    python scripts/run_extraction.py --system llm     [needs ANTHROPIC_API_KEY]
    python scripts/run_extraction.py --system hybrid  [needs ANTHROPIC_API_KEY]

Writes ``mentions_<system>.parquet``. Every arm sees exactly the same chunks, so
differences in output are differences in extraction, not in preprocessing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.extraction.schema import Mention  # noqa: E402
from src.utils.env import load_env  # noqa: E402

load_env()


def load_sample_chunks(out: Path) -> pd.DataFrame:
    sample = pd.read_parquet(out / "eval_sample.parquet")
    chunks = pd.read_parquet(out / "chunks.parquet")
    return chunks[chunks["doc_id"].isin(set(sample["doc_id"]))].reset_index(drop=True)


def to_frame(mentions: list[Mention]) -> pd.DataFrame:
    frame = pd.DataFrame([m.as_row() for m in mentions])
    if frame.empty:
        return frame
    return frame.sort_values(["doc_id", "chunk_id", "start"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", required=True, choices=["spacy", "llm", "hybrid"])
    ap.add_argument("--out", default=REPO_ROOT / "data" / "processed")
    ap.add_argument("--model", default=None, help="spaCy model or Claude model id")
    ap.add_argument("--limit", type=int, default=None, help="cap chunks, for smoke tests")
    args = ap.parse_args()

    out = Path(args.out)
    chunks = load_sample_chunks(out)
    if args.limit:
        chunks = chunks.head(args.limit)

    started = time.perf_counter()

    if args.system == "spacy":
        from src.extraction.spacy_ner import DEFAULT_MODEL, extract

        mentions = extract(chunks, model=args.model or DEFAULT_MODEL)
    else:
        from src.extraction.llm_ner import extract_hybrid, extract_llm

        if args.system == "llm":
            mentions = extract_llm(chunks, model=args.model)
        else:
            from src.extraction.spacy_ner import extract as spacy_extract

            candidates = spacy_extract(chunks)
            mentions = extract_hybrid(chunks, candidates, model=args.model)

    elapsed = time.perf_counter() - started
    frame = to_frame(mentions)
    frame.to_parquet(out / f"mentions_{args.system}.parquet", index=False)

    print(f"system              {args.system}")
    print(f"documents           {chunks['doc_id'].nunique():>8,}")
    print(f"chunks              {len(chunks):>8,}")
    print(f"mentions            {len(frame):>8,}")
    if not frame.empty:
        print(f"distinct (norm,type){frame.groupby(['norm', 'type']).ngroups:>8,}")
        print("\nby type:")
        print(frame["type"].value_counts().to_string())
    print(f"\nelapsed             {elapsed:>8.1f}s")
    print(f"wrote {out / f'mentions_{args.system}.parquet'}")


if __name__ == "__main__":
    main()
