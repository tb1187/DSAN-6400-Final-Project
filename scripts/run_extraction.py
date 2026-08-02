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


def load_sample_chunks(out: Path, chunks_path: Path | None = None) -> pd.DataFrame:
    sample = pd.read_parquet(out / "eval_sample.parquet")
    chunks = pd.read_parquet(chunks_path or (out / "chunks.parquet"))
    return chunks[chunks["doc_id"].isin(set(sample["doc_id"]))].reset_index(drop=True)


def load_corpus_chunks(out: Path, chunks_path: Path | None = None) -> pd.DataFrame:
    """Every chunk worth extracting from: the whole corpus minus unusable documents.

    ``usable`` excludes the documents whose OCR is too damaged to trust, which
    would otherwise contribute invented entities as graph nodes.
    """
    chunks = pd.read_parquet(chunks_path or (out / "chunks.parquet"))
    return chunks[chunks["usable"]].reset_index(drop=True)


def to_frame(mentions: list[Mention]) -> pd.DataFrame:
    frame = pd.DataFrame([m.as_row() for m in mentions])
    if frame.empty:
        return frame
    return frame.sort_values(["doc_id", "chunk_id", "start"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", required=True, choices=["spacy", "llm", "hybrid"])
    ap.add_argument(
        "--scope",
        default="sample",
        choices=["sample", "corpus"],
        help="sample = the 35 evaluation documents; corpus = every usable chunk. "
        "Corpus runs write to mentions_<system>_corpus.parquet and checkpoint.",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=5000,
        help="chunks between incremental writes on a corpus run",
    )
    ap.add_argument("--out", default=REPO_ROOT / "data" / "processed")
    ap.add_argument("--model", default=None, help="spaCy model or Claude model id")
    ap.add_argument(
        "--chunks",
        default=None,
        help="chunk table to extract from; defaults to data/processed/chunks.parquet. "
        "Pin a snapshot to reproduce an earlier run.",
    )
    ap.add_argument("--limit", type=int, default=None, help="cap chunks, for smoke tests")
    ap.add_argument(
        "--suffix",
        default="",
        help="appended to the output filename. Use it for smoke tests so a "
        "partial run cannot overwrite a real mention table.",
    )
    args = ap.parse_args()

    out = Path(args.out)
    chunks_path = Path(args.chunks) if args.chunks else None
    if args.scope == "corpus":
        chunks = load_corpus_chunks(out, chunks_path)
        suffix = args.suffix or "_corpus"
    else:
        chunks = load_sample_chunks(out, chunks_path)
        suffix = args.suffix
    if args.limit:
        chunks = chunks.head(args.limit)

    target = out / f"mentions_{args.system}{suffix}.parquet"

    # Resume: skip chunks already present in a previous partial run.
    done = pd.DataFrame()
    if args.scope == "corpus" and target.exists():
        done = pd.read_parquet(target)
        already = set(done["chunk_id"])
        before = len(chunks)
        chunks = chunks[~chunks["chunk_id"].isin(already)].reset_index(drop=True)
        print(f"resuming: {before - len(chunks):,} chunks already extracted, {len(chunks):,} remaining")

    started = time.perf_counter()

    if args.system == "spacy":
        from src.extraction.spacy_ner import DEFAULT_MODEL, extract, load_model

        if args.scope == "corpus":
            nlp = load_model(args.model or DEFAULT_MODEL)
            parts = [done] if len(done) else []
            step = args.checkpoint_every
            for i in range(0, len(chunks), step):
                block = chunks.iloc[i : i + step]
                parts.append(to_frame(extract(block, nlp=nlp)))
                pd.concat(parts, ignore_index=True).to_parquet(target, index=False)
                elapsed = time.perf_counter() - started
                done_n = min(i + step, len(chunks))
                rate = done_n / elapsed if elapsed else 0
                print(
                    f"  {done_n:>7,}/{len(chunks):,} chunks | {rate:>5.0f} chunks/s | "
                    f"eta {(len(chunks) - done_n) / rate / 60 if rate else 0:>5.1f} min",
                    flush=True,
                )
            frame = pd.read_parquet(target)
            print(f"\nsystem              {args.system} (corpus)")
            print(f"documents           {frame['doc_id'].nunique():>8,}")
            print(f"mentions            {len(frame):>8,}")
            print(f"distinct (norm,type){frame.groupby(['norm', 'type']).ngroups:>8,}")
            print("\nby type:")
            print(frame["type"].value_counts().to_string())
            print(f"\nelapsed             {(time.perf_counter() - started) / 60:>8.1f} min")
            print(f"wrote {target}")
            return

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
    frame.to_parquet(target, index=False)

    print(f"system              {args.system}")
    print(f"documents           {chunks['doc_id'].nunique():>8,}")
    print(f"chunks              {len(chunks):>8,}")
    print(f"mentions            {len(frame):>8,}")
    if not frame.empty:
        print(f"distinct (norm,type){frame.groupby(['norm', 'type']).ngroups:>8,}")
        print("\nby type:")
        print(frame["type"].value_counts().to_string())
    print(f"\nelapsed             {elapsed:>8.1f}s")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
