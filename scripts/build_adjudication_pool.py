"""Pool every system's mentions into one blinded adjudication sheet.

Usage:
    python scripts/build_adjudication_pool.py [--annotators 2] [--overlap 10]

Design
------
* **Pooled.** Judgements are made on the union of what the systems produced, so
  recall is measured *relative to the pool*. An entity every system missed is
  invisible to this evaluation — a real limitation that belongs in the writeup.
* **Blinded.** The sheet never says which system proposed a row, and rows are
  shuffled. The key lives in a separate file the annotators do not open.
* **Judged per distinct ``(surface, type)`` within a document**, not per
  occurrence. ``Jeffrey Epstein / PERSON`` is decided once for the document and
  applied to all its occurrences, which cuts the workload several-fold without
  changing what is being measured. Context strings are included so the decision
  can be made without opening the source file.
* **Overlapped.** The first ``--overlap`` documents go to every annotator so
  inter-annotator agreement can be computed; the rest are split.

Fill in two columns: ``verdict`` (y / n) and, when a type is wrong, ``true_type``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

WS_RE = re.compile(r"\s+")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.extraction.schema import TYPES  # noqa: E402

SYSTEMS = ("spacy", "llm", "hybrid")
CONTEXT_CHARS = 60


def _spreadsheet_safe(text: str) -> str:
    """Keep a cell from being parsed as a formula, and flatten line breaks.

    Excel and Sheets evaluate any cell starting with = + - or @, so OCR text like
    ``----------This Electronic Message`` renders as ``#NAME?`` and the annotator
    loses the context they need. A leading space defuses it and costs nothing,
    since these columns are only ever read by a human.
    """
    flattened = WS_RE.sub(" ", text).strip()
    return f" {flattened}" if flattened[:1] in "=+-@" else flattened


def _context(chunk_text: str, start: int, end: int) -> tuple[str, str]:
    if start < 0:
        return "", ""
    left = chunk_text[max(0, start - CONTEXT_CHARS) : start]
    right = chunk_text[end : end + CONTEXT_CHARS]
    return _spreadsheet_safe(left), _spreadsheet_safe(right)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=REPO_ROOT / "data" / "processed")
    ap.add_argument("--sheets", default=REPO_ROOT / "results" / "adjudication")
    ap.add_argument("--annotators", type=int, default=2)
    ap.add_argument("--overlap", type=int, default=10, help="documents judged by everyone")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    sheets = Path(args.sheets)
    sheets.mkdir(parents=True, exist_ok=True)

    chunks = pd.read_parquet(out / "chunks.parquet").set_index("chunk_id")
    sample = pd.read_parquet(out / "eval_sample.parquet").set_index("doc_id")

    frames = []
    for system in SYSTEMS:
        path = out / f"mentions_{system}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
        else:
            print(f"  (skipping {system}: {path.name} not built yet)")
    if not frames:
        raise SystemExit("No mention tables found — run scripts/run_extraction.py first.")
    mentions = pd.concat(frames, ignore_index=True)

    # One row per (document, normalised surface, proposed type).
    grouped = mentions.groupby(["doc_id", "norm", "type"], as_index=False).agg(
        surface=("surface", "first"),
        occurrences=("surface", "size"),
        chunk_id=("chunk_id", "first"),
        start=("start", "first"),
        end=("end", "first"),
        systems=("system", lambda s: "|".join(sorted(set(s)))),
        unlocated=("note", lambda n: bool((n == "unlocated").any())),
    )

    contexts = [
        _context(chunks.loc[row.chunk_id, "text"], int(row.start), int(row.end))
        if row.chunk_id in chunks.index
        else ("", "")
        for row in grouped.itertuples()
    ]
    grouped["left_context"] = [c[0] for c in contexts]
    grouped["right_context"] = [c[1] for c in contexts]
    grouped["stratum"] = grouped["doc_id"].map(sample["stratum"])
    grouped["n_systems"] = grouped["systems"].str.count(r"\|") + 1
    grouped["disputed"] = grouped["n_systems"] < len(frames)

    key = grouped.copy()
    key.insert(0, "pool_id", [f"P{i:05d}" for i in range(len(key))])

    # The annotators' view: no system column, no dispute flag, shuffled.
    sheet_columns = [
        "pool_id",
        "doc_id",
        "left_context",
        "surface",
        "right_context",
        "type",
        "occurrences",
    ]
    sheet = key[sheet_columns].copy()
    # The surface can contain a line break the chunker preserved; flatten it for
    # display only. Matching uses `norm` from the key, which is unaffected.
    sheet["surface"] = sheet["surface"].map(_spreadsheet_safe)
    sheet["verdict"] = ""
    sheet["true_type"] = ""
    sheet["notes"] = ""

    docs = list(sample.index)
    rng = pd.Series(docs).sample(frac=1, random_state=args.seed).tolist()
    shared, rest = rng[: args.overlap], rng[args.overlap :]
    splits = [rest[i :: args.annotators] for i in range(args.annotators)]

    key.to_csv(sheets / "pool_key.csv", index=False)
    for i, own in enumerate(splits, start=1):
        assigned = sheet[sheet["doc_id"].isin(set(shared) | set(own))]
        assigned = assigned.sample(frac=1, random_state=args.seed + i)
        path = sheets / f"adjudication_annotator{i}.csv"
        assigned.to_csv(path, index=False)
        print(f"annotator {i}: {len(assigned):>5,} decisions over {len(set(shared) | set(own)):>3} documents -> {path.name}")

    print(f"\npooled rows            {len(key):>7,}")
    print(f"mention occurrences    {len(mentions):>7,}")
    print(f"systems pooled         {len(frames)}  ({', '.join(sorted(mentions['system'].unique()))})")
    print(f"unanimous rows         {int((~key['disputed']).sum()):>7,}")
    print(f"disputed rows          {int(key['disputed'].sum()):>7,}")
    print(f"shared for agreement   {args.overlap} documents")
    print(f"\ntypes: {', '.join(TYPES)}")
    print(f"key (do not open while adjudicating): {sheets / 'pool_key.csv'}")


if __name__ == "__main__":
    main()
