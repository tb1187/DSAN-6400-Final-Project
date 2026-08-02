"""Draw the stratified sample used for the entity-extraction comparison.

Usage:
    python scripts/sample_eval_set.py [--n 50] [--seed 0]

Sampling design
---------------
* **Frame**: extraction-usable documents of 150–1,500 words. The word band is a
  deliberate restriction — proper-noun density is wildly skewed (median 13 spans
  per document, mean 85, max 5,391), so a handful of long documents would
  otherwise supply most of the pool and make manual adjudication intractable.
  The cost is that court filings and book scans are under-represented, which is a
  stated limitation of the comparison rather than a hidden one.
* **Strata**: ``ocr_quality`` × ``text_style``. The interesting result is not a
  single F1 number but *where* each approach wins, and OCR damage is the obvious
  candidate for a differential effect.
* **Allocation**: proportional to the frame, with a floor per stratum so the rare
  cells (noisy text) are still measurable.

Writes ``eval_sample.parquet``: one row per sampled document, with its stratum
and the counts that drove the allocation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MIN_WORDS = 150
MAX_WORDS = 1500
MIN_PER_STRATUM = 5
# A stratum needs at least this many documents in the frame to be sampled at all.
MIN_FRAME_SIZE = 8


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=REPO_ROOT / "data" / "processed")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-words", type=int, default=MIN_WORDS)
    ap.add_argument("--max-words", type=int, default=MAX_WORDS)
    args = ap.parse_args()

    out = Path(args.out)
    docs = (
        pd.read_parquet(out / "manifest.parquet")
        .set_index("doc_id")
        .join(pd.read_parquet(out / "text_stats.parquet").set_index("doc_id"))
    )

    frame = docs[
        docs["usable"]
        & docs["text_path"].notna()
        & docs["n_words"].between(args.min_words, args.max_words)
    ].copy()
    frame["stratum"] = frame["ocr_quality"] + "/" + frame["text_style"]

    sizes = frame["stratum"].value_counts()
    eligible = sizes[sizes >= MIN_FRAME_SIZE].index
    frame = frame[frame["stratum"].isin(eligible)]

    # Allocate the floor to every stratum first, then hand out what is left in
    # proportion to the frame, so ``--n`` is the total rather than a target the
    # floors push past.
    groups = dict(list(frame.groupby("stratum")))
    quota = {s: min(len(g), MIN_PER_STRATUM) for s, g in groups.items()}
    remaining = args.n - sum(quota.values())
    if remaining > 0:
        weights = {s: len(g) for s, g in groups.items()}
        total = sum(weights.values())
        for stratum in sorted(weights, key=weights.get, reverse=True):
            headroom = len(groups[stratum]) - quota[stratum]
            extra = min(headroom, round(remaining * weights[stratum] / total))
            quota[stratum] += extra
        # Rounding can leave the total slightly short; top up the largest stratum.
        shortfall = args.n - sum(quota.values())
        if shortfall > 0:
            biggest = max(groups, key=lambda s: len(groups[s]) - quota[s])
            quota[biggest] += min(shortfall, len(groups[biggest]) - quota[biggest])

    # Shuffle each stratum once, then take from the front. Because the shuffle
    # depends only on the seed, a smaller --n yields a strict subset of a larger
    # one — so resizing the sample never invalidates extraction already paid for.
    picked = [
        groups[s].sample(frac=1, random_state=args.seed).head(n).assign(stratum=s)
        for s, n in quota.items()
        if n
    ]
    sample = pd.concat(picked).sort_index()

    # Carry the frame sizes so the scorer can post-stratify. Per-stratum floors
    # deliberately over-sample the rare cells, so a pooled average over the
    # sample is not a corpus-level estimate without re-weighting.
    frame_sizes = frame["stratum"].value_counts()
    sample["frame_size"] = sample["stratum"].map(frame_sizes)
    sample["frame_share"] = sample["frame_size"] / len(frame)

    columns = [
        "stratum",
        "ocr_quality",
        "text_style",
        "doc_types",
        "n_words",
        "n_proper_noun_spans",
        "image_pages",
        "frame_size",
        "frame_share",
        "text_path",
    ]
    sample[columns].reset_index().to_parquet(out / "eval_sample.parquet", index=False)

    print(f"frame                {len(frame):>6,} documents ({args.min_words}-{args.max_words} words)")
    print(f"sampled              {len(sample):>6,} documents")
    print(f"words in sample      {sample['n_words'].sum():>6,.0f}")
    print(f"proper-noun spans    {sample['n_proper_noun_spans'].sum():>6,.0f}  (upper bound on pool size)")
    print("\nallocation:")
    for stratum, count in sample["stratum"].value_counts().items():
        print(f"  {stratum:<16} {count:>3}  of {sizes[stratum]:>5,} in frame")
    dropped = sizes[~sizes.index.isin(eligible)]
    if len(dropped):
        print(f"\nstrata too small to sample: {dict(dropped)}")
    print(f"\nwrote {out / 'eval_sample.parquet'}")


if __name__ == "__main__":
    main()
