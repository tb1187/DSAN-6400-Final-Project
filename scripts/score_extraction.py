"""Score the extraction arms against the adjudicated pool.

Usage:
    python scripts/score_extraction.py

Scoring unit is the distinct ``(document, normalised surface, type)`` triple —
the same unit that was adjudicated. Two views are reported:

``typed``
    The span must be found *and* labelled correctly.
``untyped``
    The span must be found; the label is ignored.

Reporting both separates "missed the entity" from "found it, called it the wrong
thing", which is the difference between a recall problem and a taxonomy problem
and calls for different fixes.

Recall here is **relative to the pool** — an entity no system proposed cannot be
counted as missed. The email-header probe at the end is the antidote: it is a
small slice of ground truth that exists independently of what any system found.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.extraction.schema import normalise_surface  # noqa: E402

SYSTEMS = ("spacy", "llm", "hybrid")


def load_adjudications(sheets: Path) -> pd.DataFrame:
    """Read every annotator sheet and resolve the overlapping rows."""
    files = sorted(sheets.glob("adjudication_annotator*.csv"))
    if not files:
        raise SystemExit(f"No adjudication sheets in {sheets}")

    frames = []
    for i, path in enumerate(files, start=1):
        frame = pd.read_csv(path, dtype=str).fillna("")
        frame["annotator"] = f"A{i}"
        frames.append(frame)
    judged = pd.concat(frames, ignore_index=True)
    judged = judged[judged["verdict"].str.strip() != ""]
    judged["verdict"] = judged["verdict"].str.strip().str.lower().str[0]
    judged["true_type"] = judged["true_type"].str.strip().str.upper()
    return judged


def agreement(judged: pd.DataFrame) -> tuple[float | None, int]:
    """Cohen's kappa on rows judged by more than one annotator."""
    counts = judged.groupby("pool_id")["annotator"].nunique()
    shared = counts[counts > 1].index
    if len(shared) < 2:
        return None, 0
    wide = (
        judged[judged["pool_id"].isin(shared)]
        .pivot_table(index="pool_id", columns="annotator", values="verdict", aggfunc="first")
        .dropna()
    )
    if wide.shape[1] < 2:
        return None, len(wide)
    from sklearn.metrics import cohen_kappa_score

    first, second = wide.columns[0], wide.columns[1]
    return float(cohen_kappa_score(wide[first], wide[second])), len(wide)


def build_gold(
    judged: pd.DataFrame, key: pd.DataFrame, resolution: str = "strict"
) -> pd.DataFrame:
    """The adjudicated truth: pooled rows that are genuinely entities.

    A row marked wrong but given a ``true_type`` is still an entity — it was
    simply mislabelled — so it enters the gold set under its corrected type.

    ``resolution`` decides what happens where annotators disagree:

    ``strict``   the stricter verdict wins (a row either of them rejected is out)
    ``lenient``  the more permissive verdict wins
    ``exclude``  disagreed rows are dropped from the gold set entirely

    With low inter-annotator agreement this choice does real work, so it is a
    parameter to be reported rather than a constant buried in the code.
    """
    if resolution == "exclude":
        spread = judged.groupby("pool_id")["verdict"].nunique()
        judged = judged[judged["pool_id"].isin(spread[spread == 1].index)]

    # 'n' sorts before 'y', so first == strict and last == lenient.
    ordered = judged.sort_values("verdict")
    take = "last" if resolution == "lenient" else "first"
    resolved = ordered.groupby("pool_id", as_index=False).agg(
        verdict=("verdict", take), true_type=("true_type", "max")
    )
    merged = resolved.merge(key[["pool_id", "doc_id", "norm", "type"]], on="pool_id")
    merged["gold_type"] = merged.apply(
        lambda r: r["true_type"] if r["true_type"] else r["type"], axis=1
    )
    is_entity = (merged["verdict"] == "y") | (merged["true_type"] != "")
    return merged[is_entity][["doc_id", "norm", "gold_type"]].drop_duplicates()


def score(predictions: set, gold: set) -> dict:
    tp = len(predictions & gold)
    fp = len(predictions - gold)
    fn = len(gold - predictions)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


NON_PERSON_HEADER = re.compile(
    r"mailer|daemon|postmaster|no ?reply|href|http|smtp|newsletter|notification|"
    r"\bgps\b|\d",
    re.IGNORECASE,
)


def _looks_like_a_person(norm: str) -> bool:
    """Exclude header names the parser produced that are not people."""
    return not NON_PERSON_HEADER.search(norm)


def header_probe(out: Path, mentions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Independent recall probe from parsed email headers.

    Sender and recipient display names are known deterministically, so a system
    shown that text and missing the name is a miss against ground truth that owes
    nothing to the pool.
    """
    edges_path = out / "email_edges.parquet"
    if not edges_path.exists():
        return pd.DataFrame()

    # The denominator must cover exactly the documents the numerator can draw
    # from. Scoring expected names over all sampled documents while mentions come
    # only from the adjudicated ones halves the apparent recall.
    scored_docs = set().union(*(set(f["doc_id"]) for f in mentions.values())) if mentions else set()
    edges = pd.read_parquet(edges_path)
    edges = edges[edges["doc_id"].isin(scored_docs)]

    expected = set()
    for column in ("source_name", "target_name"):
        for doc_id, name in edges[["doc_id", column]].dropna().itertuples(index=False):
            norm = normalise_surface(str(name))
            # Single tokens are too ambiguous to score fairly, and the parser
            # itself picks up non-people (mailer-daemon addresses, HTML
            # fragments, OCR noise) — counting those as missed entities would
            # measure the header parser rather than the extractors.
            if norm and len(norm.split()) >= 2 and _looks_like_a_person(norm):
                expected.add((doc_id, norm))
    if not expected:
        return pd.DataFrame()

    rows = []
    for system, frame in mentions.items():
        found = {
            (d, n)
            for d, n, t in frame[["doc_id", "norm", "type"]].itertuples(index=False)
            if t == "PERSON"
        }
        hit = len(expected & found)
        rows.append(
            {
                "system": system,
                "header_names": len(expected),
                "found": hit,
                "recall": hit / len(expected),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=REPO_ROOT / "data" / "processed")
    ap.add_argument("--sheets", default=REPO_ROOT / "results" / "adjudication")
    ap.add_argument(
        "--save",
        default=None,
        help="directory to write the result tables to as CSV, for figures and the paper",
    )
    args = ap.parse_args()
    tables: dict[str, pd.DataFrame] = {}

    out, sheets = Path(args.out), Path(args.sheets)
    key = pd.read_csv(sheets / "pool_key.csv", dtype={"pool_id": str})
    judged = load_adjudications(sheets)
    gold_frame = build_gold(judged, key)

    sample_meta = pd.read_parquet(out / "eval_sample.parquet").set_index("doc_id")
    strata = sample_meta["stratum"]
    mentions = {
        system: pd.read_parquet(out / f"mentions_{system}.parquet")
        for system in SYSTEMS
        if (out / f"mentions_{system}.parquet").exists()
    }

    # Score only documents whose pooled rows are *all* adjudicated. A partly
    # judged document would count every un-judged prediction as a false
    # positive, which silently punishes whichever system proposed the most.
    judged_ids = set(judged["pool_id"])
    per_doc = key.groupby("doc_id")["pool_id"].apply(lambda ids: set(ids) <= judged_ids)
    complete = set(per_doc[per_doc].index)
    incomplete = set(per_doc.index) - complete
    if incomplete:
        print(
            f"note: {len(incomplete)} document(s) only partly adjudicated and excluded; "
            f"scoring {len(complete)} complete document(s)."
        )
    gold_frame = gold_frame[gold_frame["doc_id"].isin(complete)]
    mentions = {s: f[f["doc_id"].isin(complete)] for s, f in mentions.items()}
    strata = strata[strata.index.isin(complete)]

    gold_typed = set(gold_frame.itertuples(index=False, name=None))
    gold_untyped = {(d, n) for d, n, _ in gold_typed}

    print(f"adjudicated rows      {judged['pool_id'].nunique():>7,} of {len(key):,} pooled")
    kappa, n_shared = agreement(judged)
    if kappa is not None:
        print(f"annotator agreement   {kappa:>7.3f} (Cohen's kappa, n={n_shared:,})")
    print(f"gold entities         {len(gold_typed):>7,}")

    print("\n== typed (span + label) ==")
    rows = []
    for system, frame in mentions.items():
        preds = set(frame[["doc_id", "norm", "type"]].itertuples(index=False, name=None))
        rows.append({"system": system, **score(preds, gold_typed)})
    tables["typed"] = pd.DataFrame(rows)
    print(tables["typed"].round(3).to_string(index=False))

    print("\n== untyped (span only) ==")
    rows = []
    for system, frame in mentions.items():
        preds = {(d, n) for d, n, _ in frame[["doc_id", "norm", "type"]].itertuples(index=False)}
        rows.append({"system": system, **score(preds, gold_untyped)})
    tables["untyped"] = pd.DataFrame(rows)
    print(tables["untyped"].round(3).to_string(index=False))

    print("\n== typed, by entity type ==")
    rows = []
    for system, frame in mentions.items():
        preds = set(frame[["doc_id", "norm", "type"]].itertuples(index=False, name=None))
        for type_ in sorted({t for _, _, t in gold_typed}):
            g = {x for x in gold_typed if x[2] == type_}
            p = {x for x in preds if x[2] == type_}
            rows.append({"system": system, "type": type_, **score(p, g)})
    tables["by_type"] = pd.DataFrame(rows)
    print(tables["by_type"].round(3).to_string(index=False))

    print("\n== typed, by stratum ==")
    rows = []
    for system, frame in mentions.items():
        preds = set(frame[["doc_id", "norm", "type"]].itertuples(index=False, name=None))
        for stratum in sorted(strata.unique()):
            docs = set(strata[strata == stratum].index)
            g = {x for x in gold_typed if x[0] in docs}
            p = {x for x in preds if x[0] in docs}
            if g or p:
                rows.append({"system": system, "stratum": stratum, **score(p, g)})
    by_stratum = tables["by_stratum"] = pd.DataFrame(rows)
    print(by_stratum.round(3).to_string(index=False))

    print("\n== corpus-weighted estimate (post-stratified) ==")
    print(
        "Per-stratum floors over-sample the rare cells, so the pooled figures above\n"
        "describe the sample, not the corpus. These re-weight each stratum by its\n"
        "share of the sampling frame."
    )
    weights = sample_meta.groupby("stratum")["frame_share"].first()
    weights = weights / weights.sum()
    weighted = []
    for system, group in by_stratum.groupby("system"):
        w = group["stratum"].map(weights).fillna(0.0)
        weighted.append(
            {
                "system": system,
                "precision": float((group["precision"] * w).sum()),
                "recall": float((group["recall"] * w).sum()),
                "f1": float((group["f1"] * w).sum()),
                "strata": len(group),
            }
        )
    tables["corpus_weighted"] = pd.DataFrame(weighted)
    print(tables["corpus_weighted"].round(3).to_string(index=False))

    probe = header_probe(out, mentions)
    if not probe.empty:
        tables["header_probe"] = probe
        print("\n== email-header recall probe (independent of the pool) ==")
        print(probe.round(3).to_string(index=False))

    unlocated = {
        system: int((frame["note"] == "unlocated").sum())
        for system, frame in mentions.items()
        if "note" in frame
    }
    if any(unlocated.values()):
        print("\n== spans the model returned that could not be found in the source ==")
        for system, count in unlocated.items():
            if count:
                print(f"  {system:<8} {count:>5,}")

    if args.save:
        target = Path(args.save)
        target.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_csv(target / f"{name}.csv", index=False)
        print(f"\nwrote {len(tables)} table(s) to {target}")


if __name__ == "__main__":
    main()
