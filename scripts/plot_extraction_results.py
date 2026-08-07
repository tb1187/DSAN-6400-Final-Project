"""Chart the entity-extraction comparison, for the slides and the paper.

Usage:
    python scripts/score_extraction.py --save results/scores
    python scripts/plot_extraction_results.py

Reads the CSVs the scorer writes rather than restating its numbers, so the
figure cannot drift from the evaluation it claims to show.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

#: Categorical slots 1-3 of the validated palette. Colour follows the *system*,
#: never its rank, so a system keeps its hue across both panels.
SYSTEM_COLOUR = {"spacy": "#2a78d6", "llm": "#eb6834", "hybrid": "#1baf7a"}
SYSTEM_LABEL = {"spacy": "spaCy", "llm": "LLM", "hybrid": "Hybrid"}
ORDER = ["spacy", "hybrid", "llm"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"


def style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "text.color": INK,
        "axes.labelcolor": INK_SOFT,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
    })


def bar_labels(ax, bars, values, size=11) -> None:
    """Direct-label every bar.

    Not decoration: the aqua slot sits below 3:1 against this surface, so the
    palette's relief rule requires visible labels. It also removes any need for
    the reader to trace a bar back to a gridline.
    """
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=size,
            color=INK,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default=REPO_ROOT / "results" / "scores")
    ap.add_argument("--out", default=REPO_ROOT / "results" / "figures" / "12_extraction_f1.png")
    args = ap.parse_args()

    scores = Path(args.scores)
    overall = pd.read_csv(scores / "corpus_weighted.csv").set_index("system")
    by_type = pd.read_csv(scores / "by_type.csv")

    style()
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.28}
    )

    # -- left: the headline number ------------------------------------------
    systems = [s for s in ORDER if s in overall.index]
    values = [overall.loc[s, "f1"] for s in systems]
    bars = left.barh(
        range(len(systems)),
        values,
        height=0.5,
        color=[SYSTEM_COLOUR[s] for s in systems],
        linewidth=0,
    )
    bar_labels(left, bars, values, size=12)
    left.set_yticks(range(len(systems)), [SYSTEM_LABEL[s] for s in systems], fontsize=12)
    left.set_xlim(0, 1.0)
    left.set_xlabel("F1 (corpus-weighted)", fontsize=10)
    left.set_title(
        "Overall accuracy", fontsize=13, color=INK, loc="left", pad=12, fontweight="bold"
    )

    # -- right: where the difference actually lives -------------------------
    types = ["PERSON", "GPE", "ORG"]
    # Thin bars with a visible surface gap between them: touching fills read as
    # one stacked mark, and the eye stops separating the systems.
    height = 0.2
    for i, system in enumerate(systems):
        subset = by_type[by_type["system"] == system].set_index("type")
        f1 = [subset.loc[t, "f1"] for t in types]
        offsets = [j + (i - 1) * (height + 0.055) for j in range(len(types))]
        bars = right.barh(
            offsets, f1, height=height, color=SYSTEM_COLOUR[system],
            linewidth=0, label=SYSTEM_LABEL[system],
        )
        bar_labels(right, bars, f1, size=9.5)

    right.set_yticks(range(len(types)), types, fontsize=12)
    right.set_xlim(0, 1.0)
    right.set_xlabel("F1 (typed: span + label)", fontsize=10)
    right.set_title(
        "ORG is the hard category for every system",
        fontsize=13, color=INK, loc="left", pad=12, fontweight="bold",
    )
    right.legend(
        frameon=False, fontsize=10, loc="lower right", ncol=3,
        bbox_to_anchor=(1.0, -0.30), labelcolor=INK_SOFT,
    )

    for ax in (left, right):
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.invert_yaxis()
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    fig.text(
        0.008, -0.02,
        "35 documents, stratified on OCR quality x text style; pooled blind adjudication, 2 annotators. "
        "Inter-annotator kappa = 0.384 - the ranking is stable, absolute values are indicative.",
        fontsize=8.5, color=INK_SOFT,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
