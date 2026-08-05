"""Run the eval suite and print a report.

Usage:
    python scripts/run_eval.py [--config config/config.yaml] [--skip-generation]

Runs, in order:
  1. Self-retrieval and paraphrase-stability sanity checks (label-free).
  2. Faithfulness check on a live RAGPipeline answer for each single_hop gold
     question (needs ANTHROPIC_API_KEY; skip with --skip-generation).

two_hop/community gold questions have no single retrievable answer to check
against, so they're listed for manual read rather than machine-scored — see
config/eval_questions.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.faithfulness import check_faithfulness  # noqa: E402
from src.evaluation.gold_set import load_gold_set  # noqa: E402
from src.evaluation.sanity_checks import paraphrase_stability_check, self_retrieval_check  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=REPO_ROOT / "config" / "config.yaml")
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    eval_cfg = config["evaluation"]
    processed = REPO_ROOT / config["paths"]["processed_data"]

    questions = load_gold_set(REPO_ROOT / eval_cfg["gold_questions_path"])
    checkable = [q for q in questions if q.machine_checkable]
    unchecked = [q for q in questions if not q.machine_checkable]

    print("Loading retriever...")
    retriever = Retriever(args.config)

    if unchecked:
        print(f"\n=== Not machine-scored ({len(unchecked)} two_hop/community questions — grade by manual read) ===")
        for q in unchecked:
            print(f"  {q.id} [{q.category}]: {q.question}")

    print("\n=== Sanity checks (label-free) ===")
    chunks = pd.read_parquet(processed / "chunks.parquet")
    usable = chunks[chunks["usable"]]
    self_ret = self_retrieval_check(retriever, usable, n=50, top_k=5)
    print(f"  Self-retrieval  top1={self_ret.top1_rate:.2f}  top5={self_ret.topk_rate:.2f}  (n={self_ret.n})")

    paraphrases_path = REPO_ROOT / "config" / "eval_paraphrases.yaml"
    if paraphrases_path.exists():
        groups = yaml.safe_load(paraphrases_path.read_text())
        stability = paraphrase_stability_check(retriever, groups)
        print(f"  Paraphrase stability (mean Jaccard)  {stability:.2f}")

    if args.skip_generation:
        return

    print("\n=== Faithfulness (live generation, single_hop questions) ===")
    from src.retrieval.rag import RAGPipeline

    pipe = RAGPipeline(retriever=retriever, config_path=args.config)
    for q in checkable:
        answer = pipe.answer(q.question)
        f = check_faithfulness(answer)
        print(f"  {q.id}  coverage={f.coverage:.2f} ({f.supported_terms}/{f.total_terms})", end="")
        if f.unsupported:
            print(f"  unsupported: {f.unsupported}")
        else:
            print()


if __name__ == "__main__":
    main()
