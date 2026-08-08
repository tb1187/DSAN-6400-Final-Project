"""Answer every gold question with both pipelines, grade each answer with an
LLM judge, and print a comparison table.

Usage:
    python scripts/grade_answers.py [--out results/graded_answers.json]

For each question in config/eval_questions.yaml: run RAGPipeline.answer() and
GraphAugmentedRAGPipeline.answer(), then ask Claude to grade each answer 0-10
against a reference (the real chunk text for machine-checkable questions,
the hand-written notes field otherwise). Full answers + grades are written to
--out; a summary table prints to stdout.

Note: this is LLM-as-judge grading, which the project intentionally avoided
using as the primary evaluation method earlier (see conversation) because the
gold set is small enough to grade by hand and self-preference bias is a real
concern when Claude grades Claude's own answers. Treat this as a quick,
coarse signal, not a substitute for reading the graded answers yourself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anthropic
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.gold_set import GoldQuestion, load_gold_set  # noqa: E402
from src.retrieval.graph_rag import GraphAugmentedRAGPipeline  # noqa: E402
from src.retrieval.rag import RAGPipeline  # noqa: E402

GRADE_PROMPT = """You are grading an AI-generated answer to a question about a legal/investigative document corpus.

Question: {question}

Reference information (real, verified facts relevant to this question — not required wording, just what's actually true):
{reference}

Answer to grade:
{answer}

Grade this answer's correctness and completeness on a 0-10 scale:
- 10 = fully correct, complete, and appropriately grounded
- 0 = confidently wrong or fabricated (e.g. inventing a connection/fact that contradicts the reference)
A correct "not found" / "no connection found" / "I don't have enough information" answer should score HIGH if the reference confirms that's actually true — do not penalize correct abstention just for being short.

Respond with ONLY valid JSON, no markdown fences: {{"grade": <integer 0-10>, "reasoning": "<one sentence>"}}"""


def _reference_text(q: GoldQuestion, chunks: pd.DataFrame) -> str:
    if q.expected_chunk_ids:
        texts = chunks.loc[chunks["chunk_id"].isin(q.expected_chunk_ids), "text"]
        if not texts.empty:
            return "\n\n".join(texts.tolist())
    return q.notes or "(no reference available for this question)"


def grade(client: anthropic.Anthropic, model: str, question: str, reference: str, answer: str) -> dict:
    prompt = GRADE_PROMPT.format(question=question, reference=reference, answer=answer)
    message = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in message.content if b.type == "text").strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"grade": None, "reasoning": f"unparseable judge output: {text[:120]!r}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=REPO_ROOT / "config" / "config.yaml")
    ap.add_argument("--out", default=REPO_ROOT / "results" / "graded_answers.json")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    questions = load_gold_set(REPO_ROOT / config["evaluation"]["gold_questions_path"])
    chunks = pd.read_parquet(REPO_ROOT / config["paths"]["processed_data"] / "chunks.parquet")

    print("Loading pipelines...")
    plain = RAGPipeline(config_path=args.config)
    graph = GraphAugmentedRAGPipeline(retriever=plain.retriever, config_path=args.config)
    client = anthropic.Anthropic()
    model = config["retrieval"]["anthropic_model"]

    rows = []
    for q in questions:
        print(f"  {q.id} [{q.category}] {q.question}")
        reference = _reference_text(q, chunks)

        plain_result = plain.answer(q.question)
        graph_result = graph.answer(q.question)

        plain_grade = grade(client, model, q.question, reference, plain_result.answer)
        graph_grade = grade(client, model, q.question, reference, graph_result.answer)

        rows.append(
            {
                "id": q.id,
                "category": q.category,
                "question": q.question,
                "plain_answer": plain_result.answer,
                "plain_grade": plain_grade["grade"],
                "plain_reasoning": plain_grade["reasoning"],
                "graph_answer": graph_result.answer,
                "graph_grade": graph_grade["grade"],
                "graph_reasoning": graph_grade["reasoning"],
                "graph_context_used": bool(graph_result.graph_context),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, default=str))

    df = pd.DataFrame(rows)
    print()
    print(df[["id", "category", "plain_grade", "graph_grade"]].to_string(index=False))
    print()
    print("mean by category:")
    print(df.groupby("category")[["plain_grade", "graph_grade"]].mean().round(2))
    print()
    print(f"wrote full answers + grades to {out_path}")


if __name__ == "__main__":
    main()
