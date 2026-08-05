"""Mine candidate relational questions from the real communication graph.

Usage:
    python scripts/mine_relation_questions.py [--out config/relation_candidates.yaml]

Reads ``nodes.parquet``/``edges.parquet`` (the ``communicated_with`` relation —
PERSON<->PERSON, high-confidence, header-derived) and surfaces real candidates
for the eval question set, stratified the way a graph-RAG comparison needs:

  1-hop     direct edges — a relational control, both systems should do fine
  2-hop     A-Z-B with a real path but no direct A-B edge — the actual test:
            can the system trace a relationship no single chunk states outright
  no-path   pairs in different connected components — negative controls, so a
            "confidently invented connection" failure mode is catchable

2-hop candidates are deliberately sampled away from the Epstein hub (degree
366 in this graph — trivially "everyone is 2 hops from him") so the stratum
tests real multi-hop tracing, not just hub recognition. Each candidate ships
with its real supporting evidence (doc_ids/subjects per edge) pulled straight
from the edges table, so a question can be hand-verified before it's promoted
into eval_questions.yaml — this script proposes candidates, it doesn't decide
final question wording.

Mining is restricted to ``PER_*`` node ids with >=2 mentions. ``nodes.parquet``
also contains ~388 ``EML_*`` stub nodes — unresolved email-address fragments
that were never merged into their matching ``PER_*`` entity (e.g. "stephen
immelt" / "immelt stephen j" exist as two separate nodes for the same real
person). Mining against the full node set produces "two-hop" chains that are
really just the same handful of people under duplicate ids — a real
entity-resolution gap upstream, not something this filter fixes, just avoids.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

HUB_DEGREE_THRESHOLD = 50  # nodes above this are excluded as 2-hop intermediaries


def build_graph(edges: pd.DataFrame, valid_ids: set[str]) -> nx.Graph:
    cw = edges[edges["relation"] == "communicated_with"]
    cw = cw[cw["src"].isin(valid_ids) & cw["dst"].isin(valid_ids)]
    graph = nx.Graph()
    for _, row in cw.iterrows():
        graph.add_edge(row["src"], row["dst"], weight=row["weight"], n_docs=row["n_docs"])
    return graph


def _evidence(edges: pd.DataFrame, a: str, b: str) -> dict:
    row = edges[
        (edges["relation"] == "communicated_with")
        & (((edges["src"] == a) & (edges["dst"] == b)) | ((edges["src"] == b) & (edges["dst"] == a)))
    ]
    if row.empty:
        return {}
    row = row.iloc[0]
    doc_ids = row["doc_ids"].split("; ")[:3] if pd.notna(row["doc_ids"]) else []
    return {
        "weight": float(row["weight"]),
        "n_docs": int(row["n_docs"]),
        "subjects": (row["subjects"] or "")[:200] if pd.notna(row["subjects"]) else "",
        "doc_ids": doc_ids,
    }


def mine_one_hop(graph: nx.Graph, nodes: pd.DataFrame, edges: pd.DataFrame, n: int, rng: random.Random) -> list[dict]:
    candidates = list(graph.edges())
    rng.shuffle(candidates)
    out = []
    for a, b in candidates:
        if len(out) >= n:
            break
        out.append(
            {
                "category": "one_hop",
                "a": nodes.loc[a, "canonical"],
                "b": nodes.loc[b, "canonical"],
                "evidence": _evidence(edges, a, b),
            }
        )
    return out


def mine_two_hop(graph: nx.Graph, nodes: pd.DataFrame, edges: pd.DataFrame, n: int, rng: random.Random) -> list[dict]:
    degrees = dict(graph.degree())
    intermediaries = [node for node, d in degrees.items() if 2 <= d < HUB_DEGREE_THRESHOLD]
    rng.shuffle(intermediaries)

    out = []
    seen_pairs = set()
    for z in intermediaries:
        if len(out) >= n:
            break
        neighbors = list(graph.neighbors(z))
        rng.shuffle(neighbors)
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a, b = neighbors[i], neighbors[j]
                if graph.has_edge(a, b):
                    continue  # direct edge exists, not a true 2-hop case
                pair_key = tuple(sorted((a, b)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                out.append(
                    {
                        "category": "two_hop",
                        "a": nodes.loc[a, "canonical"],
                        "b": nodes.loc[b, "canonical"],
                        "via": nodes.loc[z, "canonical"],
                        "a_via_evidence": _evidence(edges, a, z),
                        "via_b_evidence": _evidence(edges, z, b),
                    }
                )
                break  # one candidate per intermediary is enough diversity
            if len(out) >= n:
                break
    return out


def mine_no_path(graph: nx.Graph, nodes: pd.DataFrame, n: int, rng: random.Random) -> list[dict]:
    components = [c for c in nx.connected_components(graph) if len(c) >= 2]
    components.sort(key=len, reverse=True)
    if len(components) < 2:
        return []

    giant = list(components[0])
    others = [node for comp in components[1:] for node in comp]
    rng.shuffle(giant)
    rng.shuffle(others)

    out = []
    for a, b in zip(giant, others):
        if len(out) >= n:
            break
        out.append(
            {
                "category": "no_path",
                "a": nodes.loc[a, "canonical"],
                "b": nodes.loc[b, "canonical"],
                "note": "different connected components in the communicated_with graph — no path should be found",
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=REPO_ROOT / "config" / "relation_candidates.yaml")
    ap.add_argument("--n-per-category", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    processed = REPO_ROOT / "data" / "processed"
    nodes = pd.read_parquet(processed / "nodes.parquet").set_index("entity_id")
    edges = pd.read_parquet(processed / "edges.parquet")

    valid_ids = set(
        nodes[
            nodes.index.str.startswith("PER_") & (nodes["n_mentions"] >= 2) & (~nodes["noisy"])
        ].index
    )
    graph = build_graph(edges, valid_ids)

    rng = random.Random(args.seed)
    candidates = (
        mine_one_hop(graph, nodes, edges, args.n_per_category, rng)
        + mine_two_hop(graph, nodes, edges, args.n_per_category, rng)
        + mine_no_path(graph, nodes, args.n_per_category, rng)
    )

    out_path = Path(args.out)
    out_path.write_text(yaml.safe_dump(candidates, sort_keys=False, allow_unicode=True))

    by_cat: dict[str, int] = {}
    for c in candidates:
        by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1
    print("candidates mined:")
    for cat, count in by_cat.items():
        print(f"  {cat:<10} {count}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
