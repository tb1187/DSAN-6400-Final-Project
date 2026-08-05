"""Direct-edge lookup and bounded 1-2 hop traversal over a :class:`GraphStore`.

Two query shapes are handled differently, deliberately:

* **Two named entities** ("how is X connected to Y") — :func:`find_connection`
  looks for the specific path between exactly those two nodes: a direct edge
  first, else the single best common neighbor. This is the shape almost all
  of the corpus's real relational questions take.
* **One named entity** ("who does X talk to") — :func:`explore` fans out from
  that one seed instead, since there's no second endpoint to path toward.

Both cap what they return and rank by edge weight (communication frequency),
so a high-degree hub node doesn't dump dozens of facts into the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .graph_store import GraphStore

DEFAULT_LIMIT = 8


@dataclass(frozen=True)
class EdgeFact:
    a: str
    b: str
    weight: float
    n_docs: int
    subjects: str
    doc_ids: list[str]


@dataclass(frozen=True)
class TwoHopFact:
    a: str
    via: str
    b: str
    edge_a_via: EdgeFact
    edge_via_b: EdgeFact


def _edge_fact(store: GraphStore, a: str, b: str) -> EdgeFact | None:
    row = store.edges[
        ((store.edges["src"] == a) & (store.edges["dst"] == b))
        | ((store.edges["src"] == b) & (store.edges["dst"] == a))
    ]
    if row.empty:
        return None
    row = row.iloc[0]
    doc_ids = row["doc_ids"].split("; ")[:3] if pd.notna(row["doc_ids"]) else []
    subjects = row["subjects"] if pd.notna(row["subjects"]) else ""
    return EdgeFact(a=a, b=b, weight=float(row["weight"]), n_docs=int(row["n_docs"]), subjects=subjects, doc_ids=doc_ids)


def direct_edges(store: GraphStore, entity_id: str, limit: int = DEFAULT_LIMIT) -> list[EdgeFact]:
    if entity_id not in store.graph:
        return []
    neighbors = sorted(
        store.graph.neighbors(entity_id),
        key=lambda n: store.graph[entity_id][n]["weight"],
        reverse=True,
    )
    facts = []
    for n in neighbors[:limit]:
        fact = _edge_fact(store, entity_id, n)
        if fact:
            facts.append(fact)
    return facts


def find_connection(store: GraphStore, a_id: str, b_id: str) -> EdgeFact | TwoHopFact | None:
    """The path between two specific entities: direct edge, else best 2-hop, else None."""
    if a_id not in store.graph or b_id not in store.graph:
        return None

    direct = _edge_fact(store, a_id, b_id)
    if direct:
        return direct

    common = set(store.graph.neighbors(a_id)) & set(store.graph.neighbors(b_id))
    if not common:
        return None

    best_via = max(
        common,
        key=lambda z: store.graph[a_id][z]["weight"] * store.graph[z][b_id]["weight"],
    )
    edge_a_via = _edge_fact(store, a_id, best_via)
    edge_via_b = _edge_fact(store, best_via, b_id)
    if not edge_a_via or not edge_via_b:
        return None
    return TwoHopFact(a=a_id, via=best_via, b=b_id, edge_a_via=edge_a_via, edge_via_b=edge_via_b)


def explore(store: GraphStore, entity_id: str, limit: int = 5) -> tuple[list[EdgeFact], list[TwoHopFact]]:
    """Direct edges plus the best 2-hop facts fanning out from one seed entity."""
    direct = direct_edges(store, entity_id, limit=limit)
    direct_neighbors = {f.b for f in direct}

    two_hop: list[TwoHopFact] = []
    for first_hop in direct[: min(len(direct), limit)]:
        z = first_hop.b
        for second in direct_edges(store, z, limit=3):
            target = second.b
            if target == entity_id or target in direct_neighbors:
                continue
            two_hop.append(TwoHopFact(a=entity_id, via=z, b=target, edge_a_via=first_hop, edge_via_b=second))
    two_hop.sort(key=lambda f: f.edge_a_via.weight * f.edge_via_b.weight, reverse=True)
    return direct, two_hop[:limit]


__all__ = ["EdgeFact", "TwoHopFact", "direct_edges", "find_connection", "explore"]
