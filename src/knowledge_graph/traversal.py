"""Direct-edge lookup and bounded 1-2 hop traversal over a :class:`GraphStore`.

Two query shapes are handled differently, deliberately:

* **Two named entities** ("how is X connected to Y") — :func:`find_all_connections`
  finds *every* path between exactly those two nodes (a direct edge, plus every
  common-neighbor 2-hop path), not just the single best one. This is the shape
  almost all of the corpus's real relational questions take.
* **One named entity** ("who does X talk to") — :func:`explore` fans out from
  that one seed instead, since there's no second endpoint to path toward.

Reporting many paths as text is cheap, but attaching real chunk text for each
one isn't — see ``ConnectionResult.shortest`` and how callers use it (only
the graph_rag pipeline pulls document evidence, and only for the shortest
path, even though every path found gets described). Everything here still
caps what it returns and ranks by edge weight (communication frequency), so a
high-degree hub node doesn't dump hundreds of facts into the prompt.
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


@dataclass(frozen=True)
class ConnectionResult:
    direct: EdgeFact | None
    two_hop_paths: list[TwoHopFact]  # every common-neighbor path found, ranked by weight desc

    @property
    def shortest(self) -> EdgeFact | TwoHopFact | None:
        """The single path evidence chunks should be pulled for.

        A direct edge is always shortest when one exists. Otherwise the
        best-ranked 2-hop path — there's no way to prefer among ties at the
        same hop count other than weight, so highest-weight wins.
        """
        if self.direct:
            return self.direct
        if self.two_hop_paths:
            return self.two_hop_paths[0]
        return None

    @property
    def found(self) -> bool:
        return self.direct is not None or bool(self.two_hop_paths)


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


def find_all_connections(
    store: GraphStore, a_id: str, b_id: str, max_two_hop: int = DEFAULT_LIMIT
) -> ConnectionResult:
    """Every path between two specific entities: the direct edge (if any),
    plus every common-neighbor 2-hop path, ranked by weight and capped at
    ``max_two_hop``. A direct edge doesn't suppress 2-hop paths from being
    reported too — "they email directly, and also both know Z" is real,
    useful context — it only affects which path is ``.shortest``.
    """
    if a_id not in store.graph or b_id not in store.graph:
        return ConnectionResult(direct=None, two_hop_paths=[])

    direct = _edge_fact(store, a_id, b_id)

    common = set(store.graph.neighbors(a_id)) & set(store.graph.neighbors(b_id))
    ranked_vias = sorted(
        common,
        key=lambda z: store.graph[a_id][z]["weight"] * store.graph[z][b_id]["weight"],
        reverse=True,
    )

    two_hop: list[TwoHopFact] = []
    for z in ranked_vias[:max_two_hop]:
        edge_a_via = _edge_fact(store, a_id, z)
        edge_via_b = _edge_fact(store, z, b_id)
        if edge_a_via and edge_via_b:
            two_hop.append(TwoHopFact(a=a_id, via=z, b=b_id, edge_a_via=edge_a_via, edge_via_b=edge_via_b))

    return ConnectionResult(direct=direct, two_hop_paths=two_hop)


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


__all__ = ["EdgeFact", "TwoHopFact", "ConnectionResult", "direct_edges", "find_all_connections", "explore"]
