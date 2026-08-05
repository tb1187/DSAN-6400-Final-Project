"""Render graph traversal results into the plain-text block appended to the
vector-RAG prompt.

The LLM never sees a graph object — only this text. Every fact carries a
citation (doc_id + subject) back to the real message that evidences it, the
same way a retrieved chunk does, so a graph-derived claim is exactly as
checkable as a semantically-retrieved one.
"""

from __future__ import annotations

from .graph_store import GraphStore
from .traversal import EdgeFact, TwoHopFact

HEADER = "Known relationships from the communication graph (direct email correspondence):"


def _cite(fact: EdgeFact) -> str:
    subj = fact.subjects.split(";")[0].strip() if fact.subjects else ""
    doc = fact.doc_ids[0] if fact.doc_ids else "unknown doc"
    n = f"{fact.n_docs} message{'s' if fact.n_docs != 1 else ''}"
    if subj:
        return f"{n}; e.g. subject '{subj}', {doc}"
    return f"{n}; {doc}"


def _render_edge(store: GraphStore, fact: EdgeFact) -> str:
    a, b = store.canonical(fact.a), store.canonical(fact.b)
    return f"- {a} communicated directly with {b} ({_cite(fact)})"


def _render_two_hop(store: GraphStore, fact: TwoHopFact) -> str:
    a, via, b = store.canonical(fact.a), store.canonical(fact.via), store.canonical(fact.b)
    lines = [
        f"- {a} communicated with {via} ({_cite(fact.edge_a_via)})",
        f"- {via} communicated with {b} ({_cite(fact.edge_via_b)})",
        f"  => {a} connects to {b} via {via} (2 hops; no direct correspondence found between them)",
    ]
    return "\n".join(lines)


def render_connection(store: GraphStore, a_id: str, b_id: str, result: EdgeFact | TwoHopFact | None) -> str:
    """For a two-named-entity query: the specific path between them, or none."""
    a_name, b_name = store.canonical(a_id), store.canonical(b_id)
    if result is None:
        return (
            f"{HEADER}\n"
            f"- No connection was found between {a_name} and {b_name} within 2 hops "
            f"of direct correspondence in this corpus."
        )
    if isinstance(result, EdgeFact):
        return f"{HEADER}\n{_render_edge(store, result)}"
    return f"{HEADER}\n{_render_two_hop(store, result)}"


def render_exploration(store: GraphStore, entity_id: str, direct: list[EdgeFact], two_hop: list[TwoHopFact]) -> str:
    """For a single-named-entity query: fan out from that one seed.

    Several 2-hop facts often share the same intermediary (a high-degree hub
    the seed only has one direct edge to) — the seed->hub line is printed once
    per hub rather than repeated for every 2-hop fact through it.
    """
    if not direct and not two_hop:
        return ""
    lines = [HEADER]
    lines.extend(_render_edge(store, f) for f in direct)

    printed_first_hop: set[str] = set()
    for fact in two_hop:
        a, via, b = store.canonical(fact.a), store.canonical(fact.via), store.canonical(fact.b)
        if fact.via not in printed_first_hop:
            lines.append(f"- {a} communicated with {via} ({_cite(fact.edge_a_via)})")
            printed_first_hop.add(fact.via)
        lines.append(f"- {via} communicated with {b} ({_cite(fact.edge_via_b)})")
        lines.append(f"  => {a} connects to {b} via {via} (2 hops; no direct correspondence found between them)")
    return "\n".join(lines)


__all__ = ["render_connection", "render_exploration"]
