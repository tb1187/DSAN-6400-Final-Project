"""Load the entity/relationship graph and match query text against it.

Reuses the same quality filter developed in ``scripts/mine_relation_questions.py``:
restrict to ``PER_*`` entity ids (real NER-resolved people, not the ~388 raw
``EML_*`` email-address stub nodes that were never merged into their matching
person) with >=2 mentions, and to the ``communicated_with`` relation (direct,
header-derived correspondence — high precision, unlike the much noisier
``co_occurs_with`` mention-proximity relation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

MIN_MENTIONS = 2
MIN_NAME_LEN = 4  # single "names" shorter than this are too collision-prone to match


@dataclass
class GraphStore:
    nodes: pd.DataFrame  # indexed by entity_id
    graph: nx.Graph  # communicated_with edges only, quality-filtered nodes
    edges: pd.DataFrame  # full communicated_with edge rows, for evidence lookup
    _name_index: list[tuple[str, str]]  # (canonical_name, entity_id), longest first

    def canonical(self, entity_id: str) -> str:
        return self.nodes.loc[entity_id, "canonical"]

    def match_entities(self, query: str) -> list[str]:
        """Entity ids whose canonical name appears in ``query``, longest name first.

        A name already matched is removed from consideration so a shorter name
        contained inside it (e.g. "epstein" inside a longer alias) can't double-count.
        """
        q = query.lower()
        found: list[str] = []
        consumed = [False] * len(q)
        for name, entity_id in self._name_index:
            for m in re.finditer(re.escape(name), q):
                start, end = m.start(), m.end()
                if any(consumed[start:end]):
                    continue
                # word-boundary check so "reid" doesn't match inside "reidentify"
                before_ok = start == 0 or not q[start - 1].isalnum()
                after_ok = end == len(q) or not q[end].isalnum()
                if before_ok and after_ok:
                    found.append(entity_id)
                    for i in range(start, end):
                        consumed[i] = True
                    break
        return found


def load_graph_store(processed_dir: Path | str) -> GraphStore:
    processed_dir = Path(processed_dir)
    nodes = pd.read_parquet(processed_dir / "nodes.parquet").set_index("entity_id")
    edges = pd.read_parquet(processed_dir / "edges.parquet")

    valid = nodes[
        nodes.index.str.startswith("PER_")
        & (nodes["n_mentions"] >= MIN_MENTIONS)
        & (~nodes["noisy"])
    ]
    valid_ids = set(valid.index)

    cw = edges[
        (edges["relation"] == "communicated_with")
        & edges["src"].isin(valid_ids)
        & edges["dst"].isin(valid_ids)
    ]

    graph = nx.Graph()
    for _, row in cw.iterrows():
        graph.add_edge(row["src"], row["dst"], weight=row["weight"])

    # Some canonical names collide across entity ids (imperfect NER resolution);
    # keep the most-mentioned one so a match resolves to the better-evidenced node.
    by_name: dict[str, str] = {}
    for entity_id, row in valid.iterrows():
        name = row["canonical"]
        if not isinstance(name, str) or len(name) < MIN_NAME_LEN:
            continue
        if name not in by_name or valid.loc[by_name[name], "n_mentions"] < row["n_mentions"]:
            by_name[name] = entity_id

    name_index = sorted(by_name.items(), key=lambda pair: len(pair[0]), reverse=True)

    return GraphStore(nodes=nodes, graph=graph, edges=cw, _name_index=name_index)


__all__ = ["GraphStore", "load_graph_store"]
