"""Build the graph's node and edge tables from resolved entities.

Usage:
    python scripts/build_edges.py [--tiers 1,2]

Writes:
    entity_mentions.parquet   every mention with the entity it resolves to
    nodes.parquet             entities plus email actors NER never found
    edges.parquet             all tiers, stacked
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.knowledge_graph.edges import (  # noqa: E402
    co_recipient_edges,
    communication_edges,
    cooccurrence_edges,
    link_mentions,
    noisy_entities,
    resolve_actors,
    thread_edges,
)

EDGE_COLUMNS = [
    "src",
    "dst",
    "relation",
    "tier",
    "weight",
    "n_docs",
    "doc_ids",
    "page_bates",
    "n_rows",
    "n_cc",
    "n_chunks",
    "subjects",
    "first_sent",
    "last_sent",
    "span_days",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=REPO_ROOT / "data" / "processed")
    ap.add_argument("--mentions", default="mentions_spacy_corpus.parquet")
    ap.add_argument("--tiers", default="1,2", help="which tiers to build")
    args = ap.parse_args()

    out = Path(args.out)
    tiers = {int(t) for t in args.tiers.split(",") if t.strip()}

    entities = pd.read_parquet(out / "entities.parquet")
    aliases = pd.read_parquet(out / "entity_aliases.parquet")
    mentions = pd.read_parquet(out / args.mentions)

    linked = link_mentions(mentions, aliases)
    linked.to_parquet(out / "entity_mentions.parquet", index=False)
    unlinked = int(linked["entity_id"].isna().sum())
    print(f"mentions linked       {len(linked) - unlinked:>8,} / {len(linked):,}")
    print(f"  in prepended header {int(linked['in_header'].sum()):>8,}  (excluded from tier 2)")

    nodes = entities.copy()
    nodes["source"] = "ner"
    all_edges = []

    if 1 in tiers:
        email_edges = pd.read_parquet(out / "email_edges.parquet")
        email_messages = pd.read_parquet(out / "email_messages.parquet")
        actors = resolve_actors(email_edges, email_messages, entities, aliases)
        comms = communication_edges(actors)
        stats = comms.attrs["actor_resolution"]
        minted = actors.minted

        print(f"\ntier 1 — communicated_with")
        print(f"  actors            {comms.attrs['n_actors']:>8,}")
        for how in ("address", "name", "name_reordered", "display_name", "minted"):
            if stats.get(how):
                print(f"    {how:<16}{stats[how]:>8,}")
        print(f"  routing rows drop {comms.attrs['routing_rows_dropped']:>8,}  (teletype, not email)")
        print(f"  contested addrs   {comms.attrs['contested_addresses']:>8,}  (given to dominant claimant)")
        print(f"  self-edges dropped{comms.attrs['self_edges_dropped']:>8,}")
        print(f"  edges             {len(comms):>8,}")
        print(f"  linked to an NER entity: "
              f"{1 - len(minted) / max(comms.attrs['n_actors'], 1):.0%} of actors")

        dated = comms["first_sent"].notna()
        print(f"\n  dates: {int(dated.sum()):,}/{len(comms):,} edges dated"
              f"  ({comms.attrs['dates_unparsed']:,} rows unparsed,"
              f" {comms.attrs['dates_out_of_range']:,} out of corpus range)")
        if dated.any():
            print(f"  span:  {comms.loc[dated, 'first_sent'].min():%Y-%m-%d}"
                  f" .. {comms.loc[dated, 'last_sent'].max():%Y-%m-%d}")
            per_year = actors.rows.dropna(subset=["sent_at"]).drop_duplicates("message_key")
            counts = per_year["sent_at"].dt.year.value_counts().sort_index()
            print("  messages/year: " + "  ".join(f"{y}:{n}" for y, n in counts.items()))

        if len(minted):
            nodes = pd.concat([nodes, minted], ignore_index=True)
        all_edges.append(comms)

        print("\n  strongest ties (by distinct messages):")
        print(_label(comms.head(10), nodes, extra=["first_sent", "last_sent"]).to_string(index=False))

        corec = co_recipient_edges(actors)
        print(f"\ntier 1 — co_recipient_of")
        print(f"  edges             {len(corec):>8,}"
              f"   ({int((corec['n_cc'] > 0).sum()):,} involve a cc)")
        all_edges.append(corec)
        print("\n  most-shared audiences:")
        print(_label(corec.head(8), nodes).to_string(index=False))

        threads, thread_nodes = thread_edges(actors)
        print(f"\ntier 1 — participated_in")
        print(f"  threads           {len(thread_nodes):>8,}")
        if len(thread_nodes):
            print(f"  messages threaded {int(thread_nodes['n_mentions'].sum()):>8,}")
            print(f"  edges             {len(threads):>8,}")
            print(f"  participants/thread: median {thread_nodes['n_participants'].median():.0f}"
                  f"  max {thread_nodes['n_participants'].max()}")
            nodes = pd.concat([nodes, thread_nodes], ignore_index=True)
            all_edges.append(threads)
            print("\n  biggest threads:")
            print(
                thread_nodes.nlargest(8, "n_mentions")[
                    ["canonical", "n_mentions", "n_participants", "first_sent", "last_sent"]
                ]
                .rename(columns={"canonical": "subject", "n_mentions": "n_messages"})
                .to_string(index=False)
            )

    if 2 in tiers:
        noisy = noisy_entities(entities)
        nodes["noisy"] = nodes["entity_id"].isin(noisy)
        cooc = cooccurrence_edges(linked, exclude=noisy)
        print(f"\ntier 2 — co_occurs_with")
        print(f"  chunks used       {cooc.attrs['chunks_used']:>8,}"
              f"  ({cooc.attrs['chunks_oversized']:,} dropped as oversized)")
        print(f"  entities excluded {cooc.attrs['entities_excluded']:>8,}  (salutations, line-break spans)")
        print(f"  entities kept     {cooc.attrs['entities_frequent']:>8,}")
        print(f"  candidate pairs   {cooc.attrs['candidate_pairs']:>8,}")
        print(f"  single-doc dropped{cooc.attrs['dropped_single_doc']:>8,}")
        print(f"  edges             {len(cooc):>8,}")
        all_edges.append(cooc)

        print("\n  strongest associations (by NPMI, ≥20 chunks):")
        print(_label(cooc[cooc["n_chunks"] >= 20].head(10), nodes).to_string(index=False))
        print("\n  most co-mentioned people:")
        types = dict(zip(nodes["entity_id"], nodes["type"]))
        people = cooc[
            cooc["src"].map(types).eq("PERSON") & cooc["dst"].map(types).eq("PERSON")
        ]
        print(_label(people.nlargest(10, "n_chunks"), nodes).to_string(index=False))

    edges = pd.concat(all_edges, ignore_index=True)
    for column in EDGE_COLUMNS:
        if column not in edges:
            edges[column] = pd.NA
    edges = edges[EDGE_COLUMNS]

    nodes.to_parquet(out / "nodes.parquet", index=False)
    edges.to_parquet(out / "edges.parquet", index=False)
    print(f"\nnodes {len(nodes):,}  edges {len(edges):,}")
    print(edges.groupby(["tier", "relation"]).size().to_string())
    print(f"\nwrote {out / 'nodes.parquet'} and {out / 'edges.parquet'}")


def _label(edges: pd.DataFrame, nodes: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    """Swap entity ids for canonical names, for reading."""
    names = dict(zip(nodes["entity_id"], nodes["canonical"]))
    weight_cols = [c for c in ("weight", "n_chunks", "n_docs", *(extra or [])) if c in edges]
    out = pd.DataFrame(
        {
            "source": edges["src"].map(names),
            "target": edges["dst"].map(names),
            **{c: edges[c] for c in weight_cols},
        }
    )
    return out


if __name__ == "__main__":
    main()
