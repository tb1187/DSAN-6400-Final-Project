"""Resolve extracted mentions into canonical entities.

Usage:
    python scripts/resolve_entities.py [--mentions mentions_spacy_corpus.parquet]

Writes ``entities.parquet`` (one row per canonical entity) and
``entity_aliases.parquet`` (one row per surface form, pointing at its entity).
Email actors parsed deterministically from headers are linked in where their
names match a resolved person, so those nodes carry verified addresses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.extraction.schema import normalise_surface  # noqa: E402
from src.knowledge_graph.resolution import resolve  # noqa: E402


def load_email_actors(out: Path) -> pd.DataFrame:
    """Display-name/address pairs from the deterministic email tier.

    Drawn from two tables because neither is complete on its own: the edge list
    pairs a recipient's name with its address only when the header carried both,
    while the message table pairs sender names with sender addresses directly.
    """
    rows = []

    edges_path = out / "email_edges.parquet"
    if edges_path.exists():
        edges = pd.read_parquet(edges_path)
        for name_col, key_col in (("source_name", "source"), ("target_name", "target")):
            for name, key in edges[[name_col, key_col]].dropna().itertuples(index=False):
                rows.append(
                    {
                        "norm": normalise_surface(str(name)),
                        "address": str(key) if "@" in str(key) else "",
                    }
                )

    messages_path = out / "email_messages.parquet"
    if messages_path.exists():
        messages = pd.read_parquet(messages_path)
        senders = messages[["sender_name", "sender_address"]].dropna(subset=["sender_name"])
        for name, address in senders.itertuples(index=False):
            rows.append(
                {
                    "norm": normalise_surface(str(name)),
                    "address": str(address) if isinstance(address, str) else "",
                }
            )

    actors = pd.DataFrame(rows)
    if actors.empty:
        return pd.DataFrame(columns=["norm", "address"])
    # Every occurrence is kept, not deduplicated: how often a (name, address)
    # pair is observed is exactly the evidence the linker uses to decide whether
    # to trust it.
    return actors[actors["norm"].str.split().str.len() >= 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=REPO_ROOT / "data" / "processed")
    ap.add_argument("--mentions", default="mentions_spacy_corpus.parquet")
    args = ap.parse_args()

    out = Path(args.out)
    mentions = pd.read_parquet(out / args.mentions)
    actors = load_email_actors(out)

    entities, aliases = resolve(mentions, actors)
    entities.to_parquet(out / "entities.parquet", index=False)
    aliases.to_parquet(out / "entity_aliases.parquet", index=False)

    surface_forms = mentions.groupby(["norm", "type"]).ngroups
    print(f"surface forms in      {surface_forms:>8,}")
    print(f"entities out          {len(entities):>8,}  ({1 - len(entities)/surface_forms:.0%} reduction)")
    print(f"mentions covered      {int(entities['n_mentions'].sum()):>8,}")
    print(f"entities with address {int((entities['addresses'] != '').sum()):>8,}")
    if "actor_match" in entities.attrs:
        stats = entities.attrs["actor_match"]
        total = sum(stats.values())
        print(f"\nemail actor linking ({total:,} name occurrences):")
        for how in ("exact", "reordered", "fuzzy", "unmatched"):
            if stats.get(how):
                print(f"  {how:<10} {stats[how]:>6,}")
        print(f"  address variants merged: {entities.attrs.get('address_variants_merged', 0):,}")
    print("\nby type:")
    print(
        entities.groupby("type")
        .agg(entities=("entity_id", "size"), mentions=("n_mentions", "sum"))
        .to_string()
    )
    print("\nhow aliases were resolved:")
    print(aliases["resolution"].value_counts().to_string())
    print("\nlargest entities:")
    print(
        entities.nlargest(15, "n_mentions")[
            ["type", "canonical", "n_aliases", "n_mentions"]
        ].to_string(index=False)
    )
    print(f"\nwrote {out / 'entities.parquet'} and {out / 'entity_aliases.parquet'}")


if __name__ == "__main__":
    main()
