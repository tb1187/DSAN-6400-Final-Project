"""Extract the email communication network from the OCR text.

Usage:
    python scripts/build_email_edges.py [--out data/processed]

The load file tags only 64 documents as email, but thousands of the documents in
this production are emails that were *printed*, so their ``From:/Sent:/To:``
block survives in the extracted text. This is the deterministic tier of the
knowledge graph: high precision, exact document and page provenance, and no model
involved.

Writes two tables:

``email_messages.parquet``
    One row per header block found, including forwarded blocks nested inside a
    document. ``message_key`` is a hash of the message itself, so the same email
    quoted across several documents can be collapsed.

``email_edges.parquet``
    One row per (message, recipient) pair — the sender→recipient edges, carrying
    the document and page they were observed on.

Both keep *raw* surface forms. Resolving ``jeffrey E.`` / ``Jeffrey Epstein`` /
``jeeyacation@gmail.com`` to one actor is entity resolution's job, deliberately
not done here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ingestion.email_headers import edge_list, messages_to_frame, parse_headers  # noqa: E402
from src.ingestion.loadfile import load_text  # noqa: E402
from src.ingestion.normalize import page_at_offset, page_marks_from_raw  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=REPO_ROOT / "data" / "processed")
    args = ap.parse_args()

    out = Path(args.out)
    manifest = pd.read_parquet(out / "manifest.parquet")

    all_messages = []
    page_by_block: dict[tuple[str, int], str | None] = {}

    for doc_id, path in manifest.set_index("doc_id")["text_path"].items():
        if not isinstance(path, str):
            continue
        raw = load_text(path)
        messages = parse_headers(doc_id, raw)
        if not messages:
            continue
        marks = page_marks_from_raw(raw)
        for message in messages:
            page_by_block[(doc_id, message.block_index)] = page_at_offset(
                marks, message.char_start
            )
        all_messages.extend(messages)

    messages_df = messages_to_frame(all_messages)
    messages_df["message_key"] = [m.message_key() for m in all_messages]
    messages_df["char_start"] = [m.char_start for m in all_messages]
    messages_df["char_end"] = [m.char_end for m in all_messages]
    messages_df["page_bates"] = [
        page_by_block[(m.doc_id, m.block_index)] for m in all_messages
    ]

    edges = edge_list(all_messages)
    keys = messages_df.set_index(["doc_id", "block_index"])[["message_key", "page_bates"]]
    edges = edges.join(keys, on=["doc_id", "block_index"])

    messages_df.to_parquet(out / "email_messages.parquet", index=False)
    edges.to_parquet(out / "email_edges.parquet", index=False)

    actors = set(edges["source"]) | set(edges["target"])
    print(f"documents with headers   {messages_df['doc_id'].nunique():>8,}")
    print(f"header blocks parsed     {len(messages_df):>8,}")
    print(f"distinct messages        {messages_df['message_key'].nunique():>8,}")
    print(f"forwarded blocks         {int(messages_df['is_forwarded'].sum()):>8,}")
    print(f"sender -> recipient rows {len(edges):>8,}")
    print(f"unique directed pairs    {edges.groupby(['source', 'target']).ngroups:>8,}")
    print(f"distinct actors          {len(actors):>8,}")
    print(f"edges with a page        {int(edges['page_bates'].notna().sum()):>8,}")
    print(f"\nwrote {out / 'email_messages.parquet'}")
    print(f"wrote {out / 'email_edges.parquet'}")


if __name__ == "__main__":
    main()
