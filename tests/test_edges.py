"""Tests for graph edge construction.

Weighted towards the failures that actually happened: an address landing on the
wrong node, evidence caps corrupting counts, and boilerplate outranking people.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.knowledge_graph.edges import (
    _drop_routing_headers,
    communication_edges,
    cooccurrence_edges,
    link_mentions,
    noisy_entities,
)


def make_entities(rows):
    return pd.DataFrame(
        rows, columns=["entity_id", "type", "canonical", "n_aliases", "n_mentions", "addresses"]
    )


def make_aliases(rows):
    return pd.DataFrame(rows, columns=["entity_id", "type", "norm", "n_mentions", "resolution"])


def make_email_edges(rows):
    columns = [
        "doc_id", "block_index", "source", "target", "source_name", "target_name",
        "is_cc", "subject", "sent_raw", "message_key", "page_bates",
    ]
    return pd.DataFrame(rows, columns=columns)


EMPTY_MESSAGES = pd.DataFrame({"sender_address": pd.Series([], dtype=str)})


# --------------------------------------------------------------------------
# tier 1
# --------------------------------------------------------------------------


def test_contested_address_goes_to_dominant_claimant():
    """The bug that gave Epstein's mailbox to a 2-mention header fragment."""
    entities = make_entities([
        ("PER_1", "PERSON", "jeffrey epstein", 3, 24678, "jee@gmail.com"),
        ("PER_2", "PERSON", "darren indyke jeffrey epstein", 1, 2, "jee@gmail.com"),
    ])
    aliases = make_aliases([
        ("PER_1", "PERSON", "jeffrey epstein", 24678, "exact"),
        ("PER_2", "PERSON", "darren indyke jeffrey epstein", 2, "exact"),
    ])
    edges = make_email_edges([
        ("D1", 0, "jee@gmail.com", "reid weingarten", "Jeffrey", "Reid", False, "Re: x", "1/1/2015", "m1", "P1"),
    ])
    result, _ = communication_edges(edges, EMPTY_MESSAGES, entities, aliases)
    assert result.iloc[0]["src"] == "PER_1"
    assert result.attrs["contested_addresses"] == 1


def test_single_token_key_links_only_on_exact_alias():
    """"weingarten" should reach Reid's node; an unknown surname should not."""
    entities = make_entities([("PER_1", "PERSON", "reid weingarten", 2, 500, "")])
    aliases = make_aliases([
        ("PER_1", "PERSON", "reid weingarten", 400, "exact"),
        ("PER_1", "PERSON", "weingarten", 100, "bare_surname"),
    ])
    edges = make_email_edges([
        ("D1", 0, "weingarten", "nobodyknown", "Weingarten", "Nobody", False, "Re: x", "1/1/2015", "m1", "P1"),
    ])
    result, minted = communication_edges(edges, EMPTY_MESSAGES, entities, aliases)
    assert result.iloc[0]["src"] == "PER_1"
    assert len(minted) == 1  # the unknown target is minted, not dropped
    assert result.iloc[0]["dst"].startswith("EML_")


def test_reordered_name_matches():
    entities = make_entities([("PER_1", "PERSON", "reid weingarten", 1, 10, "")])
    aliases = make_aliases([("PER_1", "PERSON", "reid weingarten", 10, "exact")])
    edges = make_email_edges([
        ("D1", 0, "weingarten, reid", "x y", "Weingarten, Reid", "X Y", False, "s", "1/1/2015", "m1", "P1"),
    ])
    result, _ = communication_edges(edges, EMPTY_MESSAGES, entities, aliases)
    assert result.iloc[0]["src"] == "PER_1"
    assert result.attrs["actor_resolution"]["name_reordered"] == 1


def test_weight_counts_distinct_messages_not_rows():
    """A forwarded message appearing in three documents is still one message."""
    entities = make_entities([])
    aliases = make_aliases([])
    edges = make_email_edges([
        ("D1", 0, "a b", "c d", "A B", "C D", False, "s", "1/1/2015", "m1", "P1"),
        ("D2", 0, "a b", "c d", "A B", "C D", False, "s", "1/1/2015", "m1", "P2"),
        ("D3", 0, "a b", "c d", "A B", "C D", False, "s", "1/1/2015", "m2", "P3"),
    ])
    result, _ = communication_edges(edges, EMPTY_MESSAGES, entities, aliases)
    assert result.iloc[0]["weight"] == 2
    assert result.iloc[0]["n_rows"] == 3
    assert result.iloc[0]["n_docs"] == 3


def test_routing_headers_dropped_but_real_mail_kept():
    edges = make_email_edges([
        ("D1", 0, "miami", "miami attn: ssa", "Miami", "Miami Attn: SSA", False, None, None, "m1", "P1"),
        ("D2", 0, "a b", "c d", "A B", "C D", False, "Re: lunch", None, "m2", "P2"),
        ("D3", 0, "e@x.com", "f g", "E", "F G", False, None, None, "m3", "P3"),
    ])
    kept, dropped = _drop_routing_headers(edges)
    assert dropped == 1
    assert set(kept["message_key"]) == {"m2", "m3"}


# --------------------------------------------------------------------------
# tier 2
# --------------------------------------------------------------------------


def make_linked(pairs):
    """``pairs`` is (chunk_id, doc_id, entity_id, in_header)."""
    return pd.DataFrame(pairs, columns=["chunk_id", "doc_id", "entity_id", "in_header"])


def test_cooccurrence_requires_multiple_documents():
    """A pair confined to one document is a repeating signature block."""
    same_doc = make_linked(
        [(f"c{i}", "D1", e, False) for i in range(5) for e in ("E1", "E2")]
    )
    edges = cooccurrence_edges(same_doc, min_entity_df=3, min_pair_df=3, min_npmi=0.0)
    assert edges.empty
    assert edges.attrs["dropped_single_doc"] == 1

    spread = make_linked(
        [(f"c{i}", f"D{i}", e, False) for i in range(5) for e in ("E1", "E2")]
    )
    edges = cooccurrence_edges(spread, min_entity_df=3, min_pair_df=3, min_npmi=0.0)
    assert len(edges) == 1
    assert edges.iloc[0]["n_docs"] == 5


def test_n_docs_is_not_capped_by_evidence_truncation():
    """``doc_ids`` truncates for readability; ``n_docs`` must still be exact."""
    rows = [(f"c{i}", f"D{i:03d}", e, False) for i in range(50) for e in ("E1", "E2")]
    edges = cooccurrence_edges(make_linked(rows), min_entity_df=3, min_pair_df=3, min_npmi=0.0)
    assert edges.iloc[0]["n_docs"] == 50
    assert "more" in edges.iloc[0]["doc_ids"]


def test_prepended_header_mentions_excluded():
    """v2 chunking copies a header onto every body chunk of its message."""
    rows = [(f"c{i}", f"D{i}", e, True) for i in range(5) for e in ("E1", "E2")]
    edges = cooccurrence_edges(make_linked(rows), min_entity_df=3, min_pair_df=3, min_npmi=0.0)
    assert edges.empty


def test_npmi_prefers_association_over_popularity():
    """A rare pair that always co-occurs must outrank a pair of frequent entities."""
    rows = []
    for i in range(40):  # A and B are everywhere, together only sometimes
        rows.append((f"c{i}", f"D{i}", "A", False))
        rows.append((f"c{i}", f"D{i}", "B", False))
    for i in range(40, 80):
        rows.append((f"c{i}", f"D{i}", "A", False))
    for i in range(80, 120):
        rows.append((f"c{i}", f"D{i}", "B", False))
    for i in range(120, 125):  # X and Y are rare and inseparable
        rows.append((f"c{i}", f"D{i}", "X", False))
        rows.append((f"c{i}", f"D{i}", "Y", False))

    # min_npmi below zero so A-B survives to be compared: co-occurring in 40 of
    # their 80 chunks each is *less* than their frequency predicts, so its NPMI
    # is negative — which is the point.
    edges = cooccurrence_edges(make_linked(rows), min_entity_df=3, min_pair_df=3, min_npmi=-1.0)
    ranked = edges.set_index(edges["src"] + "-" + edges["dst"])["weight"]
    assert ranked["X-Y"] == pytest.approx(1.0)
    assert ranked["A-B"] < 0 < ranked["X-Y"]


def test_excluded_entities_leave_no_edges():
    rows = [(f"c{i}", f"D{i}", e, False) for i in range(5) for e in ("E1", "E2")]
    edges = cooccurrence_edges(
        make_linked(rows), exclude={"E2"}, min_entity_df=3, min_pair_df=3, min_npmi=0.0
    )
    assert edges.empty
    assert edges.attrs["entities_excluded"] == 1


def test_noisy_entities_flags_salutations_and_runaway_spans():
    entities = make_entities([
        ("PER_1", "PERSON", "dear jeffrey", 1, 5, ""),
        ("PER_2", "PERSON", "jeffrey epstein", 1, 5, ""),
        ("ORG_1", "ORG", "arizona state university college of liberal arts and", 1, 5, ""),
        ("ORG_2", "ORG", "bank of america n a", 1, 5, ""),
        ("GPE_1", "GPE", "hialeah", 1, 5, ""),  # salutation rule is PERSON-only
    ])
    assert noisy_entities(entities) == {"PER_1", "ORG_1"}


# --------------------------------------------------------------------------
# mention linking
# --------------------------------------------------------------------------


def test_link_mentions_joins_on_norm_and_type():
    mentions = pd.DataFrame({
        "norm": ["epstein", "epstein"],
        "type": ["PERSON", "ORG"],
        "note": [None, "in_prepended_header"],
    })
    aliases = make_aliases([
        ("PER_1", "PERSON", "epstein", 5, "bare_surname"),
        ("ORG_1", "ORG", "epstein", 2, "exact"),
    ])
    linked = link_mentions(mentions, aliases)
    assert list(linked["entity_id"]) == ["PER_1", "ORG_1"]
    assert list(linked["in_header"]) == [False, True]
