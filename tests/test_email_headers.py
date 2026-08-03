"""Tests for the email header parser — the deterministic tier of the graph.

These edges are meant to be the high-precision half of the knowledge graph, so
the parser's failure modes matter more than its coverage.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ingestion.email_headers import (  # noqa: E402
    edge_list,
    normalise_address,
    parse_headers,
    split_recipients,
)

FORWARDED = """From: Darren Indyke
Sent: 7/13/2016 1:37:19 PM
To: Jeffrey Epstein [jeeyacation@gmail.com]; Bennet J. Moskowitz
Subject: Fwd: the filing
Importance: High

Please see below.

----------Forwarded message----------
From: Martin Weinberg <mw@example.com>
Date: Wed, Apr 22, 2015 at 3:03 PM
To: Jeffrey Epstein
Subject: ATTORNEY-CLIENT PRIVILEGE

The attached draft is ready.
"""


# -- address and recipient handling -------------------------------------


def test_ocr_damaged_addresses_normalise_to_one_mailbox():
    variants = [
        "jeevacation@gmail.com",
        "JEEvacation@Gmail.com",
        "jeevacation@grnail.corn",
        "jeevacation © gmail.com",
    ]
    assert {normalise_address(v) for v in variants} == {"jeevacation@gmail.com"}


def test_last_comma_first_is_one_person_not_two():
    people = split_recipients("Weingarten, Reid; Starr, Ken")
    assert [p["name"] for p in people] == ["Weingarten, Reid", "Starr, Ken"]


def test_comma_still_separates_when_addresses_are_present():
    people = split_recipients("a@x.com, b@y.com")
    assert [p["address"] for p in people] == ["a@x.com", "b@y.com"]


def test_mailto_fragment_is_stripped_from_display_name():
    people = split_recipients("Jacquie Johnson [mailto:jj@example.com]")
    assert people[0]["name"] == "Jacquie Johnson"
    assert people[0]["address"] == "jj@example.com"


def test_truncated_mailto_fragment_is_also_stripped():
    """The closing bracket is often lost in the scan."""
    people = split_recipients("Jacquie Johnson [mailto")
    assert people[0]["name"] == "Jacquie Johnson"


# -- block parsing ------------------------------------------------------


def test_nested_forwarded_block_is_a_separate_message():
    messages = parse_headers("DOC_1", FORWARDED)
    assert len(messages) == 2
    assert messages[0].sender_name == "Darren Indyke"
    assert messages[1].sender_address == "mw@example.com"
    assert messages[1].is_forwarded


def test_char_offsets_point_at_the_header_block():
    messages = parse_headers("DOC_1", FORWARDED)
    for message in messages:
        span = FORWARDED[message.char_start : message.char_end]
        assert span.startswith("From:")
        assert "Subject:" in span


def test_offsets_are_ordered_and_disjoint():
    messages = parse_headers("DOC_1", FORWARDED)
    assert messages[0].char_end <= messages[1].char_start


def test_recipients_are_split_and_kept_raw():
    messages = parse_headers("DOC_1", FORWARDED)
    names = [r["name"] for r in messages[0].recipients]
    assert "Bennet J. Moskowitz" in names
    assert any(r["address"] == "jeeyacation@gmail.com" for r in messages[0].recipients)


def test_document_without_headers_yields_nothing():
    assert parse_headers("DOC_2", "A court filing with no header block.\n") == []


# -- message identity ---------------------------------------------------


def test_same_message_in_two_documents_shares_a_key():
    a = parse_headers("DOC_A", FORWARDED)[1]
    b = parse_headers("DOC_B", FORWARDED)[1]
    assert a.doc_id != b.doc_id
    assert a.message_key() == b.message_key()


def test_different_messages_get_different_keys():
    messages = parse_headers("DOC_1", FORWARDED)
    assert messages[0].message_key() != messages[1].message_key()


# -- edges --------------------------------------------------------------


def test_edges_carry_document_provenance():
    edges = edge_list(parse_headers("DOC_1", FORWARDED))
    assert set(edges["doc_id"]) == {"DOC_1"}
    assert len(edges) == 3  # 2 recipients on the outer, 1 on the forwarded


def test_self_edges_are_dropped():
    text = (
        "From: jeffrey E. <jee@example.com>\n"
        "Sent: 1/8/2018 9:25:13 PM\n"
        "To: jeffrey epstein <jee@example.com>\n"
        "Subject: Fwd: note to self\n"
    )
    assert edge_list(parse_headers("DOC_3", text)).empty


def test_address_is_preferred_over_display_name_as_the_key():
    edges = edge_list(parse_headers("DOC_1", FORWARDED))
    assert "mw@example.com" in set(edges["source"])


def test_generational_suffix_does_not_split_a_last_first_name():
    """"Thomas Jr., Landon" is one person, not two co-recipients."""
    parts = split_recipients("Thomas Jr., Landon")
    assert len(parts) == 1
    assert parts[0]["name"] == "Thomas Jr., Landon"

    # The plain form still parses, and genuine lists still split.
    assert len(split_recipients("Weingarten, Reid")) == 1
    assert len(split_recipients("Thomas Jr., Landon; Weingarten, Reid")) == 2
    assert len(split_recipients("Alice Smith, Bob Jones")) == 2
