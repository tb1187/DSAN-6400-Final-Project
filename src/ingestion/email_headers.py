"""Pull email headers out of the extracted text.

The load file only tags 64 documents as email, but the printed-to-PDF emails in the
production keep their ``From:/Sent:/To:/Subject:`` block in the OCR text — several
thousand of them, including forwarded chains. Those headers are the cheapest,
highest-precision source of ``communicated with`` edges for the knowledge graph, so
they get their own parser rather than waiting on NER.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import pandas as pd

HEADER_RE = re.compile(
    r"^[ \t]*(?P<key>From|To|Cc|CC|Bcc|Sent|Date|Subject|Attachments|Importance)[ \t]*:"
    r"[ \t]*(?P<value>.*)$",
    re.MULTILINE,
)

ADDRESS_RE = re.compile(r"[\w.\-+]+@[\w\-]+\.[\w.\-]{2,}")
# Addresses survive OCR badly: rn -> m, © -> @, .corn -> .com, l/1 confusion.
OCR_ADDRESS_RE = re.compile(r"[\w.\-+]+\s*[@©®]\s*[\w\-]+\s*\.\s*[\w.\-]{2,}")

# Boilerplate that shows up where a display name should be.
NAME_STOPWORDS = {
    "", "-", "undisclosed recipients", "redacted", "privileged", "unknown",
}

FORWARD_MARKER_RE = re.compile(
    r"-{2,}\s*(?:Original Message|Forwarded message|Begin forwarded message)\s*-{0,}",
    re.IGNORECASE,
)


def normalise_address(raw: str) -> str:
    """Repair the common OCR corruptions in an email address and lower-case it."""
    a = raw.strip().lower().replace(" ", "")
    a = a.replace("©", "@").replace("®", "@")
    a = re.sub(r"\.corn\b", ".com", a)
    a = re.sub(r"\.con\b", ".com", a)
    a = a.replace("grnail", "gmail").replace("gmai1", "gmail")
    a = a.replace("hotrnail", "hotmail").replace("yahoocom", "yahoo.com")
    a = a.strip("<>[](),;:'\"")
    return a


#: ``Weingarten, Reid`` — and ``Thomas Jr., Landon``, where the generational
#: suffix sits with the surname. Without the suffix branch that second form
#: splits into two recipients, and one person becomes two nodes who appear to
#: be corresponding with each other.
LAST_FIRST_RE = re.compile(
    r"^[A-Z][\w'\-]+(?:\s+(?:Jr|Sr|II|III|IV)\.?)?,"
    r"\s*[A-Z][\w'\-.]*(?:\s+[A-Z][\w'\-.]*)?$"
)


def _split_parts(value: str) -> list[str]:
    """Split a recipient list without breaking ``Weingarten, Reid`` in half.

    Semicolons always separate recipients. Commas only do when the fragment is not
    a bare ``Last, First`` display name.
    """
    parts = []
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if LAST_FIRST_RE.match(chunk):
            parts.append(chunk)
        else:
            parts.extend(re.split(r",\s*(?![^<\[]*[>\]])", chunk))
    return parts


def split_recipients(value: str) -> list[dict]:
    """Split a To:/Cc: value into ``{name, address}`` records."""
    out = []
    for part in _split_parts(value):
        part = part.strip()
        if not part or part.lower() in NAME_STOPWORDS:
            continue
        addr = ADDRESS_RE.search(part) or OCR_ADDRESS_RE.search(part)
        address = normalise_address(addr.group(0)) if addr else None
        name = part
        if addr:
            name = part.replace(addr.group(0), "")
        # "[mailto:x@y.com]" — and the truncated "[mailto" left behind when the
        # closing bracket was lost in the scan.
        name = re.sub(r"\[?\s*mailto\s*:?.*$", "", name, flags=re.IGNORECASE)
        name = name.strip(" <>[](),;:\"'_-\t")
        name = re.sub(r"_{2,}", "", name).strip()
        if not name and not address:
            continue
        out.append({"name": name or None, "address": address})
    return out


@dataclass
class EmailMessage:
    """One header block found in a document.

    ``char_start`` / ``char_end`` are offsets into the *raw* document text, which
    is what lets an edge cite the page it came from.
    """

    doc_id: str
    block_index: int
    is_forwarded: bool = False
    sender_name: str | None = None
    sender_address: str | None = None
    recipients: list[dict] = field(default_factory=list)
    cc: list[dict] = field(default_factory=list)
    subject: str | None = None
    sent_raw: str | None = None
    attachments: str | None = None
    char_start: int = 0
    char_end: int = 0

    @property
    def sender_key(self) -> str | None:
        """Best available identifier for the sender, address preferred."""
        if self.sender_address:
            return self.sender_address
        return self.sender_name.lower() if self.sender_name else None

    def message_key(self) -> str:
        """Stable hash of the message itself, ignoring which document it sits in.

        The same message is quoted in many documents; this collapses those copies
        so an edge can be weighted by distinct messages rather than by sightings.
        """
        targets = sorted(
            filter(
                None,
                (r["address"] or (r["name"].lower() if r["name"] else None)
                 for r in self.recipients + self.cc),
            )
        )
        payload = "|".join(
            [
                self.sender_key or "",
                ",".join(targets),
                (self.subject or "").strip().lower(),
                (self.sent_raw or "").strip().lower(),
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def parse_headers(doc_id: str, text: str) -> list[EmailMessage]:
    """Find every header block in a document, including forwarded ones.

    A block ends when a ``From:`` line is seen after the block already has one, or
    when the header lines stop being contiguous enough to belong together.
    """
    matches = list(HEADER_RE.finditer(text))
    if not matches:
        return []

    forward_spans = [m.start() for m in FORWARD_MARKER_RE.finditer(text)]
    messages: list[EmailMessage] = []
    current: EmailMessage | None = None
    last_end = None

    for m in matches:
        key = m.group("key").lower()
        value = m.group("value").strip()
        # A new From:, or a gap of more than ~400 chars, starts a new block.
        gap = last_end is not None and (m.start() - last_end) > 400
        if current is None or (key == "from" and current.sender_name is not None) or (
            key == "from" and current.sender_address is not None
        ) or gap:
            if current is not None and _is_populated(current):
                messages.append(current)
            current = EmailMessage(
                doc_id=doc_id,
                block_index=len(messages),
                is_forwarded=any(s < m.start() for s in forward_spans)
                and len(messages) > 0,
                char_start=m.start(),
            )
        last_end = m.end()
        current.char_end = m.end()

        if key == "from":
            people = split_recipients(value)
            if people:
                current.sender_name = people[0]["name"]
                current.sender_address = people[0]["address"]
        elif key == "to":
            current.recipients.extend(split_recipients(value))
        elif key in {"cc", "bcc"}:
            current.cc.extend(split_recipients(value))
        elif key == "subject":
            current.subject = value or None
        elif key in {"sent", "date"}:
            current.sent_raw = value or None
        elif key == "attachments":
            current.attachments = value or None

    if current is not None and _is_populated(current):
        messages.append(current)
    return messages


def _is_populated(msg: EmailMessage) -> bool:
    return bool(
        msg.sender_name or msg.sender_address or msg.recipients or msg.subject
    )


def messages_to_frame(messages: list[EmailMessage]) -> pd.DataFrame:
    """Flatten parsed messages into a tidy table, one row per message."""
    return pd.DataFrame(
        [
            {
                "doc_id": m.doc_id,
                "block_index": m.block_index,
                "is_forwarded": m.is_forwarded,
                "sender_name": m.sender_name,
                "sender_address": m.sender_address,
                "n_recipients": len(m.recipients),
                "recipient_names": "; ".join(
                    r["name"] for r in m.recipients if r["name"]
                )
                or None,
                "recipient_addresses": "; ".join(
                    r["address"] for r in m.recipients if r["address"]
                )
                or None,
                "cc_names": "; ".join(r["name"] for r in m.cc if r["name"]) or None,
                "subject": m.subject,
                "sent_raw": m.sent_raw,
                "attachments": m.attachments,
            }
            for m in messages
        ]
    )


def edge_list(messages: list[EmailMessage]) -> pd.DataFrame:
    """Sender -> recipient pairs, one row per (message, recipient)."""
    rows = []
    for m in messages:
        sender = m.sender_address or (m.sender_name.lower() if m.sender_name else None)
        if not sender:
            continue
        for r in m.recipients + m.cc:
            target = r["address"] or (r["name"].lower() if r["name"] else None)
            if not target or target == sender:
                continue
            rows.append(
                {
                    "doc_id": m.doc_id,
                    "block_index": m.block_index,
                    "source": sender,
                    "target": target,
                    "source_name": m.sender_name,
                    "target_name": r["name"],
                    "is_cc": r in m.cc,
                    "subject": m.subject,
                    "sent_raw": m.sent_raw,
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "EmailMessage",
    "parse_headers",
    "messages_to_frame",
    "edge_list",
    "normalise_address",
    "split_recipients",
]
