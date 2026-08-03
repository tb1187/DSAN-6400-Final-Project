"""Build graph edges over resolved entities.

Two layers so far, in increasing order of how much they can be trusted:

* **Tier 1 — ``communicated_with``.** Sender→recipient edges from headers parsed
  deterministically out of the OCR text. Highest precision in the graph: the
  relation is stated by the document, not inferred.
* **Tier 2 — ``co_occurs_with``.** Undirected association from entities being
  mentioned in the same chunk, weighted by NPMI. Cheap and broad, but it says
  only "these appear together", never *how* they are related.

Both carry evidence (``doc_ids``, page citations) so a claim in the GraphRAG arm
can be traced back to the page it came from.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import log

import pandas as pd
from dateutil.parser import UnknownTimezoneWarning
from dateutil.parser import parse as dateutil_parse

from src.extraction.schema import SALUTATION_RE, normalise_surface
from src.knowledge_graph.resolution import _Union, canonical_addresses

#: Mentions that landed inside a chunk's prepended header copy.
HEADER_NOTE = "in_prepended_header"

#: How many pieces of evidence to keep on an edge before truncating.
EVIDENCE_CAP = 20

WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------
# mention → entity
# --------------------------------------------------------------------------


def link_mentions(mentions: pd.DataFrame, aliases: pd.DataFrame) -> pd.DataFrame:
    """Attach ``entity_id`` to every mention, via its ``(norm, type)`` alias."""
    linked = mentions.merge(
        aliases[["norm", "type", "entity_id"]], on=["norm", "type"], how="left"
    )
    linked["in_header"] = linked["note"].eq(HEADER_NOTE)
    return linked


# --------------------------------------------------------------------------
# tier 1 — communication
# --------------------------------------------------------------------------


def _name_index(aliases: pd.DataFrame) -> tuple[dict, dict]:
    """Look-ups from a person's surface form to their entity."""
    person = aliases[aliases["type"] == "PERSON"]
    by_norm = dict(zip(person["norm"], person["entity_id"]))
    by_tokens: dict[tuple[str, ...], str] = {}
    for norm, entity_id in by_norm.items():
        by_tokens.setdefault(tuple(sorted(norm.split())), entity_id)
    return by_norm, by_tokens


class ActorResolver:
    """Map an email actor key onto the entity it refers to.

    Actor keys are whatever the header carried — sometimes an address, sometimes
    a display name. Tried in decreasing order of confidence: the address (a
    verified handle), then the key read as a name, then the display names seen
    alongside that key. Names are matched only on exact or reordered form, never
    fuzzily: this tier's value is its precision, and a wrong link here fabricates
    a conversation that never happened.

    Actors that resolve to nobody are *minted* as their own nodes rather than
    dropped. NER failing to find a name is not evidence that the correspondence
    did not occur, and dropping those edges would silently thin the network.
    """

    def __init__(self, entities: pd.DataFrame, aliases: pd.DataFrame, addresses: pd.Series):
        self.by_norm, self.by_tokens = _name_index(aliases)

        entity_addresses = [
            address
            for blob in entities["addresses"].fillna("")
            for address in str(blob).split("; ")
            if "@" in address
        ]
        # One map over both pools, so the entity side and the actor side land on
        # the same canonical spelling even where the pools differ.
        pool = pd.concat([addresses, pd.Series(entity_addresses, dtype=str)])
        self.address_map = canonical_addresses(pool) if len(pool) else {}

        # An address can be claimed by more than one entity, because resolution
        # attaches it to every name that matched — including junk forms like
        # "darren indyke jeffrey epstein" (2 mentions), a header string spaCy read
        # as one person. Give the address to its dominant claimant, or the real
        # Epstein node loses his own mailbox to a fragment.
        claims: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for entity_id, blob, n in zip(
            entities["entity_id"], entities["addresses"].fillna(""), entities["n_mentions"]
        ):
            for address in str(blob).split("; "):
                if "@" in address:
                    claims[self.canonical(address)].append((int(n), entity_id))
        self.address_to_entity = {a: max(c)[1] for a, c in claims.items()}
        self.contested = sum(1 for c in claims.values() if len(c) > 1)

        self.minted: dict[str, dict] = {}
        self.stats: Counter = Counter()

    def canonical(self, address: str) -> str:
        return self.address_map.get(address, address)

    def _match_name(self, norm: str) -> tuple[str | None, str]:
        if not norm:
            return None, "unmatched"
        # Single-token keys ("weingarten") are allowed on an *exact* alias hit
        # only. Resolution has already decided, with document-scoped evidence,
        # whether a bare surname belongs to a full name or stands alone; either
        # answer is better than minting a second node for the same surname.
        if norm in self.by_norm:
            return self.by_norm[norm], "name"
        if len(norm.split()) < 2:
            return None, "unmatched"
        reordered = self.by_tokens.get(tuple(sorted(norm.split())))
        if reordered:
            return reordered, "name_reordered"
        return None, "unmatched"

    def resolve(self, key: str, display_names: Counter) -> tuple[str, str]:
        if "@" in key:
            entity_id = self.address_to_entity.get(self.canonical(key))
            if entity_id:
                self.stats["address"] += 1
                return entity_id, "address"

        entity_id, how = self._match_name(normalise_surface(key))
        if entity_id:
            self.stats[how] += 1
            return entity_id, how

        for name, _ in display_names.most_common(3):
            entity_id, how = self._match_name(normalise_surface(str(name)))
            if entity_id:
                self.stats["display_name"] += 1
                return entity_id, "display_name"

        self.stats["minted"] += 1
        label = display_names.most_common(1)[0][0] if display_names else key
        node_id = f"EML_{len(self.minted):06d}"
        self.minted[key] = {
            "entity_id": node_id,
            "type": "PERSON",
            "canonical": normalise_surface(str(label)) or key,
            "n_aliases": 1,
            "n_mentions": 0,
            "addresses": self.canonical(key) if "@" in key else "",
            "source": "email_actor",
        }
        return node_id, "minted"


@dataclass
class ActorMap:
    """Email actors resolved onto entities, shared by every tier-1 builder.

    ``rows`` is the edge table after routing blocks are dropped, with ``src`` and
    ``dst`` entity ids and a parsed ``sent_at`` already attached.
    """

    rows: pd.DataFrame
    minted: pd.DataFrame
    stats: dict
    n_actors: int
    contested: int
    routing_dropped: int
    dates_unparsed: int
    dates_out_of_range: int


def resolve_actors(
    email_edges: pd.DataFrame,
    email_messages: pd.DataFrame,
    entities: pd.DataFrame,
    aliases: pd.DataFrame,
) -> ActorMap:
    """Resolve every email actor once, for all tier-1 relations to share."""
    email_edges, routing = _drop_routing_headers(email_edges)

    addresses = pd.concat(
        [
            email_edges["source"],
            email_edges["target"],
            email_messages["sender_address"].dropna(),
        ]
    ).astype(str)
    addresses = addresses[addresses.str.contains("@")]
    resolver = ActorResolver(entities, aliases, addresses)

    # Display names observed for each actor key, most common first.
    display: dict[str, Counter] = defaultdict(Counter)
    for key_col, name_col in (("source", "source_name"), ("target", "target_name")):
        pairs = email_edges[[key_col, name_col]].dropna()
        for key, name in pairs.itertuples(index=False):
            display[str(key)][str(name)] += 1

    keys = pd.unique(pd.concat([email_edges["source"], email_edges["target"]]).astype(str))
    resolved = {key: resolver.resolve(key, display[key]) for key in sorted(keys)}

    rows = email_edges.copy()
    rows["src"] = rows["source"].astype(str).map(lambda k: resolved[k][0])
    rows["dst"] = rows["target"].astype(str).map(lambda k: resolved[k][0])
    rows["sent_at"], unparsed, out_of_range = parse_sent_column(rows["sent_raw"])

    return ActorMap(
        rows=rows,
        minted=pd.DataFrame(list(resolver.minted.values())),
        stats=dict(resolver.stats),
        n_actors=len(keys),
        contested=resolver.contested,
        routing_dropped=routing,
        dates_unparsed=unparsed,
        dates_out_of_range=out_of_range,
    )


def communication_edges(actors: ActorMap) -> pd.DataFrame:
    """Directed sender→recipient edges between entities, with a date span."""
    rows = actors.rows
    # An actor mailing an address that resolves to the same person is an artefact
    # of one person holding several mailboxes, not a relationship.
    self_edges = int((rows["src"] == rows["dst"]).sum())
    rows = rows[rows["src"] != rows["dst"]]

    grouped = rows.groupby(["src", "dst"], sort=False)
    edges = grouped.agg(
        weight=("message_key", "nunique"),
        n_rows=("message_key", "size"),
        n_docs=("doc_id", "nunique"),
        n_cc=("is_cc", "sum"),
        first_sent=("sent_at", "min"),
        last_sent=("sent_at", "max"),
    ).reset_index()
    edges["doc_ids"] = grouped["doc_id"].agg(_evidence).values
    edges["page_bates"] = grouped["page_bates"].agg(_evidence).values
    edges["subjects"] = grouped["subject"].agg(lambda s: _evidence(s, cap=3)).values
    edges["span_days"] = (edges["last_sent"] - edges["first_sent"]).dt.days
    edges["relation"] = "communicated_with"
    edges["tier"] = 1
    edges = edges.sort_values("weight", ascending=False).reset_index(drop=True)

    edges.attrs["actor_resolution"] = actors.stats
    edges.attrs["n_actors"] = actors.n_actors
    edges.attrs["contested_addresses"] = actors.contested
    edges.attrs["routing_rows_dropped"] = actors.routing_dropped
    edges.attrs["self_edges_dropped"] = self_edges
    edges.attrs["dates_unparsed"] = actors.dates_unparsed
    edges.attrs["dates_out_of_range"] = actors.dates_out_of_range
    return edges


# --------------------------------------------------------------------------
# tier 1 — co-recipients
# --------------------------------------------------------------------------


def co_recipient_edges(actors: ActorMap) -> pd.DataFrame:
    """Undirected edges between people addressed on the same message.

    Distinct from ``communicated_with``: two people on one To: line have a tie
    even if neither ever wrote to the other. Both were chosen, by someone, as
    the audience for the same thing.

    Recipient rows repeat once per document a message survives in — a forwarded
    copy in seven documents lists its three recipients seven times — so rows are
    deduplicated on ``(message_key, dst)`` before pairing.
    """
    # Deduplicate for *counting* but keep every document a message survives in
    # as evidence — a forwarded copy is one message found in several places.
    docs_by_message = actors.rows.groupby("message_key")["doc_id"].agg(set)
    rows = actors.rows.drop_duplicates(subset=["message_key", "dst"])

    pair_messages: dict[tuple[str, str], set] = defaultdict(set)
    pair_docs: dict[tuple[str, str], set] = defaultdict(set)
    pair_cc: Counter = Counter()
    subjects: dict[tuple[str, str], Counter] = defaultdict(Counter)
    dates: dict[tuple[str, str], list] = defaultdict(list)

    for message_key, group in rows.groupby("message_key", sort=False):
        recipients = sorted(set(group["dst"]))
        if len(recipients) < 2:
            continue
        cc_flags = dict(zip(group["dst"], group["is_cc"]))
        for pair in combinations(recipients, 2):
            pair_messages[pair].add(message_key)
            pair_docs[pair].update(docs_by_message.get(message_key, ()))
            if all(bool(cc_flags.get(r)) for r in pair):
                pair_cc[pair] += 1
            for subject in group["subject"].dropna():
                subjects[pair][str(subject)] += 1
            dates[pair].extend(group["sent_at"].dropna())

    rows_out = []
    for pair, messages in pair_messages.items():
        when = pd.Series(dates[pair], dtype="datetime64[ns]")
        rows_out.append(
            {
                "src": pair[0],
                "dst": pair[1],
                "weight": len(messages),
                "n_docs": len(pair_docs[pair]),
                "n_cc": pair_cc[pair],
                "doc_ids": _truncate(sorted(str(d) for d in pair_docs[pair] if d)),
                "subjects": _truncate([s for s, _ in subjects[pair].most_common(3)], cap=3),
                "first_sent": when.min() if len(when) else pd.NaT,
                "last_sent": when.max() if len(when) else pd.NaT,
                "relation": "co_recipient_of",
                "tier": 1,
            }
        )

    columns = ["src", "dst", "weight", "n_docs", "n_cc", "doc_ids", "subjects",
               "first_sent", "last_sent", "relation", "tier"]
    edges = pd.DataFrame(rows_out, columns=columns)
    return edges.sort_values("weight", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# tier 1 — threads
# --------------------------------------------------------------------------

#: Leading "Re:"/"Fwd:" chains, however many deep.
SUBJECT_PREFIX_RE = re.compile(r"^\s*(?:(?:re|fw|fwd|fyi)\s*:\s*)+", re.IGNORECASE)

#: Subjects that identify nothing and would merge unrelated conversations.
EMPTY_SUBJECTS = {"", "<no subject>", "no subject", "none", "(no subject)"}

#: A thread needs at least this many messages to be a conversation.
MIN_THREAD_MESSAGES = 2

#: Two messages under one subject join the same thread only if they are this
#: close in time. Without it, Epstein — who is in almost every message — chains
#: six years of unrelated mail under "trump" into a single 61-message thread.
THREAD_WINDOW_DAYS = 30


def normalise_subject(subject) -> str:
    if not isinstance(subject, str):
        return ""
    stripped = SUBJECT_PREFIX_RE.sub("", subject).strip().lower()
    stripped = WS_RE.sub(" ", stripped)
    return "" if stripped in EMPTY_SUBJECTS or len(stripped) < 4 else stripped


def thread_edges(actors: ActorMap) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group messages into threads and link each participant to its thread.

    A shared subject alone is too weak a key — "trump" heads 63 messages and
    "jane doe" 91, spanning conversations with nothing to do with each other.
    Messages are therefore clustered within a subject only when they also share
    a participant, which splits a reused subject line back into the separate
    exchanges it actually names.

    Threads become their own nodes rather than a clique over participants: a
    thread with 12 people would otherwise contribute 66 edges and inflate every
    one of their centralities. Project to co-participation if that is wanted —
    the bipartite form keeps the choice open, and carries evidence.

    Returns ``(edges, thread_nodes)``.
    """
    rows = actors.rows.copy()
    rows["subject_norm"] = rows["subject"].map(normalise_subject)
    rows = rows[rows["subject_norm"] != ""]

    threads: list[dict] = []
    for subject, group in rows.groupby("subject_norm", sort=True):
        by_message = group.groupby("message_key")
        participants = {
            key: set(part["src"]) | set(part["dst"]) for key, part in by_message
        }
        sent = {key: part["sent_at"].min() for key, part in by_message}

        # Two messages belong together when they share a participant *and* sit
        # close in time. An undated message joins on participants alone rather
        # than being stranded.
        union = _Union()
        seen: dict[str, list[str]] = defaultdict(list)
        window = pd.Timedelta(days=THREAD_WINDOW_DAYS)
        for message_key, people in participants.items():
            union.find(message_key)
            for person in people:
                for other in seen[person]:
                    a, b = sent.get(message_key), sent.get(other)
                    if pd.isna(a) or pd.isna(b) or abs(a - b) <= window:
                        union.union(message_key, other)
                seen[person].append(message_key)

        for members in union.groups().values():
            if len(members) < MIN_THREAD_MESSAGES:
                continue
            part = group[group["message_key"].isin(members)]
            when = part["sent_at"].dropna()
            threads.append(
                {
                    "subject": subject,
                    "message_keys": set(members),
                    "participants": Counter(
                        list(part["src"]) + list(part["dst"])
                    ),
                    "doc_ids": sorted(set(str(d) for d in part["doc_id"] if d)),
                    "pages": sorted(set(str(p) for p in part["page_bates"].dropna())),
                    "first_sent": when.min() if len(when) else pd.NaT,
                    "last_sent": when.max() if len(when) else pd.NaT,
                }
            )

    threads.sort(key=lambda t: -len(t["message_keys"]))
    nodes, edges = [], []
    for i, thread in enumerate(threads):
        thread_id = f"THR_{i:04d}"
        nodes.append(
            {
                "entity_id": thread_id,
                "type": "THREAD",
                "canonical": thread["subject"],
                "n_aliases": 1,
                "n_mentions": len(thread["message_keys"]),
                "addresses": "",
                "source": "email_thread",
                "n_participants": len(thread["participants"]),
                "first_sent": thread["first_sent"],
                "last_sent": thread["last_sent"],
                "doc_ids": _truncate(thread["doc_ids"]),
            }
        )
        for participant, count in thread["participants"].items():
            edges.append(
                {
                    "src": participant,
                    "dst": thread_id,
                    "weight": count,
                    "n_docs": len(thread["doc_ids"]),
                    "doc_ids": _truncate(thread["doc_ids"]),
                    "page_bates": _truncate(thread["pages"]),
                    "subjects": thread["subject"],
                    "first_sent": thread["first_sent"],
                    "last_sent": thread["last_sent"],
                    "relation": "participated_in",
                    "tier": 1,
                }
            )

    columns = ["src", "dst", "weight", "n_docs", "doc_ids", "page_bates", "subjects",
               "first_sent", "last_sent", "relation", "tier"]
    return pd.DataFrame(edges, columns=columns), pd.DataFrame(nodes)


# --------------------------------------------------------------------------
# when a message was sent
# --------------------------------------------------------------------------

#: The corpus runs 2006–2019. Anything outside this is a misparse — usually
#: dateutil filling the *current* year in for a date it could not read.
CORPUS_YEARS = (1990, 2021)

#: A four-digit year has to be present. Without one, dateutil invents today's.
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

#: "(GMT+02:00)" marks a client that also writes the day first: 15/10/2014.
DAYFIRST_RE = re.compile(r"\(GMT[+\-]", re.IGNORECASE)


def parse_sent(raw) -> pd.Timestamp | None:
    """Parse a header's ``Sent:`` value, or return None.

    Formats are mixed — ``12/9/2016 3:46:19 PM`` beside
    ``Tuesday, November 22, 2016 2:15 PM`` beside ``15/10/2014 (GMT+02:00)``.
    dateutil swaps to day-first on its own when the month exceeds 12, so only
    genuinely ambiguous dates (both parts ≤ 12) can be misread; the ``(GMT…)``
    suffix is a reliable tell for the European clients that produce them.
    Day-level error is tolerable for the year/month trends this feeds.
    """
    if not isinstance(raw, str) or not YEAR_RE.search(raw):
        return None
    try:
        parsed = dateutil_parse(
            raw, fuzzy=True, dayfirst=bool(DAYFIRST_RE.search(raw))
        )
    except (ValueError, OverflowError, TypeError):
        return None
    return pd.Timestamp(parsed.replace(tzinfo=None))


def parse_sent_column(raw: pd.Series) -> tuple[pd.Series, int, int]:
    """Parse a column of ``Sent:`` values. Returns ``(dates, unparsed, dropped)``."""
    with warnings.catch_warnings():
        # "PST identified but not understood" — the offset is dropped either way,
        # and these are local timestamps to within a day regardless.
        warnings.simplefilter("ignore", UnknownTimezoneWarning)
        parsed = pd.Series(
            [parse_sent(value) for value in raw], index=raw.index, dtype="datetime64[ns]"
        )
    unparsed = int(parsed.isna().sum())
    low, high = CORPUS_YEARS
    in_range = parsed.dt.year.between(low, high)
    out_of_range = int((parsed.notna() & ~in_range).sum())
    return parsed.where(in_range), unparsed, out_of_range


def _drop_routing_headers(email_edges: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Discard FBI teletype routing blocks the header parser read as email.

    ``FROM: MIAMI / TO: DIRECTOR / ATTN: SSA`` has the shape of an email header
    and parses like one, but its "actors" are field offices and desks. Left in,
    ``miami`` becomes one of the highest-betweenness nodes in the network.

    The tell is that a routing block carries no address, no subject and no
    timestamp — real correspondence has at least one. That costs 18 of 4,088 rows
    across 6 documents, and every actor it removes is a routing artefact.
    """
    def present(column: str) -> pd.Series:
        return email_edges[column].notna() & email_edges[column].astype(str).str.strip().ne("")

    has_address = email_edges["source"].astype(str).str.contains("@") | email_edges[
        "target"
    ].astype(str).str.contains("@")
    keep = has_address | present("subject") | present("sent_raw")
    return email_edges[keep], int((~keep).sum())


def _evidence(series: pd.Series, cap: int = EVIDENCE_CAP) -> str:
    """Distinct values as a ``; ``-joined string, truncated with a count."""
    return _truncate([str(v) for v in pd.unique(series.dropna()) if str(v).strip()], cap)


def _truncate(values: list[str], cap: int = EVIDENCE_CAP) -> str:
    if len(values) > cap:
        return "; ".join(values[:cap]) + f"; +{len(values) - cap} more"
    return "; ".join(values)


# --------------------------------------------------------------------------
# tier 2 — co-occurrence
# --------------------------------------------------------------------------

#: An entity must appear in at least this many chunks to be worth relating.
MIN_ENTITY_DF = 3
#: A pair must co-occur in at least this many chunks.
MIN_PAIR_DF = 3
#: ...in at least this many *documents*. A pair confined to one document is
#: usually a signature block or letterhead repeating, not a relationship.
MIN_PAIR_DOCS = 2
#: Normalised PMI below this is association no stronger than chance.
MIN_NPMI = 0.2
#: Chunks naming more entities than this are lists and directories, where
#: adjacency carries no relational meaning.
MAX_ENTITIES_PER_CHUNK = 60
#: Beyond this, a "name" is a span spaCy ran across a line break — e.g.
#: "arizona state university college of liberal arts and".
MAX_ENTITY_TOKENS = 6


def noisy_entities(entities: pd.DataFrame) -> set[str]:
    """Entities that are extraction artefacts rather than actors.

    Excluded from tier 2 but left in the node table with a flag, so the choice
    is visible and reversible rather than silently applied.
    """
    canonical = entities["canonical"].fillna("")
    too_long = canonical.str.split().str.len() > MAX_ENTITY_TOKENS
    salutation = canonical.str.match(SALUTATION_RE) & entities["type"].eq("PERSON")
    return set(entities.loc[too_long | salutation, "entity_id"])


def cooccurrence_edges(
    linked: pd.DataFrame,
    exclude: set[str] | None = None,
    min_entity_df: int = MIN_ENTITY_DF,
    min_pair_df: int = MIN_PAIR_DF,
    min_pair_docs: int = MIN_PAIR_DOCS,
    min_npmi: float = MIN_NPMI,
) -> pd.DataFrame:
    """Undirected co-mention edges, weighted by normalised PMI.

    Raw co-occurrence counts rank by *popularity* — every frequent entity looks
    related to every other. NPMI asks instead whether a pair appears together
    more than their individual rates predict, which is what "related" means here.

    Mentions inside a chunk's prepended header are excluded: v2 chunking copies a
    message's header onto every one of its body chunks, so counting them would
    multiply a single header's co-occurrences by the message's chunk count. That
    layer is covered exactly by tier 1 anyway.
    """
    usable = linked[linked["entity_id"].notna() & ~linked["in_header"]]
    if exclude:
        usable = usable[~usable["entity_id"].isin(exclude)]

    per_chunk = (
        usable.groupby("chunk_id")["entity_id"].agg(lambda s: sorted(set(s))).to_dict()
    )
    oversized = {c: e for c, e in per_chunk.items() if len(e) > MAX_ENTITIES_PER_CHUNK}
    per_chunk = {c: e for c, e in per_chunk.items() if len(e) <= MAX_ENTITIES_PER_CHUNK}

    entity_df: Counter = Counter()
    for entities_here in per_chunk.values():
        entity_df.update(entities_here)
    frequent = {e for e, n in entity_df.items() if n >= min_entity_df}

    pair_df: Counter = Counter()
    pair_docs: dict[tuple[str, str], set] = defaultdict(set)
    chunk_docs = dict(zip(linked["chunk_id"], linked["doc_id"]))
    for chunk_id, entities_here in per_chunk.items():
        kept = [e for e in entities_here if e in frequent]
        doc_id = chunk_docs.get(chunk_id)
        for pair in combinations(kept, 2):
            pair_df[pair] += 1
            pair_docs[pair].add(doc_id)

    n_chunks = len(per_chunk)
    rows, dropped_single_doc = [], 0
    for (a, b), df in pair_df.items():
        if df < min_pair_df:
            continue
        docs = pair_docs[(a, b)]
        if len(docs) < min_pair_docs:
            dropped_single_doc += 1
            continue
        p_ab = df / n_chunks
        if p_ab >= 1.0:
            npmi = 1.0  # in every chunk together: perfect association by definition
        else:
            pmi = log(p_ab / ((entity_df[a] / n_chunks) * (entity_df[b] / n_chunks)))
            npmi = pmi / -log(p_ab)
        if npmi < min_npmi:
            continue
        evidence = sorted(str(d) for d in docs if d)
        rows.append(
            {
                "src": a,
                "dst": b,
                "weight": round(npmi, 4),
                "n_chunks": df,
                # Counted over every document, not just the ones kept as evidence.
                "n_docs": len(docs),
                "doc_ids": _truncate(evidence),
                "relation": "co_occurs_with",
                "tier": 2,
            }
        )

    columns = ["src", "dst", "weight", "n_chunks", "n_docs", "doc_ids", "relation", "tier"]
    edges = pd.DataFrame(rows, columns=columns)
    edges = edges.sort_values("weight", ascending=False).reset_index(drop=True)
    edges.attrs["candidate_pairs"] = len(pair_df)
    edges.attrs["chunks_used"] = n_chunks
    edges.attrs["chunks_oversized"] = len(oversized)
    edges.attrs["entities_frequent"] = len(frequent)
    edges.attrs["entities_excluded"] = len(exclude or ())
    edges.attrs["dropped_single_doc"] = dropped_single_doc
    return edges


__all__ = [
    "link_mentions",
    "resolve_actors",
    "communication_edges",
    "co_recipient_edges",
    "thread_edges",
    "cooccurrence_edges",
    "noisy_entities",
    "parse_sent",
]
