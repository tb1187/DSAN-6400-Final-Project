"""Collapse extracted surface forms into canonical entities.

Every centrality measure in Phase 3 is arithmetic over whatever node set this
produces, so getting it wrong is not a cosmetic problem: before resolution,
``epstein`` / ``jeffrey epstein`` / ``jeff epstein`` / ``jeffrey e`` are four
nodes sharing 20,000+ mentions of one person.

Three things make this corpus harder than a textbook dedup:

* **Bare surnames dominate.** ``epstein`` alone has 15,003 mentions — more than
  any full name — and the corpus also contains Mark, Ronald and Edward Jay
  Epstein. A surname is therefore a *candidate* link, never a merge on its own.
* **OCR corrupts identity.** ``jeffery epstein``, ``grnail.corn``,
  ``jeeyacation@`` — the same entity arrives spelled several ways.
* **People and organisations behave differently.** Person names cluster well on
  string similarity; organisation names do not (``Merrill Lynch`` and ``BofA
  Merrill Lynch Global Research`` are the same firm, and no edit distance will
  tell you that). PERSON is resolved aggressively, ORG and GPE conservatively.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd
from rapidfuzz import fuzz

# Legal and corporate suffixes carry no identity — "Indyke PLLC" and "Indyke" are
# the same firm. Stripped for ORG comparison only.
ORG_SUFFIX_RE = re.compile(
    r"\b(inc|incorporated|llc|llp|lp|pllc|plc|ltd|limited|co|corp|corporation|"
    r"company|group|holdings|partners|associates|pc|pa|sa|nv|ag|gmbh)\b",
    re.IGNORECASE,
)
ORG_NOISE_RE = re.compile(r"\bthe\b", re.IGNORECASE)
WS_RE = re.compile(r"\s+")

#: Similarity above which two same-surname names are treated as OCR variants.
PERSON_FUZZ = 88
#: Organisations must match far more closely — shared words are common.
ORG_FUZZ = 94


@dataclass
class Entity:
    entity_id: str
    type: str
    canonical: str
    aliases: set[str] = field(default_factory=set)
    addresses: set[str] = field(default_factory=set)
    n_mentions: int = 0
    resolution: str = "exact"


class _Union:
    """Union-find over surface forms."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for key in self.parent:
            out[self.find(key)].append(key)
        return out


# -- person names -------------------------------------------------------


def split_name(norm: str) -> tuple[list[str], str]:
    """``jeffrey e epstein`` -> ``(['jeffrey', 'e'], 'epstein')``."""
    tokens = norm.split()
    if not tokens:
        return [], ""
    return tokens[:-1], tokens[-1]


def given_names_compatible(a: list[str], b: list[str]) -> bool:
    """Whether two given-name sequences could denote the same person.

    Compatible: identical; one is an initial of the other (``j`` / ``jeffrey``);
    one is a prefix of the other (``jeffrey`` / ``jeffrey edward``); or an OCR
    variant (``jeffery`` / ``jeffrey``). Incompatible: two different names, which
    is what keeps Mark Epstein apart from Jeffrey Epstein.
    """
    if not a or not b:
        # One side is a bare surname — handled separately, never merged here.
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        if len(x) == 1 and y.startswith(x):
            continue
        if len(y) == 1 and x.startswith(y):
            continue
        if fuzz.ratio(x, y) >= PERSON_FUZZ:
            continue
        return False
    return True


def _resolve_people(forms: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    """Cluster PERSON forms. Returns (form -> cluster root, root -> resolution kind)."""
    union = _Union()
    kinds: dict[str, str] = {}

    named = forms[forms["norm"].str.split().str.len() > 1]
    by_surname: dict[str, list[str]] = defaultdict(list)
    for norm in named["norm"]:
        by_surname[split_name(norm)[1]].append(norm)

    for surname, group in by_surname.items():
        parsed = [(n, split_name(n)[0]) for n in group]
        for i, (na, ga) in enumerate(parsed):
            union.find(na)
            for nb, gb in parsed[i + 1 :]:
                if given_names_compatible(ga, gb):
                    union.union(na, nb)
                    kinds[union.find(na)] = "name_variant"
                # Reversed order: "epstein jeffrey" vs "jeffrey epstein".
                elif set(na.split()) == set(nb.split()):
                    union.union(na, nb)
                    kinds[union.find(na)] = "token_reorder"
    return {f: union.find(f) for f in named["norm"]}, kinds


def _attach_bare_surnames(
    mentions: pd.DataFrame,
    assignment: dict[str, str],
    counts: dict[str, int],
) -> dict[str, str]:
    """Attach single-token PERSON forms to a full name, preferring same-document evidence.

    ``epstein`` is only ambiguous in the abstract. Inside a document that also
    says ``jeffrey epstein`` and never mentions another Epstein, it is not
    ambiguous at all. Documents where several bearers of the surname appear are
    left alone rather than guessed.
    """
    person = mentions[mentions["type"] == "PERSON"]

    def is_full(tokens: list[str]) -> bool:
        """A usable full name: at least two tokens, the last not an initial."""
        return len(tokens) > 1 and len(tokens[-1]) > 1

    # Forms that cannot stand on their own, and how to look them up. A surname is
    # distinctive evidence; a given name is not — "bannon" identifies a person,
    # "steve" does not. So they get different treatment, and anything messier
    # than these two shapes (a garbled multi-name string like
    # "steven pfeiffer jeffrey e") is left alone rather than guessed at.
    bare: dict[str, tuple[str, str]] = {}  # form -> (lookup key, kind)
    for norm in person["norm"].unique():
        tokens = norm.split() if norm else []
        if len(tokens) == 1 and len(tokens[0]) > 1:
            bare[norm] = (tokens[0], "surname")
        elif len(tokens) == 2 and len(tokens[1]) == 1 and len(tokens[0]) > 1:
            # "jeffrey e" — a display name whose surname never appears.
            bare[norm] = (tokens[0], "given")
    if not bare:
        return {}

    # Which full names does each document use? Indexed separately by surname and
    # given name so a fragment only ever consults the index that matches it.
    full_by_doc: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for doc_id, norm in person[["doc_id", "norm"]].drop_duplicates().itertuples(index=False):
        tokens = norm.split()
        if is_full(tokens):
            root = assignment.get(norm, norm)
            full_by_doc[(doc_id, tokens[-1], "surname")].add(root)
            full_by_doc[(doc_id, tokens[0], "given")].add(root)

    # Corpus-level fallback, surnames only. "epstein" defaulting to the most
    # mentioned Epstein is a defensible dominant-sense heuristic; "steve"
    # defaulting to the most mentioned Steve is not — given names are not
    # distinctive, so those attach on document evidence or not at all.
    dominant: dict[str, str] = {}
    for norm, root in assignment.items():
        tokens = norm.split()
        if not is_full(tokens):
            continue
        best = dominant.get(tokens[-1])
        if best is None or counts.get(root, 0) > counts.get(best, 0):
            dominant[tokens[-1]] = root

    docs_by_form = person.groupby("norm")["doc_id"].unique().to_dict()
    attached: dict[str, str] = {}
    for form, (key, kind) in bare.items():
        votes: dict[str, int] = defaultdict(int)
        for doc_id in docs_by_form.get(form, ()):
            candidates = full_by_doc.get((doc_id, key, kind), set())
            # Only unambiguous documents vote: if a document names two people
            # sharing the key, it tells us nothing about which one this is.
            if len(candidates) == 1:
                votes[next(iter(candidates))] += 1
        if votes:
            attached[form] = max(votes, key=votes.get)
        elif kind == "surname" and key in dominant:
            attached[form] = dominant[key]
    return attached


# -- organisations and places -------------------------------------------


def org_key(norm: str) -> str:
    """Comparison key for an organisation: suffixes and articles carry no identity."""
    stripped = ORG_SUFFIX_RE.sub(" ", norm)
    stripped = ORG_NOISE_RE.sub(" ", stripped)
    return WS_RE.sub(" ", stripped).strip() or norm


def _resolve_conservative(forms: pd.DataFrame, threshold: int) -> dict[str, str]:
    """Merge only near-identical forms, blocked on the first token.

    Deliberately timid. Recognising that two differently-named organisations are
    the same body is a knowledge problem, not a string problem, and guessing at
    it would silently fabricate edges.
    """
    union = _Union()
    by_block: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for norm in forms["norm"]:
        key = org_key(norm)
        if key:
            # Block on the despaced prefix, not the first token: OCR and style
            # split the same name arbitrarily ("u s" vs "us", "j p morgan" vs
            # "jp morgan"), and first-token blocking never compares them.
            by_block[key.replace(" ", "")[:4]].append((norm, key.replace(" ", "")))

    for block in by_block.values():
        for i, (na, ka) in enumerate(block):
            union.find(na)
            for nb, kb in block[i + 1 :]:
                if ka == kb or fuzz.ratio(ka, kb) >= threshold:
                    union.union(na, nb)
    return {f: union.find(f) for f in forms["norm"]}


# -- assembly -----------------------------------------------------------


def resolve(
    mentions: pd.DataFrame,
    email_actors: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve mentions into entities.

    Returns ``(entities, aliases)`` — one row per canonical entity, and one row
    per surface form pointing at its entity.
    """
    counts_by_form = mentions.groupby(["norm", "type"]).size()
    forms = counts_by_form.rename("n_mentions").reset_index()
    forms = forms[forms["norm"].str.strip() != ""]

    assignment: dict[tuple[str, str], str] = {}
    kinds: dict[tuple[str, str], str] = {}

    people = forms[forms["type"] == "PERSON"]
    person_map, person_kinds = _resolve_people(people)
    root_counts: dict[str, int] = defaultdict(int)
    for norm, root in person_map.items():
        root_counts[root] += int(counts_by_form.get((norm, "PERSON"), 0))

    bare_map = _attach_bare_surnames(mentions, person_map, root_counts)
    for norm, root in person_map.items():
        assignment[(norm, "PERSON")] = root
        kinds[(norm, "PERSON")] = person_kinds.get(root, "exact")
    for norm, root in bare_map.items():
        assignment[(norm, "PERSON")] = root
        kinds[(norm, "PERSON")] = "bare_surname"

    for type_, threshold in (("ORG", ORG_FUZZ), ("GPE", ORG_FUZZ)):
        subset = forms[forms["type"] == type_]
        for norm, root in _resolve_conservative(subset, threshold).items():
            assignment[(norm, type_)] = root
            kinds[(norm, type_)] = "exact" if norm == root else "near_duplicate"

    # Anything unassigned stands alone.
    for norm, type_ in zip(forms["norm"], forms["type"]):
        assignment.setdefault((norm, type_), norm)
        kinds.setdefault((norm, type_), "singleton")

    clusters: dict[tuple[str, str], list[str]] = defaultdict(list)
    for (norm, type_), root in assignment.items():
        clusters[(root, type_)].append(norm)

    rows, alias_rows = [], []
    for i, ((root, type_), members) in enumerate(sorted(clusters.items())):
        totals = {m: int(counts_by_form.get((m, type_), 0)) for m in members}
        # Canonical form: the most-mentioned member, preferring a full name.
        canonical = max(
            members, key=lambda m: (len(m.split()) > 1, totals[m], len(m))
        )
        entity_id = f"{type_[:3]}_{i:06d}"
        rows.append(
            {
                "entity_id": entity_id,
                "type": type_,
                "canonical": canonical,
                "n_aliases": len(members),
                "n_mentions": sum(totals.values()),
            }
        )
        for member in members:
            alias_rows.append(
                {
                    "entity_id": entity_id,
                    "type": type_,
                    "norm": member,
                    "n_mentions": totals[member],
                    "resolution": kinds.get((member, type_), "exact"),
                }
            )

    entities = pd.DataFrame(rows).sort_values("n_mentions", ascending=False)
    aliases = pd.DataFrame(alias_rows)

    if email_actors is not None and len(email_actors):
        entities, aliases = _link_email_actors(entities, aliases, email_actors)
    return entities.reset_index(drop=True), aliases.reset_index(drop=True)


#: Addresses this similar are treated as OCR variants of one mailbox.
ADDRESS_FUZZ = 85
#: How many times a (name, address) pair must be observed before it is trusted.
MIN_PAIR_SUPPORT = 2


def canonical_addresses(addresses: pd.Series) -> dict[str, str]:
    """Collapse OCR variants of the same mailbox onto one canonical spelling.

    ``jeevacation@gmail.com`` also appears as ``jeeyacation@``, ``@qmail.com``
    and ``@dmail.com``. Left alone these are four identities; the most frequently
    observed spelling wins, since OCR errors are less common than correct reads.
    """
    counts = addresses.value_counts()
    union = _Union()
    items = list(counts.index)
    by_block: dict[str, list[str]] = defaultdict(list)
    for address in items:
        local, _, domain = address.partition("@")
        # Neither end is safe as a key: OCR mangles the domain ("gmail"/"qmail")
        # *and* the leading character ("jeevacation"/"eevacation"/"ueevacation").
        # Block on the tail of the local part, which survives both.
        by_block[local[-5:]].append(address)

    for block in by_block.values():
        for i, a in enumerate(block):
            union.find(a)
            for b in block[i + 1 :]:
                if fuzz.ratio(a, b) >= ADDRESS_FUZZ:
                    union.union(a, b)

    mapping: dict[str, str] = {}
    for members in union.groups().values():
        best = max(members, key=lambda m: (counts.get(m, 0), -len(m)))
        for member in members:
            mapping[member] = best
    return mapping


def _link_email_actors(
    entities: pd.DataFrame, aliases: pd.DataFrame, actors: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach email addresses to the PERSON entities whose names they match.

    An address is far stronger evidence of identity than a name string, so where
    a deterministic email actor lines up with a resolved person, the address
    becomes an attribute of that entity — the graph gains a verified handle on a
    node that otherwise exists only as a string.

    Matching is tried in decreasing order of confidence: exact form, then
    token-order-insensitive (``Weingarten, Reid`` is ``reid weingarten``), then
    fuzzy within a shared surname (OCR damage).
    """
    person = aliases[aliases["type"] == "PERSON"]
    by_norm = dict(zip(person["norm"], person["entity_id"]))
    by_tokens: dict[tuple[str, ...], str] = {}
    by_surname: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for norm, entity_id in by_norm.items():
        tokens = norm.split()
        by_tokens.setdefault(tuple(sorted(tokens)), entity_id)
        if tokens:
            by_surname[tokens[-1]].append((norm, entity_id))

    def match(norm: str) -> tuple[str | None, str]:
        if norm in by_norm:
            return by_norm[norm], "exact"
        tokens = norm.split()
        reordered = by_tokens.get(tuple(sorted(tokens)))
        if reordered:
            return reordered, "reordered"
        for candidate, entity_id in by_surname.get(tokens[-1] if tokens else "", ()):
            if fuzz.ratio(norm, candidate) >= PERSON_FUZZ:
                return entity_id, "fuzzy"
        return None, "unmatched"

    with_address = actors[actors["address"].astype(str).str.contains("@", na=False)]
    address_map = (
        canonical_addresses(with_address["address"]) if len(with_address) else {}
    )

    # A (name, address) pair seen once can be a parser artefact — a forwarded
    # header where the wrong address sat next to the wrong name. Attaching on
    # single observations put Epstein's mailbox on Steve Bannon's node. Require
    # corroboration, and match names only by exact or reordered form: fuzzy name
    # matching plus uncorroborated pairs compounds into false identity.
    support = actors.groupby(["norm", "address"]).size() if len(actors) else {}

    linked: dict[str, set[str]] = defaultdict(set)
    stats: dict[str, int] = defaultdict(int)
    for norm, address in actors[["norm", "address"]].drop_duplicates().itertuples(index=False):
        entity_id, how = match(norm)
        stats[how] += 1
        if not (entity_id and isinstance(address, str) and "@" in address):
            continue
        if how == "fuzzy":
            stats["address_skipped_fuzzy_name"] += 1
            continue
        if support.get((norm, address), 0) < MIN_PAIR_SUPPORT:
            stats["address_skipped_uncorroborated"] += 1
            continue
        linked[entity_id].add(address_map.get(address, address))

    entities = entities.copy()
    entities["addresses"] = entities["entity_id"].map(
        lambda e: "; ".join(sorted(linked.get(e, ())))
    )
    entities.attrs["actor_match"] = dict(stats)
    entities.attrs["address_variants_merged"] = len(address_map) - len(set(address_map.values()))
    return entities, aliases


__all__ = ["resolve", "split_name", "given_names_compatible", "org_key", "Entity"]
