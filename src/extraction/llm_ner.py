"""The LLM and hybrid arms of the extraction comparison.

Two modes over identical chunks:

``extract_llm``
    Blind extraction. The model sees the chunk and returns entities.

``extract_hybrid``
    Candidate-guided extraction. The model sees the chunk *and* spaCy's proposed
    spans, and is asked to confirm, correct, reject — **and add anything the
    candidate list missed**. The addition step matters: without it a hybrid can
    never exceed spaCy's recall, since it would only ever be filtering spaCy's
    output, and the headline result would be capped by construction.

Both modes return verbatim surface strings, which are then located in the chunk
here rather than trusted from the model. Models paraphrase, normalise casing and
drop punctuation; anything that cannot be found in the source is recorded with
``note="unlocated"`` instead of being silently dropped or silently counted as an
error. That count belongs in the writeup — it is a real property of using an LLM
for span extraction.
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterable

import pandas as pd

from .schema import (
    TYPES,
    Mention,
    chunk_offsets,
    document_span,
    is_plausible,
    normalise_surface,
)

DEFAULT_MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096

SYSTEM_PROMPT = """You extract named entities from documents released by a US \
congressional oversight committee. The text is OCR output from scanned pages and \
printed emails, so it contains typos, broken words and stray characters.

Extract only these types:
- PERSON: individual people, including partial names when they refer to a person
- ORG: companies, firms, agencies, institutions, publications
- GPE: countries, states, cities, and other geopolitical places

Rules:
- Copy each entity's surface form VERBATIM from the text, including any OCR damage. \
Do not correct spelling, expand abbreviations, or change capitalisation.
- Extract every occurrence you are confident about, including repeats.
- Do not extract email addresses, URLs, dates, monetary amounts, or document \
identifiers such as Bates numbers.
- A redaction placeholder such as [REDACTED] is not an entity.
- If the text contains no entities of these types, return an empty list."""

BLIND_INSTRUCTION = """Extract all PERSON, ORG and GPE entities from this text.

<text>
{text}
</text>"""

HYBRID_INSTRUCTION = """A first-pass statistical tagger proposed the candidate \
entities below for this text. It is unreliable: it mistypes people as \
organisations, tags fragments, and misses entities entirely.

Your job:
1. CONFIRM each candidate that is correct as given.
2. CORRECT any candidate whose type is wrong or whose span is wrong.
3. REJECT any candidate that is not an entity of these types.
4. ADD any entity in the text that the candidate list missed.

<candidates>
{candidates}
</candidates>

<text>
{text}
</text>"""

TOOL = {
    "name": "record_entities",
    "description": "Record the entities found in the text.",
    # strict: the API validates the model's tool input against this schema rather
    # than us hoping it matches. Without it the model occasionally returns a list
    # of bare strings instead of objects.
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "surface": {
                            "type": "string",
                            "description": "Verbatim text of the entity, copied exactly.",
                        },
                        "type": {"type": "string", "enum": list(TYPES)},
                        "verdict": {
                            "type": "string",
                            "enum": ["confirmed", "corrected", "added"],
                            "description": (
                                "confirmed/corrected for candidates, added for entities "
                                "the candidate list missed. Use 'added' in blind mode."
                            ),
                        },
                    },
                    "required": ["surface", "type", "verdict"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["entities"],
        "additionalProperties": False,
    },
}


def _client():
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The LLM arm needs the anthropic package: pip install anthropic"
        ) from exc

    from ..utils.env import ENV_PATH, load_env

    load_env()
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        raise RuntimeError(
            f"No Anthropic credentials found. Add ANTHROPIC_API_KEY to {ENV_PATH}, "
            "export it in your shell, or run `ant auth login`."
        )
    return anthropic.Anthropic()


def _call(client, model: str, instruction: str) -> list[dict]:
    """One extraction call, returning the raw entity dicts."""
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        tools=[TOOL],
        tool_choice={"type": "tool", "name": TOOL["name"]},
        messages=[{"role": "user", "content": instruction}],
    )
    if response.stop_reason == "refusal":
        return []
    for block in response.content:
        if block.type == "tool_use":
            return block.input.get("entities", [])
    return []


def _locate(text: str, surface: str, used: set[int]) -> tuple[int, int] | None:
    """Find a returned surface form in the chunk, preferring an unclaimed span."""
    for candidate in (surface, surface.strip()):
        start = 0
        while (idx := text.find(candidate, start)) != -1:
            if idx not in used:
                return idx, idx + len(candidate)
            start = idx + 1
    # Fall back to a case-insensitive whole-word search before giving up.
    pattern = re.compile(re.escape(surface.strip()), re.IGNORECASE)
    for match in pattern.finditer(text):
        if match.start() not in used:
            return match.start(), match.end()
    return None


def _to_mentions(system: str, row: dict, entities: Iterable[dict]) -> list[Mention]:
    text = row["text"]
    offset, prefix_len = chunk_offsets(row)
    used: set[int] = set()
    mentions: list[Mention] = []

    for entity in entities:
        # Belt and braces: strict tool use should guarantee the shape, but a
        # malformed item must never abort a run that costs money and minutes.
        if isinstance(entity, str):
            entity = {"surface": entity, "type": "", "verdict": "added"}
        elif not isinstance(entity, dict):
            continue
        surface = (entity.get("surface") or "").strip()
        type_ = (entity.get("type") or "").upper()
        if not is_plausible(surface, type_):
            continue
        span = _locate(text, surface, used)
        if span is None:
            # Keep it, flagged — dropping silently would inflate precision.
            mentions.append(
                Mention(
                    system=system,
                    doc_id=row["doc_id"],
                    chunk_id=row["chunk_id"],
                    start=-1,
                    end=-1,
                    doc_start=-1,
                    doc_end=-1,
                    surface=surface,
                    type=type_,
                    note="unlocated",
                )
            )
            continue
        start, end = span
        used.add(start)
        doc_start, doc_end, placement = document_span(offset, prefix_len, start, end)
        mentions.append(
            Mention(
                system=system,
                doc_id=row["doc_id"],
                chunk_id=row["chunk_id"],
                start=start,
                end=end,
                doc_start=doc_start,
                doc_end=doc_end,
                surface=text[start:end],
                type=type_,
                note=placement or entity.get("verdict"),
            )
        )
    return mentions


def _run(system: str, client, model: str, rows: list[dict], instruction_for) -> list[Mention]:
    """Drive one arm over the chunks, surviving per-chunk failures.

    A run costs real money and several minutes; losing all of it because one
    chunk came back malformed is not an acceptable failure mode. Failures are
    counted and reported rather than raised.
    """
    mentions: list[Mention] = []
    failures: list[tuple[str, str]] = []
    for row in rows:
        try:
            entities = _call(client, model, instruction_for(row))
            mentions.extend(_to_mentions(system, row, entities))
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            failures.append((row["chunk_id"], f"{type(exc).__name__}: {exc}"))
    if failures:
        print(f"  {len(failures)} chunk(s) failed and were skipped:")
        for chunk_id, message in failures[:5]:
            print(f"    {chunk_id}: {message}")
        if len(failures) > 5:
            print(f"    ... and {len(failures) - 5} more")
    return mentions


def extract_llm(chunks: pd.DataFrame, model: str | None = None) -> list[Mention]:
    """Blind extraction: the model sees only the text."""
    return _run(
        "llm",
        _client(),
        model or DEFAULT_MODEL,
        chunks.to_dict("records"),
        lambda row: BLIND_INSTRUCTION.format(text=row["text"]),
    )


def extract_hybrid(
    chunks: pd.DataFrame, candidates: Iterable[Mention], model: str | None = None
) -> list[Mention]:
    """Candidate-guided extraction: confirm, correct, reject — and add."""
    client = _client()
    model = model or DEFAULT_MODEL

    by_chunk: dict[str, list[str]] = {}
    for mention in candidates:
        key = normalise_surface(mention.surface)
        entry = f"- {mention.surface} [{mention.type}]"
        bucket = by_chunk.setdefault(mention.chunk_id, [])
        if entry not in bucket and key:
            bucket.append(entry)

    def instruction_for(row: dict) -> str:
        proposed = by_chunk.get(row["chunk_id"], [])
        return HYBRID_INSTRUCTION.format(
            candidates="\n".join(proposed) or "(none proposed)",
            text=row["text"],
        )

    return _run("hybrid", client, model, chunks.to_dict("records"), instruction_for)


def estimate_cost(chunks: pd.DataFrame, input_rate: float, output_rate: float) -> dict:
    """Rough token and dollar estimate before spending anything."""
    import tiktoken  # noqa: F401  (only used for a rough character heuristic)

    chars = int(chunks["text"].str.len().sum())
    prompt_tokens = len(chunks) * (len(SYSTEM_PROMPT) // 4) + chars // 4
    output_tokens = len(chunks) * 300
    return {
        "chunks": len(chunks),
        "input_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "usd": round(
            prompt_tokens / 1e6 * input_rate + output_tokens / 1e6 * output_rate, 2
        ),
    }


__all__ = ["extract_llm", "extract_hybrid", "estimate_cost", "DEFAULT_MODEL"]
