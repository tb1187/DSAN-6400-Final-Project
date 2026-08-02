"""The spaCy arm of the extraction comparison — and the local pipeline's tier 2.

Runs a pretrained NER model over chunks and folds its labels onto the frozen
three-type set. No custom training: the point of the comparison is what an
off-the-shelf statistical tagger gives you on this corpus, against what an LLM
gives you, at what cost.
"""

from __future__ import annotations

from typing import Iterable, Iterator

import pandas as pd

from .schema import SPACY_TYPE_MAP, Mention, is_plausible

DEFAULT_MODEL = "en_core_web_lg"
SYSTEM = "spacy"


def load_model(name: str = DEFAULT_MODEL):
    """Load a spaCy pipeline with only the components NER needs."""
    import spacy

    # The parser and lemmatiser cost time and contribute nothing here.
    return spacy.load(name, exclude=["parser", "lemmatizer", "textcat"])


def extract(
    chunks: pd.DataFrame,
    nlp=None,
    model: str = DEFAULT_MODEL,
    batch_size: int = 64,
) -> list[Mention]:
    """Extract mentions from a chunk table.

    ``chunks`` needs ``chunk_id``, ``doc_id``, ``char_start`` and ``text``.
    """
    nlp = nlp or load_model(model)
    records = chunks.to_dict("records")
    texts = (row["text"] for row in records)

    mentions: list[Mention] = []
    for row, doc in zip(records, nlp.pipe(texts, batch_size=batch_size)):
        mentions.extend(_mentions_from_doc(row, doc))
    return mentions


def _mentions_from_doc(row: dict, doc) -> Iterator[Mention]:
    offset = int(row["char_start"])
    for ent in doc.ents:
        mapped = SPACY_TYPE_MAP.get(ent.label_)
        if mapped is None or not is_plausible(ent.text, mapped):
            continue
        yield Mention(
            system=SYSTEM,
            doc_id=row["doc_id"],
            chunk_id=row["chunk_id"],
            start=ent.start_char,
            end=ent.end_char,
            doc_start=offset + ent.start_char,
            doc_end=offset + ent.end_char,
            surface=ent.text,
            type=mapped,
            note=ent.label_ if ent.label_ != mapped else None,
        )


def candidates_for_chunk(mentions: Iterable[Mention], chunk_id: str) -> list[dict]:
    """The candidate list handed to the LLM in the hybrid arm."""
    return [
        {"surface": m.surface, "type": m.type, "start": m.start}
        for m in mentions
        if m.chunk_id == chunk_id
    ]


__all__ = ["extract", "load_model", "candidates_for_chunk", "DEFAULT_MODEL", "SYSTEM"]
