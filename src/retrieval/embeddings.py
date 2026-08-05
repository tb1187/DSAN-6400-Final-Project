"""Encode text into vectors for similarity search.

Wraps a single sentence-transformers model so the index-building script and
the query-time retriever share one embedding definition and can never drift
apart. BGE models were trained with an asymmetric convention — passages are
embedded as-is, but a query is prefixed with an instruction — so passage and
query encoding are kept as separate methods rather than one generic ``encode``.
"""

from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
# Once a model is cached, nothing here needs the network — but sentence-transformers
# still pings Hugging Face to check for updates, which times out on a flaky
# connection for no benefit. Skip that check; unset HF_HUB_OFFLINE to re-enable it
# (e.g. the first time a new model name is used and needs downloading).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        # This model is small enough that CPU is plenty fast, and the GPU on
        # this box has thrown intermittent "CUDA error: unknown error" —
        # forcing CPU avoids that flakiness entirely.
        self._model = SentenceTransformer(model_name, device="cpu")

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def encode_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

    def encode_query(self, text: str) -> np.ndarray:
        vec = self._model.encode(
            QUERY_PREFIX + text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype("float32")


__all__ = ["Embedder", "DEFAULT_MODEL"]
