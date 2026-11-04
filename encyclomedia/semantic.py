"""
Optional semantic search using sentence embeddings.

Install the extra dependency to turn it on:
    pip install sentence-transformers

The first use downloads the model (~90 MB for the default) from Hugging Face and
caches it locally; after that it runs offline. If the package is missing,
load_embedder() returns None and the catalog falls back to keyword search.
"""
from __future__ import annotations

import os
import sys

DEFAULT_MODEL = os.environ.get("ENCYCLOMEDIA_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class SentenceEmbedder:
    """Wraps a sentence-transformers model. Loads lazily on first encode()."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts):
        """Return a float32 array of unit-length vectors, one row per text."""
        import numpy as np
        vectors = self._load().encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


def available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def load_embedder(model_name: str | None = None, quiet: bool = True):
    """An embedder if sentence-transformers is installed, otherwise None."""
    if not available():
        if not quiet:
            print("Semantic search is off (pip install sentence-transformers to enable).", file=sys.stderr)
        return None
    return SentenceEmbedder(model_name or DEFAULT_MODEL)
