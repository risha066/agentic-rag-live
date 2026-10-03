"""Local embedding function (no extra API key required)."""
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from config import Config


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(Config.EMBEDDING_MODEL)


class LocalEmbeddingFunction:
    """Chroma-compatible embedding function backed by sentence-transformers."""

    def __call__(self, input: List[str]) -> List[List[float]]:
        vectors = _model().encode(list(input), show_progress_bar=False)
        return vectors.tolist()

    # Chroma >=0.5 sometimes probes this for cache-key stability.
    def name(self) -> str:
        return f"local:{Config.EMBEDDING_MODEL}"


def embed_texts(texts: List[str]) -> List[List[float]]:
    return LocalEmbeddingFunction()(texts)
