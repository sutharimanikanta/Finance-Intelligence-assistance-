"""
models/embeddings.py
Sentence-transformer embedding model wrapper (used by RAG).
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from config.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Return a singleton SentenceTransformer model."""
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model '%s' loaded.", EMBEDDING_MODEL)
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            raise
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings.

    Returns:
        Float32 numpy array of shape (N, dim).
    """
    try:
        model = get_embedding_model()
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    except Exception as exc:
        logger.error("Embedding failed: %s", exc)
        raise


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    return embed_texts([query])[0]
