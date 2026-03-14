"""
models/embeddings.py
Sentence-transformer embedding model wrapper (used by RAG).
"""
import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from config.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Loaded embedding model: %s", EMBEDDING_MODEL)
    return _model


def embed_texts(texts):
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def embed_query(query):
    return embed_texts([query])[0]