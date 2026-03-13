"""
utils/rag_engine.py
─────────────────────────────────────────────────────────────────────────────
RAG engine with:
  • Multiple vector indexes  (one per uploaded document / collection)
  • Cache-aside pattern      (in-memory TTL cache keyed on query hash)
  • Accepts PDF and DOCX     (via PyPDF2 / python-docx)
  • FAISS for similarity search (lightweight, no server required)
"""

import hashlib
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config.config import (
    CACHE_TTL_SECONDS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_CACHE_ENTRIES,
    RAG_TOP_K,
    VECTOR_STORE_DIR,
)
from models.embeddings import embed_query, embed_texts
from models.llm import chat_complete

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# TEXT EXTRACTORS
# ─────────────────────────────────────────────

def _extract_pdf(path: str) -> str:
    try:
        import PyPDF2
        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text.append(t)
        return "\n".join(text)
    except Exception as exc:
        logger.error("PDF extract error (%s): %s", path, exc)
        return ""

def _extract_docx(path: str) -> str:
    try:
        import docx
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as exc:
        logger.error("DOCX extract error (%s): %s", path, exc)
        return ""

def extract_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext in (".docx", ".doc"):
        return _extract_docx(path)
    # plain text fallback
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logger.error("Text extract error (%s): %s", path, exc)
        return ""

# ─────────────────────────────────────────────
# CHUNKER
# ─────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words  = text.split()
    chunks = []
    i      = 0
    while i < len(words):
        chunk = words[i : i + size]
        chunks.append(" ".join(chunk))
        i += size - overlap
    return [c for c in chunks if c.strip()]

# ─────────────────────────────────────────────
# IN-MEMORY FAISS-LIKE INDEX (no install needed)
# ─────────────────────────────────────────────

class SimpleVectorIndex:
    """Brute-force cosine-similarity index (no FAISS required)."""

    def __init__(self):
        self.vectors: Optional[np.ndarray] = None
        self.chunks:  list[str]            = []

    def add(self, chunks: list[str]):
        embeddings = embed_texts(chunks)
        self.chunks  = chunks
        self.vectors = embeddings

    def search(self, query: str, top_k: int = RAG_TOP_K) -> list[str]:
        if self.vectors is None or len(self.chunks) == 0:
            return []
        qvec = embed_query(query)
        # Cosine similarity
        norms   = np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10
        qnorm   = np.linalg.norm(qvec) + 1e-10
        sims    = (self.vectors / norms) @ (qvec / qnorm)
        indices = np.argsort(sims)[::-1][:top_k]
        return [self.chunks[i] for i in indices]

    def save(self, path: str):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({"vectors": self.vectors, "chunks": self.chunks}, f)
        except Exception as exc:
            logger.error("Index save error: %s", exc)

    def load(self, path: str) -> bool:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.vectors = data["vectors"]
            self.chunks  = data["chunks"]
            return True
        except Exception:
            return False

# ─────────────────────────────────────────────
# CACHE-ASIDE
# ─────────────────────────────────────────────

class QueryCache:
    """Simple TTL in-memory cache."""

    def __init__(self, ttl: int = CACHE_TTL_SECONDS, max_entries: int = MAX_CACHE_ENTRIES):
        self._store:       dict[str, tuple[str, float]] = {}
        self.ttl           = ttl
        self.max_entries   = max_entries

    def _key(self, query: str, index_name: str) -> str:
        return hashlib.md5(f"{index_name}::{query}".encode()).hexdigest()

    def get(self, query: str, index_name: str) -> Optional[str]:
        k = self._key(query, index_name)
        entry = self._store.get(k)
        if entry and time.time() - entry[1] < self.ttl:
            logger.debug("Cache HIT for '%s'", query[:40])
            return entry[0]
        return None

    def set(self, query: str, index_name: str, answer: str):
        if len(self._store) >= self.max_entries:
            # Evict oldest
            oldest = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest]
        k = self._key(query, index_name)
        self._store[k] = (answer, time.time())

# ─────────────────────────────────────────────
# RAG ENGINE
# ─────────────────────────────────────────────

_ANSWER_SYSTEM = """You are a financial research assistant.
Use ONLY the provided context excerpts to answer the question.
If the context does not contain enough information, say so clearly.
Be concise and factual.
"""


class RAGEngine:
    """
    Manages multiple named vector indexes + cache-aside retrieval.

    Usage:
        rag = RAGEngine()
        rag.ingest("my_doc", "/path/to/report.pdf")
        answer = rag.query("What is the revenue outlook?")
    """

    def __init__(self):
        self._indexes: dict[str, SimpleVectorIndex] = {}
        self._cache   = QueryCache()
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    # ── document ingestion ───────────────────────────────────────────────────

    def ingest(self, name: str, file_path: str) -> int:
        """
        Extract text, chunk, embed, and store under `name`.
        Returns number of chunks created.
        """
        try:
            text   = extract_text(file_path)
            if not text.strip():
                logger.warning("No text extracted from %s", file_path)
                return 0
            chunks = chunk_text(text)
            index  = SimpleVectorIndex()
            index.add(chunks)
            self._indexes[name] = index

            # Persist to disk
            store_path = os.path.join(VECTOR_STORE_DIR, f"{name}.pkl")
            index.save(store_path)
            logger.info("Ingested '%s': %d chunks.", name, len(chunks))
            return len(chunks)
        except Exception as exc:
            logger.error("Ingest error (%s): %s", name, exc)
            return 0

    def load_existing(self, name: str) -> bool:
        """Load a persisted index from disk."""
        try:
            store_path = os.path.join(VECTOR_STORE_DIR, f"{name}.pkl")
            index      = SimpleVectorIndex()
            if index.load(store_path):
                self._indexes[name] = index
                return True
            return False
        except Exception as exc:
            logger.error("Load error (%s): %s", name, exc)
            return False

    # ── retrieval ────────────────────────────────────────────────────────────

    def query(
        self,
        query: str,
        index_name: str = "__all__",
        top_k: int = RAG_TOP_K,
        mode: str = "concise",
    ) -> str:
        """
        Retrieve relevant chunks and synthesise an answer via LLM.
        Uses cache-aside: check cache → retrieve → generate → store.
        """
        # Cache check
        cached = self._cache.get(query, index_name)
        if cached:
            return cached

        # Gather context
        contexts = self._retrieve_contexts(query, index_name, top_k)
        if not contexts:
            answer = "I couldn't find relevant information in the uploaded documents."
            self._cache.set(query, index_name, answer)
            return answer

        context_str = "\n\n---\n\n".join(contexts)
        detail_hint = "" if mode == "concise" else " Provide a detailed explanation."
        messages    = [
            {"role": "system", "content": _ANSWER_SYSTEM},
            {"role": "user",   "content":
                f"Context:\n{context_str}\n\nQuestion: {query}{detail_hint}"},
        ]
        try:
            answer = chat_complete(messages, max_tokens=512, temperature=0.2)
        except Exception as exc:
            logger.error("RAG LLM call error: %s", exc)
            answer = "RAG answer generation failed."

        self._cache.set(query, index_name, answer)
        return answer

    def _retrieve_contexts(self, query: str, index_name: str, top_k: int) -> list[str]:
        if index_name == "__all__":
            # Search across all indexes, deduplicate by content hash
            seen    = set()
            results = []
            for idx in self._indexes.values():
                for chunk in idx.search(query, top_k):
                    h = hashlib.md5(chunk.encode()).hexdigest()
                    if h not in seen:
                        seen.add(h)
                        results.append(chunk)
            return results[:top_k]
        idx = self._indexes.get(index_name)
        return idx.search(query, top_k) if idx else []

    @property
    def index_names(self) -> list[str]:
        return list(self._indexes.keys())
