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


def _extract_pdf(path):
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


def _extract_docx(path):
    try:
        import docx
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as exc:
        logger.error("DOCX extract error (%s): %s", path, exc)
        return ""


def extract_text(path):
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext in (".docx", ".doc"):
        return _extract_docx(path)
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logger.error("Text extract error (%s): %s", path, exc)
        return ""


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return [c for c in chunks if c.strip()]


class SimpleVectorIndex:
    def __init__(self):
        self.vectors = None
        self.chunks = []

    def add(self, chunks):
        self.chunks = chunks
        self.vectors = embed_texts(chunks)

    def search(self, query, top_k=RAG_TOP_K):
        if self.vectors is None or not self.chunks:
            return []
        qvec = embed_query(query)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10
        qnorm = np.linalg.norm(qvec) + 1e-10
        sims = (self.vectors / norms) @ (qvec / qnorm)
        indices = np.argsort(sims)[::-1][:top_k]
        return [self.chunks[i] for i in indices]

    def save(self, path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({"vectors": self.vectors, "chunks": self.chunks}, f)
        except Exception as exc:
            logger.error("Index save error: %s", exc)

    def load(self, path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.vectors = data["vectors"]
            self.chunks = data["chunks"]
            return True
        except Exception:
            return False


class QueryCache:
    def __init__(self, ttl=CACHE_TTL_SECONDS, max_entries=MAX_CACHE_ENTRIES):
        self._store = {}
        self.ttl = ttl
        self.max_entries = max_entries

    def _key(self, query, index_name):
        return hashlib.md5(f"{index_name}::{query}".encode()).hexdigest()

    def get(self, query, index_name):
        k = self._key(query, index_name)
        entry = self._store.get(k)
        if entry and time.time() - entry[1] < self.ttl:
            return entry[0]
        return None

    def set(self, query, index_name, answer):
        if len(self._store) >= self.max_entries:
            oldest = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest]
        self._store[self._key(query, index_name)] = (answer, time.time())


_ANSWER_SYSTEM = """You are a financial research assistant.
Use ONLY the provided context excerpts to answer the question.
If the context does not contain enough information, say so clearly.
Be concise and factual.
"""


class RAGEngine:
    def __init__(self):
        self._indexes = {}
        self._cache = QueryCache()
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    def ingest(self, name, file_path):
        text = extract_text(file_path)
        if not text.strip():
            logger.warning("No text extracted from %s", file_path)
            return 0
        chunks = chunk_text(text)
        index = SimpleVectorIndex()
        index.add(chunks)
        self._indexes[name] = index
        index.save(os.path.join(VECTOR_STORE_DIR, f"{name}.pkl"))
        logger.info("Ingested '%s': %d chunks.", name, len(chunks))
        return len(chunks)

    def load_existing(self, name):
        index = SimpleVectorIndex()
        if index.load(os.path.join(VECTOR_STORE_DIR, f"{name}.pkl")):
            self._indexes[name] = index
            return True
        return False

    def query(self, query, index_name="__all__", top_k=RAG_TOP_K, mode="concise"):
        cached = self._cache.get(query, index_name)
        if cached:
            return cached

        contexts = self._retrieve_contexts(query, index_name, top_k)
        if not contexts:
            answer = "I couldn't find relevant information in the uploaded documents."
            self._cache.set(query, index_name, answer)
            return answer

        context_str = "\n\n---\n\n".join(contexts)
        detail_hint = "" if mode == "concise" else " Provide a detailed explanation."
        messages = [
            {"role": "system", "content": _ANSWER_SYSTEM},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}{detail_hint}"},
        ]
        try:
            answer = chat_complete(messages, max_tokens=512, temperature=0.2)
        except Exception as exc:
            logger.error("RAG LLM call error: %s", exc)
            answer = "RAG answer generation failed."

        self._cache.set(query, index_name, answer)
        return answer

    def _retrieve_contexts(self, query, index_name, top_k):
        if index_name == "__all__":
            seen = set()
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
    def index_names(self):
        return list(self._indexes.keys())