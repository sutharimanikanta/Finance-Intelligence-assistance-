"""
config/config.py
All API keys and application settings.
DO NOT commit this file with real keys to version control.
"""

import os

# ─────────────────────────────────────────────
# LLM (Groq)
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────
# Web Search (Tavily)
# ─────────────────────────────────────────────
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ─────────────────────────────────────────────
# Embedding model (used by RAG)
# ─────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ─────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────
DB_PATH = "data/financial_data.db"

# ─────────────────────────────────────────────
# RAG / Vector store
# ─────────────────────────────────────────────
VECTOR_STORE_DIR = "data/vector_stores"   # parent dir; sub-dirs per index
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50
RAG_TOP_K        = 4

# ─────────────────────────────────────────────
# Cache (cache-aside pattern)
# ─────────────────────────────────────────────
CACHE_TTL_SECONDS = 300   # 5 minutes
MAX_CACHE_ENTRIES = 200

# ─────────────────────────────────────────────
# Response modes
# ─────────────────────────────────────────────
RESPONSE_MODE_CONCISE  = "concise"
RESPONSE_MODE_DETAILED = "detailed"

# ─────────────────────────────────────────────
# Conversation history window
# ─────────────────────────────────────────────
CONVERSATION_WINDOW = 4   # last N turns sent to LLM for context
