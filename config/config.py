"""
config/config.py
All API keys and application settings.
DO NOT commit this file with real keys to version control.
"""
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DB_PATH = "data/financial_data.db"

VECTOR_STORE_DIR = "data/vector_stores"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RAG_TOP_K = 4

CACHE_TTL_SECONDS = 300
MAX_CACHE_ENTRIES = 200

RESPONSE_MODE_CONCISE = "concise"
RESPONSE_MODE_DETAILED = "detailed"

CONVERSATION_WINDOW = 4