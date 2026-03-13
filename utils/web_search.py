"""
utils/web_search.py
Real-time web search via Tavily, with result synthesis by LLM.
"""

import logging
from typing import Optional

from tavily import TavilyClient

from config.config import TAVILY_API_KEY
from models.llm import chat_complete

logger = logging.getLogger(__name__)

_client: Optional[TavilyClient] = None

_SYNTHESIS_SYSTEM = """You are a concise financial news analyst.
Summarise the web search results below to directly answer the question.
Cite source titles where relevant. Be factual and brief.
"""


def get_tavily_client() -> TavilyClient:
    global _client
    if _client is None:
        try:
            _client = TavilyClient(api_key=TAVILY_API_KEY)
        except Exception as exc:
            logger.error("Tavily init error: %s", exc)
            raise
    return _client


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Run a Tavily search.

    Returns:
        List of dicts with keys: title, url, content, score.
    """
    try:
        client  = get_tavily_client()
        resp    = client.search(query, max_results=max_results, search_depth="advanced")
        results = resp.get("results", [])
        logger.info("Web search '%s': %d results.", query[:50], len(results))
        return results
    except Exception as exc:
        logger.error("Web search error: %s", exc)
        return []


def synthesise_web_results(query: str, results: list[dict], mode: str = "concise") -> str:
    """
    Use the LLM to synthesise raw search results into a readable answer.
    """
    if not results:
        return "No recent web results found for this query."

    snippets = []
    for r in results[:5]:
        title   = r.get("title", "")
        content = r.get("content", "")[:400]
        url     = r.get("url", "")
        snippets.append(f"**{title}** ({url})\n{content}")

    context  = "\n\n".join(snippets)
    detail   = "" if mode == "concise" else " Provide a detailed analysis."
    messages = [
        {"role": "system", "content": _SYNTHESIS_SYSTEM},
        {"role": "user",   "content": f"Results:\n{context}\n\nQuestion: {query}{detail}"},
    ]
    try:
        return chat_complete(messages, max_tokens=512, temperature=0.3)
    except Exception as exc:
        logger.error("Web synthesis LLM error: %s", exc)
        return "Web search results retrieved but synthesis failed."


def web_answer(query: str, mode: str = "concise") -> str:
    """Convenience wrapper: search + synthesise."""
    results = search_web(query)
    return synthesise_web_results(query, results, mode)
