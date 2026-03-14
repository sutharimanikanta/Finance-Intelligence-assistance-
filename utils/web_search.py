"""
utils/web_search.py
Real-time web search via Tavily, with result synthesis by LLM.
"""
import logging

from tavily import TavilyClient

from config.config import TAVILY_API_KEY
from models.llm import chat_complete

logger = logging.getLogger(__name__)

_client = None

_SYNTHESIS_SYSTEM = """You are a concise financial news analyst.
Summarise the web search results below to directly answer the question.
Cite source titles where relevant. Be factual and brief.
"""


def get_tavily_client():
    global _client
    if _client is None:
        _client = TavilyClient(api_key=TAVILY_API_KEY)
    return _client


def search_web(query, max_results=5):
    try:
        client = get_tavily_client()
        resp = client.search(query, max_results=max_results, search_depth="advanced")
        results = resp.get("results", [])
        logger.info("Web search '%s': %d results.", query[:50], len(results))
        return results
    except Exception as exc:
        logger.error("Web search error: %s", exc)
        return []


def synthesise_web_results(query, results, mode="concise"):
    if not results:
        return "No recent web results found for this query."

    snippets = []
    for r in results[:5]:
        title = r.get("title", "")
        content = r.get("content", "")[:400]
        url = r.get("url", "")
        snippets.append(f"{title} ({url})\n{content}")

    context = "\n\n".join(snippets)
    detail = "" if mode == "concise" else " Provide a detailed analysis."
    messages = [
        {"role": "system", "content": _SYNTHESIS_SYSTEM},
        {"role": "user", "content": f"Results:\n{context}\n\nQuestion: {query}{detail}"},
    ]
    try:
        return chat_complete(messages, max_tokens=512, temperature=0.3)
    except Exception as exc:
        logger.error("Web synthesis error: %s", exc)
        return "Web search results retrieved but synthesis failed."


def web_answer(query, mode="concise"):
    results = search_web(query)
    return synthesise_web_results(query, results, mode)