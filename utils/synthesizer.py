"""
utils/synthesizer.py
Combines SQL, RAG, and/or web answers into a single coherent response.
"""

import logging

from models.llm import chat_complete

logger = logging.getLogger(__name__)

_COMBINE_SYSTEM = """You are NeoStats, an AI-powered investment research assistant.
You have retrieved information from multiple sources for the user's query.
Synthesise the information into one coherent, professional answer.
- Lead with portfolio-specific data when available.
- Integrate research document insights naturally.
- Add web/news context at the end.
- Do not repeat yourself.
- Keep the tone analytical and concise.
"""


def combine_answers(
    query: str,
    sql_answer: str | None,
    rag_answer: str | None,
    web_answer: str | None,
    mode: str = "concise",
) -> str:
    """
    Merge answers from different sources.
    If only one source has data, return it directly.
    """
    try:
        parts = []
        if sql_answer and "couldn't find" not in sql_answer.lower():
            parts.append(f"[Portfolio Data]\n{sql_answer}")
        if rag_answer and "couldn't find" not in rag_answer.lower():
            parts.append(f"[Research Documents]\n{rag_answer}")
        if web_answer and "No recent" not in web_answer:
            parts.append(f"[Market Intelligence]\n{web_answer}")

        if not parts:
            return "I couldn't find relevant information from any of the available sources for your query."

        if len(parts) == 1:
            # Single source — no need to call LLM
            return parts[0].split("\n", 1)[1].strip()

        # Multi-source — synthesise
        detail   = "" if mode == "concise" else " Provide a detailed, structured response."
        combined = "\n\n".join(parts)
        messages = [
            {"role": "system", "content": _COMBINE_SYSTEM},
            {"role": "user",   "content":
                f"Question: {query}\n\nSources:\n{combined}{detail}"},
        ]
        return chat_complete(messages, max_tokens=800, temperature=0.2)
    except Exception as exc:
        logger.error("Synthesizer error: %s", exc)
        return "\n\n".join(p for p in [sql_answer, rag_answer, web_answer] if p)
