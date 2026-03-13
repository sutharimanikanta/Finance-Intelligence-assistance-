"""
models/llm.py
Groq LLM client wrapper.
"""

import logging
from groq import Groq
from config.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

_client: Groq | None = None


def get_llm_client() -> Groq:
    """Return a singleton Groq client."""
    global _client
    if _client is None:
        try:
            _client = Groq(api_key=GROQ_API_KEY)
            logger.info("Groq client initialised.")
        except Exception as exc:
            logger.error("Failed to initialise Groq client: %s", exc)
            raise
    return _client


def chat_complete(
    messages: list[dict],
    model: str = GROQ_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> str:
    """
    Send messages to Groq and return the assistant text.

    Args:
        messages:     OpenAI-style message list.
        model:        Model id.
        max_tokens:   Upper bound on output tokens.
        temperature:  0 = deterministic.
        json_mode:    Request JSON output (where supported).

    Returns:
        Stripped string content.
    """
    client = get_llm_client()
    kwargs: dict = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        raise
