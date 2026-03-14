"""
models/llm.py
Groq LLM client wrapper.
"""
import logging

from groq import Groq

from config.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

_client = None


def get_llm_client():
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialised.")
    return _client


def chat_complete(messages, model=GROQ_MODEL, max_tokens=1024, temperature=0.0, json_mode=False):
    client = get_llm_client()
    kwargs = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        raise