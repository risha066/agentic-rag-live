"""Thin wrapper around the Groq SDK (OpenAI-compatible chat completions)."""
from __future__ import annotations

from typing import List, Dict, Any, Optional

from groq import Groq

from config import Config

_client: Optional[Groq] = None


def get_client() -> Groq:
    global _client
    if _client is None:
        Config.validate()
        _client = Groq(api_key=Config.GROQ_API_KEY)
    return _client


def chat_completion(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
):
    """Single call to Groq's chat.completions endpoint.

    Uses the OpenAI open-weight model configured in GROQ_MODEL
    (default: openai/gpt-oss-20b) — never a Llama model.
    """
    client = get_client()
    kwargs: Dict[str, Any] = dict(
        model=Config.GROQ_MODEL,
        messages=messages,
        temperature=Config.LLM_TEMPERATURE,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    else:
        kwargs["tools"] = []
        kwargs["tool_choice"] = "none"
    # gpt-oss models on Groq accept a reasoning_effort knob.
    if "gpt-oss" in Config.GROQ_MODEL:
        kwargs["reasoning_effort"] = Config.GROQ_REASONING_EFFORT

    return client.chat.completions.create(**kwargs)
