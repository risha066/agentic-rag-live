"""Smoke tests. These mock the Groq API and network calls so the suite runs
without a real API key or internet access:

    pytest tests/test_agent.py -v
"""
from __future__ import annotations

import json
import types
from unittest.mock import patch, MagicMock

from rag.ingest import chunk_text


def test_chunk_text_basic():
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    # Overlap means consecutive chunks share a tail/head region.
    assert chunks[0][-10:] in chunks[1]


def test_chunk_text_empty():
    assert chunk_text("") == []


def _fake_tool_call(name: str, arguments: dict):
    tc = types.SimpleNamespace()
    tc.id = "call_1"
    tc.function = types.SimpleNamespace(name=name, arguments=json.dumps(arguments))
    return tc


@patch("agent.agent.chat_completion")
def test_agent_answers_without_tools(mock_chat):
    """If the model returns plain text with no tool_calls, the agent should
    return it immediately without looping."""
    from agent.agent import AgenticRAGAgent

    fake_message = types.SimpleNamespace(content="42.", tool_calls=None)
    fake_choice = types.SimpleNamespace(message=fake_message)
    mock_chat.return_value = types.SimpleNamespace(choices=[fake_choice])

    with patch("agent.agent.Config.validate"):
        agent = AgenticRAGAgent.__new__(AgenticRAGAgent)
        from agent.memory import ConversationMemory
        from agent.agent import SYSTEM_PROMPT

        agent.memory = ConversationMemory(SYSTEM_PROMPT)

    answer = agent.ask("What is 6 * 7?")
    assert answer == "42."
    mock_chat.assert_called_once()


@patch("agent.tools._web_search")
def test_tool_dispatch_web_search(mock_search):
    from agent.tools import run_tool

    mock_search.return_value = [{"title": "t", "url": "u", "snippet": "s"}]
    result = json.loads(run_tool("web_search", json.dumps({"query": "test"})))
    assert result["results"][0]["url"] == "u"


def test_tool_dispatch_unknown_tool():
    from agent.tools import run_tool

    result = json.loads(run_tool("not_a_real_tool", "{}"))
    assert "error" in result
