"""Tool definitions (OpenAI-style function-calling schemas) plus the
local Python functions that actually execute them. The agent loop in
agent.py decides which of these to call and when.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from rag.ingest import web_search as _web_search, fetch_and_index as _fetch_and_index
from rag.vectorstore import get_store

# ---------------------------------------------------------------------------
# 1. JSON schemas the LLM sees (Groq/OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live web for current information. Returns titles, "
                "URLs and short snippets. Use this first for anything that "
                "might have changed recently or that you're not confident "
                "about from memory alone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "How many results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_and_index",
            "description": (
                "Fetch one or more URLs (e.g. from web_search results), extract "
                "their clean text, chunk it, and index it into the vector store "
                "so it becomes retrievable via the `retrieve` tool. Call this on "
                "the most promising 2-4 URLs before retrieving."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of URLs to fetch and index.",
                    }
                },
                "required": ["urls"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": (
                "Semantic search over everything indexed so far (both freshly "
                "fetched live pages and any persistent knowledge base). Use "
                "this after fetch_and_index, or directly if relevant content "
                "may already be indexed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look up."},
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to return (default 6).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_datetime",
            "description": "Get the current UTC date and time. Use for time-relative questions (e.g. 'this week', 'today').",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ---------------------------------------------------------------------------
# 2. Local implementations, dispatched by name
# ---------------------------------------------------------------------------


def _tool_web_search(args: Dict[str, Any]) -> Dict[str, Any]:
    results = _web_search(args["query"], max_results=args.get("max_results"))
    return {"results": results}


def _tool_fetch_and_index(args: Dict[str, Any]) -> Dict[str, Any]:
    return _fetch_and_index(args["urls"])


def _tool_retrieve(args: Dict[str, Any]) -> Dict[str, Any]:
    store = get_store()
    hits = store.query(args["query"], top_k=args.get("top_k"))
    return {"matches": hits}


def _tool_current_datetime(_args: Dict[str, Any]) -> Dict[str, Any]:
    return {"utc_datetime": datetime.now(timezone.utc).isoformat()}


DISPATCH = {
    "web_search": _tool_web_search,
    "fetch_and_index": _tool_fetch_and_index,
    "retrieve": _tool_retrieve,
    "current_datetime": _tool_current_datetime,
}


def run_tool(name: str, arguments_json: str) -> str:
    """Execute a tool by name and return a JSON string result (or an error)."""
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"invalid tool arguments: {e}"})

    fn = DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool '{name}'"})

    try:
        result = fn(args)
    except Exception as e:  # noqa: BLE001 - surface tool failures to the LLM
        result = {"error": str(e)}
    return json.dumps(result, default=str)
