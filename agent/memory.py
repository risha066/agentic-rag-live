"""Minimal conversation memory: keeps a running list of chat messages and
trims from the middle once it gets long, preserving the system prompt and
the most recent turns (which matter most for tool-calling context)."""
from __future__ import annotations

from typing import List, Dict, Any

MAX_MESSAGES = 40  # soft cap before trimming


class ConversationMemory:
    def __init__(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

    def add(self, role: str, content: Any, **extra: Any) -> None:
        msg: Dict[str, Any] = {"role": role, "content": content}
        msg.update(extra)
        self.messages.append(msg)
        self._trim()

    def add_user(self, content: str) -> None:
        self.add("user", content)

    def add_assistant(self, content: str | None, tool_calls=None) -> None:
        msg: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)
        self._trim()

    def add_tool_result(self, tool_call_id: str, name: str, content: str) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": content,
            }
        )
        self._trim()

    def _trim(self) -> None:
        if len(self.messages) <= MAX_MESSAGES:
            return
        # Keep system prompt + most recent messages.
        system = self.messages[0]
        recent = self.messages[-(MAX_MESSAGES - 1):]
        self.messages = [system] + recent

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self._system_prompt}]
