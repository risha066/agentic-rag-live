"""The agentic RAG loop: the LLM decides, turn by turn, whether to call a
tool (search live web / fetch+index a page / retrieve from the vector
store / check the date) or to answer. This is a hand-rolled ReAct-style
loop against Groq's OpenAI-compatible tool-calling API.
"""
from __future__ import annotations

from typing import Optional

from config import Config
from agent.llm import chat_completion
from agent.memory import ConversationMemory
from agent.tools import TOOLS, run_tool

SYSTEM_PROMPT = """You are an agentic research assistant with live web access.

You have four tools:
- web_search(query): search the live web, returns titles/URLs/snippets.
- fetch_and_index(urls): fetch pages and index their text for retrieval.
- retrieve(query): semantic search over everything indexed so far.
- current_datetime(): get the current UTC date/time.

Guidelines:
1. For anything time-sensitive, current-events-related, or where you are not
   fully confident from your own knowledge, use web_search first rather than
   guessing.
2. After web_search, call fetch_and_index on the 2-4 most relevant URLs, then
   call retrieve to pull the specific passages you need before answering.
3. If the question is simple, timeless, or you already retrieved enough
   context, answer directly without more tool calls.
4. Always cite your sources inline as [1], [2], etc., and list the
   corresponding URLs at the end of your answer under "Sources:".
5. If tools fail or return nothing useful, say so plainly rather than
   fabricating an answer.
6. Be concise and directly answer what was asked before adding extra detail.
"""


class AgenticRAGAgent:
    def __init__(self) -> None:
        Config.validate()
        self.memory = ConversationMemory(SYSTEM_PROMPT)

    def ask(self, user_message: str, verbose: bool = False) -> str:
        self.memory.add_user(user_message)

        for step in range(Config.MAX_AGENT_STEPS):
            response = chat_completion(self.memory.messages, tools=TOOLS)
            choice = response.choices[0]
            msg = choice.message

            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                # Final answer.
                self.memory.add_assistant(msg.content)
                return msg.content or ""

            # Record the assistant's tool-call request, then execute each tool.
            self.memory.add_assistant(
                msg.content,
                tool_calls=[
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            )

            for tc in tool_calls:
                if verbose:
                    print(f"  -> tool: {tc.function.name}({tc.function.arguments})")
                result = run_tool(tc.function.name, tc.function.arguments)
                self.memory.add_tool_result(tc.id, tc.function.name, result)

        # Ran out of steps: ask the model to wrap up with what it has.
               # Run out of steps: ask the model to wrap up with what it has.
        # Strip tool-call/tool-result messages — gpt-oss on Groq imitates
        # those patterns and tries to call more tools even when tools=None.
               # Run out of steps: ask the model to wrap up with what it has.
        # Groq + gpt-oss imitates tool_calls if they're in the history,
        # so we can't send raw tool messages. But we also don't want to
        # lose the fetched content. Solution: flatten tool activity into
        # a plain-text "research notes" block and send that.
        research_notes: List[str] = []
        for m in self.memory.messages:
            role = m.get("role")
            if role == "tool":
                research_notes.append(f"[TOOL RESULT]\n{m.get('content', '')}")
            elif "tool_calls" in m and m["tool_calls"]:
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    research_notes.append(
                        f"[TOOL CALLED] {fn.get('name', '?')}({fn.get('arguments', '')})"
                    )

        notes_blob = "\n\n".join(research_notes) if research_notes else "(no tool results gathered)"
        if len(notes_blob) > 12000:
            notes_blob = notes_blob[:12000] + "\n\n[...truncated]"

        # Keep only the system prompt + the original user question.
        final_messages = [
            self.memory.messages[0],          # system prompt
            self.memory.messages[1],          # original user question
            {
                "role": "user",
                "content": (
                    "You've used the maximum number of tool calls. "
                    "Here are your research notes from the tools you ran:\n\n"
                    f"{notes_blob}\n\n"
                    "Now answer the original question using ONLY the information "
                    "in these notes plus your own reasoning. Cite sources where "
                    "possible. If the notes don't contain the answer, say so "
                    "explicitly instead of falling back on training data."
                ),
            },
        ]

        response = chat_completion(final_messages, tools=[], tool_choice="none")
        final = response.choices[0].message.content or ""
        self.memory.add_assistant(final)
        return final
    def reset(self) -> None:
        self.memory.reset()
