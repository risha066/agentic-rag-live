"""Optional HTTP API for the agent: POST /chat {"message": "..."}.

Run with: uvicorn app:app --reload --port 8000

Note: this demo keeps one shared agent (and thus one shared conversation
history) per process for simplicity. For multi-user production use, key
a dict of AgenticRAGAgent instances by session/user id instead.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import Config
from agent.agent import AgenticRAGAgent

app = FastAPI(title="Agentic RAG over Live Data")
_agent: AgenticRAGAgent | None = None


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    model: str


def get_agent() -> AgenticRAGAgent:
    global _agent
    if _agent is None:
        Config.validate()
        _agent = AgenticRAGAgent()
    return _agent


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    agent = get_agent()
    answer = agent.ask(req.message)
    return ChatResponse(answer=answer, model=Config.GROQ_MODEL)


@app.post("/reset")
def reset() -> dict:
    get_agent().reset()
    return {"status": "ok"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": Config.GROQ_MODEL}
