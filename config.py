"""Central configuration, loaded from environment / .env file."""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # --- Groq / LLM ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    # OpenAI open-weight model hosted on Groq. Deliberately NOT a Llama model.
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    GROQ_REASONING_EFFORT: str = os.getenv("GROQ_REASONING_EFFORT", "medium")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    # --- Embeddings / vector store ---
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    VECTOR_DB_DIR: str = os.getenv("VECTOR_DB_DIR", "./chroma_db")

    # --- Retrieval / ingestion tuning ---
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    TOP_K: int = int(os.getenv("TOP_K", "6"))

    # --- Agent loop ---
    MAX_AGENT_STEPS: int = int(os.getenv("MAX_AGENT_STEPS", "6"))

    @classmethod
    def validate(cls) -> None:
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add "
                "your key from https://console.groq.com/keys"
            )
        if "llama" in cls.GROQ_MODEL.lower():
            raise ValueError(
                f"GROQ_MODEL='{cls.GROQ_MODEL}' looks like a Llama model. "
                "This project is configured to use an OpenAI model on Groq "
                "(e.g. openai/gpt-oss-20b or openai/gpt-oss-120b) instead."
            )
