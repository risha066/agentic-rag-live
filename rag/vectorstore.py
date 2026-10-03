"""Vector store: a durable 'persistent' collection plus a per-session 'live'
collection that holds pages fetched from the web during the current
conversation. Retrieval queries both and merges results.
"""
from __future__ import annotations

import hashlib
import time
from typing import List, Dict, Any, Optional

import chromadb

from config import Config
from rag.embeddings import LocalEmbeddingFunction

_EMBED_FN = LocalEmbeddingFunction()


def _id_for(text: str, source: str) -> str:
    return hashlib.sha256(f"{source}::{text[:200]}".encode()).hexdigest()[:24]


class VectorStore:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
        self.persistent = self._client.get_or_create_collection(
            name="persistent_knowledge",
            embedding_function=_EMBED_FN,
            metadata={"hnsw:space": "cosine"},
        )
        # In-memory, ephemeral: cleared each process start. Holds live-fetched
        # pages so a session's fresh research doesn't permanently pollute the
        # durable knowledge base unless explicitly promoted.
        self._live_client = chromadb.EphemeralClient()
        self.live = self._live_client.get_or_create_collection(
            name="live_session",
            embedding_function=_EMBED_FN,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: List[str],
        source: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
        persistent: bool = False,
    ) -> int:
        """Add chunks to the live (default) or persistent collection."""
        if not chunks:
            return 0
        collection = self.persistent if persistent else self.live
        ids = [_id_for(c, source) for c in chunks]
        metadatas = [
            {
                "source": source,
                "ingested_at": time.time(),
                **(extra_metadata or {}),
            }
            for _ in chunks
        ]
        # Upsert so re-fetching the same page doesn't create duplicates.
        collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
        return len(chunks)

    def query(self, text: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Query both collections, merge, and return sorted by relevance."""
        top_k = top_k or Config.TOP_K
        results: List[Dict[str, Any]] = []
        for collection, label in ((self.live, "live"), (self.persistent, "persistent")):
            if collection.count() == 0:
                continue
            n = min(top_k, collection.count())
            res = collection.query(query_texts=[text], n_results=n)
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                results.append(
                    {
                        "text": doc,
                        "source": meta.get("source", "unknown"),
                        "collection": label,
                        "score": 1 - dist,  # cosine distance -> similarity
                    }
                )
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]


_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
