"""Live data pipeline: search the web, fetch pages, extract clean text,
chunk it, and index it into the vector store — all callable by the agent
at query time so answers reflect current information.
"""
from __future__ import annotations

from typing import List, Dict, Any

import requests
import trafilatura
from ddgs import DDGS

from config import Config
from rag.vectorstore import get_store

HEADERS = {"User-Agent": "Mozilla/5.0 (AgenticRAG/1.0)"}


def web_search(query: str, max_results: int = None) -> List[Dict[str, str]]:
    """Live web search. Returns [{title, url, snippet}, ...] — no scraping yet."""
    max_results = max_results or Config.MAX_SEARCH_RESULTS
    out: List[Dict[str, str]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            out.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
            )
    return out


def fetch_clean_text(url: str, timeout: int = 10) -> str:
    """Fetch a URL and extract main readable text (strips nav/ads/boilerplate)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        return f""  # fail closed; caller treats empty as "couldn't fetch"
    text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
    return text or ""


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    chunk_size = chunk_size or Config.CHUNK_SIZE
    overlap = overlap or Config.CHUNK_OVERLAP
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


def fetch_and_index(urls: List[str], persistent: bool = False) -> Dict[str, Any]:
    """Fetch each URL, chunk it, and index into the vector store.

    Returns a summary the agent can read to know what succeeded.
    """
    store = get_store()
    indexed: List[Dict[str, Any]] = []
    failed: List[str] = []
    for url in urls:
        text = fetch_clean_text(url)
        if not text or len(text) < 200:
            failed.append(url)
            continue
        chunks = chunk_text(text)
        count = store.add_chunks(chunks, source=url, persistent=persistent)
        indexed.append({"url": url, "chunks_indexed": count, "chars": len(text)})
    return {"indexed": indexed, "failed": failed}
