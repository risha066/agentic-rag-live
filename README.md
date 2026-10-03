\# Agentic RAG over Live Data (Groq + OpenAI GPT-OSS)



An agent that doesn't just answer from a static, pre-built index. It reasons

about what it needs, decides \*for itself\* when to search the live web, pulls

fresh pages, indexes them into a vector store on the fly, retrieves the most

relevant chunks, and only then answers — with citations back to the sources

it used.



!\[Agent trace demo](docs/screenshots/after-fix.png)



\---



\## Demo



\### Before fix — agent answered from training data, ignoring the fresh content it had just fetched:



!\[Before fix](docs/screenshots/before-fix.png)



\### After fix — agent cites live September 2026 news from the sources it fetched:



!\[After fix](docs/screenshots/after-fix.png)



\### Another query — Elon Musk in 2026 (proves the system generalizes across topics, not overfit to one):



!\[Elon Musk demo](docs/screenshots/demo-elon-musk.png)



\---



\## Why "agentic" and why "live"



\*\*Classic RAG:\*\* `query → retrieve from a fixed vector DB → generate.`



\*\*This project:\*\* `query → \*\*LLM decides\*\* which tool(s) to call (web search,

live page ingestion, vector retrieval, or just answer directly) → executes

tools → feeds results back to the LLM → repeats until it has enough grounding

→ answers.`



The vector store is continuously updated with freshly fetched content, so

answers reflect the current web, not a stale snapshot.



The agent loop is \*\*hand-rolled\*\* in `agent/agent.py` — no LangChain, no

LangGraph. Every decision the model makes is visible and auditable.



\---



\## Model / provider



\- \*\*Inference:\*\* \[Groq](https://console.groq.com) — OpenAI-compatible

&#x20; `/openai/v1/chat/completions` endpoint, used via the official `groq` Python SDK.

\- \*\*Model:\*\* `openai/gpt-oss-120b` by default (OpenAI's open-weight model,

&#x20; hosted on Groq's LPUs). \*\*No Llama models are used anywhere in this project.\*\*

&#x20; Switch to `openai/gpt-oss-20b` for faster/cheaper inference by editing `GROQ\_MODEL`

&#x20; in `.env` — both support native tool calling and reasoning effort control.

\- \*\*Embeddings:\*\* local `sentence-transformers` model (`all-MiniLM-L6-v2`)

&#x20; — free, runs on CPU, no extra API key needed. Swap it out in

&#x20; `rag/embeddings.py` if you'd rather call a hosted embeddings API.

\- \*\*Search:\*\* `ddgs` (DuckDuckGo) — free, no API key required.

\- \*\*Vector store:\*\* ChromaDB, persistent on disk under `./chroma\_db/`.



\---



\## Project layout



```

agentic-rag-live/

├── README.md

├── requirements.txt

├── .env.example              # copy to .env and fill in your keys

├── config.py                 # central settings, loaded from .env

├── main.py                   # CLI chat loop (entry point)

├── app.py                    # optional FastAPI server (POST /chat)

│

├── agent/

│   ├── \_\_init\_\_.py

│   ├── llm.py                # Groq client wrapper

│   ├── tools.py              # tool schemas + implementations

│   ├── agent.py              # the agentic tool-calling loop

│   └── memory.py             # conversation history buffer

│

├── rag/

│   ├── \_\_init\_\_.py

│   ├── embeddings.py         # embedding function wrapper

│   ├── vectorstore.py        # Chroma persistent + ephemeral "live" store

│   └── ingest.py             # live web search + fetch + chunk + index

│

├── tests/

│   ├── \_\_init\_\_.py

│   └── test\_agent.py

│

└── docs/

&#x20;   └── screenshots/

&#x20;       ├── before-fix.png

&#x20;       ├── after-fix.png

&#x20;       └── demo-elon-musk.png

```



\---



\## How it works (flow)



1\. You ask a question in `main.py`'s chat loop.

2\. `agent/agent.py` sends the conversation + tool schemas to Groq (`gpt-oss-120b`).

3\. The model decides whether to:

&#x20;  - `web\_search(query)` → `rag/ingest.py` searches the live web via `ddgs`

&#x20;  - `fetch\_and\_index(urls)` → downloads pages, chunks them, embeds them, and

&#x20;    upserts into ChromaDB so they become \*\*immediately queryable memory\*\*

&#x20;  - `retrieve(query)` → semantic similarity search over everything ingested

&#x20;    in this session and previous sessions

&#x20;  - `current\_datetime()` → fetch the current UTC date/time

4\. Tool results are fed back to the model, which may call more tools or produce

&#x20;  a final answer grounded in the retrieved + live content, with citations.

5\. The loop repeats until the model returns a plain-text final answer or the

&#x20;  step budget (`MAX\_AGENT\_STEPS`) is hit.



\---



\## Setup



```bash

\# 1. Clone and enter

git clone https://github.com/risha066/agentic-rag-live.git

cd agentic-rag-live



\# 2. Create + activate a virtual environment

python -m venv venv



\# Windows

venv\\Scripts\\Activate.ps1



\# Mac / Linux

source venv/bin/activate



\# 3. Install dependencies

pip install -r requirements.txt



\# 4. Create your .env file

cp .env.example .env

\# then edit .env and add your GROQ\_API\_KEY



\# 5. Run

python main.py

```



Then type your question at the `You:` prompt. Use `reset` to clear

conversation memory, `exit` (or `quit`) to leave.



\---



\## Required keys (in `.env`)



\- `GROQ\_API\_KEY` — \*\*required.\*\* Get one free at

&#x20; \[console.groq.com/keys](https://console.groq.com/keys).

\- `GROQ\_MODEL` — default `openai/gpt-oss-120b`.

\- `GROQ\_REASONING\_EFFORT` — `low` | `medium` | `high` (only used by gpt-oss).

\- `EMBEDDING\_MODEL` — local sentence-transformers model name.

\- `VECTOR\_DB\_DIR` — path for the persistent Chroma store.



No key is required for `fetch\_url` or `web\_search` — web search uses DuckDuckGo

via `ddgs` (no key). Embeddings run locally on CPU.



\---



\## Engineering note: fixing the `tool\_choice is none` bug



Once the agent hits `MAX\_AGENT\_STEPS`, the loop makes one final call with

`tools=None` so the model synthesizes a plain-text answer. On Groq, the

`gpt-oss` model imitated the `tool\_calls` pattern still present in message

history and tried to call a tool anyway — producing:



```

400 Bad Request: "Tool choice is none, but model called a tool"

```



\*\*Naive fix (wrong):\*\* strip all `tool\_calls` and `tool` role messages before

the final call. This stopped the crash — but it also deleted the tool results,

so the model fell back to training data and answered with stale 2024 facts

(see `before-fix.png`).



\*\*Correct fix\*\* (in `agent/agent.py`): flatten every tool call and its result

into a plain-text "research notes" block, keep only the system prompt and the

original user question, and pass that to the final synthesis call. The model

sees the data — without the pattern it was imitating.



\*\*Result:\*\* the agent now cites live September 2026 news from the sources it

actually fetched (see `after-fix.png` and `demo-elon-musk.png`).



\---



\## Engineering note: package rename



`duckduckgo-search` was renamed to `ddgs` upstream. `rag/ingest.py` imports

from `ddgs` directly. If you see a `RuntimeWarning: This package

(duckduckgo\_search) has been renamed to ddgs`, you're on an old version of the

dependency — run `pip install -U ddgs`.



\---



\## Swapping pieces



\- \*\*Different LLM:\*\* edit `agent/llm.py` only — it's the sole place the

&#x20; Groq client is instantiated.

\- \*\*Different vector DB\*\* (e.g. Pinecone, Qdrant, pgvector): re-implement

&#x20; `rag/vectorstore.py`'s `VectorStore` class. The rest of the app only calls

&#x20; `.add()`, `.query()`, and `.query\_long\_term()`.

\- \*\*Different embedding model:\*\* change `EMBEDDING\_MODEL` in `config.py`.

\- \*\*Add a new tool\*\* (e.g. a stock-price API): add its JSON schema + handler

&#x20; in `agent/tools.py`. The agent loop in `agent/agent.py` needs no changes.



\---



\## Notes / limitations



\- This is a compact reference implementation meant to be read and extended,

&#x20; not a production system — no auth, no async batching, no distributed

&#x20; vector store, no rate-limit handling on the Groq side.

\- Web search quality depends on DuckDuckGo's HTML endpoint (best-effort only).

\- The final synthesis step truncates the flattened research notes at \~12,000

&#x20; characters to stay within context limits.

\- Respect the terms of service and robots.txt of any site you fetch from.



\---



\## Stack



\- \*\*Groq\*\* — `openai/gpt-oss-120b` inference

\- \*\*ChromaDB\*\* — persistent vector store

\- \*\*sentence-transformers\*\* — local embeddings (`all-MiniLM-L6-v2`)

\- \*\*ddgs\*\* — live web search (no API key)

\- \*\*Trafilatura + BeautifulSoup\*\* — content extraction

\- \*\*Rich\*\* — CLI output

\- \*\*Python 3.10+\*\*



\---



\## License



MIT — free to use, modify, and learn from.

