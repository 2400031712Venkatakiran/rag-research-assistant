# RAG Research Assistant

Configurable RAG app for comparing retrieval strategies over one document collection.
Current stage: **Stage 2 — dense baseline** (`RETRIEVAL_METHOD=dense`).

## Architecture

    PDF -> loader -> chunker (deterministic IDs) -> Chroma (persistent)
                                                       |
    question -> Retriever (chosen by RETRIEVAL_METHOD) -> [RetrievedChunk...]
                                                       |
                                  AnswerGenerator (LLM, cites [n]) -> answer + sources

- `retrieval/base.py` defines the only contract between retrieval and everything else.
- `retrieval/factory.py` is the only place that maps a method name to a class.
- `generation/answer.py` and `main.py` never import a concrete retriever.

## Run

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env        # add your API key
    cp your.pdf documents/
    uvicorn app.main:app --reload

    curl -X POST localhost:8000/ingest -H 'content-type: application/json' -d '{}'
    curl -X POST localhost:8000/retrieve -H 'content-type: application/json' \
         -d '{"question": "How does dependency injection work?", "top_k": 4}'
    curl -X POST localhost:8000/query -H 'content-type: application/json' \
         -d '{"question": "How does dependency injection work?"}'

`/retrieve` returns chunks only (no LLM call) for inspection. `/query` adds the answer.
Metadata filter example: `"filter": {"source": "your.pdf"}`.

Docker: `docker compose up --build`. Tests: `pytest` (no API keys needed).

## Adding a retrieval method
1. New file in `app/retrieval/` implementing `Retriever`.
2. Register one line in `factory.py`.
3. Set `RETRIEVAL_METHOD=<name>`.

## Notes
- Re-ingesting a file replaces its chunks (delete by `source`, upsert by deterministic ID).
- Dense `score` is Chroma distance: lower is more similar.
