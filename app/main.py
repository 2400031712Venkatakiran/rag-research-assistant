import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException

from app.config import Settings, get_settings
from app.embeddings.embedder import build_embeddings
from app.generation.answer import AnswerGenerator, build_llm
from app.ingestion.pipeline import ingest_file
from app.retrieval.base import Retriever
from app.retrieval.factory import available_methods, get_retriever
from app.schemas.requests import IngestRequest, QueryRequest, RetrieveRequest
from app.schemas.responses import (
    IngestFileResult,
    IngestResponse,
    QueryResponse,
    RetrieveResponse,
)
from app.vectorstore.chroma import get_vectorstore

logger = logging.getLogger("app")


# --- Dependency providers (cached singletons; tests override these) ---------
@lru_cache
def get_store():
    settings = get_settings()
    return get_vectorstore(settings, build_embeddings(settings))


def get_retriever_dep(
    settings: Settings = Depends(get_settings), store=Depends(get_store)
) -> Retriever:
    return get_retriever(settings, store)  # cheap: just wraps the store


@lru_cache
def get_generator() -> AnswerGenerator:
    return AnswerGenerator(build_llm(get_settings()))


# --- App --------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    if settings.retrieval_method not in available_methods():
        raise RuntimeError(
            f"Unknown RETRIEVAL_METHOD '{settings.retrieval_method}'. "
            f"Available: {available_methods()}"
        )
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting with retrieval method: %s", settings.retrieval_method)
    yield


app = FastAPI(title="RAG Research Assistant", lifespan=lifespan)


@app.get("/health")
def health(settings: Settings = Depends(get_settings)):
    return {"status": "ok", "retrieval_method": settings.retrieval_method}


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    req: IngestRequest,
    settings: Settings = Depends(get_settings),
    store=Depends(get_store),
):
    if req.files is None:
        paths = sorted(settings.documents_dir.glob("*.pdf"))
    else:
        paths = []
        for name in req.files:
            if Path(name).name != name:
                raise HTTPException(400, f"Invalid file name: {name}")
            p = settings.documents_dir / name
            if not p.is_file():
                raise HTTPException(404, f"Not found in documents dir: {name}")
            paths.append(p)
    if not paths:
        raise HTTPException(404, "No PDF files to ingest.")

    results = []
    for p in paths:
        removed, added = ingest_file(p, store, settings)
        results.append(IngestFileResult(source=p.name, chunks_removed=removed, chunks_added=added))
    return IngestResponse(results=results)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    req: RetrieveRequest,
    settings: Settings = Depends(get_settings),
    retriever: Retriever = Depends(get_retriever_dep),
):
    """Inspect retrieved chunks without calling the LLM."""
    chunks = retriever.retrieve(req.question, req.top_k or settings.top_k, req.filter)
    return RetrieveResponse(method=retriever.name, chunks=chunks)


@app.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    settings: Settings = Depends(get_settings),
    retriever: Retriever = Depends(get_retriever_dep),
    generator: AnswerGenerator = Depends(get_generator),
):
    chunks = retriever.retrieve(req.question, req.top_k or settings.top_k, req.filter)
    try:
        result = generator.generate(req.question, chunks)
    except Exception:
        logger.exception("LLM generation failed")
        raise HTTPException(502, "Answer generation failed; see server logs.")
    return QueryResponse(
        method=retriever.name, answer=result.answer, sources=result.sources, retrieved=chunks
    )