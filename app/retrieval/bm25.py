"""BM25 keyword search over the chunks already stored in Chroma."""
import re
from typing import Any

from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from app.retrieval.base import RetrievedChunk, Retriever

_TOKEN_RE = re.compile(r"\w+")

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how in is it of on or that the "
    "this to was what when where which who why with".split()
)


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


class BM25Retriever(Retriever):
    name = "bm25"
    score_meaning = "BM25 score (higher = better; chunks matching no query word are not returned)"

    def __init__(self, store: Chroma, remove_stopwords: bool = False):
        self._store = store
        self._remove_stopwords = remove_stopwords
        self._signature: int | None = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None

    def _ensure_index(self) -> None:
        ids = self._store.get(include=[])["ids"]
        signature = hash(tuple(sorted(ids)))
        if signature == self._signature:
            return  # store unchanged since the index was built
        data = self._store.get(include=["documents", "metadatas"])
        self._ids = data["ids"]
        self._texts = data["documents"]
        self._metadatas = data["metadatas"]
        tokenized = [tokenize(t, self._remove_stopwords) for t in self._texts]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        self._signature = signature

    def retrieve(
        self, query: str, k: int, filter: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        self._ensure_index()
        if self._bm25 is None:
            return []
        query_tokens = tokenize(query, self._remove_stopwords)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        allowed = set(self._store.get(where=filter, include=[])["ids"]) if filter else None

        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: list[RetrievedChunk] = []
        for i in order:
            if scores[i] <= 0:
                break
            if allowed is not None and self._ids[i] not in allowed:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=self._ids[i],
                    content=self._texts[i],
                    metadata=self._metadatas[i],
                    rank=len(results) + 1,
                    score=float(scores[i]),
                )
            )
            if len(results) == k:
                break
        return results