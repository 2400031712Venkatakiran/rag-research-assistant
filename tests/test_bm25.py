import pytest
from langchain_core.documents import Document

from app.retrieval.bm25 import BM25Retriever, tokenize
from app.retrieval.factory import get_retriever

DOCS = [
    ("c1", "FastAPI dependency injection uses Depends for shared resources.",
     {"source": "a.pdf", "page": 1}),
    ("c2", "ChromaDB stores embeddings on disk and supports metadata filters.",
     {"source": "a.pdf", "page": 2}),
    ("c3", "Error code E1042 means the request timed out.",
     {"source": "b.pdf", "page": 1}),
    ("c4", "A backup is a separate copy kept to recover from deletion.",
     {"source": "b.pdf", "page": 2}),
]


@pytest.fixture
def filled_store(store):
    store.add_documents(
        [Document(page_content=t, metadata={**m, "chunk_id": cid}) for cid, t, m in DOCS],
        ids=[cid for cid, _, _ in DOCS],
    )
    return store


def test_tokenize():
    assert tokenize("Hello, World_1 gemini-embedding-001") == [
        "hello", "world_1", "gemini", "embedding", "001"
    ]
    assert tokenize("What is the backup", remove_stopwords=True) == ["backup"]


def test_rare_exact_term_ranks_first(filled_store):
    results = BM25Retriever(filled_store).retrieve("E1042 timed out", k=3)
    assert results[0].chunk_id == "c3"
    assert results[0].rank == 1 and results[0].score > 0


def test_no_match_returns_nothing(filled_store):
    assert BM25Retriever(filled_store).retrieve("zzzzqqq", k=3) == []


def test_scores_descend_and_ranks_are_sequential(filled_store):
    results = BM25Retriever(filled_store).retrieve("request copy", k=5)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_metadata_filter(filled_store):
    retriever = BM25Retriever(filled_store)
    query = "stores shared embeddings resources"
    in_a = retriever.retrieve(query, k=5, filter={"source": "a.pdf"})
    assert in_a and all(r.metadata["source"] == "a.pdf" for r in in_a)
    assert retriever.retrieve(query, k=5, filter={"source": "b.pdf"}) == []


def test_index_refreshes_when_store_changes(filled_store):
    retriever = BM25Retriever(filled_store)
    assert retriever.retrieve("zebra", k=3) == []
    filled_store.add_documents(
        [Document(page_content="A zebra crossed the road.",
                  metadata={"source": "c.pdf", "page": 1, "chunk_id": "c5"})],
        ids=["c5"],
    )
    assert retriever.retrieve("zebra", k=3)[0].chunk_id == "c5"


def test_empty_store_returns_nothing(store):
    assert BM25Retriever(store).retrieve("anything", k=3) == []


def test_factory_builds_bm25(settings, store):
    s = settings.model_copy(update={"retrieval_method": "bm25"})
    assert isinstance(get_retriever(s, store), BM25Retriever)