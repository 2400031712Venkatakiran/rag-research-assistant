import pytest

from app.ingestion.pipeline import ingest_file
from app.retrieval.dense import DenseRetriever
from app.retrieval.factory import get_retriever


def test_factory_returns_dense(settings, store):
    assert isinstance(get_retriever(settings, store), DenseRetriever)


def test_factory_rejects_unknown_method(settings, store):
    bad = settings.model_copy(update={"retrieval_method": "nope"})
    with pytest.raises(ValueError, match="Unknown RETRIEVAL_METHOD"):
        get_retriever(bad, store)


def test_dense_ranks_and_top_k(sample_pdf, store, settings):
    ingest_file(sample_pdf, store, settings)
    retriever = DenseRetriever(store)
    target = store.get()["documents"][0]
    results = retriever.retrieve(target, k=3)
    assert len(results) == 3
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].content == target  # identical text -> distance ~0
    assert results[0].score <= results[1].score


def test_dense_metadata_filter(sample_pdf, store, settings):
    ingest_file(sample_pdf, store, settings)
    retriever = DenseRetriever(store)
    results = retriever.retrieve("anything", k=5, filter={"page": 2})
    assert results and all(r.metadata["page"] == 2 for r in results)
