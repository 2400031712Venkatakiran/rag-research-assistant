from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.generation.answer import AnswerGenerator, format_context
from app.main import app, get_generator, get_retriever_dep
from app.retrieval.base import RetrievedChunk, Retriever


def _chunk(rank=1):
    return RetrievedChunk(
        chunk_id=f"id{rank}", content="Depends injects things.",
        metadata={"source": "a.pdf", "page": 4}, rank=rank, score=0.1,
    )


class StubRetriever(Retriever):
    name = "stub"

    def retrieve(self, query, k, filter=None):
        return [_chunk(1), _chunk(2)][:k]


def _fake_llm(text="Use Depends [1]."):
    return GenericFakeChatModel(messages=iter([AIMessage(content=text)]))


def test_context_is_numbered_with_citations():
    ctx = format_context([_chunk(1)])
    assert ctx.startswith("[1] (source: a.pdf, page 4)")


def test_generator_returns_sources_and_handles_empty():
    gen = AnswerGenerator(_fake_llm())
    out = gen.generate("q", [_chunk(1)])
    assert out.answer == "Use Depends [1]."
    assert out.sources[0].source == "a.pdf" and out.sources[0].page == 4
    assert "could not find" in gen.generate("q", []).answer


def test_query_endpoint_with_swapped_components():
    app.dependency_overrides[get_retriever_dep] = lambda: StubRetriever()
    app.dependency_overrides[get_generator] = lambda: AnswerGenerator(_fake_llm())
    try:
        client = TestClient(app)
        r = client.post("/query", json={"question": "how?", "top_k": 1})
        assert r.status_code == 200
        body = r.json()
        assert body["method"] == "stub" and len(body["retrieved"]) == 1
        r2 = client.post("/retrieve", json={"question": "how?"})
        assert r2.status_code == 200
        assert client.post("/retrieve", json={"question": ""}).status_code == 422
    finally:
        app.dependency_overrides.clear()
