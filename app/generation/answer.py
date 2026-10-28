"""Answer generation. Knows nothing about how chunks were retrieved."""
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.config import Settings
from app.retrieval.base import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a research assistant answering from technical documentation. "
    "Use ONLY the numbered context passages below. Cite passages inline like [1] or [2][3]. "
    "If the context does not contain the answer, say you could not find it in the documents. "
    "Do not invent information."
)

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "Context:\n{context}\n\nQuestion: {question}")]
)


class Source(BaseModel):
    n: int
    source: str | None = None
    page: int | None = None
    chunk_id: str


class GeneratedAnswer(BaseModel):
    answer: str
    sources: list[Source]


def build_llm(settings: Settings) -> BaseChatModel:
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        settings.llm_model, model_provider=settings.llm_provider, temperature=0
    )


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{c.rank}] (source: {c.metadata.get('source')}, page {c.metadata.get('page')})\n{c.content}"
        for c in chunks
    )


class AnswerGenerator:
    def __init__(self, llm: BaseChatModel):
        self._chain = PROMPT | llm | StrOutputParser()

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        if not chunks:
            return GeneratedAnswer(
                answer="I could not find relevant passages in the documents.", sources=[]
            )
        answer = self._chain.invoke(
            {"context": format_context(chunks), "question": question}
        )
        sources = [
            Source(
                n=c.rank,
                source=c.metadata.get("source"),
                page=c.metadata.get("page"),
                chunk_id=c.chunk_id,
            )
            for c in chunks
        ]
        return GeneratedAnswer(answer=answer, sources=sources)
