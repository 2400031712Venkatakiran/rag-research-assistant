import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def make_chunk_id(source: str, page: int, chunk_index: int, content: str) -> str:
    raw = f"{source}|{page}|{chunk_index}|{content}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def chunk_documents(
    docs: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_id"] = make_chunk_id(
            chunk.metadata["source"], chunk.metadata["page"], i, chunk.page_content
        )
    return chunks