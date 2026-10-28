import pymupdf
import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(
        chroma_dir=tmp_path / "chroma",
        documents_dir=tmp_path / "docs",
        chunk_size=200,
        chunk_overlap=20,
        top_k=3,

    )


@pytest.fixture
def store(settings):
    # Fake embeddings: no API key or network needed. Identical text -> identical vector.
    return Chroma(
        collection_name="test",
        embedding_function=DeterministicFakeEmbedding(size=32),
        persist_directory=str(settings.chroma_dir),
    )


@pytest.fixture
def sample_pdf(settings):
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    path = settings.documents_dir / "sample.pdf"
    doc = pymupdf.open()
    for text in ["FastAPI dependency injection uses Depends. " * 8,
                 "ChromaDB stores embeddings on disk. " * 8]:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 520, 760), text, fontsize=11)
    doc.save(path)
    return path
