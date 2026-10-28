import logging
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf(path: Path) -> list[Document]:
    raw = PyMuPDFLoader(str(path)).load()
    docs: list[Document] = []
    for d in raw:
        if not d.page_content.strip():
            continue
        docs.append(
            Document(
                page_content=d.page_content,
                metadata={
                    "source": path.name,
                    "page": int(d.metadata.get("page", 0)) + 1,
                    "total_pages": int(d.metadata.get("total_pages", 0)),
                },
            )
        )
    logger.info("Loaded %s: %d non-empty pages", path.name, len(docs))
    return docs