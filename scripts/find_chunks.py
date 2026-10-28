import sys

from app.config import get_settings
from app.embeddings.embedder import build_embeddings
from app.vectorstore.chroma import get_vectorstore


def main(keyword: str) -> None:
    settings = get_settings()
    store = get_vectorstore(settings, build_embeddings(settings))
    data = store.get(include=["documents", "metadatas"])

    kw = keyword.lower()
    hits = 0
    for chunk_id, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        text = doc.lower()
        if kw in text:
            hits += 1
            i = text.index(kw)
            snippet = doc[max(0, i - 80): i + 120].replace("\n", " ")
            print(f"{chunk_id}  p.{meta['page']}  ...{snippet}...")
    print(f"\n{hits} chunk(s) contain '{keyword}'")


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]))