from app.config import get_settings
from app.embeddings.embedder import build_embeddings
from app.ingestion.pipeline import ingest_file
from app.vectorstore.chroma import get_vectorstore


def main() -> None:
    settings = get_settings()
    store = get_vectorstore(settings, build_embeddings(settings))

    pdfs = sorted(settings.documents_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {settings.documents_dir.resolve()}")
        return

    print("chunks before:", len(store.get()["ids"]))
    for pdf in pdfs:
        removed, added = ingest_file(pdf, store, settings)
        print(f"{pdf.name}: removed {removed}, added {added}")
    print("chunks after:", len(store.get()["ids"]))

    sample = store.get(limit=1, include=["documents", "metadatas", "embeddings"])
    print("\nsample metadata:", sample["metadatas"][0])
    print("vector length:", len(sample["embeddings"][0]))


if __name__ == "__main__":
    main()