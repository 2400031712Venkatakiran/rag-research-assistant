from app.ingestion.chunker import chunk_documents, make_chunk_id
from app.ingestion.loader import load_pdf
from app.ingestion.pipeline import ingest_file


def test_chunk_id_is_deterministic():
    assert make_chunk_id("a.pdf", 1, 0, "x") == make_chunk_id("a.pdf", 1, 0, "x")
    assert make_chunk_id("a.pdf", 1, 0, "x") != make_chunk_id("a.pdf", 1, 0, "y")


def test_loader_normalises_metadata(sample_pdf):
    docs = load_pdf(sample_pdf)
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert docs[0].metadata["source"] == "sample.pdf"


def test_chunks_have_ids_and_indexes(sample_pdf, settings):
    chunks = chunk_documents(load_pdf(sample_pdf), settings.chunk_size, settings.chunk_overlap)
    assert len(chunks) > 2
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)


def test_reingest_does_not_duplicate(sample_pdf, store, settings):
    _, added = ingest_file(sample_pdf, store, settings)
    count_first = len(store.get()["ids"])
    removed, added_again = ingest_file(sample_pdf, store, settings)
    assert count_first == added == added_again == removed
    assert len(store.get()["ids"]) == count_first