import json
from pathlib import Path

from app.config import get_settings
from app.embeddings.embedder import build_embeddings
from app.vectorstore.chroma import get_vectorstore

DATASET = Path("app/evaluation/dataset.json")


def main() -> None:
    settings = get_settings()
    store = get_vectorstore(settings, build_embeddings(settings))
    stored_ids = set(store.get()["ids"])
    data = json.loads(DATASET.read_text(encoding="utf-8"))

    problems = 0
    answerable = 0
    seen_ids = set()
    for item in data:
        if item["id"] in seen_ids:
            print(f"DUPLICATE question id: {item['id']}")
            problems += 1
        seen_ids.add(item["id"])

        relevant = item["relevant_chunk_ids"]
        if relevant:
            answerable += 1
        elif item["type"] != "unanswerable":
            print(f"{item['id']}: no labels but type is '{item['type']}'")
            problems += 1
        for cid in relevant:
            if cid not in stored_ids:
                print(f"{item['id']}: chunk id not found in store: {cid}")
                problems += 1

    print(f"\n{len(data)} questions, {answerable} with labels, {problems} problem(s)")


if __name__ == "__main__":
    main()