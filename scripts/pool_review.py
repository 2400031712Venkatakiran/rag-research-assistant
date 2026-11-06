"""List chunks that any method ranked near the top but that are NOT labeled relevant.

Use after benchmarking each method, to find missing labels (pooling bias).
Usage: python -m scripts.pool_review dense bm25 [--top 3]
"""
import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.embeddings.embedder import build_embeddings
from app.vectorstore.chroma import get_vectorstore

RESULTS_DIR = Path("data/benchmarks")
DATASET = Path("app/evaluation/dataset.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("methods", nargs="+")
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    settings = get_settings()
    store = get_vectorstore(settings, build_embeddings(settings))
    data = store.get(include=["documents", "metadatas"])
    chunks = {i: (d, m) for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])}

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    results = {}
    for method in args.methods:
        saved = json.loads((RESULTS_DIR / f"{method}.json").read_text(encoding="utf-8"))
        results[method] = {q["id"]: q for q in saved["per_question"]}

    total = 0
    for item in dataset:
        relevant = set(item["relevant_chunk_ids"])
        if not relevant:
            continue
        candidates: dict[str, list[str]] = {}
        for method in args.methods:
            q = results[method].get(item["id"])
            if not q:
                continue
            for rank, chunk_id in enumerate(q["retrieved_ids"][: args.top], start=1):
                if chunk_id not in relevant:
                    candidates.setdefault(chunk_id, []).append(f"{method}#{rank}")
        if not candidates:
            continue
        print("=" * 90)
        print(f"{item['id']} [{item['type']}] {item['question']}")
        print(f"labeled relevant: {sorted(relevant)}")
        for chunk_id, who in candidates.items():
            total += 1
            text, meta = chunks[chunk_id]
            print(f"\n  UNLABELED {chunk_id}  p.{meta['page']}  ranked by: {', '.join(who)}")
            print("  " + text.replace("\n", " "))
    print(f"\n{total} unlabeled chunk(s) to review")


if __name__ == "__main__":
    main()