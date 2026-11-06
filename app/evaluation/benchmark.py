"""Run one retriever over the labeled dataset and report retrieval metrics.

Usage:
    python -m app.evaluation.benchmark                  # uses RETRIEVAL_METHOD from .env
    python -m app.evaluation.benchmark --method dense --ks 3 5 10
"""
import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import get_settings
from app.embeddings.embedder import build_embeddings
from app.evaluation import metrics
from app.retrieval.base import Retriever
from app.retrieval.factory import get_retriever
from app.vectorstore.chroma import get_vectorstore

DEFAULT_DATASET = Path("app/evaluation/dataset.json")
RESULTS_DIR = Path("data/benchmarks")


class EvalItem(BaseModel):
    id: str
    type: str
    question: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)


def load_dataset(path: Path) -> list[EvalItem]:
    return [EvalItem(**row) for row in json.loads(path.read_text(encoding="utf-8"))]


def score_question(retrieved: list[str], relevant: set[str], ks: list[int]) -> dict[str, float]:
    scores = {"mrr": metrics.reciprocal_rank(retrieved, relevant)}
    for k in ks:
        scores[f"p@{k}"] = metrics.precision_at_k(retrieved, relevant, k)
        scores[f"r@{k}"] = metrics.recall_at_k(retrieved, relevant, k)
        scores[f"ndcg@{k}"] = metrics.ndcg_at_k(retrieved, relevant, k)
    return scores


def run_benchmark(retriever: Retriever, items: list[EvalItem], ks: list[int]) -> dict:
    depth = max(ks)
    per_question = []
    skipped = 0
    for item in items:
        if not item.relevant_chunk_ids:  # unanswerable: no retrieval metric applies
            skipped += 1
            continue
        start = time.perf_counter()
        chunks = retriever.retrieve(item.question, k=depth)
        latency_ms = (time.perf_counter() - start) * 1000
        retrieved = [c.chunk_id for c in chunks]
        relevant = set(item.relevant_chunk_ids)
        per_question.append(
            {
                "id": item.id,
                "type": item.type,
                "question": item.question,
                "latency_ms": latency_ms,
                "retrieved_ids": retrieved,
                "relevant_ids": sorted(relevant),
                "scores": score_question(retrieved, relevant, ks),
            }
        )
    return {
        "method": retriever.name,
        "ks": ks,
        "depth": depth,
        "skipped_unanswerable": skipped,
        "per_question": per_question,
    }


def summarize(result: dict) -> dict:
    rows = result["per_question"]
    metric_names = list(rows[0]["scores"]) if rows else []

    def avg(subset: list[dict]) -> dict[str, float]:
        return {m: statistics.fmean(r["scores"][m] for r in subset) for m in metric_names}

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    latencies = [r["latency_ms"] for r in rows]
    return {
        "n_questions": len(rows),
        "overall": avg(rows) if rows else {},
        "by_type": {t: {"n": len(rs), **avg(rs)} for t, rs in by_type.items()},
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
        },
    }


def print_report(result: dict, summary: dict) -> None:
    print(f"\nMethod: {result['method']} | questions scored: {summary['n_questions']} "
          f"| unanswerable skipped: {result['skipped_unanswerable']} "
          f"| retrieval depth: {result['depth']}")
    types = list(summary["by_type"])
    header = f"{'metric':<10}{'overall':>10}" + "".join(f"{t[:11]:>12}" for t in types)
    print("\n" + header)
    print("-" * len(header))
    for m in summary["overall"]:
        line = f"{m:<10}{summary['overall'][m]:>10.3f}"
        line += "".join(f"{summary['by_type'][t][m]:>12.3f}" for t in types)
        print(line)
    print("n".ljust(10) + str(summary["n_questions"]).rjust(10)
          + "".join(str(summary["by_type"][t]["n"]).rjust(12) for t in types))

    lat = summary["latency_ms"]
    print(f"\nLatency per query (includes query embedding): mean {lat['mean']:.0f} ms, "
          f"max {lat['max']:.0f} ms")

    misses = [r for r in result["per_question"] if r["scores"]["mrr"] == 0.0]
    print(f"\nQuestions with NO relevant chunk in the top {result['depth']}: {len(misses)}")
    for r in misses:
        print(f"  {r['id']} [{r['type']}] {r['question'][:70]}")


def save_result(result: dict, summary: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{result['method']}.json"
    path.write_text(json.dumps({**result, "summary": summary}, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ks", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--method", default=None, help="Override RETRIEVAL_METHOD")
    args = parser.parse_args()

    settings = get_settings()
    if args.method:
        settings = settings.model_copy(update={"retrieval_method": args.method})
    store = get_vectorstore(settings, build_embeddings(settings))
    retriever = get_retriever(settings, store)

    result = run_benchmark(retriever, load_dataset(args.dataset), sorted(args.ks))
    if not result["per_question"]:
        print("No labeled questions found. Fill relevant_chunk_ids in the dataset first.")
        return
    summary = summarize(result)
    print_report(result, summary)
    print(f"\nSaved: {save_result(result, summary)}")


if __name__ == "__main__":
    main()