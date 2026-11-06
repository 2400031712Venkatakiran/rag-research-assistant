"""Compare saved benchmark results side by side.

Usage: python -m app.evaluation.compare dense bm25
Reads data/benchmarks/<name>.json for each name (created by the benchmark).
"""
import json
import sys
from pathlib import Path

RESULTS_DIR = Path("data/benchmarks")


def load(name: str) -> dict:
    return json.loads((RESULTS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def first_rank(question: dict) -> int | None:
    relevant = set(question["relevant_ids"])
    for rank, chunk_id in enumerate(question["retrieved_ids"], start=1):
        if chunk_id in relevant:
            return rank
    return None


def main(names: list[str]) -> None:
    results = {n: load(n) for n in names}
    base = names[0]
    two = len(names) == 2

    header = f"{'metric':<10}" + "".join(f"{n[:14]:>16}" for n in names)
    print("\n" + header + (f"{'delta':>10}" if two else ""))
    print("-" * len(header))
    for metric in results[base]["summary"]["overall"]:
        values = [results[n]["summary"]["overall"][metric] for n in names]
        line = f"{metric:<10}" + "".join(f"{v:>16.3f}" for v in values)
        if two:
            line += f"{values[1] - values[0]:>+10.3f}"
        print(line)

    lat = [results[n]["summary"]["latency_ms"]["mean"] for n in names]
    print(f"{'latency ms':<10}" + "".join(f"{v:>16.0f}" for v in lat))

    by_id = {n: {q["id"]: q for q in results[n]["per_question"]} for n in names}
    print("\nQuestions where the first relevant rank differs (None = not in top N):")
    shown = 0
    for qid, q in by_id[base].items():
        ranks = [first_rank(by_id[n][qid]) if qid in by_id[n] else "-" for n in names]
        if len(set(ranks)) > 1:
            shown += 1
            cells = "  ".join(f"{n}:{r}" for n, r in zip(names, ranks))
            print(f"  {qid} [{q['type']}] {cells}  | {q['question'][:55]}")
    if not shown:
        print("  none")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Usage: python -m app.evaluation.compare <method_a> <method_b> [...]")
    main(sys.argv[1:])