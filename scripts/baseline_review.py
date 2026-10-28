import json
from pathlib import Path

from app.config import get_settings
from app.embeddings.embedder import build_embeddings
from app.retrieval.factory import get_retriever
from app.vectorstore.chroma import get_vectorstore

QUESTIONS = Path("app/evaluation/questions.json")
OUT = Path("data/baseline_review.md")
K = 10


def main() -> None:
    settings = get_settings()
    retriever = get_retriever(settings, get_vectorstore(settings, build_embeddings(settings)))
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))

    lines = [f"# Baseline review (method={retriever.name}, k={K})\n"]
    for q in questions:
        chunks = retriever.retrieve(q["question"], k=K)
        lines.append(f"\n## {q['id']} [{q['type']}] {q['question']}\n")
        for c in chunks:
            preview = c.content[:300].replace("\n", " ")
            lines.append(
                f"- **#{c.rank}** `{c.chunk_id}` p.{c.metadata['page']} "
                f"dist={c.score:.3f}\n  {preview}\n"
            )

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT} ({len(questions)} questions)")


if __name__ == "__main__":
    main()