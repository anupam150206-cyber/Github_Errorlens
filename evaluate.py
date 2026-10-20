import json
from datetime import date
from pathlib import Path
from rag_engine.search import search_bm25, search_hybrid, search_rerank, search_vector

METHODS = {
    "Baseline: MiniLM, vector search": search_vector,
    "BM25 keyword search only": search_bm25,
    "Hybrid: vector + BM25 (RRF)": search_hybrid,
    "Hybrid + rerank (MiniLM cross-encoder)": search_rerank,
}

with open("eval_data/positives.json", encoding="utf-8") as f:
    tests = json.load(f)

results_file = Path("results/results.md")
results_file.parent.mkdir(exist_ok=True)
if not results_file.exists():
    results_file.write_text(
        "| Config | Test cases | Hit@1 | Hit@5 | MRR | Date |\n|---|---|---|---|---|---|\n",
        encoding="utf-8",
    )


def score(search_function):
    ranks = []
    for test in tests:
        results = search_function(test["query_text"])
        answer = test["answer_issue"]
        ranks.append(results.index(answer) + 1 if answer in results else None)

    hit1 = sum(1 for r in ranks if r == 1)
    hit5 = sum(1 for r in ranks if r and r <= 5)
    mrr = sum(1 / r for r in ranks if r) / len(tests)
    return hit1, hit5, mrr, ranks


with results_file.open("a", encoding="utf-8") as f:
    for name, fn in METHODS.items():
        hit1, hit5, mrr, ranks = score(fn)
        n = len(tests)
        print(f"\n{name}\n  Hit@1 {hit1}/{n} | Hit@5 {hit5}/{n} | MRR {mrr:.2f}\n  ranks: {ranks}")
        f.write(f"| {name} | {n} | {hit1}/{n} | {hit5}/{n} | {mrr:.2f} | {date.today()} |\n")

print(f"\nSaved rows to {results_file}")