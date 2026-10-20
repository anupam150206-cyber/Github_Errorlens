import json
from datetime import date
from pathlib import Path
from rag_engine.search import search_bm25, search_hybrid, search_vector
from rag_engine.search import search_bm25, search_hybrid, search_rerank, search_vector

METHODS = {
    "Hybrid + rerank (MiniLM cross-encoder)": search_rerank,
}


tests = json.load(open("eval_data/positives.json", encoding="utf-8"))

METHODS = {
    "Baseline: MiniLM, vector search": search_vector,
    "BM25 keyword search only": search_bm25,
    "Hybrid: vector + BM25 (RRF)": search_hybrid,
}

results_file = Path("results/results.md")
results_file.parent.mkdir(exist_ok=True)
if not results_file.exists():
    results_file.write_text(
        "| Config | Test cases | Hit@1 | Hit@5 | MRR | Date |\n"
        "|---|---|---|---|---|---|\n",
        encoding="utf-8",
    )


def score(search_function):
    hit1 = hit5 = 0
    rr = []
    ranks = []
    for t in tests:
        found = search_function(t["query_text"])
        answer = t["answer_issue"]
        if answer in found:
            rank = found.index(answer) + 1
            rr.append(1 / rank)
            hit1 += rank == 1
            hit5 += rank <= 5
        else:
            rank = None
            rr.append(0)
        ranks.append(rank)
    return hit1, hit5, sum(rr) / len(tests), ranks


for name, fn in METHODS.items():
    hit1, hit5, mrr, ranks = score(fn)
    n = len(tests)
    print(f"\n{name}")
    print(f"  Hit@1 {hit1}/{n} | Hit@5 {hit5}/{n} | MRR {mrr:.2f}")
    print("  ranks:", ranks)

    with results_file.open("a", encoding="utf-8") as f:
        f.write(f"| {name} | {n} | {hit1}/{n} | {hit5}/{n} | {mrr:.2f} | {date.today()} |\n")

print("\nSaved rows to", results_file)