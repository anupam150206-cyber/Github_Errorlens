import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from rag_engine.config import EMBED_MODEL

# 1. open the saved index
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)
client = QdrantClient(path="data/qdrant")
print("points in index:", client.count("issues").count)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="issues",
    embedding=embeddings,
)

# 2. load the test queries
tests = json.load(open("eval_data/positives.json", encoding="utf-8"))


# 3. search: returns issue numbers, best first, one entry per issue
def search_issues(query, top_chunks=50):
    results = vector_store.similarity_search(query, k=top_chunks)
    ranked = []
    for doc in results:
        number = doc.metadata["issue_number"]
        if number not in ranked:
            ranked.append(number)
    return ranked


# 4. score
hit1 = hit5 = 0
reciprocal_ranks = []

for t in tests:
    ranked = search_issues(t["query_text"])
    answer = t["answer_issue"]

    if answer in ranked:
        rank = ranked.index(answer) + 1
        reciprocal_ranks.append(1 / rank)
        if rank == 1:
            hit1 += 1
        if rank <= 5:
            hit5 += 1
    else:
        rank = None
        reciprocal_ranks.append(0)

    print(f"#{t['query_issue']} -> expected #{answer} | found at rank: {rank}")

n = len(tests)
print(f"\nTest cases: {n}")
print(f"Hit@1: {hit1}/{n} = {hit1 / n:.2f}")
print(f"Hit@5: {hit5}/{n} = {hit5 / n:.2f}")
print(f"MRR:   {sum(reciprocal_ranks) / n:.2f}")