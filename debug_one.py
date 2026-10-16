import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)
client = QdrantClient(path="data/qdrant")
vs = QdrantVectorStore(client=client, collection_name="issues", embedding=embeddings)

tests = json.load(open("eval_data/positives.json", encoding="utf-8"))
t = next(x for x in tests if x["query_issue"] == 31348)
answer = t["answer_issue"]

# CHECK 1: does the index contain chunks of the answer issue?
points, _ = client.scroll(
    "issues",
    scroll_filter=models.Filter(must=[models.FieldCondition(
        key="metadata.issue_number", match=models.MatchValue(value=answer))]),
    limit=100,
)
print("chunks of the answer issue in the index:", len(points))
for p in points[:3]:
    print("   ", p.payload["page_content"][:100].replace("\n", " "))

# CHECK 2: what does the query look like, and what does the search return?
print("\nQUERY TEXT (first 300 chars):\n", t["query_text"][:300])
print("\nSearch with the full query text:")
for doc, score in vs.similarity_search_with_score(t["query_text"], k=5):
    print(" ", round(score, 3), "| issue", doc.metadata["issue_number"], "|",
          doc.page_content[:70].replace("\n", " "))

# CHECK 3: search with the title only
title = t["query_text"].split("\n")[0]
print("\nSearch with the title only:", title)
for doc, score in vs.similarity_search_with_score(title, k=5):
    print(" ", round(score, 3), "| issue", doc.metadata["issue_number"], "|",
          doc.page_content[:70].replace("\n", " "))