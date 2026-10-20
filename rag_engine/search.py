from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from rag_engine.config import EMBED_MODEL

COLLECTION = "issues"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)
client = QdrantClient(path="data/qdrant")
vector_store = QdrantVectorStore(
    client=client, collection_name=COLLECTION, embedding=embeddings
)

texts, issue_numbers = [], []
offset = None
while True:
    points, offset = client.scroll(
        COLLECTION, limit=2000, offset=offset, with_payload=True, with_vectors=False
    )
    texts.extend(p.payload["page_content"] for p in points)
    issue_numbers.extend(p.payload["metadata"]["issue_number"] for p in points)
    if offset is None:
        break


def tokenize(text):
    return "".join(c if c.isalnum() or c == "_" else " " for c in text.lower()).split()


bm25 = BM25Okapi([tokenize(t) for t in texts])


def search_vector(query, top=50):
    chunks = vector_store.similarity_search(query, k=top)
    return list(dict.fromkeys(c.metadata["issue_number"] for c in chunks))


def search_bm25(query, top=50):
    scores = bm25.get_scores(tokenize(query))
    top_indices = scores.argsort()[::-1][:300]
    unique_issues = list(dict.fromkeys(issue_numbers[i] for i in top_indices))
    return unique_issues[:top]


def rrf(ranked_lists, k=60):
    scores = {}
    for items in ranked_lists:
        for rank, item in enumerate(items, start=1):
            scores[item] = scores.get(item, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def search_hybrid(query):
    return rrf([search_vector(query), search_bm25(query)])


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)


def candidate_chunks(query, per_method=30):
    candidates = {
        doc.page_content: doc.metadata["issue_number"]
        for doc in vector_store.similarity_search(query, k=per_method)
    }
    scores = bm25.get_scores(tokenize(query))
    for i in scores.argsort()[::-1][:per_method]:
        candidates[texts[i]] = issue_numbers[i]
    return candidates

def search_rerank(query, per_method=30):
    candidates = candidate_chunks(query, per_method)
    chunk_texts = list(candidates)

    scores = reranker.predict([(query[:600], text) for text in chunk_texts])

    issue_scores = {}
    for text, score in zip(chunk_texts, scores):
        issue = candidates[text]
        issue_scores[issue] = max(issue_scores.get(issue, float("-inf")), float(score))

    return sorted(issue_scores, key=issue_scores.get, reverse=True)