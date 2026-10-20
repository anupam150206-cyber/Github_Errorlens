import re

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

texts, numbers = [], []
offset = None
while True:
    points, offset = client.scroll(
        COLLECTION, limit=2000, offset=offset,
        with_payload=True, with_vectors=False,
    )
    for p in points:
        texts.append(p.payload["page_content"])
        numbers.append(p.payload["metadata"]["issue_number"])
    if offset is None:
        break

def tokenize(text):
    return re.findall(r"\w+", text.lower())


bm25 = BM25Okapi([tokenize(t) for t in texts])
print("BM25 index built over", len(texts), "chunks")

def search_vector(query, top=50):
    chunks = vector_store.similarity_search(query, k=top)
    ranked = []
    for c in chunks:
        n = c.metadata["issue_number"]
        if n not in ranked:
            ranked.append(n)
    return ranked


def search_bm25(query, top=50):
    scores = bm25.get_scores(tokenize(query))
    best_chunks = scores.argsort()[::-1][:300]
    ranked = []
    for i in best_chunks:
        n = numbers[i]
        if n not in ranked:
            ranked.append(n)
    return ranked[:top]


def rrf(lists, k=60):
    """Reciprocal Rank Fusion: merge several ranked lists into one."""
    score = {}
    for one_list in lists:
        for rank, n in enumerate(one_list, start=1):
            score[n] = score.get(n, 0) + 1 / (k + rank)
    return sorted(score, key=score.get, reverse=True)


def search_hybrid(query):
    return rrf([search_vector(query), search_bm25(query)])

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" 
reranker = CrossEncoder(RERANK_MODEL, max_length=512)


def candidate_chunks(query, per_method=30):
    """Collect the best chunks from vector search and BM25.
    Returns a dict: chunk text -> issue number (the text also removes repeats)."""
    candidates = {}

    for doc in vector_store.similarity_search(query, k=per_method):
        candidates[doc.page_content] = doc.metadata["issue_number"]

    scores = bm25.get_scores(tokenize(query))
    for i in scores.argsort()[::-1][:per_method]:
        candidates[texts[i]] = numbers[i]

    return candidates


def search_rerank(query, per_method=30):
    candidates = candidate_chunks(query, per_method)
    chunk_texts = list(candidates)

    short_query = query[:600]      
    chunk_scores = reranker.predict([(short_query, t) for t in chunk_texts])


    issue_score = {}
    for text, s in zip(chunk_texts, chunk_scores):
        issue = candidates[text]
        issue_score[issue] = max(issue_score.get(issue, float("-inf")), float(s))

    return sorted(issue_score, key=issue_score.get, reverse=True)