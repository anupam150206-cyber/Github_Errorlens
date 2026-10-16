from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace
from langchain_community import document_loaders
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

def metadata_func(record: dict, metadata: dict) -> dict:
    """Called once per issue. Copies the fields you want into Document.metadata."""
    metadata["issue_number"] = record["number"]
    metadata["title"] = record["title"]
    metadata["labels"] = record["labels"]
    metadata["closed_at"] = record["closed_at"]
    metadata["url"] = record["html_url"]
    metadata["comments"] = record["comments"]  # kept whole for the splitting step later
    return metadata


loader = JSONLoader(
    file_path="data/raw/github_issues.jsonl",
    jq_schema=".",
    content_key="body",
    metadata_func=metadata_func,
    json_lines=True,
)

docs = loader.load()


def clean(text):
    kept = []
    for line in (text or "").split("\n"):
        if line.strip().startswith(("- [x]", "- [X]", "- [ ]")):
            continue
        if line.strip().startswith("### Submission checklist"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()

for doc in docs:
    doc.page_content = clean(doc.page_content)
    for comment in doc.metadata["comments"]:
        comment["body"] = clean(comment["body"])

all_docs = []
for doc in docs:
    title = doc.metadata["title"]
    info = {"issue_number": doc.metadata["issue_number"], "url": doc.metadata["url"]}

    all_docs.append(Document(
        page_content=title + "\n\n" + doc.page_content,
        metadata={**info, "kind": "report"},
    ))

    for c in doc.metadata["comments"]:
        all_docs.append(Document(
            page_content=title + "\n\n" + c["body"],
            metadata={**info, "kind": "comment"},
        ))



splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    separators=["\n```\n", "\n\n", "\n", " "],
)

chunks = splitter.split_documents(all_docs)

print(len(docs), "issues ->", len(chunks), "chunks")
print(chunks[0].page_content[:300])


positives = json.load(open("eval_data/positives.json", encoding="utf-8"))
query_issues = {p["query_issue"] for p in positives}
index_chunks = [c for c in chunks if c.metadata["issue_number"] not in query_issues]
print(len(index_chunks), "chunks to index")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = QdrantVectorStore.from_documents(
    index_chunks,
    embeddings,
    path="data/qdrant",
    collection_name="issues",
)
print("Done. Index saved in data/qdrant")

