from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace
from langchain_community import document_loaders
from langchain_community.document_loaders import JSONLoader

load_dotenv()

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["issue_number"] = record["number"]
    metadata["title"] = record["title"]
    metadata["labels"] = record["labels"]
    metadata["closed_at"] = record["closed_at"]
    metadata["url"] = record["html_url"]
    metadata["comments"] = record["comments"]
    return metadata


loader = JSONLoader(
    file_path="data/raw/github_issues.jsonl",
    jq_schema=".",
    content_key="body",
    metadata_func=metadata_func,
    json_lines=True,
)

docs = loader.load()

