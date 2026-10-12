from langchain_community.document_loaders import JSONLoader


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

print(len(docs), "documents")
print(docs[0].page_content[:300])
print({k: v for k, v in docs[0].metadata.items() if k != "comments"})