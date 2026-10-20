# GitHub ErrorLens - Architecture

Horizontal architecture diagram showing each block of the RAG system across ingestion, hybrid retrieval, and presentation pipelines.

```mermaid
flowchart LR
    subgraph INGEST["1. Ingestion & Preprocessing"]
        A["GitHub API: Issues & Comments"] --> B["Clean & Chunk Documents"]
        B --> C1[("Qdrant Vector DB (MiniLM)")]
        B --> C2[("BM25 Keyword Index")]
        B --> C3[("Issue & Fix Metadata Store")]
    end

    subgraph SEARCH["2. Hybrid Retrieval & Reranking"]
        Q["User Query: Error / Traceback"] --> D1["Dense Vector Search"]
        Q --> D2["Sparse BM25 Search"]
        C1 -.-> D1
        C2 -.-> D2
        D1 --> E{"RRF Fusion / Cross-Encoder Reranker"}
        D2 --> E
    end

    subgraph UI["3. Resolution & UI"]
        E --> F["Maintainer Fix Lookup"]
        C3 -.-> F
        F --> G["Streamlit App UI"]
    end
```

---

## Core Blocks Overview

### 1. Ingestion & Preprocessing
- **Data Fetcher** (`scripts/fetch_issues.py`): Collects closed issues and discussion comments via the GitHub REST API.
- **Cleaner & Splitter** (`main.py`): Strips issue templates/checklists (`clean`) and chunks text with `RecursiveCharacterTextSplitter`.
- **Dual Indexing & Storage**:
  - Dense embeddings via `all-MiniLM-L6-v2` stored in Qdrant (`data/qdrant`).
  - Sparse keyword index built in-memory with `BM25Okapi` (`rag_engine/search.py`).
  - Metadata registry loaded by `load_issue_info` (`rag_engine/issues.py`).

### 2. Hybrid Retrieval & Reranking
- **Dense Search** (`search_vector`): Retrieves semantic context from Qdrant.
- **Sparse Search** (`search_bm25`): Captures exact token/traceback matches.
- **Fusion & Reranking**: Merges candidates using Reciprocal Rank Fusion via `rrf` or scores candidate chunks with `search_rerank` using `cross-encoder/ms-marco-MiniLM-L-6-v2`.

### 3. Resolution & UI
- **Fix Resolution** (`rag_engine/issues.py`): Maps top-ranked issue IDs back to author/maintainer solutions.
- **Web Interface** (`app/streamlit_app.py`): Interactive search interface displaying ranked issues, state reasons, and verified fixes.
