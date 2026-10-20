import sys
from pathlib import Path

import streamlit as st

# lets Streamlit find the rag_engine folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="GitHub ErrorLens", page_icon="🔎")


@st.cache_resource(show_spinner="Loading models and index (1 to 3 minutes the first time)...")
def load_everything():
    from rag_engine import search
    from rag_engine.issues import load_issue_info

    methods = {
        "Hybrid (vector + keyword)": search.search_hybrid,
        "Vector only": search.search_vector,
        "Keyword only (BM25)": search.search_bm25,
        "Hybrid + rerank": search.search_rerank,
    }
    return methods, load_issue_info()


methods, issues = load_everything()

st.title("🔎 Has this error happened before?")
st.caption("Searches 6,000 closed issues from langchain-ai/langchain")

error_text = st.text_area("Paste your error or traceback", height=220)
method = st.selectbox("Search method", list(methods))
top_k = st.slider("Results to show", 1, 10, 5)

if st.button("Search", type="primary") and error_text.strip():
    with st.spinner("Searching..."):
        numbers = methods[method](error_text)[:top_k]

        st.subheader(f"Top {len(numbers)} matches")
    for rank, n in enumerate(numbers, start=1):
        info = issues.get(n)
        if not info:
            continue
        st.markdown(f"**{rank}. [#{n} {info['title']}]({info['url']})**")

        closed = (info.get("closed_as") or "unknown").replace("_", " ")
        st.caption(f"Closed as: {closed}")

        if info["fix"]:
            with st.expander("Maintainer reply (likely fix)"):
                st.write(info["fix"][:2000])
        elif info["last_comment"]:
            with st.expander("Last comment (not from a maintainer)"):
                st.write(info["last_comment"][:2000])
        else:
            st.caption("No comments on this issue.")