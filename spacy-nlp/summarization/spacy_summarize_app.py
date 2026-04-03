"""Interactive spaCy summarization: run from repo root with dev deps:

uv sync --group dev
uv run --group dev streamlit run text-summarization/spacy_summarize_app.py
"""

from __future__ import annotations

import spacy
import streamlit as st
from spacy_summarize_core import summarize_text


@st.cache_resource(show_spinner="Loading en_core_web_sm…")
def load_nlp():
    return spacy.load("en_core_web_sm")


st.set_page_config(page_title="spaCy summarization", layout="centered")
st.title("spaCy extractive summarization")
st.caption(
    "Keyword frequency over sentences (en_core_web_sm). Not an abstractive LLM summary."
)

nlp = load_nlp()

with st.sidebar:
    n_summary = st.slider("Sentences in summary", min_value=1, max_value=10, value=3)
    n_keywords = st.slider("Top keywords to show", min_value=3, max_value=15, value=5)

default_demo = """The field of AI has experienced dramatic cycles of boom and bust.
There were moments when the term became so damaged that researchers avoided using it entirely.
Generative AI is transforming how we work, conduct research, and make decisions."""

with st.form("summarize_form", clear_on_submit=False):
    text = st.text_area(
        "Your text", value=default_demo, height=220, placeholder="Paste or type text…"
    )
    submitted = st.form_submit_button("Summarize", type="primary")

if submitted:
    result = summarize_text(
        text,
        nlp,
        summary_sentence_count=n_summary,
        top_keyword_count=n_keywords,
    )
    if result.error:
        st.warning(result.error)
    else:
        st.success(f"Detected **{result.num_sentences}** sentence(s).")
        st.subheader("Summary")
        st.write(result.summary or "_(empty)_")

        if result.top_keywords:
            st.subheader("Top keywords")
            st.table([{"keyword": w, "count": c} for w, c in result.top_keywords])

        if result.sentence_scores:
            st.subheader("Sentence scores (higher = more keyword overlap)")
            st.dataframe(
                [
                    {
                        "score": round(s, 3),
                        "sentence": (f"{sent[:200]}…" if len(sent) > 200 else sent),
                    }
                    for sent, s in result.sentence_scores
                ]
            )
