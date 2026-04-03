# spacy-nlp

Two spaCy use-cases built around a shared idea: maximum likelihood estimation connects a 50-line word counter to GPT-4. Both projects sit on that thread.

## Use Cases

### summarization

Extractive summarization using spaCy keyword frequencies. The algorithm scores sentences by the normalized frequency of their content words and returns the top-scoring ones. No model weights, no GPU — just counting and ranking. Works well on thematically repetitive text: news articles, research papers, earnings calls, regulatory filings.

```bash
uv run python summarization/spacy_summarization.py
# or interactive app
uv run streamlit run summarization/spacy_summarize_app.py
```

### named-entity-recognition

Brand extraction from e-commerce product titles using spaCy's pre-trained NER model (`en_core_web_sm`), with a regex fallback for common "Brand Product" title patterns. Processes thousands of records per second on CPU with no manual annotation.

```bash
uv run python -m spacy download en_core_web_sm
uv run python named-entity-recognition/scripts/brand_extractor.py
```

## Background

The article in `docs/` — *From Word Counts to World Models: MLE Is the Thread Connecting spaCy to GPT* — traces the statistical connection between these two use-cases and large language models. Both the summarizer and the NER model are trained with maximum likelihood estimation; the difference is only in what they condition on and how many parameters they use.
