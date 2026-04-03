"""Extractive summarization using spaCy keyword frequencies (same algorithm as the CLI demo)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from heapq import nlargest
from string import punctuation

from spacy.lang.en.stop_words import STOP_WORDS

POS_TAGS = frozenset({"PROPN", "ADJ", "NOUN", "VERB"})


@dataclass
class SummarizeResult:
    summary: str
    num_sentences: int
    top_keywords: list[tuple[str, int]]
    sentence_scores: list[tuple[str, float]]
    error: str | None = None


def summarize_text(
    text: str,
    nlp,
    *,
    summary_sentence_count: int = 3,
    top_keyword_count: int = 5,
) -> SummarizeResult:
    text = (text or "").strip()
    if not text:
        return SummarizeResult(
            summary="",
            num_sentences=0,
            top_keywords=[],
            sentence_scores=[],
            error="Enter some text to summarize.",
        )

    doc = nlp(text)
    sents = list(doc.sents)
    if not sents:
        return SummarizeResult(
            summary=text,
            num_sentences=0,
            top_keywords=[],
            sentence_scores=[],
            error="Could not detect any sentences in the text.",
        )

    stopwords = STOP_WORDS
    keyword: list[str] = []
    for token in doc:
        if token.text in stopwords or token.text in punctuation:
            continue
        if token.pos_ in POS_TAGS:
            keyword.append(token.text)

    if not keyword:
        picked = sents[:summary_sentence_count]
        summary = " ".join(s.text.strip().strip('"').strip() for s in picked)
        previews = [(s.text.strip().strip('"').strip(), 0.0) for s in sents]
        return SummarizeResult(
            summary=summary,
            num_sentences=len(sents),
            top_keywords=[],
            sentence_scores=previews,
            error=None,
        )

    freq_word = Counter(keyword)
    top_keywords = freq_word.most_common(top_keyword_count)
    max_freq = freq_word.most_common(1)[0][1]
    for word in freq_word:
        freq_word[word] = freq_word[word] / max_freq

    sent_strength: dict = {}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word:
                sent_strength[sent] = sent_strength.get(sent, 0) + freq_word[word.text]

    if not sent_strength:
        picked = sents[:summary_sentence_count]
        summary = " ".join(s.text.strip().strip('"').strip() for s in picked)
        previews = [(s.text.strip().strip('"').strip(), 0.0) for s in sents]
        return SummarizeResult(
            summary=summary,
            num_sentences=len(sents),
            top_keywords=top_keywords,
            sentence_scores=previews,
            error=None,
        )

    top_sents = nlargest(
        min(summary_sentence_count, len(sent_strength)),
        sent_strength,
        key=sent_strength.get,
    )
    final = [s.text.strip().strip('"').strip() for s in top_sents]
    summary = " ".join(final)

    sentence_scores = sorted(
        ((s.text.strip().strip('"').strip(), float(score)) for s, score in sent_strength.items()),
        key=lambda x: x[1],
        reverse=True,
    )

    return SummarizeResult(
        summary=summary,
        num_sentences=len(sents),
        top_keywords=top_keywords,
        sentence_scores=sentence_scores,
        error=None,
    )
