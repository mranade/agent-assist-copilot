"""Retrieval over the knowledge base.

TF-IDF with word unigrams and bigrams. For a small, well-curated KB where
agents and customers use the same product vocabulary ("late fee",
"gateway", "technician"), lexical retrieval is fast, free, explainable,
and needs no model or API key. The ``Retriever`` protocol keeps the door
open for dense or hybrid retrieval later without touching the agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .kb import Chunk, load_chunks
from .text import analyzer, topic_tokens


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int = 4) -> list[Hit]: ...


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._vectorizer = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True)
        self._matrix = self._vectorizer.fit_transform(c.text for c in chunks)
        self.vocabulary = {t for t in self._vectorizer.vocabulary_ if " " not in t}

    def coverage(self, query: str) -> float:
        """Share of the query's topic words that the KB has ever used.

        A low value means the caller is talking about something the KB
        doesn't cover at all ("TV bundle", "security cameras"), which a
        similarity score alone can't tell you. Emotion words ("ridiculous",
        "great") are left out: they describe the caller's mood, not the topic.
        """
        tokens = topic_tokens(query)
        if not tokens:
            return 0.0
        return sum(t in self.vocabulary for t in tokens) / len(tokens)

    @classmethod
    def from_kb(cls, kb_dir: Path) -> "TfidfRetriever":
        return cls(load_chunks(kb_dir))

    def search(self, query: str, k: int = 4) -> list[Hit]:
        scores = cosine_similarity(self._vectorizer.transform([query]), self._matrix)[0]
        order = scores.argsort()[::-1][:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]

    def article_score(self, query: str, kb_id: str) -> float:
        """Best score of any section of one article, whether or not it made the top k."""
        scores = cosine_similarity(self._vectorizer.transform([query]), self._matrix)[0]
        return max((float(s) for c, s in zip(self.chunks, scores) if c.kb_id == kb_id), default=0.0)
