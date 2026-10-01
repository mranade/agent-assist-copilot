"""Text normalization shared by retrieval and grading.

Customers don't talk like knowledge-base articles. They say "router" or
"modem" where the KB says "gateway", "money back" where it says "credit",
"cancelling" where it says "cancellation". Lexical retrieval only matches
words it has seen, so we close that gap in two cheap, explainable steps:

1. A small synonym map from customer vocabulary to KB vocabulary. In a real
   deployment this list grows out of the escalation log: every "no KB
   match" that turns out to be a vocabulary gap becomes a new entry.
2. Stemming (Snowball), so "cancel", "cancelling", and "cancellation"
   become the same token.
"""

from __future__ import annotations

import re

import snowballstemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_stemmer = snowballstemmer.stemmer("english")

# Multi-word phrases first, then single words. Keys are what callers say;
# values are the KB's own terms.
#
# Deliberately short: every entry comes from the sample calls or is an
# obvious domain synonym. Nothing was added to make the held-out eval set
# (data/eval/utterances.jsonl) pass; its misses are reported as-is.
PHRASE_SYNONYMS = {
    "cuts out": "drops",
}
WORD_SYNONYMS = {
    "router": "gateway",
    "modem": "gateway",
    "tech": "technician",
    "refund": "credit",
    "cost": "price",
    "expensive": "price",
}

# Conversational words that carry no support meaning. Removing them keeps
# the coverage check from being diluted by "honestly", "guys", etc.
FILLER = {
    "hi", "hey", "hello", "okay", "ok", "yeah", "yes", "sure", "thanks",
    "thank", "honestly", "guys", "like", "really", "just", "kind", "gonna",
    "wanna", "know", "want", "need", "think", "thinking", "great",
    "perfect", "doing", "today", "much", "also", "maybe", "actually",
    "please", "right", "let", "tell", "gave", "going",
    "redacted",  # the mask left behind by redact.py
    # Generic verbs that say nothing about the topic of the call.
    "make", "making", "made", "get", "getting", "look", "looking", "take",
    "give", "said", "says", "thing", "things", "stuff",
}

WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")


def apply_synonyms(text: str) -> str:
    text = text.lower()
    for phrase, replacement in PHRASE_SYNONYMS.items():
        text = text.replace(phrase, replacement)
    return " ".join(WORD_SYNONYMS.get(w, w) for w in text.split())


def stem(word: str) -> str:
    return _stemmer.stemWord(word)


def content_tokens(text: str) -> list[str]:
    """Stemmed, meaningful tokens: synonyms applied, stopwords and filler removed."""
    words = WORD_RE.findall(apply_synonyms(text))
    kept = [
        w for w in words
        if len(w) > 2 and w not in ENGLISH_STOP_WORDS and w not in FILLER and not w.isdigit()
    ]
    return _stemmer.stemWords(kept)


def _emotion_stems() -> frozenset[str]:
    """Stemmed general-purpose emotion words ("ridiculous", "great", "easy").

    Taken from VADER's lexicon (|valence| >= 1) plus the emotion words added
    for call sentiment. Support vocabulary such as "slow", "outage", and
    "cancel" is both emotional and topical, so it is always kept as a topic.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    from .sentiment import EMOTION_TERMS, SUPPORT_TERMS

    lexicon = SentimentIntensityAnalyzer().lexicon  # fresh copy, without domain edits
    words = [w for w, v in lexicon.items() if abs(v) >= 1.0 and w.isalpha()] + list(EMOTION_TERMS)
    keep = frozenset(_stemmer.stemWords(list(SUPPORT_TERMS)))
    return frozenset(_stemmer.stemWords(words)) - keep


_EMOTION: frozenset[str] | None = None


def topic_tokens(text: str) -> list[str]:
    """Content tokens minus emotion words: what the customer is talking *about*.

    Used for KB coverage and triage. "This is ridiculous" tells you how the
    customer feels, not what the call is about, so it shouldn't count as a
    topic the KB fails to cover.
    """
    global _EMOTION
    if _EMOTION is None:
        _EMOTION = _emotion_stems()
    return [t for t in content_tokens(text) if t not in _EMOTION]


def analyzer(text: str) -> list[str]:
    """Unigrams + bigrams over content tokens, for the TF-IDF vectorizer."""
    tokens = content_tokens(text)
    return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
