"""Call sentiment: how did the customer feel, and did the call help?

Uses VADER, a lexicon-based sentiment model that runs offline in
microseconds, with a small set of adjustments for support-call language.
Out of the box VADER misreads this domain in predictable ways:

* "Can I get some kind of credit?" scores clearly positive, because VADER
  treats "credit" and "kind" as positive words. On a support call, "credit"
  is a billing term and "kind of" is a hedge.
* "My internet keeps dropping" barely registers as negative, and "cancel"
  is neutral, even though both are the clearest frustration signals an ISP
  gets.

``DOMAIN_LEXICON`` fixes those. Call-level sentiment then weights later
turns more heavily, because how the customer sounds at the end of the call
says more about the outcome than how they sounded when they called in
(most people calling support start out unhappy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# VADER scores words from -4 (most negative) to +4 (most positive).
#
# Support vocabulary: words that are about the *topic* of the call and also
# carry sentiment. These stay topic words for KB coverage (text.py).
SUPPORT_TERMS = {
    # Neutralize words that are positive in everyday text but not here.
    "credit": 0.0, "kind": 0.0, "free": 0.0, "lower": 0.0, "save": 0.0,
    # Service failures and churn.
    "dropping": -1.5, "drops": -1.5, "dropped": -1.5, "down": -1.0,
    "outage": -2.0, "slow": -1.5, "broken": -2.0,
    "cancel": -1.5, "cancelling": -1.5, "canceling": -1.5, "leaving": -1.2,
    "overcharged": -2.5, "charged": -0.8, "late": -0.5, "supervisor": -1.0,
    "resolved": 2.0, "fixed": 1.8, "works": 1.2,
}
# General emotion words: how the caller feels, not what the call is about.
EMOTION_TERMS = {
    "please": 0.0, "sure": 0.0, "okay": 0.3, "honestly": 0.0,
    "ridiculous": -2.5, "unacceptable": -2.8, "frustrating": -2.2,
    "appreciate": 2.0,
}
DOMAIN_LEXICON = {**SUPPORT_TERMS, **EMOTION_TERMS}

POSITIVE, NEGATIVE = 0.10, -0.10  # call-level label thresholds


@dataclass
class CallSentiment:
    label: str                       # "positive" | "neutral" | "negative"
    score: float                     # -1..1, weighted toward the end of the call
    start: float                     # average of the first third of customer turns
    end: float                       # average of the last third
    trend: str                       # "improved" | "worsened" | "steady"
    turn_scores: list[float] = field(default_factory=list)
    most_negative: str | None = None
    most_positive: str | None = None


_analyzer = SentimentIntensityAnalyzer()
_analyzer.lexicon.update(DOMAIN_LEXICON)


def score_text(text: str) -> float:
    """-1 (very negative) to 1 (very positive)."""
    return _analyzer.polarity_scores(text)["compound"]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def call_sentiment(customer_lines: list[str]) -> CallSentiment:
    # Greetings and names score 0 and would dilute the average, so a call's
    # score only counts lines that carry some sentiment.
    scores = [score_text(t) for t in customer_lines]
    scored = [(s, t) for s, t in zip(scores, customer_lines) if s != 0.0]
    if not scored:
        return CallSentiment("neutral", 0.0, 0.0, 0.0, "steady", scores)

    values = [s for s, _ in scored]
    # Quadratic weights (1, 4, 9, ...): the last turns dominate. Linear
    # weights rated a call that ended "Great, thank you so much" as neutral.
    weights = [(i + 1) ** 2 for i in range(len(values))]
    score = sum(w * s for w, s in zip(weights, values)) / sum(weights)

    third = max(1, len(values) // 3)
    start, end = _mean(values[:third]), _mean(values[-third:])
    delta = end - start
    trend = "improved" if delta > 0.25 else "worsened" if delta < -0.25 else "steady"
    label = "positive" if score >= POSITIVE else "negative" if score <= NEGATIVE else "neutral"

    lowest = min(scored)
    highest = max(scored)
    return CallSentiment(
        label=label,
        score=round(score, 3),
        start=round(start, 3),
        end=round(end, 3),
        trend=trend,
        turn_scores=[round(s, 3) for s in scores],
        most_negative=lowest[1] if lowest[0] < 0 else None,
        most_positive=highest[1] if highest[0] > 0 else None,
    )
