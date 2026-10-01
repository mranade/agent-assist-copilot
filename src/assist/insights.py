"""Insights across saved calls: where the KB falls short, and which cards help.

Every number here is computed from the call history on request, so it is
always current and every figure can be traced back to calls and turns:

* **KB gaps**: customer lines that escalated ("No KB coverage"), grouped
  into questions the KB couldn't answer. This is the escalation log the
  synonym list and new articles should grow from.
* **Article performance**: how often each article was shown, how agents
  rated the cards, and how the calls that used it went.
* **Sentiment by topic**: calls grouped by their main article, worst first.
* **Totals**: calls, how many customer lines got a card, and how many
  rated cards were helpful.

Gap grouping is deliberately simple and explainable. A line's *gap words*
are its topic words (``text.topic_tokens``) that the KB has never used,
which are exactly the words that made the coverage check fail. Two lines
are the same gap when they share at least two gap words (one, if either
line has only one). Lines that escalated for another reason (every word is
in the KB, but no grounded answer came back) fall back to all their topic
words. See docs/decisions.md for where this breaks.
"""

from __future__ import annotations

from collections import Counter

from .text import WORD_RE, apply_synonyms, stem, topic_tokens

RATEABLE = ("suggested", "escalated")
MAX_EXAMPLES = 3


def _ratio(part: int, whole: int) -> float | None:
    return round(part / whole, 3) if whole else None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _customer_turns(record: dict):
    for i, t in enumerate(record["turns"]):
        if t["speaker"] == "customer":
            yield i, t


def gap_words(text: str, vocabulary: set[str]) -> list[str]:
    """The topic words that made this line escalate: the ones the KB has never used."""
    tokens = list(dict.fromkeys(topic_tokens(text)))
    missing = [t for t in tokens if t not in vocabulary]
    return missing or tokens


def _surface_forms(text: str) -> dict[str, str]:
    """stem -> the word the customer actually said, for readable labels."""
    words = WORD_RE.findall(apply_synonyms(text))
    return {stem(w): w for w in reversed(words)}  # first occurrence wins


def _same_gap(a: set[str], b: set[str]) -> bool:
    return len(a & b) >= min(2, len(a), len(b)) > 0


def group_gaps(lines: list[dict], vocabulary: set[str]) -> list[dict]:
    """Group escalated lines into repeated questions, most frequent first.

    ``lines`` are dicts with at least "text", plus whatever should be kept
    as an example (call id, turn index, ...).
    """
    ordered = [gap_words(line["text"], vocabulary) for line in lines]
    words = [set(w) for w in ordered]

    # Union-find: a line joins every group it shares enough gap words with.
    parent = list(range(len(lines)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(lines)):
        for j in range(i):
            if _same_gap(words[i], words[j]):
                parent[root(i)] = root(j)

    members: dict[int, list[int]] = {}
    for i in range(len(lines)):
        members.setdefault(root(i), []).append(i)

    groups = []
    for idx in members.values():
        counts = Counter(w for i in idx for w in ordered[i])  # ties keep the order they were said in
        # Label: the words at least half the lines share (all of them, for one line).
        floor = 1 if len(idx) == 1 else max(2, len(idx) / 2)
        shared = [w for w, n in counts.most_common() if n >= floor]
        surface: dict[str, str] = {}
        for i in idx:
            for k, v in _surface_forms(lines[i]["text"]).items():
                surface.setdefault(k, v)
        label_words = [surface.get(w, w) for w in shared[:3]]
        examples = sorted((lines[i] for i in idx), key=lambda x: x.get("started_at", ""), reverse=True)
        groups.append({
            "label": " · ".join(label_words) or "(no topic words)",
            "words": shared,
            "count": len(idx),
            "calls": len({lines[i].get("call_id") for i in idx}),
            "last_seen": examples[0].get("started_at"),
            "examples": examples[:MAX_EXAMPLES],
        })
    groups.sort(key=lambda g: g["last_seen"] or "", reverse=True)
    groups.sort(key=lambda g: g["count"], reverse=True)  # stable: ties stay newest first
    return groups


def main_article(record: dict) -> str | None:
    """The article credited on the most customer cards in a call (first seen wins ties)."""
    counts = Counter(kb for _, t in _customer_turns(record) for kb in t.get("kb_ids") or [])
    return counts.most_common(1)[0][0] if counts else None


def compute(records: list[dict], kb_titles: dict[str, str], vocabulary: set[str]) -> dict:
    customer = suggested = escalated = 0
    rated = helpful = 0
    gap_lines: list[dict] = []
    shown: Counter[str] = Counter()
    votes: dict[str, Counter[str]] = {}
    article_calls: dict[str, list[float]] = {}
    topics: dict[str | None, list[dict]] = {}

    for r in records:
        score = r["sentiment"]["score"]
        used: set[str] = set()
        for i, t in _customer_turns(r):
            customer += 1
            outcome = t.get("outcome")
            suggested += outcome == "suggested"
            escalated += outcome == "escalated"
            feedback = t.get("feedback") if outcome in RATEABLE else None
            if feedback:
                rated += 1
                helpful += feedback == "helpful"
            if outcome == "escalated":
                gap_lines.append({
                    "text": t["text"], "call_id": r["id"], "turn": i,
                    "call_title": r["title"], "started_at": r["started_at"],
                })
            for kb in t.get("kb_ids") or []:
                shown[kb] += 1
                used.add(kb)
                if feedback:
                    votes.setdefault(kb, Counter())[feedback] += 1
        for kb in used:
            article_calls.setdefault(kb, []).append(score)
        topics.setdefault(main_article(r), []).append(r)

    articles = []
    for kb_id, title in sorted(kb_titles.items()):
        v = votes.get(kb_id, Counter())
        n_votes = v["helpful"] + v["not_helpful"]
        articles.append({
            "kb_id": kb_id,
            "title": title,
            "shown": shown[kb_id],
            "calls": len(article_calls.get(kb_id, [])),
            "helpful": v["helpful"],
            "not_helpful": v["not_helpful"],
            "helpful_pct": _ratio(v["helpful"], n_votes),
            "avg_sentiment": _mean(article_calls.get(kb_id, [])),
        })
    articles.sort(key=lambda a: a["shown"], reverse=True)

    by_topic = []
    for kb_id, calls in topics.items():
        scores = [c["sentiment"]["score"] for c in calls]
        by_topic.append({
            "kb_id": kb_id,
            "title": kb_titles.get(kb_id, "No KB article") if kb_id else "No KB article",
            "calls": len(calls),
            "negative": sum(c["sentiment"]["label"] == "negative" for c in calls),
            "avg_sentiment": _mean(scores),
            "call_ids": [c["id"] for c in calls],
        })
    by_topic.sort(key=lambda t: (t["avg_sentiment"], -t["calls"]))  # worst first

    return {
        "totals": {
            "calls": len(records),
            "customer_turns": customer,
            "suggested": suggested,
            "escalated": escalated,
            "skipped": customer - suggested - escalated,
            "suggested_pct": _ratio(suggested, customer),
            "escalated_pct": _ratio(escalated, customer),
            "cards": suggested + escalated,
            "rated": rated,
            "helpful": helpful,
            "helpful_pct": _ratio(helpful, rated),
        },
        "gaps": group_gaps(gap_lines, vocabulary),
        "articles": articles,
        "sentiment_by_topic": by_topic,
    }
