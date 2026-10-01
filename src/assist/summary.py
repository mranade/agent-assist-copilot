"""End-of-call summary (the "wrap-up note" agents otherwise type by hand).

The default summarizer is deterministic: it assembles the note from what
the agent loop already recorded during the call (the customer's first
request, every KB article surfaced, and every question that escalated).
That makes it cheap and auditable, since each line points back to a turn
or an article. ``ClaudeSummarizer`` writes a prose note from the same
redacted transcript when ``ANTHROPIC_API_KEY`` is set.

Everything here runs on redacted text only, so account numbers and PINs
spoken during verification never reach the summary or an LLM.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .redact import redact


@dataclass
class TurnRecord:
    speaker: str
    text: str                       # already redacted
    outcome: str | None = None      # agent outcome, customer turns only
    kb_ids: list[str] = field(default_factory=list)
    headline: str | None = None
    lines: list[str] = field(default_factory=list)  # what the assist card showed, as KB text
    card: dict | None = None        # the full card (next step, say, checklist, note)
    plan: dict | None = None        # the active checklist after this turn, either speaker
    steps_marked: list[dict] = field(default_factory=list)  # checklist steps this turn completed
    feedback: str | None = None     # the agent's rating of the card: "helpful" | "not_helpful"


@dataclass
class CallSummary:
    reason: str
    articles_used: list[str]
    open_items: list[str]
    note: str
    generated_by: str


def extractive_summary(turns: list[TurnRecord], kb_titles: dict[str, str]) -> CallSummary:
    customer_turns = [t for t in turns if t.speaker == "customer"]
    first_request = next(
        (t for t in customer_turns if t.outcome in ("suggested", "escalated")), None
    )
    reason = first_request.text if first_request else "No support request captured."

    articles: list[str] = []
    for t in customer_turns:
        for kb_id in t.kb_ids:
            label = f"{kb_id} {kb_titles.get(kb_id, '')}".strip()
            if label not in articles:
                articles.append(label)

    open_items = [t.text for t in customer_turns if t.outcome == "escalated"]

    lines = [f"Reason for call: {reason}"]
    if articles:
        lines.append("Guidance used: " + "; ".join(articles) + ".")
    plan = next((t.plan for t in reversed(turns) if t.plan), None)
    if plan:
        done = sum(s["done"] for s in plan["steps"])
        lines.append(f"Checklist {plan['kb_id']} {plan['title']}: {done} of {len(plan['steps'])} steps done.")
    if open_items:
        lines.append("Not covered by the KB (follow up): " + " / ".join(open_items))
    else:
        lines.append("All customer questions were covered by the KB.")
    return CallSummary(reason, articles, open_items, "\n".join(lines), "extractive")


class ClaudeSummarizer:
    def __init__(self, model: str | None = None):
        from anthropic import Anthropic  # optional dependency

        self._client = Anthropic()
        self._model = model or os.environ.get("ASSIST_MODEL", "claude-sonnet-5")

    def summarize(self, turns: list[TurnRecord], kb_titles: dict[str, str]) -> CallSummary:
        base = extractive_summary(turns, kb_titles)
        transcript = "\n".join(f"{t.speaker.upper()}: {redact(t.text)}" for t in turns)
        prompt = (
            "Write a 3-4 sentence wrap-up note for this Relay Fiber support call, for the "
            "account history. Cover why the customer called, what the agent did or offered, "
            "and anything unresolved. Do not include numbers that were redacted.\n\n"
            f"Knowledge-base articles surfaced: {', '.join(base.articles_used) or 'none'}\n"
            f"Questions the KB did not cover: {'; '.join(base.open_items) or 'none'}\n\n"
            f"Transcript:\n{transcript}"
        )
        resp = self._client.messages.create(
            model=self._model, max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        note = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        return CallSummary(base.reason, base.articles_used, base.open_items, note, "claude")
