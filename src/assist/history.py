"""Finished calls: build the record, store it, list it, seed demo history.

A call becomes a history record when the agent clicks "End call". The
record keeps the redacted transcript, what the assist panel showed on each
customer turn (and how the agent rated each card), the wrap-up summary,
and the call's sentiment, so a supervisor can review the call later
without re-running anything. Ratings can still change after the call is
saved.

Storage is a single SQLite file (stdlib, no server). Each record is stored
as JSON in one row, which is plenty for a demo and keeps the schema out of
the way; a real deployment would normalize turns into their own table to
query across calls ("every call that escalated on KB-113").
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .agent import AssistAgent, AssistResult
from .redact import redact
from .sentiment import call_sentiment, score_text
from .steps import CallState
from .summary import TurnRecord, extractive_summary


def now() -> datetime:
    return datetime.now(timezone.utc)


def build_record(
    turns: list[TurnRecord],
    kb_titles: dict[str, str],
    *,
    source: str,
    started_at: datetime,
    ended_at: datetime,
    agent_name: str | None = None,
    summarizer=None,
) -> dict:
    """Assemble a history record from a finished call's turns."""
    summary = summarizer.summarize(turns, kb_titles) if summarizer else extractive_summary(turns, kb_titles)
    customer_lines = [t.text for t in turns if t.speaker == "customer"]
    sentiment = call_sentiment(customer_lines)

    topics = [a.split(" ", 1)[1] for a in summary.articles_used[:2] if " " in a]
    title = " · ".join(topics) if topics else "General inquiry"

    scores = iter(sentiment.turn_scores)
    turn_dicts = []
    for t in turns:
        d = asdict(t)
        d["sentiment"] = next(scores) if t.speaker == "customer" else None
        turn_dicts.append(d)

    return {
        "id": uuid.uuid4().hex[:12],
        "source": source,
        "agent": agent_name,
        "title": title,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_s": int((ended_at - started_at).total_seconds()),
        "turns": turn_dicts,
        "summary": asdict(summary),
        "sentiment": asdict(sentiment),
    }


class HistoryStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS calls ("
            " id TEXT PRIMARY KEY, started_at TEXT NOT NULL, record TEXT NOT NULL)"
        )
        self._conn.commit()

    def save(self, record: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO calls (id, started_at, record) VALUES (?, ?, ?)",
            (record["id"], record["started_at"], json.dumps(record)),
        )
        self._conn.commit()

    def get(self, call_id: str) -> dict | None:
        row = self._conn.execute("SELECT record FROM calls WHERE id = ?", (call_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def list(self) -> list[dict]:
        rows = self._conn.execute("SELECT record FROM calls ORDER BY started_at DESC").fetchall()
        out = []
        for (raw,) in rows:
            r = json.loads(raw)
            out.append({
                "id": r["id"],
                "title": r["title"],
                "agent": r.get("agent"),
                "source": r["source"],
                "started_at": r["started_at"],
                "duration_s": r["duration_s"],
                "turn_count": len(r["turns"]),
                "sentiment": {k: r["sentiment"][k] for k in ("label", "score", "trend")},
                "open_items": len(r["summary"]["open_items"]),
            })
        return out

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]

    def records(self) -> list[dict]:
        """Every saved call in full, newest first (for insights across calls)."""
        rows = self._conn.execute("SELECT record FROM calls ORDER BY started_at DESC").fetchall()
        return [json.loads(raw) for (raw,) in rows]

    def set_feedback(self, call_id: str, index: int, value: str | None) -> dict:
        """Record the agent's rating of one card on a saved call.

        Raises LookupError for an unknown call or turn, and ValueError for a
        turn that had no card to rate.
        """
        record = self.get(call_id)
        if record is None or not 0 <= index < len(record["turns"]):
            raise LookupError("Unknown call turn")
        turn = record["turns"][index]
        if not can_rate(turn["speaker"], turn["outcome"]):
            raise ValueError("That turn had no assist card to rate.")
        turn["feedback"] = value
        self.save(record)
        return turn


FEEDBACK = ("helpful", "not_helpful")


def can_rate(speaker: str, outcome: str | None) -> bool:
    """Only customer turns that showed a card (a suggestion or "no KB coverage") can be rated."""
    return speaker == "customer" and outcome in ("suggested", "escalated")


def apply_result(record: TurnRecord, result: AssistResult) -> None:
    """Keep what the assist panel showed for this turn with the call record."""
    record.plan = asdict(result.plan) if result.plan else None
    record.steps_marked = result.steps_marked
    if record.speaker != "customer":
        return
    record.outcome = result.outcome
    s = result.suggestion
    if s and result.outcome in ("suggested", "escalated"):
        record.headline = s.headline
        record.lines = s.lines
        record.card = asdict(s)
    if s and result.outcome == "suggested":
        # Credit the articles whose content was actually shown, not just the top hit.
        record.kb_ids = s.sources or [s.articles[0].kb_id]


def run_transcript(agent: AssistAgent, turns: list[dict], feedback: dict[int, str] | None = None) -> list[TurnRecord]:
    """Replay a scripted transcript through the real assist pipeline, both speakers.

    ``feedback`` (turn index -> rating), or a turn's own "feedback" key, is the
    agent's rating of that turn's card, kept only if the turn produced a card.
    """
    records: list[TurnRecord] = []
    call = CallState()
    for i, turn in enumerate(turns):
        record = TurnRecord(speaker=turn["speaker"], text=redact(turn["text"]))
        apply_result(record, agent.on_turn(turn["speaker"], record.text, call))
        rating = (feedback or {}).get(i) or turn.get("feedback")
        if rating and can_rate(record.speaker, record.outcome):
            record.feedback = rating
        records.append(record)
    return records


# How the demo agents rated the sample calls' cards (customer turn index ->
# rating). Not every card is rated, as in real use. The "bill looked higher"
# card in call-02 is the runner-up-note noise described in docs/decisions.md.
SAMPLE_FEEDBACK = {
    "call-01": {3: "helpful", 5: "helpful", 11: "helpful", 13: "helpful"},
    "call-02": {1: "helpful", 5: "not_helpful", 11: "helpful"},
    "call-03": {1: "helpful", 5: "helpful", 13: "not_helpful"},
}


def seed(store: HistoryStore, agent: AssistAgent, kb_titles: dict[str, str], root: Path) -> int:
    """Fill an empty history with demo calls, all produced by the real pipeline."""
    past = json.loads((root / "data" / "history" / "past_calls.json").read_text(encoding="utf-8"))
    samples = json.loads((root / "data" / "calls" / "calls.json").read_text(encoding="utf-8"))
    agents = {"call-01": "Sam", "call-02": "Priya", "call-03": "Marco"}
    calls = [
        (c["id"], c.get("agent"), c["days_ago"], c["turns"]) for c in past
    ] + [
        (c["id"], agents.get(c["id"]), 1, c["turns"]) for c in samples
    ]
    for i, (source, agent_name, days_ago, turns) in enumerate(calls):
        start = now() - timedelta(days=days_ago, hours=3 + i)
        end = start + timedelta(seconds=40 * len(turns))
        records = run_transcript(agent, turns, SAMPLE_FEEDBACK.get(source))
        store.save(build_record(
            records, kb_titles, source=source, started_at=start, ended_at=end,
            agent_name=agent_name,
        ))
    return len(calls)
