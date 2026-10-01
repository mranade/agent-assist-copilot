"""Turn retrieved KB chunks into a suggestion card for the human agent.

A card comes in one of two shapes:

* **Plan card**, when the call has an active issue whose article has a
  checklist (see ``steps.py``): the next step to do, a line the agent could
  say for it, the checklist with done steps ticked, and, when the customer
  asked about something else, an "Also relevant" or "Policy" note from that
  other article.
* **Lines card**, otherwise: the most relevant KB sentences, as before.

Two interchangeable implementations behind one interface:

* ``ExtractiveSuggester`` (default): never writes new text. Card lines are
  verbatim KB text, and the "Say" line is the step text with a fixed
  template around it ("Could you ...?"). Runs offline with no API key.
* ``ClaudeSuggester``: Claude writes the "Say" line (citing the KB id) and,
  for lines cards, a short reply from the retrieved excerpts, answering
  NO_ANSWER when they don't cover the question. Used automatically when
  ``ANTHROPIC_API_KEY`` is set.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Protocol

from .retriever import Hit
from .steps import STEP_SECTION, say_for_step
from .text import content_tokens

MAX_LINES = 3
MAX_NOTE_LINES = 2


@dataclass
class Article:
    kb_id: str
    title: str
    section: str


@dataclass
class PlanStep:
    text: str
    done: bool = False
    done_by: str | None = None    # "agent" or "customer"
    because: str | None = None    # why it was marked done, for "Why this card"


@dataclass
class Plan:
    kb_id: str
    title: str
    steps: list[PlanStep]
    next: int | None              # index of the next step to do; None when all are done
    say: str | None = None        # a line the agent could say for the next step
    say_by: str = "template"      # "template" or "claude"


@dataclass
class Note:
    kind: str                     # "Policy" (article has no checklist) or "Also relevant"
    kb_id: str
    title: str
    section: str
    lines: list[str]


@dataclass
class Suggestion:
    headline: str
    lines: list[str]
    articles: list[Article]
    generated_by: str = "extractive"
    grounded: bool = True
    sources: list[str] = field(default_factory=list)  # KB ids the shown content came from
    plan: Plan | None = None
    note: Note | None = None


class Suggester(Protocol):
    name: str
    def suggest(self, utterance: str, hits: list[Hit], plan: Plan | None = None,
                checklists: frozenset[str] = frozenset()) -> Suggestion: ...
    def say(self, step: str, kb_id: str, context: str) -> tuple[str, str]: ...


def _units(chunk_text: str) -> list[str]:
    """Split a chunk into list items and sentences, dropping the title line."""
    body = chunk_text.split("\n", 1)[-1]
    units: list[str] = []
    for line in body.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line).strip()
        if not line:
            continue
        units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
    return units


def _usable(chunk_text: str) -> list[str]:
    # "Use this when ..." lines are scoping notes for KB authors, not for the call.
    return [u for u in _units(chunk_text) if not u.lower().startswith("use this when")]


def _articles(hits: list[Hit], limit: int = 2) -> list[Article]:
    seen: dict[str, Article] = {}
    for h in hits:
        if h.chunk.kb_id not in seen:
            seen[h.chunk.kb_id] = Article(h.chunk.kb_id, h.chunk.title, h.chunk.section)
        if len(seen) == limit:
            break
    return list(seen.values())


def extract_lines(utterance: str, hits: list[Hit]) -> tuple[list[str], list[str]]:
    """The most relevant KB lines for a lines card, and the KB ids they came from."""
    query = set(content_tokens(utterance))
    shown = {a.kb_id for a in _articles(hits)}
    top_kb = hits[0].chunk.kb_id

    # Lines from the top article need one shared term with what the
    # customer said; lines from the runner-up article need two. That
    # keeps a weakly related second article from crowding the card.
    scored: list[tuple[int, int, str, str]] = []
    for rank, h in enumerate(hits):
        if h.chunk.kb_id not in shown:
            continue
        need = 1 if h.chunk.kb_id == top_kb else 2
        for order, unit in enumerate(_usable(h.chunk.text)):
            overlap = len(query & set(content_tokens(unit)))
            if overlap >= need:
                bonus = 1 if h.chunk.kb_id == top_kb else 0
                scored.append((-(overlap + bonus), rank * 100 + order, unit, h.chunk.kb_id))
    scored.sort()
    picked: dict[str, str] = {}  # line -> kb_id, first occurrence wins
    for _, _, unit, kb_id in scored:
        picked.setdefault(unit, kb_id)
    lines = list(picked)[:MAX_LINES]
    sources = list(dict.fromkeys(picked[ln] for ln in lines))
    if not lines:
        # Nothing shares words with the question: fall back to the first
        # retrieved section that has usable lines. (The top section can be
        # nothing but a "Use this when" note, which would leave the card
        # empty and make the grounding check escalate a good match.)
        for h in hits:
            usable = _usable(h.chunk.text)
            if usable:
                lines, sources = usable[:MAX_LINES], [h.chunk.kb_id]
                break
    return lines, sources


def build_note(utterance: str, hits: list[Hit], plan_kb: str,
               checklists: frozenset[str] = frozenset()) -> Note | None:
    """A secondary note on a plan card, from an article other than the active issue.

    If this turn's best match is another article, the customer asked about
    something else (a credit, a fee): show its most relevant lines. If the
    best match is the active issue itself, the runner-up article only gets a
    note when its lines share two words with the customer's line, the same
    bar the lines card uses for a second article.
    """
    others = [h for h in hits if h.chunk.kb_id != plan_kb]
    if not others:
        return None
    kb_id = others[0].chunk.kb_id
    asked_about_it = hits[0].chunk.kb_id == kb_id
    need = 1 if asked_about_it else 2
    query = set(content_tokens(utterance))
    scored: list[tuple[int, int, str]] = []
    for rank, h in enumerate(others):
        if h.chunk.kb_id != kb_id:
            continue
        for order, unit in enumerate(_usable(h.chunk.text)):
            overlap = len(query & set(content_tokens(unit)))
            if overlap >= need:
                scored.append((-overlap, rank * 100 + order, unit))
    lines = list(dict.fromkeys(u for _, _, u in sorted(scored)))[:MAX_NOTE_LINES]
    if not lines and asked_about_it:
        lines = _usable(others[0].chunk.text)[:MAX_NOTE_LINES]
    if not lines:
        return None
    kind = "Also relevant" if kb_id in checklists else "Policy"
    return Note(kind, kb_id, others[0].chunk.title, others[0].chunk.section, lines)


def plan_card(plan: Plan, note: Note | None, generated_by: str) -> Suggestion:
    """A card built around the active issue's checklist.

    ``lines`` holds the verbatim KB text on the card (the next step and the
    note's lines), so the history and summary can show it without the UI.
    """
    lines = [plan.steps[plan.next].text] if plan.next is not None else []
    articles = [Article(plan.kb_id, plan.title, STEP_SECTION)]
    if note:
        lines += note.lines
        articles.append(Article(note.kb_id, note.title, note.section))
    return Suggestion(
        headline=f"{plan.title} ({plan.kb_id})",
        lines=lines,
        articles=articles,
        generated_by=generated_by,
        sources=[a.kb_id for a in articles],
        plan=plan,
        note=note,
    )


class ExtractiveSuggester:
    name = "extractive"

    def suggest(self, utterance: str, hits: list[Hit], plan: Plan | None = None,
                checklists: frozenset[str] = frozenset()) -> Suggestion:
        if plan is not None:
            return plan_card(plan, build_note(utterance, hits, plan.kb_id, checklists), self.name)
        lines, sources = extract_lines(utterance, hits)
        return Suggestion(
            headline=f"{hits[0].chunk.title} ({hits[0].chunk.kb_id})",
            lines=lines,
            articles=_articles(hits),
            generated_by=self.name,
            sources=sources,
        )

    def say(self, step: str, kb_id: str, context: str) -> tuple[str, str]:
        return say_for_step(step), "template"


def claude_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


class ClaudeSuggester:
    name = "claude"

    def __init__(self, model: str | None = None):
        from anthropic import Anthropic  # optional dependency

        self._client = Anthropic()
        self._model = model or os.environ.get("ASSIST_MODEL", "claude-sonnet-5")

    def _ask(self, prompt: str) -> str:
        resp = self._client.messages.create(
            model=self._model, max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()

    def suggest(self, utterance: str, hits: list[Hit], plan: Plan | None = None,
                checklists: frozenset[str] = frozenset()) -> Suggestion:
        if plan is not None:
            # The step and the note are KB text; Claude only wrote plan.say.
            return plan_card(plan, build_note(utterance, hits, plan.kb_id, checklists), self.name)
        excerpts = "\n\n".join(f"[{h.chunk.kb_id}] {h.chunk.text}" for h in hits)
        text = self._ask(
            "You are assisting a human contact-center agent at Relay Fiber during a live call. "
            "Using ONLY the knowledge-base excerpts below, write up to 3 short lines: first a "
            "sentence the agent could say to the customer, then the next steps the agent should "
            "take. Put the KB id you relied on in brackets at the end of each line, like [KB-101]. "
            "If the excerpts do not cover what the customer asked, reply with exactly NO_ANSWER.\n\n"
            f"Excerpts:\n{excerpts}\n\nCustomer just said: {utterance}"
        )
        if text == "NO_ANSWER":
            return Suggestion("No KB coverage", [], [], self.name, grounded=False)
        lines = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()][:MAX_LINES]
        cited = list(dict.fromkeys(k for ln in lines for k in re.findall(r"KB-\d{3}", ln)))
        return Suggestion(
            headline=f"{hits[0].chunk.title} ({hits[0].chunk.kb_id})",
            lines=lines, articles=_articles(hits), generated_by=self.name, sources=cited,
        )

    def say(self, step: str, kb_id: str, context: str) -> tuple[str, str]:
        """Claude phrases the step for the customer; falls back to the template if it
        doesn't cite the article or declines."""
        text = self._ask(
            "You are helping a human contact-center agent at Relay Fiber on a live call. "
            f"The agent's next step, from knowledge-base article {kb_id}, is:\n\n{step}\n\n"
            f"Context:\n{context}\n\n"
            "Write ONE short, natural sentence the agent could say to the customer to carry out "
            "this step. Use only facts stated in the step; do not add prices, dates, or promises. "
            f"End with [{kb_id}]. If the step is internal and there is nothing to say to the "
            "customer, reply with exactly NO_ANSWER."
        )
        cited = set(re.findall(r"KB-\d{3}", text))
        if text == "NO_ANSWER" or cited != {kb_id}:
            return say_for_step(step), "template"
        return text.splitlines()[0].strip(), "claude"


def is_grounded(suggestion: Suggestion, hits: list[Hit], extra_ids: set[str] | frozenset[str] = frozenset()) -> bool:
    """Everything on the card must be traceable to the knowledge base.

    Extractive cards are verbatim KB text (plus template words around a step),
    so they pass as long as they aren't empty. For text Claude wrote we
    require a cited KB id that was actually retrieved, or that is the call's
    active issue (``extra_ids``), which catches the model citing (or
    inventing) an article it wasn't given.
    """
    if not suggestion.grounded or not (suggestion.lines or suggestion.plan):
        return False
    if suggestion.generated_by == "extractive":
        return True
    if suggestion.plan is not None:
        written = [suggestion.plan.say] if suggestion.plan.say_by == "claude" and suggestion.plan.say else []
    else:
        written = suggestion.lines
    allowed = {h.chunk.kb_id for h in hits} | set(extra_ids)
    for line in written:
        cited = set(re.findall(r"KB-\d{3}", line))
        if not cited or not cited <= allowed:
            return False
    return True
