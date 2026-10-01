"""The assist agent: a LangGraph state machine that runs on every customer turn.

    track ──▶ triage ──small talk──────────────────────────────────────────────▶ skip
                │
                ▼
    retrieve ──▶ grade_retrieval ──sufficient──▶ focus ──▶ suggest ──▶ grade_suggestion ──grounded──▶ done
      ▲                │                                                    │
      │         insufficient,                                          ungrounded
      │         earlier turns exist                                          │
      │                ▼                                                    ▼
      └──── rewrite_with_context                      insufficient, ───▶ escalate
                                                      no retry left

Agent turns never get a card. They only go through step tracking
(``on_agent_turn``), so the card can advance as the agent works.

Ideas worth calling out:

* Grading uses two signals, not one. Similarity score says "the best chunk
  looks like the question"; coverage says "the KB has ever talked about the
  things in this question". A question about a TV bundle can still score
  decently against a pricing article, but the KB has never used the words
  "TV" or "bundle", so coverage catches it.
* The retry uses conversation memory. Customers say things like "and what do
  I do with it?" that only make sense with the previous turn, so the one
  retry prepends the previous customer turn to the query. Coverage is still
  judged on the new utterance alone, so borrowed context can't make an
  out-of-scope question look answerable.
* The call has an active issue (see ``steps.py``). ``focus`` sets it from the
  first matched article that has a checklist and keeps it through side
  questions: a credit question adds a KB-109 note to the KB-101 card instead
  of replacing the troubleshooting plan. Another article only takes over when
  it is clearly a new issue: it has its own checklist, the active checklist
  doesn't link to it, it matches strongly (``NEW_ISSUE_SCORE``), and the line
  no longer matches the active article (below ``MIN_SCORE``).
* ``track`` marks checklist steps done from what was said, so the card shows
  the next step to take rather than the most similar sentences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .retriever import Hit, TfidfRetriever
from .steps import MIN_SHARED, CallState, Playbook, describe_mark, match_step, playbooks_from_chunks
from .suggest import Article, Plan, PlanStep, Suggester, Suggestion, is_grounded

MIN_SCORE = 0.13      # set from the sample calls: midpoint between in-scope and out-of-scope
MIN_COVERAGE = 0.30   # same method; see docs/decisions.md
TOP_K = 4
# A different checklist article replaces the active issue only at or above
# this score. Weakly supported: the only new-issue line in the sample calls
# scores 0.20; see docs/decisions.md.
NEW_ISSUE_SCORE = 0.18

REQUEST_RE = re.compile(
    r"\?|^(?:do|does|can|could|how|what|why|where|when|is|are|should|will|would)\b"
    r"|\b(?:i want|i need|i'd like|i would like)\b",
    re.IGNORECASE,
)


class TurnState(TypedDict, total=False):
    utterance: str
    history: list[str]          # earlier customer turns in this call, oldest first
    query: str
    retried: bool
    hits: list[Hit]
    coverage: float
    suggestion: Suggestion
    outcome: str                # "suggested" | "escalated" | "skipped"
    trace: list[str]
    call: CallState             # the call's active issue and step progress, updated in place
    marks: list[dict]           # steps this turn marked done


@dataclass
class AssistResult:
    outcome: str
    suggestion: Suggestion | None
    trace: list[str] = field(default_factory=list)
    top_score: float = 0.0
    coverage: float = 0.0
    plan: Plan | None = None                                  # the checklist after this turn
    steps_marked: list[dict] = field(default_factory=list)    # steps this turn marked done


def looks_like_request(text: str) -> bool:
    return bool(REQUEST_RE.search(text.strip()))


def track_turn(books: dict[str, Playbook], call: CallState, speaker: str, text: str) -> tuple[list[str], list[dict]]:
    """Mark the active checklist's step this turn completes, if any. Returns (trace, marks)."""
    if not call.issue:
        return [], []
    book = books[call.issue]
    mark = match_step(book, call.done, speaker, text)
    if mark is None:
        return [f"track: no {book.kb_id} step matched (needs {MIN_SHARED} shared step words)"], []
    call.done[mark.step] = mark
    return [describe_mark(book, mark)], [{
        "kb_id": book.kb_id, "step": mark.step + 1, "of": len(book.steps),
        "text": book.steps[mark.step], "by": speaker, "words": mark.words,
    }]


def plan_view(books: dict[str, Playbook], suggester: Suggester, call: CallState, context: str) -> Plan | None:
    """The active issue's checklist as the card shows it, with the next step's "Say" line."""
    if not call.issue:
        return None
    book = books[call.issue]
    steps = []
    for i, text in enumerate(book.steps):
        mark = call.done.get(i)
        because = f"{mark.speaker} said " + ", ".join(f'"{w}"' for w in mark.words) if mark else None
        steps.append(PlanStep(text, mark is not None, mark.speaker if mark else None, because))
    nxt = next((i for i in range(len(book.steps)) if i not in call.done), None)
    say, say_by = None, "template"
    if nxt is not None:
        if nxt not in call.say_cache:  # one Claude call per step, not per turn
            call.say_cache[nxt] = suggester.say(book.steps[nxt], book.kb_id, context)
        say, say_by = call.say_cache[nxt]
    return Plan(book.kb_id, book.title, steps, nxt, say, say_by)


def build_agent(retriever: TfidfRetriever, suggester: Suggester, books: dict[str, Playbook]):
    checklists = frozenset(books)

    def log(state: TurnState, *msgs: str) -> list[str]:
        return [*state.get("trace", []), *msgs]

    def track(state: TurnState) -> TurnState:
        # Runs against the plan that was active when the customer started talking.
        msgs, marks = track_turn(books, state["call"], "customer", state["utterance"])
        return {"trace": log(state, *msgs), "marks": [*state.get("marks", []), *marks]}

    def triage(state: TurnState) -> TurnState:
        # Small talk = not a question/request AND nothing the KB has ever
        # talked about. An earlier version used "short and not a question",
        # which silently dropped problem statements like "the internet is
        # completely down"; see docs/decisions.md.
        utterance = state["utterance"]
        small_talk = not looks_like_request(utterance) and retriever.coverage(utterance) == 0
        return {
            "query": utterance,
            "retried": False,
            "trace": log(state, f"triage: {'small talk, no card' if small_talk else 'support content'}"),
            "outcome": "skipped" if small_talk else "",
        }

    def route_triage(state: TurnState) -> str:
        return "skip" if state["outcome"] == "skipped" else "retrieve"

    def retrieve(state: TurnState) -> TurnState:
        hits = retriever.search(state["query"], k=TOP_K)
        coverage = retriever.coverage(state["utterance"])
        top = f"{hits[0].chunk.kb_id} score={hits[0].score:.2f}" if hits else "no hits"
        return {
            "hits": hits,
            "coverage": coverage,
            "trace": log(state, f"retrieve: {top}, coverage={coverage:.2f}"),
        }

    def grade_retrieval(state: TurnState) -> str:
        hits = state["hits"]
        if hits and hits[0].score >= MIN_SCORE and state["coverage"] >= MIN_COVERAGE:
            return "suggest"
        if not state["retried"] and state.get("history"):
            return "rewrite"
        return "escalate"

    def rewrite_with_context(state: TurnState) -> TurnState:
        previous = state["history"][-1]
        return {
            "query": f"{previous} {state['utterance']}",
            "retried": True,
            "trace": log(state, "grade: weak match, retrying with the previous customer turn as context"),
        }

    def keep_reason(active: str, hit: Hit, query: str) -> str | None:
        """Why the active issue stays, or None if ``hit`` is clearly a new issue."""
        kb_id = hit.chunk.kb_id
        if kb_id not in books:
            return f"{kb_id} has no checklist"
        if kb_id in books[active].links:
            return f"the {active} checklist links to {kb_id}"
        if hit.score < NEW_ISSUE_SCORE:
            return f"{kb_id} scores {hit.score:.2f}, below {NEW_ISSUE_SCORE} for a new issue"
        still = retriever.article_score(query, active)
        if still >= MIN_SCORE:
            return f"the line still matches {active} ({still:.2f})"
        return None

    def focus(state: TurnState) -> TurnState:
        call, top = state["call"], state["hits"][0]
        kb_id, active = top.chunk.kb_id, call.issue
        msgs: list[str] = []
        if active is None:
            if kb_id in books:
                call.set_issue(kb_id)
                msgs.append(f"issue: {kb_id} is the active issue (first match with a checklist)")
            else:
                msgs.append(f"issue: none yet; {kb_id} has no checklist, so the card shows its most relevant lines")
        elif kb_id == active:
            msgs.append(f"issue: still {active}")
        else:
            reason = keep_reason(active, top, state["query"])
            if reason is None:
                call.set_issue(kb_id)
                msgs.append(f"issue: {kb_id} replaces {active}: a strong match ({top.score:.2f}) with its own "
                            f"checklist, and the line no longer matches {active}")
            else:
                msgs.append(f"issue: stays {active}; {kb_id} goes in a note because {reason}")
        # The line that raises an issue is not tracked against its new checklist:
        # "I'm moving to a new address" mentions the address without giving it.
        return {"trace": log(state, *msgs)}

    def suggest(state: TurnState) -> TurnState:
        plan = plan_view(books, suggester, state["call"], state["utterance"])
        suggestion = suggester.suggest(state["utterance"], state["hits"], plan, checklists)
        if plan is None:
            shape = "most relevant lines"
        elif plan.next is None:
            shape = "all steps done"
        else:
            shape = f"next step {plan.next + 1} of {len(plan.steps)}"
        return {
            "suggestion": suggestion,
            "trace": log(state, f"grade: sufficient; suggest via {suggestion.generated_by} ({shape})"),
        }

    def grade_suggestion(state: TurnState) -> str:
        issue = {state["call"].issue} if state["call"].issue else set()
        return "done" if is_grounded(state["suggestion"], state["hits"], issue) else "escalate"

    def done(state: TurnState) -> TurnState:
        return {"outcome": "suggested", "trace": log(state, "done: suggestion shown to agent")}

    def escalate(state: TurnState) -> TurnState:
        return {
            "outcome": "escalated",
            "suggestion": Suggestion(
                headline="No knowledge-base coverage",
                lines=[
                    "The KB doesn't cover this. Don't guess: tell the customer plainly if it isn't something Relay Fiber offers.",
                    "If they need more help, follow the transfer steps in the escalation policy.",
                    "The question is logged as a possible KB gap.",
                ],
                articles=[Article("KB-120", "Escalation and supervisor transfers", "Overview")],
                generated_by="policy",
                grounded=False,
                # Keep the plan on screen: a side question the KB can't answer
                # doesn't change what the agent should do next.
                plan=plan_view(books, suggester, state["call"], state["utterance"]),
            ),
            "trace": log(state, "escalate: no grounded answer, flagged as KB gap"),
        }

    def skip(state: TurnState) -> TurnState:
        return {"outcome": "skipped"}

    g = StateGraph(TurnState)
    for name, fn in [
        ("track", track), ("triage", triage), ("retrieve", retrieve), ("rewrite", rewrite_with_context),
        ("focus", focus), ("suggest", suggest), ("done", done), ("escalate", escalate), ("skip", skip),
    ]:
        g.add_node(name, fn)
    g.set_entry_point("track")
    g.add_edge("track", "triage")
    g.add_conditional_edges("triage", route_triage, {"skip": "skip", "retrieve": "retrieve"})
    g.add_conditional_edges(
        "retrieve", grade_retrieval,
        {"suggest": "focus", "rewrite": "rewrite", "escalate": "escalate"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("focus", "suggest")
    g.add_conditional_edges("suggest", grade_suggestion, {"done": "done", "escalate": "escalate"})
    for terminal in ("done", "escalate", "skip"):
        g.add_edge(terminal, END)
    return g.compile()


class AssistAgent:
    def __init__(self, retriever: TfidfRetriever, suggester: Suggester,
                 playbooks: dict[str, Playbook] | None = None):
        self.playbooks = playbooks if playbooks is not None else playbooks_from_chunks(retriever.chunks)
        self._suggester = suggester
        self._graph = build_agent(retriever, suggester, self.playbooks)

    def on_customer_turn(self, utterance: str, history: list[str] | None = None,
                         call: CallState | None = None) -> AssistResult:
        """Run one customer line through the graph.

        ``call`` carries the active issue and step progress between turns;
        without it the line is handled on its own, as the eval does for single
        lines. ``history`` defaults to the call's earlier customer lines.
        """
        call = call if call is not None else CallState()
        history = history if history is not None else call.customer_lines
        state = self._graph.invoke({
            "utterance": utterance, "history": list(history), "call": call, "trace": [], "marks": [],
        })
        hits = state.get("hits") or []
        return AssistResult(
            outcome=state["outcome"],
            suggestion=state.get("suggestion"),
            trace=state.get("trace", []),
            top_score=hits[0].score if hits else 0.0,
            coverage=state.get("coverage", 0.0),
            plan=plan_view(self.playbooks, self._suggester, call, utterance),
            steps_marked=state.get("marks", []),
        )

    def on_agent_turn(self, text: str, call: CallState) -> AssistResult:
        """Agent lines get no card; they only move the checklist forward."""
        trace, marks = track_turn(self.playbooks, call, "agent", text)
        context = call.customer_lines[-1] if call.customer_lines else ""
        return AssistResult(
            outcome="none", suggestion=None, trace=trace,
            plan=plan_view(self.playbooks, self._suggester, call, context), steps_marked=marks,
        )

    def on_turn(self, speaker: str, text: str, call: CallState) -> AssistResult:
        """One turn of a call, either speaker. Updates ``call`` in place."""
        if speaker != "customer":
            return self.on_agent_turn(text, call)
        result = self.on_customer_turn(text, call=call)
        call.customer_lines.append(text)
        return result
