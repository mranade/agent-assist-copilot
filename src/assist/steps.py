"""Checklists and per-call progress through them.

Troubleshooting and process articles have a numbered "Steps for the agent"
list. A card that shows the *next* step of that list tells the agent what to
do; a card that shows the three sentences sharing the most words with the
customer's last line often doesn't (it tends to surface "Likely causes").

This module:

* parses each article's checklist (its "Steps for the agent" section, or
  else its first numbered list),
* keeps the state of one call (``CallState``): which article is the active
  issue and which of its steps are done,
* marks a step done when a turn shares enough of the step's words, and
* turns a step into a short line the agent could say (``say_for_step``).

Step tracking is deliberately simple and explainable. A turn completes a
step when it shares at least ``MIN_SHARED`` topic words with it, not counting
words that appear in more than half of the checklist's steps ("drops" is in
most KB-101 steps, so it says nothing about *which* step was done). Only an
agent turn can perform a step; a customer turn can only answer a step that
asks something ("Ask whether drops happen on wired devices too" is answered
by "it's only on Wi-Fi, nothing is wired"). One turn completes at most one
step: the first pending step, in checklist order, that clears the bar. Agents
mostly work top to bottom, and "Can I ask why you're leaving? Is it the
price?" should complete "Ask why they are leaving", not the retention offer
that happens to mention price too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .kb import Chunk
from .text import WORD_RE, apply_synonyms, stem, topic_tokens

STEP_SECTION = "Steps for the agent"
MIN_SHARED = 2

NUMBERED_RE = re.compile(r"^\s*\d+\.\s+(.*)$")
KB_REF_RE = re.compile(r"KB-\d{3}")


@dataclass(frozen=True)
class Playbook:
    kb_id: str
    title: str
    steps: tuple[str, ...]
    links: frozenset[str]  # other articles the steps point to, e.g. KB-101 step 5 -> KB-115

    def keys(self) -> list[frozenset[str]]:
        """Each step's topic words, minus words used by more than half of the steps."""
        per_step = [frozenset(topic_tokens(s)) for s in self.steps]
        counts: dict[str, int] = {}
        for tokens in per_step:
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
        common = {t for t, n in counts.items() if n * 2 > len(self.steps)}
        return [tokens - common for tokens in per_step]


def parse_steps(text: str) -> list[str]:
    """Items of the first numbered list in ``text``; wrapped lines are joined."""
    steps: list[str] = []
    for line in text.splitlines():
        m = NUMBERED_RE.match(line)
        if m:
            steps.append(m.group(1).strip())
        elif steps and line.strip():
            if line.startswith((" ", "\t")):
                steps[-1] += " " + line.strip()
            else:
                break  # a paragraph after the list ends it
    return steps


def playbooks_from_chunks(chunks: list[Chunk]) -> dict[str, Playbook]:
    by_article: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_article.setdefault(c.kb_id, []).append(c)
    books: dict[str, Playbook] = {}
    for kb_id, article_chunks in by_article.items():
        preferred = [c for c in article_chunks if c.section == STEP_SECTION]
        for chunk in preferred + article_chunks:
            steps = parse_steps(chunk.text.split("\n", 1)[-1])
            if steps:
                links = {k for s in steps for k in KB_REF_RE.findall(s)} - {kb_id}
                books[kb_id] = Playbook(kb_id, chunk.title, tuple(steps), frozenset(links))
                break
    return books


@dataclass
class StepMark:
    step: int          # 0-based index into the playbook's steps
    speaker: str       # "agent" or "customer"
    words: list[str]   # the turn's words that matched the step, for the trace
    turn: str


@dataclass
class CallState:
    """What the assist panel remembers across the turns of one call."""
    issue: str | None = None                        # KB id of the active issue
    done: dict[int, StepMark] = field(default_factory=dict)
    customer_lines: list[str] = field(default_factory=list)
    say_cache: dict[int, tuple[str, str]] = field(default_factory=dict)

    def set_issue(self, kb_id: str) -> None:
        self.issue = kb_id
        self.done = {}
        self.say_cache = {}


def _matched_words(text: str, stems: set[str]) -> list[str]:
    out: list[str] = []
    for w in WORD_RE.findall(apply_synonyms(text)):
        if stem(w) in stems and w not in out:
            out.append(w)
    return out


def match_step(book: Playbook, done: dict[int, StepMark], speaker: str, text: str) -> StepMark | None:
    """The pending step this turn completes, if any (see the module docstring)."""
    words = set(topic_tokens(text))
    for i, (step, keys) in enumerate(zip(book.steps, book.keys())):
        if i in done:
            continue
        if speaker == "customer" and not step.lower().startswith("ask"):
            continue
        shared = set(keys) & words
        if len(shared) >= MIN_SHARED:
            return StepMark(i, speaker, _matched_words(text, shared), text)
    return None


def describe_mark(book: Playbook, mark: StepMark) -> str:
    how = "agent did it" if mark.speaker == "agent" else "customer answered it"
    words = ", ".join(f'"{w}"' for w in mark.words)
    return (
        f"track: {book.kb_id} step {mark.step + 1} of {len(book.steps)} done ({how}); "
        f"the turn shares {words} with the step (needs {MIN_SHARED})"
    )


# --- Turning a step into something the agent could say ----------------------

AGENT_VERBS = {
    "run", "check", "schedule", "email", "send", "apply", "waive", "book",
    "log", "push", "process", "note", "look",
}

_YOU = [
    (r"\bthe (?:customer|caller)'s\b", "your"),
    (r"\bthe (?:customer|caller)\b", "you"),
    (r"\btheir\b", "your"),
    (r"\bthey\b", "you"),
    (r"\bthem\b", "you"),
]

_TEMPLATES = [
    (r"^ask (?:you )?(whether|if|what|why|how|when|where|which)\s+(.*)$", "Can you tell me {0} {1}?"),
    (r"^ask (?:you )?for (.*)$", "Could I get {0}?"),
    (r"^ask you to (.*)$", "Could you {0}?"),
    (r"^have you (.*)$", "Could you {0}?"),
    (r"^(?:tell|explain to) you (?:that )?(.*)$", "{0}."),
    (r"^explain (?:that )?(.*)$", "{0}."),
    (r"^remind you to (.*)$", "Please remember to {0}."),
    (r"^confirm (?:that )?(.*)$", "Just to confirm, {0}."),
    (r"^offer to (.*)$", "Would you like me to {0}?"),
    (r"^offer (?:you )?(.*)$", "I can offer you {0}."),
    (r"^recommend (.*)$", "I'd recommend {0}."),
]


def _lower_first(s: str) -> str:
    first = s.split(" ", 1)[0]
    if any(ch.isupper() for ch in first[1:]):
        return s  # "Wi-Fi", "WAN"
    return s[:1].lower() + s[1:]


def _upper_first(s: str) -> str:
    return s[:1].upper() + s[1:]


def say_for_step(step: str) -> str:
    """A short line for the agent to say, built from the step text by template.

    Only framing words are added ("Could you ...?", "I'll ..."); every fact
    comes from the step itself. The first sentence is used, parentheticals
    such as "(see KB-105)" are dropped, and a leading condition ("For
    Wi-Fi-only drops, ...") is left to the agent, who can see the full step.
    """
    s = re.sub(r"\s*\([^)]*\)", "", step).strip()
    s = re.split(r"(?<=[.!?])\s+", s)[0].rstrip(".")
    s = re.sub(r"^(?:if|for|when|once)\b[^,]*,\s*", "", s, flags=re.IGNORECASE)
    for pattern, repl in _YOU:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    for pattern, template in _TEMPLATES:
        m = re.match(pattern, s, flags=re.IGNORECASE)
        if m:
            groups = [g for g in m.groups()]
            out = template.format(*groups)
            return _upper_first(out)
    s = _lower_first(s)
    verb = s.split(" ", 1)[0]
    return f"I'll {s}." if verb in AGENT_VERBS else f"Let's {s}."
