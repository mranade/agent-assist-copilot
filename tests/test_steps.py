"""Active issue, checklist tracking, and the structured card."""

from assist.kb import load_chunks
from assist.steps import CallState, match_step, playbooks_from_chunks, say_for_step

from conftest import KB_DIR

DROPS = "My internet keeps dropping every night and my laptop loses the Wi-Fi."
CREDIT = "This has been going on for a week. Can I get some kind of credit on my bill for all the drops?"


def start_drops_call(agent):
    call = CallState()
    result = agent.on_turn("customer", DROPS, call)
    return call, result


def test_troubleshooting_articles_have_a_checklist():
    books = playbooks_from_chunks(load_chunks(KB_DIR))
    for kb_id in ("KB-101", "KB-102", "KB-103", "KB-108", "KB-112", "KB-113", "KB-115"):
        assert len(books[kb_id].steps) >= 3, kb_id
    assert "KB-109" not in books, "a policy article has no checklist"
    assert books["KB-101"].links == {"KB-105", "KB-115"}


def test_card_starts_at_the_first_step_not_the_likely_causes(agent):
    _, result = start_drops_call(agent)
    plan = result.suggestion.plan
    assert plan.kb_id == "KB-101"
    assert plan.next == 0
    assert plan.steps[0].text.startswith("Run a line check")
    assert result.suggestion.lines[0] == plan.steps[0].text, "the next step is verbatim KB text"


def test_agent_turn_marks_a_step_done_without_getting_a_card(agent):
    call, _ = start_drops_call(agent)
    result = agent.on_turn("agent", "Let me run a line check on your account. No line errors right now.", call)
    assert result.suggestion is None
    assert [m["step"] for m in result.steps_marked] == [1]
    assert result.plan.steps[0].done and result.plan.steps[0].done_by == "agent"
    assert any(t.startswith("track: KB-101 step 1 of 5 done") for t in result.trace)


def test_customer_answer_completes_the_question_step_and_card_advances(agent):
    call, _ = start_drops_call(agent)
    agent.on_turn("agent", "Let me run a line check on your account.", call)
    result = agent.on_turn("customer", "Only on Wi-Fi. None of my devices are wired.", call)
    assert result.steps_marked and result.steps_marked[0]["step"] == 2
    assert result.steps_marked[0]["by"] == "customer"
    plan = result.suggestion.plan
    assert plan.next == 2, "the card now shows step 3"
    assert "5 GHz" in plan.say


def test_customer_cannot_complete_a_step_the_agent_has_to_do(agent):
    # "Run a line check" is the agent's job, however many words the line shares with it.
    call, _ = start_drops_call(agent)
    book = agent.playbooks["KB-101"]
    assert match_step(book, call.done, "customer", "Did you run a line check on my account?") is None
    assert match_step(book, call.done, "agent", "I ran a line check on your account.").step == 0


def test_active_issue_stays_on_screen_through_a_side_question(agent):
    call, _ = start_drops_call(agent)
    agent.on_turn("agent", "Let me run a line check on your account.", call)
    result = agent.on_turn("customer", CREDIT, call)
    s = result.suggestion
    assert result.outcome == "suggested"
    assert s.plan.kb_id == "KB-101" and s.plan.next == 1, "the troubleshooting plan is still there"
    assert s.note.kb_id == "KB-109" and s.note.kind == "Policy"
    assert any("goodwill credit" in line for line in s.note.lines)
    assert call.issue == "KB-101"


def test_linked_article_stays_a_note(agent):
    # KB-101 step 5 points to KB-115, so a technician question is part of the same issue.
    call, _ = start_drops_call(agent)
    result = agent.on_turn("customer", "Will I have to pay a fee for the technician visit?", call)
    assert call.issue == "KB-101"
    assert result.suggestion.note.kb_id == "KB-115"
    assert any("links to KB-115" in t for t in result.trace)


def test_a_clearly_new_issue_replaces_the_active_one(agent):
    call, _ = start_drops_call(agent)
    agent.on_turn("agent", "Let me run a line check on your account.", call)
    result = agent.on_turn("customer", "I'm moving to a new address next month, can I take my service with me?", call)
    assert call.issue == "KB-112"
    assert result.suggestion.plan.kb_id == "KB-112"
    assert result.suggestion.plan.next == 0, "a new issue starts its own checklist"
    assert any("KB-112 replaces KB-101" in t for t in result.trace)


def test_escalation_keeps_the_plan_visible(agent):
    call, _ = start_drops_call(agent)
    result = agent.on_turn("customer", "Do you sell home security cameras?", call)
    assert result.outcome == "escalated"
    assert result.suggestion.plan.kb_id == "KB-101"


def test_articles_without_a_checklist_keep_the_lines_card(agent):
    result = agent.on_turn("customer", "Is there a discount for autopay?", CallState())
    assert result.suggestion.plan is None
    assert result.suggestion.lines


def test_say_line_only_adds_framing_words():
    assert say_for_step("Ask whether drops happen on wired devices too. Wired drops point to the line.") == (
        "Can you tell me whether drops happen on wired devices too?")
    assert say_for_step("Power-cycle the gateway (see KB-105) and confirm the firmware updated automatically.") == (
        "Let's power-cycle the gateway and confirm the firmware updated automatically.")
    assert say_for_step("If drops continue after these steps, schedule a technician visit (see KB-115). Log it.") == (
        "I'll schedule a technician visit.")
    assert say_for_step("Tell them they will get a text 30 minutes before the technician arrives.") == (
        "You will get a text 30 minutes before the technician arrives.")
