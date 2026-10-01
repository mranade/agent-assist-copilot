from assist.agent import AssistAgent
from assist.suggest import Suggestion, is_grounded


def test_small_talk_produces_no_card(agent):
    result = agent.on_customer_turn("Hi Sam, it's Dana Ortiz.")
    assert result.outcome == "skipped"
    assert result.suggestion is None


def test_problem_statement_is_not_mistaken_for_small_talk(agent):
    # Regression: "short and not a question" used to be skipped.
    result = agent.on_customer_turn("the internet is completely down")
    assert result.outcome != "skipped"


def test_in_scope_question_gets_the_right_article(agent):
    result = agent.on_customer_turn("I got charged a late fee, can you waive it?")
    assert result.outcome == "suggested"
    assert result.suggestion.articles[0].kb_id == "KB-108"
    assert result.suggestion.lines


def test_customer_vocabulary_is_bridged(agent):
    result = agent.on_customer_turn("If I cancel, what do I do with the router you gave me?")
    assert result.outcome == "suggested"
    assert {"KB-113", "KB-114"} & {a.kb_id for a in result.suggestion.articles}


def test_out_of_scope_question_escalates_after_one_retry(agent):
    history = ["I'm thinking about cancelling because the price is kind of high."]
    result = agent.on_customer_turn("Do you have a TV and streaming bundle?", history)
    assert result.outcome == "escalated"
    retrieves = [step for step in result.trace if step.startswith("retrieve")]
    assert len(retrieves) == 2, "should retry once with the previous turn as context"


def test_borrowed_context_cannot_make_off_topic_question_answerable(agent):
    # The retry prepends a very on-topic previous turn; coverage is still
    # judged on the new utterance, so it must still escalate.
    history = ["How do I get a late fee waived on my bill?"]
    result = agent.on_customer_turn("Do you sell home security cameras?", history)
    assert result.outcome == "escalated"


def test_extractive_lines_are_verbatim_kb_text(agent, retriever):
    result = agent.on_customer_turn("How much are your plans?")
    kb_text = " ".join(c.text for c in retriever.chunks)
    for line in result.suggestion.lines:
        assert line in kb_text


def test_generated_lines_must_cite_a_retrieved_article(retriever):
    hits = retriever.search("late fee waiver")
    ok = Suggestion("x", ["You can waive one late fee a year [KB-108]"], [], "claude")
    uncited = Suggestion("x", ["You can waive one late fee a year"], [], "claude")
    wrong_cite = Suggestion("x", ["Try a factory reset [KB-105]"], [], "claude")
    assert is_grounded(ok, hits)
    assert not is_grounded(uncited, hits)
    assert not is_grounded(wrong_cite, hits)


def test_ungrounded_generation_escalates(retriever):
    class MadeUpSuggester:
        name = "claude"

        def suggest(self, utterance, hits, plan=None, checklists=frozenset()):
            return Suggestion("x", ["Everyone gets a free month [KB-999]"], [], "claude")

        def say(self, step, kb_id, context):
            return "Everyone gets a free month [KB-999]", "claude"

    result = AssistAgent(retriever, MadeUpSuggester()).on_customer_turn("Can you waive my late fee?")
    assert result.outcome == "escalated"
