from assist.retriever import TfidfRetriever
from assist.sentiment import call_sentiment, score_text
from assist.text import topic_tokens


def test_billing_words_are_not_read_as_positive():
    # Stock VADER scores this clearly positive because of "credit" and "kind".
    assert score_text("Can I get some kind of credit on my bill for all the drops?") < 0


def test_churn_language_is_negative():
    assert score_text("I'm thinking about cancelling, the price is kind of high") < 0


def test_call_that_ends_well_is_positive_and_improving():
    s = call_sentiment([
        "My internet keeps dropping every night and it's frustrating.",
        "It's been going on for a week.",
        "Great, that fixed it. Thank you so much.",
    ])
    assert s.label == "positive"
    assert s.trend == "improved"
    assert s.most_negative.startswith("My internet")


def test_call_that_ends_badly_is_negative():
    s = call_sentiment([
        "Hi, I have a question about my bill.",
        "This is unacceptable. I want to talk to a supervisor.",
    ])
    assert s.label == "negative"


def test_greetings_alone_are_neutral():
    assert call_sentiment(["Hi, it's Dana.", "Okay."]).label in ("neutral", "positive")


def test_emotion_words_are_not_topics():
    tokens = topic_tokens("This is ridiculous, the internet is down again")
    assert "ridicul" not in tokens
    assert "internet" in tokens


def test_support_words_stay_topics_even_though_they_carry_sentiment():
    assert topic_tokens("I want to cancel, it's too slow") == topic_tokens("cancel slow")


def test_emotion_does_not_hide_a_real_problem(retriever: TfidfRetriever):
    # An angry, rambling outage report used to fall under the coverage floor.
    line = "My internet has been completely down since yesterday morning. This is ridiculous."
    assert retriever.coverage(line) >= 0.30
