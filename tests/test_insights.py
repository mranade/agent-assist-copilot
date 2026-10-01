from assist.insights import compute, gap_words, group_gaps, main_article


def turn(text, outcome=None, kb_ids=(), feedback=None, speaker="customer"):
    return {"speaker": speaker, "text": text, "outcome": outcome, "kb_ids": list(kb_ids), "feedback": feedback}


def call(call_id, turns, score=0.0, label="neutral", started="2026-01-01T10:00:00"):
    return {
        "id": call_id, "title": f"Call {call_id}", "started_at": started, "turns": turns,
        "sentiment": {"score": score, "label": label},
    }


def lines(*texts):
    return [{"text": t, "call_id": f"c{i}", "turn": 1, "started_at": f"2026-01-0{i + 1}"} for i, t in enumerate(texts)]


def test_gap_words_are_the_words_the_kb_lacks(retriever):
    words = gap_words("Do you guys have a TV and streaming bundle I could add?", retriever.vocabulary)
    assert {"stream", "bundl"} <= set(words)
    assert "add" not in words, "the KB uses 'add', so it didn't cause the escalation"


def test_repeated_question_is_grouped_and_ranked_first(retriever):
    groups = group_gaps(lines(
        "Do you guys have a TV and streaming bundle I could add to lower the total cost?",
        "Can I put a security camera on my account?",
        "Do you have a streaming bundle, like TV channels with the internet?",
    ), retriever.vocabulary)
    assert [g["count"] for g in groups] == [2, 1]
    bundle = groups[0]
    assert bundle["label"] == "streaming · bundle", "labelled with the words the lines share, as said"
    assert bundle["calls"] == 2
    assert {e["call_id"] for e in bundle["examples"]} == {"c0", "c2"}, "examples link back to their calls"
    assert bundle["examples"][0]["call_id"] == "c2", "newest example first"


def test_one_shared_word_is_not_enough_to_group(retriever):
    groups = group_gaps(lines(
        "Do you have a streaming bundle with sports channels?",
        "Can you stream my security camera to my phone?",
    ), retriever.vocabulary)
    assert len(groups) == 2


def test_main_article_is_the_most_credited_one():
    record = call("x", [
        turn("a", "suggested", ["KB-109"]),
        turn("b", "suggested", ["KB-101", "KB-109"]),
        turn("c", "suggested", ["KB-101"]),
        turn("d", "suggested", ["KB-101"]),
    ])
    assert main_article(record) == "KB-101"
    assert main_article(call("y", [turn("hi", "skipped")])) is None


def test_totals_articles_and_topics(retriever):
    records = [
        call("a", [
            turn("Relay Fiber, how can I help?", speaker="agent"),
            turn("Hi there", "skipped"),
            turn("My internet keeps dropping", "suggested", ["KB-101"], "helpful"),
            turn("Can I get a credit?", "suggested", ["KB-101", "KB-109"], "not_helpful"),
            turn("Do you have a streaming bundle?", "escalated", feedback="helpful"),
        ], score=-0.4, label="negative"),
        call("b", [
            turn("Can you waive my late fee?", "suggested", ["KB-108"], "helpful"),
            turn("Thanks!", "skipped"),
            turn("Also my internet keeps dropping", "suggested", ["KB-101"]),
        ], score=0.6, label="positive"),
    ]
    out = compute(records, {"KB-101": "Drops", "KB-108": "Late fees", "KB-109": "Credits", "KB-120": "Unused"},
                  retriever.vocabulary)

    t = out["totals"]
    assert (t["calls"], t["customer_turns"], t["suggested"], t["escalated"], t["skipped"]) == (2, 7, 4, 1, 2)
    assert t["suggested_pct"] == round(4 / 7, 3)
    assert t["escalated_pct"] == round(1 / 7, 3)
    assert (t["cards"], t["rated"], t["helpful"]) == (5, 4, 3), "escalation cards can be rated too"
    assert t["helpful_pct"] == 0.75

    kb101 = next(a for a in out["articles"] if a["kb_id"] == "KB-101")
    assert (kb101["shown"], kb101["calls"], kb101["helpful"], kb101["not_helpful"]) == (3, 2, 1, 1)
    assert kb101["helpful_pct"] == 0.5
    assert kb101["avg_sentiment"] == 0.1, "mean of the two calls' scores"
    unused = next(a for a in out["articles"] if a["kb_id"] == "KB-120")
    assert unused["shown"] == 0 and unused["helpful_pct"] is None and unused["avg_sentiment"] is None
    assert out["articles"][0]["kb_id"] == "KB-101", "most shown first"

    topics = out["sentiment_by_topic"]
    assert [x["kb_id"] for x in topics] == ["KB-101", "KB-108"], "worst first"
    assert topics[0]["negative"] == 1

    assert [g["count"] for g in out["gaps"]] == [1]
    assert out["gaps"][0]["examples"][0] == {
        "text": "Do you have a streaming bundle?", "call_id": "a", "turn": 4,
        "call_title": "Call a", "started_at": "2026-01-01T10:00:00",
    }


def test_no_calls_means_no_percentages(retriever):
    t = compute([], {}, retriever.vocabulary)["totals"]
    assert t["calls"] == 0 and t["suggested_pct"] is None and t["helpful_pct"] is None
