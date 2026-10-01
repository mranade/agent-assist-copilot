from assist.kb import load_articles, load_chunks
from assist.redact import redact
from assist.text import content_tokens

from conftest import KB_DIR


def test_kb_loads_all_articles_with_front_matter():
    articles = load_articles(KB_DIR)
    assert len(articles) == 20
    assert all(a.kb_id.startswith("KB-") and a.title and a.category for a in articles)


def test_chunks_carry_their_article_title():
    for chunk in load_chunks(KB_DIR):
        assert chunk.text.startswith(chunk.title)


def test_customer_words_map_to_kb_words():
    # "router" should become the KB's "gateway"; stemming unifies "cancelling"/"cancellation".
    assert content_tokens("my router") == content_tokens("my gateway")
    assert content_tokens("cancelling") == content_tokens("cancellation")


def test_filler_words_are_dropped():
    assert content_tokens("honestly guys, okay, thanks") == []


def test_redacts_pins_and_account_numbers():
    text = redact("It's 4821 and my PIN is 7090, call me at 555 1234")
    assert "4821" not in text and "7090" not in text and "1234" not in text
    assert text.count("[REDACTED]") == 3


def test_redaction_leaves_short_numbers_alone():
    assert redact("the fee is $10 for 2 days") == "the fee is $10 for 2 days"
