import json

import pytest
from fastapi.testclient import TestClient

from conftest import ROOT

CALLS = {c["id"]: c for c in json.loads((ROOT / "data" / "calls" / "calls.json").read_text(encoding="utf-8"))}


@pytest.fixture(scope="module")
def client():
    from assist.server import app

    return TestClient(app)


def new_session(client):
    return client.post("/api/sessions").json()["session_id"]


def test_status_reports_offline_modes(client):
    status = client.get("/api/status").json()
    assert status["stt_available"] is False
    assert status["generator"] == "extractive"
    assert status["kb_articles"] == 20


def test_playback_falls_back_to_script_text_without_whisper(client):
    sid = new_session(client)
    turn = client.post(f"/api/sessions/{sid}/turns/playback", json={"call_id": "call-01", "index": 3}).json()
    assert turn["transcript_source"] == "script"
    assert turn["outcome"] == "suggested"


def test_spoken_pin_is_redacted_before_storage(client):
    sid = new_session(client)
    turn = client.post(
        f"/api/sessions/{sid}/turns/text", json={"text": "It's 4821 and my PIN is 7090"}
    ).json()
    assert "4821" not in turn["text"] and "7090" not in turn["text"]


def test_live_audio_needs_whisper(client):
    sid = new_session(client)
    resp = client.post(f"/api/sessions/{sid}/turns/audio", files={"file": ("a.webm", b"x")})
    assert resp.status_code == 503


def test_live_audio_path_with_a_stub_transcriber(client, monkeypatch):
    import assist.server as server

    class StubTranscriber:
        def transcribe(self, path):
            return "Can you waive the late fee on my bill?"

    monkeypatch.setattr(server, "get_transcriber", lambda: StubTranscriber())
    sid = new_session(client)
    turn = client.post(f"/api/sessions/{sid}/turns/audio", files={"file": ("a.webm", b"x")}).json()
    assert turn["transcript_source"] == "whisper"
    assert turn["suggestion"]["articles"][0]["kb_id"] == "KB-108"


def test_playback_survives_a_transcription_crash(client, monkeypatch):
    # e.g. Whisper loaded but ffmpeg is missing: the demo must keep going.
    import assist.server as server

    class BrokenTranscriber:
        def transcribe(self, path):
            raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(server, "get_transcriber", lambda: BrokenTranscriber())
    sid = new_session(client)
    resp = client.post(f"/api/sessions/{sid}/turns/playback", json={"call_id": "call-01", "index": 3})
    assert resp.status_code == 200
    assert resp.json()["transcript_source"] == "script"


def test_mic_explains_when_whisper_is_still_loading(client, monkeypatch):
    import assist.server as server

    monkeypatch.setattr(server.stt, "status", lambda: {"state": "loading", "error": None, "model": "base.en"})
    sid = new_session(client)
    resp = client.post(f"/api/sessions/{sid}/turns/audio", files={"file": ("a.webm", b"x")})
    assert resp.status_code == 503
    assert "still loading" in resp.json()["detail"]


def test_whisper_load_failure_is_reported_not_raised(monkeypatch):
    import assist.stt as stt

    monkeypatch.setattr(stt, "_state", {"state": "loading", "error": None})

    def boom(size):
        raise OSError("model download blocked")

    monkeypatch.setattr(stt, "WhisperTranscriber", boom)
    stt._load()  # what the background thread runs
    assert stt.status()["state"] == "failed"
    assert "download blocked" in stt.status()["error"]
    assert stt.get_transcriber() is None


def play_call(client, call_id):
    sid = client.post("/api/sessions", json={"source": call_id}).json()["session_id"]
    for i in range(len(CALLS[call_id]["turns"])):
        client.post(f"/api/sessions/{sid}/turns/playback", json={"call_id": call_id, "index": i})
    return client.post(f"/api/sessions/{sid}/end").json()


def test_ending_a_call_saves_summary_and_sentiment(client):
    record = play_call(client, "call-03")
    assert any(a.startswith("KB-112") for a in record["summary"]["articles_used"])
    assert any("TV" in item for item in record["summary"]["open_items"])
    assert record["sentiment"]["label"] == "negative"
    assert record["source"] == "call-03"


def test_summary_credits_articles_whose_lines_were_shown(client):
    # The credit request's card keeps the KB-101 plan and adds a KB-109
    # (service credits) note; the summary should mention both.
    used = play_call(client, "call-01")["summary"]["articles_used"]
    assert any(a.startswith("KB-101") for a in used)
    assert any(a.startswith("KB-109") for a in used)


def test_finished_call_appears_in_history(client):
    record = play_call(client, "call-02")
    listed = client.get("/api/history").json()
    assert listed[0]["id"] == record["id"], "newest call first"
    detail = client.get(f"/api/history/{record['id']}").json()
    assert len(detail["turns"]) == len(CALLS["call-02"]["turns"])
    assert all("4821" not in t["text"] for t in detail["turns"]), "stored transcript is redacted"
    customer = [t for t in detail["turns"] if t["speaker"] == "customer"]
    assert all(t["sentiment"] is not None for t in customer)
    assert any(t["lines"] for t in customer), "the assist cards are kept with the call"
    assert any(t["card"] and t["card"]["plan"] for t in customer), "including the checklist"
    agent_turns = [t for t in detail["turns"] if t["speaker"] == "agent"]
    assert any(t["steps_marked"] for t in agent_turns), "steps the agent completed are kept too"


def test_agent_turn_advances_the_plan_without_a_card(client):
    sid = new_session(client)
    for i in range(4):  # greeting, name, "how can I help", the drops
        client.post(f"/api/sessions/{sid}/turns/playback", json={"call_id": "call-01", "index": i})
    turn = client.post(f"/api/sessions/{sid}/turns/playback", json={"call_id": "call-01", "index": 4}).json()
    assert turn["speaker"] == "agent"
    assert turn["suggestion"] is None and turn["outcome"] is None
    assert turn["steps_marked"][0]["step"] == 1
    assert turn["plan"]["kb_id"] == "KB-101" and turn["plan"]["next"] == 1


def test_sample_call_ends_with_the_checklist_done_and_the_credit_noted(client):
    record = play_call(client, "call-01")
    detail = client.get(f"/api/history/{record['id']}").json()
    last_card = [t["card"] for t in detail["turns"] if t.get("card")][-1]
    assert last_card["plan"]["kb_id"] == "KB-101"
    assert all(s["done"] for s in last_card["plan"]["steps"]), "technician visit booked"
    assert last_card["note"]["kb_id"] == "KB-109", "credit policy on the card"
    assert "5 of 5 steps done" in record["summary"]["note"]


def test_history_is_seeded_on_first_start(client):
    sources = {c["source"] for c in client.get("/api/history").json()}
    assert {"past-outage", "past-upgrade", "past-technician", "call-01"} <= sources


def test_empty_call_is_not_saved(client):
    sid = new_session(client)
    assert client.post(f"/api/sessions/{sid}/end").status_code == 422


def test_unknown_session_and_call_are_404(client):
    assert client.post("/api/sessions/nope/end").status_code == 404
    assert client.get("/api/history/nope").status_code == 404


def rate(client, sid, index, value):
    return client.post(f"/api/sessions/{sid}/turns/{index}/feedback", json={"value": value})


def test_feedback_during_the_call_is_saved_with_the_record(client):
    sid = new_session(client)
    turns = [client.post(f"/api/sessions/{sid}/turns/playback", json={"call_id": "call-01", "index": i}).json()
             for i in range(4)]
    card = turns[3]
    assert card["outcome"] == "suggested" and card["index"] == 3
    assert rate(client, sid, 3, "not_helpful").json() == {"index": 3, "feedback": "not_helpful"}
    record = client.post(f"/api/sessions/{sid}/end").json()
    assert record["turns"][3]["feedback"] == "not_helpful"
    assert record["turns"][1].get("feedback") is None, "skipped small talk has no card to rate"


def test_feedback_after_the_call_updates_the_saved_record(client):
    sid = new_session(client)
    for i in range(4):
        client.post(f"/api/sessions/{sid}/turns/playback", json={"call_id": "call-01", "index": i})
    record = client.post(f"/api/sessions/{sid}/end").json()

    # The live view may still send the rating to the (now ended) session...
    assert rate(client, sid, 3, "helpful").status_code == 200
    assert client.get(f"/api/history/{record['id']}").json()["turns"][3]["feedback"] == "helpful"
    # ...and the call history sends it to the saved call. null clears it.
    resp = client.post(f"/api/history/{record['id']}/turns/3/feedback", json={"value": None})
    assert resp.json() == {"index": 3, "feedback": None}
    assert client.get(f"/api/history/{record['id']}").json()["turns"][3]["feedback"] is None


def test_feedback_is_validated(client):
    sid = new_session(client)
    for i in range(4):
        client.post(f"/api/sessions/{sid}/turns/playback", json={"call_id": "call-01", "index": i})
    assert rate(client, sid, 0, "helpful").status_code == 422, "agent turns have no card"
    assert rate(client, sid, 1, "helpful").status_code == 422, "small talk has no card"
    assert rate(client, sid, 3, "great").status_code == 422, "only helpful / not_helpful / null"
    assert rate(client, sid, 99, "helpful").status_code == 404
    assert rate(client, "nope", 0, "helpful").status_code == 404
    assert client.post("/api/history/nope/turns/0/feedback", json={"value": "helpful"}).status_code == 404


def test_insights_show_the_seeded_tv_bundle_gap_and_ratings(client):
    data = client.get("/api/insights").json()
    top = data["gaps"][0]
    assert "streaming" in top["label"] and "bundle" in top["label"]
    assert top["count"] >= 2 and top["calls"] >= 2, "asked on more than one call"
    assert data["totals"]["rated"] > 0, "seeded calls come with feedback"
    kb118 = next(a for a in data["articles"] if a["kb_id"] == "KB-118")
    assert kb118["not_helpful"] >= 1, "the wrong 'what plans do you have' card was voted down"


def test_insights_update_when_a_call_is_rated(client):
    before = client.get("/api/insights").json()["totals"]
    sid = new_session(client)
    for i in range(4):
        client.post(f"/api/sessions/{sid}/turns/playback", json={"call_id": "call-01", "index": i})
    rate(client, sid, 3, "not_helpful")
    client.post(f"/api/sessions/{sid}/end")
    after = client.get("/api/insights").json()["totals"]
    assert after["calls"] == before["calls"] + 1
    assert after["rated"] == before["rated"] + 1
    assert after["helpful"] == before["helpful"]
