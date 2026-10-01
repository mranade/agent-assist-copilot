"""HTTP API and static UI server.

    python -m assist.server            # then open http://localhost:8000

One process serves the React UI (built into web/dist) and the API the UI
calls. Calls in progress live in memory. Finished calls are saved to a
SQLite file (data/history/history.db) and survive restarts. On first start
the history is seeded with a few demo calls, produced by running scripted
transcripts through the same pipeline.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import AssistAgent
from . import insights
from .history import HistoryStore, apply_result, build_record, can_rate, now, seed
from .kb import load_articles
from .redact import redact
from .retriever import TfidfRetriever
from .sentiment import score_text
from . import stt
from .steps import CallState
from .stt import get_transcriber
from .suggest import ClaudeSuggester, ExtractiveSuggester, claude_available
from .summary import ClaudeSummarizer, TurnRecord

ROOT = Path(__file__).resolve().parents[2]
KB_DIR = ROOT / "data" / "kb"
CALLS_PATH = ROOT / "data" / "calls" / "calls.json"
AUDIO_DIR = ROOT / "data" / "calls" / "audio"
WEB_DIST = ROOT / "web" / "dist"
DB_PATH = Path(os.environ.get("ASSIST_DB", ROOT / "data" / "history" / "history.db"))

retriever = TfidfRetriever.from_kb(KB_DIR)
suggester = ClaudeSuggester() if claude_available() else ExtractiveSuggester()
agent = AssistAgent(retriever, suggester)
kb_titles = {a.kb_id: a.title for a in load_articles(KB_DIR)}
calls = {c["id"]: c for c in json.loads(CALLS_PATH.read_text(encoding="utf-8"))}
stt.start_warmup()  # loads Whisper in the background; the app works meanwhile
history = HistoryStore(DB_PATH)
if history.count() == 0 and not os.environ.get("ASSIST_NO_SEED"):
    seed(history, agent, kb_titles, ROOT)


class Session:
    def __init__(self, source: str):
        self.source = source
        self.started_at = now()
        self.turns: list[TurnRecord] = []
        self.call = CallState()  # active issue and checklist progress


sessions: dict[str, Session] = {}
ended: dict[str, str] = {}  # session id -> saved call id, so late feedback reaches the record

app = FastAPI(title="Relay Fiber Agent Assist")


class TextTurn(BaseModel):
    speaker: str = "customer"
    text: str


class PlaybackTurn(BaseModel):
    call_id: str
    index: int


class NewSession(BaseModel):
    source: str = "live"  # a sample call id, or "live"


class Feedback(BaseModel):
    value: Literal["helpful", "not_helpful"] | None  # None clears the rating


def _session(session_id: str) -> Session:
    if session_id not in sessions:
        raise HTTPException(404, "Unknown session")
    return sessions[session_id]


def _process(session_id: str, speaker: str, text: str, source: str, stt_ms: float) -> dict:
    """Run one turn. Customer turns can get a card; agent turns only advance the
    checklist. Both return the current plan so the assist panel can update."""
    session = _session(session_id)
    clean = redact(text)
    record = TurnRecord(speaker=speaker, text=clean)
    start = time.perf_counter()
    result = agent.on_turn(speaker, clean, session.call)
    assist_ms = round((time.perf_counter() - start) * 1000)
    apply_result(record, result)
    response: dict = {
        "index": len(session.turns),  # the turn's position in the call, for feedback
        "speaker": speaker, "text": clean, "transcript_source": source,
        "stt_ms": round(stt_ms), "outcome": None, "suggestion": None,
        "trace": result.trace,
        "plan": record.plan,
        "steps_marked": result.steps_marked,
    }
    if speaker == "customer":
        response.update(
            sentiment=round(score_text(clean), 3),
            outcome=result.outcome,
            suggestion=asdict(result.suggestion) if result.suggestion else None,
            top_score=round(result.top_score, 3),
            coverage=round(result.coverage, 2),
            assist_ms=assist_ms,
        )
    session.turns.append(record)
    return response


@app.get("/api/status")
def status() -> dict:
    speech = stt.status()
    return {
        "stt_available": speech["state"] == "ready",
        "stt_state": speech["state"],        # off | loading | ready | failed
        "stt_error": speech["error"],
        "stt_model": speech["model"],
        "generator": suggester.name,
        "kb_articles": len(kb_titles),
    }


@app.get("/api/calls")
def list_calls() -> list[dict]:
    return [
        {
            "id": c["id"], "title": c["title"], "description": c["description"],
            "turns": [
                {"index": i, "speaker": t["speaker"], "audio_url": f"/audio/{c['id']}/{i:02d}.wav"}
                for i, t in enumerate(c["turns"])
            ],
        }
        for c in calls.values()
    ]


@app.post("/api/sessions")
def create_session(body: NewSession | None = None) -> dict:
    session_id = uuid.uuid4().hex[:12]
    sessions[session_id] = Session((body or NewSession()).source)
    return {"session_id": session_id}


@app.post("/api/sessions/{session_id}/turns/text")
def text_turn(session_id: str, body: TextTurn) -> dict:
    return _process(session_id, body.speaker, body.text, "typed", 0)


@app.post("/api/sessions/{session_id}/turns/playback")
def playback_turn(session_id: str, body: PlaybackTurn) -> dict:
    call = calls.get(body.call_id)
    if not call or not 0 <= body.index < len(call["turns"]):
        raise HTTPException(404, "Unknown call turn")
    turn = call["turns"][body.index]
    audio = AUDIO_DIR / body.call_id / f"{body.index:02d}.wav"
    transcriber = get_transcriber()
    start = time.perf_counter()
    text, source = turn["text"], "script"
    if transcriber and audio.exists():
        try:
            text, source = transcriber.transcribe(audio), "whisper"
        except Exception as exc:  # e.g. ffmpeg missing: keep the demo moving
            stt.log.warning("Transcription failed, using script text: %s", exc)
    stt_ms = (time.perf_counter() - start) * 1000 if source == "whisper" else 0
    return _process(session_id, turn["speaker"], text, source, stt_ms)


@app.post("/api/sessions/{session_id}/turns/audio")
async def audio_turn(session_id: str, file: UploadFile = File(...)) -> dict:
    transcriber = get_transcriber()
    if transcriber is None:
        state = stt.status()
        messages = {
            "loading": "Whisper is still loading (the first run downloads the model). Try again in a moment.",
            "failed": f"Whisper failed to load: {state['error']}",
            "off": "Speech-to-text is not installed. Run: pip install -e \".[speech]\"",
        }
        raise HTTPException(503, messages.get(state["state"], "Speech-to-text is unavailable."))
    suffix = Path(file.filename or "clip.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = Path(tmp.name)
    try:
        start = time.perf_counter()
        text = transcriber.transcribe(path)
        stt_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        raise HTTPException(500, f"Transcription failed: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)
    if not text:
        raise HTTPException(422, "No speech detected in the recording.")
    return _process(session_id, "customer", text, "whisper", stt_ms)


@app.post("/api/sessions/{session_id}/end")
def end_call(session_id: str) -> dict:
    """Finish the call: summarize, score sentiment, and save it to history."""
    session = _session(session_id)
    if not any(t.speaker == "customer" for t in session.turns):
        raise HTTPException(422, "Nothing to save: the call has no customer turns yet.")
    record = build_record(
        session.turns, kb_titles,
        source=session.source,
        started_at=session.started_at,
        ended_at=now(),
        summarizer=ClaudeSummarizer() if claude_available() else None,
    )
    history.save(record)
    del sessions[session_id]
    ended[session_id] = record["id"]
    return record


@app.post("/api/sessions/{session_id}/turns/{index}/feedback")
def session_feedback(session_id: str, index: int, body: Feedback) -> dict:
    """The agent rates a card during the call. After the call is saved, the
    same request updates the saved record instead."""
    if session_id in ended:
        return history_feedback(ended[session_id], index, body)
    session = _session(session_id)
    if not 0 <= index < len(session.turns):
        raise HTTPException(404, "Unknown turn")
    turn = session.turns[index]
    if not can_rate(turn.speaker, turn.outcome):
        raise HTTPException(422, "That turn had no assist card to rate.")
    turn.feedback = body.value
    return {"index": index, "feedback": turn.feedback}


@app.post("/api/history/{call_id}/turns/{index}/feedback")
def history_feedback(call_id: str, index: int, body: Feedback) -> dict:
    try:
        turn = history.set_feedback(call_id, index, body.value)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"index": index, "feedback": turn["feedback"]}


@app.get("/api/history")
def list_history() -> list[dict]:
    return history.list()


@app.get("/api/history/{call_id}")
def get_history(call_id: str) -> dict:
    record = history.get(call_id)
    if record is None:
        raise HTTPException(404, "Unknown call")
    return record


@app.get("/api/insights")
def get_insights() -> dict:
    return insights.compute(history.records(), kb_titles, retriever.vocabulary)


app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIST / "index.html")


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    print(f"Agent Assist running at http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
