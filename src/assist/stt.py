"""Speech-to-text with a local Whisper model (faster-whisper).

Runs on the laptop, with no audio sent to a third party, which matters for
contact-center recordings. The model (``base.en`` by default, about 150 MB)
downloads the first time it's used. Set ``ASSIST_WHISPER_MODEL`` to
``tiny.en`` for speed or ``small.en`` for accuracy.

The model loads in a background thread when the server starts, so a slow
first download never blocks the app. Until it's ready, or if it fails to load
(no internet, missing ffmpeg), ``get_transcriber()`` returns None and the app
falls back: playback mode uses the call script's text and says so in the UI,
and live-mic mode explains why it's unavailable. A demo never stalls on
speech-to-text.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path

log = logging.getLogger("assist.stt")

_lock = threading.Lock()
_state = {"state": "off", "error": None}  # off | loading | ready | failed
_transcriber = None


class WhisperTranscriber:
    def __init__(self, model_size: str):
        from faster_whisper import WhisperModel

        self.model_size = model_size
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_path: Path) -> str:
        # Browsers record webm/ogg; convert everything to 16 kHz mono WAV first.
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_path),
                 "-ar", "16000", "-ac", "1", str(wav)],
                check=True,
            )
            segments, _ = self._model.transcribe(str(wav), language="en", beam_size=1)
            return " ".join(s.text.strip() for s in segments).strip()


def model_size() -> str:
    return os.environ.get("ASSIST_WHISPER_MODEL", "base.en")


def whisper_installed() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def _load() -> None:
    global _transcriber
    try:
        transcriber = WhisperTranscriber(model_size())
    except Exception as exc:  # download blocked, disk full, bad install...
        log.warning("Whisper failed to load; falling back to script text: %s", exc)
        with _lock:
            _state.update(state="failed", error=str(exc)[:300])
        return
    with _lock:
        _transcriber = transcriber
        _state.update(state="ready", error=None)
    log.info("Whisper %s ready", model_size())


def start_warmup() -> None:
    """Begin loading Whisper in the background (no-op if disabled or not installed)."""
    with _lock:
        if _state["state"] in ("loading", "ready"):
            return
        if os.environ.get("ASSIST_DISABLE_STT") or not whisper_installed():
            _state.update(state="off", error=None)
            return
        _state.update(state="loading", error=None)
    threading.Thread(target=_load, name="whisper-warmup", daemon=True).start()


def status() -> dict:
    with _lock:
        return {"state": _state["state"], "error": _state["error"], "model": model_size()}


def get_transcriber() -> WhisperTranscriber | None:
    """The loaded model, or None if it's off, still loading, or failed."""
    with _lock:
        return _transcriber if _state["state"] == "ready" else None
