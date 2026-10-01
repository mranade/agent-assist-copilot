#!/usr/bin/env python3
"""Generate audio for the scripted sample calls in data/calls/calls.json.

Each turn becomes its own WAV file (data/calls/audio/<call-id>/<turn>.wav)
so the UI can play a call turn by turn and the backend can transcribe each
turn separately, the same way a contact-center platform streams audio per
speaker segment.

Uses Piper (free, offline neural TTS; `pip install -e ".[voices]"`) when it is
installed, and falls back to espeak-ng, which sounds robotic. Piper voices are
downloaded on first use into .cache/piper-voices/. You can also record the
lines yourself and drop the WAV files in the same paths. Nothing else needs to
change.

Usage:
    python scripts/make_call_audio.py                   # auto: Piper if installed
    python scripts/make_call_audio.py --engine piper    # require Piper
    python scripts/make_call_audio.py --call call-01    # one call only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALLS_PATH = ROOT / "data" / "calls" / "calls.json"
AUDIO_DIR = ROOT / "data" / "calls" / "audio"
PIPER_VOICE_DIR = ROOT / ".cache" / "piper-voices"

# Piper voice per speaker, matched to the names in calls.json.
PIPER_VOICES = {
    "call-01": {"agent": "en_US-ryan-medium", "customer": "en_US-lessac-medium"},  # Sam, Dana
    "call-02": {"agent": "en_US-amy-medium", "customer": "en_US-joe-medium"},  # Priya
    "call-03": {"agent": "en_US-ryan-medium", "customer": "en_US-amy-medium"},  # Marco
}
DEFAULT_PIPER_VOICES = {"agent": "en_US-ryan-medium", "customer": "en_US-lessac-medium"}

# Slightly slower than Piper's default reads more like a phone call.
PIPER_LENGTH_SCALE = 1.05

ESPEAK_VOICES = {
    "agent": ["-v", "en-us", "-s", "165", "-p", "40"],
    "customer": ["-v", "en-us+f3", "-s", "160", "-p", "55"],
}


def piper_available() -> bool:
    try:
        import piper  # noqa: F401
    except ImportError:
        return False
    return True


class PiperEngine:
    name = "piper"

    def __init__(self) -> None:
        self._loaded: dict = {}

    def voice_name(self, call_id: str, speaker: str) -> str:
        return PIPER_VOICES.get(call_id, DEFAULT_PIPER_VOICES)[speaker]

    def _voice(self, name: str):
        if name not in self._loaded:
            from piper import PiperVoice

            model = PIPER_VOICE_DIR / f"{name}.onnx"
            if not (model.exists() and model.with_suffix(".onnx.json").exists()):
                from piper.download_voices import download_voice

                print(f"Downloading Piper voice {name} ...")
                PIPER_VOICE_DIR.mkdir(parents=True, exist_ok=True)
                download_voice(name, PIPER_VOICE_DIR)
            self._loaded[name] = PiperVoice.load(str(model))
        return self._loaded[name]

    def synthesize(self, call_id: str, speaker: str, text: str, out: Path) -> None:
        from piper.config import SynthesisConfig

        voice = self._voice(self.voice_name(call_id, speaker))
        with wave.open(str(out), "wb") as wf:
            voice.synthesize_wav(text, wf, syn_config=SynthesisConfig(length_scale=PIPER_LENGTH_SCALE))


class EspeakEngine:
    name = "espeak-ng"

    def voice_name(self, call_id: str, speaker: str) -> str:
        return " ".join(ESPEAK_VOICES[speaker])

    def synthesize(self, call_id: str, speaker: str, text: str, out: Path) -> None:
        subprocess.run(["espeak-ng", *ESPEAK_VOICES[speaker], "-w", str(out), text], check=True)


def pick_engine(choice: str):
    if choice == "piper" or (choice == "auto" and piper_available()):
        if not piper_available():
            print('Piper is not installed. Run: pip install -e ".[voices]"')
            return None
        return PiperEngine()
    if not shutil.which("espeak-ng"):
        print("espeak-ng is not installed (apt install espeak-ng / brew install espeak-ng),")
        print('and Piper is not installed either (pip install -e ".[voices]").')
        return None
    return EspeakEngine()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine", choices=["auto", "piper", "espeak"], default="auto",
                        help="auto (default) uses Piper if installed, else espeak-ng")
    parser.add_argument("--call", help="only regenerate this call id, e.g. call-01")
    args = parser.parse_args(argv)

    calls = json.loads(CALLS_PATH.read_text(encoding="utf-8"))
    if args.call:
        calls = [c for c in calls if c["id"] == args.call]
        if not calls:
            print(f"No call with id {args.call!r} in {CALLS_PATH}")
            return 1

    engine = pick_engine(args.engine)
    if engine is None:
        return 1
    print(f"Engine: {engine.name}")

    count = 0
    for call in calls:
        speakers = sorted({t["speaker"] for t in call["turns"]})
        voices = ", ".join(f"{s}={engine.voice_name(call['id'], s)}" for s in speakers)
        print(f"  {call['id']}: {voices}")
        out_dir = AUDIO_DIR / call["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, turn in enumerate(call["turns"]):
            engine.synthesize(call["id"], turn["speaker"], turn["text"], out_dir / f"{i:02d}.wav")
            count += 1
    print(f"Wrote {count} audio clips to {AUDIO_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
