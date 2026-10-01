#!/usr/bin/env bash
# One-time setup for macOS/Linux. Run from VS Code (Terminal > Run Task >
# "Set up project") or from the project folder:  ./scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n==> %s\n' "$1"; }
warn() { printf '    ! %s\n' "$1"; }

step "Checking Python (3.10 or newer)"
PY=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    PY="$candidate"; break
  fi
done
[ -n "$PY" ] || { echo "Python 3.10+ not found."; exit 1; }
echo "    Using: $PY ($("$PY" --version))"

[ -x .venv/bin/python ] || { step "Creating virtual environment in .venv"; "$PY" -m venv .venv; }

step "Installing Python dependencies"
.venv/bin/python -m pip install --upgrade pip --quiet
if ! .venv/bin/python -m pip install -e ".[dev,speech]"; then
  warn "Speech-to-text install failed; retrying without it (live mic won't work)."
  .venv/bin/python -m pip install -e ".[dev]"
fi

step "Checking ffmpeg (used by Whisper to read audio)"
command -v ffmpeg >/dev/null 2>&1 && echo "    ffmpeg found." ||
  warn "ffmpeg not found. Install it (brew install ffmpeg / apt install ffmpeg) for speech-to-text."

step "Installing and building the UI"
command -v npm >/dev/null 2>&1 || { echo "npm not found. Install Node.js 18+."; exit 1; }
(cd web && npm install --no-audit --no-fund && npm run build)

[ -f .env ] || { cp .env.example .env; echo "    Created .env (optional settings)."; }

printf '\nSetup complete.\nIn VS Code: press F5 (Run Agent Assist). The app opens at http://localhost:8000\n'
