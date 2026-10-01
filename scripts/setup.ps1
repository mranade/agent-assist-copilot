# One-time setup for Windows. Run from VS Code (Terminal > Run Task > "Set up project")
# or from PowerShell in the project folder:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#
# Creates .venv, installs the Python package (with local Whisper speech-to-text),
# installs and builds the React UI, and creates .env from .env.example.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "    ! $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host ""; Write-Host "Setup stopped: $msg" -ForegroundColor Red; exit 1 }

# ---- Python -----------------------------------------------------------------
Step "Checking Python (3.10 or newer)"
$Python = $null
$PythonArgs = @()
foreach ($candidate in @("py -3", "python", "python3")) {
    $parts = $candidate.Split(" ")
    $exe = $parts[0]
    # Note: $parts[1..0] counts DOWN in PowerShell, so build the args explicitly.
    $extra = @(); if ($parts.Length -gt 1) { $extra = $parts[1..($parts.Length - 1)] }
    try {
        $version = & $exe @extra -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    } catch { continue }
    if ($LASTEXITCODE -eq 0 -and $version) {
        $version = "$($version | Select-Object -Last 1)".Trim()
        $major, $minor = $version.Split(".") | ForEach-Object { [int]$_ }
        if ($major -eq 3 -and $minor -ge 10) { $Python = $exe; $PythonArgs = $extra; break }
        Warn "Found Python $version via '$candidate', but 3.10+ is required."
    }
}
if (-not $Python) { Fail "Python 3.10+ not found. Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH') and rerun." }
Write-Host "    Using: $candidate ($version)"

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Step "Creating virtual environment in .venv"
    & $Python @PythonArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "Could not create .venv." }
} else {
    Step "Using existing .venv"
}

Step "Installing Python dependencies (this can take a few minutes the first time)"
& $VenvPython -m pip install --upgrade pip --quiet
& $VenvPython -m pip install -e ".[dev,speech]"
if ($LASTEXITCODE -ne 0) {
    Warn "Installing with speech-to-text failed. Retrying without it (sample calls and typing still work; live mic won't)."
    & $VenvPython -m pip install -e ".[dev]"
    if ($LASTEXITCODE -ne 0) { Fail "pip install failed. Scroll up for the error." }
}

# ---- ffmpeg (needed only for Whisper) ----------------------------------------
Step "Checking ffmpeg (used by Whisper to read audio)"
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Write-Host "    ffmpeg found."
} else {
    Warn "ffmpeg not found. Speech-to-text needs it. Install with:  winget install ffmpeg"
    Warn "Then open a new terminal (or restart VS Code) so it's on PATH."
}

# ---- Node / UI ----------------------------------------------------------------
Step "Installing and building the UI"
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Fail "npm not found. Install Node.js 18+ from https://nodejs.org and rerun."
}
Push-Location web
npm install --no-audit --no-fund
if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "npm install failed." }
npm run build
if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "UI build failed." }
Pop-Location

# ---- .env ------------------------------------------------------------------------
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "    Created .env (optional settings, e.g. ANTHROPIC_API_KEY)."
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "In VS Code: press F5 (Run Agent Assist). The app opens at http://localhost:8000"
Write-Host "From a terminal: .venv\Scripts\python.exe -m assist.server"
