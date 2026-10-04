# run-server.ps1 -- start the Triage OpenJev HTTP API on a Windows edge box.
#
#   powershell -ExecutionPolicy Bypass -File deploy\run-server.ps1
#
# Every decision this serves is advisory and must be reviewed by a qualified
# clinician. See docs/LIMITS.md.

[CmdletBinding()]
param(
    [int]    $Port    = 8773,
    [string] $Model   = "granite4.1:8b",
    [string] $Backend = "ollama",
    [string] $Python  = "python"
)

$ErrorActionPreference = "Stop"

# Repo root is the parent of this script's directory.
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "Triage OpenJev API"
Write-Host "  repo    : $RepoRoot"
Write-Host "  python  : $Python"
Write-Host "  model   : $Model ($Backend)"
Write-Host "  port    : $Port"

# 1. Fail early and loudly if the interpreter or OpenJev is missing, rather
#    than starting a server that cannot answer.
& $Python -c "import openjev, sys; print('openjev', openjev.__version__, 'on', sys.version.split()[0])"
if ($LASTEXITCODE -ne 0) {
    throw "OpenJev is not importable with '$Python'. Run: pip install -r requirements.txt"
}

# 2. doctor separates 'model cannot answer' from 'install is broken'. A model
#    that emits a thinking preamble fails here by design.
Write-Host ""
Write-Host "Running doctor (backend reachability + can the model answer at all)..."
& $Python -m src.triage.cli doctor --backend $Backend --model $Model
if ($LASTEXITCODE -ne 0) {
    throw "doctor failed. Fix the backend/model before serving traffic."
}

# 3. Serve. This binds locally; put an authenticating reverse proxy in front
#    before exposing it to a network -- the payload is PHI.
Write-Host ""
Write-Host "Starting API on http://127.0.0.1:$Port ... (Ctrl-C to stop)"
& $Python -m src.triage.api --port $Port --backend $Backend --model $Model
