#!/usr/bin/env bash
# Boot ModernCV — FastAPI backend + Vite frontend in a single command.
# Ctrl-C cleanly shuts both down.
#
# Usage:
#   ./run.sh                    # default ports: backend 8000, frontend 5173
#   BACKEND_PORT=9000 ./run.sh  # override
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BACKEND_PORT="${BACKEND_PORT:-8000}"

# ---- preflight checks ------------------------------------------------------

if [ ! -x ".venv/bin/uvicorn" ]; then
  echo "✗ .venv/bin/uvicorn missing." >&2
  echo "  Set up the Python env first (e.g. 'uv venv && uv sync')." >&2
  exit 1
fi

if [ ! -d "gui/node_modules" ]; then
  echo "✗ gui/node_modules missing." >&2
  echo "  Run 'cd gui && npm install' first." >&2
  exit 1
fi

# ---- shutdown handling -----------------------------------------------------

BACKEND_PID=""
cleanup() {
  if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo
    echo "› stopping backend (pid $BACKEND_PID)"
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ---- launch ----------------------------------------------------------------

echo "› starting backend  → http://localhost:${BACKEND_PORT}"
.venv/bin/uvicorn server.main:app --reload --port "$BACKEND_PORT" &
BACKEND_PID=$!

# Give uvicorn a moment so its startup logs print before Vite's banner — easier
# to read when something goes wrong on boot.
sleep 1

echo "› starting frontend → vite dev server"
cd gui
npm run dev
