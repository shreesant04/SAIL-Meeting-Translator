#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect Python binary
if [ -f "/workspaces/venv/bin/python" ]; then
    PYTHON_BIN="/workspaces/venv/bin/python"
    UVICORN_BIN="/workspaces/venv/bin/uvicorn"
elif [ -d "venv" ]; then
    PYTHON_BIN="venv/bin/python"
    UVICORN_BIN="venv/bin/uvicorn"
else
    PYTHON_BIN="python3"
    UVICORN_BIN="uvicorn"
fi

echo "============================================="
echo " Starting SAIL Meeting Translator Services"
echo "============================================="

# Ensure Piper voice exists & samples are prepared
$PYTHON_BIN backend/download_voices.py || true

# Start backend in background
echo "Starting Backend API on port 8000..."
cd backend
$UVICORN_BIN main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Cleanup handler on exit
trap "kill $BACKEND_PID 2>/dev/null || true" EXIT

# Start frontend
echo "Starting Frontend Vite server on port 5173..."
cd frontend
npm run dev -- --host 0.0.0.0
