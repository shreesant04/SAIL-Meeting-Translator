#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "    Running SAIL Meeting Translator Tests"
echo "=========================================="

# Check if running in Codespace venv or local venv/python
if [ -f "/workspaces/venv/bin/pytest" ]; then
    PYTEST_BIN="/workspaces/venv/bin/pytest"
    PYTHON_BIN="/workspaces/venv/bin/python"
elif [ -d "venv" ]; then
    PYTEST_BIN="venv/bin/pytest"
    PYTHON_BIN="venv/bin/python"
else
    PYTEST_BIN="pytest"
    PYTHON_BIN="python3"
fi

echo "--- 1. Testing Backend (Python / Pytest) ---"
cd backend
$PYTHON_BIN -m pytest tests/ -v
cd ..

echo "--- 2. Testing Frontend (Lint & Build) ---"
cd frontend
npm run lint
npm run build
cd ..

echo "=========================================="
echo "    All Tests & Verification Passed!     "
echo "=========================================="
