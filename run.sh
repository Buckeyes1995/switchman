#!/usr/bin/env bash
# Run Switchman, creating venv on first launch
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv"

if [[ ! -d "$VENV" ]]; then
    echo "Creating venv..."
    /opt/homebrew/bin/python3.13 -m venv "$VENV"
    "$VENV/bin/pip" install -q -r "$DIR/requirements.txt"
fi

exec "$VENV/bin/python" "$DIR/switchman.py"
