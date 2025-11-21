#!/bin/bash
set -euo pipefail

IP="64.23.178.61"

echo "🚀 Deploying to $IP..."

ssh root@"$IP" 'bash -s' <<'REMOTE'
set -euo pipefail

echo "➡️  (On Server) Starting deployment steps..."

PROJECT_DIR="/opt/flightintel"
SERVER_CODE_DIR="$PROJECT_DIR/server"
VENV="$PROJECT_DIR/venv"
PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"
REQUIREMENTS="$SERVER_CODE_DIR/requirements.txt"

# Log directory for Loki/Promtail
LOG_DIR="/var/log/flightintel"

cd "$PROJECT_DIR"
echo "🔄  Updating repository from GitHub..."
git fetch origin
git reset --hard origin/main

# Ensure virtualenv
if [ ! -x "$PYTHON" ]; then
  echo "🐍  Creating virtualenv..."
  python3 -m venv "$VENV"
  "$PIP" install -U pip wheel setuptools
fi

echo "📦  Installing dependencies..."
"$PIP" install -r "$REQUIREMENTS"

# Ensure log directory (file will be created by logging_utils)
echo "🧾  Ensuring log directory exists at $LOG_DIR..."
mkdir -p "$LOG_DIR"
chmod 755 "$LOG_DIR"

echo "🔄  Restarting the application service..."
tmux kill-session -t flightintel 2>/dev/null || true
tmux new -d -s flightintel "cd $SERVER_CODE_DIR && exec $PYTHON -m uvicorn main:app --host 0.0.0.0 --port 8000"

echo "✅  (On Server) Deployment complete!"
REMOTE

echo "🎉  All done!"
