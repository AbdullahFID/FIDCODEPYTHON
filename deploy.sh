#!/bin/bash
set -e # Exit immediately if a command fails

# --- Your Server IP ---
IP="64.23.178.61"

echo "🚀 Deploying to $IP..."

# This is the 'heredoc'. It sends all the commands between
# <<'REMOTE' and REMOTE to the server to be executed.
ssh root@$IP 'bash -s' <<'REMOTE'
set -e

echo "➡️  (On Server) Starting deployment steps..."

PROJECT_DIR="/opt/flightintel"

# 1. Go to project directory and PULL THE LATEST CODE
cd "$PROJECT_DIR"
echo "🔄  Updating repository from GitHub..."
git fetch origin
git reset --hard origin/main

# 2. Define Paths and Environment
VENV="$PROJECT_DIR/venv"
PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python"
REQUIREMENTS="$PROJECT_DIR/server/requirements.txt"
SERVER_CODE_DIR="$PROJECT_DIR/server"
source "$HOME/.cargo/env"

# 3. Install Dependencies
echo "📦  Installing dependencies..."
"$PIP" install -r "$REQUIREMENTS"

# 4. Restart the Application
echo "🔄  Restarting the application service..."
tmux kill-session -t flightintel 2>/dev/null || true
tmux new -d -s flightintel "cd $SERVER_CODE_DIR && exec $PYTHON -m uvicorn main:app --host 0.0.0.0 --port 8000"

# 5. Final Check
echo "✅  (On Server) Deployment complete!"
REMOTE

echo "🎉  All done!"