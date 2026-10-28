#!/usr/bin/env bash
# MRAG Orchestrator Web UI — startup helper
# Idempotent: skip if already up on port 8765.
set -euo pipefail
LOG=/BIO1/MRAG/logs/mrag_webui.log
mkdir -p "$(dirname "$LOG")"
if curl -sS --max-time 2 http://127.0.0.1:8765/ >/dev/null 2>&1; then
    echo "[start_mrag_webui] already up on :8765" >> "$LOG"
    exit 0
fi
# HOMER writes its scratch files (<random>.seq/.pos/.tmp) into the working
# directory. Started from the source tree it left 28 GB of them next to app.py,
# so run from a scratch dir instead and keep the code tree clean.
SCRATCH=/BIO1/MRAG/scratch/webui
mkdir -p "$SCRATCH"
cd "$SCRATCH"
exec env \
    MRAG_WEBUI_HOST=0.0.0.0 \
    MRAG_WEBUI_PORT=8765 \
    MRAG_BIN=/BIO1/MRAG/venvs/paper_qa/bin/mrag \
    OLLAMA_ENDPOINT=http://127.0.0.1:11500 \
    MRAG_MODEL=batiai/qwen3.6-27b:q6 \
    TMPDIR="$SCRATCH" \
    /BIO1/MRAG/venvs/paper_qa/bin/python /BIO1/MRAG/Agent/orchestrator/webui/app.py >> "$LOG" 2>&1
