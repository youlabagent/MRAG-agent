#!/usr/bin/env bash
# MRAG project: production user-mode Ollama (:11500). 2026-09-28: models dir fixed to /BIO1/MRAG/models/ollama
# (the dir the running server has used since 2026-05-26; the old /usr/share path has no batiai/qwen3.6-27b:q6).
# Originally the GLM-4.7-flash starter.
# port 11500, GPU 1, NUM_PARALLEL=4, ctx=8192
# verified May 2026: ~150 tok/s aggregate, 22.7GB VRAM
LOG="/BIO1/MRAG/logs/glm_ollama.log"
mkdir -p "$(dirname "$LOG")"
if curl -sS --max-time 2 http://127.0.0.1:11500/api/tags >/dev/null 2>&1; then
  echo "[start_glm_ollama] already up on :11500" >> "$LOG"
  exit 0
fi
exec env \
  OLLAMA_HOST=127.0.0.1:11500 \
  OLLAMA_MODELS=/BIO1/MRAG/models/ollama \
  OLLAMA_NUM_PARALLEL=4 \
  OLLAMA_MAX_LOADED_MODELS=1 \
  OLLAMA_KEEP_ALIVE=30m \
  OLLAMA_CONTEXT_LENGTH=8192 \
  CUDA_VISIBLE_DEVICES=1 \
  ollama serve >> "$LOG" 2>&1
