#!/usr/bin/env bash
# Ensure the DEDICATED target-scorer Ollama (0.34.4, user space, qwen3.8) is up -- idempotent.
# Design: 127.0.0.1:11501, GPU 2 (or whichever of GPU 0/2 is free), models dir
# /BIO1/MRAG/models/ollama-q38, loaded on demand, KEEP_ALIVE 30m. Production :11500 is never touched.
# Written 2026-09-28; tested on a sandbox port (:11503) / GPU 0 before deployment.
# Pre-deploy review (2026-09-28): --probe, and --warm re-pins a server whose GPU became unusable
# (tested on the sandbox port :11503).
#
# usage: ensure_scorer_ollama.sh [--check | --probe | --warm | --unload | --stop]
#   (none)    start the server if nothing listens on the port, else verify that it is ours;
#             then verify that the scorer model is listed
#   --warm    as above, then load the model and verify it sits 100% in VRAM of the pinned GPU at the
#             expected context (refuses to load onto a GPU that lacks the memory: no silent CPU spill).
#             The server is pinned to the GPU it was started on; when that GPU can no longer take the
#             model and nothing of ours is on it (model not loaded, no runner = no request in flight),
#             the server is restarted on the first GPU of $GPUS that can (pre-deploy review 2026-09-28)
#   --check   read-only status: exit 0 = ours and healthy, 1 = not running, 3 = something else / wrong env
#   --probe   read-only "could the scorer serve now?" (run at the START of a curated run): exit 0 = yes
#             (ours with the model loaded, or a GPU of $GPUS can take it now), 2 = not now (no GPU / RAM
#             free), 3 = it would fail anyway (foreign listener, our server with another config, binary or
#             model missing); never starts, loads, stops or locks anything
#   --unload  unload the model now (frees the GPU; the server stays up, reloads on the next request)
#   --stop    stop OUR server (identified by port + user + binary + env), never another process
# A hung server of ours (listening, identity verified, /api/version silent for 10 s) is restarted.
# exit: 0 ok | 1 not running (--check) | 2 no GPU / RAM free now, retry later | 3 refuse | 4 failed
#
# Config (env; defaults = the deployment design):
#   SCORER_PORT=11501  SCORER_GPUS="2 0"  SCORER_MODELS=/BIO1/MRAG/models/ollama-q38
#   SCORER_MODEL=qwen3.8:27b-q8_0  SCORER_CTX=32768  SCORER_KEEP_ALIVE=30m
#   SCORER_MIN_FREE_MIB=32000 (VRAM needed on the GPU: qwen3.8 Q8 at 32768 = 30.2 GB incl. context)
#   SCORER_MIN_AVAIL_GB=40 (system MemAvailable)   SCORER_STATE=/BIO1/MRAG/logs/scorer_ollama_<port>
#   SCORER_LOG=<state>/ollama.log
set -u
PORT=${SCORER_PORT:-11501}
GPUS=${SCORER_GPUS:-"2 0"}
MODELS=${SCORER_MODELS:-/BIO1/MRAG/models/ollama-q38}
MODEL=${SCORER_MODEL:-qwen3.8:27b-q8_0}
CTX=${SCORER_CTX:-32768}
KEEP=${SCORER_KEEP_ALIVE:-30m}
MIN_FREE_MIB=${SCORER_MIN_FREE_MIB:-32000}
MIN_AVAIL_GB=${SCORER_MIN_AVAIL_GB:-40}
STATE=${SCORER_STATE:-/BIO1/MRAG/logs/scorer_ollama_$PORT}
LOG=${SCORER_LOG:-$STATE/ollama.log}
BIN=/BIO1/MRAG/tools/ollama-0.34.4/bin/ollama
VER=0.34.4
CPUS=32-47
HOSTPORT=127.0.0.1:$PORT
URL=http://$HOSTPORT
PROD_PORT=11500; PROD_GPU=1; PROD_MODELS=/BIO1/MRAG/models/ollama
MODE=ensure
case "${1:-}" in "") ;; --check) MODE=check;; --probe) MODE=probe;; --warm) MODE=warm;; --unload) MODE=unload;; --stop) MODE=stop;;
  *) echo "usage: $0 [--check|--probe|--warm|--unload|--stop]" >&2; exit 3;; esac

log(){ echo "$(date '+%F %T') [ensure_scorer_ollama :$PORT] $*" >&2; }
refuse(){ log "REFUSE: $*"; exit 3; }
fail(){ log "FAILED: $*"; exit 4; }
busy(){ log "BUSY: $*"; exit 2; }

# ---- static guards: never production, never the shared system binary ----------------------------
[ "$(id -un)" = ai ] || refuse "run as user ai"
[ "$PORT" = "$PROD_PORT" ] && refuse "port $PORT is production :11500"
[ "$PORT" = 11434 ] && refuse "port 11434 is the system ollama"
for g in $GPUS; do [ "$g" = "$PROD_GPU" ] && refuse "GPU $g is the production GPU"; case $g in *[!0-9]*|"") refuse "bad GPU id '$g'";; esac; done
[ "$(readlink -f "$MODELS")" = "$PROD_MODELS" ] && refuse "models dir is the production dir $PROD_MODELS"
[ -d "$MODELS/manifests" ] && [ -d "$MODELS/blobs" ] || refuse "$MODELS is not an Ollama models dir"
[ -x "$BIN" ] || refuse "$BIN missing"
mkdir -p "$STATE" || refuse "cannot create $STATE"

pid_on_port(){ ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u; }
penv(){ tr '\0' '\n' < /proc/$1/environ 2>/dev/null | grep "^$2=" | head -1 | cut -d= -f2-; }
api(){ curl -sS --max-time "${2:-10}" "$URL$1" 2>/dev/null; }
api_version(){ api /api/version "${1:-5}" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("version",""))' 2>/dev/null; }
model_listed(){ api /api/tags 20 | python3 -c 'import json,sys; m=sys.argv[1]; sys.exit(0 if any(x.get("name")==m or x.get("model")==m for x in json.load(sys.stdin).get("models",[])) else 1)' "$MODEL" 2>/dev/null; }
gpu_uuid(){ nvidia-smi --id="$1" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' '; }
gpu_free_mib(){ nvidia-smi --id="$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }
gpu_foreign(){ # compute processes on GPU $1 that are not user ai's
  local u; u=$(gpu_uuid "$1")
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | tr -d ' ' | awk -F, -v u="$u" '$1==u{print $2}' | \
    while read -r q; do o=$(ps -o user= -p "$q" 2>/dev/null | tr -d ' '); [ -n "$o" ] && [ "$o" != ai ] && echo "$q:$o"; done
}
gpu_pids(){ local u; u=$(gpu_uuid "$1"); nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | tr -d ' ' | awk -F, -v u="$u" '$1==u{print $2}'; }
started_pids(){ # our binary serving THIS host:port + models dir (listening or not)
  local q; for q in $(pgrep -u ai -x ollama); do
    [ "$(readlink /proc/$q/exe 2>/dev/null)" = "$BIN" ] && [ "$(penv "$q" OLLAMA_HOST)" = "$HOSTPORT" ] && \
      [ "$(penv "$q" OLLAMA_MODELS)" = "$MODELS" ] && echo "$q"; done; }
children(){ ps -o pid= --ppid "$1" 2>/dev/null | tr -d ' '; }
model_loaded(){ api /api/ps 10 | python3 -c 'import json,sys; m=sys.argv[1]; sys.exit(0 if any(x.get("name")==m for x in json.load(sys.stdin).get("models",[])) else 1)' "$MODEL" 2>/dev/null; }
gpu_why_not(){ # why GPU $1 cannot take the model now ("" = it can): a foreign process or too little free memory
  local f o; f=$(gpu_free_mib "$1"); o=$(gpu_foreign "$1")
  if [ -n "$o" ]; then echo "GPU $1 now has foreign process(es) $o"
  elif [ -z "$f" ]; then echo "GPU $1: free memory unreadable"
  elif [ "$f" -lt "$MIN_FREE_MIB" ]; then echo "GPU $1 has ${f} MiB free < $MIN_FREE_MIB"; fi; }
mem_avail_gb(){ free -g | awk '/^Mem:/{print $7}'; }
in_flight(){ # $1 pid: what of ours is using the server now ("" = nothing): a runner child (a model loading
  # or loaded), or an open client connection to the port -- a request that arrived but has not spawned
  # its runner yet (GPU discovery + metadata take ~1-2 s, measured 2026-09-28) has only the latter
  local k c; k=$(children "$1" | tr '\n' ' ' | sed 's/ $//')
  [ -n "$k" ] && { echo "runner child(ren) $k of pid $1"; return; }
  c=$(ss -tnH state established "( sport = :$PORT )" 2>/dev/null | grep -c .)
  [ "$c" -gt 0 ] && echo "$c open client connection(s) to :$PORT"; }
bin_version(){ OLLAMA_HOST=127.0.0.1:1 "$BIN" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | tail -1; }

# identity of the process on the port: "ours <pid> <gpu>" / "none" / "foreign <reason>"
identify(){
  local p n; p=$(pid_on_port); n=$(echo "$p" | grep -c .)
  [ "$n" = 0 ] && { echo none; return; }
  [ "$n" = 1 ] || { echo "foreign more-than-one-listener:$(echo $p)"; return; }
  local u exe host models gpu
  u=$(ps -o user= -p "$p" | tr -d ' '); exe=$(readlink "/proc/$p/exe" 2>/dev/null)
  host=$(penv "$p" OLLAMA_HOST); models=$(penv "$p" OLLAMA_MODELS); gpu=$(penv "$p" CUDA_VISIBLE_DEVICES)
  [ "$u" = ai ] || { echo "foreign user=$u pid=$p"; return; }
  [ "$exe" = "$BIN" ] || { echo "foreign exe=$exe pid=$p"; return; }
  [ "$host" = "$HOSTPORT" ] && [ "$models" = "$MODELS" ] || { echo "foreign env HOST=$host MODELS=$models pid=$p"; return; }
  echo "ours $p $gpu"
}
# the pinned env must be exactly what this script starts (else a human decides; nothing is killed)
check_env(){ # $1 pid $2 gpu
  local p=$1 bad="" k want have
  for kv in "OLLAMA_CONTEXT_LENGTH=$CTX" "OLLAMA_MAX_LOADED_MODELS=1" "OLLAMA_NUM_PARALLEL=1" "OLLAMA_KEEP_ALIVE=$KEEP" \
            "OLLAMA_NOPRUNE=1" "OLLAMA_NO_CLOUD=1" "OLLAMA_VULKAN=0" "LLAMA_ARG_THREADS=4" "CUDA_DEVICE_ORDER=PCI_BUS_ID"; do
    k=${kv%%=*}; want=${kv#*=}; have=$(penv "$p" "$k"); [ "$have" = "$want" ] || bad="$bad $k=$have(want $want)"
  done
  case " $GPUS " in *" $2 "*) ;; *) bad="$bad CUDA_VISIBLE_DEVICES=$2(not in '$GPUS')";; esac
  [ "$(grep Cpus_allowed_list /proc/$p/status | awk '{print $2}')" = "$CPUS" ] || bad="$bad affinity"
  [ -z "$bad" ] || refuse "pid $p on :$PORT is our binary but runs with a different config:$bad -- stop it with --stop and re-run"
}

stop_ours(){ # $1 pid (identity already verified); TERM, then KILL (it is ours and dedicated)
  local p=$1 kids i; kids=$(children "$p")
  log "stopping pid $p (children: ${kids:-none})"
  kill -TERM "$p" 2>/dev/null; kill -CONT "$p" 2>/dev/null
  for i in $(seq 1 30); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
  if kill -0 "$p" 2>/dev/null; then log "pid $p ignored TERM for 30 s -> KILL"; kill -KILL "$p" 2>/dev/null; sleep 2; fi
  for k in $kids; do # llama-server children of OUR server only
    if kill -0 "$k" 2>/dev/null && [ "$(ps -o user= -p "$k" | tr -d ' ')" = ai ]; then kill -TERM "$k" 2>/dev/null; sleep 3; kill -0 "$k" 2>/dev/null && kill -KILL "$k" 2>/dev/null; fi
  done
  for i in $(seq 1 30); do [ -z "$(pid_on_port)" ] && break; sleep 1; done
  [ -z "$(pid_on_port)" ] || fail "port :$PORT still busy after stopping $p"
  rm -f "$STATE/pid"; true
}

choose_gpu(){ # first GPU of $GPUS with no foreign process and >= MIN_FREE_MIB free
  local g f o
  for g in $GPUS; do
    f=$(gpu_free_mib "$g"); o=$(gpu_foreign "$g")
    [ -n "$f" ] || continue
    if [ -z "$o" ] && [ "$f" -ge "$MIN_FREE_MIB" ]; then echo "$g"; return 0; fi
    log "GPU $g skipped: free ${f} MiB (need $MIN_FREE_MIB), foreign: ${o:-none}"
  done
  return 1
}

start_server(){
  local a g i p
  a=$(mem_avail_gb); [ "$a" -ge "$MIN_AVAIL_GB" ] || busy "system MemAvailable ${a} GiB < ${MIN_AVAIL_GB} GiB"
  [ "$(bin_version)" = "$VER" ] || refuse "$BIN is not $VER"
  g=$(choose_gpu) || busy "no GPU of '$GPUS' is free enough now"
  log "starting $BIN on $HOSTPORT GPU $g ctx $CTX models $MODELS (log $LOG)"
  echo "=== $(date '+%F %T') ensure_scorer_ollama start GPU=$g CTX=$CTX" >> "$LOG"
  # fd 9 (the flock) is closed for the server, else the lock would live as long as the server
  ( cd /home/ai || exit 1; exec 9>&-; setsid nohup taskset -c $CPUS env -i HOME=/home/ai PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
      OLLAMA_HOST=$HOSTPORT OLLAMA_MODELS=$MODELS OLLAMA_CONTEXT_LENGTH=$CTX OLLAMA_NUM_PARALLEL=1 \
      OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=$KEEP OLLAMA_NOPRUNE=1 OLLAMA_NO_CLOUD=1 \
      OLLAMA_VULKAN=0 GGML_VK_VISIBLE_DEVICES= LLAMA_ARG_THREADS=4 LLAMA_ARG_THREADS_BATCH=4 \
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g \
      "$BIN" serve >> "$LOG" 2>&1 < /dev/null & )
  local end=$((SECONDS + 60)); while [ $SECONDS -lt $end ]; do [ "$(api_version 2)" = "$VER" ] && break; sleep 1; done
  if [ "$(api_version)" != "$VER" ]; then
    for p in $(started_pids); do stop_ours "$p"; done
    fail "server did not answer /api/version in 60 s (anything it left was stopped; see $LOG)"
  fi
  set -- $(identify); [ "$1" = ours ] || fail "listener on :$PORT after start is not ours: $*"
  echo "$2" > "$STATE/pid"; log "started pid $2"
}

warm(){ # $1 pid $2 gpu [$3 = 1: already re-pinned once in this call]
  local ps why g a
  ps=$(api /api/ps 10)
  if ! echo "$ps" | python3 -c 'import json,sys; m=sys.argv[1]; sys.exit(0 if any(x.get("name")==m for x in json.load(sys.stdin).get("models",[])) else 1)' "$MODEL" 2>/dev/null; then
    why=$(gpu_why_not "$2")
    if [ -n "$why" ]; then
      # pre-deploy review 2026-09-28: re-pin. choose_gpu ran only when the server was started, so a
      # server started on GPU 2 stayed there for good and every later run failed BUSY while GPU 0 was
      # free. Restart it on another GPU only when nothing of ours is on the pinned one: the model is
      # not loaded (above), the server has no runner child and no client connection (no load or
      # request in flight).
      [ "${3:-0}" = 1 ] && busy "$why; not loading (it would spill to CPU)"
      local busy_now; busy_now=$(in_flight "$1")
      [ -z "$busy_now" ] || busy "$why, and a request is in flight ($busy_now); not loading, not re-pinning"
      g=$(choose_gpu) || busy "$why; no other GPU of '$GPUS' is free enough either; not loading (it would spill to CPU)"
      if [ "$g" != "$2" ]; then
        a=$(mem_avail_gb); [ "$a" -ge "$MIN_AVAIL_GB" ] || busy "$why; system MemAvailable ${a} GiB < ${MIN_AVAIL_GB} GiB, not re-pinning"
        log "$why and $MODEL is not loaded (no runner): re-pinning -- restarting pid $1 on GPU $g"
        stop_ours "$1"; start_server; set -- $(identify)
        [ "$1" = ours ] || fail "re-pinned server not identified as ours: $*"
        check_env "$2" "${3:-}"
        model_listed || fail "$MODEL is not listed by the re-pinned server on :$PORT (models dir $MODELS)"
        warm "$2" "${3:-}" 1; return
      fi
      log "GPU $2 can take $MODEL again"
    fi
    log "loading $MODEL on GPU $2 ..."
    local t0; t0=$(date +%s)
    curl -sS --max-time 900 "$URL/api/generate" -d "{\"model\":\"$MODEL\"}" >/dev/null || fail "load request failed"
    log "loaded in $(( $(date +%s) - t0 )) s"
    ps=$(api /api/ps 10)
  fi
  echo "$ps" | python3 -c '
import json, sys
m, ctx = sys.argv[1], int(sys.argv[2])
x = next((x for x in json.load(sys.stdin).get("models", []) if x.get("name") == m), None)
if x is None: print("model not loaded after the load request"); sys.exit(1)
if int(x.get("size_vram", 0)) != int(x.get("size", -1)): print("NOT fully on GPU: size_vram %s of %s" % (x.get("size_vram"), x.get("size"))); sys.exit(1)
if int(x.get("context_length", 0)) != ctx: print("context_length %s != %d" % (x.get("context_length"), ctx)); sys.exit(1)
print("loaded: %s size_vram=%.1f GB context_length=%d expires_at=%s" % (m, x["size_vram"] / 1e9, x["context_length"], x.get("expires_at")))
' "$MODEL" "$CTX" >&2 || fail "loaded model check failed"
  # the runner (llama-server child of our server) must be on the pinned GPU
  local kid on=""; for kid in $(children "$1"); do gpu_pids "$2" | grep -qx "$kid" && on="$kid"; done
  [ -n "$on" ] || fail "no child of pid $1 is on GPU $2 per nvidia-smi"
  log "runner pid $on is on GPU $2 ($(gpu_uuid "$2"))"
}

# ---- main ---------------------------------------------------------------------------------------------
if [ "$MODE" = check ]; then
  id=$(identify); set -- $id
  case $1 in none) log "not running"; exit 1;; foreign) refuse "port :$PORT: ${id#foreign }";; esac
  check_env "$2" "${3:-}"
  [ "$(api_version 10)" = "$VER" ] || fail "pid $2 is ours but /api/version does not answer"
  model_listed || fail "$MODEL is not listed by the server"
  log "ok: pid $2 GPU ${3:-} $VER, $MODEL listed; loaded: $(api /api/ps 10 | python3 -c 'import json,sys; print(",".join(x["name"] for x in json.load(sys.stdin).get("models",[])) or "none")' 2>/dev/null)"
  exit 0
fi

if [ "$MODE" = probe ]; then # read-only (no start / load / stop / lock): could the scorer serve now?
  mf="$MODELS/manifests/registry.ollama.ai/library/${MODEL%%:*}/${MODEL#*:}"
  [ -f "$mf" ] || refuse "$MODEL is not in $MODELS (no $mf)"
  id=$(identify); set -- $id
  case $1 in
    foreign) refuse "port :$PORT: ${id#foreign }";;
    ours)
      check_env "$2" "${3:-}"
      if [ "$(api_version 10)" = "$VER" ]; then
        model_listed || refuse "$MODEL is not listed by our server pid $2 on :$PORT (models dir $MODELS)"
        if model_loaded; then log "probe ok: pid $2, $MODEL loaded on GPU ${3:-}"; exit 0; fi
        why=$(gpu_why_not "${3:-}")
        [ -z "$why" ] && { log "probe ok: pid $2 on GPU ${3:-}, which can take $MODEL now"; exit 0; }
        log "probe: $why (--warm re-pins the server to a GPU that can)"
      else
        log "probe: pid $2 is ours but /api/version is silent (--warm restarts it)"
      fi;;
  esac
  [ "$(bin_version)" = "$VER" ] || refuse "$BIN is not $VER"
  a=$(mem_avail_gb); [ "$a" -ge "$MIN_AVAIL_GB" ] || busy "system MemAvailable ${a} GiB < ${MIN_AVAIL_GB} GiB"
  g=$(choose_gpu) || busy "no GPU of '$GPUS' is free enough now"
  log "probe ok: GPU $g can take $MODEL now"
  exit 0
fi

exec 9>"$STATE/lock"
flock -w 900 9 || busy "another ensure_scorer_ollama holds $STATE/lock for > 900 s"

id=$(identify); set -- $id
case $1 in
  foreign) refuse "port :$PORT: ${id#foreign }";;
  none)
    [ "$MODE" = stop ] && { log "nothing to stop"; exit 0; }
    [ "$MODE" = unload ] && { log "not running; nothing to unload"; exit 0; }
    start_server; set -- $(identify); [ "$1" = ours ] || fail "started server not identified as ours: $*";;
  ours)
    if [ "$MODE" = stop ]; then stop_ours "$2"; log "stopped"; exit 0; fi
    if [ "$(api_version 10)" != "$VER" ]; then
      log "pid $2 is ours but /api/version did not answer in 10 s: hung -> restart"
      stop_ours "$2"; start_server; set -- $(identify); [ "$1" = ours ] || fail "restarted server not identified as ours: $*"
    fi;;
esac
check_env "$2" "${3:-}"
model_listed || fail "$MODEL is not listed by the server on :$PORT (models dir $MODELS)"
if [ "$MODE" = unload ]; then
  curl -sS --max-time 60 "$URL/api/generate" -d "{\"model\":\"$MODEL\",\"keep_alive\":0}" >/dev/null || fail "unload request failed"
  log "unloaded $MODEL"; exit 0
fi
# a re-pin inside warm() replaces the server: report the one that now serves
[ "$MODE" = warm ] && { warm "$2" "${3:-}"; set -- $(identify); }
log "ok: pid $2 GPU ${3:-} $VER, $MODEL listed${MODE:+ ($MODE)}"
exit 0
