"""A polite lease on one CUDA device, for run steps that need a GPU (2026-10-18, Section-13 stage 2b).

WHY A LEASE.  The three RTX A6000s (49,140 MiB each) are shared, and nothing in the orchestrator
arbitrated them before this module (resources.py and resource_guard.py cap CPU and RAM only):

* GPU 1 carries the production LLM ollama on :11500.  ensure_scorer_ollama.sh refuses to touch it
  (PROD_GPU=1), and so does this module.
* The target scorer on :11501 takes the FIRST of SCORER_GPUS ("2 0") whose free VRAM is at least
  SCORER_MIN_FREE_MIB (32,000: qwen3.8 Q8 at ctx 32768 is 30.2 GiB including context), holds it
  while its model is loaded (OLLAMA_KEEP_ALIVE 30m) and releases it on --unload.

So a GPU step of a run shares a card with the scorer whether it likes it or not.  What it must not
do is take so much that the scorer's own check fails -- that is reported as "no GPU / RAM free now,
retry later" (exit 2) and it stops target scoring.  Corgi's measured peak is 4.52 GiB, so the
default cap here is 7.4 GiB = 7,577 MiB (int(7.4 * 1024)), so 49,140 - 7,577 = 41,563 MiB
stays free on an idle card, well above the scorer's 32,000.
The research setting (torch fraction 0.5, about 24.5 GiB) leaves 24 GiB and would block it.

WHAT IT DOES.  flock on <lock_dir>/_gpu<N>.lock, the convention resources.ensure_guard already uses
for _mem_guard.lock, so two concurrent runs cannot put their GPU step on the same card; plus a free-
VRAM check before handing the card over.  The lock is advisory and only between lease users: it does
not and cannot stop ollama or a foreign process from taking the card afterwards.

NOTHING HERE RAISES.  No usable card means `lease()` returns None, the caller skips its GPU arm and
the report says which arm is missing and why -- the model-arm failure policy applied to the
AlphaGenome step.  A caller that treats None as fatal is a bug in the caller.

    with gpu_lease.lease(need_gib=8) as g:
        if not g:
            record(status="skipped", reason=g.reason)      # the reason belongs in the report
        else:
            subprocess.run([...], env={**os.environ, **g.env()})

A lease is truthy, a refusal is falsy and carries .reason and .per_gpu -- so the caller can state
WHICH card refused and why, which is the whole point of not raising.
"""
from __future__ import annotations

import fcntl
import os
import subprocess
import time

PROD_GPU = 1                     # :11500, the production LLM. Never leased, mirroring ensure_scorer_ollama.sh:49
SCORER_GPUS = (2, 0)             # ensure_scorer_ollama.sh:35 default SCORER_GPUS="2 0"
SCORER_MIN_FREE_MIB = 32_000     # ensure_scorer_ollama.sh:40, so a lease must leave at least this much
DEFAULT_CAP_GIB = 7.4            # Corgi peak 4.52 GiB measured 2026-10-18; see the header
LOCK_DIR = "/BIO1/MRAG/Agent/orchestrator/webui/runs"
TOTAL_MIB = 49_140               # A6000; only used to express the cap as a torch memory fraction


def device_count(timeout_s=10):
    """How many cards the driver reports, or None when it cannot be asked (never raises)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=timeout_s)
        if out.returncode != 0:
            return None
        return len([l for l in out.stdout.splitlines() if l.strip()])
    except Exception:
        return None


def candidates(gpus=None):
    """The cards a lease may use, in preference order.

    Default: the scorer's own list REVERSED, so a lease prefers the card the scorer reaches for
    last.  The scorer takes the first of "2 0" that is free, so a lease starts at 0 and the two
    collide only when both cards are in play anyway.  PROD_GPU is never a candidate.

    MRAG_GPU_LIST overrides it.  Set but EMPTY means no card, i.e. the caller's GPU arm is off --
    `export MRAG_GPU_LIST=` has to be a way to switch it off, not a way to get the default back.
    Junk is dropped rather than raised (review 2026-10-18: `int()` on 'abc' broke the module's own
    no-raise contract), and so is an index the driver does not report, which otherwise created a
    stray _gpu-1.lock in the live runs directory before nvidia-smi refused it.
    """
    from_env = False
    if gpus is None:
        raw = os.environ.get("MRAG_GPU_LIST")
        if raw is None:
            gpus = list(reversed(SCORER_GPUS))
        else:
            from_env = True
            gpus = []
            for tok in raw.replace(",", " ").split():
                try:
                    gpus.append(int(tok))
                except ValueError:
                    pass                      # a typo must not take the GPU arm down
    n = device_count()
    out = []
    for g in gpus:
        if g == PROD_GPU or g in out or g < 0:
            continue
        if n is not None and g >= n:
            continue                          # the driver does not have this card
        out.append(g)
    if from_env and not out and os.environ.get("MRAG_GPU_LIST", "").strip() == "":
        return []                             # explicitly switched off
    return out


def free_mib(index, timeout_s=10):
    """Free VRAM of one card, or None when nvidia-smi cannot say (never raises)."""
    try:
        out = subprocess.run(["nvidia-smi", "--id=%d" % index, "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=timeout_s)
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0].strip())
    except Exception:
        return None


class Lease:
    """A held card.  Released by the context manager, or by close().  Truthy (see _NoLease)."""

    reason = ""

    def __init__(self, index, fd, cap_gib, free_before):
        self.index = index
        self.cap_gib = cap_gib
        self.free_before_mib = free_before
        self._fd = fd

    @property
    def memory_fraction(self):
        """What to pass to torch.cuda.set_per_process_memory_fraction, as a share of the whole card."""
        return round(min(1.0, (self.cap_gib * 1024.0) / TOTAL_MIB), 4)

    def env(self):
        """The environment a child needs. CUDA_VISIBLE_DEVICES=<card>, so the child sees it as cuda:0."""
        return {"CUDA_VISIBLE_DEVICES": str(self.index),
                "MRAG_GPU_MEMORY_FRACTION": str(self.memory_fraction)}

    def close(self):
        """Release.  Never raises: a card whose fd died elsewhere is already not held."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass

    def __del__(self):
        # a Lease dropped without close() would otherwise read as "leased by another run" until the
        # process exits (review 2026-10-18)
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        self.close()
        return False

    def __repr__(self):
        return ("Lease(gpu=%d, cap=%.1f GiB, fraction=%.4f, free_before=%s MiB)"
                % (self.index, self.cap_gib, self.memory_fraction, self.free_before_mib))


class _NoLease:
    """What `lease(...)` returns when no card is usable: falsy, and it carries the reason.

    __enter__ yields SELF, not None, so `with lease() as g: if not g: ... g.reason` reads the reason.
    Yielding None would throw away the one thing the caller has to report.
    """

    index = None
    cap_gib = 0.0
    memory_fraction = 0.0

    def __init__(self, reason, per_gpu):
        self.reason = reason
        self.per_gpu = dict(per_gpu)

    def __bool__(self):
        return False

    def env(self):
        """No card: a child must not see a stale CUDA_VISIBLE_DEVICES from the parent."""
        return {"CUDA_VISIBLE_DEVICES": ""}

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def close(self):
        pass

    def __repr__(self):
        return "NoLease(%s)" % self.reason


def lease(need_gib=DEFAULT_CAP_GIB, gpus=None, lock_dir=None, keep_free_mib=SCORER_MIN_FREE_MIB,
          wait_s=0, poll_s=15, log=None):
    """Lease one card, or return a falsy _NoLease whose .reason says why not.

    need_gib        what the step may use; also the cap reported as `memory_fraction`
    keep_free_mib   leave at least this much for the scorer AFTER the step's cap is taken.  With the
                    default 32,000 and need_gib 7.4 a card needs 39,577 MiB (38.7 GiB) free, i.e.
                    an idle card.
                    Pass 0 to share a card with a loaded scorer model (then the scorer cannot restart
                    there until this step ends -- say so in the report if you do).
    wait_s          keep retrying for this long (a run may start while the scorer holds a card)
    """
    lock_dir = lock_dir or os.environ.get("MRAG_GPU_LOCK_DIR") or LOCK_DIR
    need_mib = int(need_gib * 1024)
    # poll_s <= 0 would raise in time.sleep, or spin re-forking nvidia-smi for the whole wait
    poll_s = max(1, int(poll_s or 0))
    deadline = time.time() + max(0, wait_s)
    say = log or (lambda m: None)
    cands = candidates(gpus)
    if not cands:
        return _NoLease("no candidate GPU (MRAG_GPU_LIST empty, or every card is the production "
                        "GPU / not reported by the driver)", {})
    while True:
        why = {}
        for g in cands:
            fd = None
            try:
                path = os.path.join(lock_dir, "_gpu%d.lock" % g)
                fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o664)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    why[g] = "leased by another run"
                    os.close(fd)
                    continue
                f = free_mib(g)
                if f is None:
                    why[g] = "free memory unreadable (nvidia-smi)"
                elif f < need_mib + keep_free_mib:
                    why[g] = ("%d MiB free < %d needed + %d kept for the scorer"
                              % (f, need_mib, keep_free_mib))
                else:
                    say("[gpu] leased GPU %d (%d MiB free; cap %.1f GiB)" % (g, f, need_gib))
                    return Lease(g, fd, need_gib, f)
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            except Exception as e:                      # a lease never raises
                why[g] = "%s: %s" % (type(e).__name__, e)
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
        if time.time() >= deadline:
            return _NoLease("; ".join("GPU %d %s" % (g, why.get(g, "?")) for g in cands), why)
        say("[gpu] no card yet (%s); retrying in %ds"
            % ("; ".join("GPU %d %s" % (g, why.get(g, "?")) for g in cands), poll_s))
        time.sleep(min(poll_s, max(1, deadline - time.time())))
