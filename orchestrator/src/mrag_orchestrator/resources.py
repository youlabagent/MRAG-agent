"""Resource caps applied at the ``mrag`` CLI entry (2026-09-16, shared-server policy).

* ``apply_cpu_cap()``  pins this process (every thread) to ``MRAG_CPU_LIST``
  (default "32-47"; "none" disables) before any worker thread or subprocess
  exists, so every child (HOMER, R, deeptools, samtools, ...) inherits the
  mask. It exports ``MRAG_NCPUS`` and, unless set, ``MC_CORES`` (R's
  parallel::detectCores() ignores the affinity and reports 128).
* ``ncpus()``          usable CPUs (the affinity, not os.cpu_count()).
* ``start_run_guard()`` marks this process as a run root
  (``MRAG_RUN_ROOT=<pid>:<starttime>``, inherited by every descendant) and makes
  sure the account-wide memory guard daemon (``resource_guard.py --auto``) is
  running. ``MRAG_MEM_GUARD=0`` disables the daemon start.
* ``check_start_memory()`` (2026-09-26) refuses to START ``mrag run`` /
  ``mrag variants`` (exit 75) while the machine's MemAvailable is below
  ``MRAG_MEM_START_FLOOR_GB`` (default 60; 0 = off). ``MRAG_MEM_IGNORE_SYSFLOOR=1``
  starts anyway; an mrag started inside a run tree (MRAG_RUN_ROOT set) is never
  refused.
* ``install_pause_aware_timeouts()`` (called by start_run_guard) makes every
  ``subprocess`` timeout of this process (run/communicate/wait/check_output)
  count only time the process was not stopped: a SIGSTOP freeze by the memory
  guard no longer eats DiffBind/scoring/PaperQA/HOMER deadlines. After a stop
  of >= 30 s a line is printed to stderr (the run log).
  ``MRAG_PAUSE_AWARE_TIMEOUTS=0`` disables it.

Nothing here raises: a failure is reported on stderr and the run continues.
"""
from __future__ import annotations

import fcntl
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Set

DEFAULT_CPU_LIST = "32-47"
DEFAULT_GUARD_DIR = "/BIO1/MRAG/Agent/orchestrator/webui/runs"
_OFF = ("none", "off", "no", "false", "disable", "disabled")


def _warn(msg: str) -> None:
    print(f"[mrag resources] {msg}", file=sys.stderr, flush=True)


def parse_cpulist(spec: str) -> Set[int]:
    """'32-47', '0-3,8,10-15:2' -> set of CPU ids."""
    cpus: Set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            step = 1
            if ":" in b:
                b, s = b.split(":", 1)
                step = int(s)
            lo, hi = int(a), int(b)
            if lo > hi or step < 1:
                raise ValueError(f"bad cpu range {part!r}")
            cpus.update(range(lo, hi + 1, step))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError(f"empty cpu list {spec!r}")
    return cpus


def format_cpulist(cpus) -> str:
    xs = sorted(cpus)
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ",".join(out)


def ncpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def apply_cpu_cap() -> Dict[str, object]:
    """Pin every thread of this process to MRAG_CPU_LIST (intersected with the
    current affinity when that is narrower)."""
    spec = (os.environ.get("MRAG_CPU_LIST") or DEFAULT_CPU_LIST).strip()
    if spec.lower() in _OFF:
        return {"applied": False, "reason": f"MRAG_CPU_LIST={spec}"}
    try:
        want = parse_cpulist(spec)
        cur = os.sched_getaffinity(0)
    except (ValueError, AttributeError, OSError) as e:
        _warn(f"CPU cap not applied ({e})")
        return {"applied": False, "reason": str(e)}
    cpus = (want & cur) or want
    try:
        tids = [int(t) for t in os.listdir("/proc/self/task")]
    except OSError:
        tids = [0]
    errors = []
    for tid in tids:
        try:
            os.sched_setaffinity(tid, cpus)
        except OSError as e:
            errors.append(f"tid {tid}: {e}")
    now = os.sched_getaffinity(0)
    n = len(now)
    os.environ["MRAG_NCPUS"] = str(n)
    os.environ.setdefault("MC_CORES", str(n))
    if errors:
        _warn("CPU cap partly failed: " + "; ".join(errors[:3]))
    return {"applied": now == cpus, "cpus": format_cpulist(now), "n": n}


def _starttime(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            return f.read().rsplit(b")", 1)[1].split()[19].decode()
    except (OSError, IndexError):
        return "0"


def guard_dir() -> Path:
    return Path(os.environ.get("MRAG_MEM_GUARD_DIR") or DEFAULT_GUARD_DIR)


def ensure_guard() -> str:
    """Start resource_guard.py --auto unless a guard holds <dir>/_mem_guard.lock.

    Returns "running", "started", "disabled" or "error: ...". The daemon is
    double-forked (setsid -f): it must not be a descendant of this run, or it
    would sit inside the tree it may freeze.
    """
    if os.environ.get("MRAG_MEM_GUARD", "1").strip().lower() in _OFF + ("0",):
        return "disabled"
    d = guard_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(d / "_mem_guard.lock"), os.O_RDWR | os.O_CREAT, 0o664)
    except OSError as e:
        return f"error: {e}"
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return "running"
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    script = Path(__file__).with_name("resource_guard.py")
    cmd = [sys.executable, "-I", str(script), "--auto", "--dir", str(d)]
    cmd += shlex.split(os.environ.get("MRAG_MEM_GUARD_ARGS", ""))
    env = {k: v for k, v in os.environ.items() if k != "MRAG_RUN_ROOT"}
    setsid = shutil.which("setsid")
    if setsid:
        argv = [setsid, "-f"] + cmd
    else:
        argv = [sys.executable, "-c",
                "import os,subprocess,sys;"
                "subprocess.Popen(sys.argv[1:],start_new_session=True,stdin=subprocess.DEVNULL,"
                "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);os._exit(0)"] + cmd
    try:
        subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, env=env, cwd=str(d), close_fds=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        return f"error: {e}"
    return "started"


class PauseAwareClock:
    """time.monotonic() minus the time this process spent stopped (SIGSTOP).

    A daemon thread wakes every ``tick`` s. When the clock has not been
    advanced for more than ``gap`` s (by the thread or by a reader), the
    interval minus the expected tick and minus the CPU time the process used
    meanwhile counts as paused: a stopped process uses no CPU, whereas a thread
    merely starved of the GIL by a busy thread does (and a genuine stall just
    lengthens a deadline, which is harmless). Readers in a forked child (no
    ticker there) get plain time.monotonic().
    """

    def __init__(self, tick: float = 1.0, gap: float = 5.0, notify_s: float = 30.0):
        self.tick, self.gap, self.notify_s = tick, gap, notify_s
        self.pid = os.getpid()
        self.lock = threading.Lock()
        self.last = time.monotonic()
        self.last_cpu = time.process_time()
        self.paused = 0.0
        self.pending = 0.0
        self.thread = threading.Thread(target=self._loop, name="mrag-pause-aware-clock", daemon=True)
        self.thread.start()

    def _account(self, now: float, cpu: float) -> None:     # lock held
        gap = now - self.last
        if gap > self.gap:
            p = gap - self.tick - max(0.0, cpu - self.last_cpu)
            if p > 0:
                self.paused += p
                self.pending += p
        self.last, self.last_cpu = now, cpu

    def _loop(self) -> None:
        while True:
            time.sleep(self.tick)
            try:
                with self.lock:
                    self._account(time.monotonic(), time.process_time())
                    pend, self.pending = self.pending, 0.0
                if pend >= self.notify_s:
                    _warn(f"{time.strftime('%Y-%m-%d %H:%M:%S')} this run was stopped for ~{pend:.0f} s "
                          f"(memory guard freeze?); subprocess timeouts do not count that time")
            except Exception:
                pass

    def __call__(self) -> float:
        if os.getpid() != self.pid:
            return time.monotonic()
        with self.lock:
            now = time.monotonic()
            if now - self.last > self.gap:
                self._account(now, time.process_time())
            return now - self.paused


def install_pause_aware_timeouts() -> str:
    """Replace subprocess's deadline clock (``subprocess._time``) in this process."""
    if os.environ.get("MRAG_PAUSE_AWARE_TIMEOUTS", "1").strip().lower() in _OFF + ("0",):
        return "disabled"
    cur = getattr(subprocess, "_time", None)
    if isinstance(cur, PauseAwareClock):
        return "on"
    if cur is not time.monotonic:
        return "skipped (subprocess._time is not time.monotonic)"
    subprocess._time = PauseAwareClock()
    return "on"


START_REFUSED_RC = 75      # EX_TEMPFAIL: try again later


def check_start_memory(meminfo: str = "/proc/meminfo") -> int:
    """0 = go on; START_REFUSED_RC = do not start (message on stderr).

    A run peaks at ~32G PSS (card retriever + workers) and ~10G for the
    HOMER filter; starting one into a machine that other users have already
    squeezed below the floor ends in thrash or in the kernel OOM killer, which
    picks our processes first (launchers set oom_score_adj=1000)."""
    if os.environ.get("MRAG_RUN_ROOT"):
        return 0                       # nested inside a running run tree: never refuse
    if os.environ.get("MRAG_MEM_IGNORE_SYSFLOOR", "").strip().lower() in ("1", "true", "yes", "on"):
        return 0
    v = os.environ.get("MRAG_MEM_START_FLOOR_GB", "").strip()
    try:
        floor = float(v) if v else 60.0
    except ValueError:
        floor = 60.0
    if floor <= 0:
        return 0
    avail_kb = None
    try:
        with open(meminfo) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
                    break
    except (OSError, ValueError, IndexError):
        avail_kb = None
    if avail_kb is None:
        return 0                       # cannot tell: never block on a read failure
    avail = avail_kb / 1048576.0
    if avail >= floor:
        return 0
    _warn(f"REFUSING TO START: the server's MemAvailable is {avail:.1f} GiB, below the start floor "
          f"{floor:.0f} GiB (MRAG_MEM_START_FLOOR_GB). Other users hold most of the memory right now; a run "
          f"(~32 GiB peak) started now would thrash or be OOM-killed. Retry later, or set "
          f"MRAG_MEM_IGNORE_SYSFLOOR=1 to start anyway.")
    return START_REFUSED_RC


def start_run_guard() -> str:
    """Mark this process as a run root and make sure the memory guard runs."""
    try:
        os.environ["MRAG_RUN_ROOT"] = f"{os.getpid()}:{_starttime(os.getpid())}"
        state = ensure_guard()
    except Exception as e:  # never block a run on the guard
        state = f"error: {type(e).__name__}: {e}"
    try:
        clock = install_pause_aware_timeouts()
    except Exception as e:
        clock = f"error: {type(e).__name__}: {e}"
    try:
        cpus = format_cpulist(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = "?"
    _warn(f"CPUs {cpus} ({ncpus()}); memory guard {state} ({guard_dir()}/_mem_guard.log); "
          f"pause-aware subprocess timeouts {clock}")
    return state
