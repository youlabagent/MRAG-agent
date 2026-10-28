#!/usr/bin/env python3
"""Account-wide memory guard for MRAG runs (user level: no root, no cgroups).

One daemon per account, singleton by flock on <dir>/_mem_guard.lock; started
by every ``mrag run`` / ``mrag variants`` (resources.ensure_guard). Stdlib
only and Python 3.8 compatible. Every --interval s it scans /proc for the
processes of its own uid:

* run roots (freezable): ``[python] .../mrag run|variants ...`` or
  ``python -m mrag_orchestrator.cli run|variants``. A root's tree is its ppid
  descendants plus any process of this uid whose environment carries
  ``MRAG_RUN_ROOT=<root pid>:<root starttime>`` (orphans reparented to init).
* account-only: ``ollama serve`` trees - counted, never signalled.
* total: RSS of every process of this uid (--scope account, default) or of the
  run + account-only trees only (--scope tracked).

Policy:
* total > soft (MRAG_MEM_SOFT, default 0.20 x MemTotal): WARN once.
* total > hard (MRAG_MEM_HARD, 0.30) for --confirm-ticks samples in a row
  (a fork briefly double-counts RSS): SIGSTOP the unfrozen run tree that grows
  fastest. A SIGSTOP frees nothing, it only stops growth, so a tree is frozen
  only if it is part of the growth: growth = RSS now - its minimum over the
  last --growth-window-min, and the tree must have grown >= --grow-margin-gb and
  >= --min-growth-share of the growth of everything still running (run trees
  incl. hands-off ones, account-only trees, the other processes of the uid).
  When the excess comes from a hands-off run, other processes or static trees,
  nothing is frozen and OVER is logged with the growth attribution.
  Another tree is frozen only while the total keeps growing (> --grow-margin-gb
  above the total at the last freeze), and never the last running run tree
  while another run tree is stopped (frozen RSS still counts, so a cascade could
  stop every run with a total that can no longer fall).
* one frozen tree resumes per tick, FIFO, when total < MRAG_MEM_RESUME (0.85)
  x hard. Only trees this guard froze are ever resumed.
* DEADLOCK-BREAK: all live run trees (>= 2) are stopped, this guard froze at
  least one of them, and the total stays >= the resume line for --confirm-ticks
  samples: resume the oldest tree this guard froze and exempt it from freezing
  until the total drops below the resume line.
* STUCK: a tree stays frozen --stuck-minutes while even without the running
  run trees the total would stay above the resume line (waiting cannot help,
  e.g. one run tree alone above it): log STUCK and apply --on-stuck
  (MRAG_MEM_ON_STUCK): "hold" (default) keeps it frozen, "resume" resumes the
  oldest and exempts it from freezing until the total drops below the resume
  line. (A single run tree above the line stays frozen with "hold".)
* SYSTEM FLOOR (2026-09-26): the limits above only see this uid; other users
  can still exhaust the machine. When /proc/meminfo MemAvailable stays below
  MRAG_MEM_SYS_FLOOR_GB (default 40; 0 = off) for --confirm-ticks samples, the
  unfrozen run tree that grows fastest is frozen, with the same rules as above
  (growth >= --grow-margin-gb and >= --min-growth-share of the running growth;
  never the last running run tree while another run tree is stopped; another
  system-floor freeze only while MemAvailable keeps falling, i.e. more than
  --grow-margin-gb below its value at the last one). Growth here is measured on
  PSS, not RSS: a parent that forks workers doubles its RSS sum without using
  more memory (card retriever: 8 workers, ~200G RSS sum vs ~32G PSS). A tree
  that does not grow is never frozen (a SIGSTOP frees nothing). While
  MemAvailable < MRAG_MEM_SYS_RESUME_GB (default 80) nothing is resumed (the
  account resume rule above also waits for it); above it one frozen tree resumes
  per tick, FIFO. DEADLOCK-BREAK also applies when all run trees are stopped
  and MemAvailable stays below the resume mark. STUCK: frozen --stuck-minutes
  with MemAvailable still below the resume mark: log STUCK and apply
  MRAG_MEM_SYS_ON_STUCK ("hold" default | "resume" the oldest, exempt until
  MemAvailable is back above the resume mark).
* The daemon resets its own oom_score_adj to MRAG_MEM_GUARD_OOM_ADJ (default 0;
  "keep" = untouched): the run launchers set 1000, which the daemon inherited,
  and the kernel killed it with the run on 2026-09-26 19:56 -- a guard killed
  while holding a frozen tree would leave that tree stopped for good.

Notices: FREEZE / RESUME / STUCK are also appended to "<run stdout log>.memguard.txt"
(e.g. run_dir/run.log.memguard.txt) when the run root's stdout is a regular file.

Hand-over from explicit guards: while a ``_mem_guard.py --run PID ...`` process
of this uid lives, its PIDs are counted but never signalled here. A stopped
run root this guard did not freeze is logged (ORPHAN-STOPPED), never resumed.

Safety: only processes of this uid whose (pid, starttime) match a fresh scan
are signalled; never this daemon, its ancestors, a legacy guard or an
account-only tree. SIGTERM / SIGINT / SIGHUP resume everything this guard
froze before it exits. It exits by itself after --idle-exit s without run
roots and frozen trees.

Files in --dir (default /BIO1/MRAG/Agent/orchestrator/webui/runs):
  _mem_guard.lock  _mem_guard.pid  _mem_guard.log  _mem_guard.state.json
  _mem_guard.status.json
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import sys
import time
import traceback
from datetime import datetime

DEFAULT_DIR = "/BIO1/MRAG/Agent/orchestrator/webui/runs"
RUN_SUBCOMMANDS = ("run", "variants")
KB_PER_GIB = 1048576


def gib(kb):
    return kb / KB_PER_GIB


# ---------------------------------------------------------------------------
# /proc
# ---------------------------------------------------------------------------
class Proc(object):
    __slots__ = ("pid", "ppid", "state", "start", "uid", "argv", "rss_kb")

    def __init__(self, pid, ppid, state, start, uid, argv, rss_kb):
        self.pid, self.ppid, self.state, self.start = pid, ppid, state, start
        self.uid, self.argv, self.rss_kb = uid, argv, rss_kb

    def key(self):
        return "%d:%d" % (self.pid, self.start)


_PAGE_KB = max(1, os.sysconf("SC_PAGE_SIZE") // 1024)


def read_proc(pid):
    base = "/proc/%d" % pid
    uid = os.stat(base).st_uid
    with open(base + "/stat", "rb") as f:
        rest = f.read().rsplit(b")", 1)[1].split()
    with open(base + "/cmdline", "rb") as f:
        raw = f.read()
    argv = [a.decode("utf-8", "replace") for a in raw.rstrip(b"\0").split(b"\0")] if raw else []
    with open(base + "/statm", "rb") as f:
        rss_pages = int(f.read().split()[1])
    return Proc(pid, int(rest[1]), rest[0].decode(), int(rest[19]), uid, argv, rss_pages * _PAGE_KB)


def scan(uid):
    """Processes of ``uid`` (None = all). Other users' /proc entries are only
    stat()ed: reading cmdline/statm of a busy foreign process can be slow."""
    procs = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            if uid is not None and os.stat("/proc/" + d).st_uid != uid:
                continue
            p = read_proc(int(d))
        except (OSError, IndexError, ValueError):
            continue
        if uid is None or p.uid == uid:
            procs[p.pid] = p
    return procs


def env_marker(pid):
    try:
        with open("/proc/%d/environ" % pid, "rb") as f:
            data = f.read()
    except OSError:
        return None
    for item in data.split(b"\0"):
        if item.startswith(b"MRAG_RUN_ROOT="):
            return item[len(b"MRAG_RUN_ROOT="):].decode("ascii", "replace")
    return None


def stdout_log_of(pid):
    """The regular file a process writes its stdout (else stderr) to, or None."""
    for fd in (1, 2):
        try:
            p = os.readlink("/proc/%d/fd/%d" % (pid, fd))
        except OSError:
            continue
        if p.startswith("/") and not p.startswith("/dev/") and not p.endswith(" (deleted)") and os.path.isfile(p):
            return p
    return None


def mem_total_kb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    raise RuntimeError("MemTotal not found")


MEMINFO = "/proc/meminfo"


def mem_available_kb(path=None):
    """MemAvailable of the whole machine (kB), None if unreadable."""
    try:
        with open(path or MEMINFO) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def read_pss_kb(pid):
    """Proportional set size (kB) of one process from /proc/PID/smaps_rollup, None
    if unreadable. Pages shared by forked workers are split among them, so a
    fork does not look like growth (their RSS sum counts those pages N times)."""
    try:
        with open("/proc/%d/smaps_rollup" % pid, "rb") as f:
            for line in f:
                if line.startswith(b"Pss:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


# ---------------------------------------------------------------------------
# classification (pure functions, unit-tested with injected argv)
# ---------------------------------------------------------------------------
def _is_python(arg):
    return os.path.basename(arg).startswith("python")


def is_run_root_argv(argv, subcommands=RUN_SUBCOMMANDS):
    if not argv:
        return False
    i = 0
    if _is_python(argv[0]):
        i = 1
        while i < len(argv) and argv[i].startswith("-") and argv[i] != "-m":
            i += 1
        if i < len(argv) and argv[i] == "-m":
            return (i + 2 < len(argv) and argv[i + 1] in ("mrag_orchestrator.cli", "mrag_orchestrator")
                    and argv[i + 2] in subcommands)
    return (i + 1 < len(argv) and os.path.basename(argv[i]) == "mrag"
            and argv[i + 1] in subcommands)


def is_account_argv(argv):
    return bool(argv) and os.path.basename(argv[0]) == "ollama" and len(argv) > 1 and argv[1] == "serve"


def legacy_guard_runs(argv):
    """PIDs passed with --run to an explicit (legacy) _mem_guard.py, else None."""
    if not argv or "--auto" in argv:
        return None
    if not any(os.path.basename(a) == "_mem_guard.py" for a in argv[:3]):
        return None
    runs = []
    for i, a in enumerate(argv):
        v = None
        if a == "--run" and i + 1 < len(argv):
            v = argv[i + 1]
        elif a.startswith("--run="):
            v = a.split("=", 1)[1]
        if v is not None:
            try:
                runs.append(int(v))
            except ValueError:
                pass
    return runs if runs else None


def build_model(procs, self_pid, is_root=is_run_root_argv, is_account=is_account_argv,
                marker=None):
    """Classify a /proc snapshot of one uid.

    Returns dict: roots [pid], key {pid: "pid:start"}, members {root: [pids]},
    account {root: [pids]}, handsoff {root pids owned by a live legacy guard},
    protected {pids never signalled}, legacy {guard pid: [run pids]}.
    """
    marker = marker or env_marker
    kids = {}
    for p in procs.values():
        kids.setdefault(p.ppid, []).append(p.pid)
    protected = set()
    x = self_pid
    while x in procs and x not in protected:
        protected.add(x)
        x = procs[x].ppid
    protected.add(self_pid)
    legacy = {}
    for pid, p in procs.items():
        runs = legacy_guard_runs(p.argv)
        if runs is not None:
            legacy[pid] = runs
            protected.add(pid)
    handsoff_pids = set()
    for runs in legacy.values():
        handsoff_pids.update(runs)

    owner = {}          # pid -> ("acct"|"run", root)

    def claim(start, kind, root):
        stack = [start]
        while stack:
            q = stack.pop()
            if q in owner or q in protected:
                continue
            owner[q] = (kind, root)
            stack.extend(kids.get(q, []))

    acct_roots = [pid for pid, p in procs.items() if pid not in protected and is_account(p.argv)]
    for a in acct_roots:
        claim(a, "acct", a)
    cands = [pid for pid, p in procs.items()
             if pid not in protected and pid not in owner and is_root(p.argv)]
    candset = set(cands)

    def under_other_root(pid):
        seen = set()
        x = procs[pid].ppid
        while x in procs and x not in seen:
            if x in candset:
                return True
            seen.add(x)
            x = procs[x].ppid
        return False

    roots = sorted(r for r in cands if not under_other_root(r))
    key = dict((r, procs[r].key()) for r in roots)
    for r in roots:
        claim(r, "run", r)
    by_key = dict((k, r) for r, k in key.items())
    if by_key:
        for pid in sorted(procs):
            if pid in owner or pid in protected:
                continue
            m = marker(pid)
            if m and m in by_key:
                claim(pid, "run", by_key[m])
    members = dict((r, []) for r in roots)
    account = dict((a, []) for a in acct_roots)
    for pid, (kind, root) in owner.items():
        (members if kind == "run" else account)[root].append(pid)
    return {"roots": roots, "key": key, "members": members, "account": account,
            "handsoff": set(r for r in roots if r in handsoff_pids),
            "protected": protected, "legacy": legacy}


# ---------------------------------------------------------------------------
# daemon
# ---------------------------------------------------------------------------
def _frac_or_gb(env_name, default_frac, total_kb):
    v = os.environ.get(env_name, "").strip()
    try:
        x = float(v) if v else default_frac
    except ValueError:
        x = default_frac
    return x * total_kb if x <= 1.0 else x * KB_PER_GIB


def _gb_opt(cli_value, env_name, default_gb):
    """A GiB setting: the command-line value (tests) wins, then the env, then the default."""
    if cli_value is not None:
        return max(0.0, float(cli_value))
    v = os.environ.get(env_name, "").strip()
    try:
        return max(0.0, float(v)) if v else float(default_gb)
    except ValueError:
        return float(default_gb)


class Guard(object):
    def __init__(self, a):
        self.a = a
        self.uid = os.getuid()
        self.pid = os.getpid()
        self.dir = a.dir
        total_kb = mem_total_kb()
        self.soft = a.soft_gb * KB_PER_GIB if a.soft_gb else _frac_or_gb("MRAG_MEM_SOFT", 0.20, total_kb)
        self.hard = a.hard_gb * KB_PER_GIB if a.hard_gb else _frac_or_gb("MRAG_MEM_HARD", 0.30, total_kb)
        try:
            self.resume = float(os.environ.get("MRAG_MEM_RESUME", "") or a.resume)
        except ValueError:
            self.resume = a.resume
        self.on_stuck = (os.environ.get("MRAG_MEM_ON_STUCK") or a.on_stuck).strip().lower()
        # system floor (2026-09-26): the machine's MemAvailable (other users too)
        self.sys_floor = _gb_opt(a.sys_floor_gb, "MRAG_MEM_SYS_FLOOR_GB", 40.0) * KB_PER_GIB
        self.sys_resume = max(self.sys_floor,
                              _gb_opt(a.sys_resume_gb, "MRAG_MEM_SYS_RESUME_GB", 80.0) * KB_PER_GIB)
        self.sys_on_stuck = (os.environ.get("MRAG_MEM_SYS_ON_STUCK") or a.sys_on_stuck).strip().lower()
        self.meminfo = a.meminfo or MEMINFO
        self.sys_low_ticks = 0
        self.last_sys_freeze_avail = None
        rr = re.compile(a.match_run) if a.match_run else None
        ra = re.compile(a.match_account) if a.match_account else None
        self.is_root = (lambda argv: bool(argv) and bool(rr.search(" ".join(argv)))) if rr else is_run_root_argv
        self.is_account = (lambda argv: bool(argv) and bool(ra.search(" ".join(argv)))) if ra else is_account_argv
        self.frozen = []          # [{"key","pid","start","since","tree_gb"}], FIFO
        self.exempt = set()
        self.known = {}
        self.logged = set()
        self.warned = False
        self.over_ticks = 0
        self.last_freeze_total = 0
        self.stuck_since = None
        self.hist = {}            # component -> [(time, kb)] over --growth-window-min
        self.deadlock_ticks = 0
        self.last_active = time.time()
        self.stop = False

    # -- own oom_score_adj ---------------------------------------------------------
    def set_oom_adj(self):
        """Launchers put oom_score_adj=1000 on the run and the daemon inherits it;
        the guard (~10 MB) must outlive the trees it froze. An unprivileged
        process may lower the value back to what it was before it was raised."""
        v = (os.environ.get("MRAG_MEM_GUARD_OOM_ADJ") or "0").strip().lower()
        if v in ("keep", "none", "off"):
            return
        try:
            want = int(v)
            with open("/proc/self/oom_score_adj") as f:
                cur = int(f.read().strip())
            if cur != want:
                with open("/proc/self/oom_score_adj", "w") as f:
                    f.write(str(want))
                self.log("oom_score_adj %d -> %d (the guard must outlive the runs it may freeze)" % (cur, want))
        except (OSError, ValueError) as e:
            self.log("oom_score_adj left unchanged: %s" % e)

    # -- files -----------------------------------------------------------------
    def path(self, name):
        return os.path.join(self.dir, name)

    def log(self, msg):
        line = "%s auto[%d] %s\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.pid, msg)
        try:
            with open(self.path("_mem_guard.log"), "a") as f:
                f.write(line)
        except OSError:
            sys.stderr.write(line)

    def _write_json(self, name, obj):
        tmp = self.path(name + ".tmp%d" % self.pid)
        try:
            with open(tmp, "w") as f:
                json.dump(obj, f, indent=1)
            os.replace(tmp, self.path(name))
        except OSError:
            pass

    def save_state(self):
        self._write_json("_mem_guard.state.json",
                         {"pid": self.pid, "frozen": self.frozen, "exempt": sorted(self.exempt),
                          "last_freeze_total_kb": self.last_freeze_total})

    def load_state(self, procs):
        try:
            with open(self.path("_mem_guard.state.json")) as f:
                st = json.load(f)
        except (OSError, ValueError):
            return
        for fz in st.get("frozen", []):
            p = procs.get(int(fz.get("pid", -1)))
            if p is not None and p.key() == fz.get("key") and p.state in ("T", "t"):
                self.frozen.append(fz)
                self.log("ADOPT frozen run %s from previous guard %s" % (fz["key"], st.get("pid")))
        self.last_freeze_total = int(st.get("last_freeze_total_kb", 0)) if self.frozen else 0
        self.save_state()

    # -- signalling ------------------------------------------------------------
    def signal_tree(self, root_pid, root_key, sig, allow_handsoff=False):
        """Signal the current tree of a run root; returns the number of processes.
        A hands-off root (owned by a legacy guard) is only ever signalled to
        resume a freeze of this guard (allow_handsoff)."""
        done = set()
        for _ in range(2):          # second pass: children forked meanwhile
            procs = scan(self.uid)
            root = procs.get(root_pid)
            if root is None or root.key() != root_key:
                break
            m = build_model(procs, self.pid, self.is_root, self.is_account)
            if root_pid not in m["members"] or (root_pid in m["handsoff"] and not allow_handsoff):
                break
            for pid in m["members"][root_pid]:
                p = procs.get(pid)
                if p is None or p.uid != self.uid or pid in m["protected"] or pid == self.pid:
                    continue
                try:
                    if read_proc(pid).key() != p.key():
                        continue
                    os.kill(pid, sig)
                    done.add(pid)
                except OSError:
                    pass
        return len(done)

    def resume_entry(self, fz, why):
        n = self.signal_tree(int(fz["pid"]), fz["key"], signal.SIGCONT, allow_handsoff=True)
        self.log("RESUME run %s (%d procs) %s" % (fz["key"], n, why))
        self.notice(fz, "RESUMED (SIGCONT, %d procs) by the MRAG memory guard (pid %d) %s" % (n, self.pid, why))

    # -- growth attribution ------------------------------------------------------
    def _record(self, now, comps):
        """Keep (time, kb) samples of each component for --growth-window-min."""
        horizon = now - self.a.growth_window_min * 60.0
        for cid, kb in comps.items():
            h = self.hist.setdefault(cid, [])
            h.append((now, kb))
            while len(h) > 1 and h[0][0] < horizon:
                h.pop(0)
        for cid in list(self.hist):
            if cid not in comps:
                del self.hist[cid]

    def growth(self, cid):
        """RSS now minus the minimum over the window (kb, >= 0)."""
        h = self.hist.get(cid)
        if not h:
            return 0
        return h[-1][1] - min(kb for _, kb in h)

    def choose(self, roots, key, tree_kb, stopped, freezable, prefix="run:"):
        """The run tree to freeze, or (None, reason). A SIGSTOP frees nothing,
        it only stops growth: freeze the fastest-growing freezable tree, and only
        if it is a real part of the current growth."""
        a = self.a
        g = dict((r, self.growth(prefix + key[r])) for r in roots)
        pressure = (sum(max(0, g[r]) for r in roots if r not in stopped)
                    + max(0, self.growth("acct")) + max(0, self.growth("other")))
        if not freezable:
            return None, "none", g, pressure
        margin = a.grow_margin_gb * KB_PER_GIB
        cands = [r for r in freezable if g[r] >= margin and g[r] >= a.min_growth_share * pressure]
        if not cands:
            return None, "not-growing", g, pressure
        big = max(cands, key=lambda r: (g[r], tree_kb[r]))
        if len(roots) > 1 and all(r in stopped for r in roots if r != big):
            return None, "last-running", g, pressure
        return big, "", g, pressure

    def attribution(self, roots, key, m, stopped, limit=5, prefix="run:"):
        items = [(self.growth(prefix + key[r]), "run %s%s%s" % (
            key[r], " (hands-off)" if r in m["handsoff"] else "", " (stopped)" if r in stopped else ""))
            for r in roots]
        items.append((self.growth("acct"), "account-only"))
        items.append((self.growth("other"), "other processes"))
        items.sort(key=lambda x: -x[0])
        return ", ".join("%s %+.1fG" % (t, gib(v)) for v, t in items[:limit])

    def notice(self, fz, text):
        """Append a line next to the run's own stdout log (<log>.memguard.txt)."""
        path = fz.get("notice")
        if not path:
            return
        try:
            with open(path + ".memguard.txt", "a") as f:
                f.write("%s %s\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), text))
        except OSError:
            pass

    # -- system floor ------------------------------------------------------------------
    def sys_floor_step(self, now, roots, key, m, procs, tree_kb, pss_kb, stopped, freezable,
                       frozen_now, avail, total):
        """MemAvailable < sys floor for --confirm-ticks samples: freeze the run tree
        whose PSS grows fastest, with the account rules (see the module doc)."""
        a = self.a
        margin = a.grow_margin_gb * KB_PER_GIB
        if self.sys_low_ticks < a.confirm_ticks:
            return
        if (any(f.get("why") == "sysfloor" for f in self.frozen)
                and not (self.last_sys_freeze_avail is not None
                         and avail < self.last_sys_freeze_avail - margin)):
            for fz in self.frozen:
                tag = "syshold:" + fz["key"]
                if tag not in self.logged:
                    self.logged.add(tag)
                    self.log("HOLD run %s frozen; MemAvailable %.1fG < sys floor %.1fG" % (
                        fz["key"], gib(avail), gib(self.sys_floor)))
            return
        big, why, g, pressure = self.choose(roots, key, pss_kb, stopped, freezable, prefix="pss:run:")
        if big is None:
            tag = "sysnofreeze:" + why
            if tag not in self.logged:
                self.logged.add(tag)
                reason = {
                    "none": "no freezable run tree (hands-off / exempt / stopped / none)",
                    "not-growing": "no freezable run tree grew >= %.1fG PSS and >= %.0f%% of the running growth "
                                   "(the shortage comes from elsewhere; a SIGSTOP would free nothing)" % (
                                       a.grow_margin_gb, 100 * a.min_growth_share),
                    "last-running": "the growing tree is the last running run tree while the others are "
                                    "stopped (freezing it could stop every run for good)",
                }.get(why, why)
                self.log("SYSLOW MemAvailable %.1fG < sys floor %.1fG, nothing frozen: %s; this uid %.1fG RSS; "
                         "PSS growth in %g min: %s" % (gib(avail), gib(self.sys_floor), reason, gib(total),
                                                       a.growth_window_min,
                                                       self.attribution(roots, key, m, stopped, prefix="pss:run:")))
            return
        top = max((p for p in m["members"][big] if p in procs), key=lambda p: procs[p].rss_kb)
        n = self.signal_tree(big, key[big], signal.SIGSTOP)
        fz = {"key": key[big], "pid": big, "start": procs[big].start, "since": now,
              "tree_gb": round(gib(tree_kb[big]), 1), "pss_gb": round(gib(pss_kb[big]), 1),
              "growth_gb": round(gib(g[big]), 1), "why": "sysfloor", "avail_gb": round(gib(avail), 1),
              "notice": stdout_log_of(big)}
        self.frozen.append(fz)
        frozen_now.add(big)
        self.last_sys_freeze_avail = avail
        self.save_state()
        self.log("FREEZE run %s (sysfloor: MemAvailable %.1fG < floor %.1fG for %d samples; %d procs, PSS %.1fG "
                 "(RSS %.1fG), PSS grew %+.1fG in %g min = %.0f%% of the running growth, top %d %s %.1fG RSS); "
                 "resumes when MemAvailable >= %.1fG" % (
                     key[big], gib(avail), gib(self.sys_floor), self.sys_low_ticks, n, gib(pss_kb[big]),
                     gib(tree_kb[big]), gib(g[big]), a.growth_window_min,
                     100.0 * g[big] / pressure if pressure else 100.0, top,
                     os.path.basename(procs[top].argv[0]) if procs[top].argv else "?",
                     gib(procs[top].rss_kb), gib(self.sys_resume)))
        self.notice(fz, "FROZEN (SIGSTOP) by the MRAG memory guard (pid %d): the server's MemAvailable %.1fG fell "
                        "below the system floor %.1fG (other users' memory counts too) while this run's tree "
                        "(PSS %.1fG) grew %+.1fG in the last %g min. It is resumed automatically when MemAvailable "
                        "is back above %.1fG. Guard log/state: %s (_mem_guard.log, _mem_guard.status.json)." % (
                            self.pid, gib(avail), gib(self.sys_floor), gib(pss_kb[big]), gib(g[big]),
                            a.growth_window_min, gib(self.sys_resume), self.dir))

    # -- one sample --------------------------------------------------------------
    def tick(self):
        a = self.a
        now = time.time()
        procs = scan(self.uid)
        m = build_model(procs, self.pid, self.is_root, self.is_account)
        roots, key = m["roots"], m["key"]
        live = dict((key[r], r) for r in roots)
        tree_kb = dict((r, sum(procs[p].rss_kb for p in m["members"][r] if p in procs)) for r in roots)
        acct_kb = sum(procs[p].rss_kb for ms in m["account"].values() for p in ms if p in procs)
        if a.scope == "tracked":
            total = sum(tree_kb.values()) + acct_kb
        else:
            total = sum(p.rss_kb for p in procs.values())
        other_kb = max(0, total - sum(tree_kb.values()) - acct_kb)
        resume_line = self.resume * self.hard
        comps = dict(("run:" + key[r], tree_kb[r]) for r in roots)
        comps["acct"], comps["other"] = acct_kb, other_kb
        # system floor inputs: MemAvailable, and each run tree's PSS (growth series
        # "pss:run:<key>") so that a fork is not mistaken for growth
        avail = mem_available_kb(self.meminfo) if self.sys_floor > 0 else None
        sys_on = avail is not None
        sys_low = sys_on and avail < self.sys_floor
        sys_ok = (not sys_on) or avail >= self.sys_resume
        self.sys_low_ticks = self.sys_low_ticks + 1 if sys_low else 0
        pss_kb = {}
        if self.sys_floor > 0:
            for r in roots:
                s = 0
                for p in m["members"][r]:
                    if p in procs:
                        v = read_pss_kb(p)
                        s += procs[p].rss_kb if v is None else v
                pss_kb[r] = s
                comps["pss:run:" + key[r]] = s
        self._record(now, comps)
        if sys_low and "syslow" not in self.logged:
            self.logged.add("syslow")
            self.log("SYSLOW MemAvailable %.1fG < sys floor %.1fG (this uid %.1fG RSS; runs PSS: %s)" % (
                gib(avail), gib(self.sys_floor), gib(total),
                ", ".join("%s:%.1fG" % (key[r], gib(pss_kb[r])) for r in roots) or "-"))
        elif sys_on and sys_ok and "syslow" in self.logged:
            self.logged.discard("syslow")
            self.log("SYSOK MemAvailable %.1fG >= resume mark %.1fG" % (gib(avail), gib(self.sys_resume)))
        stopped = set(r for r in roots if procs[r].state in ("T", "t"))
        resumed_now, frozen_now = set(), set()
        win = a.growth_window_min

        for k, r in live.items():
            if k not in self.known:
                self.known[k] = now
                self.log("TRACK run %s%s tree=%.1fG: %s" % (
                    k, " (hands-off: legacy guard)" if r in m["handsoff"] else "",
                    gib(tree_kb[r]), " ".join(procs[r].argv)[:200]))
        for k in list(self.known):
            if k not in live:
                del self.known[k]
                self.log("END run %s" % k)
        kept = []
        for fz in self.frozen:
            if fz["key"] not in live:
                self.log("GONE frozen run %s (ended while frozen)" % fz["key"])
            elif live[fz["key"]] in m["handsoff"]:
                # an explicit legacy guard took this run over: hand it over running
                self.resume_entry(fz, "(hand-over: a legacy _mem_guard.py --run now owns it)")
                resumed_now.add(live[fz["key"]])
            else:
                kept.append(fz)
        if len(kept) != len(self.frozen):
            self.frozen = kept
            self.save_state()
        frozen_keys = set(fz["key"] for fz in self.frozen)
        for r in roots:
            if procs[r].state in ("T", "t") and key[r] not in frozen_keys and r not in m["handsoff"]:
                tag = "orphan:" + key[r]
                if tag not in self.logged:
                    self.logged.add(tag)
                    self.log("ORPHAN-STOPPED run %s is stopped but was not frozen by this guard; "
                             "not touching it (kill -CONT %d to resume)" % (key[r], r))

        if total > self.soft and not self.warned:
            self.log("WARN total=%.1fG > soft %.1fG; runs: %s; account-only %.1fG" % (
                gib(total), gib(self.soft),
                ", ".join("%s:%.1fG" % (key[r], gib(tree_kb[r])) for r in roots) or "-", gib(acct_kb)))
            self.warned = True
        elif total < 0.9 * self.soft:
            self.warned = False

        margin = a.grow_margin_gb * KB_PER_GIB
        self.over_ticks = self.over_ticks + 1 if total > self.hard else 0
        freezable = [r for r in roots if r not in m["handsoff"] and key[r] not in frozen_keys
                     and key[r] not in self.exempt and r not in stopped]
        if total > self.hard:
            if self.over_ticks >= a.confirm_ticks and (not self.frozen or total > self.last_freeze_total + margin):
                big, why, g, pressure = self.choose(roots, key, tree_kb, stopped, freezable)
                if big is not None:
                    top = max((p for p in m["members"][big] if p in procs), key=lambda p: procs[p].rss_kb)
                    n = self.signal_tree(big, key[big], signal.SIGSTOP)
                    fz = {"key": key[big], "pid": big, "start": procs[big].start, "since": now,
                          "tree_gb": round(gib(tree_kb[big]), 1), "growth_gb": round(gib(g[big]), 1),
                          "notice": stdout_log_of(big)}
                    self.frozen.append(fz)
                    frozen_now.add(big)
                    self.last_freeze_total = total
                    self.save_state()
                    self.log("FREEZE run %s (%d procs, tree %.1fG, grew %+.1fG in %g min = %.0f%% of the running "
                             "growth, top %d %s %.1fG); total=%.1fG > hard %.1fG" % (
                                 key[big], n, gib(tree_kb[big]), gib(g[big]), win,
                                 100.0 * g[big] / pressure if pressure else 100.0, top,
                                 os.path.basename(procs[top].argv[0]) if procs[top].argv else "?",
                                 gib(procs[top].rss_kb), gib(total), gib(self.hard)))
                    self.notice(fz, "FROZEN (SIGSTOP) by the MRAG memory guard (pid %d): total RSS %.1fG > hard "
                                    "%.1fG and this run's tree (%.1fG) grew %+.1fG in the last %g min. It is resumed "
                                    "automatically when the total drops below %.1fG. Guard log/state: %s "
                                    "(_mem_guard.log, _mem_guard.status.json)." % (
                                        self.pid, gib(total), gib(self.hard), gib(tree_kb[big]), gib(g[big]),
                                        win, gib(resume_line), self.dir))
                elif ("nofreeze:" + why) not in self.logged:
                    self.logged.add("nofreeze:" + why)
                    reason = {
                        "none": "no freezable run tree (hands-off / exempt / stopped / none)",
                        "not-growing": "no freezable run tree grew >= %.1fG and >= %.0f%% of the running growth "
                                       "(the excess comes from elsewhere)" % (a.grow_margin_gb,
                                                                              100 * a.min_growth_share),
                        "last-running": "the growing tree is the last running run tree while the others are "
                                        "stopped (freezing it could stop every run for good)",
                    }.get(why, why)
                    self.log("OVER total=%.1fG > hard %.1fG, nothing frozen: %s; growth in %g min: %s" % (
                        gib(total), gib(self.hard), reason, win, self.attribution(roots, key, m, stopped)))
            elif self.frozen:
                for fz in self.frozen:
                    tag = "hold:" + fz["key"]
                    if tag not in self.logged:
                        self.logged.add(tag)
                        self.log("HOLD run %s frozen; total %.1fG still > hard" % (fz["key"], gib(total)))
        elif sys_low:
            self.sys_floor_step(now, roots, key, m, procs, tree_kb, pss_kb, stopped, freezable,
                                frozen_now, avail, total)
        elif self.frozen and total < resume_line and sys_ok:
            fz = self.frozen.pop(0)
            self.logged.discard("hold:" + fz["key"])
            self.logged.discard("syshold:" + fz["key"])
            self.save_state()
            self.resume_entry(fz, "total=%.1fG < %.2f x hard%s" % (
                gib(total), self.resume,
                (", MemAvailable %.1fG >= %.1fG" % (gib(avail), gib(self.sys_resume))) if sys_on else ""))
            if fz["key"] in live:
                resumed_now.add(live[fz["key"]])
        elif self.frozen and not sys_ok:
            for fz in self.frozen:
                tag = "syshold:" + fz["key"]
                if tag not in self.logged:
                    self.logged.add(tag)
                    self.log("HOLD run %s frozen; MemAvailable %.1fG < resume mark %.1fG" % (
                        fz["key"], gib(avail), gib(self.sys_resume)))
        if total <= self.hard:
            for t in [x for x in self.logged if x.startswith("nofreeze")]:
                self.logged.discard(t)
        if not sys_low:
            for t in [x for x in self.logged if x.startswith("sysnofreeze")]:
                self.logged.discard(t)
        if total < resume_line and sys_ok and self.exempt:
            self.exempt.clear()
            self.save_state()

        # DEADLOCK-BREAK: every live run tree is stopped (our freezes, a legacy
        # guard, a manual kill -STOP) and the total cannot fall below the resume
        # line by itself: let the oldest tree this guard froze run again.
        stopped_now = (stopped - resumed_now) | frozen_now
        if (self.frozen and (total >= resume_line or not sys_ok) and len(roots) >= 2
                and all(r in stopped_now for r in roots)):
            self.deadlock_ticks += 1
            if self.deadlock_ticks >= a.confirm_ticks:
                fz = self.frozen.pop(0)
                self.exempt.add(fz["key"])
                self.logged.discard("hold:" + fz["key"])
                self.logged.discard("syshold:" + fz["key"])
                self.save_state()
                if total >= resume_line:
                    self.resume_entry(fz, "(DEADLOCK-BREAK: all %d run trees stopped and total %.1fG >= resume line "
                                          "%.1fG; exempt from freezing until the total drops below it)" % (
                                              len(roots), gib(total), gib(resume_line)))
                else:
                    self.resume_entry(fz, "(DEADLOCK-BREAK: all %d run trees stopped and MemAvailable %.1fG < resume "
                                          "mark %.1fG; exempt from freezing until MemAvailable is back above it)" % (
                                              len(roots), gib(avail), gib(self.sys_resume)))
                self.deadlock_ticks = 0
        else:
            self.deadlock_ticks = 0

        stuck = False
        if self.frozen and total >= resume_line:
            running_kb = sum(tree_kb[r] for r in roots if key[r] not in set(f["key"] for f in self.frozen))
            if total - running_kb >= resume_line:
                self.stuck_since = self.stuck_since or now
            else:
                self.stuck_since = None
            if self.stuck_since and now - self.stuck_since >= a.stuck_minutes * 60:
                stuck = True
                if "stuck" not in self.logged:
                    self.logged.add("stuck")
                    self.log("STUCK %d frozen run(s) held %.0f min and total %.1fG stays >= resume line %.1fG "
                             "even without running runs; on-stuck=%s" % (
                                 len(self.frozen), (now - self.stuck_since) / 60, gib(total), gib(resume_line),
                                 self.on_stuck))
                    for fz in self.frozen:
                        self.notice(fz, "STUCK: still frozen; total %.1fG stays >= resume line %.1fG even without "
                                        "the running runs, so waiting alone will not resume this run (on-stuck=%s). "
                                        "To let it continue: kill -TERM $(cat %s/_mem_guard.pid) (resumes "
                                        "everything the guard froze), or free memory." % (
                                            gib(total), gib(resume_line), self.on_stuck, self.dir))
                if self.on_stuck == "resume":
                    fz = self.frozen.pop(0)
                    self.exempt.add(fz["key"])
                    self.save_state()
                    self.resume_entry(fz, "(STUCK policy resume; exempt until total < resume line)")
                    self.stuck_since = None
                    self.logged.discard("stuck")
        elif self.frozen and not sys_ok:
            # held by the system floor: only other users freeing memory (or our
            # frozen trees ending) brings MemAvailable back above the resume mark
            self.stuck_since = self.stuck_since or now
            if now - self.stuck_since >= a.stuck_minutes * 60:
                stuck = True
                if "stuck" not in self.logged:
                    self.logged.add("stuck")
                    fpss = sum(pss_kb.get(live.get(f["key"]), 0) for f in self.frozen)
                    self.log("STUCK %d frozen run(s) held %.0f min and MemAvailable %.1fG stays < resume mark %.1fG "
                             "(frozen trees' PSS %.1fG); sys-on-stuck=%s" % (
                                 len(self.frozen), (now - self.stuck_since) / 60, gib(avail),
                                 gib(self.sys_resume), gib(fpss), self.sys_on_stuck))
                    for fz in self.frozen:
                        self.notice(fz, "STUCK: still frozen; the server's MemAvailable %.1fG stays below the resume "
                                        "mark %.1fG (sys-on-stuck=%s). To let it continue anyway: kill -TERM "
                                        "$(cat %s/_mem_guard.pid) (resumes everything the guard froze)." % (
                                            gib(avail), gib(self.sys_resume), self.sys_on_stuck, self.dir))
                if self.sys_on_stuck == "resume":
                    fz = self.frozen.pop(0)
                    self.exempt.add(fz["key"])
                    self.logged.discard("syshold:" + fz["key"])
                    self.save_state()
                    self.resume_entry(fz, "(STUCK sys-floor policy resume; exempt until MemAvailable >= %.1fG)" %
                                      gib(self.sys_resume))
                    self.stuck_since = None
                    self.logged.discard("stuck")
        else:
            self.stuck_since = None
            self.logged.discard("stuck")

        frozen_keys = set(f["key"] for f in self.frozen)
        self._write_json("_mem_guard.status.json", {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "pid": self.pid, "scope": a.scope,
            "total_gb": round(gib(total), 2), "soft_gb": round(gib(self.soft), 1),
            "hard_gb": round(gib(self.hard), 1), "resume_gb": round(gib(resume_line), 1),
            "account_only_gb": round(gib(acct_kb), 2), "other_gb": round(gib(other_kb), 2),
            "mem_available_gb": round(gib(avail), 1) if avail is not None else None,
            "sys_floor_gb": round(gib(self.sys_floor), 1), "sys_resume_gb": round(gib(self.sys_resume), 1),
            "sys_low": sys_low, "sys_on_stuck": self.sys_on_stuck,
            "stuck": stuck, "on_stuck": self.on_stuck, "growth_window_min": win,
            "growth_gb": {"account_only": round(gib(self.growth("acct")), 2),
                          "other": round(gib(self.growth("other")), 2)},
            "runs": [{"key": key[r], "tree_gb": round(gib(tree_kb[r]), 2), "procs": len(m["members"][r]),
                      "pss_gb": round(gib(pss_kb[r]), 2) if r in pss_kb else None,
                      "pss_growth_gb": round(gib(self.growth("pss:run:" + key[r])), 2) if r in pss_kb else None,
                      "growth_gb": round(gib(self.growth("run:" + key[r])), 2),
                      "frozen": key[r] in frozen_keys, "exempt": key[r] in self.exempt,
                      "hands_off": r in m["handsoff"], "state": procs[r].state} for r in roots],
            "frozen": self.frozen, "legacy_guards": dict((str(k), v) for k, v in m["legacy"].items()),
        })

        if roots or self.frozen:
            self.last_active = now
        elif now - self.last_active >= a.idle_exit:
            self.log("no MRAG run for %.0f s; guard exits" % (now - self.last_active))
            return False
        return True

    def shutdown(self, why):
        for fz in list(self.frozen):
            self.resume_entry(fz, "(guard exiting: %s)" % why)
        self.frozen = []
        self.save_state()
        try:
            with open(self.path("_mem_guard.pid")) as f:
                if f.read().strip() == str(self.pid):
                    os.unlink(self.path("_mem_guard.pid"))
        except OSError:
            pass

    def run(self):
        self.set_oom_adj()
        self.load_state(scan(self.uid))
        self.log("started (scope=%s soft=%.1fG hard=%.1fG resume=%.2f interval=%ss idle-exit=%ss on-stuck=%s "
                 "%s growth: window=%gmin margin=%gG share=%g%s)" % (
                     self.a.scope, gib(self.soft), gib(self.hard), self.resume, self.a.interval, self.a.idle_exit,
                     self.on_stuck,
                     ("sys-floor=%.0fG sys-resume=%.0fG sys-on-stuck=%s" % (
                         gib(self.sys_floor), gib(self.sys_resume), self.sys_on_stuck)) if self.sys_floor > 0
                     else "sys-floor=off",
                     self.a.growth_window_min, self.a.grow_margin_gb, self.a.min_growth_share,
                     (" match-run=%r" % self.a.match_run) if self.a.match_run else ""))
        why = "done"
        errors = 0
        try:
            while not self.stop:
                try:
                    if not self.tick():
                        break
                    errors = 0
                except Exception as e:
                    errors += 1
                    if errors <= 3 or errors % 100 == 0:
                        self.log("ERROR in tick (%d): %s: %s | %s" % (
                            errors, type(e).__name__, e, traceback.format_exc().replace("\n", " | ")[-800:]))
                t_end = time.time() + self.a.interval
                while not self.stop and time.time() < t_end:
                    time.sleep(min(0.5, max(0.0, t_end - time.time())))
            if self.stop:
                why = "signal"
        finally:
            self.shutdown(why)
            self.log("exit (%s)" % why)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--auto", action="store_true", help="account-wide discovery mode (marker for tools)")
    ap.add_argument("--dir", default=os.environ.get("MRAG_MEM_GUARD_DIR") or DEFAULT_DIR)
    ap.add_argument("--soft", type=float, default=0.20, help="(MRAG_MEM_SOFT wins) fraction of MemTotal")
    ap.add_argument("--hard", type=float, default=0.30, help="(MRAG_MEM_HARD wins) fraction of MemTotal")
    ap.add_argument("--soft-gb", type=float, default=0.0, help="absolute soft limit (tests)")
    ap.add_argument("--hard-gb", type=float, default=0.0, help="absolute hard limit (tests)")
    ap.add_argument("--resume", type=float, default=0.85)
    ap.add_argument("--grow-margin-gb", type=float, default=2.0,
                    help="min growth of a tree to be frozen; min total growth for another freeze")
    ap.add_argument("--growth-window-min", type=float, default=15.0,
                    help="growth = RSS now - minimum over this many minutes")
    ap.add_argument("--min-growth-share", type=float, default=0.25,
                    help="a frozen tree must carry this share of the growth of everything running")
    ap.add_argument("--confirm-ticks", type=int, default=2)
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--idle-exit", type=float, default=600.0)
    ap.add_argument("--stuck-minutes", type=float, default=30.0)
    ap.add_argument("--on-stuck", choices=("hold", "resume"), default="hold")
    ap.add_argument("--sys-floor-gb", type=float, default=None,
                    help="system floor: freeze the fastest-growing run tree while the machine's MemAvailable "
                         "is below this (else MRAG_MEM_SYS_FLOOR_GB, default 40; 0 = off)")
    ap.add_argument("--sys-resume-gb", type=float, default=None,
                    help="resume only while MemAvailable >= this (else MRAG_MEM_SYS_RESUME_GB, default 80)")
    ap.add_argument("--sys-on-stuck", choices=("hold", "resume"), default="hold",
                    help="(MRAG_MEM_SYS_ON_STUCK wins) frozen --stuck-minutes below the resume mark")
    ap.add_argument("--meminfo", default="", help="meminfo file to read MemAvailable from (tests)")
    ap.add_argument("--scope", choices=("account", "tracked"), default="account")
    ap.add_argument("--match-run", default="", help="regex on the joined cmdline instead of the mrag rule (tests)")
    ap.add_argument("--match-account", default="", help="regex instead of the ollama rule (tests)")
    ap.add_argument("--report", action="store_true", help="print one classification snapshot as JSON and exit "
                                                          "(no lock, no files, no signals)")
    ap.add_argument("--run", type=int, action="append", default=[], help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.run:
        sys.stderr.write("resource_guard.py: --run is the legacy _mem_guard.py interface\n")
        return 2
    if a.report:
        g = Guard(a)
        procs = scan(g.uid)
        m = build_model(procs, g.pid, g.is_root, g.is_account)
        tree = dict((r, sum(procs[p].rss_kb for p in m["members"][r])) for r in m["roots"])
        print(json.dumps({
            "uid": g.uid, "soft_gb": round(gib(g.soft), 1), "hard_gb": round(gib(g.hard), 1),
            "mem_available_gb": round(gib(mem_available_kb(g.meminfo) or 0), 1),
            "sys_floor_gb": round(gib(g.sys_floor), 1), "sys_resume_gb": round(gib(g.sys_resume), 1),
            "total_account_gb": round(gib(sum(p.rss_kb for p in procs.values())), 2),
            "runs": [{"pid": r, "key": m["key"][r], "hands_off": r in m["handsoff"],
                      "procs": len(m["members"][r]), "tree_gb": round(gib(tree[r]), 2),
                      "cmd": " ".join(procs[r].argv)[:160]} for r in m["roots"]],
            "account_only": dict((str(k), {"procs": len(v), "gb": round(gib(sum(procs[p].rss_kb for p in v)), 2),
                                           "cmd": " ".join(procs[k].argv)[:80]}) for k, v in m["account"].items()),
            "legacy_guards": dict((str(k), v) for k, v in m["legacy"].items()),
            "protected": sorted(m["protected"]),
        }, indent=1))
        return 0
    os.makedirs(a.dir, exist_ok=True)
    lock_fd = os.open(os.path.join(a.dir, "_mem_guard.lock"), os.O_RDWR | os.O_CREAT, 0o664)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        other = "?"
        try:
            with open(os.path.join(a.dir, "_mem_guard.pid")) as f:
                other = f.read().strip()
        except OSError:
            pass
        sys.stderr.write("memory guard already running (pid %s); exiting\n" % other)
        return 0
    g = Guard(a)
    with open(g.path("_mem_guard.pid"), "w") as f:
        f.write("%d\n" % g.pid)

    def _stop(signum, frame):
        g.stop = True
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, _stop)
    g.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
