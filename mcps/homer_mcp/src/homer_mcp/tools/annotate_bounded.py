"""Memory-bounded HOMER ``annotatePeaks.pl <peaks> <genome> -m <motifs>``.

Why: annotatePeaks.pl keeps every motif hit of every peak in Perl hashes until
it prints the table, in one single-threaded perl process (only the homer2 scan
uses -cpu). Resident memory therefore grows with scanned bp x motif count.
Measured on the original host 2026-09-16: process tree ~= 5.2 GiB + 0.95 GiB per Mb per
1000 motifs (MYC-DDK min-max, 165 Mb x 2503 motifs: ~350 GiB, 6.3 h).

What this module does (called from homer_mcp.tools.pipeline.annotate_peaks):

* ``plan_annotation`` estimates the memory of a call.
* Calls whose estimate fits ``MRAG_HOMER_ANNOT_MAX_GB`` keep the old single
  annotatePeaks.pl call unchanged; they only wait for a shared memory slot.
* Larger calls are split: the peaks are first normalised with HOMER's own
  ``bed2pos.pl -check -unique`` + ``cleanUpPeakFile.pl`` (exactly what
  annotatePeaks.pl does internally, so PeakIDs are identical), packed into
  chunks by scanned bp, annotated in parallel inside the shared slot budget,
  checked (exit code, HOMER's "Done annotating peaks file", every PeakID back,
  identical headers) and merged into the table one call would have written:
  same header (first column rewritten to the unchunked command), same rows,
  ordered by Peak Score descending as HOMER sorts them (numeric unless a score
  is non-numeric; ties keep input order, HOMER itself leaves ties in random
  hash order). The table is written to ``<out>.part`` and renamed only when
  complete; on any failure nothing is written and all chunk processes are
  killed.

Slots are lock files (``fcntl.flock``) in ``MRAG_HOMER_SLOT_DIR``, so the budget
holds across threads of one run and across concurrent MRAG runs; the kernel
drops a lock when its process dies.

Environment knobs (all optional):
  MRAG_HOMER_CHUNKING=0            kill switch: plain single call, no slots
  MRAG_HOMER_ANNOT_MAX_GB=40       memory budget of one annotatePeaks process tree
  MRAG_HOMER_TOTAL_GB=160          budget of all concurrent annotatePeaks (slots = TOTAL // MAX)
  MRAG_HOMER_SLOT_DIR=<dir>        shared slot lock files
  MRAG_HOMER_SLOT_POLL_S=5         slot polling interval
  MRAG_HOMER_CHUNK_BP=<bp>         force chunk size (tests)
  MRAG_HOMER_MAX_PARALLEL=<n>      cap on concurrent chunks of one call
  MRAG_HOMER_GIB_BASE=5.2          estimate: fixed GiB per process tree
  MRAG_HOMER_GIB_PER_MB_PER_1K=0.95  estimate: GiB per scanned Mb per 1000 motifs
"""
from __future__ import annotations

import fcntl
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SENTINEL = b"Done annotating peaks file"
DEFAULT_SLOT_DIR = "/BIO1/MRAG/Agent/run/homer_slots"   # not under webui/runs: the webui lists every dir there

# extra annotatePeaks.pl arguments that act per peak, so chunking stays exact
_CHUNK_SAFE_NOARG = {"-nmotifs", "-mscore", "-norevopp"}
_CHUNK_SAFE_ONEARG = {"-gtf"}


class NotChunkable(RuntimeError):
    """The input cannot be split exactly; the caller runs the single call."""


class SlotCancelled(RuntimeError):
    pass


def _log(msg: str) -> None:
    # stderr: stdout of an MCP stdio server is the protocol channel
    print(f"[HOMER annotate] {msg}", file=sys.stderr, flush=True)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    try:
        return float(v) if v else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    try:
        return int(float(v)) if v else default
    except ValueError:
        return default


def ncpus() -> int:
    """CPUs this process may run on (taskset / MRAG_CPU_LIST aware)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def chunking_enabled() -> bool:
    return os.environ.get("MRAG_HOMER_CHUNKING", "1").strip().lower() not in ("0", "false", "no", "off")


def max_gb() -> float:
    return max(1.0, _env_float("MRAG_HOMER_ANNOT_MAX_GB", 40.0))


def total_gb() -> float:
    return max(max_gb(), _env_float("MRAG_HOMER_TOTAL_GB", 160.0))


def n_slots() -> int:
    return max(1, int(total_gb() // max_gb()))


def est_gib(bp: float, n_motifs: int) -> float:
    base = _env_float("MRAG_HOMER_GIB_BASE", 5.2)
    k = _env_float("MRAG_HOMER_GIB_PER_MB_PER_1K", 0.95)
    return base + k * (bp / 1e6) * (max(1, n_motifs) / 1000.0)


def count_motifs(path) -> int:
    n = 0
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                n += 1
    return n


_INT = re.compile(rb"^\s*-?\d+\s*$")


def quick_scan_bp(path) -> Tuple[int, int]:
    """(regions, scanned bp) of a BED or HOMER peak file, without HOMER."""
    n = bp = 0
    with open(path, "rb") as f:
        for raw in f:
            if raw.startswith((b"#", b"track", b"browser")):
                continue
            c = raw.rstrip(b"\r\n").split(b"\t")
            if len(c) >= 3 and _INT.match(c[1]) and _INT.match(c[2]):       # BED chr start end
                w = int(c[2]) - int(c[1])
            elif len(c) >= 4 and _INT.match(c[2]) and _INT.match(c[3]):     # HOMER id chr start end
                w = int(c[3]) - int(c[2]) + 1
            else:
                continue
            n += 1
            bp += max(1, w)
    return n, bp


def chunk_safe_args(extra: Optional[Sequence[str]]) -> bool:
    extra = list(extra or [])
    i = 0
    while i < len(extra):
        a = extra[i]
        if a in _CHUNK_SAFE_NOARG:
            i += 1
        elif a in _CHUNK_SAFE_ONEARG and i + 1 < len(extra):
            i += 2
        elif a == "-size" and i + 1 < len(extra) and extra[i + 1] == "given":
            i += 2
        else:
            return False
    return True


def slots_for(est: float) -> int:
    """Shared slots a process tree of ``est`` GiB must hold."""
    return max(1, min(n_slots(), int(math.ceil(float(est) / max_gb()))))


def plan_annotation(bed, motif, extra_args=None) -> Dict[str, Any]:
    """Decide how annotate_peaks runs. Never raises.

    mode "legacy"  : chunking disabled -> old call, no slot
    mode "single"  : old call inside ``slots`` shared slots
    mode "chunked" : split run (annotate_chunked)
    """
    if not chunking_enabled():
        return {"mode": "legacy", "reason": "MRAG_HOMER_CHUNKING=0"}
    mx, ns = max_gb(), n_slots()
    p: Dict[str, Any] = {"max_gb": mx, "total_gb": total_gb(), "n_slots": ns}
    try:
        nm = count_motifs(motif)
        n, bp = quick_scan_bp(bed)
    except OSError as e:
        p.update(mode="single", slots=1, reason=f"estimate failed: {e}")
        return p
    est = est_gib(bp, nm)
    p.update(n_motifs=nm, n_regions=n, bp=bp, est_gib=round(est, 1))
    force_bp = _env_int("MRAG_HOMER_CHUNK_BP", 0)
    need = est > mx or (force_bp > 0 and bp > force_bp)
    p["slots"] = slots_for(est)
    if need and nm > 0 and n > 1 and chunk_safe_args(extra_args):
        p["mode"] = "chunked"
    else:
        p["mode"] = "single"
        if need:
            p["reason"] = "over budget but not chunkable (extra_args or single region)"
    return p


# ---------------------------------------------------------------------------
# shared slots
# ---------------------------------------------------------------------------
_local_cv = threading.Condition()
_local_used = [0]


def _slot_paths() -> Optional[List[str]]:
    d = os.environ.get("MRAG_HOMER_SLOT_DIR", DEFAULT_SLOT_DIR)
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return None
    return [os.path.join(d, f"slot_{i:02d}") for i in range(n_slots())]


def acquire_slots(k: int, label: str = "", cancel: Optional[threading.Event] = None):
    """Block until ``k`` slots are held; returns a handle for release_slots."""
    n = n_slots()
    k = max(1, min(int(k), n))
    poll = max(0.2, _env_float("MRAG_HOMER_SLOT_POLL_S", 5.0))
    paths = _slot_paths()
    t0 = time.monotonic()
    announced = False
    while True:
        if paths is not None:
            got, opened = [], 0
            for p in paths:
                try:
                    fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o664)
                except OSError:
                    continue
                opened += 1
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    os.close(fd)
                    continue
                got.append(fd)
                if len(got) == k:
                    break
            if len(got) == k:
                info = f"pid={os.getpid()} tid={threading.get_ident()} since={time.strftime('%F %T')} {label}\n"
                for fd in got:
                    try:
                        os.ftruncate(fd, 0)
                        os.pwrite(fd, info.encode(), 0)
                    except OSError:
                        pass
                return ("flock", got)
            for fd in got:
                _release_fd(fd)
            if opened == 0:          # slot dir unusable -> per-process budget
                paths = None
                continue
        else:
            with _local_cv:
                if _local_used[0] + k <= n:
                    _local_used[0] += k
                    return ("local", k)
        if cancel is not None and cancel.is_set():
            raise SlotCancelled(label)
        if not announced and time.monotonic() - t0 > 30:
            _log(f"waiting for {k} memory slot(s) of {n}: {label}")
            announced = True
        time.sleep(poll)


def _release_fd(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


def release_slots(handle) -> None:
    if not handle:
        return
    kind, val = handle
    if kind == "flock":
        for fd in val:
            _release_fd(fd)
    else:
        with _local_cv:
            _local_used[0] = max(0, _local_used[0] - val)
            _local_cv.notify_all()


@contextmanager
def slots(k: int, label: str = ""):
    h = acquire_slots(k, label)
    try:
        yield h
    finally:
        release_slots(h)


# ---------------------------------------------------------------------------
# process helpers
# ---------------------------------------------------------------------------
def _children_map() -> Dict[int, List[int]]:
    kids: Dict[int, List[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat", "rb") as f:
                ppid = int(f.read().rsplit(b")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        kids.setdefault(ppid, []).append(int(d))
    return kids


def _tree(pid: int) -> List[int]:
    kids = _children_map()
    out, stack = [], [pid]
    while stack:
        p = stack.pop()
        out.append(p)
        stack.extend(kids.get(p, []))
    return out


def kill_tree(pid: int) -> None:
    """SIGSTOP the whole tree (no more forks, nobody reparents), then SIGKILL."""
    seen: List[int] = []
    for _ in range(2):
        for p in _tree(pid):
            if p not in seen:
                seen.append(p)
            try:
                os.kill(p, signal.SIGSTOP)
            except OSError:
                pass
    for p in seen:
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass


def _vm_hwm_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


# ---------------------------------------------------------------------------
# chunked annotation
# ---------------------------------------------------------------------------
_NUMERIC_SCORE = re.compile(rb"^[\-\d\.\+e]+$")          # annotatePeaks.pl sort test
_PERL_NUM = re.compile(rb"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")


def _perl_num(s: bytes) -> float:
    m = _PERL_NUM.match(s)
    if not m:
        return 0.0
    try:
        return float(m.group(1))
    except ValueError:
        return 0.0


def _normalise(bed: Path, work: Path, perl: str, homer_bin: Path, env) -> List[Tuple[bytes, bytes, int]]:
    """HOMER's own peak clean-up; returns [(pos line, PeakID, scanned bp)] in file order."""
    tmp, pos = work / "norm.tmp", work / "norm.pos"
    for cmd, dst in (([perl, str(homer_bin / "bed2pos.pl"), str(bed), "-check", "-unique"], tmp),
                     ([perl, str(homer_bin / "cleanUpPeakFile.pl"), str(tmp)], pos)):
        with open(dst, "wb") as fo:
            r = subprocess.run(cmd, stdout=fo, stderr=subprocess.PIPE, cwd=str(work), env=env)
        if r.returncode != 0:
            raise NotChunkable(f"{Path(cmd[1]).name} rc={r.returncode}: {r.stderr[-300:]!r}")
    rows: List[Tuple[bytes, bytes, int]] = []
    ids = set()
    with open(pos, "rb") as f:
        for raw in f:
            line = raw.rstrip(b"\n").replace(b"\r", b"")
            if line.startswith(b"#"):
                continue
            c = line.split(b"\t")
            if len(c) < 4 or not c[0]:
                raise NotChunkable(f"unexpected normalised line: {line[:80]!r}")
            if re.match(rb"^[^\d\-\.]", c[2]):      # annotatePeaks.pl skips these too
                continue
            try:
                w = int(float(c[3])) - int(float(c[2])) + 1
            except ValueError:
                raise NotChunkable(f"non-numeric coordinates: {line[:80]!r}")
            if c[0] in ids:
                raise NotChunkable(f"duplicate PeakID after clean-up: {c[0][:60]!r}")
            ids.add(c[0])
            rows.append((line + b"\n", c[0], max(1, w)))
    try:
        tmp.unlink()
    except OSError:
        pass
    return rows


def _pack(rows, chunk_bp: int) -> List[List[int]]:
    """First-fit decreasing by scanned bp; returns row indices per chunk (input order)."""
    order = sorted(range(len(rows)), key=lambda i: (-rows[i][2], i))
    bins: List[List[int]] = []
    room: List[int] = []
    for i in order:
        w = rows[i][2]
        for b in range(len(bins)):
            if room[b] >= w:
                bins[b].append(i)
                room[b] -= w
                break
        else:
            bins.append([i])
            room.append(chunk_bp - w)
    for b in bins:
        b.sort()
    return bins


def _index_chunk_output(path: Path) -> Tuple[bytes, List[Tuple[bytes, bytes, int, int]]]:
    """(header line, [(PeakID, Peak Score, offset, length)])."""
    entries = []
    with open(path, "rb") as f:
        header = f.readline()
        off = len(header)
        for line in f:
            ln = len(line)
            if line.strip():
                parts = line.split(b"\t", 6)
                score = parts[5] if len(parts) > 5 else b""
                entries.append((parts[0], score, off, ln))
            off += ln
    return header, entries


def _clean_stale_workdirs(out: Path) -> None:
    """Remove chunk dirs of this output left by a killed process (pid no longer alive)."""
    prefix = f".{out.name}.chunks."
    try:
        entries = list(out.parent.iterdir())
    except OSError:
        return
    for d in entries:
        if not d.name.startswith(prefix) or not d.is_dir():
            continue
        try:
            pid = int(d.name[len(prefix):].split(".", 1)[0])
        except ValueError:
            continue
        if pid == os.getpid() or os.path.exists(f"/proc/{pid}"):
            continue
        shutil.rmtree(d, ignore_errors=True)
        _log(f"removed stale chunk dir {d}")


def annotate_chunked(*, perl: str, homer_bin: Path, script: Path, bed: Path, genome: str,
                     motif: Path, out: Path, header_args: Sequence[str],
                     extra_args: Optional[Sequence[str]], timeout_s: Optional[float],
                     env: Dict[str, str], plan: Dict[str, Any]) -> Dict[str, Any]:
    """Split, run, verify, merge. Returns a result dict (status ok / error / timeout).

    Raises NotChunkable before any HOMER work when the input cannot be split
    exactly (caller then runs the single call). ``timeout_s`` applies to each
    chunk's running time (slot waits and SIGSTOP pauses are not counted).
    """
    t0 = time.time()
    extra = list(extra_args or [])
    work = out.parent / f".{out.name}.chunks.{os.getpid()}.{threading.get_ident()}"
    part = out.parent / (out.name + ".part")
    _clean_stale_workdirs(out)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    ok = False
    try:
        rows = _normalise(bed, work, perl, homer_bin, env)
        if len(rows) < 2:
            raise NotChunkable(f"{len(rows)} peak(s) after clean-up")
        nm = int(plan.get("n_motifs") or count_motifs(motif))
        mx = max_gb()
        total_bp = sum(r[2] for r in rows)
        chunk_bp = _env_int("MRAG_HOMER_CHUNK_BP", 0)
        if chunk_bp <= 0:
            per_mb = est_gib(1e6, nm) - est_gib(0, nm)
            usable = mx - _env_float("MRAG_HOMER_GIB_BASE", 5.2) - 0.3
            chunk_bp = int(max(0.5, usable) / max(per_mb, 1e-6) * 1e6)
        chunk_bp = max(10_000, chunk_bp)
        bins = _pack(rows, chunk_bp)
        if len(bins) < 2:
            raise NotChunkable("fits in one chunk")
        ns = n_slots()
        cap = _env_int("MRAG_HOMER_MAX_PARALLEL", 0)
        ncpu = ncpus()
        W = max(1, min(len(bins), ns, ncpu, cap if cap > 0 else len(bins)))
        cpu = max(1, ncpu // W)
        chunks = []
        for i, b in enumerate(bins):
            d = work / f"chunk{i:04d}"
            d.mkdir()
            pos = d / "in.pos"
            with open(pos, "wb") as fo:
                for j in b:
                    fo.write(rows[j][0])
            bp = sum(rows[j][2] for j in b)
            est = est_gib(bp, nm)
            chunks.append({"i": i, "dir": d, "pos": pos, "rows": b, "bp": bp,
                           "est_gib": round(est, 1),
                           "slots": slots_for(est)})
        _log(f"{bed.name}: {len(rows)} peaks, {total_bp/1e6:.1f} Mb x {nm} motifs "
             f"(est {est_gib(total_bp, nm):.0f} GiB) -> {len(chunks)} chunks <= {chunk_bp/1e6:.2f} Mb, "
             f"{W} parallel of {ns} shared slots x {mx:.0f} GiB, -cpu {cpu} on {ncpu} CPUs")

        failed = threading.Event()
        lock = threading.Lock()
        procs: Dict[int, subprocess.Popen] = {}
        results: Dict[int, Dict[str, Any]] = {}

        def run_one(ch):
            i = ch["i"]
            res: Dict[str, Any] = {"i": i, "peaks": len(ch["rows"]), "bp": ch["bp"],
                                   "est_gib": ch["est_gib"], "status": "skipped"}
            results[i] = res
            if failed.is_set():
                return res
            try:
                held = acquire_slots(ch["slots"], label=f"{out.name} chunk {i+1}/{len(chunks)}", cancel=failed)
            except SlotCancelled:
                return res
            outp, errp = ch["dir"] / "out.tsv", ch["dir"] / "err.txt"
            cmd = [perl, str(script), str(ch["pos"]), genome, "-m", str(motif), "-cpu", str(cpu)] + extra
            t1 = time.monotonic()
            hwm = 0
            status = None
            try:
                if failed.is_set():
                    return res
                with open(outp, "wb") as fo, open(errp, "wb") as fe:
                    proc = subprocess.Popen(cmd, stdout=fo, stderr=fe, cwd=str(ch["dir"]), env=env)
                with lock:
                    procs[i] = proc
                active, last, last_hwm = 0.0, time.monotonic(), 0.0
                while True:
                    try:
                        proc.wait(timeout=1.0)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    now = time.monotonic()
                    gap, last = now - last, now
                    if gap < 10.0:          # a longer gap: this tree was SIGSTOPped
                        active += gap
                    if now - last_hwm > 30:
                        hwm, last_hwm = max(hwm, _vm_hwm_kb(proc.pid)), now
                    if failed.is_set():
                        status = "cancelled"
                    elif timeout_s and active > float(timeout_s):
                        status = "timeout"
                    if status:
                        kill_tree(proc.pid)
                        proc.wait()
                        break
                with lock:
                    procs.pop(i, None)
            finally:
                release_slots(held)
            res.update(rc=proc.returncode, elapsed_s=round(time.monotonic() - t1, 1),
                       perl_hwm_gib=round(hwm / 1048576, 2))
            try:
                err_tail = errp.read_bytes()[-3000:]
            except OSError:
                err_tail = b""
            if status:
                res.update(status=status, stderr_tail=err_tail.decode(errors="replace"))
                if status == "timeout":
                    failed.set()
                return res
            problems = []
            if proc.returncode != 0:
                problems.append(f"exit code {proc.returncode}")
            if SENTINEL not in err_tail:
                problems.append("no 'Done annotating peaks file' in stderr")
            header, entries = b"", []
            if not problems:
                header, entries = _index_chunk_output(outp)
                want = {rows[j][1] for j in ch["rows"]}
                got = [e[0] for e in entries]
                if len(got) != len(want) or set(got) != want:
                    problems.append(f"{len(got)} rows for {len(want)} peaks / PeakIDs differ")
                if not header.startswith(b"PeakID"):
                    problems.append("missing header")
            if problems:
                res.update(status="error", problems=problems, stderr_tail=err_tail.decode(errors="replace"))
                failed.set()
                return res
            res.update(status="ok", header=header, entries=entries, out=outp)
            return res

        with ThreadPoolExecutor(max_workers=W) as ex:
            futs = [ex.submit(run_one, ch) for ch in chunks]
            for f in futs:
                try:
                    f.result()
                except BaseException as e:           # bug in a worker, or interrupt
                    failed.set()
                    _log(f"chunk worker crashed: {type(e).__name__}: {e}")
        with lock:
            for p in list(procs.values()):
                if p.poll() is None:
                    kill_tree(p.pid)

        summary = [{k: v for k, v in r.items() if k not in ("header", "entries", "out")}
                   for _, r in sorted(results.items())]
        bad = [r for r in summary if r.get("status") != "ok"]
        if bad or len(summary) != len(chunks):
            status = "timeout" if any(r.get("status") == "timeout" for r in bad) else "error"
            _log(f"{bed.name}: chunked annotation FAILED ({len(bad)} of {len(chunks)} chunks not ok); "
                 f"no output written")
            return {"status": status, "elapsed_s": round(time.time() - t0, 1), "output": str(out),
                    "timeout_s": timeout_s, "chunking": {"chunks": len(chunks), "parallel": W,
                                                           "cpu_per_chunk": cpu, "chunk_bp": chunk_bp,
                                                           "failed": [r for r in bad if r.get("status") != "skipped"],
                                                           "per_chunk": summary}}

        # ---- merge -------------------------------------------------------------
        oks = [results[c["i"]] for c in chunks]
        rest0 = oks[0]["header"].split(b"\t", 1)[1] if b"\t" in oks[0]["header"] else b"\n"
        for r in oks[1:]:
            rest = r["header"].split(b"\t", 1)[1] if b"\t" in r["header"] else b"\n"
            if rest != rest0:
                raise RuntimeError(f"chunk {r['i']} header differs from chunk 0")
        order = {rows[j][1]: j for j in range(len(rows))}
        merged = []
        numeric = True
        for r in oks:
            ci = r["i"]
            for pid_, score, off, ln in r["entries"]:
                if numeric and (score == b"." or not _NUMERIC_SCORE.match(score)):
                    numeric = False
                merged.append((order[pid_], score, ci, off, ln))
        merged.sort(key=lambda e: e[0])
        if numeric:
            merged.sort(key=lambda e: _perl_num(e[1]), reverse=True)    # stable: ties keep input order
        else:
            merged.sort(key=lambda e: e[1], reverse=True)
        header0 = b"PeakID (cmd=annotatePeaks.pl " + b" ".join(os.fsencode(str(a)) for a in header_args) + b")"
        n_written = 0
        fhs = {r["i"]: open(r["out"], "rb") for r in oks}
        try:
            with open(part, "wb", buffering=16 * 1024 * 1024) as fo:
                fo.write(header0 + b"\t" + rest0)
                for _, _, ci, off, ln in merged:
                    fh = fhs[ci]
                    if fh.tell() != off:
                        fh.seek(off)
                    buf = fh.read(ln)
                    if len(buf) != ln:
                        raise RuntimeError(f"short read in chunk {ci}")
                    fo.write(buf if buf.endswith(b"\n") else buf + b"\n")
                    n_written += 1
        finally:
            for fh in fhs.values():
                fh.close()
        if n_written != len(rows):
            raise RuntimeError(f"merged {n_written} rows for {len(rows)} peaks")
        os.replace(part, out)
        ok = True
        elapsed = round(time.time() - t0, 1)
        _log(f"{bed.name}: merged {n_written} rows from {len(chunks)} chunks in {elapsed:.0f} s -> {out}")
        return {"status": "ok", "returncode": 0, "elapsed_s": elapsed, "output": str(out),
                "n_rows": n_written, "stderr_tail": "",
                "chunking": {"chunks": len(chunks), "parallel": W, "cpu_per_chunk": cpu,
                             "chunk_bp": chunk_bp, "sort": "numeric" if numeric else "string",
                             "per_chunk": summary}}
    finally:
        if not ok:
            try:
                part.unlink()
            except OSError:
                pass
        shutil.rmtree(work, ignore_errors=True)
