"""Single source of the NCBI E-utilities credentials for every MRAG caller.

Added 2026-09-20. Before this, every E-utilities caller
(GeneGPT/GeneRIF, the report's PubMed novelty check, the paperqa fetch/monthly
scripts, the SRA/GEO metadata lookups) identified itself anonymously and had to
stay under NCBI's 3 req/s ceiling. With a key the ceiling is 10 req/s, so the
whole point of this module is that the key, the contact address and the tool
name are configured in ONE place and every caller picks up the same limit.

    import sys
    sys.path.insert(0, "/BIO1/MRAG/Agent/libs")
    import ncbi_creds

    url = ncbi_creds.add_params(f"{EUTILS}/esearch.fcgi?db=gene&term=TP53")
    time.sleep(ncbi_creds.min_interval())          # 0.12 s keyed, 0.4 s not
    ...
    except Exception as e:
        log(ncbi_creds.redact(str(e)))             # never let the key reach a log

THE KEY IS A SECRET.  It lives only in the env file (mode 600) or in an already
exported environment variable, and it only ever appears inside a URL that is
handed straight to urllib / Biopython.  It must never be written to a log, a
status or history JSON, a manifest, a report, a run directory or a git-tracked
file.  Anything that may carry it into text -- an error message, a traceback,
a debug dump of a URL -- goes through redact() first.

Resolution order for each of the three values:
  1. the process environment (NCBI_API_KEY / NCBI_EMAIL / NCBI_TOOL), so a
     caller that already exports them does not need the file;
  2. the env file, default ~/.config/mrag/ncbi.env, overridable with
     MRAG_NCBI_ENV (used by the tests to prove the no-key fallback);
  3. nothing -- every function degrades to exactly the pre-2026-09-20
     behaviour: no api_key parameter, no tool parameter, unauthenticated
     3 req/s spacing, i.e. byte-identical URLs to before the change.

The default tool name is only attached when this machine is configured at all
(an api_key or a contact email exists) or when the caller set NCBI_TOOL itself:
a machine with NO credentials must send exactly the URL it sent before.

Nothing here raises: a missing, unreadable or malformed file is simply "no
credentials", because no MRAG run may break on a machine that has no key.
"""
from __future__ import annotations

import os
import re
import threading
import time

__all__ = [
    "creds", "has_api_key", "min_interval", "params", "add_params",
    "redact", "apply_to_entrez", "Throttle", "NcbiError", "env_path",
    "describe", "DEFAULT_TOOL", "SharedThrottle", "shared_throttle_on", "throttle_file",
]

DEFAULT_ENV_FILE = "~/.config/mrag/ncbi.env"
# Sent as &tool= so NCBI can identify the client -- but only once this machine
# has credentials of its own (see creds()).
DEFAULT_TOOL = "MRAG-Agent"

# Client-side spacing. NCBI allows 3 req/s without a key and 10 req/s with one;
# both values keep a safety margin (2.5 /s and ~8.3 /s) because the cap is
# enforced per key across every process on this shared server, and a 429 costs
# far more than the few ms of margin.
INTERVAL_NO_KEY = 0.40
INTERVAL_KEYED = 0.12

_VAR_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_KEY_RE = re.compile(r"^[A-Za-z0-9]{8,64}$")

_lock = threading.Lock()
_cache: dict | None = None


class NcbiError(RuntimeError):
    """An NCBI request failed. The message has already been redacted.

    followup 2026-09-28: genegpt.call_api also sets .transient (was the last
    error one worth retrying), .attempts and .error_class on the instance."""

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code
        self.transient = None
        self.attempts = None
        self.error_class = ""


def env_path() -> str:
    return os.path.expanduser(os.environ.get("MRAG_NCBI_ENV") or DEFAULT_ENV_FILE)


def _read_env_file(path: str) -> dict:
    """Parse a KEY=VALUE env file. Never raises; unreadable file -> {}."""
    out: dict = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                m = _VAR_RE.match(line)
                if not m:
                    continue
                val = m.group(2)
                if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                    val = val[1:-1]
                out[m.group(1)] = val
    except Exception:
        return {}
    return out


def creds(refresh: bool = False) -> dict:
    """{'api_key','email','tool','source'} -- empty strings when absent.

    Cached for the life of the process (refresh=True re-reads, which the tests
    use to flip between "key present" and "env file hidden")."""
    global _cache
    with _lock:
        if _cache is not None and not refresh:
            return dict(_cache)
        path = env_path()
        filevals = _read_env_file(path) if os.path.isfile(path) else {}

        def pick(name):
            v = os.environ.get(name)
            if v and v.strip():
                return v.strip(), "env"
            v = filevals.get(name)
            if v and v.strip():
                return v.strip(), "file"
            return "", ""

        api_key, ks = pick("NCBI_API_KEY")
        email, es = pick("NCBI_EMAIL")
        tool, ts = pick("NCBI_TOOL")
        if api_key and not _KEY_RE.match(api_key):
            # a pasted quote / trailing comment would make every request fail
            # with a 400 the caller cannot read; treat it as no key at all.
            api_key, ks = "", "rejected(malformed)"
        # LOW-2 (2026-09-20 review): the default tool name used to be filled
        # in UNCONDITIONALLY, so a machine with no key and no email started
        # sending "&tool=MRAG-Agent" on every E-utilities URL -- a new
        # parameter, not the pre-2026-09-20 behaviour this module promises.
        # It is now added only when the machine is configured (key or email)
        # or when the caller set NCBI_TOOL explicitly.
        if not tool and (api_key or email):
            tool, ts = DEFAULT_TOOL, "default"
        _cache = {"api_key": api_key, "email": email, "tool": tool,
                  "source": ks or "none", "email_source": es or "none",
                  "tool_source": ts or "none",
                  "env_file": path, "env_file_exists": bool(filevals)}
        return dict(_cache)


def describe() -> str:
    """One safe line for a log: says whether a key is in use, never its value."""
    c = creds()
    return ("ncbi: api_key=%s (%s) email=%s tool=%s interval=%.2fs"
            % ("yes" if c["api_key"] else "no", c["source"],
               c["email"] or "-", c["tool"] or "-", min_interval()))


def has_api_key() -> bool:
    return bool(creds().get("api_key"))


def min_interval() -> float:
    """Seconds to leave between two E-utilities requests from one process."""
    return INTERVAL_KEYED if has_api_key() else INTERVAL_NO_KEY


def params() -> dict:
    """Query parameters to add to every E-utilities URL (may be empty)."""
    c = creds()
    out = {}
    if c["api_key"]:
        out["api_key"] = c["api_key"]
    if c["email"]:
        out["email"] = c["email"]
    if c["tool"]:
        out["tool"] = c["tool"]
    return out


def add_params(url: str) -> str:
    """Append api_key/tool/email to an E-utilities URL, without duplicating
    parameters the caller already set.

    Only eutils.ncbi.nlm.nih.gov is touched: a URL for Europe PMC / ENA (helpers
    are shared with them) or for the BLAST URL API -- which is a different
    service that does not take an api_key and whose CGI should not be handed
    unknown parameters -- is returned unchanged."""
    u = str(url or "")
    if "eutils.ncbi.nlm.nih.gov" not in u:
        return u
    extra = params()
    if not extra:
        return u
    import urllib.parse as _up
    head, sep, frag = u.partition("#")
    base, qsep, query = head.partition("?")
    have = {k for k, _ in _up.parse_qsl(query, keep_blank_values=True)}
    add = [(k, v) for k, v in extra.items() if k not in have]
    if not add:
        return u
    q = (query + "&" if query else "") + _up.urlencode(add)
    return base + "?" + q + (sep + frag if sep else "")


def redact(text) -> str:
    """Replace the api_key anywhere in a string with a placeholder.

    Belt and braces: it blanks the literal key AND any api_key=... parameter,
    so a URL from a machine with a different key (or a key we never loaded) is
    scrubbed too."""
    s = str(text)
    key = creds().get("api_key")
    if key:
        s = s.replace(key, "<NCBI_API_KEY>")
    s = re.sub(r"(?i)(api[_-]?key[=:\"'\s]+)[A-Za-z0-9]{8,}", r"\1<NCBI_API_KEY>", s)
    return s


def apply_to_entrez(entrez_module) -> bool:
    """Configure Bio.Entrez. Returns True when a key was applied.

    Biopython drops its own spacing from 0.37 s to 0.10 s as soon as an
    api_key parameter is present (Bio/Entrez/__init__.py::_has_api_key), so
    setting the attribute is what actually buys the 10 req/s."""
    c = creds()
    try:
        if c["email"]:
            entrez_module.email = c["email"]
        if c["tool"]:
            entrez_module.tool = c["tool"]
        if c["api_key"]:
            entrez_module.api_key = c["api_key"]
            return True
        return False
    except Exception:
        return False


class Throttle:
    """Process-wide spacing between E-utilities requests.

    Shared by every caller in this process so that two code paths (say the
    GeneRIF fetch and the PubMed novelty check) cannot together exceed the
    per-key limit. The interval is read at each wait(), so hiding the env file
    and calling creds(refresh=True) immediately drops back to 3 req/s."""

    def __init__(self, interval=None):
        self._interval = interval
        self._last = 0.0
        self._lk = threading.Lock()

    def wait(self):
        iv = self._interval if self._interval is not None else min_interval()
        with self._lk:
            delta = time.time() - self._last
            if delta < iv:
                time.sleep(iv - delta)
            self._last = time.time()


# ---------------------------------------------------------------------------
# followup 2026-09-28 (GeneRIF fetch fix): ONE limiter for every MRAG process
# ---------------------------------------------------------------------------
# NCBI enforces its ceiling (10 req/s with the api_key, 3 without) per KEY
# (per IP without one) across EVERY process, but Throttle is per process: two
# MRAG processes calling E-utilities at the same time (a run's GeneRIF fetch
# and another run's PubMed checks, a workflow script, the paperqa monthly
# update) each kept their own 0.12 s spacing and together exceeded the limit.
# In the 2026-09-28 SCLC E2E the GeneRIF fetch then lost every line of 8 genes
# (581 lines, TFPI2 alone 125) without a word.  SharedThrottle keeps the
# per-process spacing AND reserves every request's time slot in one small
# state file shared by all processes of this user (flock; /dev/shm by
# default), so all of them together keep the one interval.
#   MRAG_NCBI_SHARED_THROTTLE=0   the per-process Throttle only (pre-fix)
#   MRAG_NCBI_THROTTLE_FILE=PATH  another state file (tests)
# Any problem with the file (no fcntl, unwritable, corrupt) falls back to the
# per-process spacing: this never raises and never blocks longer than the
# interval asks.
THROTTLE_FILE_DIRS = ("/dev/shm",)


def shared_throttle_on() -> bool:
    v = str(os.environ.get("MRAG_NCBI_SHARED_THROTTLE") or "").strip().lower()
    return v not in ("0", "off", "false", "no", "none")


def throttle_file() -> str:
    """The shared slot file: MRAG_NCBI_THROTTLE_FILE, else /dev/shm/mrag_ncbi_throttle.<uid>
    (the temp dir when /dev/shm is not writable)."""
    p = os.environ.get("MRAG_NCBI_THROTTLE_FILE")
    if p and p.strip():
        return os.path.expanduser(p.strip())
    try:
        uid = os.getuid()
    except Exception:
        uid = 0
    for d in THROTTLE_FILE_DIRS:
        if os.path.isdir(d) and os.access(d, os.W_OK):
            return os.path.join(d, "mrag_ncbi_throttle.%d" % uid)
    import tempfile
    return os.path.join(tempfile.gettempdir(), "mrag_ncbi_throttle.%d" % uid)


class SharedThrottle(Throttle):
    """Per-process AND cross-process spacing between E-utilities requests.

    wait() reserves the next free slot -- max(now, last reserved slot + interval,
    this process's own last slot + interval) -- in the shared file under an
    exclusive flock, then sleeps until that slot OUTSIDE the lock, so threads and
    processes queue fairly instead of spinning.  The interval is read at each
    wait() (min_interval(): 0.12 s with the api_key, 0.4 s without)."""

    MAX_AHEAD_S = 60.0     # a stored slot further ahead than this is a clock jump / junk

    def __init__(self, interval=None, path=None):
        super().__init__(interval)
        self._path = path
        self.stats = {"waits": 0, "shared": 0, "fallback": 0, "slept_s": 0.0, "error": ""}

    def _reserve(self, iv, floor):
        import fcntl
        path = self._path or throttle_file()
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            raw = os.pread(fd, 64, 0)
            try:
                last = float(raw.decode("ascii", "ignore").strip() or 0.0)
            except ValueError:
                last = 0.0
            now = time.time()
            if last > now + self.MAX_AHEAD_S:
                last = 0.0
            slot = max(now, last + iv, floor)
            os.pwrite(fd, ("%.6f" % slot).ljust(32).encode("ascii"), 0)
            return slot
        finally:
            os.close(fd)          # closing the descriptor releases the lock

    def wait(self):
        if not shared_throttle_on():
            return Throttle.wait(self)
        iv = self._interval if self._interval is not None else min_interval()
        with self._lk:
            self.stats["waits"] += 1
            floor = (self._last + iv) if self._last else 0.0
            try:
                slot = self._reserve(iv, floor)
                self.stats["shared"] += 1
            except Exception as e:           # per-process spacing only
                self.stats["fallback"] += 1
                self.stats["error"] = "%s: %s" % (type(e).__name__, str(e)[:120])
                slot = max(time.time(), floor)
            self._last = slot
        d = slot - time.time()
        if d > 0:
            self.stats["slept_s"] += d
            time.sleep(d)


SHARED = SharedThrottle()
