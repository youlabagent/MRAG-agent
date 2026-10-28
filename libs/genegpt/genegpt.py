"""NCBI E-utilities client for MRAG: GeneRIF grounding and human gene-ID resolution.

Parts of this module derive from NCBI GeneGPT (Jin et al., Bioinformatics 2024;
https://github.com/ncbi/GeneGPT): call_api grew out of GeneGPT's E-utilities call
helper.  GeneGPT's few-shot LLM tool-use loop is not used by MRAG and is not
included; what remains is deterministic (no LLM), with an NCBI throttle/backoff
(10 req/s with the shared api_key, else 3).

Public API:
  call_api(url, ...)                      -> bytes  (shared throttle, retries, payload checks)
  gene_generifs(gene, retmax=20)          -> list[{"text","pmids":[...]}]  (for report grounding)
  resolve_gene_id(symbol)                 -> {"id","how","symbol",...}  (E1 fix 2026-09-25: the
                                             local NCBI dump decides; esearch only when verified)
  gene_generifs_status(gene, retmax=20)   -> {"status": ok|empty|no_gene|failed, "lines", ...}
                                             (followup 2026-09-28: "NCBI lists no GeneRIF" and
                                             "the record could not be read" are no longer the same [])
"""
from __future__ import annotations
import json, os, re, time, urllib.request, urllib.parse
import http.client as _httpc
import random as _random
import socket as _socket
import urllib.error as _uerr
from typing import Optional

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ---------------------------------------------------------------- NCBI calls
# Credentials (2026-09-20): api_key / tool / email come from the shared loader
# ncbi_creds (env file ~/.config/mrag/ncbi.env, mode 600).  With a key NCBI
# allows 10 req/s instead of 3, so the client spacing drops 0.4 s -> 0.12 s.
# Without the file, everything below degrades to the pre-2026-09-20 behaviour.
import sys as _sys
if "/BIO1/MRAG/Agent/libs" not in _sys.path:
    _sys.path.insert(0, "/BIO1/MRAG/Agent/libs")
try:
    import ncbi_creds as _creds
except Exception:                            # no shared loader: anonymous, 3 req/s
    _creds = None

_last_ncbi = [0.0]


# ---------------------------------------------------------------------------
# followup 2026-09-28 (GeneRIF fetch fix): retries, backoff, payload checks
# ---------------------------------------------------------------------------
# WHY.  In the 2026-09-28 SCLC E2E the score runner's GeneRIF fetch lost EVERY
# line of 8 genes (581 GeneRIF lines in all, 125 of them from one gene) --
# most likely NCBI rate limiting while another
# process also called NCBI -- and nothing said so: call_api tried 3 times with
# fixed 1 s / 2 s sleeps, and every consumer turned the final exception into
# "no GeneRIF".  Now:
#  * transient errors are retried with EXPONENTIAL BACKOFF + JITTER:
#    NCBI_TRIES tries (MRAG_NCBI_TRIES), NCBI_BACKOFF_S doubling up to
#    NCBI_BACKOFF_MAX_S, each delay drawn from [d/2, d], a Retry-After header
#    honoured (capped), and no new attempt once NCBI_CALL_DEADLINE_S would be
#    passed.  Transient = HTTP 408 / 425 / 429 / 5xx, timeouts, connection
#    resets / refusals / DNS hiccups, a body cut short (IncompleteRead), and --
#    when the caller says what it expects (expect="xml" | "json") -- an empty,
#    HTML, truncated or unparseable body or an NCBI error payload that came
#    with HTTP 200 (<ERROR>..</ERROR>, {"error": ..}, esearchresult.ERROR);
#  * any other HTTP error (400 / 404 / 414 ...) is final at once;
#  * every attempt waits for the SHARED limiter (ncbi_creds.SHARED: one
#    interval across every MRAG process of this user, 0.12 s with the api_key,
#    0.4 s without);
#  * the final error is ncbi_creds.NcbiError (redacted) carrying .code,
#    .transient, .attempts and .error_class; CALL_STATS counts calls, attempts,
#    retries and failures per error class.
# The URL and the headers sent are unchanged, so with no error every request
# and every answer is byte for byte the pre-fix one.
NCBI_TRIES = 6
NCBI_BACKOFF_S = 1.0
NCBI_BACKOFF_MAX_S = 30.0
NCBI_CALL_DEADLINE_S = 180.0
TRANSIENT_HTTP = frozenset((408, 425, 429, 500, 502, 503, 504))
CALL_STATS = {"calls": 0, "attempts": 0, "retries": 0, "failures": 0, "by_class": {}, "last_error": ""}
_sleep = time.sleep              # tests replace these three (no real waiting, no randomness)
_jitter = _random.random
_now = time.time


class BadPayload(Exception):
    """HTTP 200, but the body is not a complete answer of the kind the request asked for."""


def _env_num(name, default, cast=float):
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    try:
        return cast(str(v).strip())
    except (TypeError, ValueError):
        return default


_XML_TAG = re.compile(rb"<([A-Za-z_][A-Za-z0-9_.:\-]*)")


def check_payload(body, expect=None) -> str:
    """"" when `body` is a complete answer of the `expect`ed kind ("xml" / "json";
    None = anything), else why not.  Never raises."""
    if not expect:
        return ""
    try:
        s = (body or b"").strip()
        if not s:
            return "empty body"
        head = s[:300].lower()
        if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
            return "an HTML page instead of %s" % expect
        if expect == "json":
            try:
                d = json.loads(s.decode("utf-8", "replace"))
            except Exception as e:
                return "invalid JSON (%s)" % str(e)[:80]
            if isinstance(d, dict):
                if d.get("error"):
                    return "NCBI error: %s" % str(d.get("error"))[:160]
                er = d.get("esearchresult")
                if isinstance(er, dict) and er.get("ERROR"):
                    return "NCBI error: %s" % str(er.get("ERROR"))[:160]
            return ""
        if expect == "xml":
            m = re.search(rb"<ERROR>(.*?)</ERROR>", s, re.S)
            if m:
                return "NCBI error: %s" % m.group(1).decode("utf-8", "replace").strip()[:160]
            i = 0
            while True:
                j = s.find(b"<", i)
                if j < 0:
                    return "no XML element"
                if s[j:j + 2] in (b"<?", b"<!"):
                    k = s.find(b">", j)
                    if k < 0:
                        return "truncated XML (prolog)"
                    i = k + 1
                    continue
                m = _XML_TAG.match(s, j)
                if not m:
                    return "not XML"
                root = m.group(1)
                break
            if s.endswith(b"</" + root + b">"):
                return ""
            if s.endswith(b"/>") and s.find(b">", j) == len(s) - 1:
                return ""            # a self-closing root element, e.g. <eSearchResult/>
            return "truncated XML (no closing </%s>)" % root.decode("ascii", "replace")
        return ""
    except Exception as e:          # the check itself must never be the failure
        return ""


def _error_class(e):
    """(transient?, class) of an exception raised by one attempt."""
    if isinstance(e, BadPayload):
        return True, "bad-payload"
    if isinstance(e, _uerr.HTTPError):
        c = int(getattr(e, "code", 0) or 0)
        return (c in TRANSIENT_HTTP or 500 <= c < 600), "http-%d" % c
    if isinstance(e, (_socket.timeout, TimeoutError)):
        return True, "timeout"
    if isinstance(e, _httpc.IncompleteRead):
        return True, "incomplete-read"
    if isinstance(e, ConnectionError):
        return True, "connection"
    if isinstance(e, _uerr.URLError):
        r = getattr(e, "reason", None)
        if isinstance(r, (_socket.timeout, TimeoutError)):
            return True, "timeout"
        if isinstance(r, ConnectionError):
            return True, "connection"
        return True, "url-error"         # DNS hiccup, unreachable: bounded retries
    if isinstance(e, _httpc.HTTPException):
        return True, "http-protocol"
    if isinstance(e, OSError):
        return True, "os-error"
    return False, type(e).__name__


def _retry_after(e):
    try:
        v = e.headers.get("Retry-After") if getattr(e, "headers", None) is not None else None
        return max(0.0, float(v)) if v not in (None, "") else None
    except Exception:
        return None


def _throttle_wait():
    if _creds is not None:
        _creds.SHARED.wait()                 # one interval across every MRAG process
    else:
        delta = time.time() - _last_ncbi[0]
        if delta < 0.4:                      # NCBI: <=3 req/s without api key
            time.sleep(0.4 - delta)
    _last_ncbi[0] = time.time()


def _redact(text) -> str:
    try:
        return _creds.redact(text) if _creds is not None else str(text)
    except Exception:
        return str(text)


def call_api(url: str, timeout: int = 30, retries: Optional[int] = None, expect: Optional[str] = None,
             deadline: Optional[float] = None, info: Optional[dict] = None) -> bytes:
    """GET an NCBI URL through the shared limiter, retrying transient errors with
    exponential backoff + jitter (see the block above).

    `retries` = the number of TRIES (default NCBI_TRIES / MRAG_NCBI_TRIES);
    `expect` = "xml" | "json" also checks the body (an incomplete / error answer
    is a transient error); `deadline` = seconds after which no new attempt starts
    (NCBI_CALL_DEADLINE_S); `info` (a dict) receives attempts / error_class /
    seconds of this call.

    The api_key is attached HERE and nowhere else: `url` -- the string callers
    keep and may log -- never carries it, and a final failure is re-raised with
    the message redacted, so the key cannot reach a trace, a log or a report."""
    url = url.replace(" ", "+")
    sent = _creds.add_params(url) if _creds is not None else url
    tries = int(retries) if retries else max(1, _env_num("MRAG_NCBI_TRIES", NCBI_TRIES, int))
    base = _env_num("MRAG_NCBI_BACKOFF_S", NCBI_BACKOFF_S)
    cap = _env_num("MRAG_NCBI_BACKOFF_MAX_S", NCBI_BACKOFF_MAX_S)
    dl = float(deadline) if deadline else _env_num("MRAG_NCBI_CALL_DEADLINE_S", NCBI_CALL_DEADLINE_S)
    t0 = _now()
    CALL_STATS["calls"] += 1
    last, n = None, 0
    for n in range(1, tries + 1):
        _throttle_wait()
        CALL_STATS["attempts"] += 1
        try:
            req = urllib.request.Request(sent, headers={"User-Agent": "MRAG-GeneGPT/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            why = check_payload(body, expect)
            if why:
                raise BadPayload(why)
            if info is not None:
                info.update(attempts=n, error_class="", seconds=round(_now() - t0, 2))
            return body
        except Exception as e:
            transient, cls = _error_class(e)
            last = (e, transient, cls)
            CALL_STATS["by_class"][cls] = CALL_STATS["by_class"].get(cls, 0) + 1
            if not transient or n >= tries:
                break
            d = min(cap, base * (2 ** (n - 1)))
            d = d * (0.5 + 0.5 * _jitter())
            ra = _retry_after(e)
            if ra is not None:
                d = max(d, min(ra, cap))
            if _now() - t0 + d > dl:
                break
            CALL_STATS["retries"] += 1
            _sleep(d)
    CALL_STATS["failures"] += 1
    e, transient, cls = last
    msg = _redact("%s: %s" % (type(e).__name__, e))
    CALL_STATS["last_error"] = msg[:300]
    if info is not None:
        info.update(attempts=n, error_class=cls, seconds=round(_now() - t0, 2), error=msg[:300])
    if _creds is None:
        try:
            e.transient, e.attempts, e.error_class = transient, n, cls
        except Exception:
            pass
        raise e
    err = _creds.NcbiError(msg, getattr(e, "code", None))
    err.transient, err.attempts, err.error_class = transient, n, cls
    raise err from None


_CALL_API_REAL = call_api


def _call(url, expect=None, info=None, timeout=30):
    """call_api with the followup-2026-09-28 arguments -- or without them when call_api
    has been replaced by a function that does not take them (tests stub it)."""
    f = call_api
    if f is _CALL_API_REAL:
        return f(url, timeout=timeout, expect=expect, info=info)
    try:
        return f(url, timeout=timeout, expect=expect, info=info)
    except TypeError as e:
        if "unexpected keyword" not in str(e):
            raise
        return f(url, timeout=timeout)

# ---------------------------------------------------------------- gene id (E1)
# FIX 2026-09-25 (E1, "GeneRIF gene-ID lookup").  The gene-ID lookup used to run
# esearch(db=gene, "<SYM>[sym] AND human[orgn]", retmax=1) and take the FIRST
# hit.  [sym] also matches ALIASES, and relevance order put another gene first
# for 5 of the 249 symbols one run looked up: CGA -> 1113 CHGA (chromogranin A
# lists CGA as an alias), LHFPL4 -> LHFPL3, MT1L -> MT1X, NPR3 -> NPRL3,
# UGT2B11 -> UGT2B4.  Every GeneRIF line
# the runs showed for "CGA" (237/237) was chromogranin A's, so the scorer was
# shown another gene's evidence under that symbol.
#
# Resolution order now (resolve_gene_id):
#   1. exact OFFICIAL symbol in the local NCBI Homo_sapiens.gene_info dump (the
#      same file evidence_gate.gene_fullname reads) -- one GeneID, no search;
#   1b. case-insensitive official symbol, only when that is unique;
#   2. only when the symbol is not an official symbol at all: a SYNONYM match,
#      only when exactly one gene lists it (a renamed gene, e.g. TMEM30B ->
#      CDC50B, keeps working; an ambiguous alias is refused);
#   3. only when the dump cannot decide: the old esearch, but every returned
#      record is read back (esummary) and a hit is taken ONLY when its official
#      symbol equals the query.  A hit whose symbol differs is never taken.
# MRAG_GENEGPT_ESEARCH_FALLBACK=0 switches step 3 off (dump only).
GENE_INFO_GZ = "/BIO1/MRAG/MD_check/Homo_sapiens.gene_info.gz"
_GENE_IDX: dict = {}
_RESOLVE_CACHE: dict = {}


def _gene_index(path: Optional[str] = None) -> dict:
    """{"sym": {symbol: [GeneID]}, "sym_ci": {SYMBOL: [GeneID]},
    "syn": {alias: [GeneID]}, "id2sym": {GeneID: symbol}} from the local NCBI
    dump.  Parsed once per path (~190k rows, ~1 s).  Never raises: a missing
    dump gives empty maps and resolution falls through to the verified
    esearch."""
    import os as _os, gzip as _gz
    p = path or _os.environ.get("MRAG_GENE_INFO") or GENE_INFO_GZ
    if p in _GENE_IDX:
        return _GENE_IDX[p]
    sym, sym_ci, syn, id2sym = {}, {}, {}, {}
    try:
        op = _gz.open if str(p).endswith(".gz") else open
        with op(p, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 5:
                    continue
                gid, s = f[1].strip(), f[2].strip()
                if not gid or not s or s == "-":
                    continue
                id2sym[gid] = s
                sym.setdefault(s, []).append(gid)
                sym_ci.setdefault(s.upper(), []).append(gid)
                if f[4].strip() not in ("", "-"):
                    for a in f[4].split("|"):
                        a = a.strip()
                        if a and a != "-":
                            syn.setdefault(a, []).append(gid)
    except Exception:
        sym, sym_ci, syn, id2sym = {}, {}, {}, {}
    _GENE_IDX[p] = {"sym": sym, "sym_ci": sym_ci, "syn": syn, "id2sym": id2sym, "path": p}
    return _GENE_IDX[p]


def _esearch_verified(gene: str, retmax: int = 10) -> dict:
    """Step 3: esearch, then read every hit back and keep only a record whose
    OFFICIAL symbol equals the query (exact, then case-insensitive)."""
    q = urllib.parse.quote(f"{gene}[sym] AND human[orgn]")
    # followup 2026-09-28: expect= makes an incomplete / error answer a (retried) failure
    # instead of "no hit"
    xml = _call(f"{_EUTILS}/esearch.fcgi?db=gene&retmax={int(retmax)}&term={q}", expect="xml"
                ).decode("utf-8", "ignore")
    ids = re.findall(r"<Id>(\d+)</Id>", xml)
    if not ids:
        return {"id": None, "how": "none", "symbol": "", "candidates": []}
    d = json.loads(_call(f"{_EUTILS}/esummary.fcgi?db=gene&retmode=json&id="
                         + ",".join(ids), expect="json").decode("utf-8", "ignore"))
    res = (d or {}).get("result") or {}
    cands = [(i, str((res.get(i) or {}).get("name") or "")) for i in ids]
    for want in (lambda n: n == gene, lambda n: n.upper() == gene.upper()):
        for i, n in cands:
            if n and want(n):
                return {"id": i, "how": "esearch-verified", "symbol": n,
                        "candidates": [c[0] for c in cands]}
    return {"id": None, "how": "esearch-rejected", "symbol": "",
            "candidates": ["%s:%s" % c for c in cands]}


def resolve_gene_id(gene: str, path: Optional[str] = None, live: Optional[bool] = None) -> dict:
    """Human NCBI GeneID for an official symbol, never another gene's.

    Returns {"id": str|None, "how": "official" | "official-ci" | "synonym" |
    "esearch-verified" | "ambiguous" | "esearch-rejected" | "none" | "error",
    "symbol": the resolved record's official symbol, "candidates": [...]}.
    Cached per (symbol, path).  NEVER raises."""
    import os as _os
    g = str(gene or "").strip()
    if not g:
        return {"id": None, "how": "none", "symbol": "", "candidates": []}
    ck = (g, path or _os.environ.get("MRAG_GENE_INFO") or GENE_INFO_GZ)
    if ck in _RESOLVE_CACHE:
        return dict(_RESOLVE_CACHE[ck])
    ix = _gene_index(path)
    out = None
    ids = ix["sym"].get(g) or []
    if len(ids) == 1:
        out = {"id": ids[0], "how": "official", "symbol": g, "candidates": ids}
    elif not ids:
        ci = ix["sym_ci"].get(g.upper()) or []
        if len(ci) == 1:
            out = {"id": ci[0], "how": "official-ci", "symbol": ix["id2sym"].get(ci[0], ""),
                   "candidates": ci}
        elif not ci:
            syn = sorted(set(ix["syn"].get(g) or []))
            if len(syn) == 1:
                out = {"id": syn[0], "how": "synonym", "symbol": ix["id2sym"].get(syn[0], ""),
                       "candidates": syn}
            elif len(syn) > 1:
                out = {"id": None, "how": "ambiguous", "symbol": "", "candidates": syn}
    if out is None or (out["id"] is None and out["how"] == "ambiguous"):
        amb = out
        fb = str(_os.environ.get("MRAG_GENEGPT_ESEARCH_FALLBACK", "1")).strip().lower()
        use_live = (fb not in ("0", "false", "no", "off")) if live is None else bool(live)
        if use_live:
            try:
                out = _esearch_verified(g)
            except Exception as e:
                out = {"id": None, "how": "error", "symbol": "", "candidates": [],
                       "error": str(e)[:160]}
            if out["id"] is None and amb is not None:
                out = dict(amb, esearch=out.get("how"))
        elif out is None:
            out = {"id": None, "how": "none", "symbol": "", "candidates": []}
    if out.get("how") != "error":          # a network failure is not cached
        _RESOLVE_CACHE[ck] = dict(out)
    return out


# BioGRID interaction detection-method labels are stored in the NCBI Gene
# "Interactions" subtree as type=generif commentaries whose text is the assay
# name (not a Reference-into-Function statement). Filter them out.
_INTERACTION_ASSAY_RE = re.compile(
    r"^\s*(?:Affinity Capture|Cross-Linking-MS|Two-hybrid|Reconstituted Complex|"
    r"Co-fractionation|Co-purification|Co-localization|Co-crystal|Proximity Label|"
    r"Protein-peptide|Protein-RNA|PCA|FRET|Far Western|Biochemical Activity|"
    r"Synthetic|Dosage|Phenotypic|Positive Genetic|Negative Genetic)\b",
    re.I,
)


def _gc_collect_generifs(el, out, retmax, in_interactions):
    """Recursively collect type=generif texts, skipping the Interactions
    subtree (BioGRID assay-method labels) and any assay-looking text."""
    heading = (el.findtext("Gene-commentary_heading") or "").strip().lower()
    now_int = in_interactions or (heading == "interactions")
    tp = el.find("Gene-commentary_type")
    ctype = tp.get("value") if tp is not None else ""
    if ctype == "generif" and not now_int:
        tnode = el.find("Gene-commentary_text")
        txt = tnode.text.strip() if (tnode is not None and tnode.text) else ""
        if txt and not _INTERACTION_ASSAY_RE.match(txt):
            pmids = [p.text for p in el.iter("PubMedId") if p.text]
            out.append({"text": txt, "pmids": pmids})
    for child in el.findall("Gene-commentary_comment/Gene-commentary"):
        if len(out) >= retmax:
            return
        _gc_collect_generifs(child, out, retmax, now_int)


def _generifs_from_xml(xml: str, retmax: int):
    """(records, how, error) from ONE efetch of a gene record: the GeneRIF
    statements exactly as gene_generifs() has always parsed them ("tree"; the
    naive split -- "fallback" -- only for a COMPLETE document ElementTree cannot
    read), or an error when the answer holds no Entrezgene record at all."""
    try:
        import xml.etree.ElementTree as _ET
        root = _ET.fromstring(xml)
        # followup 2026-09-28: an answer without the gene's record is a failed
        # fetch, not "this gene has no GeneRIF"
        if root.tag != "Entrezgene" and root.find(".//Entrezgene") is None:
            return [], "tree", "the answer holds no Entrezgene record"
        collected = []
        for comments in root.iter("Entrezgene_comments"):
            for el in comments.findall("Gene-commentary"):
                if len(collected) >= retmax:
                    break
                _gc_collect_generifs(el, collected, retmax, False)
        seen = set(); out = []
        for r in collected:
            if r["text"] in seen:
                continue
            seen.add(r["text"]); out.append(r)
            if len(out) >= retmax:
                break
        return out, "tree", ""
    except Exception:
        pass
    # fallback: naive split with assay-label filter
    out = []
    for block in re.split(r"<Gene-commentary>", xml):
        if "generif" not in block.lower():
            continue
        tm = re.search(r"<Gene-commentary_text>(.*?)</Gene-commentary_text>", block, re.S)
        if not tm:
            continue
        txt = tm.group(1).strip()
        if not txt or _INTERACTION_ASSAY_RE.match(txt):
            continue
        pmids = re.findall(r"<PubMedId>(\d+)</PubMedId>", block)
        out.append({"text": txt, "pmids": pmids})
        if len(out) >= retmax:
            break
    return out, "fallback", ""


def gene_generifs_status(gene: str, retmax: int = 20) -> dict:
    """followup 2026-09-28: the GeneRIF fetch WITH its outcome.  NEVER raises.

    {"gene", "status", "lines", "n", "gene_id", "how", "error", "attempts",
     "seconds", "parse"}; status is one of
      ok       the gene's NCBI Gene record was read; `lines` = its GeneRIFs (>= 1)
      empty    the record was read and lists no GeneRIF (a real "none")
      no_gene  no human Gene ID for this symbol (resolve_gene_id: the local dump,
               then a VERIFIED esearch, said so) -- nothing to fetch
      failed   the record could not be read: an NCBI / network error after
               call_api's retries, an incomplete or error answer, an answer
               without the gene's record, or a Gene-ID lookup that failed on
               the network.  `error` says which (redacted).
    For a record that was read, `lines` is exactly what gene_generifs() returned
    before this fix; before it, every failure was an unmarked []."""
    t0 = time.time()
    g = str(gene or "").strip()
    out = {"gene": g, "status": "failed", "lines": [], "n": 0, "gene_id": None, "how": "",
           "error": "", "attempts": 0, "seconds": 0.0, "parse": ""}
    try:
        try:
            res = resolve_gene_id(g)
        except Exception as e:
            res = {"id": None, "how": "error", "error": "%s: %s" % (type(e).__name__, e)}
        out["how"] = str(res.get("how") or "")
        gid = res.get("id")
        if not gid:
            if out["how"] == "error":
                out["error"] = _redact("Gene ID lookup failed: %s" % (res.get("error") or ""))[:300]
            else:
                out["status"] = "no_gene"
            return out
        out["gene_id"] = str(gid)
        info = {}
        try:
            raw = _call(f"{_EUTILS}/efetch.fcgi?db=gene&id={gid}&retmode=xml", expect="xml", info=info)
        except Exception as e:
            out["attempts"] = int(info.get("attempts") or getattr(e, "attempts", 0) or 0)
            out["error"] = _redact(str(e) if type(e).__name__ == "NcbiError"
                                   else "%s: %s" % (type(e).__name__, e))[:300]
            return out
        out["attempts"] = int(info.get("attempts") or 1)
        lines, how, err = _generifs_from_xml(raw.decode("utf-8", "ignore"), int(retmax))
        out["parse"] = how
        if err:
            out["error"] = "%s (Gene ID %s)" % (err, gid)
            return out
        out.update(lines=lines, n=len(lines), status=("ok" if lines else "empty"))
        return out
    except Exception as e:                  # never raise
        out.update(status="failed", lines=[], n=0, error=_redact("%s: %s" % (type(e).__name__, e))[:300])
        return out
    finally:
        out["seconds"] = round(time.time() - t0, 2)


def gene_generifs(gene: str, retmax: int = 20) -> list:
    """Deterministically fetch GeneRIF function statements (with PubMed IDs)
    for a human gene, for report grounding. Returns [{text, pmids:[...]}].
    Excludes the NCBI Gene 'Interactions' subtree (BioGRID assay-method
    labels such as 'Affinity Capture-MS' are stored there as type=generif).

    followup 2026-09-28: a record that could not be read RAISES (an NcbiError
    / RuntimeError naming the reason) instead of coming back as [] or as the
    part of a truncated answer; [] now always means "no GeneRIF" (or no Gene
    ID).  gene_generifs_status() gives the outcome without raising."""
    st = gene_generifs_status(gene, retmax=retmax)
    if st["status"] == "failed":
        msg = "GeneRIF fetch failed for %s: %s" % (st.get("gene"), st.get("error") or "unknown error")
        if _creds is not None:
            err = _creds.NcbiError(msg)
            err.transient, err.attempts, err.error_class = True, st.get("attempts"), "generif"
            raise err
        raise RuntimeError(msg)
    return st["lines"]
