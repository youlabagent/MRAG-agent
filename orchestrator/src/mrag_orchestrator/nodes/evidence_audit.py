"""Post-scoring EVIDENCE AUDIT of the curated targets (evaudit 2026-09-27,
revised fix2 2026-09-28).

WHY.  PI 2026-09-27: "make sure this never happens again".  Two FINAL
targets were emitted on counted evidence that did not support them -- a NEC
target (its only nec passage is a small-intestinal NET paper in the pre-2010
"carcinoma" wording) and an SCLC target (named only in a list of screened,
unaltered genes) -- and a manual review found them only
afterwards.  The deterministic filters in evidence_gate (1a-1c) catch those
two patterns before any LLM; this module is the GENERAL safety net behind
them, for the patterns nobody has written a rule for yet.

PI DECISION 2026-09-27 (binding): cohort / panel / list associations COUNT
(an SCLC target on a cohort-level association with metastasis, a panel
gene, the 'list' genes of the NEC top 20).  Screened-negative lists, the
wrong tumour type (a NET under the NEC carcinoma-only rule) and a symbol the
text uses for another gene stay invalid.  Wrongly removing a list / weak /
direct gene is the worst error, so removal is guarded (MODES below).

WHAT.  After scoring + gate + PubMed rescue (+ re-score after a filter) +
ranking, the candidates are walked in EMISSION order.  For each gene the
exact evidence the gate counted is assembled -- the domain GeneRIF lines in
full (with HOW the gate found the gene: symbol, full name, NCBI alias), the
domain passages (title + the excerpt around the gene), the rescue records'
title + the abstract window AROUND the gene + MeSH, all with PMIDs, the items
whose sentence names the gene AND the domain first -- and the scorer LLM
(same endpoint and model -- deploy 2026-09-28: the scorer profile's, qwen3.8 on
:11501 by default -- temperature 0, JSON only) answers a strict rubric:
gene_named / domain_subject / positive_finding -> verdict direct / list /
weak / invalid, a one-line reason and the supporting PMID.

MODES (MRAG_EVIDENCE_AUDIT_MODE or node param evidence_audit_mode):
  * flag (DEFAULT since fix3 2026-09-28): nothing is removed; every verdict is
    recorded, and weak / invalid verdicts are shown next to the gene in the
    report.  Why the default changed (second verification, finding 2): with
    the deterministic layers of fix2 on, removal by the LLM added NO correct
    removal on the measured data -- both FINAL invalids (the two cases under
    WHY) are already dropped by 1a / 1b / 1c before the audit, and of the
    held-out genes the only ones remove_named still removed were
    valid-but-borderline ones (one on a medullary thyroid carcinoma paper,
    one whose SCLC result lay outside the 700-character window the model
    was shown) -- and wrongly removing a list / weak / direct gene is the
    worst error (PI).  Removal stays available for an opt-in run:
  * remove_named: a gene is removed -- and replaced by the next pool
    gene -- only when the verdict is "invalid" AND the model confirms the gene
    is named in the evidence it was shown (gene_named=true) AND its own answers
    agree with "invalid" (domain_subject=false or positive_finding=false: the
    wrong tumour type, a screened-negative or a result-free mention) AND a
    second, focused call CONFIRMS the removal (confirm_removal: "keep" if any
    item reports any result for the gene in a tumour that fits the domain,
    also partial / weak / among other grades; a failed confirmation call
    keeps the gene).  On the benchmark this second call is what keeps
    borderline weak genes (e.g. one studied in NETs of all grades incl. poorly
    differentiated) in the report.  An "invalid" with
    gene_named=false is mostly "not visible in what the model was shown"
    (a gene named only by an alias, one cut out of a long abstract): the gene is
    emitted and FLAGGED -- with the label 'unverifiable', not 'invalid' (deploy
    2026-09-28, in every mode that keeps it);
  * remove: every "invalid" is removed (the 2026-09-27 build's behaviour);
  * off: the LLM audit does not run (the deterministic layers stay on).
MRAG_EVIDENCE_AUDIT_KEEP="G1,G2" (node param evidence_audit_keep): genes the
PI has overruled are never removed, only flagged.
GUARDS.  A failed call is not a verdict: the gene is emitted "unaudited"
(fail-open).  MRAG_EVIDENCE_AUDIT_FAIL=closed skips a failing gene instead,
but only while the endpoint is otherwise healthy (another call of this run
succeeded and the circuit breaker is closed).  CIRCUIT BREAKER: after
AUDIT_BREAKER_FAILS (3) consecutive failed calls, or past the time budget
(MRAG_EVIDENCE_AUDIT_BUDGET_S, 600 s), the remaining genes are emitted
"unaudited" without calling.  MALFUNCTION GUARD: when more than
AUDIT_GUARD_MAX (5) genes, or more than AUDIT_GUARD_FRAC (25 %) of >= 8
audited genes, would be removed, nothing is removed: the plain emission
order is kept and those genes are flagged "audit-suspect".
The walk stops when `n_target` genes passed, the pool ends, or `limit` (40)
genes have been audited.  A gene without any counted evidence has nothing to
audit: audit_verdict="none", no LLM call.
REPORT.  Only the flags a reader must see are displayed (Section 10 table,
Section 14 cards): weak, invalid (kept), unverifiable (an 'invalid' with
gene_named=false; deploy 2026-09-28), audit-suspect, unaudited.
direct / list stay in the JSON (curated_targets.json, scored.json).
Audits are cached by (gene, domains, evidence hash, model, AUDIT_VERSION) in
<curated_targets>/_evidence_audit_cache.json (or MRAG_EVIDENCE_AUDIT_CACHE).

SWITCH.  MRAG_EVIDENCE_AUDIT=0 or the master switch MRAG_EVIDENCE_DEFENCE=0
switches the audit AND every other 2026-09-27 layer off (fix3 2026-09-28,
second verification finding 6: MRAG_EVIDENCE_AUDIT=0 is the documented
rollback and restores the pre-2026-09-27 pipeline byte for byte);
MRAG_EVIDENCE_AUDIT_MODE=off or the node param evidence_audit=False switches
off only this LLM audit (the node then runs the plain rerank and writes no
audit field).  Default ON, mode flag.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

AUDIT_VERSION = "evaudit_20260928.7"
AUDIT_LIMIT = 40
AUDIT_WORKERS = 2               # the scorer endpoint is shared: <= 2 requests at a time
AUDIT_VERDICTS = ("direct", "list", "weak", "invalid")
AUDIT_MODES = ("flag", "remove_named", "remove", "off")
AUDIT_MODE_DEFAULT = "flag"            # fix3 2026-09-28 (was remove_named)
AUDIT_BREAKER_FAILS = 3
AUDIT_BUDGET_S = 600.0
AUDIT_CALL_TIMEOUT_S = 180
AUDIT_CALL_RETRIES = 1
AUDIT_GUARD_MAX = 5
AUDIT_GUARD_FRAC = 0.25
AUDIT_GUARD_MIN_N = 8
# fix2 2026-09-28: larger caps, and the items are ordered best-first (gene and
# domain in one sentence) BEFORE the cap, so a gene's best item is not cut
MAX_RIF, MAX_SNIP, MAX_REC = 10, 8, 6
RIF_CHARS, SNIP_CHARS, ABS_CHARS = 600, 700, 1800
# the flags the report shows (direct / list stay in the JSON)
DISPLAY_FLAGS = ("weak", "invalid", "unverifiable", "audit-suspect", "unaudited")


def invalid_label(a):
    """deploy 2026-09-28 (decision 3): the report label of a kept 'invalid' verdict.
    'unverifiable' when the audit model answered gene_named=false -- it did not find the gene
    in the items it was shown, mostly an NCBI alias or a gene outside the excerpt (bench4:
    two such genes), so its 'invalid' is not a finding about the evidence; 'invalid'
    otherwise.  The verdict itself stays 'invalid' in the JSON (audit_verdict)."""
    return "unverifiable" if (a.get("audit_flags") or {}).get("gene_named") is False else "invalid"


def _flag(v, default=True):
    if v is None or str(v).strip() == "":
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() not in ("0", "off", "false", "no", "none")


def defence_on():
    """The master switch MRAG_EVIDENCE_DEFENCE (fix2 2026-09-28); default ON.
    fix3 2026-09-28: MRAG_EVIDENCE_AUDIT=0 (the documented rollback) too."""
    return (_flag(os.environ.get("MRAG_EVIDENCE_DEFENCE"), True)
            and _flag(os.environ.get("MRAG_EVIDENCE_AUDIT"), True))


def audit_enabled(param=None):
    """Default ON; off under MRAG_EVIDENCE_DEFENCE=0 or MRAG_EVIDENCE_AUDIT=0
    (both the full rollback, fix3 2026-09-28), with the node param
    evidence_audit False, or with MRAG_EVIDENCE_AUDIT_MODE=off."""
    if not defence_on():
        return False
    if param is not None and not _flag(param):
        return False
    return audit_mode() != "off"


def audit_mode(param=None):
    """flag (default) | remove_named | remove | off; param wins over
    MRAG_EVIDENCE_AUDIT_MODE; an unknown value falls back to the default."""
    for v in (param, os.environ.get("MRAG_EVIDENCE_AUDIT_MODE")):
        v = str(v or "").strip().lower().replace("-", "_")
        if v in AUDIT_MODES:
            return v
    return AUDIT_MODE_DEFAULT


def audit_keep(param=None):
    """Genes the PI overruled: never removed, only flagged."""
    out = set()
    for v in (param, os.environ.get("MRAG_EVIDENCE_AUDIT_KEEP")):
        if isinstance(v, (list, tuple, set)):
            out.update(str(x).strip() for x in v if str(x).strip())
        elif v:
            out.update(x.strip() for x in re.split(r"[,;\s]+", str(v)) if x.strip())
    return out


def audit_limit(param=None):
    for v in (param, os.environ.get("MRAG_EVIDENCE_AUDIT_LIMIT")):
        try:
            if v is not None and str(v).strip():
                return max(1, int(v))
        except (TypeError, ValueError):
            pass
    return AUDIT_LIMIT


def audit_budget_s():
    try:
        return max(1.0, float(os.environ.get("MRAG_EVIDENCE_AUDIT_BUDGET_S") or AUDIT_BUDGET_S))
    except ValueError:
        return AUDIT_BUDGET_S


def audit_fail_closed():
    return str(os.environ.get("MRAG_EVIDENCE_AUDIT_FAIL") or "open").strip().lower() == "closed"


def _workers():
    try:
        return max(1, min(AUDIT_WORKERS, int(os.environ.get("MRAG_EVIDENCE_AUDIT_WORKERS") or AUDIT_WORKERS)))
    except ValueError:
        return AUDIT_WORKERS


def _call_timeout():
    try:
        return max(10, int(os.environ.get("MRAG_EVIDENCE_AUDIT_TIMEOUT_S") or AUDIT_CALL_TIMEOUT_S))
    except ValueError:
        return AUDIT_CALL_TIMEOUT_S


# ---------------------------------------------------------------------------
# evidence: exactly what the gate counted
# ---------------------------------------------------------------------------
def _clip(x, n):
    x = re.sub(r"\s+", " ", str(x or "")).strip()
    return x if len(x) <= n else x[:n].rstrip() + " ..."


def _counted_fallback(row, scored_doc, dspecs, eg):
    """Counted items for a scored.json written without evidence_counted (the
    evidence filters off): names_gene AND a domain tie, the v1 gate's rule."""
    g = row.get("gene", "")
    gr = (scored_doc.get("generif") or {}).get(g) or []
    vd = (scored_doc.get("vdb") or {}).get(g) or []
    try:
        ev = eg._evidence_naming_v1(g, gr, vd, None, None, dspecs) if hasattr(eg, "_evidence_naming_v1") \
            else eg.evidence_naming(g, gr, vd, domains=dspecs)
    except Exception:
        return {"generif": [], "snippets": []}
    return {"generif": [{"i": i, "pmids": [], "how": k} for i, (k, t) in
                        enumerate(zip(ev["generif_kinds"], ev["generif_ties"])) if k and t],
            "snippets": [{"i": j, "pmid": str((vd[j] or {}).get("pmid") or "")} for j, (k, t) in
                         enumerate(zip(ev["snippet_kinds"], ev["snippet_ties"])) if k and t]}


def _best_first(texts, g, nm, dspecs, eg):
    """Stable order of item indices: an item whose sentence names the gene AND
    the domain first (evidence_gate.gene_domain_cooccur, window 0)."""
    def _k(i):
        if eg is None or not dspecs:
            return (1, i)
        try:
            hit = any(eg.gene_domain_cooccur(texts[i], g, nm, sp, "", None, window=0) for sp in dspecs)
        except Exception:
            hit = False
        return (0 if hit else 1, i)
    return sorted(range(len(texts)), key=_k)


def _window(eg, text, g, nm, dspecs, width):
    """`text` cut AROUND the gene (and the nearest domain mention) when longer
    than `width`; the text's start when the gene is not in it."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(t) <= width or eg is None:
        return t
    try:
        return eg.gene_excerpt(t, g, nm, dspecs, width=width) or t
    except Exception:
        return t


def assemble_evidence(row, scored_doc, dspecs, eg=None, fetch=None, errors=None):
    """[{"id", "kind", "pmid", "title", "text", "mesh"}] -- the evidence the gate
    counted for this row (GeneRIF lines, passages, rescue records), each kind
    ordered best-first and windowed around the gene BEFORE the caps.
    `fetch(pmids) -> ({pmid: record}, err)` fills rescue records whose text
    was not stored.
    followup 2026-09-28 (fix 1): `errors` (a list) receives what `fetch` could
    not read -- {"what": "mesh" | "records", "error", "pmids"} -- instead of the
    item silently going without its MeSH headings or its abstract."""
    g = row.get("gene", "")
    gr = (scored_doc.get("generif") or {}).get(g) or []
    vd = (scored_doc.get("vdb") or {}).get(g) or []
    ec = row.get("evidence_counted")
    if not isinstance(ec, dict) and eg is not None:
        ec = _counted_fallback(row, scored_doc, dspecs, eg)
    ec = ec or {}
    nm = ""
    if eg is not None:
        try:
            nm = eg.gene_fullname(g) or ""
        except Exception:
            nm = ""
    items = []
    rifs = [x for x in (ec.get("generif") or []) if 0 <= int(x.get("i", -1)) < len(gr)]
    order = _best_first([str(gr[int(x["i"])]) for x in rifs], g, nm, dspecs, eg)
    for k in order[:MAX_RIF]:
        x = rifs[k]
        # v5: say HOW the gate found the gene in the line -- the LLM otherwise
        # rejects a line that names the gene by an NCBI alias as "a different
        # gene" despite the rubric
        how = str(x.get("how") or "")
        if how.startswith("alias:"):
            kind = ("GeneRIF (NCBI Gene record of %s; this line names %s by its NCBI alias %s)"
                    % (g, g, how.split(":", 1)[1]))
        elif how in ("full-name", "fullname", "name", "full_name"):
            kind = "GeneRIF (NCBI Gene record of %s; this line names %s by its full name)" % (g, g)
        else:
            kind = "GeneRIF (NCBI Gene record of %s)" % g
        items.append({"kind": kind, "pmid": ",".join(x.get("pmids") or []), "title": "",
                      "text": _clip(_window(eg, gr[int(x["i"])], g, nm, dspecs, RIF_CHARS), RIF_CHARS + 10),
                      "mesh": ""})
    snips = [vd[int(x.get("i", -1))] for x in (ec.get("snippets") or [])
             if 0 <= int(x.get("i", -1)) < len(vd) and isinstance(vd[int(x.get("i", -1))], dict)]

    def _stext(h):
        t = str(h.get("excerpt") or "") or str(h.get("snippet") or "")
        return _window(eg, t, g, nm, dspecs, SNIP_CHARS)
    stexts = [_stext(h) for h in snips]
    sorder = _best_first([str(h.get("title") or "") + " . " + t for h, t in zip(snips, stexts)],
                         g, nm, dspecs, eg)[:MAX_SNIP]
    snips = [snips[k] for k in sorder]
    stexts = [stexts[k] for k in sorder]
    # the paper's MeSH headings travel with its passage (what the paper is
    # about -- "Ileal Neoplasms; Enterochromaffin Cells" -- when the excerpt
    # alone does not say it); one cached efetch per gene, none when unavailable
    pmesh = {}
    if snips and fetch is not None:
        _want = [str(h.get("pmid") or "") for h in snips]
        try:
            got, _err = fetch(_want)
            pmesh = {p: r.get("mesh") for p, r in (got or {}).items() if r.get("mesh")}
            _miss = [p for p in _want if p and p not in (got or {})]
            if _err and _miss and errors is not None:
                errors.append({"what": "mesh", "error": str(_err)[:200], "pmids": _miss})
        except Exception as _fe:
            pmesh = {}
            if errors is not None:
                errors.append({"what": "mesh", "error": "%s: %s" % (type(_fe).__name__, str(_fe)[:160]),
                               "pmids": [p for p in _want if p]})
    for h, t in zip(snips, stexts):
        m = pmesh.get(str(h.get("pmid") or "")) or ""
        if isinstance(m, (list, tuple)):
            m = "; ".join(str(x) for x in m)
        items.append({"kind": "indexed literature passage", "pmid": str(h.get("pmid") or ""),
                      "title": _clip(h.get("title"), 200), "text": _clip(t, SNIP_CHARS + 10),
                      "mesh": _clip(m, 400)})
    if str(row.get("evidence_gate") or "") == "pubmed":
        pe = row.get("pubmed_evidence") or {}
        want, stored = [], {}
        for _k, d in (pe.get("by_domain") or {}).items():
            for p in d.get("verified") or []:
                if str(p) not in want:
                    want.append(str(p))
            stored.update(d.get("records") or {})
        miss = [p for p in want if p not in stored]
        if miss and fetch is not None:
            _rerr = ""
            try:
                got, _rerr = fetch(miss)
                stored.update(got or {})
            except Exception as _fe:
                _rerr = "%s: %s" % (type(_fe).__name__, str(_fe)[:160])
            _still = [p for p in miss if p not in stored]
            if _still and errors is not None:
                errors.append({"what": "records", "error": str(_rerr or "record not returned")[:200],
                               "pmids": _still})
        rtexts = [str((stored.get(p) or {}).get("title") or "") + " . " +
                  str((stored.get(p) or {}).get("abstract") or "") for p in want]
        want = [want[k] for k in _best_first(rtexts, g, nm, dspecs, eg)[:MAX_REC]]
        for p in want:
            r = stored.get(p) or {}
            mesh = r.get("mesh") or ""
            if isinstance(mesh, (list, tuple)):
                mesh = "; ".join(str(m) for m in mesh)
            # v5: an abstract longer than ABS_CHARS is cut AROUND the gene
            # mention, not from its start -- a head-clip cut the gene out of
            # one long abstract and the LLM answered "gene not named"
            ab = _window(eg, r.get("abstract"), g, nm, dspecs, ABS_CHARS)
            items.append({"kind": "PubMed record (live search)", "pmid": p,
                          "title": _clip(r.get("title"), 300),
                          "text": _clip(ab, ABS_CHARS + 10) or "(abstract not available)",
                          "mesh": _clip(mesh, 400)})
    for n, it in enumerate(items, 1):
        it["id"] = "E%d" % n
    return items


def evidence_hash(gene, keys, items, model):
    blob = json.dumps({"v": AUDIT_VERSION, "model": str(model or ""), "gene": gene,
                       "domains": list(keys), "items": [{k: it.get(k, "") for k in
                                                        ("kind", "pmid", "title", "text", "mesh")}
                                                       for it in items]},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# the rubric
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You audit the literature evidence behind a candidate target gene before it is reported. "
    "Judge ONLY the evidence items given; do not use outside knowledge about the gene or the disease. "
    "A verdict of 'invalid' removes the gene from the report: give it only when the items clearly do "
    "NOT support the gene in the domain (wrong gene, wrong disease or tumour type, a screened-negative "
    "or result-free mention). When unsure between 'invalid' and 'weak', answer 'weak'. "
    "Output ONLY one JSON object.")


def build_prompt(gene, gene_name, dspecs, items, definition):
    doms = []
    for sp in dspecs or []:
        doms.append("- %s: %s" % (sp.get("label") or sp.get("display") or sp.get("key"),
                                   definition(sp.get("key"))))
    ev = []
    for it in items:
        head = "[%s] PMID %s | %s" % (it["id"], it.get("pmid") or "-", it["kind"])
        body = []
        if it.get("title"):
            body.append("title: " + it["title"])
        body.append("text: " + it.get("text", ""))
        if it.get("mesh"):
            body.append("MeSH: " + it["mesh"])
        ev.append(head + "\n  " + "\n  ".join(body))
    many = len(dspecs or []) > 1
    return (
        "GENE: %s (official symbol%s)\n"
        "DOMAIN%s OF INTEREST (%s):\n%s\n\n"
        "EVIDENCE ITEMS (the only evidence that grounded this gene's score):\n%s\n\n"
        "Judge every item; the gene is as good as its BEST item. Answer from the items only:\n"
        "1. gene_named: is %s ITSELF named in at least one item -- not only another gene that shares an "
        "alias with it, not only a gene family, and not only a longer symbol that contains it? (A GeneRIF "
        "line belongs to %s's own NCBI Gene record, so it may name the gene by an alias.)\n"
        "2. domain_subject: in the item(s) naming the gene, is the domain%s itself what the gene is "
        "discussed in, as defined above? Another disease, a NET / carcinoid / small-intestinal or "
        "serotonin-producing tumour instead of a neuroendocrine carcinoma, or NSCLC instead of SCLC is NOT "
        "the domain; cell lines and models of the domain are the domain.\n"
        "3. positive_finding: do those item(s) report some result involving %s in the domain (expression, "
        "function, mutation or alteration found, prognosis, marker, target, or membership of a list / "
        "panel / signature that has a result, also when named together with its paralogs)? A list the "
        "study's own analysis produced -- genes identified or nominated as candidates / markers / "
        "domain-associated factors, genes up- or down-regulated or showing a trend -- IS a result for each "
        "member, even when followed by 'for further assessment'; so is a study of a gene family or paralog "
        "group that names the gene (e.g. PEA3 factors ETV4 / ETV5, IGF2BP1/2/3). A cohort-, panel- or screen-level association -- the gene named among genes whose alterations or expression correlated with an outcome, metastatic site, subtype or response in the domain -- IS a positive finding ('list'). An item's TITLE is part "
        "of the item: a title that reports the result for the family or list the text names the gene in "
        "counts for the gene. A gene that was screened "
        "or tested and found NOT altered / not different, or that is only named among genes examined, "
        "cited or listed without any observation, is NOT a positive finding.\n"
        "Verdict:\n"
        "- \"direct\": 1-3 yes and the finding is specifically about %s;\n"
        "- \"list\": 1-3 yes, but %s is one item of a list, panel, signature or screen whose POSITIVE "
        "result it shares;\n"
        "- \"weak\": 1-2 yes, but the finding about it is indirect, peripheral or cut off (a marker of "
        "non-tumour cells, a candidate or contact list, a background sentence citing the gene in the "
        "domain inside a paper about another disease, a passage that ends before its result);\n"
        "- \"invalid\": ONLY when 1 is no, or 2 is no for every item, or the gene's only mentions are "
        "screened-negative or result-free. Indirect, brief or family-level support is \"weak\" or "
        "\"list\", not \"invalid\".\n"
        "Output ONLY: {\"gene_named\": true|false, \"domain_subject\": true|false, "
        "\"positive_finding\": true|false, \"verdict\": \"direct\"|\"list\"|\"weak\"|\"invalid\", "
        "\"pmid\": \"<PMID of the best supporting item, or empty>\", \"reason\": \"<one line, at most 30 "
        "words>\"}"
        % (gene, ("; full name: " + gene_name) if gene_name else "",
           "S" if many else "", "the gene is supported if at least one is" if many else "one",
           "\n".join(doms), "\n".join(ev), gene, gene, "s" if many else "", gene, gene, gene))


def _parse(txt):
    txt = re.sub(r"<think>.*?</think>", "", str(txt or ""), flags=re.S)
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError("no JSON object in the answer (%d chars)" % len(txt))
    o = json.loads(m.group(0))
    v = str(o.get("verdict") or "").strip().lower()
    if v not in AUDIT_VERDICTS:
        raise ValueError("verdict %r is not one of %s" % (v, AUDIT_VERDICTS))
    pm = re.sub(r"[^0-9,]", "", str(o.get("pmid") or ""))
    return {"verdict": v, "reason": re.sub(r"\s+", " ", str(o.get("reason") or "")).strip()[:300],
            "pmid": pm.split(",")[0] if pm else "",
            "gene_named": bool(o.get("gene_named")), "domain_subject": bool(o.get("domain_subject")),
            "positive_finding": bool(o.get("positive_finding"))}


def _ollama_base(endpoint):
    e = str(endpoint or "").rstrip("/")
    for suf in ("/v1/chat/completions", "/chat/completions", "/v1", "/api/chat"):
        if e.endswith(suf):
            return e[: -len(suf)]
    return e


def _post(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def call_llm(endpoint, model, system, user, timeout=600, retries=2):
    """ONE audit answer (dict) or raises after `retries` extra attempts.

    The scorer's endpoint and model, temperature 0.  The Qwen3.6 scorer model
    reasons before it answers even with "/no_think" (the OpenAI-compatible
    route then needs thousands of tokens and returns an empty content when
    they run out), so the audit uses Ollama's native /api/chat with
    think=false -- as narrative._glm_chat and detector._llm_call do -- and no
    num_ctx override (the loaded model is not reloaded).  A server without
    /api/chat (HTTP 404) gets the OpenAI-compatible request with a large
    max_tokens instead."""
    err = None
    base = _ollama_base(endpoint)
    for att in range(retries + 1):
        try:
            try:
                r = _post(base + "/api/chat", {
                    "model": model, "stream": False, "think": False,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "options": {"temperature": 0, "num_predict": 700}}, timeout)
                txt = (r.get("message") or {}).get("content") or ""
            except urllib.error.HTTPError as he:
                if he.code != 404:
                    raise
                r = _post(str(endpoint), {
                    "model": model, "temperature": 0, "max_tokens": 6000,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user + "\n\n/no_think"}]}, timeout)
                txt = r["choices"][0]["message"].get("content") or ""
            return _parse(txt)
        except Exception as e:
            err = e
            time.sleep(2.0 * (att + 1))
    raise RuntimeError("%s: %s" % (type(err).__name__, str(err)[:200]))


def call_llm_raw(endpoint, model, system, user, timeout=600, retries=2, num_predict=700):
    """The answer TEXT of one chat call (native /api/chat with think=false,
    temperature 0; the OpenAI-compatible route on a 404), or raises."""
    err = None
    base = _ollama_base(endpoint)
    for att in range(retries + 1):
        try:
            try:
                r = _post(base + "/api/chat", {
                    "model": model, "stream": False, "think": False,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "options": {"temperature": 0, "num_predict": num_predict}}, timeout)
                return (r.get("message") or {}).get("content") or ""
            except urllib.error.HTTPError as he:
                if he.code != 404:
                    raise
                r = _post(str(endpoint), {
                    "model": model, "temperature": 0, "max_tokens": 6000,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user + "\n\n/no_think"}]}, timeout)
                return r["choices"][0]["message"].get("content") or ""
        except Exception as e:
            err = e
            time.sleep(2.0 * (att + 1))
    raise RuntimeError("%s: %s" % (type(err).__name__, str(err)[:200]))


# ---------------------------------------------------------------------------
# fix2 2026-09-28: a removal must be CONFIRMED by a second, focused call
# ---------------------------------------------------------------------------
CONFIRM_SYSTEM = (
    "You double-check a proposed REMOVAL of a candidate target gene from a report. Removing a gene that has "
    "any valid support is a serious error; keeping a weakly supported gene is acceptable. Judge ONLY the "
    "evidence items given; do not use outside knowledge. Output ONLY one JSON object.")


def build_confirm_prompt(gene, gene_name, dspecs, items, definition, first_reason):
    doms = "\n".join("- %s: %s" % (sp.get("label") or sp.get("display") or sp.get("key"),
                                   definition(sp.get("key"))) for sp in dspecs or [])
    ev = []
    for it in items:
        body = ["title: " + it["title"]] if it.get("title") else []
        body.append("text: " + it.get("text", ""))
        if it.get("mesh"):
            body.append("MeSH: " + it["mesh"])
        ev.append("[%s] PMID %s | %s\n  %s" % (it["id"], it.get("pmid") or "-", it["kind"], "\n  ".join(body)))
    return (
        "GENE: %s%s\nDOMAIN OF INTEREST:\n%s\n\nEVIDENCE ITEMS:\n%s\n\n"
        "An auditor proposed to REMOVE %s from the report because: %s\n\n"
        "Answer \"keep\" if at least ONE item reports ANY result involving %s -- expression, function, an "
        "alteration found, an association, a marker, membership of a list / panel / cohort / signature that has "
        "a result, also weak, partial or indirect -- in a tumour that fits the domain definition, including when "
        "the item also covers other tumour types or grades (e.g. well- AND poorly differentiated tumours). "
        "Answer \"remove\" only if EVERY item is about another tumour type or disease as defined above, or "
        "about another gene, or reports that %s was screened or tested and found NOT altered / not different, "
        "or names it without any observation.\n"
        "Output ONLY: {\"decision\": \"keep\"|\"remove\", \"reason\": \"<one line, at most 30 words>\"}"
        % (gene, (" (full name: %s)" % gene_name) if gene_name else "", doms, "\n".join(ev), gene,
           first_reason or "-", gene, gene))


def _parse_confirm(o):
    """keep / remove from a confirmation answer: a JSON text, or a dict (a
    stub may answer in the audit's own shape: verdict invalid = remove)."""
    if isinstance(o, dict):
        d = o
    else:
        txt = re.sub(r"<think>.*?</think>", "", str(o or ""), flags=re.S)
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            raise ValueError("no JSON object in the confirmation answer")
        d = json.loads(m.group(0))
    dec = str(d.get("decision") or "").strip().lower()
    if not dec and d.get("verdict"):
        dec = "remove" if str(d.get("verdict")).strip().lower() == "invalid" else "keep"
    if dec not in ("keep", "remove"):
        raise ValueError("confirmation decision %r is not keep / remove" % dec)
    return {"decision": dec, "reason": re.sub(r"\s+", " ", str(d.get("reason") or "")).strip()[:300]}


def confirm_removal(row, a, dspecs, endpoint, model, cache, definition, eg=None, llm=None, confirm_llm=None):
    """{"decision": keep|remove, "reason", "cached", "error"} for one proposed
    removal.  A failed call keeps the gene (decision keep, error set)."""
    g = row.get("gene", "")
    key = str(a.get("audit_hash") or "") + ":confirm"
    hit = cache.get(key) if isinstance(cache, dict) and a.get("audit_hash") else None
    if isinstance(hit, dict) and hit.get("decision") in ("keep", "remove"):
        return dict(hit, cached=True)
    nm = ""
    if eg is not None:
        try:
            nm = eg.gene_fullname(g) or ""
        except Exception:
            nm = ""
    user = build_confirm_prompt(g, nm, dspecs, a.get("audit_items") or [], definition, a.get("audit_reason"))
    try:
        if confirm_llm is not None:
            o = confirm_llm(endpoint, model, CONFIRM_SYSTEM, user)
        elif llm is not None:
            o = llm(endpoint, model, CONFIRM_SYSTEM, user)
        else:
            o = call_llm_raw(endpoint, model, CONFIRM_SYSTEM, user, timeout=_call_timeout(),
                             retries=AUDIT_CALL_RETRIES, num_predict=300)
        c = _parse_confirm(o)
    except Exception as e:
        return {"decision": "keep", "reason": "", "cached": False,
                "error": "confirmation call failed (the gene is kept): %s" % str(e)[:160]}
    if isinstance(cache, dict) and a.get("audit_hash"):
        cache[key] = dict(c)
    return dict(c, cached=False, error="")


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def _load_cache(path):
    try:
        return json.loads(Path(path).read_text()) if path and Path(path).exists() else {}
    except Exception:
        return {}


def _save_cache(path, cache):
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
        os.replace(tmp, p)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# the walk
# ---------------------------------------------------------------------------
def audit_one(row, scored_doc, dspecs, endpoint, model, cache, definition, eg=None, fetch=None,
              gene_name=None, llm=None):
    """Audit ONE row; returns the audit record (never raises)."""
    g = row.get("gene", "")
    keys = [sp.get("key") for sp in dspecs or []]
    _ferr = []
    items = assemble_evidence(row, scored_doc, dspecs, eg, fetch, errors=_ferr)
    base = {"audit_version": AUDIT_VERSION, "audit_n_items": len(items)}
    if not items:
        return dict(base, audit_verdict="none", audit_reason=(
            "no counted literature evidence (the gate capped the score); nothing to audit"),
            audit_pmid="", audit_cached=False, audit_items=[])
    # followup 2026-09-28 (fix 1): a live PubMed record the gene is admitted on could not be read
    # -> the audit would judge a title without its abstract: not a verdict, the gene is 'unaudited'
    _rec_err = [x for x in _ferr if x.get("what") == "records"]
    if _rec_err:
        _m = "NCBI record fetch failed for PMID %s (%s)" % (", ".join(_rec_err[0]["pmids"][:5]),
                                                            _rec_err[0]["error"][:120])
        return dict(base, audit_verdict="unaudited", audit_reason="not audited: " + _m, audit_pmid="",
                    audit_cached=False, audit_error=_m[:200], audit_items=items)
    h = evidence_hash(g, keys, items, model)
    hit = cache.get(h) if isinstance(cache, dict) else None
    if isinstance(hit, dict) and hit.get("audit_verdict") in AUDIT_VERDICTS:
        return dict(hit, audit_cached=True, audit_hash=h, audit_items=items)
    nm = gene_name if gene_name is not None else ""
    if not nm and eg is not None:
        try:
            nm = eg.gene_fullname(g)
        except Exception:
            nm = ""
    user = build_prompt(g, nm, dspecs, items, definition)
    try:
        if llm is not None:
            o = llm(endpoint, model, SYSTEM_PROMPT, user)
        else:
            o = call_llm(endpoint, model, SYSTEM_PROMPT, user, timeout=_call_timeout(),
                         retries=AUDIT_CALL_RETRIES)
    except Exception as e:
        return dict(base, audit_verdict="unaudited", audit_reason="audit call failed: " + str(e)[:200],
                    audit_pmid="", audit_cached=False, audit_error=str(e)[:200], audit_hash=h,
                    audit_items=items)
    rec = dict(base, audit_verdict=o["verdict"], audit_reason=o["reason"], audit_pmid=o["pmid"],
               audit_flags={"gene_named": o["gene_named"], "domain_subject": o["domain_subject"],
                            "positive_finding": o["positive_finding"]},
               audit_hash=h)
    if _ferr:
        # followup 2026-09-28 (fix 1): audited, but without the MeSH headings NCBI did not return;
        # said on the row and never cached (a later run with the headings asks again)
        rec["audit_error"] = ("audited without the MeSH headings of PMID %s (NCBI fetch failed: %s)"
                              % (", ".join(_ferr[0]["pmids"][:5]), _ferr[0]["error"][:100]))[:200]
        return dict(rec, audit_cached=False, audit_items=items)
    if isinstance(cache, dict):
        cache[h] = {k: v for k, v in rec.items() if k != "audit_hash"}
    return dict(rec, audit_cached=False, audit_items=items)


def removal_decision(a, mode, keep=()):
    """(remove?, why-not) for one audit record under `mode`."""
    if a.get("audit_verdict") != "invalid":
        return False, ""
    if a.get("gene") in (keep or ()):
        return False, "overruled (MRAG_EVIDENCE_AUDIT_KEEP): flagged, not removed"
    if mode in ("flag", "off"):
        return False, "audit mode 'flag': invalid verdicts are flagged, not removed"
    if mode == "remove":
        return True, ""
    fl = a.get("audit_flags") or {}
    if fl.get("gene_named") is not True:
        return False, ("the audit model did not confirm that the gene is named in the evidence it was "
                       "shown (gene_named=false): flagged, not removed")
    if fl.get("domain_subject") is True and fl.get("positive_finding") is True:
        return False, ("the audit model's own answers contradict 'invalid' (gene named, domain the subject, "
                       "positive finding): flagged, not removed")
    return True, ""


def _items_log(items, n=300):
    return [{"id": it.get("id"), "kind": it.get("kind"), "pmid": it.get("pmid"),
             "title": _clip(it.get("title"), 160), "text": _clip(it.get("text"), n)}
            for it in (items or [])]


_ROW_KEYS = ("audit_verdict", "audit_reason", "audit_pmid", "audit_version", "audit_n_items",
             "audit_flags", "audit_cached", "audit_error")


def run_audit(candidates, n_target, scored_doc, dspecs, endpoint, model, cache_path=None,
              limit=None, eg=None, fetch=None, definition=None, llm=None, log=print,
              mode=None, keep=None, budget_s=None, confirm_llm=None):
    """Walk `candidates` (emission order) and return (emitted_rows, summary).
    Audit fields are written ON the candidate rows (the scored rows):
    audit_verdict / _reason / _pmid / _version / _flags, audit_action
    ("kept" | "removed" | "flagged" | "not-audited") and, for the report,
    audit_flag (one of DISPLAY_FLAGS, only when a reader must see it)."""
    t0 = time.time()
    lim = audit_limit(limit)
    md = audit_mode(mode)
    kp = audit_keep(keep)
    budget = float(budget_s) if budget_s else audit_budget_s()
    if definition is None:
        try:
            from .. import literature as _l
            definition = _l.audit_definition
        except Exception:
            definition = (lambda k: "%s itself; NOT another cancer" % k)
    cpath = os.environ.get("MRAG_EVIDENCE_AUDIT_CACHE") or cache_path
    cache = _load_cache(cpath)
    closed = audit_fail_closed()
    emitted, removed, flagged, skipped_fail, walked = [], [], [], [], []
    counts = {v: 0 for v in AUDIT_VERDICTS + ("none", "unaudited")}
    n_llm = n_cached = n_err = n_seen = n_ok = n_confirm = 0
    consec_fail = 0
    breaker = ""
    i = 0
    stop = ""
    workers = _workers()

    def _mark(r, a, action, flag=None, note=""):
        for k in _ROW_KEYS:
            if k in a:
                r[k] = a[k]
        r["audit_action"] = action
        if flag:
            r["audit_flag"] = flag
        else:
            r.pop("audit_flag", None)
        if note:
            r["audit_note"] = note
        else:
            r.pop("audit_note", None)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        while len(emitted) < int(n_target) and i < len(candidates):
            if n_llm + n_cached >= lim:
                stop = "limit"
                break
            if not breaker and time.time() - t0 > budget:
                breaker = "time budget of %.0f s used up" % budget
            if breaker:
                # circuit open: the rest is emitted unaudited, no call
                r = candidates[i]
                i += 1
                a = {"audit_verdict": "unaudited", "audit_version": AUDIT_VERSION,
                     "audit_reason": "not audited: " + breaker, "audit_pmid": "", "audit_cached": False}
                counts["unaudited"] += 1
                _mark(r, a, "not-audited", "unaudited")
                walked.append((r, a, "emit"))
                emitted.append(r)
                continue
            need = int(n_target) - len(emitted)
            batch = []
            while (len(batch) < min(workers, need) and i < len(candidates)
                   and n_llm + n_cached + len(batch) < lim):
                batch.append(candidates[i])
                i += 1
            futs = [ex.submit(audit_one, r, scored_doc, dspecs, endpoint, model, cache, definition,
                              eg, fetch, None, llm) for r in batch]
            for r, f in zip(batch, futs):
                a = f.result()
                a["gene"] = r.get("gene")
                n_seen += 1
                v = a["audit_verdict"]
                counts[v] = counts.get(v, 0) + 1
                if v in AUDIT_VERDICTS:
                    consec_fail = 0
                    if a.get("audit_cached"):
                        n_cached += 1
                    else:
                        n_llm += 1
                        n_ok += 1
                if v == "unaudited":
                    n_err += 1
                    n_llm += 1
                    consec_fail += 1
                    if consec_fail >= AUDIT_BREAKER_FAILS and not breaker:
                        breaker = "%d consecutive audit calls failed" % consec_fail
                log("[curated] evidence audit #%d %s: %s%s -- %s"
                    % (n_seen, r.get("gene"), v, " (cached)" if a.get("audit_cached") else "",
                       str(a.get("audit_reason") or "")[:160]))
                rm, why_not = removal_decision(a, md, kp)
                if rm and md == "remove_named":
                    cf = confirm_removal(r, a, dspecs, endpoint, model, cache, definition, eg, llm, confirm_llm)
                    a["audit_confirm"] = cf
                    r["audit_confirm"] = {k: cf.get(k) for k in ("decision", "reason", "error")}
                    n_confirm += 1
                    log("[curated] evidence audit confirmation for %s: %s -- %s%s"
                        % (r.get("gene"), cf["decision"], cf.get("reason", "")[:140],
                           (" [" + cf["error"] + "]") if cf.get("error") else ""))
                    if cf["decision"] != "remove":
                        rm = False
                        why_not = ("the removal was not confirmed by the second audit call (%s): flagged, not "
                                   "removed" % (cf.get("error") or cf.get("reason") or "keep")[:160])
                if rm:
                    _mark(r, a, "removed", None)
                    removed.append({"gene": r.get("gene"), "reason": a.get("audit_reason", ""),
                                    "pmid": a.get("audit_pmid", ""), "score": r.get("score"),
                                    "flags": a.get("audit_flags") or {},
                                    "confirm": {k: (a.get("audit_confirm") or {}).get(k) for k in ("decision", "reason")},
                                    "items": _items_log(a.get("audit_items"))})
                    log("[curated] evidence audit REMOVED %s (score %s): %s [PMID %s]"
                        % (r.get("gene"), r.get("score"), str(a.get("audit_reason") or "")[:200],
                           a.get("audit_pmid") or "-"))
                    walked.append((r, a, "remove"))
                    continue
                if v == "unaudited" and closed and n_ok > 0 and not breaker:
                    skipped_fail.append(r.get("gene"))
                    _mark(r, a, "not-audited", "unaudited", "skipped: MRAG_EVIDENCE_AUDIT_FAIL=closed")
                    walked.append((r, a, "skip"))
                    continue
                if v == "invalid":
                    # deploy 2026-09-28 (decision 3): gene_named=false -> 'unverifiable'
                    _lab = invalid_label(a)
                    _mark(r, a, "flagged", _lab, why_not)
                    flagged.append({"gene": r.get("gene"), "verdict": v, "label": _lab,
                                    "reason": a.get("audit_reason", ""),
                                    "pmid": a.get("audit_pmid", ""), "why_not_removed": why_not,
                                    "flags": a.get("audit_flags") or {}})
                elif v == "weak":
                    _mark(r, a, "kept", "weak")
                    flagged.append({"gene": r.get("gene"), "verdict": v, "reason": a.get("audit_reason", ""),
                                    "pmid": a.get("audit_pmid", ""), "why_not_removed": ""})
                elif v == "unaudited":
                    _mark(r, a, "not-audited", "unaudited")
                else:
                    _mark(r, a, "kept", None)
                walked.append((r, a, "emit"))
                if len(emitted) < int(n_target):
                    emitted.append(r)
            _save_cache(cpath, cache)
    if not stop:
        stop = "target" if len(emitted) >= int(n_target) else "pool"
    # MALFUNCTION GUARD: too many removals -> remove nothing, keep the plain
    # emission order, flag the would-be removals "audit-suspect"
    guard = ""
    n_verd = sum(counts[v] for v in AUDIT_VERDICTS)
    if removed and (len(removed) > AUDIT_GUARD_MAX
                    or (n_verd >= AUDIT_GUARD_MIN_N and len(removed) > AUDIT_GUARD_FRAC * n_verd)):
        guard = ("%d of %d audited genes would have been removed (> %d, or > %d%% of >= %d): the audit "
                 "looks unreliable in this run, so nothing was removed and they are flagged "
                 "'audit-suspect'" % (len(removed), n_verd, AUDIT_GUARD_MAX, int(AUDIT_GUARD_FRAC * 100),
                                      AUDIT_GUARD_MIN_N))
        log("[curated] evidence audit GUARD: " + guard)
        emitted = []
        for r, a, what in walked:
            if what == "skip":
                continue
            if what == "remove":
                _mark(r, a, "flagged", "audit-suspect", guard)
                flagged.append({"gene": r.get("gene"), "verdict": "invalid", "reason": a.get("audit_reason", ""),
                                "pmid": a.get("audit_pmid", ""), "why_not_removed": "malfunction guard",
                                "flags": a.get("audit_flags") or {}})
            if len(emitted) < int(n_target):
                emitted.append(r)
        # a gene walked only to replace a removal is not emitted now: its
        # audit fields stay on the scored row
        removed_guarded, removed = removed, []
    else:
        removed_guarded = []
    replaced = []
    if removed:
        base = [r.get("gene") for r in candidates[:int(n_target)]]
        newg = [r.get("gene") for r in emitted]
        replaced = [{"removed": x["gene"], "added": a} for x, a in
                    zip(removed, [g for g in newg if g not in base])]
    summary = {"enabled": True, "ran": True, "version": AUDIT_VERSION, "model": model,
               "endpoint": endpoint, "mode": md, "keep": sorted(kp), "target": int(n_target),
               "limit": lim, "n_candidates": len(candidates), "n_examined": n_seen,
               "n_llm_calls": n_llm, "n_cached": n_cached, "n_errors": n_err, "verdicts": counts,
               "n_confirmations": n_confirm,
               "removed": removed, "flagged": flagged, "replaced": replaced,
               "invalid": ([{"gene": x["gene"], "reason": x["reason"], "pmid": x["pmid"],
                             "score": x["score"]} for x in removed]),
               "emitted": len(emitted), "stop": stop, "fail_mode": "closed" if closed else "open",
               "skipped_failed": skipped_fail, "circuit_breaker": breaker, "guard": guard,
               "guard_would_remove": [x["gene"] for x in removed_guarded],
               "budget_s": budget, "cache": str(cpath or ""), "seconds": round(time.time() - t0, 1)}
    notes = []
    if len(emitted) < int(n_target):
        notes.append("%d of %d targets emitted (%s)" % (
            len(emitted), int(n_target), "audit limit reached" if stop == "limit" else "candidate pool exhausted"))
    if breaker:
        notes.append("circuit breaker: " + breaker + "; the remaining genes were emitted unaudited")
    if guard:
        notes.append("malfunction guard: " + guard)
    if notes:
        summary["note"] = "; ".join(notes)
    return emitted, summary
