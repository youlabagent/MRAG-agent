"""Deterministic evidence-grounding gate for the CURATED TARGET scorer
(2026-09-23).

WHY.  The curated target scorer (nodes/curated_targets.py, step 6) shows the
LLM the evidence it retrieved for a gene -- GeneRIF lines + PaperQA vectordb
snippets -- and asks for a 0-10 domain-relevance score "using ONLY the provided
evidence".  Nothing checked that the answer was actually grounded in that
evidence.  Measured failure (ONECUT2 SCLC-only run,
2026-09-23): a gene scored 9 with a reason claiming the evidence tied it to
neuroendocrine biology, while all six retrieved vectordb snippets were about
other genes (n_vdb_mention = 0).

Section 14 Novelty already runs this class of check on its side
(viz_report_mcp/tools/narrative.py::_record_names_gene, 2026-09-20): before a
sentence may cite a PubMed record, that record's own text has to NAME the gene.
This module is the same gate for the score side, and deliberately mirrors that
function's matching rules so the two agree on what "names the gene" means:

  * the official symbol in its OWN casing (PAX6, IMPACT) -- an English word
    written as ordinary prose ("impact", "she", "set", "star") does not count;
  * ANY casing when the symbol contains a digit (Pax6, Tdrd1) -- a digit rules
    out an English word;
  * the official full name from NCBI Gene (paired box 6), case-insensitive,
    >= 5 chars;
  * word boundaries are non-alphanumeric, so CD8-positive / GRPR-targeted and
    "PAX6-WNK2 axis" count.

Differences from the novelty gate, both deliberate:
  * the full name is resolved OFFLINE from the local NCBI dump
    (Homo_sapiens.gene_info.gz) instead of an esummary call -- the score runner
    already does 249 GeneRIF fetches and must not add a second network round
    per gene.  No network calls at all happen in here.
  * GeneRIF text counts as naming evidence (it is gene-anchored literature that
    the scorer really did show the model).

Everything here is pure text over evidence the runner already has in memory.

THIRD SCOPE, "domain" (2026-09-23).  Naming the gene was never what the PI's
complaint was about.  The motivating gene scored 9 for SCLC and 19/20 of its
GeneRIF lines DO name it -- they are about eye and neural development, not lung
cancer.  So
under scope="domain" a piece of evidence only grounds a score when ONE AND THE
SAME passage names the gene AND belongs to the run's literature domain, judged
with the very tables Section 14 Novelty uses (literature.DOMAIN_TERMS /
DOMAIN_EXCLUDE through mentions_domain(), so NSCLC can never satisfy SCLC and
"head and neck" can never satisfy NEC).

"Belongs to the domain" has to be decided per evidence kind, because the three
kinds do not arrive with the same provenance:

  * a PaperQA snippet whose Qdrant payload `cancer` CONTAINS the domain key is
    domain-tied BY CORPUS, with no word test.  The retriever filtered on that
    payload, so the paper is indexed in that domain's corpus; what is kept in
    scored.json is a 220-character fragment of a full text, which very often
    does not repeat the disease name.  Re-demanding the words there would
    throw away the domain evidence the pipeline is most certain of.
  * a snippet from a corpus that is NOT the domain -- in practice the
    auto-appended pan-cancer corpus, 1122-1383 of the 1494 hits in the
    single-domain ONECUT2 runs -- must MENTION the domain in its own text.
    Pan-cancer membership only means "about cancer", which is not a tie to
    SCLC.
  * a GeneRIF line is not corpus-filtered at all, so it must MENTION the
    domain in its own text.  This is the half that decides the motivating case.

A run with several domains (the baseline: prostate + NEC + SCLC) is grounded
when ANY ONE of the user's domains is satisfied -- the same OR the Section 14
cards take over their numbered per-domain Novelty blocks.

`domains` may be corpus keys ("sclc") or the JSON-serialisable dicts
literature.domain_specs() returns.  Keys are resolved by importing
mrag_orchestrator.literature, but this module is ALSO loaded standalone, by
file path, inside the paper_qa venv (curated_targets._SCORE_RUNNER), where
that import can fail; there the node hands the specs over as plain data.
mentions_domain() below is the verbatim copy literature.mentions_domain's own
docstring asks for, so both gates agree on a domain even when the import does
not work.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# cap
# ---------------------------------------------------------------------------
# The scoring rubric (curated_targets.build_sys_prompt) ends with:
#     "If evidence says NO gene-specific evidence, set has_evidence=false and
#      score<=2."
# so 2 is the score the rubric itself allows without gene-specific literature.
# The gate therefore caps at 2 rather than zeroing: a gene may still carry
# motif/AG evidence, and 0 would be a claim the gate has not tested.
DEFAULT_CAP = 2.0

GENE_INFO_GZ = "/BIO1/MRAG/MD_check/Homo_sapiens.gene_info.gz"

_WB = r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])"
_FULLNAMES: dict | None = None


def _env_flag(name, default=True):
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() not in ("0", "false", "no", "off")


def gate_enabled(param=None):
    """Gate on unless explicitly switched off.  `param` (node/runner param)
    wins over the env var MRAG_TARGET_EVIDENCE_GATE; the safe behaviour (on)
    is the default of both."""
    if param is not None:
        return bool(param)
    return _env_flag("MRAG_TARGET_EVIDENCE_GATE", True)


SCOPES = ("all", "literature", "domain")


def gate_scope(param=None):
    """Which retrieved evidence counts as grounding the score.

      "all"        (DEFAULT, and what the PI asked for): GeneRIF lines OR
                   PaperQA snippets.  GeneRIF is gene-anchored literature that
                   the scorer really did put in front of the model.
      "literature" (stricter): only the PaperQA vectordb snippets count.
      "domain"     (2026-09-23): one passage must BOTH name the gene AND be
                   tied to the run's literature domain -- by corpus for a
                   snippet from that domain's own corpus, by the Section-14
                   domain word tables for a pan-cancer snippet or a GeneRIF
                   line.  This is the scope that answers "a gene scored 9 for
                   SCLC and nothing retrieved ties it to SCLC".  It needs
                   `domains`; with none resolvable the gate reports
                   "no-domain" and changes no score.

    WHY the knob exists.  Measured over the four ONECUT2 runs (offline replay,
    2026-09-23): under "all" the gate fires on 46/249 genes per run, but every
    one of those already scored <= 2, so nothing is capped -- GeneRIF is
    fetched per gene and almost always names it.  Under "literature" 18-21
    genes per run would be capped and the top-20 changes.  The SCLC case that
    motivated the gate sits exactly on this line: 0/6 snippets name the gene,
    but 19/20 of its GeneRIF lines do, including the very sentence the model's
    reason paraphrases.  So "all" is honest about gene naming; if the PI wants
    domain-matched LITERATURE behind a high score, "literature" is the switch.
    Param wins over MRAG_TARGET_EVIDENCE_SCOPE."""
    for v in (param, os.environ.get("MRAG_TARGET_EVIDENCE_SCOPE")):
        if v is None or str(v).strip() == "":
            continue
        v = str(v).strip().lower()
        if v in SCOPES:
            return v
    return "all"


def gate_cap(param=None):
    """Score ceiling for a gene no retrieved evidence names.  Param wins over
    MRAG_TARGET_EVIDENCE_CAP, default DEFAULT_CAP (= the rubric's <=2)."""
    for v in (param, os.environ.get("MRAG_TARGET_EVIDENCE_CAP")):
        if v is None or str(v).strip() == "":
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    return DEFAULT_CAP


# ---------------------------------------------------------------------------
# official full names, offline
# ---------------------------------------------------------------------------
def load_fullnames(path=None):
    """symbol -> official full name, from the local NCBI Homo_sapiens.gene_info
    dump.  Parsed once per process (~190k rows, < 1 s).  Never raises: a
    missing dump just means the symbol-only half of the gate runs."""
    global _FULLNAMES
    if _FULLNAMES is not None:
        return _FULLNAMES
    p = path or os.environ.get("MRAG_GENE_INFO") or GENE_INFO_GZ
    out = {}
    try:
        op = gzip.open if str(p).endswith(".gz") else open
        with op(p, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 12:
                    continue
                sym = f[2].strip()
                if not sym or sym == "-" or sym in out:
                    continue
                nm = f[11].strip()
                if not nm or nm == "-":
                    nm = f[8].strip()
                if nm and nm != "-":
                    out[sym] = nm
    except Exception:
        out = {}
    _FULLNAMES = out
    return out


def gene_fullname(gene, path=None):
    """Official full name of `gene`, or "".  Offline, no network."""
    return load_fullnames(path).get(str(gene or "").strip(), "")


# ---------------------------------------------------------------------------
# gene index: synonyms and every name of every gene, offline (2026-09-25)
# ---------------------------------------------------------------------------
# E1b ("symbol collision in text") and E3 ("shared full name") need to know
# which OTHER genes a string belongs to.  Both answers come from the same local
# Homo_sapiens.gene_info dump the full names come from -- no network, no hand
# list.  Measured cases this fixes:
#   * CGA: CHGA (chromogranin A) lists CGA as an alias, and corpus passages
#     write "chromogranin A (CGA)" -- the symbol hit is chromogranin A's;
#   * NRCAM: a PubMed record writes "NCAM (neuronal cell adhesion
#     molecule)" -- NRCAM's official full name, used there for NCAM1 (whose
#     alias is NCAM); NRCAM itself appears 0 times.
_GIDX: dict = {}
_NAMES_IX: dict = {}
_COLL_CACHE: dict = {}
_FULL_CACHE: dict = {}
COLLISION_WINDOW = 60          # characters either side of a symbol hit
# How another gene's OFFICIAL SYMBOL near a hit is treated: "paired" (default)
# rejects only a synonym-style pairing -- "NPRL3 (NPR3)", "CHGA/CGA", "CGA
# (CHGA)", "CHGA, also known as CGA"; "window" rejects any occurrence within
# COLLISION_WINDOW.  A plain co-mention ("LHFPL3 and LHFPL4", "CHGA, SYP and
# CGA") is the author telling the two genes apart, so "paired" keeps it.
COLLISION_SYMBOL_MODE = "paired"
# Round 3 (2026-09-26) -- the "/" in _SYN_SEP stays, on data.  Verifier
# finding: "UGT2B4/UGT2B11 expression" and "LHFPL3/LHFPL4 double knockout"
# (two family members listed with a slash) are refused for UGT2B11 / LHFPL4.
# True, but the proposed fix (no "/" pairing between two OFFICIAL symbols) was
# measured on all 19 parsed corpora (67,813 documents, round3/slash_pairs.py):
# every "A/B" where A and B are both official symbols and one lists the
# other as an NCBI synonym -- 65 occurrences, 25 pairs -- is a SYNONYM
# pairing in 64 (TCF7L2/TCF4 x29, HGF/SF x6, PTGS2/COX2 x3, DANCR/ANCR x3,
# PDPK1/PDK1, MST1/STK4, KDM5A/RBP2, SGPP2/SPP2, NKX2-1/TTF1, SOAT1/ACAT1,
# DFFB/CAD, ARHGAP21/ARHGAP10, ...) and a list of two genes in 1 (GLS/GLS2);
# the UGT2B4/UGT2B11 and LHFPL3/LHFPL4 forms occur 0 times.  Dropping the
# slash would make "TCF7L2/TCF4 target miRNAs" name TCF4 (a different gene)
# 29 times to rescue 1 list.  A same-stem ("family") exemption does not
# separate them either: the only same-stem occurrence in the corpora,
# "Rho GTPase activating protein 21 (ARHGAP21/ARHGAP10)", is a synonym.
# Only hits INSIDE the slash pair are refused; any other hit of the symbol in
# the passage still counts.
_SYN_SEP = r"\s*(?:[(\[/]|,?\s*(?:also\s+(?:known|called|termed)\s+as|a\.?k\.?a\.?|alias|or)\s+)\s*"


# VERIFICATION FIX V1 (2026-09-26, independent round-3 check): the one-sense-
# per-discourse rule (b') below turned the single numbered-family slash list
# of the corpora into a whole-passage refusal.  Real chunk (prostate corpus):
# "GLS/GLS2" (GLS2 lists GLS as a synonym) + "GLS inhibition
# reduced glutamine metabolism in prostate cancer cells" -- the bare GLS hits
# are glutaminase itself, and round 2 kept them; round 3 refused the passage.
# "SYM/SYM<n>" -- a symbol and the same symbol with 1-2 digits appended,
# joined by a bare slash -- enumerates numbered family members.  Only (b') is
# exempted: the hits INSIDE the pair are still refused (unchanged), and every
# other strong pairing still marks the passage.  Checked on the current
# corpora (77,397 unique documents, 846 chunks with a colliding symbol next
# to its owner's symbol): of the 16 chunks (b') changes, that GLS/GLS2 chunk is the
# only one it exempts; the other 15 (TCF7L2/TCF4 x11, MST1/STK4, NFE2L1/NRF1
# x2, MED1/MBD4) are real synonym pairings and stay refused.
def _family_slash(s):
    parts = [p.strip() for p in str(s or "").split("/")]
    if len(parts) != 2 or not all(parts) or parts[0].casefold() == parts[1].casefold():
        return False
    short, long_ = sorted(parts, key=len)
    return (long_.casefold().startswith(short.casefold())
            and re.fullmatch(r"\d{1,2}", long_[len(short):]) is not None)


def names_strict(param=None):
    """E1b/E3 on unless switched off (param wins over MRAG_NAMES_GENE_STRICT)."""
    if param is not None:
        return bool(param)
    return _env_flag("MRAG_NAMES_GENE_STRICT", True)


def _gi_path(path=None):
    return path or os.environ.get("MRAG_GENE_INFO") or GENE_INFO_GZ


def _gi_rows(path=None):
    p = _gi_path(path)
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) >= 12 and f[2].strip() and f[2].strip() != "-":
                yield f


def load_gene_index(path=None):
    """{"by_sym": {symbol: {"id", "name", "syn"}}, "syn_owners": {alias:
    [symbol]}} from the local dump; parsed once per path.  Never raises: a
    missing dump gives empty maps, and every E1b/E3 test then degrades to the
    pre-2026-09-25 behaviour (nothing extra is rejected)."""
    p = _gi_path(path)
    if p in _GIDX:
        return _GIDX[p]
    by_sym, owners = {}, {}
    try:
        for f in _gi_rows(p):
            sym = f[2].strip()
            if sym in by_sym:
                continue
            nm = f[11].strip()
            if not nm or nm == "-":
                nm = f[8].strip()
            syn = tuple(a.strip() for a in f[4].split("|") if a.strip() and a.strip() != "-")
            # round 2: description + other designations, for the own-naming
            # test of a DOMINATED symbol (symbol_sense) and the definition
            # classifier
            desig = tuple(x.strip() for x in ([f[8]] + (f[13].split("|") if len(f) > 13 else []))
                          if x.strip() and x.strip() != "-")
            by_sym[sym] = {"id": f[1].strip(), "name": "" if nm == "-" else nm, "syn": syn,
                           "desig": desig}
            for a in syn:
                owners.setdefault(a, []).append(sym)
    except Exception:
        by_sym, owners = {}, {}
    _GIDX[p] = {"by_sym": by_sym, "syn_owners": owners, "path": p}
    return _GIDX[p]


def _norm_tok(x):
    """Symbol spelling normalised for PAIRING tests only: case, hyphens, dots
    and spaces dropped (NrCAM == NRCAM, Pax-6 == PAX6)."""
    return re.sub(r"[\s\-\u2010-\u2015\u2212_.]", "", str(x or "")).casefold()


def _names_index(path=None):
    """Lazily built, only for the full-name route: every name any gene carries
    (official full name, description, other designations, synonyms), casefolded
    -> owner symbols, and every symbol/synonym spelling (_norm_tok) -> owners."""
    p = _gi_path(path)
    if p in _NAMES_IX:
        return _NAMES_IX[p]
    names, toks = {}, {}

    def _add(d, k, sym):
        v = d.get(k)
        if v is None:
            d[k] = sym
        elif isinstance(v, str):
            if v != sym:
                d[k] = (v, sym)
        elif sym not in v:
            d[k] = v + (sym,)
    try:
        for f in _gi_rows(p):
            sym = f[2].strip()
            _add(toks, _norm_tok(sym), sym)
            vals = [f[8], f[11]] + (f[13].split("|") if len(f) > 13 else [])
            syn = [a for a in f[4].split("|") if a.strip() and a.strip() != "-"]
            for a in syn:
                _add(toks, _norm_tok(a), sym)
            for v in vals + syn:
                v = re.sub(r"\s+", " ", str(v or "")).strip().casefold()
                if len(v) >= 3 and v != "-":
                    _add(names, v, sym)
    except Exception:
        names, toks = {}, {}
    _NAMES_IX[p] = {"names": names, "toks": toks}
    return _NAMES_IX[p]


def _owners(v):
    return () if v is None else ((v,) if isinstance(v, str) else tuple(v))


def colliding_genes(gene, path=None):
    """[(symbol, full name)] of the OTHER genes whose NCBI Synonyms list
    `gene` -- e.g. CGA -> [("CHGA", "chromogranin A")].  Offline."""
    g = str(gene or "").strip()
    ix = load_gene_index(path)
    owners = [o for o in ix["syn_owners"].get(g, ()) if o != g]
    # FIX round 2: a symbol that is no longer official and that exactly ONE
    # gene lists as a synonym is that gene's OLD symbol (FAM83D -> SACK1D,
    # TMEM30B -> CDC50B, the run's own gene under its new name) -- not a
    # collision with another gene.
    if g not in ix["by_sym"] and len(set(owners)) == 1:
        return []
    out = []
    for o in owners:
        out.append((o, (ix["by_sym"].get(o) or {}).get("name", "")))
    return out


def _collision_rx(gene, gene_name, path=None, ci=False):
    key = (gene, gene_name, _gi_path(path), bool(ci))
    if key in _COLL_CACHE:
        return _COLL_CACHE[key]
    others = colliding_genes(gene, path)
    rx = None
    if others:
        on = sorted({n for _, n in others if len(n) >= 5}, key=len, reverse=True)
        osy = sorted({o for o, _ in others}, key=len, reverse=True)
        sym = _pat(gene)
        # the symbol side of every pattern follows the caller's casing rule;
        # names are always matched case-insensitively
        sym_x = "(?i:%s)" % sym if ci else "(?:%s)" % sym
        alt_n = "|".join(_pat(n) for n in on)
        alt_s = "|".join(_pat(o) for o in osy)
        rx = {"others": others,
              "oname": re.compile(_WB % ("(?:%s)" % alt_n), re.I) if on else None,
              "osym": re.compile(_WB % ("(?:%s)" % alt_s)) if osy else None,
              # the other gene's symbol written AS A SYNONYM of this hit
              "osym_pair": (re.compile(
                  r"(?<![A-Za-z0-9])(?:%s)%s%s(?![A-Za-z0-9])|(?<![A-Za-z0-9])%s%s(?:%s)(?![A-Za-z0-9])"
                  % (alt_s, _SYN_SEP, sym_x, sym_x, _SYN_SEP, alt_s))
                  if osy else None),
              "own": (re.compile(_WB % _pat(gene_name), re.I)
                      if len(str(gene_name or "")) >= 5 else None),
              # "chromogranin A (CGA)" / "CGA (chromogranin A)"
              "def_other": (re.compile(
                  r"(?i:%s)\s*[(\[]\s*%s\s*[)\],;]|(?<![A-Za-z0-9])%s\s*[(\[]\s*(?i:%s)\s*[)\],;]"
                  % (alt_n, sym_x, sym_x, alt_n)) if on else None),
              "def_own": (re.compile(r"%s\s*[(\[]\s*%s\s*[)\],;]" % (_pat(gene_name), sym)
                                     + "|" + r"(?<![A-Za-z0-9])%s\s*[(\[]\s*%s\s*[)\],;]"
                                     % (sym, _pat(gene_name)), re.I)
                          if len(str(gene_name or "")) >= 5 else None)}
    _COLL_CACHE[key] = rx
    return rx


def symbol_collision(text, gene, gene_name="", spans=None, ci=False, path=None,
                     anchored=False):
    """E1b.  Which of the symbol's hits in `text` are really another gene's?

    Another gene X "claims" the symbol when X's NCBI Synonyms list it (CHGA
    lists CGA).  A hit is tied to X when
      * the passage DEFINES the symbol as X ("chromogranin A (CGA)" or
        "CGA (chromogranin A)") and never as this gene -- then every hit of the
        symbol in the passage is X's; or
      * X's official full name occurs within COLLISION_WINDOW characters of
        the hit, or X's official symbol is written as a synonym of it ("NPRL3
        (NPR3)", "CHGA/CGA"; COLLISION_SYMBOL_MODE), and this gene's own full
        name is not within the window.
    Returns (kept_spans, other) -- `other` names the claiming gene when NO hit
    survives, else "".  Offline, data-driven (gene_info Synonyms), no list.

    ROUND 2 (2026-09-26), see the block above symbol_definitions():
      (a) a symbol spelled in nucleotide letters is not counted inside a
          primer / codon context (any symbol, colliding or not);
      (b) a passage that DEFINES a colliding symbol as another gene or as a
          non-gene expansion, and never as this gene, names nothing;
      (c) a colliding symbol DOMINATED by another sense in the indexed
          literature (symbol_sense.json) needs the passage to name the gene
          itself -- skipped when `anchored` (a GeneRIF line, gene-anchored by
          NCBI);
      (d) with ci=True (the Section-14 card), a colliding symbol without a
          digit counts only in its official casing ("CgA" is not CGA)."""
    t = str(text or "")
    g = str(gene or "").strip()
    if spans is None:
        spans = [m.span() for m in re.finditer(_WB % _pat(g), t, re.I if ci else 0)]
    if not spans or not g:
        return list(spans or []), ""
    # (a) round 2: a primer / codon is not the gene
    if _NT_SYMBOL.match(g):
        sp2 = [(a, b) for a, b in spans if not _nucleotide_hit(t, a, b)]
        if not sp2:
            return [], "rejected: a nucleotide sequence (%s as a codon), not the gene" % g
        spans = sp2
    rx = _collision_rx(g, str(gene_name or ""), path, ci)
    if rx is None:
        return list(spans), ""
    # (d) round 2: case-insensitive caller, colliding digit-free symbol
    if ci and not re.search(r"\d", g):
        sp2 = [(a, b) for a, b in spans if t[a:b] == g]
        if not sp2:
            return [], "rejected: %s written as '%s' (another gene's abbreviation)" % (
                g, t[spans[0][0]:spans[0][1]])
        spans = sp2
    first = "%s (%s)" % rx["others"][0] if rx["others"][0][1] else rx["others"][0][0]
    if rx["def_other"] is not None and rx["def_other"].search(t) and not (
            rx["def_own"] is not None and rx["def_own"].search(t)):
        m = rx["def_other"].search(t)
        who = [o for o, n in rx["others"] if n and n.casefold() in m.group(0).casefold()]
        return [], ("%s (%s)" % (who[0], dict(rx["others"])[who[0]]) if who else first)
    # (b) round 2: the passage DEFINES the symbol as something else
    defs = symbol_definitions(t, g, spans, path)
    if defs and not any(k == "own" for k, _ in defs):
        k0, lf0 = defs[0]
        return [], ("rejected: %s defined in the passage as '%s' (%s)"
                    % (g, lf0[:60], ("gene " + k0[6:]) if k0.startswith("other:") else "not a gene"))
    W = COLLISION_WINDOW
    kept, hit_other = [], ""
    pair_sp = ([m.span() for m in rx["osym_pair"].finditer(t)]
               if (rx.get("osym_pair") is not None and COLLISION_SYMBOL_MODE != "window") else [])
    for a, b in spans:
        w = t[max(0, a - W):b + W]
        if COLLISION_SYMBOL_MODE == "window":
            on = (rx["oname"] is not None and rx["oname"].search(w)) or \
                 (rx["osym"] is not None and rx["osym"].search(w))
        else:
            on = (rx["oname"] is not None and rx["oname"].search(w))
            if not on:
                pm = [t[pa:pb] for pa, pb in pair_sp if pa <= a and b <= pb]
                on = re.match(r"(.+)", pm[0]) if pm else None
        if on and not (rx["own"] is not None and rx["own"].search(w)):
            hit_other = hit_other or on.group(0)
            continue
        kept.append((a, b))
    # (b') round 3: a passage that WRITES the symbol as another gene's synonym
    # ("Transcription factor 7 like 2 (TCF7L2, also known as TCF4)",
    # "TCF7L2/TCF4", "NPRL3 (NPR3)") uses it in that sense throughout (one
    # sense per discourse, as (b) does for definitions): unless the passage
    # also names this gene itself (own full name, designation, unambiguous
    # alias or own definition), its other bare hits are that gene's too.
    # Found while testing the slash rule: one chunk names TCF4 through 7
    # bare "TCF4" hits that all mean TCF7L2.  A plain "or" is not a pairing
    # strong enough for this.  Not applied to a GeneRIF line (anchored).
    if kept and pair_sp and not anchored:
        strong = [t[pa:pb] for pa, pb in pair_sp
                  if not re.search(r"\bor\b", t[pa:pb]) and not _family_slash(t[pa:pb])]
        if strong and not passage_names_own(t, g, gene_name, [], defs, path):
            return [], ("rejected: the passage writes %s as a synonym of another gene ('%s') and never "
                        "names %s itself" % (g, re.sub(r"\s+", " ", strong[0])[:60], g))
    # (c) round 2: the literature's dominant sense of the symbol is not this gene
    if kept and not anchored:
        se = symbol_sense(g)
        if se and se.get("dominated") and not passage_names_own(t, g, gene_name, kept, defs, path):
            top = sorted((se.get("other") or {}).items(), key=lambda x: -x[1])[:2]
            return [], ("rejected: %s is mostly defined as another gene in the indexed literature "
                        "(%s vs %d as %s); the passage does not name %s itself"
                        % (g, ", ".join("%s x%d" % kv for kv in top), int(se.get("own") or 0), g, g))
    return kept, ("" if kept else (hit_other and "%s" % hit_other) or first)


def generif_tied_to_other(text, gene, gene_name="", path=None):
    """E1b for a GeneRIF line (gene-anchored, so it usually does not repeat the
    symbol): the other gene -- e.g. "CHGA (chromogranin A)" -- when the line is
    about a gene that lists `gene` as an alias (that gene's official full name
    or official symbol appears) and the line does not itself name `gene`
    (collision-aware names_gene).  "" otherwise, and "" when switched off.
    Real case: the CGA record (Gene 1081) carries "Serum levels of chromogranin
    A are not predictive of ... prostate cancer" (PMID 23153859)."""
    if not names_strict():
        return ""
    t = str(text or "")
    g = str(gene or "").strip()
    others = colliding_genes(g, path)
    if not t or not others:
        return ""
    if names_gene(t, g, gene_name, path, anchored=True):
        return ""
    for o, nm in others:
        if (len(nm) >= 5 and re.search(_WB % _pat(nm), t, re.I)) or re.search(_WB % _pat(o), t):
            return "%s (%s)" % (o, nm) if nm else o
    return ""


def _longer_names(nm, path=None, gene=None):
    """OTHER genes' names that CONTAIN this full name (casefolded), e.g.
    'tudor domain containing 1' inside 'staphylococcal nuclease and tudor
    domain containing 1' (SND1).  A longer name carried only by `gene` itself
    ('calbindin 1, 28kDa' for CALB1) is not another gene's.  Cached."""
    key = ("longer", str(nm).casefold(), _gi_path(path), str(gene or ""))
    if key in _FULL_CACHE:
        return _FULL_CACHE[key]
    n = str(nm).casefold()
    ix = _names_index(path)["names"]
    out = sorted({k for k in ix if n in k and k != n
                  and any(o != gene for o in _owners(ix.get(k)))}, key=len, reverse=True)[:200]
    _FULL_CACHE[key] = out
    return out


def unambiguous_aliases(gene, path=None):
    """This gene's NCBI synonyms that no other gene uses as symbol or synonym
    (>= 3 characters).  Offline."""
    g = str(gene or "").strip()
    ix = load_gene_index(path)
    rec = ix["by_sym"].get(g) or {}
    out = []
    for a in rec.get("syn", ()):
        if len(a) < 3 or a in ix["by_sym"]:
            continue
        if any(o != g for o in ix["syn_owners"].get(a, ())):
            continue
        out.append(a)
    return out


def fullname_unique(gene, gene_name, path=None):
    """True when no OTHER gene carries this exact name (full name, description,
    other designation or synonym, casefolded)."""
    nm = re.sub(r"\s+", " ", str(gene_name or "")).strip().casefold()
    if not nm:
        return False
    ow = _owners(_names_index(path)["names"].get(nm))
    return all(o == gene for o in ow)


_PAIR_AFTER = re.compile(r"\s*[(\[]\s*([A-Za-z0-9][A-Za-z0-9\-\u2010/.]{0,15}?)\s*[)\],;]")
_PAIR_BEFORE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z0-9][A-Za-z0-9\-\u2010.]{0,15})\s*[(\[]\s*$")


# Round 2 (2026-09-26), E3 re-check: "neuronal cell adhesion molecule" is
# NRCAM's official full name and no other gene carries it, so the "unique full
# name" fallback accepted it -- but the literature uses the phrase for NCAM1
# ("neural cell adhesion molecule") and L1CAM as well: a live record
# ("... and neuronal cell adhesion molecule appear to be absent in normal
# Merkel cells", NrCAM 0 times) verified NRCAM, and "L1 neuronal cell
# adhesion molecule (L1CAM/CD171)" in two records passed because the
# pairing token was not split on "/".  Three changes:
#   * pairing tokens are split on "/" "," ";" and stripped of a modifier
#     prefix (PSA-NCAM -> NCAM, p-/c-/pro-/phospho-/pan-/anti-), and a comma
#     apposition "NCAM, neuronal cell adhesion molecule," is a pairing too;
#   * a full name WITHOUT A DESIGNATOR (no digit, single letter, Greek letter
#     or roman numeral) that is NEAR-IDENTICAL to another gene's official name
#     -- same core words once designators are dropped (CADPS / CADPS2), or one
#     word swapped for a morphological variant with the same stem once an
#     adjectival / plural ending is dropped ("neuronal" / "neural" -> "neur":
#     NRCAM / NCAM1) -- is a generic phrase: it counts only together with the
#     symbol or an unambiguous alias.
#     Data-driven from gene_info; no list.  "glypican 5", "neuropeptide Y",
#     "metallothionein 1E" carry designators; "brain derived neurotrophic
#     factor" differs from NDNF's name by an unrelated word -- all unaffected.
#     Not applied to a GeneRIF line (anchored to its gene by NCBI).
_VARIANT_END = re.compile(r"(?:ional|onal|al|ic|ous|ary|ar|s)$")


def _variant_words(x, y):
    """'neuronal' ~ 'neural' (same stem of >= 4 letters after dropping an
    adjectival / plural ending)."""
    sx, sy = _VARIANT_END.sub("", x), _VARIANT_END.sub("", y)
    return x != y and sx == sy and len(sx) >= 4


_PAIR_APPOS = re.compile(r"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9\-.]{1,15}),\s*$")
# ROUND 3 (2026-09-26): an apposition with a naming CONNECTOR is a pairing in
# either order -- "full name, termed ETV5," / "ETV5, i.e. full name," /
# "NCAM, also called full name".  Verifier finding, re-checked:
# names_gene_detail("ETS variant transcription factor 4, termed ETV5, was
# high.", "ETV4") gave ('full-name', 'unique full name').  Only a SYMBOL-CASED
# token (>= 2 capitals, or a capital and a digit) counts, so "called the ..."
# is never read as a gene.  Measured on real data (8 runs' GeneRIF lines and
# stored hits, 149 rescue records): 0 of the 383 unique-full-name acceptances
# carry such a connector, so nothing real changes.  NOT adopted: treating ANY
# other gene's symbol within ~40 characters of an unpaired full name as a
# pairing -- on the same data it would have refused 66 of 379 genuine GeneRIF
# lines ("versican V1", "gasdermin D ... IL-1beta", "dihydropyrimidine
# dehydrogenase and ERCC1") and 2 of the 4 unique-full-name rescue records,
# both genuine ("versican-TLR2-MyD88-RelB axis", "metallothionein 1E and ...
# RhoE").
_PAIR_CONN = (r"(?:also\s+(?:known|called|termed|referred\s+to|designated)\s+as|also\s+(?:called|termed|named)"
              r"|(?:known|referred\s+to|designated)\s+as|termed|called|named|designated"
              r"|a\.?k\.?a\.?|alias|i\.\s?e\.?,?)")
_PAIR_CONN_AFTER = re.compile(r"\s*,?\s*[(\[]?\s*" + _PAIR_CONN
                              + r"\s+([A-Za-z0-9][A-Za-z0-9\-\u2010/.]{0,15}?)(?=[\s,;:)\]]|\.(?:\s|$)|$)", re.I)
_PAIR_CONN_BEFORE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z0-9][A-Za-z0-9\-\u2010/.]{0,15})\s*,?\s*[(\[]?\s*"
                               + _PAIR_CONN + r"\s*,?\s*(?:the\s+)?$", re.I)


def _symbol_cased(tok):
    t = str(tok or "")
    return sum(ch.isupper() for ch in t) >= 2 or (any(ch.isupper() for ch in t)
                                                  and any(ch.isdigit() for ch in t))
_PAIR_PREFIX = re.compile(r"^(?:PSA|p|c|pro|phospho|pan|anti)-(?=[A-Za-z0-9])", re.I)
_GREEK = frozenset("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
                   "omicron pi rho sigma tau upsilon phi chi psi omega".split())
_ROMAN = frozenset("i ii iii iv v vi vii viii ix x xi xii xiii xiv xv".split())
_NAME_STOP = frozenset("of and the a an in to for with by on or".split())


def _pair_subtokens(tok):
    """'L1CAM/CD171' -> ['L1CAM', 'CD171']; 'PSA-NCAM' -> ['PSA-NCAM', 'NCAM']."""
    out = []
    for x in re.split(r"[/,;]", str(tok or "")):
        x = x.strip()
        if not x:
            continue
        out.append(x)
        y = _PAIR_PREFIX.sub("", x)
        if y and y != x:
            out.append(y)
    return out


def _is_designator(w):
    return bool(re.search(r"\d", w)) or len(w) == 1 or w in _GREEK or w in _ROMAN


def _name_core(nm):
    return frozenset(w for w in _norm_name(nm).split()
                     if w not in _NAME_STOP and not _is_designator(w))


def confusable_genes(gene, gene_name, path=None):
    """Other genes whose OFFICIAL full names read like this gene's full name
    when that name carries no designator (see above).  [] for a name with a
    designator or fewer than two core words.  Cached; offline."""
    key = ("conf", str(gene), str(gene_name or "").casefold(), _gi_path(path))
    if key in _FULL_CACHE:
        return _FULL_CACHE[key]
    out = []
    try:
        toks = _norm_name(gene_name).split()
        core = _name_core(gene_name)
        if toks and not any(_is_designator(w) for w in toks) and len(core) >= 2:
            for sym, rec in load_gene_index(path)["by_sym"].items():
                if sym == gene:
                    continue
                n2 = str(rec.get("name") or "").casefold()
                if not n2 or not any(w in n2 for w in core):
                    continue
                c2 = _name_core(n2)
                if c2 == core:
                    out.append(sym)
                    continue
                a_, b_ = core - c2, c2 - core
                if len(a_) == 1 and len(b_) == 1 and _variant_words(next(iter(a_)), next(iter(b_))):
                    out.append(sym)
    except Exception:
        out = []
    _FULL_CACHE[key] = sorted(out)
    return _FULL_CACHE[key]


def fullname_decision(text, gene, gene_name, path=None, anchored=False):
    """E3.  Does an occurrence of the official full name really name THIS gene?
    Returns (how, note): how is "full-name" or "".

      * an occurrence inside a LONGER name of another gene does not count;
      * an occurrence paired with this gene's own symbol / alias in either
        order ("neuronal cell adhesion molecule (NrCAM)") counts at once;
      * an occurrence paired with ANOTHER gene's symbol ("NCAM (neuronal cell
        adhesion molecule)", NCAM = NCAM1) does not count;
      * an unpaired occurrence counts only when this gene's symbol (any casing
        with >= 2 capitals) or an unambiguous alias also appears, or when the
        full name is unique to this gene AND (round 2) is not a generic phrase
        confusable with other genes' names (confusable_genes);
      * (round 2) pairing tokens are split on "/" and stripped of prefixes
        (PSA-NCAM, CD56/NCAM, L1CAM/CD171), and "SYM, full name," counts as a
        pairing.
    Offline, data-driven.  Never raises."""
    t = str(text or "")
    g = str(gene or "").strip()
    nm = str(gene_name or "").strip()
    if len(nm) < 5 or not t:
        return "", ""
    occ = [m.span() for m in re.finditer(_WB % _pat(nm), t, re.I)]
    if not occ:
        return "", ""
    try:
        longer = _longer_names(nm, path, g)
        lrx = [re.compile(_WB % _pat(x), re.I) for x in longer[:60]]
        lsp = [m.span() for r in lrx for m in r.finditer(t)] if lrx else []
        toks = _names_index(path)["toks"]
        rec = load_gene_index(path)["by_sym"].get(g) or {}
        own = {_norm_tok(g)} | {_norm_tok(a) for a in rec.get("syn", ())}
    except Exception:
        return "full-name", "index unavailable"
    unpaired, other = 0, []
    for a, b in occ:
        if any(la <= a and b <= lb and (la, lb) != (a, b) for la, lb in lsp):
            other.append("longer name")
            continue
        # "TOKEN[official full name]": the NCBI GeneRIF curator annotation that
        # TOKEN is this gene ("BAP[alkaline phosphatase, biomineralization
        # associated]" in an ALPL GeneRIF) -- a confirmation, not a pairing
        # with another gene.
        if a >= 2 and t[a - 1] == "[" and t[a - 2].isalnum() and t[b:b + 1] == "]":
            return "full-name", "NCBI-annotated full name"
        pt = []
        ma = _PAIR_AFTER.match(t, b)
        if ma:
            pt.append(ma.group(1))
        mb = _PAIR_BEFORE.search(t[:a])
        if mb and re.match(r"\s*[)\],;]", t[b:]):
            pt.append(mb.group(1))
        # round 2: comma apposition "NCAM, neuronal cell adhesion molecule,"
        mc = _PAIR_APPOS.search(t[max(0, a - 40):a])
        if mc and re.match(r"\s*[,;)]", t[b:]):
            pt.append(mc.group(1))
        # round 3: apposition with a naming connector, either order
        mk = _PAIR_CONN_AFTER.match(t, b)
        if mk and _symbol_cased(mk.group(1).rstrip(".")):
            pt.append(mk.group(1).rstrip("."))
        mk = _PAIR_CONN_BEFORE.search(t[max(0, a - 60):a])
        if mk and _symbol_cased(mk.group(1)):
            pt.append(mk.group(1))
        if not pt:
            unpaired += 1
            continue
        kind = ""
        # round 2: "L1CAM/CD171", "CD56/NCAM", "PSA-NCAM" are split / stripped
        for tok in [x for p_ in pt for x in _pair_subtokens(p_)]:
            n = _norm_tok(tok)
            if n in own:
                kind = "own"
                break
            if any(o != g for o in _owners(toks.get(n))):
                kind = kind or tok
        if kind == "own":
            return "full-name", "full name paired with its own symbol"
        if kind:
            other.append(kind)
        else:
            unpaired += 1
    if not unpaired:
        return "", "full name paired with another gene's symbol (%s)" % ", ".join(other[:3])
    for m in re.finditer(_WB % _pat(g), t, re.I):
        if sum(1 for ch in m.group(0) if ch.isupper()) >= 2:
            return "full-name", "full name + symbol %s" % m.group(0)
    for al in unambiguous_aliases(g, path):
        if re.search(_WB % _pat(al), t, re.I if re.search(r"\d", al) else 0):
            return "full-name", "full name + alias %s" % al
    if fullname_unique(g, nm, path):
        conf = [] if anchored else confusable_genes(g, nm, path)       # round 2
        if conf:
            return "", ("full name '%s' has no designator and reads like other genes' names "
                        "(%s); neither %s nor an unambiguous alias appears"
                        % (nm, ", ".join(conf[:4]) + (" ..." if len(conf) > 4 else ""), g))
        return "full-name", "unique full name"
    return "", "full name is not unique to %s and no symbol/alias appears" % g




# ---------------------------------------------------------------------------
# ROUND 2 (2026-09-26): symbol SENSE -- nucleotide words, in-passage
# definitions, and the literature's dominant sense of a colliding symbol
# ---------------------------------------------------------------------------
# Verifier finding (2026-09-25, re-checked): symbol_collision() only rejected a
# CGA hit when the passage itself tied CGA to CHGA.  Scanning the prostate + nec
# + sclc corpora, 43 chunks still "named" CGA after round 1 and NONE of them
# meant the glycoprotein hormone alpha subunit: 29 were primer / gRNA sequences
# ("TTT CGA TG", "5'-CGA CGC ..."), the other 14 chromogranin A in
# neuroendocrine marker lists ("CD56, INSM1, CGA and SYP", "ENO and CGA") or
# "comprehensive geriatric assessment"; stored run hits add "The Cancer Genome
# Atlas (CGA)" and "cardia gastric adenocarcinoma (CGA)".  After round 2: 0 of
# those 43 (and 0 of 79 on the case-insensitive card path).  Three general,
# data-driven rules close this:
#   (a) NUCLEOTIDE WORDS.  A symbol spelled only in nucleotide letters (CGA,
#       GAA, TAT, CAT, ...) does not name the gene where it sits in a run of
#       nucleotide tokens, next to a 5'/3' end mark, in a codon change
#       ("CGA>TGA", "CGA to TGA") or before "codon".  Structural; no list.
#   (b) IN-PASSAGE DEFINITIONS (colliding symbols only).  A passage that
#       defines the symbol -- "LONG FORM (SYM)" or "SYM (LONG FORM)", the long
#       form found as another gene's name or, failing that, by the
#       Schwartz-Hearst abbreviation algorithm -- as something that is NOT this
#       gene ("The Cancer Genome Atlas (CGA)") uses the symbol in that sense
#       throughout (one sense per discourse); unless the same passage also
#       defines it as this gene, none of its hits count.
#   (c) LITERATURE SENSE (colliding symbols only).  symbol_sense.json counts,
#       over the indexed corpora (one count per document and sense), how often
#       each colliding official symbol is DEFINED as its own gene vs as
#       ANOTHER gene (same classifier as (b); build_symbol_sense(); built
#       2026-09-26 over 65,943 unique documents).  A symbol defined as another
#       gene in >= SENSE_MIN_NOT_OWN documents and >= SENSE_RATIO x as often as
#       as itself is DOMINATED -- 8 of 1052 colliding symbols: CGA (chromogranin
#       A 7 vs 0), TTF1 (NKX2-1 62 vs 1), COX2 (PTGS2 27 vs 0), RBP2 (KDM5A 8
#       vs 1), PKD1 (PRKD1 5 vs 0), TF, C6, SF.  A bare hit of a dominated
#       symbol then needs the passage to name the gene itself
#       (official full name, an own designation, an unambiguous alias, an own
#       definition, or a specific word of its own names next to the hit).
#       GeneRIF lines are gene-anchored by NCBI and are exempt from (c).
# Only symbols another gene lists as a synonym are touched by (b)/(c): 9 of
# the 357 symbols of the 8 audited runs (7 once renamed symbols are not
# counted as collisions), 1052 of 193,733 genome-wide.  Rebuild the table
# with build_symbol_sense() when the corpora grow; a missing table disables
# (c) only.  Everything here is switched off with the
# rest of E1b by MRAG_NAMES_GENE_STRICT=0.
SYMBOL_SENSE_JSON = "/BIO1/MRAG/Agent/orchestrator/configs/symbol_sense.json"
SENSE_MIN_NOT_OWN = 5      # at least this many documents defining it as another gene ...
SENSE_RATIO = 4            # ... and at least 4x as many as define it as this gene
_SENSE: dict = {}
_NT_SYMBOL = re.compile(r"^[ACGTU]{3,}$")
_NT_SEP = "[\\s\\u00a0\\u200b-\\u200d\\u2060\\ufeff]"
_NT_ARROW = r"(?:>|→|->|➔|\bto\b|\binto\b)"
_NT_LEFT = re.compile(
    r"(?:(?<![A-Za-z0-9])[ACGTU]{3,}" + _NT_SEP + r"+"
    r"|[35]\s*['′’`´]\s*[-‐-―−]?\s*"
    r"|(?<![A-Za-z0-9])[ACGTU]{3}\s*" + _NT_ARROW + r"\s*)$")
_NT_RIGHT = re.compile(
    r"(?:" + _NT_SEP + r"+[ACGTU]{3,}(?![A-Za-z0-9])"
    r"|\s*[-‐-―−]?\s*[35]\s*['′’`´]"
    r"|\s*" + _NT_ARROW + r"\s*[ACGTU]{3}(?![A-Za-z0-9])"
    r"|[\s\-]*(?i:anti)?(?i:codons?|triplets?)\b)")
# words too common in gene names to say WHICH gene a phrase means
_GENERIC_WORDS = frozenset((
    "protein proteins factor factors receptor receptors gene genes family member "
    "subunit subunits alpha beta gamma delta epsilon kappa chain chains domain "
    "domains containing like type cell cells human kinase polypeptide complex "
    "associated related homolog homologue antigen molecule activity binding "
    "regulator regulatory hormone hormones subfamily group class isoform "
    "precursor fragment peptide transcription transcript variant enzyme "
    "channel transporter carrier cancer tumor tumour growth expressed "
    "specific interacting").split())


def _norm_name(x):
    """Name spelling for the DEFINITION classifier: casefolded, dashes /
    slashes / underscores as spaces, other punctuation dropped, leading
    article dropped."""
    import unicodedata
    x = unicodedata.normalize("NFKD", str(x or ""))
    x = "".join(ch for ch in x if not unicodedata.combining(ch)).casefold()
    x = re.sub(r"[‐-―−\-_/]", " ", x)
    x = re.sub(r"[^\w\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return re.sub(r"^(?:the|a|an) ", "", x)


def _nucleotide_hit(t, a, b):
    """(a): is the hit t[a:b] a nucleotide word (primer, codon), not a gene?"""
    return bool(_NT_LEFT.search(t[max(0, a - 16):a]) or _NT_RIGHT.match(t, b))


def _own_names(gene, path=None):
    """{"names": normalised own names, "toks": own symbol/alias tokens,
    "words": specific content words of the own names}.  Cached."""
    key = ("own", str(gene), _gi_path(path))
    if key in _FULL_CACHE:
        return _FULL_CACHE[key]
    rec = load_gene_index(path)["by_sym"].get(gene) or {}
    raw = [rec.get("name", "")] + list(rec.get("desig", ()))
    names = {_norm_name(x) for x in raw if len(_norm_name(x)) >= 3}
    toks = {_norm_tok(gene)} | {_norm_tok(a) for a in rec.get("syn", ())}
    words = {w for n in names for w in n.split()
             if len(w) >= 4 and w not in _GENERIC_WORDS and not w.isdigit()}
    out = {"names": names, "toks": toks, "words": words,
           "raw": [x for x in raw if len(str(x)) >= 8]}
    _FULL_CACHE[key] = out
    return out


def _name_owners(s, path=None):
    """Genes that carry the phrase `s` as a name / designation / synonym, or
    the token `s` as a symbol / synonym (a few spelling variants tried)."""
    ix = _names_index(path)
    c = re.sub(r"\s+", " ", str(s or "")).strip().casefold()
    out = []
    # a short single word ("Net", "cell", "Yes") is looked up as a SYMBOL
    # below (casing rule), never as a name: gene_info lists such words as
    # synonyms of some gene
    for v in ({c, re.sub(r"[‐-―−\-]", " ", c), re.sub(r"\s", "-", c),
              re.sub(r"^(?:the|a|an) ", "", c)} if (" " in c or len(c) >= 6) else ()):
        out += list(_owners(ix["names"].get(v)))
    # another gene's SYMBOL only in symbol casing (>= 2 capitals, or a capital
    # and a digit): "Kit", "Net", "set" are words, not KIT / ELK3 / SET
    s0 = str(s or "").strip()
    if re.fullmatch(r"[\w\-\u2010.]{2,15}", s0) and (
            sum(ch.isupper() for ch in s0) >= 2
            or (any(ch.isupper() for ch in s0) and any(ch.isdigit() for ch in s0))):
        out += list(_owners(ix["toks"].get(_norm_tok(s0))))
    seen, res = set(), []
    for o in out:
        if o not in seen:
            seen.add(o)
            res.append(o)
    return res


def _sh_long_form(short, long):
    """Schwartz & Hearst (2003) best long form of `short` inside `long`
    (right-to-left character match, first character at a word start), or
    None."""
    s, l = len(short) - 1, len(long) - 1
    while s >= 0:
        c = short[s].lower()
        if not c.isalnum():
            s -= 1
            continue
        while (l >= 0 and long[l].lower() != c) or (s == 0 and l > 0 and long[l - 1].isalnum()):
            l -= 1
        if l < 0:
            return None
        l -= 1
        s -= 1
    l = long.rfind(" ", 0, l + 1) + 1
    lf = long[l:].strip()
    return lf if len(lf) > len(short) else None


def _classify_exact(gene, cand, path=None):
    """'own' | 'other:<SYM>' | '' for a candidate long form that IS a gene
    name or symbol."""
    own = _own_names(gene, path)
    n = _norm_name(cand)
    if not n:
        return ""
    if n in own["names"] or (re.fullmatch(r"[\w\-‐.]{2,15}", cand.strip())
                             and _norm_tok(cand) in own["toks"]):
        return "own"
    ow = [o for o in _name_owners(cand, path) if o != gene]
    return ("other:" + ow[0]) if ow else ""


def _classify_lenient(gene, lf, path=None):
    """A Schwartz-Hearst long form that is no gene's exact name: 'own' when it
    shares a specific word with this gene's own names, else 'nongene'."""
    return "own" if set(_norm_name(lf).split()) & _own_names(gene, path)["words"] else "nongene"


def symbol_definitions(text, gene, spans=None, path=None):
    """(b): every place `text` DEFINES the symbol: [(kind, long_form)] with
    kind 'own' | 'other:<SYM>' | 'nongene'.  Forms: "LONG (SYM)" and "SYM
    (LONG)".  The long form is first looked up as a gene name (longest
    trailing word run, <= 10 words), then found by Schwartz-Hearst (<=
    min(len+5, 2*len) words).  Never raises."""
    t = str(text or "")
    g = str(gene or "").strip()
    out = []
    try:
        if spans is None:
            spans = [m.span() for m in re.finditer(_WB % _pat(g), t)]
        for a, b in spans:
            # "LONG FORM (SYM)"
            mb = re.search(r"[(\[]\s*$", t[max(0, a - 3):a])
            if mb and re.match(r"\s*[)\];,]", t[b:b + 3]):
                p = max(0, a - 3) + mb.start()
                clause = re.split(r"[.;:!?()\[\]]", t[max(0, p - 240):p])[-1]
                words = clause.split()
                kind = lf = ""
                for k in range(min(len(words), 10), 0, -1):
                    cand = " ".join(words[-k:])
                    kind = _classify_exact(g, cand, path)
                    if kind:
                        lf = cand
                        break
                nmax = min(len(g) + 5, 2 * len(g))
                sh = _sh_long_form(g, " ".join(words[-nmax:])) if words else None
                if not kind and sh:
                    lf, kind = sh, _classify_lenient(g, sh, path)
                elif (kind.startswith("other:") and sh and len(sh.split()) > len(lf.split())
                      and _classify_lenient(g, sh, path) == "own"):
                    # "membrane associated guanylate kinase inverted 1 (MAGI1)":
                    # the trailing "guanylate kinase 1" is GUK1's name, but the
                    # whole abbreviation-explaining phrase is this gene's
                    lf, kind = sh, "own"
                if kind:
                    out.append((kind, lf))
            # "SYM (LONG FORM)"
            ma = re.match(r"\s*[(\[]\s*([^()\[\]]{2,100}?)\s*[)\]]", t[b:b + 112])
            if ma:
                seg = re.split(r"[;,]", ma.group(1))[0].strip()
                kind = _classify_exact(g, seg, path)
                if not kind:
                    lf = _sh_long_form(g, seg) or ""
                    if lf and _norm_name(lf) == _norm_name(seg):
                        kind = _classify_lenient(g, lf, path)
                if kind:
                    out.append((kind, seg))
    except Exception:
        return out
    return out


def _sense_path(path=None):
    return path or os.environ.get("MRAG_SYMBOL_SENSE") or SYMBOL_SENSE_JSON


def symbol_sense(gene, path=None):
    """(c): the corpus definition counts for `gene` from symbol_sense.json:
    {"own", "other": {SYM: n}, "nongene", "examples", "dominated"} or None
    (not a colliding symbol, never defined, or no table).  Never raises."""
    p = _sense_path(path)
    if p not in _SENSE:
        try:
            _SENSE[p] = json.load(open(p)).get("symbols") or {}
        except Exception:
            _SENSE[p] = {}
    e = _SENSE[p].get(str(gene or "").strip())
    if not e:
        return None
    e = dict(e)
    # Dominance counts OTHER-GENE definitions only (E1b is about a symbol that
    # belongs to another gene); non-gene expansions still act per passage (b).
    n_oth = int(sum((e.get("other") or {}).values()))
    e["n_not_own"] = n_oth
    e["dominated"] = bool(n_oth >= SENSE_MIN_NOT_OWN and n_oth >= SENSE_RATIO * int(e.get("own") or 0))
    return e


# ROUND 3 (2026-09-26): which own-name evidence NEXT TO a bare hit may count
# for a DOMINATED symbol.  Verifier finding, re-checked: round 2 accepted ANY
# single "specific word" of the gene's names within COLLISION_WINDOW, and
# _GENERIC_WORDS did not cover ordinary words -- a real stored snippet
# ("... a common research outcome in CGA", CGA = cardia gastric
# adenocarcinoma) named CGA through "common" (from "... hormones COMMON subunit
# alpha"); constructed: "CGA is the most common circulating marker of
# neuroendocrine neoplasms" (a spurious NEC tie), "PKD1 is a potential
# therapeutic target" (transient receptor POTENTIAL), "COX2, encoded by the
# PTGS2 gene" (mitochondrially ENCODED ...), "TTF1 ... polymerase".  Now:
#   * single words no longer count.  The evidence is a FRAGMENT of one own
#     name -- 2 to 4 consecutive words, neither end a stop word or a
#     designator (digit, single letter, Greek letter, roman numeral), at least
#     one content word (>= 4 letters, not _GENERIC_WORDS) -- e.g.
#     "glycoprotein hormones", "chorionic gonadotropin", "polycystic kidney";
#   * the fragment must be RARE among gene names: its rarest word occurs in
#     the names / designations of <= OWN_FRAG_MAX_DF genes genome-wide (an
#     upper bound on the fragment's own frequency; gene_info, 193,733 genes):
#     "rna polymerase" (polymerase 297 genes), "cytochrome c oxidase"
#     (cytochrome 540), "kidney disease" (kidney 77), "receptor potential"
#     (potential 58) are dropped; "glycoprotein hormones" (hormones), "anterior
#     pituitary", "transcription termination" (termination 14) stay;
#   * a fragment that also occurs in a RIVAL gene's names (the genes that list
#     the symbol as a synonym, and those the literature defines it as --
#     symbol_sense "other") says nothing about which gene is meant;
#   * besides fragments, ONE single word still counts: the LEAD word of the
#     official full name -- its first content word ("complement" C6,
#     "polycystin" PKD1, "glycoprotein" CGA, "retinol" RBP2, "termination"
#     TTF1, "mitochondrially" COX2), matched with a light stem ("mitochondrial"
#     ~ "mitochondrially") -- unless a rival gene's names use it too.  Modifier
#     words further along ("potential", "component", "common", "encoded",
#     "cellular") no longer count;
#   * the window route does not count when a rival's official symbol, official
#     full name or a designation of >= 8 characters sits in the same window
#     ("COX2, encoded by the PTGS2 gene").
#   * an own NCBI synonym that no rival uses (rival_free_aliases: CGA's "HCG",
#     shared with CGB5 / HTC2 but never with CHGA) counts in the window too,
#     in its own casing or any symbol casing ("hCG") -- "CGA, the common
#     alpha subunit of hCG".
# Measured (all 19 parsed corpora, every chunk naming one of the dominated
# symbols CGA, PKD1, COX2, TTF1, RBP2, C6 case-sensitively; TF and SF on the
# nec / sclc / prostate corpora; round3/scan_dom_*.json): round 2's any-word
# rule kept 38 chunks.  A fragments-only rule removed 28: 12 spurious (PKD1
# x5 = protein kinase D1 via "potential", C6 x5 = "C6 glioma" / MSigDB "C6"
# / "M6/C6" via "component", RBP2 = KDM5A via "cellular", COX2 = celecoxib)
# and 16 genuine (C6 x10 "complement C1q or C6", "complement components (C6
# and C7)"; COX2 x6 mitochondrial).  With the lead word 18 are removed: the
# same 12 spurious + the 6 mitochondrial COX2 chunks (COX2 has no lead word:
# "mitochondrially" is also in the name of a rival, the pseudogene
# MTCO2P12).  The 20 chunks still kept all name their gene (read one by one).
# Real data (19 parsed corpora, every chunk naming CGA case-sensitively): the
# one chunk round 2 kept, "CGA encodes the alpha subunit of
# glycoprotein hormones" (L3MBTL2 represses CGA in PDAC -- the hormone gene),
# is still kept, by "glycoprotein hormones".  Only the window route changed;
# full name, designations, unambiguous aliases and own definitions count as
# before.  Offline, data-driven (gene_info + symbol_sense.json), no list.
OWN_FRAG_MAX_DF = 50
_WORD_DF: dict = {}
_RIVAL_RX: dict = {}


def _name_word_df(path=None):
    """{word: number of genes whose official name / designations contain it},
    over the whole gene_info dump (tokens as _norm_name makes them).  Built
    once per dump (about 2 s); {} on any failure (then no fragment is dropped
    for frequency)."""
    p = _gi_path(path)
    if p in _WORD_DF:
        return _WORD_DF[p]
    df = {}
    try:
        for rec in load_gene_index(path)["by_sym"].values():
            ws = set()
            for x in [rec.get("name", "")] + list(rec.get("desig", ())):
                x = str(x or "")
                ws.update(_norm_name(x).split() if not x.isascii()
                          else re.findall(r"[a-z0-9]+", x.casefold()))
            for w in ws:
                df[w] = df.get(w, 0) + 1
    except Exception:
        df = {}
    _WORD_DF[p] = df
    return df


def rival_genes(gene, path=None):
    """The genes competing for this symbol: those whose NCBI Synonyms list it
    (colliding_genes) and those the indexed literature defines it as
    (symbol_sense 'other').  Offline."""
    g = str(gene or "").strip()
    out = [o for o, _ in colliding_genes(g, path)]
    try:
        for o in ((symbol_sense(g) or {}).get("other") or {}):
            if o != g and o not in out:
                out.append(o)
    except Exception:
        pass
    return out


def own_fragments(gene, path=None):
    """ROUND 3: the multiword own-name fragments that may name the gene of a
    DOMINATED symbol next to a bare hit (see the block above).  Cached."""
    key = ("frag", str(gene), _gi_path(path))
    if key in _FULL_CACHE:
        return _FULL_CACHE[key]
    g = str(gene or "").strip()
    out = set()
    try:
        own = _own_names(g, path)
        rn = [" %s " % n for o in rival_genes(g, path) for n in _own_names(o, path)["names"]]
        df = _name_word_df(path)

        def _edge(w):
            return w not in _NAME_STOP and not _is_designator(w)
        for n in own["names"]:
            ws = n.split()
            for i in range(len(ws)):
                for j in range(i + 2, min(len(ws), i + 4) + 1):
                    fr = ws[i:j]
                    if not (_edge(fr[0]) and _edge(fr[-1])):
                        continue
                    if not any(len(w) >= 4 and w not in _GENERIC_WORDS and not _is_designator(w)
                               for w in fr):
                        continue
                    if df and min(df.get(w, 0) for w in fr) > OWN_FRAG_MAX_DF:
                        continue
                    f = " ".join(fr)
                    if any(" %s " % f in x for x in rn):
                        continue
                    out.add(f)
    except Exception:
        out = set()
    _FULL_CACHE[key] = frozenset(out)
    return _FULL_CACHE[key]


def _lstem(w):
    """Light stem for the lead-word match: 'mitochondrially' / 'mitochondrial'
    -> 'mitochondri', 'complements' -> 'complement'."""
    return _VARIANT_END.sub("", re.sub(r"ly$", "", str(w or "")))


def _lead_forms(w):
    """VERIFICATION FIX V2 (2026-09-26): the spellings of a LEAD word that
    count next to a bare hit -- the word, its plural, and the -al / -ally pair
    ("mitochondrial" ~ "mitochondrially").  _lstem() also strips -ary / -ic /
    -ous / -ar, so "complementary" matched C6's lead word "complement": "Rat
    C6 glioma cells were transfected with a complementary DNA" named the
    complement gene (round 2 refused it).  _lstem() stays for the RIVAL test
    in own_lead_word(), where a broader match only makes the rule stricter."""
    w = str(w or "")
    if not w:
        return frozenset()
    f = {w, w + "s", w + "es"}
    if w.endswith("ally"):
        f |= {w[:-2], w[:-2] + "s"}
    elif w.endswith("al"):
        f.add(w + "ly")
    return frozenset(f)


def own_lead_word(gene, path=None):
    """ROUND 3: the first content word (>= 4 letters, not _GENERIC_WORDS, not a
    designator) of the gene's OFFICIAL full name, or "" when there is none or
    a rival gene's names use the same word (stem).  Cached; offline."""
    key = ("lead", str(gene), _gi_path(path))
    if key in _FULL_CACHE:
        return _FULL_CACHE[key]
    g = str(gene or "").strip()
    lead = ""
    try:
        nm = (load_gene_index(path)["by_sym"].get(g) or {}).get("name", "")
        for w in _norm_name(nm).split():
            if len(w) >= 4 and w not in _GENERIC_WORDS and w not in _NAME_STOP and not _is_designator(w):
                lead = w
                break
        if lead and len(_lstem(lead)) >= 4:
            st = _lstem(lead)
            for o in rival_genes(g, path):
                if any(_lstem(x) == st for n in _own_names(o, path)["names"] for x in n.split()):
                    lead = ""
                    break
        else:
            lead = ""
    except Exception:
        lead = ""
    _FULL_CACHE[key] = lead
    return lead


def rival_free_aliases(gene, path=None):
    """ROUND 3: this gene's NCBI synonyms (>= 3 characters, not any gene's
    official symbol) that no RIVAL gene uses -- e.g. CGA's "HCG" (shared with
    CGB5 / HTC2, never with CHGA).  Near a bare hit of a dominated symbol they
    say which sense is meant even when other, non-rival genes share them.
    Cached; offline."""
    key = ("rfa", str(gene), _gi_path(path))
    if key in _FULL_CACHE:
        return _FULL_CACHE[key]
    g = str(gene or "").strip()
    out = []
    try:
        ix = load_gene_index(path)
        riv = set(rival_genes(g, path))
        for a in (ix["by_sym"].get(g) or {}).get("syn", ()):
            if len(a) < 3 or a in ix["by_sym"]:
                continue
            if riv & set(ix["syn_owners"].get(a, ())):
                continue
            out.append(a)
    except Exception:
        out = []
    _FULL_CACHE[key] = tuple(out)
    return _FULL_CACHE[key]


def _rival_rx(gene, path=None):
    """Regex of the rivals' official symbols (case-sensitive) and official full
    names / designations of >= 8 characters (case-insensitive), or None."""
    key = (str(gene), _gi_path(path))
    if key in _RIVAL_RX:
        return _RIVAL_RX[key]
    rx = None
    try:
        by = load_gene_index(path)["by_sym"]
        syms, names = set(), set()
        for o in rival_genes(gene, path):
            syms.add(o)
            rec = by.get(o) or {}
            for x in [rec.get("name", "")] + list(rec.get("desig", ())):
                if len(str(x or "")) >= 8:
                    names.add(str(x))
        alt = ["(?:%s)" % _pat(x) for x in sorted(syms, key=len, reverse=True)]
        alt += ["(?i:%s)" % _pat(x) for x in sorted(names, key=len, reverse=True)]
        if alt:
            rx = re.compile(_WB % ("(?:%s)" % "|".join(alt)))
    except Exception:
        rx = None
    _RIVAL_RX[key] = rx
    return rx


def passage_names_own(text, gene, gene_name="", spans=None, defs=None, path=None):
    """Does the passage name THIS gene by anything but the bare symbol?  Own
    official full name, an own designation (>= 8 chars), an unambiguous alias,
    an own definition, or (round 3) within COLLISION_WINDOW of a symbol hit,
    with no rival gene named in that window: a rare multiword FRAGMENT of its
    own names, the LEAD word of its official name, or an own synonym no rival
    uses (own_fragments / own_lead_word / rival_free_aliases / _rival_rx;
    round 2 took any single word)."""
    t = str(text or "")
    g = str(gene or "").strip()
    nm = str(gene_name or "").strip()
    if len(nm) >= 5 and re.search(_WB % _pat(nm), t, re.I):
        return "full name"
    own = _own_names(g, path)
    for x in own["raw"]:
        if re.search(_WB % _pat(x), t, re.I):
            return "designation '%s'" % x
    for al in unambiguous_aliases(g, path):
        if re.search(_WB % _pat(al), t, re.I if re.search(r"\d", al) else 0):
            return "alias %s" % al
    if any(k == "own" for k, _ in (defs or [])):
        return "own definition"
    W = COLLISION_WINDOW
    frags = own_fragments(g, path)
    lead = own_lead_word(g, path)
    if not spans or not (frags or lead or rival_free_aliases(g, path)):
        return ""
    lforms = _lead_forms(lead) if lead else frozenset()      # V2 (was: _lstem)
    rrx = _rival_rx(g, path)
    for a, b in (spans or []):
        wt = t[max(0, a - W):b + W]
        if rrx is not None and rrx.search(wt):
            continue                      # a rival gene is named right here
        nw = _norm_name(wt)
        w = " %s " % nw
        hit = sorted((f for f in frags if " %s " % f in w), key=len)
        if hit:
            return "own-name fragment '%s'" % hit[-1]
        if lforms and any(x in lforms for x in nw.split()):
            return "own lead word '%s'" % lead
        for al in rival_free_aliases(g, path):
            for m in re.finditer(_WB % _pat(al), wt, re.I):
                if m.group(0) == al or _symbol_cased(m.group(0)):
                    return "own alias '%s' (no rival uses it)" % m.group(0)
    return ""


def build_symbol_sense(files, out_path=None, symbols=None, path=None, progress=None):
    """Build symbol_sense.json: for every COLLIDING official symbol (another
    gene lists it as a synonym), count the definitions in the corpus documents
    (parsed/*.jsonl: title + every section) with symbol_definitions() -- the
    SAME classifier the gate uses at run time.  Case-sensitive symbol,
    definitions only (bare mentions are not counted).  Read-only on the
    corpus; writes `out_path` only."""
    import collections
    ix = load_gene_index(path)
    targets = set(symbols) if symbols else {
        s for s in ix["by_sym"] if any(o != s for o in ix["syn_owners"].get(s, ()))}
    cnt = {s: {"own": 0, "other": collections.Counter(), "nongene": 0, "examples": []}
           for s in targets}
    in_rx = re.compile(r"[(\[]\s*([A-Za-z0-9][A-Za-z0-9\-‐.]{1,14})\s*[)\];,]")
    pre_rx = re.compile(r"(?<![A-Za-z0-9\-‐.])([A-Za-z0-9][A-Za-z0-9\-‐.]{1,14})\s*[(\[]")
    n_docs, t0, stats, done = 0, time.time(), {}, set()
    for f in files:
        nd = 0
        with open(f, errors="replace") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                did = str(r.get("pmid") or r.get("doc_id") or "")
                if did and did in done:          # a paper carried into several corpora
                    continue
                done.add(did)
                txt = (r.get("title") or "") + " . " + " ".join(
                    str(s.get("text") or "") for s in (r.get("sections") or []))
                seen = set()
                for rx_ in (in_rx, pre_rx):
                    for m in rx_.finditer(txt):
                        s = m.group(1)
                        if s not in targets:
                            continue
                        a, b = m.span(1)
                        for kind, lf in symbol_definitions(txt, s, [(a, b)], path):
                            k2 = (s, kind, _norm_name(lf))
                            if k2 in seen:          # count each sense once per document
                                continue
                            seen.add(k2)
                            c = cnt[s]
                            if kind == "own":
                                c["own"] += 1
                            elif kind.startswith("other:"):
                                c["other"][kind[6:]] += 1
                            else:
                                c["nongene"] += 1
                            if kind != "own" and len(c["examples"]) < 6:
                                c["examples"].append([str(r.get("pmid") or r.get("doc_id")), kind, lf[:80]])
                nd += 1
        n_docs += nd
        stats[os.path.basename(f)] = {"docs": nd, "bytes": os.path.getsize(f)}
        if progress:
            progress("%s %d docs %.0fs" % (os.path.basename(f), nd, time.time() - t0))
    sym = {}
    for s, c in cnt.items():
        if c["own"] or c["other"] or c["nongene"]:
            sym[s] = {"own": c["own"], "other": dict(c["other"]), "nongene": c["nongene"],
                      "examples": c["examples"]}
    doc = {"built": time.strftime("%Y-%m-%dT%H:%M:%S"), "gene_info": _gi_path(path),
           "n_targets": len(targets), "n_docs": n_docs, "files": stats,
           "rule": {"dominated": "documents defining the symbol as ANOTHER gene >= %d and >= %d x "
                                 "documents defining it as its own gene (applied at run time)"
                    % (SENSE_MIN_NOT_OWN, SENSE_RATIO)},
           "symbols": sym}
    if out_path:
        tmp = out_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(doc, fh, indent=0)
        os.replace(tmp, out_path)
    return doc


# ---------------------------------------------------------------------------
# domain tie (scope="domain")
# ---------------------------------------------------------------------------
LITERATURE_SRC = "/BIO1/MRAG/Agent/orchestrator/src"


def _literature():
    """mrag_orchestrator.literature, or None when this module was loaded
    standalone into a venv that cannot see the package (the score runner)."""
    for attempt in (0, 1):
        try:
            from mrag_orchestrator import literature as L
            return L
        except Exception:
            if attempt:
                return None
            p = os.environ.get("MRAG_ORCH_SRC") or LITERATURE_SRC
            if p and p not in sys.path:
                sys.path.insert(0, p)
    return None


def mentions_domain(text, terms, exclude=()):
    """VERBATIM copy of literature.mentions_domain, which its own docstring
    asks for wherever that module cannot be imported: every `exclude` match is
    blanked with spaces of equal length, then any `terms` match counts.  All
    patterns are Python regexes applied with re.IGNORECASE; case-sensitive
    abbreviations carry a scoped (?-i:...) group."""
    t = str(text or "")
    for p in exclude:
        t = re.sub(p, lambda m: " " * len(m.group(0)), t, flags=re.I)
    return any(re.search(p, t, flags=re.I) for p in terms)


def domain_specs(domains):
    """Normalise `domains` to [{"key", "label", "terms", "exclude", "veto"}, ...].

    Accepts corpus keys ("sclc", "prostate,nec", ["sclc"]), resolved against
    literature.DOMAIN_TERMS / DOMAIN_EXCLUDE, or the dicts
    literature.domain_specs() already returns (which is how the node hands
    them to the standalone-loaded copy).  Order is kept, duplicates dropped,
    unusable entries dropped.  An EMPTY result means "the domain scope cannot
    be judged here" -- apply_gate then changes nothing rather than capping
    every gene on a test it did not run.  Never raises."""
    items = domains
    if items is None:
        return []
    if isinstance(items, str):
        items = [x for x in re.split(r"[,\s]+", items) if x]
    elif isinstance(items, dict):
        items = [items]
    keys, out = [], []
    for it in items:
        if isinstance(it, dict):
            terms = list(it.get("terms") or ())
            if terms:
                out.append({"key": str(it.get("key") or "").strip().lower(),
                            "label": str(it.get("display") or it.get("label")
                                         or it.get("key") or ""),
                            "terms": terms,
                            "exclude": list(it.get("exclude") or ()),
                            "veto": list(it.get("veto") or ())})
                if it.get("evfilter"):             # evaudit 2026-09-27
                    out[-1]["evfilter"] = it["evfilter"]
            else:                      # a spec with no terms is just a key
                keys.append(str(it.get("key") or ""))
        else:
            keys.append(str(it or ""))
    keys = [k.strip().lower() for k in keys if str(k).strip()]
    if keys:
        L = _literature()
        if L is not None:
            try:
                for sp in L.domain_specs(keys):
                    out.append({"key": sp["key"],
                                "label": sp.get("display") or sp["key"],
                                "terms": list(sp["terms"]),
                                "exclude": list(sp.get("exclude") or ()),
                                "veto": list(sp.get("veto") or ())})
                    if sp.get("evfilter"):         # evaudit 2026-09-27
                        out[-1]["evfilter"] = sp["evfilter"]
            except Exception:
                pass
    seen, uniq = set(), []
    for sp in out:
        if not sp["key"] or not sp["terms"] or sp["key"] in seen:
            continue
        seen.add(sp["key"])
        uniq.append(sp)
    return uniq


def domains_label(specs):
    """'Small cell lung cancer (SCLC)' / 'A, B' for the gate reason."""
    return ", ".join(sp.get("label") or sp.get("key") for sp in (specs or []))


def _hit_corpora(h):
    """The Qdrant `cancer` payload of a snippet as a set of corpus keys: it is
    a plain string for a paper in one corpus and a list for a paper indexed in
    several ("pancancer|nec|prostate")."""
    if not isinstance(h, dict):
        return set()
    c = h.get("cancer")
    if c is None:
        return set()
    if isinstance(c, (list, tuple, set)):
        return {str(x).strip().lower() for x in c if str(x).strip()}
    c = str(c).strip().lower()
    return {c} if c else set()


# E2 (2026-09-25): which domains need the TEXT test even for a chunk carrying
# the domain's corpus label.  The sclc corpus query ("small cell lung cancer"
# [TIAB] ...) also matches "non-small cell lung cancer", and label carry added
# sclc to papers indexed as nsclc, so a corpus label alone let NSCLC papers
# ground SCLC scores (09-25 SCLC run: three genes passed on NSCLC /
# glioblastoma passages only).
# ROUND 2 (2026-09-26): default "sclc" ONLY (was "exclude" = every domain with
# DOMAIN_EXCLUDE rules: sclc, nec, pancreatic, bladder).  Chosen on data --
# document-level precision of each corpus label (ledger snapshot 2026-09-25
# 22:15, title + abstract mention the domain after the excluded phrases are
# blanked):
#     sclc 798/3776 = 21.1%   nec 1032/1081 = 95.5%   pancreatic 1937/1949 =
#     99.4%   bladder 1131/1138 = 99.4%   (prostate 5419/5443 = 99.6%)
# Only the sclc label is a substring trap in practice.  For nec the chunk-level
# text test dropped 6 corpus ties in the 09-25 NEC run, about 4 on-domain
# (NE-prostate and amphicrine-carcinoma chunks that simply lack NE words)
# against 2 spurious (a sinonasal review, a polyomavirus paper) -- it cost
# more genuine ties than it
# removed, and a document-level test would keep the spurious ones too.
# MRAG_CORPUS_TIE_TEXT = sclc | exclude | all | none | comma-separated keys.
#
# DEVIATION FROM THE 2026-09-25 SPEC, recorded explicitly (round 3): the spec
# asked for the text requirement on "at least sclc and nec".  nec is left out
# ON PURPOSE.  Re-measured in round 3 on the 09-25 NEC run (recovered full
# chunks): 52 nec-labelled chunks name their gene; 7 of them (6 gene/PMID
# pairs) fail the text test -- 3 genuine NE-prostate / ONECUT2 chunks (2
# papers, also labelled prostate), 1 genuine amphicrine-carcinoma case report
# (labelled nec only, no NE word in the chunk or title) and 2 spurious (a
# sinonasal review labelled nec + melanoma; a polyomavirus control-region
# paper, nec only).  No score depends on them: re-gated with the text test,
# the four genes on the kept chunks still pass on another nec tie, and the
# three that would flip to "fired" (one chunk not recoverable) are at or
# below the cap, so the gate would not change their scores.
# The verifier's optional
# refinement (text test only for a chunk with no other "NEC-related" corpus
# label and no NEC word in its title) was measured too: it removes both
# spurious chunks but also the genuine amphicrine chunk, and it needs a hand-made
# list of which corpus labels count as NEC-related, so it was not adopted.
# MRAG_CORPUS_TIE_TEXT=exclude applies the text test to nec as well.
CORPUS_TIE_TEXT_DEFAULT = "sclc"
# necca 2026-09-26 (PI: nec = neuroendocrine CARCINOMA only).  nec still needs
# no text test for a corpus tie (8 of 1,104 nec-labelled documents are NET-only
# by title + abstract), but a nec-labelled passage that is NET-type ONLY --
# literature.DOMAIN_CORPUS_VETO matches, no carcinoma-type DOMAIN_TERMS pattern
# does -- no longer ties (PanNET chunks in the 09-26 NEC r2 run).
# MRAG_CORPUS_TIE_VETO=off restores the label-only tie.
CORPUS_TIE_VETO_DEFAULT = "on"
# v2 = 2026-09-25 sclc text rule; v3 = 2026-09-26 nec carcinoma-only rule.  A
# stored decision for a domain carrying a veto (nec) is trusted only from v3 on
# -- a v2 nec decision was taken under the "any neuroendocrine word" rule -- and
# is otherwise re-decided on the text at hand.
DOMAIN_TIE_VERSION = 3
_VETO_RULE_SINCE = 3


def corpus_veto_on(param=None):
    v = param if param is not None else os.environ.get("MRAG_CORPUS_TIE_VETO")
    v = str(v if v is not None and str(v).strip() else CORPUS_TIE_VETO_DEFAULT).strip().lower()
    return v not in ("off", "none", "0", "false", "no")


def _veto_patterns(spec, param=None):
    return list((spec or {}).get("veto") or ()) if corpus_veto_on(param) else []


def corpus_vetoed(spec, text, param=None):
    """necca 2026-09-26: is this passage (title + chunk) NET-type ONLY for the
    domain -- a veto pattern matches the raw text and, after the excluded
    phrases are blanked, no domain term does?  Then its corpus label does not
    tie it.  False for a domain without veto patterns or with the veto off."""
    vp = _veto_patterns(spec, param)
    if not vp:
        return False
    t = str(text or "")
    if not any(re.search(p, t, flags=re.I) for p in vp):
        return False
    return not mentions_domain(t, (spec or {}).get("terms") or (), (spec or {}).get("exclude") or ())


def _stored_version(v):
    """The `stored_v2` argument as an int rule version: True = the current rule
    (the legacy boolean callers), False / None = unknown (0), an int = that
    version."""
    if v is True:
        return DOMAIN_TIE_VERSION
    if not v:
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def corpus_needs_text(spec, param=None):
    """Does a corpus-labelled chunk of this domain also need the text test?"""
    v = param if param is not None else os.environ.get("MRAG_CORPUS_TIE_TEXT")
    v = str(v if v is not None and str(v).strip() else CORPUS_TIE_TEXT_DEFAULT).strip().lower()
    if v == "all":
        return True
    if v in ("none", "0", "off", "false", "no"):
        return False
    if v == "exclude":
        return bool((spec or {}).get("exclude"))
    return str((spec or {}).get("key") or "").lower() in {x.strip() for x in v.split(",")}


# Gene-domain CO-OCCURRENCE (2026-09-25, E2 add-on).  A passage can mention the
# domain about ANOTHER gene: a glioblastoma paper labelled sclc names one
# candidate and says "PLEKHA4 promotes tumor growth in ... small-cell lung
# cancer" -- the text test alone passes it.  For a domain listed here, a
# passage whose TITLE does not mention the domain must carry a domain mention
# and a gene mention (symbol, official full name or unambiguous alias) in the
# same or an adjacent sentence.  Default "sclc" ONLY, chosen on data (7 ONECUT2
# runs + PRAD, 43 sclc / 84 nec text-passing hits naming the gene): for sclc it
# drops 2 hits, both spurious (that glioblastoma paper; a Notch review whose
# SCLC sentences are about DLL3), and keeps a genuine SCLC-marker paper; for
# nec it would drop 8, several genuine (NE-prostate papers whose NE sentences
# sit far from the gene), so nec is not in the default.  GeneRIF lines are exempt: a
# GeneRIF is about its gene by construction.  MRAG_DOMAIN_TIE_COOC = none |
# all | exclude | comma-separated keys.
COOC_DEFAULT = "sclc"
_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+")


def cooc_required(spec, param=None):
    v = param if param is not None else os.environ.get("MRAG_DOMAIN_TIE_COOC")
    v = str(v if v is not None and str(v).strip() else COOC_DEFAULT).strip().lower()
    if v == "all":
        return True
    if v in ("none", "0", "off", "false", "no"):
        return False
    if v == "exclude":
        return bool((spec or {}).get("exclude"))
    return str((spec or {}).get("key") or "").lower() in {x.strip() for x in v.split(",")}


def gene_domain_cooccur(text, gene, gene_name="", spec=None, title="", path=None, window=1):
    """True when `title` mentions the domain, or a domain mention and a mention
    of the gene sit within `window` sentences of each other in `text` (1 =
    same or adjacent sentence, the corpus-chunk rule; 0 = same sentence, the
    PubMed-rescue rule, round 2)."""
    sp = spec or {}
    if title and mentions_domain(title, sp.get("terms") or (), sp.get("exclude") or ()):
        return True
    t = str(text or "")
    g = str(gene or "").strip()
    if not t or not g:
        return False
    tb = t
    for p in sp.get("exclude") or ():
        tb = re.sub(p, lambda m: " " * len(m.group(0)), tb, flags=re.I)
    dpos = [m.start() for p in (sp.get("terms") or ()) for m in re.finditer(p, tb, flags=re.I)]
    if not dpos:
        return False
    pats = [(g, re.I if re.search(r"\d", g) else 0)]
    if len(str(gene_name or "")) >= 5:
        pats.append((gene_name, re.I))
    try:
        pats += [(a, re.I if re.search(r"\d", a) else 0) for a in unambiguous_aliases(g, path)]
    except Exception:
        pass
    gpos = [m.start() for x, fl in pats for m in re.finditer(_WB % _pat(x), t, fl)]
    if not gpos:
        return False
    bounds, pos = [], 0
    for piece in _SENT_SPLIT.split(t):
        i = t.find(piece, pos)
        bounds.append(i)
        pos = i + len(piece)

    def _si(p):
        j = 0
        for k, b in enumerate(bounds):
            if b <= p:
                j = k
        return j
    ds = {_si(p) for p in dpos}
    gs = {_si(p) for p in gpos}
    return any(abs(a - b) <= int(window) for a in ds for b in gs)


# PROMPT EXCERPT (findfix 2026-09-26, finding 2).  The curated-target LLM used
# to be shown title + chunk[:160] of each passage -- the START of a 512-word
# chunk.  In the three 09-26 single-domain runs only 26-30 % of the literature
# lines it saw contained the gene at all, and for 40-60 % of the genes the gate
# verified (a passage naming the gene AND tied to the domain) no line it saw
# carried gene + domain; those genes were scored 0-1 with reasons such as "no
# SCLC evidence" although the gate had counted
# genuine domain passages.  The
# excerpt is cut from the FULL chunk around a mention of the gene -- symbol,
# official full name or unambiguous alias, the patterns gene_domain_cooccur
# uses -- and, given domain specs, around the gene mention nearest to a domain
# mention (excluded phrases blanked first), so both land in the window when
# they are within EXCERPT_CHARS of each other.  "" when the chunk does not
# mention the gene (the caller then falls back to the old chunk start).
EXCERPT_CHARS = 320


def gene_excerpt(text, gene, gene_name="", specs=None, width=None, path=None):
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    g = str(gene or "").strip()
    if not t or not g:
        return ""
    W = max(80, int(width or EXCERPT_CHARS))
    pats = [(g, re.I if re.search(r"\d", g) else 0)]
    if len(str(gene_name or "")) >= 5:
        pats.append((gene_name, re.I))
    try:
        pats += [(a, re.I if re.search(r"\d", a) else 0) for a in unambiguous_aliases(g, path)]
    except Exception:
        pass
    gsp = sorted({(m.start(), m.end()) for x, fl in pats
                  for m in re.finditer(_WB % _pat(x), t, fl) if m.end() > m.start()})
    if not gsp:
        return ""
    dsp = []
    for sp in (specs or []):
        tb = t
        for p in sp.get("exclude") or ():
            tb = re.sub(p, lambda m: " " * len(m.group(0)), tb, flags=re.I)
        dsp += [(m.start(), m.end()) for p in (sp.get("terms") or ())
                for m in re.finditer(p, tb, flags=re.I) if m.end() > m.start()]

    def _gap(a, b):
        return max(0, max(a[0], b[0]) - min(a[1], b[1]))
    gs = gsp[0]
    lo0, hi0 = gs
    if dsp:
        gs = min(gsp, key=lambda s: (min(_gap(s, d) for d in dsp), s[0]))
        ds = min(dsp, key=lambda d: (_gap(gs, d), d[0]))
        lo0, hi0 = gs
        if max(gs[1], ds[1]) - min(gs[0], ds[0]) <= W:
            lo0, hi0 = min(gs[0], ds[0]), max(gs[1], ds[1])
    c = (lo0 + hi0) // 2
    lo = max(0, c - W // 2)
    hi = min(len(t), lo + W)
    lo = max(0, hi - W)
    if lo > 0:                      # word boundary, never past the anchor
        k = t.find(" ", lo, lo0)
        if k != -1:
            lo = k + 1
    if hi < len(t):
        k = t.rfind(" ", hi0, hi)
        if k != -1:
            hi = k
    return ("..." if lo > 0 else "") + t[lo:hi].strip() + ("..." if hi < len(t) else "")


def _tie_one(sp, text, cs, stored=None, stored_v2=False, gene=None, gene_name="",
             title=None, path=None):
    """One domain's tie for one passage: "corpus" | "text" | ""."""
    k = sp["key"]
    strict = corpus_needs_text(sp)
    veto = _veto_patterns(sp)
    ver = _stored_version(stored_v2)
    if k in cs and not strict and not veto:
        return "corpus"                        # unchanged for non-strict domains
    if isinstance(stored, dict) and k in stored:
        s0 = str(stored.get(k) or "")
        # A stored decision was taken by the retriever on the FULL passage.  One
        # taken under the pre-2026-09-25 rule said "corpus" without any text
        # test, so for a strict domain it is re-tested below instead of trusted.
        # necca 2026-09-26: for a veto domain (nec) only a v3 decision is
        # trusted; an older one is re-decided on `text` below.
        if veto and ver < _VETO_RULE_SINCE:
            pass
        elif ver >= 2 or s0 != "corpus" or not strict:
            return s0
    if k in cs and not strict:
        # veto domain: the label ties unless the passage is NET-type only
        return "" if corpus_vetoed(sp, text) else "corpus"
    if mentions_domain(text, sp["terms"], sp.get("exclude") or ()):
        if gene and cooc_required(sp):
            _body = str(text or "")
            _tt = str(title or "")
            if _tt and _body.startswith(_tt):
                _body = _body[len(_tt):]
            if not gene_domain_cooccur(_body, gene, gene_name, sp, _tt, path):
                return ""
        return "corpus" if k in cs else "text"
    return ""


def domain_tie(text, specs, corpora=(), stored=None, stored_v2=False, gene=None,
               gene_name="", title=None, path=None):
    """How this passage ties to one of `specs`: ("corpus"|"text", key), or
    ("", "") for no tie.

    Corpus membership is tested first.  FIX 2026-09-25 (E2): for a domain whose
    corpus label is imprecise (sclc by default -- round 2; see
    corpus_needs_text / CORPUS_TIE_TEXT_DEFAULT) a corpus label is NOT enough -- the passage (chunk text + stored title) must itself
    mention the domain after the excluded phrases ("non-small cell lung
    cancer", "head and neck") are blanked; it then still reports "corpus".
    `stored` is an optional {domain_key: "corpus"|"text"|""} decision the
    retriever took on the FULL passage (only title[:120] + text[:220] survives
    into scored.json, so re-matching the stored snippet UNDERCOUNTS);
    `stored_v2` says it was taken under the 2026-09-25 rule.  A key present in
    `stored` wins for that domain, except a pre-2026-09-25 "corpus" for a
    strict domain, which is re-tested on `text`.  With `gene`, a domain listed
    in cooc_required() also needs gene-domain co-occurrence (see
    gene_domain_cooccur); without it (GeneRIF lines) that test is skipped."""
    cs = set(corpora or ())
    res = [(_tie_one(sp, text, cs, stored, stored_v2, gene, gene_name, title, path), sp["key"])
           for sp in (specs or [])]
    for how, k in res:
        if how == "corpus":
            return how, k
    for how, k in res:
        if how:
            return how, k
    return "", ""


def tie_record(full_text, corpora, specs, gene=None, gene_name="", title=None, path=None):
    """The per-domain decisions a retriever stores on a hit, taken on the FULL
    passage under the current rule: ({key: "corpus"|"text"|""}, notes) where
    notes lists the domains whose corpus label was NOT accepted.  `full_text`
    is title + " " + chunk; pass `title` (and `gene`) so the co-occurrence
    test can tell the title from the body."""
    cs = set(corpora or ())
    dec, notes = {}, {}
    for sp in (specs or []):
        how = _tie_one(sp, full_text, cs, None, True, gene, gene_name, title, path)
        # evaudit 2026-09-27: 1b / 1c on the FULL passage; the caller stamps
        # the hit domain_tie_v = tie_version() (4) so the gate trusts this
        if how and gene and retrieval_filters_on():
            _why = retrieval_filters(sp, full_text, gene, gene_name, title or "", path)
            if _why:
                dec[sp["key"]] = ""
                notes[sp["key"]] = _why
                continue
        dec[sp["key"]] = how
        if sp["key"] in cs and not how:
            notes[sp["key"]] = (
                "corpus label refused: the passage names only NET-type "
                "neuroendocrine neoplasms (no carcinoma type)"
                if corpus_vetoed(sp, full_text) and not corpus_needs_text(sp)
                else "corpus label only: the passage does not mention the domain "
                "(excluded phrases blanked)"
                if not mentions_domain(full_text, sp["terms"], sp.get("exclude") or ())
                else "corpus label + domain mention, but the domain is not mentioned "
                     "near the gene (co-occurrence rule)")
    return dec, notes


# ---------------------------------------------------------------------------
# EVIDENCE FILTERS (evaudit 2026-09-27)
# ---------------------------------------------------------------------------
# PI 2026-09-27 ("make sure this never happens again"): two FINAL
# targets rested on invalid counted evidence and were found only by a
# manual review afterwards (a NEC target: a small-intestinal NET paper in the
# pre-2010 "carcinoma" wording; an SCLC target: named only in a list of
# screened, unaltered genes).  Three deterministic filters now run BEFORE any LLM; each
# can only REMOVE a tie the rules above granted, never add one:
#   1a paper_net_verdict   nec: the PAPER behind a tie (PubMed title / abstract
#                          / keywords / MeSH) is a well-differentiated-tumour
#                          paper -> its passages, GeneRIF lines and rescue
#                          records do not tie nec (literature.NEC_PAPER_*).
#   1b nec_sentence_veto   nec: the NET-only veto on the sentence(s) naming the
#                          gene, not only on the whole chunk.
#   1c list_negative       every domain: the passage names the gene only inside
#                          an enumeration of >= 4 gene symbols that reports no /
#                          rare alterations or a negative result.
# 1b and 1c are decided at RETRIEVAL on the full chunk (tie_record), and the
# hit is stamped domain_tie_v = EVFILTER_TIE_VERSION (5); a stored decision of
# an older version is re-filtered at gate time on the text at hand (title +
# excerpt).  1a needs the PubMed record and is applied at gate time from the
# paper map the score runner builds (paper_filter_map).
# Switches (all ON by default): MRAG_NEC_NET_PAPER, MRAG_NEC_VETO_SENTENCE,
# MRAG_LIST_NEGATIVE; MRAG_GENERIF_ALIAS for the GeneRIF alias rule below.
# With all four off every function here is a no-op and the gate is the
# pre-2026-09-27 one.
EVFILTER_VERSION = 2
# fix2 2026-09-28: 5 = decided with the revised 1b / 1c (a 4 of the first
# evaudit build is re-filtered at gate time like any older decision)
# fix3 2026-09-28: 6 = the old-wording / SCLC-line / lung-NE / contrast 1b rules
# and the revised 1c negatives (a 5 is re-filtered at gate time)
EVFILTER_TIE_VERSION = 6
LIST_NEG_MIN_ITEMS = 4
_PAPER_CACHE: dict = {}


def evidence_defence_on():
    """fix2 2026-09-28: MRAG_EVIDENCE_DEFENCE=0 switches EVERY 2026-09-27
    evidence layer off at once (1a-1c, GeneRIF alias, 600-char GeneRIF display,
    re-score after a filter, evidence audit): the gate is then the
    pre-2026-09-27 one byte for byte.  Default ON.
    fix3 2026-09-28 (second verification, finding 6): MRAG_EVIDENCE_AUDIT=0
    -- the documented rollback switch -- does the same (MRAG_EVIDENCE_AUDIT_MODE
    =off turns off only the LLM audit)."""
    for k in ("MRAG_EVIDENCE_DEFENCE", "MRAG_EVIDENCE_AUDIT"):
        v = str(os.environ.get(k) or "").strip().lower()
        if v in ("0", "off", "false", "no", "none"):
            return False
    return True


def _evf_on(name, param=None):
    if not evidence_defence_on():
        return False
    v = param if param is not None else os.environ.get(name)
    v = str(v if v is not None and str(v).strip() else "on").strip().lower()
    return v not in ("0", "off", "false", "no", "none")


def sentence_veto_on(param=None):
    return _evf_on("MRAG_NEC_VETO_SENTENCE", param)


def list_negative_on(param=None):
    return _evf_on("MRAG_LIST_NEGATIVE", param)


def paper_net_on(param=None):
    return _evf_on("MRAG_NEC_NET_PAPER", param)


def generif_alias_on(param=None):
    return _evf_on("MRAG_GENERIF_ALIAS", param)


def retrieval_filters_on():
    """Are the filters decided at retrieval (1b, 1c) on?"""
    return sentence_veto_on() or list_negative_on()


def evidence_layer_on():
    """Is any evaudit 2026-09-27 evidence rule on?"""
    return retrieval_filters_on() or paper_net_on() or generif_alias_on()


def tie_version():
    """domain_tie_v to stamp on a hit decided NOW: EVFILTER_TIE_VERSION (5)
    when the retrieval-time filters ran, else the plain rule version (3)."""
    return EVFILTER_TIE_VERSION if retrieval_filters_on() else DOMAIN_TIE_VERSION


_DASHES = re.compile("[‐-―−­]")


def _flat(x):
    return re.sub(r"\s+", " ", _DASHES.sub("-", str(x or ""))).strip()


def _sentences(t):
    """[(start, end)] of the sentences of `t` (split as _SENT_SPLIT does)."""
    out, pos = [], 0
    for piece in _SENT_SPLIT.split(t):
        i = t.find(piece, pos)
        if i < 0:
            i = pos
        out.append((i, i + len(piece)))
        pos = i + len(piece)
    return out


def _sent_index(bounds, p):
    j = 0
    for k, (b, _e) in enumerate(bounds):
        if b <= p:
            j = k
        else:
            break
    return j


def _gene_spans(t, gene, gene_name="", path=None):
    """Mentions of the gene: symbol (the case rule), official full name,
    unambiguous aliases -- the patterns gene_domain_cooccur uses."""
    g = str(gene or "").strip()
    if not g or not t:
        return []
    pats = [(g, re.I if re.search(r"\d", g) else 0)]
    if len(str(gene_name or "")) >= 5:
        pats.append((gene_name, re.I))
    try:
        pats += [(a, re.I if re.search(r"\d", a) else 0) for a in unambiguous_aliases(g, path)]
    except Exception:
        pass
    return sorted({(m.start(), m.end()) for x, fl in pats
                   for m in re.finditer(_WB % _pat(x), t, fl) if m.end() > m.start()})


# ---- 1b ---------------------------------------------------------------------
# fix2 2026-09-28 (verification of the evaudit build, finding 2): the NET-word
# test used to fire on the gene's OWN name -- INSM1 "insulinoma-associated
# protein 1" read as a NET ("insulinoma") in Merkel cell carcinoma papers and
# a high-grade sinonasal carcinoma -- and on lung papers that contrast
# carcinoids with carcinomas or call high-grade NE lung cancer lines
# "pulmonary neuroendocrine tumor cells" in the title.  Now:
#   * the gene's own symbol / full name / NCBI designations / synonyms are
#     blanked before any NET-word test (window and title);
#   * a window that contrasts carcinoids with carcinomas names a carcinoma
#     (sentence_veto.contrast);
#   * the loose "neuroendocrine tumour" wording is not a NET word -- in the
#     window AND the title -- when the passage has prostate-cancer or
#     high-grade / NE-lung-cancer context (sentence_veto.loose_ok).
_OWN_NAME_RX: dict = {}
_NAME_GAP = "[\\s\\-\u2010-\u2015\u2212]*"


def _own_name_rx(gene, gene_name="", path=None):
    """One case-insensitive regex for the gene's OWN names: symbol, official
    full name, NCBI other designations and synonyms (>= 4 characters); a
    trailing number / 'protein' / 'homolog' may be dropped and any space or
    hyphen may separate the words ('Insulinoma-associated 1' and
    'insulinoma-associated protein 1' are both INSM1).  None when nothing
    usable.  Cached."""
    g = str(gene or "").strip()
    key = (g, str(gene_name or ""), _gi_path(path))
    if key in _OWN_NAME_RX:
        return _OWN_NAME_RX[key]
    names = {g} if g else set()
    if len(str(gene_name or "")) >= 4:
        names.add(str(gene_name))
    try:
        rec = load_gene_index(path)["by_sym"].get(g) or {}
        if rec.get("name"):
            names.add(rec["name"])
        names.update(x for x in (rec.get("desig") or ()) if x)
        names.update(x for x in (rec.get("syn") or ()) if len(x) >= 4)
    except Exception:
        pass
    rx = _names_rx(names)
    _OWN_NAME_RX[key] = rx
    return rx


def _names_rx(names):
    """A flexible case-insensitive regex for a set of gene names (see
    _own_name_rx), or None."""
    alts = []
    for n in sorted({re.sub(r"\s+", " ", x).strip() for x in names if len(x.strip()) >= 3},
                    key=len, reverse=True):
        words = [w for w in re.split("[\\s\\-\u2010-\u2015\u2212]+", n) if w]
        if not words:
            continue
        core, tail = list(words), []
        # an optional tail only while >= 2 words stay: "insulinoma protein"
        # (an RPS15 synonym) must not shrink to the bare NET word "insulinoma"
        while len(core) > 2 and re.fullmatch(r"\d+[A-Za-z]?|[IVX]+|protein|homolog(?:ue)?", core[-1], re.I):
            tail.insert(0, core.pop())
        body = _NAME_GAP.join(re.escape(w) for w in core)
        body += "".join("(?:%s%s)?" % (_NAME_GAP, re.escape(w)) for w in tail)
        alts.append(body)
    if not alts:
        return None
    try:
        return re.compile(r"(?<![A-Za-z0-9])(?:%s)(?![A-Za-z0-9])" % "|".join(alts), re.I)
    except re.error:
        return None


_NET_NAMED_RX: dict = {}


def _net_named_genes_rx(veto, path=None):
    """The names of EVERY gene whose official name, designation or synonym
    contains a NET word of the veto (INSM1 / INSM2 "insulinoma-associated",
    PTPRN, IAPP, MADD, KCNA5 ... -- data-driven from gene_info): blanked before
    the NET-word test, so a marker named after a tumour is not that tumour
    ("insulinoma-associated antigen-1 (INSM1)" as a marker next to another gene).
    Cached per (gene_info, veto)."""
    key = (_gi_path(path), tuple(veto or ()))
    if key in _NET_NAMED_RX:
        return _NET_NAMED_RX[key]
    names = set()
    try:
        vx = [re.compile(p, re.I) for p in veto or ()]
        for _g, rec in load_gene_index(path)["by_sym"].items():
            for n in [rec.get("name") or ""] + list(rec.get("desig") or ()) + list(rec.get("syn") or ()):
                if n and len(n) >= 5 and any(v.search(n) for v in vx):
                    names.add(n)
    except Exception:
        names = set()
    rx = _names_rx(names)
    _NET_NAMED_RX[key] = rx
    return rx


def _blank_rx(x, rx):
    return rx.sub(lambda m: " " * len(m.group(0)), x) if rx is not None else x


def _oldword_net(ow, W, tt, whole, own=None, netnamed=None):
    """fix3 2026-09-28 (1b old-wording test): the NET-type phrase that makes the
    gene's sentences + title a well-differentiated tumour in the old
    "carcinoma" wording, or "" (see literature.NEC_OLDWORD_*).  `whole` =
    title + passage (for the "(NEC)" abbreviation definition)."""
    if not ow:
        return ""
    X = _blank_rx(_blank_rx(W + " . " + tt, own), netnamed)
    Xh = X
    for p in ow.get("hg_exclude") or ():
        Xh = re.sub(p, lambda m: " " * len(m.group(0)), Xh, flags=re.I)
    defs = list(ow.get("defining") or ())
    ab = str(ow.get("abbrev") or "")
    abbrev_def = bool(ab) and any(re.search("(?:%s)%s" % (d, ab), whole, flags=re.I) for d in defs)
    if abbrev_def:
        # "serotonin-producing neuroendocrine carcinomas (NECs)": NEC is the
        # NET-defined phrase here, not a high-grade marker
        Xh = re.sub(r"(?<![A-Za-z0-9-])NECs?\b", lambda m: " " * len(m.group(0)), Xh)
    for p in ow.get("hg") or ():
        if re.search(p, Xh, flags=re.I):
            return ""
    for key in ("defining", "site", "strong"):
        for p in ow.get(key) or ():
            m = re.search(p, X, flags=re.I)
            if m:
                return m.group(0)
    return ""


def _names_carcinoma_extra(sv, W, tt, excl):
    """fix3 2026-09-28 (finding 5): "" or how the gene's sentences / the title
    name a carcinoma without a DOMAIN_TERMS word: an SCLC cell line, or a bare
    "carcinoma(s)" in a lung-NE passage (the lung NE carcinomas)."""
    X = W + " . " + tt
    for p in sv.get("sclc_lines") or ():
        m = re.search(p, X, flags=re.I)
        if m:
            return "SCLC cell line %s" % m.group(0)
    if any(re.search(p, X, flags=re.I) for p in (sv.get("lung_ne") or ())):
        Wc = W
        for p in excl or ():
            Wc = re.sub(p, lambda m: " " * len(m.group(0)), Wc, flags=re.I)
        nt = sv.get("bare_carc_not") or ""
        for m in re.finditer(r"(?<![A-Za-z])carcinomas?\b", Wc, flags=re.I):
            if nt and re.search(nt, Wc[max(0, m.start() - 40):m.start()], flags=re.I):
                continue
            return "a bare 'carcinoma' in a lung-NE passage"
    return ""


def nec_sentence_veto(spec, text, gene, gene_name="", title="", path=None, param=None):
    """1b.  "" or why the tie does not count: for a domain with a NET veto
    (nec) whose spec carries evfilter.sentence_veto, the sentence(s) naming the
    gene (+- 1 sentence) name no carcinoma type and either name a NET-type
    neoplasm themselves or sit under a NET-only title.  A passage whose gene
    sentences name neither is left to the chunk-level rule (corpus_vetoed).
    `text` may start with `title` (tie_record passes title + chunk).
    fix2 2026-09-28: the gene's own names never count as NET words, a
    carcinoid-vs-carcinoma contrast names a carcinoma, the loose
    "neuroendocrine tumour" wording is ignored when the passage
    has prostate-cancer or high-grade context (window and title).
    fix3 2026-09-28 (second verification, findings 3 / 5): the OLD-WORDING
    test comes first -- a NET-defined carcinoma phrase ("serotonin-producing
    neuroendocrine carcinoma", "midgut carcinoid-derived neuroendocrine
    carcinoma"), a small-intestinal site phrase or a strong NET marker
    (enterochromaffin, midgut, SI-NET ...) in the gene's sentences or the
    title, with no high-grade marker there, vetoes even though the sentence
    says "neuroendocrine carcinoma"; an SCLC cell line or -- in a lung-NE
    passage -- a bare "carcinoma(s)" names a carcinoma; a high-grade / poorly
    differentiated tumour contrasted with carcinoids names one; in a
    prostate-cancer passage only the gene's OWN sentence(s) are read for NET
    words (a treatment background "octreotide is approved for carcinoid" in
    the next sentence is not the gene's tumour)."""
    sp = spec or {}
    sv = (sp.get("evfilter") or {}).get("sentence_veto")
    vp = _veto_patterns(sp)
    if not sv or not vp or not gene or not sentence_veto_on(param):
        return ""
    t = _flat(text)
    tt = _flat(title)
    body = t[len(tt):].strip() if (tt and t.startswith(tt)) else t
    spans = _gene_spans(body, gene, gene_name, path)
    if not spans:
        return ""
    bounds = _sentences(body)
    idx = {_sent_index(bounds, s) for s, _e in spans}
    win = sorted({j for i in idx for j in (i - 1, i, i + 1) if 0 <= j < len(bounds)})
    W = " ".join(body[bounds[j][0]:bounds[j][1]] for j in win)
    terms, excl = sp.get("terms") or (), sp.get("exclude") or ()

    def _blank(x, pats):
        for p in pats or ():
            x = re.sub(p, lambda m: " " * len(m.group(0)), x, flags=re.I)
        return x
    own = _own_name_rx(gene, gene_name, path)
    netnamed = _net_named_genes_rx(vp, path)
    _all = tt + " " + body
    prostate = any(re.search(p, _all, flags=re.I) for p in (sv.get("loose_ok") or ()))
    loose = (prostate
             or (any(re.search(p, _all, flags=re.I) for p in (sv.get("loose_ok_hg") or ()))
                 and not any(re.search(p, _all, flags=re.I) for p in (sv.get("loose_ok_hg_block") or ()))))
    # fix3: the old "carcinoma" wording of a NET-type tumour
    ow = _oldword_net(sv.get("oldword") or {}, W, tt, _all, own, netnamed)
    if ow:
        return ("sentence veto: the sentence(s) naming %s describe a NET-type tumour in the old 'carcinoma' "
                "wording (%r) and name no high-grade tumour" % (gene, ow[:50]))

    def _net_text(x):
        # what the NET-word test reads: umbrella terms (NEN = NET + NEC) and
        # the gene's own names blanked; the loose "NE tumour" wording too when
        # the passage has prostate-cancer / high-grade context
        x = _blank(x, sv.get("umbrella"))
        x = _blank_rx(_blank_rx(x, own), netnamed)
        if loose:
            x = _blank(x, sv.get("loose_net"))
        return x
    # the loose "pulmonary neuroendocrine tumor cells" title of a high-grade
    # NE lung cancer paper is no NET-only title: _net_text blanks
    # the loose wording when the passage has high-grade / prostate context
    tt_net = _net_text(tt)
    title_veto = bool(tt) and any(re.search(p, tt_net, flags=re.I) for p in vp)
    ctx = W if (title_veto or not tt) else (W + " . " + tt)
    if mentions_domain(ctx, terms, excl):
        return ""
    # fix3: an SCLC cell line, a bare "carcinoma" in a lung-NE passage -- unless
    # the gene's sentences also name a SPECIFIC NET-type neoplasm (carcinoid,
    # midgut NET ...; the loose "neuroendocrine tumour" wording aside):
    # e.g. "NCI-H69, BON-1, NCI-H727 (human pulmonary carcinoid) and GOT1
    # (human midgut neuroendocrine tumor) cells"
    if _names_carcinoma_extra(sv, W, tt, excl):
        W_spec = _blank(_blank_rx(_blank_rx(_blank(W, sv.get("umbrella")), own), netnamed), sv.get("loose_net"))
        if not any(re.search(p, W_spec, flags=re.I) for p in vp):
            return ""
    # carcinoids contrasted with carcinomas (fix3: or with high-grade tumours)
    # in the gene's sentences: the lung NE carcinomas (SCLC / LCNEC)
    Wc = _blank(W, excl)
    if any(re.search(p, Wc, flags=re.I) for p in (sv.get("contrast") or ())):
        return ""
    # fix3: prostate context -- the gene's own sentence(s) are read one by one: a
    # sentence that names no NET-type neoplasm but names the prostate-cancer /
    # NE-carcinoma context itself is a result the veto must not take away
    # (e.g. "transcriptional downregulation of SSTR1 in ARSI-resistant
    # mCRPC"; the SSTR1 of "octreotide ... spare SSTR1 ... approved to treat ...
    # carcinoid syndrome" is treatment background); a sentence without any
    # tumour context ("ONC201 is an antagonist of ... DRD2") saves nothing
    if prostate:
        hit = ""
        for j in sorted(idx):
            sj = body[bounds[j][0]:bounds[j][1]]
            Wj = _net_text(sj)
            hj = next((m.group(0) for p in vp for m in [re.search(p, Wj, flags=re.I)] if m), "")
            if not hj and (any(re.search(p, sj, flags=re.I) for p in (sv.get("loose_ok") or ()))
                           or mentions_domain(sj, terms, excl)):
                hit = ""
                break
            hit = hit or hj
    else:
        W2 = _net_text(W)
        hit = next((m.group(0) for p in vp for m in [re.search(p, W2, flags=re.I)] if m), "")
    if hit:
        return ("sentence veto: the sentence(s) naming %s name a NET-type neoplasm (%r) and no "
                "carcinoma type" % (gene, hit[:40]))
    if title_veto and not mentions_domain(tt, terms, excl):
        return ("sentence veto: the sentence(s) naming %s name no carcinoma type and the title "
                "names only NET-type neoplasms" % gene)
    return ""


# ---- 1c ---------------------------------------------------------------------
# fix2 2026-09-28 (verification of the evaudit build, finding 1): the first
# version dropped real POSITIVE list evidence -- about a third of what it
# removed in the sclc / nec / prostate corpora -- because a negative ANYWHERE in
# the gene's sentence won ("overexpression of PTF1A, GATA4, NR5A2, and RBPJL"
# next to "absence of KRAS mutations"), the positive list was short
# (identified / found / correlated / highly expressed were missing), a next
# sentence without any gene symbol counted as "referring back", and
# "rare" was read as "absent".  PI decision 2026-09-27: cohort /
# panel / list associations COUNT; only a screened-negative list (listed, then
# reported absent / negative) does not.  The negative must now GOVERN the
# enumeration:
#   (A) same clause -- the clause of the sentence that holds the enumeration
#       (split at ; / whereas / while / but / although / however / unlike /
#       ", with no" / "and negative|positive for" ... outside the list itself)
#       carries a negative cue and no positive cue that is not itself negated;
#   (B) next sentence -- the enumeration's sentence reports nothing of its own
#       (no positive cue, no negative), the next sentence carries a negative
#       cue and no positive cue, and it refers to the SAME list: an explicit
#       collective back-reference ("none of these", "these genes / loci /
#       markers", "any of them"), or -- after a screening sentence ("we
#       examined / screened / tested / evaluated ... including ...") -- a
#       sentence that names no gene outside the list or says that only one
#       alteration was found ("We found only one KRAS mutation").
# "rare / infrequent / uncommon" is no longer a negative; "0 of 50", "results
# were negative", "unremarkable", "lack of alterations" are.
# fix3 2026-09-28 (second verification, finding 4): a negative whose clause is
# about NORMAL tissue only ("... was not observed in adjacent normal lung",
# "mutations were not detected in normal tissue") and "no significant
# difference(s)" between two groups of tumours ("... between patients treated
# with enzalutamide or abiraterone", primary tumours vs their LN
# metastases) do not govern a list -- a tumour-vs-normal comparison still does
# ("no ... differential expression between tumor and normal
# tissues"); "not detected in 0.7% of cells" is a partial result.  New
# screened-negative phrasings: "No NTRK1, NTRK2, NTRK3 or ALK fusions were
# detected", "the results were uniformly negative", "Nothing was found", "the
# mutation rate ... was 0%", "Only 1 patient (2.5%) had a KRAS mutation"; a
# "0%" and the "(2.5%)" of such an only-one phrase are not positive cues.
_LN_TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9]|[A-Za-z0-9]")
_LN_SEP = re.compile(r"^[\s,;/&()*]*(?:(?:and|or)\b)?[\s,;/&*]*$", re.I)
_LN_OBJ = r"(?:mutations?|alterations?|variants?|amplifications?|deletions?|fusions?|rearrangements?" \
          r"|aberrations?|abnormalit\w+|changes?|SNVs?|CNVs?|CNAs?)"
LIST_NEG_PATTERNS: tuple = (
    r"\bno\s+(?:[\w-]+\s+){0,3}?" + _LN_OBJ + r"\b",
    r"\bno\s+(?:statistically\s+)?significant\s+(?:associations?|correlations?|changes?)\b",
    r"\bnone\s+of\s+(?:the|these|them|those)\b", r"\bnone\s+(?:was|were|had|showed|harbou?red|carried)\b",
    r"\b(?:was|were|is|are|been)\s+not\s+(?:[\w-]+\s+)?(?:detected|found|observed|identified|mutated|altered"
    r"|amplified|present)\b",
    r"\bnot\s+(?:be\s+)?(?:detected|found|observed|identified)\b",
    r"\bonly\s+(?:one|a\s+single|1|two|2)\s+(?:[\w.,()%-]+\s+){0,5}?" + _LN_OBJ + r"\b",
    r"\bonly\s+(?:an?|one|a\s+single|two|three|\d)\s[^.;]{0,160}?\b(?:was|were)\s+(?:detected|found|identified"
    r"|observed)\b",
    r"\bwild[\s-]*type\s+for\b", r"\b(?:all|were|was|remained)\s+(?:[\w-]+\s+){0,2}?wild[\s-]*type\b",
    r"(?:mutations?|alterations?|amplifications?|fusions?|variants?)\s+(?:[\w-]+\s+){0,4}?(?:were|was|are|is)"
    r"\s+absent\b",
    r"\b(?:absence|lack)\s+of\s+(?:any\s+)?(?:[\w-]+\s+){0,2}?" + _LN_OBJ,
    r"\bnegative\s+for\b",
    r"\bdid\s+not\s+(?:harbou?r|carry|show|have|detect|find|reveal|identify|display)\b",
    r"\bfailed\s+to\s+(?:detect|identify|find)\b",
    r"\bno\s+(?:[\w-]+\s+)?(?:actionable|targetable|druggable)\b",
    r"(?<![\d.])(?:0|zero|none)\s*(?:of|/|out\s+of)\s*\d+\b",
    r"\b(?:results?|tests?|findings?|analys[ie]s|screens?|screening)\s+(?:was|were)\s+(?:all\s+|uniformly\s+"
    r"|entirely\s+|completely\s+|consistently\s+)?negative\b",
    r"\b(?:was|were)\s+(?:all\s+)?negative\s+in\s+all\b",
    r"\bunremarkable\b",
    # fix3 2026-09-28
    r"\bno\s+[A-Za-z0-9][A-Za-z0-9-]*(?:\s*,\s*[A-Za-z0-9][A-Za-z0-9-]*)*,?\s+(?:or|and|nor)\s+[A-Za-z0-9]"
    r"[A-Za-z0-9-]*\s+(?:[\w-]+\s+){0,2}?" + _LN_OBJ + r"\b",
    r"\b(?:was|were|remained)\s+(?:all\s+)?(?:uniformly|entirely|completely|consistently)\s+negative\b",
    r"\bnothing\s+(?:was|were|could\s+be)\s+(?:found|detected|identified|observed)\b",
    r"(?<![\d.])0(?:\.0+)?\s*%",
)
# fix3 2026-09-28: a comparative negative governs a list only in a
# tumour-vs-normal comparison
LIST_NEG_COMPARATIVE: tuple = (
    r"\bno\s+(?:statistically\s+)?significant\s+differences?\b",
    r"\bnot\s+(?:statistically\s+)?significantly\s+differ\w*\b",
)
_LN_NORMAL = re.compile(r"\b(?:normal|benign|healthy|non[\s-]?(?:neoplastic|tumou?r\w*|malignant|cancerous)"
                        r"|germ[\s-]?line|stroma\w*)\b", re.I)
_LN_NONTUM = re.compile(r"\bnon[\s-]?(?:neoplastic|tumou?r\w*|malignant|cancerous)\b|\btumou?r(?:al)?[\s-]+stroma\w*",
                        re.I)
_LN_TUMOUR = re.compile(r"\b(?:tumou?rs?|cancers?|carcinomas?|malignan\w+|neoplas\w+|SCLCs?|NEPCs?|m?CRPCs?"
                        r"|metasta\w+|lesions?)\b", re.I)
_LN_PARTIAL = re.compile(r"\s*(?:\w+\s+){0,2}?in\s+(?:only\s+)?(\d+(?:\.\d+)?)\s*%\s+of\b", re.I)
_LN_ONLY_PCT = re.compile(r"(\bonly\s+(?:one|a\s+single|1|two|2)\b[^.;%]{0,40}?)(\d+(?:\.\d+)?\s*%)", re.I)
# a POSITIVE finding in a clause: the enumeration is then a result, not a
# screened-negative list (a cue preceded by a negator in the same clause, or
# followed by "no / a lack of", is itself negative and does not count)
LIST_POS_PATTERNS: tuple = (
    r"\b(?:up|down)[\s-]*regulat\w*", r"\bover[\s-]*express\w*", r"\bhighly\s+express\w*", r"\bexpressed\b",
    r"\belevat\w*", r"\bincreas\w*", r"\bdecreas\w*", r"\breduc\w*", r"\bhigher\b", r"\blower\b",
    r"\benrich\w*", r"\bsignificantly\b", r"\bdifferentially\b", r"\bidentified\b", r"\bfound\b",
    r"\bdetected\b", r"\bobserved\b", r"\bcorrelat\w*", r"\bpredict\w*", r"\bprognos\w*",
    # "associated" as a RESULT (a "genes associated with approved therapies"
    # description is not one)
    r"\b(?:was|were|is|are|be|been|being)\s+(?:\w+\s+){0,2}?associated\b",
    r"\bassociated\s+with\s+(?:\w+\s+){0,2}?(?:poor|worse|better|favou?rable|unfavou?rable|shorter|longer"
    r"|increased|decreased|higher|lower|reduced|metasta\w*|survival|prognos\w*|outcomes?|risk|recurrence"
    r"|response|resistance|progression|stage|grade|relapse)",
    r"\bassociations?\s+(?:was|were|between)\b",
    r"\bnominat\w*", r"\bpositive(?:ly)?\b", r"\bimmunoreactiv\w*",
    r"\b(?:mutated|altered|amplified|deleted|methylated|hypermethylated)\b", r"\brecurrent(?:ly)?\b",
    r"(?<!in)\bfrequent(?:ly)?\b", r"\binhibit\w*", r"\bsuppress\w*", r"\binduc\w*", r"\bpromot\w*",
    r"\bactivat\w*", r"\brepress\w*", r"\bgains?\b", r"\blosses\b", r"\bshowed\s+evidence\b",
    r"\bcolocali[sz]\w*", r"\breplicated\b", r"\bco-?exist\w*", r"\bco-?occur\w*", r"\bconcurrent\w*",
    r"(?<![\d.])(?!0+(?:\.0+)?\s*%)\d+(?:\.\d+)?\s*%",       # fix3: "0%" is no positive cue
    r"[\u2191\u2193]",
)
_LN_NEGATOR = re.compile(r"\b(?:no|not|none|nothing|never|neither|nor|without|cannot|failed\s+to)\b|n't\b"
                         r"|\bonly\s+(?:one|a\s+single|a|an|1|two|2)\b", re.I)
_LN_NEG_AFTER = re.compile(r"^\s*(?:(?:a|an|any)\s+)?(?:no\b|none\b|lack\s+of|absence\s+of|nothing\b"
                           r"|only\s+(?:one|a\s+single|a|an|1|two|2)\b)", re.I)
_LN_BREAKS = re.compile(
    r";|\bwhereas\b|\bwhile\b|\bbut\b|\balthough\b|\bthough\b|\bhowever\b|\bunlike\b|\bconversely\b"
    r"|\bin\s+contrast\b|,\s*(?=with\s+(?:no|the\s+exception)\b|without\b|except\b)|,?\s*\bas\s+well\s+as\b"
    r"|,?\s*\band\s+(?=(?:(?:were|was|are|is)\s+)?(?:(?:completely|entirely|uniformly)\s+)?"
    r"(?:negative|positive|immunoreactive|express\w*|lack\w*|show\w*|no\b|not\b))", re.I)
_LN_BACKREF = re.compile(
    r"\b(?:none|any|all|each|either|neither)\s+of\s+(?:these|those|them|the\s+(?:above|listed|tested|examined"
    r"|screened|selected|\w+\s+(?:genes?|markers?|loci)))\b"
    r"|\b(?:these|those|such)\s+(?:\w+\s+){0,2}?(?:genes?|loci|locus|markers?|targets?|candidates?|transcripts?"
    r"|proteins?|alterations?|mutations?|variants?|drivers?|regions?|factors?|molecules?)\b"
    r"|\bany\s+of\s+them\b|\ball\s+of\s+them\b|\bnone\s+of\s+them\b", re.I)
_LN_SCREEN = re.compile(
    r"\b(?:examin\w*|screen\w*|test(?:ed|ing)\b|evaluat\w*|assess\w*|analy[sz]\w*|sequenc\w*|genotyp\w*"
    r"|interrogat\w*|profil\w*|investigat\w*|quer(?:y|ied)|covered|cover(?:s|ing)\b|panel\b|included\b"
    r"|including\b|searched|looked\s+for|sought|checked)", re.I)
_SYM_UP: dict = {}


def _symbolish(tok, path=None):
    """Does this token look like a gene symbol (official, exact case; mouse
    style with a digit; or an NCBI alias with a capital)?"""
    ix = load_gene_index(path)
    p = _gi_path(path)
    if p not in _SYM_UP:
        _SYM_UP[p] = {s.upper() for s in ix["by_sym"]}
    if len(tok) < 2 or not re.search(r"[A-Z0-9]", tok):
        return False
    if tok in ix["by_sym"]:
        return True
    if re.search(r"\d", tok) and tok.upper() in _SYM_UP[p]:
        return True
    return len(tok) >= 3 and re.search(r"[A-Z]", tok) is not None and tok in ix["syn_owners"]


def _enumeration_at(t, start, path=None):
    """The run of >= 2 symbol-like tokens (separated by , ; / and or, or bare
    spaces) that contains the token starting at `start`: [(s, e, tok)] or []."""
    toks = [(m.start(), m.end(), m.group(0)) for m in _LN_TOK.finditer(t)
            if m.group(0).lower() not in ("and", "or")]      # "A, B and C": the conjunction separates
    k = next((i for i, x in enumerate(toks) if x[0] == start), None)
    if k is None or not _symbolish(toks[k][2], path):
        return []
    lo = hi = k
    while lo > 0 and _symbolish(toks[lo - 1][2], path) and _LN_SEP.match(t[toks[lo - 1][1]:toks[lo][0]]) \
            and toks[lo][0] > toks[lo - 1][1]:
        lo -= 1
    while hi < len(toks) - 1 and _symbolish(toks[hi + 1][2], path) \
            and _LN_SEP.match(t[toks[hi][1]:toks[hi + 1][0]]) and toks[hi + 1][0] > toks[hi][1]:
        hi += 1
    return toks[lo:hi + 1]


def _neg_matches(s):
    """The negative cues of `s` that can govern a list (fix3 2026-09-28): a
    cue whose clause is about normal tissue only, a comparative "no
    significant difference" that is not a tumour-vs-normal comparison, and a
    partial "not detected in 0.7% of ..." are left out.  Sorted by position."""
    out = []
    for comp, pats in ((False, LIST_NEG_PATTERNS), (True, LIST_NEG_COMPARATIVE)):
        for p in pats:
            for m in re.finditer(p, s, flags=re.I):
                c0, c1 = _clause_of(s, m.start(), m.end())
                cl = s[c0:c1]
                normal = _LN_NORMAL.search(cl) is not None
                tumour = _LN_TUMOUR.search(_LN_NONTUM.sub(" ", cl)) is not None
                if comp and not (normal and tumour):
                    continue
                if not comp and normal and not tumour:
                    continue
                pm = _LN_PARTIAL.match(s[m.end():m.end() + 40])
                if pm and float(pm.group(1)) < 50:
                    continue
                out.append(m)
    return sorted(out, key=lambda m: m.start())


def _neg_hit(s):
    ms = _neg_matches(s)
    return ms[0].group(0) if ms else ""


def _pos_hit(s, blank=None, region=None):
    """The first positive cue of `s` that is not itself negated (a negator
    earlier in the same clause, or "no / a lack of" right after it); "" when
    none.  `blank` = [(a, b)] spans of `s` not to read (the enumeration);
    `region` = (a, b): only cues starting there count (negation is still read
    from the whole of `s`)."""
    x = s
    for a, b in blank or ():
        x = x[:a] + " " * (b - a) + x[b:]
    # fix3: the "(2.5%)" of "Only 1 patient (2.5%) had a KRAS mutation" is not
    # a positive result
    x = _LN_ONLY_PCT.sub(lambda m: m.group(1) + " " * len(m.group(2)), x)
    for p in LIST_POS_PATTERNS:
        for m in re.finditer(p, x, flags=re.I):
            if region is not None and not (region[0] <= m.start() < region[1]):
                continue
            # a capitalised cue in mid-clause starts a new table cell / heading
            # ("... No gene alterations Upregulation MYCL1 SOX2"): an
            # earlier negator does not reach it
            _cell = m.group(0)[:1].isupper() and x[:m.start()].strip()[-1:] not in ("", ".", ":", ";")
            if not _cell and _LN_NEGATOR.search(x[:m.start()]):
                continue
            if _LN_NEG_AFTER.match(x[m.end():m.end() + 20]):
                continue
            return m.group(0)
    return ""


LIST_NEG_GOVERN_CHARS = 100


def _neg_governs(clause, a, b, blank=None):
    """(A) a negative cue of `clause` that GOVERNS the enumeration at [a, b):
    within LIST_NEG_GOVERN_CHARS characters of it, with no positive cue in
    between (a flattened table -- "No gene alterations Upregulation MYCL1 SOX2
    RET BCL2" -- is not a negative for the genes after it).  "" when none."""
    for m in _neg_matches(clause):
        if m.end() <= a:
            g0, g1 = m.end(), a
        elif m.start() >= b:
            g0, g1 = b, m.start()
        else:
            g0 = g1 = a
        if g1 - g0 > LIST_NEG_GOVERN_CHARS:
            continue
        if g1 > g0 and _pos_hit(clause, blank, (g0, g1)):
            continue
        return m.group(0)
    return ""


def _clause_of(sent, a, b):
    """The clause of `sent` holding the span [a, b) (the enumeration): the
    stretch between the nearest clause breaks outside the span."""
    lo, hi = 0, len(sent)
    for m in _LN_BREAKS.finditer(sent):
        if m.end() <= a:
            lo = max(lo, m.start())
        elif m.start() >= b:
            hi = min(hi, m.start())
            break
    return lo, hi


def list_negative(text, gene, gene_name="", path=None, min_items=None, param=None):
    """1c.  "" or why the passage does not count: EVERY mention of the gene is
    inside an enumeration of >= LIST_NEG_MIN_ITEMS gene symbols over which a
    negative result GOVERNS (fix2 2026-09-28, see the block above): (A) the
    enumeration's own clause reports a negative and no positive, or (B) the
    enumeration's sentence reports nothing and the next sentence reports a
    negative about the same list.  A list with a positive result -- a cohort /
    panel / signature association -- always counts (PI 2026-09-27).  Typical
    case: a cohort paper listing "... actionable mutations in genes associated
    with approved therapies ..., including KRAS, EGFR, BRAF, ROS1, MET, RET,
    ALK, NTRK1, ... We found only one KRAS mutation ... in this cohort." -- a
    screened-negative list that must not tie a listed gene to the domain."""
    if not list_negative_on(param) or not gene:
        return ""
    n_min = int(min_items or LIST_NEG_MIN_ITEMS)
    t = _flat(text)
    spans = _gene_spans(t, gene, gene_name, path)
    if not spans:
        return ""
    bounds = _sentences(t)
    why = ""
    for s, _e in spans:
        run = _enumeration_at(t, s, path)
        if len(run) < n_min:
            return ""                      # a mention outside a long list: the passage counts
        si = _sent_index(bounds, s)
        s0, s1 = bounds[si]
        sent = t[s0:s1]
        ea, eb = run[0][0] - s0, run[-1][1] - s0
        c0, c1 = _clause_of(sent, ea, eb)
        clause = sent[c0:c1]
        eblank = [(max(0, ea - c0), max(0, eb - c0))]
        neg = _neg_hit(clause)
        if neg:
            if _pos_hit(clause, eblank):
                return ""                  # (A) a positive result in the list's own clause: counts
            neg = _neg_governs(clause, ea - c0, eb - c0, eblank)
            if not neg:
                return ""                  # the negative is about something else
        else:
            if _pos_hit(sent, [(ea, eb)]) or _neg_hit(sent) or si + 1 >= len(bounds):
                return ""                  # the list's sentence reports something (or nothing follows)
            nxt = t[bounds[si + 1][0]:bounds[si + 1][1]]
            neg = _neg_hit(nxt)
            if not neg or _pos_hit(nxt):
                return ""
            members = {x[2] for x in run}
            others = [m.group(0) for m in _LN_TOK.finditer(nxt)
                      if _symbolish(m.group(0), path) and m.group(0) not in members]
            only = re.search(r"\bonly\s+(?:one|a\s+single|a|an|1|two|2)\b", nxt, re.I) is not None
            refers = (_LN_BACKREF.search(nxt) is not None
                      or (_LN_SCREEN.search(sent) is not None and (not others or only)))
            if not refers:
                return ""
        why = why or ("list-negative: %s is named only in an enumeration of %d genes (%s ...) with a "
                      "negative result (%r)" % (gene, len(run), ", ".join(x[2] for x in run[:4]), neg[:60]))
    return why


def retrieval_filters(spec, text, gene, gene_name="", title="", path=None):
    """1b + 1c for one domain's tie of one passage: "" or the reason."""
    if not gene:
        return ""
    w = nec_sentence_veto(spec, text, gene, gene_name, title, path)
    if w:
        return w
    t = _flat(text)
    tt = _flat(title)
    body = t[len(tt):].strip() if (tt and t.startswith(tt)) else t
    return list_negative(body, gene, gene_name, path)


# ---- 1a ---------------------------------------------------------------------
def _mesh_set(m):
    if isinstance(m, (list, tuple, set)):
        items = [str(x) for x in m]
    else:
        items = re.split(r"\s*;\s*", str(m or ""))
    return {re.sub(r"\s+", " ", x).strip().casefold() for x in items if str(x).strip()}


_CS_NEC_ABBR = r"(?:Pan|p|GEP|GI|SI|Si|Lu|Th)?-?NECs?"


def _control_blank(x, rx):
    """fix3: blank every clause of `x` that names a control (see
    literature.NEC_PAPER_CONTROL)."""
    if not rx:
        return x
    out = x
    for m in re.finditer(rx, x, flags=re.I):
        lo = max([x.rfind(c, 0, m.start()) for c in ".;:,"] + [-1]) + 1
        his = [h for h in (x.find(c, m.end()) for c in ".;:,") if h >= 0]
        hi = min(his) if his else len(x)
        out = out[:lo] + " " * (hi - lo) + out[hi:]
    return out


def paper_net_verdict(spec, rec, param=None):
    """1a.  None when the spec carries no paper rule (or the switch is off) or
    there is no record; else {"is_net", "net": [...], "hg": [...], "strong",
    "why"} -- a well-differentiated-tumour paper is is_net=True: a NET marker
    in its title / abstract / keywords / MeSH and no high-grade /
    carcinoma-type marker.
    fix2 2026-09-28 (verification finding 4): "non-islet cell tumour" is
    blanked first; the generic MeSH parent "Neuroendocrine Tumors" alone is no
    NET marker; and a paper whose TITLE names a neuroendocrine carcinoma needs
    a STRONG text marker (enterochromaffin / serotonin-producing, carcinoid,
    well-differentiated, low grade, G1/G2, SI-NET, insulinoma ...).
    fix3 2026-09-28 (second verification, finding 3): (i) markers in a
    contrast / comparison span ("in contrast to poorly differentiated
    neuroendocrine carcinomas, ...", "compared with well-differentiated NETs")
    are not the paper's own specimens and are blanked, and a high-grade marker
    in a clause naming a control ("the small cell lung cancer line H69 as
    control") too; (ii) THE OLD WORDING: a paper whose title or first two
    abstract sentences carry a NET-defined carcinoma phrase
    (literature.NEC_OLDWORD_DEFINING: "serotonin-producing neuroendocrine
    carcinomas, which arise from enterochromaffin cells", "midgut
    carcinoid-derived ...") is NET-type unless a STRONG high-grade marker of its
    own specimens is present -- a generic "high grade", a Ki-67 value or the
    abbreviation "NEC(s)" is not one ("defining", "hg_weak" in the result)."""
    pn = ((spec or {}).get("evfilter") or {}).get("paper_net")
    if not pn or not rec or not paper_net_on(param):
        return None
    txt = _flat(" . ".join(str(rec.get(k) or "") for k in ("title", "abstract", "keywords")))
    if not txt.strip(" .") and not rec.get("mesh"):
        return None
    mesh = _mesh_set(rec.get("mesh"))

    def _blank(x, pats):
        for p in pats or ():
            x = re.sub(p, lambda m: " " * len(m.group(0)), x, flags=re.I)
        return x
    tn = _blank(_blank(txt, pn.get("net_exclude")), pn.get("contrast"))
    net_t = []
    for p in pn.get("net_text") or ():
        m = re.search(p, tn, flags=re.I)
        if m:
            net_t.append(m.group(0)[:50])
    net_m = [x for x in (pn.get("net_mesh") or ()) if x.casefold() in mesh]
    gen = {x.casefold() for x in (pn.get("net_mesh_generic") or ())}
    why = ""
    if not net_t and net_m and all(x.casefold() in gen for x in net_m):
        why = "only the generic MeSH parent (%s)" % ", ".join(net_m)
        net_m = []
    net = net_t + ["MeSH:" + x for x in net_m]
    strong = []
    for p in pn.get("net_strong") or ():
        m = re.search(p, tn, flags=re.I)
        if m:
            strong.append(m.group(0)[:50])
    tb = _control_blank(_blank(_blank(txt, pn.get("hg_exclude")), pn.get("contrast")), pn.get("control"))
    weak_p = set(pn.get("hg_weak") or ())
    hg, hg_strong = [], []
    for p in pn.get("hg_text") or ():
        m = re.search(p, tb, flags=re.I)
        if m:
            hg.append(m.group(0)[:50])
            if p not in weak_p:
                hg_strong.append(m.group(0)[:50])
    hm = ["MeSH:" + x for x in (pn.get("hg_mesh") or ()) if x.casefold() in mesh]
    hg += hm
    hg_strong += hm
    # fix3: the old wording -- a NET-defined carcinoma phrase in the title or
    # the first two abstract sentences (contrast spans blanked)
    defining = []
    if pn.get("defining"):
        ab = _flat(rec.get("abstract") or "")
        bs = _sentences(ab)
        heads = [_flat(rec.get("title") or "")] + [ab[b:e] for b, e in bs[:2]]
        for h in heads:
            h = _blank(h, pn.get("contrast"))
            for p in pn.get("defining") or ():
                m = re.search(p, h, flags=re.I)
                if m:
                    defining.append(m.group(0)[:60])
                    break
    if defining:
        is_net = not hg_strong
        why = (("old wording: the paper's own tumour is a NET-type tumour called a carcinoma (%r); weak "
                "high-grade markers ignored: %s" % (defining[0], ", ".join(x for x in hg if x not in hg_strong)[:120]
                                                   or "-")) if is_net else
               "old-wording phrase (%r) but a strong high-grade marker (%s)" % (defining[0], hg_strong[0]))
        net = (["old wording: " + defining[0]] + net)
        return {"is_net": is_net, "net": net[:6], "hg": hg[:6], "strong": (strong or defining)[:6],
                "title_nec": "", "defining": defining[:3], "hg_strong": hg_strong[:6], "why": why}
    # fix3: a title naming a small-intestinal / midgut neuroendocrine carcinoma
    # -- NET-type unless a high-grade marker other than the NEC abbreviation
    # (contrast / control blanked) is present
    if pn.get("site_title"):
        ttl = _blank(_flat(rec.get("title") or ""), pn.get("contrast"))
        ms = next((m for p in pn.get("site_title") or () for m in [re.search(p, ttl, flags=re.I)] if m), None)
        if ms is not None:
            abbr = _CS_NEC_ABBR
            hg_site = [h for h in hg if not re.fullmatch(abbr, h.strip())]
            if not hg_site:
                return {"is_net": True, "net": (["old wording (site): " + ms.group(0)[:50]] + net)[:6], "hg": hg[:6],
                        "strong": strong[:6], "title_nec": "", "defining": [ms.group(0)[:60]],
                        "hg_strong": hg_strong[:6],
                        "why": ("old wording: the title names a small-intestinal / midgut neuroendocrine carcinoma "
                                "(%r) and no high-grade marker other than the abbreviation is present"
                                % ms.group(0)[:50])}
    is_net = bool(net) and not hg
    title_nec = ""
    if is_net and pn.get("nec_title"):
        tt = _blank(_blank(_flat(rec.get("title") or ""), pn.get("hg_exclude")), pn.get("nec_title_exclude"))
        m = next((m for p in pn.get("nec_title") or () for m in [re.search(p, tt, flags=re.I)] if m), None)
        if m is not None:
            title_nec = m.group(0)[:50]
            if not strong:
                is_net = False
                why = ("the title names a neuroendocrine carcinoma (%r) and no strong NET marker is present"
                       % title_nec)
    return {"is_net": is_net, "net": net[:6], "hg": hg[:6], "strong": strong[:6],
            "title_nec": title_nec, "defining": [], "hg_strong": hg_strong[:6], "why": why}


def fetch_paper_records(pmids, timeout=40, chunk=150):
    """{pmid: {"title", "abstract", "mesh": [...], "keywords", "year"}} for the
    PMIDs (efetch, cached per process), plus an error string ("" when every
    chunk came back).  Uses genegpt.call_api (shared NCBI throttle + key).
    NEVER raises."""
    want = [str(p) for p in dict.fromkeys(pmids or []) if str(p).isdigit()]
    todo = [p for p in want if p not in _PAPER_CACHE]
    errs = []
    if todo:
        try:
            import genegpt as _gg
        except Exception:
            try:
                sys.path.insert(0, os.environ.get("MRAG_GENEGPT_DIR") or GENEGPT_DIR)
                import genegpt as _gg
            except Exception as e:
                return {p: _PAPER_CACHE[p] for p in want if p in _PAPER_CACHE}, "genegpt not importable: %s" % str(e)[:120]
        import xml.etree.ElementTree as _ET
        for i in range(0, len(todo), int(chunk)):
            part = todo[i:i + int(chunk)]
            root = None
            # fix3 2026-09-28 (second verification, finding 3): one retry --
            # a failed fetch leaves the paper-level NET test unrun
            for _att in range(2):
                try:
                    _u = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                          "?db=pubmed&retmode=xml&id=" + ",".join(part))
                    # followup 2026-09-28 (fix 1): expect="xml" -- an incomplete / NCBI-error
                    # answer is retried by call_api like a network error (a call_api without
                    # the argument, e.g. a test stub, is called as before)
                    try:
                        xml = _gg.call_api(_u, timeout=int(timeout), expect="xml")
                    except TypeError as _te:
                        if "expect" not in str(_te):
                            raise
                        xml = _gg.call_api(_u, timeout=int(timeout))
                    root = _ET.fromstring(xml)
                    break
                except Exception as e:
                    if _att == 0:
                        time.sleep(3.0)
                        continue
                    errs.append("%s: %s" % (type(e).__name__, str(e)[:120]))
            if root is None:
                continue
            for a in root.iter("PubmedArticle"):
                pm = (a.findtext(".//MedlineCitation/PMID") or "").strip()
                if not pm:
                    continue
                at = a.find(".//ArticleTitle")
                title = _flat("".join(at.itertext())) if at is not None else ""
                ab = _flat(" ".join("".join(n.itertext()) for n in a.findall(".//Abstract/AbstractText")))
                mesh = [_flat(m.findtext("DescriptorName") or "") for m in a.findall(".//MeshHeading")]
                kw = _flat(" ; ".join("".join(k.itertext()) for k in a.findall(".//Keyword")))
                yr = a.findtext(".//JournalIssue/PubDate/Year") or a.findtext(".//JournalIssue/PubDate/MedlineDate") or ""
                _PAPER_CACHE[pm] = {"title": title, "abstract": ab, "mesh": [m for m in mesh if m],
                                    "keywords": kw, "year": str(yr)[:4]}
    out = {p: _PAPER_CACHE[p] for p in want if p in _PAPER_CACHE}
    return out, "; ".join(errs)[:300]


GENEGPT_DIR = "/BIO1/MRAG/Agent/libs/genegpt"


def _pn_specs(specs):
    return [sp for sp in (specs or []) if ((sp.get("evfilter") or {}).get("paper_net"))]


def paper_filter_map(generif_map, generif_pmids_map, vdb_map, domains, gene_info=None,
                     records=None, fetch=True, lit_symbol=None):
    """1a, run by the score runner BEFORE the gate: the PMIDs of every nec
    candidate tie (a hit naming the gene with a stored tie or corpus label for
    a paper-rule domain; a GeneRIF line naming the gene whose text mentions
    that domain), their PubMed records and verdicts.  Returns
    {"enabled", "version", "domains", "n_pmids", "n_records", "n_net",
     "net": {pmid: {"net", "hg", "title"}}, "error"} -- "net" is what the gate
    takes as paper_net.  `records` (pmid -> record) skips the fetch for those.
    NEVER raises."""
    out = {"enabled": paper_net_on(), "version": EVFILTER_VERSION, "domains": [], "n_pmids": 0,
           "n_records": 0, "n_net": 0, "net": {}, "error": "", "missing": []}
    try:
        dsp = _pn_specs(domain_specs(domains))
        out["domains"] = [sp["key"] for sp in dsp]
        if not out["enabled"] or not dsp:
            return out
        keys = {sp["key"] for sp in dsp}
        lm = lit_symbol or {}
        pm = []
        for g, hs in (vdb_map or {}).items():
            for h in hs or []:
                if not isinstance(h, dict) or not h.get("names_gene"):
                    continue
                dt = h.get("domain_tie") or {}
                if any(dt.get(k) for k in keys) or (_hit_corpora(h) & keys):
                    pm.append(str(h.get("pmid") or ""))
        for g, rifs in (generif_map or {}).items():
            q = lm.get(g, g)
            nm = gene_fullname(q, gene_info)
            pl = (generif_pmids_map or {}).get(g) or []
            for i, r in enumerate(rifs or []):
                txt = r if isinstance(r, str) else _hit_text(r)
                if not any(mentions_domain(txt, sp["terms"], sp.get("exclude") or ()) for sp in dsp):
                    continue
                if not (names_gene(txt, q, nm, gene_info, anchored=True)
                        or (generif_alias_on() and generif_alias(txt, q, nm, gene_info))):
                    continue
                pm.extend(str(x) for x in (pl[i] if i < len(pl) else []) or [])
        pm = [p for p in dict.fromkeys(pm) if p.isdigit()]
        out["n_pmids"] = len(pm)
        recs = dict(records or {})
        miss = [p for p in pm if p not in recs]
        if miss and fetch:
            got, err = fetch_paper_records(miss)
            recs.update(got)
            out["error"] = err
        out["n_records"] = sum(1 for p in pm if p in recs)
        # fix3 2026-09-28: the candidate PMIDs whose record could not be read
        # (their ties count unchecked -- flag_unchecked_papers marks the rows)
        out["missing"] = [p for p in pm if p not in recs]
        for p in pm:
            r = recs.get(p)
            if not r:
                continue
            for sp in dsp:
                v = paper_net_verdict(sp, r)
                if v and v["is_net"]:
                    out["net"][p] = {"net": v["net"], "hg": v["hg"], "title": str(r.get("title") or "")[:160],
                                     "domain": sp["key"]}
                    break
        out["n_net"] = len(out["net"])
    except Exception as e:
        out["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
    return out


def flag_unchecked_papers(rows, pf, generif_pmids_map=None):
    """fix3 2026-09-28 (second verification, finding 3): a row the gate passed
    on a counted item whose PubMed record the paper-level NET test could not
    read (paper_filter_map()["missing"], e.g. an NCBI outage) was counted
    UNCHECKED.  Such a row gets evidence_flag = "NEC paper check not run" (the
    report shows it) and evidence_flag_note naming the PMIDs.  Returns the
    number of rows flagged.  NEVER raises."""
    n = 0
    try:
        miss = {str(x) for x in ((pf or {}).get("missing") or [])}
        if not miss:
            return 0
        for r in rows or []:
            if str(r.get("evidence_gate") or "") != "pass":
                continue
            ec = r.get("evidence_counted") or {}
            pm = set()
            for x in ec.get("generif") or []:
                pm.update(str(y) for y in (x.get("pmids") or []))
            for x in ec.get("snippets") or []:
                pm.add(str(x.get("pmid") or ""))
            hit = sorted(pm & miss)
            if hit:
                r["evidence_flag"] = "NEC paper check not run"
                r["evidence_flag_note"] = ("the paper-level NET test could not read PubMed record(s) %s (%s); "
                                           "the evidence was counted unchecked"
                                           % (", ".join(hit[:5]), str((pf or {}).get("error") or "fetch failed")[:120]))
                n += 1
    except Exception:
        return n
    return n


# ---- GeneRIF alias (2) ------------------------------------------------------
def generif_alias(text, gene, gene_name="", path=None, param=None):
    """"alias:<A>" when a GeneRIF line of the gene's OWN record (genegpt reads
    the record of the gene's own Gene ID, E1 fix) names the gene only by an
    NCBI synonym A that is not any gene's official symbol and whose other
    owners (genes listing the same synonym) are not named in the same line;
    "" otherwise.  Example: a GeneRIF line that writes "OPG" for TNFRSF11B.
    A line the E1b rule ties to
    another gene (generif_tied_to_other) never counts."""
    if not generif_alias_on(param):
        return ""
    g = str(gene or "").strip()
    t = str(text or "")
    if not g or not t:
        return ""
    ix = load_gene_index(path)
    rec = ix["by_sym"].get(g) or {}
    syn = sorted((a for a in rec.get("syn", ()) if len(a) >= 3 and a not in ix["by_sym"]),
                 key=len, reverse=True)
    if not syn:
        return ""
    try:
        if generif_tied_to_other(t, g, gene_name, path):
            return ""
    except Exception:
        pass
    for a in syn:
        if not re.search(_WB % _pat(a), t):
            continue
        clash = False
        for o in ix["syn_owners"].get(a, ()):
            if o == g:
                continue
            onm = (ix["by_sym"].get(o) or {}).get("name", "")
            if re.search(_WB % _pat(o), t) or (len(onm) >= 5 and re.search(_WB % _pat(onm), t, re.I)):
                clash = True
                break
        if not clash:
            return "alias:" + a
    return ""


# ---------------------------------------------------------------------------
# does this passage name the gene?
# ---------------------------------------------------------------------------
def _pat(s):
    """Escaped, with runs of whitespace relaxed (full names carry spaces and
    snippets get their newlines flattened)."""
    return re.sub(r"(?:\\[ ])+", r"\\s+", re.escape(str(s)))


def names_gene_detail(text, gene, gene_name="", gene_info=None, strict=None, anchored=False):
    """names_gene() plus a note saying WHY a hit was rejected (for audits):
    returns (how, note)."""
    t = str(text or "")
    g = str(gene or "").strip()
    if not g or not t:
        return "", ""
    st = names_strict(strict)
    note = ""
    for how, fl in (("symbol", 0), ("symbol-ci", re.I)):
        if fl and not re.search(r"\d", g):
            continue
        spans = [m.span() for m in re.finditer(_WB % _pat(g), t, fl)]
        if not spans:
            continue
        if not st:
            return how, ""
        # E1b: a symbol hit the passage ties to ANOTHER gene is not this gene's
        try:
            kept, other = symbol_collision(t, g, gene_name, spans, ci=bool(fl),
                                           path=gene_info, anchored=anchored)
        except Exception:
            kept, other = spans, ""
        if kept:
            return how, ""
        note = (other[len("rejected: "):] if str(other).startswith("rejected: ")
                else "symbol tied to %s" % other)
        break
    nm = str(gene_name or "").strip()
    if len(nm) >= 5 and re.search(_WB % _pat(nm), t, re.I):
        if not st:
            return "full-name", ""
        how, n2 = fullname_decision(t, g, nm, gene_info, anchored)   # E3
        return how, ("; ".join(x for x in (note, n2) if x) if not how else n2)
    return "", note


def names_gene(text, gene, gene_name="", gene_info=None, strict=None, anchored=False):
    """Does this passage's OWN text name the gene?  "" or how it matched
    ("symbol" / "symbol-ci" / "full-name").  Case-AWARE on purpose -- see the
    module docstring.

    FIX 2026-09-25 (E1b / E3), both switchable with MRAG_NAMES_GENE_STRICT=0:
      * a symbol hit is NOT counted when the passage ties that symbol to
        another gene that lists it as an alias (symbol_collision): "chromogranin
        A (CGA)" does not name CGA;
      * a full-name hit is NOT counted when it is paired with another gene's
        symbol, sits inside another gene's longer name, or -- unpaired -- the
        name is shared with another gene and neither the symbol nor an
        unambiguous alias appears (fullname_decision): "NCAM (neuronal cell
        adhesion molecule)" does not name NRCAM."""
    return names_gene_detail(text, gene, gene_name, gene_info, strict, anchored)[0]


def _hit_ver(h):
    """The rule version a hit's stored domain decision was taken under (0 = none)."""
    try:
        return int(h.get("domain_tie_v") or 0) if isinstance(h, dict) else 0
    except (TypeError, ValueError):
        return 0


def _hit_title(h):
    return str(h.get("title") or "") if isinstance(h, dict) else ""


def _hit_text(h):
    if isinstance(h, str):
        return h
    if isinstance(h, dict):
        return " ".join(str(h.get(k) or "") for k in ("title", "snippet", "text", "gate_text"))
    return str(h or "")


def _hit_filter_text(h):
    """evaudit 2026-09-27: the text a STORED hit is re-filtered on (1b / 1c)
    when its decision predates the filters: title + the excerpt around the gene
    (the snippet when there is none)."""
    if not isinstance(h, dict):
        return str(h or "")
    ex = str(h.get("excerpt") or "").strip()
    ex = re.sub(r"^\.\.\.|\.\.\.$", "", ex).strip()
    return ex or str(h.get("snippet") or h.get("text") or h.get("gate_text") or "")


def _pick_tie(res):
    """domain_tie()'s precedence over [(how, key)]: first corpus, then first."""
    for how, k in res:
        if how == "corpus":
            return how, k
    for how, k in res:
        if how:
            return how, k
    return "", ""


def _filtered_ties(dsp, text, corpora, stored, ver, gene, nm, title, path,
                   fgene, fname, ftext, ftitle, pmids, paper_net, refilter):
    """Per-domain ties of one item after the evaudit filters.  Returns
    ([(how, key)], [(key, why)]) -- the second list says what a filter removed."""
    res, filt = [], []
    for sp in dsp:
        how = _tie_one(sp, text, corpora, stored, ver, gene, nm, title, path)
        if how and refilter and fgene:
            why = retrieval_filters(sp, (ftitle + " " + ftext) if ftitle else ftext, fgene, fname,
                                    ftitle, path)
            if why:
                filt.append((sp["key"], why))
                how = ""
        if how and paper_net and pmids and (sp.get("evfilter") or {}).get("paper_net"):
            pl = [str(x) for x in pmids if str(x)]
            if pl and all(p in paper_net for p in pl):
                v = paper_net[pl[0]] or {}
                filt.append((sp["key"], "paper-level NET: PMID %s is a well-differentiated-tumour "
                                        "paper (%s; no high-grade / carcinoma-type marker)"
                             % (",".join(pl), ", ".join((v.get("net") or [])[:3]))))
                how = ""
        res.append((how, sp["key"]))
    return res, filt


def evidence_naming(gene, generifs=None, vdb_hits=None, gene_name=None,
                    gene_info=None, domains=None, generif_pmids=None, paper_net=None):
    """Count the retrieved evidence that really names `gene`.

    generifs: list[str] (GeneRIF lines), vdb_hits: list[dict] with
    title/snippet (the PaperQA vectordb hits the runner kept).  Returns
        {gene_name, n_naming_generif, n_naming_snippets, n_generif, n_snippets,
         generif_kinds, snippet_kinds,
         n_domain_generif, n_domain_snippets, generif_ties, snippet_ties,
         domain_keys, domain_specs}
    n_naming_snippets counts LITERATURE snippets only; the GeneRIF count is
    kept separate so both halves stay auditable.  The n_domain_* counts are
    the AND scope="domain" enforces: that SAME passage names the gene and is
    tied to one of `domains` (see domain_tie).

    evaudit 2026-09-27 (any of MRAG_NEC_NET_PAPER / MRAG_NEC_VETO_SENTENCE /
    MRAG_LIST_NEGATIVE / MRAG_GENERIF_ALIAS on, the default): the evidence
    filters 1a-1c above remove ties (never add one), a GeneRIF line naming the
    gene by an unambiguous alias counts (generif_alias), and the result also
    carries "generif_counted" / "snippet_counted" (indices of the items the
    domain scope counted) and "filtered" (what a filter removed, and why).
    `generif_pmids` is the PMID list of each GeneRIF line (parallel to
    `generifs`), `paper_net` the {pmid: verdict} map of NET-only papers
    (paper_filter_map()["net"]).  With all four switches off this is the
    pre-2026-09-27 function, byte for byte (_evidence_naming_v1)."""
    if not evidence_layer_on():
        return _evidence_naming_v1(gene, generifs, vdb_hits, gene_name, gene_info, domains)
    nm = gene_fullname(gene, gene_info) if gene_name is None else (gene_name or "")
    dsp = domain_specs(domains)
    gp = list(generif_pmids or [])
    pn = paper_net or {}
    rk, sk, rt, st, dk = [], [], [], [], set()
    rc, sc, filt = [], [], []
    for i, r in enumerate(generifs or []):
        txt = r if isinstance(r, str) else _hit_text(r)
        # a GeneRIF line is anchored to its gene by NCBI: the literature-sense
        # test (round 2, c) does not apply to it
        how = names_gene(txt, gene, nm, gene_info, anchored=True)
        if not how:
            how = generif_alias(txt, gene, nm, gene_info)
        rk.append(how)
        tie = ""
        if dsp:
            res, fl = _filtered_ties(dsp, txt, set(), None, False, None, "", None, gene_info,
                                     gene, nm, txt, "", (gp[i] if i < len(gp) else []) or [], pn, True)
            tie, key = _pick_tie(res)
            if how:
                filt += [{"kind": "generif", "i": i, "pmid": ",".join(gp[i] if i < len(gp) else []),
                          "key": k, "why": w} for k, w in fl]
                if tie:
                    dk.add(key)
                    rc.append(i)
        rt.append(tie)
    for j, h in enumerate(vdb_hits or []):
        if isinstance(h, dict) and "names_gene" in h:
            how = str(h.get("names_gene") or "")
        else:
            how = names_gene(_hit_text(h), gene, nm, gene_info)
        sk.append(how)
        tie = ""
        if dsp:
            ver = _hit_ver(h)
            res, fl = _filtered_ties(dsp, _hit_text(h), _hit_corpora(h),
                                     h.get("domain_tie") if isinstance(h, dict) else None,
                                     ver, gene, nm, _hit_title(h), gene_info,
                                     gene, nm, _hit_filter_text(h), _hit_title(h),
                                     [str(h.get("pmid") or "")] if isinstance(h, dict) else [],
                                     pn, ver < EVFILTER_TIE_VERSION)
            tie, key = _pick_tie(res)
            if how:
                filt += [{"kind": "snippet", "i": j,
                          "pmid": str(h.get("pmid") or "") if isinstance(h, dict) else "",
                          "key": k, "why": w} for k, w in fl]
                if isinstance(h, dict) and ver >= EVFILTER_TIE_VERSION:
                    for k, w in (h.get("domain_tie_note") or {}).items():
                        if str(w).startswith(("sentence veto", "list-negative")):
                            filt.append({"kind": "snippet", "i": j, "pmid": str(h.get("pmid") or ""),
                                         "key": k, "why": w, "at": "retrieval"})
                if tie:
                    dk.add(key)
                    sc.append(j)
        st.append(tie)
    return {"gene_name": nm,
            "n_generif": len(rk), "n_snippets": len(sk),
            "n_naming_generif": sum(1 for k in rk if k),
            "n_naming_snippets": sum(1 for k in sk if k),
            "generif_kinds": rk, "snippet_kinds": sk,
            "n_domain_generif": sum(1 for k, t in zip(rk, rt) if k and t),
            "n_domain_snippets": sum(1 for k, t in zip(sk, st) if k and t),
            "generif_ties": rt, "snippet_ties": st,
            "domain_keys": sorted(k for k in dk if k) if dsp else [], "domain_specs": dsp,
            "generif_counted": rc, "snippet_counted": sc, "filtered": filt}


def _evidence_naming_v1(gene, generifs=None, vdb_hits=None, gene_name=None,
                        gene_info=None, domains=None):
    """Count the retrieved evidence that really names `gene`.

    generifs: list[str] (GeneRIF lines), vdb_hits: list[dict] with
    title/snippet (the PaperQA vectordb hits the runner kept).  Returns
        {gene_name, n_naming_generif, n_naming_snippets, n_generif, n_snippets,
         generif_kinds, snippet_kinds,
         n_domain_generif, n_domain_snippets, generif_ties, snippet_ties,
         domain_keys, domain_specs}
    n_naming_snippets counts LITERATURE snippets only; the GeneRIF count is
    kept separate so both halves stay auditable.  The n_domain_* counts are
    the AND scope="domain" enforces: that SAME passage names the gene and is
    tied to one of `domains` (see domain_tie)."""
    nm = gene_fullname(gene, gene_info) if gene_name is None else (gene_name or "")
    dsp = domain_specs(domains)
    rk, sk, rt, st = [], [], [], []
    for r in (generifs or []):
        txt = r if isinstance(r, str) else _hit_text(r)
        # a GeneRIF line is anchored to its gene by NCBI: the literature-sense
        # test (round 2, c) does not apply to it
        rk.append(names_gene(txt, gene, nm, gene_info, anchored=True))
        # A GeneRIF is not corpus-filtered, so only its own words can tie it.
        rt.append(domain_tie(txt, dsp)[0] if dsp else "")
    for h in (vdb_hits or []):
        # Prefer a decision the retriever already took on the FULL passage
        # (2026-09-23 fix).  Only title[:120] + text[:220] is persisted, so
        # re-matching the stored snippet here UNDERCOUNTS: the symbol often sits
        # further into the passage, and the gate then fired on genes the
        # evidence really does name.  A stored "" is a real "does not name it",
        # so the key's presence -- not its truth -- decides.
        if isinstance(h, dict) and "names_gene" in h:
            sk.append(str(h.get("names_gene") or ""))
        else:
            sk.append(names_gene(_hit_text(h), gene, nm, gene_info))
        st.append(domain_tie(_hit_text(h), dsp, _hit_corpora(h),
                             h.get("domain_tie") if isinstance(h, dict) else None,
                             _hit_ver(h), gene, nm, _hit_title(h), gene_info)[0]
                  if dsp else "")
    dk = sorted({domain_tie(_hit_text(h), dsp, _hit_corpora(h),
                            h.get("domain_tie") if isinstance(h, dict) else None,
                            _hit_ver(h), gene, nm, _hit_title(h), gene_info)[1]
                 for h, k in zip(vdb_hits or [], sk) if k and dsp} |
                {domain_tie(r if isinstance(r, str) else _hit_text(r), dsp)[1]
                 for r, k in zip(generifs or [], rk) if k and dsp}) if dsp else []
    return {"gene_name": nm,
            "n_generif": len(rk), "n_snippets": len(sk),
            "n_naming_generif": sum(1 for k in rk if k),
            "n_naming_snippets": sum(1 for k in sk if k),
            "generif_kinds": rk, "snippet_kinds": sk,
            "n_domain_generif": sum(1 for k, t in zip(rk, rt) if k and t),
            "n_domain_snippets": sum(1 for k, t in zip(sk, st) if k and t),
            "generif_ties": rt, "snippet_ties": st,
            "domain_keys": [k for k in dk if k], "domain_specs": dsp}


def gate_reason(gene, nm="", scope="all", domains=""):
    """The deterministic sentence that replaces an ungrounded LLM reason."""
    if scope == "domain":
        return ("No retrieved evidence ties %s to %s: no GeneRIF line and no "
                "literature passage shown to the scorer both mentions this gene "
                "and belongs to this domain, so no domain-specific literature "
                "support is claimed." % (gene, domains or "the run's literature domain"))
    if scope == "literature":
        return ("No retrieved literature snippet names %s: none of the PaperQA "
                "passages shown to the scorer mentions this gene, so no "
                "literature support is claimed." % gene)
    return ("No retrieved evidence names %s: none of the GeneRIF lines or "
            "literature snippets shown to the scorer mentions this gene, so no "
            "literature support is claimed." % gene)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def apply_gate(row, generifs=None, vdb_hits=None, cap=None, enabled=None,
               gene_info=None, scope=None, domains=None, generif_pmids=None, paper_net=None):
    """Enforce the gate on ONE scored row, in place, and return it.

    Runs AFTER the LLM has answered, over the evidence already retrieved.  When
    no GeneRIF line and no literature snippet names the gene, the score may not
    claim literature support: it is capped at `cap` (rubric default 2),
    has_evidence is forced false and the reason is replaced by a deterministic
    sentence (the model's own wording is kept as reason_llm).

    Recorded on every row, gate on or off:
        score_llm, n_naming_generif, n_naming_snippets, evidence_gate
        ("fired" / "pass" / "off" / "no-domain"), evidence_gate_cap,
        evidence_gate_scope, gene_full_name -- plus, under scope="domain",
        n_domain_generif, n_domain_snippets, evidence_gate_domains and the
        domains that grounded it (evidence_gate_domain_hit).
    Only the literature score component is touched -- the candidate pool, the
    gate.json step, motif/AG evidence and the ranking rule are untouched."""
    gene = row.get("gene", "")
    if evidence_layer_on():
        ev = evidence_naming(gene, generifs, vdb_hits, gene_info=gene_info,
                             domains=domains, generif_pmids=generif_pmids, paper_net=paper_net)
    else:
        ev = evidence_naming(gene, generifs, vdb_hits, gene_info=gene_info,
                             domains=domains)
    on = gate_enabled(enabled)
    cap_v = gate_cap(cap)
    sc = gate_scope(scope)
    try:
        orig = float(row.get("score") or 0)
    except (TypeError, ValueError):
        orig = 0.0
    row["score_llm"] = row.get("score", 0)
    row["n_naming_generif"] = ev["n_naming_generif"]
    row["n_naming_snippets"] = ev["n_naming_snippets"]
    row["gene_full_name"] = ev["gene_name"]
    row["evidence_gate_cap"] = cap_v
    row["evidence_gate_scope"] = sc
    if sc == "domain":
        row["n_domain_generif"] = ev["n_domain_generif"]
        row["n_domain_snippets"] = ev["n_domain_snippets"]
        row["evidence_gate_domains"] = [sp["key"] for sp in ev["domain_specs"]]
        row["evidence_gate_domain_hit"] = ev["domain_keys"]
        if "generif_counted" in ev:
            # evaudit 2026-09-27: WHICH items the domain scope counted (indices
            # into scored.json generif[gene] / vdb[gene]) and what the evidence
            # filters removed -- the evidence audit reads exactly these
            _gp = list(generif_pmids or [])
            row["evidence_counted"] = {
                "v": EVFILTER_VERSION,
                "generif": [{"i": i, "pmids": [str(x) for x in ((_gp[i] if i < len(_gp) else []) or [])],
                             "how": ev["generif_kinds"][i]} for i in ev["generif_counted"]],
                "snippets": [{"i": j, "pmid": str((vdb_hits[j] or {}).get("pmid") or "")
                              if isinstance(vdb_hits[j], dict) else ""} for j in ev["snippet_counted"]],
                "filtered": ev["filtered"]}
        by_snip = bool(ev["n_domain_snippets"])
        by_rif = bool(ev["n_domain_generif"])
    elif sc == "literature":
        by_snip = bool(ev["n_naming_snippets"])
        by_rif = False
    else:
        by_snip = bool(ev["n_naming_snippets"])
        by_rif = bool(ev["n_naming_generif"])
    grounded = by_snip or by_rif
    # FIX 2026-09-23 (review): record WHICH evidence grounded the row, not just
    # that something did.  Under the shipped scope="all" default a row can be
    # grounded by a GeneRIF line alone -- no indexed passage names the gene and
    # no domain test is run at all -- and the report then printed "indexed
    # corpus", asserting a provenance the run never established.  "corpus" now
    # means an indexed literature passage named the gene; "generif" means only
    # an NCBI GeneRIF line did.
    src = "corpus" if by_snip else ("generif" if by_rif else "")
    if not on:
        row["evidence_gate"] = "off"
        row["evidence_source"] = src
        return row
    if sc == "domain" and not ev["domain_specs"]:
        # The domain test could not be run (no resolvable domain).  Capping
        # here would be a claim about evidence nothing looked at.
        row["evidence_gate"] = "no-domain"
        row["evidence_source"] = ""
        return row
    if grounded:
        row["evidence_gate"] = "pass"
        # 2026-09-23: where the grounding came from, so a report can tell an
        # indexed passage apart from a GeneRIF-only grounding and from the
        # live-PubMed rescue below.
        row["evidence_source"] = src
        return row
    row["evidence_gate"] = "fired"
    row["evidence_source"] = ""
    if orig > cap_v:
        row["score"] = cap_v if cap_v % 1 else int(cap_v)
    row["has_evidence"] = False
    if row.get("reason"):
        row["reason_llm"] = row["reason"]
    row["reason"] = gate_reason(gene, ev["gene_name"], sc,
                                domains_label(ev["domain_specs"]))
    return row


def apply_gate_rows(rows, generif_map=None, vdb_map=None, cap=None,
                    enabled=None, gene_info=None, scope=None, domains=None,
                    generif_pmids_map=None, paper_net=None):
    """apply_gate over a list of scored rows.  Returns a summary dict
    {n, n_fired, n_capped, capped:[(gene, before, after), ...], scope,
     domains, error}.  `error` is set when scope="domain" was asked for but no
     domain could be resolved -- no score is changed in that case."""
    gm = generif_map or {}
    vm = vdb_map or {}
    sc = gate_scope(scope)
    dsp = domain_specs(domains) if sc == "domain" else []
    fired, capped, nodom = 0, [], 0
    for r in rows:
        g = r.get("gene", "")
        before = r.get("score", 0)
        if evidence_layer_on():
            apply_gate(r, gm.get(g), vm.get(g), cap=cap, enabled=enabled,
                       gene_info=gene_info, scope=scope, domains=dsp or domains,
                       generif_pmids=(generif_pmids_map or {}).get(g), paper_net=paper_net)
        else:
            apply_gate(r, gm.get(g), vm.get(g), cap=cap, enabled=enabled,
                       gene_info=gene_info, scope=scope, domains=dsp or domains)
        if r.get("evidence_gate") == "no-domain":
            nodom += 1
        if r.get("evidence_gate") == "fired":
            fired += 1
            if r.get("score") != before:
                capped.append((g, before, r.get("score")))
    out = {"n": len(rows), "n_fired": fired, "n_capped": len(capped),
           "capped": capped, "scope": sc,
           "domains": [sp["key"] for sp in dsp]}
    if nodom:
        out["error"] = ("scope=domain but no literature domain could be "
                        "resolved; %d row(s) left untouched" % nodom)
    return out


# ---------------------------------------------------------------------------
# live PubMed rescue (2026-09-23)
# ---------------------------------------------------------------------------
# WHY.  The corpora are snapshots of very different sizes (prostate ~3.8k,
# sclc ~3.0k, nec ~0.8k parsed papers), so under scope="domain" a gene can be
# capped because OUR INDEX is thin, not because the literature is.  Measured
# over the 249 curated genes x {nec, sclc}: a scoped live check gives a
# VERIFIED record for 62 (NEC) and 47 (SCLC) genes, against 9 and 7 from the
# corpus -- and it is a strict superset, no gene loses its corpus evidence.
#
# "Verified" is the Section-14 bar, not an esearch count: the records are
# fetched back and each one must (a) name the gene (_record_names_gene,
# case-aware) and (b) pass the SAME DOMAIN_TERMS / DOMAIN_EXCLUDE test an
# indexed passage must pass.  esearch counts query matches, not gene mentions
# -- IMPACT/SCLC returns 988 records because SCLC reviews use the word
# "impact" -- so the count alone may never lift a score.
#
# SCOPE.  Only a row the gate actually CAPPED is checked (evidence_gate ==
# "fired" AND the LLM's own score was above the cap).  A run whose gate caps
# nothing makes zero NCBI calls.  Measured cost: ~13 s per domain per run.
#
# The rescued score is recorded DISTINCTLY: evidence_source="pubmed",
# evidence_gate="pubmed", and row["pubmed_evidence"] carries the PMIDs, the
# term, how each record matched and the date, so a report can always tell the
# two kinds of evidence apart.
NARRATIVE_PY = ("/BIO1/MRAG/Agent/mcps/viz_report_mcp/src/viz_report_mcp/"
                "tools/narrative.py")
_NARR_MOD: list = [None]


def _narrative(path=None):
    """viz_report_mcp.tools.narrative, loaded BY FILE PATH so this module keeps
    working when it is exec'd standalone into the paper_qa venv.  None when the
    module cannot be loaded -- the caller then simply does not rescue."""
    if _NARR_MOD[0] is None:
        try:
            import importlib
            p = path or os.environ.get("MRAG_NARRATIVE_PY") or NARRATIVE_PY
            # narrative.py uses relative imports ("from .narrative_harness
            # import ..."), so it has to be imported AS A PACKAGE MODULE -- a
            # bare spec_from_file_location load dies with "attempted relative
            # import with no known parent package".  Putting its src root on
            # sys.path is enough; viz_report_mcp never imports this package, so
            # there is no cycle.
            root = str(Path(p).resolve().parents[3])   # .../viz_report_mcp/src
            if root not in sys.path:
                sys.path.insert(0, root)
            _NARR_MOD[0] = importlib.import_module("viz_report_mcp.tools.narrative")
        except Exception as e:
            print("[evidence_gate] narrative not loadable (%s); the live "
                  "PubMed rescue is skipped" % str(e)[:160], flush=True)
            _NARR_MOD[0] = False
    return _NARR_MOD[0] or None


def _verify_takes_cooc(N):
    """Does this narrative._pubmed_verify accept the round-2 `cooc` argument?"""
    try:
        import inspect
        return "cooc" in inspect.signature(N._pubmed_verify).parameters
    except Exception:
        return False


def _verify_takes_evfilter(N):
    """Does this narrative._pubmed_verify accept the evaudit `evfilter` argument?"""
    try:
        import inspect
        return "evfilter" in inspect.signature(N._pubmed_verify).parameters
    except Exception:
        return False


def record_filters(spec, rec, gene, gene_name="", path=None):
    """evaudit 2026-09-27: 1a + 1b + 1c for one PubMed record (title,
    abstract, mesh, keywords) the rescue verifier accepted: "" or the reason
    it does not count."""
    v = paper_net_verdict(spec, rec)
    if v and v["is_net"]:
        return ("paper-level NET: a well-differentiated-tumour paper (%s; no high-grade / "
                "carcinoma-type marker)" % ", ".join(v["net"][:3]))
    title = str(rec.get("title") or "")
    ab = str(rec.get("abstract") or "")
    w = nec_sentence_veto(spec, title + " " + ab, gene, gene_name, title, path)
    if w:
        return w
    return list_negative(ab, gene, gene_name, path)


def rescue_enabled(param=None):
    """Live-PubMed rescue on unless switched off.  Param wins over
    MRAG_TARGET_EVIDENCE_PUBMED; safe behaviour (on) is the default of both."""
    if param is not None:
        return bool(param)
    return _env_flag("MRAG_TARGET_EVIDENCE_PUBMED", True)


RESCUE_RETMAX = 20          # deploy 2026-09-28 (decision 4); was 5
RESCUE_RETMAX_LEGACY = 5


def rescue_retmax(param=None):
    """PMIDs read per (gene, domain) esearch of the live-PubMed rescue; param wins over
    MRAG_TARGET_EVIDENCE_PUBMED_RETMAX.  Default 20 (deploy 2026-09-28): esearch's relevance
    order is not stable at the margin -- the query for one SCLC target returned its verifying
    record in the top 5 in only 22 of 30 identical calls (evaudit bench4) -- so with 5 the rescue
    flapped; with 20 all 7 hits are read and no other FINAL top-20 gene changed.  With the
    evidence layers off (MRAG_EVIDENCE_AUDIT=0 / MRAG_EVIDENCE_DEFENCE=0, the rollback) the
    old 5."""
    for v in (param, os.environ.get("MRAG_TARGET_EVIDENCE_PUBMED_RETMAX")):
        if v is None or str(v).strip() == "":
            continue
        try:
            return max(1, int(v))
        except (TypeError, ValueError):
            pass
    return RESCUE_RETMAX if evidence_defence_on() else RESCUE_RETMAX_LEGACY


def pubmed_specs(domains):
    """Like domain_specs(), but KEEPS the PubMed query fragment: the rescue
    needs literature.DOMAIN_PUBMED[key], which domain_specs() drops.  Only
    specs that carry BOTH terms and a pubmed fragment can be checked."""
    out = []
    raw = domains
    if isinstance(raw, dict):
        raw = [raw]
    elif isinstance(raw, str):
        raw = [x for x in re.split(r"[,\s]+", raw) if x]
    bykey = {}
    for it in (raw or []):
        if isinstance(it, dict) and it.get("pubmed"):
            bykey[str(it.get("key") or "").strip().lower()] = str(it["pubmed"])
    if not bykey:
        L = _literature()
        keys = []
        for it in (raw or []):
            keys.append(str(it.get("key") if isinstance(it, dict) else it or "").strip().lower())
        if L is not None:
            for k in keys:
                frag = (getattr(L, "DOMAIN_PUBMED", {}) or {}).get(k, "")
                if frag:
                    bykey[k] = frag
    for sp in domain_specs(domains):
        frag = bykey.get(sp["key"], "")
        if frag:
            out.append(dict(sp, pubmed=frag))
    return out


# ROUND 2 (2026-09-26): the live-PubMed RESCUE needs gene-domain
# co-occurrence too.  Verifier finding, re-checked: after round 1 one gene still
# reached the SCLC top 20 through a record whose only SCLC words are a cohort
# list ("56 patients with small cell lung cancer (SCLC)") while the gene
# appears only among "SQC-specific up-regulated genes" -- the
# record names the gene and the domain, but never together.  For the domains
# below a rescue record now needs its TITLE or author KEYWORDS to name the
# domain, or ONE SENTENCE of the abstract to name both the gene and the domain
# (window 0; the corpus-chunk rule keeps window 1, which a genuine
# SCLC-marker chunk needs).  Default "sclc,prostate",
# chosen on data: every record today's fixed verifier accepts for the
# rescue candidates of the three 09-25 runs + the verifier's live panel (147
# records) --
#   sclc      29 verified, the rule drops 2, both spurious: a cohort list and
#             "the DCC/HOPfp method" (= dicyclohexylcarbodiimide);
#   prostate  55 verified, drops 4, all spurious: "density cushion
#             centrifugation", an AML paper, an ovarian-carcinoma paper and a
#             pan-cancer list;
#   nec       18 verified, drops 3 incl. a genuine medullary-thyroid-carcinoma
#             record (domain via MeSH only) -- so nec is NOT in the default.
# The Section-14 cards call _pubmed_verify without it (a stricter verifier
# there would turn records into novelty claims).  MRAG_RESCUE_COOC = keys |
# all | exclude | none.
RESCUE_COOC_DEFAULT = "sclc,prostate"
RESCUE_COOC_WINDOW = 0


def rescue_cooc_required(spec, param=None):
    v = param if param is not None else os.environ.get("MRAG_RESCUE_COOC")
    v = str(v if v is not None and str(v).strip() else RESCUE_COOC_DEFAULT).strip().lower()
    if v == "all":
        return True
    if v in ("none", "0", "off", "false", "no"):
        return False
    if v == "exclude":
        return bool((spec or {}).get("exclude"))
    return str((spec or {}).get("key") or "").lower() in {x.strip() for x in v.split(",")}


def pubmed_rescue(rows, domains, cap=None, enabled=None, retmax=None,
                  gene_info=None, lit_symbol=None, narrative_py=None,
                  progress_every=25):
    """Live-PubMed rescue for the rows this gate CAPPED.  Mutates `rows` in
    place and returns a summary dict:
        {checked, rescued, n_calls, seconds, per_gene:{gene:{...}},
         rescued_genes:[...], domains:[key], enabled, error}

    A gene is a candidate iff evidence_gate == "fired" AND its own LLM score
    was above the cap -- i.e. the gate really took points away.  Never raises:
    any failure leaves every score exactly as the gate left it."""
    t0 = time.time()
    out = {"checked": 0, "rescued": 0, "n_calls": 0, "seconds": 0.0,
           "per_gene": {}, "rescued_genes": [], "domains": [],
           "enabled": bool(rescue_enabled(enabled)), "error": ""}
    if not out["enabled"]:
        return out
    cap_v = gate_cap(cap)
    rm = rescue_retmax(retmax)
    specs = pubmed_specs(domains)
    out["domains"] = [sp["key"] for sp in specs]
    if not specs:
        out["error"] = ("no domain carries a PubMed query fragment; nothing "
                        "was checked and no score was changed")
        return out
    cand = []
    for r in rows:
        if r.get("evidence_gate") != "fired":
            continue
        try:
            llm = float(r.get("score_llm") if r.get("score_llm") is not None
                        else r.get("score") or 0)
        except (TypeError, ValueError):
            llm = 0.0
        if llm > cap_v:
            cand.append((r, llm))
    if not cand:
        return out
    # FIX 2026-09-23 (review): a live check that FAILED is not an absence, so
    # the failures are counted here and surfaced in the summary, the log line
    # and the row -- see out["n_errors"] / evidence_source="pubmed-error".
    n_pairs = n_err_all = n_gene_err = 0
    N = _narrative(narrative_py)
    if N is None:
        out["error"] = "narrative module not loadable; no live check was run"
        return out
    lm = lit_symbol or {}

    def _check(q, sp, nm):
        res = N._pubmed_domain_check(q, sp["pubmed"], retmax=rm, gene_name=nm)
        _cw = RESCUE_COOC_WINDOW if rescue_cooc_required(sp) else None
        # evaudit 2026-09-27: the evidence filters (1a-1c) judge each
        # record the verifier accepted (followup 2026-09-28, fix 2: the
        # Section-14 cards of a curated run now do the same, without cooc)
        _ef = (evidence_layer_on() and _verify_takes_evfilter(N))
        if _ef:
            return N._pubmed_verify(q, sp, res, cooc=_cw, evfilter=True)
        if _cw is not None and _verify_takes_cooc(N):
            return N._pubmed_verify(q, sp, res, cooc=_cw)
        return N._pubmed_verify(q, sp, res)

    def _failed(res):
        return (not res.get("ok")) or bool(res.get("error")) or (
            int(res.get("count") or 0) > 0 and not res.get("verify_ok") and bool(res.get("verify_error")))
    for i, (r, llm) in enumerate(cand):
        g = str(r.get("gene") or "")
        if not g:
            continue
        q = lm.get(g, g)          # search the HUMAN symbol (ortholog route)
        out["checked"] += 1
        try:
            nm = N._gene_fullname(q)
        except Exception:
            nm = ""
        per, ver_any, n_err = {}, [], 0
        n_pairs += len(specs)
        for sp in specs:
            # followup 2026-09-28 (fix 1): a check that FAILED (NCBI error after call_api's own
            # retries) is asked ONCE more after MRAG_RESCUE_RETRY_PAUSE_S (10 s) -- a failed
            # result is no longer cached -- before it counts as failed
            _retried = False
            try:
                try:
                    res = _check(q, sp, nm)
                    _bad = _failed(res)
                except Exception:
                    _bad = True
                    res = None
                if _bad:
                    _retried = True
                    try:
                        _pz = float(os.environ.get("MRAG_RESCUE_RETRY_PAUSE_S") or 10)
                    except ValueError:
                        _pz = 10.0
                    time.sleep(max(0.0, _pz))
                    res = _check(q, sp, nm)
            except Exception as e:
                per[sp["key"]] = {"ok": False, "error": str(e)[:160],
                                  "count": 0, "verified": []}
                if _retried:
                    per[sp["key"]]["retried"] = True
                n_err += 1
                continue
            out["n_calls"] += 1
            per[sp["key"]] = {"ok": bool(res.get("ok")), "count": int(res.get("count") or 0),
                              "checked": list(res.get("checked") or []),
                              "verified": list(res.get("verified") or []),
                              "verify_how": dict(res.get("verify_how") or {}),
                              "term": res.get("term", ""), "date": res.get("date", ""),
                              "gene_name": res.get("gene_name", ""),
                              "error": res.get("error", "") or res.get("verify_error", ""),
                              "cooc_rejected": list(res.get("verify_cooc_rejected") or [])}
            if "verify_filter_rejected" in res:
                # evaudit 2026-09-27: records the evidence filters refused, and
                # the verified records' text for the evidence audit
                per[sp["key"]]["filter_rejected"] = dict(res.get("verify_filter_rejected") or {})
                per[sp["key"]]["records"] = dict(res.get("verify_records") or {})
            if _retried:
                per[sp["key"]]["retried"] = True
            # Both helpers swallow their own network errors and hand back
            # ok=False / verify_error instead of raising, so without this count
            # a timed-out NCBI reads exactly like "checked and found nothing".
            if per[sp["key"]]["error"] or not per[sp["key"]]["ok"]:
                n_err += 1
            ver_any.extend(res.get("verified") or [])
        n_err_all += n_err
        # FIX 2026-09-23 (review): the per-domain truth, kept next to the pooled
        # list.  A gene verified in ONE domain of a multi-domain run must not be
        # presented -- in the reason or in the report cell -- as verified in all
        # of them.
        ver_specs = [sp for sp in specs if (per.get(sp["key"]) or {}).get("verified")]
        rec = {"searched_as": q, "gene_name": nm, "retmax": rm,
               "n_verified": len(ver_any), "pmids": sorted(set(ver_any), key=int),
               "by_domain": per, "date": time.strftime("%Y-%m-%d"),
               "n_errors": n_err, "n_domains": len(specs),
               "verified_domains": [sp["key"] for sp in ver_specs],
               "verified_by_domain": [
                   {"key": sp["key"], "label": sp.get("label") or sp["key"],
                    "pmids": sorted(set(per[sp["key"]]["verified"]), key=int)}
                   for sp in ver_specs]}
        r["pubmed_evidence"] = rec
        out["per_gene"][g] = {"n_verified": len(ver_any), "pmids": rec["pmids"],
                              "verified_domains": rec["verified_domains"],
                              "n_errors": n_err}
        if ver_any:
            # The gate's own verdict is preserved next to the restored score:
            # a reader must be able to see that the corpus said nothing.
            r["evidence_gate_corpus"] = "fired"
            r["reason_gate"] = r.get("reason", "")
            r["evidence_gate"] = "pubmed"
            r["evidence_source"] = "pubmed"
            r["has_evidence"] = True
            r["score"] = r.get("score_llm", r.get("score"))
            # The lead clause states what the CORPUS said, so it may only
            # mention a domain when a domain test actually ran (scope="domain").
            _lead = {"domain": "Not in the indexed corpus for this domain",
                     "literature": "Not named by any indexed literature passage "
                                   "shown to the scorer"}.get(
                str(r.get("evidence_gate_scope") or ""),
                "Not named by any indexed passage or GeneRIF line shown to the "
                "scorer")
            if len(ver_specs) > 1:
                # More than one domain verified: say WHICH records verified in
                # WHICH domain.  Pooling the PMIDs behind a list of domains
                # reads as if every record covered every domain.
                _per = "; ".join(
                    "%s [PMID:%s]" % (sp.get("label") or sp["key"],
                                      "] [PMID:".join(sorted(
                                          set(per[sp["key"]]["verified"]), key=int)))
                    for sp in ver_specs)
                r["reason"] = (
                    "%s, but %d live PubMed record(s) name %s and were read back "
                    "to confirm it -- %s; the LLM score is kept."
                    % (_lead, len(set(ver_any)), g, _per))
            else:
                r["reason"] = (
                    "%s, but %d live PubMed record(s) name %s in %s and were read "
                    "back to confirm it [PMID:%s]; the LLM score is kept."
                    % (_lead, len(set(ver_any)), g, domains_label(ver_specs),
                       "] [PMID:".join(sorted(set(ver_any), key=int))))
            if r.get("reason_llm"):
                # followup 2026-09-28 (display nit, one NEC candidate): the gate counted NOTHING for this
                # gene, so the model's reason was written from evidence the domain gate did not
                # count -- say so (the evidence layers off: the old wording)
                r["reason"] = r["reason"] + (" Model reason (given on evidence the domain gate did not count): "
                                             if evidence_defence_on() else " Model reason: ") + str(r["reason_llm"])
            out["rescued"] += 1
            out["rescued_genes"].append(g)
        elif n_err and n_err == len(specs):
            # Every domain's live check failed for this gene: inconclusive, not
            # a confirmed absence.  The gate's own verdict and score stand, but
            # the row must not read like a gene nobody has published on.
            n_gene_err += 1
            r["evidence_source"] = "pubmed-error"
        if progress_every and (i + 1) % int(progress_every) == 0:
            print("[evidence_gate] pubmed rescue %d/%d checked, %d rescued (%.0fs)"
                  % (i + 1, len(cand), out["rescued"], time.time() - t0), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    out["n_pairs"] = n_pairs
    out["n_errors"] = n_err_all
    out["n_gene_errors"] = n_gene_err
    if n_pairs and n_err_all == n_pairs:
        out["error"] = ("every live PubMed check FAILED (%d/%d gene x domain "
                        "pairs); this is a failed check, NOT a confirmed "
                        "absence -- no score was changed" % (n_err_all, n_pairs))
    elif n_err_all:
        out["error"] = out["error"] or (
            "%d of %d live PubMed check(s) failed; those gene x domain pairs "
            "were not confirmed either way" % (n_err_all, n_pairs))
    try:
        out["stats"] = dict(N._PUBMED_STATS)
    except Exception:
        pass
    return out
