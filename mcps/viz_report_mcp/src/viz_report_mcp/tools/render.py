"""Final HTML report assembly via Jinja2.

Reads pre-built JSON artifacts (volcano, heatmap, trn) + markdown narrative
+ locus PNGs, fills the report.html template, writes single-file output.
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import jinja2
import pandas as pd


TEMPLATE_DIR = "/BIO1/MRAG/Agent/mcps/viz_report_mcp/templates"


def _md_to_html(md: str) -> str:
    """Minimal markdown -> HTML (paragraphs + bold + italic + code + headings).
    Avoid heavy dependencies — keep it simple and robust.
    """
    import re
    if not md:
        return ""
    lines = md.split("\n")
    out = []
    in_para = []
    def flush_para():
        if in_para:
            text = " ".join(in_para).strip()
            if text:
                # inline formatting
                text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
                text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
                text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
                text = re.sub(r"\[PMID:([0-9]+)\]", r'<a href="https://pubmed.ncbi.nlm.nih.gov/\1/" target="_blank">[PMID:\1]</a>', text)
                out.append(f"<p>{text}</p>")
            in_para.clear()
    for line in lines:
        s = line.rstrip()
        if not s:
            flush_para()
            continue
        if s.startswith("# "):
            flush_para(); out.append(f"<h2>{s[2:]}</h2>"); continue
        if s.startswith("## "):
            flush_para(); out.append(f"<h3>{s[3:]}</h3>"); continue
        if s.startswith("### "):
            flush_para(); out.append(f"<h4>{s[4:]}</h4>"); continue
        in_para.append(s)
    flush_para()
    return "\n".join(out)


def _png_to_data_url(png_path: str) -> str:
    p = Path(png_path)
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def _build_homer_known(known_dir: str, top_n: int = 10,
                       label: str = "", color: str = "#333333") -> dict:
    """Parse a HOMER findMotifsGenome knownResults output into the top-N
    enriched known motifs, with each motif's logo SVG inlined. Returns
    {label, color, motifs:[{rank, tf, full_name, consensus, pvalue,
    pct_target, pct_bg, logo_svg}]}. Missing dir/files degrade gracefully."""
    d = Path(known_dir)
    txt = d / "knownResults.txt"
    out = {"label": label, "color": color, "motifs": []}
    if not txt.exists():
        return out
    lines = txt.read_text(errors="replace").splitlines()
    for i, ln in enumerate(lines[1:top_n + 1], start=1):
        f = ln.split("\t")
        if len(f) < 9:
            continue
        name = f[0]
        # hg38 human: gene symbols are uppercase (Cistrome/HOMER metadata may
        # carry mouse-style TitleCase, e.g. Klf15 -> KLF15, Sp1 -> SP1).
        tf = (name.split("(")[0].split("/")[0].strip() or name).upper()
        logo_svg = ""
        svg_p = d / "knownResults" / f"known{i}.logo.svg"
        if svg_p.exists():
            raw = svg_p.read_text(errors="replace")
            k = raw.find("<svg")
            if k != -1:
                logo_svg = raw[k:]
                import re as _re
                _m = _re.search(r"<svg[^>]*>", logo_svg)
                if _m:
                    _tag = _m.group(0)
                    _w = _re.search(r'width="(\d+)', _tag)
                    _h = _re.search(r'height="(\d+)', _tag)
                    _vb = (f' viewBox="0 0 {_w.group(1)} {_h.group(1)}"'
                           if (_w and _h and "viewBox" not in _tag) else "")
                    _new = ('<svg' + _vb +
                            ' preserveAspectRatio="xMinYMid meet"'
                            ' style="height:22px;width:auto;display:block">')
                    logo_svg = logo_svg.replace(_tag, _new, 1)
        # dataset id = 2nd '/' segment of the HOMER motif name, e.g.
        # "MCF7-FOXA1-ChIP-Seq(GSE26831)" -> "GSE26831";
        # "U2OS-Foxo3-ChIP-Seq(E-MTAB-2701)" -> "E-MTAB-2701";
        # "JASPAR2024" / "Jurkat-NRSF-ChIP-Seq" (no parens) -> whole segment.
        _segs = name.split("/")
        _src = _segs[1].strip() if len(_segs) > 1 else ""
        _mm = re.search(r"\(([^()]*)\)\s*$", _src)
        dataset = (_mm.group(1) if _mm else _src).strip()
        out["motifs"].append({
            "rank": i,
            "tf": tf,
            "tf_label": tf,
            "dataset": dataset,
            "full_name": name,
            "consensus": f[1],
            "pvalue": f[2],
            "pct_target": f[6],
            "pct_bg": f[8],
            "logo_svg": logo_svg,
        })
    # Disambiguate repeated TF symbols within the group (e.g. FOXA1 x2 from
    # two ChIP-seq datasets): show each with its own dataset id.
    _cnt = {}
    for m in out["motifs"]:
        _cnt[m["tf"]] = _cnt.get(m["tf"], 0) + 1
    for m in out["motifs"]:
        if _cnt[m["tf"]] > 1 and m["dataset"]:
            m["tf_label"] = f'{m["tf"]} ({m["dataset"]})'
    return out


# ----------------------------------------------------------------- Section 13 ------------------
# The axis-validation node has already MEASURED everything Section 13 reports: the pair-level
# Fisher tests, their Holm correction over this run's headline family, and the descriptive triple
# list (mrag_axis.stat.report.by_model / .holm).  Nothing is recomputed below -- these helpers only
# reshape axis_validation/axis_<domain>.json into the rows, tabs and script payload the template
# prints, so every number in the section traces back to one field of that one file.

_AX_MODEL_NAMES = {"corgi": "Corgi", "alphagenome": "AlphaGenome", "both": "Both models"}

# "both models" first because it is the strongest requirement -- the same read-out passes under
# both models in the same row (mrag_axis.stat decision 4) -- so it is the view the figure opens on.
# A run with one arm has no consensus (a mouse run: Corgi is hg38-only, so only the AlphaGenome arm
# can exist), and the default then falls back to whichever arm the node did produce.
_AX_MODEL_ORDER = ("both", "alphagenome", "corgi")

# The ends in the order the table prints them.  The AR gate is a declared PRAD-only EXTRA
# requirement, reported as its own row and never folded into the enhancer headline
# (mrag_axis.stat decision 8), so it appears only when the node emitted it.
_AX_ENDS = (("pair_enhancer", "Enhancer (SE + ordinary)"),
            ("pair_enhancer_ar", "Enhancer + AR gate"),
            ("pair_target", "Target promoter"))
_AX_END_LABEL = dict(_AX_ENDS)


def _ax_num(v):
    """A measured number, or None.  NaN is 'not measured', not a value to print."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _ax_p(p):
    """A p-value the way the section prints it: below 0.01 the exponent form, which is what
    SECTION13_CONTRACT.md's own table uses; at and above it the plain number, which reads better
    (the target end's 0.089)."""
    f = _ax_num(p)
    if f is None:
        return "&mdash;"
    if f <= 0:
        return "&lt; 1 &times; 10<sup>-300</sup>"   # underflowed to 0.0 in the Fisher test
    if f >= 1e-2:
        return "%.3g" % f
    m, ex = ("%.1e" % f).split("e")
    return "%s &times; 10<sup>%d</sup>" % (m, int(ex))


def _ax_untestable(row):
    """Why this row's Fisher test could not be run, or None when it was (review 3, finding 8).

    mrag_axis.stat.stats.fisher() returns NaN -- written as null -- for a table with an empty
    margin (no network pair, no lineage-foreign pair, nobody passed, everybody passed), and None for
    a genuinely INFINITE odds ratio (some network pairs passed, no lineage-foreign one did).  Both
    reach this file as a null odds_ratio, so the row's own counts decide: an empty margin is 'not
    testable', never 'OR infinite, p 1', and it is never counted as a test above p = 0.05.  The
    node may also mark the row itself ("testable": false with a reason), which wins.  A row with
    no p-value at all is not a test either."""
    if not isinstance(row, dict):
        return None
    if row.get("testable") is False:
        return str(row.get("reason") or row.get("untestable_reason")
                   or "the node marked this test as not testable")
    if "odds_ratio" in row and _ax_num(row.get("odds_ratio")) is None:
        a, n = _ax_num(row.get("passed")), _ax_num(row.get("n"))
        b, m = _ax_num(row.get("fp")), _ax_num(row.get("n_frn"))
        if n is not None and n <= 0:
            return "no network pair at this end"
        if m is not None and m <= 0:
            return "no lineage-foreign pair at this end"
        if None not in (a, n, b, m):
            if a + b <= 0:
                return "no pair passed in either group"
            if (n - a) + (m - b) <= 0:
                return "every pair passed in both groups"
        elif _ax_num(row.get("p")) is None:
            return "no counts and no p-value recorded"
    if _ax_num(row.get("p")) is None:
        return "no p-value recorded"
    return None


def _ax_nt(reason):
    """The odds-ratio cell of a row whose test was not run: 'not testable' and why."""
    import html as _html
    return ('not testable<br><span style="font-size:11px;color:var(--muted)">%s</span>'
            % _html.escape(str(reason), quote=False))


def _ax_or(row):
    """mrag_axis.stat.stats.fisher() returns None for an INFINITE odds ratio (no lineage-foreign
    pair passed), which is a result; a row with no odds_ratio field at all is a missing one; a row
    whose table has an empty margin was never tested (_ax_untestable)."""
    _nt = _ax_untestable(row)
    if _nt:
        return _ax_nt(_nt)
    if "odds_ratio" not in row:
        return "&mdash;"
    f = _ax_num(row.get("odds_ratio"))
    return "&infin;" if f is None else "%.2f" % f


def _ax_rate(a, n):
    a, n = _ax_num(a), _ax_num(n)
    return "%.1f%%" % (100.0 * a / n) if (a is not None and n) else "n/a"


def _ax_names(v):
    """A read-out / biosample / factor field as a list of strings, in whichever shape the node
    wrote it (a list, a {name: ontology term} map, or a single string)."""
    if not v:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, dict):
        return [str(x) for x in (v.get("terms") or v.get("biosamples") or list(v.keys()))]
    return [str(x) for x in v]


def _ax_graph_ids(se_trn):
    """The node and edge ids Section 10b ACTUALLY drew, which is what the axis figure has to name.

    A target gene that is also a panel TF is drawn with the node id '<gene>__target'
    (cytoscape_trn.py builds its `tid` map that way), so an axis's TF->target edge id cannot be
    built from the gene name alone -- it is resolved against the graph instead.  Returns empty
    sets when this run drew no graph, and then the payload keeps the plain names.
    """
    els = (se_trn or {}).get("elements") or []
    if isinstance(els, dict):           # cyjs may split elements into nodes/edges
        els = list(els.get("nodes") or []) + list(els.get("edges") or [])
    nodes, edges = {}, set()
    for el in els:
        d = (el or {}).get("data") or {}
        i = d.get("id")
        if not i:
            continue
        if d.get("source"):
            edges.add(str(i))
        else:
            nodes[str(i)] = str(d.get("type") or "")
    return nodes, edges


def _ax_pick(nodes, cands, types):
    """The graph node an axis end means.  The TYPE decides, because the ids collide: a gene that is
    also a panel TF owns the bare name as a TF node and '<gene>__target' as a target node, so
    resolving the target end by name alone would thicken the factor's own edge instead.  A graph
    that labels its types differently falls back to the first candidate that exists at all."""
    for c in cands:
        if c and nodes.get(c) in types:
            return c
    for c in cands:
        if c and c in nodes:
            return c
    return None


def _ax_se_tf_payload(se_tf, row_key, nodes, edges):
    """One model set's script payload: the SE -> TF edges to draw in the supported colour, the
    super-enhancer and factor nodes they join, and each super-enhancer's factors for the click
    read-out.

    `se_tf` is the node's se_tf_edges record for this set and `row_key` the pair row the figure
    follows, the set's headline.  Only SE -> TF edges are named: the target end does not separate
    network factors from lineage-foreign ones, so no target edge is ever in this payload -- the
    script draws the network's TF -> target edges as motif-based candidates from the graph itself.
    bySE rows carry the factor's graph id AND its name in the node record, so a factor the graph
    spells differently still reads as the statistic has it.  `unmapped` counts passed edges with
    no edge in the graph; the caption states that number rather than letting them disappear.
    """
    import collections as _collections
    eset, nset, by_se, unmapped = set(), set(), _collections.defaultdict(list), 0
    for pair in ((se_tf.get("passed") or {}).get(row_key) or []):
        s, t = str(pair[0]), str(pair[1])
        se = _ax_pick(nodes, [s, s.upper()], ("SE",)) if nodes else s
        tf = _ax_pick(nodes, [t, t.upper()], ("TF",)) if nodes else t
        e1 = "%s__to__%s" % (se, tf) if (se and tf) else None
        if edges and e1 not in edges:
            e1 = None
        if not e1:
            unmapped += 1
            continue
        eset.add(e1)
        nset.update((se, tf))
        by_se[se].append([tf, t])
    return {"edges": sorted(eset), "nodes": sorted(nset), "unmapped": unmapped,
            "bySE": {k: sorted(v) for k, v in by_se.items()}}


_SE_GENE_FAM = ("free", "near")
_SE_GENE_MODELS = (("alphagenome", ("ag",)), ("corgi", ("corgi",)), ("both", ("ag", "corgi")))


def _se_gene_ctx(raw, nodes, edges):
    """ctx['axis']['se_gene'] -- the SE -> closest-gene end, and the graph ids of the proximity
    edges it supports.

    The node records this end under "se_gene"; it is DESCRIPTIVE, so it
    gets its own card and never a row of the pair-level table or the Holm family.  A pass is
    reported per control family, never pooled: the free family alone carries a distance artefact at
    the H3K4me3 read-out, which is exactly what the near family is there to show.
    """
    sg = raw.get("se_gene")
    if not isinstance(sg, dict):
        return None, {}
    rep = [str(x) for x in (sg.get("reported") or [])]
    names = {"rna": "RNA (exons)", "prom": "H3K4me3 (promoter)", "dnase": "DNase (TSS)"}
    rows, prox = [], {k: {"edges": set(), "nodes": set(), "bySE": {}} for k, _ms in _SE_GENE_MODELS}
    for r in sg.get("rows") or []:
        if not isinstance(r, dict):
            continue
        se, gene = str(r.get("se") or ""), str(r.get("gene") or "")
        cells, passed_by = {}, {}
        for rd in rep:
            cs = []
            for mk, ms in _SE_GENE_MODELS:
                if mk == "both":
                    continue
                d = ((r.get("models") or {}).get(ms[0]) or {}).get(rd)
                if not isinstance(d, dict):
                    continue
                fam = {f: bool((d.get(f) or {}).get("passed")) for f in _SE_GENE_FAM}
                drop = (d.get("near") or {}).get("drop")
                if drop is None:
                    drop = (d.get("free") or {}).get("drop")
                cs.append({"model": _AX_MODEL_NAMES.get(mk, mk), "key": mk,
                           "free": fam["free"], "near": fam["near"],
                           "both_families": fam["free"] and fam["near"],
                           "drop": ("%+.2f" % drop) if isinstance(drop, (int, float)) else ""})
            cells[rd] = cs
            passed_by[rd] = {c["key"] for c in cs if c["both_families"]}
        _d = r.get("control_near_edge_bp")
        rows.append({
            "se": se, "gene": gene, "label": str(r.get("label") or ""),
            "scrambled_kb": ("%.1f" % (r["scrambled_bp"] / 1000.0))
                            if isinstance(r.get("scrambled_bp"), (int, float)) else "",
            "n_blocks": r.get("n_blocks") or "",
            "ko_kb": ("%.1f" % (r["ko_near_edge_bp"] / 1000.0))
                     if isinstance(r.get("ko_near_edge_bp"), (int, float)) else "",
            "ctl_kb": ("%.1f&ndash;%.1f" % (_d[0] / 1000.0, _d[1] / 1000.0))
                      if isinstance(_d, (list, tuple)) and len(_d) == 2 else "",
            "cells": cells})
        for mk, ms in _SE_GENE_MODELS:
            if not any(mk in passed_by.get(rd, set()) if mk != "both"
                       else {"alphagenome", "corgi"} <= passed_by.get(rd, set()) for rd in rep):
                continue
            s_id = _ax_pick(nodes, [se, se.upper()], ("SE",)) if nodes else se
            g_id = (_ax_pick(nodes, [gene, gene.upper(), gene + "__target"],
                             ("SE_gene", "Target")) if nodes else gene)
            if not (s_id and g_id):
                continue
            # cytoscape_trn.py names a proximity edge "<SE>__near__<gene>" (the factor edges are
            # "__to__"), so both spellings are tried and an id the graph does not carry is skipped.
            e_id = next((x for x in ("%s__near__%s" % (s_id, g_id), "%s__to__%s" % (s_id, g_id))
                         if not edges or x in edges), None)
            if not e_id:
                continue
            prox[mk]["edges"].add(e_id); prox[mk]["nodes"].update((s_id, g_id))
            # Keyed by the SE's graph id, so clicking a super-enhancer can name its closest gene
            # and the read-outs it passed, the way bySE does for the axes.
            prox[mk]["bySE"][s_id] = [e_id, g_id, gene,
                                      ", ".join(names.get(rd, rd) for rd in rep
                                                if (mk in passed_by.get(rd, set())
                                                    if mk != "both"
                                                    else {"alphagenome", "corgi"} <= passed_by.get(rd, set())))]
    summary = []
    for mk, ms in _SE_GENE_MODELS:
        if mk == "both":
            continue
        for rd in rep:
            v = ((sg.get("summary") or {}).get(ms[0]) or {}).get(rd)
            if isinstance(v, dict):
                summary.append({"model": _AX_MODEL_NAMES.get(mk, mk), "read_out": names.get(rd, rd),
                                "free": "%s/%s" % (v.get("free_pass"), v.get("n")),
                                "near": "%s/%s" % (v.get("near_pass"), v.get("n_near"))})
    strict = [{"se": str(s.get("se") or ""), "gene": str(s.get("gene") or ""),
               "read_outs": ", ".join(names.get(x, x) for x in (s.get("reported") or []))}
              for s in (sg.get("strict") or []) if (s.get("reported") or [])]
    ctx = {"state": str(sg.get("status") or ""), "reason": str(sg.get("reason") or ""),
           "read_outs": [names.get(x, x) for x in rep], "rows": rows, "summary": summary,
           "strict": strict, "n_items": sg.get("n_items") or len(rows),
           "n_free": 0, "n_near": 0}
    for r in sg.get("rows") or []:
        for m in (r.get("models") or {}).values():
            for d in m.values():
                ctx["n_free"] = max(ctx["n_free"], ((d or {}).get("free") or {}).get("n") or 0)
                ctx["n_near"] = max(ctx["n_near"], ((d or {}).get("near") or {}).get("n") or 0)
    return ctx, {k: {"edges": sorted(v["edges"]), "nodes": sorted(v["nodes"]), "bySE": v["bySE"]}
                 for k, v in prox.items()}


def _axis_ctx(axis_json, se_trn=None, llm_domain=""):
    """ctx['axis'] for Section 13, shaped from the axis-validation node's axis_<domain>.json.

    `llm_domain` is accepted and unused: Section 13 no longer prints the literature domain
    (the 2026-10-23 report trim removed its Method paragraph).

    None                     -> there is no such file: the step did not run at all.
    {'models': [], 'reason'} -> the file is there but no arm produced a result.
    Either way the section states WHICH, because an empty result and a suppressed one must never
    look alike (the model-arm failure policy).  Nothing here raises: a missing or
    unreadable input becomes a stated reason.
    """
    if not axis_json or not Path(axis_json).exists():
        return None
    try:
        raw = json.loads(Path(axis_json).read_text())
    except Exception as _e:
        return {"models": [], "arm_notes": [], "reason": "",
                "state": "left a result file that could not be read (%s)" % type(_e).__name__}
    if not isinstance(raw, dict):
        return {"models": [], "arm_notes": [], "reason": "",
                "state": "left a result file that is not a JSON object"}

    # The node may nest the per-model results under "models" or carry them at the top level; both
    # are read, so a schema change on the node's side cannot silently empty the section.
    mods = raw.get("models")
    if not isinstance(mods, dict):
        mods = {k: raw[k] for k in _AX_MODEL_ORDER if isinstance(raw.get(k), dict)}
    mods = {str(k): v for k, v in mods.items() if isinstance(v, dict)}
    order = ([k for k in _AX_MODEL_ORDER if k in mods]
             + sorted(k for k in mods if k not in _AX_MODEL_ORDER))

    # An arm that did not finish is NAMED, with its reason: a failed arm is a WARN gate and the run
    # continues, so a reader has to be able to tell a negative result from
    # a missing one.  An "ok" arm needs no notice.
    _arms = raw.get("arms") or raw.get("arm_status") or {}
    _items = (list(_arms.items()) if isinstance(_arms, dict)
              else [((a or {}).get("arm"), a) for a in _arms])
    arm_notes, _noted = [], set()
    for _name, _a in _items:
        _a = _a if isinstance(_a, dict) else {"status": str(_a)}
        _st = str(_a.get("status") or "")
        if not _name or _st == "ok":
            continue
        _noted.add(str(_name))
        arm_notes.append({"arm": _AX_MODEL_NAMES.get(str(_name), str(_name)),
                          "status": _st or "not run",
                          "reason": str(_a.get("error") or _a.get("reason")
                                        or "no reason recorded")})
    # A model set the node WITHHELD -- its arm wrote no read-out column, so the set does not
    # exist, and 'both models' in particular cannot exist without both arms -- is named the same
    # way as a failed arm: a tab that is missing must never be missing without a reason.  The
    # node reuses the arm's own reason for every set that arm takes down, so a set whose arm
    # already has a note above is not repeated, and 'both' then just says it needs both arms.
    _wh = raw.get("models_withheld")
    for _name, _why in (sorted(_wh.items()) if isinstance(_wh, dict) else []):
        _name = str(_name)
        if _name in _noted:
            continue
        _why = str(_why or "no reason recorded")
        _down = [a for a in ("corgi", "alphagenome") if a in _noted]
        if _name == "both" and _down:
            _why = ("needs both arms, and the %s arm did not finish (its reason is given above)"
                    % " and the ".join(_AX_MODEL_NAMES[a] for a in _down))
        arm_notes.append({"arm": _AX_MODEL_NAMES.get(_name, _name),
                          "status": "withheld", "reason": _why})
    if not mods:
        # The section says WHICH of the three it is (finding 30): the step was skipped, it failed,
        # or it ran and scored no model set -- never one sentence for all of them.
        _rst = str(raw.get("status") or "").strip().lower()
        _state = {"skipped": "was skipped",
                  "failed": "failed"}.get(_rst, "ran but scored no model set")
        return {"models": [], "arm_notes": arm_notes, "state": _state,
                "reason": str(raw.get("reason") or "no model arm produced a result")}

    # The LEADING model set is the node's declared headline (choose_headline: declared, else the
    # consensus when both single-model headlines survive Holm) -- never "the smallest p".  It
    # decides the default tab and the one emphasised row, nothing else.
    default = (str(raw.get("default_model")) if str(raw.get("default_model")) in mods
               else order[0])
    nodes, edges = _ax_graph_ids(se_trn)
    import html as _html_mod
    _holm_by_model = raw.get("holm") if isinstance(raw.get("holm"), dict) else {}
    pair_rows, tabs, payload, tgt_tests, tgt_untest = [], [], {}, [], []
    for k in order:
        m = mods[k]
        # The headline is the AR row where the run declares one (PRAD) and the enhancer row
        # otherwise -- mrag_axis.stat.report.headline_key -- and it is the only row in the Holm
        # family, because that family is the set of headlines the section actually prints.
        head = "pair_enhancer_ar" if isinstance(m.get("pair_enhancer_ar"), dict) else "pair_enhancer"
        for key, label in _AX_ENDS:
            r = m.get(key)
            if not isinstance(r, dict):
                continue
            _h = r.get("holm", _holm_by_model.get(k)) if key == head else None
            _nt = _ax_untestable(r)
            pair_rows.append({
                "model": _AX_MODEL_NAMES.get(k, k), "end": label,
                "passed": r.get("passed"), "n": r.get("n"),
                "rate": _ax_rate(r.get("passed"), r.get("n")),
                "fp": r.get("fp"), "n_frn": r.get("n_frn"),
                "fp_rate": _ax_rate(r.get("fp"), r.get("n_frn")),
                "odds_ratio": _ax_or(r), "p": "&mdash;" if _nt else _ax_p(r.get("p")),
                "holm": _ax_p(_h) if (key == head and not _nt) else "&mdash;",
                "headline": key == head and k == default, "untestable": _nt or "",
                # The main table prints the enhancer end; the target rows move to the section's
                # details with their note.
                "end_key": key, "target": key == "pair_target",
            })
            if key == "pair_target":
                if _nt:
                    tgt_untest.append((_AX_MODEL_NAMES.get(k, k), _nt))
                else:
                    tgt_tests.append((_ax_num(r.get("p")), _AX_MODEL_NAMES.get(k, k), _ax_or(r)))
        # The figure follows this set's headline row: an SE -> TF edge is pink
        # when the node recorded it as passing that row and the row survives Holm -- the same evidence
        # as the pair-level table.  The node's se_tf_edges.subset (the drawn edges' own rate against the
        # foreign one) is descriptive and is NOT a drawing gate.  The
        # SE-TF-target triples stay in the axis JSON (descriptive) but are no longer drawn or counted:
        # their target half does not separate network factors from lineage-foreign ones, and drawing it
        # solid read as a validated link.
        name = _AX_MODEL_NAMES.get(k, k)
        hr = m.get(head) or {}
        _hh = _ax_num(hr.get("holm", _holm_by_model.get(k)))
        st = (raw.get("se_tf_edges") or {}).get(k)
        _frate = _ax_rate(hr.get("fp"), hr.get("n_frn"))
        why_not = ""
        if _ax_untestable(hr):
            why_not = ("headline test not runnable (%s)"
                       % _html_mod.escape(_ax_untestable(hr), quote=False))
        elif _hh is None:
            why_not = "no Holm <em>p</em> recorded"
        elif _hh >= 0.05:
            why_not = "headline not significant after Holm (<em>p</em> = %s)" % _ax_p(_hh)
        elif not isinstance(st, dict):
            why_not = "this axis file has no SE&rarr;TF edge list (it predates 2026-10-22)"
        elif st.get("error"):
            why_not = ("the SE&rarr;TF edge list failed (%s)"
                       % _html_mod.escape(str(st["error"]), quote=False))
        elif st.get("agrees_with_statistic") is not True:
            why_not = "the SE&rarr;TF edge list does not match the table's row"
        n_t = len(st.get("tested") or []) if isinstance(st, dict) else 0
        n_g = _ax_num(st.get("n_graph")) if isinstance(st, dict) else None
        n_nm = int(_ax_num(st.get("n_no_motif")) or 0) if isinstance(st, dict) else 0
        _rest = ""
        if n_g is not None and n_g > n_t:
            _rest = " &middot; %d network SE&rarr;TF edges untested" % (n_g - n_t)
            if n_nm:
                _rest += (" (%d name a factor not scored: %s)"
                          % (n_nm, _html_mod.escape(", ".join(st.get("no_motif_tfs") or []), quote=False)))
        if why_not:
            pl = {"edges": [], "nodes": [], "unmapped": 0, "bySE": {},
                  "caption": "<b>All SE&rarr;TF edges faded</b> for %s: %s." % (name, why_not)}
            links = None
        else:
            pl = _ax_se_tf_payload(st, head, nodes, edges)
            links = len((st.get("passed") or {}).get(head) or [])
            pl["caption"] = (
                "<b>%d of %d</b> tested SE&rarr;TF edges passed &middot; %s: %s vs %s, OR %s, Holm "
                "<em>p</em> = %s%s &middot; %s of lineage-foreign pairs also pass, so one edge alone is "
                "not proof."
                % (links, n_t, _AX_END_LABEL[head], _ax_rate(hr.get("passed"), hr.get("n")), _frate,
                   _ax_or(hr), _ax_p(_hh), _rest, _frate))
            if pl["unmapped"]:
                pl["caption"] += " %d not in the Section-10b graph (counted only)." % pl["unmapped"]
        tabs.append({"key": k, "name": name, "links": links})
        payload[k] = pl

    # The target (promoter) end is not significant in any domain or model,
    # and the section SAYS so instead of leaving it to be read off the table.  The sentences are
    # built from this run's own target-end rows, so a run whose target end DID separate is not
    # described as if it had not.
    # A target-end row whose test could not be run (an empty margin, _ax_untestable) is neither
    # above nor below 0.05: it is left out of both counts and named as not testable (finding 8).
    _ns = [x for x in tgt_tests if x[0] >= 0.05]
    _best = min(tgt_tests, default=None)
    _bt = ((", strongest: %s, OR %s, <em>p</em> = %s" % (_best[1], _best[2], _ax_p(_best[0])))
           if _best else "")
    _ut = ""
    if tgt_untest:
        _ut = ("%d target-end row%s could not be tested (%s)"
               % (len(tgt_untest), "" if len(tgt_untest) == 1 else "s",
                  "; ".join("%s: %s" % (m, _html_mod.escape(w, quote=False)) for m, w in tgt_untest)))
    # The target rows and this note sit in the section's details.
    if not tgt_tests and tgt_untest:
        target_end_note = "Not testable here: %s." % _ut
    elif not tgt_tests:
        target_end_note = ""
    else:
        target_end_note = ("Not in the Holm family. %d of %d tests above <em>p</em> = 0.05%s%s; %s."
                           % (len(_ns), len(tgt_tests), _bt, ("; " + _ut) if _ut else "",
                              "a single target pass cannot be told from chance, so no TF &rarr; "
                              "target edge is shown as passing" if len(_ns) == len(tgt_tests) else
                              "no TF &rarr; target edge is shown as passing"))

    se_gene, _prox = _se_gene_ctx(raw, nodes, edges)
    for _k, _v in (_prox or {}).items():
        if _k in payload:
            payload[_k]["prox"] = _v
    return {
        "models": tabs, "default_model": default,
        "default_model_name": _AX_MODEL_NAMES.get(default, default),
        "headline_reason": str(raw.get("headline_reason") or ""),
        "pair_rows": pair_rows, "arm_notes": arm_notes, "payload": payload,
        "enhancer_rows": [r for r in pair_rows if not r["target"]],
        "has_ar": any(r["end_key"] == "pair_enhancer_ar" for r in pair_rows),
        # The AR gate is an AlphaGenome read-out: the OTHER model sets that print an AR row need it too,
        # and only the sets this run actually scored are named (a single-arm run has no consensus).
        "ar_also": " and ".join(r["model"] for r in pair_rows
                                if r["end_key"] == "pair_enhancer_ar" and r["model"] != "AlphaGenome"),
        "target_rows": [r for r in pair_rows if r["target"]],
        "n_foreign_tfs": len(_ax_names(raw.get("foreign_tfs"))),
        "n_dropped_pairs": len(raw.get("dropped_pairs") or []),
        "target_end_note": target_end_note, "reason": "", "state": "",
        "se_gene": se_gene,
    }


_S13_RE = re.compile(r'<section id="ag-validation">.*?</section>', re.S)


# A script a kept Section 13 carries.  It is placed INSIDE the kept <section>, just before its
# </section>, so the next re-render's _S13_RE spans it and copies it verbatim with the section
# (review 3, finding 9): the marker is a JS comment, so the tag stays a plain <script>.
_S13_KEPT_JS = "// s13-kept-script"


def _s13_driver_after(html, start):
    """The new-format Section 13's own driver script (`var AX = ...`, its tabs, caption and figure),
    which the template emits right AFTER </section>: (begin, end) of that <script>...</script>, or
    None when there is none before the next section."""
    k = html.find("var AX", start)
    nxt = html.find("<section", start)
    if k < 0 or (0 <= nxt < k):
        return None
    b = html.rfind("<script", start, k)
    e = html.find("</script>", k)
    if b < 0 or e < 0:
        return None
    return b, e + len("</script>")


def keep_prior_section13(output_html, prior_html, axis_json=None):
    """For re-render helpers outside the pipeline: a run with no axis result -- every run finished
    before the axis_validation step existed -- would come out of a re-render with Section 13
    replaced by the 'no result' notice, and its published Section 13 silently gone.  This puts the
    PRIOR report's Section-13 block back into the freshly rendered `output_html`, with a notice
    that says it was kept, together with the scripts that drive it: the two calls of the old
    single-base section that live in the report-wide script area (its scatter and its table), or
    the new axis section's own driver script that follows its </section>.  Those scripts are put
    INSIDE the kept section, so a second re-render keeps them too (review 3, findings 9 and 10).

    Returns what it did, as one line for the helper's log.  It changes nothing when the run HAS an
    axis result, when there was no prior report, when the prior Section 13 was itself only the
    'no result' notice, when the fresh render's Section 13 is this run's organism skip or a stated
    step error (both are what the run itself says, review 3 finding 12), or when a new-format
    prior section's driver script cannot be found (a kept figure box with nothing behind it would
    be the empty box the template avoids)."""
    if axis_json and Path(str(axis_json)).exists():
        return "rendered from %s" % axis_json
    if not prior_html:
        return "no axis result and no earlier report: Section 13 states that the step left no result"
    p = Path(output_html)
    new = p.read_text(encoding="utf-8")
    mo, mn = _S13_RE.search(prior_html), _S13_RE.search(new)
    if not mo or not mn:
        return "no Section-13 block found in the %s report: left as rendered" % (
            "earlier" if not mo else "new")
    fresh = mn.group(0)
    if 'data-s13="skipped"' in fresh:
        return "Section 13 is this run's organism skip: left as rendered"
    if 'data-s13-state="step-error"' in fresh:
        return "Section 13 states the axis_validation step's recorded error: left as rendered"
    old = mo.group(0)
    if old.count("<section") != 1:
        return "the earlier Section-13 block is not self-contained: left as rendered"
    if 'data-s13="no-result"' in old:
        return "the earlier report had no Section-13 result either: left as rendered"
    if 'data-s13="skipped"' in old:
        return "the earlier report's Section 13 was an organism skip: left as rendered"
    scripts, n_calls = [], 0
    if 'data-s13="kept"' in old:
        # Kept before.  Its scripts are inside the block (copied verbatim with it); a block kept by
        # the first version of this function has its one carried script right after </section>.
        if _S13_KEPT_JS not in old:
            m = re.match(r"\s*<script>\s*document\.addEventListener\('DOMContentLoaded'.*?</script>",
                         prior_html[mo.end():], re.S)
            if m and "kept Section 13:" in m.group(0):
                scripts.append(m.group(0).strip().replace("<script>", "<script>\n" + _S13_KEPT_JS, 1))
                n_calls = m.group(0).count("try {")
        n_calls += old.count("console.error('kept Section 13:'")
    else:
        if 'id="cy-ax"' in old or 'class="ax-tab"' in old or 'id="ax-caption"' in old:
            # the NEW axis section: its tabs, caption and figure are driven by the script after it
            span = _s13_driver_after(prior_html, mo.end())
            if not span:
                return ("the earlier report's Section 13 is the axis section but its driver script "
                        "was not found: left as rendered")
            scripts.append(prior_html[span[0]:span[1]].replace("<script>", "<script>\n" + _S13_KEPT_JS, 1))
        # the old single-base section's scatter and table were initialised in the report-wide
        # script area, outside the section; carry those two calls with it
        calls = [ln.strip() for ln in prior_html[mo.end():].split("\n")
                 if ln.lstrip().startswith("safePlot('ag-scatter-plotly'")]
        j = prior_html.find("$('#ag-table').DataTable({", mo.end())
        if j >= 0:
            k = prior_html.find("\n  });", j)
            if k > 0:
                calls.append(prior_html[j:k + len("\n  });")].strip())
        if calls:
            scripts.append("<script>\n" + _S13_KEPT_JS + "\ndocument.addEventListener('DOMContentLoaded', function(){\n"
                           + "\n".join("  try { %s } catch (e) { console.error('kept Section 13:', e); }" % c
                                       for c in calls)
                           + "\n});\n</script>")
        n_calls = len(calls)
        i = old.find("</h2>")
        i = (i + len("</h2>")) if i >= 0 else old.find(">") + 1
        old = (old[:i] + '\n  <div class="callout" style="border-left-color:#b26a00" data-s13="kept">'
               "<strong>Kept from this run&rsquo;s earlier report.</strong> This report was re-rendered "
               "from a run that has no axis-validation result file, so Section 13 could not be drawn "
               "again; the section below is the one the earlier report carried, unchanged.</div>"
               + old[i:])
    if scripts:
        assert old.endswith("</section>")
        old = old[:-len("</section>")].rstrip("\n") + "\n" + "\n".join(scripts) + "\n</section>"
    p.write_text(new[:mn.start()] + old + new[mn.end():], encoding="utf-8")
    return ("no axis result: the earlier report's Section 13 was kept (%d chars, %d script call%s%s)"
            % (len(old), n_calls, "" if n_calls == 1 else "s",
               ", its driver script" if any("var AX" in x for x in scripts) or
               ("var AX" in old and _S13_KEPT_JS in old) else ""))


def register_render_tools(mcp):

    @mcp.tool()
    def render_report(
        output_html: str,
        report_title: str,
        study_label: str = "",
        pipeline_version: str = "0.1.0",
        # JSON artifacts (paths)
        volcano_atac_json: Optional[str] = None,
        volcano_rna_json: Optional[str] = None,
        tf_rank_json: Optional[str] = None,
        shared_dotplot_json: Optional[str] = None,
        per_tf_heatmap_json: Optional[str] = None,
        trn_json: Optional[str] = None,
        signal_heatmap_png: Optional[str] = None,
        signal_profile_png: Optional[str] = None,
        per_tf_profile_json: Optional[str] = None,
        # 2026-06-04 R-script plots (PNG fallback)
        rank_png:     Optional[str] = None,
        volcano_png:  Optional[str] = None,
        heatmap_png:  Optional[str] = None,
        # 2026-06-08 interactive Plotly versions of rank + volcano + heatmap
        rank_json:    Optional[str] = None,
        volcano_json: Optional[str] = None,
        heatmap_json: Optional[str] = None,
        heatmap_median_json: Optional[str] = None,
        # 2026-06-07 Cytoscape style (test.cys-mirrored) — replaces legacy TRN
        trn_cyjs_json:    Optional[str] = None,
        curated_targets_json: Optional[str] = None,
        # Section 10b: SE -> TF -> Target TRN (se_cyjs.json from SETrnBuilderNode)
        se_trn_json: Optional[str] = None,
        cancer_type: str = "",
        # Section 13 (stage 2c, 2026-10-18): the axis-validation node's axis_<domain>.json --
        # the pair-level tests, their Holm correction and the descriptive triple list.
        axis_json: Optional[str] = None,
        # a notice printed at the top of Section 13, e.g. that a manual target swap did not re-measure
        # it; plain text, escaped here; "" = none
        axis_note: str = "",
        # the axis_validation step's own error when it raised (or failed) before writing its
        # result file, as the run manifest records it (RenderNode reads it there); Section 13 then
        # states that error instead of the generic 'left no result file' (review 3, finding 16)
        axis_step_error: str = "",
        citation_audit: Optional[dict] = None,
        tf_table_csv: Optional[str] = None,
        # Locus tracks (list of {tf, gene, region, png_path})
        locus_tracks: Optional[List[Dict]] = None,
        # HOMER known-motif enrichment groups (list of {dir,label,color})
        homer_known_groups: Optional[List[Dict]] = None,
        # GIGGLE Cistrome TF-overlap point plots (Section 9)
        giggle_up_png: Optional[str] = None,
        giggle_down_png: Optional[str] = None,
        peakheatmap_png: Optional[str] = None,
        giggle_up_label: str = "Up in treatment",
        giggle_down_label: str = "Down in treatment",
        # Narrative (markdown text or path)
        narrative_md_path: Optional[str] = None,
        executive_summary_md: Optional[str] = None,
        # Pipeline metadata
        samples_str: str = "",
        conditions_str: str = "",
        motif_db: str = "HOMER + JASPAR v12 (2322 motifs)",
        tf_selection: str = "Pcom-deterministic Top 10 (INTEGRATED.csv)",
        diffbind_threshold: str = "FDR < 0.05",
        deseq_threshold: str = "padj < 0.05",
        # Section 2 "LLM domain" row (2026-09-17): finished label built by the
        # orchestrator from known corpus display names only (no autoescape here).
        llm_domain: str = "",
        # --- organism / genome provenance (2026-09-23, Task B) -------------
        # Section 2 has to say WHICH genome the numbers above it were produced
        # against and HOW that was decided, because every figure in the report
        # is silently conditional on it. `organism_skipped` is the list of
        # steps this organism cannot run: the sections they feed then say "not
        # available for <species>" instead of rendering themselves empty, which
        # is indistinguishable from "we looked and found nothing".
        organism: str = "",            # "Mus musculus"
        organism_common: str = "",     # "mouse"
        genome_build: str = "",        # "mm10"
        organism_detection: str = "",  # "SRA metadata: Mus musculus (8 runs)"
        organism_skipped: Optional[List[Dict]] = None,   # [{name, reason}]
        ortholog_banner: str = "",     # §14 human-ortholog disclosure ("" = human)
        # The human-ortholog route reached only Section 14, but Section 10's
        # target set is CHOSEN by the ortholog-routed literature score and
        # Section 11 enriches that same set -- so a reader of either section
        # had nothing telling them the selection rests on human literature
        # (2026-09-23). "" on a human run, so its text is unchanged.
        ortholog_overview: str = "",   # "human-ortholog route (MGI ...): 3/4 ..."
        ortholog_note: str = "",       # one short sentence for §10 / §11
        # 2026-09-28 pre-deploy review: the notice of a FAILED target step (runner-built text;
        # "" = the step did not fail, and the report is byte for byte what it was)
        curated_error: str = "",
    ) -> dict:
        """Assemble interactive HTML report from pre-built artifacts.

        All JSON files are output of other viz_report_mcp tools (plot_volcano,
        plot_tf_rank, build_trn_elements). PNGs are encoded inline
        (base64). Single-file HTML output for shareability.
        """
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
            autoescape=False,
        )
        tpl = env.get_template("report.html")

        homer_known = []
        for _g in (homer_known_groups or []):
            try:
                _hk = _build_homer_known(_g.get("dir", ""), top_n=10,
                                         label=_g.get("label", ""),
                                         color=_g.get("color", "#333333"))
                if _hk["motifs"]:
                    homer_known.append(_hk)
            except Exception:
                pass

        ctx = {
            "report_title": "MRAG Agent Report",  # always fixed, sample-agnostic
            "study_label": study_label,
            "pipeline_version": pipeline_version,
            "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "pipeline": {
                "samples": samples_str or conditions_str or "(not specified)",
                "conditions": conditions_str,
                "motif_db": motif_db,
                "tf_selection": tf_selection,
                "diffbind_threshold": diffbind_threshold,
                "deseq_threshold": deseq_threshold,
                "llm_domain": llm_domain or "(not specified)",
                # "" on a pre-organism run -> the template hides the row rather
                # than asserting a species nobody recorded.
                "organism": organism or "",
                "organism_common": organism_common or "",
                "genome_build": genome_build or "",
                "organism_detection": organism_detection or "",
            },
        }
        # Steps this organism cannot run. Rendered BOTH as a Section-2 line and
        # inside the section itself, so a reader who jumps straight to Section 9
        # is not left with a blank panel to interpret.
        _skips = [{"name": str(x.get("name", "")), "reason": str(x.get("reason", ""))}
                  for x in (organism_skipped or []) if x and x.get("name")]
        ctx["organism_skipped"] = _skips
        ctx["organism_skipped_names"] = ", ".join(x["name"] for x in _skips)
        ctx["organism_skipped_by_name"] = {x["name"]: x["reason"] for x in _skips}
        ctx["organism_label"] = (
            ("%s (%s)" % (organism, genome_build)) if (organism and genome_build)
            else (organism or genome_build or ""))
        ctx["ortholog_banner"] = ortholog_banner or ""
        import html as _html_ce
        ctx["curated_error"] = _html_ce.escape(curated_error) if curated_error else ""
        ctx["ortholog_overview"] = ortholog_overview or ""
        ctx["ortholog_note"] = ortholog_note or (
            ("Gene selection below rests on HUMAN-ortholog literature: every "
             "GeneRIF, PubMed and PaperQA lookup for this %s run was made on "
             "the human ortholog of each gene. See Section 14 for the full "
             "disclosure and the per-gene mapping."
             % (organism or organism_common or "non-human"))
            if ortholog_banner else "")

        ctx["homer_known"] = homer_known
        # GIGGLE Section 9: two static point plots embedded as data URLs
        ctx["giggle_up"] = _png_to_data_url(giggle_up_png) if giggle_up_png else ""
        ctx["giggle_down"] = _png_to_data_url(giggle_down_png) if giggle_down_png else ""
        ctx["peakheatmap"] = _png_to_data_url(peakheatmap_png) if peakheatmap_png else ""
        ctx["giggle_up_label"] = giggle_up_label
        ctx["giggle_down_label"] = giggle_down_label

        # JSON figure artifacts
        def _load_json(path):
            return json.loads(Path(path).read_text()) if path and Path(path).exists() else None

        ctx["volcano_atac"] = _load_json(volcano_atac_json)
        ctx["volcano_rna"]  = _load_json(volcano_rna_json)
        ctx["tf_rank"]      = _load_json(tf_rank_json)
        ctx["shared_dotplot"] = _load_json(shared_dotplot_json)
        ctx["per_tf_heatmap"] = _load_json(per_tf_heatmap_json)
        # Caption N comes from the figure itself, so the count always matches the heatmap's
        # actual TF axis instead of a hardcoded 6.  The blank category the heatmap uses to
        # separate the ranked panel from the SE-derived factors is a column but not a factor,
        # so it is not counted (2026-10-17).
        try:
            _x = ctx["per_tf_heatmap"]["data"][0]["x"]
            ctx["per_tf_heatmap_n"] = sum(1 for v in _x if str(v).strip())
        except Exception:
            ctx["per_tf_heatmap_n"] = 0
        ctx["trn"]          = _load_json(trn_json)
        ctx["signal_heatmap_data_url"] = _png_to_data_url(signal_heatmap_png) if signal_heatmap_png else ""
        #  R-script plot PNGs (Section 5 rank/volcano + Section 7 heatmap)
        ctx["rank_png"]    = _png_to_data_url(rank_png)    if rank_png    else ""
        ctx["volcano_png"] = _png_to_data_url(volcano_png) if volcano_png else ""
        ctx["heatmap_png"] = _png_to_data_url(heatmap_png) if heatmap_png else ""
        # Interactive Plotly JSON (preferred over PNG when available)
        def _load_json(p):
            try:
                import json as _j
                return _j.loads(Path(p).read_text()) if (p and Path(p).exists()) else None
            except Exception:
                return None
        ctx["rank_plotly"]    = _load_json(rank_json)
        ctx["volcano_plotly"] = _load_json(volcano_json)
        ctx["heatmap_plotly"] = _load_json(heatmap_json)
        ctx["heatmap_median_plotly"] = _load_json(heatmap_median_json)
        #  Cytoscape: read the cyjs.json and pass elements + style straight
        # into the template. We DON'T strip the legacy `trn` context — the
        # template prefers trn_cyjs when present and falls back to trn otherwise.
        import json as _json
        if trn_cyjs_json and Path(trn_cyjs_json).exists():
            try:
                ctx["trn_cyjs"] = _json.loads(Path(trn_cyjs_json).read_text())
            except Exception:
                ctx["trn_cyjs"] = None
        else:
            ctx["trn_cyjs"] = None
        # Section 10b (2026-09-10): SE -> TF -> Target TRN; omitted when absent.
        ctx["se_trn"] = None
        if se_trn_json and Path(se_trn_json).exists():
            try:
                _se = _json.loads(Path(se_trn_json).read_text())
                if isinstance(_se, dict) and _se.get("elements"):
                    ctx["se_trn"] = _se
            except Exception:
                ctx["se_trn"] = None
        # Section-10 target-gene table (under the TRN graph). Curated rerank
        # order. Columns: Gene | score | Regulated by TFs | Evidence source |
        # GeneRIF (cancer-field-specific PMIDs).
        # GeneRIF fetched once (GeneGPT) then cached next to curated_targets.json.
        ctx["curated_target_rows"] = []
        if curated_targets_json and Path(curated_targets_json).exists():
            try:
                from .narrative import build_target_table_rows
                import collections as _collections
                _ct = _json.loads(Path(curated_targets_json).read_text())
                # All detected TRN regulators per gene (Regulated by TFs column)
                _trn = {}
                _tsp = Path(curated_targets_json).parent / "targets_summary.json"
                if _tsp.exists():
                    _tsj = _json.loads(_tsp.read_text())
                    _rg = _collections.defaultdict(list)
                    for _tf, _lst in (_tsj.items() if isinstance(_tsj, dict) else []):
                        for _x in (_lst or []):
                            _g = _x.get("Gene Name")
                            if _g and _tf not in _rg[_g]:
                                _rg[_g].append(_tf)
                    _trn = dict(_rg)
                _cache = str(Path(curated_targets_json).parent
                             / "target_table_rows_cache.json")
                ctx["curated_target_rows"] = build_target_table_rows(
                    _ct, field=cancer_type, cache_path=_cache, trn_regulators=_trn)
            except Exception:
                ctx["curated_target_rows"] = []
            # followup 2026-09-28 (fix 1): genes whose NCBI GeneRIF record could not be read --
            # flagged rows (legend) and the scored genes not shown (the notice under the table)
            ctx["generif_fetch_failed_rows"] = [
                r.get("gene") for r in ctx["curated_target_rows"]
                if isinstance(r, dict) and "GeneRIF fetch failed" in str(r.get("evidence_flag") or "")]
            try:
                _gfj = Path(curated_targets_json).parent / "generif_fetch.json"
                if _gfj.exists():
                    _gfd = json.loads(_gfj.read_text())
                    if isinstance(_gfd, dict) and _gfd.get("failed"):
                        _gff = [str(g) for g in _gfd["failed"]]
                        _gfi = [str(g) for g in (_gfd.get("in_targets") or [])]
                        # review 2026-09-28: which of the others could have been targets (a record
                        # without "could_emit": every one of them)
                        _gfc = ([str(g) for g in (_gfd.get("could_emit") or [])] if "could_emit" in _gfd else _gff)
                        ctx["generif_fetch_notice"] = {"failed": _gff, "n": _gfd.get("n") or "", "in_targets": _gfi,
                                                       "missing": [g for g in _gff if g in _gfc and g not in _gfi],
                                                       "cannot": [g for g in _gff if g not in _gfc and g not in _gfi]}
            except Exception:
                pass
            # followup r3 (post-deploy finding 1): capped candidate genes whose live PubMed rescue check failed
            # (the notice under the table; nothing is added when no check failed)
            try:
                _prj = Path(curated_targets_json).parent / "pubmed_rescue.json"
                if _prj.exists():
                    _prd = json.loads(_prj.read_text())
                    if isinstance(_prd, dict) and _prd.get("failed"):
                        _prf = [str(g) for g in _prd["failed"]]
                        _pri = [str(g) for g in (_prd.get("in_targets") or [])]
                        _prc = [str(g) for g in (_prd.get("could_emit") or [])] if "could_emit" in _prd else _prf
                        ctx["pubmed_rescue_notice"] = {"failed": _prf, "n": _prd.get("n_candidates") or "",
                                                       "in_targets": _pri,
                                                       "missing": [g for g in _prf if g in _prc and g not in _pri],
                                                       "cannot": [g for g in _prf if g not in _prc and g not in _pri]}
            except Exception:
                pass
        # Section 13: the axis validation.  None = the step did not run; a dict with no
        # "models" = it ran and no arm produced a result -- the section states which.
        # A malformed axis file must cost Section 13, never the whole report: anything _axis_ctx
        # did not foresee becomes the section's stated reason (review 2, finding 55).
        try:
            ctx["axis"] = _axis_ctx(axis_json, ctx["se_trn"], ctx["pipeline"]["llm_domain"])
        except Exception as _axe:
            ctx["axis"] = {"models": [], "arm_notes": [], "reason": "",
                           "state": "left a result file that could not be read into the report (%s: %s)"
                                    % (type(_axe).__name__, str(_axe)[:200])}
        if ctx["axis"] is None and str(axis_step_error or "").strip():
            ctx["axis"] = {"models": [], "arm_notes": [], "step_error": True,
                           "state": "stopped with an error before it wrote its result file "
                                    "(as the run manifest records it)",
                           "reason": str(axis_step_error).strip()[:600]}
        ctx["axis_note"] = _html_ce.escape(axis_note) if axis_note else ""
        ctx["per_tf_profile"] = _load_json(per_tf_profile_json)
        ctx["citation_audit"] = citation_audit
        ctx["signal_profile_data_url"] = _png_to_data_url(signal_profile_png) if signal_profile_png else ""

        # Tables
        if tf_table_csv and Path(tf_table_csv).exists():
            df = pd.read_csv(tf_table_csv)
            _cols = list(df.columns)
            ctx["tf_table"] = {
                "data": df.values.tolist(),
                "columns": [{"title": c} for c in _cols],
                "rank_col": (_cols.index("Rank") if "Rank" in _cols else 0),
            }
        else:
            ctx["tf_table"] = None

        # Locus tracks -- inline PNG
        ctx["locus_tracks"] = []
        for t in (locus_tracks or []):
            data_url = _png_to_data_url(t.get("png_path", ""))
            if data_url:
                ctx["locus_tracks"].append({
                    "tf": t.get("tf", ""),
                    "gene": t.get("gene", ""),
                    "region": t.get("region", ""),
                    "data_url": data_url,
                })

        # Narrative
        if narrative_md_path and Path(narrative_md_path).exists():
            narr_md = Path(narrative_md_path).read_text()
        else:
            narr_md = "*(narrative not generated)*"
        # Split the References list out of the narrative body so the Notable
        # Findings cards can be placed ABOVE References (fixed report order:
        # Analysis Result -> Notable Findings -> References).
        if "## References" in narr_md:
            _body_md, _ref_tail = narr_md.split("## References", 1)
            ctx["narrative_html"] = _md_to_html(_body_md.rstrip())
            ctx["references_html"] = _md_to_html("## References" + _ref_tail)
        else:
            ctx["narrative_html"] = _md_to_html(narr_md)
            ctx["references_html"] = ""

        # Notable Findings cards (JSON written beside the narrative md)
        ctx["notable_cards"] = []
        if narrative_md_path:
            _cp = Path(narrative_md_path).parent / "notable_cards.json"
            if _cp.exists():
                try:
                    ctx["notable_cards"] = json.loads(_cp.read_text())
                except Exception:
                    ctx["notable_cards"] = []
            # followup r3 (post-deploy finding 2): the cards of a curated run that could not be rebuilt with the
            # evidence rules were WITHHELD by the runner (notable_cards_status.json beside them) -- say so
            try:
                _csp = Path(narrative_md_path).parent / "notable_cards_status.json"
                if _csp.exists():
                    _csd = json.loads(_csp.read_text())
                    if isinstance(_csd, dict) and _csd.get("withheld"):
                        ctx["notable_cards_notice"] = {"reason": str(_csd.get("reason") or ""),
                                                       "n_withheld": _csd.get("n_withheld")}
                        ctx["notable_cards"] = []
            except Exception:
                pass
        # evaudit 2026-09-27: each card shows its target's evidence-audit
        # verdict (curated_targets.json); nothing is added for a run without one
        if ctx["notable_cards"] and curated_targets_json and Path(curated_targets_json).exists():
            try:
                # fix2 2026-09-28: only the flags a reader must see (weak /
                # invalid kept / audit-suspect / unaudited) reach the cards
                _all = json.loads(Path(curated_targets_json).read_text())
                _aud = {str(r.get("gene")): r for r in _all
                        if isinstance(r, dict) and r.get("audit_flag")}
                # fix3 2026-09-28: the evidence-layer flags too
                _evf = {str(r.get("gene")): r for r in _all
                        if isinstance(r, dict) and r.get("evidence_flag")}
                for _c in ctx["notable_cards"]:
                    _e = _evf.get(str(_c.get("gene"))) if isinstance(_c, dict) else None
                    if _e:
                        _c["evidence_flag"] = str(_e.get("evidence_flag") or "")
                        _c["evidence_flag_note"] = str(_e.get("evidence_flag_note") or "")
                    _a = _aud.get(str(_c.get("gene"))) if isinstance(_c, dict) else None
                    if _a:
                        _c["audit_flag"] = str(_a.get("audit_flag") or "")
                        _c["audit_verdict"] = str(_a.get("audit_verdict") or "")
                        _c["audit_reason"] = str(_a.get("audit_reason") or "")
                        _c["audit_pmid"] = str(_a.get("audit_pmid") or "")
            except Exception:
                pass
        # followup 2026-09-28: a card's own flags (its GeneRIF record could not be read / a NEC
        # paper check could not run) join the row's evidence flag, each label once
        for _c in ctx["notable_cards"] or []:
            if not isinstance(_c, dict) or not isinstance(_c.get("card_flags"), list):
                continue
            for _f in _c["card_flags"]:
                _lab = str((_f or {}).get("label") or "")
                _labs = [x for x in str(_c.get("evidence_flag") or "").split("; ") if x]
                if not _lab or _lab in _labs:
                    continue
                _c["evidence_flag"] = "; ".join(_labs + [_lab])
                _c["evidence_flag_note"] = "; ".join(
                    x for x in (str(_c.get("evidence_flag_note") or ""), str((_f or {}).get("note") or "")) if x)

        if executive_summary_md:
            ctx["executive_summary"] = _md_to_html(executive_summary_md)
        else:
            # Use first 2 paragraphs of narrative as fallback
            paras = [p for p in narr_md.split("\n\n") if p.strip()][:2]
            ctx["executive_summary"] = _md_to_html("\n\n".join(paras))

        # Manifest
        manifest = {
            "generated_at": ctx["generated_at"],
            "title": report_title,
            "study": study_label,
            "n_tracks": len(ctx["locus_tracks"]),
            "has_volcano_atac": ctx["volcano_atac"] is not None,
            "has_volcano_rna": ctx["volcano_rna"] is not None,
            "has_tf_rank": ctx["tf_rank"] is not None,
            "has_trn": ctx["trn"] is not None,
            "has_signal_heatmap": bool(ctx["signal_heatmap_data_url"]),
            "has_per_tf_profile": bool(ctx.get("per_tf_profile")),
            "has_signal_profile": bool(ctx["signal_profile_data_url"]),
            "trn_stats": ({"n_tf": ctx["trn"].get("n_tf"),
                           "n_gene": ctx["trn"].get("n_gene"),
                           "n_edge": ctx["trn"].get("n_edge")}
                          if ctx["trn"] else None),
            "n_tf_table_rows": len(ctx["tf_table"]["data"]) if ctx["tf_table"] else 0,
            "n_axis_models": len((ctx["axis"] or {}).get("models") or []),
        }
        ctx["manifest_json"] = json.dumps(manifest, indent=2)

        html = tpl.render(**ctx)
        Path(output_html).write_text(html, encoding="utf-8")
        return {
            "ok": True,
            "output_html": output_html,
            "size_kb": round(Path(output_html).stat().st_size / 1024, 1),
            "manifest": manifest,
        }
