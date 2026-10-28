"""The SE -> closest-gene end of Section 13: the TRN's proximity edges, measured as what they claim.

An SE -> SE_gene edge of se_trn/se_cyjs.json names no transcription factor -- it says this
super-enhancer is the closest one to that gene.  So it is tested without a factor: scramble the
super-enhancer (sparing the gene's own promoter and exons) and ask whether the gene's read-out falls
further than when a size-matched stretch of the same 524-kb window is scrambled instead.

Two control families, both built by mrag_axis.prep.edits_se_direct:

  free  21 controls placed anywhere in the window.
  near  22 controls placed as close to the transcription start site as the window allows.  The
        distance the knockout itself sits at is mostly unreachable -- the super-enhancer abuts the
        promoter and the gene's own exons take the other side, so for 13 of the 20 SE/gene pairs of
        the 2026-10 runs no equally close non-exonic sequence exists.  Each row therefore carries
        both distances, and a pass is reported per family: the free family alone turned out to carry
        a distance artefact at the H3K4me3 read-out (AlphaGenome 10/10 of the OC2 pairs against the
        free family, 4/10 against the near one), which is why both are reported and never pooled.

A super-enhancer PASSES a family when its knockout's summed signal is strictly the lowest of that
family -- the same "below every member of its comparison group" rule beats() applies at the other
two ends, here with the controls as the group.  One-sided exact p under exchangeability: 1/22 for
the free family, 1/23 for the near one.  No Fisher test is quoted: like the triples, this end is
DESCRIPTIVE, it is not part of the Holm family the section's headline rests on.
"""
import math
import statistics

#: the read-outs the section reports for this end.  'dnase' is measured and
#: recorded as well, so a later read can see it, but the two reported ones are the gene's transcript
#: and the H3K4me3 at its promoter.
REPORTED = ("rna", "prom")
#: this end has no promoter peak (it needs none), so a read-out defined over the target end's 'peak'
#: window is taken over tss500, which every prediction of these layers stores.
WINDOW_FIX = {"peak": "tss500"}


def _sum(row, meta, track_sub, window):
    """The summed absolute signal of one stored prediction over one read-out window."""
    tot, ok = 0.0, False
    for k in (k for k in meta if track_sub in k):
        d = row["sums"].get(k)
        if d is not None and window in d:
            tot += abs(d[window]); ok = True
    return tot if ok else None


def ko_near_edge(item):
    """How far the knockout's nearest scrambled base sits from the gene's TSS, in bp.

    Read off the FREE family's own item, so the distance is reported even for a super-enhancer the
    near family could not place a control for (PRAD's NSMCE2) -- the row then shows the knockout's
    distance against "no near control", which is the honest reading, rather than an empty cell.
    """
    try:
        tss = int(item["tss"]) - int(item["ws"])
        blocks = [(int(p), int(p) + len(b)) for p, b in (item["tfs"]["SE"]["ko"])]
    except (KeyError, TypeError, ValueError):
        return None
    if not blocks:
        return None
    return min(0 if a <= tss < b else min(abs(a - tss), abs(b - tss)) for a, b in blocks)


def _by_role(rows, meta, roles, sub, win):
    """{item label: {role: summed signal}} for one read-out."""
    out = {}
    for r in rows:
        role = (roles.get(r["tf"]) or {}).get(r["kind"])
        if role is None or r["kind"] == "wt":
            continue
        v = _sum(r, meta, sub, win)
        if v is not None:
            out.setdefault(r["label"], {})[role] = v
    return out


def score(free_edits, near_edits, results, readouts, models=("ag", "corgi")):
    """The whole end, shaped for the record and the report.  Nothing here raises on missing input.

    free_edits / near_edits  the two edits JSONs (dicts), each carrying 'items' and 'roles'
    results                  {(model, family): stored JSONL path}; a missing file is a missing model
    readouts                 {model: {read-out: (track substring, window)}}
    """
    from mrag_axis.stat import stats as S
    items = {i["label"]: i for i in free_edits.get("items") or []}
    near_items = {i["label"]: i for i in near_edits.get("items") or []}
    if not items:
        return {"status": "skipped", "reason": "no SE -> closest-gene edits were built",
                "rows": [], "summary": {}, "strict": []}
    loaded, per = {}, {}
    for m in models:
        for fam in ("free", "near"):
            loaded[(m, fam)] = S.load(results.get((m, fam)))
    for m in models:
        rf, mf = loaded[(m, "free")]
        rn, mn = loaded[(m, "near")]
        if not rf:
            continue
        for rd, (sub, win) in (readouts.get(m) or {}).items():
            win = WINDOW_FIX.get(win, win)
            fv = _by_role(rf, mf, free_edits.get("roles") or {}, sub, win)
            nv = _by_role(rn, mn, near_edits.get("roles") or {}, sub, win) if rn else {}
            for lab in items:
                d = fv.get(lab) or {}
                ko = d.get("se_scramble")
                if ko is None:
                    continue
                free = [v for k, v in d.items() if k != "se_scramble"]
                near = list((nv.get(lab) or {}).values())
                per[(m, rd, lab)] = {
                    "ko": ko,
                    "free": {"n": len(free), "passed": bool(free) and all(ko < c for c in free),
                             "drop": (math.log2((ko + 1) / (statistics.median(free) + 1))
                                      if free else None)},
                    "near": {"n": len(near), "passed": bool(near) and all(ko < c for c in near),
                             "drop": (math.log2((ko + 1) / (statistics.median(near) + 1))
                                      if near else None)}}
    rds = [rd for rd in REPORTED if any(k[1] == rd for k in per)] + \
          [rd for rd in sorted({k[1] for k in per}) if rd not in REPORTED]
    rows = []
    for lab in sorted(items, key=lambda l: (items[l].get("se") or "", l)):
        it = items[lab]
        ni = near_items.get(lab) or {}
        ach = ni.get("achieved_near_edge_bp") or []
        ko_d = ni.get("ko_near_edge_bp")
        if ko_d is None:
            ko_d = ko_near_edge(it)
        rows.append({
            "label": lab, "se": it.get("se"), "gene": it.get("gene"),
            "scrambled_bp": it.get("n_scrambled_bases"), "n_blocks": it.get("n_blocks"),
            "ko_near_edge_bp": ko_d,
            "control_near_edge_bp": [min(ach), max(ach)] if ach else None,
            "models": {m: {rd: per[(m, rd, lab)] for rd in rds if (m, rd, lab) in per}
                       for m in models if any((m, rd, lab) in per for rd in rds)}})
    summary = {}
    for m in models:
        for rd in rds:
            ks = [k for k in per if k[0] == m and k[1] == rd]
            if not ks:
                continue
            summary.setdefault(m, {})[rd] = {
                "n": len(ks), "free_pass": sum(1 for k in ks if per[k]["free"]["passed"]),
                "n_near": sum(1 for k in ks if per[k]["near"]["n"]),
                "near_pass": sum(1 for k in ks if per[k]["near"]["passed"])}
    strict = []
    for r in rows:
        ok = [rd for rd in rds
              if all((r["models"].get(m) or {}).get(rd, {}).get(f, {}).get("passed")
                     for m in models for f in ("free", "near"))]
        if ok:
            strict.append({"se": r["se"], "gene": r["gene"], "read_outs": ok,
                           "reported": [rd for rd in ok if rd in REPORTED]})
    miss = [m for m in models if m not in summary]
    return {"status": "ok" if not miss else "partial",
            "reason": "" if not miss else "no stored predictions for: " + ", ".join(miss),
            "read_outs": rds, "reported": [rd for rd in rds if rd in REPORTED],
            "rows": rows, "summary": summary, "strict": strict,
            "n_items": len(items), "n_near_items": len(near_items)}
