"""The SE -> closest-gene end of Section 13: the proximity edge itself, with no factor in between.

se_trn/se_cyjs.json draws, per super-enhancer, an SE -> SE_gene edge to that SE's closest gene.  That
edge makes no claim about a transcription factor, so it is tested as what it says: scramble the
super-enhancer and ask whether the closest gene's read-out falls further than it does when a
size-matched stretch of the same window is scrambled instead.

Per super-enhancer:
  * window      the enhancer layer's own geometry -- ws = SE centre - HALF, 524 kb.
  * KO          the WHOLE super-enhancer scrambled, EXCEPT the closest gene's promoter
                (TSS +/- 2 kb) and its exons (+/- 200 bp) where they fall inside the SE.  Those are
                spared because the TSS sits inside the SE for many of these pairs (TRIB1 -2.6 kb,
                RABGAP1L -0.2 kb, UAP1 -0.9 kb): scrambling the promoter would drop the read-out for
                a reason that has nothing to do with the enhancer.
  * controls    21 replicates.  Each mirrors the KO's block structure -- same number of blocks, same
                lengths -- placed at random positions in the window, avoiding the SE, the promoter,
                the exons, the window edges and each other.  So a control scrambles as many bases of
                the same window, in the same pieces, somewhere the SE is not.
  * read-outs   the gene's exons (RNA), prom1k and tss500 (H3K4me3 / DNase), plus the SE region and
                the whole window for reference.

The arm format pairs each pseudo-factor with one 'ko' and one 'ctl' prediction, so the 22 edited
sequences (the KO and 21 controls) are PACKED into 11 pseudo-factors: 'SE' carries the KO in its ko
slot and control 01 in its ctl slot, 'R01'..'R10' carry controls 02..21.  For this layer the slot
names are carriers, not a ko-vs-ctl contrast, so the edits JSON records the mapping under 'roles'
and the statistic reads the raw rows through it.  Packing rather than repeating the KO in every
pair is what keeps the layer at 23 predictions per SE instead of 41.

Own seed (3), own edits file, own result files: the stored enhancer-end and target-end predictions
and their random streams are untouched.
"""
from __future__ import annotations
import collections, hashlib, json
import numpy as np
from mrag_axis.stat import config as C
from mrag_axis.prep import edits_target as PT
from mrag_axis.prep.edits_enhancer import HG38_FA
from mrag_axis.prep.edits_target import EDGE_MARGIN, EXON_PAD, PROM_EXCL, REFGENE_GTF

SEED = 3
N_CONTROL = 21          # 21 controls + the KO = 22 edited sequences = 11 packed pseudo-factors
SE_PAD = 1000           # a control block keeps this clear of the super-enhancer itself


def _merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _subtract(lo, hi, cut):
    """[lo, hi) minus the merged intervals `cut`."""
    out, p = [], lo
    for a, b in _merge(cut):
        if b <= lo or a >= hi:
            continue
        a, b = max(a, lo), min(b, hi)
        if a > p:
            out.append([p, a])
        p = max(p, b)
    if p < hi:
        out.append([p, hi])
    return out


def _scramble(wt, blocks, rng):
    """[[pos, shuffled]] for each window-relative block."""
    ed = []
    for a, b in blocks:
        sh = list(wt[a:b])
        rng.shuffle(sh)
        ed.append([int(a), "".join(sh)])
    return ed


def _place(wt, lengths, excl, window, rng, tries=20000):
    """One control replicate: a block per length, avoiding `excl` and each other.  (edits, ok)."""
    taken, ed = list(excl), []
    for L in lengths:
        for _ in range(tries):
            q = int(rng.integers(EDGE_MARGIN, window - L - EDGE_MARGIN))
            if not any(q < b and a < q + L for a, b in taken):
                taken.append([q, q + L])
                sh = list(wt[q:q + L])
                rng.shuffle(sh)
                ed.append([q, "".join(sh)])
                break
        else:
            return ed, False
    return ed, True


def se_rows(run_output):
    """[(SE label, gene, chrom, start, stop)] from the TRN's SE and SE_gene nodes."""
    p = C.manifest(run_output).get(("se_trn", "se_cyjs_json")) or f"{run_output}/se_trn/se_cyjs.json"
    try:
        els = json.load(open(p))["elements"]
    except (OSError, ValueError, KeyError) as e:
        return None, f"se_trn/se_cyjs.json unreadable: {type(e).__name__}: {e}"
    N = {e["data"]["id"]: e["data"] for e in els if "source" not in e["data"]}
    E = [e["data"] for e in els if "source" in e["data"]]
    name = lambda i: N[i].get("name", i) if i in N else i
    rows = []
    for e in E:
        if e.get("kind") != "SE_gene":
            continue
        d = N.get(e["source"]) or {}
        reg = d.get("region") or ""
        try:
            ch, a = reg.split(":")
            s, t = (int(x) for x in a.split("-"))
        except ValueError:
            continue
        rows.append((e["source"], name(e["target"]), ch, s, t))
    if not rows:
        return None, "the TRN draws no SE -> closest-gene edge with a usable region"
    return sorted(set(rows)), None


def build(run, seed=SEED, n_control=N_CONTROL, genome_fa=HG38_FA, gtf=REFGENE_GTF,
          agent_root=C.AGENT_ROOT):
    from pyfaidx import Fasta
    out = C.run_output(run, agent_root)
    G = C.corgi_channels()
    WINDOW, HALF = G["WINDOW"], G["HALF"]
    res = {"status": "ok", "reason": None, "window": WINDOW, "trn": [], "foreign": [], "items": [],
           "roles": {}}
    rows, why = se_rows(out)
    if not rows:
        return {**res, "status": "skipped", "reason": why, "detail": {"run_output": out}}
    idx, gst = PT.gene_index(gtf)
    if idx is None:
        return {**res, "status": gst["status"], "reason": gst["reason"], "detail": {"gtf": gst}}
    fa = Fasta(genome_fa)
    rng = np.random.default_rng(seed)
    dropped, skipped = [], collections.Counter()
    for se, gene, chrom, start, stop in rows:
        centre = (start + stop) // 2
        ws = centre - HALF
        try:
            clen = len(fa[chrom])
        except Exception:
            skipped["bounds"] += 1; dropped.append((se, gene, f"{chrom} not in {genome_fa}")); continue
        if ws < 0 or ws + WINDOW > clen:
            skipped["bounds"] += 1; dropped.append((se, gene, "524-kb window runs off the chromosome")); continue
        gm = PT.gene_model(gene, chrom, centre, index=idx)
        if gm is None:
            skipped["no_gene_model"] += 1
            dropped.append((se, gene, f"{gene} has no refGene transcript on {chrom}")); continue
        if not PT.tss_in_window(gm["tss"], ws, WINDOW):
            skipped["tss_outside_window"] += 1
            dropped.append((se, gene, "TSS outside [ws+1000, ws+W-1000)")); continue
        wt = str(fa[chrom][ws:ws + WINDOW].seq).upper()
        # Window-relative. Spared inside the SE, and excluded for every control block.
        spare = [[gm["tss"] - PROM_EXCL - ws, gm["tss"] + PROM_EXCL - ws]] + \
                [[a - EXON_PAD - ws, b + EXON_PAD - ws] for a, b in gm["exons"]]
        ko_blocks = _subtract(start - ws, stop - ws, spare)
        if not ko_blocks:
            skipped["se_entirely_spared"] += 1
            dropped.append((se, gene, "the promoter and exons cover the whole super-enhancer")); continue
        lengths = [b - a for a, b in ko_blocks]
        excl = _merge(spare + [[start - SE_PAD - ws, stop + SE_PAD - ws]])
        clip = lambda a, b: [max(a, ws), min(b, ws + WINDOW)]
        rec = {"label": f"D_{se}_{gene}", "source": "se_direct", "se": se, "gene": gene,
               "strand": gm["strand"], "tss": gm["tss"], "chrom": chrom, "ws": ws,
               "se_region": [start, stop], "wt_md5": hashlib.md5(wt.encode()).hexdigest(),
               "n_scrambled_bases": int(sum(lengths)), "n_blocks": len(ko_blocks),
               "windows": {"region": [[start, stop]], "whole": [[ws, ws + WINDOW]],
                           "tss500": [clip(gm["tss"] - 500, gm["tss"] + 500)],
                           "prom1k": [clip(gm["tss"] - 1000, gm["tss"] + 1000)],
                           "exons": [clip(a, b) for a, b in gm["exons"]
                                     if b > ws and a < ws + WINDOW]},
               "tfs": {}}
        ctl = []
        for k in range(n_control):
            ed, ok = _place(wt, lengths, excl, WINDOW, rng)
            if not ok:
                skipped["control_fallback"] += 1
            ctl.append(ed)
        ko = _scramble(wt, ko_blocks, rng)
        # Pack: 'SE' = (KO, control 01); 'R01'..'R10' = controls 02..21.
        slots = [("SE", ko, ctl[0], "se_scramble", "control_01")]
        for j in range(1, n_control, 2):
            slots.append((f"R{(j + 1) // 2:02d}", ctl[j], ctl[j + 1],
                          f"control_{j + 1:02d}", f"control_{j + 2:02d}"))
        for tf, a_ed, b_ed, a_role, b_role in slots:
            rec["tfs"][tf] = {"ko": a_ed, "ctl": b_ed, "n_motifs": len(a_ed),
                              "tf_class": "se_direct", "claimed": tf == "SE"}
            res["roles"][tf] = {"ko": a_role, "ctl": b_role}
        res["items"].append(rec)
    n_pairs = sum(len(i["tfs"]) for i in res["items"])
    res["detail"] = {"run_output": out, "gtf": gst, "seed": seed, "n_control": n_control,
                     "se_pairs": len(rows), "n_items": len(res["items"]),
                     "skipped": dict(skipped), "dropped_items": dropped,
                     "n_pairs": n_pairs, "n_calls": len(res["items"]) + 2 * n_pairs,
                     "genome_fa": genome_fa}
    if dropped:
        res["status"], res["reason"] = "partial", "; ".join(f"{s}/{g}: {w}" for s, g, w in dropped)
    return res


# --------------------------------------------------------------- distance-matched controls -------
NEAR_SEED = 4
NEAR_TOL = 1000         # a matched control's nearest edge is within 1 kb of the knockout's
N_NEAR = 22             # 11 packed pseudo-factors x 2 slots


def _near_edge(blocks, t):
    """Distance from the window-relative TSS `t` to the nearest edge of a block set (0 if inside)."""
    return min(0 if a <= t < b else min(abs(a - t), abs(b - t)) for a, b in blocks)


def _fill_near(free, t, d_ko, total, tol, rng):
    """One distance-matched control: `total` scrambled bases, the nearest of them within `tol` of
    `d_ko` from the window-relative TSS `t`, laid into the free intervals `free`.

    The knockout's block pattern cannot simply be translated to a matched distance -- its own gene's
    exons occupy most of what lies that close on the other side, which left 5 of the 20 SE/gene pairs
    with no matched position at all.  So what is matched is what the confound is about: how many
    bases are scrambled, and how close to the transcription start site the nearest of them is.  The
    first block is placed at the matched distance in whichever free interval reaches it; the rest is
    laid into free intervals drawn at random, so the replicates differ in everything but those two
    quantities.
    """
    near = []
    for a, b in free:
        for side in (1, -1):
            lo = max(a, t + side * (d_ko + tol)) if side > 0 else a
            hi = min(b, t - (d_ko - tol)) if side < 0 else b
            if side > 0:
                lo, hi = max(a, t + d_ko - tol), b
            else:
                lo, hi = a, min(b, t - d_ko + tol)
            if hi - lo <= 0:
                continue
            # positions whose distance from t is within tol of d_ko, inside this interval
            if side > 0:
                s0, s1 = max(lo, t + d_ko - tol), min(hi, t + d_ko + tol)
                if s1 > s0:
                    near.append((s0, s1, a, b, 1))
            else:
                e0, e1 = max(lo, t - d_ko - tol), min(hi, t - d_ko + tol)
                if e1 > e0:
                    near.append((e0, e1, a, b, -1))
    if not near:
        return None
    s0, s1, ia, ib, side = near[int(rng.integers(len(near)))]
    blocks, left = [], total
    if side > 0:
        q = int(rng.integers(s0, s1)) if s1 > s0 + 1 else int(s0)
        L = min(left, ib - q)
    else:
        e = int(rng.integers(e0 := s0, s1)) if s1 > s0 + 1 else int(s0)
        L = min(left, e - ia)
        q = e - L
    if L <= 0:
        return None
    blocks.append([q, q + L]); left -= L
    pool = [[a, b] for a, b in free]
    rng.shuffle(pool)
    for a, b in pool:
        if left <= 0:
            break
        for bq, be in blocks:
            if bq < b and a < be:                 # already used part of this interval
                a = max(a, be)
        if b - a <= 0:
            continue
        L = min(left, b - a)
        q = int(rng.integers(a, b - L)) if b - L > a else a
        if any(q < be and bq < q + L for bq, be in blocks):
            continue
        blocks.append([q, q + L]); left -= L
    if left > 0:
        return None
    return sorted(blocks)


def build_near(run, seed=NEAR_SEED, n_near=N_NEAR, genome_fa=HG38_FA, gtf=REFGENE_GTF,
               agent_root=C.AGENT_ROOT, tol=NEAR_TOL, tries=200000):
    """A second control family for the SE -> closest-gene end: DISTANCE-MATCHED controls.

    The first family places a control anywhere in the window, which never samples the distance the
    knockout itself sits at: the knockout starts at the edge of the spared promoter, 2 kb from the
    TSS, while a uniformly placed control lands 5 kb away or more (measured on all 20 SE/gene pairs).
    At the H3K4me3 read-out that difference alone can carry the result, so the knockout is also
    compared against controls that sit as close to the TSS as it does.

    A matched control is the knockout's own block pattern TRANSLATED along the window -- same number
    of blocks, same lengths, same gaps -- to a position whose nearest edge is within `tol` of the
    knockout's, avoiding the super-enhancer, the promoter, the exons and the window edges.  Matched
    distance on the super-enhancer's own side is where the super-enhancer is, so these land on the
    other side of the gene; that is the point of the family.

    Items carry the labels, windows and wt_md5 of build()'s, so the knockout already predicted for
    that item is read from the first family's result file and is not predicted again.
    """
    from pyfaidx import Fasta
    out = C.run_output(run, agent_root)
    G = C.corgi_channels()
    WINDOW, HALF = G["WINDOW"], G["HALF"]
    res = {"status": "ok", "reason": None, "window": WINDOW, "trn": [], "foreign": [], "items": [],
           "roles": {}}
    rows, why = se_rows(out)
    if not rows:
        return {**res, "status": "skipped", "reason": why, "detail": {"run_output": out}}
    idx, gst = PT.gene_index(gtf)
    if idx is None:
        return {**res, "status": gst["status"], "reason": gst["reason"], "detail": {"gtf": gst}}
    fa = Fasta(genome_fa)
    rng = np.random.default_rng(seed)
    dropped, skipped, short = [], collections.Counter(), []
    for se, gene, chrom, start, stop in rows:
        centre = (start + stop) // 2
        ws = centre - HALF
        try:
            clen = len(fa[chrom])
        except Exception:
            skipped["bounds"] += 1; dropped.append((se, gene, f"{chrom} not in {genome_fa}")); continue
        if ws < 0 or ws + WINDOW > clen:
            skipped["bounds"] += 1; dropped.append((se, gene, "524-kb window runs off the chromosome")); continue
        gm = PT.gene_model(gene, chrom, centre, index=idx)
        if gm is None or not PT.tss_in_window(gm["tss"], ws, WINDOW):
            skipped["no_gene_model"] += 1
            dropped.append((se, gene, "no refGene transcript, or the TSS is outside the window")); continue
        wt = str(fa[chrom][ws:ws + WINDOW].seq).upper()
        spare = [[gm["tss"] - PROM_EXCL - ws, gm["tss"] + PROM_EXCL - ws]] + \
                [[a - EXON_PAD - ws, b + EXON_PAD - ws] for a, b in gm["exons"]]
        ko_blocks = _subtract(start - ws, stop - ws, spare)
        if not ko_blocks:
            skipped["se_entirely_spared"] += 1
            dropped.append((se, gene, "the promoter and exons cover the whole super-enhancer")); continue
        t = gm["tss"] - ws
        d_ko = _near_edge(ko_blocks, t)
        excl = _merge(spare + [[start - SE_PAD - ws, stop + SE_PAD - ws]])
        free = _subtract(EDGE_MARGIN, WINDOW - EDGE_MARGIN, excl)
        total = sum(b - a for a, b in ko_blocks)
        # The closest a control can be placed at all.  For most of these pairs that is NOT d_ko: the
        # super-enhancer abuts the promoter and the gene's own exons take the other side, so no
        # equally close non-exonic sequence exists (measured: 13 of the 20 SE/gene pairs).  The
        # family is therefore placed as close as the window allows, and the distance it reached is
        # recorded per item so the residual gap is visible wherever the numbers are read.
        d_floor = min((0 if a <= t < b else min(abs(a - t), abs(b - t))) for a, b in free)
        band = max(d_ko, d_floor)
        shifts = []
        for _ in range(tries):
            if len(shifts) >= n_near:
                break
            blk = _fill_near(free, t, band, total, tol, rng)
            if blk and blk not in shifts:
                shifts.append(blk)
        if not shifts:
            skipped["no_matched_position"] += 1
            dropped.append((se, gene, f"no position in the window matches the knockout's {d_ko} bp "
                                      f"from the TSS without hitting the SE, the promoter or an exon"))
            continue
        if len(shifts) < n_near:
            short.append((se, gene, len(shifts)))
        clip = lambda a, b: [max(a, ws), min(b, ws + WINDOW)]
        rec = {"label": f"D_{se}_{gene}", "source": "se_near", "se": se, "gene": gene,
               "strand": gm["strand"], "tss": gm["tss"], "chrom": chrom, "ws": ws,
               "se_region": [start, stop], "wt_md5": hashlib.md5(wt.encode()).hexdigest(),
               "ko_near_edge_bp": int(d_ko), "n_matched": len(shifts),
               "control_near_edge_bp": int(band), "control_floor_bp": int(d_floor),
               "achieved_near_edge_bp": [int(_near_edge(b_, t)) for b_ in shifts],
               "windows": {"region": [[start, stop]], "whole": [[ws, ws + WINDOW]],
                           "tss500": [clip(gm["tss"] - 500, gm["tss"] + 500)],
                           "prom1k": [clip(gm["tss"] - 1000, gm["tss"] + 1000)],
                           "exons": [clip(a, b) for a, b in gm["exons"]
                                     if b > ws and a < ws + WINDOW]},
               "tfs": {}}
        ed = [_scramble(wt, b, rng) for b in shifts]
        while len(ed) < n_near:                  # fewer matched positions than slots: reuse them
            ed.append(_scramble(wt, shifts[len(ed) % len(shifts)], rng))
        for j in range(0, n_near, 2):
            tf = f"N{j // 2 + 1:02d}"
            rec["tfs"][tf] = {"ko": ed[j], "ctl": ed[j + 1], "n_motifs": len(ed[j]),
                              "tf_class": "se_near", "claimed": False}
            res["roles"][tf] = {"ko": f"near_{j + 1:02d}", "ctl": f"near_{j + 2:02d}"}
        res["items"].append(rec)
    n_pairs = sum(len(i["tfs"]) for i in res["items"])
    res["detail"] = {"run_output": out, "gtf": gst, "seed": seed, "n_near": n_near, "tol": tol,
                     "note": ("controls are placed as close to the TSS as the window allows; "
                              "control_near_edge_bp vs ko_near_edge_bp per item says how close"),
                     "se_pairs": len(rows), "n_items": len(res["items"]),
                     "skipped": dict(skipped), "dropped_items": dropped,
                     "fewer_matched_than_slots": short,
                     "n_pairs": n_pairs, "n_calls": len(res["items"]) + 2 * n_pairs,
                     "genome_fa": genome_fa}
    if dropped or short:
        res["status"] = "partial"
        res["reason"] = "; ".join([f"{s}/{g}: {w}" for s, g, w in dropped]
                                  + [f"{s}/{g}: only {n} matched positions" for s, g, n in short])
    return res


def published_form(res):
    return {"window": res["window"], "trn": res["trn"], "foreign": res["foreign"],
            "roles": res["roles"], "items": res["items"]}
