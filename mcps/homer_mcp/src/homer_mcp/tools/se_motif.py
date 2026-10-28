"""homer-mcp: SE -> TF motif presence + SE / linked-gene attributes (report Section 10, SE -> TF -> Target TRN).

MCP step 2a (2026-10-12): the body of the orchestrator's SEMotifNode
(nodes/se_motif.py), moved here unchanged as the MCP tool scan_super_enhancer_motifs; the node calls it through the
orchestrator's MCP transport.

Given the top-N super-enhancers (atac-mcp call_super_enhancers), scans them with
  * the report's TF panel PWMs (combined/top{N}.json symbols, taken from the same JASPAR-extended library the
    Step-2 chain used: homer_motif/<treat>_peak_minmax.motif; a PWM is assigned to a panel TF by
    homer_mcp.tools.tf_alias -- alias-aware (ANDR/ARE/AR-halfsite -> AR, GCR/GRE -> NR3C1, AP-2alpha -> TFAP2A),
    dimers and composites count for each member; the same rule the TF -> Target edges use), and
  * the Section-8 enriched known motifs (homer_known/<treat>/knownResults/known1..n.motif, HOMER PWMs; TF symbol
    derived as in render._build_homer_known, ':' -> '_'), every non-gene enriched motif complemented by the
    next-ranked gene-TF known motif (backfill).
HOMER annotatePeaks.pl -size given -mbed gives every motif instance; an SE -> TF edge = at least one instance
(presence rule, 2026-09-07 decision), with hits/kb and best log-odds kept for edge width / tooltips.
SE attributes: mean DiffBind Fold of peaks inside the SE (fill colour) and the ROSE CLOSEST gene's DESeq2 log2FC."""
from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from homer_mcp.tools.tf_alias import motif_matches as _mmatch, primary_symbol as _psym   # alias-aware (ANDR/ARE -> AR ...)

HOMER_BIN = "/BIO1/MRAG/Agent/bin/tools/homer/bin"
NAME_RE = re.compile(r"^(ENRICHED|PANEL)_([A-Za-z0-9_\-]+)\(")


def _tf_symbol_from_known(name: str) -> str:
    """'Fox:Ebox(Forkhead,bHLH)/Panc1-...' -> 'FOX_EBOX' (composite keeps its own name);
    'ARE(NR)/LNCAP-AR-...' -> 'AR', 'FOXA1(Forkhead)/...' -> 'FOXA1' (alias-normalised)."""
    base = (name.split("(")[0].split("/")[0].strip() or name)
    if ":" in base:
        return re.sub(r"[^A-Z0-9_\-]", "_", base.upper().replace("::", "_").replace(":", "_"))
    return re.sub(r"[^A-Z0-9_\-]", "_", _psym(name))


def _read_motif_lib(path):
    blocks, cur = [], None
    for line in open(path):
        if line.startswith(">"):
            cur = {"hdr": line.rstrip("\n").split("\t"), "rows": []}; blocks.append(cur)
        elif cur is not None and line.strip():
            cur["rows"].append(line)
    return blocks


def _base(name: str) -> str:
    return name.split("(")[0].split("/")[0].strip()


def scan_super_enhancer_motifs(se_top_tsv: str, se_homerpeak: str, panel_json: str, motif_lib: str,
                               known_dir: str, diffbind_csv: str, deseq_csv: str, output_dir: str,
                               genome: str = "hg38", cpu: int = 16, n_known: int = 10,
                               n_backfill_scan: int = 15, timeout_s: int = 3600,
                               homer_bin: str = HOMER_BIN) -> dict[str, Any]:
    """Panel + enriched TF motif instances inside the top-N super-enhancers and the SE attributes (the
    SEMotifNode stage).

    Args:
        se_top_tsv, se_homerpeak: call_super_enhancers outputs (the SE table, the HOMER peak file).
        panel_json: combined/top{N}.json (its "tfs" list, or a plain list).
        motif_lib: the JASPAR-extended library of the Step-2 chain (homer_motif/<treat>_peak_minmax.motif).
        known_dir: homer_known/<treat> (knownResults/known<i>.motif).
        diffbind_csv, deseq_csv: per-peak DiffBind table, per-gene DESeq2 table.
        output_dir: the node's se_rose/motif directory.
        genome, cpu, timeout_s, homer_bin: annotatePeaks.pl settings.
        n_known: Section-8 known motifs taken (default 10); n_backfill_scan: known motifs searched for gene-TF
            backfills (default 15 after n_known).

    Returns {motif_file, motif_map_tsv, hits_bed, annot_txt, se_tf_edges_tsv, se_tf_matrix_tsv, se_genes_tsv,
    n_tf, n_tf_with_hit, n_edges, panel_tfs, enriched_tfs, panel_without_pwm, enriched_backfill}."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    timeout = timeout_s
    se_rows = list(csv.DictReader(open(se_top_tsv), delimiter="\t"))
    if not se_rows:
        raise ValueError("empty se_top_tsv")

    # ---- 1. motif set ------------------------------------------------------
    panel = json.load(open(panel_json))
    panel_tfs = [str(t).upper() for t in (panel.get("tfs") if isinstance(panel, dict) else panel)]
    lib = _read_motif_lib(motif_lib)
    motif_lines, mapping = [], []
    for tf in panel_tfs:
        n = 0
        for b in lib:
            name = b["hdr"][1]
            if _mmatch(name, tf):
                segs = name.split("/"); src = segs[1] if len(segs) > 1 else "lib"
                hdr = list(b["hdr"]); hdr[1] = f"PANEL_{tf}({_base(name)}#{len(mapping) + 1})/{src}/Homer"
                motif_lines.append("\t".join(hdr) + "\n"); motif_lines.extend(b["rows"]); n += 1
                mapping.append(("PANEL", tf, name, hdr[1]))
        if n == 0:
            mapping.append(("PANEL", tf, "", "NO_PWM"))
    known_dir = Path(known_dir) / "knownResults"
    # gene universe (DESeq2) to tell gene TFs from family / composite motifs
    genes = set()
    try:
        for r in csv.DictReader(open(deseq_csv)):
            g = r.get("GENE") or r.get("gene") or r.get("Gene")
            if g:
                genes.add(g.upper())
    except Exception:
        genes = set()
    def _is_gene(sym):
        return ("_" not in sym and ":" not in sym
                and (sym in genes if genes else "-" not in sym))
    enriched_tfs, backfill = [], []
    def _add_known(i, tag):
        mf = known_dir / f"known{i}.motif"
        if not mf.is_file():
            return None
        b = _read_motif_lib(mf)[0]
        name = b["hdr"][1]; sym = _tf_symbol_from_known(name)
        segs = name.split("/"); src = segs[1] if len(segs) > 1 else "HOMER"
        grp = "PANEL" if sym in panel_tfs else "ENRICHED"   # e.g. ARE when AR is in the panel
        hdr = list(b["hdr"]); hdr[1] = f"{grp}_{sym}({_base(name)}#{len(mapping) + 1})/{src}/Homer"
        motif_lines.append("\t".join(hdr) + "\n"); motif_lines.extend(b["rows"])
        mapping.append((grp, sym, name, hdr[1] + tag))
        return grp, sym
    for i in range(1, n_known + 1):
        got = _add_known(i, "")
        if got and got[0] == "ENRICHED" and got[1] not in enriched_tfs:
            enriched_tfs.append(got[1])
    # Sandbox rule (curated19, 2026-09-07): every non-gene enriched motif
    # (composite / family, e.g. Fox:Ebox) is complemented by the next-ranked
    # known motif that is a gene TF (known11 = FOXD3 for ONECUT2-OE), so the
    # target panel keeps as many enriched gene TFs as there are enriched motifs.
    need = sum(1 for t in enriched_tfs if not _is_gene(t))
    i = n_known + 1
    while need > 0 and i <= n_known + int(n_backfill_scan):
        mf = known_dir / f"known{i}.motif"
        if mf.is_file():
            sym = _tf_symbol_from_known(_read_motif_lib(mf)[0]["hdr"][1])
            if _is_gene(sym) and sym not in enriched_tfs and sym not in panel_tfs:
                got = _add_known(i, f" [backfill known{i}]")
                if got and got[0] == "ENRICHED":
                    enriched_tfs.append(sym); backfill.append(f"{sym}(known{i})"); need -= 1
        i += 1
    motif_file = out / "se_motifs.motif"; motif_file.write_text("".join(motif_lines))
    map_tsv = out / "se_motif_map.tsv"
    with open(map_tsv, "w") as fh:
        fh.write("group\tsymbol\toriginal_name\tscan_name\n")
        for m in mapping:
            fh.write("\t".join(m) + "\n")
    if not motif_lines:
        raise ValueError("no PWMs selected for the SE scan")

    # ---- 2. HOMER scan (same flags as reference script 1.rose.py step4) ---
    hits_bed = out / "se_motif_hits.bed"; annot_txt = out / "se_motif_annot.txt"
    cmd = (f"export PATH={homer_bin}:$PATH; perl {homer_bin}/annotatePeaks.pl "
           f"{se_homerpeak} {genome} -m {motif_file} -size given "
           f"-mbed {hits_bed} -cpu {cpu} > {annot_txt} 2> {out / 'annotatePeaks.err'}")
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not hits_bed.is_file():
        raise RuntimeError("annotatePeaks failed: " + (out / "annotatePeaks.err").read_text()[-600:])

    # ---- 3. SE x TF presence table -----------------------------------------
    se_iv = {r["SE"]: (r["chrom"], int(r["start"]), int(r["end"])) for r in se_rows}
    se_len = {k: max(v[2] - v[1], 1) / 1000.0 for k, v in se_iv.items()}
    by_chr = {}
    for sid, (c, s, e) in se_iv.items():
        by_chr.setdefault(c, []).append((s, e, sid))
    def find_se(c, s0, e):
        s1 = s0 + 1                     # -mbed BED start is 0-based; SE START is 1-based
        for a, b, sid in by_chr.get(c, []):
            if s1 >= a and e <= b:
                return sid
        return None
    hits, best = {}, {}
    with open(hits_bed) as fh:
        for line in fh:
            if line.startswith("track"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 6:
                continue
            m = NAME_RE.match(f[3])
            if not m:
                continue
            sid = find_se(f[0], int(f[1]), int(f[2]))
            if sid is None:
                continue
            key = (sid, m.group(1), m.group(2)); sc = float(f[4])
            hits[key] = hits.get(key, 0) + 1; best[key] = max(best.get(key, -1e9), sc)
    tfs = [("PANEL", t) for t in panel_tfs] + [("ENRICHED", t) for t in enriched_tfs]
    edges_tsv = out / "se_tf_edges.tsv"; matrix_tsv = out / "se_tf_matrix.tsv"
    n_edges = 0; tf_with_hit = set()
    with open(edges_tsv, "w") as fo:
        fo.write("SE\trank\tTF\tgroup\tn_hits\thits_per_kb\tmax_score\tpresent\n")
        for r in se_rows:
            sid = r["SE"]
            for g, t in tfs:
                n = hits.get((sid, g, t), 0)
                fo.write("\t".join([sid, r["rank"], t, g, str(n), f"{n / se_len[sid]:.3f}",
                                    f"{best.get((sid, g, t), float('nan')):.3f}", "1" if n else "0"]) + "\n")
                if n:
                    n_edges += 1; tf_with_hit.add(t)
    with open(matrix_tsv, "w") as fo:
        fo.write("TF\tgroup\t" + "\t".join(r["SE"] for r in se_rows) + "\n")
        for g, t in tfs:
            fo.write(f"{t}\t{g}\t" + "\t".join(str(hits.get((r['SE'], g, t), 0)) for r in se_rows) + "\n")

    # ---- 4. SE attributes: DiffBind mean Fold in SE + closest gene DESeq2 ----
    db = list(csv.DictReader(open(diffbind_csv)))
    cols = list(db[0].keys()) if db else []
    def pick(*names):
        for c in cols:
            if c.lower() in names:
                return c
        return None
    chrk, sk, ek, fk, pk = pick("seqnames", "chr", "chrom"), pick("start"), pick("end"), pick("fold"), pick("p.value", "pvalue", "p-value")
    de = {}
    for r in csv.DictReader(open(deseq_csv)):
        g = r.get("GENE") or r.get("gene") or r.get("Gene")
        if g and g not in de:
            de[g] = r
    genes_tsv = out / "se_genes.tsv"
    with open(genes_tsv, "w") as fo:
        fo.write("SE\tregion\tclosest_gene\tn_diffbind_peaks_in_SE\tmean_Fold_SE\tmin_p_SE\tgene_log2FC\tgene_pvalue\tgene_baseMean\n")
        for r in se_rows:
            c, s, e = se_iv[r["SE"]]
            pk_ = [x for x in db if chrk and x[chrk] == c and int(float(x[sk])) < e and int(float(x[ek])) > s] if db else []
            mf = f"{sum(float(x[fk]) for x in pk_) / len(pk_):.3f}" if pk_ and fk else ""
            mp = f"{min(float(x[pk]) for x in pk_):.3g}" if pk_ and pk else ""
            g = r.get("closest_gene", ""); d = de.get(g, {})
            fo.write("\t".join([r["SE"], r["region"], g, str(len(pk_)), mf, mp,
                                d.get("log2FoldChange", d.get("log2FC", "")), d.get("pvalue", ""), d.get("baseMean", "")]) + "\n")

    return {
        "motif_file": str(motif_file), "motif_map_tsv": str(map_tsv), "hits_bed": str(hits_bed),
        "annot_txt": str(annot_txt), "se_tf_edges_tsv": str(edges_tsv), "se_tf_matrix_tsv": str(matrix_tsv),
        "se_genes_tsv": str(genes_tsv), "n_tf": len(tfs), "n_tf_with_hit": len(tf_with_hit),
        "n_edges": n_edges, "panel_tfs": ",".join(panel_tfs), "enriched_tfs": ",".join(enriched_tfs),
        "panel_without_pwm": ",".join(t for g, t, o, s in mapping if s == "NO_PWM"),
        "enriched_backfill": ",".join(backfill),
    }


def register_se_motif_tools(mcp):
    mcp.tool()(scan_super_enhancer_motifs)
