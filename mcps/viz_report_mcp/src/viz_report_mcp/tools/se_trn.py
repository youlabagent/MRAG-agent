"""viz-report-mcp: Section 10 SE -> TF -> Target TRN inputs (se_anno.txt, se_input_inter.txt, se_cyjs.json).

The second half of the orchestrator's SETrnBuilderNode (nodes/se_trn.py) -- its display-only DAStk refill
(_display_md_fallback) and the produce_se_trn_inputs call -- as the MCP tool build_se_trn; the node (which first re-runs the Step-2 column filter + DiffBind
/ RNA join on the extended TF list through the homer-mcp / trn-mcp tools) calls it through the orchestrator's MCP
transport.

ATAC MD score for graph TFs that have none in INTEGRATED.csv (colour only): an explicit atac_fallback_tsv; else
this run's DAStk file plus a DAStk pass restricted to the display-only refill beds of the missing TFs (never merged
into INTEGRATED.csv / the TF ranking)."""
from __future__ import annotations

import csv as _csv
import json
import os as _os
import subprocess as _sp
import sys as _sys
from pathlib import Path
from typing import Any, Optional

REFILL_DIR = "/BIO1/MRAG/Agent/Reference/ATAC_hg38/grch38_motifs_refill"
DASTK_PY = "/BIO1/MRAG/venvs/paper_qa/bin/python"
DASTK_DIR = "/BIO1/MRAG/Agent/bin/tools/DAStk/DAStk"
# the atac-mcp wrapper that seeds DAStk's per-motif bootstrap (reproducible p-values)
DASTK_SEEDED = str(Path(__file__).resolve().parents[4] / "ATAC_mcp" / "src" / "atac_mcp" / "tools"
                   / "dastk_seeded.py")
COMBINED_SRC = "/BIO1/MRAG/Agent/mcps/combined_mcp/src"


def _display_md_fallback(out: Path, tfs: list, integrated_csv: str, atac_fallback_tsv: str, dastk_md_csv: str,
                         refill_dir: str, genome: str, extra: dict) -> str:
    explicit = atac_fallback_tsv or ""
    if explicit:
        return explicit
    md = dastk_md_csv or ""
    have = set()
    try:
        for r in _csv.DictReader(open(integrated_csv)):
            if str(r.get("differential_MDscore", "")).strip() not in ("", "nan"):
                have.add(str(r.get("Gene", "")).upper())
    except Exception:
        pass
    missing = [t for t in tfs if t.upper() not in have]
    parts = [md] if md and Path(md).is_file() else []
    refill = Path(refill_dir or REFILL_DIR)
    dk = Path(md).parent if md else None
    if missing and refill.is_dir() and dk and (dk / "control" / "control.narrowPeak.filt").is_file() \
            and (dk / "treatment" / "treatment.narrowPeak.filt").is_file():
        _sys.path.insert(0, COMBINED_SRC)
        try:
            from combined_mcp.tools.pipeline import _hocomoco_to_symbol as _h2s
        except Exception:
            _h2s = lambda s: s
        special = {"FOXEBOX": "FOX_EBOX"}
        want = {t.upper() for t in missing}
        beds = [b for b in sorted(refill.glob("*.bed"))
                if (special.get(b.name.split("_")[0].upper()) or _h2s(b.name.split("_")[0]).upper()) in want]
        if beds:
            wd = out / "dastk_display"; bd = wd / "beds"
            py = DASTK_PY; dp = DASTK_DIR
            res = wd / "control.narrowPeak_vs_treatment.narrowPeak_differential_md_scores.txt"
            try:
                import shutil as _sh
                if bd.exists():
                    _sh.rmtree(bd)          # fresh link set on every call (no stale/dangling links)
                bd.mkdir(parents=True)
                for b in beds:
                    _os.symlink(b, bd / b.name)
                for c in ("control", "treatment"):
                    (wd / c).mkdir(exist_ok=True)
                    _sp.run([py, f"{dp}/process_atac.py", "-t", "16", "-e", str(dk / c / f"{c}.narrowPeak.filt"),
                             "-m", str(bd) + "/", "-g", genome, "-o", str(wd / c)],
                            capture_output=True, text=True, timeout=3600, check=True)
                _sp.run([py, DASTK_SEEDED, dp, "-1", str(wd / "control" / "control.narrowPeak_md_scores.txt"),
                         "-2", str(wd / "treatment" / "treatment.narrowPeak_md_scores.txt"),
                         "-m", "control", "-n", "treatment", "-o", str(wd)],
                        capture_output=True, text=True, timeout=1800, check=True)
                if res.is_file():
                    parts.append(str(res))
                    extra["display_md_refill"] = str(res)
                    extra["display_md_tfs"] = ",".join(sorted(want))
            except Exception as ex:
                extra["display_md_error"] = str(ex)[:300]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    merged = out / "atac_display_fallback.txt"
    with open(merged, "w") as fo:
        for p_ in parts:
            fo.write(open(p_).read().rstrip("\n") + "\n")
    return str(merged)


def build_se_trn(graph_anno: str, graph_inter: str, se_top_tsv: str, se_tf_edges_tsv: str, se_genes_tsv: str,
                 integrated_csv: str, output_dir: str, panel_tfs: list[str], enriched_tfs: list[str],
                 pval2_txt: Optional[str] = "", deseq_csv: Optional[str] = "", label: str = "treat",
                 atac_fallback_tsv: str = "",
                 dastk_md_csv: str = "", refill_dir: str = "", genome: str = "hg38",
                 se_method: str = "") -> dict[str, Any]:
    """SE -> TF -> Target network inputs for Section 10 (the SETrnBuilderNode stage after its extended
    Pval2 filter / join).

    Args:
        graph_anno, graph_inter: the Section-10 graph files (trn_cytoscape/anno.txt, input_inter.txt) -- the same
            Target nodes and panel-TF edges.
        se_top_tsv, se_tf_edges_tsv, se_genes_tsv: call_super_enhancers / scan_super_enhancer_motifs outputs.
        integrated_csv: INTEGRATED.csv (ATAC / RNA values of the TFs).
        output_dir: the run's se_trn/ directory.
        panel_tfs: the TF panel (upper case); enriched_tfs: the Section-8 enriched TFs not in the panel.
        pval2_txt: the extended Pval2 table (enriched-TF -> Target edges by the promoter-presence rule); "" = none.
        deseq_csv, label: per-gene DESeq2 table; treatment label.
        atac_fallback_tsv: explicit ATAC MD-score table for TFs absent from INTEGRATED.csv; else dastk_md_csv
            (this run's DAStk differential file, whose DASTK/<cond>/<cond>.narrowPeak.filt are reused) + a DAStk
            pass over the display-only refill beds (refill_dir, default Reference/ATAC_hg38/grch38_motifs_refill;
            genome = the DAStk assembly).
        se_method: the SE-calling description written into se_cyjs.json summary.se_method ("" = none).

    Returns {se_cyjs_json, se_anno_path, se_inter_path, pval2_extended, n_*..., tf_atac_missing, tf_rna_missing
    [, display_md_refill, display_md_tfs | display_md_error]} -- the node's outputs."""
    from viz_report_mcp.tools.cytoscape_trn import produce_se_trn_inputs
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    panel = list(panel_tfs)
    enriched = list(enriched_tfs)
    tfs = panel + enriched
    pval2 = pval2_txt
    extra = {}
    fallback = _display_md_fallback(out, tfs, integrated_csv, atac_fallback_tsv, dastk_md_csv, refill_dir,
                                    genome, extra)
    res = produce_se_trn_inputs(
        graph_anno_path=graph_anno, graph_inter_path=graph_inter,
        se_top_tsv=se_top_tsv, se_tf_edges_tsv=se_tf_edges_tsv, se_genes_tsv=se_genes_tsv,
        pval2_txt=pval2, integrated_csv=integrated_csv, output_dir=str(out),
        panel_tfs=panel, enriched_tfs=enriched, deseq_csv=deseq_csv,
        atac_fallback_tsv=fallback, label=label)
    if se_method:
        _cj = json.loads(Path(res["se_cyjs_json"]).read_text())
        _cj.setdefault("summary", {})["se_method"] = se_method
        Path(res["se_cyjs_json"]).write_text(json.dumps(_cj, indent=1))
    return {"se_cyjs_json": res["se_cyjs_json"], "se_anno_path": res["se_anno_path"],
            "se_inter_path": res["se_inter_path"], "pval2_extended": pval2,
            **{k: v for k, v in res.items() if k.startswith("n_")},
            "tf_atac_missing": ",".join(res.get("tf_atac_missing", [])),
            "tf_rna_missing": ",".join(res.get("tf_rna_missing", [])),
            **extra}


def register_se_trn_tools(mcp):
    mcp.tool()(build_se_trn)
