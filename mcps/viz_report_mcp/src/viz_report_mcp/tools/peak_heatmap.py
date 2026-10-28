"""viz-report-mcp: the Section-6 peak heatmap (deepTools) over the 3-way peak-set regions.

Two figure stages of the orchestrator as MCP tools, called through the orchestrator's MCP transport:
  plot_peak_heatmap          the body of PeakHeatmapNode (nodes/peakheatmap.py): Reference/peakheatmap/
                             run_peakheatmap.sh = 1.input.py (within-arm replicate narrowPeak intersect ->
                             A-specific / Shared / B-specific regions) + 2.peakheatmap_group.py (CPM bamCoverage per
                             replicate -> bigwigAverage group mean -> computeMatrix -> plotHeatmap, white,purple x2;
                             fonts enlarged via an injected matplotlibrc).  An existing peakheatmap.png (> 10 kB) in
                             output_dir is reused.
  plot_peak_heatmap_matrix   the processed-matrix ATAC path of runner.py (Step 6e-2b, no per-replicate BAMs):
                             per-arm bigWigs -> Reference/peakheatmap/run_peakheatmap_matrix.sh (DiffBind-derived
                             treat-gained / Shared / ref-gained regions, the same computeMatrix -> plotHeatmap step);
                             without bigWigs the log2-CPM peak x sample pheatmap proxy (the orchestrator's
                             adapters/atac_matrix_peakheatmap.R, passed as r_script)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Optional, Union

HELPER = "/BIO1/MRAG/Agent/Reference/peakheatmap/run_peakheatmap.sh"
MATRIX_HELPER = "/BIO1/MRAG/Agent/Reference/peakheatmap/run_peakheatmap_matrix.sh"


def plot_peak_heatmap(narrowpeak_a: list[str], bam_a: list[str], narrowpeak_b: list[str], bam_b: list[str],
                      output_dir: str, label_a: str = "arm_a", label_b: str = "arm_b",
                      threads: Union[int, str] = 32, timeout_s: Union[int, float] = 3600) -> dict[str, str]:
    """deepTools peak heatmap of two arms x two replicates over the 3-way peak-set regions (the PeakHeatmapNode
    stage).

    Args:
        narrowpeak_a, bam_a: arm A (treatment) per-replicate narrowPeak files and dedup BAMs (2 each).
        narrowpeak_b, bam_b: arm B (control), the same.
        output_dir: the run's peakheatmap/ directory (peakheatmap.png and the region BEDs).
        label_a, label_b: arm labels (column titles, region-file names).
        threads: deepTools threads.  timeout_s: for run_peakheatmap.sh.

    Returns {peakheatmap_png}; raises RuntimeError when the helper fails or writes no PNG."""
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "peakheatmap.png"
    if png.is_file() and png.stat().st_size > 10000:
        return {"peakheatmap_png": str(png)}
    cmd = ["bash", HELPER,
           str(out_dir),
           label_a, ",".join(narrowpeak_a), ",".join(bam_a),
           label_b, ",".join(narrowpeak_b), ",".join(bam_b),
           str(threads)]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=timeout_s)
    if r.returncode != 0 or not png.is_file():
        raise RuntimeError(f"peakheatmap failed: {r.stderr[-600:]}")
    return {"peakheatmap_png": str(png)}


def plot_peak_heatmap_matrix(output_png: str, diffbind_csv: str, treat_label: str, ref_label: str,
                             bigwigs_a: Optional[list[str]] = None, bigwigs_b: Optional[list[str]] = None,
                             bigwig_a_label: Optional[str] = None, bigwig_b_label: Optional[str] = None,
                             atac_count_matrix: Optional[str] = None, atac_group_tsv: Optional[str] = None,
                             r_script: str = "") -> dict[str, Any]:
    """Section-6 heatmap for processed-matrix ATAC input (runner Step 6e-2b, the stage formerly inline in the
    runner).

    With bigWigs for both arms (the existing files of bigwigs_a / bigwigs_b): run_peakheatmap_matrix.sh
    (DiffBind FDR 0.05, 16 threads, timeout 3600 s) -> step "peakheatmap_matrix_deeptools".  Otherwise the R
    log2-CPM pheatmap proxy r_script --counts atac_count_matrix --groups atac_group_tsv --diffbind diffbind_csv
    --output output_png --treat treat_label --ref ref_label (timeout 600 s) -> step "peakheatmap_matrix".

    Returns {step, returncode, stdout, stderr, peakheatmap_png ("" when no PNG > 10 kB was written), ok}; the
    caller records the step.  A timeout raises subprocess.TimeoutExpired."""
    _ph_out = Path(output_png)
    _ph_out.parent.mkdir(parents=True, exist_ok=True)
    _bws_a = [p for p in (bigwigs_a or []) if Path(p).exists()]
    _bws_b = [p for p in (bigwigs_b or []) if Path(p).exists()]
    _have_bw = bool(_bws_a and _bws_b)
    if _have_bw:
        _sh = MATRIX_HELPER
        _cmd = ["bash", _sh, str(_ph_out.parent),
                bigwig_a_label, ",".join(_bws_a),
                bigwig_b_label, ",".join(_bws_b),
                diffbind_csv, "0.05", "16"]
        _step = "peakheatmap_matrix_deeptools"; _to = 3600
    else:
        _rscript = r_script
        _cmd = ["Rscript", _rscript, "--counts", atac_count_matrix,
                "--groups", atac_group_tsv, "--diffbind", diffbind_csv,
                "--output", str(_ph_out), "--treat", treat_label, "--ref", ref_label]
        _step = "peakheatmap_matrix"; _to = 600
    _r = subprocess.run(_cmd, capture_output=True, text=True, timeout=_to)
    ok = bool(_ph_out.exists() and _ph_out.stat().st_size > 10000)
    return {"step": _step, "returncode": _r.returncode, "stdout": _r.stdout, "stderr": _r.stderr,
            "peakheatmap_png": str(_ph_out) if ok else "", "ok": ok}


def register_peak_heatmap_tools(mcp):
    mcp.tool()(plot_peak_heatmap)
    mcp.tool()(plot_peak_heatmap_matrix)
