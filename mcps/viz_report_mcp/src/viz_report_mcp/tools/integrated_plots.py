""" R-script plot wrappers (rank / volcano / heatmap).

Each tool runs ``Rscript`` against one of the bundled scripts in
``viz_report_mcp/r_scripts/``. The R scripts mirror the original Plot/*.R scripts precisely
but take CLI arguments for input/output paths.

Defaults match those scripts (top_n=10 up + 10 down genes by Rank_Combined_Score).
"""
from __future__ import annotations
import json, shutil, subprocess
from pathlib import Path
from typing import Optional


# Bundled scripts location (sibling of the tools/ directory)
_R_DIR = Path(__file__).resolve().parents[3] / "r_scripts"


def _run_rscript(script_path: Path, args: list[str], timeout_s: int = 600) -> dict:
    rscript = shutil.which("Rscript") or "/usr/bin/Rscript"
    cmd = [rscript, str(script_path), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    return {
        "rc":     proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
        "cmd":    " ".join(cmd),
    }


def register_integrated_plot_tools(mcp):

    @mcp.tool()
    def plot_rank(
        integrated_csv: str,
        output_png: str,
        top_n: int = 10,
        label_size: float = 3.0,
        width: float = 8.0,
        height: float = 8.0,
        dpi: int = 180,
        timeout_s: int = 300,
    ) -> str:
        """ rank.R wrapper — Rank vs Combined Score (Avg.Z) scatter.

        Top-N up genes labelled red, bottom-N down labelled blue.
        Requires INTEGRATED_ranked.csv with the 2026-06-04 schema
        (Gene, Combined Score (Avg.Z), Rank_Combined_Score, Pcom).
        """
        args = [
            "--integrated", integrated_csv,
            "--output",     output_png,
            "--top_n",      str(top_n),
            "--label_size", str(label_size),
            "--width",      str(width),
            "--height",     str(height),
            "--dpi",        str(dpi),
        ]
        r = _run_rscript(_R_DIR / "rank_plot.R", args, timeout_s=timeout_s)
        return json.dumps({
            "status": "ok" if r["rc"] == 0 and Path(output_png).exists() else "error",
            "output": output_png,
            **r,
        }, ensure_ascii=False, indent=2)

    @mcp.tool()
    def plot_volcano(
        integrated_csv: str,
        output_png: str,
        top_n: int = 10,
        width: float = 9.0,
        height: float = 8.0,
        dpi: int = 180,
        timeout_s: int = 300,
    ) -> str:
        """ volcano.R wrapper — log2FC (RNA-seq) vs Combined Score (Avg.Z).

        Top-N up genes by Rank_Combined_Score labelled red, bottom-N blue.
        """
        args = [
            "--integrated", integrated_csv,
            "--output",     output_png,
            "--top_n",      str(top_n),
            "--width",      str(width),
            "--height",     str(height),
            "--dpi",        str(dpi),
        ]
        r = _run_rscript(_R_DIR / "volcano_plot.R", args, timeout_s=timeout_s)
        return json.dumps({
            "status": "ok" if r["rc"] == 0 and Path(output_png).exists() else "error",
            "output": output_png,
            **r,
        }, ensure_ascii=False, indent=2)

    @mcp.tool()
    def plot_heatmap(
        integrated_csv: str,
        deseq_csv: str,
        counts_txt: str,
        output_png: str,
        top_n: int = 10,
        group_a_label: str = "OC2",
        group_b_label: str = "Cyc",
        samples_a: Optional[list[str]] = None,
        samples_b: Optional[list[str]] = None,
        color_a: str = "gold",
        color_b: str = "purple",
        width: float = 8.0,
        height: float = 10.0,
        dpi: int = 180,
        timeout_s: int = 300,
    ) -> str:
        """Heatmap.R wrapper — top-N up + top-N down genes' RSEM counts.

        scale="row" pheatmap with blue-white-red palette and Group annotation.
        ``samples_a`` / ``samples_b`` are optional comma-separated sample
        names; if omitted, columns are split in half.
        """
        args = [
            "--integrated", integrated_csv,
            "--deseq",      deseq_csv,
            "--counts",     counts_txt,
            "--output",     output_png,
            "--top_n",      str(top_n),
            "--group_a",    group_a_label,
            "--group_b",    group_b_label,
            "--color_a",    color_a,
            "--color_b",    color_b,
            "--width",      str(width),
            "--height",     str(height),
            "--dpi",        str(dpi),
        ]
        if samples_a:
            args += ["--samples_a", ",".join(samples_a)]
        if samples_b:
            args += ["--samples_b", ",".join(samples_b)]
        r = _run_rscript(_R_DIR / "heatmap_plot.R", args, timeout_s=timeout_s)
        return json.dumps({
            "status": "ok" if r["rc"] == 0 and Path(output_png).exists() else "error",
            "output": output_png,
            **r,
        }, ensure_ascii=False, indent=2)
