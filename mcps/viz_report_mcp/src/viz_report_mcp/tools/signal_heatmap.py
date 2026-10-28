"""deepTools signal heatmap + profile (Galaxy-style).

Mirrors plot_Galaxy.R:
  computeMatrix reference-point --referencePoint center -b N -a N
  plotHeatmap   --colorList ... --heatmapHeight ...
  plotProfile   --colors ... --perGroup

Used for cross-condition region-level signal visualization (rows = peaks,
cols = ±Nbp from peak center).
"""
from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from typing import List, Optional


def register_signal_heatmap_tools(mcp):

    @mcp.tool()
    def plot_signal_heatmap(
        bigwig_paths: List[str],
        sample_labels: List[str],
        regions_bed: str,
        out_dir: str,
        prefix: str = "signal",
        flank_bp: int = 5000,
        threads: int = 16,
        heatmap_color_lists: Optional[List[str]] = None,
        profile_colors: Optional[List[str]] = None,
        heatmap_height: int = 12,
        skip_zeros: bool = True,
    ) -> dict:
        """Generate deepTools signal heatmap + profile for target regions.

        Args:
            bigwig_paths: bigwig files (one per sample)
            sample_labels: display labels matching bigwig_paths
            regions_bed: target regions (e.g. top promoter peaks of selected TFs)
            out_dir: output directory
            prefix: file prefix for matrix/heatmap/profile
            flank_bp: bp upstream + downstream of region center
            heatmap_color_lists: e.g. ["white,steelblue", "white,steelblue",
                                       "white,firebrick", "white,firebrick"]
                                  (one per sample, defaults provided)
            profile_colors: line colors for profile plot
        Returns: paths to matrix.gz, heatmap_png, profile_png
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        matrix_gz   = out / f"{prefix}_matrix.gz"
        sorted_bed  = out / f"{prefix}_sorted_regions.bed"
        heatmap_png = out / f"{prefix}_heatmap.png"
        profile_png = out / f"{prefix}_profile.png"

        # 1) computeMatrix
        cmd = [
            "computeMatrix", "reference-point",
            "--referencePoint", "center",
            "-S", *bigwig_paths,
            "-R", regions_bed,
            "-b", str(flank_bp), "-a", str(flank_bp),
            "--numberOfProcessors", str(threads),
            "-o", str(matrix_gz),
            "--outFileSortedRegions", str(sorted_bed),
        ]
        if skip_zeros:
            cmd.append("--skipZeros")
        r1 = subprocess.run(cmd, capture_output=True, text=True)
        if r1.returncode != 0:
            return {"ok": False, "step": "computeMatrix",
                    "stderr": r1.stderr[-1200:]}

        # 2) plotHeatmap
        if heatmap_color_lists is None:
            # default: OC2 blue, cMyc red, paired by condition
            default = ["white,steelblue", "white,steelblue",
                       "white,firebrick", "white,firebrick"]
            heatmap_color_lists = default[: len(bigwig_paths)]
        cmd2 = [
            "plotHeatmap",
            "-m", str(matrix_gz),
            "-o", str(heatmap_png),
            "--colorList", *heatmap_color_lists,
            "--samplesLabel", *sample_labels,
            "--heatmapHeight", str(heatmap_height),
            "--whatToShow", "heatmap and colorbar",
            "--missingDataColor", "white",
            "--xAxisLabel", "",
            "--plotFileFormat", "png",
            "--dpi", "150",
        ]
        r2 = subprocess.run(cmd2, capture_output=True, text=True)
        if r2.returncode != 0:
            return {"ok": False, "step": "plotHeatmap",
                    "stderr": r2.stderr[-1200:]}

        # 3) plotProfile (perGroup)
        if profile_colors is None:
            profile_colors = ["steelblue", "lightsteelblue", "firebrick", "lightcoral"][: len(bigwig_paths)]
        cmd3 = [
            "plotProfile",
            "-m", str(matrix_gz),
            "-out", str(profile_png),
            "--colors", *profile_colors,
            "--samplesLabel", *sample_labels,
            "--perGroup",
            "--plotFileFormat", "png",
            "--dpi", "150",
        ]
        r3 = subprocess.run(cmd3, capture_output=True, text=True)
        if r3.returncode != 0:
            return {"ok": False, "step": "plotProfile",
                    "stderr": r3.stderr[-1200:]}

        try:
            from viz_report_mcp.tools._heatmap_labels import add_gene_labels
            labeled = out / f"{prefix}_heatmap_labeled.png"
            add_gene_labels(str(heatmap_png), str(sorted_bed), str(labeled), font_size=9)
            heatmap_png_final = labeled
        except Exception as e:
            heatmap_png_final = heatmap_png

        return {
            "ok": True,
            "matrix_gz": str(matrix_gz),
            "sorted_regions_bed": str(sorted_bed),
            "heatmap_png": str(heatmap_png_final),
            "profile_png": str(profile_png),
            "n_samples": len(bigwig_paths),
            "flank_bp": flank_bp,
        }
