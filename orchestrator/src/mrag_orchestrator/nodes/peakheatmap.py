"""PeakHeatmapNode — deeptools peak heatmap over 3-way peak-set regions.

Mirrors Plot/peakheatmap/1.input.py (within-arm rep narrowPeak intersect ->
A-specific / Shared / B-specific 3-way region split) + 2.peakheatmap_group.py
(CPM bamCoverage per rep -> bigwigAverage group mean -> computeMatrix ->
plotHeatmap, white,purple x2). 2026-06-29  update: completely replaces the
prior DiffBind-differential UP/DOWN region definition. FONT FIX retained
(deeptools plotHeatmap has no --fontSize; fonts enlarged via injected
matplotlibrc). Report Section 6.

MCP step 2a (2026-10-12): both stages are MCP tools, called through
capture_mcp_tools (MRAG_TOOL_TRANSPORT=mcp: an MCP tools/call; =direct: the function):
  build_specific_beds   -> atac-mcp split_condition_peaks   (mcps/ATAC_mcp/src/atac_mcp/tools/peak_sets.py)
  PeakHeatmapNode.run   -> viz-report-mcp plot_peak_heatmap (mcps/viz_report_mcp/src/viz_report_mcp/tools/
                           peak_heatmap.py)
-- the former code of each, unchanged.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path
from ._base import Node
from . import _stage_tools
_sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/viz_report_mcp/src")
from viz_report_mcp.tools.peak_heatmap import HELPER   # noqa: F401 (the tool's helper script)


def build_specific_beds(narrowpeak_a, narrowpeak_b, label_a, label_b, out_dir,
                        timeout_s: int = 900) -> dict:
    """Within-arm replicate intersect -> A-specific / Shared / B-specific BEDs.

    Byte-for-byte the same bedtools chain as run_peakheatmap.sh Step 1 (
    1.input.py), factored out because Section 8's motif enrichment needs the
    condition-specific peak sets and runs BEFORE the heatmap step. Writes into
    the peak-heatmap output dir, so the later heatmap step simply regenerates
    the same files.

    Returns {label_a: path, label_b: path, "shared": path}; {} when the inputs
    are not two-replicates-per-arm or bedtools fails.  (MCP step 2a: the atac-mcp
    tool split_condition_peaks.)
    """
    tools = _stage_tools.tools("peak_sets")
    return tools["split_condition_peaks"](
        narrowpeak_a=[str(x) for x in (narrowpeak_a or [])], narrowpeak_b=[str(x) for x in (narrowpeak_b or [])],
        label_a=label_a, label_b=label_b, output_dir=str(out_dir), timeout_s=timeout_s)


class PeakHeatmapNode(Node):
    """Inputs: narrowpeak_a (list[2]), bam_a (list[2]), label_a,
    narrowpeak_b (list[2]), bam_b (list[2]), label_b.
    Output: peakheatmap_png."""
    name = "peakheatmap"

    def run(self) -> None:
        i = self.inputs
        tools = _stage_tools.tools("peak_heatmap")
        self.outputs = tools["plot_peak_heatmap"](
            narrowpeak_a=list(i["narrowpeak_a"]), bam_a=list(i["bam_a"]),
            narrowpeak_b=list(i["narrowpeak_b"]), bam_b=list(i["bam_b"]),
            output_dir=str(self.output_dir), label_a=i.get("label_a", "arm_a"), label_b=i.get("label_b", "arm_b"),
            threads=self.params.get("threads", 32), timeout_s=self.params.get("timeout_s", 3600))

    def validate(self):
        p = Path(self.outputs.get("peakheatmap_png", ""))
        ok = p.is_file() and p.stat().st_size > 10000
        return [{"name": "peakheatmap_png", "result": "pass" if ok else "fail",
                 "msg": str(p)}]
