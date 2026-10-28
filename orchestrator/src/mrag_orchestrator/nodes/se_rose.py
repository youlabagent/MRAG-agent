"""SERoseNode — super-enhancer calling (ROSE) on the treatment-specific peak set.

Mirrors the reference 1.rose.py steps 1-3:
  1. samtools merge/sort/index the treatment-arm replicate BAMs
  2. treatment-specific peak BED (peakheatmap/<treat>_specific.bed) -> ROSE GFF
  3. ROSE_main.py (stitch 12.5 kb, no TSS exclusion) + ROSE_geneMapper.py
and extracts the top-N super-enhancers (rank by ROSE signal) as the SE layer of
the SE -> TF -> Target TRN (report Section 10b).

Cache: when <output_dir>/rose/<prefix>_SuperEnhancers_ENHANCER_TO_GENE.txt already
exists and is non-empty the ROSE run is skipped (same policy as homer_motif).
Tool: /BIO1/MRAG/Agent/bin/tools/ROSE (patched copy: scripts resolved relative
to the ROSE dir; BAM index required — a missing .bai silently yields 0 SEs).

MCP step 2a (2026-10-12): the stage is the atac-mcp tool call_super_enhancers
(mcps/ATAC_mcp/src/atac_mcp/tools/super_enhancers.py -- the former body of run(), unchanged); run() resolves the
params and calls it through capture_mcp_tools (MRAG_TOOL_TRANSPORT=mcp: an MCP tools/call; =direct: the function).
"""
from __future__ import annotations
import sys as _sys
from ._base import Node
from . import _stage_tools
_sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/ATAC_mcp/src")
from atac_mcp.tools.super_enhancers import ROSE_DIR, ROSE_PY2, SAMTOOLS, _run   # noqa: F401 (the tool's constants)


class SERoseNode(Node):
    """Inputs:
        bam_list:   treatment-arm dedup BAMs (>=1)
        peaks_bed:  treatment-specific peak BED (3+ cols)
        label:      treatment condition label (used as file prefix)
    Params:
        genome (hg38), stitch (12500), tss (0), threads (16), top_n (10),
        timeout_s (10800), rose_dir, python2
    Outputs:
        merged_bam, rose_gff, se_table (SuperEnhancers.table.txt), all_table,
        enhancer_to_gene, se_top_tsv, se_homerpeak, rose_plot_png, n_se, n_all
    """
    name = "se_rose"

    def run(self) -> None:
        p = self.params
        tools = _stage_tools.tools("super_enhancers")
        self.outputs = tools["call_super_enhancers"](
            bam_list=[str(b) for b in self.inputs["bam_list"]], peaks_bed=str(self.inputs["peaks_bed"]),
            label=str(self.inputs["label"]), output_dir=str(self.output_dir),
            genome=p.get("genome", "hg38"), threads=int(p.get("threads", 16)),
            stitch=int(p.get("stitch", 12500)), tss=int(p.get("tss", 0)),
            top_n=int(p.get("top_n", 10)), timeout_s=int(p.get("timeout_s", 10800)),
            rose_dir=p.get("rose_dir", ROSE_DIR), python2=p.get("python2", ROSE_PY2))

    def validate(self):
        gates = []
        n = int(self.outputs.get("n_se", 0) or 0)
        gates.append({"name": "n_super_enhancers", "result": "pass" if n > 0 else "fail", "value": n,
                      "msg": "" if n > 0 else "ROSE returned 0 super-enhancers: check BAM index (.bai) and peak BED"})
        top_n = int(self.params.get("top_n", 10))
        if 0 < n < top_n:
            gates.append({"name": "top_n_available", "result": "warn", "value": n,
                          "msg": f"only {n} SEs (< top_n={top_n})"})
        return gates
