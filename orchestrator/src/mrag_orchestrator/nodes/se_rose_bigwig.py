"""SERoseBigwigNode — super-enhancer calling for inputs WITHOUT read alignments.

Same ROSE procedure as SERoseNode, except the per-region signal: ROSE counts
reads from a BAM (density x length of each stitched region); here the signal is
the area under the treatment-arm ATAC bigWig tracks inside each stitched region
(sum of per-base values, averaged over the replicate bigWigs). Everything else is
ROSE's own code:
  1. peaks BED -> ROSE GFF
  2. ROSE_stitchTable.py (ROSE_main.regionStitching, 12.5 kb, TSS exclusion off,
     locus table identical to ROSE mapCollection: chrY removed, ranked by length)
  3. region signal from bigWigs (pyBigWig, 0-based half-open = ROSE [START-1, STOP))
  4. ROSE_callSuper.R (inflection-point cutoff, AllEnhancers / SuperEnhancers tables,
     Plot_points.png) with no control
  5. ROSE_geneMapper.py -> ENHANCER_TO_GENE
Used for processed-matrix ATAC (e.g. TCGA PRAD: count matrix + per-sample
insertion bigWigs) and for any run with bigWigs but no BAMs.
Outputs follow SERoseNode so SEMotifNode / SETrnBuilderNode work unchanged.

MCP step 2a (2026-10-12): the stage is the atac-mcp tool call_super_enhancers_bigwig
(mcps/ATAC_mcp/src/atac_mcp/tools/super_enhancers.py -- the former body of run(), unchanged); run() resolves the
params and calls it through capture_mcp_tools (MRAG_TOOL_TRANSPORT=mcp: an MCP tools/call; =direct: the function).
"""
from __future__ import annotations
import sys as _sys
from ._base import Node
from . import _stage_tools
_sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/ATAC_mcp/src")
from atac_mcp.tools.super_enhancers import ROSE_DIR, ROSE_PY2, _data_rows   # noqa: F401 (the tool's constants)


class SERoseBigwigNode(Node):
    """Inputs: bigwig_list (treatment-arm bigWigs), peaks_bed (treatment peak set), label
    Params: genome (hg38), stitch (12500), tss (0), top_n (10), timeout_s (7200),
            peak_set_desc (text for the report legend)
    Outputs: as SERoseNode (se_table, all_table, enhancer_to_gene, se_top_tsv, se_homerpeak,
             rose_plot_png, n_se, n_all, n_peaks, top_n) + signal_source, n_bigwigs"""
    name = "se_rose"

    def run(self) -> None:
        p = self.params
        tools = _stage_tools.tools("super_enhancers")
        self.outputs = tools["call_super_enhancers_bigwig"](
            bigwig_list=[str(b) for b in self.inputs["bigwig_list"]], peaks_bed=str(self.inputs["peaks_bed"]),
            label=str(self.inputs["label"]), output_dir=str(self.output_dir),
            genome=p.get("genome", "hg38"), stitch=int(p.get("stitch", 12500)), tss=int(p.get("tss", 0)),
            top_n=int(p.get("top_n", 10)), timeout_s=int(p.get("timeout_s", 7200)),
            rose_dir=p.get("rose_dir", ROSE_DIR), python2=p.get("python2", ROSE_PY2))

    def validate(self):
        n = int(self.outputs.get("n_se", 0) or 0)
        g = [{"name": "n_super_enhancers", "result": "pass" if n > 0 else "fail", "value": n,
              "msg": "" if n > 0 else "0 super-enhancers from bigWig signal"}]
        top_n = int(self.params.get("top_n", 10))
        if 0 < n < top_n:
            g.append({"name": "top_n_available", "result": "warn", "value": n, "msg": f"only {n} SEs (< top_n={top_n})"})
        return g
