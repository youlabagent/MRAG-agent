"""GIGGLE TF-overlap node (Section 9) — LOCAL, no Cistrome web.

2026-08-05: runs giggle search of the condition-open peak sets
({label}_specific.bed == analyst {COND}_open.bed) against the local Cistrome
hg38 TF ChIP-seq index, then follows  3.plot.R: keep the TOP-N datasets by
GIGGLE score, take per-Factor mean, plot the top-10 (treatment-open = orange,
control-open = blue). The TOP-N cut is essential — without it a Factor's mean is
diluted by its many irrelevant-cell-type datasets, so real drivers (AR/FOXA1/
HOXB13/JARID2) sink below single-dataset noise factors. NOT byte-identical to the
Cistrome DB web toolkit (our index is a slightly older/smaller collection), but
fully local + biologically equivalent.

2026-09-24: genome-aware. `params["genome"]` (the run's OrganismProfile.genome,
"hg38" | "mm10"; default "hg38") is passed to run_giggle.sh as its 7th
argument, which selects the index + GSM->Factor map for that assembly:
hg38 -> giggle_index_hg38_tf (10,005 datasets), mm10 -> giggle_index_mm10_tf
(7,624 datasets). A missing genome param keeps the historical hg38 call.

MCP step 2a (2026-10-12): the stage is the atac-mcp tool giggle_tf_overlap
(mcps/ATAC_mcp/src/atac_mcp/tools/giggle.py -- the former body of run(), unchanged); run() calls it through
capture_mcp_tools (MRAG_TOOL_TRANSPORT=mcp: an MCP tools/call; =direct: the function).
"""
from __future__ import annotations
import subprocess  # noqa: F401 (kept: tests replace subprocess.run, which the tool calls)
import sys as _sys
from pathlib import Path
from ._base import Node
from . import _stage_tools
_sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/ATAC_mcp/src")
from atac_mcp.tools.giggle import WRAPPER   # noqa: F401 (the tool's wrapper path)


class GiggleNode(Node):
    """Inputs : treat_bed, control_bed (condition-specific/open peak sets)
    Params : treat_label, control_label, top_n (default 200), timeout_s,
             genome ("hg38" | "mm10", default "hg38")
    Outputs: giggle_up_png (treatment-open), giggle_down_png (control-open)."""
    name = "giggle"

    def run(self) -> None:
        # The assembly searched is recorded in the manifest through params
        # ("genome"); outputs stay file paths only (to_manifest stats them).
        tools = _stage_tools.tools("giggle")
        self.outputs = tools["giggle_tf_overlap"](
            treat_bed=str(self.inputs["treat_bed"]), control_bed=str(self.inputs["control_bed"]),
            output_dir=str(self.output_dir),
            treat_label=self.params.get("treat_label", "treatment"),
            control_label=self.params.get("control_label", "control"),
            top_n=self.params.get("top_n", 200), genome=str(self.params.get("genome") or "hg38"),
            timeout_s=self.params.get("timeout_s", 1200))

    def validate(self):
        gates = []
        for k in ("giggle_up_png", "giggle_down_png"):
            pth = Path(self.outputs.get(k, ""))
            ok = pth.exists() and pth.stat().st_size > 5000
            gates.append({"name": k, "result": "pass" if ok else "fail", "msg": str(pth)})
        return gates
