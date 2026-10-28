"""DAStk differential motif analysis wrap.

Step 5 of the ATAC pipeline; produces the MD-score file consumed by
combined_mcp.combine_pvalues. Separated from ATACNode because it requires
treatment vs control group assignment.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/ATAC_mcp/src")
from ._base import Node, capture_mcp_tools


class DAStkNode(Node):
    """Inputs:  project_dir, treatment_files (list[str]), control_files (list[str])
    Params:    ref_dir, genome (default 'hg38'), motif_dir (genome-wide motif-hit
               BEDs; '' => the MCP's hg38/other convention), num_threads (0=auto),
               dastk_path
    Outputs:   atac_md_csv (the differential_md_scores.txt)
    """
    name = "dastk"

    def run(self) -> None:
        from atac_mcp.tools import pipeline as _p
        from atac_mcp.tools.executor import LocalExecutor
        project_dir = self.inputs["project_dir"]
        executor = LocalExecutor(work_dir=self.params.get("work_dir", project_dir))
        tools = capture_mcp_tools(lambda mcp: _p.register_pipeline_tools(mcp, executor))
        out = tools["step5_dastk"](
            project_dir=project_dir,
            treatment_files=self.inputs["treatment_files"],
            control_files=self.inputs["control_files"],
            ref_dir=self.params.get("ref_dir", ""),
            genome=self.params.get("genome", "hg38"),
            motif_dir=self.params.get("motif_dir", ""),
            num_threads=self.params.get("num_threads", 0),
            dastk_path=self.params.get("dastk_path", ""),
        )
        # Expected output (DAStk -1 control / -2 treatment convention, 2026-07):
        #   <project_dir>/DASTK/control.narrowPeak_vs_treatment.narrowPeak_differential_md_scores.txt
        md = Path(project_dir) / "DASTK" / "control.narrowPeak_vs_treatment.narrowPeak_differential_md_scores.txt"
        if not md.exists():
            raise RuntimeError(f"DAStk did not produce expected MD-score file at {md}")
        self.outputs = {"atac_md_csv": str(md)}

    def validate(self):
        p = self.outputs.get("atac_md_csv","")
        if not p or not Path(p).exists():
            return [{"name":"md_score_exists","result":"fail","msg":"no MD-score output"}]
        n = sum(1 for _ in open(p))
        return [{"name":"md_score_rows","result":"pass" if n > 50 else "warn",
                 "value":n,"msg":f"{n} motifs in MD-score table"}]
