""" Cytoscape (test.cys-style) builder node — Section 8 of the report.

Takes the ranked consensus + INTEGRATED_ranked.csv, calls the viz_report_mcp
`trn_cyjs_build` tool, and emits `trn_cyjs.json` (Cytoscape.js elements + style)
that the template wires into the cy-trn div.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/viz_report_mcp/src")
from ._base import Node, capture_mcp_tools


class TrnCytoscapeBuilderNode(Node):
    """Inputs:
        consensus_ranked_tsv:   per-peak ranked consensus from TRNRankedNode
        integrated_ranked_csv:  INTEGRATED_ranked.csv (with Combined Score column)
    Params:
        top_targets (default 10), min_tfs_per_target (default 1)
    Outputs:
        trn_cyjs_json:  Cytoscape.js elements + style JSON
        anno_path:      -format anno.txt (4 cols)
        inter_path:     -format input_inter.txt (Target,TF,Interaction)
        n_targets, n_tfs, n_edges
    """
    name = "trn_cytoscape"

    def run(self) -> None:
        from viz_report_mcp.tools.cytoscape_trn import register_trn_cyjs_tools
        tools = capture_mcp_tools(register_trn_cyjs_tools)
        out_dir = Path(self.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

        res = json.loads(tools["trn_cyjs_build"](
            consensus_ranked_tsv=self.inputs.get("consensus_ranked_tsv", ""),
            integrated_ranked_csv=self.inputs["integrated_ranked_csv"],
            output_dir=str(out_dir),
            top_targets=self.params.get("top_targets", 15),
            min_tfs_per_target=self.params.get("min_tfs_per_target", 2),
            pval2_txt=self.inputs.get("pval2_txt", ""),
            max_tfs=self.params.get("max_tfs", 6),
        ))
        if res.get("status") != "ok":
            raise RuntimeError(f"trn_cyjs_build failed: {res}")
        self.outputs = {
            "trn_cyjs_json": res["cyjs_json"],
            "anno_path":     res["anno_path"],
            "inter_path":    res["inter_path"],
            "n_targets":     res["n_targets"],
            "n_tfs":         res["n_tfs"],
            "n_edges":       res["n_edges"],
        }
