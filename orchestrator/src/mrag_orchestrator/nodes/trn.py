"""trn_mcp wrapper: HOMER-filter + DiffBind + DESeq2 -> consensus TRN table."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/trn_mcp/src")
from ._base import Node, capture_mcp_tools


class TRNNode(Node):
    """Inputs:  homer_filter_tsv, diffbind_csv, deseq_csv
    Params:   rna_collapse (default 'best_padj'),
              keep_unmatched_diffbind (False), keep_unmatched_rna (False),
              top_n (for naming the output file)
    Outputs:  consensus_tsv
    """
    name = "trn"

    def run(self) -> None:
        from trn_mcp.tools import integrate as _i
        tools = capture_mcp_tools(_i.register_integrate_tools)
        top_n = self.params.get("top_n", 6)
        out_dir = Path(self.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        out_tsv = out_dir / f"consensus_top{top_n}.tsv"
        res_str = tools["build_tfs_pval3"](
            homer_filter_tsv=self.inputs["homer_filter_tsv"],
            diffbind_csv=self.inputs["diffbind_csv"],
            deseq_csv=self.inputs["deseq_csv"],
            output_tsv=str(out_tsv),
            rna_collapse=self.params.get("rna_collapse","best_padj"),
            keep_unmatched_diffbind=self.params.get("keep_unmatched_diffbind", False),
            keep_unmatched_rna=self.params.get("keep_unmatched_rna", False),
        )
        res = json.loads(res_str)
        if res.get("status") != "ok":
            raise RuntimeError(f"build_tfs_pval3 failed: {res}")
        self.outputs = {"consensus_tsv": str(out_tsv)}

    def validate(self):
        out = self.outputs.get("consensus_tsv","")
        if not out or not Path(out).exists():
            return [{"name":"consensus_exists","result":"fail","msg":"no output"}]
        # Quick line count
        n = sum(1 for _ in open(out)) - 1
        return [{"name":"edge_count","result":"pass" if 10 <= n <= 100000 else "warn",
                 "value": n, "msg": f"{n} edges"}]
