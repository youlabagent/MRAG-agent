"""combined_mcp wrapper: combine ATAC MD-score + RNA DESeq -> INTEGRATED.csv.

Tool used: combine_and_pick(atac_md_path, rna_deseq_path, output_dir, n_list)
Outputs: INTEGRATED.csv + top{N}.json/tsv for each N in n_list
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/combined_mcp/src")
from ._base import Node, capture_mcp_tools


class CombinedNode(Node):
    """Inputs:  atac_md_csv, rna_deseq_csv
    Params:   n_list (default [4,6,8]), organism ('human'|'mouse'; selects how
              HOCOMOCO TF symbols are matched to RNA gene symbols)
    Outputs:  integrated_csv, top4_json, top6_json, top8_json (if requested)
    """
    name = "combined"

    def run(self) -> None:
        from combined_mcp.tools import pipeline as _p
        tools = capture_mcp_tools(_p.register_pipeline_tools)
        out_dir = Path(self.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        n_list = self.params.get("n_list", [4, 6, 8])
        res_str = tools["combine_and_pick"](
            atac_md_path=self.inputs["atac_md_csv"],
            rna_deseq_path=self.inputs["rna_deseq_csv"],
            output_dir=str(out_dir),
            n_list=n_list,
            organism=self.params.get("organism", "human"),
        )
        res = json.loads(res_str)
        if res.get("status") != "ok":
            raise RuntimeError(f"combine_and_pick failed: {res}")
        self.outputs = {"integrated_csv": str(out_dir / "INTEGRATED.csv")}
        for n in n_list:
            json_path = out_dir / f"top{n}.json"
            if json_path.exists():
                self.outputs[f"top{n}_json"] = str(json_path)

    def validate(self):
        gates = []
        integ = self.outputs.get("integrated_csv","")
        if integ and Path(integ).exists():
            import pandas as pd
            try:
                df = pd.read_csv(integ)
                need = {"Gene","Pcom","log2FC"}
                missing = need - set(df.columns)
                if missing:
                    gates.append({"name":"integ_schema","result":"fail",
                                  "value":list(df.columns),
                                  "msg":f"missing columns: {missing}"})
                else:
                    gates.append({"name":"integ_schema","result":"pass",
                                  "value":list(df.columns)})
                gates.append({"name":"integ_row_count","result":"pass" if len(df) > 100 else "fail",
                              "value":int(len(df)),
                              "msg":f"{len(df)} rows in INTEGRATED.csv"})
            except Exception as e:
                gates.append({"name":"integ_read","result":"fail","value":None,"msg":str(e)})
        return gates
