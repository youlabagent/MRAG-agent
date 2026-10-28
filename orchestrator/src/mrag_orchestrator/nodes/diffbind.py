"""diffbind_mcp wrapper: ATAC peak differential binding via R DiffBind.

Two-step: prepare_diffbind_sheet -> run_diffbind. We expose this as a single
node that takes either:
  - inputs.samples (list[dict] per sample with SampleID/Condition/Replicate/
                    bamReads/Peaks)  -> we'll prepare the sheet ourselves
  - inputs.sheet_csv (path to existing DiffBind sample sheet) -> skip prepare
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/diffbind_mcp/src")
from ._base import Node, capture_mcp_tools


class DiffbindNode(Node):
    """Inputs:  samples (list[dict]) OR sheet_csv (path)
    Params:   category (default 'Condition'), reference (control/baseline
              level; positive Fold = up in treatment, like DESeq2),
              min_members (2), threshold (1.0), threads (4), timeout_s (3600),
              dry_run (False)
    Outputs:  diffbind_csv, sheet_csv, status_json
    """
    name = "diffbind"

    def run(self) -> None:
        from diffbind_mcp.tools import pipeline as _p
        tools = capture_mcp_tools(_p.register_pipeline_tools)
        out_dir = Path(self.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: prepare or reuse sample sheet
        if self.inputs.get("sheet_csv"):
            sheet_path = self.inputs["sheet_csv"]
        else:
            samples = self.inputs.get("samples")
            if not samples:
                raise ValueError("DiffbindNode needs inputs.samples or inputs.sheet_csv")
            sheet_path = str(out_dir / "diffbind_sheet.csv")
            prep_res = json.loads(tools["prepare_diffbind_sheet"](
                samples=samples, output_path=sheet_path,
                default_factor=self.params.get("default_factor", "ATAC"),
                default_tissue=self.params.get("default_tissue", "human"),
                default_peak_caller=self.params.get("default_peak_caller", "bed"),
                check_files=self.params.get("check_files", True),
            ))
            if prep_res.get("status") != "ok":
                raise RuntimeError(f"prepare_diffbind_sheet failed: {prep_res}")

        # Step 2: run DiffBind
        out_csv = str(out_dir / "DiffBind_out.csv")
        run_res = json.loads(tools["run_diffbind"](
            sheet_path=sheet_path, output_csv=out_csv,
            category=self.params.get("category", "Condition"),
            reference=self.params.get("reference"),
            min_members=self.params.get("min_members", 2),
            threshold=self.params.get("threshold", 1.0),
            threads=self.params.get("threads", 4),
            timeout_s=self.params.get("timeout_s", 3600),
            dry_run=self.params.get("dry_run", False),
        ))
        if run_res.get("status") not in ("ok", "dry_run"):
            raise RuntimeError(f"run_diffbind failed: {run_res.get('message')}, stderr={run_res.get('stderr','')[:400]}")

        self.outputs = {
            "diffbind_csv": out_csv,
            "sheet_csv":    sheet_path,
            "status_json":  out_csv.replace(".csv", ".status.json"),
        }

    def validate(self):
        out = self.outputs.get("diffbind_csv","")
        if not out or not Path(out).exists():
            return [{"name":"diffbind_exists","result":"fail","msg":"no output"}]
        # Quick check: row count
        try:
            n = sum(1 for _ in open(out)) - 1
        except Exception:
            n = 0
        return [{"name":"peak_count","result":"pass" if n > 10000 else "warn",
                 "value": n, "msg":f"{n} peaks"}]
