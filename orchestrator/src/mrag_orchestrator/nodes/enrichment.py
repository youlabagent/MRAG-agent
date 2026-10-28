"""Enrichment node — Section 8 of report (Functional Enrichment of Target Programs)."""
from __future__ import annotations
import json, sys
from pathlib import Path
from ._base import Node, capture_mcp_tools


class EnrichmentNode(Node):
    """Inputs:
        ranked_consensus_tsv: TF×Gene ranked TSV from TRNRankedNode
    Params:
        min_shared_tfs: 2     (genes bound by ≥N TFs for shared-target enrichment)
        max_tfs: 6            (top-N TFs for per-TF heatmap)
        libraries: optional list of Enrichr libraries (default Hallmark+GO_BP+Reactome)
        top_terms_per_lib: 10 (shared dot plot)
        top_terms_total: 15   (per-TF heatmap rows)
    Outputs:
        shared_dotplot_json:  Plotly JSON for shared-target dot plot
        per_tf_heatmap_json:  Plotly JSON for per-TF heatmap
        shared_enrich_csv:    Full enrichment table for shared targets
        per_tf_enrich_csv:    Full per-TF enrichment table
        n_shared_genes, n_terms_shared, n_terms_pertf: int counts
    """
    name = "enrichment"

    def run(self) -> None:
        # One-shot resilience: when MRAG_REUSE_EXTERNAL=1, reuse the most recent
        # cached outputs for this (flaky external-API) step instead of re-hitting
        # the network. Lets the pipeline finish when Enrichr/AlphaGenome are down
        # even though an upstream hash change marked this step dirty. Default
        # behaviour is unchanged (guard is a no-op without the env var).
        import os as _os, glob as _glob, json as _json
        if _os.environ.get("MRAG_REUSE_EXTERNAL") == "1":
            _mroot = Path(self.output_dir).parent
            for _bak in sorted(_glob.glob(str(_mroot / "manifest.json.bak_*")),
                               reverse=True):
                try:
                    _steps = _json.load(open(_bak)).get("steps", [])
                except Exception:
                    continue
                for _st in _steps:
                    if (_st.get("name") == self.name
                            and _st.get("status") in ("ok", "skipped")
                            and _st.get("outputs")):
                        # manifest records outputs as a list of
                        # {key, path?, value?}; rebuild the {key: path-or-value}
                        # dict the node normally produces.
                        _items = _st["outputs"]
                        _outs = {}
                        _ok = True
                        for _it in _items:
                            _p = _it.get("path")
                            if _p:
                                if not Path(_p).exists():
                                    _ok = False; break
                                _outs[_it["key"]] = _p
                            else:
                                _outs[_it["key"]] = _it.get("value")
                        if _ok and _outs:
                            self.outputs = _outs
                            print(f"[{self.name}] MRAG_REUSE_EXTERNAL: reused "
                                  f"cached outputs from {Path(_bak).name}",
                                  flush=True)
                            return
        sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/enrichment_mcp/src")
        from enrichment_mcp.tools.enrich import register_enrichment_tools
        tools = capture_mcp_tools(register_enrichment_tools)

        out_dir = Path(self.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        res = json.loads(tools["build_enrichment_section"](
            ranked_consensus_tsv=self.inputs["ranked_consensus_tsv"],
            output_dir=str(out_dir),
            min_shared_tfs=self.params.get("min_shared_tfs", 2),
            max_tfs=self.params.get("max_tfs", 6),
            libraries=self.params.get("libraries"),
            top_terms_per_lib=self.params.get("top_terms_per_lib", 10),
            top_terms_total=self.params.get("top_terms_total", 15),
            dataset_context=self.params.get("dataset_context") or self.inputs.get("dataset_context"),
        ))
        shared = res.get("shared", {}) or {}
        per_tf = res.get("per_tf", {}) or {}
        self.outputs = {
            "shared_dotplot_json":  shared.get("dotplot_json", ""),
            "shared_enrich_csv":    shared.get("enrich_csv", ""),
            "n_shared_genes":       shared.get("n_shared", 0),
            "n_terms_shared":       shared.get("n_terms_total", 0),
            "per_tf_heatmap_json":  per_tf.get("heatmap_json", ""),
            "per_tf_enrich_csv":    per_tf.get("long_csv", ""),
            "n_terms_pertf":        per_tf.get("n_terms", 0),
        }

    def validate(self):
        gates = []
        ns = self.outputs.get("n_shared_genes", 0)
        nt = self.outputs.get("n_terms_shared", 0)
        gates.append({"name":"shared_genes","result":"pass" if ns >= 3 else "warn",
                      "value":ns, "msg":f"{ns} shared targets (≥2 TFs)"})
        gates.append({"name":"shared_terms","result":"pass" if nt >= 5 else "warn",
                      "value":nt, "msg":f"{nt} shared enrichment terms"})
        return gates
