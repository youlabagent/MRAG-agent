"""SETrnBuilderNode — Section 10b: SE -> TF -> Target TRN inputs.

1. Re-runs the Step-2 column filter + DiffBind/RNA join on the treatment
   min-max motif.tsv with the EXTENDED TF list (panel top-N + Section-8 enriched
   motif TFs) so enriched-TF -> Target edges can be read with the same
   promoter-presence rule the Section-10 graph uses (panel-TF edges are taken
   verbatim from the Section-10 anno/inter files).
2. Calls viz_report_mcp.tools.cytoscape_trn.produce_se_trn_inputs -> se_anno.txt,
   se_input_inter.txt, se_cyjs.json (elements + style + SE table + summary).

MCP step 2a (2026-10-12): the filter / join of step 1 already were MCP tool calls
(homer-mcp filter_target_tfs, trn-mcp join_diffbind_and_rna, through their nodes); the processed-matrix join is now
the trn-mcp tool join_matrix_tfs_pval2 (the former _matrix_join) and step 2 with its display-only DAStk refill the
viz-report-mcp tool build_se_trn (the former _display_md_fallback + produce_se_trn_inputs call), both unchanged and
called through capture_mcp_tools (MRAG_TOOL_TRANSPORT=mcp: an MCP tools/call; =direct: the function).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from ._base import Node
from . import _stage_tools
sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/viz_report_mcp/src")


class SETrnBuilderNode(Node):
    """Inputs:
        graph_anno, graph_inter        Section-10 graph files (trn_cytoscape/anno.txt, input_inter.txt)
        se_top_tsv, se_tf_edges_tsv, se_genes_tsv   SERoseNode / SEMotifNode outputs
        motif_tsv                  homer_motif/<treat>_peak_minmax.motif.tsv (extended Pval2 source)
        diffbind_csv, deseq_csv, integrated_csv
        panel_tfs (list), enriched_tfs (list), label
    Params: atac_fallback_tsv (optional table or raw DAStk differential file),
            dastk_md_csv (this run's DAStk differential file; its DASTK/<cond>/<cond>.narrowPeak.filt
            inputs are reused), refill_dir (display-only motif beds, default
            Reference/ATAC_hg38/grch38_motifs_refill), skip_extended_pval2 (bool)
    Outputs: se_cyjs_json, se_anno_path, se_inter_path, pval2_extended, n_* summary
    """
    name = "se_trn"

    def run(self) -> None:
        from .homer import FilterTargetTfsNode, JoinDiffbindRnaNode
        out = Path(self.output_dir); out.mkdir(parents=True, exist_ok=True)
        i = self.inputs; label = str(i.get("label", "treat"))
        panel = [str(t).upper() for t in i.get("panel_tfs", [])]
        enriched = [str(t).upper() for t in i.get("enriched_tfs", []) if str(t).upper() not in panel]
        tfs = panel + enriched
        tag = f"{label}_se{len(tfs)}"
        pval2 = ""
        if not self.params.get("skip_extended_pval2") and i.get("motif_tsv") and Path(i["motif_tsv"]).is_file():
            f = FilterTargetTfsNode(inputs={"motif_tsv": i["motif_tsv"]},
                                       params={"condition_label": tag, "target_tfs": tfs},
                                       output_dir=out)
            f.execute()
            if f.status != "ok":
                raise RuntimeError(f"extended Pval2 filter failed: {f.error}")
            j = JoinDiffbindRnaNode(inputs={"tfs_txt": f.outputs["tfs_txt"], "diffbind_csv": i["diffbind_csv"],
                                               "deseq_csv": i["deseq_csv"]},
                                       params={"condition_label": tag}, output_dir=out)
            j.execute()
            _n_join = int(j.outputs.get("n_rows_emitted", 0) or 0) if j.status == "ok" else 0
            if _n_join > 0:
                pval2 = j.outputs["tfs_pval2_txt"]
            elif i.get("canonical_peaks_bed") and Path(i["canonical_peaks_bed"]).is_file():
                # processed-matrix input: the Step-2 join does not fit the voom DiffBind /
                # DESeq2 tables -> same joins as curated build_pval2_from_matrix
                pval2 = _stage_tools.tools("matrix_join")["join_matrix_tfs_pval2"](
                    tfs_txt=f.outputs["tfs_txt"], canonical_peaks_bed=i["canonical_peaks_bed"],
                    diffbind_csv=i["diffbind_csv"], deseq_csv=i["deseq_csv"],
                    output_path=str(out / f"{tag}_peak_matrix.TFs.Pval2.txt"))
            elif j.status != "ok":
                raise RuntimeError(f"extended Pval2 join failed: {j.error}")
            else:
                pval2 = j.outputs["tfs_pval2_txt"]
        # SE -> TF -> Target inputs + the display-only DAStk refill (viz-report-mcp build_se_trn)
        self.outputs = _stage_tools.tools("se_trn")["build_se_trn"](
            graph_anno=i["graph_anno"], graph_inter=i["graph_inter"],
            se_top_tsv=i["se_top_tsv"], se_tf_edges_tsv=i["se_tf_edges_tsv"], se_genes_tsv=i["se_genes_tsv"],
            integrated_csv=i["integrated_csv"], output_dir=str(out), panel_tfs=panel, enriched_tfs=enriched,
            pval2_txt=pval2, deseq_csv=i.get("deseq_csv", ""), label=label,
            atac_fallback_tsv=self.params.get("atac_fallback_tsv") or "",
            dastk_md_csv=self.params.get("dastk_md_csv") or "", refill_dir=self.params.get("refill_dir") or "",
            genome=self.params.get("genome", "hg38"), se_method=self.params.get("se_method") or "")

    def validate(self):
        g = []
        n_t = int(self.outputs.get("n_targets", 0) or 0); n_se = int(self.outputs.get("n_se", 0) or 0)
        n_e = int(self.outputs.get("n_se_tf_edges", 0) or 0)
        g.append({"name": "se_trn_targets", "result": "pass" if n_t > 0 else "fail", "value": n_t,
                  "msg": "" if n_t else "no Target nodes found in the Section-10 graph"})
        g.append({"name": "se_trn_layers", "result": "pass" if (n_se > 0 and n_e > 0) else "fail",
                  "value": {"n_se": n_se, "n_se_tf_edges": n_e}, "msg": "" if (n_se and n_e) else "SE layer empty"})
        miss = self.outputs.get("tf_atac_missing", "")
        if miss:
            g.append({"name": "tf_atac_coverage", "result": "warn", "value": miss,
                      "msg": f"TFs without ATAC MD score (blue fill): {miss}"})
        return g
