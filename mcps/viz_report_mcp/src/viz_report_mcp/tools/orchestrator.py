"""High-level orchestrator: end-to-end report build from pipeline outputs.

Wraps the per-figure builders + render_report so a downstream agent (or HOMER
pipeline finisher) can produce the final HTML with a single MCP call.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, List, Dict


def register_orchestrator_tools(mcp):

    @mcp.tool()
    def build_full_report(
        output_dir: str,
        output_html: str,
        report_title: str,
        # --- Pipeline outputs ---
        integrated_csv: str,                # mcp_combined INTEGRATED.csv (TF Pcom)
        diffbind_csv: str,                  # mcp_diffbind DiffBind_out.csv
        deseq_csv: str,                     # rna_mcp DESeq2.gene.csv
        consensus_tsv: str,                 # mcp_trn consensus_all.tsv (TF x Gene)
        # Section 13 (axis validation) is built by the pipeline's axis_validation node, so this
        # one-shot tool does not produce it.
        # --- Locus & narrative ---
        locus_manifest_json: Optional[str] = None,  # list of {tf,gene,region,png_path}
        # Either supply existing narrative_md, OR set generate_narrative=True
        narrative_md_path: Optional[str] = None,
        generate_narrative: bool = False,
        retriever_script: Optional[str] = "/BIO1/MRAG/Agent/libs/paperqa/scripts/retrieve_cli.py",
        glm_endpoint: str = "http://127.0.0.1:11500",
        glm_model: str = "batiai/qwen3.6-27b:q6",
        conditions_meta: Optional[list] = None,
        # --- ATAC bigwigs (4 samples) ---
        bigwig_oc2_a: str = "",
        bigwig_oc2_b: str = "",
        bigwig_cmyc_a: str = "",
        bigwig_cmyc_b: str = "",
        # --- Choices ---
        top_n_tfs: int = 10,                # for TF table; first 6 used in per-TF profile
        per_tf_top_k: int = 6,
        # --- Pipeline metadata (passthrough) ---
        study_label: str = "",
        pipeline_version: str = "0.1.0",
    ) -> dict:
        """One-shot: build per-TF profile + volcanos + TF rank + TRN JSON then render report.

        Output:
          {output_dir}/per_tf_profile.json
          {output_dir}/volcano_atac.json
          {output_dir}/volcano_rna.json
          {output_dir}/tf_rank.json
          {output_dir}/trn.json
          {output_dir}/tf_table.csv
          + {output_html}                      (final interactive HTML)
        """
        import pandas as pd
        from .per_tf_profile import build_per_tf_profile_json
        from .volcano import _build_volcano_json  # type: ignore[attr-defined]
        from .tf_rank import _build_fc_vs_z_json  # type: ignore[attr-defined]
        from .trn_graph import _build_cy_elements, _default_cy_style  # type: ignore[attr-defined]
        from .render import register_render_tools

        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

        # ---- TF table ----
        integ = pd.read_csv(integrated_csv)
        # Deterministic Top-N: same tiebreaker as mcp_combined.pick_topn
        #   primary: Pcom asc; ties: |log2FC| desc, |MD| desc, Gene asc
        if "Pcom" in integ.columns:
            tmp = integ.copy()
            tmp["_abs_l2fc"] = tmp.get("log2FC", 0).abs()
            tmp["_abs_md"]   = tmp.get("differential_MDscore", 0).abs()
            tmp = tmp.sort_values(
                ["Pcom", "_abs_l2fc", "_abs_md", "Gene"],
                ascending=[True, False, False, True]
            )
            integ_sorted = tmp.drop(columns=["_abs_l2fc","_abs_md"]).head(top_n_tfs)
        else:
            integ_sorted = integ.head(top_n_tfs)
        tf_table_path = out / "tf_table.csv"
        integ_sorted.to_csv(tf_table_path, index=False)
        top_tfs = integ_sorted["Gene"].astype(str).tolist()
        top_6 = top_tfs[:per_tf_top_k]

        # ---- Volcanos ----
        db = pd.read_csv(diffbind_csv)
        # DiffBind has no gene col -- synthesize one from seqnames:start
        if "Gene" not in db.columns:
            db = db.copy()
            db["Gene"] = (db["seqnames"].astype(str) + ":" + db["start"].astype(str))
        v_atac = _build_volcano_json(
            db, gene_col="Gene", fc_col="Fold", pval_col="p.value", fdr_col="FDR",
            title="ATAC volcano (DiffBind)", slim=True,
        )
        (out / "volcano_atac.json").write_text(json.dumps(v_atac))

        de = pd.read_csv(deseq_csv)
        v_rna = _build_volcano_json(
            de, gene_col=("GENE" if "GENE" in de.columns else de.columns[0]),
            fc_col=("log2FoldChange" if "log2FoldChange" in de.columns else "log2FC"),
            pval_col=("pvalue" if "pvalue" in de.columns else "RNA_pvalue"),
            fdr_col=("padj" if "padj" in de.columns else None),
            title="RNA volcano (DESeq2)", slim=True,
        )
        (out / "volcano_rna.json").write_text(json.dumps(v_rna))

        # ---- TF rank ----
        tf_rank = _build_fc_vs_z_json(integ)
        (out / "tf_rank.json").write_text(json.dumps(tf_rank))

        # ---- TRN ----
        trn = _build_cy_elements(pd.read_csv(consensus_tsv, sep='\t'))
        trn['style'] = _default_cy_style()
        (out / "trn.json").write_text(json.dumps(trn))

        # ---- Per-TF profile ----
        bws = {
            "OC2":  [bigwig_oc2_a, bigwig_oc2_b],
            "cMyc": [bigwig_cmyc_a, bigwig_cmyc_b],
        }
        profile = build_per_tf_profile_json(
            consensus_tsv=consensus_tsv, bigwig_paths=bws,
            top_tfs=top_6, flank=5000, nbins=200,
        )
        (out / "per_tf_profile.json").write_text(json.dumps(profile))

        # ---- Locus tracks ----
        locus_tracks: List[Dict] = []
        if locus_manifest_json and Path(locus_manifest_json).exists():
            m = json.loads(Path(locus_manifest_json).read_text())
            for e in (m if isinstance(m, list) else m.get("loci", [])):
                if not e.get("ok", True):
                    continue
                locus_tracks.append({
                    "tf": e["tf"], "gene": e["gene"], "region": e["region"],
                    "png_path": e.get("png_path") or e.get("png"),
                })

        # ---- Optional narrative generation (with citation harness) ----
        audit_summary = None
        if generate_narrative and retriever_script and Path(retriever_script).exists():
            from .narrative import register_narrative_tools
            cap2 = {}
            class _MN:
                def tool(self):
                    def deco(fn):
                        cap2[fn.__name__] = fn
                        return fn
                    return deco
            register_narrative_tools(_MN())
            gen = cap2["generate_narrative"]

            # Build summary inputs from the artifacts we just produced
            tfs_summary = [{"name": r["Gene"], "pcom": float(r.get("Pcom", 0)),
                            "log2FC": float(r.get("log2FC", 0))}
                           for _, r in integ_sorted.iterrows()]
            (out / "tfs_summary.json").write_text(json.dumps(tfs_summary, indent=2))

            cons_df = pd.read_csv(consensus_tsv, sep="\t")
            targets_summary = {}
            for tf_name in top_tfs[:8]:
                sub = cons_df[cons_df["TF"].astype(str).str.upper() == tf_name.upper()].head(5)
                targets_summary[tf_name] = sub[["Gene Name", "Z_mean", "db_Fold_mean", "RNA_log2FC_mean"]].to_dict(orient="records")
            (out / "targets_summary.json").write_text(json.dumps(targets_summary, indent=2))

            narr_path = out / "narrative.md"
            cond_path = None
            if conditions_meta:
                cond_path = out / "conditions_meta.json"
                cond_path.write_text(json.dumps(conditions_meta, indent=2))
            r = gen(
                tfs_summary_json=str(out / "tfs_summary.json"),
                targets_summary_json=str(out / "targets_summary.json"),
                output_md=str(narr_path),
                conditions_meta_json=str(cond_path) if cond_path else None,
                retriever_script=retriever_script,
                retrieval_top_k=4,
                glm_endpoint=glm_endpoint,
                glm_model=glm_model,
                max_tokens=1800,
            )
            narrative_md_path = str(narr_path)
            audit_summary = r.get("citation_audit", {})

        # ---- Render ----
        captured = {}
        class _MCP:
            def tool(self, *a, **kw):
                def deco(fn):
                    captured[fn.__name__] = fn
                    return fn
                return deco
        register_render_tools(_MCP())
        render = captured["render_report"]
        render(
            output_html=output_html,
            report_title=report_title,
            study_label=study_label,
            pipeline_version=pipeline_version,
            volcano_atac_json=str(out / "volcano_atac.json"),
            volcano_rna_json=str(out / "volcano_rna.json"),
            tf_rank_json=str(out / "tf_rank.json"),
            trn_json=str(out / "trn.json"),
            per_tf_profile_json=str(out / "per_tf_profile.json"),
            tf_table_csv=str(tf_table_path),
            locus_tracks=locus_tracks,
            narrative_md_path=narrative_md_path,
            tf_selection=f"Pcom-deterministic Top {top_n_tfs} (INTEGRATED.csv)",
            citation_audit=audit_summary,
        )
        return {"output_html": output_html, "artifacts_dir": str(out),
                "citation_audit": audit_summary,
                "top_tfs": top_tfs, "top_6_profile": top_6,
                "n_locus_tracks": len(locus_tracks)}
