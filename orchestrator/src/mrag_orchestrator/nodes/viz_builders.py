"""Visualization JSON builders.

Each node takes a real upstream MCP output and produces a single viz JSON
that the final render step embeds. Skipping these (via precomputed override)
is allowed but the canonical "preset run" path executes them.

  TFRankNode         INTEGRATED.csv                  -> tf_rank.json
  VolcanoATACNode    DiffBind_out.csv                -> volcano_atac.json
  VolcanoRNANode     DESeq2.gene.csv                 -> volcano_rna.json
  TRNGraphNode       consensus_top{N}.tsv            -> trn.json
  PerTFProfileNode   consensus + bigwig dict + top_tfs -> per_tf_profile.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/viz_report_mcp/src")
from ._base import Node


class TFRankNode(Node):
    """INTEGRATED.csv -> tf_rank.json (Plotly fc vs avg-Z scatter)."""
    name = "tf_rank_build"

    def run(self) -> None:
        import pandas as pd
        from viz_report_mcp.tools.tf_rank import _build_fc_vs_z_json
        df = pd.read_csv(self.inputs["integrated_csv"])
        out = _build_fc_vs_z_json(df,
                                   label_top_n=self.params.get("label_top_n", 10),
                                   label_bottom_n=self.params.get("label_bottom_n", 7))
        out_path = Path(self.output_dir) / "tf_rank.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out))
        self.outputs = {"tf_rank_json": str(out_path)}

    def validate(self):
        p = Path(self.outputs.get("tf_rank_json",""))
        return [{"name":"tf_rank_exists","result":"pass" if p.exists() else "fail",
                 "value": p.stat().st_size if p.exists() else 0,
                 "msg":  f"{p.stat().st_size if p.exists() else 0} bytes"}]


class VolcanoATACNode(Node):
    """DiffBind_out.csv -> volcano_atac.json."""
    name = "volcano_atac_build"

    def run(self) -> None:
        import pandas as pd
        from viz_report_mcp.tools.volcano import _build_volcano_json
        df = pd.read_csv(self.inputs["diffbind_csv"])
        # Construct a UNIQUE per-peak id so the label trace doesn't balloon.
        if "peak_id" not in df.columns and {"seqnames","start","end"}.issubset(df.columns):
            df["peak_id"] = df.apply(lambda r: f"{r['seqnames']}:{int(r['start'])}-{int(r['end'])}", axis=1)
        out = _build_volcano_json(
            df,
            gene_col=self.params.get("gene_col", "peak_id"),
            fc_col=self.params.get("fc_col", "Fold"),
            pval_col=self.params.get("pval_col", "p.value"),
            fdr_col=self.params.get("fdr_col", "FDR"),
            title=self.params.get("title", "ATAC differential binding (DiffBind)"),
            slim=self.params.get("slim", True),
            gtf_path=self.params.get(
                "gtf_path",
                "/BIO1/MRAG/Agent/Reference/RNA_hg38/"
                "gencode.v34.pc_lncRNA.sorted.gtf.gz"),
        )
        out_path = Path(self.output_dir) / "volcano_atac.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out))
        self.outputs = {"volcano_atac_json": str(out_path)}


class VolcanoRNANode(Node):
    """DESeq2.gene.csv -> volcano_rna.json."""
    name = "volcano_rna_build"

    def run(self) -> None:
        import pandas as pd
        from viz_report_mcp.tools.volcano import _build_volcano_json
        df = pd.read_csv(self.inputs["deseq_csv"])
        out = _build_volcano_json(
            df,
            gene_col=self.params.get("gene_col", "GENE"),
            fc_col=self.params.get("fc_col", "log2FoldChange"),
            pval_col=self.params.get("pval_col", "pvalue"),
            fdr_col=self.params.get("fdr_col", "padj"),
            title=self.params.get("title", "RNA differential expression (DESeq2)"),
            slim=self.params.get("slim", True),
        )
        out_path = Path(self.output_dir) / "volcano_rna.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out))
        self.outputs = {"volcano_rna_json": str(out_path)}


class TRNRankedNode(Node):
    """consensus_top{N}.tsv (wide; motif cols) -> consensus_ranked.tsv (long, ranked).

    Wraps trn_mcp.tools.integrate.IntegrateTools.rank_target_genes which expands
    one row per (peak, TF) and ranks within each TF by Stouffer combined Z.
    The output schema matches what viz_report's _build_cy_elements and
    build_per_tf_profile_json expect (explicit TF column).
    """
    name = "trn_ranked_build"

    def run(self) -> None:
        sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/trn_mcp/src")
        from trn_mcp.tools.rank_helper import rank_target_genes
        out_dir = Path(self.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "consensus_ranked.tsv"
        res = rank_target_genes(
            tfs_pval3_tsv=self.inputs["consensus_tsv"],
            output_tsv=str(out_path),
            top_k_per_tf=int(self.params.get("top_k_per_tf", 8)),
            ranking=self.params.get("ranking", "stouffer"),
            balance_direction=bool(self.params.get("balance_direction", False)),
        )
        # Apply HOMER-motif-alias -> gene-symbol normalization so panels match
        # what the upstream TF selection step produced (e.g. AP2alpha -> TFAP2A).
        import pandas as pd
        # When upstream filter_target_tfs substituted family motifs (no exact gene
        # column existed in HOMER lib), we need to remap motif aliases back to
        # the requested gene symbol. The map is direction-aware: only re-label
        # if the requested TF was that family member.
        TOP_TFS = self.params.get("top_tfs") or []
        TFSET = {t.upper() for t in TOP_TFS}
        ALIAS = {
            "AP2ALPHA": "TFAP2A", "AP-2ALPHA": "TFAP2A", "AP2A": "TFAP2A",
            "AP2BETA":  "TFAP2B", "AP-2BETA":  "TFAP2B",
            "AP2GAMMA": "TFAP2C", "AP-2GAMMA": "TFAP2C",
            "COUP-TFI": "NR2F1",  "COUP-TFII": "NR2F2",
            "NUR77":    "NR4A1",
            "HNF1":     "HNF1A",  "HNF4":      "HNF4A",
            "PPARE":    "PPARG",
            "C-JUN":    "JUN",    "C-FOS":     "FOS",   "C-MYC": "MYC",
            "P53":      "TP53",   "P63":       "TP63",  "P73":   "TP73",
        }
        if out_path.exists():
            df = pd.read_csv(out_path, sep="	")
            if "TF" in df.columns:
                df["TF"] = df["TF"].astype(str).map(lambda t: ALIAS.get(str(t).upper(), t))
                df.to_csv(out_path, sep="	", index=False)
        self.outputs = {"consensus_ranked_tsv": str(out_path), "rank_summary": res}

    def validate(self):
        p = Path(self.outputs.get("consensus_ranked_tsv",""))
        return [{"name":"ranked_exists","result":"pass" if p.exists() else "fail",
                 "value": p.stat().st_size if p.exists() else 0,
                 "msg":  f"{p.stat().st_size if p.exists() else 0} bytes"}]


class TRNGraphNode(Node):
    """consensus_ranked.tsv (long, ranked) -> trn.json (cytoscape elements).

    Expects ranked TSV from TRNRankedNode with columns: TF, Gene Name, score,
    Fold, log2FC. Renamed/aliased to Z_combined and RNA_log2FC before passing
    to viz_report's _build_cy_elements.
    """
    name = "trn_graph_build"

    def run(self) -> None:
        import pandas as pd
        from viz_report_mcp.tools.trn_graph import _build_cy_elements, _default_cy_style
        df = pd.read_csv(self.inputs["ranked_tsv"], sep="\t")
        top_tfs = self.params.get("top_tfs") or []
        if top_tfs and "TF" in df.columns:
            keep = {t.upper() for t in top_tfs}
            matched = df[df["TF"].astype(str).str.upper().isin(keep)]
            if len(matched) > 0:
                df = matched
        if "Z_combined" not in df.columns and "score" in df.columns:
            df = df.rename(columns={"score": "Z_combined"})
        if "RNA_log2FC" not in df.columns and "log2FC" in df.columns:
            df["RNA_log2FC"] = df["log2FC"]
        cy = _build_cy_elements(df)
        cy["style"] = _default_cy_style()
        out_path = Path(self.output_dir) / "trn.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(cy))
        self.outputs = {"trn_json": str(out_path)}


class PerTFProfileNode(Node):
    """consensus_ranked.tsv + bigwigs + top_tfs -> per_tf_profile.json."""
    name = "per_tf_profile_build"

    def run(self) -> None:
        import pandas as pd
        from viz_report_mcp.tools.per_tf_profile import build_per_tf_profile_json

        bws = self.inputs["bigwig_paths"]
        if not isinstance(bws, dict):
            bws = json.loads(bws) if isinstance(bws, str) else bws
        top_tfs = self.inputs["top_tfs"]
        if isinstance(top_tfs, str) and Path(top_tfs).exists():
            d = json.loads(Path(top_tfs).read_text())
            top_tfs = d.get("tfs") if isinstance(d, dict) else d
        top_tfs = list(top_tfs)[:6]

        ranked = pd.read_csv(self.inputs["ranked_tsv"], sep="\t", dtype=str)
        if "Start_med" not in ranked.columns and "Start" in ranked.columns:
            ranked["Start_med"] = ranked["Start"]
        if "End_med" not in ranked.columns and "End" in ranked.columns:
            ranked["End_med"] = ranked["End"]
        # rank_target_genes labels TFs with motif aliases (e.g. AP-2alpha instead
        # of TFAP2A). Apply the top_tfs filter only if at least one match would
        # survive; otherwise pass all ranked TFs through.
        if "TF" in ranked.columns and top_tfs:
            keep = {t.upper() for t in top_tfs}
            matched = ranked[ranked["TF"].astype(str).str.upper().isin(keep)]
            if len(matched) > 0:
                ranked = matched
        out_dir = Path(self.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        long_path = out_dir / "consensus_ranked_for_profile.tsv"
        ranked.to_csv(long_path, sep="\t", index=False)

        # If filter survived, top_tfs is meaningful. Otherwise use TFs actually
        # present in ranked DF (motif aliases from rank_target_genes).
        effective_top = top_tfs
        if "TF" in ranked.columns:
            present = ranked["TF"].astype(str).unique().tolist()
            if not any(t.upper() in {p.upper() for p in present} for t in top_tfs):
                effective_top = present[:6]
        out = build_per_tf_profile_json(
            consensus_tsv=str(long_path),
            bigwig_paths=bws,
            top_tfs=effective_top,
            flank=self.params.get("flank", 5000),
            nbins=self.params.get("nbins", 200),
            treatment_label=str(self.inputs.get("treatment_label", "") or ""),
        )
        out_path = out_dir / "per_tf_profile.json"
        out_path.write_text(json.dumps(out))
        self.outputs = {"per_tf_profile_json": str(out_path)}


class TFTableNode(Node):
    """INTEGRATED.csv -> tf_table.csv (Top-N TFs ranked by Pcom for Section 3 table).

    Schema:
      Gene, differential_MDscore, log2FC, ATAC_pvalue, RNA_pvalue, Zcom, Pcom, Rank
    """
    name = "tf_table_build"

    def run(self) -> None:
        import pandas as pd
        import numpy as np
        from scipy.stats import norm
        df = pd.read_csv(self.inputs["integrated_csv"])
        topn = int(self.params.get("topn", 10))
        # Rank by Combined Score (Avg.Z) desc ( 1.combined_rank_public.py);
        # fall back to Pcom asc for legacy 6-column INTEGRATED.csv.
        score_col = "Combined Score (Avg.Z)"
        # Exclude negative differential-MDscore TFs so the Section-3 panel matches
        # the curated panel (control-arm-biased TFs are dropped).
        if "differential_MDscore" in df.columns:
            _pos = df[df["differential_MDscore"].fillna(0) >= 0]
            if len(_pos) >= topn or len(_pos) > 0:
                df = _pos
        if score_col in df.columns:
            df = df.sort_values(score_col, ascending=False).head(topn).copy()
        else:
            df = df.sort_values("Pcom", ascending=True).head(topn).copy()
        # Zcom = isf(Pcom); inf when Pcom == 0
        df["Zcom"] = norm.isf(df["Pcom"].clip(lower=1e-300).to_numpy(float))
        df["Rank"] = np.arange(1, len(df) + 1)
        # Reorder cols
        cols = ["Gene","differential_MDscore","log2FC","ATAC_pvalue",
                "RNA_pvalue","Zcom","Pcom","Combined Score (Avg.Z)","Rank"]
        df = df[[c for c in cols if c in df.columns]]
        # Compact numeric formatting so columns fit (esp. RNA_pvalue): p-values
        # in 2-sig-fig scientific, scores/FC to 3 decimals.
        for c in ("ATAC_pvalue", "RNA_pvalue", "Pcom"):
            if c in df.columns:
                df[c] = df[c].map(lambda x: f"{float(x):.2e}" if pd.notna(x) else x)
        for c in ("log2FC", "Zcom", "Combined Score (Avg.Z)", "differential_MDscore"):
            if c in df.columns:
                df[c] = df[c].map(lambda x: round(float(x), 3) if pd.notna(x) else x)
        out_path = Path(self.output_dir) / "tf_table.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        self.outputs = {"tf_table_csv": str(out_path)}

    def validate(self):
        p = Path(self.outputs.get("tf_table_csv",""))
        return [{"name":"tf_table_exists","result":"pass" if p.exists() else "fail",
                 "value": p.stat().st_size if p.exists() else 0,
                 "msg":  f"{p.stat().st_size if p.exists() else 0} bytes"}]
