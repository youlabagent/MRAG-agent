"""Locus snapshots node — Section 8 of report.

Picks one representative peak per TF from canonical-N ranked consensus.
**De-duplicates by region**: if multiple TFs share the same top peak/target,
they get merged into a single panel labeled "TF1, TF2, ... → gene". The TFs
that lose their rank-1 peak fall through to rank 2, 3, ... so we surface
diverse peak patterns instead of redundant identical figures.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from ._base import Node



def _peak_core(panel, frac=0.5, min_bp=400):
    """Tighten the differential-peak highlight to the peak core (central
    `frac` of the consensus peak, centered on its midpoint), so the vhighlight
    band marks the summit rather than the full broad min-max peak. Capped at
    a sensible minimum width."""
    cs, ce = int(panel["peak_start"]), int(panel["peak_end"])
    mid = (cs + ce) // 2
    half = max(int((ce - cs) * frac / 2), min_bp // 2)
    return [panel["chrom"], max(cs, mid - half), min(ce, mid + half)]


class LocusSnapshotsNode(Node):
    """Inputs: ranked_consensus_tsv, bigwigs_arm_{a,b}, arm_{a,b}_label
    Params:   n_per_tf (1), flank_bp (5000), max_tfs (8), max_panels (8)
    Outputs:  locus_manifest_json, n_snapshots
    """
    name = "locus_snapshots"

    def run(self) -> None:
        sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/viz_report_mcp/src")
        from viz_report_mcp.tools.tracks import register_track_tools
        from ._base import capture_mcp_tools
        tools = capture_mcp_tools(register_track_tools)

        import pandas as pd
        out_dir = Path(self.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

        _pool_json = self.inputs.get("curated_pool_json", "")
        if _pool_json and Path(_pool_json).exists():
            # Curated mode: the section-12 targets are the curated top-N, whose
            # peak coordinates live in curated_targets/pool.json (NOT the
            # consensus ranked tsv). Build the same-shaped frame from that pool.
            import json as _json_pool
            _prows = []
            for _e in _json_pool.loads(Path(_pool_json).read_text()):
                _g = _e.get("gene"); _ch = _e.get("chrom")
                _s = _e.get("start"); _en = _e.get("end")
                if not (_g and _ch and _s is not None and _en is not None):
                    continue
                _etfs = _e.get("rep_tfs") or _e.get("tfs") or ["TF"]
                for _tf in _etfs:
                    _prows.append({"Gene Name": str(_g), "TF": str(_tf),
                                   "Chr": str(_ch), "Start": int(_s), "End": int(_en),
                                   "Fold": float(_e.get("atac_fold", 0.0) or 0.0),
                                   "log2FC": float(_e.get("log2FC", 0.0) or 0.0),
                                   "combined_p": 0.0, "rank_within_tf": 1})
            df = pd.DataFrame(_prows)
        else:
            df = pd.read_csv(self.inputs["ranked_consensus_tsv"], sep="\t")
        max_tfs = self.params.get("max_tfs", 8)
        max_panels = self.params.get("max_panels", 8)
        flank = self.params.get("flank_bp", 5000)

        # Target selection mirrors the Cytoscape TRN network: take the top
        # target genes by best (lowest) combined_p, then AMONG those keep the
        # ones with the biggest condition difference (max |Fold| = OC2_OE vs
        # cMyc_DDK ATAC log2 fold-change). Same well-defined targets as the
        # graph, biased to the most differentially-accessible ones.
        df = df.copy()
        df["_abs_fold"] = df["Fold"].abs()
        # Candidate pool for locus panels: a WIDE set of highly-significant
        # targets (default top-60 by combined_p, all best_p≈0), from which we
        # then pick the most condition-differential (max |Fold|). The narrow
        # top-15 network set buried dramatic targets (a tenfold larger |Fold|)
        # behind weak ones; widening the pool surfaces them.
        cyto_top_n = self.params.get("locus_pool_n",
                                     self.params.get("cytoscape_top_n", 60))
        _valid = df.dropna(subset=["Gene Name"]).copy()
        _valid = _valid[~_valid["Gene Name"].astype(str).str.strip()
                        .str.lower().isin(["nan", "none", ""])]
        by_target = (_valid.groupby("Gene Name")
                           .agg(best_p=("combined_p", "min"),
                                max_fold=("_abs_fold", "max"),
                                n_tfs=("TF", "nunique")))
        # TRN targets: when the caller passes the exact Section-10 TRN
        # target-gene set (`trn_target_genes`), restrict the locus panels to
        # those genes (the network's own nodes) instead of re-deriving a wide
        # differential pool. Keeps Section-12 consistent with the TRN graph.
        # Fall back to the legacy shared-target pool when no list is provided.
        _trn = [str(g).strip() for g in self.inputs.get("trn_target_genes", [])
                if str(g).strip()]
        if _trn:
            _keep = [g for g in _trn if g in by_target.index]
            cyto_targets = by_target.loc[_keep]
        else:
            # Shared-target requirement: only show targets co-regulated by >=2
            # TFs (the TRN's convergent nodes). Single-TF dramatic targets
            # are excluded even if their |Fold| is large.
            by_target = by_target[by_target["n_tfs"] >= 2]
            cyto_targets = by_target.sort_values("best_p").head(cyto_top_n)
        if _trn:
            # Curated / TRN mode: honor the TRN rank order (trn_target_genes is
            # already rank-sorted, e.g. curated top-N) and show the first
            # max_panels network targets. No direction re-balancing.
            panel_genes = [g for g in _keep if g in cyto_targets.index][:max_panels]
        else:
            # Direction-balanced panel selection (~50/50). Each candidate
            # target's direction = sign of the Fold at its most
            # condition-differential peak: Fold>0 = increased in arm_a
            # (e.g. AR_high), Fold<0 = increased in arm_b (e.g. Basal_like).
            def _signed_fold(_g):
                _sub = _valid[_valid["Gene Name"].astype(str) == str(_g)]
                return float(_sub.loc[_sub["_abs_fold"].idxmax(), "Fold"])
            _cand = cyto_targets.copy()
            _cand["signed_fold"] = [_signed_fold(g) for g in _cand.index]
            _up = (_cand[_cand["signed_fold"] > 0]
                   .sort_values("max_fold", ascending=False).index.tolist())
            _dn = (_cand[_cand["signed_fold"] < 0]
                   .sort_values("max_fold", ascending=False).index.tolist())
            n_a = max_panels // 2
            n_b = max_panels - n_a
            sel_a, sel_b = _up[:n_a], _dn[:n_b]
            # Interleave up/down for visual alternation between conditions.
            panel_genes = []
            for i in range(max(len(sel_a), len(sel_b))):
                if i < len(sel_a):
                    panel_genes.append(sel_a[i])
                if i < len(sel_b):
                    panel_genes.append(sel_b[i])
            # Backfill if one direction is short, so we still show max_panels.
            _leftover = _up[len(sel_a):] + _dn[len(sel_b):]
            _leftover.sort(key=lambda g: float(_cand.loc[g, "max_fold"]), reverse=True)
            for g in _leftover:
                if len(panel_genes) >= max_panels:
                    break
                panel_genes.append(g)
            panel_genes = panel_genes[:max_panels]

        # Build sample list (for plot_locus_track)
        a_label = self.inputs.get("arm_a_label", "OC2")
        b_label = self.inputs.get("arm_b_label", "cMyc")
        bws_a = self.inputs.get("bigwigs_arm_a", [])
        bws_b = self.inputs.get("bigwigs_arm_b", [])
        # Track ORDER and colour both follow the arm ROLE, not its position.
        # Arm a/b order is alphabetical (sorted() upstream), which put the
        # control arm on top with the blue palette; the report convention is
        # treatment on TOP (red), control BELOW (blue).  `treatment_label` is
        # cfg.conditions[0].label (the pipeline-wide treatment convention).
        _RED, _BLUE = ["#d62728", "#e57878"], ["#1f77b4", "#4a90d9"]
        _treat = str(self.inputs.get("treatment_label", "") or "").strip().lower()

        def _is_treatment(lab: str) -> bool:
            lab = str(lab or "").strip().lower()
            if not lab or not _treat:
                return False
            return lab == _treat or lab in _treat or _treat in lab

        # treatment arm first (top), control arm second (bottom)
        _arms = [(a_label, bws_a), (b_label, bws_b)]
        if _is_treatment(b_label) and not _is_treatment(a_label):
            _arms = [_arms[1], _arms[0]]
        samples = []
        for _lab, _bws in _arms:
            for i, p in enumerate(_bws):
                samples.append([f"{_lab}_rep{i+1}", p, None])
        if not samples:
            raise ValueError("LocusSnapshotsNode needs at least one bigwig.")

        if _is_treatment(_arms[0][0]):
            track_colors = [_RED, _BLUE]      # treatment top (red), control below (blue)
        elif _is_treatment(_arms[1][0]):
            track_colors = [_BLUE, _RED]
        else:                                 # unknown roles -> positional fallback
            track_colors = None

        # One panel per selected target gene, using that gene's most
        # condition-differential peak (max |Fold|) for the genomic window.
        panels: list[dict] = []
        for gene in panel_genes:
            sub = (_valid[_valid["Gene Name"].astype(str) == str(gene)]
                   .sort_values("_abs_fold", ascending=False))
            if not len(sub):
                continue
            row = sub.iloc[0]
            tfs = [str(t) for t in sub["TF"].dropna().unique().tolist()][:6]
            panels.append({
                "tfs": tfs or ["TF"],
                "gene": str(gene),
                "chrom": str(row["Chr"]),
                "peak_start": int(row["Start"]),
                "peak_end":   int(row["End"]),
                "atac_fold":  float(row["Fold"]),
                "rna_log2fc": float(row["log2FC"]) if "log2FC" in row else None,
                "rank_within_tf": int(row["rank_within_tf"]) if "rank_within_tf" in row else 0,
            })

        # Render each panel
        manifest = []
        for panel in panels[:max_panels]:
            s2 = max(0, panel["peak_start"] - flank)
            e2 = panel["peak_end"] + flank
            region = f"{panel['chrom']}:{s2}-{e2}"
            tf_label = ", ".join(panel["tfs"])
            slug = "_".join(panel["tfs"]) + "_" + panel["gene"]
            try:
                r = tools["plot_locus_track"](
                    gene_label=panel["gene"], region=region,
                    out_dir=str(out_dir),
                    title=slug, samples=samples, colors=track_colors,
                    max_value="shared_max", width=16.5, dpi=180, flank_bp=0,
                    highlight=_peak_core(panel),
                    # The gene-model track underneath the signal. Without this
                    # the helper falls back to its hg38 default and a mouse
                    # panel gets HUMAN gene models under a mouse caption
                    # (2026-09-23).
                    gene_gtf=str(self.params.get("gene_gtf") or "") or None,
                )
            except Exception as ex:
                print(f"[locus] {tf_label}/{panel['gene']} failed: {ex}", flush=True)
                continue
            if not r.get("ok"):
                continue
            manifest.append({
                "tf":         tf_label,        # "TF1, TF2, TF3"
                "tfs":        panel["tfs"],    # list
                "gene":       panel["gene"],
                "region":     region,
                "png_path":   r.get("png"),
                "atac_fold":  panel["atac_fold"],
                "rna_log2fc": panel["rna_log2fc"],
            })

        manifest_path = out_dir / "locus_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        self.outputs = {
            "locus_manifest_json": str(manifest_path),
            "n_snapshots": len(manifest),
        }
