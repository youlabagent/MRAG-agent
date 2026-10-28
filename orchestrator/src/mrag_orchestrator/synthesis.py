"""Cross-variant synthesis: take Top 4/6/8 outputs and compute the
robust core / N-dependent fringe / stability score.

All deterministic. No LLM.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path


def _load_topn_tfs(topn_json_path: str | Path) -> list[str]:
    """combined_mcp.pick_topn output -> list of TF gene symbols."""
    d = json.loads(Path(topn_json_path).read_text())
    if isinstance(d, dict):
        return list(d.get("tfs", []))
    if isinstance(d, list):
        return list(d)
    return []


def _load_trn_edges(consensus_tsv: str | Path) -> set[tuple[str, str]]:
    """Read consensus_top{N}.tsv (HOMER + DiffBind + DESeq joined) and
    extract unique (TF_motif_column, Gene Name) edges.

    The motif columns themselves contain the TF in their prefix.
    We use 'Gene Name' (HOMER promoter gene annotation) as the target.
    Each non-empty cell in a motif column counts as an edge between
    that motif's TF and the row's Gene Name.
    """
    edges: set[tuple[str, str]] = set()
    p = Path(consensus_tsv)
    if not p.is_file():
        return edges
    with open(p) as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = reader.fieldnames or []
        # motif columns are those that look like "<TF>(Family)/..." -- after the
        # canonical base columns.  style: PeakID/Chr/Start/End/Annotation/
        # Distance to TSS/Gene Name + motif_cols + DiffBind cols + RNA cols
        base = {"PeakID","Chr","Start","End","Annotation","Distance to TSS",
                "Gene Name","Fold","DiffBind_Pval","DiffBind_FDR",
                "region_start","region_end","TYPE","log2FC","RNA_pval","RNA_padj"}
        motif_cols = [c for c in cols if c not in base]
        for row in reader:
            gene = row.get("Gene Name") or ""
            if not gene: continue
            for mc in motif_cols:
                if row.get(mc):  # non-empty motif hit
                    tf = mc.split("(")[0].split("/")[0].strip()
                    if tf:
                        edges.add((tf, gene))
    return edges


def _jaccard(a: set, b: set) -> float:
    u = a | b
    return (len(a & b) / len(u)) if u else 0.0


def synthesize_cross_variant(
    variants_artifacts: dict[int, dict],
    output_json: str | Path,
) -> dict:
    """
    Args:
      variants_artifacts: {N: {"top_json": "...", "consensus_tsv": "..."}}
                          (per N in {4,6,8})
      output_json:        where to write cross_variant_summary.json
    Returns: summary dict (also written to output_json).
    """
    per_n: dict[int, dict] = {}
    for N, art in variants_artifacts.items():
        tfs   = set(_load_topn_tfs(art["top_json"])) if art.get("top_json") else set()
        edges = _load_trn_edges(art["consensus_tsv"]) if art.get("consensus_tsv") else set()
        per_n[N] = {"tfs": tfs, "edges": edges}

    Ns = sorted(per_n.keys())
    if len(Ns) == 0:
        raise ValueError("No variant artifacts provided")

    # Pairwise / overall intersections
    tfs_all     = set.intersection(*[per_n[N]["tfs"] for N in Ns])      if Ns else set()
    tfs_union   = set.union(*[per_n[N]["tfs"] for N in Ns])             if Ns else set()
    edges_core  = set.intersection(*[per_n[N]["edges"] for N in Ns])    if Ns else set()
    edges_union = set.union(*[per_n[N]["edges"] for N in Ns])           if Ns else set()

    # Fringe TFs: in top-largest N but not in top-smallest N
    n_lo, n_hi = min(Ns), max(Ns)
    fringe_tfs    = per_n[n_hi]["tfs"] - per_n[n_lo]["tfs"]
    intermediate_tfs = (
        set.intersection(*[per_n[N]["tfs"] for N in Ns if N >= n_lo])
        - per_n[n_lo]["tfs"]
    )

    stability = _jaccard(per_n[n_lo]["tfs"], per_n[n_hi]["tfs"])

    summary = {
        "schema_version": "1.0",
        "Ns":               Ns,
        "robust_core_tfs":  sorted(tfs_all),
        "intermediate_tfs": sorted(intermediate_tfs),
        "fringe_tfs":       sorted(fringe_tfs),
        "n_tfs_per_N":      {str(N): sorted(per_n[N]["tfs"]) for N in Ns},

        "robust_edges":     sorted(list(e) for e in edges_core),
        "union_edges_count":     len(edges_union),
        "robust_edges_count":    len(edges_core),
        "edges_per_N_count": {str(N): len(per_n[N]["edges"]) for N in Ns},

        "stability_score_jaccard_lo_hi": stability,
        "tf_union_size":                 len(tfs_union),
    }
    Path(output_json).write_text(json.dumps(summary, indent=2, default=str))
    return summary


def render_cross_variant_markdown(summary: dict) -> str:
    """Generate a tight markdown block summarising the cross-variant view.

    Designed to be embedded as a new section in the final HTML.
    """
    lines = ["## Cross-variant synthesis (Top " + "/".join(str(n) for n in summary["Ns"]) + ")"]
    lines.append("")
    core = summary.get("robust_core_tfs", [])
    interm = summary.get("intermediate_tfs", [])
    fringe = summary.get("fringe_tfs", [])
    lines.append(f"- **Robust core TFs ({len(core)})**: " +
                 (", ".join(f"**{t}**" for t in core) if core else "*(none)*"))
    if interm:
        lines.append(f"- **Intermediate ({len(interm)})**: " + ", ".join(interm))
    if fringe:
        lines.append(f"- **N-dependent fringe ({len(fringe)})**: " +
                     ", ".join(f"<span style='opacity:0.6'>{t}</span>" for t in fringe))
    lines.append("")
    lines.append(f"- Stability (Jaccard, Top{min(summary['Ns'])} vs Top{max(summary['Ns'])}): "
                 f"**{summary.get('stability_score_jaccard_lo_hi', 0.0):.2f}**")
    lines.append(f"- Robust TRN edges: {summary.get('robust_edges_count', 0)} / "
                 f"union {summary.get('union_edges_count', 0)}")
    return "\n".join(lines)
