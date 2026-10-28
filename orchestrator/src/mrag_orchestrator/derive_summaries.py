"""Auto-derive aux JSONs needed by NarrativeNode:
  - tfs_summary.json:     [{name, pcom, log2FC}, ...] = the branch's top{N}.json panel
  - targets_summary.json: {TF: [{Gene Name, Z_mean, db_Fold_mean, RNA_log2FC_mean}, ...]}
                          from consensus_top{N}.tsv (per TF, top 5 by Z_mean)
  - conditions_meta.json: from Sample list (label + optional free-text fields)

These mirror viz_report_mcp.orchestrator's inline derivations so we get
identical output structure.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional


def _num_or_none(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v   # NaN -> None


def _pick_topn_panel(integrated_csv: str | Path, n: int) -> tuple[list[str], dict]:
    """Top-n TF panel for INTEGRATED.csv computed by combined_mcp.pick_topn
    itself (written to a throw-away temp dir), so the ranking precedence and the
    differential_MDscore >= 0 filter can never drift from top{N}.json.
    Returns (tfs, {gene: detail_row})."""
    import sys, shutil, tempfile
    sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/combined_mcp/src")
    try:
        from combined_mcp.tools.pipeline import register_pipeline_tools
        from .nodes._base import capture_mcp_tools
        tmp = tempfile.mkdtemp(prefix="derive_tfs_topn_")
        try:
            res = json.loads(capture_mcp_tools(register_pipeline_tools)["pick_topn"](
                str(integrated_csv), tmp, [int(n)]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if res.get("status") == "ok":
            by = res["by_n"][str(int(n))]
            return ([str(t) for t in by["tfs"]],
                    {str(r.get("Gene")): r for r in by.get("details", [])})
    except Exception:
        pass
    # Last resort (e.g. legacy INTEGRATED.csv lacking pick_topn's required
    # columns): same precedence + MDscore filter as pick_topn, inline.
    import pandas as pd
    df = pd.read_csv(integrated_csv)
    for col in ("Rank_Combined_Score", "Rank_Pcom", "Rank"):
        if col in df.columns:
            df = df.sort_values(col, ascending=True).reset_index(drop=True)
            break
    else:
        try:
            from combined_mcp.tools.pipeline import _topn_sort_key
            df = _topn_sort_key(df)
        except Exception:
            if "Pcom" in df.columns:
                df = df.sort_values("Pcom", ascending=True).reset_index(drop=True)
    if "differential_MDscore" in df.columns:
        _pos = df[df["differential_MDscore"].fillna(0) >= 0]
        if len(_pos) > 0:
            df = _pos.reset_index(drop=True)
    head = df.head(int(n))
    recs = head.to_dict(orient="records")
    return [str(r.get("Gene", "")) for r in recs], {str(r.get("Gene", "")): r for r in recs}


def derive_tfs_summary(integrated_csv: str | Path, top_n: int = 10,
                       top_json: str | Path | None = None) -> list[dict]:
    """Branch TF panel as [{name, pcom, log2FC}, ...] with EXACTLY the TF set
    and order of the report's top{N}.json (Section 3 table, TRN, homer_filter).

    2026-09-16 fix (narrtf): this used to re-sort INTEGRATED.csv by
    Rank_Combined_Score WITHOUT pick_topn's differential_MDscore >= 0 filter
    (added to pick_topn 2026-07) and was called with top_n=max(N+2, 10), so the
    narrative + executive summary discussed TFs that are not in the report
    panel (8 such TFs on PRAD).

    Source precedence:
      1. top_json (the branch's top{N}.json from combined_mcp.pick_topn): its
         "tfs" list is authoritative for membership + order; pcom/log2FC come
         from its "details" rows, else from INTEGRATED.csv.
      2. no top_json: combined_mcp.pick_topn run on INTEGRATED.csv (temp dir).
      3. pick_topn unusable: same precedence + MDscore filter inline.
    `top_n` truncates the panel (0/None = keep all of top_json).
    """
    rows_by_gene: dict = {}
    tfs: list[str] = []
    tj = Path(top_json) if top_json else None
    if tj is not None and tj.is_file():
        d = json.loads(tj.read_text())
        if isinstance(d, dict):
            tfs = [str(t) for t in (d.get("tfs") or [])]
            for r in (d.get("details") or []):
                if isinstance(r, dict) and r.get("Gene") is not None:
                    rows_by_gene.setdefault(str(r["Gene"]), r)
        elif isinstance(d, list):
            tfs = [str(t.get("name") if isinstance(t, dict) else t) for t in d]
    integ_ok = bool(integrated_csv) and Path(integrated_csv).is_file()
    if not tfs and integ_ok:
        tfs, rows_by_gene = _pick_topn_panel(integrated_csv, int(top_n or 10))
    if top_n:
        tfs = tfs[:int(top_n)]
    missing = [t for t in tfs if t not in rows_by_gene]
    if missing and integ_ok:
        import pandas as pd
        df = pd.read_csv(integrated_csv)
        if "Gene" in df.columns:
            for r in df[df["Gene"].astype(str).isin(missing)].to_dict(orient="records"):
                rows_by_gene.setdefault(str(r["Gene"]), r)
    return [{"name":   t,
             "pcom":   _num_or_none(rows_by_gene.get(t, {}).get("Pcom")),
             "log2FC": _num_or_none(rows_by_gene.get(t, {}).get("log2FC"))}
            for t in tfs]


def derive_targets_summary(consensus_tsv: str | Path,
                           tfs: list[str],
                           top_targets: int = 5) -> dict[str, list[dict]]:
    """For each TF in `tfs`, find top `top_targets` actual target genes from
    the ranked consensus (proper Stouffer Z + motif binding + direction filter).

    Wraps trn_mcp.tools.rank_helper.rank_target_genes — the same function that
    feeds the TRN graph. Previously this function used `|log2FC|` as a "proxy
    for Z_mean" AND was bugged in the binding check (string "nan" treated as
    truthy under dtype=str), so it returned identical top-genes (sorted by
    raw |log2FC|) for every TF without requiring actual motif binding.

    Returns: {tf: [{Gene Name, Z_mean, db_Fold_mean, RNA_log2FC_mean}, ...]}
    """
    import sys, json as _json, tempfile, os
    sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/trn_mcp/src")
    from trn_mcp.tools.rank_helper import rank_target_genes
    import pandas as pd

    # Run rank_target_genes on the consensus to get the canonical ranked TSV
    tmp_out = tempfile.NamedTemporaryFile(prefix="ranked_targets_", suffix=".tsv", delete=False).name
    res = _json.loads(rank_target_genes(
        tfs_pval3_tsv=str(consensus_tsv),
        output_tsv=tmp_out,
        top_k_per_tf=int(top_targets),
        ranking="stouffer",
    ))
    if res.get("status") != "ok":
        try: os.unlink(tmp_out)
        except OSError: pass
        return {tf: [] for tf in tfs}

    df = pd.read_csv(tmp_out, sep="\t")
    try: os.unlink(tmp_out)
    except OSError: pass

    # rank_target_genes uses UPPERCASE TF symbols (rank_helper._tf_symbol)
    # The map from motif alias -> gene symbol (AP2alpha -> TFAP2A, COUP-TFI -> NR2F1, etc.)
    # is applied by TRNRankedNode, NOT inside rank_target_genes. We need the same
    # alias normalization here to match user-supplied tf list.
    ALIAS = {
        "AP2ALPHA": "TFAP2A", "AP-2ALPHA": "TFAP2A", "AP2A": "TFAP2A",
        "AP2BETA":  "TFAP2B", "AP-2BETA":  "TFAP2B",
        "AP2GAMMA": "TFAP2C", "AP-2GAMMA": "TFAP2C",
        "COUP-TFI": "NR2F1",  "COUP-TFII": "NR2F2",
        "NUR77":    "NR4A1",
        "C-JUN":    "JUN",    "C-FOS":     "FOS",   "C-MYC": "MYC",
    }
    if "TF" in df.columns:
        df["TF"] = df["TF"].astype(str).str.upper().map(lambda u: ALIAS.get(u, u))

    out: dict[str, list[dict]] = {}
    for tf in tfs:
        sub = df[df["TF"].astype(str).str.upper() == tf.upper()]
        records = []
        for _, r in sub.head(top_targets).iterrows():
            records.append({
                "Gene Name":       str(r.get("Gene Name", "")),
                "Z_mean":          float(r.get("score", 0) or 0),
                "db_Fold_mean":    float(r.get("Fold", 0) or 0),
                "RNA_log2FC_mean": float(r.get("log2FC", 0) or 0),
            })
        out[tf] = records
    return out


def derive_conditions_meta(samples_or_conditions,
                           extra_meta: Optional[dict[str, dict]] = None) -> list[dict]:
    """Build conditions_meta from Sample list or Condition list.

    Args:
      samples_or_conditions: either list of Sample (then group by arm) or
                             list of Condition.
      extra_meta: optional {arm_label: {perturbation, concept, rationale}}
                  to populate free-text fields the sheet doesn't carry.
    """
    extra_meta = extra_meta or {}
    if not samples_or_conditions:
        return []
    # Detect input type
    first = samples_or_conditions[0]
    if hasattr(first, "arm") or (isinstance(first, dict) and "arm" in first):
        # list of Sample → group by arm
        seen = []
        for s in samples_or_conditions:
            arm = s.arm if hasattr(s, "arm") else s.get("arm","")
            if arm and arm not in seen: seen.append(arm)
        labels = seen
    else:
        # list of Condition
        labels = [c.label if hasattr(c,"label") else c["label"]
                  for c in samples_or_conditions]
    out = []
    for label in labels:
        m = dict(extra_meta.get(label, {}))
        out.append({
            "label":        label,
            "perturbation": m.get("perturbation", f"{label} (experimental arm)"),
            "concept":      m.get("concept",      f"{label}"),
            "rationale":    m.get("rationale",    "Comparison arm"),
        })
    return out
