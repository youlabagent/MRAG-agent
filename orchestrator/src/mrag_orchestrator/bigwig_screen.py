"""Screen per-arm replicate bigWigs by ATAC condition-specificity.

Matrix-input runs (e.g. TCGA-ATAC) ship dozens of per-rep bigWigs per arm.
Naively picking the 2 largest files gives non-representative tracks (a "largest"
sample can even be anti-correlated with its own arm, e.g. an AR_high tumour that
is more open at Basal-specific peaks) -> weak/absent diagonal in the peak
heatmap and inconsistent locus replicate tracks.

Instead, score each replicate by how cleanly it separates the two conditions on
the DiffBind-derived arm-specific peak sets, then pick the top-2 per arm.

  spec_score(sample) = mean(log1p CPM on OWN-arm-specific peaks)
                     - mean(log1p CPM on OTHER-arm-specific peaks)

Ties broken by file size (deeper coverage). Falls back to size-only when the
count matrix / DiffBind is unavailable.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional


def _sample_of(bw_path: str) -> str:
    b = Path(bw_path).name
    for suf in (".insertions.bw", ".bw"):
        if b.endswith(suf):
            return b[: -len(suf)]
    return b


def _size(p: str) -> int:
    try:
        return Path(p).stat().st_size
    except Exception:
        return 0


def screen_top2_by_specificity(
    bw_by_arm: dict[str, list[str]],
    count_matrix: str,
    group_tsv: str,
    diffbind_csv: str,
    fdr: float = 0.05,
    top_n: int = 2,
) -> Optional[dict[str, list[str]]]:
    """Return {arm: [top_n bigWig paths]} chosen by specificity score, or None
    if the inputs can't be used (caller should then fall back to size-based)."""
    try:
        import numpy as np, pandas as pd
        if len(bw_by_arm) != 2:
            return None
        arm_a, arm_b = sorted(bw_by_arm.keys())

        # --- arm-specific peak sets from DiffBind (Fold sign + FDR) ---
        db = pd.read_csv(diffbind_csv)
        need = {"seqnames", "start", "end", "Fold", "FDR"}
        if not need.issubset(db.columns):
            return None
        def _key(df):
            return (df["seqnames"].astype(str) + ":"
                    + df["start"].astype(float).astype("int64").astype(str) + "-"
                    + df["end"].astype(float).astype("int64").astype(str))
        db["key"] = _key(db)
        sig = db["FDR"].notna() & (db["FDR"] < fdr)
        # positive Fold == gained in arm_a (DiffBind treat=first condition, and
        # runner picks arm_a = sorted()[0] == treat label used for Conc_<a>).
        a_spec = set(db.loc[sig & (db["Fold"] > 0), "key"])
        b_spec = set(db.loc[sig & (db["Fold"] < 0), "key"])
        if not a_spec or not b_spec:
            return None

        # --- per-sample log1p CPM on each specific set ---
        m = pd.read_csv(count_matrix, sep="\t")
        meta = [c for c in ("seqnames", "start", "end", "name", "score")
                if c in m.columns]
        samples = [c for c in m.columns if c not in meta]
        m["key"] = _key(m)
        idx = {k: i for i, k in enumerate(m["key"])}
        ai = [idx[k] for k in a_spec if k in idx]
        bi = [idx[k] for k in b_spec if k in idx]
        if not ai or not bi:
            return None
        X = m[samples].to_numpy(dtype=float)
        col = X.sum(axis=0, keepdims=True)
        col[col == 0] = 1.0
        lc = np.log1p(X / col * 1e6)
        a_mean = lc[ai].mean(axis=0)
        b_mean = lc[bi].mean(axis=0)
        score = {s: (a_mean[j] - b_mean[j]) for j, s in enumerate(samples)}

        out: dict[str, list[str]] = {}
        for arm, own_is_a in ((arm_a, True), (arm_b, False)):
            paths = bw_by_arm[arm]
            def _sc(p):
                s = _sample_of(p)
                v = score.get(s)
                if v is None:
                    return (-1e9, _size(p))
                return (v if own_is_a else -v, _size(p))
            out[arm] = sorted(paths, key=_sc, reverse=True)[:top_n]
        return out
    except Exception:
        return None
