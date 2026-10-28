"""Processed-matrix -> combined-step adapters (flex_input extension, 2026-07-07).

Two R-backed converters that let MRAG start from processed expression/accessibility
matrices (e.g. TCGA pan-cancer log2 matrices) instead of raw FASTQ:

  RNA:  log2 / normalized gene x sample matrix (+ group labels)
        -> limma group DE + Entrez->symbol -> DESeq2.gene.csv  (deseq_csv jump)
  ATAC: peak-count matrix (seqnames,start,end,... + sample counts) (+ group labels)
        -> edgeR-voom differential accessibility -> group-enriched peak BEDs
           (treatment/control) -> existing DAStkNode -> MD-score.

Groups are supplied as a 2-col TSV (sample<tab>group); treat/ref map to
conditions[0]/[1] labels. Downstream artifacts are exactly what the existing
flex_input paths already consume, so nothing downstream changes.
"""
from __future__ import annotations
import csv
import subprocess
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
RNA_R = str(_HERE / "rna_log2_deseq.R")
ATAC_R = str(_HERE / "atac_matrix_diff.R")
RSCRIPT = "Rscript"


def build_group_tsv(sheet_path: str, sample_col: str, group_col: str,
                    out_tsv: str, require_col: Optional[str] = None) -> str:
    """Emit a sample<tab>group TSV from a detect/meta sample sheet.

    `sample_col` must match the matrix column IDs for that modality (ATAC and
    RNA can differ). `require_col` (e.g. 'has_rna') if given keeps only truthy
    rows. Duplicate samples keep the first group seen."""
    seen: dict[str, str] = {}
    with open(sheet_path, newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for r in rd:
            if require_col and str(r.get(require_col, "")).strip().lower() not in ("true", "1", "yes"):
                continue
            s = (r.get(sample_col) or "").strip()
            g = (r.get(group_col) or "").strip()
            if s and g and s not in seen:
                seen[s] = g
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w") as fh:
        fh.write("sample\tgroup\n")
        for s, g in seen.items():
            fh.write(f"{s}\t{g}\n")
    return out_tsv


def _read_header(matrix: str) -> list[str]:
    with open(matrix) as fh:
        first = fh.readline().rstrip("\n")
    sep = "\t" if "\t" in first else ","
    return [c.strip() for c in first.split(sep)]


def _infer_group_col(rows: list[dict], sample_col: str) -> Optional[str]:
    """Pick the sheet column that looks like a group/cluster label: few unique
    short values (2..6), not the sample id column."""
    if not rows:
        return None
    prefer = ("condition", "group", "arm", "subtype", "class", "label",
              "cluster", "denovo_k2", "denovo")
    cols = [c for c in rows[0].keys() if c and c != sample_col]
    for name in prefer:
        for c in cols:
            if c.lower() == name:
                return c
    best = None
    for c in cols:
        vals = {(r.get(c) or "").strip() for r in rows if (r.get(c) or "").strip()}
        if 2 <= len(vals) <= 6 and all(len(v) <= 12 for v in vals):
            best = c if best is None else best
    return best


def autobuild_group_tsvs(sheet_path: str, out_dir: str,
                         atac_matrix: Optional[str] = None,
                         rna_matrix: Optional[str] = None,
                         group_col: Optional[str] = None) -> dict:
    """From a rich metadata sheet, auto-emit per-modality sample<->group TSVs by
    matching each matrix's column headers to a sheet column. Returns
    {atac_group_tsv?, rna_group_tsv?}. This is the 'LLM/heuristic figures out the
    conversion' entry: no fixed schema required, only that some sheet column's
    values match the matrix sample headers and a low-cardinality group column
    exists (denovo_k2/group/cluster/...)."""
    rows = list(csv.DictReader(open(sheet_path, newline=""), delimiter="\t"))
    if not rows:
        return {}
    cols = [c for c in rows[0].keys() if c]
    gcol = group_col or _infer_group_col(rows, sample_col="")
    if not gcol:
        return {}
    out: dict = {}
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def _match_sample_col(matrix: str) -> Optional[str]:
        hdr = set(_read_header(matrix)[1:])
        best, best_hits = None, 0
        for c in cols:
            if c == gcol:
                continue
            vals = {(r.get(c) or "").strip() for r in rows}
            hits = len(vals & hdr)
            if hits > best_hits:
                best, best_hits = c, hits
        return best if best_hits >= 1 else None

    if atac_matrix:
        sc = _match_sample_col(atac_matrix)
        if sc:
            out["atac_group_tsv"] = build_group_tsv(
                sheet_path, sc, gcol, str(Path(out_dir) / "atac_group.tsv"))
    if rna_matrix:
        sc = _match_sample_col(rna_matrix)
        if sc:
            req = "has_rna" if "has_rna" in cols else None
            out["rna_group_tsv"] = build_group_tsv(
                sheet_path, sc, gcol, str(Path(out_dir) / "rna_group.tsv"),
                require_col=req)
    return out


def _matrix_sample_cols(matrix: str, sep: Optional[str] = None):
    """Return (df, sample_cols) for a matrix, handling both a genomic peak-count
    matrix (seqnames/start/end/... meta cols) and a gene x sample expression
    matrix (leading id/name annotation cols)."""
    import pandas as pd
    from ..detector import _PEAK_COORD0, _PEAK_META, _matrix_layout
    if sep is None:
        with open(matrix) as _f:
            sep = "\t" if "\t" in _f.readline() else ","
    df = pd.read_csv(matrix, sep=sep)
    low = {c: str(c).strip().lower() for c in df.columns}
    cols = list(df.columns)
    if low[cols[0]] in _PEAK_COORD0:
        sample_cols = [c for c in cols if low[c] not in _PEAK_META
                       and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9]
    else:
        lay = _matrix_layout(df)
        sample_cols = lay["sample_cols"] if lay else []
    return df, sample_cols


def _write_group_tsv(partition: dict, out_tsv: str) -> str:
    """partition = {group_label: [sample_col, ...]} -> sample<tab>group TSV."""
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w") as fh:
        fh.write("sample\tgroup\n")
        for g, cols in partition.items():
            for s in cols:
                fh.write(f"{s}\t{g}\n")
    return out_tsv


def llm_group_columns(sample_cols: list, modality: str,
                      endpoint: str, model: str) -> Optional[dict]:
    """Ask the LLM to split matrix sample-column NAMES into exactly two
    experimental groups when the names encode a group signal (e.g. OE/control,
    KO/WT, treated/untreated, tumor/normal). Returns
    {"treatment": {label, cols}, "control": {label, cols}} or None if the names
    carry no group signal (opaque IDs) or the reply cannot be parsed."""
    import json as _json
    from ..detector import _llm_call
    cols_txt = "\n".join(f"- {c}" for c in sample_cols)
    prompt = (
        f"You are labeling {modality}-seq sample columns for a two-group "
        f"differential analysis (treatment vs control).\n"
        f"Sample column names:\n{cols_txt}\n\n"
        "If (and ONLY if) the NAMES themselves encode two experimental groups "
        "(e.g. OE vs control, KO vs WT, treated vs untreated, drug vs DMSO, "
        "tumor vs normal), assign EVERY column to exactly one of two groups.\n"
        "If the names are opaque identifiers with no group signal (e.g. random "
        "barcodes/UUIDs/accessions), set has_signal=false.\n"
        "Reply with ONE JSON object only, no prose:\n"
        '{"has_signal": true|false, "treatment_label": "<short>", '
        '"control_label": "<short>", "treatment": ["col", ...], '
        '"control": ["col", ...]}'
    )
    try:
        raw = _llm_call(endpoint, model, prompt, num_predict=1024)
    except Exception:
        return None
    s, e = raw.find("{"), raw.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        obj = _json.loads(raw[s:e + 1])
    except Exception:
        return None
    if not obj.get("has_signal"):
        return None
    t = [c for c in obj.get("treatment", []) if c in set(sample_cols)]
    c = [c for c in obj.get("control", []) if c in set(sample_cols)]
    if len(t) < 1 or len(c) < 1 or (len(t) + len(c)) < len(sample_cols) * 0.8:
        return None
    tl = (obj.get("treatment_label") or "treatment").strip() or "treatment"
    cl = (obj.get("control_label") or "control").strip() or "control"
    if tl == cl:
        tl, cl = "treatment", "control"
    return {"treatment": {"label": tl, "cols": t},
            "control":   {"label": cl, "cols": c}}


def kmeans_group_columns(matrix: str, sep: Optional[str] = None,
                         k: int = 2, top_features: int = 25000) -> Optional[dict]:
    """De-novo two-group split of matrix SAMPLES by unsupervised k-means on the
    top-variable features (mirrors prep_prad PCA->kmeans2). Use when column names
    carry no group signal. Returns {"G1": [...], "G2": [...]} (labels are
    arbitrary; biological meaning must be verified)."""
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    df, sample_cols = _matrix_sample_cols(matrix, sep)
    if len(sample_cols) < 2 * k:
        return None
    X = df[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    var = X.var(axis=1)
    top = var.sort_values(ascending=False).head(min(top_features, len(var))).index
    M = X.loc[top].T.values                       # samples x features
    M = StandardScaler().fit_transform(M)
    ncomp = int(min(10, M.shape[0] - 1, M.shape[1]))
    if ncomp >= 2:
        M = PCA(n_components=ncomp, random_state=0).fit_transform(M)
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(M)
    part: dict = {}
    for col, l in zip(sample_cols, lab):
        part.setdefault(f"G{int(l) + 1}", []).append(col)
    if len(part) < 2:
        return None
    return part


def resolve_matrix_groups(matrix: str, modality: str, out_tsv: str,
                          endpoint: str, model: str,
                          sep: Optional[str] = None) -> Optional[dict]:
    """Two-tier group resolution when no metadata sheet is available:
      1. LLM infers groups from column-name signal (OE/control, KO/WT, ...).
      2. else unsupervised k-means k=2 on the matrix values.
    Writes a sample<->group TSV and returns
    {tsv, method: 'llm_names'|'kmeans', ordered_labels: [treat, ctrl], warning}."""
    _, sample_cols = _matrix_sample_cols(matrix, sep)
    if len(sample_cols) < 2:
        return None
    g = llm_group_columns(sample_cols, modality, endpoint, model)
    if g:
        tl, cl = g["treatment"]["label"], g["control"]["label"]
        _write_group_tsv({tl: g["treatment"]["cols"],
                          cl: g["control"]["cols"]}, out_tsv)
        return {"tsv": out_tsv, "method": "llm_names",
                "ordered_labels": [tl, cl],
                "warning": (f"{modality}: groups inferred from column NAMES by LLM "
                            f"(treatment='{tl}' vs control='{cl}'). VERIFY.")}
    part = kmeans_group_columns(matrix, sep)
    if part:
        labels = sorted(part.keys())[:2]
        _write_group_tsv(part, out_tsv)
        return {"tsv": out_tsv, "method": "kmeans",
                "ordered_labels": labels,
                "warning": (f"{modality}: no group signal in column names -> de-novo "
                            f"k=2 clustering ({', '.join(labels)}). Data-driven groups; "
                            f"treatment/control assignment is arbitrary and MUST be "
                            f"verified biologically.")}
    return None


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("adapter R failed:\n" + (p.stderr or p.stdout)[-2000:])
    return (p.stdout or "") + (p.stderr or "")


def run_rna_log2_adapter(matrix: str, group_tsv: str, out_csv: str,
                         treat: str, ref: str) -> str:
    """log2/normalized RNA matrix -> DESeq2.gene.csv (ID,GENE,baseMean,
    log2FoldChange,lfcSE,stat,pvalue,padj). coef = treat vs ref."""
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    _run([RSCRIPT, RNA_R, matrix, group_tsv, out_csv, treat, ref])
    if not Path(out_csv).exists():
        raise RuntimeError("RNA adapter produced no output CSV")
    return out_csv


def run_atac_matrix_adapter(matrix: str, group_tsv: str, out_dir: str,
                            treat: str, ref: str, floor_n: int = 5000) -> tuple[str, str]:
    """ATAC peak-count matrix -> (treatment_bed, control_bed): group-enriched
    peaks (voom differential; top `floor_n` per direction when few pass FDR).
    Returns BEDs for DAStkNode treatment_files/control_files."""
    od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)
    _run([RSCRIPT, ATAC_R, matrix, group_tsv, str(od), treat, ref, str(floor_n)])
    tb = od / "treatment_diff.bed"
    cb = od / "control_diff.bed"
    if not (tb.exists() and cb.exists()):
        raise RuntimeError("ATAC adapter produced no BED outputs")
    return str(tb), str(cb)


def write_canonical_bed_from_matrix(matrix: str, out_bed: str,
                                    sep: Optional[str] = None) -> str:
    """Emit a canonical ATAC peak BED (chrom/start/end/name) from a peak-count
    matrix whose first columns are peak coordinates (seqnames,start,end,[name]).
    This is the 'canonical ATAC BED' HOMER annotates for the matrix input path."""
    import csv as _csv
    if sep is None:
        sep = "\t" if matrix.endswith((".tsv", ".txt")) else ","
    coord0 = {"seqnames", "seqname", "chr", "chrom", "chromosome"}
    with open(matrix, newline="") as fh:
        rd = _csv.reader(fh, delimiter=sep)
        header = next(rd)
        low = [c.strip().lower() for c in header]
        ci = next((i for i, c in enumerate(low) if c in coord0), 0)
        si = low.index("start") if "start" in low else 1
        ei = low.index("end") if "end" in low else 2
        ni = low.index("name") if "name" in low else None
        n = 0
        with open(out_bed, "w") as out:
            for row in rd:
                if len(row) <= max(ci, si, ei):
                    continue
                chrom, start, end = row[ci], row[si], row[ei]
                name = row[ni] if (ni is not None and ni < len(row)) else f"peak_{n}"
                out.write(f"{chrom}\t{start}\t{end}\t{name}\n")
                n += 1
    return out_bed
