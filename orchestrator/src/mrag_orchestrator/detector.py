"""FASTQ sample auto-detection.

Two phases, sequential:
  1. Rule-based parser  — deterministic, no LLM. Covers common naming conventions
                          for cancer ATAC/RNA studies (TF perturbation, drug
                          treatment, knockout, overexpression).
  2. LLM arm grouping   — on by default (`use_llm=False` turns it off). After the
                          rule pass every sample name (plus the public SRA/GEO
                          title of an accession-named file) goes to the LLM,
                          which GROUPS the samples into arms; a rule label is
                          kept wherever it maps 1:1 onto one LLM group.

Hallucination control for the LLM phase:
  - Strict JSON output; an unparseable answer or an unreachable model keeps the
    rule-based arms (with a warning).
  - temperature = 0
  - The LLM only sees sample names and public SRA/GEO titles (no fabricated biology)
"""
from __future__ import annotations
import gzip, json, re, time, urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import (
    Sample, SampleConfidence, Condition, SampleSheet, DetectWarning,
)


# ============================================================
# Cancer-domain keyword dictionary
# ------------------------------------------------------------
# Each entry: canonical_label  ->  list of aliases (case-insensitive regex parts)
# Coverage: prostate and breast cancer TFs, common controls / vectors,
#           perturbation types and common drug perturbations.
# ============================================================

# TF / gene names commonly used in perturbation studies (overexpression/KD/KO)
TF_KEYWORDS = {
    # Prostate cancer lineage / plasticity drivers
    "ONECUT2":   [r"ONECUT2", r"OC2(?![A-Z])", r"HNF6B"],
    "MYC":       [r"c[\-_]?MYC", r"\bMYC\b", r"MYCC"],
    "AR":        [r"\bAR\b", r"AndrogenReceptor"],
    "FOXA1":     [r"FOXA1"],
    "HOXB13":    [r"HOXB13"],
    "ERG":       [r"\bERG\b", r"TMPRSS2[\-_]ERG"],
    "ETV1":      [r"ETV1"],
    "ETV4":      [r"ETV4"],
    "GATA2":     [r"GATA2"],
    "NKX3-1":    [r"NKX3[\-_]?1"],
    "REST":      [r"\bREST\b"],
    "ASCL1":     [r"ASCL1"],
    "NEUROD1":   [r"NEUROD1"],
    # Breast cancer
    "ESR1":      [r"ESR1", r"\bER\+?\b"],
    "GATA3":     [r"GATA3"],
    "FOXM1":     [r"FOXM1"],
    "STAT3":     [r"STAT3"],
    # Common controls / vectors
    "EV":        [r"\bEV\b", r"empty[\-_]?vector", r"\bMOCK\b", r"pLenti(?!\w)"],
    "GFP":       [r"\bGFP\b"],
    "shCtrl":    [r"sh[Cc]trl", r"shScramble", r"shSCR", r"shNT"],
}

# Modifier suffixes denoting perturbation type
PERTURBATION_MODIFIERS = {
    "OE":  [r"(?<![A-Za-z])OE(?![A-Za-z])", r"overexpr", r"overexpression"],
    "KD":  [r"(?<![A-Za-z])KD(?![A-Za-z])", r"knockdown", r"shRNA"],
    "KO":  [r"(?<![A-Za-z])KO(?![A-Za-z])", r"knockout", r"sgRNA", r"CRISPR"],
    "WT":  [r"(?<![A-Za-z])WT(?![A-Za-z])", r"wildtype", r"wild[\-_]type"],
    "MUT": [r"(?<![A-Za-z])MUT(?![A-Za-z])", r"mutant"],
    "DDK": [r"(?<![A-Za-z])DDK(?![A-Za-z])", r"FLAG[\-_]?tag", r"(?<![A-Za-z])HA[\-_]?tag(?![A-Za-z])"],
}

# Drug / small molecule perturbations
DRUG_KEYWORDS = {
    "ENZA":    [r"ENZA", r"enzalutamide", r"MDV3100"],
    "ABI":     [r"\bABI\b", r"abiraterone"],
    "DHT":     [r"\bDHT\b", r"dihydrotestosterone"],
    "veh":     [r"\bveh\b", r"vehicle", r"DMSO"],
    "TGFB":    [r"TGF[\-_]?B(eta)?[12]?"],
}

# Control / reference markers
CONTROL_MARKERS = [r"\bctrl\b", r"\bcontrol\b", r"\bDMSO\b", r"\bveh", r"\bWT\b",
                   r"\bmock\b", r"empty[\-_]?vector", r"\bEV\b"]

# Replicate markers
REPLICATE_PATTERNS = [
    re.compile(r"[\-_]rep(?:licate)?[\-_]?(\d+)", re.IGNORECASE),
    re.compile(r"[\-_]r(\d+)(?=[\-_\.])", re.IGNORECASE),
    re.compile(r"[\-_](\d+)(?=[\-_\.])"),    # final fallback
]

# Pair suffix patterns
PAIR_PATTERNS = [
    (re.compile(r"_R1(_|\.)"), re.compile(r"_R2(_|\.)")),
    (re.compile(r"_1(_|\.)"),  re.compile(r"_2(_|\.)")),
    (re.compile(r"\.R1(_|\.)"),re.compile(r"\.R2(_|\.)")),
    (re.compile(r"_F(_|\.)"),  re.compile(r"_R(_|\.)")),
]

# Modality hints: adapter sequences / library prep markers
MODALITY_HINTS = {
    "ATAC": [r"(?<![A-Za-z])ATAC(?![A-Za-z])", r"(?<![A-Za-z])Tn5(?![A-Za-z])", r"open[\-_]?chromatin"],
    "RNA":  [r"(?<![A-Za-z])RNA(?![A-Za-z])", r"RNA[\-_]?seq", r"(?<![A-Za-z])mRNA(?![A-Za-z])", r"polyA", r"TruSeq[\-_]?RNA"],
}


# ============================================================
# Helpers
# ============================================================

def _match_any(name: str, patterns: list[str]) -> bool:
    return any(re.search(p, name, flags=re.IGNORECASE) for p in patterns)


def _match_first(name: str, dictionary: dict[str, list[str]]) -> Optional[str]:
    """Return the canonical_label of the first dictionary entry whose any
    pattern matches name. Order matters; iterate keys in insertion order."""
    for canon, patterns in dictionary.items():
        if _match_any(name, patterns):
            return canon
    return None


def _pair_fastqs(files: list[Path]) -> list[tuple[Path, Optional[Path]]]:
    """Group FASTQ files into (R1, R2) pairs. Single-end => (R1, None)."""
    used: set[Path] = set()
    pairs: list[tuple[Path, Optional[Path]]] = []
    for f in sorted(files):
        if f in used:
            continue
        name = f.name
        matched = False
        for r1_pat, r2_pat in PAIR_PATTERNS:
            if r1_pat.search(name):
                r2_name = r1_pat.sub(lambda m: r2_pat.pattern.replace("(_|\\.)", m.group(1) if m.group(1) else "."), name)
                # Safer: just transform R1->R2 token by replacing the matched substring
                m = r1_pat.search(name)
                start, end = m.span()
                r2_token   = name[start:end].replace("R1","R2").replace("_1","_2").replace(".R1",".R2").replace("_F","_R")
                r2_guess   = f.parent / (name[:start] + r2_token + name[end:])
                if r2_guess.exists():
                    pairs.append((f, r2_guess)); used.add(f); used.add(r2_guess)
                    matched = True; break
        if not matched and f not in used:
            pairs.append((f, None)); used.add(f)
    return pairs


def _peek_read_length(fq: Path, max_reads: int = 5) -> Optional[int]:
    """Peek first few reads of a (possibly gzipped) FASTQ; return modal seq length."""
    try:
        opener = gzip.open if fq.suffix == ".gz" else open
        with opener(fq, "rt") as fh:
            lens = []
            for i, line in enumerate(fh):
                if i // 4 >= max_reads: break
                if i % 4 == 1:  lens.append(len(line.strip()))
            if lens:
                return max(set(lens), key=lens.count)
    except Exception:
        return None
    return None


def _extract_sample_basename(r1_name: str) -> str:
    """Strip pair suffix and extension to get a sample base name."""
    s = r1_name
    s = re.sub(r"\.f(ast)?q(\.gz)?$", "", s, flags=re.IGNORECASE)
    # inside-string pair tokens (e.g., _R1_, _R1.)
    for r1_pat, _ in PAIR_PATTERNS:
        s = r1_pat.sub("_", s)
    # end-of-string pair tokens (after extension already stripped)
    s = re.sub(r"(_|\.)(R1|R2|F|R|1|2)$", "", s)
    s = re.sub(r"_+$", "", s)
    return s


def _classify_arm_rule(name: str) -> tuple[str, Confidence_str]:
    """Apply keyword-based arm classification. Return (arm_label, confidence).

    Strategy:
      - If a TF name is matched AND a perturbation modifier matches => '<TF>-<MOD>'.
      - Else if a TF name alone matches => '<TF>' (medium confidence).
      - Else if a drug perturbation matches => '<DRUG>'.
      - Else if a control marker matches => 'control'.
      - Else => '' (unknown, low confidence).
    """
    tf  = _match_first(name, TF_KEYWORDS)
    mod = _match_first(name, PERTURBATION_MODIFIERS)
    drug = _match_first(name, DRUG_KEYWORDS)
    ctrl = _match_any(name, CONTROL_MARKERS)
    if tf and mod:
        return f"{tf}-{mod}", "high"
    if tf and drug:
        return f"{tf}+{drug}", "high"
    if tf:
        return tf, "medium"
    if drug:
        return drug, "medium"
    if ctrl:
        return "control", "medium"
    return "", "low"


Confidence_str = str   # forward ref (avoid import gymnastics)


def _classify_replicate_rule(name: str) -> tuple[int, Confidence_str]:
    for pat in REPLICATE_PATTERNS:
        m = pat.search(name)
        if m:
            try:
                rep = int(m.group(1))
                if 1 <= rep <= 20:
                    # high confidence if pattern was explicit "rep"
                    return rep, ("high" if "rep" in pat.pattern.lower() else "medium")
            except (ValueError, IndexError):
                continue
    return 1, "low"


def _classify_modality_rule(name: str, dir_hint: str = "") -> tuple[str, Confidence_str]:
    """Use filename + parent dir hint."""
    for canon, pats in MODALITY_HINTS.items():
        if _match_any(name + " " + dir_hint, pats):
            return canon, "high"
    return "unknown", "low"


# ============================================================
# Content-based input-stage classification (flexible input)
# ============================================================
# Beyond raw FASTQ, MRAG accepts already-processed inputs per modality:
#   ATAC:  peaks BED/narrowPeak       -> DAStk MD-score only
#   RNA:   raw counts/TPM gene matrix -> DESeq2 only
#          DESeq2 result CSV          -> jump straight to combined
# classify_input() sniffs one file; classify_inputs() groups a set and maps
# processed artifacts onto PrecomputedArtifacts keys (arm split via
# _classify_arm_rule).

_FASTQ_EXTS = (".fastq", ".fq", ".fastq.gz", ".fq.gz")
_DESEQ_STAT_COLS = {"basemean", "log2foldchange", "lfcse", "stat", "pvalue", "padj"}


def _read_head_lines(path: Path, n: int = 3) -> list[str]:
    op = gzip.open if str(path).endswith(".gz") else open
    out: list[str] = []
    try:
        with op(path, "rt", errors="replace") as f:
            for _ in range(n):
                ln = f.readline()
                if not ln:
                    break
                out.append(ln.rstrip("\n"))
    except OSError:
        return []
    return out


# Annotation/metadata columns that are NOT expression samples.
_ANNO_COLS = {"target_id", "transcript_id", "tx_id", "tx_name", "transcript",
              "length", "eff_length", "effective_length", "gene_id", "geneid",
              "ensembl_gene_id", "gene_name", "genename", "gene", "symbol",
              "name", "id", "est_counts"}
_NAME_COLS = {"gene_name", "genename", "symbol", "name"}
_ID_COLS = {"gene_id", "geneid", "target_id", "transcript_id", "tx_id",
            "ensembl_gene_id", "id"}
_PEAK_COORD0 = {"seqnames", "seqname", "chr", "chrom", "chromosome"}
_PEAK_META = {"seqnames", "seqname", "chr", "chrom", "chromosome",
              "start", "end", "width", "strand", "name", "score",
              "peak", "peak_id", "peakid"}


def _matrix_layout(df) -> Optional[dict]:
    """Identify {sample_cols, name_col, id_col} of a gene/transcript x sample
    expression matrix, tolerating leading annotation columns (gene_id,
    gene_name, length, eff_length, ...). Returns None if not a matrix."""
    import pandas as pd
    cols = list(df.columns)
    low = {c: str(c).strip().lower() for c in cols}
    sample_cols = [c for c in cols
                   if low[c] not in _ANNO_COLS
                   and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9]
    if len(sample_cols) < 2:
        return None
    name_col = next((c for c in cols if low[c] in _NAME_COLS), None)
    id_col = next((c for c in cols if low[c] in _ID_COLS), None)
    if id_col is None:
        non_sample = [c for c in cols if c not in sample_cols and c != name_col]
        id_col = non_sample[0] if non_sample else None
    return {"sample_cols": sample_cols, "name_col": name_col, "id_col": id_col}


def _looks_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _sniff_matrix_quant(path: Path, sep: str) -> Optional[str]:
    """Return 'tpm'|'counts' if path is a gene/transcript x sample expression
    matrix else None. TPM: per-column sum ~= 1e6 (full annotation) OR fractional
    values (gene-subset TPM sums < 1e6); counts: all-integer sample columns."""
    try:
        import pandas as pd
        df = pd.read_csv(path, sep=sep)
    except Exception:
        return None
    if df.shape[0] < 50:
        return None
    # A real expression matrix names its columns after samples. When a file has
    # no header row, pandas promotes the first data row, and those "names" parse
    # as numbers. That is how a headerless DAStk MD-score table (motif, p-value,
    # counts, md scores) was read as a sample matrix and, because its last column
    # holds signed score differences, classified as log2-normalized RNA.
    hdr = [str(c).strip() for c in df.columns]
    numeric_hdr = sum(1 for c in hdr if _looks_numeric(c))
    if hdr and numeric_hdr / len(hdr) > 0.5:
        return None
    lay = _matrix_layout(df)
    if not lay:
        return None
    import pandas as pd
    smp = df[lay["sample_cols"]].apply(pd.to_numeric, errors="coerce")
    colsums = smp.sum()
    if bool(((colsums > 9.0e5) & (colsums < 1.1e6)).all()):
        return "tpm"
    frac = float(((smp.round() != smp) & smp.notna()).mean().mean())
    if frac <= 0.2:
        return "counts"
    # non-integer matrix: log2-normalized (bounded/negative) vs TPM-subset
    mx = float(smp.max().max())
    has_neg = bool((smp < 0).any().any())
    if has_neg or mx < 35.0:
        return "log2"
    return "tpm"


def _peak_matrix_nsamples(path: Path, sep: str) -> int:
    """Count integer-valued (raw-count) non-meta sample columns of a peak
    matrix. Integer requirement excludes log2-normalized peak matrices and
    QC-column peak files (percentGC etc.), keeping only voom-suitable counts."""
    try:
        import pandas as pd
        df = pd.read_csv(path, sep=sep, nrows=200)
    except Exception:
        return 0
    import pandas as pd
    n = 0
    for c in df.columns:
        if str(c).strip().lower() in _PEAK_META:
            continue
        col = pd.to_numeric(df[c], errors="coerce")
        if col.notna().mean() <= 0.9:
            continue
        frac = float(((col.round() != col) & col.notna()).mean())
        if frac <= 0.02:   # integer -> raw counts
            n += 1
    return n


def _is_group_tsv(head, sep) -> bool:
    """Sample<->group mapping: 2 cols, header names sample/group."""
    cells = [c.strip().lower() for c in head[0].split(sep)]
    return (len(cells) == 2 and cells[0] in ("sample", "sample_id", "id")
            and cells[1] in ("group", "arm", "condition", "cluster"))


def classify_input(path) -> dict:
    """Sniff one file; return {kind, ...}. kind in:
      fastq | atac_peaks | rna_deseq_csv | rna_matrix | unknown
    rna_matrix additionally carries quant_source ('tpm'|'counts')."""
    p = Path(path)
    name = p.name.lower()
    if name.endswith(_FASTQ_EXTS):
        return {"kind": "fastq"}
    head = _read_head_lines(p, 3)
    if not head:
        return {"kind": "unknown"}
    first = head[0]
    sep = "\t" if "\t" in first else ("," if "," in first else None)
    if sep is None:
        return {"kind": "fastq"} if first.startswith("@") else {"kind": "unknown"}
    hdr_cells = [c.strip().lower() for c in first.split(sep)]
    if len(_DESEQ_STAT_COLS & set(hdr_cells)) >= 3:
        return {"kind": "rna_deseq_csv", "sep": sep}
    cells = first.split(sep)
    if (len(cells) >= 3 and re.match(r"^(chr)?[0-9A-Za-z_]+$", cells[0])
            and cells[1].isdigit() and cells[2].isdigit()
            and (name.endswith((".bed", ".narrowpeak")) or ".narrowpeak" in name
                 or (sep == "\t" and cells[0].startswith("chr")))):
        return {"kind": "atac_peaks"}
    low_cells = [c.strip().lower() for c in cells]
    if _is_group_tsv(head, sep):
        return {"kind": "group_tsv", "sep": sep}
    if low_cells[0] in _PEAK_COORD0 and "start" in low_cells and "end" in low_cells:
        # Genomic peak matrix: raw-count sample cols -> atac_count_matrix (voom).
        # Non-integer (log2-normalized) or QC-only peak files are ATAC-side but
        # not voom-suitable -> unknown (never fall through to RNA sniffing).
        if _peak_matrix_nsamples(p, sep) >= 2:
            return {"kind": "atac_count_matrix", "sep": sep}
        return {"kind": "unknown"}
    quant = _sniff_matrix_quant(p, sep)
    if quant == "log2":
        return {"kind": "rna_log2_matrix", "sep": sep}
    if quant:
        return {"kind": "rna_matrix", "quant_source": quant, "sep": sep}
    return {"kind": "unknown"}


def _matrix_arms(path, sep: str) -> list[str]:
    """Classify each expression-matrix column to an arm (best-effort)."""
    head = _read_head_lines(Path(path), 1)
    if not head:
        return []
    cols = head[0].split(sep)[1:]
    arms = []
    for c in cols:
        a, _ = _classify_arm_rule(c)
        if a:
            arms.append(a)
    return arms


def classify_inputs(paths) -> dict:
    """Partition a set of inputs by content and map processed artifacts onto
    (transient) precomputed suggestions. Returns
    {fastqs, precomputed, arms, unknown, notes}. ATAC peaks are grouped BY ARM
    (dict); treatment/control ordering is resolved later against conditions."""
    fastqs: list[str] = []
    unknown: list[str] = []
    notes: list[str] = []
    arms: list[str] = []
    atac_by_arm: dict[str, list[str]] = {}
    pre: dict = {}
    for pth in paths:
        sp = str(pth)
        info = classify_input(pth)
        k = info["kind"]
        if k == "fastq":
            fastqs.append(sp)
        elif k == "atac_peaks":
            arm, _c = _classify_arm_rule(Path(sp).name)
            atac_by_arm.setdefault(arm or "", []).append(sp)
            if arm:
                arms.append(arm)
        elif k == "rna_deseq_csv":
            pre["deseq_csv"] = sp
            notes.append(f"RNA DESeq2 result -> jump to combined: {Path(sp).name}")
        elif k == "rna_matrix":
            pre["rna_counts_matrix"] = sp
            pre["rna_quant_source"] = info.get("quant_source", "counts")
            arms.extend(_matrix_arms(sp, info.get("sep", "\t")))
            notes.append(f"RNA {info.get('quant_source')} matrix -> DESeq2 only: {Path(sp).name}")
        elif k == "rna_log2_matrix":
            pre["rna_log2_matrix"] = sp
            arms.extend(_matrix_arms(sp, info.get("sep", "\t")))
            notes.append(f"RNA log2/normalized matrix -> limma DE -> DESeq2.gene.csv "
                         f"(needs sample<->group TSV): {Path(sp).name}")
        elif k == "atac_count_matrix":
            pre["atac_count_matrix"] = sp
            notes.append(f"ATAC peak-count matrix -> voom differential -> DAStk "
                         f"(needs sample<->group TSV): {Path(sp).name}")
        elif k == "group_tsv":
            _n = Path(sp).name.lower()
            if "atac" in _n:
                pre["atac_group_tsv"] = sp
            elif "rna" in _n:
                pre["rna_group_tsv"] = sp
            else:
                pre.setdefault("atac_group_tsv", sp)
                pre.setdefault("rna_group_tsv", sp)
            notes.append(f"sample<->group map: {Path(sp).name}")
        else:
            unknown.append(sp)
    if atac_by_arm:
        labelled = {a: v for a, v in atac_by_arm.items() if a}
        if len(labelled) >= 2:
            pre["atac_peaks_by_arm"] = labelled
            notes.append(f"ATAC peaks BED -> DAStk only (arms: {', '.join(sorted(labelled))})")
        else:
            notes.append(f"ATAC peaks found but arm split ambiguous; "
                         f"leave for manual --precomputed")
            for v in atac_by_arm.values():
                unknown.extend(v)
    # de-dup arms preserving order
    seen = set(); arms = [a for a in arms if not (a in seen or seen.add(a))]
    return {"fastqs": fastqs, "precomputed": pre, "arms": arms,
            "unknown": unknown, "notes": notes}


# ============================================================
# Main entrypoint (rule-based)
# ============================================================

# Directory names a pipeline writes into. Their FASTQs are copies or trimmed
# derivatives of the raw ones, and scanning them produced duplicate samples plus
# orphans: trim_galore names files _R1_val_1 / _R2_val_2, which the pair patterns
# do not match, so every read ended up as its own unpaired sample of unknown
# modality. Real precomputed inputs (a TPM matrix, a peaks file, a bigWig) live
# beside these directories, not inside them, so skipping the directories keeps
# the flexible-input path intact.
WORKDIR_NAMES = {
    "trim_galore", "fastqc", "alignments", "picard_tmp", "temp_picard",
    "tmp", "_tmp", "logs",
    # DASTK is NOT excluded: it holds the per-arm narrowPeak.filt that
    # atac_peaks_by_arm needs, unlike macs2_*/ whose filtered output sits
    # one level up.
}
WORKDIR_PREFIXES = ("macs2_", "_star_", "rsem_tmp", "star_tmp")


def is_pipeline_workdir(name: str) -> bool:
    n = name.strip().lower()
    return n in WORKDIR_NAMES or n.startswith(WORKDIR_PREFIXES)


def iter_input_files(root: Path) -> list[Path]:
    """Every file under root, skipping pipeline working directories at any depth."""
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            parts = p.relative_to(root).parts[:-1]
        except ValueError:
            parts = p.parts[:-1]
        if any(is_pipeline_workdir(x) for x in parts):
            continue
        out.append(p)
    return out


def attach_organism(sheet: SampleSheet) -> SampleSheet:
    """Give the sheet (and every sample) the organism the download recorded.

    `mrag sra-fetch` writes ENA's scientific_name into sra_manifest.json and
    sra_names.json next to the FASTQs. Until 2026-09-23 detect ignored both, so
    the species died between the download and the run and every genome argument
    fell back to human. Here it is read back, per sample where the basename is
    known and per study otherwise.

    A set that really does mix species is reported as a HIGH-severity warning
    and flagged needs_user_action: resolving it to one genome is exactly the
    silent wrong answer this is meant to prevent, and `organism.resolve_many()`
    raises for the same reason further down.
    """
    from .organism import (species_by_basename, species_from_sra_dir,
                           resolve as _resolve_org)
    dirs, per_dir_map, per_dir_species = [], {}, {}
    for smp in sheet.samples:
        d = str(Path(smp.fastq_r1).parent) if smp.fastq_r1 else ""
        if d and d not in per_dir_map:
            per_dir_map[d] = species_by_basename(d)
            per_dir_species[d] = species_from_sra_dir(d)[0]
            dirs.append(d)
    if sheet.fastq_dir and sheet.fastq_dir not in per_dir_map:
        per_dir_map[sheet.fastq_dir] = species_by_basename(sheet.fastq_dir)
        per_dir_species[sheet.fastq_dir] = species_from_sra_dir(sheet.fastq_dir)[0]
        dirs.append(sheet.fastq_dir)

    found = []
    for smp in sheet.samples:
        d = str(Path(smp.fastq_r1).parent) if smp.fastq_r1 else ""
        bmap = per_dir_map.get(d, {})
        sp = bmap.get(smp.name) or ""
        if not sp:
            # basenames carry an _R1/_R2 or _2 suffix in some layouts
            for base, val in bmap.items():
                if smp.name.startswith(base) or base.startswith(smp.name):
                    sp = val; break
        if not sp:
            pool = per_dir_species.get(d) or []
            if len(set(pool)) == 1:
                sp = pool[0]
        if sp:
            smp.organism = sp
            found.append(sp)

    if not found:
        for d in dirs:
            found += per_dir_species.get(d) or []
    uniq = sorted({str(x) for x in found if str(x or "").strip()})
    if not uniq:
        return sheet
    if len(uniq) > 1:
        sheet.warnings.append(DetectWarning(
            severity="high",
            msg=("mixed species in this input: %s. MRAG aligns every sample to "
                 "ONE genome, so this set cannot be run as-is -- split it, one "
                 "organism per run." % ", ".join(uniq))))
        sheet.needs_user_action = True
        sheet.organism_source = "sra_metadata"
        sheet.organism_detail = "MIXED: " + ", ".join(uniq)
        return sheet
    try:
        prof = _resolve_org(uniq[0])
    except Exception as exc:
        sheet.warnings.append(DetectWarning(
            severity="high", msg="unsupported species %s: %s" % (uniq[0], exc)))
        sheet.needs_user_action = True
        sheet.organism_source = "sra_metadata"
        sheet.organism_detail = "unsupported: %s" % uniq[0]
        return sheet
    sheet.organism = prof.key
    sheet.organism_source = "sra_metadata"
    sheet.organism_detail = "%s (%s) from the study metadata beside the FASTQs" % (
        prof.scientific_name, prof.genome)
    return sheet


def detect_rule_based(
    fastq_paths: list[Path],
    fastq_dir: Optional[Path] = None,
) -> SampleSheet:
    """Pure rule-based detection. No LLM. Always safe to run.

    fastq_paths: list of *.fastq[.gz] (R1+R2 mixed, will be paired internally)
    fastq_dir:   optional context (used as dir hint for modality)
    """
    pairs = _pair_fastqs(fastq_paths)
    dir_hint = str(fastq_dir) if fastq_dir else ""

    samples: list[Sample] = []
    warnings: list[DetectWarning] = []
    for r1, r2 in pairs:
        base = _extract_sample_basename(r1.name)
        # Use the file's IMMEDIATE PARENT dir as modality hint (e.g. ATAC/ or RNA/)
        # in addition to the top-level dir_hint.
        per_file_hint = (r1.parent.name or "") + " " + dir_hint
        modality, mc = _classify_modality_rule(base, per_file_hint)
        arm, ac      = _classify_arm_rule(base)
        rep, rc      = _classify_replicate_rule(base)
        rl           = _peek_read_length(r1)
        samples.append(Sample(
            name=base, fastq_r1=str(r1), fastq_r2=str(r2) if r2 else None,
            modality=modality, arm=arm, replicate=rep,
            read_length=rl, paired=(r2 is not None),
            confidence=SampleConfidence(modality=mc, arm=ac, replicate=rc),
        ))

    # Roll up conditions
    cond_map: dict[str, dict] = {}
    for s in samples:
        if not s.arm: continue
        d = cond_map.setdefault(s.arm, {"n_reps":0,"modality_counts":{}})
        d["n_reps"] += 1
        d["modality_counts"][s.modality] = d["modality_counts"].get(s.modality, 0) + 1
    conditions = [Condition(label=k, n_reps=v["n_reps"],
                            modality_counts=v["modality_counts"])
                  for k, v in cond_map.items()]

    # Warnings
    n_low_arm = sum(1 for s in samples if s.confidence.arm == "low")
    if n_low_arm > 0:
        warnings.append(DetectWarning(
            severity="medium",
            msg=f"{n_low_arm}/{len(samples)} samples have low-confidence arm assignment; please verify."
        ))
    n_unknown_mod = sum(1 for s in samples if s.modality == "unknown")
    if n_unknown_mod > 0:
        warnings.append(DetectWarning(
            severity="medium",
            msg=f"{n_unknown_mod} samples have unknown modality (couldn't tell ATAC vs RNA from filename)."
        ))
    if len(conditions) != 2:
        warnings.append(DetectWarning(
            severity="high",
            msg=f"Expected exactly 2 conditions (arms) for MRAG comparison, got {len(conditions)}: {[c.label for c in conditions]}"
        ))
    needs_action = (n_low_arm > 0 or n_unknown_mod > 0 or len(conditions) != 2)

    summary = _build_human_summary(samples, conditions, warnings)
    sheet = SampleSheet(
        detected_at=datetime.now().isoformat(timespec="seconds"),
        fastq_dir=str(fastq_dir) if fastq_dir else None,
        samples=samples, conditions=conditions, warnings=warnings,
        needs_user_action=needs_action, summary_human=summary,
    )
    attach_organism(sheet)
    sheet.summary_human = _build_human_summary(
        sheet.samples, sheet.conditions, sheet.warnings)
    return sheet


def _organism_line(samples) -> str:
    """One line naming the species/genome, or nothing when it is unknown."""
    from .organism import resolve as _resolve_org
    sp = sorted({str(getattr(s, "organism", "") or "") for s in samples} - {""})
    if not sp:
        return ""
    if len(sp) > 1:
        return "Organism: MIXED (%s) - this set cannot be run as one job.\n" % ", ".join(sp)
    try:
        prof = _resolve_org(sp[0])
    except Exception:
        return "Organism: %s (unsupported by MRAG)\n" % sp[0]
    return "Organism: %s -> genome %s\n" % (prof.scientific_name, prof.genome)


def _build_human_summary(samples, conditions, warnings) -> str:
    n = len(samples)
    by_mod = {}
    for s in samples: by_mod[s.modality] = by_mod.get(s.modality, 0) + 1
    cond_str = ", ".join(f"{c.label} ({c.n_reps} reps)" for c in conditions) or "(no condition detected)"
    mod_str  = ", ".join(f"{k}: {v}" for k, v in sorted(by_mod.items()))
    org = _organism_line(samples).rstrip("\n")
    base = f"Detected {n} samples ({mod_str}). Conditions: {cond_str}."
    if org:
        base += " " + org + "."
    high_warn = sum(1 for w in warnings if w.severity == "high")
    med_warn  = sum(1 for w in warnings if w.severity == "medium")
    if high_warn: base += f" ⚠ {high_warn} HIGH severity warning(s) — needs attention."
    elif med_warn: base += f" ⚠ {med_warn} medium-severity warning(s)."
    else: base += " All confidence=high."
    return base


# ============================================================
# SRA / GEO metadata lookup (for accession-named samples)
# ------------------------------------------------------------
# Files named by SRA run accession (SRR/ERR/DRR + digits) carry no biological
# hint in the filename. We fetch the public NCBI SRA/GEO title + study name so
# the LLM arm grouping can classify the arm from real metadata rather than
# guessing from an opaque accession. Best-effort: failures fall back silently.
# ============================================================
_SRA_ACC_RE = re.compile(r"^(?:SRR|ERR|DRR)\d{5,}$", re.IGNORECASE)


def _ncbi_creds():
    """Shared NCBI credentials loader (libs/ncbi_creds.py); None when absent,
    in which case every caller keeps the anonymous 3 req/s behaviour."""
    try:
        import sys as _sys
        if "/BIO1/MRAG/Agent/libs" not in _sys.path:
            _sys.path.insert(0, "/BIO1/MRAG/Agent/libs")
        import ncbi_creds as _c
        return _c
    except Exception:
        return None


def _fetch_sra_metadata(name: str, timeout: float = 20.0) -> Optional[str]:
    """Return a short "GSM title | Study: ..." string for an SRA run accession,
    or None if the name is not an accession or the lookup fails."""
    token = name.split("_")[0].split(".")[0]
    if not _SRA_ACC_RE.match(token):
        return None
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    creds = _ncbi_creds()

    def _get(url: str, attempts: int = 3) -> Optional[str]:
        # NCBI allows 3 req/s anonymously, 10 req/s with the shared api_key
        # (ncbi_creds, 2026-09-20); retry with backoff. The key is attached
        # here only and never appears in a returned string.
        sent = creds.add_params(url) if creds is not None else url
        for i in range(attempts):
            if creds is not None:
                creds.SHARED.wait()
            try:
                with urllib.request.urlopen(sent, timeout=timeout) as r:
                    return r.read().decode("utf-8", "replace")
            except Exception:
                time.sleep(1.0 * (i + 1))
        return None

    xml = _get(f"{base}/esearch.fcgi?db=sra&term={token}&retmax=1")
    if not xml:
        return None
    m = re.search(r"<Id>(\d+)</Id>", xml)
    if not m:
        return None
    uid = m.group(1)
    if creds is None:
        time.sleep(0.4)          # NCBI: <=3 req/s without api key
    summ = _get(f"{base}/esummary.fcgi?db=sra&id={uid}")
    if not summ:
        return None
    exp = (summ.replace("&lt;", "<").replace("&gt;", ">")
                .replace("&quot;", '"').replace("&amp;", "&"))
    parts: list[str] = []
    t = re.search(r"<Title>([^<]*)</Title>", exp)
    if t:
        parts.append(t.group(1).strip())
    st = re.search(r'<Study[^>]*\bname="([^"]*)"', exp)
    if st:
        parts.append("Study: " + st.group(1).strip())
    meta = " | ".join(p for p in parts if p)
    return meta or None


# ============================================================
# LLM arm grouping
# ============================================================

def _llm_call(endpoint: str, model: str, prompt: str,
              timeout: float = 240.0, num_predict: int = 512) -> str:
    payload = {
        "model": model,
        "messages": [{"role":"user","content":prompt}],
        "stream": False, "think": False,
        "options": {"temperature": 0.0, "num_predict": num_predict, "num_ctx": 8192},
    }
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body.get("message", {}).get("content", "")


ARM_GROUP_PROMPT = """You are grouping NGS samples into EXPERIMENTAL ARMS for a differential study.

Two samples belong to the SAME arm only if every experimental variable matches
(drug, dose, duration, hormone, genotype). Samples differing in ANY treatment
variable are DIFFERENT arms. Replicate and lane numbers do not define an arm.

A donor, patient, lane or replicate identifier is NOT itself an arm. But a name
often carries BOTH an identifier and a real condition, e.g. P1_Tumor is patient P1
with condition Tumor. Strip the identifier and keep the condition. Return an empty
string ONLY when nothing but identifiers remains.

Never put the assay type in the label. ATAC-seq and RNA-seq are measurements, not
conditions: the same biological condition profiled by both assays is ONE arm, and
its ATAC samples and RNA samples must carry the identical label.

Give each distinct arm one short stable label, reused verbatim across its members.
Do not invent a condition that is not in the name.

Return STRICT JSON only, no prose and no markdown fences:
{"results":[{"id":"<id verbatim>","arm":"<label or empty string>"}]}

SAMPLES:
%s"""


_ASSAY_TOKEN_RE = re.compile(
    r"\b(?:sc|sn|bulk)?[\s_-]?(?:ATAC|RNA)(?:[\s_-]?seq)?\b"
    r"|\bDNase(?:[\s_-]?seq)?\b"
    r"|\btranscriptom\w*\b", re.I)


def _strip_assay(label: str) -> str:
    """Arm label with assay words and punctuation removed, for comparing arms."""
    return re.sub(r"[^A-Za-z0-9]+", "", _ASSAY_TOKEN_RE.sub(" ", label)).lower()


def _collapse_assay_arms(samples) -> list[str]:
    """Merge arms that differ only by assay type; return the merges performed.

    When ATAC and RNA files are named after the assay, the model labels
    "X ATAC" and "X RNA" as two arms. They are one condition measured two ways,
    and leaving them split gives every arm a single modality, which no
    differential test can use.
    """
    groups: dict[str, list] = {}
    for s in samples:
        if s.arm:
            groups.setdefault(_strip_assay(s.arm), []).append(s)
    merges = []
    for key, members in groups.items():
        labels = sorted({m.arm for m in members})
        if len(labels) < 2 or not key:
            continue
        canon = min((_ASSAY_TOKEN_RE.sub(" ", l) for l in labels), key=len)
        canon = re.sub(r"\s+", " ", canon).strip(" ,-_") or labels[0]
        for m in members:
            m.arm = canon
        merges.append("%s -> %r" % (" and ".join(repr(x) for x in labels), canon))
    return merges


def _rebuild_conditions(sheet: SampleSheet) -> SampleSheet:
    """Recompute conditions, warnings and the summary after arms changed."""
    cond_map: dict[str, dict] = {}
    for s in sheet.samples:
        if not s.arm:
            continue
        d = cond_map.setdefault(s.arm, {"n_reps": 0, "modality_counts": {}})
        d["n_reps"] += 1
        d["modality_counts"][s.modality] = d["modality_counts"].get(s.modality, 0) + 1
    sheet.conditions = [Condition(label=k, n_reps=v["n_reps"],
                                  modality_counts=v["modality_counts"])
                        for k, v in cond_map.items()]

    # Drop the stale structural warnings, then re-derive them from the new arms.
    keep = [w for w in sheet.warnings
            if "conditions (arms)" not in w.msg and "low-confidence arm" not in w.msg]
    n_unassigned = sum(1 for s in sheet.samples if not s.arm)
    if n_unassigned:
        keep.append(DetectWarning(
            severity="high",
            msg=f"{n_unassigned}/{len(sheet.samples)} sample(s) could not be assigned an arm."))
    if len(sheet.conditions) != 2:
        keep.append(DetectWarning(
            severity="high",
            msg=f"Expected exactly 2 conditions (arms) for MRAG comparison, got "
                f"{len(sheet.conditions)}: {[c.label for c in sheet.conditions]}"))
    # Each modality needs two arms to contrast, or no differential test can run.
    for mod in ("ATAC", "RNA"):
        n_arms = sum(1 for c in sheet.conditions if c.modality_counts.get(mod))
        n_samp = sum(1 for s in sheet.samples if s.modality == mod)
        if n_samp and n_arms < 2:
            keep.append(DetectWarning(
                severity="high",
                msg=f"Only {n_arms} arm carries {mod} data; a differential test needs two."))
    sheet.warnings = keep
    n_unknown_mod = sum(1 for s in sheet.samples if s.modality == "unknown")
    sheet.needs_user_action = bool(
        n_unassigned or n_unknown_mod or len(sheet.conditions) != 2
        or any(w.severity == "high" for w in keep))
    sheet.summary_human = _build_human_summary(
        sheet.samples, sheet.conditions, sheet.warnings)
    return sheet


def _adopt_rule_labels(samples, rule_arms: dict) -> list[str]:
    """Keep the canonical rule label wherever it maps 1:1 onto an LLM group.

    The two layers are good at different halves of the job. The model decides the
    partition, which the keyword tables get wrong whenever two conditions share a
    drug. The tables decide the name, mapping OC2_OE to ONECUT2-OE and cMyc_DDK to
    MYC-DDK, and those canonical labels flow into every downstream figure. Adopt a
    rule label only when it covers exactly one LLM group, so a label the rules had
    merged never pulls two groups back together.
    """
    groups: dict[str, list] = {}
    for s in samples:
        if s.arm:
            groups.setdefault(s.arm, []).append(s)
    # rule label -> the LLM groups it appears in
    spread: dict[str, set] = {}
    for llm_arm, members in groups.items():
        for m in members:
            r = rule_arms.get(m.name, "")
            if r:
                spread.setdefault(r, set()).add(llm_arm)
    renamed = []
    for llm_arm, members in groups.items():
        rules_here = {rule_arms.get(m.name, "") for m in members}
        rules_here.discard("")
        if len(rules_here) != 1:
            continue
        rule_label = rules_here.pop()
        if len(spread.get(rule_label, ())) != 1:
            continue  # the rules had merged this group with another
        if rule_label == llm_arm:
            continue
        for m in members:
            m.arm = rule_label
        renamed.append("%r -> %r" % (llm_arm, rule_label))
    return renamed


def refine_arms_with_llm(
    sheet: SampleSheet,
    endpoint: str = "http://127.0.0.1:11500",
    model: str = "batiai/qwen3.6-27b:q6",
) -> SampleSheet:
    """Re-derive every arm with the LLM, keeping the rule result as a fallback.

    The older assist only sent samples the rules left empty, so a rule that
    merged two arms was never revisited: TGFb_2h and TGFb_24h both matched the
    TGFB keyword at medium confidence and silently became one condition. The
    keyword tables also only know a prostate vocabulary, and "_" is a word
    character, so \\bcontrol\\b never fires on control_rep1. Asking the model to
    GROUP rather than to label is what keeps replicates together and time points
    apart; the rules stay as the offline fallback.
    """
    targets = [s for s in sheet.samples]
    rule_arms = {s.name: s.arm for s in targets}
    if not targets:
        return sheet

    payload = []
    n_meta = 0
    for s in targets:
        entry = {"id": s.name, "name": s.name}
        meta = _fetch_sra_metadata(s.name)
        if meta:
            entry["sra_metadata"] = meta
            n_meta += 1
        payload.append(entry)
    if n_meta:
        sheet.warnings.append(DetectWarning(
            severity="low",
            msg=f"Fetched NCBI SRA/GEO metadata for {n_meta}/{len(targets)} accession-named sample(s)."))

    raw = _llm_call(endpoint, model,
                    ARM_GROUP_PROMPT % json.dumps(payload, indent=1),
                    num_predict=2048)
    try:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        parsed = json.loads(cleaned[cleaned.index("{"):cleaned.rindex("}") + 1])["results"]
        by_id = {str(d.get("id")): (d.get("arm") or "").strip() for d in parsed}
    except Exception as e:  # noqa: BLE001 - rules still stand
        sheet.warnings.append(DetectWarning(
            severity="medium",
            msg=f"LLM arm grouping failed ({type(e).__name__}); keeping rule-based arms."))
        sheet.llm_assist_used = True
        sheet.llm_assist_model = model
        return sheet

    changed = 0
    for s in targets:
        if s.name in by_id:
            new = by_id[s.name]
            if new != s.arm:
                changed += 1
            s.arm = new
            s.confidence.arm = "high" if new else "low"
    merges = _collapse_assay_arms(targets)
    renamed = _adopt_rule_labels(targets, rule_arms)
    for r in renamed:
        sheet.warnings.append(DetectWarning(
            severity="low",
            msg=f"Kept the canonical rule label: {r}"))
    for m in merges:
        sheet.warnings.append(DetectWarning(
            severity="low",
            msg=f"Merged arms that differ only by assay type: {m}"))
    if changed:
        sheet.warnings.append(DetectWarning(
            severity="low",
            msg=f"LLM regrouping changed {changed}/{len(targets)} arm assignment(s)."))
    sheet.llm_assist_used = True
    sheet.llm_assist_model = model
    return _rebuild_conditions(sheet)


# Convenience top-level
def detect(fastq_paths: list[Path], fastq_dir: Optional[Path] = None,
           use_llm: bool = True, endpoint: str = "http://127.0.0.1:11500",
           model: str = "batiai/qwen3.6-27b:q6") -> SampleSheet:
    """Rule-based detection, then LLM regrouping of the arms.

    Regrouping is on by default because the keyword tables are prostate-specific
    and merge conditions that share a drug: on a TGFb 2h vs 24h vs control set
    the rules returned one condition of eight replicates. The rules remain the
    fallback whenever the model is unreachable or answers unparseably.
    """
    sheet = detect_rule_based(fastq_paths, fastq_dir)
    if not use_llm:
        return sheet
    try:
        return refine_arms_with_llm(sheet, endpoint=endpoint, model=model)
    except Exception as e:  # noqa: BLE001 - never fail detection over the LLM
        sheet.warnings.append(DetectWarning(
            severity="medium",
            msg=f"LLM unreachable ({type(e).__name__}); using rule-based arms."))
        return sheet
