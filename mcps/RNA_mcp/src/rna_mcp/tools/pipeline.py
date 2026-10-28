"""Pipeline tools — RNA-seq shell steps + DE analysis + visualisation."""
from __future__ import annotations

import fcntl
import glob as g
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mcp.server.fastmcp import FastMCP
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats


# ══════════════════════════════════════════════════════════════════════
# FASTQ discovery — accepts multiple vendor naming conventions
# ══════════════════════════════════════════════════════════════════════
_PE_SUFFIX_PAIRS = [
    ("_1.fastq.gz", "_2.fastq.gz"),
    ("_1.fq.gz",    "_2.fq.gz"),
    ("_R1.fastq.gz", "_R2.fastq.gz"),
    ("_R1.fq.gz",    "_R2.fq.gz"),
    ("_R1_001.fastq.gz", "_R2_001.fastq.gz"),
]
_SE_SUFFIXES = [".fastq.gz", ".fq.gz"]


def _find_fastqs(project_dir: str, sample: str) -> tuple[str, str | None, bool]:
    """Locate the raw FASTQ(s) for a sample.

    Returns (r1_path, r2_path_or_None, paired_bool). Raises FileNotFoundError
    with a helpful list of tried patterns on miss.
    """
    d = Path(project_dir)
    tried: list[str] = []
    for s1, s2 in _PE_SUFFIX_PAIRS:
        r1 = d / f"{sample}{s1}"
        r2 = d / f"{sample}{s2}"
        tried += [str(r1), str(r2)]
        if r1.exists() and r2.exists():
            return str(r1), str(r2), True
    for suf in _SE_SUFFIXES:
        se = d / f"{sample}{suf}"
        tried.append(str(se))
        if se.exists():
            return str(se), None, False
    # Glob fallback for lane-split Illumina files
    g_r1 = sorted(d.glob(f"{sample}_*R1*.fastq.gz"))
    if g_r1:
        r1 = g_r1[0]
        r2_name = r1.name.replace("R1", "R2", 1)
        r2 = d / r2_name
        if r2.exists():
            return str(r1), str(r2), True
        return str(r1), None, False
    raise FileNotFoundError(
        f"No FASTQ found for sample '{sample}' in {project_dir}. Tried: "
        + ", ".join(tried)
    )


def _canonicalize_inputs(
    project_dir: str, sample: str, r1: str, r2: str | None, paired: bool
) -> tuple[str, str | None]:
    """Symlink non-canonical FASTQ names to {sample}_1/_2 (PE) or {sample}.fastq.gz (SE).

    Keeps downstream step2/3 unaware of the original naming convention.
    Returns the canonical (r1, r2_or_None) paths used for trim_galore input.
    """
    d = Path(project_dir)
    if paired:
        c1 = d / f"{sample}_1.fastq.gz"
        c2 = d / f"{sample}_2.fastq.gz"
        if c1.resolve() != Path(r1).resolve():
            c1.unlink(missing_ok=True); c1.symlink_to(r1)
        if c2.resolve() != Path(r2).resolve():
            c2.unlink(missing_ok=True); c2.symlink_to(r2)
        return str(c1), str(c2)
    else:
        c = d / f"{sample}.fastq.gz"
        if c.resolve() != Path(r1).resolve():
            c.unlink(missing_ok=True); c.symlink_to(r1)
        return str(c), None


def _find_trimmed(project_dir: str, sample: str) -> tuple[str, str | None, bool]:
    """Locate trim_galore outputs. Raises if step1 has not produced anything."""
    d = Path(project_dir) / "trim_galore"
    pe_r1 = d / f"{sample}_1_val_1.fq.gz"
    pe_r2 = d / f"{sample}_2_val_2.fq.gz"
    if pe_r1.exists() and pe_r2.exists():
        return str(pe_r1), str(pe_r2), True
    se = d / f"{sample}_trimmed.fq.gz"
    if se.exists():
        return str(se), None, False
    raise FileNotFoundError(
        f"Trimmed FASTQ not found for '{sample}'. Run step1 first "
        f"(looked in {d})."
    )


def _list_samples(project_dir: str) -> list[str]:
    """Scan project_dir for FASTQ files and return unique sample prefixes
    across PE and SE naming conventions."""
    import re
    d = Path(project_dir)
    names: set[str] = set()
    for fq in sorted(d.glob("*.fastq.gz")) + sorted(d.glob("*.fq.gz")):
        base = fq.name
        m = re.match(
            r"^(.+?)(?:_R?1(?:_001)?|_1)\.(?:fastq|fq)\.gz$", base
        )
        if m:
            names.add(m.group(1)); continue
        # SE: strip extension only if no _2 counterpart implied
        m2 = re.match(r"^(.+?)\.(?:fastq|fq)\.gz$", base)
        if m2 and "_R2" not in base and "_2." not in base:
            # Accept as SE candidate only if no matching R1 already captured
            names.add(m2.group(1))
    return sorted(names)


def _resolve_input_path(executor, path: str) -> Path | None:
    """Resolve optional user-supplied file/dir paths under the configured work_dir."""
    if not path:
        return None
    return executor._resolve_project_dir(path)


def _bam_quickcheck(executor, bam_path: str) -> str | None:
    """Run `samtools quickcheck` on a BAM; return None if OK, else error string."""
    p = Path(bam_path)
    if not p.exists():
        return f"BAM missing: {bam_path}"
    if p.stat().st_size < 1024:
        return f"BAM too small ({p.stat().st_size} bytes), likely truncated: {bam_path}"
    out = executor.run(f"samtools quickcheck {executor.quote(bam_path)}; echo $?")
    # quickcheck prints nothing on success; non-zero status => corrupt
    if "[EXIT CODE]" in out or "[ERROR]" in out or "[TIMEOUT]" in out:
        return f"BAM failed samtools quickcheck: {bam_path}"
    return None


def _rsem_genes_total_counts(genes_results: str) -> float:
    """Sum of expected_count column in RSEM .genes.results. 0 = empty/failed run."""
    p = Path(genes_results)
    if not p.exists():
        return 0.0
    total = 0.0
    with p.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            idx = header.index("expected_count")
        except ValueError:
            return 0.0
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) > idx:
                try:
                    total += float(cols[idx])
                except ValueError:
                    pass
    return total


def _validate_deseq_inputs(
    counts_columns: list[str], samples: pd.DataFrame, reference: str = "untreated"
) -> str | None:
    """Return a user-facing error message when the DE design is invalid."""
    required_cols = {"sample", "TREATMENT"}
    if not required_cols <= set(samples.columns):
        return "mapfile must have columns 'sample' and 'TREATMENT'"

    if samples["sample"].duplicated().any():
        dup = samples.loc[samples["sample"].duplicated(), "sample"].tolist()
        return f"mapfile contains duplicate sample rows: {dup}"

    sample_names = samples["sample"].tolist()
    missing = [name for name in sample_names if name not in counts_columns]
    if missing:
        return f"Samples missing from counts matrix: {missing}"

    groups = set(samples["TREATMENT"].dropna().str.lower())
    if reference not in groups:
        return f"Reference condition '{reference}' not found in mapfile"
    if "treated" not in groups:
        return "mapfile must include at least one 'treated' sample"
    unknown = sorted(groups - {"treated", reference})
    if unknown:
        return f"Unsupported TREATMENT values for current contrast: {unknown}"
    return None


# ══════════════════════════════════════════════════════════════════════
# Core helpers (re-usable outside MCP context as well)
# ══════════════════════════════════════════════════════════════════════
def _import_rsem(
    samples: pd.DataFrame,
    rsem_dir: Path,
    suffix: str = ".genes.results",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine RSEM .genes.results files into gene × sample matrices."""
    counts_list, tpm_list = [], []
    for sample_id in samples["sample"]:
        path = rsem_dir / f"{sample_id}{suffix}"
        if not path.exists():
            raise FileNotFoundError(f"RSEM output not found: {path}")
        df = pd.read_csv(path, sep="\t")
        counts_list.append(df.set_index("gene_id")["expected_count"].rename(sample_id))
        tpm_list.append(df.set_index("gene_id")["TPM"].rename(sample_id))
    return pd.concat(counts_list, axis=1), pd.concat(tpm_list, axis=1)


def _run_deseq2(
    counts: pd.DataFrame,
    samples: pd.DataFrame,
    reference: str = "untreated",
    n_cpus: int = 8,
) -> tuple[pd.DataFrame, DeseqDataSet]:
    """Run pydeseq2 and return (results_df, fitted dds)."""
    counts_t = counts.round().astype(int).T
    metadata = samples.set_index("sample").loc[counts_t.index, ["TREATMENT"]]

    inference = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(
        counts=counts_t, metadata=metadata,
        design_factors="TREATMENT",
        ref_level=["TREATMENT", reference],
        inference=inference,
    )
    dds.deseq2()
    ds = DeseqStats(dds, contrast=["TREATMENT", "treated", reference], inference=inference)
    ds.summary()
    results = ds.results_df.sort_values("padj", na_position="last")
    return results, dds


def _plot_ma(results: pd.DataFrame, out: Path, padj_cutoff: float = 0.05) -> None:
    df = results.dropna(subset=["baseMean", "log2FoldChange"]).copy()
    df = df[df["baseMean"] > 0]  # log scale requires positive values
    if df.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No positive baseMean values", ha="center", va="center",
                transform=ax.transAxes)
        fig.savefig(out, dpi=150); plt.close(fig); return
    sig = df["padj"] < padj_cutoff
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df.loc[~sig, "baseMean"], df.loc[~sig, "log2FoldChange"],
               s=4, c="lightgray", alpha=0.6, label=f"n.s. ({(~sig).sum()})")
    ax.scatter(df.loc[sig, "baseMean"], df.loc[sig, "log2FoldChange"],
               s=6, c="#d62728", alpha=0.8, label=f"padj<{padj_cutoff} ({sig.sum()})")
    ax.set_xscale("log"); ax.axhline(0, c="k", lw=0.5)
    ax.set_xlabel("mean of normalized counts"); ax.set_ylabel("log2 fold change")
    ax.set_title("MA plot"); ax.legend(loc="best", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def _plot_volcano(results: pd.DataFrame, out: Path,
                  padj_cutoff: float = 0.10, lfc_cutoff: float = 1.0) -> None:
    df = results.dropna(subset=["log2FoldChange", "padj"]).copy()
    df["-log10padj"] = -np.log10(df["padj"].clip(lower=1e-300))
    cat = np.where((df["padj"] < padj_cutoff) & (df["log2FoldChange"] >= lfc_cutoff), "up",
          np.where((df["padj"] < padj_cutoff) & (df["log2FoldChange"] <= -lfc_cutoff), "down", "ns"))
    colors = {"up": "#d62728", "down": "#1f77b4", "ns": "#cccccc"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in ["ns", "up", "down"]:
        m = cat == c
        ax.scatter(df.loc[m, "log2FoldChange"], df.loc[m, "-log10padj"],
                   s=6, c=colors[c], alpha=0.6, label=f"{c} ({m.sum()})")
    ax.axhline(-np.log10(padj_cutoff), ls="--", c="k", lw=0.5)
    ax.axvline(-lfc_cutoff, ls="--", c="k", lw=0.5)
    ax.axvline(lfc_cutoff, ls="--", c="k", lw=0.5)
    ax.set_xlabel("log2 fold change"); ax.set_ylabel("-log10 padj")
    ax.set_title("Volcano plot"); ax.legend(loc="best", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def _plot_heatmap(
    results: pd.DataFrame,
    samples: pd.DataFrame,
    out: Path,
    dds: DeseqDataSet | None = None,
    norm: pd.DataFrame | None = None,
    top_n: int = 50,
    padj_cutoff: float = 0.10,
    lfc_cutoff: float = 1.0,
    annotation: pd.DataFrame | None = None,
    normalize: str = "zscore",
    col_cluster: bool = False,
) -> pd.DataFrame:
    """Z-score heatmap by default; normalize='none' uses raw log2 counts.

    Pass either `dds` (fitted DeseqDataSet) or `norm` (genes × samples DataFrame).
    Returns an empty DataFrame if no significant genes (no plot written).
    """
    sig = results[
        results["padj"].notna()
        & (results["padj"] < padj_cutoff)
        & (results["log2FoldChange"].abs() >= lfc_cutoff)
    ].sort_values("padj").head(top_n)
    if sig.empty:
        return sig

    if norm is None:
        if dds is None:
            raise ValueError("_plot_heatmap requires either `norm` or `dds`")
        norm = pd.DataFrame(
            dds.layers["normed_counts"],
            index=dds.obs_names, columns=dds.var_names,
        ).T  # -> genes × samples

    # Restrict to significant genes that exist in the matrix
    present = [g for g in sig.index if g in norm.index]
    if not present:
        return sig.iloc[0:0]
    mat = np.log2(norm.loc[present] + 1)  # genes × samples

    if normalize == "zscore":
        mu = mat.mean(axis=1); sd = mat.std(axis=1).replace(0, 1)
        mat = mat.sub(mu, axis=0).div(sd, axis=0)
        cmap, center, vmin, vmax = "vlag", 0, -2, 2
        cbar_label = "z-score (row)"
    else:
        cmap, center, vmin, vmax = "GnBu", None, None, None
        cbar_label = "log2(norm count + 1)"

    if annotation is not None and {"Gene.ID", "Gene.Name"}.issubset(annotation.columns):
        id_to_name = annotation.set_index("Gene.ID")["Gene.Name"].to_dict()
        mat.index = [id_to_name.get(g, g) for g in mat.index]

    col_colors = samples.set_index("sample")["TREATMENT"].map(
        {"treated": "#d62728", "untreated": "#1f77b4"}
    ).reindex(mat.columns)

    figsize = (max(5, mat.shape[1] * 0.5), max(5, min(16, mat.shape[0] * 0.18)))
    # Order columns by group (untreated then treated) when not clustering
    if not col_cluster:
        order = (samples.sort_values("TREATMENT")["sample"]
                 .loc[lambda s: s.isin(mat.columns)].tolist())
        mat = mat[order]
        col_colors = col_colors.reindex(mat.columns)

    g = sns.clustermap(
        mat, cmap=cmap, center=center, vmin=vmin, vmax=vmax,
        col_colors=col_colors, yticklabels=len(mat) <= 60, xticklabels=True,
        row_cluster=True, col_cluster=col_cluster, figsize=figsize,
        cbar_kws={"label": cbar_label}, dendrogram_ratio=(0.1, 0.08),
    )
    g.ax_heatmap.set_xlabel("")
    g.fig.suptitle(f"Top {len(mat)} significant genes ({normalize})", y=1.01)
    g.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    return sig


# ══════════════════════════════════════════════════════════════════════
# R plotting backend (publication-quality plots matching the reference analysis scripts)
# ══════════════════════════════════════════════════════════════════════
import subprocess as _subproc
import tempfile as _tempfile
import shutil as _shutil


def _run_rscript(r_code: str, cwd: str | None = None) -> tuple[int, str, str]:
    """Run R code via Rscript. Returns (returncode, stdout, stderr)."""
    rscript = _shutil.which("Rscript") or "/usr/bin/Rscript"
    with _tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as fh:
        fh.write(r_code)
        path = fh.name
    try:
        proc = _subproc.run([rscript, path], cwd=cwd, capture_output=True, text=True)
        return proc.returncode, proc.stdout, proc.stderr
    finally:
        Path(path).unlink(missing_ok=True)


def _plot_ma_r(results_csv: Path, out: Path, padj_cutoff: float = 0.05) -> None:
    """MA plot in R style — DESeq2::plotMA-equivalent on a results CSV."""
    r_code = f"""
suppressPackageStartupMessages({{ library(DESeq2) }})
res <- read.csv({_r_str(str(results_csv))}, row.names=1)
df <- res[, c("baseMean","log2FoldChange","padj")]
df$padj[is.na(df$padj)] <- 1
isDE <- df$padj < {padj_cutoff} & !is.na(df$padj)
png({_r_str(str(out))}, width=900, height=650, res=130)
par(mar=c(4.5,4.5,2.5,1))
plot(df$baseMean, df$log2FoldChange, log="x", pch=20, cex=0.45,
     col=ifelse(isDE, "#d62728", "grey60"),
     xlab="mean of normalized counts", ylab="log2 fold change",
     main=paste0("MA plot (padj<", {padj_cutoff}, ")"),
     ylim=c(-max(abs(df$log2FoldChange[is.finite(df$log2FoldChange)]), na.rm=TRUE),
             max(abs(df$log2FoldChange[is.finite(df$log2FoldChange)]), na.rm=TRUE)))
abline(h=0, col="black", lwd=0.6)
legend("topright", legend=c(paste0("padj<", {padj_cutoff}), "n.s."),
       col=c("#d62728","grey60"), pch=20, bty="n")
invisible(dev.off())
"""
    rc, _, err = _run_rscript(r_code)
    if rc != 0:
        raise RuntimeError(f"R MA plot failed: {err}")


def _plot_volcano_r(results_csv: Path, out: Path,
                    padj_cutoff: float = 0.01, lfc_cutoff: float = 2.0) -> None:
    """Volcano plot in the reference-analysis style: black/blue/red layered."""
    r_code = f"""
res <- read.csv({_r_str(str(results_csv))}, row.names=1)
res <- res[!is.na(res$pvalue), ]
png({_r_str(str(out))}, width=900, height=700, res=130)
par(mar=c(4.5,4.5,2.5,1))
with(res, plot(log2FoldChange, -log10(pvalue), pch=20, cex=0.5, col="grey50",
               main=paste0("Volcano plot (padj<", {padj_cutoff}, ", |LFC|>", {lfc_cutoff}, ")"),
               xlab="log2 fold change", ylab="-log10 p-value"))
with(subset(res, padj<{padj_cutoff}),
     points(log2FoldChange, -log10(pvalue), pch=20, cex=0.6, col="#1f77b4"))
with(subset(res, padj<{padj_cutoff} & abs(log2FoldChange)>{lfc_cutoff}),
     points(log2FoldChange, -log10(pvalue), pch=20, cex=0.7, col="#d62728"))
abline(v=c(-{lfc_cutoff}, {lfc_cutoff}), lty=2, col="black", lwd=0.5)
legend("topright",
       legend=c(paste0("padj<", {padj_cutoff}, " & |LFC|>", {lfc_cutoff}),
                paste0("padj<", {padj_cutoff}), "n.s."),
       col=c("#d62728","#1f77b4","grey50"), pch=20, bty="n")
invisible(dev.off())
"""
    rc, _, err = _run_rscript(r_code)
    if rc != 0:
        raise RuntimeError(f"R volcano plot failed: {err}")


def _plot_heatmap_r(results_csv: Path, norm_tsv: Path, mapfile: Path, out: Path,
                    annotation_tsv: Path | None = None, top_n: int = 50,
                    padj_cutoff: float = 0.10, lfc_cutoff: float = 1.0) -> int:
    """Heatmap in the reference-analysis style: gplots::heatmap.2 + RColorBrewer GnBu, scale=row, Colv=FALSE."""
    annot_arg = _r_str(str(annotation_tsv)) if annotation_tsv else "NULL"
    r_code = f"""
suppressPackageStartupMessages({{ library(RColorBrewer); library(gplots) }})
res <- read.csv({_r_str(str(results_csv))}, row.names=1)
norm <- read.table({_r_str(str(norm_tsv))}, header=TRUE, row.names=1, sep="\\t",
                   check.names=FALSE)
samples <- read.table({_r_str(str(mapfile))}, header=TRUE, sep="\\t",
                      stringsAsFactors=FALSE)
samples$TREATMENT <- tolower(samples$TREATMENT)
# Order columns: untreated then treated (reference-analysis style: Colv=FALSE)
ord <- c(samples$sample[samples$TREATMENT=="untreated"],
         samples$sample[samples$TREATMENT=="treated"])
ord <- ord[ord %in% colnames(norm)]
norm <- norm[, ord, drop=FALSE]

sig <- res[!is.na(res$padj) & res$padj < {padj_cutoff} &
           abs(res$log2FoldChange) >= {lfc_cutoff}, ]
sig <- sig[order(sig$padj), ]
top <- head(rownames(sig), {top_n})
top <- top[top %in% rownames(norm)]
if (length(top) == 0) {{ cat("0\\n"); quit(status=0) }}
mat <- log2(as.matrix(norm[top, , drop=FALSE]) + 1)

# Symbol annotation
ann_path <- {annot_arg}
if (!is.null(ann_path) && file.exists(ann_path)) {{
  ann <- read.table(ann_path, header=TRUE, sep="\\t", stringsAsFactors=FALSE,
                    quote="", comment.char="")
  if (all(c("Gene.ID","Gene.Name") %in% colnames(ann))) {{
    id2sym <- setNames(ann$Gene.Name, ann$Gene.ID)
    new_rn <- id2sym[rownames(mat)]
    new_rn[is.na(new_rn) | new_rn == ""] <- rownames(mat)[is.na(new_rn) | new_rn == ""]
    rownames(mat) <- new_rn
  }}
}}

hmcol <- colorRampPalette(brewer.pal(9, "GnBu"))(100)
treat_colors <- ifelse(samples$TREATMENT[match(colnames(mat), samples$sample)] == "treated",
                       "#d62728", "#1f77b4")

png({_r_str(str(out))}, width=1000, height=max(900, 18*length(top)+200), res=130)
heatmap.2(mat, col=hmcol, scale="row", Rowv=TRUE, Colv=FALSE,
          dendrogram="row", trace="none",
          margins=c(9, 10), cexRow=0.6, cexCol=1.0,
          ColSideColors=treat_colors, keysize=1.2, key.title=NA,
          key.xlab="z-score (row)",
          main=sprintf("Top %d significant genes (padj<{padj_cutoff}, |LFC|>={lfc_cutoff})",
                       length(top)))
legend("topright", legend=c("treated","untreated"),
       fill=c("#d62728","#1f77b4"), border=NA, bty="n", cex=0.8, inset=c(0,0))
invisible(dev.off())
cat(length(top), "\\n")
"""
    rc, stdout, err = _run_rscript(r_code)
    if rc != 0:
        raise RuntimeError(f"R heatmap failed: {err}")
    try:
        return int(stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return 0


def _r_str(s: str) -> str:
    """Quote a string for safe embedding in R code."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _run_deseq2_r(
    counts_file: Path,
    mapfile: Path,
    out_results: Path,
    out_norm: Path,
    reference: str = "untreated",
) -> dict:
    """Run R DESeq2 with the reference analysis scripts' defaults (independent filtering ON).

    Output format matches pydeseq2 columns: baseMean, log2FoldChange, lfcSE,
    stat, pvalue, padj — so downstream step6/7/8 plots work unchanged.
    """
    r_code = f"""
suppressPackageStartupMessages({{ library(DESeq2) }})
counts <- read.table({_r_str(str(counts_file))}, sep="\\t",
                    header=TRUE, row.names=1, check.names=FALSE)
samples <- read.table({_r_str(str(mapfile))}, header=TRUE, sep="\\t",
                     stringsAsFactors=FALSE)
samples$TREATMENT <- tolower(samples$TREATMENT)

# Reorder counts columns to match mapfile, drop any unmatched
common <- intersect(samples$sample, colnames(counts))
samples <- samples[match(common, samples$sample), ]
counts  <- counts[, common, drop=FALSE]
counts  <- round(as.matrix(counts))
mode(counts) <- "integer"

# Reference level first
samples$TREATMENT <- factor(samples$TREATMENT,
                            levels=unique(c({_r_str(reference.lower())},
                                            sort(unique(samples$TREATMENT)))))
samples$TREATMENT <- droplevels(samples$TREATMENT)
coldata <- data.frame(row.names=samples$sample, TREATMENT=samples$TREATMENT)

dds <- DESeqDataSetFromMatrix(countData=counts, colData=coldata,
                              design=~TREATMENT)
dds <- DESeq(dds, quiet=TRUE)
res <- results(dds, contrast=c("TREATMENT", "treated", {_r_str(reference.lower())}))
# Independent filtering applied by default; reference R DESeq2 behavior
res <- res[order(res$padj, na.last=TRUE), ]

resdf <- as.data.frame(res)
resdf <- cbind(gene_id=rownames(resdf), resdf)
write.table(resdf, {_r_str(str(out_results))}, sep=",", quote=FALSE,
            row.names=FALSE, col.names=TRUE)

# Save normalized counts (genes x samples)
norm <- counts(dds, normalized=TRUE)
normdf <- cbind(gene_id=rownames(norm), as.data.frame(norm))
write.table(normdf, {_r_str(str(out_norm))}, sep="\\t", quote=FALSE,
            row.names=FALSE, col.names=TRUE)

# Print summary as JSON-ish lines for caller
total <- nrow(resdf)
tested <- sum(!is.na(resdf$padj))
sig <- sum(!is.na(resdf$padj) & resdf$padj < 0.10 &
           abs(resdf$log2FoldChange) >= 1)
cat(sprintf("TOTAL=%d\\nTESTED=%d\\nSIGNIFICANT=%d\\n", total, tested, sig))
"""
    rc, stdout, err = _run_rscript(r_code)
    if rc != 0:
        raise RuntimeError(f"R DESeq2 failed: {err}")
    info = {"total_genes": 0, "tested_genes": 0, "significant": 0}
    for line in stdout.splitlines():
        if line.startswith("TOTAL="):
            info["total_genes"] = int(line.split("=", 1)[1])
        elif line.startswith("TESTED="):
            info["tested_genes"] = int(line.split("=", 1)[1])
        elif line.startswith("SIGNIFICANT="):
            info["significant"] = int(line.split("=", 1)[1])
    return info


# ══════════════════════════════════════════════════════════════════════
# MCP tool registration
# ══════════════════════════════════════════════════════════════════════
def register_pipeline_tools(mcp: FastMCP, executor):
    q = executor.quote

    # ── Step 1: trim_galore + fastqc ─────────────────────────────
    @mcp.tool()
    def step1_preprocessing(
        project_dir: str,
        sample: str,
        num_threads: int = 0,
        trim_galore_bin: str = "trim_galore",
    ) -> str:
        """Step 1: Trim adapters with trim_galore and run FastQC.

        Auto-detects PE/SE and naming convention (`_1/_2`, `_R1/_R2`,
        `_R1_001/_R2_001`, bare `.fastq.gz`). Non-canonical inputs are
        symlinked to `{sample}_1/_2.fastq.gz` (PE) or `{sample}.fastq.gz`
        (SE) so downstream steps see canonical names.

        Args:
            project_dir:    Absolute path containing raw FASTQs.
            sample:         Sample prefix.
            num_threads:    CPU threads. 0 = auto-detect.
            trim_galore_bin: Path to trim_galore executable.
        """
        safe_dir = str(executor._resolve_project_dir(project_dir))
        if executor.is_done(safe_dir, sample, "step1"):
            return f"[SKIP] step1 already completed for {sample}"
        try:
            r1, r2, paired = _find_fastqs(safe_dir, sample)
        except FileNotFoundError as e:
            return f"[ERROR] {e}"
        _canonicalize_inputs(safe_dir, sample, r1, r2, paired)

        res = executor.detect_resources()
        t = num_threads or res["cores"]
        s = q(sample); d = q(safe_dir); tg = q(trim_galore_bin)
        mode = "paired" if paired else "single"
        header = f"""
set -euo pipefail
cd {d}
mkdir -p trim_galore fastqc
echo '[step1] mode={mode} sample={sample}'
"""
        if paired:
            body = f"""
{tg} --paired --gzip -o trim_galore/ {s}_1.fastq.gz {s}_2.fastq.gz
fastqc -t {t} --nogroup -o fastqc/ trim_galore/{s}_1_val_1.fq.gz
fastqc -t {t} --nogroup -o fastqc/ trim_galore/{s}_2_val_2.fq.gz
"""
        else:
            body = f"""
{tg} --gzip -o trim_galore/ {s}.fastq.gz
fastqc -t {t} --nogroup -o fastqc/ trim_galore/{s}_trimmed.fq.gz
"""
        out = executor.run(header + body, cwd=safe_dir)
        if executor.is_success(out):
            executor.mark_done(safe_dir, sample, "step1")
        return out

    # ── Step 2: STAR alignment ───────────────────────────────────
    @mcp.tool()
    def step2_star(
        project_dir: str,
        sample: str,
        ref_dir: str = "",
        num_threads: int = 0,
        star_bin: str = "STAR",
    ) -> str:
        """Step 2: STAR alignment → TranscriptomeSAM.

        Args:
            project_dir: Project directory with trim_galore/ outputs.
            sample:      Sample prefix.
            ref_dir:     STAR genomeDir. Default: /BIO1/MRAG/Agent/Reference/RNA_hg38/.
            num_threads: CPU threads. 0 = auto-detect.
            star_bin:    Path to STAR executable.
        """
        safe_dir = str(executor._resolve_project_dir(project_dir))
        if executor.is_done(safe_dir, sample, "step2"):
            return f"[SKIP] step2 already completed for {sample}"
        try:
            tr1, tr2, paired = _find_trimmed(safe_dir, sample)
        except FileNotFoundError as e:
            return f"[ERROR] {e}"

        res = executor.detect_resources()
        t = num_threads or res["cores"]
        rd = ref_dir or "/BIO1/MRAG/Agent/Reference/RNA_hg38/"
        s = q(sample); d = q(safe_dir)
        # Relative path from project_dir (trim_galore/XYZ) for cleaner command
        rel_r1 = "trim_galore/" + Path(tr1).name
        rel_r2 = "trim_galore/" + Path(tr2).name if paired else None
        reads_in = f"{rel_r1} {rel_r2}" if paired else rel_r1
        cmds = f"""
set -euo pipefail
cd {d}
echo '[step2] mode={"paired" if paired else "single"} sample={sample}'
# Resilience: STAR refuses to run if outTmpDir exists. Remove stale temp + partial outputs.
rm -rf TEMP_{s}
rm -f {s}Aligned.out.sam {s}Aligned.toTranscriptome.out.bam \
      {s}Log.final.out {s}Log.out {s}Log.progress.out {s}SJ.out.tab
{q(star_bin)} --runThreadN {t} --genomeDir {q(rd)} \
  --readFilesIn {reads_in} \
  --readFilesCommand zcat --outTmpDir TEMP_{s} \
  --outFileNamePrefix {s} --quantMode TranscriptomeSAM
rm -f {s}Aligned.out.sam
"""
        out = executor.run(cmds, cwd=safe_dir)
        if not executor.is_success(out):
            return out

        # Sanity: STAR must produce a complete transcriptome BAM and final log
        bam_out = f"{safe_dir}/{sample}Aligned.toTranscriptome.out.bam"
        star_log = f"{safe_dir}/{sample}Log.final.out"
        if not Path(star_log).exists():
            return out + f"\n[ERROR] STAR finished but {star_log} missing. Not marking step2 done."
        bam_err = _bam_quickcheck(executor, bam_out)
        if bam_err:
            return out + f"\n[ERROR] {bam_err}. Not marking step2 done."

        executor.mark_done(safe_dir, sample, "step2")
        return out

    # ── Step 3: RSEM quantification ──────────────────────────────
    @mcp.tool()
    def step3_rsem(
        project_dir: str,
        sample: str,
        ref_dir: str = "",
        genome: str = "GRCh38",
        num_threads: int = 0,
        rsem_bin: str = "rsem-calculate-expression",
    ) -> str:
        """Step 3: RSEM quantification → {sample}.genes.results + isoforms.results.

        Args:
            project_dir: Project directory.
            sample:      Sample prefix.
            ref_dir:     Reference directory.
            genome:      RSEM reference prefix (e.g. 'GRCh38').
            num_threads: CPU threads. 0 = auto-detect.
            rsem_bin:    Path to rsem-calculate-expression.
        """
        safe_dir = str(executor._resolve_project_dir(project_dir))
        if executor.is_done(safe_dir, sample, "step3"):
            return f"[SKIP] step3 already completed for {sample}"
        try:
            _, _, paired = _find_trimmed(safe_dir, sample)
        except FileNotFoundError as e:
            return f"[ERROR] {e}"

        # Validate STAR output is complete before running RSEM
        bam_in = f"{safe_dir}/{sample}Aligned.toTranscriptome.out.bam"
        star_log = f"{safe_dir}/{sample}Log.final.out"
        if not Path(star_log).exists():
            return f"[ERROR] STAR Log.final.out missing for '{sample}' — step2 incomplete or failed."
        bam_err = _bam_quickcheck(executor, bam_in)
        if bam_err:
            return f"[ERROR] {bam_err}. Re-run step2 first."

        res = executor.detect_resources()
        t = num_threads or res["cores"]
        rd = ref_dir or "/BIO1/MRAG/Agent/Reference/RNA_hg38/"
        ref_prefix = str(Path(rd) / genome)  # correct path join regardless of trailing slash
        s = q(sample); d = q(safe_dir)
        pe_flag = "--paired-end" if paired else ""
        cmds = f"""
set -euo pipefail
cd {d}
echo '[step3] mode={"paired" if paired else "single"} sample={sample}'
# Resilience: clear stale RSEM intermediates from interrupted prior run
rm -rf {s}.temp {s}.stat
rm -f {s}.genes.results {s}.isoforms.results {s}.transcript.bam
{q(rsem_bin)} {pe_flag} --bam -p {t} \
  {s}Aligned.toTranscriptome.out.bam {q(ref_prefix)} {s}
"""
        out = executor.run(cmds, cwd=safe_dir)
        if not executor.is_success(out):
            return out

        # Output sanity: RSEM with empty/corrupt input still exits 0; reject empty outputs
        genes_file = f"{safe_dir}/{sample}.genes.results"
        total = _rsem_genes_total_counts(genes_file)
        if total < 1000:
            return (out + f"\n[ERROR] RSEM produced near-empty output "
                          f"(total expected_count={total:.0f}). Not marking step3 done. "
                          f"Likely the input BAM was truncated.")

        executor.mark_done(safe_dir, sample, "step3")
        return out + f"\n[step3] OK: {sample} total expected_count={total:.0f}"

    # ── Step 4: Load RSEM counts (tximport equivalent) ───────────
    @mcp.tool()
    def step4_load_counts(
        project_dir: str,
        mapfile: str,
        rsem_dir: str = "",
        suffix: str = ".genes.results",
    ) -> str:
        """Step 4: Combine per-sample RSEM outputs into counts/TPM matrices.

        Args:
            project_dir: Project directory (outputs written here).
            mapfile:     TSV path with columns 'sample' and 'TREATMENT'.
            rsem_dir:    Directory with .genes.results files. Defaults to project_dir.
            suffix:      RSEM filename suffix.
        """
        pd_dir = executor._resolve_project_dir(project_dir)
        mapfile_path = _resolve_input_path(executor, mapfile)
        samples = pd.read_csv(mapfile_path, sep="\t")
        if not {"sample", "TREATMENT"} <= set(samples.columns):
            return "[ERROR] mapfile must have columns 'sample' and 'TREATMENT'"
        samples["TREATMENT"] = samples["TREATMENT"].str.lower()

        rd = _resolve_input_path(executor, rsem_dir) or pd_dir
        counts, tpm = _import_rsem(samples, rd, suffix)
        counts.to_csv(pd_dir / "txi.rsem.counts.txt", sep="\t")
        tpm.to_csv(pd_dir / "txi.rsem.abundance.TPM.txt", sep="\t")

        executor.mark_done(str(pd_dir), "_global", "step4")
        return json.dumps({
            "counts": str(pd_dir / "txi.rsem.counts.txt"),
            "tpm":    str(pd_dir / "txi.rsem.abundance.TPM.txt"),
            "shape":  {"genes": counts.shape[0], "samples": counts.shape[1]},
            "samples": list(counts.columns),
        }, indent=2)

    # ── Step 5: DESeq2 DE (R or pydeseq2) ────────────────────────
    @mcp.tool()
    def step5_deseq2(
        project_dir: str,
        mapfile: str,
        counts_file: str = "",
        reference: str = "untreated",
        num_threads: int = 8,
        engine: str = "r",
        quant_source: str = "counts",
    ) -> str:
        """Step 5: DESeq2 differential expression analysis.

        engine='r' (default): R DESeq2 (reference implementation, the reference scripts' settings,
                              independent filtering ON by default).
        engine='python': pydeseq2 (legacy / pure-Python pipeline).

        Args:
            project_dir: Project directory (outputs written here).
            mapfile:     TSV with columns 'sample', 'TREATMENT' (treated/untreated).
            counts_file: Path to counts matrix. Defaults to txi.rsem.counts.txt.
            reference:   Reference condition name.
            num_threads: Parallel workers (only used by pydeseq2 engine).
            engine:      'r' or 'python'.
            quant_source: 'counts' (default, mirrors the reference script 1.DESeq2.R
                          via txi.rsem.counts.txt) or 'tpm' (mirrors the reference
                          script 1.DESeq2_TPM.R via txi.rsem.abundance.TPM.txt;
                          rounded to integer inside DESeq2, as that script does).
        """
        pd_dir = executor._resolve_project_dir(project_dir)
        _default_matrix = ("txi.rsem.abundance.TPM.txt"
                           if quant_source == "tpm" else "txi.rsem.counts.txt")
        counts_path = _resolve_input_path(executor, counts_file) or pd_dir / _default_matrix
        mapfile_path = _resolve_input_path(executor, mapfile)
        results_out = pd_dir / "DESeq2_results.csv"
        norm_out = pd_dir / "DESeq2_normalized_counts.tsv"

        if engine == "r":
            try:
                info = _run_deseq2_r(counts_path, mapfile_path, results_out, norm_out,
                                     reference=reference)
            except RuntimeError as e:
                return f"[ERROR] {e}"
            executor.mark_done(str(pd_dir), "_global", "step5")
            return json.dumps({
                "engine":          "r",
                "results":         str(results_out),
                "normalized":      str(norm_out),
                "total_genes":     info["total_genes"],
                "tested_genes":    info["tested_genes"],
                "significant":     info["significant"],
                "significant_thr": "padj<0.10 & |log2FC|>=1",
            }, indent=2)

        # Python (pydeseq2) engine — legacy
        counts = pd.read_csv(counts_path, sep="\t", index_col=0)
        samples = pd.read_csv(mapfile_path, sep="\t")
        samples["TREATMENT"] = samples["TREATMENT"].str.lower()
        err = _validate_deseq_inputs(list(counts.columns), samples, reference.lower())
        if err:
            return f"[ERROR] {err}"

        results, dds = _run_deseq2(counts, samples, reference, num_threads)
        results.to_csv(results_out)

        norm = pd.DataFrame(dds.layers["normed_counts"],
                            index=dds.obs_names, columns=dds.var_names).T
        norm.to_csv(norm_out, sep="\t")

        sig = results[(results["padj"] < 0.10) & (results["log2FoldChange"].abs() >= 1)]
        executor.mark_done(str(pd_dir), "_global", "step5")
        return json.dumps({
            "engine":          "python",
            "results":         str(results_out),
            "normalized":      str(norm_out),
            "total_genes":     len(results),
            "significant":     len(sig),
            "significant_thr": "padj<0.10 & |log2FC|>=1",
        }, indent=2)

    # ── Step 6-8: Plots (MA / Volcano / Heatmap) ────────────────
    @mcp.tool()
    def step6_plot_ma(
        project_dir: str,
        results_file: str = "",
        padj_cutoff: float = 0.05,
        out_name: str = "MA.png",
        engine: str = "r",
    ) -> str:
        """Step 6: MA plot from DE results.

        engine='r' (default): publication-style plot via R (DESeq2-equivalent).
        engine='python': matplotlib version (lighter, fewer R deps).
        """
        pd_dir = executor._resolve_project_dir(project_dir)
        results_path = _resolve_input_path(executor, results_file) or pd_dir / "DESeq2_results.csv"
        out = pd_dir / out_name
        if engine == "r":
            _plot_ma_r(results_path, out, padj_cutoff)
        else:
            res = pd.read_csv(results_path, index_col=0)
            _plot_ma(res, out, padj_cutoff)
        return str(out)

    @mcp.tool()
    def step7_plot_volcano(
        project_dir: str,
        results_file: str = "",
        padj_cutoff: float = 0.10,
        lfc_cutoff: float = 1.0,
        out_name: str = "Volcano.png",
        engine: str = "r",
    ) -> str:
        """Step 7: Volcano plot from DE results.

        engine='r' (default): curated-style 3-color volcano (black/blue/red).
        engine='python': matplotlib version.
        """
        pd_dir = executor._resolve_project_dir(project_dir)
        results_path = _resolve_input_path(executor, results_file) or pd_dir / "DESeq2_results.csv"
        out = pd_dir / out_name
        if engine == "r":
            _plot_volcano_r(results_path, out, padj_cutoff, lfc_cutoff)
        else:
            res = pd.read_csv(results_path, index_col=0)
            _plot_volcano(res, out, padj_cutoff, lfc_cutoff)
        return str(out)

    @mcp.tool()
    def step8_plot_heatmap(
        project_dir: str,
        mapfile: str,
        results_file: str = "",
        counts_file: str = "",
        norm_file: str = "",
        reference: str = "untreated",
        top_n: int = 50,
        padj_cutoff: float = 0.10,
        lfc_cutoff: float = 1.0,
        annotation_file: str = "",
        normalize: str = "zscore",
        out_name: str = "Heatmap.png",
        num_threads: int = 4,
        engine: str = "r",
    ) -> str:
        """Step 8: Heatmap of top significant genes.

        engine='r' (default): curated-style heatmap.2 (GnBu, Colv=FALSE, gene symbols).
        engine='python': seaborn clustermap (vlag z-score).

        Reuses DESeq2_normalized_counts.tsv from step5 if available (fast path).
        Re-fits DESeq2 only when normalized counts file is missing.

        Args:
            project_dir:     Project directory.
            mapfile:         TSV with 'sample' and 'TREATMENT'.
            results_file:    DESeq2 results CSV. Default: project_dir/DESeq2_results.csv.
            counts_file:     Counts matrix (only needed for re-fit fallback).
            norm_file:       Normalized counts TSV. Default: project_dir/DESeq2_normalized_counts.tsv.
            reference:       Reference condition (only used for re-fit fallback).
            top_n:           Number of top genes to display.
            padj_cutoff:     Significance threshold.
            lfc_cutoff:      Log2FC threshold.
            annotation_file: (optional) TSV with Gene.ID→Gene.Name mapping. R engine
                             auto-detects '/BIO1/MRAG/Agent/Reference/RNA_hg38/gene_annotation.tsv'.
            normalize:       'zscore' (default) or 'none' (raw log2 counts). [python only]
            out_name:        Output filename.
            num_threads:     pydeseq2 threads (only used for re-fit fallback).
            engine:          'r' or 'python'.
        """
        pd_dir = executor._resolve_project_dir(project_dir)
        mapfile_path = _resolve_input_path(executor, mapfile)
        results_path = _resolve_input_path(executor, results_file) or pd_dir / "DESeq2_results.csv"
        annotation_path = _resolve_input_path(executor, annotation_file)
        # Auto-detect default annotation when caller didn't pass one (R engine only)
        if annotation_path is None and engine == "r":
            default_annot = Path("/BIO1/MRAG/Agent/Reference/RNA_hg38/gene_annotation.tsv")
            if default_annot.exists():
                annotation_path = default_annot
        out = pd_dir / out_name

        if engine == "r":
            norm_path = _resolve_input_path(executor, norm_file) or pd_dir / "DESeq2_normalized_counts.tsv"
            if not norm_path.exists():
                return f"[ERROR] {norm_path} not found. Run step5 first or pass norm_file."
            n = _plot_heatmap_r(results_path, norm_path, mapfile_path, out,
                                annotation_tsv=annotation_path,
                                top_n=top_n, padj_cutoff=padj_cutoff,
                                lfc_cutoff=lfc_cutoff)
            return json.dumps({"heatmap": str(out) if n > 0 else None,
                               "n_plotted": n,
                               "engine": "r"}, indent=2)

        # Python engine path (legacy)
        samples = pd.read_csv(mapfile_path, sep="\t")
        samples["TREATMENT"] = samples["TREATMENT"].str.lower()
        results = pd.read_csv(results_path, index_col=0)
        annot = pd.read_csv(annotation_path, sep="\t") if annotation_path else None
        norm_path = _resolve_input_path(executor, norm_file) or pd_dir / "DESeq2_normalized_counts.tsv"
        if norm_path.exists():
            norm_df = pd.read_csv(norm_path, sep="\t", index_col=0)
            sig = _plot_heatmap(results, samples, out, norm=norm_df,
                                top_n=top_n, padj_cutoff=padj_cutoff,
                                lfc_cutoff=lfc_cutoff, annotation=annot,
                                normalize=normalize)
        else:
            counts_path = _resolve_input_path(executor, counts_file) or pd_dir / "txi.rsem.counts.txt"
            counts = pd.read_csv(counts_path, sep="\t", index_col=0)
            err = _validate_deseq_inputs(list(counts.columns), samples, reference.lower())
            if err:
                return f"[ERROR] {err}"
            _, dds = _run_deseq2(counts, samples, reference, num_threads)
            sig = _plot_heatmap(results, samples, out, dds=dds,
                                top_n=top_n, padj_cutoff=padj_cutoff,
                                lfc_cutoff=lfc_cutoff, annotation=annot,
                                normalize=normalize)

        if len(sig) == 0:
            return json.dumps({
                "heatmap": None,
                "n_plotted": 0,
                "message": "No significant genes met the cutoffs — no image written.",
            }, indent=2)
        return json.dumps({"heatmap": str(out), "n_plotted": len(sig)}, indent=2)

    # ── End-to-end ───────────────────────────────────────────────
    @mcp.tool()
    def run_full_pipeline(
        project_dir: str,
        ref_dir: str = "",
        genome: str = "GRCh38",
        num_threads: int = 0,
    ) -> str:
        """Run steps 1-3 (preprocess, STAR, RSEM) for every sample found.
        Steps 4-8 (counts → DE → plots) require a mapfile and should be run explicitly.

        Args:
            project_dir: Project directory containing *_1.fastq.gz files.
            ref_dir:     Reference dir.
            genome:      RSEM reference prefix.
            num_threads: CPU threads. 0 = auto-detect.
        """
        safe_dir = str(executor._resolve_project_dir(project_dir))
        lock_path = Path(safe_dir) / ".rna_pipeline_run.lock"
        lock_fh = open(lock_path, "w")
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_fh.close()
            return "[ERROR] Another pipeline run is already active on this project directory. Aborting to prevent concurrent file writes."

        try:
            rd = ref_dir or "/BIO1/MRAG/Agent/Reference/RNA_hg38/"
            samples = _list_samples(safe_dir)
            if not samples:
                return f"No FASTQ files (*.fastq.gz / *.fq.gz) found in {safe_dir}"

            # Run up to 4 samples in parallel; >4 samples queue in batches.
            res = executor.detect_resources()
            total_threads = num_threads or res["cores"]
            max_parallel = min(4, len(samples))
            threads_per_sample = max(4, total_threads // max_parallel)

            def _process_one(s: str) -> str:
                lines = [f"=== {s} ==="]
                lines.append(step1_preprocessing(safe_dir, s, threads_per_sample))
                lines.append(step2_star(safe_dir, s, rd, threads_per_sample))
                lines.append(step3_rsem(safe_dir, s, rd, genome, threads_per_sample))
                return "\n".join(lines)

            results_map: dict[str, str] = {}
            with ThreadPoolExecutor(max_workers=max_parallel) as pool:
                fut_to_sample = {pool.submit(_process_one, s): s for s in samples}
                for fut in as_completed(fut_to_sample):
                    s = fut_to_sample[fut]
                    try:
                        results_map[s] = fut.result()
                    except Exception as e:
                        results_map[s] = f"=== {s} ===\n[ERROR] {e}"

            header = (
                f"[run_full_pipeline] {len(samples)} sample(s): {samples} "
                f"(parallel: {max_parallel} workers × {threads_per_sample} threads each)"
            )
            return header + "\n" + "\n".join(results_map[s] for s in samples)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
            lock_fh.close()
