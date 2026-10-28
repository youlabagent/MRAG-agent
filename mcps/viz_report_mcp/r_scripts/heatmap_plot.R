#!/usr/bin/env Rscript
# Heatmap of top-N up + top-N down genes' RSEM counts.
# Mirrors the original Plot/heatmap.R but takes CLI args + dynamic group annotation.

suppressPackageStartupMessages({
  library(optparse); library(pheatmap)
})

op <- OptionParser(option_list = list(
  make_option("--integrated", type="character",
              help="INTEGRATED_ranked.csv path"),
  make_option("--deseq",      type="character",
              help="DESeq2.gene.csv path (with ID/GENE columns)"),
  make_option("--counts",     type="character",
              help="txi.rsem.counts.txt path (RSEM gene counts)"),
  make_option("--output",     type="character", help="output PNG path"),
  make_option("--top_n",      type="integer", default=10,
              help="number of top up + top down genes"),
  make_option("--group_a",    type="character", default="OC2",
              help="arm A label (treatment)"),
  make_option("--group_b",    type="character", default="Cyc",
              help="arm B label (control)"),
  make_option("--samples_a",  type="character", default="",
              help="optional comma-separated sample names for arm A (counts columns)"),
  make_option("--samples_b",  type="character", default="",
              help="optional comma-separated sample names for arm B (counts columns)"),
  make_option("--color_a",    type="character", default="gold"),
  make_option("--color_b",    type="character", default="purple"),
  make_option("--width",      type="numeric", default=8),
  make_option("--height",     type="numeric", default=10),
  make_option("--dpi",        type="integer", default=180)
))
o <- parse_args(op)
stopifnot(!is.null(o$integrated), !is.null(o$deseq),
          !is.null(o$counts), !is.null(o$output))

df  <- read.csv(o$integrated, check.names = FALSE)
n_total  <- nrow(df)
# INTEGRATED.csv's Gene column is the ATAC/HOCOMOCO spelling -- UPPERCASE and
# human-styled for every species. The DESeq2 table below carries the RUN's own
# namespace (mouse TitleCase: Rest, Ar, Trp53), so joining the two on Gene
# matched nothing on a mouse run and this script aborted with "No matching
# IDs" while Section 7 simply went missing (2026-09-23). combine_pvalues now
# writes RNA_Gene -- the expression-table spelling it actually joined -- for
# exactly this. Human runs have no RNA_Gene column and take the old path.
gene_key <- if ("RNA_Gene" %in% names(df)) "RNA_Gene" else "Gene"
top_up   <- df[[gene_key]][df$Rank_Combined_Score <= o$top_n]
top_down <- df[[gene_key]][df$Rank_Combined_Score >= (n_total - (o$top_n - 1))]
top_up   <- as.character(top_up); top_down <- as.character(top_down)

rna <- read.csv(o$deseq, check.names = FALSE)
# Normalise gene-name column to "GENE"
gene_col <- intersect(c("GENE","gene","Gene","gene_name"), names(rna))[1]
id_col   <- intersect(c("ID","X","gene_id"),                names(rna))[1]
stopifnot(!is.na(gene_col), !is.na(id_col))
names(rna)[names(rna) == gene_col] <- "GENE"
names(rna)[names(rna) == id_col]   <- "ID"

top_up_ID   <- rna$ID[rna$GENE %in% top_up]
top_down_ID <- rna$ID[rna$GENE %in% top_down]
all_ID <- c(top_up_ID, top_down_ID)
if (!length(all_ID)) {
  # Last resort before failing: fold case. Never fires on human (the strings
  # are already identical); on a non-human run it rescues the symbols that
  # differ only in capitalisation.
  .u <- toupper(as.character(rna$GENE))
  top_up_ID   <- rna$ID[.u %in% toupper(top_up)]
  top_down_ID <- rna$ID[.u %in% toupper(top_down)]
  all_ID <- c(top_up_ID, top_down_ID)
  if (length(all_ID)) message("[heatmap] matched INTEGRATED to DESeq2 case-insensitively")
}
if (!length(all_ID)) stop(sprintf(
  paste0("No matching IDs between INTEGRATED top-rank genes and DESeq2 file. ",
         "Joined INTEGRATED column '%s' (e.g. %s) against DESeq2 GENE (e.g. %s). ",
         "These are two different gene-symbol namespaces -- on a non-human run ",
         "INTEGRATED.csv must carry an RNA_Gene column for this join."),
  gene_key, paste(utils::head(top_up, 3), collapse=","),
  paste(utils::head(as.character(rna$GENE), 3), collapse=",")))

# Auto-detect file format: txi.rsem.counts.txt style starts with an empty
# leading tab on the header line ("\tSRR...\tSRR..."), so we read with
# row.names=1 to use the first column as gene IDs. For files that have an
# explicit "X" / "gene_id" / "ID" column we materialise that column instead.
header_line <- readLines(o$counts, n = 1)
counts <- if (startsWith(header_line, "\t")) {
  read.table(o$counts, sep="\t", header=TRUE, row.names=1, check.names=FALSE)
} else {
  read.table(o$counts, sep="\t", header=TRUE, check.names=FALSE)
}
id_counts_col <- intersect(c("X","gene_id","ID","gene","GENE","Gene"), names(counts))[1]
if (is.na(id_counts_col)) {
  counts$ID <- rownames(counts)
} else {
  names(counts)[names(counts) == id_counts_col] <- "ID"
}
counts_sub <- counts[counts$ID %in% all_ID, , drop = FALSE]
if (!nrow(counts_sub)) stop("No counts rows for selected IDs")
rownames(counts_sub) <- counts_sub$ID
counts_sub$ID <- NULL

# Replace ID with gene SYMBOL
rna_sub <- rna[rna$ID %in% all_ID, , drop = FALSE]
rownames(counts_sub) <- rna_sub$GENE[match(rownames(counts_sub), rna_sub$ID)]

# Build group annotation
arm_a_samples <- if (nchar(o$samples_a) > 0) strsplit(o$samples_a, ",")[[1]] else NULL
arm_b_samples <- if (nchar(o$samples_b) > 0) strsplit(o$samples_b, ",")[[1]] else NULL
sample_cols <- colnames(counts_sub)
if (is.null(arm_a_samples) || is.null(arm_b_samples)) {
  # Default: split columns in half, first half = arm A, second half = arm B
  n <- length(sample_cols); half <- n %/% 2
  arm_a_samples <- sample_cols[seq_len(half)]
  arm_b_samples <- sample_cols[(half+1):n]
}
group <- character(length(sample_cols))
group[sample_cols %in% arm_a_samples] <- o$group_a
group[sample_cols %in% arm_b_samples] <- o$group_b
# If no columns matched (sample names differ from counts column names, e.g.
# run names vs SRR IDs), fall back to half-split so group is never all "".
if (!any(group != "")) {
  message("NOTE: provided sample names do not match counts columns; using half-split")
  n <- length(sample_cols); half <- n %/% 2
  arm_a_samples <- sample_cols[seq_len(half)]
  arm_b_samples <- sample_cols[(half+1):n]
  group[sample_cols %in% arm_a_samples] <- o$group_a
  group[sample_cols %in% arm_b_samples] <- o$group_b
}
annotation_col <- data.frame(Group = group)
rownames(annotation_col) <- sample_cols
ann_colors <- list(); ann_colors[[ "Group" ]] <- setNames(
  c(o$color_a, o$color_b), c(o$group_a, o$group_b))

dir.create(dirname(o$output), showWarnings = FALSE, recursive = TRUE)
png(o$output, width = o$width, height = o$height, units = "in", res = o$dpi,
    bg = "white")
pheatmap(counts_sub, scale = "row",
         cluster_rows = TRUE, cluster_cols = TRUE,
         fontsize_row = 10, fontsize_col = 10,
         color = colorRampPalette(c("blue","white","red"))(100),
         annotation_col = annotation_col,
         annotation_colors = ann_colors, angle_col = 315)
invisible(dev.off())
cat(sprintf("✓ wrote %s (%d genes)\n", o$output, nrow(counts_sub)))
