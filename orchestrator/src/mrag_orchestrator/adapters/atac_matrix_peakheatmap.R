#!/usr/bin/env Rscript
# Peak accessibility heatmap for PROCESSED-MATRIX ATAC input.
# Replaces the deeptools (BAM->bigwig->computeMatrix) Section-6 heatmap, which
# needs per-replicate BAMs, with a count-matrix native equivalent:
#   rows  = peaks classified by differential accessibility (DiffBind Fold/FDR)
#           into A-specific / Shared / B-specific classes
#   cols  = samples, annotated by group
#   value = per-sample log2 CPM, row Z-scored (scale="row")
suppressPackageStartupMessages({library(optparse); library(pheatmap)})

op <- OptionParser(option_list = list(
  make_option("--counts",  type="character", help="ATAC count matrix (seqnames start end name score <samples...>)"),
  make_option("--groups",  type="character", help="group TSV (sample<TAB>group)"),
  make_option("--diffbind",type="character", help="DiffBind_out.csv (Fold,FDR; same peak order)"),
  make_option("--output",  type="character", help="output PNG"),
  make_option("--treat",   type="character", default="treatment"),
  make_option("--ref",     type="character", default="control"),
  make_option("--per_class", type="integer", default=300),
  make_option("--fdr",     type="numeric", default=0.05),
  make_option("--width",   type="numeric", default=8),
  make_option("--height",  type="numeric", default=10),
  make_option("--dpi",     type="integer", default=180)
))
o <- parse_args(op)
stopifnot(!is.null(o$counts), !is.null(o$groups), !is.null(o$diffbind), !is.null(o$output))

cm <- read.table(o$counts, sep="\t", header=TRUE, check.names=FALSE, quote="", comment.char="")
db <- read.csv(o$diffbind, check.names=FALSE)
G  <- read.table(o$groups, sep="\t", header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
# voom filterByExpr can drop low-count peaks, so DiffBind may have fewer rows
# than the raw count matrix.  Join on peak coordinate (seqnames:start:end).
key_cm <- paste(cm$seqnames, cm$start, cm$end, sep=":")
key_db <- paste(db$seqnames, db$start, db$end, sep=":")
keep <- key_cm %in% key_db
cm <- cm[keep, , drop=FALSE]
db <- db[match(key_cm[keep], key_db), , drop=FALSE]
stopifnot(nrow(cm) == nrow(db))

meta_cols <- c("seqnames","start","end","name","score")
smp <- setdiff(colnames(cm), meta_cols)
smp <- intersect(smp, G$sample)
if (length(smp) < 2) stop("no overlapping samples between count matrix and group TSV")
M <- as.matrix(cm[, smp, drop=FALSE]); mode(M) <- "numeric"; M[is.na(M)] <- 0
grp <- G$group[match(smp, G$sample)]

# CPM -> log2
libsize <- colSums(M); libsize[libsize==0] <- 1
cpm <- t(t(M) / libsize) * 1e6
L <- log2(cpm + 1)

Fold <- db$Fold; FDR <- db$FDR
cls <- rep("Shared", nrow(db))
cls[FDR < o$fdr & Fold > 0] <- o$treat
cls[FDR < o$fdr & Fold < 0] <- o$ref
# select top peaks per class
pick <- function(idx, byval, n, decreasing=TRUE) {
  if (!length(idx)) return(integer(0))
  idx[order(byval[idx], decreasing=decreasing)][seq_len(min(n, length(idx)))]
}
i_t <- pick(which(cls==o$treat), Fold,  o$per_class, TRUE)
i_r <- pick(which(cls==o$ref),   Fold,  o$per_class, FALSE)
i_s <- pick(which(cls=="Shared"), abs(Fold), o$per_class, FALSE)  # smallest |Fold| = most shared
sel <- c(i_t, i_s, i_r)
if (!length(sel)) stop("no peaks selected")
Ls <- L[sel, , drop=FALSE]
rownames(Ls) <- make.unique(as.character(cm$name[sel]))
row_class <- factor(cls[sel], levels=c(o$treat, "Shared", o$ref))

# order columns by group (treat first)
col_ord <- order(factor(grp, levels=unique(c(o$treat, o$ref, grp))))
Ls <- Ls[, col_ord, drop=FALSE]; grp2 <- grp[col_ord]; smp2 <- smp[col_ord]

annotation_col <- data.frame(Group = grp2); rownames(annotation_col) <- smp2
annotation_row <- data.frame(Class = row_class); rownames(annotation_row) <- rownames(Ls)
ann_colors <- list(
  Group = setNames(c("#D95F02", "#28347A"), c(o$treat, o$ref)),
  Class = setNames(c("#D95F02", "#B0B0B0", "#28347A"), c(o$treat, "Shared", o$ref)))

dir.create(dirname(o$output), showWarnings=FALSE, recursive=TRUE)
png(o$output, width=o$width, height=o$height, units="in", res=o$dpi, bg="white")
pheatmap(Ls, scale="row",
         cluster_rows=FALSE, cluster_cols=FALSE,
         show_rownames=FALSE, show_colnames=FALSE,
         color=colorRampPalette(c("white","#6A0DAD"))(100),
         annotation_col=annotation_col, annotation_row=annotation_row,
         annotation_colors=ann_colors,
         main=sprintf("ATAC peak accessibility (%d peaks x %d samples)", nrow(Ls), ncol(Ls)))
invisible(dev.off())
cat(sprintf("wrote %s (%d peaks: %s=%d Shared=%d %s=%d)\n",
    o$output, nrow(Ls), o$treat, length(i_t), length(i_s), o$ref, length(i_r)))
