suppressMessages({library(limma); library(org.Hs.eg.db)})
a <- commandArgs(trailingOnly=TRUE)
mat_f<-a[1]; grp_f<-a[2]; out_f<-a[3]; treat<-a[4]; ref<-a[5]
raw <- read.table(mat_f, sep="\t", header=TRUE, check.names=FALSE, quote="", comment.char="")
gene <- as.character(raw[[1]]); M <- raw[,-1,drop=FALSE]
# collapse duplicate genes by mean (log2 space)
M <- as.matrix(M); mode(M) <- "numeric"; M[is.na(M)] <- 0
if (anyDuplicated(gene)) { M <- rowsum(M, gene) / as.numeric(table(gene)[gene][!duplicated(gene)]); gene <- rownames(M) } else { rownames(M) <- gene }
G <- read.table(grp_f, sep="\t", header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
keep <- intersect(colnames(M), G$sample)
M <- M[, keep, drop=FALSE]; grp <- G$group[match(keep, G$sample)]
sel <- grp %in% c(treat, ref); M <- M[, sel, drop=FALSE]; grp <- grp[sel]
grp <- factor(grp, levels=c(ref, treat))
cat(sprintf("[rna] %d genes x %d samples (%s=%d %s=%d)\n", nrow(M), ncol(M), treat, sum(grp==treat), ref, sum(grp==ref)))
design <- model.matrix(~grp)
fit <- eBayes(lmFit(M, design))
tt <- topTable(fit, coef=2, number=Inf, sort.by="none")
ent <- rownames(M)
sym <- tryCatch(mapIds(org.Hs.eg.db, keys=ent, column="SYMBOL", keytype="ENTREZID", multiVals="first"), error=function(e) setNames(rep(NA,length(ent)),ent))
sym[is.na(sym)] <- ent[is.na(sym)]
out <- data.frame(ID=ent, GENE=unname(sym), baseMean=tt$AveExpr, log2FoldChange=tt$logFC,
                  lfcSE=NA_real_, stat=tt$t, pvalue=tt$P.Value, padj=tt$adj.P.Val, check.names=FALSE)
write.csv(out, out_f, row.names=FALSE)
cat(sprintf("[rna] wrote %s rows=%d sig(padj<0.05,|lfc|>1)=%d\n", out_f, nrow(out), sum(out$padj<0.05 & abs(out$log2FoldChange)>1, na.rm=TRUE)))
