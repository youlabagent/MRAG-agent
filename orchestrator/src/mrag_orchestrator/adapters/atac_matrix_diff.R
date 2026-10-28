suppressMessages({library(edgeR); library(limma)})
a <- commandArgs(trailingOnly=TRUE)
mat_f<-a[1]; grp_f<-a[2]; out_dir<-a[3]; treat<-a[4]; ref<-a[5]
floor_n <- ifelse(length(a)>=6, as.integer(a[6]), 5000L)
raw <- read.table(mat_f, sep="\t", header=TRUE, check.names=FALSE, quote="", comment.char="")
meta_cols <- c("seqnames","start","end","name","score")
mc <- meta_cols[meta_cols %in% colnames(raw)]
coord <- raw[, mc, drop=FALSE]
smp_cols <- setdiff(colnames(raw), mc)
M <- as.matrix(raw[, smp_cols, drop=FALSE]); mode(M) <- "numeric"
G <- read.table(grp_f, sep="\t", header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
keep <- intersect(colnames(M), G$sample); M <- M[, keep, drop=FALSE]
grp <- G$group[match(keep, G$sample)]; sel <- grp %in% c(treat, ref)
M <- M[, sel, drop=FALSE]; grp <- factor(grp[sel], levels=c(ref, treat))
cat(sprintf("[atac] %d peaks x %d samples (%s=%d %s=%d)\n", nrow(M), ncol(M), treat, sum(grp==treat), ref, sum(grp==ref)))
d <- DGEList(counts=M)
kexp <- filterByExpr(d, group=grp); d <- d[kexp,,keep.lib.sizes=FALSE]; coord <- coord[kexp,,drop=FALSE]
d <- calcNormFactors(d)
design <- model.matrix(~grp)
v <- voom(d, design)
fit <- eBayes(lmFit(v, design))
tt <- topTable(fit, coef=2, number=Inf, sort.by="none")
cat(sprintf("[atac] tested %d peaks; FDR<0.05 up=%d down=%d\n", nrow(tt),
            sum(tt$adj.P.Val<0.05 & tt$logFC>0, na.rm=TRUE), sum(tt$adj.P.Val<0.05 & tt$logFC<0, na.rm=TRUE)))
ord_up  <- order(ifelse(tt$logFC>0, tt$P.Value, Inf))
ord_dn  <- order(ifelse(tt$logFC<0, tt$P.Value, Inf))
sig_up  <- which(tt$adj.P.Val<0.05 & tt$logFC>0)
sig_dn  <- which(tt$adj.P.Val<0.05 & tt$logFC<0)
pick_up <- if(length(sig_up)>=floor_n) sig_up else head(ord_up, floor_n)
pick_dn <- if(length(sig_dn)>=floor_n) sig_dn else head(ord_dn, floor_n)
wbed <- function(idx, path){
  b <- coord[idx, c("seqnames","start","end"), drop=FALSE]
  b$name <- if("name" %in% colnames(coord)) coord$name[idx] else paste0("peak", idx)
  write.table(b, path, sep="\t", quote=FALSE, row.names=FALSE, col.names=FALSE)
}
tb <- file.path(out_dir, "treatment_diff.bed"); cb <- file.path(out_dir, "control_diff.bed")
wbed(pick_up, tb); wbed(pick_dn, cb)
cat(sprintf("[atac] wrote treatment(%s-up)=%d peaks -> %s\n", treat, length(pick_up), tb))
cat(sprintf("[atac] wrote control(%s-up)=%d peaks -> %s\n", ref, length(pick_dn), cb))

# --- DiffBind-format CSV (feeds downstream DiffBind/GIGGLE/volcano) ---
gm_t <- rowMeans(v$E[, grp==treat, drop=FALSE])
gm_r <- rowMeans(v$E[, grp==ref,   drop=FALSE])
db <- data.frame(
  seqnames = coord$seqnames,
  start    = coord$start,
  end      = coord$end,
  width    = as.integer(coord$end) - as.integer(coord$start),
  strand   = "*",
  Conc     = tt$AveExpr,
  ct       = gm_t,
  cr       = gm_r,
  Fold     = tt$logFC,
  p.value  = tt$P.Value,
  FDR      = tt$adj.P.Val,
  check.names = FALSE, stringsAsFactors = FALSE)
colnames(db)[colnames(db)=="ct"] <- paste0("Conc_", make.names(treat))
colnames(db)[colnames(db)=="cr"] <- paste0("Conc_", make.names(ref))
db <- db[, c("seqnames","start","end","width","strand","Conc",
             paste0("Conc_", make.names(treat)), paste0("Conc_", make.names(ref)),
             "Fold","p.value","FDR")]
dbf <- file.path(out_dir, "DiffBind_out.csv")
write.csv(db, dbf, row.names=FALSE, quote=FALSE)
cat(sprintf("[atac] wrote DiffBind CSV %d peaks (FDR<0.05: up=%d down=%d) -> %s\n",
            nrow(db), sum(db$Fold>0 & db$FDR<0.05, na.rm=TRUE),
            sum(db$Fold<0 & db$FDR<0.05, na.rm=TRUE), dbf))
