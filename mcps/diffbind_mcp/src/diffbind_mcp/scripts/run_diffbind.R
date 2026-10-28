#!/usr/bin/env Rscript
# MRAG DiffBind runner — parameterized version of the reference analysis script 0.DiffBind.R.
#
# Usage (called from diffbind-mcp via subprocess):
#   Rscript run_diffbind.R \
#       --sheet PATH         (sample sheet CSV, DiffBind standard format)
#       --output PATH        (output CSV with per-peak FC / p / FDR)
#       [--category COL]     (default: "Condition")
#       [--min-members N]    (default: 2)
#       [--threshold T]      (default: 1.0  -> dba.report th=1 returns all peaks)
#       [--threads N]        (default: 4 -- BiocParallel workers for dba.count)
#       [--summary PATH]     (JSON-ish status file; optional)
#
# Output schema (matches the reference analysis scripts' DiffBind_out.csv):
#   seqnames, start, end, width, strand, Conc, Conc_<groupA>, Conc_<groupB>,
#   Fold, p.value, FDR
#
# Exit codes:
#   0 = success
#   1 = parse / file / DiffBind error  (stderr carries the message)

suppressPackageStartupMessages({
  library(DiffBind)
  library(BiocParallel)
})

args <- commandArgs(trailingOnly = TRUE)
opts <- list(
  sheet       = NULL,
  output      = NULL,
  category    = "Condition",
  reference   = NULL,
  min_members = 2L,
  threshold   = 1.0,
  threads     = 4L,
  summary     = NULL
)
i <- 1
while (i <= length(args)) {
  k <- args[[i]]
  v <- if (i + 1L <= length(args)) args[[i + 1L]] else NA_character_
  if (k == "--sheet")        { opts$sheet       <- v;        i <- i + 2L }
  else if (k == "--output")  { opts$output      <- v;        i <- i + 2L }
  else if (k == "--category"){ opts$category    <- v;        i <- i + 2L }
  else if (k == "--reference"){ opts$reference  <- v;        i <- i + 2L }
  else if (k == "--min-members") { opts$min_members <- as.integer(v); i <- i + 2L }
  else if (k == "--threshold")   { opts$threshold   <- as.numeric(v); i <- i + 2L }
  else if (k == "--threads") { opts$threads     <- as.integer(v); i <- i + 2L }
  else if (k == "--summary") { opts$summary     <- v;        i <- i + 2L }
  else {
    stop(sprintf("unknown arg: %s", k))
  }
}

if (is.null(opts$sheet) || !file.exists(opts$sheet)) {
  stop(sprintf("sample sheet not found: %s", opts$sheet))
}
if (is.null(opts$output)) {
  stop("--output is required")
}

# Parallel backend: register globally so DiffBind picks it up internally.
# DiffBind 3.x exposes only `bParallel=TRUE` on dba.count, not BPPARAM.
if (opts$threads <= 1) {
  register(SerialParam())
} else {
  register(MulticoreParam(workers = opts$threads))
}

# Map --category CLI value to the DiffBind enum constant.
cat_enum <- switch(
  toupper(opts$category),
  "CONDITION" = DBA_CONDITION,
  "TREATMENT" = DBA_TREATMENT,
  "FACTOR"    = DBA_FACTOR,
  "TISSUE"    = DBA_TISSUE,
  "REPLICATE" = DBA_REPLICATE,
  stop(sprintf("unsupported --category '%s' (expected one of Condition/Treatment/Factor/Tissue/Replicate)",
               opts$category))
)

t0 <- Sys.time()
message(sprintf("[diffbind] sheet=%s",  opts$sheet))
message(sprintf("[diffbind] output=%s", opts$output))
message(sprintf("[diffbind] category=%s  min_members=%d  threshold=%s  threads=%d",
                opts$category, opts$min_members, format(opts$threshold), opts$threads))

dba_obj <- dba(sampleSheet = opts$sheet)
message(sprintf("[diffbind] dba() loaded %d samples", nrow(dba_obj$samples)))

dba_obj <- dba.count(dba_obj, bParallel = opts$threads > 1)
message("[diffbind] dba.count done")

dba_obj <- dba.contrast(dba_obj, categories = cat_enum, minMembers = opts$min_members)
n_contrasts <- length(dba_obj$contrasts)
message(sprintf("[diffbind] dba.contrast done (%d contrast(s))", n_contrasts))
if (n_contrasts < 1L) {
  stop(sprintf("no contrasts produced for category=%s minMembers=%d -- check the sheet",
               opts$category, opts$min_members))
}

dba_obj <- dba.analyze(dba_obj, bParallel = opts$threads > 1)
message("[diffbind] dba.analyze done")

report <- dba.report(dba_obj, th = opts$threshold)
report.df <- as.data.frame(report)

# --- Normalize Fold to treatment-referenced convention (mirror RNA-seq DESeq2) ---
# DiffBind's auto-contrast picks the baseline internally (often alphabetical),
# so "Fold" may be control - treatment (positive = up in control). When a
# --reference (control/baseline level) is supplied, recompute Fold from the
# per-group log2 concentrations so positive = up in the NON-reference
# (treatment) group, matching DESeq2 log2FoldChange (up = treatment).
if (!is.null(opts$reference)) {
  conc_cols <- grep("^Conc_", colnames(report.df), value = TRUE)
  ref_col   <- paste0("Conc_", make.names(opts$reference))
  if (length(conc_cols) == 2L && ref_col %in% conc_cols) {
    trt_col <- setdiff(conc_cols, ref_col)
    # DiffBind's Fold is a model-based log fold change (magnitude differs from
    # the raw Conc difference), so we correct only its SIGN -- never its value.
    # Decide the current orientation by majority agreement between sign(Fold)
    # and sign(Conc_treatment - Conc_reference) across all peaks, then flip if
    # DiffBind's positive direction points at the reference (control) group.
    d        <- report.df[[trt_col]] - report.df[[ref_col]]
    agree    <- sum(sign(report.df$Fold) ==  sign(d), na.rm = TRUE)
    disagree <- sum(sign(report.df$Fold) == -sign(d), na.rm = TRUE)
    if (disagree > agree) {
      report.df$Fold <- -report.df$Fold
      message(sprintf("[diffbind] Fold sign flipped to treatment-referenced (positive = up in %s; ref=%s; agree=%d disagree=%d)",
                      sub("^Conc_", "", trt_col), opts$reference, agree, disagree))
    } else {
      message(sprintf("[diffbind] Fold already treatment-referenced (positive = up in %s); no flip (agree=%d disagree=%d)",
                      sub("^Conc_", "", trt_col), agree, disagree))
    }
  } else {
    message(sprintf("[diffbind] WARNING: --reference '%s' -> '%s' not among Conc cols {%s}; Fold left as DiffBind default",
                    opts$reference, ref_col, paste(conc_cols, collapse = ",")))
  }
}

write.csv(report.df, opts$output, quote = FALSE, row.names = FALSE)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
message(sprintf("[diffbind] wrote %d rows to %s (%.1fs)",
                nrow(report.df), opts$output, elapsed))

if (!is.null(opts$summary)) {
  # Tiny machine-readable status -- the MCP layer can parse this.
  writeLines(sprintf(paste0(
    '{"status":"ok",',
    '"n_rows":%d,',
    '"n_samples":%d,',
    '"n_contrasts":%d,',
    '"category":"%s",',
    '"min_members":%d,',
    '"threshold":%s,',
    '"elapsed_s":%.2f,',
    '"output":"%s"}'),
    nrow(report.df), nrow(dba_obj$samples), n_contrasts,
    opts$category, opts$min_members, format(opts$threshold), elapsed,
    opts$output), opts$summary)
}
