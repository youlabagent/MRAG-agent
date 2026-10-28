"""trn-mcp: the processed-matrix join of the extended Pval2 table (Section 10, SE -> TF -> Target TRN).

MCP step 2a (2026-10-12): SETrnBuilderNode._matrix_join (nodes/se_trn.py), moved here
unchanged as the MCP tool join_matrix_tfs_pval2.  For processed-matrix input the DiffBind / DESeq2 join of
join_diffbind_and_rna does not fit the voom DiffBind / limma tables; this is the same join as the curated
build_pval2_from_matrix: TFs.txt (promoter rows + extended motif columns) + per-peak Fold (DiffBind coordinates ->
canonical peak name) + per-gene log2FC (DESeq2) -> a Pval2-style table."""
from __future__ import annotations

import csv as _csv


def join_matrix_tfs_pval2(tfs_txt: str, canonical_peaks_bed: str, diffbind_csv: str, deseq_csv: str,
                          output_path: str) -> str:
    """TFs.txt + canonical peak BED + DiffBind CSV + DESeq2 CSV -> Pval2-style table.

    Args:
        tfs_txt: filter_target_tfs output (promoter rows, extended motif columns; first column = peak name).
        canonical_peaks_bed: the matrix's canonical peak BED (chrom, start, end, name).
        diffbind_csv: per-peak table with seqnames/start/end/Fold[/p.value] (coordinates may be in exponent form).
        deseq_csv: per-gene table with GENE (or Gene/gene) and log2FoldChange (or log2FC) [, pvalue].
        output_path: where the table is written.

    Returns output_path.  Rows whose gene has no log2FC are dropped; columns added: Fold(DiffBind),
    p.value(DiffBind), log2FC, RNA_pval."""
    out_path = output_path
    coord2name = {}
    for line in open(canonical_peaks_bed):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 4:
            coord2name[(p[0], str(int(float(p[1]))), str(int(float(p[2]))))] = p[3]
    fold, pval = {}, {}
    with open(diffbind_csv) as fh:
        r = _csv.reader(fh); h = next(r)
        ci = {c: h.index(c) for c in ("seqnames", "start", "end", "Fold") if c in h}
        pi = h.index("p.value") if "p.value" in h else None
        for row in r:
            try:
                k = (row[ci["seqnames"]], str(int(float(row[ci["start"]]))), str(int(float(row[ci["end"]]))))
            except (ValueError, KeyError, IndexError):
                continue
            nm = coord2name.get(k)
            if nm is not None:
                fold[nm] = row[ci["Fold"]]
                if pi is not None:
                    pval[nm] = row[pi]
    lfc, rp = {}, {}
    with open(deseq_csv) as fh:
        r = _csv.DictReader(fh)
        for row in r:
            g = row.get("GENE") or row.get("Gene") or row.get("gene")
            v = row.get("log2FoldChange", row.get("log2FC", ""))
            if g and v not in ("", "NA") and g not in lfc:
                lfc[g] = v; rp[g] = row.get("pvalue", "")
    n = 0
    with open(tfs_txt) as fi, open(out_path, "w") as fo:
        hdr = fi.readline().rstrip("\n").split("\t")
        fo.write("\t".join(hdr + ["Fold(DiffBind)", "p.value(DiffBind)", "log2FC", "RNA_pval"]) + "\n")
        gi = hdr.index("Gene Name")
        for line in fi:
            c = line.rstrip("\n").split("\t")
            if len(c) <= gi or c[gi] not in lfc:
                continue
            c += [""] * (len(hdr) - len(c))
            fo.write("\t".join(c[:len(hdr)] + [fold.get(c[0], ""), pval.get(c[0], ""), lfc[c[gi]], rp.get(c[gi], "")]) + "\n")
            n += 1
    return out_path


def register_matrix_join_tools(mcp):
    mcp.tool()(join_matrix_tfs_pval2)
