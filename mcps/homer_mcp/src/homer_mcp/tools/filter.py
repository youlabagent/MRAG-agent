"""filter_target_tfs + combine_replicates_minmax — pure pandas tools.

These are the lightweight Top-N fork point: ``filter_target_tfs`` runs once
per (sample, N) cutoff and produces the per-TF peak table that downstream
tools (mcp_diffbind for overlap, mcp_alphagenome for in silico validation)
consume.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from mcp.server.fastmcp import FastMCP


# Standard HOMER annotatePeaks columns (positional, before motif columns).
_HOMER_BASE_COLS = [
    "PeakID", "Chr", "Start", "End", "Strand", "Peak Score",
    "Focus Ratio/Region Size", "Annotation", "Detailed Annotation",
    "Distance to TSS", "Nearest PromoterID", "Entrez ID",
    "Nearest Unigene", "Nearest Refseq", "Nearest Ensembl",
    "Gene Name", "Gene Alias", "Gene Description", "Gene Type",
    "CpG%", "GC%",
]
_N_BASE = len(_HOMER_BASE_COLS)  # 21

# Columns we keep in the filtered output for downstream tools.
_KEEP_COLS = [
    "PeakID", "Chr", "Start", "End",
    "Annotation", "Distance to TSS", "Gene Name",
]



# Gene symbol -> HOMER motif name aliases.
# HOMER's known motif DB uses common-name labels that don't always match
# HGNC gene symbols. Without this map, e.g. NR3C1 (gene symbol) would not
# match HOMER's "GRE(NR),IR3" motif column. Mirrors the reference analysis's substring-match
# convention but as an explicit lookup so we keep exact matching elsewhere.
HOMER_GENE_ALIASES = {
    "TFAP2A":  ["AP-2alpha"],
    "TFAP2C":  ["AP-2gamma"],
    "NR3C1":   ["GRE"],            # glucocorticoid receptor element
    "TWIST1":  ["Twist"],
    "TWIST2":  ["Twist2"],
    "NR2F1":   ["COUP-TFI"],         # K562-NR2F1-ChIP labeled under COUP-TFI / COUP-TFII
    "NR2F2":   ["COUP-TFII"],
    "E2F1":    ["E2F1"],
    "E2F2":    ["E2F"],                # no E2F2-specific HOMER motif; fall back to generic E2F family
    "E2F3":    ["E2F3"],
    "E2F4":    ["E2F4"],
    "NR2C2":   ["EAR2"],
    "POU5F1":  ["Oct4"],
    "POU2F1":  ["Oct2"],
    "POU2F2":  ["Oct2"],           # both POU2 genes use Oct2 motif in HOMER
    "AR":      ["AR-halfsite", "ARE"],
    "ESR1":    ["ERE"],            # estrogen response element
    "RBPJ":    ["RBPJ", "CSL"],
    "MYC":     ["c-Myc"],
    "JUN":     ["Fos"],
    "EBF1":    ["EBF1", "EBF"],
    "TCF7L1":  ["TCF7L1", "TCF"],
    "TCF7L2":  ["TCF7L2", "TCF"],
}

def _extract_motif_tf(motif_col: str) -> str:
    """Extract TF symbol from a HOMER motif column header.

    Examples:
      'Oct2(POU,Homeobox)/Bcell-Oct2-ChIP-Seq...' -> 'Oct2'
      'PPARa(NR),DR1/Liver-Ppara-ChIP-Seq...'     -> 'PPARa'
      'GRE(NR),IR3/A549-GR-ChIP-Seq...'           -> 'GRE'
    """
    # Take the substring before '(' as the symbol.
    head = motif_col.split("/", 1)[0]
    m = re.match(r"\s*([A-Za-z0-9\-\.]+)", head)
    return m.group(1) if m else head.strip()


def _match_tf_columns(motif_columns: List[str], tfs: List[str],
                      case_insensitive: bool = True,
                      use_aliases: bool = True) -> Dict[str, List[str]]:
    """Match TF gene symbols against HOMER motif column headers.

    Strategy:
      1. Exact prefix-symbol match on the TF-symbol of each motif column
         (case-insensitive when requested).
      2. If `use_aliases`, also check the HOMER_GENE_ALIASES table -- this
         covers gene symbols that HOMER labels differently (e.g. NR3C1
         is GRE, TFAP2A is AP-2alpha, TWIST1 is Twist).

    Returns ``{tf: [matching column header, ...]}``.
    """
    norm = (lambda s: s.upper()) if case_insensitive else (lambda s: s)
    tfs_n = [norm(t) for t in tfs]

    out: Dict[str, List[str]] = {t: [] for t in tfs}
    for col in motif_columns:
        sym = norm(_extract_motif_tf(col))
        for orig, t in zip(tfs, tfs_n):
            # Direct prefix-symbol match
            if sym == t:
                out[orig].append(col)
                continue
            # Alias-based match (case-insensitive)
            if use_aliases:
                aliases = HOMER_GENE_ALIASES.get(orig, [])
                if any(norm(a) == sym for a in aliases):
                    out[orig].append(col)
    return out


def register_filter_tools(mcp: FastMCP):

    @mcp.tool()
    def filter_target_tfs(
        motif_tsv: str,
        tfs: Optional[List[str]] = None,
        output_tsv: Optional[str] = None,
        only_promoter: bool = True,
        case_insensitive: bool = True,
        keep_extra_cols: Optional[List[str]] = None,
        output_txt: Optional[str] = None,
        target_tfs: Optional[List[str]] = None,
    ) -> str:
        """Subset a HOMER annotatePeaks output to selected TF columns.

        2026-09-16: the min-max motif-table filter is ``filter_target_tfs_minmax``.
        The 2026-09-12 rename had given it this same name, so a call shaped
        like it (``output_txt`` / ``target_tfs``) is forwarded there; processes
        started before the fix still call it that way.

        Mirrors the reference analysis's STEP3_TargetTfs logic, generalised:
          * Locate columns whose motif name prefix equals any of ``tfs``.
          * Keep only rows where ``Annotation`` contains 'promoter'
            (when ``only_promoter=True``).
          * Emit a TSV with PeakID/Chr/Start/End/Annotation/
            Distance to TSS/Gene Name + matched motif columns.

        Args:
            motif_tsv:    Path to HOMER annotatePeaks output TSV (the .motif.tsv).
            tfs:          List of TF gene symbols to keep (e.g. ["ETV1","FOXA1"]).
                          The caller is typically the Top-N output from
                          combined_mcp.pick_topn.
            output_tsv:   Where to write the filtered TSV.
            only_promoter:If True (default, matches the reference analysis), drop
                          rows whose Annotation does not contain 'promoter'.
            case_insensitive: Match TF symbols case-insensitively (default True).
            keep_extra_cols:  Additional HOMER base columns to keep (e.g.
                          ["Detailed Annotation", "Peak Score"]).

        Returns JSON with the matched-column report (TF -> matched headers),
        row counts before/after promoter filter, output path, and a small
        sample of the matched columns for sanity.
        """
        if output_txt is not None and output_tsv is None:
            return filter_target_tfs_minmax(motif_tsv=motif_tsv, output_txt=output_txt,
                                            target_tfs=target_tfs if target_tfs is not None else tfs)
        if tfs is None or output_tsv is None:
            raise TypeError("filter_target_tfs needs tfs and output_tsv")
        path = Path(motif_tsv).expanduser()
        if not path.is_file():
            return json.dumps({"status": "error", "message": f"not found: {path}"})
        out = Path(output_tsv).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)

        # Read header (we don't load all rows yet — file can be 500+MB).
        with path.open() as f:
            header = f.readline().rstrip("\n").split("\t")
        if len(header) <= _N_BASE:
            return json.dumps({
                "status": "error",
                "message": f"input has only {len(header)} cols; expected at least {_N_BASE + 1}",
            })

        motif_cols = header[_N_BASE:]
        n_motifs   = len(motif_cols)
        matches    = _match_tf_columns(motif_cols, list(tfs), case_insensitive)
        matched_cols: List[str] = [c for cs in matches.values() for c in cs]
        if not matched_cols:
            return json.dumps({
                "status":  "error",
                "message": "no motif columns matched the requested TFs",
                "tfs":     list(tfs),
                "n_motif_cols": n_motifs,
                "examples": motif_cols[:5],
            }, ensure_ascii=False, indent=2)

        # Build keep-list — base cols + matched motif cols
        keep_base = list(_KEEP_COLS)
        if keep_extra_cols:
            for c in keep_extra_cols:
                if c in _HOMER_BASE_COLS and c not in keep_base:
                    keep_base.append(c)
        keep_all = keep_base + matched_cols

        # The HOMER file uses raw header (col 0 starts with 'PeakID (cmd=...)').
        # Normalise: read header from file and remap PeakID-with-cmd to 'PeakID'.
        # We do this on a thin pandas pass using usecols.
        # First rewrite header for pandas matching.
        header_norm = list(header)
        if header_norm[0].startswith("PeakID"):
            header_norm[0] = "PeakID"

        # usecols = column INDEXES matching keep_all
        col_idx_map = {c: i for i, c in enumerate(header_norm)}
        missing = [c for c in keep_all if c not in col_idx_map]
        if missing:
            return json.dumps({
                "status":  "error",
                "message": f"these columns not found in header: {missing}",
            })
        keep_idx = [col_idx_map[c] for c in keep_all]
        keep_names = [header_norm[i] for i in keep_idx]

        # Stream-read the TSV (only the keep columns).
        # 2026-09-26 memfix: in row chunks. Without chunksize the C parser
        # tokenized the WHOLE file -- all ~2000 motif columns, 5-9 GB for the
        # min-max tables -- before usecols dropped them (peak RSS ~ file size).
        # Each chunk tokenizes MRAG_FILTER_CHUNK_ROWS rows (default 64) and
        # keeps only the usecols; same parser and options, dtype=str (no type
        # inference across chunks) -> same frame, same bytes written.
        _read_kw = dict(
            sep="\t", header=None,   # header=0 + skiprows=[0] dropped the first data row
            usecols=keep_idx,
            names=header_norm,        # use normalised names
            skiprows=[0],             # skip original header row
            low_memory=False,
            dtype=str,                # keep raw — downstream parses as needed
        )
        try:
            _chunk_rows = max(1, int(os.environ.get("MRAG_FILTER_CHUNK_ROWS", "") or 64))
        except ValueError:
            _chunk_rows = 64
        with pd.read_csv(path, chunksize=_chunk_rows, **_read_kw) as _reader:
            _parts = [_c for _c in _reader]
        if _parts:
            df = pd.concat(_parts, ignore_index=True) if len(_parts) > 1 else _parts[0]
        else:                         # no data rows: the former single read
            df = pd.read_csv(path, **_read_kw)
        del _parts
        df = df[keep_names]           # enforce order

        n_in = len(df)
        if only_promoter:
            df = df[df["Annotation"].fillna("").str.contains("promoter", case=False, na=False)]
        n_out = len(df)

        df.to_csv(out, sep="\t", index=False)
        return json.dumps(
            {
                "status":            "ok",
                "input":             str(path),
                "output":            str(out),
                "n_motif_cols_total": n_motifs,
                "tfs_queried":       list(tfs),
                "matched_per_tf":    matches,
                "n_matched_cols":    len(matched_cols),
                "matched_cols":      matched_cols,
                "only_promoter":     only_promoter,
                "n_rows_input":      int(n_in),
                "n_rows_filtered":   int(n_out),
                "kept_columns":      keep_names,
                "preview":           df.head(3).to_dict(orient="records"),
            },
            ensure_ascii=False, indent=2, default=str,
        )

    @mcp.tool()
    def combine_replicates_join(
        tsv_a: str,
        tsv_b: str,
        output_tsv: str,
        gene_col: str = "Gene Name",
        chr_col: str = "Chr",
    ) -> str:
        """Join two replicate filtered tables by (Chr, Gene Name).

        Exact mirror of ``centered_peak_rev.py::new()``: for two
        replicate filtered tables (output of filter_target_tfs), match
        peaks that share a gene name on the same chromosome and emit one
        row per matched pair carrying both replicates' coordinates and
        per-TF columns.

        No scaling. The output filename the reference workflow used
        (e.g. ``cMyc_DDK_peak_minmax.txt``) is misleading -- the script
        does not implement min-max; that step lives at the TF-level
        upstream (``mcp_combined.minmax_scale_integrated``, 
       ``1.minman.R``).

        Args:
            tsv_a, tsv_b: Filtered TSVs (two replicates of one condition).
            output_tsv:   Output path.
            gene_col, chr_col: Column names to join on (defaults match
                              filter_target_tfs output).

        Returns JSON with merged row count + path.
        """
        a = pd.read_csv(Path(tsv_a).expanduser(), sep="\t", dtype=str)
        b = pd.read_csv(Path(tsv_b).expanduser(), sep="\t", dtype=str)
        for df, lbl in ((a, "tsv_a"), (b, "tsv_b")):
            for c in (gene_col, chr_col):
                if c not in df.columns:
                    return json.dumps({"status": "error",
                                       "message": f"{lbl} missing column '{c}'"})

        # Suffix rename for the merge -- everything except join keys gets _A / _B
        a2 = a.rename(columns={c: f"{c}_A" for c in a.columns if c not in (chr_col, gene_col)})
        b2 = b.rename(columns={c: f"{c}_B" for c in b.columns if c not in (chr_col, gene_col)})
        merged = pd.merge(a2, b2, on=[chr_col, gene_col], how="inner")

        out = Path(output_tsv).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out, sep="\t", index=False)
        return json.dumps(
            {
                "status":     "ok",
                "output":     str(out),
                "rows_a":     int(len(a)),
                "rows_b":     int(len(b)),
                "rows_merged": int(len(merged)),
                "join_keys":  [chr_col, gene_col],
                "preview":    merged.head(3).to_dict(orient="records"),
            },
            ensure_ascii=False, indent=2, default=str,
        )

    @mcp.tool()
    def combine_replicates_minmax_bed(
        tsv_a: str,
        tsv_b: str,
        output_bed: str,
        gtf_path: str,
        peak_id_prefix: str = "peak",
        gene_col: str = "Gene Name",
        chr_col:  str = "Chr",
        start_col: str = "Start",
        end_col:   str = "End",
        annotation_col: str = "Annotation",
        promoter_keyword: str = "promoter-TSS",
        max_union_width_bp: int = 10_000_000,
    ) -> str:
        """Combine two replicate HOMER motif.tsv → peak-region UNION BED.

        Mirrors ``Step2/1.minmax.py::min_max_Group{1,2}`` (2026-06-04).
        For each gene shared between rep1 and rep2 on the same chromosome,
        emits ONE BED row per (rep1_peak × rep2_peak) match with coordinates
        equal to ``(min(start_a, start_b), max(end_a, end_b))`` — the UNION
        of both replicate peak regions. This is the algebraic simplification
        of the reference script's 4-case if/elif ladder (all four cases collapse to min/max).

        Strand is looked up per gene from the GTF annotation. Rows are
        restricted to ``promoter-TSS`` peaks only.

        Output BED columns (tab-separated, no header):
            PeakID   Chr   Start   End   Strand

        Args:
            tsv_a, tsv_b:   Replicate HOMER annotatePeaks outputs (.motif.tsv).
            output_bed:     Destination BED file.
            gtf_path:       Ensembl-style GTF for gene_name → strand mapping.
            peak_id_prefix: Prefix for PeakID column (e.g. "ONECUT2_peak").
            gene_col, chr_col, start_col, end_col, annotation_col:
                            Column names (defaults = HOMER convention).
            promoter_keyword: Substring filter on annotation_col.

        Returns JSON with row counts and path.
        """
        import re as _re

        # 1. Parse GTF for gene_name → strand
        strand_map: dict[str, str] = {}
        name_rx = _re.compile(r'gene_name\s+"([^"]+)"')
        with open(gtf_path) as fp:
            for line in fp:
                if line.startswith("#"):
                    continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 9 or cols[2] != "gene":
                    continue
                m = name_rx.search(cols[8])
                if m:
                    strand_map[m.group(1)] = cols[6]

        # 2. Read only needed columns — avoids loading 2300+ motif cols (~4-5 GB)
        _needed = [chr_col, start_col, end_col, annotation_col, gene_col]
        a = pd.read_csv(Path(tsv_a).expanduser(), sep="\t", dtype=str,
                        low_memory=False, usecols=_needed)
        for c in _needed:
            if c not in a.columns:
                return json.dumps({"status": "error",
                                   "message": f"tsv_a missing column '{c}'"})
        a_prom = a[a[annotation_col].fillna("").str.contains(
            promoter_keyword, regex=False)].copy()

        # 3. Read rep2 — keep promoter peaks
        b = pd.read_csv(Path(tsv_b).expanduser(), sep="\t", dtype=str,
                        low_memory=False, usecols=_needed)
        for c in _needed:
            if c not in b.columns:
                return json.dumps({"status": "error",
                                   "message": f"tsv_b missing column '{c}'"})
        b_prom = b[b[annotation_col].fillna("").str.contains(
            promoter_keyword, regex=False)].copy()

        # 4. Vectorized union: merge on (gene, chr), compute min(start)/max(end)
        a_s = a_prom[[gene_col, chr_col, start_col, end_col]].copy()
        b_s = b_prom[[gene_col, chr_col, start_col, end_col]].copy()
        for df in (a_s, b_s):
            df[start_col] = pd.to_numeric(df[start_col], errors="coerce")
            df[end_col]   = pd.to_numeric(df[end_col],   errors="coerce")
        a_s = a_s.dropna(subset=[start_col, end_col])
        b_s = b_s.dropna(subset=[start_col, end_col])

        merged = b_s.merge(a_s, on=[gene_col, chr_col], suffixes=("_b", "_a"))

        out = Path(output_bed).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)

        if merged.empty:
            out.write_text("")
            return json.dumps({"status": "ok", "output": str(out),
                               "rows_a_promoter": int(len(a_prom)),
                               "rows_b_promoter": int(len(b_prom)),
                               "rows_emitted": 0,
                               "n_genes_unmatched_strand": 0,
                               "n_genes_in_strand_map": len(strand_map)},
                              ensure_ascii=False, indent=2, default=str)

        merged["s_union"] = merged[[f"{start_col}_a", f"{start_col}_b"]].min(axis=1).astype(int)
        merged["e_union"] = merged[[f"{end_col}_a",   f"{end_col}_b"]  ].max(axis=1).astype(int)
        merged["strand"]  = merged[gene_col].map(strand_map)
        n_genes_unmatched_strand = int(merged["strand"].isna().sum())
        merged = merged.dropna(subset=["strand"])

        # Drop pathological cross-locus unions. Multi-copy genes (e.g. the
        # snRNA RNU1-4, snoRNA/GOLGA clusters) share ONE symbol across many
        # genomic loci; when rep1/rep2 peaks land on DIFFERENT copies of the
        # same symbol on the same chromosome, the min/max union spans mega-
        # bases (observed: RNU1-4 = 126.8 Mb on chr1). No real gene locus is
        # wider than ~2.3 Mb (DMD); such unions are artifacts AND overflow
        # HOMER homer2's sequence buffer ("ERROR: outstripped buffer!"),
        # yielding a 0-byte annotatePeaks output. Cap at max_union_width_bp
        # (default 10 Mb: keeps reference-parity multi-copy peaks up to ~6 Mb that
        # annotatePeaks handles fine, drops only the crash-inducing outliers).
        _uw = (merged["e_union"] - merged["s_union"])
        n_dropped_wide = int((_uw > max_union_width_bp).sum())
        if n_dropped_wide:
            merged = merged[_uw <= max_union_width_bp]

        n_rows_emitted = len(merged)
        bed_rows = merged.reset_index(drop=True)
        bed_rows.insert(0, "_PeakID", [f"{peak_id_prefix}{i+1}" for i in range(n_rows_emitted)])
        bed_rows[["_PeakID", chr_col, "s_union", "e_union", "strand"]].to_csv(
            out, sep="\t", index=False, header=False
        )

        return json.dumps(
            {
                "status":            "ok",
                "output":            str(out),
                "rows_a_promoter":   int(len(a_prom)),
                "rows_b_promoter":   int(len(b_prom)),
                "rows_emitted":      n_rows_emitted,
                "n_dropped_wide":    n_dropped_wide,
                "n_genes_unmatched_strand": n_genes_unmatched_strand,
                "n_genes_in_strand_map":    len(strand_map),
            },
            ensure_ascii=False, indent=2, default=str,
        )

    @mcp.tool()
    def filter_target_tfs_minmax(
        motif_tsv: str,
        output_txt: str,
        target_tfs: Optional[List[str]] = None,
        promoter_keyword: str = "promoter",
        match_mode: str = "alias",
    ) -> str:
        """Reference-exact target TF filter (mirrors ``2.findmotif_homer.py::Step2_TargetTFs``).

        2026-09-10: ``match_mode="alias"`` (default) assigns a motif column to a
        target TF through homer_mcp.tools.tf_alias (ANDR/ARE/AR-halfsite -> AR,
        GCR/GRE -> NR3C1, AP-2alpha -> TFAP2A, dimers/composites count for each
        member; case-insensitive). ``match_mode="substring"`` restores the
        reference analysis's original ``tf in column_header`` rule
        (over-inclusive: "AR" also hit ARNT/PPARG columns and missed ANDR/ARE).

        DEFAULT behavior reproduces a legacy reference workflow (2026-06-04) exactly:
          * Default TFs = ["ELF5","EHF","E2F2","EGR1","ETV5","KLF15","MAFB",
                          "ETV1","GATA3","MEF2C"]
          * TF column matching = ``substring`` of TF name anywhere in the
            motif column header (the reference analysis's
            ``any(tf in column_header)``).
          * Column header transform: strip the trailing
           ``/Homer Distance From Peak(sequence,strand,conservation)``.
          * Base columns (by FIXED HOMER position):
                [0] PeakID, [1] Chr, [2] Start, [3] End,
                [7] Annotation, [9] Distance to TSS, [15] Gene Name
          * Body filter: ``'promoter' in line[7]`` (any promoter annotation).
          * Output: tab-separated text (.txt) — NOT a CSV.

        Args:
            motif_tsv:        HOMER annotatePeaks output (.motif.tsv).
            output_txt:       Where to write filtered table (the reference analysis uses .TFs.txt).
            target_tfs:       Override the 10-TF default list.
            promoter_keyword: Substring filter for column [7] (default "promoter").

        Returns JSON with matched TF columns + row counts + output path.
        """
        DEFAULT_TFS = ["ELF5","EHF","E2F2","EGR1","ETV5","KLF15",
                       "MAFB","ETV1","GATA3","MEF2C"]
        if target_tfs is None:
            target_tfs = DEFAULT_TFS

        path = Path(motif_tsv).expanduser()
        if not path.is_file():
            return json.dumps({"status":"error", "message": f"not found: {path}"})
        out = Path(output_txt).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)

        HOMER_DIST_SUFFIX = (
            "/Homer Distance From Peak(sequence,strand,conservation)"
        )
        BASE_INDICES   = [0, 1, 2, 3, 7, 9, 15]
        BASE_HEADER    = ["PeakID","Chr","Start","End",
                          "Annotation","Distance to TSS","Gene Name"]

        tf_col_indices: List[int] = []
        matched_headers: List[str] = []
        matched_by_tf: Dict[str, List[str]] = {tf: [] for tf in target_tfs}
        n_in   = 0
        n_kept = 0

        with path.open() as fp, out.open("w") as fpout:
            for line in fp:
                n_in += 1
                if n_in == 1:
                    cols = line.rstrip("\n").split("\t")
                    # the reference loops range(1, len(cols)); we scan all motif cols
                    # (the base cols never contain TF symbols so it doesn't
                    # matter, but mirror exactly):
                    if match_mode == "alias":
                        from .tf_alias import motif_matches as _mm
                    for k in range(1, len(cols)):
                        for tf in target_tfs:
                            _hit = (_mm(cols[k].split(HOMER_DIST_SUFFIX)[0], tf)
                                    if match_mode == "alias" else (tf in cols[k]))
                            if _hit:
                                tf_col_indices.append(k)
                                stripped = cols[k].split(HOMER_DIST_SUFFIX)[0]
                                matched_headers.append(stripped)
                                matched_by_tf[tf].append(stripped)
                                break    # one TF per column (reference-analysis behaviour)
                    fpout.write(
                        "\t".join(BASE_HEADER)
                        + ("\t" + "\t".join(matched_headers)
                           if matched_headers else "")
                        + "\n"
                    )
                else:
                    cols = line.rstrip("\n").split("\t")
                    if len(cols) <= max(BASE_INDICES + tf_col_indices, default=15):
                        continue
                    if promoter_keyword not in cols[7]:
                        continue
                    base_vals = [cols[i] if i < len(cols) else "" for i in BASE_INDICES]
                    tf_vals   = [cols[i] if i < len(cols) else "" for i in tf_col_indices]
                    fpout.write(
                        "\t".join(base_vals)
                        + ("\t" + "\t".join(tf_vals) if tf_vals else "")
                        + "\n"
                    )
                    n_kept += 1

        return json.dumps(
            {
                "status":            "ok",
                "output":            str(out),
                "n_rows_input":      n_in - 1,
                "n_rows_filtered":   n_kept,
                "n_matched_tfs":     sum(1 for v in matched_by_tf.values() if v),
                "matched_by_tf":     matched_by_tf,
                "target_tfs":        target_tfs,
                "match_mode":        match_mode,
            },
            ensure_ascii=False, indent=2, default=str,
        )
