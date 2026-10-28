"""Curated target-gene selection node (feeds Section 10 graph, Section 11
enrichment and Section 14 narrative / notable findings).

Replaces the auto-derived per-TF top-12 (derive_targets_summary / Stouffer) with
the evidence-driven curated pipeline. The motif/driver PANEL is the Section-3
"TF Ranking (Top 10 by Combined Score (Avg.Z))" top-6 (== top6.json), NOT the
legacy hardcoded 10-TF Pval2 panel.

Pipeline:
  1. build a panel-6 motif table (like the manual .TFs.Pval2.top6.txt): the
     legacy .TFs.Pval2.txt only carries the hardcoded 10-TF columns and is
     MISSING any panel TF outside those 10. We therefore stream the top-20
     .motif.tsv (which always contains the top-6 panel), keep the panel motif
     columns (canonical HOCOMOCO/JASPAR token match), and join them to the
     promoter-filtered Pval2 rows (Chr/Start/End/Gene Name/Fold(DiffBind)/log2FC).
     Cached to homer_motif/{treat}.TFs.Pval2.top6.txt.
  2. pool-N candidate genes: RNA log2FC > lfc_min, representative peak binding,
     union of panel TFs over the gene's peaks >= min_union, ranked log2FC desc.
  3. per-TF motif position per (gene, panel-TF): best instance in that TF's
     column inside the representative peak (gm_start = peak_start + offset).
  4. candidates: every pool gene with a bound panel-TF motif.
  5. evidence retrieval (GeneRIF + paperQA vectordb) + LLM domain scoring 0-10,
     the domain of interest = the run's user LITERATURE domain(s) + condition
     perturbation/concept, the experimental model (cell line, cancer_type)
     stated as context (2026-09-26; MRAG_SCORE_DOMAIN_FRAME=study restores
     the old cancer_type + cell-line frame).
  6. rerank by (score desc, log2FC desc), exclude panel TFs, keep those at or
     above min_score (when set), cap at top_final.

Emits, under the branch dir the narrative reads:
    targets_summary.json     {TF: [{Gene Name, Z_mean, db_Fold_mean, RNA_log2FC_mean}]}
    curated_targets.json     ordered [{gene,score,log2FC,drivers,tfs}]; drivers = the
                             gene's bound panel TFs (pool_motifs.json)
"""
from __future__ import annotations
import json, os, re, subprocess, sys
from pathlib import Path
from ._base import Node
from . import evidence_gate as _evgate   # 2026-09-23 evidence-grounding gate
from . import _stage_tools as _stage   # MCP step 2a (2026-10-12): this node's stages as MCP tools
import sys as _sys_alias
_sys_alias.path.insert(0, "/BIO1/MRAG/Agent/mcps/homer_mcp/src")
from homer_mcp.tools.tf_alias import motif_matches as _mmatch, motif_symbols as _msym   # 2026-09-10 alias-aware TF matching

_INST_RE = re.compile(r'(-?\d+)\(([ACGTNacgtn]+),([+\-]),([-\d.eE]+)\)')

# ---------------------------------------------------------------------------
# curated evidence retrieval through the Section-14 card retriever (2026-09-23)
# ---------------------------------------------------------------------------
# WHY.  The scorer used to run its OWN pooled query -- rt.search("<GENE> in
# <cancer>", top_k=6, cancer=[<domains>, "pancancer"]) -- so a bare gene symbol
# had to win a dense-similarity contest against whole-abstract chunks.  It
# mostly lost: over the 249 curated genes only 5 (NEC-only) / 6 (SCLC-only) /
# 17 (PC-only) / 20 (3-domain) genes ended up with a snippet that BOTH names the
# gene and ties to the run's domain, against corpus ceilings of 40 / 37 / 80 /
# 90.  Section 14's card retriever already solved this: one corpus-filtered
# query PER DOMAIN, restricted to chunks whose text actually contains the gene
# symbol (_symbol_spellings), with the plain corpus query as the fallback when
# the prefilter finds nothing.  Reusing it gives 24 / 23 / 64 / 73 at k=3 and
# 25 / 23 / 68 / 75 at k=6 -- a strict superset, no gene loses evidence -- and
# it is FASTER, because the card runner's numpy fast path replaces the Qdrant
# API's per-query Python payload loop.
#
# ONE retriever: this calls narrative.batch_retrieve_card_hits, the same
# function Section 14 calls, in the same paper_qa subprocess shape.  Nothing
# here re-implements retrieval.
VIZ_REPORT_SRC = "/BIO1/MRAG/Agent/mcps/viz_report_mcp/src"
CURATED_CARD_TOP_K = 3          # per corpus; 2 corpora x 3 == the old pooled 6
CURATED_GENERIF_RETMAX = 400    # free: ONE efetch returns the whole gene record


def card_retrieval_enabled(param=None):
    """Card-retriever path on unless switched off.  Param wins over
    MRAG_CURATED_CARD_RETRIEVAL; the old pooled in-runner retrieval is still
    there and is what runs when this is off or when no domain resolves."""
    if param is not None:
        return bool(param)
    return _evgate._env_flag("MRAG_CURATED_CARD_RETRIEVAL", True)


def _int_param(param, env, default):
    for v in (param, os.environ.get(env)):
        if v is None or str(v).strip() == "":
            continue
        try:
            return max(1, int(v))
        except (TypeError, ValueError):
            pass
    return default


def card_top_k(param=None):
    return _int_param(param, "MRAG_CURATED_CARD_TOP_K", CURATED_CARD_TOP_K)


def generif_retmax(param=None):
    return _int_param(param, "MRAG_CURATED_GENERIF_RETMAX", CURATED_GENERIF_RETMAX)


# evaudit 2026-09-27: GeneRIF lines are shown to the scorer in full up to this
# many characters (was a fixed 200: one candidate's GeneRIF line lost its
# closing "no difference" result at the cut and the LLM read it as the gene
# "predicting recurrence").  MRAG_GENERIF_SHOW_CHARS=200 restores the old cut.
CURATED_GENERIF_SHOW_CHARS = 600


def generif_show_chars(param=None):
    # fix2 2026-09-28: the master switch MRAG_EVIDENCE_DEFENCE=0 restores the
    # production 200-character cut
    if hasattr(_evgate, "evidence_defence_on") and not _evgate.evidence_defence_on():
        return 200
    return _int_param(param, "MRAG_GENERIF_SHOW_CHARS", CURATED_GENERIF_SHOW_CHARS)


def _vd_record(h, gene, gene_name, dspecs, slot):
    """One card hit -> the vdb record scored.json has always carried, with the
    naming and domain decisions taken HERE on the FULL chunk text (gate_text),
    because only title[:120] + snippet[:220] is persisted."""
    full = str(h.get("title") or "") + " " + str(h.get("gate_text") or h.get("snippet") or "")
    cs = h.get("cancer")
    cs = ({str(x).strip().lower() for x in cs if str(x or "").strip()}
          if isinstance(cs, (list, tuple, set))
          else ({str(cs).strip().lower()} if str(cs or "").strip() else set()))
    rec = {"pmid": h.get("pmid"), "cancer": h.get("cancer"),
           "score": h.get("score"),
           "title": str(h.get("title") or "")[:120],
           "snippet": str(h.get("snippet") or "")[:220].replace(chr(10), " "),
           "gene_mention": bool(gene.upper() in full.upper()),
           "names_gene": _evgate.names_gene(full, gene, gene_name),
           "slot": slot}
    # findfix 2026-09-26 (finding 2): the passage the scorer LLM is shown, cut
    # from the FULL chunk around the gene and the nearest domain mention
    # (evidence_gate.gene_excerpt) instead of the chunk's first 160 characters.
    _ex = _evgate.gene_excerpt(str(h.get("gate_text") or h.get("snippet") or ""),
                               gene, gene_name, dspecs)
    if _ex:
        rec["excerpt"] = _ex
    if dspecs:
        # FIX 2026-09-25 (E2): a corpus label alone no longer ties a chunk to a
        # domain that has DOMAIN_EXCLUDE rules (sclc, nec, ...): the sclc corpus
        # holds NSCLC papers.  tie_record() applies evidence_gate's rule on the
        # FULL chunk + title; domain_tie_v marks the decision as taken under it.
        rec["domain_tie"], _notes = _evgate.tie_record(
            full, cs, dspecs, gene=gene, gene_name=gene_name,
            title=str(h.get("title") or ""))
        # evaudit 2026-09-27: 4 = decided with the evidence filters (1b / 1c)
        rec["domain_tie_v"] = _evgate.tie_version()
        if _notes:
            rec["domain_tie_note"] = _notes
    return rec


def prefetch_card_hits(genes, dom_specs, corpora, suffix, top_k=None,
                       device="cpu", gene_info=None, lit_symbol=None,
                       lit_unmapped=(), timeout=None):
    """Retrieve the curated scorer's literature through Section 14's retriever.

    `dom_specs` are literature.domain_specs() dicts (label drives the query, as
    in the cards); `corpora` is the run's vdb_cancer list -- every key in it
    that is not a domain (in practice "pancancer") becomes an extra
    corpus-filtered slot queried with the run's own vdb_query_suffix, so the
    pooled path's pan-cancer reach is kept rather than traded away.

    Returns (vdb {run symbol: [record]}, stats).  vdb is None when the batch
    failed -- the caller then leaves the in-runner retrieval to do its job."""
    import time as _time
    if VIZ_REPORT_SRC not in sys.path:
        sys.path.insert(0, VIZ_REPORT_SRC)
    from viz_report_mcp.tools.narrative import batch_retrieve_card_hits
    k = card_top_k(top_k)
    lm = dict(lit_symbol or {})
    unm = set(lit_unmapped or ())
    todo = [g for g in genes if g not in unm]
    qsym = {g: lm.get(g, g) for g in todo}
    syms = list(dict.fromkeys(qsym[g] for g in todo))
    dkeys = [d["key"] for d in (dom_specs or [])]
    extra = [{"key": c, "cancer": c, "query": "{gene} " + str(suffix or "").strip(),
              "top_k": k, "prefilter": True}
             for c in (corpora or []) if c and c not in dkeys]
    slots = dkeys + [e["key"] for e in extra]
    stats = {"enabled": True, "used": False, "top_k": k, "slots": slots,
             "n_genes": len(syms), "n_queries": len(syms) * len(slots),
             "wall_s": None, "n_errors": 0, "n_fallback_plain": 0, "error": ""}
    if not syms or not slots:
        stats["error"] = "nothing to retrieve"
        return None, stats
    print("[curated] card retrieval: %d genes x %d corpus slots (k=%d) = %d "
          "gene-name-prefiltered searches" % (len(syms), len(slots), k,
                                              len(syms) * len(slots)), flush=True)
    t0 = _time.time()
    raw = batch_retrieve_card_hits(syms, dom_specs, device=device, cross=False,
                                   domain_top_k=k, extra_slots=extra,
                                   prefilter=True, timeout=timeout)
    stats["wall_s"] = round(_time.time() - t0, 1)
    dsp = _evgate.domain_specs(dom_specs)
    names, vdb = {}, {}
    n_err = 0
    for g in todo:
        q = qsym[g]
        rec = raw.get(q) or {}
        errs = rec.get("_errors") or {}
        n_err += sum(1 for v in errs.values() if v)
        for _s, _m in (rec.get("_modes") or {}).items():
            if _m == "plain_fallback":
                stats["n_fallback_plain"] += 1
        if q not in names:
            names[q] = _evgate.gene_fullname(q, gene_info)
        seen, keep = set(), []
        for slot in slots:
            for h in (rec.get(slot) or []):
                key = (str(h.get("pmid")), str(h.get("gate_text") or h.get("snippet") or "")[:80])
                if key in seen:
                    continue
                seen.add(key)
                keep.append(_vd_record(h, q, names[q], dsp, slot))
        # what the LLM sees is the first 3 that name the gene, so put the
        # domain-tied ones first -- same ordering rule the cards use.
        keep.sort(key=lambda r: (0 if (r.get("names_gene") and
                                       any((r.get("domain_tie") or {}).values())) else
                                 (1 if r.get("names_gene") else 2),
                                 slots.index(r["slot"]),
                                 -(r.get("score") or 0)))
        vdb[g] = keep
    for g in genes:
        vdb.setdefault(g, [])       # a gene with no human ortholog: not searched
    stats["n_errors"] = n_err
    stats["n_hits"] = sum(len(v) for v in vdb.values())
    if n_err and n_err >= len(syms) * len(slots):
        stats["error"] = "every card retrieval slot failed"
        return None, stats
    stats["used"] = True
    print("[curated] card retrieval done in %.1fs: %d hits, %d slot errors, "
          "%d prefilter fallbacks to the plain corpus query"
          % (stats["wall_s"], stats["n_hits"], n_err, stats["n_fallback_plain"]),
          flush=True)
    return vdb, stats



def _parse_cell(cell):
    """HOMER motif cell -> [(offset,seq,strand,score), ...]."""
    if not cell or cell in (".", "nan", "NaN"):
        return []
    return [(int(m.group(1)), m.group(2).upper(), m.group(3), float(m.group(4)))
            for m in _INST_RE.finditer(cell)]


# ----------------------------------------------------------------------------
# panel + panel-6 motif table
# ----------------------------------------------------------------------------
def load_panel(top6_json):
    d = json.loads(Path(top6_json).read_text())
    return list(d["tfs"])


def build_panel_table(motif_tsv, pval2_txt, panel, out_path):
    """Stream the top-20 .motif.tsv, keep the panel motif columns (canonical
    token match), join to promoter-filtered Pval2 rows. Writes a Pval2-style
    table:  Chr Start End PeakID 'Gene Name' <TF(base)/motif ...> Fold(DiffBind) log2FC.
    Returns (out_path, [panel TFs actually found])."""
    import csv
    sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/combined_mcp/src")
    from combined_mcp.tools.pipeline import HOCOMOCO_TO_GENE
    panel_up = {t.upper(): t for t in panel}

    def canon(tok):
        return HOCOMOCO_TO_GENE.get(tok.upper(), tok.upper())
    sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/homer_mcp/src")
    from homer_mcp.tools.tf_alias import motif_symbols as _msym   # alias-aware (2026-09-10)

    # 1. Pval2 -> pid -> (chr,start,end,gene,fold,log2fc); this is the
    # promoter-filtered peak set.
    meta = {}
    with open(pval2_txt) as fh:
        r = csv.reader(fh, delimiter="\t")
        h = next(r)
        ci = {c: h.index(c) for c in
              ["PeakID", "Chr", "Start", "End", "Gene Name", "Fold(DiffBind)", "log2FC"]}
        for row in r:
            if len(row) <= ci["log2FC"]:
                continue
            meta[row[ci["PeakID"]]] = (row[ci["Chr"]], row[ci["Start"]], row[ci["End"]],
                                       row[ci["Gene Name"]], row[ci["Fold(DiffBind)"]],
                                       row[ci["log2FC"]])
    need = set(meta)

    # 2. motif.tsv header -> panel motif column specs
    with open(motif_tsv) as f:
        header = f.readline().rstrip("\n").split("\t")
        specs = []  # (idx, TF, newheader)
        for i, hh in enumerate(header):
            if i == 0:
                continue
            base = hh.split("(")[0]
            # every panel member of a dimer / composite column gets it (FOXA1:AR -> AR and FOXA1)
            for cu in sorted(_msym(hh)):
                if cu in panel_up:
                    specs.append((i, panel_up[cu], panel_up[cu] + "(" + base + ")/motif"))
        idxs = [s[0] for s in specs]
        # 2026-09-26 memfix: a row that is not in the promoter-filtered Pval2 set
        # costs one partition() and is dropped unsplit; a kept row is split only
        # up to the last column read (panel motif columns, Strand). Held in
        # memory: the kept panel cells (~ the size of the written table) plus
        # the current line (HOMER -m rows reach 300 MB in the 2000-motif min-max
        # tables). Same values as the former full split("\t") -> same bytes.
        _nsplit = max(idxs + [4]) + 1
        motif_by_pid = {}
        strand_by_pid = {}
        for line in f:
            _head, _tab, _ = line.partition("\t")
            pid = _head if _tab else line.rstrip("\n")
            if pid not in need:
                continue
            cols = line.rstrip("\n").split("\t", _nsplit)
            motif_by_pid[pid] = [cols[i] if i < len(cols) else "" for i in idxs]
            strand_by_pid[pid] = cols[4] if len(cols) > 4 else "+"

    # 3. write panel table (Strand = HOMER peak strand; offsets are oriented by it)
    newcols = [s[2] for s in specs]
    with open(out_path, "w") as out:
        out.write("\t".join(["Chr", "Start", "End", "Strand", "PeakID", "Gene Name"]
                            + newcols + ["Fold(DiffBind)", "log2FC"]) + "\n")
        for pid, (chrom, st, en, gene, fold, lfc) in meta.items():
            mv = motif_by_pid.get(pid, [""] * len(idxs))
            strand = strand_by_pid.get(pid, "+")
            out.write("\t".join([chrom, st, en, strand, pid, gene] + mv + [fold, lfc]) + "\n")
    found = sorted({s[1] for s in specs})
    return out_path, found


def build_pval2_from_matrix(canonical_motif_tsv, canonical_peaks_bed,
                            diffbind_csv, deseq_csv, out_path):
    """Processed-matrix equivalent of the fastq-mode Pval2 table. Writes columns
    that build_panel_table reads: PeakID, Chr, Start, End, Gene Name,
    Fold(DiffBind), log2FC. Rows are promoter-filtered (Annotation startswith
    'promoter-TSS'). Fold is joined per-peak via canonical_peaks.bed
    name<->coords; log2FC per-gene from DESeq2.gene.csv. Returns (out_path, n)."""
    import csv
    # 1. bed: (chr,start,end) -> peak name (name == motif_tsv PeakID)
    coord2name = {}
    with open(canonical_peaks_bed) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            coord2name[(p[0], p[1], p[2])] = p[3]
    # 2. DiffBind: (chr,start,end) -> Fold -> name -> Fold
    fold_by_name = {}
    with open(diffbind_csv) as fh:
        r = csv.reader(fh)
        h = next(r)
        ci = {c: h.index(c) for c in ["seqnames", "start", "end", "Fold"]}
        for row in r:
            if len(row) <= ci["Fold"]:
                continue
            key = (row[ci["seqnames"]], row[ci["start"]], row[ci["end"]])
            name = coord2name.get(key)
            if name is not None:
                fold_by_name[name] = row[ci["Fold"]]
    # 3. DESeq2: GENE -> log2FoldChange
    lfc_by_gene = {}
    with open(deseq_csv) as fh:
        r = csv.reader(fh)
        h = next(r)
        gi = h.index("GENE"); li = h.index("log2FoldChange")
        for row in r:
            if len(row) <= li:
                continue
            v = row[li]
            if v in ("", "NA"):
                continue
            lfc_by_gene[row[gi]] = v
    # 4. stream motif_tsv, promoter filter, write pval2
    n = 0
    with open(canonical_motif_tsv) as f, open(out_path, "w") as out:
        out.write("\t".join(["PeakID", "Chr", "Start", "End", "Gene Name",
                             "Fold(DiffBind)", "log2FC"]) + "\n")
        f.readline()  # skip header
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 16:
                continue
            if not cols[7].startswith("promoter-TSS"):
                continue
            pid = cols[0]; gene = cols[15]
            fold = fold_by_name.get(pid, "")
            lfc = lfc_by_gene.get(gene, "")
            out.write("\t".join([pid, cols[1], cols[2], cols[3], gene, fold, lfc]) + "\n")
            n += 1
    return out_path, n


# ----------------------------------------------------------------------------
# pool + motif scan
# ----------------------------------------------------------------------------
def build_pool(panel_table, panel, lfc_min=2.0, min_union=3, pool_n=200):
    """stage1: pool candidate target genes (log2FC>lfc_min, union of panel TFs
    bound across the gene's peaks >= min_union; representative peak = max
    (#panel TFs bound in that row, |Fold|); ranked by log2FC desc)."""
    import pandas as pd
    pv = pd.read_csv(panel_table, sep="\t", dtype=str)
    cols = list(pv.columns)
    gi = cols.index("Gene Name")
    fdi = cols.index("Fold(DiffBind)")
    motif_cols = cols[gi + 1:fdi]
    pv["log2FC"] = pd.to_numeric(pv["log2FC"], errors="coerce")
    pv["Fold(DiffBind)"] = pd.to_numeric(pv["Fold(DiffBind)"], errors="coerce")
    pv["Start"] = pd.to_numeric(pv["Start"], errors="coerce")
    pv["End"] = pd.to_numeric(pv["End"], errors="coerce")

    def row_bound(r):
        b = set()
        for c in motif_cols:
            hit = [t for t in panel if _mmatch(c, t)]
            if hit and isinstance(r[c], str) and str(r[c]).strip() not in ("", "nan", "NaN"):
                b.update(hit)
        return b

    recs = []
    for gene, sub in pv.groupby("Gene Name"):
        l = sub["log2FC"].dropna()
        if l.empty:
            continue
        lfc = float(l.iloc[0])
        if lfc <= lfc_min:
            continue
        union = set()
        best = None
        for _, r in sub.iterrows():
            b = row_bound(r)
            union |= b
            fold = r["Fold(DiffBind)"]
            fold = 0.0 if pd.isna(fold) else float(fold)
            cand = (len(b), abs(fold), r["Chr"], r["Start"], r["End"], sorted(b), fold)
            if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                best = cand
        if len(union) < min_union or best is None:
            continue
        recs.append({"gene": gene, "log2FC": round(lfc, 3), "n_tf": len(union),
                     "tfs": sorted(union), "rep_tfs": best[5], "chrom": best[2],
                     "start": int(best[3]), "end": int(best[4]),
                     "atac_fold": round(best[6], 3)})
    recs.sort(key=lambda d: -d["log2FC"])
    return recs[:pool_n]


def _panel_table_columns(header, gi, fdi, panel):
    cols = {t: [] for t in panel}
    up = {t.upper(): t for t in panel}
    for i in range(gi + 1, fdi):
        for sym in sorted(_msym(header[i])):
            if sym in up:
                cols[up[sym]].append(i)
    return cols


def scan_motifs(pool, panel_table, panel, fasta_path=None):
    """Per pool gene's representative peak, per panel TF pick the best motif
    instance (score desc). HOMER offsets are oriented by the PEAK strand:
      '+' peak: gm = peak_start + offset - 1
      '-' peak: gm = peak_end   - offset - len(motif)
    ref_span is taken from the reference genome (forward) at gm when a fasta is
    given, else the HOMER motif-strand sequence. Returns
    {gene:{chrom,peak_start,peak_end,atac_fold,log2FC,tfs:{TF:{gm_start,ref_span}}}}."""
    import csv
    fa = None
    if fasta_path:
        try:
            from pyfaidx import Fasta
            fa = Fasta(fasta_path, sequence_always_upper=True)
        except Exception:
            fa = None
    want = {(d["chrom"], d["start"], d["end"]): d for d in pool}
    need = set(want)
    out = {}
    with open(panel_table) as fh:
        r = csv.reader(fh, delimiter="\t")
        header = next(r)
        gi = header.index("Gene Name")
        fdi = header.index("Fold(DiffBind)")
        ci_chr = header.index("Chr")
        ci_start = header.index("Start")
        ci_end = header.index("End")
        ci_str = header.index("Strand") if "Strand" in header else None
        pcols = _panel_table_columns(header, gi, fdi, panel)
        for row in r:
            if len(row) <= fdi:
                continue
            try:
                key = (row[ci_chr], int(row[ci_start]), int(row[ci_end]))
            except ValueError:
                continue
            if key not in need:
                continue
            d = want[key]
            pstart = d["start"]
            pend = d["end"]
            strand = row[ci_str] if ci_str is not None else "+"
            tfmot = {}
            for tf, idxs in pcols.items():
                best = None
                for cix in idxs:
                    for off, seq, st, score in _parse_cell(row[cix]):
                        cand = (score, -off, off, seq)
                        if best is None or cand > best:
                            best = cand
                if best is not None:
                    off = best[2]
                    seq = best[3]
                    L = len(seq)
                    if strand == "-":
                        gm = pend - off - L
                    else:
                        gm = pstart + off - 1
                    ref = seq
                    if fa is not None:
                        try:
                            ref = str(fa[d["chrom"]][gm:gm + L]).upper()
                        except Exception:
                            ref = seq
                    tfmot[tf] = {"gm_start": gm, "ref_span": ref}
            out[d["gene"]] = {"chrom": d["chrom"], "peak_start": d["start"],
                              "peak_end": d["end"], "atac_fold": d["atac_fold"],
                              "log2FC": d["log2FC"], "tfs": tfmot}
            need.discard(key)
            if not need:
                break
    return out

# ----------------------------------------------------------------------------
# rerank
# ----------------------------------------------------------------------------
def rerank(scored_rows, panel, top_final=15, min_score=None):
    """Rank scored genes by (evidence score desc, log2FC desc), minus the panel.

    `top_final` is a CAP, not a quota. With `min_score` set, only genes clearing
    that score keep a slot, so the final count follows the evidence: the list
    stops where the scorer stops finding gene-specific support rather than being
    padded with score-0/1 genes that would then be ordered by log2FC alone --
    which is exactly the selection rule the curated pipeline exists to replace.
    Falls back to the plain top_final ranking if nothing clears the floor, so a
    weak dataset degrades to the old behaviour instead of emptying Sections
    10/12/15.
    """
    ps = set(panel)
    targets = [r for r in scored_rows if r["gene"] not in ps]
    # Drop targets with no driver TF (empty `drivers`): there is no TF->target
    # edge to place in the network. Since ag_gate_removal 2026-10-18 `drivers` are
    # the gene's bound panel TFs (pool_motifs.json), so a scored gene always has one
    # and this only guards rows of another origin; before, they were the TFs whose
    # AlphaGenome scramble was direction-concordant.
    _n_before = len(targets)
    targets = [r for r in targets if r.get("drivers")]
    if len(targets) < _n_before:
        print("[curated] excluded %d target(s) with no driver TF"
              % (_n_before - len(targets)), flush=True)
    targets.sort(key=lambda r: (-r["score"], -r["log2FC"]))
    if min_score is not None:
        kept = [r for r in targets if (r.get("score") or 0) >= min_score]
        if kept:
            print("[curated] evidence floor score>=%s kept %d/%d (cap %d)"
                  % (min_score, len(kept), len(targets), top_final), flush=True)
            return kept[:top_final]
        print("[curated] evidence floor score>=%s kept 0 -- falling back to top-%d"
              % (min_score, top_final), flush=True)
    return targets[:top_final]


# ----------------------------------------------------------------------------
# fix2 2026-09-28: RE-SCORE a gene whose counted evidence an evidence filter
# removed and that was re-admitted on OTHER evidence
# fix3 2026-09-28 (second verification, finding 1): IN ITS ORIGINAL BATCH
# ----------------------------------------------------------------------------
# Verification of the evaudit build (findings 6 / 8): an SCLC gene lost its
# only counted passage to the screened-negative filter, the PubMed rescue
# re-admitted it on a different, live PubMed record (valid, PI decision
# 2026-09-27), and the row kept the LLM score and reason given on the
# removed passage.  fix2 re-scored such a gene ALONE, on its counted
# evidence only.  The second verification measured that this is biased LOW:
# the same model at temperature 0 scores a gene lower alone than inside its
# 15-gene batch (12 SCLC genes with unchanged evidence: mean -1.17, 9 lower,
# 0 higher; the rescued gene 4 alone vs 7 in its batch), i.e. a de-facto
# cap on rescued list genes, which the PI forbade.  Now the gene is re-scored
# INSIDE THE BATCH IT WAS SCORED IN: the same neighbours with exactly the
# blocks they were shown (the runner writes them to _score_batches.json), and
# its own block exactly as the runner showed it MINUS every item an evidence
# filter removed PLUS the verified rescue records.  The original batch
# reproduced all 15 stored scores in that measurement, so the new score is on
# the same scale as every other gene's.  Only the gene's own answer is taken.
# No batch context (a runner of another version) or a failed call: the old
# score is kept and the row is FLAGGED (evidence_flag "score not re-checked",
# shown in the report) -- never a silent keep.  Never a cap.  Rows: re-scored
# when a filter removed at least one domain-tied item that named the gene AND
# the gene is admitted (gate "pass" or "pubmed") AND none of the items it is
# admitted on was counted before the filters.  MRAG_RESCORE_AFTER_FILTER=0
# (node param rescore_after_filter), MRAG_EVIDENCE_DEFENCE=0 or
# MRAG_EVIDENCE_AUDIT=0 switches it off.
RESCORE_VERSION = 2
RESCORE_REC_CHARS = 480


def rescore_enabled(param=None):
    if hasattr(_evgate, "evidence_defence_on") and not _evgate.evidence_defence_on():
        return False
    v = param if param is not None else os.environ.get("MRAG_RESCORE_AFTER_FILTER")
    v = str(v if v is not None and str(v).strip() else "on").strip().lower()
    return v not in ("0", "off", "false", "no", "none")


def rescore_needed(row):
    """"" or why this row is re-scored (see the block above)."""
    ec = row.get("evidence_counted") or {}
    filt = [x for x in (ec.get("filtered") or []) if isinstance(x, dict)]
    if not filt:
        return ""
    gate = str(row.get("evidence_gate") or "")
    if gate not in ("pass", "pubmed"):
        return ""
    if ec.get("snippets"):
        return ""
    if any(not str(x.get("how") or "").startswith("alias:") for x in (ec.get("generif") or [])):
        return ""
    pm = sorted({str(x.get("pmid") or "") for x in filt if str(x.get("pmid") or "")})
    return ("its counted evidence (PMID %s) was removed by the evidence filters and it was admitted on "
            "other evidence (%s)" % (", ".join(pm) or "-", "live PubMed record(s)" if gate == "pubmed"
                                     else "alias GeneRIF line(s)"))


def add_evidence_flag(row, flag, note):
    """fix3 2026-09-28: a flag the report shows next to the gene
    (evidence_flag / evidence_flag_note; several are joined with '; ')."""
    old = str(row.get("evidence_flag") or "")
    row["evidence_flag"] = flag if not old or old == flag else "%s; %s" % (old, flag)
    oldn = str(row.get("evidence_flag_note") or "")
    row["evidence_flag_note"] = note if not oldn else "%s; %s" % (oldn, note)


def _idx(x):
    try:
        return int(x.get("i"))
    except (TypeError, ValueError):
        return -1


def rescore_block(row, scored_doc, dspecs, items, rif_chars=600, lit_symbol=None, fetch=None):
    """(block text, [PMIDs shown]): the gene's block exactly as the score runner
    showed it (`items`: its head line, GeneRIF lines and literature lines with
    their indices -- _score_batches.json), minus every item an evidence filter
    removed (evidence_counted.filtered), plus a counted GeneRIF line the runner
    did not show and the verified live-PubMed rescue records."""
    g = row.get("gene", "")
    q = (lit_symbol or {}).get(g, g)
    ec = row.get("evidence_counted") or {}
    filt = [x for x in (ec.get("filtered") or []) if isinstance(x, dict)]
    fr = {_idx(x) for x in filt if x.get("kind") == "generif"}
    fs = {_idx(x) for x in filt if x.get("kind") == "snippet"}
    fpm = {str(x.get("pmid") or "") for x in filt if x.get("kind") == "snippet" and str(x.get("pmid") or "")}
    gr = (scored_doc.get("generif") or {}).get(g) or []
    nm = ""
    try:
        nm = _evgate.gene_fullname(q) or ""
    except Exception:
        nm = ""
    head = str((items or {}).get("head") or ("GENE: " + g))
    rif = [(int(i), str(ln)) for i, ln in ((items or {}).get("rif") or []) if int(i) not in fr]
    have = {i for i, _ln in rif}
    shown = []
    for x in ec.get("generif") or []:
        i = _idx(x)
        if 0 <= i < len(gr) and i not in have and i not in fr:
            r = str(gr[i])
            rif.append((i, "  - " + (r if len(r) <= rif_chars else r[:rif_chars].rstrip() + " ...")))
            have.add(i)
        shown += [str(y) for y in (x.get("pmids") or [])]
    lit = []
    for j, ln in ((items or {}).get("vd") or []):
        ln = str(ln)
        m = re.match(r"\s*-\s*\[PMID:([^\]]*)\]", ln)
        pmj = m.group(1) if m else ""
        if int(j) in fs or (pmj and pmj in fpm):
            continue
        lit.append(ln)
        if pmj:
            shown.append(pmj)
    if str(row.get("evidence_gate") or "") == "pubmed":
        pe = row.get("pubmed_evidence") or {}
        want, stored = [], {}
        for _k, d in (pe.get("by_domain") or {}).items():
            for p in d.get("verified") or []:
                if str(p) not in want:
                    want.append(str(p))
            stored.update(d.get("records") or {})
        miss = [p for p in want if p not in stored]
        if miss and fetch is not None:
            try:
                got, _e = fetch(miss)
                stored.update(got or {})
            except Exception:
                pass
        for p in want[:5]:
            rec = stored.get(p) or {}
            ab = str(rec.get("abstract") or "")
            ex = (_evgate.gene_excerpt(ab, q, nm, dspecs, width=RESCORE_REC_CHARS) or ab[:RESCORE_REC_CHARS]) if ab else ""
            lit.append("  - [PMID:%s] %s: %s" % (p, str(rec.get("title") or ""), ex))
            shown.append(p)
    lines = [head]
    if rif:
        lines.append("GeneRIF:")
        lines += [ln for _i, ln in rif]
    if lit:
        lines.append("Literature (gene-specific):")
        lines += lit
    if not rif and not lit:
        lines.append("  (NO gene-specific evidence found)")
    return "\n".join(lines), [x for x in dict.fromkeys(shown) if x]


def load_batch_context(path):
    """_score_batches.json written by the score runner, or None."""
    try:
        d = json.loads(Path(path).read_text())
        if isinstance(d, dict) and d.get("batch_of") and d.get("blocks"):
            return d
    except Exception:
        pass
    return None


def _rescore_call(endpoint, model, system, user, timeout=900, reasoning_effort=None):
    """The score runner's call(): OpenAI-compatible chat, temperature 0,
    max_tokens 8000, the JSON array of gene objects."""
    import urllib.request as _ur
    _b = {"model": model, "temperature": 0, "max_tokens": 8000, "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": user}]}
    if reasoning_effort:   # 2026-09-28: the scorer's MRAG_SCORE_REASONING_EFFORT; unset = today's bytes
        _b["reasoning_effort"] = reasoning_effort
    body = json.dumps(_b).encode()
    req = _ur.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
    with _ur.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    txt = resp["choices"][0]["message"]["content"] or ""
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        raise ValueError("no JSON array (truncated?) chars=%d" % len(txt))
    return json.loads(m.group(0))


class _RescoreLLM:
    """The re-score's call() with the scorer's reasoning_effort: _rescore_call(..., reasoning_effort=...).
    MCP step 2a (2026-10-12): a named callable instead of a lambda, so that over MCP the setting itself
    reaches the trn-mcp tool rescore_target_genes (a callable cannot be sent); the same call either way."""

    def __init__(self, reasoning_effort=None):
        self.reasoning_effort = reasoning_effort

    def __call__(self, _e, _m, _s, _u):
        return _rescore_call(_e, _m, _s, _u, reasoning_effort=self.reasoning_effort)


def rescore_after_filter(rows, scored_doc, dspecs, sys_prompt, endpoint, model, rif_chars=600,
                         lit_symbol=None, llm=None, fetch=None, log=None, retries=2, batch_ctx=None):
    """Re-score the rows rescore_needed() selects, in place, each inside its
    original scoring batch (`batch_ctx` = _score_batches.json).  Returns the
    summary {"enabled", "version", "n", "genes": [{gene, old_score, new_score,
    old_reason, new_reason, why, batch, pmids_shown, error}], "n_errors"}.
    NEVER raises."""
    log = log or (lambda m: print(m, flush=True))
    out = {"enabled": True, "version": RESCORE_VERSION, "context": "original batch", "n": 0, "n_errors": 0,
           "genes": []}
    call = llm or _rescore_call
    ctx = batch_ctx or {}
    for r in rows:
        why = rescore_needed(r)
        if not why:
            continue
        g = r.get("gene", "")
        old_s = r.get("score_llm", r.get("score"))
        old_r = r.get("reason_llm") or (r.get("reason") if str(r.get("evidence_gate")) == "pass" else "")
        rec = {"gene": g, "why": why, "old_score": old_s, "old_reason": old_r, "new_score": None,
               "new_reason": "", "batch": [], "pmids_shown": [], "error": ""}
        pm = ", ".join(sorted({str(x.get("pmid") or "") for x in
                               ((r.get("evidence_counted") or {}).get("filtered") or [])
                               if str(x.get("pmid") or "")})) or "-"
        try:
            batch = list((ctx.get("batch_of") or {}).get(g) or [])
            blocks = ctx.get("blocks") or {}
            items = (ctx.get("items") or {}).get(g)
            if not batch or not items or any(x not in blocks for x in batch if x != g):
                raise RuntimeError("no batch context for %s (the score runner did not record its scoring "
                                   "batch)" % g)
            blk, shown = rescore_block(r, scored_doc, dspecs, items, rif_chars, lit_symbol,
                                       fetch or getattr(_evgate, "fetch_paper_records", None))
            rec["pmids_shown"] = shown
            rec["batch"] = batch
            prompt = ("Score these genes.\n\n" + "\n\n".join(blk if x == g else blocks[x] for x in batch)
                      + "\n\n/no_think")
            o, err = None, ""
            for _a in range(int(retries) + 1):
                try:
                    res = call(endpoint, model, sys_prompt, prompt)
                    o = next((x for x in (res or []) if isinstance(x, dict) and x.get("gene") == g), None)
                    if o is None and len(batch) == 1 and len(res or []) == 1 and isinstance(res[0], dict):
                        o = res[0]
                    if o is not None and o.get("score") is not None:
                        break
                    err = "no score for %s in the answer" % g
                    o = None
                except Exception as e:
                    err = "%s: %s" % (type(e).__name__, str(e)[:160])
            if o is None:
                raise RuntimeError(err or "no answer")
            try:
                ns = float(o.get("score"))
                ns = int(ns) if ns == int(ns) else ns
            except (TypeError, ValueError):
                raise RuntimeError("score %r is not a number" % (o.get("score"),))
            nr = re.sub(r"\s+", " ", str(o.get("reason") or "")).strip()
            rec.update(new_score=ns, new_reason=nr)
            r["score_llm_prefilter"] = old_s
            r["reason_llm_prefilter"] = old_r
            r["score_llm"] = ns
            r["reason_llm"] = nr
            r["has_evidence"] = bool(o.get("has_evidence", True))
            r["score"] = ns
            if str(r.get("evidence_gate")) == "pubmed":
                base = re.sub(r"; the LLM score is kept\..*$", "", str(r.get("reason") or ""), flags=re.S)
                r["reason"] = ("%s; re-scored in its scoring batch on the evidence counted now, after the evidence "
                               "filters removed PMID %s (LLM score before: %s). Model reason: %s"
                               % (base, pm, old_s, nr))
            else:
                r["reason"] = nr
            r["rescored_after_filter"] = True
            r["rescore"] = {"version": RESCORE_VERSION, "context": "original batch", "batch": batch,
                            "why": why, "old_score": old_s, "old_reason": old_r, "new_score": ns,
                            "new_reason": nr, "pmids_shown": shown, "filtered_pmids": pm, "block": blk}
            log("[curated] re-score after filter (in its batch of %d): %s %s -> %s (%s)"
                % (len(batch), g, old_s, ns, nr[:120]))
        except Exception as e:
            rec["error"] = str(e)[:200]
            out["n_errors"] += 1
            r["rescored_after_filter"] = False
            r["rescore"] = {"version": RESCORE_VERSION, "why": why, "error": rec["error"],
                            "old_score": old_s, "old_reason": old_r}
            add_evidence_flag(r, "score not re-checked",
                              "the evidence filters removed PMID %s, on which the LLM score %s was given, and the "
                              "re-score failed (%s): the score shown is the old one" % (pm, old_s, rec["error"][:120]))
            log("[curated] re-score after filter FAILED for %s (%s); the old score %s is kept and FLAGGED"
                % (g, rec["error"], old_s))
        out["n"] += 1
        out["genes"].append(rec)
    return out


# ----------------------------------------------------------------------------
# evaudit 2026-09-27: target selection with the post-scoring evidence audit
# ----------------------------------------------------------------------------
def select_targets(scored, panel, n_top, min_score, gate_on, gscope, gdomains, scored_doc,
                   params=None, out_dir=None, llm=None, log=None, audit_endpoint=None, audit_model=None):
    """(top_final, audit_summary).  With the audit off -- MRAG_EVIDENCE_AUDIT=0
    (fix3 2026-09-28: which also switches every other 2026-09-27 layer off,
    like MRAG_EVIDENCE_DEFENCE=0), MRAG_EVIDENCE_AUDIT_MODE=off (the LLM audit
    only) or params["evidence_audit"] False -- this is exactly the
    pre-2026-09-27 rerank(scored, panel, n_top, min_score) and the summary is
    None (nothing is written anywhere).  On (default) and with the domain-scoped gate: the
    ranked candidates are walked in emission order by
    evidence_audit.run_audit, which writes audit_* fields on the rows and
    skips "invalid" genes; with another gate scope the audit cannot judge a
    domain and says so in the summary (targets = the plain rerank).
    deploy 2026-09-28 (decision 1): the audit calls `audit_endpoint` / `audit_model` --
    the node passes the SCORER's (scorer profile q38: qwen3.8 on :11501, native /api/chat
    with think=false; legacy profile: cfg.llm, as before); None = params["endpoint"] /
    params["model"] (cfg.llm)."""
    from . import evidence_audit as _evaudit
    params = params or {}
    log = log or (lambda m: print(m, flush=True))
    if (not _evaudit.audit_enabled(params.get("evidence_audit"))
            or _evaudit.audit_mode(params.get("evidence_audit_mode")) == "off"):
        return rerank(scored, panel, top_final=n_top, min_score=min_score), None
    if not (gate_on and gscope == "domain" and gdomains):
        return (rerank(scored, panel, top_final=n_top, min_score=min_score),
                {"enabled": True, "ran": False, "version": _evaudit.AUDIT_VERSION,
                 "reason": "the audit needs the domain-scoped gate with a resolved literature "
                           "domain (gate %s, scope %s)" % ("on" if gate_on else "off", gscope)})
    cands = rerank(scored, panel, top_final=len(scored) + 1, min_score=min_score)
    log("[curated] evidence audit: %d candidates, target %d, limit %d"
        % (len(cands), n_top, _evaudit.audit_limit(params.get("evidence_audit_limit"))))
    top, summ = _evaudit.run_audit(
        cands, n_top, scored_doc, gdomains,
        audit_endpoint or params.get("endpoint", "http://127.0.0.1:11500/v1/chat/completions"),
        audit_model or params.get("model", "qwen3.6:35b-a3b"),
        cache_path=(str(Path(out_dir) / "_evidence_audit_cache.json") if out_dir else None),
        limit=params.get("evidence_audit_limit"), eg=_evgate,
        fetch=_evgate.fetch_paper_records, llm=llm, log=log,
        mode=params.get("evidence_audit_mode"), keep=params.get("evidence_audit_keep"))
    log("[curated] evidence audit (mode %s): %d emitted, verdicts %s, removed %s, flagged %s, "
        "%d LLM calls (%d cached, %d failed), stop=%s (%ss)"
        % (summ["mode"], summ["emitted"], {k: v for k, v in summ["verdicts"].items() if v},
           [x["gene"] for x in summ["removed"]] or "-",
           ["%s:%s" % (x["gene"], x["verdict"]) for x in summ["flagged"]] or "-", summ["n_llm_calls"],
           summ["n_cached"], summ["n_errors"], summ["stop"], summ["seconds"]))
    if summ.get("note"):
        log("[curated] evidence audit note: " + summ["note"])
    return top, summ


# ----------------------------------------------------------------------------
# domain prompt from config
# ----------------------------------------------------------------------------
# PI DECISION 2026-09-26 -- the LLM scores relevance to the LITERATURE
# domain(s) the user chose as the "Domain of interest" (the list Section 2
# prints as "LLM domain"; the auto-appended pan-cancer corpus is not a user
# domain), and the experimental model is stated as context only.  Until then
# the prompt always read "Domain of interest: <cancer_type> (<cell_line>
# model)", so the 09-25 NEC-only / SCLC-only runs asked about PROSTATE
# relevance while the domain gate demanded NEC / SCLC evidence (their lists
# ended at scores 4 and 3; even two genes with NEC evidence scored <= 2).
#   literature frame (default, when the run has user literature domains):
#     "Domain of interest: <perturbation>-driven <concept> in <A> / <B>.
#      Experimental model: <cell_line> (<cancer_type>); the experiment is in
#      this model, but rate relevance to the domain of interest above."
#   study frame (MRAG_SCORE_DOMAIN_FRAME=study, or the node param
#     score_domain_frame="study"; also whenever the run has no user literature
#     domain): today's wording, byte for byte.
SCORE_DOMAIN_FRAMES = ("literature", "study")
# deploy 2026-09-28 (decision 5).  Qwen A/B 2026-09-28: whether SCLC evidence counted as NEC
# evidence flipped with the scoring batch's composition -- the rubric never said what the nec
# domain covers.  PI 2026-09-26: nec = neuroendocrine CARCINOMA only (NEPC, small-cell
# carcinoma incl. SCLC, LCNEC, Merkel cell carcinoma; NET / PanNET / carcinoid excluded), the
# scope the evidence audit is given too (literature.DOMAIN_AUDIT_DEFINITION["nec"]).  The note
# follows "Domain of interest: ..." in the literature frame whenever nec is one of the run's
# domains; no other domain has a note, so every other prompt is byte-identical.  Off with
# MRAG_SCORE_NEC_NOTE=0 and in the legacy scorer profile (MRAG_SCORER_PROFILE=legacy).
SCORE_DOMAIN_NOTES = {
    "nec": ("Neuroendocrine carcinoma (NEC) here INCLUDES small-cell lung carcinoma (SCLC) and "
            "small-cell carcinoma of other sites, large-cell neuroendocrine carcinoma (LCNEC), "
            "neuroendocrine prostate cancer (NEPC) and Merkel cell carcinoma; well-differentiated "
            "neuroendocrine tumours (NETs), carcinoids and pancreatic NETs (PanNETs) are NOT "
            "neuroendocrine carcinoma."),
}


def score_domain_notes_on(env=None):
    """The domain notes (nec) are part of the scorer's rubric: default on; off with
    MRAG_SCORE_NEC_NOTE=0 or in the legacy scorer profile (MRAG_SCORE_NEC_NOTE=1 wins)."""
    env = os.environ if env is None else env
    v = str(env.get("MRAG_SCORE_NEC_NOTE") or "").strip().lower()
    if v:
        return v not in ("0", "off", "false", "no", "none")
    return scorer_profile(env)[0] != "legacy"


def score_domain_frame(param=None):
    """(frame, source): param wins over env MRAG_SCORE_DOMAIN_FRAME; default
    "literature".  An unknown value falls back to the default and says so in
    `source` ("invalid:<value>") -- the node prints it."""
    for src, v in (("param", param), ("env", os.environ.get("MRAG_SCORE_DOMAIN_FRAME"))):
        s = str(v or "").strip().lower()
        if not s:
            continue
        if s in SCORE_DOMAIN_FRAMES:
            return s, src
        return "literature", "invalid:%s=%s" % (src, str(v)[:40])
    return "literature", "default"


def _frame_domain_keys(lit_domains):
    """The user's literature domains for the prompt: known corpus keys, user
    order, pan-cancer dropped (it is never a user domain here)."""
    try:
        from .. import literature as _l
        return [k for k in _l.known_keys(lit_domains) if k != "pancancer"]
    except Exception:
        return []


def build_sys_prompt_detail(cancer_type=None, tissue=None, cell_line=None,
                            perturbation=None, concept=None,
                            evidence_scope="all", evidence_cap=2.0,
                            evidence_domains="", lit_domains=None, frame=None):
    """build_sys_prompt() plus what it decided: {"sys_prompt", "frame",
    "frame_source", "frame_requested", "domains", "domain_of_interest",
    "experimental_model", "floor_domains"}.  frame is the frame actually USED:
    "study" when requested or when there is no user literature domain."""
    req, fsrc = score_domain_frame(frame)
    keys = _frame_domain_keys(lit_domains)
    use_lit = (req == "literature" and bool(keys))
    dom = []
    if perturbation:
        dom.append(str(perturbation).strip() + "-driven")
    if concept:
        dom.append(str(concept).strip())
    lead = " ".join(dom).strip()
    model_txt, context = "", ""
    if use_lit:
        from .. import literature as _l
        names = " / ".join(n for n in (_l.display_name(k) for k in keys) if n)
        domain = (lead + " in " + names) if lead else names
        _cl = str(cell_line or "").strip()
        _dz = str(cancer_type or tissue or "").strip()
        model_txt = ("%s (%s)" % (_cl, _dz)) if (_cl and _dz) else (_cl or _dz)
        if model_txt:
            context = ("Experimental model: " + model_txt + "; the experiment is in "
                       "this model, but rate relevance to the domain of interest above. ")
        # the floor rule names the SAME domains in the same words when the gate
        # enforces exactly these domains (otherwise it keeps the gate's own label)
        try:
            if str(evidence_domains or "").strip() == _l.domains_label(keys, include_pancancer=True):
                evidence_domains = names
        except Exception:
            pass
    else:
        ctx = cancer_type or tissue or "cancer"
        domain = lead
        if domain:
            domain = domain + " of " + ctx
        else:
            domain = ctx
        if cell_line:
            domain = domain + " (" + str(cell_line) + " model)"
    # The rubric's ceiling and the evidence gate's cap have to MEAN the same
    # thing (2026-09-23 fix).  Under scope="literature" the gate caps genes whose
    # GeneRIF block did name them, which the "NO gene-specific evidence" wording
    # does not authorise -- so under that scope the rubric says what the gate
    # actually enforces.  Under the default scope the wording is unchanged.
    try:
        cap_txt = "%g" % float(evidence_cap)
    except (TypeError, ValueError):
        cap_txt = "2"
    _sc = str(evidence_scope or "all").strip().lower()
    if _sc == "domain":
        # Same fix, one step further: under scope="domain" the gate caps every
        # gene whose evidence never ties it to the domain, so the rubric has to
        # ask for that tie rather than for gene-specific evidence in general.
        _dl = str(evidence_domains or "").strip() or "the domain of interest"
        floor_rule = ("Only evidence that is itself about " + _dl + " can support "
                      "a score above " + cap_txt + ": if every passage shown for a "
                      "gene is about some other disease or about general biology, "
                      "set has_evidence=false and score<=" + cap_txt + " however "
                      "strong that evidence is in its own field. ")
    elif _sc == "literature":
        floor_rule = ("Only a 'Literature (gene-specific)' passage can support a "
                      "score above " + cap_txt + ": if none is shown for a gene, set "
                      "has_evidence=false and score<=" + cap_txt + " however "
                      "suggestive its GeneRIF lines are. ")
    else:
        floor_rule = ("If evidence says NO gene-specific evidence, set "
                      "has_evidence=false and score<=" + cap_txt + ". ")
    # deploy 2026-09-28 (decision 5): the nec note right after the Domain of interest
    # (literature frame only; every other domain's prompt is unchanged)
    notes = [k for k in keys if k in SCORE_DOMAIN_NOTES] if (use_lit and score_domain_notes_on()) else []
    note_txt = "".join(SCORE_DOMAIN_NOTES[k] + " " for k in notes)
    _sp = ("You are a cancer genomics expert. Domain of interest: " + domain + ". " + note_txt + context +
           "For each gene, using ONLY the provided evidence, rate how relevant it is "
           "as a target gene in THIS domain on a 0-10 scale (10=direct, well-established "
           "driver in this domain; 0=no domain link). " + floor_rule +
           "Do NOT use outside knowledge "
           "beyond the evidence. Output ONLY a JSON array of objects "
           '{"gene":str,"score":number,"has_evidence":bool,"reason":str(<=20 words)}. No prose.')
    return {"sys_prompt": _sp, "frame": "literature" if use_lit else "study",
            "frame_source": fsrc, "frame_requested": req, "domains": list(keys) if use_lit else [],
            "domain_of_interest": domain, "experimental_model": model_txt,
            "floor_domains": (str(evidence_domains or "").strip() if _sc == "domain" else ""),
            "domain_notes": notes}


def build_sys_prompt(cancer_type=None, tissue=None, cell_line=None,
                     perturbation=None, concept=None,
                     evidence_scope="all", evidence_cap=2.0,
                     evidence_domains="", lit_domains=None, frame=None):
    """The scoring rubric (system prompt).  Without lit_domains -- or with
    frame / MRAG_SCORE_DOMAIN_FRAME = "study" -- the wording is exactly the
    pre-2026-09-26 one; see build_sys_prompt_detail()."""
    return build_sys_prompt_detail(cancer_type, tissue, cell_line, perturbation, concept,
                                   evidence_scope, evidence_cap, evidence_domains,
                                   lit_domains, frame)["sys_prompt"]


# 2026-09-28 (Ollama 0.34 pre-check and pre-deploy review): the TARGET SCORER's model,
# endpoint and reasoning_effort are set apart from [llm] (which every other LLM step uses).
# The DEFAULT scorer profile "q38" =
# qwen3.8:27b-q8_0 with reasoning_effort "low" on the dedicated user-space Ollama 0.34.4 at
# 127.0.0.1:11501 (GPU 2, else 0), started / loaded before scoring by
# /BIO1/MRAG/scripts/ensure_scorer_ollama.sh --warm.  The in-batch re-score after an evidence
# filter and the evidence audit follow the scorer (same model and endpoint); :11500 keeps
# serving every other LLM step with cfg.llm.
# ROLLBACK -- MRAG_SCORER_PROFILE=legacy: the pre-2026-09-28 scorer exactly: cfg.llm model and
# endpoint (qwen3.6 on :11500), no reasoning_effort, no ensure command, no preflight, no
# unscored stop, no NEC domain note: the pre-deploy request bytes and _score_params.json.
# Single MRAG_SCORE_* variables override either profile (score_llm_settings).
SCORE_REASONING_EFFORTS = ("none", "low", "medium", "high")
SCORE_MAX_UNSCORED_FRAC = 0.02
SCORER_PROFILE_DEFAULT = "q38"
SCORER_PROFILE_ALIASES = {"q38": "q38", "default": "q38", "new": "q38", "qwen3.8": "q38",
                          "legacy": "legacy", "production": "legacy", "prod": "legacy",
                          "old": "legacy", "qwen3.6": "legacy"}
SCORER_ENSURE_SCRIPT = "/BIO1/MRAG/scripts/ensure_scorer_ollama.sh"
SCORER_Q38 = {"model": "qwen3.8:27b-q8_0",
              "endpoint": "http://127.0.0.1:11501/v1/chat/completions",
              "reasoning_effort": "low",
              "ensure_cmd": SCORER_ENSURE_SCRIPT + " --warm"}
_SCORE_OFF = ("0", "off", "false", "no", "none")
# pre-deploy review 2026-09-28: the installed ensure script exits 2 when no GPU of "2 0" can take the
# model NOW (or MemAvailable is low). The curated step waits for that, bounded, before it fails.
SCORE_ENSURE_WAIT_S = 1800
SCORE_ENSURE_POLL_S = 60


def scorer_profile(env=None):
    """(profile, source): MRAG_SCORER_PROFILE = q38 (the default; also default / new /
    qwen3.8) or legacy (also production / prod / old / qwen3.6).  An unknown value RAISES:
    a typo must not silently pick a scorer."""
    env = os.environ if env is None else env
    v = str(env.get("MRAG_SCORER_PROFILE") or "").strip().lower()
    if not v:
        return SCORER_PROFILE_DEFAULT, "default"
    if v not in SCORER_PROFILE_ALIASES:
        raise ValueError("MRAG_SCORER_PROFILE=%r is not one of q38 | legacy" % v)
    return SCORER_PROFILE_ALIASES[v], "env"


def score_llm_settings(model, endpoint, env=None):
    """(model, endpoint, reasoning_effort or None, record) for the scorer.
    `model` / `endpoint` are cfg.llm's (the node params): the legacy profile's scorer.  The
    profile (scorer_profile) gives the defaults; single variables override them:
    MRAG_SCORE_MODEL            e.g. qwen3.8:27b-q8_0
    MRAG_SCORE_ENDPOINT         base URL (http://127.0.0.1:11501) or a full
                                .../chat/completions URL
    MRAG_SCORE_REASONING_EFFORT none|low|medium|high (added to the request body);
                                omit = no reasoning_effort key
    `record` is written to _score_params.json (score_llm_env): the profile unless it is
    legacy, plus every MRAG_SCORE_* override -- empty for the legacy profile without
    overrides, so that _score_params.json is the pre-deploy file byte for byte."""
    env = os.environ if env is None else env
    prof, psrc = scorer_profile(env)
    if prof == "legacy":
        dm, de, dr = model, endpoint, None
    else:
        dm, de, dr = SCORER_Q38["model"], SCORER_Q38["endpoint"], SCORER_Q38["reasoning_effort"]
    m = (env.get("MRAG_SCORE_MODEL") or "").strip()
    e = (env.get("MRAG_SCORE_ENDPOINT") or "").strip()
    r = (env.get("MRAG_SCORE_REASONING_EFFORT") or "").strip().lower()
    if r == "omit":
        dr, r = None, ""
    elif r and r not in SCORE_REASONING_EFFORTS:
        raise ValueError("MRAG_SCORE_REASONING_EFFORT=%r is not one of %s or omit" % (r, SCORE_REASONING_EFFORTS))
    if e and not e.rstrip("/").endswith("/chat/completions"):
        e = e.rstrip("/") + "/v1/chat/completions"
    rec = {} if prof == "legacy" else {"MRAG_SCORER_PROFILE": prof + (" (default)" if psrc == "default" else "")}
    rec.update({k: env.get(k) for k in ("MRAG_SCORE_MODEL", "MRAG_SCORE_ENDPOINT", "MRAG_SCORE_REASONING_EFFORT")
                if (env.get(k) or "").strip()})
    return (m or dm), (e or de), (r or dr), rec


def _score_base_url(endpoint):
    e = str(endpoint or "").rstrip("/")
    for suf in ("/v1/chat/completions", "/chat/completions"):
        if e.endswith(suf):
            return e[:-len(suf)]
    return e


def score_ensure_cmd(endpoint, env=None):
    """The command run before scoring (deploy 2026-09-28): MRAG_SCORE_ENSURE_CMD when set
    (none / off / 0 = no command); else, in the q38 profile and ONLY for the dedicated endpoint
    (127.0.0.1:11501), the installed ensure script with --warm; else no command."""
    env = os.environ if env is None else env
    v = str(env.get("MRAG_SCORE_ENSURE_CMD") or "").strip()
    if v:
        return "" if v.lower() in _SCORE_OFF else v
    if (scorer_profile(env)[0] != "legacy"
            and _score_base_url(endpoint) == _score_base_url(SCORER_Q38["endpoint"])):
        return SCORER_Q38["ensure_cmd"]
    return ""


def score_ensure_wait(cmd, env=None):
    """(wait_s, poll_s) for an ensure command that exits 2 ("busy now, retry later"): the installed
    ensure script (the q38 default) waits up to SCORE_ENSURE_WAIT_S; any other command does not wait
    (its exit codes are not ours to interpret).  MRAG_SCORE_ENSURE_WAIT_S / _POLL_S override."""
    env = os.environ if env is None else env
    dflt = float(SCORE_ENSURE_WAIT_S) if str(cmd or "").strip() == SCORER_Q38["ensure_cmd"] else 0.0
    try:
        w = float(env.get("MRAG_SCORE_ENSURE_WAIT_S") or dflt)
    except ValueError:
        w = dflt
    try:
        q = float(env.get("MRAG_SCORE_ENSURE_POLL_S") or SCORE_ENSURE_POLL_S)
    except ValueError:
        q = float(SCORE_ENSURE_POLL_S)
    return max(0.0, w), max(1.0, q)


def score_preflight(endpoint, model, env=None):
    """Fail BEFORE the score runner when the scorer cannot answer: the runner turns every
    failed call into score 0 / "unscored" and the run would finish with a log2FC-only list.
    1. MRAG_SCORE_ENSURE_CMD (optional; e.g. ".../ensure_scorer_ollama.sh --warm") must exit 0
       within MRAG_SCORE_ENSURE_TIMEOUT_S (900 s); exit 2 ("no GPU free now") is retried every
       MRAG_SCORE_ENSURE_POLL_S for up to MRAG_SCORE_ENSURE_WAIT_S (score_ensure_wait: 1800 s for
       the installed ensure script, 0 for any other command);
    2. the endpoint must list `model` (/v1/models, else Ollama /api/tags).
    MRAG_SCORE_PREFLIGHT=0 skips both.  Returns a small record; raises RuntimeError."""
    import shlex, time, urllib.request
    env = os.environ if env is None else env
    prof = scorer_profile(env)[0]
    _pfv = str(env.get("MRAG_SCORE_PREFLIGHT") or "").strip().lower()
    if _pfv in _SCORE_OFF or (not _pfv and prof == "legacy"):
        # deploy 2026-09-28: the legacy profile is the pre-deploy scorer, which had no
        # preflight (MRAG_SCORE_PREFLIGHT=1 turns it on there)
        return {"skipped": True, "profile": prof}
    cmd = score_ensure_cmd(endpoint, env)
    if cmd:
        try:
            to = float(env.get("MRAG_SCORE_ENSURE_TIMEOUT_S") or 900)
        except ValueError:
            to = 900.0
        wait_s, poll_s = score_ensure_wait(cmd, env)
        t0, n_busy, waited = time.time(), 0, 0.0
        while True:
            try:
                pr = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=to)
            except Exception as e:
                raise RuntimeError("scorer preflight: MRAG_SCORE_ENSURE_CMD failed to run (%s: %s)"
                                   % (type(e).__name__, str(e)[:200]))
            waited = time.time() - t0
            if pr.returncode != 2 or waited + poll_s > wait_s:
                break
            n_busy += 1
            _last = (((pr.stderr or "") + (pr.stdout or "")).strip().splitlines() or [""])[-1]
            print("[curated] scorer busy (ensure exit 2: %s); retry %d in %d s (waited %d of %d s, "
                  "MRAG_SCORE_ENSURE_WAIT_S)" % (_last[-300:], n_busy, poll_s, waited, wait_s), flush=True)
            time.sleep(poll_s)
        if pr.returncode != 0:
            raise RuntimeError("scorer preflight: MRAG_SCORE_ENSURE_CMD exited %d%s: %s"
                               % (pr.returncode,
                                  (" after waiting %d s for a free GPU (%d retries)" % (waited, n_busy))
                                  if n_busy else "",
                                  ((pr.stderr or "") + (pr.stdout or ""))[-800:]))
    base = _score_base_url(endpoint)
    names, errs = set(), []
    for path, key in (("/v1/models", "data"), ("/api/tags", "models")):
        try:
            with urllib.request.urlopen(base + path, timeout=30) as r:
                d = json.loads(r.read())
            for x in (d.get(key) or []):
                for k in ("id", "name", "model"):
                    if isinstance(x, dict) and x.get(k):
                        names.add(str(x[k]))
            if names:
                break
        except Exception as e:
            errs.append("%s: %s" % (path, str(e)[:120]))
    if not names:
        raise RuntimeError("scorer preflight: %s is not reachable or lists no model (%s); not scoring, "
                           "because every gene would be written as score 0 'unscored'"
                           % (base, "; ".join(errs) or "empty list"))
    if str(model) not in names:
        raise RuntimeError("scorer preflight: model %r is not served by %s (it lists: %s)"
                           % (model, base, ", ".join(sorted(names))[:400]))
    out = {"base": base, "model": model, "ensure_cmd": bool(cmd), "ensure": cmd, "profile": prof}
    if cmd and n_busy:
        out["ensure_busy_retries"] = n_busy
    return out


def score_startup_check(model, endpoint, env=None, ensure_script=None, timeout=120):
    """pre-deploy review 2026-09-28: the TARGET SCORER checked at the START of a curated run (the
    runner calls this before any node), so that a scorer that cannot work fails the run in minutes
    and not at the curated step hours later.  Read-only: nothing is started, loaded or stopped.
    `model` / `endpoint` are cfg.llm's (the legacy profile's scorer), as for score_llm_settings.
    Returns {"status": "skipped" | "ok" | "busy" | "unchecked", ...}.  RAISES only for what would
    make the curated step fail whatever the GPUs do:
      - MRAG_SCORER_PROFILE / MRAG_SCORE_* values that do not parse (ValueError);
      - the installed ensure script missing, or refusing (--probe exit 3: a foreign listener on the
        port, our server running with another config, the binary or the model missing);
      - a custom MRAG_SCORE_ENSURE_CMD whose program does not exist.
    "busy" (no GPU can take the model now, or MemAvailable is low) is a WARNING only: the GPUs may be
    free by the time the curated step runs, and it waits for them (score_preflight).  Skipped where
    score_preflight is skipped: the legacy profile (the pre-deploy scorer had no check) and
    MRAG_SCORE_PREFLIGHT=0."""
    import shlex, shutil, urllib.request
    env = os.environ if env is None else env
    m, e, _r, _rec = score_llm_settings(model, endpoint, env)
    prof = scorer_profile(env)[0]
    _pfv = str(env.get("MRAG_SCORE_PREFLIGHT") or "").strip().lower()
    if _pfv in _SCORE_OFF or (not _pfv and prof == "legacy"):
        return {"status": "skipped", "profile": prof}
    rec = {"profile": prof, "model": m, "endpoint": e}
    cmd = score_ensure_cmd(e, env)
    hint = ("fix it, or run with MRAG_SCORER_PROFILE=legacy (the pre-2026-09-28 scorer, qwen3.6 on "
            "cfg.llm)")
    if cmd and cmd == SCORER_Q38["ensure_cmd"]:
        script = ensure_script or SCORER_ENSURE_SCRIPT
        rec["ensure"] = script
        if not (os.path.isfile(script) and os.access(script, os.X_OK)):
            raise RuntimeError("scorer check: the ensure script %s is missing or not executable; the "
                               "curated step could not start the scorer -- %s" % (script, hint))
        try:
            pr = subprocess.run([script, "--probe"], capture_output=True, text=True, timeout=timeout)
        except Exception as ex:
            rec.update(status="unchecked", detail="%s --probe: %s: %s" % (script, type(ex).__name__, str(ex)[:160]))
            return rec
        _out = ((pr.stderr or "") + (pr.stdout or "")).strip()
        _last = (_out.splitlines() or [""])[-1][-400:]
        if pr.returncode == 3:
            raise RuntimeError("scorer check: %s --probe refused (exit 3): %s -- the curated step "
                               "would fail; %s" % (script, _out[-800:], hint))
        rec.update(status={0: "ok", 2: "busy"}.get(pr.returncode, "unchecked"),
                   probe_rc=pr.returncode, detail=_last)
        return rec
    if cmd:
        prog = (shlex.split(cmd) or [""])[0]
        if not (shutil.which(prog) or (os.path.isfile(prog) and os.access(prog, os.X_OK))):
            raise RuntimeError("scorer check: MRAG_SCORE_ENSURE_CMD program %r not found -- %s" % (prog, hint))
        rec.update(status="unchecked", detail="custom ensure command %r runs before scoring" % cmd)
        return rec
    # no ensure command: the endpoint is expected to be up already; a quick look, never fatal here
    base = _score_base_url(e)
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
            d = json.loads(r.read())
        names = {str(x.get("id")) for x in (d.get("data") or []) if isinstance(x, dict)}
        rec.update(status="ok" if str(m) in names else "unchecked",
                   detail=("%s lists %s" % (base, m)) if str(m) in names
                   else ("%s does not list %s now" % (base, m)))
    except Exception as ex:
        rec.update(status="unchecked", detail="%s not reachable now (%s)" % (base, str(ex)[:120]))
    return rec


def check_unscored(rows, env=None):
    """(n_unscored, n, genes) of the score runner's rows; raises RuntimeError when more than
    MRAG_SCORE_MAX_UNSCORED_FRAC (default 0.02; 1 = never) of them were never scored.  A row
    is unscored when the runner wrote its fallback {score 0, reason "unscored"} (the evidence
    gate may have moved that reason to reason_llm)."""
    env = os.environ if env is None else env
    # deploy 2026-09-28: the legacy profile never stops the node (the pre-deploy behaviour;
    # the node still prints the warning line)
    _dflt = 1.0 if scorer_profile(env)[0] == "legacy" else SCORE_MAX_UNSCORED_FRAC
    try:
        frac = float(env.get("MRAG_SCORE_MAX_UNSCORED_FRAC") or _dflt)
    except ValueError:
        frac = _dflt
    un = [r.get("gene") for r in rows
          if (r.get("score") or 0) == 0 and "unscored" in (r.get("reason"), r.get("reason_llm"))]
    n = len(rows)
    if un and len(un) > frac * n:
        raise RuntimeError("scorer: %d/%d genes UNSCORED (> %.0f%%; the endpoint failed or answered "
                           "without usable JSON), e.g. %s -- scored.json is kept, targets were NOT "
                           "selected (MRAG_SCORE_MAX_UNSCORED_FRAC=1 restores the old behaviour)"
                           % (len(un), n, 100 * frac, ", ".join(map(str, un[:10]))))
    return len(un), n, un


# ----------------------------------------------------------------------------
# followup 2026-09-28 (fix 1): a failed GeneRIF fetch is never silent
# ----------------------------------------------------------------------------
# The score runner used to turn ANY error of genegpt.gene_generifs into zero
# GeneRIF lines for that gene, with no mark anywhere (2026-09-28 SCLC E2E: 8
# genes, 581 lines, likely NCBI rate limiting).  Now the runner records every
# gene's fetch outcome (genegpt.gene_generifs_status: ok / empty = NCBI lists
# none / no_gene = no human Gene ID / failed; not_searched = no human
# ortholog) in the sidecar _generif_fetch.json, retries the failed genes in a
# second pass after MRAG_GENERIF_RETRY_PAUSE_S (30 s), and FLAGS a gene that
# still failed on its row (generif_fetch "failed", evidence_flag "GeneRIF
# fetch failed", shown in Sections 10 and 14).  The node then stops the target
# step LOUDLY -- like the unscored guard: scored.json kept, no targets -- when
# more than MRAG_GENERIF_MAX_FAILED_FRAC (0.02) of the genes failed, or when a
# failed gene COULD HAVE BEEN EMITTED as a target (generif_failed_candidates;
# MRAG_GENERIF_FAIL_ON_TOP=0: go on with it flagged instead).  With no failure
# nothing of the prompts, the requests, scored.json or the rows changes; the
# sidecar is the only addition.  This is a bug fix and is on in every scorer
# profile and evidence-defence setting.
# Review 2026-09-28 (finding 1): the first build stopped only for a failed gene
# that WAS emitted.  A gene whose lines are lost can fall OUT of the list --
# one FINAL SCLC target, grounded on 125 lines, lost them in the E2E and was not
# emitted -- and one such gene (0.4 %) passed with a warning.  Now every failed
# gene that is not a panel TF and has a driver TF stops the step, BEFORE the
# PubMed rescue, the re-score and the audit.  No score, rank or margin can
# clear it: the evidence gate caps a gene grounded only by GeneRIF lines at 2,
# below the floor (8 of the 60 FINAL targets of 2026-09-26 are grounded that
# way), so a failed gene's score says nothing about where it belonged.
GENERIF_MAX_FAILED_FRAC = 0.02
GENERIF_SIDECAR = "_generif_fetch.json"


def generif_max_failed_frac(env=None):
    env = os.environ if env is None else env
    try:
        return float(env.get("MRAG_GENERIF_MAX_FAILED_FRAC") or GENERIF_MAX_FAILED_FRAC)
    except ValueError:
        return GENERIF_MAX_FAILED_FRAC


def generif_fail_on_top(env=None):
    env = os.environ if env is None else env
    v = str(env.get("MRAG_GENERIF_FAIL_ON_TOP") or "").strip().lower()
    return v not in _SCORE_OFF


def generif_fetch_summary(scored_out, scored_doc=None):
    """The score runner's GeneRIF fetch record (_generif_fetch.json next to scored.json):
    {"n", "counts", "failed": [genes], "errors": {gene: error}, "first_pass_failed",
    "recovered": [genes]} -- {} for a runner that wrote none.  Never raises."""
    try:
        d = json.loads((Path(str(scored_out)).parent / GENERIF_SIDECAR).read_text())
    except Exception:
        return {}
    if not isinstance(d, dict):
        return {}
    genes = d.get("genes") or {}
    failed = [g for g in (d.get("failed") or []) if g in genes]
    first = list(d.get("first_pass_failed") or [])
    return {"n": int(d.get("n_genes") or len(genes)), "counts": dict(d.get("counts") or {}),
            "failed": failed, "errors": {g: str((genes.get(g) or {}).get("error") or "") for g in failed},
            "first_pass_failed": first, "recovered": [g for g in first if g not in failed],
            "seconds": d.get("seconds")}


def check_generif_fetch(summary, env=None):
    """Raise when more than MRAG_GENERIF_MAX_FAILED_FRAC (default 0.02; 1 = never) of the scored
    genes' GeneRIF records could not be read -- the score list would rest on missing evidence."""
    f = list((summary or {}).get("failed") or [])
    n = int((summary or {}).get("n") or 0)
    frac = generif_max_failed_frac(env)
    if f and n and len(f) > frac * n:
        err = (summary.get("errors") or {})
        raise RuntimeError(
            "GeneRIF fetch: the NCBI GeneRIF records of %d/%d genes could not be read, also in the second "
            "pass (> %.0f%%): e.g. %s -- scored.json and _generif_fetch.json are kept, targets were NOT "
            "selected; re-run when NCBI answers again (MRAG_GENERIF_MAX_FAILED_FRAC=1 plus, when any of them "
            "could have been a target, MRAG_GENERIF_FAIL_ON_TOP=0 continue with those genes flagged)"
            % (len(f), n, 100 * frac,
                                "; ".join("%s (%s)" % (g, str(err.get(g) or "")[:100]) for g in f[:5])))
    return len(f), n


def generif_failed_candidates(rows, panel=None):
    """Review 2026-09-28 (finding 1): the scored genes whose GeneRIF record could not be read (row
    generif_fetch "failed") that COULD HAVE BEEN EMITTED as targets: every such gene that is not a
    panel TF and has a driver TF -- the only exclusions of rerank() that do not depend on the gene's
    literature evidence.  Its score, the evidence gate's cap, the PubMed rescue and the audit were
    all decided without its lines, so no score or rank can show it would have missed the cut."""
    ps = set(panel or [])
    return [r.get("gene") for r in (rows or []) if isinstance(r, dict) and r.get("generif_fetch") == "failed"
            and r.get("gene") not in ps and r.get("drivers")]


def check_generif_candidates(rows, panel=None, summary=None, env=None):
    """Raise (unless MRAG_GENERIF_FAIL_ON_TOP=0) when a gene of generif_failed_candidates() exists:
    the target list may be wrong.  Returns the candidate genes (the caller goes on with them flagged)."""
    cand = generif_failed_candidates(rows, panel)
    if cand and generif_fail_on_top(env):
        err = ((summary or {}).get("errors") or {})
        raise RuntimeError(
            "GeneRIF fetch: the NCBI GeneRIF record of %d gene(s) that could have been EMITTED as targets could "
            "not be read, also in the second pass: %s -- their scores, the evidence gate (a gene grounded only by "
            "its GeneRIF lines is capped out of the list) and their ranks were decided without it, so the target "
            "list may be wrong; scored.json and _generif_fetch.json are kept, targets were NOT selected; re-run "
            "when NCBI answers again (MRAG_GENERIF_FAIL_ON_TOP=0 goes on with them flagged)"
            % (len(cand), "; ".join("%s (%s)" % (g, str(err.get(g) or "")[:100]) for g in cand[:8])
               + (" ..." if len(cand) > 8 else "")))
    return cand


# ----------------------------------------------------------------------------
# followup r3 2026-09-28 (post-deploy finding 1): a FAILED live-PubMed rescue check is never silent
# ----------------------------------------------------------------------------
# The rescue (evidence_gate.pubmed_rescue) re-admits a gene the domain gate CAPPED when a live PubMed record
# names it in the domain -- two SCLC and two NEC targets of the domain runs made the list only through it.  A check
# that FAILED (an NCBI error after call_api's retries and the rescue's own second try) left the gene capped: it
# fell out of the list with a log line ("pubmed rescue note") and n_errors in scored.json -- no stop, no manifest
# gate, nothing in the report (a 429 like the one of the 2026-09-28 SCLC E2E, hit during the rescue, would have
# removed such a target silently).  Now, like a failed GeneRIF fetch (fix 1): the genes the rescue could NOT decide -- a
# rescue candidate (gate "fired", own LLM score above the cap) with no verified record and at least one failed
# (or never run) domain check -- are recorded (row pubmed_rescue_check "failed" + pubmed_rescue_error; scored.json
# "pubmed_rescue_failed"), and the target step STOPS loudly when one of them could have been emitted (not a panel
# TF, has a driver TF -- as check_generif_candidates: the gate parks it at the cap, below the floor, so no score
# or rank can clear it), right after the rescue, before the re-score and the audit (MRAG_RESCUE_FAIL_ON_TOP=0 goes
# on).  Any failed check that does not stop the step (only panel TFs / genes without a driver TF, the override,
# or a rescued gene whose OTHER domain failed) is a run.log WARNING, a manifest WARN gate "pubmed_rescue", the step
# output "pubmed_rescue_failed" and branches/topN/pubmed_rescue.json -> a notice under the Section-10 table.  With
# no failed check nothing changes (rows, scored.json, outputs, report).  A bug fix: on in every scorer profile and
# evidence-defence setting (like fix 1).
RESCUE_BRANCH_FILE = "pubmed_rescue.json"


def rescue_fail_on_top(env=None):
    env = os.environ if env is None else env
    v = str(env.get("MRAG_RESCUE_FAIL_ON_TOP") or "").strip().lower()
    return v not in _SCORE_OFF


def rescue_candidates(rows, cap=None):
    """The genes evidence_gate.pubmed_rescue checks: gate verdict "fired" and the gene's own LLM score above the cap
    (its test, verbatim).  Take it BEFORE the rescue: the rescue re-labels the rows it admits."""
    try:
        cap_v = float(_evgate.gate_cap(cap))
    except Exception:
        cap_v = float(cap if cap is not None else 2)
    out = []
    for r in rows or []:
        if not isinstance(r, dict) or r.get("evidence_gate") != "fired" or not r.get("gene"):
            continue
        try:
            llm = float(r.get("score_llm") if r.get("score_llm") is not None else r.get("score") or 0)
        except (TypeError, ValueError):
            llm = 0.0
        if llm > cap_v:
            out.append(str(r["gene"]))
    return out


def rescue_failure_record(rows, resc, cand, panel=None):
    """{} when the rescue decided every candidate (verified, or every domain check answered), else
      {"version": 1, "failed": [candidate genes neither verified nor checked in every domain],
       "errors": {gene: "<domain>: <error>; ..."}, "could_emit": [failed genes that are not panel TFs and have a
       driver TF], "cannot": [the other failed genes], "partial": [ADMITTED genes whose check of another domain
       failed], "n_candidates", "n_errors", "n_pairs", "error"}
    and each failed gene's row gets pubmed_rescue_check "failed" + pubmed_rescue_error.  `cand` =
    rescue_candidates() taken before the rescue, `resc` = the rescue's summary (or the node's record of a rescue
    that raised).  A rescue that ran with no domain carrying a PubMed query fragment checked nothing BY DESIGN
    (not a failure); a switched-off rescue likewise."""
    resc = resc if isinstance(resc, dict) else {}
    cand = [str(g) for g in (cand or []) if g]
    if not cand or not resc.get("enabled"):
        return {}
    if "domains" in resc and not resc.get("domains"):
        return {}
    by_gene = {str(r.get("gene")): r for r in (rows or []) if isinstance(r, dict) and r.get("gene")}
    per = resc.get("per_gene") if isinstance(resc.get("per_gene"), dict) else {}
    rescued = {str(g) for g in (resc.get("rescued_genes") or [])}
    rescued |= {g for g in cand if (by_gene.get(g) or {}).get("evidence_gate") == "pubmed"}
    failed, partial, errors = [], [], {}
    for g in cand:
        pg = per.get(g) if isinstance(per.get(g), dict) else None
        if g in rescued:
            if pg is not None and int(pg.get("n_errors") or 0) > 0:
                partial.append(g)
            continue
        if pg is not None and int(pg.get("n_errors") or 0) == 0:
            continue                      # every domain answered: a real "not found"
        failed.append(g)
        why = []
        bd = ((by_gene.get(g) or {}).get("pubmed_evidence") or {}).get("by_domain") or {}
        if pg is not None and isinstance(bd, dict):
            for k, d in bd.items():
                d = d if isinstance(d, dict) else {}
                if d.get("error") or not d.get("ok", True):
                    why.append("%s: %s" % (k, str(d.get("error") or "check failed")[:160]))
        if not why:
            why.append(str(resc.get("error") or "the live PubMed check did not run")[:200])
        errors[g] = "; ".join(why)[:400]
    if not failed and not partial:
        return {}
    ps = set(panel or [])
    could = [g for g in failed if g not in ps and (by_gene.get(g) or {}).get("drivers")]
    for g in failed:
        r = by_gene.get(g)
        if r is not None:
            r["pubmed_rescue_check"] = "failed"
            r["pubmed_rescue_error"] = errors[g][:300]
    return {"version": 1, "failed": failed, "errors": errors, "could_emit": could,
            "cannot": [g for g in failed if g not in could], "partial": partial,
            "n_candidates": len(cand), "n_errors": int(resc.get("n_errors") or 0),
            "n_pairs": int(resc.get("n_pairs") or 0), "error": str(resc.get("error") or "")[:300]}


def check_rescue_candidates(rec, env=None):
    """Raise (unless MRAG_RESCUE_FAIL_ON_TOP=0) when a gene of rescue_failure_record()["could_emit"] exists: the
    rescue could have admitted it, so the target list may be wrong.  Returns those genes (the caller goes on)."""
    could = list((rec or {}).get("could_emit") or [])
    if could and rescue_fail_on_top(env):
        err = (rec or {}).get("errors") or {}
        raise RuntimeError(
            "PubMed rescue: the live PubMed check of %d capped gene(s) that could have been EMITTED as targets "
            "failed, also on a second try: %s -- the evidence gate found no counted evidence for them in the "
            "literature domain and the check that could have re-admitted them (a PubMed record naming the gene in "
            "the domain) never answered, so the target list may be wrong; scored.json (pubmed_rescue, "
            "pubmed_rescue_failed) is kept, targets were NOT selected; re-run when NCBI answers again "
            "(MRAG_RESCUE_FAIL_ON_TOP=0 goes on with them listed)"
            % (len(could), "; ".join("%s (%s)" % (g, str(err.get(g) or "")[:100]) for g in could[:8])
               + (" ..." if len(could) > 8 else "")))
    return could


# ----------------------------------------------------------------------------
# subprocess runner script (static; params passed via json argv[1])
# ----------------------------------------------------------------------------
_SCORE_RUNNER = r'''
import json, os, sys, time, urllib.request, re
p = json.load(open(sys.argv[1]))
genes = p["genes"]
# each gene's log2FC and driver TFs (= its bound panel TFs) come from pool_motifs.json (motifs_path)
_mot = json.load(open(p["motifs_path"]))
LFC = {g: (m or {}).get("log2FC", 0) for g, m in _mot.items()}
DRV = {g: list(((m or {}).get("tfs") or {}).keys()) for g, m in _mot.items()}
# Evidence-grounding gate (2026-09-23).  Loaded from the node package so the
# score side and the Section-14 novelty check share ONE definition of what it
# means for a passage to name a gene.  Pure text over evidence already
# retrieved below -- no extra network call.
EG = None; EG_ERR = ""
GATE_ON = bool(p.get("evidence_gate", True))
GATE_CAP = p.get("evidence_gate_cap")
GATE_SCOPE = p.get("evidence_gate_scope")
# scope="domain": the node resolved literature.domain_specs() for us, because
# mrag_orchestrator is NOT importable from this venv.
GATE_DOMAINS = p.get("evidence_gate_domains") or []
try:
    import importlib.util as _ilu
    _sp = _ilu.spec_from_file_location("mrag_evidence_gate", p["gate_module"])
    # EG must stay None when the import fails. Assigning it BEFORE exec_module
    # left a half-initialised module behind on failure, so `EG is not None`
    # was True and the first EG.<anything> below died with an AttributeError
    # AFTER the LLM scoring had already run -- instead of the intended loud
    # "evidence gate enabled but module not loadable" at the end (2026-09-23).
    _egmod = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_egmod)
    EG = _egmod
except Exception as e:
    EG_ERR = str(e)[:200]
    print("EVIDENCE_GATE import fail:", EG_ERR, flush=True)
# Domain specs only matter under scope="domain"; leaving them empty otherwise
# keeps scored.json byte-identical to a default run.
_DSPECS = (EG.domain_specs(GATE_DOMAINS)
           if (EG is not None and str(GATE_SCOPE or "").strip().lower() == "domain")
           else [])
# --- literature namespace (2026-09-23 Task B) -------------------------------
# Every retrieval below -- GeneRIF, PaperQA, the naming gate -- is indexed on
# HUMAN symbols (libs/genegpt even hard-codes "<SYM>[sym] AND human[orgn]").
# For a non-human run LIT maps this run's symbol to its human ortholog, and
# LIT_UNMAPPED lists the genes that HAVE no human ortholog. Those are NOT
# searched: a lookup of a mouse-only gene in a human index returns nothing,
# which would be scored as "no evidence" when the truth is "not searchable".
# For a human run LIT is empty and lit(g) is g, so nothing changes at all.
LIT = p.get("lit_symbol") or {}
LIT_UNMAPPED = set(p.get("lit_unmapped") or [])
def lit(g):
    return LIT.get(g, g)
if LIT or LIT_UNMAPPED:
    print("LIT_NAMESPACE human-ortholog route: %d mapped, %d with no ortholog "
          "(not searched): %s" % (len(LIT), len(LIT_UNMAPPED),
                                  sorted(LIT_UNMAPPED)[:10]), flush=True)
GR = {}
GRP = {}
try:
    sys.path.insert(0, "/BIO1/MRAG/Agent/libs/genegpt")
    from genegpt import gene_generifs
except Exception as e:
    gene_generifs = None
    print("genegpt import fail", str(e)[:120], flush=True)
t0 = time.time()
# genegpt.gene_generifs issues ONE efetch of the WHOLE gene record whatever
# retmax is -- retmax only caps the local parse -- so raising it costs nothing
# on the wire and stops a well-studied gene's older, domain-specific RIFs from
# being thrown away.  Measured over the 249 curated genes, genes with a
# domain-tied GeneRIF: NEC 6 -> 12, SCLC 3 -> 6, prostate 30 -> 46 (2026-09-23).
RIF_RETMAX = int(p.get("generif_retmax") or 20)
# followup 2026-09-28 (fix 1): every gene's fetch OUTCOME is recorded (the
# sidecar _generif_fetch.json): ok / empty (NCBI lists no GeneRIF) / no_gene (no
# human Gene ID) / failed / not_searched (no human ortholog).  A failed fetch
# used to be an unmarked [] (2026-09-28 SCLC E2E: 8 genes, 581 lines).  Failed
# genes get a SECOND pass after MRAG_GENERIF_RETRY_PAUSE_S (30 s).  Circuit
# breaker: after MRAG_GENERIF_BREAKER (5) consecutive failures a pass pauses
# for the same time and goes on; the third time it stops calling and marks the
# rest of that pass failed ("not attempted") -- NCBI is down, and hammering it
# helps nobody.  The whole fetch stops starting requests after
# MRAG_GENERIF_BUDGET_S (2400 s; a normal fetch takes ~6 min).  GR / GRP are
# filled exactly as before for every gene whose record was read, in order.
GRF = {}
_GG_STATUS = None
if gene_generifs is not None:
    try:
        from genegpt import gene_generifs_status as _GG_STATUS
    except Exception:
        _GG_STATUS = None
def _envf(name, default):
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return float(default)
def _rif_fetch(q):
    """(records, status, error, attempts) for one literature symbol; never raises"""
    if _GG_STATUS is not None:
        try:
            st = _GG_STATUS(q, retmax=RIF_RETMAX) or {}
        except Exception as e:
            return [], "failed", "%s: %s" % (type(e).__name__, str(e)[:200]), 0
        stt = st.get("status")
        if stt not in ("ok", "empty", "no_gene", "failed"):
            stt = "failed"
        return (list(st.get("lines") or []) if stt != "failed" else []), stt, \
            str(st.get("error") or "")[:300], int(st.get("attempts") or 0)
    try:
        rifs = gene_generifs(q, retmax=RIF_RETMAX) or []
        return rifs, ("ok" if rifs else "empty"), "", 1
    except Exception as e:
        return [], "failed", "%s: %s" % (type(e).__name__, str(e)[:200]), 1
_RIF_T0 = time.time()
def _rif_pass(todo, npass, brk):
    consec, trips, stop = 0, 0, ""
    for g in todo:
        if not stop and consec >= brk:
            trips += 1
            if trips > 2:
                stop = "%d consecutive GeneRIF fetches failed three times in pass %d" % (brk, npass)
            else:
                print("GENERIF_FETCH %d consecutive failures in pass %d: pausing %.0f s"
                      % (consec, npass, _envf("MRAG_GENERIF_RETRY_PAUSE_S", 30)), flush=True)
                time.sleep(max(0.0, _envf("MRAG_GENERIF_RETRY_PAUSE_S", 30)))
                consec = 0
        if not stop and time.time() - _RIF_T0 > _envf("MRAG_GENERIF_BUDGET_S", 2400):
            stop = "the GeneRIF fetch time budget (%.0f s) was used up" % _envf("MRAG_GENERIF_BUDGET_S", 2400)
        if stop:
            GRF[g] = {"status": "failed", "lit": lit(g), "n": 0, "pass": npass, "attempts": 0,
                      "error": "not attempted: " + stop}
            continue
        rifs, stt, err, att = _rif_fetch(lit(g))
        GRF[g] = {"status": stt, "lit": lit(g), "n": len(rifs), "pass": npass, "attempts": att}
        if err:
            GRF[g]["error"] = err
        if stt == "failed":
            consec += 1
            continue
        consec = 0
        GR[g] = [(r if isinstance(r, str) else (r.get("text") or str(r))) for r in rifs]
        # evaudit 2026-09-27: each line's PMIDs, for the paper-level NET test and
        # the evidence audit (kept in memory; the gate records the counted ones)
        GRP[g] = [([] if isinstance(r, str) else [str(x) for x in (r.get("pmids") or [])]) for r in rifs]
for g in genes:
    GR[g] = []
    GRP[g] = []
    if g in LIT_UNMAPPED:
        GRF[g] = {"status": "not_searched", "lit": "", "n": 0, "pass": 0, "attempts": 0}
_rif_todo = [g for g in genes if g not in LIT_UNMAPPED]
if gene_generifs is None:
    for g in _rif_todo:
        GRF[g] = {"status": "failed", "lit": lit(g), "n": 0, "pass": 0, "attempts": 0,
                  "error": "genegpt not importable"}
else:
    _rif_pass(_rif_todo, 1, max(1, int(_envf("MRAG_GENERIF_BREAKER", 5))))
_rif_f1 = [g for g in genes if (GRF.get(g) or {}).get("status") == "failed"]
if _rif_f1 and gene_generifs is not None:
    _rif_pz = max(0.0, _envf("MRAG_GENERIF_RETRY_PAUSE_S", 30))
    print("GENERIF_FETCH pass 1: %d/%d gene(s) FAILED (%s); second pass in %.0f s"
          % (len(_rif_f1), len(genes), ", ".join(_rif_f1[:10]), _rif_pz), flush=True)
    time.sleep(_rif_pz)
    _rif_pass(_rif_f1, 2, max(1, int(_envf("MRAG_GENERIF_BREAKER", 5))))
print("generif done %.1fs genes=%d retmax=%d lines=%d"
      % (time.time() - t0, len(genes), RIF_RETMAX,
         sum(len(v) for v in GR.values())), flush=True)
_GRF_FAILED = [g for g in genes if (GRF.get(g) or {}).get("status") == "failed"]
_GRF_N = {k: sum(1 for v in GRF.values() if v.get("status") == k)
          for k in ("ok", "empty", "no_gene", "not_searched", "failed")}
print("GENERIF_FETCH ok=%d empty=%d no_gene=%d not_searched=%d failed=%d%s%s"
      % (_GRF_N["ok"], _GRF_N["empty"], _GRF_N["no_gene"], _GRF_N["not_searched"], _GRF_N["failed"],
         (" [" + ", ".join(_GRF_FAILED[:12]) + "]") if _GRF_FAILED else "",
         ("; second pass recovered %d/%d" % (len(_rif_f1) - len(_GRF_FAILED), len(_rif_f1))) if _rif_f1 else ""),
      flush=True)
try:
    _rif_ncbi = {}
    try:
        import genegpt as _ggm
        _rif_ncbi = {k: v for k, v in (getattr(_ggm, "CALL_STATS", None) or {}).items()}
    except Exception:
        pass
    # "pmids": each read gene's GeneRIF PMIDs, parallel to scored.json "generif" -- with it the
    # Section-14 cards reuse exactly the lines the scorer saw (followup 2026-09-28, fix 2)
    json.dump({"version": 1, "retmax": RIF_RETMAX, "n_genes": len(genes), "counts": _GRF_N,
               "failed": _GRF_FAILED, "first_pass_failed": _rif_f1, "seconds": round(time.time() - t0, 1),
               "ncbi": _rif_ncbi, "genes": GRF,
               "pmids": {g: GRP.get(g, []) for g in genes
                         if (GRF.get(g) or {}).get("status") in ("ok", "empty")}},
              open(p["scored_out"].rsplit("/", 1)[0] + "/_generif_fetch.json", "w"), indent=1)
except Exception as _gfe:
    print("GeneRIF fetch sidecar not written: %s" % str(_gfe)[:120], flush=True)
VD = {}
class _Prefetched(Exception):
    """not an error: the node already retrieved through the card retriever"""
VD_SOURCE = "runner"
_PRE = p.get("prefetched_vdb") or None
if _PRE is not None:
    VD_SOURCE = p.get("vdb_source") or "card_retriever"
    VD = {g: list(_PRE.get(g) or []) for g in genes}
    print("VDB prefetched by the node via the Section-14 card retriever: "
          "%d genes, %d hits (no second retriever opened here)"
          % (len(VD), sum(len(v) for v in VD.values())), flush=True)
try:
    if _PRE is not None:
        raise _Prefetched()
    sys.path.insert(0, "/BIO1/MRAG/Agent/libs/paperqa")
    from mrag_paperqa.retriever import Retriever
    # Qdrant local mode is single-process: a corpus job (embed/relabel) holding the
    # lock used to make every gene score with no literature. Wait for it (2026-09-15).
    rt = None
    for _att in range(40):
        try:
            rt = Retriever(device=p.get("device", "cuda")); break
        except RuntimeError as _le:
            if "already accessed" not in str(_le):
                raise
            print("VDB locked by another process, retry in 30 s", flush=True); time.sleep(30)
    if rt is None:
        raise RuntimeError("PaperQA vector store still locked after 20 min; refusing to score without literature")
    for g in genes:
        if g in LIT_UNMAPPED:
            VD[g] = []          # nothing to search: no human ortholog
            continue
        q = lit(g) + " " + p["vdb_query_suffix"]
        try:
            hits = rt.search(q, top_k=6, cancer=p["cancer_list"])
        except Exception as e:
            hits = []
            print("search err", g, str(e)[:80], flush=True)
        keep = []; gu = lit(g).upper()
        # The strict naming decision is taken HERE, on the FULL passage, because
        # only title[:120] + snippet[:220] survives into scored.json.  Judging
        # the truncated copy made the gate blind to a symbol that appears later
        # in the passage (10-13 hits per run) -- 2026-09-23 fix.
        _nm = EG.gene_fullname(lit(g), p.get("gene_info")) if EG is not None else ""
        for h in hits:
            full = str(h.title or "") + " " + str(h.text or "")
            txt = full.upper()
            rec = {"pmid": h.pmid, "cancer": h.cancer, "score": round(h.score, 3),
                   "title": str(h.title)[:120],
                   "snippet": str(h.text)[:220].replace(chr(10), " "),
                   "gene_mention": bool(gu in txt),
                   "names_gene": (EG.names_gene(full, lit(g), _nm)
                                  if EG is not None else "")}
            # findfix 2026-09-26 (finding 2): what the LLM is shown, cut from
            # the FULL passage around the gene (see evidence_gate.gene_excerpt)
            if EG is not None and hasattr(EG, "gene_excerpt"):
                _ex = EG.gene_excerpt(str(h.text or ""), lit(g), _nm, _DSPECS)
                if _ex:
                    rec["excerpt"] = _ex
            if _DSPECS:
                # Decided HERE, on the FULL passage, for the same reason
                # names_gene is: the stored snippet is a 220-char fragment and
                # the disease name often sits outside it.
                _cs = {str(x).strip().lower()
                       for x in (h.cancer if isinstance(h.cancer, (list, tuple, set))
                                 else [h.cancer]) if str(x or "").strip()}
                # 2026-09-25 (E2): the gate module's rule, so a corpus label
                # alone cannot tie an NSCLC chunk to SCLC.
                rec["domain_tie"], _tn = EG.tie_record(full, _cs, _DSPECS, gene=lit(g),
                                                       gene_name=_nm, title=str(h.title or ""))
                rec["domain_tie_v"] = (EG.tie_version() if hasattr(EG, "tie_version")
                                       else EG.DOMAIN_TIE_VERSION)
                if _tn:
                    rec["domain_tie_note"] = _tn
            keep.append(rec)
        VD[g] = keep
except _Prefetched:
    pass
except Exception as e:
    print("VDB fail", str(e)[:120], flush=True)
    if "still locked" in str(e):
        raise
    VD = {g: [] for g in genes}
EP = p["endpoint"]; MODEL = p["model"]; SYS = p["sys_prompt"]
REASONING_EFFORT = p.get("reasoning_effort")   # 2026-09-28: only when MRAG_SCORE_REASONING_EFFORT is set
RIF_SHOW = int(p.get("generif_show") or 8)
# evaudit 2026-09-27: GeneRIF lines are shown in full up to this many
# characters (node param generif_show_chars, default 600; FLRT2's line lost
# its "no difference by recurrence" result at the old 200-character cut)
RIF_CHARS = int(p.get("generif_show_chars") or 200)
def rif_line(r):
    if len(r) <= RIF_CHARS or RIF_CHARS == 200:
        return r[:RIF_CHARS]
    return r[:RIF_CHARS].rstrip() + " ..."
def rif_show(g):
    """The RIF lines the model sees.  The gate reads the whole list, but the
    prompt can only carry a few, so -- exactly as the Section-14 cards do --
    the lines that mention one of the run's domains go first."""
    rifs = GR.get(g, [])
    if not _DSPECS or EG is None or len(rifs) <= RIF_SHOW:
        return rifs[:RIF_SHOW]
    dom, rest = [], []
    for r in rifs:
        (dom if any(EG.mentions_domain(r, sp["terms"], sp.get("exclude") or ())
                    for sp in _DSPECS) else rest).append(r)
    return (dom + rest)[:RIF_SHOW]
# findfix 2026-09-26 (finding 2).  The literature lines the model sees: only
# hits the gate itself says NAME the gene (names_gene, decided on the full
# passage -- a bare substring hit such as "chromogranin A (CGA)" for CGA is no
# longer presented as gene-specific), domain-tied ones first (the card path
# already sorts that way; the in-runner path did not), each shown as the
# excerpt around the gene instead of the chunk's first 160 characters.  With
# no gate module the old substring rule and chunk start are kept.
VD_SHOW = 3
def shown_hits(g):
    hs = VD.get(g, [])
    if EG is None:
        return [h for h in hs if h.get("gene_mention")][:VD_SHOW]
    hs = [h for h in hs if h.get("names_gene")]
    return sorted(hs, key=lambda h: 0 if any((h.get("domain_tie") or {}).values()) else 1)[:VD_SHOW]
def hit_text(h):
    return str(h.get("excerpt") or "") or str(h.get("snippet") or "")[:160]
def evblock(g):
    rifs = rif_show(g)
    vd = shown_hits(g)
    # The LLM must see the symbol the evidence is actually about, or it will
    # reason about a mouse gene from human passages without knowing it.
    lines = ["GENE: " + (("%s (searched as its human ortholog %s)"
                          % (g, lit(g))) if (LIT and lit(g) != g) else g)]
    if g in LIT_UNMAPPED:
        lines.append("  (NO human ortholog: no literature was searched for "
                     "this gene; score it 0 with has_evidence=false)")
        return "\n".join(lines)
    if rifs:
        lines.append("GeneRIF:")
        for r in rifs:
            lines.append("  - " + rif_line(r))
    if vd:
        lines.append("Literature (gene-specific):")
        for h in vd:
            lines.append("  - [PMID:%s] %s: %s" % (h["pmid"], h["title"], hit_text(h)))
    if not rifs and not vd:
        lines.append("  (NO gene-specific evidence found)")
    return "\n".join(lines)
USAGE = {"calls": 0, "max_prompt_tokens": 0, "max_prompt_chars": 0}
# fix3 2026-09-28 (second verification, finding 1): what each gene was shown
# and in which batch it was scored, so a gene whose evidence an evidence filter
# removes can be RE-SCORED in its own batch (the node, rescore_after_filter)
BATCH_OF, BLOCKS, ITEMS = {}, {}, {}
def evitems(g):
    """The runner's block of `g` as items: head line, the GeneRIF lines and the
    literature lines, each with its index in GR[g] / VD[g]."""
    allr, used, ri = GR.get(g, []), set(), []
    for r in rif_show(g):
        k = next((i for i, x in enumerate(allr) if x is r and i not in used), None)
        if k is None:
            k = next((i for i, x in enumerate(allr) if x == r and i not in used), -1)
        used.add(k)
        ri.append([k, "  - " + rif_line(r)])
    allh, vi = VD.get(g, []), []
    for h in shown_hits(g):
        k = next((j for j, x in enumerate(allh) if x is h), -1)
        vi.append([k, "  - [PMID:%s] %s: %s" % (h["pmid"], h["title"], hit_text(h))])
    return {"head": evblock(g).split("\n", 1)[0], "rif": ri, "vd": vi,
            "unmapped": g in LIT_UNMAPPED}
def call(batch):
    _blk = [evblock(g) for g in batch]
    for _g, _b in zip(batch, _blk):
        if _g not in BLOCKS:
            BLOCKS[_g] = _b
            try:
                ITEMS[_g] = evitems(_g)
            except Exception:
                pass
    prompt = "Score these genes.\n\n" + "\n\n".join(_blk) + "\n\n/no_think"
    _body = {"model": MODEL, "temperature": 0, "max_tokens": 8000, "messages": [
        {"role": "system", "content": SYS}, {"role": "user", "content": prompt}]}
    if REASONING_EFFORT:
        _body["reasoning_effort"] = REASONING_EFFORT
    body = json.dumps(_body).encode()
    req = urllib.request.Request(EP, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        _resp = json.loads(r.read())
    txt = _resp["choices"][0]["message"]["content"]
    # findfix 2026-09-26: the prompt size the endpoint reports, so a prompt the
    # server had to truncate (Ollama: num_ctx) is visible in run.log/scored.json
    try:
        _pt = int(((_resp.get("usage") or {}).get("prompt_tokens")) or 0)
    except Exception:
        _pt = 0
    USAGE["calls"] += 1
    USAGE["max_prompt_tokens"] = max(USAGE["max_prompt_tokens"], _pt)
    USAGE["max_prompt_chars"] = max(USAGE["max_prompt_chars"], len(prompt))
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        raise ValueError("no JSON array (truncated?) chars=%d" % len(txt))
    return json.loads(m.group(0))
results = {}
def score_genes(batch, depth=0):
    if not batch:
        return          # deploy 2026-09-28 (decision 6): never an empty-batch LLM call
    try:
        out = call(batch)
    except Exception as e:
        print("scoreerr n=%d d=%d %s" % (len(batch), depth, str(e)[:100]), flush=True)
        out = []
    for o in out:
        if isinstance(o, dict) and o.get("gene") in batch:
            results[o["gene"]] = o
            BATCH_OF[o["gene"]] = list(batch)
    missing = [g for g in batch if g not in results]
    if not missing:
        return
    if len(batch) == 1:
        if depth < 2:
            score_genes(batch, depth + 1)
        else:
            print("UNSCORED", batch[0], flush=True)
        return
    mid = max(1, len(missing) // 2)
    # deploy 2026-09-28 (decision 6): with exactly ONE gene missing, missing[mid:] was [] and
    # call([]) sent the LLM "Score these genes." with no gene -- one wasted call per miss
    for _half in (missing[:mid], missing[mid:]):
        if _half:
            score_genes(_half, depth + 1)
for i in range(0, len(genes), 15):
    score_genes(genes[i:i + 15])
    print("scored %d/%d (%.0fs)" % (min(i + 15, len(genes)), len(genes), time.time() - t0), flush=True)
print("LLM usage: calls=%d max_prompt_tokens=%d max_prompt_chars=%d"
      % (USAGE["calls"], USAGE["max_prompt_tokens"], USAGE["max_prompt_chars"]), flush=True)
rows = []
for g in genes:
    o = results.get(g, {"score": 0, "has_evidence": False, "reason": "unscored"})
    nvdb = sum(1 for h in VD.get(g, []) if h.get("gene_mention"))
    rows.append({"gene": g, "score": o.get("score", 0),
                 "has_evidence": o.get("has_evidence", False),
                 "reason": o.get("reason", ""), "log2FC": LFC.get(g, 0),
                 "n_rif": len(GR.get(g, [])), "n_vdb_mention": nvdb,
                 "drivers": list(DRV.get(g, []))})
# --- evidence-grounding gate: deterministic, AFTER the LLM has answered ----
# A score may not claim literature support for a gene that none of the
# retrieved GeneRIF lines and none of the retrieved snippets actually names.
# Capped to the ceiling the rubric itself allows without gene-specific
# evidence ("set has_evidence=false and score<=2").
gsum = {"n": len(rows), "n_fired": 0, "n_capped": 0, "capped": []}
if EG is not None:
    # The gate asks "does this passage NAME the gene?", and the passage is
    # human literature, so the name it must look for is the human ortholog's.
    _gate_rows = rows
    if LIT:
        _gate_rows = [dict(r, gene=lit(r["gene"])) for r in rows]
        _GRl = {lit(k): v for k, v in GR.items()}
        _VDl = {lit(k): v for k, v in VD.items()}
    else:
        _GRl, _VDl = GR, VD
    # evaudit 2026-09-27 (1a): the PubMed records behind every candidate tie
    # of a domain with a paper rule (nec), fetched once, judged before the gate
    _PF = None
    _GRPl = ({lit(k): v for k, v in GRP.items()} if LIT else GRP)
    if hasattr(EG, "evidence_layer_on") and EG.evidence_layer_on():
        if _DSPECS and hasattr(EG, "paper_filter_map"):
            _PF = EG.paper_filter_map(_GRl, _GRPl, _VDl, _DSPECS, gene_info=p.get("gene_info"))
            print("PAPER_FILTER domains=%s pmids=%d records=%d net=%d%s"
                  % (",".join(_PF.get("domains") or []) or "-", _PF.get("n_pmids", 0),
                     _PF.get("n_records", 0), _PF.get("n_net", 0),
                     (" error=" + _PF["error"]) if _PF.get("error") else ""), flush=True)
        gsum = EG.apply_gate_rows(_gate_rows, _GRl, _VDl, cap=GATE_CAP, enabled=GATE_ON,
                                  gene_info=p.get("gene_info"), scope=GATE_SCOPE,
                                  domains=(_DSPECS or GATE_DOMAINS), generif_pmids_map=_GRPl,
                                  paper_net=(_PF or {}).get("net") or {})
        # fix3 2026-09-28: a tie counted without its paper check (record not
        # read) is flagged on the row -- the report shows it
        if _PF and hasattr(EG, "flag_unchecked_papers"):
            _nf = EG.flag_unchecked_papers(_gate_rows, _PF)
            if _nf:
                print("PAPER_FILTER unchecked: %d row(s) flagged 'NEC paper check not run' (%d PMID(s) "
                      "not read)" % (_nf, len(_PF.get("missing") or [])), flush=True)
    else:
        gsum = EG.apply_gate_rows(_gate_rows, _GRl, _VDl, cap=GATE_CAP, enabled=GATE_ON,
                                  gene_info=p.get("gene_info"), scope=GATE_SCOPE,
                                  domains=(_DSPECS or GATE_DOMAINS))
    if LIT:
        # copy the gate's verdict back onto the run's own symbols
        for _r, _gr in zip(rows, _gate_rows):
            for _k, _v in _gr.items():
                if _k != "gene":
                    _r[_k] = _v
    for _g, _hits in VD.items():
        _nm2 = None
        for _h in _hits:
            # Only fill a hit the retrieval loop could not decide (EG absent at
            # that point); never overwrite a decision taken on the full passage
            # with one taken on the truncated snippet.
            if "names_gene" in _h:
                continue
            if _nm2 is None:
                _nm2 = EG.gene_fullname(lit(_g), p.get("gene_info"))
            _h["names_gene"] = EG.names_gene(
                str(_h.get("title") or "") + " " + str(_h.get("snippet") or ""),
                lit(_g), _nm2)
    print("EVIDENCE_GATE %s scope=%s domains=%s fired=%d capped=%d/%d cap=%s"
          % ("on" if GATE_ON else "off", gsum.get("scope"),
             ",".join(gsum.get("domains") or []) or "-", gsum["n_fired"],
             gsum["n_capped"], gsum["n"], EG.gate_cap(GATE_CAP)), flush=True)
    if gsum.get("error"):
        raise RuntimeError("evidence gate: " + gsum["error"])
else:
    for r in rows:
        r["evidence_gate"] = "error"
# followup 2026-09-28 (fix 1): a gene whose GeneRIF record could not be read is
# FLAGGED on its row; rows of every other gene are unchanged
for _r in rows:
    _gf = GRF.get(_r["gene"]) or {}
    if _gf.get("status") != "failed":
        continue
    _r["generif_fetch"] = "failed"
    _r["generif_fetch_error"] = str(_gf.get("error") or "fetch failed")[:200]
    _fl = "GeneRIF fetch failed"
    _nt = ("the NCBI GeneRIF record of %s could not be read (%s), also in the second pass; the gene was "
           "scored without its GeneRIF lines" % (lit(_r["gene"]), str(_gf.get("error") or "fetch failed")[:160]))
    _of = str(_r.get("evidence_flag") or "")
    _r["evidence_flag"] = _fl if (not _of or _of == _fl) else "%s; %s" % (_of, _fl)
    _on = str(_r.get("evidence_flag_note") or "")
    _r["evidence_flag_note"] = _nt if not _on else "%s; %s" % (_on, _nt)
rows.sort(key=lambda r: (-r["score"], -r["log2FC"]))
for _r in rows:
    if LIT or LIT_UNMAPPED:
        _r["lit_symbol"] = LIT.get(_r["gene"], "")
        _r["lit_searched"] = bool(_r["lit_symbol"])
        if not _r["lit_searched"]:
            # Explicit, not silent: this gene was never searched, so its score
            # must not be read as "the literature says nothing about it".
            _r["lit_reason"] = "no human ortholog; literature not searched"
json.dump({"rows": rows, "generif": GR, "vdb": VD,
           "ortholog": p.get("ortholog") or {},
           "vdb_source": VD_SOURCE, "generif_retmax": RIF_RETMAX,
           "evidence_gate": {"enabled": GATE_ON,
                             "scope": gsum.get("scope"),
                             "domains": gsum.get("domains") or [],
                             "cap": (EG.gate_cap(GATE_CAP) if EG is not None else None),
                             "module": p.get("gate_module"), "error": EG_ERR,
                             "n_fired": gsum["n_fired"], "n_capped": gsum["n_capped"],
                             "capped": gsum["capped"],
                             **({"paper_filter": globals().get("_PF")}
                                if globals().get("_PF") is not None else {})},
           # followup 2026-09-28 (fix 1): only when a GeneRIF record could not be read
           **({"generif_fetch": {"version": 1, "failed": _GRF_FAILED, "counts": _GRF_N,
                                 "errors": {g: str((GRF.get(g) or {}).get("error") or "")[:200]
                                            for g in _GRF_FAILED},
                                 "sidecar": "_generif_fetch.json"}} if _GRF_FAILED else {})},
          open(p["scored_out"], "w"), indent=1)
# findfix 2026-09-26: sidecar, so scored.json's schema is unchanged -- what the
# model was shown (prompt-evidence version) and the prompt size the endpoint
# reported (a prompt near the server's num_ctx would be truncated silently).
try:
    json.dump({"llm_usage": globals().get("USAGE") or {},
               "prompt_evidence": {"version": 2, "vdb_show": globals().get("VD_SHOW"),
                                   "excerpt_chars": getattr(EG, "EXCERPT_CHARS", None),
                                   "selection": ("names_gene, domain-tied first"
                                                 if EG is not None else "gene_mention")}},
              open(p["scored_out"].rsplit("/", 1)[0] + "/_score_prompt.json", "w"), indent=1)
except Exception as _se:
    print("score prompt sidecar not written: %s" % str(_se)[:120], flush=True)
# fix3 2026-09-28: the batches and blocks, for the re-score after a filter
# (only with the 2026-09-27 evidence layers on; nothing is written otherwise)
if EG is not None and hasattr(EG, "evidence_defence_on") and EG.evidence_defence_on():
    try:
        json.dump({"version": 1, "batch_of": BATCH_OF, "blocks": BLOCKS, "items": ITEMS},
                  open(p["scored_out"].rsplit("/", 1)[0] + "/_score_batches.json", "w"))
    except Exception as _be:
        print("score batches sidecar not written: %s" % str(_be)[:120], flush=True)
print("SCORE_DONE", len(rows), flush=True)
if EG is None and GATE_ON:
    # scored.json is written first so hours of LLM work are never lost, then we
    # stop loudly: an ungated score list is exactly the bug this gate fixes.
    raise RuntimeError("evidence gate enabled but module not loadable (%s); scored.json "
                       "written UNGATED" % (EG_ERR or p.get("gate_module")))
'''


# ----------------------------------------------------------------------------
# emit
# ----------------------------------------------------------------------------
def emit_outputs(top_final, branch_dir, motifs):
    """Write targets_summary.json and curated_targets.json.

    The TF -> target edges and each gene's log2FC / ATAC fold come from `motifs`
    (pool_motifs.json)."""
    br = Path(branch_dir)
    br.mkdir(parents=True, exist_ok=True)
    top_genes = [r["gene"] for r in top_final]

    # targets_summary.json : {TF:[{Gene Name,Z_mean,db_Fold_mean,RNA_log2FC_mean}]}
    targets = {}
    for g in top_genes:
        gd = motifs.get(g, {})
        lf = gd.get("log2FC", 0.0)
        dbf = gd.get("atac_fold", 0.0)
        for tf in gd.get("tfs", {}).keys():
            targets.setdefault(tf, []).append({
                "Gene Name": g, "Z_mean": round(abs(lf), 3),
                "db_Fold_mean": round(dbf, 3), "RNA_log2FC_mean": round(lf, 3)})
    for tf in targets:
        targets[tf].sort(key=lambda x: -x["RNA_log2FC_mean"])
    (br / "targets_summary.json").write_text(json.dumps(targets, indent=2))

    # curated_targets.json : ordered ranked rows
    (br / "curated_targets.json").write_text(json.dumps(top_final, indent=2))
    return {"targets_summary_json": str(br / "targets_summary.json"),
            "curated_targets_json": str(br / "curated_targets.json")}


# ----------------------------------------------------------------------------
# Section-10 graph + Section-11 enrichment emits (generalized from the manual
# build_graph_v2.py / build_top15_enrich.py scripts)
# ----------------------------------------------------------------------------
def emit_graph_v2(top_genes, panel_table, panel, integrated_csv, graph_dir):
    """Section-10 graph. From the curated top-N genes and their bound panel
    TFs (recomputed from the panel-6 motif table), write anno.txt /
    input_inter.txt and rebuild trn_cyjs.json. Returns (counts, g2tf) so
    Section-11 reuses the same gene->bound-TF map. Existing graph files are
    backed up to _pre_curated_backup."""
    import shutil
    import pandas as pd
    sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/viz_report_mcp/src")
    from viz_report_mcp.tools.cytoscape_trn import (
        build_trn_cyjs_elements, build_trn_cyjs_style, GROUP_LETTERS)
    graph_out = Path(graph_dir)
    graph_out.mkdir(parents=True, exist_ok=True)
    pv = pd.read_csv(panel_table, sep="\t", dtype=str)
    cols = list(pv.columns)
    gi = cols.index("Gene Name")
    fdi = cols.index("Fold(DiffBind)")
    motif_cols = cols[gi + 1:fdi]
    pv["log2FC"] = pd.to_numeric(pv["log2FC"], errors="coerce")
    pv["Fold(DiffBind)"] = pd.to_numeric(pv["Fold(DiffBind)"], errors="coerce")

    edge_rows = []
    target_rows = []
    g2tf = {}
    for gene in top_genes:
        sub = pv[pv["Gene Name"] == gene]
        if sub.empty:
            continue
        lser = sub["log2FC"].dropna()
        lfc = float(lser.iloc[0]) if not lser.empty else 0.0
        fser = sub["Fold(DiffBind)"].dropna()
        fold = float(fser.mean()) if not fser.empty else 0.0
        bound = set()
        for c in motif_cols:
            hit = [t for t in panel if _mmatch(c, t)]
            if not hit:
                continue
            vals = sub[c].astype(str).str.strip()
            if vals.replace("nan", "").ne("").any():
                bound.update(hit)
        target_rows.append({"name": gene, "Fold(ATAC-seq)": fold,
                            "log2FC(RNA-seq)": lfc, "type": "Target"})
        g2tf[gene] = sorted(bound)
        for tf in sorted(bound):
            edge_rows.append({"Gene Name": gene, "TF": tf})

    imap = {}
    if integrated_csv and Path(integrated_csv).exists():
        intg = pd.read_csv(integrated_csv)
        imap = intg.set_index("Gene").to_dict("index")
    used_tfs = sorted({e["TF"] for e in edge_rows})
    tf_rows = []
    for tf in used_tfs:
        r = imap.get(tf, {})
        tf_rows.append({"name": tf, "Fold(ATAC-seq)": r.get("differential_MDscore", ""),
                        "log2FC(RNA-seq)": r.get("log2FC", ""), "type": "TF"})
    anno = pd.DataFrame(target_rows + tf_rows,
                        columns=["name", "Fold(ATAC-seq)", "log2FC(RNA-seq)", "type"])
    present = [g for g in top_genes if g in g2tf]
    letter = {g: GROUP_LETTERS[i % len(GROUP_LETTERS)] for i, g in enumerate(sorted(present))}
    inter = pd.DataFrame([{"Target": e["Gene Name"], "TF": e["TF"],
                           "Interaction": letter[e["Gene Name"]]} for e in edge_rows],
                         columns=["Target", "TF", "Interaction"])
    bk = graph_out / "_pre_curated_backup"
    bk.mkdir(exist_ok=True)
    for f in ["anno.txt", "input_inter.txt", "trn_cyjs.json", "trn_cyjs_ag.json"]:
        if (graph_out / f).exists():
            shutil.copy2(graph_out / f, bk / f)
    anno.to_csv(graph_out / "anno.txt", sep="\t", index=False, lineterminator="\r\n")
    inter.to_csv(graph_out / "input_inter.txt", sep="\t", index=False, lineterminator="\r\n")
    elements = build_trn_cyjs_elements(str(graph_out / "anno.txt"), str(graph_out / "input_inter.txt"))
    style = build_trn_cyjs_style()
    n_edges = len([e for e in elements if "source" in e["data"]])
    (graph_out / "trn_cyjs.json").write_text(json.dumps(
        {"elements": elements, "style": style,
         "n_nodes": len(elements) - n_edges, "n_edges": n_edges}, indent=2))
    counts = {"targets": len(target_rows), "tfs": len(tf_rows), "edges": len(edge_rows)}
    return counts, g2tf


def emit_enrich(g2tf, enrich_dir, dataset_context, max_tfs=0,
                min_shared_tfs=2, top_terms_total=15, panel_tfs=None):
    """Section-11 enrichment. From the curated gene->bound-TF map build a
    synthetic ranked-consensus TSV and run enrich_shared_targets +
    enrich_per_tf into enrich_dir.

    2026-10-17: the heatmap spans every TF of the network, the panel and the ones the SE layer
    added, because that is the network Section 10 draws.  panel_tfs no longer drops the rest --
    it orders them first and names the two groups on the figure."""
    import pandas as pd
    from collections import defaultdict
    sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/enrichment_mcp/src")
    from enrichment_mcp.tools.enrich import register_enrichment_tools
    from ._base import capture_mcp_tools          # MRAG_TOOL_TRANSPORT (2026-10-12): mcp | direct
    reg = capture_mcp_tools(register_enrichment_tools)
    Path(enrich_dir).mkdir(parents=True, exist_ok=True)
    tf2genes = defaultdict(list)
    for g, tfs in g2tf.items():
        for t in tfs:
            tf2genes[t].append(g)
    tf_by_n = sorted(tf2genes, key=lambda t: -len(tf2genes[t]))
    rows = []
    for ti, t in enumerate(tf_by_n):
        for gj, g in enumerate(sorted(tf2genes[t])):
            rows.append({"TF": t, "Gene Name": g, "rank_within_tf": gj + 1,
                         "combined_p": 1e-4 * (ti + 1) + 1e-6 * gj})
    df = pd.DataFrame(rows)
    tsv = str(Path(enrich_dir) / "top15_consensus.tsv")
    df.to_csv(tsv, sep="\t", index=False)
    # one lineage lookup, shared by the two figures
    lin = reg_lineage(dataset_context, enrich_dir)
    shared = json.loads(reg["enrich_shared_targets"](
        tsv, enrich_dir, min_shared_tfs=min_shared_tfs,
        top_terms_per_lib=10, dataset_context=dataset_context, lineage=lin))
    per_tf = json.loads(reg["enrich_per_tf"](
        tsv, enrich_dir, max_tfs=max_tfs, top_terms_total=top_terms_total,
        dataset_context=dataset_context, panel_tfs=list(panel_tfs or []), lineage=lin))
    return {"lineage": (lin or {}).get("lineage", ""),
            "shared": {"status": shared.get("status"), "n_shared": shared.get("n_shared"),
                       "n_terms": shared.get("n_terms"),
                       "p_used": shared.get("p_used")},
            "per_tf": {"status": per_tf.get("status"), "tfs": per_tf.get("tfs"),
                       "n_terms": per_tf.get("n_terms"), "fill": per_tf.get("fill"),
                       "p_used": per_tf.get("p_used")}}


def reg_lineage(dataset_context, enrich_dir):
    """The run's lineage, resolved once and cached beside the figures."""
    sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/enrichment_mcp/src")
    from enrichment_mcp.tools.enrich import resolve_lineage
    return resolve_lineage(dataset_context, str(Path(enrich_dir) / "_llm_lineage.json"))


# ----------------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------------
class CuratedTargetsNode(Node):
    """Inputs:  motif_tsv, pval2_txt, top6_json
    Params:   branch_dir, cell_line, tissue, cancer_type, perturbation, concept,
              lfc_min(2.0), min_union(3), pool_n(200), top_final(15),
              min_score(None -> no evidence floor),
              model, endpoint, vdb_cancer(list), device,
              score_python,
              evidence_gate(True -> cap ungrounded literature scores;
                            env MRAG_TARGET_EVIDENCE_GATE),
              evidence_gate_cap(2.0; env MRAG_TARGET_EVIDENCE_CAP),
              evidence_gate_scope("all" = GeneRIF or snippet, default |
                            "literature" = snippets only |
                            "domain" = one passage must name the gene AND be
                            tied to the run's literature domain;
                            env MRAG_TARGET_EVIDENCE_SCOPE),
              evidence_gate_domains(scope="domain" only; corpus keys, default
                            the run's own user domains from
                            literature_domains / vdb_cancer),
              gene_info(local Homo_sapiens.gene_info.gz; env MRAG_GENE_INFO),
              curated_card_retrieval(True -> retrieve the scorer's literature
                            through the Section-14 card retriever, per domain
                            and gene-name prefiltered; False -> the legacy
                            pooled in-runner query;
                            env MRAG_CURATED_CARD_RETRIEVAL),
              curated_card_top_k(3 per corpus; env MRAG_CURATED_CARD_TOP_K),
              curated_generif_retmax(400 -- the same efetch either way;
                            env MRAG_CURATED_GENERIF_RETMAX),
              evidence_pubmed(True -> live PubMed check for the genes the gate
                            capped, records verified before they count;
                            env MRAG_TARGET_EVIDENCE_PUBMED),
              evidence_pubmed_retmax(20 -- deploy 2026-09-28; 5 with MRAG_EVIDENCE_AUDIT=0;
                            env MRAG_TARGET_EVIDENCE_PUBMED_RETMAX),
              literature_domains_user(the Section-2 user domain list; the
                            score prompt's Domain of interest, 2026-09-26),
              score_domain_frame("literature" default | "study" = the
                            pre-2026-09-26 cancer_type/cell-line wording;
                            env MRAG_SCORE_DOMAIN_FRAME),
              evidence_audit(True -> post-scoring evidence audit of the
                            emitted targets, nodes/evidence_audit.py; False
                            -> the LLM audit only is off), evidence_audit_limit(40;
                            env MRAG_EVIDENCE_AUDIT_LIMIT),
                            evidence_audit_mode(flag (default, fix3) | remove_named
                            | remove | off; env MRAG_EVIDENCE_AUDIT_MODE), evidence_audit_keep
                            (genes never removed; env MRAG_EVIDENCE_AUDIT_KEEP),
              rescore_after_filter(True; env MRAG_RESCORE_AFTER_FILTER),
              scorer (deploy 2026-09-28): env MRAG_SCORER_PROFILE = q38 (default:
                            qwen3.8:27b-q8_0, reasoning_effort low, 127.0.0.1:11501,
                            ensure_scorer_ollama.sh --warm before scoring; the re-score
                            and the evidence audit follow it) | legacy (cfg.llm model
                            and endpoint, the pre-deploy scorer byte for byte); single
                            overrides MRAG_SCORE_MODEL / _ENDPOINT / _REASONING_EFFORT /
                            _ENSURE_CMD / _PREFLIGHT / _MAX_UNSCORED_FRAC, MRAG_SCORE_NEC_NOTE,
              master switch env MRAG_EVIDENCE_DEFENCE=0 or MRAG_EVIDENCE_AUDIT=0
                            (all 2026-09-27 layers off, byte for byte; fix3),
              env MRAG_EVIDENCE_AUDIT_MODE=off (the LLM audit only; fix3),
              curated_generif_show_chars(600; env MRAG_GENERIF_SHOW_CHARS)
    Outputs:  panel_table, pool_motifs_json,
              scored_json, targets_summary_json,
              curated_targets_json,
              vdb_source, card_retrieval, pubmed_rescue, pubmed_rescued_in_top,
              evidence_audit (only when the audit is on)
    """
    name = "curated_targets"

    def run(self) -> None:
        # MCP step 2a (2026-10-12): this node's analysis stages are MCP tools --
        # trn-mcp build_panel_motif_table / build_target_pool / scan_target_motifs / score_target_genes /
        # rescore_target_genes / select_targets, paperqa-mcp retrieve_target_literature /
        # evidence_pubmed_rescue (nodes/_stage_tools.py).  MRAG_TOOL_TRANSPORT=mcp (the default): the names below
        # are adapters that make each call an MCP tools/call (the tools run this module's functions on the same
        # arguments and the rows / scored document come back into this node's objects); =direct: this module's own
        # functions and _score_via_mcp None, i.e. the node exactly as before step 2a.
        (build_panel_table, build_pool, scan_motifs, prefetch_card_hits, _evgate,
         rescore_after_filter, select_targets, _score_via_mcp) = _stage.curated_bindings(sys.modules[__name__])
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._generif_fetch, self.outputs_generif = {}, None     # followup 2026-09-28 (fix 1)
        self._generif_cand = []
        self._rescue_fail, self.outputs_rescue = {}, None      # followup r3 (post-deploy finding 1)
        motif_tsv = self.inputs["motif_tsv"]
        pval2_txt = self.inputs["pval2_txt"]
        top6_json = self.inputs["top6_json"]
        panel = load_panel(top6_json)

        # 1. panel motif table (cached next to the Pval2). The cache key carries
        # the panel SIZE -- a fixed ".top6" name made a 6 -> 10 TF switch
        # silently reuse the old 6-TF table -- and the cached file is also
        # verified to actually carry every panel TF, since two different
        # panels of the same size would otherwise collide under one name.
        pv_dir = Path(pval2_txt).parent
        base = Path(pval2_txt).name.replace(".txt", "")
        panel_table = pv_dir / (base + ".top%d.txt" % len(panel))
        _pt_stale = True
        if panel_table.exists():
            try:
                with open(panel_table) as _fh:
                    _hdr = _fh.readline().rstrip("\n").split("\t")
                _missing = [t for t in panel
                            if not any(str(c).upper().startswith(t.upper() + "(")
                                       or str(c).upper() == t.upper() for c in _hdr)]
                # 2026-09-11: the TF<->column rules evolve (aliases, composites); a
                # cached table built by older rules passes the coverage check but
                # carries the wrong columns -> always rebuild (~10 s).
                _pt_stale = True
                if _missing:
                    print("[curated] cached panel table missing %d/%d panel TFs %s "
                          "-> rebuilding" % (len(_missing), len(panel), _missing),
                          flush=True)
            except Exception:
                _pt_stale = True
        if _pt_stale or self.params.get("rebuild_panel_table"):
            print("[curated] building panel-%d motif table from .motif.tsv ..."
                  % len(panel), flush=True)
            _, found = build_panel_table(motif_tsv, pval2_txt, panel, str(panel_table))
            print("[curated] panel table cols:", found, flush=True)

        # 2. pool + 3. motif scan
        _lfc_min = float(self.params.get("lfc_min", 2.0))
        _min_union = int(self.params.get("min_union", 3))
        _pool_n = int(self.params.get("pool_n", 200))
        _min_targets = int(self.params.get("min_targets", 15))
        pool = build_pool(str(panel_table), panel, lfc_min=_lfc_min,
                          min_union=_min_union, pool_n=_pool_n)
        # Adaptive relaxation: low-fold-change contrasts (e.g. TCGA tumor
        # subtypes) yield too few candidates at the default lfc>2 / union>=3.
        # If we can't reach >=min_targets, relax shared-TF to 2 and log2FC to 1.
        if len(pool) < _min_targets and (_min_union > 2 or _lfc_min > 1.0):
            _r_union, _r_lfc = min(_min_union, 2), min(_lfc_min, 1.0)
            print("[curated] pool=%d < %d -> relaxing min_union %d->%d, lfc_min "
                  "%.2f->%.2f" % (len(pool), _min_targets, _min_union, _r_union,
                                  _lfc_min, _r_lfc), flush=True)
            _min_union, _lfc_min = _r_union, _r_lfc
            pool = build_pool(str(panel_table), panel, lfc_min=_lfc_min,
                              min_union=_min_union, pool_n=_pool_n)
            print("[curated] relaxed pool=%d" % len(pool), flush=True)
        (out_dir / "pool.json").write_text(json.dumps(pool, indent=1))
        motifs = scan_motifs(pool, str(panel_table), panel,
                             fasta_path=self.params.get("hg38_fa",
                                 "/BIO1/MRAG/Agent/Reference/ATAC_hg38/hg38.fa"))
        motifs_path = out_dir / "pool_motifs.json"
        motifs_path.write_text(json.dumps(motifs, indent=1))
        print("[curated] pool=%d motif-genes=%d" % (len(pool), len(motifs)), flush=True)

        # 5. candidates: every pool gene with a bound panel-TF motif (no AlphaGenome gate)
        passing = [g for g, v in motifs.items() if (v or {}).get("tfs")]
        print("[curated] candidates=%d/%d (a bound panel-TF motif; no AlphaGenome gate)"
              % (len(passing), len(motifs)), flush=True)
        if not passing:
            raise RuntimeError(
                "curated: no pool gene carries a panel-TF motif (pool=%d, motif-genes=%d) -- nothing to score"
                % (len(pool), len(motifs)))

        # 6. evidence + LLM domain scoring (paper_qa venv)
        # Gate settings are resolved BEFORE the prompt: the rubric the model is
        # given has to state the same ceiling the gate will enforce afterwards
        # (2026-09-23 fix).
        _gate_on = _evgate.gate_enabled(self.params.get("evidence_gate"))
        _gcap = _evgate.gate_cap(self.params.get("evidence_gate_cap"))
        _gscope = _evgate.gate_scope(self.params.get("evidence_gate_scope"))
        # scope="domain": resolve the run's own literature domains and hand the
        # score runner PLAIN specs -- it runs in the paper_qa venv, which cannot
        # import mrag_orchestrator.  vdb_cancer IS the resolved corpus list, so
        # user_domains() strips the auto-appended pan-cancer corpus from it.
        # The run's literature domains are resolved for EVERY run now, not
        # only under scope="domain": the card retriever queries per domain and
        # the live-PubMed rescue needs the domain's PubMed fragment.  Only
        # scope="domain" still treats "no domain" as fatal -- there it would
        # cap every gene on a test that never ran.
        _gdomains, _gdlabel, _gkeys = [], "", []
        try:
            from .. import literature as _lit
            _gkeys = _lit.known_keys(self.params.get("evidence_gate_domains")) or \
                _lit.user_domains(self.params.get("literature_domains")
                                  or self.params.get("vdb_cancer"))
            _gdomains = _lit.domain_specs(_gkeys)
        except Exception as _lerr:
            print("[curated] literature domains not resolvable (%s)"
                  % str(_lerr)[:140], flush=True)
        if _gate_on and _gscope == "domain":
            if not _gdomains:
                raise ValueError(
                    "evidence_gate_scope='domain' needs a literature domain, and "
                    "none could be resolved from evidence_gate_domains/"
                    "literature_domains/vdb_cancer (%r). Capping every gene on a "
                    "test that never ran would be worse than not gating."
                    % (self.params.get("vdb_cancer"),))
            # the rubric's own wording only changes under scope="domain"
            _gdlabel = _lit.domains_label(_gkeys, include_pancancer=True)
            print("[curated] evidence_gate scope=domain domains=%s"
                  % ([d["key"] for d in _gdomains],), flush=True)
        # PI decision 2026-09-26: the Domain of interest is the run's user
        # literature domain list (the Section 2 "LLM domain" row: the runner's
        # literature_domains_user, else vdb_cancer without the auto-appended
        # pan-cancer), the experimental model is context.
        _fkeys = []
        try:
            from .. import literature as _lit2
            _fkeys = [k for k in _lit2.resolve_user_domains({
                "literature_domains_user": self.params.get("literature_domains_user"),
                "literature_domains": (self.params.get("literature_domains")
                                       or self.params.get("vdb_cancer"))}) if k != "pancancer"]
        except Exception as _ferr:
            print("[curated] user literature domains not resolvable for the score "
                  "prompt (%s); study frame" % str(_ferr)[:140], flush=True)
        _sframe = build_sys_prompt_detail(
            cancer_type=self.params.get("cancer_type"),
            tissue=self.params.get("tissue"),
            cell_line=self.params.get("cell_line"),
            perturbation=self.params.get("perturbation"),
            concept=self.params.get("concept"),
            evidence_scope=(_gscope if _gate_on else "all"),
            evidence_cap=_gcap,
            evidence_domains=_gdlabel,
            lit_domains=_fkeys,
            frame=self.params.get("score_domain_frame"))
        sys_prompt = _sframe["sys_prompt"]
        self._score_frame = {k: _sframe[k] for k in (
            "frame", "frame_source", "frame_requested", "domains",
            "domain_of_interest", "experimental_model", "floor_domains")}
        if _sframe.get("domain_notes"):
            # deploy 2026-09-28 (decision 5): the nec note is part of the rubric
            self._score_frame["domain_notes"] = list(_sframe["domain_notes"])
        if str(_sframe["frame_source"]).startswith("invalid:"):
            print("[curated] WARNING score_domain_frame %s is not one of %s; using %s"
                  % (_sframe["frame_source"][8:], SCORE_DOMAIN_FRAMES, _sframe["frame_requested"]),
                  flush=True)
        print("[curated] score domain frame=%s (%s): Domain of interest: %s%s"
              % (_sframe["frame"], _sframe["frame_source"], _sframe["domain_of_interest"],
                 (" | experimental model: " + _sframe["experimental_model"])
                 if _sframe["experimental_model"] else ""), flush=True)
        if _sframe.get("domain_notes"):
            print("[curated] score rubric: domain note for %s (%s)"
                  % (", ".join(_sframe["domain_notes"]), "MRAG_SCORE_NEC_NOTE=0 removes it"), flush=True)
        vdb_suffix = self.params.get("vdb_query_suffix")
        if not vdb_suffix:
            vdb_suffix = "in " + (self.params.get("concept") or "") + " " + \
                         (self.params.get("cancer_type") or self.params.get("tissue") or "cancer")
            vdb_suffix = vdb_suffix.strip()
        # --- human-ortholog route for the literature layer -----------------
        # MRAG's evidence layer is a HUMAN index. For a non-human run every
        # gene is translated to its human ortholog before retrieval, and the
        # translation travels with the result so the report can disclose it.
        from ..orthology import get_map as _get_omap
        _omap = _get_omap(self.params.get("organism") or "human")
        _lit_hits = _omap.map_many(passing)
        _lit_symbol = {g: h.human_symbol for g, h in _lit_hits.items()
                       if h.mapped and not h.is_identity}
        _lit_unmapped = sorted(g for g, h in _lit_hits.items() if not h.mapped)
        _ortho_summary = _omap.summarise(passing)
        _ortho_summary["disclosures"] = {
            g: h.disclosure() for g, h in _lit_hits.items() if h.disclosure()}
        try:
            from ..organism import resolve as _res_org
            _oprof = _res_org(self.params.get("organism") or "human")
            _ortho_summary["banner"] = _omap.banner(_oprof.scientific_name,
                                                    _oprof.genome)
            _ortho_summary["scientific_name"] = _oprof.scientific_name
            _ortho_summary["genome"] = _oprof.genome
        except Exception:
            _ortho_summary["banner"] = _omap.banner(
                str(self.params.get("organism") or ""), "")
        self._ortholog = _ortho_summary
        if not _omap.identity:
            print("[curated] human-ortholog route: %d/%d genes mapped, %d with "
                  "no human ortholog (not searched)%s"
                  % (_ortho_summary["n_mapped"], _ortho_summary["n_total"],
                     _ortho_summary["n_unmapped"],
                     (": " + ", ".join(_lit_unmapped[:8])) if _lit_unmapped else ""),
                  flush=True)
        # --- literature retrieval, through the Section-14 card retriever ---
        _card_on = card_retrieval_enabled(self.params.get("curated_card_retrieval"))
        _rifmax = generif_retmax(self.params.get("curated_generif_retmax"))
        _rifchars = generif_show_chars(self.params.get("curated_generif_show_chars"))
        _pre, _cardstats = None, {"enabled": _card_on, "used": False,
                                  "error": "" if _card_on else "switched off"}
        if _card_on and _gdomains:
            try:
                _pre, _cardstats = prefetch_card_hits(
                    passing, _gdomains,
                    self.params.get("vdb_cancer", ["pancancer"]), vdb_suffix,
                    top_k=self.params.get("curated_card_top_k"),
                    device=self.params.get("device", "cuda"),
                    gene_info=self.params.get("gene_info"),
                    lit_symbol=_lit_symbol, lit_unmapped=_lit_unmapped)
            except Exception as _cerr:
                _pre = None
                _cardstats = {"enabled": True, "used": False,
                              "error": "%s: %s" % (type(_cerr).__name__, str(_cerr)[:200])}
                print("[curated] card retrieval failed (%s); the score runner "
                      "falls back to its own pooled retrieval" % _cardstats["error"],
                      flush=True)
        elif _card_on:
            _cardstats["error"] = ("no literature domain resolved; the score "
                                   "runner keeps its own pooled retrieval")
            print("[curated] " + _cardstats["error"], flush=True)
        score_params = out_dir / "_score_params.json"
        scored_out = out_dir / "scored.json"
        _score_model, _score_ep, _score_re, _score_env = score_llm_settings(
            self.params.get("model", "qwen3.6:35b-a3b"),
            self.params.get("endpoint", "http://127.0.0.1:11500/v1/chat/completions"))
        _score_prof = scorer_profile()
        _score_ovr = sorted(k for k in _score_env if k.startswith("MRAG_SCORE_"))
        print("[curated] scorer profile %s (%s): model=%s endpoint=%s reasoning_effort=%s%s"
              % (_score_prof[0], "MRAG_SCORER_PROFILE" if _score_prof[1] == "env" else "default",
                 _score_model, _score_ep, _score_re or "-",
                 ("; overridden by " + ", ".join(_score_ovr)) if _score_ovr else ""), flush=True)
        # 2026-09-28 (v2): fail here, not with 249 x score 0, when the scorer cannot answer
        _pf = score_preflight(_score_ep, _score_model)
        if not _pf.get("skipped"):
            print("[curated] scorer preflight ok: %s serves %s%s" % (
                _pf.get("base"), _score_model,
                (" (ensure command ran: %s)" % _pf.get("ensure")) if _pf.get("ensure_cmd") else ""), flush=True)
        _score_extra = {}
        if _score_re:
            _score_extra["reasoning_effort"] = _score_re
        if _score_env:
            _score_extra["score_llm_env"] = _score_env
        if _sframe.get("domain_notes"):
            # deploy 2026-09-28 (decision 5): the domain note(s) the rubric carries (nec)
            _score_extra["score_domain_notes"] = list(_sframe["domain_notes"])
        score_params.write_text(json.dumps({
            **_score_extra,
            "genes": passing, "motifs_path": str(motifs_path), "scored_out": str(scored_out),
            "sys_prompt": sys_prompt,
            # PI decision 2026-09-26: which frame built the rubric, and with what
            "score_domain_frame": self._score_frame["frame"],
            "score_domain_frame_source": self._score_frame["frame_source"],
            "score_domain_frame_requested": self._score_frame["frame_requested"],
            "score_domain_frame_env": os.environ.get("MRAG_SCORE_DOMAIN_FRAME", ""),
            "score_domains": self._score_frame["domains"],
            "score_domain_of_interest": self._score_frame["domain_of_interest"],
            "score_experimental_model": self._score_frame["experimental_model"],
            "model": _score_model,
            "endpoint": _score_ep,
            "vdb_query_suffix": vdb_suffix,
            "cancer_list": self.params.get("vdb_cancer", ["pancancer"]),
            "device": self.params.get("device", "cuda"),
            # evidence-grounding gate (2026-09-23); safe behaviour (on, cap 2)
            # is the default of both the param and MRAG_TARGET_EVIDENCE_GATE.
            "gate_module": str(Path(_evgate.__file__).resolve()),
            "gene_info": self.params.get("gene_info"),
            # human-ortholog route (2026-09-23 Task B); empty for a human run
            "lit_symbol": _lit_symbol,
            "lit_unmapped": _lit_unmapped,
            "ortholog": _ortho_summary,
            "evidence_gate": _gate_on,
            "evidence_gate_cap": _gcap,
            "evidence_gate_scope": _gscope,
            "evidence_gate_domains": _gdomains,
            # 2026-09-23: retrieval done by the node through the card retriever
            # (None -> the runner opens its own Retriever, unchanged), and the
            # GeneRIF parse cap, which costs nothing extra on the wire.
            "generif_retmax": _rifmax,
            # evaudit 2026-09-27 (absent = the runner's legacy 200-char cut)
            **({"generif_show_chars": _rifchars} if _rifchars != 200 else {}),
            "prefetched_vdb": _pre,
            "vdb_source": ("card_retriever" if _pre else "runner")}))
        score_runner = out_dir / "_score_runner.py"
        score_runner.write_text(_SCORE_RUNNER)
        score_py = self.params.get("score_python", "/BIO1/MRAG/venvs/paper_qa/bin/python3")
        print("[curated] evidence + LLM domain scoring (%d genes) ..." % len(passing), flush=True)
        # 2026-09-25 (cardtimeout): MRAG_CURATED_SCORE_TIMEOUT_S can only LENGTHEN
        # the 3 h limit (overloaded host); unset = unchanged.
        _sto = 10800
        try:
            _sto = max(_sto, float(os.environ.get('MRAG_CURATED_SCORE_TIMEOUT_S', '') or 0))
        except ValueError:
            pass
        # fix3 2026-09-28: a stale batch record of an earlier run must never
        # serve as the re-score context of this one
        _bpath = Path(str(scored_out)).parent / "_score_batches.json"
        if _evgate.evidence_defence_on() and _bpath.exists():
            try:
                _bpath.unlink()
            except Exception:
                pass
        # followup 2026-09-28 (fix 1): nor may an earlier run's GeneRIF fetch record
        _gfpath = Path(str(scored_out)).parent / GENERIF_SIDECAR
        if _gfpath.exists():
            try:
                _gfpath.unlink()
            except Exception:
                pass
        if _score_via_mcp is None:
            sproc = subprocess.run([score_py, str(score_runner), str(score_params)],
                                   capture_output=True, text=True, timeout=_sto)
        else:   # MCP step 2a: the LLM target scoring stage = the trn-mcp tool score_target_genes (same command)
            sproc = _score_via_mcp(score_py, score_runner, score_params, _sto)
        if sproc.returncode != 0 or not scored_out.exists():
            raise RuntimeError("scoring failed (rc=%d): %s" % (sproc.returncode, sproc.stderr[-2000:]))
        print(sproc.stdout[-500:], flush=True)
        _scored_doc = json.loads(scored_out.read_text())
        scored = _scored_doc["rows"]
        # followup 2026-09-28 (fix 1): the GeneRIF fetch outcome; too many failed genes stop here
        self._generif_fetch = generif_fetch_summary(scored_out, _scored_doc)
        if self._generif_fetch.get("recovered"):
            print("[curated] GeneRIF fetch: %d gene(s) failed in the first pass and were read in the second: %s"
                  % (len(self._generif_fetch["recovered"]), ", ".join(self._generif_fetch["recovered"][:20])),
                  flush=True)
        if self._generif_fetch.get("failed"):
            print("[curated] WARNING: the NCBI GeneRIF record of %d/%d gene(s) could NOT be read, also in the "
                  "second pass: %s -- scored without their GeneRIF lines and flagged 'GeneRIF fetch failed'"
                  % (len(self._generif_fetch["failed"]), self._generif_fetch.get("n") or len(scored),
                     "; ".join("%s (%s)" % (g, str(self._generif_fetch["errors"].get(g) or "")[:80])
                               for g in self._generif_fetch["failed"][:12])), flush=True)
        check_generif_fetch(self._generif_fetch)
        # review 2026-09-28 (finding 1): ANY failed gene that could have been emitted stops the step
        # here, before the PubMed rescue, the re-score and the audit (MRAG_GENERIF_FAIL_ON_TOP=0: go on)
        self._generif_cand = check_generif_candidates(scored, panel, self._generif_fetch)
        if self._generif_cand:
            print("[curated] WARNING: going on (MRAG_GENERIF_FAIL_ON_TOP=0) with %d gene(s) that could have been "
                  "targets although their GeneRIF record could not be read: %s"
                  % (len(self._generif_cand), ", ".join(self._generif_cand[:20])), flush=True)
        _n_unscored, _n_rows, _unscored = check_unscored(scored)
        if _n_unscored:
            print("[curated] WARNING: %d/%d genes unscored (score 0): %s"
                  % (_n_unscored, _n_rows, ", ".join(map(str, _unscored[:20]))), flush=True)

        # --- live PubMed rescue, scoped to the genes the gate actually capped -
        # The corpora are snapshots of very different sizes, so "absent from our
        # index" is at least as often "small corpus" as "nobody published it".
        # Only a row the gate CAPPED is checked, every record is fetched back and
        # must name the gene AND pass the domain terms, and the result is stored
        # distinctly (evidence_source="pubmed" + the PMIDs) so a reader can always
        # tell corpus evidence from a live record.
        _resc = {"enabled": False, "checked": 0, "rescued": 0, "rescued_genes": [],
                 "error": "gate off or no literature domain"}
        _resc_cand = []
        if _gate_on and _gdomains:
            # followup r3 (post-deploy finding 1): the genes the rescue must decide, taken before it runs
            _resc_cand = rescue_candidates(scored, _gcap)
            try:
                _resc = _evgate.pubmed_rescue(
                    scored, _gdomains, cap=_gcap,
                    enabled=self.params.get("evidence_pubmed"),
                    retmax=self.params.get("evidence_pubmed_retmax"),
                    gene_info=self.params.get("gene_info"),
                    lit_symbol=_lit_symbol)
            except Exception as _rerr:
                _resc = {"enabled": True, "checked": 0, "rescued": 0,
                         "rescued_genes": [],
                         "error": "%s: %s" % (type(_rerr).__name__, str(_rerr)[:200])}
                print("[curated] pubmed rescue failed (%s); every gate verdict "
                      "stands unchanged" % _resc["error"], flush=True)
        # FIX 2026-09-23 (review): failed=n/n makes a network failure visible
        # in the log -- without it a timed-out NCBI printed the same line as a
        # check that ran and found nothing.
        print("[curated] pubmed rescue: enabled=%s checked=%d rescued=%d%s "
              "failed=%d/%d (%ss)"
              % (_resc.get("enabled"), _resc.get("checked", 0), _resc.get("rescued", 0),
                 (" " + ",".join(_resc.get("rescued_genes") or [])
                  if _resc.get("rescued_genes") else ""),
                 _resc.get("n_errors") or 0, _resc.get("n_pairs") or 0,
                 _resc.get("seconds")), flush=True)
        if _resc.get("error"):
            print("[curated] pubmed rescue note: " + str(_resc["error"]), flush=True)
        # followup r3 (post-deploy finding 1): a candidate the rescue could not decide (a failed / never run
        # check, nothing verified) is never silent -- recorded, and the step stops when it could have been emitted
        self._rescue_fail = rescue_failure_record(scored, _resc, _resc_cand, panel)
        if self._rescue_fail:
            _rf = self._rescue_fail
            _scored_doc["pubmed_rescue_failed"] = _rf
            if _rf.get("failed"):
                print("[curated] WARNING: the live PubMed rescue check of %d/%d capped candidate gene(s) FAILED, "
                      "also on a second try, so they stay capped without a decision: %s"
                      % (len(_rf["failed"]), _rf.get("n_candidates") or 0,
                         "; ".join("%s (%s)" % (g, str(_rf["errors"].get(g) or "")[:80])
                                   for g in _rf["failed"][:12])), flush=True)
            if _rf.get("partial"):
                print("[curated] WARNING: admitted by the PubMed rescue although the check of another domain "
                      "failed: %s" % ", ".join(_rf["partial"][:20]), flush=True)
            if _rf.get("could_emit") and rescue_fail_on_top():
                # the stop leaves the rescue record in scored.json (the same fields as a finished step)
                _scored_doc["rows"] = scored
                _scored_doc["pubmed_rescue"] = _resc
                _scored_doc["card_retrieval"] = _cardstats
                scored_out.write_text(json.dumps(_scored_doc, indent=1))
            _rcould = check_rescue_candidates(_rf)
            if _rcould:
                print("[curated] WARNING: going on (MRAG_RESCUE_FAIL_ON_TOP=0) with %d gene(s) the rescue could "
                      "have admitted: %s" % (len(_rcould), ", ".join(_rcould[:20])), flush=True)
        # rewrite scored.json so the rescue is auditable even when it rescued
        # nothing (it is evidence that the check ran, not that it was skipped)
        _scored_doc["rows"] = scored
        _scored_doc["pubmed_rescue"] = _resc
        _scored_doc["card_retrieval"] = _cardstats
        # fix2 2026-09-28: a gene whose counted evidence an evidence filter
        # removed and that the rescue (or an alias GeneRIF line) re-admitted is
        # re-scored on the evidence counted now -- never its old score / reason
        _rsc = None
        if _gate_on and _gdomains and rescore_enabled(self.params.get("rescore_after_filter")):
            # 2026-09-28 (ollama034 v2): a re-score is a SCORE -> the scorer's model, endpoint
            # and reasoning_effort (deploy 2026-09-28: the evidence audit follows the scorer too)
            _rsc = rescore_after_filter(
                scored, _scored_doc, _gdomains, sys_prompt,
                _score_ep, _score_model, rif_chars=_rifchars,
                lit_symbol=_lit_symbol, batch_ctx=load_batch_context(_bpath),
                llm=(_RescoreLLM(reasoning_effort=_score_re) if _score_re else None))
            print("[curated] re-score after filter: %d gene(s)%s, %d failed"
                  % (_rsc["n"], (" " + ", ".join("%s %s->%s" % (x["gene"], x["old_score"], x["new_score"])
                                                  for x in _rsc["genes"])) if _rsc["genes"] else "",
                     _rsc["n_errors"]), flush=True)
            if _rsc["n"]:
                _scored_doc["rescore_after_filter"] = _rsc
        scored_out.write_text(json.dumps(_scored_doc, indent=1))

        # 7. rerank -> top_final, exclude panel
        _min_score = self.params.get("min_score")
        _minf = None if _min_score is None else float(_min_score)
        if _gate_on and _gscope in ("literature", "domain"):
            # Under either strict scope the gate parks every ungrounded gene AT
            # the cap, so a floor at or below the cap lets those genes fill the
            # top-N re-ordered by log2FC alone.  The floor must sit above it
            # (2026-09-23 fix).  Measured on the four domain runs with NO
            # floor: scope="literature" leaves 8-11 capped genes inside the
            # top-20 and scope="domain" 0-17, so this is not a corner case.
            _floor = _gcap + 1.0
            if _minf is None or _minf <= _gcap:
                print("[curated] evidence_gate scope=%s: min_score %s -> %s "
                      "(must exceed the cap %s)" % (_gscope, _minf, _floor, _gcap),
                      flush=True)
                _minf = _floor
        # evaudit 2026-09-27: post-scoring EVIDENCE AUDIT (select_targets ->
        # nodes/evidence_audit.py).  Off (MRAG_EVIDENCE_AUDIT_MODE=off / param
        # evidence_audit=False; MRAG_EVIDENCE_AUDIT=0 switches every layer off,
        # fix3): the old rerank, byte for byte, no audit field.
        top_final, _audit_sum = select_targets(
            scored, panel, int(self.params.get("top_final", 15)), _minf,
            _gate_on, _gscope, _gdomains, _scored_doc, self.params, out_dir,
            # deploy 2026-09-28 (decision 1): the evidence audit follows the scorer
            audit_endpoint=_score_ep, audit_model=_score_model)
        if _audit_sum is not None and _audit_sum.get("ran"):
            # scored.json carries every audited row's verdict, emitted or not
            _scored_doc["rows"] = scored
            _scored_doc["evidence_audit"] = _audit_sum
            scored_out.write_text(json.dumps(_scored_doc, indent=1))
        # followup 2026-09-28 (fix 1): a target whose GeneRIF record could not be read would be
        # shown with a score and rank decided without it -> stop loudly (scored.json is kept).
        # Review 2026-09-28: check_generif_candidates() above already stopped for every failed gene
        # that could have been emitted; this stays as a safety net and for the =0 warning.
        _gf_top = [r.get("gene") for r in top_final if r.get("generif_fetch") == "failed"]
        if _gf_top:
            if generif_fail_on_top():
                raise RuntimeError(
                    "GeneRIF fetch: the NCBI GeneRIF record of %d gene(s) that would be EMITTED as targets could "
                    "not be read, also in the second pass: %s -- their scores and ranks were decided without "
                    "it; scored.json and _generif_fetch.json are kept, targets were NOT emitted; re-run when "
                    "NCBI answers again (MRAG_GENERIF_FAIL_ON_TOP=0 emits them flagged)"
                    % (len(_gf_top), "; ".join("%s (%s)" % (g, str(((self._generif_fetch or {}).get("errors")
                                                                   or {}).get(g) or "")[:100])
                                               for g in _gf_top)))
            print("[curated] WARNING: emitting %d target(s) whose GeneRIF record could not be read, flagged "
                  "(MRAG_GENERIF_FAIL_ON_TOP=0): %s" % (len(_gf_top), ", ".join(_gf_top)), flush=True)

        # emit under branch dir
        branch_dir = self.params.get("branch_dir", str(out_dir))
        emitted = emit_outputs(top_final, branch_dir, motifs)
        # followup 2026-09-28 (fix 1): the Section-10 notice reads this beside curated_targets.json;
        # written only when a GeneRIF record could not be read, and never left over from a run before
        _gfb = Path(branch_dir) / "generif_fetch.json"
        _gfs = getattr(self, "_generif_fetch", None) or {}
        if _gfs.get("failed"):
            _gfrec = {"version": 1, "n": _gfs.get("n"), "failed": list(_gfs["failed"]),
                      "errors": dict(_gfs.get("errors") or {}),
                      "in_targets": [r.get("gene") for r in top_final if r.get("generif_fetch") == "failed"],
                      # review 2026-09-28: the failed genes that could have been targets (only with
                      # MRAG_GENERIF_FAIL_ON_TOP=0 is the run still here with any)
                      "could_emit": list(getattr(self, "_generif_cand", None) or [])}
            _gfb.write_text(json.dumps(_gfrec, indent=1))
            self.outputs_generif = _gfrec
        elif _gfb.exists():
            try:
                _gfb.unlink()
            except Exception:
                pass
        # followup r3 (post-deploy finding 1): the Section-10 notice reads this beside curated_targets.json;
        # written only when a rescue check failed, never left over from an earlier run
        _rfb = Path(branch_dir) / RESCUE_BRANCH_FILE
        _rfs = getattr(self, "_rescue_fail", None) or {}
        if _rfs:
            _rfrec = dict(_rfs)
            _rfrec["in_targets"] = [r.get("gene") for r in top_final if r.get("gene") in set(_rfs.get("failed") or [])]
            _rfb.write_text(json.dumps(_rfrec, indent=1))
            self.outputs_rescue = _rfrec
        elif _rfb.exists():
            try:
                _rfb.unlink()
            except Exception:
                pass
        self.outputs = {
            "panel_table": str(panel_table),
            "pool_motifs_json": str(motifs_path),
            "scored_json": str(scored_out),
            "curated_top_genes": [r["gene"] for r in top_final],
            "vdb_source": ("card_retriever" if _pre else "runner"),
            "card_retrieval": _cardstats,
            "pubmed_rescue": {k: _resc.get(k) for k in
                              ("enabled", "checked", "rescued", "rescued_genes",
                               "n_calls", "seconds", "domains", "error",
                               "n_errors", "n_gene_errors", "n_pairs")},
            "pubmed_rescued_in_top": [r["gene"] for r in top_final
                                      if r.get("evidence_source") == "pubmed"],
            # PI decision 2026-09-26: the frame the LLM score rubric used
            "score_domain_frame": dict(getattr(self, "_score_frame", None) or {}),
        }
        if _audit_sum is not None:
            # evaudit 2026-09-27: the audit's summary in the manifest step
            # (fix2: removed / flagged genes with reasons; the removed genes'
            # items stay in scored.json -> evidence_audit.removed[].items)
            self.outputs["evidence_audit"] = {k: v for k, v in _audit_sum.items() if k != "removed"}
            self.outputs["evidence_audit"]["removed"] = [
                {k: v for k, v in x.items() if k != "items"} for x in (_audit_sum.get("removed") or [])]
        if getattr(self, "outputs_generif", None):
            # followup 2026-09-28 (fix 1): the manifest names the genes whose GeneRIF record failed
            self.outputs["generif_fetch"] = dict(self.outputs_generif)
        if getattr(self, "outputs_rescue", None):
            # followup r3 (post-deploy finding 1): the manifest names the genes whose rescue check failed
            self.outputs["pubmed_rescue_failed"] = dict(self.outputs_rescue)
        if _rsc is not None and _rsc.get("n"):
            # fix2 2026-09-28: the re-scored genes in the manifest step
            self.outputs["rescored_after_filter"] = [
                {k: x.get(k) for k in ("gene", "old_score", "new_score", "new_reason", "error")}
                for x in _rsc["genes"]]
        self.outputs.update(emitted)

        # --- the ortholog route, written next to the targets it describes ---
        # The report reads this to print the disclosure banner and the per-gene
        # "searched as <HUMAN>" line. It is written for a human run too (with
        # identity=True and no disclosures) so a consumer never has to guess
        # whether the file is missing or the run was human.
        _o = dict(getattr(self, "_ortholog", None) or {"identity": True})
        _o["top_final"] = {
            r["gene"]: {"lit_symbol": r.get("lit_symbol", ""),
                        "lit_searched": r.get("lit_searched", True),
                        "disclosure": (_o.get("disclosures") or {}).get(r["gene"], "")}
            for r in top_final}
        _o_path = Path(self.params.get("branch_dir") or out_dir) / "ortholog.json"
        _o_path.parent.mkdir(parents=True, exist_ok=True)
        _o_path.write_text(json.dumps(_o, indent=2))
        self.outputs["ortholog_json"] = str(_o_path)
        self.outputs["ortholog_unmapped"] = list(_o.get("unmapped") or [])

        # 8. optional Section-10 graph + Section-11 enrichment emits
        top_genes = [r["gene"] for r in top_final]
        g2tf = None
        graph_dir = self.params.get("graph_dir")
        if graph_dir:
            counts, g2tf = emit_graph_v2(
                top_genes, str(panel_table), panel,
                self.params.get("integrated_csv"), graph_dir)
            print("[curated] section10 graph:", counts, flush=True)
            self.outputs["trn_cyjs_json"] = str(Path(graph_dir) / "trn_cyjs.json")
        enrich_dir = self.params.get("enrich_dir")
        if enrich_dir:
            if g2tf is None:
                g2tf = {g: sorted(motifs.get(g, {}).get("tfs", {}).keys()) for g in top_genes}
            ctx = {"cell_line": self.params.get("cell_line", ""),
                   "tissue": self.params.get("tissue", ""),
                   # the domain this run was scored against; the term filter judged relevance
                   # from a cell-line name alone until these two were threaded through
                   "cancer_type": self.params.get("cancer_type") or "",
                   "literature_domains": list(
                       self.params.get("literature_domains_user")
                       or self.params.get("vdb_cancer") or []),
                   "organism": self.params.get("organism", "human"),
                   "comparison": self.params.get("comparison", ""),
                   "perturbation": self.params.get("perturbation", ""),
                   "study_label": ""}
            en = emit_enrich(g2tf, enrich_dir, ctx,
                             max_tfs=int(self.params.get("max_tfs", 0)),
                             min_shared_tfs=int(self.params.get("min_shared_tfs", 2)),
                             top_terms_total=int(self.params.get("top_terms_total", 15)),
                             panel_tfs=self.params.get("enrich_tfs") or None)
            print("[curated] section11 enrich:", en, flush=True)
            self.outputs["enrichment"] = en

    def validate(self):
        gates = []
        ct = self.outputs.get("curated_targets_json", "")
        ok = bool(ct) and Path(ct).exists()
        n = len(self.outputs.get("curated_top_genes", []))
        gates.append({"name": "curated_targets_exists",
                      "result": "pass" if ok and n > 0 else "fail",
                      "value": n, "msg": "%d curated targets" % n})
        _gf = getattr(self, "outputs_generif", None)
        if _gf and _gf.get("failed"):
            # followup 2026-09-28 (fix 1): the manifest WARNING for genes whose GeneRIF record failed
            gates.append({"name": "generif_fetch", "result": "warn", "value": len(_gf["failed"]),
                          "msg": ("the NCBI GeneRIF record of %d/%s scored gene(s) could not be read (also in the "
                                  "second pass); they were scored without their GeneRIF lines: %s%s%s%s"
                                  % (len(_gf["failed"]), _gf.get("n") or "?", ", ".join(_gf["failed"][:12]),
                                     ("; emitted flagged: " + ", ".join(_gf["in_targets"]))
                                     if _gf.get("in_targets") else "",
                                     # review 2026-09-28: kept going although they could have been targets
                                     ("; could have been targets, not in the list (MRAG_GENERIF_FAIL_ON_TOP=0): "
                                      + ", ".join(g for g in _gf["could_emit"] if g not in (_gf.get("in_targets") or [])))
                                     if [g for g in (_gf.get("could_emit") or [])
                                         if g not in (_gf.get("in_targets") or [])] else "",
                                     ("; cannot be targets (panel TF / no driver TF): "
                                      + ", ".join(g for g in _gf["failed"] if g not in (_gf.get("could_emit") or [])))
                                     if "could_emit" in _gf and [g for g in _gf["failed"]
                                                                  if g not in (_gf.get("could_emit") or [])] else ""))})
        _rf = getattr(self, "outputs_rescue", None)
        if _rf and (_rf.get("failed") or _rf.get("partial") or _rf.get("n_errors")):
            # followup r3 (post-deploy finding 1): the manifest WARNING for failed live-PubMed rescue checks
            _rin = [g for g in (_rf.get("in_targets") or [])]
            _rmiss = [g for g in (_rf.get("could_emit") or []) if g not in _rin]
            _rcan = [g for g in (_rf.get("failed") or []) if g not in (_rf.get("could_emit") or []) and g not in _rin]
            gates.append({"name": "pubmed_rescue", "result": "warn", "value": len(_rf.get("failed") or []),
                          "msg": ("the live PubMed rescue check failed (also on a second try) for %d of %s capped "
                                  "candidate gene(s)%s%s%s%s%s"
                                  % (len(_rf.get("failed") or []), _rf.get("n_candidates") or "?",
                                     (": " + ", ".join((_rf.get("failed") or [])[:12])) if _rf.get("failed") else "",
                                     ("; in the list: " + ", ".join(_rin)) if _rin else "",
                                     ("; could have been targets, not in the list (MRAG_RESCUE_FAIL_ON_TOP=0): "
                                      + ", ".join(_rmiss)) if _rmiss else "",
                                     ("; cannot be targets (panel TF / no driver TF): " + ", ".join(_rcan))
                                     if _rcan else "",
                                     ("; admitted although the check of another domain failed: "
                                      + ", ".join(_rf.get("partial") or [])) if _rf.get("partial") else ""))})
        _o = getattr(self, "_ortholog", None)
        if _o and not _o.get("identity", True):
            gates.append({
                "name": "literature_ortholog_route",
                # not a failure: a gene with no human ortholog is a real
                # biological fact, and the report says so per gene. This gate
                # exists so the count is never invisible.
                "result": "pass" if _o.get("n_unmapped", 0) == 0 else "warn",
                "value": _o.get("n_unmapped", 0),
                "msg": ("%d/%d candidate genes mapped to a human ortholog (%s); "
                        "%d have none and were NOT searched: %s"
                        % (_o.get("n_mapped", 0), _o.get("n_total", 0),
                           _o.get("source", ""), _o.get("n_unmapped", 0),
                           ", ".join((_o.get("unmapped") or [])[:10]) or "-"))})
        return gates
