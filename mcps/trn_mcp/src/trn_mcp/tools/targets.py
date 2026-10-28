"""trn-mcp: the curated target-prioritisation stages as MCP tools (MCP step 2a, 2026-10-12).

Each tool runs one analysis stage of the orchestrator's CuratedTargetsNode (report Sections 10, 11, 14).  The
implementation is the orchestrator's own code -- mrag_orchestrator/nodes/curated_targets.py, evidence_gate.py,
evidence_audit.py -- imported when a tool is called: these tools are the MCP interface of that code, not a second
copy of it (no statistic, threshold, prompt or output differs from a direct call).

  register_target_tools(mcp, impl=None)   impl = the curated_targets module whose functions the tools run
                                          (default: import mrag_orchestrator.nodes.curated_targets; the node
                                          passes its own module, so a test's stand-ins are what runs)
Tools (the curated_targets function each runs):
  build_panel_motif_table   build_panel_table      panel-TF motif columns joined to the promoter Pval2 rows
  build_target_pool         build_pool             candidate genes (log2FC > lfc_min, >= min_union panel TFs)
  scan_target_motifs        scan_motifs            per gene x panel TF best motif instance in the rep. peak
  score_target_genes        the score runner       LLM evidence scoring 0-10 (GeneRIF + PaperQA evidence, batches
                                                   of 15, temperature 0, the scorer's reasoning_effort), then the
                                                   evidence-grounding gate (cap 2; rules 1a/1b/1c drop evidence
                                                   that does not count) -> scored.json
  rescore_target_genes      rescore_after_filter   re-score, inside its original batch, a gene an evidence filter
                                                   took counted evidence from and that was re-admitted
  select_targets            select_targets         evidence floor (min_score; 3 under a strict gate scope) ->
                                                   rerank (score, log2FC) minus the panel / no-driver genes ->
                                                   post-scoring evidence audit (default mode: flag only)
  build_matrix_pval2        build_pval2_from_matrix  processed-matrix input: the Pval2 table from the canonical
                                                   motif_tsv + DiffBind + DESeq2
Rows and the scored document travel as JSON; a tool that updates rows returns them (with the scored document),
and the orchestrator writes them back into its own objects (nodes/_stage_tools.py)."""
from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from typing import Any, Optional, Union

CURATED_MODULE = "mrag_orchestrator.nodes.curated_targets"
SCORE_PYTHON = "/BIO1/MRAG/venvs/paper_qa/bin/python3"


def _plain(x):
    """The tool's result as plain JSON values: tuples -> lists, numpy scalars -> Python numbers (json.dumps writes
    them the same way); a non-text dict key is an error."""
    t = type(x)
    if x is None or t in (str, bool, int, float):
        return x
    if t is list or t is tuple:
        return [_plain(v) for v in x]
    if t is dict:
        out = {}
        for k, v in x.items():
            if type(k) is not str:
                raise TypeError("result dict key %r is not text" % (k,))
            out[k] = _plain(v)
        return out
    mod = getattr(t, "__module__", "")
    if mod == "numpy":
        return x.item()
    if isinstance(x, bool):
        return bool(x)
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return float(x)
    if isinstance(x, str):
        return str(x)
    raise TypeError("result value of type %s.%s is not JSON" % (mod, t.__qualname__))


def register_target_tools(mcp, impl=None):
    def _ct():
        return impl if impl is not None else importlib.import_module(CURATED_MODULE)

    @mcp.tool()
    def build_panel_motif_table(motif_tsv: str, pval2_txt: str, panel: list[str], out_path: str) -> dict[str, Any]:
        """Panel-N motif table: stream the min-max .motif.tsv, keep the motif columns of the panel TFs (alias-aware
        token match), join them to the promoter-filtered Pval2 rows (Chr Start End Strand PeakID 'Gene Name'
        <TF(base)/motif ...> Fold(DiffBind) log2FC) and write out_path.

        Returns {out_path, found}: the path written and the panel TFs that have a motif column."""
        res = _ct().build_panel_table(motif_tsv, pval2_txt, panel, out_path)
        return _plain({"out_path": res[0], "found": res[1]})

    @mcp.tool()
    def build_target_pool(panel_table: str, panel: list[str], lfc_min: float = 2.0, min_union: int = 3,
                          pool_n: int = 200) -> list[dict[str, Any]]:
        """Candidate target genes: RNA log2FC > lfc_min and the union of panel TFs bound across the gene's peaks
        >= min_union; representative peak = max(# panel TFs bound, |Fold|); ranked by log2FC desc, top pool_n.

        Returns [{gene, log2FC, n_tf, tfs, rep_tfs, chrom, start, end, atac_fold}, ...]."""
        return _plain(_ct().build_pool(panel_table, panel, lfc_min=lfc_min, min_union=min_union, pool_n=pool_n))

    @mcp.tool()
    def scan_target_motifs(pool: list[dict[str, Any]], panel_table: str, panel: list[str],
                           fasta_path: Optional[str] = None) -> dict[str, Any]:
        """Per pool gene's representative peak and per panel TF the best motif instance (score desc), its genome
        position (HOMER offsets oriented by the peak strand) and reference span (from fasta_path when given).

        Returns {gene: {chrom, peak_start, peak_end, atac_fold, log2FC, tfs: {TF: {gm_start, ref_span, ...}}}}."""
        return _plain(_ct().scan_motifs(pool, panel_table, panel, fasta_path=fasta_path))

    @mcp.tool()
    def score_target_genes(params_json: str, python: str = SCORE_PYTHON, runner_py: str = "",
                           timeout_s: Union[int, float] = 10800) -> dict[str, Any]:
        """LLM evidence scoring of the candidate genes (pool genes with a bound panel-TF motif), then the
        evidence-grounding gate.

        params_json is the score parameter file (_score_params.json) the curated node writes: genes, motifs_path
        (pool_motifs.json: each gene's log2FC and bound panel TFs), scored_out, sys_prompt (the rubric), model,
        endpoint, reasoning_effort, the evidence gate's settings
        (evidence_gate, evidence_gate_cap, evidence_gate_scope, evidence_gate_domains, gate_module), the
        literature retrieval (prefetched_vdb or the runner's own query: vdb_query_suffix, cancer_list, device),
        GeneRIF settings, the human-ortholog route (lit_symbol, lit_unmapped, ortholog).  The scorer fetches each
        gene's GeneRIFs (NCBI), shows the model each gene's evidence block in batches of 15 at temperature 0, and
        caps a gene whose score no retrieved evidence grounds at the gate's cap (2); with the 2026-09-28 evidence
        layers on, rules 1a (paper-level NET test, nec), 1b (sentence-level NET veto) and 1c (screened-negative
        list) remove evidence that does not count first.

        runner_py: the score runner script (the curated node writes it); "" = write curated_targets._SCORE_RUNNER
        next to params_json.  python: the interpreter (the paper_qa venv).  timeout_s: for the whole run.

        Security: `python` and `runner_py` are executed as given (an argument list, no shell), so whoever can
        call this tool can run any program on the server with the server's permissions.  Serve it only to
        trusted clients; the orchestrator passes its own interpreter and the runner it wrote itself.

        Returns {returncode, stdout, stderr, scored_json, runner_py}; scored.json and its sidecars
        (_score_prompt.json, _score_batches.json, _generif_fetch.json) are written by the runner.  A timeout
        raises subprocess.TimeoutExpired."""
        runner = runner_py
        if not runner:
            runner = str(Path(params_json).parent / "_score_runner.py")
            Path(runner).write_text(_ct()._SCORE_RUNNER)
        r = subprocess.run([python, runner, params_json], capture_output=True, text=True, timeout=timeout_s)
        try:
            scored = str(json.loads(Path(params_json).read_text()).get("scored_out") or "")
        except Exception:
            scored = ""
        return {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr, "scored_json": scored,
                "runner_py": runner}

    @mcp.tool()
    def rescore_target_genes(scored_doc: dict[str, Any], dspecs: list[dict[str, Any]], sys_prompt: str,
                             endpoint: str, model: str, rif_chars: int = 600,
                             lit_symbol: Optional[dict[str, str]] = None, retries: int = 2,
                             batch_ctx: Optional[dict[str, Any]] = None, reasoning_effort: Optional[str] = None,
                             rows: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
        """Re-score after an evidence filter: a gene whose counted evidence the filters removed and that was
        re-admitted on other evidence is scored again INSIDE ITS ORIGINAL BATCH (batch_ctx = _score_batches.json:
        the same neighbours with the blocks they were shown, its own block minus the removed items plus the
        verified rescue records), with the scorer's model / endpoint / reasoning_effort; only its own answer is
        taken.  No batch context or a failed call: the old score is kept and the row is flagged.

        scored_doc: scored.json's content; its "rows" are the rows re-scored (or `rows`, when they are another
        list).  dspecs: literature.domain_specs() of the run.  rif_chars: GeneRIF display cut.

        Returns {summary, scored_doc[, rows]}: the re-score summary and the updated rows (in scored_doc)."""
        ct = _ct()
        rows_ = scored_doc.get("rows") if rows is None else rows
        llm = ((lambda _e, _m, _s, _u: ct._rescore_call(_e, _m, _s, _u, reasoning_effort=reasoning_effort))
               if reasoning_effort else None)
        summary = ct.rescore_after_filter(rows_, scored_doc, dspecs, sys_prompt, endpoint, model,
                                          rif_chars=rif_chars, lit_symbol=lit_symbol, llm=llm, retries=retries,
                                          batch_ctx=batch_ctx)
        out = {"summary": summary, "scored_doc": scored_doc}
        if rows is not None:
            out["rows"] = rows_
        return _plain(out)

    @mcp.tool()
    def select_targets(scored_doc: dict[str, Any], panel: list[str], n_top: int, min_score: Optional[float],
                       gate_on: bool, gscope: Optional[str], gdomains: list[dict[str, Any]],
                       params: Optional[dict[str, Any]] = None, out_dir: Optional[str] = None,
                       audit_endpoint: Optional[str] = None, audit_model: Optional[str] = None,
                       rows: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
        """Target selection: rank the scored genes by (score desc, log2FC desc), drop the panel TFs and genes
        without a driver TF, keep those at or above min_score (the evidence floor; the node raises it to cap + 1
        = 3 under the "literature" / "domain" gate scope), cap at n_top -- then, with the domain-scoped gate,
        the post-scoring evidence audit walks the candidates in emission order (the scorer's model at
        audit_endpoint / audit_model; mode from params["evidence_audit_mode"], default "flag": doubts are flagged,
        nothing is removed; cache <out_dir>/_evidence_audit_cache.json).

        scored_doc: scored.json's content; its "rows" are the scored rows (or `rows`).  params: the node params
        the selection reads (evidence_audit, evidence_audit_mode, evidence_audit_limit, evidence_audit_keep,
        endpoint, model).

        Returns {top_index, summary, scored_doc[, rows]}: the emitted targets as indices into the rows, the audit
        summary (None when the audit is off), and the rows with the audit fields."""
        ct = _ct()
        rows_ = scored_doc.get("rows") if rows is None else rows
        top, summ = ct.select_targets(rows_, panel, n_top, min_score, gate_on, gscope, gdomains, scored_doc,
                                      params, (Path(out_dir) if out_dir is not None else None),
                                      audit_endpoint=audit_endpoint, audit_model=audit_model)
        pos = {id(r): i for i, r in enumerate(rows_)}
        idx = []
        for r in top:
            if id(r) not in pos:
                raise RuntimeError("select_targets emitted a row that is not one of the scored rows (%r)"
                                   % (r.get("gene") if isinstance(r, dict) else r,))
            idx.append(pos[id(r)])
        out = {"top_index": idx, "summary": summ, "scored_doc": scored_doc}
        if rows is not None:
            out["rows"] = rows_
        return _plain(out)

    @mcp.tool()
    def build_matrix_pval2(canonical_motif_tsv: str, canonical_peaks_bed: str, diffbind_csv: str, deseq_csv: str,
                           out_path: str) -> dict[str, Any]:
        """Processed-matrix equivalent of the FASTQ-mode Pval2 table (PeakID Chr Start End 'Gene Name'
        Fold(DiffBind) log2FC): promoter rows of the canonical motif_tsv, Fold per peak via the canonical BED's
        name <-> coordinates, log2FC per gene from the DESeq2 table.

        Returns {out_path, n}: the table written and its number of rows."""
        res = _ct().build_pval2_from_matrix(canonical_motif_tsv, canonical_peaks_bed, diffbind_csv, deseq_csv,
                                            out_path)
        return _plain({"out_path": res[0], "n": res[1]})
