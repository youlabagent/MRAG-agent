"""How the nodes and the runner reach the analysis stages that became MCP tools in step 2a (2026-10-12).

Before step 2a these stages ran inside the orchestrator (node bodies, runner blocks, library calls); now each is a
tool of one of the ten MCP packages, registered on that package's stdio server.py too:

  atac-mcp       call_super_enhancers, call_super_enhancers_bigwig      SERoseNode, SERoseBigwigNode
                 giggle_tf_overlap                                      GiggleNode
                 split_condition_peaks                                  peakheatmap.build_specific_beds
  homer-mcp      scan_super_enhancer_motifs                             SEMotifNode
  viz-report-mcp plot_peak_heatmap                                      PeakHeatmapNode
                 plot_peak_heatmap_matrix                               runner Step 6e-2b (processed matrix)
                 build_se_trn                                           SETrnBuilderNode (DAStk display refill +
                                                                        produce_se_trn_inputs)
  trn-mcp        join_matrix_tfs_pval2                                  SETrnBuilderNode._matrix_join
                 build_panel_motif_table, build_target_pool,            CuratedTargetsNode: candidates,
                 scan_target_motifs, score_target_genes,                LLM scoring (+ evidence gate), re-score,
                 rescore_target_genes, select_targets,                  selection + evidence audit;
                 build_matrix_pval2                                     runner matrix Pval2
  paperqa-mcp    retrieve_target_literature, evidence_pubmed_rescue     CuratedTargetsNode: literature retrieval,
                                                                        the evidence gate's PubMed rescue
  rna-mcp        limma_group_de                                         runner: flex_adapters.run_rna_log2_adapter
  diffbind-mcp   voom_differential_accessibility                        runner: flex_adapters.run_atac_matrix_adapter

  tools(key[, impl])          {tool name: callable} of one register function, through capture_mcp_tools():
                              MCP proxies with MRAG_TOOL_TRANSPORT=mcp (default), the functions with =direct
  curated_bindings(ct)        the stage names CuratedTargetsNode.run() binds: =direct -> the curated_targets
                              module's own functions (the node exactly as before step 2a); =mcp -> adapters that
                              call the trn-mcp / paperqa-mcp tools with the same arguments and give back the same
                              results, updating the caller's rows / scored document IN PLACE (write_back_*)
  matrix_stage(name)          the runner's processed-matrix stages, the same way

Rows and documents travel as JSON.  write_back_dict / write_back_rows keep the in-place semantics of the stage
functions: the caller's list and row dicts stay the same objects, a value the tool did not change keeps its
identity, the key order follows the tool's result (= what the function left in direct mode)."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys

from . import _base
from ..tool_transport import McpArgumentError, McpTransportError, _same, transport_name

# the packages' src trees (the nodes import every package by its hard-coded path)
SRC = {
    "atac_mcp": "/BIO1/MRAG/Agent/mcps/ATAC_mcp/src",
    "rna_mcp": "/BIO1/MRAG/Agent/mcps/RNA_mcp/src",
    "diffbind_mcp": "/BIO1/MRAG/Agent/mcps/diffbind_mcp/src",
    "homer_mcp": "/BIO1/MRAG/Agent/mcps/homer_mcp/src",
    "trn_mcp": "/BIO1/MRAG/Agent/mcps/trn_mcp/src",
    "paperqa_mcp": "/BIO1/MRAG/Agent/mcps/paperqa_mcp/src",
    "viz_report_mcp": "/BIO1/MRAG/Agent/mcps/viz_report_mcp/src",
}
# key -> (package, module, register function)
REGISTERS = {
    "super_enhancers": ("atac_mcp", "atac_mcp.tools.super_enhancers", "register_super_enhancer_tools"),
    "giggle": ("atac_mcp", "atac_mcp.tools.giggle", "register_giggle_tools"),
    "peak_sets": ("atac_mcp", "atac_mcp.tools.peak_sets", "register_peak_set_tools"),
    "se_motif": ("homer_mcp", "homer_mcp.tools.se_motif", "register_se_motif_tools"),
    "peak_heatmap": ("viz_report_mcp", "viz_report_mcp.tools.peak_heatmap", "register_peak_heatmap_tools"),
    "se_trn": ("viz_report_mcp", "viz_report_mcp.tools.se_trn", "register_se_trn_tools"),
    "matrix_join": ("trn_mcp", "trn_mcp.tools.matrix_join", "register_matrix_join_tools"),
    "targets": ("trn_mcp", "trn_mcp.tools.targets", "register_target_tools"),
    "evidence": ("paperqa_mcp", "paperqa_mcp.tools.evidence", "register_evidence_tools"),
    "matrix_de": ("rna_mcp", "rna_mcp.tools.matrix_de", "register_matrix_de_tools"),
    "matrix_da": ("diffbind_mcp", "diffbind_mcp.tools.matrix_da", "register_matrix_da_tools"),
}
# the node params select_targets reads (the selection tool gets only these)
SELECT_PARAM_KEYS = ("evidence_audit", "evidence_audit_mode", "evidence_audit_limit", "evidence_audit_keep",
                     "endpoint", "model")


def register_function(key):
    pkg, mod, fn = REGISTERS[key]
    p = SRC[pkg]
    if p not in sys.path:
        sys.path.insert(0, p)
    return getattr(importlib.import_module(mod), fn)


def tools(key, impl=None):
    """{tool name: callable} of register function `key` (capture_mcp_tools; impl -> register_fn(mcp, impl=impl))."""
    reg = register_function(key)
    if impl is None:
        return _base.capture_mcp_tools(reg)
    return _base.capture_mcp_tools(lambda mcp: reg(mcp, impl=impl))


def _p(x):
    """a path argument as text (the tools take str paths)"""
    return os.fspath(x) if isinstance(x, os.PathLike) else x


# --------------------------------------------------------------------------------------------- write-back
_MISSING = object()


def write_back_rows(dst, src):
    """dst (the caller's list of rows) takes src's content in place: same list, same row dicts."""
    if not isinstance(dst, list) or not isinstance(src, list):
        raise McpTransportError("write-back: rows are not lists (%s, %s)" % (type(dst).__name__, type(src).__name__))
    if len(dst) != len(src):
        raise McpTransportError("write-back: the tool returned %d rows for %d" % (len(src), len(dst)))
    for d, s in zip(dst, src):
        if _same(d, s):
            continue
        if isinstance(d, dict) and isinstance(s, dict):
            write_back_dict(d, s)
        else:
            raise McpTransportError("write-back: a row is not a dict (%s, %s)" % (type(d).__name__, type(s).__name__))


def write_back_dict(dst, src, inplace=()):
    """dst (the caller's dict) takes src's content in place; keys in `inplace` holding row lists are written back
    row by row (write_back_rows); a value equal to the old one (same types, same order) keeps the old object."""
    if not isinstance(dst, dict) or not isinstance(src, dict):
        raise McpTransportError("write-back: not a dict (%s, %s)" % (type(dst).__name__, type(src).__name__))
    if _same(dst, src):
        return
    old = dict(dst)
    dst.clear()
    for k, v in src.items():
        ov = old.get(k, _MISSING)
        if k in inplace and isinstance(ov, list) and isinstance(v, list):
            write_back_rows(ov, v)
            dst[k] = ov
        elif ov is not _MISSING and _same(ov, v):
            dst[k] = ov
        else:
            dst[k] = v


# --------------------------------------------------------------------------------------------- the curated node
class EvidenceGateStage:
    """CuratedTargetsNode's view of evidence_gate in mcp mode: pubmed_rescue is the paperqa-mcp tool
    evidence_pubmed_rescue (adapter); every other attribute is the module's own."""

    def __init__(self, module, pubmed_rescue):
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "pubmed_rescue", pubmed_rescue)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_module"), name)

    def __repr__(self):
        return "<evidence_gate via MCP (pubmed_rescue: paperqa-mcp evidence_pubmed_rescue) %r>" % (
            object.__getattribute__(self, "_module"),)


def curated_bindings(ct):
    """(build_panel_table, build_pool, scan_motifs, prefetch_card_hits, evidence_gate, rescore_after_filter,
    select_targets, score_via_mcp) for CuratedTargetsNode.run(); ct = the curated_targets module.  direct: ct's
    own objects and score_via_mcp None (the node runs the score runner itself, as before)."""
    if transport_name() == "direct":
        return (ct.build_panel_table, ct.build_pool, ct.scan_motifs, ct.prefetch_card_hits,
                ct._evgate, ct.rescore_after_filter, ct.select_targets, None)
    T = tools("targets", impl=ct)
    P = tools("evidence", impl=ct)

    def build_panel_table(motif_tsv, pval2_txt, panel, out_path):
        d = T["build_panel_motif_table"](motif_tsv=_p(motif_tsv), pval2_txt=_p(pval2_txt), panel=list(panel),
                                         out_path=_p(out_path))
        return d["out_path"], d["found"]

    def build_pool(panel_table, panel, lfc_min=2.0, min_union=3, pool_n=200):
        return T["build_target_pool"](panel_table=_p(panel_table), panel=list(panel), lfc_min=lfc_min,
                                      min_union=min_union, pool_n=pool_n)

    def scan_motifs(pool, panel_table, panel, fasta_path=None):
        return T["scan_target_motifs"](pool=pool, panel_table=_p(panel_table), panel=list(panel),
                                       fasta_path=_p(fasta_path))

    def prefetch_card_hits(genes, dom_specs, corpora, suffix, top_k=None, device="cpu", gene_info=None,
                           lit_symbol=None, lit_unmapped=(), timeout=None):
        d = P["retrieve_target_literature"](
            genes=list(genes), dom_specs=dom_specs, corpora=list(corpora or []), suffix=suffix, top_k=top_k,
            device=device, gene_info=_p(gene_info), lit_symbol=lit_symbol,
            lit_unmapped=(list(lit_unmapped) if lit_unmapped is not None else None), timeout=timeout)
        return d["vdb"], d["stats"]

    def pubmed_rescue(rows, domains, cap=None, enabled=None, retmax=None, gene_info=None, lit_symbol=None,
                      narrative_py=None, progress_every=25):
        d = P["evidence_pubmed_rescue"](rows=rows, domains=domains, cap=cap, enabled=enabled, retmax=retmax,
                                        gene_info=_p(gene_info), lit_symbol=lit_symbol,
                                        narrative_py=_p(narrative_py), progress_every=progress_every)
        write_back_rows(rows, d["rows"])
        return d["summary"]

    def rescore_after_filter(rows, scored_doc, dspecs, sys_prompt, endpoint, model, rif_chars=600,
                             lit_symbol=None, llm=None, fetch=None, log=None, retries=2, batch_ctx=None):
        if fetch is not None or log is not None:
            raise McpArgumentError("rescore_after_filter: a fetch / log callable cannot be sent over MCP "
                                   "(run with MRAG_TOOL_TRANSPORT=direct)")
        effort = None
        if llm is not None:
            if not hasattr(llm, "reasoning_effort"):
                raise McpArgumentError("rescore_after_filter: an llm callable cannot be sent over MCP; pass "
                                       "curated_targets._RescoreLLM (the scorer's reasoning_effort)")
            effort = llm.reasoning_effort
        same = isinstance(scored_doc, dict) and rows is scored_doc.get("rows")
        d = T["rescore_target_genes"](scored_doc=scored_doc, dspecs=dspecs, sys_prompt=sys_prompt,
                                      endpoint=endpoint, model=model, rif_chars=rif_chars, lit_symbol=lit_symbol,
                                      retries=retries, batch_ctx=batch_ctx, reasoning_effort=effort,
                                      rows=(None if same else rows))
        write_back_dict(scored_doc, d["scored_doc"], inplace=("rows",))
        if not same:
            write_back_rows(rows, d["rows"])
        return d["summary"]

    def select_targets(scored, panel, n_top, min_score, gate_on, gscope, gdomains, scored_doc, params=None,
                       out_dir=None, llm=None, log=None, audit_endpoint=None, audit_model=None):
        if llm is not None or log is not None:
            raise McpArgumentError("select_targets: an llm / log callable cannot be sent over MCP "
                                   "(run with MRAG_TOOL_TRANSPORT=direct)")
        same = isinstance(scored_doc, dict) and scored is scored_doc.get("rows")
        pk = None if params is None else {k: params[k] for k in SELECT_PARAM_KEYS if k in params}
        d = T["select_targets"](scored_doc=scored_doc, panel=list(panel), n_top=n_top, min_score=min_score,
                                gate_on=gate_on, gscope=gscope, gdomains=gdomains, params=pk, out_dir=_p(out_dir),
                                audit_endpoint=audit_endpoint, audit_model=audit_model,
                                rows=(None if same else scored))
        write_back_dict(scored_doc, d["scored_doc"], inplace=("rows",))
        if not same:
            write_back_rows(scored, d["rows"])
        return [scored[i] for i in d["top_index"]], d["summary"]

    def score_via_mcp(python, runner, params, timeout):
        d = T["score_target_genes"](params_json=_p(params), python=_p(python), runner_py=_p(runner),
                                    timeout_s=timeout)
        return subprocess.CompletedProcess([_p(python), _p(runner), _p(params)], d["returncode"], d["stdout"],
                                           d["stderr"])

    return (build_panel_table, build_pool, scan_motifs, prefetch_card_hits,
            EvidenceGateStage(ct._evgate, pubmed_rescue), rescore_after_filter, select_targets, score_via_mcp)


# --------------------------------------------------------------------------------------------- runner stages
def matrix_stage(name):
    """The runner's processed-matrix stage `name` (run_atac_matrix_adapter, run_rna_log2_adapter,
    build_pval2_from_matrix): the function itself with MRAG_TOOL_TRANSPORT=direct, else a callable with its
    signature and return value that calls the MCP tool."""
    from ..adapters import flex_adapters as FA
    from . import curated_targets as CT
    if transport_name() == "direct":
        return {"run_atac_matrix_adapter": FA.run_atac_matrix_adapter,
                "run_rna_log2_adapter": FA.run_rna_log2_adapter,
                "build_pval2_from_matrix": CT.build_pval2_from_matrix}[name]
    if name == "run_atac_matrix_adapter":
        t = tools("matrix_da", impl=FA)["voom_differential_accessibility"]

        def run_atac_matrix_adapter(matrix, group_tsv, out_dir, treat, ref, floor_n=5000):
            d = t(matrix=_p(matrix), group_tsv=_p(group_tsv), out_dir=_p(out_dir), treat=treat, ref=ref,
                  floor_n=floor_n)
            return d["treatment_bed"], d["control_bed"]
        return run_atac_matrix_adapter
    if name == "run_rna_log2_adapter":
        t = tools("matrix_de", impl=FA)["limma_group_de"]

        def run_rna_log2_adapter(matrix, group_tsv, out_csv, treat, ref):
            return t(matrix=_p(matrix), group_tsv=_p(group_tsv), out_csv=_p(out_csv), treat=treat, ref=ref)
        return run_rna_log2_adapter
    if name == "build_pval2_from_matrix":
        t = tools("targets", impl=CT)["build_matrix_pval2"]

        def build_pval2_from_matrix(canonical_motif_tsv, canonical_peaks_bed, diffbind_csv, deseq_csv, out_path):
            d = t(canonical_motif_tsv=_p(canonical_motif_tsv), canonical_peaks_bed=_p(canonical_peaks_bed),
                  diffbind_csv=_p(diffbind_csv), deseq_csv=_p(deseq_csv), out_path=_p(out_path))
            return d["out_path"], d["n"]
        return build_pval2_from_matrix
    raise KeyError(name)
