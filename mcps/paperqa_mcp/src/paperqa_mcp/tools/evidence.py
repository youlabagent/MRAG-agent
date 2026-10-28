"""paperqa-mcp: the literature-evidence stages of the curated target prioritisation as MCP tools (MCP step 2a,
2026-10-12).

The implementation is the orchestrator's own code -- mrag_orchestrator/nodes/curated_targets.py and
evidence_gate.py -- imported when a tool is called (the MCP interface of that code, not a copy of it).

  register_evidence_tools(mcp, impl=None)   impl = the curated_targets module (default: import
                                            mrag_orchestrator.nodes.curated_targets); its _evgate is evidence_gate
Tools:
  retrieve_target_literature   curated_targets.prefetch_card_hits: the scorer's literature through the Section-14
                               card retriever (narrative.batch_retrieve_card_hits: one corpus-filtered,
                               gene-name-prefiltered PaperQA query per gene and literature domain, plus the run's
                               other corpora), each hit tied to the domains (evidence_gate.tie_record)
  evidence_pubmed_rescue       evidence_gate.pubmed_rescue: the evidence gate's live-PubMed rescue -- a gene the
                               gate CAPPED although its own LLM score was above the cap is searched in PubMed per
                               domain; records are read back and must name the gene in the domain (and pass the
                               evidence rules 1a-1c) before the LLM score is restored (evidence_source "pubmed")."""
from __future__ import annotations

import importlib
from typing import Any, Optional, Union

CURATED_MODULE = "mrag_orchestrator.nodes.curated_targets"


def _plain(x):
    """Plain JSON values: tuples -> lists, numpy scalars -> Python numbers; a non-text dict key is an error."""
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


def register_evidence_tools(mcp, impl=None):
    def _ct():
        return impl if impl is not None else importlib.import_module(CURATED_MODULE)

    @mcp.tool()
    def retrieve_target_literature(genes: list[str], dom_specs: list[dict[str, Any]], corpora: list[str],
                                   suffix: str, top_k: Optional[Union[int, str]] = None, device: str = "cpu",
                                   gene_info: Optional[str] = None, lit_symbol: Optional[dict[str, str]] = None,
                                   lit_unmapped: Optional[list[str]] = None,
                                   timeout: Optional[Union[int, float]] = None) -> dict[str, Any]:
        """PaperQA literature of the candidate genes through the Section-14 card retriever.

        Args:
            genes: the run's gene symbols.  dom_specs: literature.domain_specs() of the run's domains (the query
                label per domain).  corpora: the run's vdb_cancer list -- every key that is not a domain becomes an
                extra corpus slot queried with `suffix` ("{gene} <suffix>").
            top_k: hits per corpus slot (default MRAG_CURATED_CARD_TOP_K / 3).  device: the embedder's device.
            gene_info: NCBI gene_info for the official full names.  lit_symbol: run symbol -> human ortholog
                searched instead; lit_unmapped: genes with no human ortholog (not searched).  timeout: for the
                retrieval subprocess.

        Returns {vdb, stats}: vdb = {gene: [hit record ...]} (null when the batch failed: the score runner then
        retrieves itself), stats = the retrieval record (slots, queries, errors, fallbacks, wall time)."""
        vdb, stats = _ct().prefetch_card_hits(genes, dom_specs, corpora, suffix, top_k=top_k, device=device,
                                              gene_info=gene_info, lit_symbol=lit_symbol,
                                              lit_unmapped=(lit_unmapped if lit_unmapped is not None else ()),
                                              timeout=timeout)
        return _plain({"vdb": vdb, "stats": stats})

    @mcp.tool()
    def evidence_pubmed_rescue(rows: list[dict[str, Any]], domains: list[dict[str, Any]],
                               cap: Optional[float] = None, enabled: Optional[Union[bool, int, str]] = None,
                               retmax: Optional[Union[int, str]] = None, gene_info: Optional[str] = None,
                               lit_symbol: Optional[dict[str, str]] = None, narrative_py: Optional[str] = None,
                               progress_every: int = 25) -> dict[str, Any]:
        """Live-PubMed rescue of the rows the evidence gate capped.

        A row is checked iff evidence_gate == "fired" and its own LLM score was above the cap (default 2): per
        literature domain with a PubMed query fragment, PubMed is searched (retmax records, default 20; 5 with
        MRAG_EVIDENCE_AUDIT=0), every record is fetched back and must name the gene in the domain (and pass the
        evidence filters 1a-1c) before it counts; a verified gene gets evidence_gate "pubmed", its LLM score back
        and the PMIDs.  A failed check is recorded as failed, never as an absence (a second try after
        MRAG_RESCUE_RETRY_PAUSE_S).  enabled: param / MRAG_TARGET_EVIDENCE_PUBMED.

        Returns {summary, rows}: the rescue summary (checked, rescued, per_gene, errors ...) and the updated
        rows."""
        summary = _ct()._evgate.pubmed_rescue(rows, domains, cap=cap, enabled=enabled, retmax=retmax,
                                              gene_info=gene_info, lit_symbol=lit_symbol,
                                              narrative_py=narrative_py, progress_every=progress_every)
        return _plain({"summary": summary, "rows": rows})
