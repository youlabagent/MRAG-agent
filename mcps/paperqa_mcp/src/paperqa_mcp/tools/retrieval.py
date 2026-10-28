"""Retrieval tools — dense search over Qdrant + Top-N TF batch retrieval.

Qdrant local-mode locks the storage directory, so retrieval and admin tools
MUST share a single client. We use ``Retriever.__new__`` to instantiate the
Retriever object without running its ``__init__`` (which would open its own
QdrantClient and SentenceTransformer), then inject the singletons that
``admin._get_qclient`` / ``admin._get_embedder`` manage.

``year_min`` filtering is performed in this layer (not in upstream Retriever)
because the parsed corpus stores ``year`` as a string ("2026"), which makes
the upstream year filter raise a ``TypeError`` on ``str < int``.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Union

from mcp.server.fastmcp import FastMCP

# mrag_paperqa is on sys.path (injected in server.py).
from mrag_paperqa.retriever import Retriever, Passage  # type: ignore

COLLECTION = "mrag_papers"

# When the caller asks for year_min filtering we over-fetch from Qdrant
# (since the upstream filter is skipped) and post-filter here.
_YEAR_OVERFETCH = 5


# ----- Singleton retriever (shares Qdrant + embedder with admin) ----------
_RETRIEVER: Optional[Retriever] = None


def _get_retriever(device: Optional[str] = None) -> Retriever:
    """Lazy init — reuses admin's shared QdrantClient + SentenceTransformer."""
    global _RETRIEVER
    if _RETRIEVER is None:
        from paperqa_mcp.tools.admin import _get_qclient, _get_embedder
        retr = Retriever.__new__(Retriever)  # skip __init__ -> no second lock
        retr.client     = _get_qclient()
        retr.collection = COLLECTION
        retr.model      = _get_embedder(device)
        _RETRIEVER = retr
    return _RETRIEVER


def _coerce_year(y) -> Optional[int]:
    """Best-effort cast of payload year (often a string) to int."""
    if y is None:
        return None
    try:
        return int(str(y).strip())
    except Exception:
        return None


def _search(
    retr: Retriever,
    query: str,
    top_k: int,
    cancer,
    year_min: Optional[int],
    source: Optional[str],
) -> List[Passage]:
    """Wrap Retriever.search with safe year_min handling."""
    if year_min is None:
        return retr.search(query=query, top_k=top_k, cancer=cancer,
                           year_min=None, source=source)
    # Over-fetch then post-filter (str vs int year mismatch in payload)
    raw = retr.search(query=query, top_k=top_k * _YEAR_OVERFETCH, cancer=cancer,
                      year_min=None, source=source)
    out: List[Passage] = []
    for p in raw:
        y = _coerce_year(p.year)
        if y is not None and y >= year_min:
            out.append(p)
            if len(out) >= top_k:
                break
    return out


# ----- Tool registration --------------------------------------------------
def register_retrieval_tools(mcp: FastMCP):

    @mcp.tool()
    def fetch_literature(
        query: str,
        top_k: int = 8,
        cancer: Optional[Union[str, List[str]]] = None,
        year_min: Optional[int] = None,
        source: Optional[str] = None,
    ) -> str:
        """Dense-retrieve top-K passages from the paper-qa corpus.

        Uses BGE-M3 embedding + Qdrant cosine search over ~55k chunks
        (prostate / sclc / breast). Filters: cancer/source at Qdrant level,
        year_min post-filtered here.

        Args:
            query:    Natural-language query.
            top_k:    Number of passages to return (default 8).
            cancer:   "prostate" | "sclc" | "breast" or a list.
            year_min: Drop passages from papers older than this year.
            source:   E.g. "pubmed_abstract", "pmc_oa_xml",
                      "biorxiv_pdf", "landmark_2010_2023".

        Returns JSON with query/filters/n_hits/passages. Each passage:
        score, pmid, doc_id, title, cancer, year, source, section, text.
        """
        retr = _get_retriever()
        hits = _search(retr, query, top_k, cancer, year_min, source)
        return json.dumps(
            {
                "query":    query,
                "top_k":    top_k,
                "filters":  {"cancer": cancer, "year_min": year_min, "source": source},
                "n_hits":   len(hits),
                "passages": [p.to_dict() for p in hits],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    @mcp.tool()
    def fetch_literature_for_tfs(
        tfs: List[str],
        cancer: Optional[Union[str, List[str]]] = None,
        top_k_per_tf: int = 8,
        year_min: Optional[int] = None,
        source: Optional[str] = None,
        query_template: str = "{tf} transcription factor role mechanism target gene regulation in cancer",
    ) -> str:
        """Batch retrieval for a list of TF gene symbols (e.g. Top-N union).

        For each TF in ``tfs``, runs an independent dense query using
        ``query_template.format(tf=...)`` and returns per-TF passages.

        Designed to consume ``combined_mcp.pick_topn`` output --
        ``topN_union.json`` (``{"tfs": [...]}``).

        Args:
            tfs:            TF gene symbols.
            cancer:         Same as fetch_literature.
            top_k_per_tf:   Passages per TF (default 8).
            year_min:       Same as fetch_literature.
            source:         Same as fetch_literature.
            query_template: Python format string with a ``{tf}`` slot.
        """
        retr = _get_retriever()
        results = {}
        total = 0
        for tf in tfs:
            query = query_template.format(tf=tf)
            hits = _search(retr, query, top_k_per_tf, cancer, year_min, source)
            results[tf] = {
                "query":    query,
                "n":        len(hits),
                "passages": [p.to_dict() for p in hits],
            }
            total += len(hits)
        return json.dumps(
            {
                "n_tfs":            len(tfs),
                "top_k_per_tf":     top_k_per_tf,
                "filters":          {"cancer": cancer, "year_min": year_min, "source": source},
                "query_template":   query_template,
                "n_passages_total": total,
                "by_tf":            results,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    @mcp.tool()
    def check_papers_indexed(doc_ids: List[str]) -> str:
        """Check whether given doc_ids are present in the ledger.

        Args:
            doc_ids: doc_id strings (e.g. ["PMID:12345", "PMID:67890"]).
        """
        from paperqa_mcp.tools.admin import load_ledger
        ledger = load_ledger()
        papers = ledger.get("papers", {})
        status = {}
        missing: List[str] = []
        for did in doc_ids:
            if did in papers:
                status[did] = {"in_index": True, **papers[did]}
            else:
                status[did] = {"in_index": False}
                missing.append(did)
        return json.dumps(
            {
                "n_checked": len(doc_ids),
                "n_indexed": sum(1 for v in status.values() if v["in_index"]),
                "missing":   missing,
                "status":    status,
            },
            ensure_ascii=False,
            indent=2,
        )
