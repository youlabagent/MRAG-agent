"""Admin tools — incremental indexing, ledger, status, rebuild.

The ledger at ``/BIO1/MRAG/Agent/libs/paperqa/index_ledger.json`` tracks which
``doc_id`` values have been indexed and lets ``add_papers`` skip work that
is already done. Qdrant chunk IDs are deterministic
(``md5(doc_id|section|section_chunk_idx)``), so even without the ledger an
``upsert`` would be idempotent -- the ledger just lets us avoid recomputing
embeddings.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP


# ----- Constants (mirror phase5_embed.py) ---------------------------------
PAPERQA_ROOT = Path("/BIO1/MRAG/Agent/libs/paperqa")
QDRANT_PATH  = PAPERQA_ROOT / "qdrant"
LEDGER_PATH  = PAPERQA_ROOT / "index_ledger.json"
PARSED_DIR   = PAPERQA_ROOT / "parsed"
CHUNKS_DIR   = PAPERQA_ROOT / "chunks"
CANCERS      = ["prostate", "sclc", "breast"]

COLLECTION   = "mrag_papers"
EMBED_MODEL  = "BAAI/bge-m3"
EMBED_DIM    = 1024
MAX_TOKENS   = 512
OVERLAP      = 64
BATCH_SIZE   = 32


# ----- Ledger -------------------------------------------------------------
def load_ledger() -> Dict[str, Any]:
    """Load the index ledger; create empty skeleton if missing."""
    if LEDGER_PATH.exists():
        with LEDGER_PATH.open() as f:
            return json.load(f)
    return {
        "collection":  COLLECTION,
        "model":       EMBED_MODEL,
        "dim":         EMBED_DIM,
        "papers":      {},
        "last_updated": None,
    }


def save_ledger(ledger: Dict[str, Any]) -> None:
    ledger["last_updated"] = datetime.now(timezone.utc).isoformat()
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER_PATH.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, LEDGER_PATH)


# ----- Chunking (mirror phase5_embed.chunk_text/chunks_from_record) ------
def _chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP) -> List[str]:
    words = text.split()
    if len(words) <= max_tokens:
        return [text]
    chunks: List[str] = []
    step = max_tokens - overlap
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i : i + max_tokens]))
        if i + max_tokens >= len(words):
            break
    return chunks


def _chunks_from_record(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    sections = rec.get("sections", [])
    # Some parsers stored sections as a stringified Python list -- be tolerant.
    if isinstance(sections, str):
        try:
            import ast
            sections = ast.literal_eval(sections)
        except Exception:
            sections = []
    for sec in sections:
        sname = sec.get("section", "other")
        for j, piece in enumerate(_chunk_text(sec.get("text", ""))):
            out.append({
                "doc_id":            rec["doc_id"],
                "cancer":            rec.get("cancer", ""),
                "source":            rec.get("source", ""),
                "pmid":              rec.get("pmid"),
                "pmcid":             rec.get("pmcid"),
                "doi":               rec.get("doi"),
                "title":             rec.get("title", ""),
                "year":              rec.get("year"),
                "journal":           rec.get("journal", ""),
                "section":           sname,
                "section_chunk_idx": j,
                "text":              piece,
                "char_len":          len(piece),
            })
    return out


def _chunk_id(c: Dict[str, Any]) -> str:
    raw = f"{c['doc_id']}|{c['section']}|{c['section_chunk_idx']}"
    return hashlib.md5(raw.encode()).hexdigest()


# ----- Embedder + Qdrant client singletons --------------------------------
_EMBEDDER = None
_QCLIENT  = None


def _get_qclient():
    global _QCLIENT
    if _QCLIENT is None:
        from qdrant_client import QdrantClient
        _QCLIENT = QdrantClient(path=str(QDRANT_PATH))
    return _QCLIENT


def _release_qclient():
    """Release Qdrant lock so external scripts can run if needed."""
    global _QCLIENT
    if _QCLIENT is not None:
        try:
            _QCLIENT.close()
        except Exception:
            pass
        _QCLIENT = None


def _get_embedder(device: Optional[str] = None):
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        if device is None:
            device = os.environ.get("MRAG_EMBED_DEVICE", "cuda:0")
        try:
            _EMBEDDER = SentenceTransformer(EMBED_MODEL, device=device)
        except Exception:
            _EMBEDDER = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _EMBEDDER


def _payload_for(c: Dict[str, Any]) -> Dict[str, Any]:
    title = c.get("title") or ""
    journal = c.get("journal") or ""
    return {
        "doc_id":             c["doc_id"],
        "cancer":             c.get("cancer", ""),
        "source":             c.get("source", ""),
        "pmid":               c.get("pmid"),
        "pmcid":              c.get("pmcid"),
        "doi":                c.get("doi"),
        "title":              title[:300] if title else "",
        "year":               c.get("year"),
        "journal":            journal[:120] if journal else "",
        "section":            c["section"],
        "section_chunk_idx":  c["section_chunk_idx"],
        "text":               c["text"][:1500],
    }


def _embed_and_upsert(chunks: List[Dict[str, Any]], device: Optional[str] = None) -> int:
    """Embed chunks in batches and upsert into the live collection."""
    if not chunks:
        return 0
    from qdrant_client.models import PointStruct
    model  = _get_embedder(device)
    client = _get_qclient()
    n = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        vecs  = model.encode(
            [c["text"] for c in batch],
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        points = [
            PointStruct(id=_chunk_id(c), vector=v.tolist(), payload=_payload_for(c))
            for c, v in zip(batch, vecs)
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        n += len(batch)
    return n


# ----- Tool registration --------------------------------------------------
def register_admin_tools(mcp: FastMCP):

    @mcp.tool()
    def index_status() -> str:
        """Snapshot of the Qdrant index + the ledger.

        Returns:
            collection / model / dim, total chunks in Qdrant, ledger paper
            count, per-cancer ledger breakdown, and last_updated timestamp.
        """
        client = _get_qclient()
        try:
            qdrant_count = client.count(collection_name=COLLECTION, exact=True).count
        except Exception as e:
            qdrant_count = f"(error: {e})"

        ledger = load_ledger()
        per_cancer: Dict[str, int] = {}
        per_source: Dict[str, int] = {}
        for p in ledger.get("papers", {}).values():
            per_cancer[p.get("cancer", "")] = per_cancer.get(p.get("cancer", ""), 0) + 1
            per_source[p.get("source", "")] = per_source.get(p.get("source", ""), 0) + 1

        return json.dumps(
            {
                "collection":         COLLECTION,
                "embed_model":        EMBED_MODEL,
                "embed_dim":          EMBED_DIM,
                "qdrant_path":        str(QDRANT_PATH),
                "qdrant_chunks":      qdrant_count,
                "ledger_path":        str(LEDGER_PATH),
                "ledger_papers":      len(ledger.get("papers", {})),
                "ledger_last_updated": ledger.get("last_updated"),
                "per_cancer_papers":  per_cancer,
                "per_source_papers":  per_source,
            },
            ensure_ascii=False,
            indent=2,
        )

    @mcp.tool()
    def add_papers(
        parsed_jsonl_path: str,
        force_reindex: bool = False,
        dry_run: bool = False,
        device: Optional[str] = None,
    ) -> str:
        """Incrementally index papers from a phase4-parsed JSONL file.

        Each record must follow the phase4_parse.py schema (doc_id, sections,
        cancer, source, pmid, etc.). Papers whose doc_id is already in the
        ledger are skipped unless ``force_reindex=True``.

        Args:
            parsed_jsonl_path: Path to a parsed JSONL produced by
                ``scripts/phase4_parse.py``.
            force_reindex: If True, re-embed and overwrite even already-indexed
                doc_ids. Useful when the chunking strategy or payload schema
                changed.
            dry_run: If True, only count would-be additions; no embedding or
                upsert is performed.
            device: Embedder device override (e.g. "cuda:1", "cpu").

        Behaviour:
            * Reads JSONL lazily; one paper at a time.
            * Records with empty/missing ``sections`` are skipped.
            * Ledger is updated atomically only after successful upsert.

        Returns a JSON summary with counts (read, new, skipped, indexed) and
        the resulting ledger size.
        """
        path = Path(parsed_jsonl_path).expanduser()
        if not path.exists():
            return json.dumps({"status": "error", "message": f"not found: {path}"})

        ledger = load_ledger()
        papers = ledger.setdefault("papers", {})

        n_read = n_new = n_force = n_skipped = n_indexed = n_chunks = 0
        errors: List[str] = []
        new_doc_ids: List[str] = []
        t0 = time.time()

        # Stream records; collect chunks to embed in one batch per paper.
        to_index: List[Dict[str, Any]] = []
        per_paper_index: List[Dict[str, Any]] = []

        with path.open() as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                n_read += 1
                try:
                    rec = json.loads(line)
                except Exception as e:
                    errors.append(f"json parse error at line {n_read}: {e}")
                    continue
                doc_id = rec.get("doc_id")
                if not doc_id:
                    errors.append(f"missing doc_id at line {n_read}")
                    continue
                already = doc_id in papers
                if already and not force_reindex:
                    n_skipped += 1
                    continue
                if already and force_reindex:
                    n_force += 1
                else:
                    n_new += 1

                chunks = _chunks_from_record(rec)
                if not chunks:
                    errors.append(f"no chunks for {doc_id}")
                    continue
                to_index.extend(chunks)
                per_paper_index.append({
                    "doc_id":  doc_id,
                    "n_chunks": len(chunks),
                    "cancer":   rec.get("cancer", ""),
                    "source":   rec.get("source", ""),
                    "pmid":     rec.get("pmid"),
                    "year":     rec.get("year"),
                })
                new_doc_ids.append(doc_id)

        if dry_run:
            return json.dumps(
                {
                    "status":        "dry_run",
                    "input":         str(path),
                    "n_read":        n_read,
                    "n_new":         n_new,
                    "n_force":       n_force,
                    "n_skipped":     n_skipped,
                    "n_chunks_planned": len(to_index),
                    "would_index":   new_doc_ids[:20],
                    "would_index_total": len(new_doc_ids),
                    "errors":        errors[:20],
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        if to_index:
            try:
                n_chunks = _embed_and_upsert(to_index, device=device)
            except Exception as e:
                return json.dumps(
                    {
                        "status":   "error",
                        "message":  f"embed/upsert failed: {e}",
                        "n_chunks_attempted": len(to_index),
                        "errors":   errors,
                    },
                    ensure_ascii=False,
                    indent=2,
                )

        now = datetime.now(timezone.utc).isoformat()
        for entry in per_paper_index:
            papers[entry["doc_id"]] = {
                "indexed_at":  now,
                "n_chunks":    entry["n_chunks"],
                "cancer":      entry["cancer"],
                "source":      entry["source"],
                "pmid":        entry["pmid"],
                "year":        entry["year"],
            }
        n_indexed = len(per_paper_index)
        save_ledger(ledger)

        return json.dumps(
            {
                "status":         "ok",
                "input":          str(path),
                "n_read":         n_read,
                "n_new":          n_new,
                "n_force":        n_force,
                "n_skipped":      n_skipped,
                "n_papers_indexed": n_indexed,
                "n_chunks_upserted": n_chunks,
                "elapsed_s":      round(time.time() - t0, 2),
                "ledger_total":   len(papers),
                "errors":         errors[:20],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    @mcp.tool()
    def remove_paper(doc_id: str) -> str:
        """Delete all chunks belonging to a single doc_id and update the ledger.

        Args:
            doc_id: The full doc_id (e.g. "PMID:12345").
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        client = _get_qclient()
        flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])

        # Count before delete
        try:
            before = client.count(
                collection_name=COLLECTION, count_filter=flt, exact=True
            ).count
        except Exception as e:
            return json.dumps({"status": "error", "message": f"count failed: {e}"})

        if before == 0:
            ledger = load_ledger()
            removed_from_ledger = ledger.get("papers", {}).pop(doc_id, None) is not None
            if removed_from_ledger:
                save_ledger(ledger)
            return json.dumps(
                {
                    "status":  "noop",
                    "doc_id":  doc_id,
                    "message": "no chunks found in Qdrant",
                    "ledger_removed": removed_from_ledger,
                },
                ensure_ascii=False,
                indent=2,
            )

        try:
            client.delete(collection_name=COLLECTION, points_selector=flt)
        except Exception as e:
            return json.dumps({"status": "error", "message": f"delete failed: {e}"})

        ledger = load_ledger()
        had = ledger.get("papers", {}).pop(doc_id, None) is not None
        save_ledger(ledger)
        return json.dumps(
            {
                "status":            "ok",
                "doc_id":            doc_id,
                "chunks_deleted":    before,
                "ledger_had_entry":  had,
                "ledger_total":      len(ledger.get("papers", {})),
            },
            ensure_ascii=False,
            indent=2,
        )

    @mcp.tool()
    def rebuild_index(
        parsed_dir: str = "/BIO1/MRAG/Agent/libs/paperqa/parsed",
        cancers: Optional[List[str]] = None,
        confirm: bool = False,
        device: Optional[str] = None,
    ) -> str:
        """DESTRUCTIVE: drop the Qdrant collection and re-embed everything.

        Requires ``confirm=True`` to actually run. Otherwise returns a preview
        of what would be done. After rebuild the ledger is regenerated from
        the parsed files.

        Args:
            parsed_dir: Directory containing per-cancer ``{cancer}.jsonl``.
            cancers:    List of cancer labels (default ``["prostate","sclc","breast"]``).
            confirm:    Must be True to actually drop + rebuild.
            device:     Embedder device override.
        """
        if cancers is None:
            cancers = CANCERS
        parsed = Path(parsed_dir).expanduser()
        files = []
        for c in cancers:
            f = parsed / f"{c}.jsonl"
            if f.exists():
                files.append((c, f))

        if not confirm:
            n_lines = {}
            for c, f in files:
                with f.open() as fp:
                    n_lines[c] = sum(1 for _ in fp)
            return json.dumps(
                {
                    "status":           "preview",
                    "message":          "rerun with confirm=True to actually rebuild",
                    "parsed_dir":       str(parsed),
                    "files":            [{"cancer": c, "path": str(f), "papers": n_lines[c]} for c, f in files],
                    "would_drop_collection": COLLECTION,
                },
                ensure_ascii=False,
                indent=2,
            )

        # Confirmed -- recreate the collection then reindex everything.
        from qdrant_client.models import VectorParams, Distance
        client = _get_qclient()
        try:
            existing = [c.name for c in client.get_collections().collections]
            if COLLECTION in existing:
                client.delete_collection(COLLECTION)
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
        except Exception as e:
            return json.dumps({"status": "error", "message": f"recreate failed: {e}"})

        # Reset ledger and re-add via the add_papers path.
        new_ledger = {
            "collection":  COLLECTION,
            "model":       EMBED_MODEL,
            "dim":         EMBED_DIM,
            "papers":      {},
            "last_updated": None,
        }
        save_ledger(new_ledger)

        # Iterate parsed files and run add_papers in force mode for each.
        per_file: List[Dict[str, Any]] = []
        # We avoid recursive tool invocation here -- reuse the internals.
        for c, f in files:
            t0 = time.time()
            ledger = load_ledger()
            papers = ledger.setdefault("papers", {})
            chunks_buf: List[Dict[str, Any]] = []
            paper_meta: List[Dict[str, Any]] = []
            n_read = 0
            with f.open() as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    n_read += 1
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    did = rec.get("doc_id")
                    if not did:
                        continue
                    chs = _chunks_from_record(rec)
                    if not chs:
                        continue
                    chunks_buf.extend(chs)
                    paper_meta.append({
                        "doc_id":  did,
                        "n_chunks": len(chs),
                        "cancer":   rec.get("cancer", ""),
                        "source":   rec.get("source", ""),
                        "pmid":     rec.get("pmid"),
                        "year":     rec.get("year"),
                    })
            n_chunks = _embed_and_upsert(chunks_buf, device=device)
            now = datetime.now(timezone.utc).isoformat()
            for entry in paper_meta:
                papers[entry["doc_id"]] = {
                    "indexed_at": now,
                    "n_chunks":   entry["n_chunks"],
                    "cancer":     entry["cancer"],
                    "source":     entry["source"],
                    "pmid":       entry["pmid"],
                    "year":       entry["year"],
                }
            save_ledger(ledger)
            per_file.append({
                "cancer":   c,
                "papers":   len(paper_meta),
                "chunks":   n_chunks,
                "elapsed_s": round(time.time() - t0, 1),
            })

        return json.dumps(
            {
                "status":      "ok",
                "rebuilt":     True,
                "collection":  COLLECTION,
                "per_file":    per_file,
                "ledger_total": len(load_ledger().get("papers", {})),
            },
            ensure_ascii=False,
            indent=2,
        )

    @mcp.tool()
    def release_qdrant_lock() -> str:
        """Close the Qdrant client so external scripts can access the path.

        Use this before running phase5_embed.py or similar external tools.
        The next retrieval/admin call will transparently re-open the client.
        """
        _release_qclient()
        return json.dumps({"status": "ok", "message": "Qdrant client closed"})
