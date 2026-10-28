"""Admin tools — coverage lookup, cache inspection, cache pruning."""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from alphagenome_client.config import (
    cache_dir,
    PROSTATE_CELL_LINES,
    PROSTATE_MODALITY_COVERAGE,
)


def _cache_db_path():
    return cache_dir() / "cache.sqlite"


def register_admin_tools(mcp: FastMCP):

    @mcp.tool()
    def list_tissues() -> str:
        """Return the prostate cell-line ontology map + modality coverage.

        Use this before counterfactual_at_peak / batch_validate_tfs to pick
        a `cell_line` that supports the modalities you need. Notable
        constraint: among prostate models, **only PC-3 has ATAC** (the
        other lines lack ATAC tracks in the AlphaGenome metadata).

        Returns JSON with:
            cell_lines:  {name: ontology_curie}
            coverage:    {name: {modality: True|False}}
            defaults:    suggested cell_line per modality
        """
        defaults = {}
        # For each modality, pick the first cell line that supports it.
        for modality in ("ATAC", "RNA_SEQ", "CHIP_TF", "CHIP_HISTONE", "DNASE", "CAGE"):
            for cl, cov in PROSTATE_MODALITY_COVERAGE.items():
                if cov.get(modality):
                    defaults[modality] = cl
                    break
        return json.dumps(
            {
                "cell_lines": PROSTATE_CELL_LINES,
                "coverage":   PROSTATE_MODALITY_COVERAGE,
                "defaults":   defaults,
            },
            ensure_ascii=False,
            indent=2,
        )

    @mcp.tool()
    def check_coverage(
        cell_line: str,
        outputs: Optional[List[str]] = None,
    ) -> str:
        """Verify a (cell_line, outputs) combination is supported.

        Call this before a batch AlphaGenome run to fail fast if the chosen
        cell line lacks one of the requested modalities.

        Args:
            cell_line: One of the keys in PROSTATE_CELL_LINES.
            outputs:   Modalities to check. Default ["ATAC","RNA_SEQ","CHIP_TF"].

        Returns JSON with per-modality availability and an overall `ok` flag.
        """
        if outputs is None:
            outputs = ["ATAC", "RNA_SEQ", "CHIP_TF"]
        cov = PROSTATE_MODALITY_COVERAGE.get(cell_line)
        if cov is None:
            # Try case-insensitive
            for k, v in PROSTATE_MODALITY_COVERAGE.items():
                if k.lower() == cell_line.lower():
                    cov = v
                    cell_line = k
                    break
        if cov is None:
            return json.dumps(
                {
                    "ok":      False,
                    "message": f"unknown cell_line '{cell_line}'",
                    "known":   sorted(PROSTATE_MODALITY_COVERAGE.keys()),
                },
                ensure_ascii=False,
                indent=2,
            )
        per_modality = {m: bool(cov.get(m, False)) for m in outputs}
        ok = all(per_modality.values())
        missing = [m for m, v in per_modality.items() if not v]
        return json.dumps(
            {
                "ok":            ok,
                "cell_line":     cell_line,
                "ontology":      PROSTATE_CELL_LINES.get(cell_line),
                "per_modality":  per_modality,
                "missing":       missing,
            },
            ensure_ascii=False,
            indent=2,
        )

    @mcp.tool()
    def cache_stats() -> str:
        """Inspect the AlphaGenome SQLite cache.

        Returns row count, total file size, and the most-recent timestamps.
        Helpful for understanding how much downstream Top-N analysis has
        already been cached (replay is free).
        """
        path = _cache_db_path()
        if not path.exists():
            return json.dumps({"status": "empty", "path": str(path)})
        size_bytes = path.stat().st_size

        with sqlite3.connect(str(path)) as c:
            try:
                row_count = c.execute("SELECT COUNT(*) FROM ag_cache").fetchone()[0]
            except sqlite3.OperationalError:
                row_count = 0
            try:
                rows = c.execute(
                    "SELECT key, created_at FROM ag_cache "
                    "ORDER BY created_at DESC LIMIT 5"
                ).fetchall()
                latest = [
                    {"key": k[:12], "created_at": datetime.fromtimestamp(t, tz=timezone.utc).isoformat()}
                    for k, t in rows
                ]
                # Min created_at for age
                first_ts = c.execute(
                    "SELECT MIN(created_at) FROM ag_cache"
                ).fetchone()[0]
            except sqlite3.OperationalError:
                latest = []
                first_ts = None

        return json.dumps(
            {
                "path":         str(path),
                "size_bytes":   size_bytes,
                "size_mb":      round(size_bytes / 1e6, 3),
                "row_count":    row_count,
                "oldest_entry": (datetime.fromtimestamp(first_ts, tz=timezone.utc).isoformat()
                                 if first_ts else None),
                "latest_5":     latest,
            },
            ensure_ascii=False,
            indent=2,
        )

    @mcp.tool()
    def purge_cache(
        before_iso: Optional[str] = None,
        confirm: bool = False,
    ) -> str:
        """Delete cache rows older than `before_iso`.

        Args:
            before_iso: ISO-8601 timestamp (e.g. "2026-01-01T00:00:00Z").
                        Rows with created_at < this are eligible for deletion.
                        If None, ALL rows are eligible.
            confirm:    Must be True to actually delete. Otherwise a preview
                        is returned.

        Returns JSON with count of matching rows (and the deletion outcome
        when confirm=True).
        """
        path = _cache_db_path()
        if not path.exists():
            return json.dumps({"status": "empty", "path": str(path)})

        if before_iso:
            try:
                dt = datetime.fromisoformat(before_iso.replace("Z", "+00:00"))
                before_ts = dt.timestamp()
            except Exception as e:
                return json.dumps({"status": "error", "message": f"bad before_iso: {e}"})
            where = "WHERE created_at < ?"
            params = (before_ts,)
        else:
            where = ""
            params = ()

        with sqlite3.connect(str(path)) as c:
            n_match = c.execute(
                f"SELECT COUNT(*) FROM ag_cache {where}", params
            ).fetchone()[0]
            if not confirm:
                return json.dumps(
                    {
                        "status":      "preview",
                        "message":     "rerun with confirm=True to actually delete",
                        "before_iso":  before_iso,
                        "n_matching":  n_match,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            c.execute(f"DELETE FROM ag_cache {where}", params)
            c.execute("VACUUM")
            remaining = c.execute("SELECT COUNT(*) FROM ag_cache").fetchone()[0]

        return json.dumps(
            {
                "status":    "ok",
                "deleted":   n_match,
                "remaining": remaining,
            },
            ensure_ascii=False,
            indent=2,
        )
