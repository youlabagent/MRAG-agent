"""SQLite-backed cache for AlphaGenome predictions.

Cache key = SHA256 of canonical request JSON (chrom, pos, ref, alt, interval, outputs, tissue, model).
Stored value = JSON-serialised summary of the response (small).
Full responses are pickled to disk under cache_dir/blobs/<hash>.pkl on demand.
"""
from __future__ import annotations
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .config import cache_dir

_DDL = """
CREATE TABLE IF NOT EXISTS ag_cache (
    key TEXT PRIMARY KEY,
    request_json TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    blob_path TEXT,
    created_at REAL NOT NULL
)
"""


def _conn() -> sqlite3.Connection:
    db = cache_dir() / "cache.sqlite"
    c = sqlite3.connect(str(db))
    c.execute(_DDL)
    return c


def hash_request(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def get(key: str) -> dict[str, Any] | None:
    with _conn() as c:
        row = c.execute(
            "SELECT request_json, summary_json, blob_path, created_at FROM ag_cache WHERE key=?",
            (key,),
        ).fetchone()
    if not row:
        return None
    req, summary, blob, ts = row
    return {
        "key": key,
        "request": json.loads(req),
        "summary": json.loads(summary),
        "blob_path": blob,
        "created_at": ts,
    }


def put(key: str, request: dict[str, Any], summary: dict[str, Any], blob: bytes | None = None) -> None:
    blob_path = None
    if blob is not None:
        blob_dir = cache_dir() / "blobs"
        blob_dir.mkdir(exist_ok=True)
        blob_path = str(blob_dir / f"{key}.pkl")
        Path(blob_path).write_bytes(blob)
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO ag_cache VALUES (?, ?, ?, ?, ?)",
            (
                key,
                json.dumps(request, sort_keys=True),
                json.dumps(summary, sort_keys=True),
                blob_path,
                time.time(),
            ),
        )
