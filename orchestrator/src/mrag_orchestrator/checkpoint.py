"""Checkpoint / resume machinery.

Manifest-driven: a step is "cached" if a previous manifest contains an entry
with the same step name AND the same `input_hash` AND status == 'ok'.

The orchestrator passes manifests through; the executor consults the cache
before calling Node.execute().
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional


class CheckpointCache:
    """In-memory representation of a previous run's completed-step state.

    Build with `CheckpointCache.load_from_manifest(path)`. Then:
        cache.lookup(step_name, input_hash) -> dict | None
    Returns the manifest step record if it can be reused (status=='ok' and
    input_hash matches), else None.
    """

    def __init__(self, by_step: Optional[dict[str, dict]] = None):
        self._by_step = by_step or {}

    @classmethod
    def load_from_manifest(cls, path: str | Path) -> "CheckpointCache":
        p = Path(path)
        if not p.exists():
            return cls({})
        try:
            data = json.loads(p.read_text())
        except Exception:
            return cls({})
        idx: dict[str, dict] = {}
        for s in data.get("steps", []):
            name = s.get("name", "")
            if name and s.get("status") in ("ok", "provided"):
                # keep the LAST occurrence (in case a step was retried)
                idx[name] = s
        return cls(idx)

    def lookup(self, step_name: str, input_hash: str) -> Optional[dict]:
        rec = self._by_step.get(step_name)
        if not rec: return None
        if rec.get("input_hash") != input_hash: return None
        # Also verify all output FILES still exist (cache invalidation).
        # Skip scalar/metric outputs (no path separator).
        for out in rec.get("outputs", []):
            path = out.get("path","")
            if path and ("/" in path or "\\" in path):   # only check filesystem-like paths
                if not Path(path).exists():
                    return None
        return rec

    def all_cached_names(self) -> list[str]:
        return list(self._by_step.keys())


def restore_node_from_cache(node, cache_record: dict) -> None:
    """Populate a Node instance from a cached manifest record so it skips run()."""
    node.status = "skipped"      # 'skipped' = cache hit, distinct from 'ok'
    node.started_at = cache_record.get("started_at")
    node.ended_at   = cache_record.get("ended_at")
    node.outputs = {
        out["key"]: out["path"] for out in cache_record.get("outputs", [])
        if "key" in out and "path" in out
    }
    node.gates = cache_record.get("gates", [])
    node.diagnosis = cache_record.get("diagnosis")
    node.error = None
