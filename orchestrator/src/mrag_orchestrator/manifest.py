"""Provenance manifest builder.

Single source of truth for one MRAG run. Atomically updated after each step
(temp file + rename) so a Web UI can poll without race conditions.
"""
from __future__ import annotations
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from .tool_transport import transport_name as _tool_transport


def file_sha256(path: str | Path) -> str:
    """sha256 of file contents (cheap for typical pipeline outputs)."""
    p = Path(path)
    if not p.exists() or p.is_dir():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(2 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


class Manifest:
    """Live manifest accumulator. Calls to `write_atomic()` persist current state."""

    def __init__(
        self,
        run_id: str,
        config_path: Optional[str] = None,
        config_hash: Optional[str] = None,
        code_version: str = "mrag-orchestrator 0.1.0",
        llm: Optional[dict] = None,
        manifest_path: Optional[Path] = None,
    ):
        self.run_id = run_id
        self.data = {
            "schema_version": "1.0",
            "run_id":      run_id,
            "started_at":  _ts(),
            "ended_at":    None,
            "status":      "running",          # running|ok|fail|partial
            "config_path": config_path,
            "config_hash": config_hash,
            "code_version": code_version,
            # 2026-10-12: how this process called the MCP tools -- "mcp" (MCP client sessions, in-memory
            # transport) or "direct" (Python functions); MRAG_TOOL_TRANSPORT
            "tool_transport": _tool_transport(),
            "model":       llm or {},
            "inputs":      {},
            "steps":       [],
            "summary":     {},
        }
        self.path = Path(manifest_path) if manifest_path else None
        # the per-replicate HOMER scans add steps from worker threads; one writer at a time, or
        # two write_atomic calls race on the same .tmp file
        self._lock = threading.RLock()

    # ---- input declarations ----
    def declare_inputs(self, kind: str, paths: list[str | Path]) -> None:
        self.data["inputs"].setdefault(kind, [])
        for p in paths:
            self.data["inputs"][kind].append({
                "path":   str(p),
                "sha256": file_sha256(p),
                "size":   Path(p).stat().st_size if Path(p).exists() and Path(p).is_file() else 0,
            })

    # ---- step records ----
    def add_step(self, record: dict) -> None:
        # Augment outputs with sha256 if file-like + file exists
        for out in record.get("outputs", []):
            if out.get("exists") and not out.get("is_dir"):
                out["sha256"] = file_sha256(out["path"])
        with self._lock:
            self.data["steps"].append(record)

    def update_step(self, name: str, **kwargs) -> None:
        for s in self.data["steps"]:
            if s.get("name") == name:
                s.update(kwargs); return

    # ---- finalization ----
    def finalize(self, summary: Optional[dict] = None) -> None:
        self.data["ended_at"] = _ts()
        any_fail = any(s.get("status") == "fail" for s in self.data["steps"])
        any_run  = any(s.get("status") == "running" for s in self.data["steps"])
        if any_fail:
            self.data["status"] = "fail"
        elif any_run:
            self.data["status"] = "partial"
        else:
            self.data["status"] = "ok"
        if summary:
            self.data["summary"] = summary

    # ---- atomic persistence ----
    def write_atomic(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with self._lock:
            tmp.write_text(json.dumps(self.data, indent=2, default=str))
            os.replace(tmp, self.path)

    # ---- read/load ----
    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        d = json.loads(Path(path).read_text())
        m = cls(run_id=d["run_id"], manifest_path=Path(path))
        m.data = d
        return m


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
