"""Node abstract base class for the MRAG pipeline DAG.

Each pipeline step inherits from Node. The pipeline executor:
  1. computes Node.input_hash() before running
  2. checks checkpoint cache (skip if hash matches and status=='ok')
  3. calls Node.run() which populates Node.outputs
  4. runs post-flight gates via Node.validate()
  5. appends a step record to the manifest

Two execution modes:
  - computed:  Node.run() actually invokes underlying MCP tool(s)
  - provided:  Node.provided_outputs is set externally; run() just validates files exist
"""
from __future__ import annotations
import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

# Gate / Diagnosis result shapes (kept light to avoid cyclic imports)
GateResult = dict     # {"name": str, "result": "pass"|"warn"|"fail", "value": Any, "msg": str}
Diagnosis  = Optional[dict]   # populated by debug.py rule book on fail


class Node(ABC):
    """Abstract pipeline node."""

    # Subclass must set:
    name: str = "unnamed"

    def __init__(
        self,
        inputs: dict[str, Any],
        params: Optional[dict[str, Any]] = None,
        output_dir: Optional[Path] = None,
        provided_outputs: Optional[dict[str, str]] = None,
    ):
        self.inputs = inputs
        self.params = params or {}
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.provided_outputs = provided_outputs  # if set: skip computation

        # Populated during run()
        self.outputs:  dict[str, str] = {}
        self.status:   str            = "queued"   # queued|running|ok|fail|provided|skipped
        self.started_at:  Optional[str] = None
        self.ended_at:    Optional[str] = None
        self.gates:    list[GateResult] = []
        self.diagnosis: Diagnosis = None
        self.error:    Optional[str] = None

    # ---- hashing for checkpoint cache ----
    def input_hash(self) -> str:
        """sha256 of {inputs, params}. Used as cache key."""
        # File paths get hashed BY CONTENT (when small) OR by path+mtime+size (when large).
        # For now use path+mtime+size to avoid reading huge files on every run.
        def _hashable(x):
            if isinstance(x, Path) or (isinstance(x, str) and Path(x).exists()):
                p = Path(x)
                try:
                    st = p.stat()
                    return f"FILE:{p.resolve()}|mtime={int(st.st_mtime)}|size={st.st_size}"
                except OSError:
                    return f"FILE:{p}|missing"
            return x
        norm = {
            "inputs": {k: _hashable(v) for k, v in sorted(self.inputs.items())},
            "params": dict(sorted(self.params.items())),
            "node":   self.name,
        }
        s = json.dumps(norm, default=str, sort_keys=True)
        return "sha256:" + hashlib.sha256(s.encode()).hexdigest()

    # ---- execution ----
    def execute(self) -> None:
        """Top-level entry called by the pipeline executor."""
        self.started_at = _ts()
        self.status = "running"
        try:
            if self.provided_outputs:
                # Validate provided files exist + apply post-flight gates
                self._validate_provided()
                self.outputs = dict(self.provided_outputs)
                self.status = "provided"
            else:
                self.run()                         # subclass-defined
                self.status = "ok"
            self.gates = self.validate()
            if any(g.get("result") == "fail" for g in self.gates):
                self.status = "fail"
                self.error = "post-flight gate failed"
        except Exception as e:
            self.status = "fail"
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.ended_at = _ts()

    def _validate_provided(self) -> None:
        """Check that every provided_outputs path exists and is non-empty."""
        for k, path in self.provided_outputs.items():
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"provided output '{k}' not found: {path}")
            if p.is_file() and p.stat().st_size == 0:
                raise ValueError(f"provided output '{k}' is empty: {path}")

    # ---- subclass interface ----
    @abstractmethod
    def run(self) -> None:
        """Actually execute. Must populate self.outputs (dict of output path strings)."""
        ...

    def validate(self) -> list[GateResult]:
        """Post-flight gates. Default: no gates. Subclasses can override."""
        return []

    # ---- manifest record ----
    def to_manifest(self) -> dict:
        return {
            "name":         self.name,
            "status":       self.status,
            "started_at":   self.started_at,
            "ended_at":     self.ended_at,
            "input_hash":   self.input_hash(),
            "params":       self.params,
            "outputs":      [{"key": k, "path": v, **_file_metadata(v)}
                             if isinstance(v, (str, bytes, Path)) else
                             {"key": k, "value": v}
                             for k, v in self.outputs.items()],
            "gates":        self.gates,
            "diagnosis":    self.diagnosis,
            "error":        self.error,
        }


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _file_metadata(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"exists": False}
    if p.is_dir():
        return {"exists": True, "is_dir": True}
    try:
        st = p.stat()
        return {"exists": True, "size": st.st_size}
    except OSError as e:
        return {"exists": True, "stat_error": str(e)}


# ---- Helper: capture-decorated-tool pattern ----
class _CaptureMCP:
    """Reusable FakeMCP that captures @mcp.tool() functions for direct call."""
    def __init__(self):
        self.captured: dict[str, Any] = {}
    def tool(self, *a, **kw):
        def deco(fn):
            self.captured[fn.__name__] = fn
            return fn
        return deco


def capture_mcp_tools(register_fn) -> dict[str, Any]:
    """Invoke an MCP register function and return {tool name: callable}.

    MRAG_TOOL_TRANSPORT=mcp (the default since 2026-10-12): each callable sends its call as an MCP tools/call
    request, through an MCP client session over an in-memory transport, to a FastMCP server holding these
    tools (same class as the package's server.py, same register function); it returns what the tool returned
    and raises what the tool raised (mrag_orchestrator/tool_transport.py).
    MRAG_TOOL_TRANSPORT=direct: invoke the register function with a fake mcp and return the captured Python
    functions -- the behaviour before 2026-10-12."""
    from ..tool_transport import transport_name, mcp_tool_table
    if transport_name() == "mcp":
        return mcp_tool_table(register_fn)
    fake = _CaptureMCP()
    register_fn(fake)
    return fake.captured
