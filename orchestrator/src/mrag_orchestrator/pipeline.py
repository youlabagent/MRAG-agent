"""Pipeline executor: topological run of Nodes with live manifest writing.

Sequential execution only; the Top-N fanout runs in variants.py.

Usage:
    nodes = [combined_node, diffbind_node, ...]
    manifest = Manifest(run_id=..., manifest_path=...)
    Pipeline(nodes, manifest).run()
"""
from __future__ import annotations
import time

from .nodes._base import Node
from .manifest import Manifest
from .checkpoint import CheckpointCache, restore_node_from_cache
from .debug      import diagnose, DiagnosisContext


class Pipeline:
    """Linear topological executor.

    Nodes are assumed to be passed in topological order. Each node executes, manifest
    is updated atomically, and on failure the rest is skipped (status='skipped').
    """

    def __init__(self, nodes: list[Node], manifest: Manifest,
                 stop_on_fail: bool = True,
                 cache: CheckpointCache | None = None):
        self.nodes = nodes
        self.manifest = manifest
        self.stop_on_fail = stop_on_fail
        self.cache = cache or CheckpointCache({})

    def run(self) -> dict:
        """Run all nodes sequentially. Return summary dict."""
        failures = 0
        skipped: list[str] = []
        for node in self.nodes:
            if failures and self.stop_on_fail:
                # Mark remaining as skipped
                node.status = "skipped"
                node.error  = "skipped due to upstream failure"
                self.manifest.add_step(node.to_manifest())
                self.manifest.write_atomic()
                skipped.append(node.name)
                continue

            t0 = time.time()
            # Cache lookup
            ihash = node.input_hash()
            cached = self.cache.lookup(node.name, ihash)
            if cached is not None:
                restore_node_from_cache(node, cached)
                dt = time.time() - t0
                print(f"[pipeline] ▷ {node.name} cached ({dt:.2f}s)", flush=True)
            else:
                print(f"[pipeline] ▶ {node.name} ...", flush=True)
                node.execute()
                dt = time.time() - t0
            self.manifest.add_step(node.to_manifest())
            self.manifest.write_atomic()

            if node.status == "fail":
                failures += 1
                # Run auto-debug
                ctx = DiagnosisContext(
                    step_name=node.name,
                    error_text=node.error or "",
                    gate_results=node.gates,
                    node_inputs=dict(node.inputs),
                    node_outputs=dict(node.outputs),
                )
                d = diagnose(ctx)
                node.diagnosis = {
                    "rule":              d.rule,
                    "likely_cause":      d.likely_cause,
                    "suggested_actions": d.suggested_actions,
                    "severity":          d.severity,
                }
                print(f"[pipeline] ✗ {node.name} FAIL ({dt:.1f}s): {node.error}", flush=True)
                print(f"[pipeline]   → {d.rule}: {d.likely_cause}", flush=True)
            elif node.status == "provided":
                print(f"[pipeline] ▶ {node.name} provided  ({dt:.1f}s)", flush=True)
            elif node.status == "skipped":
                pass  # already printed cached marker above
            else:
                print(f"[pipeline] ✓ {node.name} ok       ({dt:.1f}s)", flush=True)

        self.manifest.finalize()
        self.manifest.write_atomic()
        return {
            "status":   self.manifest.data["status"],
            "n_total":  len(self.nodes),
            "n_failed": failures,
            "n_skipped": len(skipped),
            "manifest_path": str(self.manifest.path) if self.manifest.path else None,
        }
