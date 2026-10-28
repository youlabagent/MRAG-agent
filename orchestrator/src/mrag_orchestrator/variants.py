"""Top 4/6/8 fanout orchestrator.

Given the shared upstream artifacts (HOMER full motif TSV, DiffBind_out.csv,
DESeq2.gene.csv, plus combined_mcp outputs top{N}.json), run the per-N
downstream branch:
    HomerFilter -> TRN -> Narrative

For each N (in [4,6,8] by default). Branches are independent and can be run
sequentially or in parallel. Results: per-N artifact dict.

Cross-variant synthesis (synthesis.py) consumes the resulting dict.

This module does NOT do the final HTML render — that's the caller's job after
synthesis. variants.py just hands back the per-N artifact paths.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .nodes._base import Node
from .nodes.homer       import HomerFilterNode
from .nodes.trn         import TRNNode
from .nodes.viz_report  import NarrativeNode
from .manifest          import Manifest
from .checkpoint        import CheckpointCache, restore_node_from_cache
from .debug             import diagnose, DiagnosisContext
from .derive_summaries  import derive_tfs_summary, derive_targets_summary


def _exec_with_cache(node: Node, cache: Optional[CheckpointCache]) -> None:
    if cache is not None:
        ihash = node.input_hash()
        cached = cache.lookup(node.name, ihash)
        if cached is not None:
            restore_node_from_cache(node, cached)
            return
    node.execute()
    if node.status == "fail":
        ctx = DiagnosisContext(step_name=node.name,
                               error_text=node.error or "",
                               gate_results=node.gates,
                               node_inputs=dict(node.inputs),
                               node_outputs=dict(node.outputs))
        d = diagnose(ctx)
        node.diagnosis = {"rule": d.rule, "likely_cause": d.likely_cause,
                          "suggested_actions": d.suggested_actions,
                          "severity": d.severity}


def run_variant_branch(
    N: int,
    shared: dict[str, str],
    output_dir: Path,
    llm_endpoint: str,
    llm_model: str,
    manifest: Optional[Manifest] = None,
    run_narrative: bool = True,
    cache: Optional[CheckpointCache] = None,
) -> dict:
    """Run a single N-branch: HomerFilter -> TRN -> [Narrative].

    Args:
      N: variant size (4, 6, 8)
      shared: {top_json, motif_tsv, diffbind_csv, deseq_csv,
               tfs_summary_json, targets_summary_json, conditions_meta_json}
      output_dir: per-N branch directory (e.g. branches/top4/)
    Returns: dict with paths to all produced artifacts.
    """
    branch_dir = output_dir / f"top{N}"
    branch_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {"N": str(N), "branch_dir": str(branch_dir)}
    nodes_run: list[Node] = []

    # 1. HOMER filter
    homer_filter = HomerFilterNode(
        inputs={"motif_tsv": shared["motif_tsv"], "tfs": shared["top_json"]},
        params={"top_n": N, "only_promoter": True, "case_insensitive": True},
        output_dir=branch_dir,
    )
    homer_filter.name = f"homer_filter_top{N}"
    _exec_with_cache(homer_filter, cache)
    nodes_run.append(homer_filter)
    if homer_filter.status == "fail":
        artifacts["error"] = f"homer_filter: {homer_filter.error}"
        if manifest: _record_branch(manifest, nodes_run)
        return artifacts
    artifacts["homer_filter_tsv"] = homer_filter.outputs["homer_filter_tsv"]

    # 2. TRN
    trn = TRNNode(
        inputs={
            "homer_filter_tsv": artifacts["homer_filter_tsv"],
            "diffbind_csv":     shared["diffbind_csv"],
            "deseq_csv":        shared["deseq_csv"],
        },
        params={"top_n": N},
        output_dir=branch_dir,
    )
    trn.name = f"trn_top{N}"
    _exec_with_cache(trn, cache)
    nodes_run.append(trn)
    if trn.status == "fail":
        artifacts["error"] = f"trn: {trn.error}"
        if manifest: _record_branch(manifest, nodes_run)
        return artifacts
    artifacts["consensus_tsv"] = trn.outputs["consensus_tsv"]

    # 3. Auto-derive aux JSONs if not provided
    import json as _json
    tfs_path = shared.get("tfs_summary_json") or ""
    targets_path = shared.get("targets_summary_json") or ""
    if not tfs_path or not Path(tfs_path).exists():
        integ = shared.get("integrated_csv","") or ""
        top_json = shared.get("top_json","") or ""
        # 2026-09-16 (narrtf): the narrative TF panel is EXACTLY this branch's
        # top{N}.json (same set + order as Section 3 / TRN / homer_filter). The
        # old INTEGRATED.csv re-sort lacked pick_topn's MDscore>=0 filter and
        # took max(N+2, 10) rows; the extra rows had no consumer (NarrativeNode
        # slices [:top_n], targets use [:N], _build_notable_findings ignores
        # `tfs`), so they are dropped.
        if (top_json and Path(top_json).exists()) or (integ and Path(integ).exists()):
            tfs_full = derive_tfs_summary(integ, top_n=N, top_json=top_json or None)
            tfs_path = str(branch_dir / "tfs_summary.json")
            Path(tfs_path).write_text(_json.dumps(tfs_full, indent=2))
    if not targets_path or not Path(targets_path).exists():
        cons = artifacts.get("consensus_tsv","")
        if cons and Path(cons).exists():
            top_names = []
            if tfs_path and Path(tfs_path).exists():
                top_names = [t["name"] for t in _json.loads(Path(tfs_path).read_text())[:N]]
            if top_names:
                targets_n = derive_targets_summary(cons, top_names, top_targets=12)
                targets_path = str(branch_dir / "targets_summary.json")
                Path(targets_path).write_text(_json.dumps(targets_n, indent=2))

    # 4. Narrative
    if run_narrative:
        # Authoritative study domain for the narrative + executive-summary
        # prompts (2026-09-16): runner passes the user-supplied cancer_type /
        # tissue (never its 'prostate' fallback); other callers: `field` if set.
        if "study_cancer_type" in shared or "study_tissue" in shared:
            _study_ct = shared.get("study_cancer_type") or ""
            _study_ti = shared.get("study_tissue") or ""
        else:
            _study_ct, _study_ti = (shared.get("field") or ""), ""
        narr = NarrativeNode(
            inputs={
                "tfs_summary_json":     tfs_path,
                "targets_summary_json": targets_path,
                "conditions_meta_json": shared["conditions_meta_json"],
            },
            params={
                "top_n":     N,
                "glm_endpoint": llm_endpoint,
                "glm_model":    llm_model,
                "temperature":  0.2,
                "max_tokens":   2000,
                "generate_abstract": True,
                "field": shared.get("field", "prostate"),
                "study_cancer_type": _study_ct,
                "study_tissue":      _study_ti,
                # 2026-09-17 (llmdomain): per-domain Novelty on the Section-14
                # cards, user's literature domains in the user's order (pan-cancer
                # auto-append hidden). Part of input_hash -> a domain change
                # invalidates a cached narrative.
                "novelty_domains":   list(shared.get("literature_domains_user") or []),
                # 2026-09-23 (Task B): a non-human run's literature lookups go
                # through the human ortholog. Part of input_hash, so switching
                # organism invalidates a cached narrative.
                "organism":          shared.get("organism", "") or "",
            },
            output_dir=branch_dir,
        )
        narr.name = f"narrative_top{N}"
        _exec_with_cache(narr, cache)
        nodes_run.append(narr)
        if narr.status != "fail":
            artifacts["narrative_md"]    = narr.outputs.get("narrative_md","")
            artifacts["abstract_md"]     = narr.outputs.get("abstract_md","")
            artifacts["narrative_audit"] = narr.outputs.get("narrative_audit","")
        else:
            artifacts["narrative_error"] = narr.error

    if manifest:
        _record_branch(manifest, nodes_run)
    return artifacts


def _record_branch(manifest: Manifest, nodes: list[Node]) -> None:
    for n in nodes:
        manifest.add_step(n.to_manifest())
    manifest.write_atomic()


def run_all_variants(
    shared: dict[str, str],
    output_dir: Path,
    Ns: list[int] = (4, 6, 8),
    llm_endpoint: str = "http://127.0.0.1:11500",
    llm_model: str = "batiai/qwen3.6-27b:q6",
    parallel: bool = True,
    manifest: Optional[Manifest] = None,
    run_narrative: bool = True,
    cache: Optional[CheckpointCache] = None,
) -> dict[int, dict]:
    """Run all N-branches. shared['top_json'] is overridden per-N below.

    Args:
      shared: {motif_tsv, diffbind_csv, deseq_csv,
               tfs_summary_json, targets_summary_json, conditions_meta_json,
               topn_dir}   <- topn_dir contains top{N}.json files
    Returns: {N: artifact_dict}
    """
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    topn_dir = Path(shared["topn_dir"])

    def _make_branch_shared(N: int) -> dict[str, str]:
        s = dict(shared)
        s["top_json"] = str(topn_dir / f"top{N}.json")
        return s

    results: dict[int, dict] = {}
    if parallel:
        with ThreadPoolExecutor(max_workers=min(3, len(Ns))) as pool:
            futures = {
                pool.submit(
                    run_variant_branch,
                    N, _make_branch_shared(N), out_dir,
                    llm_endpoint, llm_model, manifest,
                    run_narrative, cache,
                ): N for N in Ns
            }
            for fut in as_completed(futures):
                N = futures[fut]
                try:
                    results[N] = fut.result()
                except Exception as e:
                    results[N] = {"N": str(N), "error": f"{type(e).__name__}: {e}"}
    else:
        for N in Ns:
            results[N] = run_variant_branch(
                N, _make_branch_shared(N), out_dir,
                llm_endpoint, llm_model, manifest,
                run_narrative, cache,
            )
    return results
