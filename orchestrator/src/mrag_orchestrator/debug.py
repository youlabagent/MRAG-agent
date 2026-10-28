"""Auto-debug rule book + diagnose() entry point.

When a Node fails (status='fail'), the executor calls diagnose() with a
DiagnosisContext bundling all the signals (exception text, log tail, gate
results, etc.). Each rule inspects the context; the FIRST matching rule
produces a structured Diagnosis. If none match, an "unknown" Diagnosis is
returned with the raw log_tail attached.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class DiagnosisContext:
    step_name:   str
    error_text:  str = ""           # error message from Node.error
    stack_trace: str = ""
    log_tail:    str = ""           # last ~100 lines of stdout/stderr
    gate_results: list = field(default_factory=list)
    node_outputs: dict = field(default_factory=dict)
    node_inputs:  dict = field(default_factory=dict)


@dataclass
class Diagnosis:
    rule:               str
    likely_cause:       str
    suggested_actions:  list[str]
    severity:           str = "error"     # error|warn|info
    rule_kind:          str = "rule_based"


# ============================================================
# Rule book (registered in order; first match wins)
# ============================================================

_RULES: list[Callable[[DiagnosisContext], Optional[Diagnosis]]] = []


def register(fn):
    _RULES.append(fn)
    return fn


@register
def ollama_endpoint_down(ctx: DiagnosisContext) -> Optional[Diagnosis]:
    s_err = (ctx.error_text + ctx.log_tail + ctx.stack_trace).lower()
    # Also search inputs/params for endpoint URL (LLM endpoint config)
    s_meta = " ".join(str(v) for v in (list(ctx.node_inputs.values()) +
                                        list(ctx.node_outputs.values()))).lower()
    name = ctx.step_name.lower()
    is_conn_refused = ("connection refused" in s_err or "connectionrefusederror" in s_err
                       or "[errno 111]" in s_err)
    is_llm_step = ("narrative" in name or "abstract" in name)
    has_ollama_hint = (":11500" in s_err or "ollama" in s_err
                       or ":11500" in s_meta or "ollama" in s_meta
                       or "11500" in s_meta or is_llm_step)
    if is_conn_refused and has_ollama_hint:
        return Diagnosis(
            rule="ollama_endpoint_down",
            likely_cause="Ollama server is not listening on 127.0.0.1:11500.",
            suggested_actions=[
                "bash /BIO1/MRAG/scripts/start_glm_ollama.sh",
                "curl -sS http://127.0.0.1:11500/api/tags  # verify reachable",
                "Check log: tail /BIO1/MRAG/logs/glm_ollama.log",
            ],
        )
    return None


@register
def ollama_model_unloaded(ctx: DiagnosisContext) -> Optional[Diagnosis]:
    s = (ctx.error_text + ctx.log_tail).lower()
    if "model not found" in s or "manifest" in s and "no such file" in s:
        return Diagnosis(
            rule="ollama_model_unloaded",
            likely_cause="Requested Ollama model tag not present locally.",
            suggested_actions=[
                "env OLLAMA_HOST=127.0.0.1:11500 ollama list",
                "env OLLAMA_HOST=127.0.0.1:11500 ollama pull <model_tag>",
            ],
        )
    return None


@register
def diffbind_zero_samples(ctx: DiagnosisContext) -> Optional[Diagnosis]:
    s = ctx.error_text + ctx.log_tail
    if ("0 samples per group" in s or "no samples in" in s.lower()
        or "members per group" in s):
        return Diagnosis(
            rule="diffbind_zero_samples",
            likely_cause="DiffBind contrast has < min_members per group "
                          "(usually means Condition column miscoded).",
            suggested_actions=[
                "Inspect the DiffBind sample sheet's Condition column "
                "(it should have at least 2 distinct values, each with ≥2 replicates).",
                "Verify SampleID / Condition / Replicate mapping in prepare_diffbind_sheet inputs.",
            ],
        )
    return None


@register
def homer_genome_missing(ctx: DiagnosisContext) -> Optional[Diagnosis]:
    s = ctx.error_text + ctx.log_tail
    if "could not find genome" in s.lower() or "no genome data" in s.lower():
        return Diagnosis(
            rule="homer_genome_missing",
            likely_cause="HOMER hg38 genome data not installed at the expected location.",
            suggested_actions=[
                "Verify HOMER_HOME points to a homer install with hg38 data:",
                "  ls /BIO1/MRAG/Agent/bin/tools/homer/data/genomes/hg38",
                "If missing: perl /BIO1/MRAG/Agent/bin/tools/homer/configureHomer.pl -install hg38",
            ],
        )
    return None


@register
def paperqa_qdrant_locked(ctx: DiagnosisContext) -> Optional[Diagnosis]:
    s = ctx.error_text + ctx.log_tail
    if "qdrant" in s.lower() and ("lock" in s.lower() or "already in use" in s.lower()):
        return Diagnosis(
            rule="paperqa_qdrant_locked",
            likely_cause="Qdrant index is locked by another process.",
            suggested_actions=[
                "Find the holder: pgrep -af 'retrieve_cli.py'",
                "Or call paperqa_mcp.release_qdrant_lock()",
            ],
        )
    return None


@register
def paperqa_empty_index(ctx: DiagnosisContext) -> Optional[Diagnosis]:
    for g in ctx.gate_results:
        if g.get("name") == "passage_count" and g.get("value", 1) == 0:
            return Diagnosis(
                rule="paperqa_empty_index",
                likely_cause="paperqa returned 0 passages for the query set.",
                suggested_actions=[
                    "Check paperqa index status: paperqa_mcp.index_status()",
                    "Verify the Qdrant collection has expected docs.",
                    "Consider broadening the query template or adding new papers via add_papers().",
                ],
                severity="warn",
            )
    return None


@register
def ag_quota_exceeded(ctx: DiagnosisContext) -> Optional[Diagnosis]:
    s = ctx.error_text + ctx.log_tail
    if "429" in s or "quota" in s.lower() or "rate limit" in s.lower():
        return Diagnosis(
            rule="ag_quota_exceeded",
            likely_cause="AlphaGenome API quota/rate limit hit.",
            suggested_actions=[
                "Reduce items count or use_cache=True.",
                "Wait and retry; or split the batch.",
                "Verify API key: env ALPHAGENOME_API_KEY",
            ],
        )
    return None


@register
def rose_zero_superenhancers(ctx: DiagnosisContext) -> Optional[Diagnosis]:
    s = ctx.error_text + ctx.log_tail
    if ("0 super-enhancers" in s or "returned 0 super-enhancers" in s
            or ("ROSE" in s and "no super" in s.lower())):
        return Diagnosis(
            rule="rose_zero_superenhancers",
            likely_cause=("ROSE found no super-enhancers: the merged treatment BAM index (.bai) "
                          "is missing/stale, so ROSE_bamToGFF silently reads zero coverage, "
                          "or the treatment-specific peak BED is empty."),
            suggested_actions=[
                "Check <out>/se_rose/<treat>_merged.sorted.bam.bai exists and is newer than the BAM.",
                "Check <out>/peakheatmap/<treat>_specific.bed is non-empty (needs 2 replicate narrowPeaks per arm).",
                "Re-run with se_rose cache cleared: rm -r <out>/se_rose/rose.",
            ],
        )
    return None


@register
def ag_invalid_tissue(ctx: DiagnosisContext) -> Optional[Diagnosis]:
    s = ctx.error_text + ctx.log_tail
    if "tissue" in s.lower() and ("not found" in s.lower() or "invalid" in s.lower()
                                   or "no match" in s.lower()):
        return Diagnosis(
            rule="ag_invalid_tissue",
            likely_cause="AlphaGenome tissue/cell-line ontology code didn't resolve.",
            suggested_actions=[
                "Try a known EFO code: 'EFO:0002074' (PC-3 ATAC).",
                "RNA/CHIP-only queries can use UBERON codes (e.g. UBERON:0002367 prostate gland).",
                "list_tissues() to enumerate supported terms.",
            ],
        )
    return None


@register
def fastq_pair_mismatch(ctx: DiagnosisContext) -> Optional[Diagnosis]:
    s = ctx.error_text + ctx.log_tail
    if "different number of lines" in s.lower() or "pair length mismatch" in s.lower():
        return Diagnosis(
            rule="fastq_pair_mismatch",
            likely_cause="R1 and R2 FASTQ have different read counts (truncated or mismatched).",
            suggested_actions=[
                "Verify R1/R2 pair: zcat <file_R1.fastq.gz> | wc -l   vs   <file_R2.fastq.gz>",
                "Re-download or re-subsample the affected sample.",
            ],
        )
    return None


@register
def disk_full(ctx: DiagnosisContext) -> Optional[Diagnosis]:
    s = ctx.error_text + ctx.log_tail
    if ("no space left" in s.lower() or "[errno 28]" in s.lower()
        or "disk quota exceeded" in s.lower()):
        return Diagnosis(
            rule="disk_full",
            likely_cause="Out of disk space on the output volume.",
            suggested_actions=[
                "df -h /BIO1",
                "Clean old report runs: rm -rf /BIO1/MRAG/Agent/reports/<old>/",
            ],
        )
    return None


@register
def post_flight_gate_failed(ctx: DiagnosisContext) -> Optional[Diagnosis]:
    if not ctx.gate_results:
        return None
    fails = [g for g in ctx.gate_results if g.get("result") == "fail"]
    if not fails:
        return None
    names = [g.get("name") for g in fails]
    return Diagnosis(
        rule="post_flight_gate_failed",
        likely_cause=f"Post-flight gate(s) reported FAIL: {names}",
        suggested_actions=[
            f"Inspect gate detail: {fails[0].get('msg','')}",
            f"Re-check inputs to step '{ctx.step_name}'.",
        ],
    )


# ============================================================
# Top-level entry
# ============================================================

def diagnose(ctx: DiagnosisContext) -> Diagnosis:
    """Run all rules in order, return first match. Else return 'unknown'."""
    for rule_fn in _RULES:
        try:
            d = rule_fn(ctx)
            if d is not None:
                return d
        except Exception:
            continue
    return Diagnosis(
        rule="unknown",
        likely_cause="No rule matched. See log_tail.",
        suggested_actions=[
            "Inspect step's log output.",
            "Add a rule in debug.py if this is a recurring failure mode.",
        ],
        severity="warn",
    )
