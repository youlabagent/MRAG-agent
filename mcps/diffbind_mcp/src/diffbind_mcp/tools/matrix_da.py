"""diffbind-mcp: edgeR-voom differential accessibility for processed ATAC peak-count matrices (MCP step 2a,
2026-10-12).

For TCGA-style input (a peak x sample count matrix + group labels) this replaces DiffBind: edgeR-voom
differential accessibility -> group-enriched treatment / control peak BEDs (the DAStk inputs; top floor_n per
direction when few pass FDR) and a DiffBind-format per-peak table (<out_dir>/DiffBind_out.csv: Fold / p.value /
FDR).  The implementation is the orchestrator's adapter (mrag_orchestrator/adapters/flex_adapters.py,
atac_matrix_diff.R), imported when the tool is called."""
from __future__ import annotations

import importlib


ADAPTER_MODULE = "mrag_orchestrator.adapters.flex_adapters"


def register_matrix_da_tools(mcp, impl=None):
    def _fa():
        return impl if impl is not None else importlib.import_module(ADAPTER_MODULE)

    @mcp.tool()
    def voom_differential_accessibility(matrix: str, group_tsv: str, out_dir: str, treat: str, ref: str,
                                        floor_n: int = 5000) -> dict[str, str]:
        """edgeR-voom differential accessibility of an ATAC peak-count matrix -> per-arm peak BEDs.

        Args:
            matrix: peak x sample count matrix (seqnames, start, end, [name], samples ...).
            group_tsv: sample<TAB>group.  out_dir: output directory (treatment_diff.bed, control_diff.bed,
                DiffBind_out.csv).  treat, ref: group labels.  floor_n: peaks kept per direction when few pass FDR.

        Returns {treatment_bed, control_bed}; raises RuntimeError when the R script fails or writes no BED."""
        tb, cb = _fa().run_atac_matrix_adapter(matrix, group_tsv, out_dir, treat=treat, ref=ref, floor_n=floor_n)
        return {"treatment_bed": tb, "control_bed": cb}
