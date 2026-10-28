"""rna-mcp: limma group differential expression for processed (log2 / normalised) RNA matrices (MCP step 2a,
2026-10-12).

For TCGA-style input (a gene x sample log2 matrix + group labels) this replaces the DESeq2 step: limma group DE +
Entrez -> symbol -> DESeq2.gene.csv (ID, GENE, baseMean, log2FoldChange, lfcSE, stat, pvalue, padj), coefficient =
treat vs ref.  The implementation is the orchestrator's adapter (mrag_orchestrator/adapters/flex_adapters.py,
rna_log2_deseq.R), imported when the tool is called."""
from __future__ import annotations

import importlib

ADAPTER_MODULE = "mrag_orchestrator.adapters.flex_adapters"


def register_matrix_de_tools(mcp, impl=None):
    def _fa():
        return impl if impl is not None else importlib.import_module(ADAPTER_MODULE)

    @mcp.tool()
    def limma_group_de(matrix: str, group_tsv: str, out_csv: str, treat: str, ref: str) -> str:
        """limma group DE of a log2 / normalised RNA matrix -> DESeq2.gene.csv.

        Args:
            matrix: gene x sample matrix (TSV/CSV).  group_tsv: sample<TAB>group.
            out_csv: the DESeq2-format CSV to write.  treat, ref: the group labels (treat vs ref).

        Returns out_csv; raises RuntimeError when the R script fails or writes nothing."""
        return _fa().run_rna_log2_adapter(matrix, group_tsv, out_csv, treat=treat, ref=ref)
