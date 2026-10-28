"""RNA_mcp wrapper: RNA-seq preprocessing pipeline (steps 1-3).

Composite tool: run_full_pipeline(project_dir, ref_dir, genome, num_threads)
Pipeline: trim_galore -> STAR -> RSEM.
Steps 4-5 (count matrix -> DESeq2) need a mapfile; DESeq2Node (deseq2.py)
runs them.

Per sample outputs (under project_dir/):
  - trim_galore/<sample>_{1,2}_val_{1,2}.fq.gz
  - STAR/<sample>Aligned.toTranscriptome.out.bam, Log.final.out, SJ.out.tab
  - RSEM/<sample>.genes.results, .isoforms.results
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/RNA_mcp/src")
from ._base import Node, capture_mcp_tools


class RNANode(Node):
    """Inputs:  project_dir
    Params:   ref_dir, genome (RSEM prefix, 'GRCh38' | 'GRCm38'), num_threads (0=auto)
    Outputs:  project_dir, per-sample bam + rsem results
    """
    name = "rna"

    def run(self) -> None:
        from rna_mcp.tools import pipeline as _p
        from rna_mcp.tools.executor import LocalExecutor

        project_dir = self.inputs["project_dir"]
        work_dir = self.params.get("work_dir", project_dir)
        executor = LocalExecutor(work_dir=work_dir)
        tools = capture_mcp_tools(lambda mcp: _p.register_pipeline_tools(mcp, executor))

        out = tools["run_full_pipeline"](
            project_dir=project_dir,
            ref_dir=self.params.get("ref_dir", ""),
            genome=self.params.get("genome", "GRCh38"),
            num_threads=self.params.get("num_threads", 0),
        )
        # MCP convention:
        #   <project_dir>/<sample>Aligned.toTranscriptome.out.bam
        #   <project_dir>/<sample>.genes.results
        #   <project_dir>/<sample>.isoforms.results
        pd = Path(project_dir)
        bams   = sorted(str(p) for p in pd.glob("*Aligned.toTranscriptome.out.bam"))
        genes  = sorted(str(p) for p in pd.glob("*.genes.results"))
        isos   = sorted(str(p) for p in pd.glob("*.isoforms.results"))

        summary_path = pd / "_rna_pipeline_summary.json"
        summary_path.write_text(json.dumps({
            "project_dir": str(pd),
            "n_bams": len(bams), "n_genes_results": len(genes),
            "n_isoforms_results": len(isos),
            "pipeline_log_tail": out[-2000:],
        }, indent=2))

        self.outputs = {
            "project_dir":   str(pd),
            "bams":          ";".join(bams),
            "genes_results": ";".join(genes),
            "iso_results":   ";".join(isos),
            "summary_json":  str(summary_path),
        }

    def validate(self):
        n_genes = len(self.outputs.get("genes_results","").split(";")) if self.outputs.get("genes_results") else 0
        n_bams  = len(self.outputs.get("bams","").split(";")) if self.outputs.get("bams") else 0
        return [
            {"name":"genes_results_count","result":"pass" if n_genes >= 2 else "fail",
             "value":n_genes, "msg":f"{n_genes} .genes.results files"},
            {"name":"bam_count","result":"pass" if n_bams >= 2 else "fail",
             "value":n_bams, "msg":f"{n_bams} STAR BAMs"},
        ]
