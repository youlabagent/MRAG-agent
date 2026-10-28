"""ATAC_mcp wrapper: ATAC-seq preprocessing pipeline (steps 0-4).

Composite tool: run_full_pipeline(project_dir, ref_dir, genome, species, num_threads)
The MCP scans `project_dir` for *_R1.fastq.gz files and processes each sample
(trim_galore -> bowtie2 -> filter -> macs2 -> annotation).

Outputs (per sample, under project_dir/<sample>/):
  - trim_galore/<sample>_R{1,2}_val_{1,2}.fq.gz
  - alignments/<sample>.sorted.dedup.bam  + .bw  + _peaks.narrowPeak.filt
  - annotations / etc

Heavy: full pipeline ~hours on full FASTQs; minutes on subsampled (~100k reads).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/ATAC_mcp/src")
from ._base import Node, capture_mcp_tools


class ATACNode(Node):
    """Inputs:  project_dir (directory containing *_R1.fastq.gz, *_R2.fastq.gz)
    Params:   ref_dir, genome ('hg38'), species ('hs'), gtf_file (peak-annotation
              GTF; '' => the MCP's {ref_dir}hg38.refGene.gtf convention),
              num_threads (0=auto)
    Outputs:  project_dir, per_sample_bam, per_sample_bigwig, per_sample_peaks
              (collected from the project tree after run)
    """
    name = "atac"

    def run(self) -> None:
        from atac_mcp.tools import pipeline as _p
        from atac_mcp.tools.executor import LocalExecutor

        project_dir = self.inputs["project_dir"]
        work_dir = self.params.get("work_dir", project_dir)
        executor = LocalExecutor(work_dir=work_dir)
        tools = capture_mcp_tools(lambda mcp: _p.register_pipeline_tools(mcp, executor))

        out = tools["run_full_pipeline"](
            project_dir=project_dir,
            ref_dir=self.params.get("ref_dir", ""),
            genome=self.params.get("genome", "hg38"),
            species=self.params.get("species", "hs"),
            # Peak annotation (step4). The MCP's default is derived from the
            # human filename convention {ref_dir}hg38.refGene.gtf, which has no
            # mouse counterpart, so the profile passes the real path through.
            gtf_file=self.params.get("gtf_file", ""),
            num_threads=self.params.get("num_threads", 0),
        )
        # Collect outputs from the project tree.
        # MCP convention:
        #   alignments/<sample>.dedup.bam
        #   <project_dir>/<sample>.bw
        #   <project_dir>/<sample>_peaks.narrowPeak.filt   (and macs2_<sample>/<sample>_peaks.narrowPeak)
        #   <project_dir>/<sample>_annotations.txt
        pd = Path(project_dir)
        bams   = sorted(str(p) for p in pd.glob("alignments/*.dedup.bam"))
        bws    = sorted(str(p) for p in pd.glob("*.bw"))
        peaks  = sorted(str(p) for p in pd.glob("*_peaks.narrowPeak.filt"))
        annots = sorted(str(p) for p in pd.glob("*_annotations.txt"))

        # Persist a small summary
        summary_path = pd / "_atac_pipeline_summary.json"
        summary_path.write_text(json.dumps({
            "project_dir": str(pd),
            "n_bams": len(bams), "n_bws": len(bws),
            "n_peaks": len(peaks), "n_annots": len(annots),
            "pipeline_log_tail": out[-2000:],
        }, indent=2))

        self.outputs = {
            "project_dir":   str(pd),
            "bams":          ";".join(bams),
            "bigwigs":       ";".join(bws),
            "peaks":         ";".join(peaks),
            "annotations":   ";".join(annots),
            "summary_json":  str(summary_path),
        }

    def validate(self):
        bams_count = len(self.outputs.get("bams","").split(";")) if self.outputs.get("bams") else 0
        peaks_count = len(self.outputs.get("peaks","").split(";")) if self.outputs.get("peaks") else 0
        gates = [
            {"name":"bam_count","result":"pass" if bams_count >= 2 else "fail",
             "value":bams_count, "msg":f"{bams_count} sorted.dedup.bam files"},
            {"name":"peaks_present","result":"pass" if peaks_count >= 2 else "fail",
             "value":peaks_count, "msg":f"{peaks_count} narrowPeak files"},
        ]
        return gates
