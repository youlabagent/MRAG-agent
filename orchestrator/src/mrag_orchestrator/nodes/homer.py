"""homer_mcp wrapper: HOMER motif discovery + annotation + Top-N TF filter."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/homer_mcp/src")
from ._base import Node, capture_mcp_tools


class HomerNode(Node):
    """Per-sample HOMER pipeline (find_motifs + concat + annotate_peaks).

    Inputs:  input_bed, sample_id
    Params:  genome (default 'hg38'), threads (4), timeout_s (5400), dry_run
    Outputs: motif_tsv (annotated peaks), motif_file (.motif),
             peak_dir  (peakAnalysis/{sample_id}/)
    """
    name = "homer"

    def run(self) -> None:
        from homer_mcp.tools import pipeline as _p
        tools = capture_mcp_tools(_p.register_pipeline_tools)
        out_root = Path(self.output_dir); out_root.mkdir(parents=True, exist_ok=True)
        sample_id = self.inputs["sample_id"]
        input_bed = self.inputs["input_bed"]
        res_str = tools["run_homer_pipeline"](
            input_bed=input_bed, sample_id=sample_id,
            output_root=str(out_root),
            genome=self.params.get("genome", "hg38"),
            threads=self.params.get("threads", 4),
            timeout_s=self.params.get("timeout_s", 5400),
            dry_run=self.params.get("dry_run", False),
            use_jaspar=self.params.get("use_jaspar", True),
            n_parallel_tasks=self.params.get("n_parallel_tasks", 1),
        )
        res = json.loads(res_str)
        if res.get("status") not in ("ok", "dry_run"):
            raise RuntimeError(f"run_homer_pipeline failed: {res}")
        self.outputs = {
            "motif_tsv":  str(out_root / f"{sample_id}.motif.tsv"),
            "motif_file": str(out_root / f"{sample_id}.motif"),
            "peak_dir":   str(out_root / "peakAnalysis" / sample_id),
        }

    def validate(self):
        tsv = self.outputs.get("motif_tsv","")
        if not tsv or not Path(tsv).exists():
            return [{"name":"motif_tsv_exists","result":"fail","msg":"no output"}]
        try:
            n = sum(1 for _ in open(tsv)) - 1
        except Exception:
            n = 0
        # Empty / near-empty output = hard fail (e.g. annotatePeaks OOM-killed
        # produces a 0-byte .motif.tsv). >100 rows = pass; 1..100 = warn.
        if n <= 0:
            result = "fail"
        elif n > 100:
            result = "pass"
        else:
            result = "warn"
        return [{"name":"motif_row_count","result":result,
                 "value": n, "msg":f"{n} annotated peaks"}]


class HomerFilterNode(Node):
    """Top-N TF filter step (fork point for variants).

    Inputs:  motif_tsv, tfs (list[str] or path to top{N}.json)
    Params:  only_promoter (True), case_insensitive (True), top_n (for naming)
    Outputs: homer_filter_tsv
    """
    name = "homer_filter"

    def run(self) -> None:
        from homer_mcp.tools import filter as _f
        tools = capture_mcp_tools(_f.register_filter_tools)
        out_dir = Path(self.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        top_n = self.params.get("top_n", 6)
        tfs = self.inputs["tfs"]
        if isinstance(tfs, str) and Path(tfs).exists():
            d = json.loads(Path(tfs).read_text())
            tfs = d.get("tfs") if isinstance(d, dict) else d
        if not tfs:
            raise ValueError("HomerFilterNode: empty TF list")
        out_tsv = out_dir / f"homer_filter_top{top_n}.tsv"
        res_str = tools["filter_target_tfs"](
            motif_tsv=self.inputs["motif_tsv"],
            tfs=list(tfs),
            output_tsv=str(out_tsv),
            only_promoter=self.params.get("only_promoter", True),
            case_insensitive=self.params.get("case_insensitive", True),
        )
        res = json.loads(res_str)
        if res.get("status") != "ok":
            raise RuntimeError(f"filter_target_tfs failed: {res}")
        self.outputs = {"homer_filter_tsv": str(out_tsv),
                        "n_rows_filtered": str(res.get("n_rows_filtered", 0)), "n_rows_input": str(res.get("n_rows_input", 0))}

    def validate(self):
        n = int(self.outputs.get("n_rows_filtered","0"))
        return [{"name":"promoter_rows","result":"pass" if n > 10 else "warn",
                 "value": n, "msg":f"{n} promoter rows kept"}]


class MinmaxBedNode(Node):
    """ ``1.minmax.py::min_max_GroupN`` mirror — per condition.

    Inputs:
        tsv_a, tsv_b: two replicate HOMER motif.tsv files (same condition).
        gtf_path:     Ensembl GTF for gene→strand lookup.
    Params:
        peak_id_prefix:   e.g. "ONECUT2_peak" or "Cyc_peak"
        condition_label:  identifier used in output filename
    Outputs:
        minmax_bed:  per-condition union BED (PeakID Chr Start End Strand)
    """
    name = "minmax_bed"

    def run(self) -> None:
        from homer_mcp.tools import filter as _f
        tools = capture_mcp_tools(_f.register_filter_tools)
        out_dir = Path(self.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        condition = self.params.get("condition_label", "cond")
        out_bed = out_dir / f"{condition}_peak_minmax.bed"
        res_str = tools["combine_replicates_minmax_bed"](
            tsv_a=self.inputs["tsv_a"],
            tsv_b=self.inputs["tsv_b"],
            output_bed=str(out_bed),
            gtf_path=self.inputs["gtf_path"],
            peak_id_prefix=self.params.get("peak_id_prefix", f"{condition}_peak"),
        )
        res = json.loads(res_str)
        if res.get("status") != "ok":
            raise RuntimeError(f"combine_replicates_minmax_bed failed: {res}")
        self.outputs = {
            "minmax_bed":   str(out_bed),
            "rows_emitted": str(res.get("rows_emitted", 0)),
        }

    def validate(self):
        n = int(self.outputs.get("rows_emitted", "0"))
        return [{"name":"minmax_bed_rows","result":"pass" if n > 100 else "warn",
                 "value": n, "msg":f"{n} union peaks"}]


class FilterTargetTfsNode(Node):
    """ ``2.findmotif_homer.py::Step2_TargetTFs`` mirror.

    Inputs:  motif_tsv (HOMER annotatePeaks output on the minmax BED).
    Params:  target_tfs (optional override), condition_label (filename slug)
    Outputs: tfs_txt (per-condition .TFs.txt)
    """
    name = "filter_target_tfs"

    def run(self) -> None:
        from homer_mcp.tools import filter as _f
        tools = capture_mcp_tools(_f.register_filter_tools)
        out_dir = Path(self.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        condition = self.params.get("condition_label", "cond")
        out_txt = out_dir / f"{condition}_peak_minmax.TFs.txt"
        res_str = tools["filter_target_tfs_minmax"](
            motif_tsv=self.inputs["motif_tsv"],
            output_txt=str(out_txt),
            target_tfs=self.params.get("target_tfs"),
        )
        res = json.loads(res_str)
        if res.get("status") != "ok":
            raise RuntimeError(f"filter_target_tfs failed: {res}")
        self.outputs = {
            "tfs_txt":         str(out_txt),
            "n_matched_tfs":   str(res.get("n_matched_tfs", 0)),
            "n_rows_filtered": str(res.get("n_rows_filtered", 0)),
        }

    def validate(self):
        n = int(self.outputs.get("n_rows_filtered", "0"))
        m = int(self.outputs.get("n_matched_tfs", "0"))
        return [
            {"name":"matched_tfs",
             "result":"pass" if m > 0 else "warn",
             "value": m, "msg":f"{m} target TFs found in motif columns"},
            {"name":"promoter_rows",
             "result":"pass" if n > 100 else "warn",
             "value": n, "msg":f"{n} promoter rows kept"},
        ]


class JoinDiffbindRnaNode(Node):
    """ ``3.Diffbind_pval.py`` mirror.

    Joins TFs.txt with DiffBind nearest-peak Fold/p.value AND DESeq2
    log2FC/pvalue. Rows whose Gene Name is not in DESeq2 are dropped
    (inner-merge behavior).

    Inputs:  tfs_txt, diffbind_csv, deseq_csv
    Params:  condition_label
    Outputs: tfs_pval2_txt (.TFs.Pval2.txt) — input to all Plot R scripts.
    """
    name = "join_diffbind_rna"

    def run(self) -> None:
        sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/trn_mcp/src")
        from trn_mcp.tools import integrate as _i
        tools = capture_mcp_tools(_i.register_integrate_tools)
        out_dir = Path(self.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        condition = self.params.get("condition_label", "cond")
        out_txt = out_dir / f"{condition}_peak_minmax.TFs.Pval2.txt"
        res_str = tools["join_diffbind_and_rna"](
            tfs_txt=self.inputs["tfs_txt"],
            diffbind_csv=self.inputs["diffbind_csv"],
            deseq_csv=self.inputs["deseq_csv"],
            output_txt=str(out_txt),
        )
        res = json.loads(res_str)
        if res.get("status") != "ok":
            raise RuntimeError(f"join_diffbind_and_rna failed: {res}")
        self.outputs = {
            "tfs_pval2_txt":  str(out_txt),
            "n_rows_emitted": str(res.get("n_rows_emitted", 0)),
            "n_db_matched":   str(res.get("n_db_matched", 0)),
            "n_rna_matched":  str(res.get("n_rna_matched", 0)),
        }

    def validate(self):
        n = int(self.outputs.get("n_rows_emitted", "0"))
        return [{"name":"joined_rows","result":"pass" if n > 100 else "warn",
                 "value": n, "msg":f"{n} TFs.Pval2 rows"}]


class HomerKnownMotifsNode(Node):
    """Run HOMER findMotifsGenome known-motif enrichment per condition on the
    min-max-scaled consensus peak BEDs, mirroring Step2/2.findmotif_homer.py
    (`findMotifsGenome.pl <bed> hg38 <out> -mask`). De-novo discovery is skipped
    (`-nomotif`) because the report only displays knownResults, which HOMER
    computes identically with or without de-novo. Produces, per arm, a
    knownResults.txt + knownResults/*.logo.svg directory that RenderNode
    Section 8 consumes — so the section is a natural full-run output rather than
    a precomputed artifact.

    Inputs : arm_a_bed, arm_b_bed, arm_a_label, arm_b_label
    Outputs: homer_known_a_dir, homer_known_b_dir, homer_known_a_label,
             homer_known_b_label
    """
    name = "homer_known_motifs"

    def run(self) -> None:
        import concurrent.futures as _cf
        from homer_mcp.tools import pipeline as _p
        tools = capture_mcp_tools(_p.register_pipeline_tools)
        out_root = Path(self.output_dir); out_root.mkdir(parents=True, exist_ok=True)
        genome    = self.params.get("genome", "hg38")
        threads   = self.params.get("threads", 16)
        timeout_s = self.params.get("timeout_s", 3600)

        arms = [("a", self.inputs["arm_a_bed"], self.inputs.get("arm_a_label", "arm_a")),
                ("b", self.inputs["arm_b_bed"], self.inputs.get("arm_b_label", "arm_b"))]

        def _run(arm):
            key, bed, label = arm
            outdir = out_root / label
            known = outdir / "knownResults.txt"
            # idempotent: reuse an existing valid knownResults (fast resume)
            if known.exists() and sum(1 for _ in open(known)) > 5:
                return key, str(outdir), {"status": "ok", "cached": True}
            res = json.loads(tools["find_motifs"](
                input_bed=bed, output_dir=str(outdir), genome=genome,
                mask_repeats=True, threads=threads,
                extra_args=["-nomotif"], timeout_s=timeout_s))
            return key, str(outdir), res

        with _cf.ThreadPoolExecutor(max_workers=2) as ex:
            results = list(ex.map(_run, arms))

        self.outputs = {}
        for key, outdir, res in results:
            if res.get("status") != "ok":
                raise RuntimeError(f"find_motifs ({key}) failed: {res}")
            self.outputs[f"homer_known_{key}_dir"] = outdir
        self.outputs["homer_known_a_label"] = arms[0][2]
        self.outputs["homer_known_b_label"] = arms[1][2]

    def validate(self):
        gates = []
        for key in ("a", "b"):
            d = self.outputs.get(f"homer_known_{key}_dir", "")
            kt = Path(d) / "knownResults.txt" if d else None
            ok = bool(kt and kt.exists() and sum(1 for _ in open(kt)) > 5)
            gates.append({"name": f"known_{key}_results",
                          "result": "pass" if ok else "fail",
                          "msg": str(kt) if kt else "no dir"})
        return gates
