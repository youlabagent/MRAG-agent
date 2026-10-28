"""DESeq2 differential expression wrap.

This is steps 4-5 of the RNA_mcp pipeline (after STAR + RSEM):
step4_load_counts (RSEM -> count matrix, FASTQ input only) then step5_deseq2
(R DESeq2 by default, pydeseq2 with engine="python"). Produces
DESeq2.gene.csv consumed by combined_mcp + TRN.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/RNA_mcp/src")
from ._base import Node, capture_mcp_tools


class DESeq2Node(Node):
    """Inputs:  project_dir, mapfile (sample -> condition assignment CSV)
    Params:    threads (4), ref_genome ('GRCh38'),
               gene_annotation (Gene.ID -> Gene.Name TSV used to turn the
               step5 DESeq2_results.csv gene_id schema into the GENE schema
               combined_mcp expects; comes from the organism profile, NOT a
               human default -- see _gene_annotation_path),
               organism ("human"|"mouse", used only to resolve the map when
               gene_annotation is not given),
               min_annotation_coverage (0.50: below this the step FAILS rather
               than writing bare Ensembl accessions into the GENE column)
    Outputs:   deseq_csv (DESeq2.gene.csv)
    """
    name = "deseq2"

    def run(self) -> None:
        from rna_mcp.tools import pipeline as _p
        from rna_mcp.tools.executor import LocalExecutor

        project_dir = self.inputs["project_dir"]
        executor = LocalExecutor(work_dir=self.params.get("work_dir", project_dir))
        tools = capture_mcp_tools(lambda mcp: _p.register_pipeline_tools(mcp, executor))

        # step5_deseq2 needs txi.rsem.counts.txt -> first call step4_load_counts
        # to aggregate per-sample RSEM .genes.results into the count matrix.
        quant_source = self.params.get("quant_source", "counts")
        # When an external matrix (counts_file) is supplied we skip step4
        # (RSEM aggregation): the matrix IS the count/TPM table already.
        if not self.inputs.get("counts_file"):
            from pathlib import Path as _Path
            # step4_load_counts writes BOTH txi.rsem.counts.txt and
            # txi.rsem.abundance.TPM.txt; require the matrix matching quant_source.
            matrix_name = ("txi.rsem.abundance.TPM.txt"
                           if quant_source == "tpm" else "txi.rsem.counts.txt")
            if not (_Path(project_dir) / matrix_name).exists():
                tools["step4_load_counts"](
                    project_dir=project_dir,
                    mapfile=self.inputs.get("mapfile", ""),
                )
        kwargs = {
            "project_dir": project_dir,
            "mapfile":     self.inputs.get("mapfile", ""),
        }
        # reference arm (control condition for log2FC denominator)
        ref = self.params.get("reference") or self.inputs.get("reference") or ""
        if ref: kwargs["reference"] = ref
        kwargs["num_threads"] = self.params.get("threads", 4)
        kwargs["engine"] = self.params.get("engine", "r")
        kwargs["quant_source"] = quant_source
        # explicit external matrix (processed TPM/counts) overrides default
        if self.inputs.get("counts_file"):
            kwargs["counts_file"] = self.inputs["counts_file"]
        out = tools["step5_deseq2"](**kwargs)
        pd_dir = Path(project_dir)
        deseq = pd_dir / "DESeq2.gene.csv"
        if not deseq.exists():
            # try common alternative locations
            for cand in [pd_dir/"STEP2_Diff/DESeq2.gene.csv",
                         pd_dir/"deseq2/DESeq2.gene.csv",
                         pd_dir/"DE/DESeq2.gene.csv"]:
                if cand.exists():
                    deseq = cand; break
        # Step5_deseq2 writes DESeq2_results.csv (gene_id schema, no GENE column).
        # combined_mcp.combine_pvalues expects ID+GENE+baseMean+log2FoldChange+...
        # If only DESeq2_results.csv exists, convert: ENSG -> Gene symbol via
        # the organism's gene_annotation.tsv (params.gene_annotation).
        # Hard-coding the human table here is what left a mouse run with bare
        # ENSMUSG ids in every downstream table while the correct mouse map sat
        # unreachable in Reference/RNA_mm10/ (2026-09-23).
        if not deseq.exists():
            results_csv = pd_dir / "DESeq2_results.csv"
            if results_csv.exists():
                import pandas as pd_lib
                gene_map_path = Path(self._gene_annotation_path())
                if not gene_map_path.exists():
                    raise RuntimeError(
                        f"gene-id->symbol map missing at {gene_map_path}. "
                        f"Without it every gene stays a bare Ensembl id and the "
                        f"ATAC/RNA join downstream silently returns nothing.")
                gmap = pd_lib.read_csv(gene_map_path, sep="\t")
                gmap.columns = [c.replace(".", "_") for c in gmap.columns]   # Gene.ID -> Gene_ID
                gmap = gmap.rename(columns={"Gene_ID": "ID", "Gene_Name": "GENE"})
                res = pd_lib.read_csv(results_csv)
                res = res.rename(columns={"gene_id": "ID"})
                merged = res.merge(gmap[["ID","GENE"]], on="ID", how="left")
                # --- coverage: does this map actually cover THESE ids? ------
                # An uncovered id used to fall through to `fillna(bare ENSG/
                # ENSMUSG)`, which produced a full-length, plausible-looking
                # DESeq2.gene.csv whose GENE column was accessions. Downstream
                # that joins against nothing and the report comes out empty
                # rather than failing. Two rescues, then a hard stop.
                cov = self._coverage(merged)
                if cov["frac"] < 0.90:
                    # Rescue 1: version suffixes, stripped on BOTH sides.
                    # ENSG00000141510 and ENSG00000141510.16 are the same gene
                    # whichever side carries the suffix, and both maps on disk
                    # are VERSIONED -- so stripping only the counts side (as
                    # this did until 2026-09-23) was dead code for them, and a
                    # version-less counts table scored 0% and hit the hard
                    # floor with an error that blamed the organism.
                    _res_bare = res["ID"].astype(str).str.split(".").str[0]
                    _map_bare = gmap["ID"].astype(str).str.split(".").str[0]
                    if (_res_bare.ne(res["ID"].astype(str)).any()
                            or _map_bare.ne(gmap["ID"].astype(str)).any()):
                        _gm = gmap[["ID", "GENE"]].assign(_bare=_map_bare)
                        _gm = _gm.drop_duplicates(subset="_bare", keep="first")
                        alt = res.assign(_bare=_res_bare).merge(
                            _gm[["_bare", "GENE"]],
                            on="_bare", how="left").drop(columns=["_bare"])
                        if self._coverage(alt)["frac"] > cov["frac"]:
                            merged, cov = alt, self._coverage(alt)
                            print("[deseq2] gene map matched on version-stripped "
                                  "Ensembl ids (%.1f%% covered)"
                                  % (100 * cov["frac"]), flush=True)
                self._annotation_coverage = cov
                min_cov = float(self.params.get("min_annotation_coverage", 0.50))
                if cov["frac"] < min_cov:
                    raise RuntimeError(
                        "the gene-id -> symbol map %s covers only %d of %d ids "
                        "(%.1f%%) in %s -- below the %.0f%% floor. Unmapped ids "
                        "would be written to the GENE column as bare "
                        "accessions, and the ATAC/RNA join downstream matches "
                        "on SYMBOLS, so the run would finish with an empty "
                        "INTEGRATED.csv instead of failing here. Example "
                        "unmapped ids: %s. Example map keys: %s. Compare the "
                        "two: if they are the same kind of accession (both "
                        "ENSG, or both ENSMUSG) this is an id-FORM mismatch, "
                        "not a species mismatch -- version suffixes are "
                        "already stripped on both sides, so look for a "
                        "different id space (RefSeq/Entrez/symbols) or an "
                        "annotation from another GENCODE release. If they are "
                        "different species, the map is the wrong one for this "
                        "organism (organism=%r expects %s)."
                        % (gene_map_path, cov["n_mapped"], cov["n_total"],
                           100 * cov["frac"], results_csv, 100 * min_cov,
                           cov["unmapped_examples"],
                           [str(x) for x in gmap["ID"].head(3).tolist()],
                           self.params.get("organism", ""), gene_map_path))
                if cov["frac"] < 0.90:
                    print("[deseq2] WARNING: %d/%d gene ids (%.1f%%) have no "
                          "symbol in %s; they keep their bare accession. "
                          "Examples: %s"
                          % (cov["n_total"] - cov["n_mapped"], cov["n_total"],
                             100 * cov["frac"], gene_map_path,
                             cov["unmapped_examples"]), flush=True)
                merged["GENE"] = merged["GENE"].fillna(merged["ID"].str.split(".").str[0])
                # Reorder cols to the DESeq2.gene.csv schema
                cols = ["ID","GENE","baseMean","log2FoldChange","lfcSE","stat","pvalue","padj"]
                merged = merged[[c for c in cols if c in merged.columns]]
                deseq = pd_dir / "DESeq2.gene.csv"
                merged.to_csv(deseq, index=False)
        if not deseq.exists():
            raise RuntimeError(f"DESeq2 did not produce DESeq2.gene.csv in {pd_dir}")
        self.outputs = {"deseq_csv": str(deseq)}

    # ---------------------------------------------------------------- helpers
    def _gene_annotation_path(self) -> str:
        """The ENSEMBL->symbol map for THIS run's organism.

        `params["gene_annotation"]` is what the runner threads in from the
        organism profile. The hard-coded human path that used to be the
        fallback here is exactly what made a mouse run produce bare ENSMUSG
        ids while Reference/RNA_mm10/gene_annotation.tsv sat unreachable
        (2026-09-23), so the fallback now goes through the profile too and an
        unknown organism fails loudly rather than defaulting to human.
        """
        explicit = str(self.params.get("gene_annotation") or "").strip()
        if explicit:
            return explicit
        org = self.params.get("organism") or ""
        from ..organism import resolve as _resolve_org, OrganismError
        try:
            return _resolve_org(org or "human").gene_annotation_tsv
        except OrganismError as e:
            raise RuntimeError(
                "no gene_annotation map was given and organism=%r could not be "
                "resolved to one: %s" % (org, e)) from None

    @staticmethod
    def _coverage(merged) -> dict:
        """How much of the id column the symbol map actually answered."""
        n = int(len(merged))
        miss = merged["GENE"].isna()
        n_un = int(miss.sum())
        return {"n_total": n, "n_mapped": n - n_un, "n_unmapped": n_un,
                "frac": (1.0 if n == 0 else (n - n_un) / float(n)),
                "unmapped_examples": [str(x) for x in
                                      merged.loc[miss, "ID"].head(5).tolist()]}

    def validate(self):
        p = self.outputs.get("deseq_csv","")
        if not p or not Path(p).exists():
            return [{"name":"deseq_exists","result":"fail","msg":"no output"}]
        n = sum(1 for _ in open(p)) - 1
        out = [{"name":"deseq_row_count","result":"pass" if n > 5000 else "warn",
                "value":n,"msg":f"{n} genes"}]
        cov = getattr(self, "_annotation_coverage", None)
        if cov:
            out.append({"name": "gene_symbol_coverage",
                        "result": "pass" if cov["frac"] >= 0.90 else "warn",
                        "value": round(cov["frac"], 4),
                        "msg": "%d/%d gene ids carry a symbol from %s"
                               % (cov["n_mapped"], cov["n_total"],
                                  self._gene_annotation_path())})
        return out
