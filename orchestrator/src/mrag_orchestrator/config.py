"""MRAG configuration schemas.

These pydantic models are the single source of truth for:
  - what a 'sample sheet' looks like (output of `mrag detect`, input of `mrag confirm`)
  - what a 'run config' looks like (output of `mrag confirm`, input of `mrag run`)
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================
# Sample sheet (detector output)
# ============================================================

Confidence = Literal["high", "medium", "low"]
Modality   = Literal["ATAC", "RNA", "unknown"]


class SampleConfidence(BaseModel):
    modality:  Confidence = "low"
    arm:       Confidence = "low"
    replicate: Confidence = "low"


class Sample(BaseModel):
    name:        str
    fastq_r1:    str
    fastq_r2:    Optional[str] = None         # None => single-end
    modality:    Modality      = "unknown"
    arm:         str           = ""           # condition label, e.g. "ONECUT2-OE"
    replicate:   int           = 1
    read_length: Optional[int] = None
    paired:      bool          = True
    # ENA/SRA scientific_name for this run ("Homo sapiens", "Mus musculus").
    # Empty when the sample did not come from a metadata-bearing download; the
    # organism is then decided further down the chain (organism.decide()).
    organism:    Optional[str] = None
    confidence:  SampleConfidence = Field(default_factory=SampleConfidence)


class Condition(BaseModel):
    label:    str                       # "ONECUT2-OE"
    n_reps:   int
    modality_counts: dict[str, int]     # {"ATAC": 2, "RNA": 2}
    perturbation: Optional[str] = None  # filled by user at confirm step
    concept:      Optional[str] = None  # biological role description
    rationale:    Optional[str] = None  # why this arm


class DetectWarning(BaseModel):
    severity: Literal["low", "medium", "high"]
    msg: str


class SampleSheet(BaseModel):
    """Output of `mrag detect`, input of `mrag confirm`."""
    schema_version: str = "1.0"
    detected_at:    str
    fastq_dir:      Optional[str] = None
    samples:        list[Sample]
    conditions:     list[Condition]
    warnings:       list[DetectWarning] = []
    needs_user_action: bool = False
    summary_human:  str = ""
    # Organism carried from the SRA/ENA metadata of the downloaded study (or
    # set by the user). `organism` is the canonical key ("human"/"mouse"),
    # `organism_detail` records the evidence so the UI can show it.
    organism:        Optional[str] = None
    organism_source: str = ""
    organism_detail: str = ""
    # Hook fields populated by LLM-assist mode (optional)
    llm_assist_used: bool = False
    llm_assist_model: Optional[str] = None
    # Flexible-input: processed (non-FASTQ) artifacts detected in the input
    # dir, mapped onto precomputed keys. atac_peaks_by_arm (dict) is resolved
    # to treatment/control against conditions in sample_sheet_to_run_config.
    precomputed_suggestion: dict = {}


# ============================================================
# Run config (confirmed sheet => full pipeline config)
# ============================================================

class LLMConfig(BaseModel):
    endpoint: str = "http://127.0.0.1:11500"
    model:    str = "batiai/qwen3.6-27b:q6"


class PaperQAConfig(BaseModel):
    retriever_script: str = "/BIO1/MRAG/Agent/libs/paperqa/scripts/retrieve_cli.py"
    venv_python:      str = "/BIO1/MRAG/venvs/paper_qa/bin/python3"


class PipelineConfig(BaseModel):
    # Default: canonical Top 10 (2026-07). Set to [4, 6, 8] explicitly to do the
    # fanout (~3x narrative LLM calls + cross-variant synthesis). Single
    # variant skips the fanout and runs ~3x faster.
    topn_variants: list[int] = [10]
    parallel:      bool = True
    # Phase B the motif-scan step chain (per-rep HOMER scan + MinmaxBed + 10-TF filter
    # + DiffBind+RNA join). Produces TFs.Pval2.txt files that are decorative
    # for the canonical report path (R plots use INTEGRATED_ranked.csv
    # directly). ~40 min on full FASTQ. Default ON; set True to skip.
    skip_motif_scan: bool = False
    # GTF path for the the motif-scan step chain (only used when skip_motif_scan=False).
    # Empty => resolved from the run's organism profile (organism.py). For a
    # human run that is the same Homo_sapiens.GRCh38.104.gtf this used to hard-code.
    gtf_path: str = ""
    # HOMER / ROSE / SE genome key. Empty => the organism profile's genome.
    # This field was READ at five places in runner.py but never declared, so
    # pydantic dropped a run.toml value silently and the escape hatch was dead
    # (2026-09-23). Declared now: a value set here is honoured and validated.
    homer_genome: str = ""
    # Super-enhancer layer (2026-09-10): ROSE on the treatment-specific peaks
    # (needs per-replicate ATAC BAMs -> skipped for processed-matrix input),
    # SE -> TF motif presence, SE -> TF -> Target TRN (report Section 10b).
    # ~15 min extra on the OC2 dataset (samtools merge + ROSE_main -n 16).
    se_analysis: bool = True
    se_top_n: int = 10
    # Optional DAStk differential MD-score table (Gene, differential_MDscore,
    # ATAC_pvalue, motif_id) used to fill ATAC values for TFs absent from
    # INTEGRATED.csv (e.g. FOX motifs whose shipped HOCOMOCO bed was empty).
    se_atac_fallback_tsv: str = ""
    # Add the Section-8 enriched motif TFs found in the SE layer (gene symbols only;
    # composite motifs such as Fox:Ebox stay SE-TRN-only nodes) to the curated
    # target panel, so Section-10 target selection, the TRN and "Regulated by TFs"
    # cover them (top-10 TFs + top-10 enriched motifs -> re-select target genes).
    se_extend_curated_panel: bool = True


# ============================================================
# Precomputed-artifact overrides (Step 6a)
# ============================================================
# When the user has already run ATAC/RNA preprocessing externally (or via
#  scripts), they can hand the orchestrator the artifact paths directly
# and skip those node executions.

class PrecomputedArtifacts(BaseModel):
    """All optional. If set, the corresponding pipeline node runs in 'provided'
    mode (validates files exist + skips computation)."""
    integrated_csv:     Optional[str] = None     # combined output (skips combined node)
    atac_md_csv:        Optional[str] = None     # DAStk differential MD-score file
    deseq_csv:          Optional[str] = None     # RNA DESeq2.gene.csv
    diffbind_out_csv:   Optional[str] = None     # DiffBind_out.csv
    motif_tsv:          Optional[str] = None     # HOMER canonical sample annotated motif
    # Partial-stage RNA input: raw expression matrix (genes x samples) that
    # skips RNA FASTQ preprocessing and enters DESeq2 directly.
    # rna_quant_source "counts" -> mirror 1.DESeq2.R, "tpm" -> 1.DESeq2_TPM.R.
    rna_counts_matrix:  Optional[str] = None
    rna_quant_source:   str = "counts"
    # Partial-stage ATAC input: per-arm peaks BED/narrowPeak that skips ATAC
    # FASTQ preprocessing and enters DAStk MD-score directly.
    atac_peaks_treatment: list = []
    atac_peaks_control:   list = []
    # Processed-matrix stage (flex adapters, 2026-07-07): log2/normalized
    # matrices that need a differential step before combined. Groups supplied
    # as a sample<tab>group TSV; treat/ref = conditions[0]/[1] labels.
    #   RNA log2 matrix  -> limma group DE  -> deseq_csv
    #   ATAC count matrix-> voom diff       -> treatment/control peak BEDs -> DAStk
    rna_log2_matrix:    Optional[str] = None
    rna_group_tsv:      Optional[str] = None
    atac_count_matrix:  Optional[str] = None
    atac_group_tsv:     Optional[str] = None
    # Optional aux artifacts used by render (passthrough to HTML)
    tfs_summary_json:       Optional[str] = None
    targets_summary_json:   Optional[str] = None
    conditions_meta_json:   Optional[str] = None
    tf_table_csv:           Optional[str] = None
    # ag_table_csv is gone with the single-base AlphaGenome step (the 2026-10-18 Section 13 rebuild): the node,
    # its render parameter and the ctx key it fed were all removed, and nothing reads this field.
    tf_rank_json:           Optional[str] = None
    trn_json:               Optional[str] = None
    per_tf_profile_json:    Optional[str] = None
    volcano_atac_json:      Optional[str] = None
    volcano_rna_json:       Optional[str] = None
    locus_manifest_json:    Optional[str] = None    # JSON list of {tf, gene, region, png_path}
    bigwig_arm_a_paths:     list = []        # e.g. ONECUT2 replicate bigwigs
    bigwig_arm_b_paths:     list = []        # e.g. cMyc replicate bigwigs
    bigwig_arm_a_label:     str  = "OC2"
    bigwig_arm_b_label:     str  = "cMyc"

    _FILE_FIELDS = (
        "integrated_csv", "atac_md_csv", "deseq_csv", "diffbind_out_csv",
        "motif_tsv", "rna_counts_matrix", "rna_log2_matrix", "rna_group_tsv",
        "atac_count_matrix", "atac_group_tsv",
    )

    def resolve_missing_paths(self, search_root=None) -> "PrecomputedArtifacts":
        """Robustness: if a precomputed file path is set but the file is missing,
        relocate it by basename under a data root (handles inputs that moved
        between sub-dirs, e.g. a group TSV that moved from ATAC/ to meta/).

        Search roots are auto-derived from the precomputed paths that DO exist
        (their parent and grandparent dirs, e.g. .../KIRP/ATAC -> .../KIRP), so
        no external root is required; an explicit ``search_root`` is added too.
        Raises a clear, actionable error if a set path cannot be recovered."""
        roots = []
        if search_root:
            roots.append(Path(search_root))
        for f in self._FILE_FIELDS:
            v = getattr(self, f, None)
            if isinstance(v, str) and v and Path(v).exists():
                pp = Path(v).resolve()
                roots += [pp.parent, pp.parent.parent]
        # de-dup, keep existing dirs
        seen, uroots = set(), []
        for r in roots:
            rs = str(r)
            if rs not in seen and r.exists():
                seen.add(rs); uroots.append(r)
        for f in self._FILE_FIELDS:
            v = getattr(self, f, None)
            if not isinstance(v, str) or not v or Path(v).exists():
                continue
            base = Path(v).name
            hit = None
            for r in uroots:
                hits = sorted(r.rglob(base))
                if hits:
                    hit = str(hits[0]); break
            if hit:
                setattr(self, f, hit)
            else:
                raise FileNotFoundError(
                    f"precomputed.{f}={v!r} does not exist and no file named "
                    f"{base!r} was found under {[str(r) for r in uroots]}. "
                    f"Re-run `mrag detect` to refresh the sample sheet paths."
                )
        return self


def load_run_config(path: str | Path) -> "RunConfig":
    """Load a TOML run config and return a RunConfig instance."""
    import tomllib
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return RunConfig(**data)


def save_run_config(cfg: "RunConfig", path: str | Path) -> None:
    """Serialize RunConfig to TOML."""
    try:
        import tomli_w
        with open(path, "wb") as f:
            tomli_w.dump(cfg.model_dump(exclude_none=True), f)
    except ImportError:
        # Fallback: JSON-as-TOML-ish (works for simple types, used for smoke only)
        Path(path).write_text(_to_toml_text(cfg.model_dump(exclude_none=True)))


def _to_toml_text(d, prefix=""):
    """Minimal TOML serializer (no nested tables of lists). Used as fallback."""
    lines = []
    # First the scalar / list keys (top-level + simple)
    for k, v in d.items():
        if isinstance(v, dict):
            continue   # handled below
        if isinstance(v, list) and v and isinstance(v[0], dict):
            for item in v:
                lines.append(f"[[{prefix}{k}]]" if not prefix else f"[[{prefix.rstrip('.')}.{k}]]")
                for ik, iv in item.items():
                    lines.append(f"{ik} = {_toml_value(iv)}")
            continue
        lines.append(f"{k} = {_toml_value(v)}")
    # Then nested dicts
    for k, v in d.items():
        if isinstance(v, dict):
            full = f"{prefix}{k}" if not prefix else f"{prefix.rstrip('.')}.{k}"
            lines.append(f"\n[{full}]")
            lines.append(_to_toml_text(v, ""))
    return "\n".join(lines) + "\n"


def _toml_value(v):
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if v is None:
        return '""'
    return f'"{v}"'


class RunConfig(BaseModel):
    """Top-level config for `mrag run`. TOML-serializable.

    Either samples (for full pipeline from FASTQ) or precomputed (to skip
    preprocessing) must give the downstream what it needs. Both can be set;
    precomputed entries override the corresponding nodes.
    """
    schema_version: str = "1.0"
    run_label:      str
    output_dir:     str
    samples:        list[Sample] = []
    conditions:     list[Condition] = []
    pipeline:       PipelineConfig    = Field(default_factory=PipelineConfig)
    llm:            LLMConfig         = Field(default_factory=LLMConfig)
    paperqa:        PaperQAConfig     = Field(default_factory=PaperQAConfig)
    precomputed:    PrecomputedArtifacts = Field(default_factory=PrecomputedArtifacts)
    canonical_N:    int = 10
    report_title:   Optional[str] = None
    conditions_str: Optional[str] = None
    # Species / genome build for the whole run. Any spelling organism.resolve()
    # knows: "mouse", "Mus musculus", "mm10", "human", "hg38", ... Unset =>
    # detected from the sample sheet's SRA metadata, then inferred from the
    # data, then defaulted to human with a visible note (organism.decide()).
    organism:       Optional[str] = None
    # Biological domain (collected from the user in chat/confirm so the LLM
    # narrative + Enrichr context + AlphaGenome tissue reflect the actual
    # dataset instead of the prostate/PC-3 default). All optional; when unset
    # the pipeline falls back to prostate/PC-3.
    cancer_type:    Optional[str] = None   # free text, e.g. "breast cancer"
    cell_line:      Optional[str] = None   # e.g. "MCF-7" or AlphaGenome key
    tissue:         Optional[str] = None   # free text or ontology CURIE (EFO:/UBERON:)
    # PaperQA corpora searched for target-gene evidence (2026-09-15). Corpus keys
    # or names, several allowed (e.g. ["prostate", "nec", "sclc"] or "PC, NEC, SCLC").
    # Empty -> derived from cancer_type. "pancancer" is always added.
    literature_domains: list[str] = []
    # When True, route Sections 10/11/14 through the evidence-driven curated
    # top-15 target pipeline (CuratedTargetsNode) instead of the auto-derived
    # per-TF top-12. Default ON (2026-07) => curated evidence pipeline is the
    # standard path; set to False in run.toml to fall back to auto-derived.
    curated_targets: bool = True
    # Evidence-grounding gate scope for CuratedTargetsNode (2026-09-25):
    #   "all"        GeneRIF line OR PaperQA snippet names the gene (node default)
    #   "literature" only PaperQA snippets count
    #   "domain"     one passage must name the gene AND be tied to the run's
    #                literature domain (evidence_gate.gate_scope docstring)
    # Unset (None) -> the node's own resolution (env MRAG_TARGET_EVIDENCE_SCOPE,
    # else "all"), i.e. byte-for-byte the old behaviour for every existing
    # run.toml. Top-level key, like literature_domains / curated_targets: it
    # must sit ABOVE the first [table] header of run.toml. Declared here
    # because pydantic silently drops an undeclared run.toml key (the
    # pipeline.homer_genome lesson of 2026-09-23).
    evidence_gate_scope: Optional[str] = None
    # Section 13 (SECTION13_CONTRACT.md): which model set LEADS the
    # section -- its default tab and the emphasised row of the pair-level table, nothing else;
    # all three sets are always printed.  "" -> the node's default rule (the consensus when both
    # arms ran and both single-model headline tests survive Holm).  It is declared, never "the
    # smallest p": OC2 leads with "both" (the default), PRAD declares "alphagenome" because Corgi
    # has no matching input condition on a bulk tumour cohort.  Top-level key, above the first
    # [table] header of run.toml.
    axis_headline: Optional[str] = None
    # Section 13: the AR cross-condition row (22Rv1 + VCaP H3K27ac on top of the enhancer end) is
    # a DECLARED PRAD-only EXTRA requirement, never a default.  Every
    # prostate-lineage run's AlphaGenome request carries those two tracks, so switching the row on
    # whenever the columns exist makes it the Holm headline of every prostate run -- including an
    # LNCaP/OC2 run, where the enhancer row is the one that leads.  Top-level key, above the first
    # [table] header of run.toml, like axis_headline.
    axis_ar_gate: bool = False

    @field_validator("axis_ar_gate", mode="before")
    @classmethod
    def _axis_ar_gate_bool(cls, v):
        """An unset key is False.  A string is accepted because run.toml is also written by hand
        and by the webui form, where a boolean arrives as text."""
        if v is None or (isinstance(v, str) and not v.strip()):
            return False
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("1", "true", "yes", "on"):
                return True
            if s in ("0", "false", "no", "off"):
                return False
            raise ValueError("axis_ar_gate must be a boolean, got %r" % (v,))
        return bool(v)

    @field_validator("axis_headline", mode="before")
    @classmethod
    def _axis_headline_known(cls, v):
        """Fail at load time on a typo: an unknown value would silently fall back to the
        default rule and lead the section with a model set nobody declared."""
        if v is None or str(v).strip() == "":
            return None
        s = str(v).strip().lower()
        if s not in ("both", "alphagenome", "corgi"):
            raise ValueError("axis_headline must be one of 'both' | 'alphagenome' | 'corgi', "
                             "got %r" % (v,))
        return s

    @field_validator("evidence_gate_scope", mode="before")
    @classmethod
    def _gate_scope_known(cls, v):
        """Fail at load time on a typo: evidence_gate.gate_scope() would
        silently fall back to "all" for an unknown value. Values mirror
        nodes/evidence_gate.SCOPES."""
        if v is None or str(v).strip() == "":
            return None
        s = str(v).strip().lower()
        if s not in ("all", "literature", "domain"):
            raise ValueError("evidence_gate_scope must be one of "
                             "'all' | 'literature' | 'domain', got %r" % (v,))
        return s

    @model_validator(mode="after")
    def _organism_resolvable(self):
        """Fail at load time, not three hours into the run.

        A typo in organism= or pipeline.homer_genome= used to be either a
        pydantic-dropped field (homer_genome) or a late crash. Resolve both
        here so `mrag run` refuses the config immediately with the list of
        accepted spellings."""
        from .organism import resolve as _resolve_org
        if self.organism:
            _resolve_org(self.organism)          # raises OrganismError on typo
        hg = getattr(self.pipeline, "homer_genome", "") or ""
        if hg:
            _resolve_org(hg)
        return self

    @field_validator("literature_domains", mode="before")
    @classmethod
    def _split_domains(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            import re as _re
            return [x.strip() for x in _re.split(r"[,;/+|]", v) if x.strip()]
        return [str(x).strip() for x in v if str(x).strip()]

    @model_validator(mode="after")
    def _canonical_in_variants(self):
        """canonical_N must be one of the computed top-N variants.

        combine only writes top{N}.json for N in topn_variants, so a
        canonical_N outside that list leaves top{canonical_N}.json missing ->
        CuratedTargetsNode raises -> the runner catches it and curated is
        silently disabled for the run (e.g. when the panel moved 6 -> 10 but
        the webui form still wrote canonical_N=6).
        Add it rather than fail: the value the user asked for is the intent.
        """
        ns = list(self.pipeline.topn_variants or [])
        if self.canonical_N not in ns:
            ns.append(self.canonical_N)
            self.pipeline.topn_variants = sorted(set(ns))
            print("[config] canonical_N=%d was missing from topn_variants; "
                  "now %s" % (self.canonical_N, self.pipeline.topn_variants),
                  flush=True)
        return self


# ---------------------------------------------------------------------------
# Tissue / cancer-type -> best AlphaGenome biosample key. Picks a line that
# matches the dataset's tissue and carries ATAC (or DNASE, used as an
# accessibility proxy by the validator). Keeps AG tissue-appropriate for any
# dataset instead of the historical hard-coded PC-3 (prostate). Fallback below.
# ---------------------------------------------------------------------------
AG_TISSUE_CELL_LINE = [
    (("prostate",),                                  "PC-3"),
    (("kidney", "renal", "rcc", "nephro", "kirp", "kirc"), "Caki2"),
    (("breast", "mammary", "brca"),                  "MCF-7"),
    (("liver", "hepat", "hcc", "lihc"),              "HepG2"),
    (("lung", "nsclc", "luad", "lusc", "pulmonary"), "A549"),
    (("colon", "colorectal", "crc", "rectal", "coad", "read"), "HCT116"),
    (("pancrea", "paad"),                            "Panc1"),
    (("leukemi", "myeloid", "cml", "aml", "blood", "hematolog"), "K562"),
]


def resolve_ag_cell_line(text):
    """Map free-text cancer_type/tissue to an AlphaGenome biosample key.

    Returns None when no keyword matches so the caller can fall back."""
    if not text:
        return None
    t = str(text).lower()
    for keys, line in AG_TISSUE_CELL_LINE:
        if any(k in t for k in keys):
            return line
    return None
