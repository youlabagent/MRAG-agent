"""Organism / genome profile: the single source of truth for every genome
asset an MRAG run touches.

Before this module the pipeline had no organism concept at all. Every genome
argument was a literal default scattered across the runner, the node wrappers
and the MCP tools ("hg38", "GRCh38", "hs", the hg38 GTF, the hg38 gene
annotation, the human HOCOMOCO motif BEDs). A mouse run therefore did not
fail -- it finished, quietly, against the human genome.

One profile now answers every genome question, `resolve()` turns any spelling
of a species into that profile, and `decide()` records WHERE the answer came
from so the manifest can show it.

mm39 is refused on purpose: AlphaGenome's mouse tracks are built on mm10
(GRCm38), and mm39 coordinates are valid numbers that silently address the
wrong locus (Reference/README_mm10.md, 2026-09-03).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Optional

REFERENCE_ROOT = Path(os.environ.get(
    "MRAG_REFERENCE_ROOT", "/BIO1/MRAG/Agent/Reference"))
ROSE_DIR = Path(os.environ.get(
    "MRAG_ROSE_DIR", "/BIO1/MRAG/Agent/bin/tools/ROSE"))
ORTHOLOGY_ROOT = Path(os.environ.get(
    "MRAG_ORTHOLOGY_DIR", "/BIO1/MRAG/Agent/Reference/orthology"))
# Cistrome TF ChIP-seq GIGGLE assets (Section 9). run_giggle.sh hardcodes the
# same directory; the profile names the two files it opens per assembly.
CISTROME_ROOT = Path(os.environ.get(
    "MRAG_CISTROME_DIR", str(REFERENCE_ROOT / "cistrome")))


class OrganismError(ValueError):
    """A species could not be resolved to a supported genome build."""


# ---------------------------------------------------------------------------
# The profile
# ---------------------------------------------------------------------------

#: Filename/path fragments that identify the species an annotation belongs to.
#: Used to CROSS-CHECK an explicitly configured path against the resolved
#: organism -- a run.toml copied from another study is the usual way a human
#: GTF ends up in a mouse run (2026-09-23).
PATH_SPECIES_TOKENS = {
    "human": ("hg38", "hg19", "grch38", "grch37", "homo_sapiens",
              "homo sapiens", "gencode.v3", "gencode.v4"),
    "mouse": ("mm10", "mm39", "grcm38", "grcm39", "mus_musculus",
              "mus musculus", "gencode.vm", "vm25"),
}


def species_of_path(path) -> str:
    """The organism key a file path names, or "" when it names none.

    Deliberately conservative: it returns "" unless exactly one species'
    tokens appear, so an unrecognised or ambiguous path is never treated as a
    conflict.
    """
    s = str(path or "").lower()
    if not s:
        return ""
    hit = [k for k, toks in PATH_SPECIES_TOKENS.items()
           if any(t in s for t in toks)]
    return hit[0] if len(hit) == 1 else ""


@dataclass(frozen=True)
class OrganismProfile:
    """Everything the pipeline needs to know about one species.

    Field names match the pipeline vocabulary, not any one tool's: `genome` is
    the assembly label the ATAC bowtie2 index / HOMER / ROSE / DAStk all key
    off, `macs2_species` is MACS2's -g, `rsem_prefix` is the RSEM reference
    prefix inside `rna_ref_dir`.
    """
    key: str                      # canonical key: "human" | "mouse"
    scientific_name: str          # "Homo sapiens" -- matches ENA scientific_name
    common_name: str              # "human"
    tax_id: int                   # 9606 | 10090
    genome: str                   # "hg38" | "mm10"
    macs2_species: str            # "hs" | "mm"
    atac_ref_dir: str             # bowtie2 index + blacklist + peak-annot GTF
    rna_ref_dir: str              # STAR genomeDir + RSEM prefix + annotation
    rsem_prefix: str              # "GRCh38" | "GRCm38"
    gene_annotation_tsv: str      # Gene.ID -> Gene.Name (ENSG/ENSMUSG -> symbol)
    peak_annotation_gtf: str      # ATAC step4 HOMER annotatePeaks -gtf
    motif_scan_gtf: str           # motif-scan chain (MinmaxBedNode) GTF
    volcano_gtf: str              # ATAC volcano gene labels
    genome_fa: str                # indexed FASTA (pyfaidx: AG ref alleles)
    chrom_sizes: str
    blacklist_bed: str            # "" => globbed from atac_ref_dir by the MCP
    homer_genome: str             # HOMER genome key (data/genomes/<x>)
    rose_genome: str              # ROSE annotation key (<x>_refseq.ucsc)
    dastk_motif_dir: str          # directory of per-motif genome-wide BEDs
    dastk_refill_dir: str         # extra motif BEDs used to refill missing TFs
    dastk_genome: str             # DAStk -g
    alphagenome_organism: str     # alphagenome_client.resolve_organism() input
    diffbind_tissue: str          # DiffBind sample-sheet Tissue column
    tx2gene: str                  # transcript->gene map ("" when none exists)
    gene_info_gz: str             # NCBI gene_info for the evidence gate ("" = none)
    capabilities: dict            # layer -> bool (see CAPABILITY_NOTES)
    # -- literature namespace (2026-09-23, Task B) -------------------------
    # MRAG's evidence layer (GeneRIF, PubMed, PaperQA, curated scoring) is
    # indexed on HUMAN symbols. A non-human run reaches it through an ortholog
    # map, and every statement it produces has to disclose that. `ortholog_
    # to_human` says whether that translation happens; `ortholog_table` is the
    # table it uses. Human maps to itself, so its literature text is unchanged.
    ortholog_to_human: bool = False
    ortholog_table: str = ""
    gene_info_is_ortholog: bool = False   # gene_info_gz is the ORTHOLOG's

    # -- AlphaGenome biosamples (2026-09-23) --------------------------------
    # AlphaGenome's biosample ontologies are PER ORGANISM. Mouse has 18 curies
    # carrying ATAC against human's 167, and none of them is a cell line -- so
    # a human cell-line curie sent with organism="mouse" is rejected by the API
    # ("Unsupported ontology: EFO:0002074"). Every item then failed while the
    # tool still returned status "ok" and Section 13 rendered an empty table
    # that read as "AlphaGenome supported none of these TFs".
    # Human is EMPTY on purpose: its cell lines are resolved by
    # alphagenome_mcp's own PROSTATE_CELL_LINES / MODALITY_COVERAGE tables and
    # that path is unchanged. {lowercase name: ontology curie}.
    alphagenome_atac_biosamples: dict = field(default_factory=dict)

    # -- Cistrome GIGGLE (Section 9) (2026-09-24) ---------------------------
    # The TF ChIP-seq GIGGLE index and its GSM -> Factor map are PER ASSEMBLY:
    # a peak set can only be searched against an index built on its own
    # coordinates. run_giggle.sh picks both from `genome` (hg38 | mm10); these
    # are the same two paths, recorded here so can("giggle") and
    # missing_assets() check exactly what the search will open.
    giggle_index_dir: str = ""
    giggle_factor_map: str = ""

    # -- derived -----------------------------------------------------------
    @property
    def is_human(self) -> bool:
        return self.key == "human"

    @property
    def rose_annotation(self) -> str:
        return str(ROSE_DIR / "annotation" / f"{self.rose_genome}_refseq.ucsc")

    @property
    def rsem_reference(self) -> str:
        return str(Path(self.rna_ref_dir) / self.rsem_prefix)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["rose_annotation"] = self.rose_annotation
        d["rsem_reference"] = self.rsem_reference
        return d

    # -- capabilities ------------------------------------------------------
    def can(self, layer: str) -> bool:
        """True when `layer` can run for this organism.

        `dastk_motifs`, `homer_genome_data` and `rose_annotation` are answered
        from disk rather than from the table, because they are exactly the
        assets that get installed later.
        """
        if layer == "dastk_motifs":
            return bool(self.dastk_motif_dir) and Path(self.dastk_motif_dir).is_dir()
        if layer == "rose_annotation":
            return Path(self.rose_annotation).is_file()
        if layer == "giggle":
            # Flag AND disk (2026-09-24): a removed index must GATE Section 9
            # with a reason, not fail the step -- a failed step renders the
            # same empty panel as "no TF overlap".
            return (bool(self.capabilities.get("giggle", False))
                    and bool(self.giggle_index_dir)
                    and Path(self.giggle_index_dir).is_dir()
                    and bool(self.giggle_factor_map)
                    and Path(self.giggle_factor_map).is_file())
        if layer == "ortholog_map":
            # Human needs no table: it IS the literature namespace.
            if not self.ortholog_to_human:
                return True
            from . import orthology as _o
            return _o.available(self.ortholog_table or None)
        if layer in ("curated_literature", "gene_info"):
            # Both are human-symbol layers. A non-human run can still use them
            # -- through the ortholog route -- but ONLY when the ortholog table
            # is installed, because without it the layer would silently look up
            # mouse symbols in a human index and report the misses as absence
            # of evidence.
            if layer == "gene_info" and not self.gene_info_gz:
                return False
            if not self.ortholog_to_human:
                return bool(self.capabilities.get(layer, False))
            return self.can("ortholog_map")
        return bool(self.capabilities.get(layer, False))

    def why_not(self, layer: str) -> str:
        if self.can(layer):
            return ""
        if layer == "dastk_motifs":
            return (
                "DAStk needs a directory of genome-wide motif-hit BEDs per "
                "assembly; %s does not exist. The %s set is "
                "HOCOMOCO_v11_p1e-6_%s.tar.gz from DAStk's own setup.py "
                "(dowell.colorado.edu/pubs/DAStk/motifs/)."
                % (self.dastk_motif_dir, self.genome, self.genome))
        if layer == "giggle" and self.capabilities.get("giggle", False):
            _miss = [p or "(unset)" for p, isdir in (
                        (self.giggle_index_dir, True),
                        (self.giggle_factor_map, False))
                     if not p or not (Path(p).is_dir() if isdir
                                      else Path(p).is_file())]
            return (
                "the Cistrome GIGGLE assets for %s are not installed (%s), so "
                "Section 9 (TF ChIP-seq overlap) is skipped. They are built "
                "from CistromeDB_%s_tranfac_v3.0.gz as documented in "
                "Reference/cistrome/README_mm10_giggle.md."
                % (self.genome, ", ".join(_miss), self.genome))
        if layer in ("ortholog_map", "curated_literature", "gene_info"):
            from pathlib import Path as _P
            from . import orthology as _o
            _tbl = self.ortholog_table or str(_o.DERIVED_TSV)
            # "present but unusable" is a different fault from "absent", and
            # telling the user to install a file that is already there wastes
            # the one chance to say what is actually wrong (2026-09-23).
            _state = ("is present but carries too few orthologs to use"
                      if _P(_tbl).is_file() else "is not installed")
            return (
                "MRAG's literature layer is indexed on HUMAN gene symbols, so a "
                "%s run reaches it through a mouse->human ortholog map -- and "
                "that map %s (%s). Install %s from %s (~15 MB) "
                "and it is parsed automatically. Without it MRAG will not "
                "present human literature for %s genes, because every miss "
                "would read as 'no evidence' rather than 'wrong namespace'."
                % (self.common_name, _state, _tbl,
                   _o.SOURCE_NAME, _o.SOURCE_URL, self.common_name))
        return CAPABILITY_NOTES.get(
            (self.key, layer),
            "%s is not available for %s" % (layer, self.scientific_name))

    def require(self, layer: str) -> None:
        if not self.can(layer):
            raise OrganismError(
                "%s cannot run for %s (%s): %s"
                % (layer, self.scientific_name, self.genome, self.why_not(layer)))

    def missing_assets(self) -> list[str]:
        """Paths this profile names that are not on disk. Cheap: stat only."""
        out = []
        checks = [
            ("atac_ref_dir", self.atac_ref_dir, "dir"),
            ("rna_ref_dir", self.rna_ref_dir, "dir"),
            ("gene_annotation_tsv", self.gene_annotation_tsv, "file"),
            ("peak_annotation_gtf", self.peak_annotation_gtf, "file"),
            ("motif_scan_gtf", self.motif_scan_gtf, "file"),
            ("genome_fa", self.genome_fa, "file"),
            ("chrom_sizes", self.chrom_sizes, "file"),
            ("rose_annotation", self.rose_annotation, "file"),
            ("volcano_gtf", self.volcano_gtf, "file"),
        ]
        # The ATAC volcano labeller queries volcano_gtf with `tabix`, which
        # needs the bgzip index beside it. Without the .tbi every label silently
        # degrades to the raw peak coordinate instead of a gene name, so the
        # index is an asset in its own right, not an implementation detail.
        if self.volcano_gtf.endswith(".gz"):
            checks.append(("volcano_gtf_tbi", self.volcano_gtf + ".tbi", "file"))
        if self.ortholog_to_human:
            checks.append(("ortholog_table", self.ortholog_table, "file"))
        # Section 9 (2026-09-24): the per-assembly GIGGLE index + factor map.
        if self.capabilities.get("giggle", False):
            checks.append(("giggle_index_dir",
                           self.giggle_index_dir or "(unset)", "dir"))
            checks.append(("giggle_factor_map",
                           self.giggle_factor_map or "(unset)", "file"))
        for name, path, kind in checks:
            if not path:
                continue
            p = Path(path)
            ok = p.is_dir() if kind == "dir" else p.is_file()
            if not ok:
                out.append("%s=%s" % (name, path))
        return out


CAPABILITY_NOTES = {
    # ("mouse", "giggle") removed 2026-09-24: the mm10 Cistrome index
    # (giggle_index_mm10_tf, 7,624 datasets) is installed. A missing asset is
    # now explained by OrganismProfile.why_not("giggle") from disk.
    ("mouse", "curated_literature"): (
        "the PaperQA corpora, the GeneRIF/PubMed checks and the curated target "
        "scoring are built on human gene symbols and human cancer literature; "
        "a mouse run reaches them through the MGI mouse->human ortholog map "
        "and every resulting statement discloses the human ortholog it "
        "describes (mrag_orchestrator.orthology)"),
    ("mouse", "gene_info"): (
        "there is no Mus_musculus.gene_info.gz; the evidence gate reads the "
        "official full name of the HUMAN ORTHOLOG from "
        "Homo_sapiens.gene_info.gz, which is disclosed with the ortholog"),
    ("mouse", "tx2gene"): (
        "no mouse transcript->gene map is installed; transcript-level "
        "(kallisto ENSMUST) RNA matrices cannot be collapsed to genes"),
}


# ---------------------------------------------------------------------------
# The two supported profiles
# ---------------------------------------------------------------------------

def _human() -> OrganismProfile:
    atac = str(REFERENCE_ROOT / "ATAC_hg38")
    rna = str(REFERENCE_ROOT / "RNA_hg38")
    return OrganismProfile(
        key="human",
        scientific_name="Homo sapiens",
        common_name="human",
        tax_id=9606,
        genome="hg38",
        macs2_species="hs",
        atac_ref_dir=atac + "/",
        rna_ref_dir=rna + "/",
        rsem_prefix="GRCh38",
        gene_annotation_tsv=rna + "/gene_annotation.tsv",
        peak_annotation_gtf=atac + "/hg38.refGene.gtf",
        motif_scan_gtf=atac + "/Homo_sapiens.GRCh38.104.gtf",
        volcano_gtf=rna + "/gencode.v34.pc_lncRNA.sorted.gtf.gz",
        genome_fa=atac + "/hg38.fa",
        # ATAC_hg38 ships STAR's chrNameLength.txt rather than a
        # UCSC-style hg38.chrom.sizes; same two columns, same purpose.
        chrom_sizes=atac + "/chrNameLength.txt",
        blacklist_bed="",
        homer_genome="hg38",
        rose_genome="hg38",
        dastk_motif_dir=atac + "/grch38_motifs",
        dastk_refill_dir=atac + "/grch38_motifs_refill",
        dastk_genome="hg38",
        alphagenome_organism="human",
        diffbind_tissue="human",
        tx2gene=rna + "/tx2gene.v34.tsv",
        gene_info_gz="/BIO1/MRAG/MD_check/Homo_sapiens.gene_info.gz",
        capabilities={
            "giggle": True, "curated_literature": True, "gene_info": True,
            "tx2gene": True, "alphagenome": True, "se_analysis": True,
        },
        # Cistrome hg38 TF index (10,005 datasets) + its GSM -> Factor map.
        giggle_index_dir=str(CISTROME_ROOT / "giggle_index_hg38_tf"),
        giggle_factor_map=str(CISTROME_ROOT / "gsm_factor_map.tsv"),
        # Human IS the literature namespace: no translation, no disclosure,
        # byte-identical report text to before the ortholog route existed.
        ortholog_to_human=False,
        ortholog_table="",
        gene_info_is_ortholog=False,
    )


def _mouse() -> OrganismProfile:
    atac = str(REFERENCE_ROOT / "ATAC_mm10")
    rna = str(REFERENCE_ROOT / "RNA_mm10")
    return OrganismProfile(
        key="mouse",
        scientific_name="Mus musculus",
        common_name="mouse",
        tax_id=10090,
        genome="mm10",
        macs2_species="mm",
        atac_ref_dir=atac + "/",
        rna_ref_dir=rna + "/",
        rsem_prefix="GRCm38",
        gene_annotation_tsv=rna + "/gene_annotation.tsv",
        # No refGene-style GTF ships with ATAC_mm10; the GENCODE vM25 file that
        # the reference was built from is the peak-annotation source, so the
        # "{ref_dir}hg38.refGene.gtf" filename convention must be overridden
        # rather than pattern-substituted.
        peak_annotation_gtf=atac + "/gencode.vM25.annotation.gtf",
        motif_scan_gtf=atac + "/gencode.vM25.annotation.gtf",
        # MUST be bgzip+tabix indexed: viz_report_mcp.tools.volcano
        # ._nearest_genes() shells out to `tabix <gtf> <region>` and, on ANY
        # error, falls back to the raw peak id "so labels never go blank".
        # The plain vM25 GTF is not queryable, so every ATAC volcano label
        # would silently have become a coordinate string. This file mirrors
        # the human gencode.v34.pc_lncRNA.sorted.gtf.gz: protein_coding +
        # the lncRNA biotype family, sorted, bgzipped, tabix-indexed.
        volcano_gtf=rna + "/gencode.vM25.pc_lncRNA.sorted.gtf.gz",
        genome_fa=atac + "/mm10.fa",
        chrom_sizes=atac + "/mm10.chrom.sizes",
        blacklist_bed=atac + "/mm10.blacklist.bed",
        homer_genome="mm10",
        rose_genome="mm10",
        dastk_motif_dir=atac + "/mm10_motifs",
        # Per-assembly, and deliberately absent: the SE-TRN display-MD refill
        # skips itself when this directory does not exist. Pointing mouse at the
        # human grch38_motifs_refill would scan mouse peaks with human motif
        # coordinates under -g mm10 and report the numbers as mouse.
        dastk_refill_dir=atac + "/mm10_motifs_refill",
        dastk_genome="mm10",
        alphagenome_organism="mouse",
        # Measured live against the AlphaGenome API on 2026-09-23: these are
        # the UBERON tissues among mouse's 18 ATAC biosamples whose meaning is
        # unambiguous. The remaining curies (CL:*, EFO:*, NTR:*) are left out
        # rather than guessed at; an explicit curie still reaches the API.
        alphagenome_atac_biosamples={
            "liver":       "UBERON:0002107",
            "kidney":      "UBERON:0002113",
            "lung":        "UBERON:0002048",
            "heart":       "UBERON:0000948",
            "stomach":     "UBERON:0000945",
            "intestine":   "UBERON:0000160",
            "forebrain":   "UBERON:0001890",
            "midbrain":    "UBERON:0001891",
            "hindbrain":   "UBERON:0002028",
            "neural tube": "UBERON:0001049",
            "limb":        "UBERON:0002101",
        },
        diffbind_tissue="mouse",
        tx2gene="",
        # No Mus_musculus.gene_info.gz exists here, and inventing one is not
        # the fix: the evidence gate wants the official full name of the gene
        # whose LITERATURE was retrieved, and that is the human ortholog.
        gene_info_gz="/BIO1/MRAG/MD_check/Homo_sapiens.gene_info.gz",
        capabilities={
            # giggle: True since 2026-09-24 -- an mm10 Cistrome index was built
            # from CistromeDB_mm10_tranfac_v3.0.gz with the human build's exact
            # top-10k rule (Reference/cistrome/README_mm10_giggle.md). It is an
            # mm10-coordinate index: nothing is lifted over or ortholog-mapped.
            # can("giggle") also requires the two paths below on disk.
            # curated_literature / gene_info are answered by can(), which
            # requires the ortholog table rather than a flag.
            "giggle": True, "curated_literature": True, "gene_info": True,
            "tx2gene": False, "alphagenome": True, "se_analysis": True,
        },
        # Cistrome mm10 TF index (7,624 datasets) + its GSM -> Factor map
        # (from the archive's own mm10_tranfac_QC.txt, 7,624/7,624 mapped).
        giggle_index_dir=str(CISTROME_ROOT / "giggle_index_mm10_tf"),
        giggle_factor_map=str(CISTROME_ROOT / "gsm_factor_map_mm10.tsv"),
        ortholog_to_human=True,
        ortholog_table=str(ORTHOLOGY_ROOT / "mouse_human_orthologs.tsv"),
        gene_info_is_ortholog=True,
    )


PROFILES: dict[str, OrganismProfile] = {"human": _human(), "mouse": _mouse()}
DEFAULT_KEY = "human"

# Every spelling we accept, lowercased. Genome builds are included because a
# run.toml is as likely to say "mm10" as "mouse".
ALIASES: dict[str, str] = {}
for _key, _words in {
    "human": ("human", "homo sapiens", "homo_sapiens", "h. sapiens", "hsapiens",
              "hs", "hg38", "grch38", "grch38.p13", "9606", "hsa"),
    "mouse": ("mouse", "mus musculus", "mus_musculus", "m. musculus", "mmusculus",
              "mm", "mm10", "grcm38", "10090", "mmu", "murine"),
}.items():
    for _w in _words:
        ALIASES[_w] = _key

# Spellings we refuse on purpose, with the reason.
REFUSED: dict[str, str] = {
    "mm39": ("mm39 / GRCm39 is refused on purpose: AlphaGenome's mouse tracks "
             "are built on mm10 (GRCm38), and mm39 coordinates are valid "
             "numbers that address the WRONG locus without raising anything "
             "(Reference/README_mm10.md, 2026-09-03). The mm39 indexes were "
             "deleted for the same reason. Use mm10."),
    "grcm39": None,   # filled below: same text as mm39
    "hg19": ("hg19 / GRCh37 has no MRAG reference tree; only hg38 is built. "
             "Lift the data over to hg38 or build an ATAC_hg19/RNA_hg19 pair."),
    "grch37": None,
}
REFUSED["grcm39"] = REFUSED["mm39"]
REFUSED["grch37"] = REFUSED["hg19"]


def _norm(spec) -> str:
    s = str(spec or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def resolve(spec) -> OrganismProfile:
    """Any spelling of a species or build -> its profile. Fails loudly.

    Accepts "human", "Homo sapiens", "hg38", "hs", 9606, "mouse", "mm10",
    "GRCm38", "Mus musculus", 10090, or a profile (returned unchanged).
    """
    if isinstance(spec, OrganismProfile):
        return spec
    s = _norm(spec)
    if not s:
        raise OrganismError(
            "no organism given; expected one of %s" % sorted(set(ALIASES)))
    if s in REFUSED:
        raise OrganismError(REFUSED[s])
    if s in ALIASES:
        return PROFILES[ALIASES[s]]
    raise OrganismError(
        "unsupported organism %r. MRAG has reference trees for %s only "
        "(accepted spellings: %s)."
        % (spec, ", ".join(p.scientific_name for p in PROFILES.values()),
           ", ".join(sorted(set(ALIASES)))))


def resolve_many(specs: Iterable, where: str = "sample set") -> OrganismProfile:
    """Resolve several species strings that MUST agree. Mixed => loud failure."""
    seen: dict[str, list[str]] = {}
    for sp in specs:
        if not str(sp or "").strip():
            continue
        prof = resolve(sp)
        seen.setdefault(prof.key, []).append(str(sp))
    if not seen:
        raise OrganismError("no usable species string in %s" % where)
    if len(seen) > 1:
        detail = "; ".join(
            "%s (from %s)" % (PROFILES[k].scientific_name,
                              ", ".join(sorted(set(v))))
            for k, v in sorted(seen.items()))
        raise OrganismError(
            "mixed species in %s: %s. MRAG aligns every sample to ONE genome, "
            "so a mixed set would align part of the data to the wrong assembly. "
            "Split the run, one organism each." % (where, detail))
    return PROFILES[next(iter(seen))]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrganismDecision:
    """What was chosen, where it came from, and what the user should be told."""
    profile: OrganismProfile
    source: str          # explicit | sra_metadata | sample_sheet | inferred | default
    detail: str          # the evidence, e.g. "sra_manifest.json: Mus musculus x8"
    note: str = ""       # non-empty when the user must SEE this decision
    evidence: tuple = ()

    @property
    def organism(self) -> str:
        return self.profile.key

    @property
    def genome(self) -> str:
        return self.profile.genome

    def as_dict(self) -> dict:
        return {
            "organism": self.profile.key,
            "scientific_name": self.profile.scientific_name,
            "tax_id": self.profile.tax_id,
            "genome": self.profile.genome,
            "source": self.source,
            "detail": self.detail,
            "note": self.note,
            "evidence": list(self.evidence),
            "capabilities": {k: self.profile.can(k) for k in
                             ("giggle", "dastk_motifs", "curated_literature",
                              "gene_info", "tx2gene", "alphagenome",
                              "ortholog_map")},
            "ortholog_to_human": self.profile.ortholog_to_human,
            "ortholog_table": self.profile.ortholog_table,
        }


_SRA_METADATA_FILES = ("sra_manifest.json", "sra_names.json")


def species_from_sra_dir(path) -> tuple[list[str], str]:
    """Species strings recorded by `mrag sra-fetch` next to the FASTQs.

    Looks in the directory and its two parents, because detect is usually
    pointed at <dest> but may be pointed at <dest>/ATAC. Returns
    (species_strings, evidence_string).
    """
    if not path:
        return [], ""
    p = Path(path)
    cands = [p, p.parent, p.parent.parent]
    for d in cands:
        for name in _SRA_METADATA_FILES:
            f = d / name
            if not f.is_file():
                continue
            try:
                raw = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            found: list[str] = []
            for s in (raw.get("samples") or []):
                if isinstance(s, dict) and s.get("species"):
                    found.append(str(s["species"]))
            for _run, rec in (raw.get("runs") or {}).items():
                if isinstance(rec, dict) and rec.get("species"):
                    found.append(str(rec["species"]))
            if found:
                uniq = sorted(set(found))
                return found, "%s: %s (%d run(s))" % (
                    f, ", ".join(uniq), len(found))
    return [], ""


_ENSG = re.compile(r"\bENSG\d{6,}")
_ENSMUSG = re.compile(r"\bENSMUSG\d{6,}")


def infer_from_artifacts(paths: Iterable[str], max_bytes: int = 262144
                         ) -> tuple[Optional[str], str]:
    """Cheap, reliable inference from files already on disk.

    Two signals, strongest first:

    1. Ensembl gene-id prefix in an expression / integrated table. ENSMUSG is
       mouse and ENSG is human by construction, so one hit is conclusive.
    2. Chromosome names in a peak BED/narrowPeak. Mouse has 19 autosomes and
       human has 22, so chr20/21/22 means human; a file that names >= 15
       distinct chromosomes and never exceeds chr19 is mouse.

    FASTQ is deliberately NOT sniffed: deciding a species from raw reads means
    aligning them, which is neither cheap nor reliable at this stage.
    Returns (key_or_None, evidence).
    """
    for p in paths:
        if not p:
            continue
        f = Path(p)
        if not f.is_file():
            continue
        try:
            head = f.open("rb").read(max_bytes).decode("utf-8", "replace")
        except OSError:
            continue
        if _ENSMUSG.search(head):
            return "mouse", "ENSMUSG gene ids in %s" % f
        if _ENSG.search(head):
            return "human", "ENSG gene ids in %s" % f
    # 2nd pass: chromosome names
    for p in paths:
        if not p:
            continue
        f = Path(p)
        if not f.is_file() or f.suffix.lower() not in (
                ".bed", ".narrowpeak", ".filt", ".broadpeak"):
            continue
        chroms: set[str] = set()
        try:
            with f.open() as fh:
                for i, line in enumerate(fh):
                    if i > 200000:
                        break
                    c = line.split("\t", 1)[0].strip()
                    if c and not c.startswith(("#", "track")):
                        chroms.add(c)
        except OSError:
            continue
        nums = set()
        for c in chroms:
            m = re.match(r"^(?:chr)?(\d+)$", c)
            if m:
                nums.add(int(m.group(1)))
        if nums & {20, 21, 22}:
            return "human", "chr20/21/22 present in %s" % f
        if len(chroms) >= 15 and nums and max(nums) <= 19:
            return "mouse", ("no autosome above chr19 across %d chromosomes in %s"
                             % (len(chroms), f))
    return None, ""


def decide(explicit=None,
           metadata_species: Optional[Iterable] = None,
           metadata_detail: str = "",
           fastq_dir=None,
           artifact_paths: Optional[Iterable[str]] = None,
           where: str = "run") -> OrganismDecision:
    """Resolve the organism for a run, in the documented precedence order.

      1. explicit user setting  (RunConfig.organism / --organism / chat field)
      2. SRA/ENA metadata       (scientific_name carried to sra_manifest.json)
      3. inference from data    (Ensembl gene-id prefix, then chromosome names)
      4. default human, WITH a visible note

    A mixed-species metadata set raises OrganismError -- never silently picks.
    """
    if explicit not in (None, ""):
        prof = resolve(explicit)
        return OrganismDecision(prof, "explicit",
                                "user set organism=%r" % (explicit,),
                                evidence=(str(explicit),))

    species = [s for s in (metadata_species or []) if str(s or "").strip()]
    if not species and fastq_dir:
        species, metadata_detail = species_from_sra_dir(fastq_dir)
    if species:
        prof = resolve_many(species, where=metadata_detail or where)
        return OrganismDecision(
            prof, "sra_metadata",
            metadata_detail or ("ENA scientific_name: %s"
                                % ", ".join(sorted(set(map(str, species))))),
            evidence=tuple(sorted(set(map(str, species)))))

    key, why = infer_from_artifacts(artifact_paths or [])
    if key:
        return OrganismDecision(
            PROFILES[key], "inferred", why,
            note=("Organism was INFERRED from the data (%s), not declared. "
                  "Set organism= in the run config if that is wrong." % why),
            evidence=(why,))

    prof = PROFILES[DEFAULT_KEY]
    return OrganismDecision(
        prof, "default",
        "no organism given, no SRA metadata, nothing to infer from",
        note=("No organism was declared and none could be detected, so this "
              "run DEFAULTS to %s/%s. If the data is not human, set "
              "organism= in the run config (or --organism) and re-run: every "
              "genome argument below follows this decision."
              % (prof.scientific_name, prof.genome)))


def species_by_basename(path) -> dict:
    """{fastq basename -> species} from the SRA metadata beside the FASTQs.

    `mrag sra-fetch` names each file after the basename it minted, and both
    sra_manifest.json (this selection) and sra_names.json (every selection ever
    downloaded here) record that basename. This is how `mrag detect` gives an
    individual sample its organism instead of only the study one.
    """
    out: dict = {}
    if not path:
        return out
    p = Path(path)
    for d in (p, p.parent, p.parent.parent):
        for name in _SRA_METADATA_FILES:
            f = d / name
            if not f.is_file():
                continue
            try:
                raw = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            for s in (raw.get("samples") or []):
                if isinstance(s, dict) and s.get("basename") and s.get("species"):
                    out.setdefault(str(s["basename"]), str(s["species"]))
            for _run, rec in (raw.get("runs") or {}).items():
                if isinstance(rec, dict) and rec.get("basename") and rec.get("species"):
                    out.setdefault(str(rec["basename"]), str(rec["species"]))
    return out
