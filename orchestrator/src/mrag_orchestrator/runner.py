"""High-level pipeline composer.

  - Load RunConfig from TOML
  - Run the ATAC / RNA preprocessing (FASTQ input) or the processed-matrix adapters,
    then combined -> diffbind -> homer -> variants -> the report sections -> render
  - Use precomputed artifacts where supplied
  - Run via the Pipeline executor + variants flow
"""
from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config     import RunConfig
from .manifest   import Manifest
from .pipeline   import Pipeline
from .checkpoint import CheckpointCache

# ---- Protected cache for the matrix-input HOMER annotation (2026-10-19) ----
# A count-matrix run (TCGA-PRAD) annotates its canonical peak catalog with the full motif library,
# ~2-3 h on 112k peaks, and reused it only when the SAME output dir already held the result, so
# every fresh run recomputed it. The result depends on exactly three things -- the peak catalog,
# the JASPAR-extended library appended to the run's known motifs, and the HOMER genome -- so the
# cache is keyed on all three: a changed catalog, a grown library or another genome is a miss,
# never a stale hit. A lookup or store failure is printed and the run carries on as before.
_CANON_CACHE = Path(os.environ.get("MRAG_HOMER_CANON_CACHE", "/BIO1/MRAG/Agent/cache/homer_canonical"))
_CANON_LIB = Path("/BIO1/MRAG/Agent/Reference/combined_v12_jaspar.motif")
_CANON_FILES = ("atac_matrix.motif", "atac_matrix.motif.tsv")


def _md5_file(path, bs=1 << 20):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def _canon_cache_key(bed, genome):
    return "%s_lib%s_%s" % (_md5_file(bed)[:16], _md5_file(_CANON_LIB)[:12], genome)


def _canon_tsv_complete(tsv, bed):
    """An annotation has a header plus one row per catalog peak (>= 99%: annotatePeaks may drop a
    peak on an unplaced contig). A truncated file from an interrupted run is not complete."""
    with open(bed) as fh:
        n_peaks = sum(1 for l in fh if l.strip())
    with open(tsv) as fh:
        n_rows = sum(1 for _ in fh) - 1
    return n_peaks > 0 and n_rows >= 0.99 * n_peaks


def _seed_canonical_from_cache(bed, homer_dir, genome):
    """Symlink a cached annotation into this run's homer_canonical/ when the key matches."""
    key = _canon_cache_key(bed, genome)
    src = _CANON_CACHE / key
    if not all((src / f).is_file() for f in _CANON_FILES):
        print(f"[homer_canonical] cache miss ({key})", flush=True)
        return False
    Path(homer_dir).mkdir(parents=True, exist_ok=True)
    for f in _CANON_FILES:
        t = Path(homer_dir) / f
        if not t.exists():
            t.symlink_to((src / f).resolve())
    print(f"[homer_canonical] cache hit ({key}): seeded from {src}", flush=True)
    return True


def _store_canonical_in_cache(bed, homer_dir, genome):
    """Copy a freshly computed, complete annotation into the cache (atomic: temp dir, rename)."""
    import shutil
    key = _canon_cache_key(bed, genome)
    dst = _CANON_CACHE / key
    if dst.is_dir():
        return False
    files = [Path(homer_dir) / f for f in _CANON_FILES]
    if not all(f.is_file() for f in files) or not _canon_tsv_complete(files[1], bed):
        print("[homer_canonical] not cached: annotation incomplete", flush=True)
        return False
    tmp = _CANON_CACHE / (".tmp_%s_%d" % (key, os.getpid()))
    tmp.mkdir(parents=True, exist_ok=False)
    for f in files:
        shutil.copyfile(f.resolve(), tmp / f.name)
    shutil.copyfile(bed, tmp / "canonical_peaks.bed")
    (tmp / "KEY.txt").write_text(
        "peak catalog md5, JASPAR library md5 (%s), HOMER genome -> %s\nstored %s\n"
        % (_CANON_LIB, key, datetime.now().isoformat(timespec="seconds")))
    tmp.rename(dst)
    print(f"[homer_canonical] cached as {dst}", flush=True)
    return True
from .variants   import run_all_variants
from .synthesis  import synthesize_cross_variant, render_cross_variant_markdown
from .nodes.combined import CombinedNode
from .nodes.diffbind import DiffbindNode
from .nodes.homer    import HomerNode
from .nodes.atac     import ATACNode
from .nodes.dastk    import DAStkNode
from .nodes.rna      import RNANode
from .nodes.deseq2   import DESeq2Node
from .nodes.viz_report import RenderNode
from .derive_summaries import derive_conditions_meta
from .organism import decide as decide_organism, resolve as resolve_organism, species_of_path
from .resources import ncpus as _ncpus


def build_run_id(label: Optional[str] = None) -> str:
    return label or datetime.now().strftime("run_%Y%m%dT%H%M%S")


_DEFAULT_TX2GENE = "/BIO1/MRAG/Agent/Reference/RNA_hg38/tx2gene.v34.tsv"


def _load_tx2gene(path: str) -> dict:
    """Load a transcript->gene-symbol map. Accepts 2-col (ENST, symbol) or
    3-col (ENST, ENSG, symbol) TSV; ENST version stripped for robust join."""
    import pandas as pd
    df = pd.read_csv(path, sep="\t", header=None, dtype=str)
    tx = df.iloc[:, 0].str.split(".").str[0]
    sym = df.iloc[:, 2] if df.shape[1] >= 3 else df.iloc[:, 1]
    return dict(zip(tx, sym))


# Input-data guard. A FASTQ run used the FASTQ directory itself as the
# ATAC / RNA project dir, so the steps' checkpoints, summaries and the DAStk / DESeq2
# outputs were written into the input data. The project dir is now a mirror under the run's output
# dir: small files are copied (the steps rewrite them), FASTQ / BAM / bigWig and any
# file >= 64 MiB are symlinked (only read by a step the checkpoint marks complete),
# and the source tree's size + mtime are compared after the steps.
_STAGE_LINK_MIN = 64 << 20
_STAGE_LINK_EXT = (".fastq.gz", ".fq.gz", ".fastq", ".fq", ".bam", ".bai",
                   ".bw", ".bigwig", ".cram", ".crai")
_STAGED: dict[str, str] = {}


def _stage_project_dir(src: str, dst: Path, out_dir: Path) -> str:
    """Mirror src under dst and return dst; return src when it already lies inside
    out_dir. Entries already in dst are kept, so a resumed run keeps its state."""
    import shutil
    src_p = Path(src).resolve()
    try:
        src_p.relative_to(Path(out_dir).resolve())
        return src
    except ValueError:
        pass
    dst = Path(dst)
    for root, dirs, files in os.walk(src_p):
        rel = Path(root).relative_to(src_p)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for name in list(dirs):
            s = Path(root) / name
            if s.is_symlink():
                dirs.remove(name)
                d = dst / rel / name
                if not (d.exists() or d.is_symlink()):
                    os.symlink(str(s.resolve()), d)
        for name in files:
            s = Path(root) / name
            d = dst / rel / name
            if d.exists() or d.is_symlink():
                continue
            if s.is_symlink() or name.lower().endswith(_STAGE_LINK_EXT) \
                    or s.stat().st_size >= _STAGE_LINK_MIN:
                os.symlink(str(s.resolve()), d)
            else:
                shutil.copy2(s, d)
    _STAGED[str(Path(src))] = str(dst)
    _STAGED[str(src_p)] = str(dst)
    print(f"[stage] project dir {src} -> {dst}", flush=True)
    return str(dst)


def _sample_dir(s) -> Path:
    """The project dir holding a FASTQ sample's outputs (the staged mirror if any)."""
    src = Path(s.fastq_r1).parent
    return Path(_STAGED.get(str(src), str(src)))


def _tree_state(src: str) -> dict:
    out = {}
    for root, _dirs, files in os.walk(src):
        for name in files:
            p = os.path.join(root, name)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            out[p] = (st.st_size, st.st_mtime_ns)
    return out


def _tree_changed(src: str, before: dict) -> list:
    after = _tree_state(src)
    return sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))


def _prepare_rna_matrix(matrix_path: str, cfg, project_dir: str,
                        tx2gene_path: str = _DEFAULT_TX2GENE):
    """Prepare an external RNA expression matrix (counts or TPM) for the R
    DESeq2 engine. Handles annotation columns (gene_id, gene_name, length,
    eff_length) and transcript-level (kallisto ENST) input, collapsing to
    gene-level (sum over transcripts, tximport-abundance style). Returns
    (tsv_path, mapfile_path); the mapfile assigns each sample column to
    treated/untreated by matching condition labels (conditions[0] = treated)."""
    import pandas as pd
    from .detector import _matrix_layout
    with open(matrix_path) as _f:
        _head = _f.readline()
    _sep = "\t" if "\t" in _head else ","
    df = pd.read_csv(matrix_path, sep=_sep)
    lay = _matrix_layout(df)
    if not lay:
        raise ValueError(f"RNA matrix: could not identify sample columns in "
                         f"{matrix_path} (header: {list(df.columns)[:8]})")
    sample_cols = lay["sample_cols"]

    # Choose gene label: prefer gene_name; else id column; ENST ids collapse to
    # gene symbol via tx2gene.
    if lay["name_col"] is not None:
        labels = df[lay["name_col"]].astype(str)
    else:
        idcol = lay["id_col"] if lay["id_col"] is not None else df.columns[0]
        ids = df[idcol].astype(str)
        if float(ids.str.startswith("ENST").mean()) > 0.5:
            if not tx2gene_path or not Path(tx2gene_path).is_file():
                raise FileNotFoundError(
                    "the RNA matrix %s is transcript-level (ENST ids) but no "
                    "transcript->gene map exists for this organism "
                    "(tx2gene=%r). Supply a gene-level matrix, or install a "
                    "tx2gene TSV and point organism.py at it -- collapsing "
                    "mouse transcripts with the human map would map them to "
                    "the wrong genes." % (matrix_path, tx2gene_path))
            t2g = _load_tx2gene(tx2gene_path)
            labels = ids.str.split(".").str[0].map(t2g)
        else:
            labels = ids

    mat = df[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    mat.index = labels
    mat = mat[mat.index.notna() & (mat.index.astype(str) != "nan")
              & (mat.index.astype(str) != "")]
    gene = mat.groupby(level=0).sum()          # collapse transcripts/dupes
    gene.index.name = "gene_id"
    tsv_path = str(Path(project_dir) / "rna_input_matrix.tsv")
    gene.to_csv(tsv_path, sep="\t")

    def _tok(s: str) -> str:
        return s.lower().replace("-", "_").replace(".", "_")
    treated_label = cfg.conditions[0].label if (cfg.conditions and len(cfg.conditions) >= 1) else ""
    tl = _tok(treated_label)
    lines = ["sample\tTREATMENT"]
    n_treated = 0
    for col in sample_cols:
        if tl and tl in _tok(str(col)):
            lines.append(f"{col}\ttreated"); n_treated += 1
        else:
            lines.append(f"{col}\tuntreated")
    if n_treated == 0:
        raise ValueError(
            f"RNA matrix: no sample column matched treated label '{treated_label}'. "
            f"Sample columns: {sample_cols}"
        )
    mapfile = Path(project_dir) / "sample_mapfile.csv"
    mapfile.write_text("\n".join(lines) + "\n")
    return tsv_path, str(mapfile)


def _config_preflight(cfg) -> None:
    """Loud effective-config echo + stale/reused-toml guard (see module note)."""
    import os
    from .config import RunConfig as _RC, PipelineConfig as _PC

    canon_def = _RC.model_fields["canonical_N"].default
    topn_def = list(_PC.model_fields["topn_variants"].default)
    cur_def = _RC.model_fields["curated_targets"].default

    _conds = " vs ".join(c.label for c in cfg.conditions) if cfg.conditions else "(none)"
    _rna = (cfg.precomputed.rna_log2_matrix or cfg.precomputed.rna_counts_matrix
            or "(fastq/none)")
    _bar = "=" * 66
    for _ln in (
        _bar,
        "  MRAG RUN - effective config (verify before it runs for hours)",
        _bar,
        "  run_label       : %s" % cfg.run_label,
        "  canonical_N     : %s   (TF panel size; current default %s)"
        % (cfg.canonical_N, canon_def),
        "  topn_variants   : %s" % (cfg.pipeline.topn_variants,),
        "  curated_targets : %s" % cfg.curated_targets,
        "  cell_line       : %s" % (cfg.cell_line or "(unset -> resolved fallback)"),
        "  cancer_type     : %s" % (cfg.cancer_type or "(unset)"),
        "  conditions      : %s" % _conds,
        "  ATAC matrix     : %s" % (cfg.precomputed.atac_count_matrix or "(fastq/none)"),
        "  RNA  matrix     : %s" % _rna,
        _bar,
    ):
        print(_ln, flush=True)

    _cond_labels = {c.label for c in cfg.conditions}
    for _w in (
        ("topn_variants=%s != default %s" % (cfg.pipeline.topn_variants, topn_def)
         if list(cfg.pipeline.topn_variants) != topn_def else None),
        ("curated_targets=%s != default %s" % (cfg.curated_targets, cur_def)
         if cfg.curated_targets != cur_def else None),
    ):
        if _w:
            print("[config] WARNING: %s" % _w, flush=True)
    for _f in ("bigwig_arm_a_label", "bigwig_arm_b_label"):
        _v = getattr(cfg.precomputed, _f, "") or ""
        if _v and _cond_labels and _v not in _cond_labels:
            print("[config] WARNING: precomputed.%s=%r matches no condition %s "
                  "(stale label from a copied toml?)"
                  % (_f, _v, sorted(_cond_labels)), flush=True)

    if cfg.canonical_N != canon_def and not os.environ.get("MRAG_ALLOW_CONFIG_DRIFT"):
        raise RuntimeError(
            "canonical_N=%d differs from the current default %d. This is the usual "
            "fingerprint of a run.toml copied from an older run. Generate the config "
            "fresh (webui form, or `mrag confirm` from the sample sheet) so it picks "
            "up current defaults -- do not reuse an old run.toml. If a non-default "
            "panel size is genuinely intended, re-run with MRAG_ALLOW_CONFIG_DRIFT=1."
            % (cfg.canonical_N, canon_def))


# ---------------------------------------------------------------------------
# 2026-09-28 pre-deploy review (finding 1): a curated run whose target step cannot work fails
# LOUDLY.  Before, a failed curated_targets step was only a 'fail' entry in the manifest: the run
# exited 0, the report rendered without Section-10 targets (or, in a reused output dir, with the
# PREVIOUS run's curated_targets.json) and the web UI's progress view showed the run finished.  The
# 2026-09-28 scorer adds expected failure paths (no GPU free, the 2 % unscored stop), so:
#   - the scorer is checked at the START of the run (read-only; the legacy profile skips it);
#   - a failed curated step prints a loud block, the report says so at the top and in Section 10
#     and shows no stale targets, and the run returns ok=False (mrag run exits 1).
def _curated_failed_message(err, at_start=False) -> str:
    """The notice for a failed target step (run.log, report banner, manifest summary, result).
    at_start: the run-start scorer check stopped the run before any step (no report)."""
    e = " ".join(str(err or "curated_targets failed").split())[:1500].rstrip(".")
    hint = ""
    if (any(k in e for k in ("scorer preflight", "scorer check", "UNSCORED", "ensure_scorer"))
            and "MRAG_SCORER_PROFILE=legacy" not in e):
        hint = (" Retry when the scorer can run (/BIO1/MRAG/scripts/ensure_scorer_ollama.sh --probe "
                "says why it cannot now), or run with MRAG_SCORER_PROFILE=legacy (the pre-2026-09-28 "
                "scorer).")
    if at_start:
        return ("Target-gene selection (curated_targets) cannot run, so the run STOPPED before any "
                "step: %s.%s" % (e, hint))
    return ("Target-gene selection (curated_targets) FAILED in this run: %s.%s Section 10 shows NO "
            "evidence-scored target genes, and no later section is based on them." % (e, hint))


def _print_loud(msg: str) -> None:
    import textwrap as _tw
    _bar = "!" * 78
    print(_bar, flush=True)
    for _ln in _tw.wrap(msg, 76):
        print("  " + _ln, flush=True)
    print(_bar, flush=True)


def _scorer_startup_gate(cfg, manifest) -> Optional[dict]:
    """At the START of a curated run: nodes.curated_targets.score_startup_check (read-only: the
    ensure script's --probe; skipped in the legacy profile and with MRAG_SCORE_PREFLIGHT=0).
    Returns None to go on (a 'scorer_check' step is recorded unless skipped; "busy" is a warning,
    since the curated step waits for a GPU), or the result to return at once: a misconfiguration that
    would make the curated step fail whatever the GPUs do, recorded as a FAILED curated_targets step
    so the run list, the status page and the progress view all show it."""
    if not bool(getattr(cfg, "curated_targets", False)):
        return None
    import time as _time
    _t0 = _time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        from .nodes.curated_targets import score_startup_check as _ssc
        rec = _ssc(cfg.llm.model, cfg.llm.endpoint.rstrip("/") + "/v1/chat/completions")
    except Exception as _e:
        err = _curated_failed_message("scorer check at run start: %s: %s" % (type(_e).__name__, _e),
                                      at_start=True)
        _print_loud(err)
        manifest.add_step({"name": "curated_targets", "status": "fail", "started_at": _t0,
                           "ended_at": _time.strftime("%Y-%m-%dT%H:%M:%S"), "error": err[:2000]})
        manifest.finalize(summary={"curated_targets_error": err})
        manifest.write_atomic()
        return {"ok": False, "failed_at": "curated_targets", "stage": "scorer_check", "error": err,
                "manifest": str(manifest.path), "final_html": ""}
    if rec.get("status") == "skipped":
        return None
    _warn = rec.get("status") != "ok"
    print("[scorer] %scheck at run start: %s -- profile %s, %s at %s%s" % (
        "WARNING " if _warn else "", rec.get("status"), rec.get("profile"), rec.get("model"),
        rec.get("endpoint"), (": " + str(rec.get("detail"))) if rec.get("detail") else ""), flush=True)
    if rec.get("status") == "busy":
        print("[scorer] WARNING the target scorer cannot load now; the curated step waits for a free GPU "
              "(MRAG_SCORE_ENSURE_WAIT_S, default 1800 s) and fails loudly if none frees up.", flush=True)
    manifest.add_step({"name": "scorer_check", "status": "ok", "started_at": _t0,
                       "ended_at": _time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "outputs": [{"key": k, "value": str(v)[:400]} for k, v in rec.items()]})
    return None


def run_from_config(cfg: RunConfig, resume_manifest: Optional[str] = None) -> dict:
    """Execute the downstream pipeline from a RunConfig.

    Uses precomputed artifacts where supplied; otherwise runs the corresponding
    node (ATAC / RNA preprocessing from cfg.samples for FASTQ input).
    """
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    _STAGED.clear()
    _config_preflight(cfg)
    run_id = build_run_id(cfg.run_label)
    manifest = Manifest(
        run_id=run_id, code_version="mrag-orchestrator 0.1.0",
        llm={"endpoint": cfg.llm.endpoint, "model": cfg.llm.model},
        manifest_path=out_dir / "manifest.json",
    )

    # Declare inputs
    for kind, paths in {
        "atac_fastqs":    [s.fastq_r1 for s in cfg.samples if s.modality == "ATAC"],
        "rna_fastqs":     [s.fastq_r1 for s in cfg.samples if s.modality == "RNA"],
        "precomputed":    [p for p in cfg.precomputed.model_dump().values()
                           if isinstance(p, str) and p and Path(p).exists()],
    }.items():
        if paths:
            manifest.declare_inputs(kind, paths)

    cache = CheckpointCache.load_from_manifest(resume_manifest) if resume_manifest else None

    pre = cfg.precomputed
    # Robustness: auto-recover precomputed file paths that moved between
    # sub-dirs (e.g. group TSV ATAC/ -> meta/); fail early+clearly if truly
    # missing. Search root = the detect --fastq-dir data root.
    pre.resolve_missing_paths(cfg.output_dir)

    # ================= Organism / genome profile (2026-09-23) =================
    # ONE decision, made before any node runs, recorded in the manifest, and
    # threaded into every genome argument below. Precedence:
    #   explicit cfg.organism > pipeline.homer_genome (an explicit genome
    #   statement) > SRA/ENA scientific_name on the samples > inference from
    #   the data on disk > human with a visible note.
    _org_fastq_dir = None
    for _s in (cfg.samples or []):
        if _s.fastq_r1:
            _org_fastq_dir = str(Path(_s.fastq_r1).parent); break
    _org_artifacts = [pre.deseq_csv, pre.rna_counts_matrix, pre.rna_log2_matrix,
                      pre.integrated_csv, pre.atac_count_matrix, pre.atac_md_csv]
    _org_artifacts += [str(x) for x in (pre.atac_peaks_treatment or [])[:1]]
    _org_artifacts += [str(x) for x in (pre.atac_peaks_control or [])[:1]]
    _explicit_org = cfg.organism or None
    _hg_cfg = (getattr(cfg.pipeline, "homer_genome", "") or "").strip()
    _hg_literal = ""          # a HOMER genome key we do not recognise as a species
    if _hg_cfg:
        try:
            _hg_prof = resolve_organism(_hg_cfg)
        except Exception:
            _hg_literal = _hg_cfg      # e.g. a custom HOMER genome install
        else:
            if _explicit_org is None:
                _explicit_org = _hg_cfg
            elif resolve_organism(_explicit_org).key != _hg_prof.key:
                raise ValueError(
                    "run.toml disagrees with itself: organism=%r is %s but "
                    "pipeline.homer_genome=%r is %s. Set them to the same "
                    "species (or drop homer_genome)."
                    % (cfg.organism, resolve_organism(_explicit_org).scientific_name,
                       _hg_cfg, _hg_prof.scientific_name))
    org_decision = decide_organism(
        explicit=_explicit_org,
        metadata_species=[getattr(s, "organism", None) for s in (cfg.samples or [])],
        fastq_dir=_org_fastq_dir,
        artifact_paths=_org_artifacts,
        where="run %s" % run_id)
    ORG = org_decision.profile
    HOMER_GENOME = _hg_literal or ORG.homer_genome
    MOTIF_SCAN_GTF = (cfg.pipeline.gtf_path or "").strip() or ORG.motif_scan_gtf
    # pipeline.gtf_path used to win unconditionally, with no organism check --
    # unlike its sibling homer_genome, which raises on a mismatch. Every run.toml
    # on disk sets it to the human GTF, so copying one as a mouse template
    # silently fed the whole motif-scan chain human gene models (2026-09-23).
    _gtf_explicit = (cfg.pipeline.gtf_path or "").strip()
    _gtf_species = species_of_path(_gtf_explicit) if _gtf_explicit else ""
    if _gtf_species and _gtf_species != ORG.key:
        raise ValueError(
            "run.toml disagrees with itself: this run resolved to %s (%s) but "
            "pipeline.gtf_path=%r is a %s annotation. The motif-scan chain "
            "(MinmaxBedNode -> combine_replicates_minmax_bed -> the TRN target "
            "tables) would read gene models from the wrong species while the "
            "ATAC peak annotation used the right ones. Drop gtf_path to take "
            "the organism's own GTF (%s), or point it at a %s annotation. A "
            "stale value copied from another run.toml is the usual cause."
            % (ORG.scientific_name, ORG.genome, _gtf_explicit,
               resolve_organism(_gtf_species).scientific_name,
               ORG.motif_scan_gtf, ORG.scientific_name))
    _org_missing = ORG.missing_assets()
    _org_rec = org_decision.as_dict()
    _org_rec["homer_genome"] = HOMER_GENOME
    _org_rec["motif_scan_gtf"] = MOTIF_SCAN_GTF
    _org_rec["profile"] = ORG.as_dict()
    _org_rec["missing_assets"] = _org_missing
    if _hg_literal:
        _org_rec["homer_genome_note"] = (
            "pipeline.homer_genome=%r is not a species MRAG knows; it is used "
            "verbatim as the HOMER/ROSE genome key." % _hg_literal)
    manifest.add_step({"name": "organism", "status": "ok",
                       "note": org_decision.note or org_decision.detail,
                       "outputs": [{"key": k, "value": v} for k, v in (
                           ("organism", ORG.key), ("genome", ORG.genome),
                           ("source", org_decision.source),
                           ("detail", org_decision.detail))],
                       "organism": _org_rec})
    manifest.write_atomic()
    print("=" * 66, flush=True)
    print("[organism] %s (%s) genome=%s  source=%s"
          % (ORG.scientific_name, ORG.key, ORG.genome, org_decision.source), flush=True)
    print("[organism] evidence: %s" % (org_decision.detail or "-"), flush=True)
    if org_decision.note:
        print("[organism] NOTE: %s" % org_decision.note, flush=True)
    if _org_missing:
        print("[organism] MISSING reference assets: %s" % ", ".join(_org_missing), flush=True)
    for _cap in ("giggle", "dastk_motifs", "curated_literature", "gene_info"):
        if not ORG.can(_cap):
            print("[organism] %s unavailable: %s" % (_cap, ORG.why_not(_cap)), flush=True)

    # ---- steps this organism cannot run --------------------------------
    # A capability gate that only prevents a step from running still leaves an
    # empty section in the report, and an empty section reads as "we looked and
    # found nothing". Every gate now registers here; the list goes into the
    # manifest, into the run log and into the report, so a skip is always
    # visible AS a skip.
    ORGANISM_SKIPS: list = []

    def _skip_for_organism(name: str, reason: str, section: str = "") -> None:
        rec = {"name": name, "reason": reason, "organism": ORG.key,
               "genome": ORG.genome}
        if section:
            rec["section"] = section
        ORGANISM_SKIPS.append(rec)
        manifest.add_step({"name": name, "status": "skipped",
                           "organism_gated": True, "note": reason,
                           "organism": ORG.key, "genome": ORG.genome})
        manifest.update_step("organism", organism_skipped=list(ORGANISM_SKIPS))
        manifest.write_atomic()
        print("[organism] SKIPPED %s for %s: %s"
              % (name, ORG.scientific_name, reason), flush=True)

    # ---- literature namespace -------------------------------------------
    # The evidence layer is a HUMAN index. A non-human run reaches it through
    # the ortholog map, and the map has to exist BEFORE the run spends hours on
    # alignment: without it every lookup misses and the misses read as "no
    # literature evidence", which is the most expensive kind of wrong.
    from .orthology import get_map as _get_ortho_map
    ORTHO = _get_ortho_map(ORG)
    if ORG.ortholog_to_human and not ORG.can("curated_literature"):
        raise FileNotFoundError(ORG.why_not("curated_literature"))
    if not ORTHO.identity:
        print("[organism] literature namespace: human, via %s (%d %s symbols "
              "mapped). Every GeneRIF / PubMed / PaperQA lookup runs on the "
              "HUMAN ortholog and the report discloses it per gene."
              % (ORTHO.source, len(ORTHO), ORG.common_name), flush=True)
    manifest.update_step("organism", ortholog={
        "identity": ORTHO.identity, "source": ORTHO.source,
        "table": ORTHO.table_path, "n_symbols": len(ORTHO),
        "banner": ORTHO.banner(ORG.scientific_name, ORG.genome)})
    manifest.write_atomic()
    print("=" * 66, flush=True)

    # 2026-09-28 pre-deploy review: the target scorer, checked after the config checks above
    # (they raise as before) and before the first compute step
    _ssg = _scorer_startup_gate(cfg, manifest)
    if _ssg is not None:
        return _ssg

    # ---- Step: combined (or provided) ----
    combined_node: Optional[CombinedNode] = None
    integrated_csv = None
    topn_dir = None
    if pre.integrated_csv and Path(pre.integrated_csv).exists():
        # Skip combined: user supplied INTEGRATED.csv directly. We still need
        # top{N}.json files; produce them inline.
        combined_dir = out_dir / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        # Copy INTEGRATED.csv into combined_dir + generate top{N}.json via pick_topn
        import shutil, sys
        sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/combined_mcp/src")
        from combined_mcp.tools.pipeline import register_pipeline_tools
        from .nodes._base import capture_mcp_tools
        tools = capture_mcp_tools(register_pipeline_tools)
        integ_local = combined_dir / "INTEGRATED.csv"
        if Path(pre.integrated_csv).resolve() != integ_local.resolve():
            shutil.copy(pre.integrated_csv, integ_local)
        import json as _json
        pick_res = _json.loads(tools["pick_topn"](
            str(integ_local), str(combined_dir), cfg.pipeline.topn_variants
        ))
        manifest.add_step({"name":"combined","status":"provided",
                           "outputs":[{"key":"integrated_csv","path":str(integ_local)}]})
        manifest.write_atomic()
        integrated_csv = str(integ_local)
        topn_dir = str(combined_dir)
    elif pre.atac_md_csv and pre.deseq_csv:
        combined_dir = out_dir / "combined"
        combined_node = CombinedNode(
            inputs={"atac_md_csv": pre.atac_md_csv, "rna_deseq_csv": pre.deseq_csv},
            params={"n_list": cfg.pipeline.topn_variants},
            output_dir=combined_dir,
        )
    else:
        # ---- Step 6c-1c/d: STAGE-AWARE PER-MODALITY PATH ----
        # Each modality resolves independently to its target artifact,
        # entering the DAG at whatever stage the input represents:
        # ATAC:  atac_md_csv (skip) | peaks BED -> DAStk | FASTQ -> full
        # RNA:   deseq_csv (skip) | counts/TPM matrix -> DESeq2 | FASTQ -> full
        # This lets mixed inputs (e.g. ATAC FASTQ + RNA TPM matrix) work.
        atac_samples = [s for s in cfg.samples if s.modality == "ATAC"]
        rna_samples  = [s for s in cfg.samples if s.modality == "RNA"]
        atac_ok = bool(pre.atac_md_csv
                       or (pre.atac_peaks_treatment and pre.atac_peaks_control)
                       or (pre.atac_count_matrix and pre.atac_group_tsv)
                       or atac_samples)
        rna_ok  = bool(pre.deseq_csv or pre.rna_counts_matrix
                       or (pre.rna_log2_matrix and pre.rna_group_tsv)
                       or rna_samples)
        if not (atac_ok and rna_ok):
            raise ValueError(
                "Cannot bootstrap pipeline. Each modality needs one input stage:\n"
                "  ATAC: precomputed.atac_md_csv OR "
                "atac_peaks_{treatment,control} OR ATAC FASTQ samples\n"
                "  RNA:  precomputed.deseq_csv OR rna_counts_matrix OR RNA FASTQ samples\n"
                "(Or supply precomputed.integrated_csv to skip both.)"
            )

        # ================= ATAC modality =================
        atac_md_csv = pre.atac_md_csv
        atac_matrix_canon_bed: Optional[str] = None
        if not atac_md_csv:
            treatment_files: list[str] = []
            control_files:   list[str] = []
            _atac_src = None
            if pre.atac_count_matrix and pre.atac_group_tsv:
                # Partial input: ATAC peak-count matrix + group TSV ->
                # edgeR/voom differential accessibility -> group-enriched
                # treatment/control peak BEDs -> DAStk MD-score.
                # MCP step 2a (2026-10-12): the diffbind-mcp tool voom_differential_accessibility
                from .nodes._stage_tools import matrix_stage as _matrix_stage
                run_atac_matrix_adapter = _matrix_stage("run_atac_matrix_adapter")
                atac_project_dir = str(out_dir / "atac_from_matrix")
                Path(atac_project_dir).mkdir(parents=True, exist_ok=True)
                _atk = cfg.conditions[0].label if (cfg.conditions and len(cfg.conditions) >= 2) else "treatment"
                _ack = cfg.conditions[1].label if (cfg.conditions and len(cfg.conditions) >= 2) else "control"
                _tbed, _cbed = run_atac_matrix_adapter(
                    pre.atac_count_matrix, pre.atac_group_tsv, atac_project_dir,
                    treat=_atk, ref=_ack)
                treatment_files = [_tbed]
                control_files   = [_cbed]
                # Keep per-arm differential BEDs for Section 8 (HOMER known):
                # the motif-scan step needs per-rep narrowPeaks (absent for matrix input),
                # so these serve as the findMotifsGenome inputs instead.
                _matrix_arm_beds = {_atk: _tbed, _ack: _cbed}
                # Canonical ATAC peak BED (full peak catalog) from the matrix
                # coordinates -> feeds the HOMER motif_tsv step (no FASTQ samples).
                from .adapters.flex_adapters import write_canonical_bed_from_matrix
                atac_matrix_canon_bed = write_canonical_bed_from_matrix(
                    pre.atac_count_matrix,
                    str(Path(atac_project_dir) / "canonical_peaks.bed"))
                # The voom adapter also emits a DiffBind-format CSV (per-peak
                # Fold/p.value/FDR) -> satisfy the DiffBind step (Section 9
                # GIGGLE + volcano) without per-sample BAMs/peaks.
                _db_from_matrix = str(Path(atac_project_dir) / "DiffBind_out.csv")
                if Path(_db_from_matrix).exists() and not pre.diffbind_out_csv:
                    pre.diffbind_out_csv = _db_from_matrix
            elif pre.atac_peaks_treatment and pre.atac_peaks_control:
                # Partial input: per-arm peaks BED/narrowPeak -> DAStk MD-score only.
                treatment_files = [str(x) for x in pre.atac_peaks_treatment]
                control_files   = [str(x) for x in pre.atac_peaks_control]
                atac_project_dir = str(out_dir / "atac_from_peaks")
                Path(atac_project_dir).mkdir(parents=True, exist_ok=True)
            else:
                # Full input: ATAC FASTQ -> preprocessing -> narrowPeak.filt per arm.
                _atac_src = str(Path(atac_samples[0].fastq_r1).parent)
                _atac_before = _tree_state(_atac_src)
                atac_project_dir = _stage_project_dir(_atac_src, out_dir / "atac_project", out_dir)
                atac_node = ATACNode(
                    inputs={"project_dir": atac_project_dir},
                    params={"num_threads": 0,
                            "ref_dir":  ORG.atac_ref_dir,
                            "genome":   ORG.genome,
                            "species":  ORG.macs2_species,
                            "gtf_file": ORG.peak_annotation_gtf},
                    output_dir=Path(atac_project_dir),
                )
                atac_node.execute()
                manifest.add_step(atac_node.to_manifest()); manifest.write_atomic()
                if atac_node.status != "ok":
                    return {"ok": False, "stage": "atac", "manifest": str(manifest.path)}
                atac_peaks_by_arm: dict[str, list[str]] = {}
                for s in atac_samples:
                    pk = Path(atac_project_dir) / f"{s.name}_peaks.narrowPeak.filt"
                    if pk.exists():
                        atac_peaks_by_arm.setdefault(s.arm or "unknown", []).append(str(pk))
                arms_present = set(atac_peaks_by_arm.keys())
                if len(arms_present) != 2:
                    raise ValueError(f"DAStk needs exactly 2 ATAC arms, got: {arms_present}")
                if cfg.conditions and len(cfg.conditions) >= 2:
                    treatment_arm = cfg.conditions[0].label
                    control_arm   = cfg.conditions[1].label
                else:
                    _sorted = sorted(arms_present)
                    treatment_arm, control_arm = _sorted[0], _sorted[1]
                if treatment_arm not in arms_present or control_arm not in arms_present:
                    raise ValueError(
                        f"DAStk arm mismatch: conditions=[{treatment_arm},{control_arm}], "
                        f"data has {arms_present}"
                    )
                treatment_files = atac_peaks_by_arm[treatment_arm]
                control_files   = atac_peaks_by_arm[control_arm]

            # DAStk scores accessibility against a directory of genome-wide
            # motif-hit BEDs. There is one such set per assembly, so a missing
            # mouse set must stop the run here: the alternative is an
            # INTEGRATED.csv built from human motif coordinates.
            if not ORG.can("dastk_motifs"):
                raise FileNotFoundError(ORG.why_not("dastk_motifs"))
            dastk_node = DAStkNode(
                inputs={
                    "project_dir":     atac_project_dir,
                    "treatment_files": treatment_files,
                    "control_files":   control_files,
                },
                params={"num_threads": 0,
                        "ref_dir":   ORG.atac_ref_dir,
                        "genome":    ORG.dastk_genome,
                        "motif_dir": ORG.dastk_motif_dir},
                output_dir=Path(atac_project_dir),
            )
            # MRAG_REUSE_ATAC_MD_CSV: reuse an existing MD-score file instead of
            # re-running DAStk.  Before the bootstrap was seeded per motif (dastk_seeded.py) a re-run
            # could reorder TFs at the top-N boundary, so comparison reruns pinned it (2026-09-16).
            _reuse_md = os.environ.get("MRAG_REUSE_ATAC_MD_CSV", "")
            if _reuse_md and Path(_reuse_md).is_file():
                dastk_node.provided_outputs = {"atac_md_csv": _reuse_md}
                print(f"[dastk] reusing existing MD scores: {_reuse_md}", flush=True)
            dastk_node.execute()
            manifest.add_step(dastk_node.to_manifest()); manifest.write_atomic()
            if dastk_node.status not in ("ok", "provided"):
                return {"ok": False, "stage": "dastk", "manifest": str(manifest.path)}
            if _atac_src:
                _chg = _tree_changed(_atac_src, _atac_before)
                if _chg:
                    print(f"[stage] ATAC steps wrote into the input dir {_atac_src}: {_chg[:10]}", flush=True)
                    return {"ok": False, "stage": "input_write_guard", "manifest": str(manifest.path)}
            atac_md_csv = dastk_node.outputs["atac_md_csv"]

        # ================= RNA modality =================
        deseq_csv = pre.deseq_csv
        if not deseq_csv:
            if pre.rna_log2_matrix and pre.rna_group_tsv:
                # Partial input: RNA log2/normalized matrix + group TSV ->
                # limma group differential expression -> DESeq2.gene.csv.
                # MCP step 2a (2026-10-12): the rna-mcp tool limma_group_de
                from .nodes._stage_tools import matrix_stage as _matrix_stage
                run_rna_log2_adapter = _matrix_stage("run_rna_log2_adapter")
                rna_project_dir = str(out_dir / "rna_from_log2")
                Path(rna_project_dir).mkdir(parents=True, exist_ok=True)
                _rtk = cfg.conditions[0].label if (cfg.conditions and len(cfg.conditions) >= 2) else "treatment"
                _rck = cfg.conditions[1].label if (cfg.conditions and len(cfg.conditions) >= 2) else "control"
                _out_csv = str(Path(rna_project_dir) / "DESeq2.gene.csv")
                deseq_csv = run_rna_log2_adapter(
                    pre.rna_log2_matrix, pre.rna_group_tsv, _out_csv,
                    treat=_rtk, ref=_rck)
            elif pre.rna_counts_matrix:
                # Partial input: raw counts/TPM matrix -> DESeq2 only. Mirrors
                # 1.DESeq2.R (counts) / 1.DESeq2_TPM.R (tpm) -- both round() the
                # matrix inside DESeq2, which _run_deseq2_r reproduces exactly.
                rna_project_dir = str(out_dir / "rna_from_matrix")
                Path(rna_project_dir).mkdir(parents=True, exist_ok=True)
                matrix_tsv, mapfile = _prepare_rna_matrix(
                    pre.rna_counts_matrix, cfg, rna_project_dir,
                    tx2gene_path=ORG.tx2gene)
                deseq_node = DESeq2Node(
                    inputs={"project_dir": rna_project_dir, "mapfile": str(mapfile),
                            "reference": "untreated", "counts_file": matrix_tsv},
                    params={"threads": 8, "quant_source": pre.rna_quant_source,
                            "gene_annotation": ORG.gene_annotation_tsv,
                            "organism": ORG.key},
                    output_dir=Path(rna_project_dir),
                )
                deseq_node.execute()
                manifest.add_step(deseq_node.to_manifest()); manifest.write_atomic()
                if deseq_node.status != "ok":
                    return {"ok": False, "stage": "deseq2", "manifest": str(manifest.path)}
                deseq_csv = deseq_node.outputs["deseq_csv"]
            else:
                # Full input: RNA FASTQ -> preprocessing -> DESeq2 (counts).
                _rna_src = str(Path(rna_samples[0].fastq_r1).parent)
                _rna_before = _tree_state(_rna_src)
                rna_project_dir = _stage_project_dir(_rna_src, out_dir / "rna_project", out_dir)
                rna_node = RNANode(
                    inputs={"project_dir": rna_project_dir},
                    params={"num_threads": 0,
                            "ref_dir": ORG.rna_ref_dir,
                            "genome":  ORG.rsem_prefix},
                    output_dir=Path(rna_project_dir),
                )
                rna_node.execute()
                manifest.add_step(rna_node.to_manifest()); manifest.write_atomic()
                if rna_node.status != "ok":
                    return {"ok": False, "stage": "rna", "manifest": str(manifest.path)}
                import re as _re
                def _strip_pair_suffix(n: str) -> str:
                    return _re.sub(r"(_|\.)([12]|R[12])$", "", n)
                treated_arm = cfg.conditions[0].label if (cfg.conditions and len(cfg.conditions) >= 1) else ""
                mapfile = Path(rna_project_dir) / "sample_mapfile.csv"
                if not mapfile.exists():
                    seen = set()
                    lines = ["sample\tTREATMENT"]
                    for s in rna_samples:
                        clean = _strip_pair_suffix(s.name)
                        if clean in seen: continue
                        seen.add(clean)
                        treat_label = "treated" if (s.arm or "") == treated_arm else "untreated"
                        lines.append(f"{clean}\t{treat_label}")
                    mapfile.write_text("\n".join(lines) + "\n")
                deseq_node = DESeq2Node(
                    inputs={"project_dir": rna_project_dir, "mapfile": str(mapfile),
                            "reference": "untreated"},
                    params={"threads": 8,
                            "gene_annotation": ORG.gene_annotation_tsv,
                            "organism": ORG.key},
                    output_dir=Path(rna_project_dir),
                )
                deseq_node.execute()
                manifest.add_step(deseq_node.to_manifest()); manifest.write_atomic()
                if deseq_node.status != "ok":
                    return {"ok": False, "stage": "deseq2", "manifest": str(manifest.path)}
                deseq_csv = deseq_node.outputs["deseq_csv"]
                _chg = _tree_changed(_rna_src, _rna_before)
                if _chg:
                    print(f"[stage] RNA steps wrote into the input dir {_rna_src}: {_chg[:10]}", flush=True)
                    return {"ok": False, "stage": "input_write_guard", "manifest": str(manifest.path)}

        # Now have atac_md_csv + deseq_csv -> feed into combined node
        combined_dir = out_dir / "combined"
        combined_node = CombinedNode(
            inputs={"atac_md_csv": atac_md_csv, "rna_deseq_csv": deseq_csv},
            params={"n_list": cfg.pipeline.topn_variants,
                    "organism": ORG.key},
            output_dir=combined_dir,
        )
        # Also push these into pre for the downstream steps
        pre.atac_md_csv = atac_md_csv
        pre.deseq_csv = deseq_csv

    pl_steps = [combined_node] if combined_node is not None else []

    # ---- Step: diffbind ----
    diffbind_provided: Optional[str] = pre.diffbind_out_csv

    # ---- Step: homer (motif TSV path) ----
    motif_tsv: Optional[str] = pre.motif_tsv

    # Execute the deterministic head (combined first)
    if pl_steps:
        p1 = Pipeline(pl_steps, manifest, stop_on_fail=True, cache=cache)
        head_summary = p1.run()
        if head_summary["status"] == "fail":
            return {"ok": False, "stage": "combined", "manifest": str(manifest.path)}
        integrated_csv = combined_node.outputs.get("integrated_csv","")
        topn_dir = str(Path(integrated_csv).parent) if integrated_csv else None

    # ---- Step 6c-1a: Generate canonical motif_tsv via HomerNode (FULL JASPAR-ext library)
    # 2026-06-07 ANALYST DECISION: HOMER annotatePeaks scores depend on the
    # full motif library background distribution. A top-N subset CHANGES
    # individual motif scores vs. the full scan. The orchestrator therefore
    # ALWAYS runs the full library (~2322 PWMs, HOCOMOCOv12 + JASPAR2024)
    # to preserve reproducibility against the reference analysis's gold values.
    if not motif_tsv and locals().get("atac_matrix_canon_bed"):
        # Matrix input path: no FASTQ ATAC samples exist. Annotate the canonical
        # peak catalog (built from the count-matrix coordinates) with the full
        # JASPAR-extended library to produce the motif_tsv.
        homer_full_node = HomerNode(
            inputs={"sample_id": "atac_matrix", "input_bed": atac_matrix_canon_bed},
            params={"genome": HOMER_GENOME, "use_jaspar": True,
                    "threads": 8, "timeout_s": 7200},
            output_dir=out_dir / "homer_canonical",
        )
        # Reuse a complete annotation from an earlier run of this output dir
        # (same rule as the motif-scan chain: file present with >= 500 rows).
        # run_homer_pipeline always re-annotates, which costs hours on the full
        # catalog and truncates the file while it runs (2026-09-16).
        _canon_tsv = out_dir / "homer_canonical" / "atac_matrix.motif.tsv"
        if not _canon_tsv.exists():
            try:
                _seed_canonical_from_cache(atac_matrix_canon_bed, out_dir / "homer_canonical",
                                           HOMER_GENOME)
            except Exception as _e:
                print(f"[homer_canonical] cache lookup skipped: {_e}", flush=True)
        if _canon_tsv.is_file() and _canon_tsv.stat().st_size > 1000:
            with open(_canon_tsv) as _cf:
                if sum(1 for _ in zip(range(501), _cf)) >= 500:
                    homer_full_node.provided_outputs = {"motif_tsv": str(_canon_tsv)}
                    print(f"[homer_canonical] reusing existing annotation: {_canon_tsv}", flush=True)
        homer_full_node.execute()
        manifest.add_step(homer_full_node.to_manifest())
        manifest.write_atomic()
        if homer_full_node.status not in ("ok", "provided"):
            return {"ok": False, "stage": "homer_canonical",
                    "manifest": str(manifest.path)}
        if homer_full_node.status == "ok":
            try:
                _store_canonical_in_cache(atac_matrix_canon_bed, out_dir / "homer_canonical",
                                          HOMER_GENOME)
            except Exception as _e:
                print(f"[homer_canonical] cache store skipped: {_e}", flush=True)
        motif_tsv = homer_full_node.outputs["motif_tsv"]

    if not motif_tsv:
        atac_samples = [s for s in cfg.samples if s.modality == "ATAC"]
        if not atac_samples:
            raise ValueError("HomerNode needs a canonical ATAC BED file. "
                             "Supply precomputed.motif_tsv OR have ATAC samples "
                             "with {sample}.input.bed / .narrowPeak.filt files.")
        # Canonical MRAG motif_tsv = the 2-replicate per-arm min-max CONSENSUS
        # (canonical). Annotating a single replicate makes the downstream
        # motif-derived results non-reproducible (1bp anchor sensitivity), so we
        # build the consensus whenever every arm has >=2 ATAC replicates and
        # only degrade to single-rep (with a loud warning) when we cannot.
        from .runner_motif_scan import build_minmax_motifs_by_arm, _arm_samples
        arm_labels = [c.label for c in cfg.conditions] if cfg.conditions else []
        all_arms_ge2 = (len(arm_labels) >= 2 and
                        all(len(_arm_samples(cfg, a)) >= 2 for a in arm_labels))
        if all_arms_ge2:
            mm = build_minmax_motifs_by_arm(cfg, out_dir, manifest,
                                            gtf_path=MOTIF_SCAN_GTF,
                                            genome=HOMER_GENOME)
            # Downstream filter_target_tfs keeps only top-TF columns regardless
            # of arm; the treatment (first) arm's consensus is the canonical
            # motif_tsv, matching the OLD [precomputed] wiring.
            motif_tsv = mm["minmax_motif_tsv_by_arm"][arm_labels[0]]
        else:
            s0 = atac_samples[0]
            print("=" * 78, flush=True)
            print("WARNING [runner 6c-1a]: <2 ATAC replicates per arm -> cannot "
                  "build min-max consensus. Falling back to SINGLE-REPLICATE "
                  f"HOMER annotate on '{s0.name}'. The canonical MRAG motif_tsv "
                  "is the 2-replicate min-max consensus; single-rep peaks yield "
                  "NON-reproducible motif-derived results (1bp anchor sensitivity). "
                  "Provide >=2 reps per arm OR a precomputed minmax motif_tsv.",
                  flush=True)
            print("=" * 78, flush=True)
            canon_bed = ""
            proj_dir = str(_sample_dir(s0))
            for cand in [Path(proj_dir) / f"{s0.name}.input.bed",
                         Path(proj_dir) / f"{s0.name}_merge.input.bed",
                         Path(proj_dir) / f"{s0.name}_peaks.narrowPeak.filt"]:
                if cand.exists():
                    canon_bed = str(cand); break
            if not canon_bed:
                raise ValueError("HomerNode needs a canonical ATAC BED file. "
                                 "Supply precomputed.motif_tsv OR have ATAC "
                                 "samples with {sample}.input.bed / "
                                 ".narrowPeak.filt files.")
            homer_full_node = HomerNode(
                inputs={"sample_id": s0.name, "input_bed": canon_bed},
                params={
                    "genome":     HOMER_GENOME,
                    "use_jaspar": True,
                    "threads":    8,      # per-task HOMER threads (annotatePeaks -cpu)
                    "timeout_s":  7200,
                },
                output_dir=out_dir / "homer_canonical",
            )
            homer_full_node.execute()
            manifest.add_step(homer_full_node.to_manifest())
            manifest.write_atomic()
            if homer_full_node.status != "ok":
                return {"ok": False, "stage": "homer_canonical",
                        "manifest": str(manifest.path)}
            motif_tsv = homer_full_node.outputs["motif_tsv"]

    # ---- Step 6c-1b: DiffBind from ATAC BAMs + peaks if not provided
    if not diffbind_provided:
        # Build samples list from cfg.samples ATAC modality
        atac_samples = [s for s in cfg.samples if s.modality == "ATAC"]
        if not atac_samples:
            raise ValueError("DiffBind needs ATAC samples in cfg or precomputed.diffbind_out_csv.")
        diff_samples = []
        for s in atac_samples:
            proj = _sample_dir(s)
            bam_candidates = [
                proj / "alignments" / f"{s.name}.dedup.bam",
                proj / "alignments" / f"{s.name}.sorted.dedup.bam",
            ]
            peak_candidates = [
                proj / f"{s.name}_peaks.narrowPeak.filt",
                proj / f"{s.name}_peaks.narrowPeak",
            ]
            bam = next((str(b) for b in bam_candidates if b.exists()), None)
            peak = next((str(p) for p in peak_candidates if p.exists()), None)
            if not bam or not peak:
                raise ValueError(
                    f"DiffBind: missing BAM or peak for ATAC sample {s.name}. "
                    f"Looked at: {bam_candidates}, {peak_candidates}"
                )
            diff_samples.append({
                "SampleID":   s.name,
                "Condition":  s.arm or "unknown",
                "Replicate":  s.replicate,
                "bamReads":   bam,
                "Peaks":      peak,
            })
        # Control/baseline = second condition (treatment = first), mirroring
        # the DESeq2 reference convention so positive Fold = up in treatment.
        _db_reference = (cfg.conditions[1].label
                         if (cfg.conditions and len(cfg.conditions) >= 2) else None)
        diffbind_node = DiffbindNode(
            inputs={"samples": diff_samples},
            params={"threads": 8, "timeout_s": 7200, "reference": _db_reference,
                    "default_tissue": ORG.diffbind_tissue},
            output_dir=out_dir / "diffbind",
        )
        diffbind_node.execute()
        manifest.add_step(diffbind_node.to_manifest())
        manifest.write_atomic()
        if diffbind_node.status != "ok":
            return {"ok": False, "stage": "diffbind",
                    "manifest": str(manifest.path)}
        diffbind_provided = diffbind_node.outputs["diffbind_csv"]
        pre.diffbind_out_csv = diffbind_provided

    # ---- Step 6e:  canonical Step 2 chain (DEFAULT, 2026-06-04) ----
    # Runs the reference 5-stage pipeline (per-rep HOMER scan → MinmaxBed →
    # minmax HOMER scan → 10-TF filter → DiffBind+RNA join) and produces
    # per-condition TFs.Pval2.txt files consumed by the Phase C R plots.
    # Cached: re-uses any *.motif.tsv files already on disk under
    # out_dir/homer_motif/. Set cfg.pipeline.skip_motif_scan=True to disable.
    motif_scan_outputs: dict = {}
    # The Step-2 chain builds per-arm min-max consensus from PER-REPLICATE
    # ATAC narrowPeaks (cfg.samples, i.e. FASTQ/BAM runs) OR a pre-seeded
    # min-max preset under out_dir/homer_motif/. Processed-matrix input (TCGA-ATAC:
    # count matrix + group TSV, no cfg.samples) has NEITHER, so the chain would
    # raise "needs >=2 replicates per arm; arm=... has 0". That is NOT a failure:
    # matrix runs cover the same sections via fallbacks (Section 8 uses the
    # DiffBind arm-specific BEDs _matrix_arm_beds; Section 10 uses the canonical
    # JASPAR motif_tsv; Section 6 uses the matrix peak-heatmap R path over the
    # per-rep bigWigs). So skip the chain cleanly with status "skipped" instead
    # of attempting it and recording a scary (and meaningless) "fail".
    _atac_reps = len([s for s in (cfg.samples or [])
                      if getattr(s, "modality", "") == "ATAC"])
    _homer_motif_root = Path(out_dir) / "homer_motif"
    _motif_seeded = bool(cfg.conditions) and len(cfg.conditions) >= 2 and all(
        (_homer_motif_root / f"{c.label}_peak_minmax.motif.tsv").is_file()
        and (_homer_motif_root / f"{c.label}_peak_minmax.motif.tsv").stat().st_size
            > 100_000_000
        for c in cfg.conditions[:2])
    _motif_can_run = _atac_reps >= 2 or _motif_seeded
    if (not getattr(cfg.pipeline, "skip_motif_scan", False) and diffbind_provided
            and not _motif_can_run):
        manifest.add_step({
            "name": "motif_scan_chain", "status": "skipped",
            "note": ("processed-matrix input has no per-replicate ATAC "
                     "narrowPeaks and no seeded min-max preset; Sections 6/8/10 "
                     "are produced from matrix fallbacks (DiffBind arm BEDs + "
                     "canonical JASPAR motif_tsv + matrix peak-heatmap)."),
        })
        manifest.write_atomic()
    if (not getattr(cfg.pipeline, "skip_motif_scan", False) and diffbind_provided
            and _motif_can_run):
        try:
            from .runner_motif_scan import run_motif_scan_chain
            # 2026-09-10: filter Step-2 motif columns by THIS run's canonical
            # top-N panel (combined/top{N}.json), not the reference analysis's
            # 2026 default TF list (taken from an older motif DB).
            _default_target_tfs = None
            try:
                _tj = (Path(topn_dir) / f"top{cfg.canonical_N}.json") if topn_dir else None
                if _tj is not None and _tj.is_file():
                    _default_target_tfs = [str(t) for t in json.loads(_tj.read_text()).get("tfs", [])] or None
            except Exception:
                _default_target_tfs = None
            motif_scan_outputs = run_motif_scan_chain(
                cfg=cfg, pre=pre,
                diffbind_csv=diffbind_provided,
                out_dir=out_dir, manifest=manifest,
                target_tfs=_default_target_tfs,
                gtf_path=MOTIF_SCAN_GTF,
                genome=HOMER_GENOME,
                use_jaspar=True,    # default ON: append JASPAR-extended lib so
                                    # top-N TFs (JASPAR-only ones too) all get
                                    # motif columns → Sections 6 + 8 cover all
            )
            manifest.add_step({
                "name": "motif_scan_chain", "status": "ok",
                "outputs": [{"key": k, "value": v}
                            for k, v in motif_scan_outputs.items()],
            })
            manifest.write_atomic()
        except Exception as _ex:
            manifest.add_step({
                "name": "motif_scan_chain", "status": "fail",
                "error": str(_ex)[:500],
            })
            manifest.write_atomic()
            # Non-fatal: the motif-scan step is an addition; existing variants/viz still run.

    # ---- Step 6e-2: HOMER known-motif enrichment per condition (Section 8) ----
    # Runs findMotifsGenome on each arm's min-max-scaled consensus BED so the
    # knownResults shown in the report are a natural full-run output (mirrors
    # 2.findmotif_homer.py). Skipped gracefully if minmax BEDs unavailable.
    homer_known_params: dict = {}
    _mm = (motif_scan_outputs or {}).get("minmax_bed_by_arm", {}) or {}
    if len(_mm) != 2:
        # Processed-matrix input: the motif-scan step (per-rep narrowPeaks) unavailable,
        # but the voom adapter emitted per-arm differential BEDs -> use those
        # so Section 8 still populates from findMotifsGenome.
        _mab = locals().get("_matrix_arm_beds", {}) or {}
        if len(_mab) == 2:
            _mm = _mab
    if len(_mm) == 2:
        try:
            from .nodes.homer import HomerKnownMotifsNode
            _arms = list(_mm.keys())
            _cond_order = [c.label for c in cfg.conditions] if cfg.conditions else _arms
            _ordered = sorted(_arms,
                              key=lambda a: _cond_order.index(a) if a in _cond_order else 99)
            _a_lab, _b_lab = _ordered[0], _ordered[1]
            # Section 8 enriches motifs on CONDITION-SPECIFIC peaks (the
            # peak-heatmap A-only / B-only split), not on each arm's min-max
            # consensus. The consensus is that arm's own replicate union and is
            # dominated by peaks open in BOTH arms, which made the two arms
            # return an identical motif ranking. The heatmap step that normally
            # builds the split runs later, so build it here (it regenerates the
            # same files when it gets there) and fall back to the consensus if
            # the per-replicate narrowPeaks are unavailable.
            _hk_a_bed, _hk_b_bed = _mm[_a_lab], _mm[_b_lab]
            try:
                from .nodes.peakheatmap import build_specific_beds
                _spec_dir = Path(out_dir) / "peakheatmap"
                _spec_ready = {
                    lab: _spec_dir / f"{lab}_specific.bed" for lab in (_a_lab, _b_lab)}
                if all(p.is_file() and p.stat().st_size > 0
                       for p in _spec_ready.values()):
                    _spec = {k: str(v) for k, v in _spec_ready.items()}
                else:
                    # Per-replicate narrowPeaks, same rule the peak-heatmap step
                    # uses further down (it derives _ph_np* after this point).
                    _hk_by_arm: dict[str, list[str]] = {}
                    for _s in (cfg.samples or []):
                        if getattr(_s, "modality", "") != "ATAC":
                            continue
                        _npk = (_sample_dir(_s)
                                / f"{_s.name}_peaks.narrowPeak.filt")
                        if _npk.is_file():
                            _hk_by_arm.setdefault(_s.arm or "unknown", []).append(str(_npk))
                    _spec = build_specific_beds(_hk_by_arm.get(_a_lab, []),
                                                _hk_by_arm.get(_b_lab, []),
                                                _a_lab, _b_lab, str(_spec_dir))
                if _spec.get(_a_lab) and _spec.get(_b_lab):
                    _hk_a_bed, _hk_b_bed = _spec[_a_lab], _spec[_b_lab]
                    print("[homer_known] condition-specific peaks: %s / %s"
                          % (_hk_a_bed, _hk_b_bed), flush=True)
                else:
                    print("[homer_known] specific-BED split unavailable; "
                          "falling back to min-max consensus BEDs", flush=True)
            except Exception as _spx:
                print("[homer_known] specific-BED split failed (%s); "
                      "falling back to min-max consensus" % str(_spx)[:200], flush=True)

            _hk = HomerKnownMotifsNode(
                inputs={"arm_a_bed": _hk_a_bed, "arm_b_bed": _hk_b_bed,
                        "arm_a_label": _a_lab, "arm_b_label": _b_lab},
                params={"genome": HOMER_GENOME,
                        "threads": max(1, min(16, _ncpus() // 2))},   # 2 arms in parallel
                output_dir=str(Path(out_dir) / "homer_known"))
            _hk.execute()
            manifest.add_step(_hk.to_manifest()); manifest.write_atomic()
            if _hk.status == "ok":
                homer_known_params = {
                    "homer_known_a_dir":   _hk.outputs["homer_known_a_dir"],
                    "homer_known_b_dir":   _hk.outputs["homer_known_b_dir"],
                    "homer_known_a_label": _a_lab,
                    "homer_known_b_label": _b_lab,
                }
        except Exception as _ex:
            manifest.add_step({"name": "homer_known_motifs", "status": "fail",
                               "error": str(_ex)[:500]})
            manifest.write_atomic()

    # ---- Step 6e-3: GIGGLE Cistrome TF-overlap plots (Section 9) ----
    # Searches DiffBind differential peaks against the permanent Cistrome hg38
    # TF ChIP-seq GIGGLE index (sample-independent reference). Produces the
    # up/down point plots. Skipped gracefully if DiffBind/index unavailable.
    # 2026-08-05: GIGGLE *search* = external Cistrome DB web toolkit
    # (dbtoolkit.cistrome.org) on the condition-open peaks; the agent PLOTS the
    # analyst-provided result.csv (GSM_ID,Factor,Biosource,GIGGLE_score) per
    # the reference 3.plot.R. Drop the two exports at
    # <out_dir>/giggle/<treat_label>_open.bed_result.csv  (+ <control_label>)
    # Skipped gracefully if absent.
    # 2026-09-24: the index is chosen by the run's assembly (ORG.genome ->
    # run_giggle.sh 7th arg): hg38 -> giggle_index_hg38_tf, mm10 ->
    # giggle_index_mm10_tf. ORG.can("giggle") checks those exact paths.
    giggle_params: dict = {}
    _gg_dir = Path(out_dir) / "giggle"
    _gg_tlab = cfg.conditions[0].label if cfg.conditions else "treatment"
    _gg_clab = (cfg.conditions[1].label
                if (cfg.conditions and len(cfg.conditions) >= 2) else "control")
    _gg_tbed = Path(out_dir) / "peakheatmap" / f"{_gg_tlab}_specific.bed"
    _gg_cbed = Path(out_dir) / "peakheatmap" / f"{_gg_clab}_specific.bed"
    if not ORG.can("giggle"):
        _skip_for_organism("giggle", ORG.why_not("giggle"),
                           section="9. GIGGLE - TF ChIP-seq Overlap (Cistrome DB)")
        manifest.write_atomic()
    elif _gg_tbed.is_file() and _gg_cbed.is_file():
        try:
            from .nodes.giggle import GiggleNode
            _gg = GiggleNode(
                inputs={"treat_bed": str(_gg_tbed),
                        "control_bed": str(_gg_cbed)},
                params={"treat_label": _gg_tlab, "control_label": _gg_clab,
                        "top_n": 200, "genome": ORG.genome},
                output_dir=str(Path(out_dir) / "giggle"))
            _gg.execute()
            manifest.add_step(_gg.to_manifest()); manifest.write_atomic()
            if _gg.status == "ok":
                giggle_params = {
                    "giggle_up_png":     _gg.outputs["giggle_up_png"],
                    "giggle_down_png":   _gg.outputs["giggle_down_png"],
                    "giggle_up_label":   f"{_gg_tlab} open peaks",
                    "giggle_down_label": f"{_gg_clab} open peaks",
                }
        except Exception as _ex:
            manifest.add_step({"name": "giggle", "status": "fail",
                               "error": str(_ex)[:500]})
            manifest.write_atomic()
    else:
        manifest.add_step({"name": "giggle", "status": "skipped",
                           "outputs": [{"key": "reason", "value":
                               "condition-specific beds not in peakheatmap/"}]})
        manifest.write_atomic()

    # ---- Step: variants fanout ----
    shared = {
        "topn_dir":              topn_dir,
        "motif_tsv":             motif_tsv,
        "diffbind_csv":          diffbind_provided,
        "deseq_csv":             pre.deseq_csv,
        "tfs_summary_json":      pre.tfs_summary_json or "",
        "targets_summary_json":  pre.targets_summary_json or "",
        "conditions_meta_json":  pre.conditions_meta_json or "",
    }
    # Auto-derive conditions_meta if not precomputed (from sample sheet conditions)
    if not shared["conditions_meta_json"]:
        cm = derive_conditions_meta(
            cfg.conditions if cfg.conditions else cfg.samples,
        )
        cm_path = out_dir / "conditions_meta.json"
        import json as _json
        cm_path.write_text(_json.dumps(cm, indent=2))
        shared["conditions_meta_json"] = str(cm_path)

    # tfs_summary + targets_summary are auto-derived per-branch in variants.py
    # using the integrated_csv we just made (combined.outputs).
    shared["integrated_csv"] = integrated_csv
    # Biological domain (user-supplied via chat/confirm). Feeds the LLM
    # narrative field + Enrichr context + AlphaGenome tissue. Falls back
    # to prostate/PC-3 (the historical MRAG default) when unset.
    # No tissue default.  This string reaches the enrichment term filter as the dataset's
    # identity, and defaulting it to prostate told every unconfigured run it was prostate data.
    # Downstream readers already handle an empty field.
    _dom_field = (cfg.cancer_type or cfg.tissue or "")
    from .config import resolve_ag_cell_line as _resolve_agcl
    # The "PC-3" fallback is a HUMAN prostate cancer cell line. Applying it on a
    # non-human run sent a human ontology curie to AlphaGenome with
    # organism="mouse" (rejected: "Unsupported ontology"), and also described
    # the study to the LLM as a PC-3 experiment (2026-09-23). Non-human runs get
    # no invented cell line; Section 13 resolves a real biosample below or is
    # skipped with its reason recorded.
    # resolve_ag_cell_line() is a HUMAN cell-line map end to end: "liver" ->
    # HepG2, "prostate" -> PC-3. Consulting it at all on a non-human run
    # invents a human cell line, so it is skipped rather than just its default.
    # An explicitly configured cfg.cell_line is still honoured -- that is the
    # user's statement, not an inference.
    _dom_cell_line = (cfg.cell_line or (
        (_resolve_agcl(cfg.cancer_type) or _resolve_agcl(cfg.tissue) or "PC-3")
        if ORG.is_human else ""))
    _dom_tissue = (cfg.tissue or "prostate")
    shared["field"] = _dom_field
    # Authoritative study domain for the narrative / executive-summary prompts
    # (2026-09-16 narrtf): user-supplied values only, never the fallback above.
    shared["study_cancer_type"] = cfg.cancer_type or ""
    shared["study_tissue"] = cfg.tissue or ""
    # PaperQA corpora for target-gene evidence (2026-09-15): explicit domains,
    # else the cancer_type text (may name several cancers), else prostate.
    from .literature import (resolve_literature_domains as _resolve_lit, describe as _describe_lit,
                             user_domains as _user_lit, domains_label as _label_lit)
    _lit_domains, _lit_unmatched = _resolve_lit(getattr(cfg, "literature_domains", None) or None,
                                                cfg.cancer_type, cfg.tissue)
    shared["literature_domains"] = _lit_domains
    # 2026-09-17 (llmdomain): the user's own domains, in the user's order, without
    # the auto-appended pan-cancer corpus -> Section 2 "LLM domain" row and the
    # numbered per-domain Novelty blocks of the Section 14 cards.
    _lit_user = _user_lit(_lit_domains, getattr(cfg, "literature_domains", None) or None)
    _lit_user_label = _label_lit(_lit_user, include_pancancer=True)
    shared["literature_domains_user"] = _lit_user
    # Section-14 card retrieval is a HUMAN index; a non-human run reaches it
    # through the ortholog map, and the card discloses which human gene each
    # statement is about.
    shared["organism"] = ORG.key
    print(f"[literature] corpora for target evidence: {_lit_domains}"
          + (f" (unmatched: {_lit_unmatched})" if _lit_unmatched else "")
          + f" | LLM domain: {_lit_user_label or '(not specified)'}", flush=True)
    manifest.add_step({"name": "literature_domains", "status": "ok",
                       "params": {"explicit": list(getattr(cfg, "literature_domains", []) or []),
                                  "cancer_type": cfg.cancer_type, "unmatched": _lit_unmatched},
                       "outputs": [{"key": "literature_domains", "value": ",".join(_lit_domains)},
                                   {"key": "literature_domains_label", "value": _describe_lit(_lit_domains)},
                                   {"key": "literature_domains_user", "value": ",".join(_lit_user)},
                                   {"key": "literature_domains_display", "value": _lit_user_label}]})

    # Sanity check
    if not shared["conditions_meta_json"]:
        raise ValueError("conditions_meta_json missing (auto-derive failed)")

    branches_dir = out_dir / "branches"
    variant_results = run_all_variants(
        shared=shared, output_dir=branches_dir,
        Ns=cfg.pipeline.topn_variants,
        llm_endpoint=cfg.llm.endpoint, llm_model=cfg.llm.model,
        parallel=cfg.pipeline.parallel, manifest=manifest,
        run_narrative=True, cache=cache,
    )

    # ---- Auto-derive bigwig paths from ATAC samples if not pre-supplied ----
    # This is the canonical wiring for full FASTQ runs — the ATAC pipeline
    # writes {sample}.bw next to the input FASTQs.
    if (not pre.bigwig_arm_a_paths or not pre.bigwig_arm_b_paths) and cfg.samples:
        atac_by_arm: dict[str, list[str]] = {}
        for s in cfg.samples:
            if s.modality != "ATAC":
                continue
            bw = _sample_dir(s) / f"{s.name}.bw"
            if bw.exists():
                atac_by_arm.setdefault(s.arm or "unknown", []).append(str(bw))
        if len(atac_by_arm) == 2:
            arm_a, arm_b = sorted(atac_by_arm.keys())
            if not pre.bigwig_arm_a_paths:
                pre.bigwig_arm_a_paths = atac_by_arm[arm_a]
                pre.bigwig_arm_a_label = arm_a
            if not pre.bigwig_arm_b_paths:
                pre.bigwig_arm_b_paths = atac_by_arm[arm_b]
                pre.bigwig_arm_b_label = arm_b

    # ---- Matrix input: derive bigwig arm paths from a bigwigs/ dir next to
    # the ATAC count matrix.  TCGA-ATAC (Corces & Granja 2018) ships per-
    # technical-replicate normalized bigWigs via the GDC manifest, and their
    # filenames == the count-matrix sample columns (== atac_group_tsv rows).
    # Wiring these enables the Per-TF signal profile + Locus snapshots for
    # processed-matrix runs (which have no per-rep BAM/FASTQ).  FASTQ runs skip
    # this because their bigwig_arm paths are already set above.
    if ((not pre.bigwig_arm_a_paths or not pre.bigwig_arm_b_paths)
            and pre.atac_count_matrix and pre.atac_group_tsv
            and Path(pre.atac_group_tsv).exists()):
        _bwdir = Path(pre.atac_count_matrix).parent / "bigwigs"
        if _bwdir.is_dir():
            import csv as _csv
            _bw_by_arm: dict[str, list[str]] = {}
            with open(pre.atac_group_tsv) as _gf:
                for _row in _csv.DictReader(_gf, delimiter="\t"):
                    _s = (_row.get("sample") or "").strip()
                    _g = (_row.get("group") or "unknown").strip()
                    if not _s:
                        continue
                    for _cand in (f"{_s}.insertions.bw", f"{_s}.bw"):
                        _p = _bwdir / _cand
                        if _p.exists():
                            _bw_by_arm.setdefault(_g, []).append(str(_p))
                            break
            if len(_bw_by_arm) == 2:
                _aa, _bb = sorted(_bw_by_arm.keys())
                # Matrix runs can have dozens of replicate bigWigs per arm (52
                # for PRAD).  Pick 2 representative replicates per arm.  Prefer
                # a *condition-specificity screen* (from the DiffBind arm-
                # specific peak sets) so the chosen reps cleanly separate the
                # two conditions and are mutually consistent -> a real diagonal
                # in the peak heatmap + coherent locus replicate tracks.  A
                # size-only pick can select an anti-correlated tumour (e.g. an
                # AR_high sample more open at Basal-specific peaks).  Fall back
                # to largest-file when the screen is unavailable.
                def _pick2(_paths):
                    try:
                        return sorted(_paths, key=lambda p: Path(p).stat().st_size,
                                      reverse=True)[:2]
                    except Exception:
                        return list(_paths[:2])
                _screened = None
                if (pre.atac_count_matrix and pre.atac_group_tsv
                        and pre.diffbind_out_csv
                        and Path(pre.diffbind_out_csv).exists()):
                    try:
                        from .bigwig_screen import screen_top2_by_specificity
                        _screened = screen_top2_by_specificity(
                            _bw_by_arm, pre.atac_count_matrix,
                            pre.atac_group_tsv, pre.diffbind_out_csv)
                    except Exception:
                        _screened = None
                def _sel(_arm):
                    if _screened and _screened.get(_arm):
                        return _screened[_arm]
                    return _pick2(_bw_by_arm[_arm])
                if not pre.bigwig_arm_a_paths:
                    pre.bigwig_arm_a_paths = _sel(_aa)
                    pre.bigwig_arm_a_label = _aa
                if not pre.bigwig_arm_b_paths:
                    pre.bigwig_arm_b_paths = _sel(_bb)
                    pre.bigwig_arm_b_label = _bb

    # ---- Step 6e-2b: Peak heatmap (deeptools) over 3-way peak-set regions
    # (A-specific / Shared / B-specific, from within-arm rep narrowPeak
    # intersect). 2026-06-29 update: replaces the prior DiffBind
    # differential UP/DOWN region definition. Font-fixed (matplotlibrc)
    # helper; bamCoverage/bigwigAverage from deeptools 3.5.6 (has
    # bigwigAverage). ----
    peakheatmap_params: dict = {}
    _ph_npa: list[str] = []; _ph_bama: list[str] = []
    _ph_npb: list[str] = []; _ph_bamb: list[str] = []
    _ph_la = ""; _ph_lb = ""
    if cfg.samples:
        _by_arm: dict[str, list] = {}
        for s in cfg.samples:
            if s.modality != "ATAC":
                continue
            sdir = _sample_dir(s)
            npk = sdir / f"{s.name}_peaks.narrowPeak.filt"
            bam = sdir / "alignments" / f"{s.name}.dedup.bam"
            if npk.is_file() and bam.is_file():
                _by_arm.setdefault(s.arm or "unknown", []).append((str(npk), str(bam)))
        if len(_by_arm) == 2:
            # Column A = treatment, column B = control (cfg.conditions order),
            # NOT sorted() -- alphabetical order put the control on the left
            # whenever its label happened to sort first.
            _cond_lbls = [c.label for c in (cfg.conditions or [])]
            arm_a, arm_b = sorted(
                _by_arm.keys(),
                key=lambda a: _cond_lbls.index(a) if a in _cond_lbls else 99)
            if len(_by_arm[arm_a]) == 2 and len(_by_arm[arm_b]) == 2:
                _ph_npa = [p for p, _ in _by_arm[arm_a]]; _ph_bama = [b for _, b in _by_arm[arm_a]]
                _ph_npb = [p for p, _ in _by_arm[arm_b]]; _ph_bamb = [b for _, b in _by_arm[arm_b]]
                _ph_la, _ph_lb = arm_a, arm_b
    # Section 6 needs per-sample peaks + BAMs. When the run did not supply them,
    # a demo dataset can stand in, but only the original host has it and using
    # the wrong BAMs would render a heatmap of the wrong samples. So the paths
    # live in MRAG_DEMO_PEAKHEATMAP (a directory holding the four narrowPeak.filt
    # and alignments/*.dedup.bam) rather than being compiled in, every file is
    # checked before use, and with the variable unset Section 6 is simply omitted.
    _is_flex_input = bool(
        pre.atac_count_matrix or pre.atac_peaks_treatment or pre.atac_peaks_control
        or pre.rna_log2_matrix or pre.rna_counts_matrix)
    _demo_root = os.environ.get("MRAG_DEMO_PEAKHEATMAP", "")
    if not (_ph_npa and _ph_npb) and not _is_flex_input and _demo_root:
        _d = Path(_demo_root)
        _STD_NP = [str(_d / f) for f in (
            "1_pLenti_ONECUT2_OE_1a_merge_peaks.narrowPeak.filt",
            "2_pLenti_ONECUT2_OE_1b_merge_peaks.narrowPeak.filt",
            "3_pLenti_cMyc_DDK_2a_merge_peaks.narrowPeak.filt",
            "4_pLenti_cMyc_DDK_2b_merge_peaks.narrowPeak.filt")]
        _STD_BAM = [str(_d / "alignments" / f) for f in (
            "1_pLenti_ONECUT2_OE_1a_merge.dedup.bam",
            "2_pLenti_ONECUT2_OE_1b_merge.dedup.bam",
            "3_pLenti_cMyc_DDK_2a_merge.dedup.bam",
            "4_pLenti_cMyc_DDK_2b_merge.dedup.bam")]
        if all(Path(p).is_file() for p in _STD_NP) and all(Path(b).is_file() for b in _STD_BAM):
            _ph_npa, _ph_npb = _STD_NP[:2], _STD_NP[2:]
            _ph_bama, _ph_bamb = _STD_BAM[:2], _STD_BAM[2:]
            _ph_la, _ph_lb = "OC2_OE", "cMyc_DDK"
    if (_ph_npa and _ph_npb and _ph_bama and _ph_bamb):
        try:
            from .nodes.peakheatmap import PeakHeatmapNode
            _ph = PeakHeatmapNode(
                inputs={"narrowpeak_a": _ph_npa, "bam_a": _ph_bama, "label_a": _ph_la,
                        "narrowpeak_b": _ph_npb, "bam_b": _ph_bamb, "label_b": _ph_lb},
                params={"threads": min(48, _ncpus())},
                output_dir=str(Path(out_dir) / "peakheatmap"))
            _ph.execute()
            manifest.add_step(_ph.to_manifest()); manifest.write_atomic()
            if _ph.status == "ok":
                peakheatmap_params = {"peakheatmap_png": _ph.outputs["peakheatmap_png"],
                                      "label_a": _ph_la, "label_b": _ph_lb}
        except Exception as _ex:
            manifest.add_step({"name": "peakheatmap", "status": "fail",
                               "error": str(_ex)[:500]})
            manifest.write_atomic()
    elif (_is_flex_input and pre.atac_count_matrix and pre.atac_group_tsv
          and pre.diffbind_out_csv and Path(pre.diffbind_out_csv).exists()):
        # Processed-matrix ATAC (no per-rep BAM/narrowPeak).  Two Section-6 paths:
        # (1) REAL curated-style deeptools signal heatmap when per-arm bigWigs are
        # present.  TCGA-ATAC ships per-rep normalized bigWigs via the GDC
        # manifest; they are averaged per arm above (bigwig_arm_*_paths).
        # 3-way regions (treat-gained / Shared / ref-gained) come from
        # DiffBind differential stats over the shared consensus peak set,
        # then computeMatrix->plotHeatmap (identical Step3 to the FASTQ
        # run_peakheatmap.sh) -> matches the reference figure.
        # (2) Fallback log2-CPM peak x sample pheatmap (a proxy, NOT the +/-3kb
        # signal heatmap) when no bigWigs are available.
        _treat, _ref = (cfg.conditions[0].label, cfg.conditions[1].label) \
                        if cfg.conditions and len(cfg.conditions) >= 2 \
                        else ("treatment", "control")
        _ph_out = Path(out_dir) / "peakheatmap" / "peakheatmap.png"
        _ph_out.parent.mkdir(parents=True, exist_ok=True)
        try:
            # MCP step 2a (2026-10-12): this stage is the viz-report-mcp tool plot_peak_heatmap_matrix -- the code
            # formerly here (deepTools on the per-arm bigWigs when both arms have them, else the R log2-CPM
            # pheatmap proxy), unchanged; the step record below is the runner's, as before
            from types import SimpleNamespace as _NS
            from .nodes import _stage_tools as _st_ph
            _phm = _st_ph.tools("peak_heatmap")["plot_peak_heatmap_matrix"](
                output_png=str(_ph_out), diffbind_csv=pre.diffbind_out_csv, treat_label=_treat, ref_label=_ref,
                bigwigs_a=list(pre.bigwig_arm_a_paths or []), bigwigs_b=list(pre.bigwig_arm_b_paths or []),
                bigwig_a_label=pre.bigwig_arm_a_label, bigwig_b_label=pre.bigwig_arm_b_label,
                atac_count_matrix=pre.atac_count_matrix, atac_group_tsv=pre.atac_group_tsv,
                r_script=str(Path(__file__).parent / "adapters" / "atac_matrix_peakheatmap.R"))
            _step = _phm["step"]
            _r = _NS(returncode=_phm["returncode"], stdout=_phm["stdout"], stderr=_phm["stderr"])
            if _ph_out.exists() and _ph_out.stat().st_size > 10000:
                peakheatmap_params = {"peakheatmap_png": str(_ph_out),
                                      "label_a": _treat, "label_b": _ref}
                manifest.add_step({"name": _step, "status": "ok",
                                   "outputs": [{"key": "peakheatmap_png",
                                                "value": str(_ph_out)}]})
            else:
                manifest.add_step({"name": _step, "status": "fail",
                                   "error": (_r.stderr or _r.stdout)[:500]})
            manifest.write_atomic()
        except Exception as _ex:
            manifest.add_step({"name": "peakheatmap_matrix", "status": "fail",
                               "error": str(_ex)[:500]})
            manifest.write_atomic()

    # ---- Step 6e-4: Super-enhancer layer (2026-09-10) ----
    # ROSE on the treatment-specific peaks (peakheatmap/<treat>_specific.bed,
    # treatment-arm dedup BAMs) -> top-N SEs; then SE -> TF motif presence
    # (panel top-N PWMs + Section-8 enriched known motifs) and SE attributes.
    # Needs per-replicate ATAC BAMs: processed-matrix input is skipped cleanly.
    # Consumed by Step 6c-2e-3 (SE -> TF -> Target TRN, report Section 10b).
    se_layer: dict = {}
    _se_deseq = str(locals().get("deseq_csv") or getattr(pre, "deseq_csv", None) or "")
    # homer_motif / homer_known files are keyed by the condition label (the motif-scan step
    # arms = cfg.conditions[*].label); the peak-heatmap BED by its arm label
    # (_ph_la, which the demo fallback sets to a fixed name).
    _se_treat = cfg.conditions[0].label if cfg.conditions else (_ph_la or "")
    _se_bed = Path(out_dir) / "peakheatmap" / f"{_se_treat}_specific.bed"
    if not _se_bed.is_file() and _ph_la:
        _se_bed = Path(out_dir) / "peakheatmap" / f"{_ph_la}_specific.bed"
    # ROSE counts reads: ALL treatment-arm ATAC replicate BAMs of THIS run (not the
    # 2-per-arm peak-heatmap selection, and never the demo fallback BAMs).
    _se_bams = []
    for _s in (cfg.samples or []):
        if getattr(_s, "modality", "") == "ATAC" and (getattr(_s, "arm", "") or "") in {_se_treat, _ph_la or _se_treat}:
            _b = _sample_dir(_s) / "alignments" / f"{_s.name}.dedup.bam"
            if _b.is_file():
                _se_bams.append(str(_b))
    # No BAMs -> bigWig ROSE (2026-09-11): processed-matrix ATAC with per-sample
    # bigWigs next to the count matrix (TCGA: <matrix dir>/bigwigs/<sample>.insertions.bw,
    # all treatment-group samples), or any run whose treatment-arm bigWigs are set.
    # Constituent peaks: the matrix adapter's treatment-gained peaks, else the
    # treatment-specific BED.
    _se_bws: list = []
    _se_bw_bed = None
    if not _se_bams:
        if pre.atac_count_matrix and pre.atac_group_tsv and Path(pre.atac_group_tsv).exists():
            _bwd = Path(pre.atac_count_matrix).parent / "bigwigs"
            if _bwd.is_dir():
                import csv as _csv_bw
                with open(pre.atac_group_tsv) as _gf:
                    for _row in _csv_bw.DictReader(_gf, delimiter="\t"):
                        if (_row.get("group") or "").strip() != _se_treat:
                            continue
                        _sn = (_row.get("sample") or "").strip()
                        for _cand in (f"{_sn}.insertions.bw", f"{_sn}.bw", f"{_sn}.bigWig", f"{_sn}.bigwig"):
                            if _sn and (_bwd / _cand).is_file():
                                _se_bws.append(str(_bwd / _cand)); break
        if not _se_bws and pre.bigwig_arm_a_paths and (pre.bigwig_arm_a_label or "") == _se_treat:
            _se_bws = [p for p in pre.bigwig_arm_a_paths if Path(p).is_file()]
        _mab_se = locals().get("_matrix_arm_beds") or {}
        for _cb in (_mab_se.get(_se_treat), str(_se_bed)):
            if _cb and Path(_cb).is_file() and Path(_cb).stat().st_size > 0:
                _se_bw_bed = _cb
                break
    _se_known = homer_known_params.get("homer_known_a_dir", "")
    _se_lib = Path(out_dir) / "homer_motif" / f"{_se_treat}_peak_minmax.motif"

    def _se_record(node):
        """manifest record + rule-book diagnosis for directly executed SE nodes"""
        if node.status == "fail":
            try:
                from .debug import diagnose, DiagnosisContext
                _d = diagnose(DiagnosisContext(step_name=node.name, error_text=node.error or "",
                                               gate_results=node.gates, node_inputs=dict(node.inputs),
                                               node_outputs=dict(node.outputs)))
                node.diagnosis = {"rule": _d.rule, "likely_cause": _d.likely_cause,
                                  "suggested_actions": _d.suggested_actions, "severity": _d.severity}
                print(f"[se] ✗ {node.name} FAIL: {node.error} -> {_d.rule}", flush=True)
            except Exception:
                pass
        manifest.add_step(node.to_manifest()); manifest.write_atomic()
    _se_panel_json = (Path(topn_dir) / f"top{cfg.canonical_N}.json") if topn_dir else None
    if not getattr(cfg.pipeline, "se_analysis", True):
        manifest.add_step({"name": "se_rose", "status": "skipped", "note": "pipeline.se_analysis = false"})
        manifest.write_atomic()
    elif not _se_bams and not (_se_bws and _se_bw_bed):
        manifest.add_step({"name": "se_rose", "status": "skipped",
                           "note": ("no treatment-arm ATAC BAMs (<sample dir>/alignments/<sample>.dedup.bam) and no "
                                    "treatment-arm bigWigs + treatment peak BED for the bigWig ROSE path")})
        manifest.write_atomic()
    elif _se_bams and not (_se_bed.is_file() and _se_bed.stat().st_size > 0):
        manifest.add_step({"name": "se_rose", "status": "skipped",
                           "note": ("no treatment-specific peak BED (peakheatmap/<treat>_specific.bed needs >=2 "
                                    "replicate narrowPeaks per arm)")})
        manifest.write_atomic()
    else:
        try:
            _se_topn = int(getattr(cfg.pipeline, "se_top_n", 10))
            if _se_bams:
                from .nodes.se_rose import SERoseNode
                _sr = SERoseNode(
                    inputs={"bam_list": list(_se_bams), "peaks_bed": str(_se_bed), "label": _se_treat},
                    params={"genome": HOMER_GENOME, "threads": 16,
                            "top_n": _se_topn},
                    output_dir=str(Path(out_dir) / "se_rose"))
                _se_method = (f"ROSE on the {_se_treat}-specific peaks (read counts from {len(_se_bams)} treatment "
                              f"BAMs), 12.5 kb stitching, ranked by ROSE signal")
            else:
                from .nodes.se_rose_bigwig import SERoseBigwigNode
                _sr = SERoseBigwigNode(
                    inputs={"bigwig_list": list(_se_bws), "peaks_bed": str(_se_bw_bed), "label": _se_treat},
                    params={"genome": HOMER_GENOME, "top_n": _se_topn},
                    output_dir=str(Path(out_dir) / "se_rose"))
                _se_method = (f"ROSE stitching (12.5 kb) and Super Enhancer cutoff on the {_se_treat}-gained peaks, "
                              f"signal = mean ATAC bigWig area over {len(_se_bws)} treatment samples (no BAM available)")
            _sr.execute(); _se_record(_sr)
            if _sr.status == "ok":
                se_layer.update({"se_top_tsv": _sr.outputs["se_top_tsv"], "se_homerpeak": _sr.outputs["se_homerpeak"],
                                 "rose_plot_png": _sr.outputs.get("rose_plot_png", ""), "label": _se_treat,
                                 "method": _se_method})
                # motif library: the Step-2 min-max library (FASTQ) or the canonical
                # library of the processed-matrix HOMER scan (<motif_tsv minus .tsv>)
                if not _se_lib.is_file() and motif_tsv and str(motif_tsv).endswith(".motif.tsv") \
                        and Path(str(motif_tsv)[:-4]).is_file():
                    _se_lib = Path(str(motif_tsv)[:-4])
                _se_missing = [k for k, v in (("homer_known_a_dir", _se_known), ("panel_json", _se_panel_json and _se_panel_json.is_file()),
                                              ("motif_lib", _se_lib.is_file()), ("diffbind_csv", diffbind_provided),
                                              ("deseq_csv", _se_deseq)) if not v]
                if _se_missing:
                    manifest.add_step({"name": "se_motif", "status": "skipped",
                                       "note": "missing inputs: " + ", ".join(_se_missing)})
                    manifest.write_atomic()
                else:
                    from .nodes.se_motif import SEMotifNode
                    _sm = SEMotifNode(
                        inputs={"se_top_tsv": _sr.outputs["se_top_tsv"], "se_homerpeak": _sr.outputs["se_homerpeak"],
                                "panel_json": str(_se_panel_json), "motif_lib": str(_se_lib),
                                "known_dir": _se_known, "diffbind_csv": str(diffbind_provided),
                                "deseq_csv": _se_deseq, "label": _se_treat},
                        params={"cpu": 16, "genome": HOMER_GENOME, "n_known": 10},
                        output_dir=str(Path(out_dir) / "se_rose" / "motif"))
                    _sm.execute(); _se_record(_sm)
                    if _sm.status == "ok":
                        se_layer.update({
                            "se_tf_edges_tsv": _sm.outputs["se_tf_edges_tsv"], "se_genes_tsv": _sm.outputs["se_genes_tsv"],
                            "panel_tfs": [t for t in str(_sm.outputs.get("panel_tfs", "")).split(",") if t],
                            "enriched_tfs": [t for t in str(_sm.outputs.get("enriched_tfs", "")).split(",") if t]})
        except Exception as _ex:
            manifest.add_step({"name": "se_layer", "status": "fail", "error": str(_ex)[:500]})
            manifest.write_atomic()

    # ---- Step: viz builders (run only when precomputed not provided) ----
    from .nodes.viz_builders import (
        TFRankNode, TFTableNode, VolcanoATACNode, VolcanoRNANode,
        TRNRankedNode, TRNGraphNode, PerTFProfileNode,
    )
    viz_dir = out_dir / "viz_built"
    viz_dir.mkdir(parents=True, exist_ok=True)
    canon_N = cfg.canonical_N
    canon_consensus = variant_results.get(canon_N, {}).get("consensus_tsv", "")
    canon_top_json  = str(Path(topn_dir) / f"top{canon_N}.json")
    # TRN graph uses the LARGEST variant so the network shows more TFs/targets
    # (matches the reference analysis's published TRN density); per-TF: canonical.
    largest_N = max(variant_results.keys()) if variant_results else canon_N
    graph_consensus = variant_results.get(largest_N, {}).get("consensus_tsv", "") or canon_consensus
    graph_top_json  = str(Path(topn_dir) / f"top{largest_N}.json")
    import json as _json_local
    top_tfs_canon = []
    if Path(canon_top_json).exists():
        _d = _json_local.loads(Path(canon_top_json).read_text())
        top_tfs_canon = _d.get("tfs") if isinstance(_d, dict) else _d
    top_tfs_graph = []
    if Path(graph_top_json).exists():
        _d = _json_local.loads(Path(graph_top_json).read_text())
        top_tfs_graph = _d.get("tfs") if isinstance(_d, dict) else _d

    viz_outputs = {}

    if not pre.tf_rank_json:
        n = TFRankNode(inputs={"integrated_csv": integrated_csv},
                       params={}, output_dir=viz_dir)
        n.execute(); manifest.add_step(n.to_manifest())
        if n.status == "ok": viz_outputs["tf_rank_json"] = n.outputs["tf_rank_json"]
    if not pre.tf_table_csv:
        n = TFTableNode(inputs={"integrated_csv": integrated_csv},
                       params={"topn": 10}, output_dir=viz_dir)
        n.execute(); manifest.add_step(n.to_manifest())
        if n.status == "ok": viz_outputs["tf_table_csv"] = n.outputs["tf_table_csv"]
    if not pre.volcano_atac_json and pre.diffbind_out_csv:
        n = VolcanoATACNode(inputs={"diffbind_csv": pre.diffbind_out_csv},
                            params={"gtf_path": ORG.volcano_gtf}, output_dir=viz_dir)
        n.execute(); manifest.add_step(n.to_manifest())
        if n.status == "ok": viz_outputs["volcano_atac_json"] = n.outputs["volcano_atac_json"]
    if not pre.volcano_rna_json and pre.deseq_csv:
        n = VolcanoRNANode(inputs={"deseq_csv": pre.deseq_csv},
                           params={}, output_dir=viz_dir)
        n.execute(); manifest.add_step(n.to_manifest())
        if n.status == "ok": viz_outputs["volcano_rna_json"] = n.outputs["volcano_rna_json"]
    # Rank the LARGEST variant consensus for the TRN graph (dense network).
    # Rank the canonical variant consensus for per-TF profile (canonical 6 TFs).
    graph_ranked_tsv = ""
    if graph_consensus and not pre.trn_json:
        n = TRNRankedNode(inputs={"consensus_tsv": graph_consensus},
                          params={"top_k_per_tf": 12, "top_tfs": top_tfs_graph},
                          output_dir=viz_dir)
        n.execute(); manifest.add_step(n.to_manifest())
        if n.status == "ok":
            # rename output to disambiguate from per-TF ranked
            from shutil import move
            new_path = Path(n.outputs["consensus_ranked_tsv"]).with_name("consensus_ranked_graph.tsv")
            move(n.outputs["consensus_ranked_tsv"], new_path)
            graph_ranked_tsv = str(new_path)
    if not pre.trn_json and graph_ranked_tsv:
        n = TRNGraphNode(inputs={"ranked_tsv": graph_ranked_tsv},
                         params={}, output_dir=viz_dir)
        n.execute(); manifest.add_step(n.to_manifest())
        if n.status == "ok": viz_outputs["trn_json"] = n.outputs["trn_json"]

    # Per-TF profile uses canonical_N consensus
    # Ranked consensus TSV feeds TRN (S10) and Enrichment (S9) — neither of
    # which needs bigwig signal.  Produce it whenever the
    # canonical consensus exists.  Only the per-TF PROFILE plot below needs
    # bigwigs, so gate that separately (skipped for processed-matrix input).
    pertf_ranked_tsv = ""
    if not pre.per_tf_profile_json and canon_consensus:
        n = TRNRankedNode(inputs={"consensus_tsv": canon_consensus},
                          params={"top_k_per_tf": 12, "top_tfs": top_tfs_canon,
                                  "balance_direction": True},
                          output_dir=viz_dir)
        n.execute(); manifest.add_step(n.to_manifest())
        if n.status == "ok":
            pertf_ranked_tsv = n.outputs["consensus_ranked_tsv"]
        if pertf_ranked_tsv and pre.bigwig_arm_a_paths and pre.bigwig_arm_b_paths:
            bws = {pre.bigwig_arm_a_label: list(pre.bigwig_arm_a_paths),
                   pre.bigwig_arm_b_label: list(pre.bigwig_arm_b_paths)}
            n = PerTFProfileNode(inputs={"ranked_tsv": pertf_ranked_tsv,
                                          "bigwig_paths": bws,
                                          "top_tfs": top_tfs_canon,
                                          "treatment_label": (cfg.conditions[0].label
                                                              if cfg.conditions else "")},
                                  params={}, output_dir=viz_dir)
            n.execute(); manifest.add_step(n.to_manifest())
            if n.status == "ok": viz_outputs["per_tf_profile_json"] = n.outputs["per_tf_profile_json"]

    # ---- Step 6c-2e:  Cytoscape (Section 8, test.cys-style) ----
    # Produces trn_cyjs.json (Cytoscape.js elements + style) mirroring the
    # reference analysis's manual test.cys: continuous mappings on ATAC Fold
    # (fill) + RNA log2FC (border), blue=ATAC-absent, gray=RNA-absent defaults,
    # diamond for TF / ellipse for Target.
    trn_cyjs_json_path = ""

    # ---- Curated top-15 target pipeline (config-gated). When on, it produces
    # the Section-10 graph (trn_cytoscape/*), Section-11 enrichment
    # (enrichment/*) and the canonical branch targets_summary.json, and the
    # default Pval2-derived §10/§11 nodes below are skipped.
    _curated_on = bool(getattr(cfg, "curated_targets", False))
    _curated_top_genes = []
    _curated_fail_msg = ""        # 2026-09-28 pre-deploy review: a failed target step is loud
    if _curated_on:
        try:
            from .nodes.curated_targets import CuratedTargetsNode
            _treat_label = (cfg.conditions[0].label
                            if (cfg.conditions and len(cfg.conditions) >= 1) else "")
            _cur_motif = out_dir / "homer_motif" / f"{_treat_label}_peak_minmax.motif.tsv"
            _cur_pval2 = out_dir / "homer_motif" / f"{_treat_label}_peak_minmax.TFs.Pval2.txt"
            # processed-matrix mode: the fastq-run min-max Pval2/motif files do
            # not exist. Build a Pval2-equivalent from the canonical motif_tsv +
            # DiffBind + DESeq2 and point curated at the canonical motif_tsv.
            if not _cur_pval2.exists():
                # MCP step 2a (2026-10-12): the trn-mcp tool build_matrix_pval2
                from .nodes._stage_tools import matrix_stage as _matrix_stage
                build_pval2_from_matrix = _matrix_stage("build_pval2_from_matrix")
                _cm_bed = locals().get("atac_matrix_canon_bed")
                if motif_tsv and _cm_bed and diffbind_provided and deseq_csv:
                    _cur_cm = out_dir / "curated_targets"
                    _cur_cm.mkdir(parents=True, exist_ok=True)
                    _built_pval2 = _cur_cm / "pval2_from_matrix.txt"
                    _pv, _pvn = build_pval2_from_matrix(
                        str(motif_tsv), str(_cm_bed),
                        str(diffbind_provided), str(deseq_csv), str(_built_pval2))
                    _cur_motif = Path(motif_tsv)
                    _cur_pval2 = Path(_pv)
                    print(f"[curated] processed-mode pval2: {_pvn} promoter rows -> {_pv}",
                          flush=True)
            _cur_top6  = Path(topn_dir) / f"top{canon_N}.json"
            # 2026-09-11: extend the curated panel with the SE-layer enriched
            # motif TFs (gene symbols only) when the SE layer ran.
            if (getattr(cfg.pipeline, "se_extend_curated_panel", True)
                    and se_layer.get("enriched_tfs") and _cur_top6.is_file()):
                try:
                    _base_tfs = [str(t) for t in json.loads(_cur_top6.read_text()).get("tfs", [])]
                    _up = {t.upper() for t in _base_tfs}
                    # gene symbols only: family / composite HOMER motifs (AP-1, ETS,
                    # GATA, NF1-halfsite, Fox:Ebox ...) are not genes -> must be in DESeq2
                    _genes = set()
                    try:
                        import csv as _csv_se
                        with open(_se_deseq) as _fh:
                            for _r in _csv_se.DictReader(_fh):
                                _gn = _r.get("GENE") or _r.get("Gene") or _r.get("gene")
                                if _gn:
                                    _genes.add(_gn.upper())
                    except Exception:
                        _genes = set()
                    _add = [t for t in se_layer["enriched_tfs"]
                            if t and "_" not in t and ":" not in t and t.upper() not in _up
                            and (t.upper() in _genes if _genes else "-" not in t)]
                    if _add:
                        _ext = Path(topn_dir) / f"top{canon_N}_se.json"
                        _ext.write_text(json.dumps({"n": len(_base_tfs) + len(_add),
                                                    "tfs": _base_tfs + _add,
                                                    "panel_tfs": _base_tfs, "se_enriched_tfs": _add,
                                                    "source": "top%d.json + SE-layer Section-8 enriched motif TFs" % canon_N},
                                                   indent=2))
                        _cur_top6 = _ext
                        print(f"[curated] panel extended with SE enriched TFs: {_add}", flush=True)
                except Exception as _pex:
                    print(f"[curated] SE panel extension skipped: {_pex}", flush=True)
            _cur_enrich_tfs = []
            try:
                _cur_pj = json.loads(Path(_cur_top6).read_text())
                _cur_panel_n = len(_cur_pj.get("tfs", [])) or canon_N
                # Section 11 keeps the ranked top-N panel even when the SE layer
                # extended the panel for the Section-10 graph/table (2026-09-12).
                _cur_enrich_tfs = _cur_pj.get("panel_tfs") or _cur_pj.get("tfs") or []
            except Exception:
                _cur_panel_n = canon_N
            _cur_branch = out_dir / "branches" / f"top{canon_N}"
            _cond0 = cfg.conditions[0] if cfg.conditions else None
            _cur_comp = (f"{cfg.conditions[0].label} vs {cfg.conditions[1].label}"
                         if (cfg.conditions and len(cfg.conditions) >= 2) else "")
            _cur_ep = cfg.llm.endpoint.rstrip("/") + "/v1/chat/completions"
            n = CuratedTargetsNode(
                inputs={"motif_tsv": str(_cur_motif), "pval2_txt": str(_cur_pval2),
                        "top6_json": str(_cur_top6)},
                params={
                    "branch_dir":     str(_cur_branch),
                    "graph_dir":        str(out_dir / "trn_cytoscape"),
                    "enrich_dir":     str(out_dir / "enrichment"),
                    "integrated_csv": integrated_csv,
                    "cell_line":      _dom_cell_line,
                    "tissue":         _dom_field,
                    "cancer_type":    cfg.cancer_type,
                    "organism":       ORG.common_name,
                    "genome_fa":      ORG.genome_fa,
                    "hg38_fa":        ORG.genome_fa,   # legacy param name
                    # For a non-human run this is still the HUMAN gene_info,
                    # because the evidence gate asks for the official full name
                    # of the gene whose literature was retrieved -- the human
                    # ortholog (ORG.gene_info_is_ortholog says so).
                    "gene_info":      ORG.gene_info_gz,
                    "perturbation":   getattr(_cond0, "perturbation", None),
                    "concept":        getattr(_cond0, "concept", None),
                    "comparison":     _cur_comp,
                    "model":          cfg.llm.model,
                    "endpoint":       _cur_ep,
                    # Candidate pool: top-N target genes by log2FC that the
                    # panel TFs bind, before evidence scoring.
                    # Raised 200 -> 250 with the 6 -> 10 TF panel: a bigger
                    # panel means more genes competing for the same slots, and
                    # at 200 the cut had crept up to log2FC 6.044, dropping
                    # a well-evidenced target just below the cut.
                    "pool_n":         250,
                    # Evidence floor, not a quota: every gene scoring >= 2 is
                    # kept, capped at 30. Below 2 the scorer reports no
                    # gene-specific evidence, so those slots would be filled by
                    # log2FC alone.
                    "min_score":      2,
                    "top_final":      20,
                    "min_targets":    25,
                    # evidence search: union of the run's literature corpora
                    "vdb_cancer":     list(_lit_domains),
                    # 2026-09-26 (PI): the score prompt's Domain of interest is
                    # the user domain list Section 2 shows as "LLM domain"
                    "literature_domains_user": list(_lit_user),
                    # 2026-09-25: run.toml evidence_gate_scope; None keeps the
                    # node default (env MRAG_TARGET_EVIDENCE_SCOPE, else "all")
                    "evidence_gate_scope": getattr(cfg, "evidence_gate_scope", None),
                    # Section-11 spans the whole network, the same TFs the Section-10 graph
                    # draws: 0 means every TF. The ranked panel is still passed, but now to
                    # order the axis and name the two groups on the figure (2026-10-17).
                    "max_tfs":        0,
                    "enrich_tfs":     _cur_enrich_tfs,
                },
                output_dir=out_dir / "curated_targets",
            )
            n.execute(); manifest.add_step(n.to_manifest())
            if n.status == "ok":
                _curated_top_genes = n.outputs.get("curated_top_genes", [])
                trn_cyjs_json_path = n.outputs.get("trn_cyjs_json", "") or trn_cyjs_json_path
            else:
                _curated_fail_msg = _curated_failed_message(n.error)
                _print_loud(_curated_fail_msg + " The run goes on to render the other sections, "
                            "then exits 1.")
                manifest.write_atomic()
        except Exception as _e:
            _curated_on = False
            manifest.add_step({"name": "curated_targets", "status": "skipped",
                               "error": str(_e)[:300]})

    if (not _curated_on) and pertf_ranked_tsv and integrated_csv:
        try:
            from .nodes.trn_cytoscape import TrnCytoscapeBuilderNode
            # TRN target genes now come from the treatment-arm min-max
            # TFs.Pval2.txt (positive RNA log2FC, >=2 shared panel TFs,
            # log2FC-desc top-N); falls back to consensus if that file is absent.
            _treat_label = (cfg.conditions[0].label
                            if (cfg.conditions and len(cfg.conditions) >= 1) else "")
            _pval2_path = (out_dir / "homer_motif"
                           / f"{_treat_label}_peak_minmax.TFs.Pval2.txt")
            n = TrnCytoscapeBuilderNode(
                inputs={
                    "consensus_ranked_tsv":  pertf_ranked_tsv,
                    "integrated_ranked_csv": integrated_csv,
                    "pval2_txt": str(_pval2_path) if _pval2_path.exists() else "",
                },
                params={"top_targets": 15, "min_tfs_per_target": 2,
                        "max_tfs": canon_N},
                output_dir=out_dir / "trn_cytoscape",
            )
            n.execute(); manifest.add_step(n.to_manifest())
            if n.status == "ok":
                trn_cyjs_json_path = n.outputs["trn_cyjs_json"]
        except Exception as _ex:
            manifest.add_step({"name":"trn_cytoscape","status":"skipped",
                               "error":str(_ex)[:300]})

    # ---- Step 6c-2e-3: SE -> TF -> Target TRN (Section 10b, 2026-09-10) ----
    # Same Target nodes as the Section-10 graph (curated when on), the
    # panel + Section-8 enriched TFs, the ROSE top-N SEs and their nearest
    # genes. Skipped (never fatal) when the SE layer or the §10 graph is absent.
    se_trn_json_path = ""
    _s10_anno = Path(out_dir) / "trn_cytoscape" / "anno.txt"
    _s10_inter = Path(out_dir) / "trn_cytoscape" / "input_inter.txt"
    if se_layer.get("se_tf_edges_tsv") and _s10_anno.is_file() and _s10_inter.is_file():
        try:
            from .nodes.se_trn import SETrnBuilderNode
            _st = SETrnBuilderNode(
                inputs={"graph_anno": str(_s10_anno), "graph_inter": str(_s10_inter),
                        "se_top_tsv": se_layer["se_top_tsv"], "se_tf_edges_tsv": se_layer["se_tf_edges_tsv"],
                        "se_genes_tsv": se_layer["se_genes_tsv"],
                        "motif_tsv": (str(Path(out_dir) / "homer_motif" / f"{se_layer['label']}_peak_minmax.motif.tsv")
                                      if (Path(out_dir) / "homer_motif" / f"{se_layer['label']}_peak_minmax.motif.tsv").is_file()
                                      else str(motif_tsv or "")),
                        "diffbind_csv": str(diffbind_provided or ""), "deseq_csv": _se_deseq,
                        "canonical_peaks_bed": str(locals().get("atac_matrix_canon_bed") or ""),
                        "integrated_csv": str(integrated_csv or ""),
                        "panel_tfs": se_layer.get("panel_tfs", []), "enriched_tfs": se_layer.get("enriched_tfs", []),
                        "label": se_layer["label"]},
                # ATAC value for TFs absent from INTEGRATED.csv (e.g. the Fox:Ebox
                # composite): an explicit table, else this run's own DAStk output.
                params={"atac_fallback_tsv": getattr(cfg.pipeline, "se_atac_fallback_tsv", "") or "",
                        "dastk_md_csv": str(getattr(pre, "atac_md_csv", "") or ""),
                        "se_method": se_layer.get("method", ""),
                        # per-assembly; a missing dir disables the refill scan
                        "refill_dir": ORG.dastk_refill_dir,
                        "genome": HOMER_GENOME},
                output_dir=str(Path(out_dir) / "se_trn"))
            _st.execute(); _se_record(_st)
            if _st.status == "ok":
                se_trn_json_path = _st.outputs["se_cyjs_json"]
        except Exception as _ex:
            manifest.add_step({"name": "se_trn", "status": "fail", "error": str(_ex)[:500]})
            manifest.write_atomic()
    elif se_layer:
        manifest.add_step({"name": "se_trn", "status": "skipped",
                           "note": "SE motif layer or Section-10 graph files (trn_cytoscape/anno.txt) missing"})
        manifest.write_atomic()

    # ---- TRN target genes: the exact Section-10 network target set
    # (curated top-N when curated_targets is on, else the 'Target' nodes
    # of trn_cyjs). Drives the Section-12 locus panels so they mirror the
    # TRN graph rather than a re-derived differential pool. ----
    _trn_target_genes = list(_curated_top_genes) if _curated_top_genes else []
    if (not _trn_target_genes) and trn_cyjs_json_path and Path(trn_cyjs_json_path).exists():
        try:
            import json as _json_trn
            _cyj = _json_trn.loads(Path(trn_cyjs_json_path).read_text())
            _els = _cyj.get("elements", _cyj)
            _nodes = _els.get("nodes", []) if isinstance(_els, dict) else _els
            for _nd in (_nodes or []):
                _d = _nd.get("data", _nd) if isinstance(_nd, dict) else {}
                if str(_d.get("type", "")).lower() == "target":
                    _nm = _d.get("name") or _d.get("id")
                    if _nm:
                        _trn_target_genes.append(str(_nm))
        except Exception:
            pass

    # ---- Step 6c-2d: Locus snapshots (Section 12) — restricted to TRN targets ----
    locus_manifest_path = ""
    # A locus panel without this organism's gene models is worse than no
    # panel: pyGenomeTracks would draw whatever genes the fallback annotation
    # has at those coordinates, under a caption naming the right gene.
    _locus_gtf_ok = bool(ORG.volcano_gtf) and Path(ORG.volcano_gtf).is_file() and (
        not str(ORG.volcano_gtf).endswith(".gz")
        or Path(str(ORG.volcano_gtf) + ".tbi").is_file())
    if (pertf_ranked_tsv
            and pre.bigwig_arm_a_paths and pre.bigwig_arm_b_paths
            and not pre.locus_manifest_json and not _locus_gtf_ok):
        _skip_for_organism(
            "locus_snapshots",
            "the gene-model track needs a tabix-indexed %s annotation and %r "
            "is missing or unindexed. Drawing the panels without it would put "
            "another species' gene models under a %s caption, so Section 12 is "
            "skipped instead."
            % (ORG.scientific_name, ORG.volcano_gtf, ORG.common_name),
            section="Section 12 (locus snapshots)")
    elif (pertf_ranked_tsv
        and pre.bigwig_arm_a_paths and pre.bigwig_arm_b_paths
        and not pre.locus_manifest_json):
        from .nodes.locus_snapshots import LocusSnapshotsNode
        n = LocusSnapshotsNode(
            inputs={
                "ranked_consensus_tsv": pertf_ranked_tsv,
                "trn_target_genes": _trn_target_genes,
                # (2026-09-28 review: not after a FAILED target step -- the file may be an
                # earlier run's, and no section after Section 10 is based on a failed selection)
                "curated_pool_json": (str(out_dir / "curated_targets" / "pool.json")
                                      if (_curated_on and not _curated_fail_msg
                                          and (out_dir / "curated_targets" / "pool.json").exists())
                                      else ""),
                "bigwigs_arm_a": list(pre.bigwig_arm_a_paths),
                "bigwigs_arm_b": list(pre.bigwig_arm_b_paths),
                "arm_a_label":   pre.bigwig_arm_a_label,
                "arm_b_label":   pre.bigwig_arm_b_label,
                # treatment arm -> red tracks, control arm -> blue tracks
                "treatment_label": (cfg.conditions[0].label
                                    if cfg.conditions else ""),
            },
            # One panel per TRN target gene (no cap): Section 12 mirrors the
            # Section-10 network rather than a truncated preview of it.
            params={"max_tfs": 8, "flank_bp": 5000, "locus_pool_n": 60,
                    "max_panels": (len(_trn_target_genes)
                                   if _trn_target_genes else 8),
                    # THIS ORGANISM's gene models. The track helper's default is
                    # hg38, and most mm10 coordinates are inside hg38 bounds, so
                    # without this a mouse panel drew human genes under a mouse
                    # caption (2026-09-23).
                    "gene_gtf": ORG.volcano_gtf},
            output_dir=out_dir / "locus_snapshots",
        )
        n.execute(); manifest.add_step(n.to_manifest())
        if n.status == "ok":
            locus_manifest_path = n.outputs["locus_manifest_json"]

    # ---- Step 6c-2f: Functional Enrichment (Section 9 — between TRN and Locus
    # in output ordering; computed here as it consumes ranked consensus).
    shared_dotplot_json_path = ""
    per_tf_heatmap_json_path = ""
    if (not _curated_on) and pertf_ranked_tsv:
        try:
            from .nodes.enrichment import EnrichmentNode
            # Build dataset_context for LLM biological-relevance filtering
            ctx_conds = cfg.conditions or []
            _comparison = ""
            _perturbation = ""
            if len(ctx_conds) >= 2:
                _comparison = f"{ctx_conds[0].label} vs {ctx_conds[1].label}"
                pertubs = [c.perturbation for c in ctx_conds if c.perturbation]
                if pertubs: _perturbation = " / ".join(pertubs)
            _dataset_context = {
                "cell_line":    _dom_cell_line,
                "tissue":       _dom_field,
                "cancer_type":  cfg.cancer_type or "",
                # the domain the run was scored against: without it the term filter judges
                # relevance from a cell-line name alone
                "literature_domains": list(getattr(cfg, "literature_domains", []) or []),
                "organism":     ORG.common_name,
                "comparison":   _comparison,
                "perturbation": _perturbation,
                "study_label":  cfg.report_title or "",
            }
            n = EnrichmentNode(
                inputs={"ranked_consensus_tsv": pertf_ranked_tsv},
                params={"min_shared_tfs": 2, "max_tfs": canon_N,
                        "top_terms_per_lib": 10, "top_terms_total": 15,
                        "dataset_context": _dataset_context},
                output_dir=out_dir / "enrichment",
            )
            n.execute(); manifest.add_step(n.to_manifest())
            if n.status == "ok":
                shared_dotplot_json_path = n.outputs.get("shared_dotplot_json","")
                per_tf_heatmap_json_path = n.outputs.get("per_tf_heatmap_json","")
        except Exception as _e:
            manifest.add_step({"name":"enrichment","status":"skipped",
                               "error": str(_e)[:300]})

    # Curated runs build Section-11 enrichment inside CuratedTargetsNode (its
    # `enrich_dir` param), so the standalone EnrichmentNode above is skipped --
    # but those artifacts still have to reach render or Section 11 renders two
    # empty figures. Pick them up from the same directory.
    # (2026-09-28 review: not after a FAILED target step -- these files would be an earlier
    # run's; Section 11 then stays empty, as the report's notice says)
    if _curated_on and not _curated_fail_msg and not (shared_dotplot_json_path or per_tf_heatmap_json_path):
        _enr_dir = out_dir / "enrichment"
        _sd = _enr_dir / "shared_targets_dotplot.json"
        _ph = _enr_dir / "per_tf_heatmap.json"
        if _sd.exists():
            shared_dotplot_json_path = str(_sd)
        if _ph.exists():
            per_tf_heatmap_json_path = str(_ph)
        print("[enrichment] curated artifacts -> shared=%s per_tf=%s"
              % (bool(shared_dotplot_json_path), bool(per_tf_heatmap_json_path)),
              flush=True)

    # ---- Step 6c-2e: Section-13 axis validation (SE -> TF -> target, two models) ----
    # SECTION13_CONTRACT.md stage 2c-2: ONE step replaces the single-base AlphaGenome node
    # (old step 6c-2e) and its single-base scatter (old 6c-2e-2).  Default-on, no switch (PI
    # decision 5): AxisValidationNode derives its whole configuration from THIS run through
    # mrag_axis.stat.config -- the ROSE table and its signal column, the TRN factors and their
    # JASPAR filter, the lineage-foreign null, the AlphaGenome biosample group, the Corgi channel
    # map -- builds the edits with mrag_axis.prep, runs the AlphaGenome and Corgi arms
    # concurrently (~90 min together against ~2.5 h serially) and scores them with
    # mrag_axis.stat, definition="unified".  A failed arm is a manifest WARN gate with a stated
    # reason, never a failed step (PI decision 3, the model-arm
    # failure policy).
    axis_json_path = ""
    # The edit items themselves are cut out of hg38 sequence and hg38 refGene, and Corgi is
    # trained on hg38 only; the statistic's read-out track names are human biosamples too.  (NOT
    # "AlphaGenome's catalogue is human": organism.py carries 18 mouse AlphaGenome ATAC biosamples
    # and the deleted node did run AlphaGenome on a mouse run that had one.)  Decide BEFORE the
    # ~90 minutes, and leave a skip the report can state (the same gate the AlphaGenome node had).
    if ORG.genome != "hg38":
        _skip_for_organism(
            "axis_validation",
            "the axis test is hg38-only: the edits are cut from hg38 sequence and hg38 refGene, "
            "Corgi is trained on hg38, and the read-outs the statistic scores are named on human "
            "biosamples. This run is %s (%s), so the edits cannot be built for it and neither "
            "arm can exist, nor the two-model consensus."
            % (ORG.scientific_name, ORG.genome),
            section="Section 13 (SE-TF-target axis validation)")
    else:
        try:
            from .nodes.axis_validation import AxisValidationNode
            n = AxisValidationNode(
                # Everything else is derived from the run's own output; these two are named so
                # that a rebuilt TRN or SE layer makes this step dirty (Node.input_hash).
                inputs={"run_output": str(out_dir),
                        "se_cyjs_json": se_trn_json_path or "",
                        "se_top_tsv": (se_layer or {}).get("se_top_tsv", "") or ""},
                params={# the label axis_<domain>.json carries: this run's own literature domain
                        "domain": (_lit_user[0] if _lit_user else "") or _dom_field or "",
                        # NOT derivable from a run, and the published OC2 numbers and PRAD
                        # disagree: "sorted" is the OC2 research order (seko2.TRN_TFS) and
                        # reproduces the published OC2 edits, "network" is the run's own se_cyjs
                        # order and reproduces PRAD.  The derivable one is the default here
                        # (mrag_axis.prep.edits_enhancer.TRN_ORDER / edits_target.TARGET_ORDER);
                        # set them to "sorted" to reproduce the published OC2 measurement.
                        "trn_order": "network",
                        "target_order": "network",
                        # which model set leads Section 13: DECLARED (run.toml axis_headline),
                        # else the node's default rule -- never the smallest p (contract)
                        "headline": getattr(cfg, "axis_headline", None) or "",
                        # the AR cross-condition row: a DECLARED PRAD-only extra requirement
                        # (run.toml axis_ar_gate), never a default -- every prostate-lineage
                        # AlphaGenome request carries the 22Rv1 / VCaP tracks, so switching it on
                        # whenever they exist would make it the Holm headline of every prostate
                        # run, LNCaP/OC2 included
                        "ar_gate": bool(getattr(cfg, "axis_ar_gate", False)),
                        # this run's own literature domain list, which Section 13 prints; the
                        # node cannot resolve it (only the runner does) and it is NOT the
                        # run.toml cancer_type text
                        "literature_domain": ", ".join(_lit_user)},
                output_dir=out_dir / "axis_validation",
            )
            n.execute(); manifest.add_step(n.to_manifest())
            # Taken whatever the status: the node writes axis_<domain>.json even when it has to
            # withhold a number, because Section 13 states which arm is missing and why.
            _axj = (n.outputs or {}).get("axis_json") or ""
            if _axj and Path(_axj).exists():
                axis_json_path = _axj
        except Exception as _e:
            # Section 13 is a WARNING surface end to end: an unimportable node must not end a run
            # that has everything else (the model-arm failure policy).
            manifest.add_step({"name": "axis_validation", "status": "skipped",
                               "error": str(_e)[:300]})
    manifest.write_atomic()

    # ---- Step: cross-variant synthesis (only when >1 variants) ----
    cross_md_path = out_dir / "cross_variant_summary.md"
    if len(variant_results) > 1:
        variants_artifacts = {
            N: {"top_json":      str(Path(topn_dir)/f"top{N}.json"),
                "consensus_tsv": r.get("consensus_tsv","")}
            for N, r in variant_results.items()
        }
        summary_path = out_dir / "cross_variant_summary.json"
        summary = synthesize_cross_variant(variants_artifacts, summary_path)
        cross_md_path.write_text(render_cross_variant_markdown(summary))
        manifest.add_step({"name": "cross_variant_synthesis", "status":"ok",
                           "outputs":[{"key":"summary_json","path":str(summary_path)},
                                      {"key":"summary_md","path":str(cross_md_path)}]})
    else:
        # Single variant -> no cross-variant section in report
        cross_md_path = ""
        manifest.add_step({"name": "cross_variant_synthesis", "status":"skipped",
                           "outputs":[]})

    # ---- Step 6e-Plots:  R-script plots (rank / volcano / heatmap) ----
    # Always-on (reference default workflow). Each node depends only on
    # INTEGRATED.csv; heatmap also needs DESeq2.gene.csv + txi.rsem.counts.txt.
    plot_dir = out_dir / "integrated_plots"
    rank_png = ""
    volcano_png = ""
    heatmap_png = ""
    rank_json = ""
    volcano_json = ""
    heatmap_json = ""
    heatmap_median_json = ""
    if integrated_csv:
        from .nodes.integrated_plots import (
            RankPlotNode, VolcanoPlotNode, HeatmapPlotNode,
        )
        # Rank
        try:
            n = RankPlotNode(
                inputs={"integrated_csv": integrated_csv},
                params={"top_n": 10},
                output_dir=plot_dir,
            )
            n.execute(); manifest.add_step(n.to_manifest())
            if n.status == "ok":
                rank_png = n.outputs["rank_png"]
        except Exception as _ex:
            manifest.add_step({"name":"rank_plot","status":"skipped",
                               "error":str(_ex)[:300]})
        # Volcano
        try:
            n = VolcanoPlotNode(
                inputs={"integrated_csv": integrated_csv},
                params={"top_n": 10},
                output_dir=plot_dir,
            )
            n.execute(); manifest.add_step(n.to_manifest())
            if n.status == "ok":
                volcano_png = n.outputs["volcano_png"]
        except Exception as _ex:
            manifest.add_step({"name":"volcano_plot","status":"skipped",
                               "error":str(_ex)[:300]})
        # Interactive Plotly versions (preferred over PNG in Section 5)
        from .nodes.integrated_scatter import (
            InteractiveRankNode, InteractiveVolcanoNode,
        )
        try:
            n = InteractiveRankNode(
                inputs={"integrated_csv": integrated_csv},
                params={"top_n": 10},
                output_dir=plot_dir,
            )
            n.execute(); manifest.add_step(n.to_manifest())
            if n.status == "ok":
                rank_json = n.outputs["rank_json"]
        except Exception as _ex:
            manifest.add_step({"name":"interactive_rank_plot","status":"skipped",
                               "error":str(_ex)[:300]})
        try:
            n = InteractiveVolcanoNode(
                inputs={"integrated_csv": integrated_csv},
                params={"top_n": 10},
                output_dir=plot_dir,
            )
            n.execute(); manifest.add_step(n.to_manifest())
            if n.status == "ok":
                volcano_json = n.outputs["volcano_json"]
        except Exception as _ex:
            manifest.add_step({"name":"interactive_volcano_plot","status":"skipped",
                               "error":str(_ex)[:300]})
        # Heatmap (needs counts file)
        # Look for RSEM counts (txi.rsem.counts.txt) next to the DESeq2 csv.
        counts_txt = ""
        if pre.deseq_csv:
            d = Path(pre.deseq_csv).parent
            for cand in [d/"txi.rsem.counts.txt", d/"rsem.counts.txt",
                         d/"counts.tsv", d/"counts.txt"]:
                if cand.exists():
                    counts_txt = str(cand); break
        # Processed-matrix input: no RSEM counts file, but the RNA log2 matrix
        # (gene x sample) IS the expression source.  Its gene IDs match the
        # DESeq2 ID column, so it serves directly as counts_txt for S7.
        if not counts_txt and pre.rna_log2_matrix and Path(pre.rna_log2_matrix).exists():
            counts_txt = pre.rna_log2_matrix
        if pre.deseq_csv and counts_txt:
            # Derive arm sample lists from cfg.samples (RNA modality), matching
            # the counts.txt column names by sample.name.
            arms_rna: dict[str, list[str]] = {}
            for s in cfg.samples:
                if s.modality == "RNA":
                    arms_rna.setdefault(s.arm or "unknown", []).append(s.name)
            # Matrix input has no cfg.samples -> derive arm membership from the
            # RNA group TSV (sample<TAB>group) so heatmap columns are annotated.
            if not arms_rna and pre.rna_group_tsv and Path(pre.rna_group_tsv).exists():
                import csv as _csv
                with open(pre.rna_group_tsv) as _gf:
                    for _row in _csv.DictReader(_gf, delimiter="\t"):
                        _g = (_row.get("group") or "unknown").strip()
                        _s = (_row.get("sample") or "").strip()
                        if _s:
                            arms_rna.setdefault(_g, []).append(_s)
            arm_a, arm_b = (cfg.conditions[0].label, cfg.conditions[1].label)                             if cfg.conditions and len(cfg.conditions) >= 2                             else ("OC2","Cyc")
            try:
                n = HeatmapPlotNode(
                    inputs={
                        "integrated_csv": integrated_csv,
                        "deseq_csv":      pre.deseq_csv,
                        "counts_txt":     counts_txt,
                        "samples_a":      arms_rna.get(arm_a),
                        "samples_b":      arms_rna.get(arm_b),
                    },
                    params={
                        "top_n":         10,
                        "group_a_label": arm_a,
                        "group_b_label": arm_b,
                    },
                    output_dir=plot_dir,
                )
                n.execute(); manifest.add_step(n.to_manifest())
                if n.status == "ok":
                    heatmap_png = n.outputs["heatmap_png"]
            except Exception as _ex:
                manifest.add_step({"name":"heatmap_plot","status":"skipped",
                                   "error":str(_ex)[:300]})
            # Interactive Plotly heatmap (Section 7)
            try:
                from .nodes.integrated_scatter import InteractiveHeatmapNode
                n = InteractiveHeatmapNode(
                    inputs={
                        "integrated_csv": integrated_csv,
                        "deseq_csv":      pre.deseq_csv,
                        "counts_txt":     counts_txt,
                        "samples_a":      arms_rna.get(arm_a),
                        "samples_b":      arms_rna.get(arm_b),
                    },
                    params={
                        "top_n":         10,
                        "group_a_label": arm_a,
                        "group_b_label": arm_b,
                    },
                    output_dir=plot_dir,
                )
                n.execute(); manifest.add_step(n.to_manifest())
                if n.status == "ok":
                    heatmap_json = n.outputs["heatmap_json"]
                    heatmap_median_json = n.outputs.get("heatmap_median_json","")
            except Exception as _ex:
                manifest.add_step({"name":"interactive_heatmap_plot","status":"skipped",
                                   "error":str(_ex)[:300]})
        manifest.write_atomic()

    # ---- Step: render ----
    canonical = cfg.canonical_N
    canon_art = variant_results.get(canonical, {})

    # ---- Rebuild Notable Findings (Section 14) for the curated targets ----
    # The NarrativeNode builds notable_cards.json during the (early) narrative
    # step, from the pre-curated targets; a curated run's Section 10 ranks the
    # CURATED ones, so the cards are rebuilt here, after them, and overwrite the
    # branch notable_cards.json that RenderNode loads.
    # 2026-09-17 (llmdomain): factored into nodes.viz_report.rebuild_notable_cards.
    # Rebuilt in curated
    # mode, when the cards are empty, or when they are stale for the per-domain
    # Novelty (no novelty_by_domain / other novelty_domains_version / other
    # domain list or order than this run's literature_domains_user).
    # followup r3 (post-deploy finding 2): in a curated run with the evidence layers on and literature domains
    # the cards must be rebuilt WITH the evidence rules (nodes.viz_report.card_evidence, strict).  If that did
    # not happen -- the rebuild raised (e.g. CardEvidenceUnavailable) or did not run -- the narrative-time cards
    # (built before the curated targets, never with the rules) are WITHHELD instead of shown: moved aside, an empty
    # notable_cards.json, a status record Section 14 shows, a FAILED manifest step with a gate, a loud log block.
    # Everything else (evidence layers off, no curated targets, no literature domains): unchanged.
    # In curated mode the cards are ordered by the curated rerank.
    _cards_strict, _cards_ok, _cards_err = False, False, ""
    _narr_md = canon_art.get("narrative_md", "")
    _branch_dir = Path(_narr_md).parent if _narr_md else None
    try:
        from .nodes.viz_report import cards_strict as _cards_strict_fn, clear_cards_status as _clear_cs
        _cards_strict = _cards_strict_fn(bool(_curated_on and not _curated_fail_msg), list(_lit_user))
        if _branch_dir is not None:
            _clear_cs(_branch_dir)          # never an earlier run's status over this run's cards
    except Exception as _ex:
        print("[notable] card evidence mode not determined (%s: %s)" % (type(_ex).__name__, _ex), flush=True)
        _cards_strict = bool(_curated_on and not _curated_fail_msg and _lit_user)
    try:
        # a curated run is what rebuilds the cards
        if _branch_dir is not None and bool(_curated_on and not _curated_fail_msg):
            from .nodes.viz_report import rebuild_notable_cards as _rebuild_nf
            _nf = _rebuild_nf(
                # (2026-09-28 review: a FAILED target step is not a curated run for
                # Section 14 -- its branch files would be an earlier run's)
                _branch_dir, curated_on=bool(_curated_on and not _curated_fail_msg),
                # Curated runs order Section 14 by the curated rerank so the
                # cards match the Section-10 target ranking.
                curated_top_genes=(list(_curated_top_genes) if _curated_top_genes else None),
                field=_dom_field, novelty_domains=list(_lit_user),
                retriever_script=getattr(cfg.paperqa, "retriever_script", "") or "",
                endpoint=cfg.llm.endpoint, model=cfg.llm.model,
                organism=ORG.key,
                ortholog_json=str(out_dir / "branches" / f"top{canonical}"
                                  / "ortholog.json"))
            _cards_ok = bool(_nf.get("rebuilt") and _nf.get("evidence") == "applied")
            if _nf.get("rebuilt"):
                manifest.add_step({
                    "name": "notable_findings_rebuild", "status": "ok",
                    "outputs": [{"key": "n_cards", "value": _nf["n_cards"]},
                                {"key": "reason", "value": _nf["reason"]},
                                {"key": "novelty_domains", "value": ",".join(_lit_user)},
                                {"key": "elapsed_s", "value": _nf["elapsed_s"]}]
                               # followup r3: an unreadable scored.json / sidecar the context was built without
                               + ([{"key": "evidence_notes", "value": _nf["evidence_notes"]}]
                                  if _nf.get("evidence_notes") else [])})
                manifest.write_atomic()
    except Exception as _ex:
        _cards_err = "%s: %s" % (type(_ex).__name__, str(_ex)[:400])
        if not _cards_strict:
            manifest.add_step({"name": "notable_findings_rebuild", "status": "skipped",
                               "error": str(_ex)[:300]})
            manifest.write_atomic()
    if _cards_strict and not _cards_ok and _branch_dir is not None:
        try:
            from .nodes.viz_report import withhold_unfiltered_cards as _withhold
            _wh = _withhold(_branch_dir, _cards_err or (
                "the Section-14 cards were not rebuilt for the curated targets (%s)"
                % "the rebuild did not run or ran without the evidence context"))
        except Exception as _ex2:
            _wh = {"withheld": False, "reason": _cards_err, "error": "%s: %s" % (type(_ex2).__name__, _ex2)}
        _cards_msg = (
            "Section-14 target-gene cards WITHHELD: they could not be rebuilt with the evidence rules of this "
            "run's curated targets (%s). The cards on disk were the narrative-time ones, built before the curated "
            "targets and without the evidence filters, so they could cite papers the evidence gate excluded%s. "
            "Rebuild them when the cause is fixed: re-run with --resume-manifest <run_dir>/output/manifest.json."
            % (_wh.get("reason") or "unknown",
               ("; %s card(s) moved to %s" % (_wh.get("n_withheld"), _wh.get("moved_to")))
               if _wh.get("moved_to") else ""))
        if not _wh.get("withheld"):
            _cards_msg += " COULD NOT EMPTY notable_cards.json (%s) -- do not use Section 14 of this report." % (
                _wh.get("error") or "?")
        _print_loud(_cards_msg)
        manifest.add_step({"name": "notable_findings_rebuild", "status": "fail", "error": _cards_msg[:2000],
                           "gates": [{"name": "notable_cards_evidence", "result": "fail",
                                      "value": _wh.get("n_withheld"), "msg": _cards_msg[:600]}],
                           "outputs": [{"key": "withheld", "value": bool(_wh.get("withheld"))},
                                       {"key": "n_withheld", "value": _wh.get("n_withheld")},
                                       {"key": "moved_to", "value": _wh.get("moved_to") or ""}]})
        manifest.write_atomic()

    try:
        from viz_report_mcp.tools.narrative import build_samples_description as _bsd
        _samples_desc = _bsd(
            [c.label for c in (cfg.conditions or [])],
            cancer_type=getattr(cfg, 'cancer_type', None),
            tissue=getattr(cfg, 'tissue', None),
            endpoint=cfg.llm.endpoint, model=cfg.llm.model)
    except Exception:
        _samples_desc = " vs ".join(c.label for c in (cfg.conditions or []))

    def _ortholog_overview_line() -> str:
        """The one-line counts the ortholog route already computes.

        Reads the ortholog.json the curated-targets node writes beside the
        genes it describes, so the line counts the genes that ACTUALLY reached
        the literature layer rather than a different set. "" on a human run --
        its report text is unchanged.
        """
        if ORTHO.identity:
            return ""
        try:
            _p = (out_dir / "branches" / f"top{canonical}" / "ortholog.json")
            _d = json.loads(_p.read_text())
            _tf = _d.get("top_final") or {}
            if not _tf:
                raise ValueError("no genes recorded")
            _n_total = len(_tf)
            _n_unmapped = sum(1 for v in _tf.values()
                              if not v.get("lit_searched", True))
            from .orthology import OVERVIEW_LINE as _OL, _fmt as _ofmt
            return _ofmt(_OL, source=ORTHO.source, n_total=_n_total,
                         n_mapped=_n_total - _n_unmapped,
                         n_unmapped=_n_unmapped)
        except Exception as _e:
            print("[organism] ortholog overview line unavailable (%s: %s)"
                  % (type(_e).__name__, str(_e)[:100]), flush=True)
            return ""

    render = RenderNode(
        inputs={
            "narrative_md":     canon_art.get("narrative_md",""),
            "abstract_md":      canon_art.get("abstract_md",""),
            "narrative_audit":  canon_art.get("narrative_audit",""),
            "cross_variant_md": str(cross_md_path) if cross_md_path else "",
            "tf_table_csv":     viz_outputs.get("tf_table_csv") or pre.tf_table_csv or "",
            "shared_dotplot_json": shared_dotplot_json_path or "",
            "per_tf_heatmap_json": per_tf_heatmap_json_path or "",
            "tf_rank_json":      viz_outputs.get("tf_rank_json")      or pre.tf_rank_json      or "",
            "trn_json":          viz_outputs.get("trn_json")          or pre.trn_json          or "",
            "locus_manifest_json": locus_manifest_path or pre.locus_manifest_json or "",
            "per_tf_profile_json": viz_outputs.get("per_tf_profile_json") or pre.per_tf_profile_json or "",
            "volcano_atac_json": viz_outputs.get("volcano_atac_json") or pre.volcano_atac_json or "",
            "volcano_rna_json":  viz_outputs.get("volcano_rna_json")  or pre.volcano_rna_json  or "",
            # 2026-06-04 R-script PNGs (Section 7 heatmap; rank/volcano fallback)
            "rank_png":      rank_png,
            "volcano_png":   volcano_png,
            "heatmap_png":   heatmap_png,
            # 2026-06-08 interactive Plotly Sections 5 + 7 (preferred over PNGs)
            "rank_json":     rank_json,
            "volcano_json":  volcano_json,
            "heatmap_json":  heatmap_json,
            "heatmap_median_json": heatmap_median_json,
            # 2026-06-07 Cytoscape (Section 8, test.cys-style)
            "trn_cyjs_json":     trn_cyjs_json_path,
            # Section-10 target ranking table (curated targets, under the TRN)
            # (not when the step FAILED: a curated_targets.json left by an earlier run in this
            # output dir must never be shown as this run's targets)
            "curated_targets_json": (str(out_dir / "branches" / f"top{canonical}"
                                         / "curated_targets.json")
                                     if (_curated_on and not _curated_fail_msg) else ""),
            "ortholog_json":     (str(out_dir / "branches" / f"top{canonical}"
                                      / "ortholog.json") if _curated_on else ""),
            # Section 10b: SE -> TF -> Target TRN (2026-09-10)
            "se_trn_json":       se_trn_json_path,
            # Section 13: the SE-TF-target axis validation (stage 2c-2, SECTION13_CONTRACT.md).
            # Replaces the removed ag_table_csv and ag_scatter_json.
            "axis_json":         axis_json_path,
            # Section 9: GIGGLE Cistrome TF-overlap plots
            "giggle_up_png":     giggle_params.get("giggle_up_png", ""),
            "giggle_down_png":   giggle_params.get("giggle_down_png", ""),
            # Section 6: peak heatmap (replaces per-TF ATAC profile)
            "peakheatmap_png":   peakheatmap_params.get("peakheatmap_png", ""),
        },
        params={
            "report_title": cfg.report_title or
                f"MRAG ({'/'.join(str(n) for n in cfg.pipeline.topn_variants)})",
            "conditions_str": cfg.conditions_str or (
                " × ".join(c.label for c in cfg.conditions) if cfg.conditions else ""
            ),
            "samples_str": _samples_desc,
            "tf_selection_label": f"Top {'/'.join(str(n) for n in cfg.pipeline.topn_variants)} (canonical={canonical})",
            "cancer_type": cfg.cancer_type or "",
            # Section 2 "LLM domain" row (2026-09-17): full search list + user list
            "literature_domains": list(_lit_domains),
            "literature_domains_user": list(_lit_user),
            "giggle_up_label":   giggle_params.get("giggle_up_label", "Up in treatment"),
            "giggle_down_label": giggle_params.get("giggle_down_label", "Down in treatment"),
            # Section 2 organism rows + the explicit skip list (2026-09-23).
            # Every figure in the report is conditional on this genome, so the
            # report states it and says what it could not do because of it.
            "organism":          ORG.scientific_name,
            "organism_common":   ORG.common_name,
            "genome_build":      ORG.genome,
            "organism_detection": ("%s (%s)" % (org_decision.source,
                                                org_decision.detail)
                                   if org_decision.detail
                                   else org_decision.source),
            "organism_skipped":  [{"name": x["name"], "reason": x["reason"]}
                                  for x in ORGANISM_SKIPS],
            "ortholog_banner":   ORTHO.banner(ORG.scientific_name, ORG.genome),
            # The counts line the ortholog route computes was never rendered
            # anywhere (2026-09-23). It belongs in Section 2, beside the genome
            # the report states, because Section 10's target set is SELECTED by
            # the ortholog-routed literature score.
            "ortholog_overview": _ortholog_overview_line(),
            # 2026-09-28 pre-deploy review: the red notice of a failed target step
            **({"curated_error": _curated_fail_msg} if _curated_fail_msg else {}),
            **homer_known_params,
        },
        output_dir=out_dir,
    )
    render.execute()
    manifest.add_step(render.to_manifest())
    # Single-variant runs have no cross-variant `summary` object — fall back
    # to the variant's own top TF list.
    if len(variant_results) > 1 and 'summary' in dir():
        finalize_kwargs = {
            "robust_core_tfs":      summary["robust_core_tfs"],
            "stability_score":      summary["stability_score_jaccard_lo_hi"],
            "final_html":           render.outputs.get("mrag_report_html",""),
            "cross_variant_summary": str(summary_path),
            "per_N_branches": {str(N): r.get("branch_dir","") for N, r in variant_results.items()},
        }
        ret_extra = {
            "robust_core_tfs":  summary["robust_core_tfs"],
            "fringe_tfs":       summary["fringe_tfs"],
            "stability":        summary["stability_score_jaccard_lo_hi"],
        }
    else:
        # Single-variant: derive top TFs from canonical variant's consensus
        canonical = cfg.canonical_N
        canon_top = []
        try:
            import json as _json
            canon_top = _json.load(open(Path(topn_dir)/f"top{canonical}.json")).get("tfs", [])
        except Exception:
            pass
        finalize_kwargs = {
            "robust_core_tfs":      canon_top,
            "stability_score":      [1.0, 1.0],   # trivial when 1 variant
            "final_html":           render.outputs.get("mrag_report_html",""),
            "cross_variant_summary": "",
            "per_N_branches": {str(N): r.get("branch_dir","") for N, r in variant_results.items()},
        }
        ret_extra = {
            "robust_core_tfs":  canon_top,
            "fringe_tfs":       [],
            "stability":        [1.0, 1.0],
        }
    if _curated_fail_msg:
        finalize_kwargs["curated_targets_error"] = _curated_fail_msg
    manifest.finalize(summary=finalize_kwargs)
    manifest.write_atomic()

    return {
        "ok": render.status != "fail" and not _curated_fail_msg,
        **ret_extra,
        **({"failed_at": "curated_targets", "error": _curated_fail_msg} if _curated_fail_msg else {}),
        "manifest":         str(manifest.path),
        "final_html":       render.outputs.get("mrag_report_html",""),
    }


def sample_sheet_to_run_config(sheet_path: str | Path,
                               output_dir: str,
                               run_label: Optional[str] = None,
                               precomputed: Optional[dict] = None,
                               canonical_N: int = 6,
                               cancer_type: Optional[str] = None,
                               cell_line: Optional[str] = None,
                               tissue: Optional[str] = None,
                               literature_domains: Optional[list[str] | str] = None,
                               organism: Optional[str] = None,
                               treatment_arm: Optional[str] = None) -> RunConfig:
    """Convert a detect-stage sample_sheet.json into a RunConfig.

    Caller can pass `precomputed={'atac_md_csv': '...', ...}` to skip
    upstream stages.

    `treatment_arm` names which condition is the TREATMENT arm. conditions[0]
    is the treatment arm everywhere downstream -- it decides every fold sign,
    the red/blue track colours, the GIGGLE labels and which arm's peaks ROSE
    calls super-enhancers on -- but the detector orders conditions by the arm
    it happened to see first, which is filename order. Until this existed
    there was no supported way to change it (2026-09-23).
    """
    from .config import SampleSheet, RunConfig, PrecomputedArtifacts
    sheet = SampleSheet.model_validate_json(Path(sheet_path).read_text())

    # ---- arm order -------------------------------------------------------
    _conds = list(sheet.conditions or [])

    def _armnorm(s) -> str:
        return str(s or "").strip().lower().replace("-", "_").replace(" ", "_")

    if treatment_arm:
        _want = _armnorm(treatment_arm)
        _i = next((i for i, c in enumerate(_conds)
                   if _armnorm(c.label) == _want), None)
        if _i is None:
            _i = next((i for i, c in enumerate(_conds)
                       if _want and (_want in _armnorm(c.label)
                                     or _armnorm(c.label) in _want)), None)
        if _i is None:
            raise ValueError(
                "--treatment-arm %r does not name any condition in this sample "
                "sheet. Its conditions are: %s"
                % (treatment_arm, [c.label for c in _conds]))
        if _i:
            _conds = [_conds[_i]] + [c for k, c in enumerate(_conds) if k != _i]

    # Merge detector's processed-input suggestion; explicit --precomputed wins.
    merged: dict = dict(sheet.precomputed_suggestion or {})
    merged.update(precomputed or {})

    # Resolve ATAC peaks grouped by arm -> treatment/control ordering. Prefer
    # conditions order (conditions[0] = treatment); fall back to sorted arms.
    by_arm = merged.pop("atac_peaks_by_arm", None)
    if by_arm and not (merged.get("atac_peaks_treatment") or merged.get("atac_peaks_control")):
        arm_keys = list(by_arm.keys())
        t_arm = c_arm = None
        if _conds and len(_conds) >= 2:
            def _pick(label):
                lab = label.lower().replace("-", "_")
                for a in arm_keys:
                    if a.lower().replace("-", "_") == lab or lab in a.lower().replace("-", "_") \
                            or a.lower().replace("-", "_") in lab:
                        return a
                return None
            t_arm = _pick(_conds[0].label)
            c_arm = _pick(_conds[1].label)
        if not (t_arm and c_arm):
            s = sorted(arm_keys)
            t_arm, c_arm = s[0], s[1]
        merged["atac_peaks_treatment"] = by_arm[t_arm]
        merged["atac_peaks_control"]   = by_arm[c_arm]

    # Keep only valid PrecomputedArtifacts fields.
    valid = set(PrecomputedArtifacts.model_fields.keys())
    merged = {k: v for k, v in merged.items() if k in valid}

    # Organism: an explicit --organism wins; otherwise inherit whatever the
    # detector recorded from the study metadata. Left None when neither knows,
    # so run_from_config still gets to infer and to say it defaulted.
    _organism = organism or sheet.organism or None
    if _organism:
        from .organism import resolve as _resolve_org
        _organism = _resolve_org(_organism).key      # normalise + validate now

    return RunConfig(
        run_label=run_label or build_run_id(),
        output_dir=output_dir,
        organism=_organism,
        samples=sheet.samples,
        conditions=_conds,
        canonical_N=canonical_N,
        precomputed=PrecomputedArtifacts(**merged),
        cancer_type=cancer_type,
        literature_domains=literature_domains or [],
        cell_line=cell_line,
        tissue=tissue,
    )
