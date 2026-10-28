"""the motif-scan step chain — the canonical default downstream of DiffBind.

Wires (one execution per run):

   ATAC sample BEDs --(HomerNode × 4)--> motif.tsv × 4
                                                |
                                                v
              MinmaxBedNode × 2 (one per condition) --> *_peak_minmax.bed
                                                |
                                                v
            HomerNode × 2 (on minmax BEDs) --> *_peak_minmax.motif.tsv
                                                |
                                                v
          FilterTargetTfsNode × 2 (10 default TFs) --> *.TFs.txt
                                                |
                                                v
   JoinDiffbindRnaNode × 2 (DiffBind + RNA join) --> *.TFs.Pval2.txt
                                                |
                                                v
                                  (consumed by Phase C R plots)

 convention: condition_A = cfg.conditions[0] (e.g. OC2/ONECUT2_OE),
                condition_B = cfg.conditions[1] (e.g. CyC/cMyc_DDK).
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import json

from .nodes.homer import (
    HomerNode, MinmaxBedNode, FilterTargetTfsNode, JoinDiffbindRnaNode,
)
from .manifest import Manifest


DEFAULT_TARGET_TFS = [
    "ELF5", "EHF", "E2F2", "EGR1", "ETV5",
    "KLF15", "MAFB", "ETV1", "GATA3", "MEF2C",
]


def _arm_samples(cfg, arm_label: str):
    """Return ATAC samples for a given arm label, ordered by replicate."""
    samples = [s for s in cfg.samples
               if s.modality == "ATAC" and (s.arm or "") == arm_label]
    samples.sort(key=lambda s: int(getattr(s, "replicate", 1)))
    return samples






def _ncpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _safe_max_workers(n_tasks: int, ram_per_worker_gb: Optional[float] = None) -> int:
    """Cap parallel HOMER workers by a RAM budget.

    2026-09-16: annotatePeaks is memory-bounded inside homer_mcp
    (tools/annotate_bounded.py). Each annotatePeaks process tree is kept under
    MRAG_HOMER_ANNOT_MAX_GB (default 40 GiB) and all of them together under
    MRAG_HOMER_TOTAL_GB (default 160 GiB; shared slot files, so across runs
    too). A worker here therefore needs about one chunk budget and the pool is
    sized against MRAG_HOMER_TOTAL_GB; extra workers only wait for slots.

    With MRAG_HOMER_CHUNKING=0 one unchunked scan needs 190-230 GiB (per-rep
    narrowPeak) to ~350-400 GiB (min-max BED), so the default is 400 GB per
    worker against 30% of MemTotal: serial.

    budget = min(psutil available RAM, the policy budget above).
    MRAG_HOMER_RAM_PER_WORKER_GB still overrides the per-worker figure.
    """
    import os
    chunking = os.environ.get("MRAG_HOMER_CHUNKING", "1").strip().lower() not in ("0", "false", "no", "off")

    def _envf(name, default):
        v = os.environ.get(name, "").strip()
        try:
            return float(v) if v else default
        except ValueError:
            return default
    if ram_per_worker_gb is None:
        ram_per_worker_gb = _envf("MRAG_HOMER_ANNOT_MAX_GB", 40.0) if chunking else 400.0
    override = os.environ.get("MRAG_HOMER_RAM_PER_WORKER_GB")
    if override:
        try: ram_per_worker_gb = float(override)
        except ValueError: pass
    mem_total_gb = 256.0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total_gb = int(line.split()[1]) / (1024**2)
                    break
    except Exception:
        pass
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / (1024**3)
    except Exception:
        avail_gb = mem_total_gb / 2
    if chunking:
        policy_gb = _envf("MRAG_HOMER_TOTAL_GB", 160.0)
    else:
        hard = _envf("MRAG_MEM_HARD", 0.30)
        policy_gb = hard * mem_total_gb if hard <= 1.0 else hard
    budget_gb = min(avail_gb, policy_gb)
    safe = max(1, min(n_tasks, int(budget_gb / max(ram_per_worker_gb, 1.0))))
    print(f"[HOMER pool] avail={avail_gb:.0f} GB, budget={budget_gb:.0f} GB "
          f"(chunking={'on' if chunking else 'off'}), per_worker={ram_per_worker_gb:.0f} GB "
          f"-> max_workers={safe} (of {n_tasks} tasks)", flush=True)
    return safe


def _validate_motif_tsv(motif_tsv: str,
                        min_size_bytes: int = 100_000,
                        min_rows: int = 1000,
                        min_motif_cols: int = 50,
                        required_tfs: Optional[list] = None) -> dict:
    """Integrity gate for a single HOMER motif.tsv output.

    Returns dict with status='ok' OR 'fail' + reasons. Used as a barrier
    after parallel scans complete: every file must pass before downstream
    steps (MinmaxBedNode, filter, join) proceed.
    """
    import csv, os
    p = Path(motif_tsv)
    reasons: list[str] = []
    if not p.exists():
        return {"status":"fail","reason":["file does not exist"],
                "motif_tsv": motif_tsv}
    size = p.stat().st_size
    if size < min_size_bytes:
        reasons.append(f"file too small ({size} < {min_size_bytes} bytes)")
    # Inspect header
    try:
        with open(motif_tsv) as f:
            header = f.readline().rstrip("\n").split("\t")
    except Exception as ex:
        return {"status":"fail","reason":[f"header read error: {ex}"],
                "motif_tsv": motif_tsv}
    if len(header) < 7:
        reasons.append(f"header has only {len(header)} cols (need >=7)")
    # Motif columns = anything beyond the 7 standard HOMER columns
    # (PeakID, Chr, Start, End, Strand, Peak Score, Focus Ratio/Region Size)
    motif_cols = [c for c in header
                  if "/Homer Distance From Peak" in c
                  or "/HOMER" in c]
    if len(motif_cols) < min_motif_cols:
        reasons.append(
            f"only {len(motif_cols)} motif columns (need >={min_motif_cols})"
        )
    # Check required TF coverage
    missing_tfs: list = []
    if required_tfs:
        for tf in required_tfs:
            if not any(tf in c for c in motif_cols):
                missing_tfs.append(tf)
        if missing_tfs:
            reasons.append(f"missing TF columns: {missing_tfs}")
    # Row count via line count
    i = 0
    with open(motif_tsv) as f:
        for i, _ in enumerate(f, 1):
            if i > min_rows: break
    if i < min_rows:
        reasons.append(f"only {i} rows (need >={min_rows})")
    return {
        "status": "fail" if reasons else "ok",
        "reason": reasons,
        "motif_tsv": motif_tsv,
        "size_mb": round(size / 1_048_576, 1),
        "n_motif_cols": len(motif_cols),
        "missing_tfs": missing_tfs,
    }

def _existing_or_run_homer(input_bed: str, sample_id: str,
                           output_root: Path, manifest: Manifest,
                           genome: str = "hg38",
                           threads: int = 12,
                           n_concurrent: int = 4,
                           use_jaspar: bool = False) -> Optional[str]:
    """Run a single-sample HOMER pipeline if its .motif.tsv doesn't exist.

    ``n_concurrent`` is the number of HOMER workers running in parallel;
    cpu_count // n_concurrent is allocated to each one's ``-cpu`` flag to
    avoid thread oversubscription.
    """
    expected = output_root / f"{sample_id}.motif.tsv"
    if expected.exists() and expected.stat().st_size > 1000:
        # Quick row-count sanity: count up to 501 lines; reject if < 500
        with open(expected) as _f:
            _n = sum(1 for _ in zip(range(501), _f))
        if _n >= 500:
            return str(expected)
        # Truncated file — delete and re-run
        expected.unlink(missing_ok=True)
        print(f"[HOMER cache] {expected.name} has only {_n} rows — re-running", flush=True)
    # find_motifs -p: share the usable CPUs (affinity) among the concurrent workers
    threads = max(1, min(int(threads), _ncpus() // max(1, int(n_concurrent))))
    node = HomerNode(
        inputs={"sample_id": sample_id, "input_bed": input_bed},
        # MRAG_HOMER_TIMEOUT_S: raise on a heavily loaded server (2 h default was
        # exceeded per replicate on 2026-09-16 at load average ~1300)
        params={"genome": genome, "threads": threads,
                "timeout_s": int(os.environ.get("MRAG_HOMER_TIMEOUT_S", "7200")),
                "use_jaspar": use_jaspar,
                "n_parallel_tasks": n_concurrent},
        output_dir=output_root,
    )
    node.execute()
    manifest.add_step(node.to_manifest()); manifest.write_atomic()
    if node.status != "ok":
        return None
    return node.outputs.get("motif_tsv")


def build_minmax_motifs_by_arm(
    cfg, out_dir: Path, manifest: Manifest,
    gtf_path: str = "/BIO1/MRAG/Agent/Reference/ATAC_hg38/Homo_sapiens.GRCh38.104.gtf",
    use_jaspar: bool = True,
    genome: str = "hg38",
) -> dict:
    """Build per-arm min-max consensus HOMER motif.tsv from ALL ATAC replicates.

    Shared by run_motif_scan_chain (Step 6e) and the runner.py Step 6c-1a
    canonical motif_tsv fallback so BOTH paths annotate the same 2-replicate
    min-max consensus peaks (never a single replicate). Requires >=2 ATAC
    replicates per arm. Writes {arm}_peak_minmax.{bed,motif.tsv} under
    out_dir/homer_motif so the seeded fast-path and both callers cache-hit.

    Returns {"minmax_bed_by_arm": {...}, "minmax_motif_tsv_by_arm": {...}}.
    """
    if not cfg.conditions or len(cfg.conditions) < 2:
        raise ValueError("min-max consensus needs exactly 2 conditions in cfg.")
    arms = [cfg.conditions[0].label, cfg.conditions[1].label]
    homer_root = out_dir / "homer_motif"
    homer_root.mkdir(parents=True, exist_ok=True)

    # ---- Seeded fast-path -------------------------------------------------
    # When both arms' min-max BED + min-max motif.tsv are already present in
    # homer_root (restored from the protected HOMER cache), skip the expensive
    # per-replicate scans + MinmaxBed. The minmax HOMER scan below still cache-
    # hits the seeded motif.tsv via _existing_or_run_homer, and downstream
    # filter/join run on the seeded artifacts.
    arm_to_minmax_bed: dict[str, str] = {}
    import concurrent.futures as _cf
    _seeded = all(
        (homer_root / f"{arm}_peak_minmax.bed").is_file()
        and (homer_root / f"{arm}_peak_minmax.motif.tsv").is_file()
        and (homer_root / f"{arm}_peak_minmax.motif.tsv").stat().st_size > 100_000_000
        for arm in arms
    )
    if _seeded:
        for arm in arms:
            arm_to_minmax_bed[arm] = str(homer_root / f"{arm}_peak_minmax.bed")
        manifest.add_step({"name": "motif_scan_minmax_seeded", "status": "ok",
                           "outputs": [{"key": "arms", "value": list(arms)}]})
        manifest.write_atomic()

    if not _seeded:
        # ---- per-replicate HOMER scans (PARALLEL — 4 reps run concurrently) ----
        # Each scan uses ~12 threads; 4 concurrent saturates ~48 cores. Cached
        # outputs are detected before submitting (no spurious work).
        arm_to_repmotifs: dict[str, list[str]] = {arm: [None, None] for arm in arms}
        tasks = []  # (arm, idx, sample)
        for arm in arms:
            reps = _arm_samples(cfg, arm)
            if len(reps) < 2:
                raise ValueError(
                    f"the motif-scan step needs ≥2 replicates per arm; arm={arm} has {len(reps)}"
                )
            for idx, sample in enumerate(reps[:2]):
                tasks.append((arm, idx, sample))

        n_concurrent_reps = len(tasks)
        from .runner import _sample_dir   # the staged mirror the ATAC step wrote its peaks into
        def _scan_one(task):
            arm, idx, sample = task
            bed = str(_sample_dir(sample) / f"{sample.name}_peaks.narrowPeak.filt")
            return arm, idx, _existing_or_run_homer(
                input_bed=bed, sample_id=sample.name,
                output_root=homer_root, manifest=manifest,
                genome=genome,
                use_jaspar=use_jaspar,
                n_concurrent=n_concurrent_reps,
            )

        with _cf.ThreadPoolExecutor(
                max_workers=_safe_max_workers(len(tasks))) as ex:
            for arm, idx, mt in ex.map(_scan_one, tasks):
                if not mt or not Path(mt).exists():
                    raise RuntimeError(
                        f"HomerNode failed for arm={arm} rep={idx}"
                    )
                arm_to_repmotifs[arm][idx] = mt

        # ---- BARRIER: validate all 4 motif.tsv files before proceeding ----
        # If ANY rep fails integrity check (size, header, motif column count,
        # row count, required TF coverage) -> halt with explicit reason.
        barrier_failures: list[dict] = []
        barrier_oks: list[dict] = []
        for arm, motif_list in arm_to_repmotifs.items():
            for idx, mt in enumerate(motif_list):
                res = _validate_motif_tsv(
                    mt,
                    min_size_bytes=100_000,
                    min_rows=1000,
                    min_motif_cols=50,
                    required_tfs=None,  # don't fail on individual rep TF gaps;
                                        # JASPAR full lib covers all top TFs
                                        # at minmax-aggregate level
                )
                (barrier_oks if res["status"] == "ok" else barrier_failures).append({
                    "arm": arm, "rep": idx + 1, **res,
                })
        manifest.add_step({
            "name": "motif_scan_homer_barrier",
            "status": "ok" if not barrier_failures else "fail",
            "outputs": [{"key": "rep_summary",
                         "value": {"passed": barrier_oks, "failed": barrier_failures}}],
        })
        manifest.write_atomic()
        if barrier_failures:
            msg = "\n".join(
                f"  - arm={f['arm']} rep{f['rep']}: {'; '.join(f['reason'])}"
                for f in barrier_failures
            )
            raise RuntimeError(
                f"the motif-scan step HOMER barrier FAILED for {len(barrier_failures)} "
                f"replicate(s). Halting before MinmaxBedNode.\n{msg}"
            )

        # ---- MinmaxBedNode per arm (PARALLEL across arms) ----
        arm_to_minmax_node: dict[str, object] = {}

        def _run_minmax(arm):
            rep_motifs = arm_to_repmotifs[arm]
            n = MinmaxBedNode(
                inputs={
                    "tsv_a": rep_motifs[0],
                    "tsv_b": rep_motifs[1],
                    "gtf_path": gtf_path,
                },
                params={"condition_label": arm,
                        "peak_id_prefix":  f"{arm}_peak"},
                output_dir=homer_root,
            )
            n.execute()
            return arm, n

        with _cf.ThreadPoolExecutor(max_workers=_safe_max_workers(len(arms))) as ex:
            for arm, node in ex.map(_run_minmax, arms):
                manifest.add_step(node.to_manifest()); manifest.write_atomic()
                if node.status != "ok":
                    raise RuntimeError(f"MinmaxBedNode failed for arm={arm}")
                arm_to_minmax_bed[arm] = node.outputs["minmax_bed"]

    # ---- HomerNode on each minmax BED (PARALLEL across 2 arms) ----
    arm_to_minmax_motif: dict[str, str] = {}
    def _scan_minmax(arm):
        sid = f"{arm}_peak_minmax"
        return arm, sid, _existing_or_run_homer(
            input_bed=arm_to_minmax_bed[arm],
            sample_id=sid,
            output_root=homer_root,
            manifest=manifest,
            genome=genome,
            use_jaspar=use_jaspar,
            n_concurrent=len(arms),
        )
    # minmax BEDs hold ~12K peaks but far more bp than the replicate BEDs (wide
    # unions): an unchunked scan peaked at ~350 GiB (2026-09-16), not 15-20 GB.
    # annotatePeaks is memory-bounded in homer_mcp now; size the pool the same
    # way as the replicate scans.
    with _cf.ThreadPoolExecutor(
            max_workers=_safe_max_workers(len(arms))) as ex:
        for arm, sid, mt in ex.map(_scan_minmax, arms):
            if not mt:
                raise RuntimeError(f"HomerNode failed for {sid}")
            arm_to_minmax_motif[arm] = mt

    # ---- BARRIER: validate both minmax motif.tsv files before filter step ----
    # Project-agnostic: only structural checks (size, col count, row count).
    # TF-specific coverage check lives in the downstream filter step which
    # knows the actual top-N TFs for THIS run's INTEGRATED.csv.
    minmax_failures: list[dict] = []
    minmax_oks: list[dict] = []
    for arm, mt in arm_to_minmax_motif.items():
        res = _validate_motif_tsv(
            mt,
            min_size_bytes=100_000,
            min_rows=500,         # minmax BEDs are smaller (consensus peaks)
            min_motif_cols=200,   # JASPAR-extended lib has 2300+ PWMs;
                                  # >=200 confirms JASPAR was actually applied
            required_tfs=None,    # never hardcode — varies per project
        )
        (minmax_oks if res["status"] == "ok" else minmax_failures).append({
            "arm": arm, **res,
        })
    manifest.add_step({
        "name": "motif_scan_minmax_barrier",
        "status": "ok" if not minmax_failures else "fail",
        "outputs": [{"key": "minmax_summary",
                     "value": {"passed": minmax_oks, "failed": minmax_failures}}],
    })
    manifest.write_atomic()
    if minmax_failures:
        msg = "\n".join(
            f"  - arm={f['arm']}: {'; '.join(f['reason'])}"
            for f in minmax_failures
        )
        raise RuntimeError(
            f"the motif-scan step minmax HOMER barrier FAILED. Halting before "
            f"FilterTargetTfsNode.\n{msg}"
        )

    return {"minmax_bed_by_arm": arm_to_minmax_bed,
            "minmax_motif_tsv_by_arm": arm_to_minmax_motif}


def run_motif_scan_chain(
    cfg, pre, diffbind_csv: str, out_dir: Path, manifest: Manifest,
    gtf_path: str = "/BIO1/MRAG/Agent/Reference/ATAC_hg38/Homo_sapiens.GRCh38.104.gtf",
    target_tfs: Optional[list[str]] = None,
    use_jaspar: bool = True,
    genome: str = "hg38",
) -> dict:
    """Run the  canonical Step 2 chain and return paths.

    Returns:
        {
          "tfs_pval2_by_arm":  {arm_label: path},
          "tfs_txt_by_arm":    {arm_label: path},
          "minmax_bed_by_arm": {arm_label: path},
          "minmax_motif_tsv_by_arm": {arm_label: path},
        }

    """
    if target_tfs is None:
        target_tfs = DEFAULT_TARGET_TFS

    if not cfg.conditions or len(cfg.conditions) < 2:
        raise ValueError("the motif-scan step needs exactly 2 conditions in cfg.")
    arms = [cfg.conditions[0].label, cfg.conditions[1].label]

    homer_root = out_dir / "homer_motif"
    homer_root.mkdir(parents=True, exist_ok=True)

    result = {
        "tfs_pval2_by_arm":       {},
        "tfs_txt_by_arm":         {},
        "minmax_bed_by_arm":      {},
        "minmax_motif_tsv_by_arm": {},
    }

    mm = build_minmax_motifs_by_arm(cfg, out_dir, manifest, gtf_path, use_jaspar,
                                    genome=genome)
    arm_to_minmax_bed = mm["minmax_bed_by_arm"]
    arm_to_minmax_motif = mm["minmax_motif_tsv_by_arm"]

    # ---- FilterTargetTfsNode ----
    arm_to_tfs_txt: dict[str, str] = {}
    for arm in arms:
        node = FilterTargetTfsNode(
            inputs={"motif_tsv": arm_to_minmax_motif[arm]},
            params={"condition_label": arm, "target_tfs": list(target_tfs)},
            output_dir=homer_root,
        )
        node.execute()
        manifest.add_step(node.to_manifest()); manifest.write_atomic()
        if node.status != "ok":
            raise RuntimeError(f"FilterTargetTfsNode failed for arm={arm}")
        arm_to_tfs_txt[arm] = node.outputs["tfs_txt"]

    # ---- JoinDiffbindRnaNode ----
    arm_to_pval2: dict[str, str] = {}
    for arm in arms:
        node = JoinDiffbindRnaNode(
            inputs={
                "tfs_txt":      arm_to_tfs_txt[arm],
                "diffbind_csv": diffbind_csv,
                "deseq_csv":    pre.deseq_csv,
            },
            params={"condition_label": arm},
            output_dir=homer_root,
        )
        node.execute()
        manifest.add_step(node.to_manifest()); manifest.write_atomic()
        if node.status != "ok":
            raise RuntimeError(f"JoinDiffbindRnaNode failed for arm={arm}")
        arm_to_pval2[arm] = node.outputs["tfs_pval2_txt"]

    result.update({
        "tfs_pval2_by_arm":        arm_to_pval2,
        "tfs_txt_by_arm":          arm_to_tfs_txt,
        "minmax_bed_by_arm":       arm_to_minmax_bed,
        "minmax_motif_tsv_by_arm": arm_to_minmax_motif,
    })
    # Persist a summary JSON for downstream consumers
    summary_path = out_dir / "motif_scan_summary.json"
    summary_path.write_text(json.dumps(result, indent=2))
    return result
