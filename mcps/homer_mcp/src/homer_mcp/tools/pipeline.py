"""HOMER perl subprocess wrappers — find_motifs / annotate_peaks / pipeline."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP


# HOMER home auto-discovery:
# 1. $HOMER_HOME if set
# 2. First candidate that has data/genomes/hg38/ populated
# 3. Fall back to /BIO1/MRAG/Agent/bin/tools/homer (bin-only; genome must be
# installed before find_motifs / annotate_peaks can run)
# The candidates are this host's installs (the second one carries the hg38 /
# mm10 genomes); on any other machine set HOMER_HOME.
_HOMER_CANDIDATES = [
    "/BIO1/MRAG/Agent/bin/tools/homer",
    "/BIO1/MRAG/HYK/homer",
]

def _discover_homer_home() -> Path:
    env = os.environ.get("HOMER_HOME")
    if env:
        return Path(env)
    for c in _HOMER_CANDIDATES:
        cp = Path(c)
        if cp.is_dir() and (cp / "data/genomes/hg38").is_dir():
            return cp
    # nothing had hg38 -- fall back to first existing candidate
    for c in _HOMER_CANDIDATES:
        if Path(c).is_dir():
            return Path(c)
    return Path(_HOMER_CANDIDATES[0])

_HOMER_HOME = _discover_homer_home()
_HOMER_BIN  = _HOMER_HOME / "bin"

_FIND_MOTIFS    = _HOMER_BIN / "findMotifsGenome.pl"
_ANNOTATE_PEAKS = _HOMER_BIN / "annotatePeaks.pl"

# JASPAR-extended motif library is the MRAG default — HOMER's built-in motif
# library lacks coverage for several common TFs (e.g. STAT6, EGR1, E2F2).
# combined_v12_jaspar.motif merges HOCOMOCOv12 + JASPAR2024 = 2322 motifs.
JASPAR_DEFAULT_MOTIF = Path("/BIO1/MRAG/Agent/Reference/combined_v12_jaspar.motif")

# perl binary (HOMER scripts have #!/usr/bin/env perl shebang but call with explicit)
_PERL = os.environ.get("HOMER_PERL") or shutil.which("perl") or "/usr/bin/perl"


def _homer_env():
    """Subprocess env with HOMER bin (homer2, bed2pos.pl, etc.) on PATH.

    HOMER's perl drivers shell out to companion executables (homer2,
    bed2pos.pl, checkPeakFile.pl, homerTools, etc.) via plain `sh`, so the
    binary directory MUST be on PATH or those calls silently fail and
    annotatePeaks writes only the header.
    """
    env = dict(os.environ)
    env["PATH"] = str(_HOMER_BIN) + ":" + env.get("PATH", "")
    return env


def _auto_cpus(n_parallel_tasks: int = 1) -> int:
    """Auto-divide the usable cpus across N concurrent tasks. Min 2.

    Use case: orchestrator runs 4 samples in parallel; each annotate_peaks call
    gets cpus // 4 threads. "cpus" is the CPU affinity of this process
    (taskset / MRAG_CPU_LIST), not the host size: os.cpu_count() reports all
    128 host cpus even under a 16-cpu pin (2026-09-16).
    """
    import os as _os
    try:
        cpu = len(_os.sched_getaffinity(0)) or 8
    except (AttributeError, OSError):
        cpu = _os.cpu_count() or 8
    n = max(1, int(n_parallel_tasks))
    return max(2, cpu // n)


def _check_homer():
    errs = []
    if not _FIND_MOTIFS.is_file():
        errs.append(f"findMotifsGenome.pl not found at {_FIND_MOTIFS}")
    if not _ANNOTATE_PEAKS.is_file():
        errs.append(f"annotatePeaks.pl not found at {_ANNOTATE_PEAKS}")
    if not Path(_PERL).is_file():
        errs.append(f"perl not found at {_PERL}")
    return errs


def register_pipeline_tools(mcp: FastMCP):

    @mcp.tool()
    def find_motifs(
        input_bed: str,
        output_dir: str,
        genome: str = "hg38",
        mask_repeats: bool = True,
        size: Optional[str] = None,
        threads: int = 4,
        extra_args: Optional[List[str]] = None,
        timeout_s: int = 3600,
        dry_run: bool = False,
    ) -> str:
        """Run HOMER findMotifsGenome.pl on a peak BED.

        Produces ``output_dir/`` with knownResults/, homerResults/,
        homerMotifs.all.motifs, nonRedundant.motifs, etc.

        Args:
            input_bed:    Peak BED (e.g. MACS2 narrowPeak or HOMER .input.bed).
            output_dir:   Output directory; will be created if missing.
            genome:       HOMER genome label (default "hg38"). Must be
                          already installed via configureHomer.pl.
            mask_repeats: Pass ``-mask`` (reference-analysis default = True).
            size:         Window size around peaks. None -> HOMER default.
                          E.g. "200" or "given".
            threads:     ``-p`` argument (default 4).
            extra_args:   Extra args appended verbatim.
            timeout_s:    Subprocess timeout (default 1h).
            dry_run:      Only return the assembled command.

        Returns JSON with elapsed time, command, exit code, and stderr tail.
        """
        errs = _check_homer()
        if errs:
            return json.dumps({"status": "error", "message": "homer not found", "errors": errs})
        bed = Path(input_bed).expanduser()
        if not bed.is_file():
            return json.dumps({"status": "error", "message": f"input BED not found: {bed}"})
        out = Path(output_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)

        cmd = [
            _PERL, str(_FIND_MOTIFS),
            str(bed), genome, str(out),
            "-p", str(int(threads)),
        ]
        if mask_repeats:
            cmd.append("-mask")
        if size:
            cmd.extend(["-size", size])
        if extra_args:
            cmd.extend(extra_args)

        if dry_run:
            return json.dumps({"status": "dry_run", "command": cmd}, ensure_ascii=False, indent=2)

        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=_homer_env())
        except subprocess.TimeoutExpired as e:
            return json.dumps({"status": "timeout", "elapsed_s": round(time.time() - t0, 1),
                               "timeout_s": timeout_s,
                               "stderr_tail": (e.stderr or "")[-2000:] if hasattr(e, "stderr") and e.stderr else ""},
                              ensure_ascii=False, indent=2, default=str)
        elapsed = round(time.time() - t0, 2)
        ok = proc.returncode == 0
        return json.dumps(
            {
                "status":     "ok" if ok else "error",
                "returncode": proc.returncode,
                "elapsed_s":  elapsed,
                "output_dir": str(out),
                "command":    cmd,
                "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
                "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
            },
            ensure_ascii=False, indent=2, default=str,
        )

    @mcp.tool()
    def concat_known_motifs(peakanalysis_dir: str, output_motif: str) -> str:
        """Concatenate all knownResults/*.motif into one HOMER motif file.

        Mirrors the reference analysis step:
            cat peakAnalysis/{sample}/knownResults/*.motif > {sample}.motif

        Args:
            peakanalysis_dir: ``peakAnalysis/{sample}/`` directory.
            output_motif:     Combined motif file path.
        """
        d = Path(peakanalysis_dir).expanduser()
        known = d / "knownResults"
        if not known.is_dir():
            return json.dumps({"status": "error",
                               "message": f"knownResults/ not found under {d}"})
        motifs = sorted(known.glob("*.motif"))
        if not motifs:
            return json.dumps({"status": "error",
                               "message": f"no .motif files in {known}"})
        out = Path(output_motif).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fout:
            for m in motifs:
                fout.write(m.read_text())
        return json.dumps(
            {"status": "ok", "n_motifs": len(motifs), "output": str(out)},
            ensure_ascii=False, indent=2,
        )

    @mcp.tool()
    def annotate_peaks(
        input_bed: str,
        output_tsv: str,
        motif_file: Optional[str] = None,
        genome: str = "hg38",
        extra_args: Optional[List[str]] = None,
        timeout_s: int = 1800,
        dry_run: bool = False,
        n_cpus: Optional[int] = None,
        n_parallel_tasks: int = 1,
    ) -> str:
        """Run HOMER annotatePeaks.pl with motif scanning.

        Mirrors :
            perl annotatePeaks.pl {sample}_merge.input.bed hg38 \
                 -m {sample}.motif > {sample}.motif.tsv

        Output schema: PeakID, Chr, Start, End, Strand, Peak Score,
        Focus Ratio/Region Size, Annotation, Detailed Annotation,
        Distance to TSS, Nearest PromoterID, Entrez ID, Nearest Unigene,
        Nearest Refseq, Nearest Ensembl, Gene Name, Gene Alias,
        Gene Description, Gene Type, CpG%, GC%,
        <motif_1>, <motif_2>, ... (one column per motif in motif_file).

        Args:
            input_bed:   Peak BED file.
            motif_file:  Concatenated .motif file (typically from
                         concat_known_motifs).
            output_tsv:  Where to write the TSV.
            genome:      HOMER genome label.
            extra_args:  Extra args appended verbatim.
            timeout_s:   Subprocess timeout (default 30 min).
            dry_run:     Only return the command.
        """
        errs = _check_homer()
        if errs:
            return json.dumps({"status": "error", "message": "homer not found", "errors": errs})
        bed = Path(input_bed).expanduser()
        if not motif_file:
            motif_file = str(JASPAR_DEFAULT_MOTIF)
        motif = Path(motif_file).expanduser()
        if not bed.is_file():
            return json.dumps({"status": "error", "message": f"BED not found: {bed}"})
        if not motif.is_file():
            return json.dumps({"status": "error", "message": f"motif file not found: {motif}"})
        out = Path(output_tsv).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            _PERL, str(_ANNOTATE_PEAKS),
            str(bed), genome,
            "-m", str(motif),
        ]
        # Auto-allocate cpus to fill host evenly when N parallel tasks share it.
        actual_cpus = int(n_cpus) if n_cpus else _auto_cpus(n_parallel_tasks)
        cmd.extend(["-cpu", str(actual_cpus)])
        if extra_args:
            cmd.extend(extra_args)

        if dry_run:
            return json.dumps(
                {"status": "dry_run", "command": cmd, "output_tsv": str(out)},
                ensure_ascii=False, indent=2,
            )

        # Memory bound (2026-09-16, annotate_bounded.py): an input whose estimated
        # annotatePeaks memory exceeds MRAG_HOMER_ANNOT_MAX_GB is annotated in
        # chunks inside the account-wide HOMER memory budget and merged into the
        # same table; an input that fits runs the single call below unchanged
        # after taking a shared memory slot. MRAG_HOMER_CHUNKING=0 = old behaviour.
        from . import annotate_bounded as _ab
        try:
            mem_plan = _ab.plan_annotation(bed, motif, extra_args)
        except Exception as _e:
            mem_plan = {"mode": "legacy", "reason": f"planning failed: {_e}"}
        if mem_plan.get("mode") == "chunked":
            try:
                res = _ab.annotate_chunked(
                    perl=_PERL, homer_bin=_HOMER_BIN, script=_ANNOTATE_PEAKS,
                    bed=bed, genome=genome, motif=motif, out=out,
                    header_args=cmd[2:], extra_args=extra_args,
                    timeout_s=timeout_s, env=_homer_env(), plan=mem_plan,
                )
                res["command"] = cmd
                res["memory_plan"] = mem_plan
                return json.dumps(res, ensure_ascii=False, indent=2, default=str)
            except _ab.NotChunkable as _e:
                mem_plan = dict(mem_plan, mode="single", reason=f"not chunkable: {_e}")
            except Exception as _e:
                return json.dumps({"status": "error", "output": str(out), "command": cmd,
                                   "message": f"chunked annotatePeaks failed: {type(_e).__name__}: {_e}",
                                   "memory_plan": mem_plan},
                                  ensure_ascii=False, indent=2, default=str)
        _slot = None
        if mem_plan.get("mode") == "single":
            _slot = _ab.acquire_slots(mem_plan.get("slots", 1), label=out.name)
        # The table goes to <out>.part and is renamed only after a clean exit
        # (2026-09-17): a timeout / crash no longer leaves a truncated
        # .motif.tsv that the HOMER cache check (>= 500 rows) would accept.
        part = out.with_name(out.name + ".part")
        try:
            t0 = time.time()
            try:
                with part.open("w") as fout:
                    proc = subprocess.run(
                        cmd,
                        stdout=fout,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=timeout_s,
                        env=_homer_env(),
                    )
            except subprocess.TimeoutExpired as e:
                part.unlink(missing_ok=True)
                return json.dumps({"status": "timeout", "elapsed_s": round(time.time() - t0, 1),
                                   "timeout_s": timeout_s}, ensure_ascii=False, indent=2)
            except BaseException:
                part.unlink(missing_ok=True)
                raise
            elapsed = round(time.time() - t0, 2)
            ok = proc.returncode == 0
            # Quick row count
            try:
                n_rows = sum(1 for _ in open(part)) - 1
            except Exception:
                n_rows = -1
            if ok:
                os.replace(part, out)
            else:
                part.unlink(missing_ok=True)
            return json.dumps(
                {
                    "status":     "ok" if ok else "error",
                    "returncode": proc.returncode,
                    "elapsed_s":  elapsed,
                    "output":     str(out),
                    "n_rows":     n_rows,
                    "command":    cmd,
                    "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
                },
                ensure_ascii=False, indent=2, default=str,
            )
        finally:
            _ab.release_slots(_slot)

    @mcp.tool()
    def annotate_peaks_top_n(
        input_bed: str,
        output_tsv: str,
        integrated_csv: str = "",
        top_tfs: Optional[List[str]] = None,
        top_n: int = 20,
        jaspar_db: Optional[str] = None,
        subset_out: str = "",
        genome: str = "hg38",
        timeout_s: int = 1800,
        n_parallel_tasks: int = 1,
    ) -> str:
        """Subset JASPAR motif lib to top-N TFs (from INTEGRATED.csv) then
        annotatePeaks.pl on the small subset. This is the FAST DEFAULT path
        for MRAG full-FASTQ HOMER runs:

          * Avoids the 10-15 min de novo find_motifs step.
          * Avoids scanning all 2322 motifs at every peak.
          * Result: same downstream consensus columns as the slow path, but
            ~50x faster (~3 min vs 2 hours per 4-sample run).

        TF source (one of):
          - `top_tfs`: explicit list (overrides everything).
          - `integrated_csv`: read INTEGRATED.csv and pick top-N via the
            canonical sort (Pcom asc + |log2FC| desc + |MD| desc + Gene asc).
        """
        from .motif_subset import subset_motif_file_impl
        import importlib, sys as _sys
        _sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/combined_mcp/src")
        from combined_mcp.tools.pipeline import _topn_sort_key
        import pandas as pd

        # Resolve target TF list
        tfs: List[str] = list(top_tfs) if top_tfs else []
        if not tfs:
            if not integrated_csv:
                return json.dumps({"status": "error",
                                   "message": "supply either top_tfs or integrated_csv"})
            df = pd.read_csv(integrated_csv)
            tfs = _topn_sort_key(df).head(int(top_n))["Gene"].tolist()

        # Resolve motif DB
        jaspar = Path(jaspar_db) if jaspar_db else JASPAR_DEFAULT_MOTIF
        if not jaspar.is_file():
            return json.dumps({"status": "error", "message": f"jaspar db not found: {jaspar}"})

        # Subset
        out_root = Path(output_tsv).parent
        subset_path = Path(subset_out) if subset_out else (out_root / f"top{top_n}_subset.motif")
        sub_res = subset_motif_file_impl(
            motif_db=str(jaspar),
            tf_symbols=tfs,
            output_motif=str(subset_path),
            include_heterodimers=True,
        )
        if not sub_res.get("ok"):
            return json.dumps({"status": "error",
                               "message": "subset_motif_file failed",
                               "subset_result": sub_res})

        # Annotate
        anno = json.loads(annotate_peaks(
            input_bed=input_bed,
            output_tsv=output_tsv,
            motif_file=str(subset_path),
            genome=genome,
            timeout_s=timeout_s,
            n_parallel_tasks=n_parallel_tasks,
        ))
        return json.dumps({
            "status": anno.get("status"),
            "annotated_tsv": output_tsv,
            "subset_motif":  str(subset_path),
            "n_tfs_requested": len(tfs),
            "n_pwms_in_subset": sub_res.get("n_kept", 0),
            "missing_tfs":   sub_res.get("missing_tfs", []),
            "elapsed_s":     anno.get("elapsed_s"),
            "tfs": tfs,
        }, ensure_ascii=False, indent=2)

    @mcp.tool()
    def run_homer_pipeline(
        input_bed: str,
        sample_id: str,
        output_root: str,
        genome: str = "hg38",
        threads: int = 4,
        timeout_s: int = 5400,
        dry_run: bool = False,
        use_jaspar: bool = True,
        n_parallel_tasks: int = 1,
    ) -> str:
        """Composite: find_motifs -> concat_known_motifs -> annotate_peaks.

        Produces under ``output_root``:
            peakAnalysis/{sample_id}/                  (HOMER motif results)
            {sample_id}.motif                          (concatenated known motifs)
            {sample_id}.motif.tsv                      (annotated peaks)

        Heavy step (~20-40 min/sample). Skips find_motifs if peakAnalysis
        is already populated (idempotent rerun).

        use_jaspar (default True): supplement the per-sample HOMER known
        motifs with the JASPAR-extended library (combined_v12_jaspar.motif —
        HOCOMOCOv12 + JASPAR2024, 2322 motifs). Critical for MRAG: without
        it, TFs like STAT6 / EGR1 / E2F2 have NO motif columns and the
        downstream TRN graph is sparse.
        """
        out_root = Path(output_root).expanduser()
        out_root.mkdir(parents=True, exist_ok=True)
        peak_dir = out_root / "peakAnalysis" / sample_id
        motif_path = out_root / f"{sample_id}.motif"
        tsv_path   = out_root / f"{sample_id}.motif.tsv"

        plan = {
            "sample_id":  sample_id,
            "input_bed":  input_bed,
            "peak_dir":   str(peak_dir),
            "motif_file": str(motif_path),
            "tsv_file":   str(tsv_path),
            "steps":      [],
        }

        steps_to_run = []
        if not (peak_dir / "knownResults").is_dir():
            steps_to_run.append("find_motifs")
        else:
            plan["steps"].append({"find_motifs": "skipped (peakAnalysis/knownResults already exists)"})
        steps_to_run.append("concat_known_motifs")
        steps_to_run.append("annotate_peaks")

        if dry_run:
            plan["steps_to_run"] = steps_to_run
            return json.dumps({"status": "dry_run", "plan": plan}, ensure_ascii=False, indent=2)

        # Internal helpers re-use the same lambdas as the tools above. We just
        # call them inline by reusing the registered tool functions via closure
        # — but for simplicity here we call subprocess directly mirroring them.
        t0 = time.time()
        results: List[Any] = []

        def _append_jaspar(target: Path):
            if not JASPAR_DEFAULT_MOTIF.exists():
                return False
            with target.open("ab") as out, JASPAR_DEFAULT_MOTIF.open("rb") as jf:
                out.write(b"\n")
                out.write(jf.read())
            return True

        # 1) find_motifs (if needed)
        if "find_motifs" in steps_to_run:
            r = json.loads(find_motifs(
                input_bed=input_bed,
                output_dir=str(peak_dir),
                genome=genome,
                mask_repeats=True,
                threads=threads,
                timeout_s=timeout_s,
            ))
            results.append({"find_motifs": r})
            if r.get("status") != "ok":
                return json.dumps({"status": "error", "where": "find_motifs",
                                   "results": results}, ensure_ascii=False, indent=2)

        # 2) concat_known_motifs
        r = json.loads(concat_known_motifs(str(peak_dir), str(motif_path)))
        results.append({"concat_known_motifs": r})
        if r.get("status") != "ok":
            return json.dumps({"status": "error", "where": "concat_known_motifs",
                               "results": results}, ensure_ascii=False, indent=2)

        # 2b) supplement with JASPAR-extended library (default ON for MRAG)
        if use_jaspar:
            ok = _append_jaspar(motif_path)
            results.append({"append_jaspar": {"appended": ok,
                                                  "motif_lib": str(JASPAR_DEFAULT_MOTIF)}})

        # 3) annotate_peaks
        r = json.loads(annotate_peaks(
            input_bed=input_bed,
            motif_file=str(motif_path),
            output_tsv=str(tsv_path),
            genome=genome,
            timeout_s=timeout_s,
            n_parallel_tasks=n_parallel_tasks,
        ))
        results.append({"annotate_peaks": r})
        if r.get("status") != "ok":
            # 2026-09-16: a failed / timed-out annotation used to be reported as
            # "ok" here, letting a truncated .motif.tsv through HomerNode.
            return json.dumps({"status": "error", "where": "annotate_peaks",
                               "elapsed_s": round(time.time() - t0, 2),
                               "plan": plan, "results": results},
                              ensure_ascii=False, indent=2, default=str)

        elapsed = round(time.time() - t0, 2)
        return json.dumps(
            {
                "status":    "ok",
                "elapsed_s": elapsed,
                "plan":      plan,
                "results":   results,
                "outputs": {
                    "peak_dir":   str(peak_dir),
                    "motif_file": str(motif_path),
                    "tsv_file":   str(tsv_path),
                },
            },
            ensure_ascii=False, indent=2, default=str,
        )

    @mcp.tool()
    def homer_check_env() -> str:
        """Verify HOMER binaries + perl + installed genomes."""
        errs = _check_homer()
        info: Dict[str, Any] = {
            "homer_home":     str(_HOMER_HOME),
            "find_motifs":    str(_FIND_MOTIFS),
            "annotate_peaks": str(_ANNOTATE_PEAKS),
            "perl":           _PERL,
            "errors":         errs,
        }
        # Discover installed genomes two ways:
        # 1. config.txt GENOMES section (declarative)
        # 2. data/genomes/* directories (actual)
        cfg = _HOMER_HOME / "config.txt"
        from_config: List[Dict[str, Any]] = []
        if cfg.is_file():
            in_genomes = False
            for line in cfg.read_text().splitlines():
                if line.strip() == "GENOMES":
                    in_genomes = True
                    continue
                if in_genomes:
                    if not line.strip() or line.startswith("#") or line.strip().isupper():
                        if line.strip().isupper() and line.strip() != "GENOMES":
                            in_genomes = False
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        from_config.append({"name": parts[0], "version": parts[1]})
        from_disk: List[str] = []
        genome_dir = _HOMER_HOME / "data/genomes"
        if genome_dir.is_dir():
            from_disk = sorted(d.name for d in genome_dir.iterdir() if d.is_dir())
        info["genomes_from_config"] = from_config
        info["genomes_on_disk"]     = from_disk
        info["has_hg38"]            = "hg38" in from_disk
        info["status"] = "ok" if (not errs and from_disk) else "error"
        if from_disk == []:
            info["hint"] = ("No genomes installed at this HOMER_HOME. "
                            "Install via configureHomer.pl -install hg38, "
                            "or set HOMER_HOME env to a HOMER with hg38, "
                            "e.g. HOMER_HOME=/path/to/homer")
        return json.dumps(info, ensure_ascii=False, indent=2)
