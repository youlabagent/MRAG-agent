"""Pipeline tools — ATAC-seq steps 0 through 5."""
from __future__ import annotations
import fcntl
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# DAStk's differential step, run through a wrapper that seeds its per-motif bootstrap (reproducible
# p-values; see dastk_seeded.py)
_DASTK_SEEDED = str(Path(__file__).with_name("dastk_seeded.py"))


_PE_SUFFIX_PAIRS = [
    ("_R1.fastq.gz", "_R2.fastq.gz"),
    ("_R1.fq.gz", "_R2.fq.gz"),
    ("_1.fastq.gz", "_2.fastq.gz"),
    ("_1.fq.gz", "_2.fq.gz"),
    ("_R1_001.fastq.gz", "_R2_001.fastq.gz"),
]


def _find_fastqs(project_dir: str, sample: str) -> tuple[str, str]:
    """Locate paired-end ATAC FASTQs across common naming conventions."""
    d = Path(project_dir)
    tried: list[str] = []
    for s1, s2 in _PE_SUFFIX_PAIRS:
        r1 = d / f"{sample}{s1}"
        r2 = d / f"{sample}{s2}"
        tried += [str(r1), str(r2)]
        if r1.exists() and r2.exists():
            return str(r1), str(r2)
    raise FileNotFoundError(
        f"No paired FASTQs found for sample '{sample}' in {project_dir}. Tried: "
        + ", ".join(tried)
    )


def _canonicalize_inputs(project_dir: str, sample: str, r1: str, r2: str) -> tuple[str, str]:
    """Symlink non-canonical FASTQ names to {sample}_R1/_R2.fastq.gz."""
    d = Path(project_dir)
    c1 = d / f"{sample}_R1.fastq.gz"
    c2 = d / f"{sample}_R2.fastq.gz"
    if c1.resolve() != Path(r1).resolve():
        c1.unlink(missing_ok=True)
        c1.symlink_to(r1)
    if c2.resolve() != Path(r2).resolve():
        c2.unlink(missing_ok=True)
        c2.symlink_to(r2)
    return str(c1), str(c2)


def _list_samples(project_dir: str) -> list[str]:
    """Return unique sample prefixes across common PE naming conventions."""
    d = Path(project_dir)
    names: set[str] = set()
    for fq in sorted(d.glob("*.fastq.gz")) + sorted(d.glob("*.fq.gz")):
        m = re.match(r"^(.+?)(?:_R1(?:_001)?|_1)\.(?:fastq|fq)\.gz$", fq.name)
        if m:
            names.add(m.group(1))
    return sorted(names)


def _bam_quickcheck(executor, bam_path: str) -> str | None:
    """Validate BAM via samtools quickcheck. Returns None on OK, error string otherwise."""
    p = Path(bam_path)
    if not p.exists():
        return f"BAM missing: {bam_path}"
    if p.stat().st_size < 1024:
        return f"BAM too small ({p.stat().st_size} bytes), likely truncated: {bam_path}"
    out = executor.run(f"samtools quickcheck {executor.quote(bam_path)}; echo $?")
    if "[EXIT CODE]" in out or "[ERROR]" in out or "[TIMEOUT]" in out:
        return f"BAM failed samtools quickcheck: {bam_path}"
    return None


def _file_nonempty(path: str, min_bytes: int = 100) -> bool:
    p = Path(path)
    return p.exists() and p.stat().st_size >= min_bytes


def register_pipeline_tools(mcp: FastMCP, executor):
    """Register each pipeline step as an MCP tool."""
    q = executor.quote  # shorthand for shlex.quote

    @mcp.tool()
    def step0_preprocessing(
        project_dir: str,
        sample: str,
        num_threads: int = 0,
    ) -> str:
        """Step 0: Trim adapters (trim_galore) and run FastQC.

        Args:
            project_dir: Absolute path to the ATAC project directory containing raw FASTQs.
            sample:      Sample prefix (e.g. 'SampleA'). Expects {sample}_R1.fastq.gz.
            num_threads: CPU threads. 0 = auto-detect.
        """
        safe_dir = str(executor._resolve_project_dir(project_dir))
        if executor.is_done(safe_dir, sample, "step0"):
            return f"[SKIP] step0 already completed for {sample}"
        try:
            r1, r2 = _find_fastqs(safe_dir, sample)
        except FileNotFoundError as e:
            return f"[ERROR] {e}"
        _canonicalize_inputs(safe_dir, sample, r1, r2)

        res = executor.detect_resources()
        t = num_threads or res["cores"]
        s = q(sample)
        d = q(safe_dir)

        cmds = f"""
set -euo pipefail
cd {d}
mkdir -p trim_galore fastqc
trim_galore --paired --gzip -o trim_galore/ {s}_R1.fastq.gz {s}_R2.fastq.gz
fastqc -t {t} --nogroup -o fastqc/ trim_galore/{s}_R1_val_1.fq.gz
fastqc -t {t} --nogroup -o fastqc/ trim_galore/{s}_R2_val_2.fq.gz
"""
        out = executor.run(cmds, cwd=safe_dir)
        if executor.is_success(out):
            executor.mark_done(safe_dir, sample, "step0")
        return out

    @mcp.tool()
    def step1_alignment(
        project_dir: str,
        sample: str,
        ref_dir: str = "",
        genome: str = "hg38",
        num_threads: int = 0,
    ) -> str:
        """Step 1: Bowtie2 alignment + SAM-to-BAM conversion + sorting + indexing.

        Args:
            project_dir: Project directory with trim_galore/ outputs.
            sample:      Sample prefix.
            ref_dir:     Reference directory containing bowtie2 index.
            genome:      Genome build name (hg38, mm10, etc.).
            num_threads: CPU threads. 0 = auto-detect.
        """
        safe_dir = str(executor._resolve_project_dir(project_dir))
        if executor.is_done(safe_dir, sample, "step1"):
            return f"[SKIP] step1 already completed for {sample}"

        res = executor.detect_resources()
        t = num_threads or res["cores"]
        rd = ref_dir or "/BIO1/MRAG/Agent/Reference/ATAC_hg38/"
        adir = f"{safe_dir}/alignments"
        s = q(sample)

        cmds = f"""
set -euo pipefail
cd {q(safe_dir)}
mkdir -p {q(adir)}
# Resilience: discard any partial outputs from a prior interrupted run
if [ -e {q(adir)}/{s}.sorted.bam ] && ! samtools quickcheck {q(adir)}/{s}.sorted.bam 2>/dev/null; then
  echo "[step1] removing corrupt sorted.bam from prior run"
  rm -f {q(adir)}/{s}.sorted.bam {q(adir)}/{s}.sorted.bam.bai
fi
if [ -e {q(adir)}/{s}.bam ] && ! samtools quickcheck {q(adir)}/{s}.bam 2>/dev/null; then
  rm -f {q(adir)}/{s}.bam
fi
if [ -s {q(adir)}/{s}.sorted.bam ]; then
  echo "[step1] reusing valid sorted.bam for {sample}"
elif [ -s {q(adir)}/{s}.sam ]; then
  echo "[step1] reusing existing SAM for {sample}, skipping bowtie2"
else
  bowtie2 -p {t} -x {q(rd)}{q(genome)} \
    -1 trim_galore/{s}_R1_val_1.fq.gz \
    -2 trim_galore/{s}_R2_val_2.fq.gz \
    -S {q(adir)}/{s}.sam 2>{q(adir)}/{s}.bowtie2Log
fi
if [ ! -s {q(adir)}/{s}.sorted.bam ]; then
  samtools view -bS {q(adir)}/{s}.sam > {q(adir)}/{s}.bam
  samtools sort -@ {t} {q(adir)}/{s}.bam > {q(adir)}/{s}.sorted.bam
  samtools index {q(adir)}/{s}.sorted.bam
  rm -f {q(adir)}/{s}.sam {q(adir)}/{s}.bam
fi
"""
        out = executor.run(cmds, cwd=safe_dir)
        if not executor.is_success(out):
            return out
        # Output sanity: sorted.bam must pass quickcheck before marking done
        sorted_bam = f"{adir}/{sample}.sorted.bam"
        bam_err = _bam_quickcheck(executor, sorted_bam)
        if bam_err:
            return out + f"\n[ERROR] {bam_err}. Not marking step1 done."
        executor.mark_done(safe_dir, sample, "step1")
        return out

    @mcp.tool()
    def step2_filtering(
        project_dir: str,
        sample: str,
        ref_dir: str = "",
        genome: str = "hg38",
        picard_jar: str = "/BIO1/picard.jar",
    ) -> str:
        """Step 2: MAPQ>=30 filter, remove chrM, mark/remove duplicates, generate bigwig.

        Args:
            project_dir: Project directory.
            sample:      Sample prefix.
            ref_dir:     Reference directory.
            genome:      Genome build name.
            picard_jar:  Path to picard.jar.
        """
        safe_dir = str(executor._resolve_project_dir(project_dir))
        if executor.is_done(safe_dir, sample, "step2"):
            return f"[SKIP] step2 already completed for {sample}"

        res = executor.detect_resources()
        # 2026-09-16: MemAvailable//4 gave ~125-250 GB heaps x 4 parallel samples
        import os as _os
        xmx = max(4, min(res["mem_gb"] // 4, int(_os.environ.get("MRAG_PICARD_XMX_MAX_GB", "16"))))
        rd = ref_dir or "/BIO1/MRAG/Agent/Reference/ATAC_hg38/"
        adir = f"{safe_dir}/alignments"
        pdir = f"{safe_dir}/TEMP_PICARD"
        s = q(sample)

        # Input validation: sorted.bam must exist and be a complete BAM
        bam_err = _bam_quickcheck(executor, f"{adir}/{sample}.sorted.bam")
        if bam_err:
            return f"[ERROR] {bam_err}. Run step1 first."

        cmds = f"""
set -euo pipefail
cd {q(safe_dir)}
mkdir -p {q(pdir)}
# Resilience: discard partial intermediates from interrupted prior run
rm -f {q(adir)}/{s}.filtered.bam {q(adir)}/{s}.filtered.noChrM.bam \
      {q(adir)}/{s}.dedup.sam {q(adir)}/{s}.dedup.bam {q(adir)}/{s}.dedup.bai \
      {q(adir)}/{s}.duplicate_metrics
bamtools filter -in {q(adir)}/{s}.sorted.bam -out {q(adir)}/{s}.filtered.bam \
  -mapQuality ">=30" -isProperPair true
samtools view {q(adir)}/{s}.filtered.bam | grep -v -E 'chrM' | \
  samtools view -bT {q(rd)}{q(genome)}.fa -o {q(adir)}/{s}.filtered.noChrM.bam
java -Xmx{xmx}g -jar {q(picard_jar)} MarkDuplicates \
  TMP_DIR={q(pdir)} VALIDATION_STRINGENCY=LENIENT \
  I={q(adir)}/{s}.filtered.noChrM.bam \
  O={q(adir)}/{s}.dedup.sam \
  M={q(adir)}/{s}.duplicate_metrics \
  REMOVE_DUPLICATES=true AS=true
java -Xmx{xmx}g -jar {q(picard_jar)} SortSam \
  TMP_DIR={q(pdir)} VALIDATION_STRINGENCY=LENIENT \
  SO=coordinate I={q(adir)}/{s}.dedup.sam \
  O={q(adir)}/{s}.dedup.bam CREATE_INDEX=true
rm -f {q(adir)}/{s}.dedup.sam
bamCoverage -p 16 -b {q(adir)}/{s}.dedup.bam -o {q(safe_dir)}/{s}.bw
"""
        out = executor.run(cmds, cwd=safe_dir)
        if not executor.is_success(out):
            return out
        # Output sanity: dedup.bam must pass quickcheck and bw must be non-empty
        bam_err = _bam_quickcheck(executor, f"{adir}/{sample}.dedup.bam")
        if bam_err:
            return out + f"\n[ERROR] {bam_err}. Not marking step2 done."
        if not _file_nonempty(f"{safe_dir}/{sample}.bw", min_bytes=10_000):
            return out + f"\n[ERROR] {sample}.bw missing or too small. Not marking step2 done."
        executor.mark_done(safe_dir, sample, "step2")
        return out

    @mcp.tool()
    def step3_peak_calling(
        project_dir: str,
        sample: str,
        ref_dir: str = "",
        species: str = "hs",
        blacklist: str = "",
    ) -> str:
        """Step 3: MACS2 peak calling + blacklist filtering.

        Args:
            project_dir: Project directory.
            sample:      Sample prefix.
            ref_dir:     Reference directory (for blacklist BED).
            species:     MACS2 genome size: hs (human) or mm (mouse).
        """
        safe_dir = str(executor._resolve_project_dir(project_dir))
        if executor.is_done(safe_dir, sample, "step3"):
            return f"[SKIP] step3 already completed for {sample}"

        rd = ref_dir or "/BIO1/MRAG/Agent/Reference/ATAC_hg38/"
        adir = f"{safe_dir}/alignments"
        s = q(sample)

        # The blacklist filename is assembly-specific, so resolve it inside the
        # reference directory rather than assuming the human one. Hardcoding it
        # forced the mm39 reference to ship its mouse blacklist under the human
        # name, which is silent and confusing.
        bl_expr = q(blacklist) if blacklist else "$(ls " + rd + "*blacklist*.bed 2>/dev/null | head -1)"

        # Input validation: dedup.bam must exist and pass quickcheck
        bam_err = _bam_quickcheck(executor, f"{adir}/{sample}.dedup.bam")
        if bam_err:
            return f"[ERROR] {bam_err}. Run step2 first."

        cmds = f"""
set -euo pipefail
cd {q(safe_dir)}
# Resilience: clear partial outputs from interrupted prior run
rm -f {q(adir)}/{s}.bed {s}_peaks.narrowPeak.filt
rm -rf macs2_{s}
bedtools bamtobed -i {q(adir)}/{s}.dedup.bam > {q(adir)}/{s}.bed
macs2 callpeak -t {q(adir)}/{s}.bed -n {s} -g {q(species)} \
  -B -q 0.05 --outdir macs2_{s} --nomodel --shift -100 --extsize 200
BL="{bl_expr}"
if [ -z "$BL" ] || [ ! -s "$BL" ]; then
  echo "no blacklist bed found in {rd}" >&2; exit 1
fi
bedtools intersect -v -a macs2_{s}/{s}_peaks.narrowPeak \
  -b "$BL" > {s}_peaks.narrowPeak.filt
"""
        out = executor.run(cmds, cwd=safe_dir)
        if not executor.is_success(out):
            return out
        # Output sanity: peaks file must have at least 1 record
        peaks_filt = f"{safe_dir}/{sample}_peaks.narrowPeak.filt"
        if not _file_nonempty(peaks_filt, min_bytes=1):
            return out + f"\n[ERROR] {sample}_peaks.narrowPeak.filt is empty. Not marking step3 done."
        executor.mark_done(safe_dir, sample, "step3")
        return out

    @mcp.tool()
    def step4_peak_annotation(
        project_dir: str,
        sample: str,
        ref_dir: str = "",
        gtf_file: str = "",
        homer_bin: str = "/BIO1/MRAG/Agent/bin/tools/homer/bin",
    ) -> str:
        """Step 4: Annotate peaks with HOMER annotatePeaks.pl.

        Args:
            project_dir: Project directory.
            sample:      Sample prefix.
            ref_dir:     Reference directory.
            gtf_file:    GTF file path.
            homer_bin:   Path to HOMER bin directory.
        """
        safe_dir = str(executor._resolve_project_dir(project_dir))
        if executor.is_done(safe_dir, sample, "step4"):
            return f"[SKIP] step4 already completed for {sample}"

        rd = ref_dir or "/BIO1/MRAG/Agent/Reference/ATAC_hg38/"
        gtf = gtf_file or f"{rd}hg38.refGene.gtf"
        s = q(sample)

        # Input validation: peaks.narrowPeak.filt must be non-empty
        peaks_filt = f"{safe_dir}/{sample}_peaks.narrowPeak.filt"
        if not _file_nonempty(peaks_filt, min_bytes=1):
            return f"[ERROR] {peaks_filt} missing or empty. Run step3 first."

        cmds = f"""
set -euo pipefail
cd {q(safe_dir)}
# Resilience: clear partial outputs from interrupted prior run
rm -f {s}.input.bed {s}_annotations.txt
# HOMER's annotatePeaks.pl spawns helper scripts (cleanUpPeakFile.pl, parseGTF.pl,
# annotateRelativePosition.pl, assignGenomeAnnotation) by name; they must be on PATH.
export PATH="{q(homer_bin)}:$PATH"
awk -F "\\t" '{{print $4 "\\t" $1 "\\t" $2 "\\t" $3 "\\t" "+"}}' \
  {s}_peaks.narrowPeak.filt > {s}.input.bed
perl {q(homer_bin)}/annotatePeaks.pl {s}.input.bed {q(rd)} -gtf {q(gtf)} \
  > {s}_annotations.txt
"""
        out = executor.run(cmds, cwd=safe_dir)
        if not executor.is_success(out):
            return out
        # Output sanity: must contain header + at least one annotation row.
        # A 376-byte header-only file is what we get when HOMER helper scripts
        # are missing from PATH — fail loudly instead of marking done.
        ann_file = Path(f"{safe_dir}/{sample}_annotations.txt")
        if not ann_file.exists():
            return out + f"\n[ERROR] {sample}_annotations.txt missing. Not marking step4 done."
        with ann_file.open() as fh:
            nlines = sum(1 for _ in fh)
        if nlines < 2:
            return out + (
                f"\n[ERROR] {sample}_annotations.txt has only the header line "
                f"(HOMER produced no annotations — likely a missing helper script or "
                f"misconfigured genome). Not marking step4 done."
            )
        executor.mark_done(safe_dir, sample, "step4")
        return out

    @mcp.tool()
    def step5_dastk(
        project_dir: str,
        treatment_files: list[str],
        control_files: list[str],
        ref_dir: str = "",
        genome: str = "hg38",
        num_threads: int = 0,
        dastk_path: str = "",
        motif_dir: str = "",
    ) -> str:
        """Step 5: DAStk differential motif analysis (treatment vs control).

        Args:
            project_dir:     Project directory.
            treatment_files: List of narrowPeak.filt files for treatment group.
            control_files:   List of narrowPeak.filt files for control group.
            ref_dir:         Reference directory with motifs subfolder.
            genome:          Genome build name.
            num_threads:     CPU threads. 0 = auto-detect.
            dastk_path:      Path to DAStk installation.
            motif_dir:       Directory of genome-wide motif-hit BEDs for THIS
                             assembly. Empty keeps the historical convention
                             (grch38_motifs/ for hg38, motifs/ otherwise),
                             which cannot name a mouse set.
        """
        safe_dir = str(executor._resolve_project_dir(project_dir))
        res = executor.detect_resources()
        t = num_threads or res["cores"]
        rd = ref_dir or "/BIO1/MRAG/Agent/Reference/ATAC_hg38/"
        dp = dastk_path or "/BIO1/MRAG/Agent/bin/tools/DAStk/DAStk"
        if motif_dir:
            motif_dir = motif_dir.rstrip("/") + "/"
        else:
            motif_dir = (
                f"{rd}grch38_motifs/"
                if "hg38" in genome.lower() or "grch38" in genome.lower()
                else f"{rd}motifs/"
            )
        if not Path(motif_dir).is_dir():
            return ("[ERROR] DAStk motif directory %s does not exist. DAStk "
                    "scores accessibility against genome-wide motif-hit BEDs, "
                    "one set per assembly (HOCOMOCO_v11_p1e-6_%s.tar.gz from "
                    "DAStk's setup.py). Without it there is nothing to score."
                    % (motif_dir, genome))

        treat_cat = " ".join(q(f) for f in treatment_files)
        ctrl_cat = " ".join(q(f) for f in control_files)

        cmds = f"""
set -euo pipefail
cd {q(safe_dir)}
mkdir -p DASTK/treatment DASTK/control
cat {treat_cat} > DASTK/treatment/treatment.narrowPeak.filt
cat {ctrl_cat} > DASTK/control/control.narrowPeak.filt
PYBIN=/BIO1/MRAG/venvs/paper_qa/bin/python
$PYBIN {q(dp)}/process_atac.py -t {t} \
  -e DASTK/treatment/treatment.narrowPeak.filt \
  -m {q(motif_dir)} -g {q(genome)} -o DASTK/treatment
$PYBIN {q(dp)}/process_atac.py -t {t} \
  -e DASTK/control/control.narrowPeak.filt \
  -m {q(motif_dir)} -g {q(genome)} -o DASTK/control
$PYBIN {q(_DASTK_SEEDED)} {q(dp)} \
  -1 DASTK/control/control.narrowPeak_md_scores.txt \
  -2 DASTK/treatment/treatment.narrowPeak_md_scores.txt \
  -m control -n treatment -b -o DASTK
"""
        out = executor.run(cmds, cwd=safe_dir)
        return out

    @mcp.tool()
    def run_full_pipeline(
        project_dir: str,
        ref_dir: str = "",
        genome: str = "hg38",
        species: str = "hs",
        num_threads: int = 0,
        gtf_file: str = "",
    ) -> str:
        """Run the complete ATAC-seq pipeline (steps 0-4) for all samples found.
        Skips already-completed steps via checkpoint. Step 5 (DAStk) requires
        manual group assignment and should be run separately.

        Args:
            project_dir: Project directory containing *_R1.fastq.gz files.
            genome:      Genome build name.
            species:     MACS2 genome size (hs/mm).
            num_threads: CPU threads. 0 = auto-detect.
            gtf_file:    Peak-annotation GTF for step4. Empty falls back to the
                         {ref_dir}hg38.refGene.gtf naming convention, which
                         only exists for the human reference -- so a non-human
                         run MUST pass this.
        """
        safe_dir = str(executor._resolve_project_dir(project_dir))
        lock_path = Path(safe_dir) / ".atac_pipeline_run.lock"
        lock_fh = open(lock_path, "w")
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_fh.close()
            return "[ERROR] Another pipeline run is already active on this project directory. Aborting to prevent concurrent file writes."

        try:
            rd = ref_dir or "/BIO1/MRAG/Agent/Reference/ATAC_hg38/"
            samples = _list_samples(safe_dir)
            if not samples:
                return f"No FASTQ files (*.fastq.gz / *.fq.gz) found in {safe_dir}"

            # Run up to 4 samples in parallel; with >4 samples they queue in batches.
            # Threads per sample = total / max_parallel so total CPU is bounded.
            res = executor.detect_resources()
            total_threads = num_threads or res["cores"]
            max_parallel = min(4, len(samples))
            threads_per_sample = max(4, total_threads // max_parallel)

            def _process_one(s: str) -> str:
                lines = [f"=== {s} ==="]
                lines.append(step0_preprocessing(safe_dir, s, threads_per_sample))
                lines.append(step1_alignment(safe_dir, s, rd, genome, threads_per_sample))
                lines.append(step2_filtering(safe_dir, s, rd, genome))
                lines.append(step3_peak_calling(safe_dir, s, rd, species))
                lines.append(step4_peak_annotation(safe_dir, s, rd, gtf_file))
                return "\n".join(lines)

            results: dict[str, str] = {}
            with ThreadPoolExecutor(max_workers=max_parallel) as pool:
                fut_to_sample = {pool.submit(_process_one, s): s for s in samples}
                for fut in as_completed(fut_to_sample):
                    s = fut_to_sample[fut]
                    try:
                        results[s] = fut.result()
                    except Exception as e:
                        results[s] = f"=== {s} ===\n[ERROR] {e}"

            # Preserve original sample order in output
            return f"[parallel: {max_parallel} workers × {threads_per_sample} threads each]\n" + \
                   "\n".join(results[s] for s in samples)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
            lock_fh.close()
