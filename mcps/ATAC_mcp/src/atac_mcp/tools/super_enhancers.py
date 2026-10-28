"""atac-mcp: super-enhancer calling with ROSE (report Section 10, the SE layer of the SE -> TF -> Target TRN).

MCP step 2a (2026-10-12): the bodies of the orchestrator's SERoseNode
(nodes/se_rose.py) and SERoseBigwigNode (nodes/se_rose_bigwig.py), moved here unchanged as two MCP tools; the
nodes now call them through the orchestrator's MCP transport (MRAG_TOOL_TRANSPORT=direct: a direct call).

  call_super_enhancers          reads from the treatment-arm ATAC BAMs (reference 1.rose.py steps 1-3):
                                  1. samtools merge/sort/index the BAMs
                                  2. treatment-specific peak BED -> ROSE GFF
                                  3. ROSE_main.py (stitch 12.5 kb, no TSS exclusion) + ROSE_geneMapper.py
                                and the top-N super-enhancers (rank by ROSE signal) as se_top<N>.tsv / .homerpeak.
                                Cache: an existing non-empty <output_dir>/rose/<label>_specific_SuperEnhancers_
                                ENHANCER_TO_GENE.txt skips the ROSE run.  BAM index required -- a missing .bai
                                silently yields 0 SEs.
  call_super_enhancers_bigwig   the same ROSE procedure for inputs WITHOUT read alignments: the per-region signal
                                is the area under the treatment-arm bigWig tracks inside each stitched region (sum of
                                per-base values, mean over the replicate bigWigs); ROSE_callSuper.R
                                (no control) and ROSE_geneMapper.py are ROSE code; ROSE_stitchTable.py is an MRAG addition to ROSE.
Both return the outputs dict the nodes record (paths, counts; the merged BAM as a value)."""
from __future__ import annotations

import csv
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

ROSE_DIR = "/BIO1/MRAG/Agent/bin/tools/ROSE"
ROSE_PY2 = "/home/ai/miniconda3/envs/rose/bin/python2"
SAMTOOLS = "/usr/local/bin/samtools"


def _run(cmd, cwd, timeout, env=None, log=None):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, capture_output=True,
                       text=True, timeout=timeout, env=env)
    if log:
        with open(log, "a") as fh:
            fh.write(f"\n$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}\n")
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd if isinstance(cmd, str) else ' '.join(cmd)}\n{r.stderr[-800:]}")
    return r


def _run_bw(cmd, cwd, timeout, log, env=None):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, capture_output=True, text=True,
                       timeout=timeout, env=env)
    with open(log, "a") as fh:
        fh.write(f"\n$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}\n")
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {(cmd if isinstance(cmd, str) else ' '.join(cmd))[:300]}\n{r.stderr[-600:]}")
    return r


def _data_rows(p: Path) -> int:
    if not p.is_file():
        return 0
    with open(p) as fh:
        return max(sum(1 for l in fh if l.strip() and not l.startswith("#")) - 1, 0)


def call_super_enhancers(bam_list: list[str], peaks_bed: str, label: str, output_dir: str,
                         genome: str = "hg38", stitch: int = 12500, tss: int = 0, threads: int = 16,
                         top_n: int = 10, timeout_s: int = 10800, rose_dir: str = ROSE_DIR,
                         python2: str = ROSE_PY2) -> dict[str, Any]:
    """ROSE super-enhancers from read alignments (the SERoseNode stage).

    Args:
        bam_list: treatment-arm dedup BAMs (>= 1); merged, sorted and indexed into <output_dir>.
        peaks_bed: treatment-specific peak BED (3+ columns), the ROSE constituent enhancers.
        label: treatment condition label, used as the file prefix (sanitised to [A-Za-z0-9._-]).
        output_dir: the node's se_rose/ directory (rose/ inside holds ROSE's own outputs, the cache).
        genome: ROSE annotation <genome>_refseq.ucsc under rose_dir/annotation.
        stitch: ROSE stitching distance (bp).  tss: TSS exclusion zone (0 = off).
        threads: samtools / ROSE threads.  top_n: super-enhancers kept for the SE layer.
        timeout_s: per external command.  rose_dir, python2: the patched ROSE copy and its python2.

    Returns {merged_bam:{path,size}, rose_gff:{path,n_peaks}, se_table, all_table, enhancer_to_gene,
    se_top_tsv, se_homerpeak, n_se, n_all, n_peaks, top_n[, rose_plot_png]}.  Raises on a failed command, a
    missing input, or 0 super-enhancers."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    log = out / "se_rose.log"
    bams = [str(b) for b in bam_list]
    bed = str(peaks_bed)
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label)).strip("_") or "treat"   # ROSE shells out
    timeout = timeout_s
    py2 = python2
    if not bams or not Path(bed).is_file():
        raise FileNotFoundError("SERoseNode needs >=1 BAM and the treatment-specific BED")
    for b in bams:
        if not Path(b).is_file():
            raise FileNotFoundError(f"BAM not found: {b}")
    annot = Path(rose_dir) / "annotation" / f"{genome}_refseq.ucsc"
    if not annot.is_file():
        raise FileNotFoundError(f"ROSE annotation missing: {annot}")

    # 1. merged / sorted / indexed treatment BAM (cached by mtime)
    merged = out / f"{label}_merged.sorted.bam"
    bai = Path(str(merged) + ".bai")
    newest_in = max(Path(b).stat().st_mtime for b in bams)
    if not (merged.is_file() and bai.is_file() and merged.stat().st_mtime >= newest_in
            and merged.stat().st_size > 0):
        tmp = out / f"{label}_merged.bam"
        if len(bams) == 1:
            _run([SAMTOOLS, "sort", "-@", str(threads), "-o", str(merged), bams[0]], out, timeout, log=log)
        else:
            _run([SAMTOOLS, "merge", "-f", "-@", str(threads), "-o", str(tmp)] + bams, out, timeout, log=log)
            _run([SAMTOOLS, "sort", "-@", str(threads), "-o", str(merged), str(tmp)], out, timeout, log=log)
            tmp.unlink(missing_ok=True)
        _run([SAMTOOLS, "index", "-@", str(threads), str(merged)], out, timeout, log=log)
    if not bai.is_file():
        raise RuntimeError("BAM index (.bai) missing after samtools index — ROSE would report 0 SEs")

    # 2. BED -> ROSE GFF (chrom, id, ., start+1, end, ., ., ., ID=id)  [ 1.rose.py step2]
    prefix = f"{label}_specific"
    gff = out / f"{prefix}.gff"
    n_peaks = 0
    with open(bed) as fi, open(gff, "w") as fo:
        for i, line in enumerate(fi, 1):
            f = line.rstrip("\n").split("\t")
            if len(f) < 3 or line.startswith(("track", "#")):
                continue
            pid = f"{label}_peak_{i}"
            fo.write("\t".join([f[0], pid, ".", str(int(f[1]) + 1), f[2], ".", ".", ".", f"ID={pid}"]) + "\n")
            n_peaks += 1
    if n_peaks == 0:
        raise ValueError(f"no peaks in {bed}")

    # 3. ROSE_main + geneMapper (cached)
    rose_out = out / "rose"; rose_out.mkdir(exist_ok=True)
    se_table = rose_out / f"{prefix}_SuperEnhancers.table.txt"
    all_table = rose_out / f"{prefix}_AllEnhancers.table.txt"
    e2g = rose_out / f"{prefix}_SuperEnhancers_ENHANCER_TO_GENE.txt"
    env = dict(os.environ)
    # ROSE_main.py spawns a bare `python .../ROSE_bamToGFF.py` and `samtools view`
    # (exit codes unchecked): put the configured python2 and samtools first on PATH.
    env["PATH"] = ":".join([str(Path(py2).parent), str(Path(SAMTOOLS).parent), "/usr/bin", "/usr/local/bin",
                            env.get("PATH", "")])
    if se_table.is_file() and e2g.is_file() and _data_rows(e2g) == 0:
        # a cached 0-SE result (header-only tables) must not be reused
        for stale in (se_table, e2g, all_table):
            stale.unlink(missing_ok=True)
    if not (se_table.is_file() and _data_rows(se_table) > 0 and e2g.is_file() and _data_rows(e2g) > 0):
        t0 = time.time()
        _run([py2, str(Path(rose_dir) / "ROSE_main.py"), "-g", str(annot), "-i", str(gff),
              "-r", str(merged), "-o", str(rose_out), "-s", str(stitch), "-t", str(tss),
              "-n", str(threads)], rose_out, timeout, env=env, log=log)
        if not se_table.is_file():
            raise RuntimeError(f"ROSE_main produced no {se_table.name}")
        _run([py2, str(Path(rose_dir) / "ROSE_geneMapper.py"), "-g", str(annot),
              "-i", str(se_table), "-o", str(rose_out)], rose_out, timeout, env=env, log=log)
        with open(log, "a") as fh:
            fh.write(f"ROSE wall time {time.time() - t0:.0f}s\n")

    # 4. top-N SE table + HOMER peak file
    rows = []
    with open(e2g) as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        sig_col = [c for c in rd.fieldnames if c not in
                   ("REGION_ID", "CHROM", "START", "STOP", "NUM_LOCI", "CONSTITUENT_SIZE",
                    "enhancerRank", "isSuper", "OVERLAP_GENES", "PROXIMAL_GENES", "CLOSEST_GENE")]
        for r in rd:
            rows.append(r)
    sig = sig_col[0] if sig_col else None
    rows.sort(key=lambda r: int(r["enhancerRank"]))
    n_se = len(rows)
    n_all = _data_rows(all_table)
    if n_se == 0:
        raise RuntimeError("ROSE returned 0 super-enhancers: check the merged BAM index (.bai) "
                           "and that the peak BED chromosome names match the BAM")
    top = rows[:top_n]
    se_top = out / f"se_top{top_n}.tsv"; hp = out / f"se_top{top_n}.homerpeak"
    with open(se_top, "w") as fo, open(hp, "w") as fh:
        fo.write("SE\trank\tchrom\tstart\tend\tregion\tsignal\tclosest_gene\toverlap_genes\tproximal_genes\tregion_id\n")
        for r in top:
            sid = f'SE{r["enhancerRank"]}'
            fo.write("\t".join([sid, r["enhancerRank"], r["CHROM"], r["START"], r["STOP"],
                                f'{r["CHROM"]}:{r["START"]}-{r["STOP"]}', r.get(sig, "") if sig else "",
                                r.get("CLOSEST_GENE", ""), r.get("OVERLAP_GENES", ""), r.get("PROXIMAL_GENES", ""),
                                r["REGION_ID"]]) + "\n")
            fh.write("\t".join([sid, r["CHROM"], r["START"], r["STOP"], "+"]) + "\n")
    png = rose_out / f"{prefix}_Plot_points.png"
    outputs = {
        # large intermediates are recorded as values (not paths) so the manifest
        # does not sha256 a multi-GB BAM on every run
        "merged_bam": {"path": str(merged), "size": merged.stat().st_size},
        "rose_gff": {"path": str(gff), "n_peaks": n_peaks},
        "se_table": str(se_table), "all_table": str(all_table), "enhancer_to_gene": str(e2g),
        "se_top_tsv": str(se_top), "se_homerpeak": str(hp),
        "n_se": n_se, "n_all": n_all, "n_peaks": n_peaks, "top_n": min(top_n, n_se),
    }
    if png.is_file():
        outputs["rose_plot_png"] = str(png)
    return outputs


def call_super_enhancers_bigwig(bigwig_list: list[str], peaks_bed: str, label: str, output_dir: str,
                                genome: str = "hg38", stitch: int = 12500, tss: int = 0, top_n: int = 10,
                                timeout_s: int = 7200, rose_dir: str = ROSE_DIR,
                                python2: str = ROSE_PY2) -> dict[str, Any]:
    """ROSE super-enhancers from bigWig signal, for inputs without read alignments (the SERoseBigwigNode stage).

    Args:
        bigwig_list: treatment-arm ATAC bigWigs (missing files are skipped; >= 1 must exist).
        peaks_bed: treatment peak set (constituent enhancers).
        label: treatment condition label (file prefix, sanitised).  output_dir: the node's se_rose/ directory.
        genome, stitch, tss, top_n, timeout_s, rose_dir, python2: as call_super_enhancers.

    Returns the call_super_enhancers outputs (no merged BAM) + n_bigwigs and signal_source."""
    import pyBigWig
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    log = out / "se_rose.log"
    bws = [str(b) for b in bigwig_list if Path(b).is_file()]
    bed = str(peaks_bed)
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label)).strip("_") or "treat"
    timeout = timeout_s
    py2 = python2
    if not bws:
        raise FileNotFoundError("SERoseBigwigNode needs >=1 bigWig")
    if not Path(bed).is_file():
        raise FileNotFoundError(f"peak BED not found: {bed}")
    annot = Path(rose_dir) / "annotation" / f"{genome}_refseq.ucsc"
    if not annot.is_file():
        raise FileNotFoundError(f"ROSE annotation missing: {annot}")

    # 1. BED -> ROSE GFF
    prefix = f"{label}_bw"
    gff = out / f"{prefix}.gff"; n_peaks = 0
    with open(bed) as fi, open(gff, "w") as fo:
        for i, line in enumerate(fi, 1):
            f = line.rstrip("\n").split("\t")
            if len(f) < 3 or line.startswith(("track", "#")):
                continue
            pid = f"{label}_peak_{i}"
            fo.write("\t".join([f[0], pid, ".", str(int(f[1]) + 1), f[2], ".", ".", ".", f"ID={pid}"]) + "\n")
            n_peaks += 1
    if n_peaks == 0:
        raise ValueError(f"no peaks in {bed}")

    rose_out = out / "rose"; rose_out.mkdir(exist_ok=True)
    se_table = rose_out / f"{prefix}_SuperEnhancers.table.txt"
    all_table = rose_out / f"{prefix}_AllEnhancers.table.txt"
    e2g = rose_out / f"{prefix}_SuperEnhancers_ENHANCER_TO_GENE.txt"
    env = dict(os.environ); env["PATH"] = ":".join([str(Path(py2).parent), "/usr/bin", "/usr/local/bin", env.get("PATH", "")])
    if se_table.is_file() and e2g.is_file() and _data_rows(e2g) == 0:
        for stale in (se_table, e2g, all_table):
            stale.unlink(missing_ok=True)
    if not (_data_rows(se_table) > 0 and _data_rows(e2g) > 0):
        t0 = time.time()
        # 2. ROSE stitching (ROSE_main.regionStitching)
        stitched = rose_out / f"{prefix}_{stitch // 1000}KB_STITCHED_table.txt"
        _run_bw([py2, str(Path(rose_dir) / "ROSE_stitchTable.py"), "-i", str(gff), "-g", str(annot),
                 "-s", str(stitch), "-t", str(tss), "-o", str(stitched),
                 "-f", str(rose_out / f"{prefix}_{stitch // 1000}KB_STITCHED.gff")], rose_out, timeout, log, env)
        # 3. bigWig area under the curve per stitched region, mean over replicates
        rows = list(csv.reader(open(stitched), delimiter="\t"))
        hdr, body = rows[0], rows[1:]
        sums = [0.0] * len(body)
        for bw in bws:
            h = pyBigWig.open(bw); chroms = h.chroms()
            for k, r in enumerate(body):
                c, s, e = r[1], int(r[2]) - 1, int(r[3])
                if c not in chroms:
                    continue
                e = min(e, chroms[c]); s = max(0, min(s, e))
                if e <= s:
                    continue
                v = h.stats(c, s, e, type="sum", exact=True)[0]
                sums[k] += float(v) if v is not None else 0.0
            h.close()
        region_map = rose_out / f"{prefix}_{stitch // 1000}KB_STITCHED_ENHANCER_REGION_MAP.txt"
        with open(region_map, "w") as fo:
            fo.write("\t".join(hdr + [f"{label}_bigwig_signal"]) + "\n")
            for r, v in zip(body, sums):
                fo.write("\t".join(r + [f"{v / len(bws):.4f}"]) + "\n")
        # 4. ROSE super-enhancer cutoff (unchanged R code), no control
        _run_bw(f"R --no-save {rose_out}/ {region_map} {prefix} NONE < {rose_dir}/ROSE_callSuper.R",
                rose_out, timeout, log, env)
        if not se_table.is_file():
            raise RuntimeError(f"ROSE_callSuper.R produced no {se_table.name}")
        # 5. ROSE gene mapper
        _run_bw([py2, str(Path(rose_dir) / "ROSE_geneMapper.py"), "-g", str(annot), "-i", str(se_table),
                 "-o", str(rose_out)], rose_out, timeout, log, env)
        with open(log, "a") as fh:
            fh.write(f"bigWig ROSE wall time {time.time() - t0:.0f}s ({len(bws)} bigWigs, {len(body)} stitched regions)\n")

    # 6. top-N table + HOMER peak file (same format as SERoseNode)
    rows = list(csv.DictReader(open(e2g), delimiter="\t"))
    std = {"REGION_ID", "CHROM", "START", "STOP", "NUM_LOCI", "CONSTITUENT_SIZE", "enhancerRank", "isSuper",
           "OVERLAP_GENES", "PROXIMAL_GENES", "CLOSEST_GENE"}
    sig = next((c for c in (rows[0].keys() if rows else []) if c not in std), None)
    rows.sort(key=lambda r: int(r["enhancerRank"]))
    n_se = len(rows); n_all = _data_rows(all_table)
    if n_se == 0:
        raise RuntimeError("ROSE (bigWig signal) returned 0 super-enhancers: check the bigWig chromosome names and the peak BED")
    top = rows[:top_n]
    se_top = out / f"se_top{top_n}.tsv"; hp = out / f"se_top{top_n}.homerpeak"
    with open(se_top, "w") as fo, open(hp, "w") as fh:
        fo.write("SE\trank\tchrom\tstart\tend\tregion\tsignal\tclosest_gene\toverlap_genes\tproximal_genes\tregion_id\n")
        for r in top:
            sid = f'SE{r["enhancerRank"]}'
            fo.write("\t".join([sid, r["enhancerRank"], r["CHROM"], r["START"], r["STOP"],
                                f'{r["CHROM"]}:{r["START"]}-{r["STOP"]}', r.get(sig, "") if sig else "",
                                r.get("CLOSEST_GENE", ""), r.get("OVERLAP_GENES", ""), r.get("PROXIMAL_GENES", ""),
                                r["REGION_ID"]]) + "\n")
            fh.write("\t".join([sid, r["CHROM"], r["START"], r["STOP"], "+"]) + "\n")
    png = rose_out / f"{prefix}_Plot_points.png"
    outputs = {"rose_gff": {"path": str(gff), "n_peaks": n_peaks},
               "se_table": str(se_table), "all_table": str(all_table), "enhancer_to_gene": str(e2g),
               "se_top_tsv": str(se_top), "se_homerpeak": str(hp),
               "n_se": n_se, "n_all": n_all, "n_peaks": n_peaks, "top_n": min(top_n, n_se),
               "n_bigwigs": len(bws),
               "signal_source": "bigWig (mean area under the curve per stitched region)"}
    if png.is_file():
        outputs["rose_plot_png"] = str(png)
    return outputs


def register_super_enhancer_tools(mcp):
    mcp.tool()(call_super_enhancers)
    mcp.tool()(call_super_enhancers_bigwig)
