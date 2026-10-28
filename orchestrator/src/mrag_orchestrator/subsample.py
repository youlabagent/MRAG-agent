"""FASTQ subsampling helper for fast pipeline smoke tests.

No external dep — uses gzip + simple line read. Takes the FIRST n_reads
from each input (and matching pair). For random sampling we'd want seqtk;
deterministic head is enough for smoke.
"""
from __future__ import annotations
import gzip
import re
import shutil
from pathlib import Path
from typing import Optional


_PAIR_TOKENS = [
    (re.compile(r"_R1(_|\.)"), "_R2"),
    (re.compile(r"_1(_|\.)"),  "_2"),
    (re.compile(r"\.R1(_|\.)"),".R2"),
    (re.compile(r"_F(_|\.)"),  "_R"),
]


def _find_pair(r1: Path) -> Optional[Path]:
    name = r1.name
    for pat, r2_tok in _PAIR_TOKENS:
        m = pat.search(name)
        if m:
            start, end = m.span()
            sep = m.group(1)
            r2_name = name[:start] + r2_tok + sep + name[end:]
            cand = r1.parent / r2_name
            if cand.exists():
                return cand
    return None


def _head_fastq(src: Path, dst: Path, n_reads: int) -> int:
    """Write first n_reads records (4 lines/read) from src to dst (both gzipped)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    n_lines = n_reads * 4
    opener = gzip.open if src.suffix == ".gz" else open
    with opener(src, "rt") as fh, gzip.open(dst, "wt") as out:
        i = 0
        for line in fh:
            if i >= n_lines: break
            out.write(line); i += 1
    return i // 4


def subsample_fastqs(
    input_fastqs: list[str | Path],
    output_dir: Path,
    n_reads: int = 100_000,
    preserve_structure: bool = False,
) -> dict:
    """Subsample a list of FASTQ files (first n_reads each, paired-aware).

    Args:
      input_fastqs: list of R1 (and possibly R2) FASTQ paths.
      output_dir:   where to write subsampled FASTQs (created if missing).
      n_reads:      reads per file (paired files get the same count).
      preserve_structure: if True, mirror parent dir names under output_dir.

    Returns dict with:
      out_dir, n_inputs, n_pairs, written: [{src, dst, n_reads_written}]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[Path] = set()
    pairs: list[tuple[Path, Optional[Path]]] = []
    for f in sorted(Path(x) for x in input_fastqs):
        if f in seen: continue
        r2 = _find_pair(f)
        pairs.append((f, r2))
        seen.add(f)
        if r2: seen.add(r2)

    written = []
    for r1, r2 in pairs:
        dst1 = output_dir / r1.name
        n1 = _head_fastq(r1, dst1, n_reads)
        written.append({"src": str(r1), "dst": str(dst1), "n_reads_written": n1})
        if r2:
            dst2 = output_dir / r2.name
            n2 = _head_fastq(r2, dst2, n_reads)
            written.append({"src": str(r2), "dst": str(dst2), "n_reads_written": n2})
    return {
        "out_dir":  str(output_dir),
        "n_inputs": len(seen),
        "n_pairs":  len(pairs),
        "written":  written,
    }
