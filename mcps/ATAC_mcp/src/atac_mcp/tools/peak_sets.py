"""atac-mcp: condition-specific peak sets (the 3-way split behind report Sections 6, 8, 9 and the SE layer).

MCP step 2a (2026-10-12): the orchestrator's nodes/peakheatmap.build_specific_beds,
moved here unchanged as the MCP tool split_condition_peaks; build_specific_beds now calls it through the
orchestrator's MCP transport.  Byte-for-byte the same bedtools chain as Reference/peakheatmap/run_peakheatmap.sh
Step 1 (1.input.py): within-arm replicate intersect -> A-specific / Shared / B-specific BEDs."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Union


def split_condition_peaks(narrowpeak_a: list[str], narrowpeak_b: list[str], label_a: str, label_b: str,
                          output_dir: str, timeout_s: Union[int, float] = 900) -> dict[str, str]:
    """Within-arm replicate intersect -> A-specific / Shared / B-specific peak BEDs.

    Args:
        narrowpeak_a, narrowpeak_b: per-replicate narrowPeak files of arm A (treatment) and arm B (control);
            the first two of each are used.
        label_a, label_b: arm labels (file names <label>_specific.bed, <label>_group.bed, <label>_r1/_r2.bed).
        output_dir: the peak-heatmap output dir (the later heatmap step regenerates the same files).
        timeout_s: for the bedtools chain.

    Returns {label_a: path, label_b: path, "shared": path}; {} when either arm has fewer than two replicates or
    a result is empty.  Raises RuntimeError when bedtools fails."""
    if not (narrowpeak_a and narrowpeak_b
            and len(narrowpeak_a) >= 2 and len(narrowpeak_b) >= 2):
        return {}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    a1, a2 = narrowpeak_a[0], narrowpeak_a[1]
    b1, b2 = narrowpeak_b[0], narrowpeak_b[1]
    script = f"""set -e
cd {out}
np2bed(){{ awk 'BEGIN{{OFS="\t"}}{{print $1,$2,$3}}' "$1" > "$2"; }}
np2bed {a1} "{label_a}_r1.bed"; np2bed {a2} "{label_a}_r2.bed"
np2bed {b1} "{label_b}_r1.bed"; np2bed {b2} "{label_b}_r2.bed"
bedtools intersect -a "{label_a}_r1.bed" -b "{label_a}_r2.bed" > "{label_a}_group.bed"
bedtools intersect -a "{label_b}_r1.bed" -b "{label_b}_r2.bed" > "{label_b}_group.bed"
bedtools intersect -a "{label_a}_group.bed" -b "{label_b}_group.bed" -v > "{label_a}_specific.bed"
bedtools intersect -a "{label_b}_group.bed" -b "{label_a}_group.bed" -v > "{label_b}_specific.bed"
bedtools intersect -a "{label_a}_group.bed" -b "{label_b}_group.bed" > shared.bed
"""
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       timeout=timeout_s)
    if r.returncode != 0:
        raise RuntimeError("specific-BED split failed: " + r.stderr[-500:])
    res = {label_a: out / f"{label_a}_specific.bed",
           label_b: out / f"{label_b}_specific.bed",
           "shared": out / "shared.bed"}
    if not all(p.is_file() and p.stat().st_size > 0 for p in res.values()):
        return {}
    return {k: str(v) for k, v in res.items()}


def register_peak_set_tools(mcp):
    mcp.tool()(split_condition_peaks)
