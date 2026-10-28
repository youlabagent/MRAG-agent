"""atac-mcp: GIGGLE TF ChIP-seq overlap of the condition-specific ATAC peak sets (report Section 9) -- LOCAL, no
Cistrome web.

MCP step 2a (2026-10-12): the body of the orchestrator's GiggleNode (nodes/giggle.py),
moved here unchanged as the MCP tool giggle_tf_overlap; the node calls it through the orchestrator's MCP transport.

Runs giggle search of the condition-open peak sets ({label}_specific.bed == analyst {COND}_open.bed) against the
local Cistrome TF ChIP-seq index of the run's assembly (Reference/cistrome/run_giggle.sh, 7th argument: hg38 ->
giggle_index_hg38_tf, 10,005 datasets; mm10 -> giggle_index_mm10_tf, 7,624), then follows 3.plot.R: keep the TOP-N
datasets by GIGGLE score, take the per-Factor mean, plot the top-10 (treatment-open = orange, control-open = blue).
The TOP-N cut is essential -- without it a Factor's mean is diluted by its many irrelevant-cell-type datasets."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Union

WRAPPER = "/BIO1/MRAG/Agent/Reference/cistrome/run_giggle.sh"


def giggle_tf_overlap(treat_bed: str, control_bed: str, output_dir: str, treat_label: str = "treatment",
                      control_label: str = "control", top_n: Union[int, str] = 200, genome: str = "hg38",
                      timeout_s: Union[int, float] = 1200) -> dict[str, str]:
    """GIGGLE search of the treatment- and control-open peak sets against the local Cistrome TF index + the
    two top-10 factor plots (the GiggleNode stage).

    Args:
        treat_bed, control_bed: condition-specific (open) peak BEDs (peakheatmap/<label>_specific.bed).
        output_dir: where run_giggle.sh writes its result tables and treatment_giggle.png / control_giggle.png.
        treat_label, control_label: the condition labels shown on the plots.
        top_n: datasets kept by GIGGLE score before the per-factor mean (default 200).
        genome: "hg38" or "mm10" -- selects the index and the GSM -> factor map.
        timeout_s: for the whole wrapper run.

    Returns {giggle_up_png, giggle_down_png}; raises RuntimeError when the wrapper fails or a plot is missing."""
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    treat = treat_bed; ctrl = control_bed
    tlab = treat_label
    clab = control_label
    top_n = str(top_n)
    genome = str(genome or "hg38")
    up_png = out_dir / "treatment_giggle.png"
    dn_png = out_dir / "control_giggle.png"
    r = subprocess.run(
        ["bash", WRAPPER, str(treat), str(ctrl), str(out_dir), tlab, clab, top_n,
         genome],
        capture_output=True, text=True, timeout=timeout_s)
    if r.returncode != 0 or not (up_png.is_file() and dn_png.is_file()):
        raise RuntimeError(f"GIGGLE failed (rc={r.returncode}): {(r.stderr or '')[-800:]}")
    return {"giggle_up_png": str(up_png), "giggle_down_png": str(dn_png)}


def register_giggle_tools(mcp):
    mcp.tool()(giggle_tf_overlap)
