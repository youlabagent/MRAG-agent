"""prepare_diffbind_sheet + run_diffbind + status/top_peaks/check_env."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from mcp.server.fastmcp import FastMCP


# ----- Constants ---------------------------------------------------------
_DIFFBIND_SHEET_COLS = [
    "SampleID",
    "Tissue",
    "Factor",
    "Condition",
    "Treatment",
    "Replicate",
    "bamReads",
    "ControlID",
    "banControl",
    "Peaks",
    "PeakCaller",
]

# Bundled R script location (ships in the wheel as package data).
_R_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_diffbind.R"

# Where to find Rscript. Override with env DIFFBIND_RSCRIPT if needed.
_RSCRIPT_BIN = os.environ.get("DIFFBIND_RSCRIPT") or shutil.which("Rscript") or "/usr/bin/Rscript"


# ----- Helpers -----------------------------------------------------------
def _coerce_sample(row: Dict[str, Any], default_factor: str = "ATAC",
                   default_tissue: str = "human",
                   default_peak_caller: str = "bed") -> Dict[str, Any]:
    """Fill in DiffBind sheet defaults for one sample dict.

    Required keys: SampleID, Condition, Replicate, bamReads, Peaks.
    Optional: Tissue, Factor, Treatment, ControlID, banControl, PeakCaller.
    """
    required = ("SampleID", "Condition", "Replicate", "bamReads", "Peaks")
    missing = [k for k in required if not row.get(k)]
    if missing:
        raise ValueError(f"sample missing required keys {missing}: {row}")
    return {
        "SampleID":   row["SampleID"],
        "Tissue":     row.get("Tissue", default_tissue),
        "Factor":     row.get("Factor", default_factor),
        "Condition":  row["Condition"],
        "Treatment":  row.get("Treatment", row["Condition"]),
        "Replicate":  int(row["Replicate"]),
        "bamReads":   row["bamReads"],
        "ControlID":  row.get("ControlID", ""),
        "banControl": row.get("banControl", ""),
        "Peaks":      row["Peaks"],
        "PeakCaller": row.get("PeakCaller", default_peak_caller),
    }


def _check_paths(samples: List[Dict[str, Any]]) -> List[str]:
    """Return list of missing-file errors (fail-fast before R subprocess)."""
    errs: List[str] = []
    for s in samples:
        for col in ("bamReads", "Peaks"):
            p = s.get(col)
            if p and not Path(p).is_file():
                errs.append(f"{s['SampleID']}: {col} not found at {p}")
    return errs


# ----- MCP tool registration ---------------------------------------------
def register_pipeline_tools(mcp: FastMCP):

    @mcp.tool()
    def prepare_diffbind_sheet(
        samples: List[Dict[str, Any]],
        output_path: str,
        default_factor: str = "ATAC",
        default_tissue: str = "human",
        default_peak_caller: str = "bed",
        check_files: bool = True,
    ) -> str:
        """Write a DiffBind-format sample sheet CSV.

        Each sample dict must contain at least:
            SampleID:    unique label
            Condition:   group label used for contrasts (e.g. "OE","CTRL")
            Replicate:   integer
            bamReads:    absolute path to dedup BAM
            Peaks:       absolute path to MACS2 narrowPeak (or bed)

        Optional keys: Tissue, Factor, Treatment, ControlID, banControl,
        PeakCaller. Defaults are applied for any missing optional key.

        Args:
            samples:             List of sample dicts as above.
            output_path:         Where to write the CSV.
            default_factor:      Factor column default (default "ATAC").
            default_tissue:      Tissue column default (default "human").
            default_peak_caller: PeakCaller column default (default "bed").
            check_files:         If True (default), fail with a list of
                                 missing BAM/Peaks before writing the sheet.

        Returns JSON status with the written sheet path and any file-check
        errors. Schema matches the reference analysis's DiffBind_input.csv exactly.
        """
        try:
            rows = [
                _coerce_sample(s, default_factor, default_tissue, default_peak_caller)
                for s in samples
            ]
        except ValueError as e:
            return json.dumps({"status": "error", "message": str(e)})

        path_errs = _check_paths(rows) if check_files else []
        if path_errs:
            return json.dumps(
                {"status": "error", "message": "missing files", "errors": path_errs},
                ensure_ascii=False,
                indent=2,
            )

        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows, columns=_DIFFBIND_SHEET_COLS)
        df.to_csv(out, index=False)

        # Per-condition replicate count for sanity (DiffBind needs >=2 per group).
        per_cond = df.groupby("Condition").size().to_dict()
        return json.dumps(
            {
                "status":           "ok",
                "output":           str(out),
                "n_samples":        int(len(df)),
                "samples_per_cond": per_cond,
                "columns":          _DIFFBIND_SHEET_COLS,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    @mcp.tool()
    def run_diffbind(
        sheet_path: str,
        output_csv: str,
        category: str = "Condition",
        reference: Optional[str] = None,
        min_members: int = 2,
        threshold: float = 1.0,
        threads: int = 4,
        timeout_s: int = 3600,
        dry_run: bool = False,
    ) -> str:
        """Run the full DiffBind R pipeline on the given sample sheet.

        This invokes the bundled R script
        ``diffbind_mcp/scripts/run_diffbind.R`` via Rscript subprocess.

        Args:
            sheet_path:   CSV from prepare_diffbind_sheet (or curated-style).
            output_csv:   Where the per-peak report is written. Schema:
                          seqnames, start, end, width, strand, Conc,
                          Conc_<groupA>, Conc_<groupB>, Fold, p.value, FDR.
            category:     Contrast category. One of "Condition" (default),
                          "Treatment", "Factor", "Tissue", "Replicate".
            reference:    Control/baseline group level. When set, Fold is
                          recomputed as Conc_treatment - Conc_reference so that
                          positive = up in the non-reference (treatment) group,
                          mirroring the RNA-seq DESeq2 convention. Requires a
                          two-group contrast. If None, DiffBind's default
                          (auto-chosen baseline) is kept.
            min_members:  Minimum members per group for a contrast (default 2).
            threshold:    dba.report th. 1.0 (default) returns all peaks.
            threads:      BiocParallel workers for dba.count (default 4).
            timeout_s:    Subprocess timeout. dba.count over many BAMs can
                          take 10-30 minutes; default 1 hour.
            dry_run:      If True, only validate inputs and return the
                          assembled R command -- no R invocation.

        Returns JSON status with elapsed time, command, output path, and the
        R-side summary (n_rows, n_samples, n_contrasts). On failure, stderr
        from R is included.
        """
        sheet = Path(sheet_path).expanduser()
        if not sheet.is_file():
            return json.dumps({"status": "error", "message": f"sheet not found: {sheet}"})
        out = Path(output_csv).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        summary_path = out.with_suffix(".status.json")

        if not _R_SCRIPT.is_file():
            return json.dumps({"status": "error",
                               "message": f"bundled R script missing: {_R_SCRIPT}"})

        cmd = [
            _RSCRIPT_BIN, str(_R_SCRIPT),
            "--sheet",        str(sheet),
            "--output",       str(out),
            "--category",     category,
            *(("--reference", str(reference)) if reference else ()),
            "--min-members",  str(int(min_members)),
            "--threshold",    str(float(threshold)),
            "--threads",      str(int(threads)),
            "--summary",      str(summary_path),
        ]

        if dry_run:
            return json.dumps(
                {
                    "status":  "dry_run",
                    "command": cmd,
                    "rscript": _RSCRIPT_BIN,
                    "script":  str(_R_SCRIPT),
                    "sheet":   str(sheet),
                    "output":  str(out),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            return json.dumps(
                {"status": "timeout",
                 "elapsed_s": round(time.time() - t0, 1),
                 "timeout_s": timeout_s,
                 "command": cmd,
                 "stderr_tail": (e.stderr or "")[-2000:] if hasattr(e, "stderr") and e.stderr else ""},
                ensure_ascii=False, indent=2, default=str,
            )
        elapsed = round(time.time() - t0, 2)

        if proc.returncode != 0:
            return json.dumps(
                {
                    "status":      "error",
                    "returncode":  proc.returncode,
                    "elapsed_s":   elapsed,
                    "command":     cmd,
                    "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
                    "stderr_tail": proc.stderr[-4000:] if proc.stderr else "",
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        # Try to read the JSON-ish summary file the R script wrote.
        r_summary: Dict[str, Any] = {}
        if summary_path.is_file():
            try:
                r_summary = json.loads(summary_path.read_text())
            except Exception:
                r_summary = {"raw": summary_path.read_text()}

        # Light verification of the output file
        try:
            n_rows = sum(1 for _ in open(out)) - 1
        except Exception:
            n_rows = -1

        return json.dumps(
            {
                "status":     "ok",
                "elapsed_s":  elapsed,
                "output":     str(out),
                "n_rows":     n_rows,
                "r_summary":  r_summary,
                "command":    cmd,
                "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    @mcp.tool()
    def diffbind_status(output_csv: str, head_n: int = 5) -> str:
        """Summarise an existing DiffBind output CSV (no R required).

        Args:
            output_csv: Path to a DiffBind_out.csv (or compatible) file.
            head_n:     Number of top rows (by smallest FDR) to preview.

        Returns JSON with row count, column list, FDR percentiles, fold
        statistics, and the top-N rows.
        """
        path = Path(output_csv).expanduser()
        if not path.is_file():
            return json.dumps({"status": "error", "message": f"not found: {path}"})
        df = pd.read_csv(path)

        # Common DiffBind schema: seqnames, start, end, ..., Fold, p.value, FDR
        result: Dict[str, Any] = {
            "status":  "ok",
            "input":   str(path),
            "n_rows":  int(len(df)),
            "columns": list(df.columns),
        }
        if "FDR" in df.columns:
            result["FDR"] = {
                "min":    float(df["FDR"].min()),
                "median": float(df["FDR"].median()),
                "p05":    float(df["FDR"].quantile(0.05)),
                "n_sig_0.05": int((df["FDR"] < 0.05).sum()),
                "n_sig_0.01": int((df["FDR"] < 0.01).sum()),
            }
        if "Fold" in df.columns:
            absfold = df["Fold"].abs()
            result["Fold"] = {
                "min":         float(df["Fold"].min()),
                "max":         float(df["Fold"].max()),
                "median_abs":  float(absfold.median()),
                "n_up_2x":     int((df["Fold"] >  1.0).sum()),
                "n_down_2x":   int((df["Fold"] < -1.0).sum()),
            }
        sort_key = "FDR" if "FDR" in df.columns else df.columns[0]
        result[f"top{head_n}_by_{sort_key}"] = (
            df.sort_values(by=sort_key).head(head_n).to_dict(orient="records")
        )
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    @mcp.tool()
    def diffbind_top_peaks(
        output_csv: str,
        top_n: int = 50,
        by: str = "FDR",
        min_abs_fold: Optional[float] = None,
        condition_direction: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """Extract the top-N peaks for downstream tools (HOMER / AlphaGenome).

        Args:
            output_csv:   DiffBind output CSV.
            top_n:        Number of peaks to return.
            by:           Sort key. "FDR" (default, ascending),
                          "abs_fold" (|Fold| descending), or "p.value".
            min_abs_fold: Pre-filter rows with |Fold| < this.
            condition_direction:
                          Optional sign filter: "up" keeps Fold > 0,
                          "down" keeps Fold < 0, None keeps both.
            save_path:    If given, write the resulting subset to TSV.

        Returns JSON with the selected rows (Gene-friendly columns when
        present + always seqnames/start/end/Fold/p.value/FDR).
        """
        path = Path(output_csv).expanduser()
        if not path.is_file():
            return json.dumps({"status": "error", "message": f"not found: {path}"})
        df = pd.read_csv(path)

        if min_abs_fold is not None and "Fold" in df.columns:
            df = df[df["Fold"].abs() >= min_abs_fold]
        if condition_direction == "up":
            df = df[df["Fold"] > 0]
        elif condition_direction == "down":
            df = df[df["Fold"] < 0]
        elif condition_direction is not None:
            return json.dumps({"status": "error",
                               "message": f"condition_direction must be 'up','down', or None, got '{condition_direction}'"})

        if by == "FDR":
            df_sorted = df.sort_values(by="FDR", ascending=True, kind="mergesort")
        elif by == "abs_fold":
            df_sorted = df.assign(_absf=df["Fold"].abs()) \
                          .sort_values(by="_absf", ascending=False, kind="mergesort") \
                          .drop(columns="_absf")
        elif by == "p.value":
            df_sorted = df.sort_values(by="p.value", ascending=True, kind="mergesort")
        else:
            return json.dumps({"status": "error",
                               "message": f"unknown sort key '{by}' (expected FDR / abs_fold / p.value)"})

        head = df_sorted.head(int(top_n))
        if save_path:
            sp = Path(save_path).expanduser()
            sp.parent.mkdir(parents=True, exist_ok=True)
            head.to_csv(sp, sep="\t", index=False)

        return json.dumps(
            {
                "status":     "ok",
                "input":      str(path),
                "n_returned": int(len(head)),
                "by":         by,
                "min_abs_fold": min_abs_fold,
                "condition_direction": condition_direction,
                "save_path":  str(save_path) if save_path else None,
                "peaks":      head.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    @mcp.tool()
    def diffbind_check_env() -> str:
        """Verify R + DiffBind environment without running an analysis."""
        if not Path(_RSCRIPT_BIN).is_file():
            return json.dumps({"status": "error",
                               "message": f"Rscript not found at {_RSCRIPT_BIN}",
                               "hint": "set DIFFBIND_RSCRIPT env or install R"})
        r_check = 'cat(R.version.string, "\\n"); '\
                  'pkgs <- c("DiffBind","BiocParallel","rtracklayer","GenomicRanges"); '\
                  'for (p in pkgs) cat(p, ":", as.character(packageVersion(p)), "\\n")'
        try:
            proc = subprocess.run(
                [_RSCRIPT_BIN, "-e", r_check],
                capture_output=True, text=True, timeout=60,
            )
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

        return json.dumps(
            {
                "status":    "ok" if proc.returncode == 0 else "error",
                "returncode": proc.returncode,
                "rscript":   _RSCRIPT_BIN,
                "stdout":    proc.stdout,
                "stderr":    proc.stderr,
                "r_script_bundled": str(_R_SCRIPT),
                "r_script_exists":  _R_SCRIPT.is_file(),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
