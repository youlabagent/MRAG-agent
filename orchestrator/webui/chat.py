"""LLM-driven chat layer.

Constrained tool-call pattern (no free-form code execution):
  - LLM returns strict JSON: {"reply": str, "actions": [{"tool": str, "args": {...}}, ...]}
  - Each action is dispatched through a whitelisted tool function
  - Tool results fed back to LLM until it has nothing more to do
  - Path arguments validated against BROWSE_ROOTS
"""
from __future__ import annotations
import json
import os
import re
import shlex
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
import sys as _sys
_sys.path.insert(0, "/BIO1/MRAG/Agent/orchestrator/webui")
import locus_gen


# ============================================================
# Tool registry
# ============================================================

TOOL_DEFS = {
    "list_data_paths": {
        "description": "List paths under /BIO1/MRAG/MRAG_data so the user can choose. No args.",
        "args_schema": {},
    },
    "detect": {
        "description": "Detect samples from a FASTQ directory. Returns sample sheet + sheet_id.",
        "args_schema": {
            "fastq_dir":  "absolute server path (must be under an allowed data root, e.g. /BIO1/MRAG/MRAG_data)",
            "llm_assist": "boolean, default false — only set true if previous detect had low-confidence samples",
        },
    },
    "start_run": {
        "description": "Build run config from sheet_id and spawn the pipeline. "
                       "Precomputed paths optional; provide as a dict.",
        "args_schema": {
            "sheet_id":    "string: use the EXACT sheet_id the last detect returned. If unsure, OMIT it and the server uses the most recent detect. NEVER invent one.",
            "canonical_N": "int 4/6/8 (default 6)",
            "organism": "string: species/genome for the run - 'human'/'hg38' or 'mouse'/'mm10'. "
                        "OMIT IT unless the user states the species or corrects you: it is read from the "
                        "study metadata of the download, and a wrong value aligns the data to the wrong genome.",
            "treatment_arm": "string: which condition is the TREATMENT arm, e.g. 'ONECUT2-OE'. "
                             "The FIRST condition is the treatment arm everywhere downstream - it sets every fold sign, "
                             "the red/blue locus-track colours and which arm's peaks ROSE calls super-enhancers on. "
                             "The detector orders arms by filename, which is often the reverse of what the user means, "
                             "so CONFIRM the arm order with the user and pass this when they name the arm of interest.",
            "cancer_type": "string: cancer/disease context for the LLM analysis, e.g. 'prostate cancer', 'breast cancer'. ASK the user for this before running.",
            "cell_line":   "string: cell line, e.g. 'PC-3', 'MCF-7' (optional; used for AlphaGenome).",
            "tissue":      "string: tissue (free text like 'prostate') or ontology CURIE (EFO:/UBERON:) for AlphaGenome.",
            "literature_domains": "string or list: literature domains searched for target-gene evidence; several can be combined, "
                                  "e.g. 'PC, NEC, SCLC' (prostate cancer, neuroendocrine carcinoma, small cell lung cancer). "
                                  "Optional; defaults to the cancer_type.",
            "precomputed": "dict of {key: path}. Keys: integrated_csv, atac_md_csv, deseq_csv, "
                           "diffbind_out_csv, motif_tsv, tf_table_csv. "
                           "All paths must be absolute and exist on the server.",
            "axis_headline": "string: which model set LEADS Section 13 (its first tab and emphasised row): "
                             "'auto' (default rule: the consensus when both arms ran and both single-model "
                             "tests survive Holm), 'both', 'alphagenome' or 'corgi'. Optional. For a bulk "
                             "prostate-adenocarcinoma (PRAD) tumour cohort the PI declaration is 'alphagenome' "
                             "with axis_ar_gate=true: SUGGEST it and let the user decide; a cell-line run "
                             "(e.g. LNCaP / OC2) stays 'auto'.",
            "axis_ar_gate": "bool: report Section 13's AR cross-condition row (a declared PRAD-only extra "
                            "requirement). Optional, default false.",
        },
    },
    "list_runs": {
        "description": "List runs with their status. No args.",
        "args_schema": {},
    },
    "get_status": {
        "description": "Get live pipeline progress of a run: overall percent, "
                       "current stage + sub-detail (e.g. 'aligning (1/4 BAM)'), and "
                       "per-stage status grouped by phase. Works even before the "
                       "manifest is written (reflects ATAC/RNA alignment progress).",
        "args_schema": {"run_id": "string"},
    },
    "open_report": {
        "description": "Returns the URL of the HTML report for a completed run.",
        "args_schema": {"run_id": "string"},
    },
    "make_peak_graph": {
        "description": "Render an ATAC peak graph (locus snapshot) for ANY target "
                       "gene on demand and show it in the plot pane. Use when the "
                       "user asks to see/view a peak, locus, or accessibility track "
                       "for a gene (e.g. 'show me the AR peak', 'FOXA1 locus 보여줘'). "
                       "Does NOT modify the report — viewing only.",
        "args_schema": {"run_id": "string (optional, defaults to last run)",
                        "gene": "gene symbol, e.g. 'AR'"},
    },
    "add_peak_to_report": {
        "description": "Generate (if needed) a peak graph for a gene AND add it to "
                       "report Section 12 (Locus Snapshots). Use when the user wants "
                       "a gene's peak INCLUDED in the report.",
        "args_schema": {"run_id": "string (optional, defaults to last run)",
                        "gene": "gene symbol to add"},
    },
    "remove_peak_from_report": {
        "description": "Remove a gene's peak graph from report Section 12. Use when "
                       "the user wants to drop/exclude a locus panel from the report.",
        "args_schema": {"run_id": "string (optional, defaults to last run)",
                        "gene": "gene symbol to remove"},
    },
    "list_report_peaks": {
        "description": "List which locus panels (genes) are currently included in "
                       "report Section 12. No gene arg.",
        "args_schema": {"run_id": "string (optional, defaults to last run)"},
    },
    "sra_lookup": {
        "description": "Expand an SRA/GEO accession to its whole study and judge whether "
                       "MRAG can analyse it. Downloads NOTHING. Always call this first "
                       "when the user mentions an accession.",
        "args_schema": {
            "accession": "One accession, or several separated by spaces or commas. "
                         "Pass the RNA accession AND the ATAC accession together when "
                         "the two were deposited apart, which is common.",
        },
    },
    "sra_fetch": {
        "description": "Download the study FASTQs. REQUIRES confirmed=true, which you may "
                       "only set after showing the user the sample table and total size "
                       "from sra_lookup and receiving an explicit yes. Runs in background. "
                       "Returns ok=true and state='started' ONLY when a download process "
                       "was really launched; anything else is a refusal or an error.",
        "args_schema": {
            "accession": "same accession(s) as sra_lookup - pass ALL of them (ATAC + RNA) "
                         "in one call; omit to reuse the accessions of the last lookup. "
                         "RUN accessions (SRR/ERR/DRR) download ONLY those runs; a study "
                         "accession (SRP/PRJNA/GSE) downloads the whole study. Use the "
                         "SAME list the user approved in sra_lookup.",
            "confirmed": "boolean; set true ONLY after the user approved the download size",
            "dest": "OMIT IT. The download goes to the `dest` sra_lookup reported, which "
                    "is the path the user was shown. A different dest is REFUSED unless "
                    "the user named that exact directory themselves.",
            "dest_confirmed": "boolean; only when the USER named a directory other than "
                              "the one sra_lookup reported, and you quoted it back to them",
        },
    },
    "sra_fetch_status": {
        "description": "Truthful state of a download directory. States: not_started "
                       "(NOTHING is downloading), starting, running, partial, done, "
                       "refused, failed. Only 'done' means the FASTQs are there.",
        "args_schema": {
            "dest": "download directory; omit to reuse the dest of the last sra_fetch "
                    "THIS SESSION actually launched",
        },
    },
    "sra_cancel": {
        "description": "STOP a download that is running. Use it when the user says stop, "
                       "cancel, abort, halt or pause. The partly transferred file is KEPT, "
                       "so sra_resume can continue from it. Downloads nothing and deletes "
                       "nothing. `killed: true` means a process was really signalled and "
                       "is really gone; ok=false means NOTHING was stopped.",
        "args_schema": {
            "dest": "download directory; omit to use the fetch this session started "
                    "(or the live one recorded on disk)",
        },
    },
    "sra_resume": {
        "description": "CONTINUE a cancelled, partial or failed download from the bytes "
                       "already on disk. Use it when the user says resume, continue or "
                       "restart the download. It can only continue a download that was "
                       "already approved and already started in that directory - it can "
                       "NEVER begin a new one, so a fresh study still needs sra_lookup "
                       "plus an explicit yes plus sra_fetch.",
        "args_schema": {
            "dest": "download directory; omit to use the fetch this session started "
                    "(or the live one recorded on disk)",
        },
    },
}

# Tolerate common hallucinated tool names the LLM emits for the locus feature
TOOL_ALIASES = {
    "generate_plot": "make_peak_graph", "plot_peak": "make_peak_graph",
    "show_peak": "make_peak_graph", "peak_graph": "make_peak_graph",
    "make_locus": "make_peak_graph", "show_locus": "make_peak_graph",
    "locus_plot": "make_peak_graph", "plot_locus": "make_peak_graph",
    "view_peak": "make_peak_graph",
    "add_peak": "add_peak_to_report", "remove_peak": "remove_peak_from_report",
    "list_peaks": "list_report_peaks",
    # download control: the model reaches for these names when the user says
    # "stop"/"resume", and an unknown tool costs a whole extra LLM turn.
    "sra_stop": "sra_cancel", "stop_download": "sra_cancel",
    "cancel_download": "sra_cancel", "sra_fetch_cancel": "sra_cancel",
    "sra_fetch_stop": "sra_cancel", "sra_abort": "sra_cancel",
    "resume_download": "sra_resume", "sra_fetch_resume": "sra_resume",
    "sra_continue": "sra_resume", "continue_download": "sra_resume",
    "sra_status": "sra_fetch_status", "sra_progress": "sra_fetch_status",
}


# ============================================================
# Section 13 declaration
# ============================================================
# Which model set LEADS Section 13 and whether the AR cross-condition row is reported are
# DECLARED by the run (run.toml top-level axis_headline / axis_ar_gate, mrag_orchestrator.config):
# never "the smallest p".  A PRAD bulk-tumour run declares
# axis_headline = "alphagenome" + axis_ar_gate = true (Corgi has no matching input condition on a
# tumour cohort); OC2 and every other run leave both unset and the node's default rule applies
# (the consensus leads when both arms ran and both single-model headlines survive Holm).  The web
# UI writes them on the user's choice; it only SUGGESTS the PRAD declaration.
AXIS_HEADLINE_CHOICES = ("auto", "both", "alphagenome", "corgi")
_PRAD_RE = re.compile(r"prostate\s+adeno|\bprad\b|tcga[-_ ]?prad", re.I)


def axis_declaration_suggestion(*texts) -> dict:
    """What the confirm form pre-selects for Section 13, from whatever the run says about its
    disease (cancer type, literature domains, tissue): prostate adenocarcinoma -> AlphaGenome + the
    AR gate, with the reason shown next to the fields; anything else -> auto / no AR gate.  It is a
    suggestion: the user's form choice is what is written."""
    t = " ".join(str(x) for x in texts if x)
    if _PRAD_RE.search(t):
        return {"axis_headline": "alphagenome", "axis_ar_gate": True,
                "why": ("suggested because this run names prostate adenocarcinoma: a bulk PRAD tumour "
                        "cohort leads Section 13 with AlphaGenome and reports the AR gate. For a "
                        "cell-line run (e.g. LNCaP / OC2) choose auto and leave the AR gate off.")}
    return {"axis_headline": "auto", "axis_ar_gate": False, "why": ""}


def _axis_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v or "").strip().lower()
    if s in ("", "0", "false", "no", "off"):
        return False
    if s in ("1", "true", "yes", "on"):
        return True
    raise ValueError("axis_ar_gate must be a boolean, got %r" % (v,))


def write_axis_declaration(config_path, axis_headline="auto", axis_ar_gate=False) -> dict:
    """Write run.toml's TOP-LEVEL axis_headline / axis_ar_gate (above the first [table] header,
    where mrag_orchestrator.config reads them).  "auto" (or "") removes axis_headline, so the
    node's default rule applies.  Every other key of the file is left byte for byte, and the
    result is re-parsed: if anything but the two keys changed, the original is restored and
    ValueError is raised.  Returns what was written."""
    import tomllib
    h = str(axis_headline or "auto").strip().lower()
    if h not in AXIS_HEADLINE_CHOICES:
        raise ValueError("axis_headline must be one of %s, got %r" % ("/".join(AXIS_HEADLINE_CHOICES),
                                                                       axis_headline))
    g = _axis_bool(axis_ar_gate)
    p = Path(config_path)
    orig = p.read_text()
    lines = orig.split("\n")
    first = next((k for k, ln in enumerate(lines) if re.match(r"\s*\[", ln)), len(lines))
    keep = [ln for k, ln in enumerate(lines)
            if not (k < first and re.match(r"\s*axis_(headline|ar_gate)\s*=", ln))]
    add = ([] if h == "auto" else ['axis_headline = "%s"' % h]) + ["axis_ar_gate = %s" % ("true" if g else "false")]
    new = "\n".join(add + keep)
    want = dict(tomllib.loads(orig))
    want.pop("axis_headline", None)
    want.pop("axis_ar_gate", None)
    got = tomllib.loads(new)
    chk = dict(got)
    if chk.pop("axis_headline", None) != (None if h == "auto" else h) or chk.pop("axis_ar_gate", None) is not g \
            or chk != want:
        raise ValueError("run.toml did not take the Section-13 declaration cleanly; left unchanged")
    p.write_text(new)
    return {"axis_headline": "" if h == "auto" else h, "axis_ar_gate": g}



# Returns Callable[[ctx, args], dict] for each tool
def _build_dispatcher(ctx: "ChatContext") -> dict[str, Callable[[dict], dict]]:
    return {
        "list_data_paths": lambda args: _t_list_data_paths(),
        "detect":          lambda args: _t_detect(args, ctx),
        "start_run":       lambda args: _t_start_run(args, ctx),
        "list_runs":       lambda args: _t_list_runs(ctx),
        "get_status":      lambda args: _t_get_status(args, ctx),
        "open_report":     lambda args: _t_open_report(args, ctx),
        "make_peak_graph":      lambda args: _t_make_peak_graph(args, ctx),
        "add_peak_to_report":   lambda args: _t_add_peak_to_report(args, ctx),
        "remove_peak_from_report": lambda args: _t_remove_peak_from_report(args, ctx),
        "list_report_peaks":    lambda args: _t_list_report_peaks(args, ctx),
        "sra_lookup":       lambda args: _t_sra_lookup(args, ctx),
        "sra_fetch":        lambda args: _t_sra_fetch(args, ctx),
        "sra_fetch_status": lambda args: _t_sra_fetch_status(args, ctx),
        "sra_cancel":       lambda args: _t_sra_cancel(args, ctx),
        "sra_resume":       lambda args: _t_sra_resume(args, ctx),
    }


# ============================================================
# Path safety
# ============================================================

# A second data root besides MRAG_data that detect / start_run / sra_fetch and the file browser
# accept.  MRAG_EXTRA_DATA_ROOT overrides the default; set it empty for none.
EXTRA_DATA_ROOT = os.environ.get("MRAG_EXTRA_DATA_ROOT", "/BIO1/MRAG/HYK")

ALLOWED_ROOTS = [
    "/BIO1/MRAG/MRAG_data",
    *([EXTRA_DATA_ROOT] if EXTRA_DATA_ROOT else []),
    "/BIO1/MRAG/Agent/reports",
    "/BIO1/MRAG/Agent/data",
    "/tmp",
]


# Run and sheet ids the chat tools accept: a single directory name under runs_root (the same rule as
# app.py's _RUN_ID_RE), so an LLM-supplied id can never point outside it.
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")


def _bad_id(ctx: "ChatContext", name: str) -> Optional[dict]:
    """None when `name` is one directory directly under ctx.runs_root, else an _error result."""
    if (isinstance(name, str) and _SAFE_ID_RE.fullmatch(name)
            and (ctx.runs_root / name).resolve().parent == ctx.runs_root.resolve()):
        return None
    return {"_error": "invalid id %r" % (name,)}


def _truthy_arg(val) -> bool:
    """Only a REAL affirmative counts as true (2026-09-23 fix).

    `if not args.get("confirmed")` was a bare truthiness test, so any non-empty
    string passed it: a model that stringified its boolean -- "false", "no",
    "0", "null", "none" -- started a 60 GB download on a shared server with no
    user approval.  Booleans the MODEL supplies now have to spell an affirmative
    out; anything else is a refusal.
    """
    if val is True:
        return True
    if val is None or val is False:
        return False
    if isinstance(val, (int, float)):
        return val == 1
    return str(val).strip().lower() in ("true", "yes", "y", "1", "on", "confirmed")


def _norm_path(p: str) -> str:
    """Comparable absolute form, for 'is this the directory the user approved?'."""
    try:
        return os.path.normpath(os.path.abspath(str(p or "").strip()))
    except Exception:                          # noqa: BLE001 - compared, never opened
        return str(p or "").strip()


def _path_allowed(p: str) -> bool:
    try:
        rp = Path(p).resolve()
    except Exception:
        return False
    for r in ALLOWED_ROOTS:
        # Compare path COMPONENTS, not string prefixes: "/BIO1/MRAG/MRAG_data_evil"
        # starts with "/BIO1/MRAG/MRAG_data" but is a different directory, and
        # sra_fetch now honours a dest the MODEL chose and writes tens of GB into
        # it -- this check is the only thing between that text and the filesystem.
        try:
            root = Path(r).resolve()
        except Exception:
            continue
        if rp == root or root in rp.parents:
            return True
    return False


def _sra_live_dest() -> str:
    """A dest on disk that a still-running fetch was launched into, or "".

    Belt and braces for a lost session (2026-09-23 fix): the child is detached
    (start_new_session=True) and outlives the web UI, so after a restart
    ctx.last_sra is gone while the download is genuinely running.  Saying
    "nothing is downloading" then would be the mirror image of the bug this
    whole change set fixes.
    """
    mod = _sra_mod()
    if mod is None or not hasattr(mod, "read_launch") or not hasattr(mod, "pid_alive"):
        return ""
    root = getattr(mod, "DATA_ROOT", None)
    if root is None:
        return ""
    lf = getattr(mod, "LAUNCH_FILE", "sra_fetch.launch.json")
    try:
        cands = sorted(Path(root).glob("*/" + lf),
                       key=lambda f: f.stat().st_mtime, reverse=True)[:20]
    except Exception:
        return ""
    for f in cands:
        try:
            rec = mod.read_launch(f.parent) or {}
            if rec.get("pid") and mod.pid_alive(int(rec["pid"])):
                return str(f.parent)
        except Exception:
            continue
    return ""


# ============================================================
# Tool implementations
# ============================================================

def _run_mrag(args: list[str], timeout: int = 600) -> dict:
    mrag = os.environ.get("MRAG_BIN", "/BIO1/MRAG/venvs/paper_qa/bin/mrag")
    try:
        proc = subprocess.run([mrag] + args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"_error": f"mrag timed out: {' '.join(shlex.quote(a) for a in args)}"}
    if proc.returncode not in (0, 3):
        return {"_error": f"mrag rc={proc.returncode}", "_stderr": proc.stderr[:400]}
    out = (proc.stdout or "").strip()
    if not out:
        return {}
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"_raw": out[:2000]}


def _t_list_data_paths() -> dict:
    root = Path("/BIO1/MRAG/MRAG_data")
    if not root.exists():
        return {"_error": f"{root} does not exist on the server"}
    items = []
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            n_fq = sum(1 for f in p.iterdir() if f.is_file()
                       and f.name.endswith((".fastq.gz", ".fq.gz", ".fastq", ".fq")))
            items.append({"path": str(p), "type": "dir", "n_fastq_inside": n_fq})
        elif p.name.endswith((".fastq.gz", ".fq.gz")):
            # skip individual fastq listing to keep payload small
            continue
    return {"root": str(root), "items": items[:50]}


def _t_detect(args: dict, ctx: "ChatContext") -> dict:
    fastq_dir = args.get("fastq_dir", "")
    if not fastq_dir or not _path_allowed(fastq_dir):
        return {"_error": f"fastq_dir not allowed: {fastq_dir}",
                "_hint": f"allowed roots: {ALLOWED_ROOTS}"}
    if not Path(fastq_dir).is_dir():
        return {"_error": f"not a directory: {fastq_dir}"}
    sheet_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = ctx.runs_root / f"sheet_{sheet_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = out_dir / "sample_sheet.json"
    args_list = ["detect", "--fastq-dir", fastq_dir, "--output", str(sheet_path)]
    if _truthy_arg(args.get("llm_assist")):
        args_list += ["--llm-assist", "--endpoint", ctx.llm_endpoint, "--model", ctx.llm_model]
    res = _run_mrag(args_list, timeout=600)
    if "_error" in res:
        return res
    if not sheet_path.exists():
        return {"_error": "detect produced no sample sheet"}
    sheet = json.loads(sheet_path.read_text())
    # Concise summary (don't blow context with full sample list)
    samples_brief = [{"name": s["name"], "modality": s["modality"], "arm": s["arm"],
                      "rep": s["replicate"], "conf_arm": s["confidence"]["arm"]}
                     for s in sheet["samples"]]
    ctx.last_sheet_id = sheet_id
    return {
        "sheet_id":          sheet_id,
        "n_samples":         len(sheet["samples"]),
        "n_conditions":      len(sheet["conditions"]),
        "needs_user_action": sheet["needs_user_action"],
        "summary":           sheet["summary_human"],
        "samples":           samples_brief,
        "conditions":        [{"label": c["label"], "n_reps": c["n_reps"]} for c in sheet["conditions"]],
        "warnings":          [w["msg"] for w in sheet.get("warnings", [])],
        "precomputed_suggestion": sheet.get("precomputed_suggestion", {}),
        "processed_inputs":  bool(sheet.get("precomputed_suggestion")),
    }


def _t_start_run(args: dict, ctx: "ChatContext") -> dict:
    sheet_id = args.get("sheet_id") or ctx.last_sheet_id
    if not sheet_id:
        return {"_error": "no sheet_id (run detect first, or supply explicitly)"}
    if _bad_id(ctx, f"sheet_{sheet_id}"):
        # an id that is not even a directory name: the LLM made it up, use the last real detect
        if not ctx.last_sheet_id or _bad_id(ctx, f"sheet_{ctx.last_sheet_id}"):
            return _bad_id(ctx, f"sheet_{sheet_id}")
        sheet_id = ctx.last_sheet_id
    sheet_path = ctx.runs_root / f"sheet_{sheet_id}" / "sample_sheet.json"
    # LLM sometimes fabricates a sheet_id; fall back to the last real detect.
    if not sheet_path.exists() and ctx.last_sheet_id and ctx.last_sheet_id != sheet_id:
        alt = ctx.runs_root / f"sheet_{ctx.last_sheet_id}" / "sample_sheet.json"
        if alt.exists():
            sheet_id, sheet_path = ctx.last_sheet_id, alt
    if not sheet_path.exists():
        return {"_error": f"sheet_id {sheet_id} not found; run detect first"}
    canonical_N = int(args.get("canonical_N") or 6)
    pre = args.get("precomputed") or {}
    # Validate every supplied path
    for k, v in list(pre.items()):
        if not v: pre.pop(k); continue
        if not _path_allowed(v):
            return {"_error": f"precomputed.{k} path not allowed: {v}"}
        if not Path(v).exists():
            return {"_error": f"precomputed.{k} does not exist: {v}"}

    run_id = f"run_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    run_dir = ctx.runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run.toml"
    output_dir  = run_dir / "output"

    args_list = ["confirm",
                 "--sheet", str(sheet_path),
                 "--output-config", str(config_path),
                 "--output-dir", str(output_dir),
                 "--run-label", run_id,
                 "--canonical-N", str(canonical_N)]
    _ld = args.get("literature_domains")
    if isinstance(_ld, (list, tuple)):
        args["literature_domains"] = ",".join(str(x) for x in _ld)
    for _flag, _key in (("--organism", "organism"),
                        ("--treatment-arm", "treatment_arm"),
                        ("--cancer-type", "cancer_type"),
                        ("--cell-line", "cell_line"),
                        ("--tissue", "tissue"),
                        ("--literature-domains", "literature_domains")):
        _val = (str(args.get(_key) or "")).strip()
        if _val:
            args_list += [_flag, _val]
    if pre:
        args_list += ["--precomputed"] + [f"{k}={v}" for k, v in pre.items()]
    confirm_res = _run_mrag(args_list, timeout=60)
    if "_error" in confirm_res:
        return confirm_res
    # Section 13's declaration, written only as the user chose it
    try:
        axis_decl = write_axis_declaration(config_path, args.get("axis_headline") or "auto",
                                           args.get("axis_ar_gate") or False)
    except Exception as _e:
        return {"_error": f"Section-13 declaration not written to run.toml: {_e}"}
    axis_sug = axis_declaration_suggestion(args.get("cancer_type"), args.get("literature_domains"),
                                           args.get("tissue"))

    # Spawn `mrag run`
    log_path = run_dir / "run.log"
    fh = open(log_path, "w")
    proc = subprocess.Popen(
        [os.environ.get("MRAG_BIN", "/BIO1/MRAG/venvs/paper_qa/bin/mrag"),
         "run", "--config", str(config_path)],
        stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
    )
    (run_dir / "pid").write_text(str(proc.pid))
    ctx.last_run_id = run_id
    return {
        "run_id":   run_id,
        "spawned":  True,
        "pid":      proc.pid,
        "status_url": f"/run/{run_id}",
        "organism": confirm_res.get("organism", ""),
        "genome": confirm_res.get("genome", ""),
        "organism_note": confirm_res.get("organism_note", ""),
        # State the arm order back to the user EVERY time, not only when they
        # set it: it is the one choice that silently inverts the whole report.
        "conditions": confirm_res.get("conditions"),
        "treatment_arm": confirm_res.get("treatment_arm", ""),
        "control_arm": confirm_res.get("control_arm", ""),
        "arm_order_note": confirm_res.get("arm_order_note", ""),
        "literature_domains": confirm_res.get("literature_domains"),
        "literature_unmatched": confirm_res.get("literature_unmatched"),
        "axis_headline": axis_decl["axis_headline"] or "auto",
        "axis_ar_gate": axis_decl["axis_ar_gate"],
        "note":     "Pipeline started. Use get_status to check progress."
                    + (" Section 13 runs with axis_headline=auto / no AR gate although this run names "
                       "prostate adenocarcinoma: tell the user that a PRAD tumour cohort normally declares "
                       "axis_headline='alphagenome' with axis_ar_gate=true." if (
                           axis_sug["axis_headline"] != "auto" and not axis_decl["axis_headline"]
                           and not axis_decl["axis_ar_gate"]) else "")
                    + (" Tell the user these literature-domain entries matched no corpus and were ignored: "
                       + ", ".join(confirm_res.get("literature_unmatched") or []) + "."
                       if confirm_res.get("literature_unmatched") else ""),
    }


def _t_list_runs(ctx: "ChatContext") -> dict:
    runs = []
    for d in sorted(ctx.runs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir() or not d.name.startswith("run_"): continue
        mp = d / "output" / "manifest.json"
        info = {"run_id": d.name, "status": "pending"}
        if mp.exists():
            try:
                m = json.loads(mp.read_text())
                info["status"] = m.get("status", "?")
                info["n_steps"] = len(m.get("steps", []))
            except Exception:
                pass
        runs.append(info)
    return {"runs": runs[:20]}


def _t_get_status(args: dict, ctx: "ChatContext") -> dict:
    run_id = args.get("run_id") or ctx.last_run_id
    if not run_id:
        return {"_error": "no run_id"}
    if _bad_id(ctx, run_id):
        return _bad_id(ctx, run_id)
    # Reuse the live progress computation (phases + percent + ATAC/RNA sub-detail).
    prog = None
    try:
        from app import run_progress  # lazy: app imports chat, avoid circular at module load
        prog = json.loads(run_progress(run_id).body)
    except Exception:
        prog = None

    mp = ctx.runs_root / run_id / "output" / "manifest.json"
    manifest = None
    if mp.exists():
        try:
            manifest = json.loads(mp.read_text())
        except Exception:
            manifest = None

    if prog:
        cur = next((s for s in prog["stages"] if s["status"] == "running"), None)
        return {
            "run_id":         run_id,
            "percent":        prog["percent"],
            "alive":          prog["alive"],
            "finished":       prog["finished"],
            "current_stage":  prog["current"],
            "current_detail": (cur.get("detail") if cur else "") or "",
            "stages_done":    prog["n_done"],
            "stages_total":   prog["n_total"],
            "stages": [
                {"phase": s["phase"], "stage": s["label"], "status": s["status"],
                 **({"detail": s["detail"]} if s.get("detail") else {})}
                for s in prog["stages"]
            ],
            "manifest_steps": (
                [{"name": s["name"], "status": s["status"]}
                 for s in manifest.get("steps", [])] if manifest else []
            ),
        }

    # Fallback if progress computation failed
    if not manifest:
        return {"run_id": run_id, "status": "pending",
                "note": "manifest not yet written; ATAC/RNA preprocessing (alignment) "
                        "likely in progress"}
    return {
        "run_id":  run_id,
        "status":  manifest.get("status"),
        "n_steps": len(manifest.get("steps", [])),
        "steps":   [{"name": s["name"], "status": s["status"]}
                    for s in manifest.get("steps", [])],
        "summary": manifest.get("summary", {}),
    }


def _t_open_report(args: dict, ctx: "ChatContext") -> dict:
    run_id = args.get("run_id") or ctx.last_run_id
    if not run_id:
        return {"_error": "no run_id"}
    if _bad_id(ctx, run_id):
        return _bad_id(ctx, run_id)
    html_path = ctx.runs_root / run_id / "output" / "mrag_report.html"
    if not html_path.exists():
        return {"_error": f"report not generated yet for {run_id}"}
    return {"run_id": run_id, "report_url": f"/report/{run_id}",
            "download_url": f"/api/download/{run_id}"}


def _t_make_peak_graph(args: dict, ctx: "ChatContext") -> dict:
    run_id = args.get("run_id") or ctx.last_run_id
    gene = (args.get("gene") or "").strip()
    if not run_id:
        return {"_error": "no run_id"}
    if _bad_id(ctx, run_id):
        return _bad_id(ctx, run_id)
    if not gene:
        return {"_error": "gene required"}
    try:
        r = locus_gen.generate(str(ctx.runs_root), run_id, gene)
    except Exception as ex:
        return {"_error": str(ex)}
    return {"run_id": run_id, "gene": r["gene"], "region": r["region"],
            "region_source": r["source"], "plot_path": r["png_rel"],
            "note": "Rendered in the plot pane under Section 12. Not yet in the "
                    "report — say 'add it to the report' to include it."}


def _t_add_peak_to_report(args: dict, ctx: "ChatContext") -> dict:
    run_id = args.get("run_id") or ctx.last_run_id
    gene = (args.get("gene") or "").strip()
    if not run_id:
        return {"_error": "no run_id"}
    if _bad_id(ctx, run_id):
        return _bad_id(ctx, run_id)
    if not gene:
        return {"_error": "gene required"}
    try:
        r = locus_gen.add_to_report(str(ctx.runs_root), run_id, gene)
    except Exception as ex:
        return {"_error": str(ex)}
    return {"run_id": run_id, "gene": r["gene"], "region": r["region"],
            "n_panels_now": r["n_panels"], "report_updated": r["spliced"],
            "note": "Added to report Section 12." if r["spliced"]
                    else "Saved, but report HTML not found to splice yet."}


def _t_remove_peak_from_report(args: dict, ctx: "ChatContext") -> dict:
    run_id = args.get("run_id") or ctx.last_run_id
    gene = (args.get("gene") or "").strip()
    if not run_id:
        return {"_error": "no run_id"}
    if _bad_id(ctx, run_id):
        return _bad_id(ctx, run_id)
    if not gene:
        return {"_error": "gene required"}
    try:
        r = locus_gen.remove_from_report(str(ctx.runs_root), run_id, gene)
    except Exception as ex:
        return {"_error": str(ex)}
    return {"run_id": run_id, "removed": r["removed"],
            "n_panels_now": r["n_panels"], "report_updated": r["spliced"]}


def _t_list_report_peaks(args: dict, ctx: "ChatContext") -> dict:
    run_id = args.get("run_id") or ctx.last_run_id
    if not run_id:
        return {"_error": "no run_id"}
    if _bad_id(ctx, run_id):
        return _bad_id(ctx, run_id)
    try:
        return locus_gen.list_report(str(ctx.runs_root), run_id)
    except Exception as ex:
        return {"_error": str(ex)}


# ============================================================
# SRA / GEO ingestion
# ============================================================

def _sra_mod():
    """The orchestrator's sra module, or None. The webui runs on the same venv."""
    try:
        import mrag_orchestrator.sra as _sra
        return _sra
    except Exception:
        return None


def _sra_accs(value) -> list[str]:
    """One accession, a list, or "SRP000001 SRP000002" -> ["SRP000001", "SRP000002"].

    The model routinely hands several accessions through as ONE string. Splitting
    here is what lets ATAC and RNA deposited as separate studies be looked up
    together.
    """
    mod = _sra_mod()
    if mod is not None:
        try:
            return mod.parse_accessions(value)
        except Exception:
            pass
    raw = list(value) if isinstance(value, (list, tuple)) else [value]
    out: list[str] = []
    for v in raw:
        for tok in re.split(r"[,;+\s]+", str(v or "")):
            tok = tok.strip().upper()
            if tok and tok not in out:
                out.append(tok)
    return out


def _sra_state(dest: str) -> dict:
    """Ask sra.fetch_state; fall back to the CLI if the import is unavailable."""
    mod = _sra_mod()
    if mod is not None and hasattr(mod, "fetch_state"):
        try:
            return mod.fetch_state(dest)
        except Exception as exc:                   # noqa: BLE001 - reported, not hidden
            return {"ok": False, "state": "failed", "dest": dest,
                    "user_message": "Could not read the download state of %s: %s: %s"
                                    % (dest, type(exc).__name__, exc)}
    res = _run_mrag(["sra-status", "--dest", dest], timeout=60)
    if "_error" in res:
        res.setdefault("ok", False)
        res.setdefault("state", "failed")
        res.setdefault("user_message", res["_error"])
    return res


def _t_sra_lookup(args: dict, ctx: "ChatContext") -> dict:
    accs = _sra_accs(args.get("accession"))
    if not accs:
        return {"ok": False, "state": "error",
                "_error": "accession required (SRR/ERR/DRR, SRP/ERP/DRP, PRJNA, GSE or GSM)",
                "user_message": "I need at least one accession (SRR/ERR/DRR, SRP/ERP/DRP, "
                                "PRJNA, GSE or GSM) before I can look anything up."}
    res = _run_mrag(["sra-lookup"] + accs, timeout=600)
    if "_error" in res:
        return {"ok": False, "state": "error", "accession": accs,
                "_error": res["_error"], "_stderr": res.get("_stderr", ""),
                "user_message": "The lookup of %s failed: %s"
                                % (", ".join(accs), res["_error"])}
    if not res.get("ok"):
        err = res.get("error", "lookup failed")
        return {"ok": False, "state": "error", "accession": accs, "_error": err,
                "user_message": "The lookup of %s failed: %s" % (", ".join(accs), err)}
    analysable = res["status"] == "ok"
    ctx.last_sra = dict(getattr(ctx, "last_sra", None) or {})
    sel = res.get("selection") or {}
    summ = str(sel.get("summary") or "")
    ctx.last_sra.update({"study": res["study"], "studies": res.get("studies", []),
                         "input": accs, "lookup_dest": res["dest"],
                         "status": res["status"], "total_human": res["total_human"],
                         "selection_summary": summ,
                         "whole_study": bool(res.get("whole_study", True))})
    brief = [{"run": s["run"], "modality": s["modality"], "arm": s["arm"],
              "rep": s["replicate"], "layout": s["layout"], "title": s["title"]}
             for s in res["samples"] if s["modality"] in ("ATAC", "RNA")][:40]
    out = {
        "ok": True, "analysable": analysable,
        "study": res["study"], "studies": res.get("studies", []),
        "input": res["input"], "dest": res["dest"],
        "n_runs": res["n_runs"], "status": res["status"],
        # The CLI has reported `selection` since run-level subsetting landed, but
        # this dict dropped it, so the model only ever saw n_runs=8 and told the
        # user the STUDY had 8 runs when it has 16 (2026-09-23 fix).
        "selection": sel, "selection_summary": summ,
        "n_selected": res.get("n_selected", res["n_runs"]),
        "n_total": res.get("n_total", res["n_runs"]),
        "whole_study": bool(res.get("whole_study", True)),
        "selected_runs": res.get("selected_runs", []),
        "n_atac": res["n_atac"], "n_rna": res["n_rna"],
        # ENA scientific_name: tell the user which genome this study will use.
        "species": res.get("species", []),
        "organism": res.get("organism", ""),
        "missing": res.get("missing", []),
        "arms": res["arms"], "per_arm": res["per_arm"],
        "total_download": res["total_human"],
        "reasons": res["reasons"], "warnings": res["warnings"],
        "samples": brief,
    }
    if not analysable:
        out["user_message"] = ("%s cannot be analysed as it stands: %s"
                               % (res["study"], " ".join(res["reasons"])))
        out["next_step"] = ("NOT analysable. Tell the user the reasons verbatim and "
                            "do NOT call sra_fetch. If `missing` names a modality, ask "
                            "the user for that accession and call sra_lookup ONCE more "
                            "with ALL accessions together.")
    else:
        _sp = ", ".join(res.get("species") or []) or "species not stated"
        out["user_message"] = ("%s is analysable: %s, %d ATAC + %d RNA runs, "
                               "arms %s - %s to download."
                               % (res["study"], _sp, res["n_atac"], res["n_rna"],
                                  ", ".join(res["arms"]),
                                  summ or res["total_human"]))
        out["next_step"] = ("Show the user the study, the sample table, the arms and "
                            "total_download, then ASK for permission. Only after an "
                            "explicit yes, call sra_fetch with confirmed=true and the "
                            "SAME accession list.")
        if not out["whole_study"]:
            out["next_step"] += (
                " This is a SUBSET of the study: say so in those words and quote "
                "selection_summary (%r) verbatim. Do NOT describe the study as "
                "having only %d runs - it has %d." % (summ, out["n_selected"],
                                                      out["n_total"]))
    return out


def _t_sra_fetch(args: dict, ctx: "ChatContext") -> dict:
    last = getattr(ctx, "last_sra", None) or {}
    accs = _sra_accs(args.get("accession")) or list(last.get("input") or []) \
        or list(last.get("studies") or []) or _sra_accs(last.get("study"))
    if not accs:
        return {"ok": False, "state": "not_started",
                "_error": "no accession; run sra_lookup first",
                "user_message": "I have no accession to download. Run sra_lookup first."}
    if not _truthy_arg(args.get("confirmed")):
        return {"ok": False, "state": "not_started",
                "_error": "download not confirmed",
                "confirmed_seen": repr(args.get("confirmed")),
                "user_message": "I did not start any download: it is not confirmed yet. "
                                "Show the user the total size from sra_lookup and get an "
                                "explicit yes first.",
                "_hint": "Downloads can be hundreds of GB. Show the user the total "
                         "size from sra_lookup, get an explicit yes, then call again "
                         "with confirmed=true."}
    # Reuse the lookup the user just approved, so the arms that were shown are
    # the arms that get downloaded.
    chk = _run_mrag(["sra-lookup"] + accs + ["--use-cache"], timeout=600)
    if "_error" in chk:
        return {"ok": False, "state": "not_started", "accession": accs,
                "_error": chk["_error"],
                "user_message": "NOTHING was downloaded. The re-check of %s failed: %s"
                                % (", ".join(accs), chk["_error"])}
    if not chk.get("ok"):
        err = chk.get("error", "lookup failed")
        return {"ok": False, "state": "not_started", "accession": accs, "_error": err,
                "user_message": "NOTHING was downloaded. The re-check of %s failed: %s"
                                % (", ".join(accs), err)}
    if chk.get("status") != "ok":
        missing = chk.get("missing", [])
        add = ("" if not missing else
               " Add the %s accession and call sra_lookup once more with ALL accessions "
               "in one call." % " and ".join(missing))
        return {"ok": False, "state": "refused", "accession": accs,
                "study": chk.get("study", ""), "missing": missing,
                "reasons": chk.get("reasons", []),
                "_error": "study is not analysable; download refused",
                "user_message": "DOWNLOAD REFUSED - nothing was transferred. %s is not "
                                "analysable: %s%s"
                                % (chk.get("study", ", ".join(accs)),
                                   " ".join(chk.get("reasons", [])), add)}
    # The user approved the destination the LOOKUP reported. Quietly honouring a
    # different one the model supplied writes tens of GB somewhere nobody agreed
    # to, when _path_allowed (which admits every allowed root) is the only check.
    approved = str(chk.get("dest") or "")
    given = str(args.get("dest") or "").strip()
    if given and _norm_path(given) not in (
            _norm_path(approved), _norm_path(last.get("fetch_dest") or "")):
        if not _truthy_arg(args.get("dest_confirmed")):
            return {"ok": False, "state": "not_started", "dest": approved,
                    "requested_dest": given,
                    "_error": "dest differs from the destination the lookup reported",
                    "user_message": "NOTHING was downloaded. I was asked to write to "
                                    "%s, but the destination for %s is %s - the path "
                                    "the user was shown and approved. Call sra_fetch "
                                    "again WITHOUT a dest to use %s."
                                    % (given, chk.get("study", ", ".join(accs)),
                                       approved, approved),
                    "next_step": "Do not choose a destination yourself. Re-call "
                                 "sra_fetch with no dest argument. Only if the USER "
                                 "named that other directory, tell them the full path, "
                                 "get a yes, and call again with dest plus "
                                 "dest_confirmed=true."}
        dest = given
    else:
        dest = approved
    if not _path_allowed(dest):
        return {"ok": False, "state": "not_started", "dest": dest,
                "_error": "dest outside the allowed roots",
                "user_message": "NOTHING was downloaded: %s is outside the directories "
                                "I may write to." % dest}
    # One download per destination (2026-09-23 fix).  Without this a retried
    # turn, a timed-out call, or a user who says yes twice launches a SECOND
    # child into the same dest: the launch record would then name only the newer
    # pid, the older child becomes an untracked orphan, both curl the same URLs
    # into the same paths, and fetch_state reports on whichever pid was written
    # last.  On a shared server that is also 60 GB of duplicated bandwidth.
    inflight = _sra_state(dest)
    if inflight.get("state") in ("starting", "running"):
        inflight = dict(inflight)
        inflight["ok"] = True
        inflight["already_running"] = True
        inflight["dest"] = dest
        inflight["user_message"] = (
            "A download into %s is ALREADY in flight (state=%s); I did NOT start a "
            "second one. %s" % (dest, inflight.get("state"),
                                inflight.get("user_message", "")))
        inflight["next_step"] = ("Do NOT call sra_fetch again for this dest. Report "
                                 "the state above and poll with sra_fetch_status.")
        ctx.last_sra = dict(last)
        ctx.last_sra.update({"study": chk.get("study", ""), "input": accs,
                             "fetch_dest": dest, "dest": dest,
                             "status": "downloading"})
        return inflight
    mod = _sra_mod()
    log = Path(dest) / "sra_fetch.log"
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {"ok": False, "state": "not_started", "dest": dest,
                "_error": str(exc),
                "user_message": "NOTHING was downloaded: could not create %s (%s)."
                                % (dest, exc)}
    mrag = os.environ.get("MRAG_BIN", "/BIO1/MRAG/venvs/paper_qa/bin/mrag")
    cmd = [mrag, "sra-fetch"] + accs + ["--dest", dest]
    try:
        with open(log, "ab") as fh:
            fh.write(("\n=== sra-fetch launched %s: %s ===\n"
                      % (datetime.now().isoformat(timespec="seconds"),
                         " ".join(shlex.quote(c) for c in cmd))).encode())
            proc = subprocess.Popen(cmd, stdout=fh, stderr=fh, start_new_session=True)
    except OSError as exc:
        return {"ok": False, "state": "failed", "dest": dest, "_error": str(exc),
                "user_message": "The download could NOT be launched: %s" % exc}
    # Record the child before anything else can poll, so "no progress file yet"
    # can never again be confused with "no fetch was ever started".
    if mod is not None and hasattr(mod, "write_launch"):
        try:
            mod.write_launch(dest, proc.pid, cmd, accs, chk.get("study", ""))
        except Exception:
            pass
    time.sleep(0.5)                 # long enough to catch an instant crash
    if proc.poll() is not None:
        st = _sra_state(dest)
        st["ok"] = False
        st.setdefault("state", "failed")
        st["user_message"] = ("The download process exited immediately (rc=%s), so "
                              "NOTHING is downloading. %s"
                              % (proc.returncode, st.get("user_message", "")))
        return st
    ctx.last_sra = dict(last)
    ctx.last_sra.update({"study": chk["study"], "studies": chk.get("studies", []),
                         "input": accs, "fetch_dest": dest, "dest": dest,
                         "fetch_pid": proc.pid, "status": "downloading",
                         "total_human": chk["total_human"]})
    return {"ok": True, "state": "started", "study": chk["study"],
            "studies": chk.get("studies", []), "accession": accs,
            "dest": dest, "lookup_dest": approved,
            "dest_overridden": bool(given and _norm_path(given) != _norm_path(approved)),
            "pid": proc.pid, "log": str(log),
            "total_download": chk["total_human"],
            "user_message": "Download started (pid %d) into %s; %s to fetch."
                            % (proc.pid, dest, chk["total_human"]),
            "next_step": "Downloading in the background. Poll with sra_fetch_status. "
                         "Only state=done means the files are there; state=failed or "
                         "not_started means they are not."}


def _t_sra_fetch_status(args: dict, ctx: "ChatContext") -> dict:
    last = getattr(ctx, "last_sra", None) or {}
    recovered = ""
    dest = str(args.get("dest") or "").strip()
    if not dest:
        # ONLY a dest this session actually launched a fetch into. The lookup dest
        # is not a download: reusing it is what made a never-started download look
        # like one that was merely slow to begin.
        dest = str(last.get("fetch_dest") or "").strip()
    if not dest:
        # The session may have been reloaded since the fetch was launched; look
        # on disk for a launch record whose pid is still alive before claiming
        # that nothing is downloading.
        dest = _sra_live_dest()
        if dest:
            recovered = dest
            ctx.last_sra = dict(last)
            ctx.last_sra["fetch_dest"] = dest
            ctx.last_sra["dest"] = dest
    if not dest:
        looked = last.get("lookup_dest") or ""
        return {"ok": False, "state": "not_started", "dest": "",
                "_error": "no fetch has been started in this session",
                "user_message": "No download has been started in this session, so there "
                                "is no progress to report. Nothing is downloading.%s"
                                % (" The last sra_lookup only inspected metadata (%s); "
                                   "a lookup never downloads anything." % looked
                                   if looked else "")}
    st = _sra_state(dest)
    if st.get("state") == "done":
        st["next_step"] = "Download complete. Call detect with fastq_dir=%s" % dest
    if recovered:
        # Say where this came from: it is a live fetch found on disk, not one
        # this session remembers launching.
        st = dict(st)
        st["recovered_from_disk"] = recovered
        st["user_message"] = (
            "This session does not remember launching a download, but a live fetch "
            "is recorded under %s. %s" % (recovered, st.get("user_message", "")))
    return st


def _t_sra_cancel(args: dict, ctx: "ChatContext") -> dict:
    """Stop the download that owns a directory. Keeps the partial file.

    The kill goes through the `mrag sra-cancel` CLI rather than an in-process
    call: that is always the on-disk version of the code, it runs in its own
    process, and it is the same command a human would type. cancel_fetch only
    ever signals a pid whose /proc cmdline still says sra-fetch, so a recycled
    pid cannot be shot.
    """
    last = getattr(ctx, "last_sra", None) or {}
    dest = str(args.get("dest") or "").strip() \
        or str(last.get("fetch_dest") or "").strip() or _sra_live_dest()
    if not dest:
        return {"ok": False, "state": "not_started", "dest": "", "killed": False,
                "_error": "no download directory to cancel",
                "user_message": "There is nothing to stop: this session never started "
                                "a download and no live fetch is recorded on disk."}
    if not _path_allowed(dest):
        return {"ok": False, "state": "not_started", "dest": dest, "killed": False,
                "_error": "dest outside the allowed roots",
                "user_message": "NOTHING was stopped: %s is outside the directories I "
                                "may touch." % dest}
    res = _run_mrag(["sra-cancel", "--dest", dest], timeout=120)
    if "_error" in res:
        st = _sra_state(dest)
        return {"ok": False, "state": st.get("state", "failed"), "dest": dest,
                "killed": False, "_error": res["_error"],
                "user_message": "I could not run the cancel command (%s), so I do NOT "
                                "know whether the download stopped. Its state is now "
                                "%r: %s" % (res["_error"], st.get("state"),
                                            st.get("user_message", ""))}
    out = dict(res)
    out.setdefault("dest", dest)
    out.setdefault("killed", False)
    out.setdefault("ok", False)
    # Keep pointing the panel and any follow-up tool at this directory, so
    # "resume" right after "stop" needs no argument from the model.
    ctx.last_sra = dict(last)
    ctx.last_sra.update({"fetch_dest": dest, "dest": dest})
    if out.get("ok") and out.get("killed"):
        out["next_step"] = ("Say once that the download was stopped and that the "
                            "partial file was kept. Offer sra_resume. Do not poll.")
    else:
        out["next_step"] = ("NOTHING was stopped. Quote user_message verbatim and ask "
                            "the user how to proceed.")
    return out


def _t_sra_resume(args: dict, ctx: "ChatContext") -> dict:
    """Continue an interrupted download; never start a new one.

    Resume is deliberately NOT gated on `confirmed`: it can only ever continue a
    transfer that a user already approved into a directory that already holds an
    interrupted fetch. Every state that is not cancelled/partial/failed is
    refused here, which is what stops "resume" from becoming an unapproved way
    to start tens of gigabytes.
    """
    last = getattr(ctx, "last_sra", None) or {}
    dest = str(args.get("dest") or "").strip() \
        or str(last.get("fetch_dest") or "").strip() or _sra_live_dest()
    if not dest:
        return {"ok": False, "state": "not_started", "dest": "", "resumed": False,
                "_error": "no download directory to resume",
                "user_message": "There is nothing to resume: no download was started "
                                "in this session and none is recorded on disk. A new "
                                "download needs sra_lookup and an explicit yes."}
    if not _path_allowed(dest):
        return {"ok": False, "state": "not_started", "dest": dest, "resumed": False,
                "_error": "dest outside the allowed roots",
                "user_message": "NOTHING was resumed: %s is outside the directories I "
                                "may write to." % dest}
    st = _sra_state(dest)
    state = str(st.get("state") or "")
    if state in ("starting", "running"):
        out = dict(st)
        out["ok"] = True
        out["resumed"] = False
        out["user_message"] = ("Nothing to resume: that download is already %s. %s"
                               % (state, st.get("user_message", "")))
        out["next_step"] = "Report the live state once. The page's panel shows it."
        return out
    if state == "done":
        out = dict(st)
        out["resumed"] = False
        out["user_message"] = ("Nothing to resume: that download already finished. %s"
                               % st.get("user_message", ""))
        return out
    if state not in ("cancelled", "partial", "failed"):
        out = dict(st)
        out["ok"] = False
        out["resumed"] = False
        out["_error"] = "nothing resumable in %s (state=%s)" % (dest, state or "unknown")
        out["user_message"] = ("NOTHING was resumed: %s holds no interrupted download "
                               "(state=%s). %s Starting a download needs sra_lookup and "
                               "an explicit yes." % (dest, state or "unknown",
                                                     st.get("user_message", "")))
        return out
    mrag = os.environ.get("MRAG_BIN", "/BIO1/MRAG/venvs/paper_qa/bin/mrag")
    cmd = [mrag, "sra-fetch", "--resume", "--dest", dest]
    log = Path(dest) / "sra_fetch.log"
    try:
        with open(log, "ab") as fh:
            fh.write(("\n=== sra-fetch --resume launched %s: %s ===\n"
                      % (datetime.now().isoformat(timespec="seconds"),
                         " ".join(shlex.quote(c) for c in cmd))).encode())
            proc = subprocess.Popen(cmd, stdout=fh, stderr=fh, start_new_session=True)
    except OSError as exc:
        return {"ok": False, "state": "failed", "dest": dest, "resumed": False,
                "_error": str(exc),
                "user_message": "The download could NOT be resumed: %s" % exc}
    # Claim the directory before anything can poll it: until the child rewrites
    # the launch record, the record still names the CANCELLED pid, and a poll in
    # that window would read "the process is gone" and call a live resume dead.
    mod = _sra_mod()
    accs = list(st.get("accessions") or [])
    if mod is not None and hasattr(mod, "write_launch"):
        try:
            mod.write_launch(dest, proc.pid, cmd, accs, st.get("study", ""))
        except Exception:
            pass
    time.sleep(0.5)                 # long enough to catch an instant crash
    if proc.poll() is not None:
        bad = _sra_state(dest)
        bad = dict(bad)
        bad["ok"] = False
        bad["resumed"] = False
        bad.setdefault("state", "failed")
        bad["user_message"] = ("The resume process exited immediately (rc=%s), so "
                               "NOTHING is downloading. %s"
                               % (proc.returncode, bad.get("user_message", "")))
        return bad
    ctx.last_sra = dict(last)
    ctx.last_sra.update({"fetch_dest": dest, "dest": dest, "fetch_pid": proc.pid,
                         "status": "downloading"})
    return {"ok": True, "state": "started", "resumed": True, "dest": dest,
            "pid": proc.pid, "accession": accs, "log": str(log),
            "user_message": "Resumed the download in %s (pid %d); it continues from "
                            "the bytes already on disk." % (dest, proc.pid),
            "next_step": "Say once that it resumed. The page's progress panel shows "
                         "it live - do NOT poll sra_fetch_status in a loop."}


# ============================================================
# Live download panel (read-only; the chat page polls this)
# ============================================================

PANEL_ACTIVE_STATES = ("starting", "running")
PANEL_SHOWN_STATES = ("starting", "running", "partial", "done",
                      "cancelled", "failed", "refused")


def _panel_left(seconds) -> str:
    """'~12 min left' — a rough figure a human reads at a glance, or ''."""
    try:
        s = int(float(seconds))
    except (TypeError, ValueError):
        return ""
    if s <= 0:
        return ""
    if s < 60:
        return "<1 min left"
    if s < 5400:
        return "~%d min left" % round(s / 60.0)
    h, m = divmod(int(round(s / 60.0)), 60)
    return "~%dh %02dm left" % (h, m)


def _panel_dest(dest: str = "", session_id: str = "") -> tuple:
    """(directory, how it was resolved) for the panel. Never invents a path.

    Resolution order is deliberately disk-first, because the panel has to
    survive a page reload AND a restart of this web UI: an explicit dest from
    the page, then the dest saved in this conversation's file, then any fetch
    on disk whose pid is still alive.
    """
    raw = str(dest or "").strip()
    if raw:
        return (raw, "given") if _path_allowed(raw) else ("", "denied")
    sid = re.sub(r"[^A-Za-z0-9_]", "", str(session_id or ""))
    if sid:
        try:
            saved = json.loads(_conv_path(sid).read_text())
        except (OSError, ValueError):
            saved = {}
        cand = str(((saved.get("last_sra") or {}).get("fetch_dest")) or "").strip()
        if cand and _path_allowed(cand):
            return cand, "session"
    live = _sra_live_dest()
    if live:
        return live, "live"
    return "", "none"


def sra_panel(dest: str = "", session_id: str = "") -> dict:
    """A compact, renderable snapshot of one download directory.

    STRICTLY read-only: it starts nothing, stops nothing and writes nothing, so
    the page can poll it every couple of seconds. `lines` is the panel exactly
    as it should read; `files`/`overall` are there for the progress bars.
    """
    d, how = _panel_dest(dest, session_id)
    base = {"active": False, "show": False, "state": "not_started", "dest": d,
            "resolved_by": how, "lines": [], "files": [], "done_runs": [],
            "pct": 0.0, "poll_ms": 0, "message": ""}
    if not d:
        base["message"] = ("That directory is not one I may read."
                           if how == "denied" else "No download is running.")
        return base
    try:
        st = _sra_state(d)
    except Exception as exc:                    # noqa: BLE001 - shown, not hidden
        base["state"] = "failed"
        base["message"] = "Could not read %s: %s: %s" % (d, type(exc).__name__, exc)
        return base

    state = str(st.get("state") or "")
    overall = st.get("overall") or {}
    sel = st.get("selection") or {}
    study = str(st.get("study") or "") or Path(d).name
    jobs = list(st.get("jobs") or [])

    files_done = overall.get("files_done", st.get("done"))
    files_total = overall.get("files_total", st.get("total_files"))
    pct = overall.get("pct", st.get("pct"))
    try:
        pct = round(float(pct), 1)
    except (TypeError, ValueError):
        pct = 0.0

    verb = {"running": "downloading", "starting": "starting",
            "done": "done", "partial": "finished with failures",
            "cancelled": "cancelled", "failed": "failed",
            "refused": "refused", "not_started": "not started"}.get(state, state)

    head = [study, verb]
    if files_total:
        head.append("%s/%s files" % (files_done if files_done is not None else 0,
                                     files_total))
    if state in ("running", "starting", "cancelled", "partial") or pct:
        head.append("%s%%" % ("%.1f" % pct).rstrip("0").rstrip("."))
    if overall.get("done_human") and overall.get("total_human"):
        head.append("(%s / %s)" % (overall["done_human"], overall["total_human"]))
    if state == "running":
        if overall.get("speed_human"):
            head.append(overall["speed_human"])
        left = _panel_left(overall.get("eta_s"))
        if left:
            head.append(left)
    lines = ["  ".join(str(h) for h in head if h)]

    files = []
    done_runs = []
    for j in jobs:
        js = str(j.get("state") or "")
        # ATAC/<arm>_rep1.fastq.gz and RNA/<arm>_rep1.fastq.gz are different
        # files with the same basename (assign_names uses <arm>_rep<N> for both),
        # so the modality directory is part of the name a reader needs.
        name = str(j.get("file") or j.get("run") or "?")
        mod = str(j.get("modality") or "").strip()
        if mod and "/" not in name:
            name = "%s/%s" % (mod, name)
        try:
            jp = round(float(j.get("pct") or 0.0), 1)
        except (TypeError, ValueError):
            jp = 0.0
        if js in ("downloading", "failed"):
            files.append({"file": name, "pct": jp, "state": js,
                          "run": j.get("run", ""), "modality": j.get("modality", "")})
        elif js == "done":
            r = str(j.get("run") or name)
            if r not in done_runs:
                done_runs.append(r)
    for f in files:
        lines.append("  %s  %s%%%s" % (f["file"], ("%.1f" % f["pct"]).rstrip("0").rstrip("."),
                                       "  FAILED" if f["state"] == "failed" else ""))
    if done_runs:
        shown = done_runs[:8]
        more = len(done_runs) - len(shown)
        lines.append("  done: %s%s" % (", ".join(shown),
                                       " (+%d more)" % more if more else ""))
    if sel.get("summary"):
        lines.append("  %s" % sel["summary"])

    return {
        "active": state in PANEL_ACTIVE_STATES,
        "show": state in PANEL_SHOWN_STATES,
        "state": state,
        "ok": bool(st.get("ok")),
        "dest": d,
        "resolved_by": how,
        "study": study,
        "pct": pct,
        "files_done": files_done,
        "files_total": files_total,
        "overall": overall,
        "selection": sel.get("summary", ""),
        "resumable": bool(st.get("resumable")),
        "files": files,
        "done_runs": done_runs,
        "lines": lines,
        "poll_ms": 2500 if state in PANEL_ACTIVE_STATES else 0,
        "message": str(st.get("user_message") or ""),
    }


class ChatContext:
    """Per-session state. Stored in-memory keyed by session_id."""
    def __init__(self, runs_root: Path, llm_endpoint: str, llm_model: str,
                 session_id: str = ""):
        self.runs_root: Path = runs_root
        self.llm_endpoint: str = llm_endpoint
        self.llm_model: str = llm_model
        self.messages: list[dict] = []        # [{role, content}]
        self.last_sheet_id: Optional[str] = None
        self.last_run_id:   Optional[str] = None
        self.last_sra: Optional[dict] = None
        self.session_id:    str = session_id   # for disk persistence
        self.title:         str = ""           # auto-derived from first user msg


# ============================================================
# Conversation persistence (left-sidebar history)
# ============================================================

CONVERSATIONS_ROOT = Path("/BIO1/MRAG/Agent/orchestrator/webui/conversations")
CONVERSATIONS_ROOT.mkdir(parents=True, exist_ok=True)


def _conv_path(session_id: str) -> Path:
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64]
    if not safe:
        raise ValueError("invalid session_id")
    return CONVERSATIONS_ROOT / f"{safe}.json"


def save_conversation(ctx: ChatContext) -> None:
    if not ctx.session_id or not ctx.messages:
        return
    # Derive title from first non-system user message if not yet set
    if not ctx.title:
        for m in ctx.messages:
            if m.get("role") == "user":
                t = (m.get("content") or "").strip().splitlines()[0]
                ctx.title = (t[:60] + ("…" if len(t) > 60 else "")) or "Untitled"
                break
    payload = {
        "session_id": ctx.session_id,
        "title":      ctx.title or "Untitled",
        "updated":    datetime.now().isoformat(timespec="seconds"),
        "messages":   ctx.messages,
        "last_sheet_id": ctx.last_sheet_id,
        "last_run_id":   ctx.last_run_id,
        # A detached fetch outlives this process.  Without persisting the dest,
        # a UI restart or a reopened conversation makes sra_fetch_status answer
        # "nothing is downloading" while the child is still downloading.
        "last_sra":      ctx.last_sra,
    }
    p = _conv_path(ctx.session_id)
    # Atomic write
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    tmp.replace(p)


def load_conversation_into(ctx: ChatContext, session_id: str) -> bool:
    p = _conv_path(session_id)
    if not p.exists():
        return False
    d = json.loads(p.read_text())
    ctx.session_id    = d.get("session_id", session_id)
    ctx.title         = d.get("title", "")
    ctx.messages      = d.get("messages", [])
    ctx.last_sheet_id = d.get("last_sheet_id")
    ctx.last_run_id   = d.get("last_run_id")
    _ls = d.get("last_sra")
    ctx.last_sra      = dict(_ls) if isinstance(_ls, dict) else None
    return True


def list_conversations() -> list[dict]:
    """List all saved conversations newest first."""
    items = []
    for f in CONVERSATIONS_ROOT.glob("*.json"):
        if f.name.endswith(".tmp"):
            continue
        try:
            d = json.loads(f.read_text())
            # Last user message for preview
            preview = ""
            for m in reversed(d.get("messages", [])):
                if m.get("role") == "user":
                    preview = (m.get("content") or "")[:80]
                    break
            items.append({
                "session_id": d.get("session_id", f.stem),
                "title":      d.get("title", "Untitled"),
                "updated":    d.get("updated", ""),
                "n_msgs":     sum(1 for m in d.get("messages", [])
                                  if m.get("role") in ("user", "assistant")),
                "preview":    preview,
            })
        except Exception:
            continue
    items.sort(key=lambda x: x.get("updated", ""), reverse=True)
    return items


def delete_conversation(session_id: str) -> bool:
    p = _conv_path(session_id)
    if p.exists():
        p.unlink()
        return True
    return False


# ============================================================
# System prompt
# ============================================================

def _build_system_prompt() -> str:
    tools_doc = []
    for name, spec in TOOL_DEFS.items():
        tools_doc.append(f'- "{name}": {spec["description"]}')
        if spec["args_schema"]:
            tools_doc.append("  args:")
            for k, desc in spec["args_schema"].items():
                tools_doc.append(f'    - {k}: {desc}')
    tools_str = "\n".join(tools_doc)

    return f"""You are the MRAG Agent. Help the user run the MRAG pipeline through conversation.

# WHAT MRAG MEANS (canonical, do NOT paraphrase or expand differently)
MRAG = Master Regulator analysis of Accessible Genome.
It identifies master-regulator transcription factors driving condition-specific
gene programs by integrating ATAC-seq (accessible chromatin) and RNA-seq
(differential expression) signals.

# TOOLS YOU CAN CALL
{tools_str}

# STRICT OUTPUT FORMAT
You MUST respond with ONLY a single JSON object, no prose outside it:
{{
  "reply":   "<text shown to the user>",
  "actions": [{{"tool": "<one of: {', '.join(TOOL_DEFS.keys())}>", "args": {{...}}}}, ...],
  "done":    true | false
}}

Rules:
- "actions" may be empty when you just want to converse.
- After you call a tool, the tool result is fed back to you as a system
  message. Reply with a new JSON object incorporating the result.
- "done": true when you have nothing more to do this turn and want to wait
  for the next user message. Use false to chain another tool call.
- NEVER fabricate file paths, sheet_ids, or run_ids. Use what tools return.
- Always call list_data_paths or detect before claiming any data exists.
- SRA/GEO accessions (SRR/ERR/DRR, SRP/ERP/DRP, PRJNA, GSE/GSM): ALWAYS sra_lookup
  first, never sra_fetch first. A single run is never enough for MRAG, so lookup
  expands it to the whole study. If status is not "ok", report the reasons verbatim
  and stop; do not download. If it is "ok", show the sample table, the arms and
  total_download, then ASK permission. Only after an explicit yes, call sra_fetch
  with confirmed=true. After sra_fetch_status reports done, run detect on dest.
  ATAC and RNA are often deposited as SEPARATE studies. Pass EVERY accession the
  user gave in ONE sra_lookup call - one string is fine, e.g. accession
  "SRP245658 SRP245659", or a list. Merging them in a single lookup is what makes
  such a study analysable; looking them up one at a time never can. If `missing`
  names a modality, ask the user for that accession and call sra_lookup once more
  with ALL of them together. Call sra_fetch with the SAME full accession list.
- SUBSETS: RUN accessions (SRR/ERR/DRR) download ONLY those runs; a study
  accession (SRP/ERP/DRP, PRJNA, GSE) downloads the WHOLE study. If the user
  names particular runs, pass exactly those to sra_lookup and then to sra_fetch.
  The lookup reports `selection` with a `summary` like "8 of 16 runs, 37.3 GB of
  60.2 GB" - quote it, and remember the size the user approved is the SELECTED
  size, not the study total. Say "a subset of the study" in those words; never
  describe a study as having only the runs you selected.
- THE DESTINATION IS THE LOOKUP'S `dest`. Never pass a dest of your own to
  sra_fetch: it is the path the user was shown and approved, and a different one
  is refused. Only pass dest (plus dest_confirmed=true) when the USER named that
  exact directory and you quoted the full path back to them.
- THE DOWNLOAD SHOWS ITSELF. The page attaches a live progress panel (overall
  percent, per-file lines, speed, ETA) to the conversation as soon as a fetch
  starts, and it refreshes on its own every few seconds. So when sra_fetch
  returns ok:true state:"started", say ONCE what you started - the runs, the
  size, the directory - and then STOP. Do NOT poll sra_fetch_status in a loop,
  do not narrate percentages, do not promise to keep the user posted, and never
  repeat the panel's numbers in prose. Call sra_fetch_status only when the USER
  asks how it is going, or when you need state=done before detect.
- STOP / RESUME: "stop", "cancel", "abort", "pause" -> sra_cancel. "resume",
  "continue", "restart the download" -> sra_resume. Report what the tool really
  returned: for sra_cancel only `killed: true` means a process was stopped, and
  for sra_resume only ok:true with state:"started" means it is going again.
  sra_resume continues the bytes already on disk and can never start a new
  study; a new study always needs sra_lookup + an explicit yes + sra_fetch.

# NEVER REPORT AN ACTION THAT DID NOT HAPPEN  (IMPORTANT)
- Every tool result carries a top-level `ok`. If `ok` is false, the call FAILED.
  Quote its `user_message` (or `_error`) to the user VERBATIM and ask how to
  proceed. Never describe a failed call as started, running, in progress,
  downloading, or successful, and never call another tool to paper over it.
- You may say a download started ONLY when sra_fetch returned `ok: true` AND
  `state: "started"`. Any other sra_fetch result means NOTHING is downloading -
  say exactly that. `state: "refused"` means the study was rejected before a
  single byte moved.
- sra_fetch_status states are literal, never optimistic:
    not_started - no download was ever launched; nothing is downloading
    starting    - the process is alive but has not written progress yet
    running     - files are being transferred (done/total_files say how far)
    partial     - finished, some files FAILED
    done        - finished; only now may you run detect on dest
    cancelled   - stopped on purpose; NOTHING is running, the partial file was
                  kept and sra_resume can continue it
    refused     - rejected before any transfer
    failed      - the process is gone; report `user_message` and the `log_tail`
  Never translate not_started, refused or failed into "still in progress".
- If the same status result comes back unchanged several times, stop polling and
  tell the user the literal state instead of repeating "still in progress".
- If user asks something unrelated to MRAG, briefly redirect to MRAG topics.
- LANGUAGE: ALWAYS reply in English, whatever language the user writes in.
  Do not switch to Korean or any other language even if asked in one.
  Tool names, file paths, sheet_ids, and run_ids stay in English (in `monospace`).
  Be concise.
- For run_label-less actions, ask for confirmation before starting heavy runs.

# ANSWERING PROGRESS / STATUS QUESTIONS  (IMPORTANT)
- When the user asks how a run is going ("how's it going", "진행 상황", "잘 되고 있어?",
  "status", "어디까지 됐어", "끝났어?"), call `get_status` — NEVER `open_report`.
- Run progress CHANGES OVER TIME. For EVERY progress/status question you MUST call
  `get_status` again in THIS turn. NEVER answer from a `get_status` result that
  appears earlier in the conversation — older results are STALE and the run has
  advanced since then. If an earlier message said the run was 100%/finished but the
  fresh `get_status` says `finished=false`, trust the FRESH result: it is NOT done.
- `open_report` is ONLY for when a run has FINISHED and the user wants the report.
  Do not call it to check progress; a missing report does NOT mean the run failed.
- Peak / locus graphs: when the user asks to SEE a gene's peak/locus/accessibility
  track, call `make_peak_graph` (viewing only — it shows in the plot pane, does NOT
  touch the report). When they want it INCLUDED in the report, call
  `add_peak_to_report`; to drop one, `remove_peak_from_report`; to see what is
  included, `list_report_peaks`. Any gene symbol works (region is resolved from the
  run's differential peaks, falling back to the refGene gene body).
- For a peak/locus request you MUST call `make_peak_graph` immediately. Do NOT
  refuse or defer based on run progress, percent, or whether the report exists —
  peak graphs are generated independently as soon as the run has peaks/bigwigs.
  If the tool returns `_error`, relay that error; otherwise confirm it is shown in
  the plot pane. Never answer a peak request with a status/percent explanation.
- CRITICAL: if your `reply` says you are generating / showing / creating / adding /
  removing something, you MUST include the matching tool call in the `actions` list
  of the SAME JSON response and set `done` to false. Never promise an action in
  `reply` with an empty `actions` list — saying "Generating SORD peak..." without a
  `make_peak_graph` action does nothing. Emit the action, then report the result.
- Interpret `get_status` fields literally and report them:
  - `finished=false` + `alive=true`  => the run is STILL RUNNING (this is normal).
  - `percent`        => overall completion (e.g. 23%).
  - `current_stage` + `current_detail` => what is happening right now
    (e.g. "HOMER Motif Scan", or "ATAC-seq Accessibility — aligning (1/4 BAM)").
  - `stages_done`/`stages_total` => e.g. 3 of 13 stages done.
- Heavy stages (ATAC alignment, HOMER motif scan, narrative) take a long time and
  show 0%/partial progress for a while — that is EXPECTED, not a stall or failure.
- Give a concise, accurate one-line summary, e.g.:
  "Running — HOMER Motif Scan (stage 4/13, ~23%). ATAC, RNA and ranking are done."
- Only say a run "finished"/"completed" when `finished=true`. Never infer completion
  from a missing report or from manifest status alone.

# PROCESSED / NON-FASTQ INPUTS (flex input)
- `detect` works on processed data too, not only FASTQ. It recognizes ATAC/RNA
  count or TPM matrices, log2-normalized matrices, peak BEDs and DESeq2 result
  CSVs, and auto-derives sample<->group labels from a metadata sheet.
- When a detect result has `processed_inputs=true` (usually `n_samples=0` with a
  populated `precomputed_suggestion` and 2 synthesized conditions), the data IS
  compatible: MRAG will start at the differential/combined step (no FASTQ needed).
  Tell the user which processed artifacts were found (from `precomputed_suggestion`)
  and which two conditions were inferred, and ASK them to verify the
  treatment-vs-control order before you `start_run`.
- To run processed data, just call `start_run` with the `sheet_id` — the detected
  `precomputed_suggestion` (matrices + group TSVs) flows through automatically; you
  do NOT need to pass `precomputed` yourself unless the user overrides a path.
- NEVER tell the user "no FASTQ found, cannot run" when `precomputed_suggestion`
  is non-empty. Processed matrices are a fully supported entry point.
- GROUPS WITHOUT A METADATA SHEET: detect resolves treatment/control groups even
  when no sample<->group sheet is uploaded -- first from the column-NAME signal
  (e.g. OE vs control, KO vs WT, treated vs untreated), and if the names are
  opaque IDs, by de-novo k=2 clustering on the matrix. If a `high` warning says
  the groups are de-novo/clustered, tell the user the grouping is data-driven and
  ASK them to confirm it is biologically meaningful (and the treatment/control
  direction) before you start the run.

# BIOLOGICAL DOMAIN (ASK BEFORE EVERY RUN)
- Before you call `start_run`, you MUST know the biological domain of the data so
  the LLM analysis, literature grounding and AlphaGenome tissue are correct (the
  pipeline otherwise defaults to prostate / PC-3, which is WRONG for other cancers).
- ASK the user, in one message: (1) which cancer / disease is this? (2) which
  tissue? (3) which cell line, if any (e.g. PC-3, MCF-7)? Accept partial answers —
  cancer type alone is enough to proceed; cell line is optional.
- Also ask which literature domain(s) the evidence should come from. Several can be
  combined: PC (prostate cancer), NEC (neuroendocrine carcinoma, incl. NEPC),
  SCLC (small cell lung cancer), or any other cancer. Pass them as
  `literature_domains` (e.g. "PC, NEC, SCLC"); if the user names none, omit it and
  the cancer_type is used.
- Pass what you learn to `start_run` as `cancer_type`, `tissue`, `cell_line`. Do NOT
  invent these; only pass values the user actually gave. If the user already stated
  the domain earlier in the conversation, reuse it and do not re-ask.
- If the user declines or says "just use the default", proceed without the args
  (prostate/PC-3 fallback) but tell them the analysis will assume prostate.

# BIGWIG REQUIRED FOR PROCESSED ATAC DATA
- For processed (matrix) ATAC input, per-sample bigWig coverage files are REQUIRED
  to run the analysis (they drive the peak heatmap, per-TF signal profiles and
  locus snapshots — the ATAC signal-level sections). A count matrix alone is not
  enough.
- The detect result reports bigWig status explicitly. When bigWigs are present,
  `precomputed_suggestion.bigwig_count` (> 0) and `bigwig_dir` are set and a `low`
  info warning says "N per-sample bigWig coverage files found ... No bigWig upload
  needed." In that case bigWigs are ALREADY on disk — DO NOT ask the user about
  bigWigs and DO NOT mention uploading them; just proceed.
- ONLY when a `high` warning explicitly says NO per-sample bigWig files were found
  (and `bigwig_count` is absent/0) should you refrain from `start_run` and tell the
  user to upload the per-sample bigWig files (the ⬆ upload panel has a dedicated
  bigWig slot). Never ask about bigWigs speculatively — rely on the detect result.

# DEFAULTS
- LLM-assist for detect: false unless user requests it or previous detect was ambiguous.
- canonical_N for runs: 6 unless user specifies otherwise.
- Default FASTQ root if user says "the usual data" or similar: /BIO1/MRAG/MRAG_data
- `detect` scans RECURSIVELY into subdirectories. /BIO1/MRAG/MRAG_data picks up both ATAC and RNA at once. Do not split into separate calls.
"""


# ============================================================
# LLM call
# ============================================================

_REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "actions": {"type": "array", "items": {
            "type": "object",
            "properties": {"tool": {"type": "string"}, "args": {"type": "object"}},
            "required": ["tool", "args"]}},
        "done": {"type": "boolean"},
    },
    "required": ["reply", "actions", "done"],
}


def _llm_call(messages: list[dict], endpoint: str, model: str,
              timeout: float = 240.0) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False, "think": False,
        # no num_ctx: the chat server's own context length applies (a different one would reload
        # the model on a server it shares with the target scoring)
        "options": {"temperature": 0.0, "num_predict": 800, "repeat_penalty": 1.1},
        # Ollama structured output: the reply object of the STRICT OUTPUT FORMAT in the system prompt
        # (qwen3.8 follows the prompt's format less reliably than JSON mode alone needs)
        "format": _REPLY_SCHEMA,
    }
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body.get("message", {}).get("content", "")


# The chat assistant runs on the dedicated target-scorer server (qwen3.8 on 127.0.0.1:11501, shared
# with the curated step's scoring, one request at a time); `mrag detect --llm-assist` keeps the run
# LLM of ChatContext.llm_endpoint / llm_model.  MRAG_CHAT_ENDPOINT / MRAG_CHAT_MODEL override it.
CHAT_ENDPOINT = os.environ.get("MRAG_CHAT_ENDPOINT", "http://127.0.0.1:11501")
CHAT_MODEL = os.environ.get("MRAG_CHAT_MODEL", "qwen3.8:27b-q8_0")
# Started on demand when the chat finds that server down (it idles out between runs); only for the
# default endpoint.  MRAG_CHAT_ENSURE_CMD=off disables it.
CHAT_ENSURE_CMD = os.environ.get("MRAG_CHAT_ENSURE_CMD", "/BIO1/MRAG/scripts/ensure_scorer_ollama.sh")


def _chat_llm(messages: list[dict]) -> str:
    """One chat completion; when the default chat server refuses the connection, start it with the
    ensure script and retry once."""
    import urllib.error
    try:
        return _llm_call(messages, CHAT_ENDPOINT, CHAT_MODEL)
    except (urllib.error.URLError, ConnectionError) as ex:
        reason = getattr(ex, "reason", ex)
        if (not isinstance(reason, ConnectionRefusedError)
                or CHAT_ENSURE_CMD.strip().lower() in ("", "off", "0", "none")
                or CHAT_ENDPOINT.rstrip("/") != "http://127.0.0.1:11501"):
            raise
        pr = subprocess.run(shlex.split(CHAT_ENSURE_CMD), capture_output=True, text=True, timeout=600)
        if pr.returncode != 0:
            tail = ((pr.stderr or "") + (pr.stdout or "")).strip().splitlines()[-1:] or [""]
            raise RuntimeError("chat server %s is down and %s exited %d: %s"
                               % (CHAT_ENDPOINT, CHAT_ENSURE_CMD, pr.returncode, tail[0][-300:]))
        return _llm_call(messages, CHAT_ENDPOINT, CHAT_MODEL)


def _parse_llm_json(raw: str) -> dict:
    """Try to parse strict JSON. Strip code fences and rescue malformed first/last braces."""
    s = raw.strip()
    # strip ```json ... ```
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # find first '{' and last '}'
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"reply": f"(LLM returned non-JSON; raw: {raw[:300]})",
            "actions": [], "done": True, "_parse_error": True}


# ============================================================
# Turn loop
# ============================================================

MAX_TOOL_TURNS_PER_USER_MSG = 6   # safety: prevent runaway tool loops
# 2026-09-23: a status poll that returns the SAME result over and over is not
# progress, it is a stuck loop. On 2026-09-18 the model spent two whole sessions
# re-reading an identical "may still be starting" and telling the user the
# download was running while nothing had ever been launched.
IDENTICAL_POLL_LIMIT = 3


def _stall_reply(stalled: dict) -> str:
    """Deterministic, truthful reply when a tool keeps returning the same thing."""
    res = stalled.get("result") or {}
    body = (res.get("user_message") or res.get("_error") or res.get("_hint")
            or json.dumps(res, default=str)[:600])
    head = "`%s` returned the same result %d times in a row" % (
        stalled.get("tool", "the tool"), stalled.get("count", 0))
    state = res.get("state")
    if state:
        head += " (state = `%s`)" % state
    return ("%s, so I stopped polling instead of repeating that it is still in "
            "progress.\n\n%s\n\nTell me how you would like to proceed."
            % (head, body))

# Uppercase tokens that look like gene symbols but are pipeline/jargon words.
_GENE_STOPWORDS = {
    "MRAG", "HOMER", "ATAC", "RNA", "TRN", "AG", "TF", "TFS", "GIGGLE", "DDK",
    "OE", "HTML", "OC2", "DNA", "SNP", "UP", "DOWN", "PCA", "ID", "URL", "CSV",
    "PNG", "PDF", "BAM", "BED", "GTF", "QC", "CHIP", "SEQ", "DEG", "FC", "FDR",
    "PI", "OK", "NO", "YES", "TSS", "BW", "HPC", "API", "LLM", "SORD2",
}
_GENE_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,7}(?:-[0-9]+)?\b")
_RUNID_RE = re.compile(r"run_[0-9A-Za-z_]+")


def _extract_gene(msg: str) -> Optional[str]:
    """Pull exactly one gene-symbol-looking uppercase token from the message.
    Case-sensitive: users type gene names in caps, lowercase words are ignored.
    Returns None if zero or >1 plausible genes (ambiguous -> let the LLM decide)."""
    cleaned = _RUNID_RE.sub(" ", msg)
    cands = [t for t in _GENE_RE.findall(cleaned) if t not in _GENE_STOPWORDS]
    # de-dup preserving order
    seen = []
    for t in cands:
        if t not in seen:
            seen.append(t)
    return seen[0] if len(seen) == 1 else None


def _peak_router(ctx: "ChatContext", msg: str) -> Optional[dict]:
    """Deterministic fast-path for peak/locus requests, bypassing the LLM (which
    otherwise parrots stale 'still running' replies from a contaminated history).
    Returns a turn-result dict if it handled the message, else None."""
    low = msg.lower()
    is_peak = any(k in low for k in ("peak", "locus", "track", "accessibility",
                                     "peak graph", "peak plot"))
    if not is_peak:
        return None
    gene = _extract_gene(msg)
    if not gene:
        return None  # ambiguous -> fall through to LLM
    run_id = ctx.last_run_id
    m = _RUNID_RE.search(msg)
    if m:
        run_id = m.group(0)
    if not run_id:
        return None
    remove = any(k in low for k in ("remove", "exclude", "drop", "빼", "제거", "삭제"))
    add = any(k in low for k in ("add", "include", "넣", "추가",
                                 "to report", "to the report", "보고서"))
    if remove:
        res = _t_remove_peak_from_report({"run_id": run_id, "gene": gene}, ctx)
        tool = "remove_peak_from_report"
        reply = (f"보고서 Section 12에서 {gene} peak를 제거했어요."
                 if not res.get("_error") else f"제거 실패: {res['_error']}")
    elif add:
        res = _t_add_peak_to_report({"run_id": run_id, "gene": gene}, ctx)
        tool = "add_peak_to_report"
        reply = (f"{gene} peak graph를 생성해 보고서 Section 12에 추가했어요 "
                 f"(region {res.get('region','')})."
                 if not res.get("_error") else f"추가 실패: {res['_error']}")
    else:
        res = _t_make_peak_graph({"run_id": run_id, "gene": gene}, ctx)
        tool = "make_peak_graph"
        if res.get("_error"):
            reply = f"{gene} peak graph 생성 실패: {res['_error']}"
        else:
            reply = (f"{gene}의 ATAC peak graph를 생성했어요 "
                     f"(region {res.get('region','')}, {res.get('region_source','')}). "
                     f"오른쪽 plot pane의 'Section 12 — Locus (custom)'에서 볼 수 있어요. "
                     f"보고서에 넣고 싶으면 '{gene} 보고서에 추가해줘'라고 말해주세요.")
    log = [{"tool": tool, "args": {"run_id": run_id, "gene": gene}, "result": res}]
    ctx.messages.append({"role": "user", "content": msg})
    ctx.messages.append({"role": "assistant",
                         "content": json.dumps({"reply": reply, "actions": [],
                                                "done": True}, ensure_ascii=False)})
    try:
        save_conversation(ctx)
    except Exception:
        pass
    return {"reply": reply, "tool_calls": log}


def chat_turn(ctx: ChatContext, user_message: str) -> dict:
    """One full turn: user message in, possibly multiple tool calls, final reply out.

    Returns: {"reply": str, "tool_calls": list[{tool, args, result}]}
    """
    if not ctx.messages:
        ctx.messages.append({"role": "system", "content": _build_system_prompt()})
    # Deterministic peak/locus fast-path (robust against contaminated history)
    routed = _peak_router(ctx, user_message)
    if routed is not None:
        return routed
    # index of THIS turn's user message — TOOL_RESULTS before this are stale
    user_idx = len(ctx.messages)
    ctx.messages.append({"role": "user", "content": user_message})

    dispatcher = _build_dispatcher(ctx)
    tool_call_log: list[dict] = []
    final_reply = "(no reply)"
    poll_seen: dict[str, int] = {}     # identical (tool, args, result) fingerprints

    for turn in range(MAX_TOOL_TURNS_PER_USER_MSG):
        # Strip stale TOOL_RESULTS from PRIOR turns: they are point-in-time
        # facts (run percent, status, etc.) that go stale and mislead the LLM
        # into re-stating old numbers. Keep only the current turn's results so
        # multi-step tool reasoning still works.
        send_msgs = [m for i, m in enumerate(ctx.messages)
                     if not (i < user_idx and m.get("role") == "system"
                             and str(m.get("content", "")).startswith("TOOL_RESULTS:"))]
        raw = _chat_llm(send_msgs)
        parsed = _parse_llm_json(raw)
        ctx.messages.append({"role": "assistant", "content": raw})

        final_reply = parsed.get("reply", "") or final_reply
        actions = parsed.get("actions") or []
        done = parsed.get("done", True)

        # Dispatch actions
        results = []
        had_unknown = False
        stalled: Optional[dict] = None
        for a in actions:
            tool = a.get("tool", "")
            args = a.get("args", {}) or {}
            # tolerate hallucinated tool names + arg names
            tool = TOOL_ALIASES.get(tool, tool)
            if tool in ("make_peak_graph", "add_peak_to_report",
                        "remove_peak_from_report") and "gene" not in args:
                for k in ("locus", "target", "gene_name", "gene_symbol", "name"):
                    if args.get(k):
                        args["gene"] = args[k]
                        break
            if tool not in dispatcher:
                valid = ", ".join(dispatcher.keys())
                results.append({"tool": tool, "args": args, "result": {
                    "_error": f"unknown tool '{tool}'. Call one of: {valid}. "
                              f"For a gene peak graph use make_peak_graph with "
                              f"arg 'gene'."}})
                had_unknown = True
                continue
            try:
                res = dispatcher[tool](args)
            except Exception as e:
                res = {"_error": f"{type(e).__name__}: {e}"}
            results.append({"tool": tool, "args": args, "result": res})
            tool_call_log.append({"tool": tool, "args": args, "result": res})
            sig = json.dumps([tool, args, res], sort_keys=True, default=str)
            poll_seen[sig] = poll_seen.get(sig, 0) + 1
            if poll_seen[sig] >= IDENTICAL_POLL_LIMIT and stalled is None:
                stalled = {"tool": tool, "args": args, "result": res,
                           "count": poll_seen[sig]}

        # Feed results back to LLM as a system message. A failed call is called
        # out explicitly: the results alone were not enough to stop the model
        # narrating an error as a download in progress.
        if results:
            failed = [r for r in results
                      if r["result"].get("ok") is False or "_error" in r["result"]]
            note = ""
            if failed:
                note = ("\n\nTOOL FAILURE: %d of the calls above did NOT succeed. "
                        "You MUST tell the user which one failed, quote its "
                        "`user_message` (or `_error`) verbatim, and ask how to "
                        "proceed. Do NOT describe a failed call as started, "
                        "running, downloading or successful." % len(failed))
            ctx.messages.append({
                "role": "system",
                "content": "TOOL_RESULTS:\n"
                           + json.dumps(results, indent=2, default=str)[:8000] + note,
            })

        # An unchanging poll result is a stuck loop, not progress: answer with the
        # literal state and stop, rather than letting the model keep re-polling.
        if stalled is not None:
            final_reply = _stall_reply(stalled)
            ctx.messages.append({"role": "assistant", "content": json.dumps(
                {"reply": final_reply, "actions": [], "done": True})})
            break

        # If the model invoked an unknown tool, force another turn so it can
        # self-correct to the real tool name instead of ending on a no-op.
        if had_unknown:
            done = False
        if (done or not actions) and not had_unknown:
            break

    # Clip conversation history if too long
    if len(ctx.messages) > 50:
        # keep system prompt + last 40 messages
        ctx.messages = ctx.messages[:1] + ctx.messages[-40:]

    # Persist conversation to disk after each turn
    try:
        save_conversation(ctx)
    except Exception as _e:
        pass  # persistence is best-effort

    out = {"reply": final_reply, "tool_calls": tool_call_log}
    # Hand the page a directory to watch whenever this turn touched a download.
    # Only the DEST travels, never a snapshot: the panel polls the progress file
    # itself, so it keeps moving between turns and survives a page reload.
    for tc in reversed(tool_call_log):
        if tc.get("tool") in ("sra_fetch", "sra_resume", "sra_cancel",
                              "sra_fetch_status"):
            dest = str((tc.get("result") or {}).get("dest") or "").strip()
            if dest:
                out["sra_dest"] = dest
                break
    return out
