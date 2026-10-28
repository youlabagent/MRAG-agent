"""MRAG Agent — Web UI (Phase 3a)

FastAPI wrapper around the `mrag` CLI. All UI text in English.

Routes:
  GET  /                              landing
  GET  /detect                        FASTQ-dir input form
  POST /api/detect                    runs `mrag detect`, returns sample sheet
  GET  /confirm/<sheet_id>            review form for a sample sheet
  POST /api/confirm                   builds run.toml + spawns `mrag run`
  GET  /run/<run_id>                  live status page (HTMX polling)
  GET  /api/status/<run_id>           returns manifest snapshot (JSON or HTML partial)
  GET  /api/debug/<run_id>            returns diagnoses JSON
  GET  /report/<run_id>               serves the final HTML report
  GET  /runs                          list of runs
  GET  /api/sra/progress              live SRA download progress (chat panel)
"""
from __future__ import annotations
import io
import json
import os
import re
import shlex
import subprocess
import zipfile
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request, Form, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from chat import ChatContext, chat_turn, EXTRA_DATA_ROOT
from chat import AXIS_HEADLINE_CHOICES, axis_declaration_suggestion, write_axis_declaration


# ============================================================
# Configuration
# ============================================================

ROOT          = Path(__file__).parent
RUNS_ROOT     = Path(os.environ.get("MRAG_WEBUI_RUNS_ROOT", ROOT / "runs"))
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
# Processed-data uploads land in a fresh run_<ts>/ directory under MRAG_data
# (an allowed detect root) with RNA/ ATAC/ meta/ subdirs, so the chat can
# immediately `detect` on them. Reused per chat session so RNA and ATAC
# uploaded in separate requests land in the SAME run dir.
UPLOAD_ROOT   = Path("/BIO1/MRAG/MRAG_data")
_UPLOAD_DIRS: dict[str, str] = {}
MRAG_BIN      = os.environ.get("MRAG_BIN", "/BIO1/MRAG/venvs/paper_qa/bin/mrag")
QWEN_ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://127.0.0.1:11500")
QWEN_MODEL    = os.environ.get("MRAG_MODEL", "batiai/qwen3.6-27b:q6")

# ============================================================
# Plot registry (canonical 2-pane plot pane)
#   ordered whitelist of relative output paths -> (title, type)
#   type in {"image","plotly","cytoscape"}
# ============================================================
PLOT_REGISTRY: list[dict] = [
    # ordered by MRAG report / pipeline section number
    {"path": "viz_built/tf_table.csv",            "title": "3. TF Ranking (Top 10, Avg.Z)",  "type": "table", "limit": 10},
    {"path": "viz_built/volcano_atac.json",       "title": "4. Volcano — ATAC",            "type": "plotly"},
    {"path": "viz_built/volcano_rna.json",        "title": "4. Volcano — RNA",             "type": "plotly"},
    {"path": "#tf-rank", "anchor": "tf-rank", "title": "5. Rank & Combined Score (2 panels)", "type": "section"},
    {"path": "peakheatmap/peakheatmap.png",       "title": "6. Peak Heatmap (Specific/Shared)", "type": "image"},
    {"path": "#heatmap", "anchor": "heatmap", "title": "7. Gene Expression Heatmap (2 panels)", "type": "section"},
    {"path": "#homer-known", "anchor": "homer-known", "title": "8. HOMER Known Motif Enrichment", "type": "section"},
    # manifest keys giggle_up_png / giggle_down_png: the step writes treatment_/control_giggle.png,
    # earlier runs wrote DiffBind_up/down.png
    {"glob": ["giggle/*up.png", "giggle/treatment_giggle.png"],
     "title": "9. GIGGLE — Up", "type": "image"},
    {"glob": ["giggle/*down.png", "giggle/control_giggle.png"],
     "title": "9. GIGGLE — Down", "type": "image"},
    # The Super Enhancer -> TF -> Target graph is the run's TRN: Section 10b as the REPORT draws it
    # (columns layout, colour scales, legend, controls) rather than a generic spring layout of
    # se_cyjs.json.
    {"path": "#trn", "anchor": "trn", "title": "10. TRN Network (Super Enhancer → TF → Target)", "type": "section"},
    {"glob": "se_rose/rose/*_Plot_points.png",     "title": "10. ROSE Super Enhancer ranking", "type": "image"},
    # Section 11 as the REPORT shows it (both panels, captions, notices), not the raw enrichment/
    # JSONs: a report whose Section 11 was rebuilt later carries figures the run's own
    # enrichment/*.json files do not.
    {"path": "#enrichment", "anchor": "enrichment", "title": "11. Functional Enrichment (2 panels)", "type": "section"},
    # Section 13 (In silico Validation) as the REPORT draws it: one entry, opened on the run's
    # leading model set, with the Both / AlphaGenome / Corgi tabs inside the section.  The section
    # draws the supported enhancer -> TF step only.
    {"path": "#ag-validation", "anchor": "ag-validation",
     "title": "13. In silico Validation (Both / AlphaGenome / Corgi)", "type": "section"},
]
# dynamic glob groups appended at request time (section 12 locus snapshots)
PLOT_GLOBS: list[dict] = [
    {"glob": "locus_snapshots/*.png", "title_prefix": "12. Locus", "type": "image"},
    {"glob": "user_locus/mrag_*.png",   "title_prefix": "12. Locus (custom)", "type": "image"},
]

# ── Pipeline progress: user-facing stages mapped onto internal manifest steps ──
PIPELINE_STAGES: list[dict] = [
    {"key": "atac",       "label": "ATAC-seq Accessibility",      "icon": "🧬",
     "steps": ["atac", "dastk"]},
    {"key": "rna",        "label": "RNA-seq Differential",        "icon": "📊",
     "steps": ["rna", "deseq2"]},
    {"key": "combined",   "label": "Master-Regulator Ranking",    "icon": "🏅",
     "steps": ["combined"]},
    {"key": "homer",      "label": "HOMER Motif Scan",            "icon": "🔍",
     "steps": ["motif_scan_minmax_seeded", "motif_scan_minmax_barrier",
               "filter_target_tfs", "motif_scan_chain",
               "homer_known_motifs", "homer_filter_top6"]},
    {"key": "join",       "label": "DiffBind Target Join",        "icon": "🔗",
     "steps": ["join_diffbind_rna"]},
    {"key": "giggle",     "label": "GIGGLE Cistrome Overlap",     "icon": "🧩",
     "steps": ["giggle"]},
    {"key": "heatmap",    "label": "Peak Heatmap",                "icon": "🔥",
     "steps": ["peakheatmap"]},
    {"key": "se",         "label": "Super Enhancers (ROSE)",       "icon": "⭐",
     "steps": ["se_rose", "se_motif", "se_layer"]},
    {"key": "curated",    "label": "Target Gene Scoring",         "icon": "🎯",
     "steps": ["curated_targets"]},
    {"key": "trn",        "label": "TRN Network",                 "icon": "🕸️",
     "steps": ["trn_top6", "trn_graph_build", "trn_ranked_build",
               "trn_cytoscape", "se_trn"]},
    {"key": "enrichment", "label": "Pathway Enrichment",          "icon": "🧪",
     "steps": ["enrichment"]},
    {"key": "axis",       "label": "SE-TF-Target Axes",          "icon": "🧠",
     "steps": ["axis_validation"]},
    {"key": "narrative",  "label": "Literature Narrative",        "icon": "📖",
     "steps": ["narrative_top6"]},
    {"key": "viz",        "label": "Interactive Figures",         "icon": "📈",
     "steps": ["tf_rank_build", "tf_table_build", "volcano_atac_build",
               "volcano_rna_build", "per_tf_profile_build", "locus_snapshots",
               "rank_plot", "volcano_plot",
               "interactive_rank_plot",
               "interactive_volcano_plot", "heatmap_plot",
               "interactive_heatmap_plot"]},
    {"key": "render",     "label": "Final Report",                "icon": "📄",
     "steps": ["render"]},
]
# flat canonical order for positional progress
_STAGE_CANON: list[str] = [n for st in PIPELINE_STAGES for n in st["steps"]]
_DONE_STATUS = {"ok", "provided", "skipped", "seeded", "cache-hit"}
# user-facing phase grouping for the 2-column progress view
PHASE_ORDER: list[str] = [
    "Alignment & Quantification", "Combined Ranking", "Motif & Targets",
    "Networks & Validation", "Plotting & Report",
]
_STAGE_PHASE: dict[str, str] = {
    "atac": "Alignment & Quantification", "rna": "Alignment & Quantification",
    "combined": "Combined Ranking",
    "homer": "Motif & Targets", "join": "Motif & Targets", "giggle": "Motif & Targets",
    "curated": "Networks & Validation",
    "trn": "Networks & Validation", "enrichment": "Networks & Validation",
    "axis": "Networks & Validation",
    "heatmap": "Plotting & Report", "narrative": "Plotting & Report",
    "viz": "Plotting & Report", "render": "Plotting & Report",
}

# section-type plots have no standalone file: they live inline in mrag_report.html.
# we extract just that <section> (+ its plotly init scripts) into a standalone mini-page.
SECTION_PLOT_DIVS: dict[str, list[str]] = {
    "tf-rank": ["rank-plotly", "volcano-plotly"],
    "heatmap": ["heatmap-median-plotly", "heatmap-plotly"],
    "homer-known": [],
    # Section 10b carries its own inline Cytoscape script inside the <section>; the page only
    # needs the library (see run_section).
    "trn": [],
    # Section 13: its graph script sits AFTER the section and reuses Section 10b's graph
    # (window.cySE), so run_section carries both.
    "ag-validation": [],
    "enrichment": ["shared-dotplot", "per-tf-heatmap"],
}



app = FastAPI(title="MRAG Agent", version="0.1.0")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
templates.env.cache = None

# In-memory chat sessions (lost on restart)
_CHAT_SESSIONS: dict[str, ChatContext] = {}
    # bypass jinja2 3.1.6 cache-key bug with starlette 0.52


# ============================================================
# Helpers
# ============================================================

# A run or sheet id names a directory directly under RUNS_ROOT ("run_<ts>_<hex>", "sheet_<id>"):
# no path separator and no leading dot, so "." / ".." never reach a parent directory.
_RUN_ID_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
# A chat session id is written into the chat page's history list: a plain token only.
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _run_dir(run_id: str) -> Path:
    """RUNS_ROOT/<run_id>; 400 unless run_id is a plain directory name whose resolved path is a
    direct child of RUNS_ROOT."""
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise HTTPException(400, "invalid run_id")
    d = RUNS_ROOT / run_id
    if d.resolve().parent != RUNS_ROOT.resolve():
        raise HTTPException(400, "invalid run_id")
    return d


def _run_mrag(args: list[str], timeout: int = 600) -> dict:
    """Invoke `mrag` synchronously and return parsed stdout as dict.
    Raises HTTPException on non-zero exit."""
    cmd = [MRAG_BIN] + args
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, detail=f"mrag timed out: {' '.join(shlex.quote(a) for a in cmd)}")
    if proc.returncode not in (0, 3):  # 3 = needs_user_action (detect)
        raise HTTPException(500, detail=f"mrag failed (rc={proc.returncode}): {proc.stderr[:500]}")
    out = (proc.stdout or "").strip()
    if not out:
        return {}
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"_raw": out}


def _spawn_mrag_run(config_path: Path, log_path: Path) -> int:
    """Start `mrag run --config` as detached subprocess. Returns PID."""
    fh = open(log_path, "w")
    proc = subprocess.Popen(
        [MRAG_BIN, "run", "--config", str(config_path)],
        stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
    )
    return proc.pid


def _list_runs(include_drafts: bool = False) -> list[dict]:
    """List actual runs (run_*). When include_drafts=True, also include sheet_*."""
    runs = []
    for d in sorted(RUNS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir(): continue
        is_draft = d.name.startswith("sheet_")
        if is_draft and not include_drafts:
            continue
        manifest_path = d / "output" / "manifest.json"
        info = {
            "run_id":  d.name,
            "created": datetime.fromtimestamp(d.stat().st_mtime).isoformat(timespec="seconds"),
            "is_draft": is_draft,
        }
        if manifest_path.exists():
            try:
                m = json.loads(manifest_path.read_text())
                info["status"]     = m.get("status", "?")
                info["n_steps"]    = len(m.get("steps", []))
                info["final_html"] = m.get("summary", {}).get("final_html", "")
            except Exception:
                info["status"] = "manifest-unreadable"
        else:
            info["status"] = "draft" if is_draft else "pending"
        runs.append(info)
    return runs


# ============================================================
# Routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"runs": _list_runs()[:10]},
    )


@app.get("/detect", response_class=HTMLResponse)
def detect_form(request: Request):
    return templates.TemplateResponse("detect.html", {"request": request, "sheet": None})


@app.post("/api/detect", response_class=HTMLResponse)
def detect_post(request: Request,
                fastq_dir: str = Form(...),
                llm_assist: str = Form("")):
    sheet_id = uuid.uuid4().hex[:10]
    run_dir = RUNS_ROOT / f"sheet_{sheet_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = run_dir / "sample_sheet.json"

    args = ["detect", "--fastq-dir", fastq_dir, "--output", str(sheet_path)]
    if llm_assist == "on":
        args += ["--llm-assist", "--endpoint", QWEN_ENDPOINT, "--model", QWEN_MODEL]
    result = _run_mrag(args, timeout=600)

    if not sheet_path.exists():
        raise HTTPException(500, "detect produced no sample sheet")
    sheet = json.loads(sheet_path.read_text())
    sheet["_sheet_id"] = sheet_id

    return templates.TemplateResponse(
        "detect.html",
        {"request": request, "sheet": sheet, "result": result,
         "fastq_dir": fastq_dir, "llm_used": (llm_assist == "on")},
    )


@app.get("/confirm/{sheet_id}", response_class=HTMLResponse)
def confirm_form(request: Request, sheet_id: str):
    sheet_path = _run_dir(f"sheet_{sheet_id}") / "sample_sheet.json"
    if not sheet_path.exists():
        raise HTTPException(404, f"sheet_id {sheet_id} not found")
    sheet = json.loads(sheet_path.read_text())
    # Section 13's declaration fields are pre-selected from what the sheet says about the disease
    # (a suggestion only; the form's choice is what goes into run.toml)
    axis_sug = axis_declaration_suggestion(
        sheet.get("cancer_type"), sheet.get("literature_domains"), sheet.get("tissue"),
        sheet.get("cohort"), sheet.get("summary_human"))
    return templates.TemplateResponse(
        "confirm.html",
        {"request": request, "sheet": sheet, "sheet_id": sheet_id, "axis_sug": axis_sug},
    )


@app.post("/api/confirm", response_class=HTMLResponse)
def confirm_post(request: Request,
                 sheet_id: str = Form(...),
                 canonical_N: int = Form(10),
                 organism: str = Form(""),
                 treatment_arm: str = Form(""),
                 integrated_csv: str = Form(""),
                 atac_md_csv: str = Form(""),
                 deseq_csv: str = Form(""),
                 diffbind_out_csv: str = Form(""),
                 motif_tsv: str = Form(""),
                 tf_table_csv: str = Form(""),
                 # Section 13: which model set leads it, and the AR gate row; written to
                 # run.toml's top level
                 axis_headline: str = Form("auto"),
                 axis_ar_gate: str = Form("")):
    if (axis_headline or "auto").strip().lower() not in AXIS_HEADLINE_CHOICES:
        raise HTTPException(400, "axis_headline must be one of " + "/".join(AXIS_HEADLINE_CHOICES))
    sheet_dir = _run_dir(f"sheet_{sheet_id}")
    sheet_path = sheet_dir / "sample_sheet.json"
    if not sheet_path.exists():
        raise HTTPException(404, f"sheet_id {sheet_id} not found")

    run_id = f"run_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run.toml"
    output_dir = run_dir / "output"

    # Build --precomputed key=val list
    pre = {
        "integrated_csv":   integrated_csv,
        "atac_md_csv":      atac_md_csv,
        "deseq_csv":        deseq_csv,
        "diffbind_out_csv": diffbind_out_csv,
        "motif_tsv":        motif_tsv,
        "tf_table_csv":     tf_table_csv,
    }
    pre_kvs = [f"{k}={v}" for k, v in pre.items() if v.strip()]

    args = ["confirm",
            "--sheet", str(sheet_path),
            "--output-config", str(config_path),
            "--output-dir", str(output_dir),
            "--run-label", run_id,
            "--canonical-N", str(canonical_N)]
    # Species/genome for the whole run. Empty => `mrag confirm` inherits the
    # sheet's detected organism (and the runner detects/defaults if neither).
    if (organism or "").strip():
        args += ["--organism", organism.strip()]
    # Which arm is the TREATMENT arm. conditions[0] drives every fold sign,
    # the red/blue track colours, the GIGGLE labels and the arm ROSE calls
    # super-enhancers on; without this the detector's filename order decided
    # it silently (2026-09-23).
    if (treatment_arm or "").strip():
        args += ["--treatment-arm", treatment_arm.strip()]
    if pre_kvs:
        args += ["--precomputed"] + pre_kvs
    _run_mrag(args, timeout=60)
    if config_path.exists():
        try:
            write_axis_declaration(config_path, axis_headline, axis_ar_gate)
        except ValueError as _e:
            raise HTTPException(500, f"Section-13 declaration not written to run.toml: {_e}")

    # Spawn `mrag run` in background
    log_path = run_dir / "run.log"
    pid = _spawn_mrag_run(config_path, log_path)
    (run_dir / "pid").write_text(str(pid))

    return RedirectResponse(url=f"/run/{run_id}", status_code=303)


@app.get("/run/{run_id}", response_class=HTMLResponse)
def run_status_page(request: Request, run_id: str):
    run_dir = _run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(404, f"run_id {run_id} not found")
    return templates.TemplateResponse(
        "run_status.html",
        {"request": request, "run_id": run_id},
    )


@app.get("/api/status/{run_id}", response_class=HTMLResponse)
def status_partial(request: Request, run_id: str):
    """HTMX partial — polled every ~1s by the run status page."""
    run_dir   = _run_dir(run_id)
    manifest_path = run_dir / "output" / "manifest.json"
    pid_file  = run_dir / "pid"
    pid_alive = False
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)  # signal 0 = check
            pid_alive = True
        except (OSError, ValueError):
            pid_alive = False
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = {"_error": "manifest unreadable"}
    return templates.TemplateResponse(
        "run_status_partial.html",
        {"request": request, "run_id": run_id, "manifest": manifest,
         "pid_alive": pid_alive, "now": datetime.now().isoformat(timespec="seconds")},
    )


@app.get("/api/debug/{run_id}", response_class=JSONResponse)
def debug_json(run_id: str):
    manifest_path = _run_dir(run_id) / "output" / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, "no manifest yet")
    result = _run_mrag(["debug", str(manifest_path)], timeout=20)
    return JSONResponse(result)


@app.get("/report/{run_id}", response_class=HTMLResponse)
def serve_report(run_id: str):
    html_path = _run_dir(run_id) / "output" / "mrag_report.html"
    if not html_path.exists():
        raise HTTPException(404, "report not generated yet")
    return FileResponse(html_path, media_type="text/html", headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"})


@app.get("/report/{run_id}/download")
def download_report(run_id: str):
    html_path = _run_dir(run_id) / "output" / "mrag_report.html"
    if not html_path.exists():
        raise HTTPException(404, "report not generated yet")
    # Same no-store policy as the view route: a report is regenerated in place
    # (re-render, resume), so a cached download hands the user a stale file that
    # looks identical to the live page they just checked.
    return FileResponse(html_path, media_type="text/html",
                        filename=f"{run_id}_mrag_report.html",
                        headers={"Cache-Control": "no-store, must-revalidate",
                                 "Pragma": "no-cache"})


def _plot_entries(run_id: str) -> list[dict]:
    out_dir = (_run_dir(run_id) / "output").resolve()
    entries: list[dict] = []
    report_fp = out_dir / "mrag_report.html"
    for spec in PLOT_REGISTRY:
        if spec["type"] == "section":
            entries.append({
                "path": spec["path"], "title": spec["title"], "type": "section",
                "ready": report_fp.exists(),
                "url": f"/api/run/{run_id}/section?id={spec['anchor']}",
            })
            continue
        if spec.get("glob"):
            # A list of patterns, first match wins: a figure the pipeline renamed is still found
            # for runs of either vintage.
            pats = spec["glob"] if isinstance(spec["glob"], (list, tuple)) else [spec["glob"]]
            matches = next((m for pat in pats for m in sorted(out_dir.glob(pat))), None)
            if matches is not None:
                rel = matches.relative_to(out_dir).as_posix()
                entries.append({
                    "path": rel, "title": spec["title"], "type": spec["type"],
                    "ready": True, "url": f"/api/run/{run_id}/plotfile?path={rel}",
                })
            else:
                entries.append({
                    "path": pats[0], "title": spec["title"], "type": spec["type"],
                    "ready": False, "url": "",
                })
            continue
        fp = (out_dir / spec["path"])
        url = f"/api/run/{run_id}/plotfile?path={spec['path']}"
        if spec.get("limit"):
            url += f"&limit={spec['limit']}"
        entries.append({
            "path": spec["path"], "title": spec["title"], "type": spec["type"],
            "ready": fp.exists(), "url": url,
        })
    for g in PLOT_GLOBS:
        for fp in sorted(out_dir.glob(g["glob"])):
            rel = fp.relative_to(out_dir).as_posix()
            entries.append({
                "path": rel, "title": f"{g['title_prefix']} — {fp.stem}", "type": g["type"],
                "ready": True, "url": f"/api/run/{run_id}/plotfile?path={rel}",
            })
    def _sec(e):
        m = re.match(r"\s*(\d+)", e["title"])
        return int(m.group(1)) if m else 999
    entries.sort(key=_sec)  # stable: keep registry order within a section
    return entries


@app.get("/api/run/{run_id}/plots", response_class=JSONResponse)
def run_plots(run_id: str):
    run_dir = _run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(404, "no such run")
    entries = _plot_entries(run_id)
    report = (run_dir / "output" / "mrag_report.html").exists()
    return JSONResponse({"run_id": run_id, "plots": entries, "report_ready": report})


def _stage_detail(run_dir: Path, key: str) -> tuple[str, float]:
    """Filesystem-based sub-progress for the long ATAC/RNA preprocessing stages.
    Returns (detail_label, fraction 0..1 within the stage)."""
    if key not in ("atac", "rna"):
        return ("", 0.0)
    toml = run_dir / "run.toml"
    if not toml.exists():
        return ("", 0.0)
    txt = toml.read_text(errors="ignore")
    import re as _re
    mod = "ATAC" if key == "atac" else "RNA"
    # all fastq_r1 paths whose sample block is this modality (path contains /ATAC/ or /RNA/)
    paths = _re.findall(r'fastq_r1\s*=\s*"([^"]+)"', txt)
    sep = "/" + mod + "/"
    mine = [x for x in paths if sep in x]
    if not mine:
        return ("", 0.0)
    n = len(mine)
    data_dir = Path(mine[0]).parent
    trim = len(list((data_dir / "trim_galore").glob("*_val_1.fq.gz")))
    bam = len(list((data_dir / "alignments").glob("*.sorted.bam")))
    if key == "atac":
        if trim < n:
            return (f"trimming reads ({trim}/{n})", 0.30 * trim / n)
        if bam < n:
            return (f"aligning ({bam}/{n} BAM)", 0.30 + 0.55 * bam / n)
        return ("peak calling / MD-score", 0.90)
    # rna
    quant = len(list(data_dir.glob("**/*.genes.results"))) + \
            len(list(data_dir.glob("**/ReadsPerGene.out.tab")))
    if trim < n:
        return (f"trimming reads ({trim}/{n})", 0.30 * trim / n)
    if quant < n:
        return (f"aligning / quantifying ({quant}/{n})", 0.30 + 0.55 * quant / n)
    return ("differential expression", 0.90)

@app.get("/api/run/{run_id}/progress", response_class=JSONResponse)

def run_progress(run_id: str):
    """User-facing pipeline progress: ordered stages with done/running/pending/error."""
    run_dir = _run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(404, "no such run")
    manifest_path = run_dir / "output" / "manifest.json"
    report_ready = (run_dir / "output" / "mrag_report.html").exists()

    # is the run process still alive?
    alive = False
    pid_file = run_dir / "pid"
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text().strip()), 0)
            alive = True
        except (OSError, ValueError):
            alive = False

    steps = []
    man_status = ""
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text())
            steps = m.get("steps", []) or []
            man_status = m.get("status", "") or ""
        except Exception:
            steps = []

    # completed step names + any error; positional index of furthest progress
    done_names: set[str] = set()
    error_names: set[str] = set()
    last_idx = -1
    for st in steps:
        if not isinstance(st, dict):
            continue
        nm = st.get("name", "")
        sstat = (st.get("status") or "").lower()
        if sstat in _DONE_STATUS:
            done_names.add(nm)
            error_names.discard(nm)          # a retried step that succeeded is done
            if nm in _STAGE_CANON:
                last_idx = max(last_idx, _STAGE_CANON.index(nm))
        elif sstat in ("error", "failed", "fail"):
            # Node.execute writes 'fail'
            error_names.add(nm)

    # NOTE: manifest top-level status is "ok" even mid-run, so it only counts as
    # finished once the process has actually exited.  A render step in THIS
    # manifest is conclusive; a report file on disk is not, because a resumed
    # run rebuilds the manifest from scratch while the previous run's report is
    # still sitting there -- treating that as "finished" pinned the progress
    # view at 100% for the whole re-run.  So the leftover-file signal only
    # counts once the process is gone.
    finished = ("render" in done_names) or (report_ready and not alive)
    if not finished and not alive and man_status.lower() in ("done", "complete", "ok", "finished"):
        finished = True

    out_stages = []
    n_done = 0
    for st in PIPELINE_STAGES:
        idxs = [_STAGE_CANON.index(n) for n in st["steps"] if n in _STAGE_CANON]
        stage_end = max(idxs) if idxs else -1
        stage_start = min(idxs) if idxs else -1
        has_err = any(n in error_names for n in st["steps"])
        # a failed stage stays "error" after the run finished
        if has_err:
            status = "error"
        elif finished:
            status = "done"
        elif last_idx >= stage_end:
            status = "done"
        elif last_idx >= stage_start:
            status = "running" if alive else "done"
        elif last_idx >= 0 and stage_start == last_idx + 1 and alive:
            status = "running"
        else:
            status = "pending"
        if status == "done":
            n_done += 1
        out_stages.append({"key": st["key"], "label": st["label"],
                           "icon": st["icon"], "status": status, "detail": "",
                           "phase": _STAGE_PHASE.get(st["key"], "Other")})

    # promote the first non-done stage to "running" if the run is alive and none flagged
    if alive and not finished and not any(s["status"] == "running" for s in out_stages):
        for s in out_stages:
            if s["status"] == "pending":
                s["status"] = "running"
                break

    run_frac = 0.0
    for s_ in out_stages:
        if s_["status"] == "running":
            det, frac = _stage_detail(run_dir, s_["key"])
            if det:
                s_["detail"] = det
                run_frac = frac
            break
    total = len(out_stages)
    pct = round(100 * (n_done + run_frac) / total) if total else 0
    cur = next((s["label"] for s in out_stages if s["status"] == "running"), None)
    return JSONResponse({
        "run_id": run_id, "alive": alive, "finished": finished,
        "report_ready": report_ready, "percent": pct,
        "n_done": n_done, "n_total": total,
        "current": cur, "stages": out_stages, "phases": PHASE_ORDER,
    })


@app.get("/api/run/{run_id}/plotfile")
def run_plotfile(run_id: str, path: str, limit: int = 0):
    out_dir = (_run_dir(run_id) / "output").resolve()
    target = (out_dir / path).resolve()
    # path traversal guard
    if not str(target).startswith(str(out_dir) + os.sep):
        raise HTTPException(403, "forbidden")
    # whitelist: registry paths or glob members
    allowed = {s["path"] for s in PLOT_REGISTRY if s.get("path")}
    rel = target.relative_to(out_dir).as_posix() if str(target).startswith(str(out_dir)) else ""
    reg_globs = [gp for s2 in PLOT_REGISTRY if s2.get("glob")
                 for gp in (s2["glob"] if isinstance(s2["glob"], (list, tuple)) else [s2["glob"]])]
    glob_ok = any(target.match(g["glob"]) for g in PLOT_GLOBS) \
              or any(target.match(gp) for gp in reg_globs)
    if rel not in allowed and not glob_ok:
        raise HTTPException(403, "not a whitelisted plot")
    if not target.exists():
        raise HTTPException(404, "plot not ready")
    if target.suffix == ".csv":
        import csv as _csv
        with open(target, newline="") as f:
            rows = list(_csv.reader(f))
        cols = rows[0] if rows else []
        data = rows[1:]
        if limit and limit > 0:
            data = data[:limit]
        return JSONResponse({"columns": cols, "rows": data},
                            headers={"Cache-Control": "no-store"})
    if target.suffix == ".json":
        txt = target.read_text()
        # JS JSON.parse rejects NaN/Infinity tokens (Python json.dump emits them) -> null
        txt = re.sub(r"\bNaN\b", "null", txt)
        txt = re.sub(r"\b-?Infinity\b", "null", txt)
        return Response(content=txt, media_type="application/json",
                        headers={"Cache-Control": "no-store"})
    return FileResponse(target, media_type="image/png", headers={"Cache-Control": "no-store"})



@app.get("/api/run/{run_id}/section", response_class=HTMLResponse)
def run_section(run_id: str, id: str):
    """Serve a single <section> of mrag_report.html as a standalone mini-page.

    Used by section-type plots (Rank/Heatmap/HOMER-known) that are rendered
    inline in the report and have no standalone figure file. We extract just
    that section + its Plotly init scripts so the chat plot-pane shows ONLY
    that figure, not the whole report.
    """
    if id not in SECTION_PLOT_DIVS:
        raise HTTPException(403, "not a whitelisted section")
    report_fp = (_run_dir(run_id) / "output" / "mrag_report.html").resolve()
    out_dir = (_run_dir(run_id) / "output").resolve()
    if not str(report_fp).startswith(str(out_dir) + os.sep) or not report_fp.exists():
        raise HTTPException(404, "report not ready")
    html = report_fp.read_text()

    # --- extract <section id="ID" ...> ... </section> ---
    m = re.search(r'<section id="' + re.escape(id) + r'"[^>]*>', html)
    if not m:
        raise HTTPException(404, "section not found")
    start = m.start()
    end = html.find("</section>", m.end())
    section_html = html[start:end + len("</section>")] if end != -1 else html[start:]
    if id == "ag-validation" and end != -1:
        # The Section-13 graph script follows the section, up to the next <section>.  It draws from
        # Section 10b's graph (window.cySE.json()), so that section is carried too, off-screen:
        # laid out at a real size (a display:none container would give Cytoscape no geometry).
        nxt = html.find("<section", end)
        tail = html[end + len("</section>"): nxt if nxt != -1 else len(html)]
        section_html += "".join(re.findall(r"<script>.*?</script>", tail, re.S))
        tm = re.search(r'<section id="trn"[^>]*>', html)
        if tm:
            te = html.find("</section>", tm.end())
            if te != -1:
                section_html = ("<div aria-hidden='true' style='position:absolute;left:-20000px;"
                                "top:0;width:1200px'>" + html[tm.start():te + len("</section>")]
                                + "</div>" + section_html)

    # --- pull each Plotly.newPlot/safePlot statement for this section's divs ---
    lines = html.splitlines()
    scripts: list[str] = []
    for div in SECTION_PLOT_DIVS[id]:
        begin = None
        for i, ln in enumerate(lines):
            if (('newPlot("' + div + '"') in ln or ('newPlot(\'' + div + '\'') in ln
                    or ('safePlot("' + div + '"') in ln or ('safePlot(\'' + div + '\'') in ln):
                begin = i
                break
        if begin is None:
            continue
        chunk = []
        for ln in lines[begin:]:
            chunk.append(ln)
            if ln.rstrip().endswith("});"):
                break
        scripts.append("\n".join(chunk))

    # --- carry over report head <style> for visual fidelity ---
    sm = re.search(r"<style>.*?</style>", html, re.S)
    style = sm.group(0) if sm else ""

    body_scripts = "\n".join(scripts)
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>"
        + ("<script src='https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js'></script>"
           if id in ("trn", "ag-validation") else "")
        + style +
        "<style>body{margin:14px;background:#fff}"
        ".figure{position:relative}"
        ".dl-btns{position:absolute;top:8px;right:8px;display:flex;gap:4px;z-index:6;opacity:.4;transition:opacity .15s}"
        ".dl-btns:hover{opacity:1}"
        ".dl-btns button{font:500 11px/1.4 system-ui,-apple-system,sans-serif;padding:3px 9px;"
        "background:rgba(255,255,255,.95);border:1px solid #cbd5e1;border-radius:5px;cursor:pointer;"
        "color:#334155;box-shadow:0 1px 3px rgba(0,0,0,.08)}"
        ".dl-btns button:hover{background:#eef2f7;border-color:#94a3b8}</style></head><body>"
        + section_html +
        "<script>function safePlot(divId,data,layout){var el=document.getElementById(divId);"
        "if(!el)return;try{Plotly.newPlot(divId,data,layout,{responsive:true,displaylogo:false});}"
        "catch(e){console.error('safePlot',divId,e);}}\n"
        + body_scripts +
        "\n(function(){"
        "function slug(s){return String(s||'figure').trim().replace(/[^A-Za-z0-9._-]+/g,'_').slice(0,60)||'figure';}"
        "function bar(w){var b=w.querySelector(':scope > .dl-btns');if(!b){b=document.createElement('div');b.className='dl-btns';w.appendChild(b);}return b;}"
        "function add(w,l,f){var x=document.createElement('button');x.type='button';x.textContent=l;"
        "x.onclick=function(e){e.preventDefault();e.stopPropagation();f();};bar(w).appendChild(x);}"
        "function fw(el){return el.closest('.figure')||el.parentElement;}"
        "document.querySelectorAll('.figure img').forEach(function(img){var s=img.getAttribute('src');"
        "if(!s||img.offsetParent===null)return;var w=fw(img);if(w.dataset.dlImg)return;w.dataset.dlImg='1';"
        "add(w,'\u2b07 PNG',function(){var a=document.createElement('a');a.href=img.src;"
        "a.download=slug(img.getAttribute('alt'))+'.png';document.body.appendChild(a);a.click();a.remove();});});"
        "function ap(){if(typeof Plotly==='undefined')return;document.querySelectorAll('.js-plotly-plot').forEach(function(gd){"
        "if(gd.dataset.dlP)return;gd.dataset.dlP='1';var w=fw(gd);"
        "add(w,'\u2b07 SVG',function(){Plotly.downloadImage(gd,{format:'svg',filename:slug(gd.id)});});"
        "add(w,'\u2b07 PNG (hi-res)',function(){Plotly.downloadImage(gd,{format:'png',scale:4,filename:slug(gd.id)});});});}"
        "window.addEventListener('load',function(){setTimeout(ap,700);});setTimeout(ap,1400);"
        "})();"
        "</script></body></html>"
    )
    page = re.sub(r"\bNaN\b", "null", page)
    page = re.sub(r"\b-?Infinity\b", "null", page)
    return HTMLResponse(content=page, headers={"Cache-Control": "no-store"})


@app.get("/api/runs_list", response_class=JSONResponse)
def runs_list_json():
    """Lightweight run-id list (most recent first) for the chat plot-pane selector."""
    ids = [r["run_id"] for r in _list_runs(include_drafts=False)]
    return JSONResponse({"runs": ids})


@app.get("/runs", response_class=HTMLResponse)
def runs_list(request: Request, drafts: int = 0):
    inc = bool(drafts)
    return templates.TemplateResponse(
        "runs.html",
        {"request": request, "runs": _list_runs(include_drafts=inc), "include_drafts": inc},
    )


@app.get("/about", response_class=HTMLResponse)
def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/contact", response_class=HTMLResponse)
def contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})




# ============================================================
# Chat
# ============================================================

@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html", {})


@app.post("/api/chat", response_class=JSONResponse)
def chat_post(request: Request, body: dict):
    session_id = body.get("session_id") or ""
    if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
        session_id = uuid.uuid4().hex[:12]
    user_msg   = (body.get("message") or "").strip()
    if not user_msg:
        return JSONResponse({"error": "empty message"}, status_code=400)
    if session_id not in _CHAT_SESSIONS:
        from chat import load_conversation_into
        c = ChatContext(
            runs_root=RUNS_ROOT,
            llm_endpoint=QWEN_ENDPOINT,
            llm_model=QWEN_MODEL,
            session_id=session_id,
        )
        load_conversation_into(c, session_id)   # restore if disk has it
        _CHAT_SESSIONS[session_id] = c
    ctx = _CHAT_SESSIONS[session_id]
    try:
        out = chat_turn(ctx, user_msg)
    except Exception as e:
        return JSONResponse(
            {"error": f"{type(e).__name__}: {e}", "session_id": session_id},
            status_code=500,
        )
    return JSONResponse({"session_id": session_id, **out})


_SAFE_EXT = {".txt", ".tsv", ".csv", ".bed", ".narrowpeak", ".narrowPeak",
             ".gz", ".mtx", ".matrix", ".bw", ".bigwig", ".bigWig"}


@app.post("/api/chat/upload", response_class=JSONResponse)
async def chat_upload(session_id: str = Form(...),
                      rna:  list[UploadFile] = File(default=[]),
                      atac: list[UploadFile] = File(default=[]),
                      bw:   list[UploadFile] = File(default=[]),
                      meta: list[UploadFile] = File(default=[])):
    """Accept processed RNA / ATAC matrices (+ optional sample<->group / metadata
    sheet) and save them under MRAG_data/_uploads/<session>/{RNA,ATAC,meta}/ so the
    chat can detect + run them (flex_input processed-data path). No FASTQ needed."""
    sess = re.sub(r"[^A-Za-z0-9_]", "", session_id or "") or uuid.uuid4().hex[:12]
    # One run_<ts>/ dir per session; reuse if this session already started one so
    # separately-uploaded RNA and ATAC files collect into the same run dir.
    prior = _UPLOAD_DIRS.get(sess)
    if prior and Path(prior).exists():
        base = Path(prior)
    else:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        base = UPLOAD_ROOT / f"run_{ts}"
        base.mkdir(parents=True, exist_ok=True)
        _UPLOAD_DIRS[sess] = str(base)
    saved: dict[str, list[str]] = {"rna": [], "atac": [], "bw": [], "meta": []}
    total = 0
    for cat, files, sub in (("rna", rna, "RNA"),
                            ("atac", atac, "ATAC"),
                            ("bw", bw, "ATAC/bigwigs"),
                            ("meta", meta, "meta")):
        for f in files:
            if not f or not f.filename:
                continue
            fn = Path(f.filename).name  # strip any client path
            if Path(fn).suffix.lower() not in {e.lower() for e in _SAFE_EXT}:
                return JSONResponse(
                    {"error": f"unsupported file type: {fn} "
                              f"(allowed: matrix/BED/TSV/CSV/txt[.gz])"},
                    status_code=400)
            d = base / sub
            d.mkdir(parents=True, exist_ok=True)
            data = await f.read()
            (d / fn).write_bytes(data)
            saved[cat].append(str(d / fn))
            total += 1
    if total == 0:
        return JSONResponse({"error": "no files uploaded"}, status_code=400)
    return JSONResponse({"ok": True, "session_id": sess,
                         "upload_dir": str(base), "saved": saved,
                         "n_files": total})


@app.get("/api/sra/progress", response_class=JSONResponse)
def sra_progress(dest: str = "", session_id: str = ""):
    """Live state of one SRA download, for the chat page's progress panel.

    Deliberately dumb and strictly read-only: it reads the progress file through
    chat.sra_panel and can start, stop or change nothing, which is what makes it
    safe to poll every couple of seconds from the browser instead of making the
    LLM poll in a loop.

    `dest` is optional. Without it the panel is resolved from the SAVED
    conversation (disk) and then from any live fetch record on disk, so a page
    reload - or a restart of this web UI - still finds the download that is
    actually running. An unreadable or disallowed directory answers show:false
    rather than an error, because a polling panel must degrade quietly.
    """
    from chat import sra_panel
    try:
        return JSONResponse(sra_panel(dest, session_id))
    except Exception as exc:                    # noqa: BLE001 - reported, not 500
        return JSONResponse({"active": False, "show": False, "state": "failed",
                             "dest": dest, "lines": [], "files": [], "pct": 0.0,
                             "poll_ms": 0,
                             "message": "%s: %s" % (type(exc).__name__, exc)})


@app.get("/api/conversations", response_class=JSONResponse)
def conversations_list():
    from chat import list_conversations
    return JSONResponse({"conversations": list_conversations()})


@app.get("/api/conversations/{session_id}", response_class=JSONResponse)
def conversations_load(session_id: str):
    """Return all messages of a saved conversation (resume)."""
    from chat import load_conversation_into, ChatContext
    c = ChatContext(runs_root=RUNS_ROOT, llm_endpoint=QWEN_ENDPOINT,
                    llm_model=QWEN_MODEL, session_id=session_id)
    if not load_conversation_into(c, session_id):
        return JSONResponse({"error": "not found"}, status_code=404)
    # Also rehydrate the in-memory session so subsequent POSTs continue properly
    _CHAT_SESSIONS[session_id] = c
    # Filter messages to user/assistant only (drop system + tool result chatter).
    # Assistant messages are stored as raw JSON ({"reply":..., "actions":..., "done":...})
    # for LLM continuity — extract just "reply" for display.
    import json as _json
    import re as _re
    def _clean_assistant(content: str) -> str:
        if not content:
            return ""
        # Try direct JSON parse, then code-fence stripped variants
        for s in (content,
                  _re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(),
                          flags=_re.MULTILINE)):
            try:
                d = _json.loads(s)
                if isinstance(d, dict) and "reply" in d:
                    return d["reply"]
            except Exception:
                pass
        return content   # fall through: show raw (better than nothing)
    visible = []
    for m in c.messages:
        if m.get("role") == "user":
            visible.append({"role": "user", "content": m.get("content", "")})
        elif m.get("role") == "assistant":
            visible.append({"role": "assistant",
                            "content": _clean_assistant(m.get("content", ""))})
    return JSONResponse({
        "session_id": session_id,
        "title":      c.title,
        "messages":   visible,
    })


@app.delete("/api/conversations/{session_id}", response_class=JSONResponse)
def conversations_delete(session_id: str):
    from chat import delete_conversation
    _CHAT_SESSIONS.pop(session_id, None)
    ok = delete_conversation(session_id)
    return JSONResponse({"ok": ok})


# ============================================================
# Polish: file browser + download
# ============================================================

# Roots that the browse endpoint will allow walking into. Anything outside
# is rejected as a safety measure.
BROWSE_ROOTS = [Path(p) for p in [
    "/BIO1/MRAG/MRAG_data",
    EXTRA_DATA_ROOT,                 # MRAG_EXTRA_DATA_ROOT; "" = none
    "/BIO1/MRAG/Agent/reports",
    "/BIO1/MRAG/Agent/data",
    "/BIO1/MRAG/Agent",
    "/tmp",
] if p]


def _path_allowed(p: Path) -> bool:
    try:
        p = p.resolve()
    except OSError:
        return False
    for root in BROWSE_ROOTS:
        try:
            p.relative_to(root.resolve())
            return True
        except (ValueError, OSError):
            continue
    return False


@app.get("/api/browse", response_class=HTMLResponse)
def browse(request: Request, path: str = "/BIO1/MRAG/MRAG_data"):
    """Tiny server-side directory browser. Returns an HTMX partial that the
    File Browser modal swaps in. Only paths under BROWSE_ROOTS are allowed."""
    p = Path(path)
    if not p.exists() or not p.is_dir() or not _path_allowed(p):
        return HTMLResponse(
            "<p>Path not allowed. Allowed roots: "
            + ", ".join(str(r) for r in BROWSE_ROOTS) + "</p>",
            status_code=400,
        )
    try:
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        return HTMLResponse("<p>Permission denied.</p>", status_code=403)
    n_fastq = sum(1 for e in entries
                  if e.is_file() and e.name.lower().endswith((".fastq.gz", ".fq.gz",
                                                              ".fastq", ".fq")))
    return templates.TemplateResponse(
        request, "browse_partial.html",
        {"path": str(p.resolve()), "parent": str(p.resolve().parent),
         "entries": entries, "n_fastq": n_fastq},
    )


@app.get("/api/download/{run_id}")
def download_run(run_id: str):
    """Return a zip of <run_dir>/output (manifest, HTML, narratives, summaries)."""
    run_dir = _run_dir(run_id) / "output"
    if not run_dir.exists():
        raise HTTPException(404, f"run_id {run_id} has no output dir yet")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in run_dir.rglob("*"):
            if fp.is_file():
                zf.write(fp, fp.relative_to(run_dir.parent))
    buf.seek(0)
    from fastapi.responses import StreamingResponse
    headers = {"Content-Disposition": f'attachment; filename="{run_id}.zip"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)




@app.delete("/api/runs/{run_id}", response_class=JSONResponse)
def delete_run(run_id: str):
    """Delete a single run or sheet directory."""
    import shutil
    d = _run_dir(run_id)
    # Safety: must be a direct child of RUNS_ROOT and must start with sheet_ or run_
    if not d.exists() or d.parent.resolve() != RUNS_ROOT.resolve():
        raise HTTPException(404, f"{run_id} not found under RUNS_ROOT")
    if not (run_id.startswith("sheet_") or run_id.startswith("run_")):
        raise HTTPException(400, f"refusing to delete: {run_id} (must be sheet_* or run_*)")
    # Kill any still-running mrag process for this run
    pid_file = d / "pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 15)
        except (OSError, ValueError):
            pass
    shutil.rmtree(d, ignore_errors=True)
    return JSONResponse({"ok": True, "deleted": run_id})


@app.post("/api/cleanup_drafts", response_class=JSONResponse)
def cleanup_drafts():
    """Delete all abandoned `sheet_*` directories (no associated run started)."""
    import shutil
    deleted = []
    for d in RUNS_ROOT.iterdir():
        if d.is_dir() and d.name.startswith("sheet_"):
            shutil.rmtree(d, ignore_errors=True)
            deleted.append(d.name)
    return JSONResponse({"ok": True, "n_deleted": len(deleted), "deleted": deleted})



if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, ...) with no reload: this process serves the code it
    # loaded NOW, for as long as it lives. Editing app.py or chat.py changes
    # nothing until it is restarted, and there was no way to tell from outside
    # which version was live. Print it, so `ps` + this line always answer the
    # question.
    try:
        import datetime as _dtm
        from pathlib import Path as _P
        for _f in ("app.py", "chat.py"):
            _p = _P(__file__).with_name(_f)
            print("[webui] serving %s modified %s" % (
                _p, _dtm.datetime.fromtimestamp(_p.stat().st_mtime)
                        .strftime("%Y-%m-%d %H:%M:%S")), flush=True)
    except Exception:
        pass
    # Loopback by default; MRAG_WEBUI_HOST=0.0.0.0 serves every interface (no authentication).
    uvicorn.run(app, host=os.environ.get("MRAG_WEBUI_HOST", "127.0.0.1"),
                port=int(os.environ.get("MRAG_WEBUI_PORT", 8765)))
