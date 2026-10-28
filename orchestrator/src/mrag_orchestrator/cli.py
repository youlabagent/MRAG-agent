"""MRAG Orchestrator CLI.

  mrag detect          scan FASTQ / processed inputs, emit sample_sheet.json
  mrag confirm         sample_sheet.json -> run config TOML
  mrag run             run the pipeline from a run config TOML (--resume-manifest reuses
                       the cached steps of an earlier manifest)
  mrag status | debug  pretty-print a manifest / the diagnoses of its failed steps
  mrag testrun         narrative + render only, from pre-computed artifacts
  mrag variants        Top-N fanout + cross-variant synthesis on pre-computed artifacts
  mrag prep-subsample  first-N-reads FASTQ subsample for smoke tests
  mrag sra-lookup | sra-fetch | sra-status | sra-cancel   public SRA/ENA studies
"""
from __future__ import annotations
import argparse, glob, json, sys
from datetime import datetime
from pathlib import Path

from .detector import detect, iter_input_files
from .manifest import Manifest
from .nodes.viz_report import NarrativeNode, RenderNode


# ====================================================================
# detect
# ====================================================================

def cmd_detect(args) -> int:
    from .detector import classify_inputs
    from .config import SampleSheet, Condition, DetectWarning
    from datetime import datetime as _dt

    all_files: list[Path] = []
    if args.fastq_dir:
        d = Path(args.fastq_dir)
        # Recursive: scan d AND every subdirectory, classify ALL files so that
        # processed (non-FASTQ) inputs -- peaks BED, counts/TPM matrix, DESeq2
        # result -- are recognized alongside raw FASTQ (flexible input).
        all_files = iter_input_files(d)
        fastq_dir = d
    else:
        for pat in args.fastqs:
            for hit in sorted(glob.glob(pat)):
                all_files.append(Path(hit))
        fastq_dir = None

    grp = classify_inputs(all_files)
    paths = [Path(x) for x in grp["fastqs"]]
    suggestion = grp["precomputed"]

    # Processed matrices (atac_count_matrix / rna_log2_matrix) need per-modality
    # sample<->group labels. If a rich metadata sheet is present (a group column
    # plus columns whose values match the matrix headers), auto-build the group
    # TSVs and derive conditions -- so the web confirm/run path needs no manual
    # --precomputed. This is the 'various inputs, conversion handled for you' path.
    _has_atac_mtx = "atac_count_matrix" in suggestion
    _has_rna_mtx  = "rna_log2_matrix" in suggestion
    if (_has_atac_mtx or _has_rna_mtx) and not (
            suggestion.get("atac_group_tsv") or suggestion.get("rna_group_tsv")):
        from .adapters.flex_adapters import autobuild_group_tsvs
        import csv as _csv
        _gdir = str((fastq_dir or Path(".")) / "_mrag_groups")
        _used = {suggestion.get(k) for k in
                 ("atac_count_matrix", "rna_log2_matrix", "rna_counts_matrix",
                  "deseq_csv")}
        _cands = [str(q) for q in all_files
                  if q.suffix.lower() in (".tsv", ".csv", ".txt")
                  and str(q) not in _used]
        for _sh in _cands:
            try:
                _g = autobuild_group_tsvs(
                    _sh, _gdir,
                    atac_matrix=suggestion.get("atac_count_matrix") if _has_atac_mtx else None,
                    rna_matrix=suggestion.get("rna_log2_matrix") if _has_rna_mtx else None)
            except Exception:
                _g = {}
            if _g:
                suggestion.update(_g)
                grp["notes"].append(
                    f"Auto sample<->group labels from {Path(_sh).name}: "
                    f"{', '.join(sorted(_g.keys()))}")
                _gt = _g.get("atac_group_tsv") or _g.get("rna_group_tsv")
                _labels: list[str] = []
                for _r in _csv.DictReader(open(_gt), delimiter="\t"):
                    _lab = _r["group"]
                    if _lab not in _labels:
                        _labels.append(_lab)
                if len(_labels) >= 2 and not grp["arms"]:
                    grp["arms"] = sorted(_labels)[:2]
                break
        else:
            grp["notes"].append(
                "Processed matrix found but no metadata sheet matched the "
                "sample headers; supply a sample<->group TSV (or --precomputed "
                "atac_group_tsv/rna_group_tsv) at confirm.")

    # Second tier: still-missing group labels are inferred WITHOUT a metadata
    # sheet -- LLM reads the column-NAME signal (OE/control, KO/WT, ...); if the
    # names are opaque IDs, fall back to de-novo k=2 clustering on the matrix.
    _need = []
    if _has_atac_mtx and not suggestion.get("atac_group_tsv"):
        _need.append(("ATAC", "atac_count_matrix", "atac_group_tsv"))
    if _has_rna_mtx and not suggestion.get("rna_group_tsv"):
        _need.append(("RNA", "rna_log2_matrix", "rna_group_tsv"))
    if _need:
        from .adapters.flex_adapters import resolve_matrix_groups
        _gdir2 = (fastq_dir or Path(".")) / "_mrag_groups"
        _ordered = None
        for _mod, _mk, _gk in _need:
            _mx = suggestion.get(_mk)
            try:
                _res = resolve_matrix_groups(
                    _mx, _mod, str(_gdir2 / f"{_gk}.tsv"),
                    endpoint=args.endpoint, model=args.model)
            except Exception as _e:
                _res = None
                grp["notes"].append(f"{_mod}: group inference failed ({_e}); "
                                    "supply a sample<->group TSV at confirm.")
            if _res:
                suggestion[_gk] = _res["tsv"]
                grp["warnings_extra"] = grp.get("warnings_extra", [])
                grp["warnings_extra"].append(
                    ("high" if _res["method"] == "kmeans" else "medium",
                     _res["warning"]))
                if _ordered is None:
                    _ordered = _res["ordered_labels"]
        if _ordered and not grp["arms"]:
            grp["arms"] = _ordered[:2]

    if not paths and not suggestion:
        print(json.dumps({"error":"no FASTQ or processed inputs found",
                          "fastq_dir": args.fastq_dir,
                          "fastqs": args.fastqs}, indent=2), file=sys.stderr)
        return 2

    if paths:
        sheet = detect(
            fastq_paths=paths, fastq_dir=fastq_dir,
            use_llm=not args.no_llm_assist, endpoint=args.endpoint, model=args.model,
        )
    else:
        # Processed-only inputs: no FASTQ to detect. Minimal sheet + synth below.
        sheet = SampleSheet(detected_at=_dt.now().isoformat(),
                            fastq_dir=str(fastq_dir) if fastq_dir else None,
                            samples=[], conditions=[])

    # Synthesize conditions from detected arms when FASTQ gave fewer than 2.
    if len(sheet.conditions) < 2 and len(grp["arms"]) >= 2:
        a0, a1 = grp["arms"][0], grp["arms"][1]
        sheet.conditions = [
            Condition(label=a0, n_reps=0, modality_counts={}),
            Condition(label=a1, n_reps=0, modality_counts={}),
        ]
        sheet.needs_user_action = True
        sheet.warnings.append(DetectWarning(severity="medium",
            msg=f"Conditions synthesized from processed inputs: treatment='{a0}' "
                f"vs control='{a1}'. VERIFY this order at confirm."))

    # Derive conditions from any resolved group TSV (autobuilt OR a detected
    # sample<->group file) when arms are still unknown, so processed-matrix runs
    # get 2 conditions for treatment/control ordering.
    if len(sheet.conditions) < 2 and not grp["arms"]:
        _gt = suggestion.get("atac_group_tsv") or suggestion.get("rna_group_tsv")
        if _gt:
            import csv as _csv2
            _labs: list[str] = []
            try:
                for _r in _csv2.DictReader(open(_gt), delimiter="\t"):
                    _lv = (_r.get("group") or "").strip()
                    if _lv and _lv not in _labs:
                        _labs.append(_lv)
            except Exception:
                _labs = []
            if len(_labs) >= 2:
                grp["arms"] = sorted(_labs)[:2]
                from .config import Condition as _Cond, DetectWarning as _DW
                a0, a1 = grp["arms"]
                # count replicates per label from each modality group TSV
                def _count_groups(path):
                    c = {}
                    try:
                        for _r in _csv2.DictReader(open(path), delimiter="\t"):
                            _lv = (_r.get("group") or "").strip()
                            if _lv:
                                c[_lv] = c.get(_lv, 0) + 1
                    except Exception:
                        pass
                    return c
                _atac_c = _count_groups(suggestion["atac_group_tsv"]) if suggestion.get("atac_group_tsv") else {}
                _rna_c = _count_groups(suggestion["rna_group_tsv"]) if suggestion.get("rna_group_tsv") else {}
                def _mc(lbl):
                    m = {}
                    if _atac_c: m["atac"] = _atac_c.get(lbl, 0)
                    if _rna_c: m["rna"] = _rna_c.get(lbl, 0)
                    return m
                sheet.conditions = [
                    _Cond(label=a0, n_reps=max(_atac_c.get(a0, 0), _rna_c.get(a0, 0)), modality_counts=_mc(a0)),
                    _Cond(label=a1, n_reps=max(_atac_c.get(a1, 0), _rna_c.get(a1, 0)), modality_counts=_mc(a1)),
                ]
                sheet.needs_user_action = True
                sheet.warnings.append(_DW(severity="medium",
                    msg=f"Conditions from group labels: treatment='{a0}' vs "
                        f"control='{a1}'. VERIFY this order at confirm."))
    # POLICY: processed ATAC matrix requires per-sample bigWig coverage to run
    # the ATAC signal sections (peak heatmap / per-TF profile / locus snapshots).
    # Auto-detect the bigwigs/ dir next to the matrix and report the result
    # explicitly (count when present, high warning when absent) so the chat LLM
    # never has to ASK the user about bigWigs that are already on disk.
    _atac_mtx = suggestion.get("atac_count_matrix")
    if _atac_mtx:
        _bwdir = Path(_atac_mtx).parent / "bigwigs"
        _bwfiles = []
        if _bwdir.is_dir():
            _bwfiles = sorted(_bwdir.glob("*.bw")) + sorted(_bwdir.glob("*.bigWig"))
        if _bwfiles:
            suggestion["bigwig_dir"] = str(_bwdir)
            suggestion["bigwig_count"] = len(_bwfiles)
            sheet.warnings.append(DetectWarning(severity="low",
                msg=(f"{len(_bwfiles)} per-sample bigWig coverage files found in "
                     f"{_bwdir} — ATAC signal plots (peak heatmap / per-TF "
                     f"profile / locus snapshots) enabled. No bigWig upload needed.")))
        else:
            sheet.needs_user_action = True
            sheet.warnings.append(DetectWarning(severity="high",
                msg=("Processed ATAC count matrix found but NO per-sample bigWig "
                     "coverage files (expected in " + str(_bwdir) + "). bigWigs are "
                     "REQUIRED for processed ATAC data — upload the per-sample .bw "
                     "files (chat upload panel has a dedicated bigWig slot) before "
                     "running.")))
    sheet.precomputed_suggestion = suggestion
    for _n in grp["notes"]:
        sheet.warnings.append(DetectWarning(severity="low", msg=_n))
    for _sev, _msg in grp.get("warnings_extra", []):
        sheet.warnings.append(DetectWarning(severity=_sev, msg=_msg))
        if _sev == "high":
            sheet.needs_user_action = True
    # An explicit --organism overrides whatever the study metadata said (and
    # supplies one when there is no metadata at all).
    _org_override = (getattr(args, "organism", None) or "").strip()
    if _org_override:
        from .organism import resolve as _resolve_org
        _prof = _resolve_org(_org_override)
        sheet.organism = _prof.key
        sheet.organism_source = "explicit"
        sheet.organism_detail = "--organism %s -> %s (%s)" % (
            _org_override, _prof.scientific_name, _prof.genome)
        for _s in sheet.samples:
            _s.organism = _prof.scientific_name
    out_json = sheet.model_dump_json(indent=2)
    if args.output:
        Path(args.output).write_text(out_json)
        print(json.dumps({"ok":True, "n_samples":len(sheet.samples),
                          "n_conditions":len(sheet.conditions),
                          "needs_user_action":sheet.needs_user_action,
                          "organism":sheet.organism or "",
                          "organism_source":sheet.organism_source,
                          "organism_detail":sheet.organism_detail,
                          "summary_human":sheet.summary_human,
                          "output":args.output}, indent=2))
    else:
        print(out_json)
    return 0 if not sheet.needs_user_action else 3


# ====================================================================
# testrun (narrative + render only)
# ====================================================================

def cmd_testrun(args) -> int:
    """Test run: skip every upstream step and feed pre-computed JSONs to
    narrative + render."""
    art = Path(args.artifacts_dir)
    if not art.exists():
        print(json.dumps({"error":f"artifacts_dir not found: {art}"}, indent=2), file=sys.stderr)
        return 2
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_label or datetime.now().strftime("testrun_%Y%m%dT%H%M%S")
    manifest = Manifest(
        run_id=run_id,
        config_path=None,
        code_version="mrag-orchestrator 0.1.0 (testrun)",
        llm={"endpoint":args.endpoint, "model":args.model},
        manifest_path=out_dir / "manifest.json",
    )
    manifest.declare_inputs("artifacts", [str(art)])

    cache = None
    if getattr(args, "resume_manifest", None):
        from .checkpoint import CheckpointCache, restore_node_from_cache
        cache = CheckpointCache.load_from_manifest(args.resume_manifest)
        print(f"[testrun] resume from {args.resume_manifest}: {len(cache.all_cached_names())} cached steps", flush=True)

    # NarrativeNode reads from artifacts_dir
    narrative_node = NarrativeNode(
        inputs={
            "tfs_summary_json":     str(art / "tfs_summary.json"),
            "targets_summary_json": str(art / "targets_summary.json"),
            "conditions_meta_json": str(art / "conditions_meta.json"),
        },
        params={
            "top_n": args.top_n,
            "glm_endpoint": args.endpoint,
            "glm_model":    args.model,
            "temperature":  0.2,
            "max_tokens":   2000,
            "generate_abstract": True,
        },
        output_dir=out_dir,
    )

    # RenderNode chains after narrative
    def build_render(narrative: NarrativeNode) -> RenderNode:
        return RenderNode(
            inputs={
                "narrative_md":   narrative.outputs.get("narrative_md"),
                "abstract_md":    narrative.outputs.get("abstract_md"),
                "narrative_audit": narrative.outputs.get("narrative_audit"),
                "volcano_atac_json": str(art / "volcano_atac.json"),
                "volcano_rna_json":  str(art / "volcano_rna.json"),
                "tf_rank_json":      str(art / "tf_rank.json"),
                "trn_json":          str(art / "trn.json"),
                "per_tf_profile_json": str(art / "per_tf_profile.json"),
                "tf_table_csv":      str(art / "tf_table.csv"),
                "locus_manifest_json": str(args.locus_manifest) if args.locus_manifest else "",
            },
            params={
                "report_title":    args.title or f"MRAG Report (testrun, Top {args.top_n})",
                "study_label":     args.study_label or "",
                "conditions_str":  args.conditions_str or "",
                "tf_selection_label": f"Pcom-deterministic Top {args.top_n}",
            },
            output_dir=out_dir,
        )

    # Sequential 2-node mini-pipeline. RenderNode is built after NarrativeNode runs
    # (its inputs depend on narrative outputs).
    if cache is not None:
        _hash = narrative_node.input_hash()
        c = cache.lookup(narrative_node.name, _hash)
        if c is not None:
            restore_node_from_cache(narrative_node, c)
            print(f"[testrun] ▷ {narrative_node.name} cached", flush=True)
        else:
            narrative_node.execute()
    else:
        narrative_node.execute()
    manifest.add_step(narrative_node.to_manifest()); manifest.write_atomic()
    if narrative_node.status == "fail":
        manifest.finalize(); manifest.write_atomic()
        print(json.dumps({"ok":False, "failed_at":"narrative",
                          "error":narrative_node.error,
                          "manifest":str(manifest.path)}, indent=2))
        return 1

    render_node = build_render(narrative_node)
    if cache is not None:
        _hash2 = render_node.input_hash()
        c2 = cache.lookup(render_node.name, _hash2)
        if c2 is not None:
            restore_node_from_cache(render_node, c2)
            print(f"[testrun] ▷ {render_node.name} cached", flush=True)
        else:
            render_node.execute()
    else:
        render_node.execute()
    manifest.add_step(render_node.to_manifest()); manifest.write_atomic()
    manifest.finalize(summary={
        "narrative_md": narrative_node.outputs.get("narrative_md"),
        "abstract_md":  narrative_node.outputs.get("abstract_md"),
        "final_html":   render_node.outputs.get("mrag_report_html"),
    })
    manifest.write_atomic()

    print(json.dumps({
        "ok": render_node.status != "fail",
        "manifest": str(manifest.path),
        "narrative_md": narrative_node.outputs.get("narrative_md"),
        "final_html":   render_node.outputs.get("mrag_report_html"),
        "status_overall": manifest.data["status"],
    }, indent=2))
    return 0 if render_node.status != "fail" else 1


def cmd_prep_subsample(args) -> int:
    """Subsample FASTQs (first n_reads each) for fast pipeline smoke tests."""
    from .subsample import subsample_fastqs
    paths = []
    if args.fastq_dir:
        for ext in ("*.fastq.gz","*.fq.gz","*.fastq","*.fq"):
            paths.extend(sorted(Path(args.fastq_dir).glob(ext)))
    else:
        for p in args.fastqs:
            for hit in sorted(glob.glob(p)):
                paths.append(Path(hit))
    if not paths:
        print(json.dumps({"error":"no FASTQ files found"}, indent=2), file=sys.stderr)
        return 2
    res = subsample_fastqs(paths, Path(args.output_dir), n_reads=args.n_reads)
    print(json.dumps({"ok":True, **{k:v for k,v in res.items() if k!="written"},
                      "first_3_written": res["written"][:3]}, indent=2))
    return 0


def cmd_variants(args) -> int:
    """Run the Top 4/6/8 fanout + cross-variant synthesis on pre-computed
    upstream artifacts. Produces:
        <output_dir>/branches/top{N}/...
        <output_dir>/cross_variant_summary.json
        <output_dir>/mrag_report.html         (single, with cross-variant section)
    """
    from .resources import start_run_guard, check_start_memory
    _rc = check_start_memory()
    if _rc:
        print(json.dumps({"ok": False, "error": "refused to start: MemAvailable below the start floor "
                          "(MRAG_MEM_START_FLOOR_GB); MRAG_MEM_IGNORE_SYSFLOOR=1 overrides"}), file=sys.stderr)
        return _rc
    start_run_guard()
    from datetime import datetime
    from .variants  import run_all_variants
    from .synthesis import synthesize_cross_variant, render_cross_variant_markdown
    from .nodes.viz_report import RenderNode
    from .manifest import Manifest

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    branches_dir = out_dir / "branches"
    Ns = list(args.Ns)

    run_id = args.run_label or datetime.now().strftime("variants_%Y%m%dT%H%M%S")
    manifest = Manifest(
        run_id=run_id, code_version="mrag-orchestrator 0.1.0 (variants)",
        llm={"endpoint": args.endpoint, "model": args.model},
        manifest_path=out_dir / "manifest.json",
    )

    shared = {
        "topn_dir":              args.topn_dir,
        "motif_tsv":             args.motif_tsv,
        "diffbind_csv":          args.diffbind_csv,
        "deseq_csv":             args.deseq_csv,
        "tfs_summary_json":      args.tfs_summary_json,
        "targets_summary_json":  args.targets_summary_json,
        "conditions_meta_json":  args.conditions_meta_json,
    }

    cache = None
    if args.resume_manifest:
        from .checkpoint import CheckpointCache
        cache = CheckpointCache.load_from_manifest(args.resume_manifest)
        print(f"[variants] resume from {args.resume_manifest}: {len(cache.all_cached_names())} cached steps", flush=True)
    print(f"[variants] running Ns={Ns}, parallel={args.parallel}", flush=True)
    results = run_all_variants(
        shared=shared, output_dir=branches_dir, Ns=Ns,
        llm_endpoint=args.endpoint, llm_model=args.model,
        parallel=args.parallel, manifest=manifest,
        run_narrative=not args.no_narrative,
        cache=cache,
    )

    # ---- Synthesis ----
    variants_artifacts = {
        N: {"top_json":      str(Path(args.topn_dir)/f"top{N}.json"),
            "consensus_tsv": r.get("consensus_tsv","")}
        for N, r in results.items()
    }
    summary_path = out_dir / "cross_variant_summary.json"
    summary = synthesize_cross_variant(variants_artifacts, summary_path)
    cross_md_path = out_dir / "cross_variant_summary.md"
    cross_md_path.write_text(render_cross_variant_markdown(summary))
    manifest.add_step({
        "name": "cross_variant_synthesis", "status": "ok",
        "outputs": [{"key":"summary_json","path":str(summary_path)},
                    {"key":"summary_md","path":str(cross_md_path)}],
        "gates": [{"name":"robust_core","result":"pass","value":len(summary["robust_core_tfs"])}],
    })

    # ---- Final single HTML using canonical N (Top 6 by default) ----
    canonical_N = args.canonical_N
    canonical_branch = branches_dir / f"top{canonical_N}"
    canonical_artifacts = results.get(canonical_N, {})
    render = RenderNode(
        inputs={
            "narrative_md":     canonical_artifacts.get("narrative_md",""),
            "abstract_md":      canonical_artifacts.get("abstract_md",""),
            "narrative_audit":  canonical_artifacts.get("narrative_audit",""),
            "cross_variant_md": str(cross_md_path),
            "tf_table_csv":     args.tf_table_csv or "",
            "tf_rank_json":     args.tf_rank_json or "",
            "trn_json":         args.trn_json or "",
            "per_tf_profile_json": args.per_tf_profile_json or "",
            "volcano_atac_json": args.volcano_atac_json or "",
            "volcano_rna_json":  args.volcano_rna_json or "",
        },
        params={
            "report_title": args.title or f"MRAG Variants (Top {'/'.join(str(n) for n in Ns)})",
            "study_label":  args.study_label or "",
            "conditions_str": args.conditions_str or "",
            "tf_selection_label": f"Pcom-deterministic Top {'/'.join(str(n) for n in Ns)} (canonical={canonical_N})",
        },
        output_dir=out_dir,
    )
    render.execute()
    manifest.add_step(render.to_manifest())
    manifest.finalize(summary={
        "robust_core_tfs":  summary["robust_core_tfs"],
        "stability_score":  summary["stability_score_jaccard_lo_hi"],
        "final_html":       render.outputs.get("mrag_report_html",""),
        "cross_variant_summary": str(summary_path),
        "per_N_branches":   {str(N): r.get("branch_dir","") for N, r in results.items()},
    })
    manifest.write_atomic()

    print(json.dumps({
        "ok": render.status != "fail",
        "robust_core_tfs": summary["robust_core_tfs"],
        "fringe_tfs":      summary["fringe_tfs"],
        "stability":       summary["stability_score_jaccard_lo_hi"],
        "manifest":        str(manifest.path),
        "final_html":      render.outputs.get("mrag_report_html",""),
    }, indent=2))
    return 0 if render.status != "fail" else 1


def cmd_status(args) -> int:
    """Pretty-print a manifest's current state."""
    m = Path(args.manifest)
    if not m.exists():
        print(json.dumps({"error":f"manifest not found: {m}"}, indent=2), file=sys.stderr)
        return 2
    d = json.loads(m.read_text())
    print(f"run_id:   {d.get('run_id')}")
    print(f"status:   {d.get('status')}")
    print(f"started:  {d.get('started_at')}")
    print(f"ended:    {d.get('ended_at')}")
    print(f"model:    {d.get('model')}")
    print(f"steps ({len(d.get('steps',[]))}):")
    for s in d.get('steps', []):
        marker = {"ok":"✓","fail":"✗","skipped":"▷","provided":"►","running":"⏳"}.get(s.get("status",""),"?")
        print(f"  {marker} {s.get('name','?'):<28} status={s.get('status'):<10} hash={s.get('input_hash','')[:24]}...")
        if s.get("diagnosis"):
            dx = s["diagnosis"]
            print(f"      → {dx.get('rule')}: {dx.get('likely_cause')}")
    summ = d.get("summary", {})
    if summ:
        print("summary:")
        for k, v in summ.items():
            if isinstance(v, (list, dict)) and len(str(v)) > 100:
                print(f"  {k}: ({type(v).__name__} of len {len(v) if hasattr(v,'__len__') else '?'})")
            else:
                print(f"  {k}: {v}")
    return 0 if d.get("status") in ("ok","running") else 1


def cmd_debug(args) -> int:
    """Print diagnoses for failed steps in a manifest. Returns 0 if no failures."""
    m = Path(args.manifest)
    if not m.exists():
        print(json.dumps({"error":f"manifest not found: {m}"}, indent=2), file=sys.stderr)
        return 2
    d = json.loads(m.read_text())
    failed = [s for s in d.get("steps", []) if s.get("status") == "fail"]
    if not failed:
        print(json.dumps({"ok":True, "n_failures":0,
                          "run_status": d.get("status")}, indent=2))
        return 0
    out = {"ok":False, "n_failures": len(failed), "diagnoses": []}
    for s in failed:
        out["diagnoses"].append({
            "step":      s.get("name"),
            "error":     s.get("error"),
            "diagnosis": s.get("diagnosis"),
            "gates":     [g for g in s.get("gates",[]) if g.get("result") == "fail"],
        })
    print(json.dumps(out, indent=2))
    return 1


def cmd_confirm(args) -> int:
    """Convert a sample_sheet.json into a run config TOML.

    The user can pass --precomputed key=path KV pairs to wire up
    pre-computed artifacts (the matching preprocessing steps are then skipped).
    """
    from .config import save_run_config
    from .runner import sample_sheet_to_run_config

    pre = {}
    for kv in (args.precomputed or []):
        if "=" not in kv:
            print(json.dumps({"error": f"--precomputed expects key=path (got: {kv})"}, indent=2), file=sys.stderr)
            return 2
        k, v = kv.split("=", 1)
        pre[k.strip()] = v.strip()

    cfg = sample_sheet_to_run_config(
        sheet_path=args.sheet, output_dir=args.output_dir,
        run_label=args.run_label, precomputed=pre, canonical_N=args.canonical_N,
        cancer_type=getattr(args, "cancer_type", None) or None,
        cell_line=getattr(args, "cell_line", None) or None,
        tissue=getattr(args, "tissue", None) or None,
        literature_domains=getattr(args, "literature_domains", None) or None,
        organism=getattr(args, "organism", None) or None,
        treatment_arm=getattr(args, "treatment_arm", None) or None,
    )
    save_run_config(cfg, args.output_config)
    from .literature import resolve_literature_domains as _rl, domains_label as _dl
    _ld, _lu = _rl(cfg.literature_domains or None, cfg.cancer_type, cfg.tissue)
    from .organism import resolve as _resolve_org, PROFILES as _PROFILES
    _org_key = cfg.organism or ""
    _org_prof = _PROFILES.get(_org_key) if _org_key else None
    print(json.dumps({
        "ok": True,
        "config":      args.output_config,
        "organism":    _org_key,
        "genome":      _org_prof.genome if _org_prof else "",
        "organism_note": ("" if _org_key else
                          "no organism declared or detected; the run will "
                          "detect it, and default to human if it cannot"),
        "literature_domains": _ld,
        # same string as the report's Section 2 "LLM domain" row (2026-09-17)
        "literature_domains_label": _dl(_ld, explicit=cfg.literature_domains or None),
        "literature_unmatched": _lu,
        "run_label":   cfg.run_label,
        # conditions[0] is the TREATMENT arm everywhere downstream: it sets
        # every fold sign, the red/blue locus-track colours, the GIGGLE labels
        # and which arm's peaks ROSE calls super-enhancers on. It used to be
        # whichever arm appeared first in filename order, with nothing saying
        # so and no way to change it (2026-09-23). Say it, every time.
        "conditions":  [c.label for c in cfg.conditions],
        "treatment_arm": (cfg.conditions[0].label if cfg.conditions else ""),
        "control_arm":   (cfg.conditions[1].label
                          if len(cfg.conditions) > 1 else ""),
        "arm_order_note": (
            "conditions[0]=%r is the TREATMENT arm and conditions[1]=%r the "
            "control; every fold sign and track colour follows this order. "
            "Pass --treatment-arm to swap them."
            % (cfg.conditions[0].label if cfg.conditions else "",
               cfg.conditions[1].label if len(cfg.conditions) > 1 else "")),
        "n_samples":   len(cfg.samples),
        "n_conditions":len(cfg.conditions),
        "precomputed_keys": [k for k, v in cfg.precomputed.model_dump().items() if v],
        "canonical_N": cfg.canonical_N,
    }, indent=2))
    return 0


def cmd_run(args) -> int:
    """Run the full downstream pipeline from a config.toml."""
    # account-wide memory guard (resource_guard.py) + run-root marker
    from .resources import start_run_guard, check_start_memory
    _rc = check_start_memory()
    if _rc:
        print(json.dumps({"ok": False, "error": "refused to start: MemAvailable below the start floor "
                          "(MRAG_MEM_START_FLOOR_GB); MRAG_MEM_IGNORE_SYSFLOOR=1 overrides"}), file=sys.stderr)
        return _rc
    start_run_guard()
    from .config import load_run_config
    from .runner import run_from_config

    cfg = load_run_config(args.config)
    try:
        result = run_from_config(cfg, resume_manifest=args.resume_manifest)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


# ====================================================================
# main
# ====================================================================

def cmd_sra_lookup(args) -> int:
    """Read-only: expand an accession to its study and judge whether MRAG can use it."""
    from .sra import lookup, SRAError
    try:
        rep = lookup(args.accession, llm_assist=not args.no_llm_assist,
                     endpoint=args.endpoint, model=args.model,
                     use_cache=args.use_cache,
                     all_runs=getattr(args, "all_runs", False))
    except SRAError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 3
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(rep, indent=2))
    v = rep["verdict"]
    print(json.dumps({
        "ok": True,
        "input": rep["input"],
        "study": rep["study"],
        "studies": rep.get("studies", []),
        "dest": rep["dest"],
        "n_runs": rep["n_runs"],
        "selection": rep.get("selection", {}),
        "selected_runs": rep.get("selected_runs", []),
        "n_selected": rep.get("n_selected", rep["n_runs"]),
        "n_total": rep.get("n_total", rep["n_runs"]),
        "whole_study": bool(rep.get("whole_study", True)),
        "from_cache": bool(rep.get("from_cache")),
        "status": v["status"],
        # ENA's scientific_name, which used to be read and then dropped here.
        "species": rep.get("species", []),
        "organism": rep.get("organism", ""),
        "missing": v.get("missing", []),
        "n_atac": v["n_atac"],
        "n_rna": v["n_rna"],
        "arms": v["arms"],
        "per_arm": v["per_arm"],
        "total_bytes": v["total_bytes"],
        "total_human": v["total_human"],
        "reasons": v["reasons"],
        "warnings": v["warnings"],
        "samples": [{"run": s["run"], "modality": s["modality"], "arm": s["arm"],
                     "replicate": s["replicate"], "layout": s["layout"],
                     "species": s.get("species", ""),
                     "title": s["title"], "bytes": s["bytes"]}
                    for s in rep["samples"]],
        "output": args.output,
    }, indent=2))
    return 0


def cmd_sra_fetch(args) -> int:
    """Download a study that lookup already cleared. Refuses an unusable one.

    Any failure is also written into <dest>/sra_progress.json, so `mrag sra-status`
    (and the chat agent polling it) reports the real reason instead of an eternal
    "the download may still be starting".
    """
    from .sra import fetch, resume_fetch, mark_failed, SRAError
    accs = list(getattr(args, "accession", None) or [])
    resume = bool(getattr(args, "resume", False))
    if not accs and not resume:
        print(json.dumps({"ok": False, "state": "not_started",
                          "error": "no accession given",
                          "user_message": "NOTHING was downloaded: no accession was "
                                          "given and --resume was not used."},
                         indent=2))
        return 2
    if resume and not getattr(args, "dest", None):
        print(json.dumps({"ok": False, "state": "not_started",
                          "error": "--resume needs --dest",
                          "user_message": "NOTHING was resumed: --resume needs the "
                                          "download directory (--dest DIR)."},
                         indent=2))
        return 2
    try:
        if resume:
            res = resume_fetch(args.dest, force=args.force,
                               llm_assist=not args.no_llm_assist,
                               use_cache=not args.no_cache,
                               accession=accs or None,
                               all_runs=getattr(args, "all_runs", False))
        else:
            res = fetch(accs, dest=args.dest, force=args.force,
                        llm_assist=not args.no_llm_assist,
                        use_cache=not args.no_cache,
                        all_runs=getattr(args, "all_runs", False))
    except SRAError as exc:
        if getattr(args, "dest", None):
            mark_failed(args.dest, str(exc))
        print(json.dumps({"ok": False, "state": "failed", "error": str(exc),
                          "user_message": "The download failed before it started: %s"
                                          % exc}, indent=2))
        return 3
    except Exception as exc:                       # noqa: BLE001 - recorded, then re-raised
        if getattr(args, "dest", None):
            mark_failed(args.dest, "%s: %s" % (type(exc).__name__, exc))
        raise
    state = res.get("state")
    res["ok"] = state in ("done", "running")
    if state == "already_running":
        res.setdefault("user_message",
                       "NOTHING new was started: a download is already running in %s."
                       % args.dest)
    elif state == "refused":
        res["user_message"] = ("Download REFUSED, nothing was transferred: %s"
                               % "; ".join(res.get("reasons") or []))
    elif state == "partial":
        res["user_message"] = ("Download finished WITH FAILURES: %s of %s files, "
                               "%s failed." % (res.get("done"), res.get("total_files"),
                                               res.get("failed")))
    elif state == "done":
        res["user_message"] = ("Download finished: %s of %s files."
                               % (res.get("done"), res.get("total_files")))
    print(json.dumps(res, indent=2))
    return 0 if res["ok"] else 3


def cmd_sra_status(args) -> int:
    """Truthful state of one download directory (see sra.fetch_state)."""
    from .sra import fetch_state
    st = fetch_state(args.dest)
    print(json.dumps(st, indent=2))
    return 0 if st.get("ok") else 3


def cmd_sra_cancel(args) -> int:
    """Stop the download running in --dest and keep the partial file.

    Kills the fetch process group (only when the fetch leads it) or the pid and
    its own children, verifies they are gone, and marks the progress file
    cancelled so `sra-status` stops claiming a download is in flight.
    """
    from .sra import cancel_fetch
    res = cancel_fetch(args.dest, grace=float(getattr(args, "grace", 6.0)))
    print(json.dumps(res, indent=2))
    return 0 if res.get("ok") else 3


def main(argv: list[str] | None = None) -> int:
    # 2026-09-16 shared-server policy: pin this command and every child it will
    # start to MRAG_CPU_LIST (default 32-47, "none" = off) before any worker
    # thread or subprocess exists (resources.py).
    try:
        from .resources import apply_cpu_cap
        apply_cpu_cap()
    except Exception as _e:
        print(f"[mrag resources] CPU cap skipped: {_e}", file=sys.stderr)
    p = argparse.ArgumentParser(prog="mrag", description="MRAG Orchestrator")
    sub = p.add_subparsers(dest="cmd", required=True)

    # detect
    pd = sub.add_parser("detect", help="Auto-detect samples from FASTQ paths")
    g = pd.add_mutually_exclusive_group(required=True)
    g.add_argument("--fastq-dir", help="Directory containing FASTQ files")
    g.add_argument("--fastqs", nargs="+", help="Explicit FASTQ paths or globs")
    pd.add_argument("--output","-o", help="Write sample_sheet.json")
    pd.add_argument("--llm-assist", action="store_true",
                    help="Deprecated: LLM arm grouping is on by default")
    pd.add_argument("--no-llm-assist", action="store_true",
                    help="Rule-based arms only (deterministic, prostate vocabulary)")
    pd.add_argument("--endpoint", default="http://127.0.0.1:11500")
    pd.add_argument("--model", default="batiai/qwen3.6-27b:q6")
    pd.add_argument("--organism", default=None,
                    help="Override the detected species: human/hg38 or mouse/mm10. "
                         "Default: read from the study metadata beside the FASTQs.")
    pd.set_defaults(func=cmd_detect)

    # sra-lookup / sra-fetch
    psl = sub.add_parser("sra-lookup",
                         help="Expand an SRA/GEO accession and judge MRAG usability (no download)")
    psl.add_argument("accession", nargs="+",
                     help="One or more accessions (SRR/SRP/PRJNA/GSE). Pass the RNA and the ATAC accession together when they were deposited apart.")
    psl.add_argument("--output", "-o", help="Write the full report JSON here")
    psl.add_argument("--no-llm-assist", action="store_true",
                     help="Use only the keyword rules for arm assignment")
    psl.add_argument("--endpoint", default="http://127.0.0.1:11500")
    psl.add_argument("--model", default="batiai/qwen3.6-27b:q6")
    psl.add_argument("--use-cache", action="store_true",
                     help="Reuse a recent cached lookup instead of re-querying")
    psl.add_argument("--all-runs", action="store_true",
                     help="Report the WHOLE study even when run accessions were "
                          "given (default: run accessions select only those runs)")
    psl.set_defaults(func=cmd_sra_lookup)

    psf = sub.add_parser("sra-fetch",
                         help="Download every usable run of a study into <dest>/<modality>/")
    psf.add_argument("accession", nargs="*",
                     help="Same accession forms as sra-lookup. Run accessions "
                          "download ONLY those runs. Optional with --resume.")
    psf.add_argument("--dest", help="Target dir (default /BIO1/MRAG/MRAG_data/<study>)")
    psf.add_argument("--force", action="store_true",
                     help="Download even when the usability check failed")
    psf.add_argument("--no-llm-assist", action="store_true",
                     help="Use only the keyword rules for arm assignment")
    psf.add_argument("--no-cache", action="store_true",
                     help="Re-run the lookup instead of reusing the approved one")
    psf.add_argument("--all-runs", action="store_true",
                     help="Fetch the WHOLE study even when run accessions were given")
    psf.add_argument("--resume", action="store_true",
                     help="Continue the cancelled or partial download in --dest; "
                          "truncated files are continued, complete ones skipped. "
                          "The accessions are read back from the launch record.")
    psf.set_defaults(func=cmd_sra_fetch)

    pss = sub.add_parser("sra-status",
                         help="Truthful state of a download dir (not_started / starting / "
                              "running / partial / cancelled / done / refused / failed)")
    pss.add_argument("--dest", required=True, help="The download directory to inspect")
    pss.set_defaults(func=cmd_sra_status)

    psc = sub.add_parser("sra-cancel",
                         help="Stop the download running in --dest; the partly "
                              "downloaded file is kept for sra-fetch --resume")
    psc.add_argument("--dest", required=True, help="The download directory to stop")
    psc.add_argument("--grace", type=float, default=6.0,
                     help="Seconds to wait for SIGTERM before SIGKILL (default 6)")
    psc.set_defaults(func=cmd_sra_cancel)

    # testrun
    pt = sub.add_parser("testrun", help="Narrative+render from pre-computed artifacts")
    pt.add_argument("--artifacts-dir", required=True,
                    help="Directory with tfs_summary.json, targets_summary.json, etc.")
    pt.add_argument("--output-dir", required=True, help="Where to write outputs")
    pt.add_argument("--top-n", type=int, default=6)
    pt.add_argument("--run-label", default=None)
    pt.add_argument("--title", default=None)
    pt.add_argument("--study-label", default="")
    pt.add_argument("--conditions-str", default="")
    pt.add_argument("--locus-manifest", default=None,
                    help="Optional manifest.json from loci_real")
    pt.add_argument("--endpoint", default="http://127.0.0.1:11500")
    pt.add_argument("--model", default="batiai/qwen3.6-27b:q6")
    pt.add_argument("--resume-manifest", default=None)
    pt.set_defaults(func=cmd_testrun)

    # prep-subsample
    psub = sub.add_parser("prep-subsample", help="Subsample FASTQs (first N reads) for fast smoke tests")
    g2 = psub.add_mutually_exclusive_group(required=True)
    g2.add_argument("--fastq-dir", help="Directory of FASTQs")
    g2.add_argument("--fastqs", nargs="+", help="Explicit FASTQ paths or globs")
    psub.add_argument("--output-dir", required=True)
    psub.add_argument("--n-reads", type=int, default=100_000,
                      help="Reads per file (default 100,000)")
    psub.set_defaults(func=cmd_prep_subsample)

    # variants
    pv = sub.add_parser("variants", help="Top 4/6/8 fanout + cross-variant synthesis + single HTML")
    pv.add_argument("--output-dir", required=True)
    pv.add_argument("--topn-dir", required=True, help="Dir containing top4.json / top6.json / top8.json (from CombinedNode)")
    pv.add_argument("--motif-tsv", required=True)
    pv.add_argument("--diffbind-csv", required=True)
    pv.add_argument("--deseq-csv", required=True)
    pv.add_argument("--tfs-summary-json", required=True)
    pv.add_argument("--targets-summary-json", required=True)
    pv.add_argument("--conditions-meta-json", required=True)
    pv.add_argument("--Ns", type=int, nargs="+", default=[4,6,8])
    pv.add_argument("--canonical-N", type=int, default=10,
                    help="Which N's narrative becomes the main one in final HTML")
    pv.add_argument("--tf-table-csv", default=None)
    pv.add_argument("--tf-rank-json", default=None)
    pv.add_argument("--trn-json", default=None)
    pv.add_argument("--per-tf-profile-json", default=None)
    pv.add_argument("--volcano-atac-json", default=None)
    pv.add_argument("--volcano-rna-json", default=None)
    pv.add_argument("--no-parallel", dest="parallel", action="store_false")
    pv.add_argument("--no-narrative", action="store_true",
                    help="Skip LLM narrative (smoke testing the chain)")
    pv.add_argument("--run-label", default=None)
    pv.add_argument("--title", default=None)
    pv.add_argument("--study-label", default="")
    pv.add_argument("--conditions-str", default="")
    pv.add_argument("--endpoint", default="http://127.0.0.1:11500")
    pv.add_argument("--model", default="batiai/qwen3.6-27b:q6")
    pv.add_argument("--resume-manifest", default=None,
                    help="Path to a prior manifest.json; cached steps with matching input_hash will be skipped")
    pv.set_defaults(func=cmd_variants, parallel=True)

    # status / debug
    ps = sub.add_parser("status", help="Pretty-print a manifest state")
    ps.add_argument("manifest")
    ps.set_defaults(func=cmd_status)

    pdbg = sub.add_parser("debug", help="Show diagnoses for failed steps in a manifest")
    pdbg.add_argument("manifest")
    pdbg.set_defaults(func=cmd_debug)

    # confirm: sample_sheet -> run_config.toml
    pc = sub.add_parser("confirm", help="Convert sample_sheet.json to run config TOML")
    pc.add_argument("--sheet", required=True, help="sample_sheet.json from `mrag detect`")
    pc.add_argument("--output-config", required=True, help="Where to write the TOML")
    pc.add_argument("--output-dir", required=True, help="Where the run should write outputs")
    pc.add_argument("--run-label", default=None)
    pc.add_argument("--canonical-N", type=int, default=10)
    pc.add_argument("--precomputed", nargs="*", default=[],
                    help="Pre-computed artifact KVs (e.g. atac_md_csv=/path deseq_csv=/path)")
    pc.add_argument("--cancer-type", default=None,
                    help="Cancer/disease context for the LLM narrative (e.g. 'breast cancer')")
    pc.add_argument("--cell-line", default=None,
                    help="Cell line for AlphaGenome (e.g. 'PC-3', 'MCF-7')")
    pc.add_argument("--tissue", default=None,
                    help="Tissue (free text) or ontology CURIE (EFO:/UBERON:) for AlphaGenome")
    pc.add_argument("--organism", default=None,
                    help="Species/genome for the whole run: human|hg38 or mouse|mm10. "
                         "Default: inherit the sample sheet's detected organism.")
    pc.add_argument("--treatment-arm", default=None,
                    help="Which condition is the TREATMENT arm (e.g. 'ONECUT2-OE'). "
                         "conditions[0] is the treatment arm everywhere "
                         "downstream -- fold signs, red/blue track colours, "
                         "GIGGLE labels, and the arm ROSE calls super-enhancers "
                         "on. Default: the detector's order, which is the arm "
                         "seen first in filename order.")
    pc.add_argument("--literature-domains", default=None,
                    help="PaperQA corpora for target evidence, comma separated; several allowed "
                         "(e.g. 'PC, NEC, SCLC' or 'prostate,nec,sclc'). Default: from --cancer-type")
    pc.set_defaults(func=cmd_confirm)

    # run: full downstream from config
    pr = sub.add_parser("run", help="Run the downstream pipeline from a config.toml")
    pr.add_argument("--config", required=True)
    pr.add_argument("--resume-manifest", default=None,
                    help="Path to a previous manifest.json; cached steps with matching input_hash will be skipped")
    pr.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
