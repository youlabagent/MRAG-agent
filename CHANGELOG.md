# Changelog

## Code-availability release (2026-10-27)

This is the first public snapshot of MRAG. It is the code that produced the paper's analyses,
cleaned for release. Except for the changes listed below, the pipeline's behaviour and the report
it writes are unchanged.

### Added

- **License.** MRAG's own code is released under the MIT License (`LICENSE`).
- **Code that ran from outside the code tree.** The repository now includes the following, so that
  the MRAG-written code the pipeline calls is all published:
  - the GIGGLE wrapper `Reference/cistrome/run_giggle.sh`;
  - the peak-heatmap scripts `Reference/peakheatmap/run_peakheatmap*.sh` with their patched
    deepTools copy;
  - the literature retriever `libs/paperqa/mrag_paperqa` and its CLI;
  - reference copies of the original host's Ollama and web UI start scripts, in `host_scripts/`;
  - MRAG's changes to ROSE (a patch and the added `ROSE_stitchTable.py`), in `third_party/ROSE/`.

  The reference data and the literature index stay out of the repository.

### Removed

- **Curated AlphaGenome per-TF step.** This step ran a per-TF AlphaGenome motif-scramble
  counterfactual next to target-gene selection. The report no longer showed its result, so the step
  is gone, along with the fields that only carried it (`ag_summary_top15` and the `ag_*` fields of
  the target-gene cards). AlphaGenome is still used by Section 13.
- **Section-8 LLM interpretation.** Section 8 (HOMER known motifs) no longer makes an LLM call to
  write an interpretation paragraph, which the report no longer displayed. The motif tables are
  unchanged.
- **CLI placeholders.** `mrag dryrun`, `mrag resume` and `mrag rerun` are removed. Resume a run with
  `mrag run --config run.toml --resume-manifest <output_dir>/manifest.json`. The
  `--ag-summary-top4/6/8` options of `mrag variants` are removed too.
- **Old-run compatibility.** The code that read run directories written by earlier internal versions
  is removed. Old `run.toml` files still load: an `[alphagenome]` table is now ignored, and the
  legacy alias of `pipeline.skip_motif_scan` is no longer read (use `skip_motif_scan`). The target
  scorer no longer falls back to a `gate.json` when its parameters name no motif table.
- **Dead code.** This covers unused nodes, helpers, configuration classes, the bundled test-run
  preset, unpublished data files, and the "Test preset" button of the web UI's plot pane. It also
  covers the GeneGPT LLM tool-use loop in `libs/genegpt`, whose NCBI client is deterministic and
  unchanged.
- **`orchestrator/src/mrag_orchestrator/paths.py`.** This module was never imported, so the
  `MRAG_REFERENCE`, `MRAG_VENV`, `MRAG_DATA`, `MRAG_RUNS` and `MRAG_SECRETS` variables it defined
  had no effect. The README lists the variables the code actually reads.

### Fixed

- **Per-replicate HOMER scans on FASTQ runs.** These scans read peak files from the FASTQ source
  directory. They now read the copies staged in the run's own `output/atac_project`, like every other
  step.
- **Manifest writes.** `manifest.json` writes are serialised. Before, the concurrent HOMER scan
  threads could interleave their updates.
- **DAStk bootstrap.** DAStk's differential MD-score p-values are now reproducible. DAStk estimates
  each motif's variance from bootstrap resamples drawn without a seed, so two runs on identical
  input gave different p-values: 300 of 346 motifs differed on the PRAD input, which could reorder
  the TF panel. MRAG now runs the unmodified DAStk script through `dastk_seeded.py`. The wrapper
  seeds each motif's resampling from its name, so two runs on the same input give byte-identical
  results. The rank correlation with an unseeded run was 0.9999.

### Changed

- **Web UI security.** The web UI binds to `127.0.0.1` by default; `MRAG_WEBUI_HOST` sets another
  address. Chat session ids are validated and escaped. Run ids and sample-sheet ids in routes and
  chat actions must name a directory directly under the runs root. The web UI still has no
  authentication.
- **Web UI chat model.** The chat assistant runs on Qwen3.8 27B (`qwen3.8:27b-q8_0`) at the
  target-scorer server (`MRAG_CHAT_ENDPOINT`, `MRAG_CHAT_MODEL`), with a structured reply schema.
  When that server is down, the chat starts it and retries once (`MRAG_CHAT_ENSURE_CMD`).
  `mrag detect --llm-assist` still uses the run's report LLM (Qwen3.6 27B).
- **Text clean-up.** User-interface strings no longer carry internal history or non-paper examples,
  and source comments no longer name internal runs or work packages; otherwise they keep their
  development notes. The report template's HTML comments are template comments now, so a rendered
  report has no HTML developer comments (the JavaScript in it keeps its code comments). The reason
  a skipped or failed Section-13 model arm states no longer ends with an internal tag.
- **Contact page.** The web UI's contact page lists names and the GitHub repository, without
  e-mail addresses.
- **Documentation.** The README is rewritten for users of the code. `NOTICE.md` lists third-party
  code, tools and services.
