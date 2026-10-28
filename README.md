# MRAG

**M**aster **R**egulator analysis of **A**ccessible **G**enome.

MRAG integrates ATAC-seq and RNA-seq from a two-condition experiment (treatment vs control) to rank
the transcription factors (TFs) that drive the difference between the conditions. It builds a
super-enhancer -> TF -> target-gene regulatory network and selects the target genes with
literature evidence. It then tests the super-enhancer -> TF edges in silico with two
sequence-to-function models, AlphaGenome and Corgi, and writes everything into one interactive HTML
report. Human (hg38) and mouse (mm10) data are supported; the in-silico validation is human-only.

> **Status: code-availability snapshot for the paper.** This repository holds the code that produced
> the paper's analyses, cleaned for release. It is not a packaged tool. Paths under `/BIO1/MRAG` are
> the original host's install locations. They are hard-coded in many modules and must be adapted to
> run elsewhere. Several required components are not part of the repository: reference data, the
> literature index, the Corgi runtime and the third-party tools. See [Portability](#portability).

## Contents

- [What MRAG does](#what-mrag-does)
- [Architecture](#architecture)
- [Inputs](#inputs)
- [Outputs: the report](#outputs-the-report)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Quick start](#quick-start)
- [Reproducibility](#reproducibility)
- [Portability](#portability)
- [Repository layout](#repository-layout)
- [Paper datasets](#paper-datasets)
- [License](#license)
- [Citation](#citation)

## What MRAG does

1. **TF ranking.** MRAG first takes differential motif accessibility from ATAC-seq: DAStk MD-scores
   over genome-wide HOCOMOCO v11 motif sites. It then takes the differential expression of each TF's
   own gene from RNA-seq (DESeq2, or limma for log2 matrices). Stouffer's method combines the two
   p-values per TF into a signed Combined Score (Avg.Z). The TF panel is the top `canonical_N`
   (default 10) TFs by Combined Score among those whose differential MD-score is not negative.
2. **Super-enhancers and the network.** ROSE calls super-enhancers on the peaks specific to the
   treatment arm. HOMER, with a merged HOCOMOCO v12 + JASPAR 2024 motif library, places TF motifs in
   the super-enhancers and in the target-gene peaks. Together they give a super-enhancer -> TF ->
   target network (Section 10).
3. **Target genes with literature evidence.** Candidate targets are genes up-regulated in the
   treatment arm whose peaks carry a motif of a panel TF. Each candidate is processed in five steps:
   - **Scoring.** An LLM rates the gene 0-10 for the run's literature domain(s). It reads the gene's
     NCBI GeneRIFs and passages retrieved from a local literature index.
   - **Evidence gate.** A deterministic gate caps any score that no retrieved evidence naming the
     gene supports.
   - **Evidence rules.** Rules drop evidence that does not count, for example papers about
     neuroendocrine tumours only, or negative results in multi-gene screens.
   - **PubMed check.** A live PubMed check can restore a capped gene.
   - **Audit.** The LLM audits the remaining evidence. By default it flags doubtful evidence and
     removes nothing.

   If NCBI cannot be read for a gene that could become a target, the step stops rather than rank
   that gene on missing evidence.
4. **In-silico validation of SE -> TF edges (Section 13).** For each super-enhancer -> TF edge, the
   TF's JASPAR motifs are scrambled inside the super-enhancer and compared with a size-matched
   control edit in the same window. AlphaGenome (remote API) and Corgi (local GPU model) predict the
   effect of each edit on the enhancer's chromatin signal. The test is at the pair level: Fisher's
   exact test against lineage-foreign control factors put through the same construction, with Holm
   correction across the headline rows. The TF -> target step is measured as well and reported in
   the section's details. In the network figure, TF -> target edges are drawn as motif-based
   candidates, not as validated links.
5. **Report.** A single HTML file with 15 sections (see [Outputs](#outputs-the-report)) and the run
   manifest. The manifest records each step's status, input hash, parameters and output files.

## Architecture

```
orchestrator/webui/   FastAPI web UI and chat assistant (optional)
orchestrator/src/     mrag CLI; runner.py builds and runs a fixed DAG of steps (nodes/)
mcps/                 10 analysis packages, each a FastMCP server with its tools
libs/                 NCBI client (GeneGPT-derived), NCBI credentials, AlphaGenome client, track plots
```

- **Deterministic orchestrator.** `runner.py` runs a fixed sequence of steps. Which steps run
  depends on the input stage that was supplied (FASTQ, peaks or matrices), on `run.toml` and on the
  organism. An LLM never decides it.
- **Stages are MCP tools.** The analysis code lives in ten MCP packages under `mcps/`. The
  orchestrator calls their tools as MCP `tools/call` requests through an MCP client session. The
  transport is in-memory, inside the same process (`tool_transport.py`): no stdio, no socket. Every
  call is checked, and the tool must receive exactly the arguments the node passed.
  `MRAG_TOOL_TRANSPORT=direct` calls the same functions without MCP.
- **Stdio servers.** Each package's `server.py` also serves its tools over stdio, 114 tools in total,
  for use from any MCP client. The pipeline does not need these servers running.

| Package | stdio server | Tools | What it provides |
|---|---|---|---|
| `mcps/ATAC_mcp` | `atac-mcp` | 21 | ATAC FASTQ pipeline (Trim Galore, bowtie2, filtering and duplicate removal, MACS2, annotation), DAStk differential MD-scores (seeded), ROSE super-enhancers from BAMs or bigWigs, GIGGLE / Cistrome overlap, condition-specific peak sets |
| `mcps/RNA_mcp` | `rna-mcp` | 22 | RNA FASTQ pipeline (Trim Galore, STAR, RSEM), DESeq2 (R by default, pydeseq2 optional), limma for log2 matrices, QC and plots |
| `mcps/combined_mcp` | `combined-mcp` | 5 | Stouffer combination of the ATAC and RNA p-values, top-N TF selection |
| `mcps/diffbind_mcp` | `diffbind-mcp` | 6 | DiffBind (R), voom differential accessibility for peak-count matrices |
| `mcps/homer_mcp` | `homer-mcp` | 12 | HOMER motif finding and annotation, replicate min-max joins, super-enhancer motif scan |
| `mcps/trn_mcp` | `trn-mcp` | 13 | TF-target tables, candidate pool and motif scan, LLM scoring with the evidence gate, re-scoring, final selection |
| `mcps/paperqa_mcp` | `paperqa-mcp` | 10 | Literature retrieval (BGE-M3 embeddings, Qdrant), PubMed check, index administration |
| `mcps/enrichment_mcp` | `python -m enrichment_mcp.server` | 4 | Hallmark, GO BP and Reactome enrichment (Enrichr libraries via gseapy) |
| `mcps/viz_report_mcp` | `viz-report-mcp` | 15 | Figures, networks, locus tracks, report text and HTML rendering |
| `mcps/alphagenome_mcp` | `alphagenome-mcp` | 6 | AlphaGenome counterfactual predictions and cache tools. Standalone only: Section 13 calls the AlphaGenome SDK directly |

**LLM roles.** Both models are served by Ollama. The code uses both Ollama's native `/api/chat` and
its OpenAI-compatible `/v1/chat/completions` routes.

| Model | Default server | Used for |
|---|---|---|
| Qwen3.8 27B (`qwen3.8:27b-q8_0`, reasoning effort `low`) | a dedicated Ollama server, `http://127.0.0.1:11501` | Target-gene evidence scoring and re-scoring, the evidence audit, the lineage-relevance judgement of Section-11 enrichment terms, and the web UI chat assistant |
| Qwen3.6 27B (`batiai/qwen3.6-27b:q6`) | Ollama at `http://127.0.0.1:11500` (`[llm]` in `run.toml`) | Report text (Section 1 summary, Section 14 interpretation and target-gene cards), and grouping of samples into arms in `mrag detect` and `mrag sra-lookup` |

The web UI chat assistant chooses among chat actions defined in `orchestrator/webui/chat.py`:
detect samples, start a run, show status, open a report, add a locus track, and SRA lookup and
download. These actions call the `mrag` CLI. They are not MCP tools.

**Step order** (`runner.py`; each step is skipped when its inputs are missing, and the skip is recorded):

1. Organism and genome decision, then a pre-check of the target-scorer LLM.
2. Per modality, from the earliest stage supplied:
   - ATAC FASTQ: alignment, then peaks, then DAStk.
   - ATAC peaks: DAStk.
   - ATAC count matrix: voom, then condition peak sets, then DAStk.
   - RNA FASTQ: STAR and RSEM, then DESeq2.
   - RNA counts or TPM matrix: DESeq2.
   - RNA log2 matrix: limma.
3. Combined TF ranking (`combined/INTEGRATED.csv`, top-N panel).
4. HOMER annotation of the consensus peaks, DiffBind, and per-replicate HOMER scans with min-max
   joins.
5. Report sections that come straight from the peaks:
   - HOMER known motifs per condition (Section 8)
   - GIGGLE TF ChIP-seq overlap (Section 9)
   - peak heatmaps (Section 6)
6. Super-enhancers (ROSE) and their motifs.
7. Target-gene selection with literature evidence, the TRN and the super-enhancer network
   (Section 10), and functional enrichment (Section 11).
8. Locus snapshots (Section 12).
9. In-silico validation (Section 13). The two models run concurrently, about 1.5 h on the paper's
   datasets.
10. Plots, report text and rendering.

## Inputs

A comparison of two conditions, each with ATAC-seq and RNA-seq. `conditions[0]` is the **treatment**
arm and `conditions[1]` the **control**; the treatment arm is listed first throughout. Use
`mrag confirm --treatment-arm` to set the order. Each modality can enter at any of these stages:

| Modality | Accepted input | Enters at |
|---|---|---|
| ATAC | paired- or single-end FASTQ per replicate | full pipeline |
| ATAC | per-arm peak files (BED / narrowPeak) | DAStk |
| ATAC | peak raw-count matrix (`seqnames start end ...` + sample columns) plus a `sample<TAB>group` TSV | voom differential accessibility, then DAStk |
| ATAC | DAStk differential MD-score table (`precomputed.atac_md_csv`) | combined ranking |
| RNA | FASTQ per replicate | full pipeline |
| RNA | raw-count or TPM gene matrix | DESeq2 |
| RNA | log2 / normalised gene matrix plus a `sample<TAB>group` TSV | limma |
| RNA | DESeq2 result table (`precomputed.deseq_csv`) | combined ranking |
| both | `INTEGRATED.csv` (`precomputed.integrated_csv`) | after the combined ranking |

`mrag detect` sniffs a directory and recognises FASTQ files, peak files, count / TPM / log2 matrices,
group TSVs and DESeq2 result tables. Set the other inputs with
`mrag confirm --precomputed key=/path` (for example `atac_md_csv=/path/md_scores.txt`) or in
`[precomputed]`. A `bigwigs/` directory next to an ATAC
count matrix, holding files named after its sample columns, enables the per-TF profiles, the locus
snapshots and bigWig-based super-enhancer calling for matrix input. Without per-replicate BAMs or
bigWigs, the sections that need them are omitted and the report says so.

## Outputs: the report

A run writes `<output_dir>/mrag_report.html` and `<output_dir>/manifest.json`, plus one directory per
step: `combined/`, `homer_motif/`, `curated_targets/`, `trn_cytoscape/`, `se_rose/`, `enrichment/`,
`axis_validation/` and others. Figures are embedded in the HTML. The interactive libraries (Plotly,
Cytoscape.js, jQuery, DataTables) are loaded from public CDNs, so the interactive parts need a
network connection when the report is opened.

| # | Section heading |
|---|---|
| 1 | Executive Summary |
| 2 | Pipeline overview |
| 3 | TF Ranking (Top 10 by Combined Score (Avg.Z)) |
| 4 | Differential Analysis |
| 5 | Rank & Combined Score volcano |
| 6 | Peak Heatmap (Condition-Specific / Shared Peaks) |
| 7 | Top-Rank Gene Expression Heatmap |
| 8 | HOMER Known Motif Enrichment |
| 9 | GIGGLE - TF ChIP-seq Overlap (Cistrome DB) |
| 10 | TRN Graph (Super Enhancer -> TF -> target gene), or (bipartite TF -> target gene) without a super-enhancer layer |
| 11 | Functional Enrichment of TF Target Programs |
| 12 | Locus Snapshots - Key TF -> Target Examples |
| 13 | In silico Validation (Enhancer -> TF) |
| 14 | Result Interpretation (with the target-gene evidence cards) |
| 15 | Manifest |

A section that could not be computed states why: no input for it, a step that failed, or a step
this organism cannot run. It is never left silently empty.

## Requirements

The versions in the note at the end of this section are those of the original host. They are a
reference, not a tested minimum.

- **Linux, Python 3.11** in one virtual environment, with these packages:
  - `pydantic`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`,
    `jinja2`, `Pillow`, `pyfaidx`, `pyBigWig`, `pydeseq2`, `gseapy`, `tomli_w`
  - `qdrant-client`, `sentence-transformers` and `torch` (BGE-M3 embeddings)
  - `fastapi`, `uvicorn` and `python-multipart` (web UI)
  - **`mcp` 1.27.1 exactly.** The in-memory transport refuses another `mcp` version; use
    `MRAG_TOOL_TRANSPORT=direct` otherwise.
  - `fastmcp` 3.x (enrichment server)
- **R 4.3 with Bioconductor**: DESeq2, DiffBind, BiocParallel, limma, edgeR, org.Hs.eg.db, plus
  optparse, pheatmap, ggplot2, ggrepel, RColorBrewer, gplots, dplyr and readr.
- **Command-line tools**:
  - Trim Galore with cutadapt, FastQC, bowtie2 (ATAC), STAR and RSEM (RNA)
  - samtools, bedtools, htslib (`tabix`, `bgzip`), Picard (Java) for duplicate removal
  - MACS2 and pyGenomeTracks
  - deepTools: `bamCoverage`, plus `bigwigAverage`, which needs deepTools 3.5 or newer, and
    `computeMatrix`. `plotHeatmap` runs from the patched deepTools 3.3.2 copy in
    `Reference/peakheatmap/deeptools_patched/`
  - HOMER with the hg38 / mm10 genome packages
  - DAStk (`process_atac.py`, `differential_md_score.py`) with its HOCOMOCO v11 motif-site BEDs
  - ROSE: the Barski-lab fork with MRAG's changes from `third_party/ROSE/`, run with Python 2
  - GIGGLE with a Cistrome DB TF ChIP-seq index
  - sra-tools `fasterq-dump` and `curl` (only for `mrag sra-fetch`)
- **Reference data** (not distributed), under `Reference/` by default:
  - genome FASTA, bowtie2, STAR and RSEM indexes, GTFs
  - the HOCOMOCO v12 + JASPAR 2024 HOMER motif library
  - the Cistrome GIGGLE index. Its wrapper `Reference/cistrome/run_giggle.sh` is in the repository
  - mouse-to-human ortholog tables for mm10
  - the NCBI `Homo_sapiens.gene_info.gz` dump
- **LLMs**: Ollama serving `qwen3.8:27b-q8_0` and `batiai/qwen3.6-27b:q6`, one GPU each on the
  original host. That host's scorer start script requires 32,000 MiB of free GPU memory before it
  loads the 27B q8 model.
- **Literature index**: a local Qdrant collection of paper chunks embedded with BGE-M3 and tagged
  by 19 cancer-type corpora. It is not in this repository. The retriever that reads it,
  `libs/paperqa/mrag_paperqa`, and its CLI `libs/paperqa/scripts/retrieve_cli.py` are.
- **NCBI E-utilities** (PubMed, Gene). An API key is optional but strongly recommended.
- **AlphaGenome**: the `alphagenome` SDK in its own venv (`libs/alphagenome_client`) and an API key.
  The API is offered for non-commercial use under its terms.
- **Corgi**: the Corgi package and its venv with `torch` and `flash-attn`, which needs an Ampere or
  newer NVIDIA GPU, plus the published checkpoint and an expression-context table
  (`Reference/corgi/expr_contexts.txt` and `expr_vectors.npy`, one 2,891-gene trans-regulator
  vector per context). Two of its rows were built for the paper's datasets: `LNCaP_OE` for OC2 and
  `PRAD_AR_high_mean`, the mean of the 12 AR_high TCGA-PRAD RNA samples. None of this is in the
  repository; the code expects it under `libs/corgi_client/` and `Reference/corgi/`.
- Network access for Enrichr, NCBI, ENA/SRA, the AlphaGenome API and the report's CDN scripts.

Versions on the original host:

- Python side: Python 3.11.15, mcp 1.27.1, fastmcp 3.3.1, pydantic 2.13, pandas 2.3, numpy 2.4,
  gseapy 1.2.1, pydeseq2 0.5.4.
- R side: R 4.3.3, DESeq2 1.42.1, DiffBind 3.12.0, limma 3.58.1, edgeR 4.0.16.
- Command-line tools: STAR 2.7.11b, RSEM 1.3.1, Trim Galore 0.6.11, cutadapt 2.8, FastQC 0.11.9,
  bowtie2 2.3.5.1, samtools 1.17, bedtools 2.27.1, MACS2 2.2.9.1, deepTools 3.3.2 and 3.5.6,
  pyGenomeTracks 3.9, DAStk 1.0.1r1.
- Models: AlphaGenome SDK 0.6.1. The Corgi venv has torch 2.6.0 (CUDA 12.4) and flash-attn 2.7.4.

## Configuration

### run.toml

`mrag confirm` writes `run.toml` from the sample sheet. The schema is `RunConfig` in
`orchestrator/src/mrag_orchestrator/config.py`. Top-level keys must come before the first `[table]`
header. pydantic ignores a key it does not know without a warning, so check the spelling.

```toml
schema_version = "1.0"
run_label   = "oc2_example"
output_dir  = "/path/to/runs/oc2_example/output"
organism    = "human"              # human|hg38 or mouse|mm10; unset = detected from metadata and data
canonical_N = 10                   # TF panel size; another value needs MRAG_ALLOW_CONFIG_DRIFT=1
cancer_type = "prostate cancer"    # free text; frames the scoring and the report text
cell_line   = "LNCaP"              # context given to the scorer
tissue      = "prostate"
literature_domains  = ["prostate"] # literature corpora; "pancancer" is always added
evidence_gate_scope = "domain"     # all | literature | domain (unset = MRAG_TARGET_EVIDENCE_SCOPE, else "all")
curated_targets     = true         # literature-evidence target selection (the default path)
# axis_headline = "both"           # Section 13 leading model set: both | alphagenome | corgi; unset = default rule
# axis_ar_gate  = false            # Section 13 AR cross-condition row (bulk prostate adenocarcinoma cohorts)
# report_title = "..."             # optional report header

conditions = [                     # conditions[0] = treatment
  { label = "ONECUT2-OE", n_reps = 4, modality_counts = { ATAC = 2, RNA = 2 } },
  { label = "MYC-DDK",    n_reps = 4, modality_counts = { ATAC = 2, RNA = 2 } },
]

[[samples]]                        # one block per FASTQ (pair); omit for matrix input
name      = "OC2_OE_rep1"
fastq_r1  = "/path/to/OC2_OE_rep1_R1.fastq.gz"
fastq_r2  = "/path/to/OC2_OE_rep1_R2.fastq.gz"   # omit for single-end
modality  = "ATAC"                 # ATAC | RNA
arm       = "ONECUT2-OE"
replicate = 1

[pipeline]
topn_variants   = [10]             # canonical_N is added if missing
skip_motif_scan = false            # per-replicate HOMER scans and min-max joins
se_analysis     = true             # super-enhancer layer
se_top_n        = 10
# gtf_path = ""                    # empty = the organism's GTF
# homer_genome = ""                # empty = the organism's HOMER genome

[llm]                              # report text and detect assist
endpoint = "http://127.0.0.1:11500"
model    = "batiai/qwen3.6-27b:q6"

[paperqa]                          # literature retriever (host paths by default)
# retriever_script = ".../libs/paperqa/scripts/retrieve_cli.py"
# venv_python      = ".../bin/python3"

[precomputed]                      # processed inputs instead of FASTQ, for example:
# rna_log2_matrix   = "/path/RNA_log2.tsv"
# rna_group_tsv     = "/path/rna_groups.tsv"
# atac_count_matrix = "/path/ATAC_counts.txt"
# atac_group_tsv    = "/path/atac_groups.tsv"
# also: atac_md_csv, deseq_csv, integrated_csv, rna_counts_matrix + rna_quant_source,
#       atac_peaks_treatment / atac_peaks_control, bigwig_arm_a_paths / bigwig_arm_b_paths
```

`mrag confirm` sets `cancer_type`, `cell_line`, `tissue`, `organism`, `literature_domains`,
`canonical_N` and the treatment arm from options. Add `evidence_gate_scope`, `axis_headline` and
`axis_ar_gate` by hand; the web UI's confirm form also has fields for the two `axis_*` keys. Only a
bulk prostate-adenocarcinoma cohort declares `axis_headline = "alphagenome"` and
`axis_ar_gate = true`. Other runs keep the default rule, in which the two-model consensus leads.

### Environment variables

These are the variables most installations need. Each one is read by the code, and the default is
the code's own default. Runs started from the web UI inherit the web UI's environment.

| Variable | Default | Effect |
|---|---|---|
| `MRAG_TOOL_TRANSPORT` | `mcp` | `mcp`: stage tools are called through MCP over the in-memory transport. `direct`: plain function calls. Recorded in `manifest.json` |
| `MRAG_SCORER_PROFILE` | `q38` | `q38`: `qwen3.8:27b-q8_0`, reasoning effort `low`, at `http://127.0.0.1:11501`. `legacy`: the `[llm]` model of `run.toml` |
| `MRAG_SCORE_MODEL`, `MRAG_SCORE_ENDPOINT`, `MRAG_SCORE_REASONING_EFFORT` | from the profile | Override one part of the scorer profile |
| `MRAG_SCORE_ENSURE_CMD` | the original host's scorer start script (q38 profile, default endpoint only) | Command run before scoring to start the scorer server. Elsewhere set it to `off` and start the server yourself |
| `MRAG_SCORE_PREFLIGHT` | on | `0` skips the scorer checks at run start and before scoring |
| `MRAG_CHAT_ENDPOINT`, `MRAG_CHAT_MODEL` | `http://127.0.0.1:11501`, `qwen3.8:27b-q8_0` | Web UI chat model |
| `MRAG_CHAT_ENSURE_CMD` | the original host's scorer start script | Started when the chat server refuses connections. `off` disables it |
| `OLLAMA_ENDPOINT`, `MRAG_MODEL` | `http://127.0.0.1:11500`, `batiai/qwen3.6-27b:q6` | Web UI: the LLM for `mrag detect` arm grouping (on unless `--no-llm-assist`) and new chat sessions |
| `MRAG_WEBUI_HOST`, `MRAG_WEBUI_PORT` | `127.0.0.1`, `8765` | Web UI bind address. `0.0.0.0` serves every interface; there is no authentication |
| `MRAG_WEBUI_RUNS_ROOT` | `orchestrator/webui/runs` | Where the web UI creates and lists runs |
| `MRAG_BIN` | `/BIO1/MRAG/venvs/paper_qa/bin/mrag` | The `mrag` executable the web UI and chat call |
| `ALPHAGENOME_API_KEY` | unset | AlphaGenome API key. If unset, the key is read from `ALPHAGENOME_KEY_FILE` (default `/BIO1/MRAG/secrets/.alphagenome_key`) |
| `NCBI_API_KEY`, `NCBI_EMAIL`, `NCBI_TOOL` | unset | NCBI E-utilities identity. If unset, read from the file `MRAG_NCBI_ENV` (default `~/.config/mrag/ncbi.env`, mode 600). The key is masked in logs |
| `MRAG_REFERENCE_ROOT` | `/BIO1/MRAG/Agent/Reference` | Root of the reference data the organism profiles use |
| `HOMER_HOME` | first host install with `data/genomes/hg38` | HOMER installation |
| `MRAG_CPU_LIST` | `32-47` | CPUs a run is pinned to. `none` disables pinning |
| `MRAG_GPU_LIST` | GPUs 0 then 2 (GPU 1 is never leased) | GPUs the Corgi arm may lease. Set but empty switches the Corgi arm off |
| `MRAG_MEM_START_FLOOR_GB` | `60` | Refuse to start a run below this much free memory. `0` disables the check |
| `MRAG_TARGET_EVIDENCE_SCOPE` | `all` | Evidence-gate scope when `run.toml` sets none |
| `MRAG_EVIDENCE_AUDIT_MODE` | `flag` | `flag`, `remove_named`, `remove` or `off` |

<details>
<summary>All environment variables read by the code</summary>

The variables below are set by MRAG itself and should not be set by hand: `MRAG_RUN_ROOT`,
`MRAG_NCPUS`, `MRAG_GPU_MEMORY_FRACTION`, and `CUDA_VISIBLE_DEVICES` for the Corgi child process.

**Paths and installs**

| Variable | Default | Effect |
|---|---|---|
| `MRAG_ROSE_DIR` | `/BIO1/MRAG/Agent/bin/tools/ROSE` | ROSE installation |
| `MRAG_ORTHOLOGY_DIR` | `/BIO1/MRAG/Agent/Reference/orthology` | Mouse-to-human ortholog tables |
| `MRAG_CISTROME_DIR` | `$MRAG_REFERENCE_ROOT/cistrome` | Where the presence of the GIGGLE index is checked. The wrapper script itself uses absolute paths under `/BIO1/MRAG/Agent` |
| `MRAG_HOME` | the repository root | Locates `libs/track_helper` for the locus snapshots |
| `MRAG_GENE_INFO`, `MRAG_HUMAN_GENE_INFO` | `/BIO1/MRAG/MD_check/Homo_sapiens.gene_info.gz` | NCBI gene_info dump: official names, synonyms, IDs |
| `MRAG_GENEGPT_DIR` | `/BIO1/MRAG/Agent/libs/genegpt` | NCBI client module |
| `MRAG_ORCHESTRATOR_SRC`, `MRAG_ORCH_SRC` | `/BIO1/MRAG/Agent/orchestrator/src` | Where the MCP packages import orchestrator modules from |
| `MRAG_EVIDENCE_GATE_PY`, `MRAG_NARRATIVE_PY` | their files under `/BIO1/MRAG/Agent` | Module files loaded by path |
| `MRAG_SYMBOL_SENSE` | `/BIO1/MRAG/Agent/orchestrator/configs/symbol_sense.json` | Gene-symbol sense table used by the evidence gate |
| `MRAG_HOMER_CANON_CACHE` | `/BIO1/MRAG/Agent/cache/homer_canonical` | Cache of HOMER annotations of matrix-input peak catalogues |
| `MRAG_HOMER_SLOT_DIR` | `/BIO1/MRAG/Agent/run/homer_slots` | Lock files that bound concurrent HOMER annotation memory |
| `MRAG_GPU_LOCK_DIR`, `MRAG_MEM_GUARD_DIR` | `/BIO1/MRAG/Agent/orchestrator/webui/runs` | GPU-lease lock files; memory-watcher state |
| `HOMER_PERL` | `perl` on `PATH` | Perl for HOMER |
| `DIFFBIND_RSCRIPT` | `Rscript` on `PATH` | R for DiffBind |
| `ALPHAGENOME_KEY_FILE` | `/BIO1/MRAG/secrets/.alphagenome_key` | AlphaGenome key file |
| `ALPHAGENOME_CACHE_DIR` | `/BIO1/MRAG/alphagenome_client/.cache` | Response cache of `alphagenome_client` |
| `ATAC_WORKDIR`, `RNA_WORKDIR` | `/BIO1/MRAG` | Working directory of the stdio `atac-mcp` / `rna-mcp` servers |
| `ATAC_MCP_PYTHON`, `RNA_MCP_PYTHON` | a host-specific Python 3 | Interpreter for those servers' sample-detection helper tools |
| `MRAG_EXTRA_DATA_ROOT` | a host-specific directory | One more data root the chat and file browser may use. Empty means none |
| `HOMER_MINMAX_CACHE`, `MRAG_PY` | host paths | Used by `orchestrator/new_run_from.sh` |
| `MRAG_GIGGLE_GENOME` | `hg38` | Genome of `Reference/cistrome/run_giggle.sh` when no genome argument is passed (the pipeline passes it) |

`host_scripts/ensure_scorer_ollama.sh` reads its own `SCORER_*` variables (port, model, GPUs,
context length, keep-alive, memory thresholds); see the script.

**LLM services**

| Variable | Default | Effect |
|---|---|---|
| `MRAG_SCORE_ENSURE_WAIT_S` | `1800` (default start script only) | How long scoring waits for a GPU that can take the scorer |
| `MRAG_SCORE_ENSURE_POLL_S` | `60` | Poll interval of that wait |
| `MRAG_SCORE_ENSURE_TIMEOUT_S` | `900` | Timeout of one start-script call |
| `MRAG_SCORE_MAX_UNSCORED_FRAC` | `0.02` (`1` in the legacy profile) | Fail the step when more genes than this fraction were never scored |
| `MRAG_SCORE_DOMAIN_FRAME` | `literature` | `study` scores for the experimental system instead of the literature domain |
| `MRAG_SCORE_NEC_NOTE` | on (off in the legacy profile) | Scoring note that small-cell, LCNEC, NEPC and Merkel count as neuroendocrine carcinoma |
| `MRAG_CURATED_SCORE_TIMEOUT_S` | unset | Can only lengthen the 3 h limit of the scoring step |
| `MRAG_ENRICH_ENDPOINT`, `MRAG_ENRICH_MODEL`, `MRAG_ENRICH_EFFORT` | the `MRAG_SCORE_*` values, else the q38 defaults | Model for the Section-11 lineage judgement |
| `MRAG_EMBED_DEVICE` | `cuda:0` (falls back to CPU) | Device of the BGE-M3 embedder in the `paperqa-mcp` index tools |

**Target-gene evidence**

| Variable | Default | Effect |
|---|---|---|
| `MRAG_TARGET_EVIDENCE_GATE` | on | Evidence-grounding gate |
| `MRAG_TARGET_EVIDENCE_CAP` | `2.0` | Score ceiling for a gene that no retrieved evidence names |
| `MRAG_TARGET_EVIDENCE_PUBMED` | on | Live PubMed check of capped genes |
| `MRAG_TARGET_EVIDENCE_PUBMED_RETMAX` | `20` (`5` with the evidence layers off) | PubMed records read per gene and domain |
| `MRAG_EVIDENCE_DEFENCE`, `MRAG_EVIDENCE_AUDIT` | on | `0` (either one) turns every evidence layer off: rules 1a-1c, alias GeneRIFs, re-scoring and the audit |
| `MRAG_NEC_NET_PAPER`, `MRAG_NEC_VETO_SENTENCE`, `MRAG_LIST_NEGATIVE` | on | Rule 1a (paper about neuroendocrine tumours only), rule 1b (sentence names such a tumour and no carcinoma), rule 1c (negative multi-gene screen) |
| `MRAG_NEC_RULE` | unset | `legacy` restores the older text rule of the `nec` domain |
| `MRAG_GENERIF_ALIAS` | on | A GeneRIF line that names the gene by an alias counts |
| `MRAG_RESCORE_AFTER_FILTER` | on | Re-score a gene in its original batch after it lost evidence |
| `MRAG_NAMES_GENE_STRICT` | on | Strict test that a passage names this gene and not a gene with a colliding symbol |
| `MRAG_CORPUS_TIE_TEXT` | `sclc` | Domains whose corpus-labelled passages also need a text match |
| `MRAG_CORPUS_TIE_VETO` | on | Domain veto patterns |
| `MRAG_DOMAIN_TIE_COOC` | `sclc` | Domains that need the gene and the domain in the same sentence |
| `MRAG_RESCUE_COOC` | `sclc,prostate` | The same rule for the PubMed check |
| `MRAG_EVIDENCE_AUDIT_KEEP` | unset | Genes the audit may flag but never remove |
| `MRAG_EVIDENCE_AUDIT_LIMIT` | `40` | Most genes the audit examines |
| `MRAG_EVIDENCE_AUDIT_BUDGET_S`, `MRAG_EVIDENCE_AUDIT_TIMEOUT_S`, `MRAG_EVIDENCE_AUDIT_WORKERS` | `600`, `180`, `2` | Audit time budget, per-call timeout, parallel requests (at most 2) |
| `MRAG_EVIDENCE_AUDIT_FAIL` | `open` | `open`: a gene whose audit call failed is kept as unaudited. `closed`: it is skipped while the endpoint is otherwise healthy |
| `MRAG_EVIDENCE_AUDIT_CACHE` | the run's own cache file | Audit cache |
| `MRAG_CURATED_CARD_RETRIEVAL` | on | Retrieve the scorer's literature with the per-domain, gene-filtered card retriever |
| `MRAG_CURATED_CARD_TOP_K` | `3` | Passages per corpus |
| `MRAG_CURATED_GENERIF_RETMAX` | `400` | GeneRIF lines fetched per gene |
| `MRAG_GENERIF_SHOW_CHARS` | `600` | Characters of each GeneRIF line shown to the scorer |
| `MRAG_GENERIF_FAIL_ON_TOP`, `MRAG_RESCUE_FAIL_ON_TOP` | on | Stop when NCBI cannot be read for a gene that could become a target. `0` flags the gene and continues |
| `MRAG_GENERIF_MAX_FAILED_FRAC` | `0.02` | Also stop when this fraction of all GeneRIF fetches failed (`1` = never) |
| `MRAG_GENERIF_BREAKER`, `MRAG_GENERIF_BUDGET_S`, `MRAG_GENERIF_RETRY_PAUSE_S`, `MRAG_RESCUE_RETRY_PAUSE_S` | `5`, `2400`, `30`, `10` | GeneRIF fetch circuit breaker, time budget and retry pauses |

**NCBI client and report cards**

| Variable | Default | Effect |
|---|---|---|
| `MRAG_NCBI_TRIES`, `MRAG_NCBI_BACKOFF_S`, `MRAG_NCBI_BACKOFF_MAX_S`, `MRAG_NCBI_CALL_DEADLINE_S` | `6`, `1.0`, `30`, `180` | Retries with exponential backoff per NCBI call |
| `MRAG_NCBI_SHARED_THROTTLE` | on | One NCBI rate limit across all MRAG processes of the user. `0` limits per process |
| `MRAG_NCBI_THROTTLE_FILE` | `/dev/shm/mrag_ncbi_throttle.<uid>` | Shared throttle file |
| `MRAG_GENEGPT_ESEARCH_FALLBACK` | `1` | Allow a verified Entrez esearch when the local gene_info dump cannot resolve a symbol |
| `MRAG_GENE_FULLNAME_OFFLINE`, `MRAG_GENE_FULLNAME_LIVE_FALLBACK` | on | Gene full names for the cards from the local dump, then from NCBI |
| `MRAG_CARD_RETRIEVER_TIMEOUT_S` | unset | Can only lengthen the card retriever's timeout |
| `MRAG_CARD_MASK_WORKERS`, `MRAG_CARD_PLAIN_CACHE` | `0`, off | Speed-ups of the card retriever that give the same results |

**Run behaviour and resources**

| Variable | Default | Effect |
|---|---|---|
| `MRAG_ALLOW_CONFIG_DRIFT` | unset | Allow a `canonical_N` other than the default |
| `MRAG_REUSE_ATAC_MD_CSV` | unset | Path of an existing DAStk MD-score table to use instead of running DAStk |
| `MRAG_REUSE_EXTERNAL` | unset | `1`: reuse enrichment results from a previous manifest backup instead of calling Enrichr |
| `MRAG_DEMO_PEAKHEATMAP`, `MRAG_DEMO_ATAC_BW`, `MRAG_DEMO_HOMER_KNOWN` | unset | Development fallbacks to a demo experiment. Leave unset |
| `MRAG_CURL_ARGS` | empty | Extra `curl` arguments for `mrag sra-fetch` |
| `MRAG_PAUSE_AWARE_TIMEOUTS` | `1` | Subprocess timeouts do not count time a run spent paused by the memory watcher |
| `MRAG_MEM_GUARD`, `MRAG_MEM_GUARD_ARGS`, `MRAG_MEM_GUARD_OOM_ADJ` | `1`, empty, `0` | Memory watcher on/off, extra arguments, its own `oom_score_adj` |
| `MRAG_MEM_SOFT`, `MRAG_MEM_HARD` | `0.20`, `0.30` | Watcher lines for this account's memory use, as a fraction of RAM (a value above 1 is read as GB): warn, then pause the fastest-growing run |
| `MRAG_MEM_RESUME`, `MRAG_MEM_ON_STUCK` | `0.85`, `hold` | Resume below this fraction of the hard line; what to do with a run paused too long |
| `MRAG_MEM_SYS_FLOOR_GB`, `MRAG_MEM_SYS_RESUME_GB`, `MRAG_MEM_SYS_ON_STUCK` | `40`, `80`, `hold` | Also pause while the machine's free memory is below the floor (`0` = off) |
| `MRAG_MEM_IGNORE_SYSFLOOR` | unset | `1` skips the start-time memory check |
| `MRAG_HOMER_TOTAL_GB`, `MRAG_HOMER_ANNOT_MAX_GB`, `MRAG_HOMER_RAM_PER_WORKER_GB` | `160`, `40`, unset | HOMER annotation memory budget, total and per worker |
| `MRAG_HOMER_CHUNKING`, `MRAG_HOMER_CHUNK_BP`, `MRAG_HOMER_MAX_PARALLEL` | `1`, `0` (auto), `0` (auto) | Chunked HOMER annotation |
| `MRAG_HOMER_GIB_BASE`, `MRAG_HOMER_GIB_PER_MB_PER_1K`, `MRAG_HOMER_SLOT_POLL_S` | `5.2`, `0.95`, `5` | HOMER memory model and slot polling |
| `MRAG_HOMER_TIMEOUT_S` | `7200` | Timeout of one per-replicate HOMER scan |
| `MRAG_FILTER_CHUNK_ROWS` | `64` | Row chunk of the HOMER target-TF filter |
| `MRAG_PICARD_XMX_MAX_GB` | `16` | Upper bound of Picard's Java heap |

</details>

## Quick start

```bash
# Install from the checkout. Editable installs, because mrag_axis (Section 13) is found
# through the source tree
pip install -e orchestrator -e mcps/enrichment_mcp

# FASTQ (or processed files) -> sample sheet: modality, arm and replicate of every file.
# Arm grouping uses the report LLM; --no-llm-assist uses keyword rules only.
mrag detect  --fastq-dir /data/my_study -o sheet.json

# Sample sheet -> run.toml
mrag confirm --sheet sheet.json --output-config run.toml --output-dir runs/my_study/output \
             --treatment-arm ONECUT2-OE --cancer-type "prostate cancer" \
             --cell-line LNCaP --tissue prostate --literature-domains prostate

# Run; exits non-zero on failure
mrag run --config run.toml

# Resume: steps whose recorded input hash is unchanged are skipped
mrag run --config run.toml --resume-manifest runs/my_study/output/manifest.json

mrag status runs/my_study/output/manifest.json   # step states
mrag debug  runs/my_study/output/manifest.json   # diagnoses of failed steps
```

Start `mrag run` and the web UI from a scratch working directory. HOMER writes temporary
`.seq` / `.pos` / `.tmp` files into the current directory.

The other subcommands are `sra-lookup`, `sra-fetch`, `sra-status` and `sra-cancel`. They expand an
SRA/ENA/GEO accession to its study, judge whether MRAG can use it, then download it, report progress
or cancel. `prep-subsample` writes the first N reads of each FASTQ for smoke tests, `testrun` renders
a report from precomputed artifacts, and `variants` runs a top-4/6/8 panel fan-out. Run `mrag <cmd> -h`
for the options. `mrag` is the console script of `orchestrator/pyproject.toml`.

**Web UI.** The web UI is optional:

```bash
MRAG_WEBUI_HOST=127.0.0.1 MRAG_WEBUI_PORT=8765 python orchestrator/webui/app.py
```

`host_scripts/start_mrag_webui.sh` is the original host's start script. It shows the full set of
variables used there; note that it binds to all interfaces.

It offers detect, confirm, run status, reports, run browsing and a chat assistant. Runs started there
land in `orchestrator/webui/runs/run_<timestamp>/`. The web UI has **no authentication**: anyone who
can reach the port can start runs, browse the allowed data roots and delete runs. It binds to the
loopback interface by default. Keep it on a trusted network, and use `MRAG_WEBUI_HOST=0.0.0.0` only
behind your own access control.

## Reproducibility

- **DAStk is seeded.** DAStk estimates each motif's MD-score variance from bootstrap resamples drawn
  without a seed inside a process pool, so identical input gave different p-values. MRAG runs the
  unmodified DAStk script through `dastk_seeded.py`, which draws each motif's resamples from
  `RandomState(crc32(motif name))`. Two runs on the same input give byte-identical MD-score tables.
- **Deterministic numbers.** The TF ranking, the motif-based network and the evidence rules are
  deterministic. Section 13's motif scrambles come from fixed seeds. The edit set still depends on
  the run's network, so a run whose network differs draws different scrambles.
- **LLM outputs are not bit-reproducible.** Target-gene scores, the evidence audit, the Section-11
  term judgements and all report text come from local LLMs. They can change with the model build,
  the Ollama version, the hardware, and how candidates are batched. The run directory keeps the
  scorer's inputs, outputs and settings: `curated_targets/`, including `_score_params.json`.
- **AlphaGenome is a remote API.** Its predictions depend on the service's model at the time of the
  run, and calls can fail or be rate-limited. A failed or skipped model arm is reported in
  Section 13 with its reason and does not fail the run. Each model arm writes its predictions
  under `axis_validation/` and skips predictions already present in its output file.
- **Corgi** runs on a local GPU (bfloat16 autocast, test-time augmentation over 3 shifts x 2
  strands). It is conditioned on an expression context resolved from the run: `LNCaP_OE` for the
  ONECUT2 dataset, `PRAD_AR_high_mean` for the TCGA-PRAD cohort. A run that matches no context row
  reports the Corgi arm as `no context`.
- **External data change over time.** NCBI GeneRIFs and PubMed, the Enrichr libraries and the
  literature index change, so the evidence available to a later run differs.
- `mrag_axis.stat` implements both the definition the report uses (`unified`) and the definition
  of the paper's research scripts (`published`). The second one re-scores stored per-pair
  predictions, which are not part of this repository.

## Portability

This snapshot was developed and run on one shared Linux server. Expect to adapt it.

- **Absolute paths.** More than 80 tracked files refer to `/BIO1/MRAG/...`. The MCP packages are
  imported by adding `/BIO1/MRAG/Agent/mcps/<package>/src` to `sys.path`; only the orchestrator
  and `enrichment_mcp` are installed as packages. A few locations have environment overrides (see
  above); the rest are literals. There are two ways to adapt:
  - Reproduce the layout, with the checkout at `/BIO1/MRAG/Agent` and the venv at
    `/BIO1/MRAG/venvs/paper_qa`, symlinks being enough.
  - Edit the paths.
- **Paths outside `/BIO1/MRAG`.** A few tools are found at fixed paths elsewhere on the original
  host: a Miniconda under `/home/ai/miniconda3` (the Python 2 environment for ROSE, deepTools 3.5.6
  for `bamCoverage` and `bigwigAverage`, and the Python of the ATAC and RNA sample-detection
  helpers), `pyGenomeTracks` under `/home/ai/.local/bin`, `/BIO1/picard.jar`, sra-tools under
  `/BIO1/sratoolkit.3.1.1-ubuntu64`, samtools at `/usr/local/bin/samtools` (super-enhancer
  calling), and the system `/usr/bin/python3` with Debian's deepTools 3.3.2 for the patched
  `plotHeatmap`. HOMER is looked up in host install directories unless
  `HOMER_HOME` is set.
- **Not in the repository:**
  - The data under `Reference/`: genome indexes, GTFs, motif files, the Cistrome GIGGLE index,
    ortholog tables and the Corgi assets.
  - `bin/`: third-party tools. They are used unmodified except ROSE, whose MRAG changes are in
    `third_party/ROSE/`.
  - The literature index (the Qdrant collection, the parsed papers) and the scripts that built it.
    The `paperqa-mcp` tools and the target-evidence steps need the index.
  - The Corgi package and its venv; the AlphaGenome venv; the NCBI gene_info dump.
- **Included from the original host:**
  - `host_scripts/`: copies of the scripts installed in `/BIO1/MRAG/scripts` there.
    `ensure_scorer_ollama.sh` starts the target-scorer Ollama server on demand, and the code calls
    it by that path. `start_glm_ollama.sh` starts the report-LLM server, and `start_mrag_webui.sh`
    starts the web UI.
    They carry the host's ports, GPUs and paths. On another host, either adapt them or set
    `MRAG_SCORE_ENSURE_CMD=off` and `MRAG_CHAT_ENSURE_CMD=off` and run the Ollama servers yourself.
  - `Reference/cistrome/run_giggle.sh` (Section 9) and `Reference/peakheatmap/run_peakheatmap*.sh`
    (Section 6). These are called by absolute path, together with the patched deepTools copy
    beside them.
  - `libs/paperqa/mrag_paperqa` and `libs/paperqa/scripts/retrieve_cli.py`, the literature
    retriever.
- **Host policy defaults.** These defaults assume a large shared server:
  - runs pin themselves to CPUs 32-47;
  - a memory watcher pauses the fastest-growing run when the user's processes take more than 30 %
    of RAM;
  - a run refuses to start below 60 GB of free memory;
  - HOMER annotation is budgeted at up to 160 GB;
  - the GPU lease never hands out GPU 1, which serves the report LLM on the original host.

  The environment variables above change all of these except the GPU-1 exclusion, which is a
  constant in `gpu_lease.py`.
- **Tests.** The development test suite is not part of this snapshot.
- **Section 13 scope.** The lineage-foreign control panels, the AlphaGenome biosample set and the
  Corgi expression contexts were curated for the paper's prostate datasets. Other lineages get less
  curated defaults or skip an arm, and the section states which.

## Repository layout

```
orchestrator/
  pyproject.toml            mrag-orchestrator; console script `mrag`
  src/mrag_orchestrator/
    cli.py                  mrag detect | confirm | run | status | debug | sra-* | prep-subsample | testrun | variants
    config.py               SampleSheet / RunConfig (run.toml schema)
    runner.py               the pipeline DAG (run_from_config)
    tool_transport.py       in-process MCP transport (MRAG_TOOL_TRANSPORT)
    detector.py, sra.py     sample detection; SRA/ENA lookup and download
    organism.py, orthology.py   hg38 / mm10 profiles; mouse -> human ortholog route for literature
    literature.py           the 19 literature corpora and their domain rules
    resources.py, resource_guard.py, gpu_lease.py   CPU pinning, memory watcher, GPU lease
    nodes/                  one module per step; curated_targets.py, evidence_gate.py and
                            evidence_audit.py hold the target-gene evidence logic
    adapters/               processed-matrix adapters (R: limma, voom, matrix heatmap)
  src/mrag_axis/            Section 13: prep (regions, motifs, edits), arm (AlphaGenome, Corgi), stat
  webui/                    app.py (FastAPI), chat.py (chat assistant), templates/
  configs/symbol_sense.json gene-symbol sense table for the evidence gate
  new_run_from.sh           start a new run directory from an existing one, sharing its HOMER cache
mcps/<package>/             pyproject.toml, src/<module>/server.py (stdio server), src/<module>/tools/
  viz_report_mcp/templates/report.html   the report template
  viz_report_mcp/r_scripts/              rank, volcano and heatmap plots (R)
Reference/                (data not included; only these scripts)
  cistrome/run_giggle.sh    GIGGLE search and plot (Section 9)
  peakheatmap/              peak heatmap scripts (Section 6) and a patched deepTools 3.3.2 copy
host_scripts/               the original host's Ollama and web UI start scripts (reference copies)
third_party/ROSE/           MRAG's changes to ROSE: a patch, an added script and ROSE's license
libs/
  paperqa/                  literature retriever (mrag_paperqa) and its CLI; the index is not included
  genegpt/genegpt.py        NCBI E-utilities client (GeneRIFs, gene IDs), derived from NCBI GeneGPT
  ncbi_creds.py             NCBI key / e-mail loading, retries, shared rate limit
  alphagenome_client/       thin wrapper around the AlphaGenome SDK (own venv)
  track_helper/             pyGenomeTracks locus plots
```

## Paper datasets

The analyses in the paper use two datasets, which also serve as configuration examples:

- **OC2**: the human prostate cancer cell line LNCaP with ONECUT2 overexpression (`ONECUT2-OE`,
  treatment) vs vector control (`MYC-DDK`). It has ATAC-seq and RNA-seq, two replicates each, from
  FASTQ. It was run once per literature domain (`prostate`, `nec`, `sclc`) with the default
  Section-13 rule. The raw data were published with *Nucleic Acids Research* 52(13):7740 (2024).
- **PRAD**: TCGA prostate adenocarcinoma, `AR_high` vs `Basal_like`, from processed matrices: an
  ATAC-seq peak-count matrix with per-sample bigWigs and an RNA-seq log2 matrix, each with a
  `sample<TAB>group` TSV. It was run with `literature_domains = ["prostate"]` and declares
  `axis_headline = "alphagenome"` and `axis_ar_gate = true`.

## License

MRAG's own code is released under the MIT License, Copyright (c) 2026 Cedars-Sinai Medical Center;
see [LICENSE](LICENSE). Third-party components and their licenses are listed in
[NOTICE.md](NOTICE.md).

## Citation

Citation information will be added with the paper.
