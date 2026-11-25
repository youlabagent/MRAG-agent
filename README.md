# MRAG: Master Regulators of Accessible Genome

**Backward-designed agentic genomics that bridges sequence foundation models and regulatory graphs
for causal network discovery.**

MRAG takes ATAC-seq and RNA-seq from a two-condition experiment (treatment vs control) and builds a
two-tier transcriptional regulatory network: super-enhancer -> master transcription factor (TF) ->
target gene. It is designed backward from that endpoint. The network is fixed first as the result to
be verified, each intermediate state gets checks that can be measured, and the analysis is then run
as a fixed, deterministic sequence of tools served over the Model Context Protocol (MCP).

- **Deterministic statistics, separate from AI reasoning.** Alignment, peak calling, differential
  analysis, motif scoring and the TF ranking run as deterministic MCP tools. Locally served
  open-source LLMs (Qwen3.6 27B and Qwen3.8 27B) score and audit the literature evidence for
  target genes, judge enrichment terms, group samples into arms and write the report text. An LLM
  never decides which steps run.
- **Sequence foundation models inside a regulatory graph.** AlphaGenome and Corgi predict the effect
  of in silico motif edits inside each super-enhancer, without fine-tuning. Each super-enhancer -> TF
  edge is tested against lineage-foreign control factors put through the same edits.
- **One dataset, several disease contexts.** The same data can be read against different literature
  domains (for example prostate cancer, neuroendocrine carcinoma or small-cell lung cancer). The TF
  panel and the candidate target pool stay the same; only the literature-scored targets and their
  interpretation change.

Everything ends in one interactive HTML report. Human (hg38) and mouse (mm10) data are supported; the
in silico validation is human-only.

> **Status: code-availability snapshot for the paper.** This repository holds the code that produced
> the paper's analyses, cleaned for release. It is not a packaged tool: host paths under `/BIO1/MRAG`
> are hard-coded, and reference data, the literature index, the Corgi runtime and the third-party
> tools are not included. See [Portability](TUTORIAL.md#portability) in the tutorial.

## Example reports

| Dataset | Comparison | Literature domain | Report |
|---|---|---|---|
| **OC2**: LNCaP prostate cancer cells with ONECUT2 overexpression | `ONECUT2-OE` vs `MYC-DDK` | Prostate cancer | [Open online](https://youlabagent.github.io/MRAG-agent/examples/PCa_cellline_reports/prostate/mrag_report.html) |
| | | Neuroendocrine carcinoma | [Open online](https://youlabagent.github.io/MRAG-agent/examples/PCa_cellline_reports/neuroendocrine/mrag_report.html) |
| | | Small cell lung cancer | [Open online](https://youlabagent.github.io/MRAG-agent/examples/PCa_cellline_reports/small_cell_lung_cancer/mrag_report.html) |
| **PRAD**: TCGA prostate adenocarcinoma | `AR_high` vs `Basal_like` | Prostate cancer | [Open online](https://youlabagent.github.io/MRAG-agent/examples/PRAD/mrag_report.html) |

The three OC2 reports come from the same data and settings; only the literature domain changes. The files,
how they were produced and how to download them are in [`examples/`](examples/).

## How MRAG works

The workflow is organized as four states, from raw data to the target network. Hard checks sit
between them: for example, the TF ranking is refused when too few TFs join between ATAC-seq and
RNA-seq, and a target gene's score is capped when no retrieved evidence names the gene.

1. **Preprocessing.** ATAC-seq and RNA-seq enter as FASTQ, peaks or matrices. Reads are trimmed and
   aligned, peaks are called, and differential accessibility and expression are computed (DiffBind,
   DESeq2, or limma for log2 matrices).
2. **Master-regulator ranking.** DAStk motif displacement (MD) scores from ATAC-seq and the
   differential expression of each TF's own gene are combined per TF with Stouffer's method into a
   signed Combined Score. The TF panel is the top 10 TFs whose differential MD-score is not
   negative. ROSE calls super-enhancers in the treatment arm, and HOMER places the panel TFs' motifs
   in the super-enhancers and in the target-gene peaks.
3. **Evidence and in silico perturbation.** Candidate targets are up-regulated genes whose peaks
   carry a panel-TF motif. A local LLM scores each candidate against NCBI GeneRIFs and passages from
   a local literature index, behind a deterministic evidence gate. In parallel, each
   super-enhancer -> TF edge is tested by scrambling the TF's motifs inside the super-enhancer;
   AlphaGenome and Corgi predict the effect, and Fisher's exact test with Holm correction compares
   network TFs with lineage-foreign control factors.
4. **Target state.** The super-enhancer -> TF -> target network, functional enrichment, locus
   snapshots and the written interpretation go into a single HTML report, with a manifest that
   records each step's status, inputs, parameters and outputs.

TF -> target edges are motif-based candidates; the in silico test is applied to the
super-enhancer -> TF edges. The details of every step are in
[What MRAG does](TUTORIAL.md#what-mrag-does).

## Architecture

```
orchestrator/   mrag CLI and the deterministic runner (a fixed DAG of steps); optional web UI
mcps/           10 analysis packages, each a FastMCP server: 114 tools in total
libs/           NCBI client, literature retriever, AlphaGenome client, locus-track plots
examples/       example reports: OC2 in three literature domains, and PRAD
```

The orchestrator calls the MCP tools in process; each package can also be served over stdio to any
MCP client. The package table, the LLM roles and the step order are in
[Architecture](TUTORIAL.md#architecture).

## Quick start

```bash
pip install -e orchestrator -e mcps/enrichment_mcp
mrag detect  --fastq-dir /data/my_study -o sheet.json
mrag confirm --sheet sheet.json --output-config run.toml --output-dir runs/my_study/output \
             --treatment-arm ONECUT2-OE --cancer-type "prostate cancer" \
             --cell-line LNCaP --tissue prostate --literature-domains prostate
mrag run     --config run.toml
```

MRAG needs external tools, reference data, Ollama-served LLMs, a literature index, an AlphaGenome API
key and Corgi on a GPU. Read [Requirements](TUTORIAL.md#requirements) and
[Configuration](TUTORIAL.md#configuration) before the first run.

## Documentation

The [tutorial](TUTORIAL.md) holds the full documentation:

- [What MRAG does](TUTORIAL.md#what-mrag-does): each step in detail
- [Architecture](TUTORIAL.md#architecture): packages, LLM roles and step order
- [Inputs](TUTORIAL.md#inputs) and [Outputs: the report](TUTORIAL.md#outputs-the-report)
- [Requirements](TUTORIAL.md#requirements)
- [Configuration](TUTORIAL.md#configuration): `run.toml` and environment variables
- [Quick start](TUTORIAL.md#quick-start): commands, resume, status and the web UI
- [Reproducibility](TUTORIAL.md#reproducibility) and [Portability](TUTORIAL.md#portability)
- [Repository layout](TUTORIAL.md#repository-layout)

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

The complete reports of the three OC2 runs (one per literature domain) and of the PRAD run are in
[`examples/`](examples/).

## License

MRAG's own code is released under the MIT License, Copyright (c) 2026 Cedars-Sinai Medical Center;
see [LICENSE](LICENSE). Third-party components and their licenses are listed in
[NOTICE.md](NOTICE.md).

## Citation

Citation information will be added with the paper.
