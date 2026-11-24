# Example reports

Two complete MRAG reports for the datasets used in the paper. Each report is one standalone HTML file
(`mrag_report.html`) written by the `viz_report` step at the end of a run.

| Example | Dataset | Comparison | Report |
|---|---|---|---|
| **OC2** | LNCaP prostate cancer cells, ATAC-seq + RNA-seq, 2 replicates per arm (processed from FASTQ) | `ONECUT2-OE` vs `MYC-DDK` | [`OC2/mrag_report.html`](OC2/mrag_report.html) |
| **PRAD** | TCGA prostate adenocarcinoma, processed matrices (ATAC-seq peak counts, 24 vs 28 samples; RNA-seq log2, 12 vs 14 samples) | `AR_high` vs `Basal_like` | [`PRAD/mrag_report.html`](PRAD/mrag_report.html) |

Both runs used `literature_domains = ["prostate"]`. The PRAD run declares `axis_headline = "alphagenome"`
and `axis_ar_gate = true` (see *Paper datasets* in the top-level README), so its Section 13 also reports the
AR-gated enhancer row.

## Viewing

Open the reports online (served by GitHub Pages):

- OC2: <https://youlabagent.github.io/MRAG-agent/examples/OC2/mrag_report.html>
- PRAD: <https://youlabagent.github.io/MRAG-agent/examples/PRAD/mrag_report.html>

The pages load Plotly 2.27.0, DataTables 1.13.7, jQuery 3.7.1 and Cytoscape 3.28.1 from public CDNs, so an
internet connection is needed to draw the plots and tables.

GitHub's own file viewer shows `.html` files as source only. To view a report offline, download the file
and open it in a browser (the libraries above are still fetched from the CDNs).

## What each report contains

Sections 1-15: executive summary, pipeline overview, TF ranking (top 10 by Combined Score), differential
ATAC-seq and RNA-seq, rank and volcano plots, peak and expression heatmaps, HOMER motif enrichment, GIGGLE
overlap with Cistrome ChIP-seq, the super-enhancer -> TF -> target network, functional enrichment, locus
snapshots, **Section 13 in-silico validation (AlphaGenome and Corgi)**, the LLM-written interpretation
with its citation check, and the run manifest.

## Provenance and reproducibility

- Generated on 2026-10-19 with `mrag-orchestrator 0.1.0`; LLM: `qwen3.6-27b:q6` served through Ollama;
  tools were called over MCP.
- The OC2 run took its DiffBind result from an earlier run of the same dataset (2026-10-12) through
  `precomputed.diffbind_out_csv`.
- Section 13 (SE -> TF enhancer end, target end, AR gate) scores stored per-pair AlphaGenome and Corgi
  predictions with the `unified` definition of `mrag_axis.stat`.
- The LLM-written interpretation (Section 14) and the AlphaGenome API predictions are not bit-for-bit
  reproducible, so a new run gives numbers that are close to, but not identical with, those in these files.
