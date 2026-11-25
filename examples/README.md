# Example reports

Complete MRAG reports for the two datasets used in the paper. Each report is one standalone HTML file
(`mrag_report.html`) written by the `viz_report` step at the end of a run.

- [`PCa_cellline_reports/`](PCa_cellline_reports/): the OC2 dataset (LNCaP prostate cancer cells,
  `ONECUT2-OE` vs `MYC-DDK`), run three times on the same data with the same settings. Only the
  literature domain changes between the runs: prostate cancer, neuroendocrine carcinoma or small cell
  lung cancer.
- [`PRAD/`](PRAD/): TCGA prostate adenocarcinoma, `AR_high` vs `Basal_like`.

| Example | Dataset | Comparison | Literature domain | Open online | Download |
|---|---|---|---|---|---|
| **OC2, prostate** | LNCaP prostate cancer cells, ATAC-seq + RNA-seq, 2 replicates per arm (processed from FASTQ) | `ONECUT2-OE` vs `MYC-DDK` | Prostate cancer (`prostate`) | [report](https://youlabagent.github.io/MRAG-agent/examples/PCa_cellline_reports/prostate/mrag_report.html) | [mrag_report.html](https://github.com/youlabagent/MRAG-agent/raw/main/examples/PCa_cellline_reports/prostate/mrag_report.html) |
| **OC2, neuroendocrine** | same data | `ONECUT2-OE` vs `MYC-DDK` | Neuroendocrine carcinoma (`nec`) | [report](https://youlabagent.github.io/MRAG-agent/examples/PCa_cellline_reports/neuroendocrine/mrag_report.html) | [mrag_report.html](https://github.com/youlabagent/MRAG-agent/raw/main/examples/PCa_cellline_reports/neuroendocrine/mrag_report.html) |
| **OC2, small cell lung cancer** | same data | `ONECUT2-OE` vs `MYC-DDK` | Small cell lung cancer (`sclc`) | [report](https://youlabagent.github.io/MRAG-agent/examples/PCa_cellline_reports/small_cell_lung_cancer/mrag_report.html) | [mrag_report.html](https://github.com/youlabagent/MRAG-agent/raw/main/examples/PCa_cellline_reports/small_cell_lung_cancer/mrag_report.html) |
| **PRAD** | TCGA prostate adenocarcinoma, processed matrices (ATAC-seq peak counts, 24 vs 28 samples; RNA-seq log2, 12 vs 14 samples) | `AR_high` vs `Basal_like` | Prostate cancer (`prostate`) | [report](https://youlabagent.github.io/MRAG-agent/examples/PRAD/mrag_report.html) | [mrag_report.html](https://github.com/youlabagent/MRAG-agent/raw/main/examples/PRAD/mrag_report.html) |

The files are 3.1 to 3.7 MB each, which is too large for GitHub's file viewer. Opening a report such as
[`PCa_cellline_reports/prostate/mrag_report.html`](PCa_cellline_reports/prostate/mrag_report.html) in the
repository only shows *"Sorry about that, but we can't show files that are this big right now."* Use
**Open online**, or **Download** the file: on the file page click the download button (top right), or
right-click the *Download* link above and choose *Save Link As...*.

The PRAD run declares `axis_headline = "alphagenome"` and `axis_ar_gate = true` (see *Paper datasets* in
the top-level README), so its Section 13 also reports the AR-gated enhancer row.

## Viewing

Open the reports online (served by GitHub Pages):

- OC2, prostate cancer: <https://youlabagent.github.io/MRAG-agent/examples/PCa_cellline_reports/prostate/mrag_report.html>
- OC2, neuroendocrine carcinoma: <https://youlabagent.github.io/MRAG-agent/examples/PCa_cellline_reports/neuroendocrine/mrag_report.html>
- OC2, small cell lung cancer: <https://youlabagent.github.io/MRAG-agent/examples/PCa_cellline_reports/small_cell_lung_cancer/mrag_report.html>
- PRAD: <https://youlabagent.github.io/MRAG-agent/examples/PRAD/mrag_report.html>

The pages load Plotly 2.27.0, DataTables 1.13.7, jQuery 3.7.1 and Cytoscape 3.28.1 from public CDNs, so an
internet connection is needed to draw the plots and tables.

GitHub's own file viewer cannot show these files (see above). To keep a copy, download the file and open it
in a browser (the libraries above are still fetched from the CDNs).

## What each report contains

Sections 1-15: executive summary, pipeline overview, TF ranking (top 10 by Combined Score), differential
ATAC-seq and RNA-seq, rank and volcano plots, peak and expression heatmaps, HOMER motif enrichment, GIGGLE
overlap with Cistrome ChIP-seq, the super-enhancer -> TF -> target network, functional enrichment, locus
snapshots, **Section 13 in-silico validation (AlphaGenome and Corgi)**, the LLM-written interpretation
with its citation check, and the run manifest.

## Provenance and reproducibility

- Generated with `mrag-orchestrator 0.1.0`; LLM: `qwen3.6-27b:q6` served through Ollama; tools were called
  over MCP.
- The three OC2 reports come from one batch of runs started on 2026-10-12 (the small cell lung cancer run
  finished on 2026-10-13). Their run configurations are identical except for `literature_domains` and the
  run paths, and all three read the same DiffBind result through `precomputed.diffbind_out_csv`. What this
  keeps the same and what changes with the domain is listed in
  [`PCa_cellline_reports/`](PCa_cellline_reports/).
- The PRAD report was generated on 2026-10-19. Its Section 13 (SE -> TF enhancer end, target end, AR gate)
  scored stored per-pair AlphaGenome and Corgi predictions with the `unified` definition of
  `mrag_axis.stat`.
- The LLM-written interpretation (Section 14) and the AlphaGenome API predictions are not bit-for-bit
  reproducible, so a new run gives numbers that are close to, but not identical with, those in these files.
