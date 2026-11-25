# PCa cell-line reports: one dataset, three literature domains

The OC2 dataset (the LNCaP prostate cancer cell line with ONECUT2 overexpression, `ONECUT2-OE`, vs vector
control, `MYC-DDK`; ATAC-seq and RNA-seq, two replicates each) was run three times with the same data and
the same configuration. Only the literature domain used to score the target genes changed
(`literature_domains` in `run.toml`).

| Folder | Literature domain | `literature_domains` | Open online | Download |
|---|---|---|---|---|
| [`prostate/`](prostate/) | Prostate cancer | `["prostate"]` | [report](https://youlabagent.github.io/MRAG-agent/examples/PCa_cellline_reports/prostate/mrag_report.html) | [mrag_report.html](https://github.com/youlabagent/MRAG-agent/raw/main/examples/PCa_cellline_reports/prostate/mrag_report.html) |
| [`neuroendocrine/`](neuroendocrine/) | Neuroendocrine carcinoma | `["nec"]` | [report](https://youlabagent.github.io/MRAG-agent/examples/PCa_cellline_reports/neuroendocrine/mrag_report.html) | [mrag_report.html](https://github.com/youlabagent/MRAG-agent/raw/main/examples/PCa_cellline_reports/neuroendocrine/mrag_report.html) |
| [`small_cell_lung_cancer/`](small_cell_lung_cancer/) | Small cell lung cancer | `["sclc"]` | [report](https://youlabagent.github.io/MRAG-agent/examples/PCa_cellline_reports/small_cell_lung_cancer/mrag_report.html) | [mrag_report.html](https://github.com/youlabagent/MRAG-agent/raw/main/examples/PCa_cellline_reports/small_cell_lung_cancer/mrag_report.html) |

## What stays the same and what changes

The steps before the literature scoring do not read the domain, so these parts are identical in the three
reports:

- the TF panel and its Combined Scores (Section 3),
- differential accessibility and expression, the rank and volcano plots, the peak and expression heatmaps,
  HOMER and GIGGLE (Sections 4-9),
- functional enrichment of the TF target programs (Section 11),
- the top-10 super-enhancers and the candidate target pool,
- the enhancer-end rows of Section 13 (super-enhancer -> TF edges).

These parts change with the domain:

- the literature-scored target genes in the super-enhancer -> TF -> target network (Section 10) and the
  locus snapshots chosen from them (Section 12),
- the target-end rows of Section 13,
- the written interpretation and its citations (Section 14),
- the executive summary, which is written for each run, and the run details (domain, times) in the
  pipeline overview and the manifest (Sections 1, 2 and 15).

The three runs started on 2026-10-12 (the small cell lung cancer run finished on 2026-10-13) and read the
same DiffBind result. See the [examples README](../README.md) for what a report contains and how the
reports were produced.
