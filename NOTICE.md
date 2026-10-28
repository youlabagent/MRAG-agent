# Third-party notices

MRAG's own code is under the MIT License (see `LICENSE`). This file lists the third-party code
that is bundled in this repository, derived into it, or loaded by it. It also lists the external
tools, models and data services the pipeline calls but does not distribute.

## Code bundled in or derived into this repository

### deepTools 3.3.2: `Reference/peakheatmap/deeptools_patched/`

This directory holds a copy of the deepTools 3.3.2 Python package
(https://github.com/deeptools/deepTools), taken from Debian's `python3-deeptools` 3.3.2+ds-2, with one
modified file, `deeptools/plotHeatmap.py`. The
changes are layout only: a taller middle region group, a wider gap between the profile and the
heatmap, and the region legend placed outside the plot. The package's test suite is omitted.
`plotheatmap_patched.py` runs the patched `plotHeatmap` for the Section-6 peak heatmaps. The full
license is in `Reference/peakheatmap/deeptools_patched/LICENSE.txt`.

- deepTools is licensed under the **MIT License**, Copyright 2016-2019 Max Planck Institute for
  Immunobiology and Epigenetics, Fidel Ramirez, Devon P Ryan, Björn Grüning, Friederike Dündar,
  Sarah Diehl, Vivek Bhardwaj, Fabian Kilpert, Andreas S Richter, Steffen Heyne, Thomas Manke.
- `deeptools/cm.py` comes from the seaborn project. It is licensed under the **BSD 3-Clause
  License**, Copyright (c) 2012-2019, Michael L. Waskom, and its full notice is at the top of the
  file.

### NCBI GeneGPT: `libs/genegpt/genegpt.py`

Parts of `libs/genegpt/genegpt.py` derive from NCBI GeneGPT (https://github.com/ncbi/GeneGPT). Its
E-utilities call helper `call_api` grew out of GeneGPT's API-call code. GeneGPT's few-shot LLM
tool-use loop is not used by MRAG and is not included.

GeneGPT is distributed under the NCBI **Public Domain Notice**. It is a "United States Government
Work" written by U.S. Government employees as part of their official duties. It cannot be
copyrighted and is freely available for use and reproduction without restriction. The NLM and the
U.S. Government do not warrant its performance or results, and they disclaim all warranties,
express or implied.

The upstream notice asks that the authors be cited in any work based on the material. Its LICENSE
file names Qiao Jin, Won Kim, Qingyu Chen, Donald C. Comeau, Lana Yeganova, John Wilbur and Zhiyong
Lu (BioCPT). The upstream README asks for this citation of GeneGPT:

> Jin Q, Yang Y, Chen Q, Lu Z. GeneGPT: augmenting large language models with domain tools for
> improved access to biomedical information. *Bioinformatics* 40(2):btae075 (2024).

### MCP Python SDK: `orchestrator/src/mrag_orchestrator/tool_transport.py`

`_keep_warnings()` in `tool_transport.py` contains an adapted copy of `Server._handle_message` from
the MCP Python SDK 1.27.1 (https://github.com/modelcontextprotocol/python-sdk), without its
`warnings.catch_warnings` block. The SDK itself is installed separately.

- The MCP Python SDK is licensed under the **MIT License**, Copyright (c) 2024 Anthropic, PBC.

### TOmicsVis: `mcps/viz_report_mcp/r_scripts/rank_plot.R`

The Section-3 rank plot follows the design of `gene_rank_plot` from TOmicsVis (Benben Miao and Wei
Dong, https://github.com/benben-miao/TOmicsVis). It is written inline with ggplot2 and ggrepel;
TOmicsVis itself is not used.

- TOmicsVis declares the **MIT License** in its package DESCRIPTION.

### ROSE changes: `third_party/ROSE/`

MRAG runs ROSE (Young lab, Whitehead Institute) from the Barski-lab fork
(https://github.com/Barski-lab/ROSE, commit e3296fe) with local changes. `third_party/ROSE/` holds
those changes as a patch against that commit (`mrag_rose.patch`: index checks, samtools error
handling, parallel read mapping, the hg38 annotation key and a header fix in the gene mapper), the
added script `ROSE_stitchTable.py` (stitching without read mapping, for bigWig input) and ROSE's
license. ROSE itself is not bundled. `ROSE_stitchTable.py` is MRAG's own code under the MIT License.

- ROSE is licensed under the **X11 License**, Copyright (C) 2013 by Whitehead Institute for
  Biomedical Research. The full text is in `third_party/ROSE/LICENSE`.

## Tools wrapped or called at run time (not bundled)

### DAStk: `mcps/ATAC_mcp/src/atac_mcp/tools/dastk_seeded.py`

DAStk (Dowell Lab, https://github.com/Dowell-Lab/DAStk) is **not bundled**. `dastk_seeded.py` loads
DAStk's own `differential_md_score.py` from a separately installed DAStk at run time. It then
replaces that module's `resample` call with one seeded per motif, so that the bootstrap p-values are
reproducible. The wrapper contains no DAStk source. The ATAC pipeline also calls DAStk's
`process_atac.py`. The original host used DAStk 1.0.1r1.

DAStk is licensed under the **BSD 3-Clause License**, Copyright (c) 2018, dowelllab.

### Corgi: `orchestrator/src/mrag_axis/arm/corgi.py`

Corgi (https://github.com/ekinda/corgi) is **not bundled**. The Section-13 Corgi arm runs a child
process in Corgi's own environment, which imports `corgi.config` and `corgi.model` to load the
checkpoint and predict. No Corgi source file is included in this repository.

- Code: **Apache License 2.0**, Copyright 2026 Ekin Deniz Aksu.
- Model weights: the published checkpoint `corgi_model.pt` from Zenodo record 18630048
  (https://zenodo.org/records/18630048), licensed **CC BY 4.0**.
- Paper: https://www.nature.com/articles/s41467-026-75527-2

### AlphaGenome: `libs/alphagenome_client`, `orchestrator/src/mrag_axis/arm/alphagenome.py`

`libs/alphagenome_client` is MRAG's own thin wrapper around the AlphaGenome Python SDK
(`alphagenome`, Google LLC, https://github.com/google-deepmind/alphagenome). The SDK is **not
bundled**; it is installed in a separate virtual environment.

- The SDK is licensed under the **Apache License 2.0**.
- The AlphaGenome API requires an API key. It is offered free of charge for non-commercial use,
  subject to the AlphaGenome Terms of Use (https://deepmind.google.com/science/alphagenome/terms).

### Other external tools

These programs are called as subprocesses and are not distributed with MRAG. The licenses below are
taken from the copies installed on the original host.

| Tool | Used for | License |
|---|---|---|
| HOMER (http://homer.ucsd.edu/homer/) | motif annotation, known-motif enrichment | GPL-3.0 |
| ROSE, the Barski-lab fork (https://github.com/Barski-lab/ROSE) of the Young lab's ROSE, with the changes above | super-enhancer calling | X11 License, Copyright (C) 2013 Whitehead Institute for Biomedical Research |
| GIGGLE (https://github.com/ryanlayer/giggle) | TF ChIP-seq overlap search | MIT, Copyright (c) 2021 Ryan Layer |
| STAR | RNA-seq alignment | MIT, Copyright (c) 2019 Alexander Dobin |
| RSEM | RNA-seq quantification | GPL-3.0 |
| Trim Galore | adapter trimming | GPL-3.0 |
| cutadapt | adapter trimming (used by Trim Galore) | MIT (Expat) |
| bowtie2 | ATAC-seq alignment | GPL-3.0-or-later |
| samtools, htslib (`tabix`, `bgzip`) | BAM and index handling | MIT (Expat) |
| bedtools | interval operations | see upstream |
| FastQC | read QC | GPL-3.0-or-later |
| deepTools | bigWig coverage, peak heatmaps | MIT |
| MACS2 | peak calling | see upstream |
| Picard | duplicate removal, BAM sorting | see upstream |
| pyGenomeTracks | locus snapshots | see upstream |
| sra-tools (`fasterq-dump`) | SRA downloads (`mrag sra-fetch`) | see upstream |
| R / Bioconductor packages (DESeq2, DiffBind, limma, edgeR, BiocParallel, org.Hs.eg.db, pheatmap, ggplot2, ggrepel, optparse, RColorBrewer, gplots, dplyr, readr) | statistics and plots | see each package |
| Ollama with the Qwen3.6 27B and Qwen3.8 27B models | LLM serving | see Ollama and the model licenses |

## JavaScript and CSS loaded by the HTML pages (not bundled)

The report template (`mcps/viz_report_mcp/templates/report.html`) and the web UI templates
(`orchestrator/webui/templates/`, `orchestrator/webui/app.py`) load these libraries from public
CDNs when a page is opened. No copy is stored in this repository.

| Library | Version and source | Used in | License |
|---|---|---|---|
| Plotly.js | 2.27.0, cdn.plot.ly | report | MIT, Copyright (c) Plotly, Inc. |
| Plotly.js | 2.35.2, cdn.plot.ly | web UI | MIT, Copyright (c) Plotly, Inc. |
| Cytoscape.js | 3.28.1, unpkg.com | report, web UI plot pane | MIT, Copyright (c) The Cytoscape Consortium |
| Cytoscape.js | 3.30.2, cdn.jsdelivr.net | web UI chat page | MIT, Copyright (c) The Cytoscape Consortium |
| jQuery | 3.7.1, code.jquery.com | report | MIT, (c) OpenJS Foundation and other contributors |
| DataTables (JS and CSS) | 1.13.7, cdn.datatables.net | report | MIT, (c) SpryMedia Ltd |
| Pico CSS | 2.x, cdn.jsdelivr.net | web UI | MIT, Copyright (c) Pico |
| htmx | 1.9.12, unpkg.com | web UI | Zero-Clause BSD (0BSD) |

## Python packages

MRAG imports these Python packages, which are installed separately and not distributed here:

- `mcp` and `fastmcp`
- `pydantic`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`,
  `jinja2`, `Pillow`
- `pyfaidx`, `pyBigWig`, `pydeseq2`, `gseapy`
- `qdrant-client`, `sentence-transformers`, `torch`
- `fastapi`, `uvicorn`, `tomli_w`
- optional: `psutil`, `toml`

Each package carries its own license.

## Models, data and services used at run time (not distributed)

| Resource | Used for | Terms |
|---|---|---|
| BAAI/bge-m3 embedding model | literature retrieval | see the model card |
| JASPAR 2024 CORE vertebrates (and HOCOMOCO v12, merged into the HOMER motif library) | motif scans, Section-13 motif scrambles | JASPAR: CC BY 4.0; HOCOMOCO: see upstream |
| HOCOMOCO v11 motif-site BEDs (from DAStk's setup) | DAStk MD-scores | see upstream |
| Cistrome DB TF ChIP-seq data (GIGGLE index built from it) | Section 9 | see Cistrome DB terms |
| Enrichr gene-set libraries (MSigDB Hallmark 2020, GO Biological Process 2023, Reactome 2022), fetched through gseapy | Section 11 | see Enrichr and each library's terms |
| NCBI E-utilities: PubMed, Gene, GeneRIF; NCBI `gene_info` | target-gene evidence | NCBI usage policies |
| ENA portal API and SRA | `mrag sra-lookup` / `sra-fetch` | EMBL-EBI and NCBI terms |
| AlphaGenome API | Section 13 | AlphaGenome Terms of Use |
| TCGA data via the GDC (paper dataset PRAD) | input data | GDC data use terms |
