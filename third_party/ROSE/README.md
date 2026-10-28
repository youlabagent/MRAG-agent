# ROSE changes

MRAG calls ROSE from `mcps/ATAC_mcp/src/atac_mcp/tools/super_enhancers.py` (`ROSE_DIR`, or
`MRAG_ROSE_DIR`). The ROSE copy it runs is the Barski-lab fork of the Young lab's ROSE
(https://github.com/Barski-lab/ROSE, commit `e3296fe`) with the changes in this directory:

- `mrag_rose.patch`, a patch against that commit:
  - `ROSE_main.py`: the options `-e` (read extension), `-f` (read floor) and `-n` (threads), and
    read mapping split over `-n` parallel workers.
  - `ROSE_bamToGFF.py`: the BAM index is looked up for this BAM only, and a missing index stops
    the run.
  - `ROSE_utils.py`: samtools errors that mean no usable data stop the run instead of yielding zero
    reads everywhere.
  - `ROSE_geneMapper.py`: the column header is taken from the header row, an `HG38` genome key,
    `-g` also accepts a path, and the annotation is found relative to the script.
  - file modes, and a few print statements.
- `ROSE_stitchTable.py`, an added script. It stitches the regions and writes ROSE's locus table
  without read mapping, for inputs that have bigWig signal but no BAM files.

To build the copy:

```bash
git clone https://github.com/Barski-lab/ROSE.git
cd ROSE
git checkout e3296fe
git apply /path/to/MRAG/third_party/ROSE/mrag_rose.patch
cp /path/to/MRAG/third_party/ROSE/ROSE_stitchTable.py .
```

ROSE also needs `annotation/<build>_refseq.ucsc`, the UCSC RefSeq gene table of each genome build
(a header line, then `bin name chrom strand txStart txEnd ...`). MRAG uses `hg38_refseq.ucsc` and
`mm10_refseq.ucsc`. These tables are not included. ROSE runs under Python 2, with `samtools` and R
on `PATH`.

ROSE is distributed under the X11 License, Copyright (C) 2013 by Whitehead Institute for Biomedical
Research. The full text is in `LICENSE`.
