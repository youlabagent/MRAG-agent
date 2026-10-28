# alphagenome_client

Thin Python wrapper around the AlphaGenome SDK used by the MRAG pipeline.

## Layout
```
alphagenome_client/
  __init__.py
  config.py    # API key + cache paths
  client.py    # DnaClient singleton + retry wrapper
  helpers.py   # high-level: counterfactual_at_peak(...)
  cache.py     # SQLite-backed response cache
  cli.py       # `ag check`, `ag cf ...`
scripts/smoke.sh
```

## Setup
```bash
cd libs/alphagenome_client   # on the original host: /BIO1/MRAG/Agent/libs/alphagenome_client
source .venv/bin/activate
uv pip install -e .
```

## Configuration
The API key is read with this precedence:

1. `ALPHAGENOME_API_KEY` env var
2. file pointed to by `ALPHAGENOME_KEY_FILE`
3. `/BIO1/MRAG/secrets/.alphagenome_key`

The cache lives under `/BIO1/MRAG/alphagenome_client/.cache` (override with
`ALPHAGENOME_CACHE_DIR`).

## Quick CLI checks
```bash
ag check                                          # key + client creation
ag cf --chrom chrX --start 67631000 --end 67632000 \
      --ref ACGT --outputs ATAC,RNA_SEQ           # one counterfactual call
```

## Python usage (used by the standalone alphagenome-mcp; Section 13 calls the SDK directly)
```python
from alphagenome_client import counterfactual_at_peak

result = counterfactual_at_peak(
    chrom="chr10", peak_start=89_624_000, peak_end=89_624_500,
    ref_at_centre="ACGTGT",
    requested_outputs=("ATAC", "RNA_SEQ", "CHIP_TF"),
    tissue="UBERON:0002367",
)
# result["tracks"]["ATAC"]["delta"]["max_abs"] etc.
```

## Notes
- Default `alt` is the reverse-complement of `ref_at_centre`, used as a heuristic
  motif-disruption allele. Override with `alt=` for an explicit edit.
- Cache key = SHA-256 of canonical request JSON. Re-running an identical query
  is essentially free.
- 30-60 s latency per call.
