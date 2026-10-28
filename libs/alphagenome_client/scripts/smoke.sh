#!/usr/bin/env bash
# Smoke test for alphagenome_client.
# Usage:
#   bash scripts/smoke.sh           # key check only
#   bash scripts/smoke.sh --full    # also runs one real prediction
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[1/3] key + client check"
python -m alphagenome_client.cli check

if [[ "${1:-}" == "--full" ]]; then
  echo
  echo "[2/3] counterfactual prediction (AR locus on chrX, ~16KB window)"
  # AR gene locus on chrX (GRCh38): ~67,544,032-67,730,619
  python -m alphagenome_client.cli cf \
    --chrom chrX --start 67631000 --end 67632000 \
    --ref ACGT --outputs ATAC,RNA_SEQ \
    --tissue UBERON:0002367 \
    --window 16384 \
    | head -80
  echo
  echo "[3/3] cache check (second call should hit cache)"
  time python -m alphagenome_client.cli cf \
    --chrom chrX --start 67631000 --end 67632000 \
    --ref ACGT --outputs ATAC,RNA_SEQ \
    --tissue UBERON:0002367 \
    --window 16384 \
    > /dev/null
fi
echo "DONE"
