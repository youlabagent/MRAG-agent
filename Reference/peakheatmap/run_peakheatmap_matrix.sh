#!/bin/bash
# Matrix-input peak heatmap (deeptools) — the REAL reference-analysis-style figure for
# processed count-matrix ATAC input (no per-rep BAM/narrowPeak).
#   * regions: 3-way split from DiffBind differential stats over the shared
#     TCGA consensus peak set -> treat-gained / non-differential (Shared) /
#     ref-gained.  (The correct analog of the reference analysis's within-arm narrowPeak-intersect
#     specific/shared split, which needs per-rep peak calls we don't have.)
#   * signal: up to 2 representative per-rep normalized bigWigs per arm
#     (TCGA insertions.bw), averaged with bigwigAverage -> one bigWig per arm,
#     exactly like run_peakheatmap.sh Step2.
#   * Step3 (computeMatrix reference-point -> patched plotHeatmap, white,purple
#     x2, mplrc fonts) IDENTICAL to run_peakheatmap.sh so the figure matches.
# Usage: run_peakheatmap_matrix.sh OUT LA "bwA1[,bwA2]" LB "bwB1[,bwB2]" diffbind.csv [FDR] [P] [CAP]
set -e
OUT="$1"; LA="$2"; BWA="$3"; LB="$4"; BWB="$5"; DB="$6"; FDR="${7:-0.05}"; P="${8:-16}"; CAP="${9:-15000}"
CONDA=/home/ai/miniconda3/bin
DTPATCH=/BIO1/MRAG/Agent/Reference/peakheatmap/deeptools_patched
PHEAT="env PYTHONPATH=$DTPATCH /usr/bin/python3 $DTPATCH/plotheatmap_patched.py"
mkdir -p "$OUT"; cd "$OUT"

cat > mplrc <<RC
font.size: 10
axes.titlesize: 11
xtick.labelsize: 8
ytick.labelsize: 9.5
legend.fontsize: 7.5
figure.subplot.hspace: 0.12
RC
export MATPLOTLIBRC="$OUT/mplrc"

# ---- per-arm mean bigWig from up to 2 representative reps (like Step2) ----
mean_arm(){ local lab="$1"; local csv="$2"; local out="$lab.bw"
  IFS=',' read -ra A <<< "$csv"
  if [ -s "$out" ]; then return; fi
  if [ "${#A[@]}" -le 1 ]; then cp "${A[0]}" "$out"; return; fi
  "$CONDA/bigwigAverage" -b "${A[@]}" -o "$out" -p "$P" >/dev/null 2>&1
}
mean_arm "$LA" "$BWA"
mean_arm "$LB" "$BWB"

# ---- 3-way region split from DiffBind (cols: 1 seqnames 2 start 3 end 6 Conc 9 Fold 11 FDR) ----
# treat-gained: FDR<thr & Fold>0 ; ref-gained: FDR<thr & Fold<0 ; Shared: not significant.
# Cap each category (by |Fold| for gained, by Conc for shared) to ~the reference analysis's scale.
awk -F',' -v thr="$FDR" 'NR>1 && $11!="NA" && $11+0<thr && $9+0>0{printf "%s\t%s\t%d\t%d\n",($9+0),$1,$2,$3}' "$DB" \
  | sort -k1,1gr | head -"$CAP" | cut -f2- > "${LA}_specific.bed"
awk -F',' -v thr="$FDR" 'NR>1 && $11!="NA" && $11+0<thr && $9+0<0{printf "%s\t%s\t%d\t%d\n",(-($9+0)),$1,$2,$3}' "$DB" \
  | sort -k1,1gr | head -"$CAP" | cut -f2- > "${LB}_specific.bed"
awk -F',' -v thr="$FDR" 'NR>1 && ($11=="NA" || $11+0>=thr){printf "%s\t%s\t%d\t%d\n",$6,$1,$2,$3}' "$DB" \
  | sort -k1,1gr | head -"$CAP" | cut -f2- > shared.bed
echo "regions: ${LA}=$(wc -l < ${LA}_specific.bed) shared=$(wc -l < shared.bed) ${LB}=$(wc -l < ${LB}_specific.bed)"

# ---- Step3 (identical to run_peakheatmap.sh): computeMatrix + patched plotHeatmap ----
[ -s matrix_group.gz ] || computeMatrix reference-point --referencePoint center \
  -R "${LA}_specific.bed" shared.bed "${LB}_specific.bed" \
  -S "$LA.bw" "$LB.bw" -b 3000 -a 3000 --skipZeros -p "$P" -o matrix_group.gz
$PHEAT -m matrix_group.gz -out peakheatmap.png \
  --regionsLabel "${LA}_specific" Shared "${LB}_specific" --xAxisLabel "" \
  --samplesLabel "$LA" "$LB" --colorList "white,purple" "white,purple" \
  --heatmapHeight 11.3 --missingDataColor white --dpi 200

echo "OK $OUT/peakheatmap.png"
