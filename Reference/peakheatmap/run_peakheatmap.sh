#!/bin/bash
# Peak heatmap (deeptools): mirrors the reference analysis Plot/peakheatmap/1.input.py (3-way region:
# A-specific / Shared / B-specific via within-arm rep narrowPeak bedtools intersect)
# + 2.peakheatmap_group.py (CPM bamCoverage per rep -> bigwigAverage group mean ->
# computeMatrix -> plotHeatmap, colorList white,purple x2). 2026-06-29 reference-analysis update
# fully replaces the prior DiffBind-UP/DOWN based region definition.
# bamCoverage/bigwigAverage taken from /home/ai/miniconda3 (deeptools 3.5.6, has
# bigwigAverage; system deeptools 3.3.2 does not). computeMatrix/plotHeatmap still
# via system deeptools + patched plotHeatmap script (taller Shared group, wider
# profile<->heatmap gap, legend outside; fonts are set through the mplrc written
# below, since --fontSize does not exist in either deeptools version).
# Usage: run_peakheatmap.sh OUT LA "npA1,npA2" "bamA1,bamA2" LB "npB1,npB2" "bamB1,bamB2" [P]
set -e
OUT="$1"; LA="$2"; NPA="$3"; BAMA="$4"; LB="$5"; NPB="$6"; BAMB="$7"; P="${8:-32}"
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

# ---- Step1 (mirrors the reference analysis 1.input.py): narrowPeak -> bed, within-arm rep
# intersect -> group bed, then 3-way specific/shared split ----
np2bed(){ awk 'BEGIN{OFS="\t"}{print $1,$2,$3}' "$1" > "$2"; }
IFS=',' read -ra NPA_ARR <<< "$NPA"; IFS=',' read -ra NPB_ARR <<< "$NPB"
# Intermediates are keyed by LABEL, not by position: with positional names
# (a1/a2/b1/b2) a re-run after an arm-order change silently reused the OTHER
# arm's cached .cpm.bw.
np2bed "${NPA_ARR[0]}" "${LA}_r1.bed"; np2bed "${NPA_ARR[1]}" "${LA}_r2.bed"
np2bed "${NPB_ARR[0]}" "${LB}_r1.bed"; np2bed "${NPB_ARR[1]}" "${LB}_r2.bed"
bedtools intersect -a "${LA}_r1.bed" -b "${LA}_r2.bed" > "${LA}_group.bed"
bedtools intersect -a "${LB}_r1.bed" -b "${LB}_r2.bed" > "${LB}_group.bed"
bedtools intersect -a "${LA}_group.bed" -b "${LB}_group.bed" -v > "${LA}_specific.bed"
bedtools intersect -a "${LB}_group.bed" -b "${LA}_group.bed" -v > "${LB}_specific.bed"
bedtools intersect -a "${LA}_group.bed" -b "${LB}_group.bed" > shared.bed

# ---- Step2 (mirrors the reference analysis 2.peakheatmap_group.py Step1_bw/Step2_bw_group):
# CPM bigwig per rep -> group-average bigwig ----
IFS=',' read -ra BAMA_ARR <<< "$BAMA"; IFS=',' read -ra BAMB_ARR <<< "$BAMB"
cpm_bw(){ local bam="$1"; local out="$2"
  [ -s "$out" ] || "$CONDA/bamCoverage" --bam "$bam" --normalizeUsing CPM -o "$out" \
    --effectiveGenomeSize 2308125299 --exactScaling --binSize 10 --extendReads 100 -p "$P" >/dev/null 2>&1; }
cpm_bw "${BAMA_ARR[0]}" "${LA}_r1.cpm.bw"; cpm_bw "${BAMA_ARR[1]}" "${LA}_r2.cpm.bw"
cpm_bw "${BAMB_ARR[0]}" "${LB}_r1.cpm.bw"; cpm_bw "${BAMB_ARR[1]}" "${LB}_r2.cpm.bw"
[ -s "$LA.bw" ] || "$CONDA/bigwigAverage" -b "${LA}_r1.cpm.bw" "${LA}_r2.cpm.bw" -o "$LA.bw" -p "$P" >/dev/null 2>&1
[ -s "$LB.bw" ] || "$CONDA/bigwigAverage" -b "${LB}_r1.cpm.bw" "${LB}_r2.cpm.bw" -o "$LB.bw" -p "$P" >/dev/null 2>&1

# ---- Step3 (mirrors the reference analysis Step3_shared): matrix + heatmap, 3 region categories, purple x2 ----
[ -s matrix_group.gz ] || computeMatrix reference-point --referencePoint center \
  -R "${LA}_specific.bed" shared.bed "${LB}_specific.bed" \
  -S "$LA.bw" "$LB.bw" -b 3000 -a 3000 --skipZeros --missingDataAsZero -p "$P" -o matrix_group.gz >/dev/null 2>&1
$PHEAT -m matrix_group.gz -out peakheatmap_group.png \
  --regionsLabel "${LA}_specific" Shared "${LB}_specific" --xAxisLabel "" \
  --samplesLabel "$LA" "$LB" --colorList "white,purple" "white,purple" \
  --heatmapHeight 11.3 --missingDataColor white --dpi 200 >/dev/null 2>&1

cp peakheatmap_group.png peakheatmap.png
echo "OK $OUT/peakheatmap.png"
