#!/bin/bash
# GIGGLE TF-overlap on condition-open peak sets, fully local (no Cistrome web).
# Matches the reference analysis script 3.plot.R METHODOLOGY: giggle search vs local Cistrome hg38
# TF index -> result.csv (all datasets) -> keep TOP-N datasets by score -> per-
# Factor mean GIGGLE top-10 point plot (treat=orange, control=blue). The TOP-N
# cut is essential: without it a Factor's mean is diluted by its many irrelevant
# cell-type datasets (AR 442 datasets), sinking real drivers below noise factors.
# Usage: run_giggle.sh <treat_bed> <control_bed> <out_dir> [treat_label] [control_label] [top_n] [genome]
#
# 2026-09-24 genome selector: [genome] (7th arg), else env MRAG_GIGGLE_GENOME,
# else hg38. It picks the index, the GSM->Factor map and the genome size used
# for giggle's significance test:
#   hg38  giggle_index_hg38_tf + gsm_factor_map.tsv, no -g (giggle's default
#         3095677412). This is the ORIGINAL behaviour: the 6-argument call is
#         byte-identical to the script before the genome selector was added.
#   mm10  giggle_index_mm10_tf + gsm_factor_map_mm10.tsv, -g 2725521370
#         (mm10 chr1-19,X,Y from ATAC_mm10/mm10.chrom.sizes; giggle's default
#         is the human chr1-22,X,Y length, which would overstate the mouse
#         background by 13.6%). See README_mm10_giggle.md.
# Factor names are upper-cased for BOTH genomes: MRAG's TF namespace (HOCOMOCO /
# DAStk, the HOMER alias table, the combined-pipeline join key) is upper-case
# for mouse as well as human.
set -e
TREAT="$(readlink -f "$1")"; CTRL="$(readlink -f "$2")"; OUT="$3"
TLAB="${4:-treatment}"; CLAB="${5:-control}"; TOPN="${6:-200}"
GENOME="${7:-${MRAG_GIGGLE_GENOME:-hg38}}"
CIS=/BIO1/MRAG/Agent/Reference/cistrome
G=/BIO1/MRAG/Agent/bin/giggle/bin/giggle
case "$GENOME" in
  hg38) IDX=$CIS/giggle_index_hg38_tf; MAP=$CIS/gsm_factor_map.tsv;      GSIZE="" ;;
  mm10) IDX=$CIS/giggle_index_mm10_tf; MAP=$CIS/gsm_factor_map_mm10.tsv; GSIZE=2725521370 ;;
  *) echo "run_giggle.sh: unsupported genome '$GENOME' (expected hg38 or mm10)" >&2; exit 2 ;;
esac
[ -d "$IDX" ] || { echo "run_giggle.sh: GIGGLE index for $GENOME not found: $IDX" >&2; exit 2; }
[ -s "$MAP" ] || { echo "run_giggle.sh: GSM->Factor map for $GENOME not found: $MAP" >&2; exit 2; }
mkdir -p "$OUT"; cd "$OUT"
cut -f1-3 "$TREAT" > up.bed
cut -f1-3 "$CTRL"  > down.bed
python3 - "$G" "$IDX" "$MAP" "$GSIZE" <<'PY'
import sys,subprocess,re
G,IDX,MAP,GSIZE=sys.argv[1:5]
m={}
for l in open(MAP):
    t=l.rstrip("\n").split("\t")
    if len(t)>=2: m[t[0]]=t[1]
for b in ("up","down"):
    subprocess.run(f"sort -k1,1 -k2,2n {b}.bed | bgzip > {b}.bed.gz",shell=True,check=True)
    cmd=[G,"search","-i",IDX,"-q",f"{b}.bed.gz","-s"]+(["-g",GSIZE] if GSIZE else [])
    raw=subprocess.run(cmd,capture_output=True,text=True).stdout
    o=open(f"{b}.bed_result.csv","w"); o.write("Factor,GIGGLE_score,GSM\n")
    for line in raw.splitlines():
        if line.startswith("#"): continue
        t=line.split("\t")
        if len(t)<8: continue
        mm=re.search(r"(GSM\d+)",t[0])
        if not mm: continue
        fac=m.get(mm.group(1))
        if fac and t[7]: o.write(f"{fac.upper()},{t[7]},{mm.group(1)}\n")
    o.close()
PY
Rscript - "$TLAB" "$CLAB" "$TOPN" <<'PY'
args<-commandArgs(trailingOnly=TRUE); tlab<-args[1]; clab<-args[2]; topn<-as.integer(args[3])
suppressMessages({library(dplyr);library(ggplot2);library(readr)})
draw<-function(csv,title,hi,outpng){
  df<-read_csv(csv,show_col_types=FALSE)
  df<-df %>% arrange(desc(GIGGLE_score)) %>% head(topn)   # TOP-N datasets cut
  s<-df%>%group_by(Factor)%>%summarize(mean_GIGGLE=mean(GIGGLE_score,na.rm=TRUE))%>%arrange(desc(mean_GIGGLE))
  t10<-head(s,10); t10$Factor<-factor(t10$Factor,levels=t10$Factor)
  p<-ggplot(t10,aes(Factor,mean_GIGGLE))+geom_point(aes(fill=mean_GIGGLE),shape=21,size=5,color="black",stroke=1)+
    scale_fill_gradient(low="white",high=hi)+theme_classic()+
    theme(axis.text.x=element_text(angle=45,hjust=1,vjust=1,color="black",size=11),
          axis.text.y=element_text(color="black",size=11),axis.title=element_text(face="bold"))+
    labs(x="",y="Average GIGGLE Score",fill="GIGGLE Score",title=title)
  ggsave(outpng,p,width=8,height=6,dpi=300)
}
draw("up.bed_result.csv",paste0(tlab,"_open"),"#D95F02","treatment_giggle.png")
draw("down.bed_result.csv",paste0(clab,"_open"),"#28347A","control_giggle.png")
PY
echo "GIGGLE done (top-$TOPN cut) -> $OUT/{treatment,control}_giggle.png"
