#!/usr/bin/env python
# MRAG addition (2026-09-11). ROSE stitching WITHOUT read mapping, for inputs that
# have no BAM (e.g. processed-matrix ATAC with bigWig signal). Uses ROSE_main's own
# regionStitching() and reproduces the locus table that mapCollection() builds
# (chrY removed, ordered by stitched length, NUM_LOCI, CONSTITUENT_SIZE), so the
# only step that differs from ROSE_main is how the per-region signal is obtained.
import os, sys
from optparse import OptionParser
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ROSE_utils
from ROSE_main import regionStitching, checkRefCollection

def main():
    p = OptionParser(usage="%prog -i input.gff -g annot.ucsc -s 12500 -t 0 -o stitched_table.txt")
    p.add_option("-i", dest="gff"); p.add_option("-g", dest="genome")
    p.add_option("-s", dest="stitch", default=12500); p.add_option("-t", dest="tss", default=0)
    p.add_option("-o", dest="out"); p.add_option("-f", dest="gff_out", default="")
    o, a = p.parse_args()
    stitch = int(o.stitch); tss = int(o.tss); removeTSS = tss != 0
    ref = ROSE_utils.gffToLocusCollection(o.gff)
    checkRefCollection(ref)
    stitched, debug = regionStitching(o.gff, stitch, tss, o.genome, removeTSS)
    if o.gff_out:
        ROSE_utils.unParseTable(ROSE_utils.locusCollectionToGFF(stitched), o.gff_out, '\t')
    loci = stitched.getLoci()
    loci = [l for l in loci if l.chr() != 'chrY']
    order = ROSE_utils.order([l.len() for l in loci], decreasing=True)
    table = [['REGION_ID', 'CHROM', 'START', 'STOP', 'NUM_LOCI', 'CONSTITUENT_SIZE']]
    for i in order:
        l = loci[i]
        size = sum(r.len() for r in ref.getOverlap(l, 'both'))
        try:
            n = int(l.ID().split('_')[0])
        except ValueError:
            n = 1
        table.append([l.ID(), l.chr(), l.start(), l.end(), n, size])
    ROSE_utils.unParseTable(table, o.out, '\t')
    print('STITCHED %d regions from %d constituents' % (len(table) - 1, len(ref.getLoci())))

if __name__ == '__main__':
    main()
