"""SEMotifNode — SE -> TF motif presence + SE / linked-gene attributes.

Given the top-N super-enhancers (SERoseNode), scans them with
  * the report's TF panel PWMs (combined/top{N}.json symbols, taken from the same
    JASPAR-extended library the Step-2 chain used: homer_motif/<treat>_peak_minmax.motif;
    a PWM is assigned to a panel TF by homer_mcp.tools.tf_alias — alias-aware
    (ANDR/ARE/AR-halfsite -> AR, GCR/GRE -> NR3C1, AP-2alpha -> TFAP2A), dimers and
    composites count for each member; the same rule produce_trn_inputs / the
    Step-2 column filter use for TF -> Target edges), and
  * the Section-8 enriched known motifs (homer_known/<treat>/knownResults/known1..n.motif,
    HOMER PWMs; TF symbol derived as in render._build_homer_known, ':' -> '_').
HOMER annotatePeaks.pl -size given -mbed gives every motif instance; an SE -> TF
edge = at least one instance (presence rule, 2026-09-07 decision), with
hits/kb and best log-odds kept for edge width / tooltips.

SE attributes: mean DiffBind Fold of peaks inside the SE (fill colour,  rule)
and the ROSE CLOSEST gene's DESeq2 log2FC (border colour of the SE-linked gene).

MCP step 2a (2026-10-12): the stage is the homer-mcp tool scan_super_enhancer_motifs
(mcps/homer_mcp/src/homer_mcp/tools/se_motif.py -- the former body of run() and its helpers, unchanged); run()
resolves the params and calls it through capture_mcp_tools (MRAG_TOOL_TRANSPORT=mcp: an MCP tools/call; =direct:
the function).
"""
from __future__ import annotations
from ._base import Node
from . import _stage_tools
import sys as _sys
_sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/homer_mcp/src")
from homer_mcp.tools.tf_alias import motif_matches as _mmatch, motif_symbols as _msym, primary_symbol as _psym   # alias-aware (ANDR/ARE -> AR ...)
from homer_mcp.tools.se_motif import (HOMER_BIN, NAME_RE, _tf_symbol_from_known, _read_motif_lib,   # noqa: F401
                                      _base)


class SEMotifNode(Node):
    """Inputs:
        se_top_tsv, se_homerpeak (SERoseNode), panel_json (combined/top{N}.json),
        motif_lib (homer_motif/<treat>_peak_minmax.motif), known_dir (homer_known/<treat>),
        diffbind_csv, deseq_csv, label
    Params: n_known (10), genome (hg38), cpu (16), timeout_s (3600), homer_bin
    Outputs: motif_file, motif_map_tsv, hits_bed, annot_txt, se_tf_edges_tsv,
             se_tf_matrix_tsv, se_genes_tsv, n_tf, n_edges, panel_tfs, enriched_tfs
    """
    name = "se_motif"

    def run(self) -> None:
        p = self.params; i = self.inputs
        tools = _stage_tools.tools("se_motif")
        self.outputs = tools["scan_super_enhancer_motifs"](
            se_top_tsv=str(i["se_top_tsv"]), se_homerpeak=str(i["se_homerpeak"]), panel_json=str(i["panel_json"]),
            motif_lib=str(i["motif_lib"]), known_dir=str(i["known_dir"]), diffbind_csv=str(i["diffbind_csv"]),
            deseq_csv=str(i["deseq_csv"]), output_dir=str(self.output_dir),
            genome=p.get("genome", "hg38"), cpu=int(p.get("cpu", 16)), n_known=int(p.get("n_known", 10)),
            n_backfill_scan=int(p.get("n_backfill_scan", 15)), timeout_s=int(p.get("timeout_s", 3600)),
            homer_bin=p.get("homer_bin", HOMER_BIN))

    def validate(self):
        n = int(self.outputs.get("n_edges", 0) or 0)
        g = [{"name": "se_tf_edges", "result": "pass" if n > 0 else "fail", "value": n,
              "msg": "" if n > 0 else "no SE->TF motif instances found"}]
        miss = self.outputs.get("panel_without_pwm", "")
        if miss:
            g.append({"name": "panel_pwm_coverage", "result": "warn", "value": miss,
                      "msg": f"panel TFs without a PWM in the library: {miss}"})
        return g
