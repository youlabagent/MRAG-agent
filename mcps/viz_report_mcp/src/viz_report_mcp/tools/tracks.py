"""Wrap mrag_track_helper.plot_atac_tracks as MCP tool.

Single locus snapshot PNG (multiple bigwig tracks stacked, shared y-axis).
"""
from __future__ import annotations

import base64
import importlib.util
import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union


HELPER_PATH = str(Path(os.environ.get("MRAG_HOME", Path(__file__).resolve().parents[5]))
                  / "libs" / "track_helper" / "mrag_track_helper.py")

# The four bigWigs of the ONECUT2-OE demo experiment. They only exist on the
# original host, so the directory comes from MRAG_DEMO_ATAC_BW and the list is
# empty when it is unset; callers pass their own tracks in the normal path.
_DEMO_BW = os.environ.get("MRAG_DEMO_ATAC_BW", "")
STANDARD_SAMPLES: List[Tuple[str, str, Optional[str]]] = ([
    ("ONECUT2_OE_1a", str(Path(_DEMO_BW) / "1_pLenti_ONECUT2_OE_1a_merge.bw"), None),
    ("ONECUT2_OE_1b", str(Path(_DEMO_BW) / "2_pLenti_ONECUT2_OE_1b_merge.bw"), None),
    ("cMyc_OE_2a",    str(Path(_DEMO_BW) / "3_pLenti_cMyc_DDK_2a_merge.bw"),    None),
    ("cMyc_OE_2b",    str(Path(_DEMO_BW) / "4_pLenti_cMyc_DDK_2b_merge.bw"),    None),
] if _DEMO_BW else [])


def _load_helper():
    spec = importlib.util.spec_from_file_location("mrag_track_helper", HELPER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mrag_track_helper"] = mod
    spec.loader.exec_module(mod)
    return mod


def register_track_tools(mcp):

    @mcp.tool()
    def plot_locus_track(
        gene_label: str,
        region: str,
        out_dir: str,
        title: Optional[str] = None,
        samples: Optional[List[List[Optional[str]]]] = None,    # [label, bigwig, narrowPeak or None]
        max_value: str = "shared_max",
        width: float = 14.0,
        dpi: int = 200,
        flank_bp: int = 0,
        highlight: Optional[List[Union[str, int]]] = None,     # [chrom, start, end] or [start, end]
        colors: Optional[List[List[str]]] = None,
        gene_gtf: Optional[str] = None,
    ) -> dict:
        """Render a single locus track PNG (multi-sample stacked).

        Args:
            gene_label: gene name for top header (e.g. "AZGP1")
            region: 'chr:start-end' string
            out_dir: directory for the PNG/PDF files
            title: short slug for filename (default: gene_label)
            samples: list of [label, bigwig_path, narrowpeak_path|None] tuples.
                     Default =  4-sample standard set
            max_value: 'shared_max' / 'shared_auto' / 'q99' / number ...
            width: cm
            dpi: image DPI
            flank_bp: extend region by ±N bp
            colors: per-condition [dark, light] hex pairs, ordered to match the
                    condition order of `samples` (2 reps per condition).
                    House convention: treatment = red, control = blue.
            gene_gtf: tabix-indexed GTF for the gene-model track. Default (None)
                    keeps the helper's HUMAN hg38 default -- which on a non-human
                    run draws the wrong species' genes under a correct caption,
                    so an organism-aware caller must pass this.
        Returns: {ok, png, pdf, max_value_used, region}
        """
        h = _load_helper()
        s = samples if samples else STANDARD_SAMPLES
        # cast lists -> tuples for helper (Sequence[Tuple] expected)
        sample_tuples = [(x[0], x[1], x[2] if len(x) > 2 else None) for x in s]

        # extend region if needed
        if flank_bp > 0:
            chrom, span = region.split(":")
            a, b = span.replace(",", "").split("-")
            region = f"{chrom}:{max(0, int(a) - flank_bp)}-{int(b) + flank_bp}"

        result = h.plot_atac_tracks(
            samples=sample_tuples,
            region=region,
            title=(title or gene_label),
            gene_title=gene_label,
            out_dir=out_dir,
            max_value=max_value,
            dpi=dpi,
            width=width,
            highlight=tuple(highlight) if highlight else None,
            colors=([tuple(c) for c in colors] if colors else None),
            **({"gene_gtf": gene_gtf} if gene_gtf else {}),
        )
        return {
            "ok": True,
            "gene_gtf": gene_gtf or "",
            "png": result.get("png"),
            "pdf": result.get("pdf"),
            "max_value_used": result.get("max_value_used"),
            "region": region,
            "gene_label": gene_label,
        }

    @mcp.tool()
    def encode_png_base64(png_path: str) -> dict:
        """Return base64-encoded PNG for inline HTML embedding."""
        p = Path(png_path)
        if not p.exists():
            return {"ok": False, "error": f"not found: {png_path}"}
        b = p.read_bytes()
        return {
            "ok": True,
            "data_url": "data:image/png;base64," + base64.b64encode(b).decode(),
            "size_bytes": len(b),
        }
