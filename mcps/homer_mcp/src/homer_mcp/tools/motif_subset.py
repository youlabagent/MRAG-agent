"""subset_motif_file — extract specific TF PWMs from a HOMER motif library.

Critical efficiency tool: annotatePeaks.pl scans EVERY PWM at EVERY peak,
so running with a 2322-PWM library when we only care about 10 selected TFs
wastes ~200x the time. Building a small motif file containing just the
PWMs for our Top-N TF list lets the full HOMER pipeline complete in seconds
instead of hours.

HOMER motif file format:
    >CONSENSUS\\tTFNAME(SOURCE)/.../...\\tlogOdds\\tlogP\\t...
    PWM row 1 (4 cols: A C G T)
    PWM row 2
    ...
    >CONSENSUS  (next motif)
    ...
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional


_HEADER_RE = re.compile(r"^>([ACGTNacgtn]+)\s+(\S+)")


def _extract_primary_tf(header_name: str) -> str:
    """Extract the primary TF symbol from a HOMER motif header name.

    Examples:
        'ETV1(HOCOMOCOv12)/HOCOMOCOv12/Homer'  -> 'ETV1'
        'TFAP4::ETV1(JASPAR2024)/JASPAR2024/Homer' -> 'TFAP4::ETV1'
        'NF-kB-p65(RHD)/GM12878-p65-ChIP-Seq(...)' -> 'NF-kB-p65'
    """
    paren = header_name.find("(")
    if paren > 0:
        return header_name[:paren]
    slash = header_name.find("/")
    if slash > 0:
        return header_name[:slash]
    return header_name


def _matches_tf(tf_header: str, target_symbols_upper: set, include_heterodimers: bool) -> bool:
    """Decide whether a motif's TF symbol matches any of our target TFs."""
    primary = _extract_primary_tf(tf_header).upper()
    if primary in target_symbols_upper:
        return True
    if include_heterodimers and "::" in primary:
        for part in primary.split("::"):
            if part.strip() in target_symbols_upper:
                return True
    return False


def _iter_motif_blocks(path: str):
    """Yield (header_line, body_lines_list) for each motif block."""
    with open(path, "r") as f:
        header = None
        body = []
        for line in f:
            if line.startswith(">"):
                if header is not None:
                    yield header, body
                header = line
                body = []
            else:
                body.append(line)
        if header is not None:
            yield header, body


def subset_motif_file_impl(
    motif_db: str,
    tf_symbols: List[str],
    output_motif: str,
    include_heterodimers: bool = True,
) -> dict:
    """Write a small HOMER motif file containing only PWMs matching tf_symbols."""
    targets_upper = {t.strip().upper() for t in tf_symbols if t and t.strip()}
    if not targets_upper:
        raise ValueError("tf_symbols cannot be empty")

    out_path = Path(output_motif)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept_motifs = []   # list of dicts: {consensus, tf_name, n_rows}
    total_seen = 0
    with open(out_path, "w") as fo:
        for header, body in _iter_motif_blocks(motif_db):
            total_seen += 1
            m = _HEADER_RE.match(header.strip())
            if not m:
                continue
            consensus, tf_name = m.group(1), m.group(2)
            if not _matches_tf(tf_name, targets_upper, include_heterodimers):
                continue
            fo.write(header)
            for b in body:
                fo.write(b)
            kept_motifs.append({
                "consensus": consensus,
                "tf_name": tf_name,
                "primary": _extract_primary_tf(tf_name),
                "n_rows": len(body),
            })

    # Coverage check
    primary_kept = {k["primary"].upper() for k in kept_motifs}
    missing = sorted(t for t in targets_upper if t not in primary_kept
                     and not any(t in (k["primary"].upper().split("::")) for k in kept_motifs))

    return {
        "ok": True,
        "output_motif": str(out_path),
        "n_total_motifs_in_db": total_seen,
        "n_kept": len(kept_motifs),
        "kept_motifs": kept_motifs,
        "missing_tfs": missing,
        "size_bytes": out_path.stat().st_size,
    }


def register_motif_subset_tools(mcp):

    @mcp.tool()
    def subset_motif_file(
        motif_db: str,
        tf_symbols: List[str],
        output_motif: str,
        include_heterodimers: bool = True,
    ) -> dict:
        """Extract specific TF PWMs from a HOMER motif library.

        Args:
            motif_db: source motif file (HOMER format, possibly thousands of PWMs)
                      e.g. /BIO1/MRAG/Agent/Reference/combined_v12_jaspar.motif
            tf_symbols: TF gene symbols to keep (case-insensitive exact match,
                        e.g. ['ETV1','GATA3',...] from mcp_combined Top-N output)
            output_motif: path for the subset motif file
            include_heterodimers: if True (default), also include heterodimer
                        motifs like 'TFAP4::ETV1' when one half is in tf_symbols
        Returns:
            {ok, output_motif, n_total_motifs_in_db, n_kept, kept_motifs[],
             missing_tfs[], size_bytes}
        """
        return subset_motif_file_impl(
            motif_db=motif_db, tf_symbols=tf_symbols,
            output_motif=output_motif,
            include_heterodimers=include_heterodimers,
        )
