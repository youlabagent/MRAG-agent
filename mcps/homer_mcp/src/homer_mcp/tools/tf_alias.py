"""TF symbol <-> motif-name aliasing (2026-09-10).

One rule for every place that has to decide whether a HOMER / JASPAR / HOCOMOCO
motif column or PWM belongs to a panel TF:
  *  Step-2 column filter (homer_mcp.tools.filter.filter_target_tfs)
  * TF -> Target edges (viz_report_mcp.tools.cytoscape_trn.produce_trn_inputs)
  * curated targets (mrag_orchestrator.nodes.curated_targets)
  * SE -> TF motif scan (mrag_orchestrator.nodes.se_motif)

A motif name is reduced to its base ("ANDR(HOCOMOCOv12)/HOCOMOCOv12/Homer" -> "ANDR",
"FOXA1:AR(Forkhead,NR)/..." -> "FOXA1:AR"), split into tokens on '::' / ':' (dimers
and composite motifs count for every member) and each token is mapped to an HGNC
symbol through HOCOMOCO short names (ANDR -> AR, GCR -> NR3C1, AP2A -> TFAP2A, ...)
and HOMER's own naming (ARE -> AR, AR-halfsite -> AR, GRE -> NR3C1, AP-2alpha -> TFAP2A,
...). Matching is case-insensitive. The composite form itself ("FOX_EBOX") is also
returned so non-gene motifs can be carried as their own node.
"""
from __future__ import annotations
import re, sys
from functools import lru_cache

try:
    sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/combined_mcp/src")
    from combined_mcp.tools.pipeline import HOCOMOCO_TO_GENE as _HOCO
except Exception:  # pragma: no cover - minimal fallback when combined_mcp is unavailable
    _HOCO = {"ANDR": "AR", "GCR": "NR3C1", "AP2A": "TFAP2A", "AP2B": "TFAP2B", "AP2C": "TFAP2C",
             "PRGR": "PGR", "ESR1": "ESR1", "NR3C1": "NR3C1"}

# HOMER known-motif naming (upper-cased base -> symbol). Extend as needed.
HOMER_NAME_ALIASES = {
    "ARE": "AR", "AR-HALFSITE": "AR", "ANDR": "AR",
    "GRE": "NR3C1", "GCR": "NR3C1", "GR": "NR3C1",
    "PGR": "PGR", "PR": "PGR", "PRE": "PGR", "PRGR": "PGR",
    "ERE": "ESR1", "ERA": "ESR1", "ESR1": "ESR1",
    "AP-2ALPHA": "TFAP2A", "AP2A": "TFAP2A", "AP-2GAMMA": "TFAP2C", "AP2C": "TFAP2C", "AP-2BETA": "TFAP2B",
    # HOMER legacy / common names (upper-cased motif base -> HGNC symbol)
    "C-MYC": "MYC", "N-MYC": "MYCN", "L-MYC": "MYCL", "MAX": "MAX",
    "OCT4": "POU5F1", "OCT2": "POU2F2", "OCT6": "POU3F1", "OCT11": "POU2F3", "BRN1": "POU3F3", "BRN2": "POU3F2",
    "ERRA": "ESRRA", "ERRB": "ESRRB", "ERRG": "ESRRG", "ERB": "ESR2",
    "FRA1": "FOSL1", "FRA2": "FOSL2", "JUNB": "JUNB", "JUND": "JUND", "C-JUN": "JUN", "C-FOS": "FOS",
    "P53": "TP53", "P63": "TP63", "P73": "TP73",
    "NFKB-P65": "RELA", "NFKB-P65-REL": "RELA", "NFKB-P50": "NFKB1", "NFKB-P52": "NFKB2",
    "PU.1": "SPI1", "SPIB": "SPIB", "EKLF": "KLF1", "NRF2": "NFE2L2", "BMAL1": "ARNTL",
    "HIF-1A": "HIF1A", "HIF-1B": "ARNT", "HIF2A": "EPAS1", "HIF-2A": "EPAS1",
    "COUP-TFII": "NR2F2", "EAR2": "NR2F6", "TR4": "NR2C2", "LRH1": "NR5A2", "SF1": "NR5A1",
    "RUNX-AML": "RUNX1", "AML1": "RUNX1", "NF-E2": "NFE2", "HNF1B": "HNF1B", "HNF4A": "HNF4A",
    "ZNF143|STAF": "ZNF143", "STAF": "ZNF143", "BORIS": "CTCFL", "NRSF": "REST", "REST-NRSF": "REST",
    "AP-2ALPHA-BETA": "TFAP2A", "LEF1": "LEF1", "TCFL2": "TCF7L2",
    "NUR77": "NR4A1", "FXR": "NR1H4", "LXR": "NR1H3", "PPARA": "PPARA", "RORGT": "RORC",
    # NOTE: family-level HOMER motifs ("GATA(Zf)", "EGR", "AP-1", "CEBP", "ETS", "TEAD",
    # "PPARE", "Sox") are deliberately NOT mapped to a single member gene.
}


# HOMER names that collide with other genes' HGNC symbols: HOMER 'Tcf4(HMG)' / 'Tcf3(HMG)'
# are the Wnt HMG factors TCF7L2 / TCF7L1, whereas JASPAR/HOCOMOCO 'TCF4' / 'TCF3' are the
# bHLH E-proteins -> alias only when the HOMER family tag is HMG.
HOMER_FAMILY_ALIASES = {("TCF4", "HMG"): "TCF7L2", ("TCF3", "HMG"): "TCF7L1"}

try:
    from combined_mcp.tools.pipeline import _hocomoco_to_symbol as _H2S   # same rule as INTEGRATED.csv
except Exception:  # pragma: no cover
    _H2S = None


def _hoco(u: str) -> str:
    # HOMER dotted NK-family names -> HGNC hyphen (Nkx3.1 -> NKX3-1, Nkx2.1 -> NKX2-1)
    if re.match(r"^NKX\d+\.\d+$", u):
        return u.replace(".", "-")
    if _H2S is not None:
        try:
            v = _H2S(u)
            if v != u:
                return v.upper()
        except Exception:
            pass
    v = _HOCO.get(u, u)
    if len(v) > 2 and v.startswith("ZN") and v[2].isdigit():
        v = "ZNF" + v[2:]
    return v


def _family(name: str) -> str:
    m = re.match(r"^[^(/]*\(([^)]*)\)", name or "")
    return (m.group(1).split(",")[0].strip().upper() if m else "")


def motif_base(name: str) -> str:
    """'PANEL_AR(Ar)/JASPAR2024/Homer' stays as given; base = text before '(' and '/'."""
    return (name or "").split("(")[0].split("/")[0].strip()


@lru_cache(maxsize=None)
def motif_symbols(name: str) -> frozenset:
    """All HGNC-style symbols a motif name stands for (upper-case), plus the
    composite form of a multi-TF motif ('FOX_EBOX', 'FOXA1_AR')."""
    base = motif_base(name)
    # strip our own scan prefixes (PANEL_/ENRICHED_/TOP10_) if present
    m = re.match(r"^(?:PANEL|ENRICHED|TOP10|TOP)_(.+)$", base)
    if m:
        base = m.group(1)
    out = set()
    fam = _family(name)
    toks = [t.strip() for t in re.split(r"::|:", base) if t.strip()]
    for tok in toks:
        u = tok.upper()
        u = HOMER_FAMILY_ALIASES.get((u, fam)) or HOMER_NAME_ALIASES.get(u) or _hoco(u)
        out.add(u)
    if len(toks) > 1:
        out.add(re.sub(r"[^A-Z0-9_\-]", "_", base.upper().replace("::", "_").replace(":", "_")))
    return frozenset(out)


def motif_matches(name: str, tf: str) -> bool:
    """True when motif `name` is a PWM/column for panel TF `tf` (alias-aware)."""
    return bool(tf) and tf.strip().upper() in motif_symbols(name)


def primary_symbol(name: str) -> str:
    """Symbol of the first token (for single-TF motifs, the TF itself)."""
    base = motif_base(name)
    m = re.match(r"^(?:PANEL|ENRICHED|TOP10|TOP)_(.+)$", base)
    if m:
        base = m.group(1)
    tok = re.split(r"::|:", base)[0].strip().upper()
    return HOMER_FAMILY_ALIASES.get((tok, _family(name))) or HOMER_NAME_ALIASES.get(tok) or _hoco(tok)
