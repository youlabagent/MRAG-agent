"""Citation harness for narrative.py — reduce PMID hallucination via:
 1) Whitelist enforcement (only retrieval-derived PMIDs allowed)
 2) Quote-grounded citation (verbatim snippet before each [PMID:X])
 3) Post-hoc parsing + validation (strip un-whitelisted, flag unverified)

Drop-in replacement for narrative.py's prompt + post-process pipeline.
"""
from __future__ import annotations
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

_BARE_CITATION = re.compile(r'\[PMID:(?P<pmid>\d+)\]')


def _norm(s: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation for fuzzy substring."""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def build_strict_prompt(
    tfs_block: str,
    targets_block: str,
    ag_block: str,
    passages: list,
    conditions: list = None,
) -> str:
    """Return prompt with whitelist + quote-grounded constraint baked in.

    passages: list of {"pmid": str, "title": str, "snippet": str}
    """
    whitelist = sorted({p["pmid"] for p in passages if p.get("pmid")})
    lit_block_lines = []
    for p in passages[:40]:
        lit_block_lines.append(
            f"[PMID:{p['pmid']}] {p.get('title','(no title)')}\n  {p.get('snippet','')[:500]}"
        )
    lit_block = "\n\n".join(lit_block_lines) if lit_block_lines else "(no passages retrieved)"

    # Build study context section
    if conditions:
        labels = [c.get("label","?") for c in conditions]
        cb = [
            "EXPERIMENTAL ARMS (perturbations applied to cells):",
            f"   Arm names = {labels} — use these EXACT strings, no abbreviations.",
            "",
            "IMPORTANT DISTINCTION:",
            "   - The \"perturbation\" / \"concept\" below is what was experimentally INTRODUCED (e.g. ONECUT2-OE = overexpression of ONECUT2).",
            "   - The \"Top TFs\" in REPORT DATA below are what was DISCOVERED as differentially active downstream.",
            "   These are DIFFERENT things. Do NOT conflate the perturbation TF with downstream identified TFs.",
            "",
            "Arm details:",
        ]
        for c in conditions:
            label = c.get("label", "?")
            pert = c.get("perturbation", "")
            concept = c.get("concept", "")
            cb.append(f"  - {label}: PERTURBATION = {pert}. CONCEPT = {concept}.")
        if any(c.get("rationale") for c in conditions):
            cb.append("\nComparison rationale (provided by experimenter):")
            for c in conditions:
                if c.get("rationale"):
                    cb.append(f"  - {c.get('label','?')}: {c['rationale']}")
        if any(c.get("rationale") for c in conditions):
            cb.append("\nComparison rationale (provided by experimenter):")
            for c in conditions:
                if c.get("rationale"):
                    cb.append(f"  - {c['rationale']}")
        cond_block = "\n".join(cb)
    else:
        cond_block = "(no study-level context provided)"

    return f"""You are a paper-grounded result writer for a cancer-transcription-regulation report.

# CITATION HARNESS — HARD RULES
1. ALLOWED CITATIONS: ONLY these {len(whitelist)} PMIDs may appear in your output:
   {whitelist}
   Citing ANY other PMID is a serious error and will be auto-removed.
2. QUOTE GROUNDING: Every [PMID:X] MUST be preceded by a short verbatim quote
   (10-30 words, in double quotes) drawn directly from the passage shown below
   for that exact PMID. Example:
       Tumor cells over-express MCT4: "lactate transporter MCT4 is up-regulated
       in hypoxic tumors" [PMID:28061459].
3. NO MEMORY CITATIONS: do NOT cite PMIDs from your training memory.
4. UNSOURCED CLAIMS: if no passage supports a claim, write it WITHOUT a citation;
   do not invent one.
5. Be result-oriented: describe what the data show + what the passages say.
   No speculative hypotheses, no hedging language.

# STUDY CONTEXT
{cond_block}

# REPORT DATA
## Top TFs
{tfs_block}

## Per-TF top targets
{targets_block}

## AlphaGenome in silico signal (SUPPORTIVE ONLY — see rules)
{ag_block}

RULES for AG:
- AG is a SUPPORTIVE in silico signal, NEVER a primary interpretive framework.
- DO NOT mention AG in the introduction, executive summary, or as the first
  evidence. Lead with ATAC + RNA + literature.
- If AG has zero direction-matched pairs, DO NOT dwell on that null result.
  Do NOT write sentences like "Despite differential expression... AlphaGenome
  yielded zero matches" or "absence of direction-consistent regulatory links".
- If AG has matched pairs, you MAY add one brief sentence noting which pairs
  it supports, but keep it under 30 words and place it AFTER the ATAC/RNA
  evidence has been established.
- Treat AG absence as silent (no commentary); treat AG support as a footnote.

## Literature passages (retrieved by paperqa — the ONLY allowed sources)
{lit_block}

# OUTPUT FORMAT
Write 4-6 paragraphs (~700-1000 words) of markdown narrative.

IMPORTANT: A study-context paragraph is already provided OUTSIDE this prompt.
Your output MUST START with the overall TF landscape paragraph (NO study context,
NO mention of "arm" / "perturbation" / "first condition" / "second condition" —
those are handled separately).

Required structure (start at Para 1 below):
- Para 1: Overall driver TF landscape — which TFs are most differential
  (use the TF names listed under "Top TFs"), what biological themes they suggest.
  Do NOT attribute TFs to specific experimental arms.
- Para 2-3: Per-TF target commentary — for each top TF, name top 1-2 targets
  and what literature says about that TF/target pair.
- Para 4: Cross-evidence consistency — how ATAC accessibility, RNA expression,
  and literature evidence converge. AlphaGenome MAY be mentioned briefly if and
  only if it has matched pairs supporting a specific finding; never lead with it,
  never describe AG null/absence results.
- Para 5: Limitations grounded in passages.

No preamble. No closing remarks. Start the markdown now."""


def validate_citations(text: str, passages: list) -> dict:
    """Verify each [PMID:X] is whitelisted + quote is in passage.

    Returns:
      {
        validated: [{pmid, quote, in_whitelist, quote_match}, ...],
        text_cleaned: str  # un-whitelisted citations replaced with [unsourced]
        n_total, n_kept, n_removed_whitelist, n_unverified_quote
      }
    """
    by_pmid = {p["pmid"]: (p.get("snippet", "") + " " + p.get("title", "")) for p in passages}
    whitelist = set(by_pmid.keys())

    validated = []
    spans_to_remove = []
    n_kept = 0; n_removed = 0; n_unverified_quote = 0

    for m in _BARE_CITATION.finditer(text):
        pmid = m.group("pmid")
        in_wl = pmid in whitelist
        # check quote in preceding text (last 300 chars before the citation)
        preceding = text[max(0, m.start() - 300):m.start()]
        # try to find a quoted snippet in preceding
        qm = re.search(r'["\u201c\u201d\u00ab\u00bb\u2018\u2019\']([^"\u201c\u201d\u00ab\u00bb\u2018\u2019\']{6,200})["\u201c\u201d\u00ab\u00bb\u2018\u2019\']\s*$', preceding)
        quote = (qm.group(1).strip() if qm else "")
        quote_match = False
        if quote and in_wl:
            psg = by_pmid[pmid]
            quote_match = _norm(quote)[:40] in _norm(psg)
        rec = {
            "pmid": pmid,
            "in_whitelist": in_wl,
            "quote": quote,
            "quote_verified": quote_match,
        }
        validated.append(rec)
        if not in_wl:
            spans_to_remove.append((m.start(), m.end(), "[unsourced]"))
            n_removed += 1
        elif not quote:
            # no preceding quote — keep citation but mark unverified
            spans_to_remove.append((m.start(), m.end(), f"[PMID:{pmid}·unverified]"))
            n_unverified_quote += 1
        elif not quote_match:
            # quote provided but doesn't appear in passage
            spans_to_remove.append((m.start(), m.end(), f"[PMID:{pmid}·misquoted]"))
            n_unverified_quote += 1
        else:
            n_kept += 1

    # rewrite text: replace flagged citations with marker
    cleaned = []
    last = 0
    for s, e, marker in sorted(spans_to_remove):
        cleaned.append(text[last:s])
        cleaned.append(marker)
        last = e
    cleaned.append(text[last:])
    text_cleaned = "".join(cleaned)

    return {
        "validated": validated,
        "text_cleaned": text_cleaned,
        "n_total": len(validated),
        "n_kept": n_kept,
        "n_removed_whitelist": n_removed,
        "n_unverified_quote": n_unverified_quote,
    }


def renumber_and_build_references(text: str, passages: list, audit_records: list) -> tuple:
    """Replace [PMID:X] / [PMID:X·unverified] / [PMID:X·misquoted] / [unsourced]
    in body with numbered refs [1], [2], ... and append a References section.

    Returns (new_text_with_refs, references_list).
    references_list: [{idx, pmid, title, status}, ...]
    """
    # passages: list of {pmid, title, snippet}
    title_by_pmid = {p["pmid"]: p.get("title", "") for p in passages}
    # audit_records: list of {pmid, in_whitelist, quote, quote_verified}
    # Map pmid -> worst status across all occurrences (verified > unverified > misquoted)
    status_by_pmid = {}
    for r in audit_records:
        pmid = r["pmid"]
        if not r.get("in_whitelist"):
            status_by_pmid[pmid] = "removed"
            continue
        q = (r.get("quote") or "").strip()
        v = bool(r.get("quote_verified"))
        if v:
            cur = "verified"
        elif q:
            cur = "misquoted"
        else:
            cur = "unverified"
        # keep the WORST status if mixed
        rank = {"verified": 0, "unverified": 1, "misquoted": 2, "removed": 3}
        prev = status_by_pmid.get(pmid)
        if prev is None or rank[cur] > rank[prev]:
            status_by_pmid[pmid] = cur

    # First pass: collect PMIDs in order of first appearance
    pat = re.compile(r"\[PMID:(\d+)(?:·(?:unverified|misquoted))?\]")
    order = []
    seen = set()
    for m in pat.finditer(text):
        pmid = m.group(1)
        if pmid not in seen:
            order.append(pmid)
            seen.add(pmid)
    idx_by_pmid = {pmid: i + 1 for i, pmid in enumerate(order)}

    # Build replacement function
    def repl(m):
        pmid = m.group(1)
        return f"[{idx_by_pmid[pmid]}]"

    new_text = pat.sub(repl, text)

    # Build references list
    refs = []
    for pmid in order:
        refs.append({
            "idx": idx_by_pmid[pmid],
            "pmid": pmid,
            "title": title_by_pmid.get(pmid, "(no title)"),
            "status": status_by_pmid.get(pmid, "unverified"),
        })

    return new_text, refs


def format_references_markdown(refs: list) -> str:
    lines = ["", "", "## References", ""]
    status_glyph = {
        "verified":    "✓",   # check
        "unverified":  "°",    # degree (subtle)
        "misquoted":   "⚠",   # warning
        "removed":     "✗",   # cross
    }
    status_text = {
        "verified":   "quote verified",
        "unverified": "quote not provided",
        "misquoted":  "quote not found in passage",
        "removed":    "PMID not in retrieval whitelist",
    }
    for r in refs:
        glyph = status_glyph.get(r["status"], "")
        st = status_text.get(r["status"], "")
        url = f"https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/"
        lines.append(
            f"[{r['idx']}] PMID:{r['pmid']} {glyph} — *{r['title']}* "
            f"([PubMed]({url})) <span style=\"color:#888;font-size:11px\">[{st}]</span>"
        )
    return "\n".join(lines)
