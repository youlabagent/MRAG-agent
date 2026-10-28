"""viz_report_mcp wrapper.

Two nodes:
  - NarrativeNode: generate narrative + abstract via Qwen3.6-27B (LLM call inside)
  - RenderNode:    assemble final HTML from per-N artifacts + narrative

Both import viz_report_mcp directly.
"""
from __future__ import annotations
import os
import csv, json, sys
from pathlib import Path
from typing import Any, Optional

# Add viz_report_mcp source to import path
sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/viz_report_mcp/src")

from ._base import Node, capture_mcp_tools


# Bump when the narrative / abstract prompt contract changes: it is part of the
# NarrativeNode params, hence of Node.input_hash, so a resumed run never reuses
# a cached narrative built by an older prompt.
# novdom_20260917: per-literature-domain Novelty on the Section-14 cards (params
# "novelty_domains"), one batch retriever process for the card retrieval.
# novdom_20260919: a would-be "unreported" domain is checked against live PubMed
# first (0 hits -> unreported, >0 -> the neutral "corpus_absent"), and the
# per-domain GeneRIF gate reads every GeneRIF instead of only the newest 8.
NARRATIVE_PROMPT_VERSION = "novdom_20260919"


def notable_cards_stale(cards, novelty_domains) -> str:
    """Why an existing notable_cards.json must be rebuilt ('' = up to date):
    empty, or (when the run has literature domains) built before per-domain
    Novelty, by another novelty_domains_version, or for another domain list /
    order than `novelty_domains` (corpus keys, user order)."""
    if not cards:
        return "cards empty"
    try:
        from ..literature import known_keys
        keys = known_keys(novelty_domains or [])
    except Exception:
        keys = [str(k) for k in (novelty_domains or []) if k]
    if not keys:
        return ""
    from viz_report_mcp.tools.narrative import NOVELTY_DOMAINS_VERSION
    for c in cards:
        if not isinstance(c, dict) or not c.get("novelty_by_domain"):
            return "cards predate per-domain Novelty"
        if c.get("novelty_domains_version") != NOVELTY_DOMAINS_VERSION:
            return f"novelty_domains_version {c.get('novelty_domains_version')!r} != {NOVELTY_DOMAINS_VERSION!r}"
        if [d.get("key") for d in c["novelty_by_domain"]] != keys:
            return (f"card Novelty domains {[d.get('key') for d in c['novelty_by_domain']]} "
                    f"!= run domains {keys}")
    return ""


class CardEvidenceUnavailable(RuntimeError):
    """followup r3 (post-deploy finding 2): the evidence context of a curated run's Section-14 cards could not be
    built, so no card may be built (or shown) for that run -- a card built without it can cite a paper the
    evidence gate's filters removed (e.g. a screened-negative gene list, an off-domain sentence)."""


# followup r3 (post-deploy finding 2): the card rule "never cite a PMID the gate's evidence filters removed" held
# only on the happy path.  card_evidence() turned ANY error (an unreadable curated_targets.json / scored.json /
# _generif_fetch.json, a failed context build) into None and the cards were then built WITHOUT the rules, with a
# print as the only trace; and a failed rebuild left the narrative-time cards (built before the curated targets
# existed, never with the rules) in place, recorded as a 'skipped' step.  Now, for a curated run with the evidence
# layers on and literature domains ("strict"): an unreadable curated_targets.json or a failed context build raises
# CardEvidenceUnavailable (no card is built); an unreadable scored.json / sidecar is left out -- the context is
# built from the rows, which carry the gate's filtered sets (only the scorer's GeneRIF-line reuse and the runner's
# 1a verdicts are lost; the card then reads / checks those itself) -- and noted; and the runner WITHHOLDS the
# cards of a curated run that were not rebuilt with the rules (withhold_unfiltered_cards: moved aside, an empty
# notable_cards.json, a status record Section 14 shows, a FAILED manifest step).  With the evidence layers off
# (MRAG_EVIDENCE_AUDIT=0 / MRAG_EVIDENCE_DEFENCE=0), a run without curated targets or without literature domains:
# the old behaviour, unchanged.
CARDS_STATUS = "notable_cards_status.json"
CARDS_UNFILTERED = "notable_cards.unfiltered.json"


def card_evidence(branch_dir, strict=False, notes=None):
    """followup 2026-09-28 (fix 2): narrative.card_evidence_context for the run of `branch_dir`
    (<out>/branches/topN): its curated_targets.json rows, <out>/curated_targets/scored.json and the
    score runner's _generif_fetch.json.  None when there are no curated rows; never raises.
    followup r3 (post-deploy finding 2) -- strict (a curated run, evidence layers on, literature domains): no
    readable curated rows or a failed context build raise CardEvidenceUnavailable; an unreadable scored.json /
    _generif_fetch.json is left out (appended to `notes`), the context then comes from the rows alone."""
    if strict:
        from viz_report_mcp.tools.narrative import card_evidence_context
        bd = Path(branch_dir)
        ctp = bd / "curated_targets.json"
        try:
            rows = json.loads(ctp.read_text())
        except Exception as e:
            raise CardEvidenceUnavailable("curated rows unreadable (%s: %s): %s"
                                          % (type(e).__name__, str(e)[:160], ctp))
        if not isinstance(rows, list) or not rows:
            raise CardEvidenceUnavailable("no curated rows in %s" % ctp)
        sdir = bd.parent.parent / "curated_targets"
        parts = []
        for fn in ("scored.json", "_generif_fetch.json"):
            fp = sdir / fn
            v = None
            if fp.exists():
                try:
                    v = json.loads(fp.read_text())
                except Exception as e:
                    msg = "%s unreadable (%s: %s); the card context was built without it" % (
                        fp, type(e).__name__, str(e)[:120])
                    print("[notable] WARNING: " + msg, flush=True)
                    if notes is not None:
                        notes.append(msg)
            parts.append(v)
        try:
            ctx = card_evidence_context(rows, parts[0], parts[1])
        except Exception as e:
            raise CardEvidenceUnavailable("card evidence context failed (%s: %s)" % (type(e).__name__, str(e)[:200]))
        if not ctx:
            raise CardEvidenceUnavailable("card evidence context empty for %d curated row(s)" % len(rows))
        return ctx
    try:
        from viz_report_mcp.tools.narrative import card_evidence_context
        bd = Path(branch_dir)
        ctp = bd / "curated_targets.json"
        rows = json.loads(ctp.read_text()) if ctp.exists() else []
        if not isinstance(rows, list) or not rows:
            return None
        sdir = bd.parent.parent / "curated_targets"
        sd = json.loads((sdir / "scored.json").read_text()) if (sdir / "scored.json").exists() else None
        sc = (json.loads((sdir / "_generif_fetch.json").read_text())
              if (sdir / "_generif_fetch.json").exists() else None)
        return card_evidence_context(rows, sd, sc)
    except Exception as e:
        print("[notable] card evidence context unavailable (%s); cards built without it" % str(e)[:160],
              flush=True)
        return None


def cards_strict(curated_on, novelty_domains) -> bool:
    """followup r3 (post-deploy finding 2): must this run's Section-14 cards be built with the evidence rules?
    A curated run with literature domains (keys, as rebuild_notable_cards gets them; known ones only -- the card
    builder has per-domain Novelty exactly then) and the evidence layers on (narrative._evidence_defence_on, the
    switch the card builder itself reads)."""
    if not (curated_on and novelty_domains):
        return False
    try:
        from ..literature import known_keys
        if not known_keys(list(novelty_domains)):
            return False
    except Exception:
        pass
    try:
        from viz_report_mcp.tools.narrative import _evidence_defence_on
        return bool(_evidence_defence_on())
    except Exception:
        return True


def withhold_unfiltered_cards(branch_dir, reason) -> dict:
    """followup r3 (post-deploy finding 2): the Section-14 cards of a curated run could not be (re)built with the
    evidence rules, so the cards on disk (the narrative-time ones: built before the curated targets existed and
    never with the rules) must not be shown.  Moves notable_cards.json to notable_cards.unfiltered.json, leaves an
    empty notable_cards.json and writes notable_cards_status.json (Section 14 shows it).  Never raises; returns the
    record ("withheld" False only when the cards could not be emptied -- the caller says so loudly)."""
    import time as _time
    bd = Path(branch_dir)
    cp, dst = bd / "notable_cards.json", bd / CARDS_UNFILTERED
    rec = {"version": 1, "withheld": False, "reason": " ".join(str(reason or "").split())[:1200],
           "n_withheld": 0, "moved_to": "", "when": _time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        if cp.exists():
            try:
                _c = json.loads(cp.read_text())
                rec["n_withheld"] = len(_c) if isinstance(_c, list) else -1
            except Exception:
                rec["n_withheld"] = -1
            if rec["n_withheld"] != 0:
                os.replace(cp, dst)
                rec["moved_to"] = str(dst)
        cp.write_text("[]")
        rec["withheld"] = True
    except Exception as e:
        rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        try:
            cp.write_text("[]")
            rec["withheld"] = True
        except Exception:
            pass
    try:
        (bd / CARDS_STATUS).write_text(json.dumps(rec, indent=1))
    except Exception as e:
        rec["status_error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
    return rec


def clear_cards_status(branch_dir) -> None:
    """followup r3: an earlier run's (or rebuild's) withheld-cards status never outlives the cards it describes."""
    try:
        (Path(branch_dir) / CARDS_STATUS).unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        print("[notable] could not remove %s (%s)" % (Path(branch_dir) / CARDS_STATUS, e), flush=True)


def rebuild_notable_cards(branch_dir, *, curated_on: bool,
                          curated_top_genes=None, field: str = "prostate",
                          novelty_domains=None, retriever_script: str = "",
                          endpoint: str = "http://127.0.0.1:11500",
                          model: str = "batiai/qwen3.6-27b:q6",
                          force: bool = False, ortholog_json: str = "",
                          organism: str = "") -> dict:
    """Runner step "notable_findings_rebuild" (factored out 2026-09-17). Rebuilds
    <branch_dir>/notable_cards.json from the branch tfs/targets summaries,
    ordered by `curated_top_genes` when given, with one Novelty entry per
    literature domain (`novelty_domains` = corpus keys in the user's order;
    empty -> legacy single-field Novelty).

    Rebuilds when `force`, in curated mode (the narrative-time cards were drawn
    from the pre-curated targets), or when the current cards are stale
    (notable_cards_stale). Returns {"rebuilt", "reason", "n_cards", "passages",
    "elapsed_s", "novelty_domains"}; raises on a real failure (caller records it)."""
    import time as _time
    t0 = _time.time()
    bd = Path(branch_dir)
    cards_fp, tfs_fp, tgt_fp = bd / "notable_cards.json", bd / "tfs_summary.json", bd / "targets_summary.json"
    cur = []
    if cards_fp.exists():
        try:
            cur = json.loads(cards_fp.read_text())
        except Exception:
            cur = []
    stale = notable_cards_stale(cur, novelty_domains)
    info = {"rebuilt": False, "reason": "", "n_cards": len(cur or []), "passages": [],
            "elapsed_s": 0.0, "novelty_domains": list(novelty_domains or [])}
    if not (force or curated_on or stale) or not (tfs_fp.exists() and tgt_fp.exists()):
        info["reason"] = "cards up to date" if not stale else "tfs/targets summary missing"
        return info
    from viz_report_mcp.tools.narrative import _build_notable_findings
    specs = None
    if novelty_domains:
        from ..literature import domain_specs
        specs = domain_specs(novelty_domains) or None
    order = list(curated_top_genes) if curated_top_genes else None
    # Human-ortholog route: ortholog.json sits in the branch dir beside the
    # targets it describes, so a re-render of a finished run finds it without
    # being told where it is.
    from viz_report_mcp.tools.narrative import load_ortholog as _load_orth
    _oj = ortholog_json or str(bd / "ortholog.json")
    # followup 2026-09-28 (fix 2): in curated mode the cards follow the curated rows' evidence
    # rules (never cite a PMID an evidence filter removed, the gate's filters on every card item,
    # rescue-only genes cite their verified rescue records, the scorer's own GeneRIF lines).
    # narrative ignores it with the evidence layers off (MRAG_EVIDENCE_AUDIT=0 / _DEFENCE=0).
    # followup r3 (post-deploy finding 2): for a curated run with the evidence layers on and literature domains
    # the context is REQUIRED -- no context, no cards (CardEvidenceUnavailable; the runner withholds the old ones)
    _strict = cards_strict(curated_on, novelty_domains)
    _evnotes = []
    _evx = card_evidence(bd, strict=True, notes=_evnotes) if _strict else (card_evidence(bd) if curated_on else None)
    cards, passages = _build_notable_findings(
        json.loads(tfs_fp.read_text()), json.loads(tgt_fp.read_text()), field=field,
        retriever_script=retriever_script or "", endpoint=endpoint, model=model,
        rank_order=order, max_cards=(len(order) if order else 15), domains=specs,
        ortholog=_load_orth(_oj), organism=organism, **({"evidence": _evx} if _evx else {}))
    cards_fp.write_text(json.dumps(cards, indent=2, ensure_ascii=False))
    clear_cards_status(bd)        # followup r3: these cards replace any withheld ones
    if force and not curated_on and not stale:
        reason = "forced rebuild"
    elif curated_on:
        reason = "curated rebuild (align §15 to curated targets)"
    elif stale == "cards empty":
        reason = "post-AG rebuild (narrative-time cards were empty)"
    else:
        reason = f"stale cards rebuilt ({stale})"
    info.update({"rebuilt": True, "reason": reason, "n_cards": len(cards),
                 "passages": passages, "elapsed_s": round(_time.time() - t0, 1)})
    # followup r3 (post-deploy finding 2): whether the evidence rules were applied (the runner withholds a curated
    # run's cards that were not rebuilt with them)
    info["evidence"] = "applied" if (_strict and _evx) else "off"
    if _evnotes:
        info["evidence_notes"] = list(_evnotes)
    print(f"[notable] rebuilt {len(cards)} card(s) in {info['elapsed_s']:.0f} s "
          f"(domains {info['novelty_domains'] or 'legacy'}; {reason})", flush=True)
    return info


def build_narrative_prompts(tfs_full, targets_n, passages, arm_A, arm_B, top_n,
                            study_cancer_type="", study_tissue=""):
    """Build (without any LLM call) the narrative + abstract prompts used by
    NarrativeNode. Split out 2026-09-16 (narrtf) so the prompt text is testable
    offline. The study domain (user-supplied cancer_type / tissue) is an
    authoritative instruction in both prompts and their system messages; with
    neither set, the model is told not to name or guess any cancer type."""
    from viz_report_mcp.tools.narrative_harness import build_strict_prompt
    import viz_report_mcp.tools.narrative as narr
    panel = list(tfs_full[:top_n])
    top_names = [t["name"] if isinstance(t, dict) else t for t in panel]
    label = narr.study_domain_label(study_cancer_type, study_tissue)
    arms = [arm_A, arm_B]
    dom_narr = narr.build_study_domain_instruction(
        study_cancer_type, study_tissue, arms=arms, cited_background_ok=True)
    dom_abs = narr.build_study_domain_instruction(
        study_cancer_type, study_tissue, arms=arms, cited_background_ok=False)
    dom_sys = (f" The study is in {label}; never describe its samples, arms or "
               f"subtypes as any other cancer type or tissue." if label else
               " The study's cancer type / tissue is not specified; never name or "
               "guess one for its samples, arms or subtypes.")

    base_prompt = build_strict_prompt(
        tfs_block=json.dumps(panel, indent=2)[:2000],
        targets_block=json.dumps(targets_n, indent=2)[:4000],
        ag_block="{}",  # AG goes to HTML table only (user preference 2026-05-26)
        passages=passages, conditions=None,
    )
    override = (
        f"\n\n{dom_narr}\n"
        f"\n# OVERRIDE OUTPUT SPEC\n"
        f"- LENGTH: 3-4 short paragraphs, 300-450 words. Results-only.\n"
        f"- COMPARISON ANCHOR (HARD RULE): every quantitative pattern statement\n"
        f"  MUST be phrased as 'in {arm_A} vs {arm_B}' or equivalent.\n"
        f"- TFs in scope (exactly these, no others): {', '.join(top_names)}.\n"
        f"- STUDY DOMAIN: {label or 'not specified'} (see the STUDY DOMAIN rules above).\n"
        f"- DO NOT mention 'AG validation' — AG is shown separately in the HTML.\n"
        f"- No Limitations paragraph.\n"
        f"- No preamble, no closing remarks. Start at Para 1.\n"
    )
    # AG is intentionally OMITTED from the abstract prompt entirely.
    # We never want the abstract to mention AlphaGenome (positive OR
    # null) — it is a supportive in silico signal shown elsewhere in
    # the report, not an abstract-level finding.
    abstract_prompt = (
        f"You are writing the abstract for a result section.\n\n"
        f"CONTEXT\n"
        f"- Study domain: {label or 'not specified'}.\n"
        f"- Study: comparison = {arm_A} vs {arm_B}.\n"
        f"- {len(panel)} driver TFs form the report's top-{len(panel)} TF panel "
        f"(integrated ATAC + RNA ranking; listed in rank order, rank 1 first).\n"
        f"- Top driver TFs (name, signed log2FC; positive => upregulated in {arm_A}):\n"
        f"  {[(t.get('name'), t.get('log2FC')) if isinstance(t, dict) else (t, None) for t in panel]}\n\n"
        f"TASK\n"
        f"Write a SHORT abstract: 3-4 sentences, 100-150 words.\n"
        f"- Frame as '{arm_A} vs {arm_B}'.\n"
        f"- Focus on the driver TFs, the integrated ATAC + RNA evidence,\n"
        f"  and the regulatory program they imply.\n"
        f"- Name only TFs from the panel above (exactly these: {', '.join(top_names)});\n"
        f"  do not add any other TF.\n"
        f"- DO NOT mention AlphaGenome, in silico predictions, AG matches,\n"
        f"  direction matches, counterfactual predictions, or any related\n"
        f"  in silico validation language. The abstract is about the\n"
        f"  experimental TF discovery, not validation tools.\n"
        f"- DO NOT use phrases like 'despite' or 'absence of'\n"
        f"  pointing at null validation results.\n"
        f"- No citations. No hedging. Plain declarative prose.\n\n"
        f"{dom_abs}\n"
    )
    return {
        "top_names":        top_names,
        "domain_label":     label,
        "narrative_system": ("You are a concise, paper-grounded, dataset-specific "
                             "result writer for genomics." + dom_sys),
        "narrative_prompt": base_prompt + override,
        "abstract_system":  "You are a concise scientific abstract writer." + dom_sys,
        "abstract_prompt":  abstract_prompt,
    }


class NarrativeNode(Node):
    """Generate narrative (Qwen3.6-27B q6) + abstract for one TF set.

    Inputs:
      tfs_summary_json     — path to tfs_summary.json (or filtered top-N)
      targets_summary_json — path to targets_summary.json (matching N)
      conditions_meta_json — path to conditions_meta.json

    Params:
      glm_endpoint, glm_model, temperature, max_tokens, top_n
      generate_abstract (default True) — also produce abstract_qwen.md
      novelty_domains — literature-domain keys, user's order (per-domain
                        Novelty on notable_cards.json; empty = legacy)

    Outputs:
      narrative_md, narrative_md_audit, narrative_md_raw, abstract_md
    """
    name = "narrative"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Node.input_hash = inputs (+file mtime/size) + params. Pin the prompt
        # version and the exact TF panel into params so a resumed run can never
        # reuse a narrative built from another panel / an older prompt, even if
        # tfs_summary.json kept its mtime (2026-09-16 narrtf).
        self.params.setdefault("prompt_version", NARRATIVE_PROMPT_VERSION)
        if "tf_panel" not in self.params:
            try:
                _tfs = json.loads(Path(self.inputs.get("tfs_summary_json") or "").read_text())
                _n = int(self.params.get("top_n", 6))
                self.params["tf_panel"] = [t["name"] if isinstance(t, dict) else str(t)
                                           for t in _tfs[:_n]]
            except Exception:
                pass

    def run(self) -> None:
        import viz_report_mcp.tools.narrative as narr
        from viz_report_mcp.tools.narrative_harness import (
            build_strict_prompt, validate_citations,
            renumber_and_build_references, format_references_markdown,
        )
        # capture generate_narrative + reuse internals
        tools = capture_mcp_tools(narr.register_narrative_tools)
        # We don't use the registered tool directly (it doesn't expose abstract).
        # Instead we use the lower-level helpers exposed in narrative.py.
        from viz_report_mcp.tools.narrative import (
            _glm_chat, _query_paperqa, _build_study_context_paragraph,
            _build_notable_findings,
        )

        p = self.params
        i = self.inputs
        out_dir = Path(self.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        top_n = p.get("top_n", 6)
        model    = p.get("glm_model", "batiai/qwen3.6-27b:q6")
        endpoint = p.get("glm_endpoint", "http://127.0.0.1:11500")
        temperature = p.get("temperature", 0.2)
        retriever  = p.get("retriever_script", "/BIO1/MRAG/Agent/libs/paperqa/scripts/retrieve_cli.py")
        venv_py    = p.get("venv_python", "/BIO1/MRAG/venvs/paper_qa/bin/python3")

        tfs_full = json.loads(Path(i["tfs_summary_json"]).read_text())
        top_names = [t["name"] if isinstance(t, dict) else t for t in tfs_full[:top_n]]

        targets_full = json.loads(Path(i["targets_summary_json"]).read_text())
        if isinstance(targets_full, dict):
            targets_n = {k: v for k, v in targets_full.items() if k in top_names}
        else:
            targets_n = [x for x in targets_full
                         if isinstance(x, dict) and
                         (x.get("tf") in top_names or x.get("TF") in top_names)]

        conditions = json.loads(Path(i["conditions_meta_json"]).read_text())
        arm_A = conditions[0]["label"]
        arm_B = conditions[1]["label"]

        # --- Study domain (authoritative; 2026-09-16 narrtf) ---
        if "study_cancer_type" in p or "study_tissue" in p:
            study_ct = p.get("study_cancer_type") or ""
            study_ti = p.get("study_tissue") or ""
        else:
            study_ct, study_ti = (p.get("field") or ""), ""

        # --- Retrieval (paperqa) ---
        study_para, study_passages = _build_study_context_paragraph(
            conditions, retriever, top_k=2, venv_python=venv_py,
        )
        passages = list(study_passages)
        used_pmids = [pp["pmid"] for pp in passages if pp.get("pmid")]
        for tf in top_names:
            hits = _query_paperqa(retriever, f"{tf} transcription factor cancer target gene regulation", top_k=5)
            for h in hits:
                pmid = h.get("pmid", "")
                if pmid and pmid not in used_pmids: used_pmids.append(pmid)
                passages.append({"pmid": pmid, "title": h.get("title", ""),
                                 "snippet": h.get("snippet") or h.get("text", "")})

        # --- Build prompts (study domain + exact TF panel; no LLM call) ---
        prompts = build_narrative_prompts(
            tfs_full, targets_n, passages, arm_A, arm_B, top_n,
            study_cancer_type=study_ct, study_tissue=study_ti)
        narr_prompt = prompts["narrative_prompt"]

        # --- Generate narrative ---
        narr_raw = _glm_chat(
            [{"role":"system","content":prompts["narrative_system"]},
             {"role":"user","content":narr_prompt}],
            endpoint=endpoint, model=model,
            temperature=temperature, max_tokens=p.get("max_tokens", 2000),
            timeout=600.0,
        )
        # flag-only: cited other-cancer background must stay verbatim
        _narr_domain_hits = narr.find_offdomain_cancer_mentions(narr_raw, study_ct, study_ti)
        if _narr_domain_hits:
            print(f"[narrative] WARNING off-domain cancer mentions (study="
                  f"{prompts['domain_label'] or 'unspecified'}): "
                  f"{[h['term'] for h in _narr_domain_hits]}", flush=True)
        _body = (study_para + "\n\n" + narr_raw) if study_para else narr_raw

        # --- Notable Findings (auto hub TF; conservative corpus-bounded novelty
        #     + hub-TF mechanism nuance). Highest hallucination risk -> guarded
        #     in narrative._build_notable_findings (_soften_novelty + cite-only). ---
        field = p.get("field", "prostate")
        # Per-domain Novelty (2026-09-17 llmdomain): params["novelty_domains"] =
        # the user's literature-domain keys in order (variants.py passes
        # shared["literature_domains_user"]); absent/empty -> legacy field Novelty.
        _nov_domains = None
        if p.get("novelty_domains"):
            try:
                from ..literature import domain_specs
                _nov_domains = domain_specs(p.get("novelty_domains")) or None
            except Exception as _e:
                print(f"[narrative] novelty domains ignored: {_e}", flush=True)
        notable_cards, notable_passages = ([], [])
        try:
            from viz_report_mcp.tools.narrative import load_ortholog as _load_orth
            _orth_map = _load_orth(p.get("ortholog_json") or "")
            notable_cards, notable_passages = _build_notable_findings(
                tfs_full, targets_full, field=field, retriever_script=retriever,
                endpoint=endpoint, model=model, domains=_nov_domains, ortholog=_orth_map,
                organism=p.get("organism", ""))
        except Exception as _e:
            print(f"[narrative] notable findings skipped: {_e}", flush=True)
        for pp in notable_passages:
            if pp.get("pmid") and pp["pmid"] not in used_pmids:
                used_pmids.append(pp["pmid"])
            passages.append(pp)

        # Notable Findings render as §15 cards (notable_cards.json), NOT markdown.
        narr_full = "## Analysis Result\n\n" + _body

        # --- Citation harness (covers Analysis Result + Notable Findings) ---
        v = validate_citations(narr_full, passages)
        numbered_text, references = renumber_and_build_references(
            v["text_cleaned"], passages, v["validated"])
        narr_final = numbered_text + format_references_markdown(references)

        # --- Abstract (separate prompt, no citations) ---
        abstract_md = ""
        _abs_domain_hits = []
        if p.get("generate_abstract", True):
            # AG is intentionally OMITTED from the abstract prompt entirely
            # (see build_narrative_prompts).
            abstract_prompt = prompts["abstract_prompt"]
            abstract_md = _glm_chat(
                [{"role":"system","content":prompts["abstract_system"]},
                 {"role":"user","content":abstract_prompt}],
                endpoint=endpoint, model=model,
                temperature=0.2, max_tokens=300, timeout=600.0,
            ).strip()
            # The abstract (= report "Paper-grounded interpretation") is uncited
            # study-level text: drop a wrong organ word ("Basal-like breast
            # cancer" -> "Basal-like cancer") and record every flagged mention.
            abstract_md, _abs_domain_hits = narr.enforce_study_domain(
                abstract_md, study_ct, study_ti, strip=True)
            if _abs_domain_hits:
                print(f"[narrative] WARNING abstract off-domain cancer mentions "
                      f"(study={prompts['domain_label'] or 'unspecified'}): "
                      f"{[(h['term'], bool(h.get('stripped'))) for h in _abs_domain_hits]}",
                      flush=True)

        # --- Save outputs ---
        narr_md_path  = out_dir / f"narrative_top{top_n}.md"
        narr_raw_path = out_dir / f"narrative_top{top_n}.md.raw.md"
        audit_path    = out_dir / f"narrative_top{top_n}.md.audit.json"
        abstract_path = out_dir / f"abstract_top{top_n}.md"

        cards_path = out_dir / "notable_cards.json"
        cards_path.write_text(json.dumps(notable_cards, indent=2, ensure_ascii=False))
        narr_md_path.write_text(narr_final)
        narr_raw_path.write_text(narr_full)
        audit = {
            "n_total": v["n_total"], "n_kept": v["n_kept"],
            "n_removed_whitelist": v["n_removed_whitelist"],
            "n_unverified_quote": v["n_unverified_quote"],
            "n_references": len(references), "references": references,
            "validated": v["validated"], "model": model,
            "comparison_direction": f"{arm_A} vs {arm_B}",
            "study_domain": {
                "label":          prompts["domain_label"],
                "cancer_type":    study_ct,
                "tissue":         study_ti,
                "prompt_version": p.get("prompt_version", NARRATIVE_PROMPT_VERSION),
                "tf_panel":       prompts["top_names"],
                "narrative_offdomain_mentions": [h["term"] for h in _narr_domain_hits],
                "abstract_offdomain_mentions":  [h["term"] for h in _abs_domain_hits],
                "abstract_stripped": [h["term"] for h in _abs_domain_hits if h.get("stripped")],
            },
        }
        audit_path.write_text(json.dumps(audit, indent=2))
        if abstract_md:
            abstract_path.write_text(abstract_md)

        self.outputs = {
            "narrative_md":         str(narr_md_path),
            "narrative_raw_md":     str(narr_raw_path),
            "narrative_audit":      str(audit_path),
            "abstract_md":          str(abstract_path) if abstract_md else "",
            "notable_cards_json":   str(cards_path),
        }


def axis_step_error_from_manifest(manifest_path) -> str:
    """The error of the run's LAST axis_validation manifest record, when that step did not finish
    (status skipped / failed / error with an error or note) and was not an organism gate; "" when
    there is no manifest, no such record, or the step finished.  Never raises."""
    try:
        man = json.loads(Path(manifest_path).read_text())
        steps = man.get("steps") if isinstance(man, dict) else None
        rec = [x for x in (steps or []) if isinstance(x, dict) and x.get("name") == "axis_validation"]
        if not rec:
            return ""
        r = rec[-1]
        st = str(r.get("status") or "").strip().lower()
        if r.get("organism_gated") or st in ("ok", "success", "done", ""):
            return ""
        why = str(r.get("error") or r.get("note") or r.get("reason") or "").strip()
        return ("%s: %s" % (st, why)) if why else ""
    except Exception:
        return ""


class RenderNode(Node):
    """Render the final interactive HTML report.

    Inputs:
      narrative_md, abstract_md, narrative_audit (paths)
      volcano_atac_json, volcano_rna_json, tf_rank_json, trn_json,
      per_tf_profile_json, tf_table_csv (paths, optional)
      axis_json (Section-13 axis validation, optional)
      locus_manifest_json (optional)

    Params:
      report_title, study_label, tf_selection_label,
      literature_domains / literature_domains_user (Section 2 "LLM domain" row;
      absent on old runs -> manifest.json / cancer_type fallback),
      organism / organism_common / genome_build / organism_detection
        (Section 2 organism rows -- the run's species, its assembly and how it
        was decided),
      organism_skipped ([{name, reason}] -- steps this organism cannot run, so
        the report says "not available for <species>" instead of rendering an
        empty section),
      ortholog_banner (the §14 human-ortholog disclosure; "" for human)
      ortholog_overview (the §2 counts line: "N/M genes mapped"; "" for human)

    Outputs:
      mrag_report_html
    """
    name = "render"

    def run(self) -> None:
        import viz_report_mcp.tools.render as render
        tools = capture_mcp_tools(render.register_render_tools)
        render_report = tools["render_report"]

        i = self.inputs; p = self.params
        out_dir = Path(self.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        out_html = out_dir / "mrag_report.html"

        audit = (json.loads(Path(i["narrative_audit"]).read_text())
                 if i.get("narrative_audit") and Path(i["narrative_audit"]).exists()
                 else None)
        abstract_md = (Path(i["abstract_md"]).read_text()
                       if i.get("abstract_md") and Path(i["abstract_md"]).exists()
                       else None)
        # Cross-variant synthesis MD is computed (for manifest provenance)
        # but NOT appended to the narrative — the user found it noise.
        narrative_md_path = i.get("narrative_md")

        locus_tracks = []
        if i.get("locus_manifest_json") and Path(i["locus_manifest_json"]).exists():
            for e in json.loads(Path(i["locus_manifest_json"]).read_text()):
                if e.get("ok", True):
                    locus_tracks.append({"tf": e["tf"], "gene": e["gene"],
                                         "region": e["region"], "png_path": e["png_path"]})

        # Section 8 HOMER known-motif groups
        _hk_groups = [
            {"dir": p.get("homer_known_a_dir",
                          os.path.join(os.environ.get("MRAG_DEMO_HOMER_KNOWN", ""), "OC2_peak_minmax")),
             "label": p.get("homer_known_a_label", "OC2_OE"),
             "color": "#d62728"},
            {"dir": p.get("homer_known_b_dir",
                          os.path.join(os.environ.get("MRAG_DEMO_HOMER_KNOWN", ""), "CyC_peak_minmax")),
             "label": p.get("homer_known_b_label", "cMyc_DDK"),
             "color": "#1f77b4"},
        ]
        # Section 2 "LLM domain" row (2026-09-17): the user's literature domains as
        # "Full name (ABBR), ..." in the user's order, auto-appended pan-cancer hidden.
        # params (new runs) -> <output_dir>/manifest.json step "literature_domains"
        # (finished runs re-rendered with their recorded params) -> cancer_type ->
        # "" (render shows "(not specified)"). Only known display names come back.
        try:
            from ..literature import resolve_user_domains, domains_label
            _llm_domain = domains_label(resolve_user_domains(p, out_dir / "manifest.json"),
                                        include_pancancer=True)
        except Exception:
            _llm_domain = ""

        # Section 13 when the axis_validation step raised or failed before it wrote its result
        # file: the runner records the error in manifest.json only, so it is read from there and
        # the section states it instead of the generic 'left no result file' (review 3, #16).  An
        # organism-gated skip is not an error (the organism_skipped param draws that callout).
        _axis_step_error = str(p.get("axis_step_error", "") or "")
        if not _axis_step_error and not (i.get("axis_json") and Path(i["axis_json"]).exists()):
            _axis_step_error = axis_step_error_from_manifest(out_dir / "manifest.json")

        # The §14 disclosure paragraph. It lives in ortholog.json so a
        # re-render of a finished run says the same thing the run said.
        _ortho_banner = p.get("ortholog_banner", "") or ""
        if not _ortho_banner and i.get("ortholog_json"):
            try:
                from viz_report_mcp.tools.narrative import ortholog_banner as _ob
                _ortho_banner = _ob(i.get("ortholog_json")) or ""
            except Exception:
                _ortho_banner = ""

        res = render_report(
            output_html       = str(out_html),
            report_title      = p.get("report_title", "MRAG Report"),
            study_label       = p.get("study_label", ""),
            volcano_atac_json = i.get("volcano_atac_json"),
            volcano_rna_json  = i.get("volcano_rna_json"),
            tf_rank_json      = i.get("tf_rank_json"),
            trn_json          = i.get("trn_json"),
            per_tf_profile_json = i.get("per_tf_profile_json"),
            tf_table_csv      = i.get("tf_table_csv"),
            shared_dotplot_json = i.get("shared_dotplot_json"),
            per_tf_heatmap_json = i.get("per_tf_heatmap_json"),
            locus_tracks      = locus_tracks,
            narrative_md_path = narrative_md_path,
            executive_summary_md = abstract_md,
            citation_audit    = audit,
            tf_selection      = p.get("tf_selection_label", ""),
            conditions_str    = p.get("conditions_str", ""),
            samples_str       = p.get("samples_str", ""),
            llm_domain        = _llm_domain,
            # 2026-06-04 R-script plot PNGs (Section 7 heatmap; Section 5 PNG fallback)
            rank_png      = i.get("rank_png")    or None,
            volcano_png   = i.get("volcano_png") or None,
            heatmap_png   = i.get("heatmap_png") or None,
            # 2026-06-08 interactive Plotly: Section 5 + Section 7
            rank_json     = i.get("rank_json")    or None,
            volcano_json  = i.get("volcano_json") or None,
            heatmap_json  = i.get("heatmap_json") or None,
            heatmap_median_json = i.get("heatmap_median_json") or None,
            # 2026-06-07 Cytoscape (test.cys-style) — Section 9 (after HOMER known)
            trn_cyjs_json     = i.get("trn_cyjs_json")   or None,
            curated_targets_json = i.get("curated_targets_json") or None,
            # Section 10b: SE -> TF -> Target TRN (2026-09-10)
            se_trn_json       = i.get("se_trn_json") or None,
            cancer_type       = p.get("cancer_type", "") or "",
            # Section 13 (stage 2c-2): axis_validation/axis_<domain>.json
            axis_json         = i.get("axis_json") or None,
            # an optional plain-text notice at the top of Section 13 (set only by re-render
            # helpers outside the pipeline); "" on every pipeline run, so its report is unchanged
            axis_note         = p.get("axis_note", "") or "",
            axis_step_error   = _axis_step_error,
            # Section 8: HOMER known-motif enrichment per condition (precomputed)
            homer_known_groups = _hk_groups,
            # Section 9: GIGGLE Cistrome TF-overlap point plots
            giggle_up_png    = i.get("giggle_up_png")   or None,
            giggle_down_png  = i.get("giggle_down_png") or None,
            peakheatmap_png  = i.get("peakheatmap_png") or None,
            giggle_up_label  = p.get("giggle_up_label", "Up in treatment"),
            giggle_down_label= p.get("giggle_down_label", "Down in treatment"),
            # --- organism / genome provenance + what it turned off ---------
            organism           = p.get("organism", ""),
            organism_common    = p.get("organism_common", ""),
            genome_build       = p.get("genome_build", ""),
            organism_detection = p.get("organism_detection", ""),
            organism_skipped   = p.get("organism_skipped") or [],
            ortholog_banner    = _ortho_banner,
            # Section-2 "Literature namespace" row + the short §10/§11 note.
            # Both are "" on a human run, so its report text is unchanged.
            ortholog_overview  = p.get("ortholog_overview", "") or "",
            # 2026-09-28 pre-deploy review: a FAILED target step says so in red
            curated_error      = p.get("curated_error", "") or "",
        )
        self.outputs = {
            "mrag_report_html": str(out_html),
        }
