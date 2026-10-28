"""Paper-grounded narrative generation: paperqa retrieval + synthesis by the
report LLM (Qwen3.6 27B), with citation harness (whitelist + quote validation
+ numbered refs).

The study-context paragraph (Para 1) is now built deterministically from
conditions_meta to avoid mislabeling of experimental arms by the report LLM.
"""
from __future__ import annotations

import re
import json
import os
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Optional

from .narrative_harness import (
    build_strict_prompt, validate_citations,
    renumber_and_build_references, format_references_markdown,
)


DEFAULT_GLM_ENDPOINT = "http://127.0.0.1:11500"
DEFAULT_GLM_MODEL = "batiai/qwen3.6-27b:q6"  # historical var name; actual model is Qwen3.6-27B q6 (replaced GLM 2026-05-26 due to sample-condition / reference hallucinations)


PROMPT_TEMPLATE = """You are an expert in cancer transcriptional regulation writing a result
section for a paper-style report. Generate a paper-grounded interpretation
of the following TF -> target gene findings.

# STRICT RULES (do not violate)
1. ONLY cite findings from the provided literature passages.
2. Be result-oriented; no speculative hypotheses.
3. Cite specific PMIDs in [PMID:nnnn] format inline.
4. Tight: 4-5 paragraphs, ~600-800 words.
5. Avoid hedging language ("might", "could").

# DATA
## Top TFs
{tfs_block}

## Per-TF top targets
{targets_block}

## Literature passages (retrieved via paperqa)
{lit_block}

# OUTPUT
Start markdown narrative now. No preamble."""


def _drop_ag_block(prompt: str) -> str:
    """narrative_harness.build_strict_prompt always writes an '## AlphaGenome in silico signal'
    block and its 'RULES for AG'; the MCP narrative has no AlphaGenome input any more, so that block
    is cut out of ITS prompt (up to the literature heading).  The pipeline's NarrativeNode builds
    its own prompt from the harness and is not affected."""
    i = prompt.find("## AlphaGenome in silico signal")
    j = prompt.find("## Literature passages", i)
    return prompt[:i] + prompt[j:] if (i >= 0 and j > i) else prompt


def _glm_chat(messages: list, endpoint: str, model: str,
              temperature: float = 0.2, max_tokens: int = 1800,
              timeout: float = 240.0) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 16384,           # bumped from 8K to allow 2000-token output without ctx pressure
            "repeat_penalty": 1.15,
            "repeat_last_n": 256,
        },
    }
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body.get("message", {}).get("content", "")


_PAPERQA_STATE = {"failures": 0, "retry_timeouts": True}
# PaperQA's Qdrant store is local-mode (one process at a time). Retrievals from
# threads of one process (parallel top-N branches) are serialised here instead of
# failing on "already accessed" (2026-09-17 llmdomain; outputs unchanged).
import threading as _threading
_RETRIEVAL_LOCK = _threading.RLock()


def _paperqa_warn(msg: str, failure: bool = True) -> None:
    """stderr (= the run log) note for a failed retrieval; rate-limited."""
    import sys
    if failure:
        _PAPERQA_STATE["failures"] += 1
    n = _PAPERQA_STATE["failures"]
    if not failure or n <= 5 or n % 25 == 0:
        print(f"[narrative] PaperQA retrieval: {msg}" + (f" (failures so far: {n})" if failure else ""),
              file=sys.stderr, flush=True)


def _query_paperqa(retriever_script: str, query: str, top_k: int = 5,
                   venv_python: str = "/BIO1/MRAG/venvs/paper_qa/bin/python",
                   timeout: float = 120.0, cancer=None) -> list:
    """Call paperqa retriever via venv subprocess. Returns hits list (best-effort).

    Failures still return [] but are reported on stderr (they were silent). A
    timeout is retried once, until a retry has itself timed out in this process
    (then a hung retriever does not double every query's wait).

    cancer (2026-09-17): corpus key(s) for the retriever's --cancer filter (a
    str, or a list sent comma-separated); None = all corpora (unchanged)."""
    import subprocess
    q = query[:80]
    attempts = 2 if _PAPERQA_STATE["retry_timeouts"] else 1
    for attempt in range(1, attempts + 1):
        try:
            cmd = [venv_python, retriever_script, "--query", query, "--top_k", str(top_k), "--json"]
            if cancer:
                cmd += ["--cancer", cancer if isinstance(cancer, str) else ",".join(cancer)]
            env = os.environ.copy()
            env["PYTHONPATH"] = "/BIO1/MRAG/Agent/libs/paperqa"
            with _RETRIEVAL_LOCK:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            if attempt < attempts:
                _paperqa_warn(f"timeout after {timeout:.0f} s for {q!r}; retrying once", failure=False)
                continue
            if attempt > 1:
                _PAPERQA_STATE["retry_timeouts"] = False
            _paperqa_warn(f"timeout after {timeout:.0f} s ({attempt} attempt(s)); no hits for {q!r}")
            return []
        except Exception as e:
            _paperqa_warn(f"{type(e).__name__}: {e}; no hits for {q!r}")
            return []
        if r.returncode != 0:
            tail = " | ".join((r.stderr or "").strip().splitlines()[-2:])[-300:]
            _paperqa_warn(f"retriever exit code {r.returncode} ({tail}); no hits for {q!r}")
            return []
        try:
            out = json.loads(r.stdout)
            return out.get("hits", []) if isinstance(out, dict) else out
        except Exception as e:
            _paperqa_warn(f"unreadable retriever output ({type(e).__name__}: {e}); no hits for {q!r}")
            return []
    return []


def _build_study_context_paragraph(conditions: list, retriever_script: str,
                                    top_k: int = 2,
                                    venv_python: str = "/BIO1/MRAG/venvs/paper_qa/bin/python") -> tuple:
    """Build Para 1 (study context) DETERMINISTICALLY.

    Returns (markdown_paragraph, passages_used).
    The LLM is bypassed entirely for the arm-identity section to avoid hallucination.
    """
    if not conditions:
        return "", []
    sentences = []
    passages = []
    for c in conditions:
        label = c.get("label", "?")
        concept = c.get("concept") or c.get("perturbation") or label
        pert = c.get("perturbation", "")
        hits = _query_paperqa(retriever_script, f"{concept} cancer mechanism",
                              top_k=top_k, venv_python=venv_python) if retriever_script else []
        cite = ""
        if hits:
            best = hits[0]
            pmid = best.get("pmid", "")
            if pmid:
                cite = f" [PMID:{pmid}]"
                passages.append({"pmid": pmid,
                                  "title": best.get("title", ""),
                                  "snippet": best.get("snippet", "") or best.get("text", "")})
        sentences.append(f"**{label}** \u2014 {pert}. Biological role: {concept}.{cite}")
    # Dedupe rationales (preserve order); arms often share the same default
    # placeholder, so joining raw would repeat it (e.g. "Comparison arm
    # Comparison arm"). Also suppress the line when the only content is the
    # default placeholder, which carries no real information.
    rationales = list(dict.fromkeys(
        c.get("rationale") for c in conditions if c.get("rationale")))
    rationale_line = ""
    if rationales and rationales != ["Comparison arm"]:
        rationale_line = "\n\n*Comparison rationale:* " + " ".join(rationales)
    para = ("This study compares the following experimental arms:\n\n" +
            "\n".join(f"- {s}" for s in sentences) + rationale_line)
    return para, passages


_NOVELTY_FORBIDDEN = re.compile(
    r"\b(first(?:\s+time|\s+report(?:ed)?)?|for the first time|never (?:been )?"
    r"(?:reported|shown|described)|previously unreported|novel discovery|"
    r"unprecedented|we (?:are the first|first report))\b", re.I)


def _soften_novelty(text: str, field: str) -> str:
    """Hard guard against absolute novelty claims (highest hallucination risk).
    Any forbidden absolute-novelty phrase is replaced with the single allowed,
    corpus-bounded, conservative phrasing."""
    repl = f"not reported in our indexed {field} literature (to our knowledge)"
    return _NOVELTY_FORBIDDEN.sub(repl, text)


_FIELD_SYNONYMS = {
    "kidney": ["kidney", "renal", "nephro", "rcc", "kirp", "kirc", "papillary renal"],
    "prostate": ["prostate", "prostatic", "prad"],
    "lung": ["lung", "pulmonary", "luad", "lusc", "nsclc", "sclc"],
    "breast": ["breast", "mammary", "brca", "tnbc"],
    "liver": ["liver", "hepatic", "hepatocellular", "hcc", "lihc"],
    "colon": ["colon", "colorectal", "coad", "crc", "rectal"],
    "pancreas": ["pancrea", "paad", "pdac"],
    "ovary": ["ovary", "ovarian"],
    "stomach": ["stomach", "gastric", "stad"],
    "bladder": ["bladder", "urothelial", "blca"],
    "brain": ["brain", "glioma", "glioblastoma", "gbm"],
    "skin": ["skin", "melanoma", "cutaneous", "skcm"],
    "thyroid": ["thyroid", "thca"],
    "cervix": ["cervix", "cervical", "cesc"],
    "esophagus": ["esophag", "oesophag", "esca"],
    "head and neck": ["head and neck", "hnsc", "oral", "laryng", "pharyng"],
}


def _field_terms(field):
    """Lowercase field synonym set. A passage counts as sample-domain evidence
    only if it literally mentions one of these terms (guards against paperQA
    nearest-neighbour false positives that surface a different cancer)."""
    f = (field or "").strip().lower()
    terms = set()
    if f:
        terms.add(f)
        terms.add(f.split()[0])
        for key, syns in _FIELD_SYNONYMS.items():
            if key in f or f in key or f in syns:
                terms.update(syns)
    terms.discard("")
    return terms


# ---------------------------------------------------------------------------
# Study-domain guard for LLM summary text (2026-09-16, narrtf)
# The narrative / executive-summary prompts used to receive NO cancer-type
# context, so the model inferred one from the arm labels (PRAD 'Basal_like' ->
# "Basal-like breast cancer subtypes"). These helpers turn the run's
# user-supplied cancer_type / tissue into an authoritative instruction and
# provide a cheap post-generation check for other-cancer mentions. Generic:
# nothing here assumes a particular cancer type.
# ---------------------------------------------------------------------------
_ONTOLOGY_ID = re.compile(r"^[A-Za-z]+:[0-9]+$")

# organ key -> adjective stems that directly precede a cancer noun
_ORGAN_ADJ_STEMS = {
    "kidney": ["kidney", "renal", "nephro"],
    "prostate": ["prostat"],
    "lung": ["lung", "pulmonary"],
    "breast": ["breast", "mammary"],
    "liver": ["liver", "hepatic", "hepatocellular"],
    "colon": ["colon", "colorectal", "rectal"],
    "pancreas": ["pancrea"],
    "ovary": ["ovar"],
    "stomach": ["stomach", "gastric"],
    "bladder": ["bladder", "urothelial"],
    "brain": ["brain"],
    "skin": ["skin", "cutaneous"],
    "thyroid": ["thyroid"],
    "cervix": ["cervix", "cervical"],
    "esophagus": ["esophag", "oesophag"],
    "head and neck": ["head and neck", "oral", "laryng", "pharyng", "nasopharyng"],
    "endometrium": ["endometri", "uterine"],
    "testis": ["testicular"],
    "adrenal": ["adrenal", "adrenocortical"],
    "gallbladder": ["gallbladder"],
    "bile duct": ["cholangio", "biliary"],
}
# organ key -> stand-alone disease names / abbreviations (flag only, never stripped)
_ORGAN_STANDALONE = {
    "brain": ["glioma", "glioblastoma", "gbm"],
    "skin": ["melanoma", "skcm"],
    "lung": ["nsclc", "sclc", "luad", "lusc"],
    "breast": ["tnbc"],
    "liver": ["hcc", "lihc"],
    "pancreas": ["pdac", "paad"],
    "kidney": ["rcc", "ccrcc", "kirc", "kirp"],
    "prostate": ["prad", "crpc", "mcrpc", "nepc"],
    "colon": ["crc", "coad"],
    "blood": ["leukemi", "leukaemi", "lymphoma", "myeloma"],
}
_CANCER_NOUN = (r"(?:cancers?|carcinomas?|adenocarcinomas?|tumou?rs?|neoplasms?|"
                r"malignanc(?:y|ies))")
_ORGAN_ALT = "|".join(sorted({re.escape(s) for v in _ORGAN_ADJ_STEMS.values() for s in v},
                             key=len, reverse=True))
_ORGAN_MENTION = re.compile(
    r"\b(?P<organs>(?:" + _ORGAN_ALT + r")\w*(?:\s*(?:,|/|\band\b|\bor\b)\s*(?:"
    + _ORGAN_ALT + r")\w*)*)"
    r"(?P<gap>(?:[\s-]+(?:small[\s-]+cell|non[\s-]+small[\s-]+cell|clear[\s-]+cell|"
    r"squamous(?:[\s-]+cell)?|cell|ductal|lobular|papillary|invasive|serous|"
    r"epithelial|neuroendocrine))*)"
    r"\s+(?P<noun>" + _CANCER_NOUN + r")\b", re.I)
_STANDALONE_MENTION = re.compile(
    r"\b(?P<term>" + "|".join(sorted({re.escape(s) for v in _ORGAN_STANDALONE.values() for s in v},
                                     key=len, reverse=True)) + r")\w*\b", re.I)


def _organ_key_of(word: str):
    w = re.sub(r"\s+", "", (word or "").strip().lower())
    for key, stems in _ORGAN_ADJ_STEMS.items():
        if any(w.startswith(s.replace(" ", "")) for s in stems):
            return key
    for key, terms in _ORGAN_STANDALONE.items():
        if any(w.startswith(t) for t in terms):
            return key
    return None


def study_domain_label(cancer_type=None, tissue=None) -> str:
    """Human label for the study domain from the run's user-supplied
    cancer_type / tissue ('' when neither is set). Ontology ids are ignored."""
    ct = str(cancer_type or "").strip()
    ti = str(tissue or "").strip()
    if ti and _ONTOLOGY_ID.match(ti):
        ti = ""
    if ct and ti and ti.lower() not in ct.lower():
        return f"{ct} ({ti})"
    return ct or ti


def _domain_organ_keys(cancer_type=None, tissue=None) -> set:
    text = study_domain_label(cancer_type, tissue).lower()
    keys = set()
    if not text:
        return keys
    for key, stems in _ORGAN_ADJ_STEMS.items():
        if key in text or any(re.search(r"\b" + re.escape(s), text) for s in stems):
            keys.add(key)
    for key, terms in _ORGAN_STANDALONE.items():
        if any(re.search(r"\b" + re.escape(t), text) for t in terms):
            keys.add(key)
    return keys


def build_study_domain_instruction(cancer_type=None, tissue=None, arms=None,
                                   cited_background_ok: bool = False) -> str:
    """Authoritative STUDY DOMAIN block for an LLM prompt.

    cited_background_ok=True (paper-grounded narrative): other cancers may be
    named only as explicitly attributed, cited literature background.
    cited_background_ok=False (abstract / executive summary): no other cancer
    type or tissue may be named at all."""
    label = study_domain_label(cancer_type, tissue)
    arm_txt = " vs ".join(str(a) for a in (arms or []) if a)
    arms_clause = f" ({arm_txt})" if arm_txt else ""
    lines = ["# STUDY DOMAIN (AUTHORITATIVE - overrides anything inferred from arm "
             "labels, TF names or literature)"]
    if label:
        lines.append(f"- The study is in {label}. All samples and experimental arms"
                     f"{arms_clause} belong to this {label} study.")
        lines.append(f"- Do not name any other cancer type or tissue for this study, its "
                     f"samples, arms or subtypes - even when an arm label (e.g. a subtype "
                     f"name) is also used in another cancer. Describe them only as {label}.")
    else:
        lines.append("- The cancer type / tissue of this study was NOT specified.")
        lines.append("- Do not name or guess any cancer type or tissue for this study, its "
                     "samples, arms or subtypes (arm labels such as subtype names do not "
                     "identify the cancer). Refer to them only by their arm labels.")
    if cited_background_ok:
        lines.append(("- Another cancer type" if label else "- A named cancer type")
                     + " may appear ONLY as explicitly attributed, cited "
                     "literature background (e.g. 'in <other cancer>, X was reported "
                     "[PMID:n]'), never as a description of this study's samples.")
    else:
        lines.append((f"- Do not name any cancer type or tissue other than {label} "
                      "anywhere in the text.") if label else
                     "- Do not name any cancer type or tissue anywhere in the text.")
    return "\n".join(lines)


def find_offdomain_cancer_mentions(text, cancer_type=None, tissue=None) -> list:
    """Cheap post-generation check: mentions of a cancer type / organ that is not
    the study domain. Returns [{term, organs, start, end, strippable,
    replacement}]. `strippable` is True only for '<organ adj> <cancer noun>'
    phrases (no qualifier such as 'cell' in between) whose organs are ALL
    off-domain and the domain itself resolved to a known organ (or no domain was
    given at all); stand-alone names (melanoma, NSCLC, ...) are flag-only."""
    if not text:
        return []
    label = study_domain_label(cancer_type, tissue)
    allowed = _domain_organ_keys(cancer_type, tissue)
    resolved = bool(allowed) or not label
    hits = []
    for m in _ORGAN_MENTION.finditer(text):
        _org = re.sub(r"head\s+and\s+neck", "headandneck", m.group("organs"), flags=re.I)
        words = [w for w in re.split(r"\s*(?:,|/|\band\b|\bor\b)\s*", _org, flags=re.I) if w.strip()]
        keys = {k for k in (_organ_key_of(w) for w in words) if k}
        off = keys - allowed
        if not off:
            continue
        # strip only the plain '<organ(s)> <noun>' form (no 'cell'/'squamous' ...
        # qualifier in between) so the remaining phrase still reads naturally
        strippable = resolved and not (keys & allowed) and not m.group("gap").strip()
        rest = m.group(0)[len(m.group("organs")):].lstrip(" \t-")
        if rest and m.group("organs")[:1].isupper():
            rest = rest[0].upper() + rest[1:]
        hits.append({"term": m.group(0), "organs": sorted(off), "start": m.start(),
                     "end": m.end(), "strippable": bool(strippable), "replacement": rest})
    spans = [(h["start"], h["end"]) for h in hits]
    for m in _STANDALONE_MENTION.finditer(text):
        if any(a <= m.start() < b for a, b in spans):
            continue
        k = _organ_key_of(m.group("term"))
        if k and k not in allowed:
            hits.append({"term": m.group(0), "organs": [k], "start": m.start(),
                         "end": m.end(), "strippable": False, "replacement": None})
    return sorted(hits, key=lambda h: h["start"])


def enforce_study_domain(text, cancer_type=None, tissue=None, strip: bool = False):
    """Flag (and with strip=True, remove the wrong organ word from) off-domain
    cancer mentions. Returns (text, hits). Use strip=True only for uncited
    study-level summaries (abstract / executive summary); cited literature
    background in the narrative must stay verbatim (flag only)."""
    hits = find_offdomain_cancer_mentions(text, cancer_type, tissue)
    if not strip or not hits:
        return text, hits
    out = text
    for h in sorted((h for h in hits if h["strippable"]), key=lambda h: -h["start"]):
        out = out[:h["start"]] + h["replacement"] + out[h["end"]:]
        h["stripped"] = True
    return out, hits


def build_samples_description(labels, cancer_type=None, tissue=None,
                             endpoint=DEFAULT_GLM_ENDPOINT,
                             model=DEFAULT_GLM_MODEL):
    """LLM one-line factual description of the samples for the report's §2
    'Samples' field. Expands the study/TCGA abbreviation in the arm labels
    (e.g. KIRP -> Kidney Renal Papillary Cell Carcinoma) and names each arm.
    Falls back to a plain 'A vs B' join on any error (never blocks rendering)."""
    labs = [str(l).strip() for l in (labels or []) if str(l).strip()]
    fallback = " vs ".join(labs) if labs else (cancer_type or tissue or "")
    if not labs:
        return fallback
    ctx = []
    if cancer_type:
        ctx.append(f"cancer_type={cancer_type}")
    # Opaque ontology ids (e.g. EFO:0002074) mislead the model into
    # inventing an unrelated disease -> drop them from the context.
    _t = str(tissue).strip() if tissue else ""
    if _t and not re.match(r"^[A-Za-z]+:[0-9]+$", _t):
        ctx.append(f"tissue={_t}")
    sys_msg = {"role": "system", "content":
        "You write ONE short factual line describing the biological samples in a "
        "genomics comparison, for a report's 'Samples' field. Expand any standard "
        "study/TCGA abbreviation found in the arm labels (e.g. KIRP = Kidney Renal "
        "Papillary Cell Carcinoma; KIRC = Kidney Renal Clear Cell Carcinoma; "
        "PRAD = Prostate Adenocarcinoma; LUAD = Lung Adenocarcinoma; LUSC = Lung "
        "Squamous Cell Carcinoma; BRCA = Breast Invasive Carcinoma) and briefly "
        "name what each arm is. Format exactly: "
        "'<STUDY> (<expanded tissue/disease>): <arm A> vs <arm B>'. "
        "<= 32 words, plain text, NO markdown, NO invented quantitative claims. "
        "If an abbreviation is unknown, keep it verbatim and do not guess. "
        "If NEITHER the labels NOR the context identify a known cancer or "
        "tissue, DO NOT invent one: output just the arms joined by ' vs ' "
        "(no parenthetical disease name)."}
    _lines = [f"Arm labels: {labs}"]
    if ctx:
        _lines.append("Context: " + "; ".join(ctx))
    _lines.append("Return ONLY the one-line description.")
    user_msg = {"role": "user", "content": "\n".join(_lines)}
    try:
        out = _glm_chat([sys_msg, user_msg], endpoint, model,
                        temperature=0.1, max_tokens=120, timeout=90.0).strip()
        out = out.splitlines()[0].strip().strip('"').strip() if out else ""
        return out or fallback
    except Exception:
        return fallback


def _pmid_links(text: str) -> str:
    return re.sub(
        r"\[PMID:([0-9]+)\]",
        r'<a href="https://pubmed.ncbi.nlm.nih.gov/\1/" target="_blank">[PMID:\1]</a>',
        text or "")


# followup 2026-09-28 (fix 1): the OUTCOME of the last GeneRIF fetch per (gene,
# retmax) -- {"status": ok | empty | no_gene | failed, "error"} -- so the
# Section-10 table and the Section-14 cards can tell "NCBI lists no GeneRIF"
# from "the record could not be read" (both used to be the same []).  Kept
# beside the function instead of in its return value because tests replace
# _gene_generifs_safe with their own; a replaced function records nothing and
# is taken at its word.
_GENERIF_STATUS: dict = {}
_GENERIF_STATUSES = ("ok", "empty", "no_gene", "failed")


def _gene_generifs_safe(gene, retmax=8):
    """Deterministic NCBI GeneRIF fetch via the GeneGPT module (esearch->efetch,
    Gene-commentary type=generif). Returns [] on any failure so narrative never
    breaks. Each item: {text, pmids:[...]}.
    followup 2026-09-28: the outcome is recorded in _GENERIF_STATUS[(gene, retmax)]
    (genegpt.gene_generifs_status; an older genegpt: ok / empty, or failed on an
    exception)."""
    key = (str(gene), int(retmax))
    try:
        import sys as _sys
        if "/BIO1/MRAG/Agent/libs/genegpt" not in _sys.path:
            _sys.path.insert(0, "/BIO1/MRAG/Agent/libs/genegpt")
        import genegpt as _gg
        _fn = getattr(_gg, "gene_generifs_status", None)
        if _fn is not None:
            st = _fn(gene, retmax=retmax) or {}
            stt = st.get("status") if st.get("status") in _GENERIF_STATUSES else "failed"
            _GENERIF_STATUS[key] = {"status": stt, "error": str(st.get("error") or "")[:300]}
            return list(st.get("lines") or []) if stt != "failed" else []
        lines = _gg.gene_generifs(gene, retmax=retmax) or []
        _GENERIF_STATUS[key] = {"status": "ok" if lines else "empty", "error": ""}
        return lines
    except Exception as e:
        _GENERIF_STATUS[key] = {"status": "failed", "error": _redact_ncbi("%s: %s" % (type(e).__name__, e))[:300]}
        return []


# --------------------------------------------------------------- NCBI access
# 2026-09-20: every E-utilities call in this module goes through ONE helper so
# the api_key / tool / email (shared loader libs/ncbi_creds.py, env file mode
# 600) and the key-aware throttle apply everywhere -- GeneRIF, the per-domain
# title gate and the ADD-1 live PubMed check alike. The key must never reach a
# report, a status JSON or a log, so every error string is redacted.
def _ncbi_creds_mod():
    try:
        import sys as _sys
        if "/BIO1/MRAG/Agent/libs" not in _sys.path:
            _sys.path.insert(0, "/BIO1/MRAG/Agent/libs")
        import ncbi_creds as _c
        return _c
    except Exception:
        return None


def _redact_ncbi(text):
    """Scrub the api_key out of anything that may be stored or printed."""
    c = _ncbi_creds_mod()
    try:
        return c.redact(text) if c is not None else str(text)
    except Exception:
        return str(text)


def _call_api_compat(gg, url, timeout=30, expect=None):
    """followup 2026-09-28 (fix 1): genegpt.call_api WITH the payload check
    (expect="xml" / "json": an incomplete or NCBI-error answer is retried like a
    network error, then raised) -- or without it for a call_api that does not
    take it (an older genegpt, a test stub)."""
    if expect:
        try:
            return gg.call_api(url, timeout=int(timeout), expect=expect)
        except TypeError as e:
            if "expect" not in str(e):
                raise
    return gg.call_api(url, timeout=int(timeout))


def _ncbi_get(url, timeout=30, expect=None):
    """One GET against NCBI: genegpt.call_api when available (shared throttle,
    3 retries, credentials, redacted errors), otherwise a plain urlopen that
    still carries the credentials. Raises on failure -- callers decide.
    followup 2026-09-28: `expect` ("xml" / "json") is passed on (_call_api_compat)."""
    _gg = None
    try:
        import sys as _sys
        if "/BIO1/MRAG/Agent/libs/genegpt" not in _sys.path:
            _sys.path.insert(0, "/BIO1/MRAG/Agent/libs/genegpt")
        import genegpt as _gg_mod
        _gg = _gg_mod
    except Exception:
        _gg = None
    if _gg is not None:
        return _call_api_compat(_gg, url, timeout, expect)
    c = _ncbi_creds_mod()
    sent = c.add_params(url) if c is not None else url
    if c is not None:
        c.SHARED.wait()
    try:
        with urllib.request.urlopen(sent, timeout=timeout) as _r:
            return _r.read()
    except Exception as e:
        if c is None:
            raise
        raise c.NcbiError(c.redact("%s: %s" % (type(e).__name__, e)),
                          getattr(e, "code", None)) from None


# followup 2026-09-28 (fix 1): the chunks of the last _pmid_titles() calls that
# FAILED (the caller resets and reads it; a replaced _pmid_titles records none)
_PMID_TITLE_ERRORS: list = []


def _pmid_titles(pmids, max_pmids=400, chunk=200):
    """Batch NCBI esummary -> {pmid: title}. Empty on any failure.

    ADD-2 (2026-09-18): chunked. It used to silently truncate at 50 PMIDs, which
    was invisible while cards used only 8 GeneRIFs but would drop most of a
    well-studied gene's 400 RIFs from the per-domain title gate. Chunks that
    fail are skipped, not fatal.
    followup 2026-09-28: ... and recorded in _PMID_TITLE_ERRORS, so a caller can
    tell a title it could not read from a paper that has none."""
    pmids = [str(p) for p in dict.fromkeys(pmids) if p][:max_pmids]
    if not pmids:
        return {}
    out = {}
    for i in range(0, len(pmids), chunk):
        part = pmids[i:i + chunk]
        try:
            url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                   "?db=pubmed&retmode=json&id=" + ",".join(part))
            d = json.loads(_ncbi_get(url, timeout=30, expect="json").decode("utf-8", "ignore"))
            res = (d or {}).get("result", {}) or {}
            for p in part:
                if p in res:
                    out[p] = (res.get(p, {}) or {}).get("title", "")
        except Exception as e:
            _PMID_TITLE_ERRORS.append("%d PubMed title(s): %s" % (len(part), _redact_ncbi(
                "%s: %s" % (type(e).__name__, e))[:200]))
            continue
    return out


def _gene_field_generif_pmids(gene, field, max_pmids=4):
    """Gene-grounded, cancer-precise GeneRIF PMIDs: the gene's GeneRIF entries
    whose RIF text OR the cited paper's TITLE mentions the cancer `field`.
    GeneRIFs are gene-specific by construction, so this cannot surface a paper
    about a different gene (unlike vectordb similarity). Title matching catches
    field-relevant papers whose one-line RIF text omits the field word (e.g.
    a prostate methylation paper whose GeneRIF says only
    'validated in an independent cohort'). Returns (field_pmids, general_pmids),
    both capped, in GeneRIF order."""
    fterms = _field_terms(field)
    general, pairs = [], []
    _nm_c = _gene_fullname_offline(gene)
    for r in _gene_generifs_safe(gene, retmax=20):
        txt = r.get("text", "") or ""
        if _tied_to_other_gene(txt, gene, _nm_c, generif=True):
            continue            # E1b: a line about the gene that uses this symbol as alias
        for pm in (r.get("pmids") or []):
            if pm and pm not in general:
                general.append(pm); pairs.append((pm, txt))
    titles = _pmid_titles([pm for pm, _ in pairs])
    field_pmids = []
    for pm, txt in pairs:
        blob = (txt + " " + (titles.get(pm, "") or "")).lower()
        if any(ft in blob for ft in fterms) and pm not in field_pmids:
            field_pmids.append(pm)
    return field_pmids[:max_pmids], general[:max_pmids]


def build_target_table_rows(curated_targets, field, cache_path=None,
                            max_pmids=4, trn_regulators=None):
    """Section-10 target-gene table rows (rendered under the TRN graph):
       Gene | score | Regulated by TFs | Evidence source | GeneRIF (field only).
    Key driver TF = curated row's drivers; GeneRIF = NCBI GeneRIF PMIDs whose text
    literally mentions the sample cancer field. The GeneRIF fetch hits the network
    only on first build; cached to cache_path so re-renders are offline.

    The table has no per-gene in-silico column: the in-silico evidence of a run is the
    pair-level axis test of Section 13, which is measured on the network's edges and
    not on one gene's promoter, so it cannot be read off a per-gene cell."""
    import json as _json
    from pathlib import Path as _Path
    cache = {}
    if cache_path and _Path(cache_path).exists():
        try:
            cache = _json.loads(_Path(cache_path).read_text()) or {}
        except Exception:
            cache = {}
    fterms = _field_terms(field)

    _gf_failed = {}

    def _field_pmids(gene):
        if gene in cache:
            return cache[gene]
        _GENERIF_STATUS.pop((str(gene), 20), None)
        del _PMID_TITLE_ERRORS[:]
        fp, gp = _gene_field_generif_pmids(gene, field, max_pmids=max_pmids)
        seen = fp or gp   # cancer-specific first; else the gene's top GeneRIFs
        # followup 2026-09-28 (fix 1): a GeneRIF record or title lookup that FAILED is
        # not cached (the next render retries) and the row says so
        _st = _GENERIF_STATUS.get((str(gene), 20)) or {}
        if _st.get("status") == "failed" or _PMID_TITLE_ERRORS:
            _gf_failed[gene] = (("GeneRIF record: " + str(_st.get("error") or "fetch failed"))
                                if _st.get("status") == "failed" else "; ".join(_PMID_TITLE_ERRORS))[:300]
            return seen
        cache[gene] = seen
        return seen

    trn_regulators = trn_regulators or {}
    rows = []
    for _i, r in enumerate(curated_targets if isinstance(curated_targets, list) else []):
        gene = r.get("gene", "")
        drivers = list(r.get("drivers") or [])
        regby = sorted(trn_regulators.get(gene, []))
        # 2026-09-23: where this gene's domain evidence came from, so the
        # table can never present a live-PubMed rescue as if it had been read
        # out of the indexed corpus.  "" for a run whose scorer predates the
        # evidence gate -- the column then just reads "-".
        _evsrc = str(r.get("evidence_source") or "")
        _pm = r.get("pubmed_evidence") or {}
        # FIX 2026-09-23 (review): three things this cell must not blur.
        #  * a GeneRIF-only grounding is NOT the indexed corpus (under the
        #    scope="all" default a row can pass on a GeneRIF line alone);
        #  * a live-PubMed rescue verified in SOME domains of a multi-domain
        #    run, so the PMIDs are labelled per domain from
        #    pubmed_evidence.verified_by_domain -- the pooled list is only the
        #    fallback for a scored.json written before this fix;
        #  * a live check that FAILED is not an absence and says so.
        _evlabel = {"corpus": "indexed corpus", "generif": "GeneRIF only",
                    "pubmed": "live PubMed (verified)",
                    "pubmed-error": "live PubMed check failed"}.get(_evsrc, "")
        _evdom = ([{"label": str(_d.get("label") or _d.get("key") or ""),
                    "pmids": [str(x) for x in (_d.get("pmids") or [])]}
                   for _d in (_pm.get("verified_by_domain") or [])
                   if _d.get("pmids")] if _evsrc == "pubmed" else [])
        rows.append({
            "rank": _i + 1,
            "gene": gene,
            "score": r.get("score", ""),
            "regulated_by": ", ".join(regby),
            "drivers": ", ".join(drivers),
            "generif_pmids": _field_pmids(gene),
            "evidence_source": _evsrc,
            "evidence_scope": str(r.get("evidence_gate_scope") or ""),
            "evidence_label": _evlabel,
            "evidence_pmids": ([str(x) for x in (_pm.get("pmids") or [])]
                               if _evsrc == "pubmed" else []),
            "evidence_by_domain": _evdom,
        })
        if gene in _gf_failed:
            # followup 2026-09-28 (fix 1): the GeneRIF column could not be built from NCBI
            rows[-1].update({"generif_fetch": "failed", "generif_fetch_error": _gf_failed[gene]})
        if r.get("audit_verdict"):
            # evaudit 2026-09-27: the evidence audit's verdict for this target.
            # fix2 2026-09-28: the report SHOWS only the flags a reader must see
            # (audit_flag: weak / invalid kept / audit-suspect / unaudited);
            # direct / list stay in the JSON
            rows[-1].update({"audit_verdict": str(r.get("audit_verdict") or ""),
                             "audit_reason": str(r.get("audit_reason") or ""),
                             "audit_pmid": str(r.get("audit_pmid") or "")})
            if r.get("audit_flag"):
                rows[-1].update({"audit_flag": str(r.get("audit_flag") or ""),
                                 "audit_note": str(r.get("audit_note") or "")})
        if r.get("evidence_flag"):
            # fix3 2026-09-28: a flag of the evidence layers a reader must see
            # ("score not re-checked": the evidence filters removed the item
            # the score was given on and the re-score failed; "NEC paper check
            # not run": the paper-level NET test could not read the record)
            rows[-1].update({"evidence_flag": str(r.get("evidence_flag") or ""),
                             "evidence_flag_note": str(r.get("evidence_flag_note") or "")})
    if cache_path:
        try:
            _Path(cache_path).write_text(_json.dumps(cache, indent=1))
        except Exception:
            pass
    return rows


# ---------------------------------------------------------------------------
# Per-domain Novelty for the Section-14 target-gene cards (2026-09-17, llmdomain)
#
# The run's literature domains (user's order, auto-appended pan-cancer hidden)
# arrive as mrag_orchestrator.literature.domain_specs() dicts -- plain data,
# because viz_report_mcp never imports the orchestrator. Each card gets one
# Novelty entry per domain, judged ONLY from that domain's corpus-filtered
# passages (+ GeneRIFs whose text/title names the domain). All retrieval for a
# card batch runs in ONE paper_qa process (one BGE-M3 load, Qdrant-lock wait)
# instead of one retrieve_cli subprocess (~30 s model load) per query.
# ---------------------------------------------------------------------------
NOVELTY_DOMAINS_VERSION = "novdom_20260920b"
# ADD-2 (2026-09-18): GeneRIFs are fetched with ONE efetch of the gene record no
# matter how many are parsed out, so a large cap costs nothing extra on the wire.
# The newest 8 still drive the generic card evidence (unchanged); the full set is
# scanned for the per-domain gate, where a well-studied gene's older, domain-
# specific RIFs used to be thrown away (ERG has 400+).
CARD_GENERIF_RETMAX = 400
CARD_GENERIF_RECENT = 8
_CARD_VENV_PYTHON = "/BIO1/MRAG/venvs/paper_qa/bin/python"
_PAPERQA_LIB = "/BIO1/MRAG/Agent/libs/paperqa"
_CARD_RETRIEVAL_STATS: dict = {}
from html import escape as _html_escape
_UNREPORTED_RE = re.compile(r"\bnot\s+(?:been\s+)?reported\b|\bto our knowledge\b", re.I)
_PMID_GROUP_RE = re.compile(r"\[\s*PMID:\s*\d+(?:\s*[,;]\s*(?:PMID:\s*)?\d+)+\s*\]", re.I)


def _split_pmid_groups(text):
    """'[PMID:1, PMID:2]' / '[PMID:1; 2]' -> '[PMID:1] [PMID:2]', so every PMID
    passes the per-PMID whitelist and becomes a PubMed link (a grouped bracket
    slipped past both)."""
    return _PMID_GROUP_RE.sub(
        lambda m: " ".join(f"[PMID:{n}]" for n in re.findall(r"\d+", m.group(0))), text or "")


_PUBMED_CACHE: dict = {}
_PUBMED_VERIFY_CACHE: dict = {}
# FIX 2026-09-20 (review): how many of a live check's records are read back and
# checked (ONE efetch either way, so this costs no extra call) -- 3 was too thin
# a sample once a record has to prove it names the gene.
PUBMED_CHECK_RETMAX = 5
# "downgraded"    = would-be "unreported" blocks the live check turned into corpus_absent
# "ne_checks"     = PI 2026-09-20 (point 3): no_evidence blocks that ran the live check
# "ne_downgraded" = of those, the ones PubMed turned into corpus_absent
# "verify_*"      = FIX 2026-09-20 (review): the efetch that checks whether the
#                   records an esearch returned really name the gene in the domain
_PUBMED_STATS: dict = {"calls": 0, "cached": 0, "errors": 0, "seconds": 0.0, "downgraded": 0,
                       "ne_checks": 0, "ne_downgraded": 0, "verify_calls": 0,
                       "verify_cached": 0, "verify_errors": 0, "verify_records": 0,
                       "verify_rejected": 0}
_GENE_NAME_CACHE: dict = {}
_GENE_NAME_STATS: dict = {"calls": 0, "cached": 0, "errors": 0, "seconds": 0.0,
                          "resolved": 0, "offline": 0, "live": 0}
# FIX 2026-09-23: the official full name comes from the LOCAL NCBI gene_info
# dump, not from a live esearch whose first hit can be another gene.  Two
# switches, both default-on, so the old behaviour is one env var away:
#   MRAG_GENE_FULLNAME_OFFLINE=0       -> skip the dump (pure legacy path)
#   MRAG_GENE_FULLNAME_LIVE_FALLBACK=0 -> never call NCBI for a name
_EVIDENCE_GATE_PY = ("/BIO1/MRAG/Agent/orchestrator/src/mrag_orchestrator/"
                     "nodes/evidence_gate.py")
_EG_MOD: list = [None]


def _evidence_gate_mod():
    """nodes/evidence_gate.py loaded BY FILE PATH (never as a package: this
    package must not import mrag_orchestrator).  None when unavailable."""
    if _EG_MOD[0] is None:
        try:
            import importlib.util as _ilu
            _p = os.environ.get("MRAG_EVIDENCE_GATE_PY") or _EVIDENCE_GATE_PY
            _sp = _ilu.spec_from_file_location("mrag_evidence_gate_narr", _p)
            _m = _ilu.module_from_spec(_sp)
            _sp.loader.exec_module(_m)
            _EG_MOD[0] = _m
        except Exception as _e:      # pragma: no cover - environment dependent
            print("[narrative] evidence_gate not loadable (%s); gene full names "
                  "fall back to the live lookup" % str(_e)[:120], flush=True)
            _EG_MOD[0] = False
    return _EG_MOD[0] or None


def _tied_to_other_gene(text, gene, gene_name="", generif=False):
    """E1b (2026-09-25) for the Section-14 counts: the other gene (e.g. "CHGA
    (chromogranin A)") when this passage's mention of `gene` is really another
    gene's -- the passage ties the symbol to a gene that lists it as an alias
    ("chromogranin A (CGA)"), or, for a GeneRIF line, the line is about that
    gene and does not name `gene`.  "" otherwise, and "" when evidence_gate is
    unavailable (legacy behaviour)."""
    _eg = _evidence_gate_mod()
    if _eg is None or not hasattr(_eg, "symbol_collision"):
        return ""
    try:
        if not _eg.names_strict():
            return ""
        if generif:
            return _eg.generif_tied_to_other(text, gene, gene_name)
        # the card gate matches the symbol case-INSENSITIVELY (_gpat), so the
        # collision test must see the same hits
        kept, other = _eg.symbol_collision(text, gene, gene_name, ci=True)
        return other if not kept else ""
    except Exception:
        return ""


def _flag(name, default=True):
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() not in ("0", "false", "no", "off")
# full names that carry no more information than the symbol itself
_GENE_NAME_BAD = re.compile(
    r"^(uncharacterized|novel transcript|novel protein|hypothetical)", re.I)


def _gene_fullname_offline(gene):
    """Official full name from the local dump only (never a network call);
    "" when unavailable.  Used by the E1b collision tests."""
    _eg = _evidence_gate_mod()
    if _eg is None:
        return ""
    try:
        return str(_eg.gene_fullname(str(gene or "").strip()) or "")
    except Exception:
        return ""


def _pubmed_reset_stats():
    _PUBMED_STATS.update({"calls": 0, "cached": 0, "errors": 0, "seconds": 0.0,
                          "downgraded": 0, "ne_checks": 0, "ne_downgraded": 0,
                          "verify_calls": 0, "verify_cached": 0, "verify_errors": 0,
                          "verify_records": 0, "verify_rejected": 0})
    _GENE_NAME_STATS.update({"calls": 0, "cached": 0, "errors": 0, "seconds": 0.0, "resolved": 0})


def _gene_fullname(gene, timeout=25.0):
    """FIX 2026-09-19: the official full name of a gene, for the ADD-1 live
    PubMed check.

    WHY. The check used to search '<SYMBOL>[tiab]' only, so a gene whose
    literature is written out in words scored 0 and kept its red novelty claim
    plus the clause "a live PubMed search returned no record either" -- a
    statement about all of PubMed that the executed query did not support.
    Measured: one SCLC candidate had 0 records for '<SYMBOL>[tiab]' and 1 (a
    1988 paper) for its quoted full name, although that gene is long-established
    SCLC biology.

    One esearch(db=gene) + one esummary(db=gene) through genegpt.call_api
    (shared key-aware spacing: 0.12 s with the NCBI api_key, 0.4 s without;
    3 retries), cached per symbol for the life of the process
    and only ever called for a (gene, domain) pair that is about to be printed
    as unreported. ~1 s per new gene.

    Official full name ONLY (esummary nomenclaturename / description). The
    'Also known as' aliases are deliberately NOT used: many are ambiguous
    (TFPI2 -> PP5/REF1, ERG -> p55), and a false hit there would silently
    suppress a genuine novelty claim.

    Returns "" when nothing usable is found; NEVER raises."""
    g = str(gene or "").strip()
    if not g:
        return ""
    if g in _GENE_NAME_CACHE:
        _GENE_NAME_STATS["cached"] += 1
        return _GENE_NAME_CACHE[g]
    t0 = time.time()
    nm = ""
    _live_failed = False
    # FIX 2026-09-23 (the gene-full-name bug).  The live lookup below is
    # esearch(db=gene, "<SYM>[sym] AND human[orgn]", retmax=1) and takes the
    # FIRST hit.  For 9 of the 249 curated genes that first hit is a DIFFERENT
    # gene -- CGA -> "chromogranin A" (that is CHGA), LHFPL4 -> LHFPL3,
    # UGT2B11 -> UGT2B4, NPR3 -> NPRL3, MT1L -> MT1X.  The wrong name then goes
    # BOTH into the PubMed term ('"chromogranin A"[tiab]') and into
    # _record_names_gene, so records about the other gene verify as naming this
    # one: all 8 verified records of one NEC candidate (2,876 records in all)
    # came through the other gene's name.  evidence_gate.gene_fullname is an
    # exact symbol lookup in the local Homo_sapiens.gene_info dump -- no search,
    # so no first-hit ambiguity -- and it agrees with the live lookup on
    # 239/249.  The live path is kept ONLY for a symbol the dump does not carry.
    if _flag("MRAG_GENE_FULLNAME_OFFLINE", True):
        _eg = _evidence_gate_mod()
        if _eg is not None:
            try:
                nm = str(_eg.gene_fullname(g) or "")
            except Exception:
                nm = ""
            if nm:
                _GENE_NAME_STATS["offline"] += 1
    if not nm and _flag("MRAG_GENE_FULLNAME_LIVE_FALLBACK", True):
        try:
            import sys as _sys
            if "/BIO1/MRAG/Agent/libs/genegpt" not in _sys.path:
                _sys.path.insert(0, "/BIO1/MRAG/Agent/libs/genegpt")
            import genegpt as _gg
            _eu = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
            _q = urllib.parse.quote(f"{g}[sym] AND human[orgn]")
            # FIX 2026-09-25 (E1): read up to 5 hits back and take only a record
            # whose OFFICIAL symbol is this one -- the first hit of a [sym]
            # search can be another gene that lists the symbol as an alias
            # (CGA -> CHGA).  No matching record -> no name.
            _xml = _call_api_compat(_gg, f"{_eu}/esearch.fcgi?db=gene&retmax=5&term={_q}",
                                    timeout, "xml").decode("utf-8", "ignore")
            _GENE_NAME_STATS["calls"] += 1
            _ids = re.findall(r"<Id>(\d+)</Id>", _xml)
            if _ids:
                _d = json.loads(_call_api_compat(
                    _gg, f"{_eu}/esummary.fcgi?db=gene&retmode=json&id={','.join(_ids)}",
                    timeout, "json").decode("utf-8", "ignore"))
                _GENE_NAME_STATS["calls"] += 1
                _res = (_d or {}).get("result") or {}
                _r = {}
                for _gid in _ids:
                    _cand = _res.get(_gid) or {}
                    # a stub/legacy esummary without "name" keeps the old
                    # single-hit behaviour
                    if str(_cand.get("name") or g).upper() == g.upper():
                        _r = _cand
                        break
                nm = str(_r.get("nomenclaturename") or _r.get("description") or "")
            if nm:
                _GENE_NAME_STATS["live"] += 1
        except Exception:
            _GENE_NAME_STATS["errors"] += 1
            nm = ""
            _live_failed = True
    # a quoted [tiab] phrase cannot carry quotes or brackets
    nm = re.sub(r"\s+", " ", re.sub(r'["\[\]()]', " ", nm)).strip()
    if len(nm) < 5 or nm.lower() == g.lower() or _GENE_NAME_BAD.match(nm):
        nm = ""
    if nm:
        _GENE_NAME_STATS["resolved"] += 1
    _GENE_NAME_STATS["seconds"] += time.time() - t0
    if not _live_failed:          # followup 2026-09-28: a failed lookup is retried next time
        _GENE_NAME_CACHE[g] = nm
    return nm


def _pubmed_domain_check(gene, query_frag, retmax=3, timeout=25.0, gene_name=""):
    """ADD-1 (2026-09-18): ONE live PubMed esearch for a (gene, domain) pair the
    indexed corpus reports as unreported.

    WHY. The corpora are snapshots of very different sizes (prostate ~3.8k,
    sclc ~3.0k, nec 787 parsed papers), so "absent from our index" is at least
    as likely to mean "small corpus" as "nobody has published this". A novelty
    claim must survive a live check against all of PubMed.

    `query_frag` is literature.DOMAIN_PUBMED[key] (carried on the domain spec as
    "pubmed"); the term sent is '<GENE>[tiab] AND (<frag>)', or, with a
    `gene_name` (FIX 2026-09-19, from _gene_fullname -- the gene side used to be
    the symbol alone and missed genes cited only by their full name),
    '(<GENE>[tiab] OR "<name>"[tiab]) AND (<frag>)'.
    Keep it identical to literature.pubmed_query(). Rate limiting and
    retries come from genegpt.call_api (0.12 s spacing with the NCBI api_key,
    0.4 s without; 3 tries), the same client the GeneRIF fetch uses. Cached per (gene, frag) for the life of the
    process, so the narrative-time build and the runner rebuild share results.

    Returns {"ok", "count", "pmids", "error", "date", "term"}; NEVER raises --
    a failed check leaves the caller's "unreported" status alone (it must never
    silently turn into a claim we did not verify)."""
    g = str(gene or "").strip()
    frag = str(query_frag or "").strip()
    date = time.strftime("%Y-%m-%d")
    if not g or not frag:
        return {"ok": False, "count": 0, "pmids": [], "date": date, "term": "",
                "gene_name": "", "error": "no PubMed query for this domain"}
    nm = re.sub(r"\s+", " ", re.sub(r'["\[\]()]', " ", str(gene_name or ""))).strip()
    term = (f'({g}[tiab] OR "{nm}"[tiab]) AND ({frag})' if nm
            else f"{g}[tiab] AND ({frag})")
    ck = (g, nm, frag)
    if ck in _PUBMED_CACHE:
        _PUBMED_STATS["cached"] += 1
        return dict(_PUBMED_CACHE[ck])
    t0 = time.time()
    try:
        import sys as _sys
        if "/BIO1/MRAG/Agent/libs/genegpt" not in _sys.path:
            _sys.path.insert(0, "/BIO1/MRAG/Agent/libs/genegpt")
        import genegpt as _gg
        url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
               "?db=pubmed&retmode=json&sort=relevance&retmax=" + str(int(retmax))
               + "&term=" + urllib.parse.quote(term))
        d = json.loads(_call_api_compat(_gg, url, timeout, "json").decode("utf-8", "ignore"))
        r = (d or {}).get("esearchresult", {}) or {}
        if r.get("ERROR"):
            raise RuntimeError(str(r.get("ERROR"))[:160])
        res = {"ok": True, "count": int(r.get("count") or 0),
               "pmids": [str(x) for x in (r.get("idlist") or []) if str(x).isdigit()][:retmax],
               "date": date, "term": term, "gene_name": nm, "error": ""}
        _PUBMED_STATS["calls"] += 1
    except Exception as e:
        res = {"ok": False, "count": 0, "pmids": [], "date": date, "term": term,
               "gene_name": nm,
               # this string lands in the report JSON: redact before storing
               "error": _redact_ncbi("%s: %s" % (type(e).__name__, e))[:200]}
        _PUBMED_STATS["calls"] += 1
        _PUBMED_STATS["errors"] += 1
    _PUBMED_STATS["seconds"] += time.time() - t0
    if res["ok"]:                 # followup 2026-09-28: a FAILED check is not cached (a retry re-asks)
        _PUBMED_CACHE[ck] = res
    return dict(res)


def _pubmed_records(pmids, timeout=30.0):
    """ONE efetch for the PMIDs a live check returned ->
    {pmid: {"title", "abstract", "mesh", "keywords"}}.

    Returns (records, error_string); NEVER raises. MeSH descriptors, qualifiers
    and substance names are kept because the domain fragments of DOMAIN_PUBMED
    are largely MeSH ("Small Cell Lung Carcinoma"[mh]), so a record can be about
    the domain without spelling it in the title or abstract."""
    pm = [str(p) for p in (pmids or []) if str(p).isdigit()]
    if not pm:
        return {}, ""
    try:
        import sys as _sys
        if "/BIO1/MRAG/Agent/libs/genegpt" not in _sys.path:
            _sys.path.insert(0, "/BIO1/MRAG/Agent/libs/genegpt")
        import genegpt as _gg
        xml = _call_api_compat(
            _gg, "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            "?db=pubmed&retmode=xml&rettype=abstract&id=" + ",".join(pm),
            timeout, "xml").decode("utf-8", "ignore")
    except Exception as e:
        # this string lands in the report JSON: redact before storing
        return {}, _redact_ncbi("%s: %s" % (type(e).__name__, e))[:200]
    from html import unescape as _unesc
    out = {}
    for blk in re.split(r"(?=<PubmedArticle[ >])", xml):
        m = re.search(r"<PMID[^>]*>(\d+)</PMID>", blk)
        if not m:
            continue

        def _t(pat, sep=" ", _b=blk):
            return re.sub(r"\s+", " ", _unesc(sep.join(
                re.sub(r"<[^>]+>", " ", x) for x in re.findall(pat, _b, re.S)))).strip()
        out[m.group(1)] = {
            "title": _t(r"<ArticleTitle[^>]*>(.*?)</ArticleTitle>")
                     + " " + _t(r"<VernacularTitle[^>]*>(.*?)</VernacularTitle>"),
            "abstract": _t(r"<AbstractText[^>]*>(.*?)</AbstractText>"),
            "mesh": " ; ".join([_t(r"<DescriptorName[^>]*>(.*?)</DescriptorName>", " ; "),
                                _t(r"<QualifierName[^>]*>(.*?)</QualifierName>", " ; "),
                                _t(r"<NameOfSubstance[^>]*>(.*?)</NameOfSubstance>", " ; ")]),
            "keywords": _t(r"<Keyword[^>]*>(.*?)</Keyword>", " ; ")}
    return out, ""


def _record_names_gene(text, gene, gene_name=""):
    """Does this record's OWN text name the gene?  Returns "" or how it matched.

    Case-AWARE on purpose. The esearch that produced the record ran
    '<SYM>[tiab]', which is case-INSENSITIVE, so repeating a case-insensitive
    test here would confirm every hit it returned -- that is exactly how
    'impact' (IMPACT), 'she' (SHE), 'set' (SET), 'star' (STAR) and 'Mars'
    (MARS) came back as records that "name the gene". Accepted:
      * the official symbol in its own casing (IMPACT, PAX6) -- an English word
        written as ordinary prose no longer counts;
      * any casing when the symbol contains a digit (Pax6, Tdrd1, Kif20a):
        a digit rules out an English word;
      * the official full name from NCBI Gene (neuropeptide Y), case-insensitive.
    Word boundaries are non-alphanumeric so CD8-positive / GRPR-targeted count.

    FIX 2026-09-25 (E3 / E1b): the decision is now evidence_gate.names_gene,
    so the rescue verifier and the score gate share ONE rule.  A full-name hit
    no longer counts when it is paired with another gene's symbol -- a record
    reading "NCAM (neuronal cell adhesion molecule)" rescued the gene whose
    official full name that is, while its symbol appears 0 times -- nor, unpaired,
    when the name is shared and neither the symbol nor an unambiguous alias
    appears; a symbol hit no longer counts when the record ties the symbol to
    another gene ("chromogranin A (CGA)").  The legacy rule below runs only
    when evidence_gate cannot be loaded."""
    t = str(text or "")
    g = str(gene or "").strip()
    if not g:
        return ""
    _eg = _evidence_gate_mod()
    if _eg is not None and hasattr(_eg, "names_gene_detail"):
        try:
            return _eg.names_gene(t, g, gene_name)
        except Exception:
            pass
    b = r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])"
    if re.search(b % re.escape(g), t):
        return "symbol"
    if re.search(r"\d", g) and re.search(b % re.escape(g), t, re.I):
        return "symbol-ci"
    nm = str(gene_name or "").strip()
    if len(nm) >= 5 and re.search(b % re.escape(nm), t, re.I):
        return "full-name"
    return ""


def _record_cooccur(rec, gene, gene_name, dom, window):
    """ROUND 2 (2026-09-26): does this PubMed record name the gene TOGETHER
    with the domain?  True when its title or author keywords mention the
    domain, or one sentence of the abstract (window 0) names both
    (evidence_gate.gene_domain_cooccur: symbol with the case rule, official
    full name or an unambiguous alias).  True when evidence_gate cannot be
    loaded (the test is then simply not applied)."""
    _eg = _evidence_gate_mod()
    if _eg is None or not hasattr(_eg, "gene_domain_cooccur"):
        return True
    try:
        head = " ; ".join(x for x in (rec.get("title") or "", rec.get("keywords") or "") if x)
        return bool(_eg.gene_domain_cooccur(rec.get("abstract") or "", gene, gene_name or "",
                                            dom, head, None, window=int(window)))
    except Exception:
        return True


def _record_evfilter(rec, gene, gene_name, dom):
    """evaudit 2026-09-27: evidence_gate.record_filters for one PubMed record
    ("" = counts).  "" when evidence_gate cannot be loaded (the filters are
    then simply not applied)."""
    _eg = _evidence_gate_mod()
    if _eg is None or not hasattr(_eg, "record_filters"):
        return ""
    try:
        return str(_eg.record_filters(dom, rec, gene, gene_name) or "")
    except Exception:
        return ""


def _pubmed_verify(gene, dom, res, timeout=30.0, cooc=None, evfilter=None):
    """FIX 2026-09-20 (review): confirm the records a live check returned really
    name the gene in this domain, BEFORE any sentence cites them.

    WHY. esearch counts query matches, not gene mentions. '<SYM>[tiab] AND
    (<domain frag>)' matches the word, not the gene, so IMPACT/SCLC returned 988
    records (the top ones are SCLC reviews that use the word "impact"), SHE/SCLC
    142 (case reports whose abstracts say "she"), MARS/NEC 3 melanoma papers
    (the NEC fragment's "Neuroendocrine Tumors"[mh] explodes over melanoma), and
    the full-name arm can hit a different gene ("tudor domain containing 1"
    inside "Staphylococcal nuclease and Tudor domain containing 1" = SND1).
    Printing those as "N records name <GENE> in this domain [PMID:...]" is a
    false statement with fabricated citations, so every record now has to pass
    two gates before it is counted or cited:
      1. _record_names_gene on title + abstract + keywords (case-aware), and
      2. _domain_mentions on title + abstract + MeSH + keywords -- the SAME
         domain test an indexed corpus passage must pass, so both kinds of
         evidence are judged by one rule.
    ONE efetch per (gene, domain) with records, cached like the esearch.

    ROUND 2 (2026-09-26): with `cooc` (a sentence window, 0 = same sentence)
    a record must also name the gene TOGETHER with the domain
    (_record_cooccur).  Only the score-side PubMed rescue passes it
    (evidence_gate.pubmed_rescue, sclc by default): one SCLC candidate was
    rescued by a paper whose only SCLC words are a cohort list.  Records
    refused for that reason are listed in "verify_cooc_rejected".

    evaudit 2026-09-27: with `evfilter` (the score-side rescue; followup 2026-09-28,
    fix 2: also the Section-14 cards of a curated run) every
    record that passed is also judged by evidence_gate.record_filters -- the
    paper-level NET test (nec), the sentence-level veto (nec) and the
    screened-negative list test -- and the refused ones are listed in
    "verify_filter_rejected" {pmid: reason}; the verified records' title /
    abstract / MeSH are returned in "verify_records" for the evidence audit.

    Returns a COPY of `res` with "checked" (PMIDs we actually read), "verified"
    (PMIDs that passed both gates), "verify_how", "verify_ok" and
    "verify_error". NEVER raises. A failed or empty efetch leaves "verified"
    empty and "verify_ok" False: the caller must then NOT claim the records
    name the gene."""
    out = dict(res or {})
    out.setdefault("checked", [])
    out.setdefault("verified", [])
    out.setdefault("verify_how", {})
    out.setdefault("verify_ok", False)
    out.setdefault("verify_error", "")
    if not out.get("ok") or int(out.get("count") or 0) <= 0:
        return out
    pmids = [str(p) for p in (out.get("pmids") or []) if str(p).isdigit()]
    if not pmids:
        out["verify_error"] = "esearch returned a count but no usable PMID"
        return out
    ck = (str(gene), str(out.get("gene_name") or ""), str(dom.get("key") or ""), tuple(pmids),
          cooc)
    if evfilter:
        ck = ck + ("evfilter",)
    if ck in _PUBMED_VERIFY_CACHE:
        _PUBMED_STATS["verify_cached"] += 1
        out.update(_PUBMED_VERIFY_CACHE[ck])
        return out
    t0 = time.time()
    recs, err = _pubmed_records(pmids, timeout=timeout)
    _PUBMED_STATS["verify_calls"] += 1
    _PUBMED_STATS["seconds"] += time.time() - t0
    if err or not recs:
        upd = {"checked": [], "verified": [], "verify_how": {}, "verify_ok": False,
               "verify_error": err or "efetch returned no record"}
        _PUBMED_STATS["verify_errors"] += 1
    else:
        checked, verified, how, cooc_rej = [], [], {}, []
        filt_rej, recs_keep = {}, {}
        for p in pmids:
            r = recs.get(p)
            if not r:
                continue
            checked.append(p)
            _w = _record_names_gene(" ".join([r["title"], r["abstract"], r["keywords"]]),
                                    gene, out.get("gene_name") or "")
            if _w and _domain_mentions(" ".join([r["title"], r["abstract"], r["mesh"],
                                                 r["keywords"]]),
                                       dom.get("terms") or (), dom.get("exclude") or ()):
                if cooc is not None and not _record_cooccur(r, gene, out.get("gene_name") or "",
                                                            dom, cooc):
                    cooc_rej.append(p)
                    continue
                if evfilter:
                    _why = _record_evfilter(r, gene, out.get("gene_name") or "", dom)
                    if _why:
                        filt_rej[p] = _why
                        continue
                    recs_keep[p] = {"title": r.get("title", ""), "abstract": str(r.get("abstract") or "")[:3000],
                                    "mesh": r.get("mesh", ""), "keywords": r.get("keywords", "")}
                verified.append(p)
                how[p] = _w
        upd = {"checked": checked, "verified": verified, "verify_how": how,
               "verify_ok": bool(checked), "verify_error": "",
               "verify_cooc_rejected": cooc_rej}
        if evfilter:
            upd["verify_filter_rejected"] = filt_rej
            upd["verify_records"] = recs_keep
        _PUBMED_STATS["verify_records"] += len(checked)
        _PUBMED_STATS["verify_rejected"] += len(checked) - len(verified)
    if not err:                   # followup 2026-09-28: a FAILED read is not cached (a retry re-reads)
        _PUBMED_VERIFY_CACHE[ck] = upd
    out.update(upd)
    return out


def _corpus_absent_text(gene, inline, res):
    """PI 2026-09-20 (point 1): the wording of the corpus_absent status.

    corpus_absent stays a category of its own -- a reader must be able to tell
    "we read the papers" (reported) from "we only counted PubMed records" --
    but it has to read immediately as NOT novel. So the sentence leads with
    "Reported in PubMed" and closes with the same "not flagged as novel." tail
    the reported branch uses: same verdict, different evidence depth. It stays
    grey in the template (.nf-nov-corpus_absent), never the red .nf-novel.

    Point 2: NO search date in the sentence. The date, the exact term sent and
    the PMIDs stay on the card JSON (entry["pubmed"]) and in the audit.

    FIX 2026-09-20 (review): the number said out loud and the PMIDs cited are
    now the VERIFIED records (_pubmed_verify), never the raw esearch count --
    that count includes records that merely contain the word. Only when every
    matching record was read and every one of them named the gene does the
    sentence keep the plain "<N> record(s) name <GENE>" shape; otherwise it says
    how many of how many, and "at least" when the query matched more records
    than were read. Called ONLY with at least one verified record.
    Deterministic Python -- the LLM never writes this."""
    n = int(res.get("count") or 0)
    ver = [str(p) for p in (res.get("verified") or [])]
    k = len(res.get("checked") or [])
    m = len(ver)
    cite = " ".join("[PMID:%s]" % p for p in ver)
    verb = "names" if m == 1 else "name"
    if m == n and k == n:
        body = "%d record%s %s %s in this domain" % (n, "" if n == 1 else "s", verb, gene)
    elif k >= n:
        body = ("%d of the %d records matching a PubMed query %s %s in this domain"
                % (m, n, verb, gene))
    else:
        body = ("at least %d of the %d records matching a PubMed query %s %s in this domain"
                % (m, n, verb, gene))
    return ("Reported in PubMed but absent from our indexed %s corpus: %s%s; "
            "not flagged as novel." % (inline, body, (" " + cite) if cite else ""))


def _pubmed_unverified_clause(res, lead):
    """FIX 2026-09-20 (review): the deterministic clause for "the query matched
    records, but none of them was confirmed to name this gene in this domain"
    (or "we could not read them"). No PMID is cited, because none was
    confirmed, and no date (point 2). `lead` is the subject, e.g.
    "a PubMed search" or "a live PubMed search for <GENE> (<full name>)"."""
    n = int(res.get("count") or 0)
    k = len(res.get("checked") or [])
    s = "" if n == 1 else "s"
    if not res.get("verify_ok"):
        return ("%s matched %d record%s that could not be retrieved for checking in "
                "this run" % (lead, n, s))
    if k >= n:
        if n == 1:
            return "%s matched 1 record, which does not name the gene in this domain" % lead
        return ("%s matched %d records, none of which names the gene in this domain"
                % (lead, n))
    return ("%s matched %d records, but none of the %d most relevant names the gene in "
            "this domain" % (lead, n, k))


def _no_evidence_resolve(gene, inline, dom, base, evfilter=None, exclude=None):
    """PI 2026-09-20 (point 3): the live PubMed check now runs for the two
    no_evidence variants too (before today it ran only for a would-be
    "unreported" block).

    WHY. "no novelty assessment is made for this domain" was printed for genes
    PubMed plainly reports in that domain -- the same blind spot ADD-1 fixed for
    the red claim, just quieter. Both variants are in scope: the corpus named
    the gene but tied it to nothing ("named"), and the corpus retrieved no
    passage naming it at all.

    `base` is the deterministic clause describing what the indexed corpus held,
    with NO trailing punctuation.

    Returns (status, text, pubmed_result_or_None):
      * VERIFIED records (FIX 2026-09-20: records that really name the gene in
        this domain, not merely query matches) -> ("corpus_absent", ...)
      * matches that none of the read records confirms -> ("no_evidence", base +
        one clause saying so, + the unchanged "no novelty assessment" tail)
      * no record at all -> ("no_evidence", base + "a PubMed search found no
        records ... either" + tail)
      * check failed, or the domain has no PubMed query -> the unchanged
        no_evidence sentence, so a search we did not run is never described.
    No date in any of them (point 2); the result dict carries it to the JSON.
    followup 2026-09-28 (fix 2): with `evfilter` the records are verified WITH the
    evidence filters and `exclude` (the row's filtered set) is never verified; when
    every record naming the gene here was excluded, the sentence says so (grey,
    no PMID) instead of claiming or denying anything.
    NEVER raises."""
    tail = "; no novelty assessment is made for this domain."
    frag = str(dom.get("pubmed") or "")
    if not frag:
        return "no_evidence", base + tail, None
    _PUBMED_STATS["ne_checks"] += 1
    _chk = _pubmed_domain_check(gene, frag, retmax=PUBMED_CHECK_RETMAX, gene_name=_gene_fullname(gene))
    res = (_pubmed_verify(gene, dom, _chk, evfilter=True) if evfilter else _pubmed_verify(gene, dom, _chk))
    if exclude:
        res = _pubmed_drop_excluded(res, exclude)
    if not res["ok"]:
        return "no_evidence", base + tail, res
    if res["verified"]:
        _PUBMED_STATS["ne_downgraded"] += 1
        return "corpus_absent", _corpus_absent_text(gene, inline, res), res
    if res.get("verify_filter_rejected"):
        return ("no_evidence", base + "; " + _pubmed_filtered_clause(res, "a PubMed search") + tail, res)
    if int(res.get("count") or 0) <= 0:
        return ("no_evidence",
                base + "; a PubMed search found no records for this gene in this "
                       "domain either" + tail, res)
    return ("no_evidence",
            base + "; " + _pubmed_unverified_clause(res, "a PubMed search") + tail, res)


# ---------------------------------------------------------------------------
# followup 2026-09-28 (fix 2): the Section-14 cards follow the gate's evidence rules
# ---------------------------------------------------------------------------
# WHY.  The card path (per-domain Qdrant hits + the gene's GeneRIF lines, then a
# live PubMed check through _pubmed_verify WITHOUT the evidence filters) could
# cite exactly the papers the score-side filters had removed: an SCLC target's
# screened-negative-list paper, a NEC target's NET paper, another NEC target's
# PanNET sentence.  With card_evidence_context() -- built from the
# curated rows by nodes.viz_report.rebuild_notable_cards -- a card now
#   * never cites a PMID of its row's filtered set (evidence_counted.filtered,
#     and the rescue's filter_rejected records) -- not in a Novelty entry, the
#     cross-domain sentence, the GeneRIF list, the prompt or its JSON;
#     review 2026-09-28 (finding 2): the gate filters per item and per domain,
#     so the card does too -- a PMID the gate filtered for domain A and counted
#     for B (another item of the paper, or the same line for B) is kept for B;
#     only a PMID filtered and counted NOWHERE is excluded card-wide, and a
#     GeneRIF line the gate itself filtered is dropped for exactly those
#     domains (from the GeneRIF list and prompt when it counted for none);
#   * applies the gate's filters to every item it would count for a domain: 1b
#     (NEC sentence veto) and 1c (screened-negative list) on the item's text,
#     1a (NET paper) on its PubMed record(s) -- the runner's verdicts first,
#     else one cached efetch;
#   * verifies its live PubMed records WITH the filters (_pubmed_verify
#     evfilter), and says so in grey when records naming the gene were all
#     excluded -- never a red novelty claim resting on invalid records;
#   * for a gene admitted ONLY on verified live-PubMed rescue records, cites
#     exactly those records (card evidence == counted evidence);
#   * reads the scorer's own GeneRIF lines (scored.json + _generif_fetch.json)
#     instead of fetching them again, so card and scorer see the same lines.
# A card none of whose items is excluded is byte for byte today's.  Off -- the
# pre-fix cards -- with the evidence layers off (MRAG_EVIDENCE_AUDIT=0 /
# MRAG_EVIDENCE_DEFENCE=0, the full rollback) or without a context (the
# narrative-time cards, a run without curated targets).
_WHY_SHORT = (("list-negative", "a screened-negative gene list"),
              ("sentence veto", "a NET-type tumour in the sentence naming the gene"),
              ("paper-level NET", "a NET-type (well-differentiated) tumour paper"),
              ("paper_net", "a NET-type (well-differentiated) tumour paper"))


def _why_short(why):
    w = str(why or "")
    for k, v in _WHY_SHORT:
        if w.startswith(k):
            return v
    return (w.split(":", 1)[0] or "an evidence filter")[:80]


def _evidence_defence_on():
    """evidence_gate.evidence_defence_on() (MRAG_EVIDENCE_DEFENCE / MRAG_EVIDENCE_AUDIT)."""
    _eg = _evidence_gate_mod()
    if _eg is not None and hasattr(_eg, "evidence_defence_on"):
        try:
            return bool(_eg.evidence_defence_on())
        except Exception:
            pass
    for _k in ("MRAG_EVIDENCE_DEFENCE", "MRAG_EVIDENCE_AUDIT"):
        if str(os.environ.get(_k) or "").strip().lower() in ("0", "off", "false", "no", "none"):
            return False
    return True


def _rif_item_key(text, pmids=None):
    """Review 2026-09-28 (finding 2): one GeneRIF line's identity -- its text and its PMIDs -- in the
    scorer's record (scored.json) and on the card, so the gate's verdict on THAT line applies to it."""
    return "%s\x1f%s" % (str(text or ""), ",".join(str(p).strip() for p in (pmids or []) if str(p).strip()))


def _hit_item_key(h):
    """Review 2026-09-28 (finding 2): one retrieved passage's identity -- PMID + the start of its
    snippet -- in scored.json's vdb hits (the scorer's hits come from the same card retriever) and on
    the card, so the gate's verdict on THAT passage applies to it.  "" when either part is missing."""
    if not isinstance(h, dict):
        return ""
    pm = str(h.get("pmid") or "").strip()
    t = re.sub(r"\s+", " ", str(h.get("snippet") or h.get("text") or "")).strip()[:150]
    return ("%s\x1f%s" % (pm, t)) if (pm and t) else ""


def card_evidence_context(rows, scored_doc=None, generif_sidecar=None):
    """followup 2026-09-28 (fix 2): the evidence facts of the curated rows that the Section-14
    cards must follow (see the block above), as plain data -- no network.
    Review 2026-09-28 (finding 2): per domain and per item, like the gate (version 2):
      {"version": 2, "genes": {run gene: {
          "exclude": {pmid: why}             PMID-wide: filtered (any domain) and counted NOWHERE -- never
                                             cited anywhere on the card
          "exclude_by_key": {key: {pmid: why}}  filtered for that domain and not counted for it -- never
                                             cited in that domain's entry
          "rif_filtered": {line id: {key: why}} the GeneRIF lines the gate itself filtered, per domain
          "rif_uncounted": [line id]         ... the ones it counted for no domain (not listed, not prompted)
          "hit_filtered": {passage id: {key: why}}, "hit_uncounted": [passage id]  the same for passages
          "gate", "counted": [pmids], "counted_by_key": {key: [pmids]},
          "rescue": {domain key: live-PubMed record}}},
       "paper_net": {pmid: 1a verdict} (the score runner's), "generif": {run gene: [{text, pmids}]},
       "generif_retmax": int}
    A counted GeneRIF line counts for the domains its text mentions (the gate's tie of a GeneRIF line
    is exactly that test), a counted passage for the domains its stored current-version tie names,
    the rescue's verified records for their domain -- each minus the domains a filter removed for that
    item.  When that cannot be told (no record text, an older stored tie), a counted item counts for
    every domain it was not filtered for: a card never drops evidence the gate counted.
    `rows` = curated_targets.json, `scored_doc` = scored.json, `generif_sidecar` =
    _generif_fetch.json (its "pmids" + "retmax" make the scorer's own GeneRIF lines reusable).
    None when `rows` is empty."""
    sd = scored_doc if isinstance(scored_doc, dict) else {}
    gtx_all = sd.get("generif") if isinstance(sd.get("generif"), dict) else {}
    vdb_all = sd.get("vdb") if isinstance(sd.get("vdb"), dict) else {}
    scx = generif_sidecar if isinstance(generif_sidecar, dict) else {}
    spm_all = scx.get("pmids") if isinstance(scx.get("pmids"), dict) else {}
    eg = _evidence_gate_mod() if (gtx_all or vdb_all) else None
    tie_v = int(getattr(eg, "EVFILTER_TIE_VERSION", 0) or 0) if eg is not None else 0
    specs_cache = {}

    def _specs(keys):
        ck = tuple(keys)
        if ck not in specs_cache:
            out = {}
            if eg is not None and keys:
                try:
                    out = {x["key"]: x for x in eg.domain_specs(list(keys))}
                except Exception:
                    out = {}
            specs_cache[ck] = out
        return specs_cache[ck]

    def _ix(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    def _txt(x):
        return x if isinstance(x, str) else (str((x or {}).get("text") or "") if isinstance(x, dict) else None)

    genes = {}
    for r in rows or []:
        if not isinstance(r, dict) or not r.get("gene"):
            continue
        g = str(r["gene"])
        ec = r.get("evidence_counted") if isinstance(r.get("evidence_counted"), dict) else {}
        pe = r.get("pubmed_evidence") if isinstance(r.get("pubmed_evidence"), dict) else {}
        bydom = pe.get("by_domain") if isinstance(pe.get("by_domain"), dict) else {}
        rescued = str(r.get("evidence_gate") or "") == "pubmed"
        filt = [x for x in (ec.get("filtered") or []) if isinstance(x, dict)]
        keys = [str(k) for k in (r.get("evidence_gate_domains") or []) if str(k)]
        for k in [str(x.get("key") or "") for x in filt] + [str(k) for k in bydom]:
            if k and k not in keys:
                keys.append(k)
        gtx = gtx_all.get(g) if isinstance(gtx_all.get(g), list) else None
        spm = spm_all.get(g) if isinstance(spm_all.get(g), list) else None
        vdb = vdb_all.get(g) if isinstance(vdb_all.get(g), list) else None
        # what the gate's filters removed: per item (kind, i) and per domain ({pmid: why})
        item_keys, fbk = {}, {}
        for x in filt:
            why = str(x.get("why") or "evidence filter")
            for k in ([str(x.get("key"))] if x.get("key") else (keys or ["*"])):
                item_keys.setdefault((str(x.get("kind") or ""), _ix(x.get("i"))), {}).setdefault(k, why)
                for p in str(x.get("pmid") or "").split(","):
                    p = p.strip()
                    if p.isdigit():
                        fbk.setdefault(k, {}).setdefault(p, why)
        for k, d in bydom.items():
            for p, w in ((d or {}).get("filter_rejected") or {}).items():
                if str(p).isdigit():
                    fbk.setdefault(str(k), {}).setdefault(str(p), str(w or "evidence filter"))
        # what it counted, and for which domain
        cbk, counted = {}, set()

        def _count(kind, i, pms, tie):
            fk = item_keys.get((kind, i)) or {}
            ks = [k for k in (tie if tie is not None else keys) if k not in fk and "*" not in fk]
            if not ks:
                ks = [k for k in keys if k not in fk] or list(keys)
            counted.update(pms)
            for k in ks:
                cbk.setdefault(k, set()).update(pms)

        for x in ec.get("generif") or []:
            if not isinstance(x, dict):
                continue
            i = _ix(x.get("i"))
            pms = {str(p) for p in (x.get("pmids") or []) if str(p)}
            t = _txt(gtx[i]) if (gtx is not None and i is not None and 0 <= i < len(gtx)) else None
            if t is not None and spm is not None and not (i < len(spm) and {str(p) for p in (spm[i] or [])} == pms):
                t = None                  # not the record the gate read
            tie, sp = None, _specs(keys)
            if t is not None and keys and all(k in sp for k in keys):
                try:
                    tie = [k for k in keys if eg.mentions_domain(t, sp[k]["terms"], sp[k].get("exclude") or ())]
                except Exception:
                    tie = None
            _count("generif", i, pms, tie)
        for x in ec.get("snippets") or []:
            if not isinstance(x, dict) or not str(x.get("pmid") or ""):
                continue
            j, p = _ix(x.get("i")), str(x.get("pmid"))
            h = vdb[j] if (vdb is not None and j is not None and 0 <= j < len(vdb)) else None
            tie = None
            if (isinstance(h, dict) and str(h.get("pmid") or "") == p and isinstance(h.get("domain_tie"), dict)
                    and tie_v and (_ix(h.get("domain_tie_v")) or 0) >= tie_v):
                dt = h["domain_tie"]
                tie = [k for k in keys if (k not in dt) or dt.get(k)]
            _count("snippet", j, {p}, tie)
        if rescued:
            for k, d in bydom.items():
                ver = {str(p) for p in ((d or {}).get("verified") or []) if str(p)}
                if ver:
                    counted.update(ver)
                    cbk.setdefault(str(k), set()).update(ver)
        ex, exk = {}, {}
        for k, d in fbk.items():
            for p, w in d.items():
                if p not in counted:
                    ex.setdefault(p, w)
                if p not in (counted if k == "*" else cbk.get(k, set())):
                    exk.setdefault(k, {})[p] = w
        # the GeneRIF lines the gate itself filtered (identified in scored.json: text + PMIDs)
        rif_f, rif_unc = {}, []
        rc_idx = {_ix(x.get("i")) for x in (ec.get("generif") or []) if isinstance(x, dict)}
        for x in filt:
            i = _ix(x.get("i"))
            if str(x.get("kind") or "") != "generif" or gtx is None or i is None or not (0 <= i < len(gtx)):
                continue
            rid = _rif_item_key(_txt(gtx[i]), str(x.get("pmid") or "").split(","))
            for k in ([str(x.get("key"))] if x.get("key") else (keys or ["*"])):
                rif_f.setdefault(rid, {}).setdefault(k, str(x.get("why") or "evidence filter"))
            if i not in rc_idx and rid not in rif_unc:
                rif_unc.append(rid)
        # ... and the passages it filtered (scored.json vdb[gene][i]: the same PMID, its snippet)
        hit_f, hit_unc = {}, []
        sc_idx = {_ix(x.get("i")) for x in (ec.get("snippets") or []) if isinstance(x, dict)}
        for x in filt:
            j = _ix(x.get("i"))
            if str(x.get("kind") or "") != "snippet" or vdb is None or j is None or not (0 <= j < len(vdb)):
                continue
            if not isinstance(vdb[j], dict) or str(vdb[j].get("pmid") or "") != str(x.get("pmid") or ""):
                continue
            hid = _hit_item_key(vdb[j])
            if not hid:
                continue
            for k in ([str(x.get("key"))] if x.get("key") else (keys or ["*"])):
                hit_f.setdefault(hid, {}).setdefault(k, str(x.get("why") or "evidence filter"))
            if j not in sc_idx and hid not in hit_unc:
                hit_unc.append(hid)
        rescue = {}
        if rescued:
            for k, d in bydom.items():
                d = d or {}
                _xk = dict(ex)
                _xk.update(exk.get("*") or {})
                _xk.update(exk.get(str(k)) or {})
                ver = [str(p) for p in (d.get("verified") or []) if str(p) not in _xk]
                if not ver:
                    continue
                chk = [str(p) for p in (d.get("checked") or [])]
                rescue[str(k)] = {"ok": True, "count": int(d.get("count") or len(chk)), "pmids": chk,
                                  "date": str(d.get("date") or pe.get("date") or ""),
                                  "term": str(d.get("term") or ""),
                                  "gene_name": str(d.get("gene_name") or pe.get("gene_name") or ""),
                                  "checked": chk, "verified": ver,
                                  "verify_how": {p: v for p, v in (d.get("verify_how") or {}).items() if p in ver},
                                  "verify_ok": True, "verify_error": "", "error": "", "source": "rescue"}
        genes[g] = {"exclude": ex, "exclude_by_key": exk, "rif_filtered": rif_f, "rif_uncounted": rif_unc,
                    "hit_filtered": hit_f, "hit_uncounted": hit_unc,
                    "gate": str(r.get("evidence_gate") or ""), "counted": sorted(counted),
                    "counted_by_key": {k: sorted(v) for k, v in cbk.items()}, "rescue": rescue}
    if not genes:
        return None
    pf = ((sd.get("evidence_gate") or {}).get("paper_filter") or {})
    ctx = {"version": 2, "genes": genes, "paper_net": dict(pf.get("net") or {})}
    if sd and isinstance(scx.get("pmids"), dict) and scx.get("retmax"):
        lines = {}
        for g, st in (scx.get("genes") or {}).items():
            if (st or {}).get("status") not in ("ok", "empty"):
                continue
            t, pm = gtx_all.get(g), scx["pmids"].get(g)
            if not isinstance(t, list) or not isinstance(pm, list) or len(t) != len(pm):
                continue
            lines[g] = [{"text": a, "pmids": [str(x) for x in (b or [])]} for a, b in zip(t, pm)]
        ctx["generif"] = lines
        ctx["generif_retmax"] = int(scx["retmax"])
    return ctx


def _card_rules(evx, run_gene, retmax):
    """The per-gene rules of card_evidence_context for one card (None: no row -> today's card)."""
    gx = ((evx or {}).get("genes") or {}).get(str(run_gene))
    if gx is None:
        return None
    cx = {"exclude": dict(gx.get("exclude") or {}), "gate": gx.get("gate") or "",
          "counted": set(gx.get("counted") or []), "rescue": dict(gx.get("rescue") or {}), "evx": evx,
          # review 2026-09-28 (finding 2): per domain and per item
          "exclude_by_key": {str(k): dict(v or {}) for k, v in (gx.get("exclude_by_key") or {}).items()},
          "counted_by_key": {str(k): set(v or []) for k, v in (gx.get("counted_by_key") or {}).items()},
          "rif_filtered": {str(k): dict(v or {}) for k, v in (gx.get("rif_filtered") or {}).items()},
          "rif_uncounted": set(gx.get("rif_uncounted") or []),
          "hit_filtered": {str(k): dict(v or {}) for k, v in (gx.get("hit_filtered") or {}).items()},
          "hit_uncounted": set(gx.get("hit_uncounted") or [])}
    cx["xset"] = set(cx["exclude"])
    gl = (evx.get("generif") or {})
    if evx.get("generif_retmax") == int(retmax) and str(run_gene) in gl:
        cx["generif"] = [dict(r, pmids=list(r.get("pmids") or [])) for r in gl[str(run_gene)]]
    return cx


def _card_excl_for(cx, key):
    """Review 2026-09-28 (finding 2): {pmid: why} the entry of domain `key` may not cite -- the PMIDs
    filtered and counted nowhere, plus the ones filtered for THIS domain and not counted for it."""
    out = dict((cx or {}).get("exclude") or {})
    out.update(((cx or {}).get("exclude_by_key") or {}).get("*") or {})
    out.update(((cx or {}).get("exclude_by_key") or {}).get(str(key or "")) or {})
    return out


def _card_paper_net(cx, sp, pmids):
    """1a verdict ({pmid: bool is_net}) of `pmids` for a paper-rule domain: the score runner's
    verdicts, the PMIDs the gate counted FOR THIS DOMAIN (they passed 1a; review 2026-09-28: not
    ones counted only for another domain), else one cached efetch.  A PMID whose record could not
    be read is left out (and noted on cx["paper_errors"])."""
    evx = cx["evx"]
    pn, cache = evx.get("paper_net") or {}, evx.setdefault("_paper_cache", {})
    ck = (cx.get("counted_by_key") or {}).get(str(sp.get("key") or "")) or set()
    out, todo = {}, []
    for p in pmids:
        if p in pn:
            out[p] = True
        elif p in ck:
            out[p] = False
        elif (sp["key"], p) in cache:
            out[p] = cache[(sp["key"], p)]
        else:
            todo.append(p)
    if todo:
        eg = _evidence_gate_mod()
        recs, err = ({}, "evidence_gate not loadable")
        if eg is not None:
            try:
                recs, err = eg.fetch_paper_records(todo)
            except Exception as e:
                recs, err = {}, "%s: %s" % (type(e).__name__, str(e)[:120])
        for p in todo:
            rec = (recs or {}).get(p)
            if not rec:
                cx.setdefault("paper_errors", []).append((p, str(err or "record not returned")[:160]))
                continue
            try:
                v = eg.paper_net_verdict(sp, rec)
            except Exception:
                v = None
            cache[(sp["key"], p)] = out[p] = bool(v and v.get("is_net"))
    return out


def _card_exclusion(cx, sp, gene, gene_name, hit=None, rif=None, key=None):
    """"" or why a card item does not count for the domain `sp` (an evidence_gate spec; `key` = its
    key): a PMID the domain may not cite (_card_excl_for), the gate's own verdict on this GeneRIF
    line / passage for this domain, 1b / 1c on the item's text (the gate's retrieval_filters, on the same text
    the gate reads: title + chunk for a passage, the line for a GeneRIF), 1a on its papers."""
    k = str(key if key is not None else (sp or {}).get("key") or "")
    pms = ([str(hit.get("pmid") or "")] if hit is not None
           else [str(p) for p in ((rif or {}).get("pmids") or [])])
    pms = [p for p in pms if p]
    ex = _card_excl_for(cx, k)
    for p in pms:
        if p in ex:
            return ex[p]
    if rif is not None:
        _rw = (cx.get("rif_filtered") or {}).get(_rif_item_key(rif.get("text"), rif.get("pmids"))) or {}
    else:
        _rw = (cx.get("hit_filtered") or {}).get(_hit_item_key(hit)) or {}
    if _rw.get(k) or _rw.get("*"):
        return _rw.get(k) or _rw.get("*")
    eg = _evidence_gate_mod()
    if eg is None or not sp:
        return ""
    try:
        if eg.retrieval_filters_on():
            if hit is not None:
                _ti = str(hit.get("title") or "")
                _w = eg.retrieval_filters(sp, _ti + " " + str(hit.get("gate_text") or hit.get("snippet")
                                                               or hit.get("text") or ""),
                                          gene, gene_name, _ti)
            else:
                _w = eg.retrieval_filters(sp, str((rif or {}).get("text") or ""), gene, gene_name, "")
            if _w:
                return _w
        if pms and (sp.get("evfilter") or {}).get("paper_net") and eg.paper_net_on():
            v = _card_paper_net(cx, sp, pms)
            if all(v.get(p) for p in pms):
                return "paper-level NET: PMID %s is a well-differentiated-tumour paper" % ",".join(pms)
    except Exception:
        return ""
    return ""


def _pubmed_drop_excluded(res, exclude):
    """followup 2026-09-28 (fix 2): a record the score-side filters removed for this gene (the curated
    row's filtered set) is never a verified record here either: moved to verify_filter_rejected."""
    ver = [str(p) for p in (res.get("verified") or [])]
    if not exclude or not any(p in exclude for p in ver):
        return res
    out = dict(res)
    rej = dict(out.get("verify_filter_rejected") or {})
    for p in ver:
        if p in exclude:
            rej[p] = exclude[p]
    out["verified"] = [p for p in ver if p not in exclude]
    out["verify_filter_rejected"] = rej
    return out


def _pubmed_filtered_clause(res, lead):
    """followup 2026-09-28 (fix 2): "<lead> matched N record(s), and the M that name the gene in this
    domain are excluded by the evidence filters (<why>)" -- no PMID is cited."""
    n = int(res.get("count") or 0)
    rej = res.get("verify_filter_rejected") or {}
    m = len(rej)
    return ("%s matched %d record%s, and the %d that name%s the gene in this domain %s excluded by the "
            "evidence filters (%s)" % (lead, n, "" if n == 1 else "s", m, "s" if m == 1 else "",
                                       "is" if m == 1 else "are",
                                       "; ".join(sorted({_why_short(w) for w in rej.values()}))))


def _card_pubmed_json(res, xset):
    """The card's copy of a live-check record, without any PMID of the row's filtered set or one the
    evidence filters rejected here (the sentence was built from the full record; no excluded paper is
    listed on the card)."""
    rej = res.get("verify_filter_rejected") or {}
    xset = set(xset) | {str(p) for p in rej}
    out = {"ok": res["ok"], "count": res["count"],
           "pmids": [p for p in (res.get("pmids") or []) if str(p) not in xset], "date": res["date"],
           "term": res["term"], "gene_name": res.get("gene_name", ""),
           "checked": [p for p in (res.get("checked") or []) if str(p) not in xset],
           "verified": [p for p in (res.get("verified") or []) if str(p) not in xset],
           "verify_how": {p: v for p, v in dict(res.get("verify_how") or {}).items() if str(p) not in xset},
           "verify_ok": bool(res.get("verify_ok")), "verify_error": res.get("verify_error", ""),
           "error": res["error"]}
    if res.get("source"):
        out["source"] = res["source"]
    if rej:
        out["n_filter_rejected"] = len(rej)
        out["filter_rejected_why"] = sorted({_why_short(w) for w in rej.values()})
    return out


def _domain_mentions(text, terms, exclude=()):
    """Reference text gate for one literature domain: verbatim copy of
    mrag_orchestrator.literature.mentions_domain (that module cannot be imported
    here). Every `exclude` match is blanked with spaces of equal length, then any
    `terms` match counts. Python regexes, re.IGNORECASE; case-sensitive
    abbreviations carry a scoped (?-i:...) group."""
    t = str(text or "")
    for p in exclude:
        t = re.sub(p, lambda m: " " * len(m.group(0)), t, flags=re.I)
    return any(re.search(p, t, flags=re.I) for p in terms)


def _norm_domains(domains):
    """domain_specs() dicts (bare keys also accepted) -> normalised list in the
    given order, duplicates dropped. None / [] -> None (= legacy single-field
    Novelty). 'inline' is the in-sentence form: 'prostate cancer (PC)'."""
    if not domains:
        return None
    out, seen = [], set()
    for d in domains:
        if isinstance(d, str):
            d = {"key": d}
        if not isinstance(d, dict):
            continue
        key = str(d.get("key") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        label = str(d.get("label") or key).strip()
        abbr = str(d.get("abbr") or "").strip()
        display = str(d.get("display") or (f"{label} ({abbr})" if abbr else label))
        inl = (label[:1].lower() + label[1:]) if (label and not label[1:2].isupper()) else label
        terms = [str(x) for x in (d.get("terms") or []) if x] or [re.escape(label)]
        out.append({"key": key, "label": label, "abbr": abbr, "display": display,
                    "inline": inl + (f" ({abbr})" if abbr else ""),
                    "terms": terms,
                    "exclude": [str(x) for x in (d.get("exclude") or []) if x],
                    "pubmed": str(d.get("pubmed") or "")})
        # followup 2026-09-28 (fix 2): the domain's evidence-filter rules travel with it, so a
        # record check WITH the filters (_pubmed_verify evfilter) applies the domain's own
        # (nec: paper-level NET test, sentence veto); nothing reads them without the filters
        if d.get("evfilter"):
            out[-1]["evfilter"] = d["evfilter"]
    return out or None


_BATCH_RETRIEVE_RUNNER = r'''
import json, os, sys, time
p = json.load(open(sys.argv[1]))
res = {"ok": False, "error": None, "init_s": None, "search_s": None, "results": {}}
rt = None
try:
    sys.path.insert(0, p.get("paperqa_lib", "/BIO1/MRAG/Agent/libs/paperqa"))
    if p.get("threads"):  # optional torch intra-op thread cap (default: library default)
        try:
            import torch
            torch.set_num_threads(int(p["threads"]))
        except Exception:
            pass
    from mrag_paperqa.retriever import Retriever
    t0 = time.time()
    # Qdrant local mode is single-process: a corpus job (embed/relabel) holding the
    # lock used to make every gene score with no literature. Wait for it (2026-09-15).
    rt = None
    for _att in range(40):
        try:
            rt = Retriever(device=p.get("device", "cuda")); break
        except RuntimeError as _le:
            if "already accessed" not in str(_le):
                raise
            print("VDB locked by another process, retry in 30 s", flush=True); time.sleep(30)
    if rt is None:
        raise RuntimeError("PaperQA vector store still locked after 20 min; refusing to score without literature")
    res["init_s"] = round(time.time() - t0, 1)
    print("CARD_RETRIEVER init %.1fs, %d queries" % (res["init_s"], len(p["queries"])), flush=True)
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny, MatchTextAny

    def _text_search(q):
        # corpus filter AND chunk text containing the gene symbol (any listed
        # spelling; Qdrant local mode = case-sensitive substring), dense-ranked.
        must = []
        c = q.get("cancer")
        if c:
            must.append(FieldCondition(key="cancer", match=(MatchValue(value=c) if isinstance(c, str)
                                                            else MatchAny(any=list(c)))))
        must.append(FieldCondition(key="text", match=MatchTextAny(text_any=q["text_any"])))
        pts = rt.client.query_points(collection_name=rt.collection, query=rt.embed(q["query"]),
                                     limit=int(q.get("top_k", 3)), query_filter=Filter(must=must),
                                     with_payload=True).points
        out = []
        for h in pts:
            pl = h.payload or {}
            if pl.get("pmid"):
                out.append({"pmid": str(pl.get("pmid")), "title": str(pl.get("title") or ""),
                            "snippet": str(pl.get("text") or "")[:500],
                            "gate_text": str(pl.get("text") or "")[:4000],
                            "score": round(float(h.score), 4), "year": pl.get("year"),
                            "cancer": pl.get("cancer")})
        return out
    # Fast path: Qdrant local mode filters by looping over all ~190k payloads in
    # Python for EVERY filtered search (2-5 s CPU each; on a busy shared host a
    # minute per search). The corpus-label and gene-text masks are computed here
    # once per corpus / gene instead, and the library's own distance function
    # scores only the masked vectors. Self-checked against the API on the first
    # filtered query; any mismatch or error falls back to the API path.
    import numpy as np
    F = None
    FAST_ON = [True]
    try:
        from qdrant_client.local.distances import calculate_distance
        _coll = rt.client._client.collections[rt.collection]
        _vn = list(_coll.vectors.keys())
        if len(_vn) == 1:
            _vn = _vn[0]
            _n = len(_coll.payload)
            F = {"V": _coll.vectors[_vn][:_n], "dist": _coll.get_vector_params(_vn).distance,
                 "alive": ~(np.asarray(_coll.deleted[:_n], dtype=bool)
                            | np.asarray(_coll.deleted_per_vector[_vn][:_n], dtype=bool)),
                 "payload": _coll.payload, "calc": calculate_distance, "dom": {}, "gene": {},
                 "texts": [(pl or {}).get("text") if isinstance((pl or {}).get("text"), str) else ""
                           for pl in _coll.payload],
                 "labels": [(pl or {}).get("cancer") for pl in _coll.payload],
                 "checked": 0}
    except Exception as _fe:
        F = None
        print("CARD_RETRIEVER fast path unavailable (%s); using the Qdrant API" % str(_fe)[:120], flush=True)

    # 2026-09-25 (cardfast): optional speed-ups for an overloaded host, env-gated
    # (unset = unchanged); both give the same numbers as the per-query code.
    #  MRAG_CARD_MASK_WORKERS=N  the gene-text masks of all queries are computed
    #      up front in N forked workers with the SAME expression _fast_search
    #      uses; the first one is re-checked in-process (mismatch -> discarded).
    #  MRAG_CARD_PLAIN_CACHE=1   corpus-only (plain / plain-fallback) searches
    #      reuse ONE normalised copy of that corpus' vectors instead of gathering
    #      and normalising it again per query -- the normalisation qdrant's
    #      cosine_similarity() applies in place, then the same query
    #      normalisation and np.dot, so the scores are bitwise the same.
    try:
        _MW = int(os.environ.get("MRAG_CARD_MASK_WORKERS", "0") or 0)
    except ValueError:
        _MW = 0
    _PLAIN_CACHE = os.environ.get("MRAG_CARD_PLAIN_CACHE", "").strip().lower() in ("1", "true", "yes", "on")
    _EPS = None
    F_PLAIN = {}
    if F is not None and _PLAIN_CACHE:
        try:
            from qdrant_client.local.distances import EPSILON as _EPS
            from qdrant_client.http import models as _qm
            if F["dist"] != _qm.Distance.COSINE:
                _PLAIN_CACHE = False
        except Exception:
            _PLAIN_CACHE = False

    def _mask_expr(ta):
        words = ta.split()
        return np.fromiter((any(w in t for w in words) for t in F["texts"]),
                           dtype=bool, count=len(F["texts"]))

    def _mask_job(ta):
        return np.packbits(_mask_expr(ta))

    if F is not None and _MW > 1:
        try:
            _tas = [ta for ta in dict.fromkeys(q.get("text_any") for q in p["queries"])
                    if ta and ta not in F["gene"]]
            if len(_tas) > 1:
                import multiprocessing as _mp
                _tm = time.time()
                _nt = len(F["texts"])
                _nw = min(_MW, len(_tas))
                with _mp.get_context("fork").Pool(_nw) as _pool:
                    for _ta, _pk in zip(_tas, _pool.imap(_mask_job, _tas, chunksize=1)):
                        F["gene"][_ta] = np.unpackbits(_pk, count=_nt).astype(bool)
                if np.array_equal(F["gene"][_tas[0]], _mask_expr(_tas[0])):
                    print("CARD_RETRIEVER gene-text masks: %d in %.0fs (%d workers, self-check ok)"
                          % (len(_tas), time.time() - _tm, _nw), flush=True)
                else:
                    F["gene"].clear()
                    print("CARD_RETRIEVER parallel gene-text masks disagree with the in-process "
                          "expression; computing them per query", flush=True)
        except Exception as _me:
            F["gene"].clear()
            print("CARD_RETRIEVER parallel gene-text masks failed (%s); computing them per query"
                  % str(_me)[:120], flush=True)

    def _fast_search(q):
        c = q.get("cancer")
        keys = [c] if isinstance(c, str) else list(c or [])
        m = F["alive"].copy()
        if keys:
            kk = tuple(keys)
            if kk not in F["dom"]:
                F["dom"][kk] = np.fromiter(
                    ((lab in keys) if isinstance(lab, str) else
                     (isinstance(lab, (list, tuple)) and any(x in keys for x in lab))
                     for lab in F["labels"]), dtype=bool, count=len(F["labels"]))
            m &= F["dom"][kk]
        if q.get("text_any"):
            ta = q["text_any"]
            if ta not in F["gene"]:
                words = ta.split()
                F["gene"][ta] = np.fromiter((any(w in t for w in words) for t in F["texts"]),
                                            dtype=bool, count=len(F["texts"]))
            m &= F["gene"][ta]
        if _PLAIN_CACHE and not q.get("text_any"):
            _ck = tuple(keys)
            if _ck not in F_PLAIN:
                _pidx = np.flatnonzero(m)
                _Vc = F["V"][_pidx]
                if _Vc.size:
                    _vn = np.linalg.norm(_Vc, axis=-1)[:, np.newaxis]
                    _Vc /= np.where(_vn != 0.0, _vn, _EPS)
                F_PLAIN[_ck] = (_pidx, _Vc)
            idx, _Vn = F_PLAIN[_ck]
            if idx.size == 0:
                return []
            qv = np.asarray(rt.embed(q["query"]), dtype=np.float32)
            _qn = np.linalg.norm(qv)
            qv /= np.where(_qn != 0.0, _qn, _EPS)
            sc = np.asarray(np.dot(_Vn, qv))
        else:
            idx = np.flatnonzero(m)
            if idx.size == 0:
                return []
            qv = np.asarray(rt.embed(q["query"]), dtype=np.float32)
            sc = np.asarray(F["calc"](qv, F["V"][idx], F["dist"]))
        out = []
        for j in np.argsort(-sc, kind="stable")[: int(q.get("top_k", 3))]:
            pl = F["payload"][int(idx[j])] or {}
            if pl.get("pmid"):
                out.append({"pmid": str(pl.get("pmid")), "title": str(pl.get("title") or ""),
                            "snippet": str(pl.get("text") or "")[:500],
                            "gate_text": str(pl.get("text") or "")[:4000],
                            "score": round(float(sc[j]), 4), "year": pl.get("year"),
                            "cancer": pl.get("cancer")})
        return out

    def _api_plain(q):
        hits = rt.search(q["query"], top_k=int(q.get("top_k", 3)), cancer=q.get("cancer") or None)
        return [{"pmid": str(h.pmid), "title": str(h.title or ""),
                 "snippet": str(h.text or "")[:500], "gate_text": str(h.text or "")[:4000],
                 "score": round(float(h.score), 4), "year": h.year, "cancer": h.cancer}
                for h in hits if h.pmid]

    def _filtered(q, api):
        """corpus (+ gene-text) filtered search via the fast path when verified."""
        if F is not None and FAST_ON[0]:
            try:
                rows = _fast_search(q)
                if F["checked"] is not True:
                    # compare with the API until a non-empty result agreed (max 3)
                    ref = api(q)
                    if [r["pmid"] for r in rows] != [r["pmid"] for r in ref]:
                        print("CARD_RETRIEVER fast path disagrees with the API on %s; API only"
                              % q["id"], flush=True)
                        FAST_ON[0] = False
                        return ref
                    F["checked"] = True if (ref or int(F["checked"]) >= 2) else int(F["checked"]) + 1
                    if F["checked"] is True:
                        print("CARD_RETRIEVER fast path verified against the API", flush=True)
                return rows
            except Exception as _fe:
                print("CARD_RETRIEVER fast path error (%s); API for %s" % (str(_fe)[:120], q["id"]),
                      flush=True)
        return api(q)
    t1 = time.time()
    for i, q in enumerate(p["queries"]):
        try:
            mode = "plain"
            if q.get("text_any"):
                mode = "gene_text"
                rows = _filtered(q, _text_search)
                if not rows and q.get("fallback_plain"):
                    mode = "plain_fallback"
            if mode != "gene_text":
                if q.get("cancer"):
                    _q = {k: v for k, v in q.items() if k != "text_any"}
                    rows = _filtered(_q, _api_plain)
                else:   # unfiltered: the API scores without a payload loop
                    rows = _api_plain(q)
            res["results"][q["id"]] = {"error": None, "mode": mode, "hits": rows}
        except Exception as e:
            res["results"][q["id"]] = {"error": ("%s: %s" % (type(e).__name__, e))[:300], "hits": []}
            print("CARD_RETRIEVER search error %s: %s" % (q["id"], str(e)[:120]), flush=True)
        if (i + 1) % 20 == 0 or i + 1 == len(p["queries"]):
            print("CARD_RETRIEVER %d/%d searches (%.0fs)" % (i + 1, len(p["queries"]), time.time() - t1), flush=True)
    res["search_s"] = round(time.time() - t1, 1)
    res["ok"] = True
except Exception as e:
    res["error"] = ("%s: %s" % (type(e).__name__, e))[:500]
    print("CARD_RETRIEVER failed: " + res["error"], flush=True)
_tmp = p["out"] + ".tmp"
with open(_tmp, "w") as fh:
    json.dump(res, fh)
os.replace(_tmp, p["out"])
try:
    if rt is not None:
        rt.client.close()
except Exception:
    pass
print("CARD_RETRIEVER_DONE", flush=True)
'''


def _run_batch_retrieval(queries, venv_python=_CARD_VENV_PYTHON, device="cpu",
                         timeout=None, threads=None):
    """Run `queries` ([{id, query, top_k, cancer}]) in ONE paper_qa process.
    Returns {"ok", "error", "init_s", "search_s", "results": {id: {"hits",
    "error"}}}; never raises. Progress / lock-wait lines go to stderr (the run
    log). The timeout (default: 20 min lock wait + init + 20 s per search) is a
    Popen.wait timeout, so the mrag pause-aware timeout patch applies. `threads`
    (optional) caps torch / BLAS threads in the worker; default = as retrieve_cli
    and the curated scoring (no cap)."""
    import collections, shutil, subprocess, sys, tempfile, threading
    if not queries:
        return {"ok": True, "error": None, "init_s": 0.0, "search_s": 0.0, "results": {}}
    wd = tempfile.mkdtemp(prefix="mrag_cardret_")
    t0 = time.time()
    try:
        runner = Path(wd) / "card_retrieve_runner.py"
        runner.write_text(_BATCH_RETRIEVE_RUNNER)
        params = Path(wd) / "params.json"
        out = Path(wd) / "hits.json"
        params.write_text(json.dumps({"queries": queries, "device": device, "threads": threads,
                                      "out": str(out), "paperqa_lib": _PAPERQA_LIB}))
        env = os.environ.copy()
        env["PYTHONPATH"] = _PAPERQA_LIB
        if threads:
            for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                env[_v] = str(threads)
        env["TOKENIZERS_PARALLELISM"] = "false"
        if timeout is None:
            timeout = 1500.0 + 20.0 * len(queries)
            # 2026-09-25 (cardtimeout): optional env override for an overloaded
            # host, where the store load alone can take 13+ min and a CPU BGE-M3
            # search 30+ s. It can only LENGTHEN the default; unset = unchanged.
            _tov = os.environ.get('MRAG_CARD_RETRIEVER_TIMEOUT_S', '').strip()
            if _tov:
                try:
                    timeout = max(timeout, float(_tov))
                except ValueError:
                    pass
        tail = collections.deque(maxlen=12)

        def _pump(proc):
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                tail.append(line)
                if line.startswith(("CARD_RETRIEVER", "VDB locked")):
                    print(f"[narrative] card retrieval: {line}", file=sys.stderr, flush=True)
        timed_out = False
        with _RETRIEVAL_LOCK:
            proc = subprocess.Popen([venv_python, str(runner), str(params)],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, env=env)
            th = threading.Thread(target=_pump, args=(proc,), daemon=True)
            th.start()
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                rc = proc.wait()
        th.join(timeout=10)
        if timed_out or not out.exists():
            msg = (f"card retriever {'timed out after %.0f s' % timeout if timed_out else 'exit code %s' % rc}"
                   f": {' | '.join(list(tail)[-3:])[-300:]}")
            _paperqa_warn(msg)
            return {"ok": False, "error": msg, "results": {}}
        res = json.loads(out.read_text())
        if not res.get("ok"):
            _paperqa_warn(f"card retriever failed: {res.get('error')}")
        return res
    except Exception as e:
        _paperqa_warn(f"card retriever {type(e).__name__}: {e}")
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "results": {}}
    finally:
        _CARD_RETRIEVAL_STATS.update({"n_queries": len(queries),
                                      "wall_s": round(time.time() - t0, 1)})
        shutil.rmtree(wd, ignore_errors=True)


def _symbol_spellings(gene):
    """Case variants of a gene symbol for the chunk-text prefilter: 'KRT5', plus
    'Krt5' (mouse style) for 4+ characters and 'krt5' for 5+ characters only
    (short variants such as 'Ar' / 'erg' are substrings of ordinary words)."""
    g = str(gene or "").strip()
    v = [g]
    if len(g) >= 4:
        v.append(g[:1].upper() + g[1:].lower())
    if len(g) >= 5:
        v.append(g.lower())
    return " ".join(dict.fromkeys(x for x in v if x and " " not in x))


def batch_retrieve_card_hits(genes, domains, venv_python=_CARD_VENV_PYTHON,
                             device="cpu", cross_top_k=3, domain_top_k=8,
                             timeout=None, cross=True, extra_slots=None,
                             prefilter=True, query_tmpl=None):
    """All Section-14 card retrieval for a batch of genes in ONE process: per
    gene the unfiltered cross-domain query of the legacy path ('{gene} gene
    function cancer', top 3) plus one corpus-filtered query per domain
    ('{gene} {domain name}', cancer=<key>, top `domain_top_k`) restricted to
    chunks whose text contains the gene symbol (_symbol_spellings); only when
    that finds nothing does the plain corpus-filtered query run instead (title-
    only or other-case mentions). A plain dense query for a bare gene symbol
    misses most gene-naming passages (probe: KRT5 in the prostate corpus, 0/8
    plain vs 6 papers with the text prefilter), which would turn into false
    'not reported' claims.

    Returns {gene: {"cross": [hit], <key>: [hit], "_errors": {slot: msg}}}; a
    hit is {pmid, title, snippet, gate_text, score, year, cancer}. Never raises:
    a failed batch marks every slot of every gene as an error.

    2026-09-23 -- the curated target scorer retrieves through THIS function too
    (one retriever, one prefilter, one fast path), so four knobs exist for it
    and every one of them defaults to the Section-14 behaviour:
      cross=False        drop the unfiltered "{gene} gene function cancer" slot
      extra_slots=[...]  further corpus-filtered slots, each
                         {"key", "cancer", "query" (may contain "{gene}"),
                          "top_k", "prefilter"} -- the scorer uses this for its
                         pan-cancer slot, which is a corpus and not a domain
      prefilter=False    plain corpus-filtered queries (no chunk-text mask)
      query_tmpl="..."   override the per-domain query; "{gene}" and "{domain}"
                         are substituted (default "{gene} {domain}")."""
    doms = _norm_domains(domains) or []
    genes = list(dict.fromkeys(str(g) for g in (genes or []) if g))
    extra = []
    for s in (extra_slots or []):
        k = str((s or {}).get("key") or "").strip()
        if not k or k in {d["key"] for d in doms} or k == "cross":
            continue
        extra.append({"key": k, "cancer": s.get("cancer") or k,
                      "query": str(s.get("query") or "{gene}"),
                      "top_k": int(s.get("top_k") or domain_top_k),
                      "prefilter": bool(s.get("prefilter", prefilter))})
    tmpl = str(query_tmpl or "{gene} {domain}")
    queries = []
    for gi, g in enumerate(genes):
        if cross:
            queries.append({"id": f"{gi}|cross", "query": f"{g} gene function cancer",
                            "top_k": int(cross_top_k), "cancer": None})
        for d in doms:
            q = {"id": f"{gi}|{d['key']}",
                 "query": tmpl.format(gene=g, domain=d["label"].lower()),
                 "top_k": int(domain_top_k), "cancer": d["key"]}
            if prefilter:
                q["text_any"] = _symbol_spellings(g)
                q["fallback_plain"] = True
            queries.append(q)
        for s in extra:
            q = {"id": f"{gi}|{s['key']}", "query": s["query"].format(gene=g),
                 "top_k": s["top_k"], "cancer": s["cancer"]}
            if s["prefilter"]:
                q["text_any"] = _symbol_spellings(g)
                q["fallback_plain"] = True
            queries.append(q)
    res = _run_batch_retrieval(queries, venv_python=venv_python, device=device,
                               timeout=timeout)
    _CARD_RETRIEVAL_STATS.update({"init_s": res.get("init_s"), "search_s": res.get("search_s"),
                                  "ok": bool(res.get("ok")), "n_genes": len(genes),
                                  "n_domains": len(doms), "n_extra": len(extra),
                                  "cross": bool(cross), "prefilter": bool(prefilter)})
    results = res.get("results") or {}
    batch_err = None if res.get("ok") else (res.get("error") or "retrieval failed")
    slots = (["cross"] if cross else []) + [d["key"] for d in doms] + [s["key"] for s in extra]
    out = {}
    for gi, g in enumerate(genes):
        rec = {"_errors": {}}
        for slot in slots:
            r = results.get(f"{gi}|{slot}")
            if r is None:
                rec[slot] = []
                rec["_errors"][slot] = batch_err or "no retrieval result"
            else:
                rec[slot] = list(r.get("hits") or [])
                if r.get("error"):
                    rec["_errors"][slot] = str(r["error"])
                if r.get("mode"):
                    rec.setdefault("_modes", {})[slot] = r["mode"]
        out[g] = rec
    return out


def _parse_novelty_by_domain(obj, doms):
    """LLM JSON -> {domain key: text}. novelty_by_domain may be a dict keyed by
    domain key / abbreviation / name, or a list ({key, text} dicts, or strings in
    domain order). A model that returned only the legacy "novelty" string has it
    applied to the FIRST domain (graceful degradation)."""
    if not isinstance(obj, dict) or not doms:
        return {}
    alias = {}
    for d in doms:
        for a in (d["key"], d["abbr"], d["label"], d["display"], d["inline"]):
            if a:
                alias.setdefault(str(a).strip().lower(), d["key"])
    out = {}
    nb = obj.get("novelty_by_domain")
    if isinstance(nb, dict):
        for k, v in nb.items():
            kk = alias.get(str(k).strip().lower())
            if kk and kk not in out and isinstance(v, str):
                out[kk] = v.strip()
    elif isinstance(nb, list):
        for i, v in enumerate(nb):
            if isinstance(v, dict):
                kk = alias.get(str(v.get("key") or v.get("domain") or "").strip().lower())
                txt = v.get("text") or v.get("novelty") or ""
            else:
                kk = doms[i]["key"] if i < len(doms) else None
                txt = v
            if kk and kk not in out and isinstance(txt, str):
                out[kk] = txt.strip()
    elif nb is None and isinstance(obj.get("novelty"), str):
        out[doms[0]["key"]] = obj["novelty"].strip()
    return out


def _auto_ortholog(organism, genes):
    """{gene: {"human","searched","disclosure"}} built on the spot.

    Returns {} for a human organism (the identity map), and {} when the MGI
    table is not installed -- in which case the caller behaves exactly as it
    did before the ortholog route existed. The orchestrator refuses a non-human
    run without the table up front, so an empty map here cannot reach a report
    unnoticed.
    """
    try:
        import sys as _sys
        _src = os.environ.get("MRAG_ORCHESTRATOR_SRC",
                              "/BIO1/MRAG/Agent/orchestrator/src")
        if _src not in _sys.path:
            _sys.path.insert(0, _src)
        from mrag_orchestrator.orthology import get_map as _gm
        m = _gm(str(organism))
        if m.identity or not len(m):
            return {}
        out = {}
        for g in genes:
            h = m.to_human(g)
            out[str(g)] = {"human": h.human_symbol or "",
                           "searched": h.mapped,
                           "disclosure": h.disclosure()}
        return out
    except Exception:
        return {}


def load_ortholog(path):
    """ortholog.json -> {run_symbol: {"human", "searched", "disclosure"}}.

    Returns {} for a human run, a missing file, or anything unreadable: the
    empty map IS the human code path, so a broken file degrades to "no
    translation" rather than to a wrong translation. A non-human run whose map
    is empty simply searches its own symbols, which is what happened before
    this existed -- but the runner refuses that case up front (the organism
    profile's `curated_literature` capability requires the ortholog table), so
    it cannot reach a report unnoticed.
    """
    if not path:
        return {}
    try:
        d = json.loads(Path(path).read_text())
    except Exception:
        return {}
    if not isinstance(d, dict) or d.get("identity"):
        return {}
    out = {}
    for g, e in (d.get("top_final") or {}).items():
        if not isinstance(e, dict):
            continue
        out[str(g)] = {"human": e.get("lit_symbol") or "",
                       "searched": bool(e.get("lit_searched", True)),
                       "disclosure": e.get("disclosure") or ""}
    # genes that were scored but not in top_final still deserve their mapping
    for g, txt in (d.get("disclosures") or {}).items():
        out.setdefault(str(g), {"human": "", "searched": True,
                                "disclosure": txt})
        if not out[str(g)].get("disclosure"):
            out[str(g)]["disclosure"] = txt
    for g in (d.get("unmapped") or []):
        out.setdefault(str(g), {"human": "", "searched": False,
                                "disclosure": ""})
        out[str(g)]["searched"] = False
    return out


def ortholog_banner(path):
    """The one-paragraph §14 disclosure for this run ('' for a human run)."""
    if not path:
        return ""
    try:
        d = json.loads(Path(path).read_text())
    except Exception:
        return ""
    if not isinstance(d, dict) or d.get("identity"):
        return ""
    return str(d.get("banner") or "")


def _build_notable_findings(tfs, targets, field, retriever_script,
                            endpoint, model, max_cards=15,
                            min_shared_tfs=2, max_ground=None,
                            rank_order=None, domains=None,
                            prefetched_hits=None, retrieve_device="cpu",
                            ortholog=None, organism="", evidence=None):
    """Deterministically select up to `max_cards` candidate target genes driven
    ENTIRELY by OBSERVED data: targets regulated by >= min_shared_tfs top TFs in
    the TRN, ranked by (number of TRN regulators) x |observed RNA log2FC|.
    Each card is grounded with GeneGPT GeneRIFs (PMID, NCBI) + paperQA passages.

    Returns (cards, passages). Each card is a dict rendered as a §15 card.
    Novelty is asserted ONLY when BOTH the GeneRIF DB and paperQA give no
    sample-domain ({field}) evidence, then passed through the 3-layer guard
    (_soften_novelty + PMID whitelist + never-when-field-evidence).

    ortholog (2026-09-23, Task B): {run_symbol: {"human": str, "searched": bool,
    "disclosure": str}} for a non-human run. Every retrieval below -- GeneRIF,
    PaperQA, the live PubMed check -- and every sentence it produces is about
    the HUMAN ortholog, because that is what the indexes contain; the card
    keeps the run's own symbol as its title and carries the disclosure naming
    the ortholog. A gene with searched=False has NO human ortholog: nothing is
    retrieved for it, and instead of vanishing (which is what a no-evidence
    gene used to do) it gets a card that says so. None / an empty map = the
    human path, unchanged.

    domains (2026-09-17, llmdomain): the run's literature domains in the user's
    order (mrag_orchestrator.literature.domain_specs dicts). With domains the
    Novelty field becomes card["novelty_by_domain"] = one entry per domain,
    {key, label, abbr, display, status: reported|unreported|no_evidence|error,
    text, html, refs, n_hits, n_named, n_retrieved}, each judged ONLY from that
    domain's corpus-filtered passages (gene named + domain terms) and the GeneRIFs
    whose text/title names the domain; PMIDs are whitelisted per domain.
    'reported', 'no_evidence' and 'error' sentences are deterministic;
    'unreported' needs no passage of that corpus naming the gene, the LLM's
    per-domain flag AND deterministic cross-domain evidence. One LLM call
    per card. All retrieval of the batch runs in ONE retriever process
    (batch_retrieve_card_hits) unless `prefetched_hits` ({gene: {"cross": [...],
    <key>: [...], "_errors": {...}}}) is supplied. card["novelty_html"] = the
    first domain's html (legacy templates). domains=None: legacy behaviour,
    unchanged.

    evidence (followup 2026-09-28, fix 2): card_evidence_context() of the curated
    rows -- the cards then follow the gate's evidence rules (see the block above
    card_evidence_context); None, or the evidence layers off, = today's cards."""
    import collections
    # --- OBSERVED candidate source: TRN regulators + differential expression ---
    # Selection is driven ENTIRELY by observed data (how many top TFs regulate
    # the target in the TRN, and the target's observed RNA log2FC / Z).
    tgt_fc, tgt_z, trn_reg = {}, {}, collections.defaultdict(set)
    for _tf, lst in (targets or {}).items():
        for x in (lst or []):
            g = x.get("Gene Name")
            if g is None:
                continue
            trn_reg[g].add(_tf)
            fc = x.get("RNA_log2FC_mean")
            if g not in tgt_fc or abs(fc or 0.0) > abs(tgt_fc[g] or 0.0):
                tgt_fc[g] = fc
            tgt_z[g] = max(tgt_z.get(g, 0.0), x.get("Z_mean") or 0.0)
    if not trn_reg:
        return [], []

    # --- candidate scoring -------------------------------------------------
    # `rank_order` (the curated rerank: evidence score desc, log2FC desc) is
    # authoritative when supplied -- it is the same ordering that decided which
    # genes are targets at all, so Section 14 stays consistent with Section 10
    # instead of re-ranking on a second, unrelated axis. Without it, fall back
    # to the observed-data score: TRN regulator count x |observed log2FC|.
    _rank_idx = {str(g).strip(): i for i, g in enumerate(rank_order or [])}
    scored = []
    for tgt, regs in trn_reg.items():
        lf = tgt_fc.get(tgt) or 0.0
        obs_score = len(regs) * abs(lf)
        scored.append((obs_score, tgt, sorted(regs), lf))
    # Candidate universe = targets regulated by >= min_shared_tfs top TFs
    # (OBSERVED TRN topology). Fall back to single-TF targets if none qualify.
    shared = [x for x in scored if len(x[2]) >= min_shared_tfs]
    scored = shared if shared else scored
    if _rank_idx:
        scored.sort(key=lambda x: _rank_idx.get(x[1], 10**6))
    else:
        scored.sort(key=lambda x: -x[0])
    # Ground at least as many candidates as we intend to show.
    _ground_n = max_ground if max_ground is not None else max(12, max_cards)
    scored = scored[:_ground_n]
    if not scored:
        return [], []

    _fterms = _field_terms(field)
    def _mentions_field(txt):
        t = (txt or "").lower()
        return any(ft in t for ft in _fterms)

    def _prompt_gene(run_sym, lit_sym):
        """What the LLM is told the gene is called. When the two differ the
        model must know the passages are about the human ortholog, or it will
        write mouse-specific claims out of human papers."""
        return (str(run_sym) if str(run_sym) == str(lit_sym)
                else "%s (mouse; all passages below are about its human "
                     "ortholog %s)" % (run_sym, lit_sym))

    # --- human-ortholog route (2026-09-23 Task B) --------------------------
    _ORTH = {str(k): (v or {}) for k, v in (ortholog or {}).items()}
    if not _ORTH and organism:
        # No explicit map was handed in (a run without curated targets, or a
        # re-render of a branch that has no ortholog.json). Build one for the
        # candidates from the same MGI table the rest of the pipeline uses --
        # never fall through to searching mouse symbols in a human index.
        _ORTH = _auto_ortholog(organism, [x[1] for x in scored])
    def _lit_sym(g):
        """The symbol the LITERATURE is indexed under for this gene."""
        return str((_ORTH.get(str(g)) or {}).get("human") or g)
    def _lit_ok(g):
        """False only when the gene has NO human ortholog: nothing to search."""
        e = _ORTH.get(str(g))
        return True if e is None else bool(e.get("searched", True))

    # --- per-domain Novelty (2026-09-17, llmdomain): ONE retriever process for
    #     the whole batch, unless the caller prefetched the hits ---
    _doms = _norm_domains(domains)
    # followup 2026-09-28 (fix 2): the curated rows' evidence rules, with the gate's own domain
    # specs (their evfilter rules); None -> today's cards
    _EVX = None
    if evidence and _doms is not None and _evidence_defence_on():
        _eg0 = _evidence_gate_mod()
        _EVX = dict(evidence)
        _EVX["specs"] = ({sp["key"]: sp for sp in _eg0.domain_specs(domains)} if _eg0 is not None else {})
    _prefetch = prefetched_hits
    _t_nf0 = time.time()
    if _doms is not None:
        _pubmed_reset_stats()   # per-build counters; the (gene, domain) cache is kept
    _t_ret, _t_llm = 0.0, 0.0
    if _doms is not None and _prefetch is None:
        if retriever_script and Path(retriever_script).exists():
            # The corpora are human literature, so the query symbol is the
            # human ortholog; the result is re-keyed back to this run's symbols
            # so the rest of the loop is unchanged.
            _q_syms = [_lit_sym(x[1]) for x in scored if _lit_ok(x[1])]
            _raw_pref = batch_retrieve_card_hits(_q_syms, _doms,
                                                 device=retrieve_device)
            _prefetch = {}
            for _x in scored:
                _g = _x[1]
                if not _lit_ok(_g):
                    _prefetch[_g] = {"_errors": {}}   # never searched, not failed
                else:
                    _prefetch[_g] = _raw_pref.get(_lit_sym(_g), {"_errors": {
                        s: "no retrieval result"
                        for s in ["cross"] + [d["key"] for d in _doms]}})
        else:
            _prefetch = {x[1]: {"_errors": {s: "PaperQA retriever not configured"
                                            for s in ["cross"] + [d["key"] for d in _doms]}}
                         for x in scored}
        _t_ret = time.time() - _t_nf0

    cards, passages, seen = [], [], set()
    for _score, _gene_run, tfs_for, lf in scored:
        # `_gene_run` is the symbol THIS RUN's data uses (mouse Trp53); `tgt` is
        # the symbol the LITERATURE is under (human TP53). They are the same
        # string on a human run, so everything below is unchanged there. Every
        # sentence built from `tgt` describes human literature and is disclosed
        # as such; only the card title and the TRN lookups use the run symbol.
        tgt = _lit_sym(_gene_run)
        _orth_entry = _ORTH.get(str(_gene_run)) or {}
        _searchable = _lit_ok(_gene_run)
        # -- GeneGPT GeneRIF grounding (deterministic, PMID-carrying) --
        # ADD-2 (2026-09-18): genegpt.gene_generifs issues ONE efetch of the whole
        # gene record whatever retmax is -- retmax only caps the parse -- so the
        # large cap is free on the wire. The full set feeds the per-domain gate;
        # the newest CARD_GENERIF_RECENT still feed the generic card evidence.
        # `rifs` equals the old retmax=8 list, except that when the record holds
        # duplicate RIF texts the newer cap back-fills the slots the old one lost
        # to de-duplication (strictly more evidence, same order).
        _rmx = CARD_GENERIF_RETMAX if _doms is not None else CARD_GENERIF_RECENT
        # followup 2026-09-28 (fix 2): this gene's evidence rules (None -> today's card)
        _cx = _card_rules(_EVX, _gene_run, _rmx) if (_EVX is not None and _searchable) else None
        _card_flags = []
        if _cx is not None and _cx.get("generif") is not None:
            rifs_all = _cx["generif"]          # the lines the scorer saw (per-run GeneRIF record)
            _gfst = {}
        else:
            _GENERIF_STATUS.pop((str(tgt), int(_rmx)), None)
            rifs_all = (_gene_generifs_safe(tgt, retmax=_rmx) if _searchable else [])
            _gfst = (_GENERIF_STATUS.get((str(tgt), int(_rmx))) or {}) if _searchable else {}
        # followup 2026-09-28 (fix 1): a GeneRIF record that could not be read is marked on the
        # card and can never support a novelty claim (it used to be an unmarked [])
        _rif_err = (str(_gfst.get("error") or "fetch failed")[:200] if _gfst.get("status") == "failed" else "")
        _rifs_scan = rifs_all                  # the per-domain scan sees every line (filters decide)
        _xset = _cx["xset"] if _cx is not None else set()
        # review 2026-09-28 (finding 2): + the lines the gate filtered and counted for no domain
        _xrif = _cx["rif_uncounted"] if _cx is not None else set()
        if _xset or _xrif:
            rifs_all = [r for r in rifs_all if not ({str(p) for p in (r.get("pmids") or [])} & _xset)
                        and _rif_item_key(r.get("text"), r.get("pmids")) not in _xrif]
        rifs = rifs_all[:CARD_GENERIF_RECENT]
        rif_pmids = list(dict.fromkeys(p for r in rifs for p in r.get("pmids", []) if p))[:6]
        rif_block = "\n".join(
            f"[PMID:{(r.get('pmids') or ['NA'])[0]}] {r.get('text', '')[:240]}"
            for r in rifs[:5])

        # -- paperQA cross-domain + field retrieval --
        cross_hits, field_hits = [], []
        dom_hits, dom_errs = {}, {}
        if _doms is not None:
            _ph = (_prefetch or {}).get(tgt)
            if not isinstance(_ph, dict):
                _ph = {"_errors": {s: "no retrieval result"
                                   for s in ["cross"] + [d["key"] for d in _doms]}}
            dom_errs = dict(_ph.get("_errors") or {})
            cross_hits = list(_ph.get("cross") or [])
            for _d in _doms:
                dom_hits[_d["key"]] = list(_ph.get(_d["key"]) or [])
        elif _prefetch is not None:
            _ph = _prefetch.get(tgt) or {}
            cross_hits = list(_ph.get("cross") or [])
            field_hits = list(_ph.get("field") or [])
        elif retriever_script and Path(retriever_script).exists() and _searchable:
            cross_hits = _query_paperqa(retriever_script, f"{tgt} gene function cancer", top_k=3)
            field_hits = _query_paperqa(retriever_script, f"{tgt} {field} cancer", top_k=3)
        _cross_all = cross_hits                # followup 2026-09-28 (fix 2): the domain scan judges these
        # review 2026-09-28 (finding 2): + the passages the gate filtered and counted for no domain
        _xhit = _cx["hit_uncounted"] if _cx is not None else set()
        if _xset or _xhit:
            cross_hits = [h for h in cross_hits if str(h.get("pmid") or "") not in _xset
                          and _hit_item_key(h) not in _xhit]
            field_hits = [h for h in field_hits if str(h.get("pmid") or "") not in _xset
                          and _hit_item_key(h) not in _xhit]
        _all_dom_hits = [h for _d in (_doms or []) for h in dom_hits.get(_d["key"], [])
                         if str(h.get("pmid") or "") not in _xset and not (_xhit and _hit_item_key(h) in _xhit)]
        for h in (cross_hits + field_hits + _all_dom_hits):
            pm = h.get("pmid", "")
            if pm and pm not in seen:
                seen.add(pm)
                passages.append({"pmid": pm, "title": h.get("title", ""),
                                 "snippet": h.get("snippet", "") or h.get("text", "")})
        for r in rifs[:6]:
            for pm in r.get("pmids", []):
                if pm and pm not in seen:
                    seen.add(pm)
                    passages.append({"pmid": pm, "title": "", "snippet": r.get("text", "")})
        cross_pmids = [h.get("pmid") for h in cross_hits if h.get("pmid")][:3]
        _gpat = re.compile(r"\b" + re.escape(tgt) + r"\b", re.I)
        def _hit_blob(h):
            return (h.get("title", "") or "") + " " + (h.get("snippet") or h.get("text") or "")
        # Require the passage to NAME THE GENE, not just the cancer field: paperQA
        # retrieves topically-near passages about a different gene in the same
        # cancer; citing those as evidence for this gene is a grounding
        # hallucination (e.g. a prostate target vs papers on other genes/compounds).
        field_pmids = [h.get("pmid") for h in field_hits
                       if h.get("pmid") and _gpat.search(_hit_blob(h))
                       and _mentions_field(_hit_blob(h))][:3]
        cross_lit = "\n".join(
            f"[PMID:{h.get('pmid')}] {h.get('title', '')} :: "
            f"{(h.get('snippet') or h.get('text') or '')[:240]}"
            for h in cross_hits if h.get("pmid"))
        # Title lookup covers every RIF in domain mode: the RIF one-liner often
        # omits the cancer word that its paper's title carries.
        _rif_src = _rifs_scan if _doms is not None else rifs
        _all_rif_pmids = [pm for r in _rif_src for pm in (r.get("pmids") or []) if pm]
        del _PMID_TITLE_ERRORS[:]
        _rif_titles = _pmid_titles(_all_rif_pmids)
        # followup 2026-09-28 (fix 1): titles that could not be read leave the domain scan incomplete
        _title_err = "; ".join(_PMID_TITLE_ERRORS)[:200]
        _rif_incomplete = ("GeneRIF record: " + _rif_err) if _rif_err else (
            ("GeneRIF paper titles: " + _title_err) if _title_err else "")
        if _rif_incomplete:
            _card_flags.append({"label": "GeneRIF fetch failed",
                                "note": "the NCBI GeneRIF record of %s could not be read completely for this card "
                                        "(%s); no novelty is claimed on it" % (tgt, _rif_incomplete)})
        def _rif_field(r):
            _b = (r.get("text", "") or "")
            for _pm in (r.get("pmids") or []):
                _b += " " + (_rif_titles.get(_pm, "") or "")
            return _mentions_field(_b)
        field_absent = (len(field_pmids) == 0) and not any(
            _rif_field(r) for r in rifs)

        if _doms is None:
            # -- per-card conservative synthesis (grounded, guarded) --
            cross_txt, nov_txt = "", ""
            pm_ok = set(rif_pmids) | set(cross_pmids) | set(field_pmids)
            sys_msg = {"role": "system", "content":
                f"You write ONE short 'Cross-domain evidence' sentence and (optionally) "
                f"ONE 'Novelty' sentence for a candidate target gene in a {field} cancer "
                "regulatory analysis. STRICT RULES:\n"
                "1. Use ONLY the provided GeneRIF and literature passages. No outside "
                "knowledge. Cite every claim inline as [PMID:xxxxxxx].\n"
                "2. Cross-domain: 1 sentence naming the cancer/tissue context the "
                "evidence actually reports (do NOT invent a domain).\n"
                f"3. Novelty: include a Novelty sentence ONLY if FIELD_ABSENT is true AND "
                f"cross-domain evidence exists in another context. The ONLY permitted "
                f"phrasing is: \"to our knowledge, this link has not been reported in the "
                f"indexed {field} literature.\" Otherwise leave Novelty empty.\n"
                "4. Dry, factual. No speculation. Output STRICT JSON: "
                '{"cross_domain": "...", "novelty": "..."}'}
            user_msg = {"role": "user", "content":
                f"TARGET GENE: {_prompt_gene(_gene_run, tgt)} ({field} cancer sample)\n"
                f"Regulated by (TRN top TFs): {', '.join(tfs_for)}\n"
                f"Observed RNA log2FC: {round(lf, 2)} ({'up' if lf > 0 else 'down'})\n"
                f"FIELD_ABSENT: {field_absent}\n\n"
                f"GeneRIF passages (NCBI):\n{rif_block or '(none)'}\n\n"
                f"paperQA cross-domain passages:\n{cross_lit or '(none)'}\n\n"
                'Return STRICT JSON now: {"cross_domain": "...", "novelty": "..."}'}
            try:
                raw = _glm_chat([sys_msg, user_msg], endpoint, model,
                                temperature=0.1, max_tokens=320, timeout=120.0).strip()
                m = re.search(r"\{.*\}", raw, re.S)
                obj = json.loads(m.group(0)) if m else {}
                cross_txt = (obj.get("cross_domain") or "").strip()
                nov_txt = (obj.get("novelty") or "").strip()
            except Exception:
                cross_txt, nov_txt = "", ""
            if not cross_txt and rifs:
                cross_txt = (rifs[0].get("text", "")[:220] +
                             (f" [PMID:{rif_pmids[0]}]" if rif_pmids else ""))

            def _keep(mm):
                return mm.group(0) if mm.group(1) in pm_ok else ""
            cross_txt = re.sub(r"\[PMID:([0-9]+)\]", _keep, cross_txt)
            nov_txt = re.sub(r"\[PMID:([0-9]+)\]", _keep, nov_txt)
            nov_txt = _soften_novelty(nov_txt, field) if nov_txt else ""
            if not field_absent:
                nov_txt = ""   # never claim novelty when field evidence exists
            if _rif_incomplete:
                nov_txt = ""   # followup 2026-09-28 (fix 1): nor on a GeneRIF check that did not complete
            # Ground the not-novel status with field-literature PMIDs so the card
            # never asserts "consistent with literature" without evidence.
            field_support = list(field_pmids)
            for _r in rifs:
                if _rif_field(_r):
                    for _pm in _r.get("pmids", []):
                        if _pm and _pm not in field_support:
                            field_support.append(_pm)
            field_support = field_support[:3]
            if field_absent and nov_txt:
                novelty_html = _pmid_links(nov_txt)
            elif field_support:
                novelty_html = _pmid_links(
                    f"Already reported in the indexed {field} literature "
                    + " ".join(f"[PMID:{p}]" for p in field_support)
                    + "; not flagged as novel.")
            else:
                novelty_html = ""
            nov_entries = None
        else:
            # ---- per-domain Novelty (llmdomain) ----
            def _gate_blob(h):
                return ((h.get("title", "") or "") + " " +
                        (h.get("gate_text") or h.get("snippet") or h.get("text") or ""))

            def _labels(h):
                c = h.get("cancer")
                return {str(x).strip().lower()
                        for x in (c if isinstance(c, (list, tuple)) else [c]) if x}
            dom_ev, dom_support, dom_nret, dom_named, dom_rif = {}, {}, {}, {}, {}
            _tgt_nm = _gene_fullname_offline(tgt)
            _xk = {}                           # followup 2026-09-28 (fix 2): {key: [(kind, why)]}
            for _d in _doms:
                _k = _d["key"]
                _gsp = (_EVX["specs"].get(_k) if _cx is not None else None)
                # the domain's own corpus-filtered hits, plus cross-domain hits whose
                # payload is labelled with this corpus (same corpus, free recall)
                _ev, _seen_pm, _named = [], set(), 0
                for h in dom_hits.get(_k, []) + [h for h in _cross_all if _k in _labels(h)]:
                    _pm = str(h.get("pmid") or "")
                    if not _pm or _pm in _seen_pm:
                        continue
                    _seen_pm.add(_pm)
                    _b = _gate_blob(h)
                    # names the gene AND this domain (not just any cancer).
                    # E1b (2026-09-25): a mention the passage ties to ANOTHER
                    # gene ("chromogranin A (CGA)") does not name this one.
                    if _gpat.search(_b) and not _tied_to_other_gene(_b, tgt, _tgt_nm):
                        if _cx is not None and _domain_mentions(_b, _d["terms"], _d["exclude"]):
                            _why = _card_exclusion(_cx, _gsp, tgt, _tgt_nm, hit=h, key=_k)
                            if _why:           # followup 2026-09-28 (fix 2): not counted, never cited
                                _xk.setdefault(_k, []).append(("passage", _why))
                                continue
                        _named += 1
                        if _domain_mentions(_b, _d["terms"], _d["exclude"]):
                            _ev.append(h)
                dom_ev[_k] = _ev
                dom_named[_k] = _named
                dom_nret[_k] = len(dom_hits.get(_k, []))
                _sup = [str(h.get("pmid")) for h in _ev][:3]
                # ADD-2: scan EVERY GeneRIF, not only the newest 8. GeneRIFs are
                # gene-specific by construction, so a domain-term match here is
                # evidence of the same quality as a corpus passage and feeds the
                # per-domain status the same way.
                _nrif = 0
                for _r in _rifs_scan:
                    _b = (_r.get("text", "") or "")
                    # E1b (2026-09-25): a GeneRIF line about the gene that uses
                    # this symbol as an alias (a CGA record line on serum
                    # chromogranin A in prostate cancer) is not this gene's.
                    if _tied_to_other_gene(_b, tgt, _tgt_nm, generif=True):
                        continue
                    for _pm in (_r.get("pmids") or []):
                        _b += " " + (_rif_titles.get(_pm, "") or "")
                    if _domain_mentions(_b, _d["terms"], _d["exclude"]):
                        if _cx is not None:
                            _why = _card_exclusion(_cx, _gsp, tgt, _tgt_nm, rif=_r, key=_k)
                            if _why:           # followup 2026-09-28 (fix 2): not counted, never cited
                                _xk.setdefault(_k, []).append(("generif", _why))
                                continue
                        _nrif += 1
                        for _pm in (_r.get("pmids") or []):
                            if _pm and _pm not in _sup:
                                _sup.append(_pm)
                dom_rif[_k] = _nrif
                dom_support[_k] = _sup[:3]
            # deterministic "cross-domain evidence exists in another context"
            _cross_ev = (any((r.get("text") or "").strip() for r in rifs)
                         or any(_gpat.search(_hit_blob(h)) for h in cross_hits)
                         or any(dom_support.values()))

            _dom_blocks = []
            for _d in _doms:
                _k = _d["key"]
                _lines = [f"[{_k}] FIELD_ABSENT: {not dom_support[_k]}"]
                if dom_ev[_k]:
                    for h in dom_ev[_k][:3]:
                        _lines.append(f"  [PMID:{h.get('pmid')}] {h.get('title', '')} :: "
                                      f"{(h.get('snippet') or h.get('text') or '')[:200]}")
                elif dom_support[_k]:
                    _lines.append("  (GeneRIF naming this domain: "
                                  + " ".join(f"[PMID:{p}]" for p in dom_support[_k]) + ")")
                elif dom_errs.get(_k):
                    _lines.append("  (retrieval from this corpus failed)")
                else:
                    _lines.append(f"  (no passage from this corpus names {tgt})")
                _dom_blocks.append("\n".join(_lines))
            _json_eg = ('{"cross_domain": "...", "novelty_by_domain": {'
                        + ", ".join(f'"{d["key"]}": "..."' for d in _doms) + "}}")
            cross_txt = ""
            nb_raw = {}
            sys_msg = {"role": "system", "content":
                f"You write ONE short 'Cross-domain evidence' sentence and, for each "
                f"literature domain, (optionally) ONE 'Novelty' sentence for a candidate "
                f"target gene in a {field} cancer regulatory analysis. STRICT RULES:\n"
                "1. Use ONLY the provided GeneRIF and literature passages. No outside "
                "knowledge. Cite every claim inline as [PMID:xxxxxxx], one PMID per "
                "bracket.\n"
                "2. Cross-domain: 1 sentence naming the cancer/tissue context the "
                "evidence actually reports (do NOT invent a domain).\n"
                "3. Novelty, separately for EACH domain key listed under DOMAINS and "
                "judged ONLY from that domain's block: write a sentence ONLY if that "
                "domain's FIELD_ABSENT is true AND the GeneRIF / cross-domain passages "
                "show evidence for this gene in another context. The ONLY permitted "
                "phrasing is: \"to our knowledge, this link has not been reported in the "
                "indexed <domain name> literature.\" Otherwise return \"\" for that "
                "domain. Return \"\" for every domain whose FIELD_ABSENT is false. Never "
                "put a PMID in a Novelty sentence and never move a PMID from one "
                "domain's block to another.\n"
                "4. Dry, factual. No speculation. Output STRICT JSON: " + _json_eg}
            user_msg = {"role": "user", "content":
                f"TARGET GENE: {_prompt_gene(_gene_run, tgt)} ({field} cancer sample)\n"
                f"Regulated by (TRN top TFs): {', '.join(tfs_for)}\n"
                f"Observed RNA log2FC: {round(lf, 2)} ({'up' if lf > 0 else 'down'})\n"
                f"DOMAINS (in order): "
                + "; ".join(f"{d['key']} = {d['inline']}" for d in _doms) + "\n\n"
                + "Per-domain literature (each block is ONLY that domain's corpus):\n"
                + "\n".join(_dom_blocks) + "\n\n"
                f"GeneRIF passages (NCBI):\n{rif_block or '(none)'}\n\n"
                f"paperQA cross-domain passages:\n{cross_lit or '(none)'}\n\n"
                "Return STRICT JSON now: " + _json_eg}
            _t_l = time.time()
            try:
                raw = _glm_chat([sys_msg, user_msg], endpoint, model,
                                temperature=0.1, max_tokens=320 + 80 * len(_doms),
                                timeout=120.0).strip()
                m = re.search(r"\{.*\}", raw, re.S)
                obj = json.loads(m.group(0)) if m else {}
                cross_txt = str(obj.get("cross_domain") or "").strip() if isinstance(obj, dict) else ""
                nb_raw = _parse_novelty_by_domain(obj, _doms)
            except Exception:
                cross_txt, nb_raw = "", {}
            _t_llm += time.time() - _t_l
            if not cross_txt and rifs:
                cross_txt = (rifs[0].get("text", "")[:220] +
                             (f" [PMID:{rif_pmids[0]}]" if rif_pmids else ""))
            pm_ok = (set(rif_pmids) | set(cross_pmids)
                     | {str(h.get("pmid")) for _k in dom_ev for h in dom_ev[_k]})

            def _keep(mm):
                return mm.group(0) if mm.group(1) in pm_ok else ""
            cross_txt = re.sub(r"\[PMID:([0-9]+)\]", _keep, _split_pmid_groups(cross_txt))
            cross_txt = re.sub(r"\s+([.,;])", r"\1", re.sub(r"[ \t]{2,}", " ", cross_txt)).strip()

            nov_entries, _earlier = [], {}
            for _d in _doms:
                if not _searchable:
                    # No human ortholog: nothing was retrieved and nothing may
                    # be asserted. Skipping here also stops a pointless live
                    # PubMed query for a symbol that exists only in mouse; the
                    # explicit no_ortholog entries are built after this block.
                    continue
                _k, _inl = _d["key"], _d["inline"]
                _sup = list(dom_support[_k])
                _ok_d = set(_sup)          # per-domain whitelist: no PMID crosses domains
                _t = re.sub(r"\[PMID:([0-9]+)\]",
                            lambda mm: mm.group(0) if mm.group(1) in _ok_d else "",
                            _split_pmid_groups(nb_raw.get(_k, "") or ""))
                _t = _soften_novelty(_t, _inl) if _t.strip() else ""
                _pmres = None
                _refs_source = ""
                # followup 2026-09-28 (fix 2): a gene admitted ONLY on verified live-PubMed rescue
                # records cites exactly those; items of this domain the filters excluded
                _resc = (_cx.get("rescue") or {}).get(_k) if _cx is not None else None
                _xw = (_xk.get(_k) or []) if _cx is not None else []
                if _resc:
                    _status = "corpus_absent"
                    _pmres = dict(_resc)
                    _sup = list(_resc["verified"])
                    _refs_source = "pubmed"
                    _text = _corpus_absent_text(tgt, _inl, _pmres)
                elif _sup:
                    _status = "reported"

                    def _cite_runs(_pms, _how="indexed under"):
                        """PMIDs -> citation string. A paper already listed under an
                        earlier domain is the same report, not independent evidence:
                        mark it (consecutive PMIDs sharing the earlier domain are
                        marked once). `_how` is "indexed under" for a corpus passage
                        and "cited for" for a GeneRIF PMID, which is evidence for the
                        domain but says nothing about corpus membership."""
                        _parts, _grp, _grp_lab = [], [], None

                        def _emit(_g, _lab):
                            if _g:
                                _parts.append(" ".join(f"[PMID:{x}]" for x in _g) + (
                                    f" (same report{'s' if len(_g) > 1 else ''}, also "
                                    f"{_how} {_lab})" if _lab else ""))
                        for _pm in _pms:
                            _lab = _earlier.get(_pm)
                            if _grp and _lab == _grp_lab:
                                _grp.append(_pm)
                                continue
                            _emit(_grp, _grp_lab)
                            _grp, _grp_lab = [_pm], _lab
                        _emit(_grp, _grp_lab)
                        return " ".join(_parts)
                    # FIX 2026-09-19: dom_support mixes corpus-passage PMIDs with
                    # GeneRIF PMIDs, and the one sentence below used to present both
                    # as "the indexed <domain> literature". A GeneRIF PMID is usually
                    # in NO indexed corpus (checked: of the 5 GeneRIF-only entries in
                    # the PRAD refresh, none of their 7 PMIDs is in parsed/*.jsonl),
                    # and ADD-2 -- scanning all 400 GeneRIFs instead of the newest 8 --
                    # is what made that case common. The status is unchanged; only the
                    # sentence now names the source it actually has.
                    _ev_pm = {str(h.get("pmid")) for h in dom_ev[_k]}
                    _corpus_pm = [p for p in _sup if p in _ev_pm]
                    _rif_pm = [p for p in _sup if p not in _ev_pm]
                    _refs_source = ("corpus" if not _rif_pm
                                    else "generif" if not _corpus_pm
                                    else "corpus+generif")
                    _rif_w = f"GeneRIF{'s' if len(_rif_pm) > 1 else ''}"
                    if not _corpus_pm and any(_kd == "passage" for _kd, _w in _xw):
                        # followup 2026-09-28 (fix 2): the corpus passage(s) naming both were excluded
                        _text = (f"Reported for {tgt} in this domain by curated NCBI "
                                 f"{_rif_w} " + _cite_runs(_rif_pm, "cited for")
                                 + f" (the passage(s) of the indexed {_inl} corpus naming both are "
                                 f"excluded by the evidence filters: "
                                 + "; ".join(sorted({_why_short(_w) for _kd, _w in _xw if _kd == "passage"}))
                                 + "); not flagged as novel.")
                    elif not _corpus_pm:
                        _text = (f"Reported for {tgt} in this domain by curated NCBI "
                                 f"{_rif_w} " + _cite_runs(_rif_pm, "cited for")
                                 + f" (no passage in the indexed {_inl} corpus names "
                                 f"both); not flagged as novel.")
                    elif not _rif_pm:
                        _text = (f"Already reported in the indexed {_inl} literature "
                                 + _cite_runs(_corpus_pm) + "; not flagged as novel.")
                    else:
                        _text = (f"Already reported in the indexed {_inl} literature "
                                 + _cite_runs(_corpus_pm)
                                 + f", with further support from curated NCBI {_rif_w} "
                                 + _cite_runs(_rif_pm, "cited for") + "; not flagged as novel.")
                elif dom_errs.get(_k):
                    _status = "error"
                    _text = (f"Retrieval from the indexed {_inl} corpus failed for this "
                             f"run; no novelty assessment is made for this domain.")
                elif _xw:
                    # followup 2026-09-28 (fix 2): every item naming the gene in this domain was
                    # excluded by the evidence filters -> neither "reported" nor a novelty claim;
                    # live PubMed (with the filters) may still find valid records
                    _status, _text, _pmres = _no_evidence_resolve(
                        tgt, _inl, _d,
                        f"The indexed {_inl} evidence naming {tgt} in this domain is excluded by the "
                        f"evidence filters ("
                        + "; ".join(sorted({_why_short(_w) for _kd, _w in _xw})) + ")",
                        evfilter=True, exclude=_card_excl_for(_cx, _k))
                elif dom_named[_k]:
                    # the corpus names the gene, but no passage ties it to this
                    # domain: ambiguous -> never a novelty claim. PI 2026-09-20
                    # (point 3): live PubMed is consulted first -- records there
                    # make this corpus_absent, not "no assessment".
                    _status, _text, _pmres = _no_evidence_resolve(
                        tgt, _inl, _d,
                        f"Passages from the indexed {_inl} corpus name {tgt}, but none "
                        f"links it to this domain",
                        **({"evfilter": True, "exclude": _card_excl_for(_cx, _k)} if _cx is not None else {}))
                elif _t and _UNREPORTED_RE.search(_t) and _cross_ev:
                    # no retrieved passage of this corpus names the gene, the gene has
                    # evidence elsewhere, and the LLM flagged the domain (today's rule).
                    # ADD-1 (2026-09-18): our corpora are snapshots of very uneven size,
                    # so before the claim is printed it is checked against live PubMed.
                    # Any record there downgrades it to corpus_absent -- a statement
                    # about OUR index, not a novelty claim. Wording is Python, never LLM.
                    _pmfrag = _d.get("pubmed") or ""
                    _pmres = _pubmed_domain_check(
                        tgt, _pmfrag, retmax=PUBMED_CHECK_RETMAX,
                        gene_name=(_gene_fullname(tgt) if _pmfrag else ""))
                    # FIX 2026-09-20 (review): a count is not a gene mention. The
                    # records themselves decide -- only records that name the gene AND
                    # pass this domain's own text gate can retire the claim.
                    # followup 2026-09-28 (fix 2): with the gate's evidence filters, and
                    # never a record of the row's filtered set
                    if _cx is not None:
                        _pmres = _pubmed_drop_excluded(_pubmed_verify(tgt, _d, _pmres, evfilter=True),
                                                       _card_excl_for(_cx, _k))
                    else:
                        _pmres = _pubmed_verify(tgt, _d, _pmres)
                    # FIX 2026-09-19: name what was actually searched. The query is
                    # one esearch over the symbol (plus the official full name when
                    # NCBI Gene gave one), not all of PubMed by every alias.
                    _searched = (f"{tgt} ({_pmres.get('gene_name')})"
                                 if _pmres.get("gene_name") else f"the symbol {tgt}")
                    if _pmres["ok"] and _pmres["verified"]:
                        _status = "corpus_absent"
                        _PUBMED_STATS["downgraded"] += 1
                        # PI 2026-09-20 (points 1-2): one shared wording, reading
                        # as "not flagged as novel", with no search date.
                        _text = _corpus_absent_text(tgt, _inl, _pmres)
                    elif _pmres["ok"] and _pmres.get("verify_filter_rejected"):
                        # followup 2026-09-28 (fix 2): the records naming the gene here are all
                        # invalid evidence -> grey, no PMID, no novelty claim
                        _status = "no_evidence"
                        _text = (f"No passage naming {tgt} was retrieved from the indexed {_inl} corpus; "
                                 + _pubmed_filtered_clause(_pmres, f"a live PubMed search for {_searched}")
                                 + "; no novelty assessment is made for this domain.")
                    elif _pmres["ok"] and _pmres["count"] > 0:
                        # matches exist but none of the records we read names the gene
                        # here (or they could not be read): the claim stands, and the
                        # sentence says exactly what the check did -- never "returned
                        # no record", which would be false.
                        _status = "unreported"
                        _text = (f"To our knowledge, this link has not been reported in the "
                                 f"indexed {_inl} literature; "
                                 + _pubmed_unverified_clause(
                                     _pmres, f"a live PubMed search for {_searched}") + ".")
                    elif _pmres["ok"]:
                        _status = "unreported"
                        # PI 2026-09-20 (point 2): same fact, no date.
                        _text = (f"To our knowledge, this link has not been reported in the "
                                 f"indexed {_inl} literature; a live PubMed search for "
                                 f"{_searched} returned no record either.")
                    else:
                        _status = "unreported"
                        _text = (f"To our knowledge, this link has not been reported in the "
                                 f"indexed {_inl} literature; the live PubMed check did not "
                                 f"complete for this run, so this is a statement about the "
                                 f"indexed corpus only.")
                else:
                    # PI 2026-09-20 (point 3): the second no_evidence variant is
                    # checked against live PubMed as well.
                    _status, _text, _pmres = _no_evidence_resolve(
                        tgt, _inl, _d,
                        f"No passage naming {tgt} was retrieved from the indexed "
                        f"{_inl} corpus",
                        **({"evfilter": True, "exclude": _card_excl_for(_cx, _k)} if _cx is not None else {}))
                if _status == "unreported" and _rif_incomplete:
                    # followup 2026-09-28 (fix 1): a novelty claim needs the gene's GeneRIF lines
                    # checked for this domain; they could not be read -> no claim
                    _status = "error"
                    _text = (f"No passage naming {tgt} was retrieved from the indexed {_inl} corpus, but "
                             f"the NCBI GeneRIF record of {tgt} could not be read completely for this run, "
                             f"so its GeneRIF lines were not checked for this domain; no novelty assessment "
                             f"is made for this domain.")
                for _pm in _sup:
                    _earlier.setdefault(_pm, _inl)
                _ent = {"key": _k, "label": _d["label"], "abbr": _d["abbr"],
                        "display": _d["display"], "status": _status, "text": _text,
                        "html": _pmid_links(_html_escape(_text, quote=False)),
                        "refs": _sup, "refs_source": _refs_source,
                        "n_hits": len(dom_ev[_k]),
                        "n_named": dom_named[_k], "n_retrieved": dom_nret[_k],
                        "n_generif": dom_rif.get(_k, 0)}
                if _pmres is not None and _cx is not None:
                    # followup 2026-09-28 (fix 2): no PMID of the row's filtered set on the card
                    _ent["pubmed"] = _card_pubmed_json(_pmres, set(_card_excl_for(_cx, _k)))
                elif _pmres is not None:
                    _ent["pubmed"] = {"ok": _pmres["ok"], "count": _pmres["count"],
                                      "pmids": _pmres["pmids"], "date": _pmres["date"],
                                      "term": _pmres["term"],
                                      "gene_name": _pmres.get("gene_name", ""),
                                      # FIX 2026-09-20 (review): which records were read
                                      # back and which of them passed both gates
                                      "checked": list(_pmres.get("checked") or []),
                                      "verified": list(_pmres.get("verified") or []),
                                      "verify_how": dict(_pmres.get("verify_how") or {}),
                                      "verify_ok": bool(_pmres.get("verify_ok")),
                                      "verify_error": _pmres.get("verify_error", ""),
                                      "error": _pmres["error"]}
                if _xw:
                    # followup 2026-09-28 (fix 2): how many items the filters excluded, and why
                    # (no PMID: an excluded paper is never cited, not even here)
                    _ent["n_excluded"] = len(_xw)
                    _ent["excluded_why"] = sorted({_why_short(_w) for _kd, _w in _xw})
                if dom_errs.get(_k):
                    _ent["retrieval_error"] = str(dom_errs[_k])[:200]
                nov_entries.append(_ent)
            # legacy keys = the FIRST (primary) domain
            novelty_html = nov_entries[0]["html"]
            field_support = list(nov_entries[0]["refs"])
            field_absent = not dom_support[_doms[0]["key"]]

        # Grounding gate: Notable Findings must have real evidence to
        # discuss. Drop candidates with no GeneRIF and no cross-domain
        # literature (they survive observed scoring but are empty cards).
        _grif = [r for r in rifs[:3] if (r.get("text") or "").strip()]
        if not _searchable:
            # No human ortholog. Before Task B this gene fell through the
            # "no evidence" guard below and vanished from Section 14 without a
            # word -- indistinguishable from a gene that was searched and found
            # wanting. It now gets a card that states why nothing was searched.
            # OVERWRITE, never "or": with no evidence at all the per-card LLM
            # can still emit an uncited sentence, and an uncited sentence about
            # a gene nobody searched is exactly the claim this branch exists to
            # prevent.
            cross_txt = str(_orth_entry.get("disclosure") or
                            "No human ortholog: no literature was searched.")
            novelty_html = _html_escape(
                str(_orth_entry.get("disclosure") or
                    "No human ortholog: no literature was searched for this "
                    "gene, so neither novelty nor prior report can be "
                    "asserted."), quote=False)
            if _doms is not None:
                nov_entries = [{
                    "key": _d["key"], "label": _d["label"], "abbr": _d["abbr"],
                    "display": _d["display"], "status": "no_ortholog",
                    "text": ("%s has no human ortholog, so the indexed %s "
                             "literature could not be searched for it; no "
                             "novelty assessment is made."
                             % (_gene_run, _d["display"])),
                    "html": _html_escape(
                        "%s has no human ortholog, so the indexed %s literature "
                        "could not be searched for it; no novelty assessment is "
                        "made." % (_gene_run, _d["display"]), quote=False),
                    "refs": [], "refs_source": "", "n_hits": 0,
                    "n_named": 0, "n_retrieved": 0, "n_generif": 0}
                    for _d in _doms]
        elif not _grif and not cross_txt.strip():
            continue
        if _cx is not None and _cx.get("paper_errors"):
            # followup 2026-09-28 (fix 2): a NEC item whose PubMed record could not be read was
            # counted without the paper-level NET test (as the gate does), and the card says so
            _pe = _cx.pop("paper_errors")
            _card_flags.append({"label": "NEC paper check not run",
                                "note": "the paper-level NET test could not read PubMed record(s) %s (%s)"
                                        % (", ".join(sorted({p for p, _e in _pe})[:5]), _pe[0][1][:120])})
        cards.append({
            "gene": _gene_run,
            # The literature symbol and the sentence that discloses it. Empty
            # strings on a human run, so the card renders exactly as before.
            "lit_symbol": (tgt if tgt != _gene_run else ""),
            "lit_searched": bool(_searchable),
            "ortholog_disclosure": str(_orth_entry.get("disclosure") or ""),
            "rna_log2fc": round(lf, 2),
            "regulated_by": tfs_for,
            "n_edges": len(tfs_for),
            "cross_domain_html": _pmid_links(cross_txt),
            "novelty_html": novelty_html,
            "generif_pmids": rif_pmids,
            "generif_items": [
                {"pmid": (r.get("pmids") or [""])[0],
                 "text": (r.get("text", "") or "").strip()}
                for r in rifs[:3] if (r.get("text") or "").strip()
            ],
            "cross_pmids": cross_pmids,
            "field_pmids": field_support,
            "_score": _score,
            "_field_absent": bool(field_absent),
        })
        if _doms is not None:
            cards[-1]["novelty_by_domain"] = nov_entries
            cards[-1]["novelty_domains_version"] = NOVELTY_DOMAINS_VERSION
        if _card_flags:
            # followup 2026-09-28: shown next to the gene (render.py joins them into evidence_flag)
            cards[-1]["card_flags"] = _card_flags
            if _rif_incomplete:
                cards[-1]["generif_fetch"] = "failed"
                cards[-1]["generif_fetch_error"] = _rif_incomplete[:300]
    # With an explicit rank_order the cards keep the curated order -- a
    # novel-first re-sort would silently override the ranking the caller asked
    # for. Otherwise: unreported-in-field leads, then combined score.
    if _rank_idx:
        cards.sort(key=lambda c: _rank_idx.get(c["gene"], 10**6))
    else:
        cards.sort(key=lambda c: (0 if c["_field_absent"] else 1, -c["_score"]))
    cards = cards[:max_cards]
    for c in cards:
        c.pop("_score", None)
        c.pop("_field_absent", None)
    if _doms is not None:
        import sys as _sys
        _ps, _gs = _PUBMED_STATS, _GENE_NAME_STATS
        print(f"[narrative] notable cards: {len(cards)} card(s), novelty domains "
              f"{[d['key'] for d in _doms]}, retrieval {_t_ret:.0f} s, LLM {_t_llm:.0f} s, "
              f"PubMed {_ps['calls']} call(s) ({_ps['cached']} cached, {_ps['errors']} failed, "
              f"{_ps['seconds']:.0f} s, {_ps['downgraded']} unreported -> corpus_absent, "
              f"{_ps['ne_checks']} no_evidence check(s) -> {_ps['ne_downgraded']} "
              f"corpus_absent), "
              f"record checks {_ps['verify_calls']} efetch(es) ({_ps['verify_cached']} cached, "
              f"{_ps['verify_errors']} failed, {_ps['verify_rejected']}/{_ps['verify_records']} "
              f"record(s) rejected), "
              f"gene names {_gs['calls']} call(s) ({_gs['resolved']} resolved, "
              f"{_gs['errors']} failed, {_gs['seconds']:.0f} s), "
              f"total {time.time() - _t_nf0:.0f} s", file=_sys.stderr, flush=True)
    return cards, passages


def register_narrative_tools(mcp):

    @mcp.tool()
    def generate_narrative(
        tfs_summary_json: str,
        targets_summary_json: str,
        output_md: str,
        conditions_meta_json: Optional[str] = None,
        retriever_script: Optional[str] = None,
        retrieval_top_k: int = 5,
        glm_endpoint: str = DEFAULT_GLM_ENDPOINT,
        glm_model: str = DEFAULT_GLM_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 1800,
        field: str = "prostate",
        novelty_domains_json: Optional[str] = None,
        ortholog_json: Optional[str] = None,
        organism: str = "",
    ) -> dict:
        """Generate paper-grounded TRN narrative with citation harness.

        Para 1 (study context) is built deterministically from conditions_meta.
        Paras 2+ are LLM-generated under whitelist/quote harness.
        novelty_domains_json: optional JSON list of literature-domain specs
        ({key, label, abbr, display, terms, exclude}) -> per-domain Novelty on
        the notable cards (default: legacy single-field Novelty).
        ortholog_json: optional path to the run's ortholog.json (written by
        CuratedTargetsNode). For a non-human run it maps every gene to the
        human ortholog whose literature is actually searched, and carries the
        per-gene disclosure the card must print. Absent / identity = the human
        path, unchanged.
        """
        tfs = json.loads(Path(tfs_summary_json).read_text())
        targets = json.loads(Path(targets_summary_json).read_text())
        # The prompt carries no AlphaGenome block (_drop_ag_block): an always-empty heading only
        # invited a sentence about a validation the report does not have.  Section 13's axis test
        # is not narrative input.
        conditions = []
        if conditions_meta_json and Path(conditions_meta_json).exists():
            conditions = json.loads(Path(conditions_meta_json).read_text())

        lit_passages = []
        harness_passages = []
        used_pmids = []

        # 1) Build deterministic study context paragraph
        study_para, study_passages = ("", [])
        if conditions and retriever_script and Path(retriever_script).exists():
            study_para, study_passages = _build_study_context_paragraph(
                conditions, retriever_script, top_k=2,
                venv_python="/BIO1/MRAG/venvs/paper_qa/bin/python3",
            )
            for pp in study_passages:
                if pp["pmid"] not in used_pmids:
                    used_pmids.append(pp["pmid"])
                lit_passages.append(f"[PMID:{pp['pmid']}] {pp.get('title','')}\n  {pp.get('snippet','')[:400]}")
                harness_passages.append(pp)

        # 2) TF-level retrieval (for downstream paragraphs)
        if retriever_script and Path(retriever_script).exists():
            tf_names = [t.get("name") if isinstance(t, dict) else t for t in tfs[:8]]
            for tf in tf_names:
                q = f"{tf} transcription factor cancer target gene regulation"
                hits = _query_paperqa(retriever_script, q, top_k=retrieval_top_k)
                for h in hits:
                    pmid = h.get("pmid", "")
                    snip = h.get("snippet", "") or h.get("text", "")
                    title = h.get("title", "")
                    if pmid and pmid not in used_pmids:
                        used_pmids.append(pmid)
                    lit_passages.append(f"[PMID:{pmid}] {title}\n  {snip[:400]}")
                    harness_passages.append({"pmid": pmid, "title": title, "snippet": snip})
        lit_block = ("\n\n".join(lit_passages[:40])
                     if lit_passages else "(no literature retrieval performed)")

        # 3) Build prompt (strict harness; conditions OMITTED since handled deterministically)
        if harness_passages:
            prompt = _drop_ag_block(build_strict_prompt(
                tfs_block=json.dumps(tfs, indent=2)[:2000],
                targets_block=json.dumps(targets, indent=2)[:4000],
                ag_block="",
                passages=harness_passages,
                conditions=None,  # skip the LLM study-context section; we prepend study_para
            ))
        else:
            prompt = PROMPT_TEMPLATE.format(
                tfs_block=json.dumps(tfs, indent=2)[:2000],
                targets_block=json.dumps(targets, indent=2)[:4000],
                lit_block=lit_block,
            )
        sys_msg = {"role": "system", "content": "You are a careful, paper-grounded result writer for genomics."}
        user_msg = {"role": "user", "content": prompt}

        t0 = time.time()
        try:
            glm_narrative = _glm_chat([sys_msg, user_msg],
                                       endpoint=glm_endpoint, model=glm_model,
                                       temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            glm_narrative = f"# GLM call failed\n\n```\n{e}\n```\n"
        dt = time.time() - t0

        # 4) Notable Findings cards (deterministic observed-data targets grounded
        #    by GeneGPT GeneRIFs + paperQA; conservative novelty, 3-layer guard).
        notable_cards, notable_passages = ([], [])
        try:
            _nov_doms = None
            if novelty_domains_json:
                try:
                    _nov_doms = json.loads(novelty_domains_json) or None
                except Exception:
                    _nov_doms = None
            notable_cards, notable_passages = _build_notable_findings(
                tfs, targets,
                field, retriever_script, glm_endpoint, glm_model,
                domains=_nov_doms, ortholog=load_ortholog(ortholog_json),
                organism=organism)
        except Exception:
            notable_cards, notable_passages = ([], [])
        for pp in notable_passages:
            if pp.get("pmid") and pp["pmid"] not in used_pmids:
                used_pmids.append(pp["pmid"])
            harness_passages.append(pp)

        # 5) Analysis Result body (Notable Findings render as §15 cards)
        body = (study_para + "\n\n" + glm_narrative) if study_para else glm_narrative
        narrative = "## Analysis Result\n\n" + body

        # 6) Citation harness: whitelist + quote + numbered refs
        audit = {"n_total": 0, "n_kept": 0, "n_removed_whitelist": 0,
                 "n_unverified_quote": 0, "n_references": 0, "references": []}
        narrative_cleaned = narrative
        if harness_passages:
            v = validate_citations(narrative, harness_passages)
            numbered_text, references = renumber_and_build_references(
                v["text_cleaned"], harness_passages, v["validated"])
            references_md = format_references_markdown(references)
            narrative_cleaned = numbered_text + references_md
            audit = {
                "n_total": v["n_total"],
                "n_kept": v["n_kept"],
                "n_removed_whitelist": v["n_removed_whitelist"],
                "n_unverified_quote": v["n_unverified_quote"],
                "n_references": len(references),
                "references": references,
                "validated": v["validated"],
            }

        Path(output_md).write_text(narrative_cleaned)
        Path(output_md + ".raw.md").write_text(narrative)
        Path(output_md + ".audit.json").write_text(json.dumps(audit, indent=2))
        cards_path = str(Path(output_md).parent / "notable_cards.json")
        Path(cards_path).write_text(json.dumps(notable_cards, indent=2, ensure_ascii=False))
        return {
            "ok": True,
            "notable_cards_json": cards_path,
            "n_notable_cards": len(notable_cards),
            "output_md": output_md,
            "audit_json": output_md + ".audit.json",
            "n_chars": len(narrative_cleaned),
            "used_pmids": used_pmids,
            "citation_audit": audit,
            "elapsed_s": round(dt, 1),
            "endpoint": glm_endpoint, "model": glm_model,
        }
