"""Enrichment core tools."""
from __future__ import annotations
import hashlib, json, os, re, urllib.request
from pathlib import Path
from typing import Optional, List

# Default libraries — MSigDB Hallmark + GO BP + Reactome (Enrichr names)
DEFAULT_LIBRARIES = ["MSigDB_Hallmark_2020", "GO_Biological_Process_2023", "Reactome_2022"]

# The term judgements below run on the SCORER profile, not the [llm] endpoint the rest of the
# pipeline uses: it is the model the curated target scores were validated on, so the domain
# reasoning here matches the reasoning that chose the targets in the first place.
def _scorer_setting(name, legacy, default):
    """MRAG_ENRICH_* first, then the scorer's own MRAG_SCORE_* so a run that moved its scorer
    moves these judgements with it, then the q38 default they were validated on."""
    v = os.environ.get(name) or os.environ.get(legacy) or default
    return v.strip() if isinstance(v, str) else v


SCORER_ENDPOINT = _scorer_setting("MRAG_ENRICH_ENDPOINT", "MRAG_SCORE_ENDPOINT",
                                  "http://127.0.0.1:11501/v1/chat/completions")
if SCORER_ENDPOINT and not SCORER_ENDPOINT.rstrip("/").endswith("chat/completions"):
    SCORER_ENDPOINT = SCORER_ENDPOINT.rstrip("/") + "/v1/chat/completions"
SCORER_MODEL = _scorer_setting("MRAG_ENRICH_MODEL", "MRAG_SCORE_MODEL", "qwen3.8:27b-q8_0")
SCORER_EFFORT = _scorer_setting("MRAG_ENRICH_EFFORT", "MRAG_SCORE_REASONING_EFFORT", "low")

# A term hit by a single target gene is an annotation lookup, not enrichment: "significant in 14
# TFs" then only means those 14 TFs bind that one gene.  In one prostate run, 146 of 186
# significant terms were single-gene and crowded the real ones out of the candidate pool.
MIN_HITS = 2
# Nominal p, not the adjusted one.  A TRN has ~20 target genes; Benjamini-Hochberg across the
# thousands of terms in three libraries leaves almost nothing -- 7 terms for one domain's shared
# panel and 2 for another, which is not a figure.  0.01 rather than 0.05 because the looser cut
# admits terms resting on nothing but a two-gene coincidence.  Every axis says "p-value", and the
# section text says the values are nominal.
PVAL_MAX = 0.01
# Some target sets simply do not share pathways: one prostate run's twenty targets left four
# terms at 0.01, against eleven for a neuroendocrine run whose targets share a strong axis.
# Rather than carry a different threshold per domain, the cut relaxes only when the strict one
# cannot fill the panel, and the figure says which cut it used.
PVAL_RELAX = 0.05
MIN_ROWS = 8          # a shorter figure beats one padded with rejected terms
MIN_PER_LIB = 3       # keep every source library visible in the dot plot
BATCH = 25            # terms per relevance call; one big call makes the model reason itself out
                      # of its token budget and return nothing

# Generic parents: significant, multi-gene, and still uninformative.  Because nearly every target
# carries the annotation they light up every column, costing a row and separating no factors.
BROAD = re.compile(
    r"^(positive |negative )?regulation of ("
    r"transcription|dna-templated transcription|gene expression|"
    r"macromolecule metabolic process|cellular process|biological process|"
    r"metabolic process|cellular metabolic process|primary metabolic process|"
    r"nitrogen compound metabolic process|multicellular organismal process|"
    r"signal transduction|cell communication|response to stimulus|"
    r"transcription by rna polymerase ii|rna biosynthetic process|"
    r"nucleobase-containing compound metabolic process)$"
    r"|^(gene expression|signal transduction|developmental biology|metabolism|"
    r"metabolism of proteins|biological oxidations|cellular responses to stimuli|"
    r"phosphate-containing compound metabolic process|"
    r"cellular macromolecule biosynthetic process|"
    r"transcription by rna polymerase ii|generic transcription pathway)$",
    re.I)

# Names of anatomical structures.  Dropped unless the structure is the sample's own, which the
# per-run lineage lookup supplies: a liver term belongs in a liver figure and nowhere else.
ANATOMICAL = re.compile(
    r"\b(brain|forebrain|midbrain|hindbrain|cerebral|cerebellum|cerebellar|hippocamp\w*|"
    r"central nervous system|spinal cord|retina\w*|eye|inner ear|cochlea\w*|"
    r"kidney|renal|nephron|liver|hepat\w*|pancrea\w*|heart|cardiac|myocard\w*|"
    r"lung|pulmonary|alveol\w*|bronch\w*|prostate|mammary|breast|ovar\w*|uter\w*|"
    r"testis|testicular|thyroid|adrenal|stomach|gastric|intestin\w*|colon\w*|"
    r"skin|epiderm\w*|skeletal system|bone|cartilage|chondrocyte|skeletal muscle|myotube|"
    r"spermatogenesis|oogenesis|placenta\w*|"
    r"axon guidance|axonogenesis|neuron projection guidance)\b", re.I)

# Synapse structure and neurotransmission, dropped in every run.  A neuroendocrine tumour
# expresses neuronal genes, which is its differentiation programme, but it has no synapses and
# does not transmit: in both runs here these terms were carried by CDH2 (an adhesion molecule)
# plus a single receptor, and the same pair produced several differently-named terms.
SYNAPTIC = re.compile(r"\b(synap\w*|postsynap\w*|presynap\w*|neurotransmitter|"
                      r"glutamatergic|gabaergic|dopaminergic|cholinergic|serotonergic|"
                      r"neurotransmission|synaptic vesicle)\b", re.I)

# Neural programme names, dropped when the run's lineage is not itself neural/neuroendocrine.
NEURAL = re.compile(r"\b(neuron\w*|neuronal|nervous system|neural|synap\w*|axon\w*|"
                    r"glutamatergic|gabaergic|dendrit\w*|neurotrophin|neurite)\b", re.I)

LIB_SHORTNAME = {
    "MSigDB_Hallmark_2020": "Hallmark",
    "GO_Biological_Process_2023": "GO_BP",
    "Reactome_2022": "Reactome",
}
LIB_COLOR = {"Hallmark": "#1f78b4", "GO_BP": "#33a02c", "Reactome": "#e31a1c"}


def _scorer_call(prompt, max_tokens=12000, timeout=900):
    """One call to the scorer model.  Raises on an empty body, which is what a model that spent
    its whole budget reasoning returns; callers retry with less to chew on."""
    body = {"model": SCORER_MODEL, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    if SCORER_EFFORT:
        body["reasoning_effort"] = SCORER_EFFORT
    req = urllib.request.Request(SCORER_ENDPOINT, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    ch = payload["choices"][0]
    txt = (ch.get("message") or {}).get("content") or ""
    if not txt.strip():
        raise ValueError(f"empty content (finish={ch.get('finish_reason')}, "
                         f"usage={payload.get('usage')})")
    txt = re.sub(r"^\s*```(?:json)?|```\s*$", "", txt.strip())
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError("no JSON in reply: " + txt[:200])
    return json.loads(m.group(0))


def _fingerprint(*parts):
    """A short digest of everything a cached answer depends on."""
    h = hashlib.md5()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()[:10]


def _cached(path, fn, fingerprint=None):
    """Deterministic at temperature 0, so the answer is kept beside the run's outputs and a
    re-render or a resumed run costs no model time.

    The file records a fingerprint of what was asked.  Without one, a run derived from another
    (new_run_from.sh copies output/ wholesale) would inherit that run's lineage and drops, so a
    prostate run's answer would decide a neuroendocrine run's figure.
    """
    try:
        if path and Path(path).exists():
            blob = json.loads(Path(path).read_text())
            if isinstance(blob, dict) and "_fp" in blob:
                if fingerprint is None or blob.get("_fp") == fingerprint:
                    return blob.get("value")
                print(f"[enrichment] cache ignored, inputs changed: {Path(path).name}",
                      flush=True)
            # a file without _fp predates this and cannot be matched to its inputs
    except Exception:
        pass
    out = fn()
    try:
        if path and out is not None:
            Path(path).write_text(json.dumps({"_fp": fingerprint, "value": out}, indent=1))
    except Exception as ex:
        print(f"[enrichment] cache write failed ({path}): {ex}", flush=True)
    return out


def resolve_lineage(dataset_context, cache_path=None):
    """What lineage does this run's target programme belong to, and what anatomy is its own?

    Asked once per run so the filters below generalise to any literature domain instead of a
    table of the domains seen so far.  Fail-open: without an answer nothing is filtered by
    lineage, which is the behaviour before this existed.
    """
    ctx = dataset_context or {}
    lines = []
    for k in ("cancer_type", "literature_domains", "tissue", "cell_line", "organism",
              "comparison", "perturbation", "study_label"):
        v = ctx.get(k)
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v if x)
        if v:
            lines.append(f"- {k}: {v}")
    if not lines:
        return {"lineage": "", "own_anatomy": [], "lineage_is_neural": False, "source": "no context"}

    def _ask():
        prompt = (
            "Describe the lineage of the tumour or sample below, for use in filtering pathway "
            "enrichment terms shown in a figure about its transcriptional regulatory network.\n\n"
            "Sample:\n" + "\n".join(lines) + "\n\n"
            "Answer three things:\n"
            '  "lineage": a short phrase for the cell lineage its target programme should belong '
            'to, e.g. "prostate epithelium (glandular)", "neuroendocrine", "hepatocyte".\n'
            '  "own_anatomy": the anatomical words that belong to THIS sample, lower case, so a '
            'term naming them is kept. For a liver cancer: ["liver","hepatocyte","hepatic"]. '
            'For a prostate adenocarcinoma: ["prostate","glandular epithelium"]. Include the '
            "organ, its cell type and common adjectives. Omit anything the sample is not.\n"
            '  "lineage_is_neural": true only when the lineage is itself neural or '
            "neuroendocrine, so that neuron, synapse and axon programmes are its own biology "
            "rather than another lineage's. A small cell lung cancer or a neuroendocrine "
            "carcinoma is true; a prostate adenocarcinoma is false even when it is known to "
            "transdifferentiate.\n\n"
            'Return JSON only: {"lineage": "...", "own_anatomy": ["..."], '
            '"lineage_is_neural": true|false}')
        return _scorer_call(prompt, max_tokens=4000, timeout=600)

    try:
        out = _cached(cache_path, _ask, _fingerprint(SCORER_MODEL, lines))
        if not isinstance(out, dict):
            raise ValueError("lineage reply not an object")
        own = [str(x).strip().lower() for x in (out.get("own_anatomy") or []) if str(x).strip()]
        res = {"lineage": str(out.get("lineage", "")).strip(),
               "own_anatomy": own,
               "lineage_is_neural": bool(out.get("lineage_is_neural")),
               "source": "llm"}
        print(f"[enrichment] lineage: {res['lineage']!r} | own anatomy: {own} | "
              f"neural lineage: {res['lineage_is_neural']}", flush=True)
        return res
    except Exception as ex:
        print(f"[enrichment] lineage lookup failed, no lineage filtering: {ex}", flush=True)
        return {"lineage": "", "own_anatomy": [], "lineage_is_neural": False, "source": "failed"}


def _own_anatomy_re(own):
    if not own:
        return None
    return re.compile("|".join(r"\b" + re.escape(w) + r"\w*" for w in own), re.I)


def structural_drops(terms, lin):
    """{term: reason} for the terms no judgement is needed on.

    Three rules, all of them about the term's name or its support rather than its biology, so
    they are applied before the model is asked anything and cost nothing.
    """
    own = _own_anatomy_re((lin or {}).get("own_anatomy"))
    out = {}
    for t in terms:
        if BROAD.match(t):
            out[t] = "generic parent"
        elif ANATOMICAL.search(t) and not (own and own.search(t)):
            out[t] = "anatomy of another tissue"
        elif SYNAPTIC.search(t):
            out[t] = "synapse/transmission, a tumour has neither"
        elif NEURAL.search(t) and not (lin or {}).get("lineage_is_neural"):
            out[t] = "neural programme, lineage is not"
    return out


def llm_drop_irrelevant(cand, lin, dataset_context, cache_path=None, floor=MIN_ROWS):
    """Which remaining terms do not belong in this run's figure.

    Batched: the judgement is local to one term, and a single call over the whole pool makes the
    model reason until its budget is gone and return nothing at all.
    `cand` is [(term, library, genes_str)].  Never filters below `floor`.
    """
    terms = [c[0] for c in cand]
    if not terms or not (lin or {}).get("lineage"):
        return {}

    def _ask():
        dom = (dataset_context or {}).get("cancer_type") or \
              (dataset_context or {}).get("tissue") or "this sample"
        dropped, failed = {}, False
        for i in range(0, len(cand), BATCH):
            chunk = cand[i:i + BATCH]
            listing = "\n".join(f'- "{t}" | genes: {g}' for t, _l, g in chunk)
            prompt = (
                f"A figure for a {dom} study lists enriched pathway/process terms.\n"
                f"The tumour's lineage here is: {lin['lineage']}.\n\n"
                f"Mark a term DROP when any of these holds:\n"
                f"  (a) it makes no biological sense in {dom} (e.g. toxin toxicity, an "
                f"unrelated tissue's programme, unrelated metabolism);\n"
                f"  (b) it is so broad that its name says almost nothing;\n"
                f"  (c) its programme belongs to a different lineage from {lin['lineage']}, even "
                f"though real genes annotate there. This figure describes a tumour's regulatory "
                f"programme, not an organ;\n"
                f"  (d) it describes synapses or neurotransmission. A neuroendocrine tumour "
                f"expresses neuronal genes -- that is its differentiation programme, and terms "
                f"about neuron development or maturation are right -- but it has no synapses and "
                f"transmits nothing.\n"
                f"Keep everything else; be conservative.\n\n"
                f'Return JSON only: {{"drop": [{{"term": <name copied verbatim>, '
                f'"reason": "<3 words>"}}]}}\n\nTerms:\n' + listing)
            try:
                out = _scorer_call(prompt, max_tokens=8000, timeout=600)
            except Exception as ex:
                # a timed-out batch must not be remembered as "nothing to drop"
                print(f"[enrichment] relevance batch kept as-is: {ex}", flush=True)
                failed = True
                continue
            names = {t for t, _l, _g in chunk}
            for d in (out.get("drop") or []):
                if not isinstance(d, dict):
                    continue
                t = d.get("term")
                if isinstance(t, str) and t in names:
                    dropped[t] = str(d.get("reason", ""))[:40]
        return None if failed else dropped

    fp = _fingerprint(SCORER_MODEL, lin.get("lineage"), terms,
                      (dataset_context or {}).get("cancer_type"))
    dropped = _cached(cache_path, _ask, fp)
    dropped = dict(dropped or {})
    dropped = {t: r for t, r in dropped.items() if t in set(terms)}
    # never filter below the floor; give back the ones listed first, which arrive most-shared first
    order = [t for t in terms if t in dropped]
    while len(terms) - len(dropped) < floor and order:
        dropped.pop(order.pop(0), None)
    return dropped


def _gene_set(g):
    return {x.strip().upper() for x in str(g).split(";") if x.strip()}


def duplicate_groups(cand, thresh=0.6):
    """Terms driven by essentially the same genes, grouped.  Names differ, the biology does not."""
    gs = {t: _gene_set(g) for t, _l, g in cand}
    order = [t for t, _l, _g in cand]
    groups, used = [], set()
    for i, a in enumerate(order):
        if a in used or not gs[a]:
            continue
        grp = [a]
        for b in order[i + 1:]:
            if b in used or not gs[b]:
                continue
            u = len(gs[a] | gs[b])
            if u and len(gs[a] & gs[b]) / u >= thresh:
                grp.append(b)
        if len(grp) > 1:
            groups.append(grp)
            used.update(grp)
    return groups


def llm_pick_in_groups(groups, lin, cache_path=None):
    """One term per duplicate group; the model picks the informative name, code keeps the rest out."""
    if not groups:
        return {}

    def _ask():
        listing = "\n".join(f"group {i}: " + " | ".join(f'"{t}"' for t in g)
                            for i, g in enumerate(groups))
        prompt = (
            f"In a figure about a {lin.get('lineage', 'tumour')} regulatory network, the terms "
            f"inside each group below are driven by essentially the same genes, so only one of "
            f"each group should take a row.\n\nFor each group pick the single most informative "
            f"term -- the one whose name tells a reader the most about the biology, preferring a "
            f"named programme or pathway over a generic parent.\n\n"
            f'Return JSON only: {{"keep": {{"<group number>": "<term copied verbatim>"}}}}\n\n'
            + listing)
        return _scorer_call(prompt, max_tokens=8000, timeout=600)

    drop = {}
    try:
        out = _cached(cache_path, _ask, _fingerprint(SCORER_MODEL, lin.get("lineage"), groups))
        keep = out.get("keep") if isinstance(out, dict) else None
        if not isinstance(keep, dict):
            keep = {}
        for i, g in enumerate(groups):
            chosen = keep.get(str(i)) or keep.get(i)
            if not isinstance(chosen, str) or chosen not in g:
                chosen = g[0]
            for t in g:
                if t != chosen:
                    drop[t] = f"duplicate of {chosen}"
    except Exception as ex:
        # one malformed reply must not take the figure down with it
        print(f"[enrichment] dedupe kept every group intact: {ex}", flush=True)
        return {}
    return drop


def _llm_compact_terms(terms, max_chars=42):
    """Shorten long term names for the axis.  Fail-open to a deterministic truncation."""
    terms = list(terms)
    if not terms:
        return {}
    longs = [t for t in terms if len(t) > max_chars]
    if not longs:
        return {t: t for t in terms}
    prompt = (
        f"Compact each biological pathway/process name to <= {max_chars} characters while "
        f"preserving the key biology. Use standard abbreviations (VEGF, EMT, GPCR, NF-kB, ER, "
        f"mRNA, ATP, DNA, Reg., Neg., Pos.). Return JSON only, an object mapping the EXACT "
        f"original term to the compact term. No commentary.\n\n"
        f'Examples:\n  "Negative Regulation Of Cellular Response To Vascular Endothelial Growth '
        f'Factor Stimulus" -> "Neg. Reg. of Cell Response to VEGF"\n'
        f'  "Regulation Of Chondrocyte Differentiation" -> "Reg. of Chondrocyte Differentiation"'
        f"\n\nTerms:\n" + "\n".join(f"- {t}" for t in longs))
    try:
        mapping = _scorer_call(prompt, max_tokens=8000, timeout=600)
        if not isinstance(mapping, dict):
            raise ValueError("LLM output not dict")
    except Exception as ex:
        print(f"[enrichment] LLM compact failed: {ex}", flush=True)
        mapping = {}
    out = {}
    for t in terms:
        v = mapping.get(t, t)
        if not isinstance(v, str) or len(v) > max_chars + 5:
            v = t if len(t) <= max_chars else t[:max_chars - 1].rstrip() + "…"
        out[t] = v
    return out


def _clean_term(term: str) -> str:
    s = re.sub(r"\s*\(GO:\d+\)\s*$", "", str(term))
    s = re.sub(r"\s*R-[A-Z]{3}-\d+\s*$", "", s)
    s = re.sub(r"\s*WP\d+\s*$", "", s)
    return s.strip()


def _hits(overlap) -> int:
    try:
        return int(str(overlap).split("/")[0])
    except Exception:
        return 0


def _enrich_genes(gene_list, libraries, background_size=20000,
                  max_retries=5, base_delay=2.0):
    """Run gseapy.enrich for each library. Enrichr API throttles aggressively;
    retry with exponential backoff on 429."""
    import gseapy, time
    import pandas as pd
    if not gene_list:
        return pd.DataFrame()
    frames = []
    for lib in libraries:
        for attempt in range(max_retries):
            try:
                res = gseapy.enrich(gene_list=list(set(gene_list)), gene_sets=lib,
                                    background=background_size, outdir=None, verbose=False)
                df = res.results.copy()
                df["Library"] = LIB_SHORTNAME.get(lib, lib)
                frames.append(df)
                time.sleep(0.6)
                break
            except Exception as ex:
                if "429" in str(ex) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"[enrichment] {lib} 429, retry in {delay:.0f}s", flush=True)
                    time.sleep(delay)
                    continue
                print(f"[enrichment] lib {lib} failed: {ex}", flush=True)
                break
    if not frames:
        return pd.DataFrame()
    import pandas as pd
    return pd.concat(frames, axis=0, ignore_index=True)


def _usable_terms(long_df, lin, dataset_context, out_dir, tag, floor=MIN_ROWS):
    """The candidate terms a figure may draw on, and an audit of everything removed.

    long_df needs Term (cleaned), Library, Genes, Overlap, P-value and Adjusted P-value.
    """
    best = None
    for p_cut in (PVAL_MAX, PVAL_RELAX):
        recs, audit = _usable_at(long_df, lin, dataset_context, out_dir, tag, floor, p_cut)
        if best is None or len(recs) > len(best[0]):
            best = (recs, audit)
        if len(recs) >= floor:
            return recs, audit
        if p_cut != PVAL_RELAX:
            print(f"[enrichment] {tag}: {len(recs)} terms at p<{p_cut}, below the {floor} a "
                  f"panel needs; relaxing to p<{PVAL_RELAX}", flush=True)
    return best


def _usable_at(long_df, lin, dataset_context, out_dir, tag, floor, p_cut):
    """One pass of the pipeline at one significance cut."""
    sig = long_df[(long_df["P-value"] < p_cut) & (long_df["hits"] >= MIN_HITS)]
    if sig.empty:
        return [], {"structural": {}, "irrelevant": {}, "duplicate": {},
                    "min_hits": int(MIN_HITS), "p_used": p_cut}
    # one row per term: its library, the union of the genes that carry it, how many TFs it is
    # significant in, and its best adjusted p
    recs = []
    for term, g in sig.groupby("Term"):
        genes = sorted({x for s in g["Genes"] for x in _gene_set(s)})
        recs.append({"Term": term, "Library": g["Library"].iloc[0], "genes": genes,
                     "n_tf": int(g["TF"].nunique()) if "TF" in g.columns else 1,
                     "best_p": float(g["P-value"].min()),
                     "best_padj": float(g["Adjusted P-value"].min())})
    recs.sort(key=lambda r: (-r["n_tf"], r["best_p"]))

    audit = {"min_hits": int(MIN_HITS), "p_used": p_cut,
             "p_field": "P-value (nominal)"}
    struct = structural_drops([r["Term"] for r in recs], lin)
    audit["structural"] = struct
    if struct:
        print(f"[enrichment] {tag}: structural drops {len(struct)}", flush=True)
    recs = [r for r in recs if r["Term"] not in struct]

    cand = [(r["Term"], r["Library"], ";".join(r["genes"][:4])) for r in recs]
    irr = llm_drop_irrelevant(cand, lin, dataset_context,
                              cache_path=str(Path(out_dir) / f"_llm_{tag}_p{p_cut}_relevance.json"),
                              floor=floor)
    audit["irrelevant"] = irr
    recs = [r for r in recs if r["Term"] not in irr]

    # the prompt shows four genes; grouping must see all of them or unrelated terms merge
    cand = [(r["Term"], r["Library"], ";".join(r["genes"][:4])) for r in recs]
    full = [(r["Term"], r["Library"], ";".join(r["genes"])) for r in recs]
    dup = llm_pick_in_groups(duplicate_groups(full), lin,
                             cache_path=str(Path(out_dir) / f"_llm_{tag}_p{p_cut}_dedupe.json"))
    audit["duplicate"] = dup
    recs = [r for r in recs if r["Term"] not in dup]
    print(f"[enrichment] {tag} (p<{p_cut}): {len(struct)} structural, {len(irr)} "
          f"irrelevant, {len(dup)} duplicate -> {len(recs)} candidates", flush=True)
    return recs, audit


def register_enrichment_tools(mcp):
    """Register enrichment tools on the given FastMCP instance."""

    @mcp.tool()
    def enrich_gene_set(genes: List[str], output_dir: str,
                        libraries: Optional[List[str]] = None,
                        background_size: int = 20000) -> str:
        """Enrich one gene list; writes enrich_gene_set.csv."""
        import pandas as pd
        out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        df = _enrich_genes(genes, libraries or DEFAULT_LIBRARIES, background_size)
        if df.empty:
            return json.dumps({"status": "empty", "n_terms": 0})
        df.to_csv(out_dir / "enrich_gene_set.csv", index=False)
        return json.dumps({"status": "ok", "n_terms": int(len(df)),
                           "csv": str(out_dir / "enrich_gene_set.csv")})

    @mcp.tool()
    def enrich_shared_targets(ranked_consensus_tsv: str, output_dir: str,
                              min_shared_tfs: int = 2,
                              libraries: Optional[List[str]] = None,
                              top_terms_per_lib: int = 10,
                              background_size: int = 20000,
                              dataset_context: Optional[dict] = None,
                              lineage: Optional[dict] = None) -> str:
        """Pathways enriched in the genes bound by >= min_shared_tfs of the network's TFs.

        Outputs: shared_targets.txt, shared_targets_enrich.csv,
                 shared_targets_dotplot.json, shared_relevance_filter.json
        """
        import numpy as np
        import pandas as pd
        out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        libs = libraries or DEFAULT_LIBRARIES

        df = pd.read_csv(ranked_consensus_tsv, sep="\t")
        tf_per_gene = df.groupby("Gene Name")["TF"].nunique().reset_index(name="n_tf")
        shared = tf_per_gene[tf_per_gene["n_tf"] >= min_shared_tfs]["Gene Name"].tolist()
        (out_dir / "shared_targets.txt").write_text("\n".join(shared))
        if not shared:
            return json.dumps({"status": "empty", "n_shared": 0,
                               "shared_genes_path": str(out_dir / "shared_targets.txt")})

        res_df = _enrich_genes(shared, libs, background_size)
        if res_df.empty:
            return json.dumps({"status": "empty", "n_shared": len(shared)})
        res_df.to_csv(out_dir / "shared_targets_enrich.csv", index=False)

        lin = lineage if lineage is not None else resolve_lineage(
            dataset_context, str(out_dir / "_llm_lineage.json"))
        work = res_df.copy()
        work["Term"] = work["Term"].map(_clean_term)
        work["hits"] = work["Overlap"].map(_hits)
        # the same floor as the heatmap: a panel of three or four rows is not worth the space,
        # and a domain whose targets share little needs the looser cut to say anything at all
        recs, audit = _usable_terms(work, lin, dataset_context, out_dir, "shared",
                                    floor=MIN_ROWS)
        (out_dir / "shared_relevance_filter.json").write_text(json.dumps(audit, indent=2))
        if not recs:
            return json.dumps({"status": "empty", "n_shared": len(shared),
                               "n_terms_total": int(len(res_df))})

        by_term = {r["Term"]: r for r in recs}
        rows = []
        for lib in dict.fromkeys(r["Library"] for r in recs):
            sub = [r for r in recs if r["Library"] == lib]
            sub.sort(key=lambda r: r["best_p"])
            rows += sub[:top_terms_per_lib]
        # keep every library present, so none vanishes from the legend
        for lib in dict.fromkeys(r["Library"] for r in recs):
            have = [r for r in rows if r["Library"] == lib]
            spare = [r for r in recs if r["Library"] == lib and r not in rows]
            while len(have) < MIN_PER_LIB and spare:
                r = spare.pop(0); rows.append(r); have.append(r)
        rows.sort(key=lambda r: (r["Library"], -r["best_p"]))

        hit_of = {}
        for t, g in work.groupby("Term"):
            hit_of[t] = int(g["hits"].max())
        neg = np.array([-np.log10(max(r["best_p"], 1e-30)) for r in rows], dtype=float)
        cmap = _llm_compact_terms([r["Term"] for r in rows])
        traces = []
        for lib in dict.fromkeys(r["Library"] for r in rows):
            idx = [i for i, r in enumerate(rows) if r["Library"] == lib]
            traces.append({
                "type": "scatter", "mode": "markers",
                "x": [float(neg[i]) for i in idx],
                "y": [cmap.get(rows[i]["Term"], rows[i]["Term"]) for i in idx],
                "marker": {"size": [float(min(10.0, hit_of.get(rows[i]["Term"], 0) * 2.5 + 5))
                                    for i in idx],
                           "color": LIB_COLOR.get(lib, "#888"),
                           "line": {"width": 0.5, "color": "#333"}},
                "name": lib,
                "hovertemplate": "<b>%{y}</b><br>-log10(p)=%{x:.2f}<extra></extra>",
            })
        plot_json = {
            "data": traces,
            "layout": {
                "title": "",
                "xaxis": {"title": {"text": "-log10(p-value) \u00b7 terms at p < "
                                            + str(audit.get("p_used", PVAL_MAX)),
                                    "font": {"size": 11}},
                          "tickfont": {"size": 10}, "automargin": True,
                          "range": [max(0.0, float(neg.min()) - 0.4), float(neg.max()) + 0.6]},
                "yaxis": {"automargin": True, "tickfont": {"size": 10}},
                "height": max(420, 24 * len(rows) + 90),
                "margin": {"l": 16, "r": 24, "t": 12, "b": 78},
                "showlegend": True,
                "legend": {"orientation": "h", "x": 0.5, "xanchor": "center",
                           "y": -0.14, "yanchor": "top", "font": {"size": 10},
                           "bgcolor": "rgba(255,255,255,0.6)",
                           "bordercolor": "#ddd", "borderwidth": 0},
                "plot_bgcolor": "white", "paper_bgcolor": "white",
            },
        }
        (out_dir / "shared_targets_dotplot.json").write_text(json.dumps(plot_json))
        return json.dumps({
            "status": "ok", "n_shared": len(shared), "n_terms_total": int(len(res_df)),
            "n_terms": len(rows), "lineage": lin.get("lineage", ""),
            "p_used": audit.get("p_used", PVAL_MAX),
            "shared_genes_path": str(out_dir / "shared_targets.txt"),
            "enrich_csv": str(out_dir / "shared_targets_enrich.csv"),
            "dotplot_json": str(out_dir / "shared_targets_dotplot.json"),
        })

    @mcp.tool()
    def enrich_per_tf(ranked_consensus_tsv: str, output_dir: str,
                      max_tfs: int = 0,
                      libraries: Optional[List[str]] = None,
                      top_terms_total: int = 15,
                      background_size: int = 20000,
                      dataset_context: Optional[dict] = None,
                      panel_tfs: Optional[List[str]] = None,
                      lineage: Optional[dict] = None) -> str:
        """Per-TF coverage heatmap over the whole TRN.

        The cell is the number of the row's genes that the factor binds, not a per-TF enrichment
        p-value: on a 20-gene target set that p-value is decided by whether one gene happens to
        carry the annotation, which left the matrix sparse and whole columns blank.  The count is
        the quantity the figure is read for and needs no background assumption.

        max_tfs = 0 means every TF in the network, which is what the Section-10 graph draws.
        panel_tfs, when given, are the ranked panel: they order first and are labelled as a group,
        the rest being the factors the super-enhancer layer added.

        Outputs: per_tf_enrich.csv, per_tf_heatmap.json, pertf_relevance_filter.json
        """
        import numpy as np
        import pandas as pd
        out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        libs = libraries or DEFAULT_LIBRARIES

        df = pd.read_csv(ranked_consensus_tsv, sep="\t")
        tf_top1 = df[df["rank_within_tf"] == 1].sort_values("combined_p", ascending=True)
        ranked = tf_top1["TF"].tolist()
        all_tfs = ranked + [t for t in df["TF"].unique() if t not in ranked]
        panel = [t for t in (panel_tfs or []) if t in all_tfs]
        if panel:
            rest = [t for t in all_tfs if t not in panel]
            rest.sort(key=lambda t: -df[df["TF"] == t]["Gene Name"].nunique())
            tf_order = panel + rest
        else:
            tf_order = all_tfs
        if max_tfs and max_tfs > 0:
            tf_order = tf_order[:max_tfs]
            panel = [t for t in panel if t in tf_order]

        # upper-cased on both sides: Enrichr returns its hit genes upper-cased whatever the
        # query looked like, so comparing them against the run's own symbols made every cell of
        # a mouse run ("Sox2" vs "SOX2") zero, and dropped human symbols like C1orf116.
        tf2genes = {t: sorted({str(g).strip().upper()
                               for g in df[df["TF"] == t]["Gene Name"].dropna()})
                    for t in tf_order}
        all_rows = []
        for tf in tf_order:
            if len(tf2genes[tf]) < 3:
                continue
            sub = _enrich_genes(tf2genes[tf], libs, background_size)
            if sub.empty:
                continue
            sub["TF"] = tf
            all_rows.append(sub)
        if not all_rows:
            return json.dumps({"status": "empty"})
        long_df = pd.concat(all_rows, ignore_index=True)
        long_df.to_csv(out_dir / "per_tf_enrich.csv", index=False)

        lin = lineage if lineage is not None else resolve_lineage(
            dataset_context, str(out_dir / "_llm_lineage.json"))
        work = long_df.copy()
        work["Term"] = work["Term"].map(_clean_term)
        work["hits"] = work["Overlap"].map(_hits)
        recs, audit = _usable_terms(work, lin, dataset_context, out_dir, "pertf",
                                    floor=MIN_ROWS)
        (out_dir / "pertf_relevance_filter.json").write_text(json.dumps(audit, indent=2))
        if not recs:
            return json.dumps({"status": "empty", "reason": "no term survived filtering"})

        rows = recs[:max(MIN_ROWS, min(top_terms_total, len(recs)))]
        term_genes = {r["Term"]: set(r["genes"]) for r in rows}
        # every factor that can be covered should be: a blank column says nothing
        covered = lambda ts: {tf for t in ts for tf in tf_order
                              if term_genes[t] & set(tf2genes.get(tf, ()))}
        repairs = []
        pool = [r for r in recs if r["Term"] not in term_genes]
        while True:
            miss = [tf for tf in tf_order
                    if tf2genes.get(tf) and tf not in covered(list(term_genes))]
            if not miss:
                break
            tf_miss, add = None, None
            for cand_tf in miss:
                add = next((r for r in pool
                            if set(r["genes"]) & set(tf2genes.get(cand_tf, ()))), None)
                if add is not None:
                    tf_miss = cand_tf
                    break
            if add is None:
                break   # no pool term covers any of the remaining factors
            pool.remove(add)
            if len(term_genes) >= len(rows):
                spare = [t for t in term_genes
                         if covered([t]) <= covered([u for u in term_genes if u != t])]
                drop = spare[-1] if spare else min(term_genes, key=lambda t: len(covered([t])))
                term_genes.pop(drop)
                repairs.append(f"{tf_miss} uncovered -> +{add['Term']!r} -{drop!r}")
            else:
                repairs.append(f"{tf_miss} uncovered -> +{add['Term']!r}")
            term_genes[add["Term"]] = set(add["genes"])
        if repairs:
            audit["coverage_repairs"] = repairs
            (out_dir / "pertf_relevance_filter.json").write_text(json.dumps(audit, indent=2))

        terms = sorted(term_genes,
                       key=lambda t: (-len(covered([t])), -len(term_genes[t]), t))
        mat = np.array([[len(term_genes[t] & set(tf2genes.get(tf, ()))) for tf in tf_order]
                        for t in terms], dtype=float)

        # a blank category opens the gap between the two groups; its cells are null so nothing
        # is drawn there, and it carries no tick
        GAP = " "
        n_panel = len(panel)
        if n_panel and n_panel < len(tf_order):
            x_disp = tf_order[:n_panel] + [GAP] + tf_order[n_panel:]
            z_disp = [list(r[:n_panel]) + [None] + list(r[n_panel:]) for r in mat.tolist()]
            n_se = len(tf_order) - n_panel
            notes = [
                {"text": "Core TFs", "x": (n_panel - 1) / 2, "xref": "x", "y": 1.045,
                 "yref": "paper", "yanchor": "bottom", "showarrow": False,
                 "xanchor": "center", "font": {"size": 11, "color": "#444"}},
                {"text": "Super-enhancer TFs", "x": n_panel + 1 + (n_se - 1) / 2, "xref": "x",
                 "y": 1.045, "yref": "paper", "yanchor": "bottom", "showarrow": False,
                 "xanchor": "center", "font": {"size": 11, "color": "#444"}},
            ]
        else:
            x_disp, z_disp, notes = list(tf_order), mat.tolist(), []

        notes = list(notes) + [
            {"text": "rows: terms at p &lt; " + str(audit.get("p_used", PVAL_MAX)),
             "x": 0, "xref": "paper", "xanchor": "left",
             "y": -0.16, "yref": "paper", "yanchor": "top",
             "showarrow": False, "font": {"size": 9, "color": "#888"}},
        ]
        cmap = _llm_compact_terms(terms)
        heatmap = {
            "data": [{
                "type": "heatmap", "z": z_disp, "x": x_disp,
                "y": [cmap.get(t, t) for t in terms],
                "colorscale": [[0.0, "#ffffff"], [0.5, "#fcae91"], [1.0, "#cb181d"]],
                "colorbar": {"title": {"text": "genes bound", "side": "right"},
                             "thickness": 10, "len": 0.5, "x": 1.02,
                             "tickfont": {"size": 9}, "titlefont": {"size": 10}},
                "hovertemplate":
                    "<b>%{y}</b><br>TF=%{x}<br>genes bound=%{z:.0f}<extra></extra>",
            }],
            "layout": {
                "title": "",
                # the group labels name the axis, so it needs no title of its own; the grid is
                # off because it would otherwise show through the null spacer column
                "xaxis": {"title": {"text": ""}, "showgrid": False, "zeroline": False,
                          "tickangle": -90, "automargin": True, "tickmode": "array",
                          "tickvals": list(tf_order), "ticktext": list(tf_order),
                          "tickfont": {"size": 9}},
                "yaxis": {"automargin": True, "tickfont": {"size": 10},
                          "showgrid": False, "zeroline": False},
                "annotations": notes,
                "height": max(440, 22 * len(terms) + 160),
                "margin": {"l": 200, "r": 20, "t": 52 if notes else 15, "b": 120},
                "plot_bgcolor": "white", "paper_bgcolor": "white",
            },
        }
        (out_dir / "per_tf_heatmap.json").write_text(json.dumps(heatmap))
        return json.dumps({
            "status": "ok", "tfs": tf_order, "n_tfs": len(tf_order),
            "n_panel_tfs": n_panel, "n_terms": len(terms),
            "lineage": lin.get("lineage", ""),
            "fill": round(float((mat >= 1).mean()), 3),
            "p_used": audit.get("p_used", PVAL_MAX),
            "long_csv": str(out_dir / "per_tf_enrich.csv"),
            "heatmap_json": str(out_dir / "per_tf_heatmap.json"),
        })

    @mcp.tool()
    def build_enrichment_section(ranked_consensus_tsv: str, output_dir: str,
                                 min_shared_tfs: int = 2, max_tfs: int = 0,
                                 libraries: Optional[List[str]] = None,
                                 top_terms_per_lib: int = 10, top_terms_total: int = 15,
                                 dataset_context: Optional[dict] = None,
                                 panel_tfs: Optional[List[str]] = None) -> str:
        """Both Section-11 figures.  The lineage is resolved once and shared by the two."""
        out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        lin = resolve_lineage(dataset_context, str(out_dir / "_llm_lineage.json"))
        s = json.loads(enrich_shared_targets(
            ranked_consensus_tsv, output_dir, min_shared_tfs=min_shared_tfs,
            libraries=libraries, top_terms_per_lib=top_terms_per_lib,
            dataset_context=dataset_context, lineage=lin))
        p = json.loads(enrich_per_tf(
            ranked_consensus_tsv, output_dir, max_tfs=max_tfs, libraries=libraries,
            top_terms_total=top_terms_total, dataset_context=dataset_context,
            panel_tfs=panel_tfs, lineage=lin))
        return json.dumps({
            "status": "ok" if (s.get("status") == "ok" or p.get("status") == "ok") else "empty",
            "lineage": lin.get("lineage", ""), "shared": s, "per_tf": p})
