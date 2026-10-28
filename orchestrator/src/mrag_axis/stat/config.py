"""Derive every constant the Section-13 research scripts hardcode, from an arbitrary MRAG run's
own output + run.toml + the production catalogues.

Read-only.  Nothing here calls AlphaGenome, Corgi or the pipeline.

Source of each rule is stated in the docstring of the function that implements it.  A rule that
cannot be satisfied returns (value=None, reason=...) rather than raising, because the node runs by
default with no on/off switch; fallback() says what the node then reports.

Difference from the prototype (config/s13cfg.py): every filesystem root is a keyword argument
whose default is the current value, and the two production modules the rules borrow (the
AlphaGenome client config and the HOMER alias table) are loaded on first use instead of at import.
Importing mrag_axis.stat must not touch /BIO1/MRAG/Agent -- stage 2c imports the package in processes that
may not have it mounted, and an import-time read makes the roots unoverridable.
"""
from __future__ import annotations
import glob, importlib.util, json, os, re, sys

AGENT_ROOT = "/BIO1/MRAG/Agent"
RUNS = f"{AGENT_ROOT}/orchestrator/webui/runs"
# The DEPLOYED copies (stage 2b): production never reads the research tree /BIO1/MRAG/corgi.
# Both are byte-identical to the research files the published numbers were measured with
# (md5 96b2e87e2344de6b85f5f4e084e8486f and 2b8b17ff52095bb2a6ecc7645f74c996, compared
# 2026-10-18), so moving the default changes no number.  mrag_axis.prep.motifs and
# mrag_axis.arm.corgi already read these two paths.
JASPAR_MEME = (f"{AGENT_ROOT}/Reference/corgi/"
               "JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt")
CORGI_CONSTANTS = f"{AGENT_ROOT}/libs/corgi_client/corgi/corgi/constants.py"


def runs_dir(agent_root=AGENT_ROOT):
    return f"{agent_root}/orchestrator/webui/runs"


# ---------------------------------------------------------------- production catalogues
def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


_CACHE = {}


def agcfg(agent_root=AGENT_ROOT):
    """alphagenome_client.config -- the biosample / substitution catalogues.  It imports nothing
    from the AlphaGenome SDK, so the file can be loaded on its own without the SDK installed."""
    k = ("agcfg", agent_root)
    if k not in _CACHE:
        _CACHE[k] = _load_mod(
            "_s13_agcfg", f"{agent_root}/libs/alphagenome_client/alphagenome_client/config.py")
    return _CACHE[k]


def tfalias(agent_root=AGENT_ROOT):
    """homer_mcp.tools.tf_alias -- production's own motif-name rule, imported rather than
    re-implemented so the package cannot drift from what the pipeline actually did."""
    k = ("tfa", agent_root)
    if k not in _CACHE:
        for p in (f"{agent_root}/mcps/homer_mcp/src", f"{agent_root}/mcps/combined_mcp/src"):
            if p not in sys.path:
                sys.path.insert(0, p)
        from homer_mcp.tools import tf_alias as TFA
        _CACHE[k] = TFA
    return _CACHE[k]


# ---------------------------------------------------------------- run basics
def run_output(run: str, agent_root=AGENT_ROOT) -> str:
    """RUN.  Rule: the orchestrator's own output_dir for the run being reported.
    Taken from run.toml `output_dir` when present, else <run_dir>/output."""
    rd = run if os.path.isdir(run) else f"{runs_dir(agent_root)}/{run}"
    if os.path.basename(rd) == "output":
        return rd
    t = run_toml(rd)
    od = t.get("output_dir")
    return od if od and os.path.isdir(od) else f"{rd}/output"


def run_toml(run_dir: str) -> dict:
    p = f"{run_dir}/run.toml"
    if not os.path.exists(p) and os.path.basename(run_dir) == "output":
        p = f"{os.path.dirname(run_dir)}/run.toml"
    if not os.path.exists(p):
        return {}
    try:
        import tomllib
        return tomllib.load(open(p, "rb"))
    except Exception:
        try:
            import toml
            return toml.load(p)
        except Exception:
            return _toml_scalars(p)


def _toml_scalars(p):
    """Last-resort scalar reader (top-level key = "value" / [sec] key = "value")."""
    out, sec = {}, None
    for l in open(p):
        l = l.rstrip("\n")
        m = re.match(r"^\[([^\]\[]+)\]\s*$", l)
        if m:
            sec = m.group(1); out.setdefault(sec, {}); continue
        m = re.match(r'^([A-Za-z_][\w]*)\s*=\s*"([^"]*)"\s*$', l)
        if m:
            (out[sec] if sec else out)[m.group(1)] = m.group(2)
    return out


def arms(run_dir: str, agent_root=AGENT_ROOT):
    """(treatment, control) arm labels.  Rule: conditions[0], conditions[1] of run.toml;
    conditions_meta.json in the output is the same list and is used when run.toml is gone."""
    t = run_toml(run_dir)
    c = t.get("conditions") or []
    labs = [x.get("label") for x in c] if c and isinstance(c[0], dict) else []
    if not labs:
        p = f"{run_output(run_dir, agent_root)}/conditions_meta.json"
        if os.path.exists(p):
            d = json.load(open(p))
            labs = [x.get("label") for x in (d if isinstance(d, list) else d.get("conditions", []))]
    return (labs + [None, None])[:2]


def manifest(out: str) -> dict:
    """{(step, key): path} from the run's own manifest.json -- the orchestrator records the real
    path of every node output, so no filename has to be guessed where it is recorded."""
    p = f"{out}/manifest.json"
    if not os.path.exists(p):
        return {}
    try:
        d = json.load(open(p))
    except Exception:
        return {}
    m = {}
    for st in d.get("steps") or []:
        nm = st.get("name")
        for o in st.get("outputs") or []:
            pa = o.get("path")
            if isinstance(pa, str) and pa.startswith("/") and o.get("key"):
                m[(nm, o["key"])] = pa
    return m


def _from_manifest(out, step, key, pattern):
    """Manifest path when recorded and present on disk, else the glob."""
    p = manifest(out).get((step, key))
    if p and os.path.exists(p):
        return p, f"manifest[{step}.{key}]"
    g = sorted(glob.glob(f"{out}/{pattern}"))
    if len(g) == 1:
        return g[0], f"glob {pattern}"
    return None, f"manifest[{step}.{key}] absent and {len(g)} matches for {pattern}"


# ---------------------------------------------------------------- ROSE
def rose_all(out: str):
    """ROSE_ALL.  Rule: manifest.json step 'se_rose' output key 'all_table'; if the manifest is
    missing, the single file matching se_rose/rose/*_AllEnhancers.table.txt."""
    return _from_manifest(out, "se_rose", "all_table",
                          "se_rose/rose/*_AllEnhancers.table.txt")


def rose_signal_col(path: str):
    """The ROSE signal column.  Rule: the name on the table's own '#Ranked by <column>' header
    line.  Fallback: the one data column that is not a fixed ROSE field.  (Research hardcoded
    endswith('.bam'), later or '_bigwig_signal'.)"""
    FIXED = {"REGION_ID", "CHROM", "START", "STOP", "NUM_LOCI",
             "CONSTITUENT_SIZE", "enhancerRank", "isSuper"}
    ranked, header = None, None
    with open(path) as fh:
        for l in fh:
            if l.startswith("#Ranked by "):
                ranked = l.strip()[len("#Ranked by "):].strip()
            elif not l.startswith("#"):
                header = [c.strip() for c in l.rstrip("\n").split("\t")]
                break
    if header is None:
        return None, "no header row"
    rest = [c for c in header if c not in FIXED]
    if ranked and ranked in header:
        return ranked, None
    if len(rest) == 1:
        return rest[0], "'#Ranked by' absent or not a column; used the single non-ROSE column"
    return None, f"cannot identify signal column (candidates {rest})"


def se10(out: str):
    """SE10.  Rule: manifest.json step 'se_rose' output key 'se_top_tsv'; fallback glob
    se_rose/se_top*.tsv.  The name is se_top<top_n>.tsv and top_n is a node PARAMETER
    (atac_mcp.tools.super_enhancers), so hardcoding 'se_top10' breaks on top_n != 10."""
    return _from_manifest(out, "se_rose", "se_top_tsv", "se_rose/se_top*.tsv")


def peak_table(out: str):
    """The HOMER peak table.  Rule: the single se_trn/*_peak_minmax.TFs.txt (the .Pval2 and
    _peak_matrix variants are different files and do not match)."""
    g = sorted(glob.glob(f"{out}/se_trn/*_peak_minmax.TFs.txt"))
    if len(g) == 1:
        return g[0], "glob se_trn/*_peak_minmax.TFs.txt"
    return None, f"{len(g)} matches for se_trn/*_peak_minmax.TFs.txt"


# ---------------------------------------------------------------- network
_NORM = lambda s: re.sub(r"[^A-Z0-9_\-]", "_", str(s).upper().replace("::", "_").replace(":", "_"))


def net_edges(out: str):
    """net_edges_*.json.  Rule: read se_trn/se_cyjs.json; nodes are the dicts without 'source',
    edges the ones with.  tfs = type=='TF' node names, targets = type=='Target',
    tf_target = kind=='TF_Target', se_tf = kind=='SE_TF'.  TF names normalised with the same
    composite form production uses ('Fox:Ebox' -> 'FOX_EBOX')."""
    p = manifest(out).get(("se_trn", "se_cyjs_json")) or f"{out}/se_trn/se_cyjs.json"
    if not os.path.exists(p):
        return None, "se_trn/se_cyjs.json missing"
    els = json.load(open(p))["elements"]
    N = [e["data"] for e in els if "source" not in e["data"]]
    E = [e["data"] for e in els if "source" in e["data"]]
    # A TF_Target edge names its target by node ID, while "targets" above (and therefore the
    # edits' gene column and the statistic's grid) is keyed by node NAME.  The two differ for a
    # target gene that is ALSO a panel TF: cytoscape_trn.py gives that Target node the id
    # '<gene>__target' and keeps its name '<gene>', so the edge read '... -> MYC__target' and
    # stats._grid, which requires the gene to be in the frame's gene column, dropped the triple --
    # the axis never reached the figure, the tab counts or the target edits' claimed factors.
    # Only the Target end is mapped: an SE_TF edge's source is the SE node's ID ('SE1'), which is
    # the SE label every edit and every ROSE row uses, while that node's NAME carries its
    # coordinates ('SE1 | chr11:...'), so both ends keep the key their consumers use.
    tgt_name = {n.get("id"): n.get("name", n.get("id"))
                for n in N if n.get("type") == "Target" and n.get("id") is not None}
    return {
        "tfs": [_NORM(n["name"]) for n in N if n.get("type") == "TF"],
        "targets": [n["name"] for n in N if n.get("type") == "Target"],
        "tf_target": [[_NORM(e["source"]), tgt_name.get(e["target"], e["target"])]
                      for e in E if e.get("kind") == "TF_Target"],
        "se_tf": [[e["source"], _NORM(e["target"])] for e in E if e.get("kind") == "SE_TF"],
    }, None


def trn_tf_names(out: str):
    """Every TRN factor, before the PWM filter.  Rule: se_cyjs TF nodes (above).
    se_trn/se_anno.txt rows with type=='TF' carry the same set already normalised and are used as
    a cross-check / fallback."""
    ne, why = net_edges(out)
    if ne:
        return ne["tfs"], None
    p = f"{out}/se_trn/se_anno.txt"
    if os.path.exists(p):
        names, hdr = [], None
        for i, l in enumerate(open(p)):
            f = l.rstrip("\n").split("\t")
            if i == 0:
                hdr = f; continue
            r = dict(zip(hdr, f))
            if r.get("type") == "TF":
                names.append(_NORM(r["name"]))
        return names, "from se_anno.txt (se_cyjs.json absent)"
    return None, why


# ---------------------------------------------------------------- JASPAR / PWMs
def jaspar_names(jaspar=JASPAR_MEME):
    """The MOTIF names of one JASPAR MEME file, upper-cased.  Cached per path."""
    k = ("jaspar", jaspar)
    if k not in _CACHE:
        s = set()
        for l in open(jaspar):
            if l.startswith("MOTIF"):
                p = l.split(None, 2)
                s.add((p[2] if len(p) > 2 else p[1]).strip().upper())
        _CACHE[k] = s
    return _CACHE[k]


def split_pwm(tf_names, jaspar=JASPAR_MEME):
    """TRN_TFS / NO_PWM.  Rule: a factor is usable iff its normalised symbol is the name of a
    JASPAR2024 CORE vertebrates non-redundant MOTIF (case-insensitive exact); everything else is
    NO_PWM.  Composite motif nodes ('FOX_EBOX') never match."""
    J = jaspar_names(jaspar)
    keep = [t for t in tf_names if t.upper() in J]
    drop = [t for t in tf_names if t.upper() not in J]
    return keep, drop


# ---------------------------------------------------------------- HOMER column map
def tf_cols(peak_tbl: str, tf_names, agent_root=AGENT_ROOT):
    """TF_COLS.  Rule: for each TRN factor T, every motif column of the peak table (columns 8..n,
    i.e. everything after Gene Name) whose motif BASE NAME resolves to T under symbol-spelling
    rules only -- case-insensitive equality, HOCOMOCO short name
    (combined_mcp.tools.pipeline._hocomoco_to_symbol: ANDR->AR, GCR->NR3C1, AP2A->TFAP2A,
    PKNX1->PKNOX1) and the ZNxxx->ZNFxxx expansion.  HOMER response-element / legacy names (ARE,
    GRE, AR-halfsite) are NOT followed, and composite columns (FOXA1:AR, Fox:Ebox, TFAP4::ETV1)
    are dropped, so every knockout stays attributable to one factor.
    Base name = text before the first '(' and before '/'.
    """
    TFA = tfalias(agent_root)
    hdr = open(peak_tbl).readline().rstrip("\n").split("\t")
    cols = hdr[7:]
    out = {}
    for T in tf_names:
        bases = []
        for c in cols:
            base = TFA.motif_base(c)
            if len([t for t in re.split(r"::|:", base) if t.strip()]) > 1:
                continue                      # composite motif
            if TFA._hoco(base.upper()) == T.upper() and base not in bases:
                bases.append(base)
        if bases:
            out[T] = bases
    return out


def name_map(tf_names, no_pwm, agent_root=AGENT_ROOT):
    """The NAME map.  Rule: the alias half is production's HOMER_NAME_ALIASES restricted to the
    aliases that actually occur for this run's factors (GR/GRE -> NR3C1 whenever NR3C1 is a TRN
    factor); the null half is exactly NO_PWM (FOX_EBOX, FOXM1, ... -> None)."""
    TFA = tfalias(agent_root)
    m = {}
    up = {t.upper() for t in tf_names}
    for alias, sym in TFA.HOMER_NAME_ALIASES.items():
        if sym in up and alias != sym and "-" not in alias and "." not in alias:
            m[alias] = sym
    for t in no_pwm:
        m[t] = None
    return m


# ---------------------------------------------------------------- lineage-foreign null
# Panels of factors that are master regulators of a lineage OTHER than the run's.  The research set
# is the "prostate" row.  Keys are matched against the same free text production's
# best_substitute() uses (cell_line + tissue + cancer_type).
FOREIGN_PANEL = {
    "prostate":   ["HNF4A", "PAX5", "MYOD1", "TBX5", "SPI1", "SOX10", "CDX2", "POU5F1"],
    "breast":     ["HNF4A", "PAX5", "MYOD1", "TBX5", "SPI1", "SOX10", "CDX2", "POU5F1"],
    "lung":       ["HNF4A", "PAX5", "MYOD1", "TBX5", "SPI1", "SOX10", "CDX2", "POU5F1"],
    "liver":      ["PAX5", "MYOD1", "TBX5", "SPI1", "SOX10", "POU5F1", "GATA3", "NEUROD1"],
    "colorectal": ["HNF4A", "PAX5", "MYOD1", "TBX5", "SPI1", "SOX10", "POU5F1", "NEUROD1"],
    "myeloid/leukaemia": ["HNF4A", "MYOD1", "TBX5", "SOX10", "CDX2", "POU5F1", "FOXA1", "AR"],
    "lymphoid":   ["HNF4A", "MYOD1", "TBX5", "SOX10", "CDX2", "POU5F1", "FOXA1", "AR"],
}
# "CDX2 is intestinal" is only a valid null for a non-intestinal run, so the panel for a lineage
# must never contain a factor of that lineage; that is enforced in neg_tfs().
LINEAGE_OWN = {
    "prostate": {"AR", "FOXA1", "NKX3-1", "HOXB13"},
    "liver": {"HNF4A", "FOXA1", "FOXA2", "FOXA3", "HNF1A", "CEBPA"},
    "colorectal": {"CDX2", "CDX1", "HNF4A", "TCF7L2"},
    "myeloid/leukaemia": {"SPI1", "CEBPA", "RUNX1", "GATA1"},
    "lymphoid": {"PAX5", "EBF1", "POU2AF1", "SPI1"},
    "breast": {"ESR1", "GATA3", "FOXA1", "TFAP2C"},
    "lung": {"NKX2-1", "FOXA1", "FOXA2", "SOX2"},
}


def lineage(run_dir: str, agent_root=AGENT_ROOT):
    """The run's lineage label.  Rule: production's own SUBSTITUTE_BY_LINEAGE keyword table applied
    to cell_line + tissue + cancer_type from run.toml -- the same free text and the same
    token-boundary matcher the AlphaGenome substitution uses."""
    A = agcfg(agent_root)
    t = run_toml(run_dir)
    hay = " ".join(str(t.get(k) or "").lower() for k in ("cell_line", "tissue", "cancer_type"))
    for label, keys, _sub in A.SUBSTITUTE_BY_LINEAGE:
        if any(A._lineage_hit(hay, k) for k in keys):
            return label, hay.strip()
    return None, hay.strip()


def neg_tfs(run_dir: str, trn_tfs, override=None, agent_root=AGENT_ROOT, jaspar=JASPAR_MEME):
    """NEG_TFS.  Rule: config override if given, else FOREIGN_PANEL[lineage(run)], minus anything
    that is a factor of this run's own lineage or already in the TRN, kept only when it has a
    JASPAR PWM.  No lineage match -> no panel (see fallback())."""
    if override:
        cand, src = list(override), "run config"
    else:
        lab, hay = lineage(run_dir, agent_root)
        if lab is None or lab not in FOREIGN_PANEL:
            return None, f"no curated foreign panel for lineage={lab!r} (text {hay!r})"
        cand, src = FOREIGN_PANEL[lab], f"FOREIGN_PANEL[{lab}]"
        own = LINEAGE_OWN.get(lab, set())
        cand = [c for c in cand if c not in own]
    J = jaspar_names(jaspar)
    up = {t.upper() for t in trn_tfs}
    out = [c for c in cand if c.upper() in J and c.upper() not in up]
    return out, src


# ---------------------------------------------------------------- AlphaGenome read-outs
def ag_biosamples(run_dir: str, need=("CHIP_HISTONE",), agent_root=AGENT_ROOT):
    """The AlphaGenome ontology terms.  Rule: every biosample in the run's lineage group that
    carries all of `need`, taken from the production catalogues in alphagenome_client.config
    (PROSTATE_CELL_LINES is the only curated group; for any other lineage the group is the single
    best_substitute()).  need=('CHIP_HISTONE',) is what the axis read-outs require at the
    enhancer end."""
    A = agcfg(agent_root)
    t = run_toml(run_dir)
    lab, _ = lineage(run_dir, agent_root)
    if lab == "prostate":
        group = dict(A.PROSTATE_CELL_LINES)
        why = "PROSTATE_CELL_LINES"
    else:
        sub, reason = A.best_substitute(t.get("cell_line"), t.get("tissue"),
                                        t.get("cancer_type"), need=need)
        group = {sub: A.CELL_LINES.get(sub)}
        why = f"best_substitute -> {sub} ({reason})"
    keep = {b: c for b, c in group.items()
            if all(A.MODALITY_COVERAGE.get(b, {}).get(m) for m in need)}
    return {"biosamples": keep, "terms": sorted(set(keep.values())), "source": why}


def ag_outputs(run_dir: str, bios=None, agent_root=AGENT_ROOT):
    """The requested AlphaGenome output types.  Rule: the modalities at least one chosen biosample
    covers, intersected with the axis read-outs' needs (DNASE, ATAC, CHIP_HISTONE, RNA_SEQ, CAGE;
    CHIP_TF only when a Pol II read-out is on)."""
    A = agcfg(agent_root)
    WANT = ["DNASE", "ATAC", "CHIP_HISTONE", "RNA_SEQ", "CAGE"]
    b = bios or ag_biosamples(run_dir, agent_root=agent_root)
    cov = [A.MODALITY_COVERAGE.get(x, {}) for x in b["biosamples"]]
    return [m for m in WANT if any(c.get(m) for c in cov)]


# ---------------------------------------------------------------- Corgi
def corgi_channels(run_dir=None, constants=CORGI_CONSTANTS):
    """The Corgi channel maps (CH / TRACKS / STRAND_PAIRS) and the window geometry.

    Rule: read corgi.constants.DEFAULT_TRACK_NAMES -- the checkpoint's own assay order -- and index
    into it by assay name.  A stranded assay is any pair of adjacent '<name>_plus' / '<name>_minus'
    entries, which gives STRAND_PAIRS and the two-channel read-outs (cage, rna_total, rna_polya).
    WINDOW/STRIDE/BIN_SIZE/CENTRAL_BINS/FLANK come from the same file.  Nothing here depends on the
    run, and nothing here loads a checkpoint: it reads constants only, never the model.
    """
    k = ("corgi", constants)
    if k not in _CACHE:
        _CACHE[k] = _load_mod("_s13_corgiconst", constants)
    K = _CACHE[k]
    names = list(K.DEFAULT_TRACK_NAMES)
    pairs, tracks = [], {}
    for i, n in enumerate(names):
        if n.endswith("_plus") and i + 1 < len(names) and names[i + 1] == n[:-5] + "_minus":
            pairs.append((i, i + 1))
            tracks[n[:-5]] = [i, i + 1]
        elif not (n.endswith("_plus") or n.endswith("_minus")):
            tracks[n] = [i]
    return {"CH": {k2: v[0] for k2, v in tracks.items() if len(v) == 1},
            "TRACKS": tracks, "STRAND_PAIRS": pairs, "n_channels": len(names),
            "WINDOW": K.WINDOW_SIZE, "HALF": K.WINDOW_SIZE // 2, "FLANK": K.FLANK,
            "NBIN": K.CENTRAL_BINS, "BIN": K.BIN_SIZE}


def expr_context(run_dir: str, agent_root=AGENT_ROOT):
    """The expression-context name.  Rule: '<cell_line>_<treatment arm>' is only a label; what the
    model needs is the 2891-gene trans-regulator vector for the treatment arm.  Derivable name: the
    treatment arm label from run.toml; derivable vector: the arm's own RNA projected onto Corgi's
    trans_regulators_final_hgnc.txt (what /BIO1/MRAG/corgi/work/B/build_expr.py does).  Returns the
    name and whether the run carries RNA to build the vector from."""
    t = run_toml(run_dir)
    tre, _ctl = arms(run_dir, agent_root)
    cl = t.get("cell_line")
    has_rna = bool((t.get("precomputed") or {}).get("rna_log2_matrix")) or \
        any((s.get("modality") or "").upper() == "RNA" for s in (t.get("samples") or []))
    return {"arm": tre, "cell_line": cl,
            "label": f"{cl}:{tre}" if cl and tre else (tre or cl),
            "rna_available": has_rna}


# ---------------------------------------------------------------- gates / exclusions
GATES = {
    "min_foreign_per_region": 4,     # axis.py / axis_models*.py / axis_v*.py: len(frn) < 4 -> skip
    "min_n_for_mannwhitney": 3,      # agstats.mannwhitney
    "min_n_for_wilcoxon": 5,         # agstats.wilcoxon_vs_zero
    "min_ordinary_for_spearman": 21,  # seko2: len(o) > 20
    "min_unassigned_for_test2": 4,   # seko2: len(b) > 3
}


def excl_pairs(edits_json: str, bin_bp: int = 128):
    """The pairs the control-edit rule drops, stated as a boundary distance.

    This is the config-side statement of the rule: a pair is excluded when a control edit lands
    within `bin_bp` of a read-out interval boundary.  mrag_axis.stat.stats.ctl_in_readout_bin() is the
    implementation the statistic uses, and it is the one to call: it asks whether the control edit
    falls in a bin the read-out actually SUMS, on a 128-bp window-relative grid.  Both reproduce
    the OC2 hand list; the stats form is the one measured on all four domains.
    """
    ed = json.load(open(edits_json))
    W = int(ed["window"])
    out = set()
    for it in ed["items"]:
        ws = it["ws"]
        bnds = []
        for _wn, ivs in it["windows"].items():
            for a, b in ivs:
                bnds += [a - ws, b - ws]
        for tf, v in it["tfs"].items():
            for pos, bases in v["ctl"]:
                if any(abs(pos - x) < bin_bp or abs(pos + len(bases) - x) < bin_bp for x in bnds):
                    out.add((it["label"], tf)); break
    return sorted(out), W


# ---------------------------------------------------------------- default-run fallback
def fallback(missing: str) -> str:
    """What the node reports when something cannot be derived (it has no on/off switch)."""
    return {
        "neg_tfs": "run the axis test with NO foreign-factor null: report every read-out effect "
                   "and the SE-vs-ordinary and assigned-vs-unassigned contrasts, and label the "
                   "axis count 'uncalibrated - no lineage-foreign null for this lineage'. Do not "
                   "print a false-positive rate.",
        "ag_terms": "fall back to the single run.toml [alphagenome] tissue term (what the old "
                    "Section 13 already uses) and mark the AlphaGenome arm 'single-biosample'; "
                    "keep the Corgi arm complete.",
        "rose_all": "skip the enhancer end, keep the promoter/target end, and say the SE layer is "
                    "unavailable because ROSE produced no AllEnhancers table.",
        "peak_table": "fall back to the SE-level items only (se_top10 + se_cyjs edges); the "
                      "promoter layer needs the HOMER peak table.",
        "expr_context": "no RNA in the run -> use the unconditioned/mean expression vector and "
                        "mark the Corgi arm 'context-free'.",
        "net_edges": "there is no TRN in this run, so there are no SE-TF-target axes to validate: "
                     "the section reports 0 axes and states this reason.",
    }.get(missing, "emit the section with the missing input named and the dependent numbers "
                   "withheld.")
