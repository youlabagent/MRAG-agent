"""Literature-corpus domains for the PaperQA vector store (2026-09-15).

The PaperQA collection tags every chunk with the corpora it belongs to
(payload "cancer", a string or a list). Target-gene evidence scoring used to
search only "pancancer"; a run now resolves one or more domains and searches
their union (+ "pancancer"). Several domains can be combined, e.g. PC + NEC +
SCLC (PI request).

Resolution order: explicit `literature_domains` -> cancer_type -> tissue ->
the historical prostate default (only when nothing at all was given). Items
that match no corpus are returned so the caller can report or re-ask.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable, Optional

CORPORA: dict[str, str] = {
    "prostate": "Prostate Cancer", "nec": "Neuroendocrine Carcinoma", "sclc": "Small Cell Lung Cancer",
    "nsclc": "Non-Small Cell Lung Cancer", "breast": "Breast Cancer", "colorectal": "Colorectal Cancer",
    "pancreatic": "Pancreatic Cancer", "melanoma": "Melanoma", "glioblastoma": "Glioblastoma / Glioma",
    "aml": "Acute Myeloid Leukemia", "ovarian": "Ovarian Cancer", "hcc": "Hepatocellular Carcinoma",
    "gastric": "Gastric Cancer", "bladder": "Bladder Cancer", "rcc": "Renal Cell Carcinoma",
    "headneck": "Head and Neck Cancer", "esophageal": "Esophageal Cancer", "lymphoma": "Lymphoma",
    "pancancer": "Pan-Cancer",
}
# short names a user may type as a whole item
ALIASES = {"pc": ["prostate"], "pca": ["prostate"], "ne": ["nec"], "nepc": ["prostate", "nec"],
           "crc": ["colorectal"], "gbm": ["glioblastoma"], "hnscc": ["headneck"], "pdac": ["pancreatic"]}

_DASH = re.compile("[‐‑‒–—―−­]")
S = r"[- ]?"                                     # optional hyphen / space (after normalisation)

_NSCLC = re.compile(rf"non{S}small{S}cell|\bNSCLC\b|lung adeno|lung squamous|\bLUAD\b|\bLUSC\b", re.I)
_RULES: list[tuple[str, tuple[str, ...]]] = [
    (rf"\bNEPC\b|neuro{S}endocrine prostat|prostat\w*(cancer|carcinoma)? with neuro{S}endocrine"
     rf"|treatment{S}(emergent|induced|related) (small{S}cell|neuro{S}endocrine)"
     rf"|\bCRPC{S}NE\b|\bNE{S}CRPC\b|\bSCN?PC\b|\bt{S}SCNC\b|\bAVPC\b|aggressive{S}variant prostat"
     rf"|small{S}cell (carcinoma )?(of the )?prostat|prostat\w* small{S}cell", ("prostate", "nec")),
    (r"prostat|\bm?CRPC\b|\bPRAD\b", ("prostate",)),
    (rf"neuro{S}endocrine|\bNECs?\b|\bLCNEC\b|merkel|small{S}cell carcinoma(?! of the lung)", ("nec",)),
    (rf"small{S}cell lung|small{S}cell carcinoma of the lung|\bSCLC\b", ("sclc",)),
    (r"breast|mammary|\bTNBC\b|\bBRCA\b(?!\s?[12])", ("breast",)),
    (r"colorectal|\bcolon\b|\brectal\b|\bCRC\b|\bCOAD\b|\bREAD\b", ("colorectal",)),
    (rf"pancrea(s|tic){S}(ductal|cancer|adeno|carcinoma|tumou?r)|\bPDAC\b|\bPAAD\b", ("pancreatic",)),
    (r"melanoma|\bSKCM\b", ("melanoma",)),
    (r"glioblastoma|glioma|\bGBM\b|\bLGG\b", ("glioblastoma",)),
    (r"acute myeloid|myeloid leuk|\bAML\b|\bLAML\b", ("aml",)),
    (r"ovar|\bHGSOC\b", ("ovarian",)),
    (r"hepatocellular|(liver|hepatic) (cancer|carcinoma)|\bHCC\b|\bLIHC\b", ("hcc",)),
    (r"gastric|stomach|\bSTAD\b", ("gastric",)),
    (r"(?<!gall)\bbladder|urothelial|\bBLCA\b", ("bladder",)),
    (rf"renal{S}(cell{S})?(carcinoma|cancer|tumou?r)|kidney (cancer|carcinoma)|clear{S}cell renal"
     r"|\bc?cRCC\b|\bRCC\b|\bKIRC\b|\bKIRP\b|\bKICH\b", ("rcc",)),
    (rf"head{S}(and|&){S}neck|\bHNSCC\b|\bHNSC\b|oral (cancer|squamous|carcinoma)|\bOSCC\b", ("headneck",)),
    (r"o?esophag|\bESCA\b|\bESCC\b", ("esophageal",)),
    (r"lymphoma|\bDLBCL\b", ("lymphoma",)),
    (rf"pan{S}cancer", ("pancancer",)),
]
_RX = [(re.compile(p, re.I), keys) for p, keys in _RULES]
_PC = re.compile(r"\bPCa?\b(?!-\d)")                # "PC"/"PCa" = prostate, unless PC-3 or pancreatic
_LUNG = re.compile(r"\blung (cancer|carcinoma|tumou?rs?|neoplasms?|malignanc)", re.I)


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", _DASH.sub("-", str(text))).strip()


def _items(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = re.split(r"[,;/+|]", value)
    return [normalise(v) for v in value if normalise(v)]


def match_item(text: str) -> list[str]:
    """Corpus keys named by one item (may be several), in the order they are
    first mentioned in the text (2026-09-17: user order; was rule order)."""
    t = normalise(text); low = t.lower()
    if low in CORPORA:
        return [low]
    if low in ALIASES:
        return list(ALIASES[low])
    hit: list[str] = []; at: dict[str, int] = {}
    def add(keys, pos):
        for k in keys:
            if k not in hit:
                hit.append(k)
            at[k] = min(at.get(k, pos), pos)
    m = _NSCLC.search(t)
    if m:
        add(("nsclc",), m.start())
    # "non-small cell lung" must not also hit SCLC: blank it with equal-length
    # padding so match positions in `rest` still index the original text.
    rest = _NSCLC.sub(lambda x: " " * len(x.group(0)), t)
    for rx, keys in _RX:
        m = rx.search(rest)
        if m:
            add(keys, m.start())
    m = _PC.search(t)
    if m and "pancrea" not in low:
        add(("prostate",), m.start())
    m = _LUNG.search(rest)
    if m and "sclc" not in hit and "nsclc" not in hit:
        add(("sclc", "nsclc"), m.start())          # generic "lung cancer": both lung corpora (historic order)
    return sorted(hit, key=lambda k: at[k])        # stable: a tie keeps rule order


def _resolve(items: list[str]) -> tuple[list[str], list[str]]:
    keys: list[str] = []; unmatched: list[str] = []
    for it in items:
        m = match_item(it)
        if m:
            keys.extend(k for k in m if k not in keys)
        else:
            unmatched.append(it)
    return keys, unmatched


def resolve_literature_domains(explicit: Optional[Iterable[str] | str] = None,
                               cancer_type: Optional[str] = None,
                               tissue: Optional[str] = None,
                               fallback: str = "prostate") -> tuple[list[str], list[str]]:
    """Return (corpus keys to search, input items that matched no corpus).

    explicit -> cancer_type -> tissue; a source is used only if it names at
    least one corpus, so unrecognised explicit entries fall back to cancer_type
    instead of silently narrowing the search. `fallback` applies only when
    nothing was given at all. "pancancer" is always included, last; the other
    keys keep the order in which the user named them."""
    unmatched: list[str] = []; keys: list[str] = []
    given = False
    for source in (explicit, cancer_type, tissue):
        items = _items(source)
        if not items:
            continue
        given = True
        k, u = _resolve(items)
        unmatched.extend(u)
        if [x for x in k if x != "pancancer"] or k:
            keys = k
            break
    if not keys and not given:
        keys, _ = _resolve(_items(fallback))
    if "pancancer" not in keys:
        keys.append("pancancer")
    # User order (first mention first; _resolve deduplicates), "pancancer" last.
    # 2026-09-17: was CORPORA declaration order, which re-sorted e.g. "SCLC, PC".
    return [k for k in keys if k != "pancancer"] + ["pancancer"], unmatched


def describe(keys: Iterable[str]) -> str:
    return ", ".join(CORPORA.get(k, k) for k in keys)


# ---------------------------------------------------------------------------
# Display names, user-domain list and per-domain text gates (2026-09-17, llmdomain)
#
# The report shows the user's literature domains as "Full name (ABBR)", in the
# user's order, without the automatically appended pan-cancer corpus: the
# Section 2 "LLM domain" row and the numbered per-domain Novelty blocks of the
# Section 14 target-gene cards. CORPORA / describe() above are unchanged: they
# are the Qdrant payload labels and the manifest "literature_domains_label"
# contract. viz_report_mcp must not import this package, so the orchestrator
# hands it plain data (domains_label() strings, domain_specs() dicts).
# ---------------------------------------------------------------------------

# key -> (full name in sentence case, official abbreviation); "" = no abbreviation.
DISPLAY: dict[str, tuple[str, str]] = {
    "prostate":     ("Prostate cancer", "PC"),
    "nec":          ("Neuroendocrine carcinoma", "NEC"),
    "sclc":         ("Small cell lung cancer", "SCLC"),
    "nsclc":        ("Non-small cell lung cancer", "NSCLC"),
    "breast":       ("Breast cancer", "BC"),
    "colorectal":   ("Colorectal cancer", "CRC"),
    "pancreatic":   ("Pancreatic ductal adenocarcinoma", "PDAC"),
    "melanoma":     ("Melanoma", "SKCM"),
    "glioblastoma": ("Glioblastoma", "GBM"),
    "aml":          ("Acute myeloid leukemia", "AML"),
    "ovarian":      ("Ovarian cancer", "OV"),
    "hcc":          ("Hepatocellular carcinoma", "HCC"),
    "gastric":      ("Gastric cancer", "GC"),
    "bladder":      ("Bladder cancer", "BLCA"),
    "rcc":          ("Renal cell carcinoma", "RCC"),
    "headneck":     ("Head and neck squamous cell carcinoma", "HNSCC"),
    "esophageal":   ("Esophageal cancer", "ESCA"),
    "lymphoma":     ("Lymphoma", ""),
    "pancancer":    ("Pan-cancer", ""),
}

_J = "[\\s\\-‐-―−]{0,3}"      # space / hyphen / unicode dash between words
_CS = "(?-i:{})".format                       # case-sensitive abbreviation inside an re.I pattern

# Text gate per corpus: a passage (title + snippet) belongs to a domain when,
# after every DOMAIN_EXCLUDE match is blanked, one DOMAIN_TERMS pattern matches
# (see mentions_domain). Python regexes, applied with re.IGNORECASE.
# NEC never matches head-and-neck / "neck"; SCLC is never satisfied by NSCLC,
# non-small cell, LUAD or LUSC.
# NEC = neuroendocrine CARCINOMA only (PI decision 2026-09-26, necca).  Counts:
# NEPC / t-NEPC / CRPC-NE, small cell carcinoma of any site (SCLC included),
# LCNEC, poorly differentiated / high-grade NE carcinoma, Merkel cell
# carcinoma, NE (trans)differentiation or transformation of a carcinoma, mixed
# NE carcinoma (MiNEN / MANEC / amphicrine).  Does NOT count on its own: NET,
# PanNET / pNET, carcinoid, well-differentiated or low/intermediate-grade NE
# neoplasms (incl. the old "well-differentiated NE carcinoma" = carcinoid
# wording), GEP-NET, pituitary NET, pheochromocytoma / paraganglioma.
# Before 2026-09-26 any "neuroendocrine" word (and any "merkel") tied a passage
# to nec, so NET-only passages grounded NEC scores (NEC run 09-26 r2: one gene
# via a PanNEN review, one via a GEP-NET atlas, two via GeneRIF lines).
# A text tie now needs one of the carcinoma-type terms below.  A bare generic
# NE word ("neuroendocrine markers / features / differentiation / lineage")
# counts only with an NE-carcinoma context: prostate cancer ANYWHERE in the
# passage (any NE word, incl. the loose "neuroendocrine tumour" wording NEPC
# papers use -- 41299174 sorts "neuroendocrine tumour cells" from NEPC models;
# "prostate-specific" antigen / membrane antigen does not count, PSMA-PET is a
# NET imaging topic), or lung / bladder / urothelial carcinoma in the SAME
# sentence (<=150 characters; NE words other than "NE tumour / neoplasm", since
# lung NE tumours include carcinoids).  A generic
# "carcinoma" is NOT such a context: "acinar cell carcinomas frequently display
# neuroendocrine differentiation" in a PanNEN review tied a gene to nec that
# way.  "NE" (upper case) counts only as the head of
# an NE compound ("NE differentiation / markers / phenotype ..."), never alone
# -- NE is also norepinephrine.  The corpus-label side lives in
# DOMAIN_CORPUS_VETO below.  Measured and hand-judged on the nec corpus.
# MRAG_NEC_RULE=legacy restores the old
# rule (domain_specs).
_NE = rf"neuro{_J}endocrine"
_NE_HEADS_NET = r"(?:tumou?rs?|neoplasms?|neoplasia|NETs?|NENs?)\b"
# a generic NE word that does not head a tumour / neoplasm noun phrase
_NE_GENERIC = (rf"(?:{_NE}(?!{_J}{_NE_HEADS_NET})"
               rf"|(?-i:\bNE\b)(?={_J}(?:differentiation|trans{_J}?differentiat\w*|transformation"
               rf"|phenotypes?|features?|markers?|lineages?|like|subtypes?|genes?|programs?"
               rf"|signatures?|plasticity|states?|cells?|morphology)\b))")
_NEC_PROSTATE_CTX = rf"(?:\bprostat(?!e{_J}specific)\w*|\bm?CRPCs?\b|(?-i:\bPCa\b)|\bcastrat\w*{_J}resistan\w*)"
_NEC_CARC_CTX = (rf"(?:{_NEC_PROSTATE_CTX}|\blung{_J}(?:cancers?|carcinomas?|adenocarcinomas?)"
                 rf"|\bLUADs?\b|\burothelial{_J}(?:cancers?|carcinomas?)"
                 rf"|\bbladder{_J}(?:cancers?|carcinomas?))")
NEC_TERMS: tuple = (
    rf"{_NE}{_J}carcinomas?",                                   # NE carcinoma (LCNEC / SCNEC / PD-NEC spelled out)
    rf"carcinomas?\s*,\s*{_NE}",                                # MeSH "Carcinoma, Neuroendocrine"
    rf"(?:carcinomas?|cancers?){_J}(?:with|showing|exhibiting|displaying){_J}(?:[a-z]+{_J}){{0,2}}?"
    rf"{_NE}{_J}(?:features?|differentiation|morphology|phenotypes?|components?|transformation"
    rf"|trans{_J}?differentiation)",                            # (adeno)carcinoma with NE differentiation
    _CS(r"\b(?:Pan|p|GEP|GI|SI|Si|Lu|Th)?-?NECs?\b"),             # NEC, PanNEC, GEP-NEC ...
    r"\bLCNECs?\b", rf"large{_J}cell{_J}{_NE}",
    r"\bNEPCs?\b", rf"\bCRPC{_J}NE\b", rf"\bNE{_J}CRPC\b",
    rf"{_NE}{_J}(?:like{_J})?(?:prostat|castrat|m?CRPC)",       # NE prostate cancer, NE CRPC
    rf"treatment{_J}(?:emergent|induced|related){_J}(?:small{_J}cell|{_NE})",
    r"\bt?-?SCNCs?\b", r"\bSCNPCs?\b", r"\bEPSCCs?\b",
    rf"small{_J}cell{_J}(?:{_NE}{_J})?(?:carcinomas?|cancers?)",  # small cell carcinoma, any site
    rf"small{_J}cell{_J}(?:[a-z]+{_J}){{1,2}}?(?:carcinomas?|cancers?)",  # small cell lung / prostate / bladder ... cancer
    rf"small{_J}cell{_J}lung", r"\bSCLCs?\b", rf"carcinomas?\s*,\s*small{_J}cell",
    rf"small{_J}cell{_J}transformation",
    rf"transform\w*{_J}(?:in)?to{_J}(?:an?{_J})?(?:SCLCs?\b|small{_J}cell|{_NE})",
    rf"merkel{_J}cell{_J}(?:carcinom\w*|carcinogen\w*|cancers?)", rf"carcinomas?\s*,\s*merkel{_J}cell",
    rf"merkel(?s:.){{0,4000}}?(?-i:\bMCCs?\b)", rf"(?-i:\bMCCs?\b)(?s:.){{0,4000}}?merkel",  # MCC next to Merkel
    rf"high{_J}grade{_J}{_NE}",
    rf"poorly{_J}differentiated{_J}(?:{_NE}|NE[CN]s?\b)",
    rf"{_NE}{_J}trans{_J}?differentiat\w*", rf"{_NE}{_J}transformation",
    r"\bMANECs?\b", _CS(r"\bMiNENs?\b"), rf"mixed{_J}adeno{_J}{_NE}",
    rf"mixed{_J}{_NE}{_J}(?:and{_J})?non{_J}{_NE}", r"amphicrine",
    # generic NE word + lung / bladder carcinoma context in the same sentence
    rf"{_NE_GENERIC}[^.;]{{0,150}}?{_NEC_CARC_CTX}",
    rf"{_NEC_CARC_CTX}[^.;]{{0,150}}?{_NE_GENERIC}",
    # any NE word (incl. "NE tumour") anywhere in a passage that names prostate cancer
    rf"(?:{_NE_GENERIC}|{_NE}{_J}(?:tumou?rs?|neoplasms?))(?s:.){{0,4000}}?{_NEC_PROSTATE_CTX}",
    rf"{_NEC_PROSTATE_CTX}(?s:.){{0,4000}}?(?:{_NE_GENERIC}|{_NE}{_J}(?:tumou?rs?|neoplasms?))",
)
# Blanked before the nec test, next to the 8 older DOMAIN_EXCLUDE['nec'] rules.
_NEC_EXCLUDE_NEW: tuple = (
    rf"\bnon{_J}SCLCs?\b",
    # old WHO wording for carcinoid / atypical carcinoid (= NET)
    rf"(?:well|moderately|highly){_J}differentiated{_J}{_NE}{_J}carcinomas?",
    rf"(?:low|intermediate){_J}grade{_J}{_NE}{_J}carcinomas?",
    rf"well{_J}differentiated{_J},?{_J}high{_J}grade{_J}{_NE}{_J}(?:tumou?rs?|neoplasms?)",
    rf"well{_J}differentiated{_J}{_NE}{_J}(?:tumou?rs?|neoplasms?)",   # NET by definition
    # "NEC" = necrotizing enterocolitis: a NEC token before an enterocolitis
    # mention, and the stretch after one up to the next NE-carcinoma word
    rf"(?-i:\bNECs?\b)(?=(?s:.){{0,4000}}?enterocolit)",
    rf"enterocolit\w*(?:(?!{_NE}|small{_J}cell|merkel|carcinom|prostat)(?s:.)){{0,4000}}",
)
# The pre-2026-09-26 rule ("any neuroendocrine word"), kept for
# MRAG_NEC_RULE=legacy (domain_specs) and for audits.
NEC_TERMS_LEGACY: tuple = (
    rf"neuro{_J}endocrine", _CS(r"\bNECs?\b"), r"\bLCNECs?\b", r"\bNEPC\b", r"merkel",
    rf"small{_J}cell{_J}(?:neuro{_J}endocrine{_J})?carcinomas?(?!{_J}of{_J}the{_J}lung)",
    r"\bt-?SCNCs?\b", r"\bSCNPC\b")


def nec_rule_legacy() -> bool:
    """MRAG_NEC_RULE=legacy restores the pre-2026-09-26 nec text rule (any
    neuroendocrine word, no corpus veto) in domain_specs()."""
    return str(os.environ.get("MRAG_NEC_RULE") or "").strip().lower() in ("legacy", "old", "any-ne")

DOMAIN_TERMS: dict[str, tuple[str, ...]] = {
    "prostate": (r"prostat", r"\bm?CRPCs?\b", r"\bPRAD\b", r"\bNEPC\b", rf"castrat\w*{_J}resistan",
                 _CS(r"\bPCa\b"), r"\bLNCaP", r"\bVCaP\b", r"\b22Rv1\b", r"\bC4-2B?\b", _CS(r"\bPC-?3\b"),
                 r"\bDU-?145\b", r"enzalutamide|abiraterone|apalutamide|darolutamide",
                 rf"androgen{_J}deprivation"),
    "nec": NEC_TERMS,          # carcinoma only (necca 2026-09-26), see above
    "sclc": (rf"small{_J}cell{_J}lung", rf"small{_J}cell{_J}(?:carcinoma|cancer)s?{_J}of{_J}the{_J}lung",
             r"\bSCLCs?\b"),
    "nsclc": (rf"\bnon{_J}small{_J}cell", r"\bNSCLCs?\b", r"\bLUAD\b", r"\bLUSC\b",
              rf"\blung{_J}(?:adeno\w*|squamous)", rf"adenocarcinomas?{_J}of{_J}the{_J}lung",
              rf"squamous{_J}cell{_J}(?:carcinoma|cancer)s?{_J}of{_J}the{_J}lung",
              r"\bA549\b", r"\bH1299\b", r"\bH1975\b"),
    "breast": (r"breast", r"mammary", r"\bTNBCs?\b", r"\bBRCA\b(?![\s\-]?[12])", r"\bMCF-?7\b",
               r"\bMCF-?10A\b", r"\bMDA-?MB-?\d+", r"\bT-?47D\b", r"\bSK-?BR-?3\b", r"\bBT-?474\b"),
    "colorectal": (r"colorectal", r"\bcolon\b", r"\bcolonic\b", r"\brect(?:al|um)\b", _CS(r"\bCRCs?\b"),
                   _CS(r"\bCOAD\b"), _CS(r"\bREAD\b"), r"\bHCT-?116\b", r"\bSW-?(?:480|620)\b",
                   r"\bHT-?29\b", r"\bCaco-?2\b", r"\bDLD-?1\b", r"\bLoVo\b"),
    "pancreatic": (rf"pancrea(?:s|tic){_J}(?:ductal|cancers?|adeno\w*|carcinomas?|tumou?rs?|neoplasms?|malignan\w*)",
                   rf"(?:cancer|carcinoma|adenocarcinoma|tumou?r)s?{_J}of{_J}the{_J}pancreas",
                   r"\bPDACs?\b", r"\bPAAD\b", r"\bPanINs?\b", r"\bPANC-?1\b", rf"\bMIA{_J}PaCa",
                   r"\bCapan-?[12]\b", r"\bBxPC-?3\b", r"\bAsPC-?1\b"),
    "melanoma": (r"melanoma", r"\bSKCM\b", r"\bA-?375\b", r"\bSK-?MEL-?\d+", r"\bB16(?:-?F\d+)?\b"),
    "glioblastoma": (r"glioblastoma", r"glioma", r"\bGBMs?\b", r"\bLGGs?\b", r"astrocytoma",
                     r"\bU-?87(?:\s?MG)?\b", r"\bU-?251(?:\s?MG)?\b", r"\bDIPG\b"),
    "aml": (rf"acute{_J}myelo(?:id|genous)", rf"myelo(?:id|genous){_J}leuk", _CS(r"\bAML\b"), r"\bLAML\b",
            r"\bMOLM-?1[34]\b", r"\bMV-?4-?11\b", r"\bKasumi-?1\b", r"\bHL-?60\b", r"\bOCI-?AML"),
    "ovarian": (r"ovar(?:y|ies|ian)", r"\bOVCAR", r"\bHGS?OCs?\b", r"\bHGSCs?\b", r"\bSKOV-?3\b",
                r"\bA2780\b", rf"fallopian{_J}tube"),
    "hcc": (r"hepatocellular", rf"(?:liver|hepatic){_J}(?:cancers?|carcinomas?|tumou?rs?)", _CS(r"\bHCCs?\b"),
            r"\bLIHC\b", r"hepatoma", r"\bHep-?G2\b", r"\bHuh-?7", r"\bHep-?3B\b"),
    "gastric": (r"gastric", r"stomach", r"\bSTAD\b", r"\bMKN-?\d+", r"\bSGC-?7901\b", r"\bHGC-?27\b",
                r"\bBGC-?823\b"),
    "bladder": (r"(?<!gall )(?<!gall-)\bbladder", r"urothelial", r"\bBLCA\b", r"\bN?MIBCs?\b",
                r"\bUM-?UC-?\d+", r"\bT24\b"),
    "rcc": (rf"renal{_J}(?:cell{_J})?(?:carcinomas?|cancers?|tumou?rs?)",
            rf"kidney{_J}(?:cancers?|carcinomas?|tumou?rs?)", rf"clear{_J}cell{_J}renal",
            r"\b(?:cc|c|ch|p)?RCCs?\b", r"\bKIR[CP]\b", r"\bKICH\b", r"\b786-?O\b", r"\bCaki-?[12]\b",
            r"\bACHN\b"),
    "headneck": (rf"\bhead{_J}(?:and|&){_J}neck", r"\bHNSCCs?\b", r"\bHNSC\b",
                 rf"\boral{_J}(?:cavity|cancers?|squamous|carcinomas?|tumou?rs?)", r"\bOSCCs?\b",
                 r"laryn(?:x|geal)", r"pharyn(?:x|geal)", r"\btongue\b", r"\bFaDu\b", r"\bCal-?27\b"),
    "esophageal": (r"o?esophag", r"\bESCA\b", r"\bESCCs?\b", r"\bbarrett", r"\bKYSE-?\d+",
                   r"\bOE-?(?:19|33)\b"),
    "lymphoma": (r"lymphoma", r"\bDLBCLs?\b", r"hodgkin", r"burkitt", r"\bPTCLs?\b", r"\bCTCLs?\b",
                 rf"mycosis{_J}fungoides"),
    "pancancer": (rf"pan{_J}cancer", r"cancer", r"carcinoma", r"tumou?r", r"neoplas", r"malignan",
                  r"oncogen", r"metasta", r"leuka?emia", r"lymphoma", r"sarcoma", r"melanoma", r"glioma",
                  r"blastoma", r"\bTCGA\b"),
}
# Blanked before the DOMAIN_TERMS test (so "non-small cell lung" cannot count as SCLC).
# FIX 2026-09-26 (evfix round 3): "neuro-endocrine" in its PHYSIOLOGY sense
# (stress axis, psycho-neuro-endocrine-immune) is not the NEC domain.  Real
# case: in the 09-25 NEC run one gene's GeneRIF lines tied to NEC through
# "circadian-neuroendocrine-immune interactions", "the highly variable
# neuroendocrine and pathological sequelae of childhood trauma" and "parental
# PTSD and neuroendocrine outcomes".  Only phrases that were physiology in
# EVERY use found are blanked -- all 19 parsed corpora (4,018 documents
# containing "ndocrin") + the GeneRIF lines of the 8 audited runs:
# "neuroendocrine-immune" (5 corpus + 1 GeneRIF), "neuroendocrine
# response(s)" (5) / "stress" (1), "neuroendocrine systems" (4, plural),
# "neuroendocrine dysregulation" (2), "psycho-neuroendocrine" (1),
# "neuroendocrine control of" (1), "neuroendocrine (and pathological)
# sequelae" and "neuroendocrine outcomes" (GeneRIF).  Deliberately NOT
# blanked, because the corpus also uses them for tumours: "neuroendocrine
# system" (singular: "SCLC is a cancer of the neuroendocrine system", "NEN
# derive from cells of the diffuse neuroendocrine system"), "... regulation",
# "... regulators" (ASCL1, INSM1), "... pathway(s)", "... signaling",
# "... alterations", "... hormones/peptides/factors", "... islet".  Pituitary
# neuroendocrine tumours (PitNET) are NOT excluded: whether they belong to the
# nec domain is an owner decision (see the evfix round-3 report).
# VERIFICATION FIX V3 (2026-09-26, independent round-3 check on the current,
# post-backfill corpora: 77,397 unique documents, 4,361 with "ndocrin"): the
# "neuroendocrine(-and)-immun\w*" pattern also blanked the PATHOLOGY phrases
# "neuroendocrine immunomarker(s)" (8), "neuroendocrine immunohistochemistry /
# -chemical" (4), "... immunostain" (1), "... immunophenotype" (1) and the
# tumour-subtype uses "Neuroendocrine and Immune Gene Expression in Small Cell
# Lung Cancer" and "male breast cancer with distinct neuroendocrine and immune
# characteristics" -- 17 of its 26 matches.  Two chunks lost their only NEC
# word that way (nsclc 42724635 "neuroendocrine immunohistochemistry" in a
# transformed-SCLC review; glioblastoma 15967023 "neuroendocrine
# immunomarkers, chromogranin and synaptophysin").  Now blanked: the compound
# "neuroendocrine(-)immune / immunity / immunology" (5 of 5 physiology) and
# "neuroendocrine and immune" only before a physiology noun (regulation,
# dysregulation, responses, systems, signalling, pathways, interactions,
# crosstalk, axis, homeostasis, function, network: 4 of 4 physiology); the
# other four round-3 patterns matched physiology in all 17 of their uses and
# are unchanged.
DOMAIN_EXCLUDE: dict[str, tuple[str, ...]] = {
    "nec": (rf"\bhead{_J}(?:and|&){_J}neck\b", r"\bneck\b", rf"\bnon{_J}small{_J}cell",
            rf"psycho{_J}neuro{_J}endocrin\w*",
            rf"neuro{_J}endocrine{_J}(?:immun(?:e|ity|olog\w*)\b|and{_J}immune{_J}(?:dys)?"
            rf"(?:regulat\w*|respons\w*|systems?|signal\w*|pathways?|interactions?|crosstalk|ax[ie]s"
            rf"|homeostasis|function\w*|networks?)\b)",
            rf"neuro{_J}endocrine{_J}(?:stress|responses?|systems|dysregulation|outcomes?)\b",
            rf"neuro{_J}endocrine{_J}control{_J}of\b",
            rf"neuro{_J}endocrine{_J}(?:and{_J}(?:pathological{_J})?)?sequelae\b",
            *_NEC_EXCLUDE_NEW),       # necca 2026-09-26, see NEC_TERMS
    "sclc": (rf"\bnon{_J}small{_J}cell", rf"\bnon{_J}SCLCs?\b", r"\bNSCLCs?\b", r"\bLUAD\b", r"\bLUSC\b",
             rf"\blung{_J}(?:adeno\w*|squamous)"),
    "pancreatic": (rf"pancrea(?:s|tic){_J}neuro{_J}endocrine", r"\bp(?:an)?NETs?\b"),
    "bladder": (rf"\bgall{_J}bladder",),
}

# CORPUS-LABEL VETO (necca 2026-09-26).  A passage whose chunk carries the
# domain's corpus label normally ties without a text test (evidence_gate
# corpus_needs_text: only sclc needs one).  For a domain listed here the label
# is refused when the passage (title + chunk) is NET-type ONLY: a pattern below
# matches it and, after the DOMAIN_EXCLUDE phrases are blanked, no DOMAIN_TERMS
# pattern does (evidence_gate.corpus_vetoed).  Chosen on data: 8 of the
# 1,104 nec-labelled documents are NET-only
# by title + abstract, so a full text test on every nec corpus tie is not
# needed -- it would also drop genuine chunks that simply carry no NE word
# (a 3'UTR study, a target studied in the Merkel cell line MCC14.2).  But
# NET-only chunks inside mixed NEN reviews did ground scores: one gene
# passed on three PanNET chunks of a multi-omic review of pancreatic
# neuroendocrine neoplasms.  A passage that also names a
# carcinoma type is kept.  MRAG_CORPUS_TIE_VETO=off switches it off.
DOMAIN_CORPUS_VETO: dict[str, tuple[str, ...]] = {
    "nec": (rf"neuro{_J}endocrine{_J}(?:tumou?rs?|neoplasms?|neoplasia)",
            # bare "NET(s)" only with a grade: it is also "neutrophil extracellular trap"
            _CS(r"\b(?:Pan|p|GEP|GI|SI|Si|Pit|Lu|Th)-?NETs?\b"), rf"(?-i:\bNETs?){_J}(?:G[1-3]\b|grade)",
            _CS(r"\b(?:Pan|p|GEP|GI|SI|Pit)?-?NENs?\b"),
            r"carcinoids?", r"insulinomas?", r"gastrinomas?", r"glucagonomas?", r"VIPomas?",
            r"somatostatinomas?", r"ph(?:a)?eochromocytomas?", r"paragangliomas?", r"\bPPGLs?\b",
            rf"islet{_J}cell{_J}tumou?rs?",
            rf"well{_J}differentiated{_J}(?:neuro{_J}endocrine|NETs?\b|NENs?\b|endocrine)",
            rf"pituitary{_J}(?:adenomas?|neuro{_J}endocrine{_J}tumou?rs?)"),
}


# ---------------------------------------------------------------------------
# EVIDENCE FILTERS (evaudit 2026-09-27).  PI 2026-09-27: "make sure this
# never happens again" -- two FINAL targets were emitted on invalid
# evidence and found only by a manual review afterwards:
#   * a NEC target: its only nec passage profiles "metastasizing
#     serotonin-producing neuroendocrine carcinomas, which arise from
#     enterochromaffin cells in the jejunum and ileum" -- small-intestinal NETs
#     in the pre-2010 "carcinoma" wording.  "neuroendocrine carcinomas" is a
#     carcinoma term and none of the 16 veto / exclude patterns matches, so the
#     carcinoma-only rule let it through;
#   * an SCLC target named only in a list of screened, unaltered genes
#     (evidence_gate.list_negative): see there.
# 1a PAPER-LEVEL NET TEST (nec).  A nec tie -- a corpus passage, a GeneRIF line
# or a PubMed-rescue record -- does not count when the PAPER it comes from is a
# well-differentiated-tumour paper: its title / abstract / keywords / MeSH carry
# one of NEC_PAPER_NET_TEXT / NEC_PAPER_NET_MESH and none of the high-grade /
# carcinoma-type markers NEC_PAPER_HG_TEXT / NEC_PAPER_HG_MESH (checked after
# NEC_PAPER_HG_EXCLUDE is blanked: "non-small cell" is not "small cell",
# "well-differentiated high-grade NET" is a NET G3).  Judged on the PubMed
# record (evidence_gate.paper_net_verdict), fetched once per PMID.  Measured
# and hand-checked on every nec-tied passage / GeneRIF / rescue record of the
# NEC r3 FINAL run.  MRAG_NEC_NET_PAPER=off
# switches it off.
# 1b SENTENCE-LEVEL VETO (nec).  DOMAIN_CORPUS_VETO above refuses a passage
# only when the WHOLE passage (title + 512-word chunk) is NET-only, so a PanNET
# chunk of a PanNEN review that names a carcinoma type anywhere else still tied
# a gene ("transcriptomic profiles differ substantially between PanNETs with
# and without MEN1-ATRX-DAXX alterations. Tumors harboring these mutations
# exhibited increased expression of liver-associated genes ...").  The veto is now also applied to the sentences
# that name the gene (+- 1 sentence, evidence_gate.nec_sentence_veto).  NEPC
# papers use "neuroendocrine tumour" loosely, so NEC_SENTENCE_LOOSE_NET does not
# count as a NET word in a passage with prostate-cancer context
# (NEC_SENTENCE_LOOSE_OK), and the umbrella terms "neuroendocrine neoplasm" /
# "NEN" (NET + NEC) never count as NET words here (NEC_SENTENCE_UMBRELLA).
# MRAG_NEC_VETO_SENTENCE=off switches it off.
_NEW = rf"neuro{_J}endocrine"
_NETAB = r"(?-i:\b(?:Pan|p|GEP|GI|SI|Pit|Lu)?-?NE[TN]s?\b)"
NEC_PAPER_NET_TEXT: tuple = (
    r"enterochromaffin",
    rf"serotonin{_J}(?:producing|secreting|secretory|positive|expressing)",
    r"\bmid{_J}?gut\b".replace("{_J}", _J),
    rf"(?:\bjejun\w*|\bile(?:al|um)\b)(?s:.){{0,100}}?(?:{_NEW}|carcinoids?|{_NETAB})",
    rf"(?:{_NEW}|carcinoids?|{_NETAB})(?s:.){{0,100}}?(?:\bjejun\w*|\bile(?:al|um)\b)",
    rf"small{_J}(?:intestin\w*|bowel){_J}(?:{_NEW}|carcinoids?|{_NETAB})",
    _CS(r"\bS[Ii]-?NETs?\b"), _CS(r"\bSINETs?\b"),
    rf"well{_J}differentiated(?s:.){{0,80}}?(?:{_NEW}|endocrine|carcinoids?|{_NETAB})",
    rf"(?:{_NEW}|{_NETAB})(?s:.){{0,60}}?well{_J}differentiated",
    rf"low{_J}grade(?s:.){{0,60}}?(?:{_NEW}|carcinoids?|{_NETAB})",
    rf"(?:{_NEW}|carcinoids?|{_NETAB})(?s:.){{0,60}}?low{_J}grade",
    # WHO grade 1 / 2 of a NET -- never the G1 phase of the cell cycle
    rf"(?-i:\bG[12]\b)(?:{_J}(?:/|and|or|,){_J}(?-i:G[12]\b))?{_J}(?:{_NETAB}|{_NEW}|carcinoids?)",
    rf"(?:{_NETAB}|{_NEW}{_J}(?:tumou?rs?|neoplasms?))[\s,(:\-]{{0,4}}(?:WHO{_J})?(?:grade{_J}[12]\b|(?-i:G[12]\b))",
    rf"grade{_J}[12](?:{_J}(?:/|and|or){_J}[12])?{_J}(?:{_NEW}|{_NETAB})",
    r"ph(?:a)?eochromocytomas?", r"paragangliomas?", _CS(r"\bPPGLs?\b"), r"carcinoids?",
    r"insulinomas?", r"gastrinomas?", r"glucagonomas?", r"\bVIPomas?", r"somatostatinomas?",
    rf"islet{_J}cell{_J}(?:tumou?rs?|neoplasms?|carcinomas?)",
    _CS(r"\b(?:Pan|p|GEP|GI|SI|Pit|Lu)-?NETs?\b"),
    rf"pancrea(?:s|tic){_J}{_NEW}{_J}(?:tumou?rs?|neoplasms?)",
    rf"pituitary{_J}(?:adenomas?|{_NEW}{_J}tumou?rs?)",
)
# MeSH descriptors (whole descriptor names, case-insensitive)
NEC_PAPER_NET_MESH: tuple = (
    "Carcinoid Tumor", "Neuroendocrine Tumors", "Malignant Carcinoid Syndrome", "Carcinoid Heart Disease",
    "Pheochromocytoma", "Paraganglioma", "Paraganglioma, Extra-Adrenal", "Adenoma, Islet Cell",
    "Insulinoma", "Gastrinoma", "Glucagonoma", "Somatostatinoma", "Vipoma", "Enterochromaffin Cells",
    "Enterochromaffin-like Cells", "Multiple Endocrine Neoplasia Type 1", "Pituitary Neoplasms",
)
NEC_PAPER_HG_TEXT: tuple = (
    rf"poorly{_J}differentiated", rf"high{_J}grade", rf"small{_J}cell",
    _CS(r"\bSCLCs?\b"), r"\bt?-?SCNCs?\b", r"\bSCNECs?\b", r"\bSCNPCs?\b", r"\bEPSCCs?\b",
    rf"large{_J}cell{_J}(?:{_NEW}|carcinomas?)", r"\bLCNECs?\b",
    r"\bt?-?NEPCs?\b", rf"\bCRPC{_J}NE\b", rf"\bNE{_J}CRPC\b", rf"{_NEW}{_J}(?:like{_J})?(?:prostat|castrat|m?CRPC)",
    rf"treatment{_J}(?:emergent|induced|related){_J}(?:small{_J}cell|{_NEW})",
    rf"{_NEW}{_J}trans{_J}?differentiat\w*", rf"{_NEW}{_J}transformation",
    rf"{_NEW}{_J}differentiation(?s:.){{0,80}}?(?:prostat|castrat|m?CRPC|adenocarcinoma|lung{_J}cancer)",
    rf"(?:prostat\w*|castrat\w*|m?CRPC|adenocarcinoma)(?s:.){{0,80}}?{_NEW}{_J}differentiation",
    r"merkel",
    rf"(?-i:\bG3\b){_J}(?:(?-i:NECs?\b)|{_NEW}{_J}carcinomas?)",
    rf"(?:(?-i:\bNECs?)|{_NEW}{_J}carcinomas?)[\s,(:\-]{{0,4}}(?:WHO{_J})?(?:grade{_J}3\b|(?-i:G3\b))",
    rf"ki{_J}67[^.;%]{{0,60}}?(?:>|\u2265|>=|greater{_J}than|more{_J}than|above|exceed\w*|over|higher{_J}than)\s*(?:20|2\d|[3-9]\d|100)(?:\.\d+)?\s*%",
    rf"ki{_J}67[^.;%<\u2264]{{0,40}}?(?<![\d.])(?:2[1-9]|[3-9]\d|100)(?:\.\d+)?\s*%",
    r"\bMANECs?\b", _CS(r"\bMiNENs?\b"), rf"mixed{_J}adeno{_J}{_NEW}", r"amphicrine",
    # the post-2010 abbreviation: a paper that writes "NEC" separates carcinomas
    # from NETs (17102592 GI NECs "extremely aggressive"; 42522028 lung NETs and
    # NECs) -- the old wording spells "neuroendocrine carcinoma" out
    _CS(r"\b(?:Pan|p|GEP|GI|SI|Lu|Th)?-?NECs?\b"),
    # fix2 2026-09-28 (verification finding 4): undifferentiated NEC (23225557),
    # combined / mixed NET + NEC (39678483 "combined pancreatic neuroendocrine
    # tumour and neuroendocrine carcinoma"), an NEC component
    r"\bundifferentiated\b",
    rf"(?:combined|mixed|collision|composite)(?s:.){{0,80}}?{_NEW}{_J}carcinomas?",
    rf"{_NEW}{_J}carcinoma{_J}components?",
)
NEC_PAPER_HG_MESH: tuple = (
    "Carcinoma, Small Cell", "Small Cell Lung Carcinoma", "Carcinoma, Large Cell", "Carcinoma, Merkel Cell",
)
NEC_PAPER_HG_EXCLUDE: tuple = (
    rf"\bnon{_J}small{_J}cell", rf"\bnon{_J}SCLCs?\b", r"\bNSCLCs?\b",
    rf"well{_J}differentiated{_J},?{_J}(?:and{_J})?high{_J}grade",
    rf"high{_J}grade{_J}(?:well{_J}differentiated{_J})?(?:{_NEW}{_J})?(?-i:NETs?\b)",
    rf"high{_J}grade{_J}(?:dysplasia|serous|glioma|lymphoma|squamous|intraepithelial|PIN\b)",
    rf"(?-i:\bNECs?\b)(?=(?s:.){{0,4000}}?enterocolit)",          # necrotizing enterocolitis
)
# fix2 2026-09-28 (verification finding 4).  Blanked before the NET-marker
# test: "non-islet cell tumour hypoglycaemia" (NICTH) is not an islet cell
# tumour (23545670, a gastric NEC).
NEC_PAPER_NET_EXCLUDE: tuple = (
    rf"non{_J}islet{_J}cell{_J}tumou?rs?(?:{_J}(?:induced{_J})?hypoglyc\w*)?", r"\bNICTH\b",
)
# The STRONG well-differentiated-tumour markers.  A paper whose TITLE names a
# neuroendocrine carcinoma is NET-type only on one of these -- not on the
# generic MeSH parent, a paraganglioma / pituitary mention in a differential
# diagnosis (24599901 C-cell NEC of the thyroid) or a background
# "pancreatic neuroendocrine tumours" (22977618, 35280094, 22889781).
# The NET paper of the NEC case (EVIDENCE FILTERS above) keeps its verdict:
# "enterochromaffin", "serotonin-producing".
NEC_PAPER_NET_STRONG: tuple = (
    r"enterochromaffin",
    rf"serotonin{_J}(?:producing|secreting|secretory|positive|expressing)",
    r"\bmid{_J}?gut\b".replace("{_J}", _J),
    rf"(?:\bjejun\w*|\bile(?:al|um)\b)(?s:.){{0,100}}?(?:{_NEW}|carcinoids?|{_NETAB})",
    rf"(?:{_NEW}|carcinoids?|{_NETAB})(?s:.){{0,100}}?(?:\bjejun\w*|\bile(?:al|um)\b)",
    rf"small{_J}(?:intestin\w*|bowel){_J}(?:{_NEW}|carcinoids?|{_NETAB})",
    _CS(r"\bS[Ii]-?NETs?\b"), _CS(r"\bSINETs?\b"),
    rf"well{_J}differentiated(?s:.){{0,80}}?(?:{_NEW}|endocrine|carcinoids?|{_NETAB})",
    rf"(?:{_NEW}|{_NETAB})(?s:.){{0,60}}?well{_J}differentiated",
    rf"low{_J}grade(?s:.){{0,60}}?(?:{_NEW}|carcinoids?|{_NETAB})",
    rf"(?:{_NEW}|carcinoids?|{_NETAB})(?s:.){{0,60}}?low{_J}grade",
    rf"(?-i:\bG[12]\b)(?:{_J}(?:/|and|or|,){_J}(?-i:G[12]\b))?{_J}(?:{_NETAB}|{_NEW}|carcinoids?)",
    rf"(?:{_NETAB}|{_NEW}{_J}(?:tumou?rs?|neoplasms?))[\s,(:\-]{{0,4}}(?:WHO{_J})?(?:grade{_J}[12]\b|(?-i:G[12]\b))",
    rf"grade{_J}[12](?:{_J}(?:/|and|or){_J}[12])?{_J}(?:{_NEW}|{_NETAB})",
    r"carcinoids?", r"insulinomas?", r"gastrinomas?", r"glucagonomas?", r"\bVIPomas?", r"somatostatinomas?",
    rf"islet{_J}cell{_J}(?:tumou?rs?|neoplasms?|carcinomas?)",
)
# the generic MeSH parent (it also indexes NEC papers): not a NET marker on its own
NEC_PAPER_NET_MESH_GENERIC: tuple = ("Neuroendocrine Tumors",)
# a TITLE naming a neuroendocrine carcinoma (the old wording for a carcinoid --
# well / moderately differentiated, low / intermediate grade NE carcinoma --
# blanked first)
NEC_PAPER_NEC_TITLE: tuple = (rf"{_NEW}{_J}carcinomas?", rf"carcinomas?\s*,\s*{_NEW}",
                              _CS(r"\b(?:Pan|p|GEP|GI|SI|Lu|Th)?-?NECs?\b"))
NEC_PAPER_NEC_TITLE_EXCLUDE: tuple = (
    rf"(?:well|moderately|highly){_J}differentiated{_J}{_NEW}{_J}carcinomas?",
    rf"(?:low|intermediate){_J}grade{_J}{_NEW}{_J}carcinomas?",
)
NEC_SENTENCE_LOOSE_NET: tuple = (rf"{_NEW}{_J}(?:tumou?rs?|neoplasms?|neoplasia)",)
# the UMBRELLA terms name NETs and NECs together (WHO 2019: NEN = NET + NEC), so
# in the gene's own sentences / the title they are not NET words: a gene
# "aberrantly expressed in tumors, such as neuroendocrine neoplasms (NENs)"
# is the subject of an SCLC / LCNEC marker paper
NEC_SENTENCE_UMBRELLA: tuple = (rf"{_NEW}{_J}(?:neoplasms?|neoplasia)",
                                _CS(r"\b(?:Pan|p|GEP|GI|SI|Pit|Lu)?-?NENs?\b"))
NEC_SENTENCE_LOOSE_OK: tuple = (rf"\bprostat(?!e{_J}specific)\w*", r"\bm?CRPCs?\b", _CS(r"\bPCa\b"),
                                rf"\bcastrat\w*{_J}resistan\w*", r"\bNEPCs?\b")
# fix2 2026-09-28 (verification finding 2): high-grade / NE lung cancer context
# also makes the loose "neuroendocrine tumour" wording no NET word (31324758
# "high-grade NE-lung cancers" under the title "pulmonary neuroendocrine tumor
# cells") -- but NOT in a passage that names well-differentiated tumours: a
# "well-differentiated high-grade NET" (NET G3, 35609558) is a NET
NEC_SENTENCE_LOOSE_OK_HG: tuple = (
    rf"high{_J}grade(?!{_J}(?:well|(?:{_NEW}{_J})?(?-i:NETs?\b)|{_NEW}{_J}tumou?r))",
    rf"poorly{_J}differentiated",
    rf"(?:\bNE|{_NEW}){_J}lung{_J}(?:cancers?|carcinomas?)")
# fix3 2026-09-28: the block needs "well-differentiated" TIED to the high grade
# (a WD high-grade NET = NET G3, 35609558 "well-differentiated (WD) high-grade
# (HG) NETs"); a passage that merely contrasts poorly with well differentiated
# tumours (a gene "overexpressed in high-grade tumors") is not blocked
NEC_SENTENCE_LOOSE_OK_HG_BLOCK: tuple = (rf"well{_J}differentiated(?:{_J}\(WD\))?(?s:.){{0,40}}?high{_J}grade",
                                         rf"high{_J}grade(?s:.){{0,40}}?well{_J}differentiated",
                                         _CS(rf"\bWD{_J}(?:\(?HG\)?|high{_J}grade)"),
                                         # ... and "well-differentiated" qualifying a NE tumour: the
                                         # "neuroendocrine tumours" of such a passage are NETs
                                         # (24658583 "well-differentiated pancreatic neuroendocrine
                                         # tumors ..., high-grade pancreatic neuroendocrine carcinomas")
                                         rf"well{_J}differentiated(?:[\s\-\u2010-\u2015\u2212]{{1,3}}[a-z]+){{0,2}}?"
                                         rf"[\s\-\u2010-\u2015\u2212]{{1,3}}(?:{_NEW}{_J}(?:tumou?rs?|neoplasms?)|(?-i:NETs?\b))",
                                         _CS(r"\b(?:NETs?{_J}G3|G3{_J}NETs?)\b"),
                                         _CS(r"\bHG{_J}NETs?\b"))
# a carcinoid-vs-carcinoma contrast in the gene's sentences names the lung NE
# carcinomas (26008974 "lower expression in carcinoids compared to carcinomas",
# 27873319 "Carcinoids and carcinomas shared ...") -- fix2 2026-09-28
NEC_SENTENCE_CONTRAST: tuple = (r"carcinoids?\b[^.;]{0,120}?\bcarcinomas?\b",
                                r"\bcarcinomas?\b[^.;]{0,120}?\bcarcinoids?\b",
                                # fix3 2026-09-28 (second verification, finding 5): "DLL3 is
                                # expressed in 80% of pulmonary neuroendocrine tumours of high
                                # grade, but rarely in carcinoids"
                                rf"(?:high{_J}grade(?!{_J}(?:well|(?:{_NEW}{_J})?(?-i:NETs?\b)|{_NEW}{_J}tumou?r))"
                                rf"|poorly{_J}differentiated)[^.;]{{0,120}}?\bcarcinoids?\b",
                                rf"\bcarcinoids?\b[^.;]{{0,120}}?(?:high{_J}grade(?!{_J}(?:well|(?:{_NEW}{_J})?"
                                rf"(?-i:NETs?\b)|{_NEW}{_J}tumou?r))|poorly{_J}differentiated)")


# ---------------------------------------------------------------------------
# fix3 2026-09-28 (second verification of the evaudit build)
# ---------------------------------------------------------------------------
_S1 = "[\\s\\-‐-―−]{1,3}"      # at least one separator between two words
# 1b OLD-WORDING TEST (finding 3).  Before 2010 a small-intestinal / midgut NET
# was often called a "(serotonin-producing) neuroendocrine carcinoma"
# (the NEC case above).  1a catches such a paper by its PubMed record, but a
# single high-grade word anywhere in the abstract switched 1a off and 1b did
# not back it up, because the gene's sentence names "neuroendocrine
# carcinoma".  Now the sentence(s) naming the gene (+- 1) and the title are
# NET-type -- the tie is vetoed -- when they carry a NET-defined carcinoma
# phrase (NEC_OLDWORD_DEFINING), a small-intestinal site phrase
# (NEC_OLDWORD_SITE) or a strong NET marker (NEC_OLDWORD_STRONG), and NO
# high-grade marker (NEC_OLDWORD_HG, after NEC_PAPER_HG_EXCLUDE is blanked; a
# bare "NEC(s)" counts as high-grade unless the passage defines it as the
# abbreviation of a NET-defined phrase).
NEC_OLDWORD_DEFINING: tuple = (
    rf"serotonin{_J}(?:producing|secreting|secretory|positive|expressing|rich)(?:{_S1}[a-z\-]+){{0,2}}?"
    rf"{_S1}(?:{_NEW}{_J})?carcinomas?\b",
    rf"enterochromaffin(?:{_J}(?:like{_J})?cells?)?(?:{_J}derived)?{_S1}(?:{_NEW}{_J})?carcinomas?\b",
    rf"{_NEW}{_J}carcinomas?(?:\s*[(\[][^)\]]{{0,20}}[)\]])?\s*,?\s*(?:which|that)\s+(?:\w+\s+){{0,2}}?"
    rf"(?:arise|arises|arising|originat\w*|deriv\w*|develop\w*|stem\w*)\s+from\s+(?:\w+\s+){{0,3}}?"
    rf"(?:enterochromaffin|serotonin|(?-i:EC){_J}cells?)",
    rf"\bmid{_J}?gut(?:{_S1}(?:carcinoid|enterochromaffin)(?:{_J}cells?)?{_J}derived)?{_S1}(?:{_NEW}{_J})?carcinomas?\b",
    rf"carcinoid(?:{_J}tumou?rs?)?{_J}derived{_S1}(?:{_NEW}{_J})?carcinomas?\b",
    rf"{_NEW}{_J}carcinomas?\s*[(\[]\s*(?:(?:typical|atypical){_S1})?carcinoids?(?:{_S1}tumou?rs?)?\s*[)\]]",
    rf"carcinoids?(?:{_S1}tumou?rs?)?\s*[(\[]\s*(?:(?:well|moderately){_J}differentiated{_S1})?{_NEW}{_J}"
    rf"carcinomas?\s*[)\]]",
)
# a site phrase: counted only when neither the gene's sentences nor the title
# carry any high-grade marker (a small-intestinal NEC of the modern wording is
# poorly differentiated / small or large cell and says so)
NEC_OLDWORD_SITE: tuple = (
    rf"(?:\bjejun\w*|\bile(?:al|um)\b|\bjejuno{_J}ileal|\bsmall{_J}(?:intestin\w*|bowel)|\bmid{_J}?gut\b)"
    rf"(?:{_S1}[a-z]+){{0,2}}?{_S1}(?:{_NEW}{_J})?carcinomas?\b",
    rf"\bcarcinomas?(?:{_S1}[a-z]+){{0,3}}?{_S1}(?:of|from|in){_S1}(?:the{_S1})?(?:[a-z]+{_S1})?"
    rf"(?:jejun\w*|ile(?:al|um)\b|jejuno{_J}ileal|small{_J}(?:intestin\w*|bowel)|mid{_J}?gut\b)",
)
NEC_OLDWORD_STRONG: tuple = (
    r"enterochromaffin",
    rf"serotonin{_J}(?:producing|secreting|secretory|positive|expressing)",
    rf"\bmid{_J}?gut\b",
    _CS(r"\bS[Ii]-?NETs?\b"), _CS(r"\bSINETs?\b"),
    rf"(?:\bjejun\w*|\bile(?:al|um)\b)(?s:.){{0,60}}?(?:{_NEW}|carcinoids?|{_NETAB})",
    rf"(?:{_NEW}|carcinoids?|{_NETAB})(?s:.){{0,60}}?(?:\bjejun\w*|\bile(?:al|um)\b)",
    rf"small{_J}(?:intestin\w*|bowel){_J}(?:{_NEW}|carcinoids?|{_NETAB})",
    rf"carcinoid(?:{_J}tumou?rs?)?{_J}derived",
)
# SCLC cell lines name a carcinoma (a GeneRIF line: "H69 cells exhibited higher
# amounts of chromatin-bound ... compared to H82 cells")
NEC_SENTENCE_SCLC_LINES: tuple = (
    _CS(r"\b(?:NCI-?)?H(?:69|82|128|146|187|196|209|345|446|510|524|526|719|748|841|889|1048|1092|1105|1184"
        r"|1417|1436|1618|1688|1694|1836|1876|1930|1963|2029|2081|2107|2141|2171|2195|2227)(?:AR|V)?\b"),
    _CS(r"\bDMS[- ]?(?:53|79|114|153|187|273|454)\b"), _CS(r"\bSHP-?77\b"), _CS(r"\bCOR-?L\d{2,3}\b"),
    _CS(r"\bGLC-?\d{1,2}\b"),
)
NEC_OLDWORD_HG: tuple = (
    rf"poorly{_J}differentiated", r"\bundifferentiated\b", rf"high{_J}grade", rf"small{_J}cell",
    _CS(r"\bSCLCs?\b"), r"\bt?-?SCNCs?\b", r"\bSCNECs?\b", r"\bSCNPCs?\b", r"\bEPSCCs?\b",
    rf"large{_J}cell", r"\bLCNECs?\b", r"\bt?-?NEPCs?\b", rf"\bCRPC{_J}NE\b", rf"\bNE{_J}CRPC\b",
    rf"\bprostat(?!e{_J}specific)\w*", r"\bm?CRPCs?\b", _CS(r"\bPCa\b"), rf"\bcastrat\w*",
    rf"treatment{_J}(?:emergent|induced|related)", rf"{_NEW}{_J}trans{_J}?differentiat\w*",
    rf"{_NEW}{_J}transformation", r"merkel", _CS(r"\bG3\b"), rf"grade{_J}(?:3|III)\b",
    rf"ki{_J}67[^.;%]{{0,60}}?(?:>|≥|>=|greater{_J}than|more{_J}than|above|exceed\w*|over|higher{_J}than)"
    rf"\s*(?:20|2\d|[3-9]\d|100)(?:\.\d+)?\s*%",
    rf"ki{_J}67[^.;%<≤]{{0,40}}?(?<![\d.])(?:2[1-9]|[3-9]\d|100)(?:\.\d+)?\s*%",
    r"\bMANECs?\b", _CS(r"\bMiNENs?\b"), rf"mixed{_J}adeno", r"amphicrine",
    rf"(?:combined|mixed|collision|composite)(?s:.){{0,80}}?{_NEW}{_J}carcinomas?",
    _CS(r"\b(?:Pan|p|GEP|GI|Lu|Th)-?NECs?\b"), _CS(r"\bNECs?\b"),
    rf"(?:\blung|pulmonary|bronch\w*){_J}(?:{_NEW}{_J})?(?:carcinomas?|cancers?)",
    rf"(?:\bNE|{_NEW}){_J}lung{_J}(?:cancers?|carcinomas?)",
    rf"{_NEW}{_J}carcinomas?{_J}of{_J}(?:the{_J})?lung",
) + NEC_SENTENCE_SCLC_LINES
# the bare "NEC(s)" is the abbreviation of a NET-defined phrase in this
# passage ("serotonin-producing neuroendocrine carcinomas (NECs)"): it is then
# not a high-grade marker
NEC_OLDWORD_ABBREV = r"\s*[(\[]\s*(?-i:NECs?)\s*[)\]]"
# 1b CARCINOMA NAMES (finding 5).  In a lung-NE passage a bare "carcinoma(s)"
# names the lung NE carcinomas (26008974 under "... neuroendocrine tumors of
# the lung": "carcinomas showed elevated expression of ..."), and an SCLC cell
# line names SCLC.  Either counts as naming a carcinoma type.
NEC_SENTENCE_LUNG_NE: tuple = (
    rf"(?:\blung|pulmonary|bronch\w*|thoracic)(?:{_S1}[a-z]+){{0,2}}?{_S1}{_NEW}",
    rf"{_NEW}{_J}(?:tumou?rs?|neoplasms?|neoplasia|carcinomas?|cells?|lesions?)(?:{_S1}[a-z]+){{0,3}}?"
    rf"{_S1}of{_S1}(?:the{_S1})?(?:lungs?|bronch\w*)",
    _CS(r"\bLu-?NE[TNC]s?\b"), _CS(r"\bP-?NENs?\b"),
)
# a bare "carcinoma" preceded by one of these is another carcinoma type
NEC_SENTENCE_BARE_CARC_NOT = (
    r"(?:squamous|adeno|non{_J}small|NSCLC|sarcomatoid|mucoepidermoid|basaloid|adenoid|salivary|giant|spindle"
    r"|urothelial|hepatocellular|renal|thyroid|breast|colorectal|gastric|well{_J}differentiated|low{_J}grade"
    r"|intermediate{_J}grade|typical|atypical)(?:{_J}cell)?{_J}$").replace("{_J}", _J)
# 1a OWN SPECIMENS (finding 3).  A marker in a contrast / comparison span
# ("in contrast to poorly differentiated neuroendocrine carcinomas, these
# tumours ...", "compared with well-differentiated NETs") describes the other
# group, not the paper's own tumours: the span is blanked before the NET and
# the high-grade tests.  A high-grade marker in a clause that names a control
# ("the small cell lung cancer line H69 as control") is blanked too.
NEC_PAPER_CONTRAST: tuple = (
    rf"\b(?:in{_J}contrast{_J}(?:to|with)|unlike|as{_J}opposed{_J}to|rather{_J}than|distinct{_J}from"
    rf"|differ\w*{_J}from|compared{_J}(?:to|with)|in{_J}comparison{_J}(?:to|with))\b[^.;,:]{{0,100}}",
)
NEC_PAPER_CONTROL: str = (rf"\b(?:as{_S1}(?:an?{_S1}|the{_S1})?(?:positive{_S1}|negative{_S1}|reference{_S1})?"
                          rf"controls?|control{_S1}(?:cell{_S1})?(?:lines?|cells?|samples?|tissues?))\b")
# 1a THE OLD WORDING.  A paper whose title or first two abstract sentences
# carry a NET-defined carcinoma phrase (NEC_OLDWORD_DEFINING) is about a
# NET-type tumour in the old "carcinoma" wording: only a STRONG high-grade
# marker (a histotype or grade of its own specimens) makes it an NEC paper;
# the weak ones -- a generic "high grade", a Ki-67 value, the abbreviation
# "NEC(s)" -- do not.
# a TITLE naming a small-intestinal / midgut neuroendocrine carcinoma
# (23184910 "... biomarker for small intestine neuroendocrine carcinomas",
# the sister paper of the NEC case's NET paper: VMAT1-positive tumours with
# mesentery metastases, abbreviated "SI-NEC") is the old wording of an SI-NET
# unless a high-grade marker other than the abbreviation is present (a Ki-67
# over 20 % counts here: 29450787 "small intestinal neuroendocrine carcinoma
# ... Ki-67 index was more than 90%" is a genuine NEC)
NEC_PAPER_SITE_TITLE: tuple = (
    rf"(?:\bjejun\w*|\bile(?:al|um)\b|\bjejuno{_J}ileal|\bsmall{_J}(?:intestin\w*|bowel)|\bmid{_J}?gut\b)"
    rf"(?:{_S1}[a-z]+)?{_S1}{_NEW}{_J}carcinomas?\b",
    rf"{_NEW}{_J}carcinomas?{_S1}(?:of|from|in){_S1}(?:the{_S1})?(?:jejun\w*|ile(?:al|um)\b|small{_J}(?:intestin\w*|bowel))",
    _CS(r"\bSI-?NECs?\b"),
)
NEC_PAPER_HG_WEAK: tuple = tuple(
    x for x in NEC_PAPER_HG_TEXT
    if x == rf"high{_J}grade" or x.startswith("ki") or x == _CS(r"\b(?:Pan|p|GEP|GI|SI|Lu|Th)?-?NECs?\b"))


def evidence_defence_on() -> bool:
    """fix2 2026-09-28: the MASTER switch of every 2026-09-27 evidence layer
    (1a-1c, GeneRIF alias, 600-char GeneRIF display, re-score after a filter,
    evidence audit).  MRAG_EVIDENCE_DEFENCE=0 -> all off, the pipeline is the
    pre-2026-09-27 one byte for byte.  Default ON."""
    # fix3 2026-09-28 (second verification, finding 6): MRAG_EVIDENCE_AUDIT=0
    # is the documented rollback switch and restores the pre-2026-09-27
    # pipeline byte for byte too (MRAG_EVIDENCE_AUDIT_MODE=off turns off only
    # the LLM audit and keeps the deterministic layers)
    for k in ("MRAG_EVIDENCE_DEFENCE", "MRAG_EVIDENCE_AUDIT"):
        v = str(os.environ.get(k) or "").strip().lower()
        if v in ("0", "off", "false", "no", "none"):
            return False
    return True


def evfilter_on(name: str) -> bool:
    """evaudit 2026-09-27 switches, all ON by default:
    MRAG_NEC_NET_PAPER (1a), MRAG_NEC_VETO_SENTENCE (1b), MRAG_LIST_NEGATIVE (1c);
    all off under MRAG_EVIDENCE_DEFENCE=0 (fix2 2026-09-28)."""
    if not evidence_defence_on():
        return False
    v = str(os.environ.get(name) or "").strip().lower()
    return v not in ("0", "off", "false", "no", "none")


def _evfilter_spec(key: str) -> dict:
    """The evidence-filter data a domain spec carries (evaudit 2026-09-27),
    only what is switched on; {} for a domain without such rules."""
    if key != "nec" or nec_rule_legacy():
        return {}
    out: dict = {}
    if evfilter_on("MRAG_NEC_NET_PAPER"):
        out["paper_net"] = {"net_text": list(NEC_PAPER_NET_TEXT), "net_mesh": list(NEC_PAPER_NET_MESH),
                            "hg_text": list(NEC_PAPER_HG_TEXT), "hg_mesh": list(NEC_PAPER_HG_MESH),
                            "hg_exclude": list(NEC_PAPER_HG_EXCLUDE),
                            # fix2 2026-09-28
                            "net_exclude": list(NEC_PAPER_NET_EXCLUDE),
                            "net_strong": list(NEC_PAPER_NET_STRONG),
                            "net_mesh_generic": list(NEC_PAPER_NET_MESH_GENERIC),
                            "nec_title": list(NEC_PAPER_NEC_TITLE),
                            "nec_title_exclude": list(NEC_PAPER_NEC_TITLE_EXCLUDE),
                            # fix3 2026-09-28
                            "contrast": list(NEC_PAPER_CONTRAST), "control": NEC_PAPER_CONTROL,
                            "defining": list(NEC_OLDWORD_DEFINING), "hg_weak": list(NEC_PAPER_HG_WEAK),
                            "site_title": list(NEC_PAPER_SITE_TITLE)}
    if evfilter_on("MRAG_NEC_VETO_SENTENCE"):
        out["sentence_veto"] = {"loose_net": list(NEC_SENTENCE_LOOSE_NET),
                                "loose_ok": list(NEC_SENTENCE_LOOSE_OK),
                                "loose_ok_hg": list(NEC_SENTENCE_LOOSE_OK_HG),
                                "loose_ok_hg_block": list(NEC_SENTENCE_LOOSE_OK_HG_BLOCK),
                                "contrast": list(NEC_SENTENCE_CONTRAST),
                                "umbrella": list(NEC_SENTENCE_UMBRELLA),
                                # fix3 2026-09-28
                                "oldword": {"defining": list(NEC_OLDWORD_DEFINING),
                                            "site": list(NEC_OLDWORD_SITE),
                                            "strong": list(NEC_OLDWORD_STRONG),
                                            "hg": list(NEC_OLDWORD_HG),
                                            "hg_exclude": list(NEC_PAPER_HG_EXCLUDE),
                                            "abbrev": NEC_OLDWORD_ABBREV},
                                "sclc_lines": list(NEC_SENTENCE_SCLC_LINES),
                                "lung_ne": list(NEC_SENTENCE_LUNG_NE),
                                "bare_carc_not": NEC_SENTENCE_BARE_CARC_NOT}
    return out


# EVIDENCE AUDIT (evaudit 2026-09-27): what each domain IS, in the words the
# audit LLM is given (nodes/evidence_audit.py).  A domain without an entry gets
# "<label> itself; not another cancer".
DOMAIN_AUDIT_DEFINITION: dict[str, str] = {
    "nec": ("neuroendocrine CARCINOMA only: neuroendocrine prostate cancer (NEPC, treatment-emergent "
            "small cell / NE prostate cancer, CRPC-NE), small cell carcinoma of any site (including small "
            "cell lung cancer), large cell neuroendocrine carcinoma, poorly differentiated / high-grade "
            "neuroendocrine carcinoma, Merkel cell carcinoma, neuroendocrine transdifferentiation of a "
            "carcinoma, mixed neuroendocrine carcinoma. NOT neuroendocrine carcinoma: well-differentiated "
            "neuroendocrine tumours (NET, carcinoid, PanNET, GEP-NET, small-intestinal / midgut / "
            "serotonin-producing enterochromaffin-cell tumours -- also when an older paper calls them "
            "'neuroendocrine carcinoma'), low / intermediate grade NE neoplasms, pheochromocytoma, "
            "paraganglioma, pituitary tumours"),
    "sclc": ("small cell lung cancer (SCLC) itself, including SCLC cell lines and SCLC transformation; NOT "
             "non-small cell lung cancer (NSCLC, lung adenocarcinoma, lung squamous cell carcinoma), not "
             "other lung diseases and not other cancers"),
    "prostate": ("prostate cancer (adenocarcinoma, castration-resistant or neuroendocrine prostate cancer, "
                 "prostate cancer cell lines or models); NOT other cancers and not benign prostate "
                 "tissue only"),
}


def audit_definition(key: str) -> str:
    """The audit LLM's definition of one domain (evaudit 2026-09-27)."""
    d = DOMAIN_AUDIT_DEFINITION.get(key)
    if d:
        return d
    return "%s itself; NOT another cancer" % (display_name(key) or key)



# ---------------------------------------------------------------------------
# ADD-1 (2026-09-18): live PubMed check for a "not in our corpus" novelty claim.
# The indexed corpora are snapshots of very different sizes (prostate ~3.8k,
# sclc ~3.0k, nec 787 parsed papers), so "absent from the indexed corpus" is not
# on its own evidence that nothing is published. One esearch per (gene, domain)
# that would otherwise be reported as unreported; any hit downgrades the claim.
#
# PRECISION FIRST. These fragments are deliberately narrower than DOMAIN_TERMS:
# a false POSITIVE only suppresses a novelty claim (safe), a false NEGATIVE
# would let a wrong novelty claim through (not safe) -- but an over-broad
# fragment makes every gene look covered and the check worthless. MeSH terms
# carry the recall, abbreviations catch the not-yet-indexed recent papers.
# NOTE on sclc: PubMed phrase search matches inside "non-small cell lung
# cancer", so the sclc fragment uses ONLY the MeSH heading and SCLC[tiab]
# (a distinct token from NSCLC) -- never the spelled-out phrase.
DOMAIN_PUBMED: dict[str, str] = {
    "prostate": ('"Prostatic Neoplasms"[mh] OR "prostate cancer"[tiab] OR "prostate carcinoma"[tiab] '
                 'OR "prostatic carcinoma"[tiab] OR "prostate adenocarcinoma"[tiab] OR CRPC[tiab] '
                 'OR mCRPC[tiab] OR NEPC[tiab] OR PRAD[tiab]'),
    "nec": ('"Carcinoma, Neuroendocrine"[mh] OR "Neuroendocrine Tumors"[mh] '
            'OR "neuroendocrine carcinoma"[tiab] OR "neuroendocrine tumor"[tiab] '
            'OR "neuroendocrine tumour"[tiab] OR "neuroendocrine neoplasm"[tiab] '
            'OR "neuroendocrine differentiation"[tiab] OR LCNEC[tiab] OR NEPC[tiab] '
            'OR "Merkel cell"[tiab]'),
    "sclc": '"Small Cell Lung Carcinoma"[mh] OR SCLC[tiab]',
    "nsclc": ('"Carcinoma, Non-Small-Cell Lung"[mh] OR NSCLC[tiab] OR "lung adenocarcinoma"[tiab] '
              'OR LUAD[tiab] OR LUSC[tiab]'),
    "breast": ('"Breast Neoplasms"[mh] OR "breast cancer"[tiab] OR "breast carcinoma"[tiab] '
               'OR "breast tumor"[tiab] OR "breast tumour"[tiab] OR TNBC[tiab]'),
    "colorectal": ('"Colorectal Neoplasms"[mh] OR "colorectal cancer"[tiab] OR "colorectal carcinoma"[tiab] '
                   'OR "colon cancer"[tiab] OR "rectal cancer"[tiab] OR CRC[tiab]'),
    "pancreatic": ('"Pancreatic Neoplasms"[mh] OR "pancreatic cancer"[tiab] '
                   'OR "pancreatic ductal adenocarcinoma"[tiab] OR "pancreatic carcinoma"[tiab] '
                   'OR PDAC[tiab]'),
    "melanoma": '"Melanoma"[mh] OR melanoma[tiab] OR SKCM[tiab]',
    "glioblastoma": ('"Glioblastoma"[mh] OR "Glioma"[mh] OR glioblastoma[tiab] OR glioma[tiab] '
                     'OR GBM[tiab]'),
    "aml": ('"Leukemia, Myeloid, Acute"[mh] OR "acute myeloid leukemia"[tiab] '
            'OR "acute myeloid leukaemia"[tiab] OR "acute myelogenous leukemia"[tiab] OR AML[tiab]'),
    "ovarian": ('"Ovarian Neoplasms"[mh] OR "ovarian cancer"[tiab] OR "ovarian carcinoma"[tiab] '
                'OR "high-grade serous"[tiab]'),
    "hcc": ('"Carcinoma, Hepatocellular"[mh] OR "hepatocellular carcinoma"[tiab] OR HCC[tiab] '
            'OR "liver cancer"[tiab]'),
    "gastric": ('"Stomach Neoplasms"[mh] OR "gastric cancer"[tiab] OR "gastric carcinoma"[tiab] '
                'OR "stomach cancer"[tiab]'),
    "bladder": ('"Urinary Bladder Neoplasms"[mh] OR "bladder cancer"[tiab] OR "bladder carcinoma"[tiab] '
                'OR "urothelial carcinoma"[tiab] OR BLCA[tiab]'),
    "rcc": ('"Carcinoma, Renal Cell"[mh] OR "renal cell carcinoma"[tiab] OR "kidney cancer"[tiab] '
            'OR "clear cell renal"[tiab] OR ccRCC[tiab]'),
    "headneck": ('"Head and Neck Neoplasms"[mh] OR "head and neck cancer"[tiab] '
                 'OR "head and neck squamous"[tiab] OR HNSCC[tiab] '
                 'OR "oral squamous cell carcinoma"[tiab]'),
    "esophageal": ('"Esophageal Neoplasms"[mh] OR "esophageal cancer"[tiab] OR "oesophageal cancer"[tiab] '
                   'OR "esophageal carcinoma"[tiab] OR "esophageal squamous cell carcinoma"[tiab] '
                   'OR ESCC[tiab]'),
    "lymphoma": '"Lymphoma"[mh] OR lymphoma[tiab] OR DLBCL[tiab]',
    "pancancer": ('"Neoplasms"[mh] OR cancer[tiab] OR carcinoma[tiab] OR tumor[tiab] OR tumour[tiab] '
                  'OR neoplasm[tiab]'),
}


def pubmed_query(key: str, gene: str = "", gene_name: str = "") -> str:
    """PubMed esearch term for one corpus key. With `gene`, the full conservative
    query: '<GENE>[tiab] AND (<domain fragment>)'. With `gene_name` (the official
    full name, FIX 2026-09-19 -- the symbol alone misses genes written out in
    words, e.g. a gene that the domain's papers name only by its full name):
    '(<GENE>[tiab] OR "<name>"[tiab]) AND (<domain fragment>)'.
    Empty for an unknown key.

    narrative._pubmed_domain_check builds the same string (viz_report_mcp cannot
    import this package) -- edit both together."""
    frag = DOMAIN_PUBMED.get(str(key or "").strip().lower(), "")
    if not frag:
        return ""
    g = str(gene or "").strip()
    if not g:
        return f"({frag})"
    nm = re.sub(r"\s+", " ", re.sub(r'["\[\]()]', " ", str(gene_name or ""))).strip()
    return (f'({g}[tiab] OR "{nm}"[tiab]) AND ({frag})' if nm
            else f"{g}[tiab] AND ({frag})")


def known_keys(keys) -> list[str]:
    """Corpus keys from a list or a comma-separated string: lower-cased, unknown
    keys dropped, deduplicated with the first occurrence kept (order preserved)."""
    if keys is None:
        return []
    if isinstance(keys, str):
        keys = keys.split(",")
    out: list[str] = []
    for k in keys:
        k = str(k).strip().lower()
        if k in CORPORA and k not in out:
            out.append(k)
    return out


def display_parts(key: str) -> tuple[str, str]:
    """(full name, abbreviation) for a known corpus key; ("", "") otherwise."""
    k = str(key).strip().lower()
    if k not in CORPORA:
        return "", ""
    return DISPLAY.get(k) or (CORPORA[k], "")


def display_name(key: str) -> str:
    """'Prostate cancer (PC)'; 'Lymphoma' when there is no abbreviation. An
    unknown key gives '' (report.html is not autoescaped, so only these
    code-defined names may reach it)."""
    full, abbr = display_parts(key)
    return f"{full} ({abbr})" if (full and abbr) else full


def _names_pancancer(explicit) -> bool:
    return any("pancancer" in match_item(it) for it in _items(explicit))


def user_domains(keys, explicit=None) -> list[str]:
    """The domains the user chose, in the user's order: `keys` (a
    resolve_literature_domains() result, list or comma string) without the
    automatically appended "pancancer". "pancancer" is kept (last) when the user
    named it in `explicit`, or when it is the only corpus that was resolved, so
    hiding it never empties the list. Unknown keys are dropped."""
    ks = known_keys(keys)
    rest = [k for k in ks if k != "pancancer"]
    if "pancancer" in ks and (not rest or _names_pancancer(explicit)):
        return rest + ["pancancer"]
    return rest


def domains_label(keys, include_pancancer: bool = False, sep: str = ", ", explicit=None) -> str:
    """'Prostate cancer (PC), Neuroendocrine carcinoma (NEC), Small cell lung
    cancer (SCLC)' in the given order. By default the auto-appended pan-cancer
    corpus is hidden (user_domains); include_pancancer=True prints the keys as
    given (use it for a list that is already user_domains()). '' when nothing
    known is left."""
    ks = known_keys(keys) if include_pancancer else user_domains(keys, explicit)
    return sep.join(n for n in (display_name(k) for k in ks) if n)


def mentions_domain(text: str, terms: Iterable[str], exclude: Iterable[str] = ()) -> bool:
    """Reference text gate for one domain (pure; copy it verbatim where this
    module cannot be imported). Every `exclude` match is blanked with spaces of
    equal length, then any `terms` match counts. All patterns are Python regexes
    applied with re.IGNORECASE; case-sensitive abbreviations carry a scoped
    (?-i:...) group."""
    t = str(text or "")
    for p in exclude:
        t = re.sub(p, lambda m: " " * len(m.group(0)), t, flags=re.I)
    return any(re.search(p, t, flags=re.I) for p in terms)


def text_mentions_domain(key: str, text: str) -> bool:
    """mentions_domain() with the DOMAIN_TERMS / DOMAIN_EXCLUDE of one corpus key."""
    specs = domain_specs([key])
    return bool(specs) and mentions_domain(text, specs[0]["terms"], specs[0]["exclude"])


def domain_specs(keys) -> list[dict]:
    """Plain, JSON-serialisable description of each domain, order kept, unknown
    keys dropped, for code that must not import mrag_orchestrator
    (viz_report_mcp). No pan-cancer hiding here: pass user_domains(...) (or
    shared["literature_domains_user"]) to get only the user's domains.

      {"key": "nec",                          # also the Qdrant payload value (Retriever.search(cancer=key))
       "label": "Neuroendocrine carcinoma",   # full name, sentence case
       "abbr": "NEC",                         # "" when none
       "display": "Neuroendocrine carcinoma (NEC)",
       "terms": [regex, ...], "exclude": [regex, ...],  # see mentions_domain()
       "veto": [regex, ...],                  # DOMAIN_CORPUS_VETO, [] for most domains
       "evfilter": {...},                     # nec only, evaudit 2026-09-27 (absent when off)
       "pubmed": '"Carcinoma, Neuroendocrine"[mh] OR ...'}   # see pubmed_query()
    """
    out: list[dict] = []
    for k in known_keys(keys):
        full, abbr = display_parts(k)
        out.append({"key": k, "label": full, "abbr": abbr, "display": display_name(k),
                    "terms": list(DOMAIN_TERMS.get(k) or (re.escape(CORPORA[k]),)),
                    "exclude": list(DOMAIN_EXCLUDE.get(k, ())),
                    "veto": list(DOMAIN_CORPUS_VETO.get(k, ())),
                    "pubmed": DOMAIN_PUBMED.get(k, "")})
        if k == "nec" and nec_rule_legacy():
            out[-1].update(terms=list(NEC_TERMS_LEGACY), veto=[],
                           exclude=[p for p in DOMAIN_EXCLUDE.get(k, ()) if p not in _NEC_EXCLUDE_NEW])
        # evaudit 2026-09-27: the nec evidence-filter data (1a paper-level NET
        # test, 1b sentence-level veto); the key is absent when switched off,
        # so a spec is then exactly the pre-2026-09-27 one
        _evf = _evfilter_spec(k)
        if _evf:
            out[-1]["evfilter"] = _evf
    return out


def manifest_user_domains(manifest) -> Optional[list[str]]:
    """User domains recorded in a run manifest (dict, or path to manifest.json):
    the last "literature_domains" step's "literature_domains_user" output (runs
    from 2026-09-17), else its "literature_domains" output with the
    auto-appended pan-cancer hidden (older runs; the step's params.explicit
    decides whether pan-cancer was the user's own choice). None when the
    manifest cannot be read or has no such step."""
    try:
        m = manifest if isinstance(manifest, dict) else json.loads(Path(manifest).read_text())
        for s in reversed(m.get("steps") or []):
            if not isinstance(s, dict) or s.get("name") != "literature_domains":
                continue
            outs = {o.get("key"): o.get("value", o.get("path"))
                    for o in (s.get("outputs") or []) if isinstance(o, dict)}
            ks = known_keys(outs.get("literature_domains_user"))
            if ks:
                return ks
            ks = user_domains(outs.get("literature_domains"), (s.get("params") or {}).get("explicit"))
            if ks:
                return ks
    except Exception:
        return None
    return None


def resolve_user_domains(params: Optional[dict] = None, manifest=None) -> list[str]:
    """User domains of a run, from the first source that yields any:
      1. params["literature_domains_user"]   (already the user list)
      2. params["literature_domains"]        (full search list; pan-cancer hidden)
      3. manifest step "literature_domains"  (finished runs replayed with old params)
      4. params["cancer_type"] / ["study_cancer_type"] resolved as at run time
      5. []                                   (caller shows "(not specified)")
    Never raises."""
    p = params if isinstance(params, dict) else {}
    steps = (lambda: known_keys(p.get("literature_domains_user")),
             lambda: user_domains(p.get("literature_domains"), p.get("literature_domains_explicit")),
             lambda: (manifest_user_domains(manifest) if manifest is not None else None),
             lambda: (user_domains(resolve_literature_domains(
                          None, p.get("cancer_type") or p.get("study_cancer_type"), None)[0])
                      if str(p.get("cancer_type") or p.get("study_cancer_type") or "").strip() else None))
    for step in steps:
        try:
            ks = step()
        except Exception:
            ks = None
        if ks:
            return list(ks)
    return []
