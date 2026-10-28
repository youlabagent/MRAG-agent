"""Mouse -> human orthology, and the disclosures that must travel with it.

WHY THIS EXISTS
---------------
MRAG's literature layer (curated target scoring, the Section-14 novelty cards,
the GeneRIF fetch, the live PubMed checks, the PaperQA corpora) is built on
HUMAN gene symbols and human disease literature. `libs/genegpt` even hard-codes
``<SYM>[sym] AND human[orgn]`` in its Entrez Gene lookup.

Run that layer on a mouse symbol and one of two things happens, both silent:

  * the symbol happens to be case-insensitively equal to the human one
    (``Onecut2`` -> human ONECUT2) and the run reports HUMAN literature as if it
    were mouse evidence, with nothing in the report saying so; or
  * it is a genuine rename (``Trp53`` -> human TP53, ``Nkx2-1`` -> NKX2-1) and
    every lookup returns nothing, so the gene silently scores 0 on evidence and
    drops out of the panel.

This module makes the first case HONEST and the second case WORK. Every mouse
symbol is mapped to its human ortholog before any literature call, and every
literature statement that results carries a disclosure naming the ortholog.
A mouse gene with NO human ortholog is reported as exactly that -- it is never
silently dropped and never silently scored as if the search had been run.

THE TABLE
---------
MGI ``HOM_MouseHumanSequence.rpt`` (Mouse Genome Informatics, Jackson Lab). It
is the smallest table that answers the question directly: two rows per
homology class, one mouse and one human, joined by "DB Class Key", so a mouse
SYMBOL maps to a human SYMBOL without a detour through Entrez ids. NCBI
``gene_orthologs.gz`` (124 MB) carries only GeneIDs and would need both
species' ``gene_info`` as well; HomoloGene has been unmaintained since 2014.

The .rpt is parsed once into a 2-column-plus-provenance TSV
(``mouse_human_orthologs.tsv``) so nothing downstream pays the 15 MB parse.

NOTE ON COORDINATES: the .rpt's "Genome Coordinates" column is GRCm39 for
mouse. We use only SYMBOLS and EntrezGene ids from it, never coordinates, so
the mm10/mm39 decision in Reference/README_mm10.md is not affected.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

ORTHOLOGY_DIR = Path(os.environ.get(
    "MRAG_ORTHOLOGY_DIR", "/BIO1/MRAG/Agent/Reference/orthology"))
#: NCBI human gene_info, used ONLY to resolve a nomenclature synonym on the
#: HUMAN side (see OrthologMap._alias_rescue). Absent -> the rescue is skipped.
HUMAN_GENE_INFO = Path(os.environ.get(
    "MRAG_HUMAN_GENE_INFO", "/BIO1/MRAG/MD_check/Homo_sapiens.gene_info.gz"))
#: A parsed table with fewer mouse symbols than this is not a usable map --
#: it is a truncated/renamed/empty file, and using it would report "no human
#: ortholog" for every gene while the banner still claimed each was mapped.
MIN_USABLE_SYMBOLS = 10000
MGI_REPORT = ORTHOLOGY_DIR / "HOM_MouseHumanSequence.rpt"
DERIVED_TSV = ORTHOLOGY_DIR / "mouse_human_orthologs.tsv"
SOURCE_URL = ("https://www.informatics.jax.org/downloads/reports/"
              "HOM_MouseHumanSequence.rpt")
SOURCE_NAME = "MGI HOM_MouseHumanSequence.rpt"

DERIVED_HEADER = ("mouse_symbol\thuman_symbol\tclass_key\tmouse_entrez\t"
                  "human_entrez\tn_human_for_mouse\tn_mouse_for_human")


class OrthologyError(RuntimeError):
    """The ortholog table is needed and is not usable."""


# ---------------------------------------------------------------------------
# Disclosure wording -- ONE place, so the report, the manifest, the card and
# the log all say the same thing.
# ---------------------------------------------------------------------------

#: Shown once, at the top of every literature-derived report section.
BANNER = (
    "Literature evidence on this run is HUMAN-ORTHOLOG literature. This is a "
    "{scientific_name} ({genome}) dataset, but MRAG's evidence layer -- GeneRIF, "
    "PubMed, the PaperQA corpora and the curated target scoring -- indexes human "
    "gene symbols and human disease literature. Each {common_name} gene was "
    "therefore mapped to its human ortholog ({source}) and every lookup was run "
    "on the HUMAN symbol. Read each statement below as evidence about the named "
    "human ortholog, not as a {common_name}-specific finding.")

#: Per gene, when an ortholog was found and used.
GENE_DISCLOSURE = (
    "Human-ortholog evidence: {query} ({common_name}) was searched as its human "
    "ortholog {human} ({source}). The statements above describe {human}.")

#: Per gene, when the mouse symbol and the human symbol differ only in case.
GENE_DISCLOSURE_SAMENAME = (
    "Human-ortholog evidence: {query} ({common_name}) was searched as its human "
    "ortholog {human}, which differs only in capitalisation ({source}). The "
    "statements above describe {human}.")

#: Per gene, when MGI lists more than one human ortholog.
GENE_DISCLOSURE_AMBIGUOUS = (
    "Human-ortholog evidence: {query} ({common_name}) has {n} human orthologs "
    "in {source} ({all_human}); {human} was searched. The statements above "
    "describe {human} only.")

#: Per gene, when the symbol is absent from the table but an NCBI human-side
#: synonym resolves it (nomenclature drift: GENCODE vM25 Arntl = MGI Bmal1).
GENE_DISCLOSURE_ALIAS = (
    "Human-ortholog evidence: {query} ({common_name}) is not listed under that "
    "spelling in {source}; NCBI human gene_info records {query_up} as a synonym "
    "of the human gene {human}, which {source} pairs with {common_name} {via}. "
    "{human} was searched, and the statements above describe {human}.")

#: Per gene, when NO human ortholog could be RESOLVED. Nothing was searched.
#: The wording deliberately describes what the lookup established, not biology:
#: the table is keyed on MGI's current mouse symbols, so a miss can mean the
#: gene has no human counterpart OR that this run's annotation spells it
#: differently than MGI does.
GENE_DISCLOSURE_NONE = (
    "No human ortholog resolved: no human ortholog could be resolved for "
    "{query} ({common_name}) from {source}, nor through an NCBI human-symbol "
    "synonym, so NO literature search was run for it. That is a limit of the "
    "ortholog table -- it is not a finding that {query} has no human "
    "counterpart. Any score or ranking shown for {query} rests on this run's "
    "ATAC and RNA evidence alone, with no literature support claimed either "
    "way.")

#: One line for the Pipeline-overview table.
OVERVIEW_LINE = (
    "human-ortholog route ({source}): {n_mapped}/{n_total} genes mapped, "
    "{n_unmapped} with no human ortholog")


def _fmt(tpl: str, **kw) -> str:
    return re.sub(r"\s+", " ", tpl.format(**kw)).strip()


# ---------------------------------------------------------------------------
# One lookup result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrthologHit:
    """What happened when one gene symbol was mapped to a human symbol.

    `relation` is the whole point -- it is what the report has to disclose:
      identity   : the run is already human; nothing was mapped.
      ortholog   : exactly one human ortholog; `human_symbol` is it.
      same_name  : one ortholog that differs only in capitalisation (Ar -> AR).
      ambiguous  : several human orthologs; `human_symbol` is the chosen one
                   -- the one whose name matches the query case-insensitively
                   when the class contains it (Cdkn2b -> CDKN2B, not CDKN2A),
                   otherwise alphabetically first for determinism.
                   `alternatives` is the rest; the disclosure names all of them.
      alias      : the query is absent from the table, but NCBI human
                   gene_info lists it as a synonym of a human gene the table
                   does pair with this species (Arntl -> BMAL1, MGI's mouse
                   spelling being Bmal1). `via` is that MGI spelling.
      none       : no human ortholog in the table. `human_symbol` is "" and NO
                   literature lookup may be performed.
    """
    query: str
    human_symbol: str
    relation: str
    source: str = SOURCE_NAME
    class_key: str = ""
    alternatives: tuple = ()
    common_name: str = "mouse"
    #: for relation "alias": the table's own spelling of this species' gene.
    via: str = ""

    @property
    def mapped(self) -> bool:
        """True when a human symbol may be used for a literature lookup."""
        return self.relation != "none" and bool(self.human_symbol)

    @property
    def is_identity(self) -> bool:
        return self.relation == "identity"

    @property
    def join_key(self) -> str:
        """Uppercase key for joining against a human-symbol namespace.

        Falls back to the uppercased query when there is no ortholog, because a
        JOIN wants a best effort; a LITERATURE LOOKUP does not and must use
        `mapped` instead.
        """
        return (self.human_symbol or self.query or "").upper()

    def disclosure(self) -> str:
        """The exact sentence the report must carry for this gene."""
        if self.relation == "identity":
            return ""
        kw = {"query": self.query, "human": self.human_symbol,
              "source": self.source, "common_name": self.common_name}
        if self.relation == "alias":
            return _fmt(GENE_DISCLOSURE_ALIAS, query_up=self.query.upper(),
                        via=self.via, **kw)
        if self.relation == "none":
            return _fmt(GENE_DISCLOSURE_NONE, **kw)
        if self.relation == "ambiguous":
            allh = ", ".join([self.human_symbol] + list(self.alternatives))
            return _fmt(GENE_DISCLOSURE_AMBIGUOUS, n=1 + len(self.alternatives),
                        all_human=allh, **kw)
        if self.relation == "same_name":
            return _fmt(GENE_DISCLOSURE_SAMENAME, **kw)
        return _fmt(GENE_DISCLOSURE, **kw)

    def as_dict(self) -> dict:
        return {"query": self.query, "human_symbol": self.human_symbol,
                "relation": self.relation, "source": self.source,
                "class_key": self.class_key, "via": self.via,
                "alternatives": list(self.alternatives),
                "mapped": self.mapped, "disclosure": self.disclosure()}


# ---------------------------------------------------------------------------
# Building the derived table
# ---------------------------------------------------------------------------

def parse_mgi(report_path=None) -> list[dict]:
    """MGI .rpt -> [{mouse_symbol, human_symbol, class_key, ...}] rows.

    The .rpt is one row per (homology class, organism). We group by class key
    and emit the mouse x human cross-product of each class, so a class with two
    mouse paralogs and one human gene yields two rows.
    """
    p = Path(report_path or MGI_REPORT)
    if not p.is_file():
        raise OrthologyError(
            "the mouse-human ortholog table is not installed: %s is missing. "
            "It is a %s download from %s (~15 MB). Without it a mouse run "
            "cannot map its genes to the human symbols the literature layer "
            "indexes." % (p, SOURCE_NAME, SOURCE_URL))
    classes: dict[str, dict[str, list[tuple]]] = {}
    _ORG_LABELS_SEEN: set = set()
    with p.open(encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            i_key = header.index("DB Class Key")
            i_org = header.index("Common Organism Name")
            i_sym = header.index("Symbol")
            i_eg = header.index("EntrezGene ID")
        except ValueError as e:
            raise OrthologyError(
                "%s does not have the expected MGI columns (DB Class Key / "
                "Common Organism Name / Symbol / EntrezGene ID); got %s"
                % (p, header)) from e
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) <= max(i_key, i_org, i_sym, i_eg):
                continue
            key, org, sym, eg = f[i_key].strip(), f[i_org].strip().lower(), \
                f[i_sym].strip(), f[i_eg].strip()
            if not key or not sym:
                continue
            _ORG_LABELS_SEEN.add(org)
            side = "mouse" if org.startswith("mouse") else (
                "human" if org.startswith("human") else "")
            if not side:
                continue
            classes.setdefault(key, {"mouse": [], "human": []})[side].append(
                (sym, eg))
    rows = []
    for key, sides in classes.items():
        mm, hh = sides["mouse"], sides["human"]
        if not mm or not hh:
            continue          # mouse-only or human-only class: no ortholog
        for ms, me in mm:
            for hs, he in hh:
                rows.append({"mouse_symbol": ms, "human_symbol": hs,
                             "class_key": key, "mouse_entrez": me,
                             "human_entrez": he,
                             "n_human_for_mouse": len(hh),
                             "n_mouse_for_human": len(mm)})
    rows.sort(key=lambda r: (r["mouse_symbol"].lower(), r["human_symbol"]))
    # An empty or implausibly small parse is NOT a valid table. It used to be
    # returned as one, and every gene then came back "no human ortholog" while
    # the report banner still said each gene had been mapped (2026-09-23).
    # The commonest cause is MGI renaming its "Common Organism Name" values,
    # which the loop above skips silently -- so say which labels were seen.
    n_mouse = len({r["mouse_symbol"].lower() for r in rows})
    if n_mouse < MIN_USABLE_SYMBOLS:
        seen = sorted(_ORG_LABELS_SEEN)[:10]
        raise OrthologyError(
            "%s parsed to only %d ortholog pair(s) over %d %s symbols, far "
            "below the %d expected. The organism labels actually found in the "
            "file were %s (rows are kept only for labels starting 'mouse' or "
            "'human'). A table this small would report 'no human ortholog' for "
            "every gene while the report still claimed each one was mapped, so "
            "it is refused here."
            % (p, len(rows), n_mouse, "mouse", MIN_USABLE_SYMBOLS, seen))
    return rows


def build_derived(report_path=None, out_path=None) -> Path:
    """Parse the MGI report once into the small TSV everything else reads."""
    rows = parse_mgi(report_path)
    out = Path(out_path or DERIVED_TSV)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write("# source: %s (%s)\n" % (SOURCE_NAME, SOURCE_URL))
        fh.write("# mouse<->human ortholog pairs derived by "
                 "mrag_orchestrator.orthology.build_derived\n")
        fh.write(DERIVED_HEADER + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in
                               DERIVED_HEADER.split("\t")) + "\n")
    tmp.replace(out)
    return out


# ---------------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------------

@dataclass
class OrthologMap:
    """mouse symbol -> human symbol, plus the provenance the report discloses.

    `IDENTITY` (see below) is the human-run instance: every lookup returns the
    symbol unchanged with relation "identity" and an EMPTY disclosure, so the
    human code path is literally unchanged text.
    """
    common_name: str = "mouse"
    source: str = SOURCE_NAME
    table_path: str = ""
    identity: bool = False
    _by_query: dict = field(default_factory=dict, repr=False)
    #: HUMAN symbol (upper) -> the rows of its class, for the alias rescue.
    _by_human: dict = field(default_factory=dict, repr=False)
    #: lazily built {HUMAN SYNONYM (upper) -> {official human symbols}}.
    _alias_idx: object = field(default=None, repr=False)

    # -- construction ------------------------------------------------------
    @classmethod
    def identity_map(cls, common_name: str = "human") -> "OrthologMap":
        return cls(common_name=common_name, source="", table_path="",
                   identity=True)

    @classmethod
    def from_tsv(cls, path=None, common_name: str = "mouse") -> "OrthologMap":
        p = Path(path or DERIVED_TSV)
        if not p.is_file():
            # Build it on demand from the .rpt when that is present.
            if Path(MGI_REPORT).is_file() and str(p) == str(DERIVED_TSV):
                build_derived()
            if not p.is_file():
                raise OrthologyError(
                    "no mouse-human ortholog table at %s (and no %s to build "
                    "it from). Install it from %s." % (p, MGI_REPORT, SOURCE_URL))
        by_query: dict[str, list[dict]] = {}
        by_human: dict[str, list[dict]] = {}
        with p.open(encoding="utf-8") as fh:
            cols = None
            for line in fh:
                if line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                if cols is None:
                    cols = f
                    continue
                if len(f) < len(cols):
                    continue
                rec = dict(zip(cols, f))
                q = rec["mouse_symbol"].strip().lower()
                if not q:
                    continue
                by_query.setdefault(q, []).append(rec)
                h = rec.get("human_symbol", "").strip().upper()
                if h:
                    by_human.setdefault(h, []).append(rec)
        # A table that is PRESENT but carries no usable rows used to pass every
        # gate: available() only stats the path, so the run proceeded and then
        # reported "no human ortholog" for every gene while the Section-14
        # banner still claimed each had been mapped (2026-09-23).
        if len(by_query) < MIN_USABLE_SYMBOLS:
            raise OrthologyError(
                "the ortholog table %s is present but carries only %d %s "
                "symbol(s) -- far below the %d expected, so it is not usable. "
                "Every gene would come back 'no human ortholog' while the "
                "report still said each one was mapped to a human ortholog. "
                "Rebuild it from %s (%s)."
                % (p, len(by_query), common_name, MIN_USABLE_SYMBOLS,
                   MGI_REPORT, SOURCE_URL))
        return cls(common_name=common_name, source=SOURCE_NAME,
                   table_path=str(p), _by_query=by_query, _by_human=by_human)

    # -- the human-synonym rescue -------------------------------------------
    def _human_alias_index(self) -> dict:
        """{HUMAN SYNONYM (upper) -> {official human symbols}} from NCBI.

        Built once, on the first miss, and only for a non-identity map. The
        file is optional: without it the rescue simply never fires.
        """
        if self._alias_idx is None:
            idx: dict = {}
            gi = Path(HUMAN_GENE_INFO)
            try:
                import gzip
                opener = gzip.open if gi.suffix == ".gz" else open
                with opener(gi, "rt", encoding="utf-8", errors="replace") as fh:
                    fh.readline()                       # header
                    for line in fh:
                        f = line.rstrip("\n").split("\t")
                        if len(f) < 5:
                            continue
                        sym, syn = f[2].strip(), f[4].strip()
                        if not sym or sym == "-" or not syn or syn == "-":
                            continue
                        for s in syn.split("|"):
                            s = s.strip().upper()
                            if s and s != "-":
                                idx.setdefault(s, set()).add(sym)
            except Exception as e:                      # never fatal
                print("[orthology] human synonym index unavailable (%s: %s); "
                      "nomenclature-drift rescue disabled"
                      % (type(e).__name__, str(e)[:120]), flush=True)
                idx = {}
            self._alias_idx = idx
        return self._alias_idx

    def _alias_rescue(self, q: str):
        """(human_symbol, table_spelling, class_key) or None.

        The table is keyed on MGI's CURRENT mouse symbols while the run's
        symbols come from GENCODE vM25 (2020), so nomenclature drift shows up
        as a miss -- and a miss was being reported as "this gene has no human
        ortholog" (2026-09-23). ``Arntl`` is the clean case: MGI calls the
        mouse gene ``Bmal1`` and the human gene ``BMAL1``, and NCBI lists
        ARNTL among BMAL1's synonyms.

        DELIBERATELY STRICT, because a name match is weaker evidence than a
        homology class: the synonym must resolve to EXACTLY ONE official human
        symbol, and that symbol must already be paired with this species in
        the table. ``Car1`` (synonym of both CA1 and NR1I3) is therefore left
        unresolved rather than guessed at.
        """
        if self.identity or not self._by_human:
            return None
        cands = self._human_alias_index().get(str(q).strip().upper())
        if not cands or len(cands) != 1:
            return None
        human = next(iter(cands))
        recs = self._by_human.get(human.upper())
        if not recs:
            return None
        return (recs[0]["human_symbol"].strip(),
                recs[0].get("mouse_symbol", "").strip(),
                recs[0].get("class_key", ""))

    # -- lookup ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._by_query)

    def to_human(self, symbol) -> OrthologHit:
        """One symbol -> OrthologHit. Never raises, never returns None."""
        q = str(symbol or "").strip()
        if self.identity:
            return OrthologHit(query=q, human_symbol=q, relation="identity",
                               source="", common_name=self.common_name)
        recs = self._by_query.get(q.lower(), [])
        if not recs:
            _al = self._alias_rescue(q)
            if _al:
                _h, _via, _ck = _al
                return OrthologHit(query=q, human_symbol=_h, relation="alias",
                                   source=self.source, class_key=_ck, via=_via,
                                   common_name=self.common_name)
            return OrthologHit(query=q, human_symbol="", relation="none",
                               source=self.source, common_name=self.common_name)
        humans = sorted({r["human_symbol"].strip() for r in recs
                         if r["human_symbol"].strip()})
        if not humans:
            return OrthologHit(query=q, human_symbol="", relation="none",
                               source=self.source, common_name=self.common_name)
        # When a homology class holds several human genes, the one that CARRIES
        # THE QUERY'S OWN NAME is the ortholog meant; taking the alphabetically
        # first one sent mouse Cdkn2b's literature to CDKN2A, Ccl3's to CCL18
        # and Chrna7's to CHRFAM7A (2026-09-23). Alphabetical order remains the
        # tie-break when no candidate matches the name, so the choice stays
        # deterministic.
        chosen = next((h for h in humans if h.upper() == q.upper()), humans[0])
        ck = next((r.get("class_key", "") for r in recs
                   if r["human_symbol"].strip() == chosen), "")
        if len(humans) > 1:
            return OrthologHit(query=q, human_symbol=chosen,
                               relation="ambiguous", source=self.source,
                               class_key=ck,
                               alternatives=tuple(h for h in humans
                                                  if h != chosen),
                               common_name=self.common_name)
        rel = "same_name" if chosen.upper() == q.upper() else "ortholog"
        return OrthologHit(query=q, human_symbol=chosen, relation=rel,
                           source=self.source, class_key=ck,
                           common_name=self.common_name)

    def map_many(self, symbols: Iterable) -> dict:
        """{symbol -> OrthologHit} preserving the caller's spelling as the key."""
        return {str(s): self.to_human(s) for s in symbols if str(s or "").strip()}

    def join_keys(self, symbols: Iterable) -> list[str]:
        """Uppercase human-namespace keys, best effort (see OrthologHit.join_key)."""
        return [self.to_human(s).join_key for s in symbols]

    def summarise(self, symbols: Iterable) -> dict:
        """Counts + the unmapped list, for the manifest and the §2 overview."""
        hits = self.map_many(symbols)
        unmapped = sorted(h.query for h in hits.values() if not h.mapped)
        amb = sorted(h.query for h in hits.values() if h.relation == "ambiguous")
        via_alias = sorted(h.query for h in hits.values()
                           if h.relation == "alias")
        return {
            "source": self.source, "table": self.table_path,
            "identity": self.identity,
            "n_total": len(hits), "n_mapped": len(hits) - len(unmapped),
            "n_unmapped": len(unmapped), "unmapped": unmapped,
            "n_ambiguous": len(amb), "ambiguous": amb,
            "n_via_synonym": len(via_alias), "via_synonym": via_alias,
            "overview_line": "" if self.identity else _fmt(
                OVERVIEW_LINE, source=self.source, n_total=len(hits),
                n_mapped=len(hits) - len(unmapped), n_unmapped=len(unmapped)),
        }

    def banner(self, scientific_name: str, genome: str) -> str:
        """The one-paragraph disclosure for the top of a literature section."""
        if self.identity:
            return ""
        return _fmt(BANNER, scientific_name=scientific_name, genome=genome,
                    common_name=self.common_name, source=self.source)


IDENTITY = OrthologMap.identity_map("human")


# ---------------------------------------------------------------------------
# Entry point used by the pipeline
# ---------------------------------------------------------------------------

def get_map(organism="human", table_path=None, required: bool = False
            ) -> OrthologMap:
    """The ortholog map for an organism key / profile / common name.

    Human (and anything already in the human symbol namespace) gets the
    IDENTITY map: same symbol, relation "identity", empty disclosure -- so the
    human code path produces byte-identical text to before this module existed.

    Mouse gets the MGI map. `required=False` (the default) degrades to the
    identity map when the table is not installed, so a pipeline that only wants
    a best-effort JOIN keeps working; the literature layer passes required=True
    because presenting human literature without the ortholog provenance is
    exactly the failure this module exists to prevent.
    """
    key = getattr(organism, "key", organism)
    key = str(key or "human").strip().lower()
    if key in ("human", "homo sapiens", "hg38", "grch38", "hs", "9606"):
        return IDENTITY
    common = getattr(organism, "common_name", None) or key
    try:
        return OrthologMap.from_tsv(table_path, common_name=str(common))
    except OrthologyError:
        if required:
            raise
        return OrthologMap(common_name=str(common), source=SOURCE_NAME,
                           table_path=str(table_path or DERIVED_TSV))


def available(table_path=None) -> bool:
    """True when a USABLE ortholog table is on disk (derived or buildable).

    "Usable" means it actually carries orthologs. A present-but-empty table
    used to pass this check -- it only stat()ed the path -- and the run then
    reported "no human ortholog" for every gene while the report banner still
    said each gene had been mapped (2026-09-23). Counting rows costs one cheap
    pass over a 1 MB TSV, so the check is done for real.
    """
    p = Path(table_path or DERIVED_TSV)
    if p.is_file():
        try:
            n = 0
            with p.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith("#") or not line.strip():
                        continue
                    n += 1
                    if n > MIN_USABLE_SYMBOLS:      # header + rows; stop early
                        return True
            return False
        except OSError:
            return False
    return Path(MGI_REPORT).is_file()
