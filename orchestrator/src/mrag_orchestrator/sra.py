"""SRA/ENA fetch: accession -> metadata -> analyzability verdict -> FASTQ download.

The chat agent gets an accession (a run, a study, or a GEO series) and expands it
to the whole study, because MRAG needs ATAC + RNA across at least two arms and a
single run can never satisfy that. Everything the go/no-go decision needs comes
from ENA metadata, so nothing is downloaded until the user has seen the sample
table and the total size.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from .detector import _classify_arm_rule, _classify_replicate_rule

ENA_API = "https://www.ebi.ac.uk/ena/portal/api/filereport"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DATA_ROOT = Path("/BIO1/MRAG/MRAG_data")
FASTERQ = "/BIO1/sratoolkit.3.1.1-ubuntu64/bin/fasterq-dump"

ENA_FIELDS = [
    "run_accession", "study_accession", "secondary_study_accession",
    "sample_accession", "experiment_accession", "sample_title", "sample_alias",
    "library_strategy", "library_layout", "library_source",
    "instrument_platform", "scientific_name", "read_count", "base_count",
    "fastq_ftp", "fastq_bytes", "fastq_md5",
]

# ENA library_strategy -> MRAG modality. Anything absent here is not analysable.
STRATEGY_TO_MODALITY = {
    "ATAC-seq": "ATAC",
    "DNase-Hypersensitivity": "ATAC",
    "RNA-Seq": "RNA",
}

SUPPORTED_SPECIES = {"Homo sapiens", "Mus musculus"}

RUN_RE = re.compile(r"^(SRR|ERR|DRR)[0-9]{5,}$", re.I)
STUDY_RE = re.compile(r"^(SRP|ERP|DRP)[0-9]{5,}$", re.I)
PRJ_RE = re.compile(r"^PRJ(NA|EB|DB)[0-9]+$", re.I)
GSE_RE = re.compile(r"^GSE[0-9]+$", re.I)
GSM_RE = re.compile(r"^GSM[0-9]+$", re.I)


class SRAError(RuntimeError):
    """Accession could not be resolved, or a remote lookup kept failing."""


def _ncbi_creds():
    """Shared NCBI credentials loader (libs/ncbi_creds.py); None when absent,
    in which case every caller keeps the anonymous 3 req/s behaviour."""
    try:
        import sys as _sys
        if "/BIO1/MRAG/Agent/libs" not in _sys.path:
            _sys.path.insert(0, "/BIO1/MRAG/Agent/libs")
        import ncbi_creds as _c
        return _c
    except Exception:
        return None


def _http(url: str, timeout: float = 30.0, attempts: int = 3) -> str:
    """GET with retries. NCBI allows 3 req/s anonymously and 10 req/s with the
    shared api_key (ncbi_creds, 2026-09-20); ENA occasionally 503s.

    The api_key is attached to NCBI URLs here and stripped again from the
    failure message, which quotes the URL and would otherwise put the secret
    into an exception that the chat agent prints."""
    creds = _ncbi_creds()
    sent = creds.add_params(url) if creds is not None else url
    ncbi = "ncbi.nlm.nih.gov" in url
    last = None
    tried = 0
    for i in range(attempts):
        tried = i + 1
        if creds is not None and ncbi:
            creds.SHARED.wait()
        try:
            with urllib.request.urlopen(sent, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001 - retried below, raised at the end
            last = exc
            code = getattr(exc, "code", None)
            # A 4xx means the REQUEST is wrong, not that the server is busy: ENA
            # answers 400 for an accession it cannot parse, and retrying that
            # three times with growing sleeps only makes the user wait ~45s for
            # a foregone failure (2026-09-23). 429 is the one 4xx a wait fixes.
            if isinstance(code, int) and 400 <= code < 500 and code != 429:
                break
            time.sleep(1.0 * (i + 1))
    # `tried`, not `attempts`: a 4xx breaks out after one try and claiming three
    # would misdescribe both the failure and how long the user waited.
    msg = ("request failed after %d attempt%s: %s (%s)"
           % (tried, "" if tried == 1 else "s", url, last))
    raise SRAError(creds.redact(msg) if creds is not None else msg)


def classify_accession(acc: str) -> str:
    """Return one of run/study/bioproject/gse/gsm, or raise."""
    a = acc.strip()
    if RUN_RE.match(a):
        return "run"
    if STUDY_RE.match(a):
        return "study"
    if PRJ_RE.match(a):
        return "bioproject"
    if GSE_RE.match(a):
        return "gse"
    if GSM_RE.match(a):
        return "gsm"
    raise SRAError(
        "unrecognised accession %r (expected SRR/ERR/DRR, SRP/ERP/DRP, PRJNA, GSE or GSM)" % acc)


def _geo_to_study(acc: str) -> str:
    """GEO series/sample -> an accession ENA understands.

    GEO accessions are not indexed in SRA, so go through the GDS record and take
    its BioProject, which ENA accepts directly.
    """
    q = urllib.parse.urlencode({"db": "gds", "term": acc + "[ACCN]",
                               "retmax": "5", "retmode": "json"})
    ids = json.loads(_http(EUTILS + "/esearch.fcgi?" + q)).get(
        "esearchresult", {}).get("idlist", [])
    if not ids:
        raise SRAError("%s is not in GEO" % acc)
    if _ncbi_creds() is None:
        time.sleep(0.4)          # anonymous NCBI ceiling; _http spaces the rest
    q2 = urllib.parse.urlencode({"db": "gds", "id": ",".join(ids), "retmode": "json"})
    res = json.loads(_http(EUTILS + "/esummary.fcgi?" + q2)).get("result", {})
    uids = [k for k in res if k != "uids"]
    for uid in uids:
        rec = res[uid]
        if (rec.get("accession") or "").upper() != acc.upper():
            continue
        proj = (rec.get("bioproject") or "").strip()
        if proj:
            return proj
    for uid in uids:
        proj = (res[uid].get("bioproject") or "").strip()
        if proj:
            return proj
    raise SRAError("%s has no linked BioProject; it may have no raw sequencing data" % acc)


def resolve_study(acc: str) -> tuple[str, str]:
    """Expand any accepted accession to (study, kind_of_input).

    Deliberately does NOT rewrite a study or BioProject the caller supplied.
    ENA reports a per-sub-study accession on every row, so canonicalising from
    row 0 silently truncates umbrella projects: PRJNA470707 holds 46 runs across
    SRP145125 (ChIP), SRP145126 (RNA) and SRP198619 (ChIP), and rewriting it to
    any single row would drop two thirds of them, modality included.
    """
    kind = classify_accession(acc)
    a = acc.strip().upper()
    if kind in ("study", "bioproject"):
        return a, kind
    if kind in ("gse", "gsm"):
        return _geo_to_study(a), kind
    rows = _ena(a, ["study_accession", "secondary_study_accession"])
    if not rows:
        raise SRAError("ENA knows no run %s" % acc)
    study = ((rows[0].get("secondary_study_accession") or "").strip()
             or (rows[0].get("study_accession") or "").strip())
    if not study:
        raise SRAError("run %s has no parent study in ENA" % acc)
    return study, kind


def _ena(accession: str, fields: list[str]) -> list[dict]:
    """One ENA filereport call -> list of row dicts (empty when nothing matches)."""
    q = urllib.parse.urlencode({
        "accession": accession, "result": "read_run",
        "fields": ",".join(fields), "format": "tsv",
    })
    text = _http(ENA_API + "?" + q)
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return []
    head = lines[0].split(chr(9))
    return [dict(zip(head, l.split(chr(9)))) for l in lines[1:]]


def fetch_runs(study: str) -> list[dict]:
    rows = _ena(study, ENA_FIELDS)
    if not rows:
        raise SRAError("ENA returned no runs for study %s" % study)
    return sorted(rows, key=lambda r: r.get("run_accession", ""))


def _int(v: str) -> int:
    try:
        return int(str(v).strip() or 0)
    except ValueError:
        return 0


def annotate(rows: list[dict]) -> list[dict]:
    """Attach modality / arm / replicate / layout / size to each ENA row.

    Arm and replicate come from the submitter sample title, reusing the same rules
    the FASTQ detector applies to filenames so the two stages agree.
    """
    out = []
    for r in rows:
        title = (r.get("sample_title") or r.get("sample_alias") or "").strip()
        strategy = (r.get("library_strategy") or "").strip()
        modality = STRATEGY_TO_MODALITY.get(strategy, "")
        if not modality:
            # Some submitters leave strategy as OTHER and only say it in the title.
            if re.search(r"ATAC", title, re.I):
                modality = "ATAC"
            elif re.search(r"RNA[-_ ]?seq|transcriptom", title, re.I):
                modality = "RNA"
        # Underscore is a word character, so the \bDHT\b style rules never fire
        # on "DHT_L001". Submitter titles are underscore-heavy; filenames are not.
        probe = title.replace("_", "-")
        arm, arm_conf = _classify_arm_rule(probe)
        rep, _ = _classify_replicate_rule(probe)
        ftp = [u for u in (r.get("fastq_ftp") or "").split(";") if u]
        sizes = [_int(b) for b in (r.get("fastq_bytes") or "").split(";") if b]
        md5s = [m for m in (r.get("fastq_md5") or "").split(";") if m]
        layout = (r.get("library_layout") or "").strip().upper()
        if not layout:
            layout = "PAIRED" if len(ftp) >= 2 else "SINGLE"
        out.append({
            "run": r.get("run_accession", ""),
            "title": title,
            "strategy": strategy,
            "modality": modality,
            "arm": arm,
            "arm_confidence": arm_conf,
            "replicate": rep,
            "layout": layout,
            "species": (r.get("scientific_name") or "").strip(),
            "platform": (r.get("instrument_platform") or "").strip(),
            "read_count": _int(r.get("read_count")),
            "fastq_ftp": ftp,
            "fastq_md5": md5s,
            # Per-file sizes are kept alongside the sum so fetch() can tell a
            # COMPLETE file from one a killed download left truncated
            # (2026-09-23 fix).  Same order as fastq_ftp.
            "fastq_sizes": sizes,
            "bytes": sum(sizes),
        })
    return out


def _human_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return "%.1f %s" % (x, unit)
        x /= 1024
    return "%.1f TB" % x




# Assay names that show up inside arm labels when a submitter titles samples by
# modality. Bounded so shRNA, snRNA and similar survive the strip.
_MODALITY_TOKEN_RE = re.compile(
    r"\b(?:sc|sn|bulk)?[\s_-]?(?:ATAC|RNA)(?:[\s_-]?seq)?\b"
    r"|\bDNase(?:[\s_-]?seq)?\b"
    r"|\btranscriptom\w*\b"
    r"|\bchromatin[\s_-]?accessibility\b", re.I)


def _strip_modality(label: str) -> str:
    """Arm label with assay words and punctuation removed, for comparing arms."""
    return re.sub(r"[^A-Za-z0-9]+", "", _MODALITY_TOKEN_RE.sub(" ", label)).lower()

def collapse_assay_arms(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    """Merge arms that differ only by assay type.

    When ATAC and RNA come from separate submissions the titles carry the assay
    name, and the model then labels "X ATAC-seq" and "X RNA-seq" as two arms.
    They are one biological condition measured two ways, and leaving them split
    gives every arm a single modality, which no differential test can use.
    """
    usable = [s for s in samples
              if s["modality"] in ("ATAC", "RNA") and s["arm"]]
    groups: dict[str, list[dict]] = {}
    for s in usable:
        groups.setdefault(_strip_modality(s["arm"]), []).append(s)
    merges: list[dict] = []
    for key, members in groups.items():
        labels = sorted({m["arm"] for m in members})
        if len(labels) < 2 or not key:
            continue
        canon = min((_MODALITY_TOKEN_RE.sub(" ", l) for l in labels), key=len)
        canon = re.sub(r"\s+", " ", canon).strip(" ,-_") or labels[0]
        for m in members:
            m["arm_before_merge"] = m["arm"]
            m["arm"] = canon
        merges.append({"from": labels, "to": canon})
    return samples, merges


def verdict(samples: list[dict]) -> dict:
    """Decide whether this study can drive an MRAG run, and say why not.

    MRAG needs accessible-chromatin and expression data for at least two arms.
    Anything that fails is reported as unusable with the specific reason, so the
    caller never starts a multi-hundred-gigabyte download on a hunch.
    """
    usable = [s for s in samples if s["modality"] in ("ATAC", "RNA")]
    dropped = [s for s in samples if s["modality"] not in ("ATAC", "RNA")]
    n_atac = sum(1 for s in usable if s["modality"] == "ATAC")
    n_rna = sum(1 for s in usable if s["modality"] == "RNA")
    reasons: list[str] = []
    # Which half of the experiment is absent, named so the caller can tell the user
    # what to ADD instead of only that the study "is not analysable".
    missing: list[str] = []

    if not usable:
        strategies = sorted({s["strategy"] or "?" for s in samples})
        missing = ["ATAC", "RNA"]
        reasons.append(
            "no ATAC-seq or RNA-Seq libraries in this study (found: %s)"
            % ", ".join(strategies))
    else:
        if n_atac == 0:
            missing.append("ATAC")
            reasons.append(
                "no ATAC-seq / DNase library; MRAG cannot score chromatin "
                "accessibility. Add the ATAC-seq accession and look both up in ONE "
                "call (sra-lookup ATAC_ACC RNA_ACC): the ATAC and the RNA half of "
                "one experiment are routinely deposited as separate studies.")
        if n_rna == 0:
            missing.append("RNA")
            reasons.append(
                "no RNA-Seq library; MRAG cannot compute differential expression. "
                "Add the RNA-seq accession and look both up in ONE call "
                "(sra-lookup ATAC_ACC RNA_ACC): the ATAC and the RNA half of one "
                "experiment are routinely deposited as separate studies.")

    # Arm assignment. Per project policy this is a hard gate: an unlabelled sample
    # would silently land in the wrong condition, so we stop instead of guessing.
    unassigned = [s for s in usable if not s["arm"]]
    arms = sorted({s["arm"] for s in usable if s["arm"]})
    if unassigned:
        shown = ", ".join("%s (%s)" % (s["run"], s["title"] or "no title") for s in unassigned[:6])
        more = "" if len(unassigned) <= 6 else " and %d more" % (len(unassigned) - 6)
        reasons.append("could not infer the experimental arm for %d sample(s): %s%s"
                       % (len(unassigned), shown, more))
    elif len(arms) < 2:
        reasons.append("only one arm detected (%s); MRAG compares two conditions"
                       % (arms[0] if arms else "none"))

    species = sorted({s["species"] for s in usable if s["species"]})
    off = [sp for sp in species if sp not in SUPPORTED_SPECIES]
    if off:
        reasons.append("unsupported species: %s (references exist for human and mouse only)"
                       % ", ".join(off))

    per_arm: dict[str, dict[str, int]] = {}
    for s in usable:
        if not s["arm"]:
            continue
        per_arm.setdefault(s["arm"], {"ATAC": 0, "RNA": 0})[s["modality"]] += 1


    # MRAG contrasts two arms per modality, so each layer needs at least two arms
    # carrying it. This also catches the failure where the labels are really the
    # assay type (arms "H9 cells, ATAC-seq" / "H9 cells, RNA-seq" are one condition
    # split by modality, not two comparable conditions).
    if per_arm:
        with_atac = [a for a, c in per_arm.items() if c["ATAC"]]
        with_rna = [a for a, c in per_arm.items() if c["RNA"]]
        if len(with_atac) < 2 and n_atac:
            reasons.append("only %d arm carries ATAC data (%s); a differential "
                           "accessibility test needs two"
                           % (len(with_atac), ", ".join(with_atac) or "none"))
        if len(with_rna) < 2 and n_rna:
            reasons.append("only %d arm carries RNA data (%s); a differential "
                           "expression test needs two"
                           % (len(with_rna), ", ".join(with_rna) or "none"))
        bare = {_strip_modality(a) for a in per_arm}
        if len(per_arm) >= 2 and len(bare) < 2:
            reasons.append("the arm labels differ only by assay type (%s), so this "
                           "is one condition profiled two ways rather than two arms"
                           % ", ".join(sorted(per_arm)))
    warnings: list[str] = []
    for arm, counts in sorted(per_arm.items()):
        for mod in ("ATAC", "RNA"):
            if counts[mod] == 1:
                warnings.append("arm %s has a single %s replicate; statistics will be weak"
                                % (arm, mod))
            elif counts[mod] == 0:
                warnings.append("arm %s has no %s sample" % (arm, mod))
    if dropped:
        warnings.append("%d run(s) ignored as non-ATAC/RNA: %s"
                        % (len(dropped), ", ".join(sorted({d["strategy"] or "?" for d in dropped}))))

    total = sum(s["bytes"] for s in usable)
    # The species was computed above only to reject unsupported ones and was
    # then thrown away, which is why nothing downstream ever knew the organism
    # (2026-09-23). Publish it, plus the canonical MRAG key when it resolves.
    organism = ""
    if len(species) == 1:
        try:
            from .organism import resolve as _resolve_org
            organism = _resolve_org(species[0]).key
        except Exception:
            organism = ""
    return {
        "status": "unusable" if reasons else "ok",
        "species": species,
        "organism": organism,
        "reasons": reasons,
        "missing": missing,
        "warnings": warnings,
        "n_atac": n_atac,
        "n_rna": n_rna,
        "arms": arms,
        "per_arm": per_arm,
        "n_dropped": len(dropped),
        "total_bytes": total,
        "total_human": _human_bytes(total),
    }


def _safe(label: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-")
    return s or "sample"


CACHE_DIR = DATA_ROOT / ".sra_cache"
CACHE_TTL = 86400.0  # seconds; an approval older than a day should be re-checked


def _cache_path(study: str) -> Path:
    return CACHE_DIR / (_safe(study) + ".json")


def _runs_key(runs) -> str:
    """Short, stable cache suffix for an explicit run selection."""
    uniq = sorted({str(r).strip().upper() for r in (runs or []) if str(r).strip()})
    joined = "-".join(uniq)
    if len(joined) <= 48:
        return joined or "none"
    import hashlib
    return "%dx%s" % (len(uniq), hashlib.sha1(joined.encode()).hexdigest()[:10])


def _cache_write(rep: dict) -> None:
    """Persist a lookup so fetch downloads exactly what the user approved.

    Keyed on cache_key, not on the study: a run subset and the whole study share
    a study name but are different approvals, and reusing one for the other
    would download runs the user never saw (2026-09-23).
    """
    try:
        p = _cache_path(rep.get("cache_key") or rep["study"])
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(rep)
        payload["_cached_at"] = time.time()
        p.write_text(json.dumps(payload, indent=2))
    except OSError:
        pass  # a cache miss later is harmless; never fail a lookup over this


def _cache_read(study: str, max_age: float = CACHE_TTL) -> Optional[dict]:
    p = _cache_path(study)
    if not p.exists():
        return None
    try:
        rep = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    age = time.time() - float(rep.get("_cached_at") or 0)
    if age > max_age:
        return None
    # Reports cached BEFORE species/organism were published at the top level
    # (2026-09-23) have neither key, and a cache entry lives a day -- so the
    # chat would have told the user "species not stated" about an unambiguously
    # Mus musculus study, moments before they authorised a 37 GB download.
    # The per-sample `species` field has always been written, so derive the
    # missing top-level keys from it rather than forcing a re-fetch.
    if "species" not in rep or "organism" not in rep:
        _sp = sorted({(s.get("species") or "").strip()
                      for s in (rep.get("samples") or [])
                      if (s.get("species") or "").strip()})
        rep.setdefault("species", _sp)
        if not rep.get("organism"):
            rep["organism"] = ""
            if len(_sp) == 1:
                try:
                    from .organism import resolve as _resolve_org2
                    rep["organism"] = _resolve_org2(_sp[0]).key
                except Exception:
                    rep["organism"] = ""
        rep["species_backfilled_from_samples"] = True
    rep["from_cache"] = True
    rep["cache_age_s"] = int(age)
    return rep


def assign_names(samples: list[dict]) -> list[dict]:
    """Give every run a destination basename the FASTQ detector can read back.

    Layout is <modality>/<arm>_rep<N>, because detect() takes the parent directory
    as the modality hint and _R1/_R2 as the pairing hint.
    """
    groups: dict[tuple, list[dict]] = {}
    for s in samples:
        if s["modality"] in ("ATAC", "RNA"):
            groups.setdefault((s["modality"], s["arm"]), []).append(s)
    for (_mod, arm), members in groups.items():
        members.sort(key=lambda x: x["run"])
        reps = [m["replicate"] for m in members]
        use_parsed = len(set(reps)) == len(reps)
        for i, m in enumerate(members):
            rep = m["replicate"] if use_parsed else i + 1
            m["replicate"] = rep
            m["basename"] = "%s_rep%d" % (_safe(arm), rep)
    return samples


ARM_PROMPT = """You are grouping NGS samples into EXPERIMENTAL ARMS for a differential study.

Two samples belong to the SAME arm only if every experimental variable matches
(drug, dose, duration, hormone, genotype). Samples differing in ANY treatment
variable are DIFFERENT arms. Replicate and lane numbers do not define an arm.

A donor, patient, lane or replicate identifier is NOT itself an arm. But a title
often carries BOTH an identifier and a real condition, e.g. P1_Tumor is patient P1
with condition Tumor. Strip the identifier and keep the condition. Return an empty
string ONLY when nothing but identifiers remains.

Never put the assay type in the label. ATAC-seq and RNA-seq are measurements,
not conditions: the same biological condition profiled by both assays is ONE
arm, and its ATAC samples and RNA samples must carry the identical label.

Give each distinct arm one short stable label, reused verbatim across its members.
Do not invent a condition that is not in the title.

Return STRICT JSON only, no prose and no markdown fences:
{"results":[{"id":"<id verbatim>","arm":"<label or empty string>"}]}

SAMPLES:
%s"""


def refine_arms(samples: list[dict],
                endpoint: str = "http://127.0.0.1:11500",
                model: str = "batiai/qwen3.6-27b:q6",
                timeout: float = 600.0) -> list[dict]:
    """Re-derive arms with the LLM, keeping the rule-based answer as a fallback.

    The keyword rules know only a prostate-centric vocabulary and, worse, collapse
    conditions that share a drug (TGFb_2h and TGFb_24h both became TGFB), which
    silently merges two arms. The model is asked to group rather than to label,
    which is what keeps replicates together and time points apart.
    """
    targets = [s for s in samples if s["modality"] in ("ATAC", "RNA")]
    if not targets:
        return samples
    payload = [{"id": s["run"], "title": s["title"] or s["run"]} for s in targets]
    body = {
        "model": model,
        "messages": [{"role": "user",
                      "content": ARM_PROMPT % json.dumps(payload, indent=1)}],
        "stream": False, "think": False,
        "options": {"temperature": 0.0, "num_predict": 2048, "num_ctx": 16384},
    }
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt = json.loads(r.read())["message"]["content"]
        parsed = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])["results"]
    except Exception as exc:
        for s in targets:
            s["arm_source"] = "rule (llm unavailable: %s)" % str(exc)[:60]
        return samples
    by_id = {str(d.get("id")): (d.get("arm") or "").strip() for d in parsed}
    for s in targets:
        s["arm_rule"] = s["arm"]
        if s["run"] in by_id:
            s["arm"] = by_id[s["run"]]
            s["arm_confidence"] = "high" if s["arm"] else "low"
            s["arm_source"] = "llm"
        else:
            s["arm_source"] = "rule (not returned by llm)"
    return samples


def parse_accessions(value) -> list[str]:
    """Accept one accession, a list, or a string separated by commas/spaces/plus.

    List ELEMENTS are split again, not just taken whole: the chat agent and the
    argparse "nargs=+" both can hand a whole request through as one element
    (["<RNA study> <ATAC study>"]), which classify_accession would otherwise
    reject as one unrecognised accession.
    """
    raw = list(value) if isinstance(value, (list, tuple)) else [value]
    items: list[str] = []
    for v in raw:
        items.extend(re.split(r"[,;+\s]+", str(v or "")))
    out: list[str] = []
    for it in items:
        a = str(it).strip().upper()
        if a and a not in out:
            out.append(a)
    if not out:
        raise SRAError("no accession given")
    return out


def combo_name(studies: list[str]) -> str:
    """One stable directory and cache key for a set of studies."""
    uniq = sorted(set(studies))
    if len(uniq) == 1:
        return uniq[0]
    joined = "_".join(uniq)
    if len(joined) <= 64:
        return joined
    import hashlib
    return "%s_and%d_%s" % (uniq[0], len(uniq) - 1,
                            hashlib.sha1(joined.encode()).hexdigest()[:8])


def _acc_error(bad: list, accs: list, good: list, label: str) -> str:
    """One message for ALL the accessions that failed.

    Raising on the first failure and listing every OTHER accession as having
    "parsed fine" would also vouch for the ones never looked at (a valid run, a
    non-existent run and a garbage token: the garbage token reported as valid).
    Only accessions that really reached `label` are named as good.
    """
    head = "; ".join("%s (accession %d of %d)" % (msg, idx, len(accs))
                     for idx, msg in bad)
    if good:
        return "%s. %s: %s" % (head, label, ", ".join(good))
    return head


def lookup(accession, llm_assist: bool = True,
           endpoint: str = "http://127.0.0.1:11500",
           model: str = "batiai/qwen3.6-27b:q6",
           use_cache: bool = False,
           all_runs: bool = False) -> dict:
    """Read-only report for one or more accessions. Downloads nothing.

    Several accessions are merged into one report because ATAC and RNA for the
    same experiment are routinely deposited as separate studies. Merging before
    arm inference is what lets the model give the ATAC and the RNA side of one
    condition the same label, which a per-study lookup could never do.

    Every successful lookup is cached, and fetch() reads that cache so the
    download uses the exact arm assignment the user approved.
    """
    accs = parse_accessions(accession)
    # Classify EVERY accession before resolving any of them, and collect all the
    # failures instead of raising on the first one. Classification is a regex, so
    # this costs no network, and it is what keeps the error message from calling
    # an unexamined accession valid (2026-09-23 fix).
    bad: list = []
    parsed: list = []
    for i, a in enumerate(accs):
        try:
            classify_accession(a)
        except SRAError as exc:
            bad.append((i + 1, str(exc)))
        else:
            parsed.append(a)
    if bad:
        raise SRAError(_acc_error(bad, accs, parsed,
                                  "The accession(s) that are syntactically valid"))
    studies, kinds = [], []
    resolved: list = []
    for i, a in enumerate(accs):
        try:
            s, k = resolve_study(a)
        except SRAError as exc:
            bad.append((i + 1, str(exc)))
            continue
        resolved.append(a)
        kinds.append(k)
        if s not in studies:
            studies.append(s)
    if bad:
        raise SRAError(_acc_error(bad, accs, resolved,
                                  "The accession(s) that resolved"))
    key = combo_name(studies)
    # Run-level subsetting (2026-09-23). A run accession now selects THAT run;
    # a study / BioProject / GEO accession still means the whole study. The mixed
    # case (two study accessions plus an explicit run list) therefore downloads
    # exactly the listed runs, which is what the split ATAC+RNA deposit needs.
    wanted = [] if all_runs else [a for a, k in zip(accs, kinds) if k == "run"]
    cache_key = key if not wanted else "%s__runs-%s" % (key, _runs_key(wanted))
    if use_cache:
        cached = _cache_read(cache_key)
        if cached:
            return cached
    rows, seen = [], set()
    for s in studies:
        for r in fetch_runs(s):
            acc = r.get("run_accession", "")
            if acc and acc not in seen:
                seen.add(acc)
                rows.append(r)
    # Whole-study totals survive the subsetting so the caller can say
    # "8 of 16 runs, 20.1 GB of 60.2 GB" instead of only the subset size.
    study_samples = annotate(rows)
    n_total = len(study_samples)
    study_bytes = sum(x["bytes"] for x in study_samples
                      if x["modality"] in ("ATAC", "RNA"))
    missing_runs: list[str] = []
    if wanted:
        have = {str(r.get("run_accession", "")).upper() for r in rows}
        missing_runs = [a for a in wanted if a not in have]
        keep = set(wanted)
        rows = [r for r in rows
                if str(r.get("run_accession", "")).upper() in keep]
        if not rows:
            raise SRAError(
                "none of the requested runs (%s) belongs to stud%s %s"
                % (", ".join(wanted), "y" if len(studies) == 1 else "ies",
                   ", ".join(studies)))
    samples = annotate(rows)
    if llm_assist:
        samples = refine_arms(samples, endpoint=endpoint, model=model)
    samples, arm_merges = collapse_assay_arms(samples)
    samples = assign_names(samples)
    v = verdict(samples)
    for m in arm_merges:
        v["warnings"].append(
            "merged %s into one arm %r: they differ only by assay type"
            % (" and ".join(repr(x) for x in m["from"]), m["to"]))
    if missing_runs:
        v["warnings"].append(
            "%d requested run(s) are not in %s and were ignored: %s"
            % (len(missing_runs), ", ".join(studies), ", ".join(missing_runs)))
    selection = {
        "mode": "runs" if wanted else "study",
        "requested_runs": list(wanted),
        "selected_runs": [s["run"] for s in samples],
        "missing_runs": missing_runs,
        "n_selected": len(samples),
        "n_total": n_total,
        "selected_bytes": v["total_bytes"],
        "selected_human": v["total_human"],
        "study_total_bytes": study_bytes,
        "study_total_human": _human_bytes(study_bytes),
    }
    selection["summary"] = (
        ("%d of %d runs, %s of %s"
         % (selection["n_selected"], n_total,
            selection["selected_human"], selection["study_total_human"]))
        if wanted else
        ("all %d runs of %s, %s"
         % (n_total, ", ".join(studies), selection["study_total_human"])))
    # A bare run resolves only to its own sub-study. Point at the fix rather than
    # letting the user conclude the experiment is unusable.
    if len(studies) == 1 and kinds[0] == "run" and (v["n_atac"] == 0 or v["n_rna"] == 0):
        v["reasons"].append(
            "you supplied run accession(s) only (%s), which select %d run(s) of "
            "sub-study %s; ATAC and RNA are often deposited separately, so add the "
            "other accession (sra-lookup RNA_ACC ATAC_ACC), use the GEO series, or "
            "pass all_runs/--all-runs to take the whole study (%d runs)"
            % (", ".join(accs), len(samples), studies[0], n_total))
    rep = {
        "input": accs,
        "input_kind": kinds,
        "study": key,
        "studies": studies,
        "cache_key": cache_key,
        "dest": str(DATA_ROOT / key),
        "n_runs": len(samples),
        "selection": selection,
        "selected_runs": selection["selected_runs"],
        "n_selected": selection["n_selected"],
        "n_total": n_total,
        "whole_study": not bool(wanted),
        "llm_assist": bool(llm_assist),
        "from_cache": False,
        "species": v.get("species", []),
        "organism": v.get("organism", ""),
        "samples": samples,
        "verdict": v,
    }
    _cache_write(rep)
    return rep


NAMES_FILE = "sra_names.json"


def read_name_registry(dest) -> dict:
    """run -> {basename, modality, arm} for everything ever downloaded here.

    Arm labels are inferred per LOOKUP, so the same run is "treated" in an 8-run
    subset and "Sorted-treated" in the whole study, while `dest` is the same
    directory for both. Without this registry the second fetch does not
    recognise the files the first one wrote: it downloads all of them again
    under the second set of names, doubling the bytes on disk and making detect
    see one biological sample as two conditions (2026-09-23 fix).
    """
    d = Path(dest)
    reg: dict = {}
    p = d / NAMES_FILE
    if p.exists():
        try:
            raw = json.loads(p.read_text())
            for run, rec in (raw.get("runs") or {}).items():
                if isinstance(rec, dict) and rec.get("basename"):
                    reg[str(run).upper()] = dict(rec)
        except (OSError, ValueError):
            reg = {}
    if reg:
        return reg
    # Directories written before the registry existed still carry the manifest.
    m = d / "sra_manifest.json"
    if m.exists():
        try:
            for s in json.loads(m.read_text()).get("samples") or []:
                if s.get("run") and s.get("basename"):
                    reg[str(s["run"]).upper()] = {
                        "basename": s["basename"],
                        "modality": s.get("modality", ""),
                        "arm": s.get("arm", ""),
                        "species": s.get("species", "")}
        except (OSError, ValueError):
            pass
    return reg


def write_name_registry(dest, samples, previous=None) -> dict:
    """Merge this selection's names into the directory's registry.

    The manifest is overwritten by every fetch and only ever describes the LAST
    selection; the registry is merged, so it still knows the names of the runs
    an earlier, wider fetch put on disk.
    """
    reg = dict(previous if previous is not None else read_name_registry(dest))
    for s in samples:
        if not s.get("run") or not s.get("basename"):
            continue
        # Record the arm the name was MINTED under, not the label this lookup
        # happens to use, so the registry explains the filenames it holds.
        reg[str(s["run"]).upper()] = {"basename": s["basename"],
                                      "modality": s.get("modality", ""),
                                      "arm": s.get("arm_previous") or s.get("arm", ""),
                                      # The organism of the data on disk. `mrag
                                      # detect` reads it back from here, so a
                                      # downloaded study no longer forgets its
                                      # species between fetch and run.
                                      "species": s.get("species", "")}
    try:
        _atomic_write_json(Path(dest) / NAMES_FILE,
                           {"updated": time.time(), "runs": reg})
    except OSError:
        pass                      # a lost registry is recoverable; a failed fetch is not
    return reg


def reconcile_names(samples: list, registry: dict) -> dict:
    """Make this selection's basenames agree with the directory's existing ones.

    Returns {"reused": [...], "deduped": [...]}; both lists are reported to the
    user, because a filename that is not the one this lookup computed is exactly
    the kind of silent difference that should never be silent.
    """
    reused: list = []
    deduped: list = []
    taken: dict = {}
    for s in samples:
        prev = (registry or {}).get(str(s.get("run", "")).upper())
        if not prev or not prev.get("basename"):
            continue
        if prev["basename"] != s.get("basename"):
            reused.append({"run": s.get("run"), "from": s.get("basename"),
                           "to": prev["basename"], "arm": s.get("arm"),
                           "previous_arm": prev.get("arm", "")})
            s["arm_previous"] = prev.get("arm", "")
            s["basename"] = prev["basename"]
        taken[(s.get("modality"), s["basename"])] = s.get("run")
    for s in samples:
        key = (s.get("modality"), s.get("basename", ""))
        if taken.get(key) in (None, s.get("run")):
            taken[key] = s.get("run")
            continue
        base, i = s["basename"], 2
        while (s.get("modality"), "%s_%d" % (base, i)) in taken:
            i += 1
        s["basename"] = "%s_%d" % (base, i)
        deduped.append({"run": s.get("run"), "from": base, "to": s["basename"]})
        taken[(s.get("modality"), s["basename"])] = s.get("run")
    return {"reused": reused, "deduped": deduped}


def _targets(s: dict, dest: Path) -> list[tuple[str, Path]]:
    """(url, path) pairs for one run, named for the detector."""
    mod_dir = dest / s["modality"]
    urls = s["fastq_ftp"]
    # ENA sometimes lists an unsplit run file alongside _1/_2; prefer the split pair.
    split = [u for u in urls if re.search(r"_[12]\.fastq\.gz$", u)]
    if s["layout"] == "PAIRED" and len(split) >= 2:
        urls = sorted(split)[:2]
    elif len(urls) > 1 and s["layout"] == "SINGLE":
        urls = urls[:1]
    out = []
    for i, u in enumerate(urls):
        suffix = ("_R%d" % (i + 1)) if len(urls) > 1 else ""
        out.append(("https://" + u.lstrip("/"),
                    mod_dir / ("%s%s.fastq.gz" % (s["basename"], suffix))))
    return out


# ------------------------------------------------------------------
# Launch bookkeeping and the honest download state machine (2026-09-23)
# ------------------------------------------------------------------
# A fetch used to be invisible until it had written its first sra_progress.json,
# so "the process died two seconds in" and "the download is still starting" were
# the same answer - and the chat agent repeated the second one forever while
# nothing was downloading (two chat conversations, 2026-09-18).
# The launch record is written the instant a fetch starts and carries the child
# pid, which is what lets fetch_state() tell a live download from a dead one, and
# both of those from a fetch that was never started at all.

PROGRESS_FILE = "sra_progress.json"
LAUNCH_FILE = "sra_fetch.launch.json"
LOG_FILE = "sra_fetch.log"
LAUNCH_TOKEN = "sra-fetch"     # must still appear in the child's /proc cmdline
STALE_AFTER = 900.0            # s of silence before a pid-less progress file
                               # counts as dead rather than merely quiet


def _atomic_write_json(path, payload: dict) -> None:
    """Write JSON so a concurrent reader never sees a half-written file.

    The progress file is rewritten every couple of seconds while a download
    runs, so a plain write_text would regularly hand `sra-status` a truncated
    document and turn a healthy download into "unreadable progress file".
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(str(tmp), str(p))


def _write_progress(dest: Path, payload: dict) -> None:
    out = dict(payload)
    out.setdefault("pid", os.getpid())
    out.setdefault("pgid", _pgid())
    out["started"] = out.get("started") or time.time()
    out["updated"] = time.time()
    _atomic_write_json(Path(dest) / PROGRESS_FILE, out)


def _pgid(pid: int = 0) -> int:
    """Process-group id of `pid` (0 = this process), or 0 when unknowable."""
    try:
        return os.getpgid(int(pid))
    except (OSError, TypeError, ValueError):
        return 0


def _pct(done, total) -> float:
    """Clamped percentage; 0.0 when the total is unknown (never a ZeroDivision)."""
    try:
        done = float(done or 0)
        total = float(total or 0)
    except (TypeError, ValueError):
        return 0.0
    if total <= 0:
        return 0.0
    return round(max(0.0, min(100.0, 100.0 * done / total)), 2)


def _fmt_eta(seconds) -> str:
    if seconds is None:
        return "unknown"
    try:
        s = int(max(0, seconds))
    except (TypeError, ValueError):
        return "unknown"
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return "%d:%02d:%02d" % (h, m, s) if h else "%d:%02d" % (m, s)


def set_job_bytes(job: dict, nbytes) -> dict:
    """Record live bytes on one job and refresh its percent."""
    try:
        total = int(job.get("bytes_total") or 0)
    except (TypeError, ValueError):
        total = 0
    try:
        n = max(0, int(nbytes or 0))
    except (TypeError, ValueError):
        n = 0
    if total and n > total:
        n = total
    job["bytes_done"] = n
    job["pct"] = _pct(n, total) if total else (
        100.0 if job.get("state") == "done" else 0.0)
    return job


def summarize_jobs(jobs, started=None, now=None, bytes_at_start=0) -> dict:
    """Overall bytes / percent / speed / ETA over a job list.

    Pure apart from time.time(), so the progress maths is unit-testable without
    a download. Speed ignores the bytes that were already on disk when this
    process started, otherwise a resumed fetch reports a fictitious rate.
    """
    jobs = list(jobs or [])
    bytes_total = sum(int(j.get("bytes_total") or 0) for j in jobs)
    bytes_done = sum(int(j.get("bytes_done") or 0) for j in jobs)
    files_total = len(jobs)
    files_done = sum(1 for j in jobs if j.get("state") == "done")
    files_failed = sum(1 for j in jobs if j.get("state") == "failed")
    now = time.time() if now is None else float(now)
    try:
        elapsed = max(0.0, now - float(started)) if started else 0.0
    except (TypeError, ValueError):
        elapsed = 0.0
    moved = max(0, bytes_done - int(bytes_at_start or 0))
    speed = (moved / elapsed) if (elapsed > 0 and moved > 0) else 0.0
    remaining = max(0, bytes_total - bytes_done)
    eta = int(remaining / speed) if (speed > 0 and bytes_total > 0) else None
    cur = ""
    for j in jobs:
        if j.get("state") == "downloading":
            cur = j.get("file") or ""
            break
    return {
        "bytes_done": bytes_done,
        "bytes_total": bytes_total,
        "pct": _pct(bytes_done, bytes_total),
        "files_done": files_done,
        "files_total": files_total,
        "files_failed": files_failed,
        "current": cur,
        "speed_bps": int(speed),
        "speed_human": _human_bytes(int(speed)) + "/s",
        "eta_s": eta,
        "eta_human": _fmt_eta(eta),
        "done_human": _human_bytes(bytes_done),
        "total_human": _human_bytes(bytes_total),
        "elapsed_s": int(elapsed),
    }


class _ProgressWriter:
    """Throttled, atomic progress writer shared by the fetch loop and the ticker."""

    # Shorter than _curl's tick interval on purpose: at exactly 2 s the two
    # clocks drift into each other and every other tick is swallowed, which
    # halves the observable update rate for no saving worth having.
    def __init__(self, dest, prog: dict, min_interval: float = 1.0,
                 bytes_at_start: int = 0):
        self.dest = Path(dest)
        self.prog = prog
        self.min_interval = float(min_interval)
        self.bytes_at_start = int(bytes_at_start or 0)
        self._last = 0.0

    def refresh(self, force: bool = False) -> dict:
        now = time.time()
        if not force and (now - self._last) < self.min_interval:
            return self.prog
        self._last = now
        ov = summarize_jobs(self.prog.get("jobs") or [],
                            started=self.prog.get("started"), now=now,
                            bytes_at_start=self.bytes_at_start)
        self.prog["overall"] = ov
        self.prog["done"] = ov["files_done"]
        # `failed` is a FILE count, exactly like `done`, and it is refreshed
        # here so a running fetch stops reporting "0 failed so far" while files
        # are already failing (2026-09-23 fix).
        self.prog["failed"] = ov["files_failed"]
        self.prog["bytes_done"] = ov["bytes_done"]
        self.prog["pct"] = ov["pct"]
        if ov["current"]:
            self.prog["current"] = ov["current"]
        _write_progress(self.dest, self.prog)
        return self.prog


def write_launch(dest, pid: int, argv=None, accessions=None, study: str = "") -> Path:
    """Record that a fetch process was started for this directory."""
    d = Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    p = d / LAUNCH_FILE
    _atomic_write_json(p, {
        "pid": int(pid),
        # The process group is what `sra-cancel` kills, so it is recorded at
        # launch: by the time someone cancels, the curl child may be the only
        # thing still running and the python parent already reaped.
        "pgid": _pgid(int(pid)),
        "started": time.time(),
        "started_human": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": [str(a) for a in (argv or [])],
        "accessions": list(accessions or []),
        "study": study,
        "dest": str(d),
        "log": str(d / LOG_FILE),
    })
    return p


def read_launch(dest) -> Optional[dict]:
    try:
        return json.loads((Path(dest) / LAUNCH_FILE).read_text())
    except (OSError, ValueError):
        return None


def mark_failed(dest, reason: str, **extra) -> None:
    """Persist a failure so the next poll reports it instead of guessing."""
    try:
        prog: dict = {}
        p = Path(dest) / PROGRESS_FILE
        if p.exists():
            try:
                prog = json.loads(p.read_text())
            except ValueError:
                prog = {}
        prog.update(extra)
        prog["state"] = "failed"
        prog["dest"] = str(dest)
        prog["reasons"] = list(prog.get("reasons") or []) + [str(reason)]
        _write_progress(Path(dest), prog)
    except OSError:
        pass


def _proc_cmdline(pid: int) -> Optional[str]:
    try:
        raw = Path("/proc/%d/cmdline" % pid).read_bytes()
    except OSError:
        return None
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()


def pid_alive(pid, token: str = LAUNCH_TOKEN) -> bool:
    """True only when the pid runs AND still looks like the fetch we launched.

    The cmdline check is what stops a recycled pid from being reported as a live
    download hours after the real one died; a zombie does not count as alive.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    cl = _proc_cmdline(pid)
    if cl is None:
        return False
    try:
        stat = Path("/proc/%d/stat" % pid).read_text()
        if stat.rsplit(")", 1)[-1].split()[0] == "Z":
            return False
    except (OSError, IndexError):
        pass
    return (token in cl) if token else True


def log_tail(path, n_lines: int = 25, n_chars: int = 2000) -> str:
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return ""
    lines = [l for l in text.splitlines() if l.strip()]
    return "\n".join(lines[-n_lines:])[-n_chars:]


def _exit_reason(tail: str) -> str:
    if not tail:
        return "the process wrote nothing to %s before it exited" % LOG_FILE
    for marker in ("Traceback (most recent call last)", "SRAError", "MemoryError",
                   "No space left on device", "Killed", "Permission denied",
                   "command not found"):
        if marker in tail:
            return "%s ends with %s" % (LOG_FILE, marker)
    return "see the %s tail" % LOG_FILE


def _files_phrase(prog: dict) -> str:
    return "%s of %s files" % (prog.get("done", 0), prog.get("total_files", "?"))


def _pct_phrase(prog: dict) -> str:
    """", 31.4% of 20.1 GB (6.3 GB)" - empty when no byte totals are known."""
    ov = prog.get("overall") or {}
    if not ov.get("bytes_total"):
        return ""
    return (", %.1f%% of %s (%s transferred)"
            % (float(ov.get("pct") or 0.0), ov.get("total_human") or "?",
               ov.get("done_human") or "?"))


def _current_phrase(prog: dict) -> str:
    """The file being transferred right now, with its own percent."""
    for j in (prog.get("jobs") or []):
        if j.get("state") == "downloading":
            tot = int(j.get("bytes_total") or 0)
            if tot:
                return (", current file %s at %.1f%% of %s"
                        % (j.get("file"), float(j.get("pct") or 0.0),
                           _human_bytes(tot)))
            return ", current file %s (size unknown)" % j.get("file")
    return (", current file %s" % prog["current"]) if prog.get("current") else ""


def _rate_phrase(prog: dict) -> str:
    ov = prog.get("overall") or {}
    if not ov.get("speed_bps"):
        return ""
    return (", %s, ETA %s" % (ov.get("speed_human") or "?",
                              ov.get("eta_human") or "unknown"))


def _notes_phrase(prog: dict) -> str:
    """Directory-level notes (reused filenames) appended to a status message."""
    ns = [str(x) for x in ((prog or {}).get("notes") or []) if str(x).strip()]
    return (" " + " ".join(ns)) if ns else ""


def _selection_phrase(prog: dict) -> str:
    sel = prog.get("selection") or {}
    if sel.get("mode") == "runs" and sel.get("n_total"):
        return " (%s of %s runs selected)" % (sel.get("n_selected"),
                                              sel.get("n_total"))
    return ""


def fetch_state(dest) -> dict:
    """The truthful state of one download directory.

    Distinguishes, and never blurs:
      not_started  nothing was ever launched here (the dir may not even exist)
      starting     a fetch process is alive but has not written progress yet
      running      progress is being written by a live process
      partial      finished, some files failed
      cancelled    stopped on purpose; the partial file was kept and can resume
      done         finished cleanly
      refused      the usability check rejected it; nothing was transferred
      failed       the process is gone, or died, or the progress file is unreadable

    Every result carries a top-level "ok" and a "user_message" that can be shown
    verbatim, so no caller can present a failure as progress.
    """
    raw = str(dest or "").strip()
    if not raw:
        return {"ok": False, "state": "not_started", "dest": "",
                "user_message": "No download directory was given and no fetch has "
                                "been started in this session, so there is nothing "
                                "to report. Nothing is downloading."}
    d = Path(raw)
    launch = read_launch(d)
    pp = d / PROGRESS_FILE
    prog: Optional[dict] = None
    if pp.exists():
        try:
            prog = json.loads(pp.read_text())
        except (OSError, ValueError) as exc:
            prog = {"state": "unreadable",
                    "_error": "%s: %s" % (type(exc).__name__, exc)}
    tail = log_tail(d / LOG_FILE)
    pid = (launch or {}).get("pid") or (prog or {}).get("pid")
    alive = pid_alive(pid)
    started = float((launch or {}).get("started") or (prog or {}).get("started") or 0)

    out: dict = {"dest": str(d), "dest_exists": d.exists(), "launched": bool(launch)}
    if launch:
        out["accessions"] = launch.get("accessions", [])
        out["started_human"] = launch.get("started_human", "")
        if launch.get("study"):
            out["study"] = launch["study"]
    if pid:
        out["pid"] = pid
        out["pid_alive"] = alive
    if started:
        out["elapsed_s"] = int(time.time() - started)
    if tail:
        out["log"] = str(d / LOG_FILE)
        out["log_tail"] = tail

    # 1. Nothing was ever started here. This is NOT "maybe starting".
    if launch is None and prog is None:
        where = ("the directory does not exist" if not d.exists()
                 else "the directory exists but holds no fetch record")
        return {**out, "ok": False, "state": "not_started",
                "user_message": "No download has been started for %s (%s). Nothing "
                                "is downloading. Run sra_lookup, get the user's "
                                "approval, then call sra_fetch with confirmed=true."
                                % (d, where)}

    state = (prog or {}).get("state", "")
    if prog:
        for k in ("study", "total_files", "done", "failed", "total_bytes",
                  "current", "from_cache", "overall", "pct", "bytes_done",
                  "selection", "selected_runs", "n_selected", "n_total",
                  "resumed", "skipped_unverified", "notes", "renamed",
                  "fallback_errors"):
            if k in prog:
                out[k] = prog[k]
        out["errors"] = (prog.get("errors") or [])[:5]
        # Per-file rows are what a caller renders a progress bar from. A huge
        # study would bury the message in them, so past 40 files only the live
        # one is passed through.
        rows = list(prog.get("jobs") or [])
        if rows:
            out["jobs"] = rows if len(rows) <= 40 else [
                j for j in rows if j.get("state") == "downloading"]
            out["n_jobs"] = len(rows)

    # 2. Finished, one way or the other.
    if state == "done":
        return {**out, "ok": True, "state": "done",
                "next_step": "Download complete. Call detect with fastq_dir=%s" % d,
                "user_message": "Download finished%s: %s under %s%s.%s"
                                % (_selection_phrase(prog), _files_phrase(prog), d,
                                   _pct_phrase(prog), _notes_phrase(prog))}
    if state == "partial":
        return {**out, "ok": True, "state": "partial",
                "next_step": "Some files failed. Show the errors and ask whether to "
                             "retry sra_fetch (it re-checks every file against the "
                             "size ENA reports and resumes the truncated ones) or "
                             "run detect anyway.",
                "user_message": "Download finished WITH FAILURES: %s transferred, %d "
                                "file(s) failed. The failures are listed in errors.%s"
                                % (_files_phrase(prog), int(prog.get("failed") or 0),
                                   _notes_phrase(prog))}
    if state == "cancelled":
        return {**out, "ok": False, "state": "cancelled", "resumable": True,
                "next_step": "Nothing is downloading. Ask whether to resume "
                             "(mrag sra-fetch --resume --dest %s) or to leave it." % d,
                "user_message": "The download was CANCELLED and NOTHING is running "
                                "now: %s complete%s. The partly downloaded file was "
                                "kept, so resuming continues from where it stopped."
                                % (_files_phrase(prog), _pct_phrase(prog))}
    if state == "refused":
        rs = list(prog.get("reasons") or [])
        out["reasons"] = rs
        out["missing"] = prog.get("missing", [])
        return {**out, "ok": False, "state": "refused",
                "user_message": "The download was REFUSED before anything was "
                                "transferred: %s"
                                % ("; ".join(rs) or "no reason recorded")}
    if state in ("failed", "unreadable"):
        rs = list(prog.get("reasons") or [])
        if prog.get("_error"):
            rs.append(str(prog["_error"]))
        out["reasons"] = rs
        return {**out, "ok": False, "state": "failed",
                "user_message": "The download FAILED and nothing is running. %s%s"
                                % ("; ".join(rs) or "No reason was recorded.",
                                   ("\nLog tail:\n" + tail) if tail else "")}

    # 3. In flight, or claiming to be.
    if state in ("running", "resolving"):
        quiet = time.time() - float(prog.get("updated") or 0)
        if alive:
            if state == "resolving":
                return {**out, "ok": True, "state": "starting",
                        "user_message": "Reading the study metadata, no bytes "
                                        "transferred yet (pid %s, %ds elapsed)."
                                        % (pid, out.get("elapsed_s", 0))}
            return {**out, "ok": True, "state": "running",
                    "user_message": "Downloading%s: %s done%s%s%s, %d failed so far.%s"
                                    % (_selection_phrase(prog),
                                       _files_phrase(prog), _pct_phrase(prog),
                                       _current_phrase(prog), _rate_phrase(prog),
                                       int(prog.get("failed") or 0),
                                       _notes_phrase(prog))}
        if not pid and quiet < STALE_AFTER:
            # Progress file from before the launch record existed: trust it while fresh.
            return {**out, "ok": True, "state": "running", "pid_unknown": True,
                    "user_message": "Downloading: %s done (no pid recorded; this "
                                    "fetch predates the launch record)."
                                    % _files_phrase(prog)}
        return {**out, "ok": False, "state": "failed",
                "user_message": "The download process is GONE but the progress file "
                                "still says %r: %s were done, so the rest was never "
                                "transferred. %s%s"
                                % (state, _files_phrase(prog), _exit_reason(tail),
                                   ("\nLog tail:\n" + tail) if tail else "")}

    # 4. A launch record but no progress file yet.
    if launch is not None and prog is None:
        if alive:
            return {**out, "ok": True, "state": "starting",
                    "user_message": "The download process (pid %s) started %ds ago "
                                    "and has not written its first progress file "
                                    "yet; it is still resolving the study metadata."
                                    % (pid, out.get("elapsed_s", 0))}
        return {**out, "ok": False, "state": "failed",
                "user_message": "The download was launched (pid %s) but that process "
                                "is gone and it never wrote a progress file, so "
                                "NOTHING was downloaded. %s%s"
                                % (pid, _exit_reason(tail),
                                   ("\nLog tail:\n" + tail) if tail else "")}

    # 5. Anything else is an unrecognised progress file; say so rather than guess.
    return {**out, "ok": False, "state": "failed",
            "user_message": "Unrecognised download state %r in %s; treat it as "
                            "failed.%s"
                            % (state, pp, ("\nLog tail:\n" + tail) if tail else "")}


def _curl(url: str, out: Path, timeout: int = 7200, resume: bool = True,
          tick=None, interval: float = 2.0) -> tuple[bool, str]:
    """Run curl and report progress WHILE it transfers.

    The transfer is polled instead of waited on, so the caller can publish a
    per-file percent every couple of seconds. Before 2026-09-23 the progress
    file only moved between files, which made a 15 GB fastq look frozen for an
    hour. stderr goes to a temp file rather than a pipe: with a pipe a chatty
    curl can fill the buffer and deadlock against our own wait().
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-fsSL", "--retry", "5", "--retry-delay", "10",
           "--connect-timeout", "60"]
    # Extra curl flags for the operator, e.g. MRAG_CURL_ARGS="--limit-rate 20M" to
    # keep a 60 GB fetch from saturating a shared link. Also what lets the resume
    # test throttle a real transfer enough to be cancelled mid-file.
    extra = os.environ.get("MRAG_CURL_ARGS", "").strip()
    if extra:
        import shlex
        cmd += shlex.split(extra)
    if resume:
        cmd += ["-C", "-"]
    cmd += ["-o", str(out), url]

    def _seen():
        try:
            return out.stat().st_size
        except OSError:
            return 0

    def _ping():
        if tick is None:
            return
        try:
            tick(_seen())
        except Exception:          # noqa: BLE001 - progress must never kill a download
            pass

    errf = tempfile.TemporaryFile(mode="w+")
    try:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=errf)
        except OSError as exc:
            return False, ("could not run curl: %s" % exc)[:300]
        deadline = time.time() + timeout
        while True:
            try:
                proc.wait(timeout=interval)
                break
            except subprocess.TimeoutExpired:
                _ping()
                if time.time() > deadline:
                    proc.kill()
                    try:
                        proc.wait(timeout=10)
                    except Exception:      # noqa: BLE001
                        pass
                    _ping()
                    return False, "timed out after %ds" % timeout
        _ping()
        try:
            errf.seek(0)
            err = (errf.read() or "").strip()[:300]
        except OSError:
            err = ""
    finally:
        errf.close()
    return (proc.returncode == 0), err


def _download_one(url: str, target: Path, want: int, tick=None,
                  timeout: int = 7200) -> tuple[bool, str, int]:
    """Transfer one file, continuing a truncated one, and VERIFY the final size.

    curl is asked to resume only when a partial file is actually there. A server
    that ignores the byte range answers 200 and curl appends the whole body to
    the partial file, which leaves a plausible-looking but corrupt fastq.gz; the
    size check below catches both that and a short transfer, and re-downloads the
    file from scratch. Returns (ok, error, bytes_on_disk).
    """
    target = Path(target)
    have = target.stat().st_size if target.exists() else 0
    want = int(want or 0)
    attempts = [True] if (have > 0 and (not want or have < want)) else []
    attempts.append(False)
    err = ""
    final = have
    for i, resume in enumerate(attempts):
        if not resume and i > 0:
            # The resumed attempt failed or produced the wrong size: the partial
            # file is worthless now, so start clean rather than keep appending.
            try:
                if target.exists():
                    target.unlink()
            except OSError:
                pass
            if tick is not None:
                try:
                    tick(0)
                except Exception:          # noqa: BLE001
                    pass
        ok, e = _curl(url, target, timeout=timeout, resume=resume, tick=tick)
        final = target.stat().st_size if target.exists() else 0
        if ok and (not want or final == want):
            return True, "", final
        err = (e or "curl failed") if not ok else (
            "transferred %d bytes but ENA reports %d" % (final, want))
        if resume:
            err = "resume failed (%s), retried from scratch" % err
            continue
        return False, err, final
    return False, err, final


# ------------------------------------------------------------------
# Stopping a download for real, and picking it up again (2026-09-23)
# ------------------------------------------------------------------

def _descendants(pid: int) -> list[int]:
    """Every descendant of pid, read straight out of /proc.

    Used only when the fetch is NOT its own process-group leader, i.e. when we
    cannot safely kill a group because the group also holds the shell that
    started it.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return []
    kids: dict[int, list[int]] = {}
    try:
        entries = [int(p.name) for p in Path("/proc").iterdir() if p.name.isdigit()]
    except OSError:
        return []
    for p in entries:
        try:
            txt = Path("/proc/%d/stat" % p).read_text()
            ppid = int(txt.rsplit(")", 1)[-1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        kids.setdefault(ppid, []).append(p)
    out: list[int] = []
    stack = list(kids.get(pid, []))
    while stack:
        c = stack.pop()
        if c == pid or c in out:
            continue
        out.append(c)
        stack.extend(kids.get(c, []))
    return sorted(out)


def _wait_gone(pid: int, timeout: float, token: str = LAUNCH_TOKEN) -> bool:
    deadline = time.time() + max(0.0, float(timeout))
    while True:
        if not pid_alive(pid, token):
            return True
        if time.time() >= deadline:
            return not pid_alive(pid, token)
        time.sleep(0.2)


def live_fetch_pid(dest) -> int:
    """pid of a fetch still running for this directory, else 0.

    Never reports our own pid, so the fetch process does not refuse to start
    because it already wrote its own launch record.
    """
    launch = read_launch(dest) or {}
    prog: dict = {}
    try:
        prog = json.loads((Path(dest) / PROGRESS_FILE).read_text())
    except (OSError, ValueError):
        prog = {}
    for cand in (launch.get("pid"), prog.get("pid")):
        try:
            cand = int(cand or 0)
        except (TypeError, ValueError):
            continue
        if cand and cand != os.getpid() and pid_alive(cand):
            return cand
    return 0


def cancel_fetch(dest, grace: float = 6.0) -> dict:
    """Stop the fetch that owns <dest>, keep the partial file, mark it cancelled.

    Only a pid whose /proc cmdline still looks like our own sra-fetch is ever
    signalled, and the process GROUP is killed only when the fetch leads that
    group (start_new_session, which is how the web UI launches it). Otherwise
    the pid and its own descendants are signalled one by one, so a fetch started
    from an interactive shell can never take the shell down with it.
    """
    raw = str(dest or "").strip()
    if not raw:
        return {"ok": False, "state": "not_started", "dest": "", "killed": False,
                "user_message": "No download directory was given, so there is "
                                "nothing to cancel. Nothing was stopped."}
    d = Path(raw)
    launch = read_launch(d)
    prog: dict = {}
    pp = d / PROGRESS_FILE
    if pp.exists():
        try:
            prog = json.loads(pp.read_text())
        except (OSError, ValueError):
            prog = {}
    if launch is None and not prog:
        return {"ok": False, "state": "not_started", "dest": str(d), "killed": False,
                "user_message": "No fetch was ever started for %s, so there is "
                                "nothing to cancel. Nothing was stopped." % d}
    try:
        pid = int((launch or {}).get("pid") or prog.get("pid") or 0)
    except (TypeError, ValueError):
        pid = 0
    try:
        rec_pgid = int((launch or {}).get("pgid") or prog.get("pgid") or 0)
    except (TypeError, ValueError):
        rec_pgid = 0
    state = str(prog.get("state") or "")
    if state in ("done", "partial", "refused", "cancelled"):
        return {"ok": True, "state": state, "dest": str(d), "pid": pid,
                "killed": False,
                "user_message": "Nothing to cancel: this download is already %r."
                                % state}

    notes: list[str] = []
    signals: list[str] = []
    killed = False
    if pid and pid_alive(pid):
        cur_pgid = _pgid(pid)
        own_pgid = _pgid(0)
        use_group = bool(cur_pgid) and cur_pgid == pid and cur_pgid != own_pgid
        if rec_pgid and cur_pgid and rec_pgid != cur_pgid:
            notes.append("process group changed since launch (%d -> %d)"
                         % (rec_pgid, cur_pgid))
        victims = [] if use_group else ([pid] + _descendants(pid))
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                if use_group:
                    os.killpg(cur_pgid, sig)
                    signals.append("%s -> process group %d" % (sig.name, cur_pgid))
                else:
                    for vp in victims:
                        try:
                            os.kill(vp, sig)
                        except OSError:
                            pass
                    signals.append("%s -> pids %s"
                                   % (sig.name,
                                      ",".join(str(x) for x in victims) or "none"))
            except OSError as exc:
                notes.append("%s failed: %s" % (sig.name, exc))
            if _wait_gone(pid, grace if sig == signal.SIGTERM else 3.0):
                killed = True
                break
        if not killed and pid_alive(pid):
            return {"ok": False, "state": "running", "dest": str(d), "pid": pid,
                    "killed": False, "signals": signals, "notes": notes,
                    "user_message": "Could NOT stop the download: pid %d is still "
                                    "alive after SIGTERM and SIGKILL, so it is still "
                                    "running and still writing files.%s"
                                    % (pid, (" " + "; ".join(notes)) if notes else "")}
    elif pid:
        cl = _proc_cmdline(pid)
        if cl is None:
            notes.append("pid %d had already exited; nothing was signalled" % pid)
        else:
            notes.append("pid %d is now a different process (%s) and was NOT "
                         "signalled" % (pid, cl[:80]))
    else:
        notes.append("no pid was recorded for this download; nothing was signalled")

    jobs = list(prog.get("jobs") or [])
    for j in jobs:
        if j.get("state") == "downloading":
            try:
                have = Path(str(j.get("path") or "")).stat().st_size
            except OSError:
                have = int(j.get("bytes_done") or 0)
            set_job_bytes(j, have)
            j["state"] = "cancelled"
    prog["jobs"] = jobs
    if jobs:
        ov = summarize_jobs(jobs, started=prog.get("started"), now=time.time())
        ov["current"] = ""
        prog["overall"] = ov
        prog["done"] = ov["files_done"]
        prog["bytes_done"] = ov["bytes_done"]
        prog["pct"] = ov["pct"]
    prog["state"] = "cancelled"
    prog["dest"] = str(d)
    prog["current"] = ""
    # Keep the DEAD fetch's group in the record. _write_progress would otherwise
    # default it to the canceller's own group, which names a live process that
    # has nothing to do with this download.
    prog["pgid"] = rec_pgid or int(prog.get("pgid") or 0)
    prog["cancelled_at"] = time.time()
    prog["cancel_notes"] = notes
    prog["cancel_signals"] = signals
    prog["reasons"] = list(prog.get("reasons") or []) + [
        "cancelled by the user at %s" % time.strftime("%Y-%m-%d %H:%M:%S")]
    _write_progress(d, prog)

    pct = float(prog.get("pct") or 0.0)
    ov = prog.get("overall") or {}
    how = ("stopped pid %d (%s)" % (pid, "; ".join(signals))) if killed else (
        "nothing was running: %s" % ("; ".join(notes) or "no process found"))
    return {"ok": True, "state": "cancelled", "dest": str(d), "pid": pid,
            "killed": killed, "signals": signals, "notes": notes,
            "resumable": True,
            "done": prog.get("done", 0),
            "total_files": prog.get("total_files", 0),
            "pct": pct,
            "overall": ov,
            "next_step": "Resume with: mrag sra-fetch --resume --dest %s" % d,
            "user_message": "Download CANCELLED (%s). %s of %s files complete, "
                            "%.1f%% of %s transferred. The partly downloaded file "
                            "was KEPT, so resuming continues from where it stopped: "
                            "mrag sra-fetch --resume --dest %s"
                            % (how, prog.get("done", 0),
                               prog.get("total_files", "?"), pct,
                               ov.get("total_human") or "the study", d)}


def resume_fetch(dest, force: bool = False, llm_assist: bool = True,
                 use_cache: bool = True, accession=None,
                 all_runs: bool = False) -> dict:
    """Continue a cancelled or partial download in <dest>.

    The accessions come from the launch record (or the manifest) so the caller
    does not have to remember which subset was approved.
    """
    raw = str(dest or "").strip()
    if not raw:
        raise SRAError("--resume needs a destination directory (--dest)")
    d = Path(raw)
    launch = read_launch(d) or {}
    prog: dict = {}
    try:
        prog = json.loads((d / PROGRESS_FILE).read_text())
    except (OSError, ValueError):
        prog = {}
    accs = list(accession or [])
    if not accs:
        accs = list(launch.get("accessions") or prog.get("accessions") or [])
    if not accs:
        try:
            accs = list(json.loads(
                (d / "sra_manifest.json").read_text()).get("input") or [])
        except (OSError, ValueError):
            accs = []
    if not accs:
        raise SRAError(
            "cannot resume %s: no accession is recorded there (no %s, no %s and no "
            "sra_manifest.json). Re-run sra-fetch with the accessions instead."
            % (d, LAUNCH_FILE, PROGRESS_FILE))
    live = live_fetch_pid(d)
    if live:
        return {"state": "already_running", "dest": str(d), "pid": live,
                "user_message": "NOTHING was resumed: a download (pid %d) is already "
                                "running in %s. Poll it with sra-status, or stop it "
                                "with sra-cancel --dest %s." % (live, d, d)}
    return fetch(accs, dest=str(d), force=force, llm_assist=llm_assist,
                 use_cache=use_cache, all_runs=all_runs, resume=True)




def _fasterq(run: str, dest: Path, basename: str, paired: bool) -> tuple[bool, str]:
    """Fallback when ENA hosts no fastq for the run."""
    tmp = dest / ("_tmp_" + run)
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = [FASTERQ, "--split-3", "--threads", "8", "--outdir", str(tmp), run]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    except subprocess.TimeoutExpired:
        return False, "fasterq-dump timed out"
    if r.returncode != 0:
        return False, (r.stderr or "").strip()[:300]
    produced = sorted(tmp.glob("*.fastq"))
    if not produced:
        return False, "fasterq-dump produced no fastq"
    for i, f in enumerate(produced):
        suffix = ("_R%d" % (i + 1)) if len(produced) > 1 else ""
        target = dest / ("%s%s.fastq.gz" % (basename, suffix))
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as fh:
            subprocess.run(["gzip", "-c", str(f)], stdout=fh, check=True)
        f.unlink()
    tmp.rmdir()
    return True, ""


def fetch(accession, dest: Optional[str] = None,
          force: bool = False, llm_assist: bool = True,
          use_cache: bool = True, all_runs: bool = False,
          resume: bool = False) -> dict:
    """Download every usable run of the study into <dest>/<modality>/.

    Refuses to start unless verdict() says ok, so an unlabelled arm or a missing
    modality stops here rather than after hundreds of gigabytes.
    """
    # When the caller names the destination we become visible from the first
    # instant: the launch record and a "resolving" progress file go down BEFORE
    # the lookup, which by itself can take minutes. Without them a poll during
    # the lookup window cannot tell a live fetch from one that never started.
    given = Path(dest) if dest else None
    # Two fetches writing the same fastq.gz would interleave their bytes, and the
    # second one's launch record would hide the first. Refuse instead.
    if given is not None:
        _live = live_fetch_pid(given)
        if _live:
            return {"state": "already_running", "dest": str(given), "pid": _live,
                    "reasons": ["a fetch (pid %d) is already running in %s"
                                % (_live, given)],
                    "user_message": "NOTHING new was started: a download (pid %d) is "
                                    "already running in %s. Poll it with sra-status, "
                                    "or stop it with sra-cancel --dest %s."
                                    % (_live, given, given)}
    accs_in: list[str] = []
    if given is not None:
        try:
            accs_in = parse_accessions(accession)
        except SRAError:
            accs_in = []
        write_launch(given, os.getpid(), sys.argv, accs_in)
        _write_progress(given, {"state": "resolving", "dest": str(given),
                                "accessions": accs_in, "total_files": 0,
                                "done": 0, "failed": 0, "errors": [],
                                "current": ""})
    try:
        rep = lookup(accession, llm_assist=llm_assist, use_cache=use_cache,
                     all_runs=all_runs)
    except Exception as exc:
        if given is not None:
            mark_failed(given, "%s: %s" % (type(exc).__name__, exc))
        raise
    v = rep["verdict"]
    if v["status"] != "ok" and not force:
        refused = {"state": "refused", "study": rep["study"],
                   "studies": rep.get("studies", []),
                   "missing": v.get("missing", []),
                   "reasons": v["reasons"], "verdict": v}
        if given is not None:
            _write_progress(given, {k: val for k, val in refused.items()
                                    if k != "verdict"})
        return refused
    dest_p = given if given is not None else DATA_ROOT / rep["study"]
    dest_p.mkdir(parents=True, exist_ok=True)
    if given is None:          # the destination only became known after the lookup
        write_launch(dest_p, os.getpid(), sys.argv,
                     rep.get("input", []), rep.get("study", ""))
    usable = [s for s in rep["samples"] if s["modality"] in ("ATAC", "RNA")]

    # Filenames belong to the DIRECTORY, not to this lookup: a run that is
    # already here keeps the name it was written under, whatever this lookup
    # called its arm. Without this, "download my 8 runs" followed by "download
    # the whole study" wrote the same 8 runs twice (2026-09-23 fix).
    registry = read_name_registry(dest_p)
    namefix = reconcile_names(usable, registry)
    write_name_registry(dest_p, usable, registry)
    notes: list = []
    if namefix["reused"]:
        notes.append(
            "%d run(s) are already in this directory under different arm labels, so "
            "the existing filenames were reused instead of downloading a second copy "
            "(%s)." % (len(namefix["reused"]),
                       ", ".join("%s: %s -> %s" % (c["run"], c["from"], c["to"])
                                 for c in namefix["reused"][:6])))
    if namefix["deduped"]:
        notes.append(
            "%d run(s) would have collided with an existing filename and were "
            "renamed (%s)." % (len(namefix["deduped"]),
                               ", ".join("%s: %s -> %s" % (c["run"], c["from"], c["to"])
                                         for c in namefix["deduped"][:6])))

    # One job per FILE, not per run: that is the unit a percent can be rendered
    # for, and the unit a cancelled download resumes at.
    jobs: list[dict] = []
    plan: list[dict] = []
    for s in usable:
        size_by_url = {}
        for _u, _b in zip(s.get("fastq_ftp") or [], s.get("fastq_sizes") or []):
            size_by_url["https://" + str(_u).lstrip("/")] = int(_b or 0)
        for url, path in _targets(s, dest_p):
            job = {"run": s["run"], "modality": s["modality"], "arm": s["arm"],
                   "replicate": s.get("replicate"),
                   "file": path.name, "path": str(path),
                   "bytes_total": int(size_by_url.get(url, 0)),
                   "bytes_done": 0, "pct": 0.0, "state": "pending"}
            jobs.append(job)
            plan.append({"url": url, "target": path, "job": job,
                         "run": s["run"], "path": str(path),
                         "bytes": job["bytes_total"]})
    sel = rep.get("selection") or {}
    prog = {"state": "running", "study": rep["study"], "dest": str(dest_p),
            "from_cache": bool(rep.get("from_cache")),
            "pid": os.getpid(), "pgid": _pgid(), "accessions": rep.get("input", []),
            "selection": sel,
            "selected_runs": rep.get("selected_runs", []),
            "n_selected": rep.get("n_selected", len(usable)),
            "n_total": rep.get("n_total", len(usable)),
            "resumed": bool(resume),
            "total_files": len(jobs), "done": 0, "failed": 0,
            "total_bytes": v["total_bytes"], "errors": [], "current": "",
            "notes": notes, "renamed": namefix["reused"],
            "renamed_conflicts": namefix["deduped"],
            "jobs": jobs, "started": time.time()}

    (dest_p / "sra_manifest.json").write_text(json.dumps(rep, indent=2))

    by_run: dict[str, dict] = {s["run"]: s for s in usable}
    failed_runs: set[str] = set()
    prog["skipped_unverified"] = 0

    # Pass 1 - what is already on disk. A non-empty file is NOT automatically a
    # complete one (2026-09-23): a killed or network-dropped download leaves a
    # truncated fastq.gz, and counting it as done reported state=done over
    # corrupt data. Only ENA's own byte count may retire a file; anything
    # shorter is recorded as partial so pass 2 continues it.
    for item in plan:
        job, target = item["job"], item["target"]
        try:
            have = target.stat().st_size if target.exists() else 0
        except OSError:
            have = 0
        want = int(job["bytes_total"])
        if have <= 0:
            continue
        if want and have == want:
            set_job_bytes(job, have)
            job["state"] = "done"
        elif not want:
            # ENA reported no size for this file: nothing to verify against, so
            # the old skip behaviour stands but is counted and disclosed.
            job["bytes_done"] = have
            job["pct"] = 100.0
            job["state"] = "done"
            job["unverified"] = True
            prog["skipped_unverified"] += 1
        elif have > want:
            job["state"] = "failed"
            job["error"] = ("existing file is %d bytes but ENA reports %d; it is "
                            "not a truncated download and was left untouched - "
                            "delete it and retry" % (have, want))
            set_job_bytes(job, want)
            prog["errors"].append({"file": job["file"], "error": job["error"]})
        else:
            set_job_bytes(job, have)       # truncated: resume from here

    bytes_at_start = sum(int(j.get("bytes_done") or 0) for j in jobs)
    writer = _ProgressWriter(dest_p, prog, bytes_at_start=bytes_at_start)
    writer.refresh(force=True)

    # Pass 2 - transfer what is left, publishing a percent every ~2 s.
    for item in plan:
        job, target = item["job"], item["target"]
        if job["state"] in ("done", "failed"):
            continue
        if int(job["bytes_done"] or 0) > 0:
            # stderr, not stdout: `mrag sra-fetch` prints a JSON document and a
            # caller parsing it must not have to strip a prose preamble first.
            print("resuming %s: %d of %d bytes already on disk"
                  % (job["file"], job["bytes_done"], job["bytes_total"]),
                  file=sys.stderr, flush=True)
        job["state"] = "downloading"
        prog["current"] = job["file"]
        writer.refresh(force=True)

        def _tick(n, _j=job, _w=writer):
            set_job_bytes(_j, n)
            _w.refresh()

        ok, err, final = _download_one(item["url"], target, job["bytes_total"],
                                       tick=_tick)
        set_job_bytes(job, final)
        if ok:
            job["state"] = "done"
            job["pct"] = 100.0
            job.pop("error", None)
        else:
            job["state"] = "failed"
            job["error"] = err
            failed_runs.add(job["run"])
            prog["errors"].append({"file": job["file"], "error": err})
        writer.refresh(force=True)

    # ENA occasionally has no mirrored fastq; retry those runs through SRA itself.
    for run in sorted(failed_runs):
        s = by_run.get(run)
        if not s:
            continue
        prog["current"] = "fasterq-dump " + run
        writer.refresh(force=True)
        ok, err = _fasterq(run, dest_p / s["modality"], s["basename"],
                           s["layout"] == "PAIRED")
        if ok:
            prog["errors"] = [e for e in prog["errors"]
                              if not e["file"].startswith(s["basename"])]
            for j in jobs:
                if j["run"] != run:
                    continue
                try:
                    j["bytes_done"] = Path(j["path"]).stat().st_size
                except OSError:
                    pass
                j["pct"] = 100.0
                j["state"] = "done"
                j["via"] = "fasterq-dump"
                j.pop("error", None)
        else:
            prog["errors"].append({"file": run, "error": "fasterq-dump: " + err})
        writer.refresh(force=True)

    # errors holds BOTH per-file errors and per-run fasterq-dump fallback errors,
    # so len(errors) could exceed total_files (16 files, 23 "failures"). Keep the
    # fallback errors visible but out of the file count (2026-09-23 fix).
    prog["fallback_errors"] = [e for e in prog["errors"]
                               if str(e.get("error", "")).startswith("fasterq-dump:")]
    prog["failed"] = sum(1 for j in jobs if j.get("state") == "failed")
    prog["state"] = "done" if not prog["errors"] else "partial"
    writer.refresh(force=True)
    prog["current"] = ""
    if isinstance(prog.get("overall"), dict):
        prog["overall"]["current"] = ""
    _write_progress(dest_p, prog)
    return prog
