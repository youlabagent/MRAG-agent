"""On-demand locus snapshot generation + report-inclusion management.

Lets the chat user render a peak graph for ANY target gene at any time, and
optionally add/remove it from report Section 12 (Locus Snapshots).
Runs inside the webui (paper_qa venv) which has pyBigWig + the pyGenomeTracks
binary, so it can call track_helper.plot_atac_tracks directly.
"""
from __future__ import annotations
import os, re, csv, json, sys, base64
from pathlib import Path

TRACK_HELPER = "/BIO1/MRAG/Agent/libs/track_helper"
GTF = "/BIO1/MRAG/Agent/Reference/ATAC_hg38/hg38.refGene.gtf"
csv.field_size_limit(10_000_000)


def out_dir_of(runs_root, run_id) -> Path:
    return (Path(runs_root) / run_id / "output").resolve()


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_").lower()


def _atac_rep_bigwigs(out_dir: Path):
    """ATAC per-replicate bigWigs used by the default locus_snapshots node
    (report Section 8 panels). Read from the atac step's 'bigwigs' output in
    the manifest so chat snapshots match the report exactly (4 ATAC reps,
    shared ATAC y-scale) instead of the peakheatmap group-mean/RNA-cpm tracks."""
    mp = out_dir / "manifest.json"
    if not mp.exists():
        return None
    try:
        m = json.loads(mp.read_text())
    except Exception:
        return None
    for st in m.get("steps", []):
        for o in st.get("outputs", []):
            if not isinstance(o, dict) or o.get("key") != "bigwigs":
                continue
            paths = [x for x in str(o.get("path", "")).split(";")
                     if x.strip().endswith(".bw") and Path(x.strip()).exists()]
            if len(paths) >= 2:
                return [x.strip() for x in paths]
    return None


def _arm_map(out_dir: Path):
    """sample-name -> arm label, parsed from run.toml (no tomllib dep)."""
    rt = out_dir.parent / "run.toml"
    m = {}
    if rt.exists():
        for blk in rt.read_text().split("[[samples]]")[1:]:
            n = re.search(r'name\s*=\s*"([^"]+)"', blk)
            a = re.search(r'arm\s*=\s*"([^"]+)"', blk)
            if n and a:
                m[n.group(1)] = a.group(1)
    return m


def _samples(out_dir: Path):
    """Track list [[label, bigwig, None], ...] for locus snapshots.

    Prefer the ATAC per-replicate bigWigs (same set the report panels use, e.g.
    the TPM4 example): labelled {arm}_rep{n} with a shared ATAC y-axis. Fall
    back to any bigWigs found in the run output only if those are unavailable."""
    bws = _atac_rep_bigwigs(out_dir)
    if bws:
        arm = _arm_map(out_dir)
        out = []
        for x in bws:
            stem = Path(x).stem
            rep = re.search(r'rep(\d+)', stem)
            label = f"{arm.get(stem, stem)}_rep{rep.group(1)}" if rep \
                    else arm.get(stem, stem)
            out.append([label, x, None])
        return out
    reps = sorted(p for p in out_dir.glob("**/*.bw")
                  if ".grp.bw" not in p.name and "user_locus" not in str(p))
    if reps:
        return [[p.stem, str(p), None] for p in reps]
    grp = sorted(out_dir.glob("peakheatmap/*.grp.bw"))
    if grp:
        return [[p.name.split(".grp")[0], str(p), None] for p in grp]
    raise FileNotFoundError("no bigwig tracks found for this run")


def _consensus_tsv(out_dir: Path):
    """Ordered peak->gene consensus tables to search. Curated ranked tables
    (one representative peak per TF target) first; the full HOMER-annotated
    per-branch consensus (all differential peaks, thousands of genes) as the
    fallback so ANY differential target gene resolves to its real peak."""
    cands = []
    for c in ("viz_built/consensus_ranked.tsv",
              "viz_built/consensus_ranked_graph.tsv"):
        fp = out_dir / c
        if fp.exists():
            cands.append(fp)
    cands += sorted((out_dir / "branches").glob("*/consensus_*.tsv"))
    return cands


def _region_from_consensus(out_dir: Path, gene: str, flank: int):
    want = gene.strip().lower()
    for fp in _consensus_tsv(out_dir):
        best = None
        with open(fp, newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                if str(row.get("Gene Name", "")).strip().lower() != want:
                    continue
                try:
                    fold = abs(float(row.get("Fold") or 0))
                except ValueError:
                    fold = 0.0
                if best is None or fold > best[0]:
                    best = (fold, row)
        if not best:
            continue  # not in this table; try the next (fuller) one
        row = best[1]
        chrom = str(row["Chr"]); s = int(float(row["Start"])); e = int(float(row["End"]))
        region = f"{chrom}:{max(0, s - flank)}-{e + flank}"
        mid = (s + e) // 2; half = max(int((e - s) * 0.25), 200)
        hl = [chrom, max(s, mid - half), min(e, mid + half)]
        return region, hl, f"differential peak (|Fold|={best[0]:.2f})"
    return None


def _region_from_gtf(gene: str, flank: int, max_body: int = 40000,
                     tss_pad: int = 15000):
    chrom = None; lo = None; hi = None; strand = None
    pat = f'gene_name "{gene}"'
    with open(GTF) as f:
        for line in f:
            if pat not in line:
                continue
            c = line.split("\t")
            if len(c) < 7 or c[2] != "transcript":
                continue
            if chrom is None:
                chrom = c[0]; strand = c[6]
            if c[0] != chrom:
                continue
            a, b = int(c[3]), int(c[4])
            lo = a if lo is None else min(lo, a)
            hi = b if hi is None else max(hi, b)
    if chrom is None:
        return None
    if hi - lo <= max_body:
        region = f"{chrom}:{max(0, lo - flank)}-{hi + flank}"
        return region, None, "gene body (refGene)"
    # Large gene, no differential peak: a whole-body window hides bp-scale
    # ATAC/RNA signal, so center a bounded window on the TSS.
    tss = lo if strand != "-" else hi
    region = f"{chrom}:{max(0, tss - tss_pad)}-{tss + tss_pad}"
    return region, None, "TSS-centered (no differential peak)"


def resolve_region(out_dir: Path, gene: str, flank: int):
    r = _region_from_consensus(out_dir, gene, flank)
    if r:
        return r
    r = _region_from_gtf(gene, flank)
    if r:
        return r
    raise ValueError(f"could not resolve a genomic region for gene '{gene}'")


def generate(runs_root, run_id: str, gene: str, flank: int = 5000) -> dict:
    out_dir = out_dir_of(runs_root, run_id)
    if not out_dir.exists():
        raise FileNotFoundError(f"no such run output: {run_id}")
    gene = gene.strip()
    region, hl, source = resolve_region(out_dir, gene, flank)
    samples = _samples(out_dir)
    udir = out_dir / "user_locus"; udir.mkdir(parents=True, exist_ok=True)
    if TRACK_HELPER not in sys.path:
        sys.path.insert(0, TRACK_HELPER)
    from mrag_track_helper import plot_atac_tracks
    res = plot_atac_tracks(
        samples=[tuple(s) for s in samples], region=region,
        title=_slug(gene), out_dir=str(udir), gene_title=gene,
        max_value="shared_max", width=16.5, dpi=180,
        highlight=tuple(hl) if hl else None, formats=("png",),
    )
    png = res.get("png")
    return {"ok": True, "gene": gene, "region": region, "source": source,
            "png": png, "png_rel": os.path.relpath(png, out_dir)}


# ---------- report Section 12 inclusion ----------

def _report_list_path(out_dir: Path) -> Path:
    return out_dir / "user_locus" / "report_locus.json"


def _seed_from_default(out_dir: Path):
    mani = out_dir / "locus_snapshots" / "locus_manifest.json"
    if not mani.exists():
        return []
    try:
        items = json.loads(mani.read_text())
    except Exception:
        return []
    return [{"gene": it.get("gene"), "tf": it.get("tf", ""),
             "region": it.get("region", ""), "png_path": it.get("png_path"),
             "source": "default"} for it in items if it.get("png_path")]


def load_report_list(out_dir: Path):
    fp = _report_list_path(out_dir)
    if fp.exists():
        try:
            return json.loads(fp.read_text())
        except Exception:
            pass
    return _seed_from_default(out_dir)


def save_report_list(out_dir: Path, panels):
    fp = _report_list_path(out_dir)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(panels, indent=2))


def _figs_html(out_dir: Path, panels):
    figs = []
    i = 0
    for t in panels:
        p = t.get("png_path")
        if not p or not Path(p).exists():
            continue
        i += 1
        b = Path(p).read_bytes()
        du = "data:image/png;base64," + base64.b64encode(b).decode()
        tf = t.get("tf", "") or "custom"
        arrow = f"{tf} &rarr; {t.get('gene')}" if tf else t.get("gene")
        figs.append(
            f'<div class="figure"><img src="{du}" alt="{t.get("gene")}">'
            f'<figcaption>Fig 4.{i} <strong>{arrow}</strong> '
            f'({t.get("region","")})</figcaption></div>')
    if not figs:
        figs.append('<p style="color:var(--muted)">No locus panels selected.</p>')
    return "\n".join(figs)


def splice_report(out_dir: Path, panels):
    rep = out_dir / "mrag_report.html"
    if not rep.exists():
        return False
    html = rep.read_text()
    section = (
        '<section id="tracks">\n'
        '  <h2>12. Locus Snapshots — Key TF → Target Examples</h2>\n'
        '  <p style="font-size:12px;color:var(--muted)">One representative '
        'target per TF plus user-added peak graphs (entered through chat). '
        'Tracks: condition group means, shared y-axis max for comparison.</p>\n'
        '  <div class="fig-grid-2">\n' + _figs_html(out_dir, panels) +
        '\n  </div>\n</section>')
    new_html, n = re.subn(r'<section id="tracks">.*?</section>', section,
                          html, count=1, flags=re.S)
    if n:
        rep.write_text(new_html)
    return bool(n)


def add_to_report(runs_root, run_id: str, gene: str, flank: int = 5000):
    out_dir = out_dir_of(runs_root, run_id)
    g = generate(runs_root, run_id, gene, flank)
    panels = load_report_list(out_dir)
    panels = [p for p in panels if str(p.get("gene", "")).lower() != gene.strip().lower()]
    panels.append({"gene": g["gene"], "tf": "custom", "region": g["region"],
                   "png_path": g["png"], "source": "user"})
    save_report_list(out_dir, panels)
    spliced = splice_report(out_dir, panels)
    return {"ok": True, "gene": g["gene"], "region": g["region"],
            "png_rel": g["png_rel"], "n_panels": len(panels), "spliced": spliced}


def remove_from_report(runs_root, run_id: str, gene: str):
    out_dir = out_dir_of(runs_root, run_id)
    panels = load_report_list(out_dir)
    before = len(panels)
    panels = [p for p in panels if str(p.get("gene", "")).lower() != gene.strip().lower()]
    save_report_list(out_dir, panels)
    spliced = splice_report(out_dir, panels)
    return {"ok": True, "removed": before - len(panels),
            "n_panels": len(panels), "spliced": spliced}


def list_report(runs_root, run_id: str):
    out_dir = out_dir_of(runs_root, run_id)
    panels = load_report_list(out_dir)
    return {"ok": True, "panels": [{"gene": p.get("gene"), "tf": p.get("tf", ""),
            "region": p.get("region", ""), "source": p.get("source", "")}
            for p in panels]}
