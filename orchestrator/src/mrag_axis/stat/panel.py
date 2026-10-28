"""Rebuild the promoter-end "panel" item set from inputs that survive stage 2.

The old ag_validation node wrote <run>/output/ag_validation/ag_items.json and agprep_D*.py grouped
it into the panel items that went into the edits JSONs.  Neither script is part of stage 2, so the
items are rebuilt here from the ranked consensus TSV + hg38.  Verified against the stored
artefacts: ag_items() is object-for-object identical to the real ag_items.json (80/80 on
the PRAD run, 96/96 on the OC2 prostate-domain run) and panel_items() reproduces every
published panel item of edits_D_prad.json (46) and edits_D.json (35), with 0 recon-only and 0
published-only (chrom, peak_start, peak_end, gene) triples.

The source file is viz_built/consensus_ranked.tsv.  It is NOT consensus_ranked_graph.tsv: the
graph variant gets 10 of the 46 PRAD peaks wrong, because it carries the graph's own peak
coordinates rather than the consensus table's.
"""
import collections, json, math

HG38_FA = "/BIO1/MRAG/Agent/Reference/ATAC_hg38/hg38.fa"
RANKED_TSV = "viz_built/consensus_ranked.tsv"
# The HOMER alias half of the NAME map, as agprep_D*.py applied it.  mrag_axis.stat.config.name_map() derives
# the same two keys from production's HOMER_NAME_ALIASES for any run whose TRN contains NR3C1.
NAME = {"GR": "NR3C1", "GRE": "NR3C1"}


def ag_items(run_output_dir, ranked=RANKED_TSV, top_peaks_per_tf=8, max_tfs=12,
             genome_fa=HG38_FA, counts=None):
    """AGValidationNode.run()'s item list, rebuilt.  Row order is the TSV's own order.

    Order is load-bearing twice over: panel_items() numbers its labels positionally, and the
    stored per-pair AlphaGenome/Corgi results are keyed by that label, so any reordering silently
    decouples a stored result from the item it was measured on.

    pandas.sort_values() is NOT stable (kind='quicksort' by default) and the rank-1 combined_p
    column is heavily tied -- 13 rank-1 rows with 5 distinct values on the OC2 run (one tie of 6,
    one of 4), 10 rows with 4 distinct values on PRAD.  Today neither tie straddles max_tfs=12, so
    the cut is unambiguous; a future run whose tie does straddle it can select a different TF set
    from the same input, and then the labels move too.  Do not add a `kind='stable'` here without
    re-deriving the stored results: it would change which TFs the 2026-10 runs chose.
    """
    import pandas as pd
    from pyfaidx import Fasta
    df = pd.read_csv(f"{run_output_dir}/{ranked}", sep="\t", low_memory=False)
    df_sub = df[df["rank_within_tf"] <= top_peaks_per_tf].copy()
    tf_order = (df[df["rank_within_tf"] == 1]
                .sort_values("combined_p", ascending=True)["TF"].tolist())[:max_tfs]
    df_sub = df_sub[df_sub["TF"].isin(tf_order)].reset_index(drop=True)
    fa = Fasta(genome_fa, sequence_always_upper=True)
    dropped = collections.Counter() if counts is None else counts
    out = []
    for _, row in df_sub.iterrows():
        chrom = str(row["Chr"]); s = int(row["Start"]); e = int(row["End"])
        centre = (s + e) // 2
        try:
            ref = str(fa[chrom][centre:centre + 1]).upper()
        except Exception:
            dropped["fasta_error"] += 1; continue
        if not ref or ref == "N":
            dropped["ref_N"] += 1; continue
        # NaN guard the old node did not have: Fold and log2FC are joined in from DiffBind and
        # the RNA table, so a peak with no RNA partner can carry NaN, and json.dump would then
        # write a bare NaN that is not valid JSON and that compares unequal to itself downstream.
        # Both 2026-10 runs have 0 such rows, so this guard does not move any published number.
        obs = {}
        for k, col, dflt in (("observed_atac_fc", "Fold", None),
                             ("observed_log2fc", "log2FC", 0.0)):
            if col not in row.index:
                if dflt is None:               # Fold is required; the old node let it KeyError
                    raise KeyError(f"{ranked} has no {col} column")
                obs[k] = dflt; continue
            v = float(row[col])
            if not math.isfinite(v):
                dropped[f"nan_{col}"] += 1
                v = None
            obs[k] = v
        out.append({"tf": str(row["TF"]), "chrom": chrom,
                    "peak_start": s, "peak_end": e, "ref": ref,
                    "observed_atac_fc": obs["observed_atac_fc"],
                    "observed_log2fc": obs["observed_log2fc"],
                    "label": f"{row['TF']}->{row['Gene Name']}"})
    return out, tf_order


def panel_items(run_output_dir, ranked=RANKED_TSV, top_peaks_per_tf=8, max_tfs=12,
                genome_fa=HG38_FA, name_map=NAME, items=None):
    """The panel items, grouped exactly as agprep_D*.py grouped them.

    One item per distinct (chrom, peak_start, peak_end, gene); the label is positional,
    P_<gene>_<k> with k counting grouped peaks in first-appearance order over the TSV.  The peak
    is re-emitted as [start-1, end], which is the half-open form the edits JSONs carry.
    """
    ag = items if items is not None else ag_items(
        run_output_dir, ranked=ranked, top_peaks_per_tf=top_peaks_per_tf, max_tfs=max_tfs,
        genome_fa=genome_fa)[0]
    byp = collections.OrderedDict()
    for it in ag:
        key = (it["chrom"], it["peak_start"], it["peak_end"], it["label"].split("->")[1])
        d = byp.setdefault(key, {"tfs": set(), "l2fc": it["observed_log2fc"],
                                 "atac": it["observed_atac_fc"]})
        d["tfs"].add(name_map.get(it["tf"], it["tf"]))
    out = []
    for k, ((chrom, ps, pe, g), d) in enumerate(byp.items(), 1):
        out.append({"label": f"P_{g}_{k}", "gene": g, "chrom": chrom,
                    "peak": [int(ps) - 1, int(pe)],
                    "obs_log2fc": d["l2fc"], "obs_atac_fold": d["atac"],
                    "claimed_tfs": sorted(d["tfs"])})
    return out


def published_panel_items(edits_json):
    """The panel items an edits JSON already carries, for the equality check."""
    return [x for x in json.load(open(edits_json))["items"] if x.get("source") == "panel"]


def diff_panel(recon, published, fields=("label", "obs_log2fc", "obs_atac_fold", "claimed_tfs")):
    """{recon_only, published_only, mismatched} over (chrom, peak_start, peak_end, gene)."""
    trip = lambda x: (x["chrom"], x["peak"][0], x["peak"][1], x["gene"])
    R, P = {trip(x): x for x in recon}, {trip(x): x for x in published}
    bad = {}
    for t in sorted(set(R) & set(P)):
        d = {}
        for f in fields:
            a, b = R[t][f], P[t][f]
            if isinstance(a, float) and isinstance(b, float):
                if abs(a - b) > 1e-12:
                    d[f] = (a, b)
            elif a != b:
                d[f] = (a, b)
        if d:
            bad[t] = d
    return {"recon_only": sorted(set(R) - set(P)), "published_only": sorted(set(P) - set(R)),
            "mismatched": bad}
