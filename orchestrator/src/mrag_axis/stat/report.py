"""The agreed definition scored per model: Corgi alone / AlphaGenome alone / both models.

From stats/s13_bymodel.py.  Pair-level unit, leave-one-out null, computed control-edit exclusion,
gene grouping at the target end, consensus intersected at row level.  Read from the stored per-pair
results; no model is called.

'both models' is not the intersection of the two single-model sets and must never be reported as
one: it means the same read-out passes under both models in the same row.
"""
import json
from . import stats as S

DOMS = ("pc", "nec", "sclc", "prad")


def by_model(domain, min_rd=1, src="all", excl="rule", cfg=None, **roots):
    """{model: the agreed definition's result} for one domain, plus the pairs the rule dropped."""
    cfg = cfg or S.domains(**roots)[domain]
    out = {}
    for name, mods in S.MODELS.items():
        r = S.axis(domain, definition="unified", src=src, min_rd=min_rd, models=mods,
                   excl=excl, cfg=cfg)
        out["dropped_pairs"] = r.pop("dropped_pairs")
        out[name] = r
    return out


def headline_key(domain, res_one_model):
    """Which row a domain's Holm family uses.  PRAD's declared extra requirement is the AR gate,
    so that is its headline; every other domain has only the enhancer row."""
    return ("enhancer_plus_AR" if domain == "prad" and "enhancer_plus_AR" in res_one_model
            else "enhancer")


def holm(res):
    """Holm correction over the headline test of every domain x model the report would quote.

    The family is the set of headline tests actually printed, which is what prices the
    multiplicity in; the target-end and graph rows are descriptive and are not in it.
    """
    heads = []
    for d, r in res.items():
        for m in S.MODELS:
            k = headline_key(d, r[m])
            heads.append((r[m][k]["p"], d, m))
    heads.sort()
    return {(d, m): min(1.0, (len(heads) - i) * q) for i, (q, d, m) in enumerate(heads)}


def table(res, out=print):
    """The per-model table, in the prototype's layout (stats/out/s13_bymodel.json's companion)."""
    h = holm(res)
    hdr = (f"{'domain':6s} {'model':12s} {'end':18s} {'pass':>5s}/{'n':<5s} {'fp':>4s}/{'n':<4s}"
           f" {'OR':>7s} {'p':>9s} {'Holm':>9s}")
    out(hdr); out("-" * len(hdr))
    for d, r in res.items():
        for m in S.MODELS:
            for k in ("enhancer", "enhancer_plus_AR", "target"):
                if k not in r[m]:
                    continue
                v = r[m][k]
                o = "inf" if v["odds_ratio"] is None else f"{v['odds_ratio']:.2f}"
                hk = headline_key(d, r[m])
                hp = f"{h[(d, m)]:9.1e}" if k == hk else f"{'-':>9s}"
                out(f"{d:6s} {m:12s} {k:18s} {v['passed']:5d}/{v['n']:<5d}"
                    f" {v['fp']:4d}/{v['n_frn']:<4d} {o:>7s} {v['p']:9.1e} {hp}")
            t = r[m]["graph_triples"]
            out(f"{d:6s} {m:12s} {'graph (descr.)':18s} {t['axes']:5d} triples / {t['n_triples']}, "
                f"{t['false']} foreign / {t['n_foreign']}, {t['SE']} SE {t['TF']} TF "
                f"{t['target']} tgt")
        out(f"{d:6s} control-edit rule drops {len(r['dropped_pairs'])} pairs: {r['dropped_pairs']}")


def run(doms=DOMS, path=None, **kw):
    """Score every domain, print the table, optionally write the JSON."""
    res = {d: by_model(d, **kw) for d in doms}
    table(res)
    if path:
        json.dump({"holm": {f"{d}|{m}": v for (d, m), v in holm(res).items()}, "domains": res},
                  open(path, "w"), indent=1, default=str)
    return res
