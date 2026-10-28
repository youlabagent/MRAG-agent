#!/bin/bash
# New run_id from a canonical run. Heavy immutable HOMER cache (homer_motif:
# motif TSVs + minmax BEDs) is SHARED via a symlink (computed once, never
# recomputed/copied). Small mutable artifacts are full-copied. Paths rewritten
# old->new. Uses the newest >=33-step manifest backup. Prints NEW dir.
set -e
SRC="$1"; LABEL="${2:-phaseFchain}"
RUNS=$(dirname "$SRC"); SRC_ID=$(basename "$SRC")
NEW_ID="run_$(date +%Y%m%dT%H%M%S)_${LABEL}"; DST="$RUNS/$NEW_ID"
mkdir -p "$DST/output"
# Seed homer_motif from the PROTECTED minmax cache (outside any run dir, never
# deleted by run cleanup). Contains the min-max BEDs + JASPAR minmax motif TSVs.
# The chain detects these (seeded fast-path) and skips the expensive per-rep
# scans + MinmaxBed. Real writable dir with per-file symlinks to the cache.
HOMER_CACHE="${HOMER_MINMAX_CACHE:-/BIO1/MRAG/Agent/cache/homer_minmax}"
mkdir -p "$DST/output/homer_motif"
if [ -d "$HOMER_CACHE" ]; then
  for f in "$HOMER_CACHE"/*peak_minmax.bed "$HOMER_CACHE"/*peak_minmax.motif.tsv; do
    [ -e "$f" ] && ln -s "$(readlink -f "$f")" "$DST/output/homer_motif/$(basename "$f")"
  done
fi
# also bring over any real (non-broken) files from the source homer_motif
if [ -d "$SRC/output/homer_motif" ]; then
  for f in "$SRC/output/homer_motif"/*; do
    b=$(basename "$f")
    [ -e "$DST/output/homer_motif/$b" ] && continue
    [ -e "$f" ] && ln -s "$(readlink -f "$f")" "$DST/output/homer_motif/$b"
  done
fi
for item in "$SRC/output"/*; do
  b=$(basename "$item"); [ "$b" = "homer_motif" ] && continue
  cp -a "$item" "$DST/output/$b"
done
cp -a "$SRC/run.toml" "$DST/run.toml" 2>/dev/null || true

# ---- Alias the seeded cache files to THIS run's condition labels ------------
# The protected cache is named after the sample prefixes it was built from
# (OC2_OE, cMyc_DDK), but the seeded fast-path in runner_motif_scan.py looks for
# "<condition label>_peak_minmax.{bed,motif.tsv}" (ONECUT2-OE, MYC-DDK). A run whose
# labels differ from the cache names therefore misses the cache and recomputes the
# per-replicate HOMER scans (~5 h on the OC2 samples).
# The mapping is DERIVED from the run's own sample names -- a cache file belongs to the
# arm whose samples carry that prefix -- and never from the order of the conditions,
# which would silently swap treatment and control. No mapping -> nothing is touched.
"${MRAG_PY:-/BIO1/MRAG/venvs/paper_qa/bin/python}" - "$DST" <<'PY' || true
import os, re, sys, tomllib
dst = sys.argv[1]
hm = os.path.join(dst, "output", "homer_motif")
try:
    cfg = tomllib.load(open(os.path.join(dst, "run.toml"), "rb"))
except Exception as e:
    print("[seed] no run.toml, cache left as seeded (%s)" % e); raise SystemExit(0)
labels = [c.get("label") for c in (cfg.get("conditions") or []) if c.get("label")]
# every spelling a sample of this arm could have given a cache file its name
cands = {}
for s in (cfg.get("samples") or []):
    arm, name = s.get("arm"), s.get("name") or ""
    if arm not in labels or not name:
        continue
    for pre in {re.sub(r"_(rep)?\d+$", "", name), name}:
        if pre:
            cands.setdefault(pre, set()).add(arm)
done = []
for f in sorted(os.listdir(hm)) if os.path.isdir(hm) else []:
    m = re.match(r"(.+?)_peak_minmax\.(bed|motif\.tsv)$", f)
    if not m:
        continue
    pre, ext = m.group(1), m.group(2)
    arms = cands.get(pre) or set()
    if len(arms) != 1:          # unknown prefix, or ambiguous: leave it alone
        continue
    arm = next(iter(arms))
    if arm == pre:
        continue                # already named for the condition
    src = os.path.realpath(os.path.join(hm, f))
    tgt = os.path.join(hm, "%s_peak_minmax.%s" % (arm, ext))
    if os.path.exists(tgt) or not os.path.exists(src):
        continue
    os.symlink(src, tgt)
    if os.path.islink(os.path.join(hm, f)):
        os.unlink(os.path.join(hm, f))   # a symlink to the same file; keep the dir tidy
    done.append("%s -> %s" % (f, os.path.basename(tgt)))
print("[seed] cache aliased to this run's condition labels: %s"
      % (", ".join(done) if done else "nothing to alias"))
PY
cp -a "$SRC"/*.json "$DST/" 2>/dev/null || true
# use newest complete (>=33-step) manifest backup
python3 - "$SRC" "$DST/output/manifest.json" "$SRC_ID" "$NEW_ID" <<'PY'
import json, glob, sys
src, dst_mp, sid, nid = sys.argv[1:5]
baks=sorted(glob.glob(src+'/output/manifest.json.bak_*'), reverse=True)
base=next((b for b in baks if len(json.load(open(b)).get('steps',[]))>=33), None) \
     or src+'/output/manifest.json'
txt=open(base).read().replace(sid, nid)
m=json.loads(txt); m['status']='running'
json.dump(m, open(dst_mp,'w'), indent=2)
PY
# rewrite any remaining old-id refs in small text files (run.toml, sheets)
grep -rIl "$SRC_ID" "$DST"/run.toml "$DST"/*.json 2>/dev/null | while read f; do
  sed -i "s|$SRC_ID|$NEW_ID|g" "$f"; done
echo "$DST"
