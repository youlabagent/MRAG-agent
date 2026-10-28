"""The JASPAR motif layer of the enhancer-end prep: PWMs, the two-strand scan, and the cap of 40.

Productionised from /BIO1/MRAG/corgi/ft/seko.py (load_pwms, scan) and the sampling line of
/BIO1/MRAG/corgi/ft/ag/agprep_A.py.  Three things here are load-bearing for reproducing
ft/ag/edits_A.json and ft/prad/edits_A_prad.json and are therefore kept bit-for-bit:

* the PWM is log-odds against a UNIFORM 0.25 background, after a 1e-3 pseudocount that is added
  before renormalising each column -- not after, which would change every score slightly;
* the cutoff is relative, worst + frac*(best - worst) with frac=0.85, applied to both the matrix
  and its reverse complement, because the experiment compares TFs of different length and
  information content against each other (seko.scan's docstring);
* when a (region, TF) has more than max_motifs hits the kept subset is
  sorted(rng.choice(n, max_motifs, replace=False)) drawn from the build's single RNG, so the draw
  sits in the stream exactly where the research script put it.

read_meme() is a local MEME reader rather than tangermeme.io.read_meme: tangermeme is installed
only in the Corgi runtime venv (/BIO1/MRAG/Agent/libs/corgi_client/.venv), and the prep has to run
wherever the node runs.  test_prep_enhancer.py checks the two readers agree on all 879 motifs of
the JASPAR2024 file to the last bit, so this substitution is measured, not assumed.
"""
from __future__ import annotations
import re
import numpy as np

# Moved with the Corgi runtime into the deployable tree; md5 96b2e87e2344de6b85f5f4e084e8486f,
# byte-identical to the research copy at /BIO1/MRAG/corgi/ext/ref/ that seko.py reads.
JASPAR_MEME = ("/BIO1/MRAG/Agent/Reference/corgi/"
               "JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt")

# seko.LUT: A,C,G,T -> 0..3 and every other byte -> -1, so an N in hg38 invalidates its window
# instead of scoring as a base.
LUT = np.full(256, -1, dtype=np.int16)
for _i, _b in enumerate("ACGT"):
    LUT[ord(_b)] = _i

_W_RE = re.compile(r"\bw=\s*(\d+)")
_CACHE: dict = {}


def read_meme(path=JASPAR_MEME):
    """{'<id> <name>': (4, w) float64 probability matrix} in FILE ORDER.

    Order and duplicates both matter: load_pwms keeps the first MOTIF per symbol, and the JASPAR2024
    CORE vertebrates non-redundant file carries TFAP2A three times and HNF4A twice, so a reader that
    reordered the file would silently swap in a different matrix for those two factors.
    """
    k = ("meme", path)
    if k in _CACHE:
        return _CACHE[k]
    out, key, want, rows = {}, None, 0, None
    with open(path) as fh:
        for line in fh:
            if line.startswith("MOTIF"):
                key, want, rows = line.split(None, 1)[1].strip(), 0, None
            elif "letter-probability matrix" in line and key is not None:
                m = _W_RE.search(line)
                want, rows = (int(m.group(1)) if m else 0), []
            elif rows is not None:
                f = line.split()
                if len(f) == 4:
                    rows.append([float(x) for x in f])
                    if len(rows) == want:
                        if key not in out:          # first MOTIF of a repeated name wins
                            out[key] = np.asarray(rows, dtype=np.float64).T
                        key, want, rows = None, 0, None
                elif f:
                    rows = None                     # URL / stray line: matrix ended early
    _CACHE[k] = out
    return out


def load_pwms(names, jaspar=JASPAR_MEME):
    """JASPAR PWMs for the requested TFs, as log-odds against a uniform background (seko.load_pwms).

    The key rule is the research one: the MOTIF line's display name upper-cased, so 'MA0004.1 Arnt'
    is ARNT; a factor with no such MOTIF is simply absent from the result, which is what makes
    TRN_TFS/NO_PWM a filter rather than an error.
    """
    want = {str(n).upper() for n in names}
    out = {}
    for key, pfm in read_meme(jaspar).items():
        nm = key.split(" ", 1)[1].strip().upper() if " " in key else key.upper()
        if nm in want and nm not in out:
            p = np.asarray(pfm, dtype=np.float64)
            p = (p + 1e-3) / (p + 1e-3).sum(axis=0, keepdims=True)
            out[nm] = np.log2(p / 0.25)
    return out


def scan(seq, lo, frac=0.85):
    """Hits scoring above `frac` of the PWM's own score range, on either strand (seko.scan).

    Returned as sorted(set(...)) of (offset, length): a palindromic motif that scores on both
    strands at the same offset is ONE hit, not two, and the sort is what makes the knockout list
    left-to-right along the region.  A hit whose window contains a non-ACGT base is dropped.
    """
    L = lo.shape[1]
    if len(seq) < L:
        return []
    idx = LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    n = len(seq) - L + 1
    win = np.lib.stride_tricks.sliding_window_view(idx, L)
    valid = (win >= 0).all(axis=1)
    cols = np.arange(L)
    best, hits = lo.max(axis=0).sum(), []
    worst = lo.min(axis=0).sum()
    cut = worst + frac * (best - worst)
    lo_rc = lo[::-1, ::-1]
    for mat in (lo, lo_rc):
        sc = np.full(n, -np.inf)
        w = win[valid]
        sc[valid] = mat[w, cols].sum(axis=1)
        for pos in np.where(sc >= cut)[0]:
            hits.append((int(pos), L))
    return sorted(set(hits))


def sample_hits(hits, max_motifs, rng):
    """The cap of 40 motifs per (region, TF), as agprep_A.py draws it.

    The rng.choice call is skipped entirely when the cap is not exceeded -- that asymmetry is part
    of the stream, so a version that always drew would desynchronise every later region.
    """
    if len(hits) <= max_motifs:
        return hits
    return [hits[j] for j in sorted(rng.choice(len(hits), max_motifs, replace=False))]
