"""Run DAStk's differential_md_score.py with a fixed random seed per motif.

DAStk (BSD-3-Clause, Dowell lab) estimates each motif's MD-score variance from 1,099 bootstrap
resamples (sklearn.utils.resample without a random_state) inside a process pool, so its differential
p-values -- and the TF ranking built on them -- changed from run to run on identical input.  This
wrapper runs the unmodified DAStk script with resample seeded from the motif name, so the same input
gives the same p-values whatever worker handles which motif.

usage: dastk_seeded.py <DAStk package dir holding differential_md_score.py> <its arguments...>
"""
import importlib.util
import multiprocessing
import sys
import zlib

_D = None        # the loaded DAStk differential_md_score module
_ORIG = None     # its unpatched get_differential_md_scores


def seeded_differential_md_scores(diff_params):
    """DAStk's per-motif function with resample drawing from RandomState(crc32(motif name))."""
    import numpy as np
    from sklearn.utils import resample
    rs = np.random.RandomState(zlib.crc32(str(diff_params[0]).encode()))
    _D.resample = lambda *a, **k: resample(*a, random_state=rs, **k)
    try:
        return _ORIG(diff_params)
    finally:
        _D.resample = resample


def main():
    global _D, _ORIG
    dastk_dir, args = sys.argv[1].rstrip("/"), sys.argv[2:]
    # the pool's workers must inherit the patched module (fork), not re-import DAStk
    multiprocessing.set_start_method("fork", force=True)
    spec = importlib.util.spec_from_file_location("dastk_differential_md_score",
                                                  dastk_dir + "/differential_md_score.py")
    _D = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_D)
    _ORIG = _D.get_differential_md_scores
    _D.get_differential_md_scores = seeded_differential_md_scores
    sys.argv = [dastk_dir + "/differential_md_score.py"] + args
    _D.main()


if __name__ == "__main__":
    main()
