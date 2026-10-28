"""CLI: `ag` command for ad-hoc smoke tests and one-off queries."""
from __future__ import annotations
import argparse
import json
import sys

from .config import DEFAULT_TISSUE_PROSTATE, load_api_key
from .client import get_client
from .helpers import counterfactual_at_peak


def cmd_check_key(_):
    try:
        k = load_api_key()
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(f"OK: key loaded (len={len(k)}, prefix='{k[:6]}...')")
    try:
        _ = get_client()  # validates the key by establishing channel
        print("OK: DnaClient created")
    except Exception as e:
        print(f"WARN: client creation failed -> {e}", file=sys.stderr)
        return 2
    return 0


def cmd_counterfactual(args):
    out = counterfactual_at_peak(
        chrom=args.chrom,
        peak_start=args.start,
        peak_end=args.end,
        ref_at_centre=args.ref,
        alt=args.alt,
        window=args.window,
        requested_outputs=args.outputs.split(","),
        tissue=args.tissue,
        use_cache=not args.no_cache,
    )
    print(json.dumps(out, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ag")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("check", help="verify API key + client creation")
    s.set_defaults(func=cmd_check_key)

    s = sub.add_parser("cf", help="counterfactual prediction at a peak")
    s.add_argument("--chrom", required=True)
    s.add_argument("--start", required=True, type=int)
    s.add_argument("--end", required=True, type=int)
    s.add_argument("--ref", required=True, help="reference allele at peak centre")
    s.add_argument("--alt", default=None, help="alt allele (default: reverse-complement of ref)")
    s.add_argument("--window", type=int, default=16384)
    s.add_argument("--outputs", default="ATAC,RNA_SEQ,CHIP_TF")
    s.add_argument("--tissue", default=DEFAULT_TISSUE_PROSTATE)
    s.add_argument("--no-cache", action="store_true")
    s.set_defaults(func=cmd_counterfactual)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
