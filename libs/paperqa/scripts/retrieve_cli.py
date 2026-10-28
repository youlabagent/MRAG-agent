#!/usr/bin/env python3
"""CLI wrapper around mrag_paperqa.Retriever for subprocess use by narrative.py."""
import argparse, json, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--cancer", default="all")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from mrag_paperqa.retriever import Retriever
    r = Retriever(device="cpu")  # cpu avoids GPU contention with GLM ollama
    # 2026-09-17 (llmdomain): a comma-separated list ("prostate,nec") is one
    # MatchAny filter; a single key stays a MatchValue filter; "all" = no filter.
    _can = [x.strip() for x in (args.cancer or "").split(",") if x.strip()]
    if not _can or any(x.lower() == "all" for x in _can):
        cancer_arg = None
    elif len(_can) == 1:
        cancer_arg = _can[0]
    else:
        cancer_arg = _can
    hits = r.search(args.query, top_k=args.top_k, cancer=cancer_arg)
    payload = {
        "hits": [
            {
                "pmid": h.pmid or "",
                "title": h.title or "",
                "snippet": (h.text or "")[:500],
                "score": h.score,
                "year": h.year,
                "cancer": h.cancer,
            }
            for h in hits if h.pmid
        ]
    }
    if args.json:
        sys.stdout.write(json.dumps(payload))
    else:
        for h in payload["hits"]:
            print(f"[PMID:{h['pmid']}] {h['title']}\n  {h['snippet'][:150]}\n")

if __name__ == "__main__":
    main()
