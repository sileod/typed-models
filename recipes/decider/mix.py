#!/usr/bin/env python3
"""Mix grouped training manifests by request share into one shuffled manifest.

  python3 recipes/decider/mix.py data/mix.jsonl --total 480000 --part data/jev_bal.jsonl:0.42 --part data/index_train.jsonl:0.3

A part smaller than its share is repeated (whole passes, then a random remainder); larger is subsampled.
"""

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--part", action="append", required=True, help="path:share")
    ap.add_argument("--total", type=int, required=True, help="requests in the output")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    parts = [(p.rsplit(":", 1)[0], float(p.rsplit(":", 1)[1])) for p in args.part]
    z = sum(w for _, w in parts)
    out, rec = [], {"total": args.total, "seed": args.seed, "parts": []}
    for path, w in parts:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        n = round(args.total * w / z)
        take = rows * (n // len(rows)) + rng.sample(rows, n % len(rows))
        out += take
        rec["parts"].append({"path": path, "share": w / z, "available": len(rows), "taken": n, "epochs": round(n / len(rows), 2)})
        print(f"{path}: {len(rows):,} available, {n:,} taken ({n / len(rows):.2f} passes)")
    rng.shuffle(out)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(args.out).with_suffix(".record.json").write_text(json.dumps(rec, indent=2))
    print(f"{args.out}: {len(out):,} requests")


if __name__ == "__main__":
    main()
