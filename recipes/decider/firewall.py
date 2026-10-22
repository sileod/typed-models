#!/usr/bin/env python3
"""Apply the template-insensitive content firewall (typed_models.firewall) to a grouped manifest: drop requests whose
content overlaps the Decision Index 0.3 suite.

  python3 recipes/decider/firewall.py data/jev_full.jsonl data/jev_cfw.jsonl
Writes the kept requests, <out>.record.json (drops per task) and <out>.dropped.jsonl (for audit).
"""

import argparse
import collections
import json
from pathlib import Path

from paths import SUITE
from typed_models.firewall import ContentFirewall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--span", type=int, default=30)
    ap.add_argument("--span_frac", type=float, default=0.2)
    args = ap.parse_args()
    fw = ContentFirewall(SUITE, frac=args.frac, span=args.span, span_frac=args.span_frac)
    print(f"suite: {len(fw.exact):,} texts, {len(fw.shingles):,} 6-grams", flush=True)
    kept = dropped = 0
    per_task, drop_task = collections.Counter(), collections.Counter()
    out = Path(args.out)
    with open(args.src, encoding="utf-8") as fi, open(out, "w", encoding="utf-8") as fo, \
            open(out.with_suffix(".dropped.jsonl"), "w", encoding="utf-8") as fd:
        for i, line in enumerate(fi):
            r = json.loads(line)
            per_task[r.get("task")] += 1
            if fw.request_hit(r.get("state") or "", r.get("options") or []):
                dropped += 1
                drop_task[r.get("task")] += 1
                fd.write(line)
            else:
                kept += 1
                fo.write(line)
            if i % 100000 == 0:
                print(f"{i:,} seen, {dropped:,} dropped", flush=True)
    rec = {"src": args.src, "frac": args.frac, "span": args.span, "span_frac": args.span_frac, "kept": kept, "dropped": dropped,
           "dropped_by_task": {t: [n, per_task[t]] for t, n in drop_task.most_common()}}
    out.with_suffix(".record.json").write_text(json.dumps(rec, indent=2))
    print(f"{out}: kept {kept:,}, dropped {dropped:,} requests")
    for t, n in drop_task.most_common(40):
        print(f"  {t}: {n}/{per_task[t]}")


if __name__ == "__main__":
    main()
