#!/usr/bin/env python3
"""Reshape a grouped training manifest along the mixing axes we ablate: per-task question cap,
question-kind ratios (choice / binary / noul / score), and option cardinality.

Question kind: "binary" is a choice question with 2 options, "choice" one with more, "noul" and
"score" (ordinal choice) as labelled by the source. Cardinality bins: 2, 3-5, 6-12, 13+.

Steps: (1) drop the kinds in --drop_kinds from every request; (2) per-task cap of --cap_q questions
(seeded request order, whole requests); (3) --kind_target: question-level thinning, each question
kept with probability (target/observed share of its kind), scaled so the most under-represented kind
keeps everything (requests share a state, so kinds co-occur and request-level sampling cannot reach
the targets); (4) --card_target: draw --n requests without replacement with probability proportional
to the mean over their questions of target/observed cardinality share.

  python3 recipes/decider/shape.py data/jev_cfw.jsonl data/jev_bal.jsonl --cap_q 600
"""

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np


def kind(k, opts):
    if k == "choice" or k is None:
        return "binary" if len(opts) == 2 else "choice"
    return k


def card(opts):
    n = len(opts)
    return "2" if n == 2 else "3-5" if n <= 5 else "6-12" if n <= 12 else "13+"


def parse(spec):
    return {k: float(v) for k, v in (x.split("=") for x in spec.split(","))} if spec else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--cap_q", type=int, default=600)
    ap.add_argument("--exclude_source", nargs="*", default=[])
    ap.add_argument("--only_source", nargs="*", default=[])
    ap.add_argument("--drop_kinds", nargs="*", default=[])
    ap.add_argument("--kind_target", help="e.g. choice=0.55,binary=0.41,noul=0.04")
    ap.add_argument("--card_target", help="e.g. 2=0.4,3-5=0.4,6-12=0.2")
    ap.add_argument("--n", type=int, help="requests to draw when targets are given (default: all eligible / 2)")
    ap.add_argument("--packed_share", type=float,
                    help="target share of packed_derived requests (jev-typed-decisions 'variant'); default: as built")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    by_task = collections.defaultdict(list)
    for line in open(args.src, encoding="utf-8"):
        r = json.loads(line)
        src = r.get("source", "")
        if src in args.exclude_source or (args.only_source and src not in args.only_source):
            continue
        ks = r.get("kinds") or ["choice"] * len(r["questions"])
        keep = [i for i, (k, o) in enumerate(zip(ks, r["options"])) if kind(k, o) not in args.drop_kinds]
        if not keep:
            continue
        if len(keep) < len(r["questions"]):
            r = dict(r, **{f: [r[f][i] for i in keep] for f in ("questions", "options", "targets", "kinds") if f in r})
        by_task[r.get("task") or src].append(r)

    pool = []
    for task in sorted(by_task):
        reqs = by_task[task]
        rng.shuffle(reqs)
        nq = 0
        for r in reqs:
            if nq >= args.cap_q:
                break
            pool.append(r)
            nq += len(r["questions"])

    if args.packed_share is not None:
        packed = [r for r in pool if r.get("variant") == "packed_derived"]
        other = [r for r in pool if r.get("variant") != "packed_derived"]
        n_packed = min(len(packed), round(args.packed_share / max(1e-9, 1 - args.packed_share) * len(other)))
        rng.shuffle(packed)
        pool = other + packed[:n_packed]

    def qfeats(r):
        ks = r.get("kinds") or ["choice"] * len(r["questions"])
        return [(kind(k, o), card(o)) for k, o in zip(ks, r["options"])]

    kt, ct = parse(args.kind_target), parse(args.card_target)
    if kt:
        obs = collections.Counter(k for r in pool for k, _ in qfeats(r))
        tot = sum(obs.values())
        ratio = {k: kt.get(k, 0.0) / (obs[k] / tot) for k in obs}
        keep_p = {k: v / max(ratio.values()) for k, v in ratio.items()}
        thinned = []
        for r in pool:
            keep = [i for i, (k, _) in enumerate(qfeats(r)) if rng.random() < keep_p[k]]
            if keep:
                thinned.append(dict(r, **{f: [r[f][i] for i in keep] for f in ("questions", "options", "targets", "kinds") if f in r}))
        pool = thinned
        kt = None
    if ct:
        obs_k, obs_c = collections.Counter(), collections.Counter()
        for r in pool:
            for k, c in qfeats(r):
                obs_k[k] += 1
                obs_c[c] += 1
        tk, tc = sum(obs_k.values()), sum(obs_c.values())
        w = []
        for r in pool:
            ws = []
            for k, c in qfeats(r):
                x = 1.0
                if kt:
                    x *= kt.get(k, 0.0) / (obs_k[k] / tk)
                if ct:
                    x *= ct.get(c, 0.0) / (obs_c[c] / tc)
                ws.append(x)
            w.append(sum(ws) / len(ws))
        w = np.array(w)
        n = min(args.n or len(pool) // 2, int((w > 0).sum()))
        idx = np.random.default_rng(args.seed).choice(len(pool), size=n, replace=False, p=w / w.sum())
        pool = [pool[i] for i in idx]
    rng.shuffle(pool)

    kinds, cards, tasks = collections.Counter(), collections.Counter(), collections.Counter()
    for r in pool:
        for k, c in qfeats(r):
            kinds[k] += 1
            cards[c] += 1
        tasks[r.get("task")] += len(r["questions"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tq = sum(kinds.values())
    packed_share = sum(r.get("variant") == "packed_derived" for r in pool) / max(1, len(pool))
    rec = {k: v for k, v in vars(args).items()} | {
        "requests": len(pool), "questions": tq, "tasks": len(tasks),
        "kinds": {k: round(v / tq, 3) for k, v in kinds.most_common()},
        "cardinality": {k: round(v / tq, 3) for k, v in cards.most_common()},
        "packed_share": round(packed_share, 4),
        "top10_task_share": round(sum(v for _, v in tasks.most_common(10)) / max(1, tq), 4)}
    Path(args.out).with_suffix(".record.json").write_text(json.dumps(rec, indent=2))
    print(f"{args.out}: {len(pool)} requests, {tq} questions, {len(tasks)} tasks; packed {packed_share:.1%}; kinds {rec['kinds']}; cards {rec['cardinality']}")


if __name__ == "__main__":
    main()
