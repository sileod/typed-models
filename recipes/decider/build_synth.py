#!/usr/bin/env python3
"""tasksource/synthetic-typed-decisions (train split) -> grouped training manifest (same schema as the
Tasksource manifest: state, questions, options, targets, kinds). Options are rendered exactly as the
engines render them (eval.formatting); score questions become ordered choices over their legend.

  python3 recipes/decider/build_synth.py --out data/synth_train.jsonl
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from typed_models.formatting import question_options, state_text

REPO, REVISION = "tasksource/synthetic-typed-decisions", "7ec5a61d5db158f637c30adeb6e66660f1dfee87"


def convert(row):
    qs, ans = json.loads(row.questions), json.loads(row.answers)
    out = {"group_id": f"synth:{row.id}", "state": state_text(row.state), "questions": [], "options": [],
           "targets": [], "kinds": [], "source": "synthetic-typed-decisions", "task": f"synth:{row.workflow}"}
    for k, q in qs.items():
        a = ans.get(k)
        if a is None:
            continue
        if q["type"] == "score":
            labels = [str(c) for c in q["criteria"]]
            opts, tgt = labels, [a["probabilities"].get(str(i), 0.0) for i in range(len(labels))]
        elif q["type"] == "noul":
            _, opts = question_options(q)
            tgt = [1 - a["noul"], a["noul"]]
        else:
            keys, opts = question_options(q)
            tgt = [a["probabilities"].get(c, 0.0) for c in keys]
        if len(opts) < 2 or sum(tgt) <= 0:
            continue
        out["questions"].append(q.get("instructions", ""))
        out["options"].append(opts)
        out["targets"].append([t / sum(tgt) for t in tgt])
        out["kinds"].append(q["type"])
    out["num_questions"] = len(out["questions"])
    return out if out["questions"] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_parquet(hf_hub_download(REPO, f"data/{args.split}-00000-of-00001.parquet", repo_type="dataset", revision=args.revision))
    rows = [r for r in (convert(r) for r in df.itertuples()) if r]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} requests, {sum(r['num_questions'] for r in rows)} decisions -> {args.out}")


if __name__ == "__main__":
    main()
