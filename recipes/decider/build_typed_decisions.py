#!/usr/bin/env python3
"""LocalLLaMA/typed-decisions `train` (Apache-2.0) -> grouped manifest, one request per case with its five questions and
the gold distributions as soft targets. `score` questions become choices over their criteria, as the board scores them.

  python3 recipes/decider/build_typed_decisions.py data/typed_decisions_train.jsonl
"""

import json
import sys

import pandas as pd
from huggingface_hub import hf_hub_download

from typed_encoders.formatting import question_options, state_text

REV = "6a7927fa"
df = pd.read_parquet(hf_hub_download("LocalLLaMA/typed-decisions", "all/train-00000-of-00001.parquet", repo_type="dataset", revision=REV))
with open(sys.argv[1], "w") as f:
    for case in df.itertuples():
        qs, gold = json.loads(case.questions), json.loads(case.gold)
        r = {"group_id": f"typed-decisions:{case.id}", "state": state_text(json.loads(case.state)), "questions": [], "options": [],
             "targets": [], "kinds": [], "source": "typed-decisions", "task": f"typed-decisions/{case.workflow}"}
        for qid, q in qs.items():
            if q["type"] == "score":
                q = dict(q, type="choice", criteria={str(i): c for i, c in enumerate(q["criteria"])})
            keys, opts = question_options(q)
            p = gold[qid]["probabilities"]
            r["questions"].append(q.get("instructions", ""))
            r["options"].append(opts)
            r["targets"].append([p.get(k, 0.0) for k in keys])
            r["kinds"].append(q["type"])
        r["num_questions"] = len(r["questions"])
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(df), "cases ->", sys.argv[1])
