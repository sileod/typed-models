#!/usr/bin/env python3
"""tasksource/tasksource-jev-typed-decisions train split at a pinned revision -> grouped request manifest.

Rows sharing (group_id, state) become one request: a state, its questions, options and target distributions
(noul rows' P(true) becomes [1 - p, p]). Then firewall.py and shape.py.

  python3 recipes/decider/build_jev.py --revision 9fe3dd87bdfa0f016cef38234c4835b797f7cbb2 --out data/jev_full.jsonl
"""

import argparse
import collections
import glob
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

REPO = "tasksource/tasksource-jev-typed-decisions"


def clean(t):
    t = str(t or "").strip()
    return t[4:].strip() if t.startswith(("[D] ", "[Q] ")) else t


def noul_target(options, target):
    t = list(target)
    if len(t) == 1 and len(options) == 2:
        p = min(1.0, max(0.0, float(t[0])))
        return [1.0 - p, p]
    if len(t) != len(options):
        raise ValueError(f"target/options mismatch: {options!r} {target!r}")
    return t


def request(members, group_id, state):
    options = [[clean(o) for o in m["options"]] if m["options"] else ["false", "true"] for m in members]
    return {"group_id": group_id, "state": clean(state), "questions": [clean(m["question"]) for m in members],
            "options": options, "targets": [noul_target(o, m["target"]) for o, m in zip(options, members)],
            "kinds": [m["kind"] for m in members], "source": "tasksource", "task": members[0]["source"],
            "num_questions": len(members),
            "variant": "packed_derived" if any(m["variant"] == "packed_derived" for m in members) else members[0]["variant"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", required=True)
    ap.add_argument("--config", default="data")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    local = snapshot_download(REPO, repo_type="dataset", revision=a.revision, allow_patterns=[f"{a.config}/train-*"])
    groups = collections.OrderedDict()
    files = sorted(glob.glob(f"{local}/{a.config}/train-*.parquet"))
    for f in files:
        cols = ["state", "kind", "options", "target", "question", "source", "group_id", "variant"]
        for batch in pq.ParquetFile(f).iter_batches(columns=cols, batch_size=50000):
            for r in batch.to_pylist():
                groups.setdefault((r["group_id"], r["state"]), []).append(r)
    out, per_task = Path(a.out), collections.Counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fo:
        for (gid, state), members in groups.items():
            r = request(members, gid, state)
            per_task[r["task"]] += r["num_questions"]
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    rec = {"repo": REPO, "revision": a.revision, "config": a.config, "files": [Path(f).name for f in files],
           "requests": len(groups), "questions": sum(per_task.values()), "tasks": len(per_task),
           "manifest_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "per_task_questions": dict(per_task.most_common())}
    out.with_suffix(".record.json").write_text(json.dumps(rec, indent=2))
    print(f"{out}: {len(groups):,} requests, {rec['questions']:,} questions, {len(per_task)} tasks")


if __name__ == "__main__":
    main()
