"""Training-data rules applied whenever a manifest is loaded.

EXCLUDED_TASKS: benchmark-only datasets with no official train split, whose pool the Decision Index samples test
items from. cladder: CLadder v1 is a benchmark release; the suite drew its test items from the same pool.
"""

import json

EXCLUDED_TASKS = {"cladder"}


def allowed(req: dict) -> bool:
    names = (str(req.get("task", "")) + "|" + str(req.get("source", ""))).lower()
    return not any(t in names for t in EXCLUDED_TASKS)


def load_manifest(path: str) -> list:
    reqs = [json.loads(line) for line in open(path, encoding="utf-8")]
    kept = [r for r in reqs if allowed(r)]
    if len(kept) != len(reqs):
        print(f"rules: dropped {len(reqs) - len(kept)} requests of {sorted(EXCLUDED_TASKS)} from {path}", flush=True)
    return kept
