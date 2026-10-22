#!/usr/bin/env python3
"""Out-of-distribution extraction eval on test-only benchmarks that no training source of ours uses:
the MRQA out-of-domain test sets (BioASQ, RelationExtraction, TextbookQA, RACE; DROP and DuoRC are left out, we train
on them) as ``text`` fields, and FinEntity (eval-only by our data rules) as a ``list`` field of entities. Same backends
and scores as eval_jev.py; written to {out}/ood_{backend}.json.

  python3 recipes/extraction/eval_ood.py tag:runs/extraction/final qa:deepset/deberta-v3-base-squad2 --out runs/extraction/eval
"""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from eval_jev import load_backend, score

MRQA = ("BioASQ", "RelationExtraction", "TextbookQA", "RACE")


def rows(per_source, seed=0):
    rng, out = random.Random(seed), []
    mrqa = pq.read_table(hf_hub_download("mrqa-workshop/mrqa", "plain_text/test-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
    for subset in MRQA:
        part = [r for r in mrqa if r["subset"] == subset]
        for r in rng.sample(part, min(per_source, len(part))):
            out.append({"source": f"mrqa/{subset}", "state": r["context"],
                        "fields": [{"name": "answer_1", "type": "text", "instruction": r["question"]}],
                        "answers": [{"values": [{"text": t} for t in r["answers"]]}]})
    fin = json.load(open(hf_hub_download("yixuantt/FinEntity", "FinEntity.json", repo_type="dataset")))
    for r in rng.sample(fin, min(per_source, len(fin))):
        out.append({"source": "finentity", "state": r["content"],
                    "fields": [{"name": "company", "type": "list", "instruction": "Every company mentioned in the text, as written."}],
                    "answers": [{"values": [{"text": a["value"]} for a in r["annotations"]]}]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backends", nargs="+", help="tag:<model>, packed:<model>, qa:<model> or gliner:<model>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_source", type=int, default=500)
    ap.add_argument("--max_length", type=int)
    a = ap.parse_args()
    data = rows(a.per_source)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for backend in a.backends:
        predict = load_backend(backend, a.max_length)
        report = score(data, [predict(r["state"], r["fields"]) for r in data])
        summary = {s: round(float(r.get("text_f1", r.get("list_f1"))), 4) for s, r in report.items()}
        summary["mean"] = round(float(np.mean(list(summary.values()))), 4)
        tag = re.sub(r"[^\w.-]+", "_", backend)
        (out / f"ood_{tag}.json").write_text(json.dumps({"backend": backend, "rows": len(data), "summary": summary, "sources": report}, indent=1))
        print(backend, summary, flush=True)


if __name__ == "__main__":
    main()
