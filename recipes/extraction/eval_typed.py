#!/usr/bin/env python3
"""Predictions for LocalLLaMA/typed-extraction `test` from a span tagger (tasknet SpanTagging export), for its score.py.

The tagger reads each field's instructions against the document and returns spans; spans become typed values:
text as written, numbers / durations parsed from the span, dates and times through dateutil (years default to the
reference date's), lists item by item. A field with no span is null. ``p_null`` = 1 - the best span's B probability.
Computed answers (totals, day counts) are out of reach of a span tagger and score as misses.

  python3 recipes/extraction/eval_typed.py runs/extraction/final runs/extraction/typed_extraction.jsonl
  python score.py <predictions.jsonl> --by-family   # score.py from the LocalLLaMA/typed-extraction dataset repository
"""
import argparse
import datetime as dt
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from dateutil import parser as dateparser
from huggingface_hub import hf_hub_download

from eval_jev import Tagger, load_backend

REPO = "LocalLLaMA/typed-extraction"
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def number(text, options):
    m = NUMBER.search(text)
    if not m:
        return None
    value = float(m.group().replace(",", ""))
    if re.match(r"\s*[kK]\b", text[m.end():]):
        value *= 1000
    elif re.match(r"\s*(?:m\b|million)", text[m.end():]):
        value *= 1_000_000
    return round(value) if options.get("integer") else value


def temporal(text, kind, reference):
    default = dt.datetime.combine(reference, dt.time()) if reference else dt.datetime(dt.date.today().year, 1, 1)
    try:
        d = dateparser.parse(text, default=default, fuzzy=True)
    except (ValueError, OverflowError):
        return None
    return {"date": d.date().isoformat(), "time": d.strftime("%H:%M"), "datetime": d.strftime("%Y-%m-%dT%H:%M")}[kind]


def typed(text, kind, options, reference):
    if kind in ("number", "duration"):
        return number(text, options)
    if kind in ("date", "time", "datetime"):
        return temporal(text, kind, reference)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--threshold", type=float, help="a token is in a span when P(O) < this (default: argmax); pick it on validation data")
    ap.add_argument("--list_threshold", type=float, help="the same for list fields (default: --threshold)")
    a = ap.parse_args()
    tagger = load_backend(f"packed:{a.model}", a.max_length) if (Path(a.model) / "head.pt").exists() else Tagger(a.model, a.max_length)
    rows = pq.read_table(hf_hub_download(REPO, "data/test.parquet", repo_type="dataset")).to_pylist()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as out:
        for i, r in enumerate(rows):
            reference = dt.date.fromisoformat(r["reference_date"]) if r["reference_date"] else None
            answers, fields = {}, json.loads(r["fields"])
            list_threshold = a.threshold if a.list_threshold is None else a.list_threshold
            found = {t: dict(zip(fields, tagger.many_spans([f["instructions"] for f in fields.values()], r["state"], t)))
                     for t in {a.threshold, list_threshold}}
            for name, f in fields.items():
                options = f.get("options") or {}
                chosen = []
                spans = found[list_threshold if f["type"] == "list" else a.threshold][name]
                for _, s, e in spans:
                    if all(e <= x or s >= y for x, y in chosen):
                        chosen.append((s, e))
                texts = [r["state"][s:e].strip().rstrip(",;:") for s, e in sorted(chosen)]
                if f["type"] == "list":
                    values = [v for v in (typed(t, options.get("items", "text"), options, reference) for t in texts) if v is not None]
                    value = list(dict.fromkeys(values))
                else:
                    best = r["state"][spans[0][1]:spans[0][2]].strip().rstrip(",;:") if spans else None
                    value = typed(best, f["type"], options, reference) if best else None
                score = spans[0][0] if spans else 0.0
                answers[name] = {"value": value, "p_null": round(1 - score, 4), "confidence": round(score, 4)}
            out.write(json.dumps({"id": r["id"], "answers": answers}) + "\n")
            if i % 100 == 0:
                print(f"{i}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
