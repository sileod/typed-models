#!/usr/bin/env python3
"""Clairic-style extraction eval on the ``extraction`` test split of tasksource/tasksource-jev-typed-decisions.

Each row is a text and typed fields; ``text`` fields take one span or null, ``list`` fields every span of a kind.
Backends:
  qa:<model>      any AutoModelForQuestionAnswering; the field instruction is the question, the context is read in
                  overlapping windows. ``text``: best span unless the null score wins; ``list``: every
                  non-overlapping span scoring above null.
  tag:<model>     a query-conditioned O / B / I tagger (tasknet SpanTagging export, e.g. train.py);
                  ``text`` keeps the most confident span, ``list`` every span.
  packed:<model>  every field of a text in one pass (train_packed.py); decoded as ``tag``.
  gliner:<model>  GLiNER with the field name as label (the instruction for numbered QA fields); ``text`` keeps
                  the top span. Needs the gliner package.
Scores per source: ``text`` fields exact match and SQuAD F1 (null counts as an answer), ``list`` fields micro F1
over normalized values. Written to {out}/{backend}.json.

  python3 recipes/extraction/eval_jev.py tag:runs/extraction/final qa:deepset/deberta-v3-base-squad2 --out runs/extraction/eval
"""
import argparse
import collections
import json
import os
import re
import string
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

DATASET, REVISION = "tasksource/tasksource-jev-typed-decisions", "v2026.10.20.7"


def normalize(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(w for w in text.split() if w not in ("a", "an", "the"))


def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = sum((collections.Counter(p) & collections.Counter(g)).values())
    if not p or not g:
        return float(p == g)
    return 0.0 if not common else 2 * common / (len(p) + len(g))


def windows(tok, question, context, max_length, stride):
    """Overlapping windows over the context; long questions are cut to half a window, the overlap to what is left."""
    ids = tok(question, add_special_tokens=False)["input_ids"]
    if len(ids) > max_length // 2:
        question = tok.decode(ids[:max_length // 2])
    room = max_length - min(len(ids), max_length // 2) - 4
    return tok(question, context, truncation="only_second", max_length=max_length, stride=min(stride, room // 2),
               return_overflowing_tokens=True, return_offsets_mapping=True, padding=True, return_tensors="pt")


class QA:
    def __init__(self, name, max_length=None, stride=128, max_answer_tokens=30, batch=16):
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(name, trust_remote_code=True).cuda().eval().half()
        self.max_length = max_length or min(self.tok.model_max_length, 2048)
        self.stride, self.max_answer_tokens, self.batch = stride, max_answer_tokens, batch

    @torch.no_grad()
    def spans(self, question, context):
        """Candidate (score, start, end) char spans over all windows, and the best null score."""
        enc = windows(self.tok, question, context, self.max_length, self.stride)
        offsets = enc.pop("offset_mapping")
        enc.pop("overflow_to_sample_mapping", None)
        starts, ends = [], []
        for i in range(0, len(enc["input_ids"]), self.batch):
            out = self.model(**{k: v[i:i + self.batch].cuda() for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")})
            starts.append(out.start_logits.float().cpu()), ends.append(out.end_logits.float().cpu())
        starts, ends = torch.cat(starts).numpy(), torch.cat(ends).numpy()
        null, candidates = -np.inf, []
        for w in range(len(starts)):
            ctx = np.array([s == 1 for s in enc.sequence_ids(w)])
            s, e = starts[w], ends[w]
            null = max(null, s[0] + e[0])
            top = np.argsort(s * ctx - 1e9 * ~ctx)[-20:]
            top_e = np.argsort(e * ctx - 1e9 * ~ctx)[-20:]
            for a in top:
                for b in top_e:
                    if ctx[a] and ctx[b] and a <= b < a + self.max_answer_tokens:
                        candidates.append((s[a] + e[b], int(offsets[w][a][0]), int(offsets[w][b][1])))
        return sorted(candidates, reverse=True), null

    def __call__(self, text, fields):
        predictions = []
        for field in fields:
            candidates, null = self.spans(field["instruction"], text)
            chosen = []
            for score, a, b in candidates:
                if score <= null or (chosen and field["type"] == "text"):
                    break
                if all(b <= x or a >= y for x, y in chosen):
                    chosen.append((a, b))
            predictions.append([text[a:b] for a, b in chosen])
        return predictions


class Tagger:
    """``name`` is a model directory, or ``<model dir>/onnx/<file>.onnx`` to run an ONNX export on CPU."""
    def __init__(self, name, max_length=None, stride=128, batch=16):
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        onnx = name.endswith(".onnx")
        self.tok = AutoTokenizer.from_pretrained(os.path.dirname(os.path.dirname(name)) if onnx else name, trust_remote_code=True)
        if onnx:
            import onnxruntime
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = int(os.environ.get("ORT_THREADS", 0))  # 0: all cores
            session = onnxruntime.InferenceSession(name, options)
            self.model = lambda input_ids, attention_mask: torch.from_numpy(
                session.run(None, {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()})[0])
        else:
            model = AutoModelForTokenClassification.from_pretrained(name, trust_remote_code=True).cuda().eval().half()
            self.model = lambda input_ids, attention_mask: model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda()).logits
        self.max_length = max_length or min(self.tok.model_max_length, 2048)
        self.stride, self.batch = stride, batch

    @torch.no_grad()
    def probs(self, question, context):
        """O / B / I probabilities of every window, and per window its (context token index, char start, char end)."""
        enc = windows(self.tok, question, context, self.max_length, self.stride)
        offsets = enc.pop("offset_mapping")
        probs = torch.cat([self.model(enc["input_ids"][i:i + self.batch], enc["attention_mask"][i:i + self.batch]).float().softmax(-1).cpu()
                           for i in range(0, len(enc["input_ids"]), self.batch)]).numpy()
        tokens = [[(t, int(offsets[w][t][0]), int(offsets[w][t][1])) for t, s in enumerate(enc.sequence_ids(w)) if s == 1]
                  for w in range(len(probs))]
        return probs, tokens

    def spans(self, question, context, threshold=None):
        """(score, start, end) character spans tagged in any window; score = the B probability of the first token.
        With ``threshold``, a token is in a span when P(O) < threshold instead of when B or I wins the argmax."""
        return decode(*self.probs(question, context), threshold)


    def many_spans(self, questions, context, threshold=None):
        return [self.spans(q, context, threshold) for q in questions]

    def __call__(self, text, fields):
        return choose(text, fields, self.many_spans([f["instruction"] for f in fields], text))


def choose(text, fields, spans):
    """Per field, its non-overlapping tagged spans as text; ``text`` fields keep the most confident one."""
    predictions = []
    for field, candidates in zip(fields, spans):
        chosen = []
        for _, a, b in candidates:
            if all(b <= x or a >= y for x, y in chosen):
                chosen.append((a, b))
        predictions.append([text[a:b] for a, b in (chosen[:1] if field["type"] == "text" else chosen)])
    return predictions


def decode(probs, tokens, threshold=None):
    found = {}
    for w in range(len(probs)):
        p = probs[w]
        tags = p.argmax(-1) if threshold is None else np.where(p[:, 0] < threshold, np.where(p[:, 1] >= p[:, 2], 1, 2), 0)
        start = None
        for item in tokens[w] + [None]:
            t = None if item is None else item[0]
            if start is not None and (t is None or tags[t] != 2):
                found[(start[1], last[2])] = max(found.get((start[1], last[2]), 0), float(p[start[0]][1:].sum()))
                start = None
            if t is not None and tags[t] in (1, 2) and start is None:
                start = item
            last = item
    return sorted(((score, a, b) for (a, b), score in found.items()), reverse=True)


class Gliner:
    def __init__(self, name, threshold=0.5):
        from gliner import GLiNER
        self.model, self.threshold = GLiNER.from_pretrained(name).cuda().eval(), threshold

    def __call__(self, text, fields):
        labels = [f["instruction"] if re.fullmatch(r"answer_\d+", f["name"]) else f["name"].replace("_", " ") for f in fields]
        entities = self.model.predict_entities(text, list(dict.fromkeys(labels)), threshold=self.threshold)
        predictions = []
        for field, label in zip(fields, labels):
            found = sorted((e for e in entities if e["label"] == label), key=lambda e: -e["score"])
            predictions.append([e["text"] for e in (found[:1] if field["type"] == "text" else found)])
        return predictions


def load_backend(backend, max_length=None):
    kind, name = backend.split(":", 1)
    if kind == "gliner":
        return Gliner(name)
    if kind == "packed":
        from train_packed import Packed
        return Packed(name, max_length)
    return {"qa": QA, "tag": Tagger}[kind](name, max_length)


def score(rows, predictions):
    """Per source: text-field exact match / F1, list-field micro precision / recall / F1."""
    by_source = collections.defaultdict(lambda: collections.defaultdict(list))
    for row, preds in zip(rows, predictions):
        s = by_source[row["source"]]
        for field, answer, pred in zip(row["fields"], row["answers"], preds):
            gold = [v["text"] for v in answer["values"]]
            if field["type"] == "text":
                s["em"].append(float(normalize(pred[0]) in map(normalize, gold)) if pred and gold else float(not pred and not gold))
                s["f1"].append(max(token_f1(pred[0], g) for g in gold) if pred and gold else float(not pred and not gold))
            else:
                p, g = set(map(normalize, pred)), set(map(normalize, gold))
                s["tp"].append(len(p & g)), s["n_pred"].append(len(p)), s["n_gold"].append(len(g))
    report = {}
    for source, s in sorted(by_source.items()):
        r = {}
        if s["em"]:
            r.update(text_em=np.mean(s["em"]), text_f1=np.mean(s["f1"]), text_fields=len(s["em"]))
        if s["tp"]:
            tp, n_pred, n_gold = sum(s["tp"]), sum(s["n_pred"]), sum(s["n_gold"])
            r.update(list_f1=2 * tp / (n_pred + n_gold) if n_pred + n_gold else 1.0, list_fields=len(s["tp"]))
        report[source] = r
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backends", nargs="+", help="qa:<model>, tag:<model>, packed:<model> or gliner:<model>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_source", type=int, default=100, help="test rows per source")
    ap.add_argument("--max_length", type=int, help="QA window (default: the model's, at most 2048)")
    a = ap.parse_args()
    test = load_dataset(DATASET, "extraction", split="test", revision=REVISION).shuffle(seed=0)
    counts, rows = collections.Counter(), []
    for row in test:  # procedural generators are our own synthetic data: not a fair benchmark
        if counts[row["source"]] < a.per_source and not row["source"].startswith("procedural/"):
            counts[row["source"]] += 1
            rows.append(row)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for backend in a.backends:
        predict = load_backend(backend, a.max_length)
        predictions = [predict(r["state"], r["fields"]) for r in rows]
        report = score(rows, predictions)
        mean = lambda key: float(np.mean([r[key] for r in report.values() if key in r]))
        summary = {"text_em": mean("text_em"), "text_f1": mean("text_f1"), "list_f1": mean("list_f1")}
        tag = re.sub(r"[^\w.-]+", "_", backend)
        (out / f"{tag}.json").write_text(json.dumps({"backend": backend, "rows": len(rows), "summary": summary,
                                                     "sources": report}, indent=1))
        print(backend, {k: round(v, 4) for k, v in summary.items()}, flush=True)


if __name__ == "__main__":
    main()
