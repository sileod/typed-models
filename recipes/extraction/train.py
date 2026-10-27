#!/usr/bin/env python3
"""Multi-task extraction encoder on tasksource, trained with tasknet as in recipes/nli/train.py.

Task 0, the exported model, is Clairic-style extraction: the ``extraction`` config of
tasksource/tasksource-jev-typed-decisions, one (field instruction, text) pair per field, context tokens tagged
O / B / I (one span, a list, or none). Auxiliary tasks: extractive question answering (``list_tasks(extractive=True)``)
and span tagging (the TokenClassification tasks, minus syntax); their heads are in {out}/final-adapter.
Per-task test F1 goes to {out}/test_results.csv.

  python3 recipes/extraction/train.py --model dmis-lab/UltraBERT-150M --max_steps 40000 --out runs/extraction
  python3 recipes/extraction/train.py ... --limit_tasks 3 --max_steps 60 --save_ratio 0.99 --eval_ratio 0.99  # smoke
"""

import argparse
import json
import os
import re
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_WATCH", "false")

import datasets  # noqa: E402
import pandas as pd  # noqa: E402
import tasknet as tn  # noqa: E402
from tasksource import list_tasks, load_task  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from typed_encoders.remote import register_remote  # noqa: E402

SYNTAX = ("pos_tags", "chunk_tags", "deprel", "conll2000")  # token tasks that tag grammar, not spans to extract
JEV, JEV_REVISION = "tasksource/tasksource-jev-typed-decisions", "v2026.10.20.7"


COUNT = re.compile(r"(how many|number of)\b", re.I)
NOT_COUNT = re.compile(r"\b(after|before|longer|shorter|more|fewer|less|than|between|difference|percent|total|combined|in all|"
                       r"altogether|yards|years|days|months|hours|minutes|points|average|per)\b", re.I)  # differences, sums, lookups
EVERY = re.compile(r"^Every (.+) mentioned in the text, as written\.$")


def how_many(pairs, n, seed=0):
    """Up to ``n`` list pairs ("Every X mentioned in the text, as written.") asked as "How many X are mentioned in the
    text?", with the same spans and their count."""
    pairs = pairs.filter(lambda q: bool(EVERY.match(q)), input_columns="question").shuffle(seed=seed)
    pairs = pairs.select(range(min(n, len(pairs))))
    return pairs.map(lambda r: {"question": EVERY.sub(r"How many \1 are mentioned in the text?", r["question"]), "count": len(r["spans"])})


def typed_extraction_pairs(counts=False):
    """(instruction, text, spans) pairs from the `train` split of LocalLLaMA/typed-extraction: null fields with no span,
    fields whose answer is written in the text with its spans (given, or the only occurrence of the value). Computed,
    relative and reworded answers have no span to tag and are left out, except, with ``counts``, "how many" counts
    (no spans, count only)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    rows = pq.read_table(hf_hub_download("LocalLLaMA/typed-extraction", "data/train.parquet", repo_type="dataset")).to_pylist()
    out = {"question": [], "context": [], "spans": [], "count": []}
    for r in rows:
        fields, gold, text = json.loads(r["fields"]), json.loads(r["gold"]), r["state"]
        for name, g in gold.items():
            spans, ins = g["spans"], fields[name]["instructions"]
            if counts and g["source"] == "computed" and COUNT.match(ins) and not NOT_COUNT.search(ins) \
                    and isinstance(g["value"], (int, float)) and 0 < g["value"] <= 30 and float(g["value"]).is_integer():
                out["question"].append(ins), out["context"].append(text), out["spans"].append([]), out["count"].append(int(g["value"]))
                continue
            if g["value"] not in (None, []) and not spans:
                values = g["value"] if isinstance(g["value"], list) else [g["value"]]
                found = [[m.start(), m.end()] for v in map(str, values) for m in re.finditer(rf"(?<!\w){re.escape(v)}(?!\w)", text)]
                if len(found) != len(values):
                    continue
                spans = found
            out["question"].append(fields[name]["instructions"])
            out["context"].append(text)
            out["spans"].append([list(x) for x in spans])
            out["count"].append(-1)
    return datasets.Dataset.from_dict(out)


def jev_extraction(rows, exclude=(), seed=0, extra=None, how_many_rows=0):
    """(instruction, text, character spans) pairs from the extraction config, ``rows`` sampled for training;
    sources starting with any ``exclude`` prefix are left out. ``extra`` pairs are added to the sampled train split; ``how_many_rows`` > 0 adds counting questions
    (see ``how_many``) and a ``count`` column, which tasknet trains as the expected number of spans."""
    def pairs(batch):
        out = {"question": [], "context": [], "spans": []}
        for text, fields, answers in zip(batch["state"], batch["fields"], batch["answers"]):
            for field, answer in zip(fields, answers):
                out["question"].append(field["instruction"])
                out["context"].append(text)
                out["spans"].append([[v["start"], v["end"]] for v in answer["values"]])
        return out
    dataset = datasets.load_dataset(JEV, "extraction", revision=JEV_REVISION)
    if exclude:
        dataset = dataset.filter(lambda source: not source.startswith(tuple(exclude)), input_columns="source")
    dataset = datasets.DatasetDict({split: d.map(pairs, batched=True, remove_columns=d.column_names).shuffle(seed=seed)
                                    for split, d in dataset.items()})
    if extra is not None or how_many_rows:
        train = dataset["train"].select(range(min(rows, len(dataset["train"]))))
        parts = [train]
        if how_many_rows:
            parts = [train.add_column("count", [-1] * len(train)), how_many(train, how_many_rows, seed)]
        if extra is not None:
            parts.append(extra if how_many_rows else extra.remove_columns("count"))
        dataset["train"] = datasets.concatenate_datasets([x.cast(parts[0].features) for x in parts]).shuffle(seed=seed)
        rows = len(dataset["train"])
    return tn.SpanTagging(dataset, name="jev-extraction", max_rows=rows, max_rows_eval=600)


def with_eval_splits(dataset, n=300):
    """tasknet validates and tests every task: carve missing validation/test splits from train."""
    for split in ("validation", "test"):
        if split not in dataset:
            parts = dataset["train"].train_test_split(test_size=min(n, len(dataset["train"]) // 10), seed=0)
            dataset["train"], dataset[split] = parts["train"], parts["test"]
    return dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dmis-lab/UltraBERT-150M")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=6, help="per-device batch; gradient accumulation fills ~28")
    ap.add_argument("--max_steps", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_ratio", type=float, default=0.1, help="validate every task each this fraction of the run")
    ap.add_argument("--save_ratio", type=float, default=0.1, help="checkpoint each this fraction of the run")
    ap.add_argument("--limit_tasks", type=int, help="smoke: first N tasks of each kind")
    ap.add_argument("--drop", default="", help="comma-separated task ids to leave out")
    ap.add_argument("--jev_rows", type=int, default=400_000, help="(field, text) training pairs of the exported task; 0 leaves it out (auxiliary tasks only)")
    ap.add_argument("--aux_rows", type=int, default=5_000, help="training rows per auxiliary task")
    ap.add_argument("--typed_extraction", action="store_true", help="add the train split of LocalLLaMA/typed-extraction (span-groundable fields)")
    ap.add_argument("--how_many", type=int, default=0, help="counting: this many 'how many X' paraphrases of list pairs, "
                    "plus the count-only 'how many' answers of typed-extraction (with --typed_extraction)")
    ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing: less memory, ~30%% slower")
    ap.add_argument("--exclude_sources", default="", help="comma-separated source prefixes left out of the extraction config, e.g. procedural/")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    datasets.logging.set_verbosity_error()
    datasets.utils.logging.disable_progress_bar()
    register_remote(a.model)

    qa = list_tasks(extractive=True).assign(extractive=True)
    tags = list_tasks().query("task_type == 'TokenClassification'").assign(extractive=False)
    tags = tags[~tags.id.map(lambda x: any(s in x for s in SYNTAX))]
    df = pd.concat([qa.head(a.limit_tasks), tags.head(a.limit_tasks)])
    df = df[~df.id.isin(a.drop.split(","))]
    extra = typed_extraction_pairs(counts=a.how_many > 0) if a.typed_extraction else None
    tasks, skipped = [jev_extraction(a.jev_rows, [x for x in a.exclude_sources.split(",") if x], extra=extra, how_many_rows=a.how_many)] if a.jev_rows else [], {}
    for i, x in enumerate(df.itertuples()):
        print(f"{len(tasks)} loaded / {i + 1} of {len(df)}", end="\r", flush=True)
        try:
            kind = tn.QuestionAnswering if x.extractive else tn.TokenClassification
            task = kind(with_eval_splits(load_task(x.id, extractive=x.extractive)), name=x.id, max_rows=a.aux_rows, max_rows_eval=300, oversampling=2)
            if task.check():
                tasks.append(task)
            else:
                skipped[x.id] = "check() failed"
        except Exception as e:  # one broken source must not stop a long run; it is logged instead
            skipped[x.id] = f"{type(e).__name__}: {e}"[:300]
    (out / "tasks.json").write_text(json.dumps({"model": a.model, "tasksource": version("tasksource"),
                                                "loaded": [t.name for t in tasks], "skipped": skipped}, indent=1))
    print(f"\n{len(tasks)} tasks, {len(skipped)} skipped -> {out / 'tasks.json'}", flush=True)

    args = dict(model_name=a.model, max_length=a.max_length, max_sequence_length=a.max_length, learning_rate=a.lr,
                max_steps=a.max_steps, output_dir=str(out / "checkpoints"),
                per_device_train_batch_size=a.batch, gradient_accumulation_steps=max(1, 28 // a.batch),
                eval_strategy="steps", eval_steps=a.eval_ratio, save_steps=a.save_ratio, logging_strategy="steps",
                logging_steps=80, warmup_ratio=0.07, bf16=True, drop_probability=0.2, batched=False, adam_epsilon=1e-6,
                remove_overflowed_examples=True, weight_decay=1e-6, seed=a.seed, report_to=[])
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = tn.Model(tasks, args)
    for m in [model.shared_encoder, *model.task_models_list]:
        m.config.update(dict(max_position_embeddings=a.max_length))
    if a.grad_ckpt:
        for m in model.task_models_list:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = tn.Trainer(model, tasks, args, tokenizer=tok)
    trainer.train()
    trainer.save_model(str(out / "final"))  # before the test pass: a metric failure must not lose the model
    tok.save_pretrained(str(out / "final"))

    res = pd.DataFrame(trainer.evaluate("test"))
    res.to_csv(out / "test_results.csv", index=False)
    print(res[["test_name", "test_f1"]].round(3).to_markdown(index=False))
    qa_f1 = res[res.test_name.isin(qa.id)].test_f1
    tag_f1 = res[res.test_name.isin(tags.id)].test_f1
    jev_f1 = res[res.test_name == "jev-extraction"].test_f1
    print(f"test F1: jev-extraction {jev_f1.mean():.4f}, QA mean {qa_f1.mean():.4f} ({len(qa_f1)}), "
          f"tagging mean {tag_f1.mean():.4f} ({len(tag_f1)})")


if __name__ == "__main__":
    main()
