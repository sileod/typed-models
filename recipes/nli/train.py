#!/usr/bin/env python3
"""Multi-task NLI encoder on tasksource (the recipe of tasksource/ModernBERT-base-nli), as a script.

All tasksource classification tasks whose id matches the NLI keyword list are trained jointly with tasknet
(shared encoder, one head per label set, task embedding added to the first token). The loaded and skipped
tasks are written to {out}/tasks.json and the per-task test accuracy to {out}/test_results.csv.

  python3 recipes/nli/train.py --model answerdotai/ModernBERT-base --out runs/nli/modernbert_base
  python3 recipes/nli/train.py --model dmis-lab/UltraBERT-150M --lr 2e-5 --out runs/nli/ultrabert
  python3 recipes/nli/train.py ... --limit_tasks 6 --max_steps 50   # smoke
  python3 recipes/nli/train.py ... --push tasksource/ModernBERT-base-nli

--label_mc K trains tasksource/zero-shot-label-nli the way the zero-shot pipeline uses it: each premise with up to K
"This example is X." candidates of its task, softmax over their entailment logits (the NLI head's entailment row,
shared with every 3-way NLI task), instead of pointwise pairs. The other NLI tasks stay pointwise.
"""

import argparse
import json
import os
from importlib.metadata import version
import re
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_WATCH", "false")

import datasets  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import tasknet as tn  # noqa: E402
import torch  # noqa: E402
import tasksource  # noqa: E402
from tasksource import list_tasks, load_task  # noqa: E402
from transformers.modeling_outputs import SequenceClassifierOutput  # noqa: E402

from typed_encoders.remote import register_remote  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

NLI = ["nli", "entailment", "rte", "fld", "scone", "rule", "logic", "nope", "cycic_classification",
       "seahorse_summarization", "puzzte", "equate", "nlgraph", "natural-language-satisfiability", "boolq",
       "mindgames", "folio", "mpe", "dnc", "recast", "fracas", "hans", "scitail", "sick/label", "super_glue/cb",
       "glue", "paws/labeled_final", "dynabench", "contrastive", "cladder", "feasibilityQA", "pdtb", "dynahate"]
MC = ["race/high", "race/middle", "race-c", "reclor", "logiqa", "cosmos_qa", "hellaswag",
      "HelpSteer3/preference", "HelpSteer3/edit_quality"]  # --mc; quality is too long for max_length 2048
MERGE = ["mmlu", "babi_nli", "blimp", "/prag", "imppres/presupposition", "gen_debiased_nli"]


MNLI_ROWS = 50_000  # --mnli_rows


def size(preprocessing_name):
    """Rows per task: 50k by default, small for probing suites, large for the core NLI sets."""
    s = dict(max_rows=50_000, max_rows_eval=180, oversampling=2)
    if preprocessing_name == "glue___mnli":
        return dict(s, max_rows=MNLI_ROWS)
    for regex, v in {
        "bigbench": dict(max_rows=5_000),
        "(linguisticprobing|imppres)": dict(max_rows=2_000, oversampling=1),
        "(robust_nli|gen_debiased_nli)": dict(max_rows=2_000, oversampling=1),
        "universal_dependencies": dict(max_rows=20_000, oversampling=1),
        "(label_nli|dpo|dataset_train_nli)": dict(max_rows=600_000, oversampling=1),
        "(anli|WANLI|dynasent__|xnli|FOL-nli)": dict(max_rows=250_000, oversampling=3),
        "(doc-nli|ConTRoL|mctest-nli|summ)": dict(max_rows=500_000, oversampling=3),
    }.items():
        if re.search(regex, preprocessing_name):
            s.update(v)
            break
    return s


def label_mc_tasks(k_max, max_rows, seed=0):
    """tasksource/zero-shot-label-nli as multiple choice: gold label + negatives from the same task's label set
    (never another label entailed by that premise). One task per choice count, since tasknet needs fixed choices."""
    rng, buckets = np.random.default_rng(seed), {}
    for split, d in datasets.load_dataset("tasksource/zero-shot-label-nli").items():
        df = d.select_columns(["premise", "hypothesis", "labels", "task"]).to_pandas()
        vocab = df.groupby("task").hypothesis.unique()
        gold = df[df.labels == 0].groupby(["task", "premise"]).hypothesis.agg(list)
        for (task, premise), golds in gold.items():
            neg = np.setdiff1d(vocab[task], golds)
            if not len(neg):
                continue
            choices = [golds[rng.integers(len(golds))], *rng.choice(neg, min(k_max - 1, len(neg)), replace=False)]
            order = rng.permutation(len(choices))
            row = {"inputs": premise, "labels": int(np.argmin(order)), **{f"choice{i}": choices[j] for i, j in enumerate(order)}}
            buckets.setdefault(len(choices), {}).setdefault(split, []).append(row)
    tasks = []
    for k, splits in sorted(buckets.items()):
        dd = datasets.DatasetDict({sp: datasets.Dataset.from_list(rows) for sp, rows in splits.items()})
        tasks.append(EntailmentMC(dd, name=f"label_nli_mc{k}", max_rows=max_rows // k, max_rows_eval=600, oversampling=1))
    return tasks


class EntailmentMC(tn.MultipleChoice):
    """Built as a 3-way sequence classifier so the head can be swapped for the shared NLI head (see EntailmentScorer)."""
    task_type = "SequenceClassification"
    num_labels: int = 3


class EntailmentScorer(torch.nn.Module):
    """Multiple-choice wrapper of a 3-way NLI model: choice score = entailment logit, cross-entropy over choices."""
    def __init__(self, model):
        super().__init__()
        self.model, self.config = model, model.config

    @property
    def classifier(self):
        return self.model.classifier

    def forward(self, input_ids, attention_mask, labels=None, **kw):
        b, c, n = input_ids.shape
        logits = self.model(input_ids=input_ids.view(-1, n), attention_mask=attention_mask.view(-1, n)).logits
        logits = logits[:, 0].view(b, c)
        loss = torch.nn.functional.cross_entropy(logits.float(), labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)


def tie_entailment(model, tasks):
    """Score EntailmentMC tasks exactly as the exported model will: task 0's head (save_model exports task 0's model,
    adapters only swap the final classifier) and the classifier shared by the 3-way entailment/neutral/contradiction tasks."""
    nli = next(m for key, m in model.models.items() if isinstance(key, tuple) and key[:1] == ("entailment",) and len(key) == 3)
    first = model.task_models_list[0]
    for i, t in enumerate(tasks):
        if isinstance(t, EntailmentMC):
            m = model.task_models_list[i]
            for name, child in first.named_children():
                if name not in (model.get_encoder_attr_name(first), "auto", "classifier"):
                    setattr(m, name, child)
            m.classifier = nli.classifier
            model.task_models_list[i] = EntailmentScorer(m)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=6, help="per-device batch; gradient accumulation fills ~28")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42, help="data order; change it when continuing from a trained --model")
    ap.add_argument("--eval_ratio", type=float, default=0.1, help="validate every task each this fraction of the run")
    ap.add_argument("--save_ratio", type=float, default=0.1, help="checkpoint each this fraction of the run")
    ap.add_argument("--limit_tasks", type=int, help="smoke: first N matching tasks")
    ap.add_argument("--label_mc", type=int, default=0, help="K>1: zero-shot-label-nli as K-way multiple choice")
    ap.add_argument("--mc", action="store_true", help="also train the MC tasks (4 options, own heads)")
    ap.add_argument("--mnli_rows", type=int, default=50_000, help="MNLI training rows (other tasks default to 50k)")
    ap.add_argument("--drop", default="", help="comma-separated task ids to leave out, e.g. dataset_train_nli")
    ap.add_argument("--attn", help="attn_implementation, e.g. flash_attention_2 (default: the model's, usually sdpa)")
    ap.add_argument("--push", help="hub repo to push the trained model to")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    global MNLI_ROWS
    MNLI_ROWS = a.mnli_rows
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    datasets.logging.set_verbosity_error()
    datasets.utils.logging.disable_progress_bar()
    tasksource.preprocess.MAX_MC_OPTIONS = 4 if a.mc else 2
    register_remote(a.model)

    df = list_tasks(multilingual="")
    nli = ~df.task_type.isin(["TokenClassification", "MultipleChoice"]) & (df.dataset_name != "tasksource/mmlu")
    nli &= df.id.map(lambda x: "merge" not in x.lower() and any(k.lower() in x.lower() for k in NLI))
    df = df[nli | (df.id.isin(MC) if a.mc else False)]
    df = df[~df.id.isin(a.drop.split(","))]
    if a.label_mc:
        df = df[df.preprocessing_name != "label_nli"]
    if a.limit_tasks:  # smoke: the first N tasks, plus the MC tasks with --mc
        df = pd.concat([df[~df.id.isin(MC)].head(a.limit_tasks), df[df.id.isin(MC)]])
    tasks, skipped = [], {}
    for i, x in enumerate(df.itertuples()):
        print(f"{len(tasks)} loaded / {i + 1} of {len(df)}", end="\r", flush=True)
        try:
            task = getattr(tn, x.task_type)(load_task(x.id), name=x.id, **size(x.preprocessing_name))
            if task.check():
                tasks.append(task)
            else:
                skipped[x.id] = "check() failed"
        except Exception as e:  # one broken source must not stop a multi-day run; it is logged instead
            skipped[x.id] = f"{type(e).__name__}: {e}"[:300]
    tasks = tn.merge_tasks(tasks, names=MERGE)
    if a.label_mc:
        tasks += label_mc_tasks(a.label_mc, size("label_nli")["max_rows"])
    (out / "tasks.json").write_text(json.dumps({"model": a.model, "tasksource": version("tasksource"),
                                                "loaded": [t.name for t in tasks], "skipped": skipped}, indent=1))
    print(f"\n{len(tasks)} tasks after merging, {len(skipped)} skipped -> {out / 'tasks.json'}", flush=True)

    large = "large" in a.model.lower()
    args = dict(model_name=a.model, max_length=a.max_length, max_sequence_length=a.max_length, learning_rate=a.lr,
                num_train_epochs=a.epochs, max_steps=a.max_steps, output_dir=str(out / "checkpoints"),
                per_device_train_batch_size=a.batch, gradient_accumulation_steps=max(1, 28 // a.batch),
                eval_strategy="steps", eval_steps=a.eval_ratio, save_steps=a.save_ratio, logging_strategy="steps", logging_steps=80, warmup_ratio=0.07,
                fp16=large, bf16=not large, drop_probability=0.2, batched=False, adam_epsilon=1e-6,
                remove_overflowed_examples=True, attn_implementation=a.attn, share_mc=True, weight_decay=1e-6, seed=a.seed,
                report_to=["wandb"] if a.wandb else [])
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = tn.Model(tasks, args)
    tie_entailment(model, tasks)
    for m in [model.shared_encoder, *model.task_models_list]:
        m.config.update(dict(max_position_embeddings=a.max_length))
    trainer = tn.Trainer(model, tasks, args, tokenizer=tok)
    trainer.train()
    trainer.save_model(str(out / "final"))  # before the test pass: a metric failure must not lose the model

    res = pd.DataFrame(trainer.evaluate("test"))
    res.to_csv(out / "test_results.csv", index=False)
    print(res[["test_name", "test_accuracy"]].round(3).to_markdown(index=False))
    print(f"mean test accuracy {res.test_accuracy.mean():.4f} over {len(res)} tasks")
    if a.push:
        trainer.push_to_hub(a.push)


if __name__ == "__main__":
    main()
