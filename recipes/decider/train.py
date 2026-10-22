#!/usr/bin/env python3
"""Train a joint decider (typed_models.decider) on a grouped request manifest (see build_*.py and mix.py).

Each question -> K option logits -> soft cross-entropy against its target distribution. Questions with more than
--max_k options keep every positive plus sampled negatives. --task_prompts renders that share of single-question
jev-typed-decisions requests with tasksource's task-specific prompts; --nota swaps or adds a "None of the other
options" choice.

  python3 recipes/decider/train.py --manifest data/mix.jsonl --out runs/decider --steps 64000 --seed 1
"""

import argparse
import random
import re
import time
from pathlib import Path

import torch
from tasksource.jev.task_prompts import STYLES, describe_options, task_style
from transformers import get_cosine_schedule_with_warmup

from typed_models.decider import Decider, shared_logits
from typed_models.rules import load_manifest

NOTA = "None of the other options"  # the jev pipeline's position-free rewording of a trailing "none of the above"
NOTA_LIKE = re.compile(r"\b(none|neither|other|all of the above|n/a)\b", re.I)


def subsample(opts, tgt, max_k, rng):
    if len(opts) <= max_k:
        return opts, tgt
    pos = [i for i, t in enumerate(tgt) if t > 0]
    neg = [i for i, t in enumerate(tgt) if t <= 0]
    keep = sorted(pos[:max_k] + rng.sample(neg, max(0, max_k - len(pos))))
    return [opts[i] for i in keep], [tgt[i] for i in keep]


def nota(opts, tgt, p, rng):
    """With prob p: drop the gold option and append NOTA as the new gold; with prob p: append NOTA as a distractor.
    Only single-answer questions with a hard target, 3+ options and no NOTA-like option already."""
    if len(opts) < 3 or sorted(tgt) != [0.0] * (len(tgt) - 1) + [1.0] or any(NOTA_LIKE.search(o) for o in opts):
        return opts, tgt
    u = rng.random()
    if u < p:
        g = tgt.index(1.0)
        return opts[:g] + opts[g + 1:] + [NOTA], [0.0] * (len(opts) - 1) + [1.0]
    if u < 2 * p:
        return opts + [NOTA], tgt + [0.0]
    return opts, tgt


def soft_ce(logits, t):
    target = torch.tensor(t, device=logits.device) / sum(t)
    return -(target * torch.log_softmax(logits, -1)).sum()


def request_batches(qs, max_length, budget):
    """Group a step's questions by request (consecutive same state) and requests into micro-batches of about
    `budget` tokens (estimated from characters, state counted once per request)."""
    reqs = []
    for item in qs:
        if reqs and reqs[-1][0][0] is item[0]:
            reqs[-1].append(item)
        else:
            reqs.append([item])
    groups, cur, used = [], [], 0
    for r in reqs:
        est = min(max_length, len(r[0][0]) // 3 + 2) + sum(len(q) // 3 + 4 + sum(min(max_length // 2, len(x) // 3 + 2) for x in o)
                                                            for _, q, o, _, _ in r)
        if cur and used + est > budget:
            groups.append(cur)
            cur, used = [], 0
        cur.append(r)
        used += est
    return groups + ([cur] if cur else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="cross-encoder/ettin-reranker-150m-v1")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=64000)
    ap.add_argument("--batch_requests", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--max_k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--task_prompts", type=float, default=0.2,
                    help="share of single-question jev requests rendered with tasksource's task-specific prompts")
    ap.add_argument("--task_styles", default="directive:30,question:20,explained:3.333,terse:3.333,conversational:3.333,"
                    "annotation:3.333,grounded:3.333,system:3.333", help=f"style:weight among {','.join(STYLES)}")
    ap.add_argument("--task_labels", type=float, default=0.0,
                    help="share of styled rows whose options get their label definitions (fixed label sets only)")
    ap.add_argument("--task_layouts", default="inline,fields")
    ap.add_argument("--nota", type=float, default=0.05, help="per eligible question: prob of replacing the gold with NOTA, and of adding it as a distractor")
    ap.add_argument("--tok_budget", type=int, default=12288, help="tokens per forward/backward micro-batch")
    ap.add_argument("--row_tokens", type=int, default=8192)
    ap.add_argument("--save_every", type=int, default=8000)
    a = ap.parse_args()
    styles = {name: float(weight or 1) for name, _, weight in (s.partition(":") for s in a.task_styles.split(","))}
    if list(Path(a.out).glob("step-*")):
        raise SystemExit(f"{a.out} already has checkpoints")

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    reqs = load_manifest(a.manifest)
    rng.shuffle(reqs)
    print(f"{len(reqs):,} requests; {a.steps * a.batch_requests:,} will be seen", flush=True)
    scorer = Decider(a.base, max_length=a.max_length)
    scorer.model.backbone.gradient_checkpointing_enable()
    scorer.model.train()
    opt = torch.optim.AdamW(scorer.model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_cosine_schedule_with_warmup(opt, a.warmup, a.steps)
    meta = {k: getattr(a, k) for k in ("manifest", "lr", "batch_requests", "max_k", "seed", "steps", "max_length", "nota",
                                       "task_prompts", "task_styles", "row_tokens", "tok_budget")}

    ri, t0, run_loss, run_n, run_acc = 0, time.time(), 0.0, 0, 0.0
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        qs = []
        for _ in range(a.batch_requests):
            r = reqs[ri % len(reqs)]
            ri += 1
            state, questions, options = r["state"], r["questions"], r["options"]
            if a.task_prompts and r.get("source") == "tasksource" and len(questions) == 1 and rng.random() < a.task_prompts:
                style = rng.choices(list(styles), weights=list(styles.values()))[0]
                styled = task_style(state, r["task"], rng.random(), rng.choice(a.task_layouts.split(",")), [style])
                if styled:
                    state, questions = styled[0], [styled[1]]
                    if rng.random() < a.task_labels:
                        options = [describe_options(options[0], r["task"])]
            for q, o, t, w in zip(questions, options, r["targets"], r.get("weights") or [1.0] * len(r["questions"])):
                if len(o) >= 2 and sum(t) > 0:
                    o, t = subsample(o, t, a.max_k, rng)
                    if a.nota:
                        o, t = nota(o, t, a.nota, rng)
                    qs.append((state or "(empty)", q, o, t, w))
        for rg in request_batches(qs, a.max_length, a.tok_budget):
            g = [it for r in rg for it in r]
            logits = shared_logits(scorer, [(r[0][0], [(q, o) for _, q, o, _, _ in r]) for r in rg], a.max_length,
                                   row_tokens=a.row_tokens, token_budget=a.tok_budget, truncate=True)
            losses = [soft_ce(z, t) for z, (_, _, _, t, _) in zip(logits, g)]
            (torch.stack([x * w for x, (*_, w) in zip(losses, g)]).sum() / len(qs)).backward()
            run_loss += sum(x.item() for x in losses)
            run_n += len(g)
            run_acc += sum(float(t[int(z.argmax())] > 0) for z, (_, _, _, t, _) in zip(logits, g))
        torch.nn.utils.clip_grad_norm_(scorer.model.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % 50 == 0:
            print(f"step {step}/{a.steps} loss {run_loss / max(1, run_n):.4f} acc {run_acc / max(1, run_n):.3f} "
                  f"lr {sch.get_last_lr()[0]:.2e} {(time.time() - t0) / step:.2f}s/step", flush=True)
            run_loss, run_n, run_acc = 0.0, 0, 0.0
        if step % a.save_every == 0 or step == a.steps:
            scorer.save_pretrained(f"{a.out}/step-{step}", step=step, train=meta)
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
