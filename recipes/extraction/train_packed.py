#!/usr/bin/env python3
"""One-pass ("packed", GLiNER-style) extraction: every field of a document in a single forward pass.

Input: ``bos, (MARK instruction_1), ..., (MARK instruction_K), sep, text``. Each field is the mean of its marker and
instruction token states; each text token gets O / B / I logits per field from a low-rank bilinear head,
``out(gelu(token(h_t) * field(f_k)))``, so K fields cost one encoder pass instead of K. Trained on the ``extraction``
config of tasksource/tasksource-jev-typed-decisions, one row per document with its fields shuffled (half the time a
random subset, so any number of fields is seen). Fields with a span past the training window are not trained on.

  python3 recipes/extraction/train_packed.py --max_steps 20000 --out runs/extraction/packed_20k
  python3 recipes/extraction/eval_jev.py packed:runs/extraction/packed_20k/final --out runs/extraction/eval
"""
import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import datasets  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments  # noqa: E402

from eval_jev import choose, decode  # noqa: E402
from train import JEV, JEV_REVISION  # noqa: E402

MARK, SEP = "<|im_start|>", "<|sep|>"


class PackedTagger(nn.Module):
    def __init__(self, encoder, dim=256):
        super().__init__()
        hidden = encoder.config.hidden_size
        self.encoder, self.token, self.field, self.out = encoder, nn.Linear(hidden, dim), nn.Linear(hidden, dim), nn.Linear(dim, 3)

    def forward(self, input_ids, attention_mask, field_ids, labels=None):
        """field_ids: B x L, k + 1 on the marker and instruction tokens of field k, else 0. Logits: B x K x L x 3."""
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pool = nn.functional.one_hot(field_ids, int(field_ids.max()) + 1)[..., 1:].to(h.dtype)  # B x L x K
        f = pool.transpose(1, 2) @ h / pool.sum(1).clamp(min=1)[..., None]
        logits = self.out(nn.functional.gelu(self.token(h)[:, None] * self.field(f)[:, :, None]))
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.functional.cross_entropy(logits.float().flatten(0, 2), labels[:, :logits.size(1)].flatten(), ignore_index=-100)
        return out

    def save(self, path):
        self.encoder.save_pretrained(path)
        torch.save({k: v for k, v in self.state_dict().items() if not k.startswith("encoder.")}, Path(path) / "head.pt")

    @classmethod
    def load(cls, path):
        model = cls(AutoModel.from_pretrained(path, trust_remote_code=True))
        model.load_state_dict(torch.load(Path(path) / "head.pt"), strict=False)
        return model


class Encoder:
    """Token ids of the packed input, field by field, and windows over long texts."""
    def __init__(self, tok, max_length, max_instruction=128):
        self.tok, self.max_length, self.max_instruction = tok, max_length, max_instruction
        self.mark, self.sep = tok.convert_tokens_to_ids([MARK, SEP])
        self.bos = tok.bos_token_id if tok.bos_token_id is not None else tok("")["input_ids"][0]

    def fields(self, instructions):
        """Groups of field indices whose prefix fits in half the input, and each field's ids (marker + instruction)."""
        ids = [[self.mark] + self.tok(x, add_special_tokens=False)["input_ids"][:self.max_instruction] for x in instructions]
        groups, size = [[]], 2
        for k, x in enumerate(ids):
            if groups[-1] and size + len(x) > self.max_length // 2:
                groups, size = groups + [[]], 2
            groups[-1].append(k)
            size += len(x)
        return groups, ids

    def prefix(self, ids):
        input_ids, field_ids = [self.bos], [0]
        for k, x in enumerate(ids):
            input_ids += x
            field_ids += [k + 1] * len(x)
        return input_ids + [self.sep], field_ids + [0]

    def windows(self, n_prefix, n_text, stride=128):
        room = self.max_length - n_prefix
        starts = list(range(0, max(1, n_text - stride), max(1, room - stride))) if n_text > room else [0]
        return [(s, min(s + room, n_text)) for s in starts]


def tags(offsets, spans):
    """O / B / I per text token, as tasknet SpanTagging."""
    out = []
    for a, b in offsets:
        overlapping = [x for x, y in spans if a < y and b > x]
        out.append(0 if not overlapping else 1 if a <= overlapping[0] else 2)
    return out


class Collator:
    def __init__(self, encoder, train=True):
        self.enc, self.train = encoder, train

    def example(self, row):
        fields = [(f["instruction"], [[v["start"], v["end"]] for v in a["values"]]) for f, a in zip(row["fields"], row["answers"])]
        random.shuffle(fields)
        if self.train and len(fields) > 1 and random.random() < 0.5:
            fields = fields[:random.randint(1, len(fields) - 1)]
        groups, ids = self.enc.fields([q for q, _ in fields])
        fields, ids = [fields[k] for k in groups[0]], [ids[k] for k in groups[0]]
        input_ids, field_ids = self.enc.prefix(ids)
        text = self.enc.tok(row["state"], add_special_tokens=False, return_offsets_mapping=True)
        end = self.enc.windows(len(input_ids), len(text["input_ids"]))[0][1]
        offsets = text["offset_mapping"][:end]
        limit = offsets[-1][1] if offsets else 0
        labels = [[-100] * len(input_ids) + (tags(offsets, spans) if all(y <= limit for _, y in spans) else [-100] * end)
                  for _, spans in fields]
        return input_ids + text["input_ids"][:end], field_ids + [0] * end, labels

    def __call__(self, rows):
        examples = [self.example(r) for r in rows]
        L, K = max(len(x[0]) for x in examples), max(len(x[2]) for x in examples)
        batch = {"input_ids": torch.zeros(len(rows), L, dtype=torch.long), "attention_mask": torch.zeros(len(rows), L, dtype=torch.long),
                 "field_ids": torch.zeros(len(rows), L, dtype=torch.long), "labels": torch.full((len(rows), K, L), -100)}
        for i, (input_ids, field_ids, labels) in enumerate(examples):
            n = len(input_ids)
            batch["input_ids"][i, :n], batch["attention_mask"][i, :n] = torch.tensor(input_ids), 1
            batch["field_ids"][i, :n] = torch.tensor(field_ids)
            batch["labels"][i, :len(labels), :n] = torch.tensor(labels)
        return batch


class Packed:
    """Inference: every field of a text in one pass per window (fields whose instructions overflow half the input go
    to further groups). Same ``spans`` contract as eval_jev.Tagger."""
    def __init__(self, name, max_length=None, batch=8):
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.model = PackedTagger.load(name).cuda().eval().half()
        self.enc, self.batch = Encoder(self.tok, max_length or 2048), batch

    @torch.no_grad()
    def many_spans(self, instructions, text, threshold=None):
        """Per instruction, (score, start, end) character spans, as Tagger.spans."""
        doc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        groups, ids = self.enc.fields(instructions)
        found = [None] * len(instructions)
        for group in groups:
            input_ids, field_ids = self.enc.prefix([ids[k] for k in group])
            windows = self.enc.windows(len(input_ids), len(doc["input_ids"]))
            probs, tokens, L = [], [], len(input_ids) + max(b - a for a, b in windows)
            for i in range(0, len(windows), self.batch):
                chunk = windows[i:i + self.batch]
                x, m, f = (torch.zeros(len(chunk), L, dtype=torch.long) for _ in range(3))
                for j, (a, b) in enumerate(chunk):
                    n = len(input_ids) + b - a
                    x[j, :n], m[j, :n], f[j, :len(field_ids)] = torch.tensor(input_ids + doc["input_ids"][a:b]), 1, torch.tensor(field_ids)
                probs.append(self.model(x.cuda(), m.cuda(), f.cuda())["logits"].float().softmax(-1).cpu())
                tokens += [[(len(input_ids) + t - a, *doc["offset_mapping"][t]) for t in range(a, b)] for a, b in chunk]
            probs = torch.cat(probs).numpy()  # W x K x L x 3
            for j, k in enumerate(group):
                found[k] = decode(probs[:, j], tokens, threshold)
        return found

    def __call__(self, text, fields):
        return choose(text, fields, self.many_spans([f["instruction"] for f in fields], text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dmis-lab/UltraBERT-150M", help="encoder to start from (a tagger directory works: its head is dropped)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=6, help="documents per device batch; gradient accumulation fills ~24")
    ap.add_argument("--max_steps", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_sources", default="", help="comma-separated source prefixes left out, e.g. procedural/")
    a = ap.parse_args()
    random.seed(a.seed)
    out = Path(a.out)
    train = datasets.load_dataset(JEV, "extraction", split="train", revision=JEV_REVISION)
    if a.exclude_sources:
        train = train.filter(lambda s: not s.startswith(tuple(a.exclude_sources.split(","))), input_columns="source")
    train = train.select_columns(["state", "fields", "answers"])
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = PackedTagger(AutoModel.from_pretrained(a.model, trust_remote_code=True))
    heads = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": a.lr}, {"params": heads, "lr": a.head_lr}],
                                  eps=1e-6, weight_decay=1e-6)
    args = TrainingArguments(output_dir=str(out / "checkpoints"), max_steps=a.max_steps, per_device_train_batch_size=a.batch,
                             gradient_accumulation_steps=max(1, 24 // a.batch), learning_rate=a.lr, warmup_steps=0.07,
                             bf16=True, logging_steps=80, save_steps=0.1, save_total_limit=2,
                             remove_unused_columns=False, dataloader_num_workers=4, seed=a.seed, report_to=[])
    Trainer(model=model, args=args, train_dataset=train, data_collator=Collator(Encoder(tok, a.max_length)),
            optimizers=(optimizer, None)).train()
    model.save(out / "final")
    tok.save_pretrained(out / "final")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
