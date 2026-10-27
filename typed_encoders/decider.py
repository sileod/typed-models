"""Joint decider: one scalar per (state, question, option), softmax over a question's options.

Shared-state layout: a request's state is written once, then each question block ("Question: q\\nOption:"),
then every option block (" o [SEP]"), with tree attention:
  state tokens see the state; question tokens see the state and their own question;
  option tokens see the state, their question and themselves (never other options).
RoPE positions continue the virtual pair [CLS] state [SEP] Question: q\\nOption: o [SEP], so position ids never
exceed one pair's length. An option's logit is a linear head on the mean of its own block's tokens.
Training truncates the state to fit ``max_length``; inference never truncates (TooLong instead).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

CONFIG_NAME, WEIGHTS_NAME = "joint_config.json", "pytorch_model.bin"
# Padded tokens per forward. Rows are length-sorted, so a small budget means length-homogeneous batches: on an A30,
# 16 short requests train 1.35x faster at 4096 than at 65536 (0.170 vs 0.231 s/step), at 6.5 vs 10.4GB. A row longer
# than the budget still runs, alone.
TOKEN_BUDGET = 4096
MASK_CELLS = 2 ** 26  # per forward: rows x length^2 dense mask cells (memory of the masks and SDPA bias)


class TooLong(ValueError):
    """An input does not fit the model's context; it is never truncated at inference."""


class JointHead(nn.Module):
    def __init__(self, backbone: nn.Module, hidden: int):
        super().__init__()
        self.backbone = backbone
        self.drop = nn.Dropout(0.1)
        self.scorer = nn.Linear(hidden, 1)


class Decider:
    def __init__(self, base: str, max_length: int = 2048, device: Optional[str] = None, weights: Optional[str] = None,
                 attn: str = "cached"):
        from transformers import AutoModel, AutoTokenizer

        self.base, self.max_length, self.attn = base, int(max_length), attn
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(base)
        backbone = AutoModel.from_pretrained(base)
        self.model = JointHead(backbone, backbone.config.hidden_size)
        if weights:
            self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.to(self.device)

    @classmethod
    def from_pretrained(cls, ckpt: str, device: Optional[str] = None, revision: Optional[str] = None) -> "Decider":
        """A training run's output dir (joint_config.json + pytorch_model.bin), or a released checkpoint, local or on
        the Hub (decider_config.json + config.json + model.safetensors, e.g. tasksource/tasksource-decider-nano)."""
        path = Path(ckpt)
        if not path.exists():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(ckpt, revision=revision))
        if (path / CONFIG_NAME).exists():
            cfg = json.loads((path / CONFIG_NAME).read_text())
            return cls(cfg["base"], cfg["max_length"], device, str(path / WEIGHTS_NAME), cfg.get("attn", "cached"))
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        cfg = json.loads((path / "decider_config.json").read_text())
        self = cls.__new__(cls)  # the release ships the whole backbone: no download of the base weights
        self.base, self.max_length, self.attn = str(path), int(cfg.get("train_max_length", 2048)), cfg.get("attn", "cached")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(path)
        backbone = AutoModel.from_config(AutoConfig.from_pretrained(path))
        self.model = JointHead(backbone, backbone.config.hidden_size)
        self.model.load_state_dict({k: v.float() for k, v in load_file(str(path / "model.safetensors")).items()})
        self.model.to(self.device)
        return self

    def save_pretrained(self, ckpt: str, **extra) -> None:
        d = Path(ckpt)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), d / WEIGHTS_NAME)
        cfg = {"base": self.base, "max_length": self.max_length, "format": "pair-v1", **extra,
               "layout": "shared", "pool": "opt", "attn": self.attn}
        (d / CONFIG_NAME).write_text(json.dumps(cfg, indent=2))

    @torch.no_grad()
    def score_many(self, items, max_length: Optional[int] = None, token_budget: int = TOKEN_BUDGET) -> List[np.ndarray]:
        """Option logits for [(state, question, options)], consecutive items with the same state sharing its encoding."""
        self.model.eval()
        reqs = []
        for state, question, options in items:
            if reqs and reqs[-1][0] == state:
                reqs[-1][1].append((question, options))
            else:
                reqs.append((state, [(question, options)]))
        return [z.cpu().numpy().astype(np.float64) for z in
                shared_logits(self, reqs, max_length or self.max_length, token_budget=token_budget, attn=self.attn)]


def _shared_tokens(tok, state, question, options):
    p = tok(state)["input_ids"]
    q = tok(f"Question: {question}\nOption:", add_special_tokens=False)["input_ids"]
    o = [tok(" " + x, add_special_tokens=False)["input_ids"] + [tok.sep_token_id] for x in options]
    return p, q, o


def _fit_train(p, q, o, L):
    """Training-time truncation to L positions per virtual pair: shorten the state first (keep at least L/2 of it),
    then the question, then each option."""
    tail = len(q) + max(map(len, o))
    if len(p) + tail > L:
        keep = max(L // 2, L - tail)
        p = p[:keep - 1] + p[-1:]
    if len(p) + len(q) + 8 > L:
        q = q[:max(8, L - len(p) - 8)]
    room = L - len(p) - len(q)
    o = [x if len(x) <= room else x[:room - 1] + x[-1:] for x in o]
    return p, q, o


def attn_full(n_questions: int, state_len: int, attn: str, auto_extra: int = 4096) -> bool:
    """"cached": the state attends only to itself, so one state encoding serves every question. "full": one row per
    question, the state also attends to its question. "auto": full when that costs at most `auto_extra` extra tokens."""
    if attn == "auto":
        return (n_questions - 1) * state_len <= auto_extra
    if attn not in ("cached", "full"):
        raise ValueError(f"unknown attn mode {attn!r}")
    return attn == "full"


def shared_rows(tok, requests, max_positions, row_tokens=16384, truncate=False, row_extra=2048, attn="cached",
                auto_extra=4096):
    """Lay requests [(state, [(question, options), ...])] out as rows (prefix_ids, [(qid, q_ids, [(oid, o_ids)])], full).
    A question whose block does not fit a row is split over several rows (question block repeated). A row holds at
    most max(2 * state, state + row_extra) tokens (capped by row_tokens); outputs do not depend on it."""
    rows, qid, oid = [], 0, 0
    for state, questions in requests:
        cur, used, p0, full = [], 0, None, False
        for question, options in questions:
            p, q, o = _shared_tokens(tok, state, question, options)
            if truncate:
                p, q, o = _fit_train(p, q, o, max_positions)
            longest = len(p) + len(q) + max(map(len, o))
            if longest > max_positions:
                raise TooLong(f"a (state, question, option) pair needs {longest} tokens > {max_positions} (no truncation)")
            if p0 is None:
                full = attn_full(len(questions), len(p), attn, auto_extra)
            if p0 is None or len(p) != len(p0) or full:
                if cur:
                    rows.append((p0, cur, full))
                p0, cur, used = p, [], len(p)
                cap = min(row_tokens, max(2 * len(p), len(p) + row_extra))
            opts = [(oid + k, x) for k, x in enumerate(o)]
            oid += len(o)
            while opts:
                take, size = [], len(q)
                for item in opts:
                    if take and used + size + len(item[1]) > cap:
                        break
                    take.append(item)
                    size += len(item[1])
                if cur and used + size > cap:
                    rows.append((p0, cur, full))
                    cur, used = [], len(p0)
                    continue
                cur.append((qid, q, take))
                used += size
                opts = opts[len(take):]
            qid += 1
        if cur:
            rows.append((p0, cur, full))
    return rows, qid, oid


def shared_logits(scorer: Decider, requests, max_positions: Optional[int] = None, row_tokens: int = 16384,
                  token_budget: int = TOKEN_BUDGET, truncate: bool = False, row_extra: int = 2048, attn: str = "cached",
                  auto_extra: int = 4096) -> List[torch.Tensor]:
    """Option logits, one [K] tensor per question (requests flattened in order). Differentiable; the caller controls
    train/eval mode and grad. Rows are batched up to `token_budget` padded tokens per forward."""
    tok, dev = scorer.tok, scorer.device
    bb = scorer.model.backbone
    w = getattr(bb.config, "sliding_window", None)
    rows, nq, no = shared_rows(tok, requests, int(max_positions or scorer.max_length), row_tokens, truncate, row_extra,
                               attn, auto_extra)
    sums = torch.zeros(no, bb.config.hidden_size, device=dev)
    cnts = torch.zeros(no, device=dev)
    lens = [len(p) + sum(len(q) + sum(len(x) for _, x in o) for _, q, o in qs) for p, qs, _ in rows]
    order = sorted(range(len(rows)), key=lambda i: lens[i])
    i = 0
    while i < len(order):
        j = i + 1
        while (j < len(order) and (j - i + 1) * lens[order[j]] <= token_budget
               and (j - i + 1) * lens[order[j]] ** 2 <= MASK_CELLS):
            j += 1
        batch = [rows[k] for k in order[i:j]]
        L = max(lens[k] for k in order[i:j])
        B = len(batch)
        ids = torch.full((B, L), tok.pad_token_id, dtype=torch.long)
        pos = torch.zeros((B, L), dtype=torch.long)
        kind = torch.full((B, L), -1, dtype=torch.long)  # 0 state, 1 question, 2 option
        qix = torch.full((B, L), -1, dtype=torch.long)
        oix = torch.full((B, L), -1, dtype=torch.long)
        rowfull = torch.tensor([f for _, _, f in batch], dtype=torch.bool)
        for r, (p, qs, _) in enumerate(batch):
            ids[r, :len(p)] = torch.tensor(p)
            pos[r, :len(p)] = torch.arange(len(p))
            kind[r, :len(p)] = 0
            c = len(p)
            for qi, q, opts in qs:
                ids[r, c:c + len(q)] = torch.tensor(q)
                pos[r, c:c + len(q)] = torch.arange(len(p), len(p) + len(q))
                kind[r, c:c + len(q)] = 1
                qix[r, c:c + len(q)] = qi
                c += len(q)
                for oi, x in opts:
                    ids[r, c:c + len(x)] = torch.tensor(x)
                    pos[r, c:c + len(x)] = torch.arange(len(p) + len(q), len(p) + len(q) + len(x))
                    kind[r, c:c + len(x)] = 2
                    qix[r, c:c + len(x)] = qi
                    oix[r, c:c + len(x)] = oi
                    c += len(x)
        ids, pos, kind, qix, oix, rowfull = (t.to(dev) for t in (ids, pos, kind, qix, oix, rowfull))
        ki, kj = kind[:, :, None], kind[:, None, :]
        allow = (kj == 0) & (ki >= 0)
        allow |= (kj == 1) & (ki >= 1) & (qix[:, :, None] == qix[:, None, :])
        allow |= (kj == 2) & (ki == 2) & (oix[:, :, None] == oix[:, None, :])
        allow |= rowfull[:, None, None] & (ki == 0) & (kj == 1)  # full rows: the state also reads its question
        allow |= torch.eye(L, dtype=torch.bool, device=dev)[None]
        masks = {"full_attention": allow[:, None]}
        if w is not None:
            masks["sliding_attention"] = (allow & ((pos[:, :, None] - pos[:, None, :]).abs() <= w))[:, None]
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            h = bb(input_ids=ids, attention_mask=masks, position_ids=pos).last_hidden_state
        h = h.float()
        sel = oix >= 0
        flat = oix[sel]
        sums = sums.index_add(0, flat, h[sel])
        cnts = cnts.index_add(0, flat, torch.ones_like(flat, dtype=cnts.dtype))
        i = j
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        z = scorer.model.scorer(scorer.model.drop(sums / cnts[:, None])).float().squeeze(-1)
    out, start = [], 0
    for _, questions in requests:
        for _, options in questions:
            out.append(z[start:start + len(options)])
            start += len(options)
    return out
