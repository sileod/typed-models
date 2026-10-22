"""Decision Index engine for this model (https://github.com/apolinario/decision-index).

    python -m decision_index run --engine decision_index_engine:DeciderEngine \
        --option repo=<repo or directory> [--option revision=<commit>] --edition 0.3 --out runs/<name>

Put this file next to `decider.py` (both are in the model repository) and on PYTHONPATH.
Kit rules: nothing is truncated and no option is dropped. A (state, question, option) pair longer than
8,192 tokens (the encoder's native context) raises `Unsupported`. One call = one request: all questions
of the request are scored together, in length-sorted padded batches.
"""

from __future__ import annotations

import sys
from pathlib import Path

from decision_index.engines import Engine, Unsupported

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decider import Decider, TooLong  # noqa: E402


class DeciderEngine(Engine):
    name = "decider"
    latency = ("In-process request wall time: tokenization plus the encoder passes of one request "
               "(all its question/option pairs batched together); excludes model loading.")

    def __init__(self, repo: str = str(Path(__file__).resolve().parent), revision: str | None = None,
                 device: str | None = None, max_length: int | None = None, token_budget: int = 65536, **options):
        super().__init__(**options)
        self.model = Decider.from_pretrained(repo, revision=revision, device=device, max_length=max_length,
                                             token_budget=int(token_budget))
        self.repo = repo
        self.provenance = {"repo": repo, "revision": revision or "main", "device": str(self.model.device),
                           "dtype": "bfloat16 autocast" if self.model.device.type == "cuda" else "float32",
                           "max_length": self.model.max_length, "token_budget": self.model.token_budget,
                           "truncation": f"none: pairs above {self.model.max_length} tokens raise Unsupported"}

    def __call__(self, state, questions):
        try:
            out = self.model.answer(state, questions)
        except TooLong as e:
            raise Unsupported(f"context window: {e}")
        except ValueError as e:
            raise Unsupported(str(e))
        return {"model": self.repo, "answers": out["answers"]}, None

    def synchronize(self):
        if self.model.device.type == "cuda":
            self.model.torch.cuda.synchronize()
