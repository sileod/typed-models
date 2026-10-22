"""Template-insensitive suite firewall: word 6-gram overlap between a training item's *content*
(state + option texts; the question template is ignored) and every string in suite-0.3 rows.

Exact hashing of (state, question) misses any rephrased template, different state rendering or reworded
option. Here a training text is a hit when it equals a suite string (normalized), when >= `frac` of its 6-grams occur
in the suite, or when it shares >= `span` 6-grams AND >= `span_frac` of them (a suite item embedded in a
longer prompt; the fraction guard keeps shared boilerplate such as NDA clauses or legal headers from
counting). Options are checked only when their joined text has >= `opt_words` words: short label sets
("entailment neutral contradiction", "1 2 3") are vocabulary, not content.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Iterable, Optional

WORD = re.compile(r"[a-z0-9]+")
SUITE_FILES = ("selected-rows.jsonl.gz", "added-rows.jsonl.gz", "gsm8k-rows.jsonl.gz")


def words(s: str):
    return WORD.findall(s.lower())


def grams(s: str, n: int = 6):
    w = words(s)
    return {hash(" ".join(w[i:i + n])) for i in range(max(0, len(w) - n + 1))}


def norm(s: str) -> str:
    return " ".join(words(s))


def leaves(x) -> Iterable[str]:
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from leaves(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from leaves(v)


class ContentFirewall:
    def __init__(self, suite_dir: Path, frac: float = 0.5, span: int = 30, span_frac: float = 0.2, min_grams: int = 4,
                 opt_words: int = 20, skip_families: Optional[Iterable[str]] = None):
        self.frac, self.span, self.span_frac, self.min_grams, self.opt_words = frac, span, span_frac, min_grams, opt_words
        self.exact, self.shingles = set(), set()
        skip = tuple(skip_families or ())
        for f in SUITE_FILES:
            for line in gzip.open(Path(suite_dir) / f, "rt", encoding="utf-8"):
                r = json.loads(line)
                if skip and str(r.get("family", "")).startswith(skip):
                    continue
                for s in leaves([r.get("state"), r.get("questions")]):
                    n = norm(s)
                    if len(n) >= 20:
                        self.exact.add(n)
                        self.shingles |= grams(s)

    def hit(self, text: str) -> bool:
        if norm(text) in self.exact:
            return True
        g = grams(text)
        if len(g) < self.min_grams:
            return False
        shared = len(g & self.shingles)
        f = shared / len(g)
        return f >= self.frac or (shared >= self.span and f >= self.span_frac)

    def request_hit(self, state: str, options: Iterable[Iterable[str]]) -> bool:
        """State, and the joined options of each question (an item's answer set is part of its content)."""
        if state and self.hit(state):
            return True
        joined = [" ".join(map(str, o)) for o in options if o]
        return any(self.hit(j) for j in joined if len(words(j)) >= self.opt_words)
