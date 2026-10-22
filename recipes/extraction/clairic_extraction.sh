#!/usr/bin/env bash
# tasksource/UltraBERT-150M-clairic-extraction: 20k steps on 400k (field, text) pairs of the extraction config of
# tasksource/tasksource-jev-typed-decisions, then the three evaluations. Run from the repository root.
set -euo pipefail
X=recipes/extraction; OUT=runs/clairic_extraction; export PYTHONPATH=$PWD
python3 $X/train.py --model dmis-lab/UltraBERT-150M --max_steps 20000 --limit_tasks 0 --out $OUT
python3 $X/eval_jev.py tag:$OUT/final --out $OUT/eval
python3 $X/eval_ood.py tag:$OUT/final --out $OUT/eval
python3 $X/eval_typed.py $OUT/final $OUT/eval/typed_extraction.jsonl
