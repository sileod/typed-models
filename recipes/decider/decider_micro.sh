#!/usr/bin/env bash
# tasksource/tasksource-decider-micro: the nano v5 recipe on the 400m reranker (see decider_nano_v5.sh for the data).
# One A100/H100-class GPU, ~8-12 h; resumable: rerunning the same command continues from <out>/resume.pt.
set -euo pipefail
R=recipes/decider; D=${TYPED_MODELS_DATA:-data}; export PYTHONPATH=$PWD
python3 $R/train.py --base cross-encoder/ettin-reranker-400m-v1 --manifest $D/mix.jsonl --steps 64000 --seed 1 \
  --out runs/decider_micro
python3 $R/package.py runs/decider_micro/step-64000 runs/decider_micro/package
