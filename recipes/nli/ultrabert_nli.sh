#!/usr/bin/env bash
# tasksource/UltraBERT-150M-nli: multi-task tasksource NLI with tasknet, 40k steps from dmis-lab/UltraBERT-150M, then
# 160k more from that checkpoint (seed 1). Run from the repository root.
set -euo pipefail
export PYTHONPATH=$PWD
python3 recipes/nli/train.py --model dmis-lab/UltraBERT-150M --max_steps 40000 --out runs/nli_40k
python3 recipes/nli/train.py --model runs/nli_40k/final --seed 1 --lr 2e-5 --max_steps 160000 --out runs/nli_200k
