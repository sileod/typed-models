#!/usr/bin/env bash
# tasksource/tasksource-decider-nano v5: data -> 64k-step training -> Hub folder. Run from the repository root with
# DECISION_INDEX_KIT pointing at a checkout of https://github.com/apolinario/decision-index whose raw benchmark
# artifacts are built (work/artifacts/benchmark-suite/raw) and TYPED_MODELS_DATA at a data directory (default data/).
set -euo pipefail
R=recipes/decider; D=${TYPED_MODELS_DATA:-data}; export PYTHONPATH=$PWD
python3 $R/download.py
# jev-typed-decisions, firewalled against the 0.3 suite, at most 600 questions per task
python3 $R/build_jev.py --revision 9fe3dd87bdfa0f016cef38234c4835b797f7cbb2 --out $D/jev_full.jsonl
python3 $R/firewall.py $D/jev_full.jsonl $D/jev_cfw.jsonl
python3 $R/shape.py $D/jev_cfw.jsonl $D/jev_bal.jsonl --cap_q 600
# Decision Index source train splits, tool use, routing, synthetic-typed-decisions, LocalLLaMA/typed-decisions train
python3 $R/build_index_train.py --manifest
python3 $R/build_tool_train.py --cap 60000 --out $D/tool_train.jsonl
python3 $R/build_router_train.py --sources xroute llmrb --out $D/router_train.jsonl
python3 $R/build_synth.py --out $D/synth_train.jsonl
python3 $R/build_typed_decisions.py $D/typed_decisions_train.jsonl
python3 $R/firewall.py $D/typed_decisions_train.jsonl $D/typed_decisions_cfw.jsonl
python3 $R/mix.py $D/mix_tool.jsonl --total 480000 --part $D/jev_bal.jsonl:0.42 --part $D/index_train.jsonl:0.3 \
  --part $D/tool_train.jsonl:0.12 --part $D/router_train.jsonl:0.08 --part $D/synth_train.jsonl:0.08
python3 $R/mix.py $D/mix.jsonl --total 480000 --part $D/mix_tool.jsonl:0.995 --part $D/typed_decisions_cfw.jsonl:0.005
# one GPU, ~10 h on an A30-class card
python3 $R/train.py --manifest $D/mix.jsonl --steps 64000 --seed 1 --out runs/decider_nano_v5
python3 $R/package.py runs/decider_nano_v5/step-64000 runs/decider_nano_v5/package
