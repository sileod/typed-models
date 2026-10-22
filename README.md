# typed-models

Training recipes for small encoders that answer **typed** questions about a text: decisions (pick among options,
with a probability for each) and extraction (the spans that answer a field, or nothing). Each recipe reproduces a
released model.

| Model | Task | Recipe |
|---|---|---|
| [tasksource/tasksource-decider-nano](https://huggingface.co/tasksource/tasksource-decider-nano) (v5, 150M) | typed decisions ([Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index), [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)) | [`recipes/decider`](recipes/decider) |
| [tasksource/UltraBERT-150M-clairic-extraction](https://huggingface.co/tasksource/UltraBERT-150M-clairic-extraction) | typed extraction ([LocalLLaMA/typed-extraction](https://huggingface.co/datasets/LocalLLaMA/typed-extraction)) | [`recipes/extraction`](recipes/extraction) |
| [tasksource/UltraBERT-150M-nli](https://huggingface.co/tasksource/UltraBERT-150M-nli) | NLI / zero-shot classification | [`recipes/nli`](recipes/nli) |

The data is [tasksource/tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions)
(its default config for decisions, its `extraction` config for extraction), plus the sources listed in each recipe.
Extraction and NLI train with [tasknet](https://github.com/sileod/tasknet).

```bash
pip install -e ".[decider,extraction,nli]"
bash recipes/decider/decider_nano_v5.sh          # needs a Decision Index kit checkout, see the script
bash recipes/extraction/clairic_extraction.sh
bash recipes/nli/ultrabert_nli.sh
```

## Decider

A joint cross-encoder (`typed_models/decider.py`): every option of a question gets a scalar, softmaxed over the
question's options. A request's state is encoded once and shared by all its questions and options (tree attention:
an option sees the state, its question and itself), so a request with many questions costs little more than one.
Trained with soft cross-entropy against target distributions on a 480k-request mix:

| Part | Share | Builder |
|---|---|---|
| jev-typed-decisions, firewalled, ≤ 600 questions per task | 42% | `build_jev.py`, `firewall.py`, `shape.py` |
| Decision Index source **train** splits, rendered like the suite | 30% | `build_index_train.py` |
| Tool selection and retrieval (glaive, hermes, xlam, ToolACE, ToolRet, Home-Assistant) | 12% | `build_tool_train.py` |
| LLM routing (xRouteBench, LLMRouterBench train data) | 8% | `build_router_train.py` |
| tasksource/synthetic-typed-decisions | 8% | `build_synth.py` |
| LocalLLaMA/typed-decisions train | 0.5% | `build_typed_decisions.py` |

Every part is firewalled against the Decision Index 0.3 suite (exact matches and word 6-gram overlap of the
content), and benchmarks without a train split are never used. All sources are pinned (`download.py`), and the
manifests rebuild byte-identical to the ones v5 was trained on.

## Extraction

A query-conditioned span tagger: the input is `field instruction [SEP] text`, and one O/B/I head tags the text,
so one model handles single values, lists and absent fields, for fields it has never seen.
`train_packed.py` is the one-pass variant: every field of a document in one forward pass (GLiNER-style marker
tokens and a field × token head). Evaluations: the extraction test split (`eval_jev.py`), out-of-domain MRQA and
FinEntity (`eval_ood.py`), and LocalLLaMA/typed-extraction (`eval_typed.py`, zero-shot).

## NLI

Multi-task training on every tasksource NLI-style task with tasknet (shared encoder, one head per label set).
