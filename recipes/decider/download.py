#!/usr/bin/env python3
"""Download the Hub train sources of build_index_train.py, build_tool_train.py and build_router_train.py, pinned
to the revisions the published deciders were built from, into $TYPED_MODELS_DATA/hub (default data/hub).
xlam-function-calling-60k is gated: accept its terms on the Hub and log in first.

  python3 recipes/decider/download.py
"""

import tarfile
import urllib.request

from huggingface_hub import snapshot_download

from paths import HUB

BANKING_TRAIN = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
SOURCES = {  # local name: (repo, revision, files)
    "ai2_arc": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453",
                ["ARC-Challenge/train-00000-of-00001.parquet", "ARC-Easy/train-00000-of-00001.parquet"]),
    "anli": ("facebook/anli", "8e4813d81f46d313dac7892e1c28076917cfcdf9",
             [f"plain_text/train_r{i}-00000-of-00001.parquet" for i in (1, 2, 3)]),
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", ["main/train-00000-of-00001.parquet"]),
    "hellaswag": ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76", ["data/train-00000-of-00001.parquet"]),
    "mmlu": ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", ["auxiliary_train/train-00000-of-00001.parquet"]),
    "MMLU-Pro": ("TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be", ["data/validation-00000-of-00001.parquet"]),
    "newyorker_caption_contest": ("jmhessel/newyorker_caption_contest", "d81cbab7d0392708d5371d3a4960e69261824db4",
                                  ["matching/train-00000-of-00002.parquet", "matching/train-00001-of-00002.parquet",
                                   "matching/validation-00000-of-00001.parquet"]),
    "When2Call": ("nvidia/When2Call", "0582f7749df63a96fdc3070932e83e72396ace53",
                  ["train/when2call_train_pref.jsonl", "train/when2call_train_sft.jsonl"]),
    "winogrande": ("allenai/winogrande", "01e74176c63542e6b0bcb004dcdea22d94fb67b5", ["winogrande_xl/train-00000-of-00001.parquet"]),
    "glaive-function-calling-v2": ("glaiveai/glaive-function-calling-v2", "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac",
                                   ["glaive-function-calling-v2.json"]),
    "hermes-function-calling-v1": ("NousResearch/hermes-function-calling-v1", "dae3e1d28cfbcf4b915c04ea1e072030529b4bda",
                                   ["func-calling-singleturn.json", "func-calling.json"]),
    "xlam-function-calling-60k": ("Salesforce/xlam-function-calling-60k", "26d14ebfe18b1f7b524bd39b404b50af5dc97866",
                                  ["xlam_function_calling_60k.json"]),
    "ToolACE": ("Team-ACE/ToolACE", "6bda777c88d21e5a204703c1ee45597a8fa4f734", ["data.json"]),
    "ToolRet-Training-20w": ("mangopy/ToolRet-Training-20w", "fdf5a317455b1e60785de7ba587496aa6cc878e4",
                             [f"ToolRet-Training-20w/train-0000{k}-of-00006.parquet" for k in (0, 1)]),
    "Home-Assistant-Requests": ("acon96/Home-Assistant-Requests", "75f5abe4a3fc4f30b6ccfeb239f4da30d22b3619",
                                ["home_assistant_train.jsonl"]),
    "xroute": ("ulab-ai/xRouteBench", "ea4b6e1b29d9a734f55f0a637baf326bad6aa681",
               [f"{c}/train.parquet" for c in ("llm_candidates", "llmrouter_generic", "memory_locomo", "memory_longmemeval",
                                               "personalized", "timeseries")]),
    "llmrb": ("NPULH/LLMRouterBench", "0e5af1b84bf73437a01a1849c0f1d2468baa93fc", ["bench-release.tar.gz"]),
}


def main():
    for name, (repo, revision, files) in SOURCES.items():
        snapshot_download(repo, repo_type="dataset", revision=revision, allow_patterns=files, local_dir=str(HUB / name))
        print(f"{name}: {repo}@{revision[:8]}", flush=True)
    if not (HUB / "llmrb/bench-release").exists():
        with tarfile.open(HUB / "llmrb/bench-release.tar.gz") as t:
            t.extractall(HUB / "llmrb", filter="data")
    (HUB / "banking77").mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(BANKING_TRAIN, HUB / "banking77/train.csv")


if __name__ == "__main__":
    main()
