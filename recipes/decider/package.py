#!/usr/bin/env python3
"""Package a decider checkpoint (train.py) as a self-contained Hub model folder: config + tokenizer of the base,
model.safetensors (backbone + head, bf16 by default), decider_config.json, and the standalone inference code
(inference/decider.py, inference/decision_index_engine.py for the Decision Index kit). Nothing is uploaded.

  python3 recipes/decider/package.py runs/decider/step-64000 runs/decider/package
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

INFERENCE = Path(__file__).resolve().parent / "inference"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                    help="stored weights; inference runs in bf16 autocast either way (decider.py loads fp32 parameters)")
    a = ap.parse_args()
    ck, out = Path(a.ckpt), Path(a.out)
    cfg = json.loads((ck / "joint_config.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    AutoConfig.from_pretrained(cfg["base"]).save_pretrained(out)
    AutoTokenizer.from_pretrained(cfg["base"]).save_pretrained(out)
    sd = torch.load(ck / "pytorch_model.bin", map_location="cpu", weights_only=True)
    if a.dtype == "bf16":
        sd = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sd.items()}
    save_file({k: v.contiguous() for k, v in sd.items()}, str(out / "model.safetensors"))
    (out / "decider_config.json").write_text(json.dumps({
        "format": "pair-v1", "arch": "joint", "base": cfg["base"], "train_max_length": cfg["max_length"],
        "inference_max_length": 8192, "step": cfg.get("step"), "train": cfg.get("train", {}),
        "layout": cfg["layout"], "pool": cfg["pool"], "attn": cfg["attn"], "weights_dtype": a.dtype}, indent=2))
    for f in INFERENCE.glob("*.py"):
        shutil.copy(f, out / f.name)
    print(f"{ck} -> {out} ({sum(v.numel() for v in sd.values()) / 1e6:.1f}M parameters)")


if __name__ == "__main__":
    main()
