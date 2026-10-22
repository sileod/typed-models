#!/usr/bin/env python3
"""Tool-selection and tool-retrieval training data, rendered in the suite's own BFCL / ToolRet framing.

Sources (train data only; none is a Decision Index test source):
  glaive   glaiveai/glaive-function-calling-v2       BFCL framing: one yes/no "invoke this tool?" per offered schema
  hermes   NousResearch/hermes-function-calling-v1   same (single-turn + multi-turn files, first user turn)
  xlam     Salesforce/xlam-function-calling-60k      same; GATED (needs an HF token with the terms accepted)
  toolace  Team-ACE/ToolACE                         BFCL framing (apache-2.0)
  toolret  mangopy/ToolRet-Training-20w              ToolRet framing: one yes/no "relevant?" per candidate tool
  homeassist acon96/Home-Assistant-Requests          device control (MIT, home-llm's own generator): per device "act on it?",
                                                     per acted device "which service?", and "does it change any device?"

Quality rules:
  * BFCL framing: positives are the tools the assistant actually called in its first reply; negatives are
    the other schemas offered in the same prompt, plus a few random schemas from other samples (distinct
    names). Prompts whose first reply calls no tool are kept at most --no_call_frac (all-"no" rows).
  * ToolRet-Training negatives come from ToolBench, where functionally equivalent APIs are labelled
    negative. A hard negative is dropped when its name+description overlaps a positive (Jaccard >=
    --neg_jaccard) or is mostly made of the query's words; kept hard negatives are capped and topped up
    with random negatives (positives of other queries).
  * Firewall: a sample is dropped when its user query matches a suite query of BFCL, ToolRet, API-Bank or
    When2Call (normalized exact match, or >= 50% of its word 6-grams found in the suite); homeassist requests
    are checked against Home-Appliance requests the same way.

  python3 recipes/decider/build_tool_train.py            # -> data/tool_train.jsonl (+ .record.json)
"""

import argparse
import ast
import gzip
import json
import random
import re
from pathlib import Path

from build_index_train import SEED, norm, to_request
from paths import DATA, HUB, SUITE

BFCL_INSTRUCTIONS = (
    "Given the complete user conversation and the published function schemas in state, mark each "
    "candidate tool yes if it should be invoked to answer the request, or no otherwise. "
    "This evaluates tool-name selection only; do not produce arguments or call ordering.\n\n"
)
BFCL_CRITERIA = {"yes": "Invoke this tool.", "no": "Do not invoke this tool."}
TOOLRET_TASK = "Assess whether this candidate is relevant/useful to the query. Use the full text below; return the probability of relevance."
TOOLRET_CRITERIA = {"no": "Not relevant/useful to the query.", "yes": "Relevant/useful to the query."}  # suite order
WORD = re.compile(r"[a-z0-9]+")


def words(s):
    return WORD.findall(str(s).lower())


def content(s):
    return {w for w in words(s) if len(w) >= 4}


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


# ---- suite firewall -----------------------------------------------------------------------------

class Firewall:
    def __init__(self):
        self.exact, self.shingles = set(), set()
        for f in ("selected-rows.jsonl.gz", "added-rows.jsonl.gz"):
            for line in gzip.open(SUITE / f, "rt", encoding="utf-8"):
                r = json.loads(line)
                fam = str(r.get("family") or r.get("benchmark") or "")
                if not any(k in fam for k in ("BFCL", "ToolRet", "API-Bank", "When2Call", "Home-Appliance")):
                    continue
                st = r.get("state") or {}
                qs = []
                if isinstance(st, dict):
                    qs.append(st.get("query") or st.get("question") or st.get("request") or "")
                    for turn in json.dumps(st.get("conversation") or st.get("dialogue") or []).split('"content": ')[1:]:
                        qs.append(turn.split('"')[1] if '"' in turn else turn)
                    for d in st.get("dialogue") or []:
                        if isinstance(d, dict):
                            qs.append(d.get("text", ""))
                for q in qs:
                    self.add(q)

    @staticmethod
    def grams(q):
        w = words(q)
        return {" ".join(w[i:i + 6]) for i in range(max(0, len(w) - 5))}

    def add(self, q):
        n = norm(q)
        if len(n) >= 20:
            self.exact.add(n)
            self.shingles |= self.grams(q)

    def hit(self, q):
        if norm(q) in self.exact:
            return True
        g = self.grams(q)
        return bool(g) and len(g & self.shingles) / len(g) >= 0.5


# ---- BFCL-framed sources ------------------------------------------------------------------------

def parse_glaive_functions(system):
    body = system.split("-", 1)[1] if "Use them if required -" in system else ""
    out, dec, i = [], json.JSONDecoder(), 0
    body = body.strip()
    while i < len(body):
        while i < len(body) and body[i] in " \n\r\t,":
            i += 1
        if i >= len(body):
            break
        try:
            obj, i = dec.raw_decode(body, i)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict) and obj.get("name"):
            out.append(obj)
    return out


def glaive():
    for i, x in enumerate(json.load(open(HUB / "glaive-function-calling-v2/glaive-function-calling-v2.json"))):
        fns = parse_glaive_functions(x["system"])
        if not fns or "USER:" not in x["chat"] or "ASSISTANT:" not in x["chat"]:
            continue
        user = x["chat"].split("USER:", 1)[1].split("ASSISTANT:", 1)[0].strip()
        reply = x["chat"].split("ASSISTANT:", 1)[1].split("USER:", 1)[0]
        called = set(re.findall(r'<functioncall>\s*\{\s*"name"\s*:\s*"([^"]+)"', reply))
        yield "glaive", i, user, fns, called


def hermes():
    seen = set()
    for f in ("func-calling-singleturn.json", "func-calling.json"):
        for x in json.load(open(HUB / "hermes-function-calling-v1" / f)):
            if x["id"] in seen:
                continue
            seen.add(x["id"])
            tools = json.loads(x["tools"]) if isinstance(x["tools"], str) else x["tools"]
            fns = [t.get("function", t) for t in tools if isinstance(t, dict)]
            conv = x["conversations"]
            hum = next((c["value"] for c in conv if c["from"] == "human"), None)
            gpt = next((c["value"] for c in conv if c["from"] == "gpt"), "")
            if not hum or not fns:
                continue
            called = set(re.findall(r'<tool_call>\s*\{\s*["\']name["\']\s*:\s*["\']([^"\']+)', gpt))
            yield "hermes", x["id"], hum.strip(), fns, called


def xlam():
    p = HUB / "xlam-function-calling-60k/xlam_function_calling_60k.json"
    if not p.exists():
        print("xlam: not downloaded (gated dataset; run `huggingface-cli login` and accept its terms) -> skipped")
        return
    for x in json.load(open(p)):
        fns = json.loads(x["tools"])
        called = {a["name"] for a in json.loads(x["answers"])}
        yield "xlam", x["id"], x["query"], fns, called


def toolace():
    for i, x in enumerate(json.load(open(HUB / "ToolACE/data.json"))):
        if "invoke:\n" not in x["system"]:
            continue
        try:
            fns = json.JSONDecoder().raw_decode(x["system"].split("invoke:\n", 1)[1].strip())[0]
        except json.JSONDecodeError:
            continue
        conv = x["conversations"]
        if len(conv) < 2 or conv[0]["from"] != "user" or conv[1]["from"] != "assistant":
            continue
        reply = conv[1]["value"].strip()
        called = {f["name"] for f in fns if reply.startswith("[") and f["name"] + "(" in reply}
        yield "toolace", i, conv[0]["value"].strip(), fns, called


def bfcl_rows(source, fw, rng, args, pool):
    stats = {"seen": 0, "firewall": 0, "no_call_dropped": 0, "bad": 0}
    rows = []
    for name, sid, user, fns, called in source():
        stats["seen"] += 1
        names = [f["name"] for f in fns if isinstance(f, dict) and f.get("name")]
        if not names or not called <= set(names):
            stats["bad"] += 1
            continue
        if fw.hit(user):
            stats["firewall"] += 1
            continue
        if not called and rng.random() >= args.no_call_frac:
            stats["no_call_dropped"] += 1
            continue
        fns = [f for f in fns if f.get("name")]
        extra = [f for f in rng.sample(pool, min(len(pool), args.random_tools)) if f["name"] not in names]
        fns = fns + extra
        rng.shuffle(fns)
        qs, ex = {}, {}
        for f in fns:
            key = "tool_" + re.sub(r"[^A-Za-z0-9_]+", "_", f["name"]).strip("_")
            qs[key] = {"type": "choice", "instructions": BFCL_INSTRUCTIONS + f"\n\nCandidate tool: {f['name']}", "criteria": BFCL_CRITERIA}
            ex[key] = "yes" if f["name"] in called else "no"
        rows.append({"id": f"tool-{name}:{sid}", "family": f"tool-selection/{name}", "split": "train",
                     "state": {"conversation": [[{"role": "user", "content": user}]], "functions": fns, "task": "BFCL tool-name selection"},
                     "questions": qs, "expected": ex})
    return rows, stats


# ---- ToolRet-framed source ----------------------------------------------------------------------

def tool_text(raw):
    try:
        d = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return str(raw)
    if not isinstance(d, dict):
        return str(raw)
    parts = [d.get(k) for k in ("name", "tool_name", "api_name", "description", "api_description", "tool_description")]
    return " ".join(str(p) for p in parts if p)


def toolret(fw, rng, args):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    rows_all = []
    for k in range(args.toolret_shards):
        p = hf_hub_download("mangopy/ToolRet-Training-20w", f"ToolRet-Training-20w/train-0000{k}-of-00006.parquet",
                            repo_type="dataset", local_dir=str(HUB / "ToolRet-Training-20w"))
        rows_all += pq.read_table(p).to_pylist()
    lit = lambda x: ast.literal_eval(x) if isinstance(x, str) else list(x)  # noqa: E731
    pos_pool = [d for r in rows_all[:20000] for d in lit(r["positive"])]
    stats = {"seen": 0, "firewall": 0, "hard_negatives": 0, "hard_kept": 0, "hard_dropped_similar": 0}
    rows = []
    for r in rows_all:
        stats["seen"] += 1
        q = r["query"]
        if fw.hit(q):
            stats["firewall"] += 1
            continue
        P, N = lit(r["positive"]), lit(r["negative"])
        pw = [content(tool_text(x)) for x in P]
        qw = content(q)
        hard = []
        for x in N:
            stats["hard_negatives"] += 1
            xw = content(tool_text(x))
            if any(jaccard(xw, w) >= args.neg_jaccard for w in pw) or (xw and len(xw & qw) / len(xw) >= 0.5):
                stats["hard_dropped_similar"] += 1
                continue
            hard.append(x)
        rng.shuffle(hard)
        hard = hard[: args.hard_negatives]
        stats["hard_kept"] += len(hard)
        pset = set(P)
        rand = [x for x in rng.sample(pos_pool, args.random_negatives + len(P)) if x not in pset][: args.random_negatives]
        cands = [(x, "yes") for x in P] + [(x, "no") for x in hard + rand]
        rng.shuffle(cands)
        qs = {f"doc_{i}": {"type": "choice", "instructions": {"task": TOOLRET_TASK, "candidate": x}, "criteria": TOOLRET_CRITERIA} for i, (x, _) in enumerate(cands)}
        ex = {f"doc_{i}": y for i, (_, y) in enumerate(cands)}
        rows.append({"id": f"tool-toolret:{r['id']}", "family": "tool-retrieval/toolret-train", "split": "train",
                     "state": {"query": q, "task": "Rank candidate documents/tools by relevance to this query."},
                     "questions": qs, "expected": ex})
    return rows, stats


# ---- device-control source ---------------------------------------------------------------------

HA_TASK = "Handle this smart-home request given the listed devices, their current states and the available services."
HA_CALL = re.compile(r"```homeassistant\n(.*?)```", re.S)


def homeassist(fw, rng, args):
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("acon96/Home-Assistant-Requests", "home_assistant_train.jsonl", repo_type="dataset",
                        local_dir=str(HUB / "Home-Assistant-Requests"))
    stats, rows, seen = {"seen": 0, "duplicate": 0, "firewall": 0, "bad": 0}, [], set()
    for i, line in enumerate(open(p)):
        stats["seen"] += 1
        system, user, reply = (c["value"] for c in json.loads(line)["conversations"][:3])
        if norm(user) in seen:  # the source repeats each request over several houses
            stats["duplicate"] += 1
            continue
        seen.add(norm(user))
        if fw.hit(user):
            stats["firewall"] += 1
            continue
        try:
            calls = [json.loads(c) for b in HA_CALL.findall(reply) for c in b.splitlines() if c.strip()]
            services = [x.strip().split("(")[0] for x in system.split("Services:", 1)[1].split("\n", 1)[0].split(",")]
            devices = [d for d in system.split("Devices:\n", 1)[1].splitlines() if d.strip()]
        except (IndexError, json.JSONDecodeError):
            stats["bad"] += 1
            continue
        ids = [d.split(" ", 1)[0] for d in devices]
        act = {c["target_device"]: c["service"] for c in calls}
        if not set(act) <= set(ids):
            stats["bad"] += 1
            continue
        same = [d for d in ids if d not in act and d.split(".")[0] in {a.split(".")[0] for a in act}]
        rest = [d for d in ids if d not in act and d not in same]
        rng.shuffle(same), rng.shuffle(rest)
        asked = sorted(set(act) | set(same[: args.ha_negatives // 2]) | set(rest[: args.ha_negatives - min(len(same), args.ha_negatives // 2)]), key=ids.index)
        qs = {"changes": {"type": "choice", "instructions": "Does handling this request change the state of any device?",
                          "criteria": {"yes": "Yes, at least one device changes.", "no": "No, nothing is changed (e.g. a status question)."}}}
        ex = {"changes": "yes" if act else "no"}
        for d in asked:
            k = "act_" + re.sub(r"\W+", "_", d)
            qs[k] = {"type": "choice", "instructions": f"Should device {d} be acted on to handle the request?",
                     "criteria": {"yes": f"Yes, act on {d}.", "no": f"No, leave {d} unchanged."}}
            ex[k] = "yes" if d in act else "no"
        for d, svc in act.items():
            opts = sorted({s for s in services if s.split(".")[0] == d.split(".")[0]} | {svc})
            if len(opts) >= 2:
                k = "svc_" + re.sub(r"\W+", "_", d)
                qs[k] = {"type": "choice", "instructions": f"Which service should be called on {d}?", "criteria": {o: o for o in opts}}
                ex[k] = svc
        rows.append({"id": f"tool-homeassist:{i}", "family": "device-control/homeassist", "split": "train",
                     "state": {"request": user, "devices": devices, "services": services, "task": HA_TASK},
                     "questions": qs, "expected": ex})
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=20000, help="max questions per family")
    ap.add_argument("--no_call_frac", type=float, default=0.3, help="share of no-call prompts kept")
    ap.add_argument("--random_tools", type=int, default=2, help="random schemas from other samples added as negatives")
    ap.add_argument("--toolret_shards", type=int, default=2)
    ap.add_argument("--neg_jaccard", type=float, default=0.25)
    ap.add_argument("--hard_negatives", type=int, default=4)
    ap.add_argument("--random_negatives", type=int, default=4)
    ap.add_argument("--ha_negatives", type=int, default=8, help="homeassist: untouched devices asked about (half same domain)")
    ap.add_argument("--out", default=str(DATA / "tool_train.jsonl"))
    args = ap.parse_args()
    rng = random.Random(SEED)
    fw = Firewall()
    print(f"firewall: {len(fw.exact)} suite queries, {len(fw.shingles)} 6-grams", flush=True)

    fam_rows, stats = {}, {}
    pool = []
    bfcl_sources = (glaive, hermes, xlam, toolace)
    for src in bfcl_sources:  # random-schema pool from all BFCL-framed sources
        pool += [f for _, _, _, fns, _ in src() for f in fns if isinstance(f, dict) and f.get("name")][:50000]
    for src in bfcl_sources:
        rows, st = bfcl_rows(src, fw, rng, args, pool)
        if rows:
            fam_rows[rows[0]["family"]] = rows
        stats[src.__name__] = st
    rows, stats["toolret"] = toolret(fw, rng, args)
    fam_rows["tool-retrieval/toolret-train"] = rows
    rows, stats["homeassist"] = homeassist(fw, rng, args)
    fam_rows["device-control/homeassist"] = rows

    out, fam_stats = [], {}
    for fam, rows in sorted(fam_rows.items()):
        rng.shuffle(rows)
        nq = 0
        for row in rows:
            if nq >= args.cap:
                break
            req = to_request(row, source="tool_train")
            if req:
                out.append(req)
                nq += len(req["questions"])
        yes = sum(o[t.index(1.0)].lower().startswith("yes") for r in out if r["task"] == fam for o, t in zip(r["options"], r["targets"]))
        fam_stats[fam] = {"rows": len(rows), "questions": nq, "yes_share": round(yes / max(1, nq), 3)}
    rng.shuffle(out)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rec = {"out": args.out, "requests": len(out), "args": vars(args), "families": fam_stats, "source_stats": stats}
    Path(args.out).with_suffix(".record.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
