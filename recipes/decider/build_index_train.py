#!/usr/bin/env python3
"""Decision Index source TRAIN splits, rendered exactly like the suite (the Dinah-0 recipe).

For each public benchmark whose source has an official train (or non-test) split, the kit's own
normalizer logic and instruction constants are applied to that split, so training rows look like
evaluation rows. Never touched: test splits, BBH (canary), benchmarks without a train split, and
"unselected pool" items of benchmarks the suite sampled from one pool (CLadder, cfcolor, BPoMP,
ChessBench, POP909): those share the evaluation distribution with no official split.

Firewall (fail-closed): a row is dropped when its state text, or any of its (state, question)
pairs, matches a suite row of edition 0.3 after whitespace/case normalization.

  python3 recipes/decider/build_index_train.py                 # rows -> data/index_train/rows/*.jsonl
  python3 recipes/decider/build_index_train.py --manifest      # + data/index_train.jsonl (capped, firewalled)
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import random
import re
import sqlite3
import unicodedata
import urllib.request
import zipfile
from decimal import Decimal
from pathlib import Path

from paths import DATA, HUB, RAW, SUITE  # noqa: E402  (also puts the kit on sys.path)

from decision_index.engines.base import text  # noqa: E402
from decision_index.suite.build import adapters_added as AA  # noqa: E402
from decision_index.suite.build.adapters_mechanical import parse_acos_line  # noqa: E402
from decision_index.suite.build.adapters_selection import num  # noqa: E402
from decision_index.suite.build.layout import choice_row  # noqa: E402
from decision_index.suite.build.normalize_text import _mc_row  # noqa: E402

from typed_models.formatting import question_options, state_text  # noqa: E402

OUT = DATA / "index_train"
SEED = 20261009
BANKING_TRAIN = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"


def pq_rows(path, drop=()):
    import pyarrow.parquet as pq

    t = pq.read_table(path)
    return t.drop_columns([c for c in drop if c in t.column_names]).to_pylist()


# ---- normalize_direct.knowledge / anli / banking77 / clinc150 on train splits -------------------

def arc():
    for fam in ("ARC-Easy", "ARC-Challenge"):
        for r in pq_rows(HUB / f"ai2_arc/{fam}/train-00000-of-00001.parquet"):
            yield choice_row(fam, "train", r["id"], r["question"], r["choices"]["text"], r["choices"]["label"].index(r["answerKey"]))


def mmlu():
    for i, r in enumerate(pq_rows(HUB / "mmlu/auxiliary_train/train-00000-of-00001.parquet")):
        r = r["train"]
        yield choice_row("MMLU", "auxiliary_train", i, r["question"], r["choices"], r["answer"])


def winogrande():
    for i, r in enumerate(pq_rows(HUB / "winogrande/winogrande_xl/train-00000-of-00001.parquet")):
        yield choice_row("WinoGrande", "train", i, "Which option correctly fills the blank?\n" + r["sentence"], [r["option1"], r["option2"]], int(r["answer"]) - 1)


def hellaswag():
    clean = lambda s: " ".join(str(s).split())  # noqa: E731
    for i, r in enumerate(pq_rows(HUB / "hellaswag/data/train-00000-of-00001.parquet")):
        yield choice_row("HellaSwag", "train", f"{i}:{r['ind']}", "Which continuation is most plausible?\n" + clean(r["ctx"]), [clean(x) for x in r["endings"]], int(r["label"]))


def anli():
    mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}
    for p in sorted((HUB / "anli/plain_text").glob("train_*.parquet")):
        for r in pq_rows(p):
            yield choice_row("ANLI", p.stem.split("-")[0], r["uid"], "Classify the relationship between the premise and hypothesis.\nPremise: " + r["premise"] + "\nHypothesis: " + r["hypothesis"], list(mapping.values()), int(r["label"]))


def banking77():
    p = HUB / "banking77/train.csv"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BANKING_TRAIN, p)
    labels = json.loads((RAW / "banking77/categories.json").read_text())
    for i, r in enumerate(csv.DictReader(p.open(newline="", encoding="utf-8"))):
        yield choice_row("BANKING77", "train", i, "Classify the banking intent of this user request:\n" + r["text"], labels, labels.index(r["category"]))


def clinc150():
    obj = json.loads((RAW / "clinc150/data_full.json").read_text())
    labels = sorted({pair[1] for pair in obj["train"]} | {"oos"})
    desc = ["out of scope: none of the listed intents" if x == "oos" else x.replace("_", " ") for x in labels]
    for split in ("train", "oos_train"):
        for i, (t, lab) in enumerate(obj[split]):
            yield choice_row("CLINC150+OOS", split, i, "Classify the intent of this user request, or choose out of scope if none applies:\n" + t, desc, labels.index(lab))


# ---- normalize_text on train splits -------------------------------------------------------------

def vast():
    labels = ["against", "favor", "neutral"]
    for r in csv.DictReader((RAW / "vast/data/VAST/vast_train.csv").open(newline="", encoding="utf-8")):
        prompt = f"Topic: {r['topic_str']}\nPost: {r['post']}\nDetermine the stance of the post toward the topic."
        yield _mc_row("VAST", "train", r["new_id"], prompt, labels, int(r["label"]))


def contractnli():
    with zipfile.ZipFile(RAW / "repos/contractnli/resources/contract-nli.zip") as z:
        data = json.loads(z.read("contract-nli/train.json"))
    criteria = {"Entailment": "The contract entails the hypothesis.", "Contradiction": "The contract contradicts the hypothesis.", "NotMentioned": "The hypothesis is neither entailed nor contradicted by the contract."}
    for doc in data["documents"]:
        ann = doc["annotation_sets"][0]["annotations"]
        qs = {k: dict(type="choice", instructions="Classify the relationship between the contract and this hypothesis:\n" + lab["hypothesis"], criteria=criteria) for k, lab in data["labels"].items()}
        yield dict(id=f'ContractNLI:train:{doc["id"]}', family="ContractNLI", split="train", state=doc["text"], questions=qs, expected={k: a["choice"] for k, a in ann.items()})


def nli4ct():
    criteria = {"Entailment": "The clinical trial evidence entails the statement.", "Contradiction": "The clinical trial evidence contradicts the statement."}
    with zipfile.ZipFile(RAW / "repos/nli4ct/training_data.zip") as z:
        names = z.namelist()
        member = next(n for n in names if n.endswith("train.json"))
        rows = json.loads(z.read(member))
        ct = {Path(n).stem: n for n in names if n.startswith("CT json/") and n.endswith(".json")}
        for uid, row in rows.items():
            section = row["Section_id"]
            state = {}
            for key, role in [("Primary_id", "primary_trial"), ("Secondary_id", "secondary_trial")]:
                if key in row:
                    trial = json.loads(z.read(ct[row[key]]))
                    state[role] = {"id": row[key], "section": section, "text": trial[section]}
            if row.get("Label") not in criteria:
                continue
            yield dict(id=f"NLI4CT-2024:train:{uid}", family="NLI4CT-2024", split="train", state=state, questions={"answer": dict(type="choice", instructions="Classify the statement against the supplied clinical trial evidence:\n" + row["Statement"], criteria=criteria)}, expected={"answer": row["Label"]})


def humicroedit():
    with zipfile.ZipFile(RAW / "downloads/humicroedit-full.zip") as z:
        src = list(csv.DictReader(io.StringIO(z.read("semeval-2020-task-7-dataset/subtask-2/train.csv").decode("utf-8-sig"))))
    for r in src:
        if r["label"] not in {"1", "2"}:
            continue
        crit = {}
        for n in (1, 2):
            rendered, c = re.subn(r"<[^<>]+/>", lambda _: r[f"edit{n}"], r[f"original{n}"])
            if c != 1:
                break
            crit[f"headline_{n}"] = rendered
        else:
            yield dict(id=f'Humicroedit:train:{r["id"]}', family="Humicroedit", split="train", state="", questions={"answer": dict(type="choice", instructions="Which edited news headline is funnier?", criteria=crit)}, expected={"answer": "headline_" + r["label"]})


def isarcasm():
    """Train files carry task A + B labels (task C pairs exist only at test time)."""
    for lang in ("En", "Ar"):
        p = RAW / f"repos/isarcasm/train/train.{lang}.csv"
        for i, row in enumerate(csv.DictReader(p.open(encoding="utf-8-sig", newline=""))):
            text = row.get("tweet") or ""
            if not text.strip() or row.get("sarcastic") not in {"0", "1"}:
                continue
            yield dict(id=f"iSarcasmEval-A-{lang}:train:{i}", family=f"iSarcasmEval-A-{lang}", split="train", state=text, questions={"sarcastic": dict(type="choice", instructions="Is this text intended to be sarcastic?", criteria={"no": "No", "yes": "Yes"})}, expected={"sarcastic": "yes" if row["sarcastic"] == "1" else "no"})
            labels = ["sarcasm", "irony", "satire", "understatement", "overstatement", "rhetorical_question"]
            if lang == "En" and row["sarcastic"] == "1" and all(row.get(x) in {"0", "1"} for x in labels):
                qs = {x: dict(type="choice", instructions=f'Does this text exhibit {x.replace("_", " ")}? Evaluate this category independently.', criteria={"no": "No", "yes": "Yes"}) for x in labels}
                yield dict(id=f"iSarcasmEval-B-En:train:{i}", family="iSarcasmEval-B-En", split="train", state=text, questions=qs, expected={x: "yes" if row[x] == "1" else "no" for x in labels})


def esci(cap):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    base = RAW / "repos/esci/shopping_queries_dataset"
    ex = pq.read_table(base / "shopping_queries_dataset_examples.parquet", filters=[("split", "=", "train")])
    idx = list(range(ex.num_rows))
    random.Random(f"{SEED}:esci").shuffle(idx)
    ex = ex.take(sorted(idx[:cap]))
    pids = pc.unique(ex["product_id"])
    prod = pq.read_table(base / "shopping_queries_dataset_products.parquet", filters=[("product_id", "in", pids.to_pylist())])
    joined = ex.join(prod, keys=["product_locale", "product_id"], join_type="left outer").sort_by([("example_id", "ascending")])
    criteria = {"E": "Exact: the product satisfies the search query.", "S": "Substitute: a product that could substitute for the requested product.", "C": "Complement: a product that complements the requested product.", "I": "Irrelevant: the product does not address the requested product need."}
    for row in joined.to_pylist():
        if row.get("product_title") is None:
            continue
        product = {k.removeprefix("product_"): row[k] for k in ["product_title", "product_description", "product_bullet_point", "product_brand", "product_color"] if row.get(k) is not None}
        yield dict(id=f'Amazon-ESCI:train:{row["example_id"]}', family="Amazon-ESCI", split="train", state={"search_query": row["query"], "product": product}, questions={"answer": dict(type="choice", instructions="Classify the relevance of this product to the search query using the ESCI categories.", criteria=criteria)}, expected={"answer": row["esci_label"]})


# ---- adapters_mechanical.build_acos on train files ----------------------------------------------

def acos():
    sources = sorted((RAW / "repos/acos/data").glob("*/*_quad_*.tsv"))
    parsed = {p: [parse_acos_line(x) for x in p.read_text(encoding="utf-8").splitlines() if x] for p in sources}
    by_domain = {}
    for p in sources:  # same inventory as the suite (built over all files)
        by_domain.setdefault(p.parent.name, set()).update(pair[0] for _, pairs in parsed[p] for pair in pairs)
    by_domain = {d: {(c, s) for c in cats for s in ("0", "1", "2")} for d, cats in by_domain.items()}
    for p in sources:
        if not p.name.endswith("_train.tsv"):
            continue
        d = p.parent.name
        invs = [sorted(by_domain[d])[i:i + 64] for i in range(0, len(by_domain[d]), 64)]
        for line_no, (review, gold) in enumerate(parsed[p], 1):
            for gi, inv in enumerate(invs, 1):
                qs, ex = {}, {}
                for cat, s in inv:
                    sn = {"0": "negative", "1": "neutral", "2": "positive"}[s]
                    k = f"{cat}__sentiment_{sn}"
                    qs[k] = {"type": "choice", "instructions": ("Given the review below, decide whether it expresses at least one " f"aspect-category/sentiment pair ({cat}, {sn}) anywhere " "in the review. Answer yes only when that exact category and " "sentiment pair is present; answer no otherwise. Multiple opposite " "sentiments are separate valid questions."), "criteria": {"yes": "The exact category/sentiment pair is present.", "no": "The exact category/sentiment pair is absent."}}
                    ex[k] = "yes" if (cat, s) in gold else "no"
                yield {"id": f"ACOS:train:{d}:{line_no}:{gi}", "family": "ACOS-category-sentiment", "split": "train", "state": {"review": review, "task": "ACOS fixed category/sentiment presence"}, "questions": qs, "expected": ex}


# ---- adapters_added on train splits -------------------------------------------------------------

def _cand(name, sid, state, questions, expected):
    return dict(id=f"{name}:train:{sid}", family=name, split="train", state=state, questions=questions, expected=expected)


def mmlu_pro():
    for r in pq_rows(HUB / "MMLU-Pro/data/validation-00000-of-00001.parquet"):
        keys = AA.letters(len(r["options"]))
        yield _cand("MMLU-Pro", r["question_id"], r["question"], {"q": dict(type="choice", instructions=AA.CHOICE, criteria=dict(zip(keys, r["options"])))}, {"q": r["answer"]})


def ragtruth():
    folder = RAW / "repos/cand-RAGTruth/dataset"
    sources = {x["source_id"]: x for x in map(json.loads, (folder / "source_info.jsonl").open(encoding="utf-8"))}
    for r in map(json.loads, (folder / "response.jsonl").open(encoding="utf-8")):
        if r["split"] != "train":
            continue
        s = sources[r["source_id"]]
        yield _cand("RAGTruth", r["id"], {"prompt": s["prompt"], "response": r["response"]}, {"q": dict(type="noul", instructions=AA.RAGTRUTH_INSTRUCTIONS, criteria=AA.RAGTRUTH_CRITERIA)}, {"q": len(r["labels"]) > 0})


def hover():
    db = sqlite3.connect(f"file:{RAW / AA.HOVER_DB[0]}?mode=ro", uri=True)
    for r in json.loads((RAW / "repos/cand-hover/data/hover/hover_train_release_v1.1.json").read_text(encoding="utf-8")):
        ev = []
        for t in dict.fromkeys(t for t, _ in r["supporting_facts"]):
            found = db.execute("SELECT id, text FROM documents WHERE id=?", (unicodedata.normalize("NFD", t),)).fetchall()
            if len(found) != 1:
                break
            ev.append({"title": t, "text": found[0][1]})
        else:
            yield _cand("HoVer", r["uid"], {"claim": r["claim"], "evidence": ev}, {"q": dict(type="choice", instructions=AA.HOVER_INSTRUCTIONS, criteria=AA.HOVER_CRITERIA)}, {"q": r["label"]})
    db.close()


def when2call():
    """Preference split: chosen vs rejected response, as a 2-option choice in the MCQ framing."""
    for i, r in enumerate(map(json.loads, (HUB / "When2Call/train/when2call_train_pref.jsonl").open(encoding="utf-8"))):
        user = [m["content"] for m in r["messages"] if m["role"] == "user"]
        if len(user) != 1:
            continue
        pair = [r["chosen_response"]["content"], r["rejected_response"]["content"]]
        order = [0, 1]
        random.Random(f"{SEED}:w2c:{i}").shuffle(order)
        keys = AA.letters(2)
        yield _cand("When2Call", i, {"tools": [json.loads(t) for t in r["tools"]], "question": user[0]}, {"q": dict(type="choice", instructions=AA.WHEN2CALL_INSTRUCTIONS, criteria={k: pair[j] for k, j in zip(keys, order)})}, {"q": keys[order.index(0)]})


def newyorker():
    for p in sorted((HUB / "newyorker_caption_contest/matching").glob("*.parquet")):
        for r in pq_rows(p, drop=("image",)):
            keys = AA.letters(len(r["caption_choices"]))
            yield _cand("New Yorker", r["instance_id"], {"scene": r["image_location"], "description": r["image_description"], "uncanny_description": r["image_uncanny_description"], "entities": r["entities"]}, {"q": dict(type="choice", instructions=AA.NEWYORKER_INSTRUCTIONS, criteria=dict(zip(keys, r["caption_choices"])))}, {"q": r["label"]})


def gsm8k():
    """0.3-style options: distractors are golds of other TRAIN problems of similar size."""
    rows = pq_rows(HUB / "gsm8k/main/train-00000-of-00001.parquet")
    golds = [Decimal(r["answer"].split("####")[-1].strip().replace(",", "")) for r in rows]
    distinct = sorted(set(golds))
    pos = {g: i for i, g in enumerate(distinct)}
    for i, (r, g) in enumerate(zip(rows, golds)):
        for count in (4, 10):
            j = pos[g]
            lo = max(0, min(j - count // 2, len(distinct) - count))
            window = distinct[lo:lo + count]
            opts = [num(x) for x in window]
            random.Random(f"{SEED}:GSM:{i}:{count}").shuffle(opts)
            yield {"id": f"GSM8K-{count}:train:{i}", "family": f"GSM8K-{count}", "split": "train", "state": {"question": r["question"], "task": f"GSM8K deterministic {count}-choice numeric selection"}, "questions": {"answer": {"type": "choice", "instructions": "Choose the numeric answer to the problem in state. Do not provide reasoning. This is a named multiple-choice adaptation of GSM8K.", "criteria": {f"option_{k}": v for k, v in enumerate(opts)}}}, "expected": {"answer": f"option_{opts.index(num(g))}"}}


BUILDERS = {
    "arc": arc, "mmlu": mmlu, "winogrande": winogrande, "hellaswag": hellaswag, "anli": anli,
    "banking77": banking77, "clinc150": clinc150, "vast": vast, "contractnli": contractnli, "nli4ct": nli4ct,
    "humicroedit": humicroedit, "isarcasm": isarcasm, "esci": lambda: esci(40000), "acos": acos,
    "mmlu_pro": mmlu_pro, "ragtruth": ragtruth, "hover": hover, "when2call": when2call,
    "newyorker": newyorker, "gsm8k": gsm8k,
}


# ---- firewall + manifest ------------------------------------------------------------------------

def norm(s):
    return " ".join(str(s).lower().split())


def h(*parts):
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:20]


def suite_fingerprints():
    states, sq = set(), set()
    for f in ("selected-rows.jsonl.gz", "added-rows.jsonl.gz", "gsm8k-rows.jsonl.gz"):
        for line in gzip.open(SUITE / f, "rt", encoding="utf-8"):
            r = json.loads(line)
            st = norm(state_text(r.get("state")))
            if len(st) >= 40:
                states.add(h(st))
            for q in (r.get("questions") or {}).values():
                sq.add(h(st, norm(q.get("instructions", ""))))
    return states, sq


ACOS_RANDOM_NEG = 3
ACOS_MAX_WEIGHT = 20.0
ACOS_CAP = 30000  # questions; ACOS is case-exact over ~228 pairs per review, so it gets 3x the default family cap


def cap_questions(row):
    """ACOS asks one yes/no question per (category, sentiment) pair (64 per chunk, ~0.5% yes in the suite) and
    scores whole reviews exactly, so both misses and false alarms fail a case. Keep every "yes", hard negatives
    (the same category with the other sentiments; the same entity with another attribute and the same
    sentiment, up to 2 per yes) and ACOS_RANDOM_NEG random "no" questions importance-weighted by
    (remaining "no" / sampled), capped at ACOS_MAX_WEIGHT, so the "no" prior stays close to the full chunk's.
    Weights are normalized to mean 1 within the request (row["weights"], aligned with row["questions"])."""
    if row["family"] != "ACOS-category-sentiment":
        return row
    rng = random.Random(f"{SEED}:acos:{row['id']}")
    ex = row["expected"]
    split = {k: (k.split("__sentiment_")[0], k.split("__sentiment_")[1]) for k in ex}
    yes = [k for k in ex if ex[k] == "yes"]
    hard = set()
    for k in yes:
        cat, sn = split[k]
        ent = cat.split("#")[0]
        hard |= {j for j, (c, s) in split.items() if c == cat and s != sn and ex[j] == "no"}
        same_ent = [j for j, (c, s) in split.items() if c != cat and c.split("#")[0] == ent and s == sn and ex[j] == "no"]
        hard |= set(rng.sample(same_ent, min(2, len(same_ent))))
    rest = sorted(k for k in ex if ex[k] == "no" and k not in hard)
    rnd = rng.sample(rest, min(ACOS_RANDOM_NEG, len(rest)))
    w_rnd = min(ACOS_MAX_WEIGHT, len(rest) / max(1, len(rnd)))
    keep = [(k, 1.0) for k in yes] + [(k, 1.0) for k in sorted(hard)] + [(k, w_rnd) for k in rnd]
    rng.shuffle(keep)
    mean_w = sum(w for _, w in keep) / len(keep)
    return dict(row, questions={k: row["questions"][k] for k, _ in keep}, expected={k: ex[k] for k, _ in keep},
                weights=[w / mean_w for _, w in keep])


def to_request(row, source="index_train"):
    """Suite-format row -> grouped manifest request, rendered exactly as the engines see it."""
    row = cap_questions(row)
    st = state_text(row["state"])
    qs, opts, tgts, kinds, wts = [], [], [], [], []
    row_w = dict(zip(row["questions"], row.get("weights") or [1.0] * len(row["questions"])))
    for qid, q in row["questions"].items():
        keys, texts = question_options(q)
        gold = row["expected"][qid]
        if q["type"] == "noul":
            gold = "true" if gold in (True, "true", "True") else "false"
        if str(gold) not in map(str, keys):
            continue
        qs.append(text(q.get("instructions", "")))
        opts.append(texts)
        tgts.append([1.0 if str(k) == str(gold) else 0.0 for k in keys])
        kinds.append(q["type"])
        wts.append(row_w[qid])
    if not qs:
        return None
    req = {"group_id": row["id"], "state": st, "questions": qs, "options": opts, "targets": tgts, "kinds": kinds,
           "source": source, "task": row["family"], "raw": {"state": row["state"], "questions": row["questions"]}}
    if any(w != 1.0 for w in wts):
        req["weights"] = wts
    return req


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--cap", type=int, default=10000, help="max QUESTIONS (decisions) per family: balances tasks")
    args = ap.parse_args()
    (OUT / "rows").mkdir(parents=True, exist_ok=True)
    for name in args.only or BUILDERS:
        n = 0
        with (OUT / "rows" / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in BUILDERS[name]():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        print(f"{name}: {n} rows", flush=True)
    if not args.manifest:
        return
    states, sq = suite_fingerprints()
    print(f"suite fingerprints: {len(states)} states, {len(sq)} state+question", flush=True)
    by_family, dropped = {}, {}
    for p in sorted((OUT / "rows").glob("*.jsonl")):
        for line in p.open(encoding="utf-8"):
            row = json.loads(line)
            st = norm(state_text(row["state"]))
            hit = (len(st) >= 40 and h(st) in states) or any(h(st, norm(q.get("instructions", ""))) in sq for q in row["questions"].values())
            if hit:
                dropped[row["family"]] = dropped.get(row["family"], 0) + 1
                continue
            req = to_request(row)
            if req:
                by_family.setdefault(row["family"], []).append(req)
    rng = random.Random(SEED)
    out, stats = [], {}
    for fam, reqs in sorted(by_family.items()):
        rng.shuffle(reqs)
        keep, nq = [], 0
        for r in reqs:
            if nq >= (ACOS_CAP if fam == "ACOS-category-sentiment" else args.cap):
                break
            keep.append(r)
            nq += len(r["questions"])
        out.extend(keep)
        stats[fam] = {"available": len(reqs), "kept": len(keep), "questions": nq, "firewall_dropped": dropped.get(fam, 0)}
    rng.shuffle(out)
    mp = DATA / "index_train.jsonl"
    mp.parent.mkdir(parents=True, exist_ok=True)
    with mp.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rec = {"manifest": mp.name, "requests": len(out), "cap": args.cap, "seed": SEED, "families": stats,
           "firewall": "drop if normalized state (>=40 chars) or any (state, instructions) matches suite-0.3 rows"}
    (mp.with_suffix(".record.json")).write_text(json.dumps(rec, indent=2))
    for fam, s in stats.items():
        print(f"  {fam:28s} avail {s['available']:7d} kept {s['kept']:6d} fw_drop {s['firewall_dropped']}")
    print(f"manifest {mp}: {len(out)} requests", flush=True)


if __name__ == "__main__":
    main()
