"""Download, tokenise and pack all data described in config["data"].

    python prepare_data.py --config config.json
    python prepare_data.py --config configs/24gb/kanvq_hybrid.json

Sources (all configurable):
  general  : FineWeb-Edu sample (language)                    -> general_{train,val}.bin
  openmath : nvidia/OpenMathInstruct-2 (NL chain-of-thought)  -> openmath_{train,val}_*  + eval_openmath.jsonl
  synth    : procedurally generated variable-chain arithmetic  -> synth_{train,val}_*     + eval_synth.jsonl
  gsm8k    : openai/gsm8k test split (evaluation only)         -> eval_gsm8k.jsonl
Runs are resumable per stage: delete a stage's output files to rebuild it.
"""
import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from hybrid.config import load_config
from hybrid.data import ReasoningWriter, format_prompt, make_chain_problem, segment_solution
from hybrid.tokenizer import load_tokenizer, train_tokenizer


def stream(ds_name, ds_config, split):
    from datasets import load_dataset
    kw = dict(split=split, streaming=True)
    return load_dataset(ds_name, ds_config, **kw) if ds_config else load_dataset(ds_name, **kw)


def openmath_iter(oc):
    sources = set(oc["sources"]) if oc.get("sources") else None
    seen_sources = {}
    for ex in stream(oc["dataset"], oc.get("config"), oc["split"]):
        src = ex.get("problem_source", "")
        seen_sources[src] = seen_sources.get(src, 0) + 1
        if sources and src not in sources:
            if sum(seen_sources.values()) == 20000 and not any(s in sources for s in seen_sources):
                print(f"WARNING: none of {sources} seen in the first 20k rows; sources present: {seen_sources}")
            continue
        sol = ex["generated_solution"].strip()
        if "\\boxed" not in sol:
            continue
        yield ex["problem"], sol, str(ex.get("expected_answer", ""))


def synth_iter(sc, seed, depths, exclude=None):
    rng = random.Random(seed)
    lo, hi = depths
    dlo, dhi = sc.get("distractors", [0, 2])
    while True:
        d = rng.randint(lo, hi)
        q, steps, a = make_chain_problem(rng, d, rng.randint(dlo, dhi))
        if exclude and q in exclude:
            continue
        yield q, steps, a, d


def tokenizer_corpus(dc):
    n = dc["tokenizer"].get("train_docs", 100000)
    if dc["general"].get("enabled", True):
        for ex in itertools.islice(stream(dc["general"]["dataset"], dc["general"].get("config"),
                                          dc["general"]["split"]), n):
            yield ex[dc["general"].get("text_field", "text")][:8000]
    if dc["openmath"].get("enabled", True):
        for p, s, _ in itertools.islice(openmath_iter(dc["openmath"]), n // 2):
            yield format_prompt(p) + s
    if dc["synth"].get("enabled", True):
        for q, st, _, _ in itertools.islice(synth_iter(dc["synth"], 7, dc["synth"]["train_depth"]), n // 10):
            yield format_prompt(q) + "".join(st)


def write_reasoning(tok, prefix_train, prefix_val, sample_iter, n_val, n_train, max_tok, dtype, eval_path):
    """sample_iter yields (question, steps:list[str], answer, depth|None)."""
    w_val, w_tr = ReasoningWriter(prefix_val, dtype), ReasoningWriter(prefix_train, dtype)
    ev = open(eval_path, "w", encoding="utf-8") if eval_path else None
    n_v = n_t = skipped = 0
    bar = tqdm(total=n_val + n_train, desc=Path(prefix_train).name)
    for chunk in iter(lambda: list(itertools.islice(sample_iter, 512)), []):
        flat, spans = [], []
        for q, steps, a, d in chunk:
            spans.append((len(flat), len(steps) + 1, q, a, d))
            flat.append(format_prompt(q))
            flat.extend(steps)
        enc = [e.ids for e in tok.encode_batch(flat, add_special_tokens=False)]
        for start, n, q, a, d in spans:
            segs = enc[start:start + n]
            if sum(map(len, segs)) + 4 > max_tok:
                skipped += 1
                continue
            if n_v < n_val:
                w_val.add(segs)
                if ev:
                    ev.write(json.dumps(dict(question=q, answer=a, depth=d)) + "\n")
                n_v += 1
            else:
                w_tr.add(segs)
                n_t += 1
            bar.update(1)
            if n_t >= n_train:
                break
        if n_t >= n_train:
            break
    bar.close()
    if ev:
        ev.close()
    (nv, tv), (nt, tt) = w_val.close(), w_tr.close()
    print(f"  val {nv} samples / {tv/1e6:.2f}M tok | train {nt} samples / {tt/1e6:.1f}M tok | skipped {skipped}")
    return dict(val_samples=nv, val_tokens=tv, train_samples=nt, train_tokens=tt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    dc = cfg["data"]
    out = Path(dc["dir"])
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # ---------------- tokenizer
    tok_path = out / "tokenizer.json"
    if not tok_path.exists():
        print("Training tokenizer ...")
        train_tokenizer(tokenizer_corpus(dc), dc["tokenizer"]["vocab_size"], tok_path)
    tok, sp = load_tokenizer(tok_path)
    V = tok.get_vocab_size()
    dtype = np.uint16 if V <= 65535 else np.uint32
    meta.update(vocab_size=V, dtype=np.dtype(dtype).name)
    print(f"tokenizer: {V} tokens, specials {sp}")

    # ---------------- general text
    gc = dc["general"]
    if gc.get("enabled", True) and not (out / "general_train.bin").exists():
        print("Tokenising general text ...")
        fv, ft = open(out / "general_val.bin", "wb"), open(out / "general_train.bin", "wb")
        nv = nt = 0
        bar = tqdm(total=gc["train_tokens"] + gc["val_tokens"], unit="tok", unit_scale=True, desc="general")
        it = (ex[gc.get("text_field", "text")] for ex in stream(gc["dataset"], gc.get("config"), gc["split"]))
        for chunk in iter(lambda: list(itertools.islice(it, 1024)), []):
            arrs = [np.asarray([sp.bos] + e.ids + [sp.eos], dtype=dtype)
                    for e in tok.encode_batch(chunk, add_special_tokens=False)]
            arr = np.concatenate(arrs)
            if nv < gc["val_tokens"]:
                arr.tofile(fv); nv += arr.size
            else:
                arr.tofile(ft); nt += arr.size
            bar.update(arr.size)
            if nt >= gc["train_tokens"]:
                break
        bar.close(); fv.close(); ft.close()
        meta["general"] = dict(train_tokens=int(nt), val_tokens=int(nv))

    # ---------------- OpenMathInstruct-2
    oc = dc["openmath"]
    if oc.get("enabled", True) and not (out / "openmath_train_index.npy").exists():
        print("Tokenising OpenMathInstruct-2 ...")
        it = ((p, segment_solution(s + "\n"), a, None) for p, s, a in openmath_iter(oc))
        meta["openmath"] = write_reasoning(tok, out / "openmath_train", out / "openmath_val", it,
                                           oc["val_samples"], oc["max_train_samples"],
                                           oc["max_tokens_per_sample"], dtype, out / "eval_openmath.jsonl")

    # ---------------- synthetic chains
    sc = dc["synth"]
    if sc.get("enabled", True) and not (out / "synth_train_index.npy").exists():
        print("Generating synthetic chain problems ...")
        lo, hi = sc["eval_depth"]
        rng = random.Random(1234)
        eval_items, eval_q = [], set()
        for d in range(lo, hi + 1):
            for _ in range(sc["eval_per_depth"]):
                q, st, a = make_chain_problem(rng, d, rng.randint(*sc.get("distractors", [0, 2])))
                eval_items.append(dict(question=q, answer=a, depth=d))
                eval_q.add(q)
        with open(out / "eval_synth.jsonl", "w", encoding="utf-8") as f:
            for e in eval_items:
                f.write(json.dumps(e) + "\n")
        it = synth_iter(sc, 42, sc["train_depth"], exclude=eval_q)
        meta["synth"] = write_reasoning(tok, out / "synth_train", out / "synth_val", it, sc["val_samples"],
                                        sc["train_samples"], 10 ** 9, dtype, None)

    # ---------------- GSM8K test (evaluation only)
    g8 = dc.get("gsm8k", {})
    if g8.get("enabled", True) and not (out / "eval_gsm8k.jsonl").exists():
        from datasets import load_dataset
        ds = load_dataset(g8["dataset"], g8.get("config"), split=g8["split"])
        with open(out / "eval_gsm8k.jsonl", "w", encoding="utf-8") as f:
            for ex in ds:
                gold = ex["answer"].split("####")[-1].strip().replace(",", "")
                f.write(json.dumps(dict(question=ex["question"], answer=gold)) + "\n")
        print(f"GSM8K test: {len(ds)} problems")

    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
