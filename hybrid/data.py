"""Data pipeline (numpy only; torch is imported lazily so this module is testable without it).

Storage (written by prepare_data.py into data.dir):
  general_{train,val}.bin           flat token stream, documents wrapped as <bos> ... <eos>
  {src}_{train,val}_tokens.bin      reasoning samples, flat tokens (no specials)
  {src}_{train,val}_index.npy       (N, 3) int64: token_start, seg_start, n_segs
  {src}_{train,val}_segs.npy        int32 segment lengths; segment 0 = prompt, 1.. = reasoning steps
  eval_*.jsonl                      {"question", "answer", ...} for accuracy evaluation

Hybrid sample construction (pretraining analogue of the paper's App. E, without LLM annotators):
  middle reasoning steps become latent with probability `latent_prob` (curriculum-ramped); the first
  and last step stay text (paper prompt: "always start and end the think block with plain text");
  adjacent latent steps merge into one span; spans shorter than `min_span_tokens` stay text.
  A latent span of n tokens becomes  <latent> SLOT x ceil(n/k) </latent>.
"""
import json
import random
import re
import threading
import queue
from pathlib import Path

import numpy as np

PROMPT_TEMPLATE = "Question: {q}\nAnswer:\n"


def format_prompt(question):
    return PROMPT_TEMPLATE.format(q=question.strip())


# --------------------------------------------------------------------------- step segmentation
_SENT_SPLIT = re.compile(r"(?<=[.!?])(\s+)(?=[A-Z\\$])")


def segment_solution(text, max_chars=280):
    """Split a solution into reasoning steps: lines, and long lines into sentences.
    Guarantees "".join(steps) == text (whitespace stays attached to the preceding step)."""
    pieces = []
    for line in text.splitlines(keepends=True):
        if len(line) <= max_chars:
            pieces.append(line)
            continue
        parts, cur = _SENT_SPLIT.split(line), ""
        for i, p in enumerate(parts):
            cur += p
            if i % 2 == 1:  # captured whitespace closes a sentence
                pieces.append(cur)
                cur = ""
        if cur:
            pieces.append(cur)
    out = []
    for p in pieces:
        if out and not p.strip():
            out[-1] += p
        else:
            out.append(p)
    return out


# --------------------------------------------------------------------------- synthetic chains
_VARS = list("abcdefghjkmnpqrstuvwxyz")


def make_chain_problem(rng, depth, n_distract=0, max_value=9999):
    """Variable-chain arithmetic with controllable depth (number of dependent steps) and
    distractor definitions. Small models can learn this from scratch, so it gives a clean,
    controllable testbed for latent reasoning depth (cf. ProsQA/ProntoQA in Coconut)."""
    names = rng.sample(_VARS, depth + 1 + n_distract)
    chain, distract = names[: depth + 1], names[depth + 1:]
    vals = {chain[0]: rng.randint(1, 20)}
    defs = [f"Let {chain[0]} = {vals[chain[0]]}."]
    steps = []
    for i in range(1, depth + 1):
        prev, nm = chain[i - 1], chain[i]
        op = rng.choice("+-*")
        if i >= 2 and rng.random() < 0.4:
            other = rng.choice(chain[: i - 1])
            otxt, oval = other, vals[other]
        else:
            oval = rng.randint(2, 9) if op == "*" else rng.randint(1, 30)
            otxt = str(oval)
        a = vals[prev]
        if op == "*" and a * oval > max_value:
            op = "+"
        if op == "-" and a - oval < 0:
            op = "+"
        res = a + oval if op == "+" else a - oval if op == "-" else a * oval
        vals[nm] = res
        defs.append(f"Let {nm} = {prev} {op} {otxt}.")
        steps.append(f"{nm} = {a} {op} {oval} = {res}.\n")
    for dn in distract:  # irrelevant definitions inserted after their dependency
        ref = rng.choice(chain)
        pos = next(t for t, s in enumerate(defs) if s.startswith(f"Let {ref} ="))
        defs.insert(rng.randint(pos + 1, len(defs)), f"Let {dn} = {ref} + {rng.randint(1, 30)}.")
    target = chain[-1]
    question = " ".join(defs) + f" What is {target}?"
    steps = [f"We need to find {target}.\n"] + steps + [f"So the answer is \\boxed{{{vals[target]}}}.\n"]
    return question, steps, str(vals[target])


# --------------------------------------------------------------------------- binary storage
class ReasoningWriter:
    def __init__(self, prefix, dtype):
        self.prefix, self.dtype = str(prefix), dtype
        self.f = open(self.prefix + "_tokens.bin", "wb")
        self.index, self.segs, self.ntok = [], [], 0

    def add(self, seg_token_lists):
        lens = [len(s) for s in seg_token_lists]
        self.index.append((self.ntok, len(self.segs), len(lens)))
        self.segs.extend(lens)
        arr = np.fromiter((t for s in seg_token_lists for t in s), dtype=self.dtype, count=sum(lens))
        arr.tofile(self.f)
        self.ntok += int(arr.size)

    def close(self):
        self.f.close()
        np.save(self.prefix + "_index.npy", np.asarray(self.index, dtype=np.int64).reshape(-1, 3))
        np.save(self.prefix + "_segs.npy", np.asarray(self.segs, dtype=np.int32))
        return len(self.index), self.ntok


class ReasoningData:
    def __init__(self, prefix, dtype):
        prefix = str(prefix)
        self.tokens = np.memmap(prefix + "_tokens.bin", dtype=dtype, mode="r")
        self.index = np.load(prefix + "_index.npy")
        self.segs = np.load(prefix + "_segs.npy")

    def __len__(self):
        return len(self.index)

    def get(self, i):
        start, s0, n = self.index[i]
        lens = self.segs[s0:s0 + n]
        out, pos = [], int(start)
        for L in lens:
            out.append(np.asarray(self.tokens[pos:pos + L], dtype=np.int64))
            pos += int(L)
        return out


def build_hybrid_items(segs, sp, k, latent_prob, rng, hcfg, text_only=False):
    """Turn one sample (list of token arrays) into item ids (L,) and slot groups (L, k)."""
    prompt, steps = segs[0], segs[1:]
    n = len(steps)
    latent = np.zeros(n, dtype=bool)
    if not text_only and latent_prob > 0 and n > 0:
        lo = 1 if hcfg.get("keep_first_step_text", True) else 0
        hi = n - 1 if hcfg.get("keep_last_step_text", True) else n
        for j in range(lo, hi):
            if rng.random() < latent_prob:
                latent[j] = True

    ids, grps = [np.array([sp.bos], dtype=np.int64), prompt], []
    grps.append(np.full((1 + len(prompt), k), -1, dtype=np.int64))
    min_span = hcfg.get("min_span_tokens", 8)
    j = 0
    while j < n:
        if latent[j]:
            e = j
            while e < n and latent[e]:
                e += 1
            span = np.concatenate(steps[j:e])
            if len(span) >= min_span:
                n_slots = -(-len(span) // k)
                g = np.full((n_slots * k,), -1, dtype=np.int64)
                g[: len(span)] = span
                ids.append(np.array([sp.lat_open] + [sp.slot] * n_slots + [sp.lat_close], dtype=np.int64))
                grps.append(np.concatenate([np.full((1, k), -1, dtype=np.int64), g.reshape(n_slots, k),
                                            np.full((1, k), -1, dtype=np.int64)]))
            else:
                ids.append(span)
                grps.append(np.full((len(span), k), -1, dtype=np.int64))
            j = e
        else:
            ids.append(steps[j])
            grps.append(np.full((len(steps[j]), k), -1, dtype=np.int64))
            j += 1
    ids.append(np.array([sp.eos], dtype=np.int64))
    grps.append(np.full((1, k), -1, dtype=np.int64))
    return np.concatenate(ids), np.concatenate(grps)


class ReasoningStream:
    """Packs hybrid samples back-to-back into rows of S+1 items (samples may span rows)."""

    def __init__(self, data, sp, k, hcfg, seed):
        self.data, self.sp, self.k, self.hcfg = data, sp, k, hcfg
        self.rng = random.Random(seed)
        self.latent_prob = 0.0
        self.hybrid = bool(hcfg.get("enabled", False))
        self._ids = np.zeros(0, dtype=np.int64)
        self._grp = np.zeros((0, k), dtype=np.int64)

    def next_row(self, n):
        while len(self._ids) < n:
            segs = self.data.get(self.rng.randrange(len(self.data)))
            text_only = (not self.hybrid) or self.rng.random() < self.hcfg.get("text_only_prob", 0.0)
            i, g = build_hybrid_items(segs, self.sp, self.k, self.latent_prob, self.rng, self.hcfg, text_only)
            self._ids = np.concatenate([self._ids, i])
            self._grp = np.concatenate([self._grp, g])
        ids, grp = self._ids[:n], self._grp[:n]
        self._ids, self._grp = self._ids[n:], self._grp[n:]
        return ids, grp


class GeneralStream:
    def __init__(self, path, dtype, k, seed):
        self.tokens = np.memmap(path, dtype=dtype, mode="r")
        self.k, self.rng = k, np.random.default_rng(seed)

    def next_row(self, n):
        s = int(self.rng.integers(0, len(self.tokens) - n - 1))
        return (np.asarray(self.tokens[s:s + n], dtype=np.int64), np.full((n, self.k), -1, dtype=np.int64))


class BatchBuilder:
    """Draws each row from a source chosen by the mixture weights."""

    def __init__(self, cfg, sp, split, seed):
        d = cfg["data"]
        root = Path(d["dir"])
        meta = json.loads((root / "meta.json").read_text())
        dtype = np.dtype(meta["dtype"])
        hcfg = dict(cfg["hybrid"])
        self.k = int(hcfg.get("compression", 1))
        self.seq = cfg["model"]["max_seq_len"]
        self.streams, weights = {}, []
        for i, (name, w) in enumerate(d["mix"].items()):
            if w <= 0 or not d.get(name, {}).get("enabled", True):
                continue
            if name == "general":
                self.streams[name] = GeneralStream(root / f"general_{split}.bin", dtype, self.k, seed + i)
            else:
                self.streams[name] = ReasoningStream(ReasoningData(root / f"{name}_{split}", dtype), sp,
                                                     self.k, hcfg, seed + i)
            weights.append(w)
        self.names = list(self.streams)
        self.weights = np.asarray(weights, dtype=np.float64) / sum(weights)
        self.rng = np.random.default_rng(seed + 999)

    def set_latent_prob(self, p):
        for s in self.streams.values():
            if isinstance(s, ReasoningStream):
                s.latent_prob = p

    def next(self, batch_size, only=None):
        n = self.seq + 1
        ids = np.empty((batch_size, n), dtype=np.int64)
        grp = np.empty((batch_size, n, self.k), dtype=np.int64)
        for b in range(batch_size):
            name = only or self.names[self.rng.choice(len(self.names), p=self.weights)]
            ids[b], grp[b] = self.streams[name].next_row(n)
        return ids, grp


class Prefetcher:
    """Background thread that builds numpy batches and converts them to (pinned) torch tensors."""

    def __init__(self, builder, batch_size, depth=4, pin=False):
        import torch

        self.torch, self.builder, self.bs, self.pin = torch, builder, batch_size, pin
        self.q = queue.Queue(maxsize=depth)
        self.stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self.stop:
            ids, grp = self.builder.next(self.bs)
            ids, grp = self.torch.from_numpy(ids), self.torch.from_numpy(grp)
            if self.pin:
                ids, grp = ids.pin_memory(), grp.pin_memory()
            self.q.put((ids, grp))

    def get(self):
        return self.q.get()


def load_jsonl(path, limit=None):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if limit and len(out) >= limit:
                    break
    return out
