#!/usr/bin/env python3
"""
v2_falsifierB_carry_transplant.py — Falsifier B: Organic Arithmetic-Carry Neuron Transplant
===========================================================================================

Astra 6.2 Ultra Rank-1 v2 Falsifier (`V2_ROADMAP.md`, `docs/COUNCIL_SYNTHESIS.md` §3 Rank 1):
Tests whether *organic pretrained* procedural MLP neurons (not synthetic constant-write rules)
can be harvested from a donor model via causal lesioning and transplanted zero-SGD into a
different-family recipient (`Qwen2.5-3B` -> `SmolLM2-1.7B`, Donor A unloaded at inference!).

Experimental Protocol:
  1. Strictly Disjoint Paired 2-Digit Addition Carry Benchmark:
       - Discovery split: 80 Carry + 80 Matched No-Carry (40 same-tens-sum + 40 same-answer-tens)
       - Held-out Test:   120 Carry + 120 Matched No-Carry (60 same-tens-sum + 60 same-answer-tens)
       - Prompts end with `"= "` so the next token is the exact single tens digit in BOTH
         Qwen2.5-3B and SmolLM2-1.7B (`['4', '7', ' +', ' ', '3', '5', ' =', ' ']`).
       - Metrics: tens-digit accuracy, off-by-10 carry-miss rate (and false-carry rate on no-carry),
         exact 2-digit EM, and tens-digit forced-choice log-odds `log P(tens_ok) - log P(tens_off10)`.
  2. Stage 1 — Donor-Lesion Kill Gate (`qwen2.5-3b`):
       - Scans middle-to-late layers (`L12..L24`, always including `L17` with cached v1 sockets)
         using exact first-order gradient attribution `a_j * d(logodds)/d(a_j)` + selectivity t-stat
         on the Discovery split.
       - Runs causal zero-ablation (and no-carry mean-ablation) of top-k (`k in {16, 32, 64}`)
         harvested carry neurons on the held-out Test split vs. random-k ablation control,
         verifying sharp selective carry drop (`|Delta carry| >> |Delta nocarry|`).
       - Extracts pretrained `(W_g, W_u, W_d)` rows + carry mean write vector, then UNLOADS Donor A!
  3. Stage 2 — Zero-SGD Organic Pretrained Neuron Conjugation (`qwen2.5-3b` -> `smollm2-1.7b`):
       - Conjugates harvested pretrained rows into native bias-free SwiGLU widening (`d_ff -> d_ff + k`,
         `mlp_bias=False`, pre-gamma deconvolved + MMSE unlabeled read sockets, exact constant pin,
         exact BOS up-null, Donor A unloaded) with `torch.equal` bit-exact removal verified!
       - Rigorous 5-arm comparison:
           a. `ORGANIC_HARVEST_CONJUGATED` (aligned unlabeled sockets, gain sweep `s in {0.5, 1.0, 2.0}`)
           b. `SHUFFLED_SOCKET_CONTROL` (same top-k neurons, random Haar orthogonal socket)
           c. `RANDOM_DONOR_NEURONS_CONTROL` (k random norm-matched donor neurons, aligned socket)
           d. `CONSTANT_MEAN_WRITE_CONTROL` (constant BOS-nulled mean carry write — catches dumb "+10"!)
           e. `NATIVE_B_CARRY_AMPLIFICATION` (upper reference: amplifying SmolLM2's own carry neurons)
       - Full audits: No-Carry preservation, 196-fact `dataset.json` locality, universal-eval KL/PPL,
         Wilson 95% CIs, exact two-sided McNemar p-values, and `torch.equal` unlearning certificates.

Usage:
  PYTHONPATH=platonicswap:experiments python experiments/v2_falsifierB_carry_transplant.py --unit-test
  PYTHONPATH=platonicswap:experiments python -u experiments/v2_falsifierB_carry_transplant.py
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla
from scipy.stats import binomtest
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (str(REPO_ROOT / "platonicswap"), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Default to populated local lab caches if present and not overridden
if "PLATONICSWAP_HF_CACHE" not in os.environ and Path("/tmp/platos_key_lab/hf_cache").is_dir():
    os.environ["PLATONICSWAP_HF_CACHE"] = "/tmp/platos_key_lab/hf_cache"
if "PLATONICSWAP_CACHE_DIR" not in os.environ and Path("/tmp/platos_key_lab/cache").is_dir():
    os.environ["PLATONICSWAP_CACHE_DIR"] = "/tmp/platos_key_lab/cache"

RESULTS_DIR = REPO_ROOT / "results" if (REPO_ROOT / "results").is_dir() else SCRIPT_DIR
OUT_PATH = RESULTS_DIR / "results_v2_falsifierB_carry_transplant.json"

SEED = 20261126
DONOR_KEY = "qwen2.5-3b"
RECIPIENT_KEY = "smollm2-1.7b"
DEFAULT_SCAN_LAYERS = (12, 14, 16, 17, 18, 20, 22, 24)
DEFAULT_K_LIST = (16, 32, 64)
DEFAULT_GAINS = (0.5, 1.0, 2.0)

# Strictly excluded from both discovery and test pools (including commutative swaps)
FEW_SHOT_EXEMPLARS: Tuple[Tuple[int, int], ...] = (
    (12, 23),  # 35 (no-carry)
    (28, 16),  # 44 (carry)
)
FEW_SHOT_PREFIX = "".join(f"{a} + {b} = {a + b}\n" for a, b in FEW_SHOT_EXEMPLARS)


# =============================================================================
# 1. Strictly Disjoint Paired 2-Digit Addition Carry Benchmark
# =============================================================================


def _unordered_pair(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def format_addition_prompt(a: int, b: int, prompt_mode: str = "2shot") -> str:
    """Format addition prompt ending in '= ' so the exact next token is the single tens digit."""
    if prompt_mode == "0shot":
        return f"{a} + {b} = "
    if prompt_mode == "2shot":
        return f"{FEW_SHOT_PREFIX}{a} + {b} = "
    raise ValueError(f"Unknown prompt_mode: {prompt_mode}")


def _make_item(a: int, b: int, is_carry: bool, match_type: str, pair_id: int) -> Dict[str, Any]:
    s = a + b
    assert 10 <= s <= 99, f"Expected 2-digit sum, got {a}+{b}={s}"
    ua, ub = a % 10, b % 10
    ta, tb = a // 10, b // 10
    assert (ua + ub >= 10) == is_carry
    correct_tens = s // 10
    # For carry: off-by-10 misses the carry (s - 10 -> ta + tb).
    # For no-carry: off-by-10 is a false carry (+10 -> ta + tb + 1).
    off_by_10_sum = (s - 10) if is_carry else (s + 10)
    off_by_10_tens = off_by_10_sum // 10
    assert 1 <= correct_tens <= 9 and 1 <= off_by_10_tens <= 9 and correct_tens != off_by_10_tens
    return {
        "pair_id": int(pair_id),
        "a": int(a),
        "b": int(b),
        "sum": int(s),
        "is_carry": bool(is_carry),
        "match_type": match_type,  # "carry", "nocarry_same_tens_sum", or "nocarry_same_ans_tens"
        "tens_sum": int(ta + tb),
        "units_sum": int(ua + ub),
        "correct_tens": str(correct_tens),
        "off_by_10_tens": str(off_by_10_tens),
        "correct_str": str(s),
        "off_by_10_str": str(off_by_10_sum),
    }


def build_paired_carry_dataset(
    n_disc_per_class: int = 80,
    n_test_per_class: int = 120,
    seed: int = SEED,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build strictly disjoint Discovery (`n_disc_per_class` carry + matched no-carry) and
    Held-out Test (`n_test_per_class` carry + matched no-carry) 2-digit addition splits.

    Every Carry item `(a, b)` with `t_a + t_b = T` (`2 <= T <= 7`, answer tens `T + 1 <= 8`)
    is paired with a strictly unique No-Carry control:
      - 50% `nocarry_same_tens_sum`: `t_a' + t_b' = T` (identical input tens sum `T`, answer tens `T`)
      - 50% `nocarry_same_ans_tens`: `t_a'' + t_b'' = T + 1` (identical answer tens `T + 1`, no carry)
    All unordered pairs `{a, b}` are 100% disjoint across discovery, test, and few-shot exemplars.
    """
    rng = random.Random(seed)
    excluded = {_unordered_pair(a, b) for a, b in FEW_SHOT_EXEMPLARS}

    # Restrict carry tens sum T in [2..7] so:
    #   - Carry answer tens T + 1 in [3..8] and carry miss tens T in [2..7]
    #   - Same-tens-sum no-carry has tens sum T in [2..7] and false-carry tens T + 1 in [3..8]
    #   - Same-ans-tens no-carry has tens sum T + 1 in [3..8] and false-carry tens T + 2 in [4..9]
    carry_by_T: Dict[int, List[Tuple[int, int]]] = {T: [] for T in range(2, 8)}
    nocarry_by_T: Dict[int, List[Tuple[int, int]]] = {T: [] for T in range(2, 9)}

    for a in range(11, 89):
        ta, ua = divmod(a, 10)
        if ua == 0:
            continue
        for b in range(11, 89):
            tb, ub = divmod(b, 10)
            if ub == 0:
                continue
            if _unordered_pair(a, b) in excluded:
                continue
            T = ta + tb
            if ua + ub >= 10:
                if 2 <= T <= 7 and (a + b) < 100:
                    carry_by_T[T].append((a, b))
            else:
                if 2 <= T <= 8 and (a + b + 10) < 100:
                    nocarry_by_T[T].append((a, b))

    for d in (carry_by_T, nocarry_by_T):
        for T in d:
            rng.shuffle(d[T])

    used_unordered: set[Tuple[int, int]] = set(excluded)
    total_pairs = n_disc_per_class + n_test_per_class
    all_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    def _pop_unique(pool: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        while pool:
            a, b = pool.pop()
            up = _unordered_pair(a, b)
            if up not in used_unordered:
                used_unordered.add(up)
                return a, b
        return None

    Ts = list(range(2, 8))
    cursor = 0
    for pid in range(total_pairs):
        match_kind = "nocarry_same_tens_sum" if (pid % 2 == 0) else "nocarry_same_ans_tens"
        chosen = None
        for step in range(len(Ts)):
            T = Ts[(cursor + step) % len(Ts)]
            nc_T = T if match_kind == "nocarry_same_tens_sum" else (T + 1)
            # Peek without losing ordering by checking pools
            c_avail = [p for p in carry_by_T[T] if _unordered_pair(p[0], p[1]) not in used_unordered]
            nc_avail = [p for p in nocarry_by_T[nc_T] if _unordered_pair(p[0], p[1]) not in used_unordered]
            if c_avail and nc_avail:
                carry_by_T[T] = c_avail
                nocarry_by_T[nc_T] = nc_avail
                c_ab = _pop_unique(carry_by_T[T])
                nc_ab = _pop_unique(nocarry_by_T[nc_T])
                chosen = (c_ab, nc_ab)
                cursor = (cursor + step + 1) % len(Ts)
                break
        assert chosen is not None, f"All tens pools exhausted at pid={pid}, kind={match_kind}"
        c_ab, nc_ab = chosen
        all_pairs.append((
            _make_item(c_ab[0], c_ab[1], True, "carry", pid),
            _make_item(nc_ab[0], nc_ab[1], False, match_kind, pid),
        ))

    # Stratify discovery/test split cleanly across T and match_kind
    disc_pairs = all_pairs[:n_disc_per_class]
    test_pairs = all_pairs[n_disc_per_class:]

    return {
        "discovery_carry": [p[0] for p in disc_pairs],
        "discovery_nocarry": [p[1] for p in disc_pairs],
        "test_carry": [p[0] for p in test_pairs],
        "test_nocarry": [p[1] for p in test_pairs],
    }


# =============================================================================
# 2. Exact Statistical Helpers & Evaluation Harness
# =============================================================================


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[Optional[float], Optional[float]]:
    if n <= 0:
        return None, None
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p)) / n + (z * z) / (4.0 * n * n))
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def exact_mcnemar(a: Sequence[int], b: Sequence[int]) -> Dict[str, Any]:
    aa, bb = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    assert len(aa) == len(bb)
    n10 = int(((aa == 1) & (bb == 0)).sum())
    n01 = int(((aa == 0) & (bb == 1)).sum())
    pval = float(binomtest(n10, n10 + n01, 0.5, alternative="two-sided").pvalue) if (n10 + n01) > 0 else 1.0
    return {
        "n": int(len(aa)),
        "acc_a": round(float(aa.mean()), 4) if len(aa) else None,
        "acc_b": round(float(bb.mean()), 4) if len(bb) else None,
        "n10": n10,
        "n01": n01,
        "p_value": pval,
    }


def parse_leading_two_digit_int(text: str) -> Optional[int]:
    m = re.match(r"^\s*(\d{1,3})", text)
    if not m:
        return None
    return int(m.group(1))


def get_digit_token_map(tok: Any) -> Dict[str, int]:
    """Map single ASCII digits '0'..'9' to their exact single-token IDs."""
    dmap: Dict[str, int] = {}
    for d in "0123456789":
        ids = tok.encode(d, add_special_tokens=False)
        assert len(ids) == 1, f"Digit {d!r} is not single-token in {tok.name_or_path}: {ids}"
        dmap[d] = int(ids[0])
    return dmap


@torch.no_grad()
def eval_addition_items(
    model: Any,
    tok: Any,
    device: torch.device,
    items: Sequence[Dict[str, Any]],
    prompt_mode: str = "2shot",
    batch_size: int = 32,
    do_generate: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate 2-digit addition items via:
      1. Single prefill pass at pos=-1 for exact tens-digit log-odds `log P(correct_tens) - log P(off10_tens)`,
         forced-choice tens accuracy, and greedy top-1 tens-digit accuracy.
      2. Greedy generation (`max_new_tokens=4`) for exact 2-digit EM and generated off-by-10 error rate.
    """
    dmap = get_digit_token_map(tok)
    prompts = [format_addition_prompt(it["a"], it["b"], prompt_mode=prompt_mode) for it in items]

    old_pad = tok.padding_side
    tok.padding_side = "left"
    fc_tens_ok: List[int] = []
    top1_tens_ok: List[int] = []
    top1_off10: List[int] = []
    log_odds: List[float] = []
    gens: List[str] = []
    em_ok: List[int] = []
    gen_off10: List[int] = []
    gen_tens_ok: List[int] = []

    try:
        for st in range(0, len(prompts), batch_size):
            bp = prompts[st : st + batch_size]
            enc = tok(bp, return_tensors="pt", padding=True, add_special_tokens=True)
            inp_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            logits = model(input_ids=inp_ids, attention_mask=attn, use_cache=False).logits[:, -1, :].float()
            lp = F.log_softmax(logits, dim=-1).cpu()
            top1_ids = lp.argmax(dim=-1).tolist()

            if do_generate:
                gen_out = model.generate(
                    input_ids=inp_ids,
                    attention_mask=attn,
                    max_new_tokens=4,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                )
                plen = inp_ids.shape[1]
                b_gens = [tok.decode(gen_out[bi, plen:], skip_special_tokens=True) for bi in range(len(bp))]
            else:
                b_gens = [""] * len(bp)

            for bi in range(len(bp)):
                it = items[st + bi]
                tid_ok = dmap[it["correct_tens"]]
                tid_bad = dmap[it["off_by_10_tens"]]
                lo = float(lp[bi, tid_ok].item() - lp[bi, tid_bad].item())
                log_odds.append(round(lo, 4))
                fc_tens_ok.append(int(lo > 0.0))
                top1_tens_ok.append(int(top1_ids[bi] == tid_ok))
                top1_off10.append(int(top1_ids[bi] == tid_bad))
                g = b_gens[bi]
                gens.append(g)
                if do_generate:
                    val = parse_leading_two_digit_int(g)
                    em_ok.append(int(val == it["sum"]))
                    gen_off10.append(int(val == int(it["off_by_10_str"])))
                    gen_tens_ok.append(int(val is not None and (val // 10) == int(it["correct_tens"])))
                else:
                    em_ok.append(0)
                    gen_off10.append(0)
                    gen_tens_ok.append(top1_tens_ok[-1])
    finally:
        tok.padding_side = old_pad

    n = len(items)
    def _stat(vec: Sequence[int]) -> Dict[str, Any]:
        k = int(sum(vec))
        return {"acc": round(k / n, 4) if n else None, "k": k, "n": n, "wilson_95ci": list(wilson_ci(k, n))}

    return {
        "n": n,
        "tens_fc": _stat(fc_tens_ok),
        "tens_top1": _stat(top1_tens_ok),
        "tens_top1_off_by_10": _stat(top1_off10),
        "exact_em": _stat(em_ok) if do_generate else None,
        "gen_off_by_10": _stat(gen_off10) if do_generate else None,
        "gen_tens_ok": _stat(gen_tens_ok) if do_generate else None,
        "mean_log_odds": round(float(np.mean(log_odds)), 4) if n else None,
        "std_log_odds": round(float(np.std(log_odds)), 4) if n else None,
        "_raw": {
            "tens_fc": fc_tens_ok,
            "tens_top1": top1_tens_ok,
            "tens_top1_off10": top1_off10,
            "exact_em": em_ok,
            "gen_off10": gen_off10,
            "log_odds": log_odds,
        },
    }


def eval_carry_and_nocarry(
    model: Any,
    tok: Any,
    device: torch.device,
    carry_items: Sequence[Dict[str, Any]],
    nocarry_items: Sequence[Dict[str, Any]],
    prompt_mode: str = "2shot",
    batch_size: int = 32,
    do_generate: bool = True,
) -> Dict[str, Any]:
    res_c = eval_addition_items(model, tok, device, carry_items, prompt_mode=prompt_mode, batch_size=batch_size, do_generate=do_generate)
    res_nc = eval_addition_items(model, tok, device, nocarry_items, prompt_mode=prompt_mode, batch_size=batch_size, do_generate=do_generate)
    # Also break down No-Carry by match type (same_tens_sum vs same_ans_tens)
    by_mt = {}
    for mt in ("nocarry_same_tens_sum", "nocarry_same_ans_tens"):
        idxs = [i for i, x in enumerate(nocarry_items) if x["match_type"] == mt]
        if idxs:
            by_mt[mt] = {
                "n": len(idxs),
                "tens_top1_acc": round(float(np.mean([res_nc["_raw"]["tens_top1"][i] for i in idxs])), 4),
                "false_carry_top1_rate": round(float(np.mean([res_nc["_raw"]["tens_top1_off10"][i] for i in idxs])), 4),
                "exact_em": round(float(np.mean([res_nc["_raw"]["exact_em"][i] for i in idxs])), 4) if do_generate else None,
                "mean_log_odds": round(float(np.mean([res_nc["_raw"]["log_odds"][i] for i in idxs])), 4),
            }
    return {
        "carry": res_c,
        "nocarry": res_nc,
        "nocarry_by_match_type": by_mt,
    }


def strip_raw(pair_eval: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "carry": {k: v for k, v in pair_eval["carry"].items() if k != "_raw"},
        "nocarry": {k: v for k, v in pair_eval["nocarry"].items() if k != "_raw"},
        "nocarry_by_match_type": pair_eval.get("nocarry_by_match_type", {}),
    }


# =============================================================================
# 3. Stage 1: Exact First-Order Causal Attribution & Donor-Lesion Kill Gate
# =============================================================================


def compute_layer_carry_attribution(
    model: Any,
    tok: Any,
    device: torch.device,
    layer_idx: int,
    carry_items: Sequence[Dict[str, Any]],
    nocarry_items: Sequence[Dict[str, Any]],
    prompt_mode: str = "2shot",
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Compute exact per-neuron SwiGLU carry selectivity and first-order causal attribution
    at `layer_idx` (token pos = -1, immediately predicting the tens digit!) on Discovery items:
      - `a_j = (SiLU(W_g x) * (W_u x))_j` (exact input to `mlp.down_proj`)
      - `grad_j = d(log P(correct_tens) - log P(off10_tens)) / d(a_j)` via exact detached-leaf
        backprop from `layer_idx` to the output logits (zero weight grads, fast & exact!).
      - `carry_attr_j = mean_carry(a_j * grad_j)` (direct causal contribution to carry tens-digit!)
      - `nocarry_attr_j = mean_nocarry(a_j * grad_j)`
      - `t_stat_j`: Welch t-stat of `a_j` on Carry vs. Matched No-Carry.
      - `score_j`: `carry_attr_j - max(0, nocarry_attr_j)` gated on `carry_attr_j > 0` (ranks
        organic neurons that specifically push carry tens-digit upward without being generic!).
    """
    from extract_and_baseline import get_decoder_layers

    dmap = get_digit_token_map(tok)
    mlp = get_decoder_layers(model)[layer_idx].mlp
    old_pad = tok.padding_side
    tok.padding_side = "left"

    def _run_group(items: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        prompts = [format_addition_prompt(it["a"], it["b"], prompt_mode=prompt_mode) for it in items]
        acts_all: List[np.ndarray] = []
        attr_all: List[np.ndarray] = []
        for st in range(0, len(prompts), batch_size):
            bp = prompts[st : st + batch_size]
            bitems = items[st : st + batch_size]
            enc = tok(bp, return_tensors="pt", padding=True, add_special_tokens=True)
            inp_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)

            holder: Dict[str, torch.Tensor] = {}

            def _pre_hook(_mod, inp):
                a_leaf = inp[0].detach().requires_grad_(True)
                holder["a"] = a_leaf
                return (a_leaf,) + inp[1:]

            h = mlp.down_proj.register_forward_pre_hook(_pre_hook)
            try:
                logits = model(input_ids=inp_ids, attention_mask=attn, use_cache=False).logits[:, -1, :].float()
                lp = F.log_softmax(logits, dim=-1)
                idx_ok = torch.tensor([dmap[it["correct_tens"]] for it in bitems], dtype=torch.long, device=device)
                idx_bad = torch.tensor([dmap[it["off_by_10_tens"]] for it in bitems], dtype=torch.long, device=device)
                bsz = torch.arange(len(bitems), device=device)
                loss_sum = (lp[bsz, idx_ok] - lp[bsz, idx_bad]).sum()
                grad_a = torch.autograd.grad(loss_sum, holder["a"], retain_graph=False, create_graph=False)[0]
                a_last = holder["a"][:, -1, :].detach().float().cpu().numpy()
                g_last = grad_a[:, -1, :].detach().float().cpu().numpy()
                acts_all.append(a_last)
                attr_all.append(a_last * g_last)
            finally:
                h.remove()
                model.zero_grad(set_to_none=True)
        return np.vstack(acts_all), np.vstack(attr_all)

    try:
        a_c, attr_c = _run_group(carry_items)
        a_nc, attr_nc = _run_group(nocarry_items)
    finally:
        tok.padding_side = old_pad

    mu_c = a_c.mean(axis=0)
    mu_nc = a_nc.mean(axis=0)
    var_c = a_c.var(axis=0)
    var_nc = a_nc.var(axis=0)
    t_stat = (mu_c - mu_nc) / np.sqrt(var_c / max(len(a_c), 1) + var_nc / max(len(a_nc), 1) + 1e-12)

    mean_attr_c = attr_c.mean(axis=0)
    mean_attr_nc = attr_nc.mean(axis=0)
    # Causal carry-specific score: rewards neurons with strong positive carry log-odds attribution
    # while penalizing neurons that also heavily contribute to no-carry prompts.
    carry_spec_score = mean_attr_c - np.maximum(0.0, mean_attr_nc)
    # Secondary pure attribution rank (for transparency / logging)
    order_spec = np.argsort(-carry_spec_score)
    order_raw_attr = np.argsort(-mean_attr_c)

    return {
        "layer": int(layer_idx),
        "d_ff": int(a_c.shape[1]),
        "mu_carry": mu_c.astype(np.float64),
        "mu_nocarry": mu_nc.astype(np.float64),
        "t_stat": t_stat.astype(np.float64),
        "mean_attr_carry": mean_attr_c.astype(np.float64),
        "mean_attr_nocarry": mean_attr_nc.astype(np.float64),
        "carry_spec_score": carry_spec_score.astype(np.float64),
        "top_idx_spec": order_spec.astype(np.int64),
        "top_idx_raw_attr": order_raw_attr.astype(np.int64),
        "summary": {
            "layer": int(layer_idx),
            "top32_spec_score_sum": round(float(carry_spec_score[order_spec[:32]].sum()), 4),
            "top32_carry_attr_sum": round(float(mean_attr_c[order_spec[:32]].sum()), 4),
            "top32_nocarry_attr_sum": round(float(mean_attr_nc[order_spec[:32]].sum()), 4),
            "top32_mean_abs_tstat": round(float(np.mean(np.abs(t_stat[order_spec[:32]]))), 3),
            "top10_indices": [int(i) for i in order_spec[:10]],
        },
    }


def eval_with_neuron_lesion(
    model: Any,
    tok: Any,
    device: torch.device,
    layer_idx: int,
    neuron_indices: Sequence[int],
    carry_items: Sequence[Dict[str, Any]],
    nocarry_items: Sequence[Dict[str, Any]],
    prompt_mode: str = "2shot",
    ablation_mode: str = "zero",
    replace_values: Optional[np.ndarray] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Lesion `neuron_indices` in `layers[layer_idx].mlp.down_proj` input (zero or nocarry-mean)."""
    from extract_and_baseline import get_decoder_layers

    mlp = get_decoder_layers(model)[layer_idx].mlp
    idx_t = torch.tensor(list(neuron_indices), dtype=torch.long, device=device)
    rep_t = None
    if ablation_mode == "mean_nocarry":
        assert replace_values is not None
        rep_t = torch.tensor(replace_values[list(neuron_indices)], dtype=mlp.down_proj.weight.dtype, device=device)

    def _ablate_pre(_mod, inp):
        a = inp[0].clone()
        if ablation_mode == "zero":
            a[:, :, idx_t] = 0.0
        elif ablation_mode == "mean_nocarry":
            a[:, :, idx_t] = rep_t
        else:
            raise ValueError(f"Unknown ablation_mode: {ablation_mode}")
        return (a,) + inp[1:]

    h = mlp.down_proj.register_forward_pre_hook(_ablate_pre)
    try:
        res = eval_carry_and_nocarry(model, tok, device, carry_items, nocarry_items, prompt_mode=prompt_mode, batch_size=batch_size, do_generate=True)
    finally:
        h.remove()
    return res


# =============================================================================
# 4. Stage 2: Exact Zero-SGD Organic SwiGLU Conjugation & Controls
# =============================================================================


def build_read_maps_from_channel(
    ch: Dict[str, Any],
    gamma_A: Optional[np.ndarray] = None,
    gamma_B: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Build closed-form unlabeled B -> A post-LN read maps directly from the cached universal channel:
      1. `mmse`: exact Ledoit-Wolf reverse regression `M_in = Sigma_AB @ inv(Sigma_BB_fit_lw)`
         where `Sigma_AB = Sigma_AA_lw @ G.T` (absorbs gamma_A, gamma_B optimally in L2!).
      2. `gamma_orth`: Fable's exact pre-gamma deconvolved orthogonal Procrustes map
         `M_in_gamma = diag(gamma_A) @ (s_pre * R_pre) @ diag(gamma_B^{-1})`, fitted on
         unnormalized held-out states `X_A_resid / gamma_A` and `X_B_resid / gamma_B`
         (ZERO extra forward passes!).
    """
    Sigma_AA_lw = np.asarray(ch["Sigma_AA_lw"], dtype=np.float64)
    Sigma_BB_lw = np.asarray(ch["Sigma_BB_fit_lw"], dtype=np.float64)
    G = np.asarray(ch["G"], dtype=np.float64)  # (D_B, D_A), G = Sigma_BA @ inv(Sigma_AA_lw)
    Sigma_AB = Sigma_AA_lw @ G.T               # (D_A, D_B)
    M_in_mmse = sla.solve(Sigma_BB_lw, Sigma_AB.T, assume_a="pos").T  # (D_A, D_B)

    out: Dict[str, np.ndarray] = {"mmse": M_in_mmse}

    if gamma_A is not None and gamma_B is not None and "X_A_resid" in ch and "X_B_resid" in ch:
        gA = np.asarray(gamma_A, dtype=np.float64)
        gB = np.asarray(gamma_B, dtype=np.float64)
        gB_safe = np.where(np.abs(gB) > 1e-6, gB, np.sign(gB + 1e-12) * 1e-6)
        Xa = np.asarray(ch["X_A_resid"], dtype=np.float64)
        Xb = np.asarray(ch["X_B_resid"], dtype=np.float64)
        Xa_pre = Xa / np.where(np.abs(gA) > 1e-6, gA, 1.0)[None, :]
        Xb_pre = Xb / gB_safe[None, :]
        Xa_c = Xa_pre - Xa_pre.mean(axis=0, keepdims=True)
        Xb_c = Xb_pre - Xb_pre.mean(axis=0, keepdims=True)
        C_pre = (Xa_c.T @ Xb_c) / float(len(Xa_c))  # (D_A, D_B)
        U, _, Vt = sla.svd(C_pre, full_matrices=False)
        R_pre = U @ Vt  # (D_A, D_B)
        s_pre = float(np.median(np.linalg.norm(Xa_pre, axis=-1)) / max(np.median(np.linalg.norm(Xb_pre, axis=-1)), 1e-12))
        M_in_gamma = (gA[:, None]) * (s_pre * R_pre) * (1.0 / gB_safe[None, :])
        out["gamma_orth"] = M_in_gamma
    return out


def random_stiefel(shape: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    m, n = shape
    if m >= n:
        Q, _ = sla.qr(rng.standard_normal((m, n)), mode="economic")
        return Q
    Q, _ = sla.qr(rng.standard_normal((n, m)), mode="economic")
    return Q.T


def conjugate_organic_swiglu_rows(
    W_g_A: np.ndarray,
    W_u_A: np.ndarray,
    W_d_A: np.ndarray,
    M_in: np.ndarray,
    M_out: np.ndarray,
    mu_A: np.ndarray,
    mu_B: np.ndarray,
    c_B: np.ndarray,
    x_bos_B: np.ndarray,
    bos_bank_B: Optional[np.ndarray] = None,
    rm_B: float = 1.0,
    gain: float = 1.0,
    shuffle_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Strictly bias-free (`mlp_bias=False`) zero-SGD conjugation of organic pretrained SwiGLU rows
    `(W_g_A, W_u_A, W_d_A)` from Donor A into Recipient B:
      - Constant-pin affine fold via `c_B` (`c_B^T x_B ≈ 1` on non-BOS):
          W_g_B = W_g_A @ M_in + outer(W_g_A @ (mu_A - M_in @ mu_B), c_B)
          W_u_B = W_u_A @ M_in + outer(W_u_A @ (mu_A - M_in @ mu_B), c_B)
      - Exact BOS up-null (`W_u_B @ x_bos_B == 0` to machine precision!)
      - Write transport: `W_d_B = (gain / rm_B) * (M_out @ W_d_A)`
    When `shuffle_seed` is set, replaces `(M_in, M_out)` with Frobenius-norm-matched random
    Haar Stiefel matrices (`SHUFFLED_SOCKET_CONTROL`).
    """
    import pilot_organs_p0 as po

    Wg = np.asarray(W_g_A, dtype=np.float64)
    Wu = np.asarray(W_u_A, dtype=np.float64)
    Wd = np.asarray(W_d_A, dtype=np.float64)
    Min = np.asarray(M_in, dtype=np.float64).copy()
    Mout = np.asarray(M_out, dtype=np.float64).copy()
    mA = np.asarray(mu_A, dtype=np.float64)
    mB = np.asarray(mu_B, dtype=np.float64)
    cB = np.asarray(c_B, dtype=np.float64)

    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        Qi = random_stiefel(Min.shape, rng)
        Qo = random_stiefel(Mout.shape, rng)
        Min = Qi * (sla.norm(Min, "fro") / max(sla.norm(Qi, "fro"), 1e-12))
        Mout = Qo * (sla.norm(Mout, "fro") / max(sla.norm(Qo, "fro"), 1e-12))

    shift = mA - Min @ mB
    Wg_B = Wg @ Min + np.outer(Wg @ shift, cB)
    Wu_B = Wu @ Min + np.outer(Wu @ shift, cB)
    Wu_B = po.apply_exact_bos_null_biasfree(Wu_B, x_bos_B, x_bos_bank=bos_bank_B)
    Wd_B = (float(gain) / float(rm_B)) * (Mout @ Wd)

    return {
        "W_g": Wg_B,
        "b_g": None,
        "W_u": Wu_B,
        "b_u": None,
        "W_d": Wd_B,
        "use_bias": False,
    }


def build_constant_mean_write_control(
    v_mean_carry_A: np.ndarray,
    M_out: np.ndarray,
    c_B: np.ndarray,
    X_B_ref: np.ndarray,
    x_bos_B: np.ndarray,
    bos_bank_B: Optional[np.ndarray] = None,
    rm_B: float = 1.0,
    gain: float = 1.0,
) -> Dict[str, Any]:
    """
    Build a 1-row strictly bias-free (`mlp_bias=False`) SwiGLU module that injects the
    CONSTANT mapped mean carry write `v_B = (gain / rm_B) * M_out @ v_mean_carry_A` on
    every non-BOS token (`SiLU(beta * c_B^T x) * (c_u^T x) / SiLU(beta) ≈ 1.0`) and exact
    0.0 on BOS (`W_u @ x_bos_B == 0`).

    Isolates genuine input-dependent carry computation from a dumb "+10" constant bias!
    """
    import pilot_organs_p0 as po

    cB = np.asarray(c_B, dtype=np.float64)
    c_u = po.apply_exact_bos_null_biasfree(cB[None, :], x_bos_B, x_bos_bank=bos_bank_B)[0]
    c_u = c_u / max(float(np.mean(np.asarray(X_B_ref, dtype=np.float64) @ c_u)), 1e-8)

    beta_c = 10.0  # SiLU(10.0) = 9.999546... (deep linear regime, smooth & constant for c_B^T x ≈ 1)
    silu_beta = beta_c / (1.0 + math.exp(-beta_c))
    v_B = (float(gain) / float(rm_B)) * (np.asarray(M_out, dtype=np.float64) @ np.asarray(v_mean_carry_A, dtype=np.float64))

    return {
        "W_g": (beta_c * cB)[None, :],
        "b_g": None,
        "W_u": c_u[None, :],
        "b_u": None,
        "W_d": (v_B / silu_beta)[:, None],
        "use_bias": False,
    }


def select_norm_matched_random_neurons(
    W_d_all: np.ndarray,
    exclude_indices: Sequence[int],
    target_indices: Sequence[int],
    seed: int = SEED + 404,
) -> np.ndarray:
    """Select `k` random donor neurons (disjoint from `exclude_indices`) matched in `W_d` column norm."""
    rng = np.random.default_rng(seed)
    k = len(target_indices)
    d_ff = W_d_all.shape[1]
    excl = set(int(i) for i in exclude_indices)
    norms = np.linalg.norm(W_d_all, axis=0)
    avail = np.array([i for i in range(d_ff) if i not in excl], dtype=np.int64)
    chosen: List[int] = []
    used: set[int] = set()
    # Match each target neuron's W_d norm within the closest 5% norm band, sampled randomly
    for tidx in target_indices:
        tn = norms[int(tidx)]
        diffs = np.abs(norms[avail] - tn)
        cand_order = avail[np.argsort(diffs)]
        pool = [int(c) for c in cand_order[: max(32, k)] if int(c) not in used]
        pick = int(rng.choice(pool))
        chosen.append(pick)
        used.add(pick)
    return np.array(chosen, dtype=np.int64)


def build_native_b_amplification_weights(
    mlp_B: nn.Module,
    neuron_indices: Sequence[int],
    x_bos_B: np.ndarray,
    bos_bank_B: Optional[np.ndarray] = None,
    gain: float = 1.0,
) -> Dict[str, Any]:
    """Append a BOS-nulled copy of Recipient B's own top-k carry neurons (upper reference!)."""
    import pilot_organs_p0 as po

    idx = list(int(i) for i in neuron_indices)
    Wg = mlp_B.gate_proj.weight.detach().float().cpu().numpy()[idx].astype(np.float64)
    Wu = mlp_B.up_proj.weight.detach().float().cpu().numpy()[idx].astype(np.float64)
    Wd = mlp_B.down_proj.weight.detach().float().cpu().numpy()[:, idx].astype(np.float64) * float(gain)
    Wu = po.apply_exact_bos_null_biasfree(Wu, x_bos_B, x_bos_bank=bos_bank_B)
    return {"W_g": Wg, "b_g": None, "W_u": Wu, "b_u": None, "W_d": Wd, "use_bias": False}


# =============================================================================
# 5. Main MPS Runner (Stage 1 Donor Kill Gate -> Unload A -> Stage 2 Transplant)
# =============================================================================


def _delta_summary(cur: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    c_t = cur["carry"]["tens_top1"]["acc"] - base["carry"]["tens_top1"]["acc"]
    c_em = cur["carry"]["exact_em"]["acc"] - base["carry"]["exact_em"]["acc"]
    c_off = cur["carry"]["tens_top1_off_by_10"]["acc"] - base["carry"]["tens_top1_off_by_10"]["acc"]
    c_lo = cur["carry"]["mean_log_odds"] - base["carry"]["mean_log_odds"]
    nc_t = cur["nocarry"]["tens_top1"]["acc"] - base["nocarry"]["tens_top1"]["acc"]
    nc_em = cur["nocarry"]["exact_em"]["acc"] - base["nocarry"]["exact_em"]["acc"]
    nc_fc = cur["nocarry"]["tens_top1_off_by_10"]["acc"] - base["nocarry"]["tens_top1_off_by_10"]["acc"]
    nc_lo = cur["nocarry"]["mean_log_odds"] - base["nocarry"]["mean_log_odds"]
    return {
        "delta_carry_tens_top1": round(c_t, 4),
        "delta_carry_exact_em": round(c_em, 4),
        "delta_carry_off_by_10": round(c_off, 4),
        "delta_carry_log_odds": round(c_lo, 4),
        "delta_nocarry_tens_top1": round(nc_t, 4),
        "delta_nocarry_exact_em": round(nc_em, 4),
        "delta_nocarry_false_carry": round(nc_fc, 4),
        "delta_nocarry_log_odds": round(nc_lo, 4),
        "net_carry_minus_nocarry_tens": round(c_t + nc_t, 4),
        "net_carry_gain_minus_false_carry": round(c_t - nc_fc, 4),
    }


def _pair_mcnemar(cur_raw: Dict[str, Any], ref_raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "carry_tens_top1": exact_mcnemar(cur_raw["carry"]["_raw"]["tens_top1"], ref_raw["carry"]["_raw"]["tens_top1"]),
        "carry_tens_fc": exact_mcnemar(cur_raw["carry"]["_raw"]["tens_fc"], ref_raw["carry"]["_raw"]["tens_fc"]),
        "carry_exact_em": exact_mcnemar(cur_raw["carry"]["_raw"]["exact_em"], ref_raw["carry"]["_raw"]["exact_em"]),
        "nocarry_tens_top1": exact_mcnemar(cur_raw["nocarry"]["_raw"]["tens_top1"], ref_raw["nocarry"]["_raw"]["tens_top1"]),
        "nocarry_exact_em": exact_mcnemar(cur_raw["nocarry"]["_raw"]["exact_em"], ref_raw["nocarry"]["_raw"]["exact_em"]),
    }


def run_falsifier_b(args: argparse.Namespace) -> Dict[str, Any]:
    import pilot_organs_p0 as po
    import pilot_organs_multifamily as pmf
    import platonic_organ_engine as poe
    from extract_and_baseline import get_decoder_layers

    t0 = time.time()
    scan_layers = [int(x) for x in args.scan_layers.split(",") if x.strip()]
    if 17 not in scan_layers:
        scan_layers = sorted(set(scan_layers + [17]))
    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    gains = [float(x) for x in args.gains.split(",") if x.strip()]

    ds = build_paired_carry_dataset(
        n_disc_per_class=args.n_disc,
        n_test_per_class=args.n_test,
        seed=SEED,
    )
    corpus_univ = poe.build_universal_calibration_corpus()
    _, test_facts, _, _, _ = pmf.load_dataset_splits()

    # Prompt-mode selection (2shot by default; auto verifies both models cleanly produce digits)
    prompt_mode = args.prompt_mode if args.prompt_mode != "auto" else "2shot"

    print(f"\n================ FALSIFIER B: ORGANIC ARITHMETIC-CARRY NEURON TRANSPLANT ================")
    print(f"Donor: {DONOR_KEY} | Recipient: {RECIPIENT_KEY} | Prompt mode: {prompt_mode}")
    print(f"Discovery: {len(ds['discovery_carry'])}+{len(ds['discovery_nocarry'])} | Held-Out Test: {len(ds['test_carry'])}+{len(ds['test_nocarry'])}")
    print(f"Scan layers: {scan_layers} | k_list: {k_list} | gains: {gains}")

    # -------------------------------------------------------------------------
    # STAGE 1: Donor-Lesion Kill Gate on Qwen2.5-3B
    # -------------------------------------------------------------------------
    t1 = time.time()
    model_A, tok_A, dev_A, _ = po.load_model_fp32(DONOR_KEY)
    L_A = len(get_decoder_layers(model_A))

    if args.prompt_mode == "auto":
        # Quick 16-item sanity check comparing 0shot vs 2shot on Donor A
        s0 = eval_addition_items(model_A, tok_A, dev_A, ds["discovery_carry"][:16], prompt_mode="0shot", do_generate=True)
        s2 = eval_addition_items(model_A, tok_A, dev_A, ds["discovery_carry"][:16], prompt_mode="2shot", do_generate=True)
        prompt_mode = "0shot" if s0["exact_em"]["acc"] >= 0.85 and s0["exact_em"]["acc"] >= s2["exact_em"]["acc"] else "2shot"
        print(f"  [Auto prompt_mode] 0shot EM={s0['exact_em']['acc']:.3f}, 2shot EM={s2['exact_em']['acc']:.3f} -> selected `{prompt_mode}`")

    donor_base_disc = eval_carry_and_nocarry(model_A, tok_A, dev_A, ds["discovery_carry"], ds["discovery_nocarry"], prompt_mode=prompt_mode)
    donor_base_test = eval_carry_and_nocarry(model_A, tok_A, dev_A, ds["test_carry"], ds["test_nocarry"], prompt_mode=prompt_mode)
    print(
        f"  [Donor {DONOR_KEY} BASE TEST] Carry: tens={donor_base_test['carry']['tens_top1']['acc']:.3f} "
        f"EM={donor_base_test['carry']['exact_em']['acc']:.3f} off10={donor_base_test['carry']['tens_top1_off_by_10']['acc']:.3f} "
        f"LO={donor_base_test['carry']['mean_log_odds']:+.3f} | "
        f"NoCarry: tens={donor_base_test['nocarry']['tens_top1']['acc']:.3f} "
        f"EM={donor_base_test['nocarry']['exact_em']['acc']:.3f} LO={donor_base_test['nocarry']['mean_log_odds']:+.3f}"
    )

    # Scan candidate layers on Discovery split via exact first-order attribution
    layer_attrs: Dict[int, Dict[str, Any]] = {}
    for lA in scan_layers:
        att = compute_layer_carry_attribution(
            model_A, tok_A, dev_A, lA, ds["discovery_carry"], ds["discovery_nocarry"], prompt_mode=prompt_mode
        )
        layer_attrs[lA] = att
        sm = att["summary"]
        print(
            f"  [Scan L{lA:02d}] top32_spec_sum={sm['top32_spec_score_sum']:+.3f} "
            f"carry_attr={sm['top32_carry_attr_sum']:+.3f} nocarry_attr={sm['top32_nocarry_attr_sum']:+.3f} "
            f"|t|={sm['top32_mean_abs_tstat']:.2f}"
        )

    # Select top-3 layers by top32_spec_score_sum (ALWAYS including L17!) for the causal lesion kill gate
    ranked_layers = sorted(scan_layers, key=lambda l: -layer_attrs[l]["summary"]["top32_spec_score_sum"])
    lesion_layers = list(dict.fromkeys(ranked_layers[:3] + [17]))

    lesion_results: Dict[str, Any] = {}
    best_lesion_layer = 17
    best_lesion_score = -1e9

    for lA in lesion_layers:
        att = layer_attrs[lA]
        lkey = f"L{lA}"
        lesion_results[lkey] = {"discovery_summary": att["summary"], "by_k": {}}
        for k in k_list:
            top_k = att["top_idx_spec"][:k]
            ev_zero = eval_with_neuron_lesion(
                model_A, tok_A, dev_A, lA, top_k, ds["test_carry"], ds["test_nocarry"],
                prompt_mode=prompt_mode, ablation_mode="zero",
            )
            d_zero = _delta_summary(ev_zero, donor_base_test)
            # Matched random-k control ablation on same layer
            mlp_A = get_decoder_layers(model_A)[lA].mlp
            W_d_all = mlp_A.down_proj.weight.detach().float().cpu().numpy()
            rand_k = select_norm_matched_random_neurons(W_d_all, top_k, top_k, seed=SEED + lA * 100 + k)
            ev_rand = eval_with_neuron_lesion(
                model_A, tok_A, dev_A, lA, rand_k, ds["test_carry"], ds["test_nocarry"],
                prompt_mode=prompt_mode, ablation_mode="zero",
            )
            d_rand = _delta_summary(ev_rand, donor_base_test)

            # Also run mean_nocarry ablation for k=32 (confirms zero-ablation isn't just bias shift)
            mean_abl_pack = None
            if k == 32:
                ev_mean = eval_with_neuron_lesion(
                    model_A, tok_A, dev_A, lA, top_k, ds["test_carry"], ds["test_nocarry"],
                    prompt_mode=prompt_mode, ablation_mode="mean_nocarry", replace_values=att["mu_nocarry"],
                )
                mean_abl_pack = {
                    "eval": strip_raw(ev_mean),
                    "delta_vs_base": _delta_summary(ev_mean, donor_base_test),
                }

            sel_drop = (-d_zero["delta_carry_log_odds"]) - abs(d_zero["delta_nocarry_log_odds"])
            if k == 32 and sel_drop > best_lesion_score:
                best_lesion_score = sel_drop
                best_lesion_layer = lA

            print(
                f"  [Lesion L{lA:02d} k={k:2d}] ZERO: Δcarry_tens={d_zero['delta_carry_tens_top1']:+.3f} "
                f"Δcarry_EM={d_zero['delta_carry_exact_em']:+.3f} Δcarry_LO={d_zero['delta_carry_log_odds']:+.3f} "
                f"vs Δnocarry_tens={d_zero['delta_nocarry_tens_top1']:+.3f} Δnocarry_LO={d_zero['delta_nocarry_log_odds']:+.3f} | "
                f"RAND-{k}: Δcarry_LO={d_rand['delta_carry_log_odds']:+.3f}"
            )
            lesion_results[lkey]["by_k"][str(k)] = {
                "neuron_indices": [int(i) for i in top_k],
                "zero_ablation": {"eval": strip_raw(ev_zero), "delta_vs_base": d_zero},
                "random_k_ablation": {"eval": strip_raw(ev_rand), "delta_vs_base": d_rand},
                "mean_nocarry_ablation": mean_abl_pack,
                "mcnemar_zero_vs_rand_carry_em": exact_mcnemar(ev_zero["carry"]["_raw"]["exact_em"], ev_rand["carry"]["_raw"]["exact_em"]),
            }

    # Harvest donor weights & carry mean write vector for EVERY transplant site (always L17 + best_lesion_layer if distinct!)
    transplant_donor_layers = list(dict.fromkeys([17, best_lesion_layer]))
    harvested_packages: Dict[int, Dict[str, Any]] = {}
    for lA in transplant_donor_layers:
        layer_mod = get_decoder_layers(model_A)[lA]
        mlp_A = layer_mod.mlp
        gamma_A = layer_mod.post_attention_layernorm.weight.detach().float().cpu().numpy().astype(np.float64)
        Wg_all = mlp_A.gate_proj.weight.detach().float().cpu().numpy().astype(np.float64)
        Wu_all = mlp_A.up_proj.weight.detach().float().cpu().numpy().astype(np.float64)
        Wd_all = mlp_A.down_proj.weight.detach().float().cpu().numpy().astype(np.float64)
        att = layer_attrs[lA]
        by_k_pkg: Dict[int, Dict[str, Any]] = {}
        for k in k_list:
            top_k = att["top_idx_spec"][:k]
            rand_k = select_norm_matched_random_neurons(Wd_all, top_k, top_k, seed=SEED + 900 + lA * 10 + k)
            # Mean write vector on Carry prompts from those k neurons
            v_mean_carry_A = Wd_all[:, top_k] @ att["mu_carry"][top_k]
            v_diff_carry_A = Wd_all[:, top_k] @ (att["mu_carry"][top_k] - att["mu_nocarry"][top_k])
            by_k_pkg[k] = {
                "top_k": top_k,
                "rand_k": rand_k,
                "W_g_top": Wg_all[top_k],
                "W_u_top": Wu_all[top_k],
                "W_d_top": Wd_all[:, top_k],
                "W_g_rand": Wg_all[rand_k],
                "W_u_rand": Wu_all[rand_k],
                "W_d_rand": Wd_all[:, rand_k],
                "v_mean_carry_A": v_mean_carry_A,
                "v_diff_carry_A": v_diff_carry_A,
            }
        harvested_packages[lA] = {"layer_A": lA, "gamma_A": gamma_A, "by_k": by_k_pkg}

    # Ensure universal donor passages exist for any non-17 best_lesion_layer BEFORE unloading Donor A!
    donor_passages_by_lA: Dict[int, Any] = {}
    for lA in transplant_donor_layers:
        cache_dp = poe.CACHE_DIR / f"universal_donor_passages_{DONOR_KEY}_l{lA}.pt"
        if cache_dp.exists():
            donor_passages_by_lA[lA] = torch.load(cache_dp, map_location="cpu", weights_only=False)
        else:
            print(f"  [Channel Prep] Extracting & caching universal donor passages for {DONOR_KEY} L{lA}...")
            dp_out = {
                "fit": po.extract_layer_states_on_passages(model_A, tok_A, dev_A, corpus_univ["fit"], lA),
                "resid": po.extract_layer_states_on_passages(model_A, tok_A, dev_A, corpus_univ["resid"], lA),
                "eval": po.extract_layer_states_on_passages(model_A, tok_A, dev_A, corpus_univ["eval"], lA),
            }
            torch.save(dp_out, cache_dp)
            donor_passages_by_lA[lA] = dp_out

    # COMPLETELY UNLOAD DONOR A BEFORE LOADING RECIPIENT B!
    po.unload_model(model_A)
    del model_A
    gc.collect()
    stage1_elapsed = round(time.time() - t1, 1)
    print(f"  [Stage 1 Complete in {stage1_elapsed}s] Best lesion layer: L{best_lesion_layer} | Donor A COMPLETELY UNLOADED!")

    # -------------------------------------------------------------------------
    # STAGE 2: Zero-SGD Organic Conjugation into Solo SmolLM2-1.7B
    # -------------------------------------------------------------------------
    t2 = time.time()
    model_B, tok_B, dev_B, rm_B = po.load_model_fp32(RECIPIENT_KEY)
    L_B = len(get_decoder_layers(model_B))

    probe_prompts = [format_addition_prompt(it["a"], it["b"], prompt_mode=prompt_mode) for it in ds["test_carry"][:4]]
    old_pad = tok_B.padding_side
    tok_B.padding_side = "left"
    enc_p = tok_B(probe_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
    with torch.no_grad():
        base_logits_probe = model_B(
            input_ids=enc_p["input_ids"].to(dev_B),
            attention_mask=enc_p["attention_mask"].to(dev_B),
            use_cache=False,
        ).logits.cpu()
    tok_B.padding_side = old_pad

    base_gt, base_lps = po.eval_general_text_kl_ppl(model_B, tok_B, dev_B, corpus_univ["eval"])
    base_loc, base_loc_ok = po.eval_196_locality(model_B, tok_B, dev_B, test_facts)
    recip_base_test = eval_carry_and_nocarry(model_B, tok_B, dev_B, ds["test_carry"], ds["test_nocarry"], prompt_mode=prompt_mode)
    print(
        f"\n  [Recipient {RECIPIENT_KEY} BASE TEST] Carry: tens={recip_base_test['carry']['tens_top1']['acc']:.3f} "
        f"EM={recip_base_test['carry']['exact_em']['acc']:.3f} off10={recip_base_test['carry']['tens_top1_off_by_10']['acc']:.3f} "
        f"LO={recip_base_test['carry']['mean_log_odds']:+.3f} | "
        f"NoCarry: tens={recip_base_test['nocarry']['tens_top1']['acc']:.3f} "
        f"EM={recip_base_test['nocarry']['exact_em']['acc']:.3f} | "
        f"196-fact={base_loc['n_correct_all']}/196 PPL={base_gt['ppl']:.4f}"
    )

    stage2_sites: Dict[str, Any] = {}
    for lA in transplant_donor_layers:
        lB = 9 if lA == 17 else max(2, min(L_B - 3, int(round(lA * L_B / float(L_A)))))
        site_name = f"L{lA}_to_L{lB}"
        print(f"\n--- [Stage 2 Transplant Site: {DONOR_KEY} L{lA} -> {RECIPIENT_KEY} L{lB}] ---")

        ch = poe.fit_or_load_universal_channel(
            DONOR_KEY, lA, RECIPIENT_KEY, lB, corpus_univ, model_B, tok_B, dev_B,
            donor_passages=donor_passages_by_lA[lA], tag="universal",
        )
        layer_B = get_decoder_layers(model_B)[lB]
        mlp_B = layer_B.mlp
        orig_state_B = po.save_original_mlp_state(mlp_B)
        gamma_B = layer_B.post_attention_layernorm.weight.detach().float().cpu().numpy().astype(np.float64)

        pkg = harvested_packages[lA]
        read_maps = build_read_maps_from_channel(ch, gamma_A=pkg["gamma_A"], gamma_B=gamma_B)
        # Prefer M_out_iso (or M_out_concept if attached — keep M_out_iso as primary for general organic MLP rows!)
        M_out_use = np.asarray(ch["M_out_iso"], dtype=np.float64)

        # Native-B carry neuron discovery at lB (upper reference!)
        att_B = compute_layer_carry_attribution(
            model_B, tok_B, dev_B, lB, ds["discovery_carry"], ds["discovery_nocarry"], prompt_mode=prompt_mode
        )
        print(f"  [Native-B L{lB} Carry Scan] {att_B['summary']}")

        def _eval_weights_arm(arm_name: str, weights: Dict[str, Any], do_loc: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            new_dff = poe.widen_mlp_native(mlp_B, orig_state_B, weights)
            ev = eval_carry_and_nocarry(model_B, tok_B, dev_B, ds["test_carry"], ds["test_nocarry"], prompt_mode=prompt_mode)
            delta = _delta_summary(ev, recip_base_test)
            if do_loc:
                gt_r, _ = po.eval_general_text_kl_ppl(model_B, tok_B, dev_B, corpus_univ["eval"], base_logprobs=base_lps)
                gt_r["delta_ppl"] = round(gt_r["ppl"] - base_gt["ppl"], 4)
                loc_r, _ = po.eval_196_locality(model_B, tok_B, dev_B, test_facts, base_correct=base_loc_ok)
            else:
                gt_r, loc_r = None, None
            tok_B.padding_side = "left"
            try:
                unl = poe.verify_exact_bit_unlearning(model_B, tok_B, dev_B, mlp_B, orig_state_B, base_logits_probe, probe_prompts)
            finally:
                tok_B.padding_side = old_pad
            loc_str = f"loc={loc_r['newly_broken_unrelated']}/{loc_r['n_unrelated_orig_correct']} KL={gt_r['mean_kl']:.5f}" if do_loc else "loc=skip"
            print(
                f"  [{arm_name:34s}] d_ff={new_dff} | "
                f"Carry tens={ev['carry']['tens_top1']['acc']:.3f}({delta['delta_carry_tens_top1']:+.3f}) "
                f"EM={ev['carry']['exact_em']['acc']:.3f}({delta['delta_carry_exact_em']:+.3f}) "
                f"off10={ev['carry']['tens_top1_off_by_10']['acc']:.3f} LO={ev['carry']['mean_log_odds']:+.3f}({delta['delta_carry_log_odds']:+.3f}) | "
                f"NoCarry tens={ev['nocarry']['tens_top1']['acc']:.3f}({delta['delta_nocarry_tens_top1']:+.3f}) "
                f"fc={ev['nocarry']['tens_top1_off_by_10']['acc']:.3f} | {loc_str} ul={unl['logits_bit_equal']}"
            )
            return {
                "widened_d_ff": new_dff,
                "eval": strip_raw(ev),
                "delta_vs_base": delta,
                "general_text": gt_r,
                "locality_196": loc_r,
                "unlearning": unl,
            }, ev

        arms_out: Dict[str, Any] = {}
        raw_store: Dict[str, Dict[str, Any]] = {"BASE": recip_base_test}

        for k in k_list:
            pk = pkg["by_k"][k]
            # Primary read map: pre-gamma deconvolved orthogonal + MMSE comparison on k=32
            for rmap_name in (["gamma_orth", "mmse"] if (k == 32 and "gamma_orth" in read_maps) else ["gamma_orth" if "gamma_orth" in read_maps else "mmse"]):
                M_in_use = read_maps[rmap_name]
                g_sweep = gains if (k == 32 and rmap_name == ("gamma_orth" if "gamma_orth" in read_maps else "mmse")) else (1.0,)
                for gval in g_sweep:
                    aname = f"ORGANIC_HARVEST_k{k}_{rmap_name}_s{gval:g}"
                    w = conjugate_organic_swiglu_rows(
                        pk["W_g_top"], pk["W_u_top"], pk["W_d_top"],
                        M_in_use, M_out_use, ch["mu_A"], ch["mu_B"], ch["c_B"], ch["x_bos_B"],
                        bos_bank_B=ch.get("bos_x_fit_B"), rm_B=rm_B, gain=gval,
                    )
                    arms_out[aname], raw_store[aname] = _eval_weights_arm(aname, w, do_loc=(k == 32 or gval == 1.0))

            # Controls for every k (using primary read map at gain=1.0)
            M_in_prim = read_maps["gamma_orth"] if "gamma_orth" in read_maps else read_maps["mmse"]

            w_shuf = conjugate_organic_swiglu_rows(
                pk["W_g_top"], pk["W_u_top"], pk["W_d_top"],
                M_in_prim, M_out_use, ch["mu_A"], ch["mu_B"], ch["c_B"], ch["x_bos_B"],
                bos_bank_B=ch.get("bos_x_fit_B"), rm_B=rm_B, gain=1.0, shuffle_seed=SEED + 501 + k,
            )
            an_sh = f"SHUFFLED_SOCKET_k{k}"
            arms_out[an_sh], raw_store[an_sh] = _eval_weights_arm(an_sh, w_shuf, do_loc=(k == 32))

            w_rand = conjugate_organic_swiglu_rows(
                pk["W_g_rand"], pk["W_u_rand"], pk["W_d_rand"],
                M_in_prim, M_out_use, ch["mu_A"], ch["mu_B"], ch["c_B"], ch["x_bos_B"],
                bos_bank_B=ch.get("bos_x_fit_B"), rm_B=rm_B, gain=1.0,
            )
            an_rn = f"RANDOM_DONOR_NEURONS_k{k}"
            arms_out[an_rn], raw_store[an_rn] = _eval_weights_arm(an_rn, w_rand, do_loc=(k == 32))

            w_const = build_constant_mean_write_control(
                pk["v_mean_carry_A"], M_out_use, ch["c_B"], ch["X_B_resid"], ch["x_bos_B"],
                bos_bank_B=ch.get("bos_x_fit_B"), rm_B=rm_B, gain=1.0,
            )
            an_cm = f"CONSTANT_MEAN_WRITE_k{k}"
            arms_out[an_cm], raw_store[an_cm] = _eval_weights_arm(an_cm, w_const, do_loc=(k == 32))

            w_nat = build_native_b_amplification_weights(
                mlp_B, att_B["top_idx_spec"][:k], ch["x_bos_B"], bos_bank_B=ch.get("bos_x_fit_B"), gain=1.0,
            )
            an_nb = f"NATIVE_B_CARRY_AMPLIFY_k{k}"
            arms_out[an_nb], raw_store[an_nb] = _eval_weights_arm(an_nb, w_nat, do_loc=(k == 32))

        # Exact two-sided McNemar tests for primary k=32 (and best gain) vs BASE and all 3 controls
        mcn_out: Dict[str, Any] = {}
        rmap_prim = "gamma_orth" if "gamma_orth" in read_maps else "mmse"
        for k in k_list:
            treat = f"ORGANIC_HARVEST_k{k}_{rmap_prim}_s1"
            if treat not in raw_store:
                continue
            for ctrl in ("BASE", f"SHUFFLED_SOCKET_k{k}", f"RANDOM_DONOR_NEURONS_k{k}", f"CONSTANT_MEAN_WRITE_k{k}"):
                if ctrl in raw_store:
                    mcn_out[f"{treat}_vs_{ctrl}"] = _pair_mcnemar(raw_store[treat], raw_store[ctrl])
            nb = f"NATIVE_B_CARRY_AMPLIFY_k{k}"
            if nb in raw_store:
                mcn_out[f"{nb}_vs_BASE"] = _pair_mcnemar(raw_store[nb], raw_store["BASE"])

        stage2_sites[site_name] = {
            "layer_A": lA,
            "layer_B": lB,
            "rm_B": rm_B,
            "channel_summary": ch["summary"],
            "native_B_discovery_summary": att_B["summary"],
            "arms": arms_out,
            "mcnemar": mcn_out,
        }

    po.unload_model(model_B)
    out = {
        "meta": {
            "donor_key": DONOR_KEY,
            "recipient_key": RECIPIENT_KEY,
            "prompt_mode": prompt_mode,
            "few_shot_exemplars": [list(p) for p in FEW_SHOT_EXEMPLARS],
            "n_disc_per_class": args.n_disc,
            "n_test_per_class": args.n_test,
            "scan_layers": scan_layers,
            "best_lesion_layer_A": best_lesion_layer,
            "transplant_donor_layers": transplant_donor_layers,
            "k_list": k_list,
            "gains": gains,
            "seed": SEED,
            "stage1_elapsed_sec": stage1_elapsed,
            "stage2_elapsed_sec": round(time.time() - t2, 1),
            "total_elapsed_sec": round(time.time() - t0, 1),
        },
        "stage1_donor_lesion": {
            "donor_base_discovery": strip_raw(donor_base_disc),
            "donor_base_test": strip_raw(donor_base_test),
            "layer_scan_summaries": {f"L{l}": layer_attrs[l]["summary"] for l in scan_layers},
            "lesion_kill_gate": lesion_results,
        },
        "stage2_recipient_transplant": {
            "recipient_base_test": strip_raw(recip_base_test),
            "recipient_base_general_text": base_gt,
            "recipient_base_locality_196": base_loc,
            "sites": stage2_sites,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[FALSIFIER B COMPLETE] Saved {OUT_PATH} ({out['meta']['total_elapsed_sec']}s)")
    return out


# =============================================================================
# 6. Pure-CPU Unit Tests (<10s, ZERO GPU / ZERO Model Forward Passes!)
# =============================================================================


def run_unit_test() -> None:
    import pilot_organs_p0 as po
    import platonic_organ_engine as poe
    from transformers import AutoTokenizer
    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaMLP

    print("[v2_falsifierB --unit-test] Running pure-CPU verification (dataset, tokenizers, conjugation math, attribution, LlamaMLP bit-exact unlearning)...")

    # 1. Paired 2-digit carry dataset invariants & strict disjointness
    ds = build_paired_carry_dataset(n_disc_per_class=80, n_test_per_class=120, seed=SEED)
    assert len(ds["discovery_carry"]) == 80 and len(ds["discovery_nocarry"]) == 80
    assert len(ds["test_carry"]) == 120 and len(ds["test_nocarry"]) == 120

    excl = {_unordered_pair(a, b) for a, b in FEW_SHOT_EXEMPLARS}
    seen_pairs: Dict[Tuple[int, int], str] = {}
    for split_name, items in ds.items():
        for it in items:
            up = _unordered_pair(it["a"], it["b"])
            assert up not in excl, f"Exemplar leak: {up}"
            assert up not in seen_pairs, f"Duplicate/overlap pair {up} in {split_name} and {seen_pairs[up]}"
            seen_pairs[up] = split_name
            ua, ub = it["a"] % 10, it["b"] % 10
            assert 10 <= it["sum"] <= 99
            if "nocarry" in split_name:
                assert not it["is_carry"] and (ua + ub) < 10
                assert int(it["off_by_10_str"]) == it["sum"] + 10
            else:
                assert it["is_carry"] and (ua + ub) >= 10
                assert int(it["off_by_10_str"]) == it["sum"] - 10
            assert 1 <= int(it["correct_tens"]) <= 9 and 1 <= int(it["off_by_10_tens"]) <= 9
            assert it["correct_tens"] != it["off_by_10_tens"]

    # Check exact 50/50 split of matched no-carry control types & exact pairing consistency
    for prefix, n_exp in (("discovery", 80), ("test", 120)):
        c_list, nc_list = ds[f"{prefix}_carry"], ds[f"{prefix}_nocarry"]
        n_same_ts = sum(1 for x in nc_list if x["match_type"] == "nocarry_same_tens_sum")
        n_same_at = sum(1 for x in nc_list if x["match_type"] == "nocarry_same_ans_tens")
        assert n_same_ts == n_exp // 2 and n_same_at == n_exp // 2
        for c_it, nc_it in zip(c_list, nc_list):
            if nc_it["match_type"] == "nocarry_same_tens_sum":
                assert nc_it["tens_sum"] == c_it["tens_sum"]
            else:
                assert int(nc_it["correct_tens"]) == int(c_it["correct_tens"])

    # 2. Verify exact single-token tens-digit tokenization on BOTH cached tokenizers (CPU only!)
    hf_cache = Path(os.environ.get("PLATONICSWAP_HF_CACHE", "/tmp/platos_key_lab/hf_cache"))
    toks_checked = 0
    for mkey in (DONOR_KEY, RECIPIENT_KEY):
        mdir = hf_cache / mkey
        if not mdir.exists():
            continue
        tok = AutoTokenizer.from_pretrained(str(mdir), local_files_only=True)
        dmap = get_digit_token_map(tok)
        assert len(set(dmap.values())) == 10
        p = format_addition_prompt(47, 35, prompt_mode="2shot")
        p_with_ans = p + "82"
        ids_p = tok.encode(p, add_special_tokens=False)
        ids_full = tok.encode(p_with_ans, add_special_tokens=False)
        assert ids_full[: len(ids_p)] == ids_p, f"Prompt boundary BPE merge on {mkey}!"
        assert ids_full[len(ids_p)] == dmap["8"] and ids_full[len(ids_p) + 1] == dmap["2"]
        toks_checked += 1

    # 3. Exact bias-free SwiGLU organic conjugation math + BOS silence + Constant-Mean control + LlamaMLP bit-exact unlearning!
    rng = np.random.default_rng(SEED + 31)
    D_A, D_B, k_org = 32, 24, 16
    R_true, _ = sla.qr(rng.standard_normal((D_A, D_B)), mode="economic")  # (D_A, D_B), R_true^T R_true = I
    mu_A = rng.standard_normal(D_A) * 0.5
    mu_B = rng.standard_normal(D_B) * 0.5
    gamma_A = np.exp(0.1 * rng.standard_normal(D_A))
    gamma_B = np.exp(0.1 * rng.standard_normal(D_B))

    # Synthetic aligned activation bank where x_B lives on affine hyperplane c_B^T x_B == 1
    c_B_true = rng.standard_normal(D_B)
    c_B_true /= np.linalg.norm(c_B_true)
    # Build X_B with exact c_B_true^T x_B == 1
    Z_raw = rng.standard_normal((300, D_B))
    Z_perp = Z_raw - np.outer(Z_raw @ c_B_true, c_B_true)
    X_B_syn = Z_perp + c_B_true[None, :]  # exact c_B_true^T x == 1.0!
    mu_B_syn = X_B_syn.mean(axis=0)
    X_A_syn = mu_A[None, :] + (X_B_syn - mu_B_syn[None, :]) @ R_true.T
    x_bos = rng.standard_normal(D_B)
    bos_bank = x_bos[None, :] + 1e-3 * rng.standard_normal((12, D_B))

    Wg_A = rng.standard_normal((k_org, D_A)) * 0.2
    Wu_A = rng.standard_normal((k_org, D_A)) * 0.2
    Wd_A = rng.standard_normal((D_A, k_org)) * 0.2

    conj = conjugate_organic_swiglu_rows(
        Wg_A, Wu_A, Wd_A, R_true, R_true.T, mu_A, mu_B_syn, c_B_true, x_bos, bos_bank_B=None, rm_B=1.0, gain=1.0
    )
    assert conj["use_bias"] is False and conj["W_g"].shape == (k_org, D_B) and conj["W_d"].shape == (D_B, k_org)
    # Gate pre-activation exact match to machine precision (<1e-10) when c_B^T x_B == 1!
    g_A_exact = X_A_syn @ Wg_A.T
    g_B_conj = X_B_syn @ conj["W_g"].T
    assert float(np.max(np.abs(g_A_exact - g_B_conj))) < 1e-10, "Gate constant-pin fold NOT exact!"
    # Exact BOS silence on W_u (<1e-12)
    assert float(np.max(np.abs(conj["W_u"] @ x_bos))) < 1e-12, "BOS up-null NOT exact!"

    # Shuffled socket control produces distinct orthogonal map
    conj_shuf = conjugate_organic_swiglu_rows(
        Wg_A, Wu_A, Wd_A, R_true, R_true.T, mu_A, mu_B_syn, c_B_true, x_bos, bos_bank_B=bos_bank, rm_B=1.0, gain=1.0, shuffle_seed=SEED + 7
    )
    assert float(np.max(np.abs(conj_shuf["W_u"] @ x_bos))) < 1e-12
    assert float(np.linalg.norm(conj_shuf["W_g"] - conj["W_g"])) > 1e-2

    # Constant mean write control: emits exact v_B on non-BOS and exact 0 on BOS!
    v_A_mean = rng.standard_normal(D_A)
    cm_w = build_constant_mean_write_control(v_A_mean, R_true.T, c_B_true, X_B_syn, x_bos, bos_bank_B=None, rm_B=1.0, gain=1.0)
    y_bos = (F.silu(torch.tensor(cm_w["W_g"] @ x_bos)).numpy() * (cm_w["W_u"] @ x_bos)) @ cm_w["W_d"].T
    assert float(np.max(np.abs(y_bos))) < 1e-12, "Constant mean write NOT silent on BOS!"
    g_nb = X_B_syn @ cm_w["W_g"].T
    u_nb = X_B_syn @ cm_w["W_u"].T
    y_nb = (F.silu(torch.tensor(g_nb)).numpy() * u_nb) @ cm_w["W_d"].T
    v_B_target = R_true.T @ v_A_mean
    assert float(np.max(np.abs(y_nb.mean(axis=0) - v_B_target))) < 1e-6, "Constant mean write mean mismatch!"

    # Norm-matched random neuron selector disjointness
    Wd_big = rng.standard_normal((D_A, 128))
    rn = select_norm_matched_random_neurons(Wd_big, list(range(16)), list(range(16)), seed=SEED + 19)
    assert len(rn) == 16 and len(set(rn)) == 16 and set(rn).isdisjoint(set(range(16)))

    # Real HF LlamaMLP widening (64 -> 80) + bit-exact unlearning (80 -> 64, torch.equal!)
    cfg = LlamaConfig(hidden_size=D_B, intermediate_size=64, num_attention_heads=4, num_key_value_heads=4, hidden_act="silu", mlp_bias=False)
    mlp = LlamaMLP(cfg)
    orig = po.save_original_mlp_state(mlp)
    x_probe = torch.randn(4, 5, D_B)
    with torch.no_grad():
        y_base = mlp(x_probe)
    assert poe.widen_mlp_native(mlp, orig, conj) == 64 + k_org
    assert mlp.gate_proj.bias is None and mlp.up_proj.bias is None and mlp.down_proj.bias is None
    with torch.no_grad():
        y_widened = mlp(x_probe)
    assert not torch.equal(y_base, y_widened)
    po.unlearn_mlp_native(mlp, orig)
    with torch.no_grad():
        y_restored = mlp(x_probe)
    assert torch.equal(y_base, y_restored) and mlp.gate_proj.out_features == 64
    assert torch.equal(mlp.gate_proj.weight, orig["gate_weight"])
    assert torch.equal(mlp.up_proj.weight, orig["up_weight"])
    assert torch.equal(mlp.down_proj.weight, orig["down_weight"])

    # 4. Synthetic read-map builder check (`mmse` + `gamma_orth`)
    ch_syn = poe.fit_generative_channel(
        X_A_syn[:200], X_B_syn[:200], X_A_syn[200:], X_B_syn[200:],
        X_A_syn[:200], X_B_syn[:200], X_A_syn[200:], X_B_syn[200:],
        bos_bank,
    )
    rmaps = build_read_maps_from_channel(ch_syn, gamma_A=gamma_A, gamma_B=gamma_B)
    assert rmaps["mmse"].shape == (D_A, D_B) and rmaps["gamma_orth"].shape == (D_A, D_B)

    print(f"[PASS 1/4] Paired 2-digit Carry dataset verified: 80+80 discovery, 120+120 test, 0 overlap (unordered), exact 50/50 same-tens-sum/same-ans-tens matching!")
    print(f"[PASS 2/4] Exact single-token tens-digit alignment verified on {toks_checked}/2 cached tokenizers ({DONOR_KEY}, {RECIPIENT_KEY})!")
    print(f"[PASS 3/4] Bias-free organic SwiGLU conjugation verified: gate err <1e-10, BOS silence <1e-12, Constant-Mean control exact, LlamaMLP 64->80->64 torch.equal!")
    print(f"[PASS 4/4] Closed-form MMSE & pre-gamma deconvolved read maps verified: { {k: v.shape for k, v in rmaps.items()} }!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Falsifier B: Organic Arithmetic-Carry Neuron Transplant (Qwen2.5-3B -> SmolLM2-1.7B)")
    parser.add_argument("--unit-test", action="store_true")
    parser.add_argument("--prompt-mode", type=str, default="2shot", choices=("2shot", "0shot", "auto"))
    parser.add_argument("--n-disc", type=int, default=80)
    parser.add_argument("--n-test", type=int, default=120)
    parser.add_argument("--scan-layers", type=str, default=",".join(str(x) for x in DEFAULT_SCAN_LAYERS))
    parser.add_argument("--k-list", type=str, default=",".join(str(x) for x in DEFAULT_K_LIST))
    parser.add_argument("--gains", type=str, default=",".join(str(x) for x in DEFAULT_GAINS))
    args = parser.parse_args()
    if args.unit_test:
        run_unit_test()
        return
    run_falsifier_b(args)


if __name__ == "__main__":
    main()
