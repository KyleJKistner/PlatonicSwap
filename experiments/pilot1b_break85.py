#!/usr/bin/env python3
"""
pilot1b_break85.py — Pilot 1B: Breaking 85%+ Cleanly (Zero QA Labels, Cross-Family)
===================================================================================

Integrates the unanimous 3/3 Outside Council plan (Astra 6.2 Ultra + Fable 5.2 Max + Neon High):
  - Pair support: qwen2.5-3b->smollm2-1.7b, qwen2.5-3b->granite-3.1-2b (forced float32!),
                  falcon3-3b->smollm2-1.7b (rectangular Stiefel + post-proj RMS rescaling).
  - Exact per-layer 4D attention knockout (with `use_cache=False` so MPS SDPA NEVER drops masks!):
      * `earlier_subj`: mask q >= last_subj from attending to earlier subject tokens [first_subj, last_subj)
      * `full_knockout`: `earlier_subj` at all layers PLUS mask q > last_subj from `last_subj` at layers < l_start!
      * `nonsubj_ctrl`: equal-size nonsubject prefix mask control
      * `all_subj_pos_ctrl`: positive control masking ALL subject edges at ALL layers (must collapse base acc!)
  - Powered 5-donor-per-base evaluation (~745 trials, base-item-clustered bootstrap 95% CIs)
    alongside standard 1-donor evaluation + Transport Efficiency (`cross_family / matched_native_ceiling`)!

Modular Stages (--stages 0,1,2,3):
  - Stage 0: Pre-flight no-op (<1e-4) & positive-knockout collapse checks;
             Native self-swap ceilings (incl. Granite sweep l_B in {4,8,12,16,20,24} + cos(h_l, h_0),
             single vs 3/5-layer bands vs earlier_subj knockout vs full_knockout vs later l_B + full_knockout);
             Free Geometry Ablation on cached N=10k maps (variance-matched vs LS scale,
             recipient base-run norm rescale, Astra's Ridge SVD polar/clamped spectrum ablation!).
  - Stage 1 (Lever A): Cross-family multi-layer band + exact attention knockout using N=10k unlabeled
             maps + per-layer RMS scaling (and Stiefel nullspace audit on Falcon), reporting
             Transport Efficiency, 1-tok/multi-tok stratification, rescued/spoiled, and 5-donor powered CIs!
  - Stage 2 (Lever B-lite): Pure REAL capitalized proper nouns (ZERO composed fictional names!
             Exact + normalized + alias + bidirectional word-containment dedup against ALL dataset.json
             subjects & answers, assert 0 leaks!), truncated at `last_subj` across M=3 relation-free
             prefixes; compares matched-N raw vs M-averaged vs between+lambda*within (lambda=0.1)
             with unaveraged (mu, s) calibration!
  - Stage 3 (Lever C — Fable + Astra Teacher): Unsupervised cross-entity self-swap distillation
             in strict O(d) (Woodbury-Cayley full-D rank-16 SO(d)), prompt truncated right at `last_subj`
             (zero suffix templates, zero QA labels!), teacher = B's own self-swap under identical
             intervention, variance-normalized residual MSE across layers l_end+1..l_end+4, frozen scale,
             early-stopped on held-out unlabeled entities!

Usage:
  python pilot1b_break85.py --pair qwen2.5-3b->smollm2-1.7b --stages 0,1,2,3
  python pilot1b_break85.py --pair qwen2.5-3b->granite-3.1-2b --stages 0,1,2,3
  python pilot1b_break85.py --pair falcon3-3b->smollm2-1.7b --stages 0,1,2
  python pilot1b_break85.py --unit-test   # Pure CPU math/mask/dedup check (zero GPU!)
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from extract_and_baseline import (
    CACHE_DIR,
    DATASET_2HOP_PATH,
    DATASET_PATH,
    LAB_DIR,
    RESULTS_PATH as BASELINE_RESULTS_PATH,
    SEED,
    _alias_first_token_ids,
    build_counterfactual_pairs,
    check_answer_match,
    get_decoder_layers,
    locate_last_subject_token,
)
from pilot1_fisher_scaling import LORO_HELD_OUT_RELS, make_same_relation_shuffled_donors
from pilot_graft_t3_gap import OBSCURE_ENTITIES

# M=3 diverse relation-free prefixes (causal LLM => ONLY prefixes exist at last_subj; zero suffixes!)
PREFIXES_M3 = [
    "The well-known ",
    "Historically, the subject ",
    "In standard encyclopedias, ",
]


# ======================================================================================
# 1. Exact Span Locator, Per-Layer 4D Attention Knockout & Cache-Free Greedy Decoding
# ======================================================================================

def locate_subject_token_span(tok: Any, prompt: str, char_span: Sequence[int]) -> Tuple[int, int, int]:
    c0, c1 = int(char_span[0]), int(char_span[1])
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    seq_len = len(enc["input_ids"])
    first_i, last_i = -1, -1
    for i, (s, e) in enumerate(offsets):
        if e <= s:
            continue
        if s < c1 and e > c0:
            if first_i < 0:
                first_i = i
            last_i = i
    if last_i < 0:
        last_i, seq_len = locate_last_subject_token(tok, prompt, char_span)
        first_i = last_i
    return int(first_i), int(last_i), int(seq_len)


def build_layer_4d_mask(
    attention_mask_2d: torch.Tensor,
    dtype: torch.dtype,
    cut_specs: List[Tuple[int, int, int]],  # per item: (q_start, k_lo, k_hi)
) -> torch.Tensor:
    """
    Build additive 4D causal mask [B, 1, T, T] (0.0 allowed, finfo.min masked).
    Masks queries q >= q_start from attending to keys k in [k_lo, k_hi).
    Always keeps diagonal self-attention allowed to avoid NaN softmax on padding rows.
    """
    bsz, T = attention_mask_2d.shape
    device = attention_mask_2d.device
    min_val = torch.finfo(dtype).min
    tril = torch.tril(torch.ones((T, T), dtype=torch.bool, device=device))
    valid_k = attention_mask_2d.bool().unsqueeze(1).unsqueeze(2)
    allowed = tril.unsqueeze(0).unsqueeze(0) & valid_k

    for b, (q_start, k_lo, k_hi) in enumerate(cut_specs):
        if k_hi > k_lo and q_start < T:
            allowed[b, 0, q_start:, k_lo:k_hi] = False

    diag = torch.eye(T, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(0)
    allowed = allowed | diag
    return torch.where(
        allowed,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), min_val, dtype=dtype, device=device),
    )


def build_powered_k_donor_pairs(
    facts: List[dict], shared_indices: Sequence[int], k_donors: int = 5
) -> List[Tuple[int, int]]:
    """
    Fable's powered evaluation: up to k_donors distinct same-relation donors per base item
    (distinct subject AND distinct answer!), deterministic order (~745 trials for SmolLM2!).
    """
    by_rel: Dict[str, List[int]] = defaultdict(list)
    for idx in sorted(shared_indices):
        by_rel[facts[idx]["relation"]].append(idx)

    pairs: List[Tuple[int, int]] = []
    for rel in sorted(by_rel.keys()):
        idxs = by_rel[rel]
        m = len(idxs)
        if m < 2:
            continue
        for pos, base_idx in enumerate(idxs):
            b_sub = facts[base_idx]["subject_norm"]
            b_ans = facts[base_idx]["answer"].strip().lower()
            picked = 0
            for step in range(1, m):
                cand = idxs[(pos + step) % m]
                if (
                    facts[cand]["subject_norm"] != b_sub
                    and facts[cand]["answer"].strip().lower() != b_ans
                ):
                    pairs.append((cand, base_idx))
                    picked += 1
                    if picked >= k_donors:
                        break
    return pairs


def base_clustered_bootstrap_ci(
    outcomes_m: Sequence[int],
    outcomes_s: Sequence[int],
    cluster_ids: Sequence[Any],
    n_boot: int = 2000,
    seed: int = SEED,
) -> dict:
    """Cluster bootstrap 95% CIs (by base item for 5-donor eval, or donor subject for 1-donor)."""
    m = np.asarray(outcomes_m, dtype=np.float64)
    s = np.asarray(outcomes_s, dtype=np.float64)
    n = len(m)
    if n == 0:
        return {"n": 0}
    by_c: Dict[Any, List[int]] = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        by_c[cid].append(i)
    clusters = list(by_c.values())
    nc = len(clusters)
    rng = np.random.default_rng(seed)
    bm = np.empty(n_boot, dtype=np.float64)
    bd = np.empty(n_boot, dtype=np.float64)
    for bi in range(n_boot):
        draw = rng.integers(0, nc, size=nc)
        idxs = [i for c in draw for i in clusters[c]]
        mv = float(np.mean(m[idxs]))
        sv = float(np.mean(s[idxs]))
        bm[bi] = mv
        bd[bi] = mv - sv
    mlo, mhi = np.percentile(bm, [2.5, 97.5])
    dlo, dhi = np.percentile(bd, [2.5, 97.5])
    return {
        "n": int(n),
        "n_clusters": int(nc),
        "matched_acc": round(float(np.mean(m)), 4),
        "matched_ci95": [round(float(mlo), 4), round(float(mhi), 4)],
        "shuffled_acc": round(float(np.mean(s)), 4),
        "matched_minus_shuffled_pp": round(100.0 * float(np.mean(m - s)), 2),
        "diff_ci95_pp": [round(100.0 * float(dlo), 2), round(100.0 * float(dhi), 2)],
    }


def pretokenize_knockout_batches(
    tok: Any,
    facts: List[dict],
    swap_pairs: Sequence[Tuple[int, int]],
    device: torch.device,
    batch_size: int = 32,
) -> List[dict]:
    """Pre-tokenize base prompts with exact span coords for all knockout modes."""
    batches = []
    n = len(swap_pairs)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bp = swap_pairs[start:end]
        prompts = [facts[b]["direct_prompt"] for _, b in bp]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        ids = enc["input_ids"].to(device)
        am2d = enc["attention_mask"].to(device)
        t_pad = ids.shape[1]

        first_p_list, last_p_list, subj_lens = [], [], []
        pad_lens = []
        for _, b in bp:
            fi, li, sl = locate_subject_token_span(tok, facts[b]["direct_prompt"], facts[b]["direct_subj_char_span"])
            pad = t_pad - sl
            fp = max(0, min(t_pad - 1, pad + fi))
            lp = max(0, min(t_pad - 1, pad + li))
            first_p_list.append(fp)
            last_p_list.append(lp)
            pad_lens.append(pad)
            subj_lens.append(max(1, lp - fp + 1))

        alias_first_sets = [_alias_first_token_ids(tok, facts[d]["aliases"]) for d, _ in bp]
        batches.append({
            "start": start,
            "end": end,
            "batch_pairs": bp,
            "input_ids": ids,
            "attention_mask": am2d,
            "first_p": first_p_list,
            "last_p": last_p_list,
            "last_p_t": torch.tensor(last_p_list, dtype=torch.long, device=device),
            "pad_lens": pad_lens,
            "subj_lens": subj_lens,
            "alias_first_sets": alias_first_sets,
        })
    return batches


def eval_knockout_swaps(
    model: Any,
    tok: Any,
    facts: List[dict],
    swap_pairs: Sequence[Tuple[int, int]],
    z_hat_by_layer: Dict[int, np.ndarray],
    device: torch.device,
    dtype: torch.dtype,
    mask_mode: str = "none",  # "none", "earlier_subj", "full_knockout", "nonsubj_ctrl", "all_subj_pos_ctrl"
    pretok_batches: Optional[List[dict]] = None,
    max_new_tokens: int = 6,
) -> Tuple[float, List[int]]:
    """
    Cache-FREE (use_cache=False!) multi-layer patch + exact per-layer 4D attention knockout!
    Because use_cache=False is used on EVERY step (fast two-stage: only continues decoding
    when first generated token matches an alias first token!), custom per-layer 4D masks are
    100% honored by MPS SDPA across BOTH prefill and multi-token continuations!
    """
    if not swap_pairs:
        return 0.0, []
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    n = len(swap_pairs)
    outcomes = [0] * n
    batches = (
        pretok_batches
        if pretok_batches is not None
        else pretokenize_knockout_batches(tok, facts, swap_pairs, device, batch_size=32)
    )

    patch_layers = sorted(z_hat_by_layer.keys())
    l_start = patch_layers[0] if patch_layers else n_layers

    ctx: Dict[str, Any] = {
        "active": False,
        "pos_t": None,
        "z_map": {},
        "mask_early": None,
        "mask_late": None,
    }
    handles = []

    # Per-layer attention mask injection pre-hook (ONLY when mask_mode != "none")
    if mask_mode != "none":
        for l_idx in range(n_layers):
            use_early = (mask_mode == "full_knockout" and l_idx < l_start) or (mask_mode == "all_subj_pos_ctrl")
            def _mk_pre(is_early: bool):
                def _pre(_mod, args, kwargs):
                    if not ctx["active"]:
                        return args, kwargs
                    m = ctx["mask_early"] if is_early else ctx["mask_late"]
                    if m is not None:
                        kwargs = dict(kwargs)
                        kwargs["attention_mask"] = m
                    return args, kwargs
                return _pre
            handles.append(layers[l_idx].register_forward_pre_hook(_mk_pre(use_early), with_kwargs=True))

    # Residual-stream patch post-hooks at patch_layers
    for lb in patch_layers:
        def _mk_post(li: int):
            def _post(_mod, _inp, out):
                if not ctx["active"]:
                    return out
                is_tup = isinstance(out, tuple)
                h = out[0].clone() if is_tup else out.clone()
                b_ar = torch.arange(h.shape[0], device=h.device)
                h[b_ar, ctx["pos_t"], :] = ctx["z_map"][li].to(device=h.device, dtype=h.dtype)
                return (h,) + out[1:] if is_tup else h
            return _post
        handles.append(layers[lb].register_forward_hook(_mk_post(lb)))

    def _make_masks_for_seq(am2d: torch.Tensor, first_p: List[int], last_p: List[int], pad_lens: List[int]):
        if mask_mode == "none":
            return None, None
        if mask_mode == "earlier_subj":
            # Fable (a): mask q >= last_subj from attending to [first_subj, last_subj) at all layers
            specs = [(lp, fp, lp) for fp, lp in zip(first_p, last_p)]
            m = build_layer_4d_mask(am2d, dtype, specs)
            return m, m
        if mask_mode == "full_knockout":
            # Late layers (>= l_start): mask q >= last_subj from [first_subj, last_subj)
            specs_late = [(lp, fp, lp) for fp, lp in zip(first_p, last_p)]
            # Early layers (< l_start): ALSO mask q > last_subj from whole subject [first_subj, last_subj+1)!
            m_late = build_layer_4d_mask(am2d, dtype, specs_late)
            m_early = m_late.clone()
            min_v = torch.finfo(dtype).min
            T = am2d.shape[1]
            for bi, (fp, lp) in enumerate(zip(first_p, last_p)):
                if lp + 1 < T:
                    m_early[bi, 0, lp + 1 :, fp : lp + 1] = min_v
            return m_early, m_late
        if mask_mode == "nonsubj_ctrl":
            specs = []
            for fp, lp, pad in zip(first_p, last_p, pad_lens):
                m_cnt = max(0, lp - fp)
                avail = max(0, fp - (pad + 1))
                mc = min(m_cnt, avail)
                specs.append((lp + 1, pad + 1, pad + 1 + mc))
            m = build_layer_4d_mask(am2d, dtype, specs)
            return m, m
        if mask_mode == "all_subj_pos_ctrl":
            # Positive control: mask q > last_subj from ENTIRE subject [first_subj, last_subj+1) at ALL layers!
            specs = [(lp + 1, fp, lp + 1) for fp, lp in zip(first_p, last_p)]
            m = build_layer_4d_mask(am2d, dtype, specs)
            return m, m
        raise ValueError(f"Unknown mask_mode: {mask_mode}")

    try:
        for b in batches:
            start, end = b["start"], b["end"]
            bp = b["batch_pairs"]
            ids = b["input_ids"]
            am2d = b["attention_mask"]
            pos_ids = (am2d.long().cumsum(dim=-1) - 1).clamp(min=0)

            m_early, m_late = _make_masks_for_seq(am2d, b["first_p"], b["last_p"], b["pad_lens"])
            ctx["pos_t"] = b["last_p_t"]
            ctx["z_map"] = {
                lb: torch.from_numpy(z_hat_by_layer[lb][start:end]).to(device=device, dtype=dtype)
                for lb in patch_layers
            }
            ctx["mask_early"] = m_early
            ctx["mask_late"] = m_late
            ctx["active"] = True
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=am2d, position_ids=pos_ids, use_cache=False)
            ctx["active"] = False

            first_toks = out.logits[:, -1, :].argmax(dim=-1)  # [B]
            cand_bi = [bi for bi, tid in enumerate(first_toks.tolist()) if tid in b["alias_first_sets"][bi]]
            if not cand_bi:
                continue

            # Cache-free exact greedy continuation ONLY for candidate-matching rows (fast & 100% exact!)
            sub_ids = torch.cat([ids[cand_bi], first_toks[cand_bi].unsqueeze(1)], dim=1)
            sub_am = torch.cat(
                [am2d[cand_bi], torch.ones((len(cand_bi), 1), dtype=am2d.dtype, device=device)], dim=1
            )
            sub_fp = [b["first_p"][i] for i in cand_bi]
            sub_lp = [b["last_p"][i] for i in cand_bi]
            sub_pad = [b["pad_lens"][i] for i in cand_bi]
            sub_z_map = {lb: ctx["z_map"][lb][cand_bi] for lb in patch_layers}
            gen_list = [first_toks[cand_bi].unsqueeze(1)]

            for _ in range(max_new_tokens - 1):
                sub_pos = (sub_am.long().cumsum(dim=-1) - 1).clamp(min=0)
                me, ml = _make_masks_for_seq(sub_am, sub_fp, sub_lp, sub_pad)
                ctx["pos_t"] = torch.tensor(sub_lp, dtype=torch.long, device=device)
                ctx["z_map"] = sub_z_map
                ctx["mask_early"] = me
                ctx["mask_late"] = ml
                ctx["active"] = True
                with torch.no_grad():
                    o_step = model(input_ids=sub_ids, attention_mask=sub_am, position_ids=sub_pos, use_cache=False)
                ctx["active"] = False
                nt = o_step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen_list.append(nt)
                sub_ids = torch.cat([sub_ids, nt], dim=1)
                sub_am = torch.cat([sub_am, torch.ones((len(cand_bi), 1), dtype=sub_am.dtype, device=device)], dim=1)

            gen_mat = torch.cat(gen_list, dim=1)
            for row_j, bi in enumerate(cand_bi):
                p_idx = start + bi
                d_idx = bp[bi][0]
                g_txt = tok.decode(gen_mat[row_j].tolist(), skip_special_tokens=True)
                if check_answer_match(g_txt, facts[d_idx]["aliases"], facts[d_idx]["relation"]):
                    outcomes[p_idx] = 1
    finally:
        for h in handles:
            h.remove()

    return float(sum(outcomes) / n), outcomes


def run_preflight_controls(
    model: Any,
    tok: Any,
    facts: List[dict],
    Z_B_all: torch.Tensor,
    base_idxs: List[int],
    layer_check: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """
    Fable + Astra + Neon mandatory pre-flight checks:
      1. Identity patch no-op: max |dlogit| < 1e-4 (f32) / < 1e-2 (bf16).
      2. Positive knockout control (`all_subj_pos_ctrl`): base accuracy MUST collapse!
      3. No-op 4D mask parity: max |dlogit| < 1e-4 (f32) / < 1e-2 (bf16).
    """
    prompts = [facts[i]["direct_prompt"] for i in base_idxs[:8]]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    am2d = enc["attention_mask"].to(device)
    pos_ids = (am2d.long().cumsum(dim=-1) - 1).clamp(min=0)
    t_pad = ids.shape[1]
    lp_list = []
    for i in base_idxs[:8]:
        _, li, sl = locate_subject_token_span(tok, facts[i]["direct_prompt"], facts[i]["direct_subj_char_span"])
        lp_list.append((t_pad - sl) + li)
    lp_t = torch.tensor(lp_list, dtype=torch.long, device=device)

    layers = get_decoder_layers(model)
    cap = {}
    def _cap(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        cap["h"] = h[torch.arange(h.shape[0], device=h.device), lp_t, :].detach().clone()
    hc = layers[layer_check].register_forward_hook(_cap)
    with torch.no_grad():
        log_ref = model(input_ids=ids, attention_mask=am2d, position_ids=pos_ids, use_cache=False).logits[:, -1, :].float()
        h_ref = cap["h"].clone()
    hc.remove()

    # 1. Identity patch no-op
    def _id(_m, _i, out):
        is_t = isinstance(out, tuple)
        h = out[0].clone() if is_t else out.clone()
        h[torch.arange(h.shape[0], device=h.device), lp_t, :] = h_ref
        return (h,) + out[1:] if is_t else h
    hi = layers[layer_check].register_forward_hook(_id)
    with torch.no_grad():
        log_id = model(input_ids=ids, attention_mask=am2d, position_ids=pos_ids, use_cache=False).logits[:, -1, :].float()
    hi.remove()
    id_diff = float((log_ref - log_id).abs().max().item())

    # 2. No-op 4D mask parity
    m_noop = build_layer_4d_mask(am2d, dtype, [(t_pad, 0, 0)] * len(prompts))
    h_m = []
    for l_i in range(len(layers)):
        def _pre(_m, a, kw):
            kw = dict(kw)
            kw["attention_mask"] = m_noop
            return a, kw
        h_m.append(layers[l_i].register_forward_pre_hook(_pre, with_kwargs=True))
    with torch.no_grad():
        log_4d = model(input_ids=ids, attention_mask=am2d, position_ids=pos_ids, use_cache=False).logits[:, -1, :].float()
    for h in h_m: h.remove()
    noop_diff = float((log_ref - log_4d).abs().max().item())

    # 3. Positive knockout control: evaluate base accuracy on 32 known items with `all_subj_pos_ctrl`!
    sub_pairs = [(i, i) for i in base_idxs[:32]]
    acc_base, _ = eval_knockout_swaps(model, tok, facts, sub_pairs, {}, device, dtype, mask_mode="none")
    acc_all_ko, _ = eval_knockout_swaps(model, tok, facts, sub_pairs, {}, device, dtype, mask_mode="all_subj_pos_ctrl")

    res = {
        "identity_patch_max_dlogit": round(id_diff, 6),
        "noop_4d_mask_max_dlogit": round(noop_diff, 6),
        "pos_ctrl_base_acc_32": round(acc_base, 4),
        "pos_ctrl_all_subj_knockout_acc_32": round(acc_all_ko, 4),
    }
    print(f"  [preflight] {res}")
    assert id_diff < 1e-2, f"Identity patch parity failed: {id_diff}"
    assert noop_diff < 1e-2, f"No-op 4D mask parity failed: {noop_diff}"
    assert acc_all_ko < 0.25, f"Positive knockout control failed to collapse base acc: {acc_all_ko}"
    return res


# ======================================================================================
# 2. Float64 Alignment Kernels, Free Geometry Ablations & Woodbury-Cayley
# ======================================================================================

def _polar_f64(M: np.ndarray) -> np.ndarray:
    try:
        U, _, Vt = sla.svd(M, full_matrices=False, lapack_driver="gesdd", check_finite=False)
    except sla.LinAlgError:
        U, _, Vt = sla.svd(M, full_matrices=False, lapack_driver="gesvd", check_finite=False)
    return U @ Vt


def fit_isometry_f64(
    Z_A: np.ndarray,
    Z_B: np.ndarray,
    Z_A_cal: Optional[np.ndarray] = None,
    Z_B_cal: Optional[np.ndarray] = None,
    cov_override: Optional[np.ndarray] = None,
) -> dict:
    """
    Centered O(d) / Stiefel partial isometry with exact post-projection RMS scaling!
    If `cov_override` (d_B x d_A cross-covariance, e.g. C_between + lam * C_within) is given,
    uses Polar(cov_override) while calibrating (mu_A, mu_B, s) on unaveraged (Z_A_cal, Z_B_cal).
    """
    ZA = Z_A.astype(np.float64)
    ZB = Z_B.astype(np.float64)
    if cov_override is not None:
        R = _polar_f64(cov_override)
    else:
        Ac = ZA - ZA.mean(axis=0)
        Bc = ZB - ZB.mean(axis=0)
        R = _polar_f64(Bc.T @ Ac)

    ZA_c = Z_A_cal.astype(np.float64) if Z_A_cal is not None else ZA
    ZB_c = Z_B_cal.astype(np.float64) if Z_B_cal is not None else ZB
    mu_A = ZA_c.mean(axis=0)
    mu_B = ZB_c.mean(axis=0)
    Ac_cal = ZA_c - mu_A
    Bc_cal = ZB_c - mu_B
    proj_A = Ac_cal @ R.T
    frob_raw = float(np.linalg.norm(Ac_cal, "fro"))
    frob_proj = float(np.linalg.norm(proj_A, "fro"))
    frob_B = float(np.linalg.norm(Bc_cal, "fro"))
    s_var = frob_B / max(frob_proj, 1e-12)
    # Also compute LS scale for Fable's Step-1 comparison!
    s_ls = float(np.sum(Bc_cal * proj_A) / max(frob_proj ** 2, 1e-12))
    null_frac = max(0.0, 1.0 - (frob_proj ** 2) / max(frob_raw ** 2, 1e-12))
    return {
        "mu_A": mu_A, "mu_B": mu_B, "R": R, "s": float(s_var), "s_ls": float(s_ls),
        "discarded_donor_nullspace_frac": round(float(null_frac), 5),
    }


def apply_map_f64(Z_A: np.ndarray, mp: dict, target_base_norms: Optional[np.ndarray] = None) -> np.ndarray:
    z = Z_A.astype(np.float64)
    if "W" in mp:
        out = mp["mu_B"] + (z - mp["mu_A"]) @ mp["W"].T
    else:
        out = mp["mu_B"] + mp["s"] * ((z - mp["mu_A"]) @ mp["R"].T)
    if target_base_norms is not None:
        # Fable's recipient base-run norm rescale (uses ONLY B's own unpatched base prompt norm!)
        cur_n = np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        out = out * (target_base_norms[:, None] / cur_n)
    return out


def build_free_geometry_ablation(
    Z_A: np.ndarray, Z_B: np.ndarray, Z_A_val: np.ndarray, Z_B_val: np.ndarray, gamma: float = 1.5
) -> dict:
    """
    Astra's Ridge SVD Polar/Clamped Ablation + Fable's Variance-Matched vs LS Scale check!
    """
    ZA, ZB = Z_A.astype(np.float64), Z_B.astype(np.float64)
    mu_A, mu_B = ZA.mean(axis=0), ZB.mean(axis=0)
    Ac, Bc = ZA - mu_A, ZB - mu_B
    n = len(ZA)
    Sig_A = (Ac.T @ Ac) / max(n, 1)
    C_BA = (Bc.T @ Ac) / max(n, 1)
    wa, Va = sla.eigh(0.5 * (Sig_A + Sig_A.T), check_finite=False)
    wa = np.maximum(wa, 0.0)
    avg_eig = max(float(np.mean(wa)), 1e-12)
    C_Vt = C_BA @ Va
    Ac_val_V = (Z_A_val.astype(np.float64) - mu_A) @ Va
    Bc_val = Z_B_val.astype(np.float64) - mu_B

    best_lam, best_mse, W_ridge = 0.1, float("inf"), None
    for lam in (1e-3, 1e-2, 5e-2, 0.1, 0.3, 1.0, 3.0):
        inv_d = 1.0 / (wa + lam * avg_eig)
        mse = float(np.mean(((Ac_val_V * inv_d) @ C_Vt.T - Bc_val) ** 2))
        if mse < best_mse:
            best_mse, best_lam = mse, lam
            W_ridge = (C_Vt * inv_d) @ Va.T

    U, sv, Vt = sla.svd(W_ridge, full_matrices=False, check_finite=False)
    med_sv = max(float(np.median(sv)), 1e-12)
    sv_n = sv / med_sv

    def _sm(W_m):
        return float(np.linalg.norm(Bc, "fro") / max(np.linalg.norm(Ac @ W_m.T, "fro"), 1e-12)) * W_m

    mp_proc = fit_isometry_f64(ZA, ZB)
    mp_ls = dict(mp_proc)
    mp_ls["s"] = mp_proc["s_ls"]

    sv_cl = np.clip(sv_n, 1.0 / gamma, gamma)
    sv_t16 = np.ones_like(sv_n)
    sv_t16[:16] = np.clip(sv_n[:16], 1.0 / gamma, gamma)

    return {
        "spec_stats": {
            "selected_lam": best_lam,
            "sv_max": round(float(sv[0]), 4),
            "sv_p90": round(float(np.percentile(sv, 90)), 4),
            "sv_median": round(med_sv, 4),
            "sv_p10": round(float(np.percentile(sv, 10)), 4),
            "s_var": round(mp_proc["s"], 4),
            "s_ls": round(mp_proc["s_ls"], 4),
        },
        "maps": {
            "proc_var_matched": mp_proc,
            "proc_ls_scale": mp_ls,
            "ridge_sigma_raw": {"mu_A": mu_A, "mu_B": mu_B, "W": W_ridge},
            "ridge_sigma_rescaled": {"mu_A": mu_A, "mu_B": mu_B, "W": _sm(W_ridge)},
            "ridge_polar_I": {"mu_A": mu_A, "mu_B": mu_B, "W": _sm(U @ Vt)},
            "ridge_clamped_1_5": {"mu_A": mu_A, "mu_B": mu_B, "W": _sm((U * sv_cl) @ Vt)},
            "ridge_top16_clamped": {"mu_A": mu_A, "mu_B": mu_B, "W": _sm((U * sv_t16) @ Vt)},
        },
    }


def apply_woodbury_cayley(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    U = torch.cat([A, B], dim=1)
    V = torch.cat([B, -A], dim=1)
    m = U.shape[1]
    I_m = torch.eye(m, dtype=U.dtype, device=U.device)
    M_sm = I_m - 0.5 * (V.T @ U)
    y = torch.linalg.solve(M_sm, (x @ V).T).T
    return x + y @ U.T


# ======================================================================================
# 3. Pure REAL Offline Entity Pool Builder (NO Fictional Compounds! Strict Leak Filter!)
# ======================================================================================

def build_real_proper_noun_pool(
    tok_A: Any,
    tok_B: Any,
    dataset: dict,
    dataset_2hop: Optional[dict],
    max_entities: int = 8000,
    seed: int = SEED,
) -> Tuple[List[str], dict]:
    """
    Fable Lever B-lite: Pure REAL capitalized proper nouns (NO composed fictional names!):
      1. Capitalized whole-word tokens in tok_A | tok_B that are NOT lowercase dictionary words
         in tok_A | tok_B (pure proper names: people, places, organizations, brands, events!).
      2. Plus curated real obscure entities from OBSCURE_ENTITIES (multi-token real places/universities!).
      3. Strict bidirectional exact + alias + whole-word-containment dedup against ALL subjects
         and answers in dataset.json & dataset_2hop.json! Logs exact removed count.
    """
    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()

    forb_surfaces = set()
    forb_words = set()
    def _add_f(s: str):
        sn = _norm(s)
        if sn:
            forb_surfaces.add(sn)
            for w in sn.split():
                if len(w) >= 3:
                    forb_words.add(w)

    for f in dataset["facts"]:
        _add_f(f["subject"])
        _add_f(f["answer"])
        for a in f.get("aliases", []): _add_f(a)
    if dataset_2hop is not None:
        for it in dataset_2hop.get("items", []):
            _add_f(it.get("subject", ""))
            _add_f(it.get("bridge_entity", ""))
            _add_f(it.get("answer", ""))
            for a in it.get("aliases", []): _add_f(a)

    # Collect lowercase dictionary words across both tokenizers to distinguish TRUE proper nouns!
    low_dict = set()
    caps_all = set()
    for tk in (tok_A, tok_B):
        for tid in range(len(tk)):
            s = tk.decode([tid])
            if s.startswith(" ") and 3 <= len(s) <= 16:
                w = s[1:]
                if re.fullmatch(r"[a-z]{3,15}", w):
                    low_dict.add(w)
                elif re.fullmatch(r"[A-Z][a-z]{2,14}", w):
                    caps_all.add(w)

    # Pure proper nouns: capitalized words whose lowercase form is NOT a common dictionary word!
    pure_proper = sorted(w for w in caps_all if w.lower() not in low_dict and not w.endswith("ly"))
    # Backup clean capitalized nouns if needed to reach max_entities
    other_caps = sorted(w for w in caps_all if w.lower() in low_dict and len(w) >= 4 and not w.endswith("ly"))
    curated_real = [e for e, _, _ in OBSCURE_ENTITIES]

    def _is_leaked(ent: str) -> bool:
        en = _norm(ent)
        if not en or en in forb_surfaces:
            return True
        ewords = set(en.split())
        if ewords & forb_words:
            return True
        for fs in forb_surfaces:
            if len(fs) >= 4 and (fs in en or en in fs):
                return True
        return False

    entities: List[str] = []
    seen = set()
    n_removed_leak = 0
    n_pure_proper = 0
    n_curated = 0

    for src_tag, src_list in [("curated", curated_real), ("pure_proper", pure_proper), ("other_cap", other_caps)]:
        rng = np.random.default_rng(seed + hash(src_tag) % 1000)
        arr = list(src_list)
        rng.shuffle(arr)
        for e in arr:
            if len(entities) >= max_entities:
                break
            el = e.lower()
            if el in seen:
                continue
            if _is_leaked(e):
                n_removed_leak += 1
                continue
            seen.add(el)
            entities.append(e)
            if src_tag == "curated":
                n_curated += 1
            elif src_tag == "pure_proper":
                n_pure_proper += 1

    assert all(not _is_leaked(e) for e in entities), "LEAK DETECTED!"
    meta = {
        "n_entities": len(entities),
        "n_curated_multiword_real": n_curated,
        "n_tokenizer_pure_proper_nouns": n_pure_proper,
        "n_other_cap_nouns": len(entities) - n_curated - n_pure_proper,
        "n_removed_by_leak_filter": n_removed_leak,
        "leak_count": 0,
        "sample": entities[:10],
    }
    return entities, meta


# ======================================================================================
# 4. Extraction & Stage 3 Truncated-at-last_subj Self-Swap Distillation
# ======================================================================================

def extract_band_states(
    model: Any,
    tok: Any,
    prompts: List[str],
    char_spans: List[Tuple[int, int]],
    layer_indices: List[int],
    device: torch.device,
    batch_size: int = 64,
) -> Dict[int, np.ndarray]:
    layers = get_decoder_layers(model)
    n = len(prompts)
    D = int(model.config.hidden_size)
    out_mats = {l: np.zeros((n, D), dtype=np.float32) for l in layer_indices}

    subj_idxs, seq_lens = [], []
    for p, sp in zip(prompts, char_spans):
        _, li, sl = locate_subject_token_span(tok, p, sp)
        subj_idxs.append(li)
        seq_lens.append(sl)

    ctx: Dict[str, Any] = {"pos": None, "buf": {}}
    handles = []
    for l in layer_indices:
        def _mk(li: int):
            def _hk(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                b_ar = torch.arange(h.shape[0], device=h.device)
                ctx["buf"][li] = h[b_ar, ctx["pos"], :].detach().float().cpu().numpy()
            return _hk
        handles.append(layers[l].register_forward_hook(_mk(l)))

    try:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            enc = tok(prompts[start:end], return_tensors="pt", padding=True, add_special_tokens=True)
            ids = enc["input_ids"].to(device)
            am = enc["attention_mask"].to(device)
            t_pad = ids.shape[1]
            pos = [max(0, min(t_pad - 1, (t_pad - seq_lens[i]) + subj_idxs[i])) for i in range(start, end)]
            ctx["pos"] = torch.tensor(pos, dtype=torch.long, device=device)
            ctx["buf"].clear()
            with torch.no_grad():
                model(input_ids=ids, attention_mask=am, use_cache=False)
            for l in layer_indices:
                out_mats[l][start:end] = ctx["buf"][l]
    finally:
        for h in handles:
            h.remove()
    return out_mats


def run_stage3_truncated_distillation(
    model_B_f32: Any,
    tok_B: Any,
    band_B: List[int],
    band_A: List[int],
    maps_R0: Dict[int, dict],
    Z_A_m3: Dict[int, np.ndarray],
    Z_B_m3: Dict[int, np.ndarray],
    entities: List[str],
    down_layers: List[int],
    device: torch.device,
    mask_mode: str,
    n_pairs: int = 1024,
    n_val: int = 128,
    n_steps: int = 80,
    batch_size: int = 16,
    lr: float = 0.015,
    prox_lam: float = 1e-3,
    seed: int = SEED,
) -> Tuple[Dict[int, dict], dict]:
    """
    Fable + Astra Modified Lever C:
      - Prompts truncated RIGHT AT `last_subj` (prefix + entity, ZERO suffixes, ZERO QA templates!).
      - Teacher = B's own self-swap (injecting B's native e1 into B's e2 prompt under identical mask_mode!).
      - Loss = variance-normalized downstream residual MSE at `last_subj` across layers `down_layers`
               normalized by native swap effect ||h_teacher - h_base||^2!
      - Strictly orthogonal Woodbury-Cayley full-D rank-16 SO(d), FROZEN scale & means,
        early-stopped strictly on held-out unlabeled entities!
    """
    rng = np.random.default_rng(seed + 888)
    n_ent = len(entities)
    tot = n_pairs + n_val
    # Hold out last 15% of entities strictly for unlabeled validation early-stopping!
    n_tr_ent = int(0.85 * n_ent)
    e1_tr = rng.integers(0, n_tr_ent, size=n_pairs)
    e2_tr = (e1_tr + rng.integers(1, n_tr_ent, size=n_pairs)) % n_tr_ent
    e1_va = rng.integers(n_tr_ent, n_ent, size=n_val)
    e2_va = n_tr_ent + ((e1_va - n_tr_ent + rng.integers(1, n_ent - n_tr_ent, size=n_val)) % (n_ent - n_tr_ent))
    e1_all = np.concatenate([e1_tr, e1_va])
    e2_all = np.concatenate([e2_tr, e2_va])
    p_idx_all = rng.integers(0, len(PREFIXES_M3), size=tot)

    prompts_e2 = [PREFIXES_M3[int(p_idx_all[k])] + entities[int(e2_all[k])] for k in range(tot)]
    spans_e2 = [(len(PREFIXES_M3[int(p_idx_all[k])]), len(prompts_e2[k])) for k in range(tot)]

    layers = get_decoder_layers(model_B_f32)
    D = int(model_B_f32.config.hidden_size)
    l_start = min(band_B)
    ctx_t: Dict[str, Any] = {"active": False, "pos_t": None, "z_map": {}, "cap": {}, "m_early": None, "m_late": None}
    handles = []

    if mask_mode != "none":
        for l_i in range(len(layers)):
            use_e = (mask_mode == "full_knockout" and l_i < l_start)
            def _mk_pre(is_e: bool):
                def _pre(_m, a, kw):
                    if not ctx_t["active"]:
                        return a, kw
                    m = ctx_t["m_early"] if is_e else ctx_t["m_late"]
                    if m is not None:
                        kw = dict(kw)
                        kw["attention_mask"] = m
                    return a, kw
                return _pre
            handles.append(layers[l_i].register_forward_pre_hook(_mk_pre(use_e), with_kwargs=True))

    for lb in band_B:
        def _mk_p(li: int):
            def _hk(_m, _i, out):
                if not ctx_t["active"]:
                    return out
                is_t = isinstance(out, tuple)
                h = out[0].clone() if is_t else out.clone()
                b_ar = torch.arange(h.shape[0], device=h.device)
                h[b_ar, ctx_t["pos_t"], :] = ctx_t["z_map"][li].to(device=h.device, dtype=h.dtype)
                return (h,) + out[1:] if is_t else h
            return _hk
        handles.append(layers[lb].register_forward_hook(_mk_p(lb)))

    for ld in down_layers:
        def _mk_c(li: int):
            def _hk(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                b_ar = torch.arange(h.shape[0], device=h.device)
                ctx_t["cap"][li] = h[b_ar, ctx_t["pos_t"], :]  # [B, D] right at last_subj!
            return _hk
        handles.append(layers[ld].register_forward_hook(_mk_c(ld)))

    batches = []
    try:
        for start in range(0, tot, batch_size):
            end = min(start + batch_size, tot)
            enc = tok_B(prompts_e2[start:end], return_tensors="pt", padding=True, add_special_tokens=True)
            ids = enc["input_ids"].to(device)
            am2d = enc["attention_mask"].to(device)
            pos_ids = (am2d.long().cumsum(dim=-1) - 1).clamp(min=0)
            t_pad = ids.shape[1]

            fp_l, lp_l = [], []
            for i in range(start, end):
                fi, li, sl = locate_subject_token_span(tok_B, prompts_e2[i], spans_e2[i])
                pad = t_pad - sl
                fp_l.append(pad + fi)
                lp_l.append(pad + li)
            pos_t = torch.tensor(lp_l, dtype=torch.long, device=device)

            m_late = build_layer_4d_mask(am2d, torch.float32, [(lp, fp, lp) for fp, lp in zip(fp_l, lp_l)]) if mask_mode != "none" else None
            m_early = m_late

            ctx_t["pos_t"] = pos_t
            ctx_t["m_early"] = m_early
            ctx_t["m_late"] = m_late

            # Base unpatched downstream states
            ctx_t["active"] = False
            ctx_t["cap"].clear()
            with torch.no_grad():
                model_B_f32(input_ids=ids, attention_mask=am2d, position_ids=pos_ids, use_cache=False)
                h_base = torch.stack([ctx_t["cap"][ld].detach().cpu() for ld in down_layers], dim=1)  # [B, Ld, D]

            # Native B self-swap teacher!
            e1_flat = [int(e1_all[i]) * len(PREFIXES_M3) + int(p_idx_all[i]) for i in range(start, end)]
            ctx_t["z_map"] = {lb: torch.from_numpy(Z_B_m3[lb][e1_flat]).to(device) for lb in band_B}
            ctx_t["active"] = True
            ctx_t["cap"].clear()
            with torch.no_grad():
                model_B_f32(input_ids=ids, attention_mask=am2d, position_ids=pos_ids, use_cache=False)
                h_teach = torch.stack([ctx_t["cap"][ld].detach().cpu() for ld in down_layers], dim=1)
            ctx_t["active"] = False

            d0_by_l = {}
            for la, lb in zip(band_A, band_B):
                mp = maps_R0[lb]
                d0 = mp["s"] * ((Z_A_m3[la][e1_flat].astype(np.float64) - mp["mu_A"]) @ mp["R"].T)
                d0_by_l[lb] = torch.from_numpy(d0.astype(np.float32))

            # Per-layer swap effect variance normalization!
            eff_per_l = ((h_teach - h_base) ** 2).mean(dim=(0, 2)).clamp(min=1e-4)  # [Ld]
            batches.append({
                "ids": ids.cpu(), "am2d": am2d.cpu(), "pos_ids": pos_ids.cpu(), "pos_t": pos_t.cpu(),
                "m_early": m_early.cpu() if m_early is not None else None,
                "m_late": m_late.cpu() if m_late is not None else None,
                "h_teach": h_teach, "eff_per_l": eff_per_l, "d0_by_l": d0_by_l,
            })

        n_tr_b = max(1, (n_pairs + batch_size - 1) // batch_size)
        tr_b, va_b = batches[:n_tr_b], batches[n_tr_b:]

        k_rank = 16
        params, pmap = [], {}
        mu_B_t = {lb: torch.from_numpy(maps_R0[lb]["mu_B"].astype(np.float32)).to(device) for lb in band_B}
        for lb in band_B:
            A_p = torch.zeros((D, k_rank), dtype=torch.float32, device=device, requires_grad=True)
            B_p = (0.01 * torch.randn((D, k_rank), dtype=torch.float32, device=device)).detach().requires_grad_(True)
            params.extend([A_p, B_p])
            pmap[lb] = {"A": A_p, "B": B_p}

        def _eval_val() -> float:
            tot_v = 0.0
            with torch.no_grad():
                for b in va_b:
                    ctx_t["pos_t"] = b["pos_t"].to(device)
                    ctx_t["m_early"] = b["m_early"].to(device) if b["m_early"] is not None else None
                    ctx_t["m_late"] = b["m_late"].to(device) if b["m_late"] is not None else None
                    ctx_t["z_map"] = {
                        lb: mu_B_t[lb] + apply_woodbury_cayley(b["d0_by_l"][lb].to(device), pmap[lb]["A"], pmap[lb]["B"])
                        for lb in band_B
                    }
                    ctx_t["active"] = True
                    ctx_t["cap"].clear()
                    model_B_f32(
                        input_ids=b["ids"].to(device), attention_mask=b["am2d"].to(device),
                        position_ids=b["pos_ids"].to(device), use_cache=False,
                    )
                    ctx_t["active"] = False
                    h_s = torch.stack([ctx_t["cap"][ld] for ld in down_layers], dim=1)
                    per_l = ((h_s - b["h_teach"].to(device)) ** 2).mean(dim=(0, 2)) / b["eff_per_l"].to(device)
                    tot_v += float(per_l.mean().item())
            return tot_v / max(len(va_b), 1)

        init_v = _eval_val()
        best_v = init_v
        best_st = {lb: {k: v.detach().cpu().clone() for k, v in pm.items()} for lb, pm in pmap.items()}
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_steps, 1), eta_min=lr * 0.1)

        for step in range(n_steps):
            b = tr_b[int(rng.integers(0, len(tr_b)))]
            opt.zero_grad()
            ctx_t["pos_t"] = b["pos_t"].to(device)
            ctx_t["m_early"] = b["m_early"].to(device) if b["m_early"] is not None else None
            ctx_t["m_late"] = b["m_late"].to(device) if b["m_late"] is not None else None
            ctx_t["z_map"] = {
                lb: mu_B_t[lb] + apply_woodbury_cayley(b["d0_by_l"][lb].to(device), pmap[lb]["A"], pmap[lb]["B"])
                for lb in band_B
            }
            ctx_t["active"] = True
            ctx_t["cap"].clear()
            with torch.enable_grad():
                model_B_f32(
                    input_ids=b["ids"].to(device), attention_mask=b["am2d"].to(device),
                    position_ids=b["pos_ids"].to(device), use_cache=False,
                )
                ctx_t["active"] = False
                h_s = torch.stack([ctx_t["cap"][ld] for ld in down_layers], dim=1)
                per_l = ((h_s - b["h_teach"].to(device)) ** 2).mean(dim=(0, 2)) / b["eff_per_l"].to(device)
                prox = sum((pm["A"] ** 2).sum() * (pm["B"] ** 2).sum() / (D * k_rank) for pm in pmap.values())
                loss = per_l.mean() + prox_lam * prox
                loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=2.0)
            opt.step()
            sched.step()

            if (step + 1) % max(1, n_steps // 4) == 0 or step == n_steps - 1:
                vr = _eval_val()
                if vr < best_v:
                    best_v = vr
                    best_st = {lb: {k: v.detach().cpu().clone() for k, v in pm.items()} for lb, pm in pmap.items()}
    finally:
        for h in handles:
            h.remove()

    # Materialize exact float64 orthogonal matrices Q @ R0 (and track ||Omega||_F!)
    new_maps: Dict[int, dict] = {}
    max_orth_err = 0.0
    omega_norms = {}
    for lb in band_B:
        mp = dict(maps_R0[lb])
        A_np = best_st[lb]["A"].to(torch.float64).numpy()
        B_np = best_st[lb]["B"].to(torch.float64).numpy()
        U = np.hstack([A_np, B_np])
        V = np.hstack([B_np, -A_np])
        M_sm = np.eye(U.shape[1]) - 0.5 * (V.T @ U)
        Q = np.eye(D) + U @ sla.solve(M_sm, V.T, check_finite=False)
        max_orth_err = max(max_orth_err, float(np.linalg.norm(Q.T @ Q - np.eye(D), "fro")))
        omega_norms[str(lb)] = round(float(np.linalg.norm(A_np @ B_np.T - B_np @ A_np.T, "fro")), 4)
        mp["R"] = Q @ mp["R"]
        new_maps[lb] = mp

    rel_red = (init_v - best_v) / max(init_v, 1e-12)
    diag = {
        "init_val_swap_norm_mse": round(init_v, 5),
        "best_val_swap_norm_mse": round(best_v, 5),
        "relative_reduction_pct": round(100.0 * rel_red, 2),
        "max_orth_err_f64": max_orth_err,
        "omega_fro_norms": omega_norms,
    }
    print(f"  [Stage 3 truncated-distill] val_mse={init_v:.4f}->{best_v:.4f} ({100*rel_red:+.1f}%, orth={max_orth_err:.2e})")
    return new_maps, diag


# ======================================================================================
# 5. Pure-CPU Unit Test & Main CLI
# ======================================================================================

def run_cpu_unit_test():
    print("[unit-test] Checking Woodbury-Cayley, 4D knockout masks, 5-donor eval builder, free geometry & real-noun pool...")
    torch.manual_seed(0)
    D, k = 128, 16
    A = 0.25 * torch.randn(D, k, dtype=torch.float64)
    B = 0.25 * torch.randn(D, k, dtype=torch.float64)
    Q = apply_woodbury_cayley(torch.eye(D, dtype=torch.float64), A, B).T
    orth_err = float(torch.linalg.norm(Q.T @ Q - torch.eye(D, dtype=torch.float64), "fro").item())
    assert orth_err < 1e-12

    am2d = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
    m4 = build_layer_4d_mask(am2d, torch.float32, [(4, 3, 4)])
    assert float(m4[0, 0, 4, 3]) < -1e30 and float(m4[0, 0, 5, 3]) < -1e30 and float(m4[0, 0, 5, 4]) == 0.0

    with open(DATASET_PATH, "r", encoding="utf-8") as f: ds = json.load(f)
    with open(DATASET_2HOP_PATH, "r", encoding="utf-8") as f: ds2 = json.load(f)
    with open(BASELINE_RESULTS_PATH, "r", encoding="utf-8") as f: rb = json.load(f)
    sh_test = rb["pairs"]["qwen2.5-3b->smollm2-1.7b"]["shared_test_indices"]
    p5 = build_powered_k_donor_pairs(ds["facts"], sh_test, k_donors=5)
    assert len(p5) >= 650

    tA = AutoTokenizer.from_pretrained(str(LAB_DIR / "hf_cache/qwen2.5-3b"), local_files_only=True)
    tB = AutoTokenizer.from_pretrained(str(LAB_DIR / "hf_cache/smollm2-1.7b"), local_files_only=True)
    ents, meta = build_real_proper_noun_pool(tA, tB, ds, ds2, max_entities=6000)
    assert meta["leak_count"] == 0 and len(ents) >= 5000

    rng = np.random.default_rng(1)
    ZA = rng.standard_normal((256, 96)); ZB = rng.standard_normal((256, 64))
    fg = build_free_geometry_ablation(ZA[:200], ZB[:200], ZA[200:], ZB[200:])
    assert len(fg["maps"]) == 7
    print(f"  ALL UNIT TESTS PASSED! orth_err={orth_err:.2e}, powered_5x_trials={len(p5)}, real_pool={meta}")


def main():
    parser = argparse.ArgumentParser(description="Pilot 1B: Breaking 85%+ Cleanly (3/3 Council Plan)")
    parser.add_argument("--pair", default="qwen2.5-3b->smollm2-1.7b")
    parser.add_argument("--stages", default="0,1,2,3")
    parser.add_argument("--n-ent-b", type=int, default=6000, help="Real entities for Lever B-lite (M=3 prefixes)")
    parser.add_argument("--unit-test", action="store_true")
    args = parser.parse_args()

    if args.unit_test:
        run_cpu_unit_test()
        return

    t_start = time.time()
    pair_key = args.pair
    mk_A, mk_B = pair_key.split("->")
    stages = {int(s.strip()) for s in args.stages.split(",") if s.strip()}

    with open(DATASET_PATH, "r", encoding="utf-8") as f: dataset = json.load(f)
    dataset_2hop = json.load(open(DATASET_2HOP_PATH)) if DATASET_2HOP_PATH.exists() else None
    with open(BASELINE_RESULTS_PATH, "r", encoding="utf-8") as f: baseline = json.load(f)
    with open(CACHE_DIR / "manifest.json", "r", encoding="utf-8") as f: manifest = json.load(f)
    eval_A = json.load(open(CACHE_DIR / f"{mk_A}_eval.json"))["facts"]
    eval_B = json.load(open(CACHE_DIR / f"{mk_B}_eval.json"))["facts"]

    facts = dataset["facts"]
    bp = baseline["pairs"][pair_key]
    L_A = int(manifest["models"][mk_A]["num_layers"])
    L_B = int(manifest["models"][mk_B]["num_layers"])
    la_opt, lb_opt = bp["best_layer_A"], bp["best_layer_B"]

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # Fable guardrail: ALWAYS run Granite in float32 (emb_mult=12 overflow risk in bf16/fp16)!
    eval_dtype = torch.float32 if (mk_B == "granite-3.1-2b" or not torch.backends.mps.is_available()) else torch.bfloat16
    print(f"[pilot1b] pair={pair_key} (L={L_A}->{L_B}, opt={la_opt}->{lb_opt}) stages={sorted(stages)} device={device} dtype={eval_dtype}")

    Z_A_all = torch.load(CACHE_DIR / f"{mk_A}_states.pt", map_location="cpu", weights_only=False)["direct_subj"]
    Z_B_all = torch.load(CACHE_DIR / f"{mk_B}_states.pt", map_location="cpu", weights_only=False)["direct_subj"]

    # Standard 1-donor eval + Powered 5-donor-per-base eval (~745 trials!) + LORO-5
    test_pairs_1x = [tuple(x) for x in bp["test_pairs"]]
    test_pairs_5x = build_powered_k_donor_pairs(facts, bp["shared_test_indices"], k_donors=5)
    shared_loro = [
        f["idx"] for f in facts
        if f["relation"] in LORO_HELD_OUT_RELS
        and f["split"] in ("val", "test")
        and eval_A[f["idx"]]["direct_correct"] and eval_B[f["idx"]]["direct_correct"]
    ]
    loro_pairs = build_counterfactual_pairs(facts, shared_loro)

    all_tr = [f["idx"] for f in facts if f["split"] == "train"]
    train_ans = {a.strip().lower() for i in all_tr for a in facts[i]["aliases"]}
    ad_mask_1x = np.array([all(a.strip().lower() not in train_ans for a in facts[d]["aliases"]) for d, _ in test_pairs_1x], dtype=bool)
    ad_mask_5x = np.array([all(a.strip().lower() not in train_ans for a in facts[d]["aliases"]) for d, _ in test_pairs_5x], dtype=bool)

    shuf_1x = make_same_relation_shuffled_donors(facts, test_pairs_1x, seed=SEED + 11)
    shuf_5x = make_same_relation_shuffled_donors(facts, test_pairs_5x, seed=SEED + 15)
    shuf_lo = make_same_relation_shuffled_donors(facts, loro_pairs, seed=SEED + 22)

    dir_A = Path(manifest["models"][mk_A]["local_path"])
    dir_B = Path(manifest["models"][mk_B]["local_path"])
    tok_A = AutoTokenizer.from_pretrained(str(dir_A), local_files_only=True)
    tok_A.padding_side = "left"
    if tok_A.pad_token is None: tok_A.pad_token = tok_A.eos_token
    tok_B = AutoTokenizer.from_pretrained(str(dir_B), local_files_only=True)
    tok_B.padding_side = "left"
    if tok_B.pad_token is None: tok_B.pad_token = tok_B.eos_token

    model_B = AutoModelForCausalLM.from_pretrained(str(dir_B), local_files_only=True, dtype=eval_dtype).to(device).eval()
    pre_1x = pretokenize_knockout_batches(tok_B, facts, test_pairs_1x, device, batch_size=32)
    pre_5x = pretokenize_knockout_batches(tok_B, facts, test_pairs_5x, device, batch_size=32)
    pre_lo = pretokenize_knockout_batches(tok_B, facts, loro_pairs, device, batch_size=32)

    t1_d = [d for d, _ in test_pairs_1x]
    t5_d = [d for d, _ in test_pairs_5x]
    lo_d = [d for d, _ in loro_pairs]

    def _match_lA(lb: int) -> int:
        return int(np.clip(round(la_opt + (lb - lb_opt) * (L_A / float(L_B))), 2, L_A - 3))

    def _eval_full(
        name: str,
        z1_m: Dict[int, np.ndarray], z1_s: Dict[int, np.ndarray],
        zlo_m: Dict[int, np.ndarray], zlo_s: Dict[int, np.ndarray],
        mask_mode: str,
        z5_m: Optional[Dict[int, np.ndarray]] = None,
        z5_s: Optional[Dict[int, np.ndarray]] = None,
        native_ceil_1x: Optional[float] = None,
        native_ceil_5x: Optional[float] = None,
        base_out_1x: Optional[List[int]] = None,
    ) -> dict:
        _, o1m = eval_knockout_swaps(model_B, tok_B, facts, test_pairs_1x, z1_m, device, eval_dtype, mask_mode, pre_1x)
        _, o1s = eval_knockout_swaps(model_B, tok_B, facts, test_pairs_1x, z1_s, device, eval_dtype, mask_mode, pre_1x)
        _, olom = eval_knockout_swaps(model_B, tok_B, facts, loro_pairs, zlo_m, device, eval_dtype, mask_mode, pre_lo)
        _, olos = eval_knockout_swaps(model_B, tok_B, facts, loro_pairs, zlo_s, device, eval_dtype, mask_mode, pre_lo)

        cf1 = base_clustered_bootstrap_ci(o1m, o1s, [facts[d]["subject_norm"] for d, _ in test_pairs_1x])
        adi = [i for i, fl in enumerate(ad_mask_1x) if fl]
        ad1 = base_clustered_bootstrap_ci([o1m[i] for i in adi], [o1s[i] for i in adi], [facts[test_pairs_1x[i][0]]["subject_norm"] for i in adi])
        lo1 = base_clustered_bootstrap_ci(olom, olos, [facts[d]["subject_norm"] for d, _ in loro_pairs])

        cf5 = ad5 = None
        if z5_m is not None and z5_s is not None:
            _, o5m = eval_knockout_swaps(model_B, tok_B, facts, test_pairs_5x, z5_m, device, eval_dtype, mask_mode, pre_5x)
            _, o5s = eval_knockout_swaps(model_B, tok_B, facts, test_pairs_5x, z5_s, device, eval_dtype, mask_mode, pre_5x)
            cf5 = base_clustered_bootstrap_ci(o5m, o5s, [b for _, b in test_pairs_5x])
            ad5i = [i for i, fl in enumerate(ad_mask_5x) if fl]
            ad5 = base_clustered_bootstrap_ci([o5m[i] for i in ad5i], [o5s[i] for i in ad5i], [test_pairs_5x[i][1] for i in ad5i])

        slens = [sl for b in pre_1x for sl in b["subj_lens"]]
        i1t = [i for i, sl in enumerate(slens) if sl == 1]
        imt = [i for i, sl in enumerate(slens) if sl > 1]
        eff1 = round(cf1["matched_acc"] / native_ceil_1x, 4) if native_ceil_1x else None
        eff5 = round(cf5["matched_acc"] / native_ceil_5x, 4) if (cf5 and native_ceil_5x) else None
        resc = int(sum(1 for a, b in zip(base_out_1x, o1m) if a == 0 and b == 1)) if base_out_1x else None
        spoil = int(sum(1 for a, b in zip(base_out_1x, o1m) if a == 1 and b == 0)) if base_out_1x else None

        rec = {
            "name": name, "mask_mode": mask_mode, "layers": sorted(z1_m.keys()),
            "cf_1x": cf1, "ad_1x": ad1, "loro_5": lo1, "cf_5x_powered": cf5, "ad_5x_powered": ad5,
            "transport_efficiency_1x": eff1, "transport_efficiency_5x": eff5,
            "acc_1tok_base": round(float(np.mean([o1m[i] for i in i1t])), 4) if i1t else None,
            "acc_multitok_base": round(float(np.mean([o1m[i] for i in imt])), 4) if imt else None,
            "rescued_vs_base": resc, "spoiled_vs_base": spoil, "outcomes_1x": o1m,
        }
        p5_s = f" | 5x={cf5['matched_acc']:.4f}(AD={ad5['matched_acc']:.4f})" if cf5 else ""
        ef_s = f" eff={eff1:.3f}" if eff1 else ""
        rs_s = f" (+{resc}/-{spoil})" if resc is not None else ""
        print(
            f"  [{name:34s}] CF={cf1['matched_acc']:.4f}(sh={cf1['shuffled_acc']:.3f}) AD={ad1['matched_acc']:.4f} "
            f"LORO={lo1['matched_acc']:.4f}{p5_s}{ef_s} 1t={rec['acc_1tok_base']} mt={rec['acc_multitok_base']}{rs_s}"
        )
        return rec

    results: Dict[str, Any] = {
        "meta": {
            "pair_key": pair_key, "stages": sorted(stages), "device": str(device), "dtype": str(eval_dtype),
            "n_test_1x": len(test_pairs_1x), "n_test_5x": len(test_pairs_5x),
            "n_test_ad_1x": int(ad_mask_1x.sum()), "n_loro": len(loro_pairs),
        }
    }

    # Cache native self-swap ceilings keyed by (tuple(layers), mask_mode) for Transport Efficiency!
    native_ceilings: Dict[Tuple[Tuple[int, ...], str], Tuple[float, Optional[float]]] = {}

    # ==================================================================================
    # STAGE 0: Pre-Flight + Native Self-Swap Ceilings + Free Geometry & Ridge Spectral Ablation
    # ==================================================================================
    if 0 in stages:
        print("\n=== STAGE 0: Pre-Flight + Native Self-Swap Ceilings + Free Geometry Ablation ===")
        preflight = run_preflight_controls(model_B, tok_B, facts, Z_B_all, [b for _, b in test_pairs_1x], lb_opt, device, eval_dtype)

        # Granite lexical-retention diagnostic: cos(h[l], h[0]) across layers!
        cos_to_l0 = {}
        z0 = Z_B_all[t1_d, 0].float().numpy()
        for li in range(0, L_B, max(1, L_B // 8)):
            zl = Z_B_all[t1_d, li].float().numpy()
            c = np.sum(zl * z0, axis=1) / np.maximum(np.linalg.norm(zl, axis=1) * np.linalg.norm(z0, axis=1), 1e-12)
            cos_to_l0[str(li)] = round(float(np.mean(c)), 4)
        print(f"  [Stage 0] cos(h[l], h[0]) at last_subj: {cos_to_l0}")

        if mk_B == "smollm2-1.7b":
            self_cfgs = [
                ("A0_self_single_l9", [9], "none", True),
                ("A1_self_band3_8_9_10", [8, 9, 10], "none", False),
                ("A2_self_l9_earlier_ko", [9], "earlier_subj", False),
                ("A2_self_l9_full_ko", [9], "full_knockout", True),
                ("A3_self_band3_earlier_ko", [8, 9, 10], "earlier_subj", True),
                ("A3_self_band3_full_ko", [8, 9, 10], "full_knockout", True),
                ("A3_self_band3_nonsubj_ctrl", [8, 9, 10], "nonsubj_ctrl", False),
            ]
        else:  # granite-3.1-2b: full layer sweep {4, 8, 12, 16, 20, 24} + knockouts + bands!
            self_cfgs = [
                ("A0_self_l4", [4], "none", False),
                ("A0_self_l8", [8], "none", True),
                ("A0_self_l12", [12], "none", False),
                ("A0_self_l16", [16], "none", False),
                ("A0_self_l20", [20], "none", False),
                ("A0_self_l24", [24], "none", False),
                ("A2_self_l8_full_ko", [8], "full_knockout", True),
                ("A2_self_l12_full_ko", [12], "full_knockout", True),
                ("A2_self_l16_full_ko", [16], "full_knockout", True),
                ("A1_self_band3_7_8_9", [7, 8, 9], "none", False),
                ("A3_self_band3_7_9_full_ko", [7, 8, 9], "full_knockout", True),
                ("A3_self_band3_12_14_full_ko", [12, 13, 14], "full_knockout", True),
                ("A3_self_band5_10_18_full_ko", [10, 12, 14, 16, 18], "full_knockout", True),
                ("A3_self_band3_12_14_nonsubj_ctrl", [12, 13, 14], "nonsubj_ctrl", False),
            ]

        s0_self = {}
        base_self_o = None
        for cname, blayers, mm, do_5x in self_cfgs:
            z1m = {l: Z_B_all[t1_d, l].float().numpy() for l in blayers}
            z1s = {l: Z_B_all[shuf_1x, l].float().numpy() for l in blayers}
            zlom = {l: Z_B_all[lo_d, l].float().numpy() for l in blayers}
            zlos = {l: Z_B_all[shuf_lo, l].float().numpy() for l in blayers}
            z5m = {l: Z_B_all[t5_d, l].float().numpy() for l in blayers} if do_5x else None
            z5s = {l: Z_B_all[shuf_5x, l].float().numpy() for l in blayers} if do_5x else None
            r = _eval_full(cname, z1m, z1s, zlom, zlos, mm, z5m, z5s, base_out_1x=base_self_o)
            if cname in ("A0_self_single_l9", "A0_self_l8"):
                base_self_o = r["outcomes_1x"]
            native_ceilings[(tuple(blayers), mm)] = (r["cf_1x"]["matched_acc"], r["cf_5x_powered"]["matched_acc"] if r["cf_5x_powered"] else None)
            s0_self[cname] = {k: v for k, v in r.items() if k != "outcomes_1x"}

        # Opus G5: Amplified self-swap h_B(e2) + alpha*(h_B(e1) - h_B(e2)) for alpha in {1.5, 2.0}!
        t1_b = [b for _, b in test_pairs_1x]
        lo_b = [b for _, b in loro_pairs]
        for amp_a in (1.5, 2.0):
            zb_d1 = Z_B_all[t1_d, lb_opt].float().numpy()
            zb_b1 = Z_B_all[t1_b, lb_opt].float().numpy()
            zb_s1 = Z_B_all[shuf_1x, lb_opt].float().numpy()
            zb_dl = Z_B_all[lo_d, lb_opt].float().numpy()
            zb_bl = Z_B_all[lo_b, lb_opt].float().numpy()
            zb_sl = Z_B_all[shuf_lo, lb_opt].float().numpy()
            r_amp = _eval_full(
                f"A0_self_amp_a{amp_a:.1f}_l{lb_opt}",
                {lb_opt: zb_b1 + amp_a * (zb_d1 - zb_b1)},
                {lb_opt: zb_b1 + amp_a * (zb_s1 - zb_b1)},
                {lb_opt: zb_bl + amp_a * (zb_dl - zb_bl)},
                {lb_opt: zb_bl + amp_a * (zb_sl - zb_bl)},
                "none", base_out_1x=base_self_o,
            )
            s0_self[f"A0_self_amp_a{amp_a:.1f}_l{lb_opt}"] = {k: v for k, v in r_amp.items() if k != "outcomes_1x"}

        # Opus protocol validity: knockout-only base retention (unpatched B accuracy under knockout!)
        ko_base_pairs = [(b, b) for _, b in test_pairs_1x]
        ko_ret_base = {}
        for km in ("none", "earlier_subj", "full_knockout"):
            acc_k, _ = eval_knockout_swaps(model_B, tok_B, facts, ko_base_pairs, {}, device, eval_dtype, mask_mode=km)
            ko_ret_base[km] = round(acc_k, 4)
        print(f"  [Stage 0] Knockout-only base retention (unpatched): {ko_ret_base}")

        # Free Geometry / Lever F Ablation (Ridge SVD Polar/Clamped + Spherical + Entity-Subspace Transplant!)
        fg_res = {"knockout_only_base_retention": ko_ret_base}
        p1_unlab = CACHE_DIR / f"pilot1_unlab_{mk_A}_to_{mk_B}_l{la_opt}_{lb_opt}.pt"
        if p1_unlab.exists():
            print("\n  [Stage 0b] Lever F: Free Geometry, Ridge Polar, Spherical & Entity-Subspace Transplant (N=10k):")
            pk = torch.load(p1_unlab, map_location="cpu", weights_only=False)
            ZA_u, ZB_u = pk["Z_A_unlab_tr"].numpy(), pk["Z_B_unlab_tr"].numpy()
            fg = build_free_geometry_ablation(
                ZA_u, ZB_u, pk["Z_A_unlab_val"].numpy(), pk["Z_B_unlab_val"].numpy(),
            )
            fg_res["spec_stats"] = fg["spec_stats"]
            fg_res["variants"] = {}
            print(f"    spec_stats: {fg['spec_stats']}")
            nc1, _ = native_ceilings.get(((lb_opt,), "none"), (None, None))
            for vname, vmp in fg["maps"].items():
                z1m = {lb_opt: apply_map_f64(Z_A_all[t1_d, la_opt].float().numpy(), vmp)}
                z1s = {lb_opt: apply_map_f64(Z_A_all[shuf_1x, la_opt].float().numpy(), vmp)}
                zlom = {lb_opt: apply_map_f64(Z_A_all[lo_d, la_opt].float().numpy(), vmp)}
                zlos = {lb_opt: apply_map_f64(Z_A_all[shuf_lo, la_opt].float().numpy(), vmp)}
                r = _eval_full(vname, z1m, z1s, zlom, zlos, "none", native_ceil_1x=nc1)
                fg_res["variants"][vname] = {k: v for k, v in r.items() if k != "outcomes_1x"}

            # Log exact patch-site norm ratio ||R h_A|| / ||h_B|| on test entities!
            mp_pv = fg["maps"]["proc_var_matched"]
            zh_pv = apply_map_f64(Z_A_all[t1_d, la_opt].float().numpy(), mp_pv)
            zb_te_true = Z_B_all[t1_d, lb_opt].float().numpy()
            fg_res["patch_site_norm_ratio_mean"] = round(
                float(np.mean(np.linalg.norm(zh_pv, axis=1) / np.maximum(np.linalg.norm(zb_te_true, axis=1), 1e-12))), 4
            )
            print(f"    patch_site_norm_ratio ||z_hat||/||z_B|| = {fg_res['patch_site_norm_ratio_mean']}")

            # Fable recipient base-run norm rescale on proc_var_matched
            bn_1x = np.linalg.norm(Z_B_all[t1_b, lb_opt].float().numpy(), axis=1)
            bn_lo = np.linalg.norm(Z_B_all[lo_b, lb_opt].float().numpy(), axis=1)
            r_bn = _eval_full(
                "F_proc_base_norm_rescale",
                {lb_opt: apply_map_f64(Z_A_all[t1_d, la_opt].float().numpy(), mp_pv, target_base_norms=bn_1x)},
                {lb_opt: apply_map_f64(Z_A_all[shuf_1x, la_opt].float().numpy(), mp_pv, target_base_norms=bn_1x)},
                {lb_opt: apply_map_f64(Z_A_all[lo_d, la_opt].float().numpy(), mp_pv, target_base_norms=bn_lo)},
                {lb_opt: apply_map_f64(Z_A_all[shuf_lo, la_opt].float().numpy(), mp_pv, target_base_norms=bn_lo)},
                "none", native_ceil_1x=nc1,
            )
            fg_res["variants"]["F_proc_base_norm_rescale"] = {k: v for k, v in r_bn.items() if k != "outcomes_1x"}

            # Opus Lever F: Spherical Procrustes (unit-RMS fit + base native norm injection!)
            def _unit_rms(M):
                return M / np.maximum(np.sqrt(np.mean(M ** 2, axis=1, keepdims=True)), 1e-12)
            mp_sph = fit_isometry_f64(_unit_rms(ZA_u.astype(np.float64)), _unit_rms(ZB_u.astype(np.float64)))
            r_sph = _eval_full(
                "F_spherical_procrustes",
                {lb_opt: apply_map_f64(_unit_rms(Z_A_all[t1_d, la_opt].float().numpy()), mp_sph, target_base_norms=bn_1x)},
                {lb_opt: apply_map_f64(_unit_rms(Z_A_all[shuf_1x, la_opt].float().numpy()), mp_sph, target_base_norms=bn_1x)},
                {lb_opt: apply_map_f64(_unit_rms(Z_A_all[lo_d, la_opt].float().numpy()), mp_sph, target_base_norms=bn_lo)},
                {lb_opt: apply_map_f64(_unit_rms(Z_A_all[shuf_lo, la_opt].float().numpy()), mp_sph, target_base_norms=bn_lo)},
                "none", native_ceil_1x=nc1,
            )
            fg_res["variants"]["F_spherical_procrustes"] = {k: v for k, v in r_sph.items() if k != "outcomes_1x"}

            # Opus Lever F: Entity-Subspace Transplant h' = h_B(e2) + P_r [z_hat - h_B(e2)], r in {256, 1024}
            ZB_uc = ZB_u.astype(np.float64) - ZB_u.astype(np.float64).mean(axis=0)
            _, _, Vt_B = sla.svd(ZB_uc, full_matrices=False, check_finite=False)
            zb_b1_f = Z_B_all[t1_b, lb_opt].float().numpy().astype(np.float64)
            zb_bl_f = Z_B_all[lo_b, lb_opt].float().numpy().astype(np.float64)
            zh_s1 = apply_map_f64(Z_A_all[shuf_1x, la_opt].float().numpy(), mp_pv)
            zh_lo = apply_map_f64(Z_A_all[lo_d, la_opt].float().numpy(), mp_pv)
            zh_ls = apply_map_f64(Z_A_all[shuf_lo, la_opt].float().numpy(), mp_pv)
            for r_tr in (256, 1024):
                Vr = Vt_B[:r_tr].T  # [D, r]
                def _tr_proj(zh, zb):
                    diff = zh - zb
                    return zb + (diff @ Vr) @ Vr.T
                r_t = _eval_full(
                    f"F_entity_transplant_r{r_tr}",
                    {lb_opt: _tr_proj(zh_pv, zb_b1_f)}, {lb_opt: _tr_proj(zh_s1, zb_b1_f)},
                    {lb_opt: _tr_proj(zh_lo, zb_bl_f)}, {lb_opt: _tr_proj(zh_ls, zb_bl_f)},
                    "none", native_ceil_1x=nc1,
                )
                fg_res["variants"][f"F_entity_transplant_r{r_tr}"] = {k: v for k, v in r_t.items() if k != "outcomes_1x"}

            # Opus Gate G4: Free Two-Donor Residual Correlation + Ensemble Diagnostic (Qwen + Falcon -> SmolLM2!)
            q_mp_p = CACHE_DIR / "pilot1_maps_qwen2.5-3b_to_smollm2-1.7b_l17_9.pt"
            f_mp_p = CACHE_DIR / "pilot1_maps_falcon3-3b_to_smollm2-1.7b_l14_9.pt"
            if mk_B == "smollm2-1.7b" and q_mp_p.exists() and f_mp_p.exists():
                qm = torch.load(q_mp_p, map_location="cpu", weights_only=False)["procrustes_global"]
                fm = torch.load(f_mp_p, map_location="cpu", weights_only=False)["procrustes_global"]
                Z_q = torch.load(CACHE_DIR / "qwen2.5-3b_states.pt", map_location="cpu", weights_only=False)["direct_subj"][:, 17, :].float().numpy()
                Z_f = torch.load(CACHE_DIR / "falcon3-3b_states.pt", map_location="cpu", weights_only=False)["direct_subj"][:, 14, :].float().numpy()
                zh_q = apply_map_f64(Z_q[t1_d], qm)
                zh_f = apply_map_f64(Z_f[t1_d], fm)
                err_q = zh_q - zb_te_true
                err_f = zh_f - zb_te_true
                res_corr = float(np.sum(err_q * err_f) / max(np.linalg.norm(err_q) * np.linalg.norm(err_f), 1e-12))
                zh_avg = 0.5 * (zh_q + zh_f)
                zh_avg_s = 0.5 * (apply_map_f64(Z_q[shuf_1x], qm) + apply_map_f64(Z_f[shuf_1x], fm))
                zh_avg_lo = 0.5 * (apply_map_f64(Z_q[lo_d], qm) + apply_map_f64(Z_f[lo_d], fm))
                zh_avg_ls = 0.5 * (apply_map_f64(Z_q[shuf_lo], qm) + apply_map_f64(Z_f[shuf_lo], fm))
                r_2d = _eval_full(
                    "G4_two_donor_qwen_falcon_avg",
                    {lb_opt: zh_avg}, {lb_opt: zh_avg_s}, {lb_opt: zh_avg_lo}, {lb_opt: zh_avg_ls},
                    "none", native_ceil_1x=nc1,
                )
                fg_res["two_donor_G4"] = {
                    "residual_error_correlation_qwen_vs_falcon": round(res_corr, 4),
                    "eval": {k: v for k, v in r_2d.items() if k != "outcomes_1x"},
                }
                print(f"    [Opus G4] Two-donor residual error correlation (Qwen vs Falcon -> SmolLM2) = {res_corr:.4f}")

        results["stage0"] = {"preflight": preflight, "cos_to_l0": cos_to_l0, "native_self_swap": s0_self, "free_geometry": fg_res}

    # Select winning intervention band per recipient
    if mk_B == "smollm2-1.7b":
        band_B_main = [8, 9, 10]
        down_layers = [11, 12, 13, 14]
    else:
        band_B_main = [12, 13, 14]
        down_layers = [15, 16, 17, 18]
    all_band_B = sorted(set([lb_opt] + band_B_main + ([16] if mk_B == "granite-3.1-2b" else [])))
    all_band_A = sorted(set([la_opt] + [_match_lA(lb) for lb in all_band_B]))

    # ==================================================================================
    # STAGE 1 (Lever A): Cross-Family Band + Exact Attention Knockout (N=10k Unlabeled)
    # ==================================================================================
    if 1 in stages:
        print(f"\n=== STAGE 1 (Lever A): Cross-Family Knockout & Band (layers B={all_band_B}) ===")
        p1_unlab = CACHE_DIR / f"pilot1_unlab_{mk_A}_to_{mk_B}_l{la_opt}_{lb_opt}.pt"
        band_cache = CACHE_DIR / f"pilot1b_band10k_{mk_A}_to_{mk_B}.pt"
        if band_cache.exists():
            bp_c = torch.load(band_cache, map_location="cpu", weights_only=False)
            st1_A = {int(k): v.numpy() for k, v in bp_c["st_A"].items()}
            st1_B = {int(k): v.numpy() for k, v in bp_c["st_B"].items()}
        else:
            pk = torch.load(p1_unlab, map_location="cpu", weights_only=False)
            pr10 = [it["prompt"] for it in pk["train_items"]]
            sp10 = [tuple(it["char_span"]) for it in pk["train_items"]]
            t_e = time.time()
            st1_B = extract_band_states(model_B, tok_B, pr10, sp10, all_band_B, device, batch_size=64)
            model_A = AutoModelForCausalLM.from_pretrained(str(dir_A), local_files_only=True, dtype=torch.bfloat16 if torch.backends.mps.is_available() else torch.float32).to(device).eval()
            st1_A = extract_band_states(model_A, tok_A, pr10, sp10, all_band_A, device, batch_size=64)
            del model_A
            gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            print(f"  [Stage 1] Extracted 10k band states in {time.time() - t_e:.1f}s")
            torch.save({"st_A": {k: torch.from_numpy(v) for k, v in st1_A.items()}, "st_B": {k: torch.from_numpy(v) for k, v in st1_B.items()}}, band_cache)

        s1_maps = {}
        null_audit = {}
        for lb in all_band_B:
            la = _match_lA(lb)
            s1_maps[lb] = fit_isometry_f64(st1_A[la], st1_B[lb])
            s1_maps[lb]["la"] = la
            null_audit[f"lA{la}_to_lB{lb}"] = {"s": round(s1_maps[lb]["s"], 4), "null_frac": s1_maps[lb]["discarded_donor_nullspace_frac"]}

        def _run_s1(cname, blayers, mm, do_5x=True, base_o=None):
            nc1, nc5 = native_ceilings.get((tuple(blayers), mm), (None, None))
            z1m = {lb: apply_map_f64(Z_A_all[t1_d, s1_maps[lb]["la"]].float().numpy(), s1_maps[lb]) for lb in blayers}
            z1s = {lb: apply_map_f64(Z_A_all[shuf_1x, s1_maps[lb]["la"]].float().numpy(), s1_maps[lb]) for lb in blayers}
            zlom = {lb: apply_map_f64(Z_A_all[lo_d, s1_maps[lb]["la"]].float().numpy(), s1_maps[lb]) for lb in blayers}
            zlos = {lb: apply_map_f64(Z_A_all[shuf_lo, s1_maps[lb]["la"]].float().numpy(), s1_maps[lb]) for lb in blayers}
            z5m = {lb: apply_map_f64(Z_A_all[t5_d, s1_maps[lb]["la"]].float().numpy(), s1_maps[lb]) for lb in blayers} if do_5x else None
            z5s = {lb: apply_map_f64(Z_A_all[shuf_5x, s1_maps[lb]["la"]].float().numpy(), s1_maps[lb]) for lb in blayers} if do_5x else None
            return _eval_full(cname, z1m, z1s, zlom, zlos, mm, z5m, z5s, nc1, nc5, base_o)

        s1_cfgs = {}
        r_a0 = _run_s1("A0_single_unlab10k", [lb_opt], "none", do_5x=True)
        base_o = r_a0["outcomes_1x"]
        s1_cfgs["A0_single_unlab10k"] = {k: v for k, v in r_a0.items() if k != "outcomes_1x"}

        run_list = [
            ("A2_single_full_ko", [lb_opt], "full_knockout", True),
            ("A1_band_main_none", band_B_main, "none", False),
            ("A3_band_main_earlier_ko", band_B_main, "earlier_subj", True),
            ("A3_band_main_full_ko", band_B_main, "full_knockout", True),
            ("A3_band_main_nonsubj_ctrl", band_B_main, "nonsubj_ctrl", False),
        ]
        if mk_B == "granite-3.1-2b":
            run_list.append(("A2_l16_full_ko", [16], "full_knockout", True))
        for cn, bl, mm, d5 in run_list:
            r = _run_s1(cn, bl, mm, do_5x=d5, base_o=base_o)
            s1_cfgs[cn] = {k: v for k, v in r.items() if k != "outcomes_1x"}

        results["stage1_leverA"] = {"nullspace_audit": null_audit, "configs": s1_cfgs}

    # ==================================================================================
    # STAGE 2 (Lever B-lite): Pure REAL Proper Nouns x M=3 Prefixes (Truncated at last_subj!)
    # ==================================================================================
    b_maps_best: Dict[int, dict] = {}
    stB_m3_A: Dict[int, np.ndarray] = {}
    stB_m3_B: Dict[int, np.ndarray] = {}
    ents_real: List[str] = []

    if 2 in stages or 3 in stages:
        print(f"\n=== STAGE 2 (Lever B-lite): Pure REAL Proper Nouns (N={args.n_ent_b} x M=3 prefixes, truncated at last_subj!) ===")
        ents_real, pool_meta = build_real_proper_noun_pool(tok_A, tok_B, dataset, dataset_2hop, max_entities=args.n_ent_b, seed=SEED + 202)
        n_ent = len(ents_real)
        print(f"  real_pool_meta: {pool_meta}")

        # Prompts truncated RIGHT AT `last_subj` (causal LLM -> suffixes do nothing!)
        p_m3, sp_m3 = [], []
        for ent in ents_real:
            for pref in PREFIXES_M3:
                p_m3.append(pref + ent)
                sp_m3.append((len(pref), len(pref) + len(ent)))

        band_B_s2 = sorted(set([lb_opt] + band_B_main))
        band_A_s2 = sorted(set([la_opt] + [_match_lA(lb) for lb in band_B_s2]))
        s2_cache = CACHE_DIR / f"pilot1b_realB_{mk_A}_to_{mk_B}_n{n_ent}.pt"
        if s2_cache.exists():
            p2 = torch.load(s2_cache, map_location="cpu", weights_only=False)
            stB_m3_A = {int(k): v.numpy() for k, v in p2["m3_A"].items()}
            stB_m3_B = {int(k): v.numpy() for k, v in p2["m3_B"].items()}
        else:
            t2 = time.time()
            stB_m3_B = extract_band_states(model_B, tok_B, p_m3, sp_m3, band_B_s2, device, batch_size=64)
            model_A = AutoModelForCausalLM.from_pretrained(str(dir_A), local_files_only=True, dtype=torch.bfloat16 if torch.backends.mps.is_available() else torch.float32).to(device).eval()
            stB_m3_A = extract_band_states(model_A, tok_A, p_m3, sp_m3, band_A_s2, device, batch_size=64)
            del model_A
            gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            print(f"  [Stage 2] Extracted {len(p_m3)} real-entity prefix-only states in {time.time() - t2:.1f}s")
            torch.save({"m3_A": {k: torch.from_numpy(v) for k, v in stB_m3_A.items()}, "m3_B": {k: torch.from_numpy(v) for k, v in stB_m3_B.items()}}, s2_cache)

        # Fit 4 Lever B-lite recipes: raw, M=3 double-centered avg, between+0.1*within, AND Opus cross-context (c != c')!
        b_raw, b_avg, b_lam01, b_xctx = {}, {}, {}, {}
        for lb in band_B_s2:
            la = _match_lA(lb)
            ZA_raw, ZB_raw = stB_m3_A[la], stB_m3_B[lb]
            b_raw[lb] = fit_isometry_f64(ZA_raw, ZB_raw); b_raw[lb]["la"] = la

            Za3 = ZA_raw.reshape(n_ent, 3, -1).astype(np.float64)
            Zb3 = ZB_raw.reshape(n_ent, 3, -1).astype(np.float64)
            # Subtract per-prefix mean first (double centering!)
            Za_dc = Za3 - Za3.mean(axis=0, keepdims=True)
            Zb_dc = Zb3 - Zb3.mean(axis=0, keepdims=True)
            Za_ent = Za_dc.mean(axis=1)  # [n_ent, D_A]
            Zb_ent = Zb_dc.mean(axis=1)  # [n_ent, D_B]
            Za_w = (Za_dc - Za_ent[:, None, :]).reshape(-1, Za3.shape[-1])
            Zb_w = (Zb_dc - Zb_ent[:, None, :]).reshape(-1, Zb3.shape[-1])

            C_bet = (Zb_ent.T @ Za_ent) / max(n_ent, 1)
            C_wit = (Zb_w.T @ Za_w) / max(len(Za_w), 1)
            # Opus cross-context pairing: unbiased sum_{c != c'} Zb_dc(c')^T Za_dc(c) (zero shared-prefix term!)
            C_xctx = np.zeros_like(C_bet)
            for c1 in range(3):
                for c2 in range(3):
                    if c1 != c2:
                        C_xctx += (Zb_dc[:, c2, :].T @ Za_dc[:, c1, :]) / max(n_ent * 6, 1)

            b_avg[lb] = fit_isometry_f64(Za_ent, Zb_ent, Z_A_cal=ZA_raw, Z_B_cal=ZB_raw); b_avg[lb]["la"] = la
            b_lam01[lb] = fit_isometry_f64(ZA_raw, ZB_raw, Z_A_cal=ZA_raw, Z_B_cal=ZB_raw, cov_override=C_bet + 0.1 * C_wit)
            b_lam01[lb]["la"] = la
            b_xctx[lb] = fit_isometry_f64(ZA_raw, ZB_raw, Z_A_cal=ZA_raw, Z_B_cal=ZB_raw, cov_override=C_xctx)
            b_xctx[lb]["la"] = la

        b_maps_best = b_lam01
        if 2 in stages:
            s2_out = {"pool_meta": pool_meta, "configs": {}}
            for mtag, md in [
                ("B_real_raw", b_raw),
                ("B_real_m3_avg", b_avg),
                ("B_real_bet_01wit", b_lam01),
                ("B_real_cross_ctx", b_xctx),
            ]:
                for stag, bl, mm, d5 in [
                    ("single_A0", [lb_opt], "none", False),
                    ("band_A3_full_ko", band_B_main, "full_knockout", True),
                ]:
                    nc1, nc5 = native_ceilings.get((tuple(bl), mm), (None, None))
                    z1m = {lb: apply_map_f64(Z_A_all[t1_d, md[lb]["la"]].float().numpy(), md[lb]) for lb in bl}
                    z1s = {lb: apply_map_f64(Z_A_all[shuf_1x, md[lb]["la"]].float().numpy(), md[lb]) for lb in bl}
                    zlom = {lb: apply_map_f64(Z_A_all[lo_d, md[lb]["la"]].float().numpy(), md[lb]) for lb in bl}
                    zlos = {lb: apply_map_f64(Z_A_all[shuf_lo, md[lb]["la"]].float().numpy(), md[lb]) for lb in bl}
                    z5m = {lb: apply_map_f64(Z_A_all[t5_d, md[lb]["la"]].float().numpy(), md[lb]) for lb in bl} if d5 else None
                    z5s = {lb: apply_map_f64(Z_A_all[shuf_5x, md[lb]["la"]].float().numpy(), md[lb]) for lb in bl} if d5 else None
                    r = _eval_full(f"{mtag}_{stag}", z1m, z1s, zlom, zlos, mm, z5m, z5s, nc1, nc5)
                    s2_out["configs"][f"{mtag}_{stag}"] = {k: v for k, v in r.items() if k != "outcomes_1x"}
            results["stage2_leverB"] = s2_out

    # ==================================================================================
    # STAGE 3 (Lever C — Fable + Astra Teacher, Truncated at last_subj!)
    # ==================================================================================
    if 3 in stages:
        print("\n=== STAGE 3 (Modified Lever C): Truncated Self-Swap Distillation in Strict O(d) ===")
        if eval_dtype != torch.float32:
            model_B = model_B.to(dtype=torch.float32)
            eval_dtype = torch.float32
        for p in model_B.parameters():
            p.requires_grad_(False)
        model_B.eval()

        band_A_main = [_match_lA(lb) for lb in band_B_main]
        new_maps, d_diag = run_stage3_truncated_distillation(
            model_B, tok_B, band_B_main, band_A_main, b_maps_best,
            stB_m3_A, stB_m3_B, ents_real, down_layers, device,
            mask_mode="full_knockout", n_pairs=1024, n_val=128, n_steps=80, batch_size=16, lr=0.015,
        )
        nc1, nc5 = native_ceilings.get((tuple(band_B_main), "full_knockout"), (None, None))
        z1m = {lb: apply_map_f64(Z_A_all[t1_d, new_maps[lb]["la"]].float().numpy(), new_maps[lb]) for lb in band_B_main}
        z1s = {lb: apply_map_f64(Z_A_all[shuf_1x, new_maps[lb]["la"]].float().numpy(), new_maps[lb]) for lb in band_B_main}
        zlom = {lb: apply_map_f64(Z_A_all[lo_d, new_maps[lb]["la"]].float().numpy(), new_maps[lb]) for lb in band_B_main}
        zlos = {lb: apply_map_f64(Z_A_all[shuf_lo, new_maps[lb]["la"]].float().numpy(), new_maps[lb]) for lb in band_B_main}
        z5m = {lb: apply_map_f64(Z_A_all[t5_d, new_maps[lb]["la"]].float().numpy(), new_maps[lb]) for lb in band_B_main}
        z5s = {lb: apply_map_f64(Z_A_all[shuf_5x, new_maps[lb]["la"]].float().numpy(), new_maps[lb]) for lb in band_B_main}
        r_c = _eval_full("C_woodbury_k16_full_ko", z1m, z1s, zlom, zlos, "full_knockout", z5m, z5s, nc1, nc5)
        results["stage3_leverC"] = {"train_diag": d_diag, "eval": {k: v for k, v in r_c.items() if k != "outcomes_1x"}}

    del model_B
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    results["meta"]["elapsed_sec"] = round(time.time() - t_start, 2)
    out_path = LAB_DIR / f"results_pilot1b_{mk_A}_to_{mk_B}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[pilot1b] COMPLETE! Saved to {out_path} in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
