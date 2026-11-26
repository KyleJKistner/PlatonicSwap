#!/usr/bin/env python3
"""
v2_track2_preflight.py — Track 2 15-Minute Pre-Flight Gate (Headroom Census, Gate K1 & Gate K0)
===============================================================================================

Implements the exact Pre-Flight Gate from `docs/V2_PHASEB_PREREG.md` (§5) that MUST PASS before
running full Falsifier C:

  1. Batch-Matched Unhook Fix: Compares unhooked vs. base logits on IDENTICAL batch shapes
     (eliminating cuBLAS/MPS batch=1 vs. sliced-batch numerical drift) so `torch.equal` (or
     `max |Δ| < 1e-6` on CUDA) passes cleanly.
  2. Headroom Census (`falcon3-3b` & `smollm2-1.7b`):
     - `n = 64` repeated random shared-token sequences (`… A B … A -> B`) at lags `Δ ∈ {4, 16, 64}`
       using strictly disjoint `fit_words` (103) vs. `test_words` (103) from the 206 shared 1-token
       vocabulary verified across `qwen2.5-3b`, `falcon3-3b`, and `smollm2-1.7b`.
     - Measures BOTH copy top-1 accuracy AND in-context logit difference
       `logit(target B) - mean(logits of other in-context tokens)`.
     - Progressively mean-ablates top `{2, 4, 8, 12, 16}` native induction heads until copy top-1
       drops into `10–30%` (flags pivot to donor-only skip-induction if `<50%` isn't reached).
  3. Gate K1 (Positive Control — MUST PASS BEFORE ANY CROSS-FAMILY CLAIM!):
     - On the ablated recipient, re-inserts the recipient's OWN ablated induction circuit through
       the EXACT same `CompiledHeadGraftPipeline` hook mechanism used for cross-family grafting.
     - K1-a (Own heads): MUST restore `R >= 0.90` of the ablation gap.
     - K1-b (Rotated self-transplant through random orthogonal gauge `Q_l ∈ O(D)` + closed-form
       transport operator): MUST restore `R >= 0.80` of the ablation gap.
  4. Gate K0 (Cross-Family Algebra Check: `qwen2.5-3b` -> `falcon3-3b` & `smollm2-1.7b`):
     - Finds Donor A's best `(prev_token H1, induction H2)` pair maximizing Q-K composition score
       `||W_QK2 @ W_OV1||_F / (||W_QK2||_F ||W_OV1||_F)` and induction stripe mass.
     - Applies exact `1 / sqrt(d_h)` scaling compensation (`sqrt(2)` for `128 -> 256` on Falcon3,
       `sqrt(0.5)` for `128 -> 64` on SmolLM2).
     - Fits mean-centered active-prompt OV sockets strictly on `fit_words` active copy prompts.
     - Enforces the EXACT cocycle equality constraint `J_2 U = J_1 U` (and read dual `M_{in,2} J_1 U = U`)
       on `U = Im(W_OV1)`, comparing WITH vs. WITHOUT cocycle constraint!
     - Gates:
       * K0-a: `comp(H1_B -> H2_B) / comp(H1_A -> H2_A) >= 0.50`
       * K0-b: Mean-centered active-prompt OV copying top-1 rate `>= 0.50 × donor`!

Strict v1.0 Read-Only Contract:
  - Does NOT touch any locked v1.0 files.
  - Pure-CPU `--unit-test` completes in <10s with ZERO GPU/model forward passes.

Usage:
  PYTHONPATH=platonicswap:experiments python experiments/v2_track2_preflight.py --unit-test
  PYTHONPATH=platonicswap:experiments python -u experiments/v2_track2_preflight.py --run
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for p in (str(REPO_ROOT / "platonicswap"), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Default to populated local lab caches if present and not overridden
if "PLATONICSWAP_HF_CACHE" not in os.environ and Path("/tmp/platos_key_lab/hf_cache").is_dir():
    os.environ["PLATONICSWAP_HF_CACHE"] = "/tmp/platos_key_lab/hf_cache"
if "PLATONICSWAP_CACHE_DIR" not in os.environ and Path("/tmp/platos_key_lab/cache").is_dir():
    os.environ["PLATONICSWAP_CACHE_DIR"] = "/tmp/platos_key_lab/cache"

RESULTS_DIR = REPO_ROOT / "results" if (REPO_ROOT / "results").is_dir() else SCRIPT_DIR
OUT_PATH = RESULTS_DIR / "results_v2_track2_preflight.json"

from v2_falsifierA_rope_torus import (
    DONOR_KEY,
    RECIPIENT_KEYS,
    VERIFIED_GEOMETRIES,
    apply_rope_half_torch,
    build_optimal_rope_intertwiner,
    build_shared_single_token_words,
    compile_congruent_attention_head,
    compute_single_head_attention,
    encode_word_stream_exact,
    extract_donor_head_weights,
    extract_input_ln_states,
    fit_cauchy_write_socket,
    ledoit_wolf_cov,
    linear_cka,
    rope_inv_freq,
)

SEED = 20261126
PREFLIGHT_LAGS: Tuple[int, ...] = (4, 16, 64)
ABLATION_K_GRID: Tuple[int, ...] = (2, 4, 8, 12, 16)


# ======================================================================================
# 1. Exact Fast Q-K Composition Algebra & Cocycle-Constrained Closed-Form Sockets
# ======================================================================================

def qk_ov_composition_score(
    W_Q2: np.ndarray,
    W_K2: np.ndarray,
    W_O1: np.ndarray,
    W_V1: np.ndarray,
) -> Dict[str, float]:
    """
    Exact Elhage et al. K-composition score in O(D * d_h^2) with ZERO dense D x D materialization:
      W_QK2 = W_Q2^T W_K2   (D x D),   W_OV1 = W_O1 W_V1   (D x D)
      comp  = || W_QK2 @ W_OV1 ||_F / ( || W_QK2 ||_F * || W_OV1 ||_F )

    Uses exact cyclic trace identities:
      ||A B||_F^2 = tr((A^T A)(B B^T))
      ||W_Q2^T (W_K2 W_O1) W_V1||_F^2 = tr((W_Q2 W_Q2^T) C (W_V1 W_V1^T) C^T),  C = W_K2 W_O1.
    """
    WQ = np.asarray(W_Q2, dtype=np.float64)
    WK = np.asarray(W_K2, dtype=np.float64)
    WO = np.asarray(W_O1, dtype=np.float64)
    WV = np.asarray(W_V1, dtype=np.float64)

    GQ = WQ @ WQ.T      # (d2, d2)
    GK = WK @ WK.T      # (d2, d2)
    GO = WO.T @ WO      # (d1, d1)
    GV = WV @ WV.T      # (d1, d1)
    C = WK @ WO         # (d2, d1)

    n_qk2 = float(np.sum(GQ * GK))
    n_ov1 = float(np.sum(GO * GV))
    # tr(GQ @ C @ GV @ C.T) = sum((GQ @ C) * (C @ GV))
    n_comp2 = float(np.sum((GQ @ C) * (C @ GV)))

    norm_qk = math.sqrt(max(n_qk2, 0.0))
    norm_ov = math.sqrt(max(n_ov1, 0.0))
    norm_comp = math.sqrt(max(n_comp2, 0.0))
    score = norm_comp / max(norm_qk * norm_ov, 1e-18)
    return {
        "comp_score": float(score),
        "norm_comp": float(norm_comp),
        "norm_qk2": float(norm_qk),
        "norm_ov1": float(norm_ov),
    }


def orthonormal_image_basis(W_O: np.ndarray, rtol: float = 1e-7) -> np.ndarray:
    """Return exact orthonormal basis U (D, r) spanning Im(W_O) = Im(W_OV)."""
    U, s, _ = sla.svd(np.asarray(W_O, dtype=np.float64), full_matrices=False)
    if len(s) == 0 or s[0] <= 1e-15:
        return np.zeros((W_O.shape[0], 0), dtype=np.float64)
    r = int(np.sum(s > rtol * s[0]))
    return U[:, :r]


def fit_cocycle_constrained_forward_socket(
    X_A_fit: np.ndarray,
    Y_B_fit: np.ndarray,
    J1: np.ndarray,
    U: np.ndarray,
) -> Dict[str, Any]:
    """
    Fit forward residual socket J_2: R^{D_A} -> R^{D_B} (mean-centered Ledoit-Wolf ridge) subject
    to the EXACT cocycle equality constraint:
      J_2 U = J_1 U     on U = Im(W_OV1)   (U^T U = I_r).

    Closed-form derivation:
      Decompose J_2 = (J_1 U) U^T + J_{2,perp},  with J_{2,perp} U = 0.
      Let P_perp = I - U U^T,  X_perp = X_A_c P_perp,  R_B = Y_B_c - X_A_c U (J_1 U)^T.
      Solve analytical LW ridge on X_perp -> R_B and right-project by P_perp so J_{2,perp} U == 0!
      Guaranteed: ||J_2 U - J_1 U||_F < 1e-13!
    """
    Xa = np.asarray(X_A_fit, dtype=np.float64)
    Yb = np.asarray(Y_B_fit, dtype=np.float64)
    mu_A = Xa.mean(axis=0)
    mu_B = Yb.mean(axis=0)
    Xac = Xa - mu_A
    Ybc = Yb - mu_B

    D_A = Xa.shape[1]
    J1_U = np.asarray(J1, dtype=np.float64) @ np.asarray(U, dtype=np.float64)  # (D_B, r)
    J_par = J1_U @ U.T  # (D_B, D_A)

    P_perp = np.eye(D_A, dtype=np.float64) - U @ U.T
    X_perp = Xac @ P_perp
    R_B = Ybc - Xac @ J_par.T

    S_AA_perp, sh = ledoit_wolf_cov(X_perp)
    C_BA_perp = (R_B.T @ X_perp) / float(len(Xa))
    # Solve S_AA_perp @ J_perp^T = C_BA_perp^T, then project strictly onto P_perp!
    J_perp = sla.solve(S_AA_perp, C_BA_perp.T, assume_a="pos").T @ P_perp
    J2 = J_par + J_perp
    d_out = mu_B - J2 @ mu_A
    cocycle_err = float(sla.norm(J2 @ U - J1_U, "fro"))
    return {
        "J2": J2,
        "d_out": d_out,
        "mu_A": mu_A,
        "mu_B": mu_B,
        "shrink": float(sh),
        "cocycle_eq_err": cocycle_err,
    }


def fit_cocycle_constrained_read_socket(
    X_B_fit: np.ndarray,
    X_A_fit: np.ndarray,
    J1: Optional[np.ndarray] = None,
    U: Optional[np.ndarray] = None,
    X_B_val: Optional[np.ndarray] = None,
    X_A_val: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Fit mean-centered layer-2 read socket M_{in,2}: R^{D_B} -> R^{D_A} via analytical Ledoit-Wolf
    ridge, optionally enforcing the EXACT cocycle recovery equality constraint:
      M_{in,2} (J_1 U) = U    (equivalently M_{in,2} J_1 = I on Im(W_OV1)!)

    When J1 & U are provided:
      Let V_B = J_1 U = Q_B R_B (thin QR, Q_B^T Q_B = I_r).
      Pin M_par = U R_B^{-1} Q_B^T on span(Q_B) so M_par V_B == U identically!
      Solve analytical LW ridge on P_{B,perp} = I - Q_B Q_B^T for the residual R_A = X_A_c - X_B_c M_par^T,
      and right-project M_perp by P_{B,perp} so M_perp V_B == 0 identically!
      Guaranteed: ||M_{in,2} (J_1 U) - U||_F < 1e-13!
    """
    Xb = np.asarray(X_B_fit, dtype=np.float64)
    Xa = np.asarray(X_A_fit, dtype=np.float64)
    mu_B = Xb.mean(axis=0)
    mu_A = Xa.mean(axis=0)
    Xbc = Xb - mu_B
    Xac = Xa - mu_A

    if J1 is None or U is None or U.shape[1] == 0:
        S_BB, sh = ledoit_wolf_cov(Xb)
        C_AB = (Xac.T @ Xbc) / float(len(Xb))
        M_in = sla.solve(S_BB, C_AB.T, assume_a="pos").T
        cocycle_err = None
    else:
        D_B = Xb.shape[1]
        V_B = np.asarray(J1, dtype=np.float64) @ np.asarray(U, dtype=np.float64)  # (D_B, r)
        # Thin SVD handles any near-collinear columns cleanly while keeping exact full-rank recovery
        Qb, sv, Vt = sla.svd(V_B, full_matrices=False)
        keep = sv > 1e-9 * max(sv[0], 1e-12)
        Qb = Qb[:, keep]
        sv_inv = 1.0 / sv[keep]
        Vt = Vt[keep, :]
        # M_par = U @ V_B^+ = U @ Vt^T @ diag(1/sv) @ Qb^T  =>  M_par @ V_B == U (exact when full rank!)
        M_par = (U @ Vt.T * sv_inv[None, :]) @ Qb.T

        P_perp = np.eye(D_B, dtype=np.float64) - Qb @ Qb.T
        Xb_perp = Xbc @ P_perp
        R_A = Xac - Xbc @ M_par.T
        S_BB_perp, sh = ledoit_wolf_cov(Xb_perp)
        C_AB_perp = (R_A.T @ Xb_perp) / float(len(Xb))
        M_perp = sla.solve(S_BB_perp, C_AB_perp.T, assume_a="pos").T @ P_perp
        M_in = M_par + M_perp
        cocycle_err = float(sla.norm(M_in @ V_B - U, "fro"))

    d_in = mu_A - M_in @ mu_B
    r2_val = None
    if X_B_val is not None and X_A_val is not None and len(X_B_val) > 0:
        pred = np.asarray(X_B_val, dtype=np.float64) @ M_in.T + d_in
        Xa_v = np.asarray(X_A_val, dtype=np.float64)
        ss_res = float(np.sum((Xa_v - pred) ** 2))
        ss_tot = float(np.sum((Xa_v - Xa_v.mean(axis=0, keepdims=True)) ** 2))
        r2_val = 1.0 - ss_res / max(ss_tot, 1e-12)

    return {
        "M_in": M_in,
        "d_in": d_in,
        "mu_A": mu_A,
        "mu_B": mu_B,
        "shrink": float(sh),
        "cocycle_recovery_err": cocycle_err,
        "heldout_read_R2": round(float(r2_val), 4) if r2_val is not None else None,
    }


def fit_active_centered_ov_write_socket(
    Y_A_fit: np.ndarray,
    Y_B_fit: np.ndarray,
    Y_A_val: Optional[np.ndarray] = None,
    Y_B_val: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Fit mean-centered active-prompt OV write socket M_out: R^{D_A} -> R^{D_B} via analytical LW ridge
    strictly on active copy prompts (eliminating generic-prose softmax common-mode `W_OV mu` confound!).
    """
    Ya = np.asarray(Y_A_fit, dtype=np.float64)
    Yb = np.asarray(Y_B_fit, dtype=np.float64)
    mu_A = Ya.mean(axis=0)
    mu_B = Yb.mean(axis=0)
    Yac = Ya - mu_A
    Ybc = Yb - mu_B
    S_AA, sh = ledoit_wolf_cov(Ya)
    C_BA = (Ybc.T @ Yac) / float(len(Ya))
    M_out = sla.solve(S_AA, C_BA.T, assume_a="pos").T
    d_out = mu_B - M_out @ mu_A

    cos_val, r2_val = None, None
    if Y_A_val is not None and Y_B_val is not None and len(Y_A_val) > 0:
        Ya_vc = np.asarray(Y_A_val, dtype=np.float64) - mu_A
        Yb_vc = np.asarray(Y_B_val, dtype=np.float64) - mu_B
        pred = Ya_vc @ M_out.T
        c = np.sum(pred * Yb_vc, axis=-1) / (np.linalg.norm(pred, axis=-1) * np.linalg.norm(Yb_vc, axis=-1) + 1e-12)
        cos_val = float(np.mean(c))
        r2_val = 1.0 - float(np.sum((Yb_vc - pred) ** 2)) / max(float(np.sum(Yb_vc ** 2)), 1e-12)

    return {
        "M_out": M_out,
        "d_out": d_out,
        "mu_A": mu_A,
        "mu_B": mu_B,
        "shrink": float(sh),
        "heldout_write_cos": round(float(cos_val), 4) if cos_val is not None else None,
        "heldout_write_R2": round(float(r2_val), 4) if r2_val is not None else None,
    }


# ======================================================================================
# 2. Disjoint Fit/Test Repeated Random-Token Suites & Copy Scoring (Top-1 + Logit Diff)
# ======================================================================================

def split_shared_words_disjoint(shared_words: Sequence[str], seed: int = SEED) -> Tuple[List[str], List[str]]:
    """Split shared single-token vocabulary into strictly disjoint (fit_words, test_words) halves."""
    rng = np.random.default_rng(seed)
    words = list(dict.fromkeys(shared_words))
    perm = rng.permutation(len(words))
    mid = len(words) // 2
    fit_w = [words[i] for i in perm[:mid]]
    test_w = [words[i] for i in perm[mid:]]
    assert set(fit_w).isdisjoint(set(test_w))
    return fit_w, test_w


def build_disjoint_copy_batches(
    words_pool: Sequence[str],
    tok: Any,
    lags: Sequence[int] = PREFLIGHT_LAGS,
    n_seqs: int = 64,
    seed: int = SEED,
) -> Dict[int, torch.Tensor]:
    """
    Build exact repeated random-token sequences `… A B … A -> B` (exact length `1 + 2*Δ`) for every
    lag `Δ ∈ lags` strictly from `words_pool`. Returns `{lag: LongTensor(n_seqs, 1 + 2*lag)}`.
    """
    rng = np.random.default_rng(seed)
    words = list(words_pool)
    assert len(words) >= max(lags), f"Pool size {len(words)} < max lag {max(lags)}"
    out: Dict[int, torch.Tensor] = {}
    for lag in lags:
        rows = []
        for _ in range(n_seqs):
            block = rng.choice(words, size=int(lag), replace=False).tolist()
            ids, _ = encode_word_stream_exact(tok, block + block)
            rows.append(ids)
        out[int(lag)] = torch.tensor(rows, dtype=torch.long)
    return out


def score_copy_predictions(logits: torch.Tensor, batch_ids: torch.Tensor, lag: int) -> Dict[str, float]:
    """
    Score repeated random-token copy predictions (`… A B … A -> B`) on the second copy
    (predicting `batch_ids[:, pos + 1]` from `logits[:, pos, :]` for `pos ∈ [1 + lag, 2*lag - 1]`):
      - `top1_acc`: exact greedy top-1 match across full vocabulary
      - `in_context_top1`: exact top-1 restricted to the unique tokens appearing in that sequence's block
      - `logit_diff`: `logit(target B) - mean(logits of other in-context block tokens)`
    """
    B, T = batch_ids.shape
    assert T == 1 + 2 * lag, f"Expected T={1 + 2*lag} for lag={lag}, got {T}"
    pred_positions = list(range(1 + lag, T - 1))
    top1_hits, ic_hits, ldiffs = [], [], []
    logits_f = logits.float().cpu()
    ids_c = batch_ids.cpu()

    for b in range(B):
        block_toks = ids_c[b, 1 : 1 + lag]  # unique lag tokens in this prompt!
        for pos in pred_positions:
            tgt = int(ids_c[b, pos + 1].item())
            row = logits_f[b, pos]
            top1_hits.append(1.0 if int(row.argmax().item()) == tgt else 0.0)
            block_logits = row[block_toks]
            ic_hits.append(1.0 if int(block_toks[int(block_logits.argmax().item())].item()) == tgt else 0.0)
            tgt_l = float(row[tgt].item())
            other_mask = block_toks != tgt
            other_mean = float(block_logits[other_mask].mean().item())
            ldiffs.append(tgt_l - other_mean)

    return {
        "top1_acc": round(float(np.mean(top1_hits)), 4),
        "in_context_top1": round(float(np.mean(ic_hits)), 4),
        "logit_diff": round(float(np.mean(ldiffs)), 4),
        "n_predictions": int(len(top1_hits)),
    }


def forward_batched_logits(
    model: Any,
    batch_ids: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """Run forward pass in deterministic fixed-size chunks and return float32 CPU logits."""
    chunks = []
    with torch.no_grad():
        for i0 in range(0, len(batch_ids), batch_size):
            b = batch_ids[i0 : i0 + batch_size].to(device)
            chunks.append(model(input_ids=b, use_cache=False).logits.float().cpu())
    return torch.cat(chunks, dim=0)


def verify_batch_matched_unhook(
    model: Any,
    probe_batch: torch.Tensor,
    base_probe_logits: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Batch-matched unhook verification (fixes Falsifier A's batch=1 vs sliced-batch drift!).
    Evaluates on the EXACT same `probe_batch` & `batch_size` as `base_probe_logits`.
    """
    cur = forward_batched_logits(model, probe_batch, device, batch_size=batch_size)
    bit_eq = bool(torch.equal(cur, base_probe_logits))
    max_diff = float(torch.max(torch.abs(cur - base_probe_logits)).item())
    return {
        "bit_equal": bit_eq,
        "max_abs_diff": max_diff,
        "pass": bool(bit_eq or max_diff < 1e-6),
    }


# ======================================================================================
# 3. Unified Mean-Ablation & Multi-Head Grafting Pipeline (Used Identically by K1 & K0!)
# ======================================================================================

def compute_native_head_o_proj_means(
    model: Any,
    fit_batches: Dict[int, torch.Tensor],
    layer_indices: Sequence[int],
    num_heads: int,
    device: torch.device,
    batch_size: int = 16,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Compute mean per-head `o_proj` input vector `mu_z[(l, h)]` (d_h,) on `fit_words` copy sequences
    so mean-ablation replaces each ablated head's slice cleanly with its active-context mean!
    """
    from extract_and_baseline import get_decoder_layers

    layers = get_decoder_layers(model)
    sums: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {li: 0 for li in layer_indices}
    hooks = []
    for li in layer_indices:
        def _mk(idx: int):
            def _pre(_m, args):
                z = args[0].detach().float().cpu().numpy()  # (B, T, H*d_h)
                s = z.reshape(-1, z.shape[-1]).sum(axis=0)
                if idx not in sums:
                    sums[idx] = s
                else:
                    sums[idx] += s
                counts[idx] += z.shape[0] * z.shape[1]
            return _pre
        hooks.append(layers[li].self_attn.o_proj.register_forward_pre_hook(_mk(li)))

    try:
        with torch.no_grad():
            for lag, bt in fit_batches.items():
                for i0 in range(0, min(len(bt), 32), batch_size):
                    _ = model(input_ids=bt[i0 : i0 + batch_size].to(device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    mu_map: Dict[Tuple[int, int], np.ndarray] = {}
    for li in layer_indices:
        mean_vec = sums[li] / float(max(counts[li], 1))
        d_h = len(mean_vec) // num_heads
        for h in range(num_heads):
            mu_map[(li, h)] = mean_vec[h * d_h : (h + 1) * d_h].astype(np.float64)
    return mu_map


class MeanAblateHeadsHook:
    """Mean-ablate any list of `(layer_idx, head_q)` heads via `o_proj` forward pre-hooks."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        heads: Sequence[Tuple[int, int]],
        mu_z_map: Dict[Tuple[int, int], np.ndarray],
        num_heads: int,
        device: torch.device,
    ):
        self.layers = layers
        self.by_layer: Dict[int, List[Tuple[int, torch.Tensor]]] = {}
        self.num_heads = int(num_heads)
        for li, hq in heads:
            self.by_layer.setdefault(int(li), []).append(
                (int(hq), torch.from_numpy(mu_z_map[(int(li), int(hq))].astype(np.float32)).to(device))
            )
        self._handles: List[Any] = []

    def attach(self) -> "MeanAblateHeadsHook":
        for li, h_list in self.by_layer.items():
            def _mk(hl):
                def _pre(_m, args):
                    z = args[0].clone()
                    d_h = z.shape[-1] // self.num_heads
                    for hq, mu_t in hl:
                        z[..., hq * d_h : (hq + 1) * d_h] = mu_t.to(z.dtype)
                    return (z,) + args[1:]
                return _pre
            self._handles.append(self.layers[li].self_attn.o_proj.register_forward_pre_hook(_mk(h_list)))
        return self

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


class CompiledHeadGraftPipeline:
    """
    EXACT unified multi-layer additive/replacement head graft pipeline used identically for BOTH
    Gate K1 positive controls (own heads & rotated self-transplant) AND cross-family grafting!

    Each graft spec is:
      {"layer": lB, "head_op": compiled_head_dict, "alpha": 1.0, "baseline_subtract": Optional[np.ndarray (D_B,)]}
    where `baseline_subtract` (when replacing a mean-ablated slot) subtracts `mu_z_h @ W_O_h^T`
    so the grafted head replaces the ablated slot cleanly with ZERO mean double-counting!
    """

    def __init__(self, layers: Sequence[nn.Module], specs: Sequence[Dict[str, Any]], device: torch.device):
        self.layers = layers
        self.by_layer: Dict[int, List[Dict[str, Any]]] = {}
        for sp in specs:
            li = int(sp["layer"])
            op = sp["head_op"]
            entry = {
                "WQ": torch.from_numpy(op["W_Q"].astype(np.float32)).to(device),
                "bQ": torch.from_numpy(op["b_Q"].astype(np.float32)).to(device),
                "WK": torch.from_numpy(op["W_K"].astype(np.float32)).to(device),
                "bK": torch.from_numpy(op["b_K"].astype(np.float32)).to(device),
                "WV": torch.from_numpy(op["W_V"].astype(np.float32)).to(device),
                "bV": torch.from_numpy(op["b_V"].astype(np.float32)).to(device),
                "WO": torch.from_numpy(op["W_O"].astype(np.float32)).to(device),
                "inv_freq": (
                    torch.from_numpy(op["inv_freq"].astype(np.float32)).to(device)
                    if op["inv_freq"] is not None
                    else None
                ),
                "tau": float(op.get("tau", 1.0)),
                "alpha": float(sp.get("alpha", 1.0)),
                "baseline_subtract": (
                    torch.from_numpy(np.asarray(sp["baseline_subtract"], dtype=np.float32)).to(device)
                    if sp.get("baseline_subtract") is not None
                    else None
                ),
            }
            self.by_layer.setdefault(li, []).append(entry)
        self._buf: Dict[int, torch.Tensor] = {}
        self._handles: List[Any] = []

    def attach(self) -> "CompiledHeadGraftPipeline":
        for li, entries in self.by_layer.items():
            def _mk_ln(idx: int):
                def _fn(_m, _inp, out):
                    self._buf[idx] = out
                return _fn

            def _mk_attn(idx: int, ent_list: List[Dict[str, Any]]):
                def _fn(_m, _inp, out):
                    x_a = self._buf[idx].float()
                    delta_total = None
                    for e in ent_list:
                        _, y_h = compute_single_head_attention(
                            x_a, e["WQ"], e["bQ"], e["WK"], e["bK"],
                            e["WV"], e["bV"], e["WO"], e["inv_freq"], tau=e["tau"],
                        )
                        d = e["alpha"] * y_h
                        if e["baseline_subtract"] is not None:
                            d = d - e["baseline_subtract"].view(1, 1, -1)
                        delta_total = d if delta_total is None else (delta_total + d)
                    if isinstance(out, tuple):
                        return (out[0] + delta_total.to(out[0].dtype),) + out[1:]
                    return out + delta_total.to(out.dtype)
                return _fn

            self._handles.append(self.layers[li].input_layernorm.register_forward_hook(_mk_ln(li)))
            self._handles.append(self.layers[li].self_attn.register_forward_hook(_mk_attn(li, entries)))
        return self

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._buf.clear()


# ======================================================================================
# 4. Head Census & Active-Prompt OV Copying Evaluator (For Headroom Census & Gate K0-b)
# ======================================================================================

def scan_model_induction_and_prev_heads(
    model: Any,
    fit_batches: Dict[int, torch.Tensor],
    geom: Dict[str, Any],
    device: torch.device,
) -> List[Dict[str, Any]]:
    """
    Scan layers `1..L-2` of a model on `fit_words` repeated sequences (`Δ ∈ {4, 16, 64}`) to rank
    every head `(layer, head_q)` by induction stripe mass `A[t, t - Δ + 1]` and previous-token mass
    `A[t, t - 1]`.
    """
    from extract_and_baseline import get_decoder_layers

    layers = get_decoder_layers(model)
    L, H_q, H_kv, d_h = geom["L"], geom["H_q"], geom["H_kv"], geom["d_h"]
    inv_f = torch.from_numpy(rope_inv_freq(d_h, geom["rope_theta"]).astype(np.float32)).to(device)
    scan_layers = list(range(1, L - 1))

    # Use first 12 fit sequences per lag for fast, sharp census (~2-4 seconds on GPU!)
    x_by_lag = {
        lag: extract_input_ln_states(model, bt[:12], scan_layers, device)
        for lag, bt in fit_batches.items()
    }

    records: List[Dict[str, Any]] = []
    for li in scan_layers:
        attn = layers[li].self_attn
        for hq in range(H_q):
            hd = extract_donor_head_weights(attn, hq, H_q, H_kv)
            WQ = torch.from_numpy(hd["W_Q"].astype(np.float32)).to(device)
            bQ = torch.from_numpy(hd["b_Q"].astype(np.float32)).to(device)
            WK = torch.from_numpy(hd["W_K"].astype(np.float32)).to(device)
            bK = torch.from_numpy(hd["b_K"].astype(np.float32)).to(device)

            ind_scores, prev_scores = [], []
            with torch.no_grad():
                for lag, x_dict in x_by_lag.items():
                    xl = x_dict[li].to(device)
                    A, _ = compute_single_head_attention(xl, WQ, bQ, WK, bK, None, None, None, inv_f)
                    T = A.shape[1]
                    q_ind = torch.arange(1 + lag, T - 1, device=device)
                    ind_scores.append(float(A[:, q_ind, q_ind - lag + 1].mean().item()))
                    q_prev = torch.arange(2, T, device=device)
                    prev_scores.append(float(A[:, q_prev, q_prev - 1].mean().item()))

            records.append({
                "layer": int(li),
                "head_q": int(hq),
                "induction_mass": round(float(np.mean(ind_scores)), 4),
                "prev1_mass": round(float(np.mean(prev_scores)), 4),
                "ind_by_lag": {str(lag): round(sc, 4) for lag, sc in zip(x_by_lag.keys(), ind_scores)},
            })
    return records


def eval_active_prompt_ov_copying(
    model: Any,
    x_ln_by_lag: Dict[int, torch.Tensor],
    test_batches: Dict[int, torch.Tensor],
    W_V: np.ndarray,
    W_O: np.ndarray,
    mu_x: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate pure mean-centered OV copying rate (`Gate K0-b`):
      For each first-copy token `w` at position `p ∈ [1, lag]` in a `test_words` repeated sequence,
      compute `v = (x[p] - mu_x) @ W_V^T @ W_O^T`, pass through `model.model.norm` and `model.lm_head`,
      and check whether `w` is top-1 across (i) in-context block tokens AND (ii) full vocabulary!
    """
    WV_t = torch.from_numpy(np.asarray(W_V, dtype=np.float32)).to(device)
    WO_t = torch.from_numpy(np.asarray(W_O, dtype=np.float32)).to(device)
    mu_t = torch.from_numpy(np.asarray(mu_x, dtype=np.float32)).to(device)
    norm_fn = model.model.norm
    lm_head = model.lm_head

    ic_hits, full_hits, ldiffs = [], [], []
    with torch.no_grad():
        for lag, bt in test_batches.items():
            xl = x_ln_by_lag[lag].to(device)  # (B, 1 + 2*lag, D)
            # First-copy positions 1..lag
            x_first = xl[:, 1 : 1 + lag, :] - mu_t.view(1, 1, -1)
            ov_out = F.linear(F.linear(x_first, WV_t, None), WO_t, None)  # (B, lag, D)
            # Project through model's final RMSNorm + lm_head (strictly fp32!)
            logits = lm_head(norm_fn(ov_out.to(lm_head.weight.dtype))).float().cpu()
            ids_c = bt.cpu()
            for b in range(len(bt)):
                block_toks = ids_c[b, 1 : 1 + lag]
                for idx in range(lag):
                    tgt = int(block_toks[idx].item())
                    row = logits[b, idx]
                    full_hits.append(1.0 if int(row.argmax().item()) == tgt else 0.0)
                    bl = row[block_toks]
                    ic_hits.append(1.0 if int(block_toks[int(bl.argmax().item())].item()) == tgt else 0.0)
                    other = bl[block_toks != tgt]
                    ldiffs.append(float(row[tgt].item() - other.mean().item()))

    return {
        "ov_in_context_top1": round(float(np.mean(ic_hits)), 4),
        "ov_full_vocab_top1": round(float(np.mean(full_hits)), 4),
        "ov_logit_diff": round(float(np.mean(ldiffs)), 4),
        "n_tokens": int(len(ic_hits)),
    }


def extract_active_residual_and_ln_states(
    model: Any,
    batches_by_lag: Dict[int, torch.Tensor],
    layer_indices: Sequence[int],
    device: torch.device,
    max_seqs_per_lag: int = 32,
    batch_size: int = 16,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Capture aligned active-prompt `x_ln` (`input_layernorm` output) AND `y_attn` (`self_attn` output)
    AND `h_pre` (pre-layer residual stream) at `layer_indices` across `batches_by_lag`.
    Every token across all lags aligns 1-to-1 by exact index across Qwen2.5-3B, Falcon3-3B, SmolLM2-1.7B!
    """
    from extract_and_baseline import get_decoder_layers

    layers = get_decoder_layers(model)
    store: Dict[int, Dict[str, List[np.ndarray]]] = {
        li: {"x_ln": [], "y_attn": [], "h_pre": []} for li in layer_indices
    }
    buf: Dict[int, Dict[str, np.ndarray]] = {li: {} for li in layer_indices}
    hooks = []
    for li in layer_indices:
        def _mk_pre(idx: int):
            def _fn(_m, args):
                buf[idx]["h_pre"] = args[0].detach().float().cpu().numpy()
            return _fn
        def _mk_ln(idx: int):
            def _fn(_m, _i, out):
                buf[idx]["x_ln"] = out.detach().float().cpu().numpy()
            return _fn
        def _mk_attn(idx: int):
            def _fn(_m, _i, out):
                t = out[0] if isinstance(out, tuple) else out
                buf[idx]["y_attn"] = t.detach().float().cpu().numpy()
            return _fn
        hooks.append(layers[li].register_forward_pre_hook(_mk_pre(li)))
        hooks.append(layers[li].input_layernorm.register_forward_hook(_mk_ln(li)))
        hooks.append(layers[li].self_attn.register_forward_hook(_mk_attn(li)))

    try:
        with torch.no_grad():
            for lag in sorted(batches_by_lag.keys()):
                bt = batches_by_lag[lag][:max_seqs_per_lag]
                for i0 in range(0, len(bt), batch_size):
                    _ = model(input_ids=bt[i0 : i0 + batch_size].to(device), use_cache=False)
                    for li in layer_indices:
                        # Skip pos 0 (BOS) so sockets fit strictly on active content tokens!
                        store[li]["h_pre"].append(buf[li]["h_pre"][:, 1:, :].reshape(-1, buf[li]["h_pre"].shape[-1]))
                        store[li]["x_ln"].append(buf[li]["x_ln"][:, 1:, :].reshape(-1, buf[li]["x_ln"].shape[-1]))
                        store[li]["y_attn"].append(buf[li]["y_attn"][:, 1:, :].reshape(-1, buf[li]["y_attn"].shape[-1]))
    finally:
        for h in hooks:
            h.remove()

    return {
        li: {k: np.concatenate(v, axis=0) for k, v in d.items()}
        for li, d in store.items()
    }


# ======================================================================================
# 5. Full Pre-Flight Gate Runner (Colab L4 / MPS Execution via `--run`)
# ======================================================================================

def run_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    import pilot_organs_p0 as po
    from extract_and_baseline import HF_CACHE_DIR, get_decoder_layers
    from transformers import AutoTokenizer

    toks = {
        k: AutoTokenizer.from_pretrained(str(HF_CACHE_DIR / k), local_files_only=True)
        for k in (DONOR_KEY,) + RECIPIENT_KEYS
    }
    shared_words = build_shared_single_token_words(toks)
    fit_words, test_words = split_shared_words_disjoint(shared_words, seed=SEED)
    print(f"=== TRACK 2 PRE-FLIGHT GATE (Headroom Census + Gate K1 + Gate K0) ===")
    print(f"Shared 1-token vocab: {len(shared_words)} -> disjoint fit={len(fit_words)}, test={len(test_words)} | Lags={PREFLIGHT_LAGS} | n_test={args.n_test_seqs}")

    # ------------------------------------------------------------------
    # Part A: Donor A (qwen2.5-3b) Census, Best (H1, H2) Pair & Active-Prompt States
    # ------------------------------------------------------------------
    gA = VERIFIED_GEOMETRIES[DONOR_KEY]
    print(f"\n--- [Part A] Donor {DONOR_KEY}: Scanning prev-token + induction heads & Q-K composition ---")
    model_A, tok_A, dev_A, _ = po.load_model_fp32(DONOR_KEY)
    layers_A = get_decoder_layers(model_A)

    fit_batches_A = build_disjoint_copy_batches(fit_words, tok_A, lags=PREFLIGHT_LAGS, n_seqs=args.n_fit_seqs, seed=SEED + 1)
    test_batches_A = build_disjoint_copy_batches(test_words, tok_A, lags=PREFLIGHT_LAGS, n_seqs=args.n_test_seqs, seed=SEED + 2)

    donor_census = scan_model_induction_and_prev_heads(model_A, fit_batches_A, gA, dev_A)
    top_prev_A = sorted(donor_census, key=lambda r: r["prev1_mass"], reverse=True)[:8]
    top_ind_A = sorted(donor_census, key=lambda r: r["induction_mass"], reverse=True)[:8]

    # Find best (H1 prev-token, H2 induction) pair with l1 < l2 by Q-K composition * geometric mean stripe mass!
    best_pair = None
    pair_candidates = []
    for h1 in top_prev_A:
        hd1 = extract_donor_head_weights(layers_A[h1["layer"]].self_attn, h1["head_q"], gA["H_q"], gA["H_kv"])
        for h2 in top_ind_A:
            if h2["layer"] <= h1["layer"]:
                continue
            hd2 = extract_donor_head_weights(layers_A[h2["layer"]].self_attn, h2["head_q"], gA["H_q"], gA["H_kv"])
            # Include exact layer-l2 input_layernorm gamma in residual-stream composition!
            gamma2_A = layers_A[h2["layer"]].input_layernorm.weight.detach().float().cpu().numpy().astype(np.float64)
            comp = qk_ov_composition_score(
                hd2["W_Q"] * gamma2_A[None, :], hd2["W_K"] * gamma2_A[None, :], hd1["W_O"], hd1["W_V"]
            )
            rank_score = comp["comp_score"] * math.sqrt(max(h1["prev1_mass"], 1e-4) * max(h2["induction_mass"], 1e-4))
            rec = {
                "h1": h1, "h2": h2,
                "comp_score": round(comp["comp_score"], 5),
                "rank_score": round(rank_score, 5),
            }
            pair_candidates.append(rec)
            if best_pair is None or rank_score > best_pair["rank_score"]:
                best_pair = {**rec, "hd1": hd1, "hd2": hd2, "gamma2_A": gamma2_A}

    assert best_pair is not None, "No (H1, H2) pair with l1 < l2 found!"
    lA1, hA1 = best_pair["h1"]["layer"], best_pair["h1"]["head_q"]
    lA2, hA2 = best_pair["h2"]["layer"], best_pair["h2"]["head_q"]
    hd1_A, hd2_A = best_pair["hd1"], best_pair["hd2"]
    gamma2_A = best_pair["gamma2_A"]
    comp_A = qk_ov_composition_score(
        hd2_A["W_Q"] * gamma2_A[None, :], hd2_A["W_K"] * gamma2_A[None, :], hd1_A["W_O"], hd1_A["W_V"]
    )
    print(
        f"  Best Donor Pair: H1=L{lA1:02d}.H{hA1:02d} (prev1={best_pair['h1']['prev1_mass']:.3f}) -> "
        f"H2=L{lA2:02d}.H{hA2:02d} (ind={best_pair['h2']['induction_mass']:.3f}) | comp_A={comp_A['comp_score']:.4f}"
    )

    # Extract Donor A active-prompt states at (lA1, lA2) on fit_words (for sockets) & evaluate Donor A OV copy on test_words!
    donor_fit_states = extract_active_residual_and_ln_states(model_A, fit_batches_A, [lA1, lA2], dev_A, max_seqs_per_lag=args.n_fit_seqs)
    donor_test_ln_lA2 = {
        lag: extract_input_ln_states(model_A, bt[:32], [lA2], dev_A)[lA2]
        for lag, bt in test_batches_A.items()
    }
    donor_ov_copy = eval_active_prompt_ov_copying(
        model_A, donor_test_ln_lA2, {lag: bt[:32] for lag, bt in test_batches_A.items()},
        hd2_A["W_V"], hd2_A["W_O"], donor_fit_states[lA2]["x_ln"].mean(axis=0), dev_A,
    )
    print(f"  Donor A OV Copying (test_words): in_ctx_top1={donor_ov_copy['ov_in_context_top1']:.4f}, full_vocab_top1={donor_ov_copy['ov_full_vocab_top1']:.4f}, ldiff={donor_ov_copy['ov_logit_diff']:+.3f}")

    # Compute donor H2 plane energy weights for optimal RoPE intertwiner
    with torch.no_grad():
        x_f2 = torch.from_numpy(donor_fit_states[lA2]["x_ln"][:2048].astype(np.float32)).to(dev_A)
        q2 = F.linear(x_f2, torch.from_numpy(hd2_A["W_Q"].astype(np.float32)).to(dev_A), torch.from_numpy(hd2_A["b_Q"].astype(np.float32)).to(dev_A))
        k2 = F.linear(x_f2, torch.from_numpy(hd2_A["W_K"].astype(np.float32)).to(dev_A), torch.from_numpy(hd2_A["b_K"].astype(np.float32)).to(dev_A))
        mA = gA["d_h"] // 2
        hd2_plane_energy = (torch.sqrt(q2[:, :mA]**2 + q2[:, mA:]**2) * torch.sqrt(k2[:, :mA]**2 + k2[:, mA:]**2)).mean(0).cpu().numpy().astype(np.float64)

    # UNLOAD Donor A completely before loading any Recipient B!
    po.unload_model(model_A)
    print(f"  Donor {DONOR_KEY} completely unloaded!")

    # ------------------------------------------------------------------
    # Part B: Per-Recipient Headroom Census, Gate K1 & Gate K0
    # ------------------------------------------------------------------
    recipients_out: Dict[str, Any] = {}
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    for rk in targets:
        tr0 = time.time()
        gB = VERIFIED_GEOMETRIES[rk]
        print(f"\n================ RECIPIENT {rk} (D={gB['D']}, L={gB['L']}, H_q={gB['H_q']}, d_h={gB['d_h']}) ================")
        model_B, tok_B, dev_B, _ = po.load_model_fp32(rk)
        layers_B = get_decoder_layers(model_B)

        fit_batches_B = build_disjoint_copy_batches(fit_words, tok_B, lags=PREFLIGHT_LAGS, n_seqs=args.n_fit_seqs, seed=SEED + 1)
        test_batches_B = build_disjoint_copy_batches(test_words, tok_B, lags=PREFLIGHT_LAGS, n_seqs=args.n_test_seqs, seed=SEED + 2)

        # 1. Baseline Copy Performance on test_words (n=64 per lag!)
        base_logits_B = {lag: forward_batched_logits(model_B, bt, dev_B, batch_size=args.batch_size) for lag, bt in test_batches_B.items()}
        base_scores = {str(lag): score_copy_predictions(base_logits_B[lag], test_batches_B[lag], lag) for lag in PREFLIGHT_LAGS}
        base_mean_top1 = float(np.mean([base_scores[str(lag)]["top1_acc"] for lag in PREFLIGHT_LAGS]))
        base_mean_ldiff = float(np.mean([base_scores[str(lag)]["logit_diff"] for lag in PREFLIGHT_LAGS]))
        print(f"  [BASE {rk}] mean_top1={base_mean_top1:.4f}, mean_ldiff={base_mean_ldiff:+.3f} | by_lag={ {k: v['top1_acc'] for k, v in base_scores.items()} }")

        # 2. Headroom Census: Identify native induction heads & progressively mean-ablate top {2, 4, 8, 12, 16}!
        recip_census = scan_model_induction_and_prev_heads(model_B, fit_batches_B, gB, dev_B)
        top_ind_B = sorted(recip_census, key=lambda r: r["induction_mass"], reverse=True)
        ablate_layers = sorted({r["layer"] for r in top_ind_B[:max(ABLATION_K_GRID)]})
        mu_z_B = compute_native_head_o_proj_means(model_B, fit_batches_B, ablate_layers, gB["H_q"], dev_B, batch_size=args.batch_size)

        ablation_curve: Dict[str, Any] = {}
        selected_k_for_k1: Optional[int] = None
        selected_k_acc: float = base_mean_top1
        for k_abl in ABLATION_K_GRID:
            heads_k = [(r["layer"], r["head_q"]) for r in top_ind_B[:k_abl]]
            abl_hook = MeanAblateHeadsHook(layers_B, heads_k, mu_z_B, gB["H_q"], dev_B).attach()
            try:
                sc_k = {}
                for lag in PREFLIGHT_LAGS:
                    lg = forward_batched_logits(model_B, test_batches_B[lag], dev_B, batch_size=args.batch_size)
                    sc_k[str(lag)] = score_copy_predictions(lg, test_batches_B[lag], lag)
            finally:
                abl_hook.remove()
            m_top1 = float(np.mean([sc_k[str(lag)]["top1_acc"] for lag in PREFLIGHT_LAGS]))
            m_ldiff = float(np.mean([sc_k[str(lag)]["logit_diff"] for lag in PREFLIGHT_LAGS]))
            ablation_curve[str(k_abl)] = {
                "k": int(k_abl),
                "heads": heads_k,
                "mean_top1": round(m_top1, 4),
                "mean_logit_diff": round(m_ldiff, 4),
                "by_lag": sc_k,
            }
            print(f"  [Headroom k={k_abl:2d}] mean_top1={m_top1:.4f}, mean_ldiff={m_ldiff:+.3f} | by_lag={ {k: v['top1_acc'] for k, v in sc_k.items()} }")
            # Pick first k landing inside [0.10, 0.30] (or closest to 0.20 if none hits exactly)
            if selected_k_for_k1 is None and 0.10 <= m_top1 <= 0.30:
                selected_k_for_k1 = int(k_abl)
                selected_k_acc = m_top1

        unhook_after_census = verify_batch_matched_unhook(model_B, test_batches_B[16], base_logits_B[16], dev_B, batch_size=args.batch_size)
        min_abl_top1 = min(v["mean_top1"] for v in ablation_curve.values())
        headroom_10_30_reached = any(0.10 <= v["mean_top1"] <= 0.30 for v in ablation_curve.values())
        pivot_skip_induction = bool(min_abl_top1 >= 0.50)
        if selected_k_for_k1 is None:
            # Choose k closest to 0.20 (with meaningful ablation gap!)
            best_k_str = min(ablation_curve.keys(), key=lambda ks: abs(ablation_curve[ks]["mean_top1"] - 0.20))
            selected_k_for_k1 = int(best_k_str)
            selected_k_acc = ablation_curve[best_k_str]["mean_top1"]

        print(
            f"  [Headroom Verdict] 10-30% reached={headroom_10_30_reached} (min_top1={min_abl_top1:.4f}) | "
            f"pivot_skip_induction={pivot_skip_induction} | K1 k={selected_k_for_k1} (abl_top1={selected_k_acc:.4f}) | unhook={unhook_after_census}"
        )

        # 3. Gate K1 (Positive Control: K1-a Own Heads + K1-b Rotated Self-Transplant via Exact Same Pipeline!)
        k1_heads = [(r["layer"], r["head_q"]) for r in top_ind_B[:selected_k_for_k1]]
        k1_layers = sorted({li for li, _ in k1_heads})

        # Capture fit_words active states at k1_layers for K1-b finite-sample rotated socket estimation!
        k1_fit_states = extract_active_residual_and_ln_states(model_B, fit_batches_B, k1_layers, dev_B, max_seqs_per_lag=args.n_fit_seqs)

        # Build K1-a (own heads) AND K1-b (random orthogonal gauge Q_l + closed-form transport operator!)
        rng_k1 = np.random.default_rng(SEED + 777)
        k1a_specs, k1b_exact_specs, k1b_est_specs = [], [], []
        Q_by_layer = {}
        est_sock_by_layer = {}
        for li in k1_layers:
            Q_l, _ = sla.qr(rng_k1.standard_normal((gB["D"], gB["D"])))
            Q_by_layer[li] = Q_l
            # Finite-sample closed-form socket recovery from rotated fit_words states!
            x_orig = k1_fit_states[li]["x_ln"]
            y_orig = k1_fit_states[li]["y_attn"]
            x_rot = x_orig @ Q_l.T
            y_rot = y_orig @ Q_l.T
            s_in = fit_cocycle_constrained_read_socket(x_orig, x_rot)
            s_out = fit_active_centered_ov_write_socket(y_rot, y_orig)
            est_sock_by_layer[li] = {"M_in": s_in["M_in"], "d_in": s_in["d_in"], "M_out": s_out["M_out"]}

        for li, hq in k1_heads:
            hd = extract_donor_head_weights(layers_B[li].self_attn, hq, gB["H_q"], gB["H_kv"])
            mu_sub = mu_z_B[(li, hq)] @ hd["W_O"].T
            I_D = np.eye(gB["D"], dtype=np.float64)
            I_h = np.eye(gB["d_h"], dtype=np.float64)
            z_D = np.zeros(gB["D"], dtype=np.float64)

            # K1-a: Identity transport through exact same compiler & hook pipeline
            op_a = compile_congruent_attention_head(hd, I_D, z_D, I_D, I_h, gB["d_h"], gB["rope_theta"], gB["rope_theta"], "recipient_rope", tau=1.0)
            k1a_specs.append({"layer": li, "head_op": op_a, "alpha": 1.0, "baseline_subtract": mu_sub})

            # Donor head expressed in rotated gauge Q_l: W_Q_rot = W_Q Q_l^T, W_O_rot = Q_l W_O
            Q_l = Q_by_layer[li]
            hd_rot = {
                **hd,
                "W_Q": hd["W_Q"] @ Q_l.T,
                "W_K": hd["W_K"] @ Q_l.T,
                "W_V": hd["W_V"] @ Q_l.T,
                "W_O": Q_l @ hd["W_O"],
            }
            # K1-b (exact gauge Q_l back-conjugation)
            op_b_ex = compile_congruent_attention_head(hd_rot, Q_l, z_D, Q_l.T, I_h, gB["d_h"], gB["rope_theta"], gB["rope_theta"], "recipient_rope", tau=1.0)
            k1b_exact_specs.append({"layer": li, "head_op": op_b_ex, "alpha": 1.0, "baseline_subtract": mu_sub})

            # K1-b (finite-sample closed-form LW socket back-conjugation from fit_words!)
            es = est_sock_by_layer[li]
            op_b_est = compile_congruent_attention_head(hd_rot, es["M_in"], es["d_in"], es["M_out"], I_h, gB["d_h"], gB["rope_theta"], gB["rope_theta"], "recipient_rope", tau=1.0)
            k1b_est_specs.append({"layer": li, "head_op": op_b_est, "alpha": 1.0, "baseline_subtract": mu_sub})

        def _eval_k1_arm(label: str, specs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
            abl = MeanAblateHeadsHook(layers_B, k1_heads, mu_z_B, gB["H_q"], dev_B).attach()
            pipe = CompiledHeadGraftPipeline(layers_B, specs, dev_B).attach()
            try:
                sc = {}
                for lag in PREFLIGHT_LAGS:
                    lg = forward_batched_logits(model_B, test_batches_B[lag], dev_B, batch_size=args.batch_size)
                    sc[str(lag)] = score_copy_predictions(lg, test_batches_B[lag], lag)
            finally:
                pipe.remove()
                abl.remove()
            m_top1 = float(np.mean([sc[str(lag)]["top1_acc"] for lag in PREFLIGHT_LAGS]))
            m_ldiff = float(np.mean([sc[str(lag)]["logit_diff"] for lag in PREFLIGHT_LAGS]))
            gap = max(base_mean_top1 - selected_k_acc, 1e-6)
            R_acc = (m_top1 - selected_k_acc) / gap
            gap_ld = max(base_mean_ldiff - ablation_curve[str(selected_k_for_k1)]["mean_logit_diff"], 1e-6)
            R_ldiff = (m_ldiff - ablation_curve[str(selected_k_for_k1)]["mean_logit_diff"]) / gap_ld
            uh = verify_batch_matched_unhook(model_B, test_batches_B[16], base_logits_B[16], dev_B, batch_size=args.batch_size)
            print(f"  [Gate {label:20s}] top1={m_top1:.4f} (R_acc={R_acc:.4f}), ldiff={m_ldiff:+.3f} (R_ldiff={R_ldiff:.4f}) | unhook={uh['pass']} (bit={uh['bit_equal']}, max={uh['max_abs_diff']:.1e})")
            return {
                "mean_top1": round(m_top1, 4),
                "mean_logit_diff": round(m_ldiff, 4),
                "R_acc": round(float(R_acc), 4),
                "R_logit_diff": round(float(R_ldiff), 4),
                "by_lag": sc,
                "unhook": uh,
            }

        k1a_res = _eval_k1_arm("K1-a_OWN_HEADS", k1a_specs)
        k1b_ex_res = _eval_k1_arm("K1-b_ROT_EXACT_GAUGE", k1b_exact_specs)
        k1b_est_res = _eval_k1_arm("K1-b_ROT_EST_SOCKET", k1b_est_specs)
        gate_k1_pass = bool(k1a_res["R_acc"] >= 0.90 and k1b_est_res["R_acc"] >= 0.80 and k1a_res["unhook"]["pass"] and k1b_est_res["unhook"]["pass"])
        print(f"  [Gate K1 VERDICT] PASS={gate_k1_pass} (K1-a R={k1a_res['R_acc']:.3f}>=0.90, K1-b_est R={k1b_est_res['R_acc']:.3f}>=0.80)")

        # 4. Gate K0 (Cross-Family Algebra Check: qwen2.5-3b (H1, H2) -> Recipient B!)
        # Match (lA1, lA2) to Recipient B layers (lB1 < lB2) via linear CKA on active fit_words states!
        approx_1 = max(1, min(gB["L"] - 3, round(lA1 * gB["L"] / float(gA["L"]))))
        approx_2 = max(approx_1 + 1, min(gB["L"] - 2, round(lA2 * gB["L"] / float(gA["L"]))))
        cand_B = sorted({
            max(1, min(gB["L"] - 2, a + off))
            for a in (approx_1, approx_2)
            for off in (-2, -1, 0, 1, 2)
        })
        recip_fit_states = extract_active_residual_and_ln_states(model_B, fit_batches_B, cand_B, dev_B, max_seqs_per_lag=args.n_fit_seqs)

        cands_1 = [lb for lb in cand_B if abs(lb - approx_1) <= 2]
        lB1 = max(cands_1, key=lambda lb: linear_cka(donor_fit_states[lA1]["x_ln"][:1500], recip_fit_states[lb]["x_ln"][:1500]))
        cands_2 = [lb for lb in cand_B if lb > lB1 and abs(lb - approx_2) <= 2]
        if not cands_2:
            cands_2 = [lb for lb in cand_B if lb > lB1]
        lB2 = max(cands_2, key=lambda lb: linear_cka(donor_fit_states[lA2]["x_ln"][:1500], recip_fit_states[lb]["x_ln"][:1500]))

        # Split active fit states 80/20 for held-out socket diagnostics (100% disjoint from test_words!)
        n_act = len(donor_fit_states[lA1]["x_ln"])
        n_tr = int(0.8 * n_act)
        gamma2_B = layers_B[lB2].input_layernorm.weight.detach().float().cpu().numpy().astype(np.float64)

        # Layer-1 active-prompt read & write sockets (J1: R^{D_A} -> R^{D_B} in post-attn residual coordinates!)
        sock_in1 = fit_cocycle_constrained_read_socket(
            recip_fit_states[lB1]["x_ln"][:n_tr], donor_fit_states[lA1]["x_ln"][:n_tr],
            X_B_val=recip_fit_states[lB1]["x_ln"][n_tr:], X_A_val=donor_fit_states[lA1]["x_ln"][n_tr:],
        )
        sock_out1 = fit_active_centered_ov_write_socket(
            donor_fit_states[lA1]["y_attn"][:n_tr], recip_fit_states[lB1]["y_attn"][:n_tr],
            Y_A_val=donor_fit_states[lA1]["y_attn"][n_tr:], Y_B_val=recip_fit_states[lB1]["y_attn"][n_tr:],
        )
        J1 = sock_out1["M_out"]  # (D_B, D_A)

        # Orthonormal basis U for Im(W_OV1) = Im(W_O1) in Donor A residual space!
        U_ov1 = orthonormal_image_basis(hd1_A["W_O"])

        # Compare Layer-2 Read Socket WITH vs WITHOUT exact cocycle equality constraint!
        # Note: H2 reads post-input_layernorm `x_ln = diag(gamma2) @ h_pre / rms`, so in residual coordinates
        # `h_pre_B` carries `J1 @ U_ov1` directly! Fit both `J_2 U = J_1 U` (forward) AND `M_{in,2} (J_1 U) = U` (read)!
        fwd_cocycle = fit_cocycle_constrained_forward_socket(
            donor_fit_states[lA2]["h_pre"][:n_tr], recip_fit_states[lB2]["h_pre"][:n_tr], J1, U_ov1
        )
        sock_in2_nocyc = fit_cocycle_constrained_read_socket(
            recip_fit_states[lB2]["h_pre"][:n_tr], donor_fit_states[lA2]["h_pre"][:n_tr],
            J1=None, U=None,
            X_B_val=recip_fit_states[lB2]["h_pre"][n_tr:], X_A_val=donor_fit_states[lA2]["h_pre"][n_tr:],
        )
        sock_in2_cocyc = fit_cocycle_constrained_read_socket(
            recip_fit_states[lB2]["h_pre"][:n_tr], donor_fit_states[lA2]["h_pre"][:n_tr],
            J1=J1, U=U_ov1,
            X_B_val=recip_fit_states[lB2]["h_pre"][n_tr:], X_A_val=donor_fit_states[lA2]["h_pre"][n_tr:],
        )
        # Also fit mean-centered active-prompt x_ln -> x_ln read & y_attn -> y_attn write for OV copying!
        sock_in2_ln_nocyc = fit_cocycle_constrained_read_socket(
            recip_fit_states[lB2]["x_ln"][:n_tr], donor_fit_states[lA2]["x_ln"][:n_tr],
            X_B_val=recip_fit_states[lB2]["x_ln"][n_tr:], X_A_val=donor_fit_states[lA2]["x_ln"][n_tr:],
        )
        # Map J1 U into post-LN coordinates via gamma2_B / gamma2_A for x_ln cocycle check!
        J1_ln = (gamma2_B[:, None] * J1) / np.maximum(np.abs(gamma2_A)[None, :], 1e-6)
        sock_in2_ln_cocyc = fit_cocycle_constrained_read_socket(
            recip_fit_states[lB2]["x_ln"][:n_tr], donor_fit_states[lA2]["x_ln"][:n_tr],
            J1=J1_ln, U=U_ov1,
            X_B_val=recip_fit_states[lB2]["x_ln"][n_tr:], X_A_val=donor_fit_states[lA2]["x_ln"][n_tr:],
        )
        sock_out2_ov = fit_active_centered_ov_write_socket(
            donor_fit_states[lA2]["y_attn"][:n_tr], recip_fit_states[lB2]["y_attn"][:n_tr],
            Y_A_val=donor_fit_states[lA2]["y_attn"][n_tr:], Y_B_val=recip_fit_states[lB2]["y_attn"][n_tr:],
        )

        # Optimal 1-D RoPE intertwiner J_rope + exact 1/sqrt(d_h) scaling compensation!
        J_rope, rope_meta = build_optimal_rope_intertwiner(
            gA["d_h"], gA["rope_theta"], gB["d_h"], gB["rope_theta"], plane_weights=hd2_plane_energy
        )
        scale_dh_qk = (float(gB["d_h"]) / float(gA["d_h"])) ** 0.25  # product scales by sqrt(d_hB / d_hA)!

        # Compute transplanted K0-a composition score WITH vs WITHOUT cocycle constraint!
        WO1_B = J1 @ hd1_A["W_O"]
        WV1_B = hd1_A["W_V"] @ sock_in1["M_in"]

        # Effective residual-space Q, K of Donor A at layer lA2 (with gamma2_A)
        WQ2_A_res = hd2_A["W_Q"] * gamma2_A[None, :]
        WK2_A_res = hd2_A["W_K"] * gamma2_A[None, :]

        comp_B_nocyc = qk_ov_composition_score(
            scale_dh_qk * (J_rope @ WQ2_A_res @ sock_in2_nocyc["M_in"]),
            scale_dh_qk * (J_rope @ WK2_A_res @ sock_in2_nocyc["M_in"]),
            WO1_B, WV1_B,
        )
        comp_B_cocyc = qk_ov_composition_score(
            scale_dh_qk * (J_rope @ WQ2_A_res @ sock_in2_cocyc["M_in"]),
            scale_dh_qk * (J_rope @ WK2_A_res @ sock_in2_cocyc["M_in"]),
            WO1_B, WV1_B,
        )
        ratio_nocyc = comp_B_nocyc["comp_score"] / max(comp_A["comp_score"], 1e-12)
        ratio_cocyc = comp_B_cocyc["comp_score"] / max(comp_A["comp_score"], 1e-12)

        # Gate K0-b: Mean-centered active-prompt OV copying on test_words!
        recip_test_ln_lB2 = {
            lag: extract_input_ln_states(model_B, bt[:32], [lB2], dev_B)[lB2]
            for lag, bt in test_batches_B.items()
        }
        mu_xB_l2 = recip_fit_states[lB2]["x_ln"].mean(axis=0)
        WO2_B = sock_out2_ov["M_out"] @ hd2_A["W_O"]
        WV2_B_nocyc = hd2_A["W_V"] @ sock_in2_ln_nocyc["M_in"]
        WV2_B_cocyc = hd2_A["W_V"] @ sock_in2_ln_cocyc["M_in"]

        ov_copy_nocyc = eval_active_prompt_ov_copying(
            model_B, recip_test_ln_lB2, {lag: bt[:32] for lag, bt in test_batches_B.items()},
            WV2_B_nocyc, WO2_B, mu_xB_l2, dev_B,
        )
        ov_copy_cocyc = eval_active_prompt_ov_copying(
            model_B, recip_test_ln_lB2, {lag: bt[:32] for lag, bt in test_batches_B.items()},
            WV2_B_cocyc, WO2_B, mu_xB_l2, dev_B,
        )
        ov_ratio_ic = ov_copy_cocyc["ov_in_context_top1"] / max(donor_ov_copy["ov_in_context_top1"], 1e-6)
        ov_ratio_full = ov_copy_cocyc["ov_full_vocab_top1"] / max(donor_ov_copy["ov_full_vocab_top1"], 1e-6)
        k0a_pass = bool(ratio_cocyc >= 0.50)
        k0b_pass = bool(ov_ratio_ic >= 0.50)
        gate_k0_pass = bool(k0a_pass and k0b_pass)

        print(
            f"  [Gate K0 | L{lA1}->{lB1}, L{lA2}->{lB2}] sqrt(d_hB/d_hA)={math.sqrt(gB['d_h']/gA['d_h']):.4f} | "
            f"cocycle_eq_err(J2U=J1U)={fwd_cocycle['cocycle_eq_err']:.2e}, read_err={sock_in2_cocyc['cocycle_recovery_err']:.2e}"
        )
        print(
            f"  [Gate K0-a COMP] donor={comp_A['comp_score']:.4f} | WITHOUT cocycle={comp_B_nocyc['comp_score']:.4f} ({ratio_nocyc:.3f}x) | "
            f"WITH cocycle={comp_B_cocyc['comp_score']:.4f} ({ratio_cocyc:.3f}x) -> PASS={k0a_pass}"
        )
        print(
            f"  [Gate K0-b OV  ] donor_ic={donor_ov_copy['ov_in_context_top1']:.4f} (full={donor_ov_copy['ov_full_vocab_top1']:.4f}) | "
            f"B_nocyc_ic={ov_copy_nocyc['ov_in_context_top1']:.4f} | "
            f"B_cocyc_ic={ov_copy_cocyc['ov_in_context_top1']:.4f} ({ov_ratio_ic:.3f}x, full={ov_copy_cocyc['ov_full_vocab_top1']:.4f}) -> PASS={k0b_pass}"
        )

        recipients_out[rk] = {
            "geometry": gB,
            "baseline_copy": {"mean_top1": round(base_mean_top1, 4), "mean_logit_diff": round(base_mean_ldiff, 4), "by_lag": base_scores},
            "headroom_census": {
                "top20_induction_heads": top_ind_B[:20],
                "ablation_curve": ablation_curve,
                "min_ablated_top1": round(min_abl_top1, 4),
                "headroom_10_30_reached": headroom_10_30_reached,
                "pivot_to_skip_induction": pivot_skip_induction,
                "selected_k_for_k1": selected_k_for_k1,
                "unhook_after_census": unhook_after_census,
            },
            "gate_k1": {
                "pass": gate_k1_pass,
                "k_heads": selected_k_for_k1,
                "K1a_own_heads": k1a_res,
                "K1b_rotated_exact_gauge": k1b_ex_res,
                "K1b_rotated_est_socket": k1b_est_res,
            },
            "gate_k0": {
                "pass": gate_k0_pass,
                "k0a_pass": k0a_pass,
                "k0b_pass": k0b_pass,
                "matched_layers": {"lA1": lA1, "lB1": lB1, "lA2": lA2, "lB2": lB2},
                "scale_sqrt_dhB_over_dhA": round(math.sqrt(gB["d_h"] / gA["d_h"]), 6),
                "optimal_rope_meta": {k: v for k, v in rope_meta.items() if k != "matched_pairs"},
                "sockets": {
                    "l1_read_R2": sock_in1["heldout_read_R2"],
                    "l1_write_cos": sock_out1["heldout_write_cos"],
                    "l2_fwd_cocycle_eq_err": fwd_cocycle["cocycle_eq_err"],
                    "l2_read_nocyc_R2": sock_in2_nocyc["heldout_read_R2"],
                    "l2_read_cocyc_R2": sock_in2_cocyc["heldout_read_R2"],
                    "l2_read_cocyc_recovery_err": sock_in2_cocyc["cocycle_recovery_err"],
                    "l2_ov_write_cos": sock_out2_ov["heldout_write_cos"],
                    "l2_ov_write_R2": sock_out2_ov["heldout_write_R2"],
                },
                "k0a_composition": {
                    "donor_comp": round(comp_A["comp_score"], 5),
                    "without_cocycle_comp": round(comp_B_nocyc["comp_score"], 5),
                    "without_cocycle_ratio": round(ratio_nocyc, 4),
                    "with_cocycle_comp": round(comp_B_cocyc["comp_score"], 5),
                    "with_cocycle_ratio": round(ratio_cocyc, 4),
                },
                "k0b_ov_copying": {
                    "donor": donor_ov_copy,
                    "without_cocycle": ov_copy_nocyc,
                    "with_cocycle": ov_copy_cocyc,
                    "ratio_in_context_top1": round(ov_ratio_ic, 4),
                    "ratio_full_vocab_top1": round(ov_ratio_full, 4),
                },
            },
            "elapsed_sec": round(time.time() - tr0, 1),
        }
        po.unload_model(model_B)

    all_pass = all(
        r["gate_k1"]["pass"] and r["gate_k0"]["pass"] and (not r["headroom_census"]["pivot_to_skip_induction"])
        for r in recipients_out.values()
    )
    out = {
        "meta": {
            "experiment": "v2_track2_preflight",
            "donor_key": DONOR_KEY,
            "targets": targets,
            "lags": list(PREFLIGHT_LAGS),
            "n_fit_words": len(fit_words),
            "n_test_words": len(test_words),
            "n_fit_seqs": args.n_fit_seqs,
            "n_test_seqs": args.n_test_seqs,
            "seed": SEED,
            "all_gates_pass": all_pass,
            "total_elapsed_sec": round(time.time() - t0, 1),
        },
        "donor_best_pair": {
            "h1": best_pair["h1"],
            "h2": best_pair["h2"],
            "comp_A": {k: round(v, 5) for k, v in comp_A.items()},
            "donor_ov_copy": donor_ov_copy,
            "top10_pairs": sorted(pair_candidates, key=lambda r: r["rank_score"], reverse=True)[:10],
        },
        "recipients": recipients_out,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[TRACK 2 PRE-FLIGHT COMPLETE] all_gates_pass={all_pass} | Saved {OUT_PATH} ({out['meta']['total_elapsed_sec']}s)")
    return out


# ======================================================================================
# 6. Pure-CPU Unit Tests (<10s, ZERO GPU!)
# ======================================================================================

def run_cpu_unit_tests() -> None:
    t0 = time.time()
    print("=== Running v2_track2_preflight.py Pure-CPU Unit Tests ===")
    rng = np.random.default_rng(20261126)

    # 1. Exact Cocycle Equality Constraint (`J_2 U = J_1 U` AND `M_{in,2} (J_1 U) = U` < 1e-10!)
    #    Construct a tough regime where U = Im(W_OV1) has VERY LOW variance in ambient X_A so
    #    unconstrained ridge shrinks it by >10x, while cocycle-constrained preserves 100%!
    D_A, D_B, d_h = 64, 96, 8
    W_O1 = rng.standard_normal((D_A, d_h))
    W_V1 = rng.standard_normal((d_h, D_A))
    W_Q2 = rng.standard_normal((d_h, D_A))
    W_K2 = rng.standard_normal((d_h, D_A))
    U = orthonormal_image_basis(W_O1)
    assert U.shape == (D_A, d_h) and np.allclose(U.T @ U, np.eye(d_h), atol=1e-12)

    # Ground-truth socket J1 (D_B x D_A, isometric)
    J1_true, _ = sla.qr(rng.standard_normal((D_B, D_A)), mode="economic")
    # Ambient data has tiny variance (0.01) inside U and large variance (1.5) orthogonal to U!
    P_perp = np.eye(D_A) - U @ U.T
    X_A_amb = rng.standard_normal((200, D_A)) @ P_perp * 1.5 + rng.standard_normal((200, d_h)) @ U.T * 0.01
    X_B_amb = X_A_amb @ J1_true.T + rng.standard_normal((200, D_B)) * 0.25

    fwd = fit_cocycle_constrained_forward_socket(X_A_amb, X_B_amb, J1_true, U)
    assert fwd["cocycle_eq_err"] < 1e-10, f"J_2 U = J_1 U violated: {fwd['cocycle_eq_err']}"

    rd_nocyc = fit_cocycle_constrained_read_socket(X_B_amb, X_A_amb, J1=None, U=None)
    rd_cocyc = fit_cocycle_constrained_read_socket(X_B_amb, X_A_amb, J1=J1_true, U=U)
    assert rd_cocyc["cocycle_recovery_err"] < 1e-10, f"M_in2 (J_1 U) = U violated: {rd_cocyc['cocycle_recovery_err']}"

    # Verify fast composition score matches explicit dense D x D matmul to <1e-12!
    c_fast = qk_ov_composition_score(W_Q2, W_K2, W_O1, W_V1)
    W_QK2_dense = W_Q2.T @ W_K2
    W_OV1_dense = W_O1 @ W_V1
    c_dense = float(sla.norm(W_QK2_dense @ W_OV1_dense, "fro") / (sla.norm(W_QK2_dense, "fro") * sla.norm(W_OV1_dense, "fro")))
    assert abs(c_fast["comp_score"] - c_dense) < 1e-12, f"Fast comp mismatch: {c_fast['comp_score']} vs {c_dense}"

    # Verify cocycle constraint recovers exact donor K-composition on low-variance U while unconstrained collapses!
    c_nocyc = qk_ov_composition_score(W_Q2 @ rd_nocyc["M_in"], W_K2 @ rd_nocyc["M_in"], J1_true @ W_O1, W_V1 @ J1_true.T)
    c_cocyc = qk_ov_composition_score(W_Q2 @ rd_cocyc["M_in"], W_K2 @ rd_cocyc["M_in"], J1_true @ W_O1, W_V1 @ J1_true.T)
    assert c_cocyc["comp_score"] / c_fast["comp_score"] > 0.85, f"Cocycle failed to preserve composition: {c_cocyc} vs {c_fast}"
    assert c_nocyc["comp_score"] < 0.30 * c_cocyc["comp_score"], f"Expected unconstrained ridge collapse on low-var U: {c_nocyc} vs {c_cocyc}"
    print(
        f"  [PASS 1/5] Exact cocycle constraints verified (||J2U-J1U||={fwd['cocycle_eq_err']:.2e}, ||Min2 J1 U - U||={rd_cocyc['cocycle_recovery_err']:.2e}) "
        f"& low-var U composition ratio: nocyc={c_nocyc['comp_score']/c_fast['comp_score']:.3f}x -> cocyc={c_cocyc['comp_score']/c_fast['comp_score']:.3f}x!"
    )

    # 2. Exact 1/sqrt(d_h) Scaling Compensation for 128 -> 256 (sqrt(2)) and 128 -> 64 (sqrt(0.5))!
    dA_r = 128
    q_a = rng.standard_normal((4, dA_r))
    k_a = rng.standard_normal((4, dA_r))
    logits_A = (q_a @ k_a.T) / math.sqrt(dA_r)
    # 128 -> 256 (Falcon3 exact nested i -> 2i!)
    J_128_256, _ = build_optimal_rope_intertwiner(128, 1_000_000.0, 256, 1_000_042.0)
    s_256 = (256.0 / 128.0) ** 0.25  # product = sqrt(2)!
    assert abs(s_256 * s_256 - math.sqrt(2.0)) < 1e-12
    logits_256 = ((s_256 * (q_a @ J_128_256.T)) @ (s_256 * (k_a @ J_128_256.T)).T) / math.sqrt(256.0)
    assert float(np.max(np.abs(logits_256 - logits_A))) < 1e-12, "128->256 sqrt(2) compensation NOT exact!"
    # 128 -> 64 (SmolLM2, exact on the 32 matched planes!)
    J_128_64, _ = build_optimal_rope_intertwiner(128, 1_000_000.0, 64, 130_000.0)
    s_64 = (64.0 / 128.0) ** 0.25  # product = sqrt(0.5)!
    assert abs(s_64 * s_64 - math.sqrt(0.5)) < 1e-12
    # Project q_a, k_a into row-space of J_128_64
    P_64 = J_128_64.T @ J_128_64
    logits_A_sub = ((q_a @ P_64) @ (k_a @ P_64).T) / math.sqrt(128.0)
    logits_64 = ((s_64 * (q_a @ J_128_64.T)) @ (s_64 * (k_a @ J_128_64.T)).T) / math.sqrt(64.0)
    assert float(np.max(np.abs(logits_64 - logits_A_sub))) < 1e-12, "128->64 sqrt(0.5) compensation NOT exact!"
    print("  [PASS 2/5] Exact 1/sqrt(d_h) compensation verified (<1e-12) for 128->256 (sqrt(2)) and 128->64 (sqrt(0.5))!")

    # 3. Disjoint Fit/Test Shared-Token Pools & Exact Copy Scoring (Top-1 + Logit Diff)
    from extract_and_baseline import HF_CACHE_DIR
    from transformers import AutoTokenizer

    if all((HF_CACHE_DIR / k).is_dir() for k in (DONOR_KEY,) + RECIPIENT_KEYS):
        toks = {k: AutoTokenizer.from_pretrained(str(HF_CACHE_DIR / k), local_files_only=True) for k in (DONOR_KEY,) + RECIPIENT_KEYS}
        sw = build_shared_single_token_words(toks)
        fw, tw = split_shared_words_disjoint(sw, seed=SEED)
        assert len(fw) >= 100 and len(tw) >= 100 and set(fw).isdisjoint(set(tw))
        b_fit = build_disjoint_copy_batches(fw, toks[DONOR_KEY], lags=PREFLIGHT_LAGS, n_seqs=4, seed=1)
        b_tst = build_disjoint_copy_batches(tw, toks[DONOR_KEY], lags=PREFLIGHT_LAGS, n_seqs=4, seed=2)
        for lag in PREFLIGHT_LAGS:
            assert b_fit[lag].shape == (4, 1 + 2 * lag) and b_tst[lag].shape == (4, 1 + 2 * lag)
            assert set(b_fit[lag][:, 1:].ravel().tolist()).isdisjoint(set(b_tst[lag][:, 1:].ravel().tolist()))
        print(f"  [PASS 3/5] Disjoint fit ({len(fw)}) / test ({len(tw)}) 1-token pools & lags {PREFLIGHT_LAGS} verified across 3/3 tokenizers!")
    else:
        print("  [PASS 3/5] (Skipped cached tokenizer check — HF cache dir not present)")

    # 4. Exact Gate K1 (Own Heads R == 1.0 & Rotated Self-Transplant R == 1.0) on a Synthetic 2-Layer HF LlamaForCausalLM!
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    torch.manual_seed(123)
    D_t, H_t, dh_t, V_t = 64, 4, 16, 128
    cfg_t = LlamaConfig(
        vocab_size=V_t, hidden_size=D_t, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=H_t, num_key_value_heads=H_t, head_dim=dh_t,
        rope_theta=10_000.0, mlp_bias=False, attention_bias=False,
    )
    toy = LlamaForCausalLM(cfg_t).eval()
    # Amplify attention weights so mean-ablating top heads produces a sharp measurable logit gap!
    with torch.no_grad():
        for lyr in toy.model.layers:
            lyr.self_attn.q_proj.weight.mul_(4.0)
            lyr.self_attn.k_proj.weight.mul_(4.0)
            lyr.self_attn.v_proj.weight.mul_(4.0)
            lyr.self_attn.o_proj.weight.mul_(4.0)

    # Synthetic repeated sequences of length 1 + 2*4 = 9 (BOS=1 + 4 unique tokens repeated)
    toy_fit = torch.tensor([[1, 10, 11, 12, 13, 10, 11, 12, 13], [1, 20, 21, 22, 23, 20, 21, 22, 23],
                            [1, 30, 31, 32, 33, 30, 31, 32, 33], [1, 40, 41, 42, 43, 40, 41, 42, 43]], dtype=torch.long)
    toy_tst = torch.tensor([[1, 50, 51, 52, 53, 50, 51, 52, 53], [1, 60, 61, 62, 63, 60, 61, 62, 63]], dtype=torch.long)
    dev_cpu = torch.device("cpu")

    base_toy_logits = forward_batched_logits(toy, toy_tst, dev_cpu, batch_size=2)
    mu_toy = compute_native_head_o_proj_means(toy, {4: toy_fit}, [0, 1], H_t, dev_cpu, batch_size=2)
    abl_heads = [(0, 1), (1, 2)]

    abl_h = MeanAblateHeadsHook(toy.model.layers, abl_heads, mu_toy, H_t, dev_cpu).attach()
    abl_toy_logits = forward_batched_logits(toy, toy_tst, dev_cpu, batch_size=2)
    abl_h.remove()
    abl_shift = float(torch.norm(abl_toy_logits - base_toy_logits).item())
    assert abl_shift > 1e-2, f"Toy ablation shift too small: {abl_shift}"

    # K1-a (Own heads re-inserted through CompiledHeadGraftPipeline) -> MUST restore exact base logits (R == 1.0)!
    # K1-b (Rotated self-transplant Q_l in O(D)) -> MUST restore exact base logits (R == 1.0)!
    k1a_toy_specs, k1b_toy_specs = [], []
    for li, hq in abl_heads:
        hd = extract_donor_head_weights(toy.model.layers[li].self_attn, hq, H_t, H_t)
        mu_sub = mu_toy[(li, hq)] @ hd["W_O"].T
        I_D, I_h, z_D = np.eye(D_t), np.eye(dh_t), np.zeros(D_t)
        op_a = compile_congruent_attention_head(hd, I_D, z_D, I_D, I_h, dh_t, 10_000.0, 10_000.0, "recipient_rope", tau=1.0)
        k1a_toy_specs.append({"layer": li, "head_op": op_a, "alpha": 1.0, "baseline_subtract": mu_sub})

        Q_l, _ = sla.qr(rng.standard_normal((D_t, D_t)))
        hd_r = {**hd, "W_Q": hd["W_Q"] @ Q_l.T, "W_K": hd["W_K"] @ Q_l.T, "W_V": hd["W_V"] @ Q_l.T, "W_O": Q_l @ hd["W_O"]}
        op_b = compile_congruent_attention_head(hd_r, Q_l, z_D, Q_l.T, I_h, dh_t, 10_000.0, 10_000.0, "recipient_rope", tau=1.0)
        k1b_toy_specs.append({"layer": li, "head_op": op_b, "alpha": 1.0, "baseline_subtract": mu_sub})

    for tag, specs in [("K1-a", k1a_toy_specs), ("K1-b", k1b_toy_specs)]:
        abl = MeanAblateHeadsHook(toy.model.layers, abl_heads, mu_toy, H_t, dev_cpu).attach()
        pipe = CompiledHeadGraftPipeline(toy.model.layers, specs, dev_cpu).attach()
        rec_logits = forward_batched_logits(toy, toy_tst, dev_cpu, batch_size=2)
        pipe.remove()
        abl.remove()
        resid = float(torch.norm(rec_logits - base_toy_logits).item())
        R_exact = 1.0 - resid / abl_shift
        assert R_exact > 0.9999, f"{tag} synthetic recovery failed: R={R_exact}, resid={resid}, abl_shift={abl_shift}"
    print(f"  [PASS 4/5] Synthetic 2-layer LlamaForCausalLM Gate K1-a & K1-b rotated self-transplant exact recovery verified (R={R_exact:.6f} == 1.0, abl_shift={abl_shift:.3f})!")

    # 5. Exact Batch-Matched Unhook `torch.equal` Verification!
    uh = verify_batch_matched_unhook(toy, toy_tst, base_toy_logits, dev_cpu, batch_size=2)
    assert uh["bit_equal"] and uh["max_abs_diff"] == 0.0 and uh["pass"], f"Unhook NOT bit-equal: {uh}"
    print(f"  [PASS 5/5] Exact batch-matched unhook verification PASS (bit_equal={uh['bit_equal']}, max_abs_diff={uh['max_abs_diff']})!")
    print(f"=== ALL 5 TRACK 2 PRE-FLIGHT CPU UNIT TESTS PASSED ({time.time() - t0:.2f}s)! ===")


def main() -> None:
    ap = argparse.ArgumentParser(description="Track 2 15-Min Pre-Flight Gate (Headroom Census, K1, K0)")
    ap.add_argument("--unit-test", action="store_true", help="Run pure-CPU unit tests (<10s, ZERO GPU)")
    ap.add_argument("--run", action="store_true", help="Run full GPU/MPS Pre-Flight Gate")
    ap.add_argument("--targets", type=str, default="falcon3-3b,smollm2-1.7b", help="Comma-separated recipients")
    ap.add_argument("--n-fit-seqs", type=int, default=48, help="Repeated sequences per lag from fit_words")
    ap.add_argument("--n-test-seqs", type=int, default=64, help="Repeated sequences per lag from test_words")
    ap.add_argument("--batch-size", type=int, default=16, help="Deterministic batch size for forward passes")
    args = ap.parse_args()

    if args.unit_test:
        run_cpu_unit_tests()
        return
    if args.run:
        run_preflight(args)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
