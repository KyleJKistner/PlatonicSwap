#!/usr/bin/env python3
"""
v2_falsifierA_rope_torus.py — Falsifier A: RoPE Torus-Dissociation Test (Prediction P1)
=======================================================================================

Tests Obstruction O1 (RoPE Torus-Commutant Rigidity) from `docs/COUNCIL_SYNTHESIS.md` (§2.2 O1,
§2.3 #2, §3 Rank 2) and `V2_ROADMAP.md`:

  A position-independent linear map S on RoPE-rotated Q/K heads must commute with the RoPE
  torus: S rho_A(t) = rho_B(t) S, restricting S to matched frequency planes only.

Verified HF model geometries:
  - Donor A:     qwen2.5-3b   (D=2048, L=36, H_q=16, H_kv=2,  d_h=128, rope_theta=1,000,000, q/k/v bias=True)
  - Recipient 1: falcon3-3b   (D=3072, L=22, H_q=12, H_kv=4,  d_h=256, rope_theta=1,000,042, bias=False)
                 -> NEARLY RoPE-NESTED (i -> 2i, max rel freq err <= 4.13e-5, phase-locked to Delta=64!)
  - Recipient 2: smollm2-1.7b (D=2048, L=24, H_q=32, H_kv=32, d_h=64,  rope_theta=130,000,   bias=False)
                 -> RoPE-INCOMMENSURATE (log-step ratio 1.705, predicts decay beyond Delta_coh ~ 4-10!)

Prediction P1: Transplanting positional/induction attention heads from qwen2.5-3b REVERSES v1.0's
residual-socket recipient ranking: Falcon3-3B (RoPE-nested) preserves sharp lag-Delta attention and
causal copying out to Delta=64 under OPTIMAL_ROPE_J, while SmolLM2-1.7B decays sharply with Delta
even though its ROPE_STRIPPED_REF (Donor-RoPE upper bound on the exact same residual socket M_in)
remains strong across all lags Delta in {1, 2, 4, 8, 16, 32, 64}.

Strict v1.0 Read-Only Contract:
  - Does NOT modify any v1.0 engine, benchmark, or result files.
  - Pure-CPU `--unit-test` completes in <10s with ZERO GPU/model forward passes.

Usage:
  PYTHONPATH=platonicswap:experiments python experiments/v2_falsifierA_rope_torus.py --unit-test
  PYTHONPATH=platonicswap:experiments python -u experiments/v2_falsifierA_rope_torus.py --run
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
from scipy.optimize import linear_sum_assignment
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
OUT_PATH = RESULTS_DIR / "results_v2_falsifierA_rope_torus.json"

SEED = 42
LAGS_DELTA: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

DONOR_KEY = "qwen2.5-3b"
RECIPIENT_KEYS: Tuple[str, ...] = ("falcon3-3b", "smollm2-1.7b")

VERIFIED_GEOMETRIES: Dict[str, Dict[str, Any]] = {
    "qwen2.5-3b":   {"D": 2048, "L": 36, "H_q": 16, "H_kv": 2,  "d_h": 128, "rope_theta": 1_000_000.0, "attn_bias": True},
    "falcon3-3b":   {"D": 3072, "L": 22, "H_q": 12, "H_kv": 4,  "d_h": 256, "rope_theta": 1_000_042.0, "attn_bias": False},
    "smollm2-1.7b": {"D": 2048, "L": 24, "H_q": 32, "H_kv": 32, "d_h": 64,  "rope_theta": 130_000.0,   "attn_bias": False},
}

# Candidate pool of simple English nouns/words (filtered at runtime to STRICTLY 1-token across all 3 tokenizers!)
SINGLE_TOKEN_CANDIDATES: Tuple[str, ...] = (
    "apple", "river", "stone", "glass", "train", "bread", "music", "clock", "light", "water",
    "paper", "chair", "table", "house", "green", "black", "white", "north", "south", "metal",
    "cloud", "storm", "flame", "earth", "ocean", "beach", "grass", "plant", "fruit", "sugar",
    "wheat", "grain", "honey", "lemon", "olive", "onion", "peach", "grape", "melon", "berry",
    "steel", "brass", "iron", "gold", "silver", "glass", "brick", "paint", "chalk", "cloth",
    "linen", "shirt", "boots", "gloves", "scarf", "crown", "sword", "shield", "arrow", "spear",
    "wheel", "motor", "cable", "lever", "screw", "chain", "frame", "board", "panel", "block",
    "tower", "vault", "porch", "floor", "walls", "doors", "steps", "roads", "lanes", "paths",
    "parks", "woods", "hills", "peaks", "caves", "lakes", "ponds", "wells", "dams", "ports",
    "ships", "boats", "sails", "oars", "decks", "wings", "tails", "horns", "claws", "teeth",
    "bones", "heart", "brain", "lungs", "nerve", "blood", "voice", "sound", "noise", "songs",
    "notes", "drums", "bells", "horns", "flute", "piano", "choir", "dance", "stage", "films",
    "books", "pages", "lines", "words", "names", "dates", "years", "weeks", "hours", "night",
    "dawn", "noon", "dusk", "frost", "sleet", "rains", "winds", "tides", "waves", "sands",
    "rocks", "soils", "roots", "stems", "seeds", "bloom", "trees", "pines", "palms", "ferns",
    "moss", "vines", "herbs", "spices", "salts", "oils", "waxes", "glues", "inks", "dyes",
    "coins", "bills", "banks", "shops", "malls", "trade", "goods", "cargo", "loads", "packs",
    "boxes", "bags", "jars", "cups", "bowls", "forks", "spoon", "knife", "plate", "stove",
    "ovens", "lamps", "bulbs", "wires", "plugs", "cells", "chips", "disks", "codes", "files",
    "texts", "lists", "maps", "plans", "chart", "graph", "scale", "meter", "gauge", "ruler",
    "proof", "logic", "rules", "facts", "ideas", "views", "goals", "tasks", "roles", "teams",
    "clubs", "games", "sport", "match", "score", "prize", "award", "medal", "title", "ranks",
)


# ======================================================================================
# 1. Exact HF RoPE Mathematics (`rotate_half` Convention) & Optimal 1-D Intertwiner J
# ======================================================================================

def rope_inv_freq(d_h: int, rope_theta: float) -> np.ndarray:
    """Exact HuggingFace inv_freq for half-split RoPE: theta_i = b^(-2i / d_h), i = 0..d_h/2-1."""
    assert d_h % 2 == 0 and d_h > 0
    idx = np.arange(0, d_h, 2, dtype=np.float64)
    return 1.0 / (float(rope_theta) ** (idx / float(d_h)))


def rope_matrix(d_h: int, rope_theta: float, t: float) -> np.ndarray:
    """
    Exact d_h x d_h orthogonal RoPE rotation matrix rho(t) in HuggingFace's half-split `rotate_half` basis:
      (rho(t) x)_i       = cos(t * theta_i) * x_i - sin(t * theta_i) * x_{i + m}
      (rho(t) x)_{i + m} = sin(t * theta_i) * x_i + cos(t * theta_i) * x_{i + m}
    where m = d_h // 2 and i in {0, ..., m - 1}.
    """
    m = d_h // 2
    angles = float(t) * rope_inv_freq(d_h, rope_theta)
    c = np.cos(angles)
    s = np.sin(angles)
    R = np.zeros((d_h, d_h), dtype=np.float64)
    idx = np.arange(m)
    R[idx, idx] = c
    R[idx, idx + m] = -s
    R[idx + m, idx] = s
    R[idx + m, idx + m] = c
    return R


def apply_rope_half_torch(
    x: torch.Tensor,
    inv_freq: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """
    Exact HuggingFace `apply_rotary_pos_emb` on a single tensor x of shape (..., T, d_h).
    `inv_freq`: (d_h // 2,), `positions`: (T,) or (B, T).
    """
    freqs = positions.to(inv_freq.dtype).unsqueeze(-1) * inv_freq  # (..., T, m)
    emb = torch.cat([freqs, freqs], dim=-1)                        # (..., T, d_h)
    cos = emb.cos().to(x.dtype)
    sin = emb.sin().to(x.dtype)
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    m = x.shape[-1] // 2
    x_rot = torch.cat((-x[..., m:], x[..., :m]), dim=-1)
    return (x * cos) + (x_rot * sin)


def build_optimal_rope_intertwiner(
    d_hA: int,
    theta_A: float,
    d_hB: int,
    theta_B: float,
    plane_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build exact 1-D optimal log-frequency RoPE intertwiner J in R^{d_hB x d_hA} preserving 2x2
    `rotate_half` plane block structure: plane i of A -> plane j(i) of B via I_2 on
    (j, j + m_B) <- (i, i + m_A).

    Uses exact linear_sum_assignment on log-frequency compatibility:
      - When m_A <= m_B (e.g. Qwen 64 -> Falcon3 128), EVERY donor plane i is injectively matched
        to its optimal recipient plane j(i) (yielding exact i -> 2i for Qwen -> Falcon3!).
      - When m_A > m_B (e.g. Qwen 64 -> SmolLM2 32), selects the m_B donor planes maximizing
        sum omega_i * exp(-|ln theta^A_i - ln theta^B_{j(i)}|) so both frequency alignment and
        (optional) donor head plane energy omega_i are respected without penalizing active planes!
    """
    mA = d_hA // 2
    mB = d_hB // 2
    fA = rope_inv_freq(d_hA, theta_A)
    fB = rope_inv_freq(d_hB, theta_B)
    log_diff = np.abs(np.log(fA)[:, None] - np.log(fB)[None, :])  # (mA, mB)

    if plane_weights is None:
        w = np.ones(mA, dtype=np.float64)
    else:
        w = np.asarray(plane_weights, dtype=np.float64).reshape(mA)
        w = np.maximum(w, 1e-12)
        w = w / np.mean(w)

    if mA <= mB:
        # Every donor plane must be matched: minimize weighted |ln fA_i - ln fB_j|
        cost = w[:, None] * log_diff
        row_ind, col_ind = linear_sum_assignment(cost)
    else:
        # Only mB donor planes can be embedded: maximize sum w_i * exp(-|ln fA_i - ln fB_j|)
        score = w[:, None] * np.exp(-log_diff)
        row_ind, col_ind = linear_sum_assignment(-score)

    J = np.zeros((d_hB, d_hA), dtype=np.float64)
    rel_errs = []
    pairs = []
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        J[j, i] = 1.0
        J[j + mB, i + mA] = 1.0
        rel_errs.append(abs(fB[j] - fA[i]) / max(fA[i], 1e-18))
        pairs.append((int(i), int(j)))

    info = {
        "m_A": int(mA),
        "m_B": int(mB),
        "n_matched_planes": int(len(pairs)),
        "matched_pairs": pairs,
        "max_rel_freq_err": float(max(rel_errs)) if rel_errs else 1.0,
        "mean_rel_freq_err": float(np.mean(rel_errs)) if rel_errs else 1.0,
        "mean_abs_log_freq_err": float(np.mean([log_diff[i, j] for i, j in pairs])) if pairs else float("inf"),
        "energy_retained_frac": float(np.sum(w[row_ind]) / np.sum(w)),
    }
    return J, info


def build_naive_index_intertwiner(d_hA: int, d_hB: int) -> np.ndarray:
    """Naive half-split index mapping: plane i -> plane i for i = 0..min(mA, mB)-1."""
    mA, mB = d_hA // 2, d_hB // 2
    k = min(mA, mB)
    J = np.zeros((d_hB, d_hA), dtype=np.float64)
    for i in range(k):
        J[i, i] = 1.0
        J[i + mB, i + mA] = 1.0
    return J


def build_haar_random_intertwiner(d_hA: int, d_hB: int, seed: int) -> np.ndarray:
    """Haar-random partial isometry J in O(d_hB, d_hA) breaking RoPE torus commutation."""
    rng = np.random.default_rng(seed)
    if d_hB >= d_hA:
        Q, _ = sla.qr(rng.standard_normal((d_hB, d_hA)), mode="economic")
        return Q
    Q, _ = sla.qr(rng.standard_normal((d_hA, d_hB)), mode="economic")
    return Q.T


def rope_commutation_error_curve(
    J: np.ndarray,
    d_hA: int,
    theta_A: float,
    d_hB: int,
    theta_B: float,
    lags: Sequence[int] = LAGS_DELTA,
) -> Dict[str, float]:
    """
    Compute exact normalized RoPE torus-commutation error:
      err(t) = || J rho_A(t) - rho_B(t) J ||_F / max(||J||_F, 1e-12)
    Notice: err(t) = 0 iff J intertwines rho_A(t) and rho_B(t) exactly on all mapped planes!
    """
    j_norm = max(float(sla.norm(J, "fro")), 1e-12)
    out: Dict[str, float] = {}
    for t in lags:
        RA = rope_matrix(d_hA, theta_A, t)
        RB = rope_matrix(d_hB, theta_B, t)
        err = float(sla.norm(J @ RA - RB @ J, "fro")) / j_norm
        out[str(t)] = round(err, 6)
    return out


# ======================================================================================
# 2. Unlabeled Residual Read/Write Sockets (Zero-Label, Parameter-Free LW + Cauchy SVD)
# ======================================================================================

def ledoit_wolf_cov(X: np.ndarray) -> Tuple[np.ndarray, float]:
    Xf = np.asarray(X, dtype=np.float64)
    n, d = Xf.shape
    Xc = Xf - Xf.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / float(n)
    mu = float(np.trace(S)) / float(d)
    sum_S2 = float(np.sum(S * S))
    delta2 = sum_S2 - d * (mu * mu)
    if delta2 <= 1e-18:
        return mu * np.eye(d, dtype=np.float64), 1.0
    row_norm2 = np.sum(Xc * Xc, axis=1)
    beta2 = (float(np.mean(row_norm2 * row_norm2)) - sum_S2) / float(n)
    shrink = float(np.clip(beta2 / delta2, 0.0, 1.0))
    return (1.0 - shrink) * S + shrink * mu * np.eye(d, dtype=np.float64), shrink


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Exact centered linear CKA between X (N, d1) and Y (N, d2)."""
    Xc = np.asarray(X, dtype=np.float64)
    Yc = np.asarray(Y, dtype=np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    Yc = Yc - Yc.mean(axis=0, keepdims=True)
    Gxy = Xc.T @ Yc
    Gxx = Xc.T @ Xc
    Gyy = Yc.T @ Yc
    num = float(np.sum(Gxy * Gxy))
    den = math.sqrt(max(float(np.sum(Gxx * Gxx)) * float(np.sum(Gyy * Gyy)), 1e-24))
    return num / den


def fit_affine_read_socket(
    X_B_fit: np.ndarray,
    X_A_fit: np.ndarray,
    X_B_val: Optional[np.ndarray] = None,
    X_A_val: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Fit unlabeled affine read socket B -> A using analytical Ledoit-Wolf shrinkage (ZERO free lambda!):
      x_A_hat = (x_B - mu_B) @ M_in.T + mu_A = x_B @ M_in.T + d_in
    """
    Xb = np.asarray(X_B_fit, dtype=np.float64)
    Xa = np.asarray(X_A_fit, dtype=np.float64)
    mu_B = Xb.mean(axis=0)
    mu_A = Xa.mean(axis=0)
    Xbc = Xb - mu_B
    Xac = Xa - mu_A
    S_BB, sh = ledoit_wolf_cov(Xb)
    C_AB = (Xac.T @ Xbc) / float(len(Xb))
    M_in = sla.solve(S_BB, C_AB.T, assume_a="pos").T  # (D_A, D_B)
    d_in = mu_A - M_in @ mu_B

    r2_val = None
    if X_B_val is not None and X_A_val is not None and len(X_B_val) > 0:
        Xb_v = np.asarray(X_B_val, dtype=np.float64)
        Xa_v = np.asarray(X_A_val, dtype=np.float64)
        pred = Xb_v @ M_in.T + d_in
        ss_res = float(np.sum((Xa_v - pred) ** 2))
        ss_tot = float(np.sum((Xa_v - Xa_v.mean(axis=0, keepdims=True)) ** 2))
        r2_val = 1.0 - ss_res / max(ss_tot, 1e-12)

    return {
        "M_in": M_in,
        "d_in": d_in,
        "mu_A": mu_A,
        "mu_B": mu_B,
        "shrink_BB": float(sh),
        "heldout_read_R2": round(float(r2_val), 4) if r2_val is not None else None,
    }


def fit_cauchy_write_socket(
    Y_A_fit: np.ndarray,
    Y_B_fit: np.ndarray,
    Y_A_val: Optional[np.ndarray] = None,
    Y_B_val: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Fit unlabeled Cauchy-weighted Stiefel write socket A -> B (post-self_attn residual space):
      w_i = 1 / (1 + (||yA_i|| / med_A)^4 + (||yB_i|| / med_B)^4)
      R_out = U @ Vt,  s_out = sum(S) / E_w[||R_out (yA - mu_A)||^2],  M_out = s_out * R_out.
    """
    Ya = np.asarray(Y_A_fit, dtype=np.float64)
    Yb = np.asarray(Y_B_fit, dtype=np.float64)
    nA = np.linalg.norm(Ya, axis=-1)
    nB = np.linalg.norm(Yb, axis=-1)
    med_A = max(float(np.median(nA)), 1e-12)
    med_B = max(float(np.median(nB)), 1e-12)
    w = 1.0 / (1.0 + (nA / med_A) ** 4 + (nB / med_B) ** 4)
    wn = w / max(float(np.sum(w)), 1e-12)
    mu_A = np.sum(wn[:, None] * Ya, axis=0)
    mu_B = np.sum(wn[:, None] * Yb, axis=0)
    Yac = Ya - mu_A
    Ybc = Yb - mu_B
    C_BA = (Ybc * wn[:, None]).T @ Yac
    U, S, Vt = sla.svd(C_BA, full_matrices=False)
    R_out = U @ Vt
    proj = Yac @ R_out.T
    denom = float(np.sum(wn * np.sum(proj * proj, axis=-1)))
    s_out = float(np.sum(S)) / max(denom, 1e-12)
    M_out = s_out * R_out

    cos_val = None
    if Y_A_val is not None and Y_B_val is not None and len(Y_A_val) > 0:
        Ya_v = np.asarray(Y_A_val, dtype=np.float64) - mu_A
        Yb_v = np.asarray(Y_B_val, dtype=np.float64) - mu_B
        pred = Ya_v @ M_out.T
        wv = 1.0 / (1.0 + (np.linalg.norm(Ya_v, axis=-1) / med_A) ** 4 + (np.linalg.norm(Yb_v, axis=-1) / med_B) ** 4)
        wv = wv / max(float(np.sum(wv)), 1e-12)
        c = np.sum(pred * Yb_v, axis=-1) / (np.linalg.norm(pred, axis=-1) * np.linalg.norm(Yb_v, axis=-1) + 1e-12)
        cos_val = float(np.sum(wv * c))

    return {
        "M_out": M_out,
        "R_out": R_out,
        "s_out": float(s_out),
        "heldout_write_cos": round(float(cos_val), 4) if cos_val is not None else None,
    }


# ======================================================================================
# 3. Congruent Attention-Head Operator + 1-D Entropy Bisection + Causal Graft Hook
# ======================================================================================

def extract_donor_head_weights(attn: nn.Module, head_q: int, num_heads: int, num_kv_heads: int) -> Dict[str, np.ndarray]:
    """
    Extract exact single-head (W_Q, b_Q, W_K, b_K, W_V, b_V, W_O) from a HF Qwen2Attention / LlamaAttention.
    Handles GQA (h_kv = h_q // (H_q // H_kv)) and Qwen2 q/k/v biases faithfully!
    """
    d_model = int(attn.q_proj.in_features)
    d_h = int(attn.q_proj.out_features) // int(num_heads)
    gqa_group = int(num_heads) // int(num_kv_heads)
    head_kv = int(head_q) // gqa_group

    q0, q1 = head_q * d_h, (head_q + 1) * d_h
    kv0, kv1 = head_kv * d_h, (head_kv + 1) * d_h

    W_Q = attn.q_proj.weight.detach().float().cpu().numpy()[q0:q1].astype(np.float64)
    W_K = attn.k_proj.weight.detach().float().cpu().numpy()[kv0:kv1].astype(np.float64)
    W_V = attn.v_proj.weight.detach().float().cpu().numpy()[kv0:kv1].astype(np.float64)
    W_O = attn.o_proj.weight.detach().float().cpu().numpy()[:, q0:q1].astype(np.float64)

    b_Q = (
        attn.q_proj.bias.detach().float().cpu().numpy()[q0:q1].astype(np.float64)
        if attn.q_proj.bias is not None
        else np.zeros(d_h, dtype=np.float64)
    )
    b_K = (
        attn.k_proj.bias.detach().float().cpu().numpy()[kv0:kv1].astype(np.float64)
        if attn.k_proj.bias is not None
        else np.zeros(d_h, dtype=np.float64)
    )
    b_V = (
        attn.v_proj.bias.detach().float().cpu().numpy()[kv0:kv1].astype(np.float64)
        if attn.v_proj.bias is not None
        else np.zeros(d_h, dtype=np.float64)
    )

    return {
        "head_q": int(head_q),
        "head_kv": int(head_kv),
        "d_model": d_model,
        "d_h": d_h,
        "W_Q": W_Q, "b_Q": b_Q,
        "W_K": W_K, "b_K": b_K,
        "W_V": W_V, "b_V": b_V,
        "W_O": W_O,
    }


def compute_single_head_attention(
    x_attn: torch.Tensor,
    W_Q: torch.Tensor, b_Q: torch.Tensor,
    W_K: torch.Tensor, b_K: torch.Tensor,
    W_V: Optional[torch.Tensor], b_V: Optional[torch.Tensor],
    W_O: Optional[torch.Tensor],
    inv_freq: Optional[torch.Tensor],
    tau: float = 1.0,
    scale_override: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Exact single-head causal self-attention on x_attn of shape (B, T, D):
      q = (x W_Q^T + b_Q),  k = (x W_K^T + b_K)
      if inv_freq is not None: apply exact HF half-split RoPE at positions 0..T-1
      logits = (tau / sqrt(d_h_eff)) * q_rot @ k_rot^T + causal_mask
      A = softmax(logits, dim=-1)
      y = (A @ (x W_V^T + b_V)) @ W_O^T   (if W_V/W_O provided)
    """
    B, T, _ = x_attn.shape
    d_h = W_Q.shape[0]
    q = F.linear(x_attn, W_Q, b_Q)  # (B, T, d_h)
    k = F.linear(x_attn, W_K, b_K)  # (B, T, d_h)

    if inv_freq is not None:
        pos = torch.arange(T, device=x_attn.device, dtype=torch.float32)
        q = apply_rope_half_torch(q, inv_freq.to(x_attn.device), pos)
        k = apply_rope_half_torch(k, inv_freq.to(x_attn.device), pos)

    scale = (1.0 / math.sqrt(float(d_h))) if scale_override is None else float(scale_override)
    logits = torch.matmul(q, k.transpose(-1, -2)) * (float(tau) * scale)
    causal = torch.triu(torch.ones((T, T), device=x_attn.device, dtype=torch.bool), diagonal=1)
    logits = logits.masked_fill(causal.unsqueeze(0), -1e9)
    A = F.softmax(logits.float(), dim=-1)

    y = None
    if W_V is not None and W_O is not None:
        v = F.linear(x_attn, W_V, b_V)
        ctx = torch.matmul(A.to(v.dtype), v)
        y = F.linear(ctx, W_O, None)
    return A, y


def mean_attention_entropy(A: torch.Tensor, min_pos: int = 2) -> float:
    """Mean Shannon entropy H(A[b, t, :]) over positions t >= min_pos."""
    A_sub = A[:, min_pos:, :].clamp(min=1e-12)
    ent = -(A_sub * A_sub.log()).sum(dim=-1)
    return float(ent.mean().item())


def compile_congruent_attention_head(
    donor_head: Dict[str, np.ndarray],
    M_in: np.ndarray,
    d_in: np.ndarray,
    M_out: np.ndarray,
    J: Optional[np.ndarray],
    d_hB: int,
    theta_A: float,
    theta_B: float,
    rope_mode: str = "recipient_rope",   # "recipient_rope" (with J), "donor_rope" (ROPE_STRIPPED_REF), or "no_rope"
    tau: float = 1.0,
) -> Dict[str, Any]:
    """
    Exact Congruent Attention-Head Operator (`docs/COUNCIL_SYNTHESIS.md` §2.3 #2):
      W_Q^B = (scale_dim) * J @ W_Q^A @ M_in,    b_Q^B = (scale_dim) * J @ (W_Q^A @ d_in + b_Q^A)
      W_K^B = (scale_dim) * J @ W_K^A @ M_in,    b_K^B = (scale_dim) * J @ (W_K^A @ d_in + b_K^A)
      W_V^B = W_V^A @ M_in,                      b_V^B = W_V^A @ d_in + b_V^A      (J_V = I_{d_hA}, internal!)
      W_O^B = M_out @ W_O^A                                                        (J_V^T = I_{d_hA}!)

    Note: Because V and O do NOT receive RoPE, J_V = I_{d_hA} cancels identically in W_O^B @ W_V^B
    (zero information loss in the V-O path even when d_hB < d_hA!), cleanly isolating O1 purely in Q-K!
    `scale_dim = (d_h_eff / d_hA) ** 0.25` exactly cancels the 1/sqrt(d_hB) vs 1/sqrt(d_hA) attention
    divisor mismatch so tau = 1.0 preserves exact donor logit scale before 1-D entropy bisection!
    """
    d_hA = int(donor_head["d_h"])
    W_Q_A, b_Q_A = donor_head["W_Q"], donor_head["b_Q"]
    W_K_A, b_K_A = donor_head["W_K"], donor_head["b_K"]
    W_V_A, b_V_A = donor_head["W_V"], donor_head["b_V"]
    W_O_A = donor_head["W_O"]

    # Map donor residual input affine transform: x_A = M_in x_B + d_in
    WQ_m = W_Q_A @ M_in
    bQ_m = W_Q_A @ d_in + b_Q_A
    WK_m = W_K_A @ M_in
    bK_m = W_K_A @ d_in + b_K_A
    WV_m = W_V_A @ M_in
    bV_m = W_V_A @ d_in + b_V_A
    WO_m = M_out @ W_O_A

    if rope_mode == "recipient_rope":
        assert J is not None and J.shape == (d_hB, d_hA)
        scale_dim = (float(d_hB) / float(d_hA)) ** 0.25
        W_Q_B = scale_dim * (J @ WQ_m)
        b_Q_B = scale_dim * (J @ bQ_m)
        W_K_B = scale_dim * (J @ WK_m)
        b_K_B = scale_dim * (J @ bK_m)
        inv_freq = rope_inv_freq(d_hB, theta_B)
        d_h_eff = d_hB
    elif rope_mode == "donor_rope":
        # Exact analytic upper bound: same residual read socket M_in, Donor A's own exact RoPE!
        W_Q_B, b_Q_B = WQ_m, bQ_m
        W_K_B, b_K_B = WK_m, bK_m
        inv_freq = rope_inv_freq(d_hA, theta_A)
        d_h_eff = d_hA
    elif rope_mode == "no_rope":
        W_Q_B, b_Q_B = WQ_m, bQ_m
        W_K_B, b_K_B = WK_m, bK_m
        inv_freq = None
        d_h_eff = d_hA
    else:
        raise ValueError(f"Unknown rope_mode: {rope_mode}")

    return {
        "W_Q": W_Q_B, "b_Q": b_Q_B,
        "W_K": W_K_B, "b_K": b_K_B,
        "W_V": WV_m,  "b_V": bV_m,
        "W_O": WO_m,
        "inv_freq": inv_freq,
        "d_h_eff": int(d_h_eff),
        "rope_mode": rope_mode,
        "tau": float(tau),
    }


def calibrate_entropy_temperature_bisection(
    head_op: Dict[str, Any],
    x_cal_B: torch.Tensor,
    target_entropy: float,
    lo: float = 0.05,
    hi: float = 16.0,
    n_iter: int = 24,
) -> float:
    """
    1-D bisection for scalar temperature tau > 0 matching donor attention entropy on calibration text.
    Higher tau -> sharper attention -> lower entropy (strictly monotone decreasing in tau!).
    """
    dev = x_cal_B.device
    WQ = torch.from_numpy(head_op["W_Q"].astype(np.float32)).to(dev)
    bQ = torch.from_numpy(head_op["b_Q"].astype(np.float32)).to(dev)
    WK = torch.from_numpy(head_op["W_K"].astype(np.float32)).to(dev)
    bK = torch.from_numpy(head_op["b_K"].astype(np.float32)).to(dev)
    inv_f = (
        torch.from_numpy(head_op["inv_freq"].astype(np.float32)).to(dev)
        if head_op["inv_freq"] is not None
        else None
    )

    def _ent(tau_v: float) -> float:
        with torch.no_grad():
            A, _ = compute_single_head_attention(x_cal_B, WQ, bQ, WK, bK, None, None, None, inv_f, tau=tau_v)
            return mean_attention_entropy(A, min_pos=2)

    e_lo = _ent(lo)
    e_hi = _ent(hi)
    if target_entropy >= e_lo:
        return float(lo)
    if target_entropy <= e_hi:
        return float(hi)

    a, b = float(lo), float(hi)
    for _ in range(n_iter):
        mid = 0.5 * (a + b)
        if _ent(mid) > target_entropy:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)


class AdditiveHeadResidualGraft:
    """
    Removable causal forward hook attaching a compiled congruent attention head as an additive
    residual head on `layer.self_attn` (exact bit-identical restoration upon `.remove()`!).
    """

    def __init__(self, layer: nn.Module, head_op: Dict[str, Any], device: torch.device, alpha_graft: float = 1.0):
        self.layer = layer
        self.alpha_graft = float(alpha_graft)
        self.tau = float(head_op["tau"])
        self.WQ = torch.from_numpy(head_op["W_Q"].astype(np.float32)).to(device)
        self.bQ = torch.from_numpy(head_op["b_Q"].astype(np.float32)).to(device)
        self.WK = torch.from_numpy(head_op["W_K"].astype(np.float32)).to(device)
        self.bK = torch.from_numpy(head_op["b_K"].astype(np.float32)).to(device)
        self.WV = torch.from_numpy(head_op["W_V"].astype(np.float32)).to(device)
        self.bV = torch.from_numpy(head_op["b_V"].astype(np.float32)).to(device)
        self.WO = torch.from_numpy(head_op["W_O"].astype(np.float32)).to(device)
        self.inv_freq = (
            torch.from_numpy(head_op["inv_freq"].astype(np.float32)).to(device)
            if head_op["inv_freq"] is not None
            else None
        )
        self._buf: Dict[str, torch.Tensor] = {}
        self._h_ln = None
        self._h_attn = None

    def attach(self) -> "AdditiveHeadResidualGraft":
        def _cap_ln(_m, _inp, out):
            self._buf["x_attn"] = out

        def _add_head(_m, _inp, out):
            x_a = self._buf["x_attn"].float()
            _, y_head = compute_single_head_attention(
                x_a, self.WQ, self.bQ, self.WK, self.bK, self.WV, self.bV, self.WO, self.inv_freq, tau=self.tau
            )
            delta = (self.alpha_graft * y_head).to(out[0].dtype if isinstance(out, tuple) else out.dtype)
            if isinstance(out, tuple):
                return (out[0] + delta,) + out[1:]
            return out + delta

        self._h_ln = self.layer.input_layernorm.register_forward_hook(_cap_ln)
        self._h_attn = self.layer.self_attn.register_forward_hook(_add_head)
        return self

    def remove(self) -> None:
        if self._h_ln is not None:
            self._h_ln.remove()
            self._h_ln = None
        if self._h_attn is not None:
            self._h_attn.remove()
            self._h_attn = None
        self._buf.clear()


# ======================================================================================
# 4. Shared Single-Token Repeated-Sequence Prompt Suite (Exact Lag Alignment Across All 3 Tokenizers!)
# ======================================================================================

def build_shared_single_token_words(tokenizers: Dict[str, Any]) -> List[str]:
    """
    Filter SINGLE_TOKEN_CANDIDATES to words w where `" " + w` encodes to STRICTLY 1 token in
    EVERY tokenizer (qwen2.5-3b, falcon3-3b, smollm2-1.7b), guaranteeing exact 1-to-1 token
    index and lag Delta alignment across all 3 architectures!
    """
    clean: List[str] = []
    seen = set()
    for w in SINGLE_TOKEN_CANDIDATES:
        if w in seen:
            continue
        seen.add(w)
        ok = True
        for tok in tokenizers.values():
            ids = tok.encode(" " + w, add_special_tokens=False)
            if len(ids) != 1:
                ok = False
                break
        if ok:
            clean.append(w)
    return clean


def build_lag_copy_sequences(
    shared_words: Sequence[str],
    lags: Sequence[int] = LAGS_DELTA,
    n_prompts_per_lag: int = 8,
    seed: int = SEED,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Build exact repeated-sequence prompts for EVERY lag Delta in {1, 2, 4, 8, 16, 32, 64}:
      - Prefix anchor: `"List:"` (exact known token count per tokenizer; or pure word stream starting with anchor `"The"`).
      - Specifically, using ONLY verified single-token words starting with a neutral anchor `"The"`:
          - For lag Delta == 1: alternating / repeated pairs `[w0, w0, w1, w1, ..., w_{M-1}, w_{M-1}]` (len 32)
            so at every odd index t, token t-1 is an exact duplicate (lag 1)!
          - For lag Delta >= 2: two back-to-back copies of a non-repeating block of length Delta:
              block = [w_0, w_1, ..., w_{Delta-1}]
              seq   = [anchor] + block + block
            At every position t in [1 + Delta, 1 + 2*Delta - 1]:
              - Exact duplicate token is at position `t - Delta` (lag == Delta)!
              - Exact induction target (next token to copy) is at position `t - Delta + 1` (lag == Delta - 1)!
    """
    rng = np.random.default_rng(seed)
    words = list(shared_words)
    assert len(words) >= max(lags), f"Need at least {max(lags)} shared single-token words, got {len(words)}"

    suite: Dict[int, List[Dict[str, Any]]] = {}
    for delta in lags:
        items: List[Dict[str, Any]] = []
        for pid in range(n_prompts_per_lag):
            if delta == 1:
                # 16 unique words each repeated twice: [w0, w0, w1, w1, ...] -> exact lag-1 copies at every odd pos
                chosen = rng.choice(words, size=16, replace=False).tolist()
                seq_words = []
                for w in chosen:
                    seq_words.extend([w, w])
                text = " ".join(seq_words)
                # Positions where prev token is identical (relative to 0-indexed word stream)
                eval_word_positions = list(range(1, len(seq_words), 2))
                items.append({
                    "lag": 1,
                    "prompt_id": pid,
                    "words": seq_words,
                    "text": text,
                    "eval_word_positions": eval_word_positions,
                })
            else:
                # Non-repeating block of length `delta` repeated twice!
                # Total words = 2 * delta (for delta=64 -> 128 words, well within context & fast!)
                block = rng.choice(words, size=delta, replace=False).tolist()
                seq_words = block + block
                text = " ".join(seq_words)
                # In second copy (word indices delta .. 2*delta - 2), predicting word[idx + 1] tests lag `delta`!
                eval_word_positions = list(range(delta, 2 * delta - 1))
                items.append({
                    "lag": int(delta),
                    "prompt_id": pid,
                    "words": seq_words,
                    "text": text,
                    "eval_word_positions": eval_word_positions,
                })
        suite[int(delta)] = items
    return suite


def encode_word_stream_exact(tok: Any, words: Sequence[str]) -> Tuple[List[int], int]:
    """
    Encode a sequence of verified shared single-token words using `" " + w` per word preceded by
    a single BOS / anchor token, guaranteeing:
      len(input_ids) == 1 + len(words)   IDENTICALLY across Qwen2.5-3B, Falcon3-3B, and SmolLM2-1.7B!
    Returns (input_ids, offset=1) where word index k sits at EXACT token index `1 + k`.
    """
    bos = tok.bos_token_id
    if bos is None:
        bos = tok.eos_token_id
    ids = [int(bos)]
    for w in words:
        w_ids = tok.encode(" " + w, add_special_tokens=False)
        assert len(w_ids) == 1, f"Word {w!r} was not 1 token: {w_ids}"
        ids.append(int(w_ids[0]))
    return ids, 1


# ======================================================================================
# 5. Stage 1 & Stage 2 Runner (MPS / GPU Execution Only via `--run`)
# ======================================================================================

def extract_input_ln_states(
    model: Any,
    input_ids_batch: torch.Tensor,
    layer_indices: Sequence[int],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """Capture `input_layernorm` outputs (B, T, D) at selected layers in a single forward pass."""
    from extract_and_baseline import get_decoder_layers

    layers = get_decoder_layers(model)
    buf: Dict[int, torch.Tensor] = {}
    hooks = []
    for li in layer_indices:
        def _mk(idx: int):
            def _fn(_m, _i, o):
                buf[idx] = o.detach().float().cpu()
            return _fn
        hooks.append(layers[li].input_layernorm.register_forward_hook(_mk(li)))
    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids_batch.to(device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return buf


def extract_attn_port_passages(
    model: Any,
    tok: Any,
    device: torch.device,
    passages: Sequence[str],
    layer_indices: Sequence[int],
    max_len: int = 128,
) -> Dict[int, Dict[str, Any]]:
    """
    Extract aligned per-passage `x_attn` (input_layernorm output) and `y_attn` (self_attn output)
    plus exact char offsets for character-offset cross-tokenizer pairing!
    """
    from extract_and_baseline import get_decoder_layers

    layers = get_decoder_layers(model)
    per_layer: Dict[int, Dict[str, List[Any]]] = {
        li: {"x_attn": [], "y_attn": [], "offsets": [], "texts": []} for li in layer_indices
    }
    cur_buf: Dict[int, Dict[str, torch.Tensor]] = {li: {} for li in layer_indices}
    hooks = []
    for li in layer_indices:
        def _mk_x(idx: int):
            def _fn(_m, _i, o):
                cur_buf[idx]["x"] = o.detach().float().cpu()
            return _fn
        def _mk_y(idx: int):
            def _fn(_m, _i, o):
                t = o[0] if isinstance(o, tuple) else o
                cur_buf[idx]["y"] = t.detach().float().cpu()
            return _fn
        hooks.append(layers[li].input_layernorm.register_forward_hook(_mk_x(li)))
        hooks.append(layers[li].self_attn.register_forward_hook(_mk_y(li)))

    try:
        with torch.no_grad():
            for txt in passages:
                enc = tok(txt, return_offsets_mapping=True, truncation=True, max_length=max_len, add_special_tokens=True)
                ids = torch.tensor([enc["input_ids"]], dtype=torch.long, device=device)
                _ = model(input_ids=ids, use_cache=False)
                offs = enc["offset_mapping"]
                for li in layer_indices:
                    per_layer[li]["x_attn"].append(cur_buf[li]["x"][0].numpy())
                    per_layer[li]["y_attn"].append(cur_buf[li]["y"][0].numpy())
                    per_layer[li]["offsets"].append(offs)
                    per_layer[li]["texts"].append(txt)
    finally:
        for h in hooks:
            h.remove()
    return per_layer


def pair_attn_port_activations(
    stA: Dict[str, List[Any]],
    stB: Dict[str, List[Any]],
) -> Dict[str, np.ndarray]:
    """Pair token activations across different tokenizers by exact character-end offset (skipping pos 0)."""
    xa_l, xb_l, ya_l, yb_l = [], [], [], []
    for xA, yA, oA, xB, yB, oB in zip(
        stA["x_attn"], stA["y_attn"], stA["offsets"],
        stB["x_attn"], stB["y_attn"], stB["offsets"],
    ):
        end_to_iA = {}
        for i in range(1, len(oA)):
            a0, a1 = oA[i]
            if a1 > a0:
                end_to_iA[a1] = i
        for j in range(1, len(oB)):
            b0, b1 = oB[j]
            if b1 > b0 and b1 in end_to_iA:
                i = end_to_iA[b1]
                xa_l.append(xA[i])
                xb_l.append(xB[j])
                ya_l.append(yA[i])
                yb_l.append(yB[j])
    return {
        "x_A": np.stack(xa_l, axis=0),
        "x_B": np.stack(xb_l, axis=0),
        "y_A": np.stack(ya_l, axis=0),
        "y_B": np.stack(yb_l, axis=0),
    }


def attention_metrics_vs_donor(
    A_donor: torch.Tensor,
    A_recip: torch.Tensor,
    target_offsets: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict[str, float]:
    """
    Compare recipient attention matrix A_recip (B, T, T) against donor A_donor (B, T, T):
      - `kl_donor_to_recip`: mean KL(A_donor || A_recip) over positions t >= 2
      - `pearson_r`: Pearson correlation between flattened lower-triangular attention entries (t >= 2)
      - `top1_match`: fraction of positions t >= 2 where argmax(A_recip[t]) == argmax(A_donor[t])
      - `target_mass_donor`, `target_mass_recip`: mean attention mass at exact target pairs `(q_pos, k_pos)`
    """
    Ad = A_donor[:, 2:, :].float().clamp(min=1e-9)
    Ar = A_recip[:, 2:, :].float().clamp(min=1e-9)
    kl = float((Ad * (Ad.log() - Ar.log())).sum(dim=-1).mean().item())
    top1 = float((A_donor[:, 2:, :].argmax(dim=-1) == A_recip[:, 2:, :].argmax(dim=-1)).float().mean().item())

    # Flatten strictly causal lower-triangular entries for rows t >= 2
    T = A_donor.shape[1]
    mask = torch.tril(torch.ones((T, T), dtype=torch.bool), diagonal=0)[2:, :]
    vd = A_donor[:, 2:, :][:, mask].float().cpu().numpy().ravel()
    vr = A_recip[:, 2:, :][:, mask].float().cpu().numpy().ravel()
    vd_c = vd - vd.mean()
    vr_c = vr - vr.mean()
    r = float(np.dot(vd_c, vr_c) / max(math.sqrt(float(np.dot(vd_c, vd_c) * np.dot(vr_c, vr_c))), 1e-12))

    res = {
        "kl_donor_to_recip": round(kl, 5),
        "pearson_r": round(r, 4),
        "top1_match": round(top1, 4),
        "entropy_donor": round(mean_attention_entropy(A_donor, min_pos=2), 4),
        "entropy_recip": round(mean_attention_entropy(A_recip, min_pos=2), 4),
    }
    if target_offsets:
        dm = [float(A_donor[:, q, k].mean().item()) for (q, k) in target_offsets]
        rm = [float(A_recip[:, q, k].mean().item()) for (q, k) in target_offsets]
        res["target_mass_donor"] = round(float(np.mean(dm)), 4)
        res["target_mass_recip"] = round(float(np.mean(rm)), 4)
    return res


def run_falsifier_a(args: argparse.Namespace) -> Dict[str, Any]:
    """Full MPS/GPU Falsifier A execution (Stage 1 Donor Head Census + Stage 2 Congruent Head Transfer)."""
    t0 = time.time()
    import pilot_organs_p0 as po
    import platonic_organ_engine as poe
    from extract_and_baseline import HF_CACHE_DIR, get_decoder_layers
    from transformers import AutoTokenizer

    # 0. Load all 3 fast tokenizers on CPU to construct exact shared 1-token word bank!
    toks = {
        k: AutoTokenizer.from_pretrained(str(HF_CACHE_DIR / k), local_files_only=True)
        for k in (DONOR_KEY,) + RECIPIENT_KEYS
    }
    shared_words = build_shared_single_token_words(toks)
    assert len(shared_words) >= 64, f"Too few shared single-token words across 3 tokenizers: {len(shared_words)}"
    lag_suite = build_lag_copy_sequences(shared_words, lags=LAGS_DELTA, n_prompts_per_lag=args.n_prompts_per_lag, seed=SEED)
    corpus = poe.build_universal_calibration_corpus()
    cal_fit_passages = corpus["fit"][:args.n_cal_fit]
    cal_val_passages = corpus["resid"][:args.n_cal_val]
    nat_eval_passages = corpus["eval"][:12]

    print(f"=== FALSIFIER A: RoPE TORUS-DISSOCIATION TEST (Prediction P1) ===")
    print(f"Shared single-token words across 3 tokenizers: {len(shared_words)} | Lags: {LAGS_DELTA}")

    # Pre-compute theoretical RoPE commutation curves for both recipients
    theory_curves: Dict[str, Any] = {}
    gA = VERIFIED_GEOMETRIES[DONOR_KEY]
    for rk in RECIPIENT_KEYS:
        gB = VERIFIED_GEOMETRIES[rk]
        J_opt, opt_info = build_optimal_rope_intertwiner(gA["d_h"], gA["rope_theta"], gB["d_h"], gB["rope_theta"])
        J_naive = build_naive_index_intertwiner(gA["d_h"], gB["d_h"])
        J_haar = build_haar_random_intertwiner(gA["d_h"], gB["d_h"], seed=SEED + 101)
        theory_curves[rk] = {
            "geometry_B": gB,
            "optimal_info": {k: v for k, v in opt_info.items() if k != "matched_pairs"},
            "commutation_err_OPTIMAL_ROPE_J": rope_commutation_error_curve(J_opt, gA["d_h"], gA["rope_theta"], gB["d_h"], gB["rope_theta"]),
            "commutation_err_NAIVE_INDEX_J": rope_commutation_error_curve(J_naive, gA["d_h"], gA["rope_theta"], gB["d_h"], gB["rope_theta"]),
            "commutation_err_HAAR_RANDOM_J": rope_commutation_error_curve(J_haar, gA["d_h"], gA["rope_theta"], gB["d_h"], gB["rope_theta"]),
        }
        print(
            f"  [Theory {DONOR_KEY} -> {rk}] max_rel_freq_err={opt_info['max_rel_freq_err']:.2e} | "
            f"comm_err(t=64): OPT={theory_curves[rk]['commutation_err_OPTIMAL_ROPE_J']['64']:.5f}, "
            f"NAIVE={theory_curves[rk]['commutation_err_NAIVE_INDEX_J']['64']:.4f}, "
            f"HAAR={theory_curves[rk]['commutation_err_HAAR_RANDOM_J']['64']:.4f}"
        )

    # ------------------------------------------------------------------
    # Stage 1: Donor Head Census on qwen2.5-3b (strictly fp32!)
    # ------------------------------------------------------------------
    print(f"\n--- Stage 1: Donor Head Census on {DONOR_KEY} (L={gA['L']}, H_q={gA['H_q']}, d_h={gA['d_h']}) ---")
    model_A, tok_A, dev_A, _ = po.load_model_fp32(DONOR_KEY)
    layers_A = get_decoder_layers(model_A)
    inv_freq_A = torch.from_numpy(rope_inv_freq(gA["d_h"], gA["rope_theta"]).astype(np.float32)).to(dev_A)

    # Encode natural text batch (truncated to 96 tokens) and repeated-sequence batches per lag
    nat_ids_A = [tok_A.encode(p, add_special_tokens=True, truncation=True, max_length=96) for p in nat_eval_passages]
    min_nat_len = min(len(x) for x in nat_ids_A)
    nat_batch_A = torch.tensor([x[:min_nat_len] for x in nat_ids_A], dtype=torch.long)

    lag_batches_A: Dict[int, torch.Tensor] = {}
    for d in LAGS_DELTA:
        rows = [encode_word_stream_exact(tok_A, it["words"])[0] for it in lag_suite[d]]
        lag_batches_A[d] = torch.tensor(rows, dtype=torch.long)

    # Census layers: skip L0 and final 2 layers (L1..L33)
    census_layers = list(range(1, gA["L"] - 2))
    x_nat_A = extract_input_ln_states(model_A, nat_batch_A, census_layers, dev_A)
    census_lags = (8, 16, 32)
    x_clag_A = {d: extract_input_ln_states(model_A, lag_batches_A[d][:4], census_layers, dev_A) for d in census_lags}

    census_records: List[Dict[str, Any]] = []
    for li in census_layers:
        attn_A = layers_A[li].self_attn
        xn = x_nat_A[li].to(dev_A)
        for hq in range(gA["H_q"]):
            hd = extract_donor_head_weights(attn_A, hq, gA["H_q"], gA["H_kv"])
            WQ = torch.from_numpy(hd["W_Q"].astype(np.float32)).to(dev_A)
            bQ = torch.from_numpy(hd["b_Q"].astype(np.float32)).to(dev_A)
            WK = torch.from_numpy(hd["W_K"].astype(np.float32)).to(dev_A)
            bK = torch.from_numpy(hd["b_K"].astype(np.float32)).to(dev_A)
            with torch.no_grad():
                An, _ = compute_single_head_attention(xn, WQ, bQ, WK, bK, None, None, None, inv_freq_A)
                T_n = An.shape[1]
                # Lag-1 previous-token mass on natural text
                pos_idx = torch.arange(2, T_n, device=dev_A)
                prev1_mass = float(An[:, pos_idx, pos_idx - 1].mean().item())
                nat_ent = mean_attention_entropy(An, min_pos=2)

                # Induction (lag P-1) & duplicate-token (lag P) mass across P in {8, 16, 32}
                ind_masses, dup_masses = [], []
                for P in census_lags:
                    xl = x_clag_A[P][li].to(dev_A)
                    Al, _ = compute_single_head_attention(xl, WQ, bQ, WK, bK, None, None, None, inv_freq_A)
                    # Second copy word indices P..2P-2 sit at token positions 1+P .. 2P-1
                    q_pos = torch.arange(1 + P, 2 * P, device=dev_A)
                    ind_masses.append(float(Al[:, q_pos, q_pos - P + 1].mean().item()))
                    dup_masses.append(float(Al[:, q_pos, q_pos - P].mean().item()))

            census_records.append({
                "layer": int(li),
                "head_q": int(hq),
                "prev1_mass_nat": round(prev1_mass, 4),
                "nat_entropy": round(nat_ent, 4),
                "induction_mass_mean": round(float(np.mean(ind_masses)), 4),
                "duplicate_mass_mean": round(float(np.mean(dup_masses)), 4),
            })

    # Select top sharp donor heads (2 previous-token + 2 induction + 1 duplicate-token, deduplicated)
    selected_heads: List[Dict[str, Any]] = []
    seen_lh = set()
    for role, key, count in [
        ("prev_token", "prev1_mass_nat", 2),
        ("induction", "induction_mass_mean", 2),
        ("duplicate", "duplicate_mass_mean", 1),
    ]:
        for rec in sorted(census_records, key=lambda r: r[key], reverse=True):
            lh = (rec["layer"], rec["head_q"])
            if lh not in seen_lh:
                seen_lh.add(lh)
                selected_heads.append({**rec, "role": role})
                if sum(1 for s in selected_heads if s["role"] == role) >= count:
                    break

    print("  Selected Donor Heads:")
    for s in selected_heads:
        print(
            f"    L{s['layer']:02d}.H{s['head_q']:02d} ({s['role']:10s}) | "
            f"prev1={s['prev1_mass_nat']:.3f} ind={s['induction_mass_mean']:.3f} dup={s['duplicate_mass_mean']:.3f} ent={s['nat_entropy']:.3f}"
        )

    sel_layers_A = sorted({s["layer"] for s in selected_heads})

    # Extract Donor A head weights, reference attention matrices, plane energy weights, and calibration port activations
    donor_heads_data: Dict[Tuple[int, int], Dict[str, Any]] = {}
    donor_attn_ref: Dict[Tuple[int, int], Dict[int, torch.Tensor]] = {}
    donor_cal_ent: Dict[Tuple[int, int], float] = {}

    x_lag_all_A = {d: extract_input_ln_states(model_A, lag_batches_A[d], sel_layers_A, dev_A) for d in LAGS_DELTA}
    cal_port_fit_A = extract_attn_port_passages(model_A, tok_A, dev_A, cal_fit_passages, sel_layers_A)
    cal_port_val_A = extract_attn_port_passages(model_A, tok_A, dev_A, cal_val_passages, sel_layers_A)

    # Also build calibration sequences for donor entropy target (first 8 val passages truncated to 64 tok)
    cal_ent_ids_A = [tok_A.encode(p, add_special_tokens=True, truncation=True, max_length=64) for p in cal_val_passages[:8]]
    m_ce_A = min(len(x) for x in cal_ent_ids_A)
    x_cal_ent_A = extract_input_ln_states(model_A, torch.tensor([x[:m_ce_A] for x in cal_ent_ids_A], dtype=torch.long), sel_layers_A, dev_A)

    for s in selected_heads:
        li, hq = s["layer"], s["head_q"]
        hd = extract_donor_head_weights(layers_A[li].self_attn, hq, gA["H_q"], gA["H_kv"])
        WQ = torch.from_numpy(hd["W_Q"].astype(np.float32)).to(dev_A)
        bQ = torch.from_numpy(hd["b_Q"].astype(np.float32)).to(dev_A)
        WK = torch.from_numpy(hd["W_K"].astype(np.float32)).to(dev_A)
        bK = torch.from_numpy(hd["b_K"].astype(np.float32)).to(dev_A)

        # Compute donor head's RoPE plane energy weights omega_i on unlabeled calibration text (strictly donor-side!)
        x_ce = x_cal_ent_A[li].to(dev_A)
        with torch.no_grad():
            q_ce = F.linear(x_ce, WQ, bQ)
            k_ce = F.linear(x_ce, WK, bK)
            mA = gA["d_h"] // 2
            q_mag = torch.sqrt(q_ce[..., :mA] ** 2 + q_ce[..., mA:] ** 2)
            k_mag = torch.sqrt(k_ce[..., :mA] ** 2 + k_ce[..., mA:] ** 2)
            hd["plane_energy"] = (q_mag * k_mag).mean(dim=(0, 1)).cpu().numpy().astype(np.float64)
            A_ce, _ = compute_single_head_attention(x_ce, WQ, bQ, WK, bK, None, None, None, inv_freq_A)
            donor_cal_ent[(li, hq)] = mean_attention_entropy(A_ce, min_pos=2)

            donor_attn_ref[(li, hq)] = {}
            for d in LAGS_DELTA:
                Ad, _ = compute_single_head_attention(x_lag_all_A[d][li].to(dev_A), WQ, bQ, WK, bK, None, None, None, inv_freq_A)
                donor_attn_ref[(li, hq)][d] = Ad.cpu()

        donor_heads_data[(li, hq)] = hd

    # UNLOAD Donor A completely before loading any Recipient B!
    po.unload_model(model_A)
    print(f"  Donor {DONOR_KEY} completely unloaded!")

    # ------------------------------------------------------------------
    # Stage 2: Transplant Into Recipient 1 (Falcon3-3B) & Recipient 2 (SmolLM2-1.7B)
    # ------------------------------------------------------------------
    recipients_out: Dict[str, Any] = {}
    for rk in RECIPIENT_KEYS:
        tr0 = time.time()
        gB = VERIFIED_GEOMETRIES[rk]
        print(f"\n--- Stage 2: SOLO Recipient {rk} (D={gB['D']}, L={gB['L']}, d_h={gB['d_h']}, theta={gB['rope_theta']}) ---")
        model_B, tok_B, dev_B, _ = po.load_model_fp32(rk)
        layers_B = get_decoder_layers(model_B)

        # Candidate B layers: proportional depth ± 2 around each donor head layer
        cand_layers_B = sorted({
            max(1, min(gB["L"] - 2, round(li * gB["L"] / float(gA["L"])) + off))
            for li in sel_layers_A
            for off in (-2, -1, 0, 1, 2)
        })
        cal_port_fit_B = extract_attn_port_passages(model_B, tok_B, dev_B, cal_fit_passages, cand_layers_B)
        cal_port_val_B = extract_attn_port_passages(model_B, tok_B, dev_B, cal_val_passages, cand_layers_B)

        # Match each donor layer lA to best recipient layer lB via unlabeled linear CKA on calibration fit!
        layer_matches: Dict[int, Dict[str, Any]] = {}
        sockets_by_lA: Dict[int, Dict[str, Any]] = {}
        for li in sel_layers_A:
            approx_lB = round(li * gB["L"] / float(gA["L"]))
            local_cands = [lb for lb in cand_layers_B if abs(lb - approx_lB) <= 2]
            best_lB, best_cka = local_cands[0], -1.0
            cka_scores = {}
            for lb in local_cands:
                pf = pair_attn_port_activations(cal_port_fit_A[li], cal_port_fit_B[lb])
                c = linear_cka(pf["x_A"][:1500], pf["x_B"][:1500])
                cka_scores[str(lb)] = round(c, 4)
                if c > best_cka:
                    best_cka, best_lB = c, lb

            pf = pair_attn_port_activations(cal_port_fit_A[li], cal_port_fit_B[best_lB])
            pv = pair_attn_port_activations(cal_port_val_A[li], cal_port_val_B[best_lB])
            read_sock = fit_affine_read_socket(pf["x_B"], pf["x_A"], pv["x_B"], pv["x_A"])
            write_sock = fit_cauchy_write_socket(pf["y_A"], pf["y_B"], pv["y_A"], pv["y_B"])
            layer_matches[li] = {
                "lA": int(li), "lB": int(best_lB), "cka_lin": round(best_cka, 4),
                "cka_window": cka_scores,
                "n_fit_pairs": int(len(pf["x_A"])), "n_val_pairs": int(len(pv["x_A"])),
                "heldout_read_R2": read_sock["heldout_read_R2"],
                "heldout_write_cos": write_sock["heldout_write_cos"],
                "write_s_out": round(write_sock["s_out"], 4),
            }
            sockets_by_lA[li] = {"lB": int(best_lB), "read": read_sock, "write": write_sock}
            print(
                f"  [Socket L{li:02d} -> L{best_lB:02d}] CKA={best_cka:.4f} | "
                f"read_R2={read_sock['heldout_read_R2']} | write_cos={write_sock['heldout_write_cos']}"
            )

        matched_layers_B = sorted({sockets_by_lA[li]["lB"] for li in sel_layers_A})

        # Encode lag copy sequences identically in Recipient B & capture unedited baseline logits + input_ln states!
        lag_batches_B: Dict[int, torch.Tensor] = {}
        base_logits_B: Dict[int, torch.Tensor] = {}
        x_lag_B: Dict[int, Dict[int, torch.Tensor]] = {}
        for d in LAGS_DELTA:
            rows = [encode_word_stream_exact(tok_B, it["words"])[0] for it in lag_suite[d]]
            bt = torch.tensor(rows, dtype=torch.long)
            lag_batches_B[d] = bt
            x_lag_B[d] = extract_input_ln_states(model_B, bt, matched_layers_B, dev_B)
            with torch.no_grad():
                base_logits_B[d] = model_B(input_ids=bt.to(dev_B), use_cache=False).logits.float().cpu()

        # Calibration batch for 1-D entropy temperature bisection
        cal_ent_ids_B = [tok_B.encode(p, add_special_tokens=True, truncation=True, max_length=64) for p in cal_val_passages[:8]]
        m_ce_B = min(len(x) for x in cal_ent_ids_B)
        x_cal_ent_B = extract_input_ln_states(model_B, torch.tensor([x[:m_ce_B] for x in cal_ent_ids_B], dtype=torch.long), matched_layers_B, dev_B)

        def _eval_causal_copy_shift(logits_graft: torch.Tensor, logits_base: torch.Tensor, batch_ids: torch.Tensor, lag: int) -> Dict[str, float]:
            """Measure downstream target-token log-prob shift on repeated-sequence copy prompts at lag `lag`."""
            lp_g = F.log_softmax(logits_graft, dim=-1)
            lp_b = F.log_softmax(logits_base, dim=-1)
            B_sz, T_len = batch_ids.shape
            if lag == 1:
                # At positions 1, 3, 5, ... (first copy of each pair), next token is an exact repeat (lag-1 copy!)
                pred_pos = list(range(1, T_len - 1, 2))
            else:
                # In second copy (positions 1 + lag .. T_len - 2), next token batch_ids[:, pos + 1] is the induction copy target!
                pred_pos = list(range(1 + lag, T_len - 1))
            d_lp_list, acc_base_l, acc_graft_l = [], [], []
            for pos in pred_pos:
                tgt_ids = batch_ids[:, pos + 1]
                g_val = lp_g[torch.arange(B_sz), pos, tgt_ids]
                b_val = lp_b[torch.arange(B_sz), pos, tgt_ids]
                d_lp_list.append(float((g_val - b_val).mean().item()))
                acc_base_l.append(float((logits_base[:, pos, :].argmax(dim=-1) == tgt_ids).float().mean().item()))
                acc_graft_l.append(float((logits_graft[:, pos, :].argmax(dim=-1) == tgt_ids).float().mean().item()))
            return {
                "delta_logp_copy_target": round(float(np.mean(d_lp_list)), 5),
                "acc_base": round(float(np.mean(acc_base_l)), 4),
                "acc_graft": round(float(np.mean(acc_graft_l)), 4),
                "delta_acc": round(float(np.mean(acc_graft_l) - np.mean(acc_base_l)), 4),
            }

        heads_out: List[Dict[str, Any]] = []
        for s in selected_heads:
            li, hq, role = s["layer"], s["head_q"], s["role"]
            hd = donor_heads_data[(li, hq)]
            sock = sockets_by_lA[li]
            lB = sock["lB"]
            M_in, d_in = sock["read"]["M_in"], sock["read"]["d_in"]
            M_out = sock["write"]["M_out"]

            # Build the 4 exact comparators for this donor head!
            J_opt, opt_meta = build_optimal_rope_intertwiner(
                gA["d_h"], gA["rope_theta"], gB["d_h"], gB["rope_theta"], plane_weights=hd["plane_energy"]
            )
            J_haar = build_haar_random_intertwiner(gA["d_h"], gB["d_h"], seed=SEED + 1000 * li + hq)
            J_naive = build_naive_index_intertwiner(gA["d_h"], gB["d_h"])

            comparators = [
                ("OPTIMAL_ROPE_J",    J_opt,   "recipient_rope"),
                ("HAAR_RANDOM_J",     J_haar,  "recipient_rope"),
                ("NAIVE_INDEX_J",     J_naive, "recipient_rope"),
                ("ROPE_STRIPPED_REF", None,    "donor_rope"),
            ]

            head_res: Dict[str, Any] = {
                "donor_layer": li,
                "donor_head_q": hq,
                "role": role,
                "recipient_layer": lB,
                "optimal_rope_meta": {k: v for k, v in opt_meta.items() if k != "matched_pairs"},
                "comparators": {},
            }
            print(f"  [Head L{li:02d}.H{hq:02d} ({role}) -> {rk} L{lB:02d}]")

            for cname, J_mat, rmode in comparators:
                op = compile_congruent_attention_head(
                    hd, M_in, d_in, M_out, J_mat, gB["d_h"], gA["rope_theta"], gB["rope_theta"], rope_mode=rmode, tau=1.0
                )
                tau_star = calibrate_entropy_temperature_bisection(
                    op, x_cal_ent_B[lB].to(dev_B), target_entropy=donor_cal_ent[(li, hq)]
                )
                op["tau"] = tau_star

                WQ = torch.from_numpy(op["W_Q"].astype(np.float32)).to(dev_B)
                bQ = torch.from_numpy(op["b_Q"].astype(np.float32)).to(dev_B)
                WK = torch.from_numpy(op["W_K"].astype(np.float32)).to(dev_B)
                bK = torch.from_numpy(op["b_K"].astype(np.float32)).to(dev_B)
                inv_f = torch.from_numpy(op["inv_freq"].astype(np.float32)).to(dev_B) if op["inv_freq"] is not None else None

                # Attach causal hook ONCE for all lags, then remove & verify exact bit restoration!
                graft = AdditiveHeadResidualGraft(layers_B[lB], op, dev_B, alpha_graft=args.alpha_graft).attach()
                by_lag: Dict[str, Any] = {}
                try:
                    for d in LAGS_DELTA:
                        xb = x_lag_B[d][lB].to(dev_B)
                        with torch.no_grad():
                            Ab, _ = compute_single_head_attention(xb, WQ, bQ, WK, bK, None, None, None, inv_f, tau=tau_star)
                            logits_g = model_B(input_ids=lag_batches_B[d].to(dev_B), use_cache=False).logits.float().cpu()

                        T_d = lag_batches_B[d].shape[1]
                        if d == 1:
                            tgt_pairs = [(p, p - 1) for p in range(2, T_d)]
                        elif role == "induction":
                            tgt_pairs = [(p, p - d + 1) for p in range(1 + d, T_d - 1)]
                        elif role == "duplicate":
                            tgt_pairs = [(p, p - d) for p in range(1 + d, T_d)]
                        else:
                            tgt_pairs = [(p, p - 1) for p in range(2, T_d)]

                        am = attention_metrics_vs_donor(donor_attn_ref[(li, hq)][d], Ab.cpu(), target_offsets=tgt_pairs)
                        cm = _eval_causal_copy_shift(logits_g, base_logits_B[d], lag_batches_B[d], d)
                        by_lag[str(d)] = {"attention": am, "causal_copy": cm}
                finally:
                    graft.remove()

                # Verify exact torch.equal bit-identical restoration on lag-1 probe
                with torch.no_grad():
                    check_logits = model_B(input_ids=lag_batches_B[1][:1].to(dev_B), use_cache=False).logits.float().cpu()
                bit_restored = bool(torch.equal(check_logits, base_logits_B[1][:1]))

                head_res["comparators"][cname] = {
                    "tau": round(tau_star, 4),
                    "bit_restored_after_unhook": bit_restored,
                    "by_lag": by_lag,
                }
                r_curve = " ".join(f"d{d}:{by_lag[str(d)]['attention']['pearson_r']:.2f}" for d in (1, 4, 16, 64))
                m_curve = " ".join(f"d{d}:{by_lag[str(d)]['attention']['target_mass_recip']:.2f}" for d in (1, 4, 16, 64))
                dlp_curve = " ".join(f"d{d}:{by_lag[str(d)]['causal_copy']['delta_logp_copy_target']:+.3f}" for d in (1, 4, 16, 64))
                print(f"    {cname:18s} tau={tau_star:5.2f} | r=[{r_curve}] | mass=[{m_curve}] | dLP=[{dlp_curve}] | unhook={bit_restored}")

            heads_out.append(head_res)

        recipients_out[rk] = {
            "geometry": gB,
            "layer_matches": {str(k): v for k, v in layer_matches.items()},
            "heads": heads_out,
            "elapsed_sec": round(time.time() - tr0, 1),
        }
        po.unload_model(model_B)

    out = {
        "meta": {
            "experiment": "v2_falsifierA_rope_torus",
            "donor_key": DONOR_KEY,
            "recipients": list(RECIPIENT_KEYS),
            "lags": list(LAGS_DELTA),
            "n_shared_single_token_words": len(shared_words),
            "n_prompts_per_lag": args.n_prompts_per_lag,
            "alpha_graft": args.alpha_graft,
            "seed": SEED,
            "total_elapsed_sec": round(time.time() - t0, 1),
        },
        "theory_commutation_curves": theory_curves,
        "donor_census_selected": selected_heads,
        "donor_census_top20_prev1": sorted(census_records, key=lambda r: r["prev1_mass_nat"], reverse=True)[:20],
        "donor_census_top20_induction": sorted(census_records, key=lambda r: r["induction_mass_mean"], reverse=True)[:20],
        "recipients": recipients_out,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[FALSIFIER A COMPLETE] Saved {OUT_PATH} ({out['meta']['total_elapsed_sec']}s)")
    return out


# ======================================================================================
# 6. Pure-CPU Unit Tests (<10s, ZERO GPU!)
# ======================================================================================

def run_cpu_unit_tests() -> None:
    t0 = time.time()
    print("=== Running v2_falsifierA_rope_torus.py Pure-CPU Unit Tests ===")

    # 1. Exact RoPE Torus-Commutation Curve (Prediction P1: Qwen -> Falcon3 vs Qwen -> SmolLM2!)
    gA = VERIFIED_GEOMETRIES["qwen2.5-3b"]
    gF = VERIFIED_GEOMETRIES["falcon3-3b"]
    gS = VERIFIED_GEOMETRIES["smollm2-1.7b"]

    J_opt_F, info_F = build_optimal_rope_intertwiner(gA["d_h"], gA["rope_theta"], gF["d_h"], gF["rope_theta"])
    J_naive_F = build_naive_index_intertwiner(gA["d_h"], gF["d_h"])
    J_haar_F = build_haar_random_intertwiner(gA["d_h"], gF["d_h"], seed=11)

    # Verify exact i -> 2i nesting for Qwen -> Falcon3!
    assert info_F["matched_pairs"] == [(i, 2 * i) for i in range(64)], f"Expected exact i->2i nesting, got {info_F['matched_pairs'][:8]}"
    assert info_F["max_rel_freq_err"] < 4.2e-5, f"Unexpected rel freq err: {info_F['max_rel_freq_err']}"
    assert np.allclose(J_opt_F.T @ J_opt_F, np.eye(gA["d_h"]), atol=1e-12)

    err_F_opt = rope_commutation_error_curve(J_opt_F, gA["d_h"], gA["rope_theta"], gF["d_h"], gF["rope_theta"])
    err_F_naive = rope_commutation_error_curve(J_naive_F, gA["d_h"], gA["rope_theta"], gF["d_h"], gF["rope_theta"])
    err_F_haar = rope_commutation_error_curve(J_haar_F, gA["d_h"], gA["rope_theta"], gF["d_h"], gF["rope_theta"])
    assert err_F_opt["64"] < 1e-3, f"Qwen->Falcon3 OPTIMAL_ROPE_J commutation error too large at t=64: {err_F_opt}"
    assert err_F_naive["16"] > 0.30, f"NAIVE_INDEX_J unexpectedly commuted on Falcon3: {err_F_naive}"
    assert err_F_haar["1"] > 0.25 and err_F_haar["16"] > 0.50, f"HAAR_RANDOM_J unexpectedly commuted: {err_F_haar}"

    J_opt_S, info_S = build_optimal_rope_intertwiner(gA["d_h"], gA["rope_theta"], gS["d_h"], gS["rope_theta"])
    err_S_opt = rope_commutation_error_curve(J_opt_S, gA["d_h"], gA["rope_theta"], gS["d_h"], gS["rope_theta"])
    assert err_S_opt["64"] > 100.0 * err_F_opt["64"], f"Expected P1 dissociation (>100x larger error on SmolLM2): F={err_F_opt['64']} vs S={err_S_opt['64']}"
    print(
        f"  [PASS 1/6] Prediction P1 RoPE commutation verified! Qwen->Falcon3 i->2i max_rel_err={info_F['max_rel_freq_err']:.2e}, "
        f"comm_err(t=64): Falcon3_OPT={err_F_opt['64']:.6f} vs SmolLM2_OPT={err_S_opt['64']:.4f} "
        f"({err_S_opt['64']/max(err_F_opt['64'],1e-12):.0f}x!), Falcon3_NAIVE={err_F_naive['64']:.4f}, HAAR={err_F_haar['64']:.4f}"
    )

    # 2. Exact HF `rotate_half` / `apply_rotary_pos_emb` Matrix Equivalence (Qwen2 & Llama)!
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as qwen_apply_rope
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_apply_rope

    rng = np.random.default_rng(2026)
    for nm, apply_fn, d_h, theta in [("Qwen2", qwen_apply_rope, 128, 1e6), ("Llama/Falcon3", llama_apply_rope, 256, 1000042.0)]:
        T = 17
        x_np = rng.standard_normal((1, 1, T, d_h)).astype(np.float32)
        x_t = torch.from_numpy(x_np)
        inv_f = torch.from_numpy(rope_inv_freq(d_h, theta).astype(np.float32))
        pos = torch.arange(T, dtype=torch.float32)
        freqs = pos.unsqueeze(-1) * inv_f
        emb = torch.cat([freqs, freqs], dim=-1).unsqueeze(0)  # (1, T, d_h)
        q_hf, _ = apply_fn(x_t, x_t, emb.cos(), emb.sin(), unsqueeze_dim=1)
        q_ours = apply_rope_half_torch(x_t[0, 0], inv_f, pos)
        assert float(torch.max(torch.abs(q_hf[0, 0] - q_ours)).item()) < 1e-6, f"{nm} apply_rope_half_torch mismatch!"
        for t_check in (0, 1, 7, 16):
            R_t = rope_matrix(d_h, theta, t_check)
            diff_mat = float(np.max(np.abs((R_t @ x_np[0, 0, t_check].astype(np.float64)) - q_hf[0, 0, t_check].numpy().astype(np.float64))))
            assert diff_mat < 1e-5, f"{nm} rope_matrix mismatch at t={t_check}: {diff_mat}"
    print("  [PASS 2/6] Exact HF Qwen2 & Llama `rotate_half` / `apply_rotary_pos_emb` equivalence (<1e-5) verified!")

    # 3. Exact Congruent Attention-Head Forward Equivalence on Real HF Qwen2Attention -> LlamaAttention!
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    torch.manual_seed(7)
    D_A, D_B = 64, 96
    d_hA, d_hB = 16, 32
    q_cfg = Qwen2Config(
        hidden_size=D_A, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, rope_theta=1_000_000.0,
    )
    attn_A = Qwen2Attention(q_cfg, layer_idx=0).eval()
    # Scale Q/K weights to realistic sharp pretrained magnitude & non-zero biases (verifies Qwen2 q/k/v bias transport!)
    with torch.no_grad():
        attn_A.q_proj.weight.mul_(12.0)
        attn_A.k_proj.weight.mul_(12.0)
        attn_A.q_proj.bias.uniform_(-1.5, 1.5)
        attn_A.k_proj.bias.uniform_(-1.5, 1.5)
        attn_A.v_proj.bias.uniform_(-0.3, 0.3)

    hd = extract_donor_head_weights(attn_A, head_q=1, num_heads=4, num_kv_heads=2)
    # Exact isometric ground-truth residual embedding A -> B (so M_in = U_in^T recovers x_A exactly!)
    U_in, _ = sla.qr(rng.standard_normal((D_B, D_A)), mode="economic")
    U_out, _ = sla.qr(rng.standard_normal((D_B, D_A)), mode="economic")
    shift_A = rng.standard_normal(D_A) * 0.2
    shift_B = rng.standard_normal(D_B) * 0.2
    M_in = U_in.T
    d_in = shift_A - M_in @ shift_B
    M_out = U_out

    xA_seq = rng.standard_normal((2, 65, D_A)) * 0.6 + shift_A
    xB_seq = (xA_seq - shift_A) @ U_in.T + shift_B
    xA_t = torch.from_numpy(xA_seq.astype(np.float32))
    xB_t = torch.from_numpy(xB_seq.astype(np.float32))

    # Donor reference attention & output
    inv_fA = torch.from_numpy(rope_inv_freq(d_hA, 1_000_000.0).astype(np.float32))
    with torch.no_grad():
        A_ref, y_ref = compute_single_head_attention(
            xA_t,
            torch.from_numpy(hd["W_Q"].astype(np.float32)), torch.from_numpy(hd["b_Q"].astype(np.float32)),
            torch.from_numpy(hd["W_K"].astype(np.float32)), torch.from_numpy(hd["b_K"].astype(np.float32)),
            torch.from_numpy(hd["W_V"].astype(np.float32)), torch.from_numpy(hd["b_V"].astype(np.float32)),
            torch.from_numpy(hd["W_O"].astype(np.float32)),
            inv_fA, tau=1.0,
        )

    J_toy_opt, _ = build_optimal_rope_intertwiner(d_hA, 1_000_000.0, d_hB, 1_000_042.0)
    op_opt = compile_congruent_attention_head(hd, M_in, d_in, M_out, J_toy_opt, d_hB, 1_000_000.0, 1_000_042.0, "recipient_rope", tau=1.0)
    with torch.no_grad():
        A_opt, y_opt = compute_single_head_attention(
            xB_t,
            torch.from_numpy(op_opt["W_Q"].astype(np.float32)), torch.from_numpy(op_opt["b_Q"].astype(np.float32)),
            torch.from_numpy(op_opt["W_K"].astype(np.float32)), torch.from_numpy(op_opt["b_K"].astype(np.float32)),
            torch.from_numpy(op_opt["W_V"].astype(np.float32)), torch.from_numpy(op_opt["b_V"].astype(np.float32)),
            torch.from_numpy(op_opt["W_O"].astype(np.float32)),
            torch.from_numpy(op_opt["inv_freq"].astype(np.float32)), tau=1.0,
        )
    max_A_diff = float(torch.max(torch.abs(A_opt - A_ref)).item())
    y_ref_mapped = torch.matmul(y_ref, torch.from_numpy(M_out.T.astype(np.float32)))
    max_y_diff = float(torch.max(torch.abs(y_opt - y_ref_mapped)).item())
    assert max_A_diff < 1e-3, f"OPTIMAL_ROPE_J failed synthetic forward equivalence: max_A_diff={max_A_diff}"
    assert max_y_diff < 1e-3, f"OPTIMAL_ROPE_J output residual mismatch: max_y_diff={max_y_diff}"

    # Verify HAAR_RANDOM_J fails on the exact same toy pair!
    J_toy_haar = build_haar_random_intertwiner(d_hA, d_hB, seed=99)
    op_haar = compile_congruent_attention_head(hd, M_in, d_in, M_out, J_toy_haar, d_hB, 1_000_000.0, 1_000_042.0, "recipient_rope", tau=1.0)
    with torch.no_grad():
        A_haar, _ = compute_single_head_attention(
            xB_t,
            torch.from_numpy(op_haar["W_Q"].astype(np.float32)), torch.from_numpy(op_haar["b_Q"].astype(np.float32)),
            torch.from_numpy(op_haar["W_K"].astype(np.float32)), torch.from_numpy(op_haar["b_K"].astype(np.float32)),
            None, None, None, torch.from_numpy(op_haar["inv_freq"].astype(np.float32)), tau=1.0,
        )
    m_opt = attention_metrics_vs_donor(A_ref, A_opt)
    m_haar = attention_metrics_vs_donor(A_ref, A_haar)
    assert m_opt["pearson_r"] > 0.9999 and m_haar["pearson_r"] < m_opt["pearson_r"] - 0.05
    print(
        f"  [PASS 3/6] Real HF Qwen2Attention (with q/k/v bias!) -> nested d_h=32 congruent forward equivalence verified "
        f"(max|dA|={max_A_diff:.2e}, max|dy|={max_y_diff:.2e}, OPT r={m_opt['pearson_r']:.4f} vs HAAR r={m_haar['pearson_r']:.4f})!"
    )

    # 4. 1-D Entropy Temperature Bisection Accuracy
    target_ent = 0.85 * mean_attention_entropy(A_ref, min_pos=2)
    tau_found = calibrate_entropy_temperature_bisection(op_opt, xB_t, target_entropy=target_ent)
    with torch.no_grad():
        A_cal, _ = compute_single_head_attention(
            xB_t,
            torch.from_numpy(op_opt["W_Q"].astype(np.float32)), torch.from_numpy(op_opt["b_Q"].astype(np.float32)),
            torch.from_numpy(op_opt["W_K"].astype(np.float32)), torch.from_numpy(op_opt["b_K"].astype(np.float32)),
            None, None, None, torch.from_numpy(op_opt["inv_freq"].astype(np.float32)), tau=tau_found,
        )
    ent_found = mean_attention_entropy(A_cal, min_pos=2)
    assert abs(ent_found - target_ent) < 1e-3, f"Bisection entropy mismatch: {ent_found} vs {target_ent}"
    print(f"  [PASS 4/6] 1-D entropy bisection verified (tau={tau_found:.4f}, target={target_ent:.4f}, got={ent_found:.4f})!")

    # 5. Verified Shared Single-Token Word Bank Across All 3 Cached Tokenizers (CPU-Only!)
    from extract_and_baseline import HF_CACHE_DIR
    from transformers import AutoTokenizer

    if all((HF_CACHE_DIR / k).is_dir() for k in (DONOR_KEY,) + RECIPIENT_KEYS):
        toks = {k: AutoTokenizer.from_pretrained(str(HF_CACHE_DIR / k), local_files_only=True) for k in (DONOR_KEY,) + RECIPIENT_KEYS}
        sw = build_shared_single_token_words(toks)
        assert len(sw) >= 120, f"Expected >=120 shared 1-token words across all 3 tokenizers, got {len(sw)}"
        suite = build_lag_copy_sequences(sw, lags=LAGS_DELTA, n_prompts_per_lag=2, seed=SEED)
        for d in LAGS_DELTA:
            it = suite[d][0]
            lens = {k: len(encode_word_stream_exact(toks[k], it["words"])[0]) for k in toks}
            assert len(set(lens.values())) == 1, f"Sequence length mismatch at lag {d}: {lens}"
            # Verify exact token repeat at lag d!
            for k in toks:
                ids, off = encode_word_stream_exact(toks[k], it["words"])
                if d == 1:
                    assert ids[off + 1] == ids[off]
                else:
                    assert ids[off + d] == ids[off] and ids[off + 2 * d - 1] == ids[off + d - 1]
        print(f"  [PASS 5/6] Shared 1-token bank ({len(sw)} words) & exact lag-Delta={{1..64}} alignment verified across 3/3 cached tokenizers!")
    else:
        print("  [PASS 5/6] (Skipped cached tokenizer check — HF cache dir not present)")

    # 6. Causal Residual Graft Hook on Real HF LlamaDecoderLayer + Exact torch.equal Unhook Restoration!
    l_cfg = LlamaConfig(
        hidden_size=D_B, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=3, num_key_value_heads=1, head_dim=d_hB,
        rope_theta=1_000_042.0, mlp_bias=False, attention_bias=False,
    )
    layer_B = LlamaDecoderLayer(l_cfg, layer_idx=0).eval()
    T_h = 12
    h_in = torch.randn(2, T_h, D_B, dtype=torch.float32)
    inv_fB = torch.from_numpy(rope_inv_freq(d_hB, 1_000_042.0).astype(np.float32))
    pos_h = torch.arange(T_h, dtype=torch.float32)
    emb_h = torch.cat([pos_h.unsqueeze(-1) * inv_fB, pos_h.unsqueeze(-1) * inv_fB], dim=-1).unsqueeze(0)
    pe_h = (emb_h.cos(), emb_h.sin())

    with torch.no_grad():
        out_base = layer_B(h_in, position_embeddings=pe_h)
    graft = AdditiveHeadResidualGraft(layer_B, op_opt, device=torch.device("cpu"), alpha_graft=1.0).attach()
    with torch.no_grad():
        out_graft = layer_B(h_in, position_embeddings=pe_h)
    graft.remove()
    with torch.no_grad():
        out_restored = layer_B(h_in, position_embeddings=pe_h)

    shift_norm = float(torch.norm(out_graft - out_base).item())
    assert shift_norm > 1e-4, f"Graft hook produced zero shift: {shift_norm}"
    assert torch.equal(out_restored, out_base), "Unhooked LlamaDecoderLayer NOT bit-identical!"
    print(f"  [PASS 6/6] Real HF LlamaDecoderLayer causal head graft (shift={shift_norm:.4f}) & exact torch.equal unhook restoration verified!")
    print(f"=== ALL 6 FALSIFIER A CPU UNIT TESTS PASSED ({time.time() - t0:.2f}s)! ===")


def main() -> None:
    ap = argparse.ArgumentParser(description="Falsifier A: RoPE Torus-Dissociation Test (Prediction P1)")
    ap.add_argument("--unit-test", action="store_true", help="Run fast pure-CPU unit tests (ZERO GPU)")
    ap.add_argument("--run", action="store_true", help="Run full Stage 1 + Stage 2 MPS experiment")
    ap.add_argument("--n-prompts-per-lag", type=int, default=8, help="Repeated-sequence prompts per lag Delta")
    ap.add_argument("--n-cal-fit", type=int, default=90, help="Unlabeled calibration fit passages for M_in/M_out")
    ap.add_argument("--n-cal-val", type=int, default=30, help="Unlabeled calibration held-out passages for socket audit")
    ap.add_argument("--alpha-graft", type=float, default=1.0, help="Causal additive residual head graft scale")
    args = ap.parse_args()

    if args.unit_test:
        run_cpu_unit_tests()
        return
    if args.run:
        run_falsifier_a(args)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
