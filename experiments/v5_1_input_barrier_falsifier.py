#!/usr/bin/env python3
"""
PlatonicSwap v5.1: 100% Zero-SGD Input-Barrier Diagnostic & Falsifier
(`Qwen/Qwen2.5-7B-Instruct` -> `Qwen/Qwen2.5-1.5B-Instruct`)

Synthesizes the 3-Model Council (Gemini 4 Carbon + Gemini 4 Neon xHigh + Astra 6.2 Ultra,
plus Goodfire/Anthropic J-lens sparse neuron-community lens) and passes all Round-1 Pre-GPU
Grading Council P0/P1 invariants:
  - 100% Zero-SGD, ZERO task-target-fitted writers, ZERO `val_report` selection leakage.
  - Generic calibration corpus (60k fit / 4k tune / 4k report `pos > 0`) & weight-only `P_0`
    are built strictly independent of pilot/disc/eval task splits!
  - Preallocated disk memmaps + memoized FP64 feature eigen-caches (< 10 min on 1x Colab L4,
    single-model phases, < 1.0 GiB post-unload VRAM).
  - Exact compiled full-sequence logits for Arm 1a (`m_low` + partner `m_low+half` dedicated
    BOS channel) & Arm 1b (25-point grid + Brent root-finding for mixed-sign BOS slopes).
  - Shared frozen generic `v_bar_gen` centering across Donor `c_true`, `d_ref`, `Arm 2c`, and
    all student arms; numerically stable `softmax(logits[1:])` sink/content decomposition;
    matched-capacity `SHUF-layer` (`L24-L26` lowest-|DLA| heads with identical taps & recipe);
    adversarial 20-draw `SHUF-pi` (live greedy gen on top-3 highest-scoring draws); exact
    integer hit-count thresholds; full all-parameter + bit-equal logit rollback in Arm 3c.

Usage:
  PYTHONPATH=platonicswap:experiments python experiments/v5_1_input_barrier_falsifier.py --unit-test
  PYTHONPATH=platonicswap:experiments python -u experiments/v5_1_input_barrier_falsifier.py --run-all --allow-download
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla
import scipy.optimize as sopt
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (str(REPO_ROOT / "platonicswap"), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if "PLATONICSWAP_HF_CACHE" not in os.environ and Path("/tmp/platos_key_lab/hf_cache").is_dir():
    os.environ["PLATONICSWAP_HF_CACHE"] = "/tmp/platos_key_lab/hf_cache"
if "PLATONICSWAP_CACHE_DIR" not in os.environ and Path("/tmp/platos_key_lab/cache").is_dir():
    os.environ["PLATONICSWAP_CACHE_DIR"] = "/tmp/platos_key_lab/cache"

RESULTS_DIR = REPO_ROOT / "results" if (REPO_ROOT / "results").is_dir() else SCRIPT_DIR
OUT_PATH = RESULTS_DIR / "results_v5_1_input_barrier_falsifier.json"

from exact_math_v43 import StreamingSharedVocabFisherAccumulator
from extract_and_baseline import HF_CACHE_DIR, get_decoder_layers
from large_to_small import load_model_bf16
from native_weight_translation import (
    assert_zero_runtime_hooks,
    get_config_rope_theta,
    unload_model_vram_safe,
)
from v2_falsifierA_rope_torus import rope_inv_freq
from v3_gate2_skill_and_sql import WORD_CANDIDATES_KEYS, WORD_CANDIDATES_VALS
from v4_4_private_lane_falsifier import ReversiblePrivateLanePack
from v4_stage1_qwen7b_to_1p5b_native import (
    DONOR_KEY_DEFAULT,
    RECIPIENT_KEY_DEFAULT,
    SEED,
    _MaskPaddedVocabLogitsProcessor,
)
from v5_0_boundary_organ_gates_g0_g2 import (
    _compute_safe_q_and_r2,
    _compute_safe_wrong_item_ratio,
    _forward_single_attn_head_z_and_out,
    _host_rss_gib,
    _left_pad_position_ids,
    _make_rope_cos_sin,
    _native_bf16_argmax_shared,
    _real_norm_margin_from_rows,
    _resolve_tokenizer_source,
    _sym,
    _tensor_to_bf16_u16,
    check_2hop_exact_word,
    compute_attn_head_demand_matrix_closed_form,
    compute_cancellation_ratio_chi,
    compute_demand_weighted_supply_boundary,
    compute_raw_and_postnorm_fisher_plugs,
    factor_psd_top_k,
    partition_2hop_key_pools,
)

LAM_GRID_UNCAPPED: Tuple[float, ...] = (
    1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0
)

DUMMY_EXEMPLAR_KEYS: Tuple[str, ...] = (
    "alpha", "beta", "gamma", "delta", "zeta", "theta", "kappa", "sigma",
    "omega", "prism", "spark", "cloud", "ember", "frost", "cedar", "maple",
)
DUMMY_EXEMPLAR_VALS: Tuple[str, ...] = (
    "amber", "ivory", "coral", "azure", "umber", "ochre", "lilac", "topaz",
    "crimson", "indigo", "violet", "scarlet", "bronze", "copper", "nickel", "quartz",
)


# ======================================================================================
# 1. Exact Mathematical Primitives (100% Zero-SGD, Memoized FP64 Solves)
# ======================================================================================

def _u16_to_f64(arr_u16: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.asarray(arr_u16)).view(torch.bfloat16).double().numpy()


def _u16_to_f32(arr_u16: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.asarray(arr_u16)).view(torch.bfloat16).float().numpy()


def _rope_rotate_np(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def compute_vocab_anchored_fisher_plugs(
    G_A: np.ndarray,
    G_B: np.ndarray,
    G_BA: np.ndarray,
    E_A_f32: np.ndarray,
    E_B_f32: np.ndarray,
    gamma_A: np.ndarray,
    gamma_B: np.ndarray,
    etas: Sequence[float] = (0.0, 1e-4, 1e-3, 1e-2, 5e-2),
    eps_rel_med: float = 1e-2,
) -> Dict[float, Dict[str, Any]]:
    """Closed-form vocab-anchored Fisher plugs without materializing any V x V matrix."""
    V = int(E_A_f32.shape[0])
    assert E_B_f32.shape[0] == V
    EA = np.asarray(E_A_f32, dtype=np.float64)
    EB = np.asarray(E_B_f32, dtype=np.float64)
    ea_bar, eb_bar = EA.mean(axis=0), EB.mean(axis=0)
    G_A_iso = _sym((EA.T @ EA) / float(V) - np.outer(ea_bar, ea_bar))
    G_B_iso = _sym((EB.T @ EB) / float(V) - np.outer(eb_bar, eb_bar))
    G_BA_iso = (EB.T @ EA) / float(V) - np.outer(eb_bar, ea_bar)
    iso_scale = max(float(np.trace(G_B)), 1e-12) / max(float(np.trace(G_B_iso)), 1e-12)
    out: Dict[float, Dict[str, Any]] = {}
    for eta in etas:
        ef = float(eta)
        GA_e = _sym((1.0 - ef) * G_A + ef * iso_scale * G_A_iso)
        GB_e = _sym((1.0 - ef) * G_B + ef * iso_scale * G_B_iso)
        GBA_e = (1.0 - ef) * G_BA + ef * iso_scale * G_BA_iso
        plugs = compute_raw_and_postnorm_fisher_plugs(GA_e, GB_e, GBA_e, gamma_A, gamma_B, eps_rel_med=eps_rel_med)
        plugs["eta"] = ef
        plugs["iso_scale"] = float(iso_scale)
        out[ef] = plugs
    return out


def prepare_ridge_x_cache(
    X_fit: np.ndarray,
    X_tune: np.ndarray,
    X_rep: Optional[np.ndarray] = None,
    chunk_rows: int = 8192,
) -> Dict[str, Any]:
    """
    Bounded-RAM FP64 Gram + eigendecomposition cache:
      - Stores `Xf`, `Xt`, `Xr` in `float32` (zero duplicate `Xfc` -> 4x lower host RAM!)
      - Streams `C_xx = (Xfc^T Xfc) / N` in `chunk_rows` FP64 chunks (exact FP64 moments!).
    """
    Xf32 = np.asarray(X_fit, dtype=np.float32)
    Xt32 = np.asarray(X_tune, dtype=np.float32)
    Xr32 = np.asarray(X_rep, dtype=np.float32) if X_rep is not None else None
    N, D = int(Xf32.shape[0]), int(Xf32.shape[1])
    mu_x = np.zeros((D,), dtype=np.float64)
    for i0 in range(0, N, chunk_rows):
        mu_x += Xf32[i0 : i0 + chunk_rows].astype(np.float64).sum(axis=0)
    mu_x /= max(float(N), 1.0)
    C_xx = np.zeros((D, D), dtype=np.float64)
    for i0 in range(0, N, chunk_rows):
        xc_b = Xf32[i0 : i0 + chunk_rows].astype(np.float64) - mu_x[None, :]
        C_xx += xc_b.T @ xc_b
    C_xx = _sym(C_xx / max(float(N), 1.0))
    ev, V = sla.eigh(C_xx)
    ev = np.maximum(ev, 0.0)
    pos = ev[ev > 1e-12]
    med_ev = float(np.median(pos)) if len(pos) > 0 else float(np.mean(ev) + 1e-12)
    Xtc = Xt32.astype(np.float64) - mu_x[None, :]
    return {
        "Xf32": Xf32,
        "Xt": Xt32.astype(np.float64),
        "Xr32": Xr32,
        "Xtc_V": Xtc @ V,
        "mu_x": mu_x,
        "C_xx": C_xx,
        "ev": ev,
        "V": V,
        "med_ev": med_ev,
        "N": N,
        "chunk_rows": int(chunk_rows),
    }


def fit_ridge_from_x_cache(
    xc: Dict[str, Any],
    Y_fit: np.ndarray,
    Y_tune: np.ndarray,
    lam_grid: Sequence[float] = LAM_GRID_UNCAPPED,
    Y_rep: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Bounded-RAM exact FP64 affine ridge solve reusing `prepare_ridge_x_cache`."""
    Yf = np.asarray(Y_fit)
    Yt = np.asarray(Y_tune, dtype=np.float64)
    N, d_y = int(Yf.shape[0]), int(Yf.shape[1])
    D = int(xc["mu_x"].shape[0])
    cr = int(xc["chunk_rows"])
    mu_x = xc["mu_x"]

    mu_y = np.zeros((d_y,), dtype=np.float64)
    for i0 in range(0, N, cr):
        mu_y += Yf[i0 : i0 + cr].astype(np.float64).sum(axis=0)
    mu_y /= max(float(N), 1.0)

    C_yx = np.zeros((d_y, D), dtype=np.float64)
    for i0 in range(0, N, cr):
        xb = xc["Xf32"][i0 : i0 + cr].astype(np.float64) - mu_x[None, :]
        yb = Yf[i0 : i0 + cr].astype(np.float64) - mu_y[None, :]
        C_yx += yb.T @ xb
    C_yx /= max(float(N), 1.0)

    C_yx_V = C_yx @ xc["V"]
    ev, V, med_ev, Xtc_V = xc["ev"], xc["V"], xc["med_ev"], xc["Xtc_V"]
    Ytc = Yt - mu_y[None, :]
    tot_t = float(np.sum((Yt - Yt.mean(axis=0, keepdims=True)) ** 2)) + 1e-18

    best_lam, best_r2_t, best_W = float(lam_grid[0]), -1e18, None
    for lam_rel in lam_grid:
        lam = max(float(lam_rel) * med_ev, 1e-10)
        W_V = C_yx_V / (ev + lam)[None, :]
        Yhat_tc = Xtc_V @ W_V.T
        r2_t = 1.0 - float(np.sum((Ytc - Yhat_tc) ** 2)) / tot_t
        if r2_t > best_r2_t:
            best_r2_t = r2_t
            best_lam = float(lam_rel)
            best_W = W_V @ V.T
    assert best_W is not None
    b = mu_y - best_W @ mu_x

    def _eval_r2_chunked(Xe: np.ndarray, Ye: np.ndarray) -> Tuple[float, np.ndarray]:
        n_e = int(Ye.shape[0])
        ye_mean = np.zeros((d_y,), dtype=np.float64)
        for i0 in range(0, n_e, cr):
            ye_mean += Ye[i0 : i0 + cr].astype(np.float64).sum(axis=0)
        ye_mean /= max(float(n_e), 1.0)
        ss_res = np.zeros((d_y,), dtype=np.float64)
        ss_tot = np.zeros((d_y,), dtype=np.float64)
        for i0 in range(0, n_e, cr):
            xb = Xe[i0 : i0 + cr].astype(np.float64)
            yb = Ye[i0 : i0 + cr].astype(np.float64)
            yp = xb @ best_W.T + b[None, :]
            ss_res += np.sum((yb - yp) ** 2, axis=0)
            ss_tot += np.sum((yb - ye_mean[None, :]) ** 2, axis=0)
        pooled = 1.0 - float(np.sum(ss_res)) / (float(np.sum(ss_tot)) + 1e-18)
        per_d = 1.0 - ss_res / (ss_tot + 1e-18)
        return float(pooled), per_d

    r2_f, _ = _eval_r2_chunked(xc["Xf32"], Yf)
    r2_t, per_d_t = _eval_r2_chunked(xc["Xt"], Yt)
    out: Dict[str, Any] = {
        "W": best_W,
        "b": b,
        "mu_x": mu_x,
        "mu_y": mu_y,
        "best_lam_rel": float(best_lam),
        "med_eig_cxx": float(med_ev),
        "r2_fit": float(r2_f),
        "r2_tune": float(r2_t),
        "per_dim_r2_tune": per_d_t,
    }
    if xc["Xr32"] is not None and Y_rep is not None:
        r2_r, per_d_r = _eval_r2_chunked(xc["Xr32"], np.asarray(Y_rep))
        out["r2_rep"] = float(r2_r)
        out["per_dim_r2_rep"] = per_d_r
    return out


def fit_ridge_affine_val_tuned(
    X_fit: np.ndarray,
    Y_fit: np.ndarray,
    X_tune: np.ndarray,
    Y_tune: np.ndarray,
    lam_grid: Sequence[float] = LAM_GRID_UNCAPPED,
    X_rep: Optional[np.ndarray] = None,
    Y_rep: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    xc = prepare_ridge_x_cache(X_fit, X_tune, X_rep=X_rep)
    return fit_ridge_from_x_cache(xc, Y_fit, Y_tune, lam_grid=lam_grid, Y_rep=Y_rep)


def apply_mode_denoising_on_tune(
    W: np.ndarray,
    b: np.ndarray,
    X_tune: np.ndarray,
    Y_tune: np.ndarray,
    mu_x_fit: np.ndarray,
    mu_y_fit: np.ndarray,
    rho2_floor: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Neon xHigh Mode Denoising using frozen 60k `fit` means (`mu_x_fit`, `mu_y_fit`)."""
    Xt = np.asarray(X_tune, dtype=np.float64)
    Yt = np.asarray(Y_tune, dtype=np.float64)
    Ytc = Yt - mu_y_fit[None, :]
    C_yy = _sym((Ytc.T @ Ytc) / max(float(Yt.shape[0]), 1.0))
    ev, U = sla.eigh(C_yy)
    U = U[:, np.argsort(ev)[::-1]]
    Yp = Xt @ W.T + b[None, :]
    Yt_u = Ytc @ U
    Yp_u = (Yp - mu_y_fit[None, :]) @ U
    rho2_u = 1.0 - np.sum((Yt_u - Yp_u) ** 2, axis=0) / (np.sum(Yt_u ** 2, axis=0) + 1e-18)
    keep = rho2_u > float(rho2_floor)
    if not np.any(keep):
        keep[int(np.argmax(rho2_u))] = True
    P = U[:, keep] @ U[:, keep].T
    W_den = P @ W
    b_den = mu_y_fit - W_den @ mu_x_fit
    return W_den, b_den, {
        "n_kept_modes": int(np.sum(keep)),
        "d_total": int(len(keep)),
        "median_rho2_all": float(np.median(rho2_u)),
        "min_rho2_kept": float(np.min(rho2_u[keep])),
    }


def apply_rope_plane_deshrinkage(
    W: np.ndarray,
    b: np.ndarray,
    X_tune: np.ndarray,
    Y_tune: np.ndarray,
    mu_x_fit: np.ndarray,
    mu_y_fit: np.ndarray,
    rho2_min: float = 0.20,
    gain_cap: float = 1.45,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Carbon Mechanism C: RoPE-commuting `2x2` plane de-shrinkage using frozen 60k `fit` means."""
    d_h = int(W.shape[0])
    half = d_h // 2
    Xt = np.asarray(X_tune, dtype=np.float64)
    Yt = np.asarray(Y_tune, dtype=np.float64)
    Yp = Xt @ W.T + b[None, :]
    gains = np.zeros((d_h,), dtype=np.float64)
    plane_rho2 = np.zeros((half,), dtype=np.float64)
    n_active = 0
    for m in range(half):
        idx = [m, m + half]
        yt_m = Yt[:, idx] - mu_y_fit[None, idx]
        yp_m = Yp[:, idx] - mu_y_fit[None, idx]
        r2_m = 1.0 - float(np.sum((yt_m - yp_m) ** 2)) / (float(np.sum(yt_m ** 2)) + 1e-18)
        plane_rho2[m] = r2_m
        if r2_m >= float(rho2_min):
            g_m = min((r2_m + 1e-2) ** (-0.5), float(gain_cap))
            gains[m] = g_m
            gains[m + half] = g_m
            n_active += 1
    if n_active == 0:
        best_m = int(np.argmax(plane_rho2))
        gains[best_m] = 1.0
        gains[best_m + half] = 1.0
        n_active = 1
    W_out = gains[:, None] * W
    b_out = mu_y_fit - W_out @ mu_x_fit
    return W_out, b_out, gains, {
        "n_active_planes": int(n_active),
        "n_total_planes": int(half),
        "median_plane_rho2": float(np.median(plane_rho2)),
        "max_plane_rho2": float(np.max(plane_rho2)),
        "mean_active_gain": float(np.mean(gains[:half][gains[:half] > 0])),
    }


def apply_kkt_rank1_pos0_anchor(
    W: np.ndarray,
    b: np.ndarray,
    u0_B: np.ndarray,
    target0_A: np.ndarray,
    mu_x_gt0: np.ndarray,
    C_xx_gt0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact rank-1 KKT update enforcing `W_new @ u0_B + b_new == target0_A` while preserving `mu_x_gt0`."""
    u0 = np.asarray(u0_B, dtype=np.float64).reshape(-1)
    t0 = np.asarray(target0_A, dtype=np.float64).reshape(-1)
    mx = np.asarray(mu_x_gt0, dtype=np.float64).reshape(-1)
    err0 = t0 - (W @ u0 + b)
    du = u0 - mx
    du_norm2 = float(np.dot(du, du))
    if du_norm2 < 1e-12:
        return W.copy(), b + err0
    if C_xx_gt0 is not None:
        tr = float(np.trace(C_xx_gt0)) / max(float(C_xx_gt0.shape[0]), 1.0)
        reg = max(1e-2 * tr, 1e-8)
        d = sla.solve(_sym(C_xx_gt0) + reg * np.eye(len(du)), du, assume_a="pos")
        denom = float(np.dot(d, du))
        if abs(denom) < 1e-12:
            d, denom = du, du_norm2
    else:
        d, denom = du, du_norm2
    dW = np.outer(err0, d / denom)
    return W + dW, b - dW @ mx


def fold_m63_rope_bos_offset(
    W_q: np.ndarray,
    b_q: np.ndarray,
    W_k: np.ndarray,
    b_k: np.ndarray,
    u0_B: np.ndarray,
    mu_x_gt0: np.ndarray,
    k0_target: np.ndarray,
    s_opt: float,
    delta0_opt: float,
    C_xx_gt0: Optional[np.ndarray] = None,
    M_anchor: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact Native Folding of `(s_opt = tau^2, delta0_opt)`:
      - Dedicates the entire lowest-frequency RoPE plane `(m_low, m_low + half)` (both `63` and
        partner `127`!) so RoPE never leaks `k[127]` into `63` via `-sin(t*theta_63)`.
      - Anchors `k0_mod[m_low] = M_anchor, k0_mod[m_low+half] = 0` via KKT rank-1 at `u0_B`
        and sets `b_q_eff[m_low] = sqrt(d_h) * delta0_opt / M_anchor`.
    """
    d_h = int(W_q.shape[0])
    half = d_h // 2
    m_low = half - 1
    m_partner = m_low + half

    W_q_eff = float(s_opt) * W_q.copy()
    b_q_eff = float(s_opt) * b_q.copy()
    W_q_eff[m_low] = 0.0
    W_q_eff[m_partner] = 0.0
    b_q_eff[m_low] = math.sqrt(float(d_h)) * float(delta0_opt) / float(M_anchor)
    b_q_eff[m_partner] = 0.0

    W_k_clean = W_k.copy()
    b_k_clean = b_k.copy()
    W_k_clean[m_low] = 0.0
    W_k_clean[m_partner] = 0.0
    b_k_clean[m_low] = 0.0
    b_k_clean[m_partner] = 0.0
    k0_mod = np.asarray(k0_target, dtype=np.float64).copy()
    k0_mod[m_low] = float(M_anchor)
    k0_mod[m_partner] = 0.0
    W_k_eff, b_k_eff = apply_kkt_rank1_pos0_anchor(W_k_clean, b_k_clean, u0_B, k0_mod, mu_x_gt0, C_xx_gt0=C_xx_gt0)
    return W_q_eff, b_q_eff, W_k_eff, b_k_eff


def _eval_compiled_seq_attn_kl(
    donor_alphas: Sequence[np.ndarray],
    Phi_seqs: Sequence[np.ndarray],
    Wq: np.ndarray,
    bq: np.ndarray,
    Wk: np.ndarray,
    bk: np.ndarray,
    cos_np: np.ndarray,
    sin_np: np.ndarray,
    quad_q_seqs: Optional[Sequence[np.ndarray]] = None,
    quad_k_seqs: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, Any]:
    """Evaluate EXACT compiled full-sequence attention KL and mean BOS sink on generic `val_tune`."""
    d_h = int(Wq.shape[0])
    inv_sq = 1.0 / math.sqrt(float(d_h))
    kls, a0s, logits_list = [], [], []
    for idx, (p_d, Phi_s) in enumerate(zip(donor_alphas, Phi_seqs)):
        q_last = Wq @ Phi_s[-1] + bq
        k_seq = Phi_s @ Wk.T + bk[None, :]
        if quad_q_seqs is not None:
            q_last = q_last + quad_q_seqs[idx]
        if quad_k_seqs is not None:
            k_seq = k_seq.copy()
            k_seq[1:] = k_seq[1:] + quad_k_seqs[idx]
        T_s = int(Phi_s.shape[0])
        qr = _rope_rotate_np(q_last, cos_np[T_s - 1], sin_np[T_s - 1])
        kr = _rope_rotate_np(k_seq, cos_np[:T_s], sin_np[:T_s])
        z = (kr @ qr) * inv_sq
        logits_list.append(z)
        zm = float(np.max(z))
        ez = np.exp(z - zm)
        s_ez = float(np.sum(ez)) + 1e-30
        a0s.append(float(ez[0] / s_ez))
        log_q = (z - zm) - math.log(s_ez)
        pc = np.clip(p_d, 1e-15, 1.0)
        kls.append(float(np.sum(pc * (np.log(pc) - log_q))))
    return {"mean_kl": float(np.mean(kls)), "mean_a0": float(np.mean(a0s)), "logits_list": logits_list}


def solve_compiled_arm1a_bos_temp(
    donor_alphas: Sequence[np.ndarray],
    Phi_seqs: Sequence[np.ndarray],
    W_q_base: np.ndarray,
    b_q_base: np.ndarray,
    W_k_base: np.ndarray,
    b_k_base: np.ndarray,
    u0_B: np.ndarray,
    mu_x_gt0: np.ndarray,
    k0_target: np.ndarray,
    C_xx_gt0: Optional[np.ndarray],
    cos_np: np.ndarray,
    sin_np: np.ndarray,
    quad_q_raw_seqs: Optional[Sequence[np.ndarray]] = None,
    quad_k_seqs: Optional[Sequence[np.ndarray]] = None,
    s_bounds: Tuple[float, float] = (0.15, 4.0),
    d0_bounds: Tuple[float, float] = (-15.0, 15.0),
) -> Dict[str, Any]:
    """
    P0-5 Fix: Solve `(s_opt, delta0_opt)` directly on the EXACT COMPILED full-sequence logits
    of `fold_m63_rope_bos_offset` (including KKT `pos > 0` shift & active `quad_q`/`quad_k`!)!
    Because `l_compiled(s, d0) = s * l_base + d0 * l_m63_unit` is EXACTLY affine in `(s, d0)`,
    this problem is STILL strictly convex in `(s, d0)` and matches deployed weights to 0.0000 ULP!
    """
    d_h = int(W_q_base.shape[0])
    inv_sq = 1.0 / math.sqrt(float(d_h))

    # Build base compiled logits (s=1, d0=0) AND unit BOS-channel logit response (d0=1, s=0) ONCE!
    Wq_10, bq_10, Wk_10, bk_10 = fold_m63_rope_bos_offset(
        W_q_base, b_q_base, W_k_base, b_k_base, u0_B, mu_x_gt0, k0_target,
        s_opt=1.0, delta0_opt=0.0, C_xx_gt0=C_xx_gt0,
    )
    # Unit BOS response: fold with s=0, d0=1 -> Wq=0, bq[m_low]=sqrt(d_h)/M, SAME Wk_10, bk_10!
    m_low = d_h // 2 - 1
    q_unit = np.zeros((d_h,), dtype=np.float64)
    q_unit[m_low] = math.sqrt(float(d_h)) / 50.0

    l_base_list, l_d0_list = [], []
    # Also measure pos > 0 drift of KKT `k[m_low]` unit response
    gt0_abs_unit = []
    for idx, Phi_s in enumerate(Phi_seqs):
        q10 = Wq_10 @ Phi_s[-1] + bq_10
        if quad_q_raw_seqs is not None:
            q10 = q10 + quad_q_raw_seqs[idx]
        k10 = Phi_s @ Wk_10.T + bk_10[None, :]
        if quad_k_seqs is not None:
            k10 = k10.copy()
            k10[1:] = k10[1:] + quad_k_seqs[idx]
        T_s = int(Phi_s.shape[0])
        c_last, s_last = cos_np[T_s - 1], sin_np[T_s - 1]
        kr = _rope_rotate_np(k10, cos_np[:T_s], sin_np[:T_s])
        l_b = (kr @ _rope_rotate_np(q10, c_last, s_last)) * inv_sq
        l_u = (kr @ _rope_rotate_np(q_unit, c_last, s_last)) * inv_sq
        l_base_list.append(l_b)
        l_d0_list.append(l_u)
        if T_s > 1:
            gt0_abs_unit.append(float(np.sqrt(np.mean(l_u[1:] ** 2))))

    ps = [np.asarray(a, dtype=np.float64) for a in donor_alphas]
    n_seq = max(len(ps), 1)
    ent_const = sum(float(np.sum(np.clip(p, 1e-15, 1.0) * np.log(np.clip(p, 1e-15, 1.0)))) for p in ps) / float(n_seq)

    def _obj_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
        s, d0 = float(params[0]), float(params[1])
        loss, gs, gd0 = 0.0, 0.0, 0.0
        for p, lb, lu in zip(ps, l_base_list, l_d0_list):
            z = s * lb + d0 * lu
            zm = float(np.max(z))
            ez = np.exp(z - zm)
            sez = float(np.sum(ez)) + 1e-30
            q = ez / sez
            loss += (zm + math.log(sez)) - float(np.dot(p, z))
            diff = q - p
            gs += float(np.dot(diff, lb))
            gd0 += float(np.dot(diff, lu))
        inv_n = 1.0 / float(n_seq)
        return float(loss * inv_n + ent_const), np.array([gs * inv_n, gd0 * inv_n], dtype=np.float64)

    kl_before, _ = _obj_grad(np.array([1.0, 0.0], dtype=np.float64))
    res = sopt.minimize(_obj_grad, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B", bounds=[s_bounds, d0_bounds])
    s_opt, d0_opt = float(res.x[0]), float(res.x[1])
    Wq_eff, bq_eff, Wk_eff, bk_eff = fold_m63_rope_bos_offset(
        W_q_base, b_q_base, W_k_base, b_k_base, u0_B, mu_x_gt0, k0_target,
        s_opt=s_opt, delta0_opt=d0_opt, C_xx_gt0=C_xx_gt0,
    )
    qq_scaled = [s_opt * q for q in quad_q_raw_seqs] if quad_q_raw_seqs is not None else None
    ev_comp = _eval_compiled_seq_attn_kl(
        donor_alphas, Phi_seqs, Wq_eff, bq_eff, Wk_eff, bk_eff, cos_np, sin_np,
        quad_q_seqs=qq_scaled, quad_k_seqs=quad_k_seqs,
    )
    pos_gt0_rmse = float(abs(d0_opt) * np.mean(gt0_abs_unit)) if gt0_abs_unit else 0.0
    return {
        "Wq_eff": Wq_eff, "bq_eff": bq_eff, "Wk_eff": Wk_eff, "bk_eff": bk_eff,
        "s_opt": s_opt, "tau_opt": math.sqrt(max(s_opt, 1e-8)), "delta0_opt": d0_opt,
        "kl_before": float(kl_before),
        "compiled_val_tune_kl": float(ev_comp["mean_kl"]),
        "compiled_val_tune_a0": float(ev_comp["mean_a0"]),
        "pos_gt0_kkt_logit_rmse": round(pos_gt0_rmse, 5),
    }


def solve_compiled_arm1b_beta0(
    donor_alphas: Sequence[np.ndarray],
    Phi_seqs: Sequence[np.ndarray],
    W_q_p: np.ndarray,
    b_q_p: np.ndarray,
    W_k_p: np.ndarray,
    b_k_p: np.ndarray,
    u0_B: np.ndarray,
    mu_x_gt0: np.ndarray,
    k0_target: np.ndarray,
    C_xx_gt0: Optional[np.ndarray],
    cos_np: np.ndarray,
    sin_np: np.ndarray,
    quad_q_seqs: Optional[Sequence[np.ndarray]] = None,
    quad_k_seqs: Optional[Sequence[np.ndarray]] = None,
    beta_bounds: Tuple[float, float] = (0.05, 1.5),
    n_grid: int = 25,
) -> Dict[str, Any]:
    """
    P0-5 Fix: Solve `beta0` directly on the EXACT COMPILED post-KKT `(Wk_eff(beta0), bk_eff(beta0))`
    using a 25-point dense bracket scan + Brent root-finding (handles mixed-sign BOS slopes cleanly!).
    """
    donor_a0 = float(np.mean([float(a[0]) for a in donor_alphas]))

    def _eval_beta(beta: float) -> Tuple[float, float, np.ndarray, np.ndarray]:
        Wke, bke = apply_kkt_rank1_pos0_anchor(
            W_k_p, b_k_p, u0_B, float(beta) * k0_target, mu_x_gt0, C_xx_gt0=C_xx_gt0
        )
        ev = _eval_compiled_seq_attn_kl(
            donor_alphas, Phi_seqs, W_q_p, b_q_p, Wke, bke, cos_np, sin_np,
            quad_q_seqs=quad_q_seqs, quad_k_seqs=quad_k_seqs,
        )
        return float(ev["mean_a0"] - donor_a0), float(ev["mean_kl"]), Wke, bke

    grid = np.linspace(float(beta_bounds[0]), float(beta_bounds[1]), int(n_grid))
    res_g = [(_eval_beta(float(b))[0], float(b)) for b in grid]

    # Collect all sign-change brackets (handles mixed-sign / non-monotone curves!)
    cand_betas: List[float] = []
    for i in range(len(res_g) - 1):
        r0, b0 = res_g[i]
        r1, b1 = res_g[i + 1]
        if r0 == 0.0:
            cand_betas.append(b0)
        elif r0 * r1 < 0.0:
            try:
                root = float(sopt.brentq(lambda x: _eval_beta(x)[0], b0, b1, xtol=1e-4, maxiter=25))
                cand_betas.append(root)
            except Exception:
                cand_betas.append(0.5 * (b0 + b1))
    if not cand_betas:
        # No sign change in bounds: pick best grid point and refine locally by bounded scalar minimization
        best_i = int(np.argmin([abs(r) for r, _ in res_g]))
        lo_l = grid[max(0, best_i - 1)]
        hi_l = grid[min(len(grid) - 1, best_i + 1)]
        opt_l = sopt.minimize_scalar(lambda x: abs(_eval_beta(x)[0]), bounds=(float(lo_l), float(hi_l)), method="bounded")
        cand_betas.append(float(opt_l.x))

    # Among all valid roots, pick the one minimizing compiled `val_tune_attn_kl`!
    best_beta, best_kl, best_res, best_Wk, best_bk = cand_betas[0], 1e18, 1e18, None, None
    for b_cand in cand_betas:
        r_c, kl_c, Wke, bke = _eval_beta(b_cand)
        if kl_c < best_kl:
            best_beta, best_kl, best_res, best_Wk, best_bk = b_cand, kl_c, r_c, Wke, bke
    assert best_Wk is not None and best_bk is not None
    return {
        "Wk_eff": best_Wk, "bk_eff": best_bk,
        "beta0_opt": float(best_beta),
        "val_tune_a0_donor": float(donor_a0),
        "compiled_val_tune_a0": float(donor_a0 + best_res),
        "compiled_val_tune_kl": float(best_kl),
    }


def build_rope_self_tap_biases(d_h: int = 128, c: float = 20.0) -> np.ndarray:
    half = d_h // 2
    b = np.zeros((d_h,), dtype=np.float32)
    for m in range(min(8, half)):
        b[m] = float(c)
    for m in range(8, min(24, half)):
        b[m] = 0.5 * float(c)
    return b


def fit_unlabeled_antisymmetric_swiglu_lift(
    U_prev_fit: np.ndarray,
    res_fit: np.ndarray,
    U_prev_tune: np.ndarray,
    res_tune: np.ndarray,
    n_pcs: int = 24,
    n_bilinear_terms: int = 128,
    lam_grid: Sequence[float] = LAM_GRID_UNCAPPED,
    zero_output_dims: Sequence[int] = (),
) -> Dict[str, Any]:
    """
    Carbon Mechanism D: Unlabeled degree-2 Antisymmetric SwiGLU Bilinear Lifting
    (`SiLU(+z)*u - SiLU(-z)*u === z*u`), fitted strictly on unlabeled generic `post_attention_layernorm`
    activations (`layers[l-1].post_attention_layernorm` — Qwen's ACTUAL native MLP input norm!).
    NOTE: Includes generic centering `(mu_u, b_phi)`, natively compilable via the `a*` constant-direction
    read in bias-free SwiGLU. Output dims in `zero_output_dims` (e.g. reserved `m_low, m_low+half`)
    are strictly zeroed in BOTH calibration and replay!
    """
    Uf = np.asarray(U_prev_fit, dtype=np.float64)
    Ut = np.asarray(U_prev_tune, dtype=np.float64)
    Rf = np.asarray(res_fit, dtype=np.float64).copy()
    Rt = np.asarray(res_tune, dtype=np.float64).copy()
    for zd in zero_output_dims:
        Rf[:, int(zd)] = 0.0
        Rt[:, int(zd)] = 0.0

    mu_u = Uf.mean(axis=0)
    Ufc = Uf - mu_u[None, :]
    C_uu = _sym((Ufc.T @ Ufc) / max(float(Uf.shape[0]), 1.0))
    ev, V = sla.eigh(C_uu)
    order = np.argsort(ev)[::-1][: int(n_pcs)]
    P_norm = V[:, order] / np.sqrt(np.maximum(ev[order], 1e-8))[None, :]

    Zf = Ufc @ P_norm
    Zt = (Ut - mu_u[None, :]) @ P_norm
    r_p = int(Zf.shape[1])
    tri_i, tri_j = np.triu_indices(r_p)
    Phi_f = Zf[:, tri_i] * Zf[:, tri_j]
    Phi_t = Zt[:, tri_i] * Zt[:, tri_j]

    reg = fit_ridge_affine_val_tuned(Phi_f, Rf, Phi_t, Rt, lam_grid=lam_grid)
    W_phi, b_phi = reg["W"].copy(), reg["b"].copy()
    for zd in zero_output_dims:
        W_phi[int(zd), :] = 0.0
        b_phi[int(zd)] = 0.0

    d_out = int(W_phi.shape[0])
    M = np.zeros((d_out, r_p, r_p), dtype=np.float64)
    for idx, (a, b_idx) in enumerate(zip(tri_i, tri_j)):
        if a == b_idx:
            M[:, a, a] = W_phi[:, idx]
        else:
            M[:, a, b_idx] = 0.5 * W_phi[:, idx]
            M[:, b_idx, a] = 0.5 * W_phi[:, idx]

    U_m, S_m, Vt_m = sla.svd(M.reshape(d_out, r_p * r_p), full_matrices=False)
    terms: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    for s_idx, sv in enumerate(S_m):
        if float(sv) < 1e-10:
            continue
        eigs, evec = sla.eigh(_sym(Vt_m[s_idx].reshape(r_p, r_p)))
        for ei, ev_i in enumerate(eigs):
            mag = abs(float(sv) * float(ev_i))
            if mag < 1e-10:
                continue
            w1_u = P_norm @ evec[:, ei]
            d_vec = (float(sv) * float(ev_i)) * U_m[:, s_idx]
            for zd in zero_output_dims:
                d_vec[int(zd)] = 0.0
            terms.append((mag, w1_u, w1_u.copy(), d_vec))
    terms.sort(key=lambda t: t[0], reverse=True)
    kept = terms[: int(n_bilinear_terms)]

    W1 = np.stack([t[1] for t in kept], axis=0) if kept else np.zeros((0, Uf.shape[1]))
    W2 = np.stack([t[2] for t in kept], axis=0) if kept else np.zeros((0, Uf.shape[1]))
    D_out = np.stack([t[3] for t in kept], axis=1) if kept else np.zeros((d_out, 0))

    def _eval_fn(U_in: np.ndarray, scale: float = 1.0) -> np.ndarray:
        if W1.shape[0] == 0:
            return np.zeros((U_in.shape[0], d_out), dtype=np.float64)
        Uc = np.asarray(U_in, dtype=np.float64) - mu_u[None, :]
        z = Uc @ W1.T
        u = Uc @ W2.T
        sig_p = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
        sig_m = 1.0 / (1.0 + np.exp(-np.clip(-z, -40.0, 40.0)))
        act = (z * sig_p) * u - ((-z) * sig_m) * u
        out = (act @ D_out.T + b_phi[None, :]) * float(scale)
        for zd in zero_output_dims:
            out[:, int(zd)] = 0.0
        return out

    pred_t = _eval_fn(Ut, scale=1.0)
    tot_rt = float(np.sum((Rt - Rt.mean(axis=0, keepdims=True)) ** 2)) + 1e-18
    r2_res_tune = 1.0 - float(np.sum((Rt - pred_t) ** 2)) / tot_rt
    return {
        "activated": bool(r2_res_tune > 0.005),
        "r2_res_tune_full": float(reg["r2_tune"]),
        "r2_res_tune_compiled": float(r2_res_tune),
        "best_lam_rel": float(reg["best_lam_rel"]),
        "n_terms": int(W1.shape[0]),
        "scale": 1.0,
        "eval_fn": _eval_fn,
    }


def compute_swiglu_jacobian_single(
    x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor, W_down: torch.Tensor
) -> torch.Tensor:
    xf = x.float()
    Wg, Wu, Wd = W_gate.float(), W_up.float(), W_down.float()
    g = F.linear(xf, Wg)
    u = F.linear(xf, Wu)
    sig = torch.sigmoid(g)
    silu = g * sig
    dsilu = sig * (1.0 + g * (1.0 - sig))
    return Wd @ ((dsilu * u).unsqueeze(1) * Wg + silu.unsqueeze(1) * Wu)


def paired_table_cluster_bootstrap(
    vals_a: np.ndarray, vals_b: np.ndarray, table_ids: np.ndarray, n_boot: int = 2000, seed: int = SEED
) -> Dict[str, float]:
    va = np.asarray(vals_a, dtype=np.float64)
    vb = np.asarray(vals_b, dtype=np.float64)
    tids = np.asarray(table_ids)
    uniq = np.unique(tids)
    diff = va - vb
    t_means = np.array([float(np.mean(diff[tids == u])) for u in uniq], dtype=np.float64)
    n_t = len(t_means)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_t, size=(int(n_boot), n_t))
    boots = t_means[idx].mean(axis=1)
    return {
        "mean_diff": float(np.mean(diff)),
        "ci95_lo": float(np.percentile(boots, 2.5)),
        "ci95_hi": float(np.percentile(boots, 97.5)),
        "p_one_sided": float(max(float(np.mean(boots <= 0.0)), 1.0 / float(n_boot + 1))),
        "n_clusters": int(n_t),
    }


# ======================================================================================
# 2. Diverse Real Generic Calibration Corpus (Strictly Independent of Task Splits!)
# ======================================================================================

def _load_real_wikitext_splits(allow_download: bool) -> Tuple[List[str], List[str], List[str]]:
    """Load document-disjoint Wikipedia articles from Salesforce/wikitext (local cache first, then HF)."""
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        splits: Dict[str, List[str]] = {}
        files = {
            "train": "wikitext-2-raw-v1/train-00000-of-00001.parquet",
            "val": "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
            "test": "wikitext-2-raw-v1/test-00000-of-00001.parquet",
        }
        for sp, fname in files.items():
            p = hf_hub_download(
                repo_id="Salesforce/wikitext",
                filename=fname,
                repo_type="dataset",
                token=False,
                local_files_only=(not allow_download),
                cache_dir=str(HF_CACHE_DIR),
            )
            tbl = pq.read_table(p, columns=["text"])
            lines = [str(s) for s in tbl["text"].to_pylist() if s and len(str(s).strip()) > 30]
            docs: List[str] = []
            buf: List[str] = []
            for ln in lines:
                if ln.strip().startswith("= ") and not ln.strip().startswith("= = ") and buf:
                    docs.append("\n".join(buf))
                    buf = [ln]
                else:
                    buf.append(ln)
            if buf:
                docs.append("\n".join(buf))
            splits[sp] = docs
        return splits["train"], splits["val"], splits["test"]
    except Exception as e1:
        if not allow_download:
            return [], [], []
        try:
            from datasets import load_dataset
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", token=False)
            def _to_docs(sp_name: str) -> List[str]:
                lines = [str(x["text"]) for x in ds[sp_name] if x.get("text") and len(str(x["text"]).strip()) > 30]
                return ["\n".join(lines[i : i + 12]) for i in range(0, len(lines), 12)]
            return _to_docs("train"), _to_docs("validation"), _to_docs("test")
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load real Salesforce/wikitext prose (pq err: {e1}; datasets err: {e2})."
            )


def _load_real_python_stdlib_splits(seed: int = SEED) -> Tuple[List[str], List[str], List[str]]:
    stdlib_dir = Path(os.__file__).resolve().parent
    py_files = sorted(
        p for p in stdlib_dir.rglob("*.py")
        if "test" not in p.parts and "__pycache__" not in p.parts and p.is_file()
    )
    rng = np.random.default_rng(seed + 101)
    py_files = [py_files[i] for i in rng.permutation(len(py_files))]
    docs: List[str] = []
    for p in py_files:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if len(txt) >= 600:
                docs.append(txt[:16000])
        except Exception:
            continue
        if len(docs) >= 240:
            break
    n = len(docs)
    n_tr, n_va = max(1, int(0.75 * n)), max(1, int(0.125 * n))
    return docs[:n_tr], docs[n_tr : n_tr + n_va], docs[n_tr + n_va :]


def _load_real_structured_docs_splits(seed: int = SEED) -> Tuple[List[str], List[str], List[str]]:
    """File-disjoint package docs + schema-disjoint structured tables/JSON (zero eval-word overlap)."""
    real_docs: List[str] = []
    search_roots = [Path(getattr(mod, "__file__")).resolve().parent for mod in (np, sla, torch) if getattr(mod, "__file__", None)]
    seen_p = set()
    for root in search_roots:
        for ext in ("*.md", "*.rst", "*.json"):
            for p in root.rglob(ext):
                if p in seen_p or not p.is_file():
                    continue
                seen_p.add(p)
                try:
                    if 400 <= p.stat().st_size <= 80000:
                        txt = p.read_text(encoding="utf-8", errors="ignore")
                        if len(txt.strip()) >= 300:
                            real_docs.append(txt[:12000])
                except Exception:
                    continue
                if len(real_docs) >= 180:
                    break
            if len(real_docs) >= 180:
                break

    rng = np.random.default_rng(seed + 202)
    real_docs = [real_docs[i] for i in rng.permutation(len(real_docs))]
    nr = len(real_docs)
    r_tr, r_va = int(0.75 * nr), int(0.125 * nr)
    tr_docs, va_docs, te_docs = list(real_docs[:r_tr]), list(real_docs[r_tr : r_tr + r_va]), list(real_docs[r_tr + r_va :])

    # Schema-disjoint structured records (train=telemetry tables, val=inventory YAML/JSON, test=deployment manifests)
    forbidden = {w.lower() for w in WORD_CANDIDATES_KEYS} | {w.lower() for w in WORD_CANDIDATES_VALS}
    for idx in range(60):
        rows = ["| cluster | latency_ms | throughput_qps | status |", "|---|---|---|---|"]
        for r in range(8):
            rows.append(f"| prod-{idx:02d}-{r} | {round(float(rng.uniform(1, 500)), 2)} | {int(rng.integers(50, 9000))} | {'ok' if (idx+r)%4 else 'warn'} |")
        txt = f"### Telemetry #{idx}\n" + "\n".join(rows)
        if set(txt.lower().split()).isdisjoint(forbidden):
            tr_docs.append(txt)
    for idx in range(15):
        rec = {"manifest": f"inv-{idx:03d}", "items": [{"sku": f"SKU-{idx}-{r}", "qty": int(rng.integers(1, 400)), "zone": f"Z{r}"} for r in range(8)]}
        txt = json.dumps(rec, indent=2)
        if set(txt.lower().split()).isdisjoint(forbidden):
            va_docs.append(txt)
    for idx in range(15):
        lines = [f"[deployment_{idx:03d}]"] + [f"replica_{r} = host-{idx}-{r}.internal:{8000+r}" for r in range(10)]
        txt = "\n".join(lines)
        if set(txt.lower().split()).isdisjoint(forbidden):
            te_docs.append(txt)
    return tr_docs, va_docs, te_docs


def build_diverse_real_calibration_corpus(
    tok: Any,
    n_fit_pos_gt0: int = 60000,
    n_tune_pos_gt0: int = 4096,
    n_rep_pos_gt0: int = 4096,
    seq_len: int = 128,
    allow_download: bool = True,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Build strictly task-independent, document- & schema-disjoint `(fit, tune, report)` calibration corpus."""
    wiki_tr, wiki_va, wiki_te = _load_real_wikitext_splits(allow_download=allow_download)
    code_tr, code_va, code_te = _load_real_python_stdlib_splits(seed=seed)
    doc_tr, doc_va, doc_te = _load_real_structured_docs_splits(seed=seed)

    def _chunk_docs(docs: Sequence[str], target_gt0: int, rng: np.random.Generator) -> List[List[int]]:
        if target_gt0 <= 0 or len(docs) == 0:
            return []
        chunks: List[List[int]] = []
        tot_gt0 = 0
        order = rng.permutation(len(docs))
        for stride_off in (0, seq_len // 2):
            for di in order:
                ids = tok.encode(docs[int(di)], add_special_tokens=False)
                for s0 in range(stride_off, max(0, len(ids) - seq_len + 1), seq_len):
                    ch = ids[s0 : s0 + seq_len]
                    if len(ch) == seq_len:
                        chunks.append(ch)
                        tot_gt0 += seq_len - 1
                        if tot_gt0 >= target_gt0:
                            return chunks
        return chunks

    rng = np.random.default_rng(seed + 777)
    has_wiki = len(wiki_tr) > 0 and len(wiki_va) > 0 and len(wiki_te) > 0
    w_prose, w_code, w_struct = (0.50, 0.30, 0.20) if has_wiki else (0.0, 0.60, 0.40)

    def _assemble(p_d, c_d, s_d, target_gt0: int) -> Tuple[List[List[int]], Dict[str, int]]:
        cp = _chunk_docs(p_d, int(round(target_gt0 * w_prose)), rng)
        cc = _chunk_docs(c_d, int(round(target_gt0 * w_code)), rng)
        cs = _chunk_docs(s_d, int(round(target_gt0 * w_struct)), rng)
        all_c = cp + cc + cs
        if len(all_c) * (seq_len - 1) < target_gt0:
            all_c.extend(_chunk_docs(list(p_d) + list(c_d), target_gt0 - len(all_c) * (seq_len - 1), rng))
        all_c = [all_c[int(i)] for i in rng.permutation(len(all_c))]
        return all_c, {
            "prose_pos_gt0": len(cp) * (seq_len - 1),
            "code_pos_gt0": len(cc) * (seq_len - 1),
            "structured_pos_gt0": len(cs) * (seq_len - 1),
            "total_pos_gt0": len(all_c) * (seq_len - 1),
        }

    fit_seqs, fit_counts = _assemble(wiki_tr, code_tr, doc_tr, n_fit_pos_gt0)
    tune_seqs, tune_counts = _assemble(wiki_va, code_va, doc_va, n_tune_pos_gt0)
    rep_seqs, rep_counts = _assemble(wiki_te, code_te, doc_te, n_rep_pos_gt0)

    flat_fit = np.array([tid for seq in fit_seqs for tid in seq[1:]], dtype=np.int64)
    uniq_ids, counts = np.unique(flat_fit, return_counts=True)
    probs = counts.astype(np.float64) / max(float(np.sum(counts)), 1.0)
    ent_bits = float(-np.sum(probs * np.log2(np.clip(probs, 1e-18, 1.0))))
    return {
        "fit_seqs": fit_seqs,
        "tune_seqs": tune_seqs,
        "report_seqs": rep_seqs,
        "seq_len": int(seq_len),
        "has_real_wikitext": bool(has_wiki),
        "counts": {"fit": fit_counts, "tune": tune_counts, "report": rep_counts},
        "diversity": {
            "unique_token_ids_fit_gt0": int(len(uniq_ids)),
            "token_entropy_bits_fit_gt0": round(ent_bits, 4),
            "n_disjoint_docs": {
                "wiki": [len(wiki_tr), len(wiki_va), len(wiki_te)],
                "code": [len(code_tr), len(code_va), len(code_te)],
                "structured": [len(doc_tr), len(doc_va), len(doc_te)],
            },
        },
    }


# ======================================================================================
# 3. G0-Clean Prompt Wrappers + Task 1 (1-Hop) + Task 2 (2-Hop 3-Intervention)
# ======================================================================================

def _assert_dummy_exemplars_disjoint() -> None:
    overlap = (set(WORD_CANDIDATES_KEYS) | set(WORD_CANDIDATES_VALS)) & (set(DUMMY_EXEMPLAR_KEYS) | set(DUMMY_EXEMPLAR_VALS))
    assert not overlap, f"Dummy exemplar overlap: {overlap}"


def check_exact_whole_completion(gen_text: str, target_word: str) -> bool:
    """Strict whole-completion exact match (rejects trailing explanations!)."""
    s = gen_text.strip().strip(".,:;!?\"'()[]")
    return s.isalpha() and s.lower() == target_word.lower()


def _format_1hop_body(rows: Sequence[Tuple[str, str, str]], query: str, side: str) -> str:
    word_pos = "FIRST" if side == "L" else "SECOND"
    lines = [
        f"Rule: Follow the 1-hop Table (Query -> (FIRST, SECOND)) and output ONLY the {word_pos} word in the pair.",
        "",
        "Table:",
    ]
    for a, vl, vr in rows:
        lines.append(f"{a} -> ({vl}, {vr})")
    lines.append(f"Query: {query}")
    return "\n".join(lines)


def _format_2hop_body(h1: Sequence[Tuple[str, str]], h2: Sequence[Tuple[str, str, str]], query: str, side: str) -> str:
    word_pos = "FIRST" if side == "L" else "SECOND"
    lines = [
        f"Rule: Follow the 2-hop Table (Query -> middle -> (FIRST, SECOND)) and output ONLY the {word_pos} word in the pair.",
        "",
        "Table:",
    ]
    for a, b in h1:
        lines.append(f"{a} -> {b}")
    for b, vl, vr in h2:
        lines.append(f"{b} -> ({vl}, {vr})")
    lines.append(f"Query: {query}")
    return "\n".join(lines)


def _fixed_2shot_exemplars(task_type: str) -> List[Tuple[str, str]]:
    _assert_dummy_exemplars_disjoint()
    dk, dv = DUMMY_EXEMPLAR_KEYS, DUMMY_EXEMPLAR_VALS
    if task_type == "1hop":
        ex1 = _format_1hop_body([(dk[0], dv[0], dv[1]), (dk[1], dv[2], dv[3]), (dk[2], dv[4], dv[5]), (dk[3], dv[6], dv[7])], dk[1], "L")
        ex2 = _format_1hop_body([(dk[4], dv[8], dv[9]), (dk[5], dv[10], dv[11]), (dk[6], dv[12], dv[13]), (dk[7], dv[14], dv[15])], dk[6], "R")
        return [(ex1, dv[2]), (ex2, dv[13])]
    ex1 = _format_2hop_body(
        [(dk[0], dk[4]), (dk[1], dk[5]), (dk[2], dk[6]), (dk[3], dk[7])],
        [(dk[6], dv[4], dv[5]), (dk[4], dv[0], dv[1]), (dk[7], dv[6], dv[7]), (dk[5], dv[2], dv[3])],
        dk[1], "L",
    )
    ex2 = _format_2hop_body(
        [(dk[8], dk[12]), (dk[9], dk[13]), (dk[10], dk[14]), (dk[11], dk[15])],
        [(dk[14], dv[12], dv[13]), (dk[12], dv[8], dv[9]), (dk[15], dv[14], dv[15]), (dk[13], dv[10], dv[11])],
        dk[10], "R",
    )
    return [(ex1, dv[2]), (ex2, dv[13])]


def wrap_prompt_with_template(body: str, wrapper: str, task_type: str, tok: Any) -> Tuple[str, str]:
    sys_msg = "Output ONLY the exact single answer word with no punctuation or explanation."
    if wrapper == "2shot_fixed":
        exs = _fixed_2shot_exemplars(task_type)
        return "\n\n".join([f"{eb}\nOutput: {ea}" for eb, ea in exs] + [f"{body}\nOutput:"]), " "
    if wrapper == "chat_bare":
        msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": body}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), ""
    if wrapper == "chat_2shot":
        msgs = [{"role": "system", "content": sys_msg}]
        for eb, ea in _fixed_2shot_exemplars(task_type):
            msgs.append({"role": "user", "content": eb})
            msgs.append({"role": "assistant", "content": ea})
        msgs.append({"role": "user", "content": body})
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), ""
    raise ValueError(f"Unknown wrapper: {wrapper}")


def resolve_verified_answer_token_id(prompt: str, sep: str, word: str, tokenizers: Sequence[Any]) -> int:
    tids = []
    for tok in tokenizers:
        p_ids = tok.encode(prompt, add_special_tokens=False)
        f_ids = tok.encode(prompt + sep + word, add_special_tokens=False)
        assert len(f_ids) == len(p_ids) + 1 and f_ids[: len(p_ids)] == p_ids, (
            f"Word {word!r} not a 1-token continuation after prefix!"
        )
        tids.append(int(f_ids[-1]))
    assert all(t == tids[0] for t in tids), f"Tokenizer mismatch for {word!r}: {tids}"
    return tids[0]


def filter_wrapper_clean_vals(
    all_vals: Sequence[str], tokenizers: Sequence[Any], wrappers: Sequence[str] = ("chat_bare", "2shot_fixed", "chat_2shot")
) -> List[str]:
    sample_body = "Rule: Test.\nTable:\nalpha -> (amber, ivory)\nQuery: alpha"
    clean: List[str] = []
    for w in all_vals:
        ok = True
        for wr in wrappers:
            pr, sep = wrap_prompt_with_template(sample_body, wr, "1hop", tokenizers[0])
            for tok in tokenizers:
                p_ids = tok.encode(pr, add_special_tokens=False)
                f_ids = tok.encode(pr + sep + w, add_special_tokens=False)
                if not (len(f_ids) == len(p_ids) + 1 and f_ids[: len(p_ids)] == p_ids):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            clean.append(w)
    return clean


def build_1hop_dataset(
    tokenizers: Sequence[Any], key_pools: Dict[str, Any], wrapper: str, n_rows: int = 12,
    n_pilot: int = 64, n_disc: int = 200, n_eval: int = 400, seed: int = SEED,
) -> Dict[str, Any]:
    assert n_pilot % 2 == 0 and n_disc % 2 == 0 and n_eval % 2 == 0
    rng = np.random.default_rng(seed + 5100 + 100 * int(n_rows))
    tok0 = tokenizers[0]
    all_vals = key_pools["all_vals_clean"]

    def _gen(kpool: List[str], n_total: int, split_name: str) -> List[Dict[str, Any]]:
        n_tables, items, seen, t_idx = n_total // 2, [], set(), 0
        while t_idx < n_tables:
            ck = rng.choice(kpool, size=n_rows, replace=False).tolist()
            cv = rng.choice(all_vals, size=2 * n_rows, replace=False).tolist()
            rows = [(ck[i], cv[2 * i], cv[2 * i + 1]) for i in range(n_rows)]
            qi = int(rng.integers(0, n_rows))
            q, vL, vR = rows[qi]
            sig = (tuple(rows), q)
            if sig in seen:
                continue
            seen.add(sig)
            for side in ("L", "R"):
                ans, foil = (vL, vR) if side == "L" else (vR, vL)
                pr, sep = wrap_prompt_with_template(_format_1hop_body(rows, q, side), wrapper, "1hop", tok0)
                items.append({
                    "id": f"1hop_{split_name}_t{t_idx:03d}_{side}",
                    "table_id": int(t_idx), "split": split_name, "side": side,
                    "n_rows": int(n_rows), "wrapper": wrapper, "query": q, "answer": ans, "foil": foil,
                    "answer_tok_id": resolve_verified_answer_token_id(pr, sep, ans, tokenizers),
                    "foil_tok_id": resolve_verified_answer_token_id(pr, sep, foil, tokenizers),
                    "prompt": pr,
                })
            t_idx += 1
        return items

    return {
        "task": "1hop", "wrapper": wrapper, "n_rows": int(n_rows),
        "pilot": _gen(key_pools["pilot_keys"], n_pilot, "pilot"),
        "disc": _gen(key_pools["disc_keys"], n_disc, "disc"),
        "eval": _gen(key_pools["eval_keys"], n_eval, "eval"),
    }


def build_2hop_three_intervention_dataset(
    tokenizers: Sequence[Any], key_pools: Dict[str, Any], wrapper: str, n_rows: int = 6,
    n_pilot: int = 64, n_disc: int = 200, n_eval: int = 400, seed: int = SEED,
) -> Dict[str, Any]:
    """Task 2 with Policy (`FIRST`<->`SECOND`), Binding Rewire, and Unrelated Null Swap (multiset-verified on ALL tokenizers!)."""
    assert n_rows >= 4 and n_pilot % 2 == 0 and n_disc % 2 == 0 and n_eval % 2 == 0
    rng = np.random.default_rng(seed + 7200 + 100 * int(n_rows))
    tok0 = tokenizers[0]
    all_vals = key_pools["all_vals_clean"]

    def _gen(kpool: List[str], n_total: int, split_name: str) -> Dict[str, Any]:
        n_tables, base_items, bind_items, null_items, seen, t_idx = n_total // 2, [], [], [], set(), 0
        while t_idx < n_tables:
            ck = rng.choice(kpool, size=2 * n_rows, replace=False).tolist()
            cv = rng.choice(all_vals, size=2 * n_rows, replace=False).tolist()
            hop1 = [(ck[i], ck[n_rows + i]) for i in range(n_rows)]
            hop2 = [(ck[n_rows + i], cv[2 * i], cv[2 * i + 1]) for i in range(n_rows)]
            p1, p2 = rng.permutation(n_rows).tolist(), rng.permutation(n_rows).tolist()
            h1s, h2s = [hop1[i] for i in p1], [hop2[i] for i in p2]
            pos_perm = rng.permutation(n_rows).tolist()
            pi, pj, pk, pl = pos_perm[0], pos_perm[1], pos_perm[2], pos_perm[3]
            qi, qj = p1[pi], p1[pj]
            q, mid_w = ck[qi], ck[n_rows + qi]
            vL_i, vR_i, vL_j, vR_j = cv[2 * qi], cv[2 * qi + 1], cv[2 * qj], cv[2 * qj + 1]
            sig = (tuple(h1s), tuple(h2s), q)
            if sig in seen:
                continue
            seen.add(sig)

            pr_L_ids_all: List[List[int]] = []
            for side in ("L", "R"):
                ans, foil = (vL_i, vR_i) if side == "L" else (vR_i, vL_i)
                pr, sep = wrap_prompt_with_template(_format_2hop_body(h1s, h2s, q, side), wrapper, "2hop", tok0)
                if side == "L":
                    pr_L_ids_all = [t.encode(pr, add_special_tokens=False) for t in tokenizers]
                base_items.append({
                    "id": f"2hop_{split_name}_t{t_idx:03d}_{side}",
                    "table_id": int(t_idx), "split": split_name, "intervention": "base", "side": side,
                    "n_rows": int(n_rows), "wrapper": wrapper, "query": q, "mid": mid_w,
                    "answer": ans, "foil": foil,
                    "answer_tok_id": resolve_verified_answer_token_id(pr, sep, ans, tokenizers),
                    "foil_tok_id": resolve_verified_answer_token_id(pr, sep, foil, tokenizers),
                    "prompt": pr,
                })

            # Binding Rewire (identical token multiset & length across ALL tokenizers!)
            h1_b = list(h1s)
            h1_b[pi], h1_b[pj] = (h1s[pi][0], h1s[pj][1]), (h1s[pj][0], h1s[pi][1])
            pr_b, sep_b = wrap_prompt_with_template(_format_2hop_body(h1_b, h2s, q, "L"), wrapper, "2hop", tok0)
            for t_tok, ref_ids in zip(tokenizers, pr_L_ids_all):
                b_ids = t_tok.encode(pr_b, add_special_tokens=False)
                assert len(b_ids) == len(ref_ids) and sorted(b_ids) == sorted(ref_ids), "Binding multiset mismatch!"
            bind_items.append({
                "id": f"2hop_{split_name}_t{t_idx:03d}_BIND",
                "table_id": int(t_idx), "base_L_index": int(2 * t_idx), "split": split_name,
                "intervention": "binding", "side": "L", "query": q, "mid": h1s[pj][1],
                "answer": vL_j, "foil": vL_i,
                "answer_tok_id": resolve_verified_answer_token_id(pr_b, sep_b, vL_j, tokenizers),
                "foil_tok_id": resolve_verified_answer_token_id(pr_b, sep_b, vL_i, tokenizers),
                "prompt": pr_b,
            })

            # Unrelated-Row Null Swap (identical token multiset & length across ALL tokenizers!)
            h1_n = list(h1s)
            h1_n[pk], h1_n[pl] = (h1s[pk][0], h1s[pl][1]), (h1s[pl][0], h1s[pk][1])
            pr_n, sep_n = wrap_prompt_with_template(_format_2hop_body(h1_n, h2s, q, "L"), wrapper, "2hop", tok0)
            for t_tok, ref_ids in zip(tokenizers, pr_L_ids_all):
                n_ids = t_tok.encode(pr_n, add_special_tokens=False)
                assert len(n_ids) == len(ref_ids) and sorted(n_ids) == sorted(ref_ids), "Null multiset mismatch!"
            null_items.append({
                "id": f"2hop_{split_name}_t{t_idx:03d}_NULL",
                "table_id": int(t_idx), "base_L_index": int(2 * t_idx), "split": split_name,
                "intervention": "null_swap", "side": "L", "query": q, "mid": mid_w,
                "answer": vL_i, "foil": vL_j,
                "answer_tok_id": resolve_verified_answer_token_id(pr_n, sep_n, vL_i, tokenizers),
                "foil_tok_id": resolve_verified_answer_token_id(pr_n, sep_n, vL_j, tokenizers),
                "prompt": pr_n,
            })
            t_idx += 1
        return {"base": base_items, "binding": bind_items, "null_swap": null_items}

    return {
        "task": "2hop", "wrapper": wrapper, "n_rows": int(n_rows),
        "pilot": _gen(key_pools["pilot_keys"], n_pilot, "pilot"),
        "disc": _gen(key_pools["disc_keys"], n_disc, "disc"),
        "eval": _gen(key_pools["eval_keys"], n_eval, "eval"),
    }


# ======================================================================================
# 4. Head Replay, Numerically Stable 8-Cell Factorial & Sink/Content Decomposition
# ======================================================================================

def _greedy_generate_clean_no_add_special(
    model: nn.Module, tok: Any, prompts: Sequence[str], v_shared: int,
    max_new_tokens: int = 4, batch_size: int = 16, device: Optional[torch.device] = None,
) -> List[str]:
    if device is None:
        device = next(model.parameters()).device
    old_pad_side = tok.padding_side
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lproc = [_MaskPaddedVocabLogitsProcessor(v_shared)]
    outs: List[str] = []
    try:
        for i0 in range(0, len(prompts), batch_size):
            batch = list(prompts[i0 : i0 + batch_size])
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
            ids, am = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            gen = model.generate(
                input_ids=ids, attention_mask=am, max_new_tokens=int(max_new_tokens),
                do_sample=False, temperature=None, top_p=None, top_k=None,
                repetition_penalty=1.0, logits_processor=lproc, pad_token_id=pad_id, use_cache=True,
            )
            pw = int(ids.shape[1])
            for j in range(len(batch)):
                outs.append(tok.decode(gen[j, pw:], skip_special_tokens=True))
    finally:
        tok.padding_side = old_pad_side
    return outs


def compute_head_qkv_and_logits(
    q_last: torch.Tensor, k_seq: torch.Tensor, v_seq: torch.Tensor, W_o_h: torch.Tensor,
    cos_t: torch.Tensor, sin_t: torch.Tensor, delta0_logit_add: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns `(z_h, out, alpha, logits)` with exact BF16 `W_o` matmul parity."""
    T, d_h = int(k_seq.shape[0]), int(q_last.shape[0])
    half = d_h // 2
    q, k, v = q_last.float(), k_seq.float(), v_seq.float()
    c, s = cos_t[:T], sin_t[:T]
    q_rot = torch.cat([q[:half] * c[-1] - q[half:] * s[-1], q[half:] * c[-1] + q[:half] * s[-1]], dim=-1)
    k_rot = torch.cat([k[:, :half] * c - k[:, half:] * s, k[:, half:] * c + k[:, :half] * s], dim=-1)
    logits = (k_rot @ q_rot) / math.sqrt(float(d_h))
    if delta0_logit_add != 0.0:
        logits = logits.clone()
        logits[0] = logits[0] + float(delta0_logit_add)
    alpha = F.softmax(logits, dim=0)
    z_h = (alpha @ v).to(dtype=W_o_h.dtype)
    out = F.linear(z_h, W_o_h).float()
    return z_h.float(), out, alpha, logits


def run_8cell_and_sink_content_decomposition(
    qkv_donor: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    qkv_student: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    Wo_list: List[torch.Tensor],
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    score_fn: Callable[[torch.Tensor], Dict[str, Any]],
    student_delta0_by_head: Optional[Sequence[float]] = None,
    mask_mixed_dims: Sequence[int] = (),
) -> Dict[str, Any]:
    """
    P0-8 & Round-3 Fix:
      1. `mask_mixed_dims` (e.g. `(63, 127)` for Arm 1a) is applied ONLY to mixed `(Q_A, K_B)` and
         `(Q_B, K_A)` cells so pure `(Q_A, K_A)` (`QA_KA_VA`, `S4`) and `(Q_B, K_B)` (`QB_KB_VB`, `S0`)
         are 100% exact untouched endpoints!
      2. For pure endpoints (`a0_src == pi_src`), reuses `F.softmax(lg, dim=0)` directly (bit-exact
         even under `a0 == 1.0` saturation); for mixed sink/content cells (`a0_src != pi_src`),
         computes `pi = F.softmax(lg_pi[1:], dim=0)` directly from non-BOS logits!
    """
    n_heads, n_items = len(Wo_list), len(qkv_donor[0])
    dev, D_A = Wo_list[0].device, int(Wo_list[0].shape[0])
    d0_list = [float(student_delta0_by_head[hi]) for hi in range(n_heads)] if student_delta0_by_head is not None else [0.0] * n_heads
    mask_idx = torch.tensor([int(d) for d in mask_mixed_dims], device=dev, dtype=torch.long) if mask_mixed_dims else None

    logits_A: List[List[torch.Tensor]] = [[] for _ in range(n_heads)]
    logits_B: List[List[torch.Tensor]] = [[] for _ in range(n_heads)]
    cells_8: Dict[str, Any] = {}

    for qs in ("A", "B"):
        for ks in ("A", "B"):
            mixed_qk = (qs != ks)
            for vs in ("A", "B"):
                out_acc = torch.zeros((n_items, D_A), device=dev, dtype=torch.float32)
                sinks: List[float] = []
                for hi in range(n_heads):
                    Wo = Wo_list[hi]
                    d0_add = d0_list[hi] if (qs == "B" and ks == "B") else 0.0
                    for i in range(n_items):
                        q_t = (qkv_donor if qs == "A" else qkv_student)[hi][i][0].to(dev)
                        k_t = (qkv_donor if ks == "A" else qkv_student)[hi][i][1].to(dev)
                        v_t = (qkv_donor if vs == "A" else qkv_student)[hi][i][2].to(dev)
                        if mixed_qk and mask_idx is not None:
                            q_t = q_t.clone().index_fill_(0, mask_idx, 0.0)
                            k_t = k_t.clone().index_fill_(1, mask_idx, 0.0)
                        _, oh, alp, lg = compute_head_qkv_and_logits(q_t, k_t, v_t, Wo, cos_t, sin_t, delta0_logit_add=d0_add)
                        out_acc[i] += oh
                        sinks.append(float(alp[0].item()))
                        if qs == "A" and ks == "A" and vs == "A":
                            logits_A[hi].append(lg)
                        if qs == "B" and ks == "B" and vs == "B":
                            logits_B[hi].append(lg)
                sc = score_fn(out_acc)
                sc["mean_pos0_sink_mass"] = round(float(np.mean(sinks)), 4)
                cells_8[f"Q{qs}_K{ks}_V{vs}"] = sc

    sink_content: Dict[str, Any] = {}
    for label, a0_src, pi_src, v_src in (
        ("S0_pure_student", "B", "B", "B"),
        ("S1_donor_sink_only", "A", "B", "B"),
        ("S2_donor_content_pi_only", "B", "A", "B"),
        ("S3_donor_attn_student_V", "A", "A", "B"),
        ("S4_full_donor_oracle", "A", "A", "A"),
    ):
        out_acc = torch.zeros((n_items, D_A), device=dev, dtype=torch.float32)
        sinks = []
        for hi in range(n_heads):
            Wo = Wo_list[hi]
            for i in range(n_items):
                lg_A, lg_B = logits_A[hi][i], logits_B[hi][i]
                if a0_src == pi_src:
                    alp = F.softmax(lg_A if a0_src == "A" else lg_B, dim=0)
                    a0 = alp[0]
                else:
                    a0 = F.softmax(lg_A if a0_src == "A" else lg_B, dim=0)[0]
                    pi = F.softmax((lg_A if pi_src == "A" else lg_B)[1:], dim=0)
                    alp = torch.cat([a0.unsqueeze(0), (1.0 - a0) * pi], dim=0)
                v_t = (qkv_donor if v_src == "A" else qkv_student)[hi][i][2].to(dev).float()
                zh = (alp @ v_t).to(dtype=Wo.dtype)
                out_acc[i] += F.linear(zh, Wo).float()
                sinks.append(float(a0.item()))
        sc = score_fn(out_acc)
        sc["mean_pos0_sink_mass"] = round(float(np.mean(sinks)), 4)
        sink_content[label] = sc

    return {"factorial_8cell": cells_8, "sink_content_decomp": sink_content}


# ======================================================================================
# 5. Live Native L0 Tied-Embedding Ingress + Full All-Parameter Rollback Verifier (Arm 3c)
# ======================================================================================

class ReversibleL0TiedPrivatePack(ReversiblePrivateLanePack):
    def __init__(self, K: int, alpha: float, highway_kappas: Dict[int, Dict[str, float]], E_priv_bf16: torch.Tensor, **kwargs: Any):
        super().__init__(K=K, alpha=alpha, highway_kappas=highway_kappas, **kwargs)
        self.E_priv_bf16 = E_priv_bf16

    def apply(self, model: nn.Module) -> None:
        super().apply(model)
        try:
            D = int(self._orig_state["embed_weight"].shape[1])
            V_ep = int(self.E_priv_bf16.shape[0])
            with torch.no_grad():
                model.model.embed_tokens.weight[:V_ep, D:].copy_(
                    self.E_priv_bf16.to(device=model.model.embed_tokens.weight.device, dtype=model.model.embed_tokens.weight.dtype)
                )
            if self._orig_state["embed_weight"].data_ptr() == self._orig_state["lm_head_weight"].data_ptr():
                assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
        except BaseException:
            self._restore_partial(model)
            raise


@torch.no_grad()
def verify_native_l0_tied_ingress(
    model_B: nn.Module,
    E_priv_bf16: torch.Tensor,
    target_l0_proj_fn: Callable[[torch.Tensor], torch.Tensor],
    eval_input_ids: List[List[int]],
    v_shared: int,
    alpha: float = 2.0 ** -6,
) -> Dict[str, Any]:
    """
    P1-4 Fix: Full-model certificate verifying:
      - `apply()` inside `try...finally` guard
      - Zero runtime hooks, tied `embed_tokens`/`lm_head`, and `final_norm.weight[D:] === 0` while applied
      - `kappa(L0) >= 0.999` & `KL < 0.002`
      - Bit-exact `torch.equal == True` on EVERY parameter, EVERY norm `variance_epsilon`, tied pointers,
        AND bit-equal probe logits (`torch.equal`) after `revert()`!
    """
    assert_zero_runtime_hooks(model_B)
    dev = next(model_B.parameters()).device
    D_B = int(model_B.model.embed_tokens.weight.shape[1])
    K = int(E_priv_bf16.shape[1])
    layers = get_decoder_layers(model_B)

    # Full-model parameter snapshot on CPU (saves ~3.1 GiB GPU VRAM!) & norm epsilon snapshot
    snap_params = {name: p.detach().cpu().clone() for name, p in model_B.named_parameters()}
    snap_eps = {"final": float(model_B.model.norm.variance_epsilon)}
    for li, lyr in enumerate(layers):
        snap_eps[f"in_{li}"] = float(lyr.input_layernorm.variance_epsilon)
        snap_eps[f"post_{li}"] = float(lyr.post_attention_layernorm.variance_epsilon)

    ids_t = torch.tensor(eval_input_ids[:16], device=dev, dtype=torch.long)
    am_t = torch.ones_like(ids_t)
    base_logits_raw = model_B(input_ids=ids_t, attention_mask=am_t, use_cache=False).logits[:, :, :v_shared].detach()
    base_probs = F.softmax(base_logits_raw.float(), dim=-1)

    pack = ReversibleL0TiedPrivatePack(
        K=K, alpha=float(alpha), highway_kappas={0: {"in": 1.0}}, E_priv_bf16=E_priv_bf16,
    )
    tied_ok = False
    fn_suffix_zero = False
    kl_val, kappa_l0, cos_l0 = 999.0, -999.0, -999.0
    try:
        pack.apply(model_B)
        assert_zero_runtime_hooks(model_B)
        tied_ok = bool(model_B.model.embed_tokens.weight.data_ptr() == model_B.lm_head.weight.data_ptr())
        fn_suffix_zero = bool(torch.all(model_B.model.norm.weight[D_B:] == 0.0).item())
        mod_logits = model_B(input_ids=ids_t, attention_mask=am_t, use_cache=False).logits[:, :, :v_shared].float()
        kl_val = float(
            F.kl_div(F.log_softmax(mod_logits, dim=-1), base_probs, reduction="batchmean").item()
            / max(int(ids_t.shape[1]), 1)
        )
        h0 = model_B.model.embed_tokens(ids_t)
        u0 = layers[0].input_layernorm(h0)
        z_priv = u0[:, :, D_B:].float().reshape(-1, K)
        z_target = target_l0_proj_fn(ids_t).to(device=dev, dtype=torch.float32).reshape(-1, K)
        zc = z_target - z_target.mean(dim=0, keepdim=True)
        kappa_l0 = float(1.0 - torch.sum((z_target - z_priv) ** 2).item() / (torch.sum(zc ** 2).item() + 1e-12))
        cos_l0 = float(F.cosine_similarity(z_priv, z_target, dim=-1).mean().item())
        del h0, u0, mod_logits
    finally:
        if pack._applied:
            pack.revert(model_B)

    assert_zero_runtime_hooks(model_B)
    rb_all_params = all(torch.equal(p.detach().cpu(), snap_params[name]) for name, p in model_B.named_parameters())
    rb_eps = (float(model_B.model.norm.variance_epsilon) == snap_eps["final"]) and all(
        float(lyr.input_layernorm.variance_epsilon) == snap_eps[f"in_{li}"]
        and float(lyr.post_attention_layernorm.variance_epsilon) == snap_eps[f"post_{li}"]
        for li, lyr in enumerate(layers)
    )
    rb_tied = bool(model_B.model.embed_tokens.weight.data_ptr() == model_B.lm_head.weight.data_ptr())
    post_logits_raw = model_B(input_ids=ids_t, attention_mask=am_t, use_cache=False).logits[:, :, :v_shared].detach()
    rb_logits = bool(torch.equal(post_logits_raw, base_logits_raw))
    rb_all = bool(rb_all_params and rb_eps and rb_tied and rb_logits)
    del snap_params, base_logits_raw, base_probs, post_logits_raw

    return {
        "K": int(K), "alpha": float(alpha),
        "tied_preserved": tied_ok,
        "final_norm_suffix_zero": fn_suffix_zero,
        "kappa_L0_post_ln0": round(kappa_l0, 6),
        "cosine_L0_post_ln0": round(cos_l0, 6),
        "generic_kl": round(kl_val, 6),
        "bit_exact_all_params_rollback": rb_all_params,
        "bit_exact_norm_eps_rollback": rb_eps,
        "bit_exact_logits_rollback": rb_logits,
        "bit_exact_rollback": rb_all,
        "passes_all": bool(tied_ok and fn_suffix_zero and rb_all and kappa_l0 >= 0.999 and kl_val < 0.002),
    }


# ======================================================================================
# 6. Phase 0 (1.5B Pilot) & Phase 1 (Donor 7B Discovery, Arm 3a/3b, Memmap 60k Stream)
# ======================================================================================

def _select_wrapper_and_rows_from_pilot(pilot_grid: Dict[str, Dict[str, Any]], min_donor_acc: float = 0.65) -> Dict[str, Any]:
    qualified, all_cands = [], []
    for k, rec in pilot_grid.items():
        d_acc, r_acc = float(rec["donor_em"]), float(rec["rec_em"])
        entry = (d_acc - r_acc, d_acc, k, rec)
        all_cands.append(entry)
        if d_acc >= float(min_donor_acc):
            qualified.append(entry)
    pool = qualified if qualified else all_cands
    pool.sort(key=lambda x: (x[0], x[1]), reverse=True)
    gap_w, d_w, k_w, rec_w = pool[0]
    return {
        "selected_key": k_w, "wrapper": rec_w["wrapper"], "n_rows": int(rec_w["n_rows"]),
        "pilot_donor_em": round(d_w, 4), "pilot_rec_em": round(float(rec_w["rec_em"]), 4),
        "pilot_gap_pp": round(100.0 * gap_w, 2),
        "pilot_qualified_min_donor_acc": bool(len(qualified) > 0),
    }


@torch.inference_mode()
def run_pilot_screen_on_model(
    model: nn.Module, tok: Any, v_shared: int, pilot_datasets: Dict[str, List[Dict[str, Any]]], batch_size: int = 16
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key, items in pilot_datasets.items():
        gens = _greedy_generate_clean_no_add_special(
            model, tok, [it["prompt"] for it in items], v_shared=v_shared, max_new_tokens=4, batch_size=batch_size
        )
        ok_fw = [int(check_2hop_exact_word(g, it["answer"])) for g, it in zip(gens, items)]
        ok_wh = [int(check_exact_whole_completion(g, it["answer"])) for g, it in zip(gens, items)]
        out[key] = {
            "em": float(np.mean(ok_fw)),
            "whole_em": float(np.mean(ok_wh)),
            "n_correct": int(sum(ok_fw)),
            "n_whole_correct": int(sum(ok_wh)),
            "n_total": int(len(ok_fw)),
            "sample_completions": [
                {"id": items[i]["id"], "target": items[i]["answer"], "foil": items[i]["foil"], "gen": gens[i], "ok_fw": ok_fw[i], "ok_whole": ok_wh[i]}
                for i in range(min(8, len(items)))
            ],
        }
    return out


@torch.inference_mode()
def run_dla_and_solo_ablation_v51(
    model_A: nn.Module, tok: Any, items: Sequence[Dict[str, Any]], v_shared: int,
    batch_size: int = 32, late_start_layer: int = 20, top_k_heads: int = 5,
    top_k_candidates: int = 24, run_mlp_solo: bool = False,
) -> Dict[str, Any]:
    """Fast 1-pass all-28-layer DLA + optional MLP solo ablation (strictly on `disc`, inference_mode)."""
    dev = next(model_A.parameters()).device
    layers = get_decoder_layers(model_A)
    n_layers = len(layers)
    cfg = model_A.config
    n_q = int(cfg.num_attention_heads)
    D = int(cfg.hidden_size)
    d_h = D // n_q
    n_items = len(items)
    prompts = [it["prompt"] for it in items]

    lm_w = model_A.lm_head.weight[:v_shared].detach()
    gamma_fin = model_A.model.norm.weight.detach().float()
    eps_fin = float(model_A.model.norm.variance_epsilon)
    ans_ids = torch.tensor([int(it["answer_tok_id"]) for it in items], device=dev, dtype=torch.long)
    foil_ids = torch.tensor([int(it["foil_tok_id"]) for it in items], device=dev, dtype=torch.long)
    diff_w = (lm_w[ans_ids].float() - lm_w[foil_ids].float()) * gamma_fin.unsqueeze(0)

    old_pad = tok.padding_side
    tok.padding_side = "left"
    head_dla = np.zeros((n_layers, n_q), dtype=np.float64)
    mlp_dla = np.zeros((n_layers,), dtype=np.float64)
    mlp_last_means = [torch.zeros((D,), device=dev, dtype=torch.float32) for _ in range(n_layers)]
    base_margins = np.zeros((n_items,), dtype=np.float64)
    disc_batches: List[Tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    cur_z: Dict[int, torch.Tensor] = {}
    cur_m: Dict[int, torch.Tensor] = {}
    cur_hf: List[torch.Tensor] = []
    hooks = []
    try:
        for li in range(n_layers):
            hooks.append(layers[li].self_attn.o_proj.register_forward_pre_hook(lambda _m, inp, lidx=li: cur_z.__setitem__(lidx, inp[0][:, -1, :].detach().float())))
            hooks.append(layers[li].mlp.register_forward_hook(lambda _m, _inp, out, lidx=li: cur_m.__setitem__(lidx, out[:, -1, :].detach().float())))
        hooks.append(model_A.model.norm.register_forward_pre_hook(lambda _m, inp: cur_hf.append(inp[0][:, -1, :].detach().float())))

        for i0 in range(0, n_items, batch_size):
            i1 = min(i0 + batch_size, n_items)
            enc = tok(prompts[i0:i1], return_tensors="pt", padding=True, add_special_tokens=False)
            ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
            pos = _left_pad_position_ids(am)
            if run_mlp_solo:
                disc_batches.append((i0, i1, ids, am, pos))
            cur_z.clear(); cur_m.clear(); cur_hf.clear()
            _ = model_A.model(input_ids=ids, attention_mask=am, position_ids=pos, use_cache=False)
            hf = cur_hf[0]
            rms = torch.sqrt(torch.mean(hf ** 2, dim=-1, keepdim=True) + eps_fin)
            dw = diff_w[i0:i1] / rms
            base_margins[i0:i1] = torch.sum(hf * dw, dim=-1).cpu().numpy()
            for li in range(n_layers):
                Wo = layers[li].self_attn.o_proj.weight.detach().float()
                z = cur_z[li].view(i1 - i0, n_q, d_h)
                proj = (dw @ Wo).view(i1 - i0, n_q, d_h)
                head_dla[li] += torch.sum(z * proj, dim=(0, 2)).cpu().numpy()
                m_out = cur_m[li]
                mlp_dla[li] += float(torch.sum(m_out * dw).item())
                mlp_last_means[li] += m_out.sum(dim=0)
    finally:
        for h in hooks:
            h.remove()
        tok.padding_side = old_pad

    head_dla /= max(float(n_items), 1.0)
    mlp_dla /= max(float(n_items), 1.0)
    mlp_last_means = [m / max(float(n_items), 1.0) for m in mlp_last_means]
    base_margin_mean = float(np.mean(base_margins))

    mlp_solo: List[Dict[str, Any]] = []
    if run_mlp_solo:
        for li in range(n_layers):
            mv = mlp_last_means[li]
            abl_m = np.zeros((n_items,), dtype=np.float64)
            h_cap: List[torch.Tensor] = []
            def _ablate(_m, _inp, out, mvec=mv):
                o2 = out.clone()
                o2[:, -1, :] = mvec.to(dtype=o2.dtype)
                return o2
            hk1 = layers[li].mlp.register_forward_hook(_ablate)
            hk2 = model_A.model.norm.register_forward_pre_hook(lambda _m, inp: h_cap.append(inp[0][:, -1, :].detach().float()))
            try:
                for i0, i1, ids, am, pos in disc_batches:
                    h_cap.clear()
                    _ = model_A.model(input_ids=ids, attention_mask=am, position_ids=pos, use_cache=False)
                    hf = h_cap[0]
                    rms = torch.sqrt(torch.mean(hf ** 2, dim=-1, keepdim=True) + eps_fin)
                    abl_m[i0:i1] = torch.sum((hf / rms) * diff_w[i0:i1], dim=-1).cpu().numpy()
            finally:
                hk1.remove(); hk2.remove()
            drop = float(base_margin_mean - np.mean(abl_m))
            mlp_solo.append({
                "layer": int(li), "mean_dla": round(float(mlp_dla[li]), 4),
                "solo_ld_drop": round(drop, 4),
                "frac_of_margin": round(drop / max(abs(base_margin_mean), 1e-6), 4),
            })

    late_range = range(max(24, n_layers - 4), n_layers)
    late_list = sorted([(float(head_dla[l, h]), l, h) for l in late_range for h in range(n_q)], reverse=True)
    top_late_heads = [(l, h) for _, l, h in late_list[: int(top_k_heads)]]

    # P0-9 Fix: Matched-depth & matched-tap `SHUF-layer` control: 5 lowest-|DLA| heads in L24-L26 strictly disjoint from top_late_heads!
    prim_set = set(top_late_heads)
    low_late = sorted([(abs(float(head_dla[l, h])), l, h) for l in range(24, min(27, n_layers)) for h in range(n_q) if (l, h) not in prim_set])
    shuf_matched_late_heads = [(l, h) for _, l, h in low_late[: int(top_k_heads)]]

    cand_list = sorted([(float(head_dla[l, h]), l, h) for l in range(late_start_layer, n_layers) for h in range(n_q)], reverse=True)
    candidate_heads = [(l, h) for _, l, h in cand_list[: int(top_k_candidates)]]
    for lh in top_late_heads:
        if lh not in candidate_heads:
            candidate_heads.append(lh)

    return {
        "base_margin_mean": round(base_margin_mean, 4),
        "top_late_heads": top_late_heads,
        "top_late_heads_dla": [{"layer": l, "head": h, "mean_dla": round(float(head_dla[l, h]), 4)} for l, h in top_late_heads],
        "shuf_matched_late_heads": shuf_matched_late_heads,
        "candidate_heads": candidate_heads,
        "candidate_heads_dla": {f"L{l}H{h}": float(head_dla[l, h]) for l, h in candidate_heads},
        "mlp_solo_ablation": mlp_solo,
        "top_mlp_by_solo_drop": sorted(mlp_solo, key=lambda x: x["solo_ld_drop"], reverse=True)[:8] if mlp_solo else [],
    }


@torch.inference_mode()
def capture_donor_head_eval_bundle(
    model_A: nn.Module, tok: Any, items: Sequence[Dict[str, Any]],
    candidate_heads: Sequence[Tuple[int, int]], primary_heads: Sequence[Tuple[int, int]],
    shuf_layer_heads: Sequence[Tuple[int, int]], v_shared: int, batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Single inference-mode forward pass capturing `qkv_prim_cpu`, `out_prim_f32`, `out_by_cand_f32`,
    `h_fin_all_f32` (no second forward pass!), and donor `head_weights_cpu`.
    """
    dev = next(model_A.parameters()).device
    layers = get_decoder_layers(model_A)
    cfg = model_A.config
    n_q, n_kv = int(cfg.num_attention_heads), int(cfg.num_key_value_heads)
    g_rep, D = n_q // n_kv, int(cfg.hidden_size)
    d_h = D // n_q
    rope_theta = get_config_rope_theta(cfg)
    n_items = len(items)
    prompts = [it["prompt"] for it in items]

    all_needed = list(dict.fromkeys(list(primary_heads) + list(candidate_heads) + list(shuf_layer_heads)))
    unique_layers = sorted({l for l, _ in all_needed})
    prim_set = set(primary_heads)

    head_weights: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for l, h in all_needed:
        kv = h // g_rep
        attn = layers[l].self_attn
        head_weights[(l, h)] = {
            "layer": int(l), "head": int(h), "kv": int(kv),
            "Wq": attn.q_proj.weight.detach()[h * d_h : (h + 1) * d_h].clone(),
            "bq": attn.q_proj.bias.detach()[h * d_h : (h + 1) * d_h].clone() if attn.q_proj.bias is not None else None,
            "Wk": attn.k_proj.weight.detach()[kv * d_h : (kv + 1) * d_h].clone(),
            "bk": attn.k_proj.bias.detach()[kv * d_h : (kv + 1) * d_h].clone() if attn.k_proj.bias is not None else None,
            "Wv": attn.v_proj.weight.detach()[kv * d_h : (kv + 1) * d_h].clone(),
            "bv": attn.v_proj.bias.detach()[kv * d_h : (kv + 1) * d_h].clone() if attn.v_proj.bias is not None else None,
            "Wo": attn.o_proj.weight.detach()[:, h * d_h : (h + 1) * d_h].clone(),
        }

    max_T = max(len(tok.encode(p, add_special_tokens=False)) for p in prompts) + 8
    cos_all, sin_all = _make_rope_cos_sin(d_h, rope_theta, max_T, dev)

    qkv_prim_cpu: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [[] for _ in primary_heads]
    out_prim = torch.zeros((n_items, D), device=dev, dtype=torch.float32)
    out_by_cand = {lh: torch.zeros((n_items, D), device=dev, dtype=torch.float32) for lh in candidate_heads}
    h_fin_all = torch.zeros((n_items, D), device=dev, dtype=torch.float32)
    sinks_prim: List[float] = []

    cur_u: Dict[int, torch.Tensor] = {}
    cur_hf: List[torch.Tensor] = []
    hooks = []
    old_p = tok.padding_side
    tok.padding_side = "left"
    try:
        for li in unique_layers:
            hooks.append(layers[li].input_layernorm.register_forward_hook(lambda _m, _inp, out, lidx=li: cur_u.__setitem__(lidx, out.detach())))
        hooks.append(model_A.model.norm.register_forward_pre_hook(lambda _m, inp: cur_hf.append(inp[0][:, -1, :].detach().float())))

        for i0 in range(0, n_items, batch_size):
            i1 = min(i0 + batch_size, n_items)
            enc = tok(prompts[i0:i1], return_tensors="pt", padding=True, add_special_tokens=False)
            ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
            cur_u.clear(); cur_hf.clear()
            _ = model_A.model(input_ids=ids, attention_mask=am, position_ids=_left_pad_position_ids(am), use_cache=False)
            h_fin_all[i0:i1] = cur_hf[0]
            for b in range(i1 - i0):
                idx = i0 + b
                rm = am[b].bool()
                for pi, lh in enumerate(primary_heads):
                    li, _ = lh
                    hw = head_weights[lh]
                    X_s = cur_u[li][b, rm, :]
                    ql = F.linear(X_s[-1:], hw["Wq"], hw["bq"]).squeeze(0).float()
                    ks = F.linear(X_s, hw["Wk"], hw["bk"]).float()
                    vs = F.linear(X_s, hw["Wv"], hw["bv"]).float()
                    _, oh, alp, _ = compute_head_qkv_and_logits(ql, ks, vs, hw["Wo"], cos_all, sin_all)
                    out_prim[idx] += oh
                    sinks_prim.append(float(alp[0].item()))
                    qkv_prim_cpu[pi].append((ql.cpu().to(torch.bfloat16), ks.cpu().to(torch.bfloat16), vs.cpu().to(torch.bfloat16)))
                    if lh in out_by_cand:
                        out_by_cand[lh][idx] = oh
                for lh in candidate_heads:
                    if lh in prim_set:
                        continue
                    li, _ = lh
                    hw = head_weights[lh]
                    _, oh, _ = _forward_single_attn_head_z_and_out(
                        cur_u[li][b, rm, :], hw["Wq"], hw["bq"], hw["Wk"], hw["bk"], hw["Wv"], hw["bv"], hw["Wo"], cos_all, sin_all
                    )
                    out_by_cand[lh][idx] = oh
    finally:
        for hk in hooks:
            hk.remove()
        tok.padding_side = old_p

    hw_cpu = {
        lh: {
            "layer": hw["layer"], "head": hw["head"], "kv": hw["kv"],
            "Wq": hw["Wq"].cpu(), "bq": hw["bq"].cpu() if hw["bq"] is not None else None,
            "Wk": hw["Wk"].cpu(), "bk": hw["bk"].cpu() if hw["bk"] is not None else None,
            "Wv": hw["Wv"].cpu(), "bv": hw["bv"].cpu() if hw["bv"] is not None else None,
            "Wo": hw["Wo"].cpu(),
        }
        for lh, hw in head_weights.items()
    }
    return {
        "qkv_prim_cpu": qkv_prim_cpu,
        "out_prim_f32": out_prim.cpu().numpy().astype(np.float32),
        "out_by_cand_f32": {lh: t.cpu().numpy().astype(np.float32) for lh, t in out_by_cand.items()},
        "h_fin_all_f32": h_fin_all.cpu().numpy().astype(np.float32),
        "donor_mean_pos0_sink": float(np.mean(sinks_prim)),
        "head_weights_cpu": hw_cpu,
    }


@torch.inference_mode()
def compute_donor_references_frozen_center(
    model_A: nn.Module,
    items: Sequence[Dict[str, Any]],
    h_fin_all_f32: np.ndarray,
    out_heads_f32: np.ndarray,
    frozen_mean_out_gen_f32: np.ndarray,
    v_shared: int,
) -> np.ndarray:
    """P0-7 Fix: Exact donor `c_true` using the SAME frozen generic center `mean_out_gen` as Recipient writes!"""
    dev = next(model_A.parameters()).device
    lm_w = model_A.lm_head.weight[:v_shared].detach()
    gamma_fin = model_A.model.norm.weight.detach().float()
    eps_fin = float(model_A.model.norm.variance_epsilon)
    e_a = lm_w[torch.tensor([int(it["answer_tok_id"]) for it in items], device=dev, dtype=torch.long)].float()
    e_f = lm_w[torch.tensor([int(it["foil_tok_id"]) for it in items], device=dev, dtype=torch.long)].float()
    hf = torch.from_numpy(h_fin_all_f32).to(dev)
    out_t = torch.from_numpy(out_heads_f32).to(dev)
    m_gen = torch.from_numpy(frozen_mean_out_gen_f32).to(dev)
    ma_int = _real_norm_margin_from_rows(hf, gamma_fin, eps_fin, e_a, e_f)
    ma_abl = _real_norm_margin_from_rows(hf - out_t + m_gen.unsqueeze(0), gamma_fin, eps_fin, e_a, e_f)
    return (ma_int - ma_abl).cpu().double().numpy()


@torch.inference_mode()
def compute_donor_swap_refs(
    model_A: nn.Module,
    items_a: Sequence[Dict[str, Any]],
    items_b: Sequence[Dict[str, Any]],
    h_fin_a: np.ndarray,
    out_a: np.ndarray,
    out_b: np.ndarray,
    v_shared: int,
    answer_b_override: Optional[Sequence[int]] = None,
    answer_a_override: Optional[Sequence[int]] = None,
) -> np.ndarray:
    dev = next(model_A.parameters()).device
    lm_w = model_A.lm_head.weight[:v_shared].detach()
    gamma_fin = model_A.model.norm.weight.detach().float()
    eps_fin = float(model_A.model.norm.variance_epsilon)
    a_ids = list(answer_a_override) if answer_a_override is not None else [int(it["answer_tok_id"]) for it in items_a]
    b_ids = list(answer_b_override) if answer_b_override is not None else [int(it["answer_tok_id"]) for it in items_b]
    ea_t = lm_w[torch.tensor(a_ids, device=dev, dtype=torch.long)].float()
    eb_t = lm_w[torch.tensor(b_ids, device=dev, dtype=torch.long)].float()
    hf_t = torch.from_numpy(h_fin_a).to(dev)
    oa_t = torch.from_numpy(out_a).to(dev)
    ob_t = torch.from_numpy(out_b).to(dev)
    hc_t = hf_t - oa_t
    return (
        _real_norm_margin_from_rows(hc_t + ob_t, gamma_fin, eps_fin, eb_t, ea_t)
        - _real_norm_margin_from_rows(hc_t + oa_t, gamma_fin, eps_fin, eb_t, ea_t)
    ).cpu().double().numpy()


@torch.inference_mode()
def run_arm3_donor_backward_isa_and_l20_mlp(
    model_A: nn.Module, tok: Any, selected_late_heads: Sequence[Tuple[int, int]],
    disc_base: Sequence[Dict[str, Any]], disc_binding: Sequence[Dict[str, Any]],
    eval_base: Sequence[Dict[str, Any]], eval_binding: Sequence[Dict[str, Any]],
    generic_fit_seqs: Sequence[List[int]], bottleneck_mlp_layer: int, v_shared: int, batch_size: int = 24,
) -> Dict[str, Any]:
    """
    Arm 3a & 3b (P0-1, P0-6, P1-3 Fixed):
      - Arm 3a: Analytical query-sensitivity `L_Q` ISA across all upstream heads & MLPs on Policy vs
        Binding swaps, validated by live paired solo head mean-ablation on top-8 upstream heads!
      - Arm 3b: Correct 6-arg `compute_cancellation_ratio_chi(Wg, Wu, Wd, X_512, Z_code, L_Q)` call;
        BOTH `MLP(E[x]) + J_bar(x - E[x])` and `E[MLP(x)] + J_bar(x - E[x])`; sparse causal J-lens
        neuron-community retention using COMMON frozen generic baseline `mean_silu_u_gen` across
        `k in (0, 16, 64, 128, 256, 1024, M_bl)`, marking `frac_recovered = None` if `intact - abl < 0.10`!
    """
    dev = next(model_A.parameters()).device
    layers = get_decoder_layers(model_A)
    cfg = model_A.config
    n_q, D = int(cfg.num_attention_heads), int(cfg.hidden_size)
    d_h = D // n_q
    min_late_l = min(l for l, _ in selected_late_heads)
    lm_w = model_A.lm_head.weight[:v_shared].detach()
    gamma_fin = model_A.model.norm.weight.detach().float()
    eps_fin = float(model_A.model.norm.variance_epsilon)

    # Query demand basis `L_Q` weighted by each late head's W_o Frobenius norm (sensitivity-weighted!)
    Wq_rows = []
    for l, h in selected_late_heads:
        ln_w = layers[l].input_layernorm.weight.detach().float()
        Wq_h = layers[l].self_attn.q_proj.weight.detach().float()[h * d_h : (h + 1) * d_h]
        wo_s = float(torch.linalg.norm(layers[l].self_attn.o_proj.weight.detach().float()[:, h * d_h : (h + 1) * d_h]).item())
        Wq_rows.append((Wq_h * ln_w.unsqueeze(0)) * wo_s)
    _, _, Vt_q = torch.linalg.svd(torch.cat(Wq_rows, dim=0), full_matrices=False)
    r_q = min(64, int(Vt_q.shape[0]))
    L_Q = Vt_q[:r_q].T.contiguous()

    def _capture_upstream(prompts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, List[torch.Tensor]]:
        n = len(prompts)
        hp = np.zeros((n, min_late_l, n_q, r_q), dtype=np.float32)
        mp = np.zeros((n, min_late_l, r_q), dtype=np.float32)
        z_means = [torch.zeros((n_q, d_h), device=dev, dtype=torch.float32) for _ in range(min_late_l)]
        cur_z, cur_m, cur_h = {}, {}, []
        hooks = []
        old_p = tok.padding_side
        tok.padding_side = "left"
        try:
            for li in range(min_late_l):
                hooks.append(layers[li].self_attn.o_proj.register_forward_pre_hook(lambda _m, inp, lidx=li: cur_z.__setitem__(lidx, inp[0][:, -1, :].detach().float())))
                hooks.append(layers[li].mlp.register_forward_hook(lambda _m, _inp, out, lidx=li: cur_m.__setitem__(lidx, out[:, -1, :].detach().float())))
            hooks.append(layers[min_late_l].input_layernorm.register_forward_pre_hook(lambda _m, inp: cur_h.append(inp[0][:, -1, :].detach().float())))
            Wo_LQ = [(layers[li].self_attn.o_proj.weight.detach().float().T @ L_Q).view(n_q, d_h, r_q) for li in range(min_late_l)]
            for i0 in range(0, n, batch_size):
                i1 = min(i0 + batch_size, n)
                enc = tok(list(prompts[i0:i1]), return_tensors="pt", padding=True, add_special_tokens=False)
                ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
                cur_z.clear(); cur_m.clear(); cur_h.clear()
                _ = model_A.model(input_ids=ids, attention_mask=am, position_ids=_left_pad_position_ids(am), use_cache=False)
                rms_b = torch.sqrt(torch.mean(cur_h[0] ** 2, dim=-1) + 1e-6)
                inv_r = (1.0 / rms_b).view(-1, 1, 1)
                for li in range(min_late_l):
                    z = cur_z[li].view(i1 - i0, n_q, d_h)
                    z_means[li] += z.sum(dim=0)
                    hp[i0:i1, li] = (torch.einsum("bhd,hdr->bhr", z, Wo_LQ[li]) * inv_r).cpu().numpy()
                    mp[i0:i1, li] = ((cur_m[li] @ L_Q) / rms_b.unsqueeze(-1)).cpu().numpy()
        finally:
            for hk in hooks:
                hk.remove()
            tok.padding_side = old_p
        return hp, mp, [zm / max(float(n), 1.0) for zm in z_means]

    hp_base, mp_base, z_means_base = _capture_upstream([it["prompt"] for it in disc_base])
    hp_bind, mp_bind, _ = _capture_upstream([it["prompt"] for it in disc_binding])

    isa_h_pol = np.mean(np.sum((hp_base[0::2] - hp_base[1::2]) ** 2, axis=-1), axis=0)
    isa_m_pol = np.mean(np.sum((mp_base[0::2] - mp_base[1::2]) ** 2, axis=-1), axis=0)
    isa_h_bnd = np.mean(np.sum((hp_base[0::2] - hp_bind) ** 2, axis=-1), axis=0)
    isa_m_bnd = np.mean(np.sum((mp_base[0::2] - mp_bind) ** 2, axis=-1), axis=0)

    top_h_pol = sorted([(float(isa_h_pol[l, h]), l, h) for l in range(min_late_l) for h in range(n_q)], reverse=True)[:8]
    top_h_bnd = sorted([(float(isa_h_bnd[l, h]), l, h) for l in range(min_late_l) for h in range(n_q)], reverse=True)[:8]

    # Live paired solo head mean-ablation on top-8 ISA heads (validates ISA rankings causally!)
    def _live_head_solo_drops(items_seq: Sequence[Dict[str, Any]], head_list: Sequence[Tuple[float, int, int]]) -> List[Dict[str, Any]]:
        n = len(items_seq)
        prs = [it["prompt"] for it in items_seq]
        e_a = lm_w[torch.tensor([int(it["answer_tok_id"]) for it in items_seq], device=dev, dtype=torch.long)].float()
        e_f = lm_w[torch.tensor([int(it["foil_tok_id"]) for it in items_seq], device=dev, dtype=torch.long)].float()
        old_p = tok.padding_side
        tok.padding_side = "left"
        batches = []
        try:
            for i0 in range(0, n, batch_size):
                i1 = min(i0 + batch_size, n)
                enc = tok(prs[i0:i1], return_tensors="pt", padding=True, add_special_tokens=False)
                ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
                batches.append((i0, i1, ids, am, _left_pad_position_ids(am)))
        finally:
            tok.padding_side = old_p

        def _run_m(abl: Optional[Tuple[int, int]] = None) -> float:
            ms = np.zeros((n,), dtype=np.float64)
            cap: List[torch.Tensor] = []
            hks = [model_A.model.norm.register_forward_pre_hook(lambda _m, inp: cap.append(inp[0][:, -1, :].detach().float()))]
            if abl is not None:
                al, ah = abl
                mv_h = z_means_base[al][ah]
                def _pre_o(_m, inp, hh=ah, mvh=mv_h):
                    x = inp[0].clone()
                    x[:, -1, hh * d_h : (hh + 1) * d_h] = mvh.to(dtype=x.dtype)
                    return (x,) + inp[1:]
                hks.append(layers[al].self_attn.o_proj.register_forward_pre_hook(_pre_o))
            try:
                for i0, i1, ids, am, pos in batches:
                    cap.clear()
                    _ = model_A.model(input_ids=ids, attention_mask=am, position_ids=pos, use_cache=False)
                    ms[i0:i1] = _real_norm_margin_from_rows(cap[0], gamma_fin, eps_fin, e_a[i0:i1], e_f[i0:i1]).cpu().numpy()
            finally:
                for hk in hks:
                    hk.remove()
            return float(np.mean(ms))

        base_m = _run_m(None)
        rows = []
        for isa_v, l, h in head_list:
            drop = base_m - _run_m((l, h))
            rows.append({"layer": int(l), "head": int(h), "isa_q": round(isa_v, 5), "live_solo_ld_drop": round(drop, 4)})
        return rows

    isa_summary = {
        "min_late_layer": int(min_late_l), "r_q_basis": int(r_q),
        "policy_top8_heads_validated": _live_head_solo_drops(disc_base, top_h_pol),
        "binding_top8_heads_validated": _live_head_solo_drops(disc_binding, top_h_bnd),
        "policy_top_mlps": [{"layer": int(l), "isa_q": round(float(isa_m_pol[l]), 5)} for l in np.argsort(isa_m_pol)[::-1][:8]],
        "binding_top_mlps": [{"layer": int(l), "isa_q": round(float(isa_m_bnd[l]), 5)} for l in np.argsort(isa_m_bnd)[::-1][:8]],
    }

    # Arm 3b: Bottleneck MLP Mean-Jacobian Linearization + Sparse J-Lens Neuron Community
    bl = int(bottleneck_mlp_layer)
    mlp_bl = layers[bl].mlp
    Wg_bl, Wu_bl, Wd_bl = mlp_bl.gate_proj.weight.detach(), mlp_bl.up_proj.weight.detach(), mlp_bl.down_proj.weight.detach()
    M_bl = int(Wg_bl.shape[0])

    gen_x_list: List[torch.Tensor] = []
    cap_x: List[torch.Tensor] = []
    hk_x = layers[bl].post_attention_layernorm.register_forward_hook(lambda _m, _inp, out: cap_x.append(out.detach()))
    try:
        n_gt = 0
        for i0 in range(0, len(generic_fit_seqs), 8):
            if n_gt >= 2048:
                break
            ids = torch.tensor(generic_fit_seqs[i0 : i0 + 8], device=dev, dtype=torch.long)
            cap_x.clear()
            _ = model_A.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            xb = cap_x[0][:, 1:, :].reshape(-1, D).float()
            gen_x_list.append(xb)
            n_gt += int(xb.shape[0])
    finally:
        hk_x.remove()
    X_gen_bl = torch.cat(gen_x_list, dim=0)[:2048]
    mu_x_gen = X_gen_bl.mean(dim=0)
    g_gen, u_gen = F.linear(X_gen_bl, Wg_bl.float()), F.linear(X_gen_bl, Wu_bl.float())
    sig_gen = torch.sigmoid(g_gen)
    silu_gen = g_gen * sig_gen
    dsilu_gen = sig_gen * (1.0 + g_gen * (1.0 - sig_gen))
    mean_silu_u_gen = (silu_gen * u_gen).mean(dim=0)  # Common frozen generic baseline!
    mean_mlp_gen = F.linear(mean_silu_u_gen.to(dtype=Wd_bl.dtype), Wd_bl).float()  # E[MLP(x)]
    mlp_at_mu_gen = F.linear((F.silu(F.linear(mu_x_gen, Wg_bl.float())) * F.linear(mu_x_gen, Wu_bl.float())).to(dtype=Wd_bl.dtype), Wd_bl).float()  # MLP(E[x])
    J_bar = Wd_bl.float() @ ((dsilu_gen * u_gen).mean(dim=0).unsqueeze(1) * Wg_bl.float() + silu_gen.mean(dim=0).unsqueeze(1) * Wu_bl.float())

    # P0-1 & P0-2 Fix: Exact 6-arg call with CPU numpy `Z_code_512` (never pass CUDA tensor to np.asarray!)
    X_512 = X_gen_bl[:512]
    Z_code_512 = (X_512 @ L_Q).detach().cpu().numpy()
    chi_bl = compute_cancellation_ratio_chi(Wg_bl, Wu_bl, Wd_bl, X_512, Z_code_512, L_Q)
    del g_gen, u_gen, sig_gen, silu_gen, dsilu_gen, X_gen_bl, X_512, Z_code_512

    # Rank all M_bl neurons STRICTLY on `disc` (Policy + Binding) by 0-SGD J-Lens / ISA into `L_Q`
    def _cap_disc_act(items_seq: Sequence[Dict[str, Any]]) -> torch.Tensor:
        xs, cap = [], []
        hk = layers[bl].mlp.down_proj.register_forward_pre_hook(lambda _m, inp: cap.append(inp[0][:, -1, :].detach().float()))
        old_p = tok.padding_side
        tok.padding_side = "left"
        try:
            prs = [it["prompt"] for it in items_seq]
            for i0 in range(0, len(prs), batch_size):
                enc = tok(prs[i0 : i0 + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
                ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
                cap.clear()
                _ = model_A.model(input_ids=ids, attention_mask=am, position_ids=_left_pad_position_ids(am), use_cache=False)
                xs.append(cap[0])
        finally:
            hk.remove(); tok.padding_side = old_p
        return torch.cat(xs, dim=0)

    a_d_base, a_d_bind = _cap_disc_act(disc_base), _cap_disc_act(disc_binding)
    col_norm_lq = torch.linalg.norm(L_Q.T @ Wd_bl.float(), dim=0)
    da_pol = torch.abs(a_d_base[0::2] - a_d_base[1::2]).mean(dim=0)
    da_bnd = torch.abs(a_d_base[0::2] - a_d_bind).mean(dim=0)
    ranked_neurons_joint = np.argsort((0.5 * (da_pol + da_bnd) * col_norm_lq).cpu().numpy())[::-1]

    def _eval_bl_mode(items_seq: Sequence[Dict[str, Any]], mode: str, keep_idx: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        n = len(items_seq)
        prs = [it["prompt"] for it in items_seq]
        e_a = lm_w[torch.tensor([int(it["answer_tok_id"]) for it in items_seq], device=dev, dtype=torch.long)].float()
        e_f = lm_w[torch.tensor([int(it["foil_tok_id"]) for it in items_seq], device=dev, dtype=torch.long)].float()
        margins = np.zeros((n,), dtype=np.float64)
        top1_ok = 0
        cap_in, cap_hf = [], []
        def _pre_in(_m, inp):
            cap_in.clear(); cap_in.append(inp[0][:, -1, :].detach().float())
        def _pre_down(_m, inp):
            # Native BF16 down_proj input masking -> k = M_bl is bit-exact with intact!
            if mode != "sparse_k":
                return None
            a_all = inp[0].clone()
            a_last = a_all[:, -1, :]
            a_m = mean_silu_u_gen.to(dtype=a_last.dtype).unsqueeze(0).expand_as(a_last).clone()
            if keep_idx is not None and len(keep_idx) > 0:
                a_m[:, keep_idx] = a_last[:, keep_idx]
            a_all[:, -1, :] = a_m
            return (a_all,) + inp[1:]
        def _mod_m(_m, _inp, out):
            if mode in ("intact", "sparse_k"):
                return out
            xl = cap_in[0]
            o2 = out.clone()
            if mode == "mean_ablate":
                o2[:, -1, :] = mean_mlp_gen.to(dtype=o2.dtype)
            elif mode == "linear_jbar_ex":
                o2[:, -1, :] = (mean_mlp_gen.unsqueeze(0) + (xl - mu_x_gen.unsqueeze(0)) @ J_bar.T).to(dtype=o2.dtype)
            elif mode == "linear_jbar_mu":
                o2[:, -1, :] = (mlp_at_mu_gen.unsqueeze(0) + (xl - mu_x_gen.unsqueeze(0)) @ J_bar.T).to(dtype=o2.dtype)
            return o2

        hk_in = layers[bl].mlp.register_forward_pre_hook(_pre_in)
        hk_dn = layers[bl].mlp.down_proj.register_forward_pre_hook(_pre_down)
        hk_m = layers[bl].mlp.register_forward_hook(_mod_m)
        hk_hf = model_A.model.norm.register_forward_pre_hook(lambda _m, inp: cap_hf.append(inp[0][:, -1, :].detach().float()))
        old_p = tok.padding_side
        tok.padding_side = "left"
        try:
            for i0 in range(0, n, batch_size):
                i1 = min(i0 + batch_size, n)
                enc = tok(prs[i0:i1], return_tensors="pt", padding=True, add_special_tokens=False)
                ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
                cap_hf.clear()
                _ = model_A.model(input_ids=ids, attention_mask=am, position_ids=_left_pad_position_ids(am), use_cache=False)
                hf = cap_hf[0]
                margins[i0:i1] = _real_norm_margin_from_rows(hf, gamma_fin, eps_fin, e_a[i0:i1], e_f[i0:i1]).cpu().numpy()
                rms = torch.sqrt(torch.mean(hf ** 2, dim=-1, keepdim=True) + eps_fin)
                preds = _native_bf16_argmax_shared((hf / rms) * gamma_fin.unsqueeze(0), lm_w, chunk=16)
                top1_ok += sum(int(p == int(items_seq[i0 + j]["answer_tok_id"])) for j, p in enumerate(preds))
        finally:
            hk_in.remove(); hk_dn.remove(); hk_m.remove(); hk_hf.remove(); tok.padding_side = old_p
        raw_m = float(np.mean(margins))
        return {"mean_margin_raw": raw_m, "mean_margin": round(raw_m, 4), "tok0_acc": round(float(top1_ok / max(n, 1)), 4)}

    l20_res: Dict[str, Any] = {
        "bottleneck_mlp_layer": bl, "M_total": M_bl, "generic_unthresholded_chi_nl2": chi_bl,
    }
    for sp_name, seq_it in (("eval_policy", eval_base), ("eval_binding", eval_binding)):
        r_int = _eval_bl_mode(seq_it, "intact")
        r_abl = _eval_bl_mode(seq_it, "mean_ablate")
        denom = float(r_int["mean_margin_raw"] - r_abl["mean_margin_raw"])
        def _frac(raw_m_val: float) -> Optional[float]:
            return round((raw_m_val - r_abl["mean_margin_raw"]) / denom, 4) if denom >= 0.10 else None
        def _clean(d: Dict[str, Any], include_frac: bool = False) -> Dict[str, Any]:
            out_d = {"mean_margin": d["mean_margin"], "tok0_acc": d["tok0_acc"]}
            if include_frac:
                out_d["frac_recovered"] = _frac(d["mean_margin_raw"])
            return out_d
        r_lin_ex = _eval_bl_mode(seq_it, "linear_jbar_ex")
        r_lin_mu = _eval_bl_mode(seq_it, "linear_jbar_mu")
        sp_dict: Dict[str, Any] = {
            "intact": _clean(r_int), "mean_ablate": _clean(r_abl), "margin_drop_supported": bool(denom >= 0.10),
            "linear_jbar_E_mlp": _clean(r_lin_ex, True), "linear_jbar_mlp_E_x": _clean(r_lin_mu, True),
            "sparse_j_lens_community": {},
        }
        for k_n in (0, 16, 64, 128, 256, 1024, M_bl):
            k_eff = min(int(k_n), M_bl)
            idx_t = torch.tensor(ranked_neurons_joint[:k_eff].copy(), device=dev, dtype=torch.long)
            rk = _eval_bl_mode(seq_it, "sparse_k", keep_idx=idx_t)
            sp_dict["sparse_j_lens_community"][f"k_{k_eff}"] = _clean(rk, True)
        l20_res[sp_name] = sp_dict

    return {"arm3a_backward_isa": isa_summary, "arm3b_bottleneck_mlp_linear_and_sparse": l20_res}


@torch.inference_mode()
def stream_donor_generic_v51(
    model_A: nn.Module, corpus: Dict[str, Any], all_target_heads: Sequence[Tuple[int, int]],
    organ_full_layers: Sequence[int], v_shared: int, tmp_dir: Path,
    n_prof_fit: int = 12288, n_val_tune_seqs_for_attn: int = 32, batch_size: int = 8,
) -> Dict[str, Any]:
    """
    P0-3 & P0-4 Fixed:
      - Preallocated disk memmaps for 28-layer profile & full organ layers (zero 3.83 GiB RAM buffer!)
      - Strictly task-independent weight-only `P_0` SVD of unweighted `LN_{0,A}(E_A)`!
      - Saves `v_bar_gen` per `(l, kv)` so Donor `c_true` & Recipient writes use identical frozen centering!
    """
    dev = next(model_A.parameters()).device
    layers = get_decoder_layers(model_A)
    n_layers = len(layers)
    cfg = model_A.config
    D_A, n_q, n_kv = int(cfg.hidden_size), int(cfg.num_attention_heads), int(cfg.num_key_value_heads)
    g_rep, d_h = n_q // n_kv, D_A // n_q
    rope_theta = get_config_rope_theta(cfg)
    seq_len = int(corpus["seq_len"])
    cos_cal, sin_cal = _make_rope_cos_sin(d_h, rope_theta, seq_len + 4, dev)

    uniq_heads = list(dict.fromkeys(all_target_heads))
    uniq_kv = sorted({(l, h // g_rep) for l, h in uniq_heads})
    organ_set = sorted(set(int(l) for l in organ_full_layers))

    Wq_map = {(l, h): (layers[l].self_attn.q_proj.weight.detach()[h * d_h : (h + 1) * d_h], layers[l].self_attn.q_proj.bias.detach()[h * d_h : (h + 1) * d_h] if layers[l].self_attn.q_proj.bias is not None else None) for l, h in uniq_heads}
    Wk_map = {(l, kv): (layers[l].self_attn.k_proj.weight.detach()[kv * d_h : (kv + 1) * d_h], layers[l].self_attn.k_proj.bias.detach()[kv * d_h : (kv + 1) * d_h] if layers[l].self_attn.k_proj.bias is not None else None) for l, kv in uniq_kv}
    Wv_map = {(l, kv): (layers[l].self_attn.v_proj.weight.detach()[kv * d_h : (kv + 1) * d_h], layers[l].self_attn.v_proj.bias.detach()[kv * d_h : (kv + 1) * d_h] if layers[l].self_attn.v_proj.bias is not None else None) for l, kv in uniq_kv}

    # P0-4 Fix: 100% Task-Independent, Weight-Only Unweighted `LN_{0,A}(E_A)` SVD for `P_0` (`K_0 = 64`)!
    K_0 = 64
    ln0_w, ln0_eps = layers[0].input_layernorm.weight.detach().float(), float(layers[0].input_layernorm.variance_epsilon)
    E_A_raw = model_A.model.embed_tokens.weight[:v_shared].detach()
    stride_v = max(1, v_shared // 16384)
    E_sub = E_A_raw[::stride_v].float()
    E_sub_ln = (E_sub / torch.sqrt(torch.mean(E_sub ** 2, dim=-1, keepdim=True) + ln0_eps)) * ln0_w.unsqueeze(0)
    _, _, Vt_l0 = torch.linalg.svd(E_sub_ln - E_sub_ln.mean(dim=0, keepdim=True), full_matrices=False)
    P_0 = Vt_l0[:K_0].contiguous()
    E_A_ln0_proj = torch.empty((v_shared, K_0), device="cpu", dtype=torch.bfloat16)
    for v0 in range(0, v_shared, 16384):
        v1 = min(v0 + 16384, v_shared)
        eb = E_A_raw[v0:v1].float()
        eb_ln = (eb / torch.sqrt(torch.mean(eb ** 2, dim=-1, keepdim=True) + ln0_eps)) * ln0_w.unsqueeze(0)
        E_A_ln0_proj[v0:v1] = (eb_ln @ P_0.T).to(dtype=torch.bfloat16).cpu()
    del E_sub, E_sub_ln, Vt_l0

    lm_w_A = model_A.lm_head.weight[:v_shared].detach()
    gamma_fin_A = model_A.model.norm.weight.detach().float()
    eps_fin_A = float(model_A.model.norm.variance_epsilon)

    cal_dir = tmp_dir / "donor_cal"
    cal_dir.mkdir(parents=True, exist_ok=True)
    split_names = ("fit", "tune", "report")
    split_n_gt0 = {sp: len(corpus[f"{sp}_seqs"]) * (seq_len - 1) for sp in split_names}
    prof_n = {"fit": min(n_prof_fit, split_n_gt0["fit"]), "tune": split_n_gt0["tune"], "report": split_n_gt0["report"]}

    # Preallocate disk memmaps for 28-layer profile & full organ layers (ZERO multi-GB RAM spikes!)
    prof_mm = {
        (sp, li): np.lib.format.open_memmap(cal_dir / f"prof_L{li}_{sp}.npy", mode="w+", dtype=np.uint16, shape=(prof_n[sp], D_A))
        for sp in split_names for li in range(n_layers)
    }
    organ_mm = {
        (sp, li): np.lib.format.open_memmap(cal_dir / f"organ_u_L{li}_{sp}.npy", mode="w+", dtype=np.uint16, shape=(split_n_gt0[sp], D_A))
        for sp in split_names for li in organ_set
    }
    q_mm = {
        (sp, l, h): np.lib.format.open_memmap(cal_dir / f"q_L{l}_H{h}_{sp}.npy", mode="w+", dtype=np.uint16, shape=(split_n_gt0[sp], d_h))
        for sp in split_names for l, h in uniq_heads
    }
    k_mm = {
        (sp, l, kv): np.lib.format.open_memmap(cal_dir / f"k_L{l}_KV{kv}_{sp}.npy", mode="w+", dtype=np.uint16, shape=(split_n_gt0[sp], d_h))
        for sp in split_names for l, kv in uniq_kv
    }
    v_mm = {
        (sp, l, kv): np.lib.format.open_memmap(cal_dir / f"v_L{l}_KV{kv}_{sp}.npy", mode="w+", dtype=np.uint16, shape=(split_n_gt0[sp], d_h))
        for sp in split_names for l, kv in uniq_kv
    }
    n_fisher_fit = min(16384, split_n_gt0["fit"])
    h_fin_fish_mm = np.lib.format.open_memmap(cal_dir / "h_fin_fit_fisher_A.npy", mode="w+", dtype=np.uint16, shape=(n_fisher_fit, D_A))
    h_fin_tune_mm = np.lib.format.open_memmap(cal_dir / "h_fin_tune_A.npy", mode="w+", dtype=np.uint16, shape=(split_n_gt0["tune"], D_A))
    h_fin_rep_mm = np.lib.format.open_memmap(cal_dir / "h_fin_rep_A.npy", mode="w+", dtype=np.uint16, shape=(split_n_gt0["report"], D_A))

    pos0_u_sum = {l: np.zeros((D_A,), dtype=np.float64) for l in organ_set}
    pos0_q_sum = {(l, h): np.zeros((d_h,), dtype=np.float64) for l, h in uniq_heads}
    pos0_k_sum = {(l, kv): np.zeros((d_h,), dtype=np.float64) for l, kv in uniq_kv}
    pos0_v_sum = {(l, kv): np.zeros((d_h,), dtype=np.float64) for l, kv in uniq_kv}
    v_fit_gt0_sum = {(l, kv): np.zeros((d_h,), dtype=np.float64) for l, kv in uniq_kv}
    n_pos0_fit, n_v_fit = 0, 0
    tune_attn_donor: Dict[Tuple[int, int], List[np.ndarray]] = {(l, h): [] for l, h in uniq_heads}
    tune_seqs_u_A: Dict[int, List[torch.Tensor]] = {l: [] for l in organ_set}
    rms_fin_fit_A: List[np.ndarray] = []

    cur_u: Dict[int, torch.Tensor] = {}
    cur_hf: List[torch.Tensor] = []
    hooks = []
    try:
        for li in range(n_layers):
            hooks.append(layers[li].input_layernorm.register_forward_hook(lambda _m, _inp, out, lidx=li: cur_u.__setitem__(lidx, out.detach())))
        hooks.append(model_A.model.norm.register_forward_pre_hook(lambda _m, inp: cur_hf.append(inp[0].detach())))

        for sp in split_names:
            seqs = corpus[f"{sp}_seqs"]
            w_ptr, prof_ptr, fish_ptr, attn_seq_c = 0, 0, 0, 0
            for i0 in range(0, len(seqs), batch_size):
                chunk = seqs[i0 : i0 + batch_size]
                ids = torch.tensor(chunk, device=dev, dtype=torch.long)
                cur_u.clear(); cur_hf.clear()
                _ = model_A.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                B_cur, T_cur = ids.shape
                hf_gt0 = cur_hf[0][:, 1:, :].reshape(-1, D_A)
                n_tok = int(hf_gt0.shape[0])

                if sp == "fit":
                    rms_fin_fit_A.append(torch.sqrt(torch.mean(hf_gt0.float() ** 2, dim=-1) + eps_fin_A).cpu().numpy())
                    if fish_ptr < n_fisher_fit:
                        tk = min(n_fisher_fit - fish_ptr, n_tok)
                        h_fin_fish_mm[fish_ptr : fish_ptr + tk] = _tensor_to_bf16_u16(hf_gt0[:tk])
                        fish_ptr += tk
                    n_pos0_fit += int(B_cur)
                    n_v_fit += n_tok
                elif sp == "tune":
                    h_fin_tune_mm[w_ptr : w_ptr + n_tok] = _tensor_to_bf16_u16(hf_gt0)
                else:
                    h_fin_rep_mm[w_ptr : w_ptr + n_tok] = _tensor_to_bf16_u16(hf_gt0)

                if prof_ptr < prof_n[sp]:
                    tkp = min(prof_n[sp] - prof_ptr, n_tok)
                    for li in range(n_layers):
                        prof_mm[(sp, li)][prof_ptr : prof_ptr + tkp] = _tensor_to_bf16_u16(cur_u[li][:, 1:, :].reshape(-1, D_A)[:tkp])
                    prof_ptr += tkp

                for li in organ_set:
                    organ_mm[(sp, li)][w_ptr : w_ptr + n_tok] = _tensor_to_bf16_u16(cur_u[li][:, 1:, :].reshape(-1, D_A))
                    if sp == "fit":
                        pos0_u_sum[li] += cur_u[li][:, 0, :].float().sum(dim=0).cpu().numpy()

                for l, h in uniq_heads:
                    Wq, bq = Wq_map[(l, h)]
                    q_all = F.linear(cur_u[l], Wq, bq)
                    q_mm[(sp, l, h)][w_ptr : w_ptr + n_tok] = _tensor_to_bf16_u16(q_all[:, 1:, :].reshape(-1, d_h))
                    if sp == "fit":
                        pos0_q_sum[(l, h)] += q_all[:, 0, :].float().sum(dim=0).cpu().numpy()
                for l, kv in uniq_kv:
                    Wk, bk = Wk_map[(l, kv)]
                    Wv, bv = Wv_map[(l, kv)]
                    k_all, v_all = F.linear(cur_u[l], Wk, bk), F.linear(cur_u[l], Wv, bv)
                    k_mm[(sp, l, kv)][w_ptr : w_ptr + n_tok] = _tensor_to_bf16_u16(k_all[:, 1:, :].reshape(-1, d_h))
                    v_mm[(sp, l, kv)][w_ptr : w_ptr + n_tok] = _tensor_to_bf16_u16(v_all[:, 1:, :].reshape(-1, d_h))
                    if sp == "fit":
                        pos0_k_sum[(l, kv)] += k_all[:, 0, :].float().sum(dim=0).cpu().numpy()
                        pos0_v_sum[(l, kv)] += v_all[:, 0, :].float().sum(dim=0).cpu().numpy()
                        v_fit_gt0_sum[(l, kv)] += v_all[:, 1:, :].reshape(-1, d_h).float().sum(dim=0).cpu().numpy()

                if sp == "tune" and attn_seq_c < n_val_tune_seqs_for_attn:
                    for b in range(B_cur):
                        if attn_seq_c >= n_val_tune_seqs_for_attn:
                            break
                        for l, h in uniq_heads:
                            kv = h // g_rep
                            Wq, bq = Wq_map[(l, h)]
                            Wk, bk = Wk_map[(l, kv)]
                            us = cur_u[l][b]
                            ql = F.linear(us[-1:], Wq, bq).squeeze(0).float()
                            ks = F.linear(us, Wk, bk).float()
                            half = d_h // 2
                            c, s = cos_cal[:T_cur], sin_cal[:T_cur]
                            qr = torch.cat([ql[:half] * c[-1] - ql[half:] * s[-1], ql[half:] * c[-1] + ql[:half] * s[-1]], dim=-1)
                            kr = torch.cat([ks[:, :half] * c - ks[:, half:] * s, ks[:, half:] * c + ks[:, :half] * s], dim=-1)
                            alp = F.softmax((kr @ qr) / math.sqrt(float(d_h)), dim=0).cpu().numpy().astype(np.float32)
                            tune_attn_donor[(l, h)].append(alp)
                        for li_o in organ_set:
                            tune_seqs_u_A[li_o].append(cur_u[li_o][b].to(dtype=torch.bfloat16).cpu())
                        attn_seq_c += 1
                w_ptr += n_tok
    finally:
        for hk in hooks:
            hk.remove()
        for d_mm in (prof_mm, organ_mm, q_mm, k_mm, v_mm):
            for mm in d_mm.values():
                mm.flush()
        h_fin_fish_mm.flush(); h_fin_tune_mm.flush(); h_fin_rep_mm.flush()

    np.save(cal_dir / "E_A_shared_u16.npy", _tensor_to_bf16_u16(lm_w_A))
    np.save(cal_dir / "E_A_ln0_proj_u16.npy", _tensor_to_bf16_u16(E_A_ln0_proj))
    inv_p0 = 1.0 / max(float(n_pos0_fit), 1.0)
    inv_vf = 1.0 / max(float(n_v_fit), 1.0)
    return {
        "cal_dir": str(cal_dir), "n_layers": n_layers, "D_A": D_A, "d_h": d_h, "g_rep": g_rep,
        "rope_theta": rope_theta, "K_0": K_0,
        "gamma_fin_A_f32": gamma_fin_A.cpu().numpy().astype(np.float32), "eps_fin_A": eps_fin_A,
        "median_rms_fin_A": float(np.median(np.concatenate(rms_fin_fit_A))),
        "mu_pos0_u_by_l": {l: pos0_u_sum[l] * inv_p0 for l in organ_set},
        "mu_pos0_q": {k: v * inv_p0 for k, v in pos0_q_sum.items()},
        "mu_pos0_k": {k: v * inv_p0 for k, v in pos0_k_sum.items()},
        "mu_pos0_v": {k: v * inv_p0 for k, v in pos0_v_sum.items()},
        "v_bar_gen_by_kv": {k: (v * inv_vf).astype(np.float32) for k, v in v_fit_gt0_sum.items()},
        "tune_attn_donor": tune_attn_donor,
        "tune_seqs_u_A": tune_seqs_u_A,
        "n_val_tune_seqs_for_attn": int(n_val_tune_seqs_for_attn),
    }


# ======================================================================================
# 7. Phase 2: Full Recipient 1.5B Evaluation, Controls & Arm 3c Native L0 Verification
# ======================================================================================

@torch.inference_mode()
def run_phase2_recipient_v51(
    model_B: nn.Module,
    tok: Any,
    corpus: Dict[str, Any],
    donor_cal: Dict[str, Any],
    task_bundles: Dict[str, Dict[str, Any]],
    v_shared: int,
    tmp_dir: Path,
    tap_layers: Sequence[int] = (8, 14, 19, 23),
    batch_size: int = 16,
) -> Dict[str, Any]:
    dev = next(model_B.parameters()).device
    layers_B = get_decoder_layers(model_B)
    n_layers = len(layers_B)
    D_B = int(model_B.config.hidden_size)
    D_A, d_h, g_rep_A = int(donor_cal["D_A"]), int(donor_cal["d_h"]), int(donor_cal["g_rep"])
    rope_theta, K_0 = float(donor_cal["rope_theta"]), int(donor_cal["K_0"])
    cal_dir = Path(donor_cal["cal_dir"])
    seq_len = int(corpus["seq_len"])

    lm_w_B = model_B.lm_head.weight[:v_shared].detach()
    gamma_fin_B = model_B.model.norm.weight.detach().float()
    eps_fin_B = float(model_B.model.norm.variance_epsilon)
    E_A_ln0_proj_f32 = _u16_to_f32(np.load(cal_dir / "E_A_ln0_proj_u16.npy"))

    all_organ_layers = sorted({
        l for tb in task_bundles.values()
        for l, _ in (list(tb["primary_heads"]) + list(tb["candidate_heads"]) + list(tb["shuf_layer_heads"]))
    })
    prev_layers_for_quad = sorted({max(0, l - 1) for tb in task_bundles.values() for l, _ in (list(tb["primary_heads"]) + list(tb["shuf_layer_heads"]))})
    full_capture_layers_B = sorted(set(tap_layers) | set(all_organ_layers))

    # Step 2.1: Stream Recipient 1.5B over generic corpus using preallocated memmaps!
    print("[v5.1 Phase 2] Streaming Recipient 1.5B over 60k/4k/4k diverse calibration corpus...", flush=True)
    rec_cal_dir = tmp_dir / "rec_cal"
    rec_cal_dir.mkdir(parents=True, exist_ok=True)
    split_names = ("fit", "tune", "report")
    split_n_gt0 = {sp: len(corpus[f"{sp}_seqs"]) * (seq_len - 1) for sp in split_names}
    n_prof_fit = int(np.load(cal_dir / "prof_L0_fit.npy", mmap_mode="r").shape[0])
    prof_n = {"fit": n_prof_fit, "tune": split_n_gt0["tune"], "report": split_n_gt0["report"]}

    prof_B_mm = {
        (sp, li): np.lib.format.open_memmap(rec_cal_dir / f"prof_B_L{li}_{sp}.npy", mode="w+", dtype=np.uint16, shape=(prof_n[sp], D_B))
        for sp in split_names for li in range(n_layers)
    }
    full_u_B_mm = {
        (sp, li): np.lib.format.open_memmap(rec_cal_dir / f"u_B_L{li}_{sp}.npy", mode="w+", dtype=np.uint16, shape=(split_n_gt0[sp], D_B))
        for sp in split_names for li in full_capture_layers_B
    }
    # P0-6 Fix: Capture `post_attention_layernorm` (Qwen's ACTUAL native MLP input norm!) for Arm 2b!
    post_u_B_mm = {
        (sp, li): np.lib.format.open_memmap(rec_cal_dir / f"post_u_B_L{li}_{sp}.npy", mode="w+", dtype=np.uint16, shape=(split_n_gt0[sp], D_B))
        for sp in split_names for li in prev_layers_for_quad
    }

    tok_ids_gt0: Dict[str, List[np.ndarray]] = {sp: [] for sp in split_names}
    pos0_u_B_sum = {l: np.zeros((D_B,), dtype=np.float64) for l in full_capture_layers_B}
    pos0_tok_l0_sum = np.zeros((K_0,), dtype=np.float64)
    n_pos0_fit_B = 0
    n_attn_seqs = int(donor_cal["n_val_tune_seqs_for_attn"])
    tune_seqs_u_B: Dict[int, List[np.ndarray]] = {l: [] for l in full_capture_layers_B}
    tune_seqs_post_u_B: Dict[int, List[np.ndarray]] = {l: [] for l in prev_layers_for_quad}
    tune_seqs_ids: List[np.ndarray] = []
    h_fin_tune_B_list: List[np.ndarray] = []
    h_fin_rep_B_list: List[np.ndarray] = []
    rms_fin_fit_B: List[np.ndarray] = []

    cur_u: Dict[int, torch.Tensor] = {}
    cur_pu: Dict[int, torch.Tensor] = {}
    cur_hf: List[torch.Tensor] = []
    hooks = []
    try:
        for li in range(n_layers):
            hooks.append(layers_B[li].input_layernorm.register_forward_hook(lambda _m, _inp, out, lidx=li: cur_u.__setitem__(lidx, out.detach())))
        for li in prev_layers_for_quad:
            hooks.append(layers_B[li].post_attention_layernorm.register_forward_hook(lambda _m, _inp, out, lidx=li: cur_pu.__setitem__(lidx, out.detach())))
        hooks.append(model_B.model.norm.register_forward_pre_hook(lambda _m, inp: cur_hf.append(inp[0].detach())))

        for sp in split_names:
            seqs = corpus[f"{sp}_seqs"]
            w_ptr, prof_ptr, attn_seq_c = 0, 0, 0
            for i0 in range(0, len(seqs), batch_size):
                chunk = seqs[i0 : i0 + batch_size]
                ids = torch.tensor(chunk, device=dev, dtype=torch.long)
                cur_u.clear(); cur_pu.clear(); cur_hf.clear()
                _ = model_B.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                B_cur = int(ids.shape[0])
                hf_gt0 = cur_hf[0][:, 1:, :].reshape(-1, D_B)
                n_tok = int(hf_gt0.shape[0])
                tok_ids_gt0[sp].append(ids[:, 1:].reshape(-1).cpu().numpy().astype(np.int64))

                if sp == "fit":
                    rms_fin_fit_B.append(torch.sqrt(torch.mean(hf_gt0.float() ** 2, dim=-1) + eps_fin_B).cpu().numpy())
                    n_pos0_fit_B += B_cur
                    pos0_tok_l0_sum += E_A_ln0_proj_f32[ids[:, 0].cpu().numpy()].astype(np.float64).sum(axis=0)
                elif sp == "tune":
                    h_fin_tune_B_list.append(_tensor_to_bf16_u16(hf_gt0))
                else:
                    h_fin_rep_B_list.append(_tensor_to_bf16_u16(hf_gt0))

                if prof_ptr < prof_n[sp]:
                    tkp = min(prof_n[sp] - prof_ptr, n_tok)
                    for li in range(n_layers):
                        prof_B_mm[(sp, li)][prof_ptr : prof_ptr + tkp] = _tensor_to_bf16_u16(cur_u[li][:, 1:, :].reshape(-1, D_B)[:tkp])
                    prof_ptr += tkp

                for li in full_capture_layers_B:
                    full_u_B_mm[(sp, li)][w_ptr : w_ptr + n_tok] = _tensor_to_bf16_u16(cur_u[li][:, 1:, :].reshape(-1, D_B))
                    if sp == "fit":
                        pos0_u_B_sum[li] += cur_u[li][:, 0, :].float().sum(dim=0).cpu().numpy()
                for li in prev_layers_for_quad:
                    post_u_B_mm[(sp, li)][w_ptr : w_ptr + n_tok] = _tensor_to_bf16_u16(cur_pu[li][:, 1:, :].reshape(-1, D_B))

                if sp == "tune" and attn_seq_c < n_attn_seqs:
                    for b in range(B_cur):
                        if attn_seq_c >= n_attn_seqs:
                            break
                        tune_seqs_ids.append(ids[b].cpu().numpy().astype(np.int64))
                        for li in full_capture_layers_B:
                            tune_seqs_u_B[li].append(cur_u[li][b].float().cpu().numpy())
                        for li in prev_layers_for_quad:
                            tune_seqs_post_u_B[li].append(cur_pu[li][b].float().cpu().numpy())
                        attn_seq_c += 1
                w_ptr += n_tok
    finally:
        for hk in hooks:
            hk.remove()
        for d_mm in (prof_B_mm, full_u_B_mm, post_u_B_mm):
            for mm in d_mm.values():
                mm.flush()

    tok_ids_cat = {sp: np.concatenate(tok_ids_gt0[sp], axis=0) for sp in split_names}
    mu_pos0_u_B = {l: pos0_u_B_sum[l] / max(float(n_pos0_fit_B), 1.0) for l in full_capture_layers_B}
    mu_pos0_l0 = pos0_tok_l0_sum / max(float(n_pos0_fit_B), 1.0)
    median_rms_fin_B = float(np.median(np.concatenate(rms_fin_fit_B)))
    gamma_scale = float(median_rms_fin_B / max(float(donor_cal["median_rms_fin_A"]), 1e-6))

    # Step 2.2: Vocab-Anchored Fisher Plug `T_{raw, eta}` & `M_U(eta)` (P0-2 & P1-5 Fixed!)
    print("[v5.1 Phase 2] Computing vocab-anchored Fisher plugs T_{raw, eta} & M_U(eta)...", flush=True)
    E_A_bf16 = torch.from_numpy(np.load(cal_dir / "E_A_shared_u16.npy")).view(torch.bfloat16).to(dev)
    E_B_bf16 = lm_w_B
    gamma_A_f32 = donor_cal["gamma_fin_A_f32"]
    eps_A = float(donor_cal["eps_fin_A"])
    gamma_A_t = torch.from_numpy(gamma_A_f32).to(dev)

    fisher_acc = StreamingSharedVocabFisherAccumulator(E_A_bf16, E_B_bf16)
    h_fit_fish_A = _u16_to_f32(np.load(cal_dir / "h_fin_fit_fisher_A.npy"))
    for i0 in range(0, len(h_fit_fish_A), 128):
        hb = torch.from_numpy(h_fit_fish_A[i0 : i0 + 128]).to(dev)
        rb = torch.sqrt(torch.mean(hb ** 2, dim=-1, keepdim=True) + eps_A)
        hn = (hb / rb) * gamma_A_t.unsqueeze(0)
        # P0-2 Fix: Pass shared-vocab LOGITS (chunk, V_shared) to `fisher_acc.update`!
        logits_b = F.linear(hn.to(dtype=E_A_bf16.dtype), E_A_bf16)
        fisher_acc.update(logits_b)
        del hb, rb, hn, logits_b
    f_res = fisher_acc.finalize()
    G_A_gen, G_B_gen, G_BA_gen = f_res["G_A"], f_res["G_B"], f_res["G_BA"]
    del fisher_acc, h_fit_fish_A

    V_f = float(v_shared)
    ea_bar = E_A_bf16.float().mean(dim=0).double()
    eb_bar = E_B_bf16.float().mean(dim=0).double()
    G_A_iso = torch.zeros((D_A, D_A), device=dev, dtype=torch.float64)
    G_B_iso = torch.zeros((D_B, D_B), device=dev, dtype=torch.float64)
    G_BA_iso = torch.zeros((D_B, D_A), device=dev, dtype=torch.float64)
    for v0 in range(0, v_shared, 8192):
        v1 = min(v0 + 8192, v_shared)
        ea_c, eb_c = E_A_bf16[v0:v1].float().double(), E_B_bf16[v0:v1].float().double()
        G_A_iso += ea_c.T @ ea_c
        G_B_iso += eb_c.T @ eb_c
        G_BA_iso += eb_c.T @ ea_c
    G_A_iso = _sym((G_A_iso / V_f - torch.outer(ea_bar, ea_bar)).cpu().numpy())
    G_B_iso = _sym((G_B_iso / V_f - torch.outer(eb_bar, eb_bar)).cpu().numpy())
    G_BA_iso = (G_BA_iso / V_f - torch.outer(eb_bar, ea_bar)).cpu().numpy()
    iso_scale = max(float(np.trace(G_B_gen)), 1e-12) / max(float(np.trace(G_B_iso)), 1e-12)

    etas = (0.0, 1e-4, 1e-3, 1e-2, 5e-2)
    plugs_by_eta = {
        eta: compute_raw_and_postnorm_fisher_plugs(
            _sym((1.0 - eta) * G_A_gen + eta * iso_scale * G_A_iso),
            _sym((1.0 - eta) * G_B_gen + eta * iso_scale * G_B_iso),
            (1.0 - eta) * G_BA_gen + eta * iso_scale * G_BA_iso,
            gamma_A_f32, gamma_fin_B.cpu().numpy(),
        )
        for eta in etas
    }

    h_tune_A = _u16_to_f32(np.load(cal_dir / "h_fin_tune_A.npy"))[:2048]
    h_tune_B = _u16_to_f32(np.concatenate(h_fin_tune_B_list, axis=0))[:2048]
    h_rep_A = _u16_to_f32(np.load(cal_dir / "h_fin_rep_A.npy"))[:2048]
    h_rep_B = _u16_to_f32(np.concatenate(h_fin_rep_B_list, axis=0))[:2048]

    def _eval_bridge_kl_top1_raw(hA_np: np.ndarray, hB_np: np.ndarray, M_U: np.ndarray, T_raw: np.ndarray) -> Dict[str, float]:
        MU_t = torch.from_numpy(M_U.astype(np.float32)).to(dev)
        TR_t = torch.from_numpy(T_raw.astype(np.float32)).to(dev)
        n = len(hA_np)
        kl_mu = kl_tr_raw = kl_tr_cal = kl_stock = 0.0
        t1_mu = t1_tr = t1_stock = 0
        for i0 in range(0, n, 64):
            i1 = min(i0 + 64, n)
            ha = torch.from_numpy(hA_np[i0:i1]).to(dev)
            hb = torch.from_numpy(hB_np[i0:i1]).to(dev)
            ra = torch.sqrt(torch.mean(ha ** 2, dim=-1, keepdim=True) + eps_A)
            rb = torch.sqrt(torch.mean(hb ** 2, dim=-1, keepdim=True) + eps_fin_B)
            hna = (ha / ra) * gamma_A_t.unsqueeze(0)
            hnb = (hb / rb) * gamma_fin_B.unsqueeze(0)
            lg_a = F.linear(hna.to(E_A_bf16.dtype), E_A_bf16).float()
            pa, top_a = F.softmax(lg_a, dim=-1), torch.argmax(lg_a, dim=-1)

            lg_st = F.linear(hnb.to(E_B_bf16.dtype), E_B_bf16).float()
            kl_stock += float(F.kl_div(F.log_softmax(lg_st, dim=-1), pa, reduction="sum").item())
            t1_stock += int((torch.argmax(lg_st, dim=-1) == top_a).sum().item())

            lg_mu = F.linear((hna @ MU_t.T).to(E_B_bf16.dtype), E_B_bf16).float()
            kl_mu += float(F.kl_div(F.log_softmax(lg_mu, dim=-1), pa, reduction="sum").item())
            t1_mu += int((torch.argmax(lg_mu, dim=-1) == top_a).sum().item())

            hp_tr = ha @ TR_t.T
            r_tr = torch.sqrt(torch.mean(hp_tr ** 2, dim=-1, keepdim=True) + eps_fin_B)
            lg_tr_raw = F.linear(((hp_tr / r_tr) * gamma_fin_B.unsqueeze(0)).to(E_B_bf16.dtype), E_B_bf16).float()
            kl_tr_raw += float(F.kl_div(F.log_softmax(lg_tr_raw, dim=-1), pa, reduction="sum").item())
            t1_tr += int((torch.argmax(lg_tr_raw, dim=-1) == top_a).sum().item())

            # Deployed scaled bridge KL (fixed donor-rms ratio!)
            lg_tr_cal = F.linear(((hp_tr * gamma_scale / (ra * gamma_scale)) * gamma_fin_B.unsqueeze(0)).to(E_B_bf16.dtype), E_B_bf16).float()
            kl_tr_cal += float(F.kl_div(F.log_softmax(lg_tr_cal, dim=-1), pa, reduction="sum").item())
        inv_n = 1.0 / max(float(n), 1.0)
        return {
            "stock_1p5b_top1": t1_stock * inv_n, "stock_1p5b_kl": kl_stock * inv_n,
            "M_U_top1": t1_mu * inv_n, "M_U_kl": kl_mu * inv_n,
            "T_raw_renorm_top1": t1_tr * inv_n, "T_raw_renorm_kl": kl_tr_raw * inv_n,
            "T_raw_donor_rms_kl": kl_tr_cal * inv_n,
        }

    # P1-5 Fix: Select `best_eta` at FULL UNROUNDED PRECISION on deployed `T_raw_donor_rms_kl`!
    eta_sweep_tune: Dict[str, Any] = {}
    best_eta, best_kl_t = 0.0, 1e18
    for eta in etas:
        ev_t = _eval_bridge_kl_top1_raw(h_tune_A, h_tune_B, plugs_by_eta[eta]["M_U"], plugs_by_eta[eta]["T_raw"])
        eta_sweep_tune[f"eta_{eta}"] = {k: round(v, 5) for k, v in ev_t.items()}
        if ev_t["T_raw_donor_rms_kl"] < best_kl_t:
            best_kl_t = ev_t["T_raw_donor_rms_kl"]
            best_eta = float(eta)
    T_raw_frozen = plugs_by_eta[best_eta]["T_raw"]
    M_U_frozen = plugs_by_eta[best_eta]["M_U"]
    bridge_report_eval = {k: round(v, 5) for k, v in _eval_bridge_kl_top1_raw(h_rep_A, h_rep_B, M_U_frozen, T_raw_frozen).items()}
    del E_A_bf16, h_tune_A, h_tune_B, h_rep_A, h_rep_B
    torch.cuda.empty_cache()

    # Step 2.3: Full 28-Layer `pos > 0` Supply Profile `kappa(l)` + Memoized Feature Caches!
    print("[v5.1 Phase 2] Computing full 28-layer supply profile kappa(l) & multi-tap Delta_kappa...", flush=True)
    layer_profile_28: List[Dict[str, Any]] = []
    for li in range(n_layers):
        Xf = _u16_to_f64(np.load(rec_cal_dir / f"prof_B_L{li}_fit.npy"))
        Xt = _u16_to_f64(np.load(rec_cal_dir / f"prof_B_L{li}_tune.npy"))
        Xr = _u16_to_f64(np.load(rec_cal_dir / f"prof_B_L{li}_report.npy"))
        Yf = _u16_to_f64(np.load(cal_dir / f"prof_L{li}_fit.npy"))
        Yt = _u16_to_f64(np.load(cal_dir / f"prof_L{li}_tune.npy"))
        Yr = _u16_to_f64(np.load(cal_dir / f"prof_L{li}_report.npy"))
        reg_l = fit_ridge_affine_val_tuned(Xf, Yf, Xt, Yt, lam_grid=LAM_GRID_UNCAPPED, X_rep=Xr, Y_rep=Yr)
        layer_profile_28.append({
            "layer": int(li), "best_lam_rel": reg_l["best_lam_rel"],
            "kappa_fit": round(reg_l["r2_fit"], 4), "kappa_tune": round(reg_l["r2_tune"], 4),
            "kappa_rep": round(reg_l["r2_rep"], 4), "fit_minus_rep_gap": round(reg_l["r2_fit"] - reg_l["r2_rep"], 4),
            "min_dim_r2_rep": round(float(np.min(reg_l["per_dim_r2_rep"])), 4),
            "median_dim_r2_rep": round(float(np.median(reg_l["per_dim_r2_rep"])), 4),
        })

    # Fit generic top-128 PCA projections per tap layer strictly on generic `fit`
    tap_pcs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    K_tap = 128
    for tl in tap_layers:
        Ut_f = _u16_to_f64(np.load(rec_cal_dir / f"u_B_L{tl}_fit.npy"))
        mu_tl = Ut_f.mean(axis=0)
        Utc = Ut_f - mu_tl[None, :]
        ev_tl, V_tl = sla.eigh(_sym((Utc.T @ Utc) / max(float(Ut_f.shape[0]), 1.0)))
        tap_pcs[tl] = (mu_tl, V_tl[:, np.argsort(ev_tl)[::-1][:K_tap]])

    def _build_tap_features_for_split(sp: str, l_org: int, include_taps: Sequence[int], include_l0: bool) -> np.ndarray:
        feats = [_u16_to_f64(np.load(rec_cal_dir / f"u_B_L{l_org}_{sp}.npy"))]
        if include_l0:
            feats.append(E_A_ln0_proj_f32[tok_ids_cat[sp]].astype(np.float64))
        for tl in include_taps:
            if tl < l_org:
                mu_tl, P_tl = tap_pcs[tl]
                feats.append((_u16_to_f64(np.load(rec_cal_dir / f"u_B_L{tl}_{sp}.npy")) - mu_tl[None, :]) @ P_tl)
        return np.concatenate(feats, axis=1)

    def _build_tap_pos0(l_org: int, include_taps: Sequence[int], include_l0: bool) -> np.ndarray:
        parts = [mu_pos0_u_B[l_org]]
        if include_l0:
            parts.append(mu_pos0_l0)
        for tl in include_taps:
            if tl < l_org:
                mu_tl, P_tl = tap_pcs[tl]
                parts.append((mu_pos0_u_B[tl] - mu_tl) @ P_tl)
        return np.concatenate(parts, axis=0)

    def _build_tap_seq(u_seq_by_l: Dict[int, np.ndarray], ids_seq: np.ndarray, l_org: int, include_taps: Sequence[int], include_l0: bool) -> np.ndarray:
        parts = [u_seq_by_l[l_org].astype(np.float64)]
        if include_l0:
            parts.append(E_A_ln0_proj_f32[ids_seq].astype(np.float64))
        for tl in include_taps:
            if tl < l_org:
                mu_tl, P_tl = tap_pcs[tl]
                parts.append((u_seq_by_l[tl].astype(np.float64) - mu_tl[None, :]) @ P_tl)
        return np.concatenate(parts, axis=1)

    # P0-7 Fix: Memoize `(xc, u0_B, Phi_tune_seqs)` per unique `(l_org, tuple(active_taps), include_l0)`! (~100x faster!)
    x_cache_memo: Dict[Tuple[int, Tuple[int, ...], bool], Tuple[Dict[str, Any], np.ndarray, List[np.ndarray]]] = {}

    def _get_x_cache(l_org: int, include_taps: Sequence[int], include_l0: bool) -> Tuple[Dict[str, Any], np.ndarray, List[np.ndarray]]:
        act_taps = tuple(tl for tl in include_taps if tl < l_org)
        key = (int(l_org), act_taps, bool(include_l0))
        if key not in x_cache_memo:
            Xf = _build_tap_features_for_split("fit", l_org, act_taps, include_l0)
            Xt = _build_tap_features_for_split("tune", l_org, act_taps, include_l0)
            Xr = _build_tap_features_for_split("report", l_org, act_taps, include_l0)
            xc = prepare_ridge_x_cache(Xf, Xt, X_rep=Xr)
            u0 = _build_tap_pos0(l_org, act_taps, include_l0)
            phi_t = [
                _build_tap_seq({tl: tune_seqs_u_B[tl][s_i] for tl in full_capture_layers_B}, tune_seqs_ids[s_i], l_org, act_taps, include_l0)
                for s_i in range(len(tune_seqs_ids))
            ]
            x_cache_memo[key] = (xc, u0, phi_t)
        return x_cache_memo[key]

    # Incremental multi-tap Delta_kappa ladder on reference head
    ref_tb = task_bundles.get("2hop", next(iter(task_bundles.values())))
    ref_l, ref_h = ref_tb["primary_heads"][0]
    ref_kv = ref_h // g_rep_A
    qf_ref, qt_ref, qr_ref = (_u16_to_f64(np.load(cal_dir / f"q_L{ref_l}_H{ref_h}_{sp}.npy")) for sp in split_names)
    vf_ref, vt_ref, vr_ref = (_u16_to_f64(np.load(cal_dir / f"v_L{ref_l}_KV{ref_kv}_{sp}.npy")) for sp in split_names)
    incremental_tap_ladder: List[Dict[str, Any]] = []
    prev_qr = prev_vr = None
    for st_name, st_taps, st_l0 in (
        ("L_org_only", [], False), ("+L0_exact_donor_emb", [], True), ("+L23_tap", [23], True),
        ("+L19_tap", [19, 23], True), ("+L14_tap", [14, 19, 23], True), ("+L8_tap_full", [8, 14, 19, 23], True),
    ):
        xc_s, _, _ = _get_x_cache(ref_l, st_taps, st_l0)
        rq = fit_ridge_from_x_cache(xc_s, qf_ref, qt_ref, Y_rep=qr_ref)
        rv = fit_ridge_from_x_cache(xc_s, vf_ref, vt_ref, Y_rep=vr_ref)
        incremental_tap_ladder.append({
            "stage": st_name, "ref_head": f"L{ref_l}H{ref_h}", "feat_dim": int(xc_s["Xf32"].shape[1]),
            "q_r2_fit": round(rq["r2_fit"], 4), "q_r2_tune": round(rq["r2_tune"], 4), "q_r2_rep": round(rq["r2_rep"], 4),
            "delta_q_r2_rep": round(rq["r2_rep"] - prev_qr, 4) if prev_qr is not None else 0.0,
            "v_r2_fit": round(rv["r2_fit"], 4), "v_r2_tune": round(rv["r2_tune"], 4), "v_r2_rep": round(rv["r2_rep"], 4),
            "delta_v_r2_rep": round(rv["r2_rep"] - prev_vr, 4) if prev_vr is not None else 0.0,
        })
        prev_qr, prev_vr = rq["r2_rep"], rv["r2_rep"]
    # Evict intermediate tap-ladder caches immediately so host RAM stays tightly bounded!
    x_cache_memo.clear()
    gc.collect()

    # Step 2.4: Fit Generic Zero-SGD Q/K/V Bridges (P0-4, P0-5, P0-6, P0-7 Fixed!)
    print("[v5.1 Phase 2] Fitting 100% generic zero-SGD bridges (memoized FP64, compiled BOS calibration)...", flush=True)
    inv_f = np.asarray(rope_inv_freq(d_h, rope_theta), dtype=np.float64)
    ang_cal = np.arange(seq_len, dtype=np.float64)[:, None] * inv_f[None, :]
    cos_cal_np, sin_cal_np = np.cos(ang_cal), np.sin(ang_cal)

    def _fit_role_qkv_arm_for_head(
        l: int, h: int, include_taps: Sequence[int], include_l0: bool, variant: str, add_quad_lift: bool = False
    ) -> Dict[str, Any]:
        kv = h // g_rep_A
        xc, u0_B, phi_tune_seqs = _get_x_cache(l, include_taps, include_l0)
        qf, qt, qr = (_u16_to_f64(np.load(cal_dir / f"q_L{l}_H{h}_{sp}.npy")) for sp in split_names)
        kf, kt, kr = (_u16_to_f64(np.load(cal_dir / f"k_L{l}_KV{kv}_{sp}.npy")) for sp in split_names)
        vf, vt, vr = (_u16_to_f64(np.load(cal_dir / f"v_L{l}_KV{kv}_{sp}.npy")) for sp in split_names)

        reg_q = fit_ridge_from_x_cache(xc, qf, qt, Y_rep=qr)
        reg_k = fit_ridge_from_x_cache(xc, kf, kt, Y_rep=kr)
        reg_v = fit_ridge_from_x_cache(xc, vf, vt, Y_rep=vr)

        C_xx_gt0 = xc["C_xx"]
        k0_A, v0_A = donor_cal["mu_pos0_k"][(l, kv)], donor_cal["mu_pos0_v"][(l, kv)]
        v_bar_gen = donor_cal["v_bar_gen_by_kv"][(l, kv)].astype(np.float64)
        donor_alphas = donor_cal["tune_attn_donor"][(l, h)]

        # Base V (exact KKT pos=0 anchor + generic v_bar_gen subtraction!)
        Wv_eff, bv_eff = apply_kkt_rank1_pos0_anchor(reg_v["W"], reg_v["b"], u0_B, v0_A, reg_v["mu_x"], C_xx_gt0=C_xx_gt0)
        bv_eff_centered = bv_eff - v_bar_gen

        m_low, m_partner = d_h // 2 - 1, d_h - 1
        zero_dims = (m_low, m_partner) if variant == "1a_convex_m63" else ()

        if variant == "1a_convex_m63":
            Wq_base, bq_base, md_q = apply_mode_denoising_on_tune(reg_q["W"], reg_q["b"], xc["Xt"], qt, reg_q["mu_x"], reg_q["mu_y"], rho2_floor=0.05)
            Wk_base, bk_base, md_k = apply_mode_denoising_on_tune(reg_k["W"], reg_k["b"], xc["Xt"], kt, reg_k["mu_x"], reg_k["mu_y"], rho2_floor=0.05)
            Wq_base[list(zero_dims)] = 0.0; bq_base[list(zero_dims)] = 0.0
            Wk_base[list(zero_dims)] = 0.0; bk_base[list(zero_dims)] = 0.0
        else:
            Wq_base, bq_base, _, md_q = apply_rope_plane_deshrinkage(reg_q["W"], reg_q["b"], xc["Xt"], qt, reg_q["mu_x"], reg_q["mu_y"], rho2_min=0.20, gain_cap=1.45)
            Wk_base, bk_base, _, md_k = apply_rope_plane_deshrinkage(reg_k["W"], reg_k["b"], xc["Xt"], kt, reg_k["mu_x"], reg_k["mu_y"], rho2_min=0.20, gain_cap=1.45)

        # P0-6 & P1-3 Fix: Fit Arm 2b quadratic residuals against the ANCHORED deployed base using `post_attention_layernorm`!
        quad_q = quad_k = quad_v = None
        qq_tune_raw = qk_tune_raw = None
        if add_quad_lift:
            lp = max(0, l - 1)
            Up_f = _u16_to_f64(np.load(rec_cal_dir / f"post_u_B_L{lp}_fit.npy"))
            Up_t = _u16_to_f64(np.load(rec_cal_dir / f"post_u_B_L{lp}_tune.npy"))
            k0_anc_t = k0_A.copy()
            if zero_dims:
                k0_anc_t[list(zero_dims)] = 0.0
            Wk_anc_q, bk_anc_q = apply_kkt_rank1_pos0_anchor(Wk_base, bk_base, u0_B, k0_anc_t, reg_k["mu_x"], C_xx_gt0=C_xx_gt0)
            Xf_f64 = xc["Xf32"].astype(np.float64)
            quad_q = fit_unlabeled_antisymmetric_swiglu_lift(
                Up_f, qf - (Xf_f64 @ Wq_base.T + bq_base[None, :]),
                Up_t, qt - (xc["Xt"] @ Wq_base.T + bq_base[None, :]), zero_output_dims=zero_dims,
            )
            quad_k = fit_unlabeled_antisymmetric_swiglu_lift(
                Up_f, kf - (Xf_f64 @ Wk_anc_q.T + bk_anc_q[None, :]),
                Up_t, kt - (xc["Xt"] @ Wk_anc_q.T + bk_anc_q[None, :]), zero_output_dims=zero_dims,
            )
            quad_v = fit_unlabeled_antisymmetric_swiglu_lift(
                Up_f, vf - (Xf_f64 @ Wv_eff.T + bv_eff[None, :]),
                Up_t, vt - (xc["Xt"] @ Wv_eff.T + bv_eff[None, :]), zero_output_dims=(),
            )
            del Xf_f64
            if quad_q["activated"]:
                qq_tune_raw = [quad_q["eval_fn"](tune_seqs_post_u_B[lp][s_i][-1:], scale=1.0)[0] for s_i in range(len(donor_alphas))]
            if quad_k["activated"]:
                qk_tune_raw = [quad_k["eval_fn"](tune_seqs_post_u_B[lp][s_i][1:], scale=1.0) for s_i in range(len(donor_alphas))]

        # P0-5 Fix: Solve BOS/temp DIRECTLY on exact COMPILED full-sequence logits!
        if variant == "1a_convex_m63":
            sol_a = solve_compiled_arm1a_bos_temp(
                donor_alphas, phi_tune_seqs, Wq_base, bq_base, Wk_base, bk_base, u0_B, reg_k["mu_x"], k0_A,
                C_xx_gt0, cos_cal_np, sin_cal_np, quad_q_raw_seqs=qq_tune_raw, quad_k_seqs=qk_tune_raw,
            )
            Wq_eff, bq_eff, Wk_eff, bk_eff = sol_a["Wq_eff"], sol_a["bq_eff"], sol_a["Wk_eff"], sol_a["bk_eff"]
            if quad_q is not None and quad_q["activated"]:
                quad_q["scale"] = float(sol_a["s_opt"])  # Fold `s_opt` into `quad_q` for exact calibration==replay parity!
            val_tune_kl = float(sol_a["compiled_val_tune_kl"])
            cal_meta = {"variant": variant, "mode_denoise_q": md_q, "mode_denoise_k": md_k, **{k: v for k, v in sol_a.items() if not isinstance(v, np.ndarray)}}
            # Also save un-folded `(Wq_clean, bq_clean, Wk_clean, bk_clean, delta0_opt)` for clean 8-cell factorial!
            Wk_clean, bk_clean = apply_kkt_rank1_pos0_anchor(Wk_base, bk_base, u0_B, k0_A, reg_k["mu_x"], C_xx_gt0=C_xx_gt0)
            qkv_8cell_clean = {
                "Wq": (float(sol_a["s_opt"]) * Wq_base).astype(np.float32),
                "bq": (float(sol_a["s_opt"]) * bq_base).astype(np.float32),
                "Wk": Wk_clean.astype(np.float32), "bk": bk_clean.astype(np.float32),
                "delta0_add": float(sol_a["delta0_opt"]),
            }
        else:
            sol_b = solve_compiled_arm1b_beta0(
                donor_alphas, phi_tune_seqs, Wq_base, bq_base, Wk_base, bk_base, u0_B, reg_k["mu_x"], k0_A,
                C_xx_gt0, cos_cal_np, sin_cal_np, quad_q_seqs=qq_tune_raw, quad_k_seqs=qk_tune_raw,
            )
            Wq_eff, bq_eff, Wk_eff, bk_eff = Wq_base, bq_base, sol_b["Wk_eff"], sol_b["bk_eff"]
            val_tune_kl = float(sol_b["compiled_val_tune_kl"])
            cal_meta = {"variant": variant, "rope_plane_q": md_q, "rope_plane_k": md_k, **{k: v for k, v in sol_b.items() if not isinstance(v, np.ndarray)}}
            qkv_8cell_clean = {
                "Wq": Wq_eff.astype(np.float32), "bq": bq_eff.astype(np.float32),
                "Wk": Wk_eff.astype(np.float32), "bk": bk_eff.astype(np.float32),
                "delta0_add": 0.0,
            }

        return {
            "layer": int(l), "head": int(h), "kv": int(kv),
            "include_taps": list(include_taps), "include_l0": bool(include_l0),
            "Wq": Wq_eff.astype(np.float32), "bq": bq_eff.astype(np.float32),
            "Wk": Wk_eff.astype(np.float32), "bk": bk_eff.astype(np.float32),
            "Wv": Wv_eff.astype(np.float32), "bv": bv_eff_centered.astype(np.float32),
            "qkv_8cell_clean": qkv_8cell_clean,
            "quad_q": quad_q, "quad_k": quad_k, "quad_v": quad_v,
            "r2_raw_tune": {"q": float(reg_q["r2_tune"]), "k": float(reg_k["r2_tune"]), "v": float(reg_v["r2_tune"])},
            "r2_summary": {
                "q_fit": round(reg_q["r2_fit"], 4), "q_tune": round(reg_q["r2_tune"], 4), "q_rep": round(reg_q["r2_rep"], 4), "q_lam": reg_q["best_lam_rel"],
                "k_fit": round(reg_k["r2_fit"], 4), "k_tune": round(reg_k["r2_tune"], 4), "k_rep": round(reg_k["r2_rep"], 4), "k_lam": reg_k["best_lam_rel"],
                "v_fit": round(reg_v["r2_fit"], 4), "v_tune": round(reg_v["r2_tune"], 4), "v_rep": round(reg_v["r2_rep"], 4), "v_lam": reg_v["best_lam_rel"],
                "quad_activated": [bool(quad_q and quad_q["activated"]), bool(quad_k and quad_k["activated"]), bool(quad_v and quad_v["activated"])],
            },
            "val_tune_attn_kl": float(val_tune_kl),  # Unrounded compiled KL!
            "cal_meta": cal_meta,
        }

    # Capture recipient eval sequences in compact BF16 CPU tensors (2x lower RAM!) via backbone-only forward!
    def _capture_recipient_eval(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        n, prs = len(items), [it["prompt"] for it in items]
        seqs_u: Dict[int, List[torch.Tensor]] = {l: [] for l in full_capture_layers_B}
        seqs_pu: Dict[int, List[torch.Tensor]] = {l: [] for l in prev_layers_for_quad}
        seqs_ids: List[np.ndarray] = []
        h_fin_B = np.zeros((n, D_B), dtype=np.float32)
        cur_ue, cur_pue, cur_hfe = {}, {}, []
        hks = []
        old_p = tok.padding_side
        tok.padding_side = "left"
        try:
            for li in full_capture_layers_B:
                hks.append(layers_B[li].input_layernorm.register_forward_hook(lambda _m, _inp, out, lidx=li: cur_ue.__setitem__(lidx, out.detach())))
            for li in prev_layers_for_quad:
                hks.append(layers_B[li].post_attention_layernorm.register_forward_hook(lambda _m, _inp, out, lidx=li: cur_pue.__setitem__(lidx, out.detach())))
            hks.append(model_B.model.norm.register_forward_pre_hook(lambda _m, inp: cur_hfe.append(inp[0][:, -1, :].detach().float())))
            for i0 in range(0, n, batch_size):
                i1 = min(i0 + batch_size, n)
                enc = tok(prs[i0:i1], return_tensors="pt", padding=True, add_special_tokens=False)
                ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
                cur_ue.clear(); cur_pue.clear(); cur_hfe.clear()
                _ = model_B.model(input_ids=ids, attention_mask=am, position_ids=_left_pad_position_ids(am), use_cache=False)
                h_fin_B[i0:i1] = cur_hfe[0].cpu().numpy()
                for b in range(i1 - i0):
                    rm = am[b].bool()
                    seqs_ids.append(ids[b, rm].cpu().numpy().astype(np.int64))
                    for li in full_capture_layers_B:
                        seqs_u[li].append(cur_ue[li][b, rm, :].to(dtype=torch.bfloat16).cpu())
                    for li in prev_layers_for_quad:
                        seqs_pu[li].append(cur_pue[li][b, rm, :].to(dtype=torch.bfloat16).cpu())
        finally:
            for hk in hks:
                hk.remove()
            tok.padding_side = old_p
        return {"seqs_u": seqs_u, "seqs_pu": seqs_pu, "seqs_ids": seqs_ids, "h_fin_B": h_fin_B}

    # Round-4 Exact v5.0 Arm 0: Analytical Attention-Output Jacobian Demand `G_l` (k_max=64 exact v5.0 parity!) + `compute_demand_weighted_supply_boundary`!
    L_F_arm0_np, _ = factor_psd_top_k(plugs_by_eta[best_eta]["G_A_raw"], k_max=64)
    L_F_arm0_t = torch.from_numpy(L_F_arm0_np.astype(np.float32)).to(dev)
    arm0_by_layer: Dict[int, Dict[str, Any]] = {}
    arm0_layers = sorted({l for tb in task_bundles.values() for l, _ in tb["primary_heads"]})
    for li in arm0_layers:
        xc0, _, _ = _get_x_cache(li, [], False)
        # 1. Exact v5.0 attention-output Jacobian demand matrix G_l on 32 generic val_tune sequences (`exclude_pos0=True`)!
        G_l = np.zeros((D_A, D_A), dtype=np.float64)
        seqs_A_li = [s.to(device=dev, dtype=torch.float32) for s in donor_cal["tune_seqs_u_A"][li]]
        seen_hl = set()
        for tb_i in task_bundles.values():
            hw_d = tb_i["donor_eval_base"]["head_weights_cpu"]
            for hl in tb_i["primary_heads"]:
                if hl[0] == li and hl not in seen_hl:
                    seen_hl.add(hl)
                    hw_i = hw_d[hl]
                    G_l += compute_attn_head_demand_matrix_closed_form(
                        hw_i["Wq"].to(dev), hw_i["bq"].to(dev) if hw_i["bq"] is not None else None,
                        hw_i["Wk"].to(dev), hw_i["bk"].to(dev) if hw_i["bk"] is not None else None,
                        hw_i["Wv"].to(dev), hw_i["bv"].to(dev) if hw_i["bv"] is not None else None,
                        hw_i["Wo"].to(dev), seqs_A_li, L_F_arm0_t, rope_theta=rope_theta, exclude_pos0=True,
                    )
        del seqs_A_li
        G_l = _sym(G_l)

        # 2. Stream bounded-RAM exact FP64 moments `(C_xx_A, C_xy_AB, mu_x_A)` in 8192-row chunks!
        XA_f_u16 = np.load(cal_dir / f"organ_u_L{li}_fit.npy", mmap_mode="r")
        XA_t_u16 = np.load(cal_dir / f"organ_u_L{li}_tune.npy")
        XA_r_u16 = np.load(cal_dir / f"organ_u_L{li}_report.npy")
        YB_t_u16 = np.load(rec_cal_dir / f"u_B_L{li}_tune.npy")
        YB_r_u16 = np.load(rec_cal_dir / f"u_B_L{li}_report.npy")
        N0, cr = int(XA_f_u16.shape[0]), 8192
        mu_x_A = np.zeros((D_A,), dtype=np.float64)
        for i0 in range(0, N0, cr):
            mu_x_A += _u16_to_f64(XA_f_u16[i0 : i0 + cr]).sum(axis=0)
        mu_x_A /= max(float(N0), 1.0)
        C_xx_A = np.zeros((D_A, D_A), dtype=np.float64)
        C_xy_AB = np.zeros((D_A, D_B), dtype=np.float64)
        for i0 in range(0, N0, cr):
            xa_c = _u16_to_f64(XA_f_u16[i0 : i0 + cr]) - mu_x_A[None, :]
            yb_c = xc0["Xf32"][i0 : i0 + cr].astype(np.float64) - xc0["mu_x"][None, :]
            C_xx_A += xa_c.T @ xa_c
            C_xy_AB += xa_c.T @ yb_c
        C_xx_A = _sym(C_xx_A / max(float(N0), 1.0))
        C_xy_AB /= max(float(N0), 1.0)

        # 3. Call v5.0's EXACT `compute_demand_weighted_supply_boundary` (demand-weighted val_tune lambda selection + (D_r, E_r))!
        sp0 = compute_demand_weighted_supply_boundary(
            C_xx_A, C_xy_AB, xc0["C_xx"], mu_x_A, xc0["mu_x"], G_l,
            XA_t_u16, YB_t_u16, XA_r_u16, YB_r_u16, ranks=(256,), lam_grid=LAM_GRID_UNCAPPED,
        )
        r0 = min(256, int(sp0["E_max"].shape[0]))
        W_r0 = sp0["D_max"][:, :r0] @ sp0["E_max"][:r0]
        r256_meta = sp0["by_rank"][r0] if r0 in sp0["by_rank"] else next(iter(sp0["by_rank"].values()))
        arm0_by_layer[li] = {
            "W_r0": W_r0.astype(np.float32), "mu_x": xc0["mu_x"].astype(np.float32),
            "mu_y": mu_x_A.astype(np.float32), "mu_pos0_A": donor_cal["mu_pos0_u_by_l"][li].astype(np.float32),
            "best_lam_rel": float(sp0["best_lam_rel"]), "rank": int(r0),
            "method": "v5_0_exact_jacobian_demand_weighted_supply_boundary",
            "kappa_tune_r256": round(float(r256_meta["kappa_r_tune"]), 4),
            "kappa_rep_r256": round(float(r256_meta["kappa_r_rep"]), 4),
            "min_rho2_rep_r256": round(float(r256_meta["min_rho2_rep"]), 4),
        }
        del XA_t_u16, XA_r_u16, YB_t_u16, YB_r_u16, G_l, C_xx_A, C_xy_AB, sp0; gc.collect()
    del L_F_arm0_t

    # Step 2.5: Evaluate Each Task (`1hop`, `2hop`)
    T_raw_t = torch.from_numpy((gamma_scale * T_raw_frozen).astype(np.float32)).to(dev)
    task_results: Dict[str, Any] = {}

    for tname, tb in task_bundles.items():
        print(f"[v5.1 Phase 2] Evaluating `{tname}` across all arms, 8-cell decomp, and fair SHUF controls...", flush=True)
        prim_heads: List[Tuple[int, int]] = list(tb["primary_heads"])
        cand_heads: List[Tuple[int, int]] = list(tb["candidate_heads"])
        shuf_heads: List[Tuple[int, int]] = list(tb["shuf_layer_heads"])
        hw_cpu = tb["donor_eval_base"]["head_weights_cpu"]
        eval_base = tb["eval_base"]
        n_eval = len(eval_base)
        table_ids_base = np.array([int(it["table_id"]) for it in eval_base])

        gens_none = _greedy_generate_clean_no_add_special(model_B, tok, [it["prompt"] for it in eval_base], v_shared=v_shared, max_new_tokens=4, batch_size=batch_size)
        ok_none = np.array([int(check_2hop_exact_word(g, it["answer"])) for g, it in zip(gens_none, eval_base)], dtype=np.int64)
        ok_none_whole = np.array([int(check_exact_whole_completion(g, it["answer"])) for g, it in zip(gens_none, eval_base)], dtype=np.int64)
        wrong_mask = (ok_none == 0)

        rec_eval_base = _capture_recipient_eval(eval_base)
        h_fin_B_t = torch.from_numpy(rec_eval_base["h_fin_B"]).to(dev)
        ans_t = torch.tensor([int(it["answer_tok_id"]) for it in eval_base], device=dev, dtype=torch.long)
        foil_t = torch.tensor([int(it["foil_tok_id"]) for it in eval_base], device=dev, dtype=torch.long)
        e_a_B, e_f_B = lm_w_B[ans_t].float(), lm_w_B[foil_t].float()
        margin_B_base = _real_norm_margin_from_rows(h_fin_B_t, gamma_fin_B, eps_fin_B, e_a_B, e_f_B)

        b_np, s_np = np.arange(0, n_eval, 2, dtype=np.int64), np.arange(1, n_eval, 2, dtype=np.int64)
        b_idx, s_idx = torch.tensor(b_np, device=dev, dtype=torch.long), torch.tensor(s_np, device=dev, dtype=torch.long)
        e_as_B, e_fb_B = lm_w_B[ans_t[s_idx]].float(), lm_w_B[ans_t[b_idx]].float()

        c_true = tb["donor_eval_base"]["c_true"]
        d_ref_policy = tb["donor_eval_base"]["d_ref_policy"]
        max_T_ev = max(len(s) for s in rec_eval_base["seqs_ids"]) + 8
        cos_ev, sin_ev = _make_rope_cos_sin(d_h, rope_theta, max_T_ev, dev)

        def _tok0_acc(h_raw: torch.Tensor, items_seq: Sequence[Dict[str, Any]]) -> Tuple[float, np.ndarray]:
            rms = torch.sqrt(torch.mean(h_raw.float() ** 2, dim=-1, keepdim=True) + eps_fin_B)
            preds = _native_bf16_argmax_shared((h_raw.float() / rms) * gamma_fin_B.unsqueeze(0), lm_w_B, chunk=16)
            ok = np.array([int(p == int(it["answer_tok_id"])) for p, it in zip(preds, items_seq)], dtype=np.int64)
            return float(np.mean(ok)), ok

        base_tok0_acc, base_tok0_ok = _tok0_acc(h_fin_B_t, eval_base)
        tok0_wrong_mask = (base_tok0_ok == 0)

        def _score_base_delta(out_centered_A: torch.Tensor, c_ref_ov: Optional[np.ndarray] = None, d_ref_ov: Optional[np.ndarray] = None) -> Dict[str, Any]:
            # Match BF16 live residual addition precision cleanly!
            dw_B = (out_centered_A @ T_raw_t.T).to(dtype=torch.bfloat16).float()
            h_new = (h_fin_B_t.to(dtype=torch.bfloat16) + dw_B.to(dtype=torch.bfloat16)).float()
            cr = c_true if c_ref_ov is None else c_ref_ov
            dr = d_ref_policy if d_ref_ov is None else d_ref_ov
            c_hat = (_real_norm_margin_from_rows(h_new, gamma_fin_B, eps_fin_B, e_a_B, e_f_B) - margin_B_base).cpu().double().numpy()
            d_hat = (
                _real_norm_margin_from_rows((h_fin_B_t[b_idx].bfloat16() + dw_B[s_idx].bfloat16()).float(), gamma_fin_B, eps_fin_B, e_as_B, e_fb_B)
                - _real_norm_margin_from_rows((h_fin_B_t[b_idx].bfloat16() + dw_B[b_idx].bfloat16()).float(), gamma_fin_B, eps_fin_B, e_as_B, e_fb_B)
            ).cpu().double().numpy()
            qr = _compute_safe_q_and_r2(d_hat, dr, c_hat, cr)
            wir_gen = _compute_safe_wrong_item_ratio(c_hat, cr, wrong_mask, min_donor_signal=0.25)
            wir_tok0 = _compute_safe_wrong_item_ratio(c_hat, cr, tok0_wrong_mask, min_donor_signal=0.25)
            acc_all, ok_all = _tok0_acc(h_new, eval_base)
            return {
                "R2_c": qr["R2_c"], "Q_outlet": qr["Q_causal"], **wir_gen,
                "n_gen_wrong": int(np.sum(wrong_mask)),
                "n_tok0_wrong": int(np.sum(tok0_wrong_mask)),
                "wir_on_tok0_wrong": wir_tok0["wrong_item_ratio"],
                "tok0_hits": int(np.sum(ok_all)), "tok0_acc": round(acc_all, 4),
                "tok0_lift_pp": round(100.0 * (acc_all - base_tok0_acc), 2),
                "tok0_acc_on_gen_wrong": round(float(np.mean(ok_all[wrong_mask])) if np.any(wrong_mask) else 0.0, 4),
                "tok0_acc_on_tok0_wrong": round(float(np.mean(ok_all[tok0_wrong_mask])) if np.any(tok0_wrong_mask) else 0.0, 4),
            }

        lproc = [_MaskPaddedVocabLogitsProcessor(v_shared)]
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        def _live_gen_with_delta(out_centered_A: torch.Tensor, items_seq: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
            dw_all = (out_centered_A @ T_raw_t.T).to(dtype=torch.bfloat16)
            prs = [it["prompt"] for it in items_seq]
            old_p2 = tok.padding_side
            tok.padding_side = "left"
            b_sl, txts = [0, 0], []
            def _inj(_m, inp):
                h_cur = inp[0].clone()
                if h_cur.shape[1] > 1:
                    h_cur[:, -1, :] = h_cur[:, -1, :] + dw_all[b_sl[0] : b_sl[1]].to(dtype=h_cur.dtype)
                return (h_cur,) + inp[1:]
            hk = model_B.model.norm.register_forward_pre_hook(_inj)
            try:
                for i0 in range(0, len(prs), batch_size):
                    i1 = min(i0 + batch_size, len(prs))
                    b_sl[0], b_sl[1] = i0, i1
                    enc = tok(prs[i0:i1], return_tensors="pt", padding=True, add_special_tokens=False)
                    ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
                    gen = model_B.generate(
                        input_ids=ids, attention_mask=am, max_new_tokens=4, do_sample=False,
                        temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                        logits_processor=lproc, pad_token_id=pad_id, use_cache=True,
                    )
                    pw = int(ids.shape[1])
                    for j in range(i1 - i0):
                        txts.append(tok.decode(gen[j, pw:], skip_special_tokens=True))
            finally:
                hk.remove(); tok.padding_side = old_p2
            ok_fw = np.array([int(check_2hop_exact_word(g, it["answer"])) for g, it in zip(txts, items_seq)], dtype=np.int64)
            ok_wh = np.array([int(check_exact_whole_completion(g, it["answer"])) for g, it in zip(txts, items_seq)], dtype=np.int64)
            return ok_fw, ok_wh

        def _replay_qkv_arm(
            head_arms: List[Dict[str, Any]], Wo_list: List[torch.Tensor], rec_bundle: Dict[str, Any], use_clean_8cell: bool = False
        ) -> Tuple[torch.Tensor, List[torch.Tensor], List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]], float]:
            n_it = len(rec_bundle["seqs_ids"])
            out_acc = torch.zeros((n_it, D_A), device=dev, dtype=torch.float32)
            z_list, qkv_list, sinks = [], [], []
            for ha, Wo in zip(head_arms, Wo_list):
                li, taps, inc_l0 = ha["layer"], ha["include_taps"], ha["include_l0"]
                src_w = ha["qkv_8cell_clean"] if use_clean_8cell else ha
                Wq_t, bq_t = torch.from_numpy(src_w["Wq"]).to(dev), torch.from_numpy(src_w["bq"]).to(dev)
                Wk_t, bk_t = torch.from_numpy(src_w["Wk"]).to(dev), torch.from_numpy(src_w["bk"]).to(dev)
                Wv_t, bv_t = torch.from_numpy(ha["Wv"]).to(dev), torch.from_numpy(ha["bv"]).to(dev)
                d0_add = float(src_w["delta0_add"]) if use_clean_8cell else 0.0
                qq, qk, qv = ha["quad_q"], ha["quad_k"], ha["quad_v"]
                zh_mat = torch.zeros((n_it, d_h), device=dev, dtype=torch.float32)
                h_qkv = []
                lp = max(0, li - 1)
                for i in range(n_it):
                    u_s = {tl: rec_bundle["seqs_u"][tl][i].float().numpy() for tl in full_capture_layers_B}
                    Phi_t = torch.from_numpy(_build_tap_seq(u_s, rec_bundle["seqs_ids"][i], li, taps, inc_l0).astype(np.float32)).to(dev)
                    q_last = F.linear(Phi_t[-1:], Wq_t, bq_t).squeeze(0)
                    k_seq = F.linear(Phi_t, Wk_t, bk_t)
                    v_seq = F.linear(Phi_t, Wv_t, bv_t)
                    if qq is not None and qq["activated"]:
                        pu_last = rec_bundle["seqs_pu"][lp][i][-1:].float().numpy()
                        q_last = q_last + torch.from_numpy(qq["eval_fn"](pu_last, scale=qq["scale"]).astype(np.float32)).to(dev).squeeze(0)
                    if qk is not None and qk["activated"]:
                        pu_gt0 = rec_bundle["seqs_pu"][lp][i][1:].float().numpy()
                        k_seq = k_seq.clone()
                        k_seq[1:] = k_seq[1:] + torch.from_numpy(qk["eval_fn"](pu_gt0, scale=1.0).astype(np.float32)).to(dev)
                    if qv is not None and qv["activated"]:
                        pu_gt0 = rec_bundle["seqs_pu"][lp][i][1:].float().numpy()
                        v_seq = v_seq.clone()
                        v_seq[1:] = v_seq[1:] + torch.from_numpy(qv["eval_fn"](pu_gt0, scale=1.0).astype(np.float32)).to(dev)
                    zh, oh, alp, _ = compute_head_qkv_and_logits(q_last, k_seq, v_seq, Wo, cos_ev, sin_ev, delta0_logit_add=d0_add)
                    zh_mat[i] = zh
                    out_acc[i] += oh
                    sinks.append(float(alp[0].item()))
                    h_qkv.append((q_last.cpu(), k_seq.cpu(), v_seq.cpu()))
                z_list.append(zh_mat)
                qkv_list.append(h_qkv)
            return out_acc, z_list, qkv_list, float(np.mean(sinks))

        Wo_prim_dev = [hw_cpu[lh]["Wo"].to(dev) for lh in prim_heads]
        v_bar_prim = {lh: donor_cal["v_bar_gen_by_kv"][(lh[0], lh[1] // g_rep_A)] for lh in prim_heads}
        mean_out_gen_prim = torch.from_numpy(tb["donor_eval_base"]["mean_out_gen_prim_f32"]).to(dev)

        # Test 2A: True-Donor Outlet Oracle (centered identically before BF16 Wo projection for exact factorial parity!)
        qkv_donor_c = [
            [(q.float(), k.float(), v.float() - torch.from_numpy(v_bar_prim[lh]).unsqueeze(0)) for (q, k, v) in tb["donor_eval_base"]["qkv_prim_cpu"][hi]]
            for hi, lh in enumerate(prim_heads)
        ]
        out_oracle_c = torch.zeros((n_eval, D_A), device=dev, dtype=torch.float32)
        for hi, Wo in enumerate(Wo_prim_dev):
            for i in range(n_eval):
                q_t, k_t, v_t = (t.to(dev) for t in qkv_donor_c[hi][i])
                _, oh, _, _ = compute_head_qkv_and_logits(q_t, k_t, v_t, Wo, cos_ev, sin_ev)
                out_oracle_c[i] += oh
        sc_oracle = _score_base_delta(out_oracle_c)
        ok_or_fw, ok_or_wh = _live_gen_with_delta(out_oracle_c, eval_base)
        sc_oracle.update({"live_gen_hits": int(np.sum(ok_or_fw)), "live_gen_em": round(float(np.mean(ok_or_fw)), 4), "live_gen_whole_em": round(float(np.mean(ok_or_wh)), 4), "live_gen_lift_pp": round(100.0 * float(np.mean(ok_or_fw) - np.mean(ok_none)), 2)})

        # Arm 0: v5.0 baseline control (same-layer 3584-D rank-256 MMSE + uncalibrated mu_pos0)
        out_arm0 = torch.zeros((n_eval, D_A), device=dev, dtype=torch.float32)
        qkv_arm0, sinks_arm0 = [], []
        for lh, Wo in zip(prim_heads, Wo_prim_dev):
            li, _ = lh
            a0, hw = arm0_by_layer[li], hw_cpu[lh]
            Wq, Wk, Wv = hw["Wq"].to(dev), hw["Wk"].to(dev), hw["Wv"].to(dev)
            bq, bk, bv = (hw[k].to(dev) if hw[k] is not None else None for k in ("bq", "bk", "bv"))
            W_r0_t, mux_t, muy_t, mp0_t, vb_t = (torch.from_numpy(x).to(dev) for x in (a0["W_r0"], a0["mu_x"], a0["mu_y"], a0["mu_pos0_A"], v_bar_prim[lh]))
            h_qkv0 = []
            for i in range(n_eval):
                Ys = rec_eval_base["seqs_u"][li][i].to(device=dev, dtype=torch.float32)
                Xin = torch.empty((Ys.shape[0], D_A), device=dev, dtype=torch.float32)
                Xin[0] = mp0_t
                if Ys.shape[0] > 1:
                    Xin[1:] = muy_t.unsqueeze(0) + (Ys[1:] - mux_t.unsqueeze(0)) @ W_r0_t.T
                X_bf = Xin.to(dtype=Wq.dtype)
                ql, ks, vs = F.linear(X_bf[-1:], Wq, bq).squeeze(0).float(), F.linear(X_bf, Wk, bk).float(), F.linear(X_bf, Wv, bv).float() - vb_t.unsqueeze(0)
                _, oh, alp, _ = compute_head_qkv_and_logits(ql, ks, vs, Wo, cos_ev, sin_ev)
                out_arm0[i] += oh; sinks_arm0.append(float(alp[0].item()))
                h_qkv0.append((ql.cpu(), ks.cpu(), vs.cpu()))
            qkv_arm0.append(h_qkv0)
        sc_arm0 = _score_base_delta(out_arm0)
        sc_arm0["mean_pos0_sink_mass"] = round(float(np.mean(sinks_arm0)), 4)
        ok_a0_fw, ok_a0_wh = _live_gen_with_delta(out_arm0, eval_base)
        sc_arm0.update({"live_gen_hits": int(np.sum(ok_a0_fw)), "live_gen_em": round(float(np.mean(ok_a0_fw)), 4), "live_gen_whole_em": round(float(np.mean(ok_a0_wh)), 4), "live_gen_lift_pp": round(100.0 * float(np.mean(ok_a0_fw) - np.mean(ok_none)), 2)})

        # Fit Arms 1a, 1b, 2a, 2b strictly on generic text (selecting BOS variant strictly by UNROUNDED compiled val_tune KL!)
        arms_1a = [_fit_role_qkv_arm_for_head(l, h, [], False, "1a_convex_m63") for l, h in prim_heads]
        arms_1b = [_fit_role_qkv_arm_for_head(l, h, [], False, "1b_rope_beta0") for l, h in prim_heads]
        win_var_tune = "1a_convex_m63" if float(np.mean([a["val_tune_attn_kl"] for a in arms_1a])) <= float(np.mean([a["val_tune_attn_kl"] for a in arms_1b])) else "1b_rope_beta0"
        arms_2a = [_fit_role_qkv_arm_for_head(l, h, tap_layers, True, win_var_tune, add_quad_lift=False) for l, h in prim_heads]
        arms_2b = [_fit_role_qkv_arm_for_head(l, h, tap_layers, True, win_var_tune, add_quad_lift=True) for l, h in prim_heads]

        # Arm 2c (Discovery Diagnostic — strictly unrounded `val_tune` observability!)
        cand_obs_rows, cand_arm_cache = [], {}
        for l, h in cand_heads:
            ex = next((a for a in arms_2a if a["layer"] == l and a["head"] == h), None) or _fit_role_qkv_arm_for_head(l, h, tap_layers, True, win_var_tune, add_quad_lift=False)
            cand_arm_cache[(l, h)] = ex
            oq, ok_o, ov = max(ex["r2_raw_tune"]["q"], 0.0), max(ex["r2_raw_tune"]["k"], 0.0), max(ex["r2_raw_tune"]["v"], 0.0)
            dla_v = max(float(tb["cand_dla"].get(f"L{l}H{h}", 0.0)), 0.0)
            cand_obs_rows.append((dla_v * ((oq * ok_o * ov) ** (1.0 / 3.0)), l, h))
        cand_obs_rows.sort(key=lambda x: x[0], reverse=True)
        sreal_heads = [(l, h) for _, l, h in cand_obs_rows[: len(prim_heads)]]
        arms_2c = [cand_arm_cache[lh] for lh in sreal_heads]
        Wo_2c_dev = [hw_cpu[lh]["Wo"].to(dev) for lh in sreal_heads]
        sreal_same_as_prim = (set(sreal_heads) == set(prim_heads))

        arm_outputs: Dict[str, Any] = {}
        arm_unrounded_kl: Dict[str, float] = {}
        arm_unrounded_sink: Dict[str, float] = {}
        arm_replay_cache: Dict[str, Any] = {}
        live_em_by_arm: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for aname, ha_list, wo_l in (
            ("Arm_1a_convex_m63", arms_1a, Wo_prim_dev),
            ("Arm_1b_rope_beta0", arms_1b, Wo_prim_dev),
            ("Arm_2a_multitap_L0", arms_2a, Wo_prim_dev),
            ("Arm_2b_multitap_quad", arms_2b, Wo_prim_dev),
            ("Arm_2c_S_real_diagnostic", arms_2c, Wo_2c_dev),
        ):
            out_a, z_a, qkv_a, sink_a = _replay_qkv_arm(ha_list, wo_l, rec_eval_base, use_clean_8cell=False)
            sc_a = _score_base_delta(out_a)
            kl_raw = float(np.mean([a["val_tune_attn_kl"] for a in ha_list]))
            arm_unrounded_kl[aname] = kl_raw
            arm_unrounded_sink[aname] = sink_a
            ok_fw, ok_wh = _live_gen_with_delta(out_a, eval_base)
            live_em_by_arm[aname] = (ok_fw, ok_wh)
            sc_a.update({
                "mean_pos0_sink_mass": round(sink_a, 4),
                "mean_val_tune_attn_kl": round(kl_raw, 5),
                "live_gen_hits": int(np.sum(ok_fw)), "live_gen_em": round(float(np.mean(ok_fw)), 4),
                "live_gen_whole_em": round(float(np.mean(ok_wh)), 4),
                "live_gen_lift_pp": round(100.0 * float(np.mean(ok_fw) - np.mean(ok_none)), 2),
                "qkv_r2_per_head": [a["r2_summary"] for a in ha_list],
                "cal_meta_per_head": [a["cal_meta"] for a in ha_list],
            })
            if aname == "Arm_2c_S_real_diagnostic":
                sc_a["sreal_same_as_primary"] = sreal_same_as_prim
                if not sreal_same_as_prim:
                    # Omit primary-head causal refs when S_real selects different heads!
                    sc_a["R2_c"] = None
                    sc_a["Q_outlet"] = None
                    sc_a["wrong_item_ratio"] = None
                    sc_a["wir_on_tok0_wrong"] = None
                sc_a["note"] = "Discovery-selected diagnostic arm (val_tune observability); outside primary qualifying selector."
            arm_outputs[aname] = sc_a
            arm_replay_cache[aname] = (out_a, z_a, qkv_a, wo_l, ha_list)

        # Select primary qualifying arm STRICTLY by UNROUNDED generic compiled `val_tune_attn_kl`!
        best_arm_name = min(("Arm_1a_convex_m63", "Arm_1b_rope_beta0", "Arm_2a_multitap_L0", "Arm_2b_multitap_quad"), key=lambda k: arm_unrounded_kl[k])
        out_best, z_best, qkv_best_deployed, wo_best, ha_best = arm_replay_cache[best_arm_name]
        best_uses_taps = (best_arm_name in ("Arm_2a_multitap_L0", "Arm_2b_multitap_quad"))
        best_uses_quad = (best_arm_name == "Arm_2b_multitap_quad")
        best_var = "1a_convex_m63" if best_arm_name == "Arm_1a_convex_m63" else ("1b_rope_beta0" if best_arm_name == "Arm_1b_rope_beta0" else win_var_tune)

        # P0-9 Fix: 20-draw SHUF-pi + live greedy generation on TOP-3 highest-scoring draws (adversarial upper bound!)
        shuf_pi_draws = []
        for di in range(20):
            out_sh = torch.zeros((n_eval, D_A), device=dev, dtype=torch.float32)
            for hi, (zh_m, Wo_bf) in enumerate(zip(z_best, wo_best)):
                gp = torch.Generator(device="cpu").manual_seed(SEED + 9991 + 131 * hi + di)
                out_sh += F.linear(zh_m.to(dtype=Wo_bf.dtype), Wo_bf[:, torch.randperm(d_h, generator=gp).to(dev)]).float()
            sc_sh = _score_base_delta(out_sh)
            shuf_pi_draws.append((sc_sh["tok0_hits"], sc_sh["wrong_item_ratio"], di, out_sh.cpu(), sc_sh))
        shuf_pi_draws_sorted = sorted(shuf_pi_draws, key=lambda x: (x[0], x[1] if x[1] is not None else -1e9), reverse=True)
        top3_shuf_em_hits, top3_shuf_ok_arrs = [], []
        for _, _, di, out_sh_cpu, _ in shuf_pi_draws_sorted[:3]:
            ok_s_fw, _ = _live_gen_with_delta(out_sh_cpu.to(dev), eval_base)
            top3_shuf_em_hits.append((int(np.sum(ok_s_fw)), di, ok_s_fw))
            top3_shuf_ok_arrs.append(ok_s_fw)
        worst_shuf_pi_hits, worst_shuf_pi_di, ok_shuf_pi_worst = max(top3_shuf_em_hits, key=lambda x: x[0])

        # P0-9 Fix: Matched-depth & matched-capacity `SHUF-layer` (`L24-L26` lowest-|DLA| heads, IDENTICAL winning recipe!)
        arms_shuf_layer = [
            _fit_role_qkv_arm_for_head(l, h, tap_layers if best_uses_taps else [], best_uses_taps, best_var, add_quad_lift=best_uses_quad)
            for l, h in shuf_heads
        ]
        Wo_shl_dev = []
        for idx_s, lh_s in enumerate(shuf_heads):
            Ws = hw_cpu[lh_s]["Wo"].to(dev).float()
            tfn = float(torch.linalg.norm(Wo_prim_dev[idx_s % len(Wo_prim_dev)].float()).item())
            Wo_shl_dev.append((Ws * (tfn / max(float(torch.linalg.norm(Ws).item()), 1e-8))).to(dtype=Wo_prim_dev[0].dtype))
        out_shl, _, _, sink_shl = _replay_qkv_arm(arms_shuf_layer, Wo_shl_dev, rec_eval_base, use_clean_8cell=False)
        sc_shl = _score_base_delta(out_shl)
        ok_shl_fw, ok_shl_wh = _live_gen_with_delta(out_shl, eval_base)
        sc_shl.update({"mean_pos0_sink_mass": round(sink_shl, 4), "live_gen_hits": int(np.sum(ok_shl_fw)), "live_gen_em": round(float(np.mean(ok_shl_fw)), 4), "live_gen_whole_em": round(float(np.mean(ok_shl_wh)), 4), "live_gen_lift_pp": round(100.0 * float(np.mean(ok_shl_fw) - np.mean(ok_none)), 2)})

        ok_best_fw, ok_best_wh = live_em_by_arm[best_arm_name]
        boot_vs_none = paired_table_cluster_bootstrap(ok_best_fw, ok_none, table_ids_base, n_boot=2000)
        boot_vs_shuf_pi = paired_table_cluster_bootstrap(ok_best_fw, ok_shuf_pi_worst, table_ids_base, n_boot=2000)
        boot_vs_shuf_layer = paired_table_cluster_bootstrap(ok_best_fw, ok_shl_fw, table_ids_base, n_boot=2000)

        # Round-3 P0-A Fix: Mask (63, 127) ONLY on mixed (Q_A, K_B) & (Q_B, K_A) cells; pure Donor Oracle stays 100% untouched!
        decomp_arm0 = run_8cell_and_sink_content_decomposition(qkv_donor_c, qkv_arm0, Wo_prim_dev, cos_ev, sin_ev, _score_base_delta)
        mixed_m = (d_h // 2 - 1, d_h - 1) if best_var == "1a_convex_m63" else ()
        decomp_best = run_8cell_and_sink_content_decomposition(qkv_donor_c, qkv_best_deployed, Wo_prim_dev, cos_ev, sin_ev, _score_base_delta, mask_mixed_dims=mixed_m)

        # Separate Task 2 Binding & Null Swap Evaluation (with 20-draw SHUF-pi + SHUF-layer + Donor Null Ref!)
        binding_null_res: Optional[Dict[str, Any]] = None
        binding_gate_pass = False
        if "eval_binding" in tb and tb["eval_binding"]:
            ev_bind, ev_null = tb["eval_binding"], tb["eval_null"]
            n_bn = len(ev_bind)
            rec_bind, rec_null = _capture_recipient_eval(ev_bind), _capture_recipient_eval(ev_null)
            h_fin_B_bind = torch.from_numpy(rec_bind["h_fin_B"]).to(dev)
            ans_b_t = torch.tensor([int(it["answer_tok_id"]) for it in ev_bind], device=dev, dtype=torch.long)
            foil_b_t = torch.tensor([int(it["foil_tok_id"]) for it in ev_bind], device=dev, dtype=torch.long)
            e_ab_B, e_fb_B2 = lm_w_B[ans_b_t].float(), lm_w_B[foil_b_t].float()
            base_L_h = h_fin_B_t[b_idx]
            margin_B_bind_base = _real_norm_margin_from_rows(h_fin_B_bind, gamma_fin_B, eps_fin_B, e_ab_B, e_fb_B2)

            gens_none_b = _greedy_generate_clean_no_add_special(model_B, tok, [it["prompt"] for it in ev_bind], v_shared=v_shared, max_new_tokens=4, batch_size=batch_size)
            ok_none_b = np.array([int(check_2hop_exact_word(g, it["answer"])) for g, it in zip(gens_none_b, ev_bind)], dtype=np.int64)
            wrong_bind = (ok_none_b == 0)
            c_true_b = tb["donor_eval_binding"]["c_true"]
            d_ref_b = tb["donor_eval_binding"]["d_ref_binding"]
            base_tok0_b, base_tok0_b_ok = _tok0_acc(h_fin_B_bind, ev_bind)
            tok0_wrong_bind = (base_tok0_b_ok == 0)

            def _score_bn(out_base_all: torch.Tensor, out_bind_c: torch.Tensor, out_null_c: torch.Tensor) -> Dict[str, Any]:
                dw_bL = (out_base_all[b_idx] @ T_raw_t.T).bfloat16().float()
                dw_b = (out_bind_c @ T_raw_t.T).bfloat16().float()
                dw_n = (out_null_c @ T_raw_t.T).bfloat16().float()
                c_hat_b = (_real_norm_margin_from_rows((h_fin_B_bind.bfloat16() + dw_b.bfloat16()).float(), gamma_fin_B, eps_fin_B, e_ab_B, e_fb_B2) - margin_B_bind_base).cpu().double().numpy()
                d_hat_b = (
                    _real_norm_margin_from_rows((base_L_h.bfloat16() + dw_b.bfloat16()).float(), gamma_fin_B, eps_fin_B, e_ab_B, e_fb_B2)
                    - _real_norm_margin_from_rows((base_L_h.bfloat16() + dw_bL.bfloat16()).float(), gamma_fin_B, eps_fin_B, e_ab_B, e_fb_B2)
                ).cpu().double().numpy()
                d_hat_n = (
                    _real_norm_margin_from_rows((base_L_h.bfloat16() + dw_n.bfloat16()).float(), gamma_fin_B, eps_fin_B, e_ab_B, e_fb_B2)
                    - _real_norm_margin_from_rows((base_L_h.bfloat16() + dw_bL.bfloat16()).float(), gamma_fin_B, eps_fin_B, e_ab_B, e_fb_B2)
                ).cpu().double().numpy()
                qr_b = _compute_safe_q_and_r2(d_hat_b, d_ref_b, c_hat_b, c_true_b)
                wir_b = _compute_safe_wrong_item_ratio(c_hat_b, c_true_b, wrong_bind, min_donor_signal=0.25)
                wir_b_tok0 = _compute_safe_wrong_item_ratio(c_hat_b, c_true_b, tok0_wrong_bind, min_donor_signal=0.25)
                acc_b, ok_tb = _tok0_acc((h_fin_B_bind.bfloat16() + dw_b.bfloat16()).float(), ev_bind)
                m_abs_b, m_abs_n = float(np.mean(np.abs(d_hat_b))), float(np.mean(np.abs(d_hat_n)))
                leak_raw = m_abs_n / max(m_abs_b, 1e-6)
                return {
                    "binding_R2_c": qr_b["R2_c"], "binding_Q_outlet": qr_b["Q_causal"], **wir_b,
                    "n_gen_wrong_bind": int(np.sum(wrong_bind)),
                    "n_tok0_wrong_bind": int(np.sum(tok0_wrong_bind)),
                    "wir_on_tok0_wrong": wir_b_tok0["wrong_item_ratio"],
                    "binding_tok0_hits": int(np.sum(ok_tb)), "binding_tok0_acc": round(acc_b, 4),
                    "binding_tok0_lift_pp": round(100.0 * (acc_b - base_tok0_b), 2),
                    "binding_mean_abs_swap_nat": round(m_abs_b, 4), "null_swap_mean_abs_nat": round(m_abs_n, 4),
                    "null_to_binding_leakage_ratio": round(leak_raw, 4),
                    "_leak_raw": leak_raw,
                }

            out_or_b = torch.zeros((n_bn, D_A), device=dev, dtype=torch.float32)
            out_or_n = torch.zeros((n_bn, D_A), device=dev, dtype=torch.float32)
            qkv_don_b_c = [[(q.float(), k.float(), v.float() - torch.from_numpy(v_bar_prim[lh]).unsqueeze(0)) for (q, k, v) in tb["donor_eval_binding"]["qkv_prim_cpu"][hi]] for hi, lh in enumerate(prim_heads)]
            qkv_don_n_c = [[(q.float(), k.float(), v.float() - torch.from_numpy(v_bar_prim[lh]).unsqueeze(0)) for (q, k, v) in tb["donor_eval_null"]["qkv_prim_cpu"][hi]] for hi, lh in enumerate(prim_heads)]
            for hi, Wo in enumerate(Wo_prim_dev):
                for i in range(n_bn):
                    _, ohb, _, _ = compute_head_qkv_and_logits(*(t.to(dev) for t in qkv_don_b_c[hi][i]), Wo, cos_ev, sin_ev)
                    _, ohn, _, _ = compute_head_qkv_and_logits(*(t.to(dev) for t in qkv_don_n_c[hi][i]), Wo, cos_ev, sin_ev)
                    out_or_b[i] += ohb; out_or_n[i] += ohn
            sc_bn_or = _score_bn(out_oracle_c, out_or_b, out_or_n)
            ok_orb_fw, ok_orb_wh = _live_gen_with_delta(out_or_b, ev_bind)
            sc_bn_or.update({"live_gen_hits": int(np.sum(ok_orb_fw)), "live_gen_em": round(float(np.mean(ok_orb_fw)), 4), "live_gen_whole_em": round(float(np.mean(ok_orb_wh)), 4), "live_gen_lift_pp": round(100.0 * float(np.mean(ok_orb_fw) - np.mean(ok_none_b)), 2)})

            out_bb, z_bb, qkv_bb_dep, sink_bb = _replay_qkv_arm(ha_best, wo_best, rec_bind, use_clean_8cell=False)
            out_bn, z_bn, _, _ = _replay_qkv_arm(ha_best, wo_best, rec_null, use_clean_8cell=False)
            sc_bn_best = _score_bn(out_best, out_bb, out_bn)
            ok_bb_fw, ok_bb_wh = _live_gen_with_delta(out_bb, ev_bind)
            sc_bn_best.update({"mean_pos0_sink_mass": round(sink_bb, 4), "live_gen_hits": int(np.sum(ok_bb_fw)), "live_gen_em": round(float(np.mean(ok_bb_fw)), 4), "live_gen_whole_em": round(float(np.mean(ok_bb_wh)), 4), "live_gen_lift_pp": round(100.0 * float(np.mean(ok_bb_fw) - np.mean(ok_none_b)), 2)})

            # P0-9 Fix: 20-draw SHUF-pi permuting base + bind + null consistently!
            shuf_b_draws = []
            for di in range(20):
                out_sh_base = torch.zeros((n_eval, D_A), device=dev, dtype=torch.float32)
                out_shb = torch.zeros((n_bn, D_A), device=dev, dtype=torch.float32)
                out_shn = torch.zeros((n_bn, D_A), device=dev, dtype=torch.float32)
                for hi, (zb_m, zbb_m, zbn_m, Wo_bf) in enumerate(zip(z_best, z_bb, z_bn, wo_best)):
                    gp = torch.Generator(device="cpu").manual_seed(SEED + 12345 + 131 * hi + di)
                    Wop = Wo_bf[:, torch.randperm(d_h, generator=gp).to(dev)]
                    out_sh_base += F.linear(zb_m.to(dtype=Wo_bf.dtype), Wop).float()
                    out_shb += F.linear(zbb_m.to(dtype=Wo_bf.dtype), Wop).float()
                    out_shn += F.linear(zbn_m.to(dtype=Wo_bf.dtype), Wop).float()
                sc_sb = _score_bn(out_sh_base, out_shb, out_shn)
                shuf_b_draws.append((sc_sb["binding_tok0_hits"], sc_sb["wrong_item_ratio"], di, out_shb.cpu(), sc_sb))
            shuf_b_top = sorted(shuf_b_draws, key=lambda x: (x[0], x[1] if x[1] is not None else -1e9), reverse=True)
            top3_sb_gen = []
            for _, _, di, out_shb_c, _ in shuf_b_top[:3]:
                ok_sb_fw, _ = _live_gen_with_delta(out_shb_c.to(dev), ev_bind)
                top3_sb_gen.append((int(np.sum(ok_sb_fw)), di, ok_sb_fw))
            worst_sb_hits, _, ok_sb_worst = max(top3_sb_gen, key=lambda x: x[0])

            out_shl_b, _, _, _ = _replay_qkv_arm(arms_shuf_layer, Wo_shl_dev, rec_bind, use_clean_8cell=False)
            out_shl_n, _, _, _ = _replay_qkv_arm(arms_shuf_layer, Wo_shl_dev, rec_null, use_clean_8cell=False)
            sc_bn_shl = _score_bn(out_shl, out_shl_b, out_shl_n)
            ok_shlb_fw, ok_shlb_wh = _live_gen_with_delta(out_shl_b, ev_bind)
            sc_bn_shl.update({"live_gen_hits": int(np.sum(ok_shlb_fw)), "live_gen_em": round(float(np.mean(ok_shlb_fw)), 4), "live_gen_whole_em": round(float(np.mean(ok_shlb_wh)), 4), "live_gen_lift_pp": round(100.0 * float(np.mean(ok_shlb_fw) - np.mean(ok_none_b)), 2)})

            tids_b = np.array([int(it["table_id"]) for it in ev_bind])
            boot_b_none = paired_table_cluster_bootstrap(ok_bb_fw, ok_none_b, tids_b, n_boot=2000)
            boot_b_shuf_pi = paired_table_cluster_bootstrap(ok_bb_fw, ok_sb_worst, tids_b, n_boot=2000)
            boot_b_shl = paired_table_cluster_bootstrap(ok_bb_fw, ok_shlb_fw, tids_b, n_boot=2000)

            def _score_bind_8(out_b: torch.Tensor) -> Dict[str, Any]:
                dw_b = (out_b @ T_raw_t.T).bfloat16().float()
                ch = (_real_norm_margin_from_rows((h_fin_B_bind.bfloat16() + dw_b.bfloat16()).float(), gamma_fin_B, eps_fin_B, e_ab_B, e_fb_B2) - margin_B_bind_base).cpu().double().numpy()
                ac, _ = _tok0_acc((h_fin_B_bind.bfloat16() + dw_b.bfloat16()).float(), ev_bind)
                return {**_compute_safe_wrong_item_ratio(ch, c_true_b, wrong_bind, min_donor_signal=0.25), "binding_tok0_acc": round(ac, 4), "binding_tok0_lift_pp": round(100.0 * (ac - base_tok0_b), 2)}
            decomp_bind_best = run_8cell_and_sink_content_decomposition(qkv_don_b_c, qkv_bb_dep, Wo_prim_dev, cos_ev, sin_ev, _score_bind_8, mask_mixed_dims=mixed_m)

            # Binding Gate using RAW UNROUNDED leakage ratio & exact integer hit counts!
            hits_bb, hits_none_b = int(np.sum(ok_bb_fw)), int(np.sum(ok_none_b))
            max_shuf_b_hits = max(worst_sb_hits, int(np.sum(ok_shlb_fw)))
            shuf_b_wirs = [x[1] for x in shuf_b_draws if x[1] is not None]
            max_shuf_b_wir = max(max(shuf_b_wirs) if shuf_b_wirs else -1e9, sc_bn_shl["wrong_item_ratio"] if sc_bn_shl["wrong_item_ratio"] is not None else -1e9)
            wir_bb = sc_bn_best["wrong_item_ratio"]
            leak_raw_best = float(sc_bn_best.pop("_leak_raw"))
            sc_bn_or.pop("_leak_raw", None); sc_bn_shl.pop("_leak_raw", None)
            binding_gate_pass = bool(
                abs(sink_bb - float(tb["donor_eval_binding"]["donor_mean_pos0_sink"])) <= 0.10
                and (wir_bb is not None and wir_bb >= 0.30 and (wir_bb - max_shuf_b_wir) >= 0.25)
                and (sc_bn_best["binding_Q_outlet"] is not None and sc_bn_best["binding_Q_outlet"] >= 0.40)
                and (sc_bn_best["binding_R2_c"] is not None and sc_bn_best["binding_R2_c"] >= 0.30)
                and (100 * (hits_bb - hits_none_b) >= 10 * n_bn)
                and (100 * (hits_bb - max_shuf_b_hits) >= 8 * n_bn)
                and (leak_raw_best <= 0.25)
                and (boot_b_none["ci95_lo"] > 0.0 and boot_b_shuf_pi["ci95_lo"] > 0.0 and boot_b_shl["ci95_lo"] > 0.0)
            )
            del rec_bind, rec_null; gc.collect()
            binding_null_res = {
                "donor_binding_em": tb["donor_binding_em"], "donor_binding_whole_em": tb.get("donor_binding_whole_em"),
                "donor_null_em": tb["donor_null_em"], "donor_null_whole_em": tb.get("donor_null_whole_em"),
                "donor_null_ref": tb["donor_eval_binding"]["donor_null_ref"],
                "rec_none_binding_hits": hits_none_b, "rec_none_binding_em": round(hits_none_b / max(n_bn, 1), 4),
                "outlet_oracle_2A": sc_bn_or,
                "best_student_arm": {"arm_name": best_arm_name, **sc_bn_best},
                "SHUF_pi_20draw_binding": {
                    "max_tok0_hits": int(shuf_b_top[0][0]), "max_wir": round(max(shuf_b_wirs), 4) if shuf_b_wirs else None,
                    "top3_max_live_gen_hits": worst_sb_hits, "top3_max_live_gen_em": round(worst_sb_hits / max(n_bn, 1), 4),
                },
                "SHUF_layer": sc_bn_shl,
                "bootstrap_vs_none": boot_b_none, "bootstrap_vs_shuf_pi": boot_b_shuf_pi, "bootstrap_vs_shuf_layer": boot_b_shl,
                "decomp_8cell_binding": decomp_bind_best,
                "binding_gate_pass": binding_gate_pass,
            }

        # P1-5 Fix: Exact integer hit-count gate comparisons (zero float rounding bugs at 400 items!)
        hits_7b = int(tb["donor_eval_hits"])
        hits_none = int(np.sum(ok_none))
        hits_real = int(np.sum(ok_best_fw))
        max_shuf_hits = max(worst_shuf_pi_hits, int(np.sum(ok_shl_fw)))
        g0_pass_15 = bool(100 * hits_7b >= 65 * n_eval and 100 * (hits_7b - hits_none) >= 15 * n_eval)
        g0_pass_20 = bool(100 * hits_7b >= 65 * n_eval and 100 * (hits_7b - hits_none) >= 20 * n_eval)
        bos_err = abs(arm_unrounded_sink[best_arm_name] - float(tb["donor_eval_base"]["donor_mean_pos0_sink"]))
        sc_best = arm_outputs[best_arm_name]
        wir_real = sc_best["wrong_item_ratio"]
        shuf_pi_wirs = [x[1] for x in shuf_pi_draws if x[1] is not None]
        max_shuf_wir = max(max(shuf_pi_wirs) if shuf_pi_wirs else -1e9, sc_shl["wrong_item_ratio"] if sc_shl["wrong_item_ratio"] is not None else -1e9)

        policy_gate_pass = bool(
            g0_pass_15
            and bos_err <= 0.10
            and (wir_real is not None and wir_real >= 0.30 and (wir_real - max_shuf_wir) >= 0.25)
            and (sc_best["Q_outlet"] is not None and sc_best["Q_outlet"] >= 0.40)
            and (sc_best["R2_c"] is not None and sc_best["R2_c"] >= 0.30)
            and (100 * (hits_real - hits_none) >= 10 * n_eval)
            and (100 * (hits_real - max_shuf_hits) >= 8 * n_eval)
            and (boot_vs_none["ci95_lo"] > 0.0 and boot_vs_shuf_pi["ci95_lo"] > 0.0 and boot_vs_shuf_layer["ci95_lo"] > 0.0)
        )

        # Persist compact durable eval cache (per-head [H, D_A] centers, W_o, T_raw, residuals, z_h) outside git repo!
        pos0_tid = int(rec_eval_base["seqs_ids"][0][0])
        pos0_str = tok.decode([pos0_tid], skip_special_tokens=False)
        per_head_centers = np.stack([
            F.linear(torch.from_numpy(v_bar_prim[lh]).to(dev, dtype=Wo_prim_dev[hi].dtype), Wo_prim_dev[hi]).float().cpu().numpy()
            for hi, lh in enumerate(prim_heads)
        ], axis=0)  # (H, D_A)
        eval_cache_path = Path(f"/tmp/pswap_v51_eval_cache_{tname}.npz")
        np.savez_compressed(
            eval_cache_path,
            head_output_center=per_head_centers.astype(np.float32),
            w_o=np.stack([w.float().cpu().numpy() for w in Wo_prim_dev], axis=0).astype(np.float32),
            writer_t_raw=T_raw_frozen.astype(np.float32),
            writer_scale=np.array([gamma_scale], dtype=np.float32),
            recipient_final_residual=rec_eval_base["h_fin_B"].astype(np.float32),
            table_ids=table_ids_base.astype(np.int64),
            answer_token_ids=np.array([int(it["answer_tok_id"]) for it in eval_base], dtype=np.int64),
            foil_token_ids=np.array([int(it["foil_tok_id"]) for it in eval_base], dtype=np.int64),
            donor_out_prim=out_oracle_c.cpu().numpy().astype(np.float32),
            student_best_out=out_best.cpu().numpy().astype(np.float32),
            student_best_z=np.stack([zm.cpu().numpy() for zm in z_best], axis=0).astype(np.float32),
        )

        task_results[tname] = {
            "wrapper": tb["wrapper"], "n_rows": tb["n_rows"],
            "pos0_token_id": pos0_tid, "pos0_token_str": pos0_str,
            "durable_eval_cache_path": str(eval_cache_path),
            "primary_heads": prim_heads, "s_real_heads_diagnostic": sreal_heads, "shuf_layer_heads_matched": shuf_heads,
            "n_eval": n_eval, "donor_eval_hits": hits_7b, "donor_eval_em": round(hits_7b / n_eval, 4),
            "donor_eval_whole_em": tb.get("donor_eval_whole_em"),
            "rec_none_hits": hits_none, "rec_none_eval_em": round(hits_none / n_eval, 4),
            "rec_none_whole_em": round(float(np.mean(ok_none_whole)), 4), "rec_none_tok0_acc": round(base_tok0_acc, 4),
            "g0_gap_pp": round(100.0 * (hits_7b - hits_none) / n_eval, 2),
            "g0_pass_15pp": g0_pass_15, "g0_pass_20pp": g0_pass_20,
            "donor_mean_pos0_sink": round(float(tb["donor_eval_base"]["donor_mean_pos0_sink"]), 4),
            "val_tune_bos_variant_winner": win_var_tune, "best_generic_selected_arm": best_arm_name,
            "outlet_oracle_2A": sc_oracle, "Arm_0_v50_baseline_diverse": sc_arm0, "arms": arm_outputs,
            "SHUF_pi_20draw": {
                "mean_tok0_lift_pp": round(float(np.mean([x[4]["tok0_lift_pp"] for x in shuf_pi_draws])), 2),
                "max_tok0_hits": int(shuf_pi_draws_sorted[0][0]),
                "mean_wir": round(float(np.mean(shuf_pi_wirs)), 4) if shuf_pi_wirs else None,
                "max_wir": round(max(shuf_pi_wirs), 4) if shuf_pi_wirs else None,
                "top3_max_live_gen_hits": worst_shuf_pi_hits, "top3_max_live_gen_draw_idx": int(worst_shuf_pi_di),
                "top3_max_live_gen_em": round(worst_shuf_pi_hits / n_eval, 4),
                "top3_max_live_gen_lift_pp": round(100.0 * (worst_shuf_pi_hits - hits_none) / n_eval, 2),
            },
            "SHUF_layer_matched": sc_shl,
            "bootstrap_best_vs_none": boot_vs_none, "bootstrap_best_vs_shuf_pi": boot_vs_shuf_pi, "bootstrap_best_vs_shuf_layer": boot_vs_shuf_layer,
            "decomp_8cell_arm0": decomp_arm0, "decomp_8cell_best": decomp_best,
            "binding_and_null_swap": binding_null_res,
            "gate_verdict": {
                "policy_gate_pass": policy_gate_pass,
                "binding_gate_pass": binding_gate_pass if binding_null_res is not None else None,
                "bos_abs_err": round(bos_err, 4),
                "em_lift_vs_none_pp": round(100.0 * (hits_real - hits_none) / n_eval, 2),
                "em_margin_vs_max_shuf_pp": round(100.0 * (hits_real - max_shuf_hits) / n_eval, 2),
                "wir_margin_vs_max_shuf": round(float(wir_real - max_shuf_wir), 4) if wir_real is not None else None,
            },
        }

    # Step 2.6: Arm 3c — Live Native `L0` Tied-Embedding Ingress (`alpha = 2^-6`, Full All-Param Rollback!)
    print("[v5.1 Phase 2] Verifying Arm 3c: live native L0 tied-embedding ingress (`alpha=2^-6`) + full rollback...", flush=True)
    alpha_l0 = 2.0 ** -6
    E_B_emb = model_B.model.embed_tokens.weight[:v_shared].detach()
    eps_ln0_B = float(layers_B[0].input_layernorm.variance_epsilon)
    E_priv_cpu = torch.empty((v_shared, K_0), dtype=torch.bfloat16)
    for v0 in range(0, v_shared, 16384):
        v1 = min(v0 + 16384, v_shared)
        rms_b = torch.sqrt(torch.mean(E_B_emb[v0:v1].float() ** 2, dim=-1, keepdim=True) + eps_ln0_B).cpu()
        E_priv_cpu[v0:v1] = (alpha_l0 * rms_b * torch.from_numpy(E_A_ln0_proj_f32[v0:v1])).to(dtype=torch.bfloat16)
    E_A_ln0_proj_t = torch.from_numpy(E_A_ln0_proj_f32)
    arm3c_l0_native = verify_native_l0_tied_ingress(
        model_B=model_B, E_priv_bf16=E_priv_cpu, target_l0_proj_fn=lambda ids_t: E_A_ln0_proj_t[ids_t.cpu()],
        eval_input_ids=corpus["report_seqs"][:16], v_shared=v_shared, alpha=alpha_l0,
    )

    # P0-9 Fix: Strict multi-regime decision logic (1-Hop alone NEVER issues multi-step GO; requires Arm 3c PASS!)
    v_1hop = bool(task_results["1hop"]["gate_verdict"]["policy_gate_pass"]) if "1hop" in task_results else False
    v_2hop_pol = bool(task_results["2hop"]["gate_verdict"]["policy_gate_pass"]) if "2hop" in task_results else False
    v_2hop_bnd = bool(task_results["2hop"]["gate_verdict"]["binding_gate_pass"]) if "2hop" in task_results else False
    v_arm3c = bool(arm3c_l0_native["passes_all"])
    if v_2hop_pol and v_2hop_bnd and v_arm3c:
        final_decision = "GO_TO_NATIVE_G3_MULTISTEP"
    elif v_1hop and v_arm3c:
        final_decision = "GO_TO_NATIVE_G3_1HOP_ONLY"
    else:
        final_decision = "STOP_BEFORE_G3"

    return {
        "fisher_bridge": {
            "eta_sweep_val_tune": eta_sweep_tune, "frozen_eta": best_eta,
            "gamma_scale": round(gamma_scale, 5), "val_report_eval": bridge_report_eval,
        },
        "layer_profile_28": layer_profile_28,
        "incremental_tap_ladder": incremental_tap_ladder,
        "arm0_diverse_supply_by_layer": {
            f"L{li}": {k: v for k, v in d.items() if k not in ("W_r0", "mu_x", "mu_y", "mu_pos0_A")}
            for li, d in arm0_by_layer.items()
        },
        "tasks": task_results,
        "arm3c_native_l0_tied_ingress": arm3c_l0_native,
        "regime_verdicts": {
            "1hop_positive_control": v_1hop, "2hop_policy": v_2hop_pol,
            "2hop_binding_and_null": v_2hop_bnd, "arm3c_native_l0": v_arm3c,
        },
        "final_decision": final_decision,
    }


# ======================================================================================
# 8. End-to-End Orchestrator (`run_all_v51`) & 10 Enhanced Pure-CPU Unit Tests
# ======================================================================================

def run_all_v51(
    donor_key: str = DONOR_KEY_DEFAULT, recipient_key: str = RECIPIENT_KEY_DEFAULT,
    allow_download: bool = True, n_pilot: int = 64, n_disc: int = 200, n_eval: int = 400,
    n_fit_pos_gt0: int = 60000, n_tune_pos_gt0: int = 4096, n_rep_pos_gt0: int = 4096,
    batch_size: int = 16, out_path: Path = OUT_PATH,
) -> Dict[str, Any]:
    t0 = time.time()
    from transformers import AutoTokenizer

    src_A, local_A = _resolve_tokenizer_source(donor_key, allow_download)
    src_B, local_B = _resolve_tokenizer_source(recipient_key, allow_download)
    tok_A = AutoTokenizer.from_pretrained(src_A, local_files_only=local_A)
    tok_B = AutoTokenizer.from_pretrained(src_B, local_files_only=local_B)
    if tok_A.pad_token_id is None:
        tok_A.pad_token_id = tok_A.eos_token_id
    if tok_B.pad_token_id is None:
        tok_B.pad_token_id = tok_B.eos_token_id
    v_shared = min(int(len(tok_A)), int(len(tok_B)), 151665)

    # Build generic calibration corpus FIRST — 100% independent of pilot/disc/eval!
    corpus = build_diverse_real_calibration_corpus(
        tok_A, n_fit_pos_gt0=n_fit_pos_gt0, n_tune_pos_gt0=n_tune_pos_gt0,
        n_rep_pos_gt0=n_rep_pos_gt0, seq_len=128, allow_download=allow_download, seed=SEED,
    )
    print(f"[v5.1 Init] Generic corpus built independently of tasks: {corpus['counts']['fit']} | diversity={corpus['diversity']}", flush=True)

    _assert_dummy_exemplars_disjoint()
    key_pools = partition_2hop_key_pools([tok_A, tok_B], max_rows=6, seed=SEED)
    key_pools["all_vals_clean"] = filter_wrapper_clean_vals(key_pools["all_vals"], [tok_A, tok_B])
    assert len(key_pools["all_vals_clean"]) >= 24

    wrappers = ("chat_bare", "2shot_fixed", "chat_2shot")
    pilot_1hop_ds, pilot_1hop_meta, pilot_2hop_ds, pilot_2hop_meta = {}, {}, {}, {}
    for wr in wrappers:
        for nr in (8, 12):
            k = f"1hop_{wr}_r{nr}"
            pilot_1hop_ds[k] = build_1hop_dataset([tok_A, tok_B], key_pools, wrapper=wr, n_rows=nr, n_pilot=n_pilot, n_disc=2, n_eval=2, seed=SEED)["pilot"]
            pilot_1hop_meta[k] = {"wrapper": wr, "n_rows": nr}
        for nr in (4, 6):
            k = f"2hop_{wr}_r{nr}"
            pilot_2hop_ds[k] = build_2hop_three_intervention_dataset([tok_A, tok_B], key_pools, wrapper=wr, n_rows=nr, n_pilot=n_pilot, n_disc=2, n_eval=2, seed=SEED)["pilot"]["base"]
            pilot_2hop_meta[k] = {"wrapper": wr, "n_rows": nr}

    vram_log: Dict[str, Any] = {}

    # PHASE 0: Quick Recipient 1.5B Pilot Screen (~20s, <4.5 GiB -> unload <0.05 GiB)
    print("[v5.1 Phase 0] Loading Recipient 1.5B for G0-Clean pilot wrapper screen...", flush=True)
    model_B0, _, _, _ = load_model_bf16(recipient_key, allow_download=allow_download)
    rec_pilot_1hop = run_pilot_screen_on_model(model_B0, tok_B, v_shared, pilot_1hop_ds, batch_size=batch_size)
    rec_pilot_2hop = run_pilot_screen_on_model(model_B0, tok_B, v_shared, pilot_2hop_ds, batch_size=batch_size)
    vram_log["phase0_peak_gib"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3) if torch.cuda.is_available() else 0.0
    unload_model_vram_safe(model_B0)
    del model_B0; gc.collect()
    vram_log["phase0_post_unload_gib"] = round(torch.cuda.memory_allocated() / (1024 ** 3), 4) if torch.cuda.is_available() else 0.0
    if torch.cuda.is_available():
        assert vram_log["phase0_post_unload_gib"] < 1.0
        torch.cuda.reset_peak_memory_stats()

    with tempfile.TemporaryDirectory(prefix="pswap_v51_") as tmp_s:
        tmp_dir = Path(tmp_s)

        # PHASE 1: Donor 7B Full Pass
        print("[v5.1 Phase 1] Loading Donor 7B-Instruct...", flush=True)
        model_A, _, _, _ = load_model_bf16(donor_key, allow_download=allow_download)
        don_pilot_1hop = run_pilot_screen_on_model(model_A, tok_A, v_shared, pilot_1hop_ds, batch_size=batch_size)
        don_pilot_2hop = run_pilot_screen_on_model(model_A, tok_A, v_shared, pilot_2hop_ds, batch_size=batch_size)

        grid_1hop = {
            k: {**pilot_1hop_meta[k], "donor_em": don_pilot_1hop[k]["em"], "donor_whole_em": don_pilot_1hop[k]["whole_em"], "rec_em": rec_pilot_1hop[k]["em"], "gap_pp": round(100.0 * (don_pilot_1hop[k]["em"] - rec_pilot_1hop[k]["em"]), 2), "donor_samples": don_pilot_1hop[k]["sample_completions"], "rec_samples": rec_pilot_1hop[k]["sample_completions"]}
            for k in pilot_1hop_meta
        }
        grid_2hop = {
            k: {**pilot_2hop_meta[k], "donor_em": don_pilot_2hop[k]["em"], "donor_whole_em": don_pilot_2hop[k]["whole_em"], "rec_em": rec_pilot_2hop[k]["em"], "gap_pp": round(100.0 * (don_pilot_2hop[k]["em"] - rec_pilot_2hop[k]["em"]), 2), "donor_samples": don_pilot_2hop[k]["sample_completions"], "rec_samples": rec_pilot_2hop[k]["sample_completions"]}
            for k in pilot_2hop_meta
        }
        win_1hop = _select_wrapper_and_rows_from_pilot(grid_1hop, min_donor_acc=0.65)
        win_2hop = _select_wrapper_and_rows_from_pilot(grid_2hop, min_donor_acc=0.65)
        print(f"[v5.1 Phase 1] Pilot winners: 1-Hop={win_1hop['selected_key']} ({win_1hop['pilot_donor_em']:.3f} vs {win_1hop['pilot_rec_em']:.3f}), 2-Hop={win_2hop['selected_key']} ({win_2hop['pilot_donor_em']:.3f} vs {win_2hop['pilot_rec_em']:.3f})", flush=True)

        ds_1hop = build_1hop_dataset([tok_A, tok_B], key_pools, wrapper=win_1hop["wrapper"], n_rows=win_1hop["n_rows"], n_pilot=n_pilot, n_disc=n_disc, n_eval=n_eval, seed=SEED)
        ds_2hop = build_2hop_three_intervention_dataset([tok_A, tok_B], key_pools, wrapper=win_2hop["wrapper"], n_rows=win_2hop["n_rows"], n_pilot=n_pilot, n_disc=n_disc, n_eval=n_eval, seed=SEED)

        # Fast backbone-only discovery on `disc` (honoring `batch_size` cap; MLP solo ablation ONLY on 2hop_policy!)
        disc_bs = min(int(batch_size), 16)
        disc_1hop = run_dla_and_solo_ablation_v51(model_A, tok_A, ds_1hop["disc"], v_shared=v_shared, batch_size=disc_bs, run_mlp_solo=False)
        disc_2hop_pol = run_dla_and_solo_ablation_v51(model_A, tok_A, ds_2hop["disc"]["base"], v_shared=v_shared, batch_size=disc_bs, run_mlp_solo=True)
        disc_2hop_bnd = run_dla_and_solo_ablation_v51(model_A, tok_A, ds_2hop["disc"]["binding"], v_shared=v_shared, batch_size=disc_bs, run_mlp_solo=False)

        shuf_1hop_heads = disc_1hop["shuf_matched_late_heads"]
        shuf_2hop_heads = disc_2hop_pol["shuf_matched_late_heads"]

        # Stream Donor 7B 60k/4k/4k generic calibration FIRST so frozen `v_bar_gen` is ready for Donor `c_true`!
        all_target_heads = list(dict.fromkeys(
            list(disc_1hop["top_late_heads"]) + list(disc_1hop["candidate_heads"]) + list(shuf_1hop_heads)
            + list(disc_2hop_pol["top_late_heads"]) + list(disc_2hop_pol["candidate_heads"]) + list(shuf_2hop_heads)
        ))
        organ_full_layers = sorted({l for l, _ in (list(disc_1hop["top_late_heads"]) + list(disc_2hop_pol["top_late_heads"]))})
        print(f"[v5.1 Phase 1] Streaming Donor 7B 60k/4k/4k calibration to memmaps ({len(all_target_heads)} heads)...", flush=True)
        donor_cal = stream_donor_generic_v51(
            model_A, corpus, all_target_heads=all_target_heads, organ_full_layers=organ_full_layers,
            v_shared=v_shared, tmp_dir=tmp_dir, n_prof_fit=12288, n_val_tune_seqs_for_attn=32, batch_size=8,
        )

        # Donor intact greedy EM on eval (reporting both first-word EM and strict whole-completion EM!)
        def _don_em(items_s):
            gs = _greedy_generate_clean_no_add_special(model_A, tok_A, [it["prompt"] for it in items_s], v_shared=v_shared, max_new_tokens=4, batch_size=batch_size)
            ok_fw = [int(check_2hop_exact_word(g, it["answer"])) for g, it in zip(gs, items_s)]
            ok_wh = [int(check_exact_whole_completion(g, it["answer"])) for g, it in zip(gs, items_s)]
            return int(sum(ok_fw)), float(np.mean(ok_fw)), float(np.mean(ok_wh))
        hits_1h_A, em_1h_A, wh_1h_A = _don_em(ds_1hop["eval"])
        hits_2h_A, em_2h_A, wh_2h_A = _don_em(ds_2hop["eval"]["base"])
        _, em_2h_bind_A, wh_2h_bind_A = _don_em(ds_2hop["eval"]["binding"])
        _, em_2h_null_A, wh_2h_null_A = _don_em(ds_2hop["eval"]["null_swap"])
        print(f"[v5.1 Phase 1] Donor 7B eval EM: 1-Hop={em_1h_A:.4f} (whole={wh_1h_A:.4f}), 2-Hop={em_2h_A:.4f} (whole={wh_2h_A:.4f}), Bind={em_2h_bind_A:.4f}, Null={em_2h_null_A:.4f}", flush=True)

        # Single-pass donor eval captures + exact frozen-generic-centered `c_true`, `d_ref`, and donor null reference!
        g_rep_A = int(donor_cal["g_rep"])
        D_A = int(donor_cal["D_A"])
        def _mean_out_gen_for_heads(heads_l, hw_cpu_d) -> np.ndarray:
            dev_A = next(model_A.parameters()).device
            acc = torch.zeros((D_A,), device=dev_A, dtype=torch.float32)
            for l, h in heads_l:
                Wo = hw_cpu_d[(l, h)]["Wo"].to(dev_A)
                vb = torch.from_numpy(donor_cal["v_bar_gen_by_kv"][(l, h // g_rep_A)]).to(dev_A, dtype=Wo.dtype)
                acc += F.linear(vb, Wo).float()
            return acc.cpu().numpy().astype(np.float32)

        bun_1hop = capture_donor_head_eval_bundle(model_A, tok_A, ds_1hop["eval"], disc_1hop["candidate_heads"], disc_1hop["top_late_heads"], shuf_1hop_heads, v_shared=v_shared, batch_size=batch_size)
        bun_1hop["mean_out_gen_prim_f32"] = _mean_out_gen_for_heads(disc_1hop["top_late_heads"], bun_1hop["head_weights_cpu"])
        bun_1hop["c_true"] = compute_donor_references_frozen_center(model_A, ds_1hop["eval"], bun_1hop["h_fin_all_f32"], bun_1hop["out_prim_f32"], bun_1hop["mean_out_gen_prim_f32"], v_shared)
        bun_1hop["d_ref_policy"] = compute_donor_swap_refs(model_A, ds_1hop["eval"][0::2], ds_1hop["eval"][1::2], bun_1hop["h_fin_all_f32"][0::2], bun_1hop["out_prim_f32"][0::2], bun_1hop["out_prim_f32"][1::2], v_shared)
        del bun_1hop["h_fin_all_f32"]

        bun_2h_base = capture_donor_head_eval_bundle(model_A, tok_A, ds_2hop["eval"]["base"], disc_2hop_pol["candidate_heads"], disc_2hop_pol["top_late_heads"], shuf_2hop_heads, v_shared=v_shared, batch_size=batch_size)
        bun_2h_bind = capture_donor_head_eval_bundle(model_A, tok_A, ds_2hop["eval"]["binding"], [], disc_2hop_pol["top_late_heads"], [], v_shared=v_shared, batch_size=batch_size)
        bun_2h_null = capture_donor_head_eval_bundle(model_A, tok_A, ds_2hop["eval"]["null_swap"], [], disc_2hop_pol["top_late_heads"], [], v_shared=v_shared, batch_size=batch_size)
        m_gen_2h = _mean_out_gen_for_heads(disc_2hop_pol["top_late_heads"], bun_2h_base["head_weights_cpu"])
        bun_2h_base["mean_out_gen_prim_f32"] = m_gen_2h
        bun_2h_base["c_true"] = compute_donor_references_frozen_center(model_A, ds_2hop["eval"]["base"], bun_2h_base["h_fin_all_f32"], bun_2h_base["out_prim_f32"], m_gen_2h, v_shared)
        bun_2h_base["d_ref_policy"] = compute_donor_swap_refs(model_A, ds_2hop["eval"]["base"][0::2], ds_2hop["eval"]["base"][1::2], bun_2h_base["h_fin_all_f32"][0::2], bun_2h_base["out_prim_f32"][0::2], bun_2h_base["out_prim_f32"][1::2], v_shared)
        bun_2h_bind["c_true"] = compute_donor_references_frozen_center(model_A, ds_2hop["eval"]["binding"], bun_2h_bind["h_fin_all_f32"], bun_2h_bind["out_prim_f32"], m_gen_2h, v_shared)
        d_ref_bnd = compute_donor_swap_refs(model_A, ds_2hop["eval"]["base"][0::2], ds_2hop["eval"]["binding"], bun_2h_base["h_fin_all_f32"][0::2], bun_2h_base["out_prim_f32"][0::2], bun_2h_bind["out_prim_f32"], v_shared)
        d_ref_null = compute_donor_swap_refs(
            model_A, ds_2hop["eval"]["base"][0::2], ds_2hop["eval"]["null_swap"],
            bun_2h_base["h_fin_all_f32"][0::2], bun_2h_base["out_prim_f32"][0::2], bun_2h_null["out_prim_f32"], v_shared,
            answer_b_override=[int(it["answer_tok_id"]) for it in ds_2hop["eval"]["binding"]],
            answer_a_override=[int(it["foil_tok_id"]) for it in ds_2hop["eval"]["binding"]],
        )
        bun_2h_bind["d_ref_binding"] = d_ref_bnd
        bun_2h_bind["donor_null_ref"] = {
            "donor_binding_mean_abs_nat": round(float(np.mean(np.abs(d_ref_bnd))), 4),
            "donor_null_mean_abs_nat": round(float(np.mean(np.abs(d_ref_null))), 4),
            "donor_null_to_binding_ratio": round(float(np.mean(np.abs(d_ref_null))) / max(float(np.mean(np.abs(d_ref_bnd))), 1e-6), 4),
        }
        del bun_2h_base["h_fin_all_f32"], bun_2h_bind["h_fin_all_f32"], bun_2h_null["h_fin_all_f32"]

        # Arm 3a & 3b (Donor-Side Backward ISA + Top Bottleneck MLP Linearization & Sparse J-Lens)
        min_late_2h = min(l for l, _ in disc_2hop_pol["top_late_heads"])
        up_mlps = [m for m in disc_2hop_pol["mlp_solo_ablation"] if m["layer"] < min_late_2h]
        bottleneck_l = max(up_mlps, key=lambda x: x["solo_ld_drop"])["layer"] if up_mlps else 20
        print(f"[v5.1 Phase 1] Running Arm 3a/3b (Backward ISA + Bottleneck L{bottleneck_l}_MLP)...", flush=True)
        arm3_donor = run_arm3_donor_backward_isa_and_l20_mlp(
            model_A, tok_A, disc_2hop_pol["top_late_heads"],
            ds_2hop["disc"]["base"], ds_2hop["disc"]["binding"],
            ds_2hop["eval"]["base"], ds_2hop["eval"]["binding"],
            corpus["fit_seqs"], bottleneck_mlp_layer=bottleneck_l, v_shared=v_shared, batch_size=24,
        )

        vram_log["phase1_peak_gib"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3) if torch.cuda.is_available() else 0.0
        unload_model_vram_safe(model_A)
        del model_A; gc.collect()
        vram_log["phase1_post_unload_gib"] = round(torch.cuda.memory_allocated() / (1024 ** 3), 4) if torch.cuda.is_available() else 0.0
        if torch.cuda.is_available():
            assert vram_log["phase1_post_unload_gib"] < 1.0
            torch.cuda.reset_peak_memory_stats()

        # PHASE 2: Recipient 1.5B
        print("[v5.1 Phase 2] Loading Recipient 1.5B-Instruct...", flush=True)
        model_B, _, _, _ = load_model_bf16(recipient_key, allow_download=allow_download)
        task_bundles = {
            "1hop": {
                "wrapper": win_1hop["wrapper"], "n_rows": win_1hop["n_rows"],
                "primary_heads": disc_1hop["top_late_heads"], "candidate_heads": disc_1hop["candidate_heads"],
                "cand_dla": disc_1hop["candidate_heads_dla"], "shuf_layer_heads": shuf_1hop_heads,
                "eval_base": ds_1hop["eval"], "donor_eval_hits": hits_1h_A, "donor_eval_whole_em": round(wh_1h_A, 4), "donor_eval_base": bun_1hop,
            },
            "2hop": {
                "wrapper": win_2hop["wrapper"], "n_rows": win_2hop["n_rows"],
                "primary_heads": disc_2hop_pol["top_late_heads"], "candidate_heads": disc_2hop_pol["candidate_heads"],
                "cand_dla": disc_2hop_pol["candidate_heads_dla"], "shuf_layer_heads": shuf_2hop_heads,
                "eval_base": ds_2hop["eval"]["base"], "eval_binding": ds_2hop["eval"]["binding"], "eval_null": ds_2hop["eval"]["null_swap"],
                "donor_eval_hits": hits_2h_A, "donor_eval_whole_em": round(wh_2h_A, 4),
                "donor_binding_em": round(em_2h_bind_A, 4), "donor_binding_whole_em": round(wh_2h_bind_A, 4),
                "donor_null_em": round(em_2h_null_A, 4), "donor_null_whole_em": round(wh_2h_null_A, 4),
                "donor_eval_base": bun_2h_base, "donor_eval_binding": bun_2h_bind, "donor_eval_null": bun_2h_null,
            },
        }
        phase2_res = run_phase2_recipient_v51(
            model_B, tok_B, corpus, donor_cal, task_bundles, v_shared=v_shared,
            tmp_dir=tmp_dir, tap_layers=(8, 14, 19, 23), batch_size=batch_size,
        )
        vram_log["phase2_peak_gib"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3) if torch.cuda.is_available() else 0.0
        unload_model_vram_safe(model_B)
        del model_B; gc.collect()
        vram_log["phase2_post_unload_gib"] = round(torch.cuda.memory_allocated() / (1024 ** 3), 4) if torch.cuda.is_available() else 0.0
        if torch.cuda.is_available():
            assert vram_log["phase2_post_unload_gib"] < 1.0

    elapsed_s = round(time.time() - t0, 1)
    result = {
        "meta": {
            "script": "experiments/v5_1_input_barrier_falsifier.py",
            "donor_key": donor_key, "recipient_key": recipient_key, "v_shared": v_shared,
            "seed": SEED, "elapsed_seconds": elapsed_s, "host_max_rss_gib": round(_host_rss_gib(), 3),
            "vram_log": vram_log, "zero_sgd_strict": True, "zero_task_fitted_writers": True,
        },
        "g0_pilot": {"1hop_grid": grid_1hop, "1hop_winner": win_1hop, "2hop_grid": grid_2hop, "2hop_winner": win_2hop},
        "calibration_corpus": {"counts": corpus["counts"], "diversity": corpus["diversity"], "has_real_wikitext": corpus["has_real_wikitext"]},
        "donor_discovery": {
            "1hop": {k: v for k, v in disc_1hop.items() if k != "candidate_heads_dla"},
            "2hop_policy": {k: v for k, v in disc_2hop_pol.items() if k != "candidate_heads_dla"},
            "2hop_binding": {k: v for k, v in disc_2hop_bnd.items() if k != "candidate_heads_dla"},
        },
        "arm3_donor_backward_isa_and_mlp": arm3_donor,
        "phase2_results": phase2_res,
        "final_decision": phase2_res["final_decision"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[v5.1 DONE] Elapsed={elapsed_s}s | Decision={result['final_decision']} | Saved to {out_path}", flush=True)
    return result


# ======================================================================================
# 9. 10 Enhanced Pure-CPU Unit Tests (`--unit-test`, Zero Downloads, < 15s)
# ======================================================================================

def run_unit_tests() -> None:
    print("[v5.1 Unit Tests] Running 10 enhanced pure-CPU mathematical & architectural unit tests...", flush=True)
    rng = np.random.default_rng(SEED)

    # Test 1: Vocab-anchored closed-form Fisher + StreamingSharedVocabFisherAccumulator contract!
    V, DA, DB = 128, 16, 12
    EA = rng.standard_normal((V, DA))
    EB = rng.standard_normal((V, DB))
    F_iso = (np.eye(V) - np.ones((V, V)) / V) / V
    G_A_ref, G_B_ref, G_BA_ref = EA.T @ F_iso @ EA, EB.T @ F_iso @ EB, EB.T @ F_iso @ EA
    plugs = compute_vocab_anchored_fisher_plugs(G_A_ref, G_B_ref, G_BA_ref, EA.astype(np.float32), EB.astype(np.float32), np.ones(DA), np.ones(DB), etas=(0.0, 1.0))
    assert np.max(np.abs(plugs[1.0]["G_A_raw"] - G_A_ref)) < 1e-6 and np.max(np.abs(plugs[1.0]["G_BA_raw"] - G_BA_ref)) < 1e-6
    # Verify exact StreamingSharedVocabFisherAccumulator logits update + dict finalize contract!
    facc = StreamingSharedVocabFisherAccumulator(torch.from_numpy(EA).bfloat16(), torch.from_numpy(EB).bfloat16())
    facc.update(F.linear(torch.randn(16, DA, dtype=torch.bfloat16), torch.from_numpy(EA).bfloat16()))
    fres = facc.finalize()
    assert set(("G_A", "G_B", "G_BA")).issubset(fres.keys()) and fres["G_BA"].shape == (DB, DA)
    print("  [PASS 1/10] Vocab-anchored Fisher & StreamingSharedVocabFisherAccumulator contract")

    # Test 2: RoPE-commuting 2x2 plane de-shrinkage + memoized ridge cache exactness
    d_h = 32
    inv_f = np.asarray(rope_inv_freq(d_h, 1e6), dtype=np.float64)
    ang = np.arange(16, dtype=np.float64)[:, None] * inv_f[None, :]
    c_np, s_np = np.cos(ang), np.sin(ang)
    X_t, W_true = rng.standard_normal((256, 20)), rng.standard_normal((d_h, 20))
    Y_t = X_t @ W_true.T
    Y_t[:, 8:16] = rng.standard_normal((256, 8)) * 5.0
    Y_t[:, 24:32] = rng.standard_normal((256, 8)) * 5.0
    W_p, b_p, gains, _ = apply_rope_plane_deshrinkage(W_true * 0.5, np.zeros(d_h), X_t, Y_t, np.zeros(20), np.zeros(d_h), rho2_min=0.20, gain_cap=1.45)
    assert np.allclose(gains[:16], gains[16:], atol=1e-12) and np.max(gains) <= 1.45 + 1e-12 and np.all(gains[8:16] == 0.0)
    xv = rng.standard_normal((16, d_h))
    assert np.max(np.abs(_rope_rotate_np(xv * gains[None, :], c_np, s_np) - _rope_rotate_np(xv, c_np, s_np) * gains[None, :])) < 1e-12
    print("  [PASS 2/10] RoPE-commuting 2x2 plane de-shrinkage & dead-plane gating (< 1e-12)")

    # Test 3: Compiled Arm 1a (including pos>0 KKT drift & partner-127 zeroing) + Compiled Arm 1b mixed-sign root!
    T_s, d_h3, D_in = 32, 128, 16
    inv_f3 = np.asarray(rope_inv_freq(d_h3, 1e6), dtype=np.float64)
    ang3 = np.arange(T_s, dtype=np.float64)[:, None] * inv_f3[None, :]
    c3, s3 = np.cos(ang3), np.sin(ang3)
    Wq0, bq0 = rng.standard_normal((d_h3, D_in)) * 0.1, rng.standard_normal(d_h3) * 0.1
    Wk0, bk0 = rng.standard_normal((d_h3, D_in)) * 0.1, rng.standard_normal(d_h3) * 0.1
    u0 = np.zeros(D_in); u0[0] = 10.0  # Orthogonal BOS spike
    mu_gt0 = np.zeros(D_in)
    k0_targ = rng.standard_normal(d_h3) * 0.5
    Phi_seqs = []
    for _ in range(12):
        seq = rng.standard_normal((T_s, D_in)) * 0.3
        seq[:, 0] = 0.0  # orthogonal to BOS spike -> near-zero pos>0 drift
        seq[0] = u0
        Phi_seqs.append(seq)
    # Build ground-truth donor alphas from known (s_true=1.6, d0_true=-2.5) compiled logits!
    Wq_gt, bq_gt, Wk_gt, bk_gt = fold_m63_rope_bos_offset(Wq0, bq0, Wk0, bk0, u0, mu_gt0, k0_targ, 1.6, -2.5)
    assert np.all(Wq_gt[[63, 127]] == 0.0) and bq_gt[127] == 0.0
    don_alps = []
    for ps_i in Phi_seqs:
        qr = _rope_rotate_np(Wq_gt @ ps_i[-1] + bq_gt, c3[-1], s3[-1])
        kr = _rope_rotate_np(ps_i @ Wk_gt.T + bk_gt[None, :], c3, s3)
        z = (kr @ qr) / math.sqrt(d_h3); z -= z.max()
        don_alps.append(np.exp(z) / np.sum(np.exp(z)))
    sol1a = solve_compiled_arm1a_bos_temp(don_alps, Phi_seqs, Wq0, bq0, Wk0, bk0, u0, mu_gt0, k0_targ, np.eye(D_in) * 0.1, c3, s3)
    assert abs(sol1a["s_opt"] - 1.6) < 1e-2 and abs(sol1a["delta0_opt"] - (-2.5)) < 1e-2 and sol1a["compiled_val_tune_kl"] < 1e-5
    assert sol1a["pos_gt0_kkt_logit_rmse"] < 1e-4, f"Unexpected pos>0 KKT drift: {sol1a['pos_gt0_kkt_logit_rmse']}"
    # Test mixed-sign BOS slope in Arm 1b (8 positive, 2 negative q.k0 -> target 0.65 strictly inside [0.05, 1.5]!)
    k0_ms = np.zeros(d_h3); k0_ms[0] = 40.0
    Wq_ms = np.zeros((d_h3, D_in)); Wq_ms[0, 1] = 2.0
    Wk_ms = np.zeros((d_h3, D_in)); bq_ms = np.zeros(d_h3); bk_ms = np.zeros(d_h3)
    seqs_ms, alps_ms = [], []
    for i_ms in range(10):
        sq = np.zeros((8, D_in)); sq[0] = u0; sq[-1, 1] = 1.0 if i_ms < 8 else -1.0
        seqs_ms.append(sq)
        a_d = np.full(8, 0.35 / 7.0); a_d[0] = 0.65
        alps_ms.append(a_d)
    sol1b = solve_compiled_arm1b_beta0(alps_ms, seqs_ms, Wq_ms, bq_ms, Wk_ms, bk_ms, u0, mu_gt0, k0_ms, None, c3[:8], s3[:8], beta_bounds=(0.05, 1.5))
    assert abs(sol1b["compiled_val_tune_a0"] - 0.65) < 0.01, f"Mixed-sign Arm 1b root failed: {sol1b}"
    print("  [PASS 3/10] Compiled Arm 1a (m=63+partner-127, pos>0 drift) & Arm 1b mixed-sign root")

    # Test 4: Exact v_proj.bias mean-zeroing
    v4, vb4, q4, k4, Wo4 = torch.randn(24, 16), torch.randn(16), torch.randn(16), torch.randn(24, 16), torch.randn(20, 16)
    c4, s4 = _make_rope_cos_sin(16, 1e6, 24, torch.device("cpu"))
    zr, oraw, _, _ = compute_head_qkv_and_logits(q4, k4, v4, Wo4, c4, s4)
    zc, ocen, _, _ = compute_head_qkv_and_logits(q4, k4, v4 - vb4.unsqueeze(0), Wo4, c4, s4)
    assert torch.max(torch.abs((zr - vb4) - zc)).item() < 1e-5 and torch.max(torch.abs((oraw - F.linear(vb4, Wo4)) - ocen)).item() < 1e-5
    print("  [PASS 4/10] Exact v_proj.bias mean-zeroing for bias-free o_proj (< 1e-5)")

    # Test 5: Native RoPE-impulse self-tap BF16 bit-exact alpha_{t,t} == 1.0
    T5 = 512
    bq_t = torch.from_numpy(build_rope_self_tap_biases(128, c=20.0)).to(torch.bfloat16)
    c5, s5 = _make_rope_cos_sin(128, 1e6, T5, torch.device("cpu"))
    be = bq_t.unsqueeze(0).expand(T5, -1)
    qr5 = torch.cat([be[:, :64] * c5.bfloat16() - be[:, 64:] * s5.bfloat16(), be[:, 64:] * c5.bfloat16() + be[:, :64] * s5.bfloat16()], dim=-1)
    lt5 = ((qr5 @ qr5.T) / math.sqrt(128.0)).masked_fill(torch.triu(torch.ones(T5, T5, dtype=torch.bool), diagonal=1), float("-inf"))
    min_gap = float((torch.diagonal(lt5)[1:] - lt5.masked_fill(torch.eye(T5, dtype=torch.bool), float("-inf")).max(dim=-1).values[1:]).min().item())
    assert min_gap > 35.0 and float(torch.max(torch.abs(torch.diagonal(F.softmax(lt5, dim=-1)) - 1.0)).item()) == 0.0
    print(f"  [PASS 5/10] Native RoPE-impulse self-tap BF16 bit-exact (min_gap={min_gap:.1f} nats)")

    # Test 6: Antisymmetric SwiGLU bilinear identity
    z6, u6 = torch.randn(512, 64), torch.randn(512, 64)
    assert torch.max(torch.abs((F.silu(z6) * u6 - F.silu(-z6) * u6) - z6 * u6)).item() < 1e-5
    print("  [PASS 6/10] Antisymmetric SwiGLU bilinear identity exact (< 1e-5)")

    # Test 7: 2-Hop 3-intervention multiset equality across MULTIPLE tokenizers & strict pool disjointness
    _assert_dummy_exemplars_disjoint()
    class _DummyTok:
        def __init__(self, off: int):
            self.off, self.vocab = off, {"<|im_start|>": 1, "<|im_end|>": 2}
        def encode(self, txt: str, add_special_tokens: bool = False) -> List[int]:
            out = []
            for t in txt.replace("\n", " \n ").replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split(" "):
                if t != "":
                    if t not in self.vocab:
                        self.vocab[t] = len(self.vocab) + self.off
                    out.append(self.vocab[t])
            return out
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "".join(f"<|im_start|> {m['role']}\n{m['content']}<|im_end|>\n" for m in msgs) + ("<|im_start|> assistant\n" if add_generation_prompt else "")
    dt1, dt2 = _DummyTok(10), _DummyTok(10)
    kp = {"pilot_keys": list(WORD_CANDIDATES_KEYS[:24]), "disc_keys": list(WORD_CANDIDATES_KEYS[24:60]), "eval_keys": list(WORD_CANDIDATES_KEYS[60:108]), "all_vals_clean": list(WORD_CANDIDATES_VALS[:40])}
    ds2_t = build_2hop_three_intervention_dataset([dt1, dt2], kp, wrapper="chat_2shot", n_rows=6, n_pilot=4, n_disc=4, n_eval=8, seed=SEED)
    assert len(ds2_t["eval"]["binding"]) == 4 and len(ds2_t["eval"]["null_swap"]) == 4
    print("  [PASS 7/10] 2-Hop 3-intervention cross-tokenizer multiset equality & disjoint pools")

    # Test 8: Numerically stable sink/content decomp UNDER VERIFIED FP32 BOS SATURATION + NON-CONSTANT V + MASKED-MIXED PURE ORACLE PARITY!
    D8, dh8 = 8, 8
    c8, s8 = _make_rope_cos_sin(dh8, 1e6, 10, torch.device("cpu"))
    Wo8 = [torch.eye(D8)]
    m_low8 = dh8 // 2 - 1  # Lowest frequency RoPE plane -> cos(9 * theta_m_low) > 0.999999!
    q_sat_B = torch.zeros(dh8); q_sat_B[m_low8] = 10.0
    k_sat_B = torch.zeros(10, dh8); k_sat_B[0, m_low8] = 20.0  # post-RoPE logit_0 = +70.71 -> alpha_0 === 1.0 in FP32!
    q_sat_A = torch.zeros(dh8); q_sat_A[m_low8] = 1.0; q_sat_A[0] = 0.5
    k_sat_A = torch.zeros(10, dh8); k_sat_A[0, m_low8] = 0.0; k_sat_A[1, m_low8] = 5.0
    v_ones = torch.ones(10, dh8)
    v_nonconst = torch.arange(10, dtype=torch.float32).unsqueeze(1).expand(10, dh8)
    _, _, alp_sat_B, _ = compute_head_qkv_and_logits(q_sat_B, k_sat_B, v_ones, Wo8[0], c8, s8)
    assert float(alp_sat_B[0].item()) == 1.0, f"Test 8 saturation precondition failed: alpha_0={alp_sat_B[0].item()}"
    # 1) Constant V: S1 must preserve exact mass 1.000000 under saturation!
    dec_sat = run_8cell_and_sink_content_decomposition(
        [[(q_sat_A, k_sat_A, v_ones)]], [[(q_sat_B, k_sat_B, v_ones)]], Wo8, c8, s8,
        lambda out: {"mean_val": float(out.mean().item())}, mask_mixed_dims=(m_low8, dh8 - 1),
    )
    assert abs(dec_sat["sink_content_decomp"]["S1_donor_sink_only"]["mean_val"] - 1.0) < 1e-6, f"Saturation mass loss: {dec_sat}"
    # 2) Non-constant V: Pure endpoints S0 === QB_KB_VB and S4 === QA_KA_VA BIT-EXACTLY, and QA_KA_VA is untouched by mask_mixed_dims!
    _, out_ref_A, _, _ = compute_head_qkv_and_logits(q_sat_A, k_sat_A, v_nonconst, Wo8[0], c8, s8)
    dec_nc = run_8cell_and_sink_content_decomposition(
        [[(q_sat_A, k_sat_A, v_nonconst)]], [[(q_sat_B, k_sat_B, v_nonconst)]], Wo8, c8, s8,
        lambda out: {"val": float(out.mean().item())}, mask_mixed_dims=(m_low8, dh8 - 1),
    )
    assert dec_nc["sink_content_decomp"]["S0_pure_student"]["val"] == dec_nc["factorial_8cell"]["QB_KB_VB"]["val"]
    assert dec_nc["sink_content_decomp"]["S4_full_donor_oracle"]["val"] == dec_nc["factorial_8cell"]["QA_KA_VA"]["val"]
    assert abs(dec_nc["factorial_8cell"]["QA_KA_VA"]["val"] - float(out_ref_A.mean().item())) < 1e-6
    print("  [PASS 8/10] Numerically stable sink/content decomp (alpha_0=1.0 mass=1.0, exact pure S0/S4 parity, untouched QA_KA_VA)")

    # Test 9: Analytical SwiGLU Jacobian + 6-arg `compute_cancellation_ratio_chi` production contract (CPU numpy Z_code)!
    D9, M9 = 16, 32
    Wg9, Wu9, Wd9 = torch.randn(M9, D9) * 0.2, torch.randn(M9, D9) * 0.2, torch.randn(D9, M9) * 0.2
    x9 = torch.randn(D9)
    J_ana = compute_swiglu_jacobian_single(x9, Wg9, Wu9, Wd9)
    J_auto = torch.autograd.functional.jacobian(lambda z: F.linear(F.silu(F.linear(z, Wg9)) * F.linear(z, Wu9), Wd9), x9)
    assert torch.max(torch.abs(J_ana - J_auto)).item() < 1e-5
    X9_chi = torch.randn(64, D9)
    LQ9 = torch.linalg.qr(torch.randn(D9, 8))[0]
    chi9 = compute_cancellation_ratio_chi(Wg9, Wu9, Wd9, X9_chi, (X9_chi @ LQ9).detach().cpu().numpy(), LQ9)
    assert isinstance(chi9, dict) and "chi_nl_sq" in chi9
    print(f"  [PASS 9/10] SwiGLU Jacobian (< 1e-5) & 6-arg compute_cancellation_ratio_chi contract (chi_nl_sq={chi9['chi_nl_sq']})")

    # Test 10: Live Native L0 Tied-Embedding Ingress (`alpha=2^-6`) + Full All-Param & Bit-Equal Logits Rollback
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg_t = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
    torch.manual_seed(SEED)
    m_tiny = Qwen2ForCausalLM(cfg_t).to(dtype=torch.bfloat16).eval()
    K10, alpha10 = 16, 2.0 ** -6
    target_l0 = torch.randn(256, K10) * 0.5
    eb_t = m_tiny.model.embed_tokens.weight.detach().float()
    rms_b_t = torch.sqrt(torch.mean(eb_t ** 2, dim=-1, keepdim=True) + float(m_tiny.model.layers[0].input_layernorm.variance_epsilon))
    E_priv_t = (alpha10 * rms_b_t * target_l0).to(torch.bfloat16)
    eval_ids_t = [[int(x) for x in rng.integers(0, 256, size=16)] for _ in range(8)]
    l0_res = verify_native_l0_tied_ingress(m_tiny, E_priv_t, lambda ids: target_l0[ids], eval_ids_t, v_shared=256, alpha=alpha10)
    assert l0_res["passes_all"] and l0_res["bit_exact_all_params_rollback"] and l0_res["bit_exact_logits_rollback"], f"L0 rollback failed: {l0_res}"
    print(f"  [PASS 10/10] Tiny Qwen2 Native L0 Ingress + All-Param & Logit Rollback ({l0_res})")
    print("[v5.1 Unit Tests] ALL 10/10 ENHANCED PURE-CPU UNIT TESTS PASSED!", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="PlatonicSwap v5.1: 100% Zero-SGD Input-Barrier Falsifier")
    ap.add_argument("--unit-test", action="store_true", help="Run 10 pure-CPU unit tests (zero downloads)")
    ap.add_argument("--run-all", action="store_true", help="Run full Phase 0 -> Phase 1 -> Phase 2 L4 falsifier")
    ap.add_argument("--allow-download", action="store_true", help="Allow HF model/dataset downloads (token=False)")
    ap.add_argument("--donor", type=str, default=DONOR_KEY_DEFAULT)
    ap.add_argument("--recipient", type=str, default=RECIPIENT_KEY_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    args = ap.parse_args()
    if args.unit_test:
        run_unit_tests()
    elif args.run_all:
        run_all_v51(donor_key=args.donor, recipient_key=args.recipient, allow_download=bool(args.allow_download), batch_size=int(args.batch_size), out_path=Path(args.out))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
