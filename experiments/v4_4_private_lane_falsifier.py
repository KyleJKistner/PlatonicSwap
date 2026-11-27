#!/usr/bin/env python3
"""
v4_4_private_lane_falsifier.py — 5-Minute Go/No-Go Private-Lane Falsifier (`L18 -> L20` on 2-Hop, v4.4.1)
=========================================================================================================

Implements the exact Fable 5.2 Max + Astra 6.2 Ultra + Astra Pro narrow-interface Platonic Organ falsifier
(`Qwen2.5-7B-Instruct -> Qwen2.5-1.5B-Instruct`, pure native in-weights `Qwen2ForCausalLM`, ZERO sidecars,
ZERO hooks during Metric 5 inertness/locality/generation, bit-exact `torch.equal == True` rollback):

  1. Position-0 Exclusion + Exact Per-Direction-Normalized Ridge + In-Loop KKT Pos-0 Sink Constraint:
     - Excludes token position 0 (BOS / attention sink) from ALL bridge covariances & Ridge fits.
     - Applies exact KKT rank-1 equality `W @ u_x0 = u_y0` INSIDE every `lambda` candidate BEFORE scoring
       held-out per-direction-normalized MSE & `R^2` (using per-token averaged pos-0 normalized complements!).
  2. Free RMSNorm Gauge Trick (`alpha = 0.1`, `D = 1536 -> D' = 1664 = 1536 + 128`):
     - `gamma' = sqrt(D / D') * [gamma_B ; (kappa_l / alpha) * 1_K]`, `eps' = (D / D') * eps`
       at highway layers `L18, L19, L20`, using the exact paired token estimator `kappa_l = mean(r_B / r_A)`.
     - Tied-embedding zero-bloat rule: `embed_tokens.weight` & `lm_head.weight` stay strictly TIED (`same data_ptr`)
       with private columns `[D:D']` identically ZERO.
     - At `L20` (no attention organ installed), `u_post` complement uses `h_in_A[20] / rms(h_mid_A[20])` so L20's
       private-lane read aligns identically with the carried incoming private lanes.
  3. Primal Interface Basis `V in R^{3584 x 128}`:
     - Top `K = 128` primal generalized-eigenvector subspace `C_bg_reg^{+1/2} Q_top` of `(centered fit contrast
       covariance pooled over L18..L20, centered pos>0 generic background covariance)`, QR-orthonormalized (`V^T V = I_K`).
  4. 4 Mandatory Arms (all on the exact same `400` strictly unseen-key 2-Hop Policy L/R contrast pairs):
     - `REAL_COPIED`: L18–L19 donor units (`+1 KV / +6 Q`, `+256 SwiGLU`), strict copied lane writes `[0 ; alpha V^T W_out,A]`.
     - `REAL_REFIT` (PREDECLARED PRIMARY ARM): L18–L19 donor units, scale-normalized background-regularized ridge
       lane writes tuned on an inner 320/80 split of the 400 fit pairs.
     - `SHUF`: wrong-layer donor units (`L6 -> L18`, `L7 -> L19`), same counts & Frobenius-norm-matched,
       identical pullback & inner-tuned ridge lane write solver.
     - `NONE`: widened 1.5B with NO organ writes at L18–L19 + freshly fitted task-local reader at L20.
  5. 5 Predeclared Pass / Kill Metrics (scored on unseen-key contrast INCREMENTS `delta_Delta = delta(L20_in - L18_in)`):
     - (1) Interface Capture `c` (generic-whitened on `delta_Delta`, donor-only): Kill < 0.60 / Go >= 0.80.
     - (2) Organ Compute Budget `beta` (donor-only on `delta_Delta`): Kill < 0.50 / Go >= 0.70.
     - (3) Relay Survival `eta` (recipient private lanes on `delta_Delta`): Kill REAL < 0.25 or REAL - SHUF < 0.15;
           Go REAL >= 0.50 and >= 0.7 * beta and SHUF <= 0.10.
     - (4) L20 Hearing: top-64 L20 donor MLP gate/up median cosine + **native BF16 SwiGLU product** `R^2_Delta`
           (paired key-group bootstrap 95% LB > 0 over `NONE_local_reader`, `NONE_pulled`, AND `SHUF`!)
           + pos>0 shared-suffix RoPE attention-logit RMSE. Kill cos < 0.50 / Go cos >= 0.80 & RMSE <= 1.0 nat.
     - (5) Inertness / Locality (100% ZERO HOOKS via hook-free decoder prefix for `rho`!):
           generic `rho <= 1e-3`, `KL(B || B') <= 0.01` nat/tok, 380-fact locality retention >= 97.0% of base-correct
           (base 1.5B = 313/380; raw `n_correct / 380` also logged) + `torch.equal == True` bit-exact rollback.
           Isolated Metric 5 failure when M1–M4 pass emits `INERTNESS_FAIL` (never silently `GO`!).

Usage:
  PYTHONPATH=platonicswap:experiments python experiments/v4_4_private_lane_falsifier.py --unit-test
  PYTHONPATH=platonicswap:experiments python -u experiments/v4_4_private_lane_falsifier.py --run-all
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla
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
OUT_PATH = RESULTS_DIR / "results_v4_4_private_lane_falsifier.json"

from exact_math_v43 import build_60k_calibration_corpus
from extract_and_baseline import HF_CACHE_DIR, get_decoder_layers
from large_to_small import load_model_bf16
from native_weight_translation import (
    assert_zero_runtime_hooks,
    bf16_to_f64_np,
    get_config_rope_theta,
    unload_model_vram_safe,
)
from v4_stage1_qwen7b_to_1p5b_native import (
    DONOR_KEY_DEFAULT,
    RECIPIENT_KEY_DEFAULT,
    SEED,
    build_2row_2hop_dataset,
    eval_certified_380_locality,
)
import v2_track1_dev_dataset as t1d


# ======================================================================================
# 1. Position-0 Masking, Streamed Moments & In-Loop KKT Pos-0 Constrained DirNorm Ridge
# ======================================================================================

def get_pos0_and_gt0_masks(attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split `attention_mask` (works identically for left-padding AND right-padding!) into:
      - `pos0_mask`: True ONLY at the first non-pad token of each sequence (BOS / attention sink)
      - `pos_gt0_mask`: True at all valid non-pad tokens STRICTLY AFTER position 0.
    """
    valid = attention_mask.bool()
    bsz = valid.shape[0]
    pos0 = torch.zeros_like(valid)
    for b in range(bsz):
        nz = torch.nonzero(valid[b], as_tuple=False)
        if nz.numel() > 0:
            pos0[b, int(nz[0, 0].item())] = True
    pos_gt0 = valid & (~pos0)
    return pos0, pos_gt0


def _stream_second_moments_u16(
    X_u16: np.ndarray,
    Y_u16: np.ndarray,
    x_cols: Optional[int] = None,
    chunk: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FP64 `(S_xx, S_yx)` in 2048-row chunks directly from lossless BF16 `uint16` arrays (<60 MB temp RAM!)."""
    n = min(len(X_u16), len(Y_u16))
    dx = int(x_cols) if x_cols is not None else int(X_u16.shape[1])
    dy = int(Y_u16.shape[1])
    S_xx = np.zeros((dx, dx), dtype=np.float64)
    S_yx = np.zeros((dy, dx), dtype=np.float64)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        xb = bf16_to_f64_np(X_u16[i0:i1])[:, :dx]
        yb = bf16_to_f64_np(Y_u16[i0:i1])
        S_xx += xb.T @ xb
        S_yx += yb.T @ xb
    inv_n = 1.0 / max(float(n), 1.0)
    S_xx *= inv_n
    S_yx *= inv_n
    S_xx = 0.5 * (S_xx + S_xx.T)
    return S_xx, S_yx


def fit_dirnorm_ridge_with_pos0_sink(
    S_xx: np.ndarray,
    S_yx: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    u_x0: Optional[np.ndarray] = None,
    u_y0: Optional[np.ndarray] = None,
    lam_grid: Sequence[float] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fit `W: R^{D_in} -> R^{D_out}` (`Y ≈ X @ W.T`) in FP64 strictly from `pos > 0` moments:
      1. Ridge `lam` scaled relative to `median_pos_eig(S_xx)` (immune to outlier spikes!).
      2. Enforces exact KKT rank-1 position-0 sink constraint `W @ u_x0 = u_y0` INSIDE EACH candidate `lam`
         BEFORE evaluating held-out per-direction-normalized MSE and `R^2` (so validation ALWAYS scores
         the exact constrained map that is returned!).
    """
    Xv = bf16_to_f64_np(X_val)
    Yv = bf16_to_f64_np(Y_val)
    D_in = S_xx.shape[0]

    eigvals, Q = sla.eigh(S_xx)
    eigvals = np.maximum(eigvals, 0.0)
    pos = eigvals[eigvals > 1e-12]
    med_eig = float(np.median(pos)) if len(pos) > 0 else float(np.trace(S_xx) / max(D_in, 1))
    p_spec = pos / (np.sum(pos) + 1e-12)
    erank = float(np.exp(-np.sum(p_spec * np.log(p_spec + 1e-12))))
    Qt_Sxy = Q.T @ S_yx.T  # (D_in, D_out)

    dir_var = np.var(Yv, axis=0)
    active_d = dir_var > 1e-12
    safe_var = np.where(active_d, dir_var, 1.0)
    yv_var_tot = float(np.sum((Yv - Yv.mean(axis=0)) ** 2)) + 1e-12

    use_sink = (u_x0 is not None and u_y0 is not None and float(np.linalg.norm(u_x0)) > 1e-12)
    ux = np.asarray(u_x0, dtype=np.float64).reshape(-1) if use_sink else None
    uy = np.asarray(u_y0, dtype=np.float64).reshape(-1) if use_sink else None
    Qt_ux = (Q.T @ ux) if ux is not None else None

    best_W: Optional[np.ndarray] = None
    best_loss = 1e18
    best_r2 = -1e9
    best_lam_rel = float(lam_grid[0])
    best_lam_abs = max(float(lam_grid[0]) * med_eig, 1e-12)
    best_sink_err: Optional[float] = None

    for lam_rel in lam_grid:
        lam_abs = max(float(lam_rel) * med_eig, 1e-12)
        W_t = Q @ (Qt_Sxy / (eigvals[:, None] + lam_abs))  # (D_in, D_out)
        sink_err = None
        if ux is not None and uy is not None and Qt_ux is not None:
            s_inv_u = Q @ (Qt_ux / (eigvals + lam_abs))
            denom = float(ux @ s_inv_u)
            if denom > 1e-18:
                resid = uy - (ux @ W_t)
                W_t = W_t + np.outer(s_inv_u, resid) / denom
                sink_err = float(np.linalg.norm((ux @ W_t) - uy) / (np.linalg.norm(uy) + 1e-12))

        pred = Xv @ W_t
        sq_err = (Yv - pred) ** 2
        per_dir_nmse = np.mean(sq_err, axis=0) / safe_var
        loss_dn = float(np.mean(per_dir_nmse[active_d])) if np.any(active_d) else float(np.mean(sq_err))
        if loss_dn < best_loss:
            best_loss = loss_dn
            best_r2 = 1.0 - float(np.sum(sq_err) / yv_var_tot)
            best_lam_rel = float(lam_rel)
            best_lam_abs = lam_abs
            best_sink_err = sink_err
            best_W = W_t.T  # (D_out, D_in)

    assert best_W is not None
    return best_W, {
        "median_eig": float(med_eig),
        "best_lam_rel": best_lam_rel,
        "best_lam_abs": float(best_lam_abs),
        "val_dirnorm_nmse": round(best_loss, 4),
        "val_r2_pos_gt0": round(best_r2, 4),
        "erank_pos_gt0": round(erank, 2),
        "pos0_sink_rel_err": round(best_sink_err, 8) if best_sink_err is not None else None,
    }


# ======================================================================================
# 2. Primal Interface Basis `V in R^{D_A x K}` & Generic-Whitened Capture `c`
# ======================================================================================

def fit_generalized_interface_basis(
    contrast_deltas_A: np.ndarray,
    bg_states_A_u16_list: Sequence[np.ndarray],
    K: int = 128,
    bg_reg_rel_med: float = 1e-2,
) -> Dict[str, Any]:
    """
    Compute top-`K` primal generalized-eigenvector signal subspace of `(C_contrast, C_bg_reg)` in FP64:
      - Streams `C_bg` directly from BF16 `uint16` arrays in 2048-row chunks (low host RAM!).
      - Computes `V_orth = qr(C_bg_reg^{+1/2} Q_top).Q` (`V^T V = I_K`) so unwhitened signal `d = C_bg^{+1/2} d_w`
        projects losslessly onto `span(V)`!
    """
    dc = np.asarray(contrast_deltas_A, dtype=np.float64)
    dc = dc - dc.mean(axis=0, keepdims=True)
    C_con = (dc.T @ dc) / max(float(len(dc)), 1.0)
    C_con = 0.5 * (C_con + C_con.T)
    D_A = C_con.shape[0]

    # Streamed centered C_bg from uint16/float arrays
    sum_bg = np.zeros(D_A, dtype=np.float64)
    sum_bg2 = np.zeros((D_A, D_A), dtype=np.float64)
    n_bg = 0
    for arr in bg_states_A_u16_list:
        for i0 in range(0, len(arr), 2048):
            blk = bf16_to_f64_np(arr[i0 : i0 + 2048])
            sum_bg += blk.sum(axis=0)
            sum_bg2 += blk.T @ blk
            n_bg += int(blk.shape[0])
    inv_nb = 1.0 / max(float(n_bg), 1.0)
    mean_bg = sum_bg * inv_nb
    C_bg = sum_bg2 * inv_nb - np.outer(mean_bg, mean_bg)
    C_bg = 0.5 * (C_bg + C_bg.T)

    evals_bg, Q_bg = sla.eigh(C_bg)
    evals_bg = np.maximum(evals_bg, 0.0)
    pos_bg = evals_bg[evals_bg > 1e-12]
    med_bg = float(np.median(pos_bg)) if len(pos_bg) > 0 else float(np.trace(C_bg) / max(D_A, 1))
    reg_abs = max(float(bg_reg_rel_med) * med_bg, 1e-10)
    inv_sqrt = 1.0 / np.sqrt(evals_bg + reg_abs)
    sqrt_bg = np.sqrt(evals_bg + reg_abs)

    A_tmp = (Q_bg.T @ C_con) * inv_sqrt[:, None]
    M_w = (A_tmp @ Q_bg) * inv_sqrt[None, :]
    M_w = 0.5 * (M_w + M_w.T)
    geigs, Q_w = sla.eigh(M_w)
    order = np.argsort(geigs)[::-1]
    k_eff = min(int(K), D_A)
    top_geigs = geigs[order[:k_eff]]
    Q_top = Q_w[:, order[:k_eff]]

    U_sig = Q_bg @ (sqrt_bg[:, None] * Q_top)
    V_orth, _ = np.linalg.qr(U_sig, mode="reduced")

    return {
        "V": V_orth,
        "Q_bg": Q_bg,
        "inv_sqrt_bg": inv_sqrt,
        "bg_reg_abs": float(reg_abs),
        "bg_med_eig": float(med_bg),
        "top_geigs": top_geigs[:10].tolist(),
        "geig_captured_frac_fit": round(float(np.sum(np.maximum(top_geigs, 0.0)) / max(float(np.sum(np.maximum(geigs, 0.0))), 1e-12)), 4),
    }


def compute_whitened_interface_capture(
    test_deltas_A: np.ndarray,
    V: np.ndarray,
    Q_bg: np.ndarray,
    inv_sqrt_bg: np.ndarray,
) -> float:
    """Metric 1 (Donor-only): exact fraction of generic-whitened contrast energy lying in `span(V)`!"""
    d = np.asarray(test_deltas_A, dtype=np.float64)
    d_w = (d @ Q_bg) * inv_sqrt_bg[None, :]
    V_w = (Q_bg.T @ V) * inv_sqrt_bg[:, None]
    B_w, _ = np.linalg.qr(V_w, mode="reduced")
    proj_e = float(np.sum((d_w @ B_w) ** 2))
    tot_e = float(np.sum(d_w ** 2)) + 1e-12
    return float(proj_e / tot_e)


# ======================================================================================
# 3. Native Reversible Private-Lane Widening (`D -> D+K`, Free RMSNorm Gauge, Tied Emb!)
# ======================================================================================

def _pad_linear_in_cols(orig: nn.Linear, add_in: int) -> nn.Linear:
    dev, dt = orig.weight.device, orig.weight.dtype
    has_b = orig.bias is not None
    out_f, in_f = orig.out_features, orig.in_features
    lin = nn.Linear(in_f + add_in, out_f, bias=has_b, device=dev, dtype=dt)
    with torch.no_grad():
        lin.weight[:, :in_f].copy_(orig.weight)
        lin.weight[:, in_f:].zero_()
        if has_b:
            lin.bias.copy_(orig.bias)
    for p in lin.parameters():
        p.requires_grad_(False)
    return lin


def _pad_linear_out_rows(orig: nn.Linear, add_out: int) -> nn.Linear:
    dev, dt = orig.weight.device, orig.weight.dtype
    has_b = orig.bias is not None
    out_f, in_f = orig.out_features, orig.in_features
    lin = nn.Linear(in_f, out_f + add_out, bias=has_b, device=dev, dtype=dt)
    with torch.no_grad():
        lin.weight[:out_f, :].copy_(orig.weight)
        lin.weight[out_f:, :].zero_()
        if has_b:
            lin.bias[:out_f].copy_(orig.bias)
            lin.bias[out_f:].zero_()
    for p in lin.parameters():
        p.requires_grad_(False)
    return lin


def _widen_reader_linear(orig: nn.Linear, K: int, append_w: torch.Tensor, append_b: Optional[torch.Tensor] = None) -> nn.Linear:
    dev, dt = orig.weight.device, orig.weight.dtype
    has_b = orig.bias is not None
    out_f, in_f = orig.out_features, orig.in_features
    add_out = int(append_w.shape[0])
    assert append_w.shape[1] == in_f + K
    lin = nn.Linear(in_f + K, out_f + add_out, bias=has_b, device=dev, dtype=dt)
    with torch.no_grad():
        lin.weight[:out_f, :in_f].copy_(orig.weight)
        lin.weight[:out_f, in_f:].zero_()
        lin.weight[out_f:, :].copy_(append_w.to(device=dev, dtype=dt))
        if has_b:
            lin.bias[:out_f].copy_(orig.bias)
            if append_b is not None:
                lin.bias[out_f:].copy_(append_b.to(device=dev, dtype=dt))
            else:
                lin.bias[out_f:].zero_()
    for p in lin.parameters():
        p.requires_grad_(False)
    return lin


def _widen_writer_linear(orig: nn.Linear, K: int, append_w: torch.Tensor) -> nn.Linear:
    dev, dt = orig.weight.device, orig.weight.dtype
    has_b = orig.bias is not None
    out_f, in_f = orig.out_features, orig.in_features
    add_in = int(append_w.shape[1])
    assert append_w.shape[0] == out_f + K
    lin = nn.Linear(in_f + add_in, out_f + K, bias=has_b, device=dev, dtype=dt)
    with torch.no_grad():
        lin.weight[:out_f, :in_f].copy_(orig.weight)
        lin.weight[out_f:, :in_f].zero_()
        lin.weight[:, in_f:].copy_(append_w.to(device=dev, dtype=dt))
        if has_b:
            lin.bias[:out_f].copy_(orig.bias)
            lin.bias[out_f:].zero_()
    for p in lin.parameters():
        p.requires_grad_(False)
    return lin


class ReversiblePrivateLanePack:
    """
    Pure native reversible widening of stock `Qwen2ForCausalLM` from `D` -> `D' = D + K`:
      - Preserves `tie_word_embeddings=True` (zero bloat: `embed_tokens` & `lm_head` share one `nn.Parameter`
        with strictly ZERO columns `[D:D']`!).
      - Exact Free RMSNorm Gauge Trick (`alpha = 0.1`):
          `gamma' = sqrt(D / D') * [gamma_B ; (kappa_l / alpha) * 1_K]`, `eps' = (D / D') * eps`
      - Atomic `BaseException` unwind in `apply()` + bit-exact `torch.equal == True` rollback in `revert()`.
    """

    def __init__(
        self,
        K: int,
        alpha: float,
        highway_kappas: Dict[int, Dict[str, float]],
        attn_organs: Optional[Dict[int, Dict[str, Any]]] = None,
        mlp_organs: Optional[Dict[int, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.K = int(K)
        self.alpha = float(alpha)
        self.highway_kappas = {int(k): dict(v) for k, v in highway_kappas.items()}
        self.attn_organs = {int(k): dict(v) for k, v in (attn_organs or {}).items()}
        self.mlp_organs = {int(k): dict(v) for k, v in (mlp_organs or {}).items()}
        self.metadata = dict(metadata or {})
        self._applied: bool = False
        self._orig_state: Dict[str, Any] = {}

    def _restore_partial(self, model: nn.Module) -> None:
        st = self._orig_state
        if "embed_weight" in st:
            model.model.embed_tokens.weight = st["embed_weight"]
            model.model.embed_tokens.embedding_dim = int(st["embed_weight"].shape[1])
        if "lm_head_weight" in st:
            model.lm_head.weight = st["lm_head_weight"]
            model.lm_head.in_features = int(st["lm_head_weight"].shape[1])
        if "final_norm" in st:
            w0, e0 = st["final_norm"]
            model.model.norm.weight = w0
            model.model.norm.variance_epsilon = e0
        layers = get_decoder_layers(model)
        for li, ldict in st.get("layers", {}).items():
            lyr = layers[li]
            if "input_layernorm" in ldict:
                w0, e0 = ldict["input_layernorm"]
                lyr.input_layernorm.weight = w0
                lyr.input_layernorm.variance_epsilon = e0
            if "post_attention_layernorm" in ldict:
                w0, e0 = ldict["post_attention_layernorm"]
                lyr.post_attention_layernorm.weight = w0
                lyr.post_attention_layernorm.variance_epsilon = e0
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if attr in ldict:
                    setattr(lyr.self_attn, attr, ldict[attr])
            for attr in ("gate_proj", "up_proj", "down_proj"):
                if attr in ldict:
                    setattr(lyr.mlp, attr, ldict[attr])
        self._orig_state.clear()
        self._applied = False

    def apply(self, model: nn.Module) -> None:
        assert not self._applied, "ReversiblePrivateLanePack already applied!"
        assert_zero_runtime_hooks(model)
        K = self.K
        D = int(model.model.embed_tokens.weight.shape[1])
        Dp = D + K
        scale_g = math.sqrt(float(D) / float(Dp))
        scale_e = float(D) / float(Dp)
        dev = model.model.embed_tokens.weight.device
        dt = model.model.embed_tokens.weight.dtype

        self._orig_state = {"layers": {}}
        try:
            orig_emb = model.model.embed_tokens.weight
            orig_lm = model.lm_head.weight
            was_tied = (orig_emb.data_ptr() == orig_lm.data_ptr())
            self._orig_state["embed_weight"] = orig_emb
            self._orig_state["lm_head_weight"] = orig_lm

            V = int(orig_emb.shape[0])
            new_emb_data = torch.zeros((V, Dp), device=dev, dtype=dt)
            new_emb_data[:, :D].copy_(orig_emb.detach())
            new_emb_param = nn.Parameter(new_emb_data, requires_grad=False)
            model.model.embed_tokens.weight = new_emb_param
            model.model.embed_tokens.embedding_dim = Dp
            if was_tied:
                model.lm_head.weight = new_emb_param
                assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
            else:
                new_lm_data = torch.zeros((int(orig_lm.shape[0]), Dp), device=dev, dtype=dt)
                new_lm_data[:, :D].copy_(orig_lm.detach())
                model.lm_head.weight = nn.Parameter(new_lm_data, requires_grad=False)
            model.lm_head.in_features = Dp

            fn = model.model.norm
            self._orig_state["final_norm"] = (fn.weight, fn.variance_epsilon)
            fn_w = torch.zeros((Dp,), device=dev, dtype=fn.weight.dtype)
            fn_w[:D].copy_(fn.weight.detach().float() * scale_g)
            fn.weight = nn.Parameter(fn_w, requires_grad=False)
            fn.variance_epsilon = float(fn.variance_epsilon) * scale_e

            layers = get_decoder_layers(model)
            for li, lyr in enumerate(layers):
                ld: Dict[str, Any] = {}
                self._orig_state["layers"][li] = ld

                for ln_name, kappa_key in (("input_layernorm", "in"), ("post_attention_layernorm", "post")):
                    ln = getattr(lyr, ln_name)
                    ld[ln_name] = (ln.weight, ln.variance_epsilon)
                    w_new = torch.zeros((Dp,), device=dev, dtype=ln.weight.dtype)
                    w_new[:D].copy_(ln.weight.detach().float() * scale_g)
                    if li in self.highway_kappas and kappa_key in self.highway_kappas[li]:
                        kappa = float(self.highway_kappas[li][kappa_key])
                        w_new[D:].fill_(scale_g * (kappa / self.alpha))
                    ln.weight = nn.Parameter(w_new, requires_grad=False)
                    ln.variance_epsilon = float(ln.variance_epsilon) * scale_e

                attn = lyr.self_attn
                for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    ld[attr] = getattr(attn, attr)
                if li in self.attn_organs:
                    sp = self.attn_organs[li]
                    g_orig = int(attn.num_key_value_groups)
                    assert int(sp["add_q_heads"]) == int(sp["add_kv_heads"]) * g_orig
                    attn.q_proj = _widen_reader_linear(ld["q_proj"], K, sp["q_w"], sp.get("q_b"))
                    attn.k_proj = _widen_reader_linear(ld["k_proj"], K, sp["k_w"], sp.get("k_b"))
                    attn.v_proj = _widen_reader_linear(ld["v_proj"], K, sp["v_w"], sp.get("v_b"))
                    attn.o_proj = _widen_writer_linear(ld["o_proj"], K, sp["o_w"])
                else:
                    attn.q_proj = _pad_linear_in_cols(ld["q_proj"], K)
                    attn.k_proj = _pad_linear_in_cols(ld["k_proj"], K)
                    attn.v_proj = _pad_linear_in_cols(ld["v_proj"], K)
                    attn.o_proj = _pad_linear_out_rows(ld["o_proj"], K)

                mlp = lyr.mlp
                for attr in ("gate_proj", "up_proj", "down_proj"):
                    ld[attr] = getattr(mlp, attr)
                if li in self.mlp_organs:
                    sp = self.mlp_organs[li]
                    mlp.gate_proj = _widen_reader_linear(ld["gate_proj"], K, sp["gate_w"])
                    mlp.up_proj = _widen_reader_linear(ld["up_proj"], K, sp["up_w"])
                    mlp.down_proj = _widen_writer_linear(ld["down_proj"], K, sp["down_w"])
                else:
                    mlp.gate_proj = _pad_linear_in_cols(ld["gate_proj"], K)
                    mlp.up_proj = _pad_linear_in_cols(ld["up_proj"], K)
                    mlp.down_proj = _pad_linear_out_rows(ld["down_proj"], K)

            self._applied = True
            assert_zero_runtime_hooks(model)
        except BaseException:
            self._restore_partial(model)
            raise

    def revert(self, model: nn.Module) -> None:
        if not self._applied and not self._orig_state:
            return
        self._restore_partial(model)
        assert_zero_runtime_hooks(model)

    def verify_bit_exact_rollback(
        self,
        model: nn.Module,
        probe_ids: torch.Tensor,
        probe_mask: torch.Tensor,
        base_logits: torch.Tensor,
        strict_assert: bool = True,
    ) -> Dict[str, Any]:
        assert not self._applied
        assert_zero_runtime_hooks(model)
        with torch.no_grad():
            cur = model(input_ids=probe_ids, attention_mask=probe_mask, use_cache=False).logits.detach().cpu()
        bit_eq = bool(torch.equal(cur, base_logits.cpu()))
        max_diff = float((cur.float() - base_logits.cpu().float()).abs().max().item())
        res = {"logits_bit_equal": bit_eq, "max_abs_diff": max_diff, "zero_runtime_hooks": True}
        if strict_assert:
            assert bit_eq and max_diff == 0.0, f"Private-lane rollback FAILED: {res}"
        return res


# ======================================================================================
# 4. 100% Zero-Hook Qwen2 Decoder Prefix Helper & State Extraction Helpers
# ======================================================================================

@torch.inference_mode()
def forward_qwen2_prefix_no_hooks(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    stop_layer: int,
) -> torch.Tensor:
    """
    100% ZERO-HOOK native Qwen2 decoder prefix returning the BIT-EXACT `hidden_states` entering `layers[stop_layer]`!
    Mirrors `Qwen2Model.forward` line-for-line (identical `create_causal_mask`, `position_ids`, and `rotary_emb`)
    with ZERO forward hooks registered anywhere in PyTorch!
    """
    assert_zero_runtime_hooks(model)
    base = model.model
    inputs_embeds = base.embed_tokens(input_ids)
    seq_len = int(inputs_embeds.shape[1])
    position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)

    import inspect
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    cm_params = inspect.signature(create_causal_mask).parameters
    mask_kwargs: Dict[str, Any] = {
        "config": base.config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    cache_position = torch.arange(seq_len, device=inputs_embeds.device)
    if "cache" in cm_params:
        mask_kwargs["cache"] = None
    elif "past_key_values" in cm_params:
        mask_kwargs["past_key_values"] = None
    if "cache_position" in cm_params:
        mask_kwargs["cache_position"] = cache_position
    causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
    if getattr(base, "has_sliding_layers", False):
        sw_params = inspect.signature(create_sliding_window_causal_mask).parameters
        sw_kwargs = {k: v for k, v in mask_kwargs.items() if k in sw_params}
        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**sw_kwargs)

    hidden_states = inputs_embeds
    position_embeddings = base.rotary_emb(hidden_states, position_ids)
    layer_types = getattr(base.config, "layer_types", ["full_attention"] * len(base.layers))
    lyr_params = inspect.signature(base.layers[0].forward).parameters
    for i in range(int(stop_layer)):
        lkw: Dict[str, Any] = {
            "attention_mask": causal_mask_mapping[layer_types[i]],
            "position_embeddings": position_embeddings,
            "position_ids": position_ids,
            "use_cache": False,
        }
        if "cache_position" in lyr_params:
            lkw["cache_position"] = cache_position
        out = base.layers[i](hidden_states, **lkw)
        hidden_states = out[0] if isinstance(out, tuple) else out
    assert_zero_runtime_hooks(model)
    return hidden_states.detach()


@torch.inference_mode()
def capture_generic_pos0_and_gt0(
    model: nn.Module,
    tok: Any,
    passages: Sequence[str],
    layer_indices: Sequence[int],
    device: torch.device,
    batch_size: int = 8,
    max_length: int = 160,
    max_gt0_tokens: int = 20_000,
    capture_pos0_tokens: bool = True,
) -> Dict[str, Any]:
    """
    Capture `h_in`, `u_in`, `h_mid`, `u_post`, per-token `pos == 0` arrays, and exact paired per-token
    `rms_in`/`rms_post` vectors on `pos > 0` tokens (truncated identically to `max_gt0_tokens`!).
    Uses decoder backbone `model.model(...)` (ZERO unused 152k lm_head projection!).
    """
    layers = get_decoder_layers(model)
    buf: Dict[int, Dict[str, torch.Tensor]] = {int(li): {} for li in layer_indices}
    fields = ("h_in", "u_in", "h_mid", "u_post")
    gt0_store: Dict[int, Dict[str, List[np.ndarray]]] = {int(li): {f: [] for f in fields} for li in layer_indices}
    p0_store: Dict[int, Dict[str, List[np.ndarray]]] = {int(li): {f: [] for f in fields} for li in layer_indices}
    rms_store: Dict[int, Dict[str, List[np.ndarray]]] = {int(li): {"in": [], "post": []} for li in layer_indices}
    n_p0 = 0
    n_gt0 = 0
    handles: List[Any] = []
    old_pad = tok.padding_side
    tok.padding_side = "left"
    backbone = model.model if hasattr(model, "model") else model
    eps_ln = float(getattr(model.config, "rms_norm_eps", 1e-6))
    try:
        for li in layer_indices:
            li = int(li)
            lyr = layers[li]
            def _mk_pre(idx: int, key: str):
                def _pre(_m, args):
                    buf[idx][key] = args[0].detach()
                return _pre
            def _mk_out(idx: int, key: str):
                def _fn(_m, _i, out):
                    buf[idx][key] = (out[0] if isinstance(out, tuple) else out).detach()
                return _fn
            handles.append(lyr.input_layernorm.register_forward_pre_hook(_mk_pre(li, "h_in")))
            handles.append(lyr.input_layernorm.register_forward_hook(_mk_out(li, "u_in")))
            handles.append(lyr.post_attention_layernorm.register_forward_pre_hook(_mk_pre(li, "h_mid")))
            handles.append(lyr.post_attention_layernorm.register_forward_hook(_mk_out(li, "u_post")))

        for i0 in range(0, len(passages), batch_size):
            bp = list(passages[i0 : i0 + batch_size])
            enc = tok(bp, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=True)
            ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            _ = backbone(input_ids=ids, attention_mask=mask, use_cache=False)

            p0_m, gt0_m = get_pos0_and_gt0_masks(mask)
            n_p0 += int(p0_m.sum().item())
            n_bt = int(gt0_m.sum().item())

            for li in layer_indices:
                li = int(li)
                for f in fields:
                    t = buf[li][f]
                    gt0_store[li][f].append(t[gt0_m].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
                    if capture_pos0_tokens:
                        p0_store[li][f].append(t[p0_m].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
                h_in_gt = buf[li]["h_in"][gt0_m].float()
                h_mid_gt = buf[li]["h_mid"][gt0_m].float()
                rms_store[li]["in"].append(torch.sqrt(h_in_gt.pow(2).mean(dim=-1) + eps_ln).cpu().numpy())
                rms_store[li]["post"].append(torch.sqrt(h_mid_gt.pow(2).mean(dim=-1) + eps_ln).cpu().numpy())
                buf[li].clear()

            n_gt0 += n_bt
            if n_gt0 >= max_gt0_tokens:
                break
    finally:
        for h in handles:
            h.remove()
        tok.padding_side = old_pad
    assert_zero_runtime_hooks(model)

    gt0_out: Dict[int, Dict[str, np.ndarray]] = {}
    p0_out: Dict[int, Dict[str, np.ndarray]] = {}
    rms_vec: Dict[int, Dict[str, np.ndarray]] = {}
    for li in layer_indices:
        li = int(li)
        gt0_out[li] = {
            f: np.ascontiguousarray(np.concatenate(gt0_store[li][f], axis=0)[:max_gt0_tokens].copy())
            for f in fields
        }
        p0_out[li] = {
            f: (np.ascontiguousarray(np.concatenate(p0_store[li][f], axis=0).copy()) if capture_pos0_tokens else np.empty((0, 0), dtype=np.uint16))
            for f in fields
        }
        rms_vec[li] = {
            "in": np.concatenate(rms_store[li]["in"], axis=0)[:max_gt0_tokens].copy(),
            "post": np.concatenate(rms_store[li]["post"], axis=0)[:max_gt0_tokens].copy(),
        }
    return {"gt0": gt0_out, "pos0": p0_out, "rms_vec": rms_vec, "n_gt0": min(n_gt0, max_gt0_tokens), "n_pos0": n_p0}


@torch.inference_mode()
def capture_donor_2hop_bundle(
    model_A: nn.Module,
    tok_A: Any,
    items: Sequence[Dict[str, Any]],
    dev_A: torch.device,
    org_layers: Tuple[int, int] = (18, 19),
    eval_layer: int = 20,
    batch_size: int = 8,
    max_length: int = 256,
) -> Dict[str, Any]:
    """
    Single Donor 7B pass over 2-Hop Policy L & R prompts:
      - Verifies 100% exact token-ID shared-suffix alignment (`f_ids[-len(s_ids):] == s_ids`).
      - Captures last-token `h_in` at `(18, 19, 20)`, `z_attn`/`y_attn`/`z_mlp`/`y_mlp` at `(18, 19)`,
        **native BF16 `z_mlp` (`down_proj` input) at `L20`** (for exact native SwiGLU product!),
        `u_post` at `L20`, and shared-suffix `h_in` at `L18` & `u_in` at `L20`.
    """
    l0, l1 = int(org_layers[0]), int(org_layers[1])
    le = int(eval_layer)
    layers_A = get_decoder_layers(model_A)
    backbone_A = model_A.model if hasattr(model_A, "model") else model_A

    suf_lens: List[int] = []
    for idx_i, it in enumerate(items):
        sp = it["recipient_prompt"]
        s_ids = tok_A.encode(sp, truncation=True, max_length=max_length, add_special_tokens=True)
        for pkey in ("donor_L_prompt", "donor_R_prompt"):
            fp = it[pkey]
            assert fp.endswith(sp)
            f_ids = tok_A.encode(fp, truncation=True, max_length=max_length, add_special_tokens=True)
            assert len(f_ids) >= len(s_ids) and f_ids[-len(s_ids) :] == s_ids, (
                f"Item {idx_i} ({pkey}): exact shared-suffix mismatch!"
            )
        suf_lens.append(len(s_ids))

    out_pol: Dict[str, Dict[str, Any]] = {}
    old_pad = tok_A.padding_side
    tok_A.padding_side = "left"

    for pol, pkey in (("L", "donor_L_prompt"), ("R", "donor_R_prompt")):
        prompts = [it[pkey] for it in items]
        buf: Dict[str, torch.Tensor] = {}
        handles: List[Any] = []
        h_in_last: Dict[int, List[np.ndarray]] = {l0: [], l1: [], le: []}
        z_attn_last: Dict[int, List[np.ndarray]] = {l0: [], l1: []}
        y_attn_last: Dict[int, List[np.ndarray]] = {l0: [], l1: []}
        z_mlp_last: Dict[int, List[np.ndarray]] = {l0: [], l1: [], le: []}
        y_mlp_last: Dict[int, List[np.ndarray]] = {l0: [], l1: []}
        u_in_le_suf: List[np.ndarray] = []
        u_post_le_last: List[np.ndarray] = []
        h_in_l0_suf: List[np.ndarray] = []

        try:
            for li in (l0, l1, le):
                def _mk_hin(k):
                    def _pre(_m, args):
                        buf[k] = args[0].detach()
                    return _pre
                handles.append(layers_A[li].input_layernorm.register_forward_pre_hook(_mk_hin(f"hin_{li}")))
            for li in (l0, l1):
                def _mk_pre(k):
                    def _pre(_m, args):
                        buf[k] = args[0].detach()
                    return _pre
                def _mk_out(k):
                    def _fn(_m, _i, o):
                        buf[k] = o.detach()
                    return _fn
                handles.append(layers_A[li].self_attn.o_proj.register_forward_pre_hook(_mk_pre(f"za_{li}")))
                handles.append(layers_A[li].self_attn.o_proj.register_forward_hook(_mk_out(f"ya_{li}")))
                handles.append(layers_A[li].mlp.down_proj.register_forward_pre_hook(_mk_pre(f"zm_{li}")))
                handles.append(layers_A[li].mlp.down_proj.register_forward_hook(_mk_out(f"ym_{li}")))
            # Capture native BF16 L20 down_proj input (`silu(gate(u)) * up(u)`) directly from Donor 7B!
            handles.append(layers_A[le].mlp.down_proj.register_forward_pre_hook(lambda _m, a: buf.__setitem__(f"zm_{le}", a[0].detach())))
            handles.append(layers_A[le].input_layernorm.register_forward_hook(lambda _m, _i, o: buf.__setitem__("uin_le", o.detach())))
            handles.append(layers_A[le].post_attention_layernorm.register_forward_hook(lambda _m, _i, o: buf.__setitem__("upost_le", o.detach())))

            for i0 in range(0, len(prompts), batch_size):
                bp = prompts[i0 : i0 + batch_size]
                bs_lens = suf_lens[i0 : i0 + batch_size]
                enc = tok_A(bp, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=True)
                ids, mask = enc["input_ids"].to(dev_A), enc["attention_mask"].to(dev_A)
                _ = backbone_A(input_ids=ids, attention_mask=mask, use_cache=False)

                for li in (l0, l1, le):
                    h_in_last[li].append(buf[f"hin_{li}"][:, -1, :].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
                    z_mlp_last[li].append(buf[f"zm_{li}"][:, -1, :].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
                for li in (l0, l1):
                    z_attn_last[li].append(buf[f"za_{li}"][:, -1, :].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
                    y_attn_last[li].append(buf[f"ya_{li}"][:, -1, :].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
                    y_mlp_last[li].append(buf[f"ym_{li}"][:, -1, :].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
                u_post_le_last.append(buf["upost_le"][:, -1, :].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
                for bi, slen in enumerate(bs_lens):
                    h_in_l0_suf.append(buf[f"hin_{l0}"][bi, -slen:, :].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
                    u_in_le_suf.append(buf["uin_le"][bi, -slen:, :].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
                buf.clear()
        finally:
            for h in handles:
                h.remove()
        assert_zero_runtime_hooks(model_A)

        out_pol[pol] = {
            "h_in_last": {li: np.concatenate(h_in_last[li], axis=0) for li in (l0, l1, le)},
            "z_attn_last": {li: np.concatenate(z_attn_last[li], axis=0) for li in (l0, l1)},
            "y_attn_last": {li: np.concatenate(y_attn_last[li], axis=0) for li in (l0, l1)},
            "z_mlp_last": {li: np.concatenate(z_mlp_last[li], axis=0) for li in (l0, l1, le)},
            "y_mlp_last": {li: np.concatenate(y_mlp_last[li], axis=0) for li in (l0, l1)},
            "u_post_le_last": np.concatenate(u_post_le_last, axis=0),
            "h_in_l0_suf": h_in_l0_suf,
            "u_in_le_suf": u_in_le_suf,
        }

    tok_A.padding_side = old_pad
    return {"L": out_pol["L"], "R": out_pol["R"], "suf_lens": suf_lens}


# ======================================================================================
# 5. Donor Unit Selection, Shuffled Control, Bootstrap CI & Pos>0 RoPE Attention RMSE
# ======================================================================================

def select_donor_units_by_lane_contrast(
    donor_layer: nn.Module,
    dz_attn_fit: np.ndarray,
    dz_mlp_fit: np.ndarray,
    V: np.ndarray,
    add_q_heads: int = 6,
    add_kv_heads: int = 1,
    add_mlp_rows: int = 256,
) -> Dict[str, Any]:
    attn = donor_layer.self_attn
    mlp = donor_layer.mlp
    d_h = int(attn.head_dim)
    g_A = int(attn.num_key_value_groups)
    n_q_A = int(attn.q_proj.out_features // d_h)
    n_kv_A = int(attn.k_proj.out_features // d_h)
    assert add_kv_heads == 1 and add_q_heads <= g_A

    Wq = attn.q_proj.weight.detach().float().cpu().numpy()
    Wk = attn.k_proj.weight.detach().float().cpu().numpy()
    Wv = attn.v_proj.weight.detach().float().cpu().numpy()
    Wo = attn.o_proj.weight.detach().float().cpu().numpy()
    bq = attn.q_proj.bias.detach().float().cpu().numpy() if attn.q_proj.bias is not None else None
    bk = attn.k_proj.bias.detach().float().cpu().numpy() if attn.k_proj.bias is not None else None
    bv = attn.v_proj.bias.detach().float().cpu().numpy() if attn.v_proj.bias is not None else None

    Vt_Wo = V.T @ Wo.astype(np.float64)
    head_attr = np.zeros(n_q_A, dtype=np.float64)
    for h in range(n_q_A):
        sl = slice(h * d_h, (h + 1) * d_h)
        proj_h = dz_attn_fit[:, sl] @ Vt_Wo[:, sl].T
        head_attr[h] = float(np.mean(np.sum(proj_h ** 2, axis=1)))

    best_kv, best_sel_q, best_score = 0, list(range(add_q_heads)), -1.0
    for kv in range(n_kv_A):
        qs = list(range(kv * g_A, (kv + 1) * g_A))
        qs.sort(key=lambda h: float(head_attr[h]), reverse=True)
        top_q = sorted(qs[:add_q_heads])
        sc = float(sum(head_attr[h] for h in top_q))
        if sc > best_score:
            best_score, best_kv, best_sel_q = sc, kv, top_q

    q_rows = np.concatenate([np.arange(h * d_h, (h + 1) * d_h) for h in best_sel_q])
    kv_rows = np.arange(best_kv * d_h, (best_kv + 1) * d_h)

    Wg = mlp.gate_proj.weight.detach().float().cpu().numpy()
    Wu = mlp.up_proj.weight.detach().float().cpu().numpy()
    Wd = mlp.down_proj.weight.detach().float().cpu().numpy()
    Vt_Wd = V.T @ Wd.astype(np.float64)
    col_e = np.sum(Vt_Wd ** 2, axis=0)
    mlp_attr = np.mean(dz_mlp_fit ** 2, axis=0) * col_e
    k_mlp = min(int(add_mlp_rows), Wg.shape[0])
    sel_n = np.sort(np.argsort(mlp_attr)[-k_mlp:])

    return {
        "best_kv_group": int(best_kv),
        "sel_q_heads": [int(h) for h in best_sel_q],
        "q_rows": q_rows,
        "add_q_heads": int(add_q_heads),
        "add_kv_heads": int(add_kv_heads),
        "q_w": Wq[q_rows].copy(),
        "q_b": bq[q_rows].copy() if bq is not None else None,
        "k_w": Wk[kv_rows].copy(),
        "k_b": bk[kv_rows].copy() if bk is not None else None,
        "v_w": Wv[kv_rows].copy(),
        "v_b": bv[kv_rows].copy() if bv is not None else None,
        "o_w": Wo[:, q_rows].copy(),
        "sel_mlp_indices": sel_n,
        "gate_w": Wg[sel_n].copy(),
        "up_w": Wu[sel_n].copy(),
        "down_w": Wd[:, sel_n].copy(),
        "gamma_in": donor_layer.input_layernorm.weight.detach().cpu().double().numpy(),
        "gamma_post": donor_layer.post_attention_layernorm.weight.detach().cpu().double().numpy(),
    }


def extract_wrong_layer_shuffled_units(
    wrong_donor_layer: nn.Module,
    real_units: Dict[str, Any],
    seed: int = SEED,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    attn = wrong_donor_layer.self_attn
    mlp = wrong_donor_layer.mlp
    d_h = int(attn.head_dim)
    g_A = int(attn.num_key_value_groups)
    n_kv_A = int(attn.k_proj.out_features // d_h)
    add_q = int(real_units["add_q_heads"])
    k_mlp = int(real_units["gate_w"].shape[0])

    kv = int(rng.integers(0, n_kv_A))
    sel_q = sorted(rng.choice(list(range(kv * g_A, (kv + 1) * g_A)), size=add_q, replace=False).tolist())
    q_rows = np.concatenate([np.arange(h * d_h, (h + 1) * d_h) for h in sel_q])
    kv_rows = np.arange(kv * d_h, (kv + 1) * d_h)
    sel_n = np.sort(rng.choice(mlp.gate_proj.weight.shape[0], size=k_mlp, replace=False))

    def _match_norm(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
        ns = float(np.linalg.norm(src)) + 1e-12
        nr = float(np.linalg.norm(ref))
        return (src * (nr / ns)).astype(src.dtype)

    Wq = attn.q_proj.weight.detach().float().cpu().numpy()[q_rows]
    Wk = attn.k_proj.weight.detach().float().cpu().numpy()[kv_rows]
    Wv = attn.v_proj.weight.detach().float().cpu().numpy()[kv_rows]
    Wo = attn.o_proj.weight.detach().float().cpu().numpy()[:, q_rows]
    bq = attn.q_proj.bias.detach().float().cpu().numpy()[q_rows] if attn.q_proj.bias is not None else None
    bk = attn.k_proj.bias.detach().float().cpu().numpy()[kv_rows] if attn.k_proj.bias is not None else None
    bv = attn.v_proj.bias.detach().float().cpu().numpy()[kv_rows] if attn.v_proj.bias is not None else None
    Wg = mlp.gate_proj.weight.detach().float().cpu().numpy()[sel_n]
    Wu = mlp.up_proj.weight.detach().float().cpu().numpy()[sel_n]
    Wd = mlp.down_proj.weight.detach().float().cpu().numpy()[:, sel_n]

    return {
        "best_kv_group": kv,
        "sel_q_heads": sel_q,
        "add_q_heads": add_q,
        "add_kv_heads": 1,
        "q_w": _match_norm(Wq, real_units["q_w"]),
        "q_b": _match_norm(bq, real_units["q_b"]) if (bq is not None and real_units["q_b"] is not None) else None,
        "k_w": _match_norm(Wk, real_units["k_w"]),
        "k_b": _match_norm(bk, real_units["k_b"]) if (bk is not None and real_units["k_b"] is not None) else None,
        "v_w": _match_norm(Wv, real_units["v_w"]),
        "v_b": _match_norm(bv, real_units["v_b"]) if (bv is not None and real_units["v_b"] is not None) else None,
        "o_w": _match_norm(Wo, real_units["o_w"]),
        "gate_w": _match_norm(Wg, real_units["gate_w"]),
        "up_w": _match_norm(Wu, real_units["up_w"]),
        "down_w": _match_norm(Wd, real_units["down_w"]),
        "gamma_in": real_units["gamma_in"].copy(),
        "gamma_post": real_units["gamma_post"].copy(),
    }


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    denom = float(np.sum((yt - yt.mean(axis=0, keepdims=True)) ** 2)) + 1e-12
    return float(1.0 - float(np.sum((yt - yp) ** 2)) / denom)


def paired_key_group_bootstrap_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    group_keys: Sequence[str],
    n_boot: int = 1000,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Return full-precision AND rounded 95% CI for paired key-group cluster bootstrap of `Delta R^2`!"""
    rng = np.random.default_rng(seed)
    groups: Dict[str, List[int]] = {}
    for idx, g in enumerate(group_keys):
        groups.setdefault(str(g), []).append(idx)
    g_list = [np.asarray(idxs, dtype=np.int64) for idxs in groups.values()]
    n_g = len(g_list)
    point = compute_r2(y_true, y_pred_a) - compute_r2(y_true, y_pred_b)
    diffs = np.empty(n_boot, dtype=np.float64)
    for bi in range(n_boot):
        pick = rng.integers(0, n_g, size=n_g)
        idxs = np.concatenate([g_list[p] for p in pick])
        diffs[bi] = compute_r2(y_true[idxs], y_pred_a[idxs]) - compute_r2(y_true[idxs], y_pred_b[idxs])
    lb95, ub95 = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff_point": round(float(point), 4),
        "ci95_low": round(float(lb95), 4),
        "ci95_high": round(float(ub95), 4),
        "_ci95_low_raw": float(lb95),
        "n_groups": n_g,
    }


def _apply_rope_single_head(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x_rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x * cos) + (x_rot * sin)


@torch.no_grad()
def compute_l20_rope_attn_logit_rmse(
    u_in_A_suf: Sequence[np.ndarray],
    u_in_B_suf: Sequence[np.ndarray],
    Wq_h_A: np.ndarray,
    bq_h_A: Optional[np.ndarray],
    Wk_kv_A: np.ndarray,
    bk_kv_A: Optional[np.ndarray],
    P_perp_20_attn: np.ndarray,
    gamma_in_20_A: np.ndarray,
    V: np.ndarray,
    rope_theta: float,
) -> Tuple[float, float]:
    """
    Compute exact RoPE attention-logit RMSE (nats) at L20's top contrast donor head:
      - Primary: `pos > 0` shared-suffix key positions (`lgB[1:] - lgA[1:]`, excluding 1.5B's pos-0 sink
        which aligns to Donor 7B's mid-sequence position ~92!).
      - Auxiliary: `all_pos` across all shared-suffix positions.
    """
    d_h = int(Wq_h_A.shape[0])
    inv_freq = 1.0 / (float(rope_theta) ** (torch.arange(0, d_h, 2, dtype=torch.float64) / float(d_h)))
    Gv = gamma_in_20_A[:, None] * V
    R_mat = np.hstack([P_perp_20_attn, Gv])
    Wq_B = Wq_h_A.astype(np.float64) @ R_mat
    Wk_B = Wk_kv_A.astype(np.float64) @ R_mat
    Wq_A_f = Wq_h_A.astype(np.float64)
    Wk_A_f = Wk_kv_A.astype(np.float64)

    sq_gt0: List[np.ndarray] = []
    sq_all: List[np.ndarray] = []
    inv_sqrt_dh = 1.0 / math.sqrt(float(d_h))
    for uA_u16, uB_u16 in zip(u_in_A_suf, u_in_B_suf):
        uA = bf16_to_f64_np(uA_u16)
        uB = bf16_to_f64_np(uB_u16)
        S = int(uB.shape[0])
        assert uA.shape[0] == S
        pos = torch.arange(S, dtype=torch.float64)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_t, sin_t = emb.cos(), emb.sin()

        qA = torch.from_numpy(uA[-1:] @ Wq_A_f.T + (bq_h_A if bq_h_A is not None else 0.0))
        kA = torch.from_numpy(uA @ Wk_A_f.T + (bk_kv_A if bk_kv_A is not None else 0.0))
        qB = torch.from_numpy(uB[-1:] @ Wq_B.T + (bq_h_A if bq_h_A is not None else 0.0))
        kB = torch.from_numpy(uB @ Wk_B.T + (bk_kv_A if bk_kv_A is not None else 0.0))

        qA_r = _apply_rope_single_head(qA, cos_t[-1:], sin_t[-1:])
        kA_r = _apply_rope_single_head(kA, cos_t, sin_t)
        qB_r = _apply_rope_single_head(qB, cos_t[-1:], sin_t[-1:])
        kB_r = _apply_rope_single_head(kB, cos_t, sin_t)

        lgA = (qA_r @ kA_r.T).squeeze(0).numpy() * inv_sqrt_dh
        lgB = (qB_r @ kB_r.T).squeeze(0).numpy() * inv_sqrt_dh
        diff2 = (lgB - lgA) ** 2
        sq_all.append(diff2)
        sq_gt0.append(diff2[1:] if S > 1 else diff2)
    return float(math.sqrt(float(np.mean(np.concatenate(sq_gt0))))), float(math.sqrt(float(np.mean(np.concatenate(sq_all)))))


@torch.no_grad()
def native_bf16_swiglu_eval(
    u_post_u16: np.ndarray,
    Wg_f64: np.ndarray,
    Wu_f64: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate pulled-back L20 MLP top-64 neurons through **native PyTorch BF16** linear projections,
    `F.silu`, and elementwise product (identical to Qwen2MLP's native BF16 execution!).
    Returns `(gate_bf16_as_f64, up_bf16_as_f64, swiglu_prod_bf16_as_f64)`.
    """
    u_t = torch.from_numpy(u_post_u16).view(torch.bfloat16).to(device=device)
    Wg_t = torch.from_numpy(Wg_f64.astype(np.float32)).to(device=device, dtype=torch.bfloat16)
    Wu_t = torch.from_numpy(Wu_f64.astype(np.float32)).to(device=device, dtype=torch.bfloat16)
    g = F.linear(u_t, Wg_t)
    u = F.linear(u_t, Wu_t)
    y = F.silu(g) * u
    return g.cpu().double().numpy(), u.cpu().double().numpy(), y.cpu().double().numpy()


# ======================================================================================
# 6. Recipient Seeded Forward (`L18 -> L20`, Pos-0 Untouched!) & 100% Zero-Hook Metric 5
# ======================================================================================

@torch.inference_mode()
def run_recipient_seeded_l18_to_l20(
    model_B: nn.Module,
    tok_B: Any,
    prompts: Sequence[str],
    seeds_l18: Sequence[np.ndarray],
    alpha: float,
    dev_B: torch.device,
    l0: int = 18,
    l1: int = 19,
    le: int = 20,
    capture_organ_z_layer: Optional[int] = None,
    z_attn_tail: int = 0,
    z_mlp_tail: int = 0,
    capture_le_u_in_suf: bool = False,
    batch_size: int = 16,
    max_length: int = 128,
) -> Dict[str, Any]:
    """
    Run widened Recipient 1.5B natively from L0 -> L20 with exact oracle private-lane seeding at `L18_in`
    strictly on `pos > 0` shared-suffix tokens (`h[bi, -slen + 1 :, -K:] = alpha * seed[1:]`), keeping
    Recipient 1.5B's position-0 attention sink untouched!
    """
    layers_B = get_decoder_layers(model_B)
    backbone_B = model_B.model if hasattr(model_B, "model") else model_B
    handles: List[Any] = []
    buf: Dict[str, torch.Tensor] = {}
    cur_seeds: List[np.ndarray] = []
    h_in_l0_last: List[np.ndarray] = []
    h_in_le_last: List[np.ndarray] = []
    u_post_le_last: List[np.ndarray] = []
    u_in_le_suf: List[np.ndarray] = []
    z_attn_out: List[np.ndarray] = []
    z_mlp_out: List[np.ndarray] = []
    old_pad = tok_B.padding_side
    tok_B.padding_side = "left"

    try:
        def _seed_pre(_m, args):
            h = args[0].clone()
            for bi, sd_np in enumerate(cur_seeds):
                slen, K_sd = sd_np.shape
                if slen > 1:
                    sd_t = torch.from_numpy(sd_np[1:]).to(device=h.device, dtype=h.dtype) * float(alpha)
                    h[bi, -slen + 1 :, -K_sd:] = sd_t
            buf["hin_l0"] = h[:, -1, :].detach()
            return (h,) + args[1:]

        handles.append(layers_B[l0].register_forward_pre_hook(_seed_pre))
        handles.append(layers_B[le].input_layernorm.register_forward_pre_hook(lambda _m, a: buf.__setitem__("hin_le", a[0][:, -1, :].detach())))
        handles.append(layers_B[le].post_attention_layernorm.register_forward_hook(lambda _m, _i, o: buf.__setitem__("upost_le", o[:, -1, :].detach())))
        if capture_le_u_in_suf:
            handles.append(layers_B[le].input_layernorm.register_forward_hook(lambda _m, _i, o: buf.__setitem__("uin_le", o.detach())))
        if capture_organ_z_layer is not None:
            li_z = int(capture_organ_z_layer)
            if z_attn_tail > 0:
                handles.append(layers_B[li_z].self_attn.o_proj.register_forward_pre_hook(lambda _m, a: buf.__setitem__("za", a[0][:, -1, -z_attn_tail:].detach())))
            if z_mlp_tail > 0:
                handles.append(layers_B[li_z].mlp.down_proj.register_forward_pre_hook(lambda _m, a: buf.__setitem__("zm", a[0][:, -1, -z_mlp_tail:].detach())))

        for i0 in range(0, len(prompts), batch_size):
            bp = list(prompts[i0 : i0 + batch_size])
            cur_seeds = list(seeds_l18[i0 : i0 + batch_size])
            enc = tok_B(bp, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=True)
            ids, mask = enc["input_ids"].to(dev_B), enc["attention_mask"].to(dev_B)
            _ = backbone_B(input_ids=ids, attention_mask=mask, use_cache=False)

            h_in_l0_last.append(buf["hin_l0"].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
            h_in_le_last.append(buf["hin_le"].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
            u_post_le_last.append(buf["upost_le"].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
            if capture_le_u_in_suf:
                for bi, sd_np in enumerate(cur_seeds):
                    slen = int(sd_np.shape[0])
                    u_in_le_suf.append(buf["uin_le"][bi, -slen:, :].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
            if "za" in buf:
                z_attn_out.append(buf["za"].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
            if "zm" in buf:
                z_mlp_out.append(buf["zm"].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
            buf.clear()
    finally:
        for h in handles:
            h.remove()
        tok_B.padding_side = old_pad
    assert_zero_runtime_hooks(model_B)

    res: Dict[str, Any] = {
        "h_in_l0_last": np.concatenate(h_in_l0_last, axis=0),
        "h_in_le_last": np.concatenate(h_in_le_last, axis=0),
        "u_post_le_last": np.concatenate(u_post_le_last, axis=0),
    }
    if capture_le_u_in_suf:
        res["u_in_le_suf"] = u_in_le_suf
    if z_attn_out:
        res["z_attn_tail"] = np.concatenate(z_attn_out, axis=0)
    if z_mlp_out:
        res["z_mlp_tail"] = np.concatenate(z_mlp_out, axis=0)
    return res


@torch.inference_mode()
def capture_inertness_reference_no_hooks(
    model_B: nn.Module,
    tok_B: Any,
    passages: Sequence[str],
    v_shared: int,
    dev_B: torch.device,
    eval_layer: int = 20,
    batch_size: int = 4,
    max_length: int = 128,
) -> Tuple[List[torch.Tensor], np.ndarray]:
    """100% ZERO-HOOK baseline reference capture for Metric 5 KL & `rho`!"""
    assert_zero_runtime_hooks(model_B)
    old_pad = tok_B.padding_side
    tok_B.padding_side = "left"
    ref_logits: List[torch.Tensor] = []
    ref_rms: List[np.ndarray] = []
    eps_ln = float(get_decoder_layers(model_B)[eval_layer].input_layernorm.variance_epsilon)
    try:
        for i0 in range(0, len(passages), batch_size):
            bp = list(passages[i0 : i0 + batch_size])
            enc = tok_B(bp, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=True)
            ids, mask = enc["input_ids"].to(dev_B), enc["attention_mask"].to(dev_B)
            vm = mask.bool()
            out = model_B(input_ids=ids, attention_mask=mask, use_cache=False)
            ref_logits.append(out.logits[vm][:, :v_shared].cpu().double())
            del out
            h_le = forward_qwen2_prefix_no_hooks(model_B, ids, mask, stop_layer=eval_layer)
            h_v = h_le[vm].cpu().double()
            ref_rms.append(torch.sqrt(h_v.pow(2).mean(dim=-1) + eps_ln).numpy())
    finally:
        tok_B.padding_side = old_pad
    assert_zero_runtime_hooks(model_B)
    return ref_logits, np.concatenate(ref_rms, axis=0)


@torch.inference_mode()
def measure_generic_inertness_no_hooks(
    model_B: nn.Module,
    tok_B: Any,
    passages: Sequence[str],
    base_logits_list: Sequence[torch.Tensor],
    base_rms_l20: np.ndarray,
    D_orig: int,
    K: int,
    v_shared: int,
    dev_B: torch.device,
    eval_layer: int = 20,
    batch_size: int = 4,
    max_length: int = 128,
) -> Dict[str, Any]:
    """100% ZERO-HOOK Metric 5 inertness: token-level `KL(B || B')` AND hook-free prefix `rho`!"""
    assert_zero_runtime_hooks(model_B)
    old_pad = tok_B.padding_side
    tok_B.padding_side = "left"
    kl_sum = 0.0
    tok_cnt = 0
    rms_cur: List[np.ndarray] = []
    eps_ln = float(get_decoder_layers(model_B)[eval_layer].input_layernorm.variance_epsilon)
    scale_g = math.sqrt(float(D_orig) / float(D_orig + K))
    try:
        for bi, i0 in enumerate(range(0, len(passages), batch_size)):
            bp = list(passages[i0 : i0 + batch_size])
            enc = tok_B(bp, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=True)
            ids, mask = enc["input_ids"].to(dev_B), enc["attention_mask"].to(dev_B)
            vm = mask.bool()
            out = model_B(input_ids=ids, attention_mask=mask, use_cache=False)
            lg_cur = out.logits[vm][:, :v_shared].cpu().double()
            lg_ref = base_logits_list[bi]
            kl_sum += float(F.kl_div(F.log_softmax(lg_cur, dim=-1), F.softmax(lg_ref, dim=-1), reduction="sum").item())
            tok_cnt += int(lg_cur.shape[0])
            del out, lg_cur

            h_le = forward_qwen2_prefix_no_hooks(model_B, ids, mask, stop_layer=eval_layer)
            h_v = h_le[vm].cpu().double()
            rms_cur.append(torch.sqrt(h_v.pow(2).mean(dim=-1) + eps_ln).numpy())
    finally:
        tok_B.padding_side = old_pad
    assert_zero_runtime_hooks(model_B)

    rms_cat = np.concatenate(rms_cur, axis=0)
    kl_raw = kl_sum / max(tok_cnt, 1)
    rho_raw = float(np.mean(np.abs(rms_cat / (scale_g * base_rms_l20 + 1e-12) - 1.0)))
    return {
        "generic_kl_nat_per_tok": round(kl_raw, 6),
        "rms_shift_rho": round(rho_raw, 6),
        "_kl_raw": kl_raw,
        "_rho_raw": rho_raw,
        "n_tokens": tok_cnt,
        "zero_hooks_verified": True,
    }


# ======================================================================================
# 7. Scale-Normalized Ridge Lane Write (Inner 320/80 Val Split — Identical for REAL & SHUF!)
# ======================================================================================

def fit_scale_normalized_ridge_lane_write(
    Z_contrast_fit: np.ndarray,
    dY_lane_fit: np.ndarray,
    Z_bg_gt0: np.ndarray,
    n_inner_train: int = 320,
    fixed_cfg: Optional[Tuple[float, float]] = None,
    fit_on_inner_only: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Scale-normalized background-regularized ridge for lane writes (`REAL_REFIT` and `SHUF` use IDENTICAL solver!):
      - Scales `lam_eye` relative to `median_pos_eig(G)` and `lam_bg` relative to `tr_G / tr_Cbg`.
      - When `fit_on_inner_only=True`: tunes `(lam_bg_rel, lam_eye_rel)` on inner `320 / 80` split and returns
        the weight fitted STRICTLY on `[:320]` so downstream sublayers NEVER see `[320:400]` during tuning!
      - When `fixed_cfg` is provided: refits on ALL `400` fit pairs using the tuned `(lam_bg_rel, lam_eye_rel)`.
    """
    Z = np.asarray(Z_contrast_fit, dtype=np.float64)
    Y = np.asarray(dY_lane_fit, dtype=np.float64)
    Zb = np.asarray(Z_bg_gt0, dtype=np.float64)
    d_in = Z.shape[1]
    n_tot = len(Z)
    n_tr = min(int(n_inner_train), max(n_tot - 20, 1))
    Z_tr, Y_tr = Z[:n_tr], Y[:n_tr]
    Z_va, Y_va = Z[n_tr:], Y[n_tr:]

    C_bg = (Zb.T @ Zb) / max(float(len(Zb)), 1.0)
    C_bg = 0.5 * (C_bg + C_bg.T)
    tr_bg = float(np.trace(C_bg)) / max(d_in, 1)

    def _solve(Z_s: np.ndarray, Y_s: np.ndarray, l_bg_rel: float, l_eye_rel: float) -> np.ndarray:
        ns = max(float(len(Z_s)), 1.0)
        G = (Z_s.T @ Z_s) / ns
        G = 0.5 * (G + G.T)
        ev = sla.eigvalsh(G)
        pos = ev[ev > 1e-12]
        med_g = float(np.median(pos)) if len(pos) > 0 else float(np.trace(G) / max(d_in, 1))
        tr_g = float(np.trace(G)) / max(d_in, 1)
        lam_eye = max(float(l_eye_rel) * med_g, 1e-8)
        lam_bg = float(l_bg_rel) * (tr_g / max(tr_bg, 1e-12))
        A = G + lam_bg * C_bg + lam_eye * np.eye(d_in, dtype=np.float64)
        B = (Z_s.T @ Y_s) / ns
        return sla.solve(A, B, assume_a="pos").T  # (K, d_in)

    if fixed_cfg is not None:
        W_final = _solve(Z, Y, fixed_cfg[0], fixed_cfg[1])
        return W_final, {"best_lam_bg_rel": fixed_cfg[0], "best_lam_eye_rel": fixed_cfg[1], "fit_n": n_tot}

    best_r2 = -1e18
    best_cfg = (0.1, 1e-2)
    best_W_tr: Optional[np.ndarray] = None
    if len(Z_va) > 0:
        for lb_r in (0.01, 0.1, 1.0):
            for le_r in (1e-3, 1e-2, 1e-1, 1.0):
                W_c = _solve(Z_tr, Y_tr, lb_r, le_r)
                r2 = compute_r2(Y_va, Z_va @ W_c.T)
                if r2 > best_r2:
                    best_r2 = r2
                    best_cfg = (lb_r, le_r)
                    best_W_tr = W_c
    if fit_on_inner_only:
        W_out = best_W_tr if best_W_tr is not None else _solve(Z_tr, Y_tr, best_cfg[0], best_cfg[1])
        return W_out, {"best_lam_bg_rel": best_cfg[0], "best_lam_eye_rel": best_cfg[1], "inner_val_r2": round(best_r2, 4), "fit_n": len(Z_tr)}
    W_final = _solve(Z, Y, best_cfg[0], best_cfg[1])
    return W_final, {"best_lam_bg_rel": best_cfg[0], "best_lam_eye_rel": best_cfg[1], "inner_val_r2": round(best_r2, 4), "fit_n": n_tot}


# ======================================================================================
# 8. Main 5-Minute Go/No-Go Falsifier Runner (Single-Model Peak VRAM < 16 GiB!)
# ======================================================================================

def run_private_lane_falsifier(args: argparse.Namespace) -> Dict[str, Any]:
    t_all = time.time()
    K = int(args.k_lanes)
    alpha = float(args.alpha)
    l0, l1, le = int(args.l0), int(args.l1), int(args.le)
    shuf_l0, shuf_l1 = 6, 7
    print(f"\n================ V4.4.1 5-MINUTE PRIVATE-LANE FALSIFIER (L{l0}->L{le}, K={K}, alpha={alpha}) ================")

    cal = build_60k_calibration_corpus(n_fit_passages=220, n_val_passages=40, seed=SEED)
    loc_facts = t1d.build_certified_locality_facts()
    vram_telemetry: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # PHASE 1: DONOR 7B ONLY (PEAK VRAM ~15 GiB, 100% UNLOADED BEFORE PHASE 2!)
    # -------------------------------------------------------------------------
    t_p1 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model_A, tok_A, dev_A, _ = load_model_bf16(args.donor_model, allow_download=args.allow_download)
    rope_theta = get_config_rope_theta(model_A.config)
    layers_A = get_decoder_layers(model_A)

    ds2 = build_2row_2hop_dataset([tok_A], n_train=args.n_fit_pairs, n_dev=0, n_test=args.n_test_pairs, seed=SEED + 101)
    assert set(ds2["train_keys"]).isdisjoint(set(ds2["test_keys"]))
    print(f"  [Data] 2-Hop key-disjoint: {len(ds2['train'])} fit pairs, {len(ds2['test'])} unseen-key test pairs (0 key overlap!)")

    # 1A. Generic pos > 0 + per-token pos == 0 capture on Donor 7B (~20k tokens, pos 0 strictly excluded!)
    don_gen_fit = capture_generic_pos0_and_gt0(model_A, tok_A, cal["fit"], [l0, l1, le], dev_A, batch_size=8, max_length=160, max_gt0_tokens=args.n_bg_tokens)
    don_gen_val = capture_generic_pos0_and_gt0(model_A, tok_A, cal["val_tune"], [l0, l1, le], dev_A, batch_size=8, max_length=160, max_gt0_tokens=2048, capture_pos0_tokens=False)
    print(f"  [Donor Generic] N_gt0_fit={don_gen_fit['n_gt0']} (pos 0 excluded!), N_pos0={don_gen_fit['n_pos0']}, N_gt0_val={don_gen_val['n_gt0']}")

    # 1B. Donor 7B 2-Hop fit (400 pairs) & unseen-key test (400 pairs)
    don_tr = capture_donor_2hop_bundle(model_A, tok_A, ds2["train"], dev_A, org_layers=(l0, l1), eval_layer=le, batch_size=8)
    don_te = capture_donor_2hop_bundle(model_A, tok_A, ds2["test"], dev_A, org_layers=(l0, l1), eval_layer=le, batch_size=8)

    # 1C. Fit Primal Interface Basis V in R^{3584 x 128} (FP64, streamed centered bg!)
    con_pool = np.concatenate([
        bf16_to_f64_np(don_tr["L"]["h_in_last"][li]) - bf16_to_f64_np(don_tr["R"]["h_in_last"][li])
        for li in (l0, l1, le)
    ], axis=0)
    bg_u16_list = [don_gen_fit["gt0"][li]["h_in"][::3] for li in (l0, l1, le)]
    basis = fit_generalized_interface_basis(con_pool, bg_u16_list, K=K, bg_reg_rel_med=1e-2)
    V = basis["V"]
    del con_pool, bg_u16_list

    # Metric 1 (Donor-only Interface Capture `c` strictly on unseen-key contrast INCREMENT `dh_A(L20) - dh_A(L18)`!)
    dh_A_te_l0 = bf16_to_f64_np(don_te["L"]["h_in_last"][l0]) - bf16_to_f64_np(don_te["R"]["h_in_last"][l0])
    dh_A_te_le = bf16_to_f64_np(don_te["L"]["h_in_last"][le]) - bf16_to_f64_np(don_te["R"]["h_in_last"][le])
    dDelta_h_A_te = dh_A_te_le - dh_A_te_l0
    metric1_c_raw = compute_whitened_interface_capture(dDelta_h_A_te, V, basis["Q_bg"], basis["inv_sqrt_bg"])
    metric1_c_endpoint = compute_whitened_interface_capture(dh_A_te_le, V, basis["Q_bg"], basis["inv_sqrt_bg"])
    print(f"  [Metric 1 | Interface Capture c] whitened unseen delta_Delta capture = {metric1_c_raw:.4f} (endpoint L{le}={metric1_c_endpoint:.4f}; Kill < 0.60 / Go >= 0.80)")

    # 1D. Select L18 & L19 real donor units + L6 & L7 wrong-layer shuffled units!
    real_units: Dict[int, Dict[str, Any]] = {}
    shuf_units: Dict[int, Dict[str, Any]] = {}
    for li, sh_li in ((l0, shuf_l0), (l1, shuf_l1)):
        dza_tr = bf16_to_f64_np(don_tr["L"]["z_attn_last"][li]) - bf16_to_f64_np(don_tr["R"]["z_attn_last"][li])
        dzm_tr = bf16_to_f64_np(don_tr["L"]["z_mlp_last"][li]) - bf16_to_f64_np(don_tr["R"]["z_mlp_last"][li])
        real_units[li] = select_donor_units_by_lane_contrast(layers_A[li], dza_tr, dzm_tr, V, add_q_heads=6, add_kv_heads=1, add_mlp_rows=args.add_mlp_rows)
        shuf_units[li] = extract_wrong_layer_shuffled_units(layers_A[sh_li], real_units[li], seed=SEED + li)

    # Metric 2 (Donor-only Organ Compute Budget `beta` on unseen-key `V^T delta_Delta h_A`!)
    dDelta_z_A_tr = (
        (bf16_to_f64_np(don_tr["L"]["h_in_last"][le]) - bf16_to_f64_np(don_tr["R"]["h_in_last"][le]))
        - (bf16_to_f64_np(don_tr["L"]["h_in_last"][l0]) - bf16_to_f64_np(don_tr["R"]["h_in_last"][l0]))
    ) @ V
    dDelta_z_A_te = dDelta_h_A_te @ V

    def _donor_selected_feats(bundle: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        feats = []
        copied_sum = np.zeros((len(bundle["L"]["h_in_last"][l0]), K), dtype=np.float64)
        for li in (l0, l1):
            ru = real_units[li]
            dza = (bf16_to_f64_np(bundle["L"]["z_attn_last"][li]) - bf16_to_f64_np(bundle["R"]["z_attn_last"][li]))[:, ru["q_rows"]]
            dzm = (bf16_to_f64_np(bundle["L"]["z_mlp_last"][li]) - bf16_to_f64_np(bundle["R"]["z_mlp_last"][li]))[:, ru["sel_mlp_indices"]]
            feats.extend([dza, dzm])
            copied_sum += dza @ (V.T @ ru["o_w"].astype(np.float64)).T
            copied_sum += dzm @ (V.T @ ru["down_w"].astype(np.float64)).T
        return np.hstack(feats), copied_sum

    F_don_tr, _ = _donor_selected_feats(don_tr)
    F_don_te, copied_don_te = _donor_selected_feats(don_te)
    C_f = (F_don_tr.T @ F_don_tr) / float(len(F_don_tr))
    ef = sla.eigvalsh(C_f)
    lam_f = max(1e-2 * float(np.median(ef[ef > 1e-12])), 1e-8)
    W_beta = sla.solve(C_f + lam_f * np.eye(F_don_tr.shape[1]), (F_don_tr.T @ dDelta_z_A_tr) / float(len(F_don_tr)), assume_a="pos")
    beta_refit_raw = compute_r2(dDelta_z_A_te, F_don_te @ W_beta)
    beta_copied_raw = compute_r2(dDelta_z_A_te, copied_don_te)
    print(f"  [Metric 2 | Organ Compute Budget beta] donor_refit R2={beta_refit_raw:.4f} | donor_copied R2={beta_copied_raw:.4f} (Kill < 0.50 / Go >= 0.70)")

    # 1E. Capture L20 donor top-64 MLP neurons (native BF16 gate/up AND actual captured `down_proj` input!) + top attn head!
    lyr_le = layers_A[le]
    Wg_le = lyr_le.mlp.gate_proj.weight.detach().cpu().double().numpy()
    Wu_le = lyr_le.mlp.up_proj.weight.detach().cpu().double().numpy()
    du_post_le_tr = bf16_to_f64_np(don_tr["L"]["u_post_le_last"]) - bf16_to_f64_np(don_tr["R"]["u_post_le_last"])
    g_le_tr = du_post_le_tr @ Wg_le.T
    u_le_tr = du_post_le_tr @ Wu_le.T
    top64_l20_mlp = np.sort(np.argsort(np.mean(g_le_tr ** 2 + u_le_tr ** 2, axis=0))[-64:])
    Wg_le_top64 = Wg_le[top64_l20_mlp].copy()
    Wu_le_top64 = Wu_le[top64_l20_mlp].copy()
    del Wg_le, Wu_le, g_le_tr, u_le_tr
    gamma_in_le_A = lyr_le.input_layernorm.weight.detach().cpu().double().numpy()
    gamma_post_le_A = lyr_le.post_attention_layernorm.weight.detach().cpu().double().numpy()

    # Exact Donor 7B native BF16 SwiGLU product contrast (captured directly from Donor 7B's `down_proj` input!)
    dyA_swiglu_te = (
        bf16_to_f64_np(don_te["L"]["z_mlp_last"][le][:, top64_l20_mlp])
        - bf16_to_f64_np(don_te["R"]["z_mlp_last"][le][:, top64_l20_mlp])
    )
    # Donor 7B native BF16 gate/up pre-activations for top-64 L20 neurons
    gA_L_te, uA_L_te, _ = native_bf16_swiglu_eval(don_te["L"]["u_post_le_last"], Wg_le_top64, Wu_le_top64, dev_A)
    gA_R_te, uA_R_te, _ = native_bf16_swiglu_eval(don_te["R"]["u_post_le_last"], Wg_le_top64, Wu_le_top64, dev_A)
    dgA_te, duA_te = gA_L_te - gA_R_te, uA_L_te - uA_R_te
    gA_L_tr, uA_L_tr, _ = native_bf16_swiglu_eval(don_tr["L"]["u_post_le_last"], Wg_le_top64, Wu_le_top64, dev_A)
    gA_R_tr, uA_R_tr, _ = native_bf16_swiglu_eval(don_tr["R"]["u_post_le_last"], Wg_le_top64, Wu_le_top64, dev_A)

    attn_le = lyr_le.self_attn
    d_h_le = int(attn_le.head_dim)
    g_le = int(attn_le.num_key_value_groups)
    n_q_le = int(attn_le.q_proj.out_features // d_h_le)
    Wq_le_all = attn_le.q_proj.weight.detach().float().cpu().numpy()
    Wk_le_all = attn_le.k_proj.weight.detach().float().cpu().numpy()
    bq_le_all = attn_le.q_proj.bias.detach().float().cpu().numpy() if attn_le.q_proj.bias is not None else None
    bk_le_all = attn_le.k_proj.bias.detach().float().cpu().numpy() if attn_le.k_proj.bias is not None else None
    du_in_le_tr = np.stack([bf16_to_f64_np(a[-1:])[0] - bf16_to_f64_np(b[-1:])[0] for a, b in zip(don_tr["L"]["u_in_le_suf"], don_tr["R"]["u_in_le_suf"])], axis=0)
    dq_le_tr = du_in_le_tr @ Wq_le_all.T.astype(np.float64)
    top_h_le = int(np.argmax([float(np.mean(dq_le_tr[:, h * d_h_le : (h + 1) * d_h_le] ** 2)) for h in range(n_q_le)]))
    top_kv_le = top_h_le // g_le
    Wq_le_top = Wq_le_all[top_h_le * d_h_le : (top_h_le + 1) * d_h_le].copy()
    bq_le_top = bq_le_all[top_h_le * d_h_le : (top_h_le + 1) * d_h_le].copy() if bq_le_all is not None else None
    Wk_le_top = Wk_le_all[top_kv_le * d_h_le : (top_kv_le + 1) * d_h_le].copy()
    bk_le_top = bk_le_all[top_kv_le * d_h_le : (top_kv_le + 1) * d_h_le].copy() if bk_le_all is not None else None
    del Wq_le_all, Wk_le_all, dq_le_tr

    def _project_seeds(suf_list: Sequence[np.ndarray]) -> List[np.ndarray]:
        return [(bf16_to_f64_np(arr) @ V).astype(np.float32) for arr in suf_list]

    seeds_tr_L = _project_seeds(don_tr["L"]["h_in_l0_suf"])
    seeds_tr_R = _project_seeds(don_tr["R"]["h_in_l0_suf"])
    seeds_te_L = _project_seeds(don_te["L"]["h_in_l0_suf"])
    seeds_te_R = _project_seeds(don_te["R"]["h_in_l0_suf"])

    # Precompute donor orthogonal-complement targets in compact lossless BF16 uint16!
    # CRITICAL P0-1 FIX: At `(li == le, tag == "post")` ONLY (L20 has no attention organ!),
    # numerator is `h_in_A[20]` while denominator retains `rms(h_mid_A[20])`!
    # CRITICAL P0-3 FIX: `pos0` sink target averages PER-TOKEN normalized complements!
    eps_A = float(getattr(model_A.config, "rms_norm_eps", 1e-6))
    don_u_perp_fit_u16: Dict[int, Dict[str, np.ndarray]] = {li: {} for li in (l0, l1, le)}
    don_u_perp_val_u16: Dict[int, Dict[str, np.ndarray]] = {li: {} for li in (l0, l1, le)}
    don_u_perp_p0_mean: Dict[int, Dict[str, np.ndarray]] = {li: {} for li in (l0, l1, le)}
    don_rms_vec = don_gen_fit["rms_vec"]
    gamma_map = {
        l0: {"in": real_units[l0]["gamma_in"], "post": real_units[l0]["gamma_post"]},
        l1: {"in": real_units[l1]["gamma_in"], "post": real_units[l1]["gamma_post"]},
        le: {"in": gamma_in_le_A, "post": gamma_post_le_A},
    }

    def _calc_u_perp_u16(h_num_u16: np.ndarray, h_den_u16: np.ndarray, u_u16: np.ndarray, gamma: np.ndarray, return_mean_f64: bool = False):
        n = len(u_u16)
        chunks_u16 = []
        s_f64 = np.zeros(u_u16.shape[1], dtype=np.float64)
        for i0 in range(0, n, 2048):
            i1 = min(i0 + 2048, n)
            Hn = bf16_to_f64_np(h_num_u16[i0:i1])
            Hd = bf16_to_f64_np(h_den_u16[i0:i1])
            U = bf16_to_f64_np(u_u16[i0:i1])
            rms = np.sqrt(np.mean(Hd ** 2, axis=-1, keepdims=True) + eps_A)
            proj_V = ((Hn / rms) @ V) @ V.T
            up = U - proj_V * gamma[None, :]
            if return_mean_f64:
                s_f64 += up.sum(axis=0)
            else:
                chunks_u16.append(torch.from_numpy(up.astype(np.float32)).to(torch.bfloat16).contiguous().view(torch.uint16).numpy())
        if return_mean_f64:
            return s_f64 / max(float(n), 1.0)
        return np.ascontiguousarray(np.concatenate(chunks_u16, axis=0))

    for li in (l0, l1, le):
        for tag, h_den_k, u_k in (("in", "h_in", "u_in"), ("post", "h_mid", "u_post")):
            h_num_k = "h_in" if (li == le and tag == "post") else h_den_k
            g_v = gamma_map[li][tag]
            don_u_perp_fit_u16[li][tag] = _calc_u_perp_u16(don_gen_fit["gt0"][li][h_num_k], don_gen_fit["gt0"][li][h_den_k], don_gen_fit["gt0"][li][u_k], g_v)
            don_u_perp_val_u16[li][tag] = _calc_u_perp_u16(don_gen_val["gt0"][li][h_num_k], don_gen_val["gt0"][li][h_den_k], don_gen_val["gt0"][li][u_k], g_v)
            don_u_perp_p0_mean[li][tag] = _calc_u_perp_u16(don_gen_fit["pos0"][li][h_num_k], don_gen_fit["pos0"][li][h_den_k], don_gen_fit["pos0"][li][u_k], g_v, return_mean_f64=True)

    don_lane_targets_tr: Dict[int, Dict[str, np.ndarray]] = {}
    for li in (l0, l1):
        don_lane_targets_tr[li] = {
            "attn": (bf16_to_f64_np(don_tr["L"]["y_attn_last"][li]) - bf16_to_f64_np(don_tr["R"]["y_attn_last"][li])) @ V,
            "mlp": (bf16_to_f64_np(don_tr["L"]["y_mlp_last"][li]) - bf16_to_f64_np(don_tr["R"]["y_mlp_last"][li])) @ V,
        }

    don_te_u_in_le_L = don_te["L"]["u_in_le_suf"]
    don_te_u_in_le_R = don_te["R"]["u_in_le_suf"]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram_telemetry["phase1_donor_peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / (1024.0 ** 3), 3)
        vram_telemetry["phase1_donor_peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / (1024.0 ** 3), 3)

    unload_model_vram_safe(model_A)
    del model_A, layers_A, don_gen_fit, don_gen_val, don_tr, don_te
    gc.collect()
    if torch.cuda.is_available():
        post_u = torch.cuda.memory_allocated() / (1024.0 ** 3)
        vram_telemetry["phase1_post_unload_gib"] = round(post_u, 3)
        print(f"  [Phase 1 Complete in {time.time() - t_p1:.1f}s] Donor 7B 100% UNLOADED! (peak_alloc={vram_telemetry['phase1_donor_peak_alloc_gib']:.2f} GiB, post_unload={post_u:.2f} GiB)")
        assert post_u < 1.0, f"Donor 7B failed to unload cleanly: {post_u:.2f} GiB"
        torch.cuda.reset_peak_memory_stats()

    # -------------------------------------------------------------------------
    # PHASE 2: SOLO RECIPIENT 1.5B — 4 ARMS + 5 PREDECLARED METRICS + ROLLBACK!
    # -------------------------------------------------------------------------
    t_p2 = time.time()
    model_B, tok_B, dev_B, _ = load_model_bf16(args.recipient_model, allow_download=args.allow_download)
    for p in model_B.parameters():
        p.requires_grad_(False)
    assert tok_A.get_vocab() == tok_B.get_vocab()
    V_shared = min(len(tok_B), int(model_B.lm_head.out_features))
    D_B = int(model_B.config.hidden_size)

    # 100% ZERO-HOOK baseline 380-fact locality, inertness reference, and rollback probe!
    base_loc, base_loc_ok = eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, batch_size=32, v_shared=V_shared)
    print(f"  [Base 1.5B 380-Fact Locality] {base_loc['n_correct']}/{base_loc['n_total']} ({base_loc['acc']:.3f})")
    inert_passages = cal["val_report"][:24]
    base_inert_logits, base_rms_l20 = capture_inertness_reference_no_hooks(model_B, tok_B, inert_passages, V_shared, dev_B, eval_layer=le)

    probe_enc = tok_B(inert_passages[:4], return_tensors="pt", padding=True, truncation=True, max_length=64)
    probe_ids, probe_mask = probe_enc["input_ids"].to(dev_B), probe_enc["attention_mask"].to(dev_B)
    with torch.no_grad():
        base_probe_logits = model_B(input_ids=probe_ids, attention_mask=probe_mask, use_cache=False).logits.detach().cpu()

    # Exact paired per-token estimator `kappa_l = mean(r_B / r_A)` on identical pos > 0 tokens!
    rec_gen_base_fit = capture_generic_pos0_and_gt0(model_B, tok_B, cal["fit"], [l0, l1, le], dev_B, batch_size=16, max_length=160, max_gt0_tokens=args.n_bg_tokens, capture_pos0_tokens=False)
    highway_kappas: Dict[int, Dict[str, float]] = {}
    for li in (l0, l1, le):
        highway_kappas[li] = {}
        for tag in ("in", "post"):
            rA = don_rms_vec[li][tag]
            rB = rec_gen_base_fit["rms_vec"][li][tag]
            n_pair = min(len(rA), len(rB))
            assert n_pair >= min(args.n_bg_tokens, 1000)
            highway_kappas[li][tag] = float(np.mean(rB[:n_pair] / np.maximum(rA[:n_pair], 1e-12)))
    del rec_gen_base_fit, don_rms_vec
    gc.collect()
    print(f"  [RMSNorm Gauge paired kappas mean(r_B/r_A)] { {f'L{k}': {kk: round(vv, 4) for kk, vv in d.items()} for k, d in highway_kappas.items()} }")

    def _fit_p_perp_current(li: int, tag: str, cur_gen_fit: Dict[str, Any], cur_gen_val: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        u_k = "u_in" if tag == "in" else "u_post"
        S_xx, S_yx = _stream_second_moments_u16(cur_gen_fit["gt0"][li][u_k], don_u_perp_fit_u16[li][tag], x_cols=D_B)
        Xb_v = bf16_to_f64_np(cur_gen_val["gt0"][li][u_k])[:, :D_B]
        Ya_v = bf16_to_f64_np(don_u_perp_val_u16[li][tag])
        n_v = min(len(Xb_v), len(Ya_v))
        u_b0 = bf16_to_f64_np(cur_gen_fit["pos0"][li][u_k])[:, :D_B].mean(axis=0)
        u_a0 = don_u_perp_p0_mean[li][tag]
        return fit_dirnorm_ridge_with_pos0_sink(S_xx, S_yx, Xb_v[:n_v], Ya_v[:n_v], u_x0=u_b0, u_y0=u_a0)

    rec_tr_prompts = [it["recipient_prompt"] for it in ds2["train"]]
    rec_te_prompts = [it["recipient_prompt"] for it in ds2["test"]]
    test_query_groups = [it["query"] for it in ds2["test"]]

    def _build_organs_pass(
        arm_name: str,
        units_src: Dict[int, Dict[str, Any]],
        fit_on_inner_only: bool = False,
        fixed_cfgs: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Tuple[float, float]]]:
        attn_organs: Dict[int, Dict[str, Any]] = {}
        mlp_organs: Dict[int, Dict[str, Any]] = {}
        p_perp_diag: Dict[str, Any] = {}
        refit_diag: Dict[str, Any] = {}
        chosen_cfgs: Dict[str, Tuple[float, float]] = {}

        for li in (l0, l1):
            u = units_src[li]
            cur_pk_pre = ReversiblePrivateLanePack(K, alpha, highway_kappas, attn_organs, mlp_organs)
            try:
                cur_pk_pre.apply(model_B)
                gf = capture_generic_pos0_and_gt0(model_B, tok_B, cal["fit"], [li], dev_B, batch_size=16, max_length=160, max_gt0_tokens=args.n_bg_tokens)
                gv = capture_generic_pos0_and_gt0(model_B, tok_B, cal["val_tune"], [li], dev_B, batch_size=16, max_length=160, max_gt0_tokens=2048, capture_pos0_tokens=False)
            finally:
                cur_pk_pre.revert(model_B)
            P_a, d_pa = _fit_p_perp_current(li, "in", gf, gv)
            del gf, gv
            p_perp_diag[f"L{li}_attn"] = d_pa
            R_a = np.hstack([P_a, u["gamma_in"][:, None] * V])
            Wq_new = torch.from_numpy((u["q_w"].astype(np.float64) @ R_a).astype(np.float32))
            Wk_new = torch.from_numpy((u["k_w"].astype(np.float64) @ R_a).astype(np.float32))
            Wv_new = torch.from_numpy((u["v_w"].astype(np.float64) @ R_a).astype(np.float32))
            bq_new = torch.from_numpy(u["q_b"].astype(np.float32)) if u["q_b"] is not None else None
            bk_new = torch.from_numpy(u["k_b"].astype(np.float32)) if u["k_b"] is not None else None
            bv_new = torch.from_numpy(u["v_b"].astype(np.float32)) if u["v_b"] is not None else None
            d_q_add = int(Wq_new.shape[0])

            if arm_name == "REAL_COPIED":
                Wo_priv = alpha * (V.T @ u["o_w"].astype(np.float64))
            else:
                probe_a = {
                    "q_w": Wq_new, "q_b": bq_new, "k_w": Wk_new, "k_b": bk_new, "v_w": Wv_new, "v_b": bv_new,
                    "o_w": torch.zeros((D_B + K, d_q_add), dtype=torch.float32),
                    "add_q_heads": u["add_q_heads"], "add_kv_heads": u["add_kv_heads"],
                }
                pk_pa = ReversiblePrivateLanePack(K, alpha, highway_kappas, {**attn_organs, li: probe_a}, mlp_organs)
                try:
                    pk_pa.apply(model_B)
                    zL = run_recipient_seeded_l18_to_l20(model_B, tok_B, rec_tr_prompts, seeds_tr_L, alpha, dev_B, l0, l1, le, capture_organ_z_layer=li, z_attn_tail=d_q_add)["z_attn_tail"]
                    zR = run_recipient_seeded_l18_to_l20(model_B, tok_B, rec_tr_prompts, seeds_tr_R, alpha, dev_B, l0, l1, le, capture_organ_z_layer=li, z_attn_tail=d_q_add)["z_attn_tail"]
                    z_bg_list = []
                    hd_bg: Any = None
                    cap_bg: Dict[str, torch.Tensor] = {}
                    old_p = tok_B.padding_side
                    tok_B.padding_side = "left"
                    try:
                        hd_bg = get_decoder_layers(model_B)[li].self_attn.o_proj.register_forward_pre_hook(lambda _m, a: cap_bg.__setitem__("z", a[0][..., -d_q_add:].detach()))
                        for j0 in range(0, min(40, len(cal["fit"])), 8):
                            enc = tok_B(cal["fit"][j0 : j0 + 8], return_tensors="pt", padding=True, truncation=True, max_length=128, add_special_tokens=True)
                            ids, msk = enc["input_ids"].to(dev_B), enc["attention_mask"].to(dev_B)
                            _ = model_B.model(input_ids=ids, attention_mask=msk, use_cache=False)
                            _, gt0 = get_pos0_and_gt0_masks(msk)
                            z_bg_list.append(cap_bg.pop("z")[gt0].float().cpu().numpy())
                    finally:
                        if hd_bg is not None:
                            hd_bg.remove()
                        tok_B.padding_side = old_p
                finally:
                    pk_pa.revert(model_B)
                dZ_a = bf16_to_f64_np(zL) - bf16_to_f64_np(zR)
                Z_bg_a = np.concatenate(z_bg_list, axis=0)
                key_a = f"L{li}_attn"
                fcfg_a = fixed_cfgs[key_a] if (fixed_cfgs is not None and key_a in fixed_cfgs) else None
                W_fit_a, rf_a = fit_scale_normalized_ridge_lane_write(
                    dZ_a, don_lane_targets_tr[li]["attn"], Z_bg_a,
                    fixed_cfg=fcfg_a, fit_on_inner_only=fit_on_inner_only,
                )
                Wo_priv = alpha * W_fit_a
                refit_diag[key_a] = rf_a
                chosen_cfgs[key_a] = (float(rf_a["best_lam_bg_rel"]), float(rf_a["best_lam_eye_rel"]))

            Wo_full = np.zeros((D_B + K, d_q_add), dtype=np.float32)
            Wo_full[D_B:, :] = Wo_priv.astype(np.float32)
            attn_organs[li] = {
                "q_w": Wq_new, "q_b": bq_new, "k_w": Wk_new, "k_b": bk_new, "v_w": Wv_new, "v_b": bv_new,
                "o_w": torch.from_numpy(Wo_full), "add_q_heads": u["add_q_heads"], "add_kv_heads": u["add_kv_heads"],
            }

            cur_pk_m = ReversiblePrivateLanePack(K, alpha, highway_kappas, attn_organs, mlp_organs)
            try:
                cur_pk_m.apply(model_B)
                gf_m = capture_generic_pos0_and_gt0(model_B, tok_B, cal["fit"], [li], dev_B, batch_size=16, max_length=160, max_gt0_tokens=args.n_bg_tokens)
                gv_m = capture_generic_pos0_and_gt0(model_B, tok_B, cal["val_tune"], [li], dev_B, batch_size=16, max_length=160, max_gt0_tokens=2048, capture_pos0_tokens=False)
            finally:
                cur_pk_m.revert(model_B)
            P_m, d_pm = _fit_p_perp_current(li, "post", gf_m, gv_m)
            del gf_m, gv_m
            p_perp_diag[f"L{li}_mlp"] = d_pm
            R_m = np.hstack([P_m, u["gamma_post"][:, None] * V])
            Wg_new = torch.from_numpy((u["gate_w"].astype(np.float64) @ R_m).astype(np.float32))
            Wu_new = torch.from_numpy((u["up_w"].astype(np.float64) @ R_m).astype(np.float32))
            d_m_add = int(Wg_new.shape[0])

            if arm_name == "REAL_COPIED":
                Wd_priv = alpha * (V.T @ u["down_w"].astype(np.float64))
            else:
                probe_m = {
                    "gate_w": Wg_new, "up_w": Wu_new,
                    "down_w": torch.zeros((D_B + K, d_m_add), dtype=torch.float32),
                }
                pk_pm = ReversiblePrivateLanePack(K, alpha, highway_kappas, attn_organs, {**mlp_organs, li: probe_m})
                try:
                    pk_pm.apply(model_B)
                    zL = run_recipient_seeded_l18_to_l20(model_B, tok_B, rec_tr_prompts, seeds_tr_L, alpha, dev_B, l0, l1, le, capture_organ_z_layer=li, z_mlp_tail=d_m_add)["z_mlp_tail"]
                    zR = run_recipient_seeded_l18_to_l20(model_B, tok_B, rec_tr_prompts, seeds_tr_R, alpha, dev_B, l0, l1, le, capture_organ_z_layer=li, z_mlp_tail=d_m_add)["z_mlp_tail"]
                    z_bg_list = []
                    hd_bg = None
                    cap_bg = {}
                    old_p = tok_B.padding_side
                    tok_B.padding_side = "left"
                    try:
                        hd_bg = get_decoder_layers(model_B)[li].mlp.down_proj.register_forward_pre_hook(lambda _m, a: cap_bg.__setitem__("z", a[0][..., -d_m_add:].detach()))
                        for j0 in range(0, min(40, len(cal["fit"])), 8):
                            enc = tok_B(cal["fit"][j0 : j0 + 8], return_tensors="pt", padding=True, truncation=True, max_length=128, add_special_tokens=True)
                            ids, msk = enc["input_ids"].to(dev_B), enc["attention_mask"].to(dev_B)
                            _ = model_B.model(input_ids=ids, attention_mask=msk, use_cache=False)
                            _, gt0 = get_pos0_and_gt0_masks(msk)
                            z_bg_list.append(cap_bg.pop("z")[gt0].float().cpu().numpy())
                    finally:
                        if hd_bg is not None:
                            hd_bg.remove()
                        tok_B.padding_side = old_p
                finally:
                    pk_pm.revert(model_B)
                dZ_m = bf16_to_f64_np(zL) - bf16_to_f64_np(zR)
                Z_bg_m = np.concatenate(z_bg_list, axis=0)
                key_m = f"L{li}_mlp"
                fcfg_m = fixed_cfgs[key_m] if (fixed_cfgs is not None and key_m in fixed_cfgs) else None
                W_fit_m, rf_m = fit_scale_normalized_ridge_lane_write(
                    dZ_m, don_lane_targets_tr[li]["mlp"], Z_bg_m,
                    fixed_cfg=fcfg_m, fit_on_inner_only=fit_on_inner_only,
                )
                Wd_priv = alpha * W_fit_m
                refit_diag[key_m] = rf_m
                chosen_cfgs[key_m] = (float(rf_m["best_lam_bg_rel"]), float(rf_m["best_lam_eye_rel"]))

            Wd_full = np.zeros((D_B + K, d_m_add), dtype=np.float32)
            Wd_full[D_B:, :] = Wd_priv.astype(np.float32)
            mlp_organs[li] = {"gate_w": Wg_new, "up_w": Wu_new, "down_w": torch.from_numpy(Wd_full)}

        return attn_organs, mlp_organs, p_perp_diag, refit_diag, chosen_cfgs

    def _run_arm(arm_name: str) -> Dict[str, Any]:
        t_arm = time.time()
        attn_organs: Dict[int, Dict[str, Any]] = {}
        mlp_organs: Dict[int, Dict[str, Any]] = {}
        p_perp_diag: Dict[str, Any] = {}
        refit_diag: Dict[str, Any] = {}

        if arm_name != "NONE":
            units_src = real_units if arm_name in ("REAL_COPIED", "REAL_REFIT") else shuf_units
            if arm_name == "REAL_COPIED":
                attn_organs, mlp_organs, p_perp_diag, refit_diag, _ = _build_organs_pass(arm_name, units_src)
            else:
                # Pass A: strict inner-320 tuning pass (upstream writes NEVER see [320:400]!)
                _, _, _, inner_diag, tuned_cfgs = _build_organs_pass(arm_name, units_src, fit_on_inner_only=True)
                # Pass B: final 400-pair refit using the leak-free tuned hyperparameters!
                attn_organs, mlp_organs, p_perp_diag, final_diag, _ = _build_organs_pass(arm_name, units_src, fixed_cfgs=tuned_cfgs)
                refit_diag = {"inner_320_tune": inner_diag, "final_400_refit": final_diag}

        final_pk = ReversiblePrivateLanePack(K, alpha, highway_kappas, attn_organs, mlp_organs, metadata={"arm": arm_name})
        try:
            final_pk.apply(model_B)
            gf_20 = capture_generic_pos0_and_gt0(model_B, tok_B, cal["fit"], [le], dev_B, batch_size=16, max_length=160, max_gt0_tokens=args.n_bg_tokens)
            gv_20 = capture_generic_pos0_and_gt0(model_B, tok_B, cal["val_tune"], [le], dev_B, batch_size=16, max_length=160, max_gt0_tokens=2048, capture_pos0_tokens=False)
            P_20_a, d_p20a = _fit_p_perp_current(le, "in", gf_20, gv_20)
            P_20_m, d_p20m = _fit_p_perp_current(le, "post", gf_20, gv_20)
            del gf_20, gv_20
            p_perp_diag[f"L{le}_attn"] = d_p20a
            p_perp_diag[f"L{le}_mlp"] = d_p20m

            out_tr_L = run_recipient_seeded_l18_to_l20(model_B, tok_B, rec_tr_prompts, seeds_tr_L, alpha, dev_B, l0, l1, le)
            out_tr_R = run_recipient_seeded_l18_to_l20(model_B, tok_B, rec_tr_prompts, seeds_tr_R, alpha, dev_B, l0, l1, le)
            out_te_L = run_recipient_seeded_l18_to_l20(model_B, tok_B, rec_te_prompts, seeds_te_L, alpha, dev_B, l0, l1, le, capture_le_u_in_suf=True)
            out_te_R = run_recipient_seeded_l18_to_l20(model_B, tok_B, rec_te_prompts, seeds_te_R, alpha, dev_B, l0, l1, le, capture_le_u_in_suf=True)

            # 100% ZERO-HOOK Metric 5 Inertness + Certified 380-Fact Locality!
            inert = measure_generic_inertness_no_hooks(model_B, tok_B, inert_passages, base_inert_logits, base_rms_l20, D_B, K, V_shared, dev_B, eval_layer=le)
            loc_res, _ = eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)
        finally:
            final_pk.revert(model_B)
            rb = final_pk.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=False)

        rb_ok = bool(rb["logits_bit_equal"] and rb["max_abs_diff"] == 0.0 and rb["zero_runtime_hooks"])
        if not rb_ok:
            # Record frozen INERTNESS_FAIL disposition cleanly BEFORE stopping on an unrestored model!
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            fail_payload = {
                "meta": {"experiment": "v4.4.1_private_lane_falsifier", "failed_arm": arm_name, "vram_telemetry": vram_telemetry},
                "decision": {"primary_arm_predeclared": "REAL_REFIT", "verdict": "INERTNESS_FAIL", "reason": f"Bit-exact rollback failed on arm {arm_name}"},
                "failed_rollback": rb,
            }
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump(fail_payload, f, indent=2)
            raise AssertionError(f"INERTNESS_FAIL recorded to {OUT_PATH}: rollback failed on arm {arm_name}: {rb}")

        # Metric 3: Relay Survival `eta` on unseen-key contrast INCREMENTS `delta_Delta z_B`!
        def _priv_delta(oL: Dict[str, Any], oR: Dict[str, Any]) -> np.ndarray:
            dz_l0 = (bf16_to_f64_np(oL["h_in_l0_last"])[:, D_B:] - bf16_to_f64_np(oR["h_in_l0_last"])[:, D_B:]) / alpha
            dz_le = (bf16_to_f64_np(oL["h_in_le_last"])[:, D_B:] - bf16_to_f64_np(oR["h_in_le_last"])[:, D_B:]) / alpha
            return dz_le - dz_l0

        dDelta_z_B_tr = _priv_delta(out_tr_L, out_tr_R)
        dDelta_z_B_te = _priv_delta(out_te_L, out_te_R)
        eta_direct_raw = compute_r2(dDelta_z_A_te, dDelta_z_B_te)

        C_z = (dDelta_z_B_tr.T @ dDelta_z_B_tr) / float(len(dDelta_z_B_tr))
        tr_z = float(np.trace(C_z)) / max(K, 1)
        W_z = sla.solve(C_z + max(1e-2 * tr_z, 1e-6) * np.eye(K), (dDelta_z_B_tr.T @ dDelta_z_A_tr) / float(len(dDelta_z_B_tr)), assume_a="pos")
        eta_readout_raw = compute_r2(dDelta_z_A_te, dDelta_z_B_te @ W_z)

        # Metric 4: L20 Hearing evaluated through NATIVE BF16 SwiGLU projection + SiLU + multiply!
        R_20_m = np.hstack([P_20_m, gamma_post_le_A[:, None] * V])
        Wg_20_B = Wg_le_top64 @ R_20_m
        Wu_20_B = Wu_le_top64 @ R_20_m
        gB_L, uB_L, yB_L = native_bf16_swiglu_eval(out_te_L["u_post_le_last"], Wg_20_B, Wu_20_B, dev_B)
        gB_R, uB_R, yB_R = native_bf16_swiglu_eval(out_te_R["u_post_le_last"], Wg_20_B, Wu_20_B, dev_B)
        dgB_te, duB_te = gB_L - gB_R, uB_L - uB_R
        dyB_swiglu_te = yB_L - yB_R

        if arm_name == "NONE":
            up_tr_L = bf16_to_f64_np(out_tr_L["u_post_le_last"])
            up_tr_R = bf16_to_f64_np(out_tr_R["u_post_le_last"])
            X_loc = np.vstack([up_tr_L, up_tr_R])
            C_loc = (X_loc.T @ X_loc) / float(len(X_loc))
            el = sla.eigvalsh(C_loc)
            lam_loc = max(1e-1 * float(np.median(el[el > 1e-12])), 1e-6)
            W_loc_g = sla.solve(C_loc + lam_loc * np.eye(X_loc.shape[1]), (X_loc.T @ np.vstack([gA_L_tr, gA_R_tr])) / float(len(X_loc)), assume_a="pos").T
            W_loc_u = sla.solve(C_loc + lam_loc * np.eye(X_loc.shape[1]), (X_loc.T @ np.vstack([uA_L_tr, uA_R_tr])) / float(len(X_loc)), assume_a="pos").T
            _, _, yB_L_loc = native_bf16_swiglu_eval(out_te_L["u_post_le_last"], W_loc_g, W_loc_u, dev_B)
            _, _, yB_R_loc = native_bf16_swiglu_eval(out_te_R["u_post_le_last"], W_loc_g, W_loc_u, dev_B)
            dyB_local = yB_L_loc - yB_R_loc
            none_local_r2_raw = compute_r2(dyA_swiglu_te, dyB_local)
        else:
            dyB_local = None
            none_local_r2_raw = None

        cos_g = np.sum(dgB_te * dgA_te, axis=0) / (np.linalg.norm(dgB_te, axis=0) * np.linalg.norm(dgA_te, axis=0) + 1e-12)
        cos_u = np.sum(duB_te * duA_te, axis=0) / (np.linalg.norm(duB_te, axis=0) * np.linalg.norm(duA_te, axis=0) + 1e-12)
        med_cos_gate = float(np.median(cos_g))
        med_cos_up = float(np.median(cos_u))
        med_cos_both_raw = float(np.median(np.concatenate([cos_g, cos_u])))
        swiglu_r2_raw = compute_r2(dyA_swiglu_te, dyB_swiglu_te)

        rmse_L_gt0, rmse_L_all = compute_l20_rope_attn_logit_rmse(don_te_u_in_le_L, out_te_L["u_in_le_suf"], Wq_le_top, bq_le_top, Wk_le_top, bk_le_top, P_20_a, gamma_in_le_A, V, rope_theta)
        rmse_R_gt0, rmse_R_all = compute_l20_rope_attn_logit_rmse(don_te_u_in_le_R, out_te_R["u_in_le_suf"], Wq_le_top, bq_le_top, Wk_le_top, bk_le_top, P_20_a, gamma_in_le_A, V, rope_theta)
        attn_rmse_gt0_raw = float(math.sqrt(0.5 * (rmse_L_gt0 ** 2 + rmse_R_gt0 ** 2)))
        attn_rmse_all_raw = float(math.sqrt(0.5 * (rmse_L_all ** 2 + rmse_R_all ** 2)))
        ret_raw = float((loc_res["n_base_correct"] - loc_res["newly_broken"]) / max(loc_res["n_base_correct"], 1))

        print(
            f"  [Arm {arm_name:11s}] eta_direct={eta_direct_raw:.4f} (readout={eta_readout_raw:.4f}) | "
            f"L20 med_cos={med_cos_both_raw:.4f} (gate={med_cos_gate:.4f}, up={med_cos_up:.4f}) | "
            f"NativeBF16 SwiGLU R2_Delta={swiglu_r2_raw:.4f} | Attn RMSE(pos>0)={attn_rmse_gt0_raw:.3f} nat | "
            f"rho={inert['_rho_raw']:.2e} KL={inert['_kl_raw']:.5f} loc={loc_res['n_correct']}/380 (ret={ret_raw:.3f}) | rb={rb['logits_bit_equal']} ({time.time()-t_arm:.1f}s)"
        )
        return {
            "eta_direct_r2": round(eta_direct_raw, 4),
            "eta_readout_r2": round(eta_readout_raw, 4),
            "l20_hearing": {
                "median_cos_gate": round(med_cos_gate, 4),
                "median_cos_up": round(med_cos_up, 4),
                "median_cos_gate_and_up": round(med_cos_both_raw, 4),
                "native_bf16_swiglu_product_r2_delta": round(swiglu_r2_raw, 4),
                "none_task_local_reader_swiglu_r2_delta": round(none_local_r2_raw, 4) if none_local_r2_raw is not None else None,
                "attn_logit_rmse_pos_gt0_nat": round(attn_rmse_gt0_raw, 4),
                "attn_logit_rmse_all_pos_nat": round(attn_rmse_all_raw, 4),
            },
            "inertness": {k: v for k, v in inert.items() if not k.startswith("_")},
            "locality_380": loc_res,
            "rollback": rb,
            "p_perp_diag": p_perp_diag,
            "refit_diag": refit_diag,
            "_raw": {
                "eta": eta_direct_raw,
                "med_cos": med_cos_both_raw,
                "attn_rmse": attn_rmse_gt0_raw,
                "rho": inert["_rho_raw"],
                "kl": inert["_kl_raw"],
                "retention": ret_raw,
                "rb_ok": rb_ok,
            },
            "_dyB_swiglu_te": dyB_swiglu_te,
            "_dyB_local_te": dyB_local,
        }

    arm_order = ("NONE", "REAL_COPIED", "REAL_REFIT", "SHUF")
    raw_arms = {name: _run_arm(name) for name in arm_order}

    boot_checks_raw: Dict[str, Any] = {}
    for real_name in ("REAL_COPIED", "REAL_REFIT"):
        dy_r = raw_arms[real_name]["_dyB_swiglu_te"]
        boot_checks_raw[f"{real_name}_vs_NONE_pulled"] = paired_key_group_bootstrap_ci(dyA_swiglu_te, dy_r, raw_arms["NONE"]["_dyB_swiglu_te"], test_query_groups)
        boot_checks_raw[f"{real_name}_vs_NONE_local_reader"] = paired_key_group_bootstrap_ci(dyA_swiglu_te, dy_r, raw_arms["NONE"]["_dyB_local_te"], test_query_groups)
        boot_checks_raw[f"{real_name}_vs_SHUF"] = paired_key_group_bootstrap_ci(dyA_swiglu_te, dy_r, raw_arms["SHUF"]["_dyB_swiglu_te"], test_query_groups)

    # Evaluate predeclared GO/KILL independently on BOTH REAL arms using FULL-PRECISION values (NO test-set cherry-picking!)
    sh_eta = raw_arms["SHUF"]["_raw"]["eta"]
    m1_go, m1_kill = (metric1_c_raw >= 0.80), (metric1_c_raw < 0.60)
    m2_go, m2_kill = (beta_refit_raw >= 0.70), (beta_refit_raw < 0.50)

    def _score_real_arm(rname: str) -> Dict[str, Any]:
        rr = raw_arms[rname]["_raw"]
        m3_go = (rr["eta"] >= 0.50) and (rr["eta"] >= 0.70 * beta_refit_raw) and (sh_eta <= 0.10)
        m3_kill = (rr["eta"] < 0.25) or ((rr["eta"] - sh_eta) < 0.15)
        m4_go = (
            rr["med_cos"] >= 0.80
            and rr["attn_rmse"] <= 1.0
            and boot_checks_raw[f"{rname}_vs_NONE_pulled"]["_ci95_low_raw"] > 0.0
            and boot_checks_raw[f"{rname}_vs_NONE_local_reader"]["_ci95_low_raw"] > 0.0
            and boot_checks_raw[f"{rname}_vs_SHUF"]["_ci95_low_raw"] > 0.0
        )
        m4_kill = rr["med_cos"] < 0.50
        m5_go = (rr["rho"] <= 1e-3) and (rr["kl"] <= 0.01) and (rr["retention"] >= 0.97) and bool(rr["rb_ok"])
        if m1_kill or m2_kill or m3_kill or m4_kill:
            v = "KILL"
        elif m1_go and m2_go and m3_go and m4_go:
            v = "GO" if m5_go else "INERTNESS_FAIL"
        else:
            v = "BORDERLINE"
        return {
            "verdict": v,
            "flags": {
                "m1_go": m1_go, "m1_kill": m1_kill, "m2_go": m2_go, "m2_kill": m2_kill,
                "m3_go": m3_go, "m3_kill": m3_kill, "m4_go": m4_go, "m4_kill": m4_kill, "m5_go": m5_go,
            },
        }

    arm_verdicts = {rname: _score_real_arm(rname) for rname in ("REAL_COPIED", "REAL_REFIT")}
    primary_arm = "REAL_REFIT"  # Predeclared primary arm (zero test-set selection bias!)
    verdict = arm_verdicts[primary_arm]["verdict"]

    arms_clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in raw_arms.items()}
    boot_checks_clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in boot_checks_raw.items()}

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram_telemetry["phase2_recipient_peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / (1024.0 ** 3), 3)
        vram_telemetry["phase2_recipient_peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / (1024.0 ** 3), 3)

    results = {
        "meta": {
            "experiment": "v4.4.1_private_lane_falsifier",
            "donor_model": args.donor_model,
            "recipient_model": args.recipient_model,
            "layers": {"l0": l0, "l1": l1, "le": le, "shuf_layers": [shuf_l0, shuf_l1]},
            "K_private_lanes": K,
            "alpha_gauge": alpha,
            "n_fit_pairs": args.n_fit_pairs,
            "n_test_pairs": args.n_test_pairs,
            "n_bg_tokens_pos_gt0": args.n_bg_tokens,
            "vram_telemetry": vram_telemetry,
            "total_elapsed_sec": round(time.time() - t_all, 1),
        },
        "metric1_interface_capture_c": {
            "whitened_unseen_delta_Delta_capture_c": round(metric1_c_raw, 4),
            "whitened_unseen_endpoint_L20_capture_c": round(metric1_c_endpoint, 4),
            "fit_geig_captured_frac": basis["geig_captured_frac_fit"],
            "top10_geigs": [round(x, 4) for x in basis["top_geigs"]],
            "go": m1_go, "kill": m1_kill,
        },
        "metric2_organ_compute_budget_beta": {
            "donor_refit_r2": round(beta_refit_raw, 4),
            "donor_copied_r2": round(beta_copied_raw, 4),
            "go": m2_go, "kill": m2_kill,
        },
        "arms": arms_clean,
        "paired_key_group_bootstrap_95ci": boot_checks_clean,
        "base_locality_380": base_loc,
        "decision": {
            "primary_arm_predeclared": primary_arm,
            "verdict": verdict,
            "per_arm_verdicts": arm_verdicts,
        },
    }

    unload_model_vram_safe(model_B)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[V4.4.1 FALSIFIER COMPLETE in {results['meta']['total_elapsed_sec']}s] PRIMARY({primary_arm})={verdict} | REAL_COPIED={arm_verdicts['REAL_COPIED']['verdict']} | Saved {OUT_PATH}")
    return results


# ======================================================================================
# 9. Pure-CPU Unit Tests (<3s, ZERO GPU!)
# ======================================================================================

def run_cpu_unit_tests() -> None:
    t0 = time.time()
    from transformers import AutoTokenizer
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    print("=== Running v4_4_private_lane_falsifier.py Pure-CPU Unit Tests (v4.4.1) ===")
    rng = np.random.default_rng(2026)

    # 1. Pos-0 masking + IN-LOOP KKT pos-0 sink constraint verified on DISJOINT val rows!
    left_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    p0, gt0 = get_pos0_and_gt0_masks(left_mask)
    assert p0[0].tolist() == [False, False, True, False, False] and p0[1].tolist() == [False, True, False, False, False]
    assert int(gt0.sum().item()) == 5
    X_all = rng.standard_normal((320, 16))
    W_true = rng.standard_normal((24, 16)) * 0.3
    Y_all = X_all @ W_true.T + 0.02 * rng.standard_normal((320, 24))
    X_tr, Y_tr = X_all[:240], Y_all[:240]
    X_va, Y_va = X_all[240:], Y_all[240:]
    u_x0 = rng.standard_normal(16) * 5.0
    u_y0 = rng.standard_normal(24) * 10.0
    W_hat, info_w = fit_dirnorm_ridge_with_pos0_sink((X_tr.T @ X_tr) / 240.0, (Y_tr.T @ X_tr) / 240.0, X_va, Y_va, u_x0=u_x0, u_y0=u_y0)
    recomp_r2 = compute_r2(Y_va, X_va @ W_hat.T)
    assert float(np.linalg.norm(W_hat @ u_x0 - u_y0)) < 1e-10
    assert abs(info_w["val_r2_pos_gt0"] - round(recomp_r2, 4)) <= 1e-4, f"Reported {info_w['val_r2_pos_gt0']} != recomputed {recomp_r2}"
    print(f"  [PASS 1/6] Pos-0 masking + in-loop KKT pos-0 sink constraint (err={info_w['pos0_sink_rel_err']:.2e}, val_r2={recomp_r2:.4f} exact!)")

    # 2. Streamed primal generalized-eigenvector interface basis V & generic-whitened capture c=1.0000!
    bg = rng.standard_normal((500, 32)).astype(np.float32)
    bg_u16 = torch.from_numpy(bg).to(torch.bfloat16).view(torch.uint16).numpy()
    con_low = rng.standard_normal((100, 4))
    V_true = np.linalg.qr(rng.standard_normal((32, 4)))[0]
    con = con_low @ V_true.T * 5.0
    bas = fit_generalized_interface_basis(con, [bg_u16], K=4)
    assert np.allclose(bas["V"].T @ bas["V"], np.eye(4), atol=1e-10)
    c_val = compute_whitened_interface_capture(con, bas["V"], bas["Q_bg"], bas["inv_sqrt_bg"])
    assert c_val > 0.99, f"Expected capture ~1.0, got {c_val}"
    print(f"  [PASS 2/6] Streamed primal generalized basis V (V^T V = I) & whitened capture c={c_val:.4f}")

    # 3. Native ReversiblePrivateLanePack (TIED emb, day-0 KL ~ 0, rollback == True) + 100% ZERO-HOOK prefix exactness!
    tok = AutoTokenizer.from_pretrained(str(HF_CACHE_DIR / "qwen2.5-3b"), local_files_only=True)
    V_tok = len(tok)
    cfg_B = Qwen2Config(vocab_size=V_tok + 16, hidden_size=48, intermediate_size=96, num_hidden_layers=4, num_attention_heads=6, num_key_value_heads=1, rope_theta=1e6, tie_word_embeddings=True)
    torch.manual_seed(7)
    mB = Qwen2ForCausalLM(cfg_B).to(dtype=torch.bfloat16).eval()
    old_p = tok.padding_side
    tok.padding_side = "left"
    enc = tok(["Table:\na -> b\nQuery: a\nOutput:", "Case 1: ocean currents flow north steadily."], return_tensors="pt", padding=True)
    tok.padding_side = old_p
    ids, msk = enc["input_ids"], enc["attention_mask"]
    with torch.no_grad():
        base_lg = mB(input_ids=ids, attention_mask=msk, use_cache=False).logits.clone()

    # Verify hook-free decoder prefix matches full model layer-2 input BIT-EXACTLY on left-padded batch!
    h_pref = forward_qwen2_prefix_no_hooks(mB, ids, msk, stop_layer=2)
    cap_ref: Dict[str, torch.Tensor] = {}
    hd_chk = mB.model.layers[2].register_forward_pre_hook(lambda _m, a: cap_ref.__setitem__("h", a[0].detach()))
    try:
        with torch.no_grad():
            _ = mB.model(input_ids=ids, attention_mask=msk, use_cache=False)
    finally:
        hd_chk.remove()
    max_pref_diff = float((h_pref.float() - cap_ref["h"].float()).abs().max().item())
    assert max_pref_diff < 1e-4, f"Zero-hook prefix mismatch vs model.model: {max_pref_diff}"

    K_t, alpha_t = 16, 0.1
    pk = ReversiblePrivateLanePack(K=K_t, alpha=alpha_t, highway_kappas={1: {"in": 1.0, "post": 1.0}, 2: {"in": 1.0, "post": 1.0}})
    pk.apply(mB)
    assert mB.lm_head.weight.data_ptr() == mB.model.embed_tokens.weight.data_ptr()
    assert_zero_runtime_hooks(mB)
    with torch.no_grad():
        wid_lg = mB(input_ids=ids, attention_mask=msk, use_cache=False).logits.clone()
    kl_day0 = float(F.kl_div(F.log_softmax(wid_lg[:, -1, :V_tok].float(), dim=-1), F.softmax(base_lg[:, -1, :V_tok].float(), dim=-1), reduction="batchmean").item())
    assert kl_day0 < 1e-3
    pk.revert(mB)
    pk.verify_bit_exact_rollback(mB, ids, msk, base_lg, strict_assert=True)
    print(f"  [PASS 3/6] ReversiblePrivateLanePack (TIED emb, KL={kl_day0:.2e}, rollback==True) & 0-hook prefix diff={max_pref_diff:.2e}!")

    # 4. Free RMSNorm Gauge Trick exact alpha-cancellation!
    z_vec = torch.randn((1, 3, K_t), dtype=torch.float32) * 0.02
    pk.apply(mB)
    cap_u: Dict[str, torch.Tensor] = {}
    h_in_c: Dict[str, torch.Tensor] = {}
    h_p = mB.model.layers[1].register_forward_pre_hook(lambda _m, a: (torch.cat([a[0][..., :48], (alpha_t * z_vec).to(a[0].dtype)], dim=-1),) + a[1:])
    h_o = mB.model.layers[1].input_layernorm.register_forward_hook(lambda _m, _i, o: cap_u.__setitem__("u", o.detach()))
    h_i = mB.model.layers[1].input_layernorm.register_forward_pre_hook(lambda _m, a: h_in_c.__setitem__("h", a[0].detach()))
    try:
        _ = mB.model(input_ids=ids[:1, :3], attention_mask=msk[:1, :3], use_cache=False)
    finally:
        h_p.remove(); h_o.remove(); h_i.remove()
        pk.revert(mB)
    h_orig_part = h_in_c["h"][..., :48].float()
    rms_orig = torch.sqrt(h_orig_part.pow(2).mean(dim=-1, keepdim=True) + float(cfg_B.rms_norm_eps))
    rel_err = float((cap_u["u"][..., 48:].float() - (z_vec / rms_orig)).norm() / ((z_vec / rms_orig).norm() + 1e-12))
    assert rel_err < 0.05
    print(f"  [PASS 4/6] Free RMSNorm Gauge Trick exact alpha-cancellation verified (rel_err={rel_err:.4f})")

    # 5. Scale-normalized ridge lane write solver (inner val split!) & native BF16 SwiGLU eval!
    Z_s = rng.standard_normal((120, 16)) * 0.05  # tiny ~0.05 activations (would be crushed by unscaled lam_eye=1.0!)
    W_s_true = rng.standard_normal((8, 16))
    Y_s = Z_s @ W_s_true.T + 1e-3 * rng.standard_normal((120, 8))
    W_s_hat, rf_i = fit_scale_normalized_ridge_lane_write(Z_s, Y_s, rng.standard_normal((200, 16)) * 0.05, n_inner_train=90)
    assert compute_r2(Y_s, Z_s @ W_s_hat.T) > 0.90, f"Scale-normalized lane write over-shrunk: {rf_i}"
    u_u16 = torch.randn((4, 48), dtype=torch.bfloat16).view(torch.uint16).numpy()
    g_bf, u_bf, y_bf = native_bf16_swiglu_eval(u_u16, rng.standard_normal((8, 48)) * 0.1, rng.standard_normal((8, 48)) * 0.1, torch.device("cpu"))
    assert g_bf.shape == (4, 8) and y_bf.shape == (4, 8) and np.all(np.isfinite(y_bf))
    print(f"  [PASS 5/6] Scale-normalized ridge lane write (inner_val_r2={rf_i['inner_val_r2']}) & native BF16 SwiGLU verified!")

    # 6. Paired key-group bootstrap + pos>0 RoPE attention-logit RMSE!
    yt = rng.standard_normal((80, 16))
    ya = yt + 0.1 * rng.standard_normal((80, 16))
    yb = rng.standard_normal((80, 16))
    ci = paired_key_group_bootstrap_ci(yt, ya, yb, [f"k{i % 10}" for i in range(80)], n_boot=200)
    assert ci["_ci95_low_raw"] > 0.5
    u_same = [torch.randn((6, 48), dtype=torch.bfloat16).view(torch.uint16).numpy()]
    u_wid = [torch.cat([torch.from_numpy(u_same[0]).view(torch.bfloat16), torch.zeros((6, 16), dtype=torch.bfloat16)], dim=-1).view(torch.uint16).numpy()]
    r_gt0, r_all = compute_l20_rope_attn_logit_rmse(u_same, u_wid, rng.standard_normal((8, 48)), None, rng.standard_normal((8, 48)), None, np.eye(48), np.ones(48), np.zeros((48, 16)), 1e6)
    assert r_gt0 < 1e-6 and r_all < 1e-6
    print(f"  [PASS 6/6] Paired key-group bootstrap (CI95=[{ci['ci95_low']}, {ci['ci95_high']}]) & pos>0 RoPE RMSE={r_gt0:.2e} verified!")
    print(f"=== ALL 6 V4.4.1 PRIVATE-LANE FALSIFIER CPU UNIT TESTS PASSED ({time.time() - t0:.2f}s)! ===")


def main() -> None:
    ap = argparse.ArgumentParser(description="v4.4.1 5-Minute Go/No-Go Private-Lane Falsifier")
    ap.add_argument("--unit-test", action="store_true")
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--donor-model", type=str, default=DONOR_KEY_DEFAULT)
    ap.add_argument("--recipient-model", type=str, default=RECIPIENT_KEY_DEFAULT)
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--k-lanes", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--l0", type=int, default=18)
    ap.add_argument("--l1", type=int, default=19)
    ap.add_argument("--le", type=int, default=20)
    ap.add_argument("--n-fit-pairs", type=int, default=400)
    ap.add_argument("--n-test-pairs", type=int, default=400)
    ap.add_argument("--n-bg-tokens", type=int, default=20000)
    ap.add_argument("--add-mlp-rows", type=int, default=256)
    args = ap.parse_args()

    if args.unit_test:
        run_cpu_unit_tests()
        return
    if args.run_all:
        run_private_lane_falsifier(args)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
