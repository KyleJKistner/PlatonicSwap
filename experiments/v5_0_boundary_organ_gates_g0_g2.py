#!/usr/bin/env python3
"""
v5_0_boundary_organ_gates_g0_g2.py — Stage 1: Boundary-First Causal Platonic Organ Gates G0 -> G1 -> G2
========================================================================================================

Strict ZERO-SGD, ZERO-task-target-fitted-writer diagnostic falsifier (`Qwen2.5-7B-Instruct` -> `Qwen2.5-1.5B-Instruct`)
synthesizing Astra 6.2 Ultra's and Fable 5.2 Max's unanimous v5.0 boundary-first theory, incorporating
100% of Pre-GPU 2-Model Grading Council P0/P1 requirements:

  - Helper contracts verified (`LARGE_TO_SMALL_MODELS` str, 4-tuple `load_model_bf16`, 2-tuple
    `fetch_or_fallback_airports` & `format_airport_prompt`, `--allow-download` CLI flag).
  - ZERO eval-set selection leakage: ALL G1 `(best_q, best_r*)` selected strictly on `disc`;
    ALL G2 `best_r` frozen strictly from generic `val_tune` (`kappa_r_tune >= 0.85` & `min_rho2_tune >= 0.80`).
  - Unified certified inlet/outlet path between G1 and G2, with donor self-test replayed on G2's
    recipient-suppliable `D_r` basis (using compact saved donor organ weights + `E_A[ans/foil]` rows).
  - Exact position-0 sink preservation for attention heads (`exclude_pos0=True` in `G_l`, native
    `pos=0` preserved in G1, discovery fixed-prefix `pos=0` sink anchored identically in G2 & `SHUF-pi`,
    with native exact `q/k/v` biases folding `mu_perp` cleanly).
  - Single key partition shared across 2-Hop 4-row & 6-row (`pilot` / `disc` / `eval` strictly disjoint),
    OurAirports `>= 96` held-out `p0`-blindspot cohort cleanly separated from full `air_eval` & known retention,
    genuine distinct-answer swap pairs with positive-energy guard, signed `wrong_item_ratio` guard,
    joint unrounded gate predicates (`G0 AND G1 AND G2`), live head restoration during generation,
    shared-vocab logit masking on EVERY `generate()`, native-BF16 chunked scoring (zero 2 GiB FP32 copy),
    and 20x faster cached `z_h` `SHUF-pi` + 1-pass `output_scores=True` discovery!

Usage:
  PYTHONPATH=platonicswap:experiments python experiments/v5_0_boundary_organ_gates_g0_g2.py --unit-test
  PYTHONPATH=platonicswap:experiments python -u experiments/v5_0_boundary_organ_gates_g0_g2.py --run-all --allow-download
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
OUT_PATH = RESULTS_DIR / "results_v5_0_boundary_organ_gates_g0_g2.json"

from exact_math_v43 import (
    StreamingSharedVocabFisherAccumulator,
    build_60k_calibration_corpus,
    solve_shared_vocab_bridge_m_u,
)
from extract_and_baseline import HF_CACHE_DIR, get_decoder_layers
from large_to_small import LARGE_TO_SMALL_MODELS, load_model_bf16
from native_weight_translation import (
    assert_zero_runtime_hooks,
    bf16_to_f64_np,
    get_config_rope_theta,
    unload_model_vram_safe,
)
from v2_falsifierA_rope_torus import rope_inv_freq
from v3_gate1_airports_enrichment import (
    check_iata_match,
    fetch_or_fallback_airports,
    format_airport_prompt,
)
from v3_gate2_skill_and_sql import (
    WORD_CANDIDATES_KEYS,
    WORD_CANDIDATES_VALS,
    filter_single_token_words,
)
from v4_4_private_lane_falsifier import get_pos0_and_gt0_masks
from v4_stage1_qwen7b_to_1p5b_native import (
    DONOR_KEY_DEFAULT,
    RECIPIENT_KEY_DEFAULT,
    SEED,
    _MaskPaddedVocabLogitsProcessor,
    greedy_generate_native,
)


# ======================================================================================
# 1. Exact Mathematical Primitives (Zero Backprop, Strict FP64 Solves)
# ======================================================================================

def _sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _left_pad_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Exact HuggingFace `generate()` position_ids for left- or right-padded masks (first real token == pos 0!)."""
    pos = attention_mask.long().cumsum(dim=-1) - 1
    return pos.masked_fill(attention_mask == 0, 1)


def _tensor_to_bf16_u16(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(dtype=torch.bfloat16).cpu().view(torch.uint16).numpy()


def _host_rss_gib() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB
    denom = float(1024 ** 3) if sys.platform == "darwin" else float(1024 ** 2)
    return round(float(rss) / denom, 3)


@torch.no_grad()
def _greedy_generate_clean(
    model: nn.Module,
    tok: Any,
    device: torch.device,
    prompts: Sequence[str],
    max_new_tokens: int = 5,
    batch_size: int = 32,
    v_shared: Optional[int] = None,
) -> List[str]:
    """
    Uniform raw greedy decoding (`do_sample=False, repetition_penalty=1.0`, shared-vocab mask,
    zero runtime hooks) so intact baselines, hooked ablations, live restorations, and raw step-0
    logits share the 100% identical decoding policy!
    """
    assert_zero_runtime_hooks(model)
    if not prompts:
        return []
    old_pad = tok.padding_side
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lproc = [_MaskPaddedVocabLogitsProcessor(v_shared)] if v_shared is not None else None
    out_txts: List[str] = []
    try:
        for i0 in range(0, len(prompts), batch_size):
            bp = list(prompts[i0 : i0 + batch_size])
            enc = tok(bp, return_tensors="pt", padding=True, add_special_tokens=True)
            ids = enc["input_ids"].to(device)
            gen_kw: Dict[str, Any] = dict(
                input_ids=ids,
                attention_mask=enc["attention_mask"].to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                use_cache=True,
                pad_token_id=pad_id,
            )
            if lproc is not None:
                gen_kw["logits_processor"] = lproc
            gen = model.generate(**gen_kw)
            plen = ids.shape[1]
            for bi in range(len(bp)):
                out_txts.append(tok.decode(gen[bi, plen:], skip_special_tokens=True))
    finally:
        tok.padding_side = old_pad
    return out_txts


def _native_bf16_argmax_shared(h_normed_f32: torch.Tensor, lm_w_bf16: torch.Tensor, chunk: int = 16) -> List[int]:
    """Chunked native-BF16 argmax over V_shared (ZERO 2.02 GiB `lm_w.float()` VRAM spike!)."""
    preds: List[int] = []
    n = int(h_normed_f32.shape[0])
    for i0 in range(0, n, chunk):
        hb = h_normed_f32[i0 : i0 + chunk].to(device=lm_w_bf16.device, dtype=lm_w_bf16.dtype)
        lg = F.linear(hb, lm_w_bf16)  # (chunk, V_shared) in BF16 (~4.8 MB!)
        preds.extend(torch.argmax(lg, dim=-1).cpu().tolist())
        del hb, lg
    return preds


def compute_raw_and_postnorm_fisher_plugs(
    G_A: np.ndarray,
    G_B: np.ndarray,
    G_BA: np.ndarray,
    gamma_A: np.ndarray,
    gamma_B: np.ndarray,
    eps_rel_med: float = 1e-2,
) -> Dict[str, Any]:
    """Compute post-norm `M_U` and exact raw pre-norm `T_raw` (`J_H = E_H diag(gamma_H)`)."""
    gA = np.asarray(gamma_A, dtype=np.float64).reshape(-1)
    gB = np.asarray(gamma_B, dtype=np.float64).reshape(-1)
    M_U, mu_meta = solve_shared_vocab_bridge_m_u(G_B, G_BA, eps_rel_med=eps_rel_med)

    G_A_raw = _sym((gA[:, None] * G_A) * gA[None, :])
    G_B_raw = _sym((gB[:, None] * G_B) * gB[None, :])
    G_BA_raw = (gB[:, None] * G_BA) * gA[None, :]

    eb = sla.eigvalsh(G_B_raw)
    pos = eb[eb > 1e-12]
    med_b_raw = float(np.median(pos)) if len(pos) > 0 else float(np.trace(G_B_raw) / max(G_B_raw.shape[0], 1))
    eps_raw = max(float(eps_rel_med) * med_b_raw, 1e-10)
    T_raw = sla.solve(G_B_raw + eps_raw * np.eye(G_B_raw.shape[0], dtype=np.float64), G_BA_raw, assume_a="pos")

    return {
        "M_U": M_U,
        "M_U_meta": mu_meta,
        "G_A_raw": G_A_raw,
        "G_B_raw": G_B_raw,
        "G_BA_raw": G_BA_raw,
        "T_raw": T_raw,
        "T_raw_meta": {
            "eps_rel_med": float(eps_rel_med),
            "median_eig_gb_raw": round(med_b_raw, 6),
            "eps_raw": round(eps_raw, 8),
        },
    }


def factor_psd_top_k(F_mat: np.ndarray, k_max: int = 128, rel_tol: float = 1e-6) -> Tuple[np.ndarray, float]:
    ev, V = sla.eigh(_sym(np.asarray(F_mat, dtype=np.float64)))
    ev = np.maximum(ev, 0.0)
    order = np.argsort(ev)[::-1]
    ev, V = ev[order], V[:, order]
    tot = float(np.sum(ev)) + 1e-18
    cutoff = float(ev[0]) * rel_tol if len(ev) > 0 else 0.0
    keep = np.where(ev > cutoff)[0][: int(k_max)]
    if len(keep) == 0:
        return np.zeros((F_mat.shape[0], 1), dtype=np.float64), 0.0
    L_F = V[:, keep] * np.sqrt(ev[keep])[None, :]
    return L_F, float(np.sum(ev[keep]) / tot)


@torch.no_grad()
def compute_mlp_demand_matrix_closed_form(
    W_g: torch.Tensor,
    W_u: torch.Tensor,
    W_d: torch.Tensor,
    X_in: torch.Tensor,
    L_F: torch.Tensor,
    chunk_size: int = 32,
) -> np.ndarray:
    """EXACT closed-form MLP demand `G = E_x[J(x)^T (L_F L_F^T) J(x)]` with ZERO backprop."""
    dev = W_g.device
    Wg_f = W_g.detach().to(device=dev, dtype=torch.float32)
    Wu_f = W_u.detach().to(device=dev, dtype=torch.float32)
    Wd_f = W_d.detach().to(device=dev, dtype=torch.float32)
    Lf_f = L_F.detach().to(device=dev, dtype=torch.float32)
    B = Lf_f.T @ Wd_f
    D = int(Wg_f.shape[1])
    G_acc = np.zeros((D, D), dtype=np.float64)
    n_tot = int(X_in.shape[0])
    assert n_tot > 0

    for i0 in range(0, n_tot, chunk_size):
        xb = X_in[i0 : i0 + chunk_size].to(device=dev, dtype=torch.float32)
        g = xb @ Wg_f.T
        u = xb @ Wu_f.T
        sig = torch.sigmoid(g)
        silu_g = g * sig
        silu_prime = sig * (1.0 + g * (1.0 - sig))
        ag = silu_prime * u
        au = silu_g
        for bi in range(xb.shape[0]):
            J_F = (B * ag[bi].unsqueeze(0)) @ Wg_f + (B * au[bi].unsqueeze(0)) @ Wu_f
            G_acc += (J_F.T @ J_F).cpu().double().numpy()
        del xb, g, u, sig, silu_g, silu_prime, ag, au

    return _sym(G_acc / float(n_tot))


def _make_rope_cos_sin(d_h: int, rope_theta: float, max_T: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = torch.from_numpy(rope_inv_freq(d_h, rope_theta)).to(device=device, dtype=torch.float32)
    pos = torch.arange(max_T, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


@torch.no_grad()
def compute_attn_head_demand_matrix_closed_form(
    W_q_h: torch.Tensor,
    b_q_h: Optional[torch.Tensor],
    W_k_kv: torch.Tensor,
    b_k_kv: Optional[torch.Tensor],
    W_v_kv: torch.Tensor,
    b_v_kv: Optional[torch.Tensor],
    W_o_h: torch.Tensor,
    X_seqs: Sequence[torch.Tensor],
    L_F: torch.Tensor,
    rope_theta: float = 1e6,
    exclude_pos0: bool = True,
) -> np.ndarray:
    """
    EXACT analytical closed-form attention-head demand matrix `G_head` at the final token `t = T-1`
    with ZERO backprop! When `exclude_pos0=True` (production default), excludes `s = 0` (BOS/sink)
    from the K/V sum so `G_head` lives strictly on `pos > 0` tokens!
    """
    dev = W_q_h.device
    Wq = W_q_h.detach().to(device=dev, dtype=torch.float32)
    Wk = W_k_kv.detach().to(device=dev, dtype=torch.float32)
    Wv = W_v_kv.detach().to(device=dev, dtype=torch.float32)
    Wo = W_o_h.detach().to(device=dev, dtype=torch.float32)
    bq = b_q_h.detach().to(device=dev, dtype=torch.float32) if b_q_h is not None else None
    bk = b_k_kv.detach().to(device=dev, dtype=torch.float32) if b_k_kv is not None else None
    bv = b_v_kv.detach().to(device=dev, dtype=torch.float32) if b_v_kv is not None else None
    Lf = L_F.detach().to(device=dev, dtype=torch.float32)

    d_h, D = int(Wq.shape[0]), int(Wq.shape[1])
    half = d_h // 2
    scale = 1.0 / math.sqrt(float(d_h))
    B_o = Lf.T @ Wo
    max_T = max(int(s.shape[0]) for s in X_seqs)
    cos_all, sin_all = _make_rope_cos_sin(d_h, rope_theta, max_T, dev)

    def _apply_rope_t(M: torch.Tensor, cos_v: torch.Tensor, sin_v: torch.Tensor) -> torch.Tensor:
        return torch.cat([M[..., :half] * cos_v + M[..., half:] * sin_v, M[..., half:] * cos_v - M[..., :half] * sin_v], dim=-1)

    G_acc = np.zeros((D, D), dtype=np.float64)
    n_seq = len(X_seqs)
    assert n_seq > 0

    for X_s in X_seqs:
        X = X_s.to(device=dev, dtype=torch.float32)
        T = int(X.shape[0])
        cos_t, sin_t = cos_all[:T], sin_all[:T]

        q_raw = X[-1] @ Wq.T + (bq if bq is not None else 0.0)
        k_raw = X @ Wk.T + (bk.unsqueeze(0) if bk is not None else 0.0)
        v_all = X @ Wv.T + (bv.unsqueeze(0) if bv is not None else 0.0)

        q_rot = torch.cat([q_raw[:half] * cos_t[-1] - q_raw[half:] * sin_t[-1], q_raw[half:] * cos_t[-1] + q_raw[:half] * sin_t[-1]], dim=-1)
        k_rot = torch.cat([k_raw[:, :half] * cos_t - k_raw[:, half:] * sin_t, k_raw[:, half:] * cos_t + k_raw[:, :half] * sin_t], dim=-1)

        alpha = F.softmax((k_rot @ q_rot) * scale, dim=0)
        dv = v_all - (alpha @ v_all).unsqueeze(0)
        dk = k_rot - (alpha @ k_rot).unsqueeze(0)

        M_vk = (alpha.unsqueeze(1) * dv).T @ dk
        J_q = _apply_rope_t(scale * (B_o @ M_vk), cos_t[-1], sin_t[-1]) @ Wq

        B_o_Wv = B_o @ Wv
        q_Wk = _apply_rope_t(q_rot.unsqueeze(0).expand(T, -1), cos_t, sin_t) @ Wk
        B_o_dv = dv @ B_o.T

        if T > 1 or not exclude_pos0:
            J_last = J_q + alpha[-1] * B_o_Wv + (scale * alpha[-1]) * torch.outer(B_o_dv[-1], q_Wk[-1])
            G_acc += (J_last.T @ J_last).cpu().double().numpy()

        s_start = 1 if exclude_pos0 else 0
        if T - 1 > s_start:
            a_prev = alpha[s_start:-1]
            dv_prev = B_o_dv[s_start:-1]
            qwk_prev = q_Wk[s_start:-1]
            a2 = a_prev * a_prev
            G_t1 = float(torch.sum(a2).item()) * (B_o_Wv.T @ B_o_Wv)
            M_cross = (B_o_Wv.T @ ((a2.unsqueeze(1) * dv_prev).T @ qwk_prev)) * scale
            w_k = a2 * torch.sum(dv_prev * dv_prev, dim=1) * (scale * scale)
            G_t3 = (qwk_prev.T * w_k.unsqueeze(0)) @ qwk_prev
            G_acc += (G_t1 + M_cross + M_cross.T + G_t3).cpu().double().numpy()

    return _sym(G_acc / float(n_seq))


def compute_demand_weighted_supply_boundary(
    C_xx: np.ndarray,
    C_xy: np.ndarray,
    C_yy: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    G: np.ndarray,
    X_tune: np.ndarray,
    Y_tune: np.ndarray,
    X_rep: np.ndarray,
    Y_rep: np.ndarray,
    ranks: Sequence[int] = (16, 32, 64, 128, 256),
    lam_grid: Sequence[float] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0),
) -> Dict[str, Any]:
    """
    Exact Fable 5.2 Max demand-weighted supply spectrum `(E_r, D_r)` + minimum-variance near-constant
    read `a*` + generic `val_tune`-selected frozen rank `frozen_r_tune` + held-out `val_report` metrics.
    """
    Cxx = _sym(np.asarray(C_xx, dtype=np.float64))
    Cyy = _sym(np.asarray(C_yy, dtype=np.float64))
    Cxy = np.asarray(C_xy, dtype=np.float64)
    mx = np.asarray(mu_x, dtype=np.float64).reshape(-1)
    my = np.asarray(mu_y, dtype=np.float64).reshape(-1)
    G_mat = _sym(np.asarray(G, dtype=np.float64))

    Xt = bf16_to_f64_np(X_tune) - mx[None, :]
    Yt = bf16_to_f64_np(Y_tune) - my[None, :]
    Xr = bf16_to_f64_np(X_rep) - mx[None, :]
    Yr = bf16_to_f64_np(Y_rep) - my[None, :]

    eg, Vg = sla.eigh(G_mat)
    eg = np.maximum(eg, 0.0)
    g_tol = max(float(np.max(eg)) * 1e-8, 1e-18) if len(eg) > 0 else 1e-18
    keep_g = eg > g_tol
    G_half = (Vg[:, keep_g] * np.sqrt(eg[keep_g])[None, :]) @ Vg[:, keep_g].T
    G_pinv_half = (Vg[:, keep_g] * (1.0 / np.sqrt(eg[keep_g]))[None, :]) @ Vg[:, keep_g].T

    ey, Vy = sla.eigh(Cyy)
    ey = np.maximum(ey, 0.0)
    pos_y = ey[ey > 1e-12]
    med_y = float(np.median(pos_y)) if len(pos_y) > 0 else float(np.trace(Cyy) / max(Cyy.shape[0], 1))
    Cxy_Vy = Cxy @ Vy

    Xt_Gh = Xt @ G_half
    Xt_Gh_tot = float(np.sum(Xt_Gh ** 2)) + 1e-18
    best_loss = 1e30
    best_lam_rel = float(lam_grid[0])
    best_A: Optional[np.ndarray] = None
    for lam_rel in lam_grid:
        lam_abs = max(float(lam_rel) * med_y, 1e-12)
        A_cand = (Cxy_Vy / (ey[None, :] + lam_abs)) @ Vy.T
        pred_Gh = (Yt @ A_cand.T) @ G_half
        loss = float(np.mean(np.sum((Xt_Gh - pred_Gh) ** 2, axis=1)))
        if loss < best_loss:
            best_loss = loss
            best_lam_rel = float(lam_rel)
            best_A = A_cand
    assert best_A is not None
    A_hat = best_A

    Sigma_s = _sym(A_hat @ Cyy @ A_hat.T)
    M_s = _sym(G_half @ Sigma_s @ G_half)
    M_x = _sym(G_half @ Cxx @ G_half)
    tr_Gx = float(np.trace(M_x)) + 1e-18
    kappa_fit_all = float(np.trace(M_s) / tr_Gx)

    lam_s, Q_s = sla.eigh(M_s)
    order = np.argsort(lam_s)[::-1]
    lam_s, Q_s = np.maximum(lam_s[order], 0.0), Q_s[:, order]

    Xr_Gh = Xr @ G_half
    Xr_Gh_tot = float(np.sum(Xr_Gh ** 2)) + 1e-18
    kappa_rep_all = 1.0 - float(np.sum((Xr_Gh - (Yr @ A_hat.T) @ G_half) ** 2) / Xr_Gh_tot)

    r_max = min(int(max(ranks)), int(np.sum(keep_g)), Cyy.shape[0])
    Q_max = Q_s[:, :r_max]
    E_max = Q_max.T @ (G_half @ A_hat)
    D_max = G_pinv_half @ Q_max

    # Generic `val_tune` and `val_report` per-mode rho_i^2
    Zt_true = Xt_Gh @ Q_max
    Zt_hat = Yt @ E_max.T
    mv_t = np.sum(Zt_true ** 2, axis=0)
    rho2_tune = np.where(mv_t > 1e-18, 1.0 - np.sum((Zt_true - Zt_hat) ** 2, axis=0) / np.maximum(mv_t, 1e-18), 0.0)

    Zr_true = Xr_Gh @ Q_max
    Zr_hat = Yr @ E_max.T
    mv_r = np.sum(Zr_true ** 2, axis=0)
    rho2_rep = np.where(mv_r > 1e-18, 1.0 - np.sum((Zr_true - Zr_hat) ** 2, axis=0) / np.maximum(mv_r, 1e-18), 0.0)

    denom_x_fit = np.sum((M_x @ Q_max) * Q_max, axis=0)
    rho2_fit = np.where(denom_x_fit > 1e-18, lam_s[:r_max] / np.maximum(denom_x_fit, 1e-18), 0.0)

    by_rank: Dict[int, Dict[str, Any]] = {}
    frozen_r_tune: Optional[int] = None
    best_r_fallback = int(ranks[0])
    best_kappa_tune = -1e9
    for r in sorted(int(rv) for rv in ranks):
        r_eff = min(r, r_max)
        kap_t = 1.0 - float(np.sum((Xt_Gh - Zt_hat[:, :r_eff] @ Q_max[:, :r_eff].T) ** 2) / Xt_Gh_tot)
        kap_r = 1.0 - float(np.sum((Xr_Gh - Zr_hat[:, :r_eff] @ Q_max[:, :r_eff].T) ** 2) / Xr_Gh_tot)
        kap_f = float(np.sum(lam_s[:r_eff]) / tr_Gx)
        min_rho_t = float(np.min(rho2_tune[:r_eff])) if r_eff > 0 else 0.0
        min_rho_r = float(np.min(rho2_rep[:r_eff])) if r_eff > 0 else 0.0
        by_rank[r] = {
            "r_eff": r_eff,
            "kappa_r_fit": kap_f,
            "kappa_r_tune": kap_t,
            "kappa_r_rep": kap_r,
            "min_rho2_tune": min_rho_t,
            "min_rho2_rep": min_rho_r,
            "median_rho2_rep": float(np.median(rho2_rep[:r_eff])) if r_eff > 0 else 0.0,
            "min_rho2_fit": float(np.min(rho2_fit[:r_eff])) if r_eff > 0 else 0.0,
        }
        # Generic-only preregistered selection: smallest r meeting kappa_tune >= 0.85 & min_rho2_tune >= 0.80!
        if frozen_r_tune is None and kap_t >= 0.85 and min_rho_t >= 0.80:
            frozen_r_tune = r
        if kap_t > best_kappa_tune:
            best_kappa_tune = kap_t
            best_r_fallback = r
    tune_criteria_met = frozen_r_tune is not None
    if frozen_r_tune is None:
        frozen_r_tune = best_r_fallback

    # Minimum-variance near-constant read a* tuned strictly on generic val_tune
    Y_tune_raw = bf16_to_f64_np(Y_tune)
    Y_rep_raw = bf16_to_f64_np(Y_rep)
    best_a_cv, best_a_star, best_a_eps = 1e9, None, 1e-3
    Vy_my = Vy.T @ my
    for eps_rel in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        eps_abs = max(float(eps_rel) * med_y, 1e-12)
        v_raw = Vy @ (Vy_my / (ey + eps_abs))
        denom = float(my @ v_raw)
        if abs(denom) < 1e-15:
            continue
        a_cand = v_raw / denom
        vt = Y_tune_raw @ a_cand
        cv_t = float(np.std(vt) / (abs(float(np.mean(vt))) + 1e-12))
        if cv_t < best_a_cv:
            best_a_cv, best_a_star, best_a_eps = cv_t, a_cand, float(eps_rel)
    assert best_a_star is not None
    vr = Y_rep_raw @ best_a_star
    a_mean_rep = float(np.mean(vr))
    a_std_rep = float(np.std(vr))
    a_cv_rep = float(a_std_rep / (abs(a_mean_rep) + 1e-12))

    return {
        "A_hat": A_hat,
        "G_half": G_half,
        "Q_max": Q_max,
        "E_max": E_max,
        "D_max": D_max,
        "a_star": best_a_star,
        "best_lam_rel": best_lam_rel,
        "frozen_r_tune": int(frozen_r_tune),
        "tune_criteria_met": bool(tune_criteria_met),
        "kappa_fit_all": kappa_fit_all,
        "kappa_rep_all": kappa_rep_all,
        "rho2_rep_all_modes": [round(float(v), 5) for v in rho2_rep],
        "by_rank": by_rank,
        "a_star_stats": {
            "eps_rel": best_a_eps,
            "mean_rep": a_mean_rep,
            "std_rep": a_std_rep,
            "cv_rep": a_cv_rep,
            "passes_2pct": bool(a_cv_rep <= 0.02),
        },
    }


def compute_astra_zero_intercept_supply(
    S_yy: np.ndarray,
    S_xy: np.ndarray,
    U_r: np.ndarray,
    X_tune: np.ndarray,
    Y_tune: np.ndarray,
    X_eval: np.ndarray,
    Y_eval: np.ndarray,
    X_diff_eval: Optional[np.ndarray] = None,
    Y_diff_eval: Optional[np.ndarray] = None,
    lam_grid: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
) -> Dict[str, Any]:
    """
    Astra 6.2 Ultra's exact zero-intercept supply plug `P in R^{r x D_B}` and worst-direction `R^2_worst`.
    Explicitly checks residual energy in the null/low-variance complement of `C_z` so a systematic bias
    in a constant target coordinate can NEVER produce a false `r2_worst = 1.0`!
    """
    U = np.asarray(U_r, dtype=np.float64)
    Syy = _sym(np.asarray(S_yy, dtype=np.float64))
    Szy = U.T @ np.asarray(S_xy, dtype=np.float64)

    ey, Vy = sla.eigh(Syy)
    ey = np.maximum(ey, 0.0)
    pos = ey[ey > 1e-12]
    med = float(np.median(pos)) if len(pos) > 0 else float(np.trace(Syy) / max(Syy.shape[0], 1))
    Szy_Vy = Szy @ Vy

    Zt = bf16_to_f64_np(X_tune) @ U
    Yt = bf16_to_f64_np(Y_tune)
    best_mse, best_P, best_lam = 1e30, None, float(lam_grid[0])
    for lam_rel in lam_grid:
        lam_abs = max(float(lam_rel) * med, 1e-12)
        P_c = (Szy_Vy / (ey[None, :] + lam_abs)) @ Vy.T
        mse = float(np.mean(np.sum((Zt - Yt @ P_c.T) ** 2, axis=1)))
        if mse < best_mse:
            best_mse, best_P, best_lam = mse, P_c, float(lam_rel)
    assert best_P is not None

    def _score(X_arr: np.ndarray, Y_arr: np.ndarray) -> Dict[str, Any]:
        Z = bf16_to_f64_np(X_arr) @ U
        Z_hat = bf16_to_f64_np(Y_arr) @ best_P.T
        E = Z - Z_hat
        Zc = Z - Z.mean(axis=0, keepdims=True)
        raw_tot_var = float(np.sum(Zc ** 2))
        pooled_r2 = (1.0 - float(np.sum(E ** 2) / raw_tot_var)) if raw_tot_var > 0.0 else (1.0 if float(np.sum(E ** 2)) == 0.0 else float("-inf"))
        C_z = _sym(Zc.T @ Zc)
        C_e = _sym(E.T @ E)
        ez, Vz = sla.eigh(C_z)
        max_ez = float(np.max(ez)) if len(ez) > 0 else 0.0
        z_tol_ident = max(max_ez * 1e-6, 1e-12)
        z_tol_nonzero = max_ez * 1e-14  # purely scale-invariant relative cutoff!
        ident = ez > z_tol_ident
        nonzero = ez > z_tol_nonzero
        true_null = ~nonzero
        null_err_e2 = float(np.sum((E @ Vz[:, true_null]) ** 2)) if np.any(true_null) else 0.0
        null_leak = bool(null_err_e2 > 1e-16 * raw_tot_var)

        r2_ident: Optional[float] = None
        if np.any(ident):
            inv_i = Vz[:, ident] * (1.0 / np.sqrt(ez[ident]))[None, :]
            r2_ident = 1.0 - float(np.max(sla.eigvalsh(_sym(inv_i.T @ C_e @ inv_i))))

        r2_unrestricted: Optional[float] = None
        if (not null_leak) and np.any(nonzero):
            inv_nz = Vz[:, nonzero] * (1.0 / np.sqrt(ez[nonzero]))[None, :]
            r2_unrestricted = 1.0 - float(np.max(sla.eigvalsh(_sym(inv_nz.T @ C_e @ inv_nz))))

        return {
            "pooled_r2": pooled_r2,
            "r2_worst_identifiable": r2_ident,
            "r2_worst_unrestricted": r2_unrestricted,
            "null_space_error_energy": null_err_e2,
            "null_space_leak": null_leak,
            "n_identifiable": int(np.sum(ident)),
            "n_nonzero_var": int(np.sum(nonzero)),
            "ident_tol": z_tol_ident,
            "n_samples": int(len(Z)),
        }

    out: Dict[str, Any] = {"P": best_P, "best_lam_rel": best_lam, "natural": _score(X_eval, Y_eval)}
    if X_diff_eval is not None and Y_diff_eval is not None and len(X_diff_eval) > 0:
        out["differences"] = _score(X_diff_eval, Y_diff_eval)
    return out


@torch.no_grad()
def compute_cancellation_ratio_chi(
    W_g: torch.Tensor,
    W_u: torch.Tensor,
    W_d: torch.Tensor,
    X_in: torch.Tensor,
    Z_code: np.ndarray,
    L_F: torch.Tensor,
) -> Dict[str, Any]:
    """Exact Fable 5.2 Max cancellation ratio `chi` & `chi^2` using rank-revealing SVD affine projection."""
    dev = W_g.device
    X = X_in.to(device=dev, dtype=torch.float32)
    Wd = W_d.detach().to(device=dev, dtype=torch.float32)
    Lf = L_F.detach().to(device=dev, dtype=torch.float32)

    g = F.linear(X.to(dtype=W_g.dtype), W_g)
    u = F.linear(X.to(dtype=W_u.dtype), W_u)
    M_act = (F.silu(g) * u).float()
    B = Lf.T @ Wd
    b_norms = torch.linalg.norm(B, dim=0)

    m_norms = torch.linalg.norm(M_act, dim=0)
    Y_raw = M_act @ B.T
    num_raw = float(torch.sum(b_norms * m_norms).item())
    den_raw = float(torch.linalg.norm(Y_raw).item()) + 1e-12
    chi_raw = num_raw / den_raw

    # Rank-revealing SVD orthonormal basis for [Z_code, 1]
    Z_aug = np.concatenate([np.asarray(Z_code, dtype=np.float64), np.ones((len(Z_code), 1), dtype=np.float64)], axis=1)
    Uz, sz, _ = sla.svd(Z_aug, full_matrices=False)
    keep_z = sz > (float(sz[0]) * 1e-8 if len(sz) > 0 else 0.0)
    Q_z_t = torch.from_numpy(Uz[:, keep_z].astype(np.float32)).to(device=dev)
    N_act = M_act - Q_z_t @ (Q_z_t.T @ M_act)
    n_norms = torch.linalg.norm(N_act, dim=0)
    Y_nl = N_act @ B.T
    num_nl = float(torch.sum(b_norms * n_norms).item())
    den_nl = float(torch.linalg.norm(Y_nl).item())
    nl_frac = float((den_nl ** 2) / (den_raw ** 2 + 1e-12))
    negligible_nl = bool(nl_frac < 1e-4)
    chi_nl = float(num_nl / (den_nl + 1e-12)) if not negligible_nl else None

    return {
        "n_neurons": int(W_g.shape[0]),
        "chi_nl": round(chi_nl, 4) if chi_nl is not None else None,
        "chi_nl_sq": round(chi_nl ** 2, 2) if chi_nl is not None else None,
        "slicing_rel_err_bound_k256": round((chi_nl ** 2 - 1.0) / 256.0, 4) if chi_nl is not None else None,
        "slicing_rel_err_bound_k2048": round((chi_nl ** 2 - 1.0) / 2048.0, 4) if chi_nl is not None else None,
        "chi_raw": round(chi_raw, 4),
        "chi_raw_sq": round(chi_raw ** 2, 2),
        "nl_energy_frac": round(nl_frac, 6),
        "negligible_nl_energy": negligible_nl,
    }


@torch.no_grad()
def run_width_compression_ladder(
    A_cal: torch.Tensor,
    B_proj: torch.Tensor,
    ladder: Sequence[int] = (18944, 8192, 4096, 2048, 1024, 512, 256),
) -> Dict[str, Any]:
    """
    Unlabeled post-G1 width-compression ladder comparing:
      1) Output-projected Wanda / leverage score `||a_j||_2 * ||b_j||_2`
      2) Vectorized Forward Cancellation-Aware Selection (`min_{|S|<=m} ||Y_full - sum_{j in S} a_j b_j^T||_F^2`,
         climbing UP the ladder `256 -> ... -> M` with exact rank-1/block residual updates + 1-swap refinement!).
    """
    dev = A_cal.device
    A = A_cal.detach().to(device=dev, dtype=torch.float32)
    B = B_proj.detach().to(device=dev, dtype=torch.float32)
    N, M = int(A.shape[0]), int(A.shape[1])
    Y_full = A @ B.T
    tot_e2 = float(torch.sum(Y_full * Y_full).item()) + 1e-12
    prod_e2 = torch.sum(A * A, dim=0) * torch.sum(B * B, dim=0)
    wanda_order = torch.argsort(prod_e2, descending=True)

    rungs_asc = sorted([min(int(m), M) for m in set(ladder)])
    in_S = torch.zeros(M, device=dev, dtype=torch.bool)
    R = Y_full.clone()

    cancel_subsets: Dict[int, np.ndarray] = {}
    cancel_rel_err: Dict[int, float] = {}
    wanda_subsets: Dict[int, np.ndarray] = {}
    wanda_rel_err: Dict[int, float] = {}

    for target_m in rungs_asc:
        if target_m >= M:
            in_S[:] = True
            R.zero_()
        else:
            while int(in_S.sum().item()) < target_m:
                gap = target_m - int(in_S.sum().item())
                step = 1 if target_m <= 64 else min(gap, max(1, min(64, gap // 12)))
                cross = torch.sum(A * (R @ B), dim=0)
                delta_add = torch.where(~in_S, -2.0 * cross + prod_e2, torch.tensor(float("inf"), device=dev))
                _, add_idx = torch.topk(delta_add, k=step, largest=False)
                in_S[add_idx] = True
                R = R - A[:, add_idx] @ B[:, add_idx].T

            if 0 < target_m < M:
                for _ in range(16):
                    cross = torch.sum(A * (R @ B), dim=0)
                    drop_delta = torch.where(in_S, 2.0 * cross + prod_e2, torch.tensor(float("inf"), device=dev))
                    best_drop = int(torch.argmin(drop_delta).item())
                    R_try = R + torch.outer(A[:, best_drop], B[:, best_drop])
                    cross_try = torch.sum(A * (R_try @ B), dim=0)
                    add_delta = torch.where(~in_S, -2.0 * cross_try + prod_e2, torch.tensor(float("inf"), device=dev))
                    best_add = int(torch.argmin(add_delta).item())
                    if float(drop_delta[best_drop].item()) + float(add_delta[best_add].item()) < -1e-6 * tot_e2:
                        in_S[best_drop] = False
                        in_S[best_add] = True
                        R = R_try - torch.outer(A[:, best_add], B[:, best_add])
                    else:
                        break

        cancel_subsets[target_m] = torch.nonzero(in_S, as_tuple=False).squeeze(-1).cpu().numpy().astype(np.int64)
        cancel_rel_err[target_m] = float(torch.sum(R * R).item() / tot_e2)

        idx_w = wanda_order[:target_m]
        Y_w = A[:, idx_w] @ B[:, idx_w].T
        wanda_subsets[target_m] = idx_w.cpu().numpy().astype(np.int64)
        wanda_rel_err[target_m] = float(torch.sum((Y_full - Y_w) ** 2).item() / tot_e2)

    return {
        "rungs": rungs_asc,
        "cancel_subsets": cancel_subsets,
        "wanda_subsets": wanda_subsets,
        "cancel_rel_err": {str(k): round(v, 6) for k, v in sorted(cancel_rel_err.items())},
        "wanda_rel_err": {str(k): round(v, 6) for k, v in sorted(wanda_rel_err.items())},
    }


# ======================================================================================
# 2. Datasets: Single-Partition 2-Hop (4-Row & 6-Row) & OurAirports Strict Scorer
# ======================================================================================

def partition_2hop_key_pools(
    tokenizers: Sequence[Any],
    max_rows: int = 6,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Partition 2-Hop keys ONCE into strictly disjoint `(pilot, disc, eval)` pools shared by ALL row counts!"""
    rng = np.random.default_rng(seed)
    raw_keys = list(dict.fromkeys(WORD_CANDIDATES_KEYS))
    raw_vals = list(dict.fromkeys(WORD_CANDIDATES_VALS))
    all_keys = filter_single_token_words(raw_keys, tokenizers)
    all_vals = filter_single_token_words(raw_vals, tokenizers)

    def _is_clean_ans(w: str) -> bool:
        for tok in tokenizers:
            p_ids = tok.encode("Output:", add_special_tokens=False)
            f_ids = tok.encode(f"Output: {w}", add_special_tokens=False)
            if not (len(f_ids) == len(p_ids) + 1 and f_ids[: len(p_ids)] == p_ids):
                return False
        return True

    all_vals = [w for w in all_vals if _is_clean_ans(w)]
    all_keys = [w for w in all_keys if w not in set(all_vals)]
    need_k = 2 * int(max_rows)
    assert len(all_keys) >= 3 * need_k, f"Need >= {3*need_k} keys, got {len(all_keys)}"
    assert len(all_vals) >= 2 * int(max_rows), f"Need >= {2*max_rows} vals, got {len(all_vals)}"

    perm_k = rng.permutation(len(all_keys))
    n_p = max(need_k + 4, int(0.20 * len(all_keys)))
    n_d = max(need_k + 6, int(0.35 * len(all_keys)))
    pilot_keys = [all_keys[i] for i in perm_k[:n_p]]
    disc_keys = [all_keys[i] for i in perm_k[n_p : n_p + n_d]]
    eval_keys = [all_keys[i] for i in perm_k[n_p + n_d :]]
    assert set(pilot_keys).isdisjoint(disc_keys)
    assert set(pilot_keys).isdisjoint(eval_keys)
    assert set(disc_keys).isdisjoint(eval_keys)
    return {"pilot_keys": pilot_keys, "disc_keys": disc_keys, "eval_keys": eval_keys, "all_vals": all_vals}


def build_in_prompt_2hop_dataset(
    tokenizers: Sequence[Any],
    key_pools: Dict[str, Any],
    n_rows: int = 6,
    n_pilot: int = 64,
    n_disc: int = 200,
    n_eval: int = 400,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Build identical-prompt multi-row 2-hop lookup dataset using a single shared disjoint key partition."""
    assert n_pilot % 2 == 0 and n_disc % 2 == 0 and n_eval % 2 == 0
    rng = np.random.default_rng(seed + 1000 * int(n_rows))
    tok0 = tokenizers[0]
    p_ref = tok0.encode("Output:", add_special_tokens=False)
    all_vals = key_pools["all_vals"]

    def _ans_tok_id(w: str) -> int:
        return int(tok0.encode(f"Output: {w}", add_special_tokens=False)[len(p_ref)])

    def _format_prompt(h1: Sequence[Tuple[str, str]], h2: Sequence[Tuple[str, str, str]], query: str, side: str) -> str:
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
        lines.append("Output:")
        return "\n".join(lines)

    def _gen_pairs(key_pool: List[str], n_total_items: int, split_name: str) -> List[Dict[str, Any]]:
        n_tables = n_total_items // 2
        items: List[Dict[str, Any]] = []
        seen = set()
        t_idx = 0
        while t_idx < n_tables:
            ck = rng.choice(key_pool, size=2 * n_rows, replace=False).tolist()
            cv = rng.choice(all_vals, size=2 * n_rows, replace=False).tolist()
            hop1 = [(ck[i], ck[n_rows + i]) for i in range(n_rows)]
            hop2 = [(ck[n_rows + i], cv[2 * i], cv[2 * i + 1]) for i in range(n_rows)]
            h1s = [hop1[i] for i in rng.permutation(n_rows).tolist()]
            h2s = [hop2[i] for i in rng.permutation(n_rows).tolist()]
            qi = int(rng.integers(0, n_rows))
            q, mid_w = ck[qi], ck[n_rows + qi]
            vL, vR = cv[2 * qi], cv[2 * qi + 1]
            sig = (tuple(h1s), tuple(h2s), q)
            if sig in seen:
                continue
            seen.add(sig)
            for side in ("L", "R"):
                ans, foil = (vL, vR) if side == "L" else (vR, vL)
                items.append({
                    "id": f"{split_name}_t{t_idx:03d}_{side}",
                    "table_id": t_idx,
                    "split": split_name,
                    "side": side,
                    "n_rows": int(n_rows),
                    "query": q, "mid": mid_w,
                    "answer": ans, "foil": foil,
                    "answer_tok_id": _ans_tok_id(ans),
                    "foil_tok_id": _ans_tok_id(foil),
                    "prompt": _format_prompt(h1s, h2s, q, side),
                })
            t_idx += 1
        return items

    return {
        "n_rows": int(n_rows),
        "pilot": _gen_pairs(key_pools["pilot_keys"], n_pilot, "pilot"),
        "disc": _gen_pairs(key_pools["disc_keys"], n_disc, "disc"),
        "eval": _gen_pairs(key_pools["eval_keys"], n_eval, "eval"),
        "n_independent_tables": {"pilot": n_pilot // 2, "disc": n_disc // 2, "eval": n_eval // 2},
        "n_keys_by_split": {k: len(key_pools[f"{k}_keys"]) for k in ("pilot", "disc", "eval")},
    }


def check_2hop_exact_word(gen_text: str, target_word: str) -> bool:
    """Strict first-word exact match (rejects digits/corruption like 'r2ed')."""
    toks = gen_text.strip().split()
    if not toks:
        return False
    w = toks[0].strip(".,:;!?\"'()[]")
    return w.isalpha() and w.lower() == target_word.lower()


def check_iata_strict_field(gen_text: str, target_iata: str) -> bool:
    """Strict answer-field whole-code IATA match: first non-empty field must equal the 3-letter IATA code."""
    toks = gen_text.strip().split()
    if not toks:
        return False
    w = toks[0].strip(".,:;!?\"'()[]")
    return len(w) == 3 and w.isalpha() and w.upper() == target_iata.upper()


def attach_airport_answer_tokens(airports: List[Dict[str, Any]], tok: Any, phrasing_idx: int = 0) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for ap in airports:
        pr, _ = format_airport_prompt(ap["name"], phrasing_idx)
        p_ids = tok.encode(pr, add_special_tokens=True)
        f_ids = tok.encode(pr + " " + ap["iata"], add_special_tokens=True)
        if len(f_ids) > len(p_ids) and f_ids[: len(p_ids)] == p_ids:
            ans_ids = [int(x) for x in f_ids[len(p_ids) :]]
        else:
            ans_ids = [int(x) for x in tok.encode(" " + ap["iata"], add_special_tokens=False)]
        enriched.append({
            **ap,
            "prompt": pr,
            "phrasing_idx": int(phrasing_idx),
            "ans_tok_ids": ans_ids,
            "first_ans_tok_id": int(ans_ids[0]),
        })
    return enriched


def build_matched_airport_swap_pairs(items: Sequence[Dict[str, Any]]) -> Tuple[List[Tuple[int, int]], Dict[int, int]]:
    """
    Build maximum-cardinality genuine bidirectional matched swap pairs `(s, b)` with STRICTLY
    DISTINCT `first_ans_tok_id` using a greedy max-bucket matcher (never wastes valid cross-ID
    pairs like `[1, 1, 1, 1, 2]`, ZERO fabricated foils!).
    """
    import heapq
    n = len(items)
    if n < 2:
        return [], {}
    buckets: Dict[int, List[int]] = {}
    for i, it in enumerate(items):
        buckets.setdefault(int(it["first_ans_tok_id"]), []).append(i)
    heap: List[Tuple[int, int]] = [(-len(idxs), tid) for tid, idxs in buckets.items()]
    heapq.heapify(heap)
    pairs: List[Tuple[int, int]] = []
    foil_map: Dict[int, int] = {}
    while len(heap) >= 2:
        neg1, tid1 = heapq.heappop(heap)
        neg2, tid2 = heapq.heappop(heap)
        ia = buckets[tid1].pop()
        ib = buckets[tid2].pop()
        pairs.append((ia, ib))
        pairs.append((ib, ia))
        foil_map[ia] = int(tid2)
        foil_map[ib] = int(tid1)
        if buckets[tid1]:
            heapq.heappush(heap, (-len(buckets[tid1]), tid1))
        if buckets[tid2]:
            heapq.heappush(heap, (-len(buckets[tid2]), tid2))
    return pairs, foil_map


def build_2hop_table_swap_pairs(items: Sequence[Dict[str, Any]]) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for k in range(0, len(items) - 1, 2):
        if items[k].get("table_id") == items[k + 1].get("table_id"):
            pairs.extend([(k, k + 1), (k + 1, k)])
    return pairs


def _compute_safe_q_and_r2(d_hat: np.ndarray, d_ref: np.ndarray, c_hat: np.ndarray, c_true: np.ndarray, min_pairs: int = 20) -> Dict[str, Any]:
    """Compute `Q_causal` & `R2_c` with positive-energy / sample-support guards (never `Q=1` on empty/zero ref!)."""
    d_e2 = float(np.sum(d_ref ** 2))
    c_var = float(np.sum((c_true - np.mean(c_true)) ** 2))
    q_valid = bool(len(d_ref) >= min_pairs and d_e2 > 1e-4)
    r2_valid = bool(len(c_true) >= 10 and c_var > 1e-4)
    q_val = (1.0 - float(np.sum((d_hat - d_ref) ** 2) / d_e2)) if q_valid else None
    r2_val = (1.0 - float(np.sum((c_hat - c_true) ** 2) / c_var)) if r2_valid else None
    return {
        "Q_causal": q_val,
        "R2_c": r2_val,
        "q_valid": q_valid,
        "r2_valid": r2_valid,
        "d_ref_e2": d_e2,
        "c_true_var": c_var,
    }


def _compute_safe_wrong_item_ratio(c_hat: np.ndarray, c_true: np.ndarray, wrong_mask: np.ndarray, min_donor_signal: float = 0.25) -> Dict[str, Any]:
    """Signed guard for `wrong_item_ratio`: NEVER rewards `-1 / -1 = +1` or fakes KILL on unsupported denom!"""
    n_w = int(np.sum(wrong_mask))
    if n_w == 0:
        return {"n_wrong": 0, "mean_c_true_wrong": 0.0, "mean_c_hat_wrong": 0.0, "wrong_item_ratio": None, "ratio_supported": False}
    mc_w = float(np.mean(c_true[wrong_mask]))
    mch_w = float(np.mean(c_hat[wrong_mask]))
    supported = bool(mc_w >= min_donor_signal)
    ratio = float(mch_w / mc_w) if supported else None
    return {
        "n_wrong": n_w,
        "mean_c_true_wrong": mc_w,
        "mean_c_hat_wrong": mch_w,
        "wrong_item_ratio": ratio,
        "ratio_supported": supported,
    }


# ======================================================================================
# 3. Single-Pass Discovery (DLA + Solo Mean-Ablation with Shared-Vocab Masking)
# ======================================================================================

@torch.no_grad()
def run_all_layer_dla_and_solo_ablation(
    model: nn.Module,
    tok: Any,
    device: torch.device,
    items: Sequence[Dict[str, Any]],
    prompt_key: str,
    ans_tok_key: str,
    foil_tok_key: str,
    exact_match_fn: Callable[[str, Dict[str, Any]], bool],
    v_shared: int,
    max_new_tokens: int = 5,
    single_token_task: bool = False,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    All-28-layer causal discovery on Donor 7B (`disc` split ONLY!):
      - ONE forward pass captures all 28 MLPs + all 784 Attention Heads -> exact DLA!
      - Solo mean-ablation probes across ALL 28 MLPs + top-16 late heads using SINGLE-PASS
        `generate(..., output_scores=True, return_dict_in_generate=True)` (or direct forward pass for
        1-token tasks) with `_MaskPaddedVocabLogitsProcessor(v_shared)` — 2x+ faster & exact!
    """
    assert_zero_runtime_hooks(model)
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    n_items = len(items)
    D_A = int(model.config.hidden_size)
    n_q = int(model.config.num_attention_heads)
    d_h = int(layers[0].self_attn.head_dim)

    prompts = [it[prompt_key] for it in items]
    ans_ids = np.array([int(it[ans_tok_key]) for it in items], dtype=np.int64)
    foil_ids = np.array([int(it[foil_tok_key]) for it in items], dtype=np.int64)

    E_A = model.lm_head.weight[:v_shared].detach()
    gA_np = model.model.norm.weight.detach().cpu().double().numpy()
    eps_A = float(getattr(model.model.norm, "variance_epsilon", getattr(model.model.norm, "eps", 1e-6)))

    mlp_outs = [np.zeros((n_items, D_A), dtype=np.float32) for _ in range(n_layers)]
    attn_z_all = [np.zeros((n_items, n_q * d_h), dtype=np.float32) for _ in range(n_layers)]
    h_final_all = np.zeros((n_items, D_A), dtype=np.float32)
    donor_logit_diff = np.zeros(n_items, dtype=np.float64)

    old_p = tok.padding_side
    tok.padding_side = "left"
    handles = []
    sl = [0, 0]
    try:
        def _mk_m(li: int):
            def _h(_m, _i, o):
                t = o[0] if isinstance(o, tuple) else o
                mlp_outs[li][sl[0] : sl[1]] = t[:, -1, :].detach().float().cpu().numpy()
            return _h

        def _mk_z(li: int):
            def _h(_m, inp):
                attn_z_all[li][sl[0] : sl[1]] = inp[0][:, -1, :].detach().float().cpu().numpy()
            return _h

        def _fin(_m, inp):
            h_final_all[sl[0] : sl[1]] = inp[0][:, -1, :].detach().float().cpu().numpy()

        for li in range(n_layers):
            handles.append(layers[li].mlp.register_forward_hook(_mk_m(li)))
            handles.append(layers[li].self_attn.o_proj.register_forward_pre_hook(_mk_z(li)))
        handles.append(model.model.norm.register_forward_pre_hook(_fin))

        for i0 in range(0, n_items, batch_size):
            i1 = min(i0 + batch_size, n_items)
            sl[0], sl[1] = i0, i1
            enc = tok(prompts[i0:i1], return_tensors="pt", padding=True, add_special_tokens=True)
            ids_b, msk_b = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            out = model(input_ids=ids_b, attention_mask=msk_b, position_ids=_left_pad_position_ids(msk_b), use_cache=False)
            lg = out.logits[:, -1, :v_shared].float()
            for bi in range(i1 - i0):
                idx = i0 + bi
                donor_logit_diff[idx] = float((lg[bi, ans_ids[idx]] - lg[bi, foil_ids[idx]]).item())
            del out, lg
    finally:
        for h in handles:
            h.remove()
        tok.padding_side = old_p
    assert_zero_runtime_hooks(model)

    rms_final = np.sqrt(np.mean(h_final_all.astype(np.float64) ** 2, axis=1) + eps_A)
    u_diff = np.zeros((n_items, D_A), dtype=np.float64)
    for i in range(n_items):
        ea = E_A[int(ans_ids[i])].float().cpu().double().numpy()
        ef = E_A[int(foil_ids[i])].float().cpu().double().numpy()
        u_diff[i] = ((ea - ef) * gA_np) / rms_final[i]

    mean_ld = float(np.mean(donor_logit_diff))
    mlp_dla: List[Dict[str, Any]] = []
    head_dla: List[Dict[str, Any]] = []
    for li in range(n_layers):
        mo = mlp_outs[li].astype(np.float64)
        c_mlp = np.sum((mo - mo.mean(axis=0, keepdims=True)) * u_diff, axis=1)
        mlp_dla.append({
            "layer": li,
            "mean_dla": float(np.mean(c_mlp)),
            "frac_of_donor_ld": float(np.mean(c_mlp) / (abs(mean_ld) + 1e-12)),
            "pos_frac": float(np.mean(c_mlp > 0)),
        })
        Wo = layers[li].self_attn.o_proj.weight.detach().float().cpu().double().numpy()
        az = attn_z_all[li].astype(np.float64)
        az_c = az - az.mean(axis=0, keepdims=True)
        for hi in range(n_q):
            sh = slice(hi * d_h, (hi + 1) * d_h)
            c_h = np.sum(az_c[:, sh] * (u_diff @ Wo[:, sh]), axis=1)
            head_dla.append({
                "layer": li,
                "head": hi,
                "mean_dla": float(np.mean(c_h)),
                "frac_of_donor_ld": float(np.mean(c_h) / (abs(mean_ld) + 1e-12)),
            })

    # Score intact with the 100% IDENTICAL decoding horizon & policy (`repetition_penalty=1.0`, shared-vocab mask)!
    eff_max_new = 1 if single_token_task else max_new_tokens
    gens_intact = _greedy_generate_clean(model, tok, device, prompts, max_new_tokens=eff_max_new, batch_size=batch_size, v_shared=v_shared)
    acc_intact = float(np.mean([int(exact_match_fn(g, it)) for g, it in zip(gens_intact, items)]))

    lproc = [_MaskPaddedVocabLogitsProcessor(v_shared)]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def _run_one_ablation(hook_fn: Callable[[], Any]) -> Tuple[float, float]:
        old_p2 = tok.padding_side
        tok.padding_side = "left"
        h_ab = None
        txts: List[str] = []
        ld_ab = np.zeros(n_items, dtype=np.float64)
        try:
            h_ab = hook_fn()
            for i0 in range(0, n_items, batch_size):
                i1 = min(i0 + batch_size, n_items)
                enc = tok(prompts[i0:i1], return_tensors="pt", padding=True, add_special_tokens=True)
                ids = enc["input_ids"].to(device)
                msk = enc["attention_mask"].to(device)
                if single_token_task:
                    fwd = model(input_ids=ids, attention_mask=msk, position_ids=_left_pad_position_ids(msk), use_cache=False)
                    lg = fwd.logits[:, -1, :v_shared].float()
                    pred_ids = torch.argmax(lg, dim=-1)
                    for bi in range(i1 - i0):
                        idx = i0 + bi
                        ld_ab[idx] = float((lg[bi, ans_ids[idx]] - lg[bi, foil_ids[idx]]).item())
                        txts.append(tok.decode([int(pred_ids[bi].item())], skip_special_tokens=True))
                    del fwd, lg
                else:
                    gen_out = model.generate(
                        input_ids=ids, attention_mask=msk, max_new_tokens=max_new_tokens,
                        do_sample=False, repetition_penalty=1.0, use_cache=True, pad_token_id=pad_id,
                        logits_processor=lproc, output_scores=True, return_dict_in_generate=True,
                    )
                    lg0 = gen_out.scores[0][:, :v_shared].float()
                    plen = ids.shape[1]
                    for bi in range(i1 - i0):
                        idx = i0 + bi
                        ld_ab[idx] = float((lg0[bi, ans_ids[idx]] - lg0[bi, foil_ids[idx]]).item())
                        txts.append(tok.decode(gen_out.sequences[bi, plen:], skip_special_tokens=True))
                    del gen_out, lg0
        finally:
            if h_ab is not None:
                h_ab.remove()
            tok.padding_side = old_p2
        em = float(np.mean([int(exact_match_fn(g, it)) for g, it in zip(txts, items)]))
        return em, float(np.mean(ld_ab))

    mlp_solo: List[Dict[str, Any]] = []
    for li in range(n_layers):
        mean_m = torch.from_numpy(mlp_outs[li].mean(axis=0)).to(device=device, dtype=layers[li].mlp.down_proj.weight.dtype)

        def _reg_mlp(layer_i=li, mv=mean_m):
            def _ab(_m, _i, out):
                o = out[0] if isinstance(out, tuple) else out
                o2 = o.clone()
                o2[:, -1, :] = mv
                return (o2,) + out[1:] if isinstance(out, tuple) else o2
            return layers[layer_i].mlp.register_forward_hook(_ab)

        em_ab, ld_ab = _run_one_ablation(_reg_mlp)
        mlp_solo.append({
            "layer": li,
            "em_ablated": round(em_ab, 4),
            "em_drop_pp": round(100.0 * (acc_intact - em_ab), 2),
            "ld_drop": round(mean_ld - ld_ab, 4),
            "mean_dla": round(mlp_dla[li]["mean_dla"], 4),
            "frac_of_donor_ld": round(mlp_dla[li]["frac_of_donor_ld"], 4),
        })

    late_heads_sorted = sorted([h for h in head_dla if h["layer"] >= 14], key=lambda x: x["mean_dla"], reverse=True)
    head_solo: List[Dict[str, Any]] = []
    for hd in late_heads_sorted[:16]:
        li, hi = int(hd["layer"]), int(hd["head"])
        sh = slice(hi * d_h, (hi + 1) * d_h)
        mz = torch.from_numpy(attn_z_all[li][:, sh].mean(axis=0)).to(device=device, dtype=layers[li].self_attn.o_proj.weight.dtype)

        def _reg_h(layer_i=li, sl_h=sh, mzv=mz):
            def _ab(_m, inp):
                x = inp[0].clone()
                x[:, -1, sl_h] = mzv
                return (x,) + inp[1:]
            return layers[layer_i].self_attn.o_proj.register_forward_pre_hook(_ab)

        em_ab, ld_ab = _run_one_ablation(_reg_h)
        head_solo.append({
            "layer": li, "head": hi,
            "mean_dla": round(hd["mean_dla"], 4),
            "frac_of_donor_ld": round(hd["frac_of_donor_ld"], 4),
            "em_ablated": round(em_ab, 4),
            "em_drop_pp": round(100.0 * (acc_intact - em_ab), 2),
            "ld_drop": round(mean_ld - ld_ab, 4),
        })

    assert_zero_runtime_hooks(model)
    return {
        "n_disc": n_items,
        "acc_intact_disc": round(acc_intact, 4),
        "mean_donor_logit_diff_disc": round(mean_ld, 4),
        "mlp_dla_all28": mlp_dla,
        "mlp_solo_ablation_all28": mlp_solo,
        "top16_late_heads_dla_and_solo": head_solo,
        "top20_all_heads_dla": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items()}
            for h in sorted(head_dla, key=lambda x: x["mean_dla"], reverse=True)[:20]
        ],
    }


# ======================================================================================
# 4. Gate G1: Whole-MLP & Late-Head Donor Self-Tests (Selection STRICTLY on `disc`!)
# ======================================================================================

@torch.no_grad()
def _extract_mlp_io_at_last_token(
    model: nn.Module, tok: Any, device: torch.device, prompts: Sequence[str], layer_idx: int, batch_size: int = 32
) -> Dict[str, np.ndarray]:
    assert_zero_runtime_hooks(model)
    layers = get_decoder_layers(model)
    D = int(model.config.hidden_size)
    n = len(prompts)
    x_in = np.zeros((n, D), dtype=np.float32)
    m_out = np.zeros((n, D), dtype=np.float32)
    h_fin = np.zeros((n, D), dtype=np.float32)
    old_p = tok.padding_side
    tok.padding_side = "left"
    handles = []
    sl = [0, 0]
    try:
        handles.append(layers[layer_idx].post_attention_layernorm.register_forward_hook(
            lambda _m, _i, o: x_in.__setitem__(slice(sl[0], sl[1]), (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach().float().cpu().numpy())
        ))
        handles.append(layers[layer_idx].mlp.register_forward_hook(
            lambda _m, _i, o: m_out.__setitem__(slice(sl[0], sl[1]), (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach().float().cpu().numpy())
        ))
        handles.append(model.model.norm.register_forward_pre_hook(
            lambda _m, inp: h_fin.__setitem__(slice(sl[0], sl[1]), inp[0][:, -1, :].detach().float().cpu().numpy())
        ))
        for i0 in range(0, n, batch_size):
            i1 = min(i0 + batch_size, n)
            sl[0], sl[1] = i0, i1
            enc = tok(list(prompts[i0:i1]), return_tensors="pt", padding=True, add_special_tokens=True)
            ids_b, msk_b = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            model(input_ids=ids_b, attention_mask=msk_b, position_ids=_left_pad_position_ids(msk_b), use_cache=False)
    finally:
        for h in handles:
            h.remove()
        tok.padding_side = old_p
    assert_zero_runtime_hooks(model)
    return {"x_in": x_in, "m_out": m_out, "h_final": h_fin}


@torch.no_grad()
def _native_bf16_mlp_forward(
    x_f32: torch.Tensor,
    W_g: torch.Tensor,
    W_u: torch.Tensor,
    W_d: torch.Tensor,
    perm_down: Optional[torch.Tensor] = None,
    subset_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x_bf = x_f32.to(dtype=W_g.dtype, device=W_g.device)
    if subset_idx is not None:
        act = F.silu(F.linear(x_bf, W_g[subset_idx])) * F.linear(x_bf, W_u[subset_idx])
        return F.linear(act, W_d[:, subset_idx]).float()
    act = F.silu(F.linear(x_bf, W_g)) * F.linear(x_bf, W_u)
    if perm_down is not None:
        act = act[:, perm_down]
    return F.linear(act, W_d).float()


@torch.no_grad()
def _real_norm_margin_from_rows(
    h_raw: torch.Tensor,
    gamma_w: torch.Tensor,
    eps: float,
    e_ans: torch.Tensor,
    e_foil: torch.Tensor,
) -> torch.Tensor:
    """Exact RMSNorm + LM-head margin using only `(e_ans, e_foil)` rows (works even after Donor 7B is unloaded!)."""
    hf = h_raw.float()
    rms = torch.sqrt(torch.mean(hf * hf, dim=-1, keepdim=True) + float(eps))
    hn = (hf / rms) * gamma_w.float().unsqueeze(0)
    return torch.sum(hn * (e_ans.float() - e_foil.float()), dim=-1)


@torch.no_grad()
def run_gate_g1_whole_mlp(
    model_A: nn.Module,
    tok: Any,
    device: torch.device,
    layer_idx: int,
    disc_items: Sequence[Dict[str, Any]],
    disc_pairs: Sequence[Tuple[int, int]],
    eval_items: Sequence[Dict[str, Any]],
    eval_pairs: Sequence[Tuple[int, int]],
    prompt_key: str,
    ans_tok_key: str,
    foil_tok_key: str,
    exact_match_fn: Callable[[str, Dict[str, Any]], bool],
    G_A_raw: np.ndarray,
    X_cal_pos_gt0_u16: np.ndarray,
    v_shared: int,
    max_new_tokens: int = 5,
    ranks: Sequence[int] = (16, 32, 64, 128, 256),
    q_list: Sequence[int] = (16, 32, 64),
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Complete Astra + Fable Gate G1 Donor Self-Test for WHOLE verbatim MLP at `layer_idx` (7B ONLY!).
    STRICT ANTI-LEAKAGE: ALL `(best_q, best_r_clamped, best_r_hom)` are selected strictly on `disc_items`
    & `disc_pairs` and frozen BEFORE scoring `eval_items`!
    """
    layers = get_decoder_layers(model_A)
    mlp = layers[layer_idx].mlp
    Wg, Wu, Wd = mlp.gate_proj.weight.detach(), mlp.up_proj.weight.detach(), mlp.down_proj.weight.detach()
    lm_w = model_A.lm_head.weight[:v_shared].detach()
    gamma_A_t = model_A.model.norm.weight.detach().float()
    eps_A = float(getattr(model_A.model.norm, "variance_epsilon", getattr(model_A.model.norm, "eps", 1e-6)))
    D_A = int(Wg.shape[1])

    io_disc = _extract_mlp_io_at_last_token(model_A, tok, device, [it[prompt_key] for it in disc_items], layer_idx, batch_size)
    io_eval = _extract_mlp_io_at_last_token(model_A, tok, device, [it[prompt_key] for it in eval_items], layer_idx, batch_size)

    # Closed-form demand matrix G on 512 generic pos>0 calibration tokens (ZERO backprop!)
    rng = np.random.default_rng(SEED + 31 * layer_idx)
    n_j = min(512, len(X_cal_pos_gt0_u16))
    X_cal_j = torch.from_numpy(bf16_to_f64_np(X_cal_pos_gt0_u16[rng.choice(len(X_cal_pos_gt0_u16), size=n_j, replace=False)]).astype(np.float32)).to(device)
    L_F_np, f_ret = factor_psd_top_k(G_A_raw, k_max=128)
    L_F_t = torch.from_numpy(L_F_np.astype(np.float32)).to(device)
    G_demand = compute_mlp_demand_matrix_closed_form(Wg, Wu, Wd, X_cal_j, L_F_t, chunk_size=32)

    n_cov = min(8192, len(X_cal_pos_gt0_u16))
    X_cov = bf16_to_f64_np(X_cal_pos_gt0_u16[:n_cov])
    mu_x_gen = X_cov.mean(axis=0)
    Xc_gen = X_cov - mu_x_gen[None, :]
    Sigma_x_gen = _sym((Xc_gen.T @ Xc_gen) / float(n_cov))

    x_disc = io_disc["x_in"].astype(np.float64)
    mu_x_disc = x_disc.mean(axis=0)
    xc_disc = x_disc - mu_x_disc[None, :]
    Sigma_comb = _sym(0.5 * Sigma_x_gen + 0.5 * ((xc_disc.T @ xc_disc) / max(float(len(xc_disc)), 1.0)))

    eg, Vg = sla.eigh(G_demand)
    eg = np.maximum(eg, 0.0)
    kg = eg > max(float(np.max(eg)) * 1e-8, 1e-18)
    G_half = (Vg[:, kg] * np.sqrt(eg[kg])[None, :]) @ Vg[:, kg].T
    G_pinv_half = (Vg[:, kg] * (1.0 / np.sqrt(eg[kg]))[None, :]) @ Vg[:, kg].T

    ev_d, Q_d = sla.eigh(_sym(G_half @ Sigma_comb @ G_half))
    Q_d = Q_d[:, np.argsort(ev_d)[::-1]]
    r_max = min(int(max(ranks)), int(np.sum(kg)))
    D_bank_raw = G_pinv_half @ Q_d[:, :r_max]
    U_mean_dir = (mu_x_disc / (np.linalg.norm(mu_x_disc) + 1e-12))[:, None]
    U_hom_full, _ = np.linalg.qr(np.concatenate([U_mean_dir, D_bank_raw], axis=1), mode="reduced")
    U_hom_full = U_hom_full[:, :r_max]

    m_disc = io_disc["m_out"].astype(np.float64)
    mc_disc = m_disc - m_disc.mean(axis=0, keepdims=True)
    _, _, Vt_m = sla.svd(mc_disc, full_matrices=False)
    V_full_q = Vt_m[: min(max(q_list), Vt_m.shape[0])].T
    m_mean_disc_t = torch.from_numpy(m_disc.mean(axis=0).astype(np.float32)).to(device)
    mu_gen_t = torch.from_numpy(mu_x_gen.astype(np.float32)).to(device)

    def _pack_split(io_sp: Dict[str, np.ndarray], items_sp: Sequence[Dict[str, Any]], pairs_sp: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
        x_t = torch.from_numpy(io_sp["x_in"]).to(device=device, dtype=torch.float32)
        m_t = torch.from_numpy(io_sp["m_out"]).to(device=device, dtype=torch.float32)
        hf_t = torch.from_numpy(io_sp["h_final"]).to(device=device, dtype=torch.float32)
        hc_t = hf_t - m_t
        paired_idx = np.array([i for i, it in enumerate(items_sp) if it.get(foil_tok_key) is not None], dtype=np.int64)
        paired_mask = np.array([it.get(foil_tok_key) is not None for it in items_sp], dtype=bool)
        p_t = torch.tensor(paired_idx, device=device, dtype=torch.long)
        a_ids_all = torch.tensor([int(it[ans_tok_key]) for it in items_sp], device=device, dtype=torch.long)
        a_ids_p = a_ids_all[p_t]
        f_ids_p = torch.tensor([int(items_sp[i][foil_tok_key]) for i in paired_idx], device=device, dtype=torch.long)
        e_a = lm_w[a_ids_p].float() if len(paired_idx) > 0 else torch.empty((0, D_A), device=device)
        e_f = lm_w[f_ids_p].float() if len(paired_idx) > 0 else torch.empty((0, D_A), device=device)
        mi = _real_norm_margin_from_rows(hf_t[p_t], gamma_A_t, eps_A, e_a, e_f) if len(paired_idx) > 0 else torch.empty(0, device=device)
        ma = _real_norm_margin_from_rows(hc_t[p_t] + m_mean_disc_t.unsqueeze(0), gamma_A_t, eps_A, e_a, e_f) if len(paired_idx) > 0 else torch.empty(0, device=device)
        c_tr = (mi - ma).cpu().double().numpy() if len(paired_idx) > 0 else np.empty((0,), dtype=np.float64)
        s_i = torch.tensor([p[0] for p in pairs_sp], device=device, dtype=torch.long) if pairs_sp else torch.empty(0, device=device, dtype=torch.long)
        b_i = torch.tensor([p[1] for p in pairs_sp], device=device, dtype=torch.long) if pairs_sp else torch.empty(0, device=device, dtype=torch.long)
        e_as = lm_w[a_ids_all[s_i]].float() if len(pairs_sp) > 0 else torch.empty((0, D_A), device=device)
        e_fb = lm_w[a_ids_all[b_i]].float() if len(pairs_sp) > 0 else torch.empty((0, D_A), device=device)
        d_rf = (
            _real_norm_margin_from_rows(hc_t[b_i] + m_t[s_i], gamma_A_t, eps_A, e_as, e_fb)
            - _real_norm_margin_from_rows(hc_t[b_i] + m_t[b_i], gamma_A_t, eps_A, e_as, e_fb)
        ).cpu().double().numpy() if len(pairs_sp) > 0 else np.empty((0,), dtype=np.float64)
        return {
            "x_t": x_t, "m_t": m_t, "hf_t": hf_t, "hc_t": hc_t,
            "paired_idx": paired_idx, "paired_mask": paired_mask, "p_t": p_t,
            "a_ids": a_ids_all, "e_a": e_a, "e_f": e_f,
            "mi": mi, "ma": ma, "c_true": c_tr, "s_i": s_i, "b_i": b_i, "e_as": e_as, "e_fb": e_fb, "d_ref": d_rf,
        }

    sp_disc = _pack_split(io_disc, disc_items, disc_pairs)
    sp_eval = _pack_split(io_eval, eval_items, eval_pairs)

    def _score_m_hat(sp: Dict[str, Any], m_hat: torch.Tensor) -> Dict[str, Any]:
        c_hat = (
            _real_norm_margin_from_rows(sp["hc_t"][sp["p_t"]] + m_hat[sp["p_t"]], gamma_A_t, eps_A, sp["e_a"], sp["e_f"]) - sp["ma"]
        ).cpu().double().numpy() if len(sp["p_t"]) > 0 else np.empty((0,), dtype=np.float64)
        d_hat = (
            _real_norm_margin_from_rows(sp["hc_t"][sp["b_i"]] + m_hat[sp["s_i"]], gamma_A_t, eps_A, sp["e_as"], sp["e_fb"])
            - _real_norm_margin_from_rows(sp["hc_t"][sp["b_i"]] + m_hat[sp["b_i"]], gamma_A_t, eps_A, sp["e_as"], sp["e_fb"])
        ).cpu().double().numpy() if len(sp["s_i"]) > 0 else np.empty((0,), dtype=np.float64)
        return _compute_safe_q_and_r2(d_hat, sp["d_ref"], c_hat, sp["c_true"])

    # 1) STRICTLY ON `disc`: Select frozen `best_q`, `best_r_clamped`, `best_r_hom`!
    best_q = min(int(q_list[-1]), V_full_q.shape[1])
    best_q_score = -1e9
    q_sweep_disc: Dict[str, Any] = {}
    q_sweep_eval: Dict[str, Any] = {}
    for q in q_list:
        q_eff = min(int(q), V_full_q.shape[1])
        Vq = torch.from_numpy(V_full_q[:, :q_eff].astype(np.float32)).to(device)
        # Note: pure projected write V_q V_q^T (m - m_mean) + m_mean (NO access to per-item complement!)
        m_proj_d = m_mean_disc_t.unsqueeze(0) + ((sp_disc["m_t"] - m_mean_disc_t.unsqueeze(0)) @ Vq) @ Vq.T
        m_proj_e = m_mean_disc_t.unsqueeze(0) + ((sp_eval["m_t"] - m_mean_disc_t.unsqueeze(0)) @ Vq) @ Vq.T
        sd = _score_m_hat(sp_disc, m_proj_d)
        se = _score_m_hat(sp_eval, m_proj_e)
        q_sweep_disc[f"q_{q_eff}"] = {"Q_causal": sd["Q_causal"], "R2_c": sd["R2_c"]}
        q_sweep_eval[f"q_{q_eff}"] = {"Q_causal": se["Q_causal"], "R2_c": se["R2_c"]}
        sc_v = sd["Q_causal"] if sd["Q_causal"] is not None else -1e9
        if sc_v > best_q_score:
            best_q_score = sc_v
            best_q = q_eff

    best_r_clamped = min(int(ranks[-1]), r_max)
    best_qc_disc = -1e9
    best_r_hom = min(int(ranks[-1]), r_max)
    best_qh_disc = -1e9
    r_sweep_eval: Dict[str, Any] = {}
    r_sweep_disc: Dict[str, Any] = {}

    for r in ranks:
        r_eff = min(int(r), r_max)
        Q_r = Q_d[:, :r_eff]
        D_r = torch.from_numpy((G_pinv_half @ Q_r).astype(np.float32)).to(device)
        E_don = torch.from_numpy((Q_r.T @ G_half).astype(np.float32)).to(device)
        U_hr = torch.from_numpy(U_hom_full[:, :r_eff].astype(np.float32)).to(device)
        mu_perp = mu_gen_t - (mu_gen_t @ E_don.T) @ D_r.T

        # Selection strictly on `disc`
        xd_cl = mu_perp.unsqueeze(0) + (sp_disc["x_t"] @ E_don.T) @ D_r.T
        xd_hm = (sp_disc["x_t"] @ U_hr) @ U_hr.T
        sc_dc = _score_m_hat(sp_disc, _native_bf16_mlp_forward(xd_cl, Wg, Wu, Wd))
        sc_dh = _score_m_hat(sp_disc, _native_bf16_mlp_forward(xd_hm, Wg, Wu, Wd))
        r_sweep_disc[f"r_{r_eff}"] = {"mean_clamped_Q": sc_dc["Q_causal"], "hom_Q": sc_dh["Q_causal"]}
        if sc_dc["Q_causal"] is not None and sc_dc["Q_causal"] > best_qc_disc:
            best_qc_disc, best_r_clamped = sc_dc["Q_causal"], r_eff
        if sc_dh["Q_causal"] is not None and sc_dh["Q_causal"] > best_qh_disc:
            best_qh_disc, best_r_hom = sc_dh["Q_causal"], r_eff

        # Read-only eval diagnostic sweep
        xe_cl = mu_perp.unsqueeze(0) + (sp_eval["x_t"] @ E_don.T) @ D_r.T
        # Stress ONLY the affine offset `mu_perp` by 2% CV jitter (matches realizable `a*` path!)
        g_j = torch.Generator(device="cpu").manual_seed(SEED + r_eff)
        jit = (1.0 + 0.02 * torch.randn((len(eval_items), 1), generator=g_j)).to(device=device, dtype=torch.float32)
        xe_jit = jit * mu_perp.unsqueeze(0) + (sp_eval["x_t"] @ E_don.T) @ D_r.T
        xe_hm = (sp_eval["x_t"] @ U_hr) @ U_hr.T

        sc_ec = _score_m_hat(sp_eval, _native_bf16_mlp_forward(xe_cl, Wg, Wu, Wd))
        sc_ej = _score_m_hat(sp_eval, _native_bf16_mlp_forward(xe_jit, Wg, Wu, Wd))
        sc_eh = _score_m_hat(sp_eval, _native_bf16_mlp_forward(xe_hm, Wg, Wu, Wd))

        # Orthogonal-complement substitution check at fixed z
        qc_orth = None
        if len(sp_eval["s_i"]) > 0:
            x_cs = xe_hm[sp_eval["s_i"]] + (sp_eval["x_t"][sp_eval["b_i"]] - xe_hm[sp_eval["b_i"]])
            m_cs = _native_bf16_mlp_forward(x_cs, Wg, Wu, Wd)
            d_cs = (
                _real_norm_margin_from_rows(sp_eval["hc_t"][sp_eval["b_i"]] + m_cs, gamma_A_t, eps_A, sp_eval["e_as"], sp_eval["e_fb"])
                - _real_norm_margin_from_rows(sp_eval["hc_t"][sp_eval["b_i"]] + sp_eval["m_t"][sp_eval["b_i"]], gamma_A_t, eps_A, sp_eval["e_as"], sp_eval["e_fb"])
            ).cpu().double().numpy()
            qc_orth = _compute_safe_q_and_r2(d_cs, sp_eval["d_ref"], sp_eval["c_true"], sp_eval["c_true"])["Q_causal"]

        r_sweep_eval[f"r_{r_eff}"] = {
            "mean_clamped_Q_causal": sc_ec["Q_causal"],
            "mean_clamped_R2_c": sc_ec["R2_c"],
            "mean_clamped_2pct_offset_jitter_Q_causal": sc_ej["Q_causal"],
            "mean_clamped_2pct_offset_jitter_R2_c": sc_ej["R2_c"],
            "homogeneous_Q_causal": sc_eh["Q_causal"],
            "homogeneous_R2_c": sc_eh["R2_c"],
            "orth_complement_sub_Q_causal": qc_orth,
        }

    # Frozen winner on eval + cached-act 20-draw SHUF-pi!
    Q_best = Q_d[:, :best_r_clamped]
    D_best = torch.from_numpy((G_pinv_half @ Q_best).astype(np.float32)).to(device)
    E_best = torch.from_numpy((Q_best.T @ G_half).astype(np.float32)).to(device)
    mu_perp_best = mu_gen_t - (mu_gen_t @ E_best.T) @ D_best.T
    x_best_eval = mu_perp_best.unsqueeze(0) + (sp_eval["x_t"] @ E_best.T) @ D_best.T
    # Cache BF16 SwiGLU activations ONCE for 20-draw SHUF-pi (20x faster!)
    x_bf = x_best_eval.to(dtype=Wg.dtype)
    act_best = F.silu(F.linear(x_bf, Wg)) * F.linear(x_bf, Wu)
    shuf_qc: List[float] = []
    shuf_r2: List[float] = []
    M_mlp = int(Wg.shape[0])
    for draw_i in range(20):
        gp = torch.Generator(device="cpu").manual_seed(SEED + 9000 + draw_i)
        m_sh = F.linear(act_best[:, torch.randperm(M_mlp, generator=gp).to(device)], Wd).float()
        sc_p = _score_m_hat(sp_eval, m_sh)
        if sc_p["Q_causal"] is not None:
            shuf_qc.append(sc_p["Q_causal"])
        if sc_p["R2_c"] is not None:
            shuf_r2.append(sc_p["R2_c"])
    del act_best

    # Ordinary-generation ablation-to-restoration with uniform repetition_penalty=1.0 & shared-vocab mask!
    prompts_eval = [it[prompt_key] for it in eval_items]
    gens_intact = _greedy_generate_clean(model_A, tok, device, prompts_eval, max_new_tokens=max_new_tokens, batch_size=batch_size, v_shared=v_shared)
    em_intact = float(np.mean([int(exact_match_fn(g, it)) for g, it in zip(gens_intact, eval_items)]))

    lproc = [_MaskPaddedVocabLogitsProcessor(v_shared)]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    U_hr_best = torch.from_numpy(U_hom_full[:, :best_r_hom].astype(np.float32)).to(device)

    def _gen_mlp_override(mode: str) -> float:
        old_p = tok.padding_side
        tok.padding_side = "left"
        h_in, h_out = None, None
        saved_x: List[torch.Tensor] = []
        txts: List[str] = []
        try:
            def _cap(_m, _i, o):
                saved_x.clear()
                saved_x.append((o[0] if isinstance(o, tuple) else o).detach())

            def _rep(_m, _i, out):
                o = out[0] if isinstance(out, tuple) else out
                o2 = o.clone()
                if mode == "ablate":
                    o2[:, -1, :] = m_mean_disc_t.to(dtype=o2.dtype)
                elif mode == "clamped":
                    xl = saved_x[0][:, -1, :].float()
                    xt = mu_perp_best.unsqueeze(0) + (xl @ E_best.T) @ D_best.T
                    o2[:, -1, :] = _native_bf16_mlp_forward(xt, Wg, Wu, Wd).to(dtype=o2.dtype)
                elif mode == "homogeneous":
                    xl = saved_x[0][:, -1, :].float()
                    xt = (xl @ U_hr_best) @ U_hr_best.T
                    o2[:, -1, :] = _native_bf16_mlp_forward(xt, Wg, Wu, Wd).to(dtype=o2.dtype)
                return (o2,) + out[1:] if isinstance(out, tuple) else o2

            h_in = layers[layer_idx].post_attention_layernorm.register_forward_hook(_cap)
            h_out = layers[layer_idx].mlp.register_forward_hook(_rep)
            for i0 in range(0, len(prompts_eval), batch_size):
                i1 = min(i0 + batch_size, len(prompts_eval))
                enc = tok(prompts_eval[i0:i1], return_tensors="pt", padding=True, add_special_tokens=True)
                ids = enc["input_ids"].to(device)
                gen = model_A.generate(
                    input_ids=ids, attention_mask=enc["attention_mask"].to(device),
                    max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0, use_cache=True,
                    pad_token_id=pad_id, logits_processor=lproc,
                )
                plen = ids.shape[1]
                for bi in range(i1 - i0):
                    txts.append(tok.decode(gen[bi, plen:], skip_special_tokens=True))
        finally:
            if h_in is not None:
                h_in.remove()
            if h_out is not None:
                h_out.remove()
            tok.padding_side = old_p
        assert_zero_runtime_hooks(model_A)
        return float(np.mean([int(exact_match_fn(g, it)) for g, it in zip(txts, eval_items)]))

    em_ablated = _gen_mlp_override("ablate")
    em_rest_clamped = _gen_mlp_override("clamped")
    em_rest_hom = _gen_mlp_override("homogeneous")
    ablation_gap = em_intact - em_ablated
    rec_gen_clamped = float((em_rest_clamped - em_ablated) / ablation_gap) if ablation_gap > 1e-6 else 0.0
    rec_gen_hom = float((em_rest_hom - em_ablated) / ablation_gap) if ablation_gap > 1e-6 else 0.0

    # Cancellation ratio chi on BOTH clamped inputs AND native inputs
    x_cal_clamped = mu_perp_best.unsqueeze(0) + (X_cal_j @ E_best.T) @ D_best.T
    Z_cal_code = ((X_cal_j.cpu().double().numpy() - mu_x_gen[None, :]) @ (Q_best.T @ G_half).T)
    chi_clamped = compute_cancellation_ratio_chi(Wg, Wu, Wd, x_cal_clamped, Z_cal_code, L_F_t)
    chi_native = compute_cancellation_ratio_chi(Wg, Wu, Wd, X_cal_j, Z_cal_code, L_F_t)

    # STRICT JOINT UNROUNDED G1 PASS PREDICATES!
    mean_margin_intact = float(sp_eval["mi"].mean().item())
    mean_c_true = float(np.mean(sp_eval["c_true"]))
    ld_share_eval = float(mean_c_true / (abs(mean_margin_intact) + 1e-12))
    win_eval = r_sweep_eval[f"r_{best_r_clamped}"]
    q_win = win_eval["mean_clamped_Q_causal"]
    r2_win = win_eval["mean_clamped_R2_c"]
    shuf_mean_q = float(np.mean(shuf_qc)) if shuf_qc else 1.0

    g1_drop_pass = bool((100.0 * ablation_gap >= 20.0) or (ld_share_eval >= 0.30))
    g1_fid_pass = bool((q_win is not None and q_win >= 0.80) and (r2_win is not None and r2_win >= 0.80) and (shuf_mean_q <= 0.10))
    g1_gen_pass = bool((rec_gen_clamped >= 0.80) and (ablation_gap >= 0.05))
    g1_pass = bool(g1_drop_pass and g1_fid_pass and g1_gen_pass)

    # Width-compression ladder ONLY runs if whole-MLP G1 PASSES!
    ladder_report: Optional[Dict[str, Any]] = None
    if g1_pass:
        A_cal = (F.silu(F.linear(x_cal_clamped.to(Wg.dtype), Wg)) * F.linear(x_cal_clamped.to(Wu.dtype), Wu)).float()
        B_proj = L_F_t.T @ Wd.float()
        lad = run_width_compression_ladder(A_cal, B_proj, ladder=(M_mlp, 8192, 4096, 2048, 1024, 512, 256))
        lad_eval: Dict[str, Any] = {}
        for m_rung in lad["rungs"]:
            idx_c = torch.from_numpy(lad["cancel_subsets"][m_rung]).to(device)
            idx_w = torch.from_numpy(lad["wanda_subsets"][m_rung]).to(device)
            sc_c = _score_m_hat(sp_eval, _native_bf16_mlp_forward(x_best_eval, Wg, Wu, Wd, subset_idx=idx_c))
            sc_w = _score_m_hat(sp_eval, _native_bf16_mlp_forward(x_best_eval, Wg, Wu, Wd, subset_idx=idx_w))
            lad_eval[str(m_rung)] = {
                "cal_rel_err_cancel": lad["cancel_rel_err"][str(m_rung)],
                "cal_rel_err_wanda": lad["wanda_rel_err"][str(m_rung)],
                "eval_Q_causal_cancel": sc_c["Q_causal"],
                "eval_Q_causal_wanda": sc_w["Q_causal"],
            }
        ladder_report = lad_eval

    return {
        "report": {
            "component": f"L{layer_idx}_MLP_whole",
            "layer": int(layer_idx),
            "n_disc": len(disc_items), "n_disc_pairs": len(disc_pairs),
            "n_eval": len(eval_items), "n_eval_pairs": len(eval_pairs),
            "fisher_top128_retained_frac": round(f_ret, 6),
            "mean_margin_intact": round(mean_margin_intact, 4),
            "mean_c_true": round(mean_c_true, 4),
            "ld_share_eval": round(ld_share_eval, 4),
            "em_intact": round(em_intact, 4),
            "em_ablated": round(em_ablated, 4),
            "ablation_drop_pp": round(100.0 * ablation_gap, 2),
            "em_restored_clamped": round(em_rest_clamped, 4),
            "rec_gen_clamped": round(rec_gen_clamped, 4),
            "em_restored_homogeneous": round(em_rest_hom, 4),
            "rec_gen_homogeneous": round(rec_gen_hom, 4),
            "frozen_from_disc": {"best_r_clamped": int(best_r_clamped), "best_r_hom": int(best_r_hom), "best_q_output": int(best_q)},
            "q_sweep_disc": q_sweep_disc,
            "q_sweep_eval": q_sweep_eval,
            "r_sweep_disc": r_sweep_disc,
            "r_sweep_eval": r_sweep_eval,
            "shuf_pi_20draws": {
                "mean_Q_causal": round(shuf_mean_q, 4) if shuf_qc else None,
                "max_Q_causal": round(float(np.max(shuf_qc)), 4) if shuf_qc else None,
                "mean_R2_c": round(float(np.mean(shuf_r2)), 4) if shuf_r2 else None,
            },
            "cancellation_chi_clamped": chi_clamped,
            "cancellation_chi_native": chi_native,
            "width_compression_ladder": ladder_report,
            "g1_checks": {
                "drop_pass": g1_drop_pass,
                "fidelity_pass": g1_fid_pass,
                "gen_recovery_pass": g1_gen_pass,
                "G1_PASS": g1_pass,
            },
        },
        # Compact CPU payload for Gate G2 + Donor Self-Test Replay on G2's `D_r` basis!
        "g2_payload": {
            "kind": "whole_mlp",
            "layer": int(layer_idx),
            "G_demand": G_demand,
            "U_hom_best": U_hom_full[:, :best_r_hom].copy(),
            "V_best": V_full_q[:, :best_q].copy(),
            "Wg_bf16": Wg.detach().to(dtype=torch.bfloat16).cpu().clone(),
            "Wu_bf16": Wu.detach().to(dtype=torch.bfloat16).cpu().clone(),
            "Wd_bf16": Wd.detach().to(dtype=torch.bfloat16).cpu().clone(),
            "m_mean_disc_f32": m_disc.mean(axis=0).astype(np.float32),
            "x_eval_A_f32": io_eval["x_in"].copy(),
            "m_eval_A_f32": io_eval["m_out"].copy(),
            "hc_eval_A_f32": sp_eval["hc_t"].cpu().numpy().astype(np.float32),
            "ma_eval_A_f32": sp_eval["ma"].cpu().numpy().astype(np.float32),
            "e_a_eval_A_f32": sp_eval["e_a"].cpu().numpy().astype(np.float32),
            "e_f_eval_A_f32": sp_eval["e_f"].cpu().numpy().astype(np.float32),
            "e_as_eval_A_f32": sp_eval["e_as"].cpu().numpy().astype(np.float32),
            "e_fb_eval_A_f32": sp_eval["e_fb"].cpu().numpy().astype(np.float32),
            "gamma_A_f32": gamma_A_t.cpu().numpy().astype(np.float32),
            "eps_A": float(eps_A),
            "c_true_eval": sp_eval["c_true"].copy(),
            "paired_mask_eval": sp_eval["paired_mask"].copy(),
            "paired_idx_eval": sp_eval["paired_idx"].copy(),
            "d_ref_swap": sp_eval["d_ref"].copy(),
            "eval_pairs": list(eval_pairs),
            "frozen_r_disc": int(best_r_clamped),
        },
    }


def _forward_single_attn_head_z_and_out(
    X_seq: torch.Tensor,
    W_q_h: torch.Tensor,
    b_q_h: Optional[torch.Tensor],
    W_k_kv: torch.Tensor,
    b_k_kv: Optional[torch.Tensor],
    W_v_kv: torch.Tensor,
    b_v_kv: Optional[torch.Tensor],
    W_o_h: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Exact single-head causal RoPE attention at last token `t = T-1`; returns `(z_h (d_h,), out (D,), alpha_pos0)`."""
    X = X_seq.to(device=W_q_h.device, dtype=W_q_h.dtype)
    T = int(X.shape[0])
    d_h = int(W_q_h.shape[0])
    half = d_h // 2
    q = F.linear(X[-1:], W_q_h, b_q_h).float().squeeze(0)
    k = F.linear(X, W_k_kv, b_k_kv).float()
    v = F.linear(X, W_v_kv, b_v_kv).float()
    c, s = cos_t[:T], sin_t[:T]
    q_rot = torch.cat([q[:half] * c[-1] - q[half:] * s[-1], q[half:] * c[-1] + q[:half] * s[-1]], dim=-1)
    k_rot = torch.cat([k[:, :half] * c - k[:, half:] * s, k[:, half:] * c + k[:, :half] * s], dim=-1)
    alpha = F.softmax((k_rot @ q_rot) / math.sqrt(float(d_h)), dim=0)
    z_h = (alpha @ v).to(dtype=W_o_h.dtype)
    out = F.linear(z_h, W_o_h).float()
    return z_h.float(), out, float(alpha[0].item())


@torch.no_grad()
def run_gate_g1_late_attn_heads(
    model_A: nn.Module,
    tok: Any,
    device: torch.device,
    selected_heads: Sequence[Tuple[int, int]],
    disc_items: Sequence[Dict[str, Any]],
    eval_items: Sequence[Dict[str, Any]],
    prompt_key: str,
    ans_tok_key: str,
    foil_tok_key: str,
    exact_match_fn: Callable[[str, Dict[str, Any]], bool],
    G_A_raw: np.ndarray,
    v_shared: int,
    max_new_tokens: int = 4,
    ranks: Sequence[int] = (16, 32, 64, 128, 256),
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Complete Fable + Astra Gate G1 Donor Self-Test for Late Attention Heads (7B ONLY!):
      - Preserves native `pos == 0` sink (`X_c[0] = X_s[0]`, clamping ONLY `pos > 0` tokens!) and logs sink mass `alpha[0]`!
      - Excludes `pos == 0` from closed-form demand `G_l`.
      - Selects `best_r` STRICTLY on `disc_items` (zero eval leakage!).
      - Caches `z_h` ONCE for 20-draw `SHUF-pi` (20x speedup!).
      - Runs ACTUAL LIVE HEAD RESTORATION during greedy generation at the exact same head sites as ablation!
    """
    assert len(selected_heads) > 0
    layers = get_decoder_layers(model_A)
    lm_w = model_A.lm_head.weight[:v_shared].detach()
    gamma_A_t = model_A.model.norm.weight.detach().float()
    eps_A = float(getattr(model_A.model.norm, "variance_epsilon", getattr(model_A.model.norm, "eps", 1e-6)))
    rope_theta = get_config_rope_theta(model_A.config)
    D_A = int(model_A.config.hidden_size)
    d_h = int(layers[0].self_attn.head_dim)
    g_A = int(layers[0].self_attn.num_key_value_groups)

    heads_by_layer: Dict[int, List[int]] = {}
    for li, hi in selected_heads:
        heads_by_layer.setdefault(int(li), []).append(int(hi))
    unique_layers = sorted(heads_by_layer.keys())

    def _extract(items_sub: Sequence[Dict[str, Any]]) -> Tuple[Dict[int, List[torch.Tensor]], np.ndarray, np.ndarray, np.ndarray]:
        assert_zero_runtime_hooks(model_A)
        prompts = [it[prompt_key] for it in items_sub]
        seqs_by_l: Dict[int, List[torch.Tensor]] = {li: [] for li in unique_layers}
        sum_out = np.zeros((len(items_sub), D_A), dtype=np.float32)
        h_fin = np.zeros((len(items_sub), D_A), dtype=np.float32)
        first_tok_ids = np.zeros(len(items_sub), dtype=np.int32)
        old_p = tok.padding_side
        tok.padding_side = "left"
        handles = []
        cur_mask: List[torch.Tensor] = []
        sl = [0, 0]
        try:
            def _mk_in(li: int):
                def _h(_m, _i, out):
                    x = (out[0] if isinstance(out, tuple) else out).detach()
                    msk = cur_mask[0]
                    for bi in range(x.shape[0]):
                        nz = torch.nonzero(msk[bi].bool(), as_tuple=False).squeeze(-1)
                        seqs_by_l[li].append(x[bi, nz].cpu().to(dtype=torch.bfloat16))
                return _h

            def _mk_z(li: int):
                Wo = layers[li].self_attn.o_proj.weight.detach().float()
                hs = heads_by_layer[li]
                def _h(_m, inp):
                    zl = inp[0][:, -1, :].detach().float()
                    acc = torch.zeros((zl.shape[0], D_A), device=zl.device, dtype=torch.float32)
                    for hi in hs:
                        sh = slice(hi * d_h, (hi + 1) * d_h)
                        acc += zl[:, sh] @ Wo[:, sh].T
                    sum_out[sl[0] : sl[1]] += acc.cpu().numpy()
                return _h

            for li in unique_layers:
                handles.append(layers[li].input_layernorm.register_forward_hook(_mk_in(li)))
                handles.append(layers[li].self_attn.o_proj.register_forward_pre_hook(_mk_z(li)))
            handles.append(model_A.model.norm.register_forward_pre_hook(
                lambda _m, inp: h_fin.__setitem__(slice(sl[0], sl[1]), inp[0][:, -1, :].detach().float().cpu().numpy())
            ))
            for i0 in range(0, len(prompts), batch_size):
                i1 = min(i0 + batch_size, len(prompts))
                sl[0], sl[1] = i0, i1
                enc = tok(prompts[i0:i1], return_tensors="pt", padding=True, add_special_tokens=True)
                ids = enc["input_ids"].to(device)
                msk = enc["attention_mask"].to(device)
                for bi in range(i1 - i0):
                    nz = torch.nonzero(msk[bi].bool(), as_tuple=False)
                    first_tok_ids[i0 + bi] = int(ids[bi, int(nz[0, 0].item())].item())
                cur_mask.clear()
                cur_mask.append(msk)
                model_A(input_ids=ids, attention_mask=msk, position_ids=_left_pad_position_ids(msk), use_cache=False)
        finally:
            for h in handles:
                h.remove()
            tok.padding_side = old_p
        assert_zero_runtime_hooks(model_A)
        return seqs_by_l, sum_out, h_fin, first_tok_ids

    seqs_disc, head_out_disc, h_fin_disc, ft_disc = _extract(disc_items)
    seqs_eval, head_out_eval, h_fin_eval, ft_eval = _extract(eval_items)
    max_T = max(max(int(s.shape[0]) for s in seqs_disc[unique_layers[0]]), max(int(s.shape[0]) for s in seqs_eval[unique_layers[0]]))
    cos_all, sin_all = _make_rope_cos_sin(d_h, rope_theta, max_T + 8, device)

    L_F_np, f_ret = factor_psd_top_k(G_A_raw, k_max=64)
    L_F_t = torch.from_numpy(L_F_np.astype(np.float32)).to(device)

    head_weights: List[Dict[str, Any]] = []
    G_by_layer: Dict[int, np.ndarray] = {}
    mu_by_layer: Dict[int, np.ndarray] = {}
    mu_pos0_by_layer: Dict[int, np.ndarray] = {}
    U_by_layer: Dict[int, np.ndarray] = {}
    r_max = int(max(ranks))

    for li in unique_layers:
        attn = layers[li].self_attn
        G_l = np.zeros((D_A, D_A), dtype=np.float64)
        disc_s = [s.to(device=device, dtype=torch.float32) for s in seqs_disc[li][:48]]
        pos_gt0_toks = np.concatenate([s[1:].float().numpy() for s in seqs_disc[li] if len(s) > 1], axis=0).astype(np.float64)
        mu_l = pos_gt0_toks.mean(axis=0)
        xc_l = pos_gt0_toks - mu_l[None, :]
        Sigma_l = _sym((xc_l.T @ xc_l) / max(float(len(xc_l)), 1.0))
        mu_by_layer[li] = mu_l
        mu_pos0_by_layer[li] = np.mean([s[0].float().numpy() for s in seqs_disc[li]], axis=0).astype(np.float64)

        for hi in heads_by_layer[li]:
            kvi = hi // g_A
            sq, skv = slice(hi * d_h, (hi + 1) * d_h), slice(kvi * d_h, (kvi + 1) * d_h)
            Wq_h = attn.q_proj.weight.detach()[sq]
            bq_h = attn.q_proj.bias.detach()[sq] if attn.q_proj.bias is not None else None
            Wk_kv = attn.k_proj.weight.detach()[skv]
            bk_kv = attn.k_proj.bias.detach()[skv] if attn.k_proj.bias is not None else None
            Wv_kv = attn.v_proj.weight.detach()[skv]
            bv_kv = attn.v_proj.bias.detach()[skv] if attn.v_proj.bias is not None else None
            Wo_h = attn.o_proj.weight.detach()[:, sq]
            G_l += compute_attn_head_demand_matrix_closed_form(
                Wq_h, bq_h, Wk_kv, bk_kv, Wv_kv, bv_kv, Wo_h, disc_s, L_F_t, rope_theta=rope_theta, exclude_pos0=True
            )
            head_weights.append({
                "layer": li, "head": hi, "kv_head": kvi,
                "Wq": Wq_h.to(torch.bfloat16).cpu().clone(),
                "bq": bq_h.to(torch.bfloat16).cpu().clone() if bq_h is not None else None,
                "Wk": Wk_kv.to(torch.bfloat16).cpu().clone(),
                "bk": bk_kv.to(torch.bfloat16).cpu().clone() if bk_kv is not None else None,
                "Wv": Wv_kv.to(torch.bfloat16).cpu().clone(),
                "bv": bv_kv.to(torch.bfloat16).cpu().clone() if bv_kv is not None else None,
                "Wo": Wo_h.to(torch.bfloat16).cpu().clone(),
            })
        G_by_layer[li] = _sym(G_l)
        eg, Vg = sla.eigh(G_by_layer[li])
        eg = np.maximum(eg, 0.0)
        kg = eg > max(float(np.max(eg)) * 1e-8, 1e-18)
        Gh = (Vg[:, kg] * np.sqrt(eg[kg])[None, :]) @ Vg[:, kg].T
        Gph = (Vg[:, kg] * (1.0 / np.sqrt(eg[kg]))[None, :]) @ Vg[:, kg].T
        ev_d, Q_d = sla.eigh(_sym(Gh @ Sigma_l @ Gh))
        Q_d = Q_d[:, np.argsort(ev_d)[::-1]]
        U_l, _ = np.linalg.qr(Gph @ Q_d[:, : min(r_max, int(np.sum(kg)))], mode="reduced")
        U_by_layer[li] = U_l

    mean_out_disc_t = torch.from_numpy(head_out_disc.mean(axis=0).astype(np.float32)).to(device)

    def _pack_h(items_sp, ho_sp, hf_sp):
        pairs = build_2hop_table_swap_pairs(items_sp)
        hf_t = torch.from_numpy(hf_sp).to(device=device, dtype=torch.float32)
        ho_t = torch.from_numpy(ho_sp).to(device=device, dtype=torch.float32)
        hc_t = hf_t - ho_t
        a_ids = torch.tensor([int(it[ans_tok_key]) for it in items_sp], device=device, dtype=torch.long)
        f_ids = torch.tensor([int(it[foil_tok_key]) for it in items_sp], device=device, dtype=torch.long)
        e_a, e_f = lm_w[a_ids].float(), lm_w[f_ids].float()
        mi = _real_norm_margin_from_rows(hf_t, gamma_A_t, eps_A, e_a, e_f)
        ma = _real_norm_margin_from_rows(hc_t + mean_out_disc_t.unsqueeze(0), gamma_A_t, eps_A, e_a, e_f)
        c_tr = (mi - ma).cpu().double().numpy()
        s_i = torch.tensor([p[0] for p in pairs], device=device, dtype=torch.long)
        b_i = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.long)
        e_as, e_fb = lm_w[a_ids[s_i]].float(), lm_w[a_ids[b_i]].float()
        d_rf = (
            _real_norm_margin_from_rows(hc_t[b_i] + ho_t[s_i], gamma_A_t, eps_A, e_as, e_fb)
            - _real_norm_margin_from_rows(hc_t[b_i] + ho_t[b_i], gamma_A_t, eps_A, e_as, e_fb)
        ).cpu().double().numpy()
        return {"hc_t": hc_t, "hf_t": hf_t, "ho_t": ho_t, "e_a": e_a, "e_f": e_f, "mi": mi, "ma": ma, "c_true": c_tr,
                "paired_mask": np.ones(len(items_sp), dtype=bool), "paired_idx": np.arange(len(items_sp), dtype=np.int64),
                "s_i": s_i, "b_i": b_i, "e_as": e_as, "e_fb": e_fb, "d_ref": d_rf, "pairs": pairs}

    sp_disc = _pack_h(disc_items, head_out_disc, h_fin_disc)
    sp_eval = _pack_h(eval_items, head_out_eval, h_fin_eval)

    def _replay_heads(seqs_sp: Dict[int, List[torch.Tensor]], sp: Dict[str, Any], r_val: int) -> Tuple[Dict[str, Any], List[torch.Tensor], torch.Tensor, float]:
        n_sp = len(seqs_sp[unique_layers[0]])
        out_hat = torch.zeros((n_sp, D_A), device=device, dtype=torch.float32)
        z_by_head: List[torch.Tensor] = []
        sink_masses: List[float] = []
        for hw in head_weights:
            li = hw["layer"]
            U_l = U_by_layer[li]
            Ur = torch.from_numpy(U_l[:, : min(int(r_val), U_l.shape[1])].astype(np.float32)).to(device)
            mu_t = torch.from_numpy(mu_by_layer[li].astype(np.float32)).to(device)
            Wq, Wk, Wv, Wo = hw["Wq"].to(device), hw["Wk"].to(device), hw["Wv"].to(device), hw["Wo"].to(device)
            bq = hw["bq"].to(device) if hw["bq"] is not None else None
            bk = hw["bk"].to(device) if hw["bk"] is not None else None
            bv = hw["bv"].to(device) if hw["bv"] is not None else None
            zh_mat = torch.zeros((n_sp, d_h), device=device, dtype=torch.float32)
            for i in range(n_sp):
                X_s = seqs_sp[li][i].to(device=device, dtype=torch.float32)
                # PRESERVE NATIVE pos == 0 SINK; clamp ONLY pos > 0 tokens!
                X_c = X_s.clone()
                if X_s.shape[0] > 1:
                    X_c[1:] = mu_t.unsqueeze(0) + ((X_s[1:] - mu_t.unsqueeze(0)) @ Ur) @ Ur.T
                zh, oh, a0 = _forward_single_attn_head_z_and_out(X_c, Wq, bq, Wk, bk, Wv, bv, Wo, cos_all, sin_all)
                zh_mat[i] = zh
                out_hat[i] += oh
                sink_masses.append(a0)
            z_by_head.append(zh_mat)
        c_hat = (_real_norm_margin_from_rows(sp["hc_t"] + out_hat, gamma_A_t, eps_A, sp["e_a"], sp["e_f"]) - sp["ma"]).cpu().double().numpy()
        d_hat = (
            _real_norm_margin_from_rows(sp["hc_t"][sp["b_i"]] + out_hat[sp["s_i"]], gamma_A_t, eps_A, sp["e_as"], sp["e_fb"])
            - _real_norm_margin_from_rows(sp["hc_t"][sp["b_i"]] + out_hat[sp["b_i"]], gamma_A_t, eps_A, sp["e_as"], sp["e_fb"])
        ).cpu().double().numpy()
        return _compute_safe_q_and_r2(d_hat, sp["d_ref"], c_hat, sp["c_true"]), z_by_head, out_hat, float(np.mean(sink_masses))

    # Select best_r STRICTLY on `disc`!
    best_r = int(ranks[-1])
    best_r2_disc = -1e9
    r_sweep_disc: Dict[str, Any] = {}
    r_sweep_eval: Dict[str, Any] = {}
    cached_eval_best: Optional[Tuple[List[torch.Tensor], torch.Tensor, float]] = None

    for r in ranks:
        sd, _, _, sm_d = _replay_heads(seqs_disc, sp_disc, int(r))
        r_sweep_disc[f"r_{int(r)}"] = {"Q_causal": sd["Q_causal"], "R2_c": sd["R2_c"], "mean_pos0_sink_mass": round(sm_d, 4)}
        if sd["R2_c"] is not None and sd["R2_c"] > best_r2_disc:
            best_r2_disc, best_r = sd["R2_c"], int(r)

    for r in ranks:
        se, zb, oh, sm_e = _replay_heads(seqs_eval, sp_eval, int(r))
        r_sweep_eval[f"r_{int(r)}"] = {"Q_causal": se["Q_causal"], "R2_c": se["R2_c"], "mean_pos0_sink_mass": round(sm_e, 4)}
        if int(r) == best_r:
            cached_eval_best = (zb, oh, sm_e)
    assert cached_eval_best is not None
    z_best_eval, out_best_eval, sink_mass_eval = cached_eval_best

    # 20-draw SHUF-pi using CACHED `z_h` in EXACT SAME BF16 W_O projection dtype as real head!
    shuf_qc: List[float] = []
    shuf_r2: List[float] = []
    Wo_devs = [hw["Wo"].to(device=device) for hw in head_weights]
    for draw_i in range(20):
        out_sh = torch.zeros((len(eval_items), D_A), device=device, dtype=torch.float32)
        for hw_i, (zh_m, Wo_bf) in enumerate(zip(z_best_eval, Wo_devs)):
            gp = torch.Generator(device="cpu").manual_seed(SEED + 7000 + 131 * hw_i + draw_i)
            out_sh += F.linear(zh_m.to(dtype=Wo_bf.dtype), Wo_bf[:, torch.randperm(d_h, generator=gp).to(device)]).float()
        c_sh = (_real_norm_margin_from_rows(sp_eval["hc_t"] + out_sh, gamma_A_t, eps_A, sp_eval["e_a"], sp_eval["e_f"]) - sp_eval["ma"]).cpu().double().numpy()
        d_sh = (
            _real_norm_margin_from_rows(sp_eval["hc_t"][sp_eval["b_i"]] + out_sh[sp_eval["s_i"]], gamma_A_t, eps_A, sp_eval["e_as"], sp_eval["e_fb"])
            - _real_norm_margin_from_rows(sp_eval["hc_t"][sp_eval["b_i"]] + out_sh[sp_eval["b_i"]], gamma_A_t, eps_A, sp_eval["e_as"], sp_eval["e_fb"])
        ).cpu().double().numpy()
        sc_p = _compute_safe_q_and_r2(d_sh, sp_eval["d_ref"], c_sh, sp_eval["c_true"])
        if sc_p["Q_causal"] is not None:
            shuf_qc.append(sc_p["Q_causal"])
        if sc_p["R2_c"] is not None:
            shuf_r2.append(sc_p["R2_c"])

    # LIVE Behavioral Ablation & ACTUAL LIVE HEAD RESTORATION on held-out eval_items (100% same generate path!)
    prompts_eval = [it[prompt_key] for it in eval_items]
    gens_intact = _greedy_generate_clean(model_A, tok, device, prompts_eval, max_new_tokens=max_new_tokens, batch_size=batch_size, v_shared=v_shared)
    em_intact = float(np.mean([int(exact_match_fn(g, it)) for g, it in zip(gens_intact, eval_items)]))

    # Compute per-head discovery mean z_h
    z_disc_sums = {(li, hi): np.zeros(d_h, dtype=np.float64) for li, hi in selected_heads}
    old_p = tok.padding_side
    tok.padding_side = "left"
    handles = []
    try:
        for li in unique_layers:
            def _mk_zh(layer_i=li):
                def _h(_m, inp):
                    z = inp[0][:, -1, :].detach().float().cpu().double().numpy()
                    for hi in heads_by_layer[layer_i]:
                        z_disc_sums[(layer_i, hi)] += z[:, hi * d_h : (hi + 1) * d_h].sum(axis=0)
                return _h
            handles.append(layers[li].self_attn.o_proj.register_forward_pre_hook(_mk_zh()))
        p_disc = [it[prompt_key] for it in disc_items]
        for i0 in range(0, len(p_disc), batch_size):
            enc = tok(p_disc[i0 : i0 + batch_size], return_tensors="pt", padding=True, add_special_tokens=True)
            ids_b, msk_b = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            model_A(input_ids=ids_b, attention_mask=msk_b, position_ids=_left_pad_position_ids(msk_b), use_cache=False)
    finally:
        for h in handles:
            h.remove()
        tok.padding_side = old_p
    mean_zh = {k: torch.from_numpy((v / max(len(disc_items), 1)).astype(np.float32)).to(device=device, dtype=layers[k[0]].self_attn.o_proj.weight.dtype)
               for k, v in z_disc_sums.items()}

    # Build lookup from (li, hi) -> cached clamped z_best_eval[hw_i] for live head restoration at step 0!
    z_restore_map = {(hw["layer"], hw["head"]): z_best_eval[hw_i] for hw_i, hw in enumerate(head_weights)}
    lproc = [_MaskPaddedVocabLogitsProcessor(v_shared)]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def _live_head_gen(mode: str) -> float:
        old_p2 = tok.padding_side
        tok.padding_side = "left"
        h_list = []
        txts: List[str] = []
        batch_idx = [0, 0]
        try:
            for li in unique_layers:
                def _mk_mod(layer_i=li):
                    def _h(_m, inp):
                        x = inp[0].clone()
                        for hi in heads_by_layer[layer_i]:
                            sh = slice(hi * d_h, (hi + 1) * d_h)
                            if mode == "ablate":
                                x[:, -1, sh] = mean_zh[(layer_i, hi)]
                            elif mode == "restore_live":
                                # Stateless prefill check (`x.shape[1] > 1` on prefill; cached decode steps have `== 1`!)
                                if x.shape[1] > 1:
                                    x[:, -1, sh] = z_restore_map[(layer_i, hi)][batch_idx[0] : batch_idx[1]].to(dtype=x.dtype)
                                else:
                                    x[:, -1, sh] = mean_zh[(layer_i, hi)]
                        return (x,) + inp[1:]
                    return _h
                h_list.append(layers[li].self_attn.o_proj.register_forward_pre_hook(_mk_mod()))
            for i0 in range(0, len(prompts_eval), batch_size):
                i1 = min(i0 + batch_size, len(prompts_eval))
                batch_idx[0], batch_idx[1] = i0, i1
                enc = tok(prompts_eval[i0:i1], return_tensors="pt", padding=True, add_special_tokens=True)
                ids = enc["input_ids"].to(device)
                gen = model_A.generate(
                    input_ids=ids, attention_mask=enc["attention_mask"].to(device),
                    max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0, use_cache=True,
                    pad_token_id=pad_id, logits_processor=lproc,
                )
                plen = ids.shape[1]
                for bi in range(i1 - i0):
                    txts.append(tok.decode(gen[bi, plen:], skip_special_tokens=True))
        finally:
            for h in h_list:
                h.remove()
            tok.padding_side = old_p2
        assert_zero_runtime_hooks(model_A)
        return float(np.mean([int(exact_match_fn(g, it)) for g, it in zip(txts, eval_items)]))

    em_joint_ab_gen = _live_head_gen("ablate")
    em_live_rest_gen = _live_head_gen("restore_live")
    gap_gen = em_intact - em_joint_ab_gen
    rec_live_gen = float((em_live_rest_gen - em_joint_ab_gen) / gap_gen) if gap_gen > 1e-6 else 0.0

    # Offline direct-write diagnostic (consistent tok-0 argmax on intact vs ablated vs restored, native BF16!)
    def _tok0_acc_from_h(h_raw: torch.Tensor) -> float:
        rms = torch.sqrt(torch.mean(h_raw.float() ** 2, dim=-1, keepdim=True) + eps_A)
        hn = (h_raw.float() / rms) * gamma_A_t.unsqueeze(0)
        preds = _native_bf16_argmax_shared(hn, lm_w, chunk=16)
        return float(np.mean([int(p == int(it[ans_tok_key])) for p, it in zip(preds, eval_items)]))

    em_intact_tok = _tok0_acc_from_h(sp_eval["hf_t"])
    em_direct_ab_tok = _tok0_acc_from_h(sp_eval["hc_t"] + mean_out_disc_t.unsqueeze(0))
    em_direct_rest_tok = _tok0_acc_from_h(sp_eval["hc_t"] + out_best_eval)
    gap_direct = em_intact_tok - em_direct_ab_tok
    rec_direct_tok = float((em_direct_rest_tok - em_direct_ab_tok) / gap_direct) if gap_direct > 1e-6 else 0.0

    # STRICT JOINT UNROUNDED G1 PASS PREDICATES (requires actual live recovery too!)
    mean_margin_intact = float(sp_eval["mi"].mean().item())
    mean_c_true = float(np.mean(sp_eval["c_true"]))
    ld_share_eval = float(mean_c_true / (abs(mean_margin_intact) + 1e-12))
    win_eval = r_sweep_eval[f"r_{best_r}"]
    q_win, r2_win = win_eval["Q_causal"], win_eval["R2_c"]
    shuf_mean_r2 = float(np.mean(shuf_r2)) if shuf_r2 else 1.0

    g1_drop_pass = bool((100.0 * gap_gen >= 20.0) or (ld_share_eval >= 0.30))
    g1_fid_pass = bool((q_win is not None and q_win >= 0.80) and (r2_win is not None and r2_win >= 0.80) and (shuf_mean_r2 <= 0.10))
    g1_rec_pass = bool((rec_live_gen >= 0.80) and (gap_gen >= 0.05))
    g1_pass = bool(g1_drop_pass and g1_fid_pass and g1_rec_pass)

    hc_disc = head_out_disc.astype(np.float64) - head_out_disc.astype(np.float64).mean(axis=0, keepdims=True)
    _, _, Vt_h = sla.svd(hc_disc, full_matrices=False)
    V_best = Vt_h[: min(64, Vt_h.shape[0])].T

    # Save last-token eval x_A per layer for Astra zero-intercept head supply check in G2!
    x_last_eval_by_l = {li: np.stack([s[-1].float().numpy() for s in seqs_eval[li]], axis=0).astype(np.float32) for li in unique_layers}

    return {
        "report": {
            "component": f"LateHeads_{len(selected_heads)}h",
            "selected_heads": [[int(l), int(h)] for l, h in selected_heads],
            "unique_layers": unique_layers,
            "n_disc": len(disc_items), "n_eval": len(eval_items),
            "first_tok_id_identical_disc_eval": bool(len(set(ft_disc.tolist() + ft_eval.tolist())) == 1),
            "mean_pos0_sink_mass_eval": round(sink_mass_eval, 4),
            "fisher_top64_retained_frac": round(f_ret, 6),
            "mean_margin_intact": round(mean_margin_intact, 4),
            "mean_c_true": round(mean_c_true, 4),
            "ld_share_eval": round(ld_share_eval, 4),
            "em_intact_gen": round(em_intact, 4),
            "em_joint_ablated_gen": round(em_joint_ab_gen, 4),
            "em_live_restored_gen": round(em_live_rest_gen, 4),
            "joint_ablation_drop_pp": round(100.0 * gap_gen, 2),
            "rec_live_gen": round(rec_live_gen, 4),
            "offline_direct_tok0": {
                "em_intact_tok": round(em_intact_tok, 4),
                "em_ablated_tok": round(em_direct_ab_tok, 4),
                "em_restored_tok": round(em_direct_rest_tok, 4),
                "rec_direct_tok": round(rec_direct_tok, 4),
            },
            "frozen_best_r_from_disc": int(best_r),
            "r_sweep_disc": r_sweep_disc,
            "r_sweep_eval": r_sweep_eval,
            "shuf_pi_20draws": {
                "mean_Q_causal": round(float(np.mean(shuf_qc)), 4) if shuf_qc else None,
                "mean_R2_c": round(shuf_mean_r2, 4) if shuf_r2 else None,
                "max_R2_c": round(float(np.max(shuf_r2)), 4) if shuf_r2 else None,
            },
            "g1_checks": {
                "drop_pass": g1_drop_pass,
                "fidelity_pass": g1_fid_pass,
                "live_gen_recovery_pass": g1_rec_pass,
                "G1_PASS": g1_pass,
            },
        },
        "g2_payload": {
            "kind": "late_attn_heads",
            "unique_layers": unique_layers,
            "head_weights": head_weights,
            "G_by_layer": G_by_layer,
            "U_by_layer": U_by_layer,
            "mu_pos0_by_layer": mu_pos0_by_layer,
            "V_best": V_best.copy(),
            "x_last_eval_by_l": x_last_eval_by_l,
            "seqs_eval_A": seqs_eval,
            "head_out_eval_A_f32": head_out_eval.copy(),
            "mean_out_disc_f32": head_out_disc.mean(axis=0).astype(np.float32),
            "hc_eval_A_f32": sp_eval["hc_t"].cpu().numpy().astype(np.float32),
            "ma_eval_A_f32": sp_eval["ma"].cpu().numpy().astype(np.float32),
            "e_a_eval_A_f32": sp_eval["e_a"].cpu().numpy().astype(np.float32),
            "e_f_eval_A_f32": sp_eval["e_f"].cpu().numpy().astype(np.float32),
            "e_as_eval_A_f32": sp_eval["e_as"].cpu().numpy().astype(np.float32),
            "e_fb_eval_A_f32": sp_eval["e_fb"].cpu().numpy().astype(np.float32),
            "gamma_A_f32": gamma_A_t.cpu().numpy().astype(np.float32),
            "eps_A": float(eps_A),
            "c_true_eval": sp_eval["c_true"].copy(),
            "paired_mask_eval": sp_eval["paired_mask"].copy(),
            "paired_idx_eval": sp_eval["paired_idx"].copy(),
            "d_ref_swap": sp_eval["d_ref"].copy(),
            "eval_pairs": list(sp_eval["pairs"]),
            "frozen_r_disc": int(best_r),
            "rope_theta": float(rope_theta),
        },
    }


# ======================================================================================
# 5. Generic Calibration Streaming & G0 >=2,000-Token `M_U` / `T_raw` Verification
# ======================================================================================

@torch.no_grad()
def stream_donor_generic_calibration(
    model_A: nn.Module,
    tok: Any,
    device: torch.device,
    corpus: Dict[str, List[str]],
    E_B_shared_bf16: torch.Tensor,
    capture_mlp_layers: Sequence[int],
    capture_attn_layers: Sequence[int],
    v_shared: int,
    tmp_dir: Path,
    max_fit_tokens: int = 55000,
    batch_size: int = 16,
    max_length: int = 128,
) -> Dict[str, Any]:
    assert_zero_runtime_hooks(model_A)
    layers = get_decoder_layers(model_A)
    D_A = int(model_A.config.hidden_size)
    E_A_shared = model_A.lm_head.weight[:v_shared].detach()
    eps_A = float(getattr(model_A.model.norm, "variance_epsilon", getattr(model_A.model.norm, "eps", 1e-6)))

    fisher_acc = StreamingSharedVocabFisherAccumulator(E_A_shared, E_B_shared_bf16, vocab_chunk=16384)
    mlp_ls = sorted(set(int(l) for l in capture_mlp_layers))
    attn_ls = sorted(set(int(l) for l in capture_attn_layers))
    site_keys = [f"mlp_{l}" for l in mlp_ls] + [f"attn_{l}" for l in attn_ls]

    def _run_split(split_name: str, texts: Sequence[str], accumulate_fisher: bool, cap_tokens: Optional[int], save_report_logits: bool) -> Dict[str, Any]:
        bufs: Dict[str, List[np.ndarray]] = {k: [] for k in site_keys}
        rms_list: List[np.ndarray] = []
        tok_id_chunks: List[np.ndarray] = []
        h_fin_list: List[np.ndarray] = []
        top1_list: List[np.ndarray] = []
        n_tok_collected = 0
        old_p = tok.padding_side
        tok.padding_side = "right"
        handles = []
        cur_gt0: List[torch.Tensor] = []
        try:
            def _mk_out(k_name: str):
                def _h(_m, _i, out):
                    x = (out[0] if isinstance(out, tuple) else out).detach()
                    bufs[k_name].append(_tensor_to_bf16_u16(x[cur_gt0[0]]))
                return _h

            def _cap_fin(_m, inp):
                xf = inp[0].detach()[cur_gt0[0]]
                rms_list.append(torch.sqrt(torch.mean(xf.float() ** 2, dim=-1) + eps_A).cpu().numpy().astype(np.float32))
                if save_report_logits:
                    h_fin_list.append(_tensor_to_bf16_u16(xf))

            for l in mlp_ls:
                handles.append(layers[l].post_attention_layernorm.register_forward_hook(_mk_out(f"mlp_{l}")))
            for l in attn_ls:
                handles.append(layers[l].input_layernorm.register_forward_hook(_mk_out(f"attn_{l}")))
            handles.append(model_A.model.norm.register_forward_pre_hook(_cap_fin))

            for i0 in range(0, len(texts), batch_size):
                if cap_tokens is not None and n_tok_collected >= cap_tokens:
                    break
                enc = tok(list(texts[i0 : i0 + batch_size]), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                ids = enc["input_ids"].to(device)
                msk = enc["attention_mask"].to(device)
                _, pos_gt0 = get_pos0_and_gt0_masks(msk)
                cur_gt0.clear()
                cur_gt0.append(pos_gt0)
                tok_id_chunks.append(ids[pos_gt0].cpu().numpy().astype(np.int32))
                out = model_A(input_ids=ids, attention_mask=msk, use_cache=False)
                lg_gt0 = out.logits[pos_gt0][:, :v_shared]
                if accumulate_fisher:
                    fisher_acc.update(lg_gt0)
                if save_report_logits:
                    top1_list.append(torch.argmax(lg_gt0, dim=-1).cpu().numpy().astype(np.int32))
                n_tok_collected += int(pos_gt0.sum().item())
                del out, lg_gt0
        finally:
            for h in handles:
                h.remove()
            tok.padding_side = old_p
        assert_zero_runtime_hooks(model_A)

        saved_paths: Dict[str, str] = {}
        for k in site_keys:
            arr = np.concatenate(bufs[k], axis=0) if bufs[k] else np.empty((0, D_A), dtype=np.uint16)
            p = tmp_dir / f"donor_{split_name}_{k}.npy"
            np.save(p, arr)
            saved_paths[k] = str(p)
            bufs[k].clear()
        res: Dict[str, Any] = {
            "n_tokens": int(n_tok_collected),
            "paths": saved_paths,
            "rms_A": np.concatenate(rms_list, axis=0) if rms_list else np.empty((0,), dtype=np.float32),
            "tok_ids": np.concatenate(tok_id_chunks, axis=0) if tok_id_chunks else np.empty((0,), dtype=np.int32),
        }
        if save_report_logits:
            p_hf = tmp_dir / f"donor_{split_name}_h_final.npy"
            np.save(p_hf, np.concatenate(h_fin_list, axis=0))
            res["h_final_path"] = str(p_hf)
            res["top1_A"] = np.concatenate(top1_list, axis=0)
        return res

    fit_info = _run_split("fit", corpus["fit"], accumulate_fisher=True, cap_tokens=max_fit_tokens, save_report_logits=False)
    tune_info = _run_split("val_tune", corpus["val_tune"], accumulate_fisher=False, cap_tokens=None, save_report_logits=True)
    rep_info = _run_split("val_report", corpus["val_report"], accumulate_fisher=False, cap_tokens=None, save_report_logits=True)
    assert fit_info["n_tokens"] >= 50000, f"Need >=50,000 generic fit pos>0 tokens, got {fit_info['n_tokens']}"
    assert rep_info["n_tokens"] >= 2000, f"Need >=2,000 generic val_report pos>0 tokens, got {rep_info['n_tokens']}"

    return {"fisher": fisher_acc.finalize(), "fit": fit_info, "val_tune": tune_info, "val_report": rep_info, "site_keys": site_keys}


@torch.no_grad()
def verify_g0_bridges_on_2000_tokens(
    model_A: nn.Module,
    E_B_shared_bf16: torch.Tensor,
    gamma_B_f64: np.ndarray,
    eps_B: float,
    fisher_res: Dict[str, np.ndarray],
    h_final_tune_u16: np.ndarray,
    h_final_rep_u16: np.ndarray,
    v_shared: int,
    chunk_toks: int = 128,
) -> Dict[str, Any]:
    """Exact G0 verification of post-norm `M_U` and raw pre-norm `T_raw` on `>= 2,000` generic `pos > 0` tokens."""
    dev = next(model_A.parameters()).device
    E_A = model_A.lm_head.weight[:v_shared].detach()
    E_B = E_B_shared_bf16.to(device=dev)
    norm_A = model_A.model.norm
    gA = norm_A.weight.detach().cpu().double().numpy()
    gB_t = torch.from_numpy(gamma_B_f64.astype(np.float32)).to(device=dev)

    def _eval_map(h_u16: np.ndarray, M_mat: np.ndarray, is_raw: bool) -> Dict[str, float]:
        M_t = torch.from_numpy(M_mat.astype(np.float32)).to(device=dev)
        n = len(h_u16)
        agree = 0
        kls: List[float] = []
        for i0 in range(0, n, chunk_toks):
            hA = torch.from_numpy(bf16_to_f64_np(h_u16[i0 : i0 + chunk_toks]).astype(np.float32)).to(device=dev)
            hA_n = norm_A(hA.to(dtype=E_A.dtype)).float()
            lg_A = F.linear(hA_n.to(E_A.dtype), E_A).float()
            lp_A = F.log_softmax(lg_A, dim=-1)
            p_A = torch.exp(lp_A)
            if is_raw:
                hB_raw = hA @ M_t.T
                rms_b = torch.sqrt(torch.mean(hB_raw ** 2, dim=-1, keepdim=True) + eps_B)
                hB_n = (hB_raw / rms_b) * gB_t.unsqueeze(0)
            else:
                hB_n = hA_n @ M_t.T
            lg_B = F.linear(hB_n.to(E_B.dtype), E_B).float()
            lp_B = F.log_softmax(lg_B, dim=-1)
            kls.extend(torch.sum(p_A * (lp_A - lp_B), dim=-1).clamp(min=0.0).cpu().tolist())
            agree += int((torch.argmax(lg_A, dim=-1) == torch.argmax(lg_B, dim=-1)).sum().item())
            del hA, hA_n, lg_A, lp_A, p_A, hB_n, lg_B, lp_B
        return {
            "n_tokens": int(n),
            "top1_agreement": float(agree / max(n, 1)),
            "mean_kl": float(np.mean(kls)),
            "median_kl": float(np.median(kls)),
        }

    best_eps_mu, best_kl_mu, best_pl_mu = 1e-2, 1e9, None
    best_eps_tr, best_kl_tr, best_pl_tr = 1e-2, 1e9, None
    for eps_rel in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
        pl = compute_raw_and_postnorm_fisher_plugs(
            fisher_res["G_A"], fisher_res["G_B"], fisher_res["G_BA"], gA, gamma_B_f64, eps_rel_med=eps_rel
        )
        sm = _eval_map(h_final_tune_u16[:512], pl["M_U"], is_raw=False)
        st = _eval_map(h_final_tune_u16[:512], pl["T_raw"], is_raw=True)
        if sm["mean_kl"] < best_kl_mu:
            best_kl_mu, best_eps_mu, best_pl_mu = sm["mean_kl"], eps_rel, pl
        if st["mean_kl"] < best_kl_tr:
            best_kl_tr, best_eps_tr, best_pl_tr = st["mean_kl"], eps_rel, pl

    assert best_pl_mu is not None and best_pl_tr is not None
    rep_mu = _eval_map(h_final_rep_u16, best_pl_mu["M_U"], is_raw=False)
    rep_tr = _eval_map(h_final_rep_u16, best_pl_tr["T_raw"], is_raw=True)
    del E_B
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "best_eps_rel_M_U": best_eps_mu,
        "best_eps_rel_T_raw": best_eps_tr,
        "M_U_val_report": rep_mu,
        "T_raw_val_report": rep_tr,
        "M_U": best_pl_mu["M_U"],
        "T_raw": best_pl_tr["T_raw"],
        "G_A_raw": best_pl_tr["G_A_raw"],
        "G_B_raw": best_pl_tr["G_B_raw"],
    }


# ======================================================================================
# 6. Phase 2: Recipient 1.5B Generic Calibration + Gate G2 (Supply + Outlet Oracle!)
# ======================================================================================

@torch.no_grad()
def stream_recipient_moments_and_gamma(
    model_B: nn.Module,
    tok: Any,
    device: torch.device,
    corpus: Dict[str, List[str]],
    donor_cal_info: Dict[str, Any],
    v_shared: int,
    max_fit_tokens: int = 55000,
    batch_size: int = 16,
    max_length: int = 128,
) -> Dict[str, Any]:
    assert_zero_runtime_hooks(model_B)
    layers = get_decoder_layers(model_B)
    D_B = int(model_B.config.hidden_size)
    eps_B = float(getattr(model_B.model.norm, "variance_epsilon", getattr(model_B.model.norm, "eps", 1e-6)))
    site_keys = list(donor_cal_info["site_keys"])

    def _run_rec_split(split_name: str, texts: Sequence[str], cap_tokens: Optional[int], check_top1_A: Optional[np.ndarray]) -> Dict[str, Any]:
        bufs: Dict[str, List[np.ndarray]] = {k: [] for k in site_keys}
        rms_list: List[np.ndarray] = []
        tok_chunks: List[np.ndarray] = []
        top1_B_list: List[np.ndarray] = []
        n_tok = 0
        old_p = tok.padding_side
        tok.padding_side = "right"
        handles = []
        cur_gt0: List[torch.Tensor] = []
        try:
            def _mk_out(k_name: str):
                def _h(_m, _i, out):
                    bufs[k_name].append(_tensor_to_bf16_u16((out[0] if isinstance(out, tuple) else out).detach()[cur_gt0[0]]))
                return _h

            for k in site_keys:
                kind, l_str = k.split("_")
                l_idx = int(l_str)
                mod = layers[l_idx].post_attention_layernorm if kind == "mlp" else layers[l_idx].input_layernorm
                handles.append(mod.register_forward_hook(_mk_out(k)))
            handles.append(model_B.model.norm.register_forward_pre_hook(
                lambda _m, inp: rms_list.append(torch.sqrt(torch.mean(inp[0].detach()[cur_gt0[0]].float() ** 2, dim=-1) + eps_B).cpu().numpy().astype(np.float32))
            ))

            for i0 in range(0, len(texts), batch_size):
                if cap_tokens is not None and n_tok >= cap_tokens:
                    break
                enc = tok(list(texts[i0 : i0 + batch_size]), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                ids = enc["input_ids"].to(device)
                msk = enc["attention_mask"].to(device)
                _, pos_gt0 = get_pos0_and_gt0_masks(msk)
                cur_gt0.clear()
                cur_gt0.append(pos_gt0)
                tok_chunks.append(ids[pos_gt0].cpu().numpy().astype(np.int32))
                out = model_B(input_ids=ids, attention_mask=msk, use_cache=False)
                if check_top1_A is not None:
                    top1_B_list.append(torch.argmax(out.logits[pos_gt0][:, :v_shared], dim=-1).cpu().numpy().astype(np.int32))
                n_tok += int(pos_gt0.sum().item())
                del out
        finally:
            for h in handles:
                h.remove()
            tok.padding_side = old_p
        assert_zero_runtime_hooks(model_B)

        rec_toks = np.concatenate(tok_chunks, axis=0)
        don_toks = donor_cal_info[split_name]["tok_ids"]
        assert len(rec_toks) == len(don_toks) and np.array_equal(rec_toks, don_toks), f"Token-ID mismatch on {split_name}!"
        out_d: Dict[str, Any] = {
            "n_tokens": int(n_tok),
            "rms_B": np.concatenate(rms_list, axis=0),
            "Y_u16": {k: np.concatenate(bufs[k], axis=0) for k in site_keys},
        }
        if check_top1_A is not None:
            out_d["stock_1p5b_top1_agree_with_7b"] = float(np.mean(np.concatenate(top1_B_list, axis=0) == check_top1_A))
        return out_d

    fit_rec = _run_rec_split("fit", corpus["fit"], cap_tokens=max_fit_tokens, check_top1_A=None)
    tune_rec = _run_rec_split("val_tune", corpus["val_tune"], cap_tokens=None, check_top1_A=None)
    rep_rec = _run_rec_split("val_report", corpus["val_report"], cap_tokens=None, check_top1_A=donor_cal_info["val_report"]["top1_A"])

    gamma_scale = float(np.median(fit_rec["rms_B"].astype(np.float64) / (donor_cal_info["fit"]["rms_A"].astype(np.float64) + 1e-12)))
    gamma_scale_rep = float(np.median(rep_rec["rms_B"].astype(np.float64) / (donor_cal_info["val_report"]["rms_A"].astype(np.float64) + 1e-12)))

    moments_by_site: Dict[str, Dict[str, Any]] = {}
    for k in site_keys:
        X_u16 = np.load(donor_cal_info["fit"]["paths"][k], mmap_mode="r")
        Y_u16 = fit_rec["Y_u16"][k]
        n = min(len(X_u16), len(Y_u16))
        D_A = int(X_u16.shape[1])
        sum_x, sum_y = np.zeros(D_A, dtype=np.float64), np.zeros(D_B, dtype=np.float64)
        S_xx = np.zeros((D_A, D_A), dtype=np.float64)
        S_yy = np.zeros((D_B, D_B), dtype=np.float64)
        S_xy = np.zeros((D_A, D_B), dtype=np.float64)
        for i0 in range(0, n, 2048):
            i1 = min(i0 + 2048, n)
            xb = bf16_to_f64_np(np.asarray(X_u16[i0:i1]))
            yb = bf16_to_f64_np(Y_u16[i0:i1])
            sum_x += xb.sum(axis=0)
            sum_y += yb.sum(axis=0)
            S_xx += xb.T @ xb
            S_yy += yb.T @ yb
            S_xy += xb.T @ yb
        inv_n = 1.0 / max(float(n), 1.0)
        mu_x, mu_y = sum_x * inv_n, sum_y * inv_n
        S_xx, S_yy, S_xy = _sym(S_xx * inv_n), _sym(S_yy * inv_n), S_xy * inv_n
        moments_by_site[k] = {
            "n_fit": int(n),
            "mu_x": mu_x, "mu_y": mu_y,
            "S_xx": S_xx, "S_yy": S_yy, "S_xy": S_xy,
            "C_xx": _sym(S_xx - np.outer(mu_x, mu_x)),
            "C_yy": _sym(S_yy - np.outer(mu_y, mu_y)),
            "C_xy": S_xy - np.outer(mu_x, mu_y),
            "X_tune_u16": np.load(donor_cal_info["val_tune"]["paths"][k]),
            "Y_tune_u16": tune_rec["Y_u16"][k],
            "X_rep_u16": np.load(donor_cal_info["val_report"]["paths"][k]),
            "Y_rep_u16": rep_rec["Y_u16"][k],
        }
        del X_u16
        fit_rec["Y_u16"][k] = None

    return {
        "n_fit_tokens": int(fit_rec["n_tokens"]),
        "n_val_report_tokens": int(rep_rec["n_tokens"]),
        "gamma_scale_fit": gamma_scale,
        "gamma_scale_rep": gamma_scale_rep,
        "stock_1p5b_top1_agree_with_7b_val_report": rep_rec["stock_1p5b_top1_agree_with_7b"],
        "moments_by_site": moments_by_site,
    }


@torch.no_grad()
def run_gate_g2_whole_mlp(
    model_B: nn.Module,
    tok: Any,
    device: torch.device,
    payload: Dict[str, Any],
    site_moments: Dict[str, Any],
    T_raw: np.ndarray,
    gamma_scale: float,
    eval_items: Sequence[Dict[str, Any]],
    rec_base_ok: Sequence[int],
    cohort_masks: Dict[str, np.ndarray],
    prompt_key: str,
    ans_tok_key: str,
    foil_tok_key: str,
    exact_match_fn: Callable[[str, Dict[str, Any]], bool],
    v_shared: int,
    max_new_tokens: int = 5,
    ranks: Sequence[int] = (16, 32, 64, 128, 256),
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Complete Gate G2 on Recipient 1.5B (Donor 7B UNLOADED, 1.5B UNTOUCHED!) for Whole-MLP Organ:
      - `frozen_r` selected STRICTLY on generic `val_tune` (zero eval leakage!).
      - Replays Donor Self-Test on G2's exact recipient-suppliable `D_r` basis so inlet certification matches!
      - Scores Test 2A (True-Donor Outlet Oracle), Test 2B (Student-Supplied exact-mean, native `a*`, Astra homogeneous `P`),
        20-draw `SHUF-pi`, and live generation at `frozen_r` across ALL cohort masks (full, `p0`-blindspot, known retention)!
    """
    assert_zero_runtime_hooks(model_B)
    layer_idx = int(payload["layer"])
    layers_B = get_decoder_layers(model_B)
    gamma_B_t = model_B.model.norm.weight.detach().float()
    eps_B = float(getattr(model_B.model.norm, "variance_epsilon", getattr(model_B.model.norm, "eps", 1e-6)))
    lm_w_B = model_B.lm_head.weight[:v_shared].detach()
    prompts = [it[prompt_key] for it in eval_items]
    n_eval = len(eval_items)
    D_B = int(model_B.config.hidden_size)

    y_eval_B = np.zeros((n_eval, D_B), dtype=np.float32)
    h_fin_B = np.zeros((n_eval, D_B), dtype=np.float32)
    old_p = tok.padding_side
    tok.padding_side = "left"
    handles = []
    sl = [0, 0]
    try:
        handles.append(layers_B[layer_idx].post_attention_layernorm.register_forward_hook(
            lambda _m, _i, o: y_eval_B.__setitem__(slice(sl[0], sl[1]), (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach().float().cpu().numpy())
        ))
        handles.append(model_B.model.norm.register_forward_pre_hook(
            lambda _m, inp: h_fin_B.__setitem__(slice(sl[0], sl[1]), inp[0][:, -1, :].detach().float().cpu().numpy())
        ))
        for i0 in range(0, n_eval, batch_size):
            i1 = min(i0 + batch_size, n_eval)
            sl[0], sl[1] = i0, i1
            enc = tok(prompts[i0:i1], return_tensors="pt", padding=True, add_special_tokens=True)
            ids_b, msk_b = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            model_B(input_ids=ids_b, attention_mask=msk_b, position_ids=_left_pad_position_ids(msk_b), use_cache=False)
    finally:
        for h in handles:
            h.remove()
        tok.padding_side = old_p
    assert_zero_runtime_hooks(model_B)

    sup = compute_demand_weighted_supply_boundary(
        site_moments["C_xx"], site_moments["C_xy"], site_moments["C_yy"],
        site_moments["mu_x"], site_moments["mu_y"], payload["G_demand"],
        site_moments["X_tune_u16"], site_moments["Y_tune_u16"],
        site_moments["X_rep_u16"], site_moments["Y_rep_u16"],
        ranks=ranks,
    )
    frozen_r = int(sup["frozen_r_tune"])

    eval_pairs = payload["eval_pairs"]
    s_np = np.array([p[0] for p in eval_pairs], dtype=np.int64) if eval_pairs else np.empty(0, dtype=np.int64)
    b_np = np.array([p[1] for p in eval_pairs], dtype=np.int64) if eval_pairs else np.empty(0, dtype=np.int64)
    x_eval_A = payload["x_eval_A_f32"].astype(np.float64)
    x_diff = x_eval_A[s_np] - x_eval_A[b_np] if len(eval_pairs) > 0 else None
    y_diff = y_eval_B.astype(np.float64)[s_np] - y_eval_B.astype(np.float64)[b_np] if len(eval_pairs) > 0 else None
    astra_sup = compute_astra_zero_intercept_supply(
        site_moments["S_yy"], site_moments["S_xy"], payload["U_hom_best"],
        site_moments["X_tune_u16"], site_moments["Y_tune_u16"],
        x_eval_A, y_eval_B, X_diff_eval=x_diff, Y_diff_eval=y_diff,
    )

    Wg, Wu, Wd = payload["Wg_bf16"].to(device), payload["Wu_bf16"].to(device), payload["Wd_bf16"].to(device)
    T_raw_t = torch.from_numpy((float(gamma_scale) * T_raw).astype(np.float32)).to(device)
    V_best = payload["V_best"]
    T_V_t = torch.from_numpy((float(gamma_scale) * (T_raw @ V_best) @ V_best.T).astype(np.float32)).to(device)

    h_fin_B_t = torch.from_numpy(h_fin_B).to(device=device, dtype=torch.float32)
    paired_idx = np.asarray(payload["paired_idx_eval"], dtype=np.int64)
    paired_mask = np.asarray(payload["paired_mask_eval"], dtype=bool)
    p_t = torch.tensor(paired_idx, device=device, dtype=torch.long)
    ans_t = torch.tensor([int(it[ans_tok_key]) for it in eval_items], device=device, dtype=torch.long)
    foil_p_t = torch.tensor([int(eval_items[i][foil_tok_key]) for i in paired_idx], device=device, dtype=torch.long)
    e_a_B = lm_w_B[ans_t[p_t]].float() if len(paired_idx) > 0 else torch.empty((0, D_B), device=device)
    e_f_B = lm_w_B[foil_p_t].float() if len(paired_idx) > 0 else torch.empty((0, D_B), device=device)
    s_idx = torch.tensor(s_np, device=device, dtype=torch.long)
    b_idx = torch.tensor(b_np, device=device, dtype=torch.long)
    e_as_B = lm_w_B[ans_t[s_idx]].float() if len(eval_pairs) > 0 else torch.empty((0, D_B), device=device)
    e_fb_B = lm_w_B[ans_t[b_idx]].float() if len(eval_pairs) > 0 else torch.empty((0, D_B), device=device)

    margin_B_base = _real_norm_margin_from_rows(h_fin_B_t[p_t], gamma_B_t, eps_B, e_a_B, e_f_B) if len(paired_idx) > 0 else torch.empty(0, device=device)
    c_true = payload["c_true_eval"]
    d_ref = payload["d_ref_swap"]
    primary_wrong_mask = cohort_masks["primary_wrong"]
    primary_wrong_paired = primary_wrong_mask[paired_idx] if len(paired_idx) > 0 else np.empty(0, dtype=bool)
    m_mean_A_t = torch.from_numpy(payload["m_mean_disc_f32"]).to(device)

    # Replay Donor Self-Test on G2's EXACT recipient-suppliable `D_r` basis at `frozen_r`!
    r_eff_fr = min(frozen_r, sup["E_max"].shape[0])
    D_fr = torch.from_numpy(sup["D_max"][:, :r_eff_fr].astype(np.float32)).to(device)
    E_don_fr = torch.from_numpy((sup["Q_max"][:, :r_eff_fr].T @ sup["G_half"]).astype(np.float32)).to(device)
    mu_x_t = torch.from_numpy(site_moments["mu_x"].astype(np.float32)).to(device)
    mu_y_t = torch.from_numpy(site_moments["mu_y"].astype(np.float32)).to(device)
    x_eval_A_t = torch.from_numpy(payload["x_eval_A_f32"]).to(device=device, dtype=torch.float32)
    x_A_on_Dfr = mu_x_t.unsqueeze(0) + ((x_eval_A_t - mu_x_t.unsqueeze(0)) @ E_don_fr.T) @ D_fr.T
    m_A_on_Dfr = _native_bf16_mlp_forward(x_A_on_Dfr, Wg, Wu, Wd)
    hc_A_t = torch.from_numpy(payload["hc_eval_A_f32"]).to(device)
    ma_A_t = torch.from_numpy(payload["ma_eval_A_f32"]).to(device)
    gA_t = torch.from_numpy(payload["gamma_A_f32"]).to(device)
    ea_A_t = torch.from_numpy(payload["e_a_eval_A_f32"]).to(device)
    ef_A_t = torch.from_numpy(payload["e_f_eval_A_f32"]).to(device)
    eas_A_t = torch.from_numpy(payload["e_as_eval_A_f32"]).to(device)
    efb_A_t = torch.from_numpy(payload["e_fb_eval_A_f32"]).to(device)
    c_rep_A = (
        _real_norm_margin_from_rows(hc_A_t[p_t] + m_A_on_Dfr[p_t], gA_t, payload["eps_A"], ea_A_t, ef_A_t) - ma_A_t
    ).cpu().double().numpy() if len(paired_idx) > 0 else np.empty((0,), dtype=np.float64)
    d_rep_A = (
        _real_norm_margin_from_rows(hc_A_t[b_idx] + m_A_on_Dfr[s_idx], gA_t, payload["eps_A"], eas_A_t, efb_A_t)
        - _real_norm_margin_from_rows(hc_A_t[b_idx] + m_A_on_Dfr[b_idx], gA_t, payload["eps_A"], eas_A_t, efb_A_t)
    ).cpu().double().numpy() if len(eval_pairs) > 0 else np.empty((0,), dtype=np.float64)
    donor_replay_on_Dfr = _compute_safe_q_and_r2(d_rep_A, d_ref, c_rep_A, c_true)

    def _tok0_acc(h_raw: torch.Tensor) -> Tuple[float, List[int]]:
        rms = torch.sqrt(torch.mean(h_raw.float() ** 2, dim=-1, keepdim=True) + eps_B)
        hn = (h_raw.float() / rms) * gamma_B_t.unsqueeze(0)
        preds = _native_bf16_argmax_shared(hn, lm_w_B, chunk=16)
        ok = [int(p == int(it[ans_tok_key])) for p, it in zip(preds, eval_items)]
        return float(np.mean(ok)), ok

    base_tok0_acc, base_tok0_ok = _tok0_acc(h_fin_B_t)

    def _score_write(delta_A_raw: torch.Tensor, use_TV: bool = False) -> Dict[str, Any]:
        dw_B = (delta_A_raw - m_mean_A_t.unsqueeze(0)) @ (T_V_t if use_TV else T_raw_t).T
        h_new = h_fin_B_t + dw_B
        c_hat = (
            _real_norm_margin_from_rows(h_new[p_t], gamma_B_t, eps_B, e_a_B, e_f_B) - margin_B_base
        ).cpu().double().numpy() if len(paired_idx) > 0 else np.empty((0,), dtype=np.float64)
        d_hat = (
            _real_norm_margin_from_rows(h_fin_B_t[b_idx] + dw_B[s_idx], gamma_B_t, eps_B, e_as_B, e_fb_B)
            - _real_norm_margin_from_rows(h_fin_B_t[b_idx] + dw_B[b_idx], gamma_B_t, eps_B, e_as_B, e_fb_B)
        ).cpu().double().numpy() if len(eval_pairs) > 0 else np.empty((0,), dtype=np.float64)
        qr = _compute_safe_q_and_r2(d_hat, d_ref, c_hat, c_true)
        wir = _compute_safe_wrong_item_ratio(c_hat, c_true, primary_wrong_paired)
        t0_acc, t0_ok = _tok0_acc(h_new)
        t0_wrong = float(np.mean([t0_ok[i] for i in range(n_eval) if primary_wrong_mask[i]])) if np.any(primary_wrong_mask) else 0.0
        return {
            "R2_c": qr["R2_c"],
            "Q_outlet_swap": qr["Q_causal"],
            "mean_c_true_all": float(np.mean(c_true)),
            "mean_c_hat_all": float(np.mean(c_hat)),
            **wir,
            "live_tok0_acc_all": t0_acc,
            "live_tok0_lift_pp_all": 100.0 * (t0_acc - base_tok0_acc),
            "live_tok0_acc_on_primary_wrong": t0_wrong,
        }

    # Test 2A: True-Donor Outlet Oracle (raw & V-projected)
    m_eval_A_t = torch.from_numpy(payload["m_eval_A_f32"]).to(device=device, dtype=torch.float32)
    outlet_oracle_raw = _score_write(m_eval_A_t, use_TV=False)
    outlet_oracle_TV = _score_write(m_eval_A_t, use_TV=True)

    # Test 2B: Read-only rank sweep + frozen_r evaluation
    y_B_t = torch.from_numpy(y_eval_B).to(device=device, dtype=torch.float32)
    a_star_t = torch.from_numpy(sup["a_star"].astype(np.float32)).to(device)
    a_val = (y_B_t @ a_star_t).unsqueeze(1)

    student_by_rank: Dict[str, Any] = {}
    for r in ranks:
        r_eff = min(int(r), sup["E_max"].shape[0])
        E_r = torch.from_numpy(sup["E_max"][:r_eff].astype(np.float32)).to(device)
        D_r = torch.from_numpy(sup["D_max"][:, :r_eff].astype(np.float32)).to(device)
        x_ex = mu_x_t.unsqueeze(0) + ((y_B_t - mu_y_t.unsqueeze(0)) @ E_r.T) @ D_r.T
        mu_p = mu_x_t - (mu_y_t @ E_r.T) @ D_r.T
        x_as = a_val * mu_p.unsqueeze(0) + (y_B_t @ E_r.T) @ D_r.T
        student_by_rank[f"r_{r_eff}"] = {
            "supply_cal_report": sup["by_rank"][int(r)],
            "exact_mean_clamped": _score_write( _native_bf16_mlp_forward(x_ex, Wg, Wu, Wd), use_TV=False ),
            "native_a_star_bias": _score_write( _native_bf16_mlp_forward(x_as, Wg, Wu, Wd), use_TV=False ),
        }

    # Astra homogeneous P path
    P_t = torch.from_numpy(astra_sup["P"].astype(np.float32)).to(device)
    U_hom_t = torch.from_numpy(payload["U_hom_best"].astype(np.float32)).to(device)
    sc_hom = _score_write(_native_bf16_mlp_forward((y_B_t @ P_t.T) @ U_hom_t.T, Wg, Wu, Wd), use_TV=False)

    # 20-draw SHUF-pi at frozen_r (native a*-bias path, cached SwiGLU act ONCE!)
    E_fr = torch.from_numpy(sup["E_max"][:r_eff_fr].astype(np.float32)).to(device)
    mu_perp_fr = mu_x_t - (mu_y_t @ E_fr.T) @ D_fr.T
    x_fr_astar = a_val * mu_perp_fr.unsqueeze(0) + (y_B_t @ E_fr.T) @ D_fr.T
    act_fr = F.silu(F.linear(x_fr_astar.to(Wg.dtype), Wg)) * F.linear(x_fr_astar.to(Wu.dtype), Wu)
    shuf20_lifts: List[float] = []
    shuf20_wirs: List[float] = []
    for di in range(20):
        gp = torch.Generator(device="cpu").manual_seed(SEED + 5555 + di)
        m_sh = F.linear(act_fr[:, torch.randperm(Wg.shape[0], generator=gp).to(device)], Wd).float()
        sc_sh = _score_write(m_sh, use_TV=False)
        shuf20_lifts.append(sc_sh["live_tok0_lift_pp_all"])
        if sc_sh["wrong_item_ratio"] is not None:
            shuf20_wirs.append(sc_sh["wrong_item_ratio"])
    del act_fr

    # Live multi-token greedy generation at frozen_r (native a*-bias path + 1 SHUF-pi gen draw, shared-vocab masked!)
    lproc = [_MaskPaddedVocabLogitsProcessor(v_shared)]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def _live_gen(perm_d: Optional[torch.Tensor] = None) -> List[int]:
        old_p2 = tok.padding_side
        tok.padding_side = "left"
        h1, h2 = None, None
        saved_y: List[torch.Tensor] = []
        txts: List[str] = []
        try:
            def _cap(_m, _i, o):
                saved_y.clear()
                saved_y.append((o[0] if isinstance(o, tuple) else o).detach())

            def _inj(_m, inp):
                h_cur = inp[0].clone()
                yl = saved_y[0][:, -1, :].float()
                xt = (yl @ a_star_t).unsqueeze(1) * mu_perp_fr.unsqueeze(0) + (yl @ E_fr.T) @ D_fr.T
                mo = _native_bf16_mlp_forward(xt, Wg, Wu, Wd, perm_down=perm_d)
                h_cur[:, -1, :] = h_cur[:, -1, :] + ((mo - m_mean_A_t.unsqueeze(0)) @ T_raw_t.T).to(dtype=h_cur.dtype)
                return (h_cur,) + inp[1:]

            h1 = layers_B[layer_idx].post_attention_layernorm.register_forward_hook(_cap)
            h2 = model_B.model.norm.register_forward_pre_hook(_inj)
            for i0 in range(0, n_eval, batch_size):
                i1 = min(i0 + batch_size, n_eval)
                enc = tok(prompts[i0:i1], return_tensors="pt", padding=True, add_special_tokens=True)
                ids = enc["input_ids"].to(device)
                gen = model_B.generate(
                    input_ids=ids, attention_mask=enc["attention_mask"].to(device),
                    max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0, use_cache=True,
                    pad_token_id=pad_id, logits_processor=lproc,
                )
                plen = ids.shape[1]
                for bi in range(i1 - i0):
                    txts.append(tok.decode(gen[bi, plen:], skip_special_tokens=True))
        finally:
            if h1 is not None:
                h1.remove()
            if h2 is not None:
                h2.remove()
            tok.padding_side = old_p2
        assert_zero_runtime_hooks(model_B)
        return [int(exact_match_fn(g, it)) for g, it in zip(txts, eval_items)]

    ok_live = _live_gen(perm_d=None)
    gp0 = torch.Generator(device="cpu").manual_seed(SEED + 5555)
    ok_shuf = _live_gen(perm_d=torch.randperm(Wg.shape[0], generator=gp0).to(device))

    def _cohort_em(ok_vec: Sequence[int], mask: np.ndarray) -> Optional[float]:
        idx = np.where(mask)[0]
        return float(np.mean([ok_vec[i] for i in idx])) if len(idx) > 0 else None

    cohort_gen_report: Dict[str, Any] = {}
    for cname, cmask in cohort_masks.items():
        b_em = _cohort_em(rec_base_ok, cmask)
        l_em = _cohort_em(ok_live, cmask)
        s_em = _cohort_em(ok_shuf, cmask)
        cohort_gen_report[cname] = {
            "n": int(np.sum(cmask)),
            "em_base": round(b_em, 4) if b_em is not None else None,
            "em_organ": round(l_em, 4) if l_em is not None else None,
            "em_shuf_pi": round(s_em, 4) if s_em is not None else None,
            "em_lift_pp": round(100.0 * (l_em - b_em), 2) if (b_em is not None and l_em is not None) else None,
        }

    # STRICT SAME-PATH UNROUNDED G2 VERDICT (gating the realizable `native_a_star_bias` path at `frozen_r`!)
    fr_entry = student_by_rank[f"r_{r_eff_fr}"]
    as_sc = fr_entry["native_a_star_bias"]
    ex_sc = fr_entry["exact_mean_clamped"]
    sup_r = fr_entry["supply_cal_report"]
    all_base = float(np.mean(rec_base_ok))
    all_live = float(np.mean(ok_live))
    all_shuf = float(np.mean(ok_shuf))
    primary_lift = (
        (_cohort_em(ok_live, cohort_masks["primary_endpoint"]) or 0.0)
        - (_cohort_em(rec_base_ok, cohort_masks["primary_endpoint"]) or 0.0)
    )
    shuf_lift = (
        (_cohort_em(ok_shuf, cohort_masks["primary_endpoint"]) or 0.0)
        - (_cohort_em(rec_base_ok, cohort_masks["primary_endpoint"]) or 0.0)
    )

    # Exact integer-count primary/shuf lift checks (immune to binary float rounding at 10% / 5% boundaries!)
    prim_idx = np.where(cohort_masks["primary_endpoint"])[0]
    n_prim = len(prim_idx)
    base_hits_p = sum(int(rec_base_ok[i]) for i in prim_idx)
    live_hits_p = sum(int(ok_live[i]) for i in prim_idx)
    shuf_hits_p = sum(int(ok_shuf[i]) for i in prim_idx)
    primary_lift_ge_10 = bool(n_prim > 0 and (100 * (live_hits_p - base_hits_p) >= 10 * n_prim))
    primary_beats_shuf_5 = bool(n_prim > 0 and (100 * (live_hits_p - shuf_hits_p) >= 5 * n_prim))

    donor_replay_pass = bool(
        (donor_replay_on_Dfr["Q_causal"] is not None and donor_replay_on_Dfr["Q_causal"] >= 0.80)
        and (donor_replay_on_Dfr["R2_c"] is not None and donor_replay_on_Dfr["R2_c"] >= 0.80)
    )
    if not as_sc["ratio_supported"]:
        g2_verdict = "INCONCLUSIVE"
    elif (
        sup["tune_criteria_met"]
        and donor_replay_pass
        and sup_r["kappa_r_rep"] >= 0.85
        and sup_r["min_rho2_rep"] >= 0.80
        and sup["a_star_stats"]["passes_2pct"]
        and (as_sc["wrong_item_ratio"] is not None and as_sc["wrong_item_ratio"] >= 0.50)
        and (as_sc["R2_c"] is not None and as_sc["R2_c"] >= 0.65)
        and primary_lift_ge_10
        and primary_beats_shuf_5
        and (max(shuf20_lifts) <= 5.0 + 1e-9)
    ):
        g2_verdict = "PASS"
    elif (as_sc["wrong_item_ratio"] is not None and as_sc["wrong_item_ratio"] < 0.15) and (
        ex_sc["wrong_item_ratio"] is not None and ex_sc["wrong_item_ratio"] < 0.15
    ):
        g2_verdict = "KILL"
    else:
        g2_verdict = "INCONCLUSIVE"

    del Wg, Wu, Wd, T_raw_t, T_V_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "component": f"L{layer_idx}_MLP_whole",
        "frozen_r_from_val_tune": int(frozen_r),
        "n_eval": n_eval,
        "base_1p5b_em_all": round(all_base, 4),
        "base_1p5b_tok0_acc_all": round(base_tok0_acc, 4),
        "donor_self_test_replayed_on_G2_Dr": donor_replay_on_Dfr,
        "supply_overall": {
            "best_lam_rel": sup["best_lam_rel"],
            "kappa_fit_all": sup["kappa_fit_all"],
            "kappa_rep_all": sup["kappa_rep_all"],
            "rho2_rep_all_modes": sup["rho2_rep_all_modes"],
            "a_star_stats": sup["a_star_stats"],
            "astra_zero_intercept_supply": {
                "best_lam_rel": astra_sup["best_lam_rel"],
                "natural": astra_sup["natural"],
                "differences": astra_sup.get("differences"),
            },
        },
        "test2a_true_donor_outlet_oracle_raw": outlet_oracle_raw,
        "test2a_true_donor_outlet_oracle_TV": outlet_oracle_TV,
        "test2b_student_supplied_by_rank": student_by_rank,
        "test2b_astra_homogeneous_P": sc_hom,
        "test2b_shuf_pi_20draws_at_frozen_r": {
            "mean_tok0_lift_pp": round(float(np.mean(shuf20_lifts)), 2),
            "max_tok0_lift_pp": round(float(np.max(shuf20_lifts)), 2),
            "mean_wir": round(float(np.mean(shuf20_wirs)), 4) if shuf20_wirs else None,
        },
        "live_full_generation_at_frozen_r": {
            "frozen_r": int(frozen_r),
            "cohorts": cohort_gen_report,
            "em_all_base": round(all_base, 4),
            "em_all_organ": round(all_live, 4),
            "em_all_shuf_pi": round(all_shuf, 4),
        },
        "G2_VERDICT": g2_verdict,
    }


@torch.no_grad()
def run_gate_g2_late_attn_heads(
    model_B: nn.Module,
    tok: Any,
    device: torch.device,
    payload: Dict[str, Any],
    moments_by_site: Dict[str, Dict[str, Any]],
    T_raw: np.ndarray,
    gamma_scale: float,
    eval_items: Sequence[Dict[str, Any]],
    rec_base_ok: Sequence[int],
    prompt_key: str,
    ans_tok_key: str,
    foil_tok_key: str,
    v_shared: int,
    ranks: Sequence[int] = (16, 32, 64, 128, 256),
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Complete Gate G2 on Recipient 1.5B (Donor 7B UNLOADED, 1.5B UNTOUCHED!) for Late Attention Heads:
      - `frozen_r` selected STRICTLY on generic `val_tune` (zero eval leakage!).
      - Anchors `pos == 0` to Donor `mu_pos0` identically in real organ & 20-draw `SHUF-pi`, mapping `pos > 0` via `(E_r, D_r)`.
      - Computes Astra zero-intercept `P` & `R^2_worst` on natural states AND matched policy differences per layer!
      - Replays Donor Self-Test on G2's exact recipient-suppliable `D_r` basis!
    """
    assert_zero_runtime_hooks(model_B)
    layers_B = get_decoder_layers(model_B)
    gamma_B_t = model_B.model.norm.weight.detach().float()
    eps_B = float(getattr(model_B.model.norm, "variance_epsilon", getattr(model_B.model.norm, "eps", 1e-6)))
    lm_w_B = model_B.lm_head.weight[:v_shared].detach()
    unique_layers = list(payload["unique_layers"])
    prompts = [it[prompt_key] for it in eval_items]
    n_eval = len(eval_items)
    D_A = int(T_raw.shape[1])
    D_B = int(model_B.config.hidden_size)
    d_h = int(payload["head_weights"][0]["Wq"].shape[0])
    rope_theta = float(payload["rope_theta"])

    seqs_B: Dict[int, List[torch.Tensor]] = {li: [] for li in unique_layers}
    h_fin_B = np.zeros((n_eval, D_B), dtype=np.float32)
    old_p = tok.padding_side
    tok.padding_side = "left"
    handles = []
    cur_mask: List[torch.Tensor] = []
    sl = [0, 0]
    try:
        def _mk_cap(li: int):
            def _h(_m, _i, out):
                x = (out[0] if isinstance(out, tuple) else out).detach()
                msk = cur_mask[0]
                for bi in range(x.shape[0]):
                    nz = torch.nonzero(msk[bi].bool(), as_tuple=False).squeeze(-1)
                    seqs_B[li].append(x[bi, nz].cpu().to(dtype=torch.bfloat16))
            return _h

        for li in unique_layers:
            handles.append(layers_B[li].input_layernorm.register_forward_hook(_mk_cap(li)))
        handles.append(model_B.model.norm.register_forward_pre_hook(
            lambda _m, inp: h_fin_B.__setitem__(slice(sl[0], sl[1]), inp[0][:, -1, :].detach().float().cpu().numpy())
        ))
        for i0 in range(0, n_eval, batch_size):
            i1 = min(i0 + batch_size, n_eval)
            sl[0], sl[1] = i0, i1
            enc = tok(prompts[i0:i1], return_tensors="pt", padding=True, add_special_tokens=True)
            cur_mask.clear()
            cur_mask.append(enc["attention_mask"].to(device))
            model_B(input_ids=enc["input_ids"].to(device), attention_mask=cur_mask[0], position_ids=_left_pad_position_ids(cur_mask[0]), use_cache=False)
    finally:
        for h in handles:
            h.remove()
        tok.padding_side = old_p
    assert_zero_runtime_hooks(model_B)

    max_T = max(int(s.shape[0]) for s in seqs_B[unique_layers[0]])
    cos_all, sin_all = _make_rope_cos_sin(d_h, rope_theta, max_T + 8, device)

    sup_by_l: Dict[int, Dict[str, Any]] = {}
    astra_by_l: Dict[str, Any] = {}
    eval_pairs = payload["eval_pairs"]
    s_np = np.array([p[0] for p in eval_pairs], dtype=np.int64)
    b_np = np.array([p[1] for p in eval_pairs], dtype=np.int64)
    for li in unique_layers:
        sm = moments_by_site[f"attn_{li}"]
        sup_by_l[li] = compute_demand_weighted_supply_boundary(
            sm["C_xx"], sm["C_xy"], sm["C_yy"], sm["mu_x"], sm["mu_y"],
            payload["G_by_layer"][li], sm["X_tune_u16"], sm["Y_tune_u16"],
            sm["X_rep_u16"], sm["Y_rep_u16"], ranks=ranks,
        )
        # Astra zero-intercept supply check for head ingress layer (natural + matched L<->R policy differences!)
        y_last_B = np.stack([s[-1].float().numpy() for s in seqs_B[li]], axis=0).astype(np.float64)
        x_last_A = payload["x_last_eval_by_l"][li].astype(np.float64)
        U_l = payload["U_by_layer"][li][:, : min(payload["frozen_r_disc"], payload["U_by_layer"][li].shape[1])]
        asup_l = compute_astra_zero_intercept_supply(
            sm["S_yy"], sm["S_xy"], U_l, sm["X_tune_u16"], sm["Y_tune_u16"],
            x_last_A, y_last_B,
            X_diff_eval=(x_last_A[s_np] - x_last_A[b_np]),
            Y_diff_eval=(y_last_B[s_np] - y_last_B[b_np]),
        )
        astra_by_l[f"L{li}"] = {"natural": asup_l["natural"], "differences": asup_l.get("differences")}

    # Joint frozen rank strictly from generic val_tune across ALL unique ingress layers!
    sorted_ranks = sorted(int(rv) for rv in ranks)
    joint_frozen_r: Optional[int] = None
    best_joint_r_fb, best_min_kap_t = sorted_ranks[0], -1e9
    for r_cand in sorted_ranks:
        min_kt = min(sup_by_l[li]["by_rank"][r_cand]["kappa_r_tune"] for li in unique_layers)
        min_rt = min(sup_by_l[li]["by_rank"][r_cand]["min_rho2_tune"] for li in unique_layers)
        if joint_frozen_r is None and min_kt >= 0.85 and min_rt >= 0.80:
            joint_frozen_r = r_cand
        if min_kt > best_min_kap_t:
            best_min_kap_t, best_joint_r_fb = min_kt, r_cand
    joint_tune_criteria_met = joint_frozen_r is not None
    frozen_r = int(joint_frozen_r) if joint_frozen_r is not None else int(best_joint_r_fb)

    T_raw_t = torch.from_numpy((float(gamma_scale) * T_raw).astype(np.float32)).to(device)
    h_fin_B_t = torch.from_numpy(h_fin_B).to(device=device, dtype=torch.float32)
    ans_t = torch.tensor([int(it[ans_tok_key]) for it in eval_items], device=device, dtype=torch.long)
    foil_t = torch.tensor([int(it[foil_tok_key]) for it in eval_items], device=device, dtype=torch.long)
    e_a_B, e_f_B = lm_w_B[ans_t].float(), lm_w_B[foil_t].float()
    s_idx = torch.tensor(s_np, device=device, dtype=torch.long)
    b_idx = torch.tensor(b_np, device=device, dtype=torch.long)
    e_as_B, e_fb_B = lm_w_B[ans_t[s_idx]].float(), lm_w_B[ans_t[b_idx]].float()

    margin_B_base = _real_norm_margin_from_rows(h_fin_B_t, gamma_B_t, eps_B, e_a_B, e_f_B)
    c_true, d_ref = payload["c_true_eval"], payload["d_ref_swap"]
    wrong_mask = np.array([int(ok) == 0 for ok in rec_base_ok], dtype=bool)
    m_mean_A_t = torch.from_numpy(payload["mean_out_disc_f32"]).to(device)

    def _tok0_acc(h_raw: torch.Tensor) -> Tuple[float, List[int]]:
        rms = torch.sqrt(torch.mean(h_raw.float() ** 2, dim=-1, keepdim=True) + eps_B)
        hn = (h_raw.float() / rms) * gamma_B_t.unsqueeze(0)
        preds = _native_bf16_argmax_shared(hn, lm_w_B, chunk=16)
        ok = [int(p == int(it[ans_tok_key])) for p, it in zip(preds, eval_items)]
        return float(np.mean(ok)), ok

    base_tok0_acc, _ = _tok0_acc(h_fin_B_t)

    def _score_write(out_A_t: torch.Tensor) -> Dict[str, Any]:
        dw_B = (out_A_t - m_mean_A_t.unsqueeze(0)) @ T_raw_t.T
        h_new = h_fin_B_t + dw_B
        c_hat = (_real_norm_margin_from_rows(h_new, gamma_B_t, eps_B, e_a_B, e_f_B) - margin_B_base).cpu().double().numpy()
        d_hat = (
            _real_norm_margin_from_rows(h_fin_B_t[b_idx] + dw_B[s_idx], gamma_B_t, eps_B, e_as_B, e_fb_B)
            - _real_norm_margin_from_rows(h_fin_B_t[b_idx] + dw_B[b_idx], gamma_B_t, eps_B, e_as_B, e_fb_B)
        ).cpu().double().numpy()
        qr = _compute_safe_q_and_r2(d_hat, d_ref, c_hat, c_true)
        wir = _compute_safe_wrong_item_ratio(c_hat, c_true, wrong_mask)
        acc_all, ok_all = _tok0_acc(h_new)
        acc_w = float(np.mean([ok_all[i] for i in range(n_eval) if wrong_mask[i]])) if np.any(wrong_mask) else 0.0
        return {
            "R2_c": qr["R2_c"],
            "Q_outlet_swap": qr["Q_causal"],
            **wir,
            "live_tok0_acc_all": acc_all,
            "live_tok0_lift_pp": 100.0 * (acc_all - base_tok0_acc),
            "live_tok0_acc_on_1p5b_wrong": acc_w,
        }

    # Test 2A: True-Donor Outlet Oracle
    outlet_oracle = _score_write(torch.from_numpy(payload["head_out_eval_A_f32"]).to(device=device, dtype=torch.float32))

    # Replay Donor Self-Test on G2's exact `D_r` basis at `frozen_r`!
    out_don_Dfr = torch.zeros((n_eval, D_A), device=device, dtype=torch.float32)
    for hw in payload["head_weights"]:
        li = hw["layer"]
        sm, sp = moments_by_site[f"attn_{li}"], sup_by_l[li]
        re = min(frozen_r, sp["E_max"].shape[0])
        D_r = torch.from_numpy(sp["D_max"][:, :re].astype(np.float32)).to(device)
        E_don = torch.from_numpy((sp["Q_max"][:, :re].T @ sp["G_half"]).astype(np.float32)).to(device)
        mu_x_t = torch.from_numpy(sm["mu_x"].astype(np.float32)).to(device)
        Wq, Wk, Wv, Wo = hw["Wq"].to(device), hw["Wk"].to(device), hw["Wv"].to(device), hw["Wo"].to(device)
        bq = hw["bq"].to(device) if hw["bq"] is not None else None
        bk = hw["bk"].to(device) if hw["bk"] is not None else None
        bv = hw["bv"].to(device) if hw["bv"] is not None else None
        for i in range(n_eval):
            X_s = payload["seqs_eval_A"][li][i].to(device=device, dtype=torch.float32)
            X_c = X_s.clone()
            if X_s.shape[0] > 1:
                X_c[1:] = mu_x_t.unsqueeze(0) + ((X_s[1:] - mu_x_t.unsqueeze(0)) @ E_don.T) @ D_r.T
            _, oh, _ = _forward_single_attn_head_z_and_out(X_c, Wq, bq, Wk, bk, Wv, bv, Wo, cos_all, sin_all)
            out_don_Dfr[i] += oh
    hc_A_t = torch.from_numpy(payload["hc_eval_A_f32"]).to(device)
    ma_A_t = torch.from_numpy(payload["ma_eval_A_f32"]).to(device)
    gA_t = torch.from_numpy(payload["gamma_A_f32"]).to(device)
    c_don_Dr = (_real_norm_margin_from_rows(hc_A_t + out_don_Dfr, gA_t, payload["eps_A"],
                torch.from_numpy(payload["e_a_eval_A_f32"]).to(device), torch.from_numpy(payload["e_f_eval_A_f32"]).to(device)) - ma_A_t).cpu().double().numpy()
    d_don_Dr = (
        _real_norm_margin_from_rows(hc_A_t[b_idx] + out_don_Dfr[s_idx], gA_t, payload["eps_A"],
            torch.from_numpy(payload["e_as_eval_A_f32"]).to(device), torch.from_numpy(payload["e_fb_eval_A_f32"]).to(device))
        - _real_norm_margin_from_rows(hc_A_t[b_idx] + out_don_Dfr[b_idx], gA_t, payload["eps_A"],
            torch.from_numpy(payload["e_as_eval_A_f32"]).to(device), torch.from_numpy(payload["e_fb_eval_A_f32"]).to(device))
    ).cpu().double().numpy()
    donor_replay_on_Dfr = _compute_safe_q_and_r2(d_don_Dr, d_ref, c_don_Dr, c_true)

    # Test 2B: Student-Supplied Late Attention Heads (anchoring pos == 0 to donor mu_pos0!)
    student_by_rank: Dict[str, Any] = {}
    z_frozen: Optional[List[torch.Tensor]] = None
    for r in ranks:
        out_hat = torch.zeros((n_eval, D_A), device=device, dtype=torch.float32)
        z_cur: List[torch.Tensor] = []
        sink_ms: List[float] = []
        for hw in payload["head_weights"]:
            li = hw["layer"]
            sm, sp = moments_by_site[f"attn_{li}"], sup_by_l[li]
            re = min(int(r), sp["E_max"].shape[0])
            E_r = torch.from_numpy(sp["E_max"][:re].astype(np.float32)).to(device)
            D_r = torch.from_numpy(sp["D_max"][:, :re].astype(np.float32)).to(device)
            mu_x_t = torch.from_numpy(sm["mu_x"].astype(np.float32)).to(device)
            mu_y_t = torch.from_numpy(sm["mu_y"].astype(np.float32)).to(device)
            mu_p0_t = torch.from_numpy(payload["mu_pos0_by_layer"][li].astype(np.float32)).to(device)
            Wq, Wk, Wv, Wo = hw["Wq"].to(device), hw["Wk"].to(device), hw["Wv"].to(device), hw["Wo"].to(device)
            bq = hw["bq"].to(device) if hw["bq"] is not None else None
            bk = hw["bk"].to(device) if hw["bk"] is not None else None
            bv = hw["bv"].to(device) if hw["bv"] is not None else None
            zh_m = torch.zeros((n_eval, d_h), device=device, dtype=torch.float32)
            for i in range(n_eval):
                Y_s = seqs_B[li][i].to(device=device, dtype=torch.float32)
                X_in = torch.empty((Y_s.shape[0], D_A), device=device, dtype=torch.float32)
                X_in[0] = mu_p0_t
                if Y_s.shape[0] > 1:
                    X_in[1:] = mu_x_t.unsqueeze(0) + ((Y_s[1:] - mu_y_t.unsqueeze(0)) @ E_r.T) @ D_r.T
                zh, oh, a0 = _forward_single_attn_head_z_and_out(X_in, Wq, bq, Wk, bk, Wv, bv, Wo, cos_all, sin_all)
                zh_m[i] = zh
                out_hat[i] += oh
                sink_ms.append(a0)
            z_cur.append(zh_m)
        min_kap = min(sup_by_l[li]["by_rank"][int(r)]["kappa_r_rep"] for li in unique_layers)
        min_rho = min(sup_by_l[li]["by_rank"][int(r)]["min_rho2_rep"] for li in unique_layers)
        student_by_rank[f"r_{int(r)}"] = {
            "min_layer_kappa_r_rep": min_kap,
            "min_layer_rho2_rep": min_rho,
            "mean_pos0_sink_mass": round(float(np.mean(sink_ms)), 4),
            **_score_write(out_hat),
        }
        if int(r) == frozen_r:
            z_frozen = z_cur
            out_frozen = out_hat.clone()
    assert z_frozen is not None

    # 20-draw SHUF-pi at frozen_r using CACHED z_frozen in EXACT SAME BF16 W_O projection dtype as real head!
    Wo_devs = [hw["Wo"].to(device=device) for hw in payload["head_weights"]]
    shuf_lifts: List[float] = []
    shuf_wirs: List[float] = []
    out_shuf_first: Optional[torch.Tensor] = None
    for di in range(20):
        out_sh = torch.zeros((n_eval, D_A), device=device, dtype=torch.float32)
        for hw_i, (zh_m, Wo_bf) in enumerate(zip(z_frozen, Wo_devs)):
            gp = torch.Generator(device="cpu").manual_seed(SEED + 6666 + 131 * hw_i + di)
            out_sh += F.linear(zh_m.to(dtype=Wo_bf.dtype), Wo_bf[:, torch.randperm(d_h, generator=gp).to(device)]).float()
        if di == 0:
            out_shuf_first = out_sh.clone()
        sc_sh = _score_write(out_sh)
        shuf_lifts.append(sc_sh["live_tok0_lift_pp"])
        if sc_sh["wrong_item_ratio"] is not None:
            shuf_wirs.append(sc_sh["wrong_item_ratio"])

    # Live greedy generation arm on Recipient 1.5B (`repetition_penalty=1.0`, shared-vocab mask, identical to intact!)
    lproc = [_MaskPaddedVocabLogitsProcessor(v_shared)]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def _live_head_gen_B(out_A_pre: torch.Tensor) -> Tuple[int, float]:
        dw_all = (out_A_pre - m_mean_A_t.unsqueeze(0)) @ T_raw_t.T  # (n_eval, D_B)
        old_p2 = tok.padding_side
        tok.padding_side = "left"
        b_sl = [0, 0]
        h_inj = None
        txts: List[str] = []
        try:
            def _inj(_m, inp):
                h_cur = inp[0].clone()
                if h_cur.shape[1] > 1:
                    h_cur[:, -1, :] = h_cur[:, -1, :] + dw_all[b_sl[0] : b_sl[1]].to(dtype=h_cur.dtype)
                return (h_cur,) + inp[1:]
            h_inj = model_B.model.norm.register_forward_pre_hook(_inj)
            for i0 in range(0, n_eval, batch_size):
                i1 = min(i0 + batch_size, n_eval)
                b_sl[0], b_sl[1] = i0, i1
                enc = tok(prompts[i0:i1], return_tensors="pt", padding=True, add_special_tokens=True)
                ids = enc["input_ids"].to(device)
                gen = model_B.generate(
                    input_ids=ids, attention_mask=enc["attention_mask"].to(device),
                    max_new_tokens=4, do_sample=False, repetition_penalty=1.0, use_cache=True,
                    pad_token_id=pad_id, logits_processor=lproc,
                )
                plen = ids.shape[1]
                for bi in range(i1 - i0):
                    txts.append(tok.decode(gen[bi, plen:], skip_special_tokens=True))
        finally:
            if h_inj is not None:
                h_inj.remove()
            tok.padding_side = old_p2
        assert_zero_runtime_hooks(model_B)
        hits = sum(int(check_2hop_exact_word(g, it["answer"])) for g, it in zip(txts, eval_items))
        return int(hits), float(hits) / max(float(n_eval), 1.0)

    base_hits_gen = int(sum(int(x) for x in rec_base_ok))
    em_base_gen = float(base_hits_gen) / max(float(n_eval), 1.0)
    live_hits_gen, em_live_gen = _live_head_gen_B(out_frozen)
    shuf_hits_gen, em_shuf_gen = _live_head_gen_B(out_shuf_first) if out_shuf_first is not None else (base_hits_gen, em_base_gen)
    live_gen_lift_pp = 100.0 * (em_live_gen - em_base_gen)
    shuf_gen_lift_pp = 100.0 * (em_shuf_gen - em_base_gen)
    # Exact integer-count inclusive threshold comparisons (never fails on exact 10% / 5% boundaries!)
    live_lift_ge_10 = bool(100 * (live_hits_gen - base_hits_gen) >= 10 * n_eval)
    live_beats_shuf_5 = bool(100 * (live_hits_gen - shuf_hits_gen) >= 5 * n_eval)

    donor_replay_pass = bool(
        (donor_replay_on_Dfr["Q_causal"] is not None and donor_replay_on_Dfr["Q_causal"] >= 0.80)
        and (donor_replay_on_Dfr["R2_c"] is not None and donor_replay_on_Dfr["R2_c"] >= 0.80)
    )
    fr_entry = student_by_rank[f"r_{frozen_r}"]
    if not fr_entry["ratio_supported"]:
        g2_verdict = "INCONCLUSIVE"
    elif (
        joint_tune_criteria_met
        and donor_replay_pass
        and fr_entry["min_layer_kappa_r_rep"] >= 0.85
        and fr_entry["min_layer_rho2_rep"] >= 0.80
        and (fr_entry["wrong_item_ratio"] is not None and fr_entry["wrong_item_ratio"] >= 0.50)
        and (fr_entry["R2_c"] is not None and fr_entry["R2_c"] >= 0.65)
        and live_lift_ge_10
        and live_beats_shuf_5
        and (fr_entry["live_tok0_lift_pp"] - max(shuf_lifts)) >= 5.0 - 1e-9
    ):
        g2_verdict = "PASS"
    elif fr_entry["wrong_item_ratio"] is not None and fr_entry["wrong_item_ratio"] < 0.15:
        g2_verdict = "KILL"
    else:
        g2_verdict = "INCONCLUSIVE"

    return {
        "component": f"LateHeads_{len(payload['head_weights'])}h",
        "native_qkv_bias_exact": True,
        "frozen_r_from_val_tune": int(frozen_r),
        "joint_tune_criteria_met": bool(joint_tune_criteria_met),
        "n_eval": n_eval,
        "n_1p5b_wrong": int(np.sum(wrong_mask)),
        "base_1p5b_em_all": round(em_base_gen, 4),
        "base_1p5b_tok0_acc_all": round(base_tok0_acc, 4),
        "donor_self_test_replayed_on_G2_Dr": donor_replay_on_Dfr,
        "live_generation_at_frozen_r": {
            "em_base": round(em_base_gen, 4),
            "em_organ": round(em_live_gen, 4),
            "em_shuf_pi": round(em_shuf_gen, 4),
            "lift_pp": round(live_gen_lift_pp, 2),
            "shuf_lift_pp": round(shuf_gen_lift_pp, 2),
        },
        "per_layer_supply_summary": {
            f"L{li}": {
                "kappa_rep_all": sup_by_l[li]["kappa_rep_all"],
                "rho2_rep_all_modes": sup_by_l[li]["rho2_rep_all_modes"],
                "by_rank": sup_by_l[li]["by_rank"],
            }
            for li in unique_layers
        },
        "astra_zero_intercept_supply_by_layer": astra_by_l,
        "test2a_true_donor_outlet_oracle": outlet_oracle,
        "test2b_student_supplied_by_rank": student_by_rank,
        "test2b_shuf_pi_20draws_at_frozen_r": {
            "mean_tok0_lift_pp": round(float(np.mean(shuf_lifts)), 2),
            "max_tok0_lift_pp": round(float(np.max(shuf_lifts)), 2),
            "mean_wir": round(float(np.mean(shuf_wirs)), 4) if shuf_wirs else None,
        },
        "G2_VERDICT": g2_verdict,
    }


# ======================================================================================
# 7. End-to-End Runner (Phase 0 -> Phase 1 -> Phase 2, Strict Cleanup & Verdicts!)
# ======================================================================================

def _resolve_tokenizer_source(model_key: str, allow_download: bool) -> Tuple[str, bool]:
    local_dir = HF_CACHE_DIR / model_key
    if local_dir.is_dir():
        return str(local_dir), True
    if allow_download and model_key in LARGE_TO_SMALL_MODELS:
        return LARGE_TO_SMALL_MODELS[model_key], False
    raise FileNotFoundError(f"Tokenizer not found locally at {local_dir}; pass --allow-download.")


def _select_2hop_difficulty_from_pilot(don_4: float, rec_4: float, don_6: float, rec_6: float) -> int:
    """Preregistered pilot selector: filter eligible (`acc_7b >= 0.65`) FIRST, then maximize `acc_7b - acc_1p5b`!"""
    cands = [(4, don_4, don_4 - rec_4), (6, don_6, don_6 - rec_6)]
    elig = [c for c in cands if c[1] >= 0.65]
    pool = elig if elig else cands
    # Tie-break toward 6-row if gaps are equal
    return int(max(pool, key=lambda x: (x[2], x[0]))[0])


def run_all_gates_g0_g2(
    donor_key: str = DONOR_KEY_DEFAULT,
    recipient_key: str = RECIPIENT_KEY_DEFAULT,
    out_path: Path = OUT_PATH,
    allow_download: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    print(f"[v5.0] Starting Stage 1 Gates G0 -> G1 -> G2 (allow_download={allow_download}) ...", flush=True)

    from transformers import AutoTokenizer
    src_A, loc_A = _resolve_tokenizer_source(donor_key, allow_download=allow_download)
    src_B, loc_B = _resolve_tokenizer_source(recipient_key, allow_download=allow_download)
    tok_A = AutoTokenizer.from_pretrained(src_A, local_files_only=loc_A)
    tok_B = AutoTokenizer.from_pretrained(src_B, local_files_only=loc_B)
    v_shared = min(len(tok_A), len(tok_B))
    assert tok_A.get_vocab() == tok_B.get_vocab(), "Donor and recipient vocabularies must match identically!"
    print(f"[v5.0] Verified exact shared vocabulary: V_shared={v_shared}", flush=True)

    # n_fit_passages=760 guarantees >= 50,000 pos>0 tokens (~63k tokens!)
    corpus = build_60k_calibration_corpus(n_fit_passages=760, n_val_passages=80, seed=SEED)
    raw_airports, air_source = fetch_or_fallback_airports(max_airports=1000)
    # Single disjoint key partition shared by BOTH 4-row and 6-row 2-Hop!
    key_pools = partition_2hop_key_pools([tok_A, tok_B], max_rows=6, seed=SEED)
    ds_2hop_4 = build_in_prompt_2hop_dataset([tok_A, tok_B], key_pools, n_rows=4, n_pilot=64, n_disc=200, n_eval=400, seed=SEED)
    ds_2hop_6 = build_in_prompt_2hop_dataset([tok_A, tok_B], key_pools, n_rows=6, n_pilot=64, n_disc=200, n_eval=400, seed=SEED)
    print(
        f"[v5.0] Datasets built: OurAirports={len(raw_airports)} ({air_source}); "
        f"2-Hop shared key pools={ds_2hop_6['n_keys_by_split']} (eval=400 items = 200 L/R tables)",
        flush=True,
    )

    def _eval_2hop_acc(m: nn.Module, tk: Any, dev_m: torch.device, items_l: Sequence[Dict[str, Any]]) -> Tuple[float, List[int]]:
        gs = _greedy_generate_clean(m, tk, dev_m, [it["prompt"] for it in items_l], max_new_tokens=4, batch_size=32, v_shared=v_shared)
        ok = [int(check_2hop_exact_word(g, it["answer"])) for g, it in zip(gs, items_l)]
        return float(np.mean(ok)), ok

    # ------------------------------------------------------------------
    # PHASE 0: Recipient 1.5B Quick G0 Screen + Extract (E_B, gamma_B)
    # ------------------------------------------------------------------
    t_p0 = time.time()
    print("[Phase 0] Loading Recipient 1.5B alone for G0 headroom screen + (E_B, gamma_B) ...", flush=True)
    model_B = None
    try:
        model_B, tok_B, device, _ = load_model_bf16(recipient_key, allow_download=allow_download)
        assert_zero_runtime_hooks(model_B)
        E_B_shared_bf16 = model_B.lm_head.weight[:v_shared].detach().to(dtype=torch.bfloat16).cpu().clone()
        gamma_B_f64 = model_B.model.norm.weight.detach().cpu().double().numpy()
        eps_B = float(getattr(model_B.model.norm, "variance_epsilon", getattr(model_B.model.norm, "eps", 1e-6)))

        air_p0 = attach_airport_answer_tokens(raw_airports, tok_B, phrasing_idx=0)
        air_p1 = attach_airport_answer_tokens(raw_airports, tok_B, phrasing_idx=1)
        rec_air_p0_gens = _greedy_generate_clean(model_B, tok_B, device, [a["prompt"] for a in air_p0], max_new_tokens=5, batch_size=64, v_shared=v_shared)
        rec_air_p1_gens = _greedy_generate_clean(model_B, tok_B, device, [a["prompt"] for a in air_p1], max_new_tokens=5, batch_size=64, v_shared=v_shared)
        rec_air_p0_ok = [int(check_iata_strict_field(g, a["iata"])) for g, a in zip(rec_air_p0_gens, air_p0)]
        rec_air_p1_ok = [int(check_iata_strict_field(g, a["iata"])) for g, a in zip(rec_air_p1_gens, air_p1)]

        rec_2h4_pilot, _ = _eval_2hop_acc(model_B, tok_B, device, ds_2hop_4["pilot"])
        rec_2h6_pilot, _ = _eval_2hop_acc(model_B, tok_B, device, ds_2hop_6["pilot"])
        rec_2h4_eval, rec_2h4_eval_ok = _eval_2hop_acc(model_B, tok_B, device, ds_2hop_4["eval"])
        rec_2h6_eval, rec_2h6_eval_ok = _eval_2hop_acc(model_B, tok_B, device, ds_2hop_6["eval"])

        p0_peak_gib = round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 3) if device.type == "cuda" else 0.0
        p0_res_gib = round(torch.cuda.max_memory_reserved(device) / (1024 ** 3), 3) if device.type == "cuda" else 0.0
    finally:
        if model_B is not None:
            unload_model_vram_safe(model_B)
            del model_B
        gc.collect()
    post_p0_gib = round(torch.cuda.memory_allocated(device) / (1024 ** 3), 3) if device.type == "cuda" else 0.0
    assert post_p0_gib < 1.0, f"Phase 0 VRAM leak: {post_p0_gib} GiB"
    print(
        f"[Phase 0] Done in {time.time()-t_p0:.1f}s (peak={p0_peak_gib}/{p0_res_gib} GiB, post={post_p0_gib} GiB, rss={_host_rss_gib()} GiB) | "
        f"1.5B Air p0={sum(rec_air_p0_ok)}/{len(air_p0)} p1={sum(rec_air_p1_ok)}/{len(air_p1)} | "
        f"2-Hop 4r={rec_2h4_eval:.3f} 6r={rec_2h6_eval:.3f}",
        flush=True,
    )

    tmp_obj = tempfile.TemporaryDirectory(prefix="v50_cal_")
    tmp_dir = Path(tmp_obj.name)
    try:
        # ------------------------------------------------------------------
        # PHASE 1: Donor 7B ONLY (G0 + 28-Layer Discovery + Gate G1!)
        # ------------------------------------------------------------------
        t_p1 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        print("[Phase 1] Loading Donor 7B alone (100% blind to 1.5B activations!) ...", flush=True)
        model_A = None
        try:
            model_A, tok_A, device, _ = load_model_bf16(donor_key, allow_download=allow_download)
            assert_zero_runtime_hooks(model_A)

            don_air_p0_gens = _greedy_generate_clean(model_A, tok_A, device, [a["prompt"] for a in air_p0], max_new_tokens=5, batch_size=64, v_shared=v_shared)
            don_air_p1_gens = _greedy_generate_clean(model_A, tok_A, device, [a["prompt"] for a in air_p1], max_new_tokens=5, batch_size=64, v_shared=v_shared)
            don_air_p0_ok = [int(check_iata_strict_field(g, a["iata"])) for g, a in zip(don_air_p0_gens, air_p0)]
            don_air_p1_ok = [int(check_iata_strict_field(g, a["iata"])) for g, a in zip(don_air_p1_gens, air_p1)]

            # Split Donor-known airports (`7B_p0 == 1`, strictly donor-only!) into disjoint disc vs eval!
            donor_known_idx = [i for i, ok in enumerate(don_air_p0_ok) if ok == 1]
            rng_ap = np.random.default_rng(SEED + 77)
            rng_ap.shuffle(donor_known_idx)
            n_air_disc = min(128, len(donor_known_idx) // 3)
            air_disc_idx = donor_known_idx[:n_air_disc]
            air_eval_idx = donor_known_idx[n_air_disc:]
            air_disc_set = set(air_disc_idx)
            air_non_disc_idx = [i for i in range(len(air_p0)) if i not in air_disc_set]

            # Build paired disc list (for DLA + disc swap selection) & UNFILTERED full eval list!
            raw_air_disc = [dict(air_p0[i]) for i in air_disc_idx]
            disc_pairs_raw, disc_foil_map = build_matched_airport_swap_pairs(raw_air_disc)
            keep_d = sorted(disc_foil_map.keys())
            d_remap = {old_i: new_i for new_i, old_i in enumerate(keep_d)}
            air_disc_items = [{**raw_air_disc[i], "foil_tok_id": disc_foil_map[i]} for i in keep_d]
            air_disc_pairs = [(d_remap[s], d_remap[b]) for s, b in disc_pairs_raw]

            # Keep 100% of UNFILTERED held-out `air_eval_idx` for behavioral generation & retention;
            # paired swap margins use ONLY items with genuine distinct-token `foil_tok_id` (None if unpaired)!
            raw_air_eval = [dict(air_p1[i]) for i in air_eval_idx]
            air_eval_pairs, eval_foil_map = build_matched_airport_swap_pairs(raw_air_eval)
            air_eval_items = [{**raw_air_eval[i], "foil_tok_id": eval_foil_map.get(i)} for i in range(len(raw_air_eval))]
            air_eval_orig_idx = list(air_eval_idx)

            air_eval_rec_ok_p1 = [rec_air_p1_ok[i] for i in air_eval_orig_idx]
            air_eval_rec_ok_p0 = [rec_air_p0_ok[i] for i in air_eval_orig_idx]
            air_eval_don_ok_p1 = [don_air_p1_ok[i] for i in air_eval_orig_idx]
            # Fixed pre-intervention cohort masks on `air_eval_items`!
            air_mask_p0_blind = np.array([r0 == 0 for r0 in air_eval_rec_ok_p0], dtype=bool)
            air_mask_p0_known = ~air_mask_p0_blind
            air_mask_p1_blind_cond = np.array([(d1 == 1 and r1 == 0) for d1, r1 in zip(air_eval_don_ok_p1, air_eval_rec_ok_p1)], dtype=bool)
            n_air_blind_p0 = int(np.sum(air_mask_p0_blind))

            # 2-Hop G0 headroom & preregistered eligible pilot selector
            don_2h4_pilot, _ = _eval_2hop_acc(model_A, tok_A, device, ds_2hop_4["pilot"])
            don_2h6_pilot, _ = _eval_2hop_acc(model_A, tok_A, device, ds_2hop_6["pilot"])
            don_2h4_eval, don_2h4_eval_ok = _eval_2hop_acc(model_A, tok_A, device, ds_2hop_4["eval"])
            don_2h6_eval, don_2h6_eval_ok = _eval_2hop_acc(model_A, tok_A, device, ds_2hop_6["eval"])

            chosen_2hop_rows = _select_2hop_difficulty_from_pilot(don_2h4_pilot, rec_2h4_pilot, don_2h6_pilot, rec_2h6_pilot)
            ds_2hop = ds_2hop_6 if chosen_2hop_rows == 6 else ds_2hop_4
            rec_2hop_eval_ok = rec_2h6_eval_ok if chosen_2hop_rows == 6 else rec_2h4_eval_ok
            don_2hop_eval_ok = don_2h6_eval_ok if chosen_2hop_rows == 6 else don_2h4_eval_ok
            acc_7b_2h_eval = don_2h6_eval if chosen_2hop_rows == 6 else don_2h4_eval
            acc_1p5b_2h_eval = rec_2h6_eval if chosen_2hop_rows == 6 else rec_2h4_eval

            # Held-out non-discovery airport G0 headroom!
            acc_7b_air_held = float(np.mean([don_air_p1_ok[i] for i in air_non_disc_idx]))
            acc_1p5b_air_held = float(np.mean([rec_air_p1_ok[i] for i in air_non_disc_idx]))
            gap_air_held_pp = 100.0 * (acc_7b_air_held - acc_1p5b_air_held)
            gap_2h_eval_pp = 100.0 * (acc_7b_2h_eval - acc_1p5b_2h_eval)

            # Exact integer-count inclusive G0 headroom predicates (never fails on exact 15% / 65% boundaries!)
            n_air_nd = len(air_non_disc_idx)
            air_nd_7b_hits = sum(int(don_air_p1_ok[i]) for i in air_non_disc_idx)
            air_nd_1p5b_hits = sum(int(rec_air_p1_ok[i]) for i in air_non_disc_idx)
            n_2h_ev = len(ds_2hop["eval"])
            twoh_7b_hits = sum(int(x) for x in don_2hop_eval_ok)
            twoh_1p5b_hits = sum(int(x) for x in rec_2hop_eval_ok)

            g0_headroom = {
                "airports": {
                    "source": air_source,
                    "n_screened": len(air_p0),
                    "n_non_disc_heldout": n_air_nd,
                    "acc_7b_non_disc_p1": round(acc_7b_air_held, 4),
                    "acc_1p5b_non_disc_p1": round(acc_1p5b_air_held, 4),
                    "gap_non_disc_p1_pp": round(gap_air_held_pp, 2),
                    "n_donor_known_p0": len(donor_known_idx),
                    "n_disc_paired": len(air_disc_items),
                    "n_eval_unfiltered_donor_known": len(air_eval_items),
                    "n_eval_paired_donor_known": len(eval_foil_map),
                    "eval_paired_coverage": round(len(eval_foil_map) / max(len(air_eval_items), 1), 4),
                    "n_eval_p0_blindspot": n_air_blind_p0,
                    "meets_96_blindspot_floor": bool(n_air_blind_p0 >= 96),
                    "n_eval_p0_known": int(np.sum(air_mask_p0_known)),
                    "n_eval_p1_cond_blindspot": int(np.sum(air_mask_p1_blind_cond)),
                    "acc_7b_eval_p1": round(float(np.mean(air_eval_don_ok_p1)), 4),
                    "acc_1p5b_eval_p1": round(float(np.mean(air_eval_rec_ok_p1)), 4),
                    "acc_1p5b_p0_blindspot_on_p1": round(float(np.mean([air_eval_rec_ok_p1[i] for i in np.where(air_mask_p0_blind)[0]])), 4) if n_air_blind_p0 > 0 else None,
                    "G0_HEADROOM_PASS": bool(n_air_nd > 0 and (100 * (air_nd_7b_hits - air_nd_1p5b_hits) >= 15 * n_air_nd) and n_air_blind_p0 >= 96),
                },
                "in_prompt_2hop": {
                    "chosen_n_rows_from_pilot": chosen_2hop_rows,
                    "4_row_pilot": {"acc_7b": round(don_2h4_pilot, 4), "acc_1p5b": round(rec_2h4_pilot, 4), "gap_pp": round(100.0 * (don_2h4_pilot - rec_2h4_pilot), 2)},
                    "6_row_pilot": {"acc_7b": round(don_2h6_pilot, 4), "acc_1p5b": round(rec_2h6_pilot, 4), "gap_pp": round(100.0 * (don_2h6_pilot - rec_2h6_pilot), 2)},
                    "4_row_eval": {"acc_7b": round(don_2h4_eval, 4), "acc_1p5b": round(rec_2h4_eval, 4), "gap_pp": round(100.0 * (don_2h4_eval - rec_2h4_eval), 2)},
                    "6_row_eval": {"acc_7b": round(don_2h6_eval, 4), "acc_1p5b": round(rec_2h6_eval, 4), "gap_pp": round(100.0 * (don_2h6_eval - rec_2h6_eval), 2)},
                    "chosen_eval": {
                        "n_rows": chosen_2hop_rows, "n_eval_items": n_2h_ev,
                        "n_independent_tables": ds_2hop["n_independent_tables"]["eval"],
                        "acc_7b": round(acc_7b_2h_eval, 4), "acc_1p5b": round(acc_1p5b_2h_eval, 4),
                        "gap_pp": round(gap_2h_eval_pp, 2),
                    },
                    "G0_HEADROOM_PASS": bool(n_2h_ev > 0 and (100 * (twoh_7b_hits - twoh_1p5b_hits) >= 15 * n_2h_ev) and (100 * twoh_7b_hits >= 65 * n_2h_ev)),
                },
            }
            print(
                f"[G0 Headroom] Air held-out p1 gap=+{gap_air_held_pp:.2f}pp (p0_blind={n_air_blind_p0}) | "
                f"2-Hop ({chosen_2hop_rows}r) gap=+{gap_2h_eval_pp:.2f}pp (7B={acc_7b_2h_eval:.3f} vs 1.5B={acc_1p5b_2h_eval:.3f})",
                flush=True,
            )

            # 1B) All-28-Layer Discovery on BOTH tasks (`disc` splits ONLY!)
            print("[Phase 1] Running single-pass all-28-layer DLA + solo mean-ablation on discovery splits ...", flush=True)
            disc_air = run_all_layer_dla_and_solo_ablation(
                model_A, tok_A, device, air_disc_items, "prompt", "first_ans_tok_id", "foil_tok_id",
                lambda g, it: check_iata_strict_field(g, it["iata"]),
                v_shared=v_shared, max_new_tokens=5, single_token_task=False, batch_size=32,
            )
            disc_2hop = run_all_layer_dla_and_solo_ablation(
                model_A, tok_A, device, ds_2hop["disc"], "prompt", "answer_tok_id", "foil_tok_id",
                lambda g, it: check_2hop_exact_word(g, it["answer"]),
                v_shared=v_shared, max_new_tokens=4, single_token_task=True, batch_size=32,
            )

            # Lock late MLP candidates strictly by DIRECT DLA (`L >= 14`) so Depth-1 L27 plugs match!
            best_air_dla_mlp = max([m for m in disc_air["mlp_dla_all28"] if m["layer"] >= 14], key=lambda x: x["mean_dla"])["layer"]
            air_mlp_candidates = [27] + ([int(best_air_dla_mlp)] if int(best_air_dla_mlp) != 27 else [])

            late_heads_2h = disc_2hop["top16_late_heads_dla_and_solo"]
            sel_2hop_heads: List[Tuple[int, int]] = []
            cum_dla = 0.0
            for hd in late_heads_2h:
                if len(sel_2hop_heads) >= 8:
                    break
                if hd["mean_dla"] <= 0.0 and len(sel_2hop_heads) >= 2:
                    break
                sel_2hop_heads.append((int(hd["layer"]), int(hd["head"])))
                cum_dla += float(hd["frac_of_donor_ld"])
                if cum_dla >= 0.50 and len(sel_2hop_heads) >= 2:
                    break
            if not sel_2hop_heads:
                sel_2hop_heads = [(int(hd["layer"]), int(hd["head"])) for hd in late_heads_2h[:4]]
            best_2hop_dla_mlp = max([m for m in disc_2hop["mlp_dla_all28"] if m["layer"] >= 14], key=lambda x: x["mean_dla"])["layer"]

            capture_mlp_layers = sorted(set(air_mlp_candidates + [int(best_2hop_dla_mlp)]))
            capture_attn_layers = sorted(set(l for l, _ in sel_2hop_heads))
            print(
                f"[Phase 1] Locked Depth-1 candidates: Air MLPs={air_mlp_candidates} | "
                f"2-Hop Heads={sel_2hop_heads} (cum DLA={cum_dla:.2f}) + MLP=L{best_2hop_dla_mlp}",
                flush=True,
            )

            # 1C) Stream >=50k Generic pos>0 Tokens + Verify G0 M_U & T_raw on >=2,000 Tokens
            don_cal = stream_donor_generic_calibration(
                model_A, tok_A, device, corpus, E_B_shared_bf16,
                capture_mlp_layers=capture_mlp_layers, capture_attn_layers=capture_attn_layers,
                v_shared=v_shared, tmp_dir=tmp_dir, max_fit_tokens=55000, batch_size=16, max_length=128,
            )
            g0_bridges = verify_g0_bridges_on_2000_tokens(
                model_A, E_B_shared_bf16, gamma_B_f64, eps_B, don_cal["fisher"],
                np.load(don_cal["val_tune"]["h_final_path"]),
                np.load(don_cal["val_report"]["h_final_path"]),
                v_shared=v_shared,
            )
            g0_bridge_pass = bool(
                don_cal["fit"]["n_tokens"] >= 50000
                and don_cal["val_report"]["n_tokens"] >= 2000
                and g0_bridges["T_raw_val_report"]["top1_agreement"] >= 0.80
            )
            print(
                f"[G0 Bridges ({don_cal['fit']['n_tokens']} fit / {don_cal['val_report']['n_tokens']} rep tok)] "
                f"M_U top1={g0_bridges['M_U_val_report']['top1_agreement']:.4f} KL={g0_bridges['M_U_val_report']['mean_kl']:.4f} | "
                f"T_raw top1={g0_bridges['T_raw_val_report']['top1_agreement']:.4f} KL={g0_bridges['T_raw_val_report']['mean_kl']:.4f} | "
                f"PASS={g0_bridge_pass}",
                flush=True,
            )

            # 1D) Gate G1 Donor Self-Tests (7B ONLY, selection strictly on `disc`!)
            print("[Phase 1] Running Gate G1 Donor Self-Tests (7B ONLY) ...", flush=True)
            g1_airports: Dict[str, Any] = {}
            g2_air_payloads: Dict[str, Any] = {}
            for l_m in air_mlp_candidates:
                X_cal_l = np.load(don_cal["fit"]["paths"][f"mlp_{l_m}"])
                res_g1 = run_gate_g1_whole_mlp(
                    model_A, tok_A, device, l_m, air_disc_items, air_disc_pairs, air_eval_items, air_eval_pairs,
                    "prompt", "first_ans_tok_id", "foil_tok_id",
                    lambda g, it: check_iata_strict_field(g, it["iata"]),
                    g0_bridges["G_A_raw"], X_cal_l, v_shared=v_shared, max_new_tokens=5, batch_size=32,
                )
                g1_airports[f"L{l_m}_MLP"] = res_g1["report"]
                g2_air_payloads[f"L{l_m}_MLP"] = res_g1["g2_payload"]
                del X_cal_l
                br = res_g1["report"]["frozen_from_disc"]["best_r_clamped"]
                qc = res_g1["report"]["r_sweep_eval"][f"r_{br}"]["mean_clamped_Q_causal"]
                print(
                    f"  [G1 Air L{l_m} MLP] drop={res_g1['report']['ablation_drop_pp']}pp Q={qc} "
                    f"rec={res_g1['report']['rec_gen_clamped']} PASS={res_g1['report']['g1_checks']['G1_PASS']}",
                    flush=True,
                )

            res_g1_2h_heads = run_gate_g1_late_attn_heads(
                model_A, tok_A, device, sel_2hop_heads, ds_2hop["disc"], ds_2hop["eval"],
                "prompt", "answer_tok_id", "foil_tok_id",
                lambda g, it: check_2hop_exact_word(g, it["answer"]),
                g0_bridges["G_A_raw"], v_shared=v_shared, max_new_tokens=4, batch_size=32,
            )
            br_h = res_g1_2h_heads["report"]["frozen_best_r_from_disc"]
            print(
                f"  [G1 2-Hop Heads ({len(sel_2hop_heads)}h)] ld_share={res_g1_2h_heads['report']['ld_share_eval']} "
                f"drop={res_g1_2h_heads['report']['joint_ablation_drop_pp']}pp rec={res_g1_2h_heads['report']['rec_live_gen']} "
                f"R2={res_g1_2h_heads['report']['r_sweep_eval'][f'r_{br_h}']['R2_c']} "
                f"PASS={res_g1_2h_heads['report']['g1_checks']['G1_PASS']}",
                flush=True,
            )

            disc_2h_pairs = build_2hop_table_swap_pairs(ds_2hop["disc"])
            eval_2h_pairs = build_2hop_table_swap_pairs(ds_2hop["eval"])
            X_cal_2h_mlp = np.load(don_cal["fit"]["paths"][f"mlp_{best_2hop_dla_mlp}"])
            res_g1_2h_mlp = run_gate_g1_whole_mlp(
                model_A, tok_A, device, int(best_2hop_dla_mlp), ds_2hop["disc"], disc_2h_pairs, ds_2hop["eval"], eval_2h_pairs,
                "prompt", "answer_tok_id", "foil_tok_id",
                lambda g, it: check_2hop_exact_word(g, it["answer"]),
                g0_bridges["G_A_raw"], X_cal_2h_mlp, v_shared=v_shared, max_new_tokens=4, batch_size=32,
            )
            del X_cal_2h_mlp

            p1_peak_gib = round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 3) if device.type == "cuda" else 0.0
            p1_res_gib = round(torch.cuda.max_memory_reserved(device) / (1024 ** 3), 3) if device.type == "cuda" else 0.0
        finally:
            if model_A is not None:
                unload_model_vram_safe(model_A)
                del model_A
            gc.collect()
        post_p1_gib = round(torch.cuda.memory_allocated(device) / (1024 ** 3), 3) if device.type == "cuda" else 0.0
        assert post_p1_gib < 1.0, f"Phase 1 VRAM leak: {post_p1_gib} GiB"
        print(f"[Phase 1] Done in {time.time()-t_p1:.1f}s (peak={p1_peak_gib}/{p1_res_gib} GiB, post={post_p1_gib} GiB, rss={_host_rss_gib()} GiB)", flush=True)

        # ------------------------------------------------------------------
        # PHASE 2: Recipient 1.5B ONLY (Generic Cal + Gate G2 Supply & Outlet!)
        # ------------------------------------------------------------------
        t_p2 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        print("[Phase 2] Loading Recipient 1.5B alone (Donor 7B 100% UNLOADED!) ...", flush=True)
        model_B = None
        try:
            model_B, tok_B, device, _ = load_model_bf16(recipient_key, allow_download=allow_download)
            assert_zero_runtime_hooks(model_B)

            rec_cal = stream_recipient_moments_and_gamma(
                model_B, tok_B, device, corpus, don_cal, v_shared=v_shared,
                max_fit_tokens=55000, batch_size=16, max_length=128,
            )
            gamma_scale = float(rec_cal["gamma_scale_fit"])
            print(
                f"[Phase 2] Generic moments streamed ({rec_cal['n_fit_tokens']} fit / {rec_cal['n_val_report_tokens']} rep) | "
                f"gamma={gamma_scale:.4f} | Stock 1.5B top1 agree={rec_cal['stock_1p5b_top1_agree_with_7b_val_report']:.4f}",
                flush=True,
            )

            air_cohorts = {
                "all_eval": np.ones(len(air_eval_items), dtype=bool),
                "primary_endpoint": air_mask_p0_blind,   # Fixed p0-screened blindspot cohort on unseen phrasing p1!
                "primary_wrong": air_mask_p0_blind,
                "p0_known_retention": air_mask_p0_known,
                "p1_same_phrasing_wrong_cond": air_mask_p1_blind_cond,
            }
            g2_airports: Dict[str, Any] = {}
            for k_name, pl in g2_air_payloads.items():
                l_m = int(pl["layer"])
                g2_res = run_gate_g2_whole_mlp(
                    model_B, tok_B, device, pl, rec_cal["moments_by_site"][f"mlp_{l_m}"],
                    g0_bridges["T_raw"], gamma_scale, air_eval_items, air_eval_rec_ok_p1, air_cohorts,
                    "prompt", "first_ans_tok_id", "foil_tok_id",
                    lambda g, it: check_iata_strict_field(g, it["iata"]),
                    v_shared=v_shared, max_new_tokens=5, batch_size=32,
                )
                g2_airports[k_name] = g2_res
                fr = g2_res["frozen_r_from_val_tune"]
                print(
                    f"  [G2 Air {k_name}] frozen_r={fr} kappa={g2_res['supply_overall']['kappa_rep_all']:.4f} "
                    f"a*_cv={g2_res['supply_overall']['a_star_stats']['cv_rep']:.4f} "
                    f"blind_lift={g2_res['live_full_generation_at_frozen_r']['cohorts']['primary_endpoint']['em_lift_pp']}pp "
                    f"-> {g2_res['G2_VERDICT']}",
                    flush=True,
                )

            g2_2hop_heads = run_gate_g2_late_attn_heads(
                model_B, tok_B, device, res_g1_2h_heads["g2_payload"], rec_cal["moments_by_site"],
                g0_bridges["T_raw"], gamma_scale, ds_2hop["eval"], rec_2hop_eval_ok,
                "prompt", "answer_tok_id", "foil_tok_id", v_shared=v_shared, batch_size=32,
            )
            fr_h = g2_2hop_heads["frozen_r_from_val_tune"]
            print(
                f"  [G2 2-Hop Heads] frozen_r={fr_h} oracle_lift=+{g2_2hop_heads['test2a_true_donor_outlet_oracle']['live_tok0_lift_pp']:.2f}pp "
                f"WIR={g2_2hop_heads['test2b_student_supplied_by_rank'][f'r_{fr_h}']['wrong_item_ratio']} "
                f"live_lift=+{g2_2hop_heads['test2b_student_supplied_by_rank'][f'r_{fr_h}']['live_tok0_lift_pp']:.2f}pp "
                f"-> {g2_2hop_heads['G2_VERDICT']}",
                flush=True,
            )

            twohop_wrong = np.array([int(ok) == 0 for ok in rec_2hop_eval_ok], dtype=bool)
            twohop_cohorts = {
                "all_eval": np.ones(len(ds_2hop["eval"]), dtype=bool),
                "primary_endpoint": np.ones(len(ds_2hop["eval"]), dtype=bool),
                "primary_wrong": twohop_wrong,
                "known_retention": ~twohop_wrong,
            }
            g2_2hop_mlp = run_gate_g2_whole_mlp(
                model_B, tok_B, device, res_g1_2h_mlp["g2_payload"], rec_cal["moments_by_site"][f"mlp_{best_2hop_dla_mlp}"],
                g0_bridges["T_raw"], gamma_scale, ds_2hop["eval"], rec_2hop_eval_ok, twohop_cohorts,
                "prompt", "answer_tok_id", "foil_tok_id",
                lambda g, it: check_2hop_exact_word(g, it["answer"]),
                v_shared=v_shared, max_new_tokens=4, batch_size=32,
            )

            assert_zero_runtime_hooks(model_B)
            p2_peak_gib = round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 3) if device.type == "cuda" else 0.0
            p2_res_gib = round(torch.cuda.max_memory_reserved(device) / (1024 ** 3), 3) if device.type == "cuda" else 0.0
        finally:
            if model_B is not None:
                unload_model_vram_safe(model_B)
                del model_B
            gc.collect()
        post_p2_gib = round(torch.cuda.memory_allocated(device) / (1024 ** 3), 3) if device.type == "cuda" else 0.0
        assert post_p2_gib < 1.0, f"Phase 2 VRAM leak: {post_p2_gib} GiB"
    finally:
        tmp_obj.cleanup()

    # STRICT JOINT STAGE 2 CONTROLLER (`G0 AND G1 AND G2` per candidate!)
    g0_air_pass = bool(g0_bridge_pass and g0_headroom["airports"]["G0_HEADROOM_PASS"])
    g0_2h_pass = bool(g0_bridge_pass and g0_headroom["in_prompt_2hop"]["G0_HEADROOM_PASS"])
    any_air_pass = g0_air_pass and any(
        g1_airports[k]["g1_checks"]["G1_PASS"] and g2_airports[k]["G2_VERDICT"] == "PASS"
        for k in g1_airports
    )
    any_2h_pass = g0_2h_pass and (
        (res_g1_2h_heads["report"]["g1_checks"]["G1_PASS"] and g2_2hop_heads["G2_VERDICT"] == "PASS")
        or (res_g1_2h_mlp["report"]["g1_checks"]["G1_PASS"] and g2_2hop_mlp["G2_VERDICT"] == "PASS")
    )
    stage2_decision = "GO_TO_GATE_G3" if (any_air_pass or any_2h_pass) else "STOP_BEFORE_G3_NO_PASSING_ORGAN"

    result = {
        "experiment": "v5_0_boundary_organ_gates_g0_g2",
        "donor": donor_key, "recipient": recipient_key, "v_shared": v_shared,
        "strict_zero_sgd": True, "zero_task_fitted_writers": True,
        "runtime_sec": round(time.time() - t0, 2),
        "host_peak_rss_gib": _host_rss_gib(),
        "vram_gib": {
            "phase0_peak_alloc": p0_peak_gib, "phase0_peak_reserved": p0_res_gib, "phase0_post_unload": post_p0_gib,
            "phase1_peak_alloc": p1_peak_gib, "phase1_peak_reserved": p1_res_gib, "phase1_post_unload": post_p1_gib,
            "phase2_peak_alloc": p2_peak_gib, "phase2_peak_reserved": p2_res_gib, "phase2_post_unload": post_p2_gib,
        },
        "gate_g0": {
            "g0_bridge_pass": g0_bridge_pass,
            "g0_airports_pass": g0_air_pass,
            "g0_in_prompt_2hop_pass": g0_2h_pass,
            "headroom": g0_headroom,
            "generic_calibration": {
                "n_fit_pos_gt0_tokens": rec_cal["n_fit_tokens"],
                "n_val_report_pos_gt0_tokens": rec_cal["n_val_report_tokens"],
                "gamma_scale_fit": rec_cal["gamma_scale_fit"],
                "gamma_scale_rep": rec_cal["gamma_scale_rep"],
                "stock_1p5b_top1_agree_with_7b_val_report": rec_cal["stock_1p5b_top1_agree_with_7b_val_report"],
                "M_U_val_report": g0_bridges["M_U_val_report"],
                "T_raw_val_report": g0_bridges["T_raw_val_report"],
                "best_eps_rel_M_U": g0_bridges["best_eps_rel_M_U"],
                "best_eps_rel_T_raw": g0_bridges["best_eps_rel_T_raw"],
            },
            "all28_discovery_airports": disc_air,
            "all28_discovery_2hop": disc_2hop,
        },
        "gate_g1_donor_self_test": {
            "airports": g1_airports,
            "in_prompt_2hop": {
                "late_heads": res_g1_2h_heads["report"],
                f"L{best_2hop_dla_mlp}_MLP": res_g1_2h_mlp["report"],
            },
        },
        "gate_g2_student_supply_and_outlet": {
            "airports": g2_airports,
            "in_prompt_2hop": {
                "late_heads": g2_2hop_heads,
                f"L{best_2hop_dla_mlp}_MLP": g2_2hop_mlp,
            },
        },
        "stage2_gate_g3_decision": stage2_decision,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[v5.0] Completed in {result['runtime_sec']}s -> {out_path} | DECISION = {stage2_decision}", flush=True)
    return result


# ======================================================================================
# 8. Pure-CPU Unit Tests (9 Tests: Math <1e-6, Null-Space Guard, Startup Contracts!)
# ======================================================================================

def run_unit_tests() -> None:
    print("[v5.0 unit-test] Running pure-CPU mathematical, anti-leakage & contract verification ...", flush=True)
    torch.manual_seed(2026)
    rng = np.random.default_rng(2026)

    # Test 1: Raw pre-norm Fisher plug algebraic identity (<1e-10)
    V, dA, dB = 64, 12, 8
    E_A, E_B = rng.standard_normal((V, dA)), rng.standard_normal((V, dB))
    gA, gB = 0.5 + rng.random(dA), 0.5 + rng.random(dB)
    probs = np.exp(rng.standard_normal((20, V)))
    probs /= probs.sum(axis=1, keepdims=True)
    F_bar = np.mean([np.diag(p) - np.outer(p, p) for p in probs], axis=0)
    plugs = compute_raw_and_postnorm_fisher_plugs(E_A.T @ F_bar @ E_A, E_B.T @ F_bar @ E_B, E_B.T @ F_bar @ E_A, gA, gB, eps_rel_med=1e-6)
    J_A, J_B = E_A * gA[None, :], E_B * gB[None, :]
    assert np.allclose(plugs["G_A_raw"], J_A.T @ F_bar @ J_A, atol=1e-10)
    assert np.allclose(plugs["G_B_raw"], J_B.T @ F_bar @ J_B, atol=1e-10)
    assert np.allclose(plugs["G_BA_raw"], J_B.T @ F_bar @ J_A, atol=1e-10)
    print("  [PASS] Test 1: Raw pre-norm Fisher plug J_H = E_H diag(gamma_H) (<1e-10)")

    # Test 2: Closed-form MLP demand G vs autograd (<1e-6!)
    D, M, kF = 16, 24, 6
    Wg, Wu, Wd = torch.randn(M, D) * 0.3, torch.randn(M, D) * 0.3, torch.randn(D, M) * 0.3
    X_in, L_F = torch.randn(10, D), torch.randn(D, kF)
    G_cf = compute_mlp_demand_matrix_closed_form(Wg, Wu, Wd, X_in, L_F, chunk_size=4)
    G_ag = np.mean([
        (J := torch.autograd.functional.jacobian(lambda x: L_F.T @ (Wd @ (F.silu(Wg @ x) * (Wu @ x))), X_in[i]).double().numpy()).T @ J
        for i in range(len(X_in))
    ], axis=0)
    rel_mlp = float(np.linalg.norm(G_cf - G_ag) / (np.linalg.norm(G_ag) + 1e-12))
    assert rel_mlp < 1e-6, f"MLP closed-form G mismatch: {rel_mlp}"
    print(f"  [PASS] Test 2: Closed-form MLP demand G matches autograd (rel_err={rel_mlp:.2e} < 1e-6)")

    # Test 3: Closed-form Attention-Head demand G vs autograd for BOTH exclude_pos0=False & True (<1e-6!)
    D, dh, T, kF = 16, 8, 5, 4
    Wq, bq = torch.randn(dh, D) * 0.3, torch.randn(dh) * 0.1
    Wk, bk = torch.randn(dh, D) * 0.3, torch.randn(dh) * 0.1
    Wv, bv = torch.randn(dh, D) * 0.3, torch.randn(dh) * 0.1
    Wo, L_Fa = torch.randn(D, dh) * 0.3, torch.randn(D, kF)
    seqs = [torch.randn(T, D) for _ in range(4)]
    cos_t, sin_t = _make_rope_cos_sin(dh, 10000.0, T, torch.device("cpu"))
    for excl in (False, True):
        G_cf_a = compute_attn_head_demand_matrix_closed_form(Wq, bq, Wk, bk, Wv, bv, Wo, seqs, L_Fa, rope_theta=10000.0, exclude_pos0=excl)
        G_ag_a = np.zeros((D, D), dtype=np.float64)
        for X_s in seqs:
            Jf = torch.autograd.functional.jacobian(
                lambda xf: L_Fa.T @ _forward_single_attn_head_z_and_out(xf.view(T, D), Wq, bq, Wk, bk, Wv, bv, Wo, cos_t, sin_t)[1],
                X_s.reshape(-1),
            ).view(kF, T, D).double().numpy()
            for t in range(1 if excl else 0, T):
                G_ag_a += Jf[:, t, :].T @ Jf[:, t, :]
        G_ag_a /= float(len(seqs))
        rel_a = float(np.linalg.norm(G_cf_a - G_ag_a) / (np.linalg.norm(G_ag_a) + 1e-12))
        assert rel_a < 1e-6, f"Attn G mismatch (exclude_pos0={excl}): {rel_a}"
    print(f"  [PASS] Test 3: Closed-form Attention-Head demand G (exclude_pos0=False & True, rel_err={rel_a:.2e} < 1e-6)")

    # Test 4: Demand-weighted supply, val_tune frozen_r, a* CV, & Astra null-space leak guard!
    N_s, dA_s, dB_s = 600, 20, 12
    Y_raw = rng.standard_normal((N_s, dB_s))
    Y_raw[:, 0] = 5.0 + 0.02 * rng.standard_normal(N_s)
    True_A = rng.standard_normal((dA_s, dB_s)) * 0.5
    X_raw = Y_raw @ True_A.T + 0.02 * rng.standard_normal((N_s, dA_s)) + rng.standard_normal((1, dA_s)) * 2.0
    G_s = rng.standard_normal((dA_s, 6))
    G_s = G_s @ G_s.T
    mu_x, mu_y = X_raw[:400].mean(axis=0), Y_raw[:400].mean(axis=0)
    Xc, Yc = X_raw[:400] - mu_x, Y_raw[:400] - mu_y
    sup = compute_demand_weighted_supply_boundary(
        (Xc.T @ Xc) / 400.0, (Xc.T @ Yc) / 400.0, (Yc.T @ Yc) / 400.0, mu_x, mu_y, G_s,
        X_raw[400:500], Y_raw[400:500], X_raw[500:], Y_raw[500:], ranks=(4, 6, 8),
    )
    assert sup["frozen_r_tune"] in (4, 6) and sup["by_rank"][sup["frozen_r_tune"]]["kappa_r_tune"] >= 0.85
    assert sup["by_rank"][6]["kappa_r_rep"] > 0.98 and sup["a_star_stats"]["passes_2pct"]
    # Verify Astra null-space leak guard catches systematic bias in a constant target coordinate!
    U_const = np.eye(dA_s, 2)
    X_const = np.zeros((100, dA_s))
    X_const[:, 0] = rng.standard_normal(100)
    X_const[:, 1] = 3.0  # constant coordinate (zero variance!)
    Y_zero = np.zeros((100, dB_s))
    Y_zero[:, 0] = X_const[:, 0]
    asup_leak = compute_astra_zero_intercept_supply(
        (Y_zero.T @ Y_zero) / 100.0 + 1e-6 * np.eye(dB_s), (X_const.T @ Y_zero) / 100.0, U_const,
        X_const, Y_zero, X_const, Y_zero,
    )
    assert asup_leak["natural"]["null_space_leak"] is True and asup_leak["natural"]["r2_worst_unrestricted"] is None
    # Also verify Astra's exact (1, 1e-5) orthogonal low-amplitude probe catches r2_worst_unrestricted <= 0.01!
    X_low = np.zeros((100, dA_s))
    X_low[:, 0] = rng.standard_normal(100)
    X_low[:, 1] = 1e-5 * rng.standard_normal(100)
    Y_low = np.zeros((100, dB_s))
    Y_low[:, 0] = X_low[:, 0]
    asup_low = compute_astra_zero_intercept_supply(
        (Y_low.T @ Y_low) / 100.0 + 1e-6 * np.eye(dB_s), (X_low.T @ Y_low) / 100.0, U_const,
        X_low, Y_low, X_low, Y_low,
    )
    assert asup_low["natural"]["r2_worst_unrestricted"] is not None and asup_low["natural"]["r2_worst_unrestricted"] < 0.05
    # Verify scale invariance: scaling BOTH targets by 1e-5 still catches the unpredicted 1e-10 coord!
    X_scaled = X_low * 1e-5
    asup_scaled = compute_astra_zero_intercept_supply(
        (Y_low.T @ Y_low) / 100.0 + 1e-6 * np.eye(dB_s), (X_scaled.T @ Y_low) / 100.0, U_const,
        X_scaled, Y_low, X_scaled, Y_low,
    )
    assert asup_scaled["natural"]["r2_worst_unrestricted"] is not None and asup_scaled["natural"]["r2_worst_unrestricted"] < 0.05
    print(f"  [PASS] Test 4: Supply (frozen_r={sup['frozen_r_tune']}, a*_cv={sup['a_star_stats']['cv_rep']:.4f}), null-space & scaled (1e-5,1e-10) guards verified")

    # Test 5: Forward cancellation-aware ladder beats Wanda on exact cancelling pairs
    A_sig, B_sig = torch.randn(64, 8), torch.randn(6, 8)
    A_p, B_p = torch.randn(64, 12) * 10.0, torch.randn(6, 12) * 10.0
    lad = run_width_compression_ladder(torch.cat([A_sig, A_p, A_p], 1), torch.cat([B_sig, B_p, -B_p], 1), ladder=(32, 16, 8))
    assert lad["cancel_rel_err"]["8"] < 1e-4 and lad["wanda_rel_err"]["8"] > 0.90
    print(f"  [PASS] Test 5: Forward cancellation ladder (m=8 err: cancel={lad['cancel_rel_err']['8']} vs wanda={lad['wanda_rel_err']['8']})")

    # Test 6: Safe Q/R2 & signed wrong_item_ratio guards (NEVER reward -1/-1=+1 or empty swap Q=1!)
    empty_qr = _compute_safe_q_and_r2(np.zeros(0), np.zeros(0), np.zeros(5), np.zeros(5))
    assert empty_qr["Q_causal"] is None and empty_qr["R2_c"] is None
    neg_wir = _compute_safe_wrong_item_ratio(np.array([-1.0, -1.0]), np.array([-1.0, -1.0]), np.array([True, True]))
    assert neg_wir["ratio_supported"] is False and neg_wir["wrong_item_ratio"] is None
    assert not check_2hop_exact_word("r2ed extra", "red") and check_2hop_exact_word(" red.", "red")
    assert not check_iata_strict_field("The code is LHR", "LHR") and check_iata_strict_field(" LHR.", "LHR")
    assert _select_2hop_difficulty_from_pilot(0.80, 0.50, 0.50, 0.10) == 4  # filters <0.65 first!
    print("  [PASS] Test 6: Empty/zero Q guard, signed WIR guard, strict scorers, & eligible pilot selector verified")

    # Test 7: Single-partition 2-Hop disjointness across 4-row AND 6-row + helper return contracts!
    local_tok_dir = Path("/tmp/platos_key_lab/hf_cache/qwen2.5-3b")
    if local_tok_dir.is_dir():
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(str(local_tok_dir))
        kp = partition_2hop_key_pools([tk, tk], max_rows=6, seed=SEED)
        d4 = build_in_prompt_2hop_dataset([tk, tk], kp, n_rows=4, n_pilot=12, n_disc=20, n_eval=24)
        d6 = build_in_prompt_2hop_dataset([tk, tk], kp, n_rows=6, n_pilot=12, n_disc=20, n_eval=24)
        p_keys_all = {it["query"] for it in d4["pilot"] + d6["pilot"]} | {it["mid"] for it in d4["pilot"] + d6["pilot"]}
        e_keys_all = {it["query"] for it in d4["eval"] + d6["eval"]} | {it["mid"] for it in d4["eval"] + d6["eval"]}
        assert p_keys_all.isdisjoint(e_keys_all), "Cross-difficulty pilot/eval key leak!"
        # Also verify airport helper 2-tuple contracts & paired matcher!
        rows_ap, src_ap = fetch_or_fallback_airports(max_airports=30, offline=True)
        enr_ap = attach_airport_answer_tokens(rows_ap, tk, phrasing_idx=0)
        pairs_ap, fmap_ap = build_matched_airport_swap_pairs(enr_ap)
        assert len(pairs_ap) >= 20 and all(enr_ap[s]["first_ans_tok_id"] != enr_ap[b]["first_ans_tok_id"] for s, b in pairs_ap)
        p_skew, fm_skew = build_matched_airport_swap_pairs([{"first_ans_tok_id": t} for t in (1, 1, 1, 1, 2)])
        assert len(p_skew) == 2 and len(fm_skew) == 2, f"Skewed matcher failed: {p_skew}"
        assert isinstance(LARGE_TO_SMALL_MODELS[DONOR_KEY_DEFAULT], str)
        print(f"  [PASS] Test 7: Cross-difficulty 2-Hop key disjointness + Airport/Model helper contracts verified ({src_ap})")

    # Test 8: Synthetic Qwen2ForCausalLM hook compatibility + exact single-head RoPE parity (<1e-6!)
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg_t = Qwen2Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        rope_theta=1000000.0, tie_word_embeddings=True,
    )
    m_tiny = Qwen2ForCausalLM(cfg_t).eval()
    ids_t = torch.tensor([[5, 12, 23, 34, 45]], dtype=torch.long)
    cap_in, cap_z = [], []
    h_a = m_tiny.model.layers[1].input_layernorm.register_forward_hook(lambda _m, _i, o: cap_in.append(o.detach()))
    h_b = m_tiny.model.layers[1].self_attn.o_proj.register_forward_pre_hook(lambda _m, inp: cap_z.append(inp[0].detach()))
    try:
        with torch.no_grad():
            m_tiny(input_ids=ids_t, attention_mask=torch.ones_like(ids_t), use_cache=False)
    finally:
        h_a.remove(); h_b.remove()
    assert_zero_runtime_hooks(m_tiny)
    attn1 = m_tiny.model.layers[1].self_attn
    dh_t, g_t, hi_t = int(attn1.head_dim), int(attn1.num_key_value_groups), 3
    kvi_t = hi_t // g_t
    ref_out = cap_z[0][0, -1, hi_t * dh_t : (hi_t + 1) * dh_t] @ attn1.o_proj.weight[:, hi_t * dh_t : (hi_t + 1) * dh_t].T
    c_t, s_t = _make_rope_cos_sin(dh_t, get_config_rope_theta(cfg_t), ids_t.shape[1], torch.device("cpu"))
    _, our_out, _ = _forward_single_attn_head_z_and_out(
        cap_in[0][0],
        attn1.q_proj.weight[hi_t * dh_t : (hi_t + 1) * dh_t],
        attn1.q_proj.bias[hi_t * dh_t : (hi_t + 1) * dh_t] if attn1.q_proj.bias is not None else None,
        attn1.k_proj.weight[kvi_t * dh_t : (kvi_t + 1) * dh_t],
        attn1.k_proj.bias[kvi_t * dh_t : (kvi_t + 1) * dh_t] if attn1.k_proj.bias is not None else None,
        attn1.v_proj.weight[kvi_t * dh_t : (kvi_t + 1) * dh_t],
        attn1.v_proj.bias[kvi_t * dh_t : (kvi_t + 1) * dh_t] if attn1.v_proj.bias is not None else None,
        attn1.o_proj.weight[:, hi_t * dh_t : (hi_t + 1) * dh_t],
        c_t, s_t,
    )
    diff_h = float(torch.max(torch.abs(ref_out - our_out)).item())
    assert diff_h < 1e-6, f"Qwen2ForCausalLM single-head RoPE mismatch: {diff_h}"
    # Also verify _native_bf16_argmax_shared matches float argmax cleanly
    h_rand = torch.randn(6, 32)
    p_chunk = _native_bf16_argmax_shared(h_rand, m_tiny.lm_head.weight.detach(), chunk=2)
    p_ref = torch.argmax(F.linear(h_rand, m_tiny.lm_head.weight.detach()), dim=-1).tolist()
    assert p_chunk == p_ref
    # Verify stateless prefill (`x.shape[1] > 1`) vs cached decode (`x.shape[1] == 1`) sequence lengths during generate!
    seen_seq_lens: List[int] = []
    h_sl = m_tiny.model.layers[1].self_attn.o_proj.register_forward_pre_hook(lambda _m, inp: seen_seq_lens.append(int(inp[0].shape[1])))
    try:
        with torch.no_grad():
            m_tiny.generate(input_ids=ids_t, attention_mask=torch.ones_like(ids_t), max_new_tokens=4, do_sample=False, repetition_penalty=1.0, use_cache=True, pad_token_id=0)
    finally:
        h_sl.remove()
    assert seen_seq_lens == [5, 1, 1, 1], f"Unexpected cached decode lengths: {seen_seq_lens}"
    # Verify left-padded forward() with _left_pad_position_ids matches generate() argmax BIT-EXACTLY in BF16!
    m_bf = m_tiny.to(torch.bfloat16)
    ids_lp = torch.tensor([[0, 0, 0, 5, 12, 23], [7, 11, 13, 17, 19, 29]], dtype=torch.long)
    msk_lp = torch.tensor([[0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1]], dtype=torch.long)
    with torch.no_grad():
        lg_fwd = m_bf(input_ids=ids_lp, attention_mask=msk_lp, position_ids=_left_pad_position_ids(msk_lp), use_cache=False).logits[:, -1, :]
        gd_lp = m_bf.generate(input_ids=ids_lp, attention_mask=msk_lp, max_new_tokens=1, do_sample=False, repetition_penalty=1.0, use_cache=True, pad_token_id=0)
    assert torch.equal(lg_fwd.argmax(dim=-1), gd_lp[:, -1]), "Left-padded forward(position_ids) vs generate() mismatch!"
    assert_zero_runtime_hooks(m_bf)
    print(f"  [PASS] Test 8: Tiny Qwen2ForCausalLM hooks, RoPE parity ({diff_h:.2e}), left-pad pos parity & prefill/decode ({seen_seq_lens}) verified")

    # Test 9: Cancellation chi rank-revealing SVD & negligible-NL flag
    chi_r = compute_cancellation_ratio_chi(Wg, Wu, Wd, X_in, rng.standard_normal((len(X_in), 3)), L_F)
    assert chi_r["chi_nl"] is not None and chi_r["chi_nl"] >= 1.0
    print(f"  [PASS] Test 9: Rank-revealing cancellation chi_nl={chi_r['chi_nl']} verified")
    print("[v5.0 unit-test] ALL 9 PURE-CPU UNIT TESTS PASSED!", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="v5.0 Boundary-First Causal Platonic Organ Gates G0 -> G1 -> G2")
    parser.add_argument("--unit-test", action="store_true", help="Run pure-CPU unit tests (zero GPU/downloads)")
    parser.add_argument("--run-all", action="store_true", help="Run full Stage 1 Gates G0 -> G1 -> G2 on GPU")
    parser.add_argument("--allow-download", action="store_true", help="Allow HuggingFace Hub downloads when not cached")
    parser.add_argument("--out", type=str, default=str(OUT_PATH), help="Output JSON path")
    args = parser.parse_args()
    if args.unit_test:
        run_unit_tests()
        return
    if args.run_all:
        run_all_gates_g0_g2(out_path=Path(args.out), allow_download=bool(args.allow_download))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
