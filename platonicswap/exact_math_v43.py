#!/usr/bin/env python3
"""
exact_math_v43.py — Unanimous Fable 5.2 Max + Astra 6.2 Ultra Exact-Math Primitives (v4.3)
==========================================================================================

Implements the exact mathematical optima for rectangular `D_A = 3584 <-> D_B = 1536` native
in-weights translation (`Qwen/Qwen2.5-7B-Instruct` -> `Qwen/Qwen2.5-1.5B-Instruct`):
  - Drops Whitened CCA + Procrustes (proven suboptimal vs OLS; confirmed empirically on L4) and
    drops bias-free SwiGLU RBF anchor kernels.
  1. Streaming FP64 Second-Moment Calibration (`N_fit = 60,000+`, `N_val >= 5,000`, ZERO RAM bloat!):
     Accumulates exact uncentered `S_AA`, `S_BB`, `S_BA` in FP64 across batches, with ridge `λ`
     scaled relative to the **median** eigenvalue and selected strictly on template-disjoint `N_val`.
  2. Exact Per-Reader Observability Score `o_j` via Schur Complement `S_{A|B}`:
       `Pt = solve(S_BB + lam*I, S_BA, assume_a='pos')`
       `S_AgB = S_AA - S_BA.T @ Pt`
       `o_j = 1.0 - ((R_A @ S_AgB) * R_A).sum(1) / ((R_A @ S_AA) * R_A).sum(1)`
     Used to score SwiGLU `(gate, up)` neurons and Attention `(Q, K, V)` heads jointly with task
     salience so `Tier 1B` NEVER transplants deaf 7B units!
  3. Exact Zero-Data Shared-Vocabulary Bridge `M_U` at Final Layer `L27` (ZERO `V x V` matrix!):
     Uses exact Fisher metrics `G_B`, `G_BA`, `G_A` streamed from Donor 7B calibration softmax `p_n`:
       `QA[n] = U_A^T p_n`, `QB[n] = U_B^T p_n`
       `G_B  = U_B^T diag(pbar) U_B - QB^T QB / N`
       `G_BA = U_B^T diag(pbar) U_A - QB^T QA / N`
       `M_U  = solve(G_B + eps*I, G_BA, assume_a='pos')`
     Plus alignment-free Gauss-Newton logit-gap writer target `tau_GN = -(G_B + lam*I)^{-1} U_B^T (p_B - p_A)`.
  4. The 3 Exact Floor Diagnostics (Gate 0B):
     - Floor #1: Retained Fisher Fraction at `k = 1536` via `fisher_rrr(S_AA, G_A, k=1536)`
     - Floor #2: Per-layer median / p90 / top-K read observability `o_j` + held-out patched KL
       recovery `R = 1 - KL(ref || patched) / KL(ref || mean_ablated)` on push, pull, and round-trip!
     - Floor #3: Nonlinear Residual Gain `Delta_NL` (tuned linear Ridge vs linear + Nyström RBF
       kernel ridge on the linear residual, strictly tuned on `cal_tune` and reported on `cal_report`!).
  5. Shared-Suffix Multi-Token Alignment helper (`align_shared_suffix_mask`) for Task C (2-Hop)
     so Attention heads are compiled across ALL table tokens matching `recipient_prompt`, not just the last token!

Usage:
  PYTHONPATH=platonicswap:experiments python platonicswap/exact_math_v43.py --unit-test
"""

from __future__ import annotations

import gc
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for _p in (str(THIS_DIR), str(REPO_ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from extract_and_baseline import get_decoder_layers
from native_weight_translation import (
    ReversibleNativeOrganPack,
    assert_zero_runtime_hooks,
    bf16_to_f64_np,
    solve_closed_form_ridge_with_bg,
)


# ======================================================================================
# 1. Streaming FP64 Second-Moment Accumulator & Median-Eigenvalue Val-Selected Ridge
# ======================================================================================

class StreamingSecondMomentAccumulator:
    """
    Streaming FP64 uncentered second-moment accumulator (`S_AA`, `S_BB`, `S_BA`) for `N_fit = 60,000+`
    tokens with ZERO large token-matrix RAM or disk bloat (`~165 MB` per layer/field!).
    """

    def __init__(self, d_a: int, d_b: int):
        self.d_a = int(d_a)
        self.d_b = int(d_b)
        self.n: int = 0
        self.sum_aa = np.zeros((d_a, d_a), dtype=np.float64)
        self.sum_bb = np.zeros((d_b, d_b), dtype=np.float64)
        self.sum_ba = np.zeros((d_b, d_a), dtype=np.float64)
        self.sum_a = np.zeros(d_a, dtype=np.float64)
        self.sum_b = np.zeros(d_b, dtype=np.float64)

    def update(self, X_a: torch.Tensor, X_b: torch.Tensor) -> None:
        """Accumulate a paired `(N_batch, D_A)` and `(N_batch, D_B)` batch in strict FP64."""
        assert X_a.shape[0] == X_b.shape[0]
        if X_a.shape[0] == 0:
            return
        a = X_a.detach().to(dtype=torch.float64).cpu().numpy()
        b = X_b.detach().to(dtype=torch.float64).cpu().numpy()
        self.n += int(a.shape[0])
        self.sum_aa += a.T @ a
        self.sum_bb += b.T @ b
        self.sum_ba += b.T @ a
        self.sum_a += a.sum(axis=0)
        self.sum_b += b.sum(axis=0)

    def moments(self, inplace: bool = False) -> Dict[str, np.ndarray]:
        assert self.n > 0, "Empty StreamingSecondMomentAccumulator!"
        inv_n = 1.0 / float(self.n)
        if inplace:
            self.sum_aa *= inv_n
            self.sum_bb *= inv_n
            self.sum_ba *= inv_n
            self.sum_a *= inv_n
            self.sum_b *= inv_n
            out = {
                "S_AA": self.sum_aa,
                "S_BB": self.sum_bb,
                "S_BA": self.sum_ba,
                "mean_A": self.sum_a,
                "mean_B": self.sum_b,
            }
            self.sum_aa = self.sum_bb = self.sum_ba = self.sum_a = self.sum_b = None  # type: ignore[assignment]
            return out
        return {
            "S_AA": self.sum_aa * inv_n,
            "S_BB": self.sum_bb * inv_n,
            "S_BA": self.sum_ba * inv_n,
            "mean_A": self.sum_a * inv_n,
            "mean_B": self.sum_b * inv_n,
        }


def fit_ridge_from_moments_median_floor(
    S_xx: np.ndarray,
    S_yx: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    n_fit: int,
    lam_grid: Sequence[float] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve `W = S_yx @ (S_xx + lam_abs * I)^{-1}` (`Y ≈ X @ W.T`) in strict FP64 with `lam_abs`
    scaled relative to the **median** positive eigenvalue of `S_xx` (Fable+Astra exact recipe —
    immune to top-outlier spikes!) and selected strictly on held-out `X_val, Y_val`.
    """
    Xv = bf16_to_f64_np(X_val)
    Yv = bf16_to_f64_np(Y_val)
    D_in = S_xx.shape[0]
    D_out = S_yx.shape[0]

    eigvals, Q = sla.eigh(S_xx)
    eigvals = np.maximum(eigvals, 0.0)
    pos = eigvals[eigvals > 1e-12]
    med_eig = float(np.median(pos)) if len(pos) > 0 else float(np.trace(S_xx) / max(D_in, 1))
    p_spec = pos / (np.sum(pos) + 1e-12)
    erank = float(np.exp(-np.sum(p_spec * np.log(p_spec + 1e-12))))
    Qt_Sxy = Q.T @ S_yx.T  # (D_in, D_out)

    yv_norm = np.linalg.norm(Yv, axis=1) + 1e-12
    yv_var = float(np.sum((Yv - Yv.mean(axis=0)) ** 2)) + 1e-12

    best_W: Optional[np.ndarray] = None
    best_r2 = -1e9
    best_cos = -1e9
    best_lam_rel = float(lam_grid[0])
    best_lam_abs = float(lam_grid[0]) * med_eig

    for lam_rel in lam_grid:
        lam_abs = max(float(lam_rel) * med_eig, 1e-12)
        W_t = Q @ (Qt_Sxy / (eigvals[:, None] + lam_abs))
        pred = Xv @ W_t
        r2 = 1.0 - float(np.sum((Yv - pred) ** 2) / yv_var)
        cos_v = float(np.mean(np.sum(pred * Yv, axis=1) / (np.linalg.norm(pred, axis=1) * yv_norm + 1e-12)))
        if r2 > best_r2:
            best_r2 = r2
            best_cos = cos_v
            best_lam_rel = float(lam_rel)
            best_lam_abs = lam_abs
            best_W = W_t.T  # (D_out, D_in)

    assert best_W is not None
    return best_W, {
        "n_fit": int(n_fit),
        "n_val": int(len(Xv)),
        "d_in": int(D_in),
        "d_out": int(D_out),
        "median_eig": float(med_eig),
        "best_lam_rel_med": best_lam_rel,
        "best_lam_abs": float(best_lam_abs),
        "val_cos": round(best_cos, 4),
        "val_r2": round(best_r2, 4),
        "erank": round(erank, 2),
    }


# ======================================================================================
# 2. Exact Per-Reader Observability Score `o_j` (Schur Complement S_{A|B})
# ======================================================================================

def compute_schur_complement_s_agb(
    S_AA: np.ndarray,
    S_BB: np.ndarray,
    S_BA: np.ndarray,
    lam_abs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact Fable+Astra pullback `Pt = (S_BB + lam*I)^{-1} S_BA` (`P = Pt.T`) and Schur complement:
      `S_AgB = S_AA - S_BA.T @ Pt` (residual unobservable covariance of Donor A given Recipient B!).
    """
    D_B = S_BB.shape[0]
    Pt = sla.solve(S_BB + float(lam_abs) * np.eye(D_B, dtype=np.float64), S_BA, assume_a="pos")  # (D_B, D_A)
    S_AgB = S_AA - S_BA.T @ Pt
    # Symmetrize cleanly for numerical hygiene
    S_AgB = 0.5 * (S_AgB + S_AgB.T)
    return Pt.T, S_AgB  # P: (D_A, D_B), S_AgB: (D_A, D_A)


def compute_reader_observability(
    R_A: np.ndarray,
    S_AA: np.ndarray,
    S_AgB: np.ndarray,
) -> np.ndarray:
    """
    Exact per-reader observability score `o_j` for rows of `R_A` `(M, D_A)`:
      `o_j = 1.0 - ((R_A @ S_AgB) * R_A).sum(1) / ((R_A @ S_AA) * R_A).sum(1)`
    Equals `1.0` when reader `r_j` is 100% predictable from Recipient B's 1,536-D stream, and `0.0`
    when `r_j` lives entirely in the 2,048-D blind spot!
    """
    R = np.asarray(R_A, dtype=np.float64)
    num = np.sum((R @ S_AgB) * R, axis=1)
    den = np.sum((R @ S_AA) * R, axis=1)
    # Zero-variance / undefined-denominator readers are STRICTLY rejected (`o_j = 0.0`, never `1.0`!).
    valid = den > 1e-18
    safe_den = np.where(valid, den, 1.0)
    o = np.where(valid, 1.0 - (num / safe_den), 0.0)
    return np.clip(o, 0.0, 1.0)


def score_layer_donor_observability(
    donor_layer: nn.Module,
    S_AA_attn: np.ndarray,
    S_AgB_attn: np.ndarray,
    S_AA_mlp: np.ndarray,
    S_AgB_mlp: np.ndarray,
) -> Dict[str, Any]:
    """Compute exact Floor #2 per-reader observability `o_j` for ALL SwiGLU neurons and Attention heads."""
    attn = donor_layer.self_attn
    mlp = donor_layer.mlp
    d_h = int(attn.head_dim)
    n_q = int(attn.q_proj.out_features // d_h)
    n_kv = int(attn.k_proj.out_features // d_h)
    g_A = int(attn.num_key_value_groups)

    Wq = attn.q_proj.weight.detach().float().cpu().numpy()
    Wk = attn.k_proj.weight.detach().float().cpu().numpy()
    Wv = attn.v_proj.weight.detach().float().cpu().numpy()
    Wg = mlp.gate_proj.weight.detach().float().cpu().numpy()
    Wu = mlp.up_proj.weight.detach().float().cpu().numpy()

    o_q_rows = compute_reader_observability(Wq, S_AA_attn, S_AgB_attn)
    o_k_rows = compute_reader_observability(Wk, S_AA_attn, S_AgB_attn)
    o_v_rows = compute_reader_observability(Wv, S_AA_attn, S_AgB_attn)

    o_q_head = np.array([float(np.mean(o_q_rows[h * d_h : (h + 1) * d_h])) for h in range(n_q)])
    o_k_kv = np.array([float(np.mean(o_k_rows[kv * d_h : (kv + 1) * d_h])) for kv in range(n_kv)])
    o_v_kv = np.array([float(np.mean(o_v_rows[kv * d_h : (kv + 1) * d_h])) for kv in range(n_kv)])
    o_head_joint = np.array([
        float((o_q_head[h] * o_k_kv[h // g_A] * o_v_kv[h // g_A]) ** (1.0 / 3.0)) for h in range(n_q)
    ])
    o_head_min = np.array([
        float(min(o_q_head[h], o_k_kv[h // g_A], o_v_kv[h // g_A])) for h in range(n_q)
    ])

    o_gate = compute_reader_observability(Wg, S_AA_mlp, S_AgB_mlp)
    o_up = compute_reader_observability(Wu, S_AA_mlp, S_AgB_mlp)
    o_mlp_joint = np.sqrt(o_gate * o_up)
    o_mlp_min = np.minimum(o_gate, o_up)

    return {
        "o_q_head": o_q_head,
        "o_k_kv": o_k_kv,
        "o_v_kv": o_v_kv,
        "o_head_joint": o_head_joint,
        "o_head_min": o_head_min,
        "o_gate": o_gate,
        "o_up": o_up,
        "o_mlp_joint": o_mlp_joint,
        "o_mlp_min": o_mlp_min,
        "summary": {
            "mlp_median_o": round(float(np.median(o_mlp_joint)), 4),
            "mlp_p90_o": round(float(np.percentile(o_mlp_joint, 90)), 4),
            "mlp_top256_mean_o": round(float(np.mean(np.sort(o_mlp_joint)[-256:])), 4),
            "mlp_n_all_constituents_ge_025": int(np.sum(o_mlp_min >= 0.25)),
            "attn_qkv_median_o": round(float(np.median(o_head_joint)), 4),
            "attn_qkv_p90_o": round(float(np.percentile(o_head_joint, 90)), 4),
            "attn_qkv_max_o": round(float(np.max(o_head_joint)), 4),
            "attn_n_all_constituents_ge_025": int(np.sum(o_head_min >= 0.25)),
        },
    }


# ======================================================================================
# 3. Exact Zero-Data Shared-Vocab Bridge `M_U`, Floor #1 Fisher-RRR & Gauss-Newton Target
# ======================================================================================

class StreamingSharedVocabFisherAccumulator:
    """
    Zero-extra-VRAM streaming shared-vocabulary Fisher accumulator (`G_A, G_B, G_BA`) with ZERO
    `V x V` matrix and ZERO FP32/FP64 full-unembedding GPU copies!
      - Holds zero-copy references to existing model BF16 `U_A[:v_shared]` & `U_B[:v_shared]` (`0 B` extra VRAM!).
      - Streams `probs @ U_A` and `probs @ U_B` in `vocab_chunk = 16,384` chunks (`< 0.25 GiB` peak VRAM!).
      - Computes `probs.sum(dim=0, dtype=torch.float64)` without allocating a full FP64 `probs` tensor!
      - Finalizes `G_A, G_B, G_BA` chunk-by-chunk on CPU (`0 B` GPU allocation during `finalize()`!).
    """

    def __init__(self, U_A_shared: torch.Tensor, U_B_shared: torch.Tensor, vocab_chunk: int = 16384):
        self.v_shared, self.d_a = int(U_A_shared.shape[0]), int(U_A_shared.shape[1])
        self.d_b = int(U_B_shared.shape[1])
        assert U_B_shared.shape[0] == self.v_shared
        self.U_A_ref = U_A_shared.detach()  # Zero-copy BF16 view on existing model weight!
        self.U_B_ref = U_B_shared.detach()  # Zero-copy BF16 view on existing model weight!
        self.vocab_chunk = int(vocab_chunk)
        self.n: int = 0
        self.sum_p = np.zeros(self.v_shared, dtype=np.float64)
        self.sum_qaqa = np.zeros((self.d_a, self.d_a), dtype=np.float64)
        self.sum_qbqb = np.zeros((self.d_b, self.d_b), dtype=np.float64)
        self.sum_qbqa = np.zeros((self.d_b, self.d_a), dtype=np.float64)

    @torch.no_grad()
    def update(self, donor_logits_shared: torch.Tensor) -> None:
        if donor_logits_shared.shape[0] == 0:
            return
        dev = self.U_A_ref.device
        probs = F.softmax(donor_logits_shared.to(device=dev, dtype=torch.float32), dim=-1)  # (N_tok, V)
        n_tok = int(probs.shape[0])
        # Streamed vocab-chunked matmul + chunked FP64 prob reduction (NEVER allocates full FP64 probs or overlapping weight chunks!)
        qa_gpu = torch.zeros((n_tok, self.d_a), device=dev, dtype=torch.float32)
        qb_gpu = torch.zeros((n_tok, self.d_b), device=dev, dtype=torch.float32)
        dev_b = self.U_B_ref.device
        for i0 in range(0, self.v_shared, self.vocab_chunk):
            i1 = min(i0 + self.vocab_chunk, self.v_shared)
            pc = probs[:, i0:i1]
            self.sum_p[i0:i1] += pc.sum(dim=0, dtype=torch.float64).cpu().numpy()
            ua_c = self.U_A_ref[i0:i1].float()
            qa_gpu.addmm_(pc, ua_c)
            del ua_c
            ub_c = self.U_B_ref[i0:i1].to(device=dev, dtype=torch.float32) if dev_b != dev else self.U_B_ref[i0:i1].float()
            qb_gpu.addmm_(pc, ub_c)
            del ub_c
        qa = qa_gpu.double().cpu().numpy()
        qb = qb_gpu.double().cpu().numpy()
        self.n += n_tok
        self.sum_qaqa += qa.T @ qa
        self.sum_qbqb += qb.T @ qb
        self.sum_qbqa += qb.T @ qa
        del probs, qa_gpu, qb_gpu

    def finalize(self) -> Dict[str, np.ndarray]:
        assert self.n > 0, "Empty StreamingSharedVocabFisherAccumulator!"
        inv_n = 1.0 / float(self.n)
        pbar = self.sum_p * inv_n
        G_A = np.zeros((self.d_a, self.d_a), dtype=np.float64)
        G_B = np.zeros((self.d_b, self.d_b), dtype=np.float64)
        G_BA = np.zeros((self.d_b, self.d_a), dtype=np.float64)
        # Copy each small vocab chunk to CPU FIRST in BF16, then cast to FP64 on CPU (ZERO GPU alloc!)
        for i0 in range(0, self.v_shared, self.vocab_chunk):
            i1 = min(i0 + self.vocab_chunk, self.v_shared)
            p_c = pbar[i0:i1]
            ua_c = self.U_A_ref[i0:i1].detach().cpu().double().numpy()
            ub_c = self.U_B_ref[i0:i1].detach().cpu().double().numpy()
            w_ua = p_c[:, None] * ua_c
            w_ub = p_c[:, None] * ub_c
            G_A += ua_c.T @ w_ua
            G_B += ub_c.T @ w_ub
            G_BA += ub_c.T @ w_ua
        G_A = 0.5 * ((G_A - self.sum_qaqa * inv_n) + (G_A - self.sum_qaqa * inv_n).T)
        G_B = 0.5 * ((G_B - self.sum_qbqb * inv_n) + (G_B - self.sum_qbqb * inv_n).T)
        G_BA = G_BA - self.sum_qbqa * inv_n
        return {"G_A": G_A, "G_B": G_B, "G_BA": G_BA, "pbar": pbar}


def solve_shared_vocab_bridge_m_u(
    G_B: np.ndarray,
    G_BA: np.ndarray,
    eps_rel_med: float = 1e-2,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Exact zero-data shared-vocabulary bridge `M_U = (G_B + eps*I)^{-1} G_BA` in post-final-RMSNorm
    coordinates (`x_B_norm ≈ M_U @ x_A_norm`)!
    """
    eb = sla.eigvalsh(G_B)
    pos = eb[eb > 1e-12]
    med_b = float(np.median(pos)) if len(pos) > 0 else float(np.trace(G_B) / max(G_B.shape[0], 1))
    eps_abs = max(float(eps_rel_med) * med_b, 1e-10)
    M_U = sla.solve(G_B + eps_abs * np.eye(G_B.shape[0], dtype=np.float64), G_BA, assume_a="pos")
    return M_U, {"eps_rel_med": float(eps_rel_med), "median_eig_gb": round(med_b, 6), "eps_abs": round(eps_abs, 8)}


def compute_fisher_rrr_retained_fraction(
    S_AA: np.ndarray,
    G_A: np.ndarray,
    ks: Sequence[int] = (256, 512, 1024, 1536),
) -> Dict[str, Any]:
    """
    Floor #1 — Exact Retained Fisher Fraction at `k = 1,536` (`fisher_rrr(S_AA, G_A, k)`):
    Computes singular values `s` of `S_AA^{1/2} G_A^{1/2}` (exact generalized Fisher spectrum of
    Donor 7B's functional logit information under `N_fit = 60,000+` calibration states):
      `retained(k) = (s[:k]**2).sum() / (s**2).sum()`.
    """
    es, Vs = sla.eigh(0.5 * (S_AA + S_AA.T))
    eg, Vg = sla.eigh(0.5 * (G_A + G_A.T))
    es_pos = np.maximum(es, 0.0)
    eg_pos = np.maximum(eg, 0.0)
    # S_AA^{1/2} G_A^{1/2} has identical singular values to diag(sqrt(es)) @ (Vs^T Vg) @ diag(sqrt(eg))!
    M = (np.sqrt(es_pos)[:, None] * (Vs.T @ Vg)) * np.sqrt(eg_pos)[None, :]
    sv = sla.svdvals(M)
    e2 = sv ** 2
    tot = float(np.sum(e2)) + 1e-12
    fracs = {f"k_{int(k)}": round(float(np.sum(e2[: min(int(k), len(e2))]) / tot), 6) for k in ks}
    return {
        "retained_fisher_by_k": fracs,
        "retained_fisher_1536": fracs.get("k_1536", round(float(np.sum(e2[: min(1536, len(e2))]) / tot), 6)),
        "dropped_fisher_tail_1536": round(1.0 - float(np.sum(e2[: min(1536, len(e2))]) / tot), 6),
        "top_sv": round(float(sv[0]), 4) if len(sv) > 0 else 0.0,
    }


def compute_gauss_newton_logit_gap_target(
    G_B: np.ndarray,
    U_B_shared: np.ndarray,
    p_A: np.ndarray,
    p_B: np.ndarray,
    lam_rel_med: float = 1e-1,
) -> np.ndarray:
    """
    Fable+Astra Alignment-Free Gauss-Newton Logit-Gap Writer Target at L27:
      `tau_GN = -(G_B + lam*I)^{-1} U_B^T (p_B - p_A) = (G_B + lam*I)^{-1} U_B^T (p_A - p_B)`
    Maps Donor 7B's probability residual directly into Recipient 1.5B's post-final-LN space!
    """
    grad_b = (p_A - p_B) @ U_B_shared  # (N, D_B)
    eb = sla.eigvalsh(G_B)
    pos = eb[eb > 1e-12]
    med_b = float(np.median(pos)) if len(pos) > 0 else float(np.trace(G_B) / max(G_B.shape[0], 1))
    lam_abs = max(float(lam_rel_med) * med_b, 1e-8)
    tau = sla.solve(G_B + lam_abs * np.eye(G_B.shape[0], dtype=np.float64), grad_b.T, assume_a="pos").T
    return tau


# ======================================================================================
# 4. Floor #3: Exact Nonlinear Residual Gain `Delta_NL` (Tuned Linear Ridge vs + Nyström Kernel)
# ======================================================================================

def compute_floor3_nonlinear_residual_gain(
    X_fit: np.ndarray,
    Y_fit: np.ndarray,
    X_tune: np.ndarray,
    Y_tune: np.ndarray,
    X_report: np.ndarray,
    Y_report: np.ndarray,
    W_lin: np.ndarray,
    n_landmarks: int = 512,
    seed: int = 20261128,
) -> Dict[str, Any]:
    """
    Floor #3 — Measure exact held-out nonlinear residual gain `Delta_NL`:
    Compares tuned linear Ridge `Y_hat_lin = X @ W_lin.T` against `Y_hat_lin + K_rbf(X, L) @ C`
    trained strictly on the linear residual `(Y_fit - X_fit @ W_lin.T)` with `(gamma, lam)` tuned
    strictly on `cal_tune` and reported strictly on untouched `cal_report`!
    """
    Xf, Yf = bf16_to_f64_np(X_fit), bf16_to_f64_np(Y_fit)
    Xt, Yt = bf16_to_f64_np(X_tune), bf16_to_f64_np(Y_tune)
    Xr, Yr = bf16_to_f64_np(X_report), bf16_to_f64_np(Y_report)

    def _metrics(pred: np.ndarray, tgt: np.ndarray) -> Dict[str, float]:
        r2 = 1.0 - float(np.sum((tgt - pred) ** 2) / (np.sum((tgt - tgt.mean(axis=0)) ** 2) + 1e-12))
        cos = float(np.mean(np.sum(pred * tgt, axis=1) / (np.linalg.norm(pred, axis=1) * np.linalg.norm(tgt, axis=1) + 1e-12)))
        return {"r2": round(r2, 4), "cos": round(cos, 4)}

    pred_lin_t = Xt @ W_lin.T
    pred_lin_r = Xr @ W_lin.T
    res_f = Yf - Xf @ W_lin.T
    lin_rep = _metrics(pred_lin_r, Yr)

    # Subsample landmarks deterministically from normalized X_fit
    rng = np.random.default_rng(seed)
    Xf_u = Xf / (np.linalg.norm(Xf, axis=1, keepdims=True) + 1e-12)
    Xt_u = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-12)
    Xr_u = Xr / (np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-12)
    k_lm = min(int(n_landmarks), len(Xf_u))
    lm_idx = rng.choice(len(Xf_u), size=k_lm, replace=False)
    L = Xf_u[lm_idx]  # (K, D_in)

    def _rbf_cosine(Xu: np.ndarray, gamma: float) -> np.ndarray:
        # 2 - 2 cos(x, l) = ||x_u - l_u||^2
        d2 = np.maximum(2.0 - 2.0 * (Xu @ L.T), 0.0)
        return np.exp(-float(gamma) * d2)

    best_tune_r2 = -1e9
    best_cfg = (1.0, 1e-1)
    best_C: Optional[np.ndarray] = None
    for gamma in (0.5, 1.5, 4.0):
        Kf = _rbf_cosine(Xf_u, gamma)
        Kt = _rbf_cosine(Xt_u, gamma)
        # Exact Nyström Kernel Ridge Gram regularizer: (Kf^T Kf / N + lam * K_LL) C = Kf^T res_f / N!
        K_LL = 0.5 * (_rbf_cosine(L, gamma) + _rbf_cosine(L, gamma).T)
        G_k = (Kf.T @ Kf) / float(len(Kf))
        rhs_k = (Kf.T @ res_f) / float(len(Kf))
        tr_k = float(np.trace(G_k)) / max(k_lm, 1)
        tr_ll = float(np.trace(K_LL)) / max(k_lm, 1)
        reg_scale = tr_k / max(tr_ll, 1e-12)
        for lam_rel in (1e-2, 1e-1, 1.0):
            A_nys = G_k + (lam_rel * reg_scale) * K_LL + (1e-6 * tr_k) * np.eye(k_lm)
            C = sla.solve(A_nys, rhs_k, assume_a="pos")  # (K, D_out)
            sc = _metrics(pred_lin_t + Kt @ C, Yt)
            if sc["r2"] > best_tune_r2:
                best_tune_r2 = sc["r2"]
                best_cfg = (gamma, lam_rel)
                best_C = C

    assert best_C is not None
    Kr = _rbf_cosine(Xr_u, best_cfg[0])
    nl_rep = _metrics(pred_lin_r + Kr @ best_C, Yr)
    return {
        "linear_cal_report": lin_rep,
        "linear_plus_kernel_cal_report": nl_rep,
        "delta_nl_r2": round(nl_rep["r2"] - lin_rep["r2"], 4),
        "delta_nl_cos": round(nl_rep["cos"] - lin_rep["cos"], 4),
        "best_gamma": best_cfg[0],
        "best_lam_rel": best_cfg[1],
        "n_landmarks": k_lm,
    }


# ======================================================================================
# 5. Calibration Corpus Scaling (`N_fit >= 60,000`, `N_val >= 6,000`, 0 Shared Prefixes!)
# ======================================================================================

def build_60k_calibration_corpus(
    n_fit_passages: int = 620,
    n_val_passages: int = 80,
    seed: int = 20261128,
) -> Dict[str, List[str]]:
    """
    Build strictly TEMPLATE-DISJOINT and DOMAIN-DISJOINT `fit` (`>= 60,000` Qwen2.5 tokens) vs
    `val` (`>= 6,000` Qwen2.5 tokens) calibration passages with ZERO shared causal prefixes!
    Strictly disjoint from all 8 SQL schemas, OurAirports, 2-hop pools, and 380 locality facts.
    """
    rng = np.random.default_rng(seed)

    fit_domains = [
        ("geophysics", "seismic wave propagation", "crustal density gradients", "mantle convection", "tectonic boundary stress"),
        ("oceanography", "thermohaline circulation", "abyssal salinity profiles", "upwelling nutrient plumes", "baroclinic instability"),
        ("botany", "stomatal conductance", "phloem sucrose transport", "xylem cavitation thresholds", "chloroplast antenna complexes"),
        ("acoustics", "harmonic overtone decay", "reverberation absorption panels", "helmholtz cavity resonance", "waveguide impedance matching"),
        ("metallurgy", "austenite phase transformation", "grain boundary precipitation", "cold-rolled work hardening", "creep rupture resistance"),
        ("optics", "diffraction grating dispersion", "birefringent polarization rotation", "fabry-perot cavity finesse", "chromatic aberration correction"),
        ("compiler design", "static single assignment graphs", "register interference coloring", "loop-invariant code motion", "dead-store elimination passes"),
        ("distributed systems", "raft leader election timers", "vector clock causal ordering", "quorum write-ahead logs", "consistent hash ring rebalancing"),
        ("relational algebra", "projection pushdown rewrites", "hash-join probe partitioning", "b-tree index leaf splits", "cost-based cardinality estimation"),
        ("signal processing", "windowed fourier spectrograms", "kalman innovation residuals", "butterworth filter phase lag", "adaptive least-mean-squares cancellation"),
        ("thermodynamics", "isentropic compressor efficiency", "carnot cycle heat rejection", "enthalpy-entropy mollier charts", "regenerative rankine preheating"),
        ("hydrology", "watershed runoff hydrographs", "unsaturated vadose infiltration", "darcy aquifer transmissivity", "evapotranspiration energy balance"),
        ("linguistics", "morphophonemic assimilation rules", "ergative case alignment", "prosodic stress hierarchies", "syntactic island constraints"),
        ("econometrics", "instrumental variable identification", "heteroskedasticity-robust standard errors", "cointegrated error-correction panels", "difference-in-differences parallel trends"),
        ("cryptography", "elliptic curve scalar multiplication", "merkle tree inclusion proofs", "sponge permutation absorption", "constant-time modular reduction"),
        ("robotics", "inverse kinematics jacobian damping", "model predictive trajectory horizons", "extended kalman pose fusion", "compliant impedance torque control"),
        ("photogrammetry", "epipolar bundle adjustment", "lens distortion polynomials", "stereo disparity rectification", "dense point-cloud triangulation"),
        ("electrochemistry", "nernst equilibrium potentials", "solid-electrolyte interphase growth", "butler-volmer exchange currents", "cyclic voltammetry hysteresis"),
        ("paleoclimatology", "oxygen isotope fractionation", "speleothem carbonate laminae", "ice-core methane chronologies", "orbital precession insolation"),
        ("structural mechanics", "euler-bernoulli beam deflection", "finite-element stiffness assembly", "buckling eigenvalue modes", "fatigue crack propagation"),
    ]
    val_domains = [
        ("archival conservation", "alkaline deacidification buffers", "iron-gall ink stabilization", "microclimate humidity buffering", "multispectral pigment imaging"),
        ("cartography", "lambert conformal conic parallels", "geodesic datum transformations", "contour interval generalization", "hillshade azimuth illumination"),
        ("ceramics", "kaolinite vitrification shrinkage", "reduction kiln glaze chemistry", "thermal shock spalling resistance", "slip-casting plaster absorption"),
        ("horticulture", "vernalization chilling accumulation", "mycorrhizal phosphorus uptake", "graft union cambium alignment", "photoperiod flowering induction"),
        ("glaciology", "basal meltwater lubrication", "firn densification profiles", "moraine sediment entrainment", "calving front flexure"),
        ("horology", "escapement impulse geometry", "isochronous balance spring coils", "jewel pivot friction reduction", "temperature-compensated bimetallic rims"),
        ("mycology", "hyphal tip extension", "spore dispersal ballistics", "lignin peroxidase secretion", "mycelial network anastomosis"),
        ("glassblowing", "annealing strain relief", "borosilicate thermal expansion", "marvering viscosity windows", "dichroic vapor deposition"),
    ]

    fit_methods = [
        "empirical calibration across controlled laboratory intervals",
        "iterative numerical integration using adaptive step control",
        "dimensional analysis linking dimensionless scaling groups",
        "comparative spectral decomposition of noisy sensor series",
        "systematic ablation across boundary condition perturbations",
        "hierarchical bayesian smoothing of sparse field observations",
        "closed-form variational approximation under linear constraints",
        "discrete-event simulation with deterministic priority queues",
        "orthogonal collocation across piecewise polynomial intervals",
        "perturbation expansion around steady operating equilibria",
    ]
    val_methods = [
        "multi-site benchmark auditing across seasonal shifts",
        "nonparametric bootstrap resampling of replicated trials",
        "blind inter-laboratory cross-checking against reference standards",
        "longitudinal drift monitoring under step-wise load ramps",
    ]

    fit_notes = [
        "When boundary assumptions hold, second-order drift remains bounded and predictable across repeated cycles.",
        "Practitioners verify stability by comparing held-out variance against theoretical noise floors before deployment.",
        "Minor measurement offsets compound nonlinearly unless each intermediate step is re-centered against invariants.",
        "Balanced regularization trades tiny bias for substantial gains in out-of-sample numerical conditioning.",
        "Modular pipelines isolate localized faults so individual components can be inspected or reverted cleanly.",
        "Careful bookkeeping of intermediate residuals exposes hidden coupling terms long before downstream saturation occurs.",
    ]
    val_notes = [
        "Field reviewers rely on transparent error budgets before certifying revised procedures for routine operations.",
        "Archiving raw diagnostic traces allows later teams to reproduce unexpected anomalies without guesswork.",
        "Consistent unit conventions eliminate silent scale discrepancies between upstream probes and downstream summaries.",
        "Periodic cross-validation on independent specimens separates durable mechanisms from transient local artifacts.",
    ]

    fit_templates = [
        (
            "Case {idx} ({d0}): analysts track {d1} alongside {d2} whenever {d3} modulates {d4}. "
            "Applying {m0} clarifies how coupled variables respond under sustained load. {c0} "
            "In secondary tests linking {e0}, {e1} and {e2} remain strongly correlated with {e3} and {e4}."
        ),
        (
            "def analyze_{idx}(samples, reg=1e-6):\n"
            "    mean_sq = sum(x * x for x in samples) / max(len(samples), 1)\n"
            "    return [x / ((mean_sq + reg) ** 0.5) for x in samples]\n"
            "# Topic {d0}: relates {d1} to {d2} via {d3} and {d4} using {m0}. {c0} "
            "Compare against {e0} where {e1} drives {e2} through {e3}."
        ),
        (
            "Memo #{idx} on {d0}:\n"
            "- Primary observable: {d1}\n"
            "- Secondary factor: {d2}\n"
            "- Coupling path: {d3} drives {d4}\n"
            "- Protocol: {m0}\n"
            "- Cross-check ({e0}): {e1} balances {e2} while tracking {e3} and {e4}.\n"
            "Summary: {c0}"
        ),
        (
            "Question {idx}: How does {d1} interact with {d2} inside {d0}?\n"
            "Answer: Perturbations in {d3} propagate directly into {d4}, which teams quantify via {m0}. "
            "An analogous mechanism appears in {e0} between {e1}, {e2}, and {e3}. {c0}"
        ),
    ]
    val_tune_templates = [
        (
            "Field Log [{idx}] | Subject: {d0} — Researchers studying {d1} observed that shifts in {d3} "
            "consistently altered both {d2} and {d4}. By scheduling {m0}, the audit team bounded residual scatter. "
            "Parallel observations from {e0} ({e1}, {e2}, {e3}) confirmed the same behavior. {c0}"
        ),
        (
            "Briefing ({idx}) concerning {d0}: Rather than isolating {d1} alone, recent workflows pair {d2} with {d4} "
            "while tracking {d3} continuously through {m0}. Teams also documented {e1} and {e2} under {e0} ({e3} -> {e4}). {c0}"
        ),
    ]
    # Strictly distinct templates & openers for `val_report` so `cal_tune` vs `cal_report` ALSO have 0 shared prefixes!
    val_report_templates = [
        (
            "/* Audit record {idx}: {d0} */\n"
            "SELECT metric_label, AVG(reading) FROM scratch_log WHERE regime = '{d1}' GROUP BY metric_label;\n"
            "Notes: {d2} co-varies with {d3} and {d4}; cross-domain check in {e0} ({e1} vs {e2}); verified via {m0}. {c0}"
        ),
        (
            "REVIEW ENTRY <{idx}> — Domain {d0}: When {d3} varies across regimes, {d1} and {d2} drift together toward {d4}. "
            "Using {m0}, inspectors cross-referenced {e0} ({e1}, {e2}, {e4}) before final sign-off. {c0}"
        ),
    ]

    def _gen(n_target: int, doms, meths, notes, tpls, start_idx: int) -> List[str]:
        out: List[str] = []
        seen = set()
        k = 0
        while len(out) < n_target:
            k += 1
            di = int(rng.integers(0, len(doms)))
            dj = (di + 1 + int(rng.integers(0, len(doms) - 1))) % len(doms)
            d, e = doms[di], doms[dj]
            m0 = meths[int(rng.integers(0, len(meths)))]
            c0 = notes[int(rng.integers(0, len(notes)))]
            tpl = tpls[k % len(tpls)]
            txt = tpl.format(
                idx=start_idx + len(out),
                d0=d[0], d1=d[1], d2=d[2], d3=d[3], d4=d[4],
                e0=e[0], e1=e[1], e2=e[2], e3=e[3], e4=e[4],
                m0=m0, c0=c0,
            )
            if txt not in seen:
                seen.add(txt)
                out.append(txt)
        return out

    n_vt = n_val_passages // 2
    n_vr = n_val_passages - n_vt
    val_tune_p = _gen(n_vt, val_domains[:4], val_methods[:2], val_notes[:2], val_tune_templates, start_idx=7000)
    val_report_p = _gen(n_vr, val_domains[4:], val_methods[2:], val_notes[2:], val_report_templates, start_idx=9000)
    return {
        "fit": _gen(n_fit_passages, fit_domains, fit_methods, fit_notes, fit_templates, start_idx=1000),
        "val_tune": val_tune_p,
        "val_report": val_report_p,
        "val": val_tune_p + val_report_p,
    }


# ======================================================================================
# 6. Shared-Suffix Multi-Token Extraction (Task C 2-Hop Table Alignment!)
# ======================================================================================

@torch.no_grad()
def extract_shared_suffix_states(
    model: nn.Module,
    tok: Any,
    full_prompts: Sequence[str],
    suffix_prompts: Sequence[str],
    layer_indices: Sequence[int],
    device: torch.device,
    batch_size: int = 8,
    max_length: int = 256,
    capture_fields: Sequence[str] = ("y_attn", "h_out"),
    z_attn_tail_slice: Optional[int] = None,
    z_mlp_tail_slice: Optional[int] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Extract ALL tokens corresponding to `suffix_prompts[i]` at the tail of `full_prompts[i]`!
    Verifies exact token-ID suffix alignment for EVERY item (zero silent misalignment!) so
    2-shot Donor 7B (`demo + table`) and 0-shot Recipient 1.5B (`table`) align across EVERY
    table token, not just the last token!
    """
    assert len(full_prompts) == len(suffix_prompts)
    layers = get_decoder_layers(model)
    fields = set(capture_fields)
    if not full_prompts:
        return {int(li): {f: np.empty((0, 0), dtype=np.uint16) for f in fields} for li in layer_indices}

    # Pre-verify STRICT exact token-ID suffix identity (`f_ids[-len(s_ids):] == s_ids`, zero silent mismatch!)
    suf_lens: List[int] = []
    for idx_p, (fp, sp) in enumerate(zip(full_prompts, suffix_prompts)):
        assert fp.endswith(sp), f"Item {idx_p}: full_prompt must end with suffix_prompt!"
        f_ids = tok.encode(fp, truncation=True, max_length=max_length, add_special_tokens=True)
        s_ids = tok.encode(sp, truncation=True, max_length=max_length, add_special_tokens=True)
        assert len(f_ids) >= len(s_ids) and f_ids[-len(s_ids) :] == s_ids, (
            f"Item {idx_p}: exact token-ID suffix alignment failed (len_f={len(f_ids)}, len_s={len(s_ids)})!"
        )
        suf_lens.append(len(s_ids))

    buf: Dict[int, Dict[str, torch.Tensor]] = {int(li): {} for li in layer_indices}
    store: Dict[int, Dict[str, List[np.ndarray]]] = {int(li): {f: [] for f in fields} for li in layer_indices}
    handles: List[Any] = []
    old_pad = tok.padding_side
    tok.padding_side = "left"
    try:
        for li in layer_indices:
            li = int(li)
            lyr = layers[li]
            if "z_attn" in fields:
                def _mk_za(idx: int, sl: Optional[int]):
                    def _pre(_m, args):
                        t = args[0].detach()
                        buf[idx]["z_attn"] = t[..., -sl:] if sl is not None else t
                    return _pre
                handles.append(lyr.self_attn.o_proj.register_forward_pre_hook(_mk_za(li, z_attn_tail_slice)))
            if "y_attn" in fields:
                def _mk_ya(idx: int):
                    def _fn(_m, _i, out):
                        buf[idx]["y_attn"] = out.detach()
                    return _fn
                handles.append(lyr.self_attn.o_proj.register_forward_hook(_mk_ya(li)))
            if "z_mlp" in fields:
                def _mk_zm(idx: int, sl: Optional[int]):
                    def _pre(_m, args):
                        t = args[0].detach()
                        buf[idx]["z_mlp"] = t[..., -sl:] if sl is not None else t
                    return _pre
                handles.append(lyr.mlp.down_proj.register_forward_pre_hook(_mk_zm(li, z_mlp_tail_slice)))
            if "h_out" in fields:
                def _mk_ho(idx: int):
                    def _fn(_m, _i, out):
                        t = out[0] if isinstance(out, tuple) else out
                        buf[idx]["h_out"] = t.detach()
                    return _fn
                handles.append(lyr.register_forward_hook(_mk_ho(li)))

        backbone = model.model if hasattr(model, "model") else model
        for i0 in range(0, len(full_prompts), batch_size):
            bp = list(full_prompts[i0 : i0 + batch_size])
            bs_lens = suf_lens[i0 : i0 + batch_size]
            enc = tok(bp, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=True)
            ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            _ = backbone(input_ids=ids, attention_mask=mask, use_cache=False)
            for li in layer_indices:
                li = int(li)
                for f in fields:
                    t = buf[li][f]
                    for bi, slen in enumerate(bs_lens):
                        arr = t[bi, -slen:, :].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy()
                        store[li][f].append(arr)
                buf[li].clear()
    finally:
        for h in handles:
            h.remove()
        tok.padding_side = old_pad
    assert_zero_runtime_hooks(model)
    return {int(li): {f: np.concatenate(store[int(li)][f], axis=0) for f in fields} for li in layer_indices}


# ======================================================================================
# 7. Pure-CPU Unit Tests (<3s, ZERO GPU!)
# ======================================================================================

def run_unit_tests() -> None:
    t0 = time.time()
    print("=== Running exact_math_v43.py Pure-CPU Unit Tests ===")

    rng = np.random.default_rng(2026)

    # 1. Verify StreamingSecondMomentAccumulator matches exact batch normal equations (<1e-12)
    N, D_A, D_B = 400, 32, 16
    Xa = rng.standard_normal((N, D_A))
    W_true = rng.standard_normal((D_B, D_A)) / math.sqrt(D_A)
    Xb = Xa @ W_true.T + 0.05 * rng.standard_normal((N, D_B))
    acc = StreamingSecondMomentAccumulator(D_A, D_B)
    for i0 in range(0, N, 64):
        acc.update(torch.from_numpy(Xa[i0 : i0 + 64]), torch.from_numpy(Xb[i0 : i0 + 64]))
    m = acc.moments()
    assert np.allclose(m["S_AA"], (Xa.T @ Xa) / N, atol=1e-12)
    assert np.allclose(m["S_BA"], (Xb.T @ Xa) / N, atol=1e-12)
    W_hat, info = fit_ridge_from_moments_median_floor(m["S_AA"], m["S_BA"], Xa[:80], Xb[:80], n_fit=N)
    assert info["val_cos"] > 0.99 and info["median_eig"] > 0.0
    print(f"  [PASS 1/5] StreamingSecondMomentAccumulator + median-eig ridge: val_cos={info['val_cos']}, med_eig={info['median_eig']}")

    # 2. Verify Exact Per-Reader Observability `o_j`: observable row -> ~1.0, orthogonal blind-spot row -> ~0.0!
    # Construct Xa where dims 0..15 are identical to Xb, and dims 16..31 are independent noise
    Xa2 = rng.standard_normal((2000, 32))
    Xb2 = Xa2[:, :16] + 0.01 * rng.standard_normal((2000, 16))
    S_AA2 = (Xa2.T @ Xa2) / 2000.0
    S_BB2 = (Xb2.T @ Xb2) / 2000.0
    S_BA2 = (Xb2.T @ Xa2) / 2000.0
    _, S_AgB2 = compute_schur_complement_s_agb(S_AA2, S_BB2, S_BA2, lam_abs=1e-5)
    R_test = np.zeros((3, 32))
    R_test[0, 0] = 1.0   # 100% observable in Xb!
    R_test[1, 20] = 1.0  # 100% in 16-D blind spot!
    # Row 2 is all-zeros (zero-variance reader -> must be strictly rejected as 0.0, NEVER 1.0!)
    o_vals = compute_reader_observability(R_test, S_AA2, S_AgB2)
    assert o_vals[0] > 0.99 and o_vals[1] < 0.05 and o_vals[2] == 0.0, f"Bad observability scores: {o_vals}"
    # Also verify tiny singular covariance ridge floor never rounds to 0.0 before Schur solve!
    S_sing = np.diag([1e-6, 0.0])
    _, info_sing = fit_ridge_from_moments_median_floor(S_sing, S_sing, np.ones((4, 2)) * 1e-3, np.ones((4, 2)) * 1e-3, n_fit=4)
    assert info_sing["best_lam_abs"] > 0.0
    _, S_agb_sing = compute_schur_complement_s_agb(S_sing, S_sing, S_sing, info_sing["best_lam_abs"])
    assert np.all(np.isfinite(S_agb_sing))
    print(f"  [PASS 2/5] Exact Schur observability o_j: observable={o_vals[0]:.4f}, blind_spot={o_vals[1]:.4f}, zero_var={o_vals[2]:.4f}, singular_lam={info_sing['best_lam_abs']:.2e}")

    # 3. Verify StreamingSharedVocabFisherAccumulator + M_U + Floor #1 Fisher-RRR + Gauss-Newton target!
    V_s, DA_s, DB_s = 256, 24, 12
    # Construct U_A = [U_B, small_tail] so top DB_s Fisher modes hold ~99%+ of Fisher info!
    U_B_s = torch.from_numpy(rng.standard_normal((V_s, DB_s)).astype(np.float32))
    U_A_s = torch.cat([U_B_s, 0.02 * torch.from_numpy(rng.standard_normal((V_s, DA_s - DB_s)).astype(np.float32))], dim=1)
    h_s = torch.from_numpy(rng.standard_normal((128, DA_s)).astype(np.float32))
    lg_s = h_s @ U_A_s.T
    facc = StreamingSharedVocabFisherAccumulator(U_A_s, U_B_s)
    facc.update(lg_s[:64])
    facc.update(lg_s[64:])
    fm = facc.finalize()
    M_U, mu_info = solve_shared_vocab_bridge_m_u(fm["G_B"], fm["G_BA"], eps_rel_med=1e-3)
    # Since U_A[:, :DB_s] == U_B, M_U[:, :DB_s] should be ~I_{DB_s}!
    err_id = float(np.linalg.norm(M_U[:, :DB_s] - np.eye(DB_s)) / math.sqrt(DB_s))
    assert err_id < 0.05, f"M_U failed identity recovery: err={err_id}"
    S_AA_s = (h_s.double().numpy().T @ h_s.double().numpy()) / 128.0
    frrr = compute_fisher_rrr_retained_fraction(S_AA_s, fm["G_A"], ks=(6, 12))
    assert frrr["retained_fisher_by_k"]["k_12"] > 0.99, f"Fisher RRR low: {frrr}"
    pA_np = F.softmax(lg_s[:8], dim=-1).double().numpy()
    pB_np = F.softmax(lg_s[:8] * 0.5, dim=-1).double().numpy()
    tau_gn = compute_gauss_newton_logit_gap_target(fm["G_B"], U_B_s.double().numpy(), pA_np, pB_np)
    assert tau_gn.shape == (8, DB_s) and np.all(np.isfinite(tau_gn))
    print(f"  [PASS 3/5] Shared-vocab M_U (id_err={err_id:.4f}), Floor #1 Fisher-RRR (k_12={frrr['retained_fisher_by_k']['k_12']}), Gauss-Newton target OK!")

    # 4. Verify Floor #3 Nonlinear Residual Gain detects genuine quadratic nonlinearity cleanly!
    X_nl = rng.standard_normal((600, 16))
    Y_nl = X_nl[:, :8] + 0.6 * (X_nl[:, :8] ** 2)
    W_l = np.hstack([np.eye(8), np.zeros((8, 8))])
    f3 = compute_floor3_nonlinear_residual_gain(X_nl[:400], Y_nl[:400], X_nl[400:500], Y_nl[400:500], X_nl[500:], Y_nl[500:], W_l, n_landmarks=128)
    assert f3["delta_nl_r2"] > 0.10, f"Floor #3 missed quadratic nonlinearity: {f3}"
    print(f"  [PASS 4/5] Floor #3 Nonlinear Gain detected Delta_NL R2 = +{f3['delta_nl_r2']:.4f}")

    # 5. Verify 60k calibration corpus yields >= 60,000 fit tokens, >= 6,000 val tokens, 0 shared prefixes!
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/tmp/platos_key_lab/hf_cache/qwen2.5-3b", local_files_only=True)
    cal = build_60k_calibration_corpus(n_fit_passages=620, n_val_passages=80)
    f_ids = [tok.encode(p, truncation=True, max_length=160) for p in cal["fit"]]
    vt_ids = [tok.encode(p, truncation=True, max_length=160) for p in cal["val_tune"]]
    vr_ids = [tok.encode(p, truncation=True, max_length=160) for p in cal["val_report"]]
    nf, nvt, nvr = sum(len(x) for x in f_ids), sum(len(x) for x in vt_ids), sum(len(x) for x in vr_ids)
    f_pref = {tuple(x[:i]) for x in f_ids for i in range(1, len(x) + 1)}
    vt_pref = {tuple(x[:i]) for x in vt_ids for i in range(1, len(x) + 1)}
    sh_fv = sum(1 for x in vt_ids + vr_ids for i in range(1, len(x) + 1) if tuple(x[:i]) in f_pref)
    sh_tr = sum(1 for x in vr_ids for i in range(1, len(x) + 1) if tuple(x[:i]) in vt_pref)
    assert nf >= 60_000 and nvt >= 3_000 and nvr >= 3_000 and sh_fv == 0 and sh_tr == 0, (
        f"60k corpus check failed: nf={nf}, nvt={nvt}, nvr={nvr}, sh_fv={sh_fv}, sh_tr={sh_tr}"
    )
    print(f"  [PASS 5/5] 60k calibration corpus verified: N_fit={nf}, N_tune={nvt}, N_report={nvr}, 0/0 shared prefixes (fit-val AND tune-report)!")
    print(f"=== ALL 5 EXACT_MATH_V43 UNIT TESTS PASSED ({time.time() - t0:.2f}s)! ===")


if __name__ == "__main__":
    if "--unit-test" in sys.argv:
        run_unit_tests()
