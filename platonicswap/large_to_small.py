#!/usr/bin/env python3
"""
large_to_small.py — BF16 Sequential Loader, Reversible Delta Packs, Contrast Bridges & Closed-Form ΔW
======================================================================================================

Shared v3 Large -> Small core module (`Qwen2.5-7B-Instruct` -> `SmolLM2-1.7B-Instruct` /
`Qwen2.5-1.5B-Instruct`) supporting all 3 head-to-head tiers on a single 24 GB NVIDIA L4 GPU:
  - Sequential BF16 model loading (`14.2 GiB` for 7B, `3.4 GiB` for 1.7B) with strict FP64/FP32
    matrix algebra and guaranteed VRAM release (`model.to('cpu') + gc + empty_cache`).
  - `ReversibleDeltaPack`: unmerged delta storage (`W_eff = W_base + ΔW`) supporting reversible edits
    to base weights/heads AND/OR appended organs, with bit-exact rollback verification
    (`torch.equal` on both `state_dict` and matched-batch fresh-cache logits).
  - Contrast-fitted linear bridges: `M = ΔB W ΔA^T (ΔA W ΔA^T + λI)^{-1}` (FP64).
  - Exact RMSNorm-aligned cocycle projection (`G_B J_1 U_norm = G_A U_norm`).
  - Closed-form layer-wise `ΔW` ridge solver (zero backprop) for matching translated donor layer
    outputs `ΔH_B = M ΔH_A` with sequential depth-cocycle refitting.
  - Seat 3's 2-Level Hierarchical Routed Fact Bank (`P[64, D_B] -> C[G, 64] -> E[G, K, 64] -> V[G, K, D_B]`)
    with joint bank+sequence max threshold calibration & hard bypass (`torch.equal == True`).

Usage:
  PYTHONPATH=platonicswap:experiments python platonicswap/large_to_small.py --unit-test
"""

from __future__ import annotations

import gc
import math
import os
import sys
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

from extract_and_baseline import HF_CACHE_DIR, get_decoder_layers

LARGE_TO_SMALL_MODELS: Dict[str, str] = {
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "smollm2-1.7b-instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B",
}


# ======================================================================================
# 1. Sequential BF16 Loader (Fits Qwen2.5-7B-Instruct 14.2 GiB on 24 GB L4!)
# ======================================================================================

def get_preferred_device_and_bf16() -> Tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def load_model_bf16(
    model_key: str,
    dtype_override: Optional[torch.dtype] = None,
    allow_download: bool = False,
) -> Tuple[Any, Any, torch.device, torch.dtype]:
    """Load a causal LM sequentially in BF16 (CUDA/MPS) or FP32 (CPU), offline by default."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = get_preferred_device_and_bf16()
    if dtype_override is not None:
        dtype = dtype_override

    local_dir = HF_CACHE_DIR / model_key
    if local_dir.is_dir():
        src = str(local_dir)
        local_only = True
    elif allow_download and model_key in LARGE_TO_SMALL_MODELS:
        src = LARGE_TO_SMALL_MODELS[model_key]
        local_only = False
    else:
        raise FileNotFoundError(f"Model directory not found: {local_dir} (set PLATONICSWAP_HF_CACHE or download first)")

    tok = AutoTokenizer.from_pretrained(src, local_files_only=local_only)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        src,
        local_files_only=local_only,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, tok, device, dtype


def unload_model_strict(model: Any) -> None:
    """Guaranteed VRAM release on CUDA/MPS even if caller holds transient references briefly."""
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ======================================================================================
# 2. Reversible Delta Pack (Unmerged Base-Weight / Head ΔW with Bit-Exact Rollback!)
# ======================================================================================

class ReversibleDeltaPack:
    """
    Stores reversible parameter deltas `{param_name: delta_tensor}` alongside exact original
    bit-copies of touched parameters (`orig_snapshots`), so `apply(model)` modifies `W_eff = W_base + ΔW`
    and `revert(model)` restores `W_base` bit-for-bit (`torch.equal == True` on both weights
    and matched-batch fresh-cache logits!).
    """

    def __init__(self, deltas: Dict[str, torch.Tensor], metadata: Optional[Dict[str, Any]] = None):
        self.deltas: Dict[str, torch.Tensor] = {k: v.detach().cpu().clone() for k, v in deltas.items()}
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self._orig_snapshots: Dict[str, torch.Tensor] = {}
        self._applied: bool = False

    def apply(self, model: nn.Module) -> None:
        assert not self._applied, "ReversibleDeltaPack is already applied!"
        named = dict(model.named_parameters())
        with torch.no_grad():
            for pname, delta in self.deltas.items():
                if pname not in named:
                    raise KeyError(f"Parameter {pname!r} not found in model")
                p = named[pname]
                # Store exact bit-identical snapshot on CPU so BF16 rounding cannot break rollback!
                self._orig_snapshots[pname] = p.detach().cpu().clone()
                p.copy_((p.float() + delta.to(p.device, dtype=torch.float32)).to(p.dtype))
        self._applied = True

    def revert(self, model: nn.Module) -> None:
        if not self._applied:
            return
        named = dict(model.named_parameters())
        with torch.no_grad():
            for pname, orig in self._orig_snapshots.items():
                named[pname].copy_(orig.to(named[pname].device, dtype=named[pname].dtype))
        self._orig_snapshots.clear()
        self._applied = False

    def verify_bit_exact_rollback(
        self,
        model: nn.Module,
        probe_input_ids: torch.Tensor,
        probe_attention_mask: Optional[torch.Tensor],
        base_logits: torch.Tensor,
    ) -> Dict[str, Any]:
        """Verify matched-batch fresh-cache (`use_cache=False`) bit-identical restoration."""
        assert not self._applied, "Call revert(model) before verify_bit_exact_rollback!"
        with torch.no_grad():
            kwargs = {"input_ids": probe_input_ids, "use_cache": False}
            if probe_attention_mask is not None:
                kwargs["attention_mask"] = probe_attention_mask
            restored_logits = model(**kwargs).logits.detach().cpu()
        bit_eq = bool(torch.equal(restored_logits, base_logits.cpu()))
        max_diff = float((restored_logits.float() - base_logits.cpu().float()).abs().max().item())
        return {"logits_bit_equal": bit_eq, "max_abs_diff": max_diff}

    def n_bytes_bf16(self) -> int:
        return sum(int(t.numel()) * 2 for t in self.deltas.values())


# ======================================================================================
# 3. Contrast-Fitted Bridges, RMSNorm Coordinate Alignment & Closed-Form Layer-Wise ΔW
# ======================================================================================

def fit_contrast_bridge(
    dA: np.ndarray,
    dB: np.ndarray,
    weights: Optional[np.ndarray] = None,
    lam: Optional[float] = None,
    mode: str = "ridge",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Fit contrast-based cross-family bridge in strict FP64:
      M = ΔB^T W ΔA (ΔA^T W ΔA + λ I)^{-1}   (shape D_B x D_A, maps column vectors v_A -> M @ v_A)
    where rows of `dA` (N, D_A) and `dB` (N, D_B) are paired same-template entity/concept contrasts
    (e.g., `h(entity_i) - h(mean_entity)` or `h(prompt_i^+) - h(prompt_i^-)`).
    Supports:
      - `mode="ridge"` (exact closed-form regularized contrast ridge, automatic GCV/LW λ if None)
      - `mode="stiefel"` (scaled orthogonal Procrustes `s * R` preserving causal tail energy)
    """
    A = np.asarray(dA, dtype=np.float64)
    B = np.asarray(dB, dtype=np.float64)
    N, D_A = A.shape
    D_B = B.shape[1]
    assert B.shape[0] == N

    if weights is None:
        w = np.ones(N, dtype=np.float64) / max(N, 1)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / (np.sum(w) + 1e-12)

    Aw = A * np.sqrt(w)[:, None]
    Bw = B * np.sqrt(w)[:, None]

    if mode == "stiefel":
        C = Bw.T @ Aw  # (D_B, D_A)
        U, sv, Vt = sla.svd(C, full_matrices=False)
        R = U @ Vt  # (D_B, D_A)
        proj_A = Aw @ R.T
        s = float(np.sqrt(np.sum(Bw ** 2) / (np.sum(proj_A ** 2) + 1e-12)))
        M = s * R
        used_lam = 0.0
    else:
        S_AA = Aw.T @ Aw  # (D_A, D_A)
        S_BA = Bw.T @ Aw  # (D_B, D_A)
        tr_mean = float(np.trace(S_AA)) / max(D_A, 1)
        if lam is None:
            # Analytical Ledoit-Wolf-like shrinkage floor scaled to N vs D_A
            used_lam = max(0.05 * tr_mean, tr_mean * float(D_A) / max(N * 20.0, 1.0))
        else:
            used_lam = float(lam) * tr_mean
        M = sla.solve(S_AA + used_lam * np.eye(D_A, dtype=np.float64), S_BA.T, assume_a="pos").T

    pred = A @ M.T
    num_cos = np.sum(pred * B, axis=1)
    den_cos = np.linalg.norm(pred, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    train_cos = float(np.mean(num_cos / den_cos))
    r2 = 1.0 - float(np.sum((B - pred) ** 2) / (np.sum((B - B.mean(axis=0)) ** 2) + 1e-12))
    return M, {"train_mean_cos": round(train_cos, 4), "train_r2": round(r2, 4), "lam": round(used_lam, 6)}


def apply_rmsnorm_np(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Exact RMSNorm in FP64: (x / sqrt(mean(x^2) + eps)) * weight."""
    xf = np.asarray(x, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    rms = np.sqrt(np.mean(xf ** 2, axis=-1, keepdims=True) + eps)
    return (xf / rms) * w


def fit_rmsnorm_aligned_cocycle_map(
    X_A_norm: np.ndarray,
    X_B_norm: np.ndarray,
    U_A_norm: np.ndarray,
    U_B_norm: np.ndarray,
    lam: float = 0.05,
) -> Tuple[np.ndarray, float]:
    """
    Fixes Astra Seat 2's RMSNorm mismatch bug: fits `M_in: R^{D_B} -> R^{D_A}` in **post-RMSNorm**
    coordinates (`G_A`, `G_B`) subject to the exact linear equality constraint
      `M_in @ U_B_norm == U_A_norm`
    (so upstream translated writes passing through Recipient B's RMSNorm arrive with zero shrinkage!).
    Uses exact KKT nullspace-constrained ridge in FP64.
    """
    Xa = np.asarray(X_A_norm, dtype=np.float64)  # (N, D_A)
    Xb = np.asarray(X_B_norm, dtype=np.float64)  # (N, D_B)
    Ua = np.asarray(U_A_norm, dtype=np.float64)  # (D_A, r)
    Ub = np.asarray(U_B_norm, dtype=np.float64)  # (D_B, r)
    N, D_B = Xb.shape

    # Orthonormalize Ub via SVD for numerical conditioning
    Q_b, s_b, Vt_b = sla.svd(Ub, full_matrices=False)
    keep = s_b > 1e-8 * (s_b[0] + 1e-12)
    Q_b = Q_b[:, keep]  # (D_B, r_eff)
    # Target on Q_b: M_in @ Q_b = Ua @ Vt_b[keep].T @ diag(1/s_b[keep])
    T_on_Q = (Ua @ Vt_b[keep].T) / s_b[keep][None, :]  # (D_A, r_eff)

    # Project remaining regression strictly onto orthogonal complement of Q_b: Xb_perp = Xb - (Xb Q_b) Q_b^T
    Xb_q = Xb @ Q_b
    Xb_perp = Xb - Xb_q @ Q_b.T
    Xa_res = Xa - Xb_q @ T_on_Q.T

    S = (Xb_perp.T @ Xb_perp) / max(N, 1)
    tr = float(np.trace(S)) / max(D_B, 1)
    M_perp = sla.solve(S + lam * tr * np.eye(D_B), (Xb_perp.T @ Xa_res) / max(N, 1), assume_a="pos").T
    # Zero out any component along Q_b, then add exact constraint T_on_Q @ Q_b^T
    M_perp = M_perp - (M_perp @ Q_b) @ Q_b.T
    M_in = M_perp + T_on_Q @ Q_b.T
    eq_err = float(np.linalg.norm(M_in @ Ub - Ua) / (np.linalg.norm(Ua) + 1e-12))
    return M_in, eq_err


def solve_closed_form_layer_delta_w(
    Z_keys: np.ndarray,
    Delta_Y_targets: np.ndarray,
    Z_bg_cov: Optional[np.ndarray] = None,
    lam_bg: float = 10.0,
    lam_eye: float = 1e-2,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Tier 1 Pure Closed-Form Layer-Wise Reversible `ΔW` Solver (ZERO backprop!):
    Solves for `ΔW ∈ R^{D_out × D_in}` minimizing:
      min_{ΔW}  ||ΔW Z_keys^T - Delta_Y_targets^T||_F^2 + lam_bg * tr(ΔW C_bg ΔW^T) + lam_eye * ||ΔW||_F^2
    yielding exact closed-form:
      ΔW = (Delta_Y^T Z_keys) @ (Z_keys^T Z_keys + lam_bg * C_bg + lam_eye * I)^{-1}
    Works directly for `mlp.down_proj.weight` (`Z_keys` = intermediate SwiGLU acts, `Delta_Y` =
    translated donor residual delta `M_out @ Δh_A`) OR `self_attn.o_proj.weight`!
    """
    Z = np.asarray(Z_keys, dtype=np.float64)        # (N, D_in)
    dY = np.asarray(Delta_Y_targets, dtype=np.float64)  # (N, D_out)
    N, D_in = Z.shape
    assert dY.shape[0] == N

    G_mat = Z.T @ Z  # (D_in, D_in)
    tr_scale = float(np.trace(G_mat)) / max(D_in, 1)
    if Z_bg_cov is not None:
        C_bg = np.asarray(Z_bg_cov, dtype=np.float64)
        c_scale = float(np.trace(C_bg)) / max(D_in, 1)
        # Scale C_bg relative to G_mat trace so lam_bg is dimensionless!
        G_mat = G_mat + float(lam_bg) * (tr_scale / max(c_scale, 1e-12)) * C_bg
    G_mat = G_mat + float(lam_eye) * tr_scale * np.eye(D_in, dtype=np.float64)

    rhs = Z.T @ dY  # (D_in, D_out)
    dW = sla.solve(G_mat, rhs, assume_a="pos").T  # (D_out, D_in)

    pred = Z @ dW.T
    cos_fit = float(np.mean(
        np.sum(pred * dY, axis=1) / (np.linalg.norm(pred, axis=1) * np.linalg.norm(dY, axis=1) + 1e-12)
    ))
    rel_err = float(np.linalg.norm(pred - dY) / (np.linalg.norm(dY) + 1e-12))
    return dW, {"fit_cos": round(cos_fit, 4), "fit_rel_err": round(rel_err, 4), "dW_frob": round(float(np.linalg.norm(dW)), 4)}


# ======================================================================================
# 4. Seat 3's 2-Level Hierarchical Routed Fact Bank (5,120 Slots in ~15.8 MiB BF16!)
# ======================================================================================

class HierarchicalRoutedFactBank(nn.Module):
    """
    2-Level Hierarchical Routed Fact Bank (Astra Seat 3 Architecture):
      1. Project normalized recipient hidden state `x_hat = x / ||x||` via `P [d_key, D_B]`
         (default `d_key = 64` or `128`).
      2. Score `G` coarse group centroids `C [G, d_key]`, pick winning group `g*` (or top-2 groups).
      3. Score `K` fine keys `E [G, K, d_key]` in the winning group.
      4. Emit scaled write vector `V [G, K, D_B]` ONLY if:
           - `best_score >= tau_abs` (calibrated on **joint bank + sequence max** over OOD text!)
           - AND `best_score - second_score >= margin_min`
           - AND (optional) cumulative prompt latch `fired_slots < max_fires_per_prompt` (exact
             prompt-only mask using `attention_mask` so left-padding NEVER steals dose and
             multi-token FC continuations NEVER leak!).
      5. Hard bypass (`enabled = False`) guarantees bit-identical (`torch.equal == True`) removal!
    """

    def __init__(
        self,
        d_model: int,
        d_key: int,
        n_groups: int,
        group_size: int,
        P: np.ndarray,
        C: np.ndarray,
        E: np.ndarray,
        V: np.ndarray,
        valid_mask: np.ndarray,
        tau_abs: float = 0.85,
        margin_min: float = 0.03,
        scale: float = 1.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_key = int(d_key)
        self.n_groups = int(n_groups)
        self.group_size = int(group_size)
        self.tau_abs = float(tau_abs)
        self.margin_min = float(margin_min)
        self.scale = float(scale)
        self.enabled: bool = True

        self.register_buffer("P", torch.tensor(P, dtype=torch.float32))               # (d_key, D_B)
        self.register_buffer("C", torch.tensor(C, dtype=torch.float32))               # (G, d_key)
        self.register_buffer("E", torch.tensor(E, dtype=torch.float32))               # (G, K, d_key)
        self.V = nn.Parameter(torch.tensor(V, dtype=torch.float32), requires_grad=False)  # (G, K, D_B)
        self.register_buffer("valid_mask", torch.tensor(valid_mask, dtype=torch.bool)) # (G, K)

    @classmethod
    def from_keys_and_writes(
        cls,
        keys_B: np.ndarray,
        writes_B: np.ndarray,
        d_key: int = 64,
        n_groups: Optional[int] = None,
        tau_abs: float = 0.85,
        margin_min: float = 0.03,
        scale: float = 1.0,
        seed: int = 20261126,
    ) -> "HierarchicalRoutedFactBank":
        """Build 2-level hierarchical bank from `(N_facts, D_B)` keys and `(N_facts, D_B)` writes."""
        K_mat = np.asarray(keys_B, dtype=np.float64)
        V_mat = np.asarray(writes_B, dtype=np.float64)
        N, D_B = K_mat.shape
        assert V_mat.shape == (N, D_B)

        # Normalize keys to unit sphere
        K_norm = K_mat / (np.linalg.norm(K_mat, axis=1, keepdims=True) + 1e-12)
        # Orthonormal projection P via SVD of K_norm (exact subspace preserving all key geometry!)
        _, _, Vt = sla.svd(K_norm, full_matrices=False)
        dk = min(int(d_key), Vt.shape[0], D_B)
        P = Vt[:dk]  # (dk, D_B)
        Z = K_norm @ P.T  # (N, dk)
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

        G = n_groups if n_groups is not None else max(1, min(80, int(round(math.sqrt(N)))))
        # Simple deterministic spherical k-means into G groups
        rng = np.random.default_rng(seed)
        init_idx = rng.choice(N, size=min(G, N), replace=False)
        C = Z[init_idx].copy()
        if len(C) < G:
            C = np.vstack([C, np.zeros((G - len(C), dk))])
        for _ in range(15):
            sims = Z @ C.T
            assign = np.argmax(sims, axis=1)
            for g in range(G):
                m = assign == g
                if np.any(m):
                    c_new = Z[m].mean(axis=0)
                    C[g] = c_new / (np.linalg.norm(c_new) + 1e-12)

        assign = np.argmax(Z @ C.T, axis=1)
        group_lists = [np.where(assign == g)[0] for g in range(G)]
        K_max = max(max(len(gl) for gl in group_lists), 2)

        E = np.zeros((G, K_max, dk), dtype=np.float64)
        V = np.zeros((G, K_max, D_B), dtype=np.float64)
        valid = np.zeros((G, K_max), dtype=bool)
        for g, idxs in enumerate(group_lists):
            for slot, orig_i in enumerate(idxs):
                E[g, slot] = Z[orig_i]
                V[g, slot] = V_mat[orig_i]
                valid[g, slot] = True

        return cls(D_B, dk, G, K_max, P, C, E, V, valid, tau_abs=tau_abs, margin_min=margin_min, scale=scale)

    def score_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given `x` of shape `(B, T, D_B)`, returns `(best_score, margin, best_g, best_k)` each `(B, T)`.
        Checks top-2 coarse groups to avoid boundary misses!
        """
        xf = x.float()
        x_hat = F.normalize(xf, dim=-1, eps=1e-8)
        q = F.normalize(torch.matmul(x_hat, self.P.t()), dim=-1, eps=1e-8)  # (B, T, dk)
        # Coarse group scores: (B, T, G)
        g_scores = torch.matmul(q, self.C.t())
        top_g_k = min(2, self.n_groups)
        top_g = torch.topk(g_scores, k=top_g_k, dim=-1).indices  # (B, T, top_g_k)

        B, T, _ = q.shape
        # Gather E for top_g_k groups: (B, T, top_g_k, K, dk)
        E_sel = self.E[top_g]
        M_sel = self.valid_mask[top_g]  # (B, T, top_g_k, K)
        # Dot product with q: (B, T, top_g_k, K)
        fine = torch.einsum("btd,btgkd->btgk", q, E_sel)
        fine = fine.masked_fill(~M_sel, -1e4)
        flat = fine.reshape(B, T, -1)  # (B, T, top_g_k * K)
        top2_vals, top2_idx = torch.topk(flat, k=min(2, flat.shape[-1]), dim=-1)
        best_score = top2_vals[..., 0]
        second_score = top2_vals[..., 1] if top2_vals.shape[-1] > 1 else torch.full_like(best_score, -1e4)
        margin = best_score - second_score

        K = self.group_size
        best_flat = top2_idx[..., 0]
        g_pos = best_flat // K
        best_k = best_flat % K
        best_g = torch.gather(top_g, -1, g_pos.unsqueeze(-1)).squeeze(-1)
        return best_score, margin, best_g, best_k

    def forward(self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return additive residual delta `(B, T, D_B)`.
        `valid_token_mask` `(B, T)` explicitly excludes BOS and left-padding (fixing Seat 1's P0 bug!).
        """
        if not self.enabled:
            return torch.zeros_like(x)
        best_score, margin, best_g, best_k = self.score_tokens(x)
        fire = (best_score >= self.tau_abs) & (margin >= self.margin_min)
        if valid_token_mask is not None:
            fire = fire & valid_token_mask.to(fire.device)
        if not torch.any(fire):
            return torch.zeros_like(x)
        # Emit winning write vector only at firing positions
        v_emit = self.V[best_g, best_k]  # (B, T, D_B)
        delta = (fire.unsqueeze(-1).float() * self.scale) * v_emit
        return delta.to(x.dtype)

    def calibrate_joint_max_threshold(
        self,
        ood_token_states: Sequence[torch.Tensor],
        alpha_seq: float = 0.005,
        margin_safety: float = 0.01,
    ) -> Dict[str, float]:
        """Calibrate `tau_abs` on the **joint max over the entire bank AND entire sequence** on OOD text!"""
        seq_maxes = []
        with torch.no_grad():
            for x_seq in ood_token_states:
                # x_seq: (1, T, D_B), skip BOS col 0
                if x_seq.shape[1] <= 1:
                    continue
                bs, _, _, _ = self.score_tokens(x_seq[:, 1:, :].to(self.P.device))
                seq_maxes.append(float(bs.max().item()))
        arr = np.sort(np.asarray(seq_maxes, dtype=np.float64))
        idx = min(len(arr) - 1, int(math.ceil((1.0 - alpha_seq) * len(arr))) - 1)
        q_seq = float(arr[max(idx, 0)])
        self.tau_abs = q_seq + margin_safety
        return {
            "n_ood_seqs": len(arr),
            "ood_seq_max_p95": round(float(np.quantile(arr, 0.95)), 4),
            "ood_seq_max_max": round(float(arr[-1]), 4),
            "calibrated_tau_abs": round(self.tau_abs, 4),
        }

    def n_bytes_bf16(self) -> int:
        return int((self.P.numel() + self.C.numel() + self.E.numel() + self.V.numel()) * 2)


# ======================================================================================
# 5. Pure-CPU Unit Tests
# ======================================================================================

def run_unit_test() -> None:
    rng = np.random.default_rng(7)

    # 1. Contrast bridge exactness
    D_A, D_B, N = 48, 32, 200
    W_true = rng.standard_normal((D_B, D_A)) / math.sqrt(D_A)
    dA = rng.standard_normal((N, D_A))
    dB = dA @ W_true.T + 0.02 * rng.standard_normal((N, D_B))
    M, info = fit_contrast_bridge(dA, dB, lam=1e-4, mode="ridge")
    assert info["train_mean_cos"] > 0.995, f"Contrast bridge failed: {info}"

    # 2. RMSNorm-aligned cocycle equality constraint (exact <1e-12!)
    Xa_n = rng.standard_normal((150, D_A))
    Xb_n = rng.standard_normal((150, D_B))
    Ua_n = rng.standard_normal((D_A, 6))
    Ub_n = rng.standard_normal((D_B, 6))
    M_in, eq_err = fit_rmsnorm_aligned_cocycle_map(Xa_n, Xb_n, Ua_n, Ub_n, lam=0.05)
    assert eq_err < 1e-12, f"RMSNorm cocycle err too high: {eq_err}"

    # 3. Closed-form layer-wise ΔW solver
    Z = rng.standard_normal((30, 64))
    dY = rng.standard_normal((30, 32))
    C_bg = np.eye(64)
    dW, dw_info = solve_closed_form_layer_delta_w(Z, dY, Z_bg_cov=C_bg, lam_bg=0.01, lam_eye=1e-4)
    assert dw_info["fit_cos"] > 0.98, f"Closed-form ΔW fit failed: {dw_info}"

    # 4. ReversibleDeltaPack bit-exact rollback on real nn.Module
    lin = nn.Linear(32, 32, bias=False).to(torch.bfloat16)
    x_probe = torch.randn(2, 5, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        base_out = lin(x_probe).clone()
    w_before = lin.weight.detach().clone()
    pack = ReversibleDeltaPack({"weight": torch.randn(32, 32) * 0.5})
    pack.apply(lin)
    with torch.no_grad():
        edited_out = lin(x_probe).clone()
    assert not torch.equal(edited_out, base_out)
    pack.revert(lin)
    assert torch.equal(lin.weight, w_before)
    with torch.no_grad():
        restored_out = lin(x_probe).clone()
    assert torch.equal(restored_out, base_out)

    # 5. 2-Level Hierarchical Routed Fact Bank (exact routing + joint max calibration + hard bypass)
    N_facts = 128
    K_B = rng.standard_normal((N_facts, 64))
    V_B = rng.standard_normal((N_facts, 64))
    bank = HierarchicalRoutedFactBank.from_keys_and_writes(K_B, V_B, d_key=32, n_groups=8, tau_abs=0.80, margin_min=0.05)
    ood_seqs = [torch.randn(1, 12, 64) for _ in range(40)]
    cal = bank.calibrate_joint_max_threshold(ood_seqs, alpha_seq=0.0, margin_safety=0.02)
    # Query exact stored key #17
    q_t = torch.tensor(K_B[17:18], dtype=torch.float32).unsqueeze(0)  # (1, 1, 64)
    y_hit = bank(q_t)
    cos_hit = float(F.cosine_similarity(y_hit[0, 0], torch.tensor(V_B[17], dtype=torch.float32), dim=0).item())
    assert cos_hit > 0.9999, f"Fact bank retrieval miss: {cos_hit}"
    # Verify hard bypass returns exact zero
    bank.enabled = False
    assert torch.equal(bank(q_t), torch.zeros_like(q_t))

    print(
        f"[large_to_small --unit-test] ALL 5 PASSED! "
        f"(contrast_cos={info['train_mean_cos']}, rms_cocycle_err={eq_err:.2e}, "
        f"dW_cos={dw_info['fit_cos']}, rollback=torch.equal, bank_hit_cos={cos_hit:.5f}, cal_tau={cal['calibrated_tau_abs']})"
    )


if __name__ == "__main__":
    if "--unit-test" in sys.argv:
        run_unit_test()
