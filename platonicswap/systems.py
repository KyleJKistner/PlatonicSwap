"""
systems.py — Systems-Level & Multi-Model Improvements to Plato's Key.

Implements:
1. Multi-Donor Platonic Barycenter (uniform, val-weighted, inverse-covariance GLS
   optimal weights per Theorem 4) with exact norm restoration & attenuation diagnostics.
2. Generalized Procrustes Analysis (GPA) into ONE shared k-dim Platonic coordinate
   space (full-overlap & partial/low-overlap synchronization + spectral O(k) sync)
   and closed-loop holonomy measurement on factual subspaces.
3. Multi-Regime Forward-Hook Patching Engine for breaking the single-layer self-swap
   ceiling:
   - Single-layer residual swap (baseline)
   - Contiguous multi-layer window residual patching (W = 2..4)
   - Additive MLP-output-only patching (Delta z_MLP — Hazy Hebbian-MLP test)
   - Dual-token patching (last_subj + last_prompt)
   - Difference-vector / residual-steering transfer & gain-tuned replacement
4. Dark-Memory Recovery & Scientific Audit helpers (same-model ICL/paraphrase ceiling,
   collateral damage / retention rate, net recall gain, weaker->stronger slice).
5. 2-Hop Unspoken Latent Bridge Transfer helpers (2-hop->2-hop & 1-hop entity key ->
   2-hop intermediate-layer bridge injection).

Row convention throughout: key matrices X have shape (N, d).
All maps are strictly closed-form (zero gradient descent).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binomtest
import torch
import torch.nn as nn


# =============================================================================
# 1. Core Closed-Form Procrustes & Mapping Primitives (Row Convention: N x d)
# =============================================================================


@dataclass
class ProcrustesMap:
    """Closed-form orthogonal/semi-orthogonal Procrustes map (row convention).

    Application: x_hat_b = mu_b + gain * scale * ((x_a - mu_a) @ W)
    where W has shape (d_a, d_b) with orthonormal columns (if d_a >= d_b)
    or orthonormal rows (if d_a <= d_b).
    """

    mu_a: np.ndarray  # (d_a,)
    mu_b: np.ndarray  # (d_b,)
    W: np.ndarray  # (d_a, d_b)
    scale: float
    name: str = "procrustes_global"
    meta: Dict[str, Any] = field(default_factory=dict)

    def transform(self, X_a: np.ndarray, gain: float = 1.0) -> np.ndarray:
        X_a = np.asarray(X_a, dtype=np.float64)
        single = X_a.ndim == 1
        if single:
            X_a = X_a[None, :]
        out = self.mu_b[None, :] + (gain * self.scale) * ((X_a - self.mu_a[None, :]) @ self.W)
        return out[0] if single else out

    def transform_centered(self, X_a: np.ndarray, gain: float = 1.0) -> np.ndarray:
        X_a = np.asarray(X_a, dtype=np.float64)
        single = X_a.ndim == 1
        if single:
            X_a = X_a[None, :]
        out = (gain * self.scale) * ((X_a - self.mu_a[None, :]) @ self.W)
        return out[0] if single else out

    def transform_delta(self, delta_a: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Map a difference vector (z_A(s') - z_A(s)) into B's space."""
        delta_a = np.asarray(delta_a, dtype=np.float64)
        single = delta_a.ndim == 1
        if single:
            delta_a = delta_a[None, :]
        out = (alpha * self.scale) * (delta_a @ self.W)
        return out[0] if single else out


def fit_procrustes(
    X_a: np.ndarray,
    X_b: np.ndarray,
    scale_mode: str = "global_frob",
) -> ProcrustesMap:
    """Fit exact Plato's Key Procrustes map from X_a (N, d_a) to X_b (N, d_b)."""
    X_a = np.asarray(X_a, dtype=np.float64)
    X_b = np.asarray(X_b, dtype=np.float64)
    assert X_a.ndim == 2 and X_b.ndim == 2 and X_a.shape[0] == X_b.shape[0]
    mu_a = X_a.mean(axis=0)
    mu_b = X_b.mean(axis=0)
    Xa_c = X_a - mu_a
    Xb_c = X_b - mu_b

    # Thin SVD of cross-covariance C = Xa_c^T Xb_c (d_a x d_b)
    C = Xa_c.T @ Xb_c
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    W = U @ Vt  # (d_a, d_b), semi-orthogonal

    norm_a = float(np.linalg.norm(Xa_c, "fro"))
    norm_b = float(np.linalg.norm(Xb_c, "fro"))
    if norm_a < 1e-12:
        scale = 1.0
    elif scale_mode == "global_frob":
        scale = norm_b / norm_a
    elif scale_mode == "ls":
        scale = float(np.sum(S)) / (norm_a ** 2)
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")

    return ProcrustesMap(
        mu_a=mu_a,
        mu_b=mu_b,
        W=W,
        scale=scale,
        name=f"procrustes_{scale_mode}",
        meta={"n_train": int(X_a.shape[0]), "d_a": int(X_a.shape[1]), "d_b": int(X_b.shape[1])},
    )


def fit_procrustes_fast_qr(
    X_a: np.ndarray,
    X_b: np.ndarray,
    scale_mode: str = "global_frob",
) -> ProcrustesMap:
    """Fast O(N^2 d) reduced-QR thin Procrustes for val-sweep layer selection when N << d.

    Exact on span(X_a); maps the orthogonal nullspace (I - Q_A Q_A^T) to 0 instead of
    arbitrary LAPACK nullspace vectors. Use exact `fit_procrustes` for final test evals.
    """
    X_a = np.asarray(X_a, dtype=np.float64)
    X_b = np.asarray(X_b, dtype=np.float64)
    if X_a.shape[0] >= min(X_a.shape[1], X_b.shape[1]):
        return fit_procrustes(X_a, X_b, scale_mode=scale_mode)
    mu_a = X_a.mean(axis=0)
    mu_b = X_b.mean(axis=0)
    Xa_c = X_a - mu_a
    Xb_c = X_b - mu_b
    Qa, Ra = np.linalg.qr(Xa_c.T, mode="reduced")  # (d_a, N), (N, N)
    Qb, Rb = np.linalg.qr(Xb_c.T, mode="reduced")  # (d_b, N), (N, N)
    U, S, Vt = np.linalg.svd(Ra @ Rb.T, full_matrices=False)
    W = (Qa @ U) @ (Qb @ Vt.T).T  # (d_a, d_b)
    norm_a = float(np.linalg.norm(Xa_c, "fro"))
    norm_b = float(np.linalg.norm(Xb_c, "fro"))
    if norm_a < 1e-12:
        scale = 1.0
    elif scale_mode == "global_frob":
        scale = norm_b / norm_a
    elif scale_mode == "ls":
        scale = float(np.sum(S)) / (norm_a ** 2)
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")
    return ProcrustesMap(
        mu_a=mu_a,
        mu_b=mu_b,
        W=W,
        scale=scale,
        name=f"procrustes_qr_{scale_mode}",
        meta={"n_train": int(X_a.shape[0]), "d_a": int(X_a.shape[1]), "d_b": int(X_b.shape[1])},
    )


def fit_random_orthogonal_control(
    X_a: np.ndarray,
    X_b: np.ndarray,
    seed: int = 42,
) -> ProcrustesMap:
    """Random Haar semi-orthogonal map with matched means and Frobenius scale."""
    X_a = np.asarray(X_a, dtype=np.float64)
    X_b = np.asarray(X_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    mu_a = X_a.mean(axis=0)
    mu_b = X_b.mean(axis=0)
    d_a, d_b = X_a.shape[1], X_b.shape[1]
    G = rng.standard_normal((d_a, d_b))
    U, _, Vt = np.linalg.svd(G, full_matrices=False)
    W = U @ Vt
    norm_a = float(np.linalg.norm(X_a - mu_a, "fro"))
    norm_b = float(np.linalg.norm(X_b - mu_b, "fro"))
    scale = norm_b / max(norm_a, 1e-12)
    return ProcrustesMap(mu_a=mu_a, mu_b=mu_b, W=W, scale=scale, name="random_orthogonal")


def fit_shuffled_procrustes_control(
    X_a: np.ndarray,
    X_b: np.ndarray,
    seed: int = 42,
) -> ProcrustesMap:
    """Procrustes fit on randomly permuted subject pairings (Plato's Key control)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X_a.shape[0])
    pmap = fit_procrustes(X_a[perm], X_b)
    pmap.name = "shuffled_procrustes"
    return pmap


def apply_difference_vector(
    pmap: ProcrustesMap,
    X_a_src: np.ndarray,
    X_a_tgt: np.ndarray,
    X_b_src: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Residual-steering difference-vector transfer:
    z_hat_B = z_B(s) + alpha * scale * (z_A(s') - z_A(s)) @ W.
    """
    delta_a = np.asarray(X_a_tgt, dtype=np.float64) - np.asarray(X_a_src, dtype=np.float64)
    return np.asarray(X_b_src, dtype=np.float64) + pmap.transform_delta(delta_a, alpha=alpha)


def apply_gain_replacement(
    pmap: ProcrustesMap,
    X_a_tgt: np.ndarray,
    alpha: float = 1.0,
    per_sample_ref: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Gain-tuned replacement (Theorem 1 optimal gain alpha >= 1).

    If per_sample_ref is provided (e.g. X_b_src), restores per-sample centered norm
    to alpha * ||per_sample_ref - mu_b||_2.
    """
    centered = pmap.transform_centered(X_a_tgt, gain=1.0)
    single = centered.ndim == 1
    if single:
        centered = centered[None, :]
    if per_sample_ref is not None:
        ref = np.asarray(per_sample_ref, dtype=np.float64)
        if ref.ndim == 1:
            ref = ref[None, :]
        ref_norm = np.linalg.norm(ref - pmap.mu_b[None, :], axis=1, keepdims=True)
        cur_norm = np.linalg.norm(centered, axis=1, keepdims=True)
        centered = centered * (alpha * ref_norm / np.maximum(cur_norm, 1e-12))
    else:
        centered = alpha * centered
    out = pmap.mu_b[None, :] + centered
    return out[0] if single else out


# =============================================================================
# 2. Pillar 1A: Multi-Donor Platonic Barycenter (Theorem 4 + Norm Restoration)
# =============================================================================


@dataclass
class BarycenterConfig:
    donor_names: List[str]
    weights: np.ndarray  # (M,) summing to 1
    weight_method: str
    norm_mode: str  # "none", "mean_norm", "rms_norm", "global_spread"
    gain: float = 1.0
    global_scale: float = 1.0
    error_cov: Optional[np.ndarray] = None  # (M, M)
    error_corr: Optional[np.ndarray] = None  # (M, M)
    mean_offdiag_rho: Optional[float] = None
    theoretical_snr_gain: Optional[float] = None
    empirical_snr_gain: Optional[float] = None


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex {w >= 0, sum w = 1}."""
    v = np.asarray(v, dtype=np.float64)
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    return np.maximum(v - theta, 0.0)


def fit_barycenter_config(
    donor_names: Sequence[str],
    mapped_fit_list: Sequence[np.ndarray],
    mu_b_list: Sequence[np.ndarray],
    X_b_fit: np.ndarray,
    val_accs: Optional[Sequence[float]] = None,
    weight_method: str = "uniform",
    norm_mode: str = "rms_norm",
    gain: float = 1.0,
    ridge: float = 1e-6,
    nonnegative_gls: bool = True,
) -> BarycenterConfig:
    """Fit multi-donor barycenter weights and global spread scale STRICTLY on fit/val data."""
    M = len(donor_names)
    assert M >= 1 and len(mapped_fit_list) == M and len(mu_b_list) == M
    X_b_fit = np.asarray(X_b_fit, dtype=np.float64)
    N_fit, d_b = X_b_fit.shape

    res_list = [np.asarray(Xh, dtype=np.float64) - X_b_fit for Xh in mapped_fit_list]
    E = np.stack([r.reshape(-1) for r in res_list], axis=0)
    Sigma_err = (E @ E.T) / max(N_fit, 1)
    diag = np.sqrt(np.maximum(np.diag(Sigma_err), 1e-12))
    corr = Sigma_err / np.outer(diag, diag)
    corr = np.clip(corr, -1.0, 1.0)
    if M > 1:
        off_mask = ~np.eye(M, dtype=bool)
        mean_rho = float(corr[off_mask].mean())
        theo_snr = float(M / max(1.0 + (M - 1) * mean_rho, 1e-9))
    else:
        mean_rho = 0.0
        theo_snr = 1.0

    if weight_method == "uniform":
        w = np.full(M, 1.0 / M, dtype=np.float64)
    elif weight_method in ("val_acc", "val_acc_sq"):
        if val_accs is None:
            raise ValueError(f"val_accs required for weight_method={weight_method}")
        scores = np.maximum(np.asarray(val_accs, dtype=np.float64), 1e-6)
        if weight_method == "val_acc_sq":
            scores = scores ** 2
        w = scores / scores.sum()
    elif weight_method == "inv_cov":
        reg = ridge * float(np.trace(Sigma_err) / M) * np.eye(M)
        S_reg = Sigma_err + reg
        if nonnegative_gls and M > 1:
            res = minimize(
                lambda x: 0.5 * float(x @ S_reg @ x),
                x0=np.full(M, 1.0 / M),
                jac=lambda x: S_reg @ x,
                bounds=[(0.0, 1.0)] * M,
                constraints={"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
                method="SLSQP",
            )
            w = _project_simplex(res.x) if res.success else np.full(M, 1.0 / M)
        else:
            inv_ones = np.linalg.solve(S_reg, np.ones(M))
            w = inv_ones / np.sum(inv_ones)
    else:
        raise ValueError(f"Unknown weight_method: {weight_method}")

    mean_single_mse = float(np.mean(np.diag(Sigma_err)))
    bary_mse = float(w @ Sigma_err @ w)
    emp_snr = mean_single_mse / max(bary_mse, 1e-12)

    mu_bar = sum(w[m] * np.asarray(mu_b_list[m], dtype=np.float64) for m in range(M))
    deltas_fit = [
        np.asarray(mapped_fit_list[m], dtype=np.float64) - np.asarray(mu_b_list[m], dtype=np.float64)[None, :]
        for m in range(M)
    ]
    delta_bar_fit = sum(w[m] * deltas_fit[m] for m in range(M))
    target_spread = float(np.linalg.norm(X_b_fit - mu_bar[None, :], "fro"))
    bary_spread = float(np.linalg.norm(delta_bar_fit, "fro"))
    global_scale = target_spread / max(bary_spread, 1e-12)

    return BarycenterConfig(
        donor_names=list(donor_names),
        weights=w,
        weight_method=weight_method,
        norm_mode=norm_mode,
        gain=float(gain),
        global_scale=float(global_scale),
        error_cov=Sigma_err,
        error_corr=corr,
        mean_offdiag_rho=mean_rho,
        theoretical_snr_gain=theo_snr,
        empirical_snr_gain=emp_snr,
    )


def aggregate_multi_donor_keys(
    mapped_keys_list: Sequence[np.ndarray],
    mu_b_list: Sequence[np.ndarray],
    config: BarycenterConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Aggregate M mapped donor key arrays into a single recipient barycenter key."""
    M = len(mapped_keys_list)
    w = config.weights
    assert len(w) == M and len(mu_b_list) == M

    single = np.asarray(mapped_keys_list[0]).ndim == 1
    arrs = [
        np.asarray(x, dtype=np.float64)[None, :] if np.asarray(x).ndim == 1 else np.asarray(x, dtype=np.float64)
        for x in mapped_keys_list
    ]
    mus = [np.asarray(mu, dtype=np.float64) for mu in mu_b_list]

    mu_bar = sum(w[m] * mus[m] for m in range(M))
    deltas = [arrs[m] - mus[m][None, :] for m in range(M)]
    delta_bar = sum(w[m] * deltas[m] for m in range(M))

    norms = [np.linalg.norm(d, axis=1, keepdims=True) for d in deltas]
    bar_norm = np.linalg.norm(delta_bar, axis=1, keepdims=True)
    mean_norm = sum(w[m] * norms[m] for m in range(M))
    rms_norm = np.sqrt(sum(w[m] * (norms[m] ** 2) for m in range(M)))

    attenuation_ratio = float(np.mean(bar_norm / np.maximum(rms_norm, 1e-12)))

    pair_cos: List[float] = []
    for i in range(M):
        for j in range(i + 1, M):
            c = np.sum(deltas[i] * deltas[j], axis=1) / np.maximum(norms[i][:, 0] * norms[j][:, 0], 1e-12)
            pair_cos.append(float(np.mean(c)))

    if config.norm_mode == "none":
        restored = config.gain * delta_bar
    elif config.norm_mode == "mean_norm":
        restored = config.gain * delta_bar * (mean_norm / np.maximum(bar_norm, 1e-12))
    elif config.norm_mode == "rms_norm":
        restored = config.gain * delta_bar * (rms_norm / np.maximum(bar_norm, 1e-12))
    elif config.norm_mode == "global_spread":
        restored = (config.gain * config.global_scale) * delta_bar
    else:
        raise ValueError(f"Unknown norm_mode: {config.norm_mode}")

    X_bary = mu_bar[None, :] + restored
    diag = {
        "mean_norm_attenuation_ratio": attenuation_ratio,
        "mean_pairwise_donor_cos": float(np.mean(pair_cos)) if pair_cos else 1.0,
        "effective_norm_boost": float(
            np.mean(np.linalg.norm(restored, axis=1, keepdims=True) / np.maximum(bar_norm, 1e-12))
        ),
    }
    return (X_bary[0] if single else X_bary), diag


# =============================================================================
# 3. Pillar 1B: Generalized Procrustes Analysis (GPA, Fig. 6) & Holonomy
# =============================================================================


@dataclass
class GPASpace:
    """Shared k-dimensional Platonic coordinate space across L LLMs (Fig. 6)."""

    models: List[str]
    k: int
    mu: Dict[str, np.ndarray]
    rms_scale: Dict[str, float]
    Q: Dict[str, np.ndarray]
    consensus: np.ndarray
    objective_history: List[float]
    converged: bool

    def encode(self, model: str, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        single = X.ndim == 1
        if single:
            X = X[None, :]
        Z = ((X - self.mu[model][None, :]) / self.rms_scale[model]) @ self.Q[model]
        return Z[0] if single else Z

    def decode(self, model: str, Z: np.ndarray, scale_boost: float = 1.0) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float64)
        single = Z.ndim == 1
        if single:
            Z = Z[None, :]
        X = self.mu[model][None, :] + (self.rms_scale[model] * scale_boost) * (Z @ self.Q[model].T)
        return X[0] if single else X

    def get_pairwise_map(
        self,
        src: str,
        dst: str,
        X_src_ref: Optional[np.ndarray] = None,
        X_dst_ref: Optional[np.ndarray] = None,
        restore_full_norm: bool = True,
    ) -> ProcrustesMap:
        W = self.Q[src] @ self.Q[dst].T
        if restore_full_norm and X_src_ref is not None and X_dst_ref is not None:
            proj_src = (np.asarray(X_src_ref, dtype=np.float64) - self.mu[src][None, :]) @ self.Q[src]
            norm_proj_src = float(np.linalg.norm(proj_src, "fro"))
            norm_dst = float(np.linalg.norm(np.asarray(X_dst_ref, dtype=np.float64) - self.mu[dst][None, :], "fro"))
            scale = norm_dst / max(norm_proj_src, 1e-12)
        else:
            scale = self.rms_scale[dst] / max(self.rms_scale[src], 1e-12)
        return ProcrustesMap(
            mu_a=self.mu[src].copy(),
            mu_b=self.mu[dst].copy(),
            W=W,
            scale=float(scale),
            name=f"gpa_k{self.k}",
            meta={"k": self.k, "src": src, "dst": dst},
        )

    def aggregate_donors_in_platonic_space(
        self,
        donor_keys: Dict[str, np.ndarray],
        recipient: str,
        weights: Optional[Sequence[float]] = None,
        norm_mode: str = "rms_norm",
        gain: float = 1.0,
        scale_boost: float = 1.0,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Fig. 6 Native Multi-Donor Platonic Barycenter directly inside the shared k-dim space!

        Encodes each donor into the shared k-dim Platonic space via Q_m, averages with
        exact norm restoration in Platonic coordinates, and decodes once via Q_recipient^T.
        Requires only O(L) model projections instead of O(L^2) pairwise maps.
        """
        donors = list(donor_keys.keys())
        M = len(donors)
        w = np.full(M, 1.0 / M, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
        w = w / np.sum(w)

        single = np.asarray(donor_keys[donors[0]]).ndim == 1
        Zs = [self.encode(m, donor_keys[m]) for m in donors]
        Zs = [z[None, :] if z.ndim == 1 else z for z in Zs]

        Z_bar = sum(w[i] * Zs[i] for i in range(M))
        norms = [np.linalg.norm(z, axis=1, keepdims=True) for z in Zs]
        bar_norm = np.linalg.norm(Z_bar, axis=1, keepdims=True)
        rms_norm = np.sqrt(sum(w[i] * (norms[i] ** 2) for i in range(M)))
        mean_norm = sum(w[i] * norms[i] for i in range(M))
        atten = float(np.mean(bar_norm / np.maximum(rms_norm, 1e-12)))

        if norm_mode == "rms_norm":
            Z_restored = gain * Z_bar * (rms_norm / np.maximum(bar_norm, 1e-12))
        elif norm_mode == "mean_norm":
            Z_restored = gain * Z_bar * (mean_norm / np.maximum(bar_norm, 1e-12))
        elif norm_mode == "none":
            Z_restored = gain * Z_bar
        else:
            raise ValueError(f"Unknown norm_mode: {norm_mode}")

        X_hat = self.decode(recipient, Z_restored, scale_boost=scale_boost)
        return (X_hat[0] if single else X_hat), {"platonic_norm_attenuation": atten}


def fit_gpa_platonic_space(
    X_dict: Dict[str, np.ndarray],
    k: Optional[int] = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> GPASpace:
    """Fit Generalized Procrustes Analysis (GPA) across L models into ONE shared k-dim space."""
    models = sorted(X_dict.keys())
    L = len(models)
    assert L >= 2
    N = X_dict[models[0]].shape[0]
    min_d = min(X_dict[m].shape[1] for m in models)
    k_eff = min_d if k is None else min(int(k), min_d, N)

    mu: Dict[str, np.ndarray] = {}
    rms_scale: Dict[str, float] = {}
    X_norm: Dict[str, np.ndarray] = {}
    for m in models:
        X = np.asarray(X_dict[m], dtype=np.float64)
        assert X.shape[0] == N
        mu_m = X.mean(axis=0)
        Xc = X - mu_m
        s_m = max(float(np.linalg.norm(Xc, "fro")) / math.sqrt(max(N, 1)), 1e-12)
        mu[m] = mu_m
        rms_scale[m] = s_m
        X_norm[m] = Xc / s_m

    X_cat = np.concatenate([X_norm[m] for m in models], axis=1)
    U_init, _, _ = np.linalg.svd(X_cat, full_matrices=False)
    G = U_init[:, :k_eff].copy()
    G /= max(float(np.linalg.norm(G, "fro")), 1e-12)

    Q: Dict[str, np.ndarray] = {}
    obj_hist: List[float] = []
    converged = False

    for it in range(max_iter):
        S_sum = np.zeros((N, k_eff), dtype=np.float64)
        for m in models:
            C_m = X_norm[m].T @ G
            U_m, _, Vt_m = np.linalg.svd(C_m, full_matrices=False)
            Q_m = U_m @ Vt_m
            Q[m] = Q_m
            S_sum += X_norm[m] @ Q_m

        phi = float(np.linalg.norm(S_sum, "fro"))
        obj_hist.append(phi)
        G_new = S_sum / max(phi, 1e-12)
        if it > 0 and abs(obj_hist[-1] - obj_hist[-2]) / max(obj_hist[-2], 1e-12) < tol:
            G = G_new
            converged = True
            break
        G = G_new

    consensus = sum(X_norm[m] @ Q[m] for m in models) / float(L)
    return GPASpace(
        models=models,
        k=k_eff,
        mu=mu,
        rms_scale=rms_scale,
        Q=Q,
        consensus=consensus,
        objective_history=obj_hist,
        converged=converged,
    )


def fit_gpa_partial_overlap(
    X_dict: Dict[str, np.ndarray],
    mask_dict: Dict[str, np.ndarray],
    k: Optional[int] = None,
    max_iter: int = 120,
    tol: float = 1e-7,
) -> GPASpace:
    """Partial-Overlap GPA Synchronization for low-overlap pairwise transfer."""
    models = sorted(X_dict.keys())
    N_total = X_dict[models[0]].shape[0]
    min_d = min(X_dict[m].shape[1] for m in models)

    counts = np.zeros(N_total, dtype=np.int64)
    for m in models:
        counts += np.asarray(mask_dict[m], dtype=bool).astype(np.int64)
    active_rows = counts >= 2
    n_active = int(active_rows.sum())
    assert n_active >= 2, "Need at least 2 facts shared by >= 2 models"

    k_eff = min_d if k is None else min(int(k), min_d, n_active)

    mu: Dict[str, np.ndarray] = {}
    rms_scale: Dict[str, float] = {}
    X_norm: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}

    for m in models:
        mask_m = np.asarray(mask_dict[m], dtype=bool)[active_rows]
        X_full = np.asarray(X_dict[m], dtype=np.float64)[active_rows]
        obs = X_full[mask_m]
        mu_m = obs.mean(axis=0)
        obs_c = obs - mu_m
        s_m = max(float(np.linalg.norm(obs_c, "fro")) / math.sqrt(max(obs.shape[0], 1)), 1e-12)
        Xn = np.zeros_like(X_full)
        Xn[mask_m] = obs_c / s_m
        mu[m] = mu_m
        rms_scale[m] = s_m
        X_norm[m] = Xn
        masks[m] = mask_m

    row_counts = sum(masks[m].astype(np.float64) for m in models)[:, None]

    X_cat = np.concatenate([X_norm[m] for m in models], axis=1)
    U_init, _, _ = np.linalg.svd(X_cat, full_matrices=False)
    G = U_init[:, :k_eff].copy()
    G /= max(float(np.linalg.norm(G, "fro")), 1e-12)

    Q: Dict[str, np.ndarray] = {}
    obj_hist: List[float] = []
    converged = False

    for it in range(max_iter):
        S_sum = np.zeros((n_active, k_eff), dtype=np.float64)
        for m in models:
            idx = masks[m]
            C_m = X_norm[m][idx].T @ G[idx]
            U_m, _, Vt_m = np.linalg.svd(C_m, full_matrices=False)
            Q_m = U_m @ Vt_m
            Q[m] = Q_m
            S_sum[idx] += X_norm[m][idx] @ Q_m

        G_raw = S_sum / np.maximum(row_counts, 1.0)
        phi = float(np.sum(G_raw * S_sum) / max(float(np.linalg.norm(G_raw, "fro")), 1e-12))
        obj_hist.append(phi)
        G_new = G_raw / max(float(np.linalg.norm(G_raw, "fro")), 1e-12)
        if it > 1 and abs(obj_hist[-1] - obj_hist[-2]) / max(abs(obj_hist[-2]), 1e-12) < tol:
            G = G_new
            converged = True
            break
        G = G_new

    consensus = sum(X_norm[m] @ Q[m] for m in models) / np.maximum(row_counts, 1.0)
    return GPASpace(
        models=models,
        k=k_eff,
        mu=mu,
        rms_scale=rms_scale,
        Q=Q,
        consensus=consensus,
        objective_history=obj_hist,
        converged=converged,
    )


def fit_spectral_orthogonal_sync(
    pairwise_R: Dict[Tuple[str, str], np.ndarray],
    models: Sequence[str],
    weights: Optional[Dict[Tuple[str, str], float]] = None,
) -> Dict[str, np.ndarray]:
    """Spectral O(k) Synchronization from pairwise k x k orthogonal rotations R_{ij}."""
    models = list(models)
    L = len(models)
    sample_R = next(iter(pairwise_R.values()))
    k = sample_R.shape[0]
    H = np.zeros((L * k, L * k), dtype=np.float64)
    idx = {m: i for i, m in enumerate(models)}

    for (mi, mj), Rij in pairwise_R.items():
        if mi == mj:
            continue
        i, j = idx[mi], idx[mj]
        w = 1.0 if weights is None else float(weights.get((mi, mj), 1.0))
        H[i * k : (i + 1) * k, j * k : (j + 1) * k] += w * Rij
        H[j * k : (j + 1) * k, i * k : (i + 1) * k] += w * Rij.T

    H = 0.5 * (H + H.T)
    _, eigvecs = np.linalg.eigh(H)
    V_top = eigvecs[:, -k:]

    Q_sync: Dict[str, np.ndarray] = {}
    for m, i in idx.items():
        block = V_top[i * k : (i + 1) * k, :]
        U, _, Vt = np.linalg.svd(block, full_matrices=False)
        Q_sync[m] = U @ Vt
    return Q_sync


def compute_closed_loop_holonomy(
    W_ab: np.ndarray,
    W_bc: np.ndarray,
    W_ca: np.ndarray,
    X_a_train: np.ndarray,
    X_a_test: Optional[np.ndarray] = None,
    ks: Sequence[int] = (16, 32, 64, 128, 256),
) -> Dict[str, Any]:
    """Measure closed-loop holonomy ||R_{CA} R_{BC} R_{AB} - I||_F on model A's factual subspace."""
    L_op = (
        np.asarray(W_ab, dtype=np.float64)
        @ np.asarray(W_bc, dtype=np.float64)
        @ np.asarray(W_ca, dtype=np.float64)
    )
    X_tr = np.asarray(X_a_train, dtype=np.float64)
    mu_a = X_tr.mean(axis=0)
    Xc_tr = X_tr - mu_a
    _, _, Vt = np.linalg.svd(Xc_tr, full_matrices=False)
    U_pca = Vt.T
    max_r = U_pca.shape[1]

    subspace_metrics: Dict[str, Dict[str, float]] = {}
    for k in ks:
        if k > max_r:
            continue
        Uk = U_pca[:, :k]
        diff_full = L_op @ Uk - Uk
        hol_full = float(np.linalg.norm(diff_full, "fro"))
        M_k = Uk.T @ L_op @ Uk
        hol_proj = float(np.linalg.norm(M_k - np.eye(k), "fro"))
        mean_diag_cos = float(np.trace(M_k) / k)
        diag_cos = np.clip(np.diag(M_k), -1.0, 1.0)
        mean_angle_deg = float(np.degrees(np.mean(np.arccos(diag_cos))))
        subspace_metrics[str(k)] = {
            "holonomy_full_frob": hol_full,
            "holonomy_full_norm": hol_full / math.sqrt(k),
            "holonomy_proj_frob": hol_proj,
            "holonomy_proj_norm": hol_proj / math.sqrt(k),
            "mean_basis_cos": mean_diag_cos,
            "mean_angle_deg": mean_angle_deg,
        }

    result: Dict[str, Any] = {"subspace": subspace_metrics}
    if X_a_test is not None and len(X_a_test) > 0:
        Xc_te = np.asarray(X_a_test, dtype=np.float64) - mu_a
        Xc_loop = Xc_te @ L_op
        rel_err = float(np.linalg.norm(Xc_loop - Xc_te, "fro") / max(float(np.linalg.norm(Xc_te, "fro")), 1e-12))
        n1 = np.linalg.norm(Xc_te, axis=1)
        n2 = np.linalg.norm(Xc_loop, axis=1)
        cos_vals = np.sum(Xc_te * Xc_loop, axis=1) / np.maximum(n1 * n2, 1e-12)
        norm_ratio = float(np.mean(n2 / np.maximum(n1, 1e-12)))
        result["empirical_test"] = {
            "rel_frob_err": rel_err,
            "mean_cos": float(np.mean(cos_vals)),
            "median_cos": float(np.median(cos_vals)),
            "norm_retention": norm_ratio,
        }
    return result


# =============================================================================
# 4. Pillar 2: Architecture-Agnostic PyTorch Multi-Regime Patching Engine
# =============================================================================


def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise AttributeError(f"Unsupported model architecture for layer introspection: {type(model)}")


def get_mlp_module(layer: nn.Module) -> nn.Module:
    if hasattr(layer, "post_feedforward_layernorm"):
        return layer.post_feedforward_layernorm
    if hasattr(layer, "mlp"):
        return layer.mlp
    if hasattr(layer, "feed_forward"):
        return layer.feed_forward
    raise AttributeError(f"Unsupported layer MLP structure: {type(layer)}")


@dataclass
class BatchPatchSpec:
    residual_replace: Dict[int, List[Tuple[List[int], torch.Tensor]]] = field(default_factory=dict)
    residual_add: Dict[int, List[Tuple[List[int], torch.Tensor]]] = field(default_factory=dict)
    mlp_replace: Dict[int, List[Tuple[List[int], torch.Tensor]]] = field(default_factory=dict)


class PatchHookManager:
    """Context manager attaching clean prefill-only forward hooks for arbitrary patch regimes."""

    def __init__(self, model: nn.Module, spec: BatchPatchSpec):
        self.model = model
        self.spec = spec
        self.handles: List[Any] = []
        self._fired: Dict[Tuple[str, int], bool] = {}

    def _patch_tensor(
        self,
        tensor: torch.Tensor,
        ops_replace: List[Tuple[List[int], torch.Tensor]],
        ops_add: List[Tuple[List[int], torch.Tensor]],
    ) -> torch.Tensor:
        out = tensor.clone()
        bsz = out.shape[0]
        seq_len = out.shape[1]
        for positions, target in ops_replace:
            tgt = target.to(device=out.device, dtype=out.dtype)
            for b in range(bsz):
                pos = positions[b]
                if 0 <= pos < seq_len:
                    out[b, pos, :] = tgt[b]
        for positions, delta in ops_add:
            dlt = delta.to(device=out.device, dtype=out.dtype)
            for b in range(bsz):
                pos = positions[b]
                if 0 <= pos < seq_len:
                    out[b, pos, :] = out[b, pos, :] + dlt[b]
        return out

    def __enter__(self) -> "PatchHookManager":
        layers = get_transformer_layers(self.model)
        self._fired.clear()

        res_layers = sorted(set(self.spec.residual_replace.keys()) | set(self.spec.residual_add.keys()))
        for l_idx in res_layers:
            ops_rep = self.spec.residual_replace.get(l_idx, [])
            ops_add = self.spec.residual_add.get(l_idx, [])

            def make_res_hook(layer_id: int, rep_list, add_list):
                def hook(_module, _inp, output):
                    key = ("res", layer_id)
                    if self._fired.get(key, False):
                        return output
                    self._fired[key] = True
                    if isinstance(output, tuple):
                        h = output[0]
                        if h.ndim == 3:
                            h_new = self._patch_tensor(h, rep_list, add_list)
                            return (h_new,) + output[1:]
                        return output
                    else:
                        if output.ndim == 3:
                            return self._patch_tensor(output, rep_list, add_list)
                        return output

                return hook

            self.handles.append(layers[l_idx].register_forward_hook(make_res_hook(l_idx, ops_rep, ops_add)))

        for l_idx, ops_rep in sorted(self.spec.mlp_replace.items()):
            mlp_mod = get_mlp_module(layers[l_idx])

            def make_mlp_hook(layer_id: int, rep_list):
                def hook(_module, _inp, output):
                    key = ("mlp", layer_id)
                    if self._fired.get(key, False):
                        return output
                    self._fired[key] = True
                    if isinstance(output, tuple):
                        h = output[0]
                        if h.ndim == 3:
                            h_new = self._patch_tensor(h, rep_list, [])
                            return (h_new,) + output[1:]
                        return output
                    else:
                        if output.ndim == 3:
                            return self._patch_tensor(output, rep_list, [])
                        return output

                return hook

            self.handles.append(mlp_mod.register_forward_hook(make_mlp_hook(l_idx, ops_rep)))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


@torch.no_grad()
def extract_mlp_outputs_batched(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_positions: Sequence[int],
    layer_indices: Sequence[int],
    batch_size: int = 16,
    device: str = "cpu",
) -> Dict[int, np.ndarray]:
    layers = get_transformer_layers(model)
    N = input_ids.shape[0]
    layer_indices = sorted(set(int(l) for l in layer_indices))
    captured: Dict[int, List[np.ndarray]] = {l: [] for l in layer_indices}

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ids_b = input_ids[start:end].to(device)
        mask_b = attention_mask[start:end].to(device)
        pos_b = list(token_positions[start:end])

        batch_buf: Dict[int, np.ndarray] = {}
        handles = []
        for l_idx in layer_indices:
            mlp_mod = get_mlp_module(layers[l_idx])

            def make_cap(lid: int):
                def hook(_mod, _inp, output):
                    h = output[0] if isinstance(output, tuple) else output
                    vecs = torch.stack([h[b, pos_b[b], :] for b in range(h.shape[0])], dim=0)
                    batch_buf[lid] = vecs.detach().float().cpu().numpy()

                return hook

            handles.append(mlp_mod.register_forward_hook(make_cap(l_idx)))

        try:
            model(input_ids=ids_b, attention_mask=mask_b, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        for l_idx in layer_indices:
            captured[l_idx].append(batch_buf[l_idx])

    return {l_idx: np.concatenate(captured[l_idx], axis=0) for l_idx in layer_indices}


@torch.no_grad()
def extract_residual_states_batched(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_positions: Sequence[int],
    layer_indices: Sequence[int],
    batch_size: int = 16,
    device: str = "cpu",
) -> Dict[int, np.ndarray]:
    N = input_ids.shape[0]
    layer_indices = sorted(set(int(l) for l in layer_indices))
    captured: Dict[int, List[np.ndarray]] = {l: [] for l in layer_indices}

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ids_b = input_ids[start:end].to(device)
        mask_b = attention_mask[start:end].to(device)
        pos_b = list(token_positions[start:end])

        out = model(input_ids=ids_b, attention_mask=mask_b, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        for l_idx in layer_indices:
            h = hs[l_idx + 1]
            vecs = torch.stack([h[b, pos_b[b], :] for b in range(h.shape[0])], dim=0)
            captured[l_idx].append(vecs.detach().float().cpu().numpy())

    return {l_idx: np.concatenate(captured[l_idx], axis=0) for l_idx in layer_indices}


@torch.no_grad()
def greedy_generate_patched(
    model: nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    patch_spec: Optional[BatchPatchSpec] = None,
    max_new_tokens: int = 8,
    device: str = "cpu",
) -> List[str]:
    ids = input_ids.to(device)
    mask = attention_mask.to(device)
    bsz, _ = ids.shape

    if patch_spec is not None:
        with PatchHookManager(model, patch_spec):
            out = model(input_ids=ids, attention_mask=mask, use_cache=True)
    else:
        out = model(input_ids=ids, attention_mask=mask, use_cache=True)

    past = out.past_key_values
    next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    generated = [next_tok]

    cur_mask = mask
    for _ in range(max_new_tokens - 1):
        cur_mask = torch.cat([cur_mask, torch.ones((bsz, 1), dtype=cur_mask.dtype, device=device)], dim=1)
        out = model(input_ids=next_tok, attention_mask=cur_mask, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(next_tok)

    gen_ids = torch.cat(generated, dim=1).cpu()
    return [tokenizer.decode(gen_ids[b], skip_special_tokens=True) for b in range(bsz)]


def tokenize_left_padded_with_positions(
    tokenizer: Any,
    prompts: Sequence[str],
    char_spans: Sequence[Tuple[int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_ids: List[List[int]] = []
    rel_subj_pos: List[int] = []

    for prompt, (_start, end) in zip(prompts, char_spans):
        full_ids = tokenizer.encode(prompt, add_special_tokens=True)
        prefix = prompt[:end]
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
        subj_idx = min(max(len(prefix_ids) - 1, 0), len(full_ids) - 1)
        all_ids.append(full_ids)
        rel_subj_pos.append(subj_idx)

    max_len = max(len(ids) for ids in all_ids)
    pad_id = tokenizer.pad_token_id
    B = len(all_ids)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    last_subj_pos: List[int] = []
    last_prompt_pos: List[int] = []

    for b, ids in enumerate(all_ids):
        L = len(ids)
        pad_len = max_len - L
        input_ids[b, pad_len:] = torch.tensor(ids, dtype=torch.long)
        attention_mask[b, pad_len:] = 1
        last_subj_pos.append(pad_len + rel_subj_pos[b])
        last_prompt_pos.append(max_len - 1)

    tokenizer.padding_side = orig_padding_side
    return input_ids, attention_mask, last_subj_pos, last_prompt_pos


def match_answer(
    generated: str,
    target: str,
    aliases: Optional[Sequence[str]] = None,
    relation: str = "",
) -> bool:
    """Clean answer match with accent folding & element_symbol case sensitivity (exact parity with extract_and_baseline)."""
    import unicodedata

    def _strip_accents(s: str) -> str:
        s = s.replace("ł", "l").replace("Ł", "L")
        return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

    cleaned = _strip_accents(generated).lstrip(" \t\n\"'“”(")
    if not cleaned:
        return False
    cleaned_low = cleaned.lower()
    case_sensitive = relation == "element_symbol"
    cands = [target] + list(aliases or [])
    for cand in cands:
        if not cand:
            continue
        c_norm = _strip_accents(cand.strip())
        if not c_norm:
            continue
        if case_sensitive:
            if cleaned.startswith(c_norm):
                rest = cleaned[len(c_norm) :]
                if not rest or not rest[0].isalnum():
                    return True
        else:
            cl = c_norm.lower()
            if cleaned_low.startswith(cl):
                rest = cleaned_low[len(cl) :]
                if not rest or not rest[0].isalnum():
                    return True
    return False


# =============================================================================
# 5. Pillar 3 & 4: Statistical, Dark-Memory, and 2-Hop Audit Helpers
# =============================================================================


def exact_mcnemar(outcomes_a: Sequence[int], outcomes_b: Sequence[int]) -> Dict[str, Any]:
    a = np.asarray(outcomes_a, dtype=np.int64)
    b = np.asarray(outcomes_b, dtype=np.int64)
    assert a.shape == b.shape and a.ndim == 1
    n_a_only = int(np.sum((a == 1) & (b == 0)))
    n_b_only = int(np.sum((a == 0) & (b == 1)))
    n_disc = n_a_only + n_b_only
    pval = 1.0 if n_disc == 0 else float(binomtest(min(n_a_only, n_b_only), n_disc, 0.5, alternative="two-sided").pvalue)
    return {
        "n": int(a.size),
        "acc_a": float(a.mean()) if a.size else 0.0,
        "acc_b": float(b.mean()) if b.size else 0.0,
        "diff_b_minus_a": float(b.mean() - a.mean()) if a.size else 0.0,
        "n_a_only": n_a_only,
        "n_b_only": n_b_only,
        "p_value": pval,
    }


def summarize_dark_memory_outcomes(
    recovered_dark: Sequence[int],
    recovered_nondark: Sequence[int],
    retained_recalled: Sequence[int],
    same_model_ceiling: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    rd = np.asarray(recovered_dark, dtype=np.int64)
    rn = np.asarray(recovered_nondark, dtype=np.int64)
    rr = np.asarray(retained_recalled, dtype=np.int64)

    n_dark = int(rd.size)
    n_nondark = int(rn.size)
    n_rec = int(rr.size)
    n_unlocked = int(rd.sum())
    n_broken = int((rr == 0).sum())
    total_tested = n_dark + n_nondark + n_rec

    summary: Dict[str, Any] = {
        "n_dark": n_dark,
        "dark_recovery_count": n_unlocked,
        "dark_recovery_rate": float(rd.mean()) if n_dark else 0.0,
        "n_nondark": n_nondark,
        "nondark_recovery_rate": float(rn.mean()) if n_nondark else 0.0,
        "n_already_recalled": n_rec,
        "retention_rate": float(rr.mean()) if n_rec else 0.0,
        "collateral_damage_rate": float(1.0 - rr.mean()) if n_rec else 0.0,
        "collateral_damage_count": n_broken,
        "net_recall_gain_count": n_unlocked + int(rn.sum()) - n_broken,
        "net_recall_gain_rate": float((n_unlocked + int(rn.sum()) - n_broken) / max(total_tested, 1)),
    }
    if same_model_ceiling is not None and len(same_model_ceiling) > 0:
        smc = np.asarray(same_model_ceiling, dtype=np.int64)
        ceil_rate = float(smc.mean())
        summary["same_model_ceiling_rate"] = ceil_rate
        summary["pct_of_same_model_ceiling"] = summary["dark_recovery_rate"] / ceil_rate if ceil_rate > 1e-12 else 0.0
    return summary


def map_layer_window(
    l_a_star: int,
    l_b_star: int,
    n_layers_a: int,
    n_layers_b: int,
    offsets: Sequence[int],
    mapping_mode: str = "proportional",
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    seen_b = set()
    ratio = float(n_layers_a) / max(float(n_layers_b), 1.0)
    for off in offsets:
        lb = int(np.clip(l_b_star + off, 0, n_layers_b - 1))
        if lb in seen_b:
            continue
        seen_b.add(lb)
        if mapping_mode == "proportional":
            la = int(np.clip(round(l_a_star + off * ratio), 0, n_layers_a - 1))
        else:
            la = int(np.clip(l_a_star + off, 0, n_layers_a - 1))
        pairs.append((la, lb))
    return pairs


# =============================================================================
# 6. Synthetic Verification Suite
# =============================================================================


class _TinyDummyBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.attn = nn.Linear(d, d, bias=False)
        self.mlp = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor):
        h = x + self.attn(x)
        h = h + self.mlp(h)
        return (h,)


class _TinyDummyModel(nn.Module):
    def __init__(self, vocab: int = 32, d: int = 16, n_layers: int = 4):
        super().__init__()

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab, d)
                self.layers = nn.ModuleList([_TinyDummyBlock(d) for _ in range(n_layers)])

        self.model = _Inner()
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False, past_key_values=None):
        h = self.model.embed(input_ids)
        hs = [h]
        for layer in self.model.layers:
            h = layer(h)[0]
            hs.append(h)
        logits = self.lm_head(h)

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        out.hidden_states = tuple(hs) if output_hidden_states else None
        out.past_key_values = past_key_values
        return out


def run_synthetic_tests() -> Dict[str, Any]:
    rng = np.random.default_rng(12345)
    report: Dict[str, Any] = {}

    N_tr, N_te = 400, 200
    k_true = 32
    dims = {"A1": 48, "A2": 64, "A3": 56, "B": 40}
    Z_true_tr = rng.standard_normal((N_tr, k_true))
    Z_true_te = rng.standard_normal((N_te, k_true))

    Q_true = {}
    X_tr, X_te = {}, {}
    shared_err_tr = rng.standard_normal((N_tr, k_true)) * 0.15
    shared_err_te = rng.standard_normal((N_te, k_true)) * 0.15
    noise_scales = {"A1": 0.35, "A2": 0.40, "A3": 0.55, "B": 0.05}
    for name, d in dims.items():
        G = rng.standard_normal((k_true, d))
        U, _, Vt = np.linalg.svd(G, full_matrices=False)
        Qm = U @ Vt
        Q_true[name] = Qm
        mu = rng.standard_normal(d) * 0.5
        sc = rng.uniform(0.7, 1.8)
        e_tr = shared_err_tr + rng.standard_normal((N_tr, k_true)) * noise_scales[name]
        e_te = shared_err_te + rng.standard_normal((N_te, k_true)) * noise_scales[name]
        X_tr[name] = mu + sc * ((Z_true_tr + e_tr) @ Qm)
        X_te[name] = mu + sc * ((Z_true_te + e_te) @ Qm)

    donors = ["A1", "A2", "A3"]
    pmaps = {m: fit_procrustes(X_tr[m], X_tr["B"]) for m in donors}
    for m, pm in pmaps.items():
        assert np.allclose(pm.W.T @ pm.W, np.eye(dims["B"]), atol=1e-10)

    mapped_tr = [pmaps[m].transform(X_tr[m]) for m in donors]
    mapped_te = [pmaps[m].transform(X_te[m]) for m in donors]
    mu_list = [pmaps[m].mu_b for m in donors]
    single_mses = [float(np.mean(np.sum((Xh - X_te["B"]) ** 2, axis=1))) for Xh in mapped_te]

    cfg_unif = fit_barycenter_config(donors, mapped_tr, mu_list, X_tr["B"], weight_method="uniform", norm_mode="rms_norm")
    cfg_gls = fit_barycenter_config(donors, mapped_tr, mu_list, X_tr["B"], weight_method="inv_cov", norm_mode="rms_norm")
    cfg_none = fit_barycenter_config(donors, mapped_tr, mu_list, X_tr["B"], weight_method="uniform", norm_mode="none")

    Xb_unif, diag_unif = aggregate_multi_donor_keys(mapped_te, mu_list, cfg_unif)
    Xb_gls, _ = aggregate_multi_donor_keys(mapped_te, mu_list, cfg_gls)
    Xb_none, _ = aggregate_multi_donor_keys(mapped_te, mu_list, cfg_none)

    mse_unif = float(np.mean(np.sum((Xb_unif - X_te["B"]) ** 2, axis=1)))
    mse_gls = float(np.mean(np.sum((Xb_gls - X_te["B"]) ** 2, axis=1)))
    mse_none = float(np.mean(np.sum((Xb_none - X_te["B"]) ** 2, axis=1)))

    assert cfg_gls.weights[0] > cfg_gls.weights[2]
    assert mse_none < min(single_mses)
    assert mse_gls < mse_unif + 1e-6
    assert diag_unif["mean_norm_attenuation_ratio"] < 0.99
    assert diag_unif["effective_norm_boost"] * diag_unif["mean_norm_attenuation_ratio"] >= 1.0 - 1e-9
    restored_norms = np.linalg.norm(Xb_unif - sum(cfg_unif.weights[m] * mu_list[m] for m in range(3))[None, :], axis=1)
    target_rms = np.sqrt(
        sum(
            cfg_unif.weights[m] * np.sum((mapped_te[m] - mu_list[m][None, :]) ** 2, axis=1)
            for m in range(3)
        )
    )
    assert np.allclose(restored_norms, target_rms, rtol=1e-6, atol=1e-6)

    report["test1_barycenter"] = {
        "single_mses": single_mses,
        "mse_bary_unrescaled": mse_none,
        "mse_bary_rms_restored": mse_unif,
        "mse_bary_gls_restored": mse_gls,
        "gls_weights": cfg_gls.weights.tolist(),
        "mean_offdiag_rho": cfg_gls.mean_offdiag_rho,
        "theoretical_snr_gain": cfg_gls.theoretical_snr_gain,
        "empirical_snr_gain": cfg_gls.empirical_snr_gain,
        "attenuation_ratio": diag_unif["mean_norm_attenuation_ratio"],
    }

    gpa = fit_gpa_platonic_space(X_tr, k=k_true, max_iter=100)
    diffs = np.diff(gpa.objective_history)
    assert np.all(diffs >= -1e-9)
    for m in gpa.models:
        assert np.allclose(gpa.Q[m].T @ gpa.Q[m], np.eye(k_true), atol=1e-10)

    L_gpa = (gpa.Q["A1"] @ gpa.Q["A2"].T) @ (gpa.Q["A2"] @ gpa.Q["A3"].T) @ (gpa.Q["A3"] @ gpa.Q["A1"].T)
    hol_gpa_on_Q = float(np.linalg.norm(L_gpa @ gpa.Q["A1"] - gpa.Q["A1"], "fro"))
    assert hol_gpa_on_Q < 1e-10

    W_12 = fit_procrustes(X_tr["A1"], X_tr["A2"]).W
    W_23 = fit_procrustes(X_tr["A2"], X_tr["A3"]).W
    W_31 = fit_procrustes(X_tr["A3"], X_tr["A1"]).W
    hol_pair = compute_closed_loop_holonomy(W_12, W_23, W_31, X_tr["A1"], X_te["A1"], ks=(8, 16, 32))
    W_r1 = fit_random_orthogonal_control(X_tr["A1"], X_tr["A2"], seed=1).W
    W_r2 = fit_random_orthogonal_control(X_tr["A2"], X_tr["A3"], seed=2).W
    W_r3 = fit_random_orthogonal_control(X_tr["A3"], X_tr["A1"], seed=3).W
    hol_rand = compute_closed_loop_holonomy(W_r1, W_r2, W_r3, X_tr["A1"], X_te["A1"], ks=(8, 16, 32))
    assert hol_pair["subspace"]["16"]["holonomy_full_norm"] < hol_rand["subspace"]["16"]["holonomy_full_norm"] * 0.5
    assert hol_pair["empirical_test"]["mean_cos"] > 0.75

    masks = {
        "A1": np.arange(N_tr) < 210,
        "A3": np.arange(N_tr) >= 190,
        "A2": np.ones(N_tr, dtype=bool),
        "B": np.ones(N_tr, dtype=bool),
    }
    direct_idx = masks["A1"] & masks["A3"]
    assert int(direct_idx.sum()) == 20
    pm_direct_low = fit_procrustes(X_tr["A1"][direct_idx], X_tr["A3"][direct_idx])
    gpa_partial = fit_gpa_partial_overlap(X_tr, masks, k=k_true)
    pm_gpa_low = gpa_partial.get_pairwise_map("A1", "A3", X_tr["A1"][masks["A1"]], X_tr["A3"][masks["A3"]])

    mse_direct_low = float(np.mean(np.sum((pm_direct_low.transform(X_te["A1"]) - X_te["A3"]) ** 2, axis=1)))
    mse_gpa_low = float(np.mean(np.sum((pm_gpa_low.transform(X_te["A1"]) - X_te["A3"]) ** 2, axis=1)))
    assert mse_gpa_low < mse_direct_low

    sub_R = {
        ("A1", "A2"): gpa.Q["A1"].T @ W_12 @ gpa.Q["A2"],
        ("A2", "A3"): gpa.Q["A2"].T @ W_23 @ gpa.Q["A3"],
        ("A3", "A1"): gpa.Q["A3"].T @ W_31 @ gpa.Q["A1"],
    }
    Q_sync = fit_spectral_orthogonal_sync(sub_R, ["A1", "A2", "A3"])
    for m, Qm in Q_sync.items():
        assert np.allclose(Qm.T @ Qm, np.eye(k_true), atol=1e-10)

    report["test2_gpa_holonomy"] = {
        "gpa_iters": len(gpa.objective_history),
        "gpa_converged": gpa.converged,
        "gpa_exact_holonomy_on_Q": hol_gpa_on_Q,
        "pairwise_holonomy_k16": hol_pair["subspace"]["16"]["holonomy_full_norm"],
        "random_holonomy_k16": hol_rand["subspace"]["16"]["holonomy_full_norm"],
        "pairwise_emp_cos": hol_pair["empirical_test"]["mean_cos"],
        "low_overlap_20_direct_mse": mse_direct_low,
        "low_overlap_20_gpa_sync_mse": mse_gpa_low,
    }

    torch.manual_seed(0)
    dummy = _TinyDummyModel(vocab=32, d=16, n_layers=4)
    dummy.eval()
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.long)
    mask = torch.ones_like(ids)

    res_states = extract_residual_states_batched(dummy, ids, mask, [2, 3], [0, 1, 2, 3])
    mlp_states = extract_mlp_outputs_batched(dummy, ids, mask, [2, 3], [0, 1, 2, 3])
    assert res_states[1].shape == (2, 16) and mlp_states[1].shape == (2, 16)

    base_logits = dummy(ids, attention_mask=mask).logits
    spec_id_res = BatchPatchSpec(
        residual_replace={
            1: [([2, 3], torch.tensor(res_states[1]))],
            2: [([2, 3], torch.tensor(res_states[2]))],
        }
    )
    with PatchHookManager(dummy, spec_id_res):
        patched_id_logits = dummy(ids, attention_mask=mask).logits
    assert torch.allclose(base_logits, patched_id_logits, atol=1e-5)

    spec_id_mlp = BatchPatchSpec(mlp_replace={1: [([2, 3], torch.tensor(mlp_states[1]))]})
    with PatchHookManager(dummy, spec_id_mlp):
        patched_mlp_logits = dummy(ids, attention_mask=mask).logits
    assert torch.allclose(base_logits, patched_mlp_logits, atol=1e-5)

    delta = torch.ones((2, 16)) * 5.0
    spec_delta = BatchPatchSpec(residual_add={1: [([2, 3], delta)]})
    with PatchHookManager(dummy, spec_delta):
        out_delta = dummy(ids, attention_mask=mask, output_hidden_states=True)
    h1_base = dummy(ids, attention_mask=mask, output_hidden_states=True).hidden_states[2]
    h1_delta = out_delta.hidden_states[2]
    assert torch.allclose(h1_delta[0, 2] - h1_base[0, 2], delta[0], atol=1e-5)
    assert torch.allclose(h1_delta[0, 1] - h1_base[0, 1], torch.zeros(16), atol=1e-6)

    mc = exact_mcnemar([1, 0, 0, 0, 0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert mc["n_b_only"] == 7 and mc["n_a_only"] == 0 and mc["p_value"] < 0.02
    dm = summarize_dark_memory_outcomes([1, 1, 0, 0], [0, 0], [1, 1, 1, 0], same_model_ceiling=[1, 1, 1, 0])
    assert abs(dm["dark_recovery_rate"] - 0.5) < 1e-9
    assert abs(dm["collateral_damage_rate"] - 0.25) < 1e-9
    assert abs(dm["pct_of_same_model_ceiling"] - (0.5 / 0.75)) < 1e-9

    report["test3_hooks_and_stats"] = {
        "identity_res_max_err": float((base_logits - patched_id_logits).abs().max().item()),
        "identity_mlp_max_err": float((base_logits - patched_mlp_logits).abs().max().item()),
        "mcnemar_p": mc["p_value"],
        "dark_summary": dm,
    }
    report["all_passed"] = True
    return report


if __name__ == "__main__":
    import json

    res = run_synthetic_tests()
    print(json.dumps(res, indent=2))
