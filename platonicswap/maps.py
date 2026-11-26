"""
maps.py — Closed-Form Cross-Model Representation Alignment & Layer Selection Library
====================================================================================

Implements BOTH the Plato's Key (Cheng et al., Hazy Research 2026) baseline maps AND
closed-form geometric, low-dimensional compression, and zero-intervention layer-selection
improvements (strictly closed-form; zero gradient descent):

1. Baselines & Controls:
   - `ProcrustesGlobalMap` (`procrustes_global`):
       z_hat = mu_B + s * R (z_A - mu_A),  R = U V^T from SVD(Z_B Z_A^T),  s = ||Z_B||_F / ||Z_A||_F.
   - `ProcrustesLSMap` (`procrustes_ls`):
       Least-squares scalar scale s_LS = tr(R^T Z_B Z_A^T) / ||R Z_A||_F^2.
   - `RidgeMap` (`ridge_raw`, `ridge_rescaled`):
       Fast economy-SVD ridge W(lam) = Z_B Z_A^T (Z_A Z_A^T + lam I)^{-1}, with or without
       spread restoration s = ||Z_B||_F / ||W Z_A||_F.
   - `VecMapWhitenedMap` (`vecmap_whitened`):
       Whitened orthogonal alignment Q_VM = polar(Sigma_{B,lam}^{-1/2} Z_B Z_A^T Sigma_{A,lam}^{-1/2})
       mapped back via Sigma_{B,lam}^{+1/2} Q_VM Sigma_{A,lam}^{-1/2} (note -1/2 on Sigma_B in polar).
   - `RandomControlMap` & `ShuffledProcrustesMap`:
       Haar-random semi-orthogonal rotation and row-shuffled paired Procrustes controls.

2. Proposed Mapping Geometry Improvements:
   - `MongeProcrustesMap` (`monge_procrustes`):
       Exact Covariance-Preserving Paired Optimal Map solving
         min_{T Sigma_A T^T = Sigma_B} ||Z_B - T Z_A||_F^2
       in closed form via T* = Sigma_{B,lam_B}^{+1/2} Q* Sigma_{A,lam_A}^{-1/2} where
         Q* = polar(Sigma_{B,lam_B}^{+1/2} Z_B Z_A^T Sigma_{A,lam_A}^{-1/2})
       (crucial +1/2 exponent on Sigma_B inside polar factor vs. VecMap's -1/2), computed in
       O(N^2 d) via exact Woodbury / economy-SVD shrinkage covariance powers.
   - `RMSNormWeightedProcrustesMap` (`rmsnorm_weighted_procrustes`):
       Aligns keys under recipient B's frozen downstream Pre-RMSNorm gain metric gamma_B
       (both orthogonal readout-weighted polar(M_B Z_B Z_A^T) and post-RMSNorm similarity
       D_B^{-1} polar(D_B Z_B (D_A Z_A)^T) D_A).
   - `ReadoutWeightedProcrustesMap` (`readout_weighted_procrustes`):
       Weights Procrustes alignment by recipient B's frozen downstream MLP/Attention input
       operator Gram metric M_B = (1 - eta) I + eta G_readout.
   - `PerSampleNormGainMap` (`per_sample_norm_and_gain`):
       Per-sample recipient norm matching (centered ||z_hat - mu_B||_2 = ||z_B_orig - mu_B||_2
       or total ||z_hat||_2 = ||z_B_orig||_2) + validation-tuned gain alpha to overcome
       downstream residual leakage in Pre-RMSNorm residual streams.
   - `RelationCenteredProcrustesMap` (`relation_centered_procrustes`):
       Fits semi-orthogonal R on within-relation residuals (z_i - mu_{rel(i)}) so top singular
       modes align entity variations rather than surface prompt-template centroids; supports
       `test_mode="global_mean"` (zero relation label needed at test time; LORO-compatible)
       and `test_mode="relation_mean"`.

3. Low-Dimensional Compression & Surgical Subspace Injection:
   - `SeparatePCAProcrustesMap` (`pca_procrustes`):
       Plato's Key Fig. 4b baseline (separate top-k PCA per model + k x k Procrustes), with
       both `fill_mode="mean"` and `fill_mode="surgical"`.
   - `PLSProcrustesMap` (`pls_procrustes`):
       Cross-Covariance SVD rank-k compression truncating Z_B Z_A^T = U Sigma V^T to rank k,
       with `fill_mode="mean"`, `fill_mode="surgical"`
         z_hat = (I - U_k U_k^T) z_B + U_k (mu_Bk + s_k V_k^T (z_A - mu_A)),
       and `fill_mode="soft_surgical"` interpolating orthogonal-complement retention (1 - beta)
       with exact Pythagorean spread preservation for all beta in [0, 1].

4. Zero-Intervention Layer Selection Proxies (`compute_layer_selection_grids`):
   - `raw_mknn` (PRH / Plato's Key baseline)
   - `within_relation_mknn` (removes prompt-template clustering)
   - `paraphrase_entity_mknn` & `within_rel_paraphrase_mknn` (cross-template entity invariance)
   - `linear_cka` & `within_relation_cka`
   - `paraphrase_procrustes_cosine` (split-half cross-template Procrustes cosine)

Row-major convention:
  All public methods take sample matrices `X_A` of shape `(N, d_A)` and `X_B` of shape `(N, d_B)`
  (or 1D single vectors `(d_A,)`). Internally, centered column matrices match Plato's Key notation:
  `Z_A = (X_A - mu_A).T` of shape `(d_A, N)` and `Z_B = (X_B - mu_B).T` of shape `(d_B, N)`.
  All linear-algebra factorizations run in IEEE float64 for machine precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg as sla

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

ArrayLike = Union[np.ndarray, "torch.Tensor"]
EPS = 1e-12


# ---------------------------------------------------------------------------
# Tensor / Array Conversion Helpers (Float64 math, preserve caller container)
# ---------------------------------------------------------------------------

@dataclass
class _InputMeta:
    is_torch: bool
    dtype: Any
    device: Any
    was_1d: bool


def _to_f64_2d(x: ArrayLike) -> Tuple[np.ndarray, _InputMeta]:
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        was_1d = (x.ndim == 1)
        arr = x.detach().cpu().to(torch.float64).numpy()
        if was_1d:
            arr = arr[None, :]
        return np.ascontiguousarray(arr, dtype=np.float64), _InputMeta(
            is_torch=True, dtype=x.dtype, device=x.device, was_1d=was_1d
        )
    arr = np.asarray(x, dtype=np.float64)
    was_1d = (arr.ndim == 1)
    if was_1d:
        arr = arr[None, :]
    return np.ascontiguousarray(arr, dtype=np.float64), _InputMeta(
        is_torch=False, dtype=np.asarray(x).dtype, device=None, was_1d=was_1d
    )


def _from_f64_2d(arr: np.ndarray, meta: _InputMeta) -> ArrayLike:
    out = arr[0] if meta.was_1d else arr
    if meta.is_torch:
        t = torch.from_numpy(np.ascontiguousarray(out))
        return t.to(device=meta.device, dtype=meta.dtype)
    if np.issubdtype(meta.dtype, np.floating):
        return out.astype(meta.dtype, copy=False)
    return out


def _polar_factor(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute thin SVD M = U diag(s) Vt and semi-orthogonal polar factor Q = U @ Vt.
    For M of shape (d_B, d_A) with p = min(d_B, d_A):
      U: (d_B, p), s: (p,), Vt: (p, d_A), Q: (d_B, d_A).
    """
    try:
        U, s, Vt = sla.svd(M, full_matrices=False, lapack_driver="gesdd", check_finite=False)
    except sla.LinAlgError:
        U, s, Vt = sla.svd(M, full_matrices=False, lapack_driver="gesvd", check_finite=False)
    Q = U @ Vt
    return Q, U, s, Vt


def _polar_factor_lowrank_factors(
    Left_B: np.ndarray,
    Right_A: np.ndarray,
    min_rank: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast O(N^2 (d_A + d_B)) exact nonzero-singular-space SVD of `M = Left_B @ Right_A.T`
    when `N = Left_B.shape[1] < min(d_B, d_A)` (20x-100x faster than dense d_B x d_A SVD, AND
    eliminates arbitrary unaligned LAPACK nullspace completion on out-of-span test noise!).
    When `N >= min(d_B, d_A) // 2`, falls back seamlessly to dense `_polar_factor`.
    If `min_rank > r` (e.g. k=512 when N=400), completes orthonormal columns up to `min_rank`.
    """
    d_B, n = Left_B.shape
    d_A, n2 = Right_A.shape
    assert n == n2
    p_max = min(d_B, d_A)
    if n >= max(p_max // 2, min_rank) and p_max <= 512:
        return _polar_factor(Left_B @ Right_A.T)
    if n >= p_max and min_rank <= p_max:
        return _polar_factor(Left_B @ Right_A.T)

    Q_B, R_B = np.linalg.qr(Left_B, mode="reduced")   # Q_B: (d_B, r_B), R_B: (r_B, n)
    Q_A, R_A = np.linalg.qr(Right_A, mode="reduced")  # Q_A: (d_A, r_A), R_A: (r_A, n)
    C_small = R_B @ R_A.T                              # (r_B, r_A)
    _, U_s, s_vals, Vt_s = _polar_factor(C_small)
    # Keep strictly positive singular modes (plus up to min_rank if requested)
    tol = 1e-10 * max(float(s_vals[0]) if s_vals.size else 0.0, EPS)
    r_pos = int(np.sum(s_vals > tol))
    r_keep = max(r_pos, min(len(s_vals), min_rank))
    U = Q_B @ U_s[:, :r_keep]      # (d_B, r_keep)
    Vt = Vt_s[:r_keep, :] @ Q_A.T  # (r_keep, d_A)
    s_out = s_vals[:r_keep]

    target_k = min(min_rank, p_max)
    if target_k > r_keep:
        # Complete orthonormal frame up to target_k using deterministic QR against standard basis
        need = target_k - r_keep
        rng_det = np.random.default_rng(20261125)
        pad_B = rng_det.standard_normal((d_B, need))
        pad_B -= U @ (U.T @ pad_B)
        Q_pad_B, _ = np.linalg.qr(pad_B, mode="reduced")
        pad_A = rng_det.standard_normal((d_A, need))
        V_cur = Vt.T
        pad_A -= V_cur @ (V_cur.T @ pad_A)
        Q_pad_A, _ = np.linalg.qr(pad_A, mode="reduced")
        U = np.hstack([U, Q_pad_B[:, :need]])
        Vt = np.vstack([Vt, Q_pad_A[:, :need].T])
        s_out = np.concatenate([s_out, np.zeros(need, dtype=np.float64)])

    Q = U @ Vt
    return Q, U, s_out, Vt


@dataclass
class _ShrinkageCovSpec:
    """
    Exact O(N r) representation of shrinkage covariance
        Sigma_lam = (1 - lam) * (Z Z^T / N) + lam * tau * I_d,
    where Z has shape (d, N), economy SVD Z = U diag(s) Vt (U: d x r),
    empirical eigenvalues mu_i = s_i^2 / N, and tau = tr(Sigma_hat) / d = sum(mu_i) / d.

    For any real power p (e.g. -1/2 or +1/2):
        Sigma_lam^p = d_perp^p I_d + U diag(d_i^p - d_perp^p) U^T
    when d_perp = lam * tau > 0, and U diag(mu_i^p) U^T when lam == 0 (for p >= 0).
    """
    U: np.ndarray          # (d, r)
    s_vals: np.ndarray     # (r,) singular values of Z
    Vt: np.ndarray         # (r, N) right singular vectors of Z
    eig_emp: np.ndarray    # (r,) = s^2 / N
    tau: float             # scalar mean variance per coordinate
    d: int

    @classmethod
    def from_centered_cols(cls, Z: np.ndarray) -> "_ShrinkageCovSpec":
        d, n = Z.shape
        try:
            U, s, Vt = sla.svd(Z, full_matrices=False, lapack_driver="gesdd", check_finite=False)
        except sla.LinAlgError:
            U, s, Vt = sla.svd(Z, full_matrices=False, lapack_driver="gesvd", check_finite=False)
        eig_emp = (s ** 2) / max(n, 1)
        tau = float(np.sum(eig_emp) / max(d, 1))
        return cls(U=U, s_vals=s, Vt=Vt, eig_emp=eig_emp, tau=max(tau, EPS), d=d)

    def apply_power_left(self, X: np.ndarray, lam: float, power: float) -> np.ndarray:
        """Compute Sigma_lam^power @ X for X of shape (d, m) in O(d r m) time."""
        lam = float(np.clip(lam, 0.0, 1.0))
        d_sub = (1.0 - lam) * self.eig_emp + lam * self.tau
        d_perp = lam * self.tau
        UX = self.U.T @ X  # (r, m)
        if d_perp <= EPS:
            if power < 0:
                # Pseudo-inverse on positive empirical eigenvalues only
                mask = d_sub > (1e-10 * max(float(np.max(d_sub)), EPS))
                pow_sub = np.where(mask, np.maximum(d_sub, EPS) ** power, 0.0)
                return self.U @ (pow_sub[:, None] * UX)
            pow_sub = np.maximum(d_sub, 0.0) ** power
            return self.U @ (pow_sub[:, None] * UX)
        pow_sub = d_sub ** power
        pow_perp = d_perp ** power
        diff = pow_sub - pow_perp
        return pow_perp * X + self.U @ (diff[:, None] * UX)

    def to_dense_power(self, lam: float, power: float) -> np.ndarray:
        """Materialize Sigma_lam^power of shape (d, d)."""
        lam = float(np.clip(lam, 0.0, 1.0))
        d_sub = (1.0 - lam) * self.eig_emp + lam * self.tau
        d_perp = lam * self.tau
        if d_perp <= EPS:
            if power < 0:
                mask = d_sub > (1e-10 * max(float(np.max(d_sub)), EPS))
                pow_sub = np.where(mask, np.maximum(d_sub, EPS) ** power, 0.0)
            else:
                pow_sub = np.maximum(d_sub, 0.0) ** power
            return (self.U * pow_sub[None, :]) @ self.U.T
        pow_sub = d_sub ** power
        pow_perp = d_perp ** power
        diff = pow_sub - pow_perp
        M = (self.U * diff[None, :]) @ self.U.T
        M.flat[:: self.d + 1] += pow_perp
        return M

    def ledoit_wolf_shrinkage(self, Z: np.ndarray) -> float:
        """
        Oracle Approximating / Ledoit-Wolf closed-form linear shrinkage intensity lam in [0, 1]
        toward spherical target tau * I_d for centered column matrix Z (d, N).
        Computed in O(N^2 d) without forming d x d matrices.
        """
        d, n = Z.shape
        if n <= 1:
            return 1.0
        # tr(Sigma_hat^2) = sum(eig_emp^2)
        tr_S2 = float(np.sum(self.eig_emp ** 2))
        tr_S_sq_over_d = (self.d * (self.tau ** 2))
        denom = tr_S2 - tr_S_sq_over_d
        if denom <= EPS:
            return 1.0
        # sum_i ||z_i||^4:
        col_norms_sq = np.sum(Z ** 2, axis=0)  # (N,)
        # Sample variance of rank-1 outer products:
        # b_bar = (1 / N^2) * (sum_i ||z_i||^4 - N * tr(Sigma_hat^2))
        numer = (float(np.sum(col_norms_sq ** 2)) / (n ** 2)) - (tr_S2 / n)
        lam = numer / denom
        return float(np.clip(lam, 0.01, 1.0))


# ---------------------------------------------------------------------------
# Base Interface for All Closed-Form Alignment Maps
# ---------------------------------------------------------------------------

class BaseKeyMap:
    """Common interface for closed-form donor->recipient representation maps."""

    name: str = "base"

    def transform(
        self,
        X_A: ArrayLike,
        X_B_orig: Optional[ArrayLike] = None,
        relations: Optional[Sequence[str]] = None,
    ) -> ArrayLike:
        arr_A, meta_A = _to_f64_2d(X_A)
        arr_B = None
        if X_B_orig is not None:
            arr_B, _ = _to_f64_2d(X_B_orig)
            if arr_B.shape[0] != arr_A.shape[0]:
                raise ValueError(
                    f"X_B_orig batch {arr_B.shape[0]} != X_A batch {arr_A.shape[0]}"
                )
        out_f64 = self._transform_f64(arr_A, arr_B, relations=relations)
        return _from_f64_2d(out_f64, meta_A)

    def _transform_f64(
        self,
        X_A: np.ndarray,
        X_B_orig: Optional[np.ndarray] = None,
        relations: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        raise NotImplementedError


class AffineKeyMap(BaseKeyMap):
    """
    Standard centered affine map:
        z_hat_B = mu_B + (z_A - mu_A) @ T.T
    where T has shape (d_B, d_A), mu_A has shape (d_A,), mu_B has shape (d_B,).
    """

    def __init__(self, name: str = "affine"):
        self.name = name
        self.mu_A: Optional[np.ndarray] = None
        self.mu_B: Optional[np.ndarray] = None
        self.T: Optional[np.ndarray] = None  # (d_B, d_A)
        self.scale: float = 1.0
        self.R: Optional[np.ndarray] = None  # optional directional matrix before scalar scale
        self.meta: Dict[str, Any] = {}

    def _transform_f64(
        self,
        X_A: np.ndarray,
        X_B_orig: Optional[np.ndarray] = None,
        relations: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        assert self.T is not None and self.mu_A is not None and self.mu_B is not None
        return self.mu_B[None, :] + (X_A - self.mu_A[None, :]) @ self.T.T


# ---------------------------------------------------------------------------
# 1. Plato's Key Baseline & Standard Baselines / Controls
# ---------------------------------------------------------------------------

class ProcrustesGlobalMap(AffineKeyMap):
    """
    Plato's Key primary map (`procrustes_global`):
        Z_B Z_A^T = U Sigma V^T,   R = U V^T,
        s = ||Z_B||_F / ||Z_A||_F  (or ||Z_B||_F / ||R Z_A||_F if post_proj_norm=True),
        z_hat_B = mu_B + s * R (z_A - mu_A).
    """

    def __init__(self, post_proj_norm: bool = False):
        super().__init__(name="procrustes_postproj" if post_proj_norm else "procrustes_global")
        self.post_proj_norm = post_proj_norm

    def fit(self, X_A: ArrayLike, X_B: ArrayLike) -> "ProcrustesGlobalMap":
        A, _ = _to_f64_2d(X_A)
        B, _ = _to_f64_2d(X_B)
        self.mu_A = A.mean(axis=0)
        self.mu_B = B.mean(axis=0)
        Z_A = (A - self.mu_A).T  # (d_A, N)
        Z_B = (B - self.mu_B).T  # (d_B, N)

        R, _, s_vals, _ = _polar_factor(Z_B @ Z_A.T)
        norm_B = float(sla.norm(Z_B, "fro"))
        if self.post_proj_norm:
            denom = float(sla.norm(R @ Z_A, "fro"))
        else:
            denom = float(sla.norm(Z_A, "fro"))
        self.scale = norm_B / max(denom, EPS)
        self.R = R
        self.T = self.scale * R
        self.meta = {
            "norm_A": float(sla.norm(Z_A, "fro")),
            "norm_B": norm_B,
            "scale": self.scale,
            "nuclear_corr": float(np.sum(s_vals) / max(norm_B * float(sla.norm(Z_A, "fro")), EPS)),
        }
        return self


class SpanProcrustesMap(AffineKeyMap):
    """
    Reduced-QR Span-Restricted Procrustes (`procrustes_span`):
    Eliminates high-dimensional nullspace hallucination when `N_train < min(d_A, d_B)`.
    Standard thin SVD of `Z_B Z_A^T` (`d_B x d_A`) pads the `r <= N_train - 1` positive singular
    modes with `min(d_A, d_B) - r` zero-singular-value modes formed from arbitrary LAPACK
    Householder reflections, injecting unaligned orthogonal noise on out-of-span test components.
    `SpanProcrustesMap` restricts `R_span = U_pos V_pos^T` strictly to the `r` positive singular
    modes via O(N^2 (d_A + d_B)) reduced-QR SVD (zero hyperparameters, zero validation data by default).
    """

    def __init__(self, calibrate_val_norm: bool = False):
        super().__init__(name="procrustes_span_valcal" if calibrate_val_norm else "procrustes_span")
        self.calibrate_val_norm = bool(calibrate_val_norm)

    def fit(
        self,
        X_A: ArrayLike,
        X_B: ArrayLike,
        X_A_val: Optional[ArrayLike] = None,
        X_B_val: Optional[ArrayLike] = None,
    ) -> "SpanProcrustesMap":
        A, _ = _to_f64_2d(X_A)
        B, _ = _to_f64_2d(X_B)
        self.mu_A = A.mean(axis=0)
        self.mu_B = B.mean(axis=0)
        Z_A = (A - self.mu_A).T
        Z_B = (B - self.mu_B).T

        R_span, U_pos, s_vals, _ = _polar_factor_lowrank_factors(Z_B, Z_A)
        norm_A = float(sla.norm(Z_A, "fro"))
        norm_B = float(sla.norm(Z_B, "fro"))
        denom_tr = float(sla.norm(R_span @ Z_A, "fro"))
        scale = norm_B / max(denom_tr, EPS)

        if self.calibrate_val_norm and X_A_val is not None and X_B_val is not None:
            Ava, _ = _to_f64_2d(X_A_val)
            Bva, _ = _to_f64_2d(X_B_val)
            ZA_va = (Ava - self.mu_A).T
            ZB_va = (Bva - self.mu_B).T
            scale = float(sla.norm(ZB_va, "fro")) / max(float(sla.norm(R_span @ ZA_va, "fro")), EPS)

        self.scale = scale
        self.R = R_span
        self.T = scale * R_span
        self.meta = {
            "rank_pos": int(U_pos.shape[1]),
            "nullspace_dims_zeroed": int(min(Z_A.shape[0], Z_B.shape[0]) - U_pos.shape[1]),
            "norm_A": norm_A,
            "norm_B": norm_B,
            "scale": self.scale,
            "calibrate_val_norm": self.calibrate_val_norm,
            "nuclear_corr": float(np.sum(s_vals) / max(norm_B * norm_A, EPS)),
        }
        return self


class HaarNullspaceProcrustesMap(AffineKeyMap):
    """
    Equally-Valid Haar-Random Orthogonal Nullspace Completion of Procrustes (`procrustes_haar_null`):
    When `r = rank(Z_B Z_A^T) < p = min(d_B, d_A)`, ANY matrix of the form
        R(Q_perp) = U_pos V_pos^T + U_perp Q_perp V_perp^T,   Q_perp in O(p - r),
    is an exact global maximizer of `max_{R^T R = I} tr(R^T Z_B Z_A^T)` and produces bit-for-bit
    identical training predictions (`R(Q_perp) Z_A == R_span Z_A`). Comparing independent Haar
    draws `Q_perp` against `SpanProcrustesMap` (`Q_perp = 0`) and `ProcrustesGlobalMap` (LAPACK's
    deterministic Householder completion) causally isolates out-of-span nullspace variance injection.
    """

    def __init__(self, seed: int = 0):
        super().__init__(name=f"procrustes_haar_null_s{seed}")
        self.seed = int(seed)

    def fit(self, X_A: ArrayLike, X_B: ArrayLike) -> "HaarNullspaceProcrustesMap":
        A, _ = _to_f64_2d(X_A)
        B, _ = _to_f64_2d(X_B)
        self.mu_A = A.mean(axis=0)
        self.mu_B = B.mean(axis=0)
        Z_A = (A - self.mu_A).T
        Z_B = (B - self.mu_B).T
        d_A, d_B = Z_A.shape[0], Z_B.shape[0]
        p = min(d_A, d_B)

        _, U_pos, s_vals, Vt_pos = _polar_factor_lowrank_factors(Z_B, Z_A)
        r = int(U_pos.shape[1])
        V_pos = Vt_pos.T  # (d_A, r)

        if r < p:
            need = p - r
            rng = np.random.default_rng(self.seed)
            G_B = rng.standard_normal((d_B, need))
            G_B -= U_pos @ (U_pos.T @ G_B)
            U_perp, _ = np.linalg.qr(G_B, mode="reduced")
            U_perp -= U_pos @ (U_pos.T @ U_perp)
            U_perp, _ = np.linalg.qr(U_perp, mode="reduced")

            G_A = rng.standard_normal((d_A, need))
            G_A -= V_pos @ (V_pos.T @ G_A)
            V_perp, _ = np.linalg.qr(G_A, mode="reduced")
            V_perp -= V_pos @ (V_pos.T @ V_perp)
            V_perp, _ = np.linalg.qr(V_perp, mode="reduced")

            Q_haar, _, _, _ = _polar_factor(rng.standard_normal((need, need)))
            R_full = (U_pos @ Vt_pos) + (U_perp @ Q_haar @ V_perp.T)
        else:
            R_full = U_pos @ Vt_pos

        norm_A = float(sla.norm(Z_A, "fro"))
        norm_B = float(sla.norm(Z_B, "fro"))
        self.scale = norm_B / max(norm_A, EPS)
        self.R = R_full
        self.T = self.scale * R_full
        self.meta = {
            "seed": self.seed,
            "rank_pos": r,
            "nullspace_dims_randomized": max(0, p - r),
            "scale": self.scale,
            "nuclear_corr": float(np.sum(s_vals) / max(norm_B * norm_A, EPS)),
        }
        return self


class ReducedRankRegressionMap(AffineKeyMap):
    """
    Closed-Form Ridge-Regularized Reduced-Rank Regression (`rrr_raw`, `rrr_rescaled`; Izenman 1975):
    Solves `min_{rank(W) <= k} ||Z_B - W Z_A||_F^2 + lam ||W||_F^2` in closed form via:
      1. Ridge predictor `Z_B_hat(lam) = W_ridge(lam) Z_A` using economy SVD of `Z_A`,
      2. Exact top-k left singular subspace `U_{hat,k}` of `Z_B_hat(lam)`,
      3. `W_{RRR,k} = U_{hat,k} U_{hat,k}^T W_ridge(lam)`.
    When `rescaled=True`, restores subspace Frobenius spread `s_k = ||U_{hat,k}^T Z_B||_F / ||W_{RRR,k} Z_A||_F`
    to isolate RRR's subspace choice from least-squares norm shrinkage at matched rank `k`.
    """

    DEFAULT_GRID: Tuple[float, ...] = (1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 30.0)

    def __init__(self, rank: int = 32, lam_rel: float = 1.0, rescaled: bool = True):
        super().__init__(name=f"rrr_{'rescaled' if rescaled else 'raw'}_k{int(rank)}")
        self.rank = int(rank)
        self.lam_rel = float(lam_rel)
        self.rescaled = bool(rescaled)

    @classmethod
    def fit_sweep_tuned(
        cls,
        X_A_train: ArrayLike,
        X_B_train: ArrayLike,
        X_A_val: ArrayLike,
        X_B_val: ArrayLike,
        ranks: Sequence[int] = (16, 32, 64, 128),
        lam_grid: Sequence[float] = DEFAULT_GRID,
    ) -> Dict[int, Tuple["ReducedRankRegressionMap", "ReducedRankRegressionMap"]]:
        """
        Fit `(rrr_raw, rrr_rescaled)` across all `ranks` simultaneously, tuning `lam_rel`
        strictly on `(X_A_val, X_B_val)` MSE in O(N^2 d).
        """
        A_tr, _ = _to_f64_2d(X_A_train)
        B_tr, _ = _to_f64_2d(X_B_train)
        A_va, _ = _to_f64_2d(X_A_val)
        B_va, _ = _to_f64_2d(X_B_val)

        mu_A = A_tr.mean(axis=0)
        mu_B = B_tr.mean(axis=0)
        Z_A = (A_tr - mu_A).T
        Z_B = (B_tr - mu_B).T
        A_va_c = A_va - mu_A[None, :]
        B_va_c = B_va - mu_B[None, :]

        try:
            U_A, s_A, Vt_A = sla.svd(Z_A, full_matrices=False, lapack_driver="gesdd", check_finite=False)
        except sla.LinAlgError:
            U_A, s_A, Vt_A = sla.svd(Z_A, full_matrices=False, lapack_driver="gesvd", check_finite=False)

        mean_sq = float(np.sum(s_A ** 2)) / max(Z_A.shape[0], 1)
        ZB_V = Z_B @ Vt_A.T  # (d_B, r_A)
        CA_va = A_va_c @ U_A  # (N_val, r_A)
        max_k = max(int(k) for k in ranks)

        # Precompute SVD of Z_B_hat(lam) = (ZB_V * (s_A^2 / (s_A^2 + lam))) @ Vt_A
        # Since Vt_A has orthonormal rows, left singular vectors of Z_B_hat(lam) EQUAL left singular vectors of M_lam = ZB_V * (s_A^2 / (s_A^2 + lam))!
        lam_cache = []
        for lam_rel in lam_grid:
            lam = max(float(lam_rel) * mean_sq, EPS)
            filt = s_A / (s_A ** 2 + lam)
            shrink = filt * s_A  # s_A^2 / (s_A^2 + lam)
            M_lam = ZB_V * shrink[None, :]  # (d_B, r_A)
            Q_B, R_B = np.linalg.qr(M_lam, mode="reduced")
            U_small, _, _ = sla.svd(R_B, full_matrices=False, check_finite=False)
            k_cap = min(max_k, U_small.shape[1])
            U_hat_max = Q_B @ U_small[:, :k_cap]  # (d_B, k_cap)
            W_core = ZB_V * filt[None, :]         # (d_B, r_A), so W_ridge = W_core @ U_A.T
            lam_cache.append((float(lam_rel), U_hat_max, W_core, shrink, M_lam))

        out: Dict[int, Tuple[ReducedRankRegressionMap, ReducedRankRegressionMap]] = {}
        for k in ranks:
            k_int = int(k)
            best_raw_mse = np.inf
            best_resc_mse = np.inf
            best_raw: Optional[ReducedRankRegressionMap] = None
            best_resc: Optional[ReducedRankRegressionMap] = None

            for lam_rel, U_hat_max, W_core, shrink, M_lam in lam_cache:
                ke = min(k_int, U_hat_max.shape[1])
                Uk = U_hat_max[:, :ke]                # (d_B, ke)
                # Core map from U_A coords (r_A) -> Uk coords (ke): G_k = Uk.T @ W_core  (ke, r_A)
                G_k = Uk.T @ W_core
                pred_tr_sub_norm = float(sla.norm(Uk.T @ M_lam, "fro"))
                targ_tr_sub_norm = float(sla.norm(Uk.T @ Z_B, "fro"))
                s_resc = targ_tr_sub_norm / max(pred_tr_sub_norm, EPS)

                # Validation predictions: (N_val, r_A) @ G_k.T @ Uk.T
                va_lat = CA_va @ G_k.T  # (N_val, ke)
                pred_va_raw = va_lat @ Uk.T
                mse_raw = float(np.mean((pred_va_raw - B_va_c) ** 2))
                if mse_raw < best_raw_mse:
                    best_raw_mse = mse_raw
                    W_raw = (Uk @ G_k) @ U_A.T
                    m_r = cls(rank=k_int, lam_rel=lam_rel, rescaled=False)
                    m_r.mu_A, m_r.mu_B = mu_A, mu_B
                    m_r.scale = 1.0
                    m_r.R = W_raw
                    m_r.T = W_raw
                    m_r.meta = {"rank": ke, "lam_rel": lam_rel, "rescaled": False, "scale": 1.0}
                    best_raw = m_r

                pred_va_resc = (s_resc * va_lat) @ Uk.T
                mse_resc = float(np.mean((pred_va_resc - B_va_c) ** 2))
                if mse_resc < best_resc_mse:
                    best_resc_mse = mse_resc
                    W_base = (Uk @ G_k) @ U_A.T
                    m_s = cls(rank=k_int, lam_rel=lam_rel, rescaled=True)
                    m_s.mu_A, m_s.mu_B = mu_A, mu_B
                    m_s.scale = s_resc
                    m_s.R = W_base
                    m_s.T = s_resc * W_base
                    m_s.meta = {"rank": ke, "lam_rel": lam_rel, "rescaled": True, "scale": s_resc}
                    best_resc = m_s

            assert best_raw is not None and best_resc is not None
            out[k_int] = (best_raw, best_resc)
        return out


class ProcrustesLSMap(AffineKeyMap):
    """
    Least-squares scaled Procrustes (`procrustes_ls`):
        min_{s, R^T R = I} ||Z_B - s R Z_A||_F^2
        => R = polar(Z_B Z_A^T),  s_LS = tr(R^T Z_B Z_A^T) / ||R Z_A||_F^2.
    """

    def __init__(self):
        super().__init__(name="procrustes_ls")

    def fit(self, X_A: ArrayLike, X_B: ArrayLike) -> "ProcrustesLSMap":
        A, _ = _to_f64_2d(X_A)
        B, _ = _to_f64_2d(X_B)
        self.mu_A = A.mean(axis=0)
        self.mu_B = B.mean(axis=0)
        Z_A = (A - self.mu_A).T
        Z_B = (B - self.mu_B).T

        R, _, s_vals, _ = _polar_factor(Z_B @ Z_A.T)
        proj_norm_sq = float(np.sum((R @ Z_A) ** 2))
        s_ls = float(np.sum(s_vals)) / max(proj_norm_sq, EPS)
        s_global = float(sla.norm(Z_B, "fro")) / max(float(sla.norm(Z_A, "fro")), EPS)
        self.scale = s_ls
        self.R = R
        self.T = s_ls * R
        self.meta = {
            "scale_ls": s_ls,
            "scale_global": s_global,
            "shrinkage_rho": s_ls / max(s_global, EPS),
        }
        return self


def fit_procrustes_family(
    X_A: ArrayLike, X_B: ArrayLike, fast_qr_svd: bool = False
) -> Tuple["ProcrustesGlobalMap", "ProcrustesGlobalMap", "ProcrustesLSMap"]:
    """
    Compute ONE shared thin SVD of `Z_B @ Z_A.T` and return `(proc_global, proc_postproj, proc_ls)`.
    When `fast_qr_svd=True`, uses O(N^2 (d_A + d_B)) reduced-QR SVD (~30x faster when N << d).
    """
    A, _ = _to_f64_2d(X_A)
    B, _ = _to_f64_2d(X_B)
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    Z_A = (A - mu_A).T
    Z_B = (B - mu_B).T

    if fast_qr_svd:
        R, _, s_vals, _ = _polar_factor_lowrank_factors(Z_B, Z_A)
    else:
        R, _, s_vals, _ = _polar_factor(Z_B @ Z_A.T)
    norm_A = float(sla.norm(Z_A, "fro"))
    norm_B = float(sla.norm(Z_B, "fro"))
    RZ_A = R @ Z_A
    proj_norm = float(sla.norm(RZ_A, "fro"))
    nuc = float(np.sum(s_vals))

    m_glob = ProcrustesGlobalMap(post_proj_norm=False)
    m_glob.mu_A, m_glob.mu_B = mu_A, mu_B
    m_glob.R = R
    m_glob.scale = norm_B / max(norm_A, EPS)
    m_glob.T = m_glob.scale * R
    m_glob.meta = {
        "norm_A": norm_A,
        "norm_B": norm_B,
        "scale": m_glob.scale,
        "nuclear_corr": nuc / max(norm_B * norm_A, EPS),
    }

    m_post = ProcrustesGlobalMap(post_proj_norm=True)
    m_post.mu_A, m_post.mu_B = mu_A, mu_B
    m_post.R = R
    m_post.scale = norm_B / max(proj_norm, EPS)
    m_post.T = m_post.scale * R
    m_post.meta = {
        "norm_A": norm_A,
        "norm_B": norm_B,
        "scale": m_post.scale,
        "nuclear_corr": nuc / max(norm_B * norm_A, EPS),
    }

    m_ls = ProcrustesLSMap()
    m_ls.mu_A, m_ls.mu_B = mu_A, mu_B
    m_ls.R = R
    m_ls.scale = nuc / max(proj_norm ** 2, EPS)
    m_ls.T = m_ls.scale * R
    m_ls.meta = {
        "scale_ls": m_ls.scale,
        "scale_global": m_glob.scale,
        "shrinkage_rho": m_ls.scale / max(m_glob.scale, EPS),
    }
    return m_glob, m_post, m_ls


class RandomControlMap(AffineKeyMap):
    """Haar-random semi-orthogonal map with Plato's Key global scale s = ||Z_B||_F / ||Z_A||_F."""

    def __init__(self, seed: int = 0):
        super().__init__(name="random_orthogonal")
        self.seed = seed

    def fit(self, X_A: ArrayLike, X_B: ArrayLike) -> "RandomControlMap":
        A, _ = _to_f64_2d(X_A)
        B, _ = _to_f64_2d(X_B)
        self.mu_A = A.mean(axis=0)
        self.mu_B = B.mean(axis=0)
        Z_A = (A - self.mu_A).T
        Z_B = (B - self.mu_B).T
        rng = np.random.default_rng(self.seed)
        G = rng.standard_normal((Z_B.shape[0], Z_A.shape[0]))
        R, _, _, _ = _polar_factor(G)
        s = float(sla.norm(Z_B, "fro")) / max(float(sla.norm(Z_A, "fro")), EPS)
        self.scale = s
        self.R = R
        self.T = s * R
        return self


class ShuffledProcrustesMap(AffineKeyMap):
    """Row-permuted Procrustes control (destroys pairing while preserving marginal spectra)."""

    def __init__(self, seed: int = 0):
        super().__init__(name="shuffled_procrustes")
        self.seed = seed

    def fit(self, X_A: ArrayLike, X_B: ArrayLike) -> "ShuffledProcrustesMap":
        A, _ = _to_f64_2d(X_A)
        B, _ = _to_f64_2d(X_B)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(A.shape[0])
        self.mu_A = A.mean(axis=0)
        self.mu_B = B.mean(axis=0)
        Z_A = (A[perm] - self.mu_A).T
        Z_B = (B - self.mu_B).T
        R, _, _, _ = _polar_factor(Z_B @ Z_A.T)
        s = float(sla.norm(Z_B, "fro")) / max(float(sla.norm(Z_A, "fro")), EPS)
        self.scale = s
        self.R = R
        self.T = s * R
        return self


class RidgeMap(AffineKeyMap):
    """
    Closed-form Ridge Regression (`ridge_raw` and `ridge_rescaled`):
        W(lam) = Z_B Z_A^T (Z_A Z_A^T + lam I_{d_A})^{-1}
    parameterized scale-invariantly via `lam = lam_rel * (||Z_A||_F^2 / d_A)`.
    Computed exactly in O(N^2 d) via economy SVD of Z_A:
        Z_A = U_A diag(s_A) V_A^T  =>  W = (Z_B V_A) diag(s_A / (s_A^2 + lam)) U_A^T.
    When `rescaled=True` (`ridge_rescaled`), rescales W by `||Z_B||_F / ||W Z_A||_F`.
    """

    DEFAULT_GRID: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)

    def __init__(self, lam_rel: float = 1.0, rescaled: bool = False):
        super().__init__(name="ridge_rescaled" if rescaled else "ridge_raw")
        self.lam_rel = float(lam_rel)
        self.rescaled = bool(rescaled)

    def fit(self, X_A: ArrayLike, X_B: ArrayLike) -> "RidgeMap":
        A, _ = _to_f64_2d(X_A)
        B, _ = _to_f64_2d(X_B)
        self.mu_A = A.mean(axis=0)
        self.mu_B = B.mean(axis=0)
        Z_A = (A - self.mu_A).T  # (d_A, N)
        Z_B = (B - self.mu_B).T  # (d_B, N)

        try:
            U_A, s_A, Vt_A = sla.svd(Z_A, full_matrices=False, lapack_driver="gesdd", check_finite=False)
        except sla.LinAlgError:
            U_A, s_A, Vt_A = sla.svd(Z_A, full_matrices=False, lapack_driver="gesvd", check_finite=False)

        mean_sq = float(np.sum(s_A ** 2)) / max(Z_A.shape[0], 1)
        lam = max(self.lam_rel * mean_sq, EPS)
        filt = s_A / (s_A ** 2 + lam)
        # W = Z_B @ Vt_A.T @ diag(filt) @ U_A.T
        ZB_V = Z_B @ Vt_A.T  # (d_B, r)
        W = (ZB_V * filt[None, :]) @ U_A.T  # (d_B, d_A)

        norm_B = float(sla.norm(Z_B, "fro"))
        # ||W Z_A||_F = ||(ZB_V * (filt * s_A))||_F since Vt_A has orthonormal rows!
        pred_norm = float(sla.norm(ZB_V * (filt * s_A)[None, :], "fro"))
        if self.rescaled:
            self.scale = norm_B / max(pred_norm, EPS)
        else:
            self.scale = 1.0
        self.R = W
        self.T = self.scale * W
        self.meta = {
            "lam_rel": self.lam_rel,
            "lam_abs": lam,
            "rescaled": self.rescaled,
            "scale": self.scale,
            "train_pred_norm_ratio": pred_norm / max(norm_B, EPS),
        }
        return self

    @classmethod
    def fit_tuned(
        cls,
        X_A_train: ArrayLike,
        X_B_train: ArrayLike,
        X_A_val: ArrayLike,
        X_B_val: ArrayLike,
        rescaled: bool = False,
        lam_grid: Sequence[float] = DEFAULT_GRID,
        val_eval_fn: Optional[Callable[["RidgeMap"], float]] = None,
    ) -> "RidgeMap":
        """
        Tune `lam_rel` STRICTLY on validation split `(X_A_val, X_B_val)`.
        If `val_eval_fn` is provided (e.g. validation transfer accuracy), maximizes `val_eval_fn(m)`.
        Otherwise maximizes validation cosine similarity (rescaled) or negative MSE (raw)
        using a single shared SVD of `X_A_train`.
        """
        A_tr, _ = _to_f64_2d(X_A_train)
        B_tr, _ = _to_f64_2d(X_B_train)
        A_va, _ = _to_f64_2d(X_A_val)
        B_va, _ = _to_f64_2d(X_B_val)

        mu_A = A_tr.mean(axis=0)
        mu_B = B_tr.mean(axis=0)
        Z_A = (A_tr - mu_A).T
        Z_B = (B_tr - mu_B).T
        try:
            U_A, s_A, Vt_A = sla.svd(Z_A, full_matrices=False, lapack_driver="gesdd", check_finite=False)
        except sla.LinAlgError:
            U_A, s_A, Vt_A = sla.svd(Z_A, full_matrices=False, lapack_driver="gesvd", check_finite=False)

        mean_sq = float(np.sum(s_A ** 2)) / max(Z_A.shape[0], 1)
        ZB_V = Z_B @ Vt_A.T
        norm_B = float(sla.norm(Z_B, "fro"))

        best_score = -np.inf
        best_map: Optional[RidgeMap] = None
        scores_by_lam: Dict[str, float] = {}

        for lam_rel in lam_grid:
            lam = max(float(lam_rel) * mean_sq, EPS)
            filt = s_A / (s_A ** 2 + lam)
            W = (ZB_V * filt[None, :]) @ U_A.T
            pred_norm = float(sla.norm(ZB_V * (filt * s_A)[None, :], "fro"))
            scale = (norm_B / max(pred_norm, EPS)) if rescaled else 1.0

            cand = cls(lam_rel=float(lam_rel), rescaled=rescaled)
            cand.mu_A = mu_A
            cand.mu_B = mu_B
            cand.R = W
            cand.scale = scale
            cand.T = scale * W
            cand.meta = {
                "lam_rel": float(lam_rel),
                "lam_abs": lam,
                "rescaled": rescaled,
                "scale": scale,
            }

            if val_eval_fn is not None:
                score = float(val_eval_fn(cand))
            else:
                pred_va = cand._transform_f64(A_va)
                score = -float(np.mean((pred_va - B_va) ** 2))
            scores_by_lam[str(lam_rel)] = score
            if score > best_score:
                best_score = score
                best_map = cand

        assert best_map is not None
        best_map.meta["val_scores"] = scores_by_lam
        best_map.meta["best_val_score"] = best_score
        return best_map


# ---------------------------------------------------------------------------
# 2. VecMap Whitened Baseline vs. Monge-Procrustes (Covariance-Preserving OT)
# ---------------------------------------------------------------------------

class _CovarianceTransportBase(AffineKeyMap):
    """
    Shared fast O(N^2 d) engine for:
      - `VecMapWhitenedMap` (`sigma_B_polar_power = -0.5`)
      - `MongeProcrustesMap` (`sigma_B_polar_power = +0.5`)

    Both construct `T = s_cal * Sigma_{B,lam_B}^{+1/2} Q Sigma_{A,lam_A}^{-1/2}`
    so that `T Sigma_{A,lam_A} T^T = Sigma_{B,lam_B}` (up to rectangular rank & calibration),
    differing strictly in the polar alignment objective:
      - VecMap: Q = polar(Sigma_{B,lam_B}^{-1/2} Z_B Z_A^T Sigma_{A,lam_A}^{-1/2})
                (minimizes error in whitened space, over-weighting noisy tail eigenvalues of B!)
      - Monge:  Q = polar(Sigma_{B,lam_B}^{+1/2} Z_B Z_A^T Sigma_{A,lam_A}^{-1/2})
                (exact closed-form minimizer of ||Z_B - T Z_A||_F^2 in recipient B's space!).
    """

    def __init__(
        self,
        name: str,
        sigma_B_polar_power: float,
        lam_A: Optional[float] = None,
        lam_B: Optional[float] = None,
        calibrate_frobenius: bool = True,
    ):
        super().__init__(name=name)
        self.sigma_B_polar_power = float(sigma_B_polar_power)
        self.lam_A = lam_A
        self.lam_B = lam_B
        self.calibrate_frobenius = bool(calibrate_frobenius)

    def fit(self, X_A: ArrayLike, X_B: ArrayLike) -> "_CovarianceTransportBase":
        A, _ = _to_f64_2d(X_A)
        B, _ = _to_f64_2d(X_B)
        self.mu_A = A.mean(axis=0)
        self.mu_B = B.mean(axis=0)
        Z_A = (A - self.mu_A).T
        Z_B = (B - self.mu_B).T

        spec_A = _ShrinkageCovSpec.from_centered_cols(Z_A)
        spec_B = _ShrinkageCovSpec.from_centered_cols(Z_B)

        lam_A_eff = spec_A.ledoit_wolf_shrinkage(Z_A) if self.lam_A is None else float(self.lam_A)
        if self.lam_B is None:
            # For Monge (+0.5), Sigma_B^{+1/2} never inverts small eigenvalues; mild LW shrinkage
            # preserves full rank on out-of-span directions when N < d_B.
            lam_B_eff = spec_B.ledoit_wolf_shrinkage(Z_B)
        else:
            lam_B_eff = float(self.lam_B)

        return self._fit_from_specs(Z_A, Z_B, spec_A, spec_B, lam_A_eff, lam_B_eff)

    @staticmethod
    def _compute_core(
        spec_A: _ShrinkageCovSpec,
        spec_B: _ShrinkageCovSpec,
        V_cross: np.ndarray,
        lam_A_eff: float,
        lam_B_eff: float,
        sigma_B_polar_power: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute r_B x r_A core matrices `(Q_core, T_core, pred_train_norm)` in O(r^3) (~4ms!)
        without forming any d_B x d_A matrices!
        """
        la = float(np.clip(lam_A_eff, 0.0, 1.0))
        lb = float(np.clip(lam_B_eff, 0.0, 1.0))

        dA = (1.0 - la) * spec_A.eig_emp + la * spec_A.tau
        if la * spec_A.tau <= EPS:
            maskA = dA > (1e-10 * max(float(np.max(dA)), EPS))
            inv_half_A = np.where(maskA, np.maximum(dA, EPS) ** (-0.5), 0.0)
        else:
            inv_half_A = dA ** (-0.5)

        dB = (1.0 - lb) * spec_B.eig_emp + lb * spec_B.tau
        if lb * spec_B.tau <= EPS:
            maskB = dB > (1e-10 * max(float(np.max(dB)), EPS))
            if sigma_B_polar_power < 0:
                pow_polar_B = np.where(maskB, np.maximum(dB, EPS) ** sigma_B_polar_power, 0.0)
            else:
                pow_polar_B = np.maximum(dB, 0.0) ** sigma_B_polar_power
            half_B = np.maximum(dB, 0.0) ** 0.5
        else:
            pow_polar_B = dB ** sigma_B_polar_power
            half_B = dB ** 0.5

        left_w = pow_polar_B * spec_B.s_vals
        right_w = inv_half_A * spec_A.s_vals
        C_core = (left_w[:, None] * V_cross) * right_w[None, :]  # (r_B, r_A)
        Q_core, _, _, _ = _polar_factor(C_core)
        T_core = (half_B[:, None] * Q_core) * inv_half_A[None, :]  # (r_B, r_A)
        pred_train_norm = float(sla.norm(T_core * spec_A.s_vals[None, :], "fro"))
        return Q_core, T_core, pred_train_norm

    def _fit_from_specs(
        self,
        Z_A: np.ndarray,
        Z_B: np.ndarray,
        spec_A: _ShrinkageCovSpec,
        spec_B: _ShrinkageCovSpec,
        lam_A_eff: float,
        lam_B_eff: float,
        V_cross: Optional[np.ndarray] = None,
    ) -> "_CovarianceTransportBase":
        if V_cross is None:
            V_cross = spec_B.Vt @ spec_A.Vt.T
        Q_core, T_core, pred_norm = self._compute_core(
            spec_A, spec_B, V_cross, lam_A_eff, lam_B_eff, self.sigma_B_polar_power
        )
        norm_B = float(np.sqrt(np.sum(spec_B.s_vals ** 2)))
        s_cal = (norm_B / max(pred_norm, EPS)) if self.calibrate_frobenius else 1.0

        self.R = None  # Lazily avoid redundant d_B x d_A allocation unless needed
        self.scale = s_cal
        self.T = s_cal * ((spec_B.U @ T_core) @ spec_A.U.T)
        self.meta = {
            "lam_A": lam_A_eff,
            "lam_B": lam_B_eff,
            "sigma_B_polar_power": self.sigma_B_polar_power,
            "s_cal": s_cal,
            "raw_pred_norm_ratio": pred_norm / max(norm_B, EPS),
        }
        return self

    @classmethod
    def fit_tuned(
        cls,
        X_A_train: ArrayLike,
        X_B_train: ArrayLike,
        X_A_val: ArrayLike,
        X_B_val: ArrayLike,
        lam_A_grid: Sequence[float] = (0.05, 0.15, 0.3, 0.5, 0.75, 1.0),
        lam_B_grid: Sequence[float] = (0.0, 0.1, 0.3, 0.6, 1.0),
        calibrate_frobenius: bool = True,
        val_eval_fn: Optional[Callable[["_CovarianceTransportBase"], float]] = None,
    ) -> "_CovarianceTransportBase":
        """
        Tune `(lam_A, lam_B)` STRICTLY on validation split `(X_A_val, X_B_val)` in O(r^3) per grid point (~4ms!).
        """
        A_tr, _ = _to_f64_2d(X_A_train)
        B_tr, _ = _to_f64_2d(X_B_train)
        A_va, _ = _to_f64_2d(X_A_val)
        B_va, _ = _to_f64_2d(X_B_val)

        mu_A = A_tr.mean(axis=0)
        mu_B = B_tr.mean(axis=0)
        Z_A = (A_tr - mu_A).T
        Z_B = (B_tr - mu_B).T
        spec_A = _ShrinkageCovSpec.from_centered_cols(Z_A)
        spec_B = _ShrinkageCovSpec.from_centered_cols(Z_B)
        V_cross = spec_B.Vt @ spec_A.Vt.T
        norm_B_tr = float(np.sqrt(np.sum(spec_B.s_vals ** 2)))

        # Pre-project validation set into (U_A, U_B) ONCE so grid evaluation takes <0.1ms per point!
        A_va_c = A_va - mu_A[None, :]
        B_va_c = B_va - mu_B[None, :]
        CA_va = A_va_c @ spec_A.U   # (N_val, r_A)
        CB_va = B_va_c @ spec_B.U   # (N_val, r_B)
        norm_B_va = float(sla.norm(B_va_c, "fro"))
        perp_B_va_sq = max(float(np.sum(B_va_c ** 2) - np.sum(CB_va ** 2)), 0.0)
        n_va_elems = max(B_va_c.size, 1)
        p_B_polar = -0.5 if (cls is VecMapWhitenedMap) else +0.5

        lw_A = spec_A.ledoit_wolf_shrinkage(Z_A)
        lw_B = spec_B.ledoit_wolf_shrinkage(Z_B)
        candidates: List[Tuple[float, float]] = [(lw_A, lw_B)]
        for la in lam_A_grid:
            for lb in lam_B_grid:
                if cls is VecMapWhitenedMap and lb <= 0.0 and Z_B.shape[1] <= Z_B.shape[0]:
                    continue
                candidates.append((float(la), float(lb)))

        best_score = -np.inf
        best_params: Optional[Tuple[float, float, np.ndarray, float]] = None
        best_map_custom: Optional[_CovarianceTransportBase] = None
        grid_scores: Dict[str, float] = {}

        for la, lb in candidates:
            key = f"({la:.3f},{lb:.3f})"
            if key in grid_scores:
                continue
            _, T_core, pred_tr_norm = cls._compute_core(spec_A, spec_B, V_cross, la, lb, p_B_polar)
            s_cal = (norm_B_tr / max(pred_tr_norm, EPS)) if calibrate_frobenius else 1.0

            pred_va_sub = (CA_va @ T_core.T) * s_cal  # (N_val, r_B)
            if calibrate_frobenius and Z_A.shape[1] < min(Z_A.shape[0], Z_B.shape[0]):
                v_rat = norm_B_va / max(float(sla.norm(pred_va_sub, "fro")), EPS)
                s_cal *= v_rat
                pred_va_sub *= v_rat

            if val_eval_fn is not None:
                cand = cls(lam_A=la, lam_B=lb, calibrate_frobenius=calibrate_frobenius)
                cand.mu_A, cand.mu_B = mu_A, mu_B
                cand.scale = s_cal
                cand.T = s_cal * ((spec_B.U @ T_core) @ spec_A.U.T)
                cand.meta = {"lam_A": la, "lam_B": lb, "sigma_B_polar_power": p_B_polar, "s_cal": s_cal}
                score = float(val_eval_fn(cand))
                if score > best_score:
                    best_score = score
                    best_map_custom = cand
            else:
                mse_va = (float(np.sum((CB_va - pred_va_sub) ** 2)) + perp_B_va_sq) / n_va_elems
                score = -mse_va
                if score > best_score:
                    best_score = score
                    best_params = (la, lb, T_core, s_cal)
            grid_scores[key] = score

        if val_eval_fn is not None:
            assert best_map_custom is not None
            best_map = best_map_custom
        else:
            assert best_params is not None
            la_b, lb_b, T_core_b, s_cal_b = best_params
            best_map = cls(lam_A=la_b, lam_B=lb_b, calibrate_frobenius=calibrate_frobenius)
            best_map.mu_A, best_map.mu_B = mu_A, mu_B
            best_map.scale = s_cal_b
            # Materialize full (d_B, d_A) matrix ONCE for the winning hyperparameters!
            best_map.T = s_cal_b * ((spec_B.U @ T_core_b) @ spec_A.U.T)
            best_map.meta = {
                "lam_A": la_b,
                "lam_B": lb_b,
                "sigma_B_polar_power": p_B_polar,
                "s_cal": s_cal_b,
            }

        best_map.meta["val_grid_scores"] = grid_scores
        best_map.meta["best_val_score"] = best_score
        return best_map


class VecMapWhitenedMap(_CovarianceTransportBase):
    """
    Whitened Rotation / VecMap baseline (`vecmap_whitened`):
        Q_VM = polar(Sigma_{B,lam_B}^{-1/2} Z_B Z_A^T Sigma_{A,lam_A}^{-1/2}),
        T_VM = s_cal * Sigma_{B,lam_B}^{+1/2} Q_VM Sigma_{A,lam_A}^{-1/2}.
    """

    def __init__(
        self,
        lam_A: Optional[float] = None,
        lam_B: Optional[float] = None,
        calibrate_frobenius: bool = True,
    ):
        super().__init__(
            name="vecmap_whitened",
            sigma_B_polar_power=-0.5,
            lam_A=lam_A,
            lam_B=lam_B,
            calibrate_frobenius=calibrate_frobenius,
        )


class MongeProcrustesMap(_CovarianceTransportBase):
    """
    Monge-Procrustes: Exact Closed-Form Covariance-Preserving Paired Optimal Transport (`monge_procrustes`):
        min_{T Sigma_A T^T = Sigma_B} ||Z_B - T Z_A||_F^2
        => Q* = polar(Sigma_{B,lam_B}^{+1/2} Z_B Z_A^T Sigma_{A,lam_A}^{-1/2}),
           T* = s_cal * Sigma_{B,lam_B}^{+1/2} Q* Sigma_{A,lam_A}^{-1/2}.
    Notice the crucial `+1/2` power on Sigma_B inside the polar factor!
    """

    def __init__(
        self,
        lam_A: Optional[float] = None,
        lam_B: Optional[float] = None,
        calibrate_frobenius: bool = True,
    ):
        super().__init__(
            name="monge_procrustes",
            sigma_B_polar_power=+0.5,
            lam_A=lam_A,
            lam_B=lam_B,
            calibrate_frobenius=calibrate_frobenius,
        )


# ---------------------------------------------------------------------------
# 3. RMSNorm-Weighted & Downstream-Readout-Weighted Procrustes
# ---------------------------------------------------------------------------

class RMSNormWeightedProcrustesMap(AffineKeyMap):
    """
    RMSNorm-Weighted Procrustes (`rmsnorm_weighted_procrustes`):
    Solves the exact closed-form RMSNorm-metric alignment problem:
        min_{T Sigma_{A,lam_A} T^T = Sigma_{B,lam_B}} || diag(w_B) (Z_B - T Z_A) ||_F^2
    whose exact global optimum (without any commutativity assumption!) is:
        M_B = diag(w_B^2),
        Q*  = polar(Sigma_{B,lam_B}^{+1/2} M_B Z_B Z_A^T Sigma_{A,lam_A}^{-1/2}),
        T*  = s_cal * Sigma_{B,lam_B}^{+1/2} Q* Sigma_{A,lam_A}^{-1/2}.
    Note:
      - Setting `(lam_A=1.0, lam_B=1.0)` recovers strictly semi-orthogonal weighted Procrustes
        `R = polar(M_B Z_B Z_A^T)`.
      - Setting `lam_A < 1` prevents high-variance donor nuisance directions from rotating into
        recipient B's high-gain RMSNorm channels!
      - Also supports `mode="post_norm_similarity"` (`D_B^{-1} polar((D_B Z_B)(D_A Z_A)^T) D_A`).
    """

    def __init__(
        self,
        power: float = 1.0,
        reg: float = 0.05,
        mode: str = "metric_ot",
        lam_A: Optional[float] = None,
        lam_B: Optional[float] = None,
    ):
        super().__init__(name="rmsnorm_weighted_procrustes")
        self.power = float(power)
        self.reg = float(reg)
        self.mode = mode
        self.lam_A = lam_A
        self.lam_B = lam_B

    @staticmethod
    def _build_weights(gamma: ArrayLike, d_expected: int, power: float, reg: float) -> np.ndarray:
        g, _ = _to_f64_2d(gamma)
        g = g.reshape(-1)
        if g.shape[0] != d_expected:
            raise ValueError(f"gamma length {g.shape[0]} != expected dim {d_expected}")
        g_sq = g ** 2
        mean_sq = max(float(np.mean(g_sq)), EPS)
        base = (1.0 - reg) * g_sq + reg * mean_sq
        w = base ** (0.5 * power)
        return w / max(float(np.sqrt(np.mean(w ** 2))), EPS)

    def fit(
        self,
        X_A: ArrayLike,
        X_B: ArrayLike,
        gamma_B: ArrayLike,
        gamma_A: Optional[ArrayLike] = None,
    ) -> "RMSNormWeightedProcrustesMap":
        A, _ = _to_f64_2d(X_A)
        B, _ = _to_f64_2d(X_B)
        self.mu_A = A.mean(axis=0)
        self.mu_B = B.mean(axis=0)
        Z_A = (A - self.mu_A).T
        Z_B = (B - self.mu_B).T

        w_B = self._build_weights(gamma_B, Z_B.shape[0], power=self.power, reg=self.reg)

        if self.mode == "orthogonal_metric":
            R, _, _, _ = _polar_factor_lowrank_factors((w_B ** 2)[:, None] * Z_B, Z_A)
            norm_B = float(sla.norm(Z_B, "fro"))
            denom = float(sla.norm(R @ Z_A, "fro"))
            self.scale = norm_B / max(denom, EPS)
            self.R = R
            self.T = self.scale * R
            lam_A_eff, lam_B_eff = 1.0, 1.0
        elif self.mode == "metric_ot":
            spec_A = _ShrinkageCovSpec.from_centered_cols(Z_A)
            spec_B = _ShrinkageCovSpec.from_centered_cols(Z_B)
            lam_A_eff = spec_A.ledoit_wolf_shrinkage(Z_A) if self.lam_A is None else float(self.lam_A)
            lam_B_eff = spec_B.ledoit_wolf_shrinkage(Z_B) if self.lam_B is None else float(self.lam_B)

            ZA_inv_half = spec_A.apply_power_left(Z_A, lam=lam_A_eff, power=-0.5)
            MB_ZB = (w_B ** 2)[:, None] * Z_B
            Left_B = spec_B.apply_power_left(MB_ZB, lam=lam_B_eff, power=+0.5)
            Q, _, _, _ = _polar_factor_lowrank_factors(Left_B, ZA_inv_half)

            Q_right = spec_A.apply_power_left(Q.T, lam=lam_A_eff, power=-0.5).T
            T_raw = spec_B.apply_power_left(Q_right, lam=lam_B_eff, power=+0.5)
            norm_B = float(sla.norm(Z_B, "fro"))
            pred_norm = float(sla.norm(T_raw @ Z_A, "fro"))
            self.scale = norm_B / max(pred_norm, EPS)
            self.R = Q
            self.T = self.scale * T_raw
        elif self.mode == "post_norm_similarity":
            if gamma_A is not None:
                w_A = self._build_weights(gamma_A, Z_A.shape[0], power=self.power, reg=self.reg)
            else:
                w_A = np.ones(Z_A.shape[0], dtype=np.float64)
            Q, _, _, _ = _polar_factor_lowrank_factors(w_B[:, None] * Z_B, w_A[:, None] * Z_A)
            T_raw = (1.0 / np.maximum(w_B, 1e-4))[:, None] * Q * w_A[None, :]
            norm_B = float(sla.norm(Z_B, "fro"))
            pred_norm = float(sla.norm(T_raw @ Z_A, "fro"))
            self.scale = norm_B / max(pred_norm, EPS)
            self.R = Q
            self.T = self.scale * T_raw
            lam_A_eff, lam_B_eff = 1.0, 1.0
        else:
            raise ValueError(f"Unknown RMSNormWeightedProcrustesMap mode: {self.mode}")

        self.meta = {
            "power": self.power,
            "reg": self.reg,
            "mode": self.mode,
            "lam_A": lam_A_eff,
            "lam_B": lam_B_eff,
            "gamma_B_cv": float(np.std(w_B) / max(float(np.mean(w_B)), EPS)),
        }
        return self

    @classmethod
    def fit_tuned(
        cls,
        X_A_train: ArrayLike,
        X_B_train: ArrayLike,
        X_A_val: ArrayLike,
        X_B_val: ArrayLike,
        gamma_B: ArrayLike,
        gamma_A: Optional[ArrayLike] = None,
        configs: Sequence[Tuple[str, float, float]] = (
            ("orthogonal_metric", 0.5, 0.05),
            ("orthogonal_metric", 1.0, 0.05),
            ("metric_ot", 0.5, 0.05),
            ("metric_ot", 1.0, 0.05),
            ("post_norm_similarity", 0.5, 0.10),
            ("post_norm_similarity", 1.0, 0.10),
        ),
        val_eval_fn: Optional[Callable[["RMSNormWeightedProcrustesMap"], float]] = None,
    ) -> "RMSNormWeightedProcrustesMap":
        A_va, _ = _to_f64_2d(X_A_val)
        B_va, _ = _to_f64_2d(X_B_val)
        w_B_eval = cls._build_weights(gamma_B, B_va.shape[1], power=1.0, reg=0.01)

        best_score = -np.inf
        best_map: Optional[RMSNormWeightedProcrustesMap] = None
        scores: Dict[str, float] = {}
        for mode, pwr, reg in configs:
            cand = cls(power=pwr, reg=reg, mode=mode).fit(
                X_A_train, X_B_train, gamma_B=gamma_B, gamma_A=gamma_A
            )
            if A_va.shape[0] > 1 and cand.T is not None and cand.mu_A is not None and cand.mu_B is not None:
                vp = (A_va - cand.mu_A[None, :]) @ cand.T.T
                vt = B_va - cand.mu_B[None, :]
                v_rat = float(sla.norm(vt, "fro")) / max(float(sla.norm(vp, "fro")), EPS)
                cand.scale *= v_rat
                cand.T = cand.T * v_rat
            if val_eval_fn is not None:
                score = float(val_eval_fn(cand))
            else:
                pred = cand._transform_f64(A_va)
                diff = (pred - B_va) * w_B_eval[None, :]
                score = -float(np.mean(diff ** 2))
            key = f"{mode}_p{pwr}_r{reg}"
            scores[key] = score
            if score > best_score:
                best_score = score
                best_map = cand
        assert best_map is not None
        best_map.meta["val_scores"] = scores
        best_map.meta["best_val_score"] = best_score
        return best_map


class ReadoutWeightedProcrustesMap(AffineKeyMap):
    """
    Readout-Weighted Procrustes (`readout_weighted_procrustes`):
    Solves the exact closed-form downstream-readout-weighted alignment problem:
        min_{T Sigma_{A,lam_A} T^T = Sigma_{B,lam_B}} || M_B^{1/2} (Z_B - T Z_A) ||_F^2
    where `M_B = (1 - eta) I_{d_B} + eta * G_readout`, via:
        Q* = polar(Sigma_{B,lam_B}^{+1/2} M_B Z_B Z_A^T Sigma_{A,lam_A}^{-1/2}),
        T* = s_cal * Sigma_{B,lam_B}^{+1/2} Q* Sigma_{A,lam_A}^{-1/2}.
    When `(lam_A=1.0, lam_B=1.0)` (`orthogonal_only=True`), reduces to strictly semi-orthogonal
    `R = polar(M_B Z_B Z_A^T)`.
    """

    def __init__(
        self,
        eta: float = 0.7,
        orthogonal_only: bool = False,
        lam_A: Optional[float] = None,
        lam_B: Optional[float] = None,
    ):
        super().__init__(name="readout_weighted_procrustes")
        self.eta = float(np.clip(eta, 0.0, 1.0))
        self.orthogonal_only = bool(orthogonal_only)
        self.lam_A = lam_A
        self.lam_B = lam_B

    @staticmethod
    def build_lowrank_readout_metric(
        weight_matrices: Sequence[ArrayLike],
        gamma_B: Optional[ArrayLike] = None,
        rank: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray]:
        G = None
        g_vec = None
        if gamma_B is not None:
            g_arr, _ = _to_f64_2d(gamma_B)
            g_vec = g_arr.reshape(-1)

        for W_in in weight_matrices:
            W, _ = _to_f64_2d(W_in)
            if g_vec is not None:
                W = W * g_vec[None, :]
            nrm = max(float(sla.norm(W, "fro")), EPS)
            W_norm = W / nrm
            gram_piece = W_norm.T @ W_norm
            G = gram_piece if G is None else (G + gram_piece)

        assert G is not None, "weight_matrices must be non-empty"
        G = 0.5 * (G + G.T)
        eigvals, eigvecs = sla.eigh(G)
        idx = np.argsort(eigvals)[::-1][: min(rank, G.shape[0])]
        top_eigs = np.maximum(eigvals[idx], 0.0)
        U_read = eigvecs[:, idx]
        s_read = np.sqrt(top_eigs)
        return U_read, s_read

    def fit(
        self,
        X_A: ArrayLike,
        X_B: ArrayLike,
        U_read: Optional[ArrayLike] = None,
        s_read: Optional[ArrayLike] = None,
        gram_B: Optional[ArrayLike] = None,
        weight_matrices: Optional[Sequence[ArrayLike]] = None,
        gamma_B: Optional[ArrayLike] = None,
    ) -> "ReadoutWeightedProcrustesMap":
        A, _ = _to_f64_2d(X_A)
        B, _ = _to_f64_2d(X_B)
        self.mu_A = A.mean(axis=0)
        self.mu_B = B.mean(axis=0)
        Z_A = (A - self.mu_A).T
        Z_B = (B - self.mu_B).T
        d_B = Z_B.shape[0]

        if weight_matrices is not None and U_read is None and gram_B is None:
            U_read, s_read = self.build_lowrank_readout_metric(weight_matrices, gamma_B=gamma_B)

        if U_read is not None:
            U_r, _ = _to_f64_2d(U_read)
            if s_read is not None:
                s_r, _ = _to_f64_2d(s_read)
                w_eig = (s_r.reshape(-1)) ** 2
            else:
                w_eig = np.ones(U_r.shape[1], dtype=np.float64)
            tr_val = max(float(np.sum(w_eig)), EPS)
            w_eig = w_eig * (d_B / tr_val)
            MB_ZB = (1.0 - self.eta) * Z_B + self.eta * (U_r @ (w_eig[:, None] * (U_r.T @ Z_B)))
        elif gram_B is not None:
            G, _ = _to_f64_2d(gram_B)
            tr_val = max(float(np.trace(G)), EPS)
            G_norm = G * (d_B / tr_val)
            MB_ZB = (1.0 - self.eta) * Z_B + self.eta * (G_norm @ Z_B)
        else:
            raise ValueError("Provide (U_read[, s_read]), gram_B, or weight_matrices.")

        norm_B = float(sla.norm(Z_B, "fro"))
        if self.orthogonal_only:
            R, _, _, _ = _polar_factor_lowrank_factors(MB_ZB, Z_A)
            denom = float(sla.norm(R @ Z_A, "fro"))
            self.scale = norm_B / max(denom, EPS)
            self.R = R
            self.T = self.scale * R
            lam_A_eff, lam_B_eff = 1.0, 1.0
        else:
            spec_A = _ShrinkageCovSpec.from_centered_cols(Z_A)
            spec_B = _ShrinkageCovSpec.from_centered_cols(Z_B)
            lam_A_eff = spec_A.ledoit_wolf_shrinkage(Z_A) if self.lam_A is None else float(self.lam_A)
            lam_B_eff = spec_B.ledoit_wolf_shrinkage(Z_B) if self.lam_B is None else float(self.lam_B)

            ZA_inv_half = spec_A.apply_power_left(Z_A, lam=lam_A_eff, power=-0.5)
            Left_B = spec_B.apply_power_left(MB_ZB, lam=lam_B_eff, power=+0.5)
            Q, _, _, _ = _polar_factor_lowrank_factors(Left_B, ZA_inv_half)

            Q_right = spec_A.apply_power_left(Q.T, lam=lam_A_eff, power=-0.5).T
            T_raw = spec_B.apply_power_left(Q_right, lam=lam_B_eff, power=+0.5)
            pred_norm = float(sla.norm(T_raw @ Z_A, "fro"))
            self.scale = norm_B / max(pred_norm, EPS)
            self.R = Q
            self.T = self.scale * T_raw

        self.meta = {
            "eta": self.eta,
            "orthogonal_only": self.orthogonal_only,
            "lam_A": lam_A_eff,
            "lam_B": lam_B_eff,
            "scale": self.scale,
        }
        return self

    @classmethod
    def fit_tuned(
        cls,
        X_A_train: ArrayLike,
        X_B_train: ArrayLike,
        X_A_val: ArrayLike,
        X_B_val: ArrayLike,
        U_read: Optional[ArrayLike] = None,
        s_read: Optional[ArrayLike] = None,
        gram_B: Optional[ArrayLike] = None,
        eta_grid: Sequence[float] = (0.25, 0.5, 0.75, 0.95),
        val_eval_fn: Optional[Callable[["ReadoutWeightedProcrustesMap"], float]] = None,
    ) -> "ReadoutWeightedProcrustesMap":
        A_va, _ = _to_f64_2d(X_A_val)
        B_va, _ = _to_f64_2d(X_B_val)

        best_score = -np.inf
        best_map: Optional[ReadoutWeightedProcrustesMap] = None
        scores: Dict[str, float] = {}

        U_r = _to_f64_2d(U_read)[0] if U_read is not None else None
        s_r = _to_f64_2d(s_read)[0].reshape(-1) if s_read is not None else None
        G_r = _to_f64_2d(gram_B)[0] if gram_B is not None else None

        for orth in (True, False):
            for eta in eta_grid:
                cand = cls(eta=float(eta), orthogonal_only=orth).fit(
                    X_A_train, X_B_train, U_read=U_r, s_read=s_r, gram_B=G_r
                )
                if A_va.shape[0] > 1 and cand.T is not None and cand.mu_A is not None and cand.mu_B is not None:
                    vp = (A_va - cand.mu_A[None, :]) @ cand.T.T
                    vt = B_va - cand.mu_B[None, :]
                    v_rat = float(sla.norm(vt, "fro")) / max(float(sla.norm(vp, "fro")), EPS)
                    cand.scale *= v_rat
                    cand.T = cand.T * v_rat
                if val_eval_fn is not None:
                    score = float(val_eval_fn(cand))
                else:
                    diff = (cand._transform_f64(A_va) - B_va)
                    if U_r is not None:
                        proj = diff @ U_r
                        if s_r is not None:
                            proj = proj * s_r[None, :]
                        score = -float(np.mean(proj ** 2))
                    elif G_r is not None:
                        score = -float(np.mean(np.sum((diff @ G_r) * diff, axis=1)))
                    else:
                        score = -float(np.mean(diff ** 2))
                tag = f"{'orth' if orth else 'ot'}_e{eta:.2f}"
                scores[tag] = score
                if score > best_score:
                    best_score = score
                    best_map = cand
        assert best_map is not None
        best_map.meta["val_scores"] = scores
        best_map.meta["best_val_score"] = best_score
        return best_map


# ---------------------------------------------------------------------------
# 4. Per-Sample Recipient Norm Matching & Validation-Tuned Gain
# ---------------------------------------------------------------------------

class PerSampleNormGainMap(BaseKeyMap):
    """
    Per-Sample Norm Matching & Residual-Leakage Gain (`per_sample_norm_and_gain`):
    Wraps any centered affine base map (default `ProcrustesGlobalMap`) that predicts
        delta_B(z_A) = (z_A - mu_A) @ T_base.T,   mu_B = base_map.mu_B.

    Supports three norm modes + gain `alpha` (e.g. `alpha in [0.9, 1.0, 1.1, 1.2, 1.35, 1.5]`):
      - `norm_mode="global"`:
          `z_hat = mu_B + alpha * delta_B` (no `X_B_orig` required at test time).
      - `norm_mode="centered_match"`:
          Matches recipient's exact per-sample centered norm `||z_B_orig - mu_B||_2`:
          `z_hat = mu_B + alpha * (||z_B_orig - mu_B||_2 / ||delta_B||_2) * delta_B`
          (falls back to `"global"` if `X_B_orig` is None).
      - `norm_mode="total_match"`:
          First forms `z_tilde = mu_B + alpha * delta_B`, then matches recipient's exact
          total residual norm `||z_B_orig||_2`:
          `z_hat = (||z_B_orig||_2 / ||z_tilde||_2) * z_tilde` (optionally multiplied by `total_gain`).
    """

    DEFAULT_ALPHAS: Tuple[float, ...] = (0.9, 1.0, 1.1, 1.2, 1.35, 1.5)

    def __init__(
        self,
        base_map: Optional[AffineKeyMap] = None,
        norm_mode: str = "centered_match",
        alpha: float = 1.0,
        total_gain: float = 1.0,
    ):
        self.base_map = base_map if base_map is not None else ProcrustesGlobalMap()
        self.norm_mode = norm_mode
        self.alpha = float(alpha)
        self.total_gain = float(total_gain)
        self.name = f"per_sample_{norm_mode}_a{self.alpha:g}"
        self.meta: Dict[str, Any] = {}

    def fit(self, X_A: ArrayLike, X_B: ArrayLike) -> "PerSampleNormGainMap":
        if self.base_map.T is None:
            self.base_map.fit(X_A, X_B)  # type: ignore[attr-defined]
        self.meta = {
            "base_map": self.base_map.name,
            "norm_mode": self.norm_mode,
            "alpha": self.alpha,
            "total_gain": self.total_gain,
        }
        return self

    def _transform_f64(
        self,
        X_A: np.ndarray,
        X_B_orig: Optional[np.ndarray] = None,
        relations: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        assert self.base_map.T is not None and self.base_map.mu_A is not None and self.base_map.mu_B is not None
        mu_B = self.base_map.mu_B[None, :]
        delta_B = (X_A - self.base_map.mu_A[None, :]) @ self.base_map.T.T  # (N, d_B)

        if self.norm_mode == "global" or X_B_orig is None:
            return mu_B + self.alpha * delta_B

        if self.norm_mode == "centered_match":
            target_centered_norm = np.linalg.norm(X_B_orig - mu_B, axis=1, keepdims=True)
            cur_norm = np.maximum(np.linalg.norm(delta_B, axis=1, keepdims=True), EPS)
            return mu_B + (self.alpha * target_centered_norm / cur_norm) * delta_B

        if self.norm_mode == "total_match":
            z_tilde = mu_B + self.alpha * delta_B
            target_total_norm = np.linalg.norm(X_B_orig, axis=1, keepdims=True)
            cur_total_norm = np.maximum(np.linalg.norm(z_tilde, axis=1, keepdims=True), EPS)
            return (self.total_gain * target_total_norm / cur_total_norm) * z_tilde

        raise ValueError(f"Unknown norm_mode: {self.norm_mode}")

    @classmethod
    def fit_tuned(
        cls,
        X_A_train: ArrayLike,
        X_B_train: ArrayLike,
        val_eval_fn: Callable[["PerSampleNormGainMap"], float],
        base_map: Optional[AffineKeyMap] = None,
        norm_modes: Sequence[str] = ("global", "centered_match", "total_match"),
        alpha_grid: Sequence[float] = DEFAULT_ALPHAS,
    ) -> "PerSampleNormGainMap":
        """
        Tune `(norm_mode, alpha)` STRICTLY on validation swap interventions via `val_eval_fn`.
         Note: `base_map` is fit ONCE on `(X_A_train, X_B_train)` and reused across all candidates.
        """
        fitted_base = base_map if (base_map is not None and base_map.T is not None) else ProcrustesGlobalMap().fit(X_A_train, X_B_train)
        best_score = -np.inf
        best_map: Optional[PerSampleNormGainMap] = None
        scores: Dict[str, float] = {}

        for mode in norm_modes:
            for alpha in alpha_grid:
                cand = cls(base_map=fitted_base, norm_mode=mode, alpha=float(alpha))
                cand.meta = {"base_map": fitted_base.name, "norm_mode": mode, "alpha": float(alpha)}
                score = float(val_eval_fn(cand))
                scores[f"{mode}_a{alpha:g}"] = score
                if score > best_score:
                    best_score = score
                    best_map = cand

        assert best_map is not None
        best_map.name = "per_sample_norm_and_gain"
        best_map.meta["val_scores"] = scores
        best_map.meta["best_val_score"] = best_score
        return best_map


# ---------------------------------------------------------------------------
# 5. Relation-Centered Procrustes (Eliminating Prompt-Template Centroid Bias)
# ---------------------------------------------------------------------------

class RelationCenteredProcrustesMap(BaseKeyMap):
    """
    Within-Relation Centered Procrustes (`relation_centered_procrustes`):
    Removes between-relation prompt-template offsets (`mu_{rel(i)}`) before fitting R:
        z_check_{A,i} = z_{A,i} - mu_{A, rel(i)},   z_check_{B,i} = z_{B,i} - mu_{B, rel(i)},
        R_within = polar(Z_check_B Z_check_A^T).

    Supports two test-time application modes (zero test leakage!):
      - `test_mode="global_mean"` (default; requires NO relation labels at test time & works on unseen LORO relations!):
          `z_hat_B = mu_B + s_global * R_within (z_A - mu_A)`,
          where `s_global = ||Z_B_global||_F / ||R_within Z_A_global||_F`.
      - `test_mode="relation_mean"` (uses training relation centroids on seen relations;
          gracefully falls back to `global_mean` for unseen relations):
          `z_hat_B = mu_{B,r} + s_within * R_within (z_A - mu_{A,r})`,
          where `s_within = ||Z_check_B||_F / ||R_within Z_check_A||_F`.
    """

    def __init__(self, test_mode: str = "global_mean", between_weight: Optional[float] = None):
        if test_mode not in ("global_mean", "relation_mean"):
            raise ValueError(f"Invalid test_mode: {test_mode}")
        self.test_mode = test_mode
        self.between_weight = None if between_weight is None else float(np.clip(between_weight, 0.0, 1.0))
        self.name = f"relation_centered_{test_mode}"
        self.mu_A: Optional[np.ndarray] = None
        self.mu_B: Optional[np.ndarray] = None
        self.rel_mu_A: Dict[str, np.ndarray] = {}
        self.rel_mu_B: Dict[str, np.ndarray] = {}
        self.R: Optional[np.ndarray] = None
        self.s_global: float = 1.0
        self.s_within: float = 1.0
        self.T_global: Optional[np.ndarray] = None
        self.T_within: Optional[np.ndarray] = None
        self.meta: Dict[str, Any] = {}

    def fit(
        self,
        X_A: ArrayLike,
        X_B: ArrayLike,
        relations: Sequence[str],
        X_A_val: Optional[ArrayLike] = None,
        X_B_val: Optional[ArrayLike] = None,
    ) -> "RelationCenteredProcrustesMap":
        A, _ = _to_f64_2d(X_A)
        B, _ = _to_f64_2d(X_B)
        if len(relations) != A.shape[0]:
            raise ValueError(f"len(relations)={len(relations)} != N={A.shape[0]}")

        self.mu_A = A.mean(axis=0)
        self.mu_B = B.mean(axis=0)
        Z_A_glob = (A - self.mu_A).T
        Z_B_glob = (B - self.mu_B).T

        rels_arr = np.asarray(relations)
        uniq_rels = sorted(set(relations))
        A_within = np.empty_like(A)
        B_within = np.empty_like(B)
        self.rel_mu_A = {}
        self.rel_mu_B = {}
        cent_A_cols = []
        cent_B_cols = []

        for r in uniq_rels:
            mask = (rels_arr == r)
            nr = float(np.sum(mask))
            mu_Ar = A[mask].mean(axis=0)
            mu_Br = B[mask].mean(axis=0)
            self.rel_mu_A[str(r)] = mu_Ar
            self.rel_mu_B[str(r)] = mu_Br
            A_within[mask] = A[mask] - mu_Ar
            B_within[mask] = B[mask] - mu_Br
            cent_A_cols.append(np.sqrt(nr) * (mu_Ar - self.mu_A))
            cent_B_cols.append(np.sqrt(nr) * (mu_Br - self.mu_B))

        Z_A_w = A_within.T
        Z_B_w = B_within.T
        C_A_between = np.column_stack(cent_A_cols) if cent_A_cols else np.zeros((A.shape[1], 1))
        C_B_between = np.column_stack(cent_B_cols) if cent_B_cols else np.zeros((B.shape[1], 1))

        # Exact ANOVA decomposition: Z_B_glob @ Z_A_glob.T == Z_B_w @ Z_A_w.T + C_B_between @ C_A_between.T!
        # For `relation_mean` (which subtracts mu_{A,r} explicitly at test time), pure within-relation gamma=0 is optimal.
        # For `global_mean` (zero relation label at test time!), gamma=0 annihilates prompt-template centroids in span(Z_w)^perp,
        # whereas tuning `gamma_between in (0.0, 0.03, 0.1, 0.25, 0.5, 1.0)` on validation de-weights template dominance (1/n_r)
        # while keeping the template subspace aligned!
        def _fit_for_gamma(gamma_b: float) -> np.ndarray:
            if gamma_b <= 0.0:
                R_cand, _, _, _ = _polar_factor_lowrank_factors(Z_B_w, Z_A_w)
                return R_cand
            s_b = np.sqrt(gamma_b)
            Left_aug = np.hstack([Z_B_w, s_b * C_B_between])
            Right_aug = np.hstack([Z_A_w, s_b * C_A_between])
            R_cand, _, _, _ = _polar_factor_lowrank_factors(Left_aug, Right_aug)
            return R_cand

        chosen_gamma = 0.0 if self.between_weight is None else self.between_weight
        if self.between_weight is None and self.test_mode == "global_mean" and X_A_val is not None and X_B_val is not None:
            Ava, _ = _to_f64_2d(X_A_val)
            Bva, _ = _to_f64_2d(X_B_val)
            ZA_va = (Ava - self.mu_A).T
            ZB_va = (Bva - self.mu_B).T
            norm_ZB_va = float(sla.norm(ZB_va, "fro"))
            best_mse = np.inf
            best_R = None
            for g_cand in (0.0, 0.03, 0.08, 0.15, 0.3):
                Rc = _fit_for_gamma(g_cand)
                RZA = Rc @ ZA_va
                sc = norm_ZB_va / max(float(sla.norm(RZA, "fro")), EPS)
                mse = float(np.mean((sc * RZA - ZB_va) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_R = Rc
                    chosen_gamma = g_cand
            assert best_R is not None
            R = best_R
        else:
            R = _fit_for_gamma(chosen_gamma)

        self.R = R
        if X_A_val is not None and X_B_val is not None:
            Ava, _ = _to_f64_2d(X_A_val)
            Bva, _ = _to_f64_2d(X_B_val)
            ZA_va = (Ava - self.mu_A).T
            ZB_va = (Bva - self.mu_B).T
            self.s_global = float(sla.norm(ZB_va, "fro")) / max(float(sla.norm(R @ ZA_va, "fro")), EPS)
            self.s_within = float(sla.norm(Z_B_w, "fro")) / max(float(sla.norm(R @ Z_A_w, "fro")), EPS)
            # Adjust within-relation out-of-span shrinkage by validation global ratio
            tr_glob_rat = float(sla.norm(Z_B_glob, "fro")) / max(float(sla.norm(R @ Z_A_glob, "fro")), EPS)
            self.s_within *= (self.s_global / max(tr_glob_rat, EPS))
        else:
            self.s_global = float(sla.norm(Z_B_glob, "fro")) / max(float(sla.norm(R @ Z_A_glob, "fro")), EPS)
            self.s_within = float(sla.norm(Z_B_w, "fro")) / max(float(sla.norm(R @ Z_A_w, "fro")), EPS)
        self.T_global = self.s_global * R
        self.T_within = self.s_within * R

        between_var_frac_B = 1.0 - float(np.sum(Z_B_w ** 2)) / max(float(np.sum(Z_B_glob ** 2)), EPS)
        between_var_frac_A = 1.0 - float(np.sum(Z_A_w ** 2)) / max(float(np.sum(Z_A_glob ** 2)), EPS)
        self.meta = {
            "num_relations": len(uniq_rels),
            "test_mode": self.test_mode,
            "gamma_between": chosen_gamma,
            "s_global": self.s_global,
            "s_within": self.s_within,
            "between_var_frac_A": between_var_frac_A,
            "between_var_frac_B": between_var_frac_B,
        }
        return self

    def _transform_f64(
        self,
        X_A: np.ndarray,
        X_B_orig: Optional[np.ndarray] = None,
        relations: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        assert self.T_global is not None and self.T_within is not None
        assert self.mu_A is not None and self.mu_B is not None

        if self.test_mode == "global_mean" or relations is None:
            return self.mu_B[None, :] + (X_A - self.mu_A[None, :]) @ self.T_global.T

        if len(relations) != X_A.shape[0]:
            raise ValueError(f"len(relations)={len(relations)} != batch={X_A.shape[0]}")

        out = np.empty((X_A.shape[0], self.mu_B.shape[0]), dtype=np.float64)
        rels_arr = np.asarray([str(r) for r in relations])
        for r in set(rels_arr):
            idx = np.where(rels_arr == r)[0]
            if r in self.rel_mu_A and r in self.rel_mu_B:
                mu_Ar = self.rel_mu_A[r]
                mu_Br = self.rel_mu_B[r]
                out[idx] = mu_Br[None, :] + (X_A[idx] - mu_Ar[None, :]) @ self.T_within.T
            else:
                out[idx] = self.mu_B[None, :] + (X_A[idx] - self.mu_A[None, :]) @ self.T_global.T
        return out


# ---------------------------------------------------------------------------
# 6. Low-Dimensional Compression: Separate-PCA vs. Cross-Covariance SVD (PLS)
#    + Surgical Residual-Preserving Subspace Injection
# ---------------------------------------------------------------------------

class SeparatePCAProcrustesMap(BaseKeyMap):
    """
    Plato's Key Fig. 4b Rank-k Compression Baseline (`pca_procrustes`):
    Projects A and B separately onto their top-k PCA subspaces `P_Ak` (d_A, k) and `P_Bk` (d_B, k),
    fits k x k orthogonal Procrustes `R_k` with `s_k = ||P_Bk^T Z_B||_F / ||P_Ak^T Z_A||_F`,
    and reconstructs in R^{d_B} via either:
      - `fill_mode="mean"` (exact Plato's Key Fig. 4b):
          `z_hat = mu_B + P_Bk (s_k R_k P_Ak^T (z_A - mu_A))`
          (sets recipient's remaining d_B - k components to their training mean).
      - `fill_mode="surgical"` (ablation control isolating subspace selection vs. injection):
          `z_hat = (I - P_Bk P_Bk^T) z_B_orig + P_Bk (P_Bk^T mu_B + s_k R_k P_Ak^T (z_A - mu_A))`.
    """

    def __init__(self, rank: int = 256, fill_mode: str = "mean"):
        if fill_mode not in ("mean", "surgical"):
            raise ValueError(f"Unknown fill_mode: {fill_mode}")
        self.rank = int(rank)
        self.fill_mode = fill_mode
        self.name = f"pca_{fill_mode}_k{self.rank}"
        self.mu_A: Optional[np.ndarray] = None
        self.mu_B: Optional[np.ndarray] = None
        self.P_Ak: Optional[np.ndarray] = None   # (d_A, k_eff)
        self.P_Bk: Optional[np.ndarray] = None   # (d_B, k_eff)
        self.R_k: Optional[np.ndarray] = None    # (k_eff, k_eff)
        self.s_k: float = 1.0
        self.meta: Dict[str, Any] = {}

    @staticmethod
    def _top_k_pca_basis(Z: np.ndarray, k: int) -> np.ndarray:
        """Return orthonormal columns (d, k) for centered column matrix Z (d, N)."""
        d, n = Z.shape
        full = k > min(d, n)
        try:
            U, _, _ = sla.svd(Z, full_matrices=full, lapack_driver="gesdd", check_finite=False)
        except sla.LinAlgError:
            U, _, _ = sla.svd(Z, full_matrices=full, lapack_driver="gesvd", check_finite=False)
        return np.ascontiguousarray(U[:, :k])

    def fit(self, X_A: ArrayLike, X_B: ArrayLike) -> "SeparatePCAProcrustesMap":
        A, _ = _to_f64_2d(X_A)
        B, _ = _to_f64_2d(X_B)
        self.mu_A = A.mean(axis=0)
        self.mu_B = B.mean(axis=0)
        Z_A = (A - self.mu_A).T
        Z_B = (B - self.mu_B).T
        k_eff = min(self.rank, Z_A.shape[0], Z_B.shape[0])

        self.P_Ak = self._top_k_pca_basis(Z_A, k_eff)
        self.P_Bk = self._top_k_pca_basis(Z_B, k_eff)
        return self._fit_from_bases(Z_A, Z_B, self.P_Ak, self.P_Bk)

    def _fit_from_bases(
        self,
        Z_A: np.ndarray,
        Z_B: np.ndarray,
        P_Ak: np.ndarray,
        P_Bk: np.ndarray,
    ) -> "SeparatePCAProcrustesMap":
        self.P_Ak = P_Ak
        self.P_Bk = P_Bk
        C_A = P_Ak.T @ Z_A  # (k, N)
        C_B = P_Bk.T @ Z_B  # (k, N)
        R_k, _, _, _ = _polar_factor(C_B @ C_A.T)
        self.R_k = R_k
        self.s_k = float(sla.norm(C_B, "fro")) / max(float(sla.norm(C_A, "fro")), EPS)
        self.meta = {
            "rank": P_Ak.shape[1],
            "fill_mode": self.fill_mode,
            "s_k": self.s_k,
            "var_explained_A": float(np.sum(C_A ** 2)) / max(float(np.sum(Z_A ** 2)), EPS),
            "var_explained_B": float(np.sum(C_B ** 2)) / max(float(np.sum(Z_B ** 2)), EPS),
        }
        return self

    def _transform_f64(
        self,
        X_A: np.ndarray,
        X_B_orig: Optional[np.ndarray] = None,
        relations: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        assert self.P_Ak is not None and self.P_Bk is not None and self.R_k is not None
        assert self.mu_A is not None and self.mu_B is not None
        # Mapped k-dim centered offset: (N, k)
        c_hat = self.s_k * ((X_A - self.mu_A[None, :]) @ self.P_Ak @ self.R_k.T)
        delta_sub = c_hat @ self.P_Bk.T  # (N, d_B) in col(P_Bk)

        if self.fill_mode == "mean" or X_B_orig is None:
            return self.mu_B[None, :] + delta_sub

        # Surgical: (I - P_Bk P_Bk^T) z_B_orig + P_Bk (P_Bk^T mu_B + c_hat)
        #         = z_B_orig - P_Bk P_Bk^T (z_B_orig - mu_B) + delta_sub
        proj_B = ((X_B_orig - self.mu_B[None, :]) @ self.P_Bk) @ self.P_Bk.T
        return X_B_orig - proj_B + delta_sub


class PLSProcrustesMap(BaseKeyMap):
    """
    Cross-Covariance SVD Rank-k Compression (`pls_procrustes`):
    Truncates `Z_B Z_A^T = U Sigma V^T` directly to its top-k shared singular modes `U_k, V_k`
    (avoiding spending rank budget on unshared model-specific nuisance variance!).

    Supports three reconstruction / injection modes:
      1. `fill_mode="mean"`:
           `z_hat = mu_B + U_k (s_k V_k^T (z_A - mu_A))`
           with `s_k = ||U_k^T Z_B||_F / ||V_k^T Z_A||_F` (or full-spread `||Z_B||_F / ||V_k^T Z_A||_F`
           when `boost_mean_norm=True`).
      2. `fill_mode="surgical"` (Surgical Residual-Preserving Injection, parameter-free beta=0):
           `z_hat = (I - U_k U_k^T) z_B_orig + U_k (mu_Bk + s_k V_k^T (z_A - mu_A))`
           where `mu_Bk = U_k^T mu_B` and `s_k = ||U_k^T Z_B||_F / ||V_k^T Z_A||_F`.
           By Pythagoras, preserves exact full-dimensional spread `||Z_hat - mu_B||_F == ||Z_B||_F`
           for EVERY rank k, while retaining recipient B's prompt/syntactic state in `col(U_k)^perp`!
      3. `fill_mode="soft_surgical"` (Validation-tuned orthogonal-complement attenuation `beta in [0, 1]`):
           `z_hat = mu_B + (1 - beta)(I - U_k U_k^T)(z_B_orig - mu_B) + U_k (s_k(beta) V_k^T (z_A - mu_A))`
           where `s_k(beta) = sqrt(||Z_B||_F^2 - (1 - beta)^2 ||(I - U_k U_k^T) Z_B||_F^2) / ||V_k^T Z_A||_F`
           preserves exact spread `||Z_hat - mu_B||_F == ||Z_B||_F` for all `beta in [0, 1]`
           (`beta=0` recovers `surgical`; `beta=1` recovers norm-restored `mean-fill`).
    """

    def __init__(
        self,
        rank: int = 256,
        fill_mode: str = "surgical",
        beta: float = 0.0,
        boost_mean_norm: bool = False,
    ):
        if fill_mode not in ("mean", "surgical", "soft_surgical"):
            raise ValueError(f"Unknown fill_mode: {fill_mode}")
        self.rank = int(rank)
        self.fill_mode = fill_mode
        self.beta = float(np.clip(beta, 0.0, 1.0))
        self.boost_mean_norm = bool(boost_mean_norm)
        suffix = f"_b{self.beta:g}" if fill_mode == "soft_surgical" else ""
        self.name = f"pls_{fill_mode}_k{self.rank}{suffix}"
        self.mu_A: Optional[np.ndarray] = None
        self.mu_B: Optional[np.ndarray] = None
        self.U_k: Optional[np.ndarray] = None  # (d_B, k_eff)
        self.V_k: Optional[np.ndarray] = None  # (d_A, k_eff)
        self.s_k: float = 1.0
        self.meta: Dict[str, Any] = {}

    def fit(self, X_A: ArrayLike, X_B: ArrayLike) -> "PLSProcrustesMap":
        A, _ = _to_f64_2d(X_A)
        B, _ = _to_f64_2d(X_B)
        mu_A = A.mean(axis=0)
        mu_B = B.mean(axis=0)
        Z_A = (A - mu_A).T
        Z_B = (B - mu_B).T
        _, U, s_vals, Vt = _polar_factor_lowrank_factors(Z_B, Z_A, min_rank=self.rank)
        return self._fit_from_cross_svd(mu_A, mu_B, Z_A, Z_B, U, s_vals, Vt)

    def _fit_from_cross_svd(
        self,
        mu_A: np.ndarray,
        mu_B: np.ndarray,
        Z_A: np.ndarray,
        Z_B: np.ndarray,
        U: np.ndarray,
        s_vals: np.ndarray,
        Vt: np.ndarray,
    ) -> "PLSProcrustesMap":
        k_eff = min(self.rank, U.shape[1])
        self.mu_A = mu_A
        self.mu_B = mu_B
        self.U_k = np.ascontiguousarray(U[:, :k_eff])
        self.V_k = np.ascontiguousarray(Vt[:k_eff, :].T)

        norm_B_sq = float(np.sum(Z_B ** 2))
        sub_B_sq = float(np.sum((self.U_k.T @ Z_B) ** 2))
        perp_B_sq = max(norm_B_sq - sub_B_sq, 0.0)
        sub_A_norm = max(float(sla.norm(self.V_k.T @ Z_A, "fro")), EPS)

        if self.fill_mode == "mean":
            numer = np.sqrt(norm_B_sq) if self.boost_mean_norm else np.sqrt(sub_B_sq)
            self.s_k = float(numer) / sub_A_norm
        elif self.fill_mode == "surgical":
            self.s_k = float(np.sqrt(sub_B_sq)) / sub_A_norm
        elif self.fill_mode == "soft_surgical":
            rem_B_sq = max(norm_B_sq - ((1.0 - self.beta) ** 2) * perp_B_sq, 0.0)
            self.s_k = float(np.sqrt(rem_B_sq)) / sub_A_norm

        self.meta = {
            "rank": k_eff,
            "fill_mode": self.fill_mode,
            "beta": self.beta,
            "s_k": self.s_k,
            "nuclear_captured_frac": float(np.sum(s_vals[:k_eff])) / max(float(np.sum(s_vals)), EPS),
            "subspace_var_frac_B": sub_B_sq / max(norm_B_sq, EPS),
        }
        return self

    def _transform_f64(
        self,
        X_A: np.ndarray,
        X_B_orig: Optional[np.ndarray] = None,
        relations: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        assert self.U_k is not None and self.V_k is not None
        assert self.mu_A is not None and self.mu_B is not None

        # Donor shared-subspace injection: U_k (s_k V_k^T (z_A - mu_A))  -> shape (N, d_B)
        delta_shared = (self.s_k * ((X_A - self.mu_A[None, :]) @ self.V_k)) @ self.U_k.T

        if self.fill_mode == "mean" or X_B_orig is None:
            return self.mu_B[None, :] + delta_shared

        # Centered recipient state & its orthogonal complement (I - U_k U_k^T)(z_B_orig - mu_B)
        z_B_centered = X_B_orig - self.mu_B[None, :]
        z_B_proj = (z_B_centered @ self.U_k) @ self.U_k.T
        z_B_perp = z_B_centered - z_B_proj

        if self.fill_mode == "surgical":
            # Exact identity: mu_B + z_B_perp + delta_shared
            #               = (I - U_k U_k^T) z_B_orig + U_k (U_k^T mu_B + s_k V_k^T (z_A - mu_A))
            return self.mu_B[None, :] + z_B_perp + delta_shared

        # soft_surgical
        return self.mu_B[None, :] + (1.0 - self.beta) * z_B_perp + delta_shared


def fit_compression_sweep(
    X_A_train: ArrayLike,
    X_B_train: ArrayLike,
    ranks: Sequence[int] = (8, 16, 32, 64, 128, 256, 512),
    soft_betas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> Dict[int, Dict[str, Any]]:
    """
    Fit ALL compression maps across `ranks` using a SINGLE cross-covariance SVD and
    SINGLE top-max(k) PCA per model — 10x faster than refitting per rank!

    Returns `sweep[k]` containing:
      - `"pca_mean"`: SeparatePCAProcrustesMap(k, "mean")
      - `"pca_surgical"`: SeparatePCAProcrustesMap(k, "surgical")
      - `"pls_mean"`: PLSProcrustesMap(k, "mean")
      - `"pls_mean_boosted"`: PLSProcrustesMap(k, "mean", boost_mean_norm=True)
      - `"pls_surgical"`: PLSProcrustesMap(k, "surgical")
      - `"pls_soft_candidates"`: list of PLSProcrustesMap(k, "soft_surgical", beta=b) for val tuning
    """
    A, _ = _to_f64_2d(X_A_train)
    B, _ = _to_f64_2d(X_B_train)
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    Z_A = (A - mu_A).T
    Z_B = (B - mu_B).T

    max_k = min(max(ranks), Z_A.shape[0], Z_B.shape[0])
    P_A_max = SeparatePCAProcrustesMap._top_k_pca_basis(Z_A, max_k)
    P_B_max = SeparatePCAProcrustesMap._top_k_pca_basis(Z_B, max_k)
    _, U, s_vals, Vt = _polar_factor_lowrank_factors(Z_B, Z_A, min_rank=max_k)

    out: Dict[int, Dict[str, Any]] = {}
    for k in ranks:
        k_eff = min(int(k), Z_A.shape[0], Z_B.shape[0])
        P_Ak = np.ascontiguousarray(P_A_max[:, :k_eff])
        P_Bk = np.ascontiguousarray(P_B_max[:, :k_eff])

        pca_mean = SeparatePCAProcrustesMap(rank=k, fill_mode="mean")
        pca_mean.mu_A, pca_mean.mu_B = mu_A, mu_B
        pca_mean._fit_from_bases(Z_A, Z_B, P_Ak, P_Bk)

        pca_surg = SeparatePCAProcrustesMap(rank=k, fill_mode="surgical")
        pca_surg.mu_A, pca_surg.mu_B = mu_A, mu_B
        pca_surg._fit_from_bases(Z_A, Z_B, P_Ak, P_Bk)

        pls_mean = PLSProcrustesMap(rank=k, fill_mode="mean", boost_mean_norm=False)._fit_from_cross_svd(
            mu_A, mu_B, Z_A, Z_B, U, s_vals, Vt
        )
        pls_mean_boost = PLSProcrustesMap(rank=k, fill_mode="mean", boost_mean_norm=True)._fit_from_cross_svd(
            mu_A, mu_B, Z_A, Z_B, U, s_vals, Vt
        )
        pls_surg = PLSProcrustesMap(rank=k, fill_mode="surgical")._fit_from_cross_svd(
            mu_A, mu_B, Z_A, Z_B, U, s_vals, Vt
        )
        soft_cands = [
            PLSProcrustesMap(rank=k, fill_mode="soft_surgical", beta=float(b))._fit_from_cross_svd(
                mu_A, mu_B, Z_A, Z_B, U, s_vals, Vt
            )
            for b in soft_betas
        ]

        out[int(k)] = {
            "pca_mean": pca_mean,
            "pca_surgical": pca_surg,
            "pls_mean": pls_mean,
            "pls_mean_boosted": pls_mean_boost,
            "pls_surgical": pls_surg,
            "pls_soft_candidates": soft_cands,
        }
    return out


# ---------------------------------------------------------------------------
# 7. Zero-Intervention Layer Selection Proxies (Vectorized O((L_A+L_B) N^2 d))
# ---------------------------------------------------------------------------

def _row_l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)
    return X / norms


def _knn_indicator_matrix(X_centered: np.ndarray, k_nn: int) -> np.ndarray:
    """
    Return binary directed k-NN matrix `A` of shape `(N, N)` (float64 0.0/1.0)
    under cosine similarity on centered row matrix `X_centered` (N, d), excluding self.
    """
    n = X_centered.shape[0]
    k_eff = min(max(int(k_nn), 1), max(n - 1, 1))
    Xn = _row_l2_normalize(X_centered)
    sim = Xn @ Xn.T  # (N, N)
    np.fill_diagonal(sim, -np.inf)
    # argpartition top k_eff per row
    nn_idx = np.argpartition(sim, -k_eff, axis=1)[:, -k_eff:]
    A = np.zeros((n, n), dtype=np.float64)
    rows = np.arange(n)[:, None]
    A[rows, nn_idx] = 1.0
    return A


def _within_relation_center(X: np.ndarray, relations: Sequence[str]) -> np.ndarray:
    rels = np.asarray(relations)
    out = np.empty_like(X)
    for r in set(relations):
        m = (rels == r)
        out[m] = X[m] - X[m].mean(axis=0, keepdims=True)
    return out


def _within_relation_knn_flat(X: np.ndarray, relations: Sequence[str], k_nn: int) -> Tuple[np.ndarray, float]:
    """
    Compute block-diagonal within-relation binary k-NN indicator vector so that
    dot(v_A, v_B) gives the exact relation-macro-averaged mutual k-NN overlap!
    """
    rels = np.asarray(relations)
    uniq = sorted(set(relations))
    parts_A = []
    n_valid_rels = 0
    for r in uniq:
        idx = np.where(rels == r)[0]
        nr = len(idx)
        if nr < 3:
            continue
        k_r = min(int(k_nn), max(1, nr - 1), max(2, nr // 3))
        Sub = X[idx] - X[idx].mean(axis=0, keepdims=True)
        Ar = _knn_indicator_matrix(Sub, k_r) / np.sqrt(float(nr * k_r))
        parts_A.append(Ar.reshape(-1))
        n_valid_rels += 1
    if not parts_A:
        raise ValueError("Need at least one relation with >= 3 examples for within-relation mKNN.")
    vec = np.concatenate(parts_A, axis=0) / np.sqrt(float(n_valid_rels))
    return vec, float(n_valid_rels)


def _normalized_gram_flat(X_centered: np.ndarray, debiased: bool = True) -> np.ndarray:
    """
    Return flattened normalized Gram matrix `K / ||K||_F` of shape `(N*N,)` so that
    `dot(g_A, g_B) == LinearCKA(X_A, X_B)`!
    Supports unbiased HSIC/CKA centering (`debiased=True`, Kornblith et al. 2019) as well
    as standard centered Linear CKA (`debiased=False`).
    """
    K = X_centered @ X_centered.T
    n = K.shape[0]
    if debiased and n > 3:
        np.fill_diagonal(K, 0.0)
        row_sums = np.sum(K, axis=1)
        total_sum = float(np.sum(row_sums))
        # Song et al. / Kornblith et al. unbiased HSIC matrix:
        K_tilde = (
            K
            - (row_sums[:, None] + row_sums[None, :]) / (n - 2.0)
            + total_sum / ((n - 1.0) * (n - 2.0))
        )
        np.fill_diagonal(K_tilde, 0.0)
        nrm = max(float(sla.norm(K_tilde, "fro")), EPS)
        return (K_tilde / nrm).reshape(-1)
    nrm = max(float(sla.norm(K, "fro")), EPS)
    return (K / nrm).reshape(-1)


def compute_layer_selection_grids(
    H_A_direct: ArrayLike,
    H_B_direct: ArrayLike,
    relations: Sequence[str],
    H_A_para: Optional[ArrayLike] = None,
    H_B_para: Optional[ArrayLike] = None,
    H_A_last: Optional[ArrayLike] = None,
    H_B_last: Optional[ArrayLike] = None,
    k_nn: int = 10,
    exclude_embedding_layer0: bool = False,
) -> Dict[str, Any]:
    """
    Compute ALL zero-intervention `(l_A, l_B)` layer-selection proxy grids across every
    layer pair of donor A (`L_A` layers) and recipient B (`L_B` layers) in milliseconds!

    Accepts `(L, N, d)` arrays/tensors (note: if cache stores `(N, L, d)`, pass `H.transpose(0, 1)`).
    Includes:
      - `raw_mknn_uncentered` (exact match to `extract_and_baseline.compute_mknn`)
      - `raw_mknn` (centered cosine mKNN)
      - `within_relation_mknn` (removes prompt-template clustering)
      - `paraphrase_entity_mknn` & `within_rel_paraphrase_mknn` (cross-template entity invariance)
      - `linear_cka` & `within_relation_cka`
      - When `H_A_last` & `H_B_last` (`(L, N, d)` static last-prompt-token states) are supplied:
        parameter-free zero-intervention mechanistic key-window gates `phi_key(l) * psi_unread(l)`
        (`gated_within_rel_mknn`, `gated_paraphrase_mknn`, `gated_within_rel_cka`), where
        `phi_key(l) = 1 - CKA_wrel(H_subj[l], H_subj[0])` measures contextualization beyond raw
        token embeddings and `psi_unread(l) = 1 - CKA_wrel(H_last[l], H_last[L-1])` measures
        remaining causal capacity before the last token commits to the final answer!
    """
    if _HAS_TORCH and isinstance(H_A_direct, torch.Tensor):
        HA = H_A_direct.detach().cpu().to(torch.float64).numpy()
    else:
        HA = np.asarray(H_A_direct, dtype=np.float64)

    if _HAS_TORCH and isinstance(H_B_direct, torch.Tensor):
        HB = H_B_direct.detach().cpu().to(torch.float64).numpy()
    else:
        HB = np.asarray(H_B_direct, dtype=np.float64)

    L_A, N, _ = HA.shape
    L_B, N_B, _ = HB.shape
    if N != N_B or len(relations) != N:
        raise ValueError(f"Mismatched sample count: HA={N}, HB={N_B}, relations={len(relations)}")

    k_eff = min(max(int(k_nn), 1), max(N - 1, 1))

    raw_unc_A = np.empty((L_A, N * N), dtype=np.float64)
    raw_unc_B = np.empty((L_B, N * N), dtype=np.float64)
    raw_knn_A = np.empty((L_A, N * N), dtype=np.float64)
    raw_knn_B = np.empty((L_B, N * N), dtype=np.float64)
    wrel_knn_A_list: List[np.ndarray] = []
    wrel_knn_B_list: List[np.ndarray] = []
    cka_A = np.empty((L_A, N * N), dtype=np.float64)
    cka_B = np.empty((L_B, N * N), dtype=np.float64)
    wrel_cka_A = np.empty((L_A, N * N), dtype=np.float64)
    wrel_cka_B = np.empty((L_B, N * N), dtype=np.float64)

    for la in range(L_A):
        Xa_c = HA[la] - HA[la].mean(axis=0, keepdims=True)
        Xa_w = _within_relation_center(HA[la], relations)
        raw_unc_A[la] = _knn_indicator_matrix(HA[la], k_eff).reshape(-1) / np.sqrt(float(N * k_eff))
        raw_knn_A[la] = _knn_indicator_matrix(Xa_c, k_eff).reshape(-1) / np.sqrt(float(N * k_eff))
        vec_wa, _ = _within_relation_knn_flat(HA[la], relations, k_eff)
        wrel_knn_A_list.append(vec_wa)
        cka_A[la] = _normalized_gram_flat(Xa_c, debiased=False)
        wrel_cka_A[la] = _normalized_gram_flat(Xa_w, debiased=False)

    for lb in range(L_B):
        Xb_c = HB[lb] - HB[lb].mean(axis=0, keepdims=True)
        Xb_w = _within_relation_center(HB[lb], relations)
        raw_unc_B[lb] = _knn_indicator_matrix(HB[lb], k_eff).reshape(-1) / np.sqrt(float(N * k_eff))
        raw_knn_B[lb] = _knn_indicator_matrix(Xb_c, k_eff).reshape(-1) / np.sqrt(float(N * k_eff))
        vec_wb, _ = _within_relation_knn_flat(HB[lb], relations, k_eff)
        wrel_knn_B_list.append(vec_wb)
        cka_B[lb] = _normalized_gram_flat(Xb_c, debiased=False)
        wrel_cka_B[lb] = _normalized_gram_flat(Xb_w, debiased=False)

    wrel_knn_A = np.stack(wrel_knn_A_list, axis=0)
    wrel_knn_B = np.stack(wrel_knn_B_list, axis=0)

    grids: Dict[str, np.ndarray] = {
        "raw_mknn_uncentered": raw_unc_A @ raw_unc_B.T,
        "raw_mknn": raw_knn_A @ raw_knn_B.T,
        "within_relation_mknn": wrel_knn_A @ wrel_knn_B.T,
        "linear_cka": cka_A @ cka_B.T,
        "within_relation_cka": wrel_cka_A @ wrel_cka_B.T,
    }

    # Paraphrase-invariant cross-template entity mKNN & CKA (when paraphrase caches provided)
    if H_B_para is not None:
        if _HAS_TORCH and isinstance(H_B_para, torch.Tensor):
            HB_p = H_B_para.detach().cpu().to(torch.float64).numpy()
        else:
            HB_p = np.asarray(H_B_para, dtype=np.float64)

        para_knn_B = np.empty((L_B, N * N), dtype=np.float64)
        para_wrel_B_list: List[np.ndarray] = []
        para_cka_B = np.empty((L_B, N * N), dtype=np.float64)
        para_wrel_cka_B = np.empty((L_B, N * N), dtype=np.float64)
        for lb in range(L_B):
            Xbp_c = HB_p[lb] - HB_p[lb].mean(axis=0, keepdims=True)
            Xbp_w = _within_relation_center(HB_p[lb], relations)
            para_knn_B[lb] = _knn_indicator_matrix(Xbp_c, k_eff).reshape(-1) / np.sqrt(float(N * k_eff))
            vw_bp, _ = _within_relation_knn_flat(HB_p[lb], relations, k_eff)
            para_wrel_B_list.append(vw_bp)
            para_cka_B[lb] = _normalized_gram_flat(Xbp_c, debiased=False)
            para_wrel_cka_B[lb] = _normalized_gram_flat(Xbp_w, debiased=False)
        para_wrel_B = np.stack(para_wrel_B_list, axis=0)

        if H_A_para is not None:
            if _HAS_TORCH and isinstance(H_A_para, torch.Tensor):
                HA_p = H_A_para.detach().cpu().to(torch.float64).numpy()
            else:
                HA_p = np.asarray(H_A_para, dtype=np.float64)
            para_knn_A = np.empty((L_A, N * N), dtype=np.float64)
            para_wrel_A_list: List[np.ndarray] = []
            para_cka_A = np.empty((L_A, N * N), dtype=np.float64)
            para_wrel_cka_A = np.empty((L_A, N * N), dtype=np.float64)
            for la in range(L_A):
                Xap_c = HA_p[la] - HA_p[la].mean(axis=0, keepdims=True)
                Xap_w = _within_relation_center(HA_p[la], relations)
                para_knn_A[la] = _knn_indicator_matrix(Xap_c, k_eff).reshape(-1) / np.sqrt(float(N * k_eff))
                vw_ap, _ = _within_relation_knn_flat(HA_p[la], relations, k_eff)
                para_wrel_A_list.append(vw_ap)
                para_cka_A[la] = _normalized_gram_flat(Xap_c, debiased=False)
                para_wrel_cka_A[la] = _normalized_gram_flat(Xap_w, debiased=False)
            para_wrel_A = np.stack(para_wrel_A_list, axis=0)
            # Symmetric cross-template mKNN & CKA: 0.5 * (sim(A_dir, B_para) + sim(A_para, B_dir))
            grids["paraphrase_entity_mknn"] = 0.5 * (
                (raw_knn_A @ para_knn_B.T) + (para_knn_A @ raw_knn_B.T)
            )
            grids["within_rel_paraphrase_mknn"] = 0.5 * (
                (wrel_knn_A @ para_wrel_B.T) + (para_wrel_A @ wrel_knn_B.T)
            )
            grids["paraphrase_cka"] = 0.5 * (
                (cka_A @ para_cka_B.T) + (para_cka_A @ cka_B.T)
            )
            grids["within_rel_paraphrase_cka"] = 0.5 * (
                (wrel_cka_A @ para_wrel_cka_B.T) + (para_wrel_cka_A @ wrel_cka_B.T)
            )
        else:
            grids["paraphrase_entity_mknn"] = raw_knn_A @ para_knn_B.T
            grids["within_rel_paraphrase_mknn"] = wrel_knn_A @ para_wrel_B.T
            grids["paraphrase_cka"] = cka_A @ para_cka_B.T
            grids["within_rel_paraphrase_cka"] = wrel_cka_A @ para_wrel_cka_B.T

    # Parameter-free zero-intervention mechanistic key-window gating (when H_A_last & H_B_last provided):
    #   phi_key(l)    = 1 - CKA_wrel(H_subj[l], H_subj[0])     (contextualization away from raw token embedding)
    #   psi_unread(l) = 1 - CKA_wrel(H_last[l], H_last[L-1])   (uncommitted capacity before t_last holds final answer)
    if H_A_last is not None and H_B_last is not None:
        if _HAS_TORCH and isinstance(H_A_last, torch.Tensor):
            HA_l = H_A_last.detach().cpu().to(torch.float64).numpy()
        else:
            HA_l = np.asarray(H_A_last, dtype=np.float64)
        if _HAS_TORCH and isinstance(H_B_last, torch.Tensor):
            HB_l = H_B_last.detach().cpu().to(torch.float64).numpy()
        else:
            HB_l = np.asarray(H_B_last, dtype=np.float64)

        gA_last_final = _normalized_gram_flat(_within_relation_center(HA_l[L_A - 1], relations), debiased=False)
        gB_last_final = _normalized_gram_flat(_within_relation_center(HB_l[L_B - 1], relations), debiased=False)

        gate_A = np.empty(L_A, dtype=np.float64)
        for la in range(L_A):
            phi_a = max(0.0, 1.0 - float(np.dot(wrel_cka_A[la], wrel_cka_A[0])))
            gl_a = _normalized_gram_flat(_within_relation_center(HA_l[la], relations), debiased=False)
            psi_a = max(0.0, 1.0 - float(np.dot(gl_a, gA_last_final)))
            gate_A[la] = phi_a * psi_a

        gate_B = np.empty(L_B, dtype=np.float64)
        for lb in range(L_B):
            phi_b = max(0.0, 1.0 - float(np.dot(wrel_cka_B[lb], wrel_cka_B[0])))
            gl_b = _normalized_gram_flat(_within_relation_center(HB_l[lb], relations), debiased=False)
            psi_b = max(0.0, 1.0 - float(np.dot(gl_b, gB_last_final)))
            gate_B[lb] = phi_b * psi_b

        gate_2d = gate_A[:, None] * gate_B[None, :]
        grids["gated_within_rel_mknn"] = gate_2d * grids["within_relation_mknn"]
        grids["gated_within_rel_cka"] = gate_2d * grids["within_relation_cka"]
        if "within_rel_paraphrase_mknn" in grids:
            grids["gated_paraphrase_mknn"] = gate_2d * grids["within_rel_paraphrase_mknn"]

    selected_layers: Dict[str, Tuple[int, int]] = {}
    selected_scores: Dict[str, float] = {}
    start_a = 1 if (exclude_embedding_layer0 and L_A > 1) else 0
    start_b = 1 if (exclude_embedding_layer0 and L_B > 1) else 0

    for name, G in grids.items():
        sub = G[start_a:, start_b:]
        flat_idx = int(np.argmax(sub))
        ia, ib = divmod(flat_idx, sub.shape[1])
        la_best, lb_best = start_a + ia, start_b + ib
        selected_layers[name] = (int(la_best), int(lb_best))
        selected_scores[name] = float(G[la_best, lb_best])

    return {
        "grids": grids,
        "selected_layers": selected_layers,
        "selected_scores": selected_scores,
    }


# ---------------------------------------------------------------------------
# Convenience Factory Function
# ---------------------------------------------------------------------------

def fit_map(
    name: str,
    X_A_train: ArrayLike,
    X_B_train: ArrayLike,
    *,
    X_A_val: Optional[ArrayLike] = None,
    X_B_val: Optional[ArrayLike] = None,
    relations_train: Optional[Sequence[str]] = None,
    gamma_B: Optional[ArrayLike] = None,
    gamma_A: Optional[ArrayLike] = None,
    U_read: Optional[ArrayLike] = None,
    s_read: Optional[ArrayLike] = None,
    gram_B: Optional[ArrayLike] = None,
    rank: int = 256,
    alpha: float = 1.0,
    norm_mode: str = "centered_match",
    beta: float = 0.0,
    seed: int = 0,
    val_eval_fn: Optional[Callable[[Any], float]] = None,
) -> BaseKeyMap:
    """Unified factory for fitting any baseline or improved closed-form alignment map."""
    key = name.lower().strip()
    if key == "procrustes_global":
        return ProcrustesGlobalMap(post_proj_norm=False).fit(X_A_train, X_B_train)
    if key == "procrustes_span":
        return SpanProcrustesMap(calibrate_val_norm=False).fit(X_A_train, X_B_train)
    if key == "procrustes_span_valcal":
        return SpanProcrustesMap(calibrate_val_norm=True).fit(
            X_A_train, X_B_train, X_A_val=X_A_val, X_B_val=X_B_val
        )
    if key == "procrustes_postproj":
        return ProcrustesGlobalMap(post_proj_norm=True).fit(X_A_train, X_B_train)
    if key == "procrustes_ls":
        return ProcrustesLSMap().fit(X_A_train, X_B_train)
    if key == "random_orthogonal":
        return RandomControlMap(seed=seed).fit(X_A_train, X_B_train)
    if key == "shuffled_procrustes":
        return ShuffledProcrustesMap(seed=seed).fit(X_A_train, X_B_train)
    if key in ("ridge_raw", "ridge_rescaled"):
        rescaled = (key == "ridge_rescaled")
        if X_A_val is not None and X_B_val is not None:
            return RidgeMap.fit_tuned(
                X_A_train, X_B_train, X_A_val, X_B_val, rescaled=rescaled, val_eval_fn=val_eval_fn
            )
        return RidgeMap(lam_rel=1.0, rescaled=rescaled).fit(X_A_train, X_B_train)
    if key == "vecmap_whitened":
        if X_A_val is not None and X_B_val is not None:
            return VecMapWhitenedMap.fit_tuned(
                X_A_train, X_B_train, X_A_val, X_B_val, val_eval_fn=val_eval_fn
            )
        return VecMapWhitenedMap().fit(X_A_train, X_B_train)
    if key == "monge_procrustes":
        if X_A_val is not None and X_B_val is not None:
            return MongeProcrustesMap.fit_tuned(
                X_A_train, X_B_train, X_A_val, X_B_val, val_eval_fn=val_eval_fn
            )
        return MongeProcrustesMap().fit(X_A_train, X_B_train)
    if key == "rmsnorm_weighted_procrustes":
        if gamma_B is None:
            raise ValueError("rmsnorm_weighted_procrustes requires gamma_B")
        if X_A_val is not None and X_B_val is not None:
            return RMSNormWeightedProcrustesMap.fit_tuned(
                X_A_train, X_B_train, X_A_val, X_B_val, gamma_B=gamma_B, gamma_A=gamma_A, val_eval_fn=val_eval_fn
            )
        return RMSNormWeightedProcrustesMap().fit(X_A_train, X_B_train, gamma_B=gamma_B, gamma_A=gamma_A)
    if key == "readout_weighted_procrustes":
        if X_A_val is not None and X_B_val is not None:
            return ReadoutWeightedProcrustesMap.fit_tuned(
                X_A_train, X_B_train, X_A_val, X_B_val, U_read=U_read, s_read=s_read, gram_B=gram_B, val_eval_fn=val_eval_fn
            )
        return ReadoutWeightedProcrustesMap().fit(
            X_A_train, X_B_train, U_read=U_read, s_read=s_read, gram_B=gram_B
        )
    if key == "per_sample_norm_and_gain":
        if val_eval_fn is not None:
            return PerSampleNormGainMap.fit_tuned(X_A_train, X_B_train, val_eval_fn=val_eval_fn)
        return PerSampleNormGainMap(norm_mode=norm_mode, alpha=alpha).fit(X_A_train, X_B_train)
    if key in ("relation_centered_procrustes", "relation_centered_global"):
        if relations_train is None:
            raise ValueError("relation_centered_procrustes requires relations_train")
        return RelationCenteredProcrustesMap(test_mode="global_mean").fit(
            X_A_train, X_B_train, relations=relations_train
        )
    if key == "relation_centered_relation":
        if relations_train is None:
            raise ValueError("relation_centered_relation requires relations_train")
        return RelationCenteredProcrustesMap(test_mode="relation_mean").fit(
            X_A_train, X_B_train, relations=relations_train
        )
    if key == "pca_mean":
        return SeparatePCAProcrustesMap(rank=rank, fill_mode="mean").fit(X_A_train, X_B_train)
    if key == "pca_surgical":
        return SeparatePCAProcrustesMap(rank=rank, fill_mode="surgical").fit(X_A_train, X_B_train)
    if key == "pls_mean":
        return PLSProcrustesMap(rank=rank, fill_mode="mean").fit(X_A_train, X_B_train)
    if key == "pls_surgical":
        return PLSProcrustesMap(rank=rank, fill_mode="surgical").fit(X_A_train, X_B_train)
    if key == "pls_soft_surgical":
        return PLSProcrustesMap(rank=rank, fill_mode="soft_surgical", beta=beta).fit(X_A_train, X_B_train)
    raise ValueError(f"Unknown map name: {name}")
