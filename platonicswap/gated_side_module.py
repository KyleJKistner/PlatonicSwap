#!/usr/bin/env python3
"""
gated_side_module.py — Non-Destructive Gated Native-Key Side-Module for
Donor-Free Cross-Family Knowledge Compilation & Modular Multi-Donor Hot-Swap.

Architecture:
  - Leaves 100% of Recipient B's base weights (`MLP_orig`, attention, norms, lm_head) FROZEN.
  - Reads `x_t` = exact `post_attention_layernorm` input to `layers_B[l_B].mlp` at EVERY
    real token `t` of every prompt (ZERO oracle position mask!).
  - Computes shrinkage-whitened normalized keys `z_hat = unit((x_t - mu_bg) @ W_white)`
    using background covariance `C_bg` fitted STRICTLY on `dataset.json` train prompts.
  - Each modular bank `b` (compiled independently from one donor, e.g. 32 Qwen entity->guild
    neurons OR 8 Falcon guild->color neurons) stores:
      * `U_gate`: `(M, D)` unit-normalized whitened train prototypes for task+entity gating,
      * `D_route`: `(M, D)` mean-subtracted unit-normalized prototypes for pure entity/guild routing,
      * `theta`: single scalar threshold calibrated STRICTLY on train prompts vs `X_bg_train`
        (0% train background FPR, zero test peeking!),
      * `V`: `(M, D)` template-averaged transported donor payloads (`pm.transform_delta(d_donor)`).
  - At inference (with ALL donors completely unloaded from memory):
      If `max_m <u_m, z_hat_t> > theta`, adds `alpha * V[argmax_m <d_m, z_hat_t>]` into
      the residual stream at `layers_B[l_B]`.
  - Records exact per-token, per-prompt, and last-token firing telemetry per bank!
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from extract_and_baseline import get_decoder_layers


@torch.no_grad()
def extract_post_attn_ln_inputs(
    model: torch.nn.Module,
    tok: Any,
    prompts: Sequence[str],
    layer_idx: int,
    device: torch.device,
    all_tokens: bool = False,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract exact `post_attention_layernorm` inputs to `layers[layer_idx].mlp`.

    If `all_tokens=True`, returns ALL real (non-padding) tokens across all prompts `(N_tok, D)`.
    If `all_tokens=False`, returns the final token `x[:, -1, :]` of each prompt `(N_prompts, D)`.
    """
    layers = get_decoder_layers(model)
    buf: Dict[str, torch.Tensor] = {}

    def _pre_hook(_mod, args):
        buf["x"] = args[0].detach().float().cpu()

    handle = layers[layer_idx].mlp.register_forward_pre_hook(_pre_hook)
    outs: List[np.ndarray] = []
    try:
        for start in range(0, len(prompts), batch_size):
            batch = list(prompts[start : start + batch_size])
            enc = tok(batch, return_tensors="pt", padding=True).to(device)
            _ = model(**enc, use_cache=False)
            x = buf["x"]  # (B, T, D)
            if all_tokens:
                mask = enc["attention_mask"].detach().cpu().bool()  # (B, T)
                outs.append(x[mask].numpy())
            else:
                outs.append(x[:, -1, :].numpy())
    finally:
        handle.remove()
    return np.concatenate(outs, axis=0)


def fit_bg_whitener(
    X_bg_all: np.ndarray,
    shrink_rel: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit symmetric shrinkage ZCA whitener `W_white = (C_bg + lam I)^{-1/2}` on CPU float64."""
    X = np.asarray(X_bg_all, dtype=np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(len(X), 1)
    lam = float(shrink_rel) * float(np.trace(cov)) / float(cov.shape[0])
    cov_reg = cov + lam * np.eye(cov.shape[0], dtype=np.float64)
    evals, evecs = np.linalg.eigh(cov_reg)
    evals = np.maximum(evals, 1e-12)
    W_white = (evecs * (1.0 / np.sqrt(evals))[None, :]) @ evecs.T
    return mu, W_white


def _whiten_unit(X: np.ndarray, mu: np.ndarray, W_white: np.ndarray) -> np.ndarray:
    Z = (np.asarray(X, dtype=np.float64) - mu[None, :]) @ W_white
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(norms, 1e-12)


@dataclass
class GatedNativeKeyBank:
    """One modular donor-free bank of `M` gated native-key neurons."""

    name: str
    U_gate: np.ndarray          # (M, D) float64 unit-normalized whitened prototypes
    D_route: np.ndarray         # (M, D) float64 mean-subtracted unit-normalized routing keys
    V_payload: np.ndarray       # (M, D) float64 template-averaged transported donor deltas
    theta: float                # calibrated scalar gate threshold
    alpha: float                # payload gain multiplier
    calibration: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def compile(
        cls,
        name: str,
        X_tr_last: np.ndarray,
        V_tr_deltas: np.ndarray,
        entity_ids: Sequence[int],
        n_entities: int,
        X_bg_all: np.ndarray,
        mu_bg: np.ndarray,
        W_white: np.ndarray,
        alpha: float = 2.0,
        margin_frac: float = 0.35,
        center_payloads: bool = False,
    ) -> "GatedNativeKeyBank":
        """Compile a modular gated bank STRICTLY from training prompts + background train tokens."""
        Z_tr = _whiten_unit(X_tr_last, mu_bg, W_white)       # (N_tr, D)
        Z_bg = _whiten_unit(X_bg_all, mu_bg, W_white)        # (N_bg, D)
        eids = np.asarray(entity_ids, dtype=np.int64)
        V_tr = np.asarray(V_tr_deltas, dtype=np.float64)
        D = Z_tr.shape[1]

        U_gate = np.zeros((n_entities, D), dtype=np.float64)
        V_proto = np.zeros((n_entities, D), dtype=np.float64)
        for m in range(n_entities):
            idx = np.where(eids == m)[0]
            assert len(idx) > 0, f"No training rows for prototype {m} in bank {name}"
            u = Z_tr[idx].mean(axis=0)
            U_gate[m] = u / max(float(np.linalg.norm(u)), 1e-12)
            V_proto[m] = V_tr[idx].mean(axis=0)
        if center_payloads and n_entities > 1:
            V_proto = V_proto - V_proto.mean(axis=0, keepdims=True)

        # Mean-subtracted discriminative routing keys (removes shared task/template offset!)
        u_mean = U_gate.mean(axis=0, keepdims=True)
        D_raw = U_gate - u_mean
        D_route = D_raw / np.maximum(np.linalg.norm(D_raw, axis=1, keepdims=True), 1e-12)

        # Calibrate scalar gate threshold strictly on Z_tr vs Z_bg (ZERO test data!)
        s_pos = (Z_tr @ U_gate.T).max(axis=1)
        s_neg = (Z_bg @ U_gate.T).max(axis=1)
        pos_min = float(s_pos.min())
        pos_p05 = float(np.quantile(s_pos, 0.05))
        pos_mean = float(s_pos.mean())
        neg_max = float(s_neg.max())
        neg_p999 = float(np.quantile(s_neg, 0.999))
        neg_p99 = float(np.quantile(s_neg, 0.99))

        if pos_min > neg_max:
            theta = float((1.0 - margin_frac) * neg_max + margin_frac * pos_min)
            calib_mode = "clean_gap_max_neg_to_min_pos"
        elif pos_p05 > neg_p999:
            theta = float((1.0 - margin_frac) * neg_p999 + margin_frac * pos_p05)
            calib_mode = "p999_neg_to_p05_pos"
        else:
            theta = float(neg_p999 + 0.02)
            calib_mode = "fallback_p999_plus_0.02"

        train_tpr = float((s_pos > theta).mean())
        train_bg_fpr = float((s_neg > theta).mean())
        # Also verify train routing accuracy
        route_preds = (Z_tr @ D_route.T).argmax(axis=1)
        train_route_acc = float((route_preds == eids).mean())

        calib = {
            "n_prototypes": int(n_entities),
            "n_train_rows": int(len(Z_tr)),
            "n_bg_tokens": int(len(Z_bg)),
            "calib_mode": calib_mode,
            "s_pos_min": pos_min,
            "s_pos_p05": pos_p05,
            "s_pos_mean": pos_mean,
            "s_neg_max": neg_max,
            "s_neg_p999": neg_p999,
            "s_neg_p99": neg_p99,
            "gap_pos_min_minus_neg_max": pos_min - neg_max,
            "theta": theta,
            "train_gate_tpr": train_tpr,
            "train_bg_token_fpr": train_bg_fpr,
            "train_route_acc": train_route_acc,
        }
        return cls(
            name=name,
            U_gate=U_gate,
            D_route=D_route,
            V_payload=V_proto,
            theta=theta,
            alpha=float(alpha),
            calibration=calib,
        )

    def with_alpha(self, alpha: float) -> "GatedNativeKeyBank":
        return GatedNativeKeyBank(
            name=self.name,
            U_gate=self.U_gate,
            D_route=self.D_route,
            V_payload=self.V_payload,
            theta=self.theta,
            alpha=float(alpha),
            calibration=dict(self.calibration),
        )


class GatedSideModuleMount:
    """Context manager mounting 1 or more modular `GatedNativeKeyBank`s at `layers_B[l_B]`.

    - Runs identically on EVERY real token of every prompt (ZERO oracle position mask!).
    - Leaves 100% of Recipient B's base weights untouched.
    - Tracks exact firing telemetry per bank (real tokens, prompts, and last-token positions).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer_idx: int,
        mu_bg: np.ndarray,
        W_white: np.ndarray,
        banks: Sequence[GatedNativeKeyBank],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        compete_across_banks: bool = False,
    ):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.device = device
        self.dtype = dtype
        self.compete_across_banks = bool(compete_across_banks)
        self.mu_bg_t = torch.from_numpy(mu_bg.astype(np.float32)).to(device=device, dtype=dtype)
        self.W_white_t = torch.from_numpy(W_white.astype(np.float32)).to(device=device, dtype=dtype)
        self.banks_t: List[Dict[str, Any]] = []
        for b in banks:
            self.banks_t.append({
                "name": b.name,
                "U": torch.from_numpy(b.U_gate.astype(np.float32)).to(device=device, dtype=dtype),
                "D": torch.from_numpy(b.D_route.astype(np.float32)).to(device=device, dtype=dtype),
                "V": torch.from_numpy((b.alpha * b.V_payload).astype(np.float32)).to(device=device, dtype=dtype),
                "theta": float(b.theta),
            })
        self._handles: List[Any] = []
        self._buf: Dict[str, Any] = {}
        self.reset_telemetry()

    def reset_telemetry(self) -> None:
        self.telemetry: Dict[str, Any] = {
            "total_real_tokens": 0,
            "total_prompts": 0,
            "per_bank": {
                b["name"]: {
                    "fired_real_tokens": 0,
                    "fired_prompts_any_tok": 0,
                    "fired_last_tok": 0,
                }
                for b in self.banks_t
            },
            "any_bank_fired_real_tokens": 0,
            "any_bank_fired_prompts": 0,
            "any_bank_fired_last_tok": 0,
        }

    def get_telemetry_summary(self) -> Dict[str, Any]:
        tot_tok = max(self.telemetry["total_real_tokens"], 1)
        tot_pr = max(self.telemetry["total_prompts"], 1)
        out = {
            "total_real_tokens": self.telemetry["total_real_tokens"],
            "total_prompts": self.telemetry["total_prompts"],
            "any_bank_token_fire_rate": self.telemetry["any_bank_fired_real_tokens"] / tot_tok,
            "any_bank_prompt_fire_rate": self.telemetry["any_bank_fired_prompts"] / tot_pr,
            "any_bank_last_tok_fire_rate": self.telemetry["any_bank_fired_last_tok"] / tot_pr,
            "per_bank": {},
        }
        for bname, d in self.telemetry["per_bank"].items():
            out["per_bank"][bname] = {
                "token_fire_rate": d["fired_real_tokens"] / tot_tok,
                "prompt_fire_rate": d["fired_prompts_any_tok"] / tot_pr,
                "last_tok_fire_rate": d["fired_last_tok"] / tot_pr,
                "fired_real_tokens": d["fired_real_tokens"],
                "fired_prompts_any_tok": d["fired_prompts_any_tok"],
                "fired_last_tok": d["fired_last_tok"],
            }
        return out

    def __enter__(self) -> "GatedSideModuleMount":
        layers = get_decoder_layers(self.model)

        def _model_pre_hook(_mod, args, kwargs):
            mask = kwargs.get("attention_mask", None)
            self._buf["attn_mask"] = mask

        def _mlp_pre_hook(_mod, args):
            self._buf["x_in"] = args[0]  # (B, T, D)

        def _layer_post_hook(_mod, _inp, out):
            x_in = self._buf.get("x_in", None)
            if x_in is None or not self.banks_t:
                return out
            h = out[0] if isinstance(out, tuple) else out  # (B, T, D)
            B, T, D = x_in.shape
            attn_mask = self._buf.get("attn_mask", None)
            if attn_mask is not None and attn_mask.shape == (B, T):
                real_mask = attn_mask.bool()
            else:
                real_mask = torch.ones((B, T), dtype=torch.bool, device=x_in.device)

            # Whiten & normalize EVERY token (no oracle position mask!)
            z = (x_in - self.mu_bg_t) @ self.W_white_t
            z_hat = z / torch.clamp(torch.linalg.norm(z, dim=-1, keepdim=True), min=1e-8)

            delta_total = torch.zeros_like(h)
            margins = []
            fires = []
            deltas = []
            for b in self.banks_t:
                gate_s = torch.matmul(z_hat, b["U"].T).max(dim=-1).values  # (B, T)
                margin = gate_s - b["theta"]
                fired = (margin > 0.0) & real_mask  # (B, T)
                route_idx = torch.matmul(z_hat, b["D"].T).argmax(dim=-1)  # (B, T)
                payload = b["V"][route_idx]  # (B, T, D)
                margins.append(margin)
                fires.append(fired)
                deltas.append(payload)

            if self.compete_across_banks and len(self.banks_t) > 1:
                m_stack = torch.stack(margins, dim=-1)  # (B, T, K)
                winner = m_stack.argmax(dim=-1)         # (B, T)
                for bi in range(len(self.banks_t)):
                    fires[bi] = fires[bi] & (winner == bi)

            any_fired = torch.zeros((B, T), dtype=torch.bool, device=x_in.device)
            self.telemetry["total_real_tokens"] += int(real_mask.sum().item())
            self.telemetry["total_prompts"] += int(B)
            for bi, b in enumerate(self.banks_t):
                f_b = fires[bi]
                any_fired = any_fired | f_b
                if f_b.any():
                    delta_total = delta_total + f_b.unsqueeze(-1).to(h.dtype) * deltas[bi]
                tb = self.telemetry["per_bank"][b["name"]]
                tb["fired_real_tokens"] += int(f_b.sum().item())
                tb["fired_prompts_any_tok"] += int(f_b.any(dim=1).sum().item())
                tb["fired_last_tok"] += int(f_b[:, -1].sum().item())

            self.telemetry["any_bank_fired_real_tokens"] += int(any_fired.sum().item())
            self.telemetry["any_bank_fired_prompts"] += int(any_fired.any(dim=1).sum().item())
            self.telemetry["any_bank_fired_last_tok"] += int(any_fired[:, -1].sum().item())

            h_new = h + delta_total
            if isinstance(out, tuple):
                return (h_new,) + out[1:]
            return h_new

        self._handles.append(self.model.register_forward_pre_hook(_model_pre_hook, with_kwargs=True))
        self._handles.append(layers[self.layer_idx].mlp.register_forward_pre_hook(_mlp_pre_hook))
        self._handles.append(layers[self.layer_idx].register_forward_hook(_layer_post_hook))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._buf.clear()
