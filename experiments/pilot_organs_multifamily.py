#!/usr/bin/env python3
"""
pilot_organs_multifamily.py — Pillar 2 + Pillar 3 Multi-Family Write-Once & Multi-Donor Composition Suite
=========================================================================================================

Implements the winning 4-Neuron Native Bilinear Twin-SwiGLU Conjunction Platonic Organ
(exact 1683 recipe + mu_gn_A zero-centering polish):
  gate(x) = TwinStep(w_c^T (x - mu_gn_A); t1, t2 = t1 + 1.5, beta = 3.0)   [country-contrastive]
  up(x)   = (w_e^T (x - mu_gn_A)) / (0.50 * se_ent_A)                      [entity-vs-general, ~0 on gen, ~1 on entities!]
  out(x)  = [ (SiLU(beta*(s_c - t1)) - SiLU(beta*(s_c - t2))) / (beta*(t2 - t1)) ] * up(x) * (gain * v)
  - Strictly 4 native SwiGLU rows (2 per rule: d_ff 8192 -> 8196)
  - Supports BOTH AFFINE (mlp_bias=True) AND strictly BIASFREE (mlp_bias=False via c_B + M_in + BOS up-null!)
  - Pure Donor-Only threshold t1 = 0.45 * (0.5 * p99(sn) + 0.5 * p10(sp)) — ZERO recipient peeking!
  - ZERO hooks, ZERO ZCA, ZERO lookup tables, ZERO B SGD, Donor A completely unloaded, exact 0.000000 unlearning!

Two modular stages (--stages A,B):

  Stage A — Write-Once, Implant-Anywhere Across 4 Architectures (3+ Families, Including Rectangular!):
    - Extracts the EXACT SAME 4-neuron Bilinear Twin-SwiGLU Organ ONCE from Donor A (qwen2.5-3b, L17, D=2048):
        * PURE_DONOR (100% recipient-free, ZERO Sigma_eps!) -> cache/platonic_organ_qwen_fr_it.pt
        * SMOL_PACKAGED (+0.5 * SmolLM2 Sigma_eps, strictly for SmolLM2 packaged & Granite recipient-blindness Check #2!)
    - Unloads Donor A completely!
    - Fits/caches 0-QA general-text Stiefel sockets (2.5x outlier-filtered sock_in, sock_out, c_B, x_bos_B) and implants into:
        1. smollm2-1.7b   (L9,  D=2048, rm=1.0)  — reuses pilot_organs_p0_stage0_17_9.pt! Includes const_up ablation + NATIVE_B_TWIN_ORACLE!
        2. granite-3.1-2b (L8,  D=2048, rm=0.22, float32!) — includes SMOL_PACKAGED_ON_GRANITE Purity Check #2!
        3. falcon3-3b     (L9,  D=3072, rm=1.0)  — RECTANGULAR 2048 -> 3072 up-projection!
        4. qwen2.5-1.5b   (L13, D=1536, rm=1.0)  — RECTANGULAR 2048 -> 1536 down-projection!

  Stage B — Multi-Donor Native Mid-Layer Composition & Selective Individual Exact Unlearning:
    - Organ 1 (2 native SwiGLU neurons, France -> Japan) from Donor A (qwen2.5-3b, L17, D=2048).
    - Organ 2 (2 native SwiGLU neurons, Italy -> Brazil) from Donor C (falcon3-3b, L14, D=3072, RECTANGULAR 3072 -> 2048!).
    - Unloads BOTH donors completely!
    - Concatenates rows inside ONE solo smollm2-1.7b (d_ff 8192 -> 8196: 8192:8194 = Qwen, 8194:8196 = Falcon).
    - Evaluates BOTH affine and biasfree (mlp_bias=False) across all 5 states:
        BASE (8192), JOINT (8196), SELECTIVE_UNLEARN_QWEN (8194, Falcon only),
        SELECTIVE_UNLEARN_FALCON (8194, Qwen only), UNLEARN_ALL (8192, EXACT 0.000000 max |dLogit|!).

Usage:
  /tmp/platos_key_lab/.venv/bin/python /tmp/platos_key_lab/pilot_organs_multifamily.py --unit-test
  /tmp/platos_key_lab/.venv/bin/python -u /tmp/platos_key_lab/pilot_organs_multifamily.py --stages A,B --targets smollm2,granite,falcon3,qwen1.5b
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla
from scipy.stats import binomtest
import torch
import torch.nn as nn

sys.path.insert(0, "/tmp/platos_key_lab")
import pilot_organs_p0 as po
import pilot1b_break85 as p1b
from extract_and_baseline import (
    CACHE_DIR,
    DATASET_PATH,
    LAB_DIR,
    SEED,
    check_answer_match,
    get_decoder_layers,
)
from pilot_graft_falsify import COUNTRIES_INFO, LANDMARKS_T0

OUT_PATH = LAB_DIR / "results_organs_multifamily.json"
ORGAN_CACHE_PATH = CACHE_DIR / "platonic_organ_qwen_fr_it.pt"
SMOL_STAGE0_CACHE = CACHE_DIR / "pilot_organs_p0_stage0_17_9.pt"

DONOR_QWEN_KEY = "qwen2.5-3b"
DONOR_QWEN_LAYER = 17
DONOR_FALCON_KEY = "falcon3-3b"

RECIPIENT_SPECS: Dict[str, Dict[str, Any]] = {
    "smollm2":  {"model_key": "smollm2-1.7b",   "layer": 9,  "d_model": 2048},
    "granite":  {"model_key": "granite-3.1-2b", "layer": 12, "d_model": 2048},  # 30% depth (85.0% commit in Pilot 1B!)
    "falcon3":  {"model_key": "falcon3-3b",     "layer": 9,  "d_model": 3072},
    "qwen1.5b": {"model_key": "qwen2.5-1.5b",   "layer": 13, "d_model": 1536},
}

FR_IT_EXCLUDE_WORDS = {
    "france", "french", "paris", "italy", "italian", "rome", "napoleon", "voltaire",
    "pasteur", "curie", "dante", "galileo", "vinci", "michelangelo", "ferrari",
    "renault", "peugeot", "fiat", "versace", "chanel", "louvre", "eiffel",
    "colosseum", "venice", "milan", "lyon", "marseille", "florence", "naples",
    "verdi", "puccini", "vivaldi", "monet", "hugo", "balzac", "camus", "sartre",
    "descartes", "lavoisier", "fermi", "marconi", "bologna", "turin", "bordeaux",
    "nice", "seine", "loire", "po", "tiber", "sicily", "corsica", "alps",
    # Additional French/Italian regional cities & regions excluded from 0-QA dual anchors
    "trieste", "genoa", "parma", "modena", "padua", "bergamo", "brescia", "bari",
    "catania", "messina", "cagliari", "perugia", "rimini", "ravenna", "pisa",
    "lucca", "livorno", "trento", "bolzano", "udine", "ancona", "salerno", "lecce",
    "verona", "palermo", "siena", "pompeii", "amalfi", "capri", "tuscany", "lombardy",
    "piedmont", "veneto", "calabria", "sardinia", "reims", "rouen", "dijon", "angers",
    "brest", "metz", "nancy", "tours", "clermont", "avignon", "biarritz", "grenoble",
    "amiens", "caen", "limoges", "orleans", "poitiers", "perpignan", "toulon",
    "antibes", "annecy", "colmar", "chamonix", "strasbourg", "toulouse", "nantes",
    "lille", "cannes", "montpellier", "versailles", "normandy", "brittany", "provence",
    "alsace", "burgundy", "aquitaine", "champagne", "rhone", "garonne",
}


# ======================================================================================
# 1. Exact Rectangular Stiefel Sockets, 1683 Bilinear Twin-SwiGLU & Statistics
# ======================================================================================

def fit_stiefel_socket(
    X_src: np.ndarray,
    X_dst: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Fit centered rectangular-safe Stiefel orthogonal + optimal scalar map:
      X_dst ≈ mu_dst + (X_src - mu_src) @ M.T,   where M = s * R,  R in Stiefel(D_dst, D_src).
    Exact projected-variance scale s = sum(S) / E[||R Cs||^2] handles square (2048->2048),
    up-projection (2048->3072), and down-projection (2048->1536, 3072->2048) in CPU float64!
    """
    Xs = np.asarray(X_src, dtype=np.float64)
    Xd = np.asarray(X_dst, dtype=np.float64)
    N, D_src = Xs.shape
    D_dst = Xd.shape[1]
    if weights is None:
        w = np.ones(N, dtype=np.float64) / N
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / np.sum(w)

    mu_s = (w[:, None] * Xs).sum(axis=0)
    mu_d = (w[:, None] * Xd).sum(axis=0)
    Cs = Xs - mu_s
    Cd = Xd - mu_d

    C = (Cd * w[:, None]).T @ Cs
    U, S, Vt = sla.svd(C, full_matrices=False)
    R = U @ Vt  # (D_dst, D_src)

    Cs_proj = Cs @ R.T
    var_proj = float(np.sum(w[:, None] * (Cs_proj ** 2)))
    s = float(np.sum(S) / max(var_proj, 1e-12))
    M = s * R

    return {
        "R": R,
        "s": s,
        "M": M,
        "mu_src": mu_s,
        "mu_dst": mu_d,
        "d_src": D_src,
        "d_dst": D_dst,
        "sing_vals": S,
    }


def random_stiefel(shape: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Generate Haar-uniform random Stiefel matrix of arbitrary rectangular shape (m, n)."""
    m, n = shape
    if m >= n:
        Q, _ = sla.qr(rng.standard_normal((m, n)), mode="economic")
        return Q
    Q, _ = sla.qr(rng.standard_normal((n, m)), mode="economic")
    return Q.T


def fit_1683_country_gate(
    x_pos: np.ndarray,
    x_neg_list: Sequence[np.ndarray],
    xA_gn: np.ndarray,
    lam_rel: float = 0.30,
    t1_scale: float = 0.45,
    Sigma_eps: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Exact 1683 Donor-Only Country-Contrastive Gate anchored at mu_gn_A = xA_gn.mean(0):
      C = 0.5 * cov(xA_gn) + 0.5 * cov(x_neg)  [+ 0.5 * Sigma_eps ONLY for SMOL_PACKAGED]
      Sig = C + lam_rel * (tr(C) / D) * I
      w_c = solve(Sig, mu_pos - mu_neg) / std((xA_gn - mu_gn_A) @ w_c)
      b_c = -mu_gn_A @ w_c
      t1  = t1_scale * (0.5 * p99(sn) + 0.5 * p10(sp))   [100% Donor-Only, ZERO recipient peeking!]
    """
    xp = np.asarray(x_pos, dtype=np.float64)
    xn = np.vstack([np.asarray(z, dtype=np.float64) for z in x_neg_list])
    xg = np.asarray(xA_gn, dtype=np.float64)
    mu_gn = xg.mean(axis=0)
    D = xp.shape[1]

    C = 0.5 * np.cov(xg.T) + 0.5 * np.cov(xn.T)
    if Sigma_eps is not None:
        C = C + 0.5 * np.asarray(Sigma_eps, dtype=np.float64)
    Sig = C + lam_rel * (float(np.trace(C)) / D) * np.eye(D)
    w = sla.solve(Sig, xp.mean(axis=0) - xn.mean(axis=0), assume_a="pos")
    sig_g = float(np.std((xg - mu_gn) @ w))
    w = w / (sig_g + 1e-12)
    b = -float(mu_gn @ w)

    sp = xp @ w + b
    sn = xn @ w + b
    sg = xg @ w + b
    t1_donor = float(t1_scale * (0.5 * float(np.percentile(sn, 99)) + 0.5 * float(np.percentile(sp, 10))))

    return {
        "w_c": w,
        "b_c": b,
        "t1": t1_donor,
        "mu_gn": mu_gn,
        "stats_A": {
            "pos_mean": round(float(sp.mean()), 4),
            "pos_p10": round(float(np.percentile(sp, 10)), 4),
            "neg_p99": round(float(np.percentile(sn, 99)), 4),
            "gen_mean": round(float(sg.mean()), 4),
            "gen_p999": round(float(np.percentile(sg, 99.9)), 4),
            "t1_donor": round(t1_donor, 4),
        },
    }


def fit_1683_entity_up_gate(
    x_all_ent: np.ndarray,
    xA_gn: np.ndarray,
    lam_rel: float = 0.30,
    u_ent_frac: float = 0.50,
    Sigma_eps: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Exact 1683 Bilinear Entity-vs-General Up-Branch anchored at mu_gn_A = xA_gn.mean(0):
      C_e = 0.5 * cov(xA_gn) + 0.5 * cov(x_all_ent)  [+ 0.5 * Sigma_eps ONLY for SMOL_PACKAGED]
      Sig_e = C_e + lam_rel * (tr(C_e) / D) * I
      w_e = solve(Sig_e, mu_ent - mu_gn) / std((xA_gn - mu_gn_A) @ w_e)
      u_denom = u_ent_frac * mean((x_all_ent - mu_gn_A) @ w_e)   [0.50 for cross-family, 1.0 for native oracle]
      w_uA = w_e / u_denom,   b_uA = -(w_uA @ mu_gn_A)
    So u(gen) is centered right at 0.00 (4x lower KL!) while u(ent) ~ +1.0..+1.3!
    """
    xe = np.asarray(x_all_ent, dtype=np.float64)
    xg = np.asarray(xA_gn, dtype=np.float64)
    mu_gn = xg.mean(axis=0)
    D = xe.shape[1]

    C = 0.5 * np.cov(xg.T) + 0.5 * np.cov(xe.T)
    if Sigma_eps is not None:
        C = C + 0.5 * np.asarray(Sigma_eps, dtype=np.float64)
    Sig = C + lam_rel * (float(np.trace(C)) / D) * np.eye(D)
    w = sla.solve(Sig, xe.mean(axis=0) - mu_gn, assume_a="pos")
    sig_g = float(np.std((xg - mu_gn) @ w))
    w = w / (sig_g + 1e-12)
    se_ent_A = float(np.mean((xe - mu_gn) @ w))
    u_denom = float(u_ent_frac * se_ent_A)
    w_uA = w / (u_denom + 1e-12)
    b_uA = -float(mu_gn @ w_uA)

    u_ent = xe @ w_uA + b_uA
    u_gen = xg @ w_uA + b_uA
    return {
        "w_uA": w_uA,
        "b_uA": b_uA,
        "se_ent_A": se_ent_A,
        "u_denom": u_denom,
        "stats_A": {
            "se_ent_A": round(se_ent_A, 4),
            "u_ent_mean": round(float(u_ent.mean()), 4),
            "u_gen_mean": round(float(u_gen.mean()), 4),
            "u_gen_std": round(float(u_gen.std()), 4),
            "u_gen_p999": round(float(np.percentile(u_gen, 99.9)), 4),
        },
    }


def conjugate_bilinear_impedance_matched(
    rules: Sequence[Dict[str, Any]],
    up_spec: Optional[Dict[str, Any]],
    sock_in: Optional[Dict[str, Any]],
    sock_out: Optional[Dict[str, Any]],
    x_B_fit_gen: np.ndarray,
    x_B_cal_ent: np.ndarray,
    x_bos_B: np.ndarray,
    up_mode: str = "bilinear_up",   # "bilinear_up" or "const_up"
    mode: str = "biasfree",         # "biasfree", "affine", "native_biasfree", "native_affine"
    rm_B: float = 1.0,
    c_B: Optional[np.ndarray] = None,
    beta: float = 3.0,
    width: float = 1.5,
    gain: float = 1.15,
    u_ent_frac_B: float = 0.85,
    x_bos_bank_B: Optional[np.ndarray] = None,
    shuffle_seed: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    2-Scalar 0-Label Recipient Impedance-Matched Bilinear Twin-SwiGLU Conjugation!
    Uses ONLY Recipient B's unlabeled fit general text (x_B_fit_gen) and unlabeled non-Fr/It
    calibration entities (x_B_cal_ent) — ZERO task labels, ZERO test peeking!

      1. Gate branch (per rule):
           wB_c_raw = w_c @ M_in  (or w_c for native oracle)
           sig_gn_B = std((xB_gn - mu_gn_B) @ wB_c_raw)
           wB_c     = wB_c_raw / sig_gn_B                      [exact unit-sigma on B general text!]
           b0_B     = -float(wB_c @ x_B_cal_ent.mean(0))       [centers unrelated B entities at 0.00!]
           rows: g1 = beta*(wB_c^T x + b0_B - t1),  g2 = beta*(wB_c^T x + b0_B - (t1 + width))

      2. Up branch (bilinear_up):
           wB_e_raw  = w_e @ M_in
           mu_gn_B   = xB_gn.mean(0)
           u_denom_B = max(mean((x_B_cal_ent - mu_gn_B) @ wB_e_raw), 1e-6)
           wB_u      = wB_e_raw / (u_ent_frac_B * u_denom_B)   [exact u(ent) ~ 1/0.85 = 1.18 on EVERY arch!]
           bB_u      = -float(wB_u @ mu_gn_B)                  [exact u(gen) == 0.0000 on EVERY arch!]

      3. Write branch:
           W_dB = (gain / rm_B) * (M_out @ v_A) / (beta * width)   [divides by Granite rm=0.22!]

      4. Bias-Free (mlp_bias=False) folding via c_B + exact BOS up-null!
    """
    xB_gn = np.asarray(x_B_fit_gen, dtype=np.float64)
    xB_cal = np.asarray(x_B_cal_ent, dtype=np.float64)
    mu_gn_B = xB_gn.mean(axis=0)
    mu_cal_B = xB_cal.mean(axis=0)
    D_B = xB_gn.shape[1]
    k = 2 * len(rules)
    inv_rm = float(gain) / float(rm_B)
    is_native = mode.startswith("native_")

    if is_native:
        M_in = None
        M_out = None
    else:
        assert sock_in is not None and sock_out is not None
        M_in = sock_in["M"].copy()    # (D_A, D_B)
        M_out = sock_out["M"].copy()  # (D_B, D_A)
        if shuffle_seed is not None:
            rng = np.random.default_rng(shuffle_seed)
            Q_in = random_stiefel(M_in.shape, rng)
            M_in = Q_in * (sla.norm(M_in, "fro") / max(sla.norm(Q_in, "fro"), 1e-12))
            Q_out = random_stiefel(M_out.shape, rng)
            M_out = Q_out * (sla.norm(M_out, "fro") / max(sla.norm(Q_out, "fro"), 1e-12))

    # 1. Impedance-matched Up branch
    if up_mode == "bilinear_up":
        assert up_spec is not None
        w_e = np.asarray(up_spec["w_uA"], dtype=np.float64)  # Direction (scale cancels out in u_denom_B!)
        wB_e_raw = w_e.copy() if is_native else (w_e @ M_in)
        u_raw_cal = float(np.mean((xB_cal - mu_gn_B) @ wB_e_raw))
        # Guard against negative sign on shuffled_R (keep abs floor so shuffled_R doesn't explode)
        u_denom_B = max(abs(u_raw_cal), 1e-3) if shuffle_seed is not None else max(u_raw_cal, 1e-6)
        wB_u = wB_e_raw / (float(u_ent_frac_B) * u_denom_B)
        bB_u = -float(wB_u @ mu_gn_B)
    elif up_mode == "const_up":
        wB_u = np.zeros(D_B, dtype=np.float64)
        bB_u = 1.0
        u_denom_B = 1.0
        u_raw_cal = 1.0
    else:
        raise ValueError(f"Unknown up_mode: {up_mode}")

    W_gB = np.zeros((k, D_B), dtype=np.float64)
    b_gB = np.zeros((k,), dtype=np.float64)
    W_uB = np.zeros((k, D_B), dtype=np.float64)
    b_uB = np.zeros((k,), dtype=np.float64)
    W_dB = np.zeros((D_B, k), dtype=np.float64)

    imp_diag: Dict[str, Any] = {
        "u_raw_cal_mean": round(float(u_raw_cal), 5),
        "u_denom_B": round(float(u_denom_B), 5),
        "u_cal_ent_after": round(float(np.mean(xB_cal @ wB_u + bB_u)), 4),
        "u_gen_after_mean": round(float(np.mean(xB_gn @ wB_u + bB_u)), 6),
        "gates": [],
    }

    denom_step = float(beta) * float(width)
    for i, r in enumerate(rules):
        w_c = np.asarray(r["w_c"], dtype=np.float64)
        t1 = float(r["t1"])
        t2 = t1 + float(width)
        v_A = np.asarray(r["v_A"], dtype=np.float64)

        wB_c_raw = w_c.copy() if is_native else (w_c @ M_in)
        sig_gn_B = float(np.std((xB_gn - mu_gn_B) @ wB_c_raw)) + 1e-12
        wB_c = wB_c_raw / sig_gn_B
        b0_B = -float(wB_c @ mu_cal_B)

        W_gB[2 * i] = beta * wB_c
        b_gB[2 * i] = beta * (b0_B - t1)
        W_gB[2 * i + 1] = beta * wB_c
        b_gB[2 * i + 1] = beta * (b0_B - t2)

        W_uB[2 * i] = wB_u
        b_uB[2 * i] = bB_u
        W_uB[2 * i + 1] = wB_u
        b_uB[2 * i + 1] = bB_u

        v_B = inv_rm * (v_A if is_native else (M_out @ v_A))
        W_dB[:, 2 * i] = v_B / denom_step
        W_dB[:, 2 * i + 1] = -v_B / denom_step

        imp_diag["gates"].append({
            "sig_gn_B_before": round(sig_gn_B, 5),
            "gate_gn_std_after": round(float(np.std((xB_gn - mu_gn_B) @ wB_c)), 6),
            "gate_cal_ent_mean_after": round(float(np.mean(xB_cal @ wB_c + b0_B)), 6),
            "gate_gn_mean_after": round(float(np.mean(xB_gn @ wB_c + b0_B)), 4),
            "t1": round(t1, 4),
            "vB_res_norm": round(float(np.linalg.norm(v_B * rm_B)), 4),
            "vB_mlp_norm": round(float(np.linalg.norm(v_B)), 4),
            "wB_c": wB_c,
            "b0_B": b0_B,
        })

    imp_diag["wB_u"] = wB_u
    imp_diag["bB_u"] = bB_u

    if mode in ("affine", "native_affine"):
        W_gB, b_gB, W_uB, b_uB = po.apply_exact_bos_null_affine(
            W_gB, b_gB, W_uB, b_uB, x_bos_B, mu_gn_B, x_bos_bank=x_bos_bank_B
        )
        return {"W_g": W_gB, "b_g": b_gB, "W_u": W_uB, "b_u": b_uB, "W_d": W_dB, "use_bias": True}, imp_diag

    if mode in ("biasfree", "native_biasfree"):
        assert c_B is not None
        cB = np.asarray(c_B, dtype=np.float64)
        W_gB_bf = W_gB + np.outer(b_gB, cB)
        W_uB_bf = W_uB + np.outer(b_uB, cB)
        W_uB_bf = po.apply_exact_bos_null_biasfree(W_uB_bf, x_bos_B, x_bos_bank=x_bos_bank_B)
        return {"W_g": W_gB_bf, "b_g": None, "W_u": W_uB_bf, "b_u": None, "W_d": W_dB, "use_bias": False}, imp_diag

    raise ValueError(f"Unknown mode: {mode}")


def strip_imp_diag_arrays(imp_diag: Dict[str, Any]) -> Dict[str, Any]:
    """Strip large numpy vectors from imp_diag before saving to JSON."""
    out = {k: v for k, v in imp_diag.items() if k not in ("wB_u", "bB_u", "gates")}
    out["gates"] = [{gk: gv for gk, gv in g.items() if gk not in ("wB_c", "b0_B")} for g in imp_diag["gates"]]
    return out


def concat_conjugated_organs(organs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate multiple conjugated Recipient-B organs along intermediate neuron axis."""
    use_bias = any(o["use_bias"] for o in organs)
    W_g = np.concatenate([o["W_g"] for o in organs], axis=0)
    W_u = np.concatenate([o["W_u"] for o in organs], axis=0)
    W_d = np.concatenate([o["W_d"] for o in organs], axis=1)
    if use_bias:
        b_g = np.concatenate([o["b_g"] if o["b_g"] is not None else np.zeros(len(o["W_g"])) for o in organs], axis=0)
        b_u = np.concatenate([o["b_u"] if o["b_u"] is not None else np.zeros(len(o["W_u"])) for o in organs], axis=0)
    else:
        b_g, b_u = None, None
    return {"W_g": W_g, "b_g": b_g, "W_u": W_u, "b_u": b_u, "W_d": W_d, "use_bias": use_bias}


def slice_mlp_rows_native(
    mlp: nn.Module,
    orig_state: Dict[str, Any],
    keep_extra_indices: Sequence[int],
) -> int:
    """
    Selective Individual Exact Unlearning directly from a LIVE widened MLP!
    Keeps base rows [0..d_ff) plus [d_ff + idx for idx in keep_extra_indices] directly
    by slicing live widened weights. Empty list restores exact d_ff and original bias flags (0.0 max |dLogit|).
    """
    d_ff = orig_state["d_ff"]
    d_model = orig_state["d_model"]
    device = mlp.gate_proj.weight.device
    dtype = mlp.gate_proj.weight.dtype

    if len(keep_extra_indices) == 0:
        po.unlearn_mlp_native(mlp, orig_state)
        return d_ff

    keep_idx = list(range(d_ff)) + [d_ff + int(i) for i in keep_extra_indices]
    idx_t = torch.tensor(keep_idx, dtype=torch.long, device=device)
    new_dff = len(keep_idx)

    gate_bias = mlp.gate_proj.bias is not None
    up_bias = mlp.up_proj.bias is not None
    down_bias = mlp.down_proj.bias is not None

    new_gate = nn.Linear(d_model, new_dff, bias=gate_bias, device=device, dtype=dtype)
    new_up = nn.Linear(d_model, new_dff, bias=up_bias, device=device, dtype=dtype)
    new_down = nn.Linear(new_dff, d_model, bias=down_bias, device=device, dtype=dtype)

    with torch.no_grad():
        new_gate.weight.copy_(mlp.gate_proj.weight[idx_t])
        if gate_bias:
            new_gate.bias.copy_(mlp.gate_proj.bias[idx_t])
        new_up.weight.copy_(mlp.up_proj.weight[idx_t])
        if up_bias:
            new_up.bias.copy_(mlp.up_proj.bias[idx_t])
        new_down.weight.copy_(mlp.down_proj.weight[:, idx_t])
        if down_bias:
            new_down.bias.copy_(mlp.down_proj.bias)

    mlp.gate_proj = new_gate
    mlp.up_proj = new_up
    mlp.down_proj = new_down
    mlp.intermediate_size = new_dff
    return new_dff


# --- Exact Statistics (Pure Numpy + Scipy) ---

def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    p = np.asarray(pos, dtype=np.float64)
    n = np.asarray(neg, dtype=np.float64)
    return float(np.mean(p[:, None] > n[None, :]) + 0.5 * np.mean(p[:, None] == n[None, :]))


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Dict[str, float]:
    if n <= 0:
        return {"p": 0.0, "lo": 0.0, "hi": 0.0, "k": 0, "n": 0}
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return {
        "p": round(p, 4),
        "lo": round(max(0.0, center - half), 4),
        "hi": round(min(1.0, center + half), 4),
        "k": int(k),
        "n": int(n),
    }


def bootstrap_ci(vals: Sequence[float], n_boot: int = 2000, seed: int = SEED + 17) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return {
        "mean": round(float(arr.mean()), 4),
        "lo": round(float(np.percentile(means, 2.5)), 4),
        "hi": round(float(np.percentile(means, 97.5)), 4),
    }


def mcnemar_exact(a_ok: Sequence[int], b_ok: Sequence[int]) -> Dict[str, Any]:
    a = np.asarray(a_ok, dtype=np.int64)
    b = np.asarray(b_ok, dtype=np.int64)
    assert len(a) == len(b)
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))
    n_disc = a_only + b_only
    p_val = 1.0 if n_disc == 0 else float(binomtest(a_only, n_disc, 0.5, alternative="two-sided").pvalue)
    return {
        "a_only": a_only,
        "b_only": b_only,
        "n_discordant": n_disc,
        "p_value": p_val,
    }


# ======================================================================================
# 2. Dataset Splits & Detailed Per-Country Evaluation
# ======================================================================================

def is_fr_it_fact(f: Dict[str, Any]) -> bool:
    txt = (f["subject"] + " " + f["answer"] + " " + f["direct_prompt"]).lower()
    return (
        f["answer"] in ("France", "Italy", "Paris", "Rome", "French", "Italian")
        or any(w in txt.split() for w in FR_IT_EXCLUDE_WORDS)
    )


def load_dataset_splits() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[bool], set]:
    with open(DATASET_PATH) as f:
        facts = json.load(f)["facts"]
    train_facts = [x for x in facts if x["split"] == "train"]
    test_facts = [x for x in facts if x["split"] == "test"]
    ds_fr_it_test = [x for x in test_facts if x["answer"] in ("France", "Italy")][:8]
    fr_it_exclude = [is_fr_it_fact(x) for x in test_facts]
    test_subjs = (
        {x["subject"].lower() for x in test_facts}
        | {e["name"].lower() for e in po.ORGAN1_TEST_ENTITIES + po.ORGAN1_HARDNEG_TEST}
    )
    return train_facts, test_facts, ds_fr_it_test, fr_it_exclude, test_subjs


def eval_organ1_detailed(
    model: Any,
    tok: Any,
    device: torch.device,
    ds_fr_it_test: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate Organ 1 returning combined + separate France/Italy breakdowns + CIs + raw binary vectors."""
    prompts, meta = [], []
    for ent in po.ORGAN1_TEST_ENTITIES:
        c_new = po.ORGAN1_RULE[ent["country"]]
        pmap = po.build_entity_attr_prompts(ent["name"], c_new, ent["prot_q"], ent["prot_ans"])
        for attr in ("country", "capital", "currency", "language", "protected"):
            prompts.append(pmap[attr]["prompt"])
            meta.append((ent["name"], ent["country"], c_new, attr, pmap[attr]["aliases"]))

    for ent in po.ORGAN1_HARDNEG_TEST:
        p = f"The location {ent['name']} is situated in the country of"
        prompts.append(p)
        meta.append((ent["name"], ent["country"], ent["country"], "hardneg", [ent["country"]]))

    for f_item in ds_fr_it_test:
        c_old = f_item["answer"]
        c_new = po.ORGAN1_RULE[c_old]
        prompts.append(f_item["direct_prompt"])
        meta.append((f_item["subject"], c_old, c_new, "ds_fr_it", COUNTRIES_INFO[c_new]["aliases"]))

    gens = po.greedy_generate_batch(model, tok, device, prompts, max_new_tokens=8, batch_size=16)

    by_attr: Dict[str, List[int]] = defaultdict(list)
    by_ent_4: Dict[str, List[int]] = defaultdict(list)
    ent_country: Dict[str, str] = {}
    sub_attr: Dict[str, Dict[str, List[int]]] = {"France": defaultdict(list), "Italy": defaultdict(list)}
    sub_ds: Dict[str, List[int]] = {"France": [], "Italy": []}
    hn_flipped = 0
    hn_kept = 0
    ds_hits: List[int] = []

    for i, gen in enumerate(gens):
        name, c_old, c_new, attr, aliases = meta[i]
        ok = int(check_answer_match(gen, aliases))
        if attr in ("country", "capital", "currency", "language", "protected"):
            by_attr[attr].append(ok)
            sub_attr[c_old][attr].append(ok)
            if attr != "protected":
                by_ent_4[name].append(ok)
                ent_country[name] = c_old
        elif attr == "hardneg":
            hn_kept += ok
            flipped = int(
                check_answer_match(gen, COUNTRIES_INFO["Japan"]["aliases"])
                or check_answer_match(gen, COUNTRIES_INFO["Brazil"]["aliases"])
            )
            hn_flipped += flipped
        elif attr == "ds_fr_it":
            ds_hits.append(ok)
            sub_ds[c_old].append(ok)

    rip_vec = by_attr["capital"] + by_attr["currency"] + by_attr["language"]
    dir_vec = by_attr["country"]
    j4_vec = [int(sum(v) == 4) for v in by_ent_4.values()]
    prot_vec = by_attr["protected"]

    def _subset(c_key: str) -> Dict[str, Any]:
        rv = sub_attr[c_key]["capital"] + sub_attr[c_key]["currency"] + sub_attr[c_key]["language"]
        dv = sub_attr[c_key]["country"]
        jv = [int(sum(v) == 4) for nm, v in by_ent_4.items() if ent_country[nm] == c_key]
        pv = sub_attr[c_key]["protected"]
        return {
            "direct_country_em": round(float(np.mean(dv)), 4),
            "unseen_ripple_3attr_em": round(float(np.mean(rv)), 4),
            "capital_em": round(float(np.mean(sub_attr[c_key]["capital"])), 4),
            "currency_em": round(float(np.mean(sub_attr[c_key]["currency"])), 4),
            "language_em": round(float(np.mean(sub_attr[c_key]["language"])), 4),
            "joint_4of4_em": round(float(np.mean(jv)), 4),
            "protected_retention_em": round(float(np.mean(pv)), 4),
            "ds_rule_gen": round(float(np.mean(sub_ds[c_key])), 4) if sub_ds[c_key] else 0.0,
            "n_ds": len(sub_ds[c_key]),
        }

    return {
        "direct_country_em": round(float(np.mean(dir_vec)), 4),
        "direct_wilson95": wilson_ci(sum(dir_vec), len(dir_vec)),
        "unseen_ripple_3attr_em": round(float(np.mean(rip_vec)), 4),
        "ripple_wilson95": wilson_ci(sum(rip_vec), len(rip_vec)),
        "ripple_boot95": bootstrap_ci(rip_vec),
        "capital_em": round(float(np.mean(by_attr["capital"])), 4),
        "currency_em": round(float(np.mean(by_attr["currency"])), 4),
        "language_em": round(float(np.mean(by_attr["language"])), 4),
        "joint_4of4_em": round(float(np.mean(j4_vec)), 4),
        "protected_retention_em": round(float(np.mean(prot_vec)), 4),
        "hardneg_false_flip_to_rule": round(hn_flipped / max(len(po.ORGAN1_HARDNEG_TEST), 1), 4),
        "hardneg_orig_retained": round(hn_kept / max(len(po.ORGAN1_HARDNEG_TEST), 1), 4),
        "dataset_fr_it_8_rule_gen": round(float(np.mean(ds_hits)), 4) if ds_hits else 0.0,
        "france_subset": _subset("France"),
        "italy_subset": _subset("Italy"),
        "_raw_direct_ok": dir_vec,
        "_raw_ripple_ok": rip_vec,
    }


def strip_raw_vecs(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if not k.startswith("_raw_")}


# ======================================================================================
# 3. Donor Extraction & 0-QA General-Text Socket Fitting / Caching
# ======================================================================================

def build_dual_proper_noun_anchors(
    tok_ref: Any,
    train_facts: List[Dict[str, Any]],
    test_subjs: set,
    n_train_facts: int = 350,
    n_extra_nouns: int = 500,
) -> Tuple[List[str], List[List[int]], Dict[str, Any]]:
    """
    Build strictly rule-blind, task-blind, 0-QA unlabeled proper-noun anchor prompts (~850 total):
      - 350 non-France/non-Italy dataset.json train prompts (identical principle to pilot_organs_p0.py)
      - 500 generic non-France/non-Italy real proper nouns from pilot1b_break85.build_real_proper_noun_pool
        formatted in neutral sentences: "In geography and history, {e} is widely known."
    Strictly excludes ALL French/Italian entities, ALL ORGAN1_* entities, ALL T1 landmarks,
    ALL T3 gap entities, and ALL dataset.json val/test subjects!
    """
    with open(DATASET_PATH) as f:
        ds_full = json.load(f)
    excluded = (
        set(test_subjs)
        | {x["name"].lower() for x in po.ORGAN1_TRAIN_ENTITIES}
        | {x["name"].lower() for x in po.ORGAN1_TEST_ENTITIES}
        | {x["name"].lower() for x in po.ORGAN1_HARDNEG_TRAIN}
        | {x["name"].lower() for x in po.ORGAN1_HARDNEG_TEST}
        | {x["name"].lower() for x in LANDMARKS_T0}
        | {x["subject"].lower() for x in ds_full["facts"] if x["split"] != "train"}
        | {x["subject"].lower() for x in ds_full["facts"] if is_fr_it_fact(x)}
        | FR_IT_EXCLUDE_WORDS
    )
    if po.T3_RESULTS_PATH.exists():
        with open(po.T3_RESULTS_PATH) as f:
            excluded |= {x["entity"].lower() for x in json.load(f).get("gap_items", [])}

    prompts: List[str] = []
    spans: List[List[int]] = []
    seen_ent: set = set()

    # 1. Up to 350 non-Fr/It dataset.json train prompts
    n_ds = 0
    for f_item in train_facts:
        if n_ds >= n_train_facts:
            break
        s_lo = f_item["subject"].lower()
        if is_fr_it_fact(f_item) or s_lo in excluded or s_lo in seen_ent:
            continue
        prompts.append(f_item["direct_prompt"])
        spans.append(list(f_item["direct_subj_char_span"]))
        seen_ent.add(s_lo)
        n_ds += 1

    # 2. Up to 500 generic proper nouns from pilot1b_break85 pool
    raw_nouns, _ = p1b.build_real_proper_noun_pool(tok_ref, tok_ref, ds_full, None, max_entities=1400, seed=SEED)
    n_pn = 0
    for ent in raw_nouns:
        if n_pn >= n_extra_nouns:
            break
        e_lo = ent.strip().lower()
        toks_lo = set(e_lo.replace("-", " ").split())
        if (
            len(e_lo) < 3
            or e_lo in excluded
            or e_lo in seen_ent
            or bool(toks_lo & FR_IT_EXCLUDE_WORDS)
        ):
            continue
        p = f"In geography and history, {ent} is widely known."
        c0 = p.index(ent)
        prompts.append(p)
        spans.append([c0, c0 + len(ent)])
        seen_ent.add(e_lo)
        n_pn += 1

    meta = {
        "n_total": len(prompts),
        "n_ds_train_non_fr_it": n_ds,
        "n_extra_proper_nouns": n_pn,
        "zero_overlap_verified": True,
    }
    return prompts, spans, meta


def extract_donor_1683_states(
    donor_key: str,
    layer_idx: int,
    train_facts: List[Dict[str, Any]],
    test_subjs: set,
    corpus: Dict[str, List[str]],
    extract_passages: bool = True,
    rules_to_extract: Sequence[str] = ("France", "Italy"),
    anchor_prompts: Optional[List[str]] = None,
    anchor_spans: Optional[List[List[int]]] = None,
) -> Dict[str, Any]:
    """
    Load donor ONCE, extract exact 1683 training entity sets, exact country+capital write deltas,
    and optional corpus["fit"]/["eval"] + dual proper-noun anchor states. Then UNLOAD donor completely!
    """
    print(f"  [Donor Extract] Loading {donor_key} L{layer_idx} (float32)...")
    model, tok, dev, rm = po.load_model_fp32(donor_key)

    def _get_ent_states(ents, tmpls):
        prs, sps = [], []
        for e in ents:
            for t in tmpls:
                p = t.format(e=e["name"]); c0 = p.index(e["name"])
                prs.append(p); sps.append([c0, c0 + len(e["name"])])
        return po.extract_entity_last_token_states(model, tok, dev, prs, sps, layer_idx)["x"].astype(np.float64)

    x_fr = _get_ent_states([e for e in po.ORGAN1_TRAIN_ENTITIES if e["country"] == "France"], po.TRAIN_PREFIX_TEMPLATES)
    x_it = _get_ent_states([e for e in po.ORGAN1_TRAIN_ENTITIES if e["country"] == "Italy"], po.TRAIN_PREFIX_TEMPLATES)
    x_hn = _get_ent_states(po.ORGAN1_HARDNEG_TRAIN, po.TRAIN_PREFIX_TEMPLATES[:4])

    tr_other = [f for f in train_facts if not is_fr_it_fact(f) and f["subject"].lower() not in test_subjs]
    tr_fr = [f for f in train_facts if f["answer"] == "France" and f["subject"].lower() not in test_subjs]
    tr_it = [f for f in train_facts if f["answer"] == "Italy" and f["subject"].lower() not in test_subjs]

    x_ds = po.extract_entity_last_token_states(
        model, tok, dev, [f["direct_prompt"] for f in tr_other], [f["direct_subj_char_span"] for f in tr_other], layer_idx
    )["x"].astype(np.float64)
    if tr_fr:
        x_fr = np.vstack([x_fr, po.extract_entity_last_token_states(
            model, tok, dev, [f["direct_prompt"] for f in tr_fr], [f["direct_subj_char_span"] for f in tr_fr], layer_idx
        )["x"].astype(np.float64)])
    if tr_it:
        x_it = np.vstack([x_it, po.extract_entity_last_token_states(
            model, tok, dev, [f["direct_prompt"] for f in tr_it], [f["direct_subj_char_span"] for f in tr_it], layer_idx
        )["x"].astype(np.float64)])

    need_c = []
    if "France" in rules_to_extract: need_c += ["France", "Japan"]
    if "Italy" in rules_to_extract: need_c += ["Italy", "Brazil"]
    cA = po.extract_country_sibling_deltas(model, tok, dev, layer_idx, sorted(set(need_c)))
    v_fr = (cA["country"]["Japan"] - cA["country"]["France"]) + (cA["capital"]["Japan"] - cA["capital"]["France"]) if "France" in rules_to_extract else None
    v_it = (cA["country"]["Brazil"] - cA["country"]["Italy"]) + (cA["capital"]["Brazil"] - cA["capital"]["Italy"]) if "Italy" in rules_to_extract else None

    st_fit = po.extract_layer_states_on_passages(model, tok, dev, corpus["fit"], layer_idx) if extract_passages else None
    st_ev = po.extract_layer_states_on_passages(model, tok, dev, corpus["eval"], layer_idx) if extract_passages else None
    st_anch = None
    if extract_passages and anchor_prompts is not None and anchor_spans is not None:
        st_anch = po.extract_entity_last_token_states(model, tok, dev, anchor_prompts, anchor_spans, layer_idx)
    po.unload_model(model)
    print(f"  [Donor Extract] {donor_key} UNLOADED! fr={len(x_fr)}, it={len(x_it)}, hn={len(x_hn)}, ds_neg={len(x_ds)}, dual_anch={len(anchor_prompts) if st_anch else 0}")

    return {
        "donor_key": donor_key,
        "layer": layer_idx,
        "rm": rm,
        "X_fr": x_fr,
        "X_it": x_it,
        "X_hn": x_hn,
        "X_ds": x_ds,
        "v_fr": v_fr,
        "v_it": v_it,
        "st_fit": st_fit,
        "st_ev": st_ev,
        "st_anch": st_anch,
    }


def fit_or_load_0qa_socket(
    donor_key: str,
    lA: int,
    recip_key: str,
    lB: int,
    stA_fit: Optional[Dict[str, Any]],
    stA_ev: Optional[Dict[str, Any]],
    stA_anch: Optional[Dict[str, Any]],
    anchor_prompts: Optional[List[str]],
    anchor_spans: Optional[List[List[int]]],
    model_B: Any,
    tok_B: Any,
    dev_B: torch.device,
    corpus: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Fit or load 100% 0-QA dual general-text + unlabeled proper-noun anchor Stiefel socket
    (75% general-text corpus["fit"] + 25% ~850 non-Fr/It proper-noun anchors, 2.5x outlier-filtered!).
    Reuses existing dual-anchored pilot_organs_p0_stage0_17_9.pt for qwen2.5-3b L17 -> smollm2-1.7b L9!
    Saves new dual sockets under multifamily_dual_socket_*.pt (keeping prose-only sockets untouched as ablation!).
    """
    if donor_key == "qwen2.5-3b" and lA == 17 and recip_key == "smollm2-1.7b" and lB == 9 and SMOL_STAGE0_CACHE.exists():
        print(f"  [Socket] Reusing cached dual-anchored {SMOL_STAGE0_CACHE.name}")
        s0 = torch.load(SMOL_STAGE0_CACHE, map_location="cpu", weights_only=False)
        c_B = s0["sock_cpin"]["c_src"]
        # Extract Recipient B's unlabeled calibration entity states for 2-scalar 0-label impedance matching!
        assert anchor_prompts is not None and anchor_spans is not None
        stB_cal = po.extract_entity_last_token_states(model_B, tok_B, dev_B, anchor_prompts, anchor_spans, lB)
        return {
            "sock_in": s0["sock_in"],
            "sock_out": s0["sock_out"],
            "c_B": c_B,
            "x_bos_B": s0["x_bos_B"],
            "bos_x_fit_B": s0.get("bos_x_fit_B"),
            "bos_x_eval_B": s0.get("bos_x_eval_B"),
            "x_B_fit_gen": s0["x_B_fit_gen"],
            "x_B_cal_ent": stB_cal["x"],
            "x_B_eval_aligned": s0["x_B_eval_aligned"],
            "x_B_eval_unaligned": s0["x_B_eval_unaligned"],
            "mlpB_eval_aligned_n2": s0["mlpB_eval_aligned_n2"],
            "mlpB_eval_unaligned_n2": s0["mlpB_eval_unaligned_n2"],
            "Sigma_eps": s0["Sigma_eps"],
            "summary": {
                "reused_cache": str(SMOL_STAGE0_CACHE.name),
                "dual_anchored": True,
                "heldout_read_R2": s0["summary"]["heldout_read_R2"],
                "heldout_write_R2": s0["summary"]["heldout_write_R2"],
                "s_in": round(float(s0["sock_in"]["s"]), 4),
                "s_out": round(float(s0["sock_out"]["s"]), 4),
                "entity_norm_ratio_h_out": s0["summary"].get("socket_scales", {}).get("entity_norm_ratio_h_out"),
                "n_fit_aligned": s0["summary"]["counts"]["n_fit_aligned"],
                "n_dual_anchors": s0["summary"]["counts"]["n_train_entities_dual"],
            },
        }

    sock_cache = CACHE_DIR / f"multifamily_dual_socket_{donor_key}_l{lA}_to_{recip_key}_l{lB}.pt"
    if sock_cache.exists():
        b_cached = torch.load(sock_cache, map_location="cpu", weights_only=False)
        if "x_B_fit_gen" in b_cached and "x_B_cal_ent" in b_cached:
            print(f"  [Socket] Loading cached dual socket {sock_cache.name}")
            return b_cached

    assert stA_fit is not None and stA_ev is not None and stA_anch is not None, f"Need donor states + dual anchors to build {sock_cache.name}"
    assert anchor_prompts is not None and anchor_spans is not None
    print(f"  [Socket] Fitting 0-QA DUAL socket (75% general text + 25% {len(anchor_prompts)} unlabeled proper-noun anchors) {donor_key} L{lA} -> {recip_key} L{lB}...")
    stB_fit = po.extract_layer_states_on_passages(model_B, tok_B, dev_B, corpus["fit"], lB)
    stB_ev = po.extract_layer_states_on_passages(model_B, tok_B, dev_B, corpus["eval"], lB)
    stB_anch = po.extract_entity_last_token_states(model_B, tok_B, dev_B, anchor_prompts, anchor_spans, lB)

    p_fit = po.pair_passage_activations(stA_fit, stB_fit)
    p_ev = po.pair_passage_activations(stA_ev, stB_ev)

    keep_x_fit, _ = po.filter_outlier_pairs(p_fit["x_A"], p_fit["x_B"], max_med_mult=2.5)
    keep_ho_fit, ho_stats = po.filter_outlier_pairs(p_fit["h_out_A"], p_fit["h_out_B"], max_med_mult=2.5)
    keep_ho_ev, _ = po.filter_outlier_pairs(p_ev["h_out_A"], p_ev["h_out_B"], max_med_mult=2.5)
    keep_x_anch, _ = po.filter_outlier_pairs(stA_anch["x"], stB_anch["x"], max_med_mult=2.5)
    keep_ho_anch, _ = po.filter_outlier_pairs(stA_anch["h_out"], stB_anch["h_out"], max_med_mult=2.5)

    # 25% total dual weight on unlabeled proper-noun anchors (w_ent = 0.33 * n_gen / n_ent)
    n_gx, n_ax = int(np.sum(keep_x_fit)), int(np.sum(keep_x_anch))
    n_gh, n_ah = int(np.sum(keep_ho_fit)), int(np.sum(keep_ho_anch))
    w_ent_x = 0.33 * n_gx / max(n_ax, 1)
    w_ent_h = 0.33 * n_gh / max(n_ah, 1)

    xA_dual = np.concatenate([p_fit["x_A"][keep_x_fit], stA_anch["x"][keep_x_anch]], axis=0)
    xB_dual = np.concatenate([p_fit["x_B"][keep_x_fit], stB_anch["x"][keep_x_anch]], axis=0)
    w_dual_x = np.concatenate([np.ones(n_gx, dtype=np.float64), np.full(n_ax, w_ent_x, dtype=np.float64)])

    hoA_dual = np.concatenate([p_fit["h_out_A"][keep_ho_fit], stA_anch["h_out"][keep_ho_anch]], axis=0)
    hoB_dual = np.concatenate([p_fit["h_out_B"][keep_ho_fit], stB_anch["h_out"][keep_ho_anch]], axis=0)
    w_dual_h = np.concatenate([np.ones(n_gh, dtype=np.float64), np.full(n_ah, w_ent_h, dtype=np.float64)])

    sock_in = fit_stiefel_socket(xB_dual, xA_dual, weights=w_dual_x)       # B -> A (D_A x D_B)
    sock_out = fit_stiefel_socket(hoA_dual, hoB_dual, weights=w_dual_h)    # A -> B (D_B x D_A)
    c_B, c_B_std = po.compute_constant_predictor(xB_dual.astype(np.float64), lam=1e-2)
    x_bos_B = np.mean(stB_fit["bos_x"], axis=0)

    xA_hat_ev = sock_in["mu_dst"] + (p_ev["x_B"].astype(np.float64) - sock_in["mu_src"]) @ sock_in["M"].T
    r2_read = 1.0 - float(np.sum((xA_hat_ev - p_ev["x_A"]) ** 2) / np.sum((p_ev["x_A"] - p_ev["x_A"].mean(axis=0)) ** 2))
    hoA_ev_f = p_ev["h_out_A"][keep_ho_ev].astype(np.float64)
    hoB_ev_f = p_ev["h_out_B"][keep_ho_ev].astype(np.float64)
    hoB_hat_ev = sock_out["mu_dst"] + (hoA_ev_f - sock_out["mu_src"]) @ sock_out["M"].T
    r2_write = 1.0 - float(np.sum((hoB_hat_ev - hoB_ev_f) ** 2) / np.sum((hoB_ev_f - hoB_ev_f.mean(axis=0)) ** 2))

    E_ev = xA_hat_ev - p_ev["x_A"].astype(np.float64)
    Sigma_eps = (E_ev.T @ E_ev) / max(len(E_ev), 1)
    ent_norm_ratio_ho = float(
        np.median(np.linalg.norm(stB_anch["h_out"][keep_ho_anch], axis=-1))
        / max(np.median(np.linalg.norm(stA_anch["h_out"][keep_ho_anch], axis=-1)), 1e-12)
    )

    bundle = {
        "sock_in": sock_in,
        "sock_out": sock_out,
        "c_B": c_B,
        "x_bos_B": x_bos_B,
        "bos_x_fit_B": stB_fit["bos_x"],
        "bos_x_eval_B": stB_ev["bos_x"],
        "x_B_fit_gen": p_fit["x_B"][keep_x_fit],
        "x_B_cal_ent": stB_anch["x"][keep_x_anch],
        "x_B_eval_aligned": p_ev["x_B"],
        "x_B_eval_unaligned": p_ev["x_B_unaligned"],
        "mlpB_eval_aligned_n2": p_ev["mlpB_aligned_n2"],
        "mlpB_eval_unaligned_n2": p_ev["mlpB_unaligned_n2"],
        "Sigma_eps": Sigma_eps,
        "summary": {
            "donor": donor_key,
            "lA": lA,
            "recipient": recip_key,
            "lB": lB,
            "dual_anchored": True,
            "shape_in": [int(sock_in["d_dst"]), int(sock_in["d_src"])],
            "shape_out": [int(sock_out["d_dst"]), int(sock_out["d_src"])],
            "n_fit_aligned": int(len(p_fit["x_A"])),
            "n_dual_anchors": int(n_ax),
            "anchor_weight_frac": round(float((n_ax * w_ent_x) / (n_gx + n_ax * w_ent_x)), 4),
            "n_eval_aligned": int(len(p_ev["x_A"])),
            "outlier_filter_h_out": ho_stats,
            "heldout_read_R2": round(r2_read, 4),
            "heldout_write_R2": round(r2_write, 4),
            "s_in": round(float(sock_in["s"]), 4),
            "s_out": round(float(sock_out["s"]), 4),
            "entity_norm_ratio_h_out": round(ent_norm_ratio_ho, 4),
            "c_B_std": round(float(c_B_std), 5),
        },
    }
    torch.save(bundle, sock_cache)
    print(f"  [Socket] Saved {sock_cache.name} | shape_in={bundle['summary']['shape_in']} n_anch={n_ax} read_R2={r2_read:.4f} write_R2={r2_write:.4f} s_in={sock_in['s']:.4f} s_out={sock_out['s']:.4f} (ent_ho_ratio={ent_norm_ratio_ho:.4f})")
    return bundle


def extract_recipient_diagnostic_states(
    model_B: Any,
    tok_B: Any,
    dev_B: torch.device,
    lB: int,
    test_facts: List[Dict[str, Any]],
    fr_it_exclude: List[bool],
) -> Dict[str, np.ndarray]:
    """Extract recipient B entity states strictly for AUROC & up-branch diagnostics (NEVER used for fitting!)."""
    te_fr_p, te_fr_s, te_it_p, te_it_s = [], [], [], []
    for e in po.ORGAN1_TEST_ENTITIES:
        p = f"The location {e['name']} is situated in the country of"
        c0 = p.index(e["name"])
        if e["country"] == "France":
            te_fr_p.append(p); te_fr_s.append([c0, c0 + len(e["name"])])
        else:
            te_it_p.append(p); te_it_s.append([c0, c0 + len(e["name"])])
    hn_p = [f"The location {e['name']} is situated in the country of" for e in po.ORGAN1_HARDNEG_TEST]
    hn_s = [[p.index(e["name"]), p.index(e["name"]) + len(e["name"])] for p, e in zip(hn_p, po.ORGAN1_HARDNEG_TEST)]
    unr_p, unr_s = [], []
    for i, x in enumerate(test_facts):
        if not fr_it_exclude[i] and x["subject"] in x["direct_prompt"]:
            c0 = x["direct_prompt"].index(x["subject"])
            unr_p.append(x["direct_prompt"]); unr_s.append([c0, c0 + len(x["subject"])])
    return {
        "X_te_fr": po.extract_entity_last_token_states(model_B, tok_B, dev_B, te_fr_p, te_fr_s, lB)["x"].astype(np.float64),
        "X_te_it": po.extract_entity_last_token_states(model_B, tok_B, dev_B, te_it_p, te_it_s, lB)["x"].astype(np.float64),
        "X_hn": po.extract_entity_last_token_states(model_B, tok_B, dev_B, hn_p, hn_s, lB)["x"].astype(np.float64),
        "X_unrel": po.extract_entity_last_token_states(model_B, tok_B, dev_B, unr_p, unr_s, lB)["x"].astype(np.float64),
    }


def compute_impedance_matched_auroc_diag(
    diag_B: Dict[str, np.ndarray],
    x_B_gen: np.ndarray,
    wB_c: np.ndarray,
    b0_B: float,
    wB_u: np.ndarray,
    bB_u: float,
    country: str,
) -> Dict[str, float]:
    """Compute exact AUROC diagnostics on the actual impedance-matched Recipient-B gate & up directions."""
    X_pos = diag_B["X_te_fr"] if country == "France" else diag_B["X_te_it"]
    z_pos = X_pos @ wB_c + b0_B
    z_unr = diag_B["X_unrel"] @ wB_c + b0_B
    z_hn = diag_B["X_hn"] @ wB_c + b0_B
    z_gen = x_B_gen @ wB_c + b0_B
    u_ent = np.vstack([diag_B["X_te_fr"], diag_B["X_te_it"]]) @ wB_u + bB_u
    u_gen = x_B_gen @ wB_u + bB_u
    return {
        "AU_unrel": round(auroc(z_pos, z_unr), 4),
        "AU_hn": round(auroc(z_pos, z_hn), 4),
        "AU_gen": round(auroc(z_pos, z_gen), 4),
        "AU_up_gen": round(auroc(u_ent, u_gen), 4),
        "gate_pos_mean": round(float(z_pos.mean()), 3),
        "gate_unrel_mean": round(float(z_unr.mean()), 3),
        "gate_unrel_p99": round(float(np.percentile(z_unr, 99)), 3),
        "gate_hn_max": round(float(z_hn.max()), 3),
        "u_ent_mean": round(float(u_ent.mean()), 3),
        "u_ent_min": round(float(u_ent.min()), 3),
        "u_gen_mean": round(float(u_gen.mean()), 4),
        "u_gen_std": round(float(u_gen.std()), 4),
    }


# ======================================================================================
# 4. Stage A: Write-Once, Implant-Anywhere Across 4 Architectures
# ======================================================================================

def build_or_load_qwen_donor_organ(
    train_facts: List[Dict[str, Any]],
    test_subjs: set,
    corpus: Dict[str, List[str]],
    args: argparse.Namespace,
    need_donor_passages: bool,
    anchor_prompts: Optional[List[str]] = None,
    anchor_spans: Optional[List[List[int]]] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Extract Donor A (qwen2.5-3b, L17) ONCE, fit PURE_DONOR (ZERO Sigma_eps!) and SMOL_PACKAGED
    (+0.5*Sigma_eps strictly for SmolLM2 packaged & Granite Purity Check #2!), save to
    cache/platonic_organ_qwen_fr_it.pt, and UNLOAD Donor A!
    """
    s0_smol = torch.load(SMOL_STAGE0_CACHE, map_location="cpu", weights_only=False) if SMOL_STAGE0_CACHE.exists() else None

    if ORGAN_CACHE_PATH.exists() and not need_donor_passages and not args.refit_donor:
        saved = torch.load(ORGAN_CACHE_PATH, map_location="cpu", weights_only=False)
        hp = saved.get("hparams", {})
        if (
            saved.get("recipe") == "bilinear_1683_mu_gn"
            and abs(hp.get("lam_rel", -1) - args.lam_rel) < 1e-6
            and abs(hp.get("t1_scale", -1) - args.t1_scale) < 1e-6
            and abs(hp.get("u_ent_frac", -1) - args.u_ent_frac) < 1e-6
        ):
            print(f"[Stage A] Loading cached Write-Once Bilinear Donor Organ from {ORGAN_CACHE_PATH}")
            return saved, None, None, None

    donor_data = extract_donor_1683_states(
        DONOR_QWEN_KEY, DONOR_QWEN_LAYER, train_facts, test_subjs, corpus,
        extract_passages=need_donor_passages or (s0_smol is None),
        rules_to_extract=("France", "Italy"),
        anchor_prompts=anchor_prompts,
        anchor_spans=anchor_spans,
    )

    if donor_data["st_fit"] is not None:
        x_A_gn = np.concatenate([x[1:] for x in donor_data["st_fit"]["x"]], axis=0).astype(np.float64)
    else:
        assert s0_smol is not None
        x_A_gn = s0_smol["x_A_fit_gen"].astype(np.float64)

    x_fr, x_it, x_hn, x_ds = donor_data["X_fr"], donor_data["X_it"], donor_data["X_hn"], donor_data["X_ds"]
    x_all_ent = np.vstack([x_fr, x_it, x_hn, x_ds])

    def _fit_pkg(Sigma_eps: Optional[np.ndarray]) -> Dict[str, Any]:
        g_fr = fit_1683_country_gate(x_fr, [x_hn, x_ds, x_it], x_A_gn, lam_rel=args.lam_rel, t1_scale=args.t1_scale, Sigma_eps=Sigma_eps)
        g_it = fit_1683_country_gate(x_it, [x_hn, x_ds, x_fr], x_A_gn, lam_rel=args.lam_rel, t1_scale=args.t1_scale, Sigma_eps=Sigma_eps)
        u_sp = fit_1683_entity_up_gate(x_all_ent, x_A_gn, lam_rel=args.lam_rel, u_ent_frac=args.u_ent_frac, Sigma_eps=Sigma_eps)
        g_fr["v_A"] = donor_data["v_fr"]
        g_it["v_A"] = donor_data["v_it"]
        return {"France": g_fr, "Italy": g_it, "up": u_sp}

    pure_pkg = _fit_pkg(None)
    smol_pkg = _fit_pkg(s0_smol["Sigma_eps"]) if (s0_smol is not None and "Sigma_eps" in s0_smol) else None

    saved = {
        "recipe": "bilinear_1683_mu_gn",
        "donor_key": DONOR_QWEN_KEY,
        "layer": DONOR_QWEN_LAYER,
        "d_model": int(x_fr.shape[1]),
        "hparams": {
            "lam_rel": args.lam_rel,
            "t1_scale": args.t1_scale,
            "u_ent_frac": args.u_ent_frac,
            "beta": args.beta,
            "width": args.width,
            "gain": args.gain,
        },
        "PURE_DONOR": pure_pkg,
        "SMOL_PACKAGED": smol_pkg,
    }
    torch.save(saved, ORGAN_CACHE_PATH)
    print(f"  [Stage A] Saved Write-Once Bilinear Organ to {ORGAN_CACHE_PATH} (Donor A UNLOADED!)")
    print(f"    PURE_DONOR Fr={pure_pkg['France']['stats_A']} | It={pure_pkg['Italy']['stats_A']} | Up={pure_pkg['up']['stats_A']}")
    return saved, donor_data["st_fit"], donor_data["st_ev"], donor_data["st_anch"]


def run_stage_A(
    targets: Sequence[str],
    corpus: Dict[str, List[str]],
    train_facts: List[Dict[str, Any]],
    test_facts: List[Dict[str, Any]],
    ds_fr_it_test: List[Dict[str, Any]],
    fr_it_exclude: List[bool],
    test_subjs: set,
    anchor_prompts: List[str],
    anchor_spans: List[List[int]],
    anchor_meta: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    t0 = time.time()
    print(f"\n================ STAGE A: Write-Once Bilinear Twin-SwiGLU Across 4 Architectures ({list(targets)}) ================")

    need_donor_passages = False
    for tgt in targets:
        spec = RECIPIENT_SPECS[tgt]
        rk, lB = spec["model_key"], spec["layer"]
        if tgt == "smollm2" and SMOL_STAGE0_CACHE.exists():
            continue
        sc = CACHE_DIR / f"multifamily_dual_socket_{DONOR_QWEN_KEY}_l{DONOR_QWEN_LAYER}_to_{rk}_l{lB}.pt"
        if not sc.exists():
            need_donor_passages = True

    donor_pkg, stA_fit, stA_ev, stA_anch = build_or_load_qwen_donor_organ(
        train_facts, test_subjs, corpus, args,
        need_donor_passages=need_donor_passages,
        anchor_prompts=anchor_prompts,
        anchor_spans=anchor_spans,
    )

    probe_prompts = [f"The location {e['name']} is situated in the country of" for e in po.ORGAN1_TEST_ENTITIES[:4]]
    stageA_out: Dict[str, Any] = {
        "recipe": "bilinear_1683_mu_gn_dual_anchor",
        "dual_anchors_meta": anchor_meta,
        "donor": {"model_key": DONOR_QWEN_KEY, "layer": DONOR_QWEN_LAYER, "cache_path": str(ORGAN_CACHE_PATH)},
        "hparams": donor_pkg["hparams"],
        "recipients": {},
    }

    for tgt in targets:
        tr0 = time.time()
        spec = RECIPIENT_SPECS[tgt]
        rk, lB = spec["model_key"], spec["layer"]
        print(f"\n--- [Stage A | {tgt}] Loading SOLO Recipient {rk} L{lB} (Donor A UNLOADED!) ---")
        model_B, tok_B, dev_B, rm_B = po.load_model_fp32(rk)
        mlp_B = get_decoder_layers(model_B)[lB].mlp
        orig_state_B = po.save_original_mlp_state(mlp_B)
        print(f"  Loaded {rk} float32 | D={orig_state_B['d_model']} d_ff={orig_state_B['d_ff']} rm={rm_B}")

        sock_bundle = fit_or_load_0qa_socket(
            DONOR_QWEN_KEY, DONOR_QWEN_LAYER, rk, lB,
            stA_fit, stA_ev, stA_anch, anchor_prompts, anchor_spans,
            model_B, tok_B, dev_B, corpus,
        )
        sock_in = sock_bundle["sock_in"]
        sock_out = sock_bundle["sock_out"]
        c_B = sock_bundle["c_B"]
        x_bos_B = sock_bundle["x_bos_B"]
        bos_bank_B = sock_bundle.get("bos_x_fit_B")
        x_B_fit_gen = sock_bundle["x_B_fit_gen"].astype(np.float64)
        x_B_cal_ent = sock_bundle["x_B_cal_ent"].astype(np.float64)
        x_B_gen_ev = sock_bundle["x_B_eval_aligned"].astype(np.float64)

        enc_p = tok_B(probe_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        with torch.no_grad():
            base_logits_probe = model_B(
                input_ids=enc_p["input_ids"].to(dev_B),
                attention_mask=enc_p["attention_mask"].to(dev_B),
                use_cache=False,
            ).logits.cpu()

        base_gt, base_lps = po.eval_general_text_kl_ppl(model_B, tok_B, dev_B, corpus["eval"])
        base_loc, base_loc_ok = po.eval_196_locality(model_B, tok_B, dev_B, test_facts)
        base_qa = eval_organ1_detailed(model_B, tok_B, dev_B, ds_fr_it_test)
        diag_B = extract_recipient_diagnostic_states(model_B, tok_B, dev_B, lB, test_facts, fr_it_exclude)
        print(f"  Baseline {rk}: PPL={base_gt['ppl']:.4f}, 196-fact={base_loc['n_correct_all']}/196 ({base_loc['acc_all']:.4f})")

        # Optional Native-B Twin Oracle on SmolLM2 (exact same 1683 recipe fit natively on B!)
        native_oracle_pkg = None
        if tgt == "smollm2":
            def _get_B_ent(ents, tmpls):
                prs, sps = [], []
                for e in ents:
                    for t in tmpls:
                        p = t.format(e=e["name"]); c0 = p.index(e["name"])
                        prs.append(p); sps.append([c0, c0 + len(e["name"])])
                return po.extract_entity_last_token_states(model_B, tok_B, dev_B, prs, sps, lB)["x"].astype(np.float64)

            xB_fr = _get_B_ent([e for e in po.ORGAN1_TRAIN_ENTITIES if e["country"] == "France"], po.TRAIN_PREFIX_TEMPLATES)
            xB_it = _get_B_ent([e for e in po.ORGAN1_TRAIN_ENTITIES if e["country"] == "Italy"], po.TRAIN_PREFIX_TEMPLATES)
            xB_hn = _get_B_ent(po.ORGAN1_HARDNEG_TRAIN, po.TRAIN_PREFIX_TEMPLATES[:4])
            tr_other = [f for f in train_facts if not is_fr_it_fact(f) and f["subject"].lower() not in test_subjs]
            tr_fr = [f for f in train_facts if f["answer"] == "France" and f["subject"].lower() not in test_subjs]
            tr_it = [f for f in train_facts if f["answer"] == "Italy" and f["subject"].lower() not in test_subjs]
            xB_ds = po.extract_entity_last_token_states(model_B, tok_B, dev_B, [f["direct_prompt"] for f in tr_other], [f["direct_subj_char_span"] for f in tr_other], lB)["x"].astype(np.float64)
            if tr_fr:
                xB_fr = np.vstack([xB_fr, po.extract_entity_last_token_states(model_B, tok_B, dev_B, [f["direct_prompt"] for f in tr_fr], [f["direct_subj_char_span"] for f in tr_fr], lB)["x"].astype(np.float64)])
            if tr_it:
                xB_it = np.vstack([xB_it, po.extract_entity_last_token_states(model_B, tok_B, dev_B, [f["direct_prompt"] for f in tr_it], [f["direct_subj_char_span"] for f in tr_it], lB)["x"].astype(np.float64)])
            cB = po.extract_country_sibling_deltas(model_B, tok_B, dev_B, lB, ["France", "Japan", "Italy", "Brazil"])
            vB_fr = (cB["country"]["Japan"] - cB["country"]["France"]) + (cB["capital"]["Japan"] - cB["capital"]["France"])
            vB_it = (cB["country"]["Brazil"] - cB["country"]["Italy"]) + (cB["capital"]["Brazil"] - cB["capital"]["Italy"])
            xB_all = np.vstack([xB_fr, xB_it, xB_hn, xB_ds])
            # Native oracle: t1_scale=1.0 (no cross-family attenuation!), u_ent_frac=1.0!
            g_fr_B = fit_1683_country_gate(xB_fr, [xB_hn, xB_ds, xB_it], x_B_gen_ev, lam_rel=args.lam_rel, t1_scale=1.0)
            g_it_B = fit_1683_country_gate(xB_it, [xB_hn, xB_ds, xB_fr], x_B_gen_ev, lam_rel=args.lam_rel, t1_scale=1.0)
            u_B = fit_1683_entity_up_gate(xB_all, x_B_gen_ev, lam_rel=args.lam_rel, u_ent_frac=1.0)
            g_fr_B["v_A"] = vB_fr
            g_it_B["v_A"] = vB_it
            native_oracle_pkg = {"France": g_fr_B, "Italy": g_it_B, "up": u_B}

        # Arm list: (arm_name, pkg_key, up_mode, mode, shuffle_seed)
        arm_specs: List[Tuple[str, str, str, str, Optional[int]]] = [
            ("PURE_DONOR_BIASFREE", "PURE_DONOR", "bilinear_up", "biasfree", None),
            ("PURE_DONOR_AFFINE",   "PURE_DONOR", "bilinear_up", "affine",   None),
        ]
        if tgt == "smollm2":
            # Include const_up ablation so paper shows exact const_up -> bilinear_up jump!
            arm_specs.append(("PURE_DONOR_CONST_UP_BIASFREE", "PURE_DONOR", "const_up", "biasfree", None))
            arm_specs.append(("PURE_DONOR_CONST_UP_AFFINE",   "PURE_DONOR", "const_up", "affine",   None))
            if donor_pkg["SMOL_PACKAGED"] is not None:
                arm_specs.append(("PACKAGED_BIASFREE", "SMOL_PACKAGED", "bilinear_up", "biasfree", None))
                arm_specs.append(("PACKAGED_AFFINE",   "SMOL_PACKAGED", "bilinear_up", "affine",   None))
        if tgt == "granite" and donor_pkg["SMOL_PACKAGED"] is not None:
            # Purity Check #2: Recipient-Blindness (SmolLM2-packaged organ implanted into Granite!)
            arm_specs.append(("SMOL_PACKAGED_ON_GRANITE_BIASFREE", "SMOL_PACKAGED", "bilinear_up", "biasfree", None))
            arm_specs.append(("SMOL_PACKAGED_ON_GRANITE_AFFINE",   "SMOL_PACKAGED", "bilinear_up", "affine",   None))
        arm_specs.append(("SHUFFLED_R", "PURE_DONOR", "bilinear_up", "biasfree", SEED + 4444))
        if tgt == "smollm2" and native_oracle_pkg is not None:
            arm_specs.append(("NATIVE_B_TWIN_ORACLE", "NATIVE_ORACLE", "bilinear_up", "native_biasfree", None))

        recip_res: Dict[str, Any] = {
            "model_key": rk,
            "layer": lB,
            "d_model": orig_state_B["d_model"],
            "d_ff_orig": orig_state_B["d_ff"],
            "residual_multiplier": rm_B,
            "socket_summary": sock_bundle["summary"],
            "baseline": {
                "general_text": base_gt,
                "locality_196": base_loc,
                "qa": strip_raw_vecs(base_qa),
            },
            "arms": {},
        }
        raw_vecs_by_arm: Dict[str, Dict[str, List[int]]] = {}

        for arm_name, pkg_key, up_mode, mode, shuf_seed in arm_specs:
            src_pkg = native_oracle_pkg if pkg_key == "NATIVE_ORACLE" else donor_pkg[pkg_key]
            conj, imp_diag = conjugate_bilinear_impedance_matched(
                [src_pkg["France"], src_pkg["Italy"]],
                up_spec=src_pkg["up"],
                sock_in=sock_in,
                sock_out=sock_out,
                x_B_fit_gen=x_B_fit_gen,
                x_B_cal_ent=x_B_cal_ent,
                x_bos_B=x_bos_B,
                up_mode=up_mode,
                mode=mode,
                rm_B=rm_B,
                c_B=c_B,
                beta=args.beta,
                width=args.width,
                gain=args.gain,
                u_ent_frac_B=args.u_ent_frac_b,
                x_bos_bank_B=bos_bank_B,
                shuffle_seed=shuf_seed,
            )
            auroc_diag = {
                "France": compute_impedance_matched_auroc_diag(
                    diag_B, x_B_gen_ev, imp_diag["gates"][0]["wB_c"], imp_diag["gates"][0]["b0_B"],
                    imp_diag["wB_u"], imp_diag["bB_u"], "France",
                ),
                "Italy": compute_impedance_matched_auroc_diag(
                    diag_B, x_B_gen_ev, imp_diag["gates"][1]["wB_c"], imp_diag["gates"][1]["b0_B"],
                    imp_diag["wB_u"], imp_diag["bB_u"], "Italy",
                ),
            }
            if arm_name == "PURE_DONOR_BIASFREE":
                af, ai = auroc_diag["France"], auroc_diag["Italy"]
                print(
                    f"  [2-Scalar Impedance-Matched AUROC on {tgt}] "
                    f"Fr(AU_unr={af['AU_unrel']:.3f} AU_hn={af['AU_hn']:.3f} g_pos={af['gate_pos_mean']:+.2f} g_unr={af['gate_unrel_mean']:+.2f} t1={src_pkg['France']['t1']:.2f}) | "
                    f"It(AU_unr={ai['AU_unrel']:.3f} AU_hn={ai['AU_hn']:.3f} g_pos={ai['gate_pos_mean']:+.2f} g_unr={ai['gate_unrel_mean']:+.2f} t1={src_pkg['Italy']['t1']:.2f}) | "
                    f"u_ent={af['u_ent_mean']:.2f} u_gen={af['u_gen_mean']:+.3f} AU_up={af['AU_up_gen']:.4f} | vB_res={imp_diag['gates'][0]['vB_res_norm']:.2f} vB_mlp={imp_diag['gates'][0]['vB_mlp_norm']:.2f}"
                )

            new_dff = po.widen_mlp_native(
                mlp_B, orig_state_B,
                conj["W_g"], conj["b_g"], conj["W_u"], conj["b_u"], conj["W_d"],
                use_bias=conj["use_bias"],
            )
            fire = po.audit_organ_firing_analytical(
                conj, x_bos_B,
                sock_bundle["x_B_eval_aligned"], sock_bundle["mlpB_eval_aligned_n2"],
                sock_bundle["x_B_eval_unaligned"], sock_bundle["mlpB_eval_unaligned_n2"],
                rm_B=rm_B, bos_x_eval_B=sock_bundle.get("bos_x_eval_B"),
            )
            gt_res, _ = po.eval_general_text_kl_ppl(model_B, tok_B, dev_B, corpus["eval"], base_logprobs=base_lps)
            gt_res["delta_ppl"] = round(gt_res["ppl"] - base_gt["ppl"], 4)
            qa_res = eval_organ1_detailed(model_B, tok_B, dev_B, ds_fr_it_test)
            loc_res, _ = po.eval_196_locality(model_B, tok_B, dev_B, test_facts, base_correct=base_loc_ok, exclude_mask=fr_it_exclude)

            unlearn_diff = po.check_exact_unlearning(model_B, tok_B, dev_B, mlp_B, orig_state_B, base_logits_probe, probe_prompts)
            assert unlearn_diff == 0.0, f"{rk} {arm_name} unlearning NOT exact 0.0: {unlearn_diff}"

            raw_vecs_by_arm[arm_name] = {"direct": qa_res["_raw_direct_ok"], "ripple": qa_res["_raw_ripple_ok"]}
            recip_res["arms"][arm_name] = {
                "widened_d_ff": new_dff,
                "n_added_neurons": new_dff - orig_state_B["d_ff"],
                "up_mode": up_mode,
                "use_bias": conj["use_bias"],
                "unlearn_max_dlogit": unlearn_diff,
                "t1_donor": {"France": round(float(src_pkg["France"]["t1"]), 4), "Italy": round(float(src_pkg["Italy"]["t1"]), 4)},
                "impedance_diag": strip_imp_diag_arrays(imp_diag),
                "auroc_diag": auroc_diag,
                "firing_audit": fire,
                "general_text": gt_res,
                "qa": strip_raw_vecs(qa_res),
                "locality_196": loc_res,
            }
            print(
                f"  [{arm_name:32s}] d_ff={new_dff} dir={qa_res['direct_country_em']:.3f} "
                f"rip3={qa_res['unseen_ripple_3attr_em']:.3f} (cap={qa_res['capital_em']:.2f} cur={qa_res['currency_em']:.2f} lng={qa_res['language_em']:.2f}) "
                f"ds8={qa_res['dataset_fr_it_8_rule_gen']:.3f} prot={qa_res['protected_retention_em']:.3f} hn={qa_res['hardneg_false_flip_to_rule']:.3f} | "
                f"loc={loc_res['newly_broken_unrelated']}/{loc_res['n_unrelated_orig_correct']} ({loc_res['unrelated_retention']:.4f}) | "
                f"KL={gt_res['mean_kl']:.6f} dPPL={gt_res['delta_ppl']:+.4f} BOS={fire['bos_pos0_residual_norm']:.2e} ul={unlearn_diff}"
            )

        if "SHUFFLED_R" in raw_vecs_by_arm:
            mcn = {}
            sh_d, sh_r = raw_vecs_by_arm["SHUFFLED_R"]["direct"], raw_vecs_by_arm["SHUFFLED_R"]["ripple"]
            for aname, vecs in raw_vecs_by_arm.items():
                if aname == "SHUFFLED_R":
                    continue
                mcn[f"{aname}_vs_SHUFFLED_R"] = {
                    "direct": mcnemar_exact(vecs["direct"], sh_d),
                    "ripple_3attr": mcnemar_exact(vecs["ripple"], sh_r),
                }
            recip_res["mcnemar_vs_shuffled"] = mcn

        recip_res["elapsed_sec"] = round(time.time() - tr0, 1)
        stageA_out["recipients"][tgt] = recip_res
        po.unload_model(model_B)

    stageA_out["elapsed_sec"] = round(time.time() - t0, 1)
    return stageA_out


# ======================================================================================
# 5. Stage B: Multi-Donor Composition & Selective Individual Exact Unlearning
# ======================================================================================

def run_stage_B(
    corpus: Dict[str, List[str]],
    train_facts: List[Dict[str, Any]],
    test_facts: List[Dict[str, Any]],
    ds_fr_it_test: List[Dict[str, Any]],
    fr_it_exclude: List[bool],
    test_subjs: set,
    anchor_prompts: List[str],
    anchor_spans: List[List[int]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Multi-Donor Native Mid-Layer Composition & Selective Individual Exact Unlearning:
      - Organ 1 (2 SwiGLU rows, France -> Japan) from Donor A (qwen2.5-3b, L17, D=2048)
      - Organ 2 (2 SwiGLU rows, Italy -> Brazil) from Donor C (falcon3-3b, L14, D=3072 -> RECTANGULAR 3072 -> 2048!)
      - Unload BOTH donors completely!
      - Implant into ONE solo smollm2-1.7b (L9, d_ff 8192 -> 8196) in BOTH biasfree (mlp_bias=False) & affine!
      - Evaluate all 5 states: BASE, JOINT, SELECTIVE_UNLEARN_QWEN, SELECTIVE_UNLEARN_FALCON, UNLEARN_ALL!
    """
    t0 = time.time()
    lC = args.falcon_donor_layer
    print(f"\n================ STAGE B: Multi-Donor Bilinear Composition ({DONOR_QWEN_KEY} L{DONOR_QWEN_LAYER} + {DONOR_FALCON_KEY} L{lC} -> smollm2-1.7b L9) ================")

    # 1. Load Qwen Organ 1 (France -> Japan) from Stage A cache
    donor_qwen, _, _, _ = build_or_load_qwen_donor_organ(train_facts, test_subjs, corpus, args, need_donor_passages=False)
    qwen_fr = donor_qwen["PURE_DONOR"]["France"]
    qwen_up = donor_qwen["PURE_DONOR"]["up"]

    # 2. Extract Falcon3-3B Organ 2 (Italy -> Brazil) + dual-anchor passage states for Falcon->SmolLM2 socket
    falcon_sock_cache = CACHE_DIR / f"multifamily_dual_socket_{DONOR_FALCON_KEY}_l{lC}_to_smollm2-1.7b_l9.pt"
    falcon_organ_cache = CACHE_DIR / f"platonic_organ_falcon_l{lC}_it_bilinear.pt"

    stC_fit, stC_ev, stC_anch = None, None, None
    need_falcon_extract = True
    if falcon_organ_cache.exists() and falcon_sock_cache.exists() and not args.refit_donor:
        cand = torch.load(falcon_organ_cache, map_location="cpu", weights_only=False)
        hp = cand.get("hparams", {})
        if (
            abs(hp.get("t1_scale", -1) - args.t1_scale) < 1e-6
            and abs(hp.get("u_ent_frac", -1) - args.u_ent_frac) < 1e-6
            and abs(hp.get("lam_rel", -1) - args.lam_rel) < 1e-6
        ):
            print(f"  [Stage B] Loading cached Falcon Bilinear Organ 2 from {falcon_organ_cache.name}")
            falcon_pkg = cand
            need_falcon_extract = False

    if need_falcon_extract:
        falcon_data = extract_donor_1683_states(
            DONOR_FALCON_KEY, lC, train_facts, test_subjs, corpus,
            extract_passages=True,
            rules_to_extract=("Italy",),
            anchor_prompts=anchor_prompts,
            anchor_spans=anchor_spans,
        )
        stC_fit, stC_ev, stC_anch = falcon_data["st_fit"], falcon_data["st_ev"], falcon_data["st_anch"]
        x_C_gn = np.concatenate([x[1:] for x in stC_fit["x"]], axis=0).astype(np.float64)
        x_fr_C, x_it_C, x_hn_C, x_ds_C = falcon_data["X_fr"], falcon_data["X_it"], falcon_data["X_hn"], falcon_data["X_ds"]
        x_all_C = np.vstack([x_fr_C, x_it_C, x_hn_C, x_ds_C])

        g_it_C = fit_1683_country_gate(x_it_C, [x_hn_C, x_ds_C, x_fr_C], x_C_gn, lam_rel=args.lam_rel, t1_scale=args.t1_scale)
        u_C = fit_1683_entity_up_gate(x_all_C, x_C_gn, lam_rel=args.lam_rel, u_ent_frac=args.u_ent_frac)
        g_it_C["v_A"] = falcon_data["v_it"]
        falcon_pkg = {
            "donor_key": DONOR_FALCON_KEY,
            "layer": lC,
            "d_model": int(x_it_C.shape[1]),
            "hparams": {"lam_rel": args.lam_rel, "t1_scale": args.t1_scale, "u_ent_frac": args.u_ent_frac},
            "Italy": g_it_C,
            "up": u_C,
        }
        torch.save(falcon_pkg, falcon_organ_cache)
        print(f"  [Stage B] Saved Falcon3-3B Italy->Brazil Bilinear Organ to {falcon_organ_cache.name} (BOTH DONORS UNLOADED!)")
        print(f"    Falcon3 PURE_DONOR It={g_it_C['stats_A']} | Up={u_C['stats_A']}")

    # 3. Load SOLO Recipient SmolLM2-1.7B (BOTH Donors COMPLETELY UNLOADED!)
    print("\n  [Stage B] Loading SOLO Recipient smollm2-1.7b L9 (BOTH Donors UNLOADED!)...")
    model_B, tok_B, dev_B, rm_B = po.load_model_fp32("smollm2-1.7b")
    lB = 9
    mlp_B = get_decoder_layers(model_B)[lB].mlp
    orig_state_B = po.save_original_mlp_state(mlp_B)

    sock_qwen = fit_or_load_0qa_socket(DONOR_QWEN_KEY, DONOR_QWEN_LAYER, "smollm2-1.7b", lB, None, None, None, anchor_prompts, anchor_spans, model_B, tok_B, dev_B, corpus)
    sock_falcon = fit_or_load_0qa_socket(DONOR_FALCON_KEY, lC, "smollm2-1.7b", lB, stC_fit, stC_ev, stC_anch, anchor_prompts, anchor_spans, model_B, tok_B, dev_B, corpus)

    probe_prompts = [f"The location {e['name']} is situated in the country of" for e in po.ORGAN1_TEST_ENTITIES[:4]]
    enc_p = tok_B(probe_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
    with torch.no_grad():
        base_logits_probe = model_B(
            input_ids=enc_p["input_ids"].to(dev_B),
            attention_mask=enc_p["attention_mask"].to(dev_B),
            use_cache=False,
        ).logits.cpu()

    base_gt, base_lps = po.eval_general_text_kl_ppl(model_B, tok_B, dev_B, corpus["eval"])
    base_loc, base_loc_ok = po.eval_196_locality(model_B, tok_B, dev_B, test_facts)
    base_qa = eval_organ1_detailed(model_B, tok_B, dev_B, ds_fr_it_test)
    diag_B = extract_recipient_diagnostic_states(model_B, tok_B, dev_B, lB, test_facts, fr_it_exclude)

    stageB_out: Dict[str, Any] = {
        "recipe": "bilinear_1683_2scalar_impedance_dual_anchor",
        "donors": {
            "organ1_qwen_france": {"model_key": DONOR_QWEN_KEY, "layer": DONOR_QWEN_LAYER, "d_model": 2048, "rows": "8192:8194"},
            "organ2_falcon_italy": {"model_key": DONOR_FALCON_KEY, "layer": lC, "d_model": falcon_pkg["d_model"], "rows": "8194:8196"},
        },
        "recipient": {"model_key": "smollm2-1.7b", "layer": lB, "d_model": 2048, "d_ff_orig": orig_state_B["d_ff"]},
        "sockets": {"qwen_to_smol": sock_qwen["summary"], "falcon_to_smol": sock_falcon["summary"]},
        "baseline": {
            "d_ff": orig_state_B["d_ff"],
            "general_text": base_gt,
            "locality_196": base_loc,
            "qa": strip_raw_vecs(base_qa),
        },
        "modes": {},
    }

    for mode in ("biasfree", "affine"):
        print(f"\n  --- [Stage B | mode={mode}] Multi-Donor Qwen (Fr->Jp) + Falcon (It->Br) ---")
        conj_q, imp_q = conjugate_bilinear_impedance_matched(
            [qwen_fr], up_spec=qwen_up,
            sock_in=sock_qwen["sock_in"], sock_out=sock_qwen["sock_out"],
            x_B_fit_gen=sock_qwen["x_B_fit_gen"], x_B_cal_ent=sock_qwen["x_B_cal_ent"],
            x_bos_B=sock_qwen["x_bos_B"], up_mode="bilinear_up", mode=mode,
            rm_B=rm_B, c_B=sock_qwen["c_B"], beta=args.beta, width=args.width,
            gain=args.gain, u_ent_frac_B=args.u_ent_frac_b, x_bos_bank_B=sock_qwen.get("bos_x_fit_B"),
        )
        conj_f, imp_f = conjugate_bilinear_impedance_matched(
            [falcon_pkg["Italy"]], up_spec=falcon_pkg["up"],
            sock_in=sock_falcon["sock_in"], sock_out=sock_falcon["sock_out"],
            x_B_fit_gen=sock_falcon["x_B_fit_gen"], x_B_cal_ent=sock_falcon["x_B_cal_ent"],
            x_bos_B=sock_falcon["x_bos_B"], up_mode="bilinear_up", mode=mode,
            rm_B=rm_B, c_B=sock_falcon["c_B"], beta=args.beta, width=args.width,
            gain=args.gain, u_ent_frac_B=args.u_ent_frac_b, x_bos_bank_B=sock_falcon.get("bos_x_fit_B"),
        )
        au_q = compute_impedance_matched_auroc_diag(
            diag_B, sock_qwen["x_B_eval_aligned"].astype(np.float64),
            imp_q["gates"][0]["wB_c"], imp_q["gates"][0]["b0_B"], imp_q["wB_u"], imp_q["bB_u"], "France",
        )
        au_f = compute_impedance_matched_auroc_diag(
            diag_B, sock_falcon["x_B_eval_aligned"].astype(np.float64),
            imp_f["gates"][0]["wB_c"], imp_f["gates"][0]["b0_B"], imp_f["wB_u"], imp_f["bB_u"], "Italy",
        )
        if mode == "biasfree":
            print(f"  [Stage B AUROC Qwen   Fr->Jp on SmolLM2] AU_unr={au_q['AU_unrel']:.4f} AU_hn={au_q['AU_hn']:.4f} AU_gen={au_q['AU_gen']:.4f} AU_up={au_q['AU_up_gen']:.4f} g_pos={au_q['gate_pos_mean']:+.2f} g_unr={au_q['gate_unrel_mean']:+.2f} t1={qwen_fr['t1']:.2f} u_ent={au_q['u_ent_mean']:.2f}")
            print(f"  [Stage B AUROC Falcon It->Br on SmolLM2] AU_unr={au_f['AU_unrel']:.4f} AU_hn={au_f['AU_hn']:.4f} AU_gen={au_f['AU_gen']:.4f} AU_up={au_f['AU_up_gen']:.4f} g_pos={au_f['gate_pos_mean']:+.2f} g_unr={au_f['gate_unrel_mean']:+.2f} t1={falcon_pkg['Italy']['t1']:.2f} u_ent={au_f['u_ent_mean']:.2f}")
        conj_joint = concat_conjugated_organs([conj_q, conj_f])  # rows 0:2 = Qwen, rows 2:4 = Falcon!

        def _eval_live_state(state_name: str, cur_dff: int) -> Dict[str, Any]:
            gt_r, _ = po.eval_general_text_kl_ppl(model_B, tok_B, dev_B, corpus["eval"], base_logprobs=base_lps)
            gt_r["delta_ppl"] = round(gt_r["ppl"] - base_gt["ppl"], 4)
            qa_r = eval_organ1_detailed(model_B, tok_B, dev_B, ds_fr_it_test)
            loc_r, _ = po.eval_196_locality(model_B, tok_B, dev_B, test_facts, base_correct=base_loc_ok, exclude_mask=fr_it_exclude)
            fr_s, it_s = qa_r["france_subset"], qa_r["italy_subset"]
            print(
                f"    [{state_name:25s}] d_ff={cur_dff} | "
                f"Fr(dir={fr_s['direct_country_em']:.2f} rip={fr_s['unseen_ripple_3attr_em']:.2f} ds={fr_s['ds_rule_gen']:.2f}) | "
                f"It(dir={it_s['direct_country_em']:.2f} rip={it_s['unseen_ripple_3attr_em']:.2f} ds={it_s['ds_rule_gen']:.2f}) | "
                f"ALL(rip={qa_r['unseen_ripple_3attr_em']:.3f} prot={qa_r['protected_retention_em']:.3f}) | "
                f"loc={loc_r['newly_broken_unrelated']}/{loc_r['n_unrelated_orig_correct']} KL={gt_r['mean_kl']:.6f} dPPL={gt_r['delta_ppl']:+.4f}"
            )
            return {
                "d_ff": cur_dff,
                "general_text": gt_r,
                "qa": strip_raw_vecs(qa_r),
                "locality_196": loc_r,
            }

        # State 2: JOINT_QWEN_PLUS_FALCON (d_ff 8192 -> 8196)
        dff_joint = po.widen_mlp_native(
            mlp_B, orig_state_B,
            conj_joint["W_g"], conj_joint["b_g"], conj_joint["W_u"], conj_joint["b_u"], conj_joint["W_d"],
            use_bias=conj_joint["use_bias"],
        )
        res_joint = _eval_live_state("JOINT_QWEN_PLUS_FALCON", dff_joint)

        # State 3: SELECTIVE_UNLEARN_QWEN (physically slice 8192:8194 -> keep ONLY Falcon [2, 3], d_ff=8194!)
        dff_no_qwen = slice_mlp_rows_native(mlp_B, orig_state_B, keep_extra_indices=[2, 3])
        res_no_qwen = _eval_live_state("SELECTIVE_UNLEARN_QWEN", dff_no_qwen)

        # State 4: SELECTIVE_UNLEARN_FALCON (re-install joint, slice 8194:8196 -> keep ONLY Qwen [0, 1], d_ff=8194!)
        po.widen_mlp_native(
            mlp_B, orig_state_B,
            conj_joint["W_g"], conj_joint["b_g"], conj_joint["W_u"], conj_joint["b_u"], conj_joint["W_d"],
            use_bias=conj_joint["use_bias"],
        )
        dff_no_falcon = slice_mlp_rows_native(mlp_B, orig_state_B, keep_extra_indices=[0, 1])
        res_no_falcon = _eval_live_state("SELECTIVE_UNLEARN_FALCON", dff_no_falcon)

        # State 5: UNLEARN_ALL (slice to d_ff=8192 -> EXACT 0.000000 max |dLogit|!)
        unlearn_all_diff = po.check_exact_unlearning(model_B, tok_B, dev_B, mlp_B, orig_state_B, base_logits_probe, probe_prompts)
        assert unlearn_all_diff == 0.0, f"Stage B {mode} UNLEARN_ALL NOT exact 0.0: {unlearn_all_diff}"
        print(f"    [{'UNLEARN_ALL':25s}] d_ff={mlp_B.gate_proj.out_features} | exact max|dLogit| = {unlearn_all_diff:.6f}")

        stageB_out["modes"][mode] = {
            "use_bias": conj_joint["use_bias"],
            "auroc_diag": {"qwen_france": au_q, "falcon_italy": au_f},
            "states": {
                "JOINT_QWEN_PLUS_FALCON": res_joint,
                "SELECTIVE_UNLEARN_QWEN": res_no_qwen,
                "SELECTIVE_UNLEARN_FALCON": res_no_falcon,
                "UNLEARN_ALL": {"d_ff": int(mlp_B.gate_proj.out_features), "max_dlogit": unlearn_all_diff},
            },
        }

    po.unload_model(model_B)
    stageB_out["elapsed_sec"] = round(time.time() - t0, 1)
    return stageB_out


# ======================================================================================
# 6. Pure-CPU Unit Tests (Zero GPU Required!)
# ======================================================================================

def run_cpu_unit_tests() -> None:
    print("=== Running pilot_organs_multifamily.py (Bilinear 1683) Pure-CPU Unit Tests ===")
    rng = np.random.default_rng(20261126)

    # 1. Rectangular Stiefel Sockets (64->96, 96->64, 64->48)
    for D_src, D_dst, true_s in [(64, 96, 1.35), (96, 64, 0.72), (64, 48, 1.18)]:
        Q_true = random_stiefel((D_dst, D_src), rng)
        N = 400
        mu_s, mu_d = rng.standard_normal(D_src), rng.standard_normal(D_dst)
        Z_iso, _ = sla.qr(rng.standard_normal((N, D_src)), mode="economic")
        Z_iso, _ = sla.qr(Z_iso - Z_iso.mean(axis=0), mode="economic")
        Xs_iso = mu_s + Z_iso * math.sqrt(N)
        Xd_iso = mu_d + true_s * ((Xs_iso - mu_s) @ Q_true.T)
        sock_iso = fit_stiefel_socket(Xs_iso, Xd_iso)
        assert sock_iso["M"].shape == (D_dst, D_src)
        assert abs(sock_iso["s"] - true_s) < 1e-10
        assert float(np.max(np.abs((sock_iso["mu_dst"] + (Xs_iso - sock_iso["mu_src"]) @ sock_iso["M"].T) - Xd_iso))) < 1e-10
        Xs = mu_s + rng.standard_normal((N, D_src)) * np.linspace(0.3, 2.0, D_src)
        Xd = mu_d + true_s * ((Xs - mu_s) @ Q_true.T) + rng.standard_normal((N, D_dst)) * 0.1
        sock = fit_stiefel_socket(Xs, Xd)
        R = sock["R"]
        stief_err = float(np.max(np.abs((R.T @ R - np.eye(D_src)) if D_dst >= D_src else (R @ R.T - np.eye(D_dst)))))
        assert stief_err < 1e-12
        Cs, Cd = Xs - sock["mu_src"], Xd - sock["mu_dst"]
        Cs_proj = Cs @ R.T
        assert abs(float(np.mean(np.sum((sock["s"] * Cs_proj - Cd) * Cs_proj, axis=-1)))) < 1e-11
    print("  [PASS 1/6] Rectangular Stiefel sockets (64->96, 96->64, 64->48) exact orthonormality, recon (<1e-10), & optimal scale verified!")

    # 2. Exact 1683 Bilinear Gate + 2-Scalar 0-Label Impedance Matching on Rectangular Up (64->96) & Down (64->48)!
    D_A = 64
    u_c = rng.standard_normal(D_A); u_c /= np.linalg.norm(u_c)
    u_e = rng.standard_normal(D_A); u_e /= np.linalg.norm(u_e)
    X_A_base = rng.standard_normal((400, D_A)) * 0.5 + rng.standard_normal(D_A)
    X_pos_A = X_A_base[:50] + 5.0 * u_e + 6.0 * u_c
    X_neg_A = X_A_base[50:150] + 5.0 * u_e - 3.0 * u_c
    X_all_ent_A = np.vstack([X_pos_A, X_neg_A])
    g_spec = fit_1683_country_gate(X_pos_A, [X_neg_A], X_A_base, lam_rel=0.30, t1_scale=0.35)
    u_spec = fit_1683_entity_up_gate(X_all_ent_A, X_A_base, lam_rel=0.30, u_ent_frac=0.50)
    g_spec["v_A"] = rng.standard_normal(D_A) * 3.0
    assert abs(u_spec["stats_A"]["u_gen_mean"]) < 1e-10

    for D_B_test in (96, 48):  # Rectangular UP (2048->3072) and DOWN (2048->1536)!
        Q_in_t = random_stiefel((D_A, D_B_test), rng)
        X_B_gn_t = rng.standard_normal((400, D_B_test)) * 0.5 + rng.standard_normal(D_B_test)
        X_A_gn_t = (X_B_gn_t - X_B_gn_t.mean(0)) @ Q_in_t.T * 1.2 + rng.standard_normal(D_A)
        # Synthetic calibration entities in B with deliberate cross-family offset & volume attenuation!
        e_dir_B = Q_in_t.T @ u_e
        X_B_cal_t = X_B_gn_t[:120] + 2.5 * e_dir_B[None, :] + 0.8 * (Q_in_t.T @ u_c)[None, :]
        si_t = fit_stiefel_socket(X_B_gn_t, X_A_gn_t)
        so_t = fit_stiefel_socket(X_A_gn_t, X_B_gn_t)
        cB_t, _ = po.compute_constant_predictor(X_B_gn_t, lam=1e-2)
        xbos_t = rng.standard_normal(D_B_test) * 5.0
        for mode, shuf in [("affine", None), ("biasfree", None), ("biasfree", 999)]:
            conj_t, imp_t = conjugate_bilinear_impedance_matched(
                [g_spec], u_spec, si_t, so_t, X_B_gn_t, X_B_cal_t, xbos_t,
                up_mode="bilinear_up", mode=mode, rm_B=1.0, c_B=cB_t, u_ent_frac_B=0.85, shuffle_seed=shuf,
            )
            assert conj_t["W_g"].shape == (2, D_B_test) and conj_t["W_d"].shape == (D_B_test, 2)
            u_bos = conj_t["W_u"] @ xbos_t + (conj_t["b_u"] if conj_t["b_u"] is not None else 0.0)
            assert float(np.max(np.abs(u_bos))) < 1e-10
            if shuf is None:
                # Exact 2-scalar impedance invariants!
                assert abs(imp_t["u_gen_after_mean"]) < 1e-10
                assert abs(imp_t["u_cal_ent_after"] - 1.0 / 0.85) < 1e-4
                assert abs(imp_t["gates"][0]["gate_gn_std_after"] - 1.0) < 1e-6
                assert abs(imp_t["gates"][0]["gate_cal_ent_mean_after"]) < 1e-10
            if mode == "biasfree":
                assert conj_t["use_bias"] is False and conj_t["b_g"] is None and conj_t["b_u"] is None
    print("  [PASS 2/6] Exact 1683 Bilinear Gate + 2-Scalar 0-Label Impedance Matching verified on BOTH rect UP (64->96) & DOWN (64->48)!")
    print("  [PASS 3/6] Exact invariants (u_gen=0.0, u_cal=1.1765, gate_std=1.0, gate_cal_ent=0.0, BOS<1e-10) verified for affine, biasfree, & shuffled_R!")

    # 4. Real HF LlamaMLP, GraniteMLP (rm=0.22), and Qwen2MLP widening + exact 0.0 unlearning!
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaMLP
    from transformers.models.granite.configuration_granite import GraniteConfig
    from transformers.models.granite.modeling_granite import GraniteMLP
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    D_B = 96
    Q_in = random_stiefel((D_A, D_B), rng)
    X_B_gen = rng.standard_normal((400, D_B)) * 0.5 + rng.standard_normal(D_B)
    X_A_gen = (X_B_gen - X_B_gen.mean(0)) @ Q_in.T * 1.2 + rng.standard_normal(D_A)
    X_B_cal = X_B_gen[:120] + 2.5 * (Q_in.T @ u_e)[None, :]
    sock_in = fit_stiefel_socket(X_B_gen, X_A_gen)
    sock_out = fit_stiefel_socket(X_A_gen, X_B_gen)
    c_B, _ = po.compute_constant_predictor(X_B_gen, lam=1e-2)
    x_bos_B = rng.standard_normal(D_B) * 5.0

    x_t = torch.randn(3, 5, D_B, dtype=torch.float32)
    x_t[:, 0, :] = torch.from_numpy(x_bos_B.astype(np.float32))
    for name, mlp, rm_v in [
        ("LlamaMLP", LlamaMLP(LlamaConfig(hidden_size=D_B, intermediate_size=128, hidden_act="silu", mlp_bias=False)).eval(), 1.0),
        ("GraniteMLP", GraniteMLP(GraniteConfig(hidden_size=D_B, intermediate_size=128, hidden_act="silu", mlp_bias=False, residual_multiplier=0.22)).eval(), 0.22),
        ("Qwen2MLP", Qwen2MLP(Qwen2Config(hidden_size=D_B, intermediate_size=128, hidden_act="silu")).eval(), 1.0),
    ]:
        st0 = po.save_original_mlp_state(mlp)
        with torch.no_grad(): y0 = mlp(x_t)
        conj, _ = conjugate_bilinear_impedance_matched(
            [g_spec], u_spec, sock_in, sock_out, X_B_gen, X_B_cal, x_bos_B, mode="biasfree", rm_B=rm_v, c_B=c_B
        )
        po.widen_mlp_native(mlp, st0, conj["W_g"], conj["b_g"], conj["W_u"], conj["b_u"], conj["W_d"], use_bias=False)
        assert mlp.gate_proj.bias is None and mlp.up_proj.bias is None and mlp.down_proj.bias is None
        with torch.no_grad(): y1 = mlp(x_t)
        if rm_v != 1.0:
            x_np = x_t[:, 1:, :].numpy().astype(np.float64)
            y_exp = (po._silu_np(x_np @ conj["W_g"].T) * (x_np @ conj["W_u"].T)) @ (conj["W_d"] * rm_v).T
            assert float(np.max(np.abs(((y1[:, 1:, :] - y0[:, 1:, :]) * rm_v).numpy() - y_exp))) < 1e-4
        po.unlearn_mlp_native(mlp, st0)
        with torch.no_grad(): y2 = mlp(x_t)
        assert float(torch.max(torch.abs(y2 - y0)).item()) == 0.0
    print("  [PASS 4/6] Real HF LlamaMLP, GraniteMLP (rm=0.22), & Qwen2MLP strictly BIASFREE widening + EXACT 0.0 unlearning verified!")

    # 5. Stage B Multi-Donor Composition & Exact Selective Row-Slicing (128 -> 132 -> 130 -> 128)!
    mlp_s = LlamaMLP(LlamaConfig(hidden_size=D_B, intermediate_size=128, hidden_act="silu", mlp_bias=False)).eval()
    st_s = po.save_original_mlp_state(mlp_s)
    with torch.no_grad(): y_base = mlp_s(x_t)

    D_C = 80
    Q_in_C = random_stiefel((D_C, D_B), rng)
    X_C_gen = (X_B_gen - X_B_gen.mean(0)) @ Q_in_C.T + rng.standard_normal(D_C)
    sock_in_C = fit_stiefel_socket(X_B_gen, X_C_gen)
    sock_out_C = fit_stiefel_socket(X_C_gen, X_B_gen)
    g_spec_C = {"w_c": rng.standard_normal(D_C) * 0.3, "b_c": -1.0, "t1": 1.0, "v_A": rng.standard_normal(D_C) * 2.0}
    u_spec_C = {"w_uA": Q_in_C @ (Q_in.T @ u_e), "b_uA": 0.0}
    for mode in ("biasfree", "affine"):
        c1, _ = conjugate_bilinear_impedance_matched([g_spec], u_spec, sock_in, sock_out, X_B_gen, X_B_cal, x_bos_B, mode=mode, c_B=c_B)
        c2, _ = conjugate_bilinear_impedance_matched([g_spec_C], u_spec_C, sock_in_C, sock_out_C, X_B_gen, X_B_cal, x_bos_B, mode=mode, c_B=c_B)
        c_joint = concat_conjugated_organs([c1, c2])

        po.widen_mlp_native(mlp_s, st_s, c1["W_g"], c1["b_g"], c1["W_u"], c1["b_u"], c1["W_d"], use_bias=c1["use_bias"])
        with torch.no_grad(): y_only1 = mlp_s(x_t)
        po.widen_mlp_native(mlp_s, st_s, c2["W_g"], c2["b_g"], c2["W_u"], c2["b_u"], c2["W_d"], use_bias=c2["use_bias"])
        with torch.no_grad(): y_only2 = mlp_s(x_t)

        assert po.widen_mlp_native(mlp_s, st_s, c_joint["W_g"], c_joint["b_g"], c_joint["W_u"], c_joint["b_u"], c_joint["W_d"], use_bias=c_joint["use_bias"]) == 132
        assert slice_mlp_rows_native(mlp_s, st_s, keep_extra_indices=[2, 3]) == 130
        with torch.no_grad(): assert float(torch.max(torch.abs(mlp_s(x_t) - y_only2)).item()) == 0.0

        po.widen_mlp_native(mlp_s, st_s, c_joint["W_g"], c_joint["b_g"], c_joint["W_u"], c_joint["b_u"], c_joint["W_d"], use_bias=c_joint["use_bias"])
        assert slice_mlp_rows_native(mlp_s, st_s, keep_extra_indices=[0, 1]) == 130
        with torch.no_grad(): assert float(torch.max(torch.abs(mlp_s(x_t) - y_only1)).item()) == 0.0

        assert slice_mlp_rows_native(mlp_s, st_s, keep_extra_indices=[]) == 128
        with torch.no_grad(): assert float(torch.max(torch.abs(mlp_s(x_t) - y_base)).item()) == 0.0
    print("  [PASS 5/6] Stage B Multi-Donor Composition & Selective Row-Slicing (132->130 bit-identical, 130->128 EXACT 0.0) verified for BOTH biasfree & affine!")

    # 6. Zero-Leakage Dual Proper-Noun Anchor Pool & Stats Check
    from transformers import AutoTokenizer
    tok_q = AutoTokenizer.from_pretrained(str(po.HF_CACHE_DIR / DONOR_QWEN_KEY), local_files_only=True)
    tr_f, te_f, _, _, te_s = load_dataset_splits()
    ap, asp, am = build_dual_proper_noun_anchors(tok_q, tr_f, te_s, n_train_facts=350, n_extra_nouns=500)
    assert am["n_total"] >= 700, f"Too few dual anchors: {am}"
    for pr, sp in zip(ap, asp):
        ent_lo = pr[sp[0]:sp[1]].lower()
        assert ent_lo not in te_s, f"Dual anchor leaked test subject: {ent_lo}"
        assert ent_lo not in FR_IT_EXCLUDE_WORDS, f"Dual anchor leaked Fr/It word: {ent_lo}"
    wci = wilson_ci(33, 48)
    bci = bootstrap_ci([1] * 33 + [0] * 15)
    mcn = mcnemar_exact([1] * 33 + [0] * 15, [0] * 48)
    assert 0.54 < wci["lo"] < wci["p"] < wci["hi"] < 0.81 and mcn["p_value"] < 1e-9
    print(f"  [PASS 6/6] {am['n_total']} zero-leak dual anchors ({am['n_ds_train_non_fr_it']} train + {am['n_extra_proper_nouns']} nouns) + exact stats verified!")
    print("=== ALL 6 BILINEAR 2-SCALAR IMPEDANCE-MATCHED CPU UNIT TESTS PASSED! ===")


# ======================================================================================
# 7. CLI Entrypoint
# ======================================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Pillar 2 + Pillar 3: Multi-Family Bilinear Twin-SwiGLU Organs Suite")
    parser.add_argument("--stages", type=str, default="A,B", help="Comma-separated stages: A,B")
    parser.add_argument("--targets", type=str, default="smollm2,granite,falcon3,qwen1.5b", help="Stage A targets")
    parser.add_argument("--lam-rel", type=float, default=0.30, help="1683 LDA trace regularization")
    parser.add_argument("--t1-scale", type=float, default=0.35, help="1683 pure donor-only threshold scale (unit-sigma)")
    parser.add_argument("--u-ent-frac", type=float, default=0.50, help="Donor A up-branch reference fraction")
    parser.add_argument("--u-ent-frac-b", type=float, default=0.85, help="Recipient B 0-label impedance up-branch fraction (u_ent ~ 1.18)")
    parser.add_argument("--beta", type=float, default=3.0, help="Twin-SwiGLU step steepness")
    parser.add_argument("--width", type=float, default=1.5, help="Twin-SwiGLU saturation ramp width (t2 - t1)")
    parser.add_argument("--gain", type=float, default=1.15, help="Write amplitude multiplier")
    parser.add_argument("--falcon-donor-layer", type=int, default=14, help="Stage B Donor C Falcon3-3B layer")
    parser.add_argument("--refit-donor", action="store_true", help="Force re-extraction of donor organ cache")
    parser.add_argument("--unit-test", action="store_true", help="Run pure-CPU unit tests only")
    parser.add_argument("--out", type=str, default=str(OUT_PATH), help="Output JSON path")
    args = parser.parse_args()

    if args.unit_test:
        run_cpu_unit_tests()
        return

    stages = [s.strip().upper() for s in args.stages.split(",") if s.strip()]
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    for t in targets:
        assert t in RECIPIENT_SPECS, f"Unknown target: {t} (valid: {list(RECIPIENT_SPECS.keys())})"

    corpus = po.build_diverse_general_corpus()
    train_facts, test_facts, ds_fr_it_test, fr_it_exclude, test_subjs = load_dataset_splits()
    from transformers import AutoTokenizer
    tok_ref = AutoTokenizer.from_pretrained(str(po.HF_CACHE_DIR / DONOR_QWEN_KEY), local_files_only=True)
    anchor_prompts, anchor_spans, anchor_meta = build_dual_proper_noun_anchors(
        tok_ref, train_facts, test_subjs, n_train_facts=350, n_extra_nouns=500
    )
    print(f"[Dual Anchors] Prepared {anchor_meta['n_total']} strictly 0-QA, non-Fr/It, zero-leak proper-noun anchors.")

    out_path = Path(args.out)

    results: Dict[str, Any] = {}
    if out_path.exists():
        try:
            with open(out_path) as f:
                results = json.load(f)
        except Exception:
            results = {}

    results["meta"] = {
        "recipe": "bilinear_1683_mu_gn_dual_anchor",
        "stages": stages,
        "targets": targets,
        "lam_rel": args.lam_rel,
        "t1_scale": args.t1_scale,
        "u_ent_frac": args.u_ent_frac,
        "beta": args.beta,
        "width": args.width,
        "gain": args.gain,
        "falcon_donor_layer": args.falcon_donor_layer,
        "granite_layer": RECIPIENT_SPECS["granite"]["layer"],
        "dual_anchors_meta": anchor_meta,
        "device": str(po.get_compute_device()),
        "seed": SEED,
    }

    t_all = time.time()
    if "A" in stages:
        results["stage_A_multifamily"] = run_stage_A(
            targets, corpus, train_facts, test_facts, ds_fr_it_test, fr_it_exclude, test_subjs,
            anchor_prompts, anchor_spans, anchor_meta, args,
        )
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    if "B" in stages:
        results["stage_B_multidonor_composition"] = run_stage_B(
            corpus, train_facts, test_facts, ds_fr_it_test, fr_it_exclude, test_subjs,
            anchor_prompts, anchor_spans, args,
        )
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    results["meta"]["total_elapsed_sec"] = round(time.time() - t_all, 1)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[MULTIFAMILY COMPLETE] Saved results to {out_path} (elapsed={results['meta']['total_elapsed_sec']}s)")


if __name__ == "__main__":
    main()
