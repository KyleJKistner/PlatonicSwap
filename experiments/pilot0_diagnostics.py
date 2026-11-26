#!/usr/bin/env python3
"""
pilot0_diagnostics.py — Pilot 0: Zero-Training Cache Diagnostics + Causal Fisher AUC
=====================================================================================

Implements all four Pilot 0 diagnostics cleanly:
  - P0a: Error taxonomy of Procrustes counterfactual swap failures
         (correct_donor, base_subject_survived, right_type_wrong_entity, off_manifold_garbled_other)
         on qwen2.5-3b->smollm2-1.7b, falcon3-3b->smollm2-1.7b, qwen2.5-3b->granite-3.1-2b.
  - P0b: Anchor-count learning curve N in {147, 293, 587} (plus shared-correct pool)
         measuring held-out geometry (centered cosine, R^2/MSE, in-span energy) AND
         train vs test counterfactual swap accuracy (pp/doubling slope).
  - P0c: Dark-memory slice BEYOND GlobalFV (dark facts that stay wrong under GlobalFV in
         results_referee_controls.json): matched vs shuffled vs random donor key swaps
         across all 20 pairs + per-recipient + focus pairs.
  - P0d: Causal Fisher Gradient Diagnostic at swap site l_B on
         qwen2.5-3b->smollm2-1.7b and qwen2.5-3b->granite-3.1-2b:
         (1) ROC-AUC of |g_i^T delta_i| (and signed -g_i^T delta_i, L2 ||delta_i||,
             global Fisher norm ||F_B^{1/2} delta_i||) predicting per-fact swap failure;
         (2) First-order Taylor fidelity corr(g_i^T delta_i, delta log p);
         (3) Empirical Fisher matrix F_B = (1/N) sum g_i g_i^T eigenspectrum,
             effective rank (participation ratio & Shannon), and overlap with top PCA of Z_B.

Usage:
  python pilot0_diagnostics.py --mode cpu      # Pure-CPU cache-only parts (P0c + P0b geometry)
  python pilot0_diagnostics.py --mode all      # Full P0a + P0b + P0c + P0d (uses MPS if available)
  python pilot0_diagnostics.py --mode all --smoke  # Quick CPU/MPS smoke test (4 items)
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from extract_and_baseline import (
    CACHE_DIR,
    DATASET_PATH,
    LAB_DIR,
    RESULTS_PATH as BASELINE_RESULTS_PATH,
    SEED,
    apply_linear_map,
    build_counterfactual_pairs,
    check_answer_match,
    eval_patched_swaps,
    fit_procrustes_global,
    get_decoder_layers,
    pretokenize_swap_batches,
    _greedy_decode_from_prefill,
)

REFEREE_CONTROLS_PATH = LAB_DIR / "results_referee_controls.json"
SYSTEMS_RESULTS_PATH = LAB_DIR / "results_systems.json"
TRIAL_LEDGER_PATH = LAB_DIR / "trial_ledger.csv"
OUT_ALL_PATH = LAB_DIR / "results_pilot0.json"
OUT_CPU_PATH = LAB_DIR / "results_pilot0_cpu.json"

FOCUS_PAIRS_P0AB = [
    "qwen2.5-3b->smollm2-1.7b",
    "falcon3-3b->smollm2-1.7b",
    "qwen2.5-3b->granite-3.1-2b",
]
FOCUS_PAIRS_P0D = [
    "qwen2.5-3b->smollm2-1.7b",
    "qwen2.5-3b->granite-3.1-2b",
]


# ======================================================================================
# Utilities: Exact ROC-AUC, Span Procrustes, Geometry Metrics, Bootstrap CIs
# ======================================================================================

def compute_roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """Exact ROC-AUC via average-rank Mann-Whitney U (handles ties identically to sklearn)."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    # Rank scores with average ranks for ties (1-indexed)
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s_sorted[j] == s_sorted[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos_ranks = float(np.sum(ranks[y == 1]))
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def bootstrap_roc_auc_ci(
    y_true: Sequence[int], scores: Sequence[float], n_boot: int = 2000, seed: int = SEED
) -> Tuple[Optional[float], Optional[float]]:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) < 4 or np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = compute_roc_auc(y[idx], s[idx])
        if a is not None:
            vals.append(a)
    if not vals:
        return None, None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def fit_procrustes_span(Z_A: np.ndarray, Z_B: np.ndarray) -> dict:
    """
    Span-restricted Procrustes: exact SVD on reduced QR spans (eliminates LAPACK nullspace noise).
    """
    Z_A = Z_A.astype(np.float64)
    Z_B = Z_B.astype(np.float64)
    mu_A = Z_A.mean(axis=0)
    mu_B = Z_B.mean(axis=0)
    A_c = Z_A - mu_A
    B_c = Z_B - mu_B
    Q_A, R_A = np.linalg.qr(A_c.T, mode="reduced")  # [D_A, rA]
    Q_B, R_B = np.linalg.qr(B_c.T, mode="reduced")  # [D_B, rB]
    U_s, s_vals, Vt_s = np.linalg.svd(R_B @ R_A.T, full_matrices=False)
    tol = 1e-10 * max(float(s_vals[0]) if s_vals.size else 0.0, 1e-12)
    r_pos = int(np.sum(s_vals > tol))
    U = Q_B @ U_s[:, :r_pos]
    Vt = Vt_s[:r_pos, :] @ Q_A.T
    R = U @ Vt
    s = float(np.linalg.norm(B_c, "fro") / max(np.linalg.norm(A_c, "fro"), 1e-12))
    return {"mu_A": mu_A, "mu_B": mu_B, "R": R, "s": s, "rank": r_pos, "Q_A": Q_A}


def compute_geometry_metrics(
    Z_A_tr: np.ndarray,
    Z_B_tr: np.ndarray,
    Z_A_te: np.ndarray,
    Z_B_te: np.ndarray,
    mp: dict,
) -> dict:
    """Compute held-out & train cosine similarity, relative MSE, R^2, and in-span energy."""
    Z_hat_tr = apply_linear_map(Z_A_tr, mp)
    Z_hat_te = apply_linear_map(Z_A_te, mp)
    mu_B = mp["mu_B"]

    def _stats(Z_true: np.ndarray, Z_pred: np.ndarray) -> dict:
        Zt = Z_true.astype(np.float64)
        Zp = Z_pred.astype(np.float64)
        Zt_c = Zt - mu_B
        Zp_c = Zp - mu_B
        # Per-sample centered cosine
        num_c = np.sum(Zt_c * Zp_c, axis=1)
        den_c = np.maximum(np.linalg.norm(Zt_c, axis=1) * np.linalg.norm(Zp_c, axis=1), 1e-12)
        cos_c = float(np.mean(num_c / den_c))
        # Per-sample raw cosine
        num_r = np.sum(Zt * Zp, axis=1)
        den_r = np.maximum(np.linalg.norm(Zt, axis=1) * np.linalg.norm(Zp, axis=1), 1e-12)
        cos_r = float(np.mean(num_r / den_r))
        # Normalized MSE & R^2
        mse_num = float(np.sum((Zp - Zt) ** 2))
        var_den = float(np.sum(Zt_c ** 2))
        nmse = mse_num / max(var_den, 1e-12)
        r2 = 1.0 - nmse
        mean_l2 = float(np.mean(np.linalg.norm(Zp - Zt, axis=1)))
        return {
            "centered_cosine": round(cos_c, 5),
            "raw_cosine": round(cos_r, 5),
            "normalized_mse": round(nmse, 5),
            "r2": round(r2, 5),
            "mean_l2_err": round(mean_l2, 5),
        }

    # In-span fraction of centered test donor vectors in training donor span
    A_tr_c = Z_A_tr.astype(np.float64) - mp["mu_A"]
    A_te_c = Z_A_te.astype(np.float64) - mp["mu_A"]
    Q_A, _ = np.linalg.qr(A_tr_c.T, mode="reduced")
    proj = (A_te_c @ Q_A) @ Q_A.T
    in_span_energy = float(np.sum(proj ** 2) / max(np.sum(A_te_c ** 2), 1e-12))

    return {
        "train": _stats(Z_B_tr, Z_hat_tr),
        "test": _stats(Z_B_te, Z_hat_te),
        "test_in_train_span_energy_frac": round(in_span_energy, 5),
    }


# ======================================================================================
# P0c: Pure-CPU Dark Slice Beyond GlobalFV
# ======================================================================================

def run_p0c_dark_beyond_globalfv(
    dataset: dict,
    baseline: dict,
    ref_ctrl: dict,
    systems: dict,
) -> dict:
    """
    On dark facts that stay WRONG under GlobalFV (results_referee_controls.json),
    compare matched vs shuffled vs random donor key swaps (from results_baseline.json
    and results_systems.json) across all 20 pairs, per-recipient, and focus pairs.
    """
    facts = dataset["facts"]
    test_idxs = [f["idx"] for f in facts if f["split"] == "test"]
    idx_to_test_pos = {idx: pos for pos, idx in enumerate(test_idxs)}

    # Load answer-disjoint flags & query_only_1B outcomes from trial_ledger.csv if present
    ledger_map: Dict[Tuple[str, int], dict] = {}
    if TRIAL_LEDGER_PATH.exists():
        with open(TRIAL_LEDGER_PATH, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ledger_map[(row["pair"], int(row["fact_idx"]))] = {
                    "global_ad": int(row["global_train_answer_disjoint"]),
                    "same_rel_ad": int(row["same_rel_answer_disjoint"]),
                    "q1b": int(row["query_only_1B_correct"]),
                    "baseline_key": int(row["baseline_key_correct"]),
                    "canon_key": int(row["canon_key_correct"]),
                    "random_key": int(row["random_key_correct"]),
                    "subtype": row["recipient_subtype"],
                }

    per_pair: Dict[str, dict] = {}
    by_recipient: Dict[str, dict] = defaultdict(lambda: {
        "n_dark_total": 0,
        "n_dark_wrong_gfv": 0,
        "matched_hits": 0,
        "shuffled_hits": 0,
        "random_hits": 0,
        "q1b_hits": 0,
        "monge_hits": 0,
        "same_model_hits": 0,
        "disc_m1_s0": 0,
        "disc_m0_s1": 0,
        "n_dark_wrong_gfv_ad": 0,
        "matched_hits_ad": 0,
        "shuffled_hits_ad": 0,
        "random_hits_ad": 0,
        "n_dark_wrong_gfv_nm": 0,
        "matched_hits_nm": 0,
        "shuffled_hits_nm": 0,
    })

    pooled_all = {
        "n_dark_total": 0,
        "n_dark_wrong_gfv": 0,
        "matched_hits": 0,
        "shuffled_hits": 0,
        "random_hits": 0,
        "q1b_hits": 0,
        "monge_hits": 0,
        "same_model_hits": 0,
        "disc_m1_s0": 0,
        "disc_m0_s1": 0,
        "n_dark_wrong_gfv_ad": 0,
        "matched_hits_ad": 0,
        "shuffled_hits_ad": 0,
        "random_hits_ad": 0,
        "n_dark_wrong_gfv_nm": 0,
        "matched_hits_nm": 0,
        "shuffled_hits_nm": 0,
    }
    pooled_cross_family = dict(pooled_all)

    def _same_family(a: str, b: str) -> bool:
        return a.startswith("qwen") and b.startswith("qwen")

    for pair_key, bp in sorted(baseline["pairs"].items()):
        donor, rec = bp["donor"], bp["recipient"]
        dm = bp["dark_memory"]["test_only"]
        dark_idxs = dm["dark_fact_indices"]
        m_out = dm["outcomes"]
        s_out = dm["shuffled_control_outcomes"]
        r_out = dm["random_control_outcomes"]

        rc_p = ref_ctrl["pairs"][pair_key]
        gfv_out = rc_p["control_A_global_icl_fv"]["test_outcomes"]
        gfv_nm_out = rc_p["control_A_global_norm_matched"]["test_outcomes"]

        # Optional systems outcomes on dark_ids
        sys_sc = systems["dark_memory"][rec]["pairs"][pair_key]["scopes"]["test_only"]
        sys_dark_pos = {fid: k for k, fid in enumerate(sys_sc["dark_ids"])}
        sys_monge = sys_sc["methods"]["monge"]["dark_outcomes"]
        sys_ceil = sys_sc["same_model_ceiling_outcomes"]

        rows = []
        n_wg = 0
        m_h = s_h = r_h = q_h = monge_h = ceil_h = 0
        d10 = d01 = 0
        n_wg_ad = m_ad = s_ad = r_ad = 0
        n_wg_nm = m_nm = s_nm = 0

        for k, fid in enumerate(dark_idxs):
            tpos = idx_to_test_pos[fid]
            gfv_ok = int(gfv_out[tpos])
            gfv_nm_ok = int(gfv_nm_out[tpos])
            m_ok = int(m_out[k])
            s_ok = int(s_out[k])
            r_ok = int(r_out[k])
            lm = ledger_map.get((pair_key, fid), {})
            q1b_ok = int(lm.get("q1b", 0))
            is_ad = int(lm.get("global_ad", 0))
            sp_idx = sys_dark_pos.get(fid, None)
            mg_ok = int(sys_monge[sp_idx]) if sp_idx is not None else 0
            cl_ok = int(sys_ceil[sp_idx]) if sp_idx is not None else 0

            if gfv_nm_ok == 0:
                n_wg_nm += 1
                m_nm += m_ok
                s_nm += s_ok

            if gfv_ok == 0:
                n_wg += 1
                m_h += m_ok
                s_h += s_ok
                r_h += r_ok
                q_h += q1b_ok
                monge_h += mg_ok
                ceil_h += cl_ok
                if m_ok == 1 and s_ok == 0:
                    d10 += 1
                elif m_ok == 0 and s_ok == 1:
                    d01 += 1
                if is_ad:
                    n_wg_ad += 1
                    m_ad += m_ok
                    s_ad += s_ok
                    r_ad += r_ok
                rows.append({
                    "fact_idx": fid,
                    "fact_id": facts[fid]["id"],
                    "subject": facts[fid]["subject"],
                    "relation": facts[fid]["relation"],
                    "answer": facts[fid]["answer"],
                    "global_ad": is_ad,
                    "matched": m_ok,
                    "shuffled": s_ok,
                    "random": r_ok,
                    "monge": mg_ok,
                    "same_model_ceil": cl_ok,
                    "q1b": q1b_ok,
                })

        per_pair[pair_key] = {
            "donor": donor,
            "recipient": rec,
            "cross_family": not _same_family(donor, rec),
            "n_dark_total": len(dark_idxs),
            "n_dark_wrong_under_globalfv": n_wg,
            "globalfv_resolved_dark_frac": round(1.0 - (n_wg / len(dark_idxs)), 4) if dark_idxs else None,
            "matched_hits": m_h,
            "shuffled_hits": s_h,
            "random_hits": r_h,
            "q1b_hits": q_h,
            "monge_hits": monge_h,
            "same_model_ceil_hits": ceil_h,
            "matched_rate": round(m_h / n_wg, 4) if n_wg else None,
            "shuffled_rate": round(s_h / n_wg, 4) if n_wg else None,
            "random_rate": round(r_h / n_wg, 4) if n_wg else None,
            "matched_minus_shuffled_pp": round(100.0 * (m_h - s_h) / n_wg, 2) if n_wg else None,
            "discordant_m1_s0": d10,
            "discordant_m0_s1": d01,
            "n_dark_wrong_globalfv_ad": n_wg_ad,
            "matched_hits_ad": m_ad,
            "shuffled_hits_ad": s_ad,
            "items": rows,
        }

        for acc_dict in (
            [by_recipient[rec], pooled_all]
            + ([pooled_cross_family] if not _same_family(donor, rec) else [])
        ):
            acc_dict["n_dark_total"] += len(dark_idxs)
            acc_dict["n_dark_wrong_gfv"] += n_wg
            acc_dict["matched_hits"] += m_h
            acc_dict["shuffled_hits"] += s_h
            acc_dict["random_hits"] += r_h
            acc_dict["q1b_hits"] += q_h
            acc_dict["monge_hits"] += monge_h
            acc_dict["same_model_hits"] += ceil_h
            acc_dict["disc_m1_s0"] += d10
            acc_dict["disc_m0_s1"] += d01
            acc_dict["n_dark_wrong_gfv_ad"] += n_wg_ad
            acc_dict["matched_hits_ad"] += m_ad
            acc_dict["shuffled_hits_ad"] += s_ad
            acc_dict["random_hits_ad"] += r_ad
            acc_dict["n_dark_wrong_gfv_nm"] += n_wg_nm
            acc_dict["matched_hits_nm"] += m_nm
            acc_dict["shuffled_hits_nm"] += s_nm

    def _finalize_bucket(d: dict) -> dict:
        n = d["n_dark_wrong_gfv"]
        n_ad = d["n_dark_wrong_gfv_ad"]
        n_nm = d["n_dark_wrong_gfv_nm"]
        out = dict(d)
        out["globalfv_resolved_dark_frac"] = (
            round(1.0 - n / d["n_dark_total"], 4) if d["n_dark_total"] else None
        )
        out["matched_rate"] = round(d["matched_hits"] / n, 4) if n else None
        out["shuffled_rate"] = round(d["shuffled_hits"] / n, 4) if n else None
        out["random_rate"] = round(d["random_hits"] / n, 4) if n else None
        out["q1b_rate"] = round(d["q1b_hits"] / n, 4) if n else None
        out["monge_rate"] = round(d["monge_hits"] / n, 4) if n else None
        out["same_model_ceil_rate"] = round(d["same_model_hits"] / n, 4) if n else None
        out["matched_minus_shuffled_pp"] = (
            round(100.0 * (d["matched_hits"] - d["shuffled_hits"]) / n, 2) if n else None
        )
        out["matched_rate_ad"] = round(d["matched_hits_ad"] / n_ad, 4) if n_ad else None
        out["shuffled_rate_ad"] = round(d["shuffled_hits_ad"] / n_ad, 4) if n_ad else None
        out["matched_minus_shuffled_ad_pp"] = (
            round(100.0 * (d["matched_hits_ad"] - d["shuffled_hits_ad"]) / n_ad, 2) if n_ad else None
        )
        out["matched_rate_nm"] = round(d["matched_hits_nm"] / n_nm, 4) if n_nm else None
        out["shuffled_rate_nm"] = round(d["shuffled_hits_nm"] / n_nm, 4) if n_nm else None
        out["matched_minus_shuffled_nm_pp"] = (
            round(100.0 * (d["matched_hits_nm"] - d["shuffled_hits_nm"]) / n_nm, 2) if n_nm else None
        )
        return out

    return {
        "pooled_all_20_pairs": _finalize_bucket(pooled_all),
        "pooled_cross_family_18_pairs": _finalize_bucket(pooled_cross_family),
        "by_recipient": {r: _finalize_bucket(v) for r, v in sorted(by_recipient.items())},
        "focus_pairs": {pk: per_pair[pk] for pk in FOCUS_PAIRS_P0AB if pk in per_pair},
        "per_pair_summary": {
            pk: {k: v for k, v in d.items() if k != "items"}
            for pk, d in per_pair.items()
        },
    }


# ======================================================================================
# P0a: Error Taxonomy of Procrustes Swap Failures (requires model forward passes)
# ======================================================================================

def eval_swaps_with_generations(
    model: Any,
    tok: Any,
    facts: List[dict],
    eval_B: List[dict],
    swap_pairs: Sequence[Tuple[int, int]],
    z_hat_mat: np.ndarray,
    layer_B: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 32,
    max_new_tokens: int = 6,
    pretok_batches: Optional[List[dict]] = None,
) -> Tuple[List[int], List[str]]:
    """
    Exact patched counterfactual swap decoding full greedy continuations (max_new_tokens=6)
    for EVERY item so failures can be classified accurately.
    """
    layers = get_decoder_layers(model)
    target_layer = layers[layer_B]
    n = len(swap_pairs)
    outcomes = [0] * n
    gen_texts = [""] * n

    batches = (
        pretok_batches
        if pretok_batches is not None
        else pretokenize_swap_batches(tok, facts, eval_B, swap_pairs, device, batch_size=batch_size)
    )
    ctx = {"active": False, "subj_pos": None, "z_patch": None}

    def patch_hook(_mod, _inp, out):
        if not ctx["active"]:
            return out
        if isinstance(out, tuple):
            h = out[0].clone()
            is_tup = True
        else:
            h = out.clone()
            is_tup = False
        if h.shape[1] <= 1:
            return out
        bsz = h.shape[0]
        b_ar = torch.arange(bsz, device=h.device)
        h[b_ar, ctx["subj_pos"], :] = ctx["z_patch"].to(device=h.device, dtype=h.dtype)
        return (h,) + out[1:] if is_tup else h

    handle = target_layer.register_forward_hook(patch_hook)
    try:
        for b in batches:
            start, end = b["start"], b["end"]
            batch_pairs = b["batch_pairs"]
            input_ids = b["input_ids"]
            attention_mask = b["attention_mask"]

            ctx["subj_pos"] = b["subj_pos"]
            ctx["z_patch"] = torch.from_numpy(z_hat_mat[start:end]).to(device=device, dtype=dtype)
            ctx["active"] = True
            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
            ctx["active"] = False

            first_logits = out.logits[:, -1, :]
            gen_ids = _greedy_decode_from_prefill(
                model, first_logits, out.past_key_values, attention_mask, max_new_tokens=max_new_tokens
            )
            for bi in range(end - start):
                p_idx = start + bi
                d_idx = batch_pairs[bi][0]
                g_txt = tok.decode(gen_ids[bi].tolist(), skip_special_tokens=True)
                gen_texts[p_idx] = g_txt
                if check_answer_match(g_txt, facts[d_idx]["aliases"], facts[d_idx]["relation"]):
                    outcomes[p_idx] = 1
    finally:
        handle.remove()

    return outcomes, gen_texts


def classify_p0a_taxonomy(
    dataset: dict,
    swap_pairs: Sequence[Tuple[int, int]],
    outcomes: Sequence[int],
    gen_texts: Sequence[str],
    z_hat: np.ndarray,
    z_true_donor: np.ndarray,
    z_true_base: np.ndarray,
    train_answer_set: set,
) -> dict:
    """
    Classify every counterfactual swap into:
      - correct_donor
      - base_subject_survived
      - right_type_wrong_entity
      - off_manifold_garbled_other (with sub-breakdown: garbled_nonalpha vs wrong_type_or_other)
    """
    facts = dataset["facts"]
    # Build comprehensive per-relation valid entity alias bank
    rel_all_aliases: Dict[str, List[str]] = defaultdict(list)
    for rel, cands in dataset["relation_candidate_answers"].items():
        rel_all_aliases[rel].extend(cands)
    for f in facts:
        rel_all_aliases[f["relation"]].extend(f["aliases"])
    # Deduplicate (longest first so multi-word answers match cleanly)
    rel_alias_bank = {
        rel: sorted(set(a for a in als if a), key=len, reverse=True)
        for rel, als in rel_all_aliases.items()
    }

    cats: List[str] = []
    subcats: List[str] = []
    examples_by_cat: Dict[str, List[dict]] = defaultdict(list)

    # Geometry diagnostics by category
    cos_to_donor = np.sum(z_hat * z_true_donor, axis=1) / np.maximum(
        np.linalg.norm(z_hat, axis=1) * np.linalg.norm(z_true_donor, axis=1), 1e-12
    )
    l2_to_donor = np.linalg.norm(z_hat - z_true_donor, axis=1)
    l2_to_base = np.linalg.norm(z_hat - z_true_base, axis=1)

    ad_flags = []
    for p_idx, ((d_idx, b_idx), ok, g_txt) in enumerate(zip(swap_pairs, outcomes, gen_texts)):
        fd = facts[d_idx]
        fb = facts[b_idx]
        rel = fd["relation"]
        is_ad = int(all(a.strip().lower() not in train_answer_set for a in fd["aliases"]))
        ad_flags.append(is_ad)

        if ok == 1 or check_answer_match(g_txt, fd["aliases"], rel):
            cat = "correct_donor"
            subcat = "correct_donor"
        elif check_answer_match(g_txt, fb["aliases"], rel):
            cat = "base_subject_survived"
            subcat = "base_subject_survived"
        else:
            # Exclude donor & base aliases from right-type candidate pool
            excl = {a.strip().lower() for a in list(fd["aliases"]) + list(fb["aliases"])}
            other_cands = [a for a in rel_alias_bank[rel] if a.strip().lower() not in excl]
            if check_answer_match(g_txt, other_cands, rel):
                cat = "right_type_wrong_entity"
                subcat = "right_type_wrong_entity"
            else:
                cat = "off_manifold_garbled_other"
                s = g_txt.strip()
                if not s or not any(ch.isalpha() for ch in s[:8]):
                    subcat = "empty_or_nonalpha"
                elif s.split()[0].lower() in {
                    "the", "a", "an", "in", "of", "to", "is", "was", "and", "or", "for", "on", "at", "by", "its", "their"
                }:
                    subcat = "function_word_continuation"
                else:
                    subcat = "wrong_type_or_other_word"

        cats.append(cat)
        subcats.append(subcat)
        if cat != "correct_donor" and len(examples_by_cat[cat]) < 12:
            examples_by_cat[cat].append({
                "pair_idx": p_idx,
                "relation": rel,
                "donor_subject": fd["subject"],
                "donor_answer": fd["answer"],
                "base_subject": fb["subject"],
                "base_answer": fb["answer"],
                "generated": g_txt,
                "subcat": subcat,
                "answer_disjoint": is_ad,
                "cos_to_donor": round(float(cos_to_donor[p_idx]), 4),
                "l2_to_donor": round(float(l2_to_donor[p_idx]), 4),
            })

    n_total = len(cats)
    counts = Counter(cats)
    n_fail = n_total - counts["correct_donor"]
    sub_counts = Counter(subcats)

    def _slice_summary(mask: Sequence[bool]) -> dict:
        sel = [c for c, m in zip(cats, mask) if m]
        nt = len(sel)
        ct = Counter(sel)
        nf = nt - ct["correct_donor"]
        return {
            "n_total": nt,
            "n_correct": ct["correct_donor"],
            "n_failures": nf,
            "acc": round(ct["correct_donor"] / nt, 4) if nt else None,
            "counts": {
                k: ct[k]
                for k in ("correct_donor", "right_type_wrong_entity", "base_subject_survived", "off_manifold_garbled_other")
            },
            "frac_of_all": {
                k: round(ct[k] / nt, 4) if nt else None
                for k in ("correct_donor", "right_type_wrong_entity", "base_subject_survived", "off_manifold_garbled_other")
            },
            "frac_of_failures": {
                k: round(ct[k] / nf, 4) if nf else None
                for k in ("right_type_wrong_entity", "base_subject_survived", "off_manifold_garbled_other")
            },
        }

    geom_by_cat = {}
    for k in ("correct_donor", "right_type_wrong_entity", "base_subject_survived", "off_manifold_garbled_other"):
        idxs = [i for i, c in enumerate(cats) if c == k]
        if idxs:
            geom_by_cat[k] = {
                "n": len(idxs),
                "mean_cos_to_donor": round(float(np.mean(cos_to_donor[idxs])), 4),
                "mean_l2_to_donor": round(float(np.mean(l2_to_donor[idxs])), 4),
                "mean_l2_to_base": round(float(np.mean(l2_to_base[idxs])), 4),
            }

    return {
        "overall": _slice_summary([True] * n_total),
        "answer_disjoint": _slice_summary([bool(x) for x in ad_flags]),
        "answer_seen_in_train": _slice_summary([not bool(x) for x in ad_flags]),
        "off_manifold_subcounts": {
            k: sub_counts[k]
            for k in ("empty_or_nonalpha", "function_word_continuation", "wrong_type_or_other_word")
        },
        "geometry_by_category": geom_by_cat,
        "failure_examples": dict(examples_by_cat),
        "per_item_category": cats,
    }


# ======================================================================================
# P0b: Anchor-Count Learning Curve N in {147, 293, 587}
# ======================================================================================

def run_p0b_for_pair(
    pair_key: str,
    bp: dict,
    facts: List[dict],
    Z_A_layer: np.ndarray,
    Z_B_layer: np.ndarray,
    all_train_indices: List[int],
    train_answer_set: set,
    model_B: Optional[Any] = None,
    tok_B: Optional[Any] = None,
    eval_B: Optional[List[dict]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    smoke: bool = False,
) -> dict:
    """
    Evaluate N in {147, 293, 587} (all train facts) AND shared-correct train pool
    for both ProcrustesGlobal and ProcrustesSpan:
      - Geometry (train/test centered cosine, R^2, NMSE, in-span energy) [pure CPU]
      - Counterfactual swap accuracy on fixed train pairs & held-out test pairs [if model_B given]
    """
    test_pairs = [tuple(x) for x in bp["test_pairs"]]
    if smoke:
        test_pairs = test_pairs[:8]
    test_donor_idxs = [d for d, _ in test_pairs]
    Z_A_te = Z_A_layer[test_donor_idxs]
    Z_B_te = Z_B_layer[test_donor_idxs]

    ad_test_mask = np.array(
        [all(a.strip().lower() not in train_answer_set for a in facts[d]["aliases"]) for d, _ in test_pairs],
        dtype=bool,
    )

    shared_train = list(bp["shared_train_indices"])
    # Fixed nested train evaluation pairs drawn strictly from shared_train so we can check train swap acc
    # across N=147, 293, 587 on the EXACT same train pairs (making the 147 subset always include them
    # in seed 0, or sampling cleanly across 3 seeds!)
    rng_master = np.random.default_rng(SEED + abs(hash(pair_key)) % 10000)

    # Pre-tokenize test batches once if model is available
    test_pretok = None
    if model_B is not None:
        test_pretok = pretokenize_swap_batches(tok_B, facts, eval_B, test_pairs, device, batch_size=32)

    n_seeds = 1 if smoke else 3
    layer_B = bp["best_layer_B"]

    def _eval_pool(pool_indices: List[int], n_grid: List[int], pool_name: str) -> dict:
        pool_arr = np.array(pool_indices, dtype=np.int64)
        pool_results = {}
        for N in n_grid:
            N_eff = min(N, len(pool_arr))
            seeds_to_run = 1 if N_eff == len(pool_arr) else n_seeds
            for method_name, fit_fn in (
                ("procrustes_global", fit_procrustes_global),
                ("procrustes_span", fit_procrustes_span),
            ):
                seed_records = []
                for s_idx in range(seeds_to_run):
                    rng = np.random.default_rng(SEED + 1000 * s_idx + N_eff)
                    if N_eff == len(pool_arr):
                        sub_idx = pool_arr
                    else:
                        sub_idx = np.sort(rng.choice(pool_arr, size=N_eff, replace=False))
                    ZA_tr = Z_A_layer[sub_idx]
                    ZB_tr = Z_B_layer[sub_idx]
                    mp = fit_fn(ZA_tr, ZB_tr)
                    geom = compute_geometry_metrics(ZA_tr, ZB_tr, Z_A_te, Z_B_te, mp)

                    rec = {"seed_idx": s_idx, "N": int(N_eff), "geom": geom}
                    if model_B is not None:
                        # Test swap acc
                        z_hat_te = apply_linear_map(Z_A_te, mp)
                        te_acc, te_out = eval_patched_swaps(
                            model_B, tok_B, facts, eval_B, test_pairs, z_hat_te,
                            layer_B, device, dtype, pretok_batches=test_pretok,
                        )
                        te_out_arr = np.array(te_out, dtype=np.float64)
                        ad_acc = float(te_out_arr[ad_test_mask].mean()) if ad_test_mask.any() else None

                        # Train swap acc on pairs built from shared-correct subset inside sub_idx (capped at 120 pairs)
                        sub_shared = [int(i) for i in sub_idx if i in set(shared_train)]
                        tr_pairs = build_counterfactual_pairs(facts, sub_shared)
                        if len(tr_pairs) > (8 if smoke else 120):
                            tr_pairs = tr_pairs[: (8 if smoke else 120)]
                        if tr_pairs:
                            tr_d_idxs = [d for d, _ in tr_pairs]
                            z_hat_tr = apply_linear_map(Z_A_layer[tr_d_idxs], mp)
                            tr_acc, _ = eval_patched_swaps(
                                model_B, tok_B, facts, eval_B, tr_pairs, z_hat_tr,
                                layer_B, device, dtype, batch_size=32,
                            )
                        else:
                            tr_acc = None

                        rec["test_swap_acc"] = round(te_acc, 4)
                        rec["test_swap_ad_acc"] = round(ad_acc, 4) if ad_acc is not None else None
                        rec["train_swap_acc"] = round(tr_acc, 4) if tr_acc is not None else None
                        rec["n_train_eval_pairs"] = len(tr_pairs)
                    seed_records.append(rec)

                # Aggregate across seeds
                agg = {
                    "N": int(N_eff),
                    "n_seeds": seeds_to_run,
                    "test_centered_cosine_mean": round(float(np.mean([r["geom"]["test"]["centered_cosine"] for r in seed_records])), 5),
                    "train_centered_cosine_mean": round(float(np.mean([r["geom"]["train"]["centered_cosine"] for r in seed_records])), 5),
                    "test_r2_mean": round(float(np.mean([r["geom"]["test"]["r2"] for r in seed_records])), 5),
                    "test_nmse_mean": round(float(np.mean([r["geom"]["test"]["normalized_mse"] for r in seed_records])), 5),
                    "test_in_span_energy_mean": round(float(np.mean([r["geom"]["test_in_train_span_energy_frac"] for r in seed_records])), 5),
                }
                if model_B is not None:
                    te_accs = [r["test_swap_acc"] for r in seed_records]
                    tr_accs = [r["train_swap_acc"] for r in seed_records if r["train_swap_acc"] is not None]
                    ad_accs = [r["test_swap_ad_acc"] for r in seed_records if r["test_swap_ad_acc"] is not None]
                    agg["test_swap_acc_mean"] = round(float(np.mean(te_accs)), 4)
                    agg["test_swap_acc_std"] = round(float(np.std(te_accs)), 4)
                    agg["test_swap_ad_acc_mean"] = round(float(np.mean(ad_accs)), 4) if ad_accs else None
                    agg["train_swap_acc_mean"] = round(float(np.mean(tr_accs)), 4) if tr_accs else None
                    agg["train_minus_test_gap_pp"] = (
                        round(100.0 * (agg["train_swap_acc_mean"] - agg["test_swap_acc_mean"]), 2)
                        if agg["train_swap_acc_mean"] is not None
                        else None
                    )
                agg["seeds"] = seed_records
                pool_results.setdefault(method_name, {})[str(N_eff)] = agg

        # Compute pp/doubling slopes for each method
        slopes = {}
        for method_name, by_n in pool_results.items():
            ns = sorted(int(k) for k in by_n.keys())
            m_slopes = {}
            for a, b in zip(ns[:-1], ns[1:]):
                doublings = math.log2(b / a)
                cos_gain = by_n[str(b)]["test_centered_cosine_mean"] - by_n[str(a)]["test_centered_cosine_mean"]
                entry = {
                    "doublings": round(doublings, 3),
                    "centered_cosine_per_doubling": round(cos_gain / doublings, 5),
                    "r2_per_doubling": round(
                        (by_n[str(b)]["test_r2_mean"] - by_n[str(a)]["test_r2_mean"]) / doublings, 5
                    ),
                }
                if model_B is not None:
                    acc_pp = 100.0 * (by_n[str(b)]["test_swap_acc_mean"] - by_n[str(a)]["test_swap_acc_mean"])
                    entry["swap_acc_pp_per_doubling"] = round(acc_pp / doublings, 2)
                m_slopes[f"{a}->{b}"] = entry
            slopes[method_name] = m_slopes

        return {"pool_name": pool_name, "pool_size": len(pool_arr), "by_method": pool_results, "slopes": slopes}

    return {
        "pair_key": pair_key,
        "layers": [bp["best_layer_A"], bp["best_layer_B"]],
        "all_train_pool": _eval_pool(all_train_indices, [147, 293, 587], "all_train_587"),
        "shared_correct_pool": _eval_pool(shared_train, [147, 293, len(shared_train)], "shared_correct_train"),
    }


# ======================================================================================
# P0d: Causal Fisher Gradient Diagnostic & Spectrum (MPS float32 exact backward)
# ======================================================================================

def compute_exact_gradients_at_layer(
    model: Any,
    tok: Any,
    prompts: List[str],
    subj_tok_idxs: List[int],
    seq_lens: List[int],
    target_tok_ids: List[int],
    layer_B: int,
    device: torch.device,
    z_override: Optional[np.ndarray] = None,
    sample_unlabeled: bool = False,
    batch_size: int = 16,
    seed: int = SEED,
) -> dict:
    """
    Compute exact gradients g_i = grad_{z_B} log p_B(y_i | z_B) at layer_B, subj_tok_idx
    using float32 forward/backward with frozen model weights (only activation grad!).
    Also returns logp_target, margin_top1_vs_other, and top1_pred_id at the evaluated z_B.
    """
    layers = get_decoder_layers(model)
    target_layer = layers[layer_B]
    n = len(prompts)
    D = int(model.config.hidden_size)

    grads = np.zeros((n, D), dtype=np.float32)
    z_clean = np.zeros((n, D), dtype=np.float32)
    logp_tgt = np.zeros(n, dtype=np.float32)
    margins = np.zeros(n, dtype=np.float32)
    top1_ids = np.zeros(n, dtype=np.int64)
    sampled_ids = np.zeros(n, dtype=np.int64)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    ctx: Dict[str, Any] = {"subj_pos": None, "z_override": None, "z_leaf": None, "z_captured": None}

    def hook_fn(_mod, _inp, out):
        is_tup = isinstance(out, tuple)
        h = out[0] if is_tup else out
        bsz = h.shape[0]
        b_ar = torch.arange(bsz, device=h.device)
        spos = ctx["subj_pos"]
        h_subj = h[b_ar, spos, :].detach()
        ctx["z_captured"] = h_subj
        if ctx["z_override"] is not None:
            z_leaf = ctx["z_override"].to(device=h.device, dtype=h.dtype).detach().requires_grad_(True)
        else:
            z_leaf = h_subj.clone().requires_grad_(True)
        ctx["z_leaf"] = z_leaf
        # Out-of-place scatter-like replacement so autograd flows cleanly into z_leaf
        h_new = h.clone()
        h_new[b_ar, spos, :] = z_leaf
        return (h_new,) + out[1:] if is_tup else h_new

    handle = target_layer.register_forward_hook(hook_fn)
    try:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            b_prompts = prompts[start:end]
            enc = tok(b_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            t_pad = input_ids.shape[1]
            spos = []
            for i in range(start, end):
                pad_len = t_pad - seq_lens[i]
                pos = max(0, min(t_pad - 1, pad_len + subj_tok_idxs[i]))
                spos.append(pos)
            ctx["subj_pos"] = torch.tensor(spos, dtype=torch.long, device=device)
            ctx["z_override"] = (
                torch.from_numpy(z_override[start:end].astype(np.float32))
                if z_override is not None
                else None
            )

            with torch.enable_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                logits = out.logits[:, -1, :].float()  # [B, V]
                log_probs = F.log_softmax(logits, dim=-1)
                bsz = logits.shape[0]
                b_ar = torch.arange(bsz, device=device)

                if sample_unlabeled:
                    probs_cpu = torch.softmax(logits.detach().cpu(), dim=-1)
                    y_samp = torch.multinomial(probs_cpu, num_samples=1, generator=gen).squeeze(-1)
                    y_ids = y_samp.to(device)
                    sampled_ids[start:end] = y_samp.numpy()
                else:
                    y_ids = torch.tensor(target_tok_ids[start:end], dtype=torch.long, device=device)

                lp_y = log_probs[b_ar, y_ids]
                loss_sum = lp_y.sum()
                g_leaf = torch.autograd.grad(loss_sum, ctx["z_leaf"], retain_graph=False)[0]

            grads[start:end] = g_leaf.detach().float().cpu().numpy()
            z_clean[start:end] = ctx["z_captured"].detach().float().cpu().numpy()
            logp_tgt[start:end] = lp_y.detach().float().cpu().numpy()
            top1 = logits.detach().argmax(dim=-1)
            top1_ids[start:end] = top1.cpu().numpy()

            # Compute margin = logp(target) - max_{w != target} logp(w)
            lp_det = log_probs.detach()
            lp_other = lp_det.clone()
            lp_other[b_ar, y_ids] = -1e9
            max_other = lp_other.max(dim=-1).values
            margins[start:end] = (lp_y.detach() - max_other).float().cpu().numpy()

            del out, logits, log_probs, g_leaf
    finally:
        handle.remove()

    return {
        "grads": grads,
        "z_clean": z_clean,
        "logp_tgt": logp_tgt,
        "margins": margins,
        "top1_ids": top1_ids,
        "sampled_ids": sampled_ids,
    }


def analyze_fisher_spectrum(G: np.ndarray, Z_states: np.ndarray) -> dict:
    """
    Compute exact eigenspectrum, effective rank, and overlap with top activation PCA
    of empirical Fisher matrix F = (1/N) G^T G in float64 on CPU.
    """
    N, D = G.shape
    Gf = G.astype(np.float64)
    # Thin SVD of G / sqrt(N): singular values s_k => eigenvalues lam_k = s_k^2
    _, s_f, Vt_f = np.linalg.svd(Gf / math.sqrt(max(N, 1)), full_matrices=False)
    eig = s_f ** 2
    tr = float(np.sum(eig))
    pr_eff_rank = float((tr ** 2) / max(float(np.sum(eig ** 2)), 1e-30))
    p = eig / max(tr, 1e-30)
    p_pos = p[p > 1e-18]
    shannon_eff_rank = float(math.exp(-float(np.sum(p_pos * np.log(p_pos)))))

    cum = np.cumsum(eig) / max(tr, 1e-30)
    cum_at_k = {}
    for k in (1, 4, 8, 16, 32, 64, 128, 256, 512):
        if k <= len(cum):
            cum_at_k[str(k)] = round(float(cum[k - 1]), 5)

    # Compare Fisher top-k subspace vs Activation PCA top-k subspace!
    Zc = Z_states.astype(np.float64) - Z_states.astype(np.float64).mean(axis=0, keepdims=True)
    _, s_z, Vt_z = np.linalg.svd(Zc / math.sqrt(max(len(Zc), 1)), full_matrices=False)
    eig_z = s_z ** 2
    tr_z = float(np.sum(eig_z))
    pr_z = float((tr_z ** 2) / max(float(np.sum(eig_z ** 2)), 1e-30))

    # How much Fisher trace lies inside the top-k activation PCs?
    # tr(P_Z_k F) / tr(F) = ||(G / sqrt(N)) @ V_Z_k||_F^2 / tr(F)
    fisher_in_pca_k = {}
    subspace_overlap_k = {}
    for k in (8, 16, 32, 64, 128, 256):
        if k <= min(Vt_z.shape[0], Vt_f.shape[0]):
            Vk_z = Vt_z[:k].T  # [D, k]
            Vk_f = Vt_f[:k].T  # [D, k]
            proj_energy = float(np.sum(((Gf / math.sqrt(max(N, 1))) @ Vk_z) ** 2)) / max(tr, 1e-30)
            fisher_in_pca_k[str(k)] = round(proj_energy, 5)
            subspace_overlap_k[str(k)] = round(float(np.sum((Vk_f.T @ Vk_z) ** 2) / k), 5)

    return {
        "N": int(N),
        "D": int(D),
        "trace_F": round(tr, 6),
        "top10_eigenvalues": [round(float(x), 6) for x in eig[:10]],
        "effective_rank_participation_ratio": round(pr_eff_rank, 2),
        "effective_rank_shannon": round(shannon_eff_rank, 2),
        "cum_fisher_energy_top_k": cum_at_k,
        "activation_pca_eff_rank_pr": round(pr_z, 2),
        "fisher_energy_captured_by_top_k_activation_pca": fisher_in_pca_k,
        "top_k_subspace_overlap_fisher_vs_pca": subspace_overlap_k,
    }


def run_p0d_for_pair(
    pair_key: str,
    bp: dict,
    facts: List[dict],
    eval_B: List[dict],
    Z_A_layer: np.ndarray,
    Z_B_layer: np.ndarray,
    model_B_f32: Any,
    tok_B: Any,
    device: torch.device,
    all_train_indices: List[int],
    smoke: bool = False,
) -> dict:
    """
    Exact Fisher gradient diagnostic for pair A -> B at swap site l_B:
      1. Exact gradients g_i on test counterfactual swap pairs (at self-swap target z_B_donor)
         + forward pass at mapped z_hat_B to measure exact delta logp & verify swap outcomes.
      2. ROC-AUC of |g_i^T delta_i| (Fable's Go bar >= 0.75!) and baselines.
      3. Empirical Fisher matrix F_B on train facts (both gold-answer and unlabeled sampled-y!)
         + eigenspectrum, effective rank, and overlap with activation PCA.
    """
    layer_B = bp["best_layer_B"]
    shared_train = list(bp["shared_train_indices"])
    mp = fit_procrustes_global(Z_A_layer[shared_train], Z_B_layer[shared_train])

    test_pairs = [tuple(x) for x in bp["test_pairs"]]
    test_outcomes = list(bp["test_outcomes"])
    self_outcomes = list(bp["self_swap_at_lB_outcomes"])
    if smoke:
        test_pairs = test_pairs[:8]
        test_outcomes = test_outcomes[:8]
        self_outcomes = self_outcomes[:8]

    donor_idxs = [d for d, _ in test_pairs]
    base_idxs = [b for _, b in test_pairs]
    z_B_donor = Z_B_layer[donor_idxs].astype(np.float32)
    z_hat_B = apply_linear_map(Z_A_layer[donor_idxs], mp).astype(np.float32)
    delta = (z_hat_B - z_B_donor).astype(np.float64)  # [N_test, D]

    # Target answer first token ID for each donor fact
    def _first_ans_tok(fact_idx: int) -> int:
        ans = facts[fact_idx]["answer"]
        ids = tok_B.encode(" " + ans, add_special_tokens=False)
        if not ids:
            ids = tok_B.encode(ans, add_special_tokens=False)
        return int(ids[0])

    test_prompts = [facts[b]["direct_prompt"] for b in base_idxs]
    test_subj_pos = [eval_B[b]["subj_tok_idx_direct"] for b in base_idxs]
    test_seq_lens = [eval_B[b]["seq_len_direct"] for b in base_idxs]
    test_tgt_ids = [_first_ans_tok(d) for d in donor_idxs]

    # 1a. Exact gradients at self-swap point z_B_donor on counterfactual base prompt b_i
    t0 = time.time()
    res_self = compute_exact_gradients_at_layer(
        model_B_f32, tok_B, test_prompts, test_subj_pos, test_seq_lens, test_tgt_ids,
        layer_B=layer_B, device=device, z_override=z_B_donor, sample_unlabeled=False, batch_size=16,
    )
    # 1b. Exact logp / margin at Procrustes-mapped point z_hat_B (for Taylor fidelity check!)
    res_hat = compute_exact_gradients_at_layer(
        model_B_f32, tok_B, test_prompts, test_subj_pos, test_seq_lens, test_tgt_ids,
        layer_B=layer_B, device=device, z_override=z_hat_B, sample_unlabeled=False, batch_size=16,
    )
    print(f"  [P0d {pair_key}] test gradients + Taylor check done in {time.time() - t0:.1f}s")

    # 2. Train empirical Fisher matrix (gold answer + unlabeled sampled-y)
    tr_idxs = all_train_indices[: (16 if smoke else len(all_train_indices))]
    tr_prompts = [facts[i]["direct_prompt"] for i in tr_idxs]
    tr_subj_pos = [eval_B[i]["subj_tok_idx_direct"] for i in tr_idxs]
    tr_seq_lens = [eval_B[i]["seq_len_direct"] for i in tr_idxs]
    tr_tgt_ids = [_first_ans_tok(i) for i in tr_idxs]

    t1 = time.time()
    res_tr_gold = compute_exact_gradients_at_layer(
        model_B_f32, tok_B, tr_prompts, tr_subj_pos, tr_seq_lens, tr_tgt_ids,
        layer_B=layer_B, device=device, z_override=None, sample_unlabeled=False, batch_size=16,
    )
    res_tr_unlab = compute_exact_gradients_at_layer(
        model_B_f32, tok_B, tr_prompts, tr_subj_pos, tr_seq_lens, tr_tgt_ids,
        layer_B=layer_B, device=device, z_override=None, sample_unlabeled=True, batch_size=16,
    )
    print(f"  [P0d {pair_key}] train Fisher (N={len(tr_idxs)}, gold + unlabeled) done in {time.time() - t1:.1f}s")

    # Save Fisher gradients to cache for reuse / auditing
    if not smoke:
        rec_name = bp["recipient"]
        torch.save(
            {
                "pair_key": pair_key,
                "layer_B": layer_B,
                "train_indices": tr_idxs,
                "G_train_gold": torch.from_numpy(res_tr_gold["grads"]),
                "G_train_unlab": torch.from_numpy(res_tr_unlab["grads"]),
                "G_test_self": torch.from_numpy(res_self["grads"]),
                "delta_test": torch.from_numpy(delta.astype(np.float32)),
            },
            CACHE_DIR / f"pilot0_fisher_{rec_name}_l{layer_B}.pt",
        )

    # 3. Compute ROC-AUC metrics & Taylor fidelity
    G_te = res_self["grads"].astype(np.float64)  # [N_test, D]
    G_tr_gold = res_tr_gold["grads"].astype(np.float64)
    G_tr_unlab = res_tr_unlab["grads"].astype(np.float64)

    g_dot_delta = np.sum(G_te * delta, axis=1)  # first-order approx of logp(z_hat) - logp(z_self)
    abs_g_dot_delta = np.abs(g_dot_delta)
    neg_g_dot_delta = -g_dot_delta
    l2_delta = np.linalg.norm(delta, axis=1)
    rel_l2_delta = l2_delta / np.maximum(np.linalg.norm(z_B_donor.astype(np.float64), axis=1), 1e-12)
    g_norm = np.linalg.norm(G_te, axis=1)
    # Global train Fisher Mahalanobis norm: sqrt(delta^T F_tr delta) = ||G_tr @ delta^T||_2 / sqrt(N_tr)
    fisher_gold_norm = np.linalg.norm(delta @ G_tr_gold.T, axis=1) / math.sqrt(len(G_tr_gold))
    fisher_unlab_norm = np.linalg.norm(delta @ G_tr_unlab.T, axis=1) / math.sqrt(len(G_tr_unlab))
    # Actual log-prob change
    actual_dlogp = (res_hat["logp_tgt"] - res_self["logp_tgt"]).astype(np.float64)
    margin_self = res_self["margins"].astype(np.float64)
    margin_norm_score = neg_g_dot_delta / np.maximum(margin_self, 0.25)

    # First-order Taylor correlation
    if np.std(g_dot_delta) > 1e-12 and np.std(actual_dlogp) > 1e-12:
        taylor_pearson = float(np.corrcoef(g_dot_delta, actual_dlogp)[0, 1])
    else:
        taylor_pearson = 0.0
    # Also midpoint gradient correlation (trapezoidal / path-averaged)
    g_mid = 0.5 * (G_te + res_hat["grads"].astype(np.float64))
    mid_dot_delta = np.sum(g_mid * delta, axis=1)
    if np.std(mid_dot_delta) > 1e-12 and np.std(actual_dlogp) > 1e-12:
        midpoint_pearson = float(np.corrcoef(mid_dot_delta, actual_dlogp)[0, 1])
    else:
        midpoint_pearson = 0.0

    # Failure labels: fail = 1 means Procrustes swap FAILED (0 means succeeded)
    # Check BOTH cached outcomes (from bf16 baseline) AND f32 top1 match
    y_fail_cached = np.array([1 - int(o) for o in test_outcomes], dtype=np.int64)
    y_fail_f32_tok1 = np.array(
        [0 if int(res_hat["top1_ids"][i]) == int(test_tgt_ids[i]) else 1 for i in range(len(test_pairs))],
        dtype=np.int64,
    )
    self_valid_mask = np.array([int(o) == 1 for o in self_outcomes], dtype=bool)

    predictors = {
        "abs_g_dot_delta": abs_g_dot_delta,
        "neg_g_dot_delta": neg_g_dot_delta,
        "abs_g_mid_dot_delta": np.abs(mid_dot_delta),
        "neg_g_mid_dot_delta": -mid_dot_delta,
        "margin_normalized_neg_g_dot_delta": margin_norm_score,
        "actual_neg_delta_logp": -actual_dlogp,
        "l2_delta": l2_delta,
        "rel_l2_delta": rel_l2_delta,
        "grad_norm_only": g_norm,
        "global_train_fisher_gold_norm": fisher_gold_norm,
        "global_train_fisher_unlab_norm": fisher_unlab_norm,
        "neg_margin_self": -margin_self,
    }

    def _auc_table(y_fail: np.ndarray, mask: np.ndarray) -> dict:
        ym = y_fail[mask]
        tbl = {
            "n": int(len(ym)),
            "n_fail": int(np.sum(ym == 1)),
            "n_pass": int(np.sum(ym == 0)),
            "predictors": {},
        }
        for pname, sc in predictors.items():
            scm = sc[mask]
            auc = compute_roc_auc(ym, scm)
            lo, hi = bootstrap_roc_auc_ci(ym, scm, n_boot=500 if smoke else 2000)
            tbl["predictors"][pname] = {
                "auc": round(auc, 4) if auc is not None else None,
                "ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
                "mean_on_fail": round(float(np.mean(scm[ym == 1])), 5) if np.any(ym == 1) else None,
                "mean_on_pass": round(float(np.mean(scm[ym == 0])), 5) if np.any(ym == 0) else None,
            }
        return tbl

    # 4. Empirical Fisher spectra (train gold, train unlabeled, test swap)
    spec_tr_gold = analyze_fisher_spectrum(G_tr_gold, Z_B_layer[tr_idxs])
    spec_tr_unlab = analyze_fisher_spectrum(G_tr_unlab, Z_B_layer[tr_idxs])
    spec_te_swap = analyze_fisher_spectrum(G_te, z_B_donor)

    # Also check subspace alignment between Gold-Train Fisher and Unlabeled-Train Fisher!
    _, _, Vt_g = np.linalg.svd(G_tr_gold / math.sqrt(len(G_tr_gold)), full_matrices=False)
    _, _, Vt_u = np.linalg.svd(G_tr_unlab / math.sqrt(len(G_tr_unlab)), full_matrices=False)
    gold_vs_unlab_overlap = {}
    for k in (8, 16, 32, 64, 128):
        if k <= min(Vt_g.shape[0], Vt_u.shape[0]):
            gold_vs_unlab_overlap[str(k)] = round(float(np.sum((Vt_g[:k] @ Vt_u[:k].T) ** 2) / k), 5)

    return {
        "pair_key": pair_key,
        "layers": [bp["best_layer_A"], layer_B],
        "taylor_diagnostics": {
            "first_order_pearson_r": round(taylor_pearson, 4),
            "midpoint_trapezoidal_pearson_r": round(midpoint_pearson, 4),
            "mean_g_dot_delta": round(float(np.mean(g_dot_delta)), 4),
            "mean_actual_dlogp": round(float(np.mean(actual_dlogp)), 4),
            "mean_abs_g_dot_delta": round(float(np.mean(abs_g_dot_delta)), 4),
            "mean_l2_delta": round(float(np.mean(l2_delta)), 4),
            "f32_vs_bf16_outcome_agreement": round(
                float(np.mean(y_fail_cached == y_fail_f32_tok1)), 4
            ),
        },
        "auc_all_test_cached_outcomes": _auc_table(y_fail_cached, np.ones(len(y_fail_cached), dtype=bool)),
        "auc_self_swap_valid_cached_outcomes": _auc_table(y_fail_cached, self_valid_mask),
        "auc_all_test_f32_tok1": _auc_table(y_fail_f32_tok1, np.ones(len(y_fail_f32_tok1), dtype=bool)),
        "fisher_spectrum_train_gold": spec_tr_gold,
        "fisher_spectrum_train_unlabeled": spec_tr_unlab,
        "fisher_spectrum_test_swap": spec_te_swap,
        "gold_vs_unlabeled_fisher_top_k_overlap": gold_vs_unlab_overlap,
    }


# ======================================================================================
# Main Orchestrator
# ======================================================================================

def main():
    parser = argparse.ArgumentParser(description="Pilot 0: Cache diagnostics + Causal Fisher AUC")
    parser.add_argument("--mode", choices=["cpu", "all"], default="all")
    parser.add_argument("--smoke", action="store_true", help="Fast 8-item smoke test")
    args = parser.parse_args()

    t_start = time.time()
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(BASELINE_RESULTS_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    with open(REFEREE_CONTROLS_PATH, "r", encoding="utf-8") as f:
        ref_ctrl = json.load(f)
    with open(SYSTEMS_RESULTS_PATH, "r", encoding="utf-8") as f:
        systems = json.load(f)
    with open(CACHE_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    facts = dataset["facts"]
    all_train_indices = [f["idx"] for f in facts if f["split"] == "train"]
    train_answer_set = set()
    for i in all_train_indices:
        for a in facts[i]["aliases"]:
            train_answer_set.add(a.strip().lower())

    has_mps = torch.backends.mps.is_available()
    print(f"[pilot0] mode={args.mode}, smoke={args.smoke}, mps_available={has_mps}")

    # Load cached direct_subj states for the needed models
    needed_models = sorted({
        m for pk in set(FOCUS_PAIRS_P0AB + FOCUS_PAIRS_P0D)
        for m in pk.split("->")
    })
    states_direct: Dict[str, torch.Tensor] = {}
    evals: Dict[str, dict] = {}
    for mk in needed_models:
        st = torch.load(CACHE_DIR / f"{mk}_states.pt", map_location="cpu", weights_only=False)
        states_direct[mk] = st["direct_subj"]
        del st
        with open(CACHE_DIR / f"{mk}_eval.json", "r", encoding="utf-8") as f:
            evals[mk] = json.load(f)

    results: Dict[str, Any] = {
        "meta": {
            "mode": args.mode,
            "smoke": args.smoke,
            "mps_available": has_mps,
            "seed": SEED,
            "focus_pairs_p0ab": FOCUS_PAIRS_P0AB,
            "focus_pairs_p0d": FOCUS_PAIRS_P0D,
        }
    }

    # ---- P0c (Pure CPU, always runs!) ----
    print("\n[P0c] Running dark-slice-beyond-GlobalFV analysis across all 20 pairs...")
    p0c = run_p0c_dark_beyond_globalfv(dataset, baseline, ref_ctrl, systems)
    results["P0c_dark_beyond_globalfv"] = p0c
    p_all = p0c["pooled_all_20_pairs"]
    p_cf = p0c["pooled_cross_family_18_pairs"]
    print(
        f"  [P0c ALL 20 pairs] dark_total={p_all['n_dark_total']} -> wrong_under_GlobalFV={p_all['n_dark_wrong_gfv']} "
        f"(GlobalFV solved {p_all['globalfv_resolved_dark_frac']:.1%}) | "
        f"matched={p_all['matched_hits']}/{p_all['n_dark_wrong_gfv']} ({p_all['matched_rate']:.1%}) vs "
        f"shuffled={p_all['shuffled_hits']}/{p_all['n_dark_wrong_gfv']} ({p_all['shuffled_rate']:.1%}) vs "
        f"random={p_all['random_hits']}/{p_all['n_dark_wrong_gfv']} ({p_all['random_rate']:.1%}) | "
        f"diff={p_all['matched_minus_shuffled_pp']:+.2f} pp (disc m1s0={p_all['disc_m1_s0']}, m0s1={p_all['disc_m0_s1']})"
    )
    print(
        f"  [P0c Cross-Fam 18] wrong_under_GlobalFV={p_cf['n_dark_wrong_gfv']} | "
        f"matched={p_cf['matched_rate']:.1%} vs shuffled={p_cf['shuffled_rate']:.1%} "
        f"(diff={p_cf['matched_minus_shuffled_pp']:+.2f} pp)"
    )

    if args.mode == "cpu":
        # Pure-CPU P0b geometry curve (cosine / R^2 / NMSE / in-span energy across N in {147, 293, 587})
        print("\n[P0b-CPU] Running pure-CPU geometry learning curves (N in {147, 293, 587})...")
        p0b_cpu = {}
        for pk in FOCUS_PAIRS_P0AB:
            bp = baseline["pairs"][pk]
            mk_A, mk_B = bp["donor"], bp["recipient"]
            ZA = states_direct[mk_A][:, bp["best_layer_A"], :].float().numpy()
            ZB = states_direct[mk_B][:, bp["best_layer_B"], :].float().numpy()
            p0b_cpu[pk] = run_p0b_for_pair(
                pk, bp, facts, ZA, ZB, all_train_indices, train_answer_set,
                model_B=None, smoke=args.smoke,
            )
            at = p0b_cpu[pk]["all_train_pool"]["by_method"]["procrustes_global"]
            sp = p0b_cpu[pk]["all_train_pool"]["by_method"]["procrustes_span"]
            print(
                f"  [P0b-CPU {pk}] ProcGlobal cos_c: "
                + ", ".join(f"N={n}:{at[n]['test_centered_cosine_mean']:.4f}(R2={at[n]['test_r2_mean']:.3f})" for n in ("147", "293", "587"))
                + " | Span cos_c: "
                + ", ".join(f"N={n}:{sp[n]['test_centered_cosine_mean']:.4f}" for n in ("147", "293", "587"))
            )
        results["P0b_geometry_cpu"] = p0b_cpu
        results["meta"]["elapsed_sec"] = round(time.time() - t_start, 2)
        with open(OUT_CPU_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[pilot0] Saved pure-CPU diagnostics to {OUT_CPU_PATH} in {time.time() - t_start:.1f}s")
        return

    # ---- Full Mode (MPS GPU or CPU): group by RECIPIENT model B so each recipient loads ONCE! ----
    device = torch.device("mps") if has_mps else torch.device("cpu")
    eval_dtype = torch.bfloat16 if has_mps else torch.float32

    pairs_by_rec: Dict[str, List[str]] = defaultdict(list)
    for pk in FOCUS_PAIRS_P0AB:
        pairs_by_rec[pk.split("->")[1]].append(pk)

    results["P0a_error_taxonomy"] = {}
    results["P0b_learning_curve"] = {}
    results["P0d_fisher_diagnostic"] = {}

    for rec_mk, rec_pairs in pairs_by_rec.items():
        model_dir = Path(manifest["models"][rec_mk]["local_path"])
        print(f"\n[pilot0] === Loading recipient {rec_mk} on {device} (eval dtype={eval_dtype}) ===")
        tok_B = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        tok_B.padding_side = "left"
        if tok_B.pad_token is None:
            tok_B.pad_token = tok_B.eos_token
        model_B = AutoModelForCausalLM.from_pretrained(
            str(model_dir), local_files_only=True, dtype=eval_dtype
        )
        model_B.to(device)
        model_B.eval()
        eval_B = evals[rec_mk]["facts"]

        # P0a + P0b in fast bf16 (or f32 on CPU)
        for pk in rec_pairs:
            bp = baseline["pairs"][pk]
            mk_A = bp["donor"]
            la, lb = bp["best_layer_A"], bp["best_layer_B"]
            ZA = states_direct[mk_A][:, la, :].float().numpy()
            ZB = states_direct[rec_mk][:, lb, :].float().numpy()

            # P0a
            t0 = time.time()
            shared_tr = list(bp["shared_train_indices"])
            mp = fit_procrustes_global(ZA[shared_tr], ZB[shared_tr])
            t_pairs = [tuple(x) for x in bp["test_pairs"]]
            if args.smoke:
                t_pairs = t_pairs[:8]
            d_idxs = [d for d, _ in t_pairs]
            b_idxs = [b for _, b in t_pairs]
            z_hat = apply_linear_map(ZA[d_idxs], mp)
            outs, gens = eval_swaps_with_generations(
                model_B, tok_B, facts, eval_B, t_pairs, z_hat, lb, device, eval_dtype, batch_size=32
            )
            tax = classify_p0a_taxonomy(
                dataset, t_pairs, outs, gens, z_hat, ZB[d_idxs], ZB[b_idxs], train_answer_set
            )
            tax["layers"] = [la, lb]
            tax["baseline_cached_acc"] = bp["test_transfer_acc"]
            results["P0a_error_taxonomy"][pk] = tax
            ov = tax["overall"]
            ff = ov["frac_of_failures"]
            print(
                f"  [P0a {pk}] acc={ov['acc']:.4f} (cached={bp['test_transfer_acc']:.4f}), "
                f"n_fail={ov['n_failures']}/{ov['n_total']} | "
                f"right_type_wrong_ent={ff['right_type_wrong_entity']:.1%}, "
                f"base_survived={ff['base_subject_survived']:.1%}, "
                f"off_manifold={ff['off_manifold_garbled_other']:.1%} ({time.time() - t0:.1f}s)"
            )

            # P0b
            t1 = time.time()
            p0b_res = run_p0b_for_pair(
                pk, bp, facts, ZA, ZB, all_train_indices, train_answer_set,
                model_B=model_B, tok_B=tok_B, eval_B=eval_B,
                device=device, dtype=eval_dtype, smoke=args.smoke,
            )
            results["P0b_learning_curve"][pk] = p0b_res
            at = p0b_res["all_train_pool"]["by_method"]["procrustes_global"]
            sl = p0b_res["all_train_pool"]["slopes"]["procrustes_global"]
            print(
                f"  [P0b {pk}] all-train ProcGlobal test_acc: "
                + ", ".join(f"N={n}:{at[n]['test_swap_acc_mean']:.3f}(tr={at[n]['train_swap_acc_mean']:.3f})" for n in ("147", "293", "587"))
                + f" | slopes={sl} ({time.time() - t1:.1f}s)"
            )

        # P0d: exact float32 gradients for pairs whose recipient is rec_mk
        p0d_pairs_here = [pk for pk in FOCUS_PAIRS_P0D if pk.endswith(f"->{rec_mk}")]
        if p0d_pairs_here:
            print(f"\n  [P0d] Casting {rec_mk} to float32 on {device} (frozen weights) for exact Fisher gradients...")
            if eval_dtype != torch.float32:
                model_B = model_B.to(dtype=torch.float32)
            for p in model_B.parameters():
                p.requires_grad_(False)
            model_B.eval()
            for pk in p0d_pairs_here:
                bp = baseline["pairs"][pk]
                mk_A = bp["donor"]
                ZA = states_direct[mk_A][:, bp["best_layer_A"], :].float().numpy()
                ZB = states_direct[rec_mk][:, bp["best_layer_B"], :].float().numpy()
                p0d_res = run_p0d_for_pair(
                    pk, bp, facts, eval_B, ZA, ZB, model_B, tok_B, device,
                    all_train_indices=all_train_indices, smoke=args.smoke,
                )
                results["P0d_fisher_diagnostic"][pk] = p0d_res
                a_all = p0d_res["auc_all_test_cached_outcomes"]["predictors"]
                a_sv = p0d_res["auc_self_swap_valid_cached_outcomes"]["predictors"]
                sp_g = p0d_res["fisher_spectrum_train_gold"]
                sp_u = p0d_res["fisher_spectrum_train_unlabeled"]
                print(
                    f"  [P0d {pk}] AUC(all): |g.d|={a_all['abs_g_dot_delta']['auc']}, "
                    f"-g.d={a_all['neg_g_dot_delta']['auc']}, L2={a_all['l2_delta']['auc']}, "
                    f"F_gold={a_all['global_train_fisher_gold_norm']['auc']}, "
                    f"F_unlab={a_all['global_train_fisher_unlab_norm']['auc']}"
                )
                print(
                    f"  [P0d {pk}] AUC(self-valid): |g.d|={a_sv['abs_g_dot_delta']['auc']}, "
                    f"-g.d={a_sv['neg_g_dot_delta']['auc']}, L2={a_sv['l2_delta']['auc']} | "
                    f"Taylor r={p0d_res['taylor_diagnostics']['first_order_pearson_r']:.3f} "
                    f"(mid={p0d_res['taylor_diagnostics']['midpoint_trapezoidal_pearson_r']:.3f})"
                )
                print(
                    f"  [P0d {pk}] Fisher eff_rank (PR/Shannon): gold={sp_g['effective_rank_participation_ratio']}/{sp_g['effective_rank_shannon']}, "
                    f"unlab={sp_u['effective_rank_participation_ratio']}/{sp_u['effective_rank_shannon']} | "
                    f"Fisher-in-top64-PCA={sp_g['fisher_energy_captured_by_top_k_activation_pca'].get('64')}"
                )

        del model_B
        gc.collect()
        if has_mps:
            torch.mps.empty_cache()

    results["meta"]["elapsed_sec"] = round(time.time() - t_start, 2)
    out_path = LAB_DIR / ("results_pilot0_smoke.json" if args.smoke else "results_pilot0.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[pilot0] COMPLETE! Saved to {out_path} in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
