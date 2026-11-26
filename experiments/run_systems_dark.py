#!/usr/bin/env python3
"""
run_systems_dark.py — Full Systems-Level & Multi-Model Plato's Key Experiment Suite.

Runs all 5 experimental pillars on cached cross-family LLMs:
  1. Multi-Donor Platonic Barycenter (uniform, val-weighted, Theorem 4 GLS inv_cov,
     norm restoration ablations, Monge-Procrustes barycenter, Fig. 6 native GPA barycenter)
  2. Generalized Procrustes Analysis (GPA, Fig. 6) + Closed-Loop Holonomy + Low-Overlap Sync
  3. Breaking the Single-Layer Self-Swap Ceiling (Gain-Tuned Replacement, Difference-Vector
     Steering, Contiguous Multi-Layer Windows W=2..4, Additive MLP-Output-Only Delta z_MLP,
     Dual-Token last_subj + last_prompt, Combined Window+Dual-Token)
  4. Dark-Memory Recovery & Scientific Audit (Same-Model ICL/Paraphrase Unlocking Ceiling,
     Single-Donor Procrustes vs Monge-Procrustes vs Multi-Donor Barycenter vs Multi-Layer,
     Collateral Damage / Retention Rate on already-recalled facts, Net Recall Gain,
     Weaker-Donor -> Stronger-Recipient slice)
  5. 2-Hop Unspoken Latent Bridge Transfer (2-Hop->2-Hop zero-shot via 1-hop map,
     1-Hop Bridge Entity Key -> 2-Hop intermediate-layer injection incl. cross-relation
     no-value-leakage control, and 2-hop compositional dark-memory rescue)

All hyperparameters (layer windows, gain alpha, dual-token layer, shrinkage lam, barycenter
weights & global spread scale) are selected STRICTLY on validation splits (zero test leakage).
Saves per-example binary outcomes in `results_systems.json` for independent audit.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

LAB_DIR = Path(__file__).resolve().parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

import systems
import maps as map_lib
from extract_and_baseline import (
    CACHE_DIR,
    DATASET_2HOP_PATH,
    DATASET_PATH,
    RESULTS_PATH as BASELINE_RESULTS_PATH,
    SEED,
    _alias_first_token_ids,
    _greedy_decode_from_prefill,
    build_counterfactual_pairs,
    check_answer_match,
    get_decoder_layers,
    load_model_and_tokenizer,
    locate_last_subject_token,
)

OUTPUT_PATH = LAB_DIR / "results_systems.json"

MODEL_FAMILIES: Dict[str, str] = {
    "qwen2.5-1.5b": "qwen2.5",
    "qwen2.5-3b": "qwen2.5",
    "smollm2-1.7b": "smollm2",
    "granite-3.1-2b": "granite",
    "falcon3-3b": "falcon3",
}


def get_family(model_key: str) -> str:
    return MODEL_FAMILIES.get(model_key, model_key.split("-")[0])


_PROC_MEMO: Dict[Tuple[Any, ...], systems.ProcrustesMap] = {}


def cached_fit_procrustes(
    tag: Tuple[Any, ...],
    X_a: np.ndarray,
    X_b: np.ndarray,
) -> systems.ProcrustesMap:
    """Exact memoization cache for fit_procrustes (zero approximation, avoids duplicate SVDs)."""
    if tag not in _PROC_MEMO:
        _PROC_MEMO[tag] = systems.fit_procrustes(X_a, X_b)
    return _PROC_MEMO[tag]


def _json_clean(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 6)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    return obj


# =============================================================================
# Fast Exact Multi-Regime Patched Evaluation Engine (Two-Stage Exact Decoding)
# =============================================================================


@torch.no_grad()
def eval_multi_regime_swaps(
    model: Any,
    tok: Any,
    prompts: Sequence[str],
    subj_tok_idxs: Sequence[int],
    seq_lens: Sequence[int],
    target_aliases_list: Sequence[Sequence[str]],
    relations: Sequence[str],
    residual_replace: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    residual_add: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    mlp_replace: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    residual_multiplier: float = 1.0,
    batch_size: int = 32,
    max_new_tokens: int = 6,
    fast_first_tok_only: bool = False,
) -> Tuple[float, List[int]]:
    """
    Evaluate arbitrary multi-layer / multi-token / MLP-only / additive-delta patches
    using exact two-stage greedy decoding matching `extract_and_baseline.py`.

    Each dict maps `layer_idx (0-based)` -> `{"subj": np.ndarray(N, D), "last": np.ndarray(N, D)}`.
    For `mlp_replace`, cached `direct_mlp_subj` already includes `residual_multiplier`,
    so we divide by `residual_multiplier` before writing to `layer.mlp` output so the
    model's `residual + mlp_out * residual_multiplier` injects the exact target delta!
    """
    n = len(prompts)
    if n == 0:
        return 0.0, []

    res_rep = residual_replace or {}
    res_add = residual_add or {}
    mlp_rep = mlp_replace or {}

    alias_first_sets = [_alias_first_token_ids(tok, aliases) for aliases in target_aliases_list]
    outcomes = [0] * n
    inv_rm = 1.0 / residual_multiplier if abs(residual_multiplier) > 1e-12 else 1.0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b_prompts = list(prompts[start:end])
        enc = tok(b_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        t_pad = input_ids.shape[1]
        bsz = end - start

        spos: List[int] = []
        lpos: List[int] = [t_pad - 1] * bsz
        for bi in range(bsz):
            idx = start + bi
            pad_len = t_pad - int(seq_lens[idx])
            spos.append(max(0, min(t_pad - 1, pad_len + int(subj_tok_idxs[idx]))))

        spec = systems.BatchPatchSpec()
        for l_idx, pos_map in res_rep.items():
            ops = []
            if "subj" in pos_map:
                ops.append((spos, torch.from_numpy(np.asarray(pos_map["subj"][start:end], dtype=np.float32))))
            if "last" in pos_map:
                ops.append((lpos, torch.from_numpy(np.asarray(pos_map["last"][start:end], dtype=np.float32))))
            if ops:
                spec.residual_replace[int(l_idx)] = ops

        for l_idx, pos_map in res_add.items():
            ops = []
            if "subj" in pos_map:
                ops.append((spos, torch.from_numpy(np.asarray(pos_map["subj"][start:end], dtype=np.float32))))
            if "last" in pos_map:
                ops.append((lpos, torch.from_numpy(np.asarray(pos_map["last"][start:end], dtype=np.float32))))
            if ops:
                spec.residual_add[int(l_idx)] = ops

        for l_idx, pos_map in mlp_rep.items():
            ops = []
            if "subj" in pos_map:
                arr = np.asarray(pos_map["subj"][start:end], dtype=np.float32) * inv_rm
                ops.append((spos, torch.from_numpy(arr)))
            if "last" in pos_map:
                arr = np.asarray(pos_map["last"][start:end], dtype=np.float32) * inv_rm
                ops.append((lpos, torch.from_numpy(arr)))
            if ops:
                spec.mlp_replace[int(l_idx)] = ops

        with systems.PatchHookManager(model, spec):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=(not fast_first_tok_only),
            )

        first_logits = out.logits[:, -1, :]
        first_toks = first_logits.argmax(dim=-1).tolist()

        if fast_first_tok_only:
            for bi, tok_id in enumerate(first_toks):
                p_idx = start + bi
                if tok_id in alias_first_sets[p_idx]:
                    outcomes[p_idx] = 1
            continue

        cand_bi = [bi for bi, tok_id in enumerate(first_toks) if tok_id in alias_first_sets[start + bi]]
        if not cand_bi:
            continue

        gen_ids = _greedy_decode_from_prefill(
            model, first_logits, out.past_key_values, attention_mask, max_new_tokens=max_new_tokens
        )
        for bi in cand_bi:
            p_idx = start + bi
            g_txt = tok.decode(gen_ids[bi].tolist(), skip_special_tokens=True)
            if check_answer_match(g_txt, target_aliases_list[p_idx], relations[p_idx]):
                outcomes[p_idx] = 1

    return float(sum(outcomes) / n), outcomes


def _prep_fact_eval_args(
    facts: List[dict],
    eval_B: List[dict],
    swap_pairs: Sequence[Tuple[int, int]],
) -> Tuple[List[str], List[int], List[int], List[List[str]], List[str]]:
    prompts = [facts[b]["direct_prompt"] for _, b in swap_pairs]
    spos = [eval_B[b]["subj_tok_idx_direct"] for _, b in swap_pairs]
    slens = [eval_B[b]["seq_len_direct"] for _, b in swap_pairs]
    aliases = [facts[d]["aliases"] for d, _ in swap_pairs]
    rels = [facts[d]["relation"] for d, _ in swap_pairs]
    return prompts, spos, slens, aliases, rels


def fit_best_monge_on_val(
    model_B: Any,
    tok_B: Any,
    facts: List[dict],
    eval_B: List[dict],
    val_pairs: Sequence[Tuple[int, int]],
    ZA_tr: np.ndarray,
    ZB_tr: np.ndarray,
    ZA_val: np.ndarray,
    layer_B: int,
    device: torch.device,
    dtype: torch.dtype,
    res_mult: float,
    lam_grid: Sequence[float] = (0.05, 0.15, 0.30, 0.50),
) -> Tuple[map_lib.MongeProcrustesMap, float, float]:
    """Tune Monge-Procrustes shrinkage strictly on val_pairs (zero test leakage)."""
    p_args = _prep_fact_eval_args(facts, eval_B, val_pairs)
    best_map = None
    best_lam = lam_grid[0]
    best_acc = -1.0
    for lam in lam_grid:
        mp = map_lib.MongeProcrustesMap(lam_A=lam, lam_B=lam).fit(ZA_tr, ZB_tr)
        z_val = mp.transform(ZA_val)
        acc, _ = eval_multi_regime_swaps(
            model_B, tok_B, *p_args,
            residual_replace={layer_B: {"subj": z_val}},
            device=device, dtype=dtype, residual_multiplier=res_mult,
            fast_first_tok_only=True,
        )
        if acc > best_acc:
            best_acc = acc
            best_lam = lam
            best_map = mp
    return best_map, float(best_lam), float(best_acc)


# =============================================================================
# Helper: Load Cached Data Cleanly
# =============================================================================


def load_all_caches(model_keys: Sequence[str]) -> dict:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(DATASET_2HOP_PATH, "r", encoding="utf-8") as f:
        dataset_2hop = json.load(f)
    with open(BASELINE_RESULTS_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    evals, evals_2hop, states, states_2hop, model_infos = {}, {}, {}, {}, {}
    for mk in model_keys:
        with open(CACHE_DIR / f"{mk}_eval.json", "r", encoding="utf-8") as f:
            ev = json.load(f)
        evals[mk] = ev
        model_infos[mk] = ev["model_info"]
        with open(CACHE_DIR / f"{mk}_2hop_eval.json", "r", encoding="utf-8") as f:
            evals_2hop[mk] = json.load(f)
        states[mk] = torch.load(CACHE_DIR / f"{mk}_states.pt", map_location="cpu", weights_only=False)
        states_2hop[mk] = torch.load(CACHE_DIR / f"{mk}_2hop.pt", map_location="cpu", weights_only=False)

    return {
        "dataset": dataset,
        "dataset_2hop": dataset_2hop,
        "baseline": baseline,
        "evals": evals,
        "evals_2hop": evals_2hop,
        "states": states,
        "states_2hop": states_2hop,
        "model_infos": model_infos,
    }


def _select_donor_layer_for_target_lB(
    model_B: Any,
    tok_B: Any,
    facts: List[dict],
    eval_B: List[dict],
    shared_train: Sequence[int],
    val_pairs: Sequence[Tuple[int, int]],
    ZA_all: torch.Tensor,
    ZB_all: torch.Tensor,
    base_la: int,
    target_lb: int,
    L_A: int,
    device: torch.device,
    dtype: torch.dtype,
    res_mult: float,
) -> Tuple[int, float]:
    """Select best donor layer l_A in [base_la-2 .. base_la+2] for a fixed recipient layer target_lb on val."""
    cands = sorted(set(int(np.clip(base_la + d, 0, L_A - 1)) for d in (-2, -1, 0, 1, 2)))
    ZB_tr = ZB_all[shared_train, target_lb].float().numpy()
    val_donors = [d for d, _ in val_pairs]
    p_args = _prep_fact_eval_args(facts, eval_B, val_pairs)
    best_la = base_la
    best_acc = -1.0
    for la in cands:
        mp = systems.fit_procrustes_fast_qr(ZA_all[shared_train, la].float().numpy(), ZB_tr)
        z_val = mp.transform(ZA_all[val_donors, la].float().numpy())
        acc, _ = eval_multi_regime_swaps(
            model_B, tok_B, *p_args,
            residual_replace={target_lb: {"subj": z_val}},
            device=device, dtype=dtype, residual_multiplier=res_mult,
            fast_first_tok_only=True,
        )
        if acc > best_acc:
            best_acc = acc
            best_la = la
    return best_la, float(best_acc)


# =============================================================================
# PILLAR 1: Multi-Donor Platonic Barycenter + PILLAR 2: GPA & Holonomy
# =============================================================================


def run_holonomy_and_gpa_geometry(cache: dict, model_keys: Sequence[str]) -> dict:
    """Pure linear-algebra Pillar 1B: Closed-loop holonomy across all triples + GPA space fit."""
    facts = cache["dataset"]["facts"]
    evals = cache["evals"]
    states = cache["states"]
    baseline = cache["baseline"]

    splits_idx = {sp: [f["idx"] for f in facts if f["split"] == sp] for sp in ("train", "val", "test")}
    # Common layer per model = val-optimal self-swap layer
    canon_layer = {mk: int(baseline["self_swap"][mk]["best_layer"]) for mk in model_keys}

    common_train = [
        i for i in splits_idx["train"]
        if all(evals[mk]["facts"][i]["direct_correct"] for mk in model_keys)
    ]
    common_val = [
        i for i in splits_idx["val"]
        if all(evals[mk]["facts"][i]["direct_correct"] for mk in model_keys)
    ]
    common_test = [
        i for i in splits_idx["test"]
        if all(evals[mk]["facts"][i]["direct_correct"] for mk in model_keys)
    ]

    X_tr = {mk: states[mk]["direct_subj"][common_train, canon_layer[mk]].float().numpy() for mk in model_keys}
    X_te = {mk: states[mk]["direct_subj"][common_test, canon_layer[mk]].float().numpy() for mk in model_keys}

    # Fit GPA at multiple dimensions k
    gpa_spaces: Dict[int, systems.GPASpace] = {}
    gpa_summary: Dict[str, Any] = {}
    max_k = min(min(X_tr[mk].shape[1] for mk in model_keys), len(common_train))
    k_list = [k for k in (64, 128, 256, 512) if k <= max_k]
    if max_k not in k_list:
        k_list.append(max_k)

    for k in k_list:
        g = systems.fit_gpa_platonic_space(X_tr, k=k, max_iter=150, tol=1e-6)
        gpa_spaces[k] = g
        # Pairwise test cosine in Platonic space
        pair_cos = []
        for m1, m2 in itertools.combinations(model_keys, 2):
            Z1 = g.encode(m1, X_te[m1])
            Z2 = g.encode(m2, X_te[m2])
            c = np.sum(Z1 * Z2, axis=1) / np.maximum(np.linalg.norm(Z1, axis=1) * np.linalg.norm(Z2, axis=1), 1e-12)
            pair_cos.append(float(np.mean(c)))
        gpa_summary[str(k)] = {
            "k": k,
            "iters": len(g.objective_history),
            "converged": g.converged,
            "final_objective": float(g.objective_history[-1]),
            "mean_test_platonic_cos": float(np.mean(pair_cos)),
        }

    # Closed-loop holonomy across all C(5, 3) = 10 undirected triples (each oriented A->B->C->A)
    pairwise_W = {}
    for m1 in model_keys:
        for m2 in model_keys:
            if m1 == m2:
                continue
            pairwise_W[(m1, m2)] = systems.fit_procrustes(X_tr[m1], X_tr[m2]).W

    holonomy_triples = {}
    g_full = gpa_spaces[max_k]
    for idx_t, (mA, mB, mC) in enumerate(itertools.combinations(model_keys, 3)):
        key = f"{mA}->{mB}->{mC}->{mA}"
        hol_pair = systems.compute_closed_loop_holonomy(
            pairwise_W[(mA, mB)], pairwise_W[(mB, mC)], pairwise_W[(mC, mA)],
            X_tr[mA], X_te[mA], ks=(16, 32, 64, 128, 256),
        )
        # GPA maps on exact same triple
        W_g_ab = g_full.Q[mA] @ g_full.Q[mB].T
        W_g_bc = g_full.Q[mB] @ g_full.Q[mC].T
        W_g_ca = g_full.Q[mC] @ g_full.Q[mA].T
        hol_gpa = systems.compute_closed_loop_holonomy(
            W_g_ab, W_g_bc, W_g_ca, X_tr[mA], X_te[mA], ks=(16, 32, 64, 128, 256),
        )
        # Exact zero holonomy check on span(Q_A)
        L_gpa = W_g_ab @ W_g_bc @ W_g_ca
        hol_gpa_on_Q = float(np.linalg.norm(L_gpa @ g_full.Q[mA] - g_full.Q[mA], "fro"))

        # Random & shuffled controls
        Wr_ab = systems.fit_random_orthogonal_control(X_tr[mA], X_tr[mB], seed=SEED + 100 * idx_t + 1).W
        Wr_bc = systems.fit_random_orthogonal_control(X_tr[mB], X_tr[mC], seed=SEED + 100 * idx_t + 2).W
        Wr_ca = systems.fit_random_orthogonal_control(X_tr[mC], X_tr[mA], seed=SEED + 100 * idx_t + 3).W
        hol_rand = systems.compute_closed_loop_holonomy(
            Wr_ab, Wr_bc, Wr_ca, X_tr[mA], X_te[mA], ks=(16, 32, 64, 128, 256),
        )
        Ws_ab = systems.fit_shuffled_procrustes_control(X_tr[mA], X_tr[mB], seed=SEED + 200 * idx_t + 1).W
        Ws_bc = systems.fit_shuffled_procrustes_control(X_tr[mB], X_tr[mC], seed=SEED + 200 * idx_t + 2).W
        Ws_ca = systems.fit_shuffled_procrustes_control(X_tr[mC], X_tr[mA], seed=SEED + 200 * idx_t + 3).W
        hol_shuf = systems.compute_closed_loop_holonomy(
            Ws_ab, Ws_bc, Ws_ca, X_tr[mA], X_te[mA], ks=(16, 32, 64, 128, 256),
        )

        holonomy_triples[key] = {
            "pairwise_procrustes": hol_pair,
            "gpa_full": hol_gpa,
            "gpa_exact_holonomy_on_Q": hol_gpa_on_Q,
            "random_control": hol_rand,
            "shuffled_control": hol_shuf,
        }

    # Aggregate median holonomy metrics across all 10 triples
    agg_hol = {}
    for k_str in ("16", "32", "64", "128", "256"):
        vals_p = [v["pairwise_procrustes"]["subspace"][k_str]["holonomy_full_norm"] for v in holonomy_triples.values() if k_str in v["pairwise_procrustes"]["subspace"]]
        vals_r = [v["random_control"]["subspace"][k_str]["holonomy_full_norm"] for v in holonomy_triples.values() if k_str in v["random_control"]["subspace"]]
        vals_s = [v["shuffled_control"]["subspace"][k_str]["holonomy_full_norm"] for v in holonomy_triples.values() if k_str in v["shuffled_control"]["subspace"]]
        vals_ang = [v["pairwise_procrustes"]["subspace"][k_str]["mean_angle_deg"] for v in holonomy_triples.values() if k_str in v["pairwise_procrustes"]["subspace"]]
        if vals_p:
            agg_hol[k_str] = {
                "median_pairwise_hol_norm": float(np.median(vals_p)),
                "median_random_hol_norm": float(np.median(vals_r)),
                "median_shuffled_hol_norm": float(np.median(vals_s)),
                "median_pairwise_angle_deg": float(np.median(vals_ang)),
            }

    agg_hol["median_emp_test_cos_pairwise"] = float(
        np.median([v["pairwise_procrustes"]["empirical_test"]["mean_cos"] for v in holonomy_triples.values()])
    )
    agg_hol["median_emp_test_cos_random"] = float(
        np.median([v["random_control"]["empirical_test"]["mean_cos"] for v in holonomy_triples.values()])
    )
    agg_hol["max_gpa_exact_holonomy_on_Q"] = float(
        max(v["gpa_exact_holonomy_on_Q"] for v in holonomy_triples.values())
    )

    return {
        "canonical_layers": canon_layer,
        "n_common_train": len(common_train),
        "n_common_val": len(common_val),
        "n_common_test": len(common_test),
        "common_train_indices": common_train,
        "common_val_indices": common_val,
        "common_test_indices": common_test,
        "gpa_fit_summary": gpa_summary,
        "holonomy_aggregate": agg_hol,
        "holonomy_triples": holonomy_triples,
        "_gpa_spaces": gpa_spaces,  # stripped before JSON dump
    }


def eval_recipient_barycenter_and_gpa(
    mk_B: str,
    model_B: Any,
    tok_B: Any,
    device: torch.device,
    dtype: torch.dtype,
    cache: dict,
    model_keys: Sequence[str],
    gpa_geom: dict,
) -> Tuple[dict, dict, dict]:
    """Run Multi-Donor Barycenter + GPA Full Transfer + Low-Overlap GPA Sync for recipient mk_B."""
    facts = cache["dataset"]["facts"]
    evals = cache["evals"]
    states = cache["states"]
    baseline = cache["baseline"]
    eval_B = evals[mk_B]["facts"]
    res_mult = float(cache["model_infos"][mk_B].get("residual_multiplier", 1.0))

    canon_layer = gpa_geom["canonical_layers"]
    l_B = canon_layer[mk_B]
    common_train = gpa_geom["common_train_indices"]
    common_val = gpa_geom["common_val_indices"]
    common_test = gpa_geom["common_test_indices"]

    val_pairs = build_counterfactual_pairs(facts, common_val)
    test_pairs = build_counterfactual_pairs(facts, common_test)
    val_donors = [d for d, _ in val_pairs]
    test_donors = [d for d, _ in test_pairs]
    val_args = _prep_fact_eval_args(facts, eval_B, val_pairs)
    test_args = _prep_fact_eval_args(facts, eval_B, test_pairs)

    ZB_tr = states[mk_B]["direct_subj"][common_train, l_B].float().numpy()
    ZB_val_all = states[mk_B]["direct_subj"][common_val, l_B].float().numpy()
    ZB_te_self = states[mk_B]["direct_subj"][test_donors, l_B].float().numpy()

    self_acc, self_out = eval_multi_regime_swaps(
        model_B, tok_B, *test_args,
        residual_replace={l_B: {"subj": ZB_te_self}},
        device=device, dtype=dtype, residual_multiplier=res_mult,
    )

    all_donors = [m for m in model_keys if m != mk_B]
    cross_fam_donors = [m for m in all_donors if get_family(m) != get_family(mk_B)]

    # Fit each donor -> B at common l_B (with donor l_A tuned on val around baseline best_layer_A)
    donor_maps_proc: Dict[str, systems.ProcrustesMap] = {}
    donor_maps_monge: Dict[str, Any] = {}
    donor_la: Dict[str, int] = {}
    single_results: Dict[str, Any] = {}
    mapped_val_common: Dict[str, np.ndarray] = {}
    mapped_val_pairs: Dict[str, np.ndarray] = {}
    mapped_test_pairs: Dict[str, np.ndarray] = {}
    mapped_val_monge: Dict[str, np.ndarray] = {}
    mapped_test_monge: Dict[str, np.ndarray] = {}

    for mA in all_donors:
        L_A = int(cache["model_infos"][mA]["num_layers"])
        base_la = int(baseline["pairs"][f"{mA}->{mk_B}"]["best_layer_A"])
        la, _ = _select_donor_layer_for_target_lB(
            model_B, tok_B, facts, eval_B, common_train, val_pairs,
            states[mA]["direct_subj"], states[mk_B]["direct_subj"],
            base_la, l_B, L_A, device, dtype, res_mult,
        )
        donor_la[mA] = la
        ZA_tr = states[mA]["direct_subj"][common_train, la].float().numpy()
        ZA_val_c = states[mA]["direct_subj"][common_val, la].float().numpy()
        ZA_val_p = states[mA]["direct_subj"][val_donors, la].float().numpy()
        ZA_te_p = states[mA]["direct_subj"][test_donors, la].float().numpy()

        mp = systems.fit_procrustes(ZA_tr, ZB_tr)
        donor_maps_proc[mA] = mp
        mapped_val_common[mA] = mp.transform(ZA_val_c)
        mapped_val_pairs[mA] = mp.transform(ZA_val_p)
        mapped_test_pairs[mA] = mp.transform(ZA_te_p)

        val_acc_m, _ = eval_multi_regime_swaps(
            model_B, tok_B, *val_args,
            residual_replace={l_B: {"subj": mapped_val_pairs[mA]}},
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        te_acc_m, te_out_m = eval_multi_regime_swaps(
            model_B, tok_B, *test_args,
            residual_replace={l_B: {"subj": mapped_test_pairs[mA]}},
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )

        # Monge-Procrustes single donor
        mp_monge, best_lam, monge_val_acc = fit_best_monge_on_val(
            model_B, tok_B, facts, eval_B, val_pairs,
            ZA_tr, ZB_tr, ZA_val_p, l_B, device, dtype, res_mult,
        )
        donor_maps_monge[mA] = (mp_monge, best_lam)
        mapped_val_monge[mA] = mp_monge.transform(ZA_val_c)
        mapped_test_monge[mA] = mp_monge.transform(ZA_te_p)
        m_te_acc, m_te_out = eval_multi_regime_swaps(
            model_B, tok_B, *test_args,
            residual_replace={l_B: {"subj": mapped_test_monge[mA]}},
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )

        single_results[mA] = {
            "layer_A": la,
            "layer_B": l_B,
            "cross_family": get_family(mA) != get_family(mk_B),
            "procrustes_val_acc": val_acc_m,
            "procrustes_test_acc": te_acc_m,
            "procrustes_test_outcomes": te_out_m,
            "monge_lam": best_lam,
            "monge_val_acc": monge_val_acc,
            "monge_test_acc": m_te_acc,
            "monge_test_outcomes": m_te_out,
        }

    # Multi-Donor Barycenter subsets:
    # - M=2 top-2 cross-family (by val_acc)
    # - M=3 pure cross-family (all cross-family donors, up to 3)
    # - M=all (all 4 other models)
    cross_sorted = sorted(cross_fam_donors, key=lambda m: single_results[m]["procrustes_val_acc"], reverse=True)
    all_sorted = sorted(all_donors, key=lambda m: single_results[m]["procrustes_val_acc"], reverse=True)

    bary_groups: Dict[str, List[str]] = {}
    if len(cross_sorted) >= 2:
        bary_groups["M2_cross_family_top2"] = cross_sorted[:2]
    if len(cross_sorted) >= 3:
        bary_groups["M3_cross_family"] = cross_sorted[:3]
    if len(all_sorted) >= 3:
        bary_groups[f"M{len(all_sorted)}_all_donors"] = all_sorted

    bary_results: Dict[str, Any] = {
        "recipient": mk_B,
        "layer_B": l_B,
        "n_test_pairs": len(test_pairs),
        "test_pairs": test_pairs,
        "self_swap_test_acc": self_acc,
        "self_swap_outcomes": self_out,
        "single_donors": single_results,
        "groups": {},
    }

    for gname, dlist in bary_groups.items():
        val_accs_g = [single_results[m]["procrustes_val_acc"] for m in dlist]
        mu_list_g = [donor_maps_proc[m].mu_b for m in dlist]
        mv_common_g = [mapped_val_common[m] for m in dlist]
        mv_pairs_g = [mapped_val_pairs[m] for m in dlist]
        mt_pairs_g = [mapped_test_pairs[m] for m in dlist]

        group_out: Dict[str, Any] = {
            "donors": dlist,
            "M": len(dlist),
            "best_single_test_acc": max(single_results[m]["procrustes_test_acc"] for m in dlist),
            "mean_single_test_acc": float(np.mean([single_results[m]["procrustes_test_acc"] for m in dlist])),
            "variants": {},
        }

        # Evaluate uniform vs val_acc vs inv_cov AND norm_mode none vs rms_norm vs global_spread
        combos = [
            ("uniform_no_norm", "uniform", "none"),
            ("uniform_rms_norm", "uniform", "rms_norm"),
            ("uniform_global_spread", "uniform", "global_spread"),
            ("val_weighted_rms_norm", "val_acc", "rms_norm"),
            ("inv_cov_gls_rms_norm", "inv_cov", "rms_norm"),
        ]
        for vname, w_meth, n_mode in combos:
            cfg = systems.fit_barycenter_config(
                dlist, mv_common_g, mu_list_g, ZB_val_all,
                val_accs=val_accs_g, weight_method=w_meth, norm_mode=n_mode, gain=1.0,
            )
            z_te_bary, diag = systems.aggregate_multi_donor_keys(mt_pairs_g, mu_list_g, cfg)
            acc_v, out_v = eval_multi_regime_swaps(
                model_B, tok_B, *test_args,
                residual_replace={l_B: {"subj": z_te_bary}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            group_out["variants"][vname] = {
                "weights": cfg.weights.tolist(),
                "mean_offdiag_rho": cfg.mean_offdiag_rho,
                "error_corr": cfg.error_corr.tolist() if cfg.error_corr is not None else None,
                "theoretical_snr_gain": cfg.theoretical_snr_gain,
                "empirical_snr_gain": cfg.empirical_snr_gain,
                "norm_attenuation_ratio": diag["mean_norm_attenuation_ratio"],
                "mean_pairwise_donor_cos": diag["mean_pairwise_donor_cos"],
                "test_acc": acc_v,
                "test_outcomes": out_v,
            }

        # Also tune gain alpha on val for inv_cov_gls_rms_norm!
        cfg_gls = systems.fit_barycenter_config(
            dlist, mv_common_g, mu_list_g, ZB_val_all,
            val_accs=val_accs_g, weight_method="inv_cov", norm_mode="rms_norm", gain=1.0,
        )
        best_g_alpha, best_g_val = 1.0, -1.0
        for alpha in (0.95, 1.0, 1.1, 1.2, 1.35):
            cfg_a = systems.BarycenterConfig(
                donor_names=cfg_gls.donor_names, weights=cfg_gls.weights,
                weight_method=cfg_gls.weight_method, norm_mode="rms_norm",
                gain=alpha, global_scale=cfg_gls.global_scale,
            )
            z_va, _ = systems.aggregate_multi_donor_keys(mv_pairs_g, mu_list_g, cfg_a)
            ava, _ = eval_multi_regime_swaps(
                model_B, tok_B, *val_args,
                residual_replace={l_B: {"subj": z_va}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
                fast_first_tok_only=True,
            )
            if ava > best_g_val:
                best_g_val = ava
                best_g_alpha = alpha
        cfg_gls_gain = systems.BarycenterConfig(
            donor_names=cfg_gls.donor_names, weights=cfg_gls.weights,
            weight_method=cfg_gls.weight_method, norm_mode="rms_norm",
            gain=best_g_alpha, global_scale=cfg_gls.global_scale,
        )
        z_te_gg, _ = systems.aggregate_multi_donor_keys(mt_pairs_g, mu_list_g, cfg_gls_gain)
        acc_gg, out_gg = eval_multi_regime_swaps(
            model_B, tok_B, *test_args,
            residual_replace={l_B: {"subj": z_te_gg}},
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        group_out["variants"]["inv_cov_gls_rms_val_gain"] = {
            "val_selected_gain": best_g_alpha,
            "weights": cfg_gls.weights.tolist(),
            "test_acc": acc_gg,
            "test_outcomes": out_gg,
        }

        # Multi-Donor Monge-Procrustes Barycenter (combining @map-improver + Multi-Donor!)
        mv_monge_g = [mapped_val_monge[m] for m in dlist]
        mt_monge_g = [mapped_test_monge[m] for m in dlist]
        mu_monge_g = [donor_maps_monge[m][0].mu_B for m in dlist]
        cfg_m_bary = systems.fit_barycenter_config(
            dlist, mv_monge_g, mu_monge_g, ZB_val_all,
            val_accs=[single_results[m]["monge_val_acc"] for m in dlist],
            weight_method="inv_cov", norm_mode="rms_norm", gain=1.0,
        )
        z_te_mb, diag_mb = systems.aggregate_multi_donor_keys(mt_monge_g, mu_monge_g, cfg_m_bary)
        acc_mb, out_mb = eval_multi_regime_swaps(
            model_B, tok_B, *test_args,
            residual_replace={l_B: {"subj": z_te_mb}},
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        group_out["variants"]["monge_inv_cov_rms_norm"] = {
            "weights": cfg_m_bary.weights.tolist(),
            "mean_offdiag_rho": cfg_m_bary.mean_offdiag_rho,
            "empirical_snr_gain": cfg_m_bary.empirical_snr_gain,
            "norm_attenuation_ratio": diag_mb["mean_norm_attenuation_ratio"],
            "test_acc": acc_mb,
            "test_outcomes": out_mb,
        }

        # Fig. 6 Native GPA Platonic-Space Barycenter (O(L) projections!)
        max_k = max(gpa_geom["_gpa_spaces"].keys())
        g_full = gpa_geom["_gpa_spaces"][max_k]
        donor_raw_te = {m: states[m]["direct_subj"][test_donors, canon_layer[m]].float().numpy() for m in dlist}
        # Scale boost to restore full recipient spread from k-dim GPA projection
        proj_B_tr = (states[mk_B]["direct_subj"][common_train, l_B].float().numpy() - g_full.mu[mk_B]) @ g_full.Q[mk_B]
        full_B_norm = float(np.linalg.norm(states[mk_B]["direct_subj"][common_train, l_B].float().numpy() - g_full.mu[mk_B], "fro"))
        proj_B_norm = float(np.linalg.norm(proj_B_tr, "fro"))
        s_boost = full_B_norm / max(proj_B_norm, 1e-12)
        z_gpa_bary, diag_gb = g_full.aggregate_donors_in_platonic_space(
            donor_raw_te, mk_B, weights=None, norm_mode="rms_norm", gain=1.0, scale_boost=s_boost
        )
        acc_gb, out_gb = eval_multi_regime_swaps(
            model_B, tok_B, *test_args,
            residual_replace={l_B: {"subj": z_gpa_bary}},
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        group_out["variants"]["fig6_native_gpa_barycenter"] = {
            "k": max_k,
            "scale_boost": s_boost,
            "platonic_norm_attenuation": diag_gb["platonic_norm_attenuation"],
            "test_acc": acc_gb,
            "test_outcomes": out_gb,
        }

        bary_results["groups"][gname] = group_out
        print(
            f"  [bary {mk_B} {gname}] single_mean={group_out['mean_single_test_acc']:.3f} "
            f"single_best={group_out['best_single_test_acc']:.3f} | "
            f"no_norm={group_out['variants']['uniform_no_norm']['test_acc']:.3f} "
            f"unif_rms={group_out['variants']['uniform_rms_norm']['test_acc']:.3f} "
            f"gls_rms={group_out['variants']['inv_cov_gls_rms_norm']['test_acc']:.3f} "
            f"monge_bary={acc_mb:.3f} gpa_bary={acc_gb:.3f} (self={self_acc:.3f}, rho={cfg_gls.mean_offdiag_rho:.2f})"
        )

    # GPA Pairwise Transfer Curve across k for all donors -> mk_B
    gpa_transfer_res: Dict[str, Any] = {}
    for mA in all_donors:
        pair_key = f"{mA}->{mk_B}"
        l_A_c = canon_layer[mA]
        ZA_tr_c = states[mA]["direct_subj"][common_train, l_A_c].float().numpy()
        ZA_te_c = states[mA]["direct_subj"][test_donors, l_A_c].float().numpy()
        mp_dir = systems.fit_procrustes(ZA_tr_c, ZB_tr)
        dir_acc, dir_out = eval_multi_regime_swaps(
            model_B, tok_B, *test_args,
            residual_replace={l_B: {"subj": mp_dir.transform(ZA_te_c)}},
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        k_res = {}
        for k, gspace in sorted(gpa_geom["_gpa_spaces"].items()):
            mp_g = gspace.get_pairwise_map(mA, mk_B, ZA_tr_c, ZB_tr, restore_full_norm=True)
            acc_g, out_g = eval_multi_regime_swaps(
                model_B, tok_B, *test_args,
                residual_replace={l_B: {"subj": mp_g.transform(ZA_te_c)}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            k_res[str(k)] = {
                "test_acc": acc_g,
                "rel_to_direct": acc_g / max(dir_acc, 1e-9),
                "test_outcomes": out_g,
            }
        gpa_transfer_res[pair_key] = {
            "layers": [l_A_c, l_B],
            "direct_procrustes_acc": dir_acc,
            "direct_procrustes_outcomes": dir_out,
            "gpa_by_k": k_res,
        }

    # Low-Overlap GPA Synchronization Experiment:
    # For cross-family donor -> mk_B, mask out direct (mA, mk_B) overlap down to N_ov in {25, 50, 100}
    # while bridge models retain full training overlap with BOTH mA and mk_B!
    low_overlap_res: Dict[str, Any] = {}
    rng_lo = np.random.default_rng(SEED + 777 + hash(mk_B) % 1000)
    N_c = len(common_train)
    X_tr_all = {m: states[m]["direct_subj"][common_train, canon_layer[m]].float().numpy() for m in model_keys}

    # Pick top-2 cross-family donors for clean, fast low-overlap sweep
    for mA in cross_sorted[:2]:
        pair_key = f"{mA}->{mk_B}"
        l_A_c = canon_layer[mA]
        ZA_te_c = states[mA]["direct_subj"][test_donors, l_A_c].float().numpy()
        perm = rng_lo.permutation(N_c)
        sweep = {}
        for n_ov in (25, 50, 100):
            if n_ov >= N_c - 40:
                continue
            rem = N_c - n_ov
            half = rem // 2
            # mA sees [0 .. n_ov+half), mk_B sees [0..n_ov) U [n_ov+half..N_c) -> exact direct overlap = n_ov!
            # Bridge models see ALL N_c facts!
            masks = {m: np.ones(N_c, dtype=bool) for m in model_keys}
            mask_A = np.zeros(N_c, dtype=bool)
            mask_B = np.zeros(N_c, dtype=bool)
            mask_A[perm[: n_ov + half]] = True
            mask_B[perm[:n_ov]] = True
            mask_B[perm[n_ov + half :]] = True
            masks[mA] = mask_A
            masks[mk_B] = mask_B
            ov_idx = mask_A & mask_B
            assert int(ov_idx.sum()) == n_ov

            # Direct Procrustes on tiny n_ov overlap
            mp_low = systems.fit_procrustes(X_tr_all[mA][ov_idx], X_tr_all[mk_B][ov_idx])
            acc_low, out_low = eval_multi_regime_swaps(
                model_B, tok_B, *test_args,
                residual_replace={l_B: {"subj": mp_low.transform(ZA_te_c)}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )

            # Partial-Overlap GPA Sync across all 5 models (k=256)
            k_sync = min(256, N_c)
            gpa_p = systems.fit_gpa_partial_overlap(X_tr_all, masks, k=k_sync, max_iter=50, tol=1e-5)
            mp_sync = gpa_p.get_pairwise_map(mA, mk_B, X_tr_all[mA][mask_A], X_tr_all[mk_B][mask_B], restore_full_norm=True)
            acc_sync, out_sync = eval_multi_regime_swaps(
                model_B, tok_B, *test_args,
                residual_replace={l_B: {"subj": mp_sync.transform(ZA_te_c)}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )

            mc = systems.exact_mcnemar(out_low, out_sync)
            sweep[str(n_ov)] = {
                "n_direct_overlap": n_ov,
                "k_sync": k_sync,
                "direct_low_acc": acc_low,
                "direct_low_outcomes": out_low,
                "gpa_sync_acc": acc_sync,
                "gpa_sync_outcomes": out_sync,
                "gain": acc_sync - acc_low,
                "mcnemar_p": mc["p_value"],
            }
            print(
                f"  [low-overlap {pair_key} N={n_ov}] direct={acc_low:.3f} -> GPA_sync={acc_sync:.3f} "
                f"(+{acc_sync - acc_low:.3f}, p={mc['p_value']:.4f})"
            )
        low_overlap_res[pair_key] = sweep

    return bary_results, gpa_transfer_res, low_overlap_res


# =============================================================================
# PILLAR 3: Breaking the Single-Layer Self-Swap Ceiling
# =============================================================================


def eval_recipient_ceiling_breakers(
    mk_B: str,
    model_B: Any,
    tok_B: Any,
    device: torch.device,
    dtype: torch.dtype,
    cache: dict,
    model_keys: Sequence[str],
) -> dict:
    """Evaluate Multi-Layer Window, MLP-Only, Dual-Token, Difference-Vector, & Gain Steering
    for BOTH Self-Swap Ceiling AND Cross-Model Transfer into recipient mk_B."""
    facts = cache["dataset"]["facts"]
    evals = cache["evals"]
    states = cache["states"]
    baseline = cache["baseline"]
    eval_B = evals[mk_B]["facts"]
    L_B = int(cache["model_infos"][mk_B]["num_layers"])
    res_mult = float(cache["model_infos"][mk_B].get("residual_multiplier", 1.0))

    ZB_subj = states[mk_B]["direct_subj"]
    ZB_last = states[mk_B]["direct_last"]
    ZB_mlp = states[mk_B]["direct_mlp_subj"]

    # ------------------------------------------------------------------
    # Part A: Self-Swap Ceiling Breakers on mk_B
    # ------------------------------------------------------------------
    splits_idx = {sp: [f["idx"] for f in facts if f["split"] == sp] for sp in ("train", "val", "test")}
    b_train_rec = [i for i in splits_idx["train"] if eval_B[i]["direct_correct"]]
    b_val_rec = [i for i in splits_idx["val"] if eval_B[i]["direct_correct"]]
    b_test_rec = [i for i in splits_idx["test"] if eval_B[i]["direct_correct"]]
    self_val_pairs = build_counterfactual_pairs(facts, b_val_rec)
    self_test_pairs = build_counterfactual_pairs(facts, b_test_rec)
    sv_args = _prep_fact_eval_args(facts, eval_B, self_val_pairs)
    st_args = _prep_fact_eval_args(facts, eval_B, self_test_pairs)
    sv_d = [d for d, _ in self_val_pairs]
    sv_b = [b for _, b in self_val_pairs]
    st_d = [d for d, _ in self_test_pairs]
    st_b = [b for _, b in self_test_pairs]

    l_B_self = int(baseline["self_swap"][mk_B]["best_layer"])
    mu_B_self = ZB_subj[b_train_rec, l_B_self].float().numpy().mean(axis=0)

    self_regimes: Dict[str, Any] = {"best_single_layer": l_B_self, "n_test_pairs": len(self_test_pairs), "regimes": {}}

    # 1. Single-layer baseline
    acc_s1, out_s1 = eval_multi_regime_swaps(
        model_B, tok_B, *st_args,
        residual_replace={l_B_self: {"subj": ZB_subj[st_d, l_B_self].float().numpy()}},
        device=device, dtype=dtype, residual_multiplier=res_mult,
    )
    self_regimes["regimes"]["single_layer_residual"] = {"test_acc": acc_s1, "outcomes": out_s1}

    # 2. Gain-tuned single-layer self-swap (Theorem 1!)
    alpha_grid = (0.85, 1.0, 1.15, 1.3, 1.5, 1.75)
    best_a_rep, best_a_rep_val = 1.0, -1.0
    for alpha in alpha_grid:
        z_v = mu_B_self[None, :] + alpha * (ZB_subj[sv_d, l_B_self].float().numpy() - mu_B_self[None, :])
        av, _ = eval_multi_regime_swaps(
            model_B, tok_B, *sv_args, residual_replace={l_B_self: {"subj": z_v}},
            device=device, dtype=dtype, residual_multiplier=res_mult, fast_first_tok_only=True,
        )
        if av > best_a_rep_val:
            best_a_rep_val, best_a_rep = av, alpha
    z_te_gr = mu_B_self[None, :] + best_a_rep * (ZB_subj[st_d, l_B_self].float().numpy() - mu_B_self[None, :])
    acc_gr, out_gr = eval_multi_regime_swaps(
        model_B, tok_B, *st_args, residual_replace={l_B_self: {"subj": z_te_gr}},
        device=device, dtype=dtype, residual_multiplier=res_mult,
    )
    self_regimes["regimes"]["single_layer_gain_tuned"] = {
        "val_alpha": best_a_rep, "test_acc": acc_gr, "outcomes": out_gr,
    }

    # 3. Difference-vector steering self-swap: z_B(s) + alpha * (z_B(s') - z_B(s))
    best_a_diff, best_a_diff_val = 1.0, -1.0
    for alpha in (0.85, 1.0, 1.15, 1.3, 1.5, 1.75):
        delta_v = alpha * (ZB_subj[sv_d, l_B_self].float().numpy() - ZB_subj[sv_b, l_B_self].float().numpy())
        av, _ = eval_multi_regime_swaps(
            model_B, tok_B, *sv_args, residual_add={l_B_self: {"subj": delta_v}},
            device=device, dtype=dtype, residual_multiplier=res_mult, fast_first_tok_only=True,
        )
        if av > best_a_diff_val:
            best_a_diff_val, best_a_diff = av, alpha
    delta_te = best_a_diff * (ZB_subj[st_d, l_B_self].float().numpy() - ZB_subj[st_b, l_B_self].float().numpy())
    acc_df, out_df = eval_multi_regime_swaps(
        model_B, tok_B, *st_args, residual_add={l_B_self: {"subj": delta_te}},
        device=device, dtype=dtype, residual_multiplier=res_mult,
    )
    self_regimes["regimes"]["single_layer_diff_steering"] = {
        "val_alpha": best_a_diff, "test_acc": acc_df, "outcomes": out_df,
    }

    # 4. Contiguous Multi-Layer Window Residual Patching (W = 2, 3, 4)
    window_cands = {
        "window_W2_residual": [[-1, 0], [0, 1]],
        "window_W3_residual": [[-1, 0, 1], [0, 1, 2], [-2, -1, 0]],
        "window_W4_residual": [[-1, 0, 1, 2], [-2, -1, 0, 1]],
    }
    best_self_win_offsets: Dict[str, List[int]] = {}
    for wname, off_list in window_cands.items():
        best_off, best_w_val = off_list[0], -1.0
        for offs in off_list:
            lbs = sorted(set(int(np.clip(l_B_self + o, 0, L_B - 1)) for o in offs))
            rep_v = {lb: {"subj": ZB_subj[sv_d, lb].float().numpy()} for lb in lbs}
            av, _ = eval_multi_regime_swaps(
                model_B, tok_B, *sv_args, residual_replace=rep_v,
                device=device, dtype=dtype, residual_multiplier=res_mult, fast_first_tok_only=True,
            )
            if av > best_w_val:
                best_w_val, best_off = av, offs
        best_self_win_offsets[wname] = best_off
        lbs = sorted(set(int(np.clip(l_B_self + o, 0, L_B - 1)) for o in best_off))
        rep_t = {lb: {"subj": ZB_subj[st_d, lb].float().numpy()} for lb in lbs}
        at, ot = eval_multi_regime_swaps(
            model_B, tok_B, *st_args, residual_replace=rep_t,
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        self_regimes["regimes"][wname] = {"layers": lbs, "offsets": best_off, "test_acc": at, "outcomes": ot}

    # 5. Additive MLP-Output-Only Patching (Delta z_MLP, W = 1, 2, 3, 4 + cumulative 0..l_B) — Hebbian-MLP test!
    for w_sz, offs in [(1, [0]), (2, [-1, 0]), (3, [-1, 0, 1]), (4, [-1, 0, 1, 2])]:
        lbs = sorted(set(int(np.clip(l_B_self + o, 0, L_B - 1)) for o in offs))
        mlp_t = {lb: {"subj": ZB_mlp[st_d, lb].float().numpy()} for lb in lbs}
        at, ot = eval_multi_regime_swaps(
            model_B, tok_B, *st_args, mlp_replace=mlp_t,
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        self_regimes["regimes"][f"mlp_only_W{w_sz}"] = {"layers": lbs, "test_acc": at, "outcomes": ot}

    lbs_cum = list(range(0, l_B_self + 1))
    mlp_cum_t = {lb: {"subj": ZB_mlp[st_d, lb].float().numpy()} for lb in lbs_cum}
    at_mc, ot_mc = eval_multi_regime_swaps(
        model_B, tok_B, *st_args, mlp_replace=mlp_cum_t,
        device=device, dtype=dtype, residual_multiplier=res_mult,
    )
    self_regimes["regimes"]["mlp_only_cumulative_0_to_lB"] = {
        "layers": lbs_cum, "test_acc": at_mc, "outcomes": ot_mc,
    }

    # 6. Dual-Token Patching (last_subj + last_prompt): same-layer AND val-optimal mid-layer
    acc_dt_same, out_dt_same = eval_multi_regime_swaps(
        model_B, tok_B, *st_args,
        residual_replace={
            l_B_self: {
                "subj": ZB_subj[st_d, l_B_self].float().numpy(),
                "last": ZB_last[st_d, l_B_self].float().numpy(),
            }
        },
        device=device, dtype=dtype, residual_multiplier=res_mult,
    )
    self_regimes["regimes"]["dual_token_same_layer"] = {
        "layer": l_B_self, "test_acc": acc_dt_same, "outcomes": out_dt_same,
    }

    # Last-prompt-only ablation at l_B_self
    acc_lp_only, out_lp_only = eval_multi_regime_swaps(
        model_B, tok_B, *st_args,
        residual_replace={l_B_self: {"last": ZB_last[st_d, l_B_self].float().numpy()}},
        device=device, dtype=dtype, residual_multiplier=res_mult,
    )
    self_regimes["regimes"]["last_prompt_only_same_layer"] = {
        "layer": l_B_self, "test_acc": acc_lp_only, "outcomes": out_lp_only,
    }

    # Combined W3 Residual Window + Dual-Token at l_B_self
    lbs_w3 = self_regimes["regimes"]["window_W3_residual"]["layers"]
    rep_comb = {lb: {"subj": ZB_subj[st_d, lb].float().numpy()} for lb in lbs_w3}
    rep_comb[l_B_self]["last"] = ZB_last[st_d, l_B_self].float().numpy()
    acc_comb, out_comb = eval_multi_regime_swaps(
        model_B, tok_B, *st_args, residual_replace=rep_comb,
        device=device, dtype=dtype, residual_multiplier=res_mult,
    )
    self_regimes["regimes"]["window_W3_plus_dual_token"] = {
        "layers": lbs_w3, "test_acc": acc_comb, "outcomes": out_comb,
    }

    print(
        f"  [ceiling SELF {mk_B}] W1={acc_s1:.3f} | gain({best_a_rep})={acc_gr:.3f} diff({best_a_diff})={acc_df:.3f} | "
        f"W2={self_regimes['regimes']['window_W2_residual']['test_acc']:.3f} "
        f"W3={self_regimes['regimes']['window_W3_residual']['test_acc']:.3f} "
        f"W4={self_regimes['regimes']['window_W4_residual']['test_acc']:.3f} | "
        f"MLP_W1={self_regimes['regimes']['mlp_only_W1']['test_acc']:.3f} "
        f"MLP_W3={self_regimes['regimes']['mlp_only_W3']['test_acc']:.3f} | "
        f"dual={acc_dt_same:.3f} W3+dual={acc_comb:.3f}"
    )

    # ------------------------------------------------------------------
    # Part B: Cross-Model Ceiling Breakers for ALL donors -> mk_B
    # ------------------------------------------------------------------
    pair_regimes: Dict[str, Any] = {}
    for mk_A in model_keys:
        if mk_A == mk_B:
            continue
        pair_key = f"{mk_A}->{mk_B}"
        bp = baseline["pairs"][pair_key]
        l_A = int(bp["best_layer_A"])
        l_B = int(bp["best_layer_B"])
        L_A = int(cache["model_infos"][mk_A]["num_layers"])

        sh_tr = bp["shared_train_indices"]
        val_pairs = [tuple(x) for x in bp["val_pairs"]]
        test_pairs = [tuple(x) for x in bp["test_pairs"]]
        v_d, v_b = [d for d, _ in val_pairs], [b for _, b in val_pairs]
        t_d, t_b = [d for d, _ in test_pairs], [b for _, b in test_pairs]
        v_args = _prep_fact_eval_args(facts, eval_B, val_pairs)
        t_args = _prep_fact_eval_args(facts, eval_B, test_pairs)

        ZA_subj = states[mk_A]["direct_subj"]
        ZA_last = states[mk_A]["direct_last"]
        ZA_mlp = states[mk_A]["direct_mlp_subj"]

        pm_base = cached_fit_procrustes(
            ("subj", mk_A, l_A, mk_B, l_B, len(sh_tr), sh_tr[0], sh_tr[-1]),
            ZA_subj[sh_tr, l_A].float().numpy(), ZB_subj[sh_tr, l_B].float().numpy()
        )
        z_base_te = pm_base.transform(ZA_subj[t_d, l_A].float().numpy())
        acc_b1, out_b1 = eval_multi_regime_swaps(
            model_B, tok_B, *t_args, residual_replace={l_B: {"subj": z_base_te}},
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )

        # Matched self-swap on exact same test_pairs for clean retention ratios
        acc_self_w1, out_self_w1 = eval_multi_regime_swaps(
            model_B, tok_B, *t_args,
            residual_replace={l_B: {"subj": ZB_subj[t_d, l_B].float().numpy()}},
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )

        p_res: Dict[str, Any] = {
            "layers": [l_A, l_B],
            "n_test_pairs": len(test_pairs),
            "test_pairs": test_pairs,
            "regimes": {
                "single_layer_baseline": {
                    "test_acc": acc_b1, "self_acc": acc_self_w1,
                    "outcomes": out_b1, "self_outcomes": out_self_w1,
                }
            },
        }

        # Gain-tuned replacement (val-tuned alpha)
        best_a, best_av = 1.0, -1.0
        ZA_v = ZA_subj[v_d, l_A].float().numpy()
        for alpha in (0.9, 1.0, 1.15, 1.3, 1.5):
            z_v = pm_base.transform(ZA_v, gain=alpha)
            av, _ = eval_multi_regime_swaps(
                model_B, tok_B, *v_args, residual_replace={l_B: {"subj": z_v}},
                device=device, dtype=dtype, residual_multiplier=res_mult, fast_first_tok_only=True,
            )
            if av > best_av:
                best_av, best_a = av, alpha
        acc_g, out_g = eval_multi_regime_swaps(
            model_B, tok_B, *t_args,
            residual_replace={l_B: {"subj": pm_base.transform(ZA_subj[t_d, l_A].float().numpy(), gain=best_a)}},
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        p_res["regimes"]["gain_tuned_replacement"] = {
            "val_alpha": best_a, "test_acc": acc_g, "outcomes": out_g,
        }

        # Difference-vector steering: z_B(s) + alpha * s * (z_A(s') - z_A(s)) @ W
        best_ad, best_adv = 1.0, -1.0
        dA_val = ZA_subj[v_d, l_A].float().numpy() - ZA_subj[v_b, l_A].float().numpy()
        for alpha in (0.85, 1.0, 1.15, 1.3, 1.5):
            delta_v = pm_base.transform_delta(dA_val, alpha=alpha)
            av, _ = eval_multi_regime_swaps(
                model_B, tok_B, *v_args, residual_add={l_B: {"subj": delta_v}},
                device=device, dtype=dtype, residual_multiplier=res_mult, fast_first_tok_only=True,
            )
            if av > best_adv:
                best_adv, best_ad = av, alpha
        dA_te = ZA_subj[t_d, l_A].float().numpy() - ZA_subj[t_b, l_A].float().numpy()
        acc_dv, out_dv = eval_multi_regime_swaps(
            model_B, tok_B, *t_args,
            residual_add={l_B: {"subj": pm_base.transform_delta(dA_te, alpha=best_ad)}},
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        p_res["regimes"]["diff_vector_steering"] = {
            "val_alpha": best_ad, "test_acc": acc_dv, "outcomes": out_dv,
        }

        # Contiguous Multi-Layer Residual Windows (W = 2, 3, 4) — select offset pattern on val!
        for wname, off_list in window_cands.items():
            best_lp, best_wv = None, -1.0
            for offs in off_list:
                lpairs = systems.map_layer_window(l_A, l_B, L_A, L_B, offs, mapping_mode="proportional")
                rep_v = {}
                for la_w, lb_w in lpairs:
                    mp_w = systems.fit_procrustes_fast_qr(
                        ZA_subj[sh_tr, la_w].float().numpy(), ZB_subj[sh_tr, lb_w].float().numpy()
                    )
                    rep_v[lb_w] = {"subj": mp_w.transform(ZA_subj[v_d, la_w].float().numpy())}
                av, _ = eval_multi_regime_swaps(
                    model_B, tok_B, *v_args, residual_replace=rep_v,
                    device=device, dtype=dtype, residual_multiplier=res_mult, fast_first_tok_only=True,
                )
                if av > best_wv:
                    best_wv, best_lp = av, lpairs

            rep_t, self_t = {}, {}
            for la_w, lb_w in best_lp:
                mp_w = cached_fit_procrustes(
                    ("subj", mk_A, la_w, mk_B, lb_w, len(sh_tr), sh_tr[0], sh_tr[-1]),
                    ZA_subj[sh_tr, la_w].float().numpy(), ZB_subj[sh_tr, lb_w].float().numpy()
                )
                rep_t[lb_w] = {"subj": mp_w.transform(ZA_subj[t_d, la_w].float().numpy())}
                self_t[lb_w] = {"subj": ZB_subj[t_d, lb_w].float().numpy()}
            acc_wt, out_wt = eval_multi_regime_swaps(
                model_B, tok_B, *t_args, residual_replace=rep_t,
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            acc_ws, out_ws = eval_multi_regime_swaps(
                model_B, tok_B, *t_args, residual_replace=self_t,
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            p_res["regimes"][wname] = {
                "layer_pairs": best_lp, "test_acc": acc_wt, "self_acc": acc_ws,
                "outcomes": out_wt, "self_outcomes": out_ws,
            }

        # Additive MLP-Output-Only Patching (W = 1, 3 + cumulative 0..l_B) — Hebbian-MLP cross-model test!
        for w_sz, offs in [(1, [0]), (3, [-1, 0, 1])]:
            lpairs = systems.map_layer_window(l_A, l_B, L_A, L_B, offs, mapping_mode="proportional")
            mlp_t, mlp_self_t = {}, {}
            for la_w, lb_w in lpairs:
                mp_m = cached_fit_procrustes(
                    ("mlp", mk_A, la_w, mk_B, lb_w, len(sh_tr), sh_tr[0], sh_tr[-1]),
                    ZA_mlp[sh_tr, la_w].float().numpy(), ZB_mlp[sh_tr, lb_w].float().numpy()
                )
                mlp_t[lb_w] = {"subj": mp_m.transform(ZA_mlp[t_d, la_w].float().numpy())}
                mlp_self_t[lb_w] = {"subj": ZB_mlp[t_d, lb_w].float().numpy()}
            acc_mt, out_mt = eval_multi_regime_swaps(
                model_B, tok_B, *t_args, mlp_replace=mlp_t,
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            acc_ms, out_ms = eval_multi_regime_swaps(
                model_B, tok_B, *t_args, mlp_replace=mlp_self_t,
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            p_res["regimes"][f"mlp_only_W{w_sz}"] = {
                "layer_pairs": lpairs, "test_acc": acc_mt, "self_acc": acc_ms,
                "outcomes": out_mt, "self_outcomes": out_ms,
            }

        # Cumulative MLP-only 0..l_B (proportional donor layer for each recipient layer 0..l_B)
        lp_cum = [
            (int(np.clip(round(lb_w * (L_A / max(L_B, 1))), 0, L_A - 1)), lb_w)
            for lb_w in range(0, l_B + 1)
        ]
        mlp_ct, mlp_cs = {}, {}
        for la_w, lb_w in lp_cum:
            mp_m = cached_fit_procrustes(
                ("mlp", mk_A, la_w, mk_B, lb_w, len(sh_tr), sh_tr[0], sh_tr[-1]),
                ZA_mlp[sh_tr, la_w].float().numpy(), ZB_mlp[sh_tr, lb_w].float().numpy(),
            )
            mlp_ct[lb_w] = {"subj": mp_m.transform(ZA_mlp[t_d, la_w].float().numpy())}
            mlp_cs[lb_w] = {"subj": ZB_mlp[t_d, lb_w].float().numpy()}
        acc_mct, out_mct = eval_multi_regime_swaps(
            model_B, tok_B, *t_args, mlp_replace=mlp_ct,
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        acc_mcs, out_mcs = eval_multi_regime_swaps(
            model_B, tok_B, *t_args, mlp_replace=mlp_cs,
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        p_res["regimes"]["mlp_only_cumulative_0_to_lB"] = {
            "layer_pairs": lp_cum, "test_acc": acc_mct, "self_acc": acc_mcs,
            "outcomes": out_mct, "self_outcomes": out_mcs,
        }

        # Dual-Token Patching (same-layer last_subj + last_prompt)
        pm_last = systems.fit_procrustes(
            ZA_last[sh_tr, l_A].float().numpy(), ZB_last[sh_tr, l_B].float().numpy()
        )
        z_last_te = pm_last.transform(ZA_last[t_d, l_A].float().numpy())
        acc_dt, out_dt = eval_multi_regime_swaps(
            model_B, tok_B, *t_args,
            residual_replace={l_B: {"subj": z_base_te, "last": z_last_te}},
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        acc_dts, out_dts = eval_multi_regime_swaps(
            model_B, tok_B, *t_args,
            residual_replace={
                l_B: {
                    "subj": ZB_subj[t_d, l_B].float().numpy(),
                    "last": ZB_last[t_d, l_B].float().numpy(),
                }
            },
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        acc_lpo, out_lpo = eval_multi_regime_swaps(
            model_B, tok_B, *t_args,
            residual_replace={l_B: {"last": z_last_te}},
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        p_res["regimes"]["dual_token_same_layer"] = {
            "layers": [l_A, l_B], "test_acc": acc_dt, "self_acc": acc_dts,
            "outcomes": out_dt, "self_outcomes": out_dts,
        }
        p_res["regimes"]["last_prompt_only_same_layer"] = {
            "layers": [l_A, l_B], "test_acc": acc_lpo, "outcomes": out_lpo,
        }

        # Combined W3 Window + Same-Layer Dual-Token!
        lp_w3 = p_res["regimes"]["window_W3_residual"]["layer_pairs"]
        rep_c, self_c = {}, {}
        for la_w, lb_w in lp_w3:
            mp_w = cached_fit_procrustes(
                ("subj", mk_A, la_w, mk_B, lb_w, len(sh_tr), sh_tr[0], sh_tr[-1]),
                ZA_subj[sh_tr, la_w].float().numpy(), ZB_subj[sh_tr, lb_w].float().numpy()
            )
            rep_c[lb_w] = {"subj": mp_w.transform(ZA_subj[t_d, la_w].float().numpy())}
            self_c[lb_w] = {"subj": ZB_subj[t_d, lb_w].float().numpy()}
        rep_c.setdefault(l_B, {})["last"] = z_last_te
        self_c.setdefault(l_B, {})["last"] = ZB_last[t_d, l_B].float().numpy()
        acc_ct, out_ct = eval_multi_regime_swaps(
            model_B, tok_B, *t_args, residual_replace=rep_c,
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        acc_cs, out_cs = eval_multi_regime_swaps(
            model_B, tok_B, *t_args, residual_replace=self_c,
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        p_res["regimes"]["window_W3_plus_dual_token"] = {
            "layer_pairs": lp_w3, "test_acc": acc_ct, "self_acc": acc_cs,
            "outcomes": out_ct, "self_outcomes": out_cs,
        }

        pair_regimes[pair_key] = p_res
        print(
            f"  [ceiling {pair_key}] W1={acc_b1:.3f} (self={acc_self_w1:.3f}) | "
            f"gain={acc_g:.3f} diff={acc_dv:.3f} | "
            f"W2={p_res['regimes']['window_W2_residual']['test_acc']:.3f} "
            f"W3={p_res['regimes']['window_W3_residual']['test_acc']:.3f} "
            f"W4={p_res['regimes']['window_W4_residual']['test_acc']:.3f} | "
            f"MLP3={p_res['regimes']['mlp_only_W3']['test_acc']:.3f} | "
            f"dual={acc_dt:.3f} W3+dual={acc_ct:.3f} (self={acc_cs:.3f})"
        )

    return {"self_swap": self_regimes, "pairs": pair_regimes}


# =============================================================================
# PILLAR 4: Dark-Memory Recovery + Scientific Audit
# =============================================================================


def eval_recipient_dark_memory_audit(
    mk_B: str,
    model_B: Any,
    tok_B: Any,
    device: torch.device,
    dtype: torch.dtype,
    cache: dict,
    model_keys: Sequence[str],
) -> dict:
    """Comprehensive dark-memory benchmark + same-model unlocking ceiling + collateral damage audit."""
    facts = cache["dataset"]["facts"]
    evals = cache["evals"]
    states = cache["states"]
    baseline = cache["baseline"]
    eval_B = evals[mk_B]["facts"]
    L_B = int(cache["model_infos"][mk_B]["num_layers"])
    res_mult = float(cache["model_infos"][mk_B].get("residual_multiplier", 1.0))
    acc_B_direct = float(evals[mk_B]["summary"]["direct_acc"])

    ZB_subj = states[mk_B]["direct_subj"]
    ZB_icl3 = states[mk_B]["icl3_subj"]
    ZB_para = states[mk_B]["paraphrase_subj"]
    l_B_self = int(baseline["self_swap"][mk_B]["best_layer"])

    splits_idx = {sp: [f["idx"] for f in facts if f["split"] == sp] for sp in ("train", "val", "test")}
    # Evaluate on BOTH strict test_only and val_plus_test (strictly non-train; larger dark-memory N!)
    nontrain_idx = splits_idx["val"] + splits_idx["test"]

    # ------------------------------------------------------------------
    # Control #1: Same-Model Unlocking Ceiling (B's OWN ICL-3 / Paraphrase key -> Direct prompt!)
    # ------------------------------------------------------------------
    all_dark_B = [i for i in nontrain_idx if eval_B[i]["is_dark_memory_candidate"]]
    icl3_pass_dark = [i for i in all_dark_B if eval_B[i]["icl3_correct"]]
    para_pass_dark = [i for i in all_dark_B if eval_B[i]["paraphrase_correct"]]

    same_model_ceiling: Dict[str, Any] = {
        "n_total_dark_nontrain": len(all_dark_B),
        "n_icl3_pass": len(icl3_pass_dark),
        "n_para_pass": len(para_pass_dark),
        "per_fact_icl3_w1": {},
        "per_fact_para_w1": {},
        "per_fact_best_w1": {},
        "per_fact_best_w3": {},
    }

    lbs_w3 = sorted(set(int(np.clip(l_B_self + o, 0, L_B - 1)) for o in (-1, 0, 1)))

    if icl3_pass_dark:
        pairs_i = [(i, i) for i in icl3_pass_dark]
        args_i = _prep_fact_eval_args(facts, eval_B, pairs_i)
        acc_i1, out_i1 = eval_multi_regime_swaps(
            model_B, tok_B, *args_i,
            residual_replace={l_B_self: {"subj": ZB_icl3[icl3_pass_dark, l_B_self].float().numpy()}},
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        acc_i3, out_i3 = eval_multi_regime_swaps(
            model_B, tok_B, *args_i,
            residual_replace={lb: {"subj": ZB_icl3[icl3_pass_dark, lb].float().numpy()} for lb in lbs_w3},
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        for idx_f, o1, o3 in zip(icl3_pass_dark, out_i1, out_i3):
            same_model_ceiling["per_fact_icl3_w1"][str(idx_f)] = int(o1)
            same_model_ceiling["per_fact_best_w1"][str(idx_f)] = max(
                same_model_ceiling["per_fact_best_w1"].get(str(idx_f), 0), int(o1)
            )
            same_model_ceiling["per_fact_best_w3"][str(idx_f)] = max(
                same_model_ceiling["per_fact_best_w3"].get(str(idx_f), 0), int(o3)
            )
        same_model_ceiling["icl3_w1_acc"] = acc_i1
        same_model_ceiling["icl3_w3_acc"] = acc_i3
    else:
        same_model_ceiling["icl3_w1_acc"] = 0.0
        same_model_ceiling["icl3_w3_acc"] = 0.0

    if para_pass_dark:
        pairs_p = [(i, i) for i in para_pass_dark]
        args_p = _prep_fact_eval_args(facts, eval_B, pairs_p)
        acc_p1, out_p1 = eval_multi_regime_swaps(
            model_B, tok_B, *args_p,
            residual_replace={l_B_self: {"subj": ZB_para[para_pass_dark, l_B_self].float().numpy()}},
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        acc_p3, out_p3 = eval_multi_regime_swaps(
            model_B, tok_B, *args_p,
            residual_replace={lb: {"subj": ZB_para[para_pass_dark, lb].float().numpy()} for lb in lbs_w3},
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        for idx_f, o1, o3 in zip(para_pass_dark, out_p1, out_p3):
            same_model_ceiling["per_fact_para_w1"][str(idx_f)] = int(o1)
            same_model_ceiling["per_fact_best_w1"][str(idx_f)] = max(
                same_model_ceiling["per_fact_best_w1"].get(str(idx_f), 0), int(o1)
            )
            same_model_ceiling["per_fact_best_w3"][str(idx_f)] = max(
                same_model_ceiling["per_fact_best_w3"].get(str(idx_f), 0), int(o3)
            )
        same_model_ceiling["para_w1_acc"] = acc_p1
        same_model_ceiling["para_w3_acc"] = acc_p3
    else:
        same_model_ceiling["para_w1_acc"] = 0.0
        same_model_ceiling["para_w3_acc"] = 0.0

    union_eligible = sorted(set(icl3_pass_dark) | set(para_pass_dark))
    same_model_ceiling["n_icl_or_para_eligible"] = len(union_eligible)
    same_model_ceiling["best_own_key_w1_rate_on_eligible"] = (
        float(np.mean([same_model_ceiling["per_fact_best_w1"].get(str(i), 0) for i in union_eligible]))
        if union_eligible else 0.0
    )
    same_model_ceiling["best_own_key_w3_rate_on_eligible"] = (
        float(np.mean([same_model_ceiling["per_fact_best_w3"].get(str(i), 0) for i in union_eligible]))
        if union_eligible else 0.0
    )
    same_model_ceiling["best_own_key_w1_rate_on_all_dark"] = (
        float(np.mean([same_model_ceiling["per_fact_best_w1"].get(str(i), 0) for i in all_dark_B]))
        if all_dark_B else 0.0
    )

    print(
        f"  [dark CEILING {mk_B}] n_dark={len(all_dark_B)} | own ICL3(W1={same_model_ceiling['icl3_w1_acc']:.3f}, W3={same_model_ceiling['icl3_w3_acc']:.3f}) "
        f"| own Para(W1={same_model_ceiling['para_w1_acc']:.3f}, W3={same_model_ceiling['para_w3_acc']:.3f}) "
        f"| best_own_W1_all={same_model_ceiling['best_own_key_w1_rate_on_all_dark']:.3f}"
    )

    # ------------------------------------------------------------------
    # Single-Donor Dark-Memory & Collateral Damage Audit (all donors -> mk_B)
    # ------------------------------------------------------------------
    pair_dark: Dict[str, Any] = {}
    donor_maps_at_self_lb: Dict[str, Dict[str, Any]] = {}
    all_donors = [m for m in model_keys if m != mk_B]

    for mk_A in all_donors:
        pair_key = f"{mk_A}->{mk_B}"
        bp = baseline["pairs"][pair_key]
        eval_A = evals[mk_A]["facts"]
        acc_A_direct = float(evals[mk_A]["summary"]["direct_acc"])
        L_A = int(cache["model_infos"][mk_A]["num_layers"])
        l_A = int(bp["best_layer_A"])
        l_B = int(bp["best_layer_B"])
        sh_tr = bp["shared_train_indices"]
        val_pairs = [tuple(x) for x in bp["val_pairs"]]
        val_d = [d for d, _ in val_pairs]

        ZA_subj = states[mk_A]["direct_subj"]
        ZA_tr = ZA_subj[sh_tr, l_A].float().numpy()
        ZB_tr = ZB_subj[sh_tr, l_B].float().numpy()

        mp_proc = cached_fit_procrustes(
            ("subj", mk_A, l_A, mk_B, l_B, len(sh_tr), sh_tr[0], sh_tr[-1]),
            ZA_tr, ZB_tr
        )
        mp_monge, monge_lam, _ = fit_best_monge_on_val(
            model_B, tok_B, facts, eval_B, val_pairs,
            ZA_tr, ZB_tr, ZA_subj[val_d, l_A].float().numpy(),
            l_B, device, dtype, res_mult,
        )
        # Also tune gain alpha for Monge-Procrustes strictly on val_pairs (never on dark memories!)
        best_mg_alpha, best_mg_val = 1.0, -1.0
        val_args_dm = _prep_fact_eval_args(facts, eval_B, val_pairs)
        zm_val_base = mp_monge.transform(ZA_subj[val_d, l_A].float().numpy())
        for alpha in (1.0, 1.15, 1.3):
            z_va = mp_monge.mu_B[None, :] + alpha * (zm_val_base - mp_monge.mu_B[None, :])
            ava, _ = eval_multi_regime_swaps(
                model_B, tok_B, *val_args_dm,
                residual_replace={l_B: {"subj": z_va}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
                fast_first_tok_only=True,
            )
            if ava > best_mg_val:
                best_mg_val, best_mg_alpha = ava, alpha
        mp_rand = systems.fit_random_orthogonal_control(ZA_tr, ZB_tr, seed=SEED + 501 + hash(pair_key) % 1000)

        # W3 window maps for multi-layer dark memory unlocking
        lp_w3 = systems.map_layer_window(l_A, l_B, L_A, L_B, [-1, 0, 1], mapping_mode="proportional")
        w3_maps = [
            (
                la_w, lb_w,
                cached_fit_procrustes(
                    ("subj", mk_A, la_w, mk_B, lb_w, len(sh_tr), sh_tr[0], sh_tr[-1]),
                    ZA_subj[sh_tr, la_w].float().numpy(), ZB_subj[sh_tr, lb_w].float().numpy(),
                ),
            )
            for la_w, lb_w in lp_w3
        ]

        # Also store donor map into common l_B_self for Multi-Donor Barycenter dark-memory unlocking!
        la_self, _ = _select_donor_layer_for_target_lB(
            model_B, tok_B, facts, eval_B, sh_tr, val_pairs,
            ZA_subj, ZB_subj, l_A, l_B_self, L_A, device, dtype, res_mult,
        )
        mp_self_proc = systems.fit_procrustes(
            ZA_subj[sh_tr, la_self].float().numpy(), ZB_subj[sh_tr, l_B_self].float().numpy()
        )
        mp_self_monge, _, _ = fit_best_monge_on_val(
            model_B, tok_B, facts, eval_B, val_pairs,
            ZA_subj[sh_tr, la_self].float().numpy(), ZB_subj[sh_tr, l_B_self].float().numpy(),
            ZA_subj[val_d, la_self].float().numpy(), l_B_self, device, dtype, res_mult,
        )
        donor_maps_at_self_lb[mk_A] = {
            "la": la_self,
            "proc": mp_self_proc,
            "monge": mp_self_monge,
        }

        scope_res = {}
        for scope_name, scope_ids in [("test_only", splits_idx["test"]), ("val_plus_test", nontrain_idx)]:
            dark_ids = [
                i for i in scope_ids
                if eval_A[i]["direct_correct"] and (not eval_B[i]["direct_correct"]) and eval_B[i]["is_dark_memory_candidate"]
            ]
            nondark_ids = [
                i for i in scope_ids
                if eval_A[i]["direct_correct"] and (not eval_B[i]["direct_correct"]) and (not eval_B[i]["is_dark_memory_candidate"])
            ]
            # Already-recalled facts by BOTH A and B (Control #2: Collateral Damage / Retention Rate!)
            recalled_ids = [
                i for i in scope_ids
                if eval_A[i]["direct_correct"] and eval_B[i]["direct_correct"]
            ]

            def _eval_subset(id_list: List[int]) -> Dict[str, List[int]]:
                if not id_list:
                    return {k: [] for k in ("procrustes", "monge", "monge_gain", "window_W3", "random")}
                id_pairs = [(i, i) for i in id_list]
                p_args = _prep_fact_eval_args(facts, eval_B, id_pairs)
                ZA_sub = ZA_subj[id_list, l_A].float().numpy()
                _, o_proc = eval_multi_regime_swaps(
                    model_B, tok_B, *p_args,
                    residual_replace={l_B: {"subj": mp_proc.transform(ZA_sub)}},
                    device=device, dtype=dtype, residual_multiplier=res_mult,
                )
                zm_sub = mp_monge.transform(ZA_sub)
                _, o_monge = eval_multi_regime_swaps(
                    model_B, tok_B, *p_args,
                    residual_replace={l_B: {"subj": zm_sub}},
                    device=device, dtype=dtype, residual_multiplier=res_mult,
                )
                zm_gain_sub = mp_monge.mu_B[None, :] + best_mg_alpha * (zm_sub - mp_monge.mu_B[None, :])
                _, o_mg = eval_multi_regime_swaps(
                    model_B, tok_B, *p_args,
                    residual_replace={l_B: {"subj": zm_gain_sub}},
                    device=device, dtype=dtype, residual_multiplier=res_mult,
                )
                rep_w3 = {lb_w: {"subj": mp_w.transform(ZA_subj[id_list, la_w].float().numpy())} for la_w, lb_w, mp_w in w3_maps}
                _, o_w3 = eval_multi_regime_swaps(
                    model_B, tok_B, *p_args,
                    residual_replace=rep_w3,
                    device=device, dtype=dtype, residual_multiplier=res_mult,
                )
                _, o_rand = eval_multi_regime_swaps(
                    model_B, tok_B, *p_args,
                    residual_replace={l_B: {"subj": mp_rand.transform(ZA_sub)}},
                    device=device, dtype=dtype, residual_multiplier=res_mult,
                )
                return {
                    "procrustes": o_proc,
                    "monge": o_monge,
                    "monge_gain": o_mg,
                    "window_W3": o_w3,
                    "random": o_rand,
                }

            out_dark = _eval_subset(dark_ids)
            out_nd = _eval_subset(nondark_ids)
            out_rec = _eval_subset(recalled_ids)
            ceil_vec = [same_model_ceiling["per_fact_best_w1"].get(str(i), 0) for i in dark_ids]

            methods_summary = {}
            for mname in ("procrustes", "monge", "monge_gain", "window_W3", "random"):
                s = systems.summarize_dark_memory_outcomes(
                    out_dark[mname], out_nd[mname], out_rec[mname], same_model_ceiling=ceil_vec
                )
                s["dark_outcomes"] = out_dark[mname]
                s["nondark_outcomes"] = out_nd[mname]
                s["retained_outcomes"] = out_rec[mname]
                methods_summary[mname] = s

            scope_res[scope_name] = {
                "dark_ids": dark_ids,
                "nondark_ids": nondark_ids,
                "recalled_ids": recalled_ids,
                "same_model_ceiling_outcomes": ceil_vec,
                "methods": methods_summary,
            }

        pair_dark[pair_key] = {
            "donor": mk_A,
            "recipient": mk_B,
            "donor_direct_acc": acc_A_direct,
            "recipient_direct_acc": acc_B_direct,
            "weaker_donor_to_stronger_recipient": acc_A_direct < acc_B_direct,
            "layers": [l_A, l_B],
            "monge_lam": monge_lam,
            "monge_val_gain": best_mg_alpha,
            "scopes": scope_res,
        }
        vt = scope_res["val_plus_test"]["methods"]
        tag = "WEAKER->STRONGER" if acc_A_direct < acc_B_direct else "STRONGER->WEAKER"
        print(
            f"  [dark {pair_key} {tag}] n_dark={vt['procrustes']['n_dark']} | "
            f"proc={vt['procrustes']['dark_recovery_rate']:.3f} (ret={vt['procrustes']['retention_rate']:.3f}) | "
            f"monge={vt['monge']['dark_recovery_rate']:.3f} (ret={vt['monge']['retention_rate']:.3f}) | "
            f"W3={vt['window_W3']['dark_recovery_rate']:.3f} (ret={vt['window_W3']['retention_rate']:.3f}) | "
            f"rand={vt['random']['dark_recovery_rate']:.3f}"
        )

    # ------------------------------------------------------------------
    # Multi-Donor Platonic Barycenter Dark-Memory Recovery & Collateral Damage!
    # For every non-train dark memory of B recalled by >= 2 donors, aggregate ALL recalling donors!
    # Also test collateral damage on facts recalled by ALL donors + B!
    # ------------------------------------------------------------------
    multi_dark_res: Dict[str, Any] = {}
    for scope_name, scope_ids in [("test_only", splits_idx["test"]), ("val_plus_test", nontrain_idx)]:
        # Dark memories of B recalled by >= 2 donors
        dm_items = []
        for i in scope_ids:
            if (not eval_B[i]["direct_correct"]) and eval_B[i]["is_dark_memory_candidate"]:
                rec_donors = [m for m in all_donors if evals[m]["facts"][i]["direct_correct"]]
                if len(rec_donors) >= 2:
                    dm_items.append((i, rec_donors))
        # Already-recalled facts recalled by ALL donors + B (for clean multi-donor collateral damage)
        rec_all_ids = [
            i for i in scope_ids
            if eval_B[i]["direct_correct"] and all(evals[m]["facts"][i]["direct_correct"] for m in all_donors)
        ]

        if dm_items:
            dm_ids = [i for i, _ in dm_items]
            dm_pairs = [(i, i) for i in dm_ids]
            dm_args = _prep_fact_eval_args(facts, eval_B, dm_pairs)

            z_bary_proc_list, z_bary_monge_list, z_single_best_list = [], [], []
            for i, dlist in dm_items:
                # Aggregate recalling donors with uniform + rms_norm
                mapped_p = [
                    donor_maps_at_self_lb[m]["proc"].transform(
                        states[m]["direct_subj"][i, donor_maps_at_self_lb[m]["la"]].float().numpy()
                    )
                    for m in dlist
                ]
                mus_p = [donor_maps_at_self_lb[m]["proc"].mu_b for m in dlist]
                cfg_u = systems.BarycenterConfig(
                    donor_names=dlist, weights=np.full(len(dlist), 1.0 / len(dlist)),
                    weight_method="uniform", norm_mode="rms_norm", gain=1.0,
                )
                zp, _ = systems.aggregate_multi_donor_keys(mapped_p, mus_p, cfg_u)
                z_bary_proc_list.append(zp)

                mapped_m = [
                    donor_maps_at_self_lb[m]["monge"].transform(
                        states[m]["direct_subj"][i, donor_maps_at_self_lb[m]["la"]].float().numpy()
                    )
                    for m in dlist
                ]
                mus_m = [donor_maps_at_self_lb[m]["monge"].mu_B for m in dlist]
                zm, _ = systems.aggregate_multi_donor_keys(mapped_m, mus_m, cfg_u)
                z_bary_monge_list.append(zm)

                # Single-donor comparison: pick the highest-direct-acc recalling donor for this fact
                best_d = max(dlist, key=lambda m: evals[m]["summary"]["direct_acc"])
                z_single_best_list.append(
                    donor_maps_at_self_lb[best_d]["proc"].transform(
                        states[best_d]["direct_subj"][i, donor_maps_at_self_lb[best_d]["la"]].float().numpy()
                    )
                )

            acc_bp, out_bp = eval_multi_regime_swaps(
                model_B, tok_B, *dm_args,
                residual_replace={l_B_self: {"subj": np.stack(z_bary_proc_list, axis=0)}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            acc_bm, out_bm = eval_multi_regime_swaps(
                model_B, tok_B, *dm_args,
                residual_replace={l_B_self: {"subj": np.stack(z_bary_monge_list, axis=0)}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            acc_sb, out_sb = eval_multi_regime_swaps(
                model_B, tok_B, *dm_args,
                residual_replace={l_B_self: {"subj": np.stack(z_single_best_list, axis=0)}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            ceil_m = [same_model_ceiling["per_fact_best_w1"].get(str(i), 0) for i in dm_ids]
        else:
            dm_ids, acc_bp, out_bp, acc_bm, out_bm, acc_sb, out_sb, ceil_m = [], 0.0, [], 0.0, [], 0.0, [], []

        # Collateral damage on rec_all_ids using all M=4 donors barycenter vs single donor!
        if rec_all_ids:
            rec_pairs = [(i, i) for i in rec_all_ids]
            rec_args = _prep_fact_eval_args(facts, eval_B, rec_pairs)
            cfg_all = systems.BarycenterConfig(
                donor_names=all_donors, weights=np.full(len(all_donors), 1.0 / len(all_donors)),
                weight_method="uniform", norm_mode="rms_norm", gain=1.0,
            )
            mp_rec_p = [
                donor_maps_at_self_lb[m]["proc"].transform(
                    states[m]["direct_subj"][rec_all_ids, donor_maps_at_self_lb[m]["la"]].float().numpy()
                )
                for m in all_donors
            ]
            z_rec_bp, _ = systems.aggregate_multi_donor_keys(
                mp_rec_p, [donor_maps_at_self_lb[m]["proc"].mu_b for m in all_donors], cfg_all
            )
            ret_bp, out_ret_bp = eval_multi_regime_swaps(
                model_B, tok_B, *rec_args,
                residual_replace={l_B_self: {"subj": z_rec_bp}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )

            mp_rec_m = [
                donor_maps_at_self_lb[m]["monge"].transform(
                    states[m]["direct_subj"][rec_all_ids, donor_maps_at_self_lb[m]["la"]].float().numpy()
                )
                for m in all_donors
            ]
            z_rec_bm, _ = systems.aggregate_multi_donor_keys(
                mp_rec_m, [donor_maps_at_self_lb[m]["monge"].mu_B for m in all_donors], cfg_all
            )
            ret_bm, out_ret_bm = eval_multi_regime_swaps(
                model_B, tok_B, *rec_args,
                residual_replace={l_B_self: {"subj": z_rec_bm}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            best_d0 = max(all_donors, key=lambda m: evals[m]["summary"]["direct_acc"])
            z_rec_sb = donor_maps_at_self_lb[best_d0]["proc"].transform(
                states[best_d0]["direct_subj"][rec_all_ids, donor_maps_at_self_lb[best_d0]["la"]].float().numpy()
            )
            ret_sb, out_ret_sb = eval_multi_regime_swaps(
                model_B, tok_B, *rec_args,
                residual_replace={l_B_self: {"subj": z_rec_sb}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
        else:
            ret_bp, out_ret_bp, ret_bm, out_ret_bm, ret_sb, out_ret_sb = 0.0, [], 0.0, [], 0.0, []

        multi_dark_res[scope_name] = {
            "n_dark_ge2_donors": len(dm_ids),
            "dark_ids": dm_ids,
            "same_model_ceiling_outcomes": ceil_m,
            "bary_procrustes_dark_acc": acc_bp,
            "bary_procrustes_dark_outcomes": out_bp,
            "bary_monge_dark_acc": acc_bm,
            "bary_monge_dark_outcomes": out_bm,
            "best_single_donor_dark_acc": acc_sb,
            "best_single_donor_dark_outcomes": out_sb,
            "n_recalled_all_donors": len(rec_all_ids),
            "bary_procrustes_retention": ret_bp,
            "bary_procrustes_retention_outcomes": out_ret_bp,
            "bary_monge_retention": ret_bm,
            "bary_monge_retention_outcomes": out_ret_bm,
            "best_single_donor_retention": ret_sb,
            "best_single_donor_retention_outcomes": out_ret_sb,
        }

    vt_m = multi_dark_res["val_plus_test"]
    print(
        f"  [dark MULTI-DONOR {mk_B}] n_dark(>=2)={vt_m['n_dark_ge2_donors']} | "
        f"single_best={vt_m['best_single_donor_dark_acc']:.3f} (ret={vt_m['best_single_donor_retention']:.3f}) | "
        f"bary_proc={vt_m['bary_procrustes_dark_acc']:.3f} (ret={vt_m['bary_procrustes_retention']:.3f}) | "
        f"bary_monge={vt_m['bary_monge_dark_acc']:.3f} (ret={vt_m['bary_monge_retention']:.3f})"
    )

    return {
        "recipient": mk_B,
        "recipient_direct_acc": acc_B_direct,
        "same_model_ceiling": same_model_ceiling,
        "pairs": pair_dark,
        "multi_donor": multi_dark_res,
    }


# =============================================================================
# PILLAR 5: 2-Hop Unspoken Latent Bridge Transfer
# =============================================================================


def build_2hop_counterfactual_pairs(items: List[dict], valid_indices: Sequence[int]) -> List[Tuple[int, int]]:
    """Pair items within the same hop_type with distinct subject, bridge_entity, and answer."""
    by_type: Dict[str, List[int]] = defaultdict(list)
    for idx in sorted(valid_indices):
        by_type[items[idx]["hop_type"]].append(idx)

    pairs: List[Tuple[int, int]] = []
    for ht in sorted(by_type.keys()):
        idxs = by_type[ht]
        m = len(idxs)
        if m < 2:
            continue
        for pos, d_idx in enumerate(idxs):
            ds = items[d_idx]["subject"].strip().lower()
            db = items[d_idx]["bridge_entity"].strip().lower()
            da = items[d_idx]["answer"].strip().lower()
            for step in range(1, m):
                c = idxs[(pos + step) % m]
                if (
                    items[c]["subject"].strip().lower() != ds
                    and items[c]["bridge_entity"].strip().lower() != db
                    and items[c]["answer"].strip().lower() != da
                ):
                    pairs.append((d_idx, c))
                    break
    return pairs


def extract_cross_relation_bridge_keys(
    model: Any,
    tok: Any,
    items_2hop: List[dict],
    layer_indices: Sequence[int],
    device: torch.device,
) -> Dict[int, np.ndarray]:
    """Extract 1-hop entity keys for each 2-hop item's `bridge_entity` using a COMPLETELY DIFFERENT
    1-hop relation (`country_continent`) that has ZERO overlap with the 2-hop answer (capital/currency/language)!
    This provides an airtight no-value-leakage control for 1-hop -> 2-hop bridge transfer!
    """
    prompts = []
    spans = []
    for x in items_2hop:
        ent = x["bridge_entity"]
        # None of dataset_2hop's hop2_relations are country_continent!
        pref = "The country of "
        p = f"{pref}{ent} is located on the continent of"
        prompts.append(p)
        spans.append((len(pref), len(pref) + len(ent)))

    ids, mask, spos, _ = systems.tokenize_left_padded_with_positions(tok, prompts, spans)
    return systems.extract_residual_states_batched(
        model, ids, mask, spos, layer_indices=layer_indices, batch_size=32, device=str(device)
    )


def eval_recipient_2hop_bridge(
    mk_B: str,
    model_B: Any,
    tok_B: Any,
    device: torch.device,
    dtype: torch.dtype,
    cache: dict,
    model_keys: Sequence[str],
    cross_rel_bridge_cache: Dict[str, Dict[int, np.ndarray]],
) -> dict:
    """Evaluate 2-Hop->2-Hop transfer, 1-Hop Entity Key -> 2-Hop Bridge Injection (same-rel & cross-rel),
    and 2-hop compositional dark-memory rescue for recipient mk_B."""
    facts = cache["dataset"]["facts"]
    items_2 = cache["dataset_2hop"]["items"]
    evals = cache["evals"]
    evals_2 = cache["evals_2hop"]
    states = cache["states"]
    states_2 = cache["states_2hop"]
    baseline = cache["baseline"]

    ev2_B = evals_2[mk_B]["items"]
    L_B = int(cache["model_infos"][mk_B]["num_layers"])
    res_mult = float(cache["model_infos"][mk_B].get("residual_multiplier", 1.0))

    Z2_B_subj = states_2[mk_B]["states_subj"]
    Z2_B_last = states_2[mk_B]["states_last"]
    Z2_B_hop2 = states_2[mk_B]["hop2_subj"]

    # Helper to prep 2-hop args
    def _prep_2hop(pair_list: Sequence[Tuple[int, int]]):
        pr = [items_2[b]["direct_prompt"] for _, b in pair_list]
        sp = [ev2_B[b]["subj_tok_idx_direct"] for _, b in pair_list]
        sl = [ev2_B[b]["seq_len_direct"] for _, b in pair_list]
        al = [items_2[d]["aliases"] for d, _ in pair_list]
        rl = [items_2[d]["hop2_relation"] for d, _ in pair_list]
        return pr, sp, sl, al, rl

    # Candidate intermediate bridge layers for 1-hop -> 2-hop injection
    bridge_fracs = (0.25, 0.35, 0.45, 0.55)
    bridge_layers_B = sorted(set(int(np.clip(round(f * L_B), 1, L_B - 2)) for f in bridge_fracs))

    res_B: Dict[str, Any] = {
        "recipient": mk_B,
        "direct_2hop_acc": evals_2[mk_B]["summary"]["direct_2hop_acc"],
        "hop1_acc": evals_2[mk_B]["summary"]["hop1_acc"],
        "hop2_acc": evals_2[mk_B]["summary"]["hop2_acc"],
        "pairs": {},
    }

    for mk_A in model_keys:
        if mk_A == mk_B:
            continue
        pair_key = f"{mk_A}->{mk_B}"
        ev2_A = evals_2[mk_A]["items"]
        bp = baseline["pairs"][pair_key]
        l_A = int(bp["best_layer_A"])
        l_B = int(bp["best_layer_B"])
        L_A = int(cache["model_infos"][mk_A]["num_layers"])
        # Filter 1-hop training indices to strictly exclude ANY subject or bridge_entity
        # appearing anywhere in dataset_2hop (zero cross-dataset entity leakage per #52 audit!)
        all_2hop_entities = set()
        for x2 in items_2:
            all_2hop_entities.add(x2["subject"].strip().lower())
            all_2hop_entities.add(x2["bridge_entity"].strip().lower())
        sh_tr_1hop = [
            i for i in bp["shared_train_indices"]
            if facts[i]["subject_norm"].strip().lower() not in all_2hop_entities
        ]

        # Fit 1-hop Procrustes map (ZERO 2-hop training & ZERO 2-hop entity overlap!)
        pm_1hop = systems.fit_procrustes(
            states[mk_A]["direct_subj"][sh_tr_1hop, l_A].float().numpy(),
            states[mk_B]["direct_subj"][sh_tr_1hop, l_B].float().numpy(),
        )
        pm_rand = systems.fit_random_orthogonal_control(
            states[mk_A]["direct_subj"][sh_tr_1hop, l_A].float().numpy(),
            states[mk_B]["direct_subj"][sh_tr_1hop, l_B].float().numpy(),
            seed=SEED + 901 + hash(pair_key) % 1000,
        )

        # Shared 2-hop recalled items (evaluate on ALL shared 2-hop items since 1-hop map uses ZERO 2-hop data!)
        # Also break out test-split-only subset for strictness!
        shared_2hop_all = [
            i for i in range(len(items_2))
            if ev2_A[i]["direct_correct"] and ev2_B[i]["direct_correct"]
        ]
        pairs_2hop = build_2hop_counterfactual_pairs(items_2, shared_2hop_all)
        if not pairs_2hop:
            continue
        d2 = [d for d, _ in pairs_2hop]
        args_2 = _prep_2hop(pairs_2hop)

        # ---- Exp 5A: 2-Hop -> 2-Hop Transfer via 1-Hop Procrustes Map (W=1 and W=3) ----
        z2_A_subj = states_2[mk_A]["states_subj"][d2, l_A].float().numpy()
        acc_2h_w1, out_2h_w1 = eval_multi_regime_swaps(
            model_B, tok_B, *args_2,
            residual_replace={l_B: {"subj": pm_1hop.transform(z2_A_subj)}},
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        acc_2h_self, out_2h_self = eval_multi_regime_swaps(
            model_B, tok_B, *args_2,
            residual_replace={l_B: {"subj": Z2_B_subj[d2, l_B].float().numpy()}},
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        acc_2h_rand, out_2h_rand = eval_multi_regime_swaps(
            model_B, tok_B, *args_2,
            residual_replace={l_B: {"subj": pm_rand.transform(z2_A_subj)}},
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )

        # W3 window 2-hop -> 2-hop using 1-hop maps
        lp_w3 = systems.map_layer_window(l_A, l_B, L_A, L_B, [-1, 0, 1], mapping_mode="proportional")
        rep_2h_w3, self_2h_w3 = {}, {}
        for la_w, lb_w in lp_w3:
            mp_w = systems.fit_procrustes(
                states[mk_A]["direct_subj"][sh_tr_1hop, la_w].float().numpy(),
                states[mk_B]["direct_subj"][sh_tr_1hop, lb_w].float().numpy(),
            )
            rep_2h_w3[lb_w] = {"subj": mp_w.transform(states_2[mk_A]["states_subj"][d2, la_w].float().numpy())}
            self_2h_w3[lb_w] = {"subj": Z2_B_subj[d2, lb_w].float().numpy()}
        acc_2h_w3, out_2h_w3 = eval_multi_regime_swaps(
            model_B, tok_B, *args_2, residual_replace=rep_2h_w3,
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )
        acc_2h_w3_self, out_2h_w3_self = eval_multi_regime_swaps(
            model_B, tok_B, *args_2, residual_replace=self_2h_w3,
            device=device, dtype=dtype, residual_multiplier=res_mult,
        )

        # ---- Exp 5B: 1-Hop Bridge Entity Key -> 2-Hop Intermediate-Layer Injection! ----
        # Test BOTH same-relation 1-hop key (hop2_subj) AND cross-relation 1-hop key (country_continent)!
        # Sweep recipient injection layer lb in bridge_layers_B at BOTH `subj` and `last` token positions!
        bridge_curve = {}
        for lb_inj in bridge_layers_B:
            la_inj = int(np.clip(round(lb_inj * (L_A / L_B)), 0, L_A - 1))
            # 1-hop map at (la_inj, lb_inj) for `subj` -> `subj` and `subj` -> `last`
            mp_s2s = systems.fit_procrustes(
                states[mk_A]["direct_subj"][sh_tr_1hop, la_inj].float().numpy(),
                states[mk_B]["direct_subj"][sh_tr_1hop, lb_inj].float().numpy(),
            )
            mp_s2l = systems.fit_procrustes(
                states[mk_A]["direct_subj"][sh_tr_1hop, la_inj].float().numpy(),
                states[mk_B]["direct_last"][sh_tr_1hop, lb_inj].float().numpy(),
            )
            mp_r_inj = systems.fit_random_orthogonal_control(
                states[mk_A]["direct_subj"][sh_tr_1hop, la_inj].float().numpy(),
                states[mk_B]["direct_subj"][sh_tr_1hop, lb_inj].float().numpy(),
                seed=SEED + 1200 + lb_inj,
            )

            z_hop2_A = states_2[mk_A]["hop2_subj"][d2, la_inj].float().numpy()
            z_hop2_B = Z2_B_hop2[d2, lb_inj].float().numpy()

            # Inject at `last_subj` of 2-hop prompt
            acc_h2_subj, out_h2_subj = eval_multi_regime_swaps(
                model_B, tok_B, *args_2,
                residual_replace={lb_inj: {"subj": mp_s2s.transform(z_hop2_A)}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            acc_h2_subj_self, out_h2_subj_self = eval_multi_regime_swaps(
                model_B, tok_B, *args_2,
                residual_replace={lb_inj: {"subj": z_hop2_B}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            acc_h2_subj_rand, out_h2_subj_rand = eval_multi_regime_swaps(
                model_B, tok_B, *args_2,
                residual_replace={lb_inj: {"subj": mp_r_inj.transform(z_hop2_A)}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            # Inject at `last_prompt` of 2-hop prompt
            acc_h2_last, out_h2_last = eval_multi_regime_swaps(
                model_B, tok_B, *args_2,
                residual_replace={lb_inj: {"last": mp_s2l.transform(z_hop2_A)}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )

            entry = {
                "la": la_inj,
                "lb": lb_inj,
                "rel_depth": lb_inj / L_B,
                "hop2_1hop_to_2hop_subj_acc": acc_h2_subj,
                "hop2_1hop_to_2hop_subj_outcomes": out_h2_subj,
                "hop2_1hop_to_2hop_subj_self_acc": acc_h2_subj_self,
                "hop2_1hop_to_2hop_subj_self_outcomes": out_h2_subj_self,
                "hop2_1hop_to_2hop_subj_rand_acc": acc_h2_subj_rand,
                "hop2_1hop_to_2hop_last_acc": acc_h2_last,
                "hop2_1hop_to_2hop_last_outcomes": out_h2_last,
            }

            # Cross-relation no-value-leakage 1-hop bridge key (country_continent -> 2-hop!)
            if mk_A in cross_rel_bridge_cache and la_inj in cross_rel_bridge_cache[mk_A]:
                z_xref_A = cross_rel_bridge_cache[mk_A][la_inj][d2]
                acc_xr, out_xr = eval_multi_regime_swaps(
                    model_B, tok_B, *args_2,
                    residual_replace={lb_inj: {"subj": mp_s2s.transform(z_xref_A)}},
                    device=device, dtype=dtype, residual_multiplier=res_mult,
                )
                entry["cross_rel_1hop_to_2hop_subj_acc"] = acc_xr
                entry["cross_rel_1hop_to_2hop_subj_outcomes"] = out_xr
            if mk_B in cross_rel_bridge_cache and lb_inj in cross_rel_bridge_cache[mk_B]:
                z_xref_B = cross_rel_bridge_cache[mk_B][lb_inj][d2]
                acc_xr_s, out_xr_s = eval_multi_regime_swaps(
                    model_B, tok_B, *args_2,
                    residual_replace={lb_inj: {"subj": z_xref_B}},
                    device=device, dtype=dtype, residual_multiplier=res_mult,
                )
                entry["cross_rel_1hop_to_2hop_subj_self_acc"] = acc_xr_s
                entry["cross_rel_1hop_to_2hop_subj_self_outcomes"] = out_xr_s

            bridge_curve[str(lb_inj)] = entry

        # ---- Exp 5C: 2-Hop Compositional Dark-Memory Rescue! ----
        # Items where B knows BOTH hop1 and hop2, fails 2-hop direct, and donor A passes 2-hop direct!
        comp_dark = [
            i for i in range(len(items_2))
            if ev2_A[i]["direct_correct"]
            and (not ev2_B[i]["direct_correct"])
            and ev2_B[i]["hop1_correct"]
            and ev2_B[i]["hop2_correct"]
        ]
        if comp_dark:
            cd_pairs = [(i, i) for i in comp_dark]
            cd_args = _prep_2hop(cd_pairs)
            acc_cd_2h, out_cd_2h = eval_multi_regime_swaps(
                model_B, tok_B, *cd_args,
                residual_replace={l_B: {"subj": pm_1hop.transform(states_2[mk_A]["states_subj"][comp_dark, l_A].float().numpy())}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            acc_cd_1h, out_cd_1h = eval_multi_regime_swaps(
                model_B, tok_B, *cd_args,
                residual_replace={l_B: {"subj": pm_1hop.transform(states_2[mk_A]["hop2_subj"][comp_dark, l_A].float().numpy())}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            acc_cd_rand, out_cd_rand = eval_multi_regime_swaps(
                model_B, tok_B, *cd_args,
                residual_replace={l_B: {"subj": pm_rand.transform(states_2[mk_A]["states_subj"][comp_dark, l_A].float().numpy())}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
        else:
            acc_cd_2h, out_cd_2h, acc_cd_1h, out_cd_1h, acc_cd_rand, out_cd_rand = 0.0, [], 0.0, [], 0.0, []

        best_1h_subj = max(v["hop2_1hop_to_2hop_subj_acc"] for v in bridge_curve.values())
        best_xr_subj = max(v.get("cross_rel_1hop_to_2hop_subj_acc", 0.0) for v in bridge_curve.values())

        res_B["pairs"][pair_key] = {
            "n_2hop_pairs": len(pairs_2hop),
            "n_filtered_1hop_train": len(sh_tr_1hop),
            "zero_2hop_entity_overlap": True,
            "pairs_2hop": pairs_2hop,
            "pairs_2hop_splits": [[items_2[d]["split"], items_2[b]["split"]] for d, b in pairs_2hop],
            "exp5a_2hop_to_2hop": {
                "layers": [l_A, l_B],
                "w1_acc": acc_2h_w1,
                "w1_outcomes": out_2h_w1,
                "w1_self_acc": acc_2h_self,
                "w1_self_outcomes": out_2h_self,
                "w1_rand_acc": acc_2h_rand,
                "w1_rand_outcomes": out_2h_rand,
                "w3_acc": acc_2h_w3,
                "w3_outcomes": out_2h_w3,
                "w3_self_acc": acc_2h_w3_self,
                "w3_self_outcomes": out_2h_w3_self,
            },
            "exp5b_1hop_entity_to_2hop_bridge": bridge_curve,
            "exp5c_compositional_dark_rescue": {
                "n_comp_dark": len(comp_dark),
                "item_indices": comp_dark,
                "via_donor_2hop_acc": acc_cd_2h,
                "via_donor_2hop_outcomes": out_cd_2h,
                "via_donor_1hop_bridge_acc": acc_cd_1h,
                "via_donor_1hop_bridge_outcomes": out_cd_1h,
                "random_control_acc": acc_cd_rand,
                "random_control_outcomes": out_cd_rand,
            },
        }
        print(
            f"  [2-hop {pair_key}] n={len(pairs_2hop)} | 2h->2h W1={acc_2h_w1:.3f} W3={acc_2h_w3:.3f} (self={acc_2h_self:.3f}, rand={acc_2h_rand:.3f}) | "
            f"1h->2h best={best_1h_subj:.3f} cross_rel={best_xr_subj:.3f} | "
            f"comp_dark({len(comp_dark)}): 2h={acc_cd_2h:.3f} 1h_bridge={acc_cd_1h:.3f}"
        )

    return res_B


# =============================================================================
# Aggregate Summary Builder
# =============================================================================


def build_aggregate_summary(results: dict) -> dict:
    agg: Dict[str, Any] = {}

    # 1. Multi-donor barycenter aggregate
    if "barycenter" in results and results["barycenter"]:
        m3_single_mean, m3_single_best, m3_no_norm, m3_unif_rms, m3_gls_rms, m3_monge, m3_gpa, m3_rhos = [], [], [], [], [], [], [], []
        for rec, d in results["barycenter"].items():
            g = d["groups"].get("M3_cross_family") or next(iter(d["groups"].values()), None)
            if not g:
                continue
            m3_single_mean.append(g["mean_single_test_acc"])
            m3_single_best.append(g["best_single_test_acc"])
            m3_no_norm.append(g["variants"]["uniform_no_norm"]["test_acc"])
            m3_unif_rms.append(g["variants"]["uniform_rms_norm"]["test_acc"])
            m3_gls_rms.append(g["variants"]["inv_cov_gls_rms_norm"]["test_acc"])
            m3_monge.append(g["variants"]["monge_inv_cov_rms_norm"]["test_acc"])
            m3_gpa.append(g["variants"]["fig6_native_gpa_barycenter"]["test_acc"])
            if g["variants"]["inv_cov_gls_rms_norm"]["mean_offdiag_rho"] is not None:
                m3_rhos.append(g["variants"]["inv_cov_gls_rms_norm"]["mean_offdiag_rho"])
        if m3_single_mean:
            agg["barycenter_M3_cross_family"] = {
                "median_single_mean_acc": float(np.median(m3_single_mean)),
                "median_single_best_acc": float(np.median(m3_single_best)),
                "median_bary_no_norm_acc": float(np.median(m3_no_norm)),
                "median_bary_unif_rms_acc": float(np.median(m3_unif_rms)),
                "median_bary_gls_rms_acc": float(np.median(m3_gls_rms)),
                "median_bary_monge_rms_acc": float(np.median(m3_monge)),
                "median_fig6_gpa_bary_acc": float(np.median(m3_gpa)),
                "median_cross_family_rho_err": float(np.median(m3_rhos)) if m3_rhos else None,
            }

    # 2. Ceiling breakers aggregate across all 20 pairs & 5 self-swaps
    if "ceiling_breakers" in results and results["ceiling_breakers"]:
        self_by_reg = defaultdict(list)
        pair_by_reg = defaultdict(list)
        pair_self_by_reg = defaultdict(list)
        for rec, d in results["ceiling_breakers"].items():
            for rname, rv in d["self_swap"]["regimes"].items():
                self_by_reg[rname].append(rv["test_acc"])
            for pkey, pv in d["pairs"].items():
                for rname, rv in pv["regimes"].items():
                    pair_by_reg[rname].append(rv["test_acc"])
                    if "self_acc" in rv:
                        pair_self_by_reg[rname].append(rv["self_acc"])
        agg["ceiling_breakers_median_self"] = {k: float(np.median(v)) for k, v in self_by_reg.items()}
        agg["ceiling_breakers_median_cross_model"] = {k: float(np.median(v)) for k, v in pair_by_reg.items()}
        agg["ceiling_breakers_median_matched_self"] = {k: float(np.median(v)) for k, v in pair_self_by_reg.items()}

    # 3. Dark memory audit aggregate (pooled val_plus_test & test_only + weaker->stronger slice!)
    if "dark_memory" in results and results["dark_memory"]:
        for scope in ("test_only", "val_plus_test"):
            pooled = defaultdict(lambda: {"dark": [], "nd": [], "ret": [], "ceil": []})
            weaker = defaultdict(lambda: {"dark": [], "ret": []})
            stronger = defaultdict(lambda: {"dark": [], "ret": []})
            multi_bp, multi_bm, multi_sb, multi_ret_bp, multi_ret_bm, multi_ret_sb = [], [], [], [], [], []
            for rec, d in results["dark_memory"].items():
                md = d["multi_donor"][scope]
                multi_bp.extend(md["bary_procrustes_dark_outcomes"])
                multi_bm.extend(md["bary_monge_dark_outcomes"])
                multi_sb.extend(md["best_single_donor_dark_outcomes"])
                multi_ret_bp.extend(md["bary_procrustes_retention_outcomes"])
                multi_ret_bm.extend(md["bary_monge_retention_outcomes"])
                multi_ret_sb.extend(md["best_single_donor_retention_outcomes"])

                for pkey, pv in d["pairs"].items():
                    sc = pv["scopes"][scope]
                    is_w2s = pv["weaker_donor_to_stronger_recipient"]
                    for mname, mv in sc["methods"].items():
                        pooled[mname]["dark"].extend(mv["dark_outcomes"])
                        pooled[mname]["nd"].extend(mv["nondark_outcomes"])
                        pooled[mname]["ret"].extend(mv["retained_outcomes"])
                        pooled[mname]["ceil"].extend(sc["same_model_ceiling_outcomes"])
                        bucket = weaker if is_w2s else stronger
                        bucket[mname]["dark"].extend(mv["dark_outcomes"])
                        bucket[mname]["ret"].extend(mv["retained_outcomes"])

            agg[f"dark_memory_{scope}"] = {
                "single_donor_pooled": {
                    mname: systems.summarize_dark_memory_outcomes(
                        v["dark"], v["nd"], v["ret"], same_model_ceiling=v["ceil"]
                    )
                    for mname, v in pooled.items()
                },
                "weaker_to_stronger_pooled": {
                    mname: {
                        "n_dark": len(v["dark"]),
                        "dark_recovery_rate": float(np.mean(v["dark"])) if v["dark"] else 0.0,
                        "retention_rate": float(np.mean(v["ret"])) if v["ret"] else 0.0,
                    }
                    for mname, v in weaker.items()
                },
                "stronger_to_weaker_pooled": {
                    mname: {
                        "n_dark": len(v["dark"]),
                        "dark_recovery_rate": float(np.mean(v["dark"])) if v["dark"] else 0.0,
                        "retention_rate": float(np.mean(v["ret"])) if v["ret"] else 0.0,
                    }
                    for mname, v in stronger.items()
                },
                "multi_donor_pooled": {
                    "n_dark_ge2": len(multi_bp),
                    "best_single_donor_dark_rate": float(np.mean(multi_sb)) if multi_sb else 0.0,
                    "bary_procrustes_dark_rate": float(np.mean(multi_bp)) if multi_bp else 0.0,
                    "bary_monge_dark_rate": float(np.mean(multi_bm)) if multi_bm else 0.0,
                    "n_recalled_all": len(multi_ret_bp),
                    "best_single_donor_retention": float(np.mean(multi_ret_sb)) if multi_ret_sb else 0.0,
                    "bary_procrustes_retention": float(np.mean(multi_ret_bp)) if multi_ret_bp else 0.0,
                    "bary_monge_retention": float(np.mean(multi_ret_bm)) if multi_ret_bm else 0.0,
                },
            }

    # 4. 2-Hop aggregate
    if "two_hop_bridge" in results and results["two_hop_bridge"]:
        w1_accs, w3_accs, self_accs, rand_accs, b1h_accs, bxr_accs = [], [], [], [], [], []
        cd_2h, cd_1h, cd_rand = [], [], []
        for rec, d in results["two_hop_bridge"].items():
            for pkey, pv in d["pairs"].items():
                e5a = pv["exp5a_2hop_to_2hop"]
                w1_accs.append(e5a["w1_acc"])
                w3_accs.append(e5a["w3_acc"])
                self_accs.append(e5a["w1_self_acc"])
                rand_accs.append(e5a["w1_rand_acc"])
                curve = pv["exp5b_1hop_entity_to_2hop_bridge"]
                b1h_accs.append(max(v["hop2_1hop_to_2hop_subj_acc"] for v in curve.values()))
                bxr_accs.append(max(v.get("cross_rel_1hop_to_2hop_subj_acc", 0.0) for v in curve.values()))
                e5c = pv["exp5c_compositional_dark_rescue"]
                cd_2h.extend(e5c["via_donor_2hop_outcomes"])
                cd_1h.extend(e5c["via_donor_1hop_bridge_outcomes"])
                cd_rand.extend(e5c["random_control_outcomes"])
        if w1_accs:
            agg["two_hop_bridge"] = {
                "n_pairs": len(w1_accs),
                "median_2hop_to_2hop_w1": float(np.median(w1_accs)),
                "median_2hop_to_2hop_w3": float(np.median(w3_accs)),
                "median_2hop_self_w1": float(np.median(self_accs)),
                "median_2hop_rand_w1": float(np.median(rand_accs)),
                "median_1hop_to_2hop_bridge_best_layer": float(np.median(b1h_accs)),
                "median_cross_rel_1hop_to_2hop_best_layer": float(np.median(bxr_accs)),
                "pooled_comp_dark_n": len(cd_2h),
                "pooled_comp_dark_via_2hop": float(np.mean(cd_2h)) if cd_2h else 0.0,
                "pooled_comp_dark_via_1hop_bridge": float(np.mean(cd_1h)) if cd_1h else 0.0,
                "pooled_comp_dark_rand": float(np.mean(cd_rand)) if cd_rand else 0.0,
            }

    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sections",
        nargs="+",
        default=["all"],
        choices=["all", "barycenter", "gpa", "ceiling", "dark", "2hop"],
    )
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    t_global = time.time()
    sec_set = set(args.sections)
    if "all" in sec_set:
        sec_set = {"barycenter", "gpa", "ceiling", "dark", "2hop"}

    with open(CACHE_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    model_keys = sorted(manifest["models"].keys())
    if args.models:
        wanted = set(args.models)
        model_keys = [m for m in model_keys if m in wanted]
    print(f"[systems-dark] Running sections={sorted(sec_set)} across models={model_keys}")

    cache = load_all_caches(model_keys)

    out_path = Path(args.output)
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {}

    results["meta"] = {
        "models": model_keys,
        "families": {m: get_family(m) for m in model_keys},
        "seed": SEED,
        "sections_run": sorted(sec_set),
    }

    # Pre-compute GPA geometry & closed-loop holonomy (fast CPU linear algebra)
    gpa_geom = None
    if "barycenter" in sec_set or "gpa" in sec_set:
        t0 = time.time()
        print("\n=== [Pillar 1B] Computing GPA Platonic Space & Closed-Loop Holonomy ===")
        gpa_geom = run_holonomy_and_gpa_geometry(cache, model_keys)
        results["gpa_and_holonomy"] = {k: v for k, v in gpa_geom.items() if k != "_gpa_spaces"}
        print(f"  Done in {time.time()-t0:.1f}s | Holonomy aggregate:", json.dumps(gpa_geom["holonomy_aggregate"], indent=2))

    # Pre-extract cross-relation 1-hop bridge keys once per model if running 2hop!
    cross_rel_bridge_cache: Dict[str, Dict[int, np.ndarray]] = {}
    if "2hop" in sec_set:
        print("\n=== [Pillar 5 Prep] Extracting cross-relation 1-hop bridge entity keys (country_continent) ===")
        items_2 = cache["dataset_2hop"]["items"]
        for mk in model_keys:
            L_m = int(cache["model_infos"][mk]["num_layers"])
            # Extract all early/mid layers in one single forward pass (0 extra cost!)
            cand_ls = list(range(0, min(L_m, int(0.70 * L_m) + 2)))
            model_m, tok_m, dev_m, _ = load_model_and_tokenizer(Path(cache["model_infos"][mk]["local_path"]))
            cross_rel_bridge_cache[mk] = extract_cross_relation_bridge_keys(model_m, tok_m, items_2, cand_ls, dev_m)
            del model_m
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    results.setdefault("barycenter", {})
    results.setdefault("gpa_pairwise_transfer", {})
    results.setdefault("low_overlap_gpa_sync", {})
    results.setdefault("ceiling_breakers", {})
    results.setdefault("dark_memory", {})
    results.setdefault("two_hop_bridge", {})

    # Group all model forward passes by recipient mk_B so each LLM is loaded ONLY ONCE!
    for mk_B in model_keys:
        t_rec = time.time()
        print(f"\n================ RECIPIENT: {mk_B} ================")
        model_B, tok_B, device, dtype = load_model_and_tokenizer(Path(cache["model_infos"][mk_B]["local_path"]))

        if ("barycenter" in sec_set or "gpa" in sec_set) and gpa_geom is not None:
            bary_r, gpa_tr, lo_r = eval_recipient_barycenter_and_gpa(
                mk_B, model_B, tok_B, device, dtype, cache, model_keys, gpa_geom
            )
            results["barycenter"][mk_B] = bary_r
            results["gpa_pairwise_transfer"].update(gpa_tr)
            results["low_overlap_gpa_sync"].update(lo_r)

        if "ceiling" in sec_set:
            results["ceiling_breakers"][mk_B] = eval_recipient_ceiling_breakers(
                mk_B, model_B, tok_B, device, dtype, cache, model_keys
            )

        if "dark" in sec_set:
            results["dark_memory"][mk_B] = eval_recipient_dark_memory_audit(
                mk_B, model_B, tok_B, device, dtype, cache, model_keys
            )

        if "2hop" in sec_set:
            results["two_hop_bridge"][mk_B] = eval_recipient_2hop_bridge(
                mk_B, model_B, tok_B, device, dtype, cache, model_keys, cross_rel_bridge_cache
            )

        del model_B
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Incremental checkpoint save after each recipient!
        results["aggregate"] = build_aggregate_summary(results)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_json_clean(results), f, indent=2)
        print(f"=== Checkpoint saved for {mk_B} in {time.time() - t_rec:.1f}s -> {out_path} ===")

    print(f"\n[systems-dark] ALL DONE in {time.time() - t_global:.1f}s!")
    print("[systems-dark] Aggregate Summary:\n", json.dumps(_json_clean(results["aggregate"]), indent=2))


if __name__ == "__main__":
    main()
