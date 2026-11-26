#!/usr/bin/env python3
"""
v2_step0_rw_diagnostic.py — Per-Item Gate Telemetry & 2×2 Native/Transferred Read×Write Decomposition
=====================================================================================================

Strictly READ-ONLY on frozen v1.0 engine (`platonic_organ_engine.py`, SHA-256 `22ef57c9...`)
and single-execution Cap-2 suite (`bio_ripple_sealed.py`, SHA-256 `d321fd02...`).

Resolves the exact Round 2 Council attribution gap on all 4 recipients (`smollm2`, `granite`,
`falcon3`, `qwen1.5b`) using cached v1.0 sockets (zero donor forward passes):
  1. Logs exact per-item gate pre-activation `z = x_B · w_c`, BOS-nulled pin `u = x_B · c_u`,
     and realized twin-SwiGLU write amplitude `phi` at the subject token (and max non-BOS token)
     across all 48 Cap-2 test species (direct + 4-attr ripple) and 24 hard negatives.
  2. Measures actual `cos(v_B_push, v_B_native)` and `||v_B_push|| / ||v_B_native||` for both rules.
  3. Evaluates the complete 2×2 Read×Write factorial + forced gold-position controls:
       - PUSH_READ_x_PUSH_WRITE     (exact v1.0 organ)
       - NATIVE_READ_x_PUSH_WRITE   (native B gate driving pushed-forward write)
       - PUSH_READ_x_NATIVE_WRITE   (pushed-forward gate driving native B write)
       - NATIVE_READ_x_NATIVE_WRITE (exact v1.0 NATIVE_B_ORACLE)
       - FORCED_GOLD_PUSH_WRITE     (forced phi=1.0 at species token, pushed write)
       - FORCED_GOLD_NATIVE_WRITE   (forced phi=1.0 at species token, native write)
  4. Scores BOTH first-token FC AND full-continuation FC (fixing BPE single-token artifacts such
     as Granite `" re"` vs `" m"`), plus EM and saved greedy generations, per-attribute accuracy,
     causal miss taxonomy (`gate_miss`, `ramp_underdrive`, `write_limited`, `verbalization_gap`),
     and exact single-direction empirical ROC (test species TPR vs. HardNeg FPR).

Usage:
  PYTHONPATH=platonicswap:experiments python experiments/v2_step0_rw_diagnostic.py --unit-test
  PYTHONPATH=platonicswap:experiments python experiments/v2_step0_rw_diagnostic.py --targets smollm2,granite,falcon3,qwen1.5b
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for p in (str(REPO_ROOT / "platonicswap"), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Default to populated local lab caches if present and not overridden
if "PLATONICSWAP_HF_CACHE" not in os.environ and Path("/tmp/platos_key_lab/hf_cache").is_dir():
    os.environ["PLATONICSWAP_HF_CACHE"] = "/tmp/platos_key_lab/hf_cache"
if "PLATONICSWAP_CACHE_DIR" not in os.environ and Path("/tmp/platos_key_lab/cache").is_dir():
    os.environ["PLATONICSWAP_CACHE_DIR"] = "/tmp/platos_key_lab/cache"

RESULTS_DIR = REPO_ROOT / "results" if (REPO_ROOT / "results").is_dir() else SCRIPT_DIR
OUT_PATH = RESULTS_DIR / "results_v2_step0_rw_diagnostic.json"

EXPECTED_ENGINE_SHA = "22ef57c9d196e12f2cd3c9ecc79ec0813803df43d5a6009f5422904948802306"
EXPECTED_SEALED_SHA = "d321fd0255c47cd3ab0113bb966acdf248296ca12da0796c0e21dfdb160bdf6d"


def _silu_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    # Numerically stable sigmoid
    sig = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
    return x * sig


def compute_twin_swiglu_phi(
    x: np.ndarray,
    W_g_2: np.ndarray,
    W_u_2: np.ndarray,
    beta_gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given token states x (N, D) and the 2 bias-free rows (W_g_2, W_u_2) for one rule,
    returns (u_pin, phi) where:
      y_organ(x) = phi * v_B
    """
    g1 = x @ W_g_2[0]
    g2 = x @ W_g_2[1]
    u1 = x @ W_u_2[0]
    u2 = x @ W_u_2[1]
    phi = (_silu_np(g1) * u1 - _silu_np(g2) * u2) / float(beta_gamma)
    return u1, phi


def build_hybrid_organ_weights(
    read_compiled: Dict[str, Any],
    write_compiled: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Combine the gate rows (W_g, W_u) from `read_compiled` with the write vectors v_B
    from `write_compiled`, rescaling W_d by the reader's (beta * Gamma) so that
    a saturated reader (phi = 1.0) emits the exact write vector of `write_compiled`.
    """
    wr = read_compiled["weights"]
    ww = write_compiled["weights"]
    n_rules = len(read_compiled["rules"])
    assert n_rules == len(write_compiled["rules"])
    W_d_new = np.zeros_like(wr["W_d"])
    for i in range(n_rules):
        bg_r = float(read_compiled["rules"][i]["beta"]) * float(read_compiled["rules"][i]["Gamma"])
        bg_w = float(write_compiled["rules"][i]["beta"]) * float(write_compiled["rules"][i]["Gamma"])
        v_B_w = ww["W_d"][:, 2 * i] * bg_w
        W_d_new[:, 2 * i] = v_B_w / bg_r
        W_d_new[:, 2 * i + 1] = -v_B_w / bg_r
    return {
        "W_g": wr["W_g"].copy(),
        "b_g": None if wr["b_g"] is None else wr["b_g"].copy(),
        "W_u": wr["W_u"].copy(),
        "b_u": None if wr["b_u"] is None else wr["b_u"].copy(),
        "W_d": W_d_new,
        "use_bias": wr["use_bias"],
    }


def compute_binary_roc(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> Dict[str, Any]:
    """Exact AUROC and TPR at specific HardNeg FPR caps (0%, <=5%, <=10%)."""
    p = np.asarray(pos_scores, dtype=np.float64)
    n = np.asarray(neg_scores, dtype=np.float64)
    if len(p) == 0 or len(n) == 0:
        return {"auroc": None, "tpr_at_0pct_fpr": None, "tpr_at_5pct_fpr": None, "tpr_at_10pct_fpr": None}
    # Mann-Whitney U
    wins = float(np.sum(p[:, None] > n[None, :]) + 0.5 * np.sum(p[:, None] == n[None, :]))
    auroc = wins / (len(p) * len(n))

    def _tpr_at_fpr_cap(cap: float) -> Tuple[float, float, float]:
        # Threshold strictly above top k allowed negatives
        max_neg_allowed = int(math.floor(cap * len(n) + 1e-12))
        n_sorted = np.sort(n)
        if max_neg_allowed == 0:
            thr = float(n_sorted[-1]) + 1e-12
        elif max_neg_allowed >= len(n):
            thr = -1e18
        else:
            thr = float(n_sorted[-(max_neg_allowed + 1)]) + 1e-12
        realized_fpr = float(np.mean(n >= thr))
        realized_tpr = float(np.mean(p >= thr))
        return round(realized_tpr, 4), round(realized_fpr, 4), round(thr, 5)

    t0, f0, thr0 = _tpr_at_fpr_cap(0.0)
    t5, f5, thr5 = _tpr_at_fpr_cap(0.05)
    t10, f10, thr10 = _tpr_at_fpr_cap(0.10)
    return {
        "auroc": round(auroc, 4),
        "tpr_at_0pct_fpr": t0, "fpr_0": f0, "thr_0": thr0,
        "tpr_at_5pct_fpr": t5, "fpr_5": f5, "thr_5": thr5,
        "tpr_at_10pct_fpr": t10, "fpr_10": f10, "thr_10": thr10,
        "pos_mean": round(float(np.mean(p)), 4),
        "pos_min": round(float(np.min(p)), 4),
        "neg_mean": round(float(np.mean(n)), 4),
        "neg_max": round(float(np.max(n)), 4),
    }


def locate_species_last_token(tok: Any, prompt: str, species: str) -> int:
    """Locate the index of the last sub-token overlapping the first mention of `species` (case-insensitive)."""
    from extract_and_baseline import locate_last_subject_token
    idx_c = prompt.lower().find(species.lower())
    assert idx_c >= 0, f"Species {species!r} not found in prompt {prompt!r}"
    tok_idx, _ = locate_last_subject_token(tok, prompt, (idx_c, idx_c + len(species)))
    return tok_idx


@torch.no_grad()
def extract_item_x_mid_and_max_phi(
    model: Any,
    tok: Any,
    device: torch.device,
    layer_idx: int,
    items: List[Dict[str, Any]],
    compiled_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Single forward pass per item (unbatched or right-padded to exact unpadded indices) capturing
    post_attention_layernorm(h_mid) at the species token AND across all non-BOS tokens, then
    evaluating exact (z, u_pin, phi) for every compiled organ in `compiled_map`.
    """
    from extract_and_baseline import get_decoder_layers

    layer = get_decoder_layers(model)[layer_idx]
    captured: List[torch.Tensor] = []

    def _hook(mod, inp, out):
        captured.append(out[0].detach().float().cpu())

    h = layer.post_attention_layernorm.register_forward_hook(_hook)
    records: List[Dict[str, Any]] = []
    try:
        for it in items:
            prompt = it["prompt"]
            species = it["species"]
            subj_idx = locate_species_last_token(tok, prompt, species)
            enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
            input_ids = enc["input_ids"].to(device)
            captured.clear()
            model(input_ids=input_ids, use_cache=False)
            X_seq = captured[0].numpy()  # (T, D)
            T = X_seq.shape[0]
            x_subj = X_seq[subj_idx : subj_idx + 1]  # (1, D)
            x_nonbos = X_seq[1:] if T > 1 else X_seq

            rec: Dict[str, Any] = {
                "species": species,
                "rule": it.get("rule", it.get("sub_group", "")),
                "attr": it.get("attr", "direct"),
                "subj_tok_idx": int(subj_idx),
                "seq_len": int(T),
                "gates": {},
            }
            for cname, comp in compiled_map.items():
                W_g = comp["weights"]["W_g"]
                W_u = comp["weights"]["W_u"]
                g_rec = {}
                for ri, rule in enumerate(comp["rules"]):
                    rname = rule["name"]
                    w_c = rule["w_c"]
                    bg = float(rule["beta"]) * float(rule["Gamma"])
                    z_subj = float((x_subj @ w_c)[0])
                    u_subj, phi_subj = compute_twin_swiglu_phi(x_subj, W_g[2 * ri : 2 * ri + 2], W_u[2 * ri : 2 * ri + 2], bg)
                    z_all = x_nonbos @ w_c
                    _, phi_all = compute_twin_swiglu_phi(x_nonbos, W_g[2 * ri : 2 * ri + 2], W_u[2 * ri : 2 * ri + 2], bg)
                    g_rec[rname] = {
                        "z_subj": round(z_subj, 5),
                        "u_subj": round(float(u_subj[0]), 5),
                        "phi_subj": round(float(phi_subj[0]), 5),
                        "z_max_nonbos": round(float(np.max(z_all)), 5),
                        "phi_max_nonbos": round(float(np.max(phi_all)), 5),
                        "t1": float(rule["t1"]),
                        "t2": float(rule["t2"]),
                    }
                rec["gates"][cname] = g_rec
            records.append(rec)
    finally:
        h.remove()
    return records


@torch.no_grad()
def _full_continuation_logp(
    model: Any,
    tok: Any,
    device: torch.device,
    prompt_ids: List[int],
    cand_text: str,
) -> float:
    """Exact sum of token log-probabilities for the full multi-token continuation `cand_text`."""
    cand_ids = tok.encode(cand_text, add_special_tokens=False)
    if not cand_ids:
        return -1e9
    full = torch.tensor([prompt_ids + cand_ids], dtype=torch.long, device=device)
    logits = model(input_ids=full, use_cache=False).logits[0].float()  # (T_p + T_c, V)
    p_len = len(prompt_ids)
    total = 0.0
    for k, tid in enumerate(cand_ids):
        lp = F.log_softmax(logits[p_len - 1 + k], dim=-1)[tid].item()
        total += float(lp)
    return total


@torch.no_grad()
def eval_items_full_telemetry(
    model: Any,
    tok: Any,
    device: torch.device,
    items: List[Dict[str, Any]],
    is_hn: bool = False,
    compute_full_cont: bool = True,
    batch_size: int = 24,
) -> Dict[str, Any]:
    """
    Evaluate items returning per-item:
      - first-token FC (`fc1_new`, `fc1_log_odds`)
      - full-continuation FC (`fc_full_new`, `fc_full_log_odds`) — removes Granite BPE first-token bias!
      - greedy EM (`em_new`, `em_old`, `gen`)
    """
    import pilot_organs_p0 as po
    from bio_ripple_sealed import _first_tok_max_logp
    from extract_and_baseline import check_answer_match

    prompts = [x["prompt"] for x in items]
    gens = po.greedy_generate_batch(model, tok, device, prompts, max_new_tokens=8, batch_size=batch_size)

    fc1_new, fc1_lo = [], []
    em_new, em_old = [], []

    old_pad = tok.padding_side
    tok.padding_side = "left"
    try:
        for st in range(0, len(prompts), batch_size):
            bp = prompts[st : st + batch_size]
            enc = tok(bp, return_tensors="pt", padding=True, add_special_tokens=True)
            out = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device), use_cache=False)
            lp = F.log_softmax(out.logits[:, -1, :].float(), dim=-1).cpu()
            for bi in range(len(bp)):
                it = items[st + bi]
                g = gens[st + bi]
                if is_hn:
                    lp_o = _first_tok_max_logp(lp[bi], tok, it["orig_fc"])
                    lp_f = _first_tok_max_logp(lp[bi], tok, it["forbid_fc"])
                    fc1_new.append(int(lp_f > lp_o))
                    fc1_lo.append(round(lp_f - lp_o, 4))
                    em_new.append(int(check_answer_match(g, it["forbid_flip"])))
                    em_old.append(int(check_answer_match(g, it["orig_aliases"])))
                else:
                    lp_n = _first_tok_max_logp(lp[bi], tok, it["new_fc"])
                    lp_o = _first_tok_max_logp(lp[bi], tok, it["old_fc"])
                    fc1_new.append(int(lp_n > lp_o))
                    fc1_lo.append(round(lp_n - lp_o, 4))
                    em_new.append(int(check_answer_match(g, it["new_aliases"])))
                    em_old.append(int(check_answer_match(g, it["old_aliases"])))
    finally:
        tok.padding_side = old_pad

    # Full-continuation FC on Direct + HardNeg (and any item where a candidate has >1 sub-token)
    fc_full_new, fc_full_lo = list(fc1_new), list(fc1_lo)
    if compute_full_cont:
        for i, it in enumerate(items):
            pos_cands = it["forbid_fc"] if is_hn else it["new_fc"]
            neg_cands = it["orig_fc"] if is_hn else it["old_fc"]
            needs_multi = any(len(tok.encode(c, add_special_tokens=False)) > 1 for c in list(pos_cands) + list(neg_cands))
            if not needs_multi:
                continue
            p_ids = tok.encode(it["prompt"], add_special_tokens=True)
            lp_pos = max(_full_continuation_logp(model, tok, device, p_ids, c) for c in pos_cands)
            lp_neg = max(_full_continuation_logp(model, tok, device, p_ids, c) for c in neg_cands)
            fc_full_new[i] = int(lp_pos > lp_neg)
            fc_full_lo[i] = round(lp_pos - lp_neg, 4)

    return {
        "fc1_new": fc1_new,
        "fc1_log_odds": fc1_lo,
        "fc_full_new": fc_full_new,
        "fc_full_log_odds": fc_full_lo,
        "em_new": em_new,
        "em_old": em_old,
        "gens": gens,
    }


@torch.no_grad()
def eval_forced_gold_write(
    model: Any,
    tok: Any,
    device: torch.device,
    layer_idx: int,
    rm_B: float,
    items: List[Dict[str, Any]],
    rule_v_map: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    Forced-fire gold-position write control (`phi == 1.0` at the species last sub-token, 0 elsewhere):
    adds `rm_B * v_B_rule` directly to `mlp` output at `subj_tok_idx` during prefill and greedy gen.
    Isolates pure write potency with 0% gate misses and 0% under-drive.
    """
    from bio_ripple_sealed import _first_tok_max_logp
    from extract_and_baseline import check_answer_match, get_decoder_layers

    mlp = get_decoder_layers(model)[layer_idx].mlp
    fc1_new, fc1_lo, fc_full_new, fc_full_lo, em_new, gens = [], [], [], [], [], []

    for it in items:
        v_np = rule_v_map[it["rule"]]
        v_t = torch.tensor(v_np, dtype=torch.float32, device=device)  # MLP output space (pre-rm_B, since model scales by rm_B!)
        subj_idx = locate_species_last_token(tok, it["prompt"], it["species"])

        def _make_hook(s_idx: int, vec: torch.Tensor):
            def _hook(mod, inp, out):
                if out.shape[1] > s_idx:
                    out = out.clone()
                    out[:, s_idx, :] = out[:, s_idx, :] + vec.to(out.dtype)
                return out
            return _hook

        h = mlp.register_forward_hook(_make_hook(subj_idx, v_t))
        try:
            p_ids = tok.encode(it["prompt"], add_special_tokens=True)
            inp = torch.tensor([p_ids], dtype=torch.long, device=device)
            # First-token FC
            logits = model(input_ids=inp, use_cache=False).logits[0, -1, :].float().cpu()
            lp = F.log_softmax(logits, dim=-1)
            lp_n = _first_tok_max_logp(lp, tok, it["new_fc"])
            lp_o = _first_tok_max_logp(lp, tok, it["old_fc"])
            fc1_new.append(int(lp_n > lp_o))
            fc1_lo.append(round(lp_n - lp_o, 4))

            # Full-continuation FC
            needs_multi = any(len(tok.encode(c, add_special_tokens=False)) > 1 for c in list(it["new_fc"]) + list(it["old_fc"]))
            if needs_multi:
                lp_nf = max(_full_continuation_logp(model, tok, device, p_ids, c) for c in it["new_fc"])
                lp_of = max(_full_continuation_logp(model, tok, device, p_ids, c) for c in it["old_fc"])
                fc_full_new.append(int(lp_nf > lp_of))
                fc_full_lo.append(round(lp_nf - lp_of, 4))
            else:
                fc_full_new.append(fc1_new[-1])
                fc_full_lo.append(fc1_lo[-1])

            # Greedy 8-token generation (use_cache=False so hook applies cleanly at subj_idx every step!)
            cur = inp
            gen_ids = []
            for _ in range(8):
                nxt = int(model(input_ids=cur, use_cache=False).logits[0, -1, :].argmax().item())
                gen_ids.append(nxt)
                cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
            g = tok.decode(gen_ids, skip_special_tokens=True)
            gens.append(g)
            em_new.append(int(check_answer_match(g, it["new_aliases"])))
        finally:
            h.remove()

    return {
        "fc1_new": fc1_new,
        "fc1_log_odds": fc1_lo,
        "fc_full_new": fc_full_new,
        "fc_full_log_odds": fc_full_lo,
        "em_new": em_new,
        "gens": gens,
    }


def classify_failure_taxonomy(
    items: List[Dict[str, Any]],
    gate_recs: List[Dict[str, Any]],
    gate_key: str,
    eval_res: Dict[str, Any],
    use_full_fc: bool = True,
) -> Dict[str, Any]:
    """
    Classify each item into 5 mutually exclusive buckets using actual subject-token gate phi:
      - gate_miss:          phi < 0.05  (and FC fails)
      - ramp_underdrive:    0.05 <= phi < 0.90 (and FC fails)
      - write_limited:      phi >= 0.90 (and FC fails)
      - verbalization_gap:  FC succeeds, EM fails
      - success_em:         EM succeeds
    Also reports `phi` mean/quantiles and per-rule & per-attribute accuracy.
    """
    fc_vec = eval_res["fc_full_new"] if use_full_fc else eval_res["fc1_new"]
    em_vec = eval_res["em_new"]
    buckets = {"gate_miss": 0, "ramp_underdrive": 0, "write_limited": 0, "verbalization_gap": 0, "success_em": 0}
    by_rule: Dict[str, Dict[str, Any]] = {}
    by_attr: Dict[str, Dict[str, Any]] = {}
    phis_subj, phis_max = [], []

    for i, it in enumerate(items):
        rname = it["rule"]
        attr = it.get("attr", "direct")
        g = gate_recs[i]["gates"][gate_key][rname]
        phi_s = float(g["phi_subj"])
        phi_m = float(g["phi_max_nonbos"])
        phis_subj.append(phi_s)
        phis_max.append(phi_m)
        fc_ok = int(fc_vec[i])
        em_ok = int(em_vec[i])

        if em_ok:
            b = "success_em"
        elif fc_ok:
            b = "verbalization_gap"
        elif phi_s < 0.05:
            b = "gate_miss"
        elif phi_s < 0.90:
            b = "ramp_underdrive"
        else:
            b = "write_limited"
        buckets[b] += 1

        for grp_dict, gkey in ((by_rule, rname), (by_attr, f"{rname}::{attr}")):
            if gkey not in grp_dict:
                grp_dict[gkey] = {"n": 0, "fc1": 0, "fc_full": 0, "em": 0, "phis": [], "gate_miss": 0, "underdrive": 0, "write_lim": 0, "verb_gap": 0}
            d = grp_dict[gkey]
            d["n"] += 1
            d["fc1"] += int(eval_res["fc1_new"][i])
            d["fc_full"] += int(eval_res["fc_full_new"][i])
            d["em"] += em_ok
            d["phis"].append(phi_s)
            if b == "gate_miss":
                d["gate_miss"] += 1
            elif b == "ramp_underdrive":
                d["underdrive"] += 1
            elif b == "write_limited":
                d["write_lim"] += 1
            elif b == "verbalization_gap":
                d["verb_gap"] += 1

    def _summarize_group(gd: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        out = {}
        for k, d in gd.items():
            n = d["n"]
            out[k] = {
                "n": n,
                "fc1_acc": round(d["fc1"] / n, 4),
                "fc_full_acc": round(d["fc_full"] / n, 4),
                "em_acc": round(d["em"] / n, 4),
                "mean_phi_subj": round(float(np.mean(d["phis"])), 4),
                "frac_phi_ge_09": round(float(np.mean(np.array(d["phis"]) >= 0.90)), 4),
                "frac_phi_lt_005": round(float(np.mean(np.array(d["phis"]) < 0.05)), 4),
                "gate_miss": d["gate_miss"],
                "underdrive": d["underdrive"],
                "write_lim": d["write_lim"],
                "verb_gap": d["verb_gap"],
            }
        return out

    n_tot = len(items)
    return {
        "n": n_tot,
        "fc1_acc": round(float(np.mean(eval_res["fc1_new"])), 4),
        "fc_full_acc": round(float(np.mean(eval_res["fc_full_new"])), 4),
        "em_acc": round(float(np.mean(em_vec)), 4),
        "mean_fc1_log_odds": round(float(np.mean(eval_res["fc1_log_odds"])), 4),
        "mean_fc_full_log_odds": round(float(np.mean(eval_res["fc_full_log_odds"])), 4),
        "mean_phi_subj": round(float(np.mean(phis_subj)), 4),
        "median_phi_subj": round(float(np.median(phis_subj)), 4),
        "frac_phi_subj_ge_09": round(float(np.mean(np.array(phis_subj) >= 0.90)), 4),
        "frac_phi_subj_lt_005": round(float(np.mean(np.array(phis_subj) < 0.05)), 4),
        "mean_phi_max_nonbos": round(float(np.mean(phis_max)), 4),
        "buckets": buckets,
        "by_rule": _summarize_group(by_rule),
        "by_attr": _summarize_group(by_attr),
    }


def run_unit_test() -> None:
    import bio_ripple_sealed as brs
    import platonic_organ_engine as poe

    assert poe.get_engine_sha256() == EXPECTED_ENGINE_SHA, "Engine SHA-256 mismatch!"
    assert brs.compute_sealed_sha256() == EXPECTED_SEALED_SHA, "Sealed Cap-2 SHA-256 mismatch!"

    # Verify hybrid organ weight algebra + twin-SwiGLU phi exactness on synthetic data
    rng = np.random.default_rng(7)
    D = 32
    w_c_r = rng.standard_normal(D); w_c_r /= np.linalg.norm(w_c_r)
    c_B = np.zeros(D); c_B[0] = 1.0
    t1_r, t2_r, bg_r = 1.0, 2.0, 14.0
    beta_r = bg_r / (t2_r - t1_r)
    v_push = rng.standard_normal(D)
    v_nat = rng.standard_normal(D)

    def _mk_comp(beta, Gamma, t1, t2, v):
        W_g = np.stack([beta * (w_c_r - t1 * c_B), beta * (w_c_r - t2 * c_B)], axis=0)
        W_u = np.stack([c_B, c_B], axis=0)
        W_d = np.stack([v / (beta * Gamma), -v / (beta * Gamma)], axis=1)
        return {
            "rules": [{"name": "r0", "beta": beta, "Gamma": Gamma, "t1": t1, "t2": t2, "w_c": w_c_r}],
            "weights": {"W_g": W_g, "b_g": None, "W_u": W_u, "b_u": None, "W_d": W_d, "use_bias": False},
        }

    c_r = _mk_comp(beta_r, 1.0, t1_r, t2_r, v_nat)
    c_w = _mk_comp(20.0, 0.7, 0.5, 1.2, v_push)
    hyb = build_hybrid_organ_weights(c_r, c_w)
    # Test saturated token (z = 10.0, x[0] = 1.0)
    x_sat = c_B + 10.0 * w_c_r
    x_sat[0] = 1.0
    y = hyb["W_d"] @ (_silu_np(hyb["W_g"] @ x_sat) * (hyb["W_u"] @ x_sat))
    err = float(np.linalg.norm(y - v_push) / np.linalg.norm(v_push))
    assert err < 1e-4, f"Hybrid write mismatch: {err}"

    roc = compute_binary_roc([3.0, 4.0, 5.0, 6.0], [0.0, 1.0, 2.0, 3.5])
    assert abs(roc["auroc"] - 0.9375) < 1e-6 and roc["tpr_at_0pct_fpr"] == 0.75
    print(f"[v2_step0_rw_diagnostic --unit-test] ALL PASSED! (hybrid rel_err={err:.2e}, engine_sha={EXPECTED_ENGINE_SHA[:12]}, sealed_sha={EXPECTED_SEALED_SHA[:12]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit-test", action="store_true")
    ap.add_argument("--targets", type=str, default="smollm2,granite,falcon3,qwen1.5b")
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    args = ap.parse_args()

    if args.unit_test:
        run_unit_test()
        return

    import bio_ripple_sealed as brs
    import pilot_organs_p0 as po
    import platonic_organ_engine as poe
    from extract_and_baseline import CACHE_DIR, get_decoder_layers

    # Provenance lock assertion
    assert poe.get_engine_sha256() == EXPECTED_ENGINE_SHA, "Frozen v1 engine modified!"
    assert brs.compute_sealed_sha256() == EXPECTED_SEALED_SHA, "Sealed Cap-2 modified!"

    t0 = time.time()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    corpus_univ = poe.build_universal_calibration_corpus()
    suite = brs.build_cap2_eval_items(in_context_ceiling=False)

    cap2_rules_cache = CACHE_DIR / f"cap2_cb_rules_{poe.DONOR_QWEN_KEY}_l{poe.DONOR_QWEN_LAYER}.pt"
    assert cap2_rules_cache.exists(), f"Missing cached donor rules: {cap2_rules_cache}"
    cap2_donor_rules = torch.load(cap2_rules_cache, map_location="cpu", weights_only=False)

    out: Dict[str, Any] = {
        "meta": {
            "engine_sha256": EXPECTED_ENGINE_SHA,
            "sealed_cap2_sha256": EXPECTED_SEALED_SHA,
            "targets": targets,
            "note": "Read-only v2 Step 0 diagnostic: per-item gate telemetry, 2x2 ReadxWrite hybrid decomposition, forced-gold write control, full-continuation FC, and empirical single-direction ROC.",
        },
        "recipients": {},
    }

    for tgt in targets:
        tr0 = time.time()
        spec = poe.RECIPIENT_SPECS[tgt]
        rk, lB = spec["model_key"], spec["layer"]
        ch_cache = CACHE_DIR / f"universal_channel_{poe.DONOR_QWEN_KEY}_l{poe.DONOR_QWEN_LAYER}_to_{rk}_l{lB}.pt"
        assert ch_cache.exists(), f"Missing frozen socket: {ch_cache}"

        print(f"\n================ [v2 Step 0 | {tgt}] {rk} L{lB} ================")
        model_B, tok_B, dev_B, rm_B = po.load_model_fp32(rk)
        mlp_B = get_decoder_layers(model_B)[lB].mlp
        orig_state_B = po.save_original_mlp_state(mlp_B)

        ch_univ = poe.fit_or_load_universal_channel(
            poe.DONOR_QWEN_KEY, poe.DONOR_QWEN_LAYER, rk, lB, corpus_univ, model_B, tok_B, dev_B, tag="universal"
        )
        b_cap2_native = brs.extract_cap2_donor_data(model_B, tok_B, dev_B, lB)["rules"]

        # Compile exact v1.0 Pushforward & Native-B Oracle organs
        c_push = poe.compile_channel_pushforward_organ(cap2_donor_rules, ch_univ, rm_B=rm_B, mode="biasfree", force_abstained=False)
        c_nat = poe.compile_channel_pushforward_organ(b_cap2_native, channel=None, rm_B=rm_B, mode="biasfree", native_B_stats=ch_univ, force_abstained=False)
        assert c_push["weights"] is not None and c_nat["weights"] is not None

        # 1. Measure actual write-vector cosine & norm ratio (previously unmeasured!)
        write_diag = {}
        v_push_map, v_nat_map = {}, {}
        for i, r in enumerate(c_push["rules"]):
            rname = r["name"]
            bg_p = float(r["beta"]) * float(r["Gamma"])
            bg_n = float(c_nat["rules"][i]["beta"]) * float(c_nat["rules"][i]["Gamma"])
            vp = c_push["weights"]["W_d"][:, 2 * i] * bg_p
            vn = c_nat["weights"]["W_d"][:, 2 * i] * bg_n
            v_push_map[rname] = vp
            v_nat_map[rname] = vn
            cos_v = float(np.dot(vp, vn) / (np.linalg.norm(vp) * np.linalg.norm(vn) + 1e-12))
            norm_ratio = float(np.linalg.norm(vp) / (np.linalg.norm(vn) + 1e-12))
            cos_w = float(np.dot(r["w_c"], c_nat["rules"][i]["w_c"]) / (np.linalg.norm(r["w_c"]) * np.linalg.norm(c_nat["rules"][i]["w_c"]) + 1e-12))
            write_diag[rname] = {
                "cos_vB_push_vs_native": round(cos_v, 4),
                "norm_ratio_push_over_native": round(norm_ratio, 4),
                "cos_wc_push_vs_native": round(cos_w, 4),
                "vB_push_res_norm": round(float(np.linalg.norm(vp) * rm_B), 4),
                "vB_native_res_norm": round(float(np.linalg.norm(vn) * rm_B), 4),
            }
            print(f"  [{rname:16s}] cos(v_push, v_nat)={cos_v:.4f} | ||v_push||/||v_nat||={norm_ratio:.3f} | cos(w_push, w_nat)={cos_w:.4f}")

        # 2. Extract exact per-item gate pre-activations z & phi on unedited model_B!
        compiled_map = {"PUSH": c_push, "NATIVE": c_nat}
        gate_dir = extract_item_x_mid_and_max_phi(model_B, tok_B, dev_B, lB, suite["direct"], compiled_map)
        gate_rip = extract_item_x_mid_and_max_phi(model_B, tok_B, dev_B, lB, suite["ripple"], compiled_map)
        gate_hn = extract_item_x_mid_and_max_phi(model_B, tok_B, dev_B, lB, suite["hardneg"], compiled_map)

        # Compute exact single-direction empirical ROC (test species direct vs 24 HardNegs)
        roc_summary: Dict[str, Any] = {}
        for gkey in ("PUSH", "NATIVE"):
            roc_summary[gkey] = {}
            for r in c_push["rules"]:
                rname = r["name"]
                pos_z = [rec["gates"][gkey][rname]["z_subj"] for rec in gate_dir if rec["rule"] == rname]
                neg_z = [rec["gates"][gkey][rname]["z_subj"] for rec in gate_hn]
                pos_phi = [rec["gates"][gkey][rname]["phi_subj"] for rec in gate_dir if rec["rule"] == rname]
                neg_phi = [rec["gates"][gkey][rname]["phi_subj"] for rec in gate_hn]
                roc = compute_binary_roc(pos_z, neg_z)
                roc["realized_pos_phi_mean"] = round(float(np.mean(pos_phi)), 4)
                roc["realized_pos_phi_ge_09"] = round(float(np.mean(np.array(pos_phi) >= 0.90)), 4)
                roc["realized_pos_phi_lt_005"] = round(float(np.mean(np.array(pos_phi) < 0.05)), 4)
                roc["realized_hn_phi_ge_05"] = round(float(np.mean(np.array(neg_phi) >= 0.50)), 4)
                roc["realized_hn_phi_ge_005"] = round(float(np.mean(np.array(neg_phi) >= 0.05)), 4)
                roc_summary[gkey][rname] = roc
                print(
                    f"  [GATE {gkey:6s} | {rname:15s}] AUROC(sp vs HN)={roc['auroc']:.4f} | "
                    f"TPR@0%/5%/10%FPR={roc['tpr_at_0pct_fpr']:.2f}/{roc['tpr_at_5pct_fpr']:.2f}/{roc['tpr_at_10pct_fpr']:.2f} | "
                    f"pos phi>=0.9: {roc['realized_pos_phi_ge_09']:.2f} (<0.05: {roc['realized_pos_phi_lt_005']:.2f}) | "
                    f"HN phi>=0.5: {roc['realized_hn_phi_ge_05']:.2f}"
                )

        # 3. Evaluate BASE + 2x2 Hybrid ReadxWrite Arms + Forced-Gold Write Controls!
        arms_weights = {
            "PUSH_READ_x_PUSH_WRITE": ("PUSH", c_push["weights"]),
            "NATIVE_READ_x_PUSH_WRITE": ("NATIVE", build_hybrid_organ_weights(c_nat, c_push)),
            "PUSH_READ_x_NATIVE_WRITE": ("PUSH", build_hybrid_organ_weights(c_push, c_nat)),
            "NATIVE_READ_x_NATIVE_WRITE": ("NATIVE", c_nat["weights"]),
        }

        arm_results: Dict[str, Any] = {}
        # BASE (unedited)
        b_dir = eval_items_full_telemetry(model_B, tok_B, dev_B, suite["direct"], is_hn=False)
        b_rip = eval_items_full_telemetry(model_B, tok_B, dev_B, suite["ripple"], is_hn=False)
        b_hn = eval_items_full_telemetry(model_B, tok_B, dev_B, suite["hardneg"], is_hn=True)
        arm_results["BASE"] = {
            "direct_fc1": round(float(np.mean(b_dir["fc1_new"])), 4),
            "direct_fc_full": round(float(np.mean(b_dir["fc_full_new"])), 4),
            "direct_em": round(float(np.mean(b_dir["em_new"])), 4),
            "ripple_fc1": round(float(np.mean(b_rip["fc1_new"])), 4),
            "ripple_fc_full": round(float(np.mean(b_rip["fc_full_new"])), 4),
            "ripple_em": round(float(np.mean(b_rip["em_new"])), 4),
            "hn_ff_fc1": round(float(np.mean(b_hn["fc1_new"])), 4),
            "hn_ff_fc_full": round(float(np.mean(b_hn["fc_full_new"])), 4),
            "hn_ff_em": round(float(np.mean(b_hn["em_new"])), 4),
        }

        for arm_name, (gkey, w_dict) in arms_weights.items():
            poe.widen_mlp_native(mlp_B, orig_state_B, w_dict)
            try:
                e_dir = eval_items_full_telemetry(model_B, tok_B, dev_B, suite["direct"], is_hn=False)
                e_rip = eval_items_full_telemetry(model_B, tok_B, dev_B, suite["ripple"], is_hn=False)
                e_hn = eval_items_full_telemetry(model_B, tok_B, dev_B, suite["hardneg"], is_hn=True)
            finally:
                po.unlearn_mlp_native(mlp_B, orig_state_B)

            tax_dir = classify_failure_taxonomy(suite["direct"], gate_dir, gkey, e_dir, use_full_fc=True)
            tax_rip = classify_failure_taxonomy(suite["ripple"], gate_rip, gkey, e_rip, use_full_fc=True)
            hn_ff_fc1 = round(float(np.mean(e_hn["fc1_new"])), 4)
            hn_ff_full = round(float(np.mean(e_hn["fc_full_new"])), 4)
            hn_ff_em = round(float(np.mean(e_hn["em_new"])), 4)

            # Save concise per-item direct sample ledger (all 48 direct items with gen + phi + fc_full + em!)
            direct_ledger = []
            for i, it in enumerate(suite["direct"]):
                rname = it["rule"]
                g = gate_dir[i]["gates"][gkey][rname]
                direct_ledger.append({
                    "species": it["species"],
                    "rule": rname,
                    "phi_subj": g["phi_subj"],
                    "z_subj": g["z_subj"],
                    "fc1": e_dir["fc1_new"][i],
                    "fc_full": e_dir["fc_full_new"][i],
                    "fc_full_lo": e_dir["fc_full_log_odds"][i],
                    "em": e_dir["em_new"][i],
                    "gen": e_dir["gens"][i],
                })

            arm_results[arm_name] = {
                "direct": tax_dir,
                "ripple": tax_rip,
                "hardneg": {"ff_fc1": hn_ff_fc1, "ff_fc_full": hn_ff_full, "ff_em": hn_ff_em},
                "direct_ledger": direct_ledger,
            }
            print(
                f"  [{arm_name:26s}] DIR fc1={tax_dir['fc1_acc']:.3f} full={tax_dir['fc_full_acc']:.3f} em={tax_dir['em_acc']:.3f} "
                f"(buckets={tax_dir['buckets']}) | "
                f"RIP fc1={tax_rip['fc1_acc']:.3f} full={tax_rip['fc_full_acc']:.3f} em={tax_rip['em_acc']:.3f} | "
                f"HN ff={hn_ff_full:.3f}/{hn_ff_em:.3f}"
            )

        # 4. Forced-Gold Write Controls (phi == 1.0 at species token, Direct + Ripple!)
        for fname, vmap in (("FORCED_GOLD_PUSH_WRITE", v_push_map), ("FORCED_GOLD_NATIVE_WRITE", v_nat_map)):
            fd = eval_forced_gold_write(model_B, tok_B, dev_B, lB, rm_B, suite["direct"], vmap)
            fr = eval_forced_gold_write(model_B, tok_B, dev_B, lB, rm_B, suite["ripple"], vmap)
            arm_results[fname] = {
                "direct_fc1": round(float(np.mean(fd["fc1_new"])), 4),
                "direct_fc_full": round(float(np.mean(fd["fc_full_new"])), 4),
                "direct_em": round(float(np.mean(fd["em_new"])), 4),
                "direct_mean_lo": round(float(np.mean(fd["fc_full_log_odds"])), 4),
                "ripple_fc1": round(float(np.mean(fr["fc1_new"])), 4),
                "ripple_fc_full": round(float(np.mean(fr["fc_full_new"])), 4),
                "ripple_em": round(float(np.mean(fr["em_new"])), 4),
                "ripple_mean_lo": round(float(np.mean(fr["fc_full_log_odds"])), 4),
                "direct_by_rule": {
                    rn: {
                        "fc_full": round(float(np.mean([fd["fc_full_new"][i] for i, it in enumerate(suite["direct"]) if it["rule"] == rn])), 4),
                        "em": round(float(np.mean([fd["em_new"][i] for i, it in enumerate(suite["direct"]) if it["rule"] == rn])), 4),
                    }
                    for rn in ("Bird->Fish", "Mammal->Reptile")
                },
                "ripple_by_rule": {
                    rn: {
                        "fc_full": round(float(np.mean([fr["fc_full_new"][i] for i, it in enumerate(suite["ripple"]) if it["rule"] == rn])), 4),
                        "em": round(float(np.mean([fr["em_new"][i] for i, it in enumerate(suite["ripple"]) if it["rule"] == rn])), 4),
                    }
                    for rn in ("Bird->Fish", "Mammal->Reptile")
                },
                "sample_mammal_gens": [fd["gens"][i] for i, it in enumerate(suite["direct"]) if it["rule"] == "Mammal->Reptile"][:8],
            }
            print(
                f"  [{fname:26s}] DIR fc1={arm_results[fname]['direct_fc1']:.3f} full={arm_results[fname]['direct_fc_full']:.3f} em={arm_results[fname]['direct_em']:.3f} | "
                f"RIP fc1={arm_results[fname]['ripple_fc1']:.3f} full={arm_results[fname]['ripple_fc_full']:.3f} em={arm_results[fname]['ripple_em']:.3f}"
            )

        out["recipients"][tgt] = {
            "model_key": rk,
            "layer": lB,
            "rm": rm_B,
            "write_direction_audit": write_diag,
            "gate_roc_summary": roc_summary,
            "arms": arm_results,
            "elapsed_sec": round(time.time() - tr0, 1),
        }

        po.unload_model(model_B)
        del model_B
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Save incremental checkpoint after each recipient
        out["meta"]["elapsed_sec"] = round(time.time() - t0, 1)
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n[v2_step0_rw_diagnostic] COMPLETE in {time.time() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
