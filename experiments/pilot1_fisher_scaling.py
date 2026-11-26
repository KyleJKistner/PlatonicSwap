#!/usr/bin/env python3
"""
pilot1_fisher_scaling.py — Pilot 1 (Track 1): Unlabeled N >> D Scaling +
Closed-Form Causal-Fisher Pullback Procrustes + Low-DOF SO(32)/SO(64) MSE vs KL
================================================================================

Discovery pair (default): qwen2.5-3b -> smollm2-1.7b (equal-width D=2048 cross-family;
reserves granite-3.1-2b & falcon3-3b for independent replication, though --pair supports them).

What this script runs:
  1. Builds a clean, 100% leak-free UNLABELED entity-mention corpus (N = 10,000 distinct
     entities + 512 unseen unlabeled validation entities) with exact aligned word-final
     character boundaries across both tokenizers, STRICTLY excluding every subject, answer,
     alias, and >=4-char content word in dataset.json val/test (and dataset_2hop.json val/test).
  2. Extracts mid-layer hidden states at (l_A, l_B) on all 10,512 unlabeled entity spans
     (fast bf16 on MPS) AND computes exact UNLABELED Causal Fisher matrices on Recipient B
     (N_fisher = 2,000 unlabeled spans in float32 on MPS, frozen weights):
       - F_B_end: clause-end continuation sampled-y score Fisher (zero gold labels!)
       - F_B_multi: Neon's exact unbiased multi-step causal Fisher sum_{s >= t_ent} F_{t->s}
  3. Sweeps N in {500, 2000, 10000} (plus Round-1 QA train reference) comparing closed-form maps
     (all algebra in float64 on CPU Accelerate/AMX):
       - procrustes_global (Plato's Key O(d) + global scale)
       - procrustes_span (span-restricted QR-SVD)
       - ridge_gl & ridge_gl_rescaled (lambda selected strictly by unlabeled held-out MSE)
       - zca_whitened_procrustes (Sigma_B^{1/2} Q Sigma_A^{-1/2}, Q = Polar(Sigma_B^{1/2} Z_B Z_A_tilde^T))
       - fisher_pullback_zca (Carbon's exact closed-form Causal-Fisher Pullback Procrustes:
           Q*_Fisher = Polar(Sigma_B^{1/2} F_{B,gamma} (1/N Z_B Z_A_tilde^T)), gamma selected by
           unlabeled continuation KL!)
       - fisher_orth_procrustes (Pure O(d) Fisher-weighted Procrustes minimizing
           ||F_{B,gamma}^{1/2} (s R Z_A - Z_B)||_F^2 via closed-form polar + Riemannian refinement)
       - fisher_diag_procrustes (Diagonal-Fisher Procrustes isolating axis/RMSNorm scaling)
       - Rank-k in {32, 64} compression: PCA-Procrustes vs PLS vs Fisher-weighted rank-k!
  4. Low-DOF SO(32) [496 DOF] & SO(64) [2016 DOF] head-to-head:
       R(A) = [I + U (Q(A) - I) U^T] R_0,  Q(A) = (I - A)(I + A)^{-1} in SO(r)
       with R_0, mu_A, mu_B, s FROZEN and U = top-r eigenvectors of unlabeled F_B!
       Compares inside the EXACT SAME 496-DOF / 2016-DOF class:
         - so{r}_mse_unlab vs so{r}_kl_unlab (strictly unlabeled text continuation KL!)
         - so{r}_mse_train10 vs so{r}_kl_selfswap_train10 (strictly on 10 train relations,
           evaluated zero-shot on 5 completely unseen LORO relation families!)
  5. Evaluates EVERY method on:
       - Held-out test Counterfactual Flip (CF-Flip, zero GlobalFV!), matched vs shuffled donor,
         answer-disjoint slice, and subject-clustered 95% bootstrap CIs;
       - Held-out LORO 5 unseen relation families (matched vs shuffled donor + 95% CI);
       - Unlabeled held-out continuation KL guardrail (general-language preservation).

Usage:
  python pilot1_fisher_scaling.py --pair qwen2.5-3b->smollm2-1.7b
  python pilot1_fisher_scaling.py --pair qwen2.5-3b->smollm2-1.7b --smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from extract_and_baseline import (
    CACHE_DIR,
    DATASET_2HOP_PATH,
    DATASET_PATH,
    LAB_DIR,
    RESULTS_PATH as BASELINE_RESULTS_PATH,
    SEED,
    apply_linear_map,
    build_counterfactual_pairs,
    eval_patched_swaps,
    fit_procrustes_global,
    get_decoder_layers,
    locate_last_subject_token,
    pretokenize_swap_batches,
)

# 5 diverse held-out LORO relation families (disjoint answer types: sports, people, symbols, countries, US capitals)
LORO_HELD_OUT_RELS = {
    "athlete_sport",
    "company_founder",
    "element_symbol",
    "historical_figure_country",
    "us_state_capital",
}

# Diverse natural non-QA narrative prefixes (ALL grammatically clean subject-introducers!)
UNLAB_PREFIXES = [
    "Historically, ",
    "In modern records, ",
    "According to historians, ",
    "Around the world, ",
    "In many accounts, ",
    "Over the centuries, ",
    "By all accounts, ",
    "In public records, ",
    "In recent decades, ",
    "Across the region, ",
    "In encyclopedias, ",
    "In classical accounts, ",
    "In international reports, ",
    "Among scholars, ",
    "In standard references, ",
    "Throughout history, ",
]

UNLAB_CONTINUATIONS = [
    " is widely known for",
    " is closely associated with",
    " played a notable role in",
    " has a long history in",
    " became prominent during",
    " is primarily remembered for",
    " originated in the context of",
    " is recognized across",
    " was historically linked to",
    " remains an important part of",
    " is frequently discussed in",
    " developed strong ties to",
]

STOPWORDS_LOWER = {
    "about", "above", "actually", "after", "again", "against", "almost", "alone", "along",
    "already", "also", "although", "always", "among", "another", "around", "because", "become",
    "becomes", "before", "behind", "being", "below", "between", "beyond", "both", "certainly",
    "clearly", "could", "definitely", "doing", "down", "during", "each", "either", "else",
    "enough", "especially", "even", "ever", "every", "finally", "from", "further", "generally",
    "have", "having", "here", "herself", "himself", "however", "indeed", "instead", "into",
    "itself", "just", "keep", "least", "less", "like", "likely", "made", "make", "making",
    "many", "maybe", "meanwhile", "might", "more", "most", "mostly", "much", "must", "myself",
    "near", "neither", "never", "nevertheless", "next", "none", "nothing", "obviously", "often",
    "once", "only", "onto", "other", "others", "otherwise", "ours", "ourselves", "over",
    "perhaps", "please", "possibly", "probably", "quite", "rather", "really", "recently", "same",
    "several", "shall", "should", "simply", "since", "some", "somehow", "someone", "something",
    "sometimes", "somewhat", "somewhere", "still", "such", "suddenly", "than", "that", "their",
    "theirs", "them", "themselves", "then", "there", "therefore", "these", "they", "thing",
    "things", "this", "those", "though", "through", "throughout", "thus", "today", "together",
    "tomorrow", "toward", "towards", "truly", "under", "unless", "until", "upon", "used",
    "using", "usually", "very", "want", "were", "what", "whatever", "when", "whenever", "where",
    "whereas", "wherever", "whether", "which", "while", "whoever", "whole", "whom", "whose",
    "will", "with", "within", "without", "would", "yesterday", "your", "yours", "yourself",
    "yourselves",
}


# ======================================================================================
# 1. Clean, Leak-Free Unlabeled Entity-Mention Corpus Builder (N = 10,000 + 512 Val)
# ======================================================================================

def build_exclusion_set(dataset: dict, dataset_2hop: Optional[dict] = None) -> Tuple[set, set]:
    """
    Build strict exclusion sets from ALL val & test facts in dataset.json (and dataset_2hop.json):
      - exact normalized subjects, answers, aliases
      - every alphabetic word token of length >= 4 appearing in any val/test subject or answer/alias
    """
    exact_excl = set()
    word_excl = set()

    def _add_str(s: str):
        sl = s.strip().lower()
        if sl:
            exact_excl.add(sl)
        for w in re.findall(r"[a-z]+", sl):
            if len(w) >= 4:
                word_excl.add(w)

    for f in dataset["facts"]:
        if f["split"] in ("val", "test"):
            _add_str(f["subject"])
            _add_str(f["answer"])
            for a in f.get("aliases", []):
                _add_str(a)

    if dataset_2hop is not None:
        for it in dataset_2hop.get("items", []):
            if it.get("split") in ("val", "test"):
                _add_str(it.get("subject", ""))
                _add_str(it.get("bridge_entity", ""))
                _add_str(it.get("answer", ""))
                for a in it.get("aliases", []):
                    _add_str(a)

    return exact_excl, word_excl


def build_unlabeled_entity_corpus(
    tok_A: Any,
    tok_B: Any,
    dataset: dict,
    dataset_2hop: Optional[dict],
    n_train: int = 10000,
    n_val: int = 512,
    seed: int = SEED,
) -> Tuple[List[dict], List[dict], dict]:
    """
    Build N = n_train + n_val distinct unlabeled entity-mention sentences with:
      - STRICTLY zero overlap with any val/test subject, answer, alias, or >=4-char word;
      - EXACT aligned word-final character boundaries across both tok_A and tok_B!
    """
    exact_excl, word_excl = build_exclusion_set(dataset, dataset_2hop)

    def _extract_clean_vocab_words(tok: Any) -> Tuple[set, set]:
        caps = set()
        lows = set()
        for tid in range(len(tok)):
            s = tok.decode([tid])
            if s.startswith(" ") and len(s) >= 4:
                w = s[1:]
                if re.fullmatch(r"[A-Z][a-z]{2,15}", w):
                    caps.add(w)
                elif re.fullmatch(r"[a-z]{4,15}", w):
                    lows.add(w)
        return caps, lows

    cap_A, low_A = _extract_clean_vocab_words(tok_A)
    cap_B, low_B = _extract_clean_vocab_words(tok_B)

    def _is_excluded(w: str) -> bool:
        wl = w.lower()
        return (
            (wl in exact_excl)
            or (wl in word_excl)
            or (wl in STOPWORDS_LOWER)
            or (wl.endswith("ly") and len(wl) >= 6)
            or (w[0].islower() and (wl.endswith("ed") or wl.endswith("ing") or wl.endswith("est") or wl.endswith("ier")))
        )

    cap_inter = sorted(w for w in (cap_A & cap_B) if not _is_excluded(w))
    low_inter = sorted(
        w for w in (low_A & low_B)
        if not _is_excluded(w) and w.capitalize() not in (cap_A & cap_B)
    )
    # If tokenizer intersection has < n_total + 200 words (e.g. Granite BPE), append clean
    # union words (multi-token in one model, still strictly boundary-verified offset_end == c1!).
    cap_extra = sorted(w for w in ((cap_A | cap_B) - (cap_A & cap_B)) if not _is_excluded(w))
    low_extra = sorted(
        w for w in ((low_A | low_B) - (low_A & low_B))
        if not _is_excluded(w) and w.capitalize() not in (cap_A | cap_B)
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(cap_inter)
    rng.shuffle(low_inter)
    rng.shuffle(cap_extra)
    rng.shuffle(low_extra)

    # Prioritize capitalized shared words -> capitalized union -> lowercase shared -> lowercase union
    n_total = n_train + n_val
    pool_words = list(cap_inter) + list(cap_extra) + list(low_inter) + list(low_extra)
    assert len(pool_words) >= n_total + 200, (
        f"Not enough clean shared entity words: {len(pool_words)} < {n_total + 200}"
    )

    items: List[dict] = []
    n_cap_used = 0
    n_low_used = 0
    n_boundary_rejected = 0

    for idx_w, w in enumerate(pool_words):
        if len(items) >= n_total:
            break
        pref = UNLAB_PREFIXES[idx_w % len(UNLAB_PREFIXES)]
        cont = UNLAB_CONTINUATIONS[(idx_w * 7 + idx_w // len(UNLAB_PREFIXES)) % len(UNLAB_CONTINUATIONS)]
        # For lowercase nouns after comma/verb, add 'the ' so syntax is completely natural
        is_cap = w[0].isupper()
        if (not is_cap) and (not pref.endswith("the ")):
            pref_eff = pref + "the "
        else:
            pref_eff = pref
        prompt = pref_eff + w + cont
        c0 = len(pref_eff)
        c1 = c0 + len(w)

        # Verify EXACT aligned word-final character boundary in BOTH tokenizers!
        enc_A = tok_A(prompt, return_offsets_mapping=True, add_special_tokens=True)
        enc_B = tok_B(prompt, return_offsets_mapping=True, add_special_tokens=True)

        def _exact_end_tok(offsets: Sequence[Tuple[int, int]]) -> int:
            last_i = -1
            for ti, (s, e) in enumerate(offsets):
                if e <= s:
                    continue
                if s < c1 and e > c0:
                    last_i = ti
            if last_i < 0:
                return -1
            # Require exact word-final boundary match (offset_end == c1)
            if offsets[last_i][1] != c1:
                return -1
            return last_i

        tA = _exact_end_tok(enc_A["offset_mapping"])
        tB = _exact_end_tok(enc_B["offset_mapping"])
        if tA < 0 or tB < 0:
            n_boundary_rejected += 1
            continue

        items.append({
            "entity": w,
            "is_capitalized": is_cap,
            "prompt": prompt,
            "char_span": [c0, c1],
            "tok_idx_A": int(tA),
            "seq_len_A": int(len(enc_A["input_ids"])),
            "tok_idx_B": int(tB),
            "seq_len_B": int(len(enc_B["input_ids"])),
        })
        if is_cap:
            n_cap_used += 1
        else:
            n_low_used += 1

    assert len(items) == n_total, f"Built {len(items)} != {n_total}"

    # Deterministic shuffle so nested prefixes N=500, 2000, 10000 have identical cap/low mix!
    perm = rng.permutation(n_total)
    items = [items[i] for i in perm]
    train_items = items[:n_train]
    val_items = items[n_train:]

    # Final leak-audit assertion
    val_test_forbidden = exact_excl | word_excl
    leaks = [it["entity"] for it in items if it["entity"].lower() in val_test_forbidden]
    assert len(leaks) == 0, f"LEAKAGE DETECTED: {leaks[:5]}"

    meta = {
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_capitalized_proper_nouns": n_cap_used,
        "n_lowercase_content_nouns": n_low_used,
        "n_boundary_rejected": n_boundary_rejected,
        "val_test_overlap_count": 0,
        "sample_prompts": [train_items[i]["prompt"] for i in range(5)],
    }
    return train_items, val_items, meta


# ======================================================================================
# 2. Fast Batched Unlabeled State & Multi-Step Causal Fisher Extraction
# ======================================================================================

def extract_unlabeled_layer_states(
    model: Any,
    tok: Any,
    items: List[dict],
    layer_idx: int,
    side: str,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract hidden state [N, D] (float32) at (layer_idx, entity word-final token)."""
    layers = get_decoder_layers(model)
    target_layer = layers[layer_idx]
    n = len(items)
    D = int(model.config.hidden_size)
    out_mat = np.zeros((n, D), dtype=np.float32)

    tok_key = "tok_idx_A" if side == "A" else "tok_idx_B"
    len_key = "seq_len_A" if side == "A" else "seq_len_B"
    ctx: Dict[str, Any] = {"pos": None, "buf": None}

    def hook_fn(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        bsz = h.shape[0]
        b_ar = torch.arange(bsz, device=h.device)
        ctx["buf"] = h[b_ar, ctx["pos"], :].detach().float().cpu().numpy()

    handle = target_layer.register_forward_hook(hook_fn)
    try:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            prompts = [items[i]["prompt"] for i in range(start, end)]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            t_pad = input_ids.shape[1]
            pos = []
            for i in range(start, end):
                pad_len = t_pad - items[i][len_key]
                pos.append(max(0, min(t_pad - 1, pad_len + items[i][tok_key])))
            ctx["pos"] = torch.tensor(pos, dtype=torch.long, device=device)
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            out_mat[start:end] = ctx["buf"]
    finally:
        handle.remove()
    return out_mat


def extract_unlabeled_causal_fisher(
    model_B_f32: Any,
    tok_B: Any,
    items: List[dict],
    layer_B: int,
    device: torch.device,
    batch_size: int = 16,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute TWO exact UNLABELED Causal Fisher gradient banks [N, D] on Recipient B (float32):
      1. G_end: grad_{z_B(t_ent)} log p_B(y_hat_last | x) with y_hat_last ~ p_B(. | x)
      2. G_multi: Neon's exact unbiased multi-step causal Fisher:
           grad_{z_B(t_ent)} sum_{s = t_ent}^{T-1} log p_B(y_hat_s | x_{<=s}),
         where y_hat_s ~ p_B(. | x_{<=s}) independently at each post-entity position s.
    Zero gold labels, zero QA prompts, frozen model weights!
    """
    layers = get_decoder_layers(model_B_f32)
    target_layer = layers[layer_B]
    n = len(items)
    D = int(model_B_f32.config.hidden_size)
    G_end = np.zeros((n, D), dtype=np.float32)
    G_multi = np.zeros((n, D), dtype=np.float32)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    ctx: Dict[str, Any] = {"pos": None, "z_leaf": None}

    def hook_fn(_mod, _inp, out):
        is_tup = isinstance(out, tuple)
        h = out[0] if is_tup else out
        bsz = h.shape[0]
        b_ar = torch.arange(bsz, device=h.device)
        z_leaf = h[b_ar, ctx["pos"], :].detach().clone().requires_grad_(True)
        ctx["z_leaf"] = z_leaf
        h_new = h.clone()
        h_new[b_ar, ctx["pos"], :] = z_leaf
        return (h_new,) + out[1:] if is_tup else h_new

    handle = target_layer.register_forward_hook(hook_fn)
    try:
        for mode in ("end", "multi"):
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                prompts = [items[i]["prompt"] for i in range(start, end)]
                enc = tok_B(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                bsz, t_pad = input_ids.shape
                pos = []
                for i in range(start, end):
                    pad_len = t_pad - items[i]["seq_len_B"]
                    pos.append(max(0, min(t_pad - 1, pad_len + items[i]["tok_idx_B"])))
                ctx["pos"] = torch.tensor(pos, dtype=torch.long, device=device)

                with torch.enable_grad():
                    out = model_B_f32(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                    if mode == "end":
                        logits_last = out.logits[:, -1, :].float()  # [B, V]
                        lp_last = F.log_softmax(logits_last, dim=-1)
                        probs_cpu = torch.softmax(logits_last.detach().cpu(), dim=-1)
                        y_samp = torch.multinomial(probs_cpu, num_samples=1, generator=gen).squeeze(-1).to(device)
                        b_ar = torch.arange(bsz, device=device)
                        loss = lp_last[b_ar, y_samp].sum()
                    else:
                        # Multi-step unbiased Fisher over all post-entity positions s in [pos[b], t_pad - 1]
                        logits_all = out.logits.float()  # [B, T, V]
                        lp_all = F.log_softmax(logits_all, dim=-1)
                        # Sample independently at every position >= min(pos)
                        min_p = min(pos)
                        probs_sub = torch.softmax(logits_all[:, min_p:, :].detach().cpu(), dim=-1)  # [B, T', V]
                        t_sub = probs_sub.shape[1]
                        y_sub = torch.multinomial(
                            probs_sub.reshape(-1, probs_sub.shape[-1]), num_samples=1, generator=gen
                        ).reshape(bsz, t_sub).to(device)
                        # Gather logp at sampled tokens and mask strictly to s >= pos[b]
                        lp_sub = lp_all[:, min_p:, :].gather(dim=-1, index=y_sub.unsqueeze(-1)).squeeze(-1)  # [B, T']
                        t_coords = torch.arange(min_p, t_pad, device=device).unsqueeze(0)  # [1, T']
                        mask = (t_coords >= ctx["pos"].unsqueeze(1)).float()
                        # Scale by 1 / sqrt(n_steps) so trace is comparable across lengths
                        loss = (lp_sub * mask).sum()

                    g = torch.autograd.grad(loss, ctx["z_leaf"], retain_graph=False)[0]
                if mode == "end":
                    G_end[start:end] = g.detach().float().cpu().numpy()
                else:
                    G_multi[start:end] = g.detach().float().cpu().numpy()
                del out, g
    finally:
        handle.remove()

    return G_end, G_multi


# ======================================================================================
# 3. Closed-Form Alignment Library (Float64 CPU: Procrustes, Span, Ridge, ZCA, Fisher)
# ======================================================================================

def _sym_eigh_power(M: np.ndarray, power: float, shrinkage_frac: float = 0.02) -> np.ndarray:
    """Compute (M + lam * (tr(M)/d) * I)^power in float64 symmetrically."""
    d = M.shape[0]
    tr = float(np.trace(M))
    lam = shrinkage_frac * max(tr / d, 1e-12)
    M_reg = 0.5 * (M + M.T) + lam * np.eye(d, dtype=np.float64)
    w, V = sla.eigh(M_reg, check_finite=False)
    w = np.maximum(w, 1e-12)
    return (V * (w ** power)) @ V.T


def _polar(M: np.ndarray) -> np.ndarray:
    try:
        U, _, Vt = sla.svd(M, full_matrices=False, lapack_driver="gesdd", check_finite=False)
    except sla.LinAlgError:
        U, _, Vt = sla.svd(M, full_matrices=False, lapack_driver="gesvd", check_finite=False)
    return U @ Vt


def apply_affine_map(Z_A: np.ndarray, mp: dict) -> np.ndarray:
    """General affine map z_hat = mu_B + (z_A - mu_A) @ W^T (where W = s * R for orthogonal maps)."""
    z = Z_A.astype(np.float64)
    if "W" in mp:
        return mp["mu_B"] + (z - mp["mu_A"]) @ mp["W"].T
    return mp["mu_B"] + mp["s"] * ((z - mp["mu_A"]) @ mp["R"].T)


def fit_procrustes_span_f64(Z_A: np.ndarray, Z_B: np.ndarray) -> dict:
    ZA = Z_A.astype(np.float64)
    ZB = Z_B.astype(np.float64)
    mu_A = ZA.mean(axis=0)
    mu_B = ZB.mean(axis=0)
    Ac = ZA - mu_A
    Bc = ZB - mu_B
    if Ac.shape[0] >= min(Ac.shape[1], Bc.shape[1]):
        R = _polar(Bc.T @ Ac)
    else:
        QA, RA = np.linalg.qr(Ac.T, mode="reduced")
        QB, RB = np.linalg.qr(Bc.T, mode="reduced")
        Us, sv, Vts = sla.svd(RB @ RA.T, full_matrices=False, check_finite=False)
        tol = 1e-10 * max(float(sv[0]) if sv.size else 0.0, 1e-12)
        r = int(np.sum(sv > tol))
        R = (QB @ Us[:, :r]) @ (Vts[:r, :] @ QA.T)
    s = float(np.linalg.norm(Bc, "fro") / max(np.linalg.norm(Ac, "fro"), 1e-12))
    return {"mu_A": mu_A, "mu_B": mu_B, "R": R, "s": s}


def fit_ridge_gl(
    Z_A: np.ndarray,
    Z_B: np.ndarray,
    Z_A_val: np.ndarray,
    Z_B_val: np.ndarray,
    lam_grid: Sequence[float] = (1e-3, 1e-2, 5e-2, 0.1, 0.3, 1.0, 3.0, 10.0),
) -> Tuple[dict, dict]:
    """
    Fit Ridge-GL(d) W = (B_c^T A_c) (A_c^T A_c + lam * avg_eig * I)^{-1},
    choosing lam STRICTLY by unlabeled held-out MSE (zero QA leakage!).
    Returns (ridge_raw_map, ridge_rescaled_map).
    """
    ZA = Z_A.astype(np.float64)
    ZB = Z_B.astype(np.float64)
    mu_A = ZA.mean(axis=0)
    mu_B = ZB.mean(axis=0)
    Ac = ZA - mu_A
    Bc = ZB - mu_B
    n, dA = Ac.shape
    Sig_A = (Ac.T @ Ac) / max(n, 1)
    C_BA = (Bc.T @ Ac) / max(n, 1)
    w_a, V_a = sla.eigh(0.5 * (Sig_A + Sig_A.T), check_finite=False)
    w_a = np.maximum(w_a, 0.0)
    avg_eig = max(float(np.mean(w_a)), 1e-12)

    Ac_val = Z_A_val.astype(np.float64) - mu_A
    Bc_val = Z_B_val.astype(np.float64) - mu_B
    C_Vt = C_BA @ V_a  # [dB, dA]
    Ac_val_V = Ac_val @ V_a  # [n_val, dA]

    best_lam = lam_grid[0]
    best_mse = float("inf")
    best_W = None
    for lam in lam_grid:
        inv_diag = 1.0 / (w_a + lam * avg_eig)
        pred_val = (Ac_val_V * inv_diag) @ C_Vt.T
        mse = float(np.mean((pred_val - Bc_val) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_lam = lam
            best_W = (C_Vt * inv_diag) @ V_a.T

    pred_tr = Ac @ best_W.T
    s_resc = float(np.linalg.norm(Bc, "fro") / max(np.linalg.norm(pred_tr, "fro"), 1e-12))
    mp_raw = {"mu_A": mu_A, "mu_B": mu_B, "W": best_W, "lam": best_lam, "val_mse": best_mse}
    mp_resc = {"mu_A": mu_A, "mu_B": mu_B, "W": s_resc * best_W, "lam": best_lam, "s_resc": s_resc}
    return mp_raw, mp_resc


def precompute_zca_cov(Z_A: np.ndarray, Z_B: np.ndarray, shrink_cov: float = 0.02) -> dict:
    """Precompute ZCA whitening/coloring matrices and cross-covariance ONCE per (N, shrink_cov)."""
    ZA = Z_A.astype(np.float64)
    ZB = Z_B.astype(np.float64)
    mu_A = ZA.mean(axis=0)
    mu_B = ZB.mean(axis=0)
    Ac = ZA - mu_A
    Bc = ZB - mu_B
    n = len(ZA)
    Sig_A = (Ac.T @ Ac) / max(n, 1)
    Sig_B = (Bc.T @ Bc) / max(n, 1)
    Sig_A_inv_half = _sym_eigh_power(Sig_A, -0.5, shrinkage_frac=shrink_cov)
    Sig_B_half = _sym_eigh_power(Sig_B, +0.5, shrinkage_frac=shrink_cov)
    Ac_tilde = Ac @ Sig_A_inv_half
    C_cross = (Bc.T @ Ac_tilde) / max(n, 1)
    return {
        "mu_A": mu_A,
        "mu_B": mu_B,
        "Ac": Ac,
        "Bc_frob": float(np.linalg.norm(Bc, "fro")),
        "Sig_A_inv_half": Sig_A_inv_half,
        "Sig_B_half": Sig_B_half,
        "C_cross": C_cross,
        "shrink_cov": shrink_cov,
    }


def fit_zca_and_fisher_pullback(
    Z_A: np.ndarray,
    Z_B: np.ndarray,
    F_B: Optional[np.ndarray] = None,
    shrink_cov: float = 0.02,
    fisher_damp: float = 0.1,
    diag_fisher_only: bool = False,
    cov_cache: Optional[dict] = None,
) -> dict:
    """
    Carbon's exact closed-form Theorem (ZCA-Whitened & Causal-Fisher Pullback Procrustes):
      Z_A_tilde = Sigma_{A,lam}^{-1/2} (Z_A - mu_A)
      If F_B is None (ZCA-Whitened Procrustes / Monge):
        Q* = Polar(Sigma_{B,lam}^{1/2} * (1/N Z_B_c Z_A_tilde^T))
      If F_B is provided (Closed-Form Causal-Fisher Pullback Procrustes):
        F_{B,gamma} = F_B_norm + gamma * I
        Q*_Fisher = Polar(Sigma_{B,lam}^{1/2} * F_{B,gamma} * (1/N Z_B_c Z_A_tilde^T))
      Map: M = Sigma_{B,lam}^{1/2} @ Q* @ Sigma_{A,lam}^{-1/2}
    Note: unit-RMS / Granite invariant because Sigma_B^{1/2} carries exact recipient scale!
    """
    cc = cov_cache if cov_cache is not None else precompute_zca_cov(Z_A, Z_B, shrink_cov=shrink_cov)
    mu_A, mu_B = cc["mu_A"], cc["mu_B"]
    Ac, Bc_frob = cc["Ac"], cc["Bc_frob"]
    Sig_A_inv_half, Sig_B_half, C_cross = cc["Sig_A_inv_half"], cc["Sig_B_half"], cc["C_cross"]

    if F_B is None:
        M_polar = Sig_B_half @ C_cross
    else:
        dB = F_B.shape[0]
        F_use = np.diag(np.diag(F_B)) if diag_fisher_only else 0.5 * (F_B + F_B.T)
        tr_F = max(float(np.trace(F_use)) / dB, 1e-12)
        F_reg = (F_use / tr_F) + fisher_damp * np.eye(dB, dtype=np.float64)
        M_polar = Sig_B_half @ (F_reg @ C_cross)

    Q = _polar(M_polar)
    W = Sig_B_half @ Q @ Sig_A_inv_half
    pred_tr = Ac @ W.T
    s_match = float(Bc_frob / max(np.linalg.norm(pred_tr, "fro"), 1e-12))
    return {
        "mu_A": mu_A,
        "mu_B": mu_B,
        "W": s_match * W,
        "W_raw": W,
        "Q": Q,
        "s_match": s_match,
        "shrink_cov": shrink_cov,
        "fisher_damp": fisher_damp,
    }


def fit_fisher_orthogonal_procrustes(
    Z_A: np.ndarray,
    Z_B: np.ndarray,
    F_B: np.ndarray,
    fisher_damp: float = 0.1,
    diag_only: bool = False,
    n_iter: int = 12,
) -> dict:
    """
    Pure O(d) Fisher-Weighted Procrustes (strictly orthogonal R in O(d), NO non-orthogonal shear!):
      min_{R in O(d)} || F_{B,gamma}^{1/2} (s R Z_A_c - Z_B_c) ||_F^2
    Solved in closed-form majorization-minimization (exact monotonic polar iterations on CPU f64):
      At each step, gradient of 0.5 * Tr(R^T F R Sig_A) - Tr(R^T F C_BA) is F R Sig_A - F C_BA;
      majorizer with L = lam_max(F) * lam_max(Sig_A) gives exact polar update:
        R_{t+1} = Polar(R_t - (1/L) (F R_t Sig_A - F C_BA))
      Guaranteed strictly R^T R = I and monotonically decreasing Fisher-weighted objective!
    """
    ZA = Z_A.astype(np.float64)
    ZB = Z_B.astype(np.float64)
    mu_A = ZA.mean(axis=0)
    mu_B = ZB.mean(axis=0)
    Ac = ZA - mu_A
    Bc = ZB - mu_B
    n, dA = Ac.shape
    dB = Bc.shape[1]
    s = float(np.linalg.norm(Bc, "fro") / max(np.linalg.norm(Ac, "fro"), 1e-12))

    F_use = np.diag(np.diag(F_B)) if diag_only else 0.5 * (F_B + F_B.T)
    tr_F = max(float(np.trace(F_use)) / dB, 1e-12)
    F_reg = (F_use / tr_F) + fisher_damp * np.eye(dB, dtype=np.float64)

    Sig_A = (s * s) * (Ac.T @ Ac) / max(n, 1)
    C_BA = s * (Bc.T @ Ac) / max(n, 1)

    # Closed-form linear Fisher polar initialization
    R = _polar(F_reg @ C_BA)

    if n_iter > 0:
        lam_F = float(sla.eigh(F_reg, eigvals_only=True, subset_by_index=[dB - 1, dB - 1])[0])
        lam_A = float(sla.eigh(Sig_A, eigvals_only=True, subset_by_index=[dA - 1, dA - 1])[0])
        step = 1.0 / max(lam_F * lam_A, 1e-12)
        FC = F_reg @ C_BA
        for _ in range(n_iter):
            grad = (F_reg @ R) @ Sig_A - FC
            R = _polar(R - step * grad)

    return {"mu_A": mu_A, "mu_B": mu_B, "R": R, "s": s, "fisher_damp": fisher_damp}


def fit_rank_k_compression(
    Z_A: np.ndarray,
    Z_B: np.ndarray,
    k: int,
    mode: str,
    F_B: Optional[np.ndarray] = None,
    fisher_damp: float = 0.05,
) -> dict:
    """
    Rank-k compressed maps (k in {32, 64}) comparing:
      - 'pca': Separate top-k PCA per model + k x k Procrustes (Plato's Key baseline)
      - 'pls': Top-k cross-covariance SVD U_k V_k^T (Round-1 champion)
      - 'fisher_pls': Top-k Fisher-weighted cross-covariance SVD in F_{B,gamma}^{1/2} metric!
    """
    ZA = Z_A.astype(np.float64)
    ZB = Z_B.astype(np.float64)
    mu_A = ZA.mean(axis=0)
    mu_B = ZB.mean(axis=0)
    Ac = ZA - mu_A
    Bc = ZB - mu_B

    if mode == "pca":
        _, _, VtA = sla.svd(Ac, full_matrices=False, check_finite=False)
        _, _, VtB = sla.svd(Bc, full_matrices=False, check_finite=False)
        PA = VtA[:k]  # [k, dA]
        PB = VtB[:k]  # [k, dB]
        Ak = Ac @ PA.T
        Bk = Bc @ PB.T
        Rk = _polar(Bk.T @ Ak)
        sk = float(np.linalg.norm(Bk, "fro") / max(np.linalg.norm(Ak, "fro"), 1e-12))
        return {"mu_A": mu_A, "mu_B": mu_B, "R": PB.T @ Rk @ PA, "s": sk}

    if mode == "pls":
        U, _, Vt = sla.svd(Bc.T @ Ac, full_matrices=False, check_finite=False)
        Uk = U[:, :k]
        Vtk = Vt[:k, :]
        Rk = Uk @ Vtk
        # Spread-preserving scale inside projected rank-k subspace
        Ak_norm = float(np.linalg.norm(Ac @ Vtk.T, "fro"))
        Bk_norm = float(np.linalg.norm(Bc @ Uk, "fro"))
        sk = Bk_norm / max(Ak_norm, 1e-12)
        return {"mu_A": mu_A, "mu_B": mu_B, "R": Rk, "s": sk}

    if mode == "fisher_pls":
        assert F_B is not None
        dB = F_B.shape[0]
        F_sym = 0.5 * (F_B + F_B.T)
        tr_F = max(float(np.trace(F_sym)) / dB, 1e-12)
        F_norm = F_sym / tr_F
        F_half = _sym_eigh_power(F_norm, +0.5, shrinkage_frac=fisher_damp)
        F_inv_half = _sym_eigh_power(F_norm, -0.5, shrinkage_frac=fisher_damp)
        # Solve rank-k PLS inside Fisher-whitened target space F^{1/2} B_c
        Bf = Bc @ F_half
        U, _, Vt = sla.svd(Bf.T @ Ac, full_matrices=False, check_finite=False)
        Uk = U[:, :k]
        Vtk = Vt[:k, :]
        # Map back to ambient B space: W = F^{-1/2} Uk Vtk (or pure orthogonal projection onto F^{-1/2} Uk)
        # Keep exact scale match on training target norm inside the subspace
        W_raw = (F_inv_half @ Uk) @ Vtk
        pred_tr = Ac @ W_raw.T
        s_m = float(np.linalg.norm(Bc @ (F_half @ Uk), "fro") / max(np.linalg.norm(pred_tr @ F_half, "fro"), 1e-12))
        return {"mu_A": mu_A, "mu_B": mu_B, "W": s_m * W_raw}

    raise ValueError(f"Unknown mode: {mode}")


# ======================================================================================
# 4. Low-DOF SO(r) [r=32: 496 DOF, r=64: 2016 DOF] MSE vs Causal-KL Trainer
# ======================================================================================

def fit_low_dof_so_r(
    model_B_f32: Any,
    tok_B: Any,
    R0: np.ndarray,
    mu_A: np.ndarray,
    mu_B: np.ndarray,
    s_scale: float,
    U_basis: np.ndarray,
    Z_A_train: np.ndarray,
    Z_B_train: np.ndarray,
    prompts_train: List[str],
    subj_pos_train: List[int],
    seq_lens_train: List[int],
    layer_B: int,
    device: torch.device,
    loss_type: str,  # "mse" or "kl"
    n_steps: int = 80,
    batch_size: int = 16,
    lr: float = 0.08,
    val_pack: Optional[dict] = None,
    seed: int = SEED,
) -> Tuple[dict, dict]:
    """
    Parameterize R(A) = [I + U (Q(A) - I) U^T] R_0  with exact Cayley Q(A) = (I - A)(I + A)^{-1} in SO(r),
    A = 0.5*(W - W^T) skew-symmetric (strictly r*(r-1)/2 DOF: 496 for r=32, 2016 for r=64!).
    R_0, mu_A, mu_B, s_scale, and U_basis are 100% FROZEN!
    Compares MSE vs Causal-KL inside the EXACT same low-DOF SO(r) parameter family!
    """
    r = U_basis.shape[1]
    dof = r * (r - 1) // 2
    n_tr = len(Z_A_train)
    assert len(Z_B_train) == n_tr

    # Pre-transform z_base_hat = mu_B + s * (z_A - mu_A) @ R_0^T  (frozen!)
    # Notice: mu_B + s * R(A) (z_A - mu_A)
    #   = mu_B + (I + U (Q(A) - I) U^T) [s * R_0 (z_A - mu_A)]
    #   = z_0_hat + U (Q(A) - I) (U^T d_0)
    # where d_0 = s * R_0 (z_A - mu_A)! That's O(r * d) per batch and numerically ultra-clean!
    d0_tr = s_scale * ((Z_A_train.astype(np.float64) - mu_A) @ R0.T)  # [N, D]
    z0_tr = mu_B + d0_tr
    proj0_tr = d0_tr @ U_basis  # [N, r]
    perp0_tr = z0_tr - proj0_tr @ U_basis.T  # [N, D], component orthogonal to U + mu_B!

    U_t = torch.from_numpy(U_basis.astype(np.float32)).to(device)  # [D, r]
    I_r = torch.eye(r, dtype=torch.float32, device=device)
    W_param = torch.zeros((r, r), dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([W_param], lr=lr)

    def _get_Q() -> torch.Tensor:
        A = 0.5 * (W_param - W_param.T)
        # Cayley map Q = (I - A)(I + A)^{-1} in SO(r) (exact orthogonality!)
        return torch.linalg.solve(I_r + A, I_r - A)

    if loss_type == "mse":
        # Exact MSE inside SO(r): min_{Q in SO(r)} || (perp0 + proj0 @ Q^T @ U^T) - Z_B ||_F^2
        # Since U has orthonormal columns, || U (Q p - b_u) ||_2^2 = || Q p - b_u ||_2^2!
        # Wait: solve BOTH via Adam (matching KL budget) AND verify exact closed-form SO(r) Procrustes!
        b_u_tr = (Z_B_train.astype(np.float64) - mu_B) @ U_basis  # [N, r]
        p_t = torch.from_numpy(proj0_tr.astype(np.float32)).to(device)
        b_t = torch.from_numpy(b_u_tr.astype(np.float32)).to(device)
        losses = []
        for step in range(n_steps):
            opt.zero_grad()
            Q = _get_Q()
            loss = F.mse_loss(p_t @ Q.T, b_t)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        Q_np = _get_Q().detach().cpu().to(torch.float64).numpy()
        # Verify orthogonality
        orth_err = float(np.linalg.norm(Q_np.T @ Q_np - np.eye(r), "fro"))
        R_final = (np.eye(R0.shape[0]) + U_basis @ (Q_np - np.eye(r)) @ U_basis.T) @ R0
        mp = {"mu_A": mu_A, "mu_B": mu_B, "R": R_final, "s": s_scale, "Q_sub": Q_np}
        diag = {
            "loss_type": "mse",
            "r": r,
            "dof": dof,
            "n_steps": n_steps,
            "init_loss": round(losses[0], 6),
            "final_loss": round(losses[-1], 6),
            "orth_err": orth_err,
            "Q_displacement_fro": round(float(np.linalg.norm(Q_np - np.eye(r), "fro")), 4),
        }
        return mp, diag

    # loss_type == "kl": optimize D_KL(p_B(. | z_B_target) || p_B(. | z_hat(Q))) on MPS float32!
    layers = get_decoder_layers(model_B_f32)
    target_layer = layers[layer_B]

    # Pre-tokenize ALL training batches AND precompute target teacher log-probs p_B(. | z_B_train) ONCE!
    # That saves 50% of forward passes during training!
    rng = np.random.default_rng(seed)
    pre_batches = []
    ctx_hook: Dict[str, Any] = {"active": False, "pos": None, "z_patch": None}

    def patch_hook(_mod, _inp, out):
        if not ctx_hook["active"]:
            return out
        is_tup = isinstance(out, tuple)
        h = out[0] if is_tup else out
        bsz = h.shape[0]
        b_ar = torch.arange(bsz, device=h.device)
        h_new = h.clone()
        h_new[b_ar, ctx_hook["pos"], :] = ctx_hook["z_patch"].to(device=h.device, dtype=h.dtype)
        return (h_new,) + out[1:] if is_tup else h_new

    handle = target_layer.register_forward_hook(patch_hook)
    try:
        # 1. Precompute teacher log-probs at Z_B_train (exact self-swap target!)
        for start in range(0, n_tr, batch_size):
            end = min(start + batch_size, n_tr)
            b_prompts = prompts_train[start:end]
            enc = tok_B(b_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            t_pad = input_ids.shape[1]
            pos = []
            for i in range(start, end):
                pad_len = t_pad - seq_lens_train[i]
                pos.append(max(0, min(t_pad - 1, pad_len + subj_pos_train[i])))
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)

            ctx_hook["pos"] = pos_t
            ctx_hook["z_patch"] = torch.from_numpy(Z_B_train[start:end].astype(np.float32)).to(device)
            ctx_hook["active"] = True
            with torch.no_grad():
                out_t = model_B_f32(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                lp_teacher = F.log_softmax(out_t.logits[:, -1, :].float(), dim=-1)
                # Store top-256 probabilities + tail mass? Actually full [B, V] on CPU or MPS for N=256-512
                # is only 512 * 49152 * 2 bytes = 50 MB in fp16! Exact full-vocab KL!
                lp_teacher_fp16 = lp_teacher.to(torch.float16).cpu()
            ctx_hook["active"] = False
            pre_batches.append({
                "input_ids": input_ids.cpu(),
                "attention_mask": attention_mask.cpu(),
                "pos": pos_t.cpu(),
                "proj0": torch.from_numpy(proj0_tr[start:end].astype(np.float32)),
                "perp0": torch.from_numpy(perp0_tr[start:end].astype(np.float32)),
                "lp_teacher_fp16": lp_teacher_fp16,
            })

        # Evaluate initial KL at Q = I on first 4 batches (and val_pack if provided)
        def _eval_kl_on_batches(batches_list: List[dict], Q_mat: torch.Tensor) -> float:
            tot_kl = 0.0
            tot_n = 0
            with torch.no_grad():
                for b in batches_list:
                    p0 = b["proj0"].to(device)
                    perp0 = b["perp0"].to(device)
                    z_hat = perp0 + (p0 @ Q_mat.T) @ U_t.T
                    ctx_hook["pos"] = b["pos"].to(device)
                    ctx_hook["z_patch"] = z_hat
                    ctx_hook["active"] = True
                    out_s = model_B_f32(
                        input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device),
                        use_cache=False,
                    )
                    ctx_hook["active"] = False
                    lp_s = F.log_softmax(out_s.logits[:, -1, :].float(), dim=-1)
                    lp_t = b["lp_teacher_fp16"].to(device=device, dtype=torch.float32)
                    p_t = torch.exp(lp_t)
                    kl = (p_t * (lp_t - lp_s)).sum(dim=-1)
                    tot_kl += float(kl.sum().item())
                    tot_n += len(kl)
            return tot_kl / max(tot_n, 1)

        init_train_kl = _eval_kl_on_batches(pre_batches[: min(8, len(pre_batches))], I_r)

        # Optional validation early-stopping batches (strictly unlabeled or train-val, NEVER test!)
        val_batches = None
        best_val_kl = float("inf")
        best_Q_np = np.eye(r, dtype=np.float64)
        if val_pack is not None:
            val_batches = []
            z_a_v = val_pack["Z_A"]
            z_b_v = val_pack["Z_B"]
            d0_v = s_scale * ((z_a_v.astype(np.float64) - mu_A) @ R0.T)
            p0_v = d0_v @ U_basis
            perp0_v = (mu_B + d0_v) - p0_v @ U_basis.T
            nv = len(z_a_v)
            for start in range(0, nv, batch_size):
                end = min(start + batch_size, nv)
                enc = tok_B(val_pack["prompts"][start:end], return_tensors="pt", padding=True, add_special_tokens=True)
                ids = enc["input_ids"].to(device)
                am = enc["attention_mask"].to(device)
                t_pad = ids.shape[1]
                pos = [max(0, min(t_pad - 1, (t_pad - val_pack["seq_lens"][i]) + val_pack["subj_pos"][i])) for i in range(start, end)]
                pos_t = torch.tensor(pos, dtype=torch.long, device=device)
                ctx_hook["pos"] = pos_t
                ctx_hook["z_patch"] = torch.from_numpy(z_b_v[start:end].astype(np.float32)).to(device)
                ctx_hook["active"] = True
                with torch.no_grad():
                    out_v = model_B_f32(input_ids=ids, attention_mask=am, use_cache=False)
                    lp_v = F.log_softmax(out_v.logits[:, -1, :].float(), dim=-1).to(torch.float16).cpu()
                ctx_hook["active"] = False
                val_batches.append({
                    "input_ids": ids.cpu(),
                    "attention_mask": am.cpu(),
                    "pos": pos_t.cpu(),
                    "proj0": torch.from_numpy(p0_v[start:end].astype(np.float32)),
                    "perp0": torch.from_numpy(perp0_v[start:end].astype(np.float32)),
                    "lp_teacher_fp16": lp_v,
                })
            best_val_kl = _eval_kl_on_batches(val_batches, I_r)

        # 2. Mini-batch SGD/Adam on SO(r) Cayley parameter A
        n_batches = len(pre_batches)
        step_losses = []
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_steps, 1), eta_min=lr * 0.1)
        for step in range(n_steps):
            b = pre_batches[int(rng.integers(0, n_batches))]
            opt.zero_grad()
            Q = _get_Q()
            p0 = b["proj0"].to(device)
            perp0 = b["perp0"].to(device)
            z_hat = perp0 + (p0 @ Q.T) @ U_t.T

            ctx_hook["pos"] = b["pos"].to(device)
            ctx_hook["z_patch"] = z_hat
            ctx_hook["active"] = True
            with torch.enable_grad():
                out_s = model_B_f32(
                    input_ids=b["input_ids"].to(device),
                    attention_mask=b["attention_mask"].to(device),
                    use_cache=False,
                )
                ctx_hook["active"] = False
                lp_s = F.log_softmax(out_s.logits[:, -1, :].float(), dim=-1)
                lp_t = b["lp_teacher_fp16"].to(device=device, dtype=torch.float32)
                p_t = torch.exp(lp_t)
                kl_loss = (p_t * (lp_t - lp_s)).sum(dim=-1).mean()
                kl_loss.backward()
            torch.nn.utils.clip_grad_norm_([W_param], max_norm=2.0)
            opt.step()
            scheduler.step()
            step_losses.append(float(kl_loss.item()))

            if val_batches is not None and ((step + 1) % max(1, n_steps // 4) == 0 or step == n_steps - 1):
                vkl = _eval_kl_on_batches(val_batches, _get_Q().detach())
                if vkl < best_val_kl:
                    best_val_kl = vkl
                    best_Q_np = _get_Q().detach().cpu().to(torch.float64).numpy()

        final_train_kl = _eval_kl_on_batches(pre_batches[: min(8, len(pre_batches))], _get_Q().detach())
        Q_np = best_Q_np if val_batches is not None else _get_Q().detach().cpu().to(torch.float64).numpy()
    finally:
        handle.remove()

    orth_err = float(np.linalg.norm(Q_np.T @ Q_np - np.eye(r), "fro"))
    R_final = (np.eye(R0.shape[0]) + U_basis @ (Q_np - np.eye(r)) @ U_basis.T) @ R0
    mp = {"mu_A": mu_A, "mu_B": mu_B, "R": R_final, "s": s_scale, "Q_sub": Q_np}
    diag = {
        "loss_type": "kl",
        "r": r,
        "dof": dof,
        "n_steps": n_steps,
        "init_train_kl": round(init_train_kl, 5),
        "final_train_kl": round(final_train_kl, 5),
        "best_val_kl": round(best_val_kl, 5) if val_batches is not None else None,
        "orth_err": orth_err,
        "Q_displacement_fro": round(float(np.linalg.norm(Q_np - np.eye(r), "fro")), 4),
    }
    return mp, diag


# ======================================================================================
# 5. Evaluation Suite: CF-Flip (Matched vs Shuffled) + LORO + Unlabeled KL + Subject CIs
# ======================================================================================

def subject_clustered_bootstrap_ci(
    outcomes_matched: Sequence[int],
    outcomes_shuffled: Sequence[int],
    donor_subjects: Sequence[str],
    n_boot: int = 2000,
    seed: int = SEED,
) -> dict:
    """
    Subject-clustered bootstrap 95% CIs (clustering by donor subject_norm,
    so multiple facts with the same subject are resampled together!).
    """
    m = np.asarray(outcomes_matched, dtype=np.float64)
    s = np.asarray(outcomes_shuffled, dtype=np.float64)
    n = len(m)
    if n == 0:
        return {"n": 0}

    by_subj: Dict[str, List[int]] = defaultdict(list)
    for i, subj in enumerate(donor_subjects):
        by_subj[subj].append(i)
    clusters = list(by_subj.values())
    n_c = len(clusters)

    rng = np.random.default_rng(seed)
    boot_m = np.empty(n_boot, dtype=np.float64)
    boot_d = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draw = rng.integers(0, n_c, size=n_c)
        idxs = [i for c_id in draw for i in clusters[c_id]]
        bm = float(np.mean(m[idxs]))
        bs = float(np.mean(s[idxs]))
        boot_m[b] = bm
        boot_d[b] = bm - bs

    m_lo, m_hi = np.percentile(boot_m, [2.5, 97.5])
    d_lo, d_hi = np.percentile(boot_d, [2.5, 97.5])
    return {
        "n": int(n),
        "n_unique_subjects": int(n_c),
        "matched_acc": round(float(np.mean(m)), 4),
        "matched_ci95": [round(float(m_lo), 4), round(float(m_hi), 4)],
        "shuffled_acc": round(float(np.mean(s)), 4),
        "matched_minus_shuffled_pp": round(100.0 * float(np.mean(m - s)), 2),
        "diff_ci95_pp": [round(100.0 * float(d_lo), 2), round(100.0 * float(d_hi), 2)],
    }


def eval_unlabeled_continuation_kl(
    model_B: Any,
    tok_B: Any,
    val_items: List[dict],
    Z_A_val: np.ndarray,
    Z_B_val: np.ndarray,
    mp: dict,
    layer_B: int,
    device: torch.device,
    dtype: torch.dtype,
    teacher_lp_cache: Optional[List[torch.Tensor]] = None,
    batch_size: int = 32,
) -> Tuple[float, List[torch.Tensor]]:
    """
    Evaluate mean D_KL(p_B(. | z_B_true) || p_B(. | z_hat_B)) on unseen unlabeled entity continuations.
    Caches teacher log-probs on first call for speed.
    """
    layers = get_decoder_layers(model_B)
    target_layer = layers[layer_B]
    z_hat = apply_affine_map(Z_A_val, mp).astype(np.float32)
    n = len(val_items)

    ctx: Dict[str, Any] = {"active": False, "pos": None, "z_patch": None}

    def hook_fn(_mod, _inp, out):
        if not ctx["active"]:
            return out
        is_tup = isinstance(out, tuple)
        h = out[0].clone() if is_tup else out.clone()
        bsz = h.shape[0]
        b_ar = torch.arange(bsz, device=h.device)
        h[b_ar, ctx["pos"], :] = ctx["z_patch"].to(device=h.device, dtype=h.dtype)
        return (h,) + out[1:] if is_tup else h

    handle = target_layer.register_forward_hook(hook_fn)
    build_cache = teacher_lp_cache is None
    new_cache: List[torch.Tensor] = [] if build_cache else teacher_lp_cache
    tot_kl = 0.0
    tot_n = 0
    try:
        for b_idx, start in enumerate(range(0, n, batch_size)):
            end = min(start + batch_size, n)
            prompts = [val_items[i]["prompt"] for i in range(start, end)]
            enc = tok_B(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
            ids = enc["input_ids"].to(device)
            am = enc["attention_mask"].to(device)
            t_pad = ids.shape[1]
            pos = [max(0, min(t_pad - 1, (t_pad - val_items[i]["seq_len_B"]) + val_items[i]["tok_idx_B"])) for i in range(start, end)]
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)

            if build_cache:
                ctx["pos"] = pos_t
                ctx["z_patch"] = torch.from_numpy(Z_B_val[start:end].astype(np.float32)).to(device)
                ctx["active"] = True
                with torch.no_grad():
                    ot = model_B(input_ids=ids, attention_mask=am, use_cache=False)
                    lp_t = F.log_softmax(ot.logits[:, -1, :].float(), dim=-1)
                ctx["active"] = False
                new_cache.append(lp_t.to(torch.float16).cpu())
            else:
                lp_t = new_cache[b_idx].to(device=device, dtype=torch.float32)

            ctx["pos"] = pos_t
            ctx["z_patch"] = torch.from_numpy(z_hat[start:end]).to(device)
            ctx["active"] = True
            with torch.no_grad():
                os_out = model_B(input_ids=ids, attention_mask=am, use_cache=False)
                lp_s = F.log_softmax(os_out.logits[:, -1, :].float(), dim=-1)
            ctx["active"] = False

            p_t = torch.exp(lp_t)
            kl = (p_t * (lp_t - lp_s)).sum(dim=-1)
            tot_kl += float(kl.sum().item())
            tot_n += len(kl)
    finally:
        handle.remove()

    return tot_kl / max(tot_n, 1), new_cache


def evaluate_map_full(
    name: str,
    mp: dict,
    model_B: Any,
    tok_B: Any,
    facts: List[dict],
    eval_B: List[dict],
    Z_A_facts: np.ndarray,
    Z_B_facts: np.ndarray,
    test_pairs: List[Tuple[int, int]],
    test_pretok: List[dict],
    loro_pairs: List[Tuple[int, int]],
    loro_pretok: List[dict],
    shuf_test_donor_idxs: List[int],
    shuf_loro_donor_idxs: List[int],
    ad_test_mask: np.ndarray,
    F_B_eval: np.ndarray,
    val_items_kl: List[dict],
    Z_A_val_kl: np.ndarray,
    Z_B_val_kl: np.ndarray,
    teacher_lp_cache: Optional[List[torch.Tensor]],
    layer_B: int,
    device: torch.device,
    dtype: torch.dtype,
    n_boot: int = 2000,
) -> Tuple[dict, List[torch.Tensor]]:
    """
    Comprehensive zero-GlobalFV evaluation of a candidate map:
      - CF-Flip test (matched vs shuffled donor + answer-disjoint + subject-clustered 95% CI)
      - LORO 5 unseen relation families (matched vs shuffled donor + subject-clustered 95% CI)
      - Held-out geometry (centered cosine, R^2, Fisher-weighted MSE)
      - Unlabeled continuation KL guardrail
    """
    # 1. Test CF-Flip (matched + shuffled donor)
    t_d_idxs = [d for d, _ in test_pairs]
    z_hat_te = apply_affine_map(Z_A_facts[t_d_idxs], mp)
    z_hat_te_shuf = apply_affine_map(Z_A_facts[shuf_test_donor_idxs], mp)
    _, out_te_m = eval_patched_swaps(
        model_B, tok_B, facts, eval_B, test_pairs, z_hat_te, layer_B, device, dtype, pretok_batches=test_pretok
    )
    _, out_te_s = eval_patched_swaps(
        model_B, tok_B, facts, eval_B, test_pairs, z_hat_te_shuf, layer_B, device, dtype, pretok_batches=test_pretok
    )
    te_subjs = [facts[d]["subject_norm"] for d, _ in test_pairs]
    te_stats = subject_clustered_bootstrap_ci(out_te_m, out_te_s, te_subjs, n_boot=n_boot)

    # Answer-disjoint slice
    ad_idxs = [i for i, flag in enumerate(ad_test_mask) if flag]
    te_ad_stats = subject_clustered_bootstrap_ci(
        [out_te_m[i] for i in ad_idxs],
        [out_te_s[i] for i in ad_idxs],
        [te_subjs[i] for i in ad_idxs],
        n_boot=n_boot,
    )

    # 2. LORO 5 unseen relation families (matched + shuffled donor)
    l_d_idxs = [d for d, _ in loro_pairs]
    z_hat_lo = apply_affine_map(Z_A_facts[l_d_idxs], mp)
    z_hat_lo_shuf = apply_affine_map(Z_A_facts[shuf_loro_donor_idxs], mp)
    _, out_lo_m = eval_patched_swaps(
        model_B, tok_B, facts, eval_B, loro_pairs, z_hat_lo, layer_B, device, dtype, pretok_batches=loro_pretok
    )
    _, out_lo_s = eval_patched_swaps(
        model_B, tok_B, facts, eval_B, loro_pairs, z_hat_lo_shuf, layer_B, device, dtype, pretok_batches=loro_pretok
    )
    lo_subjs = [facts[d]["subject_norm"] for d, _ in loro_pairs]
    lo_stats = subject_clustered_bootstrap_ci(out_lo_m, out_lo_s, lo_subjs, n_boot=n_boot)

    # 3. Held-out geometry on test donor facts (centered cosine, R^2, Fisher-weighted error)
    zb_te = Z_B_facts[t_d_idxs].astype(np.float64)
    mu_ref = zb_te.mean(axis=0)
    zb_c = zb_te - mu_ref
    zh_c = z_hat_te - mu_ref
    cos_c = float(np.mean(
        np.sum(zb_c * zh_c, axis=1)
        / np.maximum(np.linalg.norm(zb_c, axis=1) * np.linalg.norm(zh_c, axis=1), 1e-12)
    ))
    delta = z_hat_te - zb_te
    r2 = 1.0 - float(np.sum(delta ** 2)) / max(float(np.sum(zb_c ** 2)), 1e-12)
    # Normalized Fisher-weighted MSE: E[delta^T F_B delta] / E[z_c^T F_B z_c]
    tr_f_err = float(np.sum((delta @ F_B_eval) * delta))
    tr_f_tot = float(np.sum((zb_c @ F_B_eval) * zb_c))
    fisher_r2 = 1.0 - tr_f_err / max(tr_f_tot, 1e-12)

    # 4. Unlabeled held-out continuation KL guardrail
    unlab_kl, new_cache = eval_unlabeled_continuation_kl(
        model_B, tok_B, val_items_kl, Z_A_val_kl, Z_B_val_kl, mp,
        layer_B, device, dtype, teacher_lp_cache=teacher_lp_cache,
    )

    res = {
        "name": name,
        "cf_flip_test": te_stats,
        "cf_flip_test_answer_disjoint": te_ad_stats,
        "loro_5_unseen_relations": lo_stats,
        "geometry_test": {
            "centered_cosine": round(cos_c, 5),
            "r2_euclidean": round(r2, 5),
            "r2_fisher": round(fisher_r2, 5),
            "mean_l2_err": round(float(np.mean(np.linalg.norm(delta, axis=1))), 5),
        },
        "unlabeled_val_continuation_kl": round(unlab_kl, 5),
    }
    print(
        f"  [{name:32s}] CF-Flip={te_stats['matched_acc']:.3f} (shuf={te_stats['shuffled_acc']:.3f}, "
        f"net={te_stats['matched_minus_shuffled_pp']:+5.1f}pp, CI={te_stats['diff_ci95_pp']}) | "
        f"AD={te_ad_stats['matched_acc']:.3f} | LORO={lo_stats['matched_acc']:.3f} "
        f"(net={lo_stats['matched_minus_shuffled_pp']:+5.1f}pp) | "
        f"cos={cos_c:.3f}, R2_F={fisher_r2:.3f}, unlabKL={unlab_kl:.3f}"
    )
    return res, new_cache


def make_same_relation_shuffled_donors(
    facts: List[dict], swap_pairs: List[Tuple[int, int]], seed: int = SEED
) -> List[int]:
    """
    For each swap pair (d_idx, b_idx), deterministically select a WRONG donor d_shuf
    from the SAME relation with d_shuf != d_idx, d_shuf != b_idx, and answer != donor_answer.
    """
    rng = np.random.default_rng(seed)
    by_rel: Dict[str, List[int]] = defaultdict(list)
    for d, _ in swap_pairs:
        by_rel[facts[d]["relation"]].append(d)

    out = []
    for d, b in swap_pairs:
        rel = facts[d]["relation"]
        d_ans = facts[d]["answer"].strip().lower()
        cands = [
            x for x in by_rel[rel]
            if x != d and x != b and facts[x]["answer"].strip().lower() != d_ans
        ]
        if not cands:
            cands = [x for x in by_rel[rel] if x != d]
        if not cands:
            cands = [d]
        out.append(int(cands[int(rng.integers(0, len(cands)))]))
    return out


# ======================================================================================
# 6. Main Pilot 1 Pipeline
# ======================================================================================

def main():
    parser = argparse.ArgumentParser(description="Pilot 1: Unlabeled N>>D + Fisher Pullback + SO(32)/SO(64)")
    parser.add_argument("--pair", default="qwen2.5-3b->smollm2-1.7b")
    parser.add_argument("--smoke", action="store_true", help="Fast end-to-end smoke test")
    args = parser.parse_args()

    t_start = time.time()
    pair_key = args.pair
    mk_A, mk_B = pair_key.split("->")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dataset_2hop = None
    if DATASET_2HOP_PATH.exists():
        with open(DATASET_2HOP_PATH, "r", encoding="utf-8") as f:
            dataset_2hop = json.load(f)
    with open(BASELINE_RESULTS_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    with open(CACHE_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(CACHE_DIR / f"{mk_A}_eval.json", "r", encoding="utf-8") as f:
        eval_A_full = json.load(f)
    with open(CACHE_DIR / f"{mk_B}_eval.json", "r", encoding="utf-8") as f:
        eval_B_full = json.load(f)

    facts = dataset["facts"]
    eval_A = eval_A_full["facts"]
    eval_B = eval_B_full["facts"]
    bp = baseline["pairs"][pair_key]
    layer_A = bp["best_layer_A"]
    layer_B = bp["best_layer_B"]

    has_mps = torch.backends.mps.is_available()
    device = torch.device("mps") if has_mps else torch.device("cpu")
    eval_dtype = torch.bfloat16 if has_mps else torch.float32

    n_unlab_max = 120 if args.smoke else 10000
    n_unlab_val = 32 if args.smoke else 512
    n_fisher = 48 if args.smoke else 2000
    n_grid = [40, 80, 120] if args.smoke else [500, 2000, 10000]
    n_kl_train = 32 if args.smoke else 512
    so_steps = 8 if args.smoke else 100
    n_boot = 200 if args.smoke else 2000

    print(
        f"[pilot1] pair={pair_key} layers=({layer_A},{layer_B}) device={device} "
        f"smoke={args.smoke} N_grid={n_grid} N_fisher={n_fisher}"
    )

    # Load cached factual states for A and B
    st_A = torch.load(CACHE_DIR / f"{mk_A}_states.pt", map_location="cpu", weights_only=False)
    Z_A_facts = st_A["direct_subj"][:, layer_A, :].float().numpy()
    del st_A
    st_B = torch.load(CACHE_DIR / f"{mk_B}_states.pt", map_location="cpu", weights_only=False)
    Z_B_facts = st_B["direct_subj"][:, layer_B, :].float().numpy()
    del st_B

    # Load tokenizers & build clean unlabeled entity-mention corpus
    dir_A = Path(manifest["models"][mk_A]["local_path"])
    dir_B = Path(manifest["models"][mk_B]["local_path"])
    tok_A = AutoTokenizer.from_pretrained(str(dir_A), local_files_only=True)
    tok_A.padding_side = "left"
    if tok_A.pad_token is None:
        tok_A.pad_token = tok_A.eos_token
    tok_B = AutoTokenizer.from_pretrained(str(dir_B), local_files_only=True)
    tok_B.padding_side = "left"
    if tok_B.pad_token is None:
        tok_B.pad_token = tok_B.eos_token

    cache_file = CACHE_DIR / (
        f"pilot1_unlab_{mk_A}_to_{mk_B}_l{layer_A}_{layer_B}{'_smoke' if args.smoke else ''}.pt"
    )
    if cache_file.exists():
        print(f"[pilot1] Loading cached unlabeled states & Fisher gradients from {cache_file}")
        pack = torch.load(cache_file, map_location="cpu", weights_only=False)
        train_items = pack["train_items"]
        val_items = pack["val_items"]
        corpus_meta = pack["corpus_meta"]
        Z_A_unlab_tr = pack["Z_A_unlab_tr"].numpy()
        Z_A_unlab_val = pack["Z_A_unlab_val"].numpy()
        Z_B_unlab_tr = pack["Z_B_unlab_tr"].numpy()
        Z_B_unlab_val = pack["Z_B_unlab_val"].numpy()
        G_B_end = pack["G_B_end"].numpy()
        G_B_multi = pack["G_B_multi"].numpy()
    else:
        print(f"[pilot1] Building leak-free unlabeled entity corpus (N={n_unlab_max} + {n_unlab_val} val)...")
        train_items, val_items, corpus_meta = build_unlabeled_entity_corpus(
            tok_A, tok_B, dataset, dataset_2hop, n_train=n_unlab_max, n_val=n_unlab_val, seed=SEED
        )
        print(f"  corpus_meta: {json.dumps(corpus_meta)}")

        # Extract Donor A unlabeled states (bf16 on MPS) then unload A!
        t_a = time.time()
        print(f"[pilot1] Extracting Donor A ({mk_A}) states at layer {layer_A} on {device}...")
        model_A = AutoModelForCausalLM.from_pretrained(str(dir_A), local_files_only=True, dtype=eval_dtype)
        model_A.to(device).eval()
        Z_A_unlab_tr = extract_unlabeled_layer_states(model_A, tok_A, train_items, layer_A, "A", device, batch_size=64)
        Z_A_unlab_val = extract_unlabeled_layer_states(model_A, tok_A, val_items, layer_A, "A", device, batch_size=64)
        del model_A
        gc.collect()
        if has_mps:
            torch.mps.empty_cache()
        print(f"  Donor A extraction done in {time.time() - t_a:.1f}s")

        # Extract Recipient B unlabeled states (bf16) AND exact float32 unlabeled Causal Fisher!
        t_b = time.time()
        print(f"[pilot1] Extracting Recipient B ({mk_B}) states at layer {layer_B} on {device}...")
        model_B = AutoModelForCausalLM.from_pretrained(str(dir_B), local_files_only=True, dtype=eval_dtype)
        model_B.to(device).eval()
        Z_B_unlab_tr = extract_unlabeled_layer_states(model_B, tok_B, train_items, layer_B, "B", device, batch_size=64)
        Z_B_unlab_val = extract_unlabeled_layer_states(model_B, tok_B, val_items, layer_B, "B", device, batch_size=64)
        print(f"  Recipient B forward states done in {time.time() - t_b:.1f}s")

        t_f = time.time()
        print(f"[pilot1] Computing exact UNLABELED Causal Fisher (end + multi-step, N={n_fisher}) in float32...")
        if eval_dtype != torch.float32:
            model_B = model_B.to(dtype=torch.float32)
        for p in model_B.parameters():
            p.requires_grad_(False)
        model_B.eval()
        G_B_end, G_B_multi = extract_unlabeled_causal_fisher(
            model_B, tok_B, train_items[:n_fisher], layer_B, device, batch_size=16, seed=SEED
        )
        del model_B
        gc.collect()
        if has_mps:
            torch.mps.empty_cache()
        print(f"  Unlabeled Causal Fisher extraction done in {time.time() - t_f:.1f}s")

        torch.save(
            {
                "train_items": train_items,
                "val_items": val_items,
                "corpus_meta": corpus_meta,
                "Z_A_unlab_tr": torch.from_numpy(Z_A_unlab_tr),
                "Z_A_unlab_val": torch.from_numpy(Z_A_unlab_val),
                "Z_B_unlab_tr": torch.from_numpy(Z_B_unlab_tr),
                "Z_B_unlab_val": torch.from_numpy(Z_B_unlab_val),
                "G_B_end": torch.from_numpy(G_B_end),
                "G_B_multi": torch.from_numpy(G_B_multi),
            },
            cache_file,
        )

    # Form empirical unlabeled Fisher matrices in float64 on CPU
    F_B_end = (G_B_end.astype(np.float64).T @ G_B_end.astype(np.float64)) / max(len(G_B_end), 1)
    F_B_multi = (G_B_multi.astype(np.float64).T @ G_B_multi.astype(np.float64)) / max(len(G_B_multi), 1)

    def _fisher_eff_rank(F_mat: np.ndarray) -> dict:
        w = sla.eigh(0.5 * (F_mat + F_mat.T), eigvals_only=True, check_finite=False)
        w = np.maximum(w[::-1], 0.0)
        tr = float(np.sum(w))
        pr = float((tr ** 2) / max(float(np.sum(w ** 2)), 1e-30))
        p = w / max(tr, 1e-30)
        p = p[p > 1e-18]
        sh = float(math.exp(-float(np.sum(p * np.log(p)))))
        cum = np.cumsum(w) / max(tr, 1e-30)
        return {
            "eff_rank_pr": round(pr, 2),
            "eff_rank_shannon": round(sh, 2),
            "top32_frac": round(float(cum[min(31, len(cum) - 1)]), 4),
            "top64_frac": round(float(cum[min(63, len(cum) - 1)]), 4),
            "top256_frac": round(float(cum[min(255, len(cum) - 1)]), 4),
        }

    fisher_summary = {
        "F_B_end_unlab": _fisher_eff_rank(F_B_end),
        "F_B_multi_unlab": _fisher_eff_rank(F_B_multi),
    }
    print(f"[pilot1] Unlabeled Fisher spectra: {json.dumps(fisher_summary)}")

    # Prepare evaluation sets:
    #   (a) Standard held-out test CF-Flip (exact test_pairs from results_baseline.json)
    test_pairs = [tuple(x) for x in bp["test_pairs"]]
    #   (b) LORO 5 completely unseen relation families (all jointly-recalled test+val facts in those 5 relations;
    #       for strict zero-leakage against train-10 models, use ALL jointly-recalled facts in those 5 rels across val+test!)
    shared_loro_eval = [
        f["idx"] for f in facts
        if f["relation"] in LORO_HELD_OUT_RELS
        and f["split"] in ("val", "test")
        and eval_A[f["idx"]]["direct_correct"]
        and eval_B[f["idx"]]["direct_correct"]
    ]
    loro_pairs = build_counterfactual_pairs(facts, shared_loro_eval)
    if args.smoke:
        test_pairs = test_pairs[:12]
        loro_pairs = loro_pairs[:12]

    all_train_indices = [f["idx"] for f in facts if f["split"] == "train"]
    train_answer_set = {
        a.strip().lower() for i in all_train_indices for a in facts[i]["aliases"]
    }
    ad_test_mask = np.array(
        [all(a.strip().lower() not in train_answer_set for a in facts[d]["aliases"]) for d, _ in test_pairs],
        dtype=bool,
    )
    shuf_test_donors = make_same_relation_shuffled_donors(facts, test_pairs, seed=SEED + 11)
    shuf_loro_donors = make_same_relation_shuffled_donors(facts, loro_pairs, seed=SEED + 22)

    # Load Recipient B ONCE for all evaluations + low-DOF SO(32)/SO(64) training!
    print(f"\n[pilot1] Loading Recipient B ({mk_B}) on {device} (eval_dtype={eval_dtype})...")
    model_B = AutoModelForCausalLM.from_pretrained(str(dir_B), local_files_only=True, dtype=eval_dtype)
    model_B.to(device).eval()

    test_pretok = pretokenize_swap_batches(tok_B, facts, eval_B, test_pairs, device, batch_size=32)
    loro_pretok = pretokenize_swap_batches(tok_B, facts, eval_B, loro_pairs, device, batch_size=32)

    val_items_kl = val_items[: (24 if args.smoke else 256)]
    Z_A_val_kl = Z_A_unlab_val[: len(val_items_kl)]
    Z_B_val_kl = Z_B_unlab_val[: len(val_items_kl)]
    teacher_lp_cache: Optional[List[torch.Tensor]] = None

    results: Dict[str, Any] = {
        "meta": {
            "pair_key": pair_key,
            "layers": [layer_A, layer_B],
            "device": str(device),
            "smoke": args.smoke,
            "seed": SEED,
            "n_grid": n_grid,
            "n_fisher": n_fisher,
            "loro_held_out_relations": sorted(LORO_HELD_OUT_RELS),
            "n_test_pairs": len(test_pairs),
            "n_test_ad_pairs": int(ad_test_mask.sum()),
            "n_loro_pairs": len(loro_pairs),
        },
        "corpus_meta": corpus_meta,
        "unlabeled_fisher_spectra": fisher_summary,
        "qa_train_reference": {},
        "unlabeled_scaling_ladder": {},
        "rank_k_compression": {},
        "low_dof_so_comparison": {},
    }

    def _eval_and_record(bucket: dict, key: str, mp: dict) -> dict:
        nonlocal teacher_lp_cache
        r, teacher_lp_cache = evaluate_map_full(
            key, mp, model_B, tok_B, facts, eval_B, Z_A_facts, Z_B_facts,
            test_pairs, test_pretok, loro_pairs, loro_pretok,
            shuf_test_donors, shuf_loro_donors, ad_test_mask, F_B_multi,
            val_items_kl, Z_A_val_kl, Z_B_val_kl, teacher_lp_cache,
            layer_B, device, eval_dtype, n_boot=n_boot,
        )
        bucket[key] = r
        return r

    # ---- A. Round-1 Reference Baselines (QA train facts) ----
    print("\n=== A. Round-1 QA-Train Reference Baselines ===")
    shared_train = list(bp["shared_train_indices"])
    mp_qa_shared = fit_procrustes_global(Z_A_facts[shared_train], Z_B_facts[shared_train])
    _eval_and_record(results["qa_train_reference"], f"qa_shared_procrustes_N{len(shared_train)}", mp_qa_shared)

    mp_qa_all = fit_procrustes_global(Z_A_facts[all_train_indices], Z_B_facts[all_train_indices])
    _eval_and_record(results["qa_train_reference"], f"qa_alltrain_procrustes_N{len(all_train_indices)}", mp_qa_all)

    mp_qa_span = fit_procrustes_span_f64(Z_A_facts[all_train_indices], Z_B_facts[all_train_indices])
    _eval_and_record(results["qa_train_reference"], f"qa_alltrain_span_N{len(all_train_indices)}", mp_qa_span)

    # Also evaluate strict Train-10-Relations QA Procrustes (so LORO is 100% unseen-relation for QA baseline too!)
    train10_indices = [i for i in all_train_indices if facts[i]["relation"] not in LORO_HELD_OUT_RELS]
    mp_qa_tr10 = fit_procrustes_global(Z_A_facts[train10_indices], Z_B_facts[train10_indices])
    _eval_and_record(results["qa_train_reference"], f"qa_train10rels_procrustes_N{len(train10_indices)}", mp_qa_tr10)

    # ---- B. Unlabeled Entity-Span Scaling Ladder N in {500, 2000, 10000} ----
    print("\n=== B. Unlabeled Scaling Ladder + Closed-Form Causal-Fisher Pullback ===")
    mp_unlab_best_proc = None
    mp_unlab_best_fisher = None

    for N in n_grid:
        print(f"\n--- Unlabeled N = {N} ---")
        ZA_n = Z_A_unlab_tr[:N]
        ZB_n = Z_B_unlab_tr[:N]
        b_n = {}

        # 1. Vanilla Procrustes O(d)
        mp_proc = fit_procrustes_global(ZA_n, ZB_n)
        _eval_and_record(b_n, "procrustes_global", mp_proc)
        if N == n_grid[-1]:
            mp_unlab_best_proc = mp_proc

        # 2. Span-Procrustes
        mp_span = fit_procrustes_span_f64(ZA_n, ZB_n)
        _eval_and_record(b_n, "procrustes_span", mp_span)

        # 3. Ridge-GL(d) (raw & rescaled; lambda selected strictly on unlabeled val MSE!)
        mp_ridge_raw, mp_ridge_resc = fit_ridge_gl(ZA_n, ZB_n, Z_A_unlab_val, Z_B_unlab_val)
        r_rr = _eval_and_record(b_n, "ridge_gl_raw", mp_ridge_raw)
        r_rr["selected_lam"] = mp_ridge_raw["lam"]
        r_rs = _eval_and_record(b_n, "ridge_gl_rescaled", mp_ridge_resc)
        r_rs["selected_lam"] = mp_ridge_resc["lam"]

        # Precompute ZCA covariance factors ONCE per shrink_cov (6x speedup!)
        cov_caches = {sc: precompute_zca_cov(ZA_n, ZB_n, shrink_cov=sc) for sc in (0.02, 0.1)}

        # 4. ZCA-Whitened Procrustes (H5 intrinsic conditioning without Fisher)
        best_zca_mp = None
        best_zca_kl = float("inf")
        for sc in (0.02, 0.1):
            cand = fit_zca_and_fisher_pullback(ZA_n, ZB_n, F_B=None, shrink_cov=sc, cov_cache=cov_caches[sc])
            kl_v, teacher_lp_cache = eval_unlabeled_continuation_kl(
                model_B, tok_B, val_items_kl, Z_A_val_kl, Z_B_val_kl, cand,
                layer_B, device, eval_dtype, teacher_lp_cache=teacher_lp_cache,
            )
            if kl_v < best_zca_kl:
                best_zca_kl = kl_v
                best_zca_mp = cand
        r_zca = _eval_and_record(b_n, "zca_whitened_procrustes", best_zca_mp)
        r_zca["selected_shrink_cov"] = best_zca_mp["shrink_cov"]

        # 5. Closed-Form Causal-Fisher Pullback Procrustes (Carbon Theorem! Select gamma strictly on unlabeled KL!)
        for f_name, F_mat in (("fisher_multi", F_B_multi), ("fisher_end", F_B_end)):
            best_f_mp = None
            best_f_kl = float("inf")
            for sc in (0.02, 0.1):
                for gam in (0.01, 0.1, 0.5):
                    cand = fit_zca_and_fisher_pullback(
                        ZA_n, ZB_n, F_B=F_mat, shrink_cov=sc, fisher_damp=gam,
                        diag_fisher_only=False, cov_cache=cov_caches[sc],
                    )
                    kl_v, teacher_lp_cache = eval_unlabeled_continuation_kl(
                        model_B, tok_B, val_items_kl, Z_A_val_kl, Z_B_val_kl, cand,
                        layer_B, device, eval_dtype, teacher_lp_cache=teacher_lp_cache,
                    )
                    if kl_v < best_f_kl:
                        best_f_kl = kl_v
                        best_f_mp = cand
            r_fp = _eval_and_record(b_n, f"pullback_zca_{f_name}", best_f_mp)
            r_fp["selected_shrink_cov"] = best_f_mp["shrink_cov"]
            r_fp["selected_fisher_damp"] = best_f_mp["fisher_damp"]
            if N == n_grid[-1] and f_name == "fisher_multi":
                mp_unlab_best_fisher = best_f_mp

        # 6. Pure Orthogonal O(d) Fisher-Weighted Procrustes + Diagonal Fisher control (at largest N & N=2000)
        best_fo_mp = None
        if N in (n_grid[-1], n_grid[len(n_grid) // 2]):
            best_fo_kl = float("inf")
            for gam in (0.02, 0.1, 0.5):
                cand = fit_fisher_orthogonal_procrustes(ZA_n, ZB_n, F_B_multi, fisher_damp=gam, diag_only=False, n_iter=12)
                kl_v, teacher_lp_cache = eval_unlabeled_continuation_kl(
                    model_B, tok_B, val_items_kl, Z_A_val_kl, Z_B_val_kl, cand,
                    layer_B, device, eval_dtype, teacher_lp_cache=teacher_lp_cache,
                )
                if kl_v < best_fo_kl:
                    best_fo_kl = kl_v
                    best_fo_mp = cand
            r_fo = _eval_and_record(b_n, "fisher_orth_Od_multi", best_fo_mp)
            r_fo["selected_fisher_damp"] = best_fo_mp["fisher_damp"]

            mp_fdiag = fit_zca_and_fisher_pullback(
                ZA_n, ZB_n, F_B=F_B_multi, shrink_cov=0.1, fisher_damp=0.1,
                diag_fisher_only=True, cov_cache=cov_caches[0.1],
            )
            _eval_and_record(b_n, "pullback_zca_diag_fisher", mp_fdiag)

        # 7. At largest N: isolate domain mean shift (unlabeled R/W + train10 mean calibration, 0 LORO/test leakage!)
        #    AND save full-rank N=10k maps to cache so Pilot 2 can immediately reuse them!
        if N == n_grid[-1]:
            muA_t10 = Z_A_facts[train10_indices].astype(np.float64).mean(axis=0)
            muB_t10 = Z_B_facts[train10_indices].astype(np.float64).mean(axis=0)
            for mtag, mbase in (
                ("procrustes_global_t10mean", mp_proc),
                ("zca_whitened_t10mean", best_zca_mp),
                ("pullback_zca_fisher_multi_t10mean", mp_unlab_best_fisher),
                ("fisher_orth_Od_multi_t10mean", best_fo_mp),
            ):
                if mbase is None:
                    continue
                mcal = dict(mbase)
                mcal["mu_A"] = muA_t10
                mcal["mu_B"] = muB_t10
                _eval_and_record(b_n, mtag, mcal)

            if not args.smoke:
                maps_save_path = CACHE_DIR / f"pilot1_maps_{mk_A}_to_{mk_B}_l{layer_A}_{layer_B}.pt"
                torch.save(
                    {
                        "pair_key": pair_key,
                        "layers": [layer_A, layer_B],
                        "N": N,
                        "procrustes_global": {k: mp_proc[k] for k in ("mu_A", "mu_B", "R", "s")},
                        "zca_whitened": {k: best_zca_mp[k] for k in ("mu_A", "mu_B", "W", "s_match", "shrink_cov")},
                        "pullback_zca_fisher_multi": {
                            k: mp_unlab_best_fisher[k]
                            for k in ("mu_A", "mu_B", "W", "s_match", "shrink_cov", "fisher_damp")
                        },
                        "fisher_orth_Od_multi": {
                            k: best_fo_mp[k] for k in ("mu_A", "mu_B", "R", "s", "fisher_damp")
                        } if best_fo_mp is not None else None,
                    },
                    maps_save_path,
                )
                print(f"  [pilot1] Saved full-rank N={N} maps to {maps_save_path}")

        results["unlabeled_scaling_ladder"][str(N)] = b_n

    # ---- C. Rank-k Compression (k=32, 64) at Max Unlabeled N ----
    print(f"\n=== C. Rank-k Compression (k=32, 64) at Unlabeled N={n_grid[-1]} ===")
    ZA_max = Z_A_unlab_tr[: n_grid[-1]]
    ZB_max = Z_B_unlab_tr[: n_grid[-1]]
    for k in (32, 64):
        for mode in ("pca", "pls", "fisher_pls"):
            mp_k = fit_rank_k_compression(ZA_max, ZB_max, k=k, mode=mode, F_B=F_B_multi, fisher_damp=0.05)
            _eval_and_record(results["rank_k_compression"], f"k{k}_{mode}", mp_k)

    # ---- D. Low-DOF SO(32) [496 DOF] & SO(64) [2016 DOF] MSE vs Causal-KL Head-to-Head ----
    print("\n=== D. Low-DOF SO(32) & SO(64) Head-to-Head: MSE vs Causal-KL ===")
    # Cast Recipient B to float32 (frozen weights) for exact low-DOF Cayley SO(r) KL optimization!
    if eval_dtype != torch.float32:
        model_B = model_B.to(dtype=torch.float32)
    for p in model_B.parameters():
        p.requires_grad_(False)
    model_B.eval()
    teacher_lp_cache = None  # refresh cache cleanly in f32

    # Top-r eigenvectors of UNLABELED multi-step Fisher F_B_multi (zero QA labels!)
    w_f, V_f = sla.eigh(0.5 * (F_B_multi + F_B_multi.T), check_finite=False)
    V_f_desc = V_f[:, ::-1]  # [D, D], columns are descending Fisher eigenvectors

    # Unlabeled KL training set (disjoint from val_items_kl!)
    unlab_kl_items = train_items[:n_kl_train]
    Z_A_kl_unlab = Z_A_unlab_tr[:n_kl_train]
    Z_B_kl_unlab = Z_B_unlab_tr[:n_kl_train]
    unlab_val_pack = {
        "Z_A": Z_A_unlab_val[: (16 if args.smoke else 128)],
        "Z_B": Z_B_unlab_val[: (16 if args.smoke else 128)],
        "prompts": [it["prompt"] for it in val_items[: (16 if args.smoke else 128)]],
        "subj_pos": [it["tok_idx_B"] for it in val_items[: (16 if args.smoke else 128)]],
        "seq_lens": [it["seq_len_B"] for it in val_items[: (16 if args.smoke else 128)]],
    }

    # Train-only 10-relations self-swap KL set (strictly excluding ALL 5 LORO held-out relations + test split!)
    tr10_shared = [
        i for i in shared_train if facts[i]["relation"] not in LORO_HELD_OUT_RELS
    ]
    tr10_pairs = build_counterfactual_pairs(facts, tr10_shared)
    if args.smoke:
        tr10_pairs = tr10_pairs[:24]
    tr10_d = [d for d, _ in tr10_pairs]
    tr10_b = [b for _, b in tr10_pairs]
    tr10_prompts = [facts[b]["direct_prompt"] for b in tr10_b]
    tr10_spos = [eval_B[b]["subj_tok_idx_direct"] for b in tr10_b]
    tr10_slen = [eval_B[b]["seq_len_direct"] for b in tr10_b]
    # Val pack for train-10 early stopping: val split facts ONLY from those SAME 10 train relations!
    val10_shared = [
        i for i in bp["shared_val_indices"] if facts[i]["relation"] not in LORO_HELD_OUT_RELS
    ]
    val10_pairs = build_counterfactual_pairs(facts, val10_shared)
    if args.smoke:
        val10_pairs = val10_pairs[:16]
    val10_pack = {
        "Z_A": Z_A_facts[[d for d, _ in val10_pairs]],
        "Z_B": Z_B_facts[[d for d, _ in val10_pairs]],
        "prompts": [facts[b]["direct_prompt"] for _, b in val10_pairs],
        "subj_pos": [eval_B[b]["subj_tok_idx_direct"] for _, b in val10_pairs],
        "seq_lens": [eval_B[b]["seq_len_direct"] for _, b in val10_pairs],
    }

    # Frozen base R_0: use N=10k unlabeled Procrustes (pure zero-QA base!)
    R0_base = mp_unlab_best_proc["R"]
    muA_base = mp_unlab_best_proc["mu_A"]
    muB_base = mp_unlab_best_proc["mu_B"]
    s_base = mp_unlab_best_proc["s"]

    # Also test starting from QA train10 Procrustes for the train10 self-swap comparison!
    R0_qa10 = mp_qa_tr10["R"]
    muA_qa10 = mp_qa_tr10["mu_A"]
    muB_qa10 = mp_qa_tr10["mu_B"]
    s_qa10 = mp_qa_tr10["s"]

    for r_dim in (32, 64):
        U_r = np.ascontiguousarray(V_f_desc[:, :r_dim])
        print(f"\n--- SO({r_dim}) [DOF = {r_dim * (r_dim - 1) // 2}] ---")

        # 1. Strictly UNLABELED MSE vs UNLABELED Continuation KL (frozen unlabeled R0!)
        for loss_t in ("mse", "kl"):
            mp_so, diag_so = fit_low_dof_so_r(
                model_B, tok_B, R0_base, muA_base, muB_base, s_base, U_r,
                Z_A_kl_unlab, Z_B_kl_unlab,
                [it["prompt"] for it in unlab_kl_items],
                [it["tok_idx_B"] for it in unlab_kl_items],
                [it["seq_len_B"] for it in unlab_kl_items],
                layer_B, device, loss_type=loss_t, n_steps=so_steps, batch_size=16,
                lr=(0.015 if loss_t == "kl" else 0.06),
                val_pack=unlab_val_pack if loss_t == "kl" else None, seed=SEED + r_dim,
            )
            key = f"so{r_dim}_{loss_t}_unlab_from_unlabR0"
            rec = _eval_and_record(results["low_dof_so_comparison"], key, mp_so)
            rec["train_diag"] = diag_so

        # 2. Train-10-relations Self-Swap MSE vs KL (frozen unlabeled R0, 5 LORO relations 100% unseen!)
        for loss_t in ("mse", "kl"):
            mp_so, diag_so = fit_low_dof_so_r(
                model_B, tok_B, R0_base, muA_base, muB_base, s_base, U_r,
                Z_A_facts[tr10_d], Z_B_facts[tr10_d],
                tr10_prompts, tr10_spos, tr10_slen,
                layer_B, device, loss_type=loss_t, n_steps=so_steps, batch_size=16,
                lr=(0.015 if loss_t == "kl" else 0.06),
                val_pack=val10_pack if loss_t == "kl" else None, seed=SEED + 100 + r_dim,
            )
            key = f"so{r_dim}_{loss_t}_selfswap_tr10_from_unlabR0"
            rec = _eval_and_record(results["low_dof_so_comparison"], key, mp_so)
            rec["train_diag"] = diag_so

        # 3. Also Train-10-relations Self-Swap MSE vs KL starting from frozen QA train10 R0 (for r=32)!
        if r_dim == 32:
            for loss_t in ("mse", "kl"):
                mp_so, diag_so = fit_low_dof_so_r(
                    model_B, tok_B, R0_qa10, muA_qa10, muB_qa10, s_qa10, U_r,
                    Z_A_facts[tr10_d], Z_B_facts[tr10_d],
                    tr10_prompts, tr10_spos, tr10_slen,
                    layer_B, device, loss_type=loss_t, n_steps=so_steps, batch_size=16,
                    lr=(0.015 if loss_t == "kl" else 0.06),
                    val_pack=val10_pack if loss_t == "kl" else None, seed=SEED + 200 + r_dim,
                )
                key = f"so{r_dim}_{loss_t}_selfswap_tr10_from_qa10R0"
                rec = _eval_and_record(results["low_dof_so_comparison"], key, mp_so)
                rec["train_diag"] = diag_so

    del model_B
    gc.collect()
    if has_mps:
        torch.mps.empty_cache()

    results["meta"]["elapsed_sec"] = round(time.time() - t_start, 2)
    if args.smoke:
        out_path = LAB_DIR / f"results_pilot1_{mk_A}_to_{mk_B}_smoke.json"
    elif pair_key == "qwen2.5-3b->smollm2-1.7b":
        out_path = LAB_DIR / "results_pilot1.json"
    else:
        out_path = LAB_DIR / f"results_pilot1_{mk_A}_to_{mk_B}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[pilot1] COMPLETE! Saved to {out_path} in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
