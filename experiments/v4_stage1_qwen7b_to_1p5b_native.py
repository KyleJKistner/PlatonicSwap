#!/usr/bin/env python3
"""
v4_stage1_qwen7b_to_1p5b_native.py — Stage 1 Pure Native Weight Translation (Qwen2.5-7B -> Qwen2.5-1.5B)
=========================================================================================================

Resolves 100% of P0 & P1 blockers from the Pre-GPU 2-Model Grading Council (Astra 6.2 Ultra + Neon High):
  - Safe `get_config_rope_theta` (`transformers 5.x` `rope_parameters`) + explicit shared valid vocab
    slicing `V_shared` (`152,064` vs `151,936` logit-width fix + exact token-to-ID mapping check).
  - ZERO external sidecars and ZERO runtime forward hooks at inference (`assert_zero_runtime_hooks`).
  - FP32 master-weight training for ONLY appended native slices (`_TrainableAppendedSliceLinear`,
    100% removed before inference -> fixes BF16 AdamW ULP stagnation & eliminates full-matrix Adam RAM!).
  - Fixed-batch `loss_start`/`loss_end` convergence assertion (`assert loss_end < loss_start`), true
    dev-split LR selection (`lr ∈ {1e-4, 3e-4}`), and EQUAL step budgets across Tier 2, Tier 2 Ctrl,
    and Tier 3!
  - Strict anti-leakage on Task B Airports: screens strictly on `p0`, splits strict blindspots into
    disjoint `dev` (feature selection + LR tuning) vs `eval` (`p0` donor-predicted implantation; `p1`,
    `p2`, and ordered `check_route_match` routes are 100% unseen!), plus exact multi-token Gate 1a check
    and empty-cohort guard.
  - Task A SQL `max_length = 384` + BOTH 80-query AND 40-query unseen-schema headroom reporting.
  - Task C 2-Row 2-Hop Policy Swap reports BOTH 2-way `fc` AND strict 4-way in-table leaf `fc4`,
    strict `torch.equal` rollback gate on BOTH Policy L and Policy R, and certified 380-fact locality
    on BOTH policies!
  - Host RAM bounded (<3 GiB calibration arrays, separate `max_tokens_cap=1024` `z_mlp` pass,
    zero-copy `unload_model_vram_safe`).

Usage:
  PYTHONPATH=platonicswap:experiments python experiments/v4_stage1_qwen7b_to_1p5b_native.py --unit-test
  PYTHONPATH=platonicswap:experiments python -u experiments/v4_stage1_qwen7b_to_1p5b_native.py --run-all
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (str(REPO_ROOT / "platonicswap"), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if "PLATONICSWAP_HF_CACHE" not in os.environ and Path("/tmp/platos_key_lab/hf_cache").is_dir():
    os.environ["PLATONICSWAP_HF_CACHE"] = "/tmp/platos_key_lab/hf_cache"
if "PLATONICSWAP_CACHE_DIR" not in os.environ and Path("/tmp/platos_key_lab/cache").is_dir():
    os.environ["PLATONICSWAP_CACHE_DIR"] = "/tmp/platos_key_lab/cache"

RESULTS_DIR = REPO_ROOT / "results" if (REPO_ROOT / "results").is_dir() else SCRIPT_DIR
OUT_PATH = RESULTS_DIR / "results_v4_stage1_qwen7b_to_1p5b_native.json"

from extract_and_baseline import HF_CACHE_DIR, check_answer_match, get_decoder_layers
from large_to_small import load_model_bf16
from native_weight_translation import (
    ReversibleNativeOrganPack,
    _TrainableAppendedSliceLinear,
    assert_zero_runtime_hooks,
    bf16_to_f64_np,
    build_large_calibration_corpus,
    compile_tier1a_sequential_native_delta_w,
    compile_tier1b_native_widened_heads_and_swiglu,
    compile_tier1c_native_anchor_swiglu_unroller,
    extract_donor_salient_heads_and_neurons,
    extract_layer_token_states,
    fit_val_selected_ridge_bridge,
    get_config_rope_theta,
    run_gate0b_rectangular_bridge_diagnostic,
    unload_model_vram_safe,
)
import v2_track1_dev_dataset as t1d
from v3_gate1_airports_enrichment import (
    build_route_items,
    check_iata_match,
    check_route_match,
    extract_iata_code,
    fetch_or_fallback_airports,
    format_airport_prompt,
)
from v3_gate2_skill_and_sql import (
    WORD_CANDIDATES_KEYS,
    WORD_CANDIDATES_VALS,
    build_text_to_sqlite_benchmark,
    execute_sql_on_schema,
    extract_sql_from_generation,
    filter_single_token_words,
    sql_results_match,
)

SEED = 20261128
DONOR_KEY_DEFAULT = "qwen2.5-7b-instruct"
RECIPIENT_KEY_DEFAULT = "qwen2.5-1.5b-instruct"
DEFAULT_COMPILE_LAYERS: Tuple[int, ...] = (14, 16, 18, 20, 22, 24, 26, 27)
SQL_TRAIN_SCHEMAS: Tuple[str, ...] = (
    "logistics_shipments",
    "hospital_appointments",
    "library_loans",
    "solar_power_grid",
)
SQL_TEST_SCHEMAS: Tuple[str, ...] = (
    "retail_inventory",
    "university_courses",
    "flight_manifests",
    "sensor_telemetry",
)


# ======================================================================================
# 1. Task C Dataset: 2-Row Shuffled Counterfactual 2-Hop Policy Swap (Donor EM ~90%+)
# ======================================================================================

def format_2row_2hop_prompts(
    hop1_rows: Sequence[Tuple[str, str]],
    hop2_rows: Sequence[Tuple[str, str, str]],
    query: str,
) -> Dict[str, str]:
    lines = ["Table:"]
    for a, b in hop1_rows:
        lines.append(f"{a} -> {b}")
    for b, vL, vR in hop2_rows:
        lines.append(f"{b} -> ({vL}, {vR})")
    lines.append(f"Query: {query}")
    lines.append("Output:")
    recip_prompt = "\n".join(lines)

    demo_L = (
        "Rule: Follow the 2-hop Table (Query -> middle -> (first, second)) and output ONLY the FIRST word in the pair.\n\n"
        "Table:\ncat -> sun\nfox -> owl\nowl -> (big, odd)\nsun -> (yes, no)\nQuery: cat\nOutput: yes\n\n"
        "Table:\nfox -> owl\ncat -> sun\nsun -> (yes, no)\nowl -> (big, odd)\nQuery: fox\nOutput: big\n\n"
    )
    demo_R = (
        "Rule: Follow the 2-hop Table (Query -> middle -> (first, second)) and output ONLY the SECOND word in the pair.\n\n"
        "Table:\ncat -> sun\nfox -> owl\nowl -> (big, odd)\nsun -> (yes, no)\nQuery: cat\nOutput: no\n\n"
        "Table:\nfox -> owl\ncat -> sun\nsun -> (yes, no)\nowl -> (big, odd)\nQuery: fox\nOutput: odd\n\n"
    )
    return {
        "recipient_prompt": recip_prompt,
        "donor_L_prompt": demo_L + recip_prompt,
        "donor_R_prompt": demo_R + recip_prompt,
    }


def build_2row_2hop_dataset(
    tokenizers: Sequence[Any],
    n_train: int = 192,
    n_dev: int = 64,
    n_test: int = 256,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Strictly key-disjoint 2-row shuffled 2-hop tables (`train`, `dev`, `test`) with all 4 leaf values logged."""
    rng = np.random.default_rng(seed)
    all_keys = filter_single_token_words(WORD_CANDIDATES_KEYS, tokenizers)
    all_vals = filter_single_token_words(WORD_CANDIDATES_VALS, tokenizers)
    all_keys = [w for w in all_keys if w not in set(all_vals)]
    assert len(all_keys) >= 40 and len(all_vals) >= 24

    perm_k = rng.permutation(len(all_keys))
    mid_k = len(all_keys) // 2
    train_keys = [all_keys[i] for i in perm_k[:mid_k]]
    test_keys = [all_keys[i] for i in perm_k[mid_k:]]
    assert set(train_keys).isdisjoint(set(test_keys))

    def _gen_split(key_pool: List[str], n_items: int, split_name: str, seen_sigs: set) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        while len(items) < n_items:
            ck = rng.choice(key_pool, size=4, replace=False).tolist()
            cv = rng.choice(all_vals, size=4, replace=False).tolist()
            hop1 = [(ck[0], ck[2]), (ck[1], ck[3])]
            hop2 = [(ck[2], cv[0], cv[1]), (ck[3], cv[2], cv[3])]
            i1 = rng.permutation(2).tolist()
            i2 = rng.permutation(2).tolist()
            h1s = [hop1[i] for i in i1]
            h2s = [hop2[i] for i in i2]
            qi = int(rng.integers(0, 2))
            q, mid_w = ck[qi], ck[2 + qi]
            vL, vR = cv[2 * qi], cv[2 * qi + 1]
            other_i = 1 - qi
            dist_L, dist_R = cv[2 * other_i], cv[2 * other_i + 1]
            sig = (tuple(h1s), tuple(h2s), q)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            pr = format_2row_2hop_prompts(h1s, h2s, q)
            items.append({
                "id": f"{split_name}_{len(items):03d}",
                "split": split_name,
                "query": q, "mid": mid_w,
                "val_L": vL, "val_R": vR,
                "dist_L": dist_L, "dist_R": dist_R,
                **pr,
            })
        return items

    seen_tr = set()
    train_items = _gen_split(train_keys, n_train, "train", seen_tr)
    dev_items = _gen_split(train_keys, n_dev, "dev", seen_tr)
    test_items = _gen_split(test_keys, n_test, "test", set())
    return {"train": train_items, "dev": dev_items, "test": test_items, "train_keys": train_keys, "test_keys": test_keys}


# ======================================================================================
# 2. Native Zero-Hook Evaluation, Certified Locality & FP32 Master-Weight Native Polish
# ======================================================================================

class _MaskPaddedVocabLogitsProcessor:
    """Mask untrained padded lm_head rows `>= v_shared` during greedy generation (ZERO model hooks!)."""

    def __init__(self, v_shared: int):
        self.v_shared = int(v_shared)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[-1] > self.v_shared:
            scores[:, self.v_shared :] = -float("inf")
        return scores


@torch.no_grad()
def greedy_generate_native(
    model: nn.Module,
    tok: Any,
    device: torch.device,
    prompts: Sequence[str],
    max_new_tokens: int = 8,
    batch_size: int = 32,
    v_shared: Optional[int] = None,
) -> List[str]:
    """Stock HuggingFace generation with fresh DynamicCache, shared-vocab masking, and ZERO runtime hooks!"""
    assert_zero_runtime_hooks(model)
    if not prompts:
        return []
    old_pad = tok.padding_side
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lproc = [_MaskPaddedVocabLogitsProcessor(v_shared)] if v_shared is not None else None
    out_txts: List[str] = []
    try:
        for i0 in range(0, len(prompts), batch_size):
            bp = list(prompts[i0 : i0 + batch_size])
            enc = tok(bp, return_tensors="pt", padding=True, add_special_tokens=True)
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            gen_kw: Dict[str, Any] = dict(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_id,
            )
            if lproc is not None:
                gen_kw["logits_processor"] = lproc
            gen = model.generate(**gen_kw)
            plen = ids.shape[1]
            for bi in range(len(bp)):
                out_txts.append(tok.decode(gen[bi, plen:], skip_special_tokens=True))
    finally:
        tok.padding_side = old_pad
    return out_txts


def eval_certified_380_locality(
    model: nn.Module,
    tok: Any,
    device: torch.device,
    loc_facts: Sequence[Dict[str, Any]],
    base_ok: Optional[List[int]] = None,
    batch_size: int = 32,
    v_shared: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[int]]:
    prompts = [f["direct_prompt"] for f in loc_facts]
    gens = greedy_generate_native(model, tok, device, prompts, max_new_tokens=8, batch_size=batch_size, v_shared=v_shared)
    ok_vec = [int(check_answer_match(g, f["aliases"], f.get("relation", ""))) for g, f in zip(gens, loc_facts)]
    n_ok = int(sum(ok_vec))
    res: Dict[str, Any] = {"n_total": len(loc_facts), "n_correct": n_ok, "acc": round(n_ok / max(len(loc_facts), 1), 4)}
    if base_ok is not None:
        n_base = int(sum(base_ok))
        broken = int(sum(1 for b, c in zip(base_ok, ok_vec) if b == 1 and c == 0))
        res["n_base_correct"] = n_base
        res["newly_broken"] = broken
        res["retention"] = round((n_base - broken) / max(n_base, 1), 4)
    return res, ok_vec


def format_sql_chat_prompt(tok: Any, ddl: str, question: str) -> str:
    prompt = (
        "You are an expert SQLite assistant. Given the SQLite schema below, write ONLY a single valid "
        "SQLite SELECT query answering the question. Do not include any explanation.\n\n"
        f"Schema:\n{ddl}\n\nQuestion: {question}\nSQL:\n"
    )
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return prompt


@torch.no_grad()
def eval_sql_queries_native(
    model: nn.Module,
    tok: Any,
    device: torch.device,
    queries: Sequence[Dict[str, Any]],
    gold_cache: Dict[str, List[Tuple[Any, ...]]],
    batch_size: int = 8,
    v_shared: Optional[int] = None,
) -> Dict[str, Any]:
    assert_zero_runtime_hooks(model)
    prompts = [format_sql_chat_prompt(tok, it["ddl"], it["question"]) for it in queries]
    gens = greedy_generate_native(model, tok, device, prompts, max_new_tokens=96, batch_size=batch_size, v_shared=v_shared)
    valid_hits, exec_hits = 0, 0
    by_schema: Dict[str, Dict[str, int]] = {}
    pred_sqls: List[str] = []
    for it, gen_txt in zip(queries, gens):
        pred_sql = extract_sql_from_generation(gen_txt)
        pred_sqls.append(pred_sql)
        ok, rows, _ = execute_sql_on_schema(it["ddl"], it["seed"], pred_sql)
        matched = bool(ok and sql_results_match(gold_cache[it["id"]], rows, order_sensitive=it["order_sensitive"]))
        valid_hits += int(ok)
        exec_hits += int(matched)
        sc = by_schema.setdefault(it["schema_id"], {"n": 0, "valid": 0, "exec_match": 0})
        sc["n"] += 1
        sc["valid"] += int(ok)
        sc["exec_match"] += int(matched)
    n = max(len(queries), 1)
    return {
        "n_queries": len(queries),
        "exec_match_count": exec_hits,
        "exec_match_rate": round(exec_hits / float(n), 4),
        "syntax_valid_rate": round(valid_hits / float(n), 4),
        "by_schema": {k: {**v, "exec_rate": round(v["exec_match"] / v["n"], 4)} for k, v in by_schema.items()},
        "_pred_sqls": pred_sqls,
    }


def make_random_zero_control_pack(template_pack: ReversibleNativeOrganPack, seed: int = SEED + 99) -> ReversibleNativeOrganPack:
    gen = torch.Generator().manual_seed(seed)
    attn_c: Dict[int, Dict[str, Any]] = {}
    for li, sp in template_pack.attn_widen.items():
        attn_c[li] = {
            "q_w": torch.randn(sp["q_w"].shape, generator=gen, dtype=torch.float32) * 0.02,
            "q_b": torch.zeros_like(sp["q_b"]) if sp.get("q_b") is not None else None,
            "k_w": torch.randn(sp["k_w"].shape, generator=gen, dtype=torch.float32) * 0.02,
            "k_b": torch.zeros_like(sp["k_b"]) if sp.get("k_b") is not None else None,
            "v_w": torch.randn(sp["v_w"].shape, generator=gen, dtype=torch.float32) * 0.02,
            "v_b": torch.zeros_like(sp["v_b"]) if sp.get("v_b") is not None else None,
            "o_w": torch.zeros_like(sp["o_w"]),
            "add_q_heads": sp["add_q_heads"], "add_kv_heads": sp["add_kv_heads"],
        }
    mlp_c: Dict[int, Dict[str, Any]] = {}
    for li, sp in template_pack.mlp_widen.items():
        mlp_c[li] = {
            "gate_w": torch.randn(sp["gate_w"].shape, generator=gen, dtype=torch.float32) * 0.02,
            "up_w": torch.randn(sp["up_w"].shape, generator=gen, dtype=torch.float32) * 0.02,
            "down_w": torch.zeros_like(sp["down_w"]),
        }
    return ReversibleNativeOrganPack(attn_widen=attn_c, mlp_widen=mlp_c, metadata={"tier": "Tier2_Ctrl_RandomZeroInit"})


def _attach_training_only_fp32_slices(
    model_B: nn.Module,
    init_pack: ReversibleNativeOrganPack,
    orig_attn: Dict[int, Dict[str, Any]],
    orig_mlp: Dict[int, Dict[str, Any]],
) -> List[nn.Parameter]:
    """
    Attach training-only FP32 master-weight slice adapters into caller-owned `orig_attn`/`orig_mlp`
    maps BEFORE mutating any layer, so even a mid-loop OOM unwinds every partial replacement cleanly!
    """
    layers = get_decoder_layers(model_B)
    params: List[nn.Parameter] = []

    for li, sp in init_pack.attn_widen.items():
        attn = layers[li].self_attn
        orig_attn[li] = {"q_proj": attn.q_proj, "k_proj": attn.k_proj, "v_proj": attn.v_proj, "o_proj": attn.o_proj}
        attn.q_proj = _TrainableAppendedSliceLinear(attn.q_proj, sp["q_w"], sp.get("q_b"), mode="row")
        attn.k_proj = _TrainableAppendedSliceLinear(attn.k_proj, sp["k_w"], sp.get("k_b"), mode="row")
        attn.v_proj = _TrainableAppendedSliceLinear(attn.v_proj, sp["v_w"], sp.get("v_b"), mode="row")
        attn.o_proj = _TrainableAppendedSliceLinear(attn.o_proj, sp["o_w"], None, mode="col")
        for m in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
            params.append(m.append_weight)
            if m.append_bias is not None:
                params.append(m.append_bias)

    for li, sp in init_pack.mlp_widen.items():
        mlp = layers[li].mlp
        orig_mlp[li] = {"gate_proj": mlp.gate_proj, "up_proj": mlp.up_proj, "down_proj": mlp.down_proj}
        mlp.gate_proj = _TrainableAppendedSliceLinear(mlp.gate_proj, sp["gate_w"], None, mode="row")
        mlp.up_proj = _TrainableAppendedSliceLinear(mlp.up_proj, sp["up_w"], None, mode="row")
        mlp.down_proj = _TrainableAppendedSliceLinear(mlp.down_proj, sp["down_w"], None, mode="col")
        for m in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            params.append(m.append_weight)
    return params


def _extract_and_detach_training_slices(
    model_B: nn.Module,
    init_pack: ReversibleNativeOrganPack,
    orig_attn: Dict[int, Dict[str, Any]],
    orig_mlp: Dict[int, Dict[str, Any]],
    tier_label: str,
) -> ReversibleNativeOrganPack:
    layers = get_decoder_layers(model_B)
    new_attn: Dict[int, Dict[str, Any]] = {}
    new_mlp: Dict[int, Dict[str, Any]] = {}
    for li, orig in orig_attn.items():
        attn = layers[li].self_attn
        sp = init_pack.attn_widen[li]
        if isinstance(attn.q_proj, _TrainableAppendedSliceLinear) and isinstance(attn.o_proj, _TrainableAppendedSliceLinear):
            new_attn[li] = {
                "q_w": attn.q_proj.append_weight.detach().float().cpu().clone(),
                "q_b": attn.q_proj.append_bias.detach().float().cpu().clone() if attn.q_proj.append_bias is not None else None,
                "k_w": attn.k_proj.append_weight.detach().float().cpu().clone(),
                "k_b": attn.k_proj.append_bias.detach().float().cpu().clone() if attn.k_proj.append_bias is not None else None,
                "v_w": attn.v_proj.append_weight.detach().float().cpu().clone(),
                "v_b": attn.v_proj.append_bias.detach().float().cpu().clone() if attn.v_proj.append_bias is not None else None,
                "o_w": attn.o_proj.append_weight.detach().float().cpu().clone(),
                "add_q_heads": sp["add_q_heads"], "add_kv_heads": sp["add_kv_heads"],
            }
        attn.q_proj = orig["q_proj"]
        attn.k_proj = orig["k_proj"]
        attn.v_proj = orig["v_proj"]
        attn.o_proj = orig["o_proj"]
    for li, orig in orig_mlp.items():
        mlp = layers[li].mlp
        if isinstance(mlp.gate_proj, _TrainableAppendedSliceLinear) and isinstance(mlp.down_proj, _TrainableAppendedSliceLinear):
            new_mlp[li] = {
                "gate_w": mlp.gate_proj.append_weight.detach().float().cpu().clone(),
                "up_w": mlp.up_proj.append_weight.detach().float().cpu().clone(),
                "down_w": mlp.down_proj.append_weight.detach().float().cpu().clone(),
            }
        mlp.gate_proj = orig["gate_proj"]
        mlp.up_proj = orig["up_proj"]
        mlp.down_proj = orig["down_proj"]
    assert_zero_runtime_hooks(model_B)
    return ReversibleNativeOrganPack(attn_widen=new_attn, mlp_widen=new_mlp, metadata={"tier": tier_label})


def _make_sft_batch(encoded_pairs: Sequence[Tuple[List[int], List[int]]], idxs: Sequence[int], pad_id: int, device: torch.device):
    ml = max(len(encoded_pairs[i][0]) for i in idxs)
    inp = torch.full((len(idxs), ml), pad_id, dtype=torch.long, device=device)
    att = torch.zeros((len(idxs), ml), dtype=torch.long, device=device)
    lbl = torch.full((len(idxs), ml), -100, dtype=torch.long, device=device)
    for bi, idx in enumerate(idxs):
        ii, ll = encoded_pairs[idx]
        inp[bi, : len(ii)] = torch.tensor(ii, dtype=torch.long, device=device)
        att[bi, : len(ii)] = 1
        lbl[bi, : len(ll)] = torch.tensor(ll, dtype=torch.long, device=device)
    return inp, att, lbl


def polish_widened_native_pack_sft(
    model_B: nn.Module,
    tok_B: Any,
    device_B: torch.device,
    init_pack: ReversibleNativeOrganPack,
    prompt_target_pairs: Sequence[Tuple[str, str]],
    v_shared: Optional[int] = None,
    lr: float = 2e-4,
    steps: int = 40,
    batch_size: int = 4,
    seed: int = SEED,
    strict_loss_decrease: bool = True,
) -> Tuple[ReversibleNativeOrganPack, Dict[str, Any]]:
    """
    FP32 master-weight native polish on ONLY appended slices (exception-safe attach inside `try:`,
    100% detached in `finally:`, SFT CE normalized strictly over `[:v_shared]`, fixed-batch
    `loss_start`/`loss_end` convergence assertion!).
    """
    t0 = time.time()
    pad_id = tok_B.pad_token_id if tok_B.pad_token_id is not None else tok_B.eos_token_id
    enc_pairs: List[Tuple[List[int], List[int]]] = []
    for pr, tgt in prompt_target_pairs:
        p_ids = tok_B.encode(pr, add_special_tokens=True)
        t_ids = tok_B.encode(tgt, add_special_tokens=False)
        if t_ids:
            enc_pairs.append((p_ids + t_ids, [-100] * len(p_ids) + t_ids))
    assert len(enc_pairs) > 0

    for p in model_B.parameters():
        p.requires_grad_(False)

    orig_attn: Dict[int, Dict[str, Any]] = {}
    orig_mlp: Dict[int, Dict[str, Any]] = {}
    out_pack: Optional[ReversibleNativeOrganPack] = None
    try:
        params = _attach_training_only_fp32_slices(model_B, init_pack, orig_attn, orig_mlp)
        probe_idx = list(range(min(8, len(enc_pairs))))
        p_inp, p_att, p_lbl = _make_sft_batch(enc_pairs, probe_idx, pad_id, device_B)

        def _sft_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            sl = logits[:, :-1, :v_shared] if v_shared is not None else logits[:, :-1, :]
            return F.cross_entropy(sl.float().contiguous().view(-1, sl.size(-1)), labels[:, 1:].contiguous().view(-1), ignore_index=-100)

        with torch.no_grad():
            loss_start = float(_sft_ce(model_B(input_ids=p_inp, attention_mask=p_att, use_cache=False).logits, p_lbl).item())

        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=lr * 0.1)
        rng = np.random.default_rng(seed)

        for _ in range(steps):
            b_idx = rng.choice(len(enc_pairs), size=min(batch_size, len(enc_pairs)), replace=False).tolist()
            inp, att, lbl = _make_sft_batch(enc_pairs, b_idx, pad_id, device_B)
            out = model_B(input_ids=inp, attention_mask=att, use_cache=False)
            loss = _sft_ce(out.logits, lbl)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

        with torch.no_grad():
            loss_end = float(_sft_ce(model_B(input_ids=p_inp, attention_mask=p_att, use_cache=False).logits, p_lbl).item())
    finally:
        out_pack = _extract_and_detach_training_slices(model_B, init_pack, orig_attn, orig_mlp, "Tier2_NativePolish")
        for p in model_B.parameters():
            p.requires_grad_(False)

    assert out_pack is not None
    info = {
        "steps": int(steps), "lr": float(lr),
        "loss_start": round(loss_start, 4), "loss_end": round(loss_end, 4),
        "loss_decreased": bool(loss_end < loss_start),
        "train_sec": round(time.time() - t0, 2),
    }
    if strict_loss_decrease:
        assert info["loss_decreased"], f"Native polish failed to decrease loss on fixed probe batch: {info}"
    return out_pack, info


def polish_widened_native_pack_teacher_kl(
    model_B: nn.Module,
    tok_B: Any,
    device_B: torch.device,
    init_pack: ReversibleNativeOrganPack,
    prompts: Sequence[str],
    teacher_logits_shared: torch.Tensor,
    v_shared: int,
    lr: float = 3e-4,
    steps: int = 80,
    batch_size: int = 8,
    top_k: int = 16,
    seed: int = SEED,
    strict_loss_decrease: bool = True,
) -> Tuple[ReversibleNativeOrganPack, Dict[str, Any]]:
    """FP32 master-weight native polish via Donor 7B top-K KL (exception-safe attach inside `try:`)."""
    t0 = time.time()
    probs_t = F.softmax(teacher_logits_shared[:, :v_shared].float(), dim=-1)
    top_v, top_i = torch.topk(probs_t, k=top_k, dim=-1)
    top_v = top_v / (top_v.sum(dim=-1, keepdim=True) + 1e-12)

    for p in model_B.parameters():
        p.requires_grad_(False)
    orig_attn: Dict[int, Dict[str, Any]] = {}
    orig_mlp: Dict[int, Dict[str, Any]] = {}
    old_pad = tok_B.padding_side
    tok_B.padding_side = "left"
    out_pack: Optional[ReversibleNativeOrganPack] = None
    try:
        params = _attach_training_only_fp32_slices(model_B, init_pack, orig_attn, orig_mlp)
        probe_idx = list(range(min(16, len(prompts))))
        p_enc = tok_B([prompts[i] for i in probe_idx], return_tensors="pt", padding=True, add_special_tokens=True)
        p_ids, p_att = p_enc["input_ids"].to(device_B), p_enc["attention_mask"].to(device_B)
        p_ti, p_tv = top_i[probe_idx].to(device_B), top_v[probe_idx].to(device_B)

        def _eval_probe() -> float:
            with torch.no_grad():
                o = model_B(input_ids=p_ids, attention_mask=p_att, use_cache=False)
                lps = F.log_softmax(o.logits[:, -1, :v_shared].float(), dim=-1)
                return float(-(p_tv * torch.gather(lps, 1, p_ti)).sum(dim=-1).mean().item())

        loss_start = _eval_probe()
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=lr * 0.1)
        rng = np.random.default_rng(seed)

        for _ in range(steps):
            b_idx = rng.choice(len(prompts), size=min(batch_size, len(prompts)), replace=False)
            enc = tok_B([prompts[i] for i in b_idx], return_tensors="pt", padding=True, add_special_tokens=True)
            out = model_B(
                input_ids=enc["input_ids"].to(device_B),
                attention_mask=enc["attention_mask"].to(device_B),
                use_cache=False,
            )
            lps = F.log_softmax(out.logits[:, -1, :v_shared].float(), dim=-1)
            loss = -(top_v[b_idx].to(device_B) * torch.gather(lps, 1, top_i[b_idx].to(device_B))).sum(dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

        loss_end = _eval_probe()
    finally:
        tok_B.padding_side = old_pad
        out_pack = _extract_and_detach_training_slices(model_B, init_pack, orig_attn, orig_mlp, "Tier2_NativeKL")
        for p in model_B.parameters():
            p.requires_grad_(False)

    assert out_pack is not None
    info = {
        "steps": int(steps), "lr": float(lr),
        "loss_start": round(loss_start, 4), "loss_end": round(loss_end, 4),
        "loss_decreased": bool(loss_end < loss_start),
        "train_sec": round(time.time() - t0, 2),
    }
    if strict_loss_decrease:
        assert info["loss_decreased"], f"Native KL polish failed to decrease loss: {info}"
    return out_pack, info


def train_tier3_teacher_lora_merged_native(
    model_B: nn.Module,
    tok_B: Any,
    device_B: torch.device,
    layer_indices: Sequence[int],
    mode: str,
    prompt_target_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    prompts_kl: Optional[Sequence[str]] = None,
    teacher_logits_shared: Optional[torch.Tensor] = None,
    v_shared: Optional[int] = None,
    rank: int = 16,
    alpha: float = 32.0,
    lr: float = 3e-4,
    steps: int = 40,
    batch_size: int = 8,
    seed: int = SEED,
) -> Tuple[ReversibleNativeOrganPack, Dict[str, Any]]:
    """Fair Tier 3 Rank-16 Teacher-LoRA merged into native `ReversibleNativeOrganPack` (ZERO hooks at inference!)."""
    from v3_gate1_airports_enrichment import attach_lora_adapters, detach_lora_to_delta_pack

    t0 = time.time()
    for p in model_B.parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed)
    targets = ("q_proj", "k_proj", "v_proj", "o_proj", "down_proj")
    attached = attach_lora_adapters(model_B, layer_indices, target_modules=targets, rank=rank, alpha=alpha)
    lora_params = [p for m in attached.values() for p in (m.lora_A, m.lora_B)]
    opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=lr * 0.1)
    rng = np.random.default_rng(seed)
    pad_id = tok_B.pad_token_id if tok_B.pad_token_id is not None else tok_B.eos_token_id
    old_pad = tok_B.padding_side

    try:
        if mode == "sft":
            assert prompt_target_pairs is not None
            enc_pairs = []
            for pr, tgt in prompt_target_pairs:
                p_ids = tok_B.encode(pr, add_special_tokens=True)
                t_ids = tok_B.encode(tgt, add_special_tokens=False)
                if t_ids:
                    enc_pairs.append((p_ids + t_ids, [-100] * len(p_ids) + t_ids))
            p_inp, p_att, p_lbl = _make_sft_batch(enc_pairs, list(range(min(8, len(enc_pairs)))), pad_id, device_B)
            def _sft_ce3(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                sl = logits[:, :-1, :v_shared] if v_shared is not None else logits[:, :-1, :]
                return F.cross_entropy(sl.float().contiguous().view(-1, sl.size(-1)), labels[:, 1:].contiguous().view(-1), ignore_index=-100)

            with torch.no_grad():
                o0 = model_B(input_ids=p_inp, attention_mask=p_att, use_cache=False)
                loss_start = float(_sft_ce3(o0.logits, p_lbl).item())
            for _ in range(steps):
                b_idx = rng.choice(len(enc_pairs), size=min(batch_size, len(enc_pairs)), replace=False).tolist()
                inp, att, lbl = _make_sft_batch(enc_pairs, b_idx, pad_id, device_B)
                out = model_B(input_ids=inp, attention_mask=att, use_cache=False)
                loss = _sft_ce3(out.logits, lbl)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lora_params, 1.0); opt.step(); sched.step()
            with torch.no_grad():
                o1 = model_B(input_ids=p_inp, attention_mask=p_att, use_cache=False)
                loss_end = float(_sft_ce3(o1.logits, p_lbl).item())
        else:
            assert prompts_kl is not None and teacher_logits_shared is not None and v_shared is not None
            tok_B.padding_side = "left"
            probs_t = F.softmax(teacher_logits_shared[:, :v_shared].float(), dim=-1)
            top_v, top_i = torch.topk(probs_t, k=16, dim=-1)
            top_v = top_v / (top_v.sum(dim=-1, keepdim=True) + 1e-12)
            p_idx = list(range(min(16, len(prompts_kl))))
            p_enc = tok_B([prompts_kl[i] for i in p_idx], return_tensors="pt", padding=True, add_special_tokens=True)
            p_ids, p_att = p_enc["input_ids"].to(device_B), p_enc["attention_mask"].to(device_B)
            with torch.no_grad():
                lps0 = F.log_softmax(model_B(input_ids=p_ids, attention_mask=p_att, use_cache=False).logits[:, -1, :v_shared].float(), dim=-1)
                loss_start = float(-(top_v[p_idx].to(device_B) * torch.gather(lps0, 1, top_i[p_idx].to(device_B))).sum(dim=-1).mean().item())
            for _ in range(steps):
                b_idx = rng.choice(len(prompts_kl), size=min(batch_size, len(prompts_kl)), replace=False)
                enc = tok_B([prompts_kl[i] for i in b_idx], return_tensors="pt", padding=True, add_special_tokens=True)
                out = model_B(input_ids=enc["input_ids"].to(device_B), attention_mask=enc["attention_mask"].to(device_B), use_cache=False)
                lps = F.log_softmax(out.logits[:, -1, :v_shared].float(), dim=-1)
                loss = -(top_v[b_idx].to(device_B) * torch.gather(lps, 1, top_i[b_idx].to(device_B))).sum(dim=-1).mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lora_params, 1.0); opt.step(); sched.step()
            with torch.no_grad():
                lps1 = F.log_softmax(model_B(input_ids=p_ids, attention_mask=p_att, use_cache=False).logits[:, -1, :v_shared].float(), dim=-1)
                loss_end = float(-(top_v[p_idx].to(device_B) * torch.gather(lps1, 1, top_i[p_idx].to(device_B))).sum(dim=-1).mean().item())
    finally:
        tok_B.padding_side = old_pad
        old_pack, raw_lora_bytes = detach_lora_to_delta_pack(model_B, attached, layer_indices, target_modules=targets)

    native_pack = ReversibleNativeOrganPack(deltas=old_pack.deltas, metadata={"tier": "Tier3_MergedTeacherLoRA"})
    assert_zero_runtime_hooks(model_B)
    info = {
        "steps": int(steps), "lr": float(lr), "rank": int(rank),
        "loss_start": round(loss_start, 4), "loss_end": round(loss_end, 4),
        "loss_decreased": bool(loss_end < loss_start),
        "raw_lora_factor_bytes_bf16": int(raw_lora_bytes),
        "merged_dense_delta_bytes_bf16": native_pack.n_bytes_bf16(),
        "train_sec": round(time.time() - t0, 2),
    }
    return native_pack, info


# ======================================================================================
# 3. Task C Scorers (EM + 2-Way FC + Strict 4-Way In-Table Leaf FC4!)
# ======================================================================================

def _eval_2hop_logits(logits_shared: torch.Tensor, items: Sequence[Dict[str, Any]], tok: Any, pol: str) -> Dict[str, Any]:
    em_v, fc2_v, fc4_v = [], [], []
    for i, it in enumerate(items):
        tL = tok.encode(" " + it["val_L"], add_special_tokens=False)[0]
        tR = tok.encode(" " + it["val_R"], add_special_tokens=False)[0]
        dL = tok.encode(" " + it["dist_L"], add_special_tokens=False)[0]
        dR = tok.encode(" " + it["dist_R"], add_special_tokens=False)[0]
        row = logits_shared[i]
        pred = int(row.argmax().item())
        tgt, oth = (tL, tR) if pol == "L" else (tR, tL)
        em_v.append(1 if pred == tgt else 0)
        fc2_v.append(1 if float(row[tgt].item()) > float(row[oth].item()) else 0)
        fc4_v.append(1 if float(row[tgt].item()) > max(float(row[oth].item()), float(row[dL].item()), float(row[dR].item())) else 0)
    return {
        "em": round(float(np.mean(em_v)), 4),
        "fc": round(float(np.mean(fc2_v)), 4),
        "fc4": round(float(np.mean(fc4_v)), 4),
        "_em": em_v, "_fc": fc2_v, "_fc4": fc4_v,
    }


def _joint_2hop(rL: Dict[str, Any], rR: Dict[str, Any]) -> Dict[str, float]:
    eL, eR = np.asarray(rL["_em"]), np.asarray(rR["_em"])
    fL, fR = np.asarray(rL["_fc"]), np.asarray(rR["_fc"])
    qL, qR = np.asarray(rL["_fc4"]), np.asarray(rR["_fc4"])
    return {
        "left_em": rL["em"], "right_em": rR["em"], "joint_swap_em": round(float(np.mean((eL == 1) & (eR == 1))), 4),
        "left_fc": rL["fc"], "right_fc": rR["fc"], "joint_swap_fc": round(float(np.mean((fL == 1) & (fR == 1))), 4),
        "left_fc4": rL["fc4"], "right_fc4": rR["fc4"], "joint_swap_fc4": round(float(np.mean((qL == 1) & (qR == 1))), 4),
    }


@torch.no_grad()
def _last_tok_logits(model: nn.Module, tok: Any, prompts: Sequence[str], dev: torch.device, bs: int = 16, v_cap: Optional[int] = None) -> torch.Tensor:
    assert_zero_runtime_hooks(model)
    assert len(prompts) > 0, "_last_tok_logits called with empty prompt list!"
    old_pad = tok.padding_side
    tok.padding_side = "left"
    out_l = []
    try:
        for i0 in range(0, len(prompts), bs):
            enc = tok(list(prompts[i0 : i0 + bs]), return_tensors="pt", padding=True, add_special_tokens=True)
            lg = model(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev), use_cache=False).logits[:, -1, :]
            if v_cap is not None:
                lg = lg[:, :v_cap]
            out_l.append(lg.float().cpu())
    finally:
        tok.padding_side = old_pad
    return torch.cat(out_l, dim=0)


# ======================================================================================
# 4. Main Stage 1 GPU Runner (`Qwen2.5-7B-Instruct` -> `Qwen2.5-1.5B-Instruct`)
# ======================================================================================

def run_stage1_all(args: argparse.Namespace) -> Dict[str, Any]:
    t_all = time.time()
    compile_layers = list(DEFAULT_COMPILE_LAYERS)
    cache_dir = Path("/tmp/v4_stage1_donor_cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n================ V4.1 STAGE 1: PURE NATIVE WEIGHT TRANSLATION ({args.donor_model} -> {args.recipient_model}) ================")
    print(f"Compiled native layer band ({len(compile_layers)} layers): {compile_layers}")

    cal_corpus = build_large_calibration_corpus(n_fit_passages=args.n_fit_passages, n_val_passages=args.n_val_passages, seed=SEED)
    sql_bench = build_text_to_sqlite_benchmark()
    sql_gold: Dict[str, List[Tuple[Any, ...]]] = {}
    for it in sql_bench:
        ok, rows, err = execute_sql_on_schema(it["ddl"], it["seed"], it["gold_sql"])
        assert ok and rows is not None and len(rows) > 0, f"Gold SQL failed: {it['id']}: {err}"
        sql_gold[it["id"]] = rows
    sql_train = [it for it in sql_bench if it["schema_id"] in SQL_TRAIN_SCHEMAS]
    sql_test = [it for it in sql_bench if it["schema_id"] in SQL_TEST_SCHEMAS]
    # Split SQL train into 30 core fit + 10 dev for clean LR tuning!
    sql_fit_sub, sql_dev_sub = sql_train[:30], sql_train[30:]

    airports, air_src = fetch_or_fallback_airports(max_airports=args.max_airports, seed=SEED, offline=args.offline)
    loc_facts = t1d.build_certified_locality_facts()

    # =========================================================================
    # PHASE 1: DONOR 7B EXTRACTION (100% UNLOADED VIA ZERO-COPY VRAM FREE!)
    # =========================================================================
    t_d0 = time.time()
    model_A, tok_A, dev_A, _ = load_model_bf16(args.donor_model, allow_download=args.allow_download)
    D_A = int(model_A.config.hidden_size)
    L_A = len(get_decoder_layers(model_A))
    head_dim_A = int(getattr(model_A.config, "head_dim", D_A // model_A.config.num_attention_heads))
    rope_A = get_config_rope_theta(model_A.config)
    v_A_out = int(model_A.lm_head.out_features)
    tok_A_vocab = tok_A.get_vocab()
    v_tok_A = int(len(tok_A))
    print(f"  [Phase 1] Loaded Donor {args.donor_model} (L={L_A}, D={D_A}, head_dim={head_dim_A}, rope_theta={rope_A:g}, lm_head={v_A_out}, tok_len={v_tok_A})")

    twohop_ds = build_2row_2hop_dataset([tok_A], n_train=args.n_2hop_train, n_dev=64, n_test=args.n_2hop_test, seed=SEED)

    # 1A. Stream calibration states (capped cleanly at N_fit=12,000, N_val=2,500 -> bounded host RAM!)
    cal_fit_A = extract_layer_token_states(
        model_A, tok_A, cal_corpus["fit"], compile_layers, dev_A,
        batch_size=args.batch_size, max_length=160, positions="all_nonpad",
        capture_fields=("u_in", "y_attn", "u_post", "h_out"), max_tokens_cap=args.cal_fit_cap,
    )
    cal_val_A = extract_layer_token_states(
        model_A, tok_A, cal_corpus["val"], compile_layers, dev_A,
        batch_size=args.batch_size, max_length=160, positions="all_nonpad",
        capture_fields=("u_in", "y_attn", "u_post", "h_out"), max_tokens_cap=args.cal_val_cap,
    )
    n_cal_fit = int(len(cal_fit_A[compile_layers[0]]["h_out"]))
    n_cal_val = int(len(cal_val_A[compile_layers[0]]["h_out"]))
    print(f"  [Calibration Tokens] N_fit={n_cal_fit} (>=10,000 >> D_A={D_A}) | N_val={n_cal_val} (>=2,000, template-disjoint)")
    assert n_cal_fit >= 10_000 and n_cal_val >= 2_000

    for li in compile_layers:
        np.savez(cache_dir / f"cal_fit_A_L{li}.npz", **cal_fit_A[li])
        np.savez(cache_dir / f"cal_val_A_L{li}.npz", **cal_val_A[li])
    del cal_fit_A, cal_val_A
    gc.collect()

    # Separate tiny pass (1024 tokens -> ~310 MB total!) for Donor background SwiGLU salience baseline!
    donor_bg_z_mlp = {
        li: st["z_mlp"]
        for li, st in extract_layer_token_states(
            model_A, tok_A, cal_corpus["fit"][:40], compile_layers, dev_A,
            batch_size=args.batch_size, max_length=160, positions="all_nonpad",
            capture_fields=("z_mlp",), max_tokens_cap=1024,
        ).items()
    }

    donor_val_last_logits = _last_tok_logits(model_A, tok_A, cal_corpus["val"], dev_A, bs=args.batch_size, v_cap=v_tok_A)
    donor_val_last_h27 = extract_layer_token_states(
        model_A, tok_A, cal_corpus["val"], [compile_layers[-1]], dev_A,
        batch_size=args.batch_size, max_length=160, positions="last", capture_fields=("h_out",),
    )[compile_layers[-1]]["h_out"]

    # 1B. Task A (SQL, max_length=384!): Eval Donor 7B across all 80 queries + extract Donor SQL train states
    donor_sql_all = eval_sql_queries_native(model_A, tok_A, dev_A, sql_bench, sql_gold, batch_size=8, v_shared=v_tok_A)
    donor_sql_unseen_40 = eval_sql_queries_native(model_A, tok_A, dev_A, sql_test, sql_gold, batch_size=8, v_shared=v_tok_A)
    donor_sql_train_preds = [
        donor_sql_all["_pred_sqls"][i] for i, it in enumerate(sql_bench) if it["schema_id"] in SQL_TRAIN_SCHEMAS
    ]
    sql_train_prompts = [format_sql_chat_prompt(tok_A, it["ddl"], it["question"]) for it in sql_train]
    sql_full_teacher_texts = [p + pred for p, pred in zip(sql_train_prompts, donor_sql_train_preds)]
    # Count tokens in first 30 SQL fit queries so dev LR tuning compiles strictly on `sql_fit_sub` (0 dev overlap!)
    n_sql_fit30_toks = sum(len(tok_A.encode(t, truncation=True, max_length=384, add_special_tokens=True)) for t in sql_full_teacher_texts[:30])
    donor_sql_states = extract_layer_token_states(
        model_A, tok_A, sql_full_teacher_texts, compile_layers, dev_A,
        batch_size=4, max_length=384, positions="all_nonpad",
        capture_fields=("y_attn", "h_out", "z_attn", "z_mlp"),
    )
    donor_sql_weights_fit30 = extract_donor_salient_heads_and_neurons(
        model_A, compile_layers,
        {li: donor_sql_states[li]["z_attn"][:n_sql_fit30_toks] for li in compile_layers},
        {li: donor_sql_states[li]["z_mlp"][:n_sql_fit30_toks] for li in compile_layers},
        donor_bg_z_mlp, add_q_heads=6, add_kv_heads=1, add_mlp_rows=args.add_mlp_rows,
    )
    donor_sql_weights = extract_donor_salient_heads_and_neurons(
        model_A, compile_layers,
        {li: donor_sql_states[li].pop("z_attn") for li in compile_layers},
        {li: donor_sql_states[li].pop("z_mlp") for li in compile_layers},
        donor_bg_z_mlp, add_q_heads=6, add_kv_heads=1, add_mlp_rows=args.add_mlp_rows,
    )
    gc.collect()
    print(f"  [Task A | Donor 7B SQL] 80q={donor_sql_all['exec_match_rate']:.3f} | 40q_unseen={donor_sql_unseen_40['exec_match_rate']:.3f} | syntax={donor_sql_all['syntax_valid_rate']:.3f}")

    # 1C. Task B (Airports): Screen strictly on `p0` ONLY (`p1`/`p2`/routes untouched!), early empty-cohort guard!
    air_all_p0 = [format_airport_prompt(ap["name"], 0)[0] for ap in airports]
    donor_p0_gens = greedy_generate_native(model_A, tok_A, dev_A, air_all_p0, max_new_tokens=6, batch_size=32, v_shared=v_tok_A)
    airports_7b_p0 = []
    for ap, g in zip(airports, donor_p0_gens):
        if check_iata_match(g, ap["iata"]):
            rec = dict(ap)
            rec["donor_p0_pred_iata"] = extract_iata_code(g) or ap["iata"]
            airports_7b_p0.append(rec)
    print(f"  [Task B | Donor 7B Airports (p0-only screen)] exact IATA: {len(airports_7b_p0)}/{len(airports)}")

    rng_air = np.random.default_rng(SEED + 7)
    perm_a = rng_air.permutation(len(airports_7b_p0)).tolist()
    n_air_dev_pool = min(max(4, int(round(0.30 * len(airports_7b_p0)))), max(len(airports_7b_p0) - 1, 0))
    air_dev_pool = [airports_7b_p0[i] for i in perm_a[:n_air_dev_pool]]
    air_eval_pool = [airports_7b_p0[i] for i in perm_a[n_air_dev_pool:]]

    donor_air_dev_st: Dict[int, Dict[str, np.ndarray]] = {}
    donor_air_weights: Dict[int, Dict[str, Any]] = {}
    donor_air_eval_pool_st: Dict[int, Dict[str, np.ndarray]] = {}
    if len(air_dev_pool) >= 4 and len(air_eval_pool) >= 8:
        donor_air_dev_st = extract_layer_token_states(
            model_A, tok_A, [format_airport_prompt(ap["name"], 0)[0] for ap in air_dev_pool], compile_layers, dev_A,
            batch_size=16, max_length=64, positions="last", capture_fields=("y_attn", "h_out", "z_attn", "z_mlp"),
        )
        donor_air_weights = extract_donor_salient_heads_and_neurons(
            model_A, compile_layers,
            {li: donor_air_dev_st[li].pop("z_attn") for li in compile_layers},
            {li: donor_air_dev_st[li].pop("z_mlp") for li in compile_layers},
            donor_bg_z_mlp, add_q_heads=6, add_kv_heads=1, add_mlp_rows=args.add_mlp_rows,
        )
        donor_air_eval_pool_st = extract_layer_token_states(
            model_A, tok_A, [format_airport_prompt(ap["name"], 0)[0] for ap in air_eval_pool], compile_layers, dev_A,
            batch_size=16, max_length=64, positions="last", capture_fields=("y_attn", "h_out"),
        )
        gc.collect()

    # 1D. Task C (2-Row 2-Hop): Verify Donor 7B ceiling (EM + FC + FC4!) + extract teacher logits/states
    d2_te_L = _last_tok_logits(model_A, tok_A, [x["donor_L_prompt"] for x in twohop_ds["test"]], dev_A, bs=16, v_cap=v_tok_A)
    d2_te_R = _last_tok_logits(model_A, tok_A, [x["donor_R_prompt"] for x in twohop_ds["test"]], dev_A, bs=16, v_cap=v_tok_A)
    donor_2hop_sum = _joint_2hop(
        _eval_2hop_logits(d2_te_L, twohop_ds["test"], tok_A, "L"),
        _eval_2hop_logits(d2_te_R, twohop_ds["test"], tok_A, "R"),
    )
    print(
        f"  [Task C | Donor 7B 2-Row 2-Hop Sealed] joint_em={donor_2hop_sum['joint_swap_em']:.3f} "
        f"(L={donor_2hop_sum['left_em']:.3f}, R={donor_2hop_sum['right_em']:.3f}) | "
        f"joint_fc={donor_2hop_sum['joint_swap_fc']:.3f} | joint_fc4={donor_2hop_sum['joint_swap_fc4']:.3f}"
    )

    d2_tr_L_logits = _last_tok_logits(model_A, tok_A, [x["donor_L_prompt"] for x in twohop_ds["train"]], dev_A, bs=16, v_cap=v_tok_A)
    d2_tr_R_logits = _last_tok_logits(model_A, tok_A, [x["donor_R_prompt"] for x in twohop_ds["train"]], dev_A, bs=16, v_cap=v_tok_A)
    d2_tr_L_st = extract_layer_token_states(
        model_A, tok_A, [x["donor_L_prompt"] for x in twohop_ds["train"]], compile_layers, dev_A,
        batch_size=8, max_length=256, positions="last", capture_fields=("y_attn", "h_out", "z_attn", "z_mlp"),
    )
    d2_tr_R_st = extract_layer_token_states(
        model_A, tok_A, [x["donor_R_prompt"] for x in twohop_ds["train"]], compile_layers, dev_A,
        batch_size=8, max_length=256, positions="last", capture_fields=("y_attn", "h_out", "z_attn", "z_mlp"),
    )
    # Also capture Donor 7B states on strictly held-out `twohop_ds["dev"]` at L20 & L27 for Gate 0B + Deafness Audit!
    g0b_layers = [20, 27]
    d2_dv_L_st = extract_layer_token_states(
        model_A, tok_A, [x["donor_L_prompt"] for x in twohop_ds["dev"]], g0b_layers, dev_A,
        batch_size=8, max_length=256, positions="last", capture_fields=("h_out", "z_attn", "z_mlp"),
    )
    d2_dv_R_st = extract_layer_token_states(
        model_A, tok_A, [x["donor_R_prompt"] for x in twohop_ds["dev"]], g0b_layers, dev_A,
        batch_size=8, max_length=256, positions="last", capture_fields=("h_out",),
    )
    donor_2hop_weights = {
        "L": extract_donor_salient_heads_and_neurons(
            model_A, compile_layers,
            {li: d2_tr_L_st[li].pop("z_attn") for li in compile_layers},
            {li: d2_tr_L_st[li].pop("z_mlp") for li in compile_layers},
            donor_bg_z_mlp, add_q_heads=6, add_kv_heads=1, add_mlp_rows=args.add_mlp_rows,
        ),
        "R": extract_donor_salient_heads_and_neurons(
            model_A, compile_layers,
            {li: d2_tr_R_st[li].pop("z_attn") for li in compile_layers},
            {li: d2_tr_R_st[li].pop("z_mlp") for li in compile_layers},
            donor_bg_z_mlp, add_q_heads=6, add_kv_heads=1, add_mlp_rows=args.add_mlp_rows,
        ),
    }
    # Keep ONLY the selected head/neuron dev columns for Deafness Audit (tiny <1 MB!)
    d_head_A = int(head_dim_A)
    donor_dev_deaf_ref: Dict[int, Dict[str, np.ndarray]] = {}
    for gl in g0b_layers:
        dw_L = donor_2hop_weights["L"][gl]
        q_cols = np.concatenate([np.arange(h * d_head_A, (h + 1) * d_head_A) for h in dw_L["sel_q_heads"]])
        m_cols = np.asarray(dw_L["sel_mlp_indices"], dtype=np.int64)
        donor_dev_deaf_ref[gl] = {
            "z_attn_sel": np.ascontiguousarray(d2_dv_L_st[gl].pop("z_attn")[:, q_cols].copy()),
            "z_mlp_sel": np.ascontiguousarray(d2_dv_L_st[gl].pop("z_mlp")[:, m_cols].copy()),
            "h_out_L": d2_dv_L_st[gl]["h_out"],
            "h_out_R": d2_dv_R_st[gl]["h_out"],
        }
    del d2_dv_L_st, d2_dv_R_st
    gc.collect()

    # ZERO-COPY STRICT UNLOAD OF DONOR 7B!
    unload_model_vram_safe(model_A)
    del model_A, donor_bg_z_mlp
    gc.collect()
    print(f"  [Phase 1 Complete in {time.time() - t_d0:.1f}s] Donor {args.donor_model} 100% UNLOADED (zero-copy VRAM free)!")

    # =========================================================================
    # PHASE 2: SOLO RECIPIENT 1.5B + SHARED VOCAB CHECK + WELL-CONDITIONED BRIDGES
    # =========================================================================
    model_B, tok_B, dev_B, _ = load_model_bf16(args.recipient_model, allow_download=args.allow_download)
    for p in model_B.parameters():
        p.requires_grad_(False)
    D_B = int(model_B.config.hidden_size)
    L_B = len(get_decoder_layers(model_B))
    v_B_out = int(model_B.lm_head.out_features)
    assert L_A == L_B, f"Stage 1 requires identical depth: {L_A} != {L_B}"
    assert tok_A_vocab == tok_B.get_vocab(), "Stage 1 requires exact token-to-ID mapping identity!"
    V_shared = min(v_tok_A, len(tok_B), v_A_out, v_B_out)
    print(f"  [Phase 2] Loaded SOLO Recipient {args.recipient_model} (L={L_B}, D={D_B}, lm_head={v_B_out}, V_shared={V_shared})")

    base_loc, base_loc_ok = eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, batch_size=32, v_shared=V_shared)
    print(f"  [Certified 380-Fact Locality Baseline] {base_loc['n_correct']}/{base_loc['n_total']} ({base_loc['acc']:.3f})")
    assert base_loc["n_correct"] >= 250, f"Locality baseline unexpectedly low: {base_loc['n_correct']}/380!"

    probe_enc = tok_B(cal_corpus["val"][:4], return_tensors="pt", padding=True, truncation=True, max_length=64)
    probe_ids, probe_mask = probe_enc["input_ids"].to(dev_B), probe_enc["attention_mask"].to(dev_B)
    with torch.no_grad():
        base_probe_logits = model_B(input_ids=probe_ids, attention_mask=probe_mask, use_cache=False).logits.detach().cpu()

    # Fit exact RMSNorm-aligned bridges on template-disjoint calibration splits (N_fit >= 10k, N_val >= 2k!)
    cal_fit_B = extract_layer_token_states(
        model_B, tok_B, cal_corpus["fit"], compile_layers, dev_B,
        batch_size=args.batch_size, max_length=160, positions="all_nonpad",
        capture_fields=("u_in", "y_attn", "u_post", "h_out"), max_tokens_cap=args.cal_fit_cap,
    )
    cal_val_B = extract_layer_token_states(
        model_B, tok_B, cal_corpus["val"], compile_layers, dev_B,
        batch_size=args.batch_size, max_length=160, positions="all_nonpad",
        capture_fields=("u_in", "y_attn", "u_post", "h_out"), max_tokens_cap=args.cal_val_cap,
    )

    bridges: Dict[int, Dict[str, np.ndarray]] = {}
    bridge_diag: Dict[str, Any] = {}
    for li in compile_layers:
        fA = np.load(cache_dir / f"cal_fit_A_L{li}.npz")
        vA = np.load(cache_dir / f"cal_val_A_L{li}.npz")
        fB, vB = cal_fit_B[li], cal_val_B[li]
        assert len(fA["h_out"]) == len(fB["h_out"]) and len(vA["h_out"]) == len(vB["h_out"])

        P_attn, i_pa = fit_val_selected_ridge_bridge(fB["u_in"], fA["u_in"], vB["u_in"], vA["u_in"])
        P_mlp, i_pm = fit_val_selected_ridge_bridge(fB["u_post"], fA["u_post"], vB["u_post"], vA["u_post"])
        M_attn, i_ma = fit_val_selected_ridge_bridge(fA["y_attn"], fB["y_attn"], vA["y_attn"], vB["y_attn"])
        M_out, i_mo = fit_val_selected_ridge_bridge(fA["h_out"], fB["h_out"], vA["h_out"], vB["h_out"])
        bridges[li] = {"P_attn": P_attn, "P_mlp": P_mlp, "M_attn": M_attn, "M_out": M_out}
        bridge_diag[f"L{li}"] = {"P_attn": i_pa, "P_mlp": i_pm, "M_attn": i_ma, "M_out": i_mo}
        print(
            f"  [Bridge L{li:02d} | N_fit={i_mo['n_fit']}, N_val={i_mo['n_val']}, erank={i_mo['erank']:.1f}] "
            f"P_attn={i_pa['val_cos']:.4f} | P_mlp={i_pm['val_cos']:.4f} | M_attn={i_ma['val_cos']:.4f} | M_out={i_mo['val_cos']:.4f} (R2={i_mo['val_r2']:.4f})"
        )
    # Gate 0B: Rectangular Width-Bridge (`3584 <-> 1536`, 2,048-D Nullspace) & Curvature Diagnostic (L20 & L27)!
    # Strictly fit on `cal_fit` / `twohop["train"]`, evaluated on held-out `cal_val` / `twohop["dev"]` (ZERO test leakage!)
    r2_tr_g0b = extract_layer_token_states(
        model_B, tok_B, [x["recipient_prompt"] for x in twohop_ds["train"]], g0b_layers, dev_B,
        batch_size=16, max_length=128, positions="last", capture_fields=("h_out",),
    )
    r2_dv_g0b = extract_layer_token_states(
        model_B, tok_B, [x["recipient_prompt"] for x in twohop_ds["dev"]], g0b_layers, dev_B,
        batch_size=16, max_length=128, positions="last", capture_fields=("h_out",),
    )
    gate0b_diag: Dict[str, Any] = {}
    for gl in g0b_layers:
        fA_g = np.load(cache_dir / f"cal_fit_A_L{gl}.npz")
        vA_g = np.load(cache_dir / f"cal_val_A_L{gl}.npz")
        hA_tr_mid = 0.5 * (bf16_to_f64_np(d2_tr_L_st[gl]["h_out"]) + bf16_to_f64_np(d2_tr_R_st[gl]["h_out"]))
        hA_dv_mid = 0.5 * (bf16_to_f64_np(donor_dev_deaf_ref[gl]["h_out_L"]) + bf16_to_f64_np(donor_dev_deaf_ref[gl]["h_out_R"]))
        hA_dv_con = bf16_to_f64_np(donor_dev_deaf_ref[gl]["h_out_L"]) - bf16_to_f64_np(donor_dev_deaf_ref[gl]["h_out_R"])
        g0b_l = run_gate0b_rectangular_bridge_diagnostic(
            fA_g["h_out"], cal_fit_B[gl]["h_out"], vA_g["h_out"], cal_val_B[gl]["h_out"],
            X_task_fit=hA_tr_mid, Y_task_fit=r2_tr_g0b[gl]["h_out"],
            X_task_dev=hA_dv_mid, Y_task_dev=r2_dv_g0b[gl]["h_out"],
            H_A_contrast_dev=hA_dv_con, n_anchors=256, seed=SEED + gl,
        )
        # Measure Pulled-Back Deafness (`cos(Z_pulled_back, Z_donor)`) on held-out `twohop_ds["dev"]`!
        dw_L = donor_2hop_weights["L"][gl]
        P_a, P_m = bridges[gl]["P_attn"], bridges[gl]["P_mlp"]
        Wq_p = torch.from_numpy((dw_L["q_w"].astype(np.float64) @ P_a).astype(np.float32))
        Wk_p = torch.from_numpy((dw_L["k_w"].astype(np.float64) @ P_a).astype(np.float32))
        Wv_p = torch.from_numpy((dw_L["v_w"].astype(np.float64) @ P_a).astype(np.float32))
        bq_p = torch.from_numpy(dw_L["q_b"].astype(np.float32)) if dw_L.get("q_b") is not None else None
        bk_p = torch.from_numpy(dw_L["k_b"].astype(np.float32)) if dw_L.get("k_b") is not None else None
        bv_p = torch.from_numpy(dw_L["v_b"].astype(np.float32)) if dw_L.get("v_b") is not None else None
        Wg_p = torch.from_numpy((dw_L["gate_w"].astype(np.float64) @ P_m).astype(np.float32))
        Wu_p = torch.from_numpy((dw_L["up_w"].astype(np.float64) @ P_m).astype(np.float32))
        pk_deaf = ReversibleNativeOrganPack(
            attn_widen={gl: {"q_w": Wq_p, "q_b": bq_p, "k_w": Wk_p, "k_b": bk_p, "v_w": Wv_p, "v_b": bv_p,
                             "o_w": torch.zeros((D_B, int(Wq_p.shape[0])), dtype=torch.float32),
                             "add_q_heads": int(dw_L["add_q_heads"]), "add_kv_heads": int(dw_L["add_kv_heads"])}},
            mlp_widen={gl: {"gate_w": Wg_p, "up_w": Wu_p, "down_w": torch.zeros((D_B, int(Wg_p.shape[0])), dtype=torch.float32)}},
        )
        pk_deaf.apply(model_B)
        try:
            st_pb = extract_layer_token_states(
                model_B, tok_B, [x["recipient_prompt"] for x in twohop_ds["dev"]], [gl], dev_B,
                batch_size=16, max_length=128, positions="last", capture_fields=("z_attn", "z_mlp"),
                z_attn_tail_slice=int(Wq_p.shape[0]), z_mlp_tail_slice=int(Wg_p.shape[0]),
            )[gl]
        finally:
            pk_deaf.revert(model_B)
        pk_deaf.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
        za_pb, za_dn = bf16_to_f64_np(st_pb["z_attn"]), bf16_to_f64_np(donor_dev_deaf_ref[gl]["z_attn_sel"])
        zm_pb, zm_dn = bf16_to_f64_np(st_pb["z_mlp"]), bf16_to_f64_np(donor_dev_deaf_ref[gl]["z_mlp_sel"])
        cos_a_pb = float(np.mean(np.sum(za_pb * za_dn, axis=1) / (np.linalg.norm(za_pb, axis=1) * np.linalg.norm(za_dn, axis=1) + 1e-12)))
        cos_m_pb = float(np.mean(np.sum(zm_pb * zm_dn, axis=1) / (np.linalg.norm(zm_pb, axis=1) * np.linalg.norm(zm_dn, axis=1) + 1e-12)))
        g0b_l["pulled_back_deafness_audit"] = {
            "cos_z_attn_pulled_back_vs_donor": round(cos_a_pb, 4),
            "cos_z_mlp_pulled_back_vs_donor": round(cos_m_pb, 4),
        }
        gate0b_diag[f"L{gl}"] = g0b_l
        print(
            f"  [Gate 0B L{gl:02d}] Ridge_val={g0b_l['1_global_prose_ridge']['cal_val']['cos']:.4f} | "
            f"CCA_val={g0b_l['2_whitened_cca_procrustes']['cal_val']['cos']:.4f} (top1536_var={g0b_l['2_whitened_cca_procrustes']['top_db_pc_variance_frac']:.3f}) | "
            f"Local_dev={g0b_l.get('3_local_skill_neighborhood_bridge', {}).get('task_dev', {}).get('cos', 0):.4f} | "
            f"NL_SwiGLU_val={g0b_l['4_nonlinear_anchor_swiglu_kernel']['cal_val']['cos']:.4f} (dev={g0b_l['4_nonlinear_anchor_swiglu_kernel'].get('task_dev', {}).get('cos', 0):.4f}) | "
            f"Deafness(attn={cos_a_pb:.3f}, mlp={cos_m_pb:.3f}) | ContrastSurv={g0b_l['nullspace_contrast_survival']['contrast_energy_in_bridge_rowspace']:.3f}"
        )

    del cal_fit_B, cal_val_B, r2_tr_g0b, r2_dv_g0b, donor_dev_deaf_ref
    shutil.rmtree(cache_dir, ignore_errors=True)
    gc.collect()

    # Gate 0 Diagnostic on strictly shared valid vocab `[:V_shared]`!
    with torch.no_grad():
        h27_mapped = torch.from_numpy((bf16_to_f64_np(donor_val_last_h27) @ bridges[compile_layers[-1]]["M_out"].T).astype(np.float32)).to(dev_B, dtype=model_B.dtype)
        mapped_lg = model_B.lm_head(model_B.model.norm(h27_mapped))[:, :V_shared].float().cpu()
        recip_val_lg = _last_tok_logits(model_B, tok_B, cal_corpus["val"], dev_B, bs=args.batch_size, v_cap=V_shared)
        donor_val_sh = donor_val_last_logits[:, :V_shared]
    top1_mapped_vs_7b = float((mapped_lg.argmax(-1) == donor_val_sh.argmax(-1)).float().mean().item())
    top1_base1p5_vs_7b = float((recip_val_lg.argmax(-1) == donor_val_sh.argmax(-1)).float().mean().item())
    kl_mapped_to_7b = float(F.kl_div(F.log_softmax(mapped_lg, dim=-1), F.softmax(donor_val_sh, dim=-1), reduction="batchmean").item())
    gate0_diag = {
        "n_val_passages": len(cal_corpus["val"]), "v_shared": V_shared,
        "mapped_L27_top1_agree_with_7b": round(top1_mapped_vs_7b, 4),
        "base_1p5b_top1_agree_with_7b": round(top1_base1p5_vs_7b, 4),
        "mapped_L27_kl_to_7b": round(kl_mapped_to_7b, 4),
    }
    print(f"  [Gate 0 L27 Width-Bridge] mapped_top1_agree_7B={top1_mapped_vs_7b:.3f} (base_1.5B={top1_base1p5_vs_7b:.3f}) | KL={kl_mapped_to_7b:.3f}")

    results: Dict[str, Any] = {
        "meta": {
            "experiment": "v4.1_stage1_qwen7b_to_1p5b_native",
            "donor_model": args.donor_model, "recipient_model": args.recipient_model,
            "compile_layers": compile_layers, "v_shared": V_shared,
            "n_cal_fit_tokens": n_cal_fit, "n_cal_val_tokens": n_cal_val,
            "zero_external_sidecars": True, "zero_runtime_hooks_at_inference": True,
            "cache_contract": "use_cache=False or fresh DynamicCache (no StaticCache)",
            "seed": SEED,
        },
        "bridge_diagnostics": bridge_diag,
        "gate0_l27_bridge_diagnostic": gate0_diag,
        "gate0b_rectangular_width_and_curvature": gate0b_diag,
        "base_locality_380": base_loc,
    }

    # =========================================================================
    # TASK A: REAL TEXT-TO-SQLITE (4 Train Schemas -> 4 Strictly Unseen Schemas)
    # =========================================================================
    print("\n================ TASK A: TEXT-TO-SQLITE (4 Train Schemas -> 4 Unseen Schemas) ================")
    base_sql_all = eval_sql_queries_native(model_B, tok_B, dev_B, sql_bench, sql_gold, batch_size=8, v_shared=V_shared)
    base_sql_test = eval_sql_queries_native(model_B, tok_B, dev_B, sql_test, sql_gold, batch_size=8, v_shared=V_shared)
    print(
        f"  [SQL Headroom] 80q: 7B={donor_sql_all['exec_match_rate']:.3f} vs 1.5B={base_sql_all['exec_match_rate']:.3f} | "
        f"40q_unseen: 7B={donor_sql_unseen_40['exec_match_rate']:.3f} vs 1.5B={base_sql_test['exec_match_rate']:.3f} "
        f"(unseen gap={100*(donor_sql_unseen_40['exec_match_rate']-base_sql_test['exec_match_rate']):+.1f} pp)"
    )

    sql_tiers: Dict[str, Any] = {}

    pk_sql_1a, d_sql_1a = compile_tier1a_sequential_native_delta_w(
        model_B, tok_B, dev_B, compile_layers, bridges, donor_sql_states,
        sql_full_teacher_texts, cal_corpus["fit"][:40], positions="all_nonpad",
        lam_bg=5.0, lam_eye=0.05, batch_size=4, max_length=384,
    )
    pk_sql_1a.apply(model_B)
    r_sql_1a = eval_sql_queries_native(model_B, tok_B, dev_B, sql_test, sql_gold, batch_size=8, v_shared=V_shared)
    l_sql_1a, _ = eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)
    pk_sql_1a.revert(model_B)
    rb_sql_1a = pk_sql_1a.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
    sql_tiers["Tier1A_NATIVE_SEQ_DELTA_W_0SGD"] = {
        "unseen_exec_rate": r_sql_1a["exec_match_rate"], "unseen_syntax_rate": r_sql_1a["syntax_valid_rate"],
        "by_schema": r_sql_1a["by_schema"], "locality_380": l_sql_1a, "n_bytes_bf16": pk_sql_1a.n_bytes_bf16(),
        "compile_sec": d_sql_1a["compile_sec"], "rollback": rb_sql_1a,
    }
    print(f"  [SQL Tier 1A Native Seq ΔW 0-SGD] unseen_exec={r_sql_1a['exec_match_rate']:.3f} | loc_broken={l_sql_1a['newly_broken']} | rb=True")

    pk_sql_1b, d_sql_1b = compile_tier1b_native_widened_heads_and_swiglu(
        model_B, tok_B, dev_B, compile_layers, bridges, donor_sql_weights, donor_sql_states,
        sql_full_teacher_texts, cal_corpus["fit"][:40], positions="all_nonpad",
        lam_bg=5.0, lam_eye=0.05, batch_size=4, max_length=384,
    )
    pk_sql_1b.apply(model_B)
    r_sql_1b = eval_sql_queries_native(model_B, tok_B, dev_B, sql_test, sql_gold, batch_size=8, v_shared=V_shared)
    l_sql_1b, _ = eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)
    pk_sql_1b.revert(model_B)
    rb_sql_1b = pk_sql_1b.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
    sql_tiers["Tier1B_NATIVE_WIDENED_GQA_SWIGLU_0SGD"] = {
        "unseen_exec_rate": r_sql_1b["exec_match_rate"], "unseen_syntax_rate": r_sql_1b["syntax_valid_rate"],
        "by_schema": r_sql_1b["by_schema"], "locality_380": l_sql_1b, "n_bytes_bf16": pk_sql_1b.n_bytes_bf16(),
        "compile_sec": d_sql_1b["compile_sec"], "rollback": rb_sql_1b,
    }
    print(f"  [SQL Tier 1B Native Widened 0-SGD] unseen_exec={r_sql_1b['exec_match_rate']:.3f} | bytes={pk_sql_1b.n_bytes_bf16()/1e6:.1f}MB | loc_broken={l_sql_1b['newly_broken']} | rb=True")

    # Tier 1C: Native Anchor-SwiGLU Unroller (0-SGD, 100% Native Qwen2MLP, dev-tuned tau in {1, 2, 4}!)
    donor_sql_fit30_st = {li: {k: v[:n_sql_fit30_toks] for k, v in donor_sql_states[li].items()} for li in compile_layers}
    best_sql_tau, best_sql_dev_1c = 2.0, -1.0
    for cand_tau in (1.0, 2.0, 4.0):
        pk_c, _ = compile_tier1c_native_anchor_swiglu_unroller(
            model_B, tok_B, dev_B, compile_layers, bridges, donor_sql_fit30_st,
            sql_full_teacher_texts[:30], cal_corpus["fit"][:40], n_anchors=args.add_mlp_rows, tau=cand_tau,
            positions="all_nonpad", lam_bg=5.0, lam_eye=0.05, batch_size=4, max_length=384, seed=SEED + 15,
        )
        pk_c.apply(model_B)
        sc_d = eval_sql_queries_native(model_B, tok_B, dev_B, sql_dev_sub, sql_gold, batch_size=8, v_shared=V_shared)["exec_match_rate"]
        pk_c.revert(model_B)
        pk_c.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
        if sc_d > best_sql_dev_1c:
            best_sql_dev_1c, best_sql_tau = sc_d, cand_tau
    pk_sql_1c, d_sql_1c = compile_tier1c_native_anchor_swiglu_unroller(
        model_B, tok_B, dev_B, compile_layers, bridges, donor_sql_states,
        sql_full_teacher_texts, cal_corpus["fit"][:40], n_anchors=args.add_mlp_rows, tau=best_sql_tau,
        positions="all_nonpad", lam_bg=5.0, lam_eye=0.05, batch_size=4, max_length=384, seed=SEED + 16,
    )
    pk_sql_1c.apply(model_B)
    r_sql_1c = eval_sql_queries_native(model_B, tok_B, dev_B, sql_test, sql_gold, batch_size=8, v_shared=V_shared)
    l_sql_1c, _ = eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)
    pk_sql_1c.revert(model_B)
    rb_sql_1c = pk_sql_1c.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
    sql_tiers["Tier1C_NATIVE_ANCHOR_SWIGLU_0SGD"] = {
        "unseen_exec_rate": r_sql_1c["exec_match_rate"], "unseen_syntax_rate": r_sql_1c["syntax_valid_rate"],
        "by_schema": r_sql_1c["by_schema"], "locality_380": l_sql_1c, "n_bytes_bf16": pk_sql_1c.n_bytes_bf16(),
        "tau": best_sql_tau, "compile_sec": d_sql_1c["compile_sec"], "rollback": rb_sql_1c,
    }
    print(f"  [SQL Tier 1C Native Anchor-SwiGLU] unseen_exec={r_sql_1c['exec_match_rate']:.3f} (tau={best_sql_tau}) | bytes={pk_sql_1c.n_bytes_bf16()/1e6:.1f}MB | loc_broken={l_sql_1c['newly_broken']} | rb=True")

    # Dev LR selection strictly compiling initializer AND selecting heads/neurons on 30 `sql_fit_sub` queries (0 dev overlap!)
    pk_sql_fit30_1b, _ = compile_tier1b_native_widened_heads_and_swiglu(
        model_B, tok_B, dev_B, compile_layers, bridges, donor_sql_weights_fit30, donor_sql_fit30_st,
        sql_full_teacher_texts[:30], cal_corpus["fit"][:40], positions="all_nonpad",
        lam_bg=5.0, lam_eye=0.05, batch_size=4, max_length=384,
    )
    del donor_sql_states, donor_sql_fit30_st, donor_sql_weights, donor_sql_weights_fit30
    gc.collect()

    sql_fit_pairs = list(zip(sql_train_prompts[:30], donor_sql_train_preds[:30]))
    sql_all_pairs = list(zip(sql_train_prompts, donor_sql_train_preds))
    best_sql_lr_t2, best_sql_dev_t2 = 1e-4, -1.0
    for cand_lr in (1e-4, 3e-4):
        pk_c, _ = polish_widened_native_pack_sft(model_B, tok_B, dev_B, pk_sql_fit30_1b, sql_fit_pairs, v_shared=V_shared, lr=cand_lr, steps=20, batch_size=4, seed=SEED + 10)
        pk_c.apply(model_B)
        dev_sc = eval_sql_queries_native(model_B, tok_B, dev_B, sql_dev_sub, sql_gold, batch_size=8, v_shared=V_shared)["exec_match_rate"]
        pk_c.revert(model_B)
        pk_c.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
        if dev_sc > best_sql_dev_t2:
            best_sql_dev_t2, best_sql_lr_t2 = dev_sc, cand_lr
    print(f"  [SQL Dev LR Tuning] Selected Tier 2 lr={best_sql_lr_t2} (dev_exec={best_sql_dev_t2:.3f})")

    SQL_STEPS = 40
    for tname, init_pk in (
        ("Tier2_NATIVE_INIT_TINY_POLISH", pk_sql_1b),
        ("Tier2_CTRL_RECIPIENT_ONLY_NATIVE", make_random_zero_control_pack(pk_sql_1b, seed=SEED + 11)),
    ):
        pk_t2, info_t2 = polish_widened_native_pack_sft(
            model_B, tok_B, dev_B, init_pk, sql_all_pairs, v_shared=V_shared, lr=best_sql_lr_t2, steps=SQL_STEPS, batch_size=4, seed=SEED + 12,
        )
        pk_t2.apply(model_B)
        r_t2 = eval_sql_queries_native(model_B, tok_B, dev_B, sql_test, sql_gold, batch_size=8, v_shared=V_shared)
        l_t2, _ = eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)
        pk_t2.revert(model_B)
        rb_t2 = pk_t2.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
        sql_tiers[tname] = {
            "unseen_exec_rate": r_t2["exec_match_rate"], "unseen_syntax_rate": r_t2["syntax_valid_rate"],
            "by_schema": r_t2["by_schema"], "locality_380": l_t2, "n_bytes_bf16": pk_t2.n_bytes_bf16(),
            "train_info": info_t2, "rollback": rb_t2,
        }
        print(f"  [SQL {tname:32s}] unseen_exec={r_t2['exec_match_rate']:.3f} | loss={info_t2['loss_start']}->{info_t2['loss_end']} | loc_broken={l_t2['newly_broken']} | rb=True")

    best_sql_lr_t3, best_sql_dev_t3 = 1e-4, -1.0
    for cand_lr in (1e-4, 3e-4):
        pk_c, _ = train_tier3_teacher_lora_merged_native(model_B, tok_B, dev_B, compile_layers, mode="sft", prompt_target_pairs=sql_fit_pairs, v_shared=V_shared, lr=cand_lr, steps=20, batch_size=4, seed=SEED + 13)
        pk_c.apply(model_B)
        dev_sc = eval_sql_queries_native(model_B, tok_B, dev_B, sql_dev_sub, sql_gold, batch_size=8, v_shared=V_shared)["exec_match_rate"]
        pk_c.revert(model_B)
        pk_c.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
        if dev_sc > best_sql_dev_t3:
            best_sql_dev_t3, best_sql_lr_t3 = dev_sc, cand_lr

    pk_sql_3, info_sql_3 = train_tier3_teacher_lora_merged_native(
        model_B, tok_B, dev_B, compile_layers, mode="sft", prompt_target_pairs=sql_all_pairs,
        v_shared=V_shared, rank=16, alpha=32.0, lr=best_sql_lr_t3, steps=SQL_STEPS, batch_size=4, seed=SEED + 14,
    )
    pk_sql_3.apply(model_B)
    r_sql_3 = eval_sql_queries_native(model_B, tok_B, dev_B, sql_test, sql_gold, batch_size=8, v_shared=V_shared)
    l_sql_3, _ = eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)
    pk_sql_3.revert(model_B)
    rb_sql_3 = pk_sql_3.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
    sql_tiers["Tier3_TEACHER_LORA_MERGED_NATIVE"] = {
        "unseen_exec_rate": r_sql_3["exec_match_rate"], "unseen_syntax_rate": r_sql_3["syntax_valid_rate"],
        "by_schema": r_sql_3["by_schema"], "locality_380": l_sql_3,
        "raw_lora_factor_bytes_bf16": info_sql_3["raw_lora_factor_bytes_bf16"],
        "merged_dense_delta_bytes_bf16": info_sql_3["merged_dense_delta_bytes_bf16"],
        "train_info": info_sql_3, "rollback": rb_sql_3,
    }
    print(f"  [SQL Tier 3 Merged Teacher-LoRA   ] unseen_exec={r_sql_3['exec_match_rate']:.3f} | lr={best_sql_lr_t3} | loss={info_sql_3['loss_start']}->{info_sql_3['loss_end']} | loc_broken={l_sql_3['newly_broken']} | rb=True")

    results["task_a_text_to_sqlite"] = {
        "donor_7b_all_80": {k: v for k, v in donor_sql_all.items() if not k.startswith("_")},
        "donor_7b_unseen_40": {k: v for k, v in donor_sql_unseen_40.items() if not k.startswith("_")},
        "base_1p5b_all_80": {k: v for k, v in base_sql_all.items() if not k.startswith("_")},
        "base_1p5b_unseen_40": {k: v for k, v in base_sql_test.items() if not k.startswith("_")},
        "headroom_gap_all80_pp": round(100.0 * (donor_sql_all["exec_match_rate"] - base_sql_all["exec_match_rate"]), 2),
        "headroom_gap_unseen40_pp": round(100.0 * (donor_sql_unseen_40["exec_match_rate"] - base_sql_test["exec_match_rate"]), 2),
        "tiers": sql_tiers,
    }
    del pk_sql_1a, pk_sql_1b, pk_sql_1c, pk_sql_fit30_1b, pk_sql_3
    gc.collect()

    # =========================================================================
    # TASK B: OURAIRPORTS IATA ENRICHMENT (Strict Anti-Leakage: p0-Screen, Disjoint Dev/Eval!)
    # =========================================================================
    print("\n================ TASK B: OURAIRPORTS IATA ENRICHMENT (p0-Screen, Disjoint Dev/Eval) ================")
    dev_airports: List[Dict[str, Any]] = []
    eval_airports: List[Dict[str, Any]] = []
    routes_eval: List[Dict[str, Any]] = []
    air_tiers: Dict[str, Any] = {}
    g1a_out: Dict[str, Any] = {"skipped": True}

    if len(air_dev_pool) >= 4 and len(air_eval_pool) >= 8:
        dev_p0_gens = greedy_generate_native(model_B, tok_B, dev_B, [format_airport_prompt(ap["name"], 0)[0] for ap in air_dev_pool], max_new_tokens=6, batch_size=32, v_shared=V_shared)
        eval_p0_gens = greedy_generate_native(model_B, tok_B, dev_B, [format_airport_prompt(ap["name"], 0)[0] for ap in air_eval_pool], max_new_tokens=6, batch_size=32, v_shared=V_shared)
        dev_blind_idx = [i for i, (ap, g) in enumerate(zip(air_dev_pool, dev_p0_gens)) if not check_iata_match(g, ap["iata"])]
        eval_blind_idx = [i for i, (ap, g) in enumerate(zip(air_eval_pool, eval_p0_gens)) if not check_iata_match(g, ap["iata"])]
        dev_airports = [air_dev_pool[i] for i in dev_blind_idx]
        eval_airports = [air_eval_pool[i] for i in eval_blind_idx]
        routes_eval = build_route_items(eval_airports, max_routes=48, seed=SEED + 17)
        print(f"  [Airports Partition] Dev-pool blindspots={len(dev_airports)}/{len(air_dev_pool)} | Eval-pool blindspots={len(eval_airports)}/{len(air_eval_pool)} | Routes={len(routes_eval)}")

    if len(eval_airports) >= 8:
        # Gate 1a: Multi-token full 3-letter IATA decode via temporary mid-layer (L18) prefill injection
        # (handles 2-token codes like " LAX" cleanly; temporary diagnostic hook strictly removed in `finally:`!)
        l_mid = 18
        hA_eval_mid = bf16_to_f64_np(donor_air_eval_pool_st[l_mid]["h_out"][eval_blind_idx])
        h_mapped_mid = torch.from_numpy((hA_eval_mid @ bridges[l_mid]["M_out"].T).astype(np.float32))
        h_perm_mid = h_mapped_mid.roll(shifts=1, dims=0)
        eval_p0_prompts = [format_airport_prompt(ap["name"], 0)[0] for ap in eval_airports]

        def _gate1a_midlayer_generate(mapped_vecs: Optional[torch.Tensor]) -> float:
            if mapped_vecs is None:
                gs = greedy_generate_native(model_B, tok_B, dev_B, eval_p0_prompts, max_new_tokens=6, batch_size=16, v_shared=V_shared)
            else:
                gs = []
                old_pad = tok_B.padding_side
                tok_B.padding_side = "left"
                pad_id = tok_B.pad_token_id if tok_B.pad_token_id is not None else tok_B.eos_token_id
                lproc = [_MaskPaddedVocabLogitsProcessor(V_shared)]
                try:
                    for i0 in range(0, len(eval_p0_prompts), 16):
                        bp = eval_p0_prompts[i0 : i0 + 16]
                        enc = tok_B(bp, return_tensors="pt", padding=True, add_special_tokens=True)
                        ids, mask = enc["input_ids"].to(dev_B), enc["attention_mask"].to(dev_B)
                        plen = ids.shape[1]
                        v_batch = mapped_vecs[i0 : i0 + len(bp)].to(dev_B, dtype=model_B.dtype)
                        fired = {"done": False}
                        def _inj(_m, _i, out):
                            if fired["done"]:
                                return out
                            is_tup = isinstance(out, tuple)
                            h = out[0] if is_tup else out
                            if h.shape[1] == plen:
                                fired["done"] = True
                                h = h.clone()
                                h[:, -1, :] = v_batch
                                return (h,) + out[1:] if is_tup else h
                            return out
                        hdl = get_decoder_layers(model_B)[l_mid].register_forward_hook(_inj)
                        try:
                            with torch.no_grad():
                                gen = model_B.generate(input_ids=ids, attention_mask=mask, max_new_tokens=6, do_sample=False, use_cache=True, pad_token_id=pad_id, logits_processor=lproc)
                        finally:
                            hdl.remove()
                        for bi in range(len(bp)):
                            gs.append(tok_B.decode(gen[bi, plen:], skip_special_tokens=True))
                finally:
                    tok_B.padding_side = old_pad
                assert_zero_runtime_hooks(model_B)
            hits = sum(int(check_iata_match(g, ap["iata"])) for g, ap in zip(gs, eval_airports))
            return round(hits / max(len(eval_airports), 1), 4)

        g1a_out = {
            "skipped": False,
            "inject_layer": l_mid,
            "mapped_full_iata": _gate1a_midlayer_generate(h_mapped_mid),
            "self_full_iata": _gate1a_midlayer_generate(None),
            "perm_full_iata": _gate1a_midlayer_generate(h_perm_mid),
        }
        print(f"  [Airports Gate 1a L{l_mid} Multi-Token IATA] MAPPED={g1a_out['mapped_full_iata']:.3f} vs SELF={g1a_out['self_full_iata']:.3f}, PERM={g1a_out['perm_full_iata']:.3f}")

        def _eval_air_cohort(cohort: Sequence[Dict[str, Any]], routes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
            out_p = {}
            for pi in (0, 1, 2):
                prs = [format_airport_prompt(ap["name"], pi)[0] for ap in cohort]
                gs = greedy_generate_native(model_B, tok_B, dev_B, prs, max_new_tokens=6, batch_size=32, v_shared=V_shared)
                k = sum(int(check_iata_match(g, ap["iata"])) for g, ap in zip(gs, cohort))
                out_p[f"p{pi}"] = {"acc": round(k / max(len(cohort), 1), 4), "k": k, "n": len(cohort)}
            uk = out_p["p1"]["k"] + out_p["p2"]["k"]
            un = out_p["p1"]["n"] + out_p["p2"]["n"]
            out_p["unseen_p1_p2_pooled"] = {"acc": round(uk / max(un, 1), 4), "k": uk, "n": un}
            if routes:
                r_gs = greedy_generate_native(model_B, tok_B, dev_B, [r["prompt"] for r in routes], max_new_tokens=12, batch_size=32, v_shared=V_shared)
                both, ordered, either = 0, 0, 0
                for r, g in zip(routes, r_gs):
                    rm = check_route_match(g, r["iata_a"], r["iata_b"])
                    both += int(rm["both"])
                    ordered += int(rm["ordered"])
                    either += int(rm["either"])
                out_p["routes"] = {
                    "n": len(routes),
                    "both_acc": round(both / len(routes), 4),
                    "ordered_acc": round(ordered / len(routes), 4),
                    "either_acc": round(either / len(routes), 4),
                }
            return out_p

        air_tiers["BASE"] = _eval_air_cohort(eval_airports, routes_eval)
        donor_air_eval_st = {li: {k: v[eval_blind_idx] for k, v in donor_air_eval_pool_st[li].items()} for li in compile_layers}
        eval_air_pairs = [(format_airport_prompt(ap["name"], 0)[0], " " + ap["donor_p0_pred_iata"]) for ap in eval_airports]
        dev_air_pairs = [(format_airport_prompt(ap["name"], 0)[0], " " + ap["donor_p0_pred_iata"]) for ap in dev_airports]

        pk_a1a, _ = compile_tier1a_sequential_native_delta_w(
            model_B, tok_B, dev_B, compile_layers, bridges, donor_air_eval_st,
            eval_p0_prompts, cal_corpus["fit"][:40], positions="last", lam_bg=5.0, lam_eye=0.05, batch_size=16, max_length=64,
        )
        pk_a1a.apply(model_B)
        ev_a1a = _eval_air_cohort(eval_airports, routes_eval)
        lc_a1a, _ = eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)
        pk_a1a.revert(model_B)
        rb_a1a = pk_a1a.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
        air_tiers["Tier1A_NATIVE_SEQ_DELTA_W_0SGD"] = {**ev_a1a, "locality_380": lc_a1a, "n_bytes_bf16": pk_a1a.n_bytes_bf16(), "rollback": rb_a1a}
        print(f"  [Air Tier 1A Native Seq ΔW 0-SGD] p0={ev_a1a['p0']['acc']:.3f} unseen={ev_a1a['unseen_p1_p2_pooled']['acc']:.3f} | loc_broken={lc_a1a['newly_broken']} | rb=True")

        pk_a1b, _ = compile_tier1b_native_widened_heads_and_swiglu(
            model_B, tok_B, dev_B, compile_layers, bridges, donor_air_weights, donor_air_eval_st,
            eval_p0_prompts, cal_corpus["fit"][:40], positions="last", lam_bg=5.0, lam_eye=0.05, batch_size=16, max_length=64,
        )
        pk_a1b.apply(model_B)
        ev_a1b = _eval_air_cohort(eval_airports, routes_eval)
        lc_a1b, _ = eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)
        pk_a1b.revert(model_B)
        rb_a1b = pk_a1b.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
        air_tiers["Tier1B_NATIVE_WIDENED_GQA_SWIGLU_0SGD"] = {**ev_a1b, "locality_380": lc_a1b, "n_bytes_bf16": pk_a1b.n_bytes_bf16(), "rollback": rb_a1b}
        print(f"  [Air Tier 1B Native Widened 0-SGD] p0={ev_a1b['p0']['acc']:.3f} unseen={ev_a1b['unseen_p1_p2_pooled']['acc']:.3f} | loc_broken={lc_a1b['newly_broken']} | rb=True")

        # Tier 1C: Native Anchor-SwiGLU Unroller (0-SGD, 100% Native Qwen2MLP!)
        best_air_tau = 2.0
        air_tau_mode = "default_fallback_insufficient_dev"
        if len(dev_airports) >= 2:
            air_tau_mode = "dev_tuned"
            donor_air_dev_blind_st_1c = {li: {k: v[dev_blind_idx] for k, v in donor_air_dev_st[li].items()} for li in compile_layers}
            best_air_dev_1c = -1.0
            for cand_tau in (1.0, 2.0, 4.0):
                pk_c, _ = compile_tier1c_native_anchor_swiglu_unroller(
                    model_B, tok_B, dev_B, compile_layers, bridges, donor_air_dev_blind_st_1c,
                    [format_airport_prompt(ap["name"], 0)[0] for ap in dev_airports], cal_corpus["fit"][:40],
                    n_anchors=min(args.add_mlp_rows, len(dev_airports)), tau=cand_tau, positions="last",
                    lam_bg=5.0, lam_eye=0.05, batch_size=16, max_length=64, seed=SEED + 25,
                )
                pk_c.apply(model_B)
                sc_d = _eval_air_cohort(dev_airports, [])["unseen_p1_p2_pooled"]["acc"]
                pk_c.revert(model_B)
                pk_c.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
                if sc_d > best_air_dev_1c:
                    best_air_dev_1c, best_air_tau = sc_d, cand_tau
        pk_a1c, d_a1c = compile_tier1c_native_anchor_swiglu_unroller(
            model_B, tok_B, dev_B, compile_layers, bridges, donor_air_eval_st,
            eval_p0_prompts, cal_corpus["fit"][:40], n_anchors=min(args.add_mlp_rows, len(eval_airports)),
            tau=best_air_tau, positions="last", lam_bg=5.0, lam_eye=0.05, batch_size=16, max_length=64, seed=SEED + 26,
        )
        pk_a1c.apply(model_B)
        ev_a1c = _eval_air_cohort(eval_airports, routes_eval)
        lc_a1c, _ = eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)
        pk_a1c.revert(model_B)
        rb_a1c = pk_a1c.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
        air_tiers["Tier1C_NATIVE_ANCHOR_SWIGLU_0SGD"] = {**ev_a1c, "tau": best_air_tau, "tau_selection_mode": air_tau_mode, "locality_380": lc_a1c, "n_bytes_bf16": pk_a1c.n_bytes_bf16(), "compile_sec": d_a1c["compile_sec"], "rollback": rb_a1c}
        print(f"  [Air Tier 1C Native Anchor-SwiGLU] p0={ev_a1c['p0']['acc']:.3f} unseen={ev_a1c['unseen_p1_p2_pooled']['acc']:.3f} (tau={best_air_tau}, {air_tau_mode}) | loc_broken={lc_a1c['newly_broken']} | rb=True")

        best_air_lr_t2, best_air_dev_t2 = 3e-4, -1.0
        if len(dev_airports) >= 4:
            donor_air_dev_blind_st = {li: {k: v[dev_blind_idx] for k, v in donor_air_dev_st[li].items()} for li in compile_layers}
            pk_dev_1b, _ = compile_tier1b_native_widened_heads_and_swiglu(
                model_B, tok_B, dev_B, compile_layers, bridges, donor_air_weights, donor_air_dev_blind_st,
                [format_airport_prompt(ap["name"], 0)[0] for ap in dev_airports], cal_corpus["fit"][:40],
                positions="last", lam_bg=5.0, lam_eye=0.05, batch_size=16, max_length=64,
            )
            for cand_lr in (1e-4, 3e-4):
                pk_c, _ = polish_widened_native_pack_sft(model_B, tok_B, dev_B, pk_dev_1b, dev_air_pairs, v_shared=V_shared, lr=cand_lr, steps=25, batch_size=8, seed=SEED + 20)
                pk_c.apply(model_B)
                sc_d = _eval_air_cohort(dev_airports, [])["unseen_p1_p2_pooled"]["acc"]
                pk_c.revert(model_B)
                pk_c.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
                if sc_d > best_air_dev_t2:
                    best_air_dev_t2, best_air_lr_t2 = sc_d, cand_lr

        AIR_STEPS = 40
        for tname, init_pk in (
            ("Tier2_NATIVE_INIT_TINY_POLISH", pk_a1b),
            ("Tier2_CTRL_RECIPIENT_ONLY_NATIVE", make_random_zero_control_pack(pk_a1b, seed=SEED + 21)),
        ):
            pk_t2, info_t2 = polish_widened_native_pack_sft(
                model_B, tok_B, dev_B, init_pk, eval_air_pairs, v_shared=V_shared, lr=best_air_lr_t2, steps=AIR_STEPS, batch_size=8, seed=SEED + 22,
            )
            pk_t2.apply(model_B)
            ev_t2 = _eval_air_cohort(eval_airports, routes_eval)
            lc_t2, _ = eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)
            pk_t2.revert(model_B)
            rb_t2 = pk_t2.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
            air_tiers[tname] = {**ev_t2, "locality_380": lc_t2, "n_bytes_bf16": pk_t2.n_bytes_bf16(), "train_info": info_t2, "rollback": rb_t2}
            print(f"  [Air {tname:32s}] p0={ev_t2['p0']['acc']:.3f} unseen={ev_t2['unseen_p1_p2_pooled']['acc']:.3f} | loss={info_t2['loss_start']}->{info_t2['loss_end']} | loc_broken={lc_t2['newly_broken']} | rb=True")

        best_air_lr_t3, best_air_dev_t3 = 3e-4, -1.0
        if len(dev_airports) >= 4:
            for cand_lr in (1e-4, 3e-4):
                pk_c, _ = train_tier3_teacher_lora_merged_native(model_B, tok_B, dev_B, compile_layers, mode="sft", prompt_target_pairs=dev_air_pairs, v_shared=V_shared, lr=cand_lr, steps=25, batch_size=8, seed=SEED + 23)
                pk_c.apply(model_B)
                sc_d = _eval_air_cohort(dev_airports, [])["unseen_p1_p2_pooled"]["acc"]
                pk_c.revert(model_B)
                pk_c.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
                if sc_d > best_air_dev_t3:
                    best_air_dev_t3, best_air_lr_t3 = sc_d, cand_lr

        pk_a3, info_a3 = train_tier3_teacher_lora_merged_native(
            model_B, tok_B, dev_B, compile_layers, mode="sft", prompt_target_pairs=eval_air_pairs,
            v_shared=V_shared, rank=16, alpha=32.0, lr=best_air_lr_t3, steps=AIR_STEPS, batch_size=8, seed=SEED + 24,
        )
        pk_a3.apply(model_B)
        ev_a3 = _eval_air_cohort(eval_airports, routes_eval)
        lc_a3, _ = eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)
        pk_a3.revert(model_B)
        rb_a3 = pk_a3.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
        air_tiers["Tier3_TEACHER_LORA_MERGED_NATIVE"] = {
            **ev_a3, "locality_380": lc_a3,
            "raw_lora_factor_bytes_bf16": info_a3["raw_lora_factor_bytes_bf16"],
            "merged_dense_delta_bytes_bf16": info_a3["merged_dense_delta_bytes_bf16"],
            "train_info": info_a3, "rollback": rb_a3,
        }
        print(f"  [Air Tier 3 Merged Teacher-LoRA   ] p0={ev_a3['p0']['acc']:.3f} unseen={ev_a3['unseen_p1_p2_pooled']['acc']:.3f} | lr={best_air_lr_t3} | loss={info_a3['loss_start']}->{info_a3['loss_end']} | loc_broken={lc_a3['newly_broken']} | rb=True")

    del donor_air_dev_st, donor_air_eval_pool_st, donor_air_weights
    gc.collect()

    results["task_b_ourairports_enrichment"] = {
        "source": air_src, "n_screened": len(airports), "n_7b_p0_ok": len(airports_7b_p0),
        "n_dev_blind": len(dev_airports), "n_eval_blind": len(eval_airports),
        "gate1a_l27_exact_3letter": g1a_out, "tiers": air_tiers,
    }

    # =========================================================================
    # TASK C: COUNTERFACTUAL 2-ROW 2-HOP POLICY SWAP (ZERO Labels, Strict Rollback Both Policies!)
    # =========================================================================
    print("\n================ TASK C: COUNTERFACTUAL 2-ROW 2-HOP POLICY SWAP (ZERO Labels, ZERO Hooks) ================")
    tr_2h_prompts = [x["recipient_prompt"] for x in twohop_ds["train"]]
    dv_2h_prompts = [x["recipient_prompt"] for x in twohop_ds["dev"]]
    te_2h_prompts = [x["recipient_prompt"] for x in twohop_ds["test"]]
    base_2h_lg = _last_tok_logits(model_B, tok_B, te_2h_prompts, dev_B, bs=16, v_cap=V_shared)
    b2_L = _eval_2hop_logits(base_2h_lg, twohop_ds["test"], tok_B, "L")
    b2_R = _eval_2hop_logits(base_2h_lg, twohop_ds["test"], tok_B, "R")
    print(f"  [2-Hop Base 1.5B] L_em={b2_L['em']:.3f}, R_em={b2_R['em']:.3f}, L_fc4={b2_L['fc4']:.3f}, R_fc4={b2_R['fc4']:.3f}")

    twohop_tiers: Dict[str, Any] = {}
    pol_st_map = {"L": d2_tr_L_st, "R": d2_tr_R_st}
    pol_lg_map = {"L": d2_tr_L_logits[:, :V_shared], "R": d2_tr_R_logits[:, :V_shared]}

    # Tier 1A: Sequential Native ΔW (0-SGD)
    t1a_res, t1a_loc, t1a_rb, t1a_bytes = {}, {}, {}, 0
    for pol in ("L", "R"):
        pk, _ = compile_tier1a_sequential_native_delta_w(
            model_B, tok_B, dev_B, compile_layers, bridges, pol_st_map[pol],
            tr_2h_prompts, cal_corpus["fit"][:40], positions="last", lam_bg=5.0, lam_eye=0.05, batch_size=16, max_length=128,
        )
        t1a_bytes = pk.n_bytes_bf16()
        pk.apply(model_B)
        t1a_res[pol] = _eval_2hop_logits(_last_tok_logits(model_B, tok_B, te_2h_prompts, dev_B, bs=16, v_cap=V_shared), twohop_ds["test"], tok_B, pol)
        t1a_loc[pol], _ = eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)
        pk.revert(model_B)
        t1a_rb[pol] = pk.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
    s_1a = _joint_2hop(t1a_res["L"], t1a_res["R"])
    twohop_tiers["Tier1A_NATIVE_SEQ_DELTA_W_0SGD"] = {**s_1a, "n_bytes_bf16": t1a_bytes, "locality_380": t1a_loc, "rollback": t1a_rb}
    print(f"  [2-Hop Tier 1A Native Seq ΔW 0-SGD] joint_em={s_1a['joint_swap_em']:.3f} | joint_fc4={s_1a['joint_swap_fc4']:.3f} | loc=({t1a_loc['L']['newly_broken']},{t1a_loc['R']['newly_broken']}) | rb=True")

    # Tier 1B: Native Widened GQA Heads + SwiGLU Pullback (0-SGD)
    t1b_packs: Dict[str, ReversibleNativeOrganPack] = {}
    t1b_res, t1b_loc, t1b_rb = {}, {}, {}
    for pol in ("L", "R"):
        pk, _ = compile_tier1b_native_widened_heads_and_swiglu(
            model_B, tok_B, dev_B, compile_layers, bridges, donor_2hop_weights[pol], pol_st_map[pol],
            tr_2h_prompts, cal_corpus["fit"][:40], positions="last", lam_bg=5.0, lam_eye=0.05, batch_size=16, max_length=128,
        )
        t1b_packs[pol] = pk
        pk.apply(model_B)
        t1b_res[pol] = _eval_2hop_logits(_last_tok_logits(model_B, tok_B, te_2h_prompts, dev_B, bs=16, v_cap=V_shared), twohop_ds["test"], tok_B, pol)
        t1b_loc[pol], _ = eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)
        pk.revert(model_B)
        t1b_rb[pol] = pk.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
    s_1b = _joint_2hop(t1b_res["L"], t1b_res["R"])
    twohop_tiers["Tier1B_NATIVE_WIDENED_GQA_SWIGLU_0SGD"] = {**s_1b, "n_bytes_bf16": t1b_packs["L"].n_bytes_bf16(), "locality_380": t1b_loc, "rollback": t1b_rb}
    print(f"  [2-Hop Tier 1B Native Widened 0-SGD] joint_em={s_1b['joint_swap_em']:.3f} | joint_fc4={s_1b['joint_swap_fc4']:.3f} | loc=({t1b_loc['L']['newly_broken']},{t1b_loc['R']['newly_broken']}) | rb=True")

    # Tier 1C: Native Anchor-SwiGLU Unroller (0-SGD, 100% Native Qwen2MLP, dev-tuned tau in {1, 2, 4}!)
    best_2h_tau, best_2h_dev_1c = 2.0, -1.0
    for cand_tau in (1.0, 2.0, 4.0):
        pk_c, _ = compile_tier1c_native_anchor_swiglu_unroller(
            model_B, tok_B, dev_B, compile_layers, bridges, pol_st_map["R"],
            tr_2h_prompts, cal_corpus["fit"][:40], n_anchors=min(args.add_mlp_rows, len(tr_2h_prompts)),
            tau=cand_tau, positions="last", lam_bg=5.0, lam_eye=0.05, batch_size=16, max_length=128, seed=SEED + 28,
        )
        pk_c.apply(model_B)
        sc_d = _eval_2hop_logits(_last_tok_logits(model_B, tok_B, dv_2h_prompts, dev_B, bs=16, v_cap=V_shared), twohop_ds["dev"], tok_B, "R")["fc4"]
        pk_c.revert(model_B)
        pk_c.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
        if sc_d > best_2h_dev_1c:
            best_2h_dev_1c, best_2h_tau = sc_d, cand_tau

    t1c_res, t1c_loc, t1c_rb, t1c_bytes = {}, {}, {}, 0
    for pol in ("L", "R"):
        pk, _ = compile_tier1c_native_anchor_swiglu_unroller(
            model_B, tok_B, dev_B, compile_layers, bridges, pol_st_map[pol],
            tr_2h_prompts, cal_corpus["fit"][:40], n_anchors=min(args.add_mlp_rows, len(tr_2h_prompts)),
            tau=best_2h_tau, positions="last", lam_bg=5.0, lam_eye=0.05, batch_size=16, max_length=128, seed=SEED + 29,
        )
        t1c_bytes = pk.n_bytes_bf16()
        pk.apply(model_B)
        t1c_res[pol] = _eval_2hop_logits(_last_tok_logits(model_B, tok_B, te_2h_prompts, dev_B, bs=16, v_cap=V_shared), twohop_ds["test"], tok_B, pol)
        t1c_loc[pol], _ = eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)
        pk.revert(model_B)
        t1c_rb[pol] = pk.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
    s_1c = _joint_2hop(t1c_res["L"], t1c_res["R"])
    twohop_tiers["Tier1C_NATIVE_ANCHOR_SWIGLU_0SGD"] = {**s_1c, "tau": best_2h_tau, "n_bytes_bf16": t1c_bytes, "locality_380": t1c_loc, "rollback": t1c_rb}
    print(f"  [2-Hop Tier 1C Native Anchor-SwiGLU] joint_em={s_1c['joint_swap_em']:.3f} | joint_fc4={s_1c['joint_swap_fc4']:.3f} (tau={best_2h_tau}) | loc=({t1c_loc['L']['newly_broken']},{t1c_loc['R']['newly_broken']}) | rb=True")

    # Dev LR selection on `twohop_ds["dev"]` (`lr ∈ {1e-4, 3e-4}`), EQUAL 80 steps across T2 / T2-Ctrl / T2-Shuf / T3!
    best_2h_lr_t2, best_2h_dev_t2 = 3e-4, -1.0
    for cand_lr in (1e-4, 3e-4):
        pk_c, _ = polish_widened_native_pack_teacher_kl(model_B, tok_B, dev_B, t1b_packs["R"], tr_2h_prompts, pol_lg_map["R"], V_shared, lr=cand_lr, steps=40, batch_size=8, seed=SEED + 30, strict_loss_decrease=False)
        pk_c.apply(model_B)
        sc_d = _eval_2hop_logits(_last_tok_logits(model_B, tok_B, dv_2h_prompts, dev_B, bs=16, v_cap=V_shared), twohop_ds["dev"], tok_B, "R")["em"]
        pk_c.revert(model_B)
        pk_c.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
        if sc_d > best_2h_dev_t2:
            best_2h_dev_t2, best_2h_lr_t2 = sc_d, cand_lr

    TWOHOP_STEPS = 80
    for tname, use_init, shuf in (
        ("Tier2_NATIVE_INIT_TINY_POLISH", True, False),
        ("Tier2_CTRL_RECIPIENT_ONLY_NATIVE", False, False),
        ("Tier2_CTRL_SHUFFLED_SUPERVISION", False, True),
    ):
        ev_p, loc_p, tr_p, rb_p = {}, {}, {}, {}
        n_b = 0
        for pol in ("L", "R"):
            init_pk = t1b_packs[pol] if use_init else make_random_zero_control_pack(t1b_packs[pol], seed=SEED + (31 if pol == "L" else 32))
            t_lg = pol_lg_map[pol]
            if shuf:
                rng_s = np.random.default_rng(SEED + (41 if pol == "L" else 42))
                t_lg = t_lg[rng_s.permutation(len(t_lg))]
            pk2, info2 = polish_widened_native_pack_teacher_kl(
                model_B, tok_B, dev_B, init_pk, tr_2h_prompts, t_lg, V_shared,
                lr=best_2h_lr_t2, steps=TWOHOP_STEPS, batch_size=8, seed=SEED + 50,
                strict_loss_decrease=(not shuf),
            )
            n_b = pk2.n_bytes_bf16()
            tr_p[pol] = info2
            pk2.apply(model_B)
            ev_p[pol] = _eval_2hop_logits(_last_tok_logits(model_B, tok_B, te_2h_prompts, dev_B, bs=16, v_cap=V_shared), twohop_ds["test"], tok_B, pol)
            loc_p[pol], _ = eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)
            pk2.revert(model_B)
            rb_p[pol] = pk2.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
        sm2 = _joint_2hop(ev_p["L"], ev_p["R"])
        twohop_tiers[tname] = {**sm2, "n_bytes_bf16": n_b, "locality_380": loc_p, "train_info": tr_p, "rollback": rb_p}
        print(f"  [2-Hop {tname:32s}] joint_em={sm2['joint_swap_em']:.3f} (L={sm2['left_em']:.3f}, R={sm2['right_em']:.3f}) | joint_fc4={sm2['joint_swap_fc4']:.3f} | loc=({loc_p['L']['newly_broken']},{loc_p['R']['newly_broken']}) | rb=True")

    best_2h_lr_t3, best_2h_dev_t3 = 3e-4, -1.0
    for cand_lr in (1e-4, 3e-4):
        pk_c, _ = train_tier3_teacher_lora_merged_native(model_B, tok_B, dev_B, compile_layers, mode="teacher_kl", prompts_kl=tr_2h_prompts, teacher_logits_shared=pol_lg_map["R"], v_shared=V_shared, lr=cand_lr, steps=40, batch_size=8, seed=SEED + 58)
        pk_c.apply(model_B)
        sc_d = _eval_2hop_logits(_last_tok_logits(model_B, tok_B, dv_2h_prompts, dev_B, bs=16, v_cap=V_shared), twohop_ds["dev"], tok_B, "R")["em"]
        pk_c.revert(model_B)
        pk_c.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
        if sc_d > best_2h_dev_t3:
            best_2h_dev_t3, best_2h_lr_t3 = sc_d, cand_lr

    t3_res, t3_loc, t3_tr, t3_rb, t3_raw_b, t3_dense_b = {}, {}, {}, {}, 0, 0
    for pol in ("L", "R"):
        pk3, info3 = train_tier3_teacher_lora_merged_native(
            model_B, tok_B, dev_B, compile_layers, mode="teacher_kl",
            prompts_kl=tr_2h_prompts, teacher_logits_shared=pol_lg_map[pol], v_shared=V_shared,
            rank=16, alpha=32.0, lr=best_2h_lr_t3, steps=TWOHOP_STEPS, batch_size=8, seed=SEED + 60,
        )
        t3_tr[pol] = info3
        t3_raw_b = info3["raw_lora_factor_bytes_bf16"]
        t3_dense_b = info3["merged_dense_delta_bytes_bf16"]
        pk3.apply(model_B)
        t3_res[pol] = _eval_2hop_logits(_last_tok_logits(model_B, tok_B, te_2h_prompts, dev_B, bs=16, v_cap=V_shared), twohop_ds["test"], tok_B, pol)
        t3_loc[pol], _ = eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)
        pk3.revert(model_B)
        t3_rb[pol] = pk3.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
    sm3 = _joint_2hop(t3_res["L"], t3_res["R"])
    twohop_tiers["Tier3_TEACHER_LORA_MERGED_NATIVE"] = {
        **sm3, "raw_lora_factor_bytes_bf16": t3_raw_b, "merged_dense_delta_bytes_bf16": t3_dense_b,
        "locality_380": t3_loc, "train_info": t3_tr, "rollback": t3_rb,
    }
    print(f"  [2-Hop Tier 3 Merged Teacher-LoRA   ] joint_em={sm3['joint_swap_em']:.3f} (L={sm3['left_em']:.3f}, R={sm3['right_em']:.3f}) | joint_fc4={sm3['joint_swap_fc4']:.3f} | loc=({t3_loc['L']['newly_broken']},{t3_loc['R']['newly_broken']}) | rb=True")

    results["task_c_2row_2hop_policy_swap"] = {
        "n_train": len(twohop_ds["train"]), "n_dev": len(twohop_ds["dev"]), "n_test_sealed": len(twohop_ds["test"]),
        "donor_7b_test": donor_2hop_sum,
        "base_1p5b": {"left_em": b2_L["em"], "right_em": b2_R["em"], "left_fc": b2_L["fc"], "right_fc": b2_R["fc"], "left_fc4": b2_L["fc4"], "right_fc4": b2_R["fc4"]},
        "tiers": twohop_tiers,
    }
    results["meta"]["total_elapsed_sec"] = round(time.time() - t_all, 1)

    unload_model_vram_safe(model_B)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[V4.1 STAGE 1 COMPLETE in {results['meta']['total_elapsed_sec']}s] Saved {OUT_PATH}")
    return results


# ======================================================================================
# 5. Pure-CPU Unit Tests (<3s, ZERO GPU!) — BF16 Convergence + Zero Hooks + Rollback!
# ======================================================================================

def run_cpu_unit_tests() -> None:
    t0 = time.time()
    from transformers import AutoTokenizer
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    print("=== Running v4_stage1_qwen7b_to_1p5b_native.py Pure-CPU Unit Tests (v4.1) ===")

    tok = AutoTokenizer.from_pretrained(str(HF_CACHE_DIR / "qwen2.5-3b"), local_files_only=True)
    ds = build_2row_2hop_dataset([tok], n_train=192, n_dev=64, n_test=256, seed=SEED)
    assert len(ds["train"]) == 192 and len(ds["dev"]) == 64 and len(ds["test"]) == 256
    assert set(ds["train_keys"]).isdisjoint(set(ds["test_keys"]))
    assert all(len({it["val_L"], it["val_R"], it["dist_L"], it["dist_R"]}) == 4 for it in ds["test"][:16])
    print("  [PASS 1/3] 2-row 2-hop dataset verified (4 distinct 1-token leaves per table, disjoint train/test keys)")

    lf = t1d.build_certified_locality_facts()
    assert len(lf) == 380
    assert check_answer_match(" " + lf[0]["aliases"][0] + ".", lf[0]["aliases"], lf[0].get("relation", ""))
    assert not check_route_match("The man can fly", "MAN", "CAN")["both"]
    assert check_route_match("JFK to LAX", "JFK", "LAX")["ordered"]
    print("  [PASS 2/3] Certified 380-fact locality + word-boundary ordered check_route_match verified")

    # Verify BF16 model + FP32 master-weight native polish at realistic lr=3e-4 with nonzero init (|w|=0.08)!
    torch.manual_seed(11)
    cfg = Qwen2Config(
        vocab_size=max(tok.vocab_size, len(tok)) + 32, hidden_size=96, intermediate_size=192, num_hidden_layers=3,
        num_attention_heads=12, num_key_value_heads=2, rope_theta=1_000_000.0,
    )
    toy = Qwen2ForCausalLM(cfg).to(dtype=torch.bfloat16).eval()
    base_sd = {k: v.detach().clone() for k, v in toy.state_dict().items()}
    enc = tok(["Table:\noak -> pine\nOutput:", "Table:\nrain -> snow\nOutput:"], return_tensors="pt", padding=True)
    with torch.no_grad():
        base_lg = toy(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False).logits.clone()

    d_h = 8
    init_pk = ReversibleNativeOrganPack(
        attn_widen={1: {
            "q_w": torch.full((6 * d_h, 96), 0.08), "q_b": torch.zeros(6 * d_h),
            "k_w": torch.full((1 * d_h, 96), 0.08), "k_b": torch.zeros(1 * d_h),
            "v_w": torch.full((1 * d_h, 96), 0.08), "v_b": torch.zeros(1 * d_h),
            "o_w": torch.full((96, 6 * d_h), 0.08), "add_q_heads": 6, "add_kv_heads": 1,
        }},
        mlp_widen={1: {
            "gate_w": torch.full((16, 96), 0.08), "up_w": torch.full((16, 96), 0.08), "down_w": torch.full((96, 16), 0.08),
        }},
    )
    pol_pk, pinfo = polish_widened_native_pack_sft(
        toy, tok, torch.device("cpu"), init_pk,
        [("Table:\noak -> pine\nOutput:", " red"), ("Table:\nrain -> snow\nOutput:", " blue")],
        v_shared=len(tok), lr=3e-4, steps=6, batch_size=2, seed=SEED, strict_loss_decrease=True,
    )
    assert pinfo["loss_decreased"], f"BF16 FP32-master polish stalled: {pinfo}"
    assert_zero_runtime_hooks(toy)
    assert all(torch.equal(v, base_sd[k]) for k, v in toy.state_dict().items())
    pol_pk.apply(toy)
    assert_zero_runtime_hooks(toy)
    pol_pk.revert(toy)
    rb = pol_pk.verify_bit_exact_rollback(toy, enc["input_ids"], enc["attention_mask"], base_lg, strict_assert=True)
    print(f"  [PASS 3/4] BF16 + FP32-master native polish at lr=3e-4 (|w|=0.08): loss {pinfo['loss_start']}->{pinfo['loss_end']}, ZERO hooks, torch.equal rollback==True!")

    # 4. Verify Tier 1C Native Anchor-SwiGLU Unroller end-to-end on BF16 toy Qwen2 (ZERO hooks, torch.equal rollback!)
    rng_u = np.random.default_rng(17)
    fake_bridges = {1: {"M_out": rng_u.standard_normal((96, 64)) * 0.1}}
    fake_donor = {1: {"h_out": torch.randn(2, 64, dtype=torch.bfloat16).view(torch.uint16).numpy()}}
    pk_1c, _ = compile_tier1c_native_anchor_swiglu_unroller(
        toy, tok, torch.device("cpu"), [1], fake_bridges, fake_donor,
        ["Table:\noak -> pine\nOutput:", "Table:\nrain -> snow\nOutput:"],
        ["Sample background passage one.", "Sample background passage two."],
        n_anchors=2, tau=2.0, positions="last", batch_size=2, max_length=32, seed=SEED,
    )
    pk_1c.apply(toy)
    assert_zero_runtime_hooks(toy)
    assert toy.model.layers[1].mlp.gate_proj.out_features == 192 + 2
    pk_1c.revert(toy)
    pk_1c.verify_bit_exact_rollback(toy, enc["input_ids"], enc["attention_mask"], base_lg, strict_assert=True)
    print(f"  [PASS 4/4] Tier 1C Native Anchor-SwiGLU Unroller verified end-to-end on BF16 Qwen2 (ZERO hooks, torch.equal==True)!")
    print(f"=== ALL 4 V4.2 STAGE 1 CPU UNIT TESTS PASSED ({time.time() - t0:.2f}s)! ===")


def main() -> None:
    ap = argparse.ArgumentParser(description="v4.1 Stage 1: Pure Native Weight Translation (Qwen2.5-7B -> Qwen2.5-1.5B)")
    ap.add_argument("--unit-test", action="store_true")
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--donor-model", type=str, default=DONOR_KEY_DEFAULT)
    ap.add_argument("--recipient-model", type=str, default=RECIPIENT_KEY_DEFAULT)
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--n-fit-passages", type=int, default=165)
    ap.add_argument("--n-val-passages", type=int, default=45)
    ap.add_argument("--cal-fit-cap", type=int, default=12000)
    ap.add_argument("--cal-val-cap", type=int, default=2500)
    ap.add_argument("--max-airports", type=int, default=1000)
    ap.add_argument("--n-2hop-train", type=int, default=192)
    ap.add_argument("--n-2hop-test", type=int, default=256)
    ap.add_argument("--add-mlp-rows", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    if args.unit_test:
        run_cpu_unit_tests()
        return
    if args.run_all:
        run_stage1_all(args)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
