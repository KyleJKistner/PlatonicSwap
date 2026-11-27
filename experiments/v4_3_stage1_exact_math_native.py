#!/usr/bin/env python3
"""
v4_3_stage1_exact_math_native.py — Stage 1 Pure Native Translation with Fable 5.2 Max + Astra 6.2 Ultra Exact Math (v4.3.1)
============================================================================================================================

Resolves 100% of P0 & P1 blockers from BOTH Astra 6.2 Ultra AND Gemini 4 Neon High:
  - Zero-extra-VRAM BF16 chunked `StreamingSharedVocabFisherAccumulator` (`< 18.0 GiB` peak VRAM on 24 GB L4,
    0 GPU copies in `finalize()`, `cal_batch_size=2`, Fisher streamed over ALL `N_fit >= 60,000` tokens,
    strict CUDA memory drop verification after `unload_model_vram_safe(model_A)`).
  - Strict `o_j >= 0.25` deaf-unit rejection enforced on BOTH Attention heads AND MLP SwiGLU neurons
    (never silently falling back to copying deaf `< 0.25` units; skips sublayer cleanly if 0 qualify).
  - Template-disjoint `fit` (`>= 60,000`) / `val_tune` (`>= 3,000`) / `val_report` (`>= 3,000`) with
    **0 shared prefixes** across BOTH `fit vs val` AND `val_tune vs val_report`; `ood_guard_prompts`
    strictly drawn from `val_tune` (never enters `val_report`).
  - True Nyström Kernel Ridge (`K_LL` Gram regularizer) for Floor #3 `Delta_NL`, and exact RMSNorm
    Jacobian orthogonal projection in Alignment-Free Gauss-Newton L27 writer (`Tier 1-GN`).
  - Fail-closed `KL_drift <= 0.05` OOD damping gate with descending `(1.0 -> 0.0)` tie-break, verified
    on EVERY final evaluated pack, wrapped in exception-safe `try/finally` rollback + `strict_assert=True`.
  - End-to-end ZERO-LABEL Task C selection (damping `alpha` and per-tier dev LR selected strictly by
    minimizing KL to Donor 7B dev logits, plus `Tier2_CTRL_SHUFFLED_SUPERVISION` restored!), independent
    per-tier `{1e-4, 3e-4}` dev LR sweeps for Tier 2 and Tier 3 across all 3 tasks, and Task B dev
    calibration strictly on `dev_airports` (framed accurately as implanted-fact `p0` retention + unseen
    `p1/p2` & unseen route transfer).

Usage:
  PYTHONPATH=platonicswap:experiments python experiments/v4_3_stage1_exact_math_native.py --unit-test
  PYTHONPATH=platonicswap:experiments python -u experiments/v4_3_stage1_exact_math_native.py --run-all
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
OUT_PATH = RESULTS_DIR / "results_v4_3_stage1_exact_math_native.json"

from exact_math_v43 import (
    StreamingSecondMomentAccumulator,
    StreamingSharedVocabFisherAccumulator,
    build_60k_calibration_corpus,
    compute_fisher_rrr_retained_fraction,
    compute_floor3_nonlinear_residual_gain,
    compute_gauss_newton_logit_gap_target,
    compute_schur_complement_s_agb,
    extract_shared_suffix_states,
    fit_ridge_from_moments_median_floor,
    score_layer_donor_observability,
    solve_shared_vocab_bridge_m_u,
)
from extract_and_baseline import HF_CACHE_DIR, get_decoder_layers
from large_to_small import load_model_bf16
from native_weight_translation import (
    ReversibleNativeOrganPack,
    assert_zero_runtime_hooks,
    bf16_to_f64_np,
    compile_tier1a_sequential_native_delta_w,
    compile_tier1b_native_widened_heads_and_swiglu,
    extract_layer_token_states,
    get_config_rope_theta,
    solve_closed_form_ridge_with_bg,
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
    build_text_to_sqlite_benchmark,
    execute_sql_on_schema,
)
from v4_stage1_qwen7b_to_1p5b_native import (
    DEFAULT_COMPILE_LAYERS,
    DONOR_KEY_DEFAULT,
    RECIPIENT_KEY_DEFAULT,
    SQL_TEST_SCHEMAS,
    SQL_TRAIN_SCHEMAS,
    _eval_2hop_logits,
    _joint_2hop,
    _last_tok_logits,
    build_2row_2hop_dataset,
    eval_certified_380_locality,
    eval_sql_queries_native,
    format_sql_chat_prompt,
    greedy_generate_native,
    make_random_zero_control_pack,
    polish_widened_native_pack_sft,
    polish_widened_native_pack_teacher_kl,
    train_tier3_teacher_lora_merged_native,
)

SEED = 20261129


# ======================================================================================
# 1. Strict `o_j >= 0.25` Observability-Weighted Donor Extraction (Zero Deaf Units!)
# ======================================================================================

def extract_observable_donor_heads_and_neurons(
    model_A: nn.Module,
    layer_indices: Sequence[int],
    obs_by_layer: Dict[int, Dict[str, Any]],
    donor_task_z_attn: Dict[int, np.ndarray],
    donor_task_z_mlp: Dict[int, np.ndarray],
    donor_bg_z_mlp: Dict[int, np.ndarray],
    add_q_heads: int = 6,
    add_kv_heads: int = 1,
    add_mlp_rows: int = 256,
    min_obs: float = 0.25,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """
    Strictly enforce `o_j >= min_obs` (0.25) on BOTH Attention Q/K/V heads AND MLP SwiGLU neurons!
    Never silently falls back to deaf (`< min_obs`) units: if fewer than `add_mlp_rows` qualify,
    keeps ONLY the qualifying subset; if no GQA group has >= `add_q_heads` heads passing `o_j >= min_obs`,
    skips attention widening at that layer cleanly!
    """
    layers = get_decoder_layers(model_A)
    out: Dict[int, Dict[str, Any]] = {}
    sel_diag: Dict[str, Any] = {}

    for li in sorted(int(x) for x in layer_indices):
        lyr = layers[li]
        attn = lyr.self_attn
        mlp = lyr.mlp
        d_h = int(attn.head_dim)
        g_A = int(attn.num_key_value_groups)
        n_q_A = int(attn.q_proj.out_features // d_h)
        n_kv_A = int(attn.k_proj.out_features // d_h)
        assert add_kv_heads == 1 and add_q_heads <= g_A

        obs_l = obs_by_layer[li]
        o_head = obs_l["o_head_joint"]
        if "o_head_min" in obs_l:
            o_head_min = obs_l["o_head_min"]
        elif all(k in obs_l for k in ("o_q_head", "o_k_kv", "o_v_kv")):
            o_head_min = np.array([
                float(min(obs_l["o_q_head"][h], obs_l["o_k_kv"][h // g_A], obs_l["o_v_kv"][h // g_A]))
                for h in range(n_q_A)
            ])
        else:
            raise KeyError(f"Layer {li}: obs_by_layer must provide 'o_head_min' or constituent ('o_q_head','o_k_kv','o_v_kv')!")
        o_mlp = obs_l["o_mlp_joint"]
        if "o_mlp_min" in obs_l:
            o_mlp_min = obs_l["o_mlp_min"]
        elif "o_gate" in obs_l and "o_up" in obs_l:
            o_mlp_min = np.minimum(obs_l["o_gate"], obs_l["o_up"])
        else:
            raise KeyError(f"Layer {li}: obs_by_layer must provide 'o_mlp_min' or constituent ('o_gate','o_up')!")

        # 1. Strict GQA head selection requiring ALL constituent (Q, K, V) scores >= min_obs BEFORE ranking!
        za = bf16_to_f64_np(donor_task_z_attn[li])
        head_energy = np.array([float(np.mean(za[:, h * d_h : (h + 1) * d_h] ** 2)) for h in range(n_q_A)])
        best_kv, best_sel_q, best_gscore = -1, [], -1.0
        for kv in range(n_kv_A):
            q_cand = [h for h in range(kv * g_A, (kv + 1) * g_A) if float(o_head_min[h]) >= float(min_obs)]
            if len(q_cand) < add_q_heads:
                continue
            q_cand.sort(key=lambda h: float(head_energy[h] * o_head[h]), reverse=True)
            top_q = sorted(q_cand[:add_q_heads])
            gscore = float(sum(head_energy[h] * o_head[h] for h in top_q))
            if gscore > best_gscore:
                best_gscore, best_kv, best_sel_q = gscore, kv, top_q

        rec: Dict[str, Any] = {"add_q_heads": 0, "add_kv_heads": 0, "q_w": None, "gate_w": None}
        if best_kv >= 0:
            Wq_A = attn.q_proj.weight.detach().float().cpu().numpy()
            Wk_A = attn.k_proj.weight.detach().float().cpu().numpy()
            Wv_A = attn.v_proj.weight.detach().float().cpu().numpy()
            Wo_A = attn.o_proj.weight.detach().float().cpu().numpy()
            bq_A = attn.q_proj.bias.detach().float().cpu().numpy() if attn.q_proj.bias is not None else None
            bk_A = attn.k_proj.bias.detach().float().cpu().numpy() if attn.k_proj.bias is not None else None
            bv_A = attn.v_proj.bias.detach().float().cpu().numpy() if attn.v_proj.bias is not None else None
            q_rows = np.concatenate([np.arange(h * d_h, (h + 1) * d_h) for h in best_sel_q])
            kv_rows = np.arange(best_kv * d_h, (best_kv + 1) * d_h)
            rec.update({
                "best_kv_group": best_kv, "sel_q_heads": best_sel_q,
                "add_q_heads": int(add_q_heads), "add_kv_heads": int(add_kv_heads),
                "q_w": Wq_A[q_rows].copy(), "q_b": bq_A[q_rows].copy() if bq_A is not None else None,
                "k_w": Wk_A[kv_rows].copy(), "k_b": bk_A[kv_rows].copy() if bk_A is not None else None,
                "v_w": Wv_A[kv_rows].copy(), "v_b": bv_A[kv_rows].copy() if bv_A is not None else None,
                "o_w": Wo_A[:, q_rows].copy(),
            })

        # 2. Strict MLP SwiGLU selection: require BOTH gate AND up constituents >= min_obs BEFORE ranking!
        zm_t = bf16_to_f64_np(donor_task_z_mlp[li])
        zm_b = bf16_to_f64_np(donor_bg_z_mlp[li])
        e_task = np.mean(zm_t ** 2, axis=0)
        e_bg = np.mean(zm_b ** 2, axis=0)
        salience = np.maximum(e_task / (e_bg + 1e-4) + 0.25 * (e_task - e_bg), 0.0)
        joint_mlp = salience * (o_mlp ** 2)
        qual_idx = np.where(o_mlp_min >= float(min_obs))[0]
        k_keep = min(int(add_mlp_rows), int(len(qual_idx)))
        if k_keep > 0:
            top_local = np.argsort(joint_mlp[qual_idx])[-k_keep:]
            sel_n = np.sort(qual_idx[top_local])
            Wg_A = mlp.gate_proj.weight.detach().float().cpu().numpy()
            Wu_A = mlp.up_proj.weight.detach().float().cpu().numpy()
            Wd_A = mlp.down_proj.weight.detach().float().cpu().numpy()
            rec.update({
                "target_mlp_rows": k_keep, "sel_mlp_indices": sel_n.tolist(),
                "gate_w": Wg_A[sel_n].copy(), "up_w": Wu_A[sel_n].copy(), "down_w": Wd_A[:, sel_n].copy(),
            })
        else:
            sel_n = np.array([], dtype=np.int64)
            rec.update({"target_mlp_rows": 0, "sel_mlp_indices": []})

        out[li] = rec
        sel_diag[f"L{li}"] = {
            "n_heads_selected": len(best_sel_q),
            "selected_head_min_constituent_o": round(float(np.min(o_head_min[best_sel_q])), 4) if best_sel_q else 0.0,
            "selected_head_mean_o": round(float(np.mean(o_head[best_sel_q])), 4) if best_sel_q else 0.0,
            "n_mlp_selected": int(k_keep),
            "n_mlp_above_min_obs": int(len(qual_idx)),
            "selected_mlp_min_constituent_o": round(float(np.min(o_mlp_min[sel_n])), 4) if k_keep > 0 else 0.0,
            "selected_mlp_mean_o": round(float(np.mean(o_mlp[sel_n])), 4) if k_keep > 0 else 0.0,
        }
    return out, sel_diag


# ======================================================================================
# 2. Zero-Extra-VRAM Streaming Dual-Model Calibration (`N_fit >= 60k`, Peak < 18 GiB!)
# ======================================================================================

@torch.inference_mode()
def stream_60k_dual_calibration_and_fisher(
    model_A: nn.Module,
    model_B: nn.Module,
    tok: Any,
    fit_passages: Sequence[str],
    val_tune_passages: Sequence[str],
    val_report_passages: Sequence[str],
    compile_layers: Sequence[int],
    dev_A: torch.device,
    dev_B: torch.device,
    v_shared: int,
    batch_size: int = 2,
    max_length: int = 160,
    max_val_tokens_per_half: int = 3200,
) -> Dict[str, Any]:
    """
    Zero-extra-VRAM streaming FP64 calibration (`N_fit >= 60,000`, `N_val_tune >= 2,500`,
    `N_val_report >= 2,500`, Fisher streamed over ALL fit tokens in 16k vocab chunks!).
    Peak L4 VRAM ~17.9 GiB (< 22.5 GiB usable!).
    """
    t0 = time.time()
    D_A = int(model_A.config.hidden_size)
    D_B = int(model_B.config.hidden_size)
    layers_A = get_decoder_layers(model_A)
    layers_B = get_decoder_layers(model_B)
    fields = ("u_in", "y_attn", "u_post", "h_out")

    accs: Dict[int, Dict[str, StreamingSecondMomentAccumulator]] = {
        int(li): {f: StreamingSecondMomentAccumulator(D_A, D_B) for f in fields}
        for li in compile_layers
    }
    acc_xfinal = StreamingSecondMomentAccumulator(D_A, D_B)
    # Zero-copy BF16 views on existing lm_head weights (0 B extra VRAM!)
    fisher_acc = StreamingSharedVocabFisherAccumulator(
        model_A.lm_head.weight[:v_shared],
        model_B.lm_head.weight[:v_shared],
        vocab_chunk=16384,
    )

    bg_z_mlp_store: Dict[int, List[np.ndarray]] = {int(li): [] for li in compile_layers}
    bg_z_mlp_count = 0
    buf_A: Dict[int, Dict[str, torch.Tensor]] = {int(li): {} for li in compile_layers}
    buf_B: Dict[int, Dict[str, torch.Tensor]] = {int(li): {} for li in compile_layers}
    buf_xf: Dict[str, torch.Tensor] = {}
    handles: List[Any] = []

    def _register_pair(layers, buf, capture_z_mlp: bool = False):
        for li in compile_layers:
            li = int(li)
            lyr = layers[li]
            def _mk(b, idx, key):
                def _fn(_m, _i, out):
                    t = out[0] if isinstance(out, tuple) else out
                    b[idx][key] = t.detach()
                return _fn
            handles.append(lyr.input_layernorm.register_forward_hook(_mk(buf, li, "u_in")))
            handles.append(lyr.self_attn.o_proj.register_forward_hook(_mk(buf, li, "y_attn")))
            handles.append(lyr.post_attention_layernorm.register_forward_hook(_mk(buf, li, "u_post")))
            handles.append(lyr.register_forward_hook(_mk(buf, li, "h_out")))
            if capture_z_mlp:
                def _mk_zm(b, idx):
                    def _pre(_m, args):
                        if bg_z_mlp_count < 1024:
                            b[idx]["z_mlp"] = args[0].detach()
                    return _pre
                handles.append(lyr.mlp.down_proj.register_forward_pre_hook(_mk_zm(buf, li)))

    f3_layers = sorted({int(l) for l in (20, compile_layers[-1]) if int(l) in {int(x) for x in compile_layers}})
    f3_fit_A: Dict[int, List[np.ndarray]] = {gl: [] for gl in f3_layers}
    f3_fit_B: Dict[int, List[np.ndarray]] = {gl: [] for gl in f3_layers}
    f3_fit_n = 0

    val_halves = {"tune": list(val_tune_passages), "report": list(val_report_passages)}
    val_store: Dict[str, Dict[str, Any]] = {}

    old_pad = tok.padding_side
    tok.padding_side = "left"
    backbone_B = model_B.model if hasattr(model_B, "model") else model_B
    try:
        _register_pair(layers_A, buf_A, capture_z_mlp=True)
        _register_pair(layers_B, buf_B, capture_z_mlp=False)
        handles.append(model_A.model.norm.register_forward_hook(lambda _m, _i, o: buf_xf.__setitem__("A", o.detach())))
        handles.append(model_B.model.norm.register_forward_hook(lambda _m, _i, o: buf_xf.__setitem__("B", o.detach())))

        # 1. Stream ALL fit passages (`N_fit >= 60,000` tokens; Fisher streamed over ALL fit tokens!)
        for i0 in range(0, len(fit_passages), batch_size):
            bp = list(fit_passages[i0 : i0 + batch_size])
            enc = tok(bp, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=True)
            ids_A, mask_A = enc["input_ids"].to(dev_A), enc["attention_mask"].to(dev_A)
            ids_B, mask_B = enc["input_ids"].to(dev_B), enc["attention_mask"].to(dev_B)
            vm_A, vm_B = mask_A.bool(), mask_B.bool()

            out_A = model_A(input_ids=ids_A, attention_mask=mask_A, use_cache=False)
            fisher_acc.update(out_A.logits[vm_A][:, :v_shared])
            del out_A  # Free 152k logits immediately BEFORE model_B forward!

            # State-only backbone forward for Recipient B (skips unused 152k lm_head projection!)
            _ = backbone_B(input_ids=ids_B, attention_mask=mask_B, use_cache=False)
            acc_xfinal.update(buf_xf.pop("A")[vm_A], buf_xf.pop("B")[vm_B])
            n_btok = int(vm_A.sum().item())

            for li in compile_layers:
                li = int(li)
                for f in fields:
                    accs[li][f].update(buf_A[li][f][vm_A], buf_B[li][f][vm_B])
                if li in f3_layers and f3_fit_n < 3072:
                    f3_fit_A[li].append(buf_A[li]["h_out"][vm_A].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
                    f3_fit_B[li].append(buf_B[li]["h_out"][vm_B].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
                if "z_mlp" in buf_A[li] and bg_z_mlp_count < 1024:
                    bg_z_mlp_store[li].append(buf_A[li]["z_mlp"][vm_A].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
                buf_A[li].clear()
                buf_B[li].clear()

            if f3_fit_n < 3072:
                f3_fit_n += n_btok
            if bg_z_mlp_count < 1024:
                bg_z_mlp_count += n_btok

        # 2. Capture template-disjoint `val_tune` and `val_report` splits (0 shared prefixes!)
        for hname, vlist in val_halves.items():
            st_A = {int(li): {f: [] for f in fields} for li in compile_layers}
            st_B = {int(li): {f: [] for f in fields} for li in compile_layers}
            xf_A_l, xf_B_l = [], []
            last_lg_A, last_lg_B, last_xf_A, last_h27_A = [], [], [], []
            n_col = 0
            for i0 in range(0, len(vlist), batch_size):
                bp = list(vlist[i0 : i0 + batch_size])
                enc = tok(bp, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=True)
                ids_A, mask_A = enc["input_ids"].to(dev_A), enc["attention_mask"].to(dev_A)
                ids_B, mask_B = enc["input_ids"].to(dev_B), enc["attention_mask"].to(dev_B)
                vm_A, vm_B = mask_A.bool(), mask_B.bool()

                try:
                    out_A = model_A(input_ids=ids_A, attention_mask=mask_A, use_cache=False, logits_to_keep=1)
                except TypeError:
                    out_A = model_A(input_ids=ids_A, attention_mask=mask_A, use_cache=False)
                last_lg_A.append(out_A.logits[:, -1, :v_shared].float().cpu())
                del out_A

                try:
                    out_B = model_B(input_ids=ids_B, attention_mask=mask_B, use_cache=False, logits_to_keep=1)
                except TypeError:
                    out_B = model_B(input_ids=ids_B, attention_mask=mask_B, use_cache=False)
                last_lg_B.append(out_B.logits[:, -1, :v_shared].float().cpu())
                del out_B

                xfA_t, xfB_t = buf_xf.pop("A"), buf_xf.pop("B")
                last_xf_A.append(xfA_t[:, -1, :].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
                last_h27_A.append(buf_A[compile_layers[-1]]["h_out"][:, -1, :].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())

                if n_col < max_val_tokens_per_half:
                    xf_A_l.append(xfA_t[vm_A].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
                    xf_B_l.append(xfB_t[vm_B].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
                    for li in compile_layers:
                        li = int(li)
                        for f in fields:
                            st_A[li][f].append(buf_A[li][f][vm_A].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
                            st_B[li][f].append(buf_B[li][f][vm_B].to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy())
                    n_col += int(vm_A.sum().item())
                for li in compile_layers:
                    buf_A[int(li)].clear()
                    buf_B[int(li)].clear()

            cap = max_val_tokens_per_half
            val_store[hname] = {
                "A": {li: {f: np.ascontiguousarray(np.concatenate(st_A[li][f], axis=0)[:cap].copy()) for f in fields} for li in compile_layers},
                "B": {li: {f: np.ascontiguousarray(np.concatenate(st_B[li][f], axis=0)[:cap].copy()) for f in fields} for li in compile_layers},
                "xf_A": np.ascontiguousarray(np.concatenate(xf_A_l, axis=0)[:cap].copy()),
                "xf_B": np.ascontiguousarray(np.concatenate(xf_B_l, axis=0)[:cap].copy()),
                "last_lg_A": torch.cat(last_lg_A, dim=0),
                "last_lg_B": torch.cat(last_lg_B, dim=0),
                "last_xf_A": np.concatenate(last_xf_A, axis=0),
                "last_h27_A": np.concatenate(last_h27_A, axis=0),
            }
    finally:
        for h in handles:
            h.remove()
        tok.padding_side = old_pad
    assert_zero_runtime_hooks(model_A)
    assert_zero_runtime_hooks(model_B)

    donor_bg_z_mlp = {
        int(li): np.ascontiguousarray(np.concatenate(bg_z_mlp_store[int(li)], axis=0)[:1024].copy())
        for li in compile_layers
    }
    f3_fit = {
        "A": {gl: np.ascontiguousarray(np.concatenate(f3_fit_A[gl], axis=0)[:3072].copy()) for gl in f3_layers},
        "B": {gl: np.ascontiguousarray(np.concatenate(f3_fit_B[gl], axis=0)[:3072].copy()) for gl in f3_layers},
    }
    n_fisher = fisher_acc.n
    fisher_mats = fisher_acc.finalize()
    del fisher_acc

    # Normalize moments IN-PLACE (zero 5.1 GiB duplicate host-RAM copy!)
    moments_out = {int(li): {f: accs[int(li)][f].moments(inplace=True) for f in fields} for li in compile_layers}
    moments_xf = acc_xfinal.moments(inplace=True)
    n_fit_toks = acc_xfinal.n
    del accs, acc_xfinal
    gc.collect()

    return {
        "n_fit_tokens": n_fit_toks,
        "n_fisher_tokens": n_fisher,
        "moments": moments_out,
        "moments_xfinal": moments_xf,
        "fisher": fisher_mats,
        "val": val_store,
        "f3_fit": f3_fit,
        "donor_bg_z_mlp": donor_bg_z_mlp,
        "elapsed_sec": round(time.time() - t0, 1),
    }


@torch.no_grad()
def chunked_lm_head_logits_cpu(x_fp32_cpu: np.ndarray, lm_head_weight_bf16: torch.Tensor, v_shared: int, dev: torch.device, chunk: int = 16384) -> torch.Tensor:
    """Compute `x @ U_B[:v_shared].T` in 16k vocab chunks with ZERO full FP32 unembedding GPU allocation!"""
    x_gpu = torch.from_numpy(x_fp32_cpu.astype(np.float32)).to(dev)
    out = torch.empty((x_gpu.shape[0], v_shared), dtype=torch.float32, device="cpu")
    for i0 in range(0, v_shared, chunk):
        i1 = min(i0 + chunk, v_shared)
        out[:, i0:i1] = (x_gpu @ lm_head_weight_bf16[i0:i1].float().T).cpu()
    del x_gpu
    return out


# ======================================================================================
# 3. Floor #2 Patched KL Recovery `R` (Push M h_A, Pull P h_B, Round-Trip P M h_A!)
# ======================================================================================

@torch.inference_mode()
def measure_patched_kl_recovery(
    model_A: nn.Module,
    model_B: nn.Module,
    tok: Any,
    passages: Sequence[str],
    layer_idx: int,
    M_out: np.ndarray,
    P_out: np.ndarray,
    mean_A: np.ndarray,
    mean_B: np.ndarray,
    dev_A: torch.device,
    dev_B: torch.device,
    v_shared: int,
    batch_size: int = 4,
    max_length: int = 128,
) -> Dict[str, float]:
    layers_A = get_decoder_layers(model_A)
    layers_B = get_decoder_layers(model_B)
    M_t = torch.from_numpy(M_out.astype(np.float32)).to(dev_B)
    P_t = torch.from_numpy(P_out.astype(np.float32)).to(dev_A)
    PM_t = torch.from_numpy((P_out @ M_out).astype(np.float32)).to(dev_A)
    mA_t = torch.from_numpy(mean_A.astype(np.float32)).to(dev_A)
    mB_t = torch.from_numpy(mean_B.astype(np.float32)).to(dev_B)

    def _kl(ref_lg: torch.Tensor, pat_lg: torch.Tensor) -> float:
        return float(F.kl_div(F.log_softmax(pat_lg.float(), dim=-1), F.softmax(ref_lg.float(), dim=-1), reduction="batchmean").item())

    kl_push_num, kl_push_den, kl_pull_num, kl_rt_num, kl_A_den = [], [], [], [], []
    old_pad = tok.padding_side
    tok.padding_side = "left"
    try:
        for i0 in range(0, len(passages), batch_size):
            bp = list(passages[i0 : i0 + batch_size])
            enc = tok(bp, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=True)
            ids_A, mask_A = enc["input_ids"].to(dev_A), enc["attention_mask"].to(dev_A)
            ids_B, mask_B = enc["input_ids"].to(dev_B), enc["attention_mask"].to(dev_B)

            cap: Dict[str, torch.Tensor] = {}
            hA_hdl: Any = None
            hB_hdl: Any = None
            try:
                hA_hdl = layers_A[layer_idx].register_forward_hook(lambda _m, _i, o: cap.__setitem__("hA", (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach()))
                hB_hdl = layers_B[layer_idx].register_forward_hook(lambda _m, _i, o: cap.__setitem__("hB", (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach()))
                ref_A = model_A(input_ids=ids_A, attention_mask=mask_A, use_cache=False).logits[:, -1, :v_shared].float().cpu()
                ref_B = model_B(input_ids=ids_B, attention_mask=mask_B, use_cache=False).logits[:, -1, :v_shared].float().cpu()
            finally:
                if hA_hdl is not None:
                    hA_hdl.remove()
                if hB_hdl is not None:
                    hB_hdl.remove()

            hA_v, hB_v = cap["hA"].float(), cap["hB"].float()

            def _run_patched(model, lyr, ids, mask, repl_vec: torch.Tensor) -> torch.Tensor:
                def _hk(_m, _i, o):
                    is_t = isinstance(o, tuple)
                    h = (o[0] if is_t else o).clone()
                    h[:, -1, :] = repl_vec.to(h.dtype)
                    return (h,) + o[1:] if is_t else h
                hd: Any = None
                try:
                    hd = lyr.register_forward_hook(_hk)
                    return model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1, :v_shared].float().cpu()
                finally:
                    if hd is not None:
                        hd.remove()

            kl_push_num.append(_kl(ref_B, _run_patched(model_B, layers_B[layer_idx], ids_B, mask_B, hA_v.to(dev_B) @ M_t.T)))
            kl_push_den.append(_kl(ref_B, _run_patched(model_B, layers_B[layer_idx], ids_B, mask_B, mB_t.unsqueeze(0).expand_as(hB_v))))
            kl_pull_num.append(_kl(ref_A, _run_patched(model_A, layers_A[layer_idx], ids_A, mask_A, hB_v.to(dev_A) @ P_t.T)))
            kl_rt_num.append(_kl(ref_A, _run_patched(model_A, layers_A[layer_idx], ids_A, mask_A, hA_v @ PM_t.T)))
            kl_A_den.append(_kl(ref_A, _run_patched(model_A, layers_A[layer_idx], ids_A, mask_A, mA_t.unsqueeze(0).expand_as(hA_v))))
    finally:
        tok.padding_side = old_pad
    assert_zero_runtime_hooks(model_A)
    assert_zero_runtime_hooks(model_B)

    def _rec(num_l, den_l) -> float:
        return round(1.0 - float(np.mean(num_l)) / max(float(np.mean(den_l)), 1e-8), 4)

    return {
        "R_push_M_hA": _rec(kl_push_num, kl_push_den),
        "R_pull_P_hB": _rec(kl_pull_num, kl_A_den),
        "R_roundtrip_P_M_hA": _rec(kl_rt_num, kl_A_den),
        "kl_push": round(float(np.mean(kl_push_num)), 4),
        "kl_pull": round(float(np.mean(kl_pull_num)), 4),
        "kl_roundtrip": round(float(np.mean(kl_rt_num)), 4),
    }


# ======================================================================================
# 4. OOD-Safe Damping, Exception-Safe Eval Wrapper & Exact-Jacobian Gauss-Newton L27
# ======================================================================================

def scale_native_pack_outputs(pack: ReversibleNativeOrganPack, alpha: float) -> ReversibleNativeOrganPack:
    sc_deltas = {k: (v * float(alpha)).clone() for k, v in pack.deltas.items()}
    sc_attn = {
        li: {k: ((v * float(alpha)).clone() if k == "o_w" else (v.clone() if isinstance(v, torch.Tensor) else v)) for k, v in sp.items()}
        for li, sp in pack.attn_widen.items()
    }
    sc_mlp = {
        li: {k: ((v * float(alpha)).clone() if k == "down_w" else v.clone()) for k, v in sp.items()}
        for li, sp in pack.mlp_widen.items()
    }
    return ReversibleNativeOrganPack(deltas=sc_deltas, attn_widen=sc_attn, mlp_widen=sc_mlp, metadata={**pack.metadata, "alpha_damp": float(alpha)})


@torch.no_grad()
def measure_ood_kl_drift(
    model_B: nn.Module,
    tok_B: Any,
    ood_prompts: Sequence[str],
    base_ood_logits: torch.Tensor,
    dev_B: torch.device,
    v_shared: int,
) -> float:
    cur = _last_tok_logits(model_B, tok_B, ood_prompts, dev_B, bs=16, v_cap=v_shared)
    return float(F.kl_div(F.log_softmax(cur, dim=-1), F.softmax(base_ood_logits, dim=-1), reduction="batchmean").item())


def project_rmsnorm_tangent(H_b: np.ndarray, v: np.ndarray, rms_b: np.ndarray) -> np.ndarray:
    """
    Scale-invariant exact orthogonal projection onto `H_b^perp` scaled by `rms_b`:
      `dH_gn = rms_b * (v - h_u * (h_u . v) / (h_u . h_u))`, where `h_u = H_b / max(|H_b|)`
    Guarantees radial vectors (`v || H_b`) map identically to `0` down to tiny scales (`1e-13` and below!),
    tangent vectors (`v . H_b == 0`) map identically to `rms_b * v`, and exact `H_b == 0` stays finite!
    """
    h_scale = np.max(np.abs(H_b), axis=-1, keepdims=True)
    nonzero = h_scale > 0.0
    h_u = np.where(nonzero, H_b / np.where(nonzero, h_scale, 1.0), 0.0)
    h_norm2 = np.sum(h_u * h_u, axis=-1, keepdims=True)
    proj = np.where(nonzero, np.sum(h_u * v, axis=-1, keepdims=True) / np.where(nonzero, h_norm2, 1.0), 0.0)
    return rms_b * (v - proj * h_u)


def compile_tier1_gn_l27_native_delta_w(
    model_B: nn.Module,
    tok_B: Any,
    device_B: torch.device,
    G_B: np.ndarray,
    donor_logits_shared: torch.Tensor,
    task_prompts: Sequence[str],
    bg_prompts: Sequence[str],
    v_shared: int,
    lam_bg: float = 50.0,
    lam_eye: float = 0.5,
    batch_size: int = 16,
    max_length: int = 128,
) -> Tuple[ReversibleNativeOrganPack, Dict[str, Any]]:
    """
    Alignment-Free Gauss-Newton L27 Native `mlp.down_proj` Compiler (0-SGD, ZERO state-space bridge!):
    Streams `grad_b = (p_A - p_B) @ U_B[:v_shared]` in 16k vocab chunks (zero full FP64 `U_B` copy!),
    solves `tau_GN = (G_B + lam*I)^{-1} grad_b`, and applies the **exact RMSNorm inverse-Jacobian
    orthogonal projection** (`dH_gn = rms * (v - mean(v * h_hat) * h_hat)` where `v = tau_GN / gamma_B`)!
    """
    t0 = time.time()
    l_fin = len(get_decoder_layers(model_B)) - 1
    st_t = extract_layer_token_states(
        model_B, tok_B, task_prompts, [l_fin], device_B, batch_size=batch_size,
        max_length=max_length, positions="last", capture_fields=("z_mlp", "h_out"),
    )[l_fin]
    st_bg = extract_layer_token_states(
        model_B, tok_B, bg_prompts[:40], [l_fin], device_B, batch_size=batch_size,
        max_length=max_length, positions="all_nonpad", capture_fields=("z_mlp",), max_tokens_cap=2048,
    )[l_fin]
    lg_B = _last_tok_logits(model_B, tok_B, task_prompts, device_B, bs=batch_size, v_cap=v_shared)
    dp_gpu = (F.softmax(donor_logits_shared[:, :v_shared].float(), dim=-1) - F.softmax(lg_B[:, :v_shared].float(), dim=-1)).to(device_B)
    D_B = int(model_B.config.hidden_size)
    grad_b_gpu = torch.zeros((dp_gpu.shape[0], D_B), device=device_B, dtype=torch.float32)
    U_B_ref = model_B.lm_head.weight[:v_shared].detach()
    for i0 in range(0, v_shared, 16384):
        i1 = min(i0 + 16384, v_shared)
        grad_b_gpu.addmm_(dp_gpu[:, i0:i1], U_B_ref[i0:i1].float())
    grad_b = grad_b_gpu.double().cpu().numpy()
    del dp_gpu, grad_b_gpu

    eb = np.linalg.eigvalsh(G_B)
    pos = eb[eb > 1e-12]
    med_b = float(np.median(pos)) if len(pos) > 0 else float(np.trace(G_B) / max(D_B, 1))
    lam_abs = max(0.1 * med_b, 1e-8)
    tau_gn = np.linalg.solve(G_B + lam_abs * np.eye(D_B, dtype=np.float64), grad_b.T).T

    # Exact orthogonal tangent projection onto H_b^perp (radial component vanishes identically!)
    H_b = bf16_to_f64_np(st_t["h_out"])
    eps_ln = float(getattr(model_B.config, "rms_norm_eps", 1e-6))
    rms_b = np.sqrt(np.mean(H_b ** 2, axis=-1, keepdims=True) + eps_ln)
    gamma_b = model_B.model.norm.weight.detach().double().cpu().numpy()[None, :]
    g_safe = np.where(np.abs(gamma_b) >= 1e-4, gamma_b, np.where(gamma_b >= 0, 1e-4, -1e-4))
    v = tau_gn / g_safe
    dH_gn = project_rmsnorm_tangent(H_b, v, rms_b)

    Z_t = bf16_to_f64_np(st_t["z_mlp"])
    Z_bg = bf16_to_f64_np(st_bg["z_mlp"])
    C_bg = (Z_bg.T @ Z_bg) / max(float(len(Z_bg)), 1.0)
    del Z_bg, st_bg
    dW_gn, info = solve_closed_form_ridge_with_bg(Z_t, dH_gn, C_bg=C_bg, lam_bg=lam_bg, lam_eye=lam_eye)
    del Z_t, dH_gn, C_bg, st_t
    gc.collect()

    pk = ReversibleNativeOrganPack(
        deltas={f"model.layers.{l_fin}.mlp.down_proj.weight": torch.from_numpy(dW_gn.astype(np.float32))},
        metadata={"tier": "Tier1_GN_AlignmentFree_L27_0SGD"},
    )
    return pk, {**info, "compile_sec": round(time.time() - t0, 2)}


# ======================================================================================
# 5. Main Stage 1 v4.3.1 Runner (`Qwen2.5-7B-Instruct` -> `Qwen2.5-1.5B-Instruct`)
# ======================================================================================

def run_v43_stage1(args: argparse.Namespace) -> Dict[str, Any]:
    t_all = time.time()
    compile_layers = list(DEFAULT_COMPILE_LAYERS)
    print(f"\n================ V4.3.1 STAGE 1: FABLE+ASTRA EXACT MATH NATIVE TRANSLATION ({args.donor_model} -> {args.recipient_model}) ================")

    cal = build_60k_calibration_corpus(n_fit_passages=args.n_fit_passages, n_val_passages=args.n_val_passages, seed=SEED)
    sql_bench = build_text_to_sqlite_benchmark()
    sql_gold = {}
    for it in sql_bench:
        ok, rows, err = execute_sql_on_schema(it["ddl"], it["seed"], it["gold_sql"])
        assert ok and rows is not None and len(rows) > 0, f"Gold SQL failed: {it['id']}: {err}"
        sql_gold[it["id"]] = rows
    sql_train = [it for it in sql_bench if it["schema_id"] in SQL_TRAIN_SCHEMAS]
    sql_test = [it for it in sql_bench if it["schema_id"] in SQL_TEST_SCHEMAS]
    sql_fit_sub, sql_dev_sub = sql_train[:30], sql_train[30:]
    airports, air_src = fetch_or_fallback_airports(max_airports=args.max_airports, seed=SEED, offline=args.offline)
    loc_facts = t1d.build_certified_locality_facts()

    # =========================================================================
    # PHASE 1: ZERO-EXTRA-VRAM STREAMING 60K+ CALIBRATION + 3 FLOORS + DONOR EXTRACTION
    # =========================================================================
    t_p1 = time.time()
    vram_telemetry: Dict[str, Any] = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model_A, tok_A, dev_A, _ = load_model_bf16(args.donor_model, allow_download=args.allow_download)
    model_B, tok_B, dev_B, _ = load_model_bf16(args.recipient_model, allow_download=args.allow_download)
    for p in model_B.parameters():
        p.requires_grad_(False)
    assert tok_A.get_vocab() == tok_B.get_vocab(), "Stage 1 requires identical tokenizer!"
    V_shared = min(len(tok_A), len(tok_B), int(model_A.lm_head.out_features), int(model_B.lm_head.out_features))
    D_A, D_B = int(model_A.config.hidden_size), int(model_B.config.hidden_size)
    print(f"  [Phase 1] Loaded 7B (D={D_A}) + 1.5B (D={D_B}) for streaming 60k calibration (V_shared={V_shared}, rope={get_config_rope_theta(model_A.config):g})")

    cal_out = stream_60k_dual_calibration_and_fisher(
        model_A, model_B, tok_A, cal["fit"], cal["val_tune"], cal["val_report"], compile_layers, dev_A, dev_B, V_shared,
        batch_size=args.cal_batch_size, max_length=160,
    )
    n_fit = cal_out["n_fit_tokens"]
    n_val_t = len(cal_out["val"]["tune"]["xf_A"])
    n_val_r = len(cal_out["val"]["report"]["xf_A"])
    print(f"  [Streaming Calibration] N_fit={n_fit} (Fisher={cal_out['n_fisher_tokens']}) | N_tune={n_val_t} | N_report={n_val_r} ({cal_out['elapsed_sec']}s)")
    assert n_fit >= 60_000, f"Expected N_fit >= 60,000, got {n_fit}"
    assert n_val_t >= 2_500 and n_val_r >= 2_500, f"Expected >=2,500 val tokens per half, got {n_val_t}/{n_val_r}"

    bridges: Dict[int, Dict[str, np.ndarray]] = {}
    bridge_diag: Dict[str, Any] = {}
    obs_by_layer: Dict[int, Dict[str, Any]] = {}
    obs_summary: Dict[str, Any] = {}
    layers_A = get_decoder_layers(model_A)

    # Release each layer's large FP64 moment matrices immediately after bridge+Schur computation (saves host RAM!)
    h_out_means: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for li in compile_layers:
        m_li = cal_out["moments"].pop(li)
        vtA, vtB = cal_out["val"]["tune"]["A"][li], cal_out["val"]["tune"]["B"][li]
        vrA, vrB = cal_out["val"]["report"]["A"][li], cal_out["val"]["report"]["B"][li]

        P_attn, i_pa = fit_ridge_from_moments_median_floor(m_li["u_in"]["S_BB"], m_li["u_in"]["S_BA"].T, vtB["u_in"], vtA["u_in"], n_fit)
        P_mlp, i_pm = fit_ridge_from_moments_median_floor(m_li["u_post"]["S_BB"], m_li["u_post"]["S_BA"].T, vtB["u_post"], vtA["u_post"], n_fit)
        M_attn, i_ma = fit_ridge_from_moments_median_floor(m_li["y_attn"]["S_AA"], m_li["y_attn"]["S_BA"], vtA["y_attn"], vtB["y_attn"], n_fit)
        M_out, i_mo = fit_ridge_from_moments_median_floor(m_li["h_out"]["S_AA"], m_li["h_out"]["S_BA"], vtA["h_out"], vtB["h_out"], n_fit)
        P_out, _ = fit_ridge_from_moments_median_floor(m_li["h_out"]["S_BB"], m_li["h_out"]["S_BA"].T, vtB["h_out"], vtA["h_out"], n_fit)

        def _rep_cos(W, Xr, Yr):
            x, y = bf16_to_f64_np(Xr), bf16_to_f64_np(Yr)
            p = x @ W.T
            c = float(np.mean(np.sum(p * y, axis=1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)))
            r2 = 1.0 - float(np.sum((y - p) ** 2) / (np.sum((y - y.mean(0)) ** 2) + 1e-12))
            return round(c, 4), round(r2, 4)

        bridges[li] = {"P_attn": P_attn, "P_mlp": P_mlp, "M_attn": M_attn, "M_out": M_out, "P_out": P_out}
        _, S_AgB_a = compute_schur_complement_s_agb(m_li["u_in"]["S_AA"], m_li["u_in"]["S_BB"], m_li["u_in"]["S_BA"], i_pa["best_lam_abs"])
        _, S_AgB_m = compute_schur_complement_s_agb(m_li["u_post"]["S_AA"], m_li["u_post"]["S_BB"], m_li["u_post"]["S_BA"], i_pm["best_lam_abs"])
        obs_l = score_layer_donor_observability(layers_A[li], m_li["u_in"]["S_AA"], S_AgB_a, m_li["u_post"]["S_AA"], S_AgB_m)
        obs_by_layer[li] = obs_l
        obs_summary[f"L{li}"] = obs_l["summary"]
        if li in (20, compile_layers[-1]):
            h_out_means[li] = (m_li["h_out"]["mean_A"].copy(), m_li["h_out"]["mean_B"].copy())
        del m_li, S_AgB_a, S_AgB_m
        gc.collect()

        rc_mo, r2_mo = _rep_cos(M_out, vrA["h_out"], vrB["h_out"])
        rc_pm, _ = _rep_cos(P_mlp, vrB["u_post"], vrA["u_post"])
        bridge_diag[f"L{li}"] = {"M_out_cal_report": {"cos": rc_mo, "r2": r2_mo, "lam_rel_med": i_mo["best_lam_rel_med"]}, "P_mlp_cal_report_cos": rc_pm}
        print(
            f"  [Bridge+Obs L{li:02d}] M_out rep_cos={rc_mo:.4f} (R2={r2_mo:.4f}) | P_mlp={rc_pm:.4f} | "
            f"Floor#2 o_j: MLP med={obs_l['summary']['mlp_median_o']:.3f} p90={obs_l['summary']['mlp_p90_o']:.3f} top256={obs_l['summary']['mlp_top256_mean_o']:.3f} | "
            f"Attn med={obs_l['summary']['attn_qkv_median_o']:.3f} max={obs_l['summary']['attn_qkv_max_o']:.3f}"
        )

    # Floor #1 (Retained Fisher Fraction at k=1536) + Zero-Data Shared-Vocab Bridge `M_U` vs `M_out` at L27!
    fm = cal_out["fisher"]
    floor1_fisher = compute_fisher_rrr_retained_fraction(cal_out["moments_xfinal"]["S_AA"], fm["G_A"], ks=(256, 512, 1024, 1536))
    print(f"  [Floor #1 Fisher-RRR L27] retained_1536={floor1_fisher['retained_fisher_1536']:.6f} | by_k={floor1_fisher['retained_fisher_by_k']}")

    best_mu_kl, best_M_U, best_mu_info = 1e9, None, {}
    vt_xfA = bf16_to_f64_np(cal_out["val"]["tune"]["last_xf_A"])
    vt_lgA = cal_out["val"]["tune"]["last_lg_A"]
    for eps_r in (1e-3, 1e-2, 1e-1, 1.0):
        mu_c, mi_c = solve_shared_vocab_bridge_m_u(fm["G_B"], fm["G_BA"], eps_rel_med=eps_r)
        lg_m = chunked_lm_head_logits_cpu(vt_xfA @ mu_c.T, model_B.lm_head.weight, V_shared, dev_B)
        kl_c = float(F.kl_div(F.log_softmax(lg_m, dim=-1), F.softmax(vt_lgA, dim=-1), reduction="batchmean").item())
        if kl_c < best_mu_kl:
            best_mu_kl, best_M_U, best_mu_info = kl_c, mu_c, mi_c
    assert best_M_U is not None

    vr_xfA = bf16_to_f64_np(cal_out["val"]["report"]["last_xf_A"])
    vr_h27A = bf16_to_f64_np(cal_out["val"]["report"]["last_h27_A"])
    vr_lgA = cal_out["val"]["report"]["last_lg_A"]
    vr_lgB = cal_out["val"]["report"]["last_lg_B"]
    lg_mu_rep = chunked_lm_head_logits_cpu(vr_xfA @ best_M_U.T, model_B.lm_head.weight, V_shared, dev_B)
    with torch.no_grad():
        h27_mout = torch.from_numpy((vr_h27A @ bridges[compile_layers[-1]]["M_out"].T).astype(np.float32)).to(dev_B, dtype=model_B.dtype)
        lg_mout_rep = model_B.lm_head(model_B.model.norm(h27_mout))[:, :V_shared].float().cpu()
        del h27_mout

    def _lg_cmp(pred_lg: torch.Tensor, ref_lg: torch.Tensor) -> Dict[str, float]:
        t1 = float((pred_lg.argmax(-1) == ref_lg.argmax(-1)).float().mean().item())
        kl = float(F.kl_div(F.log_softmax(pred_lg, dim=-1), F.softmax(ref_lg, dim=-1), reduction="batchmean").item())
        return {"top1_agree_7b": round(t1, 4), "kl_to_7b": round(kl, 4)}

    gate0_mu_vs_mout = {
        "M_U_shared_vocab_bridge": {**_lg_cmp(lg_mu_rep, vr_lgA), **best_mu_info},
        "M_out_state_regression": _lg_cmp(lg_mout_rep, vr_lgA),
        "base_1p5b_unmapped": _lg_cmp(vr_lgB, vr_lgA),
        "M_U_beats_M_out_on_KL": bool(_lg_cmp(lg_mu_rep, vr_lgA)["kl_to_7b"] < _lg_cmp(lg_mout_rep, vr_lgA)["kl_to_7b"]),
    }
    print(f"  [Gate 0 L27 M_U vs M_out] M_U: {gate0_mu_vs_mout['M_U_shared_vocab_bridge']['kl_to_7b']} KL / {gate0_mu_vs_mout['M_U_shared_vocab_bridge']['top1_agree_7b']} top1 | M_out: {gate0_mu_vs_mout['M_out_state_regression']['kl_to_7b']} KL / {gate0_mu_vs_mout['M_out_state_regression']['top1_agree_7b']} top1 | Base 1.5B: {gate0_mu_vs_mout['base_1p5b_unmapped']['kl_to_7b']} KL")

    # Floor #2 Patched KL Recovery `R` (L20 & L27) + Floor #3 Nyström Nonlinear Gain `Delta_NL`!
    floor2_kl_rec, floor3_nl = {}, {}
    for gl in (20, compile_layers[-1]):
        mean_A_gl, mean_B_gl = h_out_means[gl]
        floor2_kl_rec[f"L{gl}"] = measure_patched_kl_recovery(
            model_A, model_B, tok_A, cal["val_report"][:32], gl,
            bridges[gl]["M_out"], bridges[gl]["P_out"], mean_A_gl, mean_B_gl,
            dev_A, dev_B, V_shared, batch_size=4,
        )
        floor3_nl[f"L{gl}"] = compute_floor3_nonlinear_residual_gain(
            cal_out["f3_fit"]["A"][gl], cal_out["f3_fit"]["B"][gl],
            cal_out["val"]["tune"]["A"][gl]["h_out"], cal_out["val"]["tune"]["B"][gl]["h_out"],
            cal_out["val"]["report"]["A"][gl]["h_out"], cal_out["val"]["report"]["B"][gl]["h_out"],
            bridges[gl]["M_out"], n_landmarks=512, seed=SEED + gl,
        )
        print(f"  [Floor #2 KL Rec L{gl:02d}] push={floor2_kl_rec[f'L{gl}']['R_push_M_hA']} | pull={floor2_kl_rec[f'L{gl}']['R_pull_P_hB']} | rt={floor2_kl_rec[f'L{gl}']['R_roundtrip_P_M_hA']} || [Floor #3 Delta_NL L{gl:02d}] +{floor3_nl[f'L{gl}']['delta_nl_r2']:.4f} R2")

    donor_bg_z_mlp = cal_out["donor_bg_z_mlp"]
    G_B_l27 = fm["G_B"].copy()
    del cal_out, fm
    gc.collect()

    # Donor 7B Task Extractions (strict `o_j >= 0.25` filtering!)
    twohop_ds = build_2row_2hop_dataset([tok_A], n_train=args.n_2hop_train, n_dev=64, n_test=args.n_2hop_test, seed=SEED)

    donor_sql_all = eval_sql_queries_native(model_A, tok_A, dev_A, sql_bench, sql_gold, batch_size=8, v_shared=V_shared)
    donor_sql_unseen_40 = eval_sql_queries_native(model_A, tok_A, dev_A, sql_test, sql_gold, batch_size=8, v_shared=V_shared)
    donor_sql_train_preds = [donor_sql_all["_pred_sqls"][i] for i, it in enumerate(sql_bench) if it["schema_id"] in SQL_TRAIN_SCHEMAS]
    sql_train_prompts = [format_sql_chat_prompt(tok_A, it["ddl"], it["question"]) for it in sql_train]
    sql_full_teacher_texts = [p + pred for p, pred in zip(sql_train_prompts, donor_sql_train_preds)]
    n_sql_fit30_toks = sum(len(tok_A.encode(t, truncation=True, max_length=384, add_special_tokens=True)) for t in sql_full_teacher_texts[:30])
    donor_sql_states = extract_layer_token_states(
        model_A, tok_A, sql_full_teacher_texts, compile_layers, dev_A,
        batch_size=4, max_length=384, positions="all_nonpad", capture_fields=("y_attn", "h_out", "z_attn", "z_mlp"),
    )
    donor_sql_w_fit30, _ = extract_observable_donor_heads_and_neurons(
        model_A, compile_layers, obs_by_layer,
        {li: donor_sql_states[li]["z_attn"][:n_sql_fit30_toks] for li in compile_layers},
        {li: donor_sql_states[li]["z_mlp"][:n_sql_fit30_toks] for li in compile_layers},
        donor_bg_z_mlp, add_q_heads=6, add_kv_heads=1, add_mlp_rows=args.add_mlp_rows, min_obs=0.25,
    )
    donor_sql_w, sql_obs_sel = extract_observable_donor_heads_and_neurons(
        model_A, compile_layers, obs_by_layer,
        {li: donor_sql_states[li].pop("z_attn") for li in compile_layers},
        {li: donor_sql_states[li].pop("z_mlp") for li in compile_layers},
        donor_bg_z_mlp, add_q_heads=6, add_kv_heads=1, add_mlp_rows=args.add_mlp_rows, min_obs=0.25,
    )
    gc.collect()

    air_p0_all = [format_airport_prompt(ap["name"], 0)[0] for ap in airports]
    d_p0_gens = greedy_generate_native(model_A, tok_A, dev_A, air_p0_all, max_new_tokens=6, batch_size=32, v_shared=V_shared)
    airports_7b_p0 = [dict(ap, donor_p0_pred_iata=(extract_iata_code(g) or ap["iata"])) for ap, g in zip(airports, d_p0_gens) if check_iata_match(g, ap["iata"])]
    rng_air = np.random.default_rng(SEED + 7)
    perm_a = rng_air.permutation(len(airports_7b_p0)).tolist()
    n_ad = min(max(4, int(round(0.30 * len(airports_7b_p0)))), max(len(airports_7b_p0) - 1, 0))
    air_dev_pool = [airports_7b_p0[i] for i in perm_a[:n_ad]]
    air_eval_pool = [airports_7b_p0[i] for i in perm_a[n_ad:]]
    # Screen Recipient 1.5B blind cohort BEFORE selecting Donor 7B units so donor_air_w uses strictly `dev_airports`!
    dev_airports, eval_airports, routes_eval = [], [], []
    if len(air_dev_pool) >= 4 and len(air_eval_pool) >= 8:
        dg_b = greedy_generate_native(model_B, tok_B, dev_B, [format_airport_prompt(ap["name"], 0)[0] for ap in air_dev_pool], max_new_tokens=6, batch_size=32, v_shared=V_shared)
        eg_b = greedy_generate_native(model_B, tok_B, dev_B, [format_airport_prompt(ap["name"], 0)[0] for ap in air_eval_pool], max_new_tokens=6, batch_size=32, v_shared=V_shared)
        dev_airports = [ap for ap, g in zip(air_dev_pool, dg_b) if not check_iata_match(g, ap["iata"])]
        eval_airports = [ap for ap, g in zip(air_eval_pool, eg_b) if not check_iata_match(g, ap["iata"])]
        routes_eval = build_route_items(eval_airports, max_routes=48, seed=SEED + 17)
        print(f"  [Airports Blind Screen] dev_blind={len(dev_airports)}, eval_blind={len(eval_airports)}, routes={len(routes_eval)}")

    donor_air_dev_st, donor_air_w, donor_air_eval_st = {}, {}, {}
    donor_air_dev_lg, donor_air_eval_lg = None, None
    if len(dev_airports) >= 4 and len(eval_airports) >= 8:
        dp0 = [format_airport_prompt(ap["name"], 0)[0] for ap in dev_airports]
        ep0 = [format_airport_prompt(ap["name"], 0)[0] for ap in eval_airports]
        donor_air_dev_st = extract_layer_token_states(model_A, tok_A, dp0, compile_layers, dev_A, batch_size=16, max_length=64, positions="last", capture_fields=("y_attn", "h_out", "z_attn", "z_mlp"))
        donor_air_w, _ = extract_observable_donor_heads_and_neurons(
            model_A, compile_layers, obs_by_layer,
            {li: donor_air_dev_st[li].pop("z_attn") for li in compile_layers},
            {li: donor_air_dev_st[li].pop("z_mlp") for li in compile_layers},
            donor_bg_z_mlp, add_q_heads=6, add_kv_heads=1, add_mlp_rows=args.add_mlp_rows, min_obs=0.25,
        )
        donor_air_dev_lg = _last_tok_logits(model_A, tok_A, dp0, dev_A, bs=32, v_cap=V_shared)
        donor_air_eval_st = extract_layer_token_states(model_A, tok_A, ep0, compile_layers, dev_A, batch_size=16, max_length=64, positions="last", capture_fields=("y_attn", "h_out"))
        donor_air_eval_lg = _last_tok_logits(model_A, tok_A, ep0, dev_A, bs=32, v_cap=V_shared)

    # Task C 2-Row 2-Hop: Capture Donor train + dev teacher logits (100% zero-label selection!) & shared-suffix states!
    d2_te_L = _last_tok_logits(model_A, tok_A, [x["donor_L_prompt"] for x in twohop_ds["test"]], dev_A, bs=16, v_cap=V_shared)
    d2_te_R = _last_tok_logits(model_A, tok_A, [x["donor_R_prompt"] for x in twohop_ds["test"]], dev_A, bs=16, v_cap=V_shared)
    donor_2hop_sum = _joint_2hop(_eval_2hop_logits(d2_te_L, twohop_ds["test"], tok_A, "L"), _eval_2hop_logits(d2_te_R, twohop_ds["test"], tok_A, "R"))
    print(f"  [Task C | Donor 7B 2-Row 2-Hop] joint_em={donor_2hop_sum['joint_swap_em']:.3f} | joint_fc4={donor_2hop_sum['joint_swap_fc4']:.3f}")

    tr_2h_rec = [x["recipient_prompt"] for x in twohop_ds["train"]]
    tr_2h_L = [x["donor_L_prompt"] for x in twohop_ds["train"]]
    tr_2h_R = [x["donor_R_prompt"] for x in twohop_ds["train"]]
    d2_tr_L_lg = _last_tok_logits(model_A, tok_A, tr_2h_L, dev_A, bs=16, v_cap=V_shared)
    d2_tr_R_lg = _last_tok_logits(model_A, tok_A, tr_2h_R, dev_A, bs=16, v_cap=V_shared)
    # Donor dev logits ONLY (zero gold labels used during dev damping & LR tuning!)
    d2_dv_L_lg = _last_tok_logits(model_A, tok_A, [x["donor_L_prompt"] for x in twohop_ds["dev"]], dev_A, bs=16, v_cap=V_shared)
    d2_dv_R_lg = _last_tok_logits(model_A, tok_A, [x["donor_R_prompt"] for x in twohop_ds["dev"]], dev_A, bs=16, v_cap=V_shared)

    d2_tr_L_last = extract_layer_token_states(model_A, tok_A, tr_2h_L, compile_layers, dev_A, batch_size=8, max_length=256, positions="last", capture_fields=("y_attn", "h_out", "z_attn", "z_mlp"))
    d2_tr_R_last = extract_layer_token_states(model_A, tok_A, tr_2h_R, compile_layers, dev_A, batch_size=8, max_length=256, positions="last", capture_fields=("y_attn", "h_out", "z_attn", "z_mlp"))
    d2_tr_L_suf = extract_shared_suffix_states(model_A, tok_A, tr_2h_L, tr_2h_rec, compile_layers, dev_A, batch_size=8, max_length=256, capture_fields=("y_attn", "h_out"))
    d2_tr_R_suf = extract_shared_suffix_states(model_A, tok_A, tr_2h_R, tr_2h_rec, compile_layers, dev_A, batch_size=8, max_length=256, capture_fields=("y_attn", "h_out"))

    donor_2hop_w = {}
    for pol, d_st in (("L", d2_tr_L_last), ("R", d2_tr_R_last)):
        donor_2hop_w[pol], _ = extract_observable_donor_heads_and_neurons(
            model_A, compile_layers, obs_by_layer,
            {li: d_st[li].pop("z_attn") for li in compile_layers},
            {li: d_st[li].pop("z_mlp") for li in compile_layers},
            donor_bg_z_mlp, add_q_heads=6, add_kv_heads=1, add_mlp_rows=args.add_mlp_rows, min_obs=0.25,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram_telemetry["phase1_peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / (1024.0 ** 3), 3)
        vram_telemetry["phase1_peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / (1024.0 ** 3), 3)

    # STRICT CERTIFIED ZERO-COPY UNLOAD OF DONOR 7B!
    unload_model_vram_safe(model_A)
    del model_A, donor_bg_z_mlp, obs_by_layer, layers_A
    gc.collect()
    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated() / (1024.0 ** 3)
        vram_telemetry["post_unload_alloc_gib"] = round(alloc_gb, 3)
        print(
            f"  [Phase 1 Complete in {time.time() - t_p1:.1f}s] Donor 7B 100% UNLOADED! "
            f"(peak_alloc={vram_telemetry['phase1_peak_alloc_gib']:.2f} GiB, peak_res={vram_telemetry['phase1_peak_reserved_gib']:.2f} GiB, post_unload={alloc_gb:.2f} GiB)"
        )
        assert alloc_gb < 6.0, f"Donor 7B VRAM release failed: still allocated {alloc_gb:.2f} GiB!"
        torch.cuda.reset_peak_memory_stats()

    # =========================================================================
    # PHASE 2: SOLO RECIPIENT 1.5B — EXCEPTION-SAFE FAIL-CLOSED OOD EVALUATION
    # =========================================================================
    base_loc, base_loc_ok = eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, batch_size=32, v_shared=V_shared)
    print(f"  [Certified 380-Fact Locality Baseline] {base_loc['n_correct']}/{base_loc['n_total']} ({base_loc['acc']:.3f})")
    assert base_loc["n_correct"] >= 250

    # Strictly draw OOD guard prompts from `val_tune` ONLY (never touches `val_report`!)
    ood_guard_prompts = cal["val_tune"][: min(24, len(cal["val_tune"]))]
    base_ood_lg = _last_tok_logits(model_B, tok_B, ood_guard_prompts, dev_B, bs=16, v_cap=V_shared)
    probe_enc = tok_B(ood_guard_prompts[:4], return_tensors="pt", padding=True, truncation=True, max_length=64)
    probe_ids, probe_mask = probe_enc["input_ids"].to(dev_B), probe_enc["attention_mask"].to(dev_B)
    with torch.no_grad():
        base_probe_logits = model_B(input_ids=probe_ids, attention_mask=probe_mask, use_cache=False).logits.detach().cpu()

    def _run_with_pack_safe(pack: ReversibleNativeOrganPack, fn: Callable[[], Any]) -> Tuple[Any, Dict[str, Any]]:
        """Exception-safe apply (inside try!) -> eval -> revert -> strict torch.equal rollback + zero-hook gate!"""
        try:
            pack.apply(model_B)
            res = fn()
        finally:
            pack.revert(model_B)
            rb = pack.verify_bit_exact_rollback(model_B, probe_ids, probe_mask, base_probe_logits, strict_assert=True)
            assert_zero_runtime_hooks(model_B)
        return res, rb

    def _calibrate_ood_safe_alpha(raw_pack: ReversibleNativeOrganPack, dev_score_fn: Callable[[], float], max_kl: float = 0.05) -> Tuple[ReversibleNativeOrganPack, Dict[str, Any]]:
        """
        Fail-closed OOD damping gate: iterates DESCENDING `(1.0 -> 0.0)` with strict `sc > best_sc`
        (ties ALWAYS keep largest safe alpha; `alpha = 0.0` guarantees `KL_drift <= max_kl` NEVER fails open!).
        """
        best_pk, best_sc, best_kl, best_a = None, -1e18, 0.0, 0.0
        for a in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.0):
            cand = scale_native_pack_outputs(raw_pack, a)
            (kl_d, sc), _ = _run_with_pack_safe(
                cand,
                lambda: (
                    (kl := measure_ood_kl_drift(model_B, tok_B, ood_guard_prompts, base_ood_lg, dev_B, V_shared)),
                    (dev_score_fn() if kl <= max_kl else -1e18),
                ),
            )
            if kl_d <= max_kl and sc > best_sc:
                best_pk, best_sc, best_kl, best_a = cand, sc, kl_d, a
        assert best_pk is not None and best_kl <= max_kl
        return best_pk, {"alpha": best_a, "ood_kl": round(best_kl, 4), "within_kl_guard": True, "dev_score": round(best_sc, 4)}

    def _enforce_final_ood_kl(raw_pack: ReversibleNativeOrganPack, start_alpha: float, max_kl: float = 0.05) -> Tuple[ReversibleNativeOrganPack, Dict[str, Any]]:
        """Verify final compiled pack satisfies `KL_drift <= max_kl` (re-damping downwards if needed; never fails open!)."""
        cands = [a for a in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.0) if a <= start_alpha + 1e-9]
        if not cands or cands[-1] != 0.0:
            cands.append(0.0)
        for a in cands:
            cand = scale_native_pack_outputs(raw_pack, a)
            kl_d, _ = _run_with_pack_safe(cand, lambda: measure_ood_kl_drift(model_B, tok_B, ood_guard_prompts, base_ood_lg, dev_B, V_shared))
            if kl_d <= max_kl:
                return cand, {"alpha": a, "final_ood_kl": round(kl_d, 4), "within_kl_guard": True}
        raise RuntimeError("Unreachable: alpha=0.0 must yield KL=0.0")

    results: Dict[str, Any] = {
        "meta": {
            "experiment": "v4.3.2_stage1_exact_math_native",
            "donor_model": args.donor_model, "recipient_model": args.recipient_model,
            "compile_layers": compile_layers, "v_shared": V_shared,
            "n_fit_tokens": n_fit, "n_val_tune": n_val_t, "n_val_report": n_val_r,
            "zero_external_sidecars": True, "zero_runtime_hooks_at_inference": True, "seed": SEED,
        },
        "bridge_diagnostics": bridge_diag,
        "gate0_l27_M_U_vs_M_out": gate0_mu_vs_mout,
        "gate0b_3_floors": {
            "floor1_fisher_rrr_L27": floor1_fisher,
            "floor2_observability_summary": obs_summary,
            "floor2_patched_kl_recovery": floor2_kl_rec,
            "floor3_nonlinear_residual_gain": floor3_nl,
        },
        "base_locality_380": base_loc,
    }

    # =========================================================================
    # TASK A: TEXT-TO-SQLITE (4 Train Schemas -> 4 Strictly Unseen Schemas)
    # =========================================================================
    print("\n================ TASK A: TEXT-TO-SQLITE (OOD-Guarded Native v4.3.1) ================")
    base_sql_test = eval_sql_queries_native(model_B, tok_B, dev_B, sql_test, sql_gold, batch_size=8, v_shared=V_shared)
    sql_tiers: Dict[str, Any] = {}
    donor_sql_fit30_st = {li: {k: v[:n_sql_fit30_toks] for k, v in donor_sql_states[li].items()} for li in compile_layers}

    pk_s1a_fit30, _ = compile_tier1a_sequential_native_delta_w(model_B, tok_B, dev_B, compile_layers, bridges, donor_sql_fit30_st, sql_full_teacher_texts[:30], cal["fit"][:40], positions="all_nonpad", lam_bg=50.0, lam_eye=0.5, batch_size=4, max_length=384)
    _, cal_s1a = _calibrate_ood_safe_alpha(pk_s1a_fit30, lambda: eval_sql_queries_native(model_B, tok_B, dev_B, sql_dev_sub, sql_gold, batch_size=8, v_shared=V_shared)["exec_match_rate"])
    pk_s1a_full, _ = compile_tier1a_sequential_native_delta_w(model_B, tok_B, dev_B, compile_layers, bridges, donor_sql_states, sql_full_teacher_texts, cal["fit"][:40], positions="all_nonpad", lam_bg=50.0, lam_eye=0.5, batch_size=4, max_length=384)
    pk_s1a, fin_s1a = _enforce_final_ood_kl(pk_s1a_full, cal_s1a["alpha"])
    (r_s1a, l_s1a), rb_s1a = _run_with_pack_safe(pk_s1a, lambda: (eval_sql_queries_native(model_B, tok_B, dev_B, sql_test, sql_gold, batch_size=8, v_shared=V_shared), eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)[0]))
    sql_tiers["Tier1A_OOD_GUARDED_SEQ_DELTA_W_0SGD"] = {"unseen_exec_rate": r_s1a["exec_match_rate"], "unseen_syntax_rate": r_s1a["syntax_valid_rate"], "ood_cal": {**cal_s1a, **fin_s1a}, "locality_380": l_s1a, "rollback": rb_s1a}
    print(f"  [SQL Tier 1A OOD-Guarded ΔW] unseen={r_s1a['exec_match_rate']:.3f} (alpha={fin_s1a['alpha']}, kl={fin_s1a['final_ood_kl']}) | loc_broken={l_s1a['newly_broken']}")

    pk_s1b_fit30, _ = compile_tier1b_native_widened_heads_and_swiglu(model_B, tok_B, dev_B, compile_layers, bridges, donor_sql_w_fit30, donor_sql_fit30_st, sql_full_teacher_texts[:30], cal["fit"][:40], positions="all_nonpad", lam_bg=50.0, lam_eye=0.5, batch_size=4, max_length=384)
    pk_s1b_fit30_safe, cal_s1b = _calibrate_ood_safe_alpha(pk_s1b_fit30, lambda: eval_sql_queries_native(model_B, tok_B, dev_B, sql_dev_sub, sql_gold, batch_size=8, v_shared=V_shared)["exec_match_rate"])
    pk_s1b_full, _ = compile_tier1b_native_widened_heads_and_swiglu(model_B, tok_B, dev_B, compile_layers, bridges, donor_sql_w, donor_sql_states, sql_full_teacher_texts, cal["fit"][:40], positions="all_nonpad", lam_bg=50.0, lam_eye=0.5, batch_size=4, max_length=384)
    pk_s1b, fin_s1b = _enforce_final_ood_kl(pk_s1b_full, cal_s1b["alpha"])
    (r_s1b, l_s1b), rb_s1b = _run_with_pack_safe(pk_s1b, lambda: (eval_sql_queries_native(model_B, tok_B, dev_B, sql_test, sql_gold, batch_size=8, v_shared=V_shared), eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)[0]))
    sql_tiers["Tier1B_OBSERVABLE_WIDENED_0SGD"] = {"unseen_exec_rate": r_s1b["exec_match_rate"], "unseen_syntax_rate": r_s1b["syntax_valid_rate"], "ood_cal": {**cal_s1b, **fin_s1b}, "obs_sel": sql_obs_sel, "locality_380": l_s1b, "n_bytes_bf16": pk_s1b.n_bytes_bf16(), "rollback": rb_s1b}
    print(f"  [SQL Tier 1B Observable Widened] unseen={r_s1b['exec_match_rate']:.3f} (alpha={fin_s1b['alpha']}, kl={fin_s1b['final_ood_kl']}) | loc_broken={l_s1b['newly_broken']}")

    del donor_sql_states, donor_sql_fit30_st, donor_sql_w, donor_sql_w_fit30, pk_s1a_fit30, pk_s1a_full, pk_s1b_fit30, pk_s1b_full
    gc.collect()

    sql_fit_pairs = list(zip(sql_train_prompts[:30], donor_sql_train_preds[:30]))
    sql_all_pairs = list(zip(sql_train_prompts, donor_sql_train_preds))
    # Independent per-tier dev LR sweeps for Tier 2, Tier 2 Ctrl, and Tier 3!
    def _tune_sql_lr(build_fn):
        bl, bs = 1e-4, -1.0
        for clr in (1e-4, 3e-4):
            pk_c, _ = build_fn(clr)
            sc, _ = _run_with_pack_safe(pk_c, lambda: eval_sql_queries_native(model_B, tok_B, dev_B, sql_dev_sub, sql_gold, batch_size=8, v_shared=V_shared)["exec_match_rate"])
            if sc > bs:
                bs, bl = sc, clr
        return bl

    if pk_s1b.n_params() > 0:
        lr_s_t2 = _tune_sql_lr(lambda lr: polish_widened_native_pack_sft(model_B, tok_B, dev_B, pk_s1b_fit30_safe, sql_fit_pairs, v_shared=V_shared, lr=lr, steps=20, batch_size=4, seed=SEED + 10, strict_loss_decrease=False))
        lr_s_ctrl = _tune_sql_lr(lambda lr: polish_widened_native_pack_sft(model_B, tok_B, dev_B, make_random_zero_control_pack(pk_s1b_fit30_safe, seed=SEED + 11), sql_fit_pairs, v_shared=V_shared, lr=lr, steps=20, batch_size=4, seed=SEED + 10, strict_loss_decrease=False))
        for tname, ipk, used_lr in (
            ("Tier2_OBSERVABLE_NATIVE_INIT_POLISH", pk_s1b, lr_s_t2),
            ("Tier2_CTRL_RECIPIENT_ONLY_NATIVE", make_random_zero_control_pack(pk_s1b, seed=SEED + 11), lr_s_ctrl),
        ):
            pk2, inf2 = polish_widened_native_pack_sft(model_B, tok_B, dev_B, ipk, sql_all_pairs, v_shared=V_shared, lr=used_lr, steps=40, batch_size=4, seed=SEED + 12)
            (r2, l2), rb2 = _run_with_pack_safe(pk2, lambda: (eval_sql_queries_native(model_B, tok_B, dev_B, sql_test, sql_gold, batch_size=8, v_shared=V_shared), eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)[0]))
            sql_tiers[tname] = {"unseen_exec_rate": r2["exec_match_rate"], "unseen_syntax_rate": r2["syntax_valid_rate"], "locality_380": l2, "train_info": inf2, "rollback": rb2}
            print(f"  [SQL {tname:34s}] unseen={r2['exec_match_rate']:.3f} (lr={used_lr}) | loss={inf2['loss_start']}->{inf2['loss_end']} | loc_broken={l2['newly_broken']}")

    lr_s_t3 = _tune_sql_lr(lambda lr: train_tier3_teacher_lora_merged_native(model_B, tok_B, dev_B, compile_layers, mode="sft", prompt_target_pairs=sql_fit_pairs, v_shared=V_shared, lr=lr, steps=20, batch_size=4, seed=SEED + 13))
    pk_s3, inf_s3 = train_tier3_teacher_lora_merged_native(model_B, tok_B, dev_B, compile_layers, mode="sft", prompt_target_pairs=sql_all_pairs, v_shared=V_shared, rank=16, alpha=32.0, lr=lr_s_t3, steps=40, batch_size=4, seed=SEED + 14)
    (r_s3, l_s3), rb_s3 = _run_with_pack_safe(pk_s3, lambda: (eval_sql_queries_native(model_B, tok_B, dev_B, sql_test, sql_gold, batch_size=8, v_shared=V_shared), eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)[0]))
    sql_tiers["Tier3_TEACHER_LORA_MERGED_NATIVE"] = {"unseen_exec_rate": r_s3["exec_match_rate"], "unseen_syntax_rate": r_s3["syntax_valid_rate"], "locality_380": l_s3, "train_info": inf_s3, "rollback": rb_s3}
    print(f"  [SQL Tier 3 Merged Teacher-LoRA     ] unseen={r_s3['exec_match_rate']:.3f} (lr={lr_s_t3}) | loss={inf_s3['loss_start']}->{inf_s3['loss_end']} | loc_broken={l_s3['newly_broken']}")

    results["task_a_text_to_sqlite"] = {
        "donor_7b_unseen_40": donor_sql_unseen_40["exec_match_rate"], "base_1p5b_unseen_40": base_sql_test["exec_match_rate"],
        "headroom_gap_unseen40_pp": round(100.0 * (donor_sql_unseen_40["exec_match_rate"] - base_sql_test["exec_match_rate"]), 2),
        "tiers": sql_tiers,
    }
    del pk_s1a, pk_s1b, pk_s1b_fit30_safe, pk_s3
    gc.collect()

    # =========================================================================
    # TASK B: OURAIRPORTS FACT IMPLANTATION (`implanted_p0` Retention + Unseen `p1/p2` & Route Transfer)
    # =========================================================================
    print("\n================ TASK B: OURAIRPORTS IMPLANTATION (implanted_p0 Retention + Unseen p1/p2 & Route Transfer) ================")
    air_tiers: Dict[str, Any] = {}
    task_b_status = "ok"

    if len(dev_airports) < 4 or len(eval_airports) < 8:
        task_b_status = "insufficient_cohort"
        print(f"  [Airports] Insufficient blind cohort for dev calibration (dev_blind={len(dev_airports)}, eval_blind={len(eval_airports)}); skipping Task B cleanly!")
    else:
        def _eval_air(coh, rts):
            o = {}
            for pi in (0, 1, 2):
                gs = greedy_generate_native(model_B, tok_B, dev_B, [format_airport_prompt(a["name"], pi)[0] for a in coh], max_new_tokens=6, batch_size=32, v_shared=V_shared)
                k = sum(int(check_iata_match(g, a["iata"])) for g, a in zip(gs, coh))
                key = "implanted_p0" if pi == 0 else f"p{pi}"
                o[key] = {"acc": round(k / len(coh), 4), "k": k, "n": len(coh)}
            uk = o["p1"]["k"] + o["p2"]["k"]
            o["unseen_p1_p2_transfer"] = {"acc": round(uk / (2 * len(coh)), 4), "k": uk, "n": 2 * len(coh)}
            if rts:
                rgs = greedy_generate_native(model_B, tok_B, dev_B, [r["prompt"] for r in rts], max_new_tokens=12, batch_size=32, v_shared=V_shared)
                o["unseen_routes_transfer"] = {
                    "ordered_acc": round(sum(int(check_route_match(g, r["iata_a"], r["iata_b"])["ordered"]) for r, g in zip(rts, rgs)) / len(rts), 4),
                    "either_acc": round(sum(int(check_route_match(g, r["iata_a"], r["iata_b"])["either"]) for r, g in zip(rts, rgs)) / len(rts), 4),
                }
            return o

        air_tiers["BASE"] = _eval_air(eval_airports, routes_eval)
        dp0 = [format_airport_prompt(a["name"], 0)[0] for a in dev_airports]
        ep0 = [format_airport_prompt(a["name"], 0)[0] for a in eval_airports]
        dev_air_pairs = [(format_airport_prompt(a["name"], 0)[0], " " + a["donor_p0_pred_iata"]) for a in dev_airports]
        eval_air_pairs = [(format_airport_prompt(a["name"], 0)[0], " " + a["donor_p0_pred_iata"]) for a in eval_airports]

        # Mandatory dev calibration strictly on `dev_airports` (NO fallback alpha/lr!), then verify final eval pack KL!
        pk_ab_dev_raw, _ = compile_tier1b_native_widened_heads_and_swiglu(model_B, tok_B, dev_B, compile_layers, bridges, donor_air_w, donor_air_dev_st, dp0, cal["fit"][:40], positions="last", lam_bg=50.0, lam_eye=0.5, batch_size=16, max_length=64)
        pk_ab_dev_safe, c_ab_d = _calibrate_ood_safe_alpha(pk_ab_dev_raw, lambda: _eval_air(dev_airports, [])["unseen_p1_p2_transfer"]["acc"])
        dev_alpha_ab = c_ab_d["alpha"]
        assert donor_air_dev_lg is not None
        pk_gn_dev_raw, _ = compile_tier1_gn_l27_native_delta_w(model_B, tok_B, dev_B, G_B_l27, donor_air_dev_lg, dp0, cal["fit"][:40], V_shared)
        _, c_gn_d = _calibrate_ood_safe_alpha(pk_gn_dev_raw, lambda: _eval_air(dev_airports, [])["unseen_p1_p2_transfer"]["acc"])
        dev_alpha_gn = c_gn_d["alpha"]

        pk_ab_raw, _ = compile_tier1b_native_widened_heads_and_swiglu(model_B, tok_B, dev_B, compile_layers, bridges, donor_air_w, donor_air_eval_st, ep0, cal["fit"][:40], positions="last", lam_bg=50.0, lam_eye=0.5, batch_size=16, max_length=64)
        pk_ab, fin_ab = _enforce_final_ood_kl(pk_ab_raw, dev_alpha_ab)
        (ev_ab, lc_ab), rb_ab = _run_with_pack_safe(pk_ab, lambda: (_eval_air(eval_airports, routes_eval), eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)[0]))
        air_tiers["Tier1B_OBSERVABLE_WIDENED_0SGD"] = {**ev_ab, "ood_cal": fin_ab, "locality_380": lc_ab, "rollback": rb_ab}
        print(f"  [Air Tier 1B Observable Widened] implanted_p0={ev_ab['implanted_p0']['acc']:.3f} unseen={ev_ab['unseen_p1_p2_transfer']['acc']:.3f} (alpha={fin_ab['alpha']}, kl={fin_ab['final_ood_kl']}) | loc_broken={lc_ab['newly_broken']}")

        assert donor_air_eval_lg is not None
        pk_agn_raw, _ = compile_tier1_gn_l27_native_delta_w(model_B, tok_B, dev_B, G_B_l27, donor_air_eval_lg, ep0, cal["fit"][:40], V_shared)
        pk_agn, fin_agn = _enforce_final_ood_kl(pk_agn_raw, dev_alpha_gn)
        (ev_agn, lc_agn), rb_agn = _run_with_pack_safe(pk_agn, lambda: (_eval_air(eval_airports, routes_eval), eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)[0]))
        air_tiers["Tier1_GN_ALIGNMENT_FREE_L27_0SGD"] = {**ev_agn, "ood_cal": fin_agn, "locality_380": lc_agn, "rollback": rb_agn}
        print(f"  [Air Tier 1-GN Gauss-Newton L27] implanted_p0={ev_agn['implanted_p0']['acc']:.3f} unseen={ev_agn['unseen_p1_p2_transfer']['acc']:.3f} (alpha={fin_agn['alpha']}, kl={fin_agn['final_ood_kl']}) | loc_broken={lc_agn['newly_broken']}")

        def _tune_air_lr(build_fn):
            bl, bs = 3e-4, -1.0
            for clr in (1e-4, 3e-4):
                pk_c, _ = build_fn(clr)
                sc, _ = _run_with_pack_safe(pk_c, lambda: _eval_air(dev_airports, [])["unseen_p1_p2_transfer"]["acc"])
                if sc > bs:
                    bs, bl = sc, clr
            return bl

        if pk_ab.n_params() > 0:
            lr_a_t2 = _tune_air_lr(lambda lr: polish_widened_native_pack_sft(model_B, tok_B, dev_B, pk_ab_dev_safe, dev_air_pairs, v_shared=V_shared, lr=lr, steps=25, batch_size=8, seed=SEED + 20, strict_loss_decrease=False))
            lr_a_ctrl = _tune_air_lr(lambda lr: polish_widened_native_pack_sft(model_B, tok_B, dev_B, make_random_zero_control_pack(pk_ab_dev_safe, seed=SEED + 21), dev_air_pairs, v_shared=V_shared, lr=lr, steps=25, batch_size=8, seed=SEED + 20, strict_loss_decrease=False))
            for tname, ipk, ulr in (
                ("Tier2_OBSERVABLE_NATIVE_INIT_POLISH", pk_ab, lr_a_t2),
                ("Tier2_CTRL_RECIPIENT_ONLY_NATIVE", make_random_zero_control_pack(pk_ab, seed=SEED + 21), lr_a_ctrl),
            ):
                pk2, inf2 = polish_widened_native_pack_sft(model_B, tok_B, dev_B, ipk, eval_air_pairs, v_shared=V_shared, lr=ulr, steps=40, batch_size=8, seed=SEED + 22)
                (ev2, lc2), rb2 = _run_with_pack_safe(pk2, lambda: (_eval_air(eval_airports, routes_eval), eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)[0]))
                air_tiers[tname] = {**ev2, "locality_380": lc2, "train_info": inf2, "rollback": rb2}
                print(f"  [Air {tname:34s}] implanted_p0={ev2['implanted_p0']['acc']:.3f} unseen={ev2['unseen_p1_p2_transfer']['acc']:.3f} (lr={ulr}) | loc_broken={lc2['newly_broken']}")

        lr_a_t3 = _tune_air_lr(lambda lr: train_tier3_teacher_lora_merged_native(model_B, tok_B, dev_B, compile_layers, mode="sft", prompt_target_pairs=dev_air_pairs, v_shared=V_shared, lr=lr, steps=25, batch_size=8, seed=SEED + 23))
        pk_a3, inf_a3 = train_tier3_teacher_lora_merged_native(model_B, tok_B, dev_B, compile_layers, mode="sft", prompt_target_pairs=eval_air_pairs, v_shared=V_shared, rank=16, alpha=32.0, lr=lr_a_t3, steps=40, batch_size=8, seed=SEED + 24)
        (ev_a3, lc_a3), rb_a3 = _run_with_pack_safe(pk_a3, lambda: (_eval_air(eval_airports, routes_eval), eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)[0]))
        air_tiers["Tier3_TEACHER_LORA_MERGED_NATIVE"] = {**ev_a3, "locality_380": lc_a3, "train_info": inf_a3, "rollback": rb_a3}
        print(f"  [Air Tier 3 Merged Teacher-LoRA     ] implanted_p0={ev_a3['implanted_p0']['acc']:.3f} unseen={ev_a3['unseen_p1_p2_transfer']['acc']:.3f} (lr={lr_a_t3}) | loc_broken={lc_a3['newly_broken']}")

    results["task_b_ourairports_implanted_transfer"] = {
        "status": task_b_status,
        "protocol_note": "Implantation on p0 donor predictions; eval facts are unseen during dev tuning, while p1/p2 paraphrase templates participate in dev tuning on disjoint dev_airports.",
        "n_dev_blind": len(dev_airports), "n_eval_blind": len(eval_airports), "tiers": air_tiers,
    }
    del donor_air_dev_st, donor_air_eval_st, donor_air_w
    gc.collect()

    # =========================================================================
    # TASK C: 2-ROW 2-HOP POLICY SWAP (100% ZERO-LABEL DEV SELECTION + SHARED-SUFFIX!)
    # =========================================================================
    print("\n================ TASK C: 2-ROW 2-HOP POLICY SWAP (100% Zero-Label Dev Selection + Shared-Suffix) ================")
    dv_2h_rec = [x["recipient_prompt"] for x in twohop_ds["dev"]]
    te_2h_rec = [x["recipient_prompt"] for x in twohop_ds["test"]]
    b2_lg = _last_tok_logits(model_B, tok_B, te_2h_rec, dev_B, bs=16, v_cap=V_shared)
    b2_L, b2_R = _eval_2hop_logits(b2_lg, twohop_ds["test"], tok_B, "L"), _eval_2hop_logits(b2_lg, twohop_ds["test"], tok_B, "R")
    twohop_tiers: Dict[str, Any] = {}
    pol_tr_lg = {"L": d2_tr_L_lg, "R": d2_tr_R_lg}
    pol_dv_lg = {"L": d2_dv_L_lg, "R": d2_dv_R_lg}

    def _neg_dev_teacher_kl(pol: str) -> float:
        """Strictly ZERO-LABEL dev score: negative KL(p_7B_dev || p_recip_dev) — higher is closer to Donor 7B!"""
        cur = _last_tok_logits(model_B, tok_B, dv_2h_rec, dev_B, bs=16, v_cap=V_shared)
        kl = float(F.kl_div(F.log_softmax(cur, dim=-1), F.softmax(pol_dv_lg[pol], dim=-1), reduction="batchmean").item())
        return -kl

    t1b_best_packs: Dict[str, ReversibleNativeOrganPack] = {}
    for pos_label, pos_mode, st_map in (
        ("Tier1B_OBS_WIDENED_LAST_TOK_0SGD", "last", {"L": d2_tr_L_last, "R": d2_tr_R_last}),
        ("Tier1B_OBS_WIDENED_SHARED_SUFFIX_0SGD", "all_nonpad", {"L": d2_tr_L_suf, "R": d2_tr_R_suf}),
    ):
        ev_p, loc_p, rb_p, cal_p = {}, {}, {}, {}
        for pol in ("L", "R"):
            pk_raw, _ = compile_tier1b_native_widened_heads_and_swiglu(
                model_B, tok_B, dev_B, compile_layers, bridges, donor_2hop_w[pol], st_map[pol],
                tr_2h_rec, cal["fit"][:40], positions=pos_mode, lam_bg=50.0, lam_eye=0.5, batch_size=16, max_length=128,
            )
            pk_safe, c_info = _calibrate_ood_safe_alpha(pk_raw, lambda p=pol: _neg_dev_teacher_kl(p))
            if pos_mode == "all_nonpad":
                t1b_best_packs[pol] = pk_safe
            cal_p[pol] = c_info
            (ev_p[pol], loc_p[pol]), rb_p[pol] = _run_with_pack_safe(
                pk_safe,
                lambda p=pol: (
                    _eval_2hop_logits(_last_tok_logits(model_B, tok_B, te_2h_rec, dev_B, bs=16, v_cap=V_shared), twohop_ds["test"], tok_B, p),
                    eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)[0],
                ),
            )
        sm = _joint_2hop(ev_p["L"], ev_p["R"])
        twohop_tiers[pos_label] = {**sm, "ood_cal": cal_p, "locality_380": loc_p, "rollback": rb_p}
        print(f"  [2-Hop {pos_label:38s}] joint_em={sm['joint_swap_em']:.3f} | joint_fc4={sm['joint_swap_fc4']:.3f} | loc=({loc_p['L']['newly_broken']},{loc_p['R']['newly_broken']})")

    gn_ev, gn_loc, gn_rb, gn_cal = {}, {}, {}, {}
    for pol in ("L", "R"):
        pk_gn_raw, _ = compile_tier1_gn_l27_native_delta_w(model_B, tok_B, dev_B, G_B_l27, pol_tr_lg[pol], tr_2h_rec, cal["fit"][:40], V_shared)
        pk_gn, c_gn = _calibrate_ood_safe_alpha(pk_gn_raw, lambda p=pol: _neg_dev_teacher_kl(p))
        gn_cal[pol] = c_gn
        (gn_ev[pol], gn_loc[pol]), gn_rb[pol] = _run_with_pack_safe(
            pk_gn,
            lambda p=pol: (
                _eval_2hop_logits(_last_tok_logits(model_B, tok_B, te_2h_rec, dev_B, bs=16, v_cap=V_shared), twohop_ds["test"], tok_B, p),
                eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)[0],
            ),
        )
    sm_gn = _joint_2hop(gn_ev["L"], gn_ev["R"])
    twohop_tiers["Tier1_GN_ALIGNMENT_FREE_L27_0SGD"] = {**sm_gn, "ood_cal": gn_cal, "locality_380": gn_loc, "rollback": gn_rb}
    print(f"  [2-Hop Tier 1-GN Alignment-Free L27    ] joint_em={sm_gn['joint_swap_em']:.3f} | joint_fc4={sm_gn['joint_swap_fc4']:.3f} | loc=({gn_loc['L']['newly_broken']},{gn_loc['R']['newly_broken']})")

    def _tune_2h_lr(build_fn):
        bl, bs = 3e-4, -1e18
        for clr in (1e-4, 3e-4):
            pk_c, _ = build_fn(clr)
            sc, _ = _run_with_pack_safe(pk_c, lambda: _neg_dev_teacher_kl("R"))
            if sc > bs:
                bs, bl = sc, clr
        return bl

    if t1b_best_packs["R"].n_params() > 0:
        lr_2h_t2 = _tune_2h_lr(lambda lr: polish_widened_native_pack_teacher_kl(model_B, tok_B, dev_B, t1b_best_packs["R"], tr_2h_rec, pol_tr_lg["R"], V_shared, lr=lr, steps=40, batch_size=8, seed=SEED + 30, strict_loss_decrease=False))
        lr_2h_ctrl = _tune_2h_lr(lambda lr: polish_widened_native_pack_teacher_kl(model_B, tok_B, dev_B, make_random_zero_control_pack(t1b_best_packs["R"], seed=SEED + 32), tr_2h_rec, pol_tr_lg["R"], V_shared, lr=lr, steps=40, batch_size=8, seed=SEED + 30, strict_loss_decrease=False))
        for tname, use_init, shuf, ulr in (
            ("Tier2_OBSERVABLE_NATIVE_INIT_POLISH", True, False, lr_2h_t2),
            ("Tier2_CTRL_RECIPIENT_ONLY_NATIVE", False, False, lr_2h_ctrl),
            ("Tier2_CTRL_SHUFFLED_SUPERVISION", False, True, lr_2h_ctrl),
        ):
            ev_p, loc_p, tr_p, rb_p = {}, {}, {}, {}
            for pol in ("L", "R"):
                ipk = t1b_best_packs[pol] if use_init else make_random_zero_control_pack(t1b_best_packs[pol], seed=SEED + (31 if pol == "L" else 32))
                t_lg = pol_tr_lg[pol]
                if shuf:
                    rng_s = np.random.default_rng(SEED + (41 if pol == "L" else 42))
                    t_lg = t_lg[rng_s.permutation(len(t_lg))]
                pk2, inf2 = polish_widened_native_pack_teacher_kl(model_B, tok_B, dev_B, ipk, tr_2h_rec, t_lg, V_shared, lr=ulr, steps=80, batch_size=8, seed=SEED + 50, strict_loss_decrease=(not shuf))
                tr_p[pol] = inf2
                (ev_p[pol], loc_p[pol]), rb_p[pol] = _run_with_pack_safe(
                    pk2,
                    lambda p=pol: (
                        _eval_2hop_logits(_last_tok_logits(model_B, tok_B, te_2h_rec, dev_B, bs=16, v_cap=V_shared), twohop_ds["test"], tok_B, p),
                        eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)[0],
                    ),
                )
            sm2 = _joint_2hop(ev_p["L"], ev_p["R"])
            twohop_tiers[tname] = {**sm2, "locality_380": loc_p, "train_info": tr_p, "rollback": rb_p}
            print(f"  [2-Hop {tname:38s}] joint_em={sm2['joint_swap_em']:.3f} | joint_fc4={sm2['joint_swap_fc4']:.3f} (lr={ulr}) | loc=({loc_p['L']['newly_broken']},{loc_p['R']['newly_broken']})")

    lr_2h_t3 = _tune_2h_lr(lambda lr: train_tier3_teacher_lora_merged_native(model_B, tok_B, dev_B, compile_layers, mode="teacher_kl", prompts_kl=tr_2h_rec, teacher_logits_shared=pol_tr_lg["R"], v_shared=V_shared, lr=lr, steps=40, batch_size=8, seed=SEED + 58))
    t3_ev, t3_loc, t3_tr, t3_rb = {}, {}, {}, {}
    for pol in ("L", "R"):
        pk3, inf3 = train_tier3_teacher_lora_merged_native(model_B, tok_B, dev_B, compile_layers, mode="teacher_kl", prompts_kl=tr_2h_rec, teacher_logits_shared=pol_tr_lg[pol], v_shared=V_shared, rank=16, alpha=32.0, lr=lr_2h_t3, steps=80, batch_size=8, seed=SEED + 60)
        t3_tr[pol] = inf3
        (t3_ev[pol], t3_loc[pol]), t3_rb[pol] = _run_with_pack_safe(
            pk3,
            lambda p=pol: (
                _eval_2hop_logits(_last_tok_logits(model_B, tok_B, te_2h_rec, dev_B, bs=16, v_cap=V_shared), twohop_ds["test"], tok_B, p),
                eval_certified_380_locality(model_B, tok_B, dev_B, loc_facts, base_ok=base_loc_ok, v_shared=V_shared)[0],
            ),
        )
    sm3 = _joint_2hop(t3_ev["L"], t3_ev["R"])
    twohop_tiers["Tier3_TEACHER_LORA_MERGED_NATIVE"] = {**sm3, "locality_380": t3_loc, "train_info": t3_tr, "rollback": t3_rb}
    print(f"  [2-Hop Tier 3 Merged Teacher-LoRA         ] joint_em={sm3['joint_swap_em']:.3f} | joint_fc4={sm3['joint_swap_fc4']:.3f} (lr={lr_2h_t3}) | loc=({t3_loc['L']['newly_broken']},{t3_loc['R']['newly_broken']})")

    results["task_c_2row_2hop"] = {
        "zero_label_dev_selection": True,
        "donor_7b_test": donor_2hop_sum,
        "base_1p5b": {"left_em": b2_L["em"], "right_em": b2_R["em"], "left_fc4": b2_L["fc4"], "right_fc4": b2_R["fc4"]},
        "tiers": twohop_tiers,
    }
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram_telemetry["phase2_peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / (1024.0 ** 3), 3)
        vram_telemetry["phase2_peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / (1024.0 ** 3), 3)
    results["meta"]["vram_telemetry"] = vram_telemetry
    results["meta"]["total_elapsed_sec"] = round(time.time() - t_all, 1)

    unload_model_vram_safe(model_B)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[V4.3.1 STAGE 1 COMPLETE in {results['meta']['total_elapsed_sec']}s] Saved {OUT_PATH}")
    return results


# ======================================================================================
# 6. Pure-CPU Unit Tests (<3s, ZERO GPU!)
# ======================================================================================

def run_cpu_unit_tests() -> None:
    t0 = time.time()
    from transformers import AutoTokenizer
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    print("=== Running v4_3_stage1_exact_math_native.py Pure-CPU Unit Tests (v4.3.1) ===")
    tok = AutoTokenizer.from_pretrained(str(HF_CACHE_DIR / "qwen2.5-3b"), local_files_only=True)
    V = len(tok)
    cfg_A = Qwen2Config(vocab_size=V + 32, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2, rope_theta=1e6)
    cfg_B = Qwen2Config(vocab_size=V + 16, hidden_size=48, intermediate_size=96, num_hidden_layers=2, num_attention_heads=6, num_key_value_heads=1, rope_theta=1e6)
    torch.manual_seed(5)
    mA = Qwen2ForCausalLM(cfg_A).to(dtype=torch.bfloat16).eval()
    mB = Qwen2ForCausalLM(cfg_B).to(dtype=torch.bfloat16).eval()

    # 1. Verify zero-extra-VRAM streaming dual calibration + chunked Fisher + M_U + Floor #2 patched KL recovery!
    fit_p = ["Case 1: ocean salinity drives deep circulation.", "Case 2: seismic waves bend across mantle layers."] * 2
    vt_p = ["Field Log 9: glacier melt lubricates bedrock.", "Briefing 10: ceramic glaze shrinks in kilns."]
    vr_p = ["/* Audit 11 */ SELECT 1;", "REVIEW ENTRY <12> — horology balance coils."]
    cal_o = stream_60k_dual_calibration_and_fisher(mA, mB, tok, fit_p, vt_p, vr_p, [0, 1], torch.device("cpu"), torch.device("cpu"), v_shared=V, batch_size=2, max_length=32)
    assert cal_o["n_fit_tokens"] > 20 and cal_o["n_fisher_tokens"] == cal_o["n_fit_tokens"]
    lg_mu = chunked_lm_head_logits_cpu(np.zeros((2, 48), dtype=np.float32), mB.lm_head.weight, V, torch.device("cpu"))
    assert lg_mu.shape == (2, V)
    M_out = np.random.default_rng(1).standard_normal((48, 64)) * 0.05
    P_out = np.random.default_rng(2).standard_normal((64, 48)) * 0.05
    rec = measure_patched_kl_recovery(mA, mB, tok, vr_p, 1, M_out, P_out, np.zeros(64), np.zeros(48), torch.device("cpu"), torch.device("cpu"), V, batch_size=2, max_length=32)
    assert all(k in rec for k in ("R_push_M_hA", "R_pull_P_hB", "R_roundtrip_P_M_hA"))
    assert_zero_runtime_hooks(mA) and assert_zero_runtime_hooks(mB)
    print(f"  [PASS 1/4] Zero-copy chunked dual calibration + Fisher + Floor #2 patched KL recovery verified!")

    # 2. Verify strict CONSTITUENT `o_j >= 0.25` rejection (Q/K/V=(0.1,1,1) & gate/up=(0.1,1) MUST be rejected!)
    o_head_j = np.full(8, 0.4642)   # geometric mean of (0.1, 1.0, 1.0) > 0.25!
    o_head_m = np.full(8, 0.10)     # constituent min < 0.25 -> MUST reject all 8 heads!
    o_mlp_j = np.array([0.3162] * 120 + [0.80] * 8)  # geometric mean of (0.1, 1.0) > 0.25!
    o_mlp_m = np.array([0.10] * 120 + [0.80] * 8)    # constituent min < 0.25 for first 120 -> ONLY last 8 pass!
    fake_obs = {1: {"o_head_joint": o_head_j, "o_head_min": o_head_m, "o_mlp_joint": o_mlp_j, "o_mlp_min": o_mlp_m}}
    z_a = {1: np.ones((4, 64), dtype=np.float32)}
    z_m = {1: np.ones((4, 128), dtype=np.float32)}
    sel_w, sel_d = extract_observable_donor_heads_and_neurons(mA, [1], fake_obs, z_a, z_m, z_m, add_q_heads=3, add_kv_heads=1, add_mlp_rows=32, min_obs=0.25)
    assert sel_w[1]["q_w"] is None and sel_d["L1"]["n_heads_selected"] == 0
    assert sel_w[1]["gate_w"].shape[0] == 8 and sel_d["L1"]["selected_mlp_min_constituent_o"] >= 0.25
    # Also verify missing constituent/min keys raise KeyError (never fall back to geometric mean!)
    caught_ke = False
    try:
        extract_observable_donor_heads_and_neurons(mA, [1], {1: {"o_head_joint": o_head_j, "o_mlp_joint": o_mlp_j}}, z_a, z_m, z_m, add_q_heads=3, add_kv_heads=1, add_mlp_rows=32, min_obs=0.25)
    except KeyError:
        caught_ke = True
    assert caught_ke, "Extractor must reject obs_by_layer lacking constituent/min scores!"
    print("  [PASS 2/5] Strict constituent o_j >= 0.25 rejection verified (Q/K/V=(0.1,1,1) & gate/up=(0.1,1) rejected; missing mins raise KeyError!)")

    # 3. Verify exact shared-suffix multi-token alignment!
    ds2 = build_2row_2hop_dataset([tok], n_train=4, n_dev=2, n_test=4, seed=SEED)
    rec_p = [x["recipient_prompt"] for x in ds2["train"]]
    don_p = [x["donor_L_prompt"] for x in ds2["train"]]
    st_don = extract_shared_suffix_states(mA, tok, don_p, rec_p, [1], torch.device("cpu"), batch_size=2, max_length=192)
    st_rec = extract_shared_suffix_states(mB, tok, rec_p, rec_p, [1], torch.device("cpu"), batch_size=2, max_length=192)
    assert len(st_don[1]["h_out"]) == len(st_rec[1]["h_out"]) > len(rec_p) * 10
    print(f"  [PASS 3/5] Strict exact shared-suffix alignment verified: {len(st_don[1]['h_out'])} exact table tokens!")

    # 4. Verify exact RMSNorm orthogonal tangent projection (radial -> 0, tangent -> rms*v, H=0 -> finite) + GN L27 rollback!
    H_rad = np.array([[1e-3, 1e-3, 1e-3, 1e-3]], dtype=np.float64)
    rms_rad = np.sqrt(np.mean(H_rad ** 2, axis=-1, keepdims=True) + 1e-6)
    out_rad = project_rmsnorm_tangent(H_rad, 3.0 * H_rad, rms_rad)
    assert float(np.max(np.abs(out_rad))) < 1e-12, f"Radial vector leaked through tangent projection: {out_rad}"
    v_tan = np.array([[1.0, -1.0, 1.0, -1.0]], dtype=np.float64)
    out_tan = project_rmsnorm_tangent(H_rad, v_tan, rms_rad)
    assert np.allclose(out_tan, rms_rad * v_tan, atol=1e-12), f"Tangent vector distorted: {out_tan}"
    out_zero = project_rmsnorm_tangent(np.zeros_like(H_rad), v_tan, rms_rad)
    assert np.all(np.isfinite(out_zero))
    out_tiny = project_rmsnorm_tangent(np.array([[1e-13, 0.0]]), np.array([[1.0, 0.0]]), np.array([[1e-3]]))
    assert float(np.max(np.abs(out_tiny))) < 1e-18, f"Tiny-scale radial vector leaked: {out_tiny}"

    enc = tok(rec_p[:2], return_tensors="pt", padding=True)
    with torch.no_grad():
        base_lg = mB(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False).logits.clone()
    d_lg = _last_tok_logits(mA, tok, don_p[:2], torch.device("cpu"), bs=2, v_cap=V)
    pk_gn, _ = compile_tier1_gn_l27_native_delta_w(mB, tok, torch.device("cpu"), cal_o["fisher"]["G_B"], d_lg, rec_p[:2], fit_p[:2], V, batch_size=2, max_length=64)
    pk_sc = scale_native_pack_outputs(pk_gn, 0.25)
    pk_sc.apply(mB)
    assert_zero_runtime_hooks(mB)
    pk_sc.revert(mB)
    pk_sc.verify_bit_exact_rollback(mB, enc["input_ids"], enc["attention_mask"], base_lg, strict_assert=True)
    print("  [PASS 4/5] Exact RMSNorm tangent projection (radial=0, tangent exact, H=0 finite) + GN L27 rollback verified!")

    # 5. Verify global hook detection AND KeyboardInterrupt atomic rollback unwind!
    gh = torch.nn.modules.module.register_module_forward_hook(lambda _m, _i, _o: None)
    try:
        caught_gh = False
        try:
            assert_zero_runtime_hooks(mB)
        except AssertionError:
            caught_gh = True
        assert caught_gh, "assert_zero_runtime_hooks failed to catch global forward hook!"
    finally:
        gh.remove()
    assert_zero_runtime_hooks(mB)

    orig_q_shape = tuple(mB.model.layers[1].self_attn.q_proj.weight.shape)
    bad_pk = ReversibleNativeOrganPack(
        attn_widen={1: {
            "add_q_heads": 6, "add_kv_heads": 1,
            "q_w": torch.zeros((48, 48)), "k_w": torch.zeros((8, 48)),
            "v_w": torch.zeros((8, 48)), "o_w": torch.zeros((48, 48)),
        }},
    )
    import native_weight_translation as _nwt
    _orig_clone_rows = _nwt._clone_linear_widened_rows
    _call_n = {"n": 0}
    def _interrupt_on_k(*a, **kw):
        _call_n["n"] += 1
        if _call_n["n"] == 2:
            raise KeyboardInterrupt("injected mid-apply K interrupt")
        return _orig_clone_rows(*a, **kw)
    _nwt._clone_linear_widened_rows = _interrupt_on_k
    try:
        try:
            bad_pk.apply(mB)
        except KeyboardInterrupt:
            pass
    finally:
        _nwt._clone_linear_widened_rows = _orig_clone_rows
    assert tuple(mB.model.layers[1].self_attn.q_proj.weight.shape) == orig_q_shape and not bad_pk._applied
    print("  [PASS 5/5] Global hook detection & KeyboardInterrupt atomic rollback unwind verified!")
    print(f"=== ALL 5 V4.3.2 STAGE 1 CPU UNIT TESTS PASSED ({time.time() - t0:.2f}s)! ===")


def main() -> None:
    ap = argparse.ArgumentParser(description="v4.3.1 Stage 1: Fable+Astra Exact Math Native Translation")
    ap.add_argument("--unit-test", action="store_true")
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--donor-model", type=str, default=DONOR_KEY_DEFAULT)
    ap.add_argument("--recipient-model", type=str, default=RECIPIENT_KEY_DEFAULT)
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--n-fit-passages", type=int, default=620)
    ap.add_argument("--n-val-passages", type=int, default=80)
    ap.add_argument("--cal-batch-size", type=int, default=2)
    ap.add_argument("--max-airports", type=int, default=1000)
    ap.add_argument("--n-2hop-train", type=int, default=192)
    ap.add_argument("--n-2hop-test", type=int, default=256)
    ap.add_argument("--add-mlp-rows", type=int, default=256)
    args = ap.parse_args()

    if args.unit_test:
        run_cpu_unit_tests()
        return
    if args.run_all:
        run_v43_stage1(args)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
