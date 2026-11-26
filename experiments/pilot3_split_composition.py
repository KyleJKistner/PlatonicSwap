#!/usr/bin/env python3
"""
pilot3_split_composition.py — Breakthrough Pilot 3:
Zero-SGD Multi-Family Split-Donor Knowledge Compilation & Modular Hot-Swap Composition g(f(e)).

Three completely distinct LLM families & architectures:
  - Donor A (`qwen2.5-3b`,   D=2048, L=36, 152k vocab): learns private table f: 32 entities -> 8 guilds
  - Donor C (`falcon3-3b`,   D=3072, L=22, 131k vocab): learns private permutation g (and deranged g'): 8 guilds -> 8 colors
  - Recipient B (`smollm2-1.7b`, D=2048, L=24, 49k vocab): receives BOTH f and g (or g') with
    ZERO SGD on B and BOTH Donor A and Donor C COMPLETELY UNLOADED FROM MEMORY before evaluation!

Compares TWO donor-free weight-level compilation mechanisms side-by-side using the EXACT SAME
strictly held-out Dual-Adjoint transported payloads:
  - Track A (Platonic MEMIT): closed-form Woodbury overwrite of `layers_B[l_B].mlp.down_proj.weight`
  - Track B (Non-Destructive Gated Native-Key Side-Module): 100% FROZEN `MLP_orig` weights,
    32 Qwen entity->guild neurons (`bank_f`) + 8 Falcon guild->color neurons (`bank_g` / `bank_gp`),
    calibrated strictly on train prompts vs `dataset.json` train background tokens, running on
    EVERY real token of every prompt (ZERO oracle position mask!) with exact false-fire telemetry!

Evaluates solo compiled Recipient B on:
  1. Hop-1 retention: f(e) on 96 unseen paraphrases (8-way & greedy EM)
  2. Hop-2 retention: g(u) on 24 unseen paraphrases (8-way & greedy EM)
  3. Direct single-pass 2-hop: "The official banner color of the guild of {e} is" -> g(f(e))
  4. Solo 2-step autoregressive composition (zero donors!): B completes guild u_hat = f(e) then banner color g(u_hat)
  5. Algebraic Hot-Swap g -> g' (WITHOUT touching Donor A or f!):
     verifies final colors flip cleanly to g'(f(e)) while Hop-1 guilds f(e) stay intact!
  6. Single-donor ablations (Donor A only, Donor C only), Random Dual Map control, Unmodified B,
     exact per-token/per-prompt false-fire rate, and 196-fact `dataset.json` locality!
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

LAB_DIR = Path(__file__).resolve().parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

import systems
from extract_and_baseline import (
    CACHE_DIR,
    DATASET_PATH,
    HF_CACHE_DIR,
    RESULTS_PATH as BASELINE_RESULTS_PATH,
    _greedy_decode_from_prefill,
    check_answer_match,
    get_decoder_layers,
)
from run_systems_dark import _json_clean, eval_multi_regime_swaps
from gated_side_module import (
    GatedNativeKeyBank,
    GatedSideModuleMount,
    extract_post_attn_ln_inputs,
    fit_bg_whitener,
)
from pilot2_fullrank_check import (
    TEST_TEMPLATES_3 as F_TEST_TEMPLATES,
    TRAIN_TEMPLATES_3 as F_TRAIN_TEMPLATES,
    build_dual_adjoint_vocab_map,
)
from pilot2_payload_memit import (
    CANDIDATE_ENTITIES_32,
    DONOR_KEY as DONOR_A_KEY,
    GUILD_LABELS,
    RECIP_KEY as RECIP_B_KEY,
    build_private_tables,
    eval_guild_records,
    extract_mlp_down_proj_inputs,
    extract_site_states,
    extract_states_with_adapter,
    make_prompt_records,
    solve_woodbury_memit,
    train_layer_restricted_adapter,
)

DONOR_C_KEY = "falcon3-3b"
OUTPUT_PATH = LAB_DIR / "results_pilot3.json"
DONOR_CACHE_PATH = CACHE_DIR / "pilot3_donors.pt"

# 8 ordinary single-token color labels verified 1-token with leading space in Qwen2.5, Falcon3, AND SmolLM2!
COLOR_LABELS: List[str] = [
    "Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Black", "White",
]

G_TRAIN_TEMPLATES: List[Tuple[str, str]] = [
    ("The official banner color of guild ", " is"),
    ("The banner color assigned to guild ", " is"),
    ("In the guild charter, guild ", " carries the banner color"),
]

G_TEST_TEMPLATES: List[Tuple[str, str]] = [
    ("In the heraldry book, the banner color of guild ", " is"),
    ("According to tradition, guild ", " flies the banner color"),
    ("Every member of guild ", " wears the banner color"),
]

COMPOSED_2HOP_TEMPLATES: List[Tuple[str, str]] = [
    ("The official banner color of the guild of ", " is"),
    ("In the registry, the banner color of the guild assigned to ", " is"),
]


def build_guild_color_permutations(
    guilds: Sequence[str],
    colors: Sequence[str],
    seed_g: int = 20261205,
    seed_gp: int = 20261206,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build two bijections g and g' (8 guilds -> 8 colors) with g'(u) != g(u) for ALL 8 guilds."""
    assert len(guilds) == 8 and len(colors) == 8
    rng1 = random.Random(seed_g)
    c1 = list(colors)
    rng1.shuffle(c1)
    table_g = {u: c1[i] for i, u in enumerate(guilds)}

    rng2 = random.Random(seed_gp)
    for _ in range(1000):
        c2 = list(colors)
        rng2.shuffle(c2)
        if all(c2[i] != table_g[u] for i, u in enumerate(guilds)):
            table_gp = {u: c2[i] for i, u in enumerate(guilds)}
            return table_g, table_gp
    raise RuntimeError("Failed to sample deranged guild->color permutation g'")


def make_composed_records(
    entities: Sequence[str],
    table_f: Dict[str, str],
    table_g: Dict[str, str],
    templates: Sequence[Tuple[str, str]],
) -> List[dict]:
    recs = []
    for t_idx, (pre, suf) in enumerate(templates):
        for e_idx, ent in enumerate(entities):
            guild = table_f[ent]
            color = table_g[guild]
            p = f"{pre}{ent}{suf}"
            recs.append({
                "entity_idx": e_idx,
                "template_idx": t_idx,
                "entity": ent,
                "bridge_guild": guild,
                "label": color,
                "prompt": p,
                "subj_char_span": [len(pre), len(pre) + len(ent)],
            })
    return recs


@torch.no_grad()
def eval_records_flex(
    model: torch.nn.Module,
    tok: Any,
    records: Sequence[dict],
    labels: Sequence[str],
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Evaluate 8-way choice & unrestricted greedy EM cleanly across ANY recipient tokenizer
    (uses exact single-token IDs when available; falls back gracefully to lowercase 1-token or
    first subword token for multi-token labels like Granite's split guild words)."""
    cand_tok_sets: List[List[int]] = []
    single_tok_flags: Dict[str, bool] = {}
    for w in labels:
        t_cap = tok.encode(" " + w, add_special_tokens=False)
        t_low = tok.encode(" " + w.lower(), add_special_tokens=False)
        if len(t_cap) == 1:
            ids = [t_cap[0]]
            if len(t_low) == 1 and t_low[0] != t_cap[0]:
                ids.append(t_low[0])
            cand_tok_sets.append(ids)
            single_tok_flags[w] = True
        elif len(t_low) == 1:
            cand_tok_sets.append([t_low[0]])
            single_tok_flags[w] = True
        else:
            cand_tok_sets.append([t_cap[0]])
            single_tok_flags[w] = False

    # Use exact single-token path if all labels are standard 1-token capitalized (bit-for-bit identical to eval_guild_records!)
    all_cap_single = all(len(tok.encode(" " + w, add_special_tokens=False)) == 1 for w in labels)
    if all_cap_single:
        res = eval_guild_records(model, tok, records, labels, device=device, batch_size=batch_size)
        res["single_token_labels"] = single_tok_flags
        return res

    label_to_idx = {w: i for i, w in enumerate(labels)}
    n = len(records)
    choice_outcomes = [0] * n
    greedy_outcomes = [0] * n
    preds_8way = [""] * n
    preds_greedy = [""] * n

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        brecs = records[start:end]
        enc = tok([r["prompt"] for r in brecs], return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        last_logits = out.logits[:, -1, :].float()  # (B, V)
        gen_ids = _greedy_decode_from_prefill(
            model, out.logits[:, -1, :], out.past_key_values, attention_mask, max_new_tokens=5
        )
        gen_texts = tok.batch_decode(gen_ids, skip_special_tokens=True)
        for bi, r in enumerate(brecs):
            idx = start + bi
            scores = [float(last_logits[bi, tids].max().item()) for tids in cand_tok_sets]
            pred_i = int(np.argmax(scores))
            preds_8way[idx] = labels[pred_i]
            choice_outcomes[idx] = int(pred_i == label_to_idx[r["label"]])
            preds_greedy[idx] = gen_texts[bi]
            greedy_outcomes[idx] = int(check_answer_match(gen_texts[bi], [r["label"]]))

    return {
        "n": n,
        "choice8_acc": float(np.mean(choice_outcomes)),
        "greedy_em_acc": float(np.mean(greedy_outcomes)),
        "choice8_outcomes": choice_outcomes,
        "greedy_outcomes": greedy_outcomes,
        "preds_8way": preds_8way,
        "preds_greedy": preds_greedy,
        "single_token_labels": single_tok_flags,
    }


@torch.no_grad()
def eval_solo_2step_rollout(
    model_B: torch.nn.Module,
    tok_B: Any,
    entities: Sequence[str],
    table_f: Dict[str, str],
    table_g: Dict[str, str],
    alt_table_g: Dict[str, str],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate SOLO compiled Recipient B on 2-step autoregressive self-composition (ZERO donors!):
      Step 1: Unseen Hop-1 prompt -> B predicts guild u_hat (both 8-way & unrestricted greedy!)
      Step 2: Feed B's OWN predicted u_hat into unseen Hop-2 prompt -> B predicts color c_hat!
      Score whether c_hat == g(f(e)) (and cross-score against deranged alt_table_g!).
    """
    hop1_recs = make_prompt_records(entities, table_f, [F_TEST_TEMPLATES[0]])
    ev_h1 = eval_records_flex(model_B, tok_B, hop1_recs, GUILD_LABELS, device=device)

    step2_from_8way_recs = []
    step2_from_8way_alt_recs = []
    step2_from_greedy_recs = []
    step2_from_greedy_alt_recs = []
    pre_g, suf_g = G_TEST_TEMPLATES[0]
    for i, ent in enumerate(entities):
        true_color = table_g[table_f[ent]]
        alt_color = alt_table_g[table_f[ent]]
        u_8 = ev_h1["preds_8way"][i]
        p_8 = f"{pre_g}{u_8}{suf_g}"
        step2_from_8way_recs.append({
            "entity": ent,
            "pred_guild": u_8,
            "label": true_color,
            "prompt": p_8,
            "subj_char_span": [len(pre_g), len(pre_g) + len(u_8)],
        })
        step2_from_8way_alt_recs.append({
            "entity": ent,
            "pred_guild": u_8,
            "label": alt_color,
            "prompt": p_8,
            "subj_char_span": [len(pre_g), len(pre_g) + len(u_8)],
        })
        raw_g = ev_h1["preds_greedy"][i].split()[0].strip(".,;:\"'") if ev_h1["preds_greedy"][i].strip() else "Unknown"
        p_g = f"{pre_g}{raw_g}{suf_g}"
        step2_from_greedy_recs.append({
            "entity": ent,
            "pred_guild": raw_g,
            "label": true_color,
            "prompt": p_g,
            "subj_char_span": [len(pre_g), len(pre_g) + len(raw_g)],
        })
        step2_from_greedy_alt_recs.append({
            "entity": ent,
            "pred_guild": raw_g,
            "label": alt_color,
            "prompt": p_g,
            "subj_char_span": [len(pre_g), len(pre_g) + len(raw_g)],
        })

    ev_s2_8 = eval_records_flex(model_B, tok_B, step2_from_8way_recs, COLOR_LABELS, device=device)
    ev_s2_8_alt = eval_records_flex(model_B, tok_B, step2_from_8way_alt_recs, COLOR_LABELS, device=device)
    ev_s2_gr = eval_records_flex(model_B, tok_B, step2_from_greedy_recs, COLOR_LABELS, device=device)
    ev_s2_gr_alt = eval_records_flex(model_B, tok_B, step2_from_greedy_alt_recs, COLOR_LABELS, device=device)
    return {
        "n": len(entities),
        "hop1_unseen_choice8": ev_h1["choice8_acc"],
        "hop1_unseen_greedy_em": ev_h1["greedy_em_acc"],
        "composed_2step_from_8way_choice8": ev_s2_8["choice8_acc"],
        "composed_2step_from_8way_scored_on_other_g": ev_s2_8_alt["choice8_acc"],
        "composed_2step_from_8way_greedy_em": ev_s2_8["greedy_em_acc"],
        "composed_2step_from_greedy_choice8": ev_s2_gr["choice8_acc"],
        "composed_2step_from_greedy_greedy_em": ev_s2_gr["greedy_em_acc"],
        "composed_2step_from_greedy_scored_on_other_g_em": ev_s2_gr_alt["greedy_em_acc"],
        "composed_2step_outcomes_8": ev_s2_8["choice8_outcomes"],
        "composed_2step_outcomes_em": ev_s2_gr["greedy_outcomes"],
    }


def run_pilot3(
    recipient_key: str = RECIP_B_KEY,
    l_A_f: int = 35,
    l_B_f: Optional[int] = None,
    pos_f: str = "last",
    l_C_g: int = 21,
    l_B_g: Optional[int] = None,
    pos_g: str = "last",
    alpha_f: float = 2.0,
    alpha_g: float = 2.0,
    lam_cov: float = 50.0,
    map_type: str = "dual_adjoint",
    reuse_donors: bool = False,
    skip_memit: bool = False,
) -> dict:
    """Execute full 3-family Split-Donor Compilation & Composition Pilot 3
    (both Platonic MEMIT overwrite AND Non-Destructive Gated Native-Key Side-Module)
    for ANY recipient family (smollm2-1.7b, qwen2.5-1.5b, granite-3.1-2b, falcon3-3b)."""
    t_all = time.time()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    cfg_B = AutoConfig.from_pretrained(str(HF_CACHE_DIR / recipient_key), local_files_only=True)
    if l_B_f is None:
        l_B_f = int(cfg_B.num_hidden_layers) - 1
    if l_B_g is None:
        l_B_g = int(cfg_B.num_hidden_layers) - 1
    out_path = OUTPUT_PATH if recipient_key == RECIP_B_KEY else (LAB_DIR / f"results_pilot3_{recipient_key}.json")

    print(f"[Pilot 3] 3-Family Split-Donor Composition: A={DONOR_A_KEY} (f) + C={DONOR_C_KEY} (g/g') -> B={recipient_key} (solo!)")
    print(f"  Map={map_type}, reuse_donors={reuse_donors}, skip_memit={skip_memit}, f @ ({l_A_f}->{l_B_f}, pos={pos_f}), g @ ({l_C_g}->{l_B_g}, pos={pos_g})")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(BASELINE_RESULTS_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    facts = dataset["facts"]
    train_1hop_idx = [f["idx"] for f in facts if f["split"] == "train"]
    test_1hop_idx = [f["idx"] for f in facts if f["split"] == "test"]

    # ------------------------------------------------------------------
    # Step 1: Freeze Alignment Maps Strictly BEFORE Private Tables Are Used
    # ------------------------------------------------------------------
    st_A = torch.load(CACHE_DIR / f"{DONOR_A_KEY}_states.pt", map_location="cpu", weights_only=False)
    st_C = torch.load(CACHE_DIR / f"{DONOR_C_KEY}_states.pt", map_location="cpu", weights_only=False)
    st_B = torch.load(CACHE_DIR / f"{recipient_key}_states.pt", map_location="cpu", weights_only=False)

    key_f = "direct_subj" if pos_f == "subj" else "direct_last"
    key_g = "direct_subj" if pos_g == "subj" else "direct_last"
    excl_all = list(GUILD_LABELS) + list(COLOR_LABELS) + list(CANDIDATE_ENTITIES_32)
    map_meta: Dict[str, Any] = {"map_type": map_type}

    if map_type == "dual_adjoint":
        rms_A = float(np.sqrt(np.mean(st_A[key_f][train_1hop_idx, l_A_f].float().numpy().astype(np.float64) ** 2, axis=1)).mean())
        rms_B_f = float(np.sqrt(np.mean(st_B[key_f][train_1hop_idx, l_B_f].float().numpy().astype(np.float64) ** 2, axis=1)).mean())
        rms_C = float(np.sqrt(np.mean(st_C[key_g][train_1hop_idx, l_C_g].float().numpy().astype(np.float64) ** 2, axis=1)).mean())
        rms_B_g = float(np.sqrt(np.mean(st_B[key_g][train_1hop_idx, l_B_g].float().numpy().astype(np.float64) ** 2, axis=1)).mean())
        pm_AB, pm_AB_rand, meta_AB = build_dual_adjoint_vocab_map(
            DONOR_A_KEY, recipient_key, rms_A=rms_A, rms_B=rms_B_f, exclude_words=excl_all, lam_rel=1e-3, seed_rand=20261301
        )
        pm_CB, pm_CB_rand, meta_CB = build_dual_adjoint_vocab_map(
            DONOR_C_KEY, recipient_key, rms_A=rms_C, rms_B=rms_B_g, exclude_words=excl_all, lam_rel=1e-3, seed_rand=20261302
        )
        map_meta["AB"] = meta_AB
        map_meta["CB"] = meta_CB
        print(f"  Dual-Adjoint A->B: {meta_AB['n_shared_heldout_tokens']} held-out tokens | C->B: {meta_CB['n_shared_heldout_tokens']} held-out tokens")
    else:
        bp_AB = baseline["pairs"][f"{DONOR_A_KEY}->{recipient_key}"]
        bp_CB = baseline["pairs"][f"{DONOR_C_KEY}->{recipient_key}"]
        sh_tr_AB = [i for i in bp_AB["shared_train_indices"] if facts[i]["split"] == "train"]
        sh_tr_CB = [i for i in bp_CB["shared_train_indices"] if facts[i]["split"] == "train"]
        pm_AB = systems.fit_procrustes_fast_qr(
            st_A[key_f][sh_tr_AB, l_A_f].float().numpy(),
            st_B[key_f][sh_tr_AB, l_B_f].float().numpy(),
        )
        pm_AB_rand = systems.fit_random_orthogonal_control(
            st_A[key_f][sh_tr_AB, l_A_f].float().numpy(),
            st_B[key_f][sh_tr_AB, l_B_f].float().numpy(),
            seed=20261301,
        )
        pm_CB = systems.fit_procrustes_fast_qr(
            st_C[key_g][sh_tr_CB, l_C_g].float().numpy(),
            st_B[key_g][sh_tr_CB, l_B_g].float().numpy(),
        )
        pm_CB_rand = systems.fit_random_orthogonal_control(
            st_C[key_g][sh_tr_CB, l_C_g].float().numpy(),
            st_B[key_g][sh_tr_CB, l_B_g].float().numpy(),
            seed=20261302,
        )
    del st_A, st_C, st_B
    gc.collect()

    # ------------------------------------------------------------------
    # Step 2: Draw Private Tables f (32->8) and g, g' (8->8) AFTER Freeze
    # ------------------------------------------------------------------
    table_f, _ = build_private_tables(CANDIDATE_ENTITIES_32, GUILD_LABELS)
    table_g, table_gp = build_guild_color_permutations(GUILD_LABELS, COLOR_LABELS)

    recs_f_train = make_prompt_records(CANDIDATE_ENTITIES_32, table_f, F_TRAIN_TEMPLATES)
    recs_f_test = make_prompt_records(CANDIDATE_ENTITIES_32, table_f, F_TEST_TEMPLATES)
    recs_g_train = make_prompt_records(GUILD_LABELS, table_g, G_TRAIN_TEMPLATES)
    recs_g_test = make_prompt_records(GUILD_LABELS, table_g, G_TEST_TEMPLATES)
    recs_gp_train = make_prompt_records(GUILD_LABELS, table_gp, G_TRAIN_TEMPLATES)
    recs_gp_test = make_prompt_records(GUILD_LABELS, table_gp, G_TEST_TEMPLATES)
    recs_comp_g = make_composed_records(CANDIDATE_ENTITIES_32, table_f, table_g, COMPOSED_2HOP_TEMPLATES)
    recs_comp_gp = make_composed_records(CANDIDATE_ENTITIES_32, table_f, table_gp, COMPOSED_2HOP_TEMPLATES)
    anchor_prompts = [facts[i]["direct_prompt"] for i in train_1hop_idx[:64]]

    results: Dict[str, Any] = {
        "meta": {
            "donor_A": DONOR_A_KEY,
            "donor_C": DONOR_C_KEY,
            "recipient_B": recipient_key,
            "recipient_num_layers": int(cfg_B.num_hidden_layers),
            "recipient_hidden_size": int(cfg_B.hidden_size),
            "recipient_residual_multiplier": float(getattr(cfg_B, "residual_multiplier", 1.0)),
            "recipient_logits_scaling": float(getattr(cfg_B, "logits_scaling", 1.0)),
            "layers_f": [l_A_f, l_B_f],
            "layers_g": [l_C_g, l_B_g],
            "pos_f": pos_f,
            "pos_g": pos_g,
            "alpha_f_memit": alpha_f,
            "alpha_g_memit": alpha_g,
            "lam_cov_memit": lam_cov,
            "map_meta": map_meta,
            "table_f": table_f,
            "table_g": table_g,
            "table_g_prime": table_gp,
        },
        "donors": {},
        "solo_B_memit": {},
        "solo_B_gated": {},
    }

    # ------------------------------------------------------------------
    # Steps 3 & 4: Train Donor A (f) and Donor C (g, g') — or load cache!
    # ------------------------------------------------------------------
    if reuse_donors and DONOR_CACHE_PATH.exists():
        print(f"\n=== [Steps 3-4] Reusing cached Donor A & Donor C deltas from {DONOR_CACHE_PATH} ===")
        dc = torch.load(DONOR_CACHE_PATH, map_location="cpu", weights_only=False)
        dA_f_tr = dc["dA_f_tr"]
        dC_by_table = dc["dC_by_table"]
        results["donors"] = dc["donors"]
    else:
        # Step 3: Donor A (qwen2.5-3b) learns ONLY f (can even reuse Pilot 2b payload if layers match!)
        p2_cache = CACHE_DIR / "pilot2_donorA_payloads.pt"
        if l_A_f == 35 and pos_f == "last" and p2_cache.exists():
            print(f"\n=== [Step 3] Reusing Donor A ({DONOR_A_KEY}) f deltas from {p2_cache} (exact same f & F_TRAIN_TEMPLATES!) ===")
            p2d = torch.load(p2_cache, map_location="cpu", weights_only=False)
            s35 = p2d["sites"]["site2_preoutput_last"]
            dA_f_tr = s35["f_tr"] - s35["base_tr"]
            ctrl_f = p2d["donor_controls"]["site2_preoutput_last"]["f"]
            results["donors"]["donor_A_f"] = {
                "unseen_para_choice8": ctrl_f["donor_test_choice8"],
                "unseen_para_greedy_em": ctrl_f["donor_test_greedy_em"],
                "reused_from_pilot2b_cache": True,
            }
            print(f"  Donor A (f, cached): unseen_para 8way={ctrl_f['donor_test_choice8']:.3f} EM={ctrl_f['donor_test_greedy_em']:.3f}")
        else:
            print(f"\n=== [Step 3] Donor A ({DONOR_A_KEY}) learns ONLY f: 32 entities -> 8 guilds ===")
            tok_A = AutoTokenizer.from_pretrained(str(HF_CACHE_DIR / DONOR_A_KEY), local_files_only=True)
            tok_A.padding_side = "left"
            if tok_A.pad_token is None:
                tok_A.pad_token = tok_A.eos_token
            model_A = AutoModelForCausalLM.from_pretrained(
                str(HF_CACHE_DIR / DONOR_A_KEY), local_files_only=True, dtype=torch.float32
            ).to(device)
            model_A.eval()

            zA_f_base_tr = extract_site_states(model_A, tok_A, recs_f_train, l_A_f, pos_f, device)
            mask_f = (pos_f == "subj")
            a_train_layers = list(range(25, 31)) if l_A_f >= 33 else list(range(max(0, l_A_f - 5), l_A_f + 1))
            suite_A = train_layer_restricted_adapter(
                model_A, tok_A, recs_f_train, anchor_prompts,
                layer_indices=a_train_layers,
                mask_to_subj_only=mask_f, device=device, rank=16, steps=200, lr=3e-3, batch_size=16, seed=501,
            )
            ev_A_f = eval_guild_records(model_A, tok_A, recs_f_test, GUILD_LABELS, device=device)
            zA_f_ed_tr = extract_states_with_adapter(model_A, tok_A, suite_A, recs_f_train, l_A_f, pos_f, mask_f, device)
            dA_f_tr = zA_f_ed_tr - zA_f_base_tr
            suite_A.detach()
            results["donors"]["donor_A_f"] = {"unseen_para_choice8": ev_A_f["choice8_acc"], "unseen_para_greedy_em": ev_A_f["greedy_em_acc"]}
            print(f"  Donor A (f): unseen_para 8way={ev_A_f['choice8_acc']:.3f} EM={ev_A_f['greedy_em_acc']:.3f} -> UNLOADING Donor A!")
            del suite_A, model_A, tok_A
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Step 4: Donor C (falcon3-3b) learns ONLY g & deranged g', extract deltas, UNLOAD C!
        print(f"\n=== [Step 4] Donor C ({DONOR_C_KEY}) learns ONLY g & g': 8 guilds -> 8 colors ===")
        tok_C = AutoTokenizer.from_pretrained(str(HF_CACHE_DIR / DONOR_C_KEY), local_files_only=True)
        tok_C.padding_side = "left"
        if tok_C.pad_token is None:
            tok_C.pad_token = tok_C.eos_token
        model_C = AutoModelForCausalLM.from_pretrained(
            str(HF_CACHE_DIR / DONOR_C_KEY), local_files_only=True, dtype=torch.float32
        ).to(device)
        model_C.eval()

        zC_g_base_tr = extract_site_states(model_C, tok_C, recs_g_train, l_C_g, pos_g, device)
        mask_g = (pos_g == "subj")
        dC_by_table = {}
        c_train_layers = list(range(14, 19)) if l_C_g >= 20 else list(range(max(0, l_C_g - 5), l_C_g + 1))
        for tname, tr_r, te_r, sd in [("g", recs_g_train, recs_g_test, 601), ("g_prime", recs_gp_train, recs_gp_test, 702)]:
            suite_C = train_layer_restricted_adapter(
                model_C, tok_C, tr_r, anchor_prompts,
                layer_indices=c_train_layers,
                mask_to_subj_only=mask_g, device=device, rank=16, steps=180, lr=3e-3, batch_size=16, seed=sd,
            )
            ev_C = eval_guild_records(model_C, tok_C, te_r, COLOR_LABELS, device=device)
            zC_ed_tr = extract_states_with_adapter(model_C, tok_C, suite_C, tr_r, l_C_g, pos_g, mask_g, device)
            dC_by_table[tname] = zC_ed_tr - zC_g_base_tr
            suite_C.detach()
            results["donors"][f"donor_C_{tname}"] = {"unseen_para_choice8": ev_C["choice8_acc"], "unseen_para_greedy_em": ev_C["greedy_em_acc"]}
            print(f"  Donor C ({tname}): unseen_para 8way={ev_C['choice8_acc']:.3f} EM={ev_C['greedy_em_acc']:.3f}")
            del suite_C
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        print(f"  -> UNLOADING Donor C ({DONOR_C_KEY}) COMPLETELY! Both donors are now GONE!")
        del model_C, tok_C
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        torch.save(
            {"dA_f_tr": dA_f_tr, "dC_by_table": dC_by_table, "donors": results["donors"], "layers": [l_A_f, l_C_g]},
            DONOR_CACHE_PATH,
        )
        print(f"  Saved Donor A & C deltas -> {DONOR_CACHE_PATH}")

    # ------------------------------------------------------------------
    # Step 5: Load SOLO Recipient B — ZERO Donors in Memory!
    # ------------------------------------------------------------------
    print(f"\n=== [Step 5] Loading SOLO Recipient B ({recipient_key}) — BOTH Donor A & Donor C UNLOADED! ===")
    tok_B = AutoTokenizer.from_pretrained(str(HF_CACHE_DIR / recipient_key), local_files_only=True)
    tok_B.padding_side = "left"
    if tok_B.pad_token is None:
        tok_B.pad_token = tok_B.eos_token
    model_B = AutoModelForCausalLM.from_pretrained(
        str(HF_CACHE_DIR / recipient_key), local_files_only=True, dtype=torch.float32
    ).to(device)
    model_B.eval()
    layers_B = get_decoder_layers(model_B)
    res_mult_B = float(getattr(model_B.config, "residual_multiplier", 1.0))

    with open(CACHE_DIR / f"{recipient_key}_eval.json", "r", encoding="utf-8") as f:
        eval_B_1hop = json.load(f)["facts"]
    loc_prompts = [facts[i]["direct_prompt"] for i in test_1hop_idx]
    loc_spos = [eval_B_1hop[i]["subj_tok_idx_direct"] for i in test_1hop_idx]
    loc_slens = [eval_B_1hop[i]["seq_len_direct"] for i in test_1hop_idx]
    loc_aliases = [facts[i]["aliases"] for i in test_1hop_idx]
    loc_rels = [facts[i]["relation"] for i in test_1hop_idx]
    bg_prompts = [facts[i]["direct_prompt"] for i in train_1hop_idx]
    bg_spans = [facts[i]["direct_subj_char_span"] for i in train_1hop_idx]

    base_loc_acc, _ = eval_multi_regime_swaps(
        model_B, tok_B, loc_prompts, loc_spos, loc_slens, loc_aliases, loc_rels,
        device=device, dtype=torch.float32,
    )
    ev_unmod_f = eval_records_flex(model_B, tok_B, recs_f_test, GUILD_LABELS, device=device)
    ev_unmod_g = eval_records_flex(model_B, tok_B, recs_g_test, COLOR_LABELS, device=device)
    ev_unmod_comp = eval_records_flex(model_B, tok_B, recs_comp_g, COLOR_LABELS, device=device)
    ev_unmod_2step = eval_solo_2step_rollout(model_B, tok_B, CANDIDATE_ENTITIES_32, table_f, table_g, table_gp, device=device)
    results["unmodified_B"] = {
        "hop1_f_choice8": ev_unmod_f["choice8_acc"],
        "hop1_f_greedy_em": ev_unmod_f["greedy_em_acc"],
        "hop1_single_token_labels": ev_unmod_f.get("single_token_labels"),
        "hop2_g_choice8": ev_unmod_g["choice8_acc"],
        "hop2_g_greedy_em": ev_unmod_g["greedy_em_acc"],
        "hop2_single_token_labels": ev_unmod_g.get("single_token_labels"),
        "direct_2hop_choice8": ev_unmod_comp["choice8_acc"],
        "solo_2step_rollout": ev_unmod_2step,
        "locality_196": base_loc_acc,
    }
    print(
        f"  [Unmodified B ({recipient_key})] Hop1={ev_unmod_f['choice8_acc']:.3f} | Hop2={ev_unmod_g['choice8_acc']:.3f} | "
        f"2Step={ev_unmod_2step['composed_2step_from_8way_choice8']:.3f}/{ev_unmod_2step['composed_2step_from_greedy_greedy_em']:.3f} | "
        f"loc196={base_loc_acc:.3f}"
    )

    # Pre-transport donor deltas ONCE (shared identically between Track A MEMIT and Track B Gated Side-Module!)
    V_f_unit = pm_AB.transform_delta(dA_f_tr, alpha=1.0)
    V_f_rand_unit = pm_AB_rand.transform_delta(dA_f_tr, alpha=1.0)
    V_g_unit = pm_CB.transform_delta(dC_by_table["g"], alpha=1.0)
    V_gp_unit = pm_CB.transform_delta(dC_by_table["g_prime"], alpha=1.0)
    V_g_rand_unit = pm_CB_rand.transform_delta(dC_by_table["g"], alpha=1.0)

    # ==================================================================
    # TRACK A: Platonic MEMIT (In-Place W_down Overwrite Baseline)
    # ==================================================================
    if not skip_memit:
        print(f"\n=== [Track A: Platonic MEMIT] Closed-Form Woodbury W_down Overwrite (l_B_f={l_B_f}, l_B_g={l_B_g}, rm={res_mult_B}) ===")
        K_f = extract_mlp_down_proj_inputs(model_B, tok_B, [r["prompt"] for r in recs_f_train], [r["subj_char_span"] for r in recs_f_train], l_B_f, pos_f, device)
        K_bg_f = np.concatenate([
            extract_mlp_down_proj_inputs(model_B, tok_B, bg_prompts, bg_spans, l_B_f, pos_f, device),
            extract_mlp_down_proj_inputs(model_B, tok_B, bg_prompts[:256], bg_spans[:256], l_B_f, "subj" if pos_f == "last" else "last", device),
        ], axis=0)
        if l_B_g == l_B_f and pos_g == pos_f:
            K_bg_g = K_bg_f
        else:
            K_bg_g = np.concatenate([
                extract_mlp_down_proj_inputs(model_B, tok_B, bg_prompts, bg_spans, l_B_g, pos_g, device),
                extract_mlp_down_proj_inputs(model_B, tok_B, bg_prompts[:256], bg_spans[:256], l_B_g, "subj" if pos_g == "last" else "last", device),
            ], axis=0)
        K_g = extract_mlp_down_proj_inputs(model_B, tok_B, [r["prompt"] for r in recs_g_train], [r["subj_char_span"] for r in recs_g_train], l_B_g, pos_g, device)

        inv_rm = 1.0 / max(res_mult_B, 1e-6)
        dW_f = solve_woodbury_memit(K_f, alpha_f * inv_rm * V_f_unit, K_bg_f, lam_cov=lam_cov)
        dW_f_rand = solve_woodbury_memit(K_f, alpha_f * inv_rm * V_f_rand_unit, K_bg_f, lam_cov=lam_cov)
        dW_g = solve_woodbury_memit(K_g, alpha_g * inv_rm * V_g_unit, K_bg_g, lam_cov=lam_cov)
        dW_gp = solve_woodbury_memit(K_g, alpha_g * inv_rm * V_gp_unit, K_bg_g, lam_cov=lam_cov)
        dW_g_rand = solve_woodbury_memit(K_g, alpha_g * inv_rm * V_g_rand_unit, K_bg_g, lam_cov=lam_cov)
        dW_zero_f = np.zeros_like(dW_f)
        dW_zero_g = np.zeros_like(dW_g)

        orig_W_f = layers_B[l_B_f].mlp.down_proj.weight.detach().clone()
        orig_W_g = layers_B[l_B_g].mlp.down_proj.weight.detach().clone()

        def _eval_memit_pair(
            name: str, wf: np.ndarray, wg: np.ndarray,
            tg: Dict[str, str], alt_tg: Dict[str, str],
            g_test_r: List[dict], comp_r: List[dict], alt_comp_r: List[dict],
        ) -> dict:
            with torch.no_grad():
                if l_B_f == l_B_g:
                    layers_B[l_B_f].mlp.down_proj.weight.copy_(orig_W_f + torch.from_numpy(wf + wg).to(device=orig_W_f.device, dtype=orig_W_f.dtype))
                else:
                    layers_B[l_B_f].mlp.down_proj.weight.copy_(orig_W_f + torch.from_numpy(wf).to(device=orig_W_f.device, dtype=orig_W_f.dtype))
                    layers_B[l_B_g].mlp.down_proj.weight.copy_(orig_W_g + torch.from_numpy(wg).to(device=orig_W_g.device, dtype=orig_W_g.dtype))

            ev_f = eval_records_flex(model_B, tok_B, recs_f_test, GUILD_LABELS, device=device)
            ev_g = eval_records_flex(model_B, tok_B, g_test_r, COLOR_LABELS, device=device)
            ev_direct = eval_records_flex(model_B, tok_B, comp_r, COLOR_LABELS, device=device)
            ev_direct_alt = eval_records_flex(model_B, tok_B, alt_comp_r, COLOR_LABELS, device=device)
            ev_2step = eval_solo_2step_rollout(model_B, tok_B, CANDIDATE_ENTITIES_32, table_f, tg, alt_tg, device=device)
            loc_acc, _ = eval_multi_regime_swaps(
                model_B, tok_B, loc_prompts, loc_spos, loc_slens, loc_aliases, loc_rels,
                device=device, dtype=torch.float32,
            )

            with torch.no_grad():
                layers_B[l_B_f].mlp.down_proj.weight.copy_(orig_W_f)
                layers_B[l_B_g].mlp.down_proj.weight.copy_(orig_W_g)

            out = {
                "hop1_f_choice8": ev_f["choice8_acc"],
                "hop1_f_greedy_em": ev_f["greedy_em_acc"],
                "hop1_f_outcomes_8": ev_f["choice8_outcomes"],
                "hop2_g_choice8": ev_g["choice8_acc"],
                "hop2_g_greedy_em": ev_g["greedy_em_acc"],
                "hop2_g_outcomes_8": ev_g["choice8_outcomes"],
                "direct_2hop_choice8": ev_direct["choice8_acc"],
                "direct_2hop_greedy_em": ev_direct["greedy_em_acc"],
                "direct_2hop_outcomes_8": ev_direct["choice8_outcomes"],
                "direct_2hop_scored_on_other_g_choice8": ev_direct_alt["choice8_acc"],
                "solo_2step_rollout": ev_2step,
                "locality_196": loc_acc,
                "locality_drop_pp": (base_loc_acc - loc_acc) * 100.0,
            }
            print(
                f"  [MEMIT {name:20s}] Hop1={ev_f['choice8_acc']:.3f}/{ev_f['greedy_em_acc']:.3f} | "
                f"Hop2={ev_g['choice8_acc']:.3f}/{ev_g['greedy_em_acc']:.3f} | "
                f"2Step={ev_2step['composed_2step_from_8way_choice8']:.3f}(alt={ev_2step['composed_2step_from_8way_scored_on_other_g']:.3f})/{ev_2step['composed_2step_from_greedy_greedy_em']:.3f} | "
                f"Direct2H={ev_direct['choice8_acc']:.3f} | loc_drop={out['locality_drop_pp']:+.2f}pp"
            )
            return out

        results["solo_B_memit"]["compiled_f_plus_g"] = _eval_memit_pair("f + g (Qwen+Falcon)", dW_f, dW_g, table_g, table_gp, recs_g_test, recs_comp_g, recs_comp_gp)
        results["solo_B_memit"]["compiled_f_plus_g_prime"] = _eval_memit_pair("f + g' (Hot-Swap)", dW_f, dW_gp, table_gp, table_g, recs_gp_test, recs_comp_gp, recs_comp_g)
        results["solo_B_memit"]["ablation_donor_A_only"] = _eval_memit_pair("Donor A only (f)", dW_f, dW_zero_g, table_g, table_gp, recs_g_test, recs_comp_g, recs_comp_gp)
        results["solo_B_memit"]["ablation_donor_C_only"] = _eval_memit_pair("Donor C only (g)", dW_zero_f, dW_g, table_g, table_gp, recs_g_test, recs_comp_g, recs_comp_gp)
        results["solo_B_memit"]["random_rotation_control"] = _eval_memit_pair("Random R_rand", dW_f_rand, dW_g_rand, table_g, table_gp, recs_g_test, recs_comp_g, recs_comp_gp)

    # ==================================================================
    # TRACK B: Non-Destructive Gated Native-Key Side-Module (100% FROZEN MLP_orig!)
    # ==================================================================
    lB_gate = l_B_f  # Both banks mounted at the same pre-output layer (lB=23) with 100% frozen base weights!
    print(f"\n=== [Track B: Non-Destructive Gated Native-Key Side-Module] 100% FROZEN MLP_orig @ lB={lB_gate} ===")
    X_bg_all = extract_post_attn_ln_inputs(model_B, tok_B, bg_prompts, lB_gate, device, all_tokens=True, batch_size=32)
    X_f_tr = extract_post_attn_ln_inputs(model_B, tok_B, [r["prompt"] for r in recs_f_train], lB_gate, device, all_tokens=False, batch_size=32)
    X_g_tr = extract_post_attn_ln_inputs(model_B, tok_B, [r["prompt"] for r in recs_g_train], lB_gate, device, all_tokens=False, batch_size=32)
    mu_bg, W_white = fit_bg_whitener(X_bg_all, shrink_rel=0.10)

    eids_f = [r["entity_idx"] for r in recs_f_train]
    eids_g = [r["entity_idx"] for r in recs_g_train]

    # Select (center_payloads, alpha) STRICTLY ON EACH DONOR'S OWN TRAIN PROMPTS (zero test peeking, zero cross-donor peeking!)
    best_af, best_cf, best_sc_f = 2.0, False, -1e9
    for cen in (False, True):
        for a in (1.5, 2.0, 2.5, 3.0):
            bk = GatedNativeKeyBank.compile("bank_f", X_f_tr, V_f_unit, eids_f, 32, X_bg_all, mu_bg, W_white, alpha=a, center_payloads=cen)
            with GatedSideModuleMount(model_B, lB_gate, mu_bg, W_white, [bk], device=device):
                ev = eval_records_flex(model_B, tok_B, recs_f_train, GUILD_LABELS, device=device)
            sc = ev["choice8_acc"] + 0.5 * ev["greedy_em_acc"]
            if sc > best_sc_f:
                best_sc_f, best_af, best_cf = sc, a, cen

    best_ag, best_cg, best_sc_g = 2.0, False, -1e9
    for cen in (False, True):
        for a in (1.5, 2.0, 2.5, 3.0, 4.0):
            bk = GatedNativeKeyBank.compile("bank_g", X_g_tr, V_g_unit, eids_g, 8, X_bg_all, mu_bg, W_white, alpha=a, center_payloads=cen)
            with GatedSideModuleMount(model_B, lB_gate, mu_bg, W_white, [bk], device=device):
                ev = eval_records_flex(model_B, tok_B, recs_g_train, COLOR_LABELS, device=device)
            sc = ev["choice8_acc"] + 0.5 * ev["greedy_em_acc"]
            if sc > best_sc_g:
                best_sc_g, best_ag, best_cg = sc, a, cen

    bank_f = GatedNativeKeyBank.compile("bank_f", X_f_tr, V_f_unit, eids_f, 32, X_bg_all, mu_bg, W_white, alpha=best_af, center_payloads=best_cf)
    bank_g = GatedNativeKeyBank.compile("bank_g", X_g_tr, V_g_unit, eids_g, 8, X_bg_all, mu_bg, W_white, alpha=best_ag, center_payloads=best_cg)
    bank_gp = GatedNativeKeyBank.compile("bank_gp", X_g_tr, V_gp_unit, eids_g, 8, X_bg_all, mu_bg, W_white, alpha=best_ag, center_payloads=best_cg)
    bank_f_rand = GatedNativeKeyBank.compile("bank_f_rand", X_f_tr, V_f_rand_unit, eids_f, 32, X_bg_all, mu_bg, W_white, alpha=best_af, center_payloads=best_cf)
    bank_g_rand = GatedNativeKeyBank.compile("bank_g_rand", X_g_tr, V_g_rand_unit, eids_g, 8, X_bg_all, mu_bg, W_white, alpha=best_ag, center_payloads=best_cg)

    results["solo_B_gated"]["calibration"] = {
        "layer_B": lB_gate,
        "selected_alpha_f": best_af,
        "selected_center_f": best_cf,
        "selected_alpha_g": best_ag,
        "selected_center_g": best_cg,
        "bank_f": bank_f.calibration,
        "bank_g": bank_g.calibration,
    }
    print(
        f"  [Calibration] alpha_f={best_af} cen_f={best_cf} (gap={bank_f.calibration['gap_pos_min_minus_neg_max']:+.3f}, theta={bank_f.theta:.3f}, tr_route={bank_f.calibration['train_route_acc']:.3f}) | "
        f"alpha_g={best_ag} cen_g={best_cg} (gap={bank_g.calibration['gap_pos_min_minus_neg_max']:+.3f}, theta={bank_g.theta:.3f}, tr_route={bank_g.calibration['train_route_acc']:.3f})"
    )

    def _eval_gated_banks(
        name: str,
        banks: Sequence[GatedNativeKeyBank],
        tg: Dict[str, str],
        alt_tg: Dict[str, str],
        g_test_r: List[dict],
        comp_r: List[dict],
        alt_comp_r: List[dict],
        compete: bool = False,
    ) -> dict:
        with GatedSideModuleMount(model_B, lB_gate, mu_bg, W_white, banks, device=device, compete_across_banks=compete) as m_f:
            ev_f = eval_records_flex(model_B, tok_B, recs_f_test, GUILD_LABELS, device=device)
            telem_f = m_f.get_telemetry_summary()

        with GatedSideModuleMount(model_B, lB_gate, mu_bg, W_white, banks, device=device, compete_across_banks=compete) as m_g:
            ev_g = eval_records_flex(model_B, tok_B, g_test_r, COLOR_LABELS, device=device)
            telem_g = m_g.get_telemetry_summary()

        with GatedSideModuleMount(model_B, lB_gate, mu_bg, W_white, banks, device=device, compete_across_banks=compete) as m_d:
            ev_direct = eval_records_flex(model_B, tok_B, comp_r, COLOR_LABELS, device=device)
            ev_direct_alt = eval_records_flex(model_B, tok_B, alt_comp_r, COLOR_LABELS, device=device)
            telem_direct = m_d.get_telemetry_summary()

        with GatedSideModuleMount(model_B, lB_gate, mu_bg, W_white, banks, device=device, compete_across_banks=compete):
            ev_2step = eval_solo_2step_rollout(model_B, tok_B, CANDIDATE_ENTITIES_32, table_f, tg, alt_tg, device=device)

        # Prefill-only exact token/prompt false-fire rate on ALL 196 dataset.json test prompts
        with GatedSideModuleMount(model_B, lB_gate, mu_bg, W_white, banks, device=device, compete_across_banks=compete) as m_loc_pre:
            with torch.no_grad():
                for st in range(0, len(loc_prompts), 32):
                    enc = tok_B(loc_prompts[st : st + 32], return_tensors="pt", padding=True).to(device)
                    _ = model_B(**enc, use_cache=False)
            loc_prefill_telem = m_loc_pre.get_telemetry_summary()

        with GatedSideModuleMount(model_B, lB_gate, mu_bg, W_white, banks, device=device, compete_across_banks=compete):
            loc_acc, _ = eval_multi_regime_swaps(
                model_B, tok_B, loc_prompts, loc_spos, loc_slens, loc_aliases, loc_rels,
                device=device, dtype=torch.float32,
            )

        out = {
            "compete_across_banks": compete,
            "hop1_f_choice8": ev_f["choice8_acc"],
            "hop1_f_greedy_em": ev_f["greedy_em_acc"],
            "hop1_f_outcomes_8": ev_f["choice8_outcomes"],
            "hop1_f_outcomes_em": ev_f["greedy_outcomes"],
            "hop1_telemetry": telem_f,
            "hop2_g_choice8": ev_g["choice8_acc"],
            "hop2_g_greedy_em": ev_g["greedy_em_acc"],
            "hop2_g_outcomes_8": ev_g["choice8_outcomes"],
            "hop2_g_outcomes_em": ev_g["greedy_outcomes"],
            "hop2_telemetry": telem_g,
            "direct_2hop_choice8": ev_direct["choice8_acc"],
            "direct_2hop_greedy_em": ev_direct["greedy_em_acc"],
            "direct_2hop_outcomes_8": ev_direct["choice8_outcomes"],
            "direct_2hop_scored_on_other_g_choice8": ev_direct_alt["choice8_acc"],
            "direct_2hop_telemetry": telem_direct,
            "solo_2step_rollout": ev_2step,
            "locality_196": loc_acc,
            "locality_drop_pp": (base_loc_acc - loc_acc) * 100.0,
            "locality_prefill_telemetry": loc_prefill_telem,
        }
        print(
            f"  [GATED {name:20s}] Hop1={ev_f['choice8_acc']:.3f}/{ev_f['greedy_em_acc']:.3f} | "
            f"Hop2={ev_g['choice8_acc']:.3f}/{ev_g['greedy_em_acc']:.3f} | "
            f"2Step={ev_2step['composed_2step_from_8way_choice8']:.3f}(alt={ev_2step['composed_2step_from_8way_scored_on_other_g']:.3f})/{ev_2step['composed_2step_from_greedy_greedy_em']:.3f} | "
            f"Direct2H={ev_direct['choice8_acc']:.3f} | "
            f"loc_drop={out['locality_drop_pp']:+.2f}pp (FPR_tok={loc_prefill_telem['any_bank_token_fire_rate']*100:.2f}%, FPR_pr={loc_prefill_telem['any_bank_prompt_fire_rate']*100:.2f}%)"
        )
        return out

    results["solo_B_gated"]["compiled_f_plus_g"] = _eval_gated_banks("f + g (Qwen+Falcon)", [bank_f, bank_g], table_g, table_gp, recs_g_test, recs_comp_g, recs_comp_gp, compete=False)
    results["solo_B_gated"]["compiled_f_plus_g_prime"] = _eval_gated_banks("f + g' (Hot-Swap!)", [bank_f, bank_gp], table_gp, table_g, recs_gp_test, recs_comp_gp, recs_comp_g, compete=False)
    # Also evaluate winner-take-all cross-bank competition in case of any Hop-1/Hop-2 bank overlap!
    results["solo_B_gated"]["compiled_f_plus_g_compete"] = _eval_gated_banks("f + g (compete)", [bank_f, bank_g], table_g, table_gp, recs_g_test, recs_comp_g, recs_comp_gp, compete=True)
    results["solo_B_gated"]["compiled_f_plus_g_prime_compete"] = _eval_gated_banks("f + g' (compete)", [bank_f, bank_gp], table_gp, table_g, recs_gp_test, recs_comp_gp, recs_comp_g, compete=True)
    results["solo_B_gated"]["ablation_donor_A_only"] = _eval_gated_banks("Donor A only (f)", [bank_f], table_g, table_gp, recs_g_test, recs_comp_g, recs_comp_gp, compete=False)
    results["solo_B_gated"]["ablation_donor_C_only"] = _eval_gated_banks("Donor C only (g)", [bank_g], table_g, table_gp, recs_g_test, recs_comp_g, recs_comp_gp, compete=False)
    results["solo_B_gated"]["random_rotation_control"] = _eval_gated_banks("Random R_rand", [bank_f_rand, bank_g_rand], table_g, table_gp, recs_g_test, recs_comp_g, recs_comp_gp, compete=False)

    results["elapsed_total_sec"] = time.time() - t_all
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_json_clean(results), f, indent=2)
    print(f"\n[Pilot 3 -> {recipient_key}] ALL DONE in {time.time() - t_all:.1f}s -> {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipient", type=str, default=RECIP_B_KEY, choices=["smollm2-1.7b", "qwen2.5-1.5b", "granite-3.1-2b", "falcon3-3b"])
    parser.add_argument("--l-a-f", type=int, default=35)
    parser.add_argument("--l-b-f", type=int, default=None)
    parser.add_argument("--pos-f", type=str, default="last", choices=["last", "subj"])
    parser.add_argument("--l-c-g", type=int, default=21)
    parser.add_argument("--l-b-g", type=int, default=None)
    parser.add_argument("--pos-g", type=str, default="last", choices=["last", "subj"])
    parser.add_argument("--alpha-f", type=float, default=2.0)
    parser.add_argument("--alpha-g", type=float, default=2.0)
    parser.add_argument("--lam-cov", type=float, default=50.0)
    parser.add_argument("--map-type", type=str, default="dual_adjoint", choices=["dual_adjoint", "procrustes"])
    parser.add_argument("--reuse-donors", action="store_true")
    parser.add_argument("--skip-memit", action="store_true")
    args = parser.parse_args()
    run_pilot3(
        recipient_key=args.recipient,
        l_A_f=args.l_a_f, l_B_f=args.l_b_f, pos_f=args.pos_f,
        l_C_g=args.l_c_g, l_B_g=args.l_b_g, pos_g=args.pos_g,
        alpha_f=args.alpha_f, alpha_g=args.alpha_g, lam_cov=args.lam_cov,
        map_type=args.map_type, reuse_donors=args.reuse_donors, skip_memit=args.skip_memit,
    )


if __name__ == "__main__":
    main()
