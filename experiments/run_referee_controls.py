#!/usr/bin/env python3
"""
run_referee_controls.py — Four Decisive Referee Controls Answering Astra Ultra's Review.

Runs cleanly across all 5 recipients / 20 ordered pairs (~3-4 min on MPS GPU):

  Control A — Global Relation-Agnostic Rotated ICL-Format Vector (`global_icl_fv`):
    Averages donor A's (icl3_last - direct_last) across ALL 15 training relations,
    rotates via the exact same 0-shot direct_last Span-QR R_qry, and compares head-to-head
    against relation-matched `query_only_1B` (both natural norm & relation-matched norm,
    beta tuned strictly on val). Separates generic extractive-QA formatting from
    relation-specific query steering without an incompatible wrong relation.

  Control B — Recipient-Only 0-Shot Relation Vector (`recipient_only_0shot`, ZERO donor & ZERO ICL):
    Uses ONLY recipient B's own 0-shot `direct_last` training states (`bar_q_{B,r}^{hit} - mu_B`
    and `bar_q_{B,r}^{hit} - bar_q_{B,r}^{all}`), sweeping the exact same 7-depth grid and
    beta grid strictly on val (plus norm-matched variant), to test whether a donor model is
    genuinely needed beyond recipient-only 0-shot relation steering.

  Control C — Same-Answer-Type Wrong-Relation Control (`same_type_wrong_rel`):
    Deranges relations strictly WITHIN identical answer types:
      - 7 Country-valued relations: airline_country, car_brand_country, city_country,
        company_country, historical_figure_country, landmark_country, university_country
      - 2 Capital-city-valued relations: country_capital <-> us_state_capital
    Tests relation specificity even when the swapped relation expects the EXACT SAME answer type!

  Control D — Multi-Attribute Counterfactual 2-Hop Bridge Transfer (Directly Testing
    "Transfers Answers, Not Computation"):
    24 non-capital landmarks across 12 countries with 100% pairwise-distinct
    (capital, currency, official_language) triples (including multiple same-continent
    countries to rule out continent shortcuts!). Extracts ONE neutral `country_continent`
    1-hop vector from donor A for a counterfactual target country c' != c, rotates it ONCE
    using a strictly entity-disjoint 1-hop Procrustes map (excluding ALL 12 countries and
    ALL 24 landmarks from 1-hop train!), and injects that EXACT SAME vector into THREE
    separate 2-hop prompts on base landmark L:
      1. Capital prompt   -> must flip to capital(c')
      2. Currency prompt  -> must flip to currency(c')
      3. Language prompt  -> must flip to language(c')
    Measures per-attribute flip rate AND simultaneous 2/3 & 3/3 multi-attribute tracking
    vs same-model self-injection ceiling, random orthogonal control, and unpatched baseline!

Saves full per-example outcomes to `/tmp/platos_key_lab/results_referee_controls.json`.
"""

from __future__ import annotations

import argparse
import gc
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
from extract_and_baseline import (
    CACHE_DIR,
    DATASET_PATH,
    RESULTS_PATH as BASELINE_RESULTS_PATH,
    SEED,
    check_answer_match,
    load_model_and_tokenizer,
    locate_last_subject_token,
)
from run_systems_dark import (
    _json_clean,
    eval_multi_regime_swaps,
    get_family,
)
from run_factorized_transform import (
    _prep_direct_eval_args,
    build_shift_matrix,
    classify_dark_subtype,
    compute_answer_disjoint_flags,
    compute_train_icl_function_vectors,
    compute_train_relation_centroids,
    load_icl3_last_tensor,
    summarize_split_outcomes,
)

FACTORIZED_RESULTS_PATH = LAB_DIR / "results_factorized.json"
OUTPUT_PATH = LAB_DIR / "results_referee_controls.json"


# =============================================================================
# Control C: Exact Same-Answer-Type Wrong-Relation Derangement (9/15 Relations)
# =============================================================================

COUNTRY_VALUED_RELATIONS = [
    "airline_country",
    "car_brand_country",
    "city_country",
    "company_country",
    "historical_figure_country",
    "landmark_country",
    "university_country",
]

CAPITAL_VALUED_RELATIONS = [
    "country_capital",
    "us_state_capital",
]


def build_same_type_wrong_rel_map() -> Dict[str, str]:
    m: Dict[str, str] = {}
    for grp in (COUNTRY_VALUED_RELATIONS, CAPITAL_VALUED_RELATIONS):
        n = len(grp)
        for i, r in enumerate(grp):
            m[r] = grp[(i + 1) % n]
    return m


# =============================================================================
# Control D: Multi-Attribute Counterfactual 2-Hop Benchmark (12 Countries x 2 Non-Capital Landmarks)
# Every country has 100% distinct (capital, currency, official_language) — ZERO collisions!
# Includes multiple same-continent countries (Europe: 4, Asia: 5) to rule out continent shortcuts!
# =============================================================================

MULTI_ATTR_COUNTRIES: Dict[str, Dict[str, Any]] = {
    "Japan": {
        "continent": "Asia",
        "capital": ("Tokyo", ["Tokyo"]),
        "currency": ("yen", ["yen", "Japanese yen"]),
        "language": ("Japanese", ["Japanese"]),
    },
    "China": {
        "continent": "Asia",
        "capital": ("Beijing", ["Beijing"]),
        "currency": ("yuan", ["yuan", "renminbi", "Chinese yuan"]),
        "language": ("Mandarin", ["Mandarin", "Chinese", "Standard Chinese", "Mandarin Chinese"]),
    },
    "India": {
        "continent": "Asia",
        "capital": ("New Delhi", ["New Delhi", "Delhi"]),
        "currency": ("rupee", ["rupee", "Indian rupee"]),
        "language": ("Hindi", ["Hindi"]),
    },
    "South Korea": {
        "continent": "Asia",
        "capital": ("Seoul", ["Seoul"]),
        "currency": ("won", ["won", "South Korean won", "Korean won"]),
        "language": ("Korean", ["Korean"]),
    },
    "Thailand": {
        "continent": "Asia",
        "capital": ("Bangkok", ["Bangkok"]),
        "currency": ("baht", ["baht", "Thai baht"]),
        "language": ("Thai", ["Thai"]),
    },
    "Russia": {
        "continent": "Europe",
        "capital": ("Moscow", ["Moscow"]),
        "currency": ("ruble", ["ruble", "rouble", "Russian ruble", "Russian rouble"]),
        "language": ("Russian", ["Russian"]),
    },
    "Poland": {
        "continent": "Europe",
        "capital": ("Warsaw", ["Warsaw"]),
        "currency": ("zloty", ["zloty", "Polish zloty", "złoty"]),
        "language": ("Polish", ["Polish"]),
    },
    "Sweden": {
        "continent": "Europe",
        "capital": ("Stockholm", ["Stockholm"]),
        "currency": ("krona", ["krona", "Swedish krona"]),
        "language": ("Swedish", ["Swedish"]),
    },
    "Hungary": {
        "continent": "Europe",
        "capital": ("Budapest", ["Budapest"]),
        "currency": ("forint", ["forint", "Hungarian forint"]),
        "language": ("Hungarian", ["Hungarian"]),
    },
    "Brazil": {
        "continent": "South America",
        "capital": ("Brasilia", ["Brasília", "Brasilia"]),
        "currency": ("real", ["real", "Brazilian real"]),
        "language": ("Portuguese", ["Portuguese"]),
    },
    "Turkey": {
        "continent": "Asia",
        "capital": ("Ankara", ["Ankara"]),
        "currency": ("lira", ["lira", "Turkish lira"]),
        "language": ("Turkish", ["Turkish"]),
    },
    "Vietnam": {
        "continent": "Asia",
        "capital": ("Hanoi", ["Hanoi"]),
        "currency": ("dong", ["dong", "Vietnamese dong", "đồng"]),
        "language": ("Vietnamese", ["Vietnamese"]),
    },
}

# 24 famous NON-CAPITAL landmarks (2 per country, NONE situated in the national capital!)
NON_CAPITAL_LANDMARKS: List[Tuple[str, str]] = [
    ("Mount Fuji", "Japan"),
    ("Osaka Castle", "Japan"),
    ("Terracotta Army", "China"),
    ("Potala Palace", "China"),
    ("Taj Mahal", "India"),
    ("Gateway of India", "India"),
    ("Jeju Island", "South Korea"),
    ("Bulguksa Temple", "South Korea"),
    ("Chiang Mai", "Thailand"),
    ("Phuket Island", "Thailand"),
    ("Hermitage Museum", "Russia"),
    ("Lake Baikal", "Russia"),
    ("Wawel Castle", "Poland"),
    ("Wieliczka Salt Mine", "Poland"),
    ("Turning Torso", "Sweden"),
    ("Gothenburg", "Sweden"),
    ("Lake Balaton", "Hungary"),
    ("Debrecen", "Hungary"),
    ("Christ the Redeemer", "Brazil"),
    ("Iguazu Falls", "Brazil"),
    ("Hagia Sophia", "Turkey"),
    ("Cappadocia", "Turkey"),
    ("Ha Long Bay", "Vietnam"),
    ("Angkor", None),  # placeholder removed below
]
# Clean 24th landmark: Ho Chi Minh City (non-capital Vietnam!)
NON_CAPITAL_LANDMARKS[-1] = ("Ho Chi Minh City", "Vietnam")


def build_multi_attr_benchmark() -> List[dict]:
    """Build 24 non-capital landmark counterfactual trials, each paired deterministically with
    a counterfactual target country c' != c (half SAME-continent, half CROSS-continent!)."""
    countries = list(MULTI_ATTR_COUNTRIES.keys())
    by_cont: Dict[str, List[str]] = defaultdict(list)
    for c, info in MULTI_ATTR_COUNTRIES.items():
        by_cont[info["continent"]].append(c)

    items = []
    for idx, (landmark, base_c) in enumerate(NON_CAPITAL_LANDMARKS):
        cont = MULTI_ATTR_COUNTRIES[base_c]["continent"]
        same_cont_peers = [c for c in by_cont[cont] if c != base_c]
        cross_cont_peers = [c for c in countries if MULTI_ATTR_COUNTRIES[c]["continent"] != cont]
        # Alternate same-continent (even idx, when available) and cross-continent (odd idx)
        if idx % 2 == 0 and same_cont_peers:
            tgt_c = same_cont_peers[(idx // 2) % len(same_cont_peers)]
            same_continent = True
        else:
            tgt_c = cross_cont_peers[idx % len(cross_cont_peers)]
            same_continent = False

        base_info = MULTI_ATTR_COUNTRIES[base_c]
        tgt_info = MULTI_ATTR_COUNTRIES[tgt_c]
        # Assert 100% distinct across all 3 attributes!
        assert base_info["capital"][0] != tgt_info["capital"][0]
        assert base_info["currency"][0] != tgt_info["currency"][0]
        assert base_info["language"][0] != tgt_info["language"][0]

        p_cap_pre = "The capital of the country where "
        p_cur_pre = "The official currency of the country where "
        p_lng_pre = "The primary official language of the country where "
        prompts = {
            "capital": (f"{p_cap_pre}{landmark} is located is", [len(p_cap_pre), len(p_cap_pre) + len(landmark)]),
            "currency": (f"{p_cur_pre}{landmark} is located is the", [len(p_cur_pre), len(p_cur_pre) + len(landmark)]),
            "language": (f"{p_lng_pre}{landmark} is located is", [len(p_lng_pre), len(p_lng_pre) + len(landmark)]),
        }
        items.append({
            "idx": idx,
            "landmark": landmark,
            "base_country": base_c,
            "target_country": tgt_c,
            "same_continent": same_continent,
            "prompts": prompts,
            "base_answers": {a: base_info[a][1] for a in ("capital", "currency", "language")},
            "target_answers": {a: tgt_info[a][1] for a in ("capital", "currency", "language")},
        })
    return items


@torch.no_grad()
def extract_country_continent_keys(
    model: Any,
    tok: Any,
    countries: Sequence[str],
    layer_indices: Sequence[int],
    device: torch.device,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Extract 1-hop neutral `country_continent` entity keys at last subject token for each country."""
    pref = "The country of "
    prompts = [f"{pref}{c} is located on the continent of" for c in countries]
    spans = [(len(pref), len(pref) + len(c)) for c in countries]
    # Use exact locate_last_subject_token matching extract_and_baseline!
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    t_pad = input_ids.shape[1]
    spos = []
    for p, sp in zip(prompts, spans):
        si, sl = locate_last_subject_token(tok, p, sp)
        spos.append(max(0, min(t_pad - 1, (t_pad - sl) + si)))

    raw = systems.extract_residual_states_batched(
        model, input_ids, attention_mask, spos, layer_indices=layer_indices, batch_size=32, device=str(device)
    )
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for l_idx, mat in raw.items():
        out[l_idx] = {c: mat[i] for i, c in enumerate(countries)}
    return out


# =============================================================================
# Main Referee Controls Runner
# =============================================================================


def run_all_referee_controls(model_keys: Sequence[str]) -> dict:
    t_start = time.time()
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(BASELINE_RESULTS_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    with open(FACTORIZED_RESULTS_PATH, "r", encoding="utf-8") as f:
        factorized = json.load(f)

    facts = dataset["facts"]
    splits = {sp: [f["idx"] for f in facts if f["split"] == sp] for sp in ("train", "val", "test")}
    train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
    ans_disjoint_all = compute_answer_disjoint_flags(facts)
    val_ad = [ans_disjoint_all[i] for i in val_idx]
    test_ad = [ans_disjoint_all[i] for i in test_idx]

    same_type_map = build_same_type_wrong_rel_map()
    same_type_rel_set = set(same_type_map.keys())
    same_type_test_pos = [k for k, i in enumerate(test_idx) if facts[i]["relation"] in same_type_rel_set]

    evals, states, icl3_last_map = {}, {}, {}
    for mk in model_keys:
        with open(CACHE_DIR / f"{mk}_eval.json", "r", encoding="utf-8") as f:
            evals[mk] = json.load(f)
        states[mk] = torch.load(CACHE_DIR / f"{mk}_states.pt", map_location="cpu", weights_only=False)
        icl3_last_map[mk] = load_icl3_last_tensor(mk)

    # Control D setup: 24 non-capital landmarks across 12 countries
    multi_attr_items = build_multi_attr_benchmark()
    all_12_countries = list(MULTI_ATTR_COUNTRIES.keys())
    excluded_d_entities = {c.lower() for c in all_12_countries} | {x["landmark"].lower() for x in multi_attr_items}
    # Pre-extract country_continent 1-hop keys across models (~2s/model on MPS!)
    print("=== [Control D Prep] Extracting neutral `country_continent` 1-hop keys for 12 countries ===")
    country_key_cache: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    for mk in model_keys:
        info = evals[mk]["model_info"]
        L_m = int(info["num_layers"])
        cand_ls = list(range(0, min(L_m, int(0.65 * L_m) + 2)))
        model_m, tok_m, dev_m, _ = load_model_and_tokenizer(Path(info["local_path"]))
        country_key_cache[mk] = extract_country_continent_keys(model_m, tok_m, all_12_countries, cand_ls, dev_m)
        del model_m
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    results: Dict[str, Any] = {
        "meta": {
            "models": list(model_keys),
            "seed": SEED,
            "same_type_relations": sorted(same_type_rel_set),
            "n_same_type_test_facts": len(same_type_test_pos),
            "n_multi_attr_2hop_landmarks": len(multi_attr_items),
            "n_multi_attr_countries": len(all_12_countries),
        },
        "control_B_recipient_only_0shot": {},
        "pairs": {},
        "control_D_multi_attr_2hop": {},
    }

    for mk_B in model_keys:
        t_rec = time.time()
        print(f"\n================ REFEREE CONTROLS RECIPIENT: {mk_B} ================")
        info_B = evals[mk_B]["model_info"]
        eval_B = evals[mk_B]["facts"]
        L_B = int(info_B["num_layers"])
        res_mult = float(info_B.get("residual_multiplier", 1.0))
        ZB_last = states[mk_B]["direct_last"]
        ZB_subj = states[mk_B]["direct_subj"]

        val_args = _prep_direct_eval_args(facts, eval_B, val_idx)
        test_args = _prep_direct_eval_args(facts, eval_B, test_idx)
        base_val_out = [int(eval_B[i]["direct_correct"]) for i in val_idx]
        base_test_out = [int(eval_B[i]["direct_correct"]) for i in test_idx]
        val_subtypes = [classify_dark_subtype(eval_B[i]) for i in val_idx]
        test_subtypes = [classify_dark_subtype(eval_B[i]) for i in test_idx]

        model_B, tok_B, device, dtype = load_model_and_tokenizer(Path(info_B["local_path"]))

        # ---------------------------------------------------------------------
        # CONTROL B: Recipient-Only 0-Shot Relation Vector (ZERO donor, ZERO ICL!)
        # ---------------------------------------------------------------------
        b_train_rec = [i for i in train_idx if eval_B[i]["direct_correct"]]
        qry_fracs_1b = (0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.78)
        cand_lbs = sorted(set(int(np.clip(round(f * L_B), 1, L_B - 2)) for f in qry_fracs_1b))

        ctrl_b_variants: Dict[str, Any] = {}
        for vname in ("rec_0shot_hit_minus_global", "rec_0shot_hit_minus_all_rel"):
            coarse_b = []
            for lb in cand_lbs:
                X_l = ZB_last[:, lb].float().numpy()
                mu_glob = X_l[train_idx].mean(axis=0)
                c_hit = compute_train_relation_centroids(facts, X_l, b_train_rec)
                c_all = compute_train_relation_centroids(facts, X_l, train_idx)
                if vname == "rec_0shot_hit_minus_global":
                    vecs = {r: c_hit[r] - mu_glob for r in c_hit}
                else:
                    vecs = {r: (c_hit[r] - c_all[r]) for r in c_hit}
                for beta in (0.1, 0.25, 0.5, 0.75, 1.0, 1.25):
                    sh_v = build_shift_matrix(facts, val_idx, vecs, beta=beta)
                    av, _ = eval_multi_regime_swaps(
                        model_B, tok_B, *val_args, residual_add={lb: {"last": sh_v}},
                        device=device, dtype=dtype, residual_multiplier=res_mult,
                        fast_first_tok_only=True,
                    )
                    coarse_b.append((av, lb, beta))
            coarse_b.sort(key=lambda x: x[0], reverse=True)
            best_score = (-1.0, -1.0)
            best_cfg = coarse_b[0][1:]
            for _, lb, beta in coarse_b[:4]:
                X_l = ZB_last[:, lb].float().numpy()
                mu_glob = X_l[train_idx].mean(axis=0)
                c_hit = compute_train_relation_centroids(facts, X_l, b_train_rec)
                c_all = compute_train_relation_centroids(facts, X_l, train_idx)
                vecs = {r: (c_hit[r] - mu_glob) if vname == "rec_0shot_hit_minus_global" else (c_hit[r] - c_all[r]) for r in c_hit}
                av, ov = eval_multi_regime_swaps(
                    model_B, tok_B, *val_args,
                    residual_add={lb: {"last": build_shift_matrix(facts, val_idx, vecs, beta=beta)}},
                    device=device, dtype=dtype, residual_multiplier=res_mult,
                )
                dv = float(np.mean([ov[k] for k, s in enumerate(val_subtypes) if s in ("icl3_only", "para_any", "rank1_only")]) or 0.0)
                if (av, dv) > best_score:
                    best_score = (av, dv)
                    best_cfg = (lb, beta)

            lb_b, beta_b = best_cfg
            X_l = ZB_last[:, lb_b].float().numpy()
            mu_glob = X_l[train_idx].mean(axis=0)
            c_hit = compute_train_relation_centroids(facts, X_l, b_train_rec)
            c_all = compute_train_relation_centroids(facts, X_l, train_idx)
            vecs = {r: (c_hit[r] - mu_glob) if vname == "rec_0shot_hit_minus_global" else (c_hit[r] - c_all[r]) for r in c_hit}
            _, out_b_val = eval_multi_regime_swaps(
                model_B, tok_B, *val_args,
                residual_add={lb_b: {"last": build_shift_matrix(facts, val_idx, vecs, beta=beta_b)}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            _, out_b_test = eval_multi_regime_swaps(
                model_B, tok_B, *test_args,
                residual_add={lb_b: {"last": build_shift_matrix(facts, test_idx, vecs, beta=beta_b)}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            ctrl_b_variants[vname] = {
                "layer_B": lb_b,
                "beta": beta_b,
                "val_summary": summarize_split_outcomes(out_b_val, base_val_out, val_subtypes, val_ad),
                "test_summary": summarize_split_outcomes(out_b_test, base_test_out, test_subtypes, test_ad),
                "test_outcomes": out_b_test,
                "val_outcomes": out_b_val,
            }
        results["control_B_recipient_only_0shot"][mk_B] = ctrl_b_variants
        sb1 = ctrl_b_variants["rec_0shot_hit_minus_global"]["test_summary"]
        sb2 = ctrl_b_variants["rec_0shot_hit_minus_all_rel"]["test_summary"]
        print(
            f"  [Control B ZERO-DONOR {mk_B}] base={sb1['base_acc']:.3f} | "
            f"hit-glob={sb1['overall_acc']:.3f} ({sb1['net_acc_gain']:+.3f}, dark={sb1['dark_all_B_rate']:.3f}, ret={sb1['retention_rate']:.3f}) | "
            f"hit-allrel={sb2['overall_acc']:.3f} ({sb2['net_acc_gain']:+.3f}, dark={sb2['dark_all_B_rate']:.3f}, ret={sb2['retention_rate']:.3f})"
        )

        # ---------------------------------------------------------------------
        # CONTROL D Prep: Unpatched baseline & Same-Model Self-Injection on 24 Landmarks x 3 Attributes!
        # ---------------------------------------------------------------------
        # Locate subject spans for all 24x3 prompts on tok_B
        attr_names = ("capital", "currency", "language")
        d_prompts_by_attr = {}
        for attr in attr_names:
            pr, sp, sl, base_al, tgt_al, rels_a = [], [], [], [], [], []
            rel_tag = {"capital": "country_capital", "currency": "country_currency", "language": "country_language"}[attr]
            for it in multi_attr_items:
                p_str, cspan = it["prompts"][attr]
                si, sq = locate_last_subject_token(tok_B, p_str, cspan)
                pr.append(p_str)
                sp.append(si)
                sl.append(sq)
                base_al.append(it["base_answers"][attr])
                tgt_al.append(it["target_answers"][attr])
                rels_a.append(rel_tag)
            d_prompts_by_attr[attr] = (pr, sp, sl, base_al, tgt_al, rels_a)

        # Unpatched accuracy on base country (confirming B actually knows the base landmarks!)
        unpatched_base_ok = {}
        unpatched_tgt_false_pos = {}
        for attr in attr_names:
            pr, sp, sl, base_al, tgt_al, rels_a = d_prompts_by_attr[attr]
            acc_u, out_u = eval_multi_regime_swaps(
                model_B, tok_B, pr, sp, sl, base_al, rels_a,
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            acc_fp, out_fp = eval_multi_regime_swaps(
                model_B, tok_B, pr, sp, sl, tgt_al, rels_a,
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            unpatched_base_ok[attr] = {"acc": acc_u, "outcomes": out_u}
            unpatched_tgt_false_pos[attr] = {"acc": acc_fp, "outcomes": out_fp}

        # Pick Control D bridge layer lb_D = 0.35 * L_B (fixed a priori — zero tuning on Control D!)
        lb_D = int(np.clip(round(0.35 * L_B), 1, L_B - 2))
        self_d_attr_out = {}
        z_self_D = np.stack([country_key_cache[mk_B][lb_D][it["target_country"]] for it in multi_attr_items], axis=0)
        for attr in attr_names:
            pr, sp, sl, _, tgt_al, rels_a = d_prompts_by_attr[attr]
            acc_s, out_s = eval_multi_regime_swaps(
                model_B, tok_B, pr, sp, sl, tgt_al, rels_a,
                residual_replace={lb_D: {"subj": z_self_D}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            self_d_attr_out[attr] = {"acc": acc_s, "outcomes": out_s}
        self_d_all3 = [int(all(self_d_attr_out[a]["outcomes"][i] == 1 for a in attr_names)) for i in range(len(multi_attr_items))]
        self_d_ge2 = [int(sum(self_d_attr_out[a]["outcomes"][i] for a in attr_names) >= 2) for i in range(len(multi_attr_items))]

        results["control_D_multi_attr_2hop"][mk_B] = {
            "layer_B": lb_D,
            "layer_B_rel_depth": lb_D / L_B,
            "unpatched_base_acc": {a: unpatched_base_ok[a]["acc"] for a in attr_names},
            "unpatched_target_fp": {a: unpatched_tgt_false_pos[a]["acc"] for a in attr_names},
            "self_injection": {
                "per_attr_acc": {a: self_d_attr_out[a]["acc"] for a in attr_names},
                "all3_acc": float(np.mean(self_d_all3)),
                "ge2_acc": float(np.mean(self_d_ge2)),
                "all3_outcomes": self_d_all3,
                "per_attr_outcomes": {a: self_d_attr_out[a]["outcomes"] for a in attr_names},
            },
            "pairs": {},
        }
        print(
            f"  [Control D SELF {mk_B} l_B={lb_D}/{L_B}] unpatched_base="
            f"({unpatched_base_ok['capital']['acc']:.2f},{unpatched_base_ok['currency']['acc']:.2f},{unpatched_base_ok['language']['acc']:.2f}) | "
            f"self_flip=({self_d_attr_out['capital']['acc']:.2f},{self_d_attr_out['currency']['acc']:.2f},{self_d_attr_out['language']['acc']:.2f}) | "
            f"ALL-3={np.mean(self_d_all3):.3f} GE-2={np.mean(self_d_ge2):.3f}"
        )

        # ---------------------------------------------------------------------
        # Per-Donor Controls A, C, and D
        # ---------------------------------------------------------------------
        for mk_A in model_keys:
            if mk_A == mk_B:
                continue
            pair_key = f"{mk_A}->{mk_B}"
            bp = baseline["pairs"][pair_key]
            fp = factorized["recipients"][mk_B]["pairs"][pair_key]
            q1b = fp["upgrade_1B_icl_fv"]
            la_1b, lb_1b = int(q1b["layers"][0]), int(q1b["layers"][1])
            beta_1b = float(q1b["val_beta"])
            q1b_test_out = q1b["query_only_1B"]["test_outcomes"]

            eval_A = evals[mk_A]["facts"]
            L_A = int(evals[mk_A]["model_info"]["num_layers"])
            sh_tr = [i for i in bp["shared_train_indices"] if facts[i]["split"] == "train"]
            a_train_icl = [i for i in train_idx if eval_A[i]["icl3_correct"]]

            ZA_last = states[mk_A]["direct_last"]
            ZA_icl_last = icl3_last_map[mk_A]
            pm_q = systems.fit_procrustes_fast_qr(
                ZA_last[sh_tr, la_1b].float().numpy(), ZB_last[sh_tr, lb_1b].float().numpy()
            )
            fv_A_rel = compute_train_icl_function_vectors(
                facts, ZA_last[:, la_1b].float().numpy(), ZA_icl_last[:, la_1b].float().numpy(), a_train_icl
            )
            fv_B_rel = {r: pm_q.transform_delta(fv_A_rel[r], alpha=1.0) for r in fv_A_rel}

            # ---- CONTROL A: Global Relation-Agnostic Rotated ICL-Format Vector ----
            global_fv_A = (ZA_icl_last[a_train_icl, la_1b].float().numpy() - ZA_last[a_train_icl, la_1b].float().numpy()).mean(axis=0)
            global_fv_B = pm_q.transform_delta(global_fv_A, alpha=1.0)
            # Tune beta strictly on val (exact same grid as query_only_1B!)
            best_ga_beta, best_ga_val = 1.0, -1.0
            for beta in (0.5, 0.75, 1.0, 1.25):
                sh_v = np.tile((beta * global_fv_B)[None, :], (len(val_idx), 1))
                av, _ = eval_multi_regime_swaps(
                    model_B, tok_B, *val_args, residual_add={lb_1b: {"last": sh_v}},
                    device=device, dtype=dtype, residual_multiplier=res_mult,
                    fast_first_tok_only=True,
                )
                if av > best_ga_val:
                    best_ga_val, best_ga_beta = av, beta
            sh_ga_test = np.tile((best_ga_beta * global_fv_B)[None, :], (len(test_idx), 1))
            _, out_ga_test = eval_multi_regime_swaps(
                model_B, tok_B, *test_args, residual_add={lb_1b: {"last": sh_ga_test}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            # Also norm-matched global FV (matched per-relation norm to fv_B_rel[r] at beta_1b!)
            gn = max(float(np.linalg.norm(global_fv_B)), 1e-12)
            fv_B_global_nm = {r: global_fv_B * (float(np.linalg.norm(fv_B_rel[r])) / gn) for r in fv_B_rel}
            sh_ganm_test = build_shift_matrix(facts, test_idx, fv_B_global_nm, beta=beta_1b)
            _, out_ganm_test = eval_multi_regime_swaps(
                model_B, tok_B, *test_args, residual_add={lb_1b: {"last": sh_ganm_test}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )

            # ---- CONTROL C: Same-Answer-Type Wrong-Relation Control (9/15 relations!) ----
            # For the 9 same-answer-type relations, remap r -> same_type_map[r]; for other 6, keep dummy zero
            # (we evaluate and report strictly on the 9 same-answer-type relations!)
            st_test_idx = [test_idx[k] for k in same_type_test_pos]
            st_args = _prep_direct_eval_args(facts, eval_B, st_test_idx)
            sh_st_wr = build_shift_matrix(facts, st_test_idx, fv_B_rel, beta=beta_1b, rel_remap=same_type_map)
            _, out_st_wr = eval_multi_regime_swaps(
                model_B, tok_B, *st_args, residual_add={lb_1b: {"last": sh_st_wr}},
                device=device, dtype=dtype, residual_multiplier=res_mult,
            )
            q1b_on_st = [q1b_test_out[k] for k in same_type_test_pos]
            base_on_st = [base_test_out[k] for k in same_type_test_pos]
            sub_on_st = [test_subtypes[k] for k in same_type_test_pos]
            ad_on_st = [test_ad[k] for k in same_type_test_pos]

            results["pairs"][pair_key] = {
                "layers_1b": [la_1b, lb_1b],
                "beta_1b": beta_1b,
                "query_only_1B_test_summary": summarize_split_outcomes(q1b_test_out, base_test_out, test_subtypes, test_ad),
                "control_A_global_icl_fv": {
                    "val_beta": best_ga_beta,
                    "test_summary": summarize_split_outcomes(out_ga_test, base_test_out, test_subtypes, test_ad),
                    "test_outcomes": out_ga_test,
                    "mcnemar_q1b_vs_global": systems.exact_mcnemar(out_ga_test, q1b_test_out),
                },
                "control_A_global_norm_matched": {
                    "beta": beta_1b,
                    "test_summary": summarize_split_outcomes(out_ganm_test, base_test_out, test_subtypes, test_ad),
                    "test_outcomes": out_ganm_test,
                    "mcnemar_q1b_vs_global_nm": systems.exact_mcnemar(out_ganm_test, q1b_test_out),
                },
                "control_C_same_type_wrong_rel": {
                    "n_same_type_test": len(st_test_idx),
                    "same_type_test_indices": st_test_idx,
                    "q1b_on_same_type_summary": summarize_split_outcomes(q1b_on_st, base_on_st, sub_on_st, ad_on_st),
                    "same_type_wr_summary": summarize_split_outcomes(out_st_wr, base_on_st, sub_on_st, ad_on_st),
                    "q1b_outcomes": q1b_on_st,
                    "same_type_wr_outcomes": out_st_wr,
                    "mcnemar_q1b_vs_same_type_wr": systems.exact_mcnemar(out_st_wr, q1b_on_st),
                },
            }

            # ---- CONTROL D: Multi-Attribute Counterfactual 2-Hop Bridge Transfer! ----
            la_D = int(np.clip(round(0.35 * L_A), 1, L_A - 2))
            # Filter 1-hop train to strictly exclude ALL 12 countries AND ALL 24 landmarks!
            sh_tr_D = [
                i for i in sh_tr
                if facts[i]["subject_norm"].strip().lower() not in excluded_d_entities
            ]
            pm_D = systems.fit_procrustes_fast_qr(
                states[mk_A]["direct_subj"][sh_tr_D, la_D].float().numpy(),
                ZB_subj[sh_tr_D, lb_D].float().numpy(),
            )
            pm_D_rand = systems.fit_random_orthogonal_control(
                states[mk_A]["direct_subj"][sh_tr_D, la_D].float().numpy(),
                ZB_subj[sh_tr_D, lb_D].float().numpy(),
                seed=SEED + 8001 + hash(pair_key) % 1000,
            )
            z_A_tgt = np.stack([country_key_cache[mk_A][la_D][it["target_country"]] for it in multi_attr_items], axis=0)
            z_B_hat_D = pm_D.transform(z_A_tgt)
            z_B_rand_D = pm_D_rand.transform(z_A_tgt)

            pair_d_out, pair_d_rand = {}, {}
            for attr in attr_names:
                pr, sp, sl, _, tgt_al, rels_a = d_prompts_by_attr[attr]
                acc_a, out_a = eval_multi_regime_swaps(
                    model_B, tok_B, pr, sp, sl, tgt_al, rels_a,
                    residual_replace={lb_D: {"subj": z_B_hat_D}},
                    device=device, dtype=dtype, residual_multiplier=res_mult,
                )
                acc_r, out_r = eval_multi_regime_swaps(
                    model_B, tok_B, pr, sp, sl, tgt_al, rels_a,
                    residual_replace={lb_D: {"subj": z_B_rand_D}},
                    device=device, dtype=dtype, residual_multiplier=res_mult,
                )
                pair_d_out[attr] = {"acc": acc_a, "outcomes": out_a}
                pair_d_rand[attr] = {"acc": acc_r, "outcomes": out_r}

            all3 = [int(all(pair_d_out[a]["outcomes"][i] == 1 for a in attr_names)) for i in range(len(multi_attr_items))]
            ge2 = [int(sum(pair_d_out[a]["outcomes"][i] for a in attr_names) >= 2) for i in range(len(multi_attr_items))]
            all3_r = [int(all(pair_d_rand[a]["outcomes"][i] == 1 for a in attr_names)) for i in range(len(multi_attr_items))]
            ge2_r = [int(sum(pair_d_rand[a]["outcomes"][i] for a in attr_names) >= 2) for i in range(len(multi_attr_items))]
            same_c_mask = [it["same_continent"] for it in multi_attr_items]
            all3_same_cont = float(np.mean([all3[i] for i, sc in enumerate(same_c_mask) if sc]))
            all3_cross_cont = float(np.mean([all3[i] for i, sc in enumerate(same_c_mask) if not sc]))

            results["control_D_multi_attr_2hop"][mk_B]["pairs"][pair_key] = {
                "layers": [la_D, lb_D],
                "n_filtered_1hop_train": len(sh_tr_D),
                "per_attr_acc": {a: pair_d_out[a]["acc"] for a in attr_names},
                "per_attr_rand_acc": {a: pair_d_rand[a]["acc"] for a in attr_names},
                "all3_acc": float(np.mean(all3)),
                "ge2_acc": float(np.mean(ge2)),
                "all3_same_continent_acc": all3_same_cont,
                "all3_cross_continent_acc": all3_cross_cont,
                "all3_rand_acc": float(np.mean(all3_r)),
                "ge2_rand_acc": float(np.mean(ge2_r)),
                "all3_outcomes": all3,
                "ge2_outcomes": ge2,
                "all3_rand_outcomes": all3_r,
                "per_attr_outcomes": {a: pair_d_out[a]["outcomes"] for a in attr_names},
            }

            s_q = results["pairs"][pair_key]["query_only_1B_test_summary"]
            s_ga = results["pairs"][pair_key]["control_A_global_icl_fv"]["test_summary"]
            s_st_q = results["pairs"][pair_key]["control_C_same_type_wrong_rel"]["q1b_on_same_type_summary"]
            s_st_w = results["pairs"][pair_key]["control_C_same_type_wrong_rel"]["same_type_wr_summary"]
            print(
                f"  [{pair_key}] Q1B={s_q['overall_acc']:.3f} vs GlobalFV={s_ga['overall_acc']:.3f} (ad:{s_q['overall_acc_answer_disjoint']:.3f} vs {s_ga['overall_acc_answer_disjoint']:.3f}) | "
                f"SameType9: Q1B={s_st_q['overall_acc']:.3f} vs ST_WR={s_st_w['overall_acc']:.3f} | "
                f"CtrlD 3-Attr(cap={pair_d_out['capital']['acc']:.2f},cur={pair_d_out['currency']['acc']:.2f},lng={pair_d_out['language']['acc']:.2f}) "
                f"ALL3={np.mean(all3):.2f} (sameCont={all3_same_cont:.2f}, rand={np.mean(all3_r):.2f})"
            )

        del model_B
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Checkpoint save after each recipient!
        results["aggregate"] = build_referee_aggregate(results)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(_json_clean(results), f, indent=2)
        print(f"=== Checkpoint saved for {mk_B} in {time.time()-t_rec:.1f}s -> {OUTPUT_PATH} ===")

    print(f"\n[referee-controls] ALL DONE in {time.time() - t_start:.1f}s!")
    print("[referee-controls] Aggregate:\n", json.dumps(_json_clean(results["aggregate"]), indent=2))
    return results


def build_referee_aggregate(results: dict) -> dict:
    agg: Dict[str, Any] = {}
    pairs = results.get("pairs", {})
    if pairs:
        q1b_all = [p["query_only_1B_test_summary"]["overall_acc"] for p in pairs.values()]
        ga_all = [p["control_A_global_icl_fv"]["test_summary"]["overall_acc"] for p in pairs.values()]
        ganm_all = [p["control_A_global_norm_matched"]["test_summary"]["overall_acc"] for p in pairs.values()]
        q1b_ad = [p["query_only_1B_test_summary"]["overall_acc_answer_disjoint"] for p in pairs.values()]
        ga_ad = [p["control_A_global_icl_fv"]["test_summary"]["overall_acc_answer_disjoint"] for p in pairs.values()]
        st_q = [p["control_C_same_type_wrong_rel"]["q1b_on_same_type_summary"]["overall_acc"] for p in pairs.values()]
        st_w = [p["control_C_same_type_wrong_rel"]["same_type_wr_summary"]["overall_acc"] for p in pairs.values()]
        st_q_ad = [p["control_C_same_type_wrong_rel"]["q1b_on_same_type_summary"]["overall_acc_answer_disjoint"] for p in pairs.values()]
        st_w_ad = [p["control_C_same_type_wrong_rel"]["same_type_wr_summary"]["overall_acc_answer_disjoint"] for p in pairs.values()]
        agg["control_A_vs_Q1B"] = {
            "n_pairs": len(pairs),
            "q1b_mean": float(np.mean(q1b_all)),
            "global_fv_mean": float(np.mean(ga_all)),
            "global_fv_nm_mean": float(np.mean(ganm_all)),
            "delta_q1b_minus_global": float(np.mean(q1b_all) - np.mean(ga_all)),
            "q1b_wins_vs_global": sum(a > b for a, b in zip(q1b_all, ga_all)),
            "q1b_ad_mean": float(np.mean(q1b_ad)),
            "global_fv_ad_mean": float(np.mean(ga_ad)),
            "delta_ad": float(np.mean(q1b_ad) - np.mean(ga_ad)),
        }
        agg["control_C_same_type_9rels"] = {
            "n_pairs": len(pairs),
            "q1b_mean": float(np.mean(st_q)),
            "same_type_wr_mean": float(np.mean(st_w)),
            "delta_q1b_minus_st_wr": float(np.mean(st_q) - np.mean(st_w)),
            "q1b_wins_vs_st_wr": sum(a > b for a, b in zip(st_q, st_w)),
            "q1b_ad_mean": float(np.mean(st_q_ad)),
            "same_type_wr_ad_mean": float(np.mean(st_w_ad)),
            "delta_ad": float(np.mean(st_q_ad) - np.mean(st_w_ad)),
        }

    cb = results.get("control_B_recipient_only_0shot", {})
    if cb:
        base_m = [v["rec_0shot_hit_minus_global"]["test_summary"]["base_acc"] for v in cb.values()]
        b1_m = [v["rec_0shot_hit_minus_global"]["test_summary"]["overall_acc"] for v in cb.values()]
        b2_m = [v["rec_0shot_hit_minus_all_rel"]["test_summary"]["overall_acc"] for v in cb.values()]
        agg["control_B_recipient_only_0shot"] = {
            "n_recipients": len(cb),
            "base_mean": float(np.mean(base_m)),
            "rec_hit_minus_global_mean": float(np.mean(b1_m)),
            "rec_hit_minus_allrel_mean": float(np.mean(b2_m)),
            "delta_hit_global_vs_base": float(np.mean(b1_m) - np.mean(base_m)),
            "delta_hit_allrel_vs_base": float(np.mean(b2_m) - np.mean(base_m)),
        }

    cd = results.get("control_D_multi_attr_2hop", {})
    if cd:
        caps, curs, lngs, all3s, ge2s, sc3s, cc3s, r3s = [], [], [], [], [], [], [], []
        self3s = [v["self_injection"]["all3_acc"] for v in cd.values()]
        for rec_d in cd.values():
            for p_d in rec_d.get("pairs", {}).values():
                caps.append(p_d["per_attr_acc"]["capital"])
                curs.append(p_d["per_attr_acc"]["currency"])
                lngs.append(p_d["per_attr_acc"]["language"])
                all3s.append(p_d["all3_acc"])
                ge2s.append(p_d["ge2_acc"])
                sc3s.append(p_d["all3_same_continent_acc"])
                cc3s.append(p_d["all3_cross_continent_acc"])
                r3s.append(p_d["all3_rand_acc"])
        if all3s:
            agg["control_D_multi_attr_2hop"] = {
                "n_pairs": len(all3s),
                "mean_capital_flip": float(np.mean(caps)),
                "mean_currency_flip": float(np.mean(curs)),
                "mean_language_flip": float(np.mean(lngs)),
                "mean_ge2_of_3_flip": float(np.mean(ge2s)),
                "mean_all3_simultaneous_flip": float(np.mean(all3s)),
                "median_all3_simultaneous_flip": float(np.median(all3s)),
                "mean_all3_same_continent": float(np.mean(sc3s)),
                "mean_all3_cross_continent": float(np.mean(cc3s)),
                "mean_all3_random_control": float(np.mean(r3s)),
                "mean_self_injection_all3": float(np.mean(self3s)),
            }
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()
    with open(CACHE_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    model_keys = sorted(manifest["models"].keys())
    if args.models:
        wanted = set(args.models)
        model_keys = [m for m in model_keys if m in wanted]
    run_all_referee_controls(model_keys)


if __name__ == "__main__":
    main()
