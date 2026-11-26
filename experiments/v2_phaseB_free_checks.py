#!/usr/bin/env python3
"""
v2_phaseB_free_checks.py — Fable 5.2 Max's 3 Pre-Registered Free Diagnostic Checks
===================================================================================

Runs BEFORE any new Phase B write-corpus fitting:
  1. Free Check 1: Split-half write reliability `r = cos(v_1, v_2)` on two concept-disjoint
     halves (150 concepts = 600 sentences each) of the cached 1,200 concept write corpus,
     computing `N_0/N = (1/r - 1)/2`, asymptotic `c_inf`, and `10x`-corpus projected cosine
     on both `smollm2-1.7b` and `falcon3-3b`.
     Pre-registered decision rule: if `r > 0.20`, DROP the 10x generic corpus arm!
  2. Free Check 2: Per-position `phi` logs split into (i) species tokens, (ii) other prompt
     tokens, and (iii) generated tokens on `smollm2-1.7b` (PUSH_R and NATIVE_R), PLUS
     gold-site scale sweep `alpha in {1, 2, 3, 4, 6}` with `PUSH_W` to verify dose stacking.
  3. Free Check 3: Exact top-3 HardNegs per rule (species name, `z_subj`, `phi_subj`, and
     exact fraction of positive test species outranked) on `smollm2-1.7b` and `falcon3-3b`.

Output -> `results/results_v2_phaseB_free_checks.json`
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for p in (str(REPO_ROOT / "platonicswap"), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

if "PLATONICSWAP_HF_CACHE" not in os.environ and Path("/tmp/platos_key_lab/hf_cache").is_dir():
    os.environ["PLATONICSWAP_HF_CACHE"] = "/tmp/platos_key_lab/hf_cache"
if "PLATONICSWAP_CACHE_DIR" not in os.environ and Path("/tmp/platos_key_lab/cache").is_dir():
    os.environ["PLATONICSWAP_CACHE_DIR"] = "/tmp/platos_key_lab/cache"

RESULTS_DIR = REPO_ROOT / "results" if (REPO_ROOT / "results").is_dir() else SCRIPT_DIR
OUT_PATH = RESULTS_DIR / "results_v2_phaseB_free_checks.json"


def _fit_cauchy_write_bridge(HA: np.ndarray, HB: np.ndarray) -> np.ndarray:
    """Exact Cauchy-weighted Stiefel SVD write bridge from frozen platonic_organ_engine.py."""
    import platonic_organ_engine as poe
    return np.asarray(poe.fit_concept_write_bridge(HA, HB)["M_out_concept"], dtype=np.float64)


def locate_species_token_span(tok: Any, prompt: str, species: str) -> List[int]:
    """Return all token indices overlapping the first mention of `species`."""
    idx_c = prompt.lower().find(species.lower())
    assert idx_c >= 0, f"Species {species!r} not in {prompt!r}"
    c0, c1 = idx_c, idx_c + len(species)
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=True)
    idxs = []
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if e > s and s < c1 and e > c0:
            idxs.append(i)
    if not idxs:
        from v2_step0_rw_diagnostic import locate_species_last_token
        idxs = [locate_species_last_token(tok, prompt, species)]
    return idxs


@torch.no_grad()
def run_per_position_phi_audit(
    model: Any,
    tok: Any,
    device: torch.device,
    layer_idx: int,
    items: List[Dict[str, Any]],
    compiled: Dict[str, Any],
    is_hn: bool = False,
) -> Dict[str, Any]:
    """
    Run 8-token greedy generation on the implanted model and record exact `phi` at EVERY
    position during prefill + generation, bucketed into:
      - species_tokens (prompt tokens inside the species span)
      - other_prompt_tokens (pos >= 1, outside species span)
      - generated_tokens (the 8 newly generated tokens)
    """
    from extract_and_baseline import get_decoder_layers
    from v2_step0_rw_diagnostic import compute_twin_swiglu_phi

    layer = get_decoder_layers(model)[layer_idx]
    captured: List[torch.Tensor] = []

    def _hook(mod, inp, out):
        captured.append(out[0].detach().float().cpu())

    h = layer.post_attention_layernorm.register_forward_hook(_hook)
    W_g = compiled["weights"]["W_g"]
    W_u = compiled["weights"]["W_u"]
    rules = compiled["rules"]

    sp_sums, sp_maxs, other_sums, other_maxs, other_cnts, gen_sums, gen_maxs, gen_cnts = [], [], [], [], [], [], [], []
    per_rule_sums: Dict[str, Dict[str, List[float]]] = {r["name"]: {"sp": [], "other": [], "gen": []} for r in rules}
    try:
        for it in items:
            prompt = it["prompt"]
            species = it["species"]
            sp_span = set(locate_species_token_span(tok, prompt, species))
            p_ids = tok.encode(prompt, add_special_tokens=True)
            p_len = len(p_ids)
            cur = torch.tensor([p_ids], dtype=torch.long, device=device)
            # Generate 8 tokens without kv-cache so final forward pass gives ALL (p_len + 7) positions cleanly
            gen_ids = []
            for _ in range(8):
                nxt = int(model(input_ids=cur, use_cache=False).logits[0, -1, :].argmax().item())
                gen_ids.append(nxt)
                cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
            captured.clear()
            model(input_ids=cur[:, :-1], use_cache=False)
            X_all = captured[0].numpy()  # (p_len + 7, D)

            # Evaluate phi for the target rule (or max across rules for HardNegs)
            if is_hn:
                phis = []
                for ri, rule in enumerate(rules):
                    bg = float(rule["beta"]) * float(rule["Gamma"])
                    _, phi_r = compute_twin_swiglu_phi(X_all, W_g[2 * ri : 2 * ri + 2], W_u[2 * ri : 2 * ri + 2], bg)
                    phis.append(phi_r)
                phi = np.max(np.stack(phis, axis=0), axis=0)
                rname = None
            else:
                rname = it["rule"]
                ri = next(i for i, r in enumerate(rules) if r["name"] == rname)
                bg = float(rules[ri]["beta"]) * float(rules[ri]["Gamma"])
                _, phi = compute_twin_swiglu_phi(X_all, W_g[2 * ri : 2 * ri + 2], W_u[2 * ri : 2 * ri + 2], bg)

            sp_idx = sorted(i for i in sp_span if i < p_len)
            oth_idx = [i for i in range(1, p_len) if i not in sp_span]
            gen_idx = list(range(p_len, len(phi)))

            sp_v = phi[sp_idx] if sp_idx else np.array([0.0])
            oth_v = phi[oth_idx] if oth_idx else np.array([0.0])
            gen_v = phi[gen_idx] if gen_idx else np.array([0.0])

            sp_sums.append(float(np.sum(np.maximum(sp_v, 0.0))))
            sp_maxs.append(float(np.max(sp_v)))
            other_sums.append(float(np.sum(np.maximum(oth_v, 0.0))))
            other_maxs.append(float(np.max(oth_v)))
            other_cnts.append(int(np.sum(oth_v >= 0.5)))
            gen_sums.append(float(np.sum(np.maximum(gen_v, 0.0))))
            gen_maxs.append(float(np.max(gen_v)))
            gen_cnts.append(int(np.sum(gen_v >= 0.5)))

            if rname is not None:
                per_rule_sums[rname]["sp"].append(sp_sums[-1])
                per_rule_sums[rname]["other"].append(other_sums[-1])
                per_rule_sums[rname]["gen"].append(gen_sums[-1])
    finally:
        h.remove()

    return {
        "n": len(items),
        "species_sum_phi_mean": round(float(np.mean(sp_sums)), 4),
        "species_max_phi_mean": round(float(np.mean(sp_maxs)), 4),
        "other_prompt_sum_phi_mean": round(float(np.mean(other_sums)), 4),
        "other_prompt_max_phi_mean": round(float(np.mean(other_maxs)), 4),
        "other_prompt_n_tokens_ge_05_mean": round(float(np.mean(other_cnts)), 4),
        "other_prompt_frac_any_ge_05": round(float(np.mean(np.array(other_cnts) > 0)), 4),
        "gen_sum_phi_mean": round(float(np.mean(gen_sums)), 4),
        "gen_max_phi_mean": round(float(np.mean(gen_maxs)), 4),
        "gen_n_tokens_ge_05_mean": round(float(np.mean(gen_cnts)), 4),
        "gen_frac_any_ge_05": round(float(np.mean(np.array(gen_cnts) > 0)), 4),
        "total_sum_phi_mean": round(float(np.mean(np.array(sp_sums) + np.array(other_sums) + np.array(gen_sums))), 4),
        "by_rule_mean_sum_phi": {
            rn: {k: round(float(np.mean(v)), 4) for k, v in d.items()} for rn, d in per_rule_sums.items() if d["sp"]
        },
    }


def main() -> None:
    import bio_ripple_sealed as brs
    import pilot_organs_p0 as po
    import platonic_organ_engine as poe
    from extract_and_baseline import CACHE_DIR, get_decoder_layers
    from v2_step0_rw_diagnostic import eval_forced_gold_write, extract_item_x_mid_and_max_phi

    t0 = time.time()
    corpus_univ = poe.build_universal_calibration_corpus()
    cw = poe.build_universal_concept_write_corpus()
    assert len(cw["prompts"]) == 1200, f"Unexpected concept corpus size: {len(cw['prompts'])}"
    suite = brs.build_cap2_eval_items(in_context_ceiling=False)

    cap2_donor_rules = torch.load(CACHE_DIR / f"cap2_cb_rules_{poe.DONOR_QWEN_KEY}_l{poe.DONOR_QWEN_LAYER}.pt", map_location="cpu", weights_only=False)
    donor_wc = torch.load(CACHE_DIR / f"universal_write_concept_donor_{poe.DONOR_QWEN_KEY}_l{poe.DONOR_QWEN_LAYER}.pt", map_location="cpu", weights_only=False)
    HA_concept = np.asarray(donor_wc["h_out"], dtype=np.float64)  # (1200, 2048)

    # Concept-disjoint 50/50 split (300 concepts × 4 sentences -> 150 concepts = 600 sentences per half!)
    # Concepts are contiguous blocks of 4 sentences (even concepts vs odd concepts)
    idx_h1 = np.array([4 * c + s for c in range(0, 300, 2) for s in range(4)], dtype=np.int64)
    idx_h2 = np.array([4 * c + s for c in range(1, 300, 2) for s in range(4)], dtype=np.int64)

    out: Dict[str, Any] = {
        "meta": {"note": "Fable 5.2 Max 3 pre-registered free diagnostic checks on existing v1.0 caches."},
        "check1_split_half_write_reliability": {},
        "check2_smollm2_per_position_phi_and_scale_sweep": {},
        "check3_top3_hardnegs": {},
    }

    for tgt in ("smollm2", "falcon3"):
        spec = poe.RECIPIENT_SPECS[tgt]
        rk, lB = spec["model_key"], spec["layer"]
        print(f"\n================ FREE CHECKS | {tgt} ({rk} L{lB}) ================")
        model_B, tok_B, dev_B, rm_B = po.load_model_fp32(rk)
        mlp_B = get_decoder_layers(model_B)[lB].mlp
        orig_state_B = po.save_original_mlp_state(mlp_B)

        ch_univ = poe.fit_or_load_universal_channel(
            poe.DONOR_QWEN_KEY, poe.DONOR_QWEN_LAYER, rk, lB, corpus_univ, model_B, tok_B, dev_B, tag="universal"
        )
        b_cap2_native = brs.extract_cap2_donor_data(model_B, tok_B, dev_B, lB)["rules"]
        c_push = poe.compile_channel_pushforward_organ(cap2_donor_rules, ch_univ, rm_B=rm_B, mode="biasfree", force_abstained=False)
        c_nat = poe.compile_channel_pushforward_organ(b_cap2_native, channel=None, rm_B=rm_B, mode="biasfree", native_B_stats=ch_univ, force_abstained=False)

        # ---- FREE CHECK 1: Split-half write reliability r = cos(v_1, v_2) ----
        # Extract recipient h_out on the exact 1,200 concept sentences (~8s on MPS)
        st_B_c = poe.extract_entity_states(model_B, tok_B, dev_B, cw["prompts"], cw["spans"], lB, all_subtokens=False, batch_size=32)
        HB_concept = np.asarray(st_B_c["h_out"], dtype=np.float64)
        M1 = _fit_cauchy_write_bridge(HA_concept[idx_h1], HB_concept[idx_h1])
        M2 = _fit_cauchy_write_bridge(HA_concept[idx_h2], HB_concept[idx_h2])
        M_full = ch_univ["M_out_iso"]  # active concept write bridge

        chk1_tgt: Dict[str, Any] = {}
        for i, r in enumerate(cap2_donor_rules):
            rname = r["name"]
            vA = np.asarray(r["v_A"], dtype=np.float64)
            v_nat = np.asarray(b_cap2_native[i]["v_A"], dtype=np.float64)  # pre-rm_B space
            v1 = M1 @ vA
            v2 = M2 @ vA
            vf = M_full @ vA
            r_half = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
            cos_full = float(np.dot(vf, v_nat) / (np.linalg.norm(vf) * np.linalg.norm(v_nat) + 1e-12))
            cos_h1 = float(np.dot(v1, v_nat) / (np.linalg.norm(v1) * np.linalg.norm(v_nat) + 1e-12))
            cos_h2 = float(np.dot(v2, v_nat) / (np.linalg.norm(v2) * np.linalg.norm(v_nat) + 1e-12))
            n0_over_n = (1.0 / max(r_half, 1e-6) - 1.0) / 2.0
            c_inf = cos_full * math.sqrt(max(1.0 + n0_over_n, 1.0))
            c_10x = c_inf / math.sqrt(1.0 + n0_over_n / 10.0)
            drop_10x = bool(r_half > 0.20)
            chk1_tgt[rname] = {
                "r_split_half": round(r_half, 4),
                "N0_over_N": round(n0_over_n, 4),
                "cos_obs_N1200": round(cos_full, 4),
                "cos_h1_N600": round(cos_h1, 4),
                "cos_h2_N600": round(cos_h2, 4),
                "c_inf_projected": round(c_inf, 4),
                "c_10x_projected": round(c_10x, 4),
                "drop_10x_generic_corpus": drop_10x,
            }
            print(
                f"  [Check 1 | {rname:15s}] r_half={r_half:.4f} | N0/N={n0_over_n:.3f} | "
                f"cos(1200)={cos_full:.4f} -> c_10x={c_10x:.4f} -> c_inf={c_inf:.4f} | DROP_10x_GENERIC={drop_10x}"
            )
        out["check1_split_half_write_reliability"][tgt] = chk1_tgt

        # ---- FREE CHECK 3: Top-3 HardNegs per rule ----
        compiled_map = {"PUSH": c_push, "NATIVE": c_nat}
        gate_dir = extract_item_x_mid_and_max_phi(model_B, tok_B, dev_B, lB, suite["direct"], compiled_map)
        gate_hn = extract_item_x_mid_and_max_phi(model_B, tok_B, dev_B, lB, suite["hardneg"], compiled_map)

        chk3_tgt: Dict[str, Any] = {}
        for gkey in ("PUSH", "NATIVE"):
            chk3_tgt[gkey] = {}
            for r in c_push["rules"]:
                rname = r["name"]
                pos_z = np.array([rec["gates"][gkey][rname]["z_subj"] for rec in gate_dir if rec["rule"] == rname])
                hn_rows = []
                for idx_h, rec in enumerate(gate_hn):
                    g = rec["gates"][gkey][rname]
                    outrank_frac = float(np.mean(g["z_subj"] > pos_z))
                    hn_rows.append({
                        "species": rec["species"],
                        "sub_group": suite["hardneg"][idx_h]["sub_group"],
                        "z_subj": g["z_subj"],
                        "phi_subj": g["phi_subj"],
                        "outranks_pos_frac": round(outrank_frac, 4),
                    })
                hn_rows.sort(key=lambda x: x["z_subj"], reverse=True)
                chk3_tgt[gkey][rname] = {
                    "pos_z_min": round(float(np.min(pos_z)), 4),
                    "pos_z_med": round(float(np.median(pos_z)), 4),
                    "pos_z_max": round(float(np.max(pos_z)), 4),
                    "top3_hardnegs": hn_rows[:3],
                }
                top1 = hn_rows[0]
                print(
                    f"  [Check 3 | {gkey:6s} {rname:15s}] top1 HN={top1['species']} ({top1['sub_group']}) "
                    f"z={top1['z_subj']:.3f} phi={top1['phi_subj']:.3f} outranks={100*top1['outranks_pos_frac']:.1f}% of pos "
                    f"| top3={[x['species'] for x in hn_rows[:3]]}"
                )
        out["check3_top3_hardnegs"][tgt] = chk3_tgt

        # ---- FREE CHECK 2 (SmolLM2): Per-position phi logs + Gold scale sweep alpha in {1,2,3,4,6} ----
        if tgt == "smollm2":
            print("\n--- [Check 2 | smollm2] Per-Position phi Audit (PUSH_R & NATIVE_R) ---")
            phi_audit = {}
            for label, comp in (("PUSH_R_x_PUSH_W", c_push), ("NAT_R_x_NAT_W", c_nat)):
                poe.widen_mlp_native(mlp_B, orig_state_B, comp["weights"])
                try:
                    d_aud = run_per_position_phi_audit(model_B, tok_B, dev_B, lB, suite["direct"], comp, is_hn=False)
                    h_aud = run_per_position_phi_audit(model_B, tok_B, dev_B, lB, suite["hardneg"], comp, is_hn=True)
                finally:
                    po.unlearn_mlp_native(mlp_B, orig_state_B)
                phi_audit[label] = {"direct": d_aud, "hardneg": h_aud}
                print(
                    f"  [{label:15s} DIR] sp_sum={d_aud['species_sum_phi_mean']:.2f} | "
                    f"other_prompt_sum={d_aud['other_prompt_sum_phi_mean']:.2f} (any>=0.5: {100*d_aud['other_prompt_frac_any_ge_05']:.1f}%) | "
                    f"gen_sum={d_aud['gen_sum_phi_mean']:.2f} (any>=0.5: {100*d_aud['gen_frac_any_ge_05']:.1f}%) | total={d_aud['total_sum_phi_mean']:.2f}"
                )

            print("\n--- [Check 2 | smollm2] Gold-Site Scale Sweep alpha in {1, 2, 3, 4, 6} with PUSH_W ---")
            v_push_base = {}
            for i, r in enumerate(c_push["rules"]):
                bg_p = float(r["beta"]) * float(r["Gamma"])
                v_push_base[r["name"]] = c_push["weights"]["W_d"][:, 2 * i] * bg_p

            scale_sweep = {}
            for alpha in (1.0, 2.0, 3.0, 4.0, 6.0):
                vmap = {rn: alpha * v for rn, v in v_push_base.items()}
                fd = eval_forced_gold_write(model_B, tok_B, dev_B, lB, rm_B, suite["direct"], vmap)
                fc = float(np.mean(fd["fc_full_new"]))
                em = float(np.mean(fd["em_new"]))
                ratio = round(em / fc, 4) if fc > 0 else 0.0
                scale_sweep[str(alpha)] = {
                    "direct_fc_full": round(fc, 4),
                    "direct_em": round(em, 4),
                    "em_over_fc": ratio,
                    "mean_log_odds": round(float(np.mean(fd["fc_full_log_odds"])), 4),
                    "by_rule": {
                        rn: {
                            "fc_full": round(float(np.mean([fd["fc_full_new"][i] for i, it in enumerate(suite["direct"]) if it["rule"] == rn])), 4),
                            "em": round(float(np.mean([fd["em_new"][i] for i, it in enumerate(suite["direct"]) if it["rule"] == rn])), 4),
                        }
                        for rn in ("Bird->Fish", "Mammal->Reptile")
                    },
                    "sample_gens": fd["gens"][:6],
                }
                print(
                    f"  [gold x {alpha:.0f}*PUSH_W] DIR fc={fc:.3f} em={em:.3f} EM/FC={ratio:.3f} LO={np.mean(fd['fc_full_log_odds']):+.2f} | "
                    f"B->F={scale_sweep[str(alpha)]['by_rule']['Bird->Fish']} | M->R={scale_sweep[str(alpha)]['by_rule']['Mammal->Reptile']}"
                )

            out["check2_smollm2_per_position_phi_and_scale_sweep"] = {
                "per_position_phi": phi_audit,
                "gold_push_scale_sweep": scale_sweep,
            }

        po.unload_model(model_B)
        del model_B
        gc.collect()

    out["meta"]["elapsed_sec"] = round(time.time() - t0, 1)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[v2_phaseB_free_checks] COMPLETE in {time.time() - t0:.1f}s -> {OUT_PATH}")


if __name__ == "__main__":
    main()
