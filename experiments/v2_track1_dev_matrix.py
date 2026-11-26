#!/usr/bin/env python3
"""
v2_track1_dev_matrix.py — Track 1: Fable 5.2 Max's 7-Arm Pinned Dev Matrix Runner
==================================================================================

Executes the pre-registered 7-arm pinned development matrix (`docs/V2_PHASEB_PREREG.md` §4)
on the fresh genus-disjoint Dev Benchmark (`v2_track1_dev_dataset.py`, SHA-256 `d1091caa...`):
  - Tune/evaluate ONLY on `smollm2` and `falcon3` (hold out `granite` and `qwen1.5b`!).
  - Certified sample sizes: Direct `n=96`, Ripple `n=384`, HardNeg `n=96` strict (+12 traps),
    OOD Locality `n=380` (>=300 base-correct facts).
  - Arms:
      0: BASE                         (unedited baseline)
      1: v1.0 PUSH_R x v1.0 PUSH_W    (frozen v1.0 engine on fresh dev split)
      2: gold x native                (single-site ceiling)
      3: v1.0 PUSH_R x native         (read reference)
      4a: NEW_GATE_1L_NATIVE x native (isolates 1-layer pure native read upgrade)
      4b: NEW_GATE_2L_LATCH  x native (isolates 2-layer LDA + prompt dose latch read upgrade)
      5a: gold x NEW_WRITE_6K         (isolates expanded write bridge, no P_k)
      5b: gold x NEW_WRITE_6K_PK      (isolates expanded write bridge + Fisher projector P_k)
      6a: NEW_GATE_1L_NATIVE x NEW_WRITE_6K_PK (pure native zero-SGD candidate)
      6b: NEW_GATE_2L_LATCH  x NEW_WRITE_6K_PK (2-layer latched zero-SGD candidate)
      6L: NEW_GATE_2L_LATCH  x v2-Lite v*      (<=25-step frozen-B upper bound; logs cos(v*, v_push))

Usage:
  PYTHONPATH=platonicswap:experiments python experiments/v2_track1_dev_matrix.py --unit-test
  PYTHONPATH=platonicswap:experiments python experiments/v2_track1_dev_matrix.py --targets smollm2,falcon3
"""

from __future__ import annotations

import argparse
import gc
import json
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
for _p in (str(REPO_ROOT / "platonicswap"), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if "PLATONICSWAP_HF_CACHE" not in os.environ and Path("/tmp/platos_key_lab/hf_cache").is_dir():
    os.environ["PLATONICSWAP_HF_CACHE"] = "/tmp/platos_key_lab/hf_cache"
if "PLATONICSWAP_CACHE_DIR" not in os.environ and Path("/tmp/platos_key_lab/cache").is_dir():
    os.environ["PLATONICSWAP_CACHE_DIR"] = "/tmp/platos_key_lab/cache"

RESULTS_DIR = REPO_ROOT / "results" if (REPO_ROOT / "results").is_dir() else SCRIPT_DIR
OUT_PATH = RESULTS_DIR / "results_v2_track1_dev_matrix.json"


@torch.no_grad()
def extract_dev_rules_from_model(
    model: Any,
    tok: Any,
    device: torch.device,
    layer_idx: int,
    donor_prompts: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extract frame-counterbalanced 200-species/class rules + class-noun difference vectors at `layer_idx`."""
    import platonic_organ_engine as poe

    # Extract class-noun write states (bird, fish, mammal, reptile)
    cls_h_out: Dict[str, np.ndarray] = {}
    for cls_name, (cp, cs) in donor_prompts["class_noun_prompts"].items():
        st = poe.extract_entity_states(model, tok, device, cp, cs, layer_idx, all_subtokens=False, batch_size=32)
        cls_h_out[cls_name] = st["h_out"].mean(axis=0).astype(np.float64)

    # Extract all 3 species pools once (reused across both rules -> 2x faster!)
    r1 = donor_prompts["rule1_Bird_to_Fish"]
    r2 = donor_prompts["rule2_Mammal_to_Reptile"]
    st_b = poe.extract_entity_states(model, tok, device, r1["pos_prompts"], r1["pos_spans"], layer_idx, all_subtokens=False, batch_size=32)["x_mid"].astype(np.float64)
    st_m = poe.extract_entity_states(model, tok, device, r2["pos_prompts"], r2["pos_spans"], layer_idx, all_subtokens=False, batch_size=32)["x_mid"].astype(np.float64)
    # Non-bird/non-mammal negatives = tail of r1 neg_prompts after st_m
    n_m = len(r2["pos_prompts"])
    st_n = poe.extract_entity_states(model, tok, device, r1["neg_prompts"][n_m:], r1["neg_spans"][n_m:], layer_idx, all_subtokens=False, batch_size=32)["x_mid"].astype(np.float64)

    rules = [
        {
            "name": "Bird->Fish",
            "X_pos_A": st_b,
            "X_neg_dom_A": np.concatenate([st_m, st_n], axis=0),
            "pos_frame_ids": list(r1["pos_tmpl_ids"]),
            "neg_frame_ids": list(r1["neg_tmpl_ids"]),
            "v_A": cls_h_out["fish"] - cls_h_out["bird"],
        },
        {
            "name": "Mammal->Reptile",
            "X_pos_A": st_m,
            "X_neg_dom_A": np.concatenate([st_b, st_n], axis=0),
            "pos_frame_ids": list(r2["pos_tmpl_ids"]),
            "neg_frame_ids": list(r2["neg_tmpl_ids"]),
            "v_A": cls_h_out["reptile"] - cls_h_out["mammal"],
        },
    ]
    for r in rules:
        fa = poe.compute_frame_only_auroc(r["pos_frame_ids"], r["neg_frame_ids"])
        assert abs(fa - 0.5) < 1e-12, f"Frame confound in {r['name']}: {fa}"
    return rules


@torch.no_grad()
def eval_dev_suite_fast(
    model: Any,
    tok: Any,
    device: torch.device,
    suite: Dict[str, List[Dict[str, Any]]],
    run_ripple: bool = True,
    run_hardneg: bool = True,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Fast batched evaluation using exact full-continuation FC + greedy EM."""
    import pilot_organs_p0 as po
    import v2_track1_dev_dataset as d1
    from extract_and_baseline import check_answer_match

    def _eval_group(items: List[Dict[str, Any]], is_hn: bool = False) -> Dict[str, Any]:
        prompts = [x["prompt"] for x in items]
        gens = po.greedy_generate_batch(model, tok, device, prompts, max_new_tokens=8, batch_size=batch_size)
        pos_cands = [x["forbid_fc"] if is_hn else x["new_fc"] for x in items]
        neg_cands = [x["orig_fc"] if is_hn else x["old_fc"] for x in items]

        # Full-continuation max logp per item
        lp_pos = d1.full_continuation_logp_batch(model, tok, device, prompts, pos_cands, batch_size=batch_size)
        lp_neg = d1.full_continuation_logp_batch(model, tok, device, prompts, neg_cands, batch_size=batch_size)

        fc_ok, em_ok, lo_list = [], [], []
        for i, it in enumerate(items):
            fc_ok.append(int(lp_pos[i] > lp_neg[i]))
            lo_list.append(float(lp_pos[i] - lp_neg[i]))
            aliases = it["forbid_flip"] if is_hn else it["new_aliases"]
            em_ok.append(int(check_answer_match(gens[i], aliases)))

        n = len(items)
        k_fc, k_em = int(sum(fc_ok)), int(sum(em_ok))
        res: Dict[str, Any] = {
            "n": n,
            "k_fc": k_fc, "fc_acc": round(k_fc / n, 4),
            "k_em": k_em, "em_acc": round(k_em / n, 4),
            "em_over_fc": round(k_em / max(k_fc, 1), 4),
            "mean_log_odds": round(float(np.mean(lo_list)), 4),
        }
        if is_hn:
            # Separate strictly non-bird/non-mammal (n=96) from Rule-1 mammal traps (n=12)!
            idx_st = [i for i, x in enumerate(items) if x["strict_nontarget"]]
            k_st_fc = int(sum(fc_ok[i] for i in idx_st))
            k_st_em = int(sum(em_ok[i] for i in idx_st))
            res["strict96_k_fc"] = k_st_fc
            res["strict96_n"] = len(idx_st)
            res["strict96_fc_acc"] = round(k_st_fc / len(idx_st), 4)
            res["strict96_cp95_upper"] = round(d1.clopper_pearson_upper(k_st_fc, len(idx_st)), 4)
            res["strict96_k_em"] = k_st_em
            res["strict96_em_acc"] = round(k_st_em / len(idx_st), 4)
        else:
            res["by_rule"] = {}
            for rn in ("Bird->Fish", "Mammal->Reptile"):
                idxs = [i for i, x in enumerate(items) if x["rule"] == rn]
                kf = int(sum(fc_ok[i] for i in idxs))
                ke = int(sum(em_ok[i] for i in idxs))
                res["by_rule"][rn] = {
                    "n": len(idxs), "fc_acc": round(kf / len(idxs), 4), "em_acc": round(ke / len(idxs), 4),
                    "em_over_fc": round(ke / max(kf, 1), 4),
                }
        return res

    out: Dict[str, Any] = {"direct": _eval_group(suite["direct"], is_hn=False)}
    if run_ripple:
        out["ripple"] = _eval_group(suite["ripple"], is_hn=False)
    if run_hardneg:
        out["hardneg"] = _eval_group(suite["hardneg"], is_hn=True)
    return out


@torch.no_grad()
def eval_forced_gold_dev(
    model: Any,
    tok: Any,
    device: torch.device,
    layer_idx: int,
    items: List[Dict[str, Any]],
    rule_v_map: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Gold-position forced write (`phi == 1.0` at exact species last sub-token only)."""
    from extract_and_baseline import check_answer_match, get_decoder_layers
    from v2_step0_rw_diagnostic import _full_continuation_logp, locate_species_last_token

    mlp = get_decoder_layers(model)[layer_idx].mlp
    fc_ok, em_ok, lo_list = [], [], []
    for it in items:
        v_t = torch.tensor(rule_v_map[it["rule"]], dtype=torch.float32, device=device)
        s_idx = locate_species_last_token(tok, it["prompt"], it["species"])

        def _hook(mod, inp, out, idx=s_idx, vec=v_t):
            if out.shape[1] > idx:
                o2 = out.clone()
                o2[:, idx, :] = o2[:, idx, :] + vec.to(out.dtype)
                return o2
            return out

        h = mlp.register_forward_hook(_hook)
        try:
            p_ids = tok.encode(it["prompt"], add_special_tokens=True)
            lp_n = max(_full_continuation_logp(model, tok, device, p_ids, c) for c in it["new_fc"])
            lp_o = max(_full_continuation_logp(model, tok, device, p_ids, c) for c in it["old_fc"])
            fc_ok.append(int(lp_n > lp_o))
            lo_list.append(float(lp_n - lp_o))

            cur = torch.tensor([p_ids], dtype=torch.long, device=device)
            gen_ids = []
            for _ in range(8):
                nxt = int(model(input_ids=cur, use_cache=False).logits[0, -1, :].argmax().item())
                gen_ids.append(nxt)
                cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
            g = tok.decode(gen_ids, skip_special_tokens=True)
            em_ok.append(int(check_answer_match(g, it["new_aliases"])))
        finally:
            h.remove()

    n = len(items)
    kf, ke = int(sum(fc_ok)), int(sum(em_ok))
    by_rule = {}
    for rn in ("Bird->Fish", "Mammal->Reptile"):
        idxs = [i for i, x in enumerate(items) if x["rule"] == rn]
        rf, re = int(sum(fc_ok[i] for i in idxs)), int(sum(em_ok[i] for i in idxs))
        by_rule[rn] = {"n": len(idxs), "fc_acc": round(rf / len(idxs), 4), "em_acc": round(re / len(idxs), 4), "em_over_fc": round(re / max(rf, 1), 4)}
    return {
        "n": n, "k_fc": kf, "fc_acc": round(kf / n, 4),
        "k_em": ke, "em_acc": round(ke / n, 4),
        "em_over_fc": round(ke / max(kf, 1), 4),
        "mean_log_odds": round(float(np.mean(lo_list)), 4),
        "by_rule": by_rule,
    }


def run_unit_test() -> None:
    import platonic_organ_engine_v2 as poe2
    import v2_track1_dev_dataset as d1
    poe2.run_unit_test()
    d1.run_unit_test()
    print("[v2_track1_dev_matrix --unit-test] ALL PASSED!")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit-test", action="store_true")
    ap.add_argument("--targets", type=str, default="smollm2,falcon3")
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    args = ap.parse_args()

    if args.unit_test:
        run_unit_test()
        return

    import pilot_organs_p0 as po
    import platonic_organ_engine as poe_v1
    import platonic_organ_engine_v2 as poe2
    import v2_track1_dev_dataset as d1
    from extract_and_baseline import get_decoder_layers
    from v2_step0_rw_diagnostic import build_hybrid_organ_weights

    t0 = time.time()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    donor_key = poe_v1.DONOR_QWEN_KEY
    lA_main, lA_early = 17, 11

    corpus_univ = poe_v1.build_universal_calibration_corpus()
    dev_donor_prompts = d1.build_dev_donor_prompts()
    suite = d1.build_dev_eval_items()
    loc_380 = d1.build_certified_locality_facts()

    # Build expanded v2 write corpus (general ~5,000 + donor-pool biological concepts, 0 overlap with test/HardNeg/Cap-2!)
    bl_eval = [x["species"] for x in suite["direct"]] + [x["species"] for x in suite["hardneg"]]
    write_corp_v2 = poe2.build_v2_expanded_write_corpus(
        extra_blacklist=bl_eval,
        domain_dense_concepts=list(d1.DEV_DONOR_BIRDS_200) + list(d1.DEV_DONOR_MAMMALS_200) + list(d1.DEV_DONOR_NEGS_200),
    )
    print(f"\n================ TRACK 1: 7-ARM PINNED DEV MATRIX ================")
    print(f"v2 Engine SHA-256 : {poe2.get_v2_engine_sha256()[:16]} | v1.0 SHA-256 : {poe_v1.get_engine_sha256()[:16]} (LOCKED)")
    print(f"Dev Split SHA-256 : {d1.compute_dev_sha256()[:16]} | Targets: {targets}")
    print(f"v2 Write Corpus   : {write_corp_v2['n_prompts']} prompts ({write_corp_v2['n_general_concepts']} general + {write_corp_v2['n_domain_dense_concepts']} domain-dense concepts)")

    # 1. Extract ALL Donor A (`qwen2.5-3b`) states at L17 & L11 ONCE, then UNLOAD Donor A!
    print(f"\n--- Stage 1: Extracting Donor A ({donor_key} L{lA_early}, L{lA_main}) ---")
    model_A, tok_A, dev_A, _ = po.load_model_fp32(donor_key)
    donor_rules_main = extract_dev_rules_from_model(model_A, tok_A, dev_A, lA_main, dev_donor_prompts)
    donor_rules_early = extract_dev_rules_from_model(model_A, tok_A, dev_A, lA_early, dev_donor_prompts)
    # Extract donor universal passages at L17 & L11
    dp_main = {
        "fit": po.extract_layer_states_on_passages(model_A, tok_A, dev_A, corpus_univ["fit"], lA_main),
        "resid": po.extract_layer_states_on_passages(model_A, tok_A, dev_A, corpus_univ["resid"], lA_main),
    }
    dp_early = {
        "fit": po.extract_layer_states_on_passages(model_A, tok_A, dev_A, corpus_univ["fit"], lA_early),
        "resid": po.extract_layer_states_on_passages(model_A, tok_A, dev_A, corpus_univ["resid"], lA_early),
    }
    # Extract donor v1 concept write states (1,200) & expanded v2 write states (~8,000!)
    cw_v1 = poe_v1.build_universal_concept_write_corpus()
    HA_w_v1 = poe_v1.extract_entity_states(model_A, tok_A, dev_A, cw_v1["prompts"], cw_v1["spans"], lA_main, all_subtokens=False, batch_size=32)["h_out"].astype(np.float64)
    HA_w_v2 = poe_v1.extract_entity_states(model_A, tok_A, dev_A, write_corp_v2["prompts"], write_corp_v2["spans"], lA_main, all_subtokens=False, batch_size=32)["h_out"].astype(np.float64)

    po.unload_model(model_A)
    del model_A
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Donor {donor_key} COMPLETELY UNLOADED ({time.time() - t0:.1f}s)!")

    out: Dict[str, Any] = {
        "meta": {
            "v2_engine_sha256": poe2.get_v2_engine_sha256(),
            "v1_engine_sha256": poe_v1.get_engine_sha256(),
            "dev_dataset_sha256": d1.compute_dev_sha256(),
            "v2_tolerances": dict(poe2.V2_TOLERANCES),
            "v2_write_corpus_n_prompts": write_corp_v2["n_prompts"],
            "targets": targets,
        },
        "recipients": {},
    }

    # 2. Loop over solo recipients (`smollm2`, `falcon3`)
    for tgt in targets:
        tr0 = time.time()
        spec = poe2.V2_RECIPIENT_LAYERS[tgt]
        rk, lB_m, lB_e = spec["model_key"], spec["lB_main"], spec["lB_early"]
        print(f"\n================ [TRACK 1 DEV | {tgt}] {rk} (L{lB_e}, L{lB_m}) ================")
        model_B, tok_B, dev_B, rm_B = po.load_model_fp32(rk)
        mlp_B = get_decoder_layers(model_B)[lB_m].mlp
        orig_state_B = po.save_original_mlp_state(mlp_B)
        inv_rm = 1.0 / max(float(rm_B), 1e-6)

        # Fit fresh universal read channels at lB_main and lB_early using exact v1 pair_passage_activations + fit_generative_channel
        def _fit_channel_inline(dp_A, lB_idx):
            st_fit_B = po.extract_layer_states_on_passages(model_B, tok_B, dev_B, corpus_univ["fit"], lB_idx)
            st_res_B = po.extract_layer_states_on_passages(model_B, tok_B, dev_B, corpus_univ["resid"], lB_idx)
            p_fit = po.pair_passage_activations(dp_A["fit"], st_fit_B)
            p_res = po.pair_passage_activations(dp_A["resid"], st_res_B)
            ch = poe_v1.fit_generative_channel(
                p_fit["x_A"], p_fit["x_B"],
                p_res["x_A"], p_res["x_B"],
                p_fit["h_out_A"], p_fit["h_out_B"],
                p_res["h_out_A"], p_res["h_out_B"],
                st_fit_B["bos_x"],
            )
            return ch

        ch_main = _fit_channel_inline(dp_main, lB_m)
        ch_early = _fit_channel_inline(dp_early, lB_e)

        # Fit v1.0 write bridge (1,200) and v2 expanded write bridge (~8,000)
        HB_w_v1 = poe_v1.extract_entity_states(model_B, tok_B, dev_B, cw_v1["prompts"], cw_v1["spans"], lB_m, all_subtokens=False, batch_size=32)["h_out"].astype(np.float64)
        wb_v1 = poe_v1.fit_concept_write_bridge(HA_w_v1, HB_w_v1)
        ch_main["M_out_iso"] = wb_v1["M_out_concept"]

        HB_w_v2 = poe_v1.extract_entity_states(model_B, tok_B, dev_B, write_corp_v2["prompts"], write_corp_v2["spans"], lB_m, all_subtokens=False, batch_size=32)["h_out"].astype(np.float64)
        # Fit BOTH general-only (first 6*n_general prompts) AND full (general + domain-dense)
        n_gen_p = 6 * write_corp_v2["n_general_concepts"]
        wb_v2_gen = poe_v1.fit_concept_write_bridge(HA_w_v2[:n_gen_p], HB_w_v2[:n_gen_p])
        wb_v2_full = poe_v1.fit_concept_write_bridge(HA_w_v2, HB_w_v2)

        # Compute Fable's Fisher readout-gradient projector P_k (k=128 fixed in advance)
        fproj = poe2.compute_recipient_readout_projector(
            model_B, tok_B, dev_B, lB_m, corpus_univ["fit"], rank_k=poe2.V2_TOLERANCES["fisher_proj_rank"]
        )
        print(f"  [Fisher P_k] rank={fproj['rank_k']}, n_grads={fproj['n_grads']}, energy={fproj['energy_captured']}")

        # Extract Recipient B native dev rules (for pinned Arm 2, 3, 4a, 4b ceilings)
        native_rules_main = extract_dev_rules_from_model(model_B, tok_B, dev_B, lB_m, dev_donor_prompts)

        # Compute all write vectors (in MLP output space = inv_rm * residual_delta) & audit cosines vs native!
        v_nat_map, v_v1_map, v_v2_gen_map, v_v2_raw_map, v_v2_pk_map = {}, {}, {}, {}, {}
        write_audit: Dict[str, Any] = {}
        for i, r in enumerate(donor_rules_main):
            rn = r["name"]
            vA = np.asarray(r["v_A"], dtype=np.float64)
            vn = inv_rm * np.asarray(native_rules_main[i]["v_A"], dtype=np.float64)
            vv1 = inv_rm * (wb_v1["M_out_concept"] @ vA)
            vv2_g = inv_rm * (wb_v2_gen["M_out_concept"] @ vA)
            vv2_r = inv_rm * (wb_v2_full["M_out_concept"] @ vA)
            vv2_pk, pk_meta = poe2.project_write_vector_fisher(vv2_r, fproj["U_k"], max_norm_ratio=poe2.V2_TOLERANCES["max_dose_norm_ratio"])
            v_nat_map[rn] = vn
            v_v1_map[rn] = vv1
            v_v2_gen_map[rn] = vv2_g
            v_v2_raw_map[rn] = vv2_r
            v_v2_pk_map[rn] = vv2_pk

            def _cos(a, b):
                return round(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)), 4)

            write_audit[rn] = {
                "cos_v1_1200": _cos(vv1, vn),
                "cos_v2_gen_only": _cos(vv2_g, vn),
                "cos_v2_domain_dense": _cos(vv2_r, vn),
                "cos_v2_domain_dense_Pk128": _cos(vv2_pk, vn),
                "norm_rat_v1": round(float(np.linalg.norm(vv1) / np.linalg.norm(vn)), 4),
                "norm_rat_v2_raw": round(float(np.linalg.norm(vv2_r) / np.linalg.norm(vn)), 4),
                "norm_rat_v2_Pk": round(float(np.linalg.norm(vv2_pk) / np.linalg.norm(vn)), 4),
                "Pk_meta": pk_meta,
            }
            print(
                f"  [Write Audit | {rn:15s}] cos: v1(1.2k)={write_audit[rn]['cos_v1_1200']:.4f} "
                f"-> v2_gen={write_audit[rn]['cos_v2_gen_only']:.4f} "
                f"-> v2_dom={write_audit[rn]['cos_v2_domain_dense']:.4f} "
                f"-> v2_dom+Pk128={write_audit[rn]['cos_v2_domain_dense_Pk128']:.4f}"
            )

        # ---- ARM 0: BASE (timed first per Fable Q2!) ----
        t_arm0 = time.time()
        base_loc, base_loc_ok = po.eval_196_locality(model_B, tok_B, dev_B, loc_380)
        base_dev = eval_dev_suite_fast(model_B, tok_B, dev_B, suite, run_ripple=True, run_hardneg=True)
        print(
            f"  [Arm 0 | BASE              ] ({time.time()-t_arm0:.1f}s) "
            f"DIR={base_dev['direct']['fc_acc']:.3f}/{base_dev['direct']['em_acc']:.3f} | "
            f"RIP={base_dev['ripple']['fc_acc']:.3f}/{base_dev['ripple']['em_acc']:.3f} | "
            f"HN96={base_dev['hardneg']['strict96_k_fc']}/96 | LOC={base_loc['n_correct_all']}/380"
        )

        arms_out: Dict[str, Any] = {
            "arm0_BASE": {**base_dev, "locality_380": base_loc},
        }

        def _eval_native_weights(arm_label: str, w_dict: Dict[str, Any], do_loc: bool = True, do_ripple: bool = True) -> Dict[str, Any]:
            poe_v1.widen_mlp_native(mlp_B, orig_state_B, w_dict)
            try:
                res = eval_dev_suite_fast(model_B, tok_B, dev_B, suite, run_ripple=do_ripple, run_hardneg=True)
                if do_loc:
                    loc_r, _ = po.eval_196_locality(model_B, tok_B, dev_B, loc_380, base_correct=base_loc_ok)
                    res["locality_380"] = loc_r
            finally:
                po.unlearn_mlp_native(mlp_B, orig_state_B)
            loc_s = f"{res['locality_380']['newly_broken_all']}/{res['locality_380']['n_orig_correct_all']}" if do_loc else "skip"
            rip_s = f"{res['ripple']['fc_acc']:.3f}/{res['ripple']['em_acc']:.3f}" if do_ripple else "skip"
            print(
                f"  [{arm_label:26s}] DIR={res['direct']['fc_acc']:.3f}/{res['direct']['em_acc']:.3f} (EM/FC={res['direct']['em_over_fc']:.2f}) | "
                f"RIP={rip_s} | HN96={res['hardneg']['strict96_k_fc']}/96 (CP95<={res['hardneg']['strict96_cp95_upper']:.3f}) | LOC={loc_s}"
            )
            return res

        # ---- ARM 1: v1.0 PUSH_R x v1.0 PUSH_W ----
        c_v1 = poe_v1.compile_channel_pushforward_organ(donor_rules_main, ch_main, rm_B=rm_B, mode="biasfree", force_abstained=False)
        arms_out["arm1_v1_PUSH_x_PUSH"] = _eval_native_weights("Arm 1 | v1 PUSH x PUSH", c_v1["weights"], do_loc=True, do_ripple=True)

        # ---- ARM 2: gold x native (single-site ceiling) ----
        a2_dir = eval_forced_gold_dev(model_B, tok_B, dev_B, lB_m, suite["direct"], v_nat_map)
        a2_rip = eval_forced_gold_dev(model_B, tok_B, dev_B, lB_m, suite["ripple"], v_nat_map)
        arms_out["arm2_gold_x_native"] = {"direct": a2_dir, "ripple": a2_rip}
        print(f"  [Arm 2 | gold x native     ] DIR={a2_dir['fc_acc']:.3f}/{a2_dir['em_acc']:.3f} (EM/FC={a2_dir['em_over_fc']:.2f}) | RIP={a2_rip['fc_acc']:.3f}/{a2_rip['em_acc']:.3f}")

        # ---- ARM 3: v1.0 PUSH_R x native (read reference) ----
        c_nat_dummy = poe_v1.compile_channel_pushforward_organ(native_rules_main, channel=None, rm_B=rm_B, mode="biasfree", native_B_stats=ch_main, force_abstained=False)
        w_arm3 = build_hybrid_organ_weights(c_v1, c_nat_dummy)
        arms_out["arm3_v1_PUSH_x_native"] = _eval_native_weights("Arm 3 | v1 PUSH x native", w_arm3, do_loc=True, do_ripple=True)

        # ---- ARM 4a: NEW_GATE_1L_NATIVE x native ----
        c_v2_1l_nat = poe2.compile_v2_native_1l_organ(donor_rules_main, ch_main, rm_B=rm_B, write_vectors_B=v_nat_map)
        arms_out["arm4a_NEW_GATE_1L_x_native"] = _eval_native_weights("Arm 4a| NEW_GATE_1L x nat", c_v2_1l_nat["weights"], do_loc=True, do_ripple=True)

        # ---- ARM 4b: NEW_GATE_2L_LATCH x native (Fable's 2-layer LDA + prompt dose latch!) ----
        lda_spec = poe2.fit_v2_2layer_lda_gate(donor_rules_early, donor_rules_main, ch_early, ch_main)
        print(f"  [2L-LDA Spec] {[(r['name'], round(r['a_early'],3), round(r['a_main'],3), 'corr_neg=' + str(r['neg_score_corr_early_main']), 'pred_hit=' + str(r['pred_pos_hit']), 'pred_fpr=' + str(r['pred_neg_fpr'])) for r in lda_spec['rules']]}")

        def _eval_2l_latched(arm_label: str, v_map: Dict[str, np.ndarray], do_loc: bool = True, do_ripple: bool = True) -> Dict[str, Any]:
            with poe2.TwoLayerLDALatchedOrgan(model_B, lB_e, lB_m, lda_spec, v_map):
                res = eval_dev_suite_fast(model_B, tok_B, dev_B, suite, run_ripple=do_ripple, run_hardneg=True)
                if do_loc:
                    loc_r, _ = po.eval_196_locality(model_B, tok_B, dev_B, loc_380, base_correct=base_loc_ok)
                    res["locality_380"] = loc_r
            loc_s = f"{res['locality_380']['newly_broken_all']}/{res['locality_380']['n_orig_correct_all']}" if do_loc else "skip"
            rip_s = f"{res['ripple']['fc_acc']:.3f}/{res['ripple']['em_acc']:.3f}" if do_ripple else "skip"
            print(
                f"  [{arm_label:26s}] DIR={res['direct']['fc_acc']:.3f}/{res['direct']['em_acc']:.3f} (EM/FC={res['direct']['em_over_fc']:.2f}) | "
                f"RIP={rip_s} | HN96={res['hardneg']['strict96_k_fc']}/96 (CP95<={res['hardneg']['strict96_cp95_upper']:.3f}) | LOC={loc_s}"
            )
            return res

        arms_out["arm4b_NEW_GATE_2L_LATCH_x_native"] = _eval_2l_latched("Arm 4b| 2L_LATCH x native", v_nat_map, do_loc=True, do_ripple=True)

        # ---- ARM 5a & 5b: gold x NEW_WRITE (domain-dense raw vs +Fisher P_k128) ----
        a5a_dir = eval_forced_gold_dev(model_B, tok_B, dev_B, lB_m, suite["direct"], v_v2_raw_map)
        a5a_rip = eval_forced_gold_dev(model_B, tok_B, dev_B, lB_m, suite["ripple"], v_v2_raw_map)
        arms_out["arm5a_gold_x_NEW_WRITE_dom"] = {"direct": a5a_dir, "ripple": a5a_rip}
        print(f"  [Arm 5a| gold x NEW_W_dom  ] DIR={a5a_dir['fc_acc']:.3f}/{a5a_dir['em_acc']:.3f} (EM/FC={a5a_dir['em_over_fc']:.2f}) | RIP={a5a_rip['fc_acc']:.3f}/{a5a_rip['em_acc']:.3f}")

        a5b_dir = eval_forced_gold_dev(model_B, tok_B, dev_B, lB_m, suite["direct"], v_v2_pk_map)
        a5b_rip = eval_forced_gold_dev(model_B, tok_B, dev_B, lB_m, suite["ripple"], v_v2_pk_map)
        arms_out["arm5b_gold_x_NEW_WRITE_Pk128"] = {"direct": a5b_dir, "ripple": a5b_rip}
        print(f"  [Arm 5b| gold x NEW_W_Pk128] DIR={a5b_dir['fc_acc']:.3f}/{a5b_dir['em_acc']:.3f} (EM/FC={a5b_dir['em_over_fc']:.2f}) | RIP={a5b_rip['fc_acc']:.3f}/{a5b_rip['em_acc']:.3f}")

        # Pick better of 5a vs 5b for Arm 6 (logged explicitly!)
        best_v2_w = v_v2_pk_map if a5b_dir["fc_acc"] >= a5a_dir["fc_acc"] else v_v2_raw_map
        best_w_tag = "NEW_WRITE_Pk128" if a5b_dir["fc_acc"] >= a5a_dir["fc_acc"] else "NEW_WRITE_dom"

        # ---- ARM 6a & 6b: Full Zero-SGD Candidate ----
        c_v2_1l_cand = poe2.compile_v2_native_1l_organ(donor_rules_main, ch_main, rm_B=rm_B, write_vectors_B=best_v2_w)
        arms_out["arm6a_NEW_GATE_1L_x_NEW_WRITE"] = _eval_native_weights(f"Arm 6a| 1L x {best_w_tag[:12]}", c_v2_1l_cand["weights"], do_loc=True, do_ripple=True)
        arms_out["arm6b_NEW_GATE_2L_LATCH_x_NEW_WRITE"] = _eval_2l_latched(f"Arm 6b| 2L x {best_w_tag[:12]}", best_v2_w, do_loc=True, do_ripple=True)

        # ---- ARM 6L: v2-Lite Upper-Bound Control (<=25 steps on frozen B, donor-pool species only!) ----
        v_star_map, vstar_meta = {}, {}
        for rn, sp_pool, tgt_s, old_s in [
            ("Bird->Fish", d1.DEV_DONOR_BIRDS_200, " fish", " bird"),
            ("Mammal->Reptile", d1.DEV_DONOR_MAMMALS_200, " reptile", " mammal"),
        ]:
            vs, vm = poe2.fit_v2_lite_write_vector(model_B, tok_B, dev_B, lB_m, best_v2_w[rn], sp_pool, tgt_s, old_s, n_steps=25)
            vm["cos_vstar_vs_native"] = round(float(np.dot(vs, v_nat_map[rn]) / (np.linalg.norm(vs) * np.linalg.norm(v_nat_map[rn]) + 1e-12)), 4)
            v_star_map[rn] = vs
            vstar_meta[rn] = vm
            print(f"  [v2-Lite v* | {rn:15s}] loss {vm['loss_start']}->{vm['loss_end']} | cos(v*, v_push)={vm['cos_vstar_vs_vpush']:.4f} | cos(v*, v_nat)={vm['cos_vstar_vs_native']:.4f}")
        arms_out["arm6L_2L_LATCH_x_v2Lite_vstar"] = {
            **_eval_2l_latched("Arm 6L| 2L x v2-Lite v*", v_star_map, do_loc=True, do_ripple=True),
            "vstar_meta": vstar_meta,
        }

        out["recipients"][tgt] = {
            "model_key": rk,
            "lB_early": lB_e,
            "lB_main": lB_m,
            "rm_B": rm_B,
            "selected_v2_write": best_w_tag,
            "fisher_projector": {"rank_k": fproj["rank_k"], "n_grads": fproj["n_grads"], "energy_captured": fproj["energy_captured"]},
            "write_audit": write_audit,
            "lda_2layer_meta": [{k: v for k, v in r.items() if k not in ("w_early", "w_main")} for r in lda_spec["rules"]],
            "arms": arms_out,
            "elapsed_sec": round(time.time() - tr0, 1),
        }

        po.unload_model(model_B)
        del model_B
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        out["meta"]["elapsed_sec"] = round(time.time() - t0, 1)
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n[v2_track1_dev_matrix] COMPLETE in {time.time() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
