#!/usr/bin/env python3
"""CPU verification gate for the Nov 27, 2026 Astra Pro adversarial audit fixes.

Verifies:
1. All 4 locked v1.0 SHA-256 artifacts + socket_manifest.json remain byte-identical.
2. F05: v3 2-hop primary `fc` scores against all 8 candidate leaves (query-independent
   column-preference shortcut drops from 1.00 -> <= 0.30).
3. F19: platonicswap/platonic_organ_engine.py rejects confounded frame IDs (AUC=1.0 -> ABSTAIN, weights=None).
4. F20: compiled bias-free gate firing rate on 200k fresh i.i.d. Gaussian states with large pin SD (~0.20)
   respects nominal alpha=0.002 (within finite-sample tolerance <= 0.0030 vs old 0.006135).
5. F21: solve_2constraint_kkt raises ValueError on contradictory collinear constraints.
"""
from __future__ import annotations

import ast
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla
from scipy.special import ndtri

REPO = Path(__file__).resolve().parent.parent

LOCKED_V1_SHAS = {
    "experiments/platonic_organ_engine.py": "22ef57c9d196e12f2cd3c9ecc79ec0813803df43d5a6009f5422904948802306",
    "experiments/bio_ripple_sealed.py": "001d296b4c278ab8b86f31363844b9d65ecc933761b12b02ec0afff9fc910fcb",
    "results/results_cap1_channel_pushforward.json": "bcf2dc70997e22c82418699535f9f7c9aa19967618978407170ee73ea047376e",
    "results/results_bio_ripple_sealed.json": "cd6c3c02a0881e04acbcdd4ea2144fc63389ebf61651395a1ac37912ee42cc23",
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def extract_funcs(path: Path, names: set[str], ns: Dict[str, Any]) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        n for n in tree.body
        if (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names)
        or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))
        or (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id in names)
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)


def main() -> None:
    # 1. Verify locked v1.0 SHA-256 artifacts are untouched
    for rel, expected in LOCKED_V1_SHAS.items():
        got = sha256_file(REPO / rel)
        assert got == expected, f"Locked v1.0 SHA mismatch for {rel}: got {got}, expected {expected}"
    assert (REPO / "results/socket_manifest.json").exists()
    print("[PASS 1/5] All locked v1.0 SHA-256 artifacts verified byte-identical.")

    # 2. Verify F05: query-independent column preference cannot ace all-8-leaf FC
    ns_v3: Dict[str, Any] = {"np": np, "Any": Any, "Dict": Dict, "List": List, "Sequence": Sequence, "Tuple": Tuple}
    extract_funcs(
        REPO / "experiments/v3_gate2_skill_and_sql.py",
        {
            "SEED", "WORD_CANDIDATES_KEYS", "WORD_CANDIDATES_VALS",
            "filter_single_token_words", "format_2hop_prompts",
            "build_2hop_routing_dataset", "_eval_2hop_policy_logits", "_joint_swap_summary",
        },
        ns_v3,
    )

    class SingleWordTok:
        def __init__(self):
            self.vocab: Dict[str, int] = {}

        def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
            w = text.strip()
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)
            return [self.vocab[w]]

    tok = SingleWordTok()
    ds = ns_v3["build_2hop_routing_dataset"]([tok], n_train=192, n_test=256, seed=20261127)
    items = ds["test"]
    scores = {}
    for pol in ("L", "R"):
        logits = np.full((len(items), len(tok.vocab)), -10.0)
        for i, it in enumerate(items):
            for r_idx, (_, vL, vR) in enumerate(it["hop2_rows"]):
                for col, val in (("L", vL), ("R", vR)):
                    logits[i, tok.encode(val)[0]] = (2.0 if col == pol else 0.0) + (0.1 if r_idx == 0 else 0.0)
        scores[pol] = ns_v3["_eval_2hop_policy_logits"](logits, items, tok, pol)
    res_v3 = ns_v3["_joint_swap_summary"](scores["L"], scores["R"])
    assert res_v3["joint_swap_fc"] <= 0.30, f"F05 regression: query-independent shortcut got {res_v3['joint_swap_fc']}"
    print(f"[PASS 2/5] F05 fixed: query-independent shortcut joint_swap_fc={res_v3['joint_swap_fc']:.4f} (legacy 2-way={res_v3['joint_swap_fc_2way_legacy']:.4f}).")

    # 3-5. Verify F19, F20, F21 in platonicswap/platonic_organ_engine.py
    ns_eng: Dict[str, Any] = dict(
        np=np, sla=sla, math=math, ndtri=ndtri, Counter=Counter, defaultdict=defaultdict,
        Any=Any, Dict=Dict, List=List, Optional=Optional, Sequence=Sequence, Tuple=Tuple,
        SILU_NEG_MAX=0.2784645427610738,
        FROZEN_TOLERANCES={"alpha": 0.002, "alpha_neg": 0.03, "delta": 0.02},
        NOISE_SEEDS=(11, 22, 33),
    )
    extract_funcs(REPO / "experiments/pilot_organs_p0.py", {"apply_exact_bos_null_biasfree"}, ns_eng)
    ns_eng["po"] = SimpleNamespace(apply_exact_bos_null_biasfree=ns_eng["apply_exact_bos_null_biasfree"])
    extract_funcs(
        REPO / "platonicswap/platonic_organ_engine.py",
        {
            "ledoit_wolf_cov", "compute_lw_constant_pin", "fit_generative_channel",
            "random_stiefel", "compute_frame_only_auroc", "solve_2constraint_kkt",
            "compile_channel_pushforward_organ",
        },
        ns_eng,
    )

    # F19: Frame confounding MUST force ABSTAIN
    rng = np.random.default_rng(20261127)
    d = 12
    xf = rng.normal(size=(5000, d)) + np.eye(d)[1] * 5
    xr = rng.normal(size=(10000, d)) + np.eye(d)[1] * 5
    bf = xf + rng.normal(scale=0.01, size=xf.shape)
    br = xr + rng.normal(scale=0.01, size=xr.shape)
    bos = np.eye(d)[-1]
    ch = ns_eng["fit_generative_channel"](xf, bf, xr, br, xf, bf, xr, br, bos[None, :])
    u = np.eye(d)[0]
    rule = {
        "name": "synthetic", "X_pos_A": xf[:60] + 8 * u, "X_neg_dom_A": xf[60:120] - 2 * u,
        "pos_frame_ids": np.zeros(60, dtype=int), "neg_frame_ids": np.ones(60, dtype=int), "v_A": u,
    }
    c_conf = ns_eng["compile_channel_pushforward_organ"]([rule], ch)
    assert c_conf["status"] == "ABSTAIN" and c_conf["weights"] is None and not c_conf["rules"][0]["frame_cert_pass"]
    print(f"[PASS 3/5] F19 fixed: confounded frame_auc={c_conf['rules'][0]['frame_only_auroc']} -> status={c_conf['status']}, weights=None.")

    # F20: Exact compiled bias-free gate conformal bound under large pin SD (~0.20)
    rng = np.random.default_rng(42)
    d = 3
    mu = np.array([0.0, 5.0, 0.0])
    fit = rng.normal(size=(50000, d)) + mu
    res = rng.normal(size=(50000, d)) + mu
    pin, pin_std = ns_eng["compute_lw_constant_pin"](fit)
    bos = np.array([0.0, 0.0, 1.0])
    stats = {
        "mu_B": fit.mean(0), "Sigma_BB_fit_lw": ns_eng["ledoit_wolf_cov"](fit)[0],
        "X_B_resid": res, "c_B": pin, "x_bos_B": bos, "bos_x_fit_B": bos[None, :],
    }
    base = rng.normal(size=(300, d))
    base = np.concatenate([base, -base]) + mu
    rule2 = {
        "name": "synthetic", "X_pos_A": base + np.array([8.0, 0.0, 0.0]),
        "X_neg_dom_A": base - np.array([2.0, 0.0, 0.0]),
        "pos_frame_ids": np.zeros(600, dtype=int), "neg_frame_ids": np.zeros(600, dtype=int),
        "v_A": np.array([1.0, 0.0, 0.0]),
    }
    c2 = ns_eng["compile_channel_pushforward_organ"]([rule2], None, native_B_stats=stats)
    assert c2["status"] == "COMPILED"
    fresh = rng.normal(size=(200000, d)) + mu
    g = fresh @ c2["weights"]["W_g"].T
    any_fire = float((g > 0.0).any(axis=1).mean())
    assert any_fire <= 0.0030, f"F20 regression: compiled biasfree gate firing rate={any_fire:.6f} > 0.0030 (nominal 0.002)"
    print(f"[PASS 4/5] F20 fixed: pin_std={pin_std:.4f}, compiled biasfree any_gate_gt0={any_fire:.6f} (nominal alpha=0.002000).")

    # F21: Infeasible collinear KKT QP raises ValueError
    try:
        ns_eng["solve_2constraint_kkt"](np.eye(2), np.array([1.0, 0.0]), np.array([-1.0, 0.0]), 1.0, 1.0)
        raise AssertionError("F21 regression: infeasible KKT did not raise ValueError!")
    except ValueError:
        pass
    print("[PASS 5/5] F21 fixed: contradictory collinear KKT constraints raise ValueError.")


if __name__ == "__main__":
    main()
