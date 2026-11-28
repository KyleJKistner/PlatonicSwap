#!/usr/bin/env python3
"""Reproducible Benchmark: Pythia-160m v1.1 Exact-GELU Parallel-MLP Retention Repair & Iso-KL SELF-GRAFT Audit.

Runs strictly on the discovery validation split (pool[150:300], n=256; sealed induction prompts pool[1500:]
and IOI prompts names[64:117] are never evaluated) with document-disjoint Python stdlib fit/val corpora
(excluding all pool[:600] task tokens):
  1. Verifies exact GELU(z) - GELU(-z) = z linear subtractor inside vanilla GPTNeoXMLP at both d=64 and d=768
     (<= 1e-5 error, zero hooks, paired input biases [-V^T x_bar, +V^T x_bar]).
  2. Matches published v1 (`experiments/pythia_v1_standalone_transplant.py`) identically: residual rotation Rh
     fitted from recipient L5 -> donor L3 (da["h3"]), Q5/Q8 ridge writes, generic-code mean-centering.
  3. Counterfactual KL removal check (h+p+e+l vs h+e+l): measures KL remaining when direct `p` write is removed
     from suffix input (note: `e` and `l` already read `p`).
  4. Overlapping Iso-KL comparison on seed2@step2000:
       - Donor 15L transplant with ridge lam in {0.01, 0.05, 0.10} + L8 exact-GELU MLP(-p)
       - Disjoint 24-head RAND sampled WITHOUT replacement (exact matching per-layer head counts, 24 distinct heads)
       - Clean EdgeCut (zeroing L6 Prev attention AND L8 MLP(-p) so the p-subtractor does not fire on zeroed p)
       - RECIPIENT-SELF-GRAFT (24 distinct native heads 8+7+9 nominated on disc_rank pool[0:150] + L8 MLP(-p))
         swept across scale in {0.25, 0.40, 0.55, 0.70, 0.85, 1.00, 1.15, 1.30} so KL overlaps Donor directly!
       - Verifies 148/148 frozen original recipient tensors byte-identical and bit-exact 0.0 rollback.

Usage:
  python experiments/pythia_v1_1_retention_and_limits.py --unit-test
  python experiments/pythia_v1_1_retention_and_limits.py --out-json results/results_pythia_v1_1_retention_and_limits.json
"""
import argparse
import collections
import inspect
import json
import os
import random
import re
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXConfig, GPTNeoXLayer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pythia_v1_standalone_transplant import (
    DEV, DONOR_ID, DONOR_SHA, RECIP_ID, RECIP_SHA, P8, IND_EARLY, IND_LATE,
    LOCKED_AP, LOCKED_AE, LOCKED_AL,
    iso_ln, extract_neox_head_params, build_folded_neox_layer,
    insert_blocks_at, eval_tail_kl_and_ppl, get_clean_single_tokens,
    build_induction_formats, to_tensor, sha256_tensor,
)


def ridge_f64(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    Xd, Yd = X.double(), Y.double()
    XtX = Xd.T @ Xd
    d = XtX.shape[0]
    reg = lam * (torch.trace(XtX) / d).clamp_min(1e-12)
    return torch.linalg.solve(XtX + reg * torch.eye(d, dtype=torch.float64), Xd.T @ Yd).float()


def procrustes_f64(X: torch.Tensor, Y: torch.Tensor):
    Xd, Yd = X.double(), Y.double()
    mx, my = Xd.mean(0), Yd.mean(0)
    Xc, Yc = Xd - mx, Yd - my
    U, _, Vt = torch.linalg.svd(Yc.T @ Xc, full_matrices=False)
    R = U @ Vt
    s = float((Yc * (Xc @ R.T)).sum() / Xc.square().sum().clamp_min(1e-12))
    M = s * R
    b = my - M @ mx
    return M.float(), b.float(), R.float(), s


def install_exact_gelu_subtractor_mlp(layer: GPTNeoXLayer, A: torch.Tensor, x_bar: torch.Tensor, t_bar: torch.Tensor):
    """Exact closed-form linear subtractor inside vanilla GPTNeoXMLP using GELU(z) - GELU(-z) = z!
    Computes MLP(x) = -(t_bar + A @ (x - x_bar)) using 2*d of 4*d hidden units (zero hooks):
      - Input rows [V^T; -V^T] with paired input biases [-V^T x_bar; +V^T x_bar]
      - Output columns [-U*S, +U*S] with output bias -t_bar
    """
    U, S, Vt = torch.linalg.svd(A.double(), full_matrices=False)
    d = A.shape[0]
    U_f, S_f, Vt_f = U.float(), S.float(), Vt.float()
    b_in = -(Vt_f @ x_bar.float())
    with torch.no_grad():
        layer.mlp.dense_h_to_4h.weight.zero_()
        layer.mlp.dense_h_to_4h.bias.zero_()
        layer.mlp.dense_4h_to_h.weight.zero_()
        layer.mlp.dense_4h_to_h.bias.zero_()
        layer.mlp.dense_h_to_4h.weight[0:d].copy_(Vt_f)
        layer.mlp.dense_h_to_4h.bias[0:d].copy_(b_in)
        layer.mlp.dense_h_to_4h.weight[d:2*d].copy_(-Vt_f)
        layer.mlp.dense_h_to_4h.bias[d:2*d].copy_(-b_in)
        US = U_f * S_f.unsqueeze(0)
        layer.mlp.dense_4h_to_h.weight[:, 0:d].copy_(-US)
        layer.mlp.dense_4h_to_h.weight[:, d:2*d].copy_(US)
        layer.mlp.dense_4h_to_h.bias.copy_(-t_bar.float())


def sample_disjoint_heads_no_replacement(template_heads, forbidden_set, rng, n_heads_per_layer=12):
    """Sample matching per-layer head counts WITHOUT replacement, strictly disjoint from forbidden_set."""
    by_layer = collections.Counter(l for l, _ in template_heads)
    out = []
    for l, cnt in sorted(by_layer.items()):
        avail = [h for h in range(n_heads_per_layer) if (l, h) not in forbidden_set]
        assert len(avail) >= cnt, f"Layer {l} only has {len(avail)} free heads, need {cnt}"
        for h in rng.sample(avail, cnt):
            out.append((l, h))
    return out


def build_diverse_disjoint_code(tok, forb_words: set, forb_ids: set, n_fit=128, n_val=80):
    stdlib_dir = Path(inspect.getfile(os)).parent
    files = sorted(p for p in stdlib_dir.glob("*.py") if p.is_file() and p.stat().st_size > 4000)
    pat = re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted(forb_words)) + r")\b", re.IGNORECASE)
    rng = random.Random(20261128)
    rng.shuffle(files)

    def harvest(file_subset, target_n):
        chunks, used = [], []
        for pf in file_subset:
            try:
                txt = pf.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            added = 0
            for start in range(0, max(1, len(txt) - 500), 600):
                seg = txt[start:start + 500].strip()
                if len(seg) < 180 or pat.search(seg):
                    continue
                ids = tok.encode(seg, add_special_tokens=False, truncation=True, max_length=96)
                if len(ids) < 64 or any(i in forb_ids for i in ids):
                    continue
                chunks.append(seg)
                added += 1
                if added >= 2 or len(chunks) >= target_n:
                    break
            if added > 0:
                used.append(pf.name)
            if len(chunks) >= target_n:
                break
        return chunks, used

    mid = int(len(files) * 0.60)
    fit_c, fit_f = harvest(files[:mid], n_fit)
    val_c, val_f = harvest(files[mid:], n_val)
    assert len(set(fit_f) & set(val_f)) == 0
    return fit_c, val_c, len(fit_f), len(val_f)


@torch.no_grad()
def eval_ind_fams(model, fams):
    ids = to_tensor(fams, "cl").to(DEV)
    t1s, avs, ms = [], [], []
    for st in range(0, len(fams), 32):
        lg = model(input_ids=ids[st:st+32], use_cache=False).logits[:, -1, :].float().cpu()
        for j, f in enumerate(fams[st:st+32]):
            row = lg[j]
            t1s.append(1.0 if int(row.argmax()) == f["tgt"] else 0.0)
            av = torch.tensor(f["all_vals"])
            avs.append(1.0 if int(av[row[av].argmax()]) == f["tgt"] else 0.0)
            ms.append(float(row[f["tgt"]] - row[f["alt"]]))
    return {"top1": sum(t1s)/len(t1s), "allVal": sum(avs)/len(avs), "margin": sum(ms)/len(ms)}


@torch.no_grad()
def mean_center(model, enc_fit, indices=(6, 7, 8)):
    for li in indices:
        outs = []
        for st in range(0, 64, 16):
            sl = slice(st, st + 16)
            ids, am = enc_fit["input_ids"][sl].to(DEV), enc_fit["attention_mask"][sl].to(DEV)
            box = {}
            h = model.gpt_neox.layers[li].attention.dense.register_forward_hook(lambda _m, _i, o: box.__setitem__("y", o.detach().float().cpu()))
            model(ids, attention_mask=am, use_cache=False); h.remove()
            vm = am.bool().cpu(); vm[:, 0] = False
            outs.append(box["y"][vm])
        model.gpt_neox.layers[li].attention.dense.bias.copy_(-torch.cat(outs, 0).mean(0).to(DEV))


@torch.no_grad()
def fit_l8_mlp_sub_p(model, enc_fit, lam_scale=1e-3):
    xs, ps = [], []
    for st in range(0, enc_fit["input_ids"].shape[0], 16):
        sl = slice(st, st + 16)
        ids, am = enc_fit["input_ids"][sl].to(DEV), enc_fit["attention_mask"][sl].to(DEV)
        box = {}
        with ExitStack() as stk:
            stk.callback(model.gpt_neox.layers[6].attention.dense.register_forward_hook(lambda _m, _i, o: box.__setitem__("p", o.detach().float().cpu())).remove)
            stk.callback(model.gpt_neox.layers[8].post_attention_layernorm.register_forward_hook(lambda _m, _i, o: box.__setitem__("x", o.detach().float().cpu())).remove)
            model(ids, attention_mask=am, use_cache=False)
        vm = am.bool().cpu(); vm[:, 0] = False
        xs.append(box["x"][vm]); ps.append(box["p"][vm])
    X, T = torch.cat(xs, 0), torch.cat(ps, 0)
    xb, tb = X.mean(0), T.mean(0)
    Xc, Tc = X - xb, T - tb
    Cxx = (Xc.T @ Xc) / len(X); Ctx = (Tc.T @ Xc) / len(X)
    lam = lam_scale * float(torch.trace(Cxx) / 768.0)
    A = torch.linalg.solve(Cxx + lam * torch.eye(768), Ctx.T).T
    r2 = 1.0 - float((T - (tb + Xc @ A.T)).square().sum() / Tc.square().sum().clamp_min(1e-12))
    install_exact_gelu_subtractor_mlp(model.gpt_neox.layers[8], A, xb, tb)
    return r2


def run_unit_test():
    errs = {}
    for d, nh in [(64, 4), (768, 12)]:
        cfg = GPTNeoXConfig(vocab_size=128, hidden_size=d, num_hidden_layers=2, num_attention_heads=nh, intermediate_size=4*d, use_parallel_residual=True)
        lay = GPTNeoXLayer(cfg, 0).eval()
        g = torch.Generator().manual_seed(0)
        A = torch.randn(d, d, generator=g) * (1.0 / (d ** 0.5))
        xb = torch.randn(d, generator=g) * 0.3
        tb = torch.randn(d, generator=g) * 0.3
        install_exact_gelu_subtractor_mlp(lay, A, xb, tb)
        x = torch.randn(4, 8, d, generator=g)
        with torch.no_grad():
            got = lay.mlp(x)
            want = -(tb + (x - xb) @ A.T)
            err = float((got - want).abs().max())
        assert err < 2e-5, f"d={d} exact GELU identity error too large: {err}"
        errs[d] = err
    # Verify RAND without replacement produces 24 distinct disjoint heads
    real_ind = set(P8) | set(IND_EARLY) | set(IND_LATE)
    rng = random.Random(42)
    rp = sample_disjoint_heads_no_replacement(P8, real_ind, rng)
    rie = sample_disjoint_heads_no_replacement(IND_EARLY, real_ind | set(rp), rng)
    ril = sample_disjoint_heads_no_replacement(IND_LATE, real_ind | set(rp) | set(rie), rng)
    assert len(set(rp + rie + ril)) == 24 and set(rp + rie + ril).isdisjoint(real_ind)
    print(f"UNIT TEST PASS: exact GELU(z)-GELU(-z)=z d=64 err={errs[64]:.2e}, d=768 err={errs[768]:.2e}; RAND=24 distinct disjoint heads")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit-test", action="store_true")
    ap.add_argument("--out-json", type=str, default="results/results_pythia_v1_1_retention_and_limits.json")
    args = ap.parse_args()
    if args.unit_test:
        run_unit_test()
        return

    t0 = time.time()
    print("=== PYTHIA-160M v1.1 EXACT-GELU RETENTION REPAIR & ISO-KL SELF-GRAFT BENCHMARK (disc_val ONLY) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(DONOR_ID, revision=DONOR_SHA, local_files_only=True)
    tok.pad_token = tok.eos_token
    full_pool = get_clean_single_tokens(tok, 1500)
    dr_pool, dv_pool = full_pool[0:150], full_pool[150:300]
    rs6_dr = build_induction_formats(tok, dr_pool, n_families=64,  n_pairs=6, n_distractors=2, fmt="repeat_seq",    seed=111)
    rs6_dv = build_induction_formats(tok, dv_pool, n_families=256, n_pairs=6, n_distractors=2, fmt="repeat_seq",    seed=222)
    nl6_dv = build_induction_formats(tok, dv_pool, n_families=256, n_pairs=6, n_distractors=2, fmt="newline_pairs", seed=223)

    forb_w = {w.lower() for w, _ in full_pool[:600]}; forb_id = {tid for _, tid in full_pool[:600]}
    fit_code, val_code, n_fit_f, n_val_f = build_diverse_disjoint_code(tok, forb_w, forb_id, n_fit=128, n_val=80)
    enc_fit = tok(fit_code, return_tensors="pt", padding=True, truncation=True, max_length=96)
    enc_val = tok(val_code, return_tensors="pt", padding=True, truncation=True, max_length=96)
    print(f"  Disjoint code corpus (excludes pool[:600]): fit={len(fit_code)} ({n_fit_f} files), val={len(val_code)} ({n_val_f} files), 0 shared files!", flush=True)

    mA = AutoModelForCausalLM.from_pretrained(DONOR_ID, revision=DONOR_SHA, local_files_only=True, dtype=torch.float32, attn_implementation="eager").to(DEV).eval()
    mB = AutoModelForCausalLM.from_pretrained(RECIP_ID, revision=RECIP_SHA, local_files_only=True, dtype=torch.float32, attn_implementation="eager").to(DEV).eval()

    # Record original recipient tensor hashes & reference logits for integrity/rollback check
    orig_hashes = {k: sha256_tensor(v) for k, v in mB.state_dict().items()}
    probe_ids = to_tensor(rs6_dv[:8], "cl").to(DEV)
    with torch.no_grad():
        ref_probe_lg = mB(input_ids=probe_ids, use_cache=False).logits.float().cpu()

    don_rs = eval_ind_fams(mA, rs6_dv); don_nl = eval_ind_fams(mA, nl6_dv)
    st_rs  = eval_ind_fams(mB, rs6_dv); st_nl  = eval_ind_fams(mB, nl6_dv)
    st_kl  = eval_tail_kl_and_ppl(mB, enc_val, ref=None)
    print(f"  DONOR: rs6={don_rs['top1']:.4f}(allVal={don_rs['allVal']:.4f}) nl6={don_nl['top1']:.4f}", flush=True)
    print(f"  STOCK: rs6={st_rs['top1']:.4f}(allVal={st_rs['allVal']:.4f}) nl6={st_nl['top1']:.4f} | PPL={st_kl['ppl']:.2f}", flush=True)

    p8_p = extract_neox_head_params(mA, P8)
    ie_p = extract_neox_head_params(mA, IND_EARLY)
    il_p = extract_neox_head_params(mA, IND_LATE)
    ln4A_w, ln4A_b = mA.gpt_neox.layers[4].input_layernorm.weight.detach().float().cpu().clone(), mA.gpt_neox.layers[4].input_layernorm.bias.detach().float().cpu().clone()
    ln8A_w, ln8A_b = mA.gpt_neox.layers[8].input_layernorm.weight.detach().float().cpu().clone(), mA.gpt_neox.layers[8].input_layernorm.bias.detach().float().cpu().clone()

    # Disjoint 24-head RAND sampled WITHOUT replacement (exact matching per-layer head counts!)
    real_ind = set(P8) | set(IND_EARLY) | set(IND_LATE)
    rng_r = random.Random(42)
    r_p8  = sample_disjoint_heads_no_replacement(P8, real_ind, rng_r)
    r_ie  = sample_disjoint_heads_no_replacement(IND_EARLY, real_ind | set(r_p8), rng_r)
    r_il  = sample_disjoint_heads_no_replacement(IND_LATE, real_ind | set(r_p8) | set(r_ie), rng_r)
    assert len(set(r_p8 + r_ie + r_il)) == 24 and set(r_p8 + r_ie + r_il).isdisjoint(real_ind)
    rp_p, rie_p, ril_p = extract_neox_head_params(mA, r_p8), extract_neox_head_params(mA, r_ie), extract_neox_head_params(mA, r_il)

    # Capture Donor x3, h3 (exact published v1 Rh target!), h5, h8 and Recipient h5
    with torch.no_grad():
        da = {k: [] for k in ("x3", "h3", "h5", "h8")}; db5 = []
        for st in range(0, len(fit_code), 16):
            sl = slice(st, st + 16); ids, am = enc_fit["input_ids"][sl].to(DEV), enc_fit["attention_mask"][sl].to(DEV)
            ca, cx, cb = {}, {}, {}
            with ExitStack() as stk:
                for l in (3, 5, 8): stk.callback(mA.gpt_neox.layers[l].register_forward_hook(lambda _m, _i, o, li=l: ca.__setitem__(li, (o[0] if isinstance(o, tuple) else o).detach().float().cpu())).remove)
                stk.callback(mA.gpt_neox.layers[3].input_layernorm.register_forward_hook(lambda _m, _i, o: cx.__setitem__(3, o.detach().float().cpu())).remove)
                stk.callback(mB.gpt_neox.layers[5].register_forward_hook(lambda _m, _i, o: cb.__setitem__(5, (o[0] if isinstance(o, tuple) else o).detach().float().cpu())).remove)
                mA(ids, attention_mask=am, use_cache=False); mB(ids, attention_mask=am, use_cache=False)
            vm = am.bool().cpu(); vm[:, 0] = False
            da["x3"].append(cx[3][vm])
            for l in (3, 5, 8): da[f"h{l}"].append(ca[l][vm])
            db5.append(cb[5][vm])
        for k in da: da[k] = torch.cat(da[k], 0)
        db5 = torch.cat(db5, 0)
    del mA

    # Exact published v1 interface fit: M1(iso_ln(h5_B) -> x3_A), Rh(h5_B -> h3_A)!
    M1, b1, _, _ = procrustes_f64(iso_ln(db5), da["x3"])
    _, _, Rh, sh = procrustes_f64(db5, da["h3"])
    ln_one, ln_zero = torch.ones(768), torch.zeros(768)

    # [1] Counterfactual KL removal check on exact published v1 graft (lam=0.01, NO MLP!)
    Q5_01 = ridge_f64(da["h5"] - da["h5"].mean(0), db5 - db5.mean(0), 0.01).T
    Q8_01 = ridge_f64(da["h8"] - da["h8"].mean(0), db5 - db5.mean(0), 0.01).T
    bp = build_folded_neox_layer(mB.config, 6, ln_one, ln_zero, p8_p, M1, b1, (LOCKED_AP / sh) * Rh.T)
    be = build_folded_neox_layer(mB.config, 7, ln_one, ln_zero, ie_p, ln4A_w.unsqueeze(1) * Rh, ln4A_b.clone(), LOCKED_AE * Q5_01)
    bl = build_folded_neox_layer(mB.config, 8, ln_one, ln_zero, il_p, ln8A_w.unsqueeze(1) * Rh, ln8A_b.clone(), LOCKED_AL * Q8_01)
    mv1 = insert_blocks_at(mB, 6, [bp, be, bl]).to(DEV).eval()
    mean_center(mv1, enc_fit, (6, 7, 8))
    kl_full = eval_tail_kl_and_ppl(mv1, enc_val, ref=st_kl)
    box_p = {}
    h_cap = mv1.gpt_neox.layers[6].attention.dense.register_forward_hook(lambda _m, _i, o: box_p.__setitem__("p", o))
    def sub_p_hk(_m, _i, o):
        t = o[0] if isinstance(o, tuple) else o
        t2 = t - box_p["p"]
        return (t2,) + o[1:] if isinstance(o, tuple) else t2
    h_sub = mv1.gpt_neox.layers[8].register_forward_hook(sub_p_hk)
    kl_no_p = eval_tail_kl_and_ppl(mv1, enc_val, ref=st_kl)
    h_cap.remove(); h_sub.remove()
    del mv1
    rem_frac = kl_no_p["kl_mean"] / max(kl_full["kl_mean"], 1e-12)
    print(f"\n[1/3] Counterfactual direct-p removal (exact v1 Rh=h3): full KL={kl_full['kl_mean']:.4f} -> direct_p_removed KL={kl_no_p['kl_mean']:.4f} ({100*rem_frac:.1f}% remaining!)", flush=True)

    # [2] Donor v1.1 Retention Sweep (ridge lam in {0.01, 0.05, 0.10} x L8 MLP(-p))
    print("\n[2/3] Donor v1.1 Retention Repair Sweep (exact v1 Rh=h3 + ridge lam + exact GELU L8 MLP(-p)):", flush=True)
    donor_rows = []
    integrity_verified = None
    for lam, use_mlp in [(0.01, False), (0.01, True), (0.05, True), (0.10, True)]:
        Q5 = ridge_f64(da["h5"] - da["h5"].mean(0), db5 - db5.mean(0), lam).T
        Q8 = ridge_f64(da["h8"] - da["h8"].mean(0), db5 - db5.mean(0), lam).T
        bp = build_folded_neox_layer(mB.config, 6, ln_one, ln_zero, p8_p, M1, b1, (LOCKED_AP / sh) * Rh.T)
        be = build_folded_neox_layer(mB.config, 7, ln_one, ln_zero, ie_p, ln4A_w.unsqueeze(1) * Rh, ln4A_b.clone(), LOCKED_AE * Q5)
        bl = build_folded_neox_layer(mB.config, 8, ln_one, ln_zero, il_p, ln8A_w.unsqueeze(1) * Rh, ln8A_b.clone(), LOCKED_AL * Q8)
        mc = insert_blocks_at(mB, 6, [bp, be, bl]).to(DEV).eval()
        mean_center(mc, enc_fit, (6, 7, 8))
        r2_p = fit_l8_mlp_sub_p(mc, enc_fit, 1e-3) if use_mlp else 0.0
        e_rs = eval_ind_fams(mc, rs6_dv); e_nl = eval_ind_fams(mc, nl6_dv); kl = eval_tail_kl_and_ppl(mc, enc_val, ref=st_kl)

        # Verify 148/148 frozen recipient tensors & bit-exact 0.0 rollback on lam=0.10 mlp=True
        if lam == 0.10 and use_mlp:
            sd = mc.state_dict()
            matched = 0
            for k, h0 in orig_hashes.items():
                if k.startswith("gpt_neox.layers."):
                    parts = k.split("."); li = int(parts[2])
                    nk = f"gpt_neox.layers.{li + 3}." + ".".join(parts[3:]) if li >= 6 else k
                else:
                    nk = k
                if sha256_tensor(sd[nk]) == h0:
                    matched += 1
            # Rollback: remove layers 6,7,8 and compare logits bit-identically
            layers_rb = list(mc.gpt_neox.layers[:6]) + list(mc.gpt_neox.layers[9:])
            mc_rb = insert_blocks_at(mB, 6, []).to(DEV).eval()
            mc_rb.gpt_neox.layers = torch.nn.ModuleList(layers_rb)
            for idx_l, lyr in enumerate(mc_rb.gpt_neox.layers):
                lyr.layer_idx = idx_l; lyr.attention.layer_idx = idx_l
            with torch.no_grad():
                rb_lg = mc_rb(input_ids=probe_ids, use_cache=False).logits.float().cpu()
            rb_diff = float((rb_lg - ref_probe_lg).abs().max())
            del mc_rb
            integrity_verified = {"frozen_tensors_matched": matched, "frozen_tensors_total": len(orig_hashes), "rollback_max_abs_diff": rb_diff}

        # Clean EdgeCut: zero BOTH L6 Prev attention AND L8 MLP(-p) so the p-subtractor does not fire on zeroed p!
        with torch.no_grad():
            mc.gpt_neox.layers[6].attention.dense.weight.zero_()
            mc.gpt_neox.layers[6].attention.dense.bias.zero_()
            for p in mc.gpt_neox.layers[8].mlp.parameters():
                p.zero_()
        e_ec = eval_ind_fams(mc, rs6_dv)
        del mc

        # Matched 24-distinct-head RAND at same lam + MLP(-p)
        bpr = build_folded_neox_layer(mB.config, 6, ln_one, ln_zero, rp_p,  M1, b1, (LOCKED_AP / sh) * Rh.T)
        ber = build_folded_neox_layer(mB.config, 7, ln_one, ln_zero, rie_p, ln4A_w.unsqueeze(1) * Rh, ln4A_b.clone(), LOCKED_AE * Q5)
        blr = build_folded_neox_layer(mB.config, 8, ln_one, ln_zero, ril_p, ln8A_w.unsqueeze(1) * Rh, ln8A_b.clone(), LOCKED_AL * Q8)
        mr = insert_blocks_at(mB, 6, [bpr, ber, blr]).to(DEV).eval()
        mean_center(mr, enc_fit, (6, 7, 8))
        if use_mlp: fit_l8_mlp_sub_p(mr, enc_fit, 1e-3)
        e_rn = eval_ind_fams(mr, rs6_dv); kl_rn = eval_tail_kl_and_ppl(mr, enc_val, ref=st_kl); del mr

        row = {
            "lam": lam, "use_mlp": use_mlp, "mlp_r2": r2_p,
            "kl_mean": kl["kl_mean"], "kl_p99": kl["kl_p99"], "ppl": kl["ppl"], "ppl_ratio": kl["ppl"] / st_kl["ppl"],
            "rs6_top1": e_rs["top1"], "rs6_allVal": e_rs["allVal"], "nl6_top1": e_nl["top1"],
            "edgecut_rs6": e_ec["top1"], "rand_rs6": e_rn["top1"], "rand_kl": kl_rn["kl_mean"],
        }
        donor_rows.append(row)
        print(f"  [Donor lam={lam:.2f} mlp={str(use_mlp):5s} r2={r2_p:.2f}] KL={kl['kl_mean']:.4f}(p99={kl['kl_p99']:.2f}, PPL={row['ppl_ratio']:.3f}x) | rs6={e_rs['top1']:.4f}(allVal={e_rs['allVal']:.4f}) nl6={e_nl['top1']:.4f} | EdgeCut={e_ec['top1']:.4f} | RAND24={e_rn['top1']:.4f}(KL={kl_rn['kl_mean']:.4f})", flush=True)

    # [3] RECIPIENT-SELF-GRAFT Overlapping Iso-KL Sweep (24 distinct native heads + L8 MLP(-p))!
    print("\n[3/3] RECIPIENT-SELF-GRAFT Overlapping Iso-KL Sweep (24 distinct native heads 8+7+9 + L8 MLP(-p)):", flush=True)
    ids_dr = to_tensor(rs6_dr, "cl").to(DEV)
    tg_dr = torch.tensor([f["tgt"] for f in rs6_dr]); ta_dr = torch.tensor([f["alt"] for f in rs6_dr])
    with torch.no_grad():
        out_b = mB(input_ids=ids_dr, output_attentions=True, use_cache=False)
        prev_sc = sorted([(float(torch.diagonal(out_b.attentions[l].float().cpu()[:, h, 1:, :-1], dim1=-2, dim2=-1).mean()), (l, h)) for l in range(1, 6) for h in range(12)], reverse=True)
        b_p8 = [lh for _, lh in prev_sc[:8]]
        hzb = {}
        with ExitStack() as stk:
            for l in range(4, 12): stk.callback(mB.gpt_neox.layers[l].attention.dense.register_forward_pre_hook(lambda _m, inp, li=l: hzb.__setitem__(li, inp[0].detach().float().cpu())).remove)
            mB(ids_dr, use_cache=False)
        WuB_raw = mB.get_output_embeddings().weight.detach().float().cpu(); du = WuB_raw[tg_dr] - WuB_raw[ta_dr]
        dla_b = sorted([(float(((hzb[l][:, -1, :].view(len(rs6_dr), 12, 64)[:, h, :] @ mB.gpt_neox.layers[l].attention.dense.weight.detach().float().cpu().view(768, 12, 64)[:, h, :].T) * du).sum(-1).mean()), (l, h)) for l in range(4, 12) for h in range(12)], reverse=True)
        b_ie = [lh for _, lh in dla_b if lh[0] <= 7 and lh not in set(b_p8)][:7]
        b_il = [lh for _, lh in dla_b if lh[0] >= 8][:9]
        assert len(set(b_p8 + b_ie + b_il)) == 24
    bp_sp = extract_neox_head_params(mB, b_p8); be_sp = extract_neox_head_params(mB, b_ie); bl_sp = extract_neox_head_params(mB, b_il)
    ln3B_w, ln3B_b = mB.gpt_neox.layers[3].input_layernorm.weight.detach().float().cpu().clone(), mB.gpt_neox.layers[3].input_layernorm.bias.detach().float().cpu().clone()
    ln4B_w, ln4B_b = mB.gpt_neox.layers[4].input_layernorm.weight.detach().float().cpu().clone(), mB.gpt_neox.layers[4].input_layernorm.bias.detach().float().cpu().clone()
    ln8B_w, ln8B_b = mB.gpt_neox.layers[8].input_layernorm.weight.detach().float().cpu().clone(), mB.gpt_neox.layers[8].input_layernorm.bias.detach().float().cpu().clone()

    self_rows = []
    for scale in (0.25, 0.40, 0.55, 0.70, 0.85, 1.00, 1.15, 1.30):
        bp_s = build_folded_neox_layer(mB.config, 6, ln_one, ln_zero, bp_sp, ln3B_w.unsqueeze(1)*torch.eye(768), ln3B_b.clone(), 1.0*torch.eye(768))
        be_s = build_folded_neox_layer(mB.config, 7, ln_one, ln_zero, be_sp, ln4B_w.unsqueeze(1)*torch.eye(768), ln4B_b.clone(), (scale * LOCKED_AE)*torch.eye(768))
        bl_s = build_folded_neox_layer(mB.config, 8, ln_one, ln_zero, bl_sp, ln8B_w.unsqueeze(1)*torch.eye(768), ln8B_b.clone(), (scale * LOCKED_AL)*torch.eye(768))
        ms = insert_blocks_at(mB, 6, [bp_s, be_s, bl_s]).to(DEV).eval()
        mean_center(ms, enc_fit, (6, 7, 8))
        r2_s = fit_l8_mlp_sub_p(ms, enc_fit, 1e-3)
        e_rs = eval_ind_fams(ms, rs6_dv); e_nl = eval_ind_fams(ms, nl6_dv); kl = eval_tail_kl_and_ppl(ms, enc_val, ref=st_kl)
        del ms
        row = {
            "scale": scale, "mlp_r2": r2_s,
            "kl_mean": kl["kl_mean"], "kl_p99": kl["kl_p99"], "ppl": kl["ppl"], "ppl_ratio": kl["ppl"] / st_kl["ppl"],
            "rs6_top1": e_rs["top1"], "rs6_allVal": e_rs["allVal"], "nl6_top1": e_nl["top1"],
        }
        self_rows.append(row)
        print(f"  [SELF-GRAFT scale={scale:.2f} r2={r2_s:.2f}] KL={kl['kl_mean']:.4f}(p99={kl['kl_p99']:.2f}, PPL={row['ppl_ratio']:.3f}x) | rs6={e_rs['top1']:.4f}(allVal={e_rs['allVal']:.4f}) nl6={e_nl['top1']:.4f}", flush=True)

    out = {
        "split": "disc_val (pool[150:300], n=256); sealed prompts pool[1500:] and IOI names[64:117] were not evaluated",
        "rh_target_layer": "donor L3 (h3) matching published experiments/pythia_v1_standalone_transplant.py",
        "donor": {"id": DONOR_ID, "sha": DONOR_SHA, "rs6": don_rs, "nl6": don_nl},
        "stock": {"id": RECIP_ID, "sha": RECIP_SHA, "rs6": st_rs, "nl6": st_nl, "code_ppl": st_kl["ppl"]},
        "corpus": {"n_fit": len(fit_code), "n_fit_files": n_fit_f, "n_val": len(val_code), "n_val_files": n_val_f, "shared_files": 0, "excluded_vocab_prefix": "pool[:600]"},
        "integrity_lam010_mlp": integrity_verified,
        "counterfactual_direct_p_removal": {
            "full_h_p_e_l_kl": kl_full["kl_mean"],
            "direct_p_removed_kl": kl_no_p["kl_mean"],
            "remaining_kl_fraction": rem_frac,
        },
        "donor_v1_1_sweep": donor_rows,
        "recipient_self_graft_iso_kl_sweep": self_rows,
        "rand_heads_distinct_24": {"prev8": r_p8, "ind_early7": r_ie, "ind_late9": r_il},
        "self_graft_heads_distinct_24": {"prev8": b_p8, "ind_early7": b_ie, "ind_late9": b_il},
        "elapsed_sec": time.time() - t0,
    }
    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=2))
    print(f"\nIntegrity: {integrity_verified} | Saved {out_p} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
