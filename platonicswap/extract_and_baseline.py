#!/usr/bin/env python3
"""
Hidden-state extraction, model factual evaluation, and exact Plato's Key baseline reproduction.

Supports incremental per-model extraction so additional models can be added with zero rework.
All model loads use local_files_only=True from /tmp/platos_key_lab/hf_cache/.

Usage:
  python extract_and_baseline.py --mode extract
  python extract_and_baseline.py --mode baseline
  python extract_and_baseline.py --mode all
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_DEFAULT_LAB_DIR = _REPO_ROOT / "experiments" if (_REPO_ROOT / "experiments").is_dir() else _THIS_DIR
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" if (_REPO_ROOT / "data").is_dir() else _DEFAULT_LAB_DIR
_DEFAULT_RESULTS_DIR = _REPO_ROOT / "results" if (_REPO_ROOT / "results").is_dir() else _DEFAULT_LAB_DIR

LAB_DIR = Path(os.environ.get("PLATONICSWAP_LAB_DIR", str(_DEFAULT_LAB_DIR)))
HF_CACHE_DIR = Path(os.environ.get("PLATONICSWAP_HF_CACHE", str(LAB_DIR / "hf_cache")))
CACHE_DIR = Path(os.environ.get("PLATONICSWAP_CACHE_DIR", str(LAB_DIR / "cache")))
DATASET_PATH = Path(os.environ.get("PLATONICSWAP_DATASET", str(_DEFAULT_DATA_DIR / "dataset.json")))
DATASET_2HOP_PATH = Path(os.environ.get("PLATONICSWAP_DATASET_2HOP", str(_DEFAULT_DATA_DIR / "dataset_2hop.json")))
RESULTS_PATH = Path(os.environ.get("PLATONICSWAP_BASELINE_RESULTS", str(_DEFAULT_RESULTS_DIR / "results_baseline.json")))

SEED = 20261125


def get_device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.bfloat16
    # On Apple Silicon CPU (sandboxed), float32 uses native AMX and is ~5x faster than bfloat16.
    return torch.device("cpu"), torch.float32


def load_model_and_tokenizer(model_dir: Path) -> Tuple[Any, Any, torch.device, torch.dtype]:
    device, dtype = get_device_and_dtype()
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype=dtype,
    )
    model.to(device)
    model.eval()
    return model, tok, device, dtype


def get_decoder_layers(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Unsupported model architecture: {type(model).__name__}")


def get_final_norm(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    return None


def locate_last_subject_token(
    tok: Any, prompt: str, char_span: Sequence[int]
) -> Tuple[int, int]:
    """
    Return (subj_tok_idx, seq_len) in unpadded token coordinates using exact offset_mapping.
    subj_tok_idx is the index of the last token overlapping the subject character span [c0, c1).
    """
    c0, c1 = int(char_span[0]), int(char_span[1])
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    seq_len = len(ids)

    subj_idx = -1
    for i, (s, e) in enumerate(offsets):
        if e <= s:
            continue
        # Token overlaps [c0, c1)
        if s < c1 and e > c0:
            subj_idx = i
    if subj_idx < 0:
        # Fallback: tokenize prefix + subject
        prefix_subj = prompt[:c1]
        ps_ids = tok(prefix_subj, add_special_tokens=True)["input_ids"]
        subj_idx = min(len(ps_ids) - 1, seq_len - 1)
    return subj_idx, seq_len


def _strip_accents(s: str) -> str:
    # Handle special Latin characters like Polish ł -> l alongside NFKD combining marks
    s = s.replace("ł", "l").replace("Ł", "L")
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def check_answer_match(gen_text: str, aliases: Sequence[str], relation: str = "") -> bool:
    """
    Check if generated continuation starts with any acceptable answer alias at a word boundary.
    For element_symbol, require exact case on the symbol so 'He' doesn't match 'he'.
    """
    cleaned = _strip_accents(gen_text).lstrip(" \t\"'(")
    if not cleaned:
        return False
    cleaned_low = cleaned.lower()
    case_sensitive = (relation == "element_symbol")
    for alias in aliases:
        if not alias:
            continue
        a_norm = _strip_accents(alias)
        if case_sensitive:
            if cleaned.startswith(a_norm):
                rest = cleaned[len(a_norm):]
                if not rest or not rest[0].isalnum():
                    return True
        else:
            al = a_norm.lower()
            if cleaned_low.startswith(al):
                rest = cleaned_low[len(al):]
                if not rest or not rest[0].isalnum():
                    return True
    return False


def _alias_first_token_ids(tok: Any, aliases: Sequence[str]) -> set:
    """Collect candidate first token IDs (with and without leading space) for fast exact pre-filtering."""
    out = set()
    for a in aliases:
        for variant in (f" {a}", a, f" {a.lower()}", f" {a.capitalize()}"):
            ids = tok.encode(variant, add_special_tokens=False)
            if ids:
                out.add(ids[0])
    return out


def extract_frozen_weights(model: Any, out_path: Path, top_k: int = 64) -> dict:
    """
    Extract frozen RMSNorm gamma vectors and top-k input singular subspaces of downstream
    MLP ([gate_proj; up_proj]) and Attention ([q_proj; k_proj; v_proj]) weights per layer.
    """
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    hidden_size = int(model.config.hidden_size)
    res_mult = float(getattr(model.config, "residual_multiplier", 1.0))

    in_ln = torch.zeros((n_layers, hidden_size), dtype=torch.float32)
    post_ln = torch.zeros((n_layers, hidden_size), dtype=torch.float32)
    mlp_v = torch.zeros((n_layers, top_k, hidden_size), dtype=torch.float16)
    mlp_s = torch.zeros((n_layers, top_k), dtype=torch.float32)
    attn_v = torch.zeros((n_layers, top_k, hidden_size), dtype=torch.float16)
    attn_s = torch.zeros((n_layers, top_k), dtype=torch.float32)

    q_rank = min(top_k + 8, hidden_size)
    with torch.no_grad():
        for l_idx, layer in enumerate(layers):
            if hasattr(layer, "input_layernorm") and hasattr(layer.input_layernorm, "weight"):
                in_ln[l_idx] = layer.input_layernorm.weight.detach().float().cpu()
            if hasattr(layer, "post_attention_layernorm") and hasattr(layer.post_attention_layernorm, "weight"):
                post_ln[l_idx] = layer.post_attention_layernorm.weight.detach().float().cpu()

            # MLP input projection matrix [out, D] -> right singular vectors V^T [k, D]
            mlp_mats = []
            for attr in ("gate_proj", "up_proj", "input_linear"):
                if hasattr(layer.mlp, attr):
                    mlp_mats.append(getattr(layer.mlp, attr).weight.detach().float().cpu())
            if mlp_mats:
                W_mlp = torch.cat(mlp_mats, dim=0)
                _, S, V = torch.svd_lowrank(W_mlp, q=q_rank, niter=2)
                k_eff = min(top_k, S.shape[0])
                mlp_s[l_idx, :k_eff] = S[:k_eff]
                mlp_v[l_idx, :k_eff, :] = V[:, :k_eff].T.to(torch.float16)

            # Attention QKV input projection matrix
            attn_mats = []
            for attr in ("q_proj", "k_proj", "v_proj"):
                if hasattr(layer.self_attn, attr):
                    attn_mats.append(getattr(layer.self_attn, attr).weight.detach().float().cpu())
            if attn_mats:
                W_attn = torch.cat(attn_mats, dim=0)
                _, S, V = torch.svd_lowrank(W_attn, q=q_rank, niter=2)
                k_eff = min(top_k, S.shape[0])
                attn_s[l_idx, :k_eff] = S[:k_eff]
                attn_v[l_idx, :k_eff, :] = V[:, :k_eff].T.to(torch.float16)

        final_norm = get_final_norm(model)
        if final_norm is not None and hasattr(final_norm, "weight"):
            final_gamma = final_norm.weight.detach().float().cpu()
        else:
            final_gamma = torch.ones(hidden_size, dtype=torch.float32)

    payload = {
        "num_layers": n_layers,
        "hidden_size": hidden_size,
        "residual_multiplier": res_mult,
        "input_layernorm_gamma": in_ln,
        "post_attention_layernorm_gamma": post_ln,
        "final_norm_gamma": final_gamma,
        "mlp_gate_up_top_v": mlp_v,
        "mlp_gate_up_top_s": mlp_s,
        "attn_qkv_top_v": attn_v,
        "attn_qkv_top_s": attn_s,
    }
    torch.save(payload, out_path)
    return {
        "num_layers": n_layers,
        "hidden_size": hidden_size,
        "residual_multiplier": res_mult,
    }


def _greedy_decode_from_prefill(
    model: Any,
    first_logits: torch.Tensor,
    past_key_values: Any,
    attention_mask: torch.Tensor,
    max_new_tokens: int = 6,
) -> torch.Tensor:
    """
    Continue greedy decoding from prefill logits + KV cache for max_new_tokens total tokens.
    Returns [B, max_new_tokens] token IDs.
    """
    next_tok = first_logits.argmax(dim=-1, keepdim=True)  # [B, 1]
    gen_ids = [next_tok]
    cur_mask = attention_mask
    pkv = past_key_values

    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            cur_mask = torch.cat(
                [cur_mask, torch.ones((cur_mask.shape[0], 1), dtype=cur_mask.dtype, device=cur_mask.device)],
                dim=1,
            )
            out = model(
                input_ids=next_tok,
                attention_mask=cur_mask,
                past_key_values=pkv,
                use_cache=True,
            )
            pkv = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_ids.append(next_tok)
    return torch.cat(gen_ids, dim=1)


def _rank_candidates_single(
    model: Any,
    tok: Any,
    prompt_ids: torch.Tensor,
    first_logprobs: torch.Tensor,
    candidates: Sequence[str],
    correct_answer: str,
    device: torch.device,
) -> Tuple[int, float]:
    """
    Compute 1-indexed rank of correct_answer among up to 50 within-relation candidate answers
    by exact sequence log-probability log P(" " + cand | prompt).
    Uses fast branch-and-bound: single-token candidates are scored in O(1) from first_logprobs;
    multi-token candidates are scored exactly via a single batched continuation pass if needed.
    """
    cand_tok_lists: List[List[int]] = []
    cand_names: List[str] = list(candidates)
    # Ensure correct_answer is in candidate pool
    corr_idx = -1
    for idx, c in enumerate(cand_names):
        if c.strip().lower() == correct_answer.strip().lower():
            corr_idx = idx
            break
    if corr_idx < 0:
        cand_names.append(correct_answer)
        corr_idx = len(cand_names) - 1

    scores = np.full(len(cand_names), -1e9, dtype=np.float64)
    multi_indices: List[int] = []

    for idx, c in enumerate(cand_names):
        ids = tok.encode(" " + c, add_special_tokens=False)
        cand_tok_lists.append(ids)
        if not ids:
            continue
        if len(ids) == 1:
            scores[idx] = float(first_logprobs[ids[0]].item())
        else:
            # Upper bound is first token log-prob (since subsequent log-probs <= 0)
            scores[idx] = float(first_logprobs[ids[0]].item())
            multi_indices.append(idx)

    # First evaluate exact score of correct_answer if multi-token
    def _eval_multi_exact(idxs: List[int]) -> None:
        if not idxs:
            return
        # Batch evaluate all requested multi-token candidates by concatenating prompt + cand[:-1]
        seqs = []
        p_list = prompt_ids.tolist()
        for ci in idxs:
            c_ids = cand_tok_lists[ci]
            seqs.append(p_list + c_ids[:-1])
        max_len = max(len(s) for s in seqs)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        inp = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
        amask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
        for bi, s in enumerate(seqs):
            # Left pad
            offset = max_len - len(s)
            inp[bi, offset:] = torch.tensor(s, dtype=torch.long, device=device)
            amask[bi, offset:] = 1
        with torch.no_grad():
            logits = model(input_ids=inp, attention_mask=amask, use_cache=False).logits
        for bi, ci in enumerate(idxs):
            c_ids = cand_tok_lists[ci]
            # Position of prompt's last token predicts c_ids[0], next predicts c_ids[1], etc.
            n_c = len(c_ids)
            start_pos = max_len - n_c  # last token of prompt is at max_len - (n_c - 1) - 1 = max_len - n_c
            lp_sum = 0.0
            for t in range(n_c):
                lps = torch.log_softmax(logits[bi, start_pos + t].float(), dim=-1)
                lp_sum += float(lps[c_ids[t]].item())
            scores[ci] = lp_sum

    if corr_idx in multi_indices:
        _eval_multi_exact([corr_idx])
        multi_indices = [ci for ci in multi_indices if ci != corr_idx]

    corr_score = scores[corr_idx]
    # Exact branch-and-bound: only multi-token candidates whose first-token upper bound >= corr_score
    # can possibly beat correct_answer!
    need_exact = [ci for ci in multi_indices if scores[ci] >= corr_score - 1e-6]
    if need_exact:
        _eval_multi_exact(need_exact)

    rank = 1 + int(np.sum(scores > corr_score + 1e-9))
    return rank, float(corr_score)


def extract_prompts_batched(
    model: Any,
    tok: Any,
    prompts: Sequence[str],
    char_spans: Sequence[Sequence[int]],
    device: torch.device,
    batch_size: int = 32,
    capture_last: bool = False,
    capture_mlp: bool = False,
    do_generate: bool = True,
    max_new_tokens: int = 6,
    return_first_logprobs: bool = False,
) -> dict:
    """
    Single-pass batched extraction of hidden states at all layers + greedy generation.
    Layer index l in [0, L-1] is the output of decoder layer l (residual stream after layer l).
    """
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    hidden_size = int(model.config.hidden_size)
    res_mult = float(getattr(model.config, "residual_multiplier", 1.0))
    n = len(prompts)

    subj_tok_idxs = []
    seq_lens = []
    for p, span in zip(prompts, char_spans):
        si, sl = locate_last_subject_token(tok, p, span)
        subj_tok_idxs.append(si)
        seq_lens.append(sl)

    states_subj = torch.zeros((n, n_layers, hidden_size), dtype=torch.float16)
    states_last = torch.zeros((n, n_layers, hidden_size), dtype=torch.float16) if capture_last else None
    states_mlp = torch.zeros((n, n_layers, hidden_size), dtype=torch.float16) if capture_mlp else None
    gen_texts: List[str] = [""] * n
    first_logprobs_list: Optional[List[torch.Tensor]] = [None] * n if return_first_logprobs else None

    # Mutable batch context read by hooks
    ctx = {"active": False, "start": 0, "subj_pos": None}

    handles = []
    for l_idx, layer in enumerate(layers):
        def make_layer_hook(li: int):
            def hook(_mod, _inp, out):
                if not ctx["active"]:
                    return
                h = out[0] if isinstance(out, tuple) else out  # [B, T, D]
                if h.shape[1] <= 1:
                    return
                bsz = h.shape[0]
                s0 = ctx["start"]
                b_ar = torch.arange(bsz, device=h.device)
                spos = ctx["subj_pos"]
                states_subj[s0:s0 + bsz, li, :] = h[b_ar, spos, :].detach().to(torch.float16).cpu()
                if capture_last:
                    states_last[s0:s0 + bsz, li, :] = h[:, -1, :].detach().to(torch.float16).cpu()
            return hook
        handles.append(layer.register_forward_hook(make_layer_hook(l_idx)))

        if capture_mlp and hasattr(layer, "mlp"):
            def make_mlp_hook(li: int):
                def hook(_mod, _inp, out):
                    if not ctx["active"]:
                        return
                    h = out[0] if isinstance(out, tuple) else out
                    if h.shape[1] <= 1:
                        return
                    bsz = h.shape[0]
                    s0 = ctx["start"]
                    b_ar = torch.arange(bsz, device=h.device)
                    spos = ctx["subj_pos"]
                    val = h[b_ar, spos, :].detach()
                    if res_mult != 1.0:
                        val = val * res_mult
                    states_mlp[s0:s0 + bsz, li, :] = val.to(torch.float16).cpu()
                return hook
            handles.append(layer.mlp.register_forward_hook(make_mlp_hook(l_idx)))

    try:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_prompts = list(prompts[start:end])
            enc = tok(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            t_pad = input_ids.shape[1]

            # Left-padded positions of last subject token
            spos_list = []
            for bi in range(end - start):
                idx = start + bi
                pad_len = t_pad - seq_lens[idx]
                pos = max(0, min(t_pad - 1, pad_len + subj_tok_idxs[idx]))
                spos_list.append(pos)
            ctx["start"] = start
            ctx["subj_pos"] = torch.tensor(spos_list, dtype=torch.long, device=device)
            ctx["active"] = True

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=do_generate)
            ctx["active"] = False

            first_logits = out.logits[:, -1, :]  # [B, V]
            if return_first_logprobs:
                lps = torch.log_softmax(first_logits.float(), dim=-1).cpu()
                for bi in range(end - start):
                    first_logprobs_list[start + bi] = lps[bi]

            if do_generate:
                gen_ids = _greedy_decode_from_prefill(
                    model, first_logits, out.past_key_values, attention_mask, max_new_tokens=max_new_tokens
                )
                for bi in range(end - start):
                    gen_texts[start + bi] = tok.decode(gen_ids[bi].tolist(), skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()

    return {
        "states_subj": states_subj,
        "states_last": states_last,
        "states_mlp": states_mlp,
        "gen_texts": gen_texts,
        "first_logprobs": first_logprobs_list,
        "subj_tok_idxs": subj_tok_idxs,
        "seq_lens": seq_lens,
    }


def extract_single_model(
    model_key: str,
    model_dir: Path,
    dataset: dict,
    dataset_2hop: dict,
    force: bool = False,
) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    eval_path = CACHE_DIR / f"{model_key}_eval.json"
    states_path = CACHE_DIR / f"{model_key}_states.pt"
    weights_path = CACHE_DIR / f"{model_key}_weights.pt"
    hop2_states_path = CACHE_DIR / f"{model_key}_2hop.pt"
    hop2_eval_path = CACHE_DIR / f"{model_key}_2hop_eval.json"

    if (
        not force
        and eval_path.exists()
        and states_path.exists()
        and weights_path.exists()
        and hop2_states_path.exists()
        and hop2_eval_path.exists()
    ):
        print(f"[extract] {model_key}: already cached, skipping (use --force to recompute).")
        with open(eval_path, "r", encoding="utf-8") as f:
            ev = json.load(f)
        return ev["model_info"]

    t0 = time.time()
    print(f"[extract] Loading {model_key} from {model_dir} ...")
    model, tok, device, dtype = load_model_and_tokenizer(model_dir)
    n_layers = len(get_decoder_layers(model))
    hidden_size = int(model.config.hidden_size)
    vocab_size = int(model.config.vocab_size)
    arch = type(model).__name__
    print(f"  arch={arch}, L={n_layers}, D={hidden_size}, V={vocab_size}, device={device}, dtype={dtype}")

    # 1. Extract frozen weights metadata
    t_w = time.time()
    w_info = extract_frozen_weights(model, weights_path, top_k=64)
    print(f"  saved weights in {time.time() - t_w:.1f}s")

    facts = dataset["facts"]
    rel_cands = dataset["relation_candidate_answers"]
    n_facts = len(facts)

    # 2. Direct prompt pass (states_subj, states_last, states_mlp, generation, first_logprobs)
    t_d = time.time()
    direct_res = extract_prompts_batched(
        model,
        tok,
        [f["direct_prompt"] for f in facts],
        [f["direct_subj_char_span"] for f in facts],
        device=device,
        batch_size=32,
        capture_last=True,
        capture_mlp=True,
        do_generate=True,
        max_new_tokens=6,
        return_first_logprobs=True,
    )
    print(f"  direct pass ({n_facts} facts) done in {time.time() - t_d:.1f}s")

    # 3. Paraphrase prompt pass (states_subj + generation)
    t_p = time.time()
    para_res = extract_prompts_batched(
        model,
        tok,
        [f["paraphrase_prompt"] for f in facts],
        [f["para_subj_char_span"] for f in facts],
        device=device,
        batch_size=32,
        capture_last=False,
        capture_mlp=False,
        do_generate=True,
        max_new_tokens=6,
        return_first_logprobs=False,
    )
    print(f"  paraphrase pass done in {time.time() - t_p:.1f}s")

    # 4. 3-shot ICL prompt pass (states_subj + generation; smaller batch_size=16 for longer prompts)
    t_i = time.time()
    icl_res = extract_prompts_batched(
        model,
        tok,
        [f["icl3_prompt"] for f in facts],
        [f["icl3_subj_char_span"] for f in facts],
        device=device,
        batch_size=16,
        capture_last=False,
        capture_mlp=False,
        do_generate=True,
        max_new_tokens=6,
        return_first_logprobs=False,
    )
    print(f"  icl3 pass done in {time.time() - t_i:.1f}s")

    # 5. Evaluate correctness & within-relation candidate ranking (for all facts, especially misses)
    t_r = time.time()
    per_fact_eval: List[dict] = []
    rel_stats = defaultdict(lambda: {"n": 0, "direct": 0, "para": 0, "icl3": 0, "rank1": 0, "dark_cand": 0})

    for idx, f in enumerate(facts):
        rel = f["relation"]
        aliases = f["aliases"]
        d_gen = direct_res["gen_texts"][idx]
        p_gen = para_res["gen_texts"][idx]
        i_gen = icl_res["gen_texts"][idx]

        d_ok = check_answer_match(d_gen, aliases, rel)
        p_ok = check_answer_match(p_gen, aliases, rel)
        i_ok = check_answer_match(i_gen, aliases, rel)

        # First-token probability of canonical answer
        ans_ids = tok.encode(" " + f["answer"], add_special_tokens=False)
        lp_vec = direct_res["first_logprobs"][idx]
        first_prob = float(math.exp(lp_vec[ans_ids[0]].item())) if ans_ids else 0.0

        # Exact within-relation ranking among up to 50 candidate answers
        # (Exact multi-token evaluation for every direct miss; for direct hits where greedy
        # generation already matches the answer, rank is 1 and we record first-token log-prob.)
        if not d_ok:
            p_ids = torch.tensor(tok.encode(f["direct_prompt"], add_special_tokens=True), dtype=torch.long)
            rank_in_rel, ans_lp = _rank_candidates_single(
                model, tok, p_ids, lp_vec, rel_cands[rel], f["answer"], device
            )
        else:
            rank_in_rel = 1
            ans_lp = float(lp_vec[ans_ids[0]].item()) if ans_ids else -100.0
        rank_top1 = (rank_in_rel == 1)
        is_dark_cand = (not d_ok) and (i_ok or p_ok or rank_top1)

        per_fact_eval.append({
            "idx": idx,
            "id": f["id"],
            "relation": rel,
            "split": f["split"],
            "direct_correct": d_ok,
            "direct_gen": d_gen,
            "direct_first_tok_prob": round(first_prob, 6),
            "direct_ans_logprob": round(ans_lp, 4),
            "paraphrase_correct": p_ok,
            "paraphrase_gen": p_gen,
            "icl3_correct": i_ok,
            "icl3_gen": i_gen,
            "rank_in_relation": rank_in_rel,
            "rank_top1": rank_top1,
            "is_dark_memory_candidate": is_dark_cand,
            "subj_tok_idx_direct": direct_res["subj_tok_idxs"][idx],
            "seq_len_direct": direct_res["seq_lens"][idx],
            "subj_tok_idx_paraphrase": para_res["subj_tok_idxs"][idx],
            "seq_len_paraphrase": para_res["seq_lens"][idx],
            "subj_tok_idx_icl3": icl_res["subj_tok_idxs"][idx],
            "seq_len_icl3": icl_res["seq_lens"][idx],
        })
        st = rel_stats[rel]
        st["n"] += 1
        st["direct"] += int(d_ok)
        st["para"] += int(p_ok)
        st["icl3"] += int(i_ok)
        st["rank1"] += int(rank_top1)
        st["dark_cand"] += int(is_dark_cand)

    print(f"  ranking & eval done in {time.time() - t_r:.1f}s")

    # Save 1-hop hidden states
    torch.save(
        {
            "fact_ids": [f["id"] for f in facts],
            "direct_subj": direct_res["states_subj"],
            "direct_last": direct_res["states_last"],
            "direct_mlp_subj": direct_res["states_mlp"],
            "paraphrase_subj": para_res["states_subj"],
            "icl3_subj": icl_res["states_subj"],
        },
        states_path,
    )

    model_info = {
        "model_key": model_key,
        "local_path": str(model_dir),
        "arch": arch,
        "num_layers": n_layers,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "residual_multiplier": w_info["residual_multiplier"],
        "dtype_used": str(dtype),
    }

    summary = {
        "n_facts": n_facts,
        "direct_acc": round(sum(x["direct_correct"] for x in per_fact_eval) / n_facts, 4),
        "paraphrase_acc": round(sum(x["paraphrase_correct"] for x in per_fact_eval) / n_facts, 4),
        "icl3_acc": round(sum(x["icl3_correct"] for x in per_fact_eval) / n_facts, 4),
        "rank_top1_acc": round(sum(x["rank_top1"] for x in per_fact_eval) / n_facts, 4),
        "n_direct_misses": sum(not x["direct_correct"] for x in per_fact_eval),
        "n_dark_memory_candidates": sum(x["is_dark_memory_candidate"] for x in per_fact_eval),
        "by_relation": {
            r: {
                "n": st["n"],
                "direct_acc": round(st["direct"] / st["n"], 4),
                "para_acc": round(st["para"] / st["n"], 4),
                "icl3_acc": round(st["icl3"] / st["n"], 4),
                "rank1_acc": round(st["rank1"] / st["n"], 4),
                "dark_cand": st["dark_cand"],
            }
            for r, st in sorted(rel_stats.items())
        },
    }
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump({"model_info": model_info, "summary": summary, "facts": per_fact_eval}, f, indent=2)

    # 6. 2-hop latent bridge dataset extraction & eval
    t_2 = time.time()
    items_2 = dataset_2hop["items"]
    n_2 = len(items_2)
    h2_dir = extract_prompts_batched(
        model,
        tok,
        [x["direct_prompt"] for x in items_2],
        [x["direct_subj_char_span"] for x in items_2],
        device=device,
        batch_size=32,
        capture_last=True,
        capture_mlp=False,
        do_generate=True,
        max_new_tokens=6,
    )
    h2_hop1 = extract_prompts_batched(
        model,
        tok,
        [x["hop1_prompt"] for x in items_2],
        [[x["hop1_prompt"].index(x["subject"]), x["hop1_prompt"].index(x["subject"]) + len(x["subject"])] for x in items_2],
        device=device,
        batch_size=32,
        capture_last=False,
        capture_mlp=False,
        do_generate=True,
        max_new_tokens=6,
    )
    h2_hop2 = extract_prompts_batched(
        model,
        tok,
        [x["hop2_prompt"] for x in items_2],
        [[len(x["hop2_prefix"]), len(x["hop2_prefix"]) + len(x["bridge_entity"])] for x in items_2],
        device=device,
        batch_size=32,
        capture_last=False,
        capture_mlp=False,
        do_generate=True,
        max_new_tokens=6,
    )

    hop2_eval = []
    for idx, x in enumerate(items_2):
        d_ok = check_answer_match(h2_dir["gen_texts"][idx], x["aliases"], x["hop2_relation"])
        h1_aliases = [x["bridge_entity"]]
        if x["bridge_entity"] == "United Kingdom":
            h1_aliases += ["the United Kingdom", "UK", "Britain", "England"]
        elif x["bridge_entity"] == "United States":
            h1_aliases += ["the United States", "USA", "US", "America"]
        elif x["bridge_entity"] == "Netherlands":
            h1_aliases += ["the Netherlands", "Holland"]
        elif x["bridge_entity"] == "Czech Republic":
            h1_aliases += ["the Czech Republic", "Czechia"]
        h1_ok = check_answer_match(h2_hop1["gen_texts"][idx], h1_aliases, x["bridge_relation"])
        h2_ok = check_answer_match(h2_hop2["gen_texts"][idx], x["aliases"], x["hop2_relation"])
        hop2_eval.append({
            "idx": idx,
            "id": x["id"],
            "hop_type": x["hop_type"],
            "split": x["split"],
            "direct_correct": d_ok,
            "direct_gen": h2_dir["gen_texts"][idx],
            "hop1_correct": h1_ok,
            "hop1_gen": h2_hop1["gen_texts"][idx],
            "hop2_correct": h2_ok,
            "hop2_gen": h2_hop2["gen_texts"][idx],
            "subj_tok_idx_direct": h2_dir["subj_tok_idxs"][idx],
            "seq_len_direct": h2_dir["seq_lens"][idx],
            "subj_tok_idx_hop1": h2_hop1["subj_tok_idxs"][idx],
            "seq_len_hop1": h2_hop1["seq_lens"][idx],
            "subj_tok_idx_hop2": h2_hop2["subj_tok_idxs"][idx],
            "seq_len_hop2": h2_hop2["seq_lens"][idx],
        })

    torch.save(
        {
            "item_ids": [x["id"] for x in items_2],
            "states_subj": h2_dir["states_subj"],
            "states_last": h2_dir["states_last"],
            "hop1_subj": h2_hop1["states_subj"],
            "hop2_subj": h2_hop2["states_subj"],
        },
        hop2_states_path,
    )
    hop2_summary = {
        "n_items": n_2,
        "direct_2hop_acc": round(sum(x["direct_correct"] for x in hop2_eval) / n_2, 4),
        "hop1_acc": round(sum(x["hop1_correct"] for x in hop2_eval) / n_2, 4),
        "hop2_acc": round(sum(x["hop2_correct"] for x in hop2_eval) / n_2, 4),
    }
    with open(hop2_eval_path, "w", encoding="utf-8") as f:
        json.dump({"model_info": model_info, "summary": hop2_summary, "items": hop2_eval}, f, indent=2)

    print(
        f"[extract] {model_key} DONE in {time.time() - t0:.1f}s | "
        f"direct={summary['direct_acc']:.3f}, para={summary['paraphrase_acc']:.3f}, "
        f"icl3={summary['icl3_acc']:.3f}, rank1={summary['rank_top1_acc']:.3f}, "
        f"dark_cands={summary['n_dark_memory_candidates']}/{summary['n_direct_misses']} | "
        f"2hop={hop2_summary['direct_2hop_acc']:.3f}"
    )

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return model_info


def update_cache_manifest(dataset: dict, dataset_2hop: dict, model_infos: Dict[str, dict]) -> dict:
    manifest = {
        "lab_dir": str(LAB_DIR),
        "dataset_path": str(DATASET_PATH),
        "dataset_2hop_path": str(DATASET_2HOP_PATH),
        "n_facts": dataset["meta"]["n_facts"],
        "n_relations": dataset["meta"]["n_relations"],
        "relations": dataset["meta"]["relations"],
        "split_counts": dataset["meta"]["split_counts"],
        "n_2hop_items": dataset_2hop["meta"]["n_items"],
        "tensor_convention": {
            "layer_index_meaning": "l in [0, L-1] is post-decoder-layer residual stream (output of model.model.layers[l]); patching replaces layers[l] output at subj_tok_idx",
            "direct_subj": "[N, L, D] float16 residual stream at last subject token of direct_prompt",
            "direct_last": "[N, L, D] float16 residual stream at last prompt token of direct_prompt",
            "direct_mlp_subj": "[N, L, D] float16 residual_multiplier * mlp(x) delta at last subject token",
            "paraphrase_subj": "[N, L, D] float16 residual stream at last subject token of paraphrase_prompt",
            "icl3_subj": "[N, L, D] float16 residual stream at last subject token of icl3_prompt",
        },
        "models": {},
    }
    for mk, info in sorted(model_infos.items()):
        manifest["models"][mk] = {
            **info,
            "states_path": str(CACHE_DIR / f"{mk}_states.pt"),
            "weights_path": str(CACHE_DIR / f"{mk}_weights.pt"),
            "eval_path": str(CACHE_DIR / f"{mk}_eval.json"),
            "hop2_states_path": str(CACHE_DIR / f"{mk}_2hop.pt"),
            "hop2_eval_path": str(CACHE_DIR / f"{mk}_2hop_eval.json"),
        }
    with open(CACHE_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ======================================================================================
# Plato's Key Exact Baseline: Procrustes + Global Scale, Controls, mKNN, Chains, PCA, Dark
# ======================================================================================

def fit_procrustes_global(Z_A: np.ndarray, Z_B: np.ndarray) -> dict:
    """
    Exact Plato's Key closed-form map:
      z_hat_B = mu_B + s * (z_A - mu_A) @ R^T
    where Z_B_c^T Z_A_c = U Sigma V^T, R = U V^T (shape [D_B, D_A]), s = ||Z_B_c||_F / ||Z_A_c||_F.
    Here inputs Z_A [N, D_A], Z_B [N, D_B] have one key per row.
    """
    Z_A = Z_A.astype(np.float64)
    Z_B = Z_B.astype(np.float64)
    mu_A = Z_A.mean(axis=0)
    mu_B = Z_B.mean(axis=0)
    A_c = Z_A - mu_A
    B_c = Z_B - mu_B
    # Z_B_col Z_A_col^T = B_c^T @ A_c  shape [D_B, D_A]
    M = B_c.T @ A_c
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt  # [D_B, D_A], semi-orthogonal
    norm_A = np.linalg.norm(A_c, ord="fro")
    norm_B = np.linalg.norm(B_c, ord="fro")
    s = float(norm_B / max(norm_A, 1e-12))
    return {"mu_A": mu_A, "mu_B": mu_B, "R": R, "s": s}


def precompute_layer_qr(Z_tr: np.ndarray) -> dict:
    """Precompute mean, Frobenius norm, and reduced QR (Q, R) of Z_c^T once per layer (float32)."""
    Z = Z_tr.astype(np.float32)
    mu = Z.mean(axis=0)
    Z_c = Z - mu
    frob = float(np.linalg.norm(Z_c, ord="fro"))
    Q, R_qr = np.linalg.qr(Z_c.T, mode="reduced")  # Q: [D, r], R_qr: [r, N]
    return {"mu": mu, "frob": frob, "Q": Q, "R_qr": R_qr}


def fast_qr_procrustes_predict(qr_A: dict, qr_B: dict, Z_A_eval: np.ndarray) -> np.ndarray:
    """
    74x faster Procrustes prediction using precomputed per-layer reduced QR:
      M = B_c^T A_c = Q_B (R_B R_A^T) Q_A^T, small r x r SVD takes ~16ms vs 1450ms!
      Factored apply avoids forming D_B x D_A matrix.
    """
    Ur, _, Vrt = np.linalg.svd(qr_B["R_qr"] @ qr_A["R_qr"].T, full_matrices=False)
    s = qr_B["frob"] / max(qr_A["frob"], 1e-12)
    # pred = mu_B + s * (((Z_A - mu_A) @ Q_A) @ (Vrt.T @ Ur.T)) @ Q_B.T
    z_proj = (Z_A_eval.astype(np.float32) - qr_A["mu"]) @ qr_A["Q"]
    z_rot = z_proj @ (Vrt.T @ Ur.T)
    return qr_B["mu"] + s * (z_rot @ qr_B["Q"].T)


def apply_linear_map(z_A: np.ndarray, mp: dict) -> np.ndarray:
    z = z_A.astype(np.float64)
    return mp["mu_B"] + mp["s"] * ((z - mp["mu_A"]) @ mp["R"].T)


def fit_random_orthogonal_control(Z_A: np.ndarray, Z_B: np.ndarray, seed: int) -> dict:
    Z_A = Z_A.astype(np.float64)
    Z_B = Z_B.astype(np.float64)
    mu_A = Z_A.mean(axis=0)
    mu_B = Z_B.mean(axis=0)
    A_c = Z_A - mu_A
    B_c = Z_B - mu_B
    d_A, d_B = Z_A.shape[1], Z_B.shape[1]
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d_B, d_A))
    U, _, Vt = np.linalg.svd(G, full_matrices=False)
    R = U @ Vt
    s = float(np.linalg.norm(B_c, "fro") / max(np.linalg.norm(A_c, "fro"), 1e-12))
    return {"mu_A": mu_A, "mu_B": mu_B, "R": R, "s": s}


def fit_shuffled_procrustes_control(Z_A: np.ndarray, Z_B: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(Z_A.shape[0])
    return fit_procrustes_global(Z_A[perm], Z_B)


def precompute_pca_bases(Z_A: np.ndarray, Z_B: np.ndarray) -> dict:
    """Compute centered SVDs of Z_A and Z_B once for the entire separate-PCA curve."""
    Z_A = Z_A.astype(np.float64)
    Z_B = Z_B.astype(np.float64)
    mu_A = Z_A.mean(axis=0)
    mu_B = Z_B.mean(axis=0)
    A_c = Z_A - mu_A
    B_c = Z_B - mu_B
    _, _, Vt_A = np.linalg.svd(A_c, full_matrices=False)
    _, _, Vt_B = np.linalg.svd(B_c, full_matrices=False)
    return {"mu_A": mu_A, "mu_B": mu_B, "A_c": A_c, "B_c": B_c, "Vt_A": Vt_A, "Vt_B": Vt_B}


def fit_pca_procrustes_from_cache(pca_cache: dict, k: int) -> dict:
    """Plato's Key Sec. 2 separate-PCA compression at rank k using precomputed PCA bases."""
    Vt_A, Vt_B = pca_cache["Vt_A"], pca_cache["Vt_B"]
    k_eff = min(k, Vt_A.shape[0], Vt_B.shape[0])
    P_A = Vt_A[:k_eff]
    P_B = Vt_B[:k_eff]
    A_k = pca_cache["A_c"] @ P_A.T
    B_k = pca_cache["B_c"] @ P_B.T
    M = B_k.T @ A_k
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R_k = U @ Vt
    s_k = float(np.linalg.norm(B_k, "fro") / max(np.linalg.norm(A_k, "fro"), 1e-12))
    R_full = P_B.T @ R_k @ P_A
    return {"mu_A": pca_cache["mu_A"], "mu_B": pca_cache["mu_B"], "R": R_full, "s": s_k, "k": k_eff}


def fit_pca_procrustes(Z_A: np.ndarray, Z_B: np.ndarray, k: int) -> dict:
    return fit_pca_procrustes_from_cache(precompute_pca_bases(Z_A, Z_B), k)


def compute_mknn_grid(Z_A_layers: torch.Tensor, Z_B_layers: torch.Tensor, k: int = 10) -> np.ndarray:
    """
    Exact vectorized mutual k-nearest-neighbor similarity across ALL (L_A, L_B) pairs.
    Inputs: Z_A_layers [N, L_A, D_A], Z_B_layers [N, L_B, D_B]. Returns [L_A, L_B] float64.
    """
    n, L_A, _ = Z_A_layers.shape
    L_B = Z_B_layers.shape[1]
    k_eff = min(k, max(1, n - 1))

    def _knn_indicators(Z_layers: torch.Tensor, L: int) -> np.ndarray:
        G = np.zeros((L, n * n), dtype=np.float32)
        rows = np.arange(n)[:, None]
        for li in range(L):
            X = Z_layers[:, li, :].float().numpy()
            X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
            sim = X @ X.T
            np.fill_diagonal(sim, -np.inf)
            nn = np.argpartition(-sim, k_eff, axis=1)[:, :k_eff]
            mat = np.zeros((n, n), dtype=np.float32)
            mat[rows, nn] = 1.0
            G[li] = mat.reshape(-1)
        return G

    G_A = _knn_indicators(Z_A_layers, L_A)
    G_B = _knn_indicators(Z_B_layers, L_B)
    return ((G_A @ G_B.T) / float(n * k_eff)).astype(np.float64)


def build_counterfactual_pairs(
    facts: List[dict], shared_indices: Sequence[int]
) -> List[Tuple[int, int]]:
    """
    Deterministically pair each fact i in shared_indices (donor key) with a base prompt j
    in shared_indices from the SAME relation with distinct subject and distinct answer.
    Returns list of (donor_idx, base_idx).
    """
    by_rel: Dict[str, List[int]] = defaultdict(list)
    for idx in sorted(shared_indices):
        by_rel[facts[idx]["relation"]].append(idx)

    pairs: List[Tuple[int, int]] = []
    for rel in sorted(by_rel.keys()):
        idxs = by_rel[rel]
        m = len(idxs)
        if m < 2:
            continue
        for pos, donor_idx in enumerate(idxs):
            d_sub = facts[donor_idx]["subject_norm"]
            d_ans = facts[donor_idx]["answer"].strip().lower()
            # Deterministic offset walk
            chosen = None
            for step in range(1, m):
                cand = idxs[(pos + step) % m]
                if (
                    facts[cand]["subject_norm"] != d_sub
                    and facts[cand]["answer"].strip().lower() != d_ans
                ):
                    chosen = cand
                    break
            if chosen is not None:
                pairs.append((donor_idx, chosen))
    return pairs


def pretokenize_swap_batches(
    tok: Any,
    facts: List[dict],
    eval_B: List[dict],
    swap_pairs: Sequence[Tuple[int, int]],
    device: torch.device,
    batch_size: int = 32,
) -> List[dict]:
    """Pre-tokenize base prompts and compute left-padded subject positions once per pair set."""
    batches = []
    n = len(swap_pairs)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_pairs = swap_pairs[start:end]
        prompts = [facts[b_idx]["direct_prompt"] for _, b_idx in batch_pairs]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        t_pad = input_ids.shape[1]
        spos = []
        for _, b_idx in batch_pairs:
            pad_len = t_pad - eval_B[b_idx]["seq_len_direct"]
            pos = max(0, min(t_pad - 1, pad_len + eval_B[b_idx]["subj_tok_idx_direct"]))
            spos.append(pos)
        alias_first_sets = [
            _alias_first_token_ids(tok, facts[d_idx]["aliases"]) for d_idx, _ in batch_pairs
        ]
        batches.append({
            "start": start,
            "end": end,
            "batch_pairs": batch_pairs,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "subj_pos": torch.tensor(spos, dtype=torch.long, device=device),
            "alias_first_sets": alias_first_sets,
        })
    return batches


def eval_patched_swaps(
    model: Any,
    tok: Any,
    facts: List[dict],
    eval_B: List[dict],
    swap_pairs: Sequence[Tuple[int, int]],
    z_hat_mat: np.ndarray,
    layer_B: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 32,
    max_new_tokens: int = 6,
    fast_first_tok_only: bool = False,
    pretok_batches: Optional[List[dict]] = None,
) -> Tuple[float, List[int]]:
    """
    Patch recipient `model` at `layer_B` at the last subject token of each base prompt `j`,
    replacing it with `z_hat_mat[p]` (intended to elicit donor fact `i`'s answer).
    Uses exact two-stage decoding: only runs continuation steps on items whose first generated
    token matches an alias first token (100% exact vs full greedy decoding!).
    """
    if not swap_pairs:
        return 0.0, []

    layers = get_decoder_layers(model)
    target_layer = layers[layer_B]
    n = len(swap_pairs)
    outcomes = [0] * n

    batches = (
        pretok_batches
        if pretok_batches is not None
        else pretokenize_swap_batches(tok, facts, eval_B, swap_pairs, device, batch_size=batch_size)
    )

    ctx = {"active": False, "subj_pos": None, "z_patch": None}

    def patch_hook(_mod, _inp, out):
        if not ctx["active"]:
            return out
        if isinstance(out, tuple):
            h = out[0].clone()
            is_tup = True
        else:
            h = out.clone()
            is_tup = False
        if h.shape[1] <= 1:
            return out
        bsz = h.shape[0]
        b_ar = torch.arange(bsz, device=h.device)
        h[b_ar, ctx["subj_pos"], :] = ctx["z_patch"].to(device=h.device, dtype=h.dtype)
        return (h,) + out[1:] if is_tup else h

    handle = target_layer.register_forward_hook(patch_hook)
    try:
        for b in batches:
            start, end = b["start"], b["end"]
            batch_pairs = b["batch_pairs"]
            input_ids = b["input_ids"]
            attention_mask = b["attention_mask"]
            alias_first_sets = b["alias_first_sets"]

            ctx["subj_pos"] = b["subj_pos"]
            ctx["z_patch"] = torch.from_numpy(z_hat_mat[start:end]).to(device=device, dtype=dtype)
            ctx["active"] = True

            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=(not fast_first_tok_only),
                )
            ctx["active"] = False

            first_logits = out.logits[:, -1, :]
            first_toks = first_logits.argmax(dim=-1).tolist()

            if fast_first_tok_only:
                for bi, tok_id in enumerate(first_toks):
                    if tok_id in alias_first_sets[bi]:
                        outcomes[start + bi] = 1
                continue

            cand_bi = [
                bi for bi, tok_id in enumerate(first_toks)
                if tok_id in alias_first_sets[bi]
            ]
            if not cand_bi:
                continue

            gen_ids = _greedy_decode_from_prefill(
                model, first_logits, out.past_key_values, attention_mask, max_new_tokens=max_new_tokens
            )
            for bi in cand_bi:
                p_idx = start + bi
                d_idx = batch_pairs[bi][0]
                g_txt = tok.decode(gen_ids[bi].tolist(), skip_special_tokens=True)
                if check_answer_match(g_txt, facts[d_idx]["aliases"], facts[d_idx]["relation"]):
                    outcomes[p_idx] = 1
    finally:
        handle.remove()

    acc = float(sum(outcomes) / n)
    return acc, outcomes


def run_baseline(model_keys: Sequence[str]) -> dict:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    facts = dataset["facts"]

    # Load all cached evals & direct_subj states into CPU RAM (small: ~100MB/model)
    evals: Dict[str, dict] = {}
    states_direct: Dict[str, torch.Tensor] = {}
    model_infos: Dict[str, dict] = {}
    for mk in model_keys:
        with open(CACHE_DIR / f"{mk}_eval.json", "r", encoding="utf-8") as f:
            ev = json.load(f)
        evals[mk] = ev
        model_infos[mk] = ev["model_info"]
        st = torch.load(CACHE_DIR / f"{mk}_states.pt", map_location="cpu", weights_only=False)
        states_direct[mk] = st["direct_subj"]  # [N, L, D] float16
        del st

    splits_idx = {
        sp: [f["idx"] for f in facts if f["split"] == sp]
        for sp in ("train", "val", "test")
    }

    results: dict = {
        "meta": {
            "models": list(model_keys),
            "n_facts": len(facts),
            "split_counts": dataset["meta"]["split_counts"],
            "seed": SEED,
        },
        "model_summaries": {mk: evals[mk]["summary"] for mk in model_keys},
        "self_swap": {},
        "pairs": {},
        "chains": {},
    }

    # Group all evaluations by RECIPIENT model B so we only load each model ONCE!
    for mk_B in model_keys:
        t_rec = time.time()
        print(f"\n[baseline] === Recipient: {mk_B} ===")
        model_B, tok_B, device, dtype = load_model_and_tokenizer(Path(model_infos[mk_B]["local_path"]))
        eval_B = evals[mk_B]["facts"]
        L_B = model_infos[mk_B]["num_layers"]
        Z_B_all = states_direct[mk_B]  # [N, L_B, D_B]

        # 1. Self-swap sweep on B (val split -> pick best l_B -> eval test split)
        b_val_rec = [i for i in splits_idx["val"] if eval_B[i]["direct_correct"]]
        b_test_rec = [i for i in splits_idx["test"] if eval_B[i]["direct_correct"]]
        self_val_pairs = build_counterfactual_pairs(facts, b_val_rec)
        self_test_pairs = build_counterfactual_pairs(facts, b_test_rec)
        self_val_pretok = pretokenize_swap_batches(tok_B, facts, eval_B, self_val_pairs, device)

        self_val_curve = {}
        best_self_l = 0
        best_self_val = -1.0
        for l_b in range(L_B):
            z_patch = np.stack([Z_B_all[d_idx, l_b].float().numpy() for d_idx, _ in self_val_pairs], axis=0)
            acc, _ = eval_patched_swaps(
                model_B, tok_B, facts, eval_B, self_val_pairs, z_patch, l_b, device, dtype,
                fast_first_tok_only=True, pretok_batches=self_val_pretok,
            )
            self_val_curve[str(l_b)] = round(acc, 4)
            if acc > best_self_val:
                best_self_val = acc
                best_self_l = l_b

        z_self_val = np.stack([Z_B_all[d_idx, best_self_l].float().numpy() for d_idx, _ in self_val_pairs], axis=0)
        self_val_exact, _ = eval_patched_swaps(
            model_B, tok_B, facts, eval_B, self_val_pairs, z_self_val, best_self_l, device, dtype,
            pretok_batches=self_val_pretok,
        )
        z_self_test = np.stack([Z_B_all[d_idx, best_self_l].float().numpy() for d_idx, _ in self_test_pairs], axis=0)
        self_test_exact, self_test_outcomes = eval_patched_swaps(
            model_B, tok_B, facts, eval_B, self_test_pairs, z_self_test, best_self_l, device, dtype
        )
        results["self_swap"][mk_B] = {
            "best_layer": best_self_l,
            "best_layer_rel_depth": round(best_self_l / L_B, 4),
            "val_acc": round(self_val_exact, 4),
            "test_acc": round(self_test_exact, 4),
            "n_val_pairs": len(self_val_pairs),
            "n_test_pairs": len(self_test_pairs),
            "val_curve_first_tok": self_val_curve,
            "test_pairs": self_test_pairs,
            "test_outcomes": self_test_outcomes,
        }
        print(
            f"  [self-swap] {mk_B}: best l_B={best_self_l}/{L_B} ({best_self_l/L_B:.2f}), "
            f"val={self_val_exact:.3f}, test={self_test_exact:.3f} (n={len(self_test_pairs)})"
        )

        # 2. All ordered pairs (A -> B)
        for mk_A in model_keys:
            if mk_A == mk_B:
                continue
            pair_key = f"{mk_A}->{mk_B}"
            t_p = time.time()
            eval_A = evals[mk_A]["facts"]
            L_A = model_infos[mk_A]["num_layers"]
            Z_A_all = states_direct[mk_A]

            shared_train = [
                i for i in splits_idx["train"]
                if eval_A[i]["direct_correct"] and eval_B[i]["direct_correct"]
            ]
            shared_val = [
                i for i in splits_idx["val"]
                if eval_A[i]["direct_correct"] and eval_B[i]["direct_correct"]
            ]
            shared_test = [
                i for i in splits_idx["test"]
                if eval_A[i]["direct_correct"] and eval_B[i]["direct_correct"]
            ]
            val_pairs = build_counterfactual_pairs(facts, shared_val)
            test_pairs = build_counterfactual_pairs(facts, shared_test)
            val_pretok = pretokenize_swap_batches(tok_B, facts, eval_B, val_pairs, device)
            test_pretok = pretokenize_swap_batches(tok_B, facts, eval_B, test_pairs, device)

            # 2a. Fast QR-accelerated coarse-to-fine layer search on VAL split ONLY
            cand_A = list(range(1, max(3, int(0.68 * L_A)), 2))
            cand_B = list(range(1, max(3, int(0.65 * L_B)), 2))
            if best_self_l not in cand_B and best_self_l < L_B:
                cand_B.append(best_self_l)
                cand_B.sort()

            val_donor_idxs = [d for d, _ in val_pairs]
            test_donor_idxs = [d for d, _ in test_pairs]

            # Precompute reduced QR ONCE per candidate layer (~48ms each instead of 1450ms x 145!)
            qr_cache_A: Dict[int, dict] = {}
            qr_cache_B: Dict[int, dict] = {}
            z_val_A_cache: Dict[int, np.ndarray] = {}

            def _get_qr_A(la: int) -> dict:
                if la not in qr_cache_A:
                    qr_cache_A[la] = precompute_layer_qr(Z_A_all[shared_train, la].float().numpy())
                    z_val_A_cache[la] = Z_A_all[val_donor_idxs, la].float().numpy()
                return qr_cache_A[la]

            def _get_qr_B(lb: int) -> dict:
                if lb not in qr_cache_B:
                    qr_cache_B[lb] = precompute_layer_qr(Z_B_all[shared_train, lb].float().numpy())
                return qr_cache_B[lb]

            coarse_scores: List[Tuple[float, int, int]] = []
            for l_b in cand_B:
                qB = _get_qr_B(l_b)
                for l_a in cand_A:
                    qA = _get_qr_A(l_a)
                    z_hat = fast_qr_procrustes_predict(qA, qB, z_val_A_cache[l_a])
                    acc, _ = eval_patched_swaps(
                        model_B, tok_B, facts, eval_B, val_pairs, z_hat, l_b, device, dtype,
                        fast_first_tok_only=True, pretok_batches=val_pretok,
                    )
                    coarse_scores.append((acc, l_a, l_b))

            coarse_scores.sort(key=lambda x: x[0], reverse=True)
            # Fine stride-1 refinement around top-3 coarse peaks using fast QR-SVD first,
            # then re-scoring the top-6 fine peaks with exact full-SVD Plato's Key Procrustes!
            fine_candidates = set()
            for _, la0, lb0 in coarse_scores[:3]:
                for dla in (-1, 0, 1):
                    for dlb in (-1, 0, 1):
                        la = la0 + dla
                        lb = lb0 + dlb
                        if 0 <= la < L_A and 0 <= lb < L_B:
                            fine_candidates.add((la, lb))

            fine_qr_scores: List[Tuple[float, int, int]] = []
            for la, lb in sorted(fine_candidates):
                z_hat = fast_qr_procrustes_predict(_get_qr_A(la), _get_qr_B(lb), z_val_A_cache[la])
                acc, _ = eval_patched_swaps(
                    model_B, tok_B, facts, eval_B, val_pairs, z_hat, lb, device, dtype,
                    fast_first_tok_only=False, pretok_batches=val_pretok,
                )
                fine_qr_scores.append((acc, la, lb))
            fine_qr_scores.sort(key=lambda x: x[0], reverse=True)

            # Final exact full-SVD Procrustes selection over top-6 fine candidates on VAL split
            best_val_acc = -1.0
            best_la, best_lb = fine_qr_scores[0][1], fine_qr_scores[0][2]
            for _, la, lb in fine_qr_scores[:6]:
                ZA_tr_f = Z_A_all[shared_train, la].float().numpy()
                ZB_tr_f = Z_B_all[shared_train, lb].float().numpy()
                mp_f = fit_procrustes_global(ZA_tr_f, ZB_tr_f)
                z_hat = apply_linear_map(z_val_A_cache[la], mp_f)
                acc, _ = eval_patched_swaps(
                    model_B, tok_B, facts, eval_B, val_pairs, z_hat, lb, device, dtype,
                    fast_first_tok_only=False, pretok_batches=val_pretok,
                )
                if acc > best_val_acc:
                    best_val_acc = acc
                    best_la, best_lb = la, lb

            # 2b. Evaluate exact Plato's Key baseline + self-swap + controls on TEST at (best_la, best_lb)
            ZA_tr = Z_A_all[shared_train, best_la].float().numpy()
            ZB_tr = Z_B_all[shared_train, best_lb].float().numpy()
            ZA_te = Z_A_all[test_donor_idxs, best_la].float().numpy()
            ZB_te_self = Z_B_all[test_donor_idxs, best_lb].float().numpy()

            mp_base = fit_procrustes_global(ZA_tr, ZB_tr)
            z_hat_base = apply_linear_map(ZA_te, mp_base)
            test_acc, test_outcomes = eval_patched_swaps(
                model_B, tok_B, facts, eval_B, test_pairs, z_hat_base, best_lb, device, dtype,
                pretok_batches=test_pretok,
            )

            # Self-swap at the EXACT same l_B and EXACT same test_pairs (apples-to-apples retention!)
            self_at_lb_acc, self_at_lb_outcomes = eval_patched_swaps(
                model_B, tok_B, facts, eval_B, test_pairs, ZB_te_self, best_lb, device, dtype,
                pretok_batches=test_pretok,
            )

            # Controls: random orthogonal + shuffled Procrustes
            mp_rand = fit_random_orthogonal_control(ZA_tr, ZB_tr, seed=SEED + hash(pair_key) % 10000)
            rand_acc, rand_outcomes = eval_patched_swaps(
                model_B, tok_B, facts, eval_B, test_pairs, apply_linear_map(ZA_te, mp_rand), best_lb, device, dtype,
                pretok_batches=test_pretok,
            )
            mp_shuf = fit_shuffled_procrustes_control(ZA_tr, ZB_tr, seed=SEED + 17 + hash(pair_key) % 10000)
            shuf_acc, shuf_outcomes = eval_patched_swaps(
                model_B, tok_B, facts, eval_B, test_pairs, apply_linear_map(ZA_te, mp_shuf), best_lb, device, dtype,
                pretok_batches=test_pretok,
            )

            # 2c. Exact vectorized mKNN layer selection on ALL (l_A, l_B) using shared_val
            mknn_mat = compute_mknn_grid(Z_A_all[shared_val], Z_B_all[shared_val], k=10)
            flat_idx = int(np.argmax(mknn_mat))
            mknn_la, mknn_lb = divmod(flat_idx, L_B)
            best_mknn = float(mknn_mat[mknn_la, mknn_lb])
            mp_mknn = fit_procrustes_global(
                Z_A_all[shared_train, mknn_la].float().numpy(),
                Z_B_all[shared_train, mknn_lb].float().numpy(),
            )
            mknn_test_acc, mknn_outcomes = eval_patched_swaps(
                model_B, tok_B, facts, eval_B, test_pairs,
                apply_linear_map(Z_A_all[test_donor_idxs, mknn_la].float().numpy(), mp_mknn),
                mknn_lb, device, dtype, pretok_batches=test_pretok,
            )

            # 2d. Separate-PCA compression curve at (best_la, best_lb) (precomputed PCA bases!)
            pca_bases = precompute_pca_bases(ZA_tr, ZB_tr)
            pca_curve = {}
            for k_dim in (16, 32, 64, 128, 256, 512):
                mp_pca = fit_pca_procrustes_from_cache(pca_bases, k=k_dim)
                z_pca = apply_linear_map(ZA_te, mp_pca)
                acc_k, out_k = eval_patched_swaps(
                    model_B, tok_B, facts, eval_B, test_pairs, z_pca, best_lb, device, dtype,
                    pretok_batches=test_pretok,
                )
                pca_curve[str(k_dim)] = {
                    "test_acc": round(acc_k, 4),
                    "rel_to_full": round(acc_k / max(test_acc, 1e-9), 4),
                    "outcomes": out_k,
                }

            # 2e. Dark-memory recovery (Plato's Key Sec. 3):
            # Evaluate on TEST facts (and also VAL+TEST non-train for larger sample size, reported separately!)
            dark_results = {}
            for scope_name, scope_indices in [
                ("test_only", splits_idx["test"]),
                ("val_plus_test", splits_idx["val"] + splits_idx["test"]),
            ]:
                misses = [
                    i for i in scope_indices
                    if eval_A[i]["direct_correct"] and (not eval_B[i]["direct_correct"])
                ]
                dark_mems = [i for i in misses if eval_B[i]["is_dark_memory_candidate"]]
                non_dark_misses = [i for i in misses if not eval_B[i]["is_dark_memory_candidate"]]

                if dark_mems:
                    dm_pairs = [(i, i) for i in dark_mems]  # same-subject unlock!
                    ZA_dm = Z_A_all[dark_mems, best_la].float().numpy()
                    dm_rec, dm_out = eval_patched_swaps(
                        model_B, tok_B, facts, eval_B, dm_pairs, apply_linear_map(ZA_dm, mp_base), best_lb, device, dtype
                    )
                    dm_rand, dm_rand_out = eval_patched_swaps(
                        model_B, tok_B, facts, eval_B, dm_pairs, apply_linear_map(ZA_dm, mp_rand), best_lb, device, dtype
                    )
                    dm_shuf, dm_shuf_out = eval_patched_swaps(
                        model_B, tok_B, facts, eval_B, dm_pairs, apply_linear_map(ZA_dm, mp_shuf), best_lb, device, dtype
                    )
                else:
                    dm_rec, dm_out = 0.0, []
                    dm_rand, dm_rand_out = 0.0, []
                    dm_shuf, dm_shuf_out = 0.0, []

                if non_dark_misses:
                    nd_pairs = [(i, i) for i in non_dark_misses]
                    ZA_nd = Z_A_all[non_dark_misses, best_la].float().numpy()
                    nd_rec, nd_out = eval_patched_swaps(
                        model_B, tok_B, facts, eval_B, nd_pairs, apply_linear_map(ZA_nd, mp_base), best_lb, device, dtype
                    )
                else:
                    nd_rec, nd_out = 0.0, []

                dark_results[scope_name] = {
                    "n_misses": len(misses),
                    "n_dark_memories": len(dark_mems),
                    "dark_fraction_of_misses": round(len(dark_mems) / max(len(misses), 1), 4),
                    "dark_fact_indices": dark_mems,
                    "recovery_acc": round(dm_rec, 4),
                    "n_recovered": int(sum(dm_out)),
                    "outcomes": dm_out,
                    "random_control_acc": round(dm_rand, 4),
                    "random_control_outcomes": dm_rand_out,
                    "shuffled_control_acc": round(dm_shuf, 4),
                    "shuffled_control_outcomes": dm_shuf_out,
                    "n_non_dark_misses": len(non_dark_misses),
                    "non_dark_recovery_acc": round(nd_rec, 4),
                    "non_dark_outcomes": nd_out,
                }

            retention = test_acc / max(self_at_lb_acc, 1e-9)
            results["pairs"][pair_key] = {
                "donor": mk_A,
                "recipient": mk_B,
                "n_shared_train": len(shared_train),
                "n_shared_val": len(shared_val),
                "n_shared_test": len(shared_test),
                "n_val_pairs": len(val_pairs),
                "n_test_pairs": len(test_pairs),
                "shared_train_indices": shared_train,
                "shared_val_indices": shared_val,
                "shared_test_indices": shared_test,
                "val_pairs": val_pairs,
                "test_pairs": test_pairs,
                "best_layer_A": best_la,
                "best_layer_B": best_lb,
                "best_layer_A_rel_depth": round(best_la / L_A, 4),
                "best_layer_B_rel_depth": round(best_lb / L_B, 4),
                "procrustes_global_scale": round(mp_base["s"], 6),
                "val_transfer_acc": round(best_val_acc, 4),
                "test_transfer_acc": round(test_acc, 4),
                "test_outcomes": test_outcomes,
                "self_swap_at_lB_test_acc": round(self_at_lb_acc, 4),
                "self_swap_at_lB_outcomes": self_at_lb_outcomes,
                "retention_vs_self_swap": round(retention, 4),
                "random_control_test_acc": round(rand_acc, 4),
                "random_control_outcomes": rand_outcomes,
                "shuffled_control_test_acc": round(shuf_acc, 4),
                "shuffled_control_outcomes": shuf_outcomes,
                "mknn_selection": {
                    "layer_A": mknn_la,
                    "layer_B": mknn_lb,
                    "layer_A_rel_depth": round(mknn_la / L_A, 4),
                    "layer_B_rel_depth": round(mknn_lb / L_B, 4),
                    "val_mknn_score": round(best_mknn, 4),
                    "test_transfer_acc": round(mknn_test_acc, 4),
                    "test_outcomes": mknn_outcomes,
                },
                "separate_pca_curve": pca_curve,
                "dark_memory": dark_results,
            }
            print(
                f"  [{pair_key}] l=({best_la},{best_lb}) [{best_la/L_A:.2f},{best_lb/L_B:.2f}] | "
                f"test={test_acc:.3f} (self={self_at_lb_acc:.3f}, ret={retention:.2f}) | "
                f"rand={rand_acc:.3f} shuf={shuf_acc:.3f} | mKNN=({mknn_la},{mknn_lb})->{mknn_test_acc:.3f} | "
                f"PCA256={pca_curve['256']['test_acc']:.3f} | "
                f"dark(v+t)={dark_results['val_plus_test']['n_recovered']}/{dark_results['val_plus_test']['n_dark_memories']} "
                f"({dark_results['val_plus_test']['recovery_acc']:.3f}) [{time.time() - t_p:.1f}s]"
            )

        del model_B
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # 3. 2-Hop Model Chains (A -> B -> C), grouped by endpoint recipient C
    print("\n[baseline] === Evaluating 2-Hop Model Chains (A -> B -> C) ===")
    chain_map_cache: Dict[Tuple[str, int, str, int], dict] = {}

    def _get_chain_map(m1: str, l1: int, m2: str, l2: int) -> dict:
        key = (m1, l1, m2, l2)
        if key not in chain_map_cache:
            e1, e2 = evals[m1]["facts"], evals[m2]["facts"]
            tr = [i for i in splits_idx["train"] if e1[i]["direct_correct"] and e2[i]["direct_correct"]]
            chain_map_cache[key] = fit_procrustes_global(
                states_direct[m1][tr, l1].float().numpy(),
                states_direct[m2][tr, l2].float().numpy(),
            )
        return chain_map_cache[key]

    for mk_C in model_keys:
        model_C, tok_C, device, dtype = load_model_and_tokenizer(Path(model_infos[mk_C]["local_path"]))
        eval_C = evals[mk_C]["facts"]
        Z_C_all = states_direct[mk_C]

        for mk_A in model_keys:
            if mk_A == mk_C:
                continue
            for mk_B in model_keys:
                if mk_B in (mk_A, mk_C):
                    continue
                chain_key = f"{mk_A}->{mk_B}->{mk_C}"
                p_AB = results["pairs"][f"{mk_A}->{mk_B}"]
                p_BC = results["pairs"][f"{mk_B}->{mk_C}"]

                # Per Plato's Key Sec. 2: recipient layer of (A->B) is donor layer of (B->C)
                l_A = p_AB["best_layer_A"]
                l_B = p_AB["best_layer_B"]
                l_C = p_BC["best_layer_B"]

                eval_A = evals[mk_A]["facts"]
                eval_B = evals[mk_B]["facts"]
                Z_A_all = states_direct[mk_A]

                te_ABC = [
                    i for i in splits_idx["test"]
                    if eval_A[i]["direct_correct"] and eval_B[i]["direct_correct"] and eval_C[i]["direct_correct"]
                ]
                te_pairs = build_counterfactual_pairs(facts, te_ABC)
                if not te_pairs:
                    continue
                te_donors = [d for d, _ in te_pairs]

                mp_AB = _get_chain_map(mk_A, l_A, mk_B, l_B)
                mp_BC = _get_chain_map(mk_B, l_B, mk_C, l_C)
                mp_AC = _get_chain_map(mk_A, l_A, mk_C, l_C)

                # Chained prediction without running B!
                z_mid = apply_linear_map(Z_A_all[te_donors, l_A].float().numpy(), mp_AB)
                z_chain = apply_linear_map(z_mid, mp_BC)
                z_direct = apply_linear_map(Z_A_all[te_donors, l_A].float().numpy(), mp_AC)

                chain_acc, chain_out = eval_patched_swaps(
                    model_C, tok_C, facts, eval_C, te_pairs, z_chain, l_C, device, dtype
                )
                direct_acc, direct_out = eval_patched_swaps(
                    model_C, tok_C, facts, eval_C, te_pairs, z_direct, l_C, device, dtype
                )
                ratio = chain_acc / max(direct_acc, 1e-9)
                results["chains"][chain_key] = {
                    "layers": [l_A, l_B, l_C],
                    "n_test_pairs": len(te_pairs),
                    "chain_acc": round(chain_acc, 4),
                    "direct_acc": round(direct_acc, 4),
                    "chain_retention": round(ratio, 4),
                    "chain_outcomes": chain_out,
                    "direct_outcomes": direct_out,
                }
                print(
                    f"  [chain] {chain_key} l=({l_A},{l_B},{l_C}): "
                    f"chain={chain_acc:.3f} vs direct={direct_acc:.3f} (ratio={ratio:.3f}, n={len(te_pairs)})"
                )

        del model_C
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # 4. Aggregate medians (matching Plato's Key headline numbers)
    pair_vals = list(results["pairs"].values())
    if pair_vals:
        def _med(xs): return round(float(np.median(xs)), 4)
        results["aggregate"] = {
            "n_pairs": len(pair_vals),
            "median_test_transfer_acc": _med([p["test_transfer_acc"] for p in pair_vals]),
            "median_self_swap_at_lB_acc": _med([p["self_swap_at_lB_test_acc"] for p in pair_vals]),
            "median_retention_vs_self_swap": _med([p["retention_vs_self_swap"] for p in pair_vals]),
            "median_random_control_acc": _med([p["random_control_test_acc"] for p in pair_vals]),
            "median_shuffled_control_acc": _med([p["shuffled_control_test_acc"] for p in pair_vals]),
            "median_mknn_transfer_acc": _med([p["mknn_selection"]["test_transfer_acc"] for p in pair_vals]),
            "median_pca_rel_to_full": {
                str(k): _med([p["separate_pca_curve"][str(k)]["rel_to_full"] for p in pair_vals])
                for k in (16, 32, 64, 128, 256, 512)
            },
            "median_chain_retention": _med([c["chain_retention"] for c in results["chains"].values()]) if results["chains"] else None,
            "dark_memory_pooled_test_only": {
                "n_misses": sum(p["dark_memory"]["test_only"]["n_misses"] for p in pair_vals),
                "n_dark": sum(p["dark_memory"]["test_only"]["n_dark_memories"] for p in pair_vals),
                "n_recovered": sum(p["dark_memory"]["test_only"]["n_recovered"] for p in pair_vals),
                "recovery_rate": round(
                    sum(p["dark_memory"]["test_only"]["n_recovered"] for p in pair_vals)
                    / max(sum(p["dark_memory"]["test_only"]["n_dark_memories"] for p in pair_vals), 1),
                    4,
                ),
            },
            "dark_memory_pooled_val_plus_test": {
                "n_misses": sum(p["dark_memory"]["val_plus_test"]["n_misses"] for p in pair_vals),
                "n_dark": sum(p["dark_memory"]["val_plus_test"]["n_dark_memories"] for p in pair_vals),
                "n_recovered": sum(p["dark_memory"]["val_plus_test"]["n_recovered"] for p in pair_vals),
                "recovery_rate": round(
                    sum(p["dark_memory"]["val_plus_test"]["n_recovered"] for p in pair_vals)
                    / max(sum(p["dark_memory"]["val_plus_test"]["n_dark_memories"] for p in pair_vals), 1),
                    4,
                ),
            },
        }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[baseline] Saved {RESULTS_PATH}")
    if "aggregate" in results:
        print("[baseline] Aggregate summary:", json.dumps(results["aggregate"], indent=2))
    return results


def discover_ready_models() -> List[Tuple[str, Path]]:
    ready = []
    if not HF_CACHE_DIR.exists():
        return ready
    for d in sorted(HF_CACHE_DIR.iterdir()):
        if d.is_dir() and (d / "config.json").exists():
            # Verify weights exist (safetensors or bin)
            has_weights = any(d.glob("*.safetensors")) or any(d.glob("*.bin"))
            if has_weights:
                ready.append((d.name, d))
    return ready


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["extract", "baseline", "all"], default="all")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model folder names in hf_cache/")
    parser.add_argument("--force", action="store_true", help="Re-extract even if cached")
    args = parser.parse_args()

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(DATASET_2HOP_PATH, "r", encoding="utf-8") as f:
        dataset_2hop = json.load(f)

    ready = discover_ready_models()
    if args.models:
        wanted = set(args.models)
        ready = [(k, p) for k, p in ready if k in wanted]
    if not ready:
        raise RuntimeError(f"No ready models found in {HF_CACHE_DIR}")

    print(f"Discovered ready models: {[k for k, _ in ready]}")

    model_infos: Dict[str, dict] = {}
    if args.mode in ("extract", "all"):
        for mk, mdir in ready:
            info = extract_single_model(mk, mdir, dataset, dataset_2hop, force=args.force)
            model_infos[mk] = info
            # Update manifest incrementally after each model so peers can inspect immediately!
            # Merge with any previously cached models too
            all_cached = {}
            for ep in sorted(CACHE_DIR.glob("*_eval.json")):
                if ep.name.endswith("_2hop_eval.json"):
                    continue
                with open(ep, "r", encoding="utf-8") as f:
                    ev = json.load(f)
                all_cached[ev["model_info"]["model_key"]] = ev["model_info"]
            update_cache_manifest(dataset, dataset_2hop, all_cached)

    if args.mode in ("baseline", "all"):
        mkeys = [k for k, _ in ready]
        run_baseline(mkeys)


if __name__ == "__main__":
    main()
