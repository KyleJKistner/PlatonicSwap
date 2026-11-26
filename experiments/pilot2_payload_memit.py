#!/usr/bin/env python3
"""
pilot2_payload_memit.py — Breakthrough Pilot 2:
"Pointer or Payload?" + Donor-Free Platonic MEMIT Weight Compilation.

Tests whether brand-new PRIVATE knowledge (two randomly assigned 32-entity -> 8-guild
tables f and f', drawn AFTER freezing the cross-family alignment map on `dataset.json`)
taught ONLY to Donor A (`qwen2.5-3b`, D=2048) can be:
  - Test 2A (Live "Pointer or Payload?" Crossing):
      Transferred zero-shot across families into frozen Recipient B (`smollm2-1.7b`, D=2048)
      at BOTH:
        (i)  Site 1: Early Subject Key `(l_A=17, l_B=9, last_subj)` (exact Plato's Key site;
             Donor A LoRA masked strictly to `last_subj` so binding MUST pass through key site!)
        (ii) Site 2: Mid-Late Attribute Payload `(l_A=27, l_B=18, last_prompt)` (75% depth,
             6 full layers pre-unembedding) & `(l_A=30, l_B=20, last_prompt)` (83% depth, 4 layers pre-unembedding)
      Evaluating BOTH Full-State Swap AND Delta-Teleportation (`z_B + s * R * (z_A^edit - z_A^base)`),
      plus `matched_f`, `matched_f_prime` (full derangement flip!), `shuffled_donor`, `random_rotation`,
      and `unmodified_B` on UNSEEN PARAPHRASE TEMPLATES (both 8-way choice & unrestricted greedy EM).
  - Test 2B (Donor-Free Platonic MEMIT Compile — THROW DONOR A AWAY!):
      Compiles the teleported deltas directly into solo Recipient B's native `mlp.down_proj`
      weight matrix in closed form via exact CPU float64 Woodbury MEMIT (0 gradient steps on B,
      Donor A completely deleted from memory!), evaluating solo B on:
        (a) unseen paraphrases of all 32 entities for BOTH f and f',
        (b) 8-way choice AND unrestricted greedy exact match,
        (c) locality retention on all 196 `dataset.json` standard test facts!

Hardware guardrails (per Fable 5.2 Max & Gemini 4 Carbon):
  - Donor A tiny LoRA backward passes run in `float32` on MPS/CPU (zero bf16 grad bugs!).
  - All Procrustes SVDs and Woodbury MEMIT linear solves run in `float64` on CPU.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    locate_last_subject_token,
)
from run_systems_dark import _json_clean, eval_multi_regime_swaps

OUTPUT_PATH = LAB_DIR / "results_pilot2.json"

DONOR_KEY = "qwen2.5-3b"
RECIP_KEY = "smollm2-1.7b"

# 8 ordinary single-token guild labels verified 1-token with leading space in Qwen2.5, SmolLM2, AND Falcon3!
GUILD_LABELS: List[str] = [
    "Cedar", "Flint", "Silver", "Copper", "Bronze", "Coral", "Ruby", "Pearl",
]

# 32 familiar historical/scientific/literary entities (100% disjoint from dataset.json train/val/test!)
CANDIDATE_ENTITIES_32: List[str] = [
    "Ada Lovelace", "Wolfgang Mozart", "Jane Austen", "Mark Twain",
    "Charles Dickens", "Ernest Hemingway", "George Orwell", "Virginia Woolf",
    "Plato", "Julius Caesar", "Martin Luther King", "Winston Churchill",
    "Amelia Earhart", "Neil Armstrong", "Stephen Hawking", "Richard Feynman",
    "Carl Sagan", "Rosalind Franklin", "Johannes Kepler", "Pythagoras",
    "Homer", "Dante Alighieri", "Grace Hopper", "Katherine Johnson",
    "Emily Dickinson", "Alexander Fleming", "Gregor Mendel", "Oscar Wilde",
    "James Clerk Maxwell", "John von Neumann", "Max Planck", "Claude Shannon",
]

TRAIN_TEMPLATES: List[Tuple[str, str]] = [
    ("The official guild of ", " is"),
    ("The guild assigned to ", " is"),
]

TEST_PARAPHRASE_TEMPLATES: List[Tuple[str, str]] = [
    ("In the registry, the guild of ", " is"),
    ("According to the archive, ", " belongs to guild"),
]


# =============================================================================
# 1. Private Table Generation (f and Deranged f', Drawn AFTER Map Freeze)
# =============================================================================


def build_private_tables(
    entities: Sequence[str],
    labels: Sequence[str],
    seed_f: int = 20261201,
    seed_f_prime: int = 20261202,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build two balanced 32->8 tables (4 entities/guild) where f'(e) != f(e) for ALL 32 entities."""
    assert len(entities) == 32 and len(labels) == 8
    rng1 = random.Random(seed_f)
    pool1 = list(labels) * 4
    rng1.shuffle(pool1)
    table_f = {e: pool1[i] for i, e in enumerate(entities)}

    # Draw balanced derangement f' with f'(e) != f(e) for all e
    rng2 = random.Random(seed_f_prime)
    for _ in range(1000):
        pool2 = list(labels) * 4
        rng2.shuffle(pool2)
        if all(pool2[i] != table_f[e] for i, e in enumerate(entities)):
            table_fp = {e: pool2[i] for i, e in enumerate(entities)}
            return table_f, table_fp
    raise RuntimeError("Failed to sample balanced derangement f'")


def make_prompt_records(
    entities: Sequence[str],
    table: Dict[str, str],
    templates: Sequence[Tuple[str, str]],
) -> List[dict]:
    recs = []
    for t_idx, (pre, suf) in enumerate(templates):
        for e_idx, ent in enumerate(entities):
            p = f"{pre}{ent}{suf}"
            recs.append({
                "entity_idx": e_idx,
                "template_idx": t_idx,
                "entity": ent,
                "label": table[ent],
                "prompt": p,
                "subj_char_span": [len(pre), len(pre) + len(ent)],
            })
    return recs


# =============================================================================
# 2. Position-Aware Layer-Restricted LoRA Module (Runs Cleanly in float32 on MPS)
# =============================================================================


class MaskedResidualLoRA(nn.Module):
    """Low-rank adapter attached to a transformer layer's MLP (`gate/up/down` or direct residual block).

    Attaches directly to a layer's output (or MLP) with optional token-position masking so
    Site-1 training can strictly constrain all updates to `last_subj` (guaranteeing the
    new knowledge passes through `(l_A, last_subj)` without leaking directly to `last_prompt`!).
    """

    def __init__(self, d_model: int, rank: int = 16, init_scale: float = 0.02):
        super().__init__()
        self.A = nn.Parameter(torch.randn(d_model, rank) * (1.0 / math.sqrt(d_model)))
        self.B = nn.Parameter(torch.zeros(rank, d_model))
        self.scale = 1.0 / math.sqrt(rank)
        self.active_mask: Optional[torch.Tensor] = None  # [B, T, 1] or None (all tokens)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, D]
        delta = (h.float() @ self.A) @ self.B * self.scale
        if self.active_mask is not None:
            delta = delta * self.active_mask.to(device=delta.device, dtype=delta.dtype)
        return h + delta.to(dtype=h.dtype)


class LayerRestrictedAdapterSuite:
    """Manages MaskedResidualLoRA modules on a contiguous subset of layers `layer_indices`."""

    def __init__(
        self,
        model: nn.Module,
        layer_indices: Sequence[int],
        rank: int = 16,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.layer_indices = list(layer_indices)
        d_model = int(model.config.hidden_size)
        self.adapters = nn.ModuleDict({
            str(l): MaskedResidualLoRA(d_model, rank=rank).to(device=device, dtype=torch.float32)
            for l in self.layer_indices
        })
        self.handles: List[Any] = []
        self.enabled = True

    def parameters(self):
        return self.adapters.parameters()

    def set_token_mask(self, mask: Optional[torch.Tensor]):
        for ad in self.adapters.values():
            ad.active_mask = mask

    def attach(self):
        self.detach()
        layers = get_decoder_layers(self.model)
        for l in self.layer_indices:
            ad = self.adapters[str(l)]

            def make_hook(adapter: MaskedResidualLoRA):
                def hook(_mod, _inp, out):
                    if not self.enabled:
                        return out
                    if isinstance(out, tuple):
                        h = out[0]
                        if h.ndim == 3:
                            return (adapter(h),) + out[1:]
                        return out
                    if out.ndim == 3:
                        return adapter(out)
                    return out

                return hook

            self.handles.append(layers[l].register_forward_hook(make_hook(ad)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# =============================================================================
# 3. Evaluation Helpers: 8-Way Choice + Unrestricted Greedy EM + Token Logprobs
# =============================================================================


def get_label_token_ids(tok: Any, labels: Sequence[str]) -> List[int]:
    ids = []
    for w in labels:
        t = tok.encode(" " + w, add_special_tokens=False)
        assert len(t) == 1, f"Expected single token for ' {w}', got {t}"
        ids.append(t[0])
    return ids


@torch.no_grad()
def eval_guild_records(
    model: nn.Module,
    tok: Any,
    records: Sequence[dict],
    labels: Sequence[str],
    device: torch.device,
    patch_layer: Optional[int] = None,
    patch_pos_kind: str = "subj",  # "subj" or "last"
    patch_replace: Optional[np.ndarray] = None,
    patch_add: Optional[np.ndarray] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Evaluate BOTH 8-way choice accuracy AND unrestricted greedy exact match on guild records."""
    label_tok_ids = get_label_token_ids(tok, labels)
    label_to_idx = {w: i for i, w in enumerate(labels)}
    n = len(records)

    choice_outcomes = [0] * n
    greedy_outcomes = [0] * n
    preds_8way: List[str] = [""] * n
    preds_greedy: List[str] = [""] * n
    target_probs: List[float] = [0.0] * n

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        brecs = records[start:end]
        prompts = [r["prompt"] for r in brecs]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        t_pad = input_ids.shape[1]
        bsz = end - start

        spos, lpos = [], [t_pad - 1] * bsz
        for r in brecs:
            si, sl = locate_last_subject_token(tok, r["prompt"], r["subj_char_span"])
            spos.append(max(0, min(t_pad - 1, (t_pad - sl) + si)))

        spec = None
        if patch_layer is not None and (patch_replace is not None or patch_add is not None):
            spec = systems.BatchPatchSpec()
            pos_use = spos if patch_pos_kind == "subj" else lpos
            if patch_replace is not None:
                spec.residual_replace[int(patch_layer)] = [
                    (pos_use, torch.from_numpy(np.asarray(patch_replace[start:end], dtype=np.float32)))
                ]
            if patch_add is not None:
                spec.residual_add[int(patch_layer)] = [
                    (pos_use, torch.from_numpy(np.asarray(patch_add[start:end], dtype=np.float32)))
                ]

        if spec is not None:
            with systems.PatchHookManager(model, spec):
                out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

        logits = out.logits[:, -1, :].float()
        probs_full = torch.softmax(logits, dim=-1)
        sub_logits = logits[:, label_tok_ids]  # [B, 8]
        best_8 = sub_logits.argmax(dim=-1).cpu().tolist()

        # Greedy 3-token decode for unrestricted EM
        gen_ids = _greedy_decode_from_prefill(
            model, out.logits[:, -1, :], out.past_key_values, attention_mask, max_new_tokens=3
        )

        for bi, r in enumerate(brecs):
            idx = start + bi
            tgt_label = r["label"]
            tgt_i = label_to_idx[tgt_label]
            p_label = labels[best_8[bi]]
            preds_8way[idx] = p_label
            choice_outcomes[idx] = int(best_8[bi] == tgt_i)
            target_probs[idx] = float(probs_full[bi, label_tok_ids[tgt_i]].item())

            g_txt = tok.decode(gen_ids[bi].tolist(), skip_special_tokens=True)
            preds_greedy[idx] = g_txt.strip()
            greedy_outcomes[idx] = int(check_answer_match(g_txt, [tgt_label], "guild"))

    return {
        "n": n,
        "choice8_acc": float(np.mean(choice_outcomes)),
        "greedy_em_acc": float(np.mean(greedy_outcomes)),
        "mean_target_prob": float(np.mean(target_probs)),
        "choice8_outcomes": choice_outcomes,
        "greedy_outcomes": greedy_outcomes,
        "preds_8way": preds_8way,
        "preds_greedy": preds_greedy,
    }


@torch.no_grad()
def extract_site_states(
    model: nn.Module,
    tok: Any,
    records: Sequence[dict],
    layer_idx: int,
    pos_kind: str,  # "subj" or "last"
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract post-block residual hidden states at `layer_idx` and `pos_kind` (float64 numpy)."""
    layers = get_decoder_layers(model)
    n = len(records)
    outs = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        brecs = records[start:end]
        prompts = [r["prompt"] for r in brecs]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        t_pad = input_ids.shape[1]
        bsz = end - start
        if pos_kind == "subj":
            pos = []
            for r in brecs:
                si, sl = locate_last_subject_token(tok, r["prompt"], r["subj_char_span"])
                pos.append(max(0, min(t_pad - 1, (t_pad - sl) + si)))
        else:
            pos = [t_pad - 1] * bsz

        buf = {}

        def cap_hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            vecs = torch.stack([h[b, pos[b], :] for b in range(h.shape[0])], dim=0)
            buf["x"] = vecs.detach().float().cpu().numpy()

        handle = layers[layer_idx].register_forward_hook(cap_hook)
        try:
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            handle.remove()
        outs.append(buf["x"])
    return np.concatenate(outs, axis=0).astype(np.float64)


@torch.no_grad()
def extract_mlp_down_proj_inputs(
    model: nn.Module,
    tok: Any,
    prompts: Sequence[str],
    char_spans: Sequence[Sequence[int]],
    layer_idx: int,
    pos_kind: str,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract exact post-SwiGLU hidden inputs k in R^{d_ff} entering `layers[layer_idx].mlp.down_proj`."""
    layers = get_decoder_layers(model)
    down_proj = layers[layer_idx].mlp.down_proj
    n = len(prompts)
    outs = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b_prompts = list(prompts[start:end])
        b_spans = char_spans[start:end]
        enc = tok(b_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        t_pad = input_ids.shape[1]
        bsz = end - start
        if pos_kind == "subj":
            pos = []
            for p, sp in zip(b_prompts, b_spans):
                si, sl = locate_last_subject_token(tok, p, sp)
                pos.append(max(0, min(t_pad - 1, (t_pad - sl) + si)))
        else:
            pos = [t_pad - 1] * bsz

        buf = {}

        def pre_hook(_mod, inp):
            k_tens = inp[0]  # [B, T, d_ff]
            vecs = torch.stack([k_tens[b, pos[b], :] for b in range(k_tens.shape[0])], dim=0)
            buf["k"] = vecs.detach().float().cpu().numpy()

        handle = down_proj.register_forward_pre_hook(pre_hook)
        try:
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            handle.remove()
        outs.append(buf["k"])
    return np.concatenate(outs, axis=0).astype(np.float64)


# =============================================================================
# 4. Exact Closed-Form Woodbury MEMIT Weight Solver (float64 on CPU, ~15ms!)
# =============================================================================


def solve_woodbury_memit(
    K_edit: np.ndarray,      # [M, d_ff]
    Delta_V: np.ndarray,     # [M, d_model]
    K_bg: np.ndarray,        # [N_bg, d_ff]
    lam_cov: float = 100.0,
    lam_iso: float = 1e-2,
) -> np.ndarray:
    """Exact closed-form MEMIT update minimizing:
        ||Delta_W K_edit^T - Delta_V^T||_F^2 + (lam_cov / N_bg) ||Delta_W K_bg^T||_F^2 + lam_iso ||Delta_W||_F^2
    Returns Delta_W of shape [d_model, d_ff] (exact PyTorch `down_proj.weight` shape!).

    Uses exact Woodbury identity on U = [K_edit^T, sqrt(lam_cov / N_bg) K_bg^T] of shape [d_ff, M + N_bg]
    so the linear solve is only (M + N_bg) x (M + N_bg) (~650x650 -> ~15ms in float64!).
    """
    K_e = np.asarray(K_edit, dtype=np.float64).T  # [d_ff, M]
    V_e = np.asarray(Delta_V, dtype=np.float64).T  # [d_model, M]
    K_0 = np.asarray(K_bg, dtype=np.float64).T  # [d_ff, N_bg]
    N_bg = max(K_0.shape[1], 1)

    U = np.concatenate([K_e, math.sqrt(lam_cov / N_bg) * K_0], axis=1)  # [d_ff, M + N_bg]
    # Small Gram matrix G = U^T U + lam_iso I
    G = U.T @ U
    G[np.diag_indices_from(G)] += lam_iso

    # Compute A^{-1} K_e = (1/lam_iso) * (K_e - U G^{-1} (U^T K_e))
    UtKe = U.T @ K_e  # [M + N_bg, M]
    sol = np.linalg.solve(G, UtKe)  # [M + N_bg, M]
    Ainv_Ke = (K_e - U @ sol) / lam_iso  # [d_ff, M]
    Delta_W = V_e @ Ainv_Ke.T  # [d_model, d_ff]
    return Delta_W


# =============================================================================
# 5. Layer-Restricted Adapter Training (Strictly float32, Fast ~40s on MPS!)
# =============================================================================


def train_layer_restricted_adapter(
    model: nn.Module,
    tok: Any,
    train_records: Sequence[dict],
    anchor_prompts: Sequence[str],
    layer_indices: Sequence[int],
    mask_to_subj_only: bool,
    device: torch.device,
    rank: int = 16,
    steps: int = 180,
    lr: float = 3e-3,
    batch_size: int = 16,
    kl_weight: float = 0.05,
    seed: int = 42,
) -> LayerRestrictedAdapterSuite:
    """Train a compact layer-restricted residual LoRA strictly on `layer_indices`.

    When `mask_to_subj_only=True`, the LoRA activates ONLY at `last_subj` (0 update on
    all suffix tokens including `last_prompt`), guaranteeing 100% of the new binding
    passes through the early subject key site!
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    suite = LayerRestrictedAdapterSuite(model, layer_indices=layer_indices, rank=rank, device=device)

    # Pre-tokenize train prompts + target single-token IDs
    prompts = [r["prompt"] for r in train_records]
    enc_all = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
    all_ids = enc_all["input_ids"]
    all_mask = enc_all["attention_mask"]
    t_pad = all_ids.shape[1]
    all_spos = []
    all_targets = []
    for r in train_records:
        si, sl = locate_last_subject_token(tok, r["prompt"], r["subj_char_span"])
        all_spos.append(max(0, min(t_pad - 1, (t_pad - sl) + si)))
        tid = tok.encode(" " + r["label"], add_special_tokens=False)
        assert len(tid) == 1
        all_targets.append(tid[0])
    all_targets_t = torch.tensor(all_targets, dtype=torch.long)

    # Pre-compute base logits on 32 locality anchor prompts for gentle KL preservation
    anc_enc = tok(list(anchor_prompts[:32]), return_tensors="pt", padding=True, add_special_tokens=True)
    anc_ids = anc_enc["input_ids"].to(device)
    anc_mask = anc_enc["attention_mask"].to(device)
    with torch.no_grad():
        base_anc_logp = F.log_softmax(
            model(input_ids=anc_ids, attention_mask=anc_mask, use_cache=False).logits[:, -1, :].float(), dim=-1
        )
        base_anc_p = base_anc_logp.exp()

    # Freeze base model completely
    for p in model.parameters():
        p.requires_grad_(False)
    suite.attach()

    opt = torch.optim.AdamW(suite.parameters(), lr=lr, weight_decay=1e-4)
    n = len(train_records)
    idxs = list(range(n))

    for step in range(steps):
        rng.shuffle(idxs)
        b_idx = idxs[:batch_size]
        ids_b = all_ids[b_idx].to(device)
        mask_b = all_mask[b_idx].to(device)
        tgt_b = all_targets_t[b_idx].to(device)

        if mask_to_subj_only:
            tmask = torch.zeros((len(b_idx), ids_b.shape[1], 1), dtype=torch.float32, device=device)
            for bi, gi in enumerate(b_idx):
                tmask[bi, all_spos[gi], 0] = 1.0
            suite.set_token_mask(tmask)
        else:
            suite.set_token_mask(None)

        logits = model(input_ids=ids_b, attention_mask=mask_b, use_cache=False).logits[:, -1, :].float()
        loss_ce = F.cross_entropy(logits, tgt_b)

        # Gentle locality KL on anchor prompts every 4 steps
        if kl_weight > 0.0 and (step % 4 == 0):
            suite.set_token_mask(None)
            anc_logits = model(input_ids=anc_ids, attention_mask=anc_mask, use_cache=False).logits[:, -1, :].float()
            anc_logp = F.log_softmax(anc_logits, dim=-1)
            loss_kl = F.kl_div(anc_logp, base_anc_p, reduction="batchmean")
            loss = loss_ce + kl_weight * loss_kl
        else:
            loss = loss_ce

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(suite.parameters(), 1.0)
        opt.step()

        if (step + 1) % 60 == 0 or step == steps - 1:
            acc_b = float((logits.argmax(dim=-1) == tgt_b).float().mean().item())
            print(f"    [train step {step+1:3d}/{steps}] ce={loss_ce.item():.4f} batch_greedy={acc_b:.3f}")

    suite.set_token_mask(None)
    return suite


def eval_with_subj_mask_if_needed(
    model: nn.Module,
    tok: Any,
    suite: LayerRestrictedAdapterSuite,
    records: Sequence[dict],
    labels: Sequence[str],
    mask_to_subj_only: bool,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate trained adapter suite cleanly (honoring mask_to_subj_only if set)."""
    if not mask_to_subj_only:
        suite.set_token_mask(None)
        return eval_guild_records(model, tok, records, labels, device=device)

    # Evaluate in one batch or per-batch with exact subject mask on prefill
    label_tok_ids = get_label_token_ids(tok, labels)
    label_to_idx = {w: i for i, w in enumerate(labels)}
    n = len(records)
    c8, gr = [0] * n, [0] * n
    with torch.no_grad():
        for start in range(0, n, 32):
            end = min(start + 32, n)
            brecs = records[start:end]
            enc = tok([r["prompt"] for r in brecs], return_tensors="pt", padding=True, add_special_tokens=True)
            ids = enc["input_ids"].to(device)
            am = enc["attention_mask"].to(device)
            t_pad = ids.shape[1]
            tmask = torch.zeros((end - start, t_pad, 1), dtype=torch.float32, device=device)
            for bi, r in enumerate(brecs):
                si, sl = locate_last_subject_token(tok, r["prompt"], r["subj_char_span"])
                pos = max(0, min(t_pad - 1, (t_pad - sl) + si))
                tmask[bi, pos, 0] = 1.0
            suite.set_token_mask(tmask)
            out = model(input_ids=ids, attention_mask=am, use_cache=True)
            # Disable mask for KV-cached single-token decode steps (adapters disabled during decode)
            suite.enabled = False
            gen_ids = _greedy_decode_from_prefill(model, out.logits[:, -1, :], out.past_key_values, am, max_new_tokens=3)
            suite.enabled = True
            sub_l = out.logits[:, -1, :].float()[:, label_tok_ids]
            b8 = sub_l.argmax(dim=-1).cpu().tolist()
            for bi, r in enumerate(brecs):
                idx = start + bi
                c8[idx] = int(b8[bi] == label_to_idx[r["label"]])
                g_txt = tok.decode(gen_ids[bi].tolist(), skip_special_tokens=True)
                gr[idx] = int(check_answer_match(g_txt, [r["label"]], "guild"))
    suite.set_token_mask(None)
    return {"n": n, "choice8_acc": float(np.mean(c8)), "greedy_em_acc": float(np.mean(gr)), "choice8_outcomes": c8, "greedy_outcomes": gr}


def extract_states_with_adapter(
    model: nn.Module,
    tok: Any,
    suite: LayerRestrictedAdapterSuite,
    records: Sequence[dict],
    layer_idx: int,
    pos_kind: str,
    mask_to_subj_only: bool,
    device: torch.device,
) -> np.ndarray:
    """Extract edited Donor A states at `(layer_idx, pos_kind)` honoring subject mask if active."""
    if not mask_to_subj_only:
        suite.set_token_mask(None)
        return extract_site_states(model, tok, records, layer_idx, pos_kind, device=device)

    layers = get_decoder_layers(model)
    outs = []
    with torch.no_grad():
        for start in range(0, len(records), 32):
            end = min(start + 32, len(records))
            brecs = records[start:end]
            enc = tok([r["prompt"] for r in brecs], return_tensors="pt", padding=True, add_special_tokens=True)
            ids = enc["input_ids"].to(device)
            am = enc["attention_mask"].to(device)
            t_pad = ids.shape[1]
            spos = []
            tmask = torch.zeros((end - start, t_pad, 1), dtype=torch.float32, device=device)
            for bi, r in enumerate(brecs):
                si, sl = locate_last_subject_token(tok, r["prompt"], r["subj_char_span"])
                pos = max(0, min(t_pad - 1, (t_pad - sl) + si))
                spos.append(pos)
                tmask[bi, pos, 0] = 1.0
            suite.set_token_mask(tmask)
            pos_use = spos if pos_kind == "subj" else [t_pad - 1] * (end - start)
            buf = {}

            def cap(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                buf["x"] = torch.stack([h[b, pos_use[b], :] for b in range(h.shape[0])], dim=0).detach().float().cpu().numpy()

            # Register capture AFTER adapter hook so we capture the post-adapter state!
            handle = layers[layer_idx].register_forward_hook(cap)
            try:
                model(input_ids=ids, attention_mask=am, use_cache=False)
            finally:
                handle.remove()
            outs.append(buf["x"])
    suite.set_token_mask(None)
    return np.concatenate(outs, axis=0).astype(np.float64)


# =============================================================================
# 6. Full Pilot 2 Runner
# =============================================================================


def run_pilot2() -> dict:
    t_all = time.time()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[Pilot 2] Device={device} (using float32 for MPS backward safety, CPU float64 for SVD/MEMIT)")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(BASELINE_RESULTS_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    facts = dataset["facts"]
    used_subjs = {f["subject_norm"].strip().lower() for f in facts}
    for e in CANDIDATE_ENTITIES_32:
        assert e.lower() not in used_subjs, f"Entity {e} overlaps dataset.json!"

    # ------------------------------------------------------------------
    # Step 1: Freeze Alignment Maps STRICTLY on dataset.json train facts
    # ------------------------------------------------------------------
    print("\n=== [Step 1] Freezing Alignment Maps Strictly on dataset.json Train Facts ===")
    bp = baseline["pairs"][f"{DONOR_KEY}->{RECIP_KEY}"]
    sh_tr = [i for i in bp["shared_train_indices"] if facts[i]["split"] == "train"]
    l_A_key, l_B_key = int(bp["best_layer_A"]), int(bp["best_layer_B"])  # (17, 9)
    # Site 2 mid-late attribute payload sites: 75% depth (27, 18) and 83% depth (30, 20)
    site_specs = {
        "site1_early_key_subj": {"l_A": l_A_key, "l_B": l_B_key, "pos": "subj", "state_key": "direct_subj", "lora_layers_A": list(range(l_A_key - 5, l_A_key + 1)), "mask_subj": True},
        "site2_midlate_last_75pct": {"l_A": 27, "l_B": 18, "pos": "last", "state_key": "direct_last", "lora_layers_A": list(range(22, 28)), "mask_subj": False},
        "site2_late_last_83pct": {"l_A": 30, "l_B": 20, "pos": "last", "state_key": "direct_last", "lora_layers_A": list(range(25, 31)), "mask_subj": False},
    }

    st_A = torch.load(CACHE_DIR / f"{DONOR_KEY}_states.pt", map_location="cpu", weights_only=False)
    st_B = torch.load(CACHE_DIR / f"{RECIP_KEY}_states.pt", map_location="cpu", weights_only=False)
    frozen_maps: Dict[str, dict] = {}
    for sname, sp in site_specs.items():
        ZA_tr = st_A[sp["state_key"]][sh_tr, sp["l_A"]].float().numpy()
        ZB_tr = st_B[sp["state_key"]][sh_tr, sp["l_B"]].float().numpy()
        pm_span = systems.fit_procrustes_fast_qr(ZA_tr, ZB_tr)
        pm_dense = systems.fit_procrustes(ZA_tr, ZB_tr)
        pm_rand = systems.fit_random_orthogonal_control(ZA_tr, ZB_tr, seed=20261210 + hash(sname) % 1000)
        frozen_maps[sname] = {"span": pm_span, "dense": pm_dense, "rand": pm_rand}
        print(f"  Frozen {sname}: l=({sp['l_A']},{sp['l_B']}) pos={sp['pos']} scale={pm_span.scale:.3f} (N_tr={len(sh_tr)})")
    del st_A, st_B
    gc.collect()

    # ------------------------------------------------------------------
    # Step 2: Draw Fresh Private Tables f & f' AFTER Freezing Maps
    # ------------------------------------------------------------------
    table_f, table_fp = build_private_tables(CANDIDATE_ENTITIES_32, GUILD_LABELS)
    recs_f_train = make_prompt_records(CANDIDATE_ENTITIES_32, table_f, TRAIN_TEMPLATES)
    recs_f_test = make_prompt_records(CANDIDATE_ENTITIES_32, table_f, TEST_PARAPHRASE_TEMPLATES)
    recs_fp_train = make_prompt_records(CANDIDATE_ENTITIES_32, table_fp, TRAIN_TEMPLATES)
    recs_fp_test = make_prompt_records(CANDIDATE_ENTITIES_32, table_fp, TEST_PARAPHRASE_TEMPLATES)
    anchor_prompts = [facts[i]["direct_prompt"] for i in sh_tr[:64]]

    results: Dict[str, Any] = {
        "meta": {
            "donor": DONOR_KEY,
            "recipient": RECIP_KEY,
            "guild_labels": GUILD_LABELS,
            "entities_32": CANDIDATE_ENTITIES_32,
            "table_f": table_f,
            "table_f_prime": table_fp,
            "train_templates": TRAIN_TEMPLATES,
            "test_paraphrase_templates": TEST_PARAPHRASE_TEMPLATES,
            "n_frozen_map_train": len(sh_tr),
            "site_specs": site_specs,
        },
        "donor_A_controls": {},
        "native_B_ceiling": {},
        "test2A_live_crossing": {},
        "test2B_donor_free_memit": {},
    }

    # ------------------------------------------------------------------
    # Step 3: Load Donor A (qwen2.5-3b), Extract Base States, Train f & f'
    #         Layer-Restricted Adapters, Verify Self-Swap >= 80%, Extract Edited States,
    #         THEN COMPLETELY UNLOAD DONOR A!
    # ------------------------------------------------------------------
    print(f"\n=== [Step 3] Loading Donor A ({DONOR_KEY}) in float32 on {device} ===")
    tok_A = AutoTokenizer.from_pretrained(str(HF_CACHE_DIR / DONOR_KEY), local_files_only=True)
    tok_A.padding_side = "left"
    if tok_A.pad_token is None:
        tok_A.pad_token = tok_A.eos_token
    model_A = AutoModelForCausalLM.from_pretrained(
        str(HF_CACHE_DIR / DONOR_KEY), local_files_only=True, dtype=torch.float32
    ).to(device)
    model_A.eval()

    # Extract Donor A's UNEDITED base states at all 3 sites on train & test prompts
    base_A_states: Dict[str, Dict[str, np.ndarray]] = {}
    for sname, sp in site_specs.items():
        base_A_states[sname] = {
            "train": extract_site_states(model_A, tok_A, recs_f_train, sp["l_A"], sp["pos"], device),
            "test": extract_site_states(model_A, tok_A, recs_f_test, sp["l_A"], sp["pos"], device),
        }

    # Train Donor A adapters for each site and each table (f and f'), verify positive controls,
    # capture edited states, and discard adapter before next run!
    donor_edited_states: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {s: {} for s in site_specs}

    for sname, sp in site_specs.items():
        results["donor_A_controls"][sname] = {}
        for tname, tr_recs, te_recs, seed_t in [
            ("f", recs_f_train, recs_f_test, 101),
            ("f_prime", recs_fp_train, recs_fp_test, 202),
        ]:
            t0 = time.time()
            print(f"\n  [Donor A train] site={sname} table={tname} layers={sp['lora_layers_A']} mask_subj={sp['mask_subj']}")
            suite = train_layer_restricted_adapter(
                model_A, tok_A, tr_recs, anchor_prompts,
                layer_indices=sp["lora_layers_A"],
                mask_to_subj_only=sp["mask_subj"],
                device=device, rank=16, steps=180, lr=4e-3 if sp["mask_subj"] else 3e-3,
                batch_size=16, kl_weight=0.05, seed=seed_t,
            )
            ev_te = eval_with_subj_mask_if_needed(model_A, tok_A, suite, te_recs, GUILD_LABELS, sp["mask_subj"], device)
            # Extract edited states on train (for Test 2B MEMIT) and test (for Test 2A Live Crossing + Self-Swap!)
            z_tr_ed = extract_states_with_adapter(model_A, tok_A, suite, tr_recs, sp["l_A"], sp["pos"], sp["mask_subj"], device)
            z_te_ed = extract_states_with_adapter(model_A, tok_A, suite, te_recs, sp["l_A"], sp["pos"], sp["mask_subj"], device)
            donor_edited_states[sname][tname] = {"train": z_tr_ed, "test": z_te_ed}

            # Mandatory Positive Control: Donor A -> Donor A Self-Swap at l_A (with adapter DETACHED!)
            suite.detach()
            ev_self = eval_guild_records(
                model_A, tok_A, te_recs, GUILD_LABELS, device=device,
                patch_layer=sp["l_A"], patch_pos_kind=sp["pos"], patch_replace=z_te_ed,
            )
            results["donor_A_controls"][sname][tname] = {
                "donor_test_choice8": ev_te["choice8_acc"],
                "donor_test_greedy_em": ev_te["greedy_em_acc"],
                "donor_self_swap_choice8": ev_self["choice8_acc"],
                "donor_self_swap_greedy_em": ev_self["greedy_em_acc"],
                "elapsed_sec": time.time() - t0,
            }
            print(
                f"    -> Donor A ({sname}, {tname}): unseen_para 8way={ev_te['choice8_acc']:.3f} EM={ev_te['greedy_em_acc']:.3f} | "
                f"SELF-SWAP at l_A={sp['l_A']} 8way={ev_self['choice8_acc']:.3f} EM={ev_self['greedy_em_acc']:.3f} [{time.time()-t0:.1f}s]"
            )
            del suite
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    # =========================================================================
    # THROW DONOR A AWAY COMPLETELY BEFORE LOADING RECIPIENT B!!
    # =========================================================================
    print(f"\n=== [UNLOADING DONOR A ({DONOR_KEY}) COMPLETELY FROM MEMORY!] ===")
    del model_A, tok_A
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ------------------------------------------------------------------
    # Step 4: Load Frozen Recipient B (`smollm2-1.7b`) + Run Test 2A & Test 2B!
    # ------------------------------------------------------------------
    print(f"\n=== [Step 4] Loading Solo Recipient B ({RECIP_KEY}) in float32 on {device} ===")
    tok_B = AutoTokenizer.from_pretrained(str(HF_CACHE_DIR / RECIP_KEY), local_files_only=True)
    tok_B.padding_side = "left"
    if tok_B.pad_token is None:
        tok_B.pad_token = tok_B.eos_token
    model_B = AutoModelForCausalLM.from_pretrained(
        str(HF_CACHE_DIR / RECIP_KEY), local_files_only=True, dtype=torch.float32
    ).to(device)
    model_B.eval()

    # Evaluate Unmodified Recipient B on f, f', AND 196-fact dataset.json locality baseline!
    ev_B_unmod_f = eval_guild_records(model_B, tok_B, recs_f_test, GUILD_LABELS, device=device)
    ev_B_unmod_fp = eval_guild_records(model_B, tok_B, recs_fp_test, GUILD_LABELS, device=device)

    with open(CACHE_DIR / f"{RECIP_KEY}_eval.json", "r", encoding="utf-8") as f:
        eval_B_1hop = json.load(f)["facts"]
    test_1hop_idx = [f["idx"] for f in facts if f["split"] == "test"]
    train_1hop_idx = [f["idx"] for f in facts if f["split"] == "train"]
    loc_prompts = [facts[i]["direct_prompt"] for i in test_1hop_idx]
    loc_spos = [eval_B_1hop[i]["subj_tok_idx_direct"] for i in test_1hop_idx]
    loc_slens = [eval_B_1hop[i]["seq_len_direct"] for i in test_1hop_idx]
    loc_aliases = [facts[i]["aliases"] for i in test_1hop_idx]
    loc_rels = [facts[i]["relation"] for i in test_1hop_idx]

    base_loc_acc, base_loc_out = eval_multi_regime_swaps(
        model_B, tok_B, loc_prompts, loc_spos, loc_slens, loc_aliases, loc_rels,
        device=device, dtype=torch.float32,
    )
    results["unmodified_B"] = {
        "f_test_choice8": ev_B_unmod_f["choice8_acc"],
        "f_test_greedy_em": ev_B_unmod_f["greedy_em_acc"],
        "fp_test_choice8": ev_B_unmod_fp["choice8_acc"],
        "fp_test_greedy_em": ev_B_unmod_fp["greedy_em_acc"],
        "locality_196_test_acc": base_loc_acc,
    }
    print(
        f"  [Unmodified B] f: 8way={ev_B_unmod_f['choice8_acc']:.3f} EM={ev_B_unmod_f['greedy_em_acc']:.3f} | "
        f"f': 8way={ev_B_unmod_fp['choice8_acc']:.3f} EM={ev_B_unmod_fp['greedy_em_acc']:.3f} | "
        f"locality(196)={base_loc_acc:.3f}"
    )

    # ---- TEST 2A: Live "Pointer or Payload?" Crossing at All 3 Sites ----
    print("\n=== [Test 2A] Live 'Pointer or Payload?' Cross-Family Crossing (Unseen Paraphrases) ===")
    rng_shuf = np.random.default_rng(999)
    # Deterministic cyclic entity shift for shuffled_donor (ensures wrong entity & wrong guild!)
    shuf_perm = np.array([(i + 1) % 32 + (i // 32) * 32 for i in range(len(recs_f_test))])

    for sname, sp in site_specs.items():
        pm_span = frozen_maps[sname]["span"]
        pm_dense = frozen_maps[sname]["dense"]
        pm_rand = frozen_maps[sname]["rand"]
        l_B = sp["l_B"]
        pos = sp["pos"]

        zA_base_te = base_A_states[sname]["test"]
        zA_f_te = donor_edited_states[sname]["f"]["test"]
        zA_fp_te = donor_edited_states[sname]["f_prime"]["test"]
        dA_f_te = zA_f_te - zA_base_te
        dA_fp_te = zA_fp_te - zA_base_te

        # 1. Full-state swap (Span-Procrustes & Dense Procrustes)
        ev_full_f_span = eval_guild_records(
            model_B, tok_B, recs_f_test, GUILD_LABELS, device=device,
            patch_layer=l_B, patch_pos_kind=pos, patch_replace=pm_span.transform(zA_f_te),
        )
        ev_full_fp_span = eval_guild_records(
            model_B, tok_B, recs_fp_test, GUILD_LABELS, device=device,
            patch_layer=l_B, patch_pos_kind=pos, patch_replace=pm_span.transform(zA_fp_te),
        )
        ev_full_f_dense = eval_guild_records(
            model_B, tok_B, recs_f_test, GUILD_LABELS, device=device,
            patch_layer=l_B, patch_pos_kind=pos, patch_replace=pm_dense.transform(zA_f_te),
        )
        # Cross-table check: f' donor scored against OLD table f (must DROP below chance!)
        ev_fp_scored_on_f = eval_guild_records(
            model_B, tok_B, recs_f_test, GUILD_LABELS, device=device,
            patch_layer=l_B, patch_pos_kind=pos, patch_replace=pm_span.transform(zA_fp_te),
        )

        # 2. Delta-Teleportation: z_B^base + alpha * s * R * (z_A^edit - z_A^base)
        delta_results = {}
        for alpha in (1.0, 1.5, 2.0):
            ev_df = eval_guild_records(
                model_B, tok_B, recs_f_test, GUILD_LABELS, device=device,
                patch_layer=l_B, patch_pos_kind=pos, patch_add=pm_span.transform_delta(dA_f_te, alpha=alpha),
            )
            ev_dfp = eval_guild_records(
                model_B, tok_B, recs_fp_test, GUILD_LABELS, device=device,
                patch_layer=l_B, patch_pos_kind=pos, patch_add=pm_span.transform_delta(dA_fp_te, alpha=alpha),
            )
            delta_results[f"alpha_{alpha}"] = {
                "f_choice8": ev_df["choice8_acc"],
                "f_greedy_em": ev_df["greedy_em_acc"],
                "f_outcomes_8": ev_df["choice8_outcomes"],
                "f_outcomes_em": ev_df["greedy_outcomes"],
                "fp_choice8": ev_dfp["choice8_acc"],
                "fp_greedy_em": ev_dfp["greedy_em_acc"],
                "fp_outcomes_8": ev_dfp["choice8_outcomes"],
                "fp_outcomes_em": ev_dfp["greedy_outcomes"],
            }

        # 3. Controls: shuffled_donor (wrong entity) & random_rotation (R_rand)
        ev_shuf_full = eval_guild_records(
            model_B, tok_B, recs_f_test, GUILD_LABELS, device=device,
            patch_layer=l_B, patch_pos_kind=pos, patch_replace=pm_span.transform(zA_f_te[shuf_perm]),
        )
        ev_shuf_delta = eval_guild_records(
            model_B, tok_B, recs_f_test, GUILD_LABELS, device=device,
            patch_layer=l_B, patch_pos_kind=pos, patch_add=pm_span.transform_delta(dA_f_te[shuf_perm], alpha=1.0),
        )
        ev_rand_full = eval_guild_records(
            model_B, tok_B, recs_f_test, GUILD_LABELS, device=device,
            patch_layer=l_B, patch_pos_kind=pos, patch_replace=pm_rand.transform(zA_f_te),
        )
        ev_rand_delta = eval_guild_records(
            model_B, tok_B, recs_f_test, GUILD_LABELS, device=device,
            patch_layer=l_B, patch_pos_kind=pos, patch_add=pm_rand.transform_delta(dA_f_te, alpha=1.0),
        )

        results["test2A_live_crossing"][sname] = {
            "full_swap_span_f": {"choice8": ev_full_f_span["choice8_acc"], "greedy_em": ev_full_f_span["greedy_em_acc"], "outcomes_8": ev_full_f_span["choice8_outcomes"], "outcomes_em": ev_full_f_span["greedy_outcomes"]},
            "full_swap_span_fp": {"choice8": ev_full_fp_span["choice8_acc"], "greedy_em": ev_full_fp_span["greedy_em_acc"], "outcomes_8": ev_full_fp_span["choice8_outcomes"], "outcomes_em": ev_full_fp_span["greedy_outcomes"]},
            "full_swap_dense_f": {"choice8": ev_full_f_dense["choice8_acc"], "greedy_em": ev_full_f_dense["greedy_em_acc"]},
            "fp_scored_against_old_f": {"choice8": ev_fp_scored_on_f["choice8_acc"], "greedy_em": ev_fp_scored_on_f["greedy_em_acc"]},
            "delta_teleport": delta_results,
            "shuffled_donor_full": {"choice8": ev_shuf_full["choice8_acc"], "greedy_em": ev_shuf_full["greedy_em_acc"]},
            "shuffled_donor_delta": {"choice8": ev_shuf_delta["choice8_acc"], "greedy_em": ev_shuf_delta["greedy_em_acc"]},
            "random_rotation_full": {"choice8": ev_rand_full["choice8_acc"], "greedy_em": ev_rand_full["greedy_em_acc"]},
            "random_rotation_delta": {"choice8": ev_rand_delta["choice8_acc"], "greedy_em": ev_rand_delta["greedy_em_acc"]},
        }
        d1 = delta_results["alpha_1.0"]
        d15 = delta_results["alpha_1.5"]
        print(
            f"  [Test 2A {sname:26s}] full_f: 8way={ev_full_f_span['choice8_acc']:.3f} EM={ev_full_f_span['greedy_em_acc']:.3f} | "
            f"full_f': 8way={ev_full_fp_span['choice8_acc']:.3f} EM={ev_full_fp_span['greedy_em_acc']:.3f} (on_old_f={ev_fp_scored_on_f['choice8_acc']:.3f}) | "
            f"delta(1.0) f={d1['f_choice8']:.3f}/{d1['f_greedy_em']:.3f} f'={d1['fp_choice8']:.3f}/{d1['fp_greedy_em']:.3f} | "
            f"delta(1.5) f={d15['f_choice8']:.3f}/{d15['f_greedy_em']:.3f} | shuf={ev_shuf_delta['choice8_acc']:.3f} rand={ev_rand_delta['choice8_acc']:.3f}"
        )

    # ---- TEST 2B: Donor-Free Platonic MEMIT Compile (Zero SGD on B, Donor A ALREADY GONE!) ----
    print("\n=== [Test 2B] Donor-Free Platonic MEMIT Compile into Solo Recipient B's mlp.down_proj ===")
    layers_B = get_decoder_layers(model_B)
    bg_prompts = [facts[i]["direct_prompt"] for i in train_1hop_idx]
    bg_spans = [facts[i]["direct_subj_char_span"] for i in train_1hop_idx]
    tr_prompts = [r["prompt"] for r in recs_f_train]
    tr_spans = [r["subj_char_span"] for r in recs_f_train]

    for sname, sp in site_specs.items():
        l_B = sp["l_B"]
        pos = sp["pos"]
        pm_span = frozen_maps[sname]["span"]
        pm_rand = frozen_maps[sname]["rand"]

        # Extract B's native post-SwiGLU MLP keys on the 64 train prompts + dataset.json background facts (both subj & last tokens for clean locality!)
        K_edit = extract_mlp_down_proj_inputs(model_B, tok_B, tr_prompts, tr_spans, l_B, pos, device)
        K_bg_pos = extract_mlp_down_proj_inputs(model_B, tok_B, bg_prompts, bg_spans, l_B, pos, device)
        other_pos = "last" if pos == "subj" else "subj"
        K_bg_other = extract_mlp_down_proj_inputs(model_B, tok_B, bg_prompts[:256], bg_spans[:256], l_B, other_pos, device)
        K_bg = np.concatenate([K_bg_pos, K_bg_other], axis=0)
        zB_tr_base = extract_site_states(model_B, tok_B, recs_f_train, l_B, pos, device)

        zA_base_tr = base_A_states[sname]["train"]
        dA_f_tr = donor_edited_states[sname]["f"]["train"] - zA_base_tr
        dA_fp_tr = donor_edited_states[sname]["f_prime"]["train"] - zA_base_tr
        zA_f_tr = donor_edited_states[sname]["f"]["train"]

        # Save exact original weight for clean restoration between MEMIT compiles!
        orig_W = layers_B[l_B].mlp.down_proj.weight.detach().clone()

        def _compile_and_eval(delta_V: np.ndarray, lam_cov: float, eval_recs: Sequence[dict], check_loc: bool = True) -> dict:
            dW = solve_woodbury_memit(K_edit, delta_V, K_bg, lam_cov=lam_cov, lam_iso=1e-2)
            with torch.no_grad():
                layers_B[l_B].mlp.down_proj.weight.copy_(orig_W + torch.from_numpy(dW).to(device=orig_W.device, dtype=orig_W.dtype))
            ev_te = eval_guild_records(model_B, tok_B, eval_recs, GUILD_LABELS, device=device)
            ev_tr = eval_guild_records(model_B, tok_B, recs_f_train if eval_recs is recs_f_test else recs_fp_train, GUILD_LABELS, device=device)
            loc_acc = None
            if check_loc:
                loc_acc, _ = eval_multi_regime_swaps(
                    model_B, tok_B, loc_prompts, loc_spos, loc_slens, loc_aliases, loc_rels,
                    device=device, dtype=torch.float32,
                )
            with torch.no_grad():
                layers_B[l_B].mlp.down_proj.weight.copy_(orig_W)
            return {
                "lam_cov": lam_cov,
                "train_choice8": ev_tr["choice8_acc"],
                "train_greedy_em": ev_tr["greedy_em_acc"],
                "test_unseen_para_choice8": ev_te["choice8_acc"],
                "test_unseen_para_greedy_em": ev_te["greedy_em_acc"],
                "test_outcomes_8": ev_te["choice8_outcomes"],
                "test_outcomes_em": ev_te["greedy_outcomes"],
                "locality_196_acc": loc_acc,
                "locality_drop_pp": (base_loc_acc - loc_acc) * 100.0 if loc_acc is not None else None,
            }

        # Select best (lam_cov, alpha) strictly on TRAIN templates (`train_choice8` + locality preservation, NEVER test paraphrases!)
        # Actually report clean grid over lam_cov in (10.0, 50.0, 200.0) at alpha=1.0 AND alpha=1.5!
        memit_sweep = {}
        best_cfg = None
        best_tr_score = -1.0
        for target_mode in ("delta_1.0", "delta_1.5", "full_minus_base"):
            for lam_cov in (10.0, 50.0, 200.0):
                if target_mode == "delta_1.0":
                    dV_f = pm_span.transform_delta(dA_f_tr, alpha=1.0)
                elif target_mode == "delta_1.5":
                    dV_f = pm_span.transform_delta(dA_f_tr, alpha=1.5)
                else:
                    dV_f = pm_span.transform(zA_f_tr) - zB_tr_base
                key_m = f"{target_mode}_lam{int(lam_cov)}"
                res_m = _compile_and_eval(dV_f, lam_cov, recs_f_test, check_loc=True)
                memit_sweep[key_m] = res_m
                # Selection strictly by train EM & locality <= 3pp (ZERO peeking at test_unseen_para!)
                if res_m["locality_drop_pp"] is not None and res_m["locality_drop_pp"] <= 3.5:
                    sc = res_m["train_choice8"] + 0.5 * res_m["train_greedy_em"]
                    if sc > best_tr_score:
                        best_tr_score = sc
                        best_cfg = (target_mode, lam_cov)

        if best_cfg is None:
            best_cfg = ("delta_1.5", 50.0)
        sel_mode, sel_lam = best_cfg

        # Evaluate selected config on REASSIGNED table f' (AND cross-check f' compiled B against old f!)
        if sel_mode == "delta_1.0":
            dV_fp = pm_span.transform_delta(dA_fp_tr, alpha=1.0)
            dV_rand = pm_rand.transform_delta(dA_f_tr, alpha=1.0)
            dV_shuf = pm_span.transform_delta(dA_f_tr[shuf_perm], alpha=1.0)
        elif sel_mode == "delta_1.5":
            dV_fp = pm_span.transform_delta(dA_fp_tr, alpha=1.5)
            dV_rand = pm_rand.transform_delta(dA_f_tr, alpha=1.5)
            dV_shuf = pm_span.transform_delta(dA_f_tr[shuf_perm], alpha=1.5)
        else:
            dV_fp = pm_span.transform(donor_edited_states[sname]["f_prime"]["train"]) - zB_tr_base
            dV_rand = pm_rand.transform(zA_f_tr) - zB_tr_base
            dV_shuf = pm_span.transform(zA_f_tr[shuf_perm]) - zB_tr_base

        res_fp_sel = _compile_and_eval(dV_fp, sel_lam, recs_fp_test, check_loc=True)
        res_fp_on_old_f = _compile_and_eval(dV_fp, sel_lam, recs_f_test, check_loc=False)
        res_rand_sel = _compile_and_eval(dV_rand, sel_lam, recs_f_test, check_loc=True)
        res_shuf_sel = _compile_and_eval(dV_shuf, sel_lam, recs_f_test, check_loc=False)

        sel_f = memit_sweep[f"{sel_mode}_lam{int(sel_lam)}"]
        results["test2B_donor_free_memit"][sname] = {
            "layer_B": l_B,
            "selected_config_by_train_and_loc": {"mode": sel_mode, "lam_cov": sel_lam},
            "compiled_f": sel_f,
            "compiled_f_prime": res_fp_sel,
            "compiled_fp_scored_on_old_f": res_fp_on_old_f,
            "random_rotation_control": res_rand_sel,
            "shuffled_donor_control": res_shuf_sel,
            "full_sweep": memit_sweep,
        }
        print(
            f"  [Test 2B SOLO MEMIT {sname:26s}] sel=({sel_mode}, lam={int(sel_lam)}) | "
            f"f unseen_para: 8way={sel_f['test_unseen_para_choice8']:.3f} EM={sel_f['test_unseen_para_greedy_em']:.3f} loc_drop={sel_f['locality_drop_pp']:+.2f}pp | "
            f"f' unseen_para: 8way={res_fp_sel['test_unseen_para_choice8']:.3f} EM={res_fp_sel['test_unseen_para_greedy_em']:.3f} (on_old_f={res_fp_on_old_f['test_unseen_para_choice8']:.3f}) | "
            f"rand={res_rand_sel['test_unseen_para_choice8']:.3f} shuf={res_shuf_sel['test_unseen_para_choice8']:.3f}"
        )

    # ------------------------------------------------------------------
    # Step 5: Native B Ceiling Adapter (Verify B Can Learn f & f' >= 80%!)
    # ------------------------------------------------------------------
    print("\n=== [Step 5] Native Recipient B Ceiling Adapter (layers 14..18) ===")
    for tname, tr_recs, te_recs, seed_t in [
        ("f", recs_f_train, recs_f_test, 303),
        ("f_prime", recs_fp_train, recs_fp_test, 404),
    ]:
        suite_B = train_layer_restricted_adapter(
            model_B, tok_B, tr_recs, anchor_prompts,
            layer_indices=list(range(14, 19)), mask_to_subj_only=False,
            device=device, rank=16, steps=180, lr=3e-3, batch_size=16, kl_weight=0.05, seed=seed_t,
        )
        ev_nb = eval_guild_records(model_B, tok_B, te_recs, GUILD_LABELS, device=device)
        loc_nb, _ = eval_multi_regime_swaps(
            model_B, tok_B, loc_prompts, loc_spos, loc_slens, loc_aliases, loc_rels,
            device=device, dtype=torch.float32,
        )
        suite_B.detach()
        results["native_B_ceiling"][tname] = {
            "test_unseen_para_choice8": ev_nb["choice8_acc"],
            "test_unseen_para_greedy_em": ev_nb["greedy_em_acc"],
            "locality_196_acc": loc_nb,
            "locality_drop_pp": (base_loc_acc - loc_nb) * 100.0,
        }
        print(
            f"  [Native B ceiling {tname}] unseen_para 8way={ev_nb['choice8_acc']:.3f} EM={ev_nb['greedy_em_acc']:.3f} "
            f"loc_drop={(base_loc_acc - loc_nb)*100.0:+.2f}pp"
        )
        del suite_B

    results["elapsed_total_sec"] = time.time() - t_all
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(_json_clean(results), f, indent=2)
    print(f"\n[Pilot 2] ALL DONE in {time.time() - t_all:.1f}s -> {OUTPUT_PATH}")
    return results


if __name__ == "__main__":
    run_pilot2()
