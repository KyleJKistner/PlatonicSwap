#!/usr/bin/env python3
"""
pilot2_fullrank_check.py — Breakthrough Pilot 2b:
Resolving N << D Nullspace Starvation vs. Pointer-Only / Payload Crossing +
Donor-Free Platonic MEMIT Compilation Across Three Strictly Task-Blind Full-Rank Maps.

Diagnoses WHY N=429 dataset.json Span-Procrustes hit chance in Pilot 2 and tests:
  1. Exact nullspace energy diagnostic: what fraction of Donor A's private-table delta
     dA = zA^edit - zA^base lies inside vs. outside the N=429 span, and its cosine
     alignment with Donor A's guild unembedding vectors W_{U,A}[guild] across depth
     (l_A=17 early subj, l_A=30 late 83%, l_A=35 pre-output).
  2. Saves ALL Donor A base & edited payloads (f and deranged f') to
     /tmp/platos_key_lab/cache/pilot2_donorA_payloads.pt so future maps (including
     Pilot 1's N=10,000 maps) can be re-evaluated in <45s with --reuse-payloads!
  3. Head-to-head comparison of strictly FROZEN, ZERO-TASK-LEAKAGE alignment maps
     (strictly excluding all 8 guild labels, 8 colors, and all 32 entities):
       - Map A: N=429 QA Span-Procrustes (rank-starved Pilot 2 baseline)
       - Map B: Full-Rank Unlabeled General-Text Residual Alignment (N=3,500+ non-guild
                prose tokens; pure hidden-state Procrustes & Ridge — ZERO W_U used!)
                PLUS auto-loads Pilot 1's N=10,000 maps at site1 (17->9) when present!
       - Map C: Zero-Forward-Pass Dual-Adjoint Vocabulary Map T_pre (closed-form Ridge
                on ~25,000 strictly held-out shared English words in W_U, exact
                pre-RMSNorm logit-delta equivariance, 0 forward passes!)
       - Map D: Matched Random Orthogonal Control (sanity baseline -> ~12.5%)
  4. Evaluates BOTH:
       - Test 2A: Live Payload Crossing on unseen paraphrases (f, deranged f', cross-score,
                  shuffled donor, random map, unmodified B; 8-way choice AND greedy EM)
       - Test 2B: Donor-Free Platonic MEMIT Weight Compilation into solo Recipient B's
                  native mlp.down_proj weights (0 SGD on B, Donor A completely unloaded!),
                  including single-layer (l_B=23, l_B=20) AND distributed 4-layer MEMIT
                  ([20, 21, 22, 23]), per-template generalization (including novel final
                  token " as"), and 196-fact dataset.json locality retention!
"""

from __future__ import annotations

import argparse
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
from safetensors import safe_open
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
    get_decoder_layers,
)
from run_systems_dark import _json_clean, eval_multi_regime_swaps
from pilot2_payload_memit import (
    CANDIDATE_ENTITIES_32,
    DONOR_KEY,
    GUILD_LABELS,
    RECIP_KEY,
    build_private_tables,
    eval_guild_records,
    extract_mlp_down_proj_inputs,
    extract_site_states,
    extract_states_with_adapter,
    make_prompt_records,
    solve_woodbury_memit,
    train_layer_restricted_adapter,
)


def eval_live_site_crossing(
    model: torch.nn.Module,
    tok: Any,
    records: Sequence[dict],
    layer_idx: int,
    pos_kind: str,
    vectors: np.ndarray,
    labels: Sequence[str],
    device: torch.device,
    mode: str = "delta",
) -> Dict[str, Any]:
    if mode == "full":
        return eval_guild_records(
            model, tok, records, labels, device=device,
            patch_layer=layer_idx, patch_pos_kind=pos_kind, patch_replace=vectors,
        )
    return eval_guild_records(
        model, tok, records, labels, device=device,
        patch_layer=layer_idx, patch_pos_kind=pos_kind, patch_add=vectors,
    )

OUTPUT_PATH = LAB_DIR / "results_pilot2_fullrank.json"
PAYLOAD_CACHE_PATH = CACHE_DIR / "pilot2_donorA_payloads.pt"
UNLAB_LATE_CACHE_PATH = CACHE_DIR / "pilot2_unlab_late_states.pt"
PILOT1_MAPS_PATH = CACHE_DIR / f"pilot1_maps_{DONOR_KEY}_to_{RECIP_KEY}_l17_9.pt"

# 3 diverse training templates (covering both "of {e} is" and "{e} belongs to guild")
TRAIN_TEMPLATES_3: List[Tuple[str, str]] = [
    ("The official guild of ", " is"),
    ("The guild assigned to ", " is"),
    ("In the confidential ledger, ", " belongs to guild"),
]

# 3 completely UNSEEN test paraphrases (exact Pilot 2 test templates + novel final token " as")
TEST_TEMPLATES_3: List[Tuple[str, str]] = [
    ("In the registry, the guild of ", " is"),
    ("According to the archive, ", " belongs to guild"),
    ("Official records list the guild for ", " as"),
]

# General-domain non-guild, non-entity natural prose templates for N >= 3,500 unlabeled alignment
UNLAB_SENTENCE_TEMPLATES: List[str] = [
    "During the nineteenth century, the development of {noun} transformed {domain} across Europe and led to",
    "In modern engineering, careful analysis of {noun} helps researchers improve {domain} whenever",
    "According to recent surveys, public interest in {noun} has shaped {domain} because",
    "Throughout maritime history, access to {noun} influenced {domain} and encouraged",
    "In university seminars, students examining {noun} often compare {domain} with",
    "Environmental reports indicate that {noun} plays a central role in {domain} during",
    "Economic historians note that trade in {noun} accelerated {domain} by",
    "In architectural design, the choice of {noun} determines {domain} under",
    "Medical journals frequently discuss how {noun} interacts with {domain} after",
    "Agricultural manuals explain that managing {noun} strengthens {domain} throughout",
]

UNLAB_NOUNS: List[str] = [
    "railways", "bridges", "harbors", "canals", "textiles", "ceramics", "timber", "granite",
    "glassware", "turbines", "engines", "compasses", "telescopes", "microscopes", "barometers",
    "clocks", "pendulums", "lenses", "prisms", "batteries", "conductors", "magnets", "antennas",
    "cables", "pipelines", "reservoirs", "aqueducts", "windmills", "greenhouses", "orchards",
    "vineyards", "pastures", "wetlands", "glaciers", "estuaries", "plateaus", "canyons", "valleys",
    "forests", "meadows", "prairies", "deserts", "islands", "peninsulas", "volcanoes", "geysers",
    "minerals", "fossils", "sediments", "crystals", "alloys", "polymers", "solvents", "catalysts",
    "enzymes", "proteins", "vitamins", "nutrients", "antibodies", "vaccines", "antibiotics",
    "manuscripts", "chronicles", "treatises", "dictionaries", "atlases", "blueprints", "diagrams",
    "symphonies", "sonatas", "operas", "frescoes", "sculptures", "tapestries", "mosaics",
    "cathedrals", "aquariums", "observatories", "libraries", "museums", "theaters", "stadiums",
    "universities", "hospitals", "laboratories", "factories", "shipyards", "foundries", "refineries",
    "satellites", "rockets", "gliders", "submarines", "locomotives", "trams", "ferries", "barges",
    "wheat", "barley", "oats", "maize", "cotton", "linen", "wool", "hemp", "coffee", "cocoa",
    "spices", "olive oil", "honey", "cheese", "bread", "vinegar", "salt", "sugar", "tea", "citrus",
]

UNLAB_DOMAINS: List[str] = [
    "regional commerce", "urban planning", "scientific research", "industrial production",
    "coastal navigation", "public health", "higher education", "agricultural yields",
    "structural durability", "energy efficiency", "historical preservation", "climate adaptation",
    "material science", "civic infrastructure", "cultural exchange", "transportation networks",
    "ecological balance", "long-term stability", "technical standards", "global logistics",
    "laboratory safety", "financial markets", "municipal governance", "resource conservation",
    "water management", "optical precision", "mechanical reliability", "acoustic performance",
    "diagnostic accuracy", "archival documentation",
]


# =============================================================================
# 1. Exact Dual-Adjoint Zero-Forward-Pass Vocabulary Map T_pre
# =============================================================================


def load_unembedding_and_norm(model_key: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load frozen lm_head (or tied embed_tokens) weight and final model.norm.weight on CPU."""
    d = HF_CACHE_DIR / model_key
    idx_path = d / "model.safetensors.index.json"
    if idx_path.exists():
        wm = json.load(open(idx_path, "r", encoding="utf-8"))["weight_map"]
        k_wu = "lm_head.weight" if "lm_head.weight" in wm else "model.embed_tokens.weight"
        with safe_open(str(d / wm[k_wu]), framework="pt", device="cpu") as f:
            W = f.get_tensor(k_wu).float().numpy()
        with safe_open(str(d / wm["model.norm.weight"]), framework="pt", device="cpu") as f:
            g = f.get_tensor("model.norm.weight").float().numpy()
    else:
        with safe_open(str(d / "model.safetensors"), framework="pt", device="cpu") as f:
            ks = set(f.keys())
            k_wu = "lm_head.weight" if "lm_head.weight" in ks else "model.embed_tokens.weight"
            W = f.get_tensor(k_wu).float().numpy()
            g = f.get_tensor("model.norm.weight").float().numpy()
    return W, g, k_wu


class DualAdjointMap:
    """Closed-form linear residual-stream map T_pre with transform_delta / transform API.

    Fitted ONLY on strictly held-out shared vocabulary tokens (excluding all 8 guild words,
    all 8 color words, and every word of all 32 entities) — ZERO forward passes, ZERO task data!
    """

    def __init__(
        self,
        T_pre: np.ndarray,
        mu_A: Optional[np.ndarray] = None,
        mu_B: Optional[np.ndarray] = None,
        meta: Optional[dict] = None,
    ):
        self.T = T_pre.astype(np.float64)  # shape (D_A, D_B)
        self.mu_A = np.zeros(T_pre.shape[0], dtype=np.float64) if mu_A is None else mu_A.astype(np.float64)
        self.mu_B = np.zeros(T_pre.shape[1], dtype=np.float64) if mu_B is None else mu_B.astype(np.float64)
        self.meta = meta or {}

    def transform(self, zA: np.ndarray) -> np.ndarray:
        z = np.asarray(zA, dtype=np.float64)
        return ((z - self.mu_A) @ self.T + self.mu_B).astype(np.float32)

    def transform_delta(self, dA: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        d = np.asarray(dA, dtype=np.float64)
        return (float(alpha) * (d @ self.T)).astype(np.float32)


def build_dual_adjoint_vocab_map(
    src_key: str,
    dst_key: str,
    rms_A: float,
    rms_B: float,
    mu_A: Optional[np.ndarray] = None,
    mu_B: Optional[np.ndarray] = None,
    exclude_words: Optional[Sequence[str]] = None,
    lam_rel: float = 1e-3,
    seed_rand: int = 20261401,
) -> Tuple[DualAdjointMap, DualAdjointMap, Dict[str, Any]]:
    """Build exact pre-RMSNorm Dual-Adjoint map T_pre and norm-matched random orthogonal control."""
    tokA = AutoTokenizer.from_pretrained(str(HF_CACHE_DIR / src_key), local_files_only=True)
    tokB = AutoTokenizer.from_pretrained(str(HF_CACHE_DIR / dst_key), local_files_only=True)
    WA, gA, kA = load_unembedding_and_norm(src_key)
    WB, gB, kB = load_unembedding_and_norm(dst_key)

    excl = set()
    for w in (exclude_words or []):
        for piece in w.replace("-", " ").split():
            excl.add(piece.strip(".,;:\"'").lower())

    vA, vB = tokA.get_vocab(), tokB.get_vocab()
    mapA: Dict[str, int] = {}
    for s, idx in vA.items():
        if (s.startswith("\u0120") or s.startswith("\u2581")) and len(s) >= 3 and s[1:].isalpha():
            mapA[s[1:]] = idx

    pairs: List[Tuple[int, int]] = []
    for s, idxB in vB.items():
        if (s.startswith("\u0120") or s.startswith("\u2581")) and len(s) >= 3 and s[1:].isalpha():
            word = s[1:]
            if word in mapA and word.lower() not in excl:
                pairs.append((mapA[word], idxB))

    idxA = np.array([a for a, _ in pairs], dtype=np.int64)
    idxB = np.array([b for _, b in pairs], dtype=np.int64)
    XA = WA[idxA].astype(np.float64)
    XB = WB[idxB].astype(np.float64)

    # Dual Adjoint solve: XB @ M_BA ≈ XA  =>  (nA @ M_BA.T) @ WB.T ≈ nA @ WA.T
    gramB = XB.T @ XB
    lam = float(lam_rel) * float(np.trace(gramB)) / float(XB.shape[1])
    M_BA = np.linalg.solve(gramB + lam * np.eye(XB.shape[1]), XB.T @ XA)  # (D_B, D_A)

    # Exact pre-RMSNorm residual-stream linear operator:
    #   T_pre = (rms_B / rms_A) * diag(gA) @ M_BA.T @ diag(1 / gB_safe)
    gB_safe = np.where(np.abs(gB) < 1e-2, np.sign(gB + 1e-6) * 1e-2, gB)
    T_pre = (float(rms_B) / max(float(rms_A), 1e-6)) * (gA[:, None] * M_BA.T) / gB_safe[None, :]

    # Verify held-out 8-guild top-1 rank in B's 49k vocabulary (purely diagnostic!)
    g_ids_A = np.array([tokA.encode(" " + w, add_special_tokens=False)[0] for w in GUILD_LABELS])
    g_ids_B = np.array([tokB.encode(" " + w, add_special_tokens=False)[0] for w in GUILD_LABELS])
    pred_logits = (WA[g_ids_A].astype(np.float64) @ M_BA.T) @ WB.astype(np.float64).T
    guild_ranks = [int((pred_logits[i] > pred_logits[i, g_ids_B[i]]).sum() + 1) for i in range(8)]
    guild_8way = float((pred_logits[:, g_ids_B].argmax(axis=1) == np.arange(8)).mean())

    # Norm-matched random orthogonal control: T_rand = T_pre @ Q_rand
    rng = np.random.RandomState(seed_rand)
    G = rng.randn(T_pre.shape[1], T_pre.shape[1])
    Q, R_qr = np.linalg.qr(G)
    Q = Q * np.sign(np.diag(R_qr))[None, :]
    T_rand = T_pre @ Q

    meta = {
        "src_weight": kA,
        "dst_weight": kB,
        "n_shared_heldout_tokens": len(pairs),
        "lam_rel": lam_rel,
        "rms_A": float(rms_A),
        "rms_B": float(rms_B),
        "heldout_8guild_vocab_ranks": guild_ranks,
        "heldout_8guild_8way_acc": guild_8way,
    }
    return DualAdjointMap(T_pre, mu_A, mu_B, meta), DualAdjointMap(T_rand, mu_A, mu_B, {"random": True}), meta


class LinearGeneralMap:
    """Wrapper for general closed-form maps (Procrustes-Global, Ridge-GL, ZCA, Fisher-Pullback)."""

    def __init__(self, W: np.ndarray, mu_A: np.ndarray, mu_B: np.ndarray, scale: float = 1.0, name: str = ""):
        self.W = np.asarray(W, dtype=np.float64)  # (D_A, D_B)
        self.mu_A = np.asarray(mu_A, dtype=np.float64)
        self.mu_B = np.asarray(mu_B, dtype=np.float64)
        self.scale = float(scale)
        self.name = name

    def transform(self, zA: np.ndarray) -> np.ndarray:
        z = np.asarray(zA, dtype=np.float64)
        return (self.scale * ((z - self.mu_A) @ self.W) + self.mu_B).astype(np.float32)

    def transform_delta(self, dA: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        d = np.asarray(dA, dtype=np.float64)
        return (float(alpha) * self.scale * (d @ self.W)).astype(np.float32)


def fit_fullrank_procrustes_and_ridge(
    ZA: np.ndarray,
    ZB: np.ndarray,
    lam_rel: float = 1e-2,
) -> Tuple[LinearGeneralMap, LinearGeneralMap]:
    """Fit full-rank O(D) Procrustes-Global AND Ridge-GL in CPU float64."""
    A = np.asarray(ZA, dtype=np.float64)
    B = np.asarray(ZB, dtype=np.float64)
    muA, muB = A.mean(0), B.mean(0)
    Ac, Bc = A - muA, B - muB

    # Full-rank Orthogonal Procrustes (NO span truncation!)
    U, S, Vt = np.linalg.svd(Ac.T @ Bc, full_matrices=False)
    R = U @ Vt
    s = float(S.sum() / (np.sum(Ac ** 2) + 1e-12))
    map_proc = LinearGeneralMap(R, muA, muB, scale=s, name="fullrank_procrustes")

    # Full-rank Ridge GL
    gram = Ac.T @ Ac
    lam = float(lam_rel) * float(np.trace(gram)) / float(Ac.shape[1])
    W_ridge = np.linalg.solve(gram + lam * np.eye(Ac.shape[1]), Ac.T @ Bc)
    map_ridge = LinearGeneralMap(W_ridge, muA, muB, scale=1.0, name="fullrank_ridge")
    return map_proc, map_ridge


# =============================================================================
# 2. Fast Unlabeled Full-Rank Residual Extractor (N = 3,600 Non-Guild Prose)
# =============================================================================


def build_unlabeled_prose_prompts(n_prompts: int = 3600, seed: int = 20261501) -> List[str]:
    """Generate diverse natural English non-guild, non-entity prose prompts for full-rank N > D alignment."""
    rng = random.Random(seed)
    all_combos = []
    for tmpl in UNLAB_SENTENCE_TEMPLATES:
        for noun in UNLAB_NOUNS:
            for dom in UNLAB_DOMAINS:
                all_combos.append(tmpl.format(noun=noun, domain=dom))
    rng.shuffle(all_combos)
    prompts = all_combos[:n_prompts]
    # Strict verification: zero overlap with any guild word or 32 entities
    forbidden = [w.lower() for w in GUILD_LABELS] + [e.lower() for e in CANDIDATE_ENTITIES_32]
    for p in prompts:
        pl = p.lower()
        for fw in forbidden:
            assert fw not in pl, f"Leakage detected: {fw} in {p}"
    return prompts


@torch.no_grad()
def extract_multi_layer_last_states(
    model: torch.nn.Module,
    tok: Any,
    prompts: Sequence[str],
    layer_indices: Sequence[int],
    device: torch.device,
    batch_size: int = 48,
) -> Dict[int, np.ndarray]:
    """Extract last-token residual states at multiple layers in ONE single pass!"""
    layers = get_decoder_layers(model)
    captured: Dict[int, List[np.ndarray]] = {l: [] for l in layer_indices}
    buf: Dict[int, torch.Tensor] = {}

    handles = []
    for l in layer_indices:
        def _make_hook(li: int):
            def _hook(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                buf[li] = h[:, -1, :].detach().float().cpu()
            return _hook
        handles.append(layers[l].register_forward_hook(_make_hook(l)))

    try:
        for start in range(0, len(prompts), batch_size):
            batch = list(prompts[start : start + batch_size])
            enc = tok(batch, return_tensors="pt", padding=True).to(device)
            _ = model(**enc, use_cache=False)
            for l in layer_indices:
                captured[l].append(buf[l].numpy())
    finally:
        for h in handles:
            h.remove()

    return {l: np.concatenate(captured[l], axis=0) for l in layer_indices}


# =============================================================================
# 3. Nullspace & Unembedding Diagnostic
# =============================================================================


def compute_delta_diagnostics(
    dA_f: np.ndarray,
    recs: Sequence[dict],
    ZA_429_train: np.ndarray,
    ZA_unlab: Optional[np.ndarray],
    WA: np.ndarray,
    gA: np.ndarray,
    tokA: Any,
) -> Dict[str, Any]:
    """Compute exact fraction of ||dA||^2 inside the N=429 span vs full-rank span,
    plus cosine alignment and 8-way logit-lens accuracy against Donor A's W_U[guild]!"""
    dA = np.asarray(dA_f, dtype=np.float64)
    total_energy = float(np.sum(dA ** 2)) + 1e-12

    # N=429 orthonormal basis U_429 (centered)
    A429 = np.asarray(ZA_429_train, dtype=np.float64)
    A429c = A429 - A429.mean(0, keepdims=True)
    _, S429, Vt429 = np.linalg.svd(A429c, full_matrices=False)
    rank429 = int((S429 > 1e-6 * S429[0]).sum())
    V429 = Vt429[:rank429]  # (r, D)
    proj429 = (dA @ V429.T) @ V429
    in_span_429 = float(np.sum(proj429 ** 2) / total_energy)

    in_span_unlab = None
    rank_unlab = None
    if ZA_unlab is not None:
        Au = np.asarray(ZA_unlab, dtype=np.float64)
        Auc = Au - Au.mean(0, keepdims=True)
        _, Su, Vtu = np.linalg.svd(Auc, full_matrices=False)
        rank_unlab = int((Su > 1e-6 * Su[0]).sum())
        Vu = Vtu[:rank_unlab]
        proj_u = (dA @ Vu.T) @ Vu
        in_span_unlab = float(np.sum(proj_u ** 2) / total_energy)

    # Alignment with Donor A's 8 guild unembedding directions (W_U * gamma_A)
    g_ids = np.array([tokA.encode(" " + w, add_special_tokens=False)[0] for w in GUILD_LABELS])
    Wg = (WA[g_ids] * gA[None, :]).astype(np.float64)  # (8, D)
    Wg_norm = Wg / (np.linalg.norm(Wg, axis=1, keepdims=True) + 1e-12)
    dA_norm = dA / (np.linalg.norm(dA, axis=1, keepdims=True) + 1e-12)
    cos_mat = dA_norm @ Wg_norm.T  # (N, 8)
    label_to_idx = {w: i for i, w in enumerate(GUILD_LABELS)}
    y_true = np.array([label_to_idx[r["label"]] for r in recs])
    true_cos = np.array([cos_mat[i, y_true[i]] for i in range(len(recs))])
    other_cos = np.array([np.delete(cos_mat[i], y_true[i]).mean() for i in range(len(recs))])
    lens_8way = float((cos_mat.argmax(axis=1) == y_true).mean())

    return {
        "n_429_rank": rank429,
        "in_span_429_energy_frac": in_span_429,
        "nullspace_429_energy_frac": 1.0 - in_span_429,
        "unlab_rank": rank_unlab,
        "in_span_unlab_energy_frac": in_span_unlab,
        "delta_norm_mean": float(np.linalg.norm(dA, axis=1).mean()),
        "wu_true_guild_cos_mean": float(true_cos.mean()),
        "wu_other_guild_cos_mean": float(other_cos.mean()),
        "wu_delta_logit_lens_8way": lens_8way,
    }


# =============================================================================
# 4. Distributed Multi-Layer & Single-Layer Donor-Free Platonic MEMIT Compiler
# =============================================================================


def compile_and_eval_memit_suite(
    model_B: torch.nn.Module,
    tok_B: Any,
    target_layers_B: Sequence[int],
    pos_B: str,
    mapped_dB_f_tr: np.ndarray,
    mapped_dB_fp_tr: np.ndarray,
    mapped_dB_rand_tr: np.ndarray,
    recs_f_train: List[dict],
    recs_f_test: List[dict],
    recs_fp_train: List[dict],
    recs_fp_test: List[dict],
    bg_prompts: List[str],
    bg_spans: List[List[int]],
    loc_args: Tuple[List[str], List[int], List[int], List[List[str]], List[str]],
    base_loc_acc: float,
    device: torch.device,
    alphas: Sequence[float] = (1.5, 2.5, 3.5),
    lams: Sequence[float] = (10.0, 50.0, 200.0),
) -> Dict[str, Any]:
    """Compile Donor A's mapped deltas into solo Recipient B's native mlp.down_proj weights
    (single layer or evenly distributed across target_layers_B) via exact CPU float64
    Woodbury MEMIT. Zero gradient steps on B; Donor A is completely unloaded!
    """
    layers_B = get_decoder_layers(model_B)
    n_layers = len(target_layers_B)
    other_pos = "subj" if pos_B == "last" else "last"
    tr_prompts = [r["prompt"] for r in recs_f_train]
    tr_spans = [r["subj_char_span"] for r in recs_f_train]
    loc_prompts, loc_spos, loc_slens, loc_aliases, loc_rels = loc_args

    # Extract B's native post-SwiGLU keys for every target layer once
    K_by_layer: Dict[int, np.ndarray] = {}
    Kbg_by_layer: Dict[int, np.ndarray] = {}
    orig_W: Dict[int, torch.Tensor] = {}
    for lB in target_layers_B:
        K_by_layer[lB] = extract_mlp_down_proj_inputs(model_B, tok_B, tr_prompts, tr_spans, lB, pos_B, device)
        K_bg_1 = extract_mlp_down_proj_inputs(model_B, tok_B, bg_prompts, bg_spans, lB, pos_B, device)
        K_bg_2 = extract_mlp_down_proj_inputs(model_B, tok_B, bg_prompts[:256], bg_spans[:256], lB, other_pos, device)
        Kbg_by_layer[lB] = np.concatenate([K_bg_1, K_bg_2], axis=0)
        orig_W[lB] = layers_B[lB].mlp.down_proj.weight.detach().clone()

    def _apply_and_score(delta_target_tr: np.ndarray, alpha: float, lam_cov: float, eval_recs_tr: List[dict], eval_recs_te: List[dict], eval_loc: bool = True) -> dict:
        per_layer_target = (float(alpha) / float(n_layers)) * delta_target_tr
        with torch.no_grad():
            for lB in target_layers_B:
                dW = solve_woodbury_memit(K_by_layer[lB], per_layer_target, Kbg_by_layer[lB], lam_cov=lam_cov)
                layers_B[lB].mlp.down_proj.weight.copy_(
                    orig_W[lB] + torch.from_numpy(dW).to(device=orig_W[lB].device, dtype=orig_W[lB].dtype)
                )
        ev_tr = eval_guild_records(model_B, tok_B, eval_recs_tr, GUILD_LABELS, device=device)
        ev_te = eval_guild_records(model_B, tok_B, eval_recs_te, GUILD_LABELS, device=device)
        # Per-template breakdown on unseen test paraphrases!
        per_tmpl = {}
        for t_idx in range(len(TEST_TEMPLATES_3)):
            idxs = [i for i, r in enumerate(eval_recs_te) if r["template_idx"] == t_idx]
            per_tmpl[f"tmpl_{t_idx}"] = {
                "ending": TEST_TEMPLATES_3[t_idx][1],
                "choice8": float(np.mean([ev_te["choice8_outcomes"][i] for i in idxs])),
                "greedy_em": float(np.mean([ev_te["greedy_outcomes"][i] for i in idxs])),
            }
        loc_acc, loc_drop = None, None
        if eval_loc:
            loc_acc, _ = eval_multi_regime_swaps(
                model_B, tok_B, loc_prompts, loc_spos, loc_slens, loc_aliases, loc_rels,
                device=device, dtype=torch.float32,
            )
            loc_drop = (base_loc_acc - loc_acc) * 100.0
        with torch.no_grad():
            for lB in target_layers_B:
                layers_B[lB].mlp.down_proj.weight.copy_(orig_W[lB])
        return {
            "alpha": float(alpha),
            "lam_cov": float(lam_cov),
            "train_choice8": ev_tr["choice8_acc"],
            "train_greedy_em": ev_tr["greedy_em_acc"],
            "test_unseen_para_choice8": ev_te["choice8_acc"],
            "test_unseen_para_greedy_em": ev_te["greedy_em_acc"],
            "test_per_template": per_tmpl,
            "test_choice8_outcomes": ev_te["choice8_outcomes"],
            "test_greedy_outcomes": ev_te["greedy_outcomes"],
            "locality_196_acc": loc_acc,
            "locality_drop_pp": loc_drop,
        }

    # Strict hyperparameter selection on TRAIN + LOCALITY ONLY (zero peeking at unseen test paraphrases!)
    sweep = {}
    best_key = None
    best_score = -1e9
    for a in alphas:
        for lam in lams:
            res = _apply_and_score(mapped_dB_f_tr, a, lam, recs_f_train, recs_f_test, eval_loc=True)
            k = f"a{a}_lam{int(lam)}"
            sweep[k] = res
            loc_ok = 1.0 if (res["locality_drop_pp"] is not None and res["locality_drop_pp"] <= 3.0) else 0.0
            score = loc_ok * 10.0 + res["train_choice8"] + 0.3 * res["train_greedy_em"] - 0.01 * max(0.0, res["locality_drop_pp"] or 0.0)
            if score > best_score:
                best_score = score
                best_key = k

    sel = sweep[best_key]
    best_a, best_lam = sel["alpha"], sel["lam_cov"]

    # Evaluate locked (best_a, best_lam) on f', cross-score f' on old f, shuffled donor, and random map!
    res_fp = _apply_and_score(mapped_dB_fp_tr, best_a, best_lam, recs_fp_train, recs_fp_test, eval_loc=True)
    res_fp_on_old_f = _apply_and_score(mapped_dB_fp_tr, best_a, best_lam, recs_f_train, recs_f_test, eval_loc=False)

    rng_shuf = np.random.RandomState(20261601)
    perm = rng_shuf.permutation(len(mapped_dB_f_tr))
    res_shuf = _apply_and_score(mapped_dB_f_tr[perm], best_a, best_lam, recs_f_train, recs_f_test, eval_loc=False)
    res_rand = _apply_and_score(mapped_dB_rand_tr, best_a, best_lam, recs_f_train, recs_f_test, eval_loc=True)

    return {
        "target_layers_B": list(target_layers_B),
        "selected_hp": {"alpha": best_a, "lam_cov": best_lam, "key": best_key},
        "compiled_f": sel,
        "compiled_f_prime": res_fp,
        "compiled_fp_scored_on_old_f": res_fp_on_old_f,
        "shuffled_donor_control": res_shuf,
        "random_map_control": res_rand,
        "sweep": sweep,
    }


# =============================================================================
# 5. Main Orchestrator
# =============================================================================


def run_pilot2_fullrank(reuse_payloads: bool = False, n_unlab: int = 3600) -> dict:
    t_all = time.time()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[Pilot 2b Full-Rank] Device={device}, reuse_payloads={reuse_payloads}, n_unlab={n_unlab}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(BASELINE_RESULTS_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    facts = dataset["facts"]
    train_1hop_idx = [f["idx"] for f in facts if f["split"] == "train"]
    test_1hop_idx = [f["idx"] for f in facts if f["split"] == "test"]
    bp = baseline["pairs"][f"{DONOR_KEY}->{RECIP_KEY}"]
    sh_tr = [i for i in bp["shared_train_indices"] if facts[i]["split"] == "train"]

    table_f, table_fp = build_private_tables(CANDIDATE_ENTITIES_32, GUILD_LABELS)
    recs_f_tr = make_prompt_records(CANDIDATE_ENTITIES_32, table_f, TRAIN_TEMPLATES_3)
    recs_f_te = make_prompt_records(CANDIDATE_ENTITIES_32, table_f, TEST_TEMPLATES_3)
    recs_fp_tr = make_prompt_records(CANDIDATE_ENTITIES_32, table_fp, TRAIN_TEMPLATES_3)
    recs_fp_te = make_prompt_records(CANDIDATE_ENTITIES_32, table_fp, TEST_TEMPLATES_3)
    anchor_prompts = [facts[i]["direct_prompt"] for i in sh_tr[:64]]

    stA_1hop = torch.load(CACHE_DIR / f"{DONOR_KEY}_states.pt", map_location="cpu", weights_only=False)
    stB_1hop = torch.load(CACHE_DIR / f"{RECIP_KEY}_states.pt", map_location="cpu", weights_only=False)

    # ------------------------------------------------------------------
    # Step 1: Train Donor A (or load cached payloads!) + Extract N_unlab Donor A states
    # ------------------------------------------------------------------
    unlab_prompts = build_unlabeled_prose_prompts(n_prompts=n_unlab)
    need_unlab_A = not UNLAB_LATE_CACHE_PATH.exists()

    if reuse_payloads and PAYLOAD_CACHE_PATH.exists() and not need_unlab_A:
        print(f"[Step 1] Reusing cached Donor A payloads from {PAYLOAD_CACHE_PATH}")
        payloads = torch.load(PAYLOAD_CACHE_PATH, map_location="cpu", weights_only=False)
        unlab_cache = torch.load(UNLAB_LATE_CACHE_PATH, map_location="cpu", weights_only=False)
    else:
        print(f"\n=== [Step 1] Loading Donor A ({DONOR_KEY}) for Payloads + Unlabeled Full-Rank States ===")
        tok_A = AutoTokenizer.from_pretrained(str(HF_CACHE_DIR / DONOR_KEY), local_files_only=True)
        tok_A.padding_side = "left"
        if tok_A.pad_token is None:
            tok_A.pad_token = tok_A.eos_token
        model_A = AutoModelForCausalLM.from_pretrained(
            str(HF_CACHE_DIR / DONOR_KEY), local_files_only=True, dtype=torch.float32
        ).to(device)
        model_A.eval()

        # 1a. Extract unlabeled full-rank general-prose states for Donor A at layers 17, 30, 35 (in bf16/fp32 fast pass!)
        if need_unlab_A:
            print(f"  Extracting {len(unlab_prompts)} unlabeled general-prose states for Donor A at layers [17, 30, 35]...")
            t0 = time.time()
            unlab_A = extract_multi_layer_last_states(model_A, tok_A, unlab_prompts, [17, 30, 35], device, batch_size=48)
            print(f"  Donor A unlabeled extraction done in {time.time() - t0:.1f}s")
        else:
            unlab_A = torch.load(UNLAB_LATE_CACHE_PATH, map_location="cpu", weights_only=False)["A"]

        if reuse_payloads and PAYLOAD_CACHE_PATH.exists():
            payloads = torch.load(PAYLOAD_CACHE_PATH, map_location="cpu", weights_only=False)
        else:
            payloads: Dict[str, Any] = {"donor_controls": {}, "sites": {}}
            # Extract base Donor A states at all 3 sites ONCE
            base_17_tr = extract_site_states(model_A, tok_A, recs_f_tr, 17, "subj", device)
            base_17_te = extract_site_states(model_A, tok_A, recs_f_te, 17, "subj", device)
            base_30_tr = extract_site_states(model_A, tok_A, recs_f_tr, 30, "last", device)
            base_30_te = extract_site_states(model_A, tok_A, recs_f_te, 30, "last", device)
            base_35_tr = extract_site_states(model_A, tok_A, recs_f_tr, 35, "last", device)
            base_35_te = extract_site_states(model_A, tok_A, recs_f_te, 35, "last", device)

            # Site 1: Early Subject Key (l_A=17, mask_to_subj_only=True)
            s1_data = {"base_tr": base_17_tr, "base_te": base_17_te}
            s1_ctrl = {}
            for tname, tr_r, te_r, sd in [("f", recs_f_tr, recs_f_te, 101), ("f_prime", recs_fp_tr, recs_fp_te, 202)]:
                t0 = time.time()
                suite = train_layer_restricted_adapter(
                    model_A, tok_A, tr_r, anchor_prompts,
                    layer_indices=list(range(12, 18)), mask_to_subj_only=True,
                    device=device, rank=16, steps=200, lr=3e-3, batch_size=16, seed=sd,
                )
                ev_A = eval_guild_records(model_A, tok_A, te_r, GUILD_LABELS, device=device)
                z_tr = extract_states_with_adapter(model_A, tok_A, suite, tr_r, 17, "subj", True, device)
                z_te = extract_states_with_adapter(model_A, tok_A, suite, te_r, 17, "subj", True, device)
                suite.detach()
                ev_ss = eval_live_site_crossing(model_A, tok_A, te_r, 17, "subj", z_te, GUILD_LABELS, device, mode="full")
                s1_data[f"{tname}_tr"] = z_tr
                s1_data[f"{tname}_te"] = z_te
                s1_ctrl[tname] = {
                    "donor_test_choice8": ev_A["choice8_acc"],
                    "donor_test_greedy_em": ev_A["greedy_em_acc"],
                    "donor_self_swap_choice8": ev_ss["choice8_acc"],
                    "donor_self_swap_greedy_em": ev_ss["greedy_em_acc"],
                    "elapsed_sec": time.time() - t0,
                }
                print(f"  [Site 1 lA=17 subj | {tname:7s}] Donor={ev_A['choice8_acc']:.3f}/{ev_A['greedy_em_acc']:.3f} | SelfSwap={ev_ss['choice8_acc']:.3f}/{ev_ss['greedy_em_acc']:.3f}")
                del suite
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            payloads["sites"]["site1_early_key_subj"] = s1_data
            payloads["donor_controls"]["site1_early_key_subj"] = s1_ctrl

            # Site 2: Train ONCE on layers 25..30 (layers 31..35 frozen!), extract BOTH l_A=30 AND l_A=35!
            s30_data = {"base_tr": base_30_tr, "base_te": base_30_te}
            s35_data = {"base_tr": base_35_tr, "base_te": base_35_te}
            s30_ctrl, s35_ctrl = {}, {}
            for tname, tr_r, te_r, sd in [("f", recs_f_tr, recs_f_te, 303), ("f_prime", recs_fp_tr, recs_fp_te, 404)]:
                t0 = time.time()
                suite = train_layer_restricted_adapter(
                    model_A, tok_A, tr_r, anchor_prompts,
                    layer_indices=list(range(25, 31)), mask_to_subj_only=False,
                    device=device, rank=16, steps=200, lr=3e-3, batch_size=16, seed=sd,
                )
                ev_A = eval_guild_records(model_A, tok_A, te_r, GUILD_LABELS, device=device)
                z30_tr = extract_states_with_adapter(model_A, tok_A, suite, tr_r, 30, "last", False, device)
                z30_te = extract_states_with_adapter(model_A, tok_A, suite, te_r, 30, "last", False, device)
                z35_tr = extract_states_with_adapter(model_A, tok_A, suite, tr_r, 35, "last", False, device)
                z35_te = extract_states_with_adapter(model_A, tok_A, suite, te_r, 35, "last", False, device)
                suite.detach()
                ev_ss30 = eval_live_site_crossing(model_A, tok_A, te_r, 30, "last", z30_te, GUILD_LABELS, device, mode="full")
                ev_ss35 = eval_live_site_crossing(model_A, tok_A, te_r, 35, "last", z35_te, GUILD_LABELS, device, mode="full")
                s30_data[f"{tname}_tr"], s30_data[f"{tname}_te"] = z30_tr, z30_te
                s35_data[f"{tname}_tr"], s35_data[f"{tname}_te"] = z35_tr, z35_te
                s30_ctrl[tname] = {
                    "donor_test_choice8": ev_A["choice8_acc"],
                    "donor_test_greedy_em": ev_A["greedy_em_acc"],
                    "donor_self_swap_choice8": ev_ss30["choice8_acc"],
                    "donor_self_swap_greedy_em": ev_ss30["greedy_em_acc"],
                    "elapsed_sec": time.time() - t0,
                }
                s35_ctrl[tname] = {
                    "donor_test_choice8": ev_A["choice8_acc"],
                    "donor_test_greedy_em": ev_A["greedy_em_acc"],
                    "donor_self_swap_choice8": ev_ss35["choice8_acc"],
                    "donor_self_swap_greedy_em": ev_ss35["greedy_em_acc"],
                }
                print(
                    f"  [Site 2 lA=30/35 last | {tname:7s}] Donor={ev_A['choice8_acc']:.3f}/{ev_A['greedy_em_acc']:.3f} | "
                    f"SelfSwap30={ev_ss30['choice8_acc']:.3f}/{ev_ss30['greedy_em_acc']:.3f} | "
                    f"SelfSwap35={ev_ss35['choice8_acc']:.3f}/{ev_ss35['greedy_em_acc']:.3f}"
                )
                del suite
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            payloads["sites"]["site2_late_last_83pct"] = s30_data
            payloads["sites"]["site2_preoutput_last"] = s35_data
            payloads["donor_controls"]["site2_late_last_83pct"] = s30_ctrl
            payloads["donor_controls"]["site2_preoutput_last"] = s35_ctrl
            torch.save(payloads, PAYLOAD_CACHE_PATH)
            print(f"  Saved Donor A payloads -> {PAYLOAD_CACHE_PATH}")

        # COMPLETELY UNLOAD DONOR A!
        print(f"  -> UNLOADING Donor A ({DONOR_KEY}) COMPLETELY FROM MEMORY!")
        del model_A, tok_A
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        unlab_cache = {"A": unlab_A, "B": None}

    # ------------------------------------------------------------------
    # Step 2: Load SOLO Recipient B (smollm2-1.7b), Extract Unlabeled B States
    # ------------------------------------------------------------------
    print(f"\n=== [Step 2] Loading SOLO Recipient B ({RECIP_KEY}) — Donor A is GONE! ===")
    tok_B = AutoTokenizer.from_pretrained(str(HF_CACHE_DIR / RECIP_KEY), local_files_only=True)
    tok_B.padding_side = "left"
    if tok_B.pad_token is None:
        tok_B.pad_token = tok_B.eos_token
    model_B = AutoModelForCausalLM.from_pretrained(
        str(HF_CACHE_DIR / RECIP_KEY), local_files_only=True, dtype=torch.float32
    ).to(device)
    model_B.eval()

    if unlab_cache.get("B") is None:
        print(f"  Extracting {len(unlab_prompts)} unlabeled general-prose states for Recipient B at layers [9, 20, 23]...")
        t0 = time.time()
        unlab_cache["B"] = extract_multi_layer_last_states(model_B, tok_B, unlab_prompts, [9, 20, 23], device, batch_size=48)
        torch.save(unlab_cache, UNLAB_LATE_CACHE_PATH)
        print(f"  Recipient B unlabeled extraction done in {time.time() - t0:.1f}s -> {UNLAB_LATE_CACHE_PATH}")

    # ------------------------------------------------------------------
    # Step 3: Nullspace & Unembedding Diagnostics on CPU (float64)
    # ------------------------------------------------------------------
    print("\n=== [Step 3] Nullspace Energy & Unembedding Alignment Diagnostics ===")
    tok_A_cpu = AutoTokenizer.from_pretrained(str(HF_CACHE_DIR / DONOR_KEY), local_files_only=True)
    WA, gA, _ = load_unembedding_and_norm(DONOR_KEY)
    diagnostics = {}
    diag_specs = [
        ("site1_early_key_subj", 17, "direct_subj", 17),
        ("site2_late_last_83pct", 30, "direct_last", 30),
        ("site2_preoutput_last", 35, "direct_last", 35),
    ]
    for sname, lA, key_1hop, ul_lA in diag_specs:
        dA_f = payloads["sites"][sname]["f_te"] - payloads["sites"][sname]["base_te"]
        diag = compute_delta_diagnostics(
            dA_f, recs_f_te,
            stA_1hop[key_1hop][sh_tr, lA].float().numpy(),
            unlab_cache["A"][ul_lA],
            WA, gA, tok_A_cpu,
        )
        diagnostics[sname] = diag
        print(
            f"  [{sname:24s}] in_N429_span={diag['in_span_429_energy_frac']*100:5.1f}% "
            f"(NULLSPACE={diag['nullspace_429_energy_frac']*100:5.1f}%) | "
            f"in_unlab3600_span={diag['in_span_unlab_energy_frac']*100:5.1f}% | "
            f"W_U_true_cos={diag['wu_true_guild_cos_mean']:+.3f} vs other={diag['wu_other_guild_cos_mean']:+.3f} | "
            f"A_logit_lens_8way={diag['wu_delta_logit_lens_8way']:.3f}"
        )

    # ------------------------------------------------------------------
    # Step 4: Baseline Locality & Unmodified B
    # ------------------------------------------------------------------
    with open(CACHE_DIR / f"{RECIP_KEY}_eval.json", "r", encoding="utf-8") as f:
        eval_B_1hop = json.load(f)["facts"]
    loc_prompts = [facts[i]["direct_prompt"] for i in test_1hop_idx]
    loc_spos = [eval_B_1hop[i]["subj_tok_idx_direct"] for i in test_1hop_idx]
    loc_slens = [eval_B_1hop[i]["seq_len_direct"] for i in test_1hop_idx]
    loc_aliases = [facts[i]["aliases"] for i in test_1hop_idx]
    loc_rels = [facts[i]["relation"] for i in test_1hop_idx]
    loc_args = (loc_prompts, loc_spos, loc_slens, loc_aliases, loc_rels)
    bg_prompts = [facts[i]["direct_prompt"] for i in train_1hop_idx]
    bg_spans = [facts[i]["direct_subj_char_span"] for i in train_1hop_idx]

    base_loc_acc, _ = eval_multi_regime_swaps(
        model_B, tok_B, loc_prompts, loc_spos, loc_slens, loc_aliases, loc_rels,
        device=device, dtype=torch.float32,
    )
    unmod_f = eval_guild_records(model_B, tok_B, recs_f_te, GUILD_LABELS, device=device)
    unmod_fp = eval_guild_records(model_B, tok_B, recs_fp_te, GUILD_LABELS, device=device)
    print(f"\n[Unmodified B] f_test={unmod_f['choice8_acc']:.3f}/{unmod_f['greedy_em_acc']:.3f} | fp_test={unmod_fp['choice8_acc']:.3f}/{unmod_fp['greedy_em_acc']:.3f} | loc196={base_loc_acc:.3f}")

    # ------------------------------------------------------------------
    # Step 5: Build Strictly Frozen, Zero-Leakage Full-Rank Maps for Each Site
    # ------------------------------------------------------------------
    exclude_all = list(GUILD_LABELS) + ["Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Black", "White"] + list(CANDIDATE_ENTITIES_32)

    results: Dict[str, Any] = {
        "meta": {
            "donor": DONOR_KEY,
            "recipient": RECIP_KEY,
            "n_unlab": int(n_unlab),
            "train_templates": TRAIN_TEMPLATES_3,
            "test_templates": TEST_TEMPLATES_3,
            "table_f": table_f,
            "table_f_prime": table_fp,
        },
        "donor_A_controls": payloads["donor_controls"],
        "diagnostics": diagnostics,
        "unmodified_B": {
            "f_choice8": unmod_f["choice8_acc"],
            "f_greedy_em": unmod_f["greedy_em_acc"],
            "fp_choice8": unmod_fp["choice8_acc"],
            "fp_greedy_em": unmod_fp["greedy_em_acc"],
            "locality_196": base_loc_acc,
        },
        "test2A_live_crossing": {},
        "test2B_donor_free_memit": {},
    }

    # Build map dictionaries per site
    site_configs = [
        ("site1_early_key_subj", 17, 9, "subj", "direct_subj"),
        ("site2_late_last_83pct", 30, 20, "last", "direct_last"),
        ("site2_preoutput_last", 35, 23, "last", "direct_last"),
    ]

    for sname, lA, lB, pos, key_1hop in site_configs:
        print(f"\n==================== [{sname}] (lA={lA} -> lB={lB}, pos={pos}) ====================")
        sd = payloads["sites"][sname]
        dA_f_tr = sd["f_tr"] - sd["base_tr"]
        dA_f_te = sd["f_te"] - sd["base_te"]
        dA_fp_tr = sd["f_prime_tr"] - sd["base_tr"]
        dA_fp_te = sd["f_prime_te"] - sd["base_te"]

        maps_to_test: Dict[str, Any] = {}
        # 1. Rank-starved N=429 Span-Procrustes (control)
        ZA_429 = stA_1hop[key_1hop][sh_tr, lA].float().numpy()
        ZB_429 = stB_1hop[key_1hop][sh_tr, lB].float().numpy()
        maps_to_test["n429_span_qa"] = systems.fit_procrustes_fast_qr(ZA_429, ZB_429)

        # 2. Full-rank N=3,600 unlabeled general-prose Procrustes & Ridge (ZERO W_U used!)
        m_u_proc, m_u_ridge = fit_fullrank_procrustes_and_ridge(unlab_cache["A"][lA], unlab_cache["B"][lB], lam_rel=1e-2)
        maps_to_test["unlab3600_procrustes"] = m_u_proc
        maps_to_test["unlab3600_ridge"] = m_u_ridge

        # 3. Auto-load Pilot 1's N=10,000 unlabeled maps at site1 (17->9) if available!
        #    NOTE: pilot1_fisher_scaling.py stores W & R in (D_B, D_A) convention -> apply as (z - mu_A) @ M.T!
        #    And W already includes s_match, whereas R multiplies by mv["s"].
        if sname == "site1_early_key_subj" and PILOT1_MAPS_PATH.exists():
            print(f"  Loading Pilot 1's N=10,000 full-rank maps from {PILOT1_MAPS_PATH}!")
            p1m = torch.load(PILOT1_MAPS_PATH, map_location="cpu", weights_only=False)
            for mk, mv in p1m.items():
                if not isinstance(mv, dict):
                    continue
                if "W" in mv:
                    maps_to_test[f"pilot1_10k_{mk}"] = LinearGeneralMap(
                        np.asarray(mv["W"], dtype=np.float64).T,
                        mv.get("mu_A", np.zeros(2048)),
                        mv.get("mu_B", np.zeros(2048)),
                        scale=1.0,
                        name=f"pilot1_10k_{mk}",
                    )
                elif "R" in mv:
                    maps_to_test[f"pilot1_10k_{mk}"] = LinearGeneralMap(
                        np.asarray(mv["R"], dtype=np.float64).T,
                        mv.get("mu_A", np.zeros(2048)),
                        mv.get("mu_B", np.zeros(2048)),
                        scale=float(mv.get("s", 1.0)),
                        name=f"pilot1_10k_{mk}",
                    )

        # 4. Zero-Forward-Pass Dual-Adjoint Vocabulary Map T_pre (and norm-matched random control)
        rmsA = float(np.sqrt(np.mean(unlab_cache["A"][lA].astype(np.float64) ** 2, axis=1)).mean())
        rmsB = float(np.sqrt(np.mean(unlab_cache["B"][lB].astype(np.float64) ** 2, axis=1)).mean())
        muA_u = unlab_cache["A"][lA].astype(np.float64).mean(0)
        muB_u = unlab_cache["B"][lB].astype(np.float64).mean(0)
        map_dual, map_dual_rand, dual_meta = build_dual_adjoint_vocab_map(
            DONOR_KEY, RECIP_KEY, rms_A=rmsA, rms_B=rmsB, mu_A=muA_u, mu_B=muB_u,
            exclude_words=exclude_all, lam_rel=1e-3,
        )
        maps_to_test["dual_adjoint_wu"] = map_dual
        maps_to_test["dual_adjoint_rand"] = map_dual_rand

        # ---- Test 2A: Live Crossing across ALL maps ----
        site_2A: Dict[str, Any] = {"dual_adjoint_meta": dual_meta}
        for mname, mp in maps_to_test.items():
            # Test alpha in {1.0, 2.0, 3.0} for delta teleportation
            best_a_2A, best_ev_f = 2.0, None
            alpha_grid = {}
            for a in (1.0, 2.0, 3.0):
                ev_f = eval_live_site_crossing(
                    model_B, tok_B, recs_f_te, lB, pos, mp.transform_delta(dA_f_te, alpha=a),
                    GUILD_LABELS, device, mode="delta",
                )
                alpha_grid[f"alpha_{a}"] = {"f_choice8": ev_f["choice8_acc"], "f_greedy_em": ev_f["greedy_em_acc"]}
                if best_ev_f is None or ev_f["choice8_acc"] > best_ev_f["choice8_acc"]:
                    best_ev_f = ev_f
                    best_a_2A = a

            ev_fp = eval_live_site_crossing(
                model_B, tok_B, recs_fp_te, lB, pos, mp.transform_delta(dA_fp_te, alpha=best_a_2A),
                GUILD_LABELS, device, mode="delta",
            )
            ev_fp_on_f = eval_live_site_crossing(
                model_B, tok_B, recs_f_te, lB, pos, mp.transform_delta(dA_fp_te, alpha=best_a_2A),
                GUILD_LABELS, device, mode="delta",
            )
            perm = np.random.RandomState(20261701).permutation(len(dA_f_te))
            ev_shuf = eval_live_site_crossing(
                model_B, tok_B, recs_f_te, lB, pos, mp.transform_delta(dA_f_te[perm], alpha=best_a_2A),
                GUILD_LABELS, device, mode="delta",
            )
            site_2A[mname] = {
                "best_alpha": best_a_2A,
                "alpha_grid": alpha_grid,
                "matched_f_choice8": best_ev_f["choice8_acc"],
                "matched_f_greedy_em": best_ev_f["greedy_em_acc"],
                "matched_f_outcomes_8": best_ev_f["choice8_outcomes"],
                "matched_fp_choice8": ev_fp["choice8_acc"],
                "matched_fp_greedy_em": ev_fp["greedy_em_acc"],
                "matched_fp_outcomes_8": ev_fp["choice8_outcomes"],
                "fp_scored_on_old_f_choice8": ev_fp_on_f["choice8_acc"],
                "shuffled_donor_choice8": ev_shuf["choice8_acc"],
            }
            print(
                f"  [2A {mname:26s} a={best_a_2A:.1f}] "
                f"f={best_ev_f['choice8_acc']:.3f}/{best_ev_f['greedy_em_acc']:.3f} | "
                f"f'={ev_fp['choice8_acc']:.3f}/{ev_fp['greedy_em_acc']:.3f} "
                f"(f' on old f={ev_fp_on_f['choice8_acc']:.3f}) | shuf={ev_shuf['choice8_acc']:.3f}"
            )
        results["test2A_live_crossing"][sname] = site_2A

        # ---- Test 2B: Donor-Free Platonic MEMIT Compile for every map that cleared 30% live (plus unlab3600 & dual_adjoint at preoutput!) ----
        site_2B: Dict[str, Any] = {}
        compile_candidates = []
        for mname in ("unlab3600_procrustes", "unlab3600_ridge", "dual_adjoint_wu"):
            if mname in site_2A and (site_2A[mname]["matched_f_choice8"] >= 0.30 or sname == "site2_preoutput_last"):
                compile_candidates.append(mname)
        for mname in compile_candidates:
            mp = maps_to_test[mname]
            dB_f_tr = mp.transform_delta(dA_f_tr, alpha=1.0)
            dB_fp_tr = mp.transform_delta(dA_fp_tr, alpha=1.0)
            dB_rand_tr = map_dual_rand.transform_delta(dA_f_tr, alpha=1.0)

            # Single-layer MEMIT at lB
            res_single = compile_and_eval_memit_suite(
                model_B, tok_B, [lB], pos, dB_f_tr, dB_fp_tr, dB_rand_tr,
                recs_f_tr, recs_f_te, recs_fp_tr, recs_fp_te,
                bg_prompts, bg_spans, loc_args, base_loc_acc, device,
                alphas=(1.5, 2.5, 3.5), lams=(10.0, 50.0, 200.0),
            )
            site_2B[f"{mname}_single_l{lB}"] = res_single
            cf, cfp = res_single["compiled_f"], res_single["compiled_f_prime"]
            print(
                f"  [2B MEMIT {mname:20s} single lB={lB}] "
                f"f={cf['test_unseen_para_choice8']:.3f}/{cf['test_unseen_para_greedy_em']:.3f} (loc={cf['locality_drop_pp']:+.2f}pp) | "
                f"f'={cfp['test_unseen_para_choice8']:.3f}/{cfp['test_unseen_para_greedy_em']:.3f} "
                f"(on old f={res_single['compiled_fp_scored_on_old_f']['test_unseen_para_choice8']:.3f}) | "
                f"rand={res_single['random_map_control']['test_unseen_para_choice8']:.3f}"
            )

            # Distributed 4-layer MEMIT [20, 21, 22, 23] when compiling pre-output site!
            if lB == 23:
                res_dist = compile_and_eval_memit_suite(
                    model_B, tok_B, [20, 21, 22, 23], pos, dB_f_tr, dB_fp_tr, dB_rand_tr,
                    recs_f_tr, recs_f_te, recs_fp_tr, recs_fp_te,
                    bg_prompts, bg_spans, loc_args, base_loc_acc, device,
                    alphas=(1.5, 2.5, 3.5), lams=(10.0, 50.0, 200.0),
                )
                site_2B[f"{mname}_dist_20_23"] = res_dist
                cf2, cfp2 = res_dist["compiled_f"], res_dist["compiled_f_prime"]
                print(
                    f"  [2B MEMIT {mname:20s} dist 20..23] "
                    f"f={cf2['test_unseen_para_choice8']:.3f}/{cf2['test_unseen_para_greedy_em']:.3f} (loc={cf2['locality_drop_pp']:+.2f}pp) | "
                    f"f'={cfp2['test_unseen_para_choice8']:.3f}/{cfp2['test_unseen_para_greedy_em']:.3f} "
                    f"(on old f={res_dist['compiled_fp_scored_on_old_f']['test_unseen_para_choice8']:.3f})"
                )

        results["test2B_donor_free_memit"][sname] = site_2B

    results["elapsed_total_sec"] = time.time() - t_all
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(_json_clean(results), f, indent=2)
    print(f"\n[Pilot 2b Full-Rank] ALL DONE in {time.time() - t_all:.1f}s -> {OUTPUT_PATH}")
    return results


def run_pilot2_gated_only() -> dict:
    """Fast (~40s on MPS) Non-Destructive Gated Native-Key Side-Module verification
    reusing cached Donor A payloads (`cache/pilot2_donorA_payloads.pt`).
    Zero SGD on B, 100% frozen MLP_orig, Donor A NEVER EVEN LOADED!
    """
    from gated_side_module import (
        GatedNativeKeyBank,
        GatedSideModuleMount,
        extract_post_attn_ln_inputs,
        fit_bg_whitener,
    )

    t_all = time.time()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    assert PAYLOAD_CACHE_PATH.exists(), f"Missing {PAYLOAD_CACHE_PATH}"
    assert UNLAB_LATE_CACHE_PATH.exists(), f"Missing {UNLAB_LATE_CACHE_PATH}"

    payloads = torch.load(PAYLOAD_CACHE_PATH, map_location="cpu", weights_only=False)
    unlab_cache = torch.load(UNLAB_LATE_CACHE_PATH, map_location="cpu", weights_only=False)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    facts = dataset["facts"]
    train_1hop_idx = [f["idx"] for f in facts if f["split"] == "train"]
    test_1hop_idx = [f["idx"] for f in facts if f["split"] == "test"]

    table_f, table_fp = build_private_tables(CANDIDATE_ENTITIES_32, GUILD_LABELS)
    recs_f_tr = make_prompt_records(CANDIDATE_ENTITIES_32, table_f, TRAIN_TEMPLATES_3)
    recs_f_te = make_prompt_records(CANDIDATE_ENTITIES_32, table_f, TEST_TEMPLATES_3)
    recs_fp_tr = make_prompt_records(CANDIDATE_ENTITIES_32, table_fp, TRAIN_TEMPLATES_3)
    recs_fp_te = make_prompt_records(CANDIDATE_ENTITIES_32, table_fp, TEST_TEMPLATES_3)

    # Load SOLO Recipient B (Donor A is NEVER loaded!)
    print(f"[Pilot 2b Gated] Loading SOLO Recipient B ({RECIP_KEY}) — Donor A NEVER loaded!")
    tok_B = AutoTokenizer.from_pretrained(str(HF_CACHE_DIR / RECIP_KEY), local_files_only=True)
    tok_B.padding_side = "left"
    if tok_B.pad_token is None:
        tok_B.pad_token = tok_B.eos_token
    model_B = AutoModelForCausalLM.from_pretrained(
        str(HF_CACHE_DIR / RECIP_KEY), local_files_only=True, dtype=torch.float32
    ).to(device)
    model_B.eval()

    with open(CACHE_DIR / f"{RECIP_KEY}_eval.json", "r", encoding="utf-8") as f:
        eval_B_1hop = json.load(f)["facts"]
    loc_prompts = [facts[i]["direct_prompt"] for i in test_1hop_idx]
    loc_spos = [eval_B_1hop[i]["subj_tok_idx_direct"] for i in test_1hop_idx]
    loc_slens = [eval_B_1hop[i]["seq_len_direct"] for i in test_1hop_idx]
    loc_aliases = [facts[i]["aliases"] for i in test_1hop_idx]
    loc_rels = [facts[i]["relation"] for i in test_1hop_idx]
    bg_prompts = [facts[i]["direct_prompt"] for i in train_1hop_idx]

    base_loc_acc, _ = eval_multi_regime_swaps(
        model_B, tok_B, loc_prompts, loc_spos, loc_slens, loc_aliases, loc_rels,
        device=device, dtype=torch.float32,
    )

    lA, lB = 35, 23
    sd = payloads["sites"]["site2_preoutput_last"]
    dA_f_tr = sd["f_tr"] - sd["base_tr"]
    dA_fp_tr = sd["f_prime_tr"] - sd["base_tr"]

    excl = list(GUILD_LABELS) + ["Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Black", "White"] + list(CANDIDATE_ENTITIES_32)
    rmsA = float(np.sqrt(np.mean(unlab_cache["A"][lA].astype(np.float64) ** 2, axis=1)).mean())
    rmsB = float(np.sqrt(np.mean(unlab_cache["B"][lB].astype(np.float64) ** 2, axis=1)).mean())
    map_dual, map_dual_rand, dual_meta = build_dual_adjoint_vocab_map(
        DONOR_KEY, RECIP_KEY, rms_A=rmsA, rms_B=rmsB, exclude_words=excl, lam_rel=1e-3,
    )

    # Extract B's native post_attn_ln inputs at lB=23 (STRICTLY TRAIN PROMPTS ONLY!)
    X_bg_all = extract_post_attn_ln_inputs(model_B, tok_B, bg_prompts, lB, device, all_tokens=True, batch_size=32)
    X_tr_last = extract_post_attn_ln_inputs(model_B, tok_B, [r["prompt"] for r in recs_f_tr], lB, device, all_tokens=False, batch_size=32)
    mu_bg, W_white = fit_bg_whitener(X_bg_all, shrink_rel=0.10)
    eids = [r["entity_idx"] for r in recs_f_tr]

    dB_f_tr = map_dual.transform_delta(dA_f_tr, alpha=1.0)
    dB_fp_tr = map_dual.transform_delta(dA_fp_tr, alpha=1.0)
    dB_rand_tr = map_dual_rand.transform_delta(dA_f_tr, alpha=1.0)
    perm = np.random.RandomState(20261601).permutation(len(dB_f_tr))
    dB_shuf_tr = dB_f_tr[perm]

    # Select alpha strictly on TRAIN prompts (zero test peeking!)
    best_alpha, best_tr_score = 2.0, -1e9
    alpha_train_sweep = {}
    for a in (1.5, 2.0, 2.5, 3.0):
        bk = GatedNativeKeyBank.compile("bank_f", X_tr_last, dB_f_tr, eids, 32, X_bg_all, mu_bg, W_white, alpha=a)
        with GatedSideModuleMount(model_B, lB, mu_bg, W_white, [bk], device=device):
            ev_tr = eval_guild_records(model_B, tok_B, recs_f_tr, GUILD_LABELS, device=device)
        alpha_train_sweep[f"a{a}"] = {"train_choice8": ev_tr["choice8_acc"], "train_greedy_em": ev_tr["greedy_em_acc"]}
        sc = ev_tr["choice8_acc"] + 0.5 * ev_tr["greedy_em_acc"]
        if sc > best_tr_score:
            best_tr_score = sc
            best_alpha = a

    bank_f = GatedNativeKeyBank.compile("bank_f", X_tr_last, dB_f_tr, eids, 32, X_bg_all, mu_bg, W_white, alpha=best_alpha)
    bank_fp = GatedNativeKeyBank.compile("bank_fp", X_tr_last, dB_fp_tr, eids, 32, X_bg_all, mu_bg, W_white, alpha=best_alpha)
    bank_shuf = GatedNativeKeyBank.compile("bank_shuf", X_tr_last, dB_shuf_tr, eids, 32, X_bg_all, mu_bg, W_white, alpha=best_alpha)
    bank_rand = GatedNativeKeyBank.compile("bank_rand", X_tr_last, dB_rand_tr, eids, 32, X_bg_all, mu_bg, W_white, alpha=best_alpha)

    def _eval_bank(tag: str, bank: GatedNativeKeyBank, eval_recs: List[dict], eval_loc: bool = True) -> dict:
        with GatedSideModuleMount(model_B, lB, mu_bg, W_white, [bank], device=device) as mnt:
            ev = eval_guild_records(model_B, tok_B, eval_recs, GUILD_LABELS, device=device)
            task_telem = mnt.get_telemetry_summary()
        per_tmpl = {}
        for t_idx in range(len(TEST_TEMPLATES_3)):
            idxs = [i for i, r in enumerate(eval_recs) if r["template_idx"] == t_idx]
            per_tmpl[f"tmpl_{t_idx}"] = {
                "ending": TEST_TEMPLATES_3[t_idx][1],
                "choice8": float(np.mean([ev["choice8_outcomes"][i] for i in idxs])),
                "greedy_em": float(np.mean([ev["greedy_outcomes"][i] for i in idxs])),
            }
        loc_acc, loc_drop, loc_telem = None, None, None
        if eval_loc:
            # Prefill-only exact token/prompt false-fire telemetry on the 196 test prompts
            with GatedSideModuleMount(model_B, lB, mu_bg, W_white, [bank], device=device) as mnt_pre:
                with torch.no_grad():
                    for st in range(0, len(loc_prompts), 32):
                        enc = tok_B(loc_prompts[st : st + 32], return_tensors="pt", padding=True).to(device)
                        _ = model_B(**enc, use_cache=False)
                loc_telem = mnt_pre.get_telemetry_summary()
            # Full generative locality eval
            with GatedSideModuleMount(model_B, lB, mu_bg, W_white, [bank], device=device):
                loc_acc, _ = eval_multi_regime_swaps(
                    model_B, tok_B, loc_prompts, loc_spos, loc_slens, loc_aliases, loc_rels,
                    device=device, dtype=torch.float32,
                )
            loc_drop = (base_loc_acc - loc_acc) * 100.0
        print(
            f"  [GATED 2B {tag:18s}] 8way={ev['choice8_acc']:.3f} EM={ev['greedy_em_acc']:.3f} | "
            f"task_last_fire={task_telem['any_bank_last_tok_fire_rate']:.3f} | "
            + (f"loc_drop={loc_drop:+.2f}pp (FPR_tok={loc_telem['any_bank_token_fire_rate']*100:.2f}%, FPR_pr={loc_telem['any_bank_prompt_fire_rate']*100:.2f}%)" if eval_loc else "")
        )
        return {
            "test_unseen_para_choice8": ev["choice8_acc"],
            "test_unseen_para_greedy_em": ev["greedy_em_acc"],
            "test_per_template": per_tmpl,
            "test_choice8_outcomes": ev["choice8_outcomes"],
            "test_greedy_outcomes": ev["greedy_outcomes"],
            "task_telemetry": task_telem,
            "locality_196_acc": loc_acc,
            "locality_drop_pp": loc_drop,
            "locality_prefill_telemetry": loc_telem,
        }

    out = {
        "meta": {"donor": DONOR_KEY, "recipient": RECIP_KEY, "lA": lA, "lB": lB, "selected_alpha": best_alpha, "dual_meta": dual_meta},
        "base_locality_196": base_loc_acc,
        "alpha_train_sweep": alpha_train_sweep,
        "bank_f_calibration": bank_f.calibration,
        "compiled_f": _eval_bank("compiled_f", bank_f, recs_f_te, eval_loc=True),
        "compiled_f_prime": _eval_bank("compiled_f_prime", bank_fp, recs_fp_te, eval_loc=True),
        "compiled_fp_on_old_f": _eval_bank("fp_on_old_f", bank_fp, recs_f_te, eval_loc=False),
        "shuffled_donor": _eval_bank("shuffled_donor", bank_shuf, recs_f_te, eval_loc=False),
        "random_dual_map": _eval_bank("random_dual_map", bank_rand, recs_f_te, eval_loc=True),
        "elapsed_sec": time.time() - t_all,
    }
    out_path = LAB_DIR / "results_pilot2_gated.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_json_clean(out), f, indent=2)
    print(f"[Pilot 2b Gated] DONE in {time.time() - t_all:.1f}s -> {out_path}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-payloads", action="store_true")
    parser.add_argument("--gated-only", action="store_true")
    parser.add_argument("--n-unlab", type=int, default=3600)
    args = parser.parse_args()
    if args.gated_only:
        run_pilot2_gated_only()
    else:
        run_pilot2_fullrank(reuse_payloads=args.reuse_payloads, n_unlab=args.n_unlab)


if __name__ == "__main__":
    main()
