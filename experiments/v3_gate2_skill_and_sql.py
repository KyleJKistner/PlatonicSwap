#!/usr/bin/env python3
"""
v3_gate2_skill_and_sql.py — 48h Gate 2: Counterfactual 2-Hop Routing Policy Swap + Text-to-SQLite Headroom
===========================================================================================================

Real Large -> Small Skill Implantation (`Qwen2.5-7B-Instruct` -> `SmolLM2-1.7B-Instruct`) built on
`platonicswap/large_to_small.py`:

  Part A (`--run-2hop`): Counterfactual 2-Hop Routing Policy Swap Falsifier
    - Shuffled 2-hop lookup tables (`192` train / `256` sealed test, strictly disjoint query/mid word pools).
    - Donor L (`Qwen2.5-7B-Instruct`, "pick FIRST") -> `val_L`; Donor R ("pick SECOND") -> `val_R` on
      the EXACT same table. Recipient `SmolLM2-1.7B-Instruct` receives the IDENTICAL instruction-free
      prompt (`"Table:\\n...\\nQuery: ...\\nOutput:"`).
    - Head-to-head 3-Tier comparison (Donor 7B 100% unloaded at inference, `torch.equal` rollback on ALL tiers!):
        * Tier 1:  `CLOSED_FORM_SEQ_DELTA_W` (ZERO backprop; sequential L12 -> L18 reversible ΔW with
                   contrast bridges + RMSNorm-aligned cocycle!)
        * Tier 2A: `SIDECAR_2BLOCK_DISTILL_WITH_HINT` (Removable 2-block recipient-RoPE attn + width-64 SwiGLU,
                   zero-init, hard bypass `torch.equal == True`, distilled ONLY from Donor 7B logits +
                   aligned mid-layer activation hint — ZERO ground-truth labels!)
        * Tier 2B: `SIDECAR_OUTPUT_ONLY_CTRL` (same 2-block sidecar, Donor 7B logits ONLY, no internal hint)
        * Tier 2C: `SIDECAR_SHUFFLED_SUPERVISION_CTRL` (shuffled table <-> donor-logit pairs; must collapse!)
        * Tier 3:  `TEACHER_LORA_BASELINE` (fair rank-16 LoRA distilled on exact same 192 Donor 7B logit pairs)
    - Verifies modular hot-swap (Organ L -> Organ R -> bypass/revert `torch.equal == True`) + 100-fact OOD locality.

  Part B (`--run-sql-headroom`): Real Text-to-SQLite Headroom Check
    - 80 self-contained, execution-verified Text-to-SQLite queries across 8 unfamiliar multi-table schemas
      (10 queries/schema) scored by strict live `sqlite3` execution match + SQL syntax validity.
    - Evaluates `Qwen2.5-7B-Instruct` vs. `SmolLM2-1.7B-Instruct` against the `>= 80%` vs `<= 40%` bar.

Strict v1.0 Read-Only Contract:
  - Does NOT touch any locked v1.0 files.
  - Pure-CPU `--unit-test` completes in <5s with ZERO GPU/model forward passes.

Usage:
  PYTHONPATH=platonicswap:experiments python experiments/v3_gate2_skill_and_sql.py --unit-test
  PYTHONPATH=platonicswap:experiments python -u experiments/v3_gate2_skill_and_sql.py --run-all
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
OUT_PATH = RESULTS_DIR / "results_v3_gate2_skill_and_sql.json"

from extract_and_baseline import DATASET_PATH, HF_CACHE_DIR, get_decoder_layers
from large_to_small import (
    LARGE_TO_SMALL_MODELS,
    ReversibleDeltaPack,
    apply_rmsnorm_np,
    fit_contrast_bridge,
    fit_rmsnorm_aligned_cocycle_map,
    load_model_bf16,
    solve_closed_form_layer_delta_w,
    unload_model_strict,
)
from v2_falsifierA_rope_torus import apply_rope_half_torch, rope_inv_freq

SEED = 20261127

# Clean 1-token word pool (filtered at runtime to STRICTLY 1-token with leading space across both tokenizers!)
WORD_CANDIDATES_KEYS: Tuple[str, ...] = (
    "oak", "pine", "moss", "kelp", "reed", "bark", "fern", "ash", "elm", "birch",
    "sage", "mint", "plum", "pear", "fig", "lime", "date", "corn", "oats", "rye",
    "flax", "hemp", "vine", "root", "stem", "leaf", "seed", "twig", "cone", "bulb",
    "clay", "silt", "loam", "peat", "sand", "rock", "jade", "ruby", "opal", "onyx",
    "zinc", "lead", "iron", "gold", "tin", "coal", "salt", "lime", "talc", "mica",
    "rain", "snow", "hail", "mist", "wind", "gale", "gust", "tide", "wave", "foam",
    "lake", "pond", "cove", "cape", "reef", "dune", "peak", "hill", "vale", "glen",
    "drum", "bell", "horn", "harp", "lute", "fife", "gong", "pipe", "flute", "reed",
    "bolt", "nail", "gear", "axle", "wire", "cord", "rope", "knot", "hook", "ring",
    "lamp", "wick", "oven", "kiln", "mill", "loom", "plow", "cart", "sled", "raft",
    "bowl", "cup", "jar", "jug", "pot", "pan", "tray", "dish", "fork", "spoon",
    "coat", "vest", "belt", "boot", "shoe", "sock", "hat", "cap", "hood", "cape",
    "book", "page", "card", "note", "list", "plan", "map", "sign", "seal", "flag",
    "coin", "mint", "bank", "shop", "mart", "dock", "pier", "port", "road", "lane",
)

WORD_CANDIDATES_VALS: Tuple[str, ...] = (
    "red", "blue", "green", "black", "white", "gray", "pink", "brown", "gold", "silver",
    "north", "south", "east", "west", "left", "right", "high", "low", "warm", "cold",
    "fast", "slow", "loud", "soft", "hard", "dark", "bright", "sweet", "sour", "sharp",
    "round", "flat", "wide", "deep", "tall", "short", "thick", "thin", "clean", "dry",
    "wet", "fresh", "plain", "rich", "pure", "bold", "calm", "wild", "mild", "clear",
    "dense", "light", "heavy", "solid", "smooth", "rough", "tight", "loose", "prime", "grand",
)


# ======================================================================================
# 1. Part A Dataset: Counterfactual Shuffled 2-Hop Routing Tables (192 Train / 256 Sealed Test)
# ======================================================================================

def filter_single_token_words(words: Sequence[str], tokenizers: Sequence[Any]) -> List[str]:
    """Keep unique words where `' ' + w` encodes to STRICTLY 1 token in every provided tokenizer."""
    out: List[str] = []
    seen = set()
    for w in words:
        if w in seen:
            continue
        seen.add(w)
        if all(len(t.encode(" " + w, add_special_tokens=False)) == 1 for t in tokenizers):
            out.append(w)
    return out


def format_2hop_prompts(
    hop1_rows: Sequence[Tuple[str, str]],
    hop2_rows: Sequence[Tuple[str, str, str]],
    query: str,
) -> Dict[str, str]:
    """
    Build IDENTICAL 0-shot instruction-free recipient prompt + 2-shot Donor L ("FIRST") &
    Donor R ("SECOND") prompts! Demo words (`cat, fox, bee, ant, sun, owl, elk, bat, yes, no,
    big, odd, fun, joy, shy, sad`) are strictly disjoint from `WORD_CANDIDATES_KEYS` & `VALS`.
    """
    lines = ["Table:"]
    for a, b in hop1_rows:
        lines.append(f"{a} -> {b}")
    for b, vL, vR in hop2_rows:
        lines.append(f"{b} -> ({vL}, {vR})")
    lines.append(f"Query: {query}")
    lines.append("Output:")
    recip_prompt = "\n".join(lines)

    demo_L = (
        "Rule: Follow the 2-hop Table (Query -> middle -> (first, second)) and output ONLY the FIRST word in the pair.\n\n"
        "Table:\n"
        "cat -> sun\nfox -> owl\nbee -> elk\nant -> bat\n"
        "elk -> (big, odd)\nsun -> (yes, no)\nbat -> (fun, joy)\nowl -> (shy, sad)\n"
        "Query: cat\nOutput: yes\n\n"
        "Table:\n"
        "ant -> bat\nbee -> elk\nfox -> owl\ncat -> sun\n"
        "owl -> (shy, sad)\nbat -> (fun, joy)\nsun -> (yes, no)\nelk -> (big, odd)\n"
        "Query: fox\nOutput: shy\n\n"
    )
    demo_R = (
        "Rule: Follow the 2-hop Table (Query -> middle -> (first, second)) and output ONLY the SECOND word in the pair.\n\n"
        "Table:\n"
        "cat -> sun\nfox -> owl\nbee -> elk\nant -> bat\n"
        "elk -> (big, odd)\nsun -> (yes, no)\nbat -> (fun, joy)\nowl -> (shy, sad)\n"
        "Query: cat\nOutput: no\n\n"
        "Table:\n"
        "ant -> bat\nbee -> elk\nfox -> owl\ncat -> sun\n"
        "owl -> (shy, sad)\nbat -> (fun, joy)\nsun -> (yes, no)\nelk -> (big, odd)\n"
        "Query: fox\nOutput: sad\n\n"
    )
    donor_L_prompt = demo_L + recip_prompt
    donor_R_prompt = demo_R + recip_prompt
    return {
        "recipient_prompt": recip_prompt,
        "donor_L_prompt": donor_L_prompt,
        "donor_R_prompt": donor_R_prompt,
    }


def build_2hop_routing_dataset(
    tokenizers: Sequence[Any],
    n_train: int = 192,
    n_test: int = 256,
    seed: int = SEED,
) -> Dict[str, Any]:
    """
    Generate `n_train` train tables + `n_test` strictly sealed test tables with:
      - Strictly DISJOINT key-word pools between train and test (0 shared query or mid-hop words!)
      - Strictly 1-token leaf answers (`' ' + val_L`, `' ' + val_R`) across all tokenizers
      - Shuffled 4-row Hop 1 (`a -> b`) and shuffled 4-row Hop 2 (`b -> (vL, vR)`) with `vL != vR`
      - Zero table-signature overlap between train and test!
    """
    rng = np.random.default_rng(seed)
    all_keys = filter_single_token_words(WORD_CANDIDATES_KEYS, tokenizers)
    all_vals = filter_single_token_words(WORD_CANDIDATES_VALS, tokenizers)
    # Remove any overlap between key words and leaf value words so routing roles never collide
    all_keys = [w for w in all_keys if w not in set(all_vals)]
    assert len(all_keys) >= 40, f"Need >=40 1-token key words, got {len(all_keys)}"
    assert len(all_vals) >= 24, f"Need >=24 1-token val words, got {len(all_vals)}"

    # Strictly split key pool into disjoint train vs test halves (0 shared query/mid words!)
    perm_k = rng.permutation(len(all_keys))
    mid_k = len(all_keys) // 2
    train_keys = [all_keys[i] for i in perm_k[:mid_k]]
    test_keys = [all_keys[i] for i in perm_k[mid_k:]]
    assert set(train_keys).isdisjoint(set(test_keys))

    def _gen_split(key_pool: List[str], n_items: int, split_name: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        seen_sigs = set()
        while len(items) < n_items:
            chosen_k = rng.choice(key_pool, size=8, replace=False).tolist()
            hop1_src = chosen_k[:4]
            hop1_mid = chosen_k[4:8]
            chosen_v = rng.choice(all_vals, size=8, replace=False).tolist()
            hop1_rows = [(hop1_src[i], hop1_mid[i]) for i in range(4)]
            hop2_rows = [(hop1_mid[i], chosen_v[2 * i], chosen_v[2 * i + 1]) for i in range(4)]
            # Independently shuffle Hop 1 and Hop 2 rows!
            idx1 = rng.permutation(4).tolist()
            idx2 = rng.permutation(4).tolist()
            hop1_shuf = [hop1_rows[i] for i in idx1]
            hop2_shuf = [hop2_rows[i] for i in idx2]
            q_i = int(rng.integers(0, 4))
            query, mid_word = hop1_src[q_i], hop1_mid[q_i]
            val_L, val_R = chosen_v[2 * q_i], chosen_v[2 * q_i + 1]
            sig = (tuple(hop1_shuf), tuple(hop2_shuf), query)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            pr = format_2hop_prompts(hop1_shuf, hop2_shuf, query)
            items.append({
                "id": f"{split_name}_{len(items):03d}",
                "split": split_name,
                "query": query,
                "mid": mid_word,
                "val_L": val_L,
                "val_R": val_R,
                "hop1_rows": hop1_shuf,
                "hop2_rows": hop2_shuf,
                **pr,
            })
        return items

    train_items = _gen_split(train_keys, n_train, "train")
    test_items = _gen_split(test_keys, n_test, "test")
    return {
        "train": train_items,
        "test": test_items,
        "train_key_vocab": train_keys,
        "test_key_vocab": test_keys,
        "val_vocab": all_vals,
    }


# ======================================================================================
# 2. Removable 2-Block Sidecar Organ (Seat 2/3) & Reversible Rank-16 LoRA Baseline (Tier 3)
# ======================================================================================

class SidecarBlock(nn.Module):
    """
    Single removable sidecar block (Seat 2/3 Architecture):
      - RMSNorm on incoming residual `x` (`D_B`)
      - 1 recipient-RoPE causal attention head (`d_h`, using recipient's exact `rope_theta`)
      - Width-`mlp_width` (64) SwiGLU MLP
      - BOTH `o_proj` and `down_proj` are exact ZERO-INITIALIZED so at init `block(x) == 0` identically!
    """

    def __init__(self, d_model: int, head_dim: int = 64, mlp_width: int = 64, rope_theta: float = 130_000.0, eps: float = 1e-5):
        super().__init__()
        self.d_model = int(d_model)
        self.head_dim = int(head_dim)
        self.mlp_width = int(mlp_width)
        self.eps = float(eps)

        self.norm_w = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim, bias=False)
        self.o_proj = nn.Linear(head_dim, d_model, bias=False)

        self.gate_proj = nn.Linear(d_model, mlp_width, bias=False)
        self.up_proj = nn.Linear(d_model, mlp_width, bias=False)
        self.down_proj = nn.Linear(mlp_width, d_model, bias=False)

        # Exact zero-init output projections -> bit-exact identity at step 0!
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.down_proj.weight)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)

        inv_f = rope_inv_freq(head_dim, rope_theta).astype(np.float32)
        self.register_buffer("inv_freq", torch.from_numpy(inv_f), persistent=False)

    def _rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.norm_w

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        xf = x.float()
        B, T, _ = xf.shape
        h = self._rmsnorm(xf)

        # 1 recipient-RoPE causal attention head
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        pos = torch.arange(T, device=xf.device, dtype=torch.float32)
        if attention_mask is not None and attention_mask.shape[0] == B:
            if attention_mask.shape[1] < T:
                extra = torch.ones((B, T - attention_mask.shape[1]), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, extra], dim=1)
            elif attention_mask.shape[1] > T:
                attention_mask = attention_mask[:, :T]
            # Handle left-padded sequences cleanly: real positions start at first non-pad token!
            pos = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0).float()
        else:
            attention_mask = None
        q = apply_rope_half_torch(q, self.inv_freq.to(xf.device), pos)
        k = apply_rope_half_torch(k, self.inv_freq.to(xf.device), pos)

        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(self.head_dim))
        causal = torch.triu(torch.ones((T, T), device=xf.device, dtype=torch.bool), diagonal=1)
        logits = logits.masked_fill(causal.unsqueeze(0), -1e9)
        if attention_mask is not None:
            pad_k = (attention_mask == 0).unsqueeze(1)  # (B, 1, T)
            logits = logits.masked_fill(pad_k, -1e9)
        attn = F.softmax(logits, dim=-1)
        y_attn = self.o_proj(torch.matmul(attn, v))

        # Width-64 SwiGLU MLP on post-sidecar-attn residual
        h2 = self._rmsnorm(xf + y_attn)
        y_mlp = self.down_proj(F.silu(self.gate_proj(h2)) * self.up_proj(h2))
        delta = y_attn + y_mlp
        if attention_mask is not None:
            delta = delta * attention_mask.unsqueeze(-1).float()
        return delta.to(x.dtype)


class Removable2BlockSidecarOrgan(nn.Module):
    """
    Removable 2-Block Sidecar Organ (Seat 2/3 Architecture):
      - Attaches two zero-init `SidecarBlock`s via forward hooks at `(layer1, layer2)` of a frozen
        recipient model (base weights 100% frozen!).
      - Hard bypass (`self.enabled = False`) OR `.detach()` restores bit-identical (`torch.equal == True`)
        model outputs on matched-batch fresh-cache logits!
      - Records per-block residual deltas in `self.last_deltas` for mid-layer Platonic hint distillation!
    """

    def __init__(
        self,
        d_model: int,
        layer_indices: Tuple[int, int] = (12, 18),
        head_dim: int = 64,
        mlp_width: int = 64,
        rope_theta: float = 130_000.0,
    ):
        super().__init__()
        assert len(layer_indices) == 2
        self.layer_indices = (int(layer_indices[0]), int(layer_indices[1]))
        self.blocks = nn.ModuleList([
            SidecarBlock(d_model, head_dim=head_dim, mlp_width=mlp_width, rope_theta=rope_theta),
            SidecarBlock(d_model, head_dim=head_dim, mlp_width=mlp_width, rope_theta=rope_theta),
        ])
        self.enabled: bool = True
        self._handles: List[Any] = []
        self._cur_attn_mask: Optional[torch.Tensor] = None
        self.last_deltas: Dict[int, torch.Tensor] = {}

    def set_attention_mask(self, mask: Optional[torch.Tensor]) -> None:
        self._cur_attn_mask = mask

    def attach(self, model: nn.Module) -> "Removable2BlockSidecarOrgan":
        self.detach()
        layers = get_decoder_layers(model)
        for bi, li in enumerate(self.layer_indices):
            def _mk_hook(block_idx: int, layer_idx: int):
                def _hook(_m, _inp, out):
                    if not self.enabled:
                        return out
                    is_tup = isinstance(out, tuple)
                    h = out[0] if is_tup else out
                    delta = self.blocks[block_idx](h, attention_mask=self._cur_attn_mask)
                    self.last_deltas[layer_idx] = delta
                    h_new = h + delta
                    return (h_new,) + out[1:] if is_tup else h_new
                return _hook
            self._handles.append(layers[li].register_forward_hook(_mk_hook(bi, li)))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.last_deltas.clear()
        self._cur_attn_mask = None

    def n_params(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def n_bytes_bf16(self) -> int:
        return self.n_params() * 2


class ReversibleLoRALinearHook:
    """
    Lightweight, self-contained Rank-r LoRA adapter attached via forward hooks (ZERO external peft
    dependency, base weights 100% frozen, bit-exact `torch.equal` rollback on `.detach()`!).
    """

    def __init__(self, targets: Dict[str, nn.Linear], rank: int = 16, alpha: float = 32.0, device: torch.device = torch.device("cpu")):
        self.targets = targets
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.params = nn.ModuleDict()
        for name, lin in targets.items():
            safe = name.replace(".", "__")
            A = nn.Linear(lin.in_features, rank, bias=False, device=device, dtype=torch.float32)
            B = nn.Linear(rank, lin.out_features, bias=False, device=device, dtype=torch.float32)
            nn.init.kaiming_uniform_(A.weight, a=math.sqrt(5))
            nn.init.zeros_(B.weight)
            self.params[safe] = nn.Sequential(A, B)
        self.enabled: bool = True
        self._handles: List[Any] = []

    def attach(self) -> "ReversibleLoRALinearHook":
        self.detach()
        for name, lin in self.targets.items():
            safe = name.replace(".", "__")
            mod = self.params[safe]
            def _mk(m):
                def _hook(_mod, inp, out):
                    if not self.enabled:
                        return out
                    delta = self.scale * m(inp[0].float())
                    return out + delta.to(out.dtype)
                return _hook
            self._handles.append(lin.register_forward_hook(_mk(mod)))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def parameters(self):
        return self.params.parameters()

    def n_params(self) -> int:
        return sum(int(p.numel()) for p in self.params.parameters())

    def n_bytes_bf16(self) -> int:
        return self.n_params() * 2


# ======================================================================================
# 3. Part B Benchmark: 80 Self-Contained Execution-Verified Text-to-SQLite Queries (8 Schemas × 10)
# ======================================================================================

def build_text_to_sqlite_benchmark() -> List[Dict[str, Any]]:
    """
    80 self-contained, execution-verified Text-to-SQLite queries across 8 unfamiliar multi-table
    schemas (10 queries/schema: logistics, hospital, library, solar grid, retail, university,
    flights, sensor telemetry). Every gold SQL returns a non-empty deterministic result in sqlite3!
    """
    schemas: List[Dict[str, Any]] = [
        # 1. Logistics Shipments
        {
            "schema_id": "logistics_shipments",
            "ddl": """
CREATE TABLE warehouses (wh_id INTEGER PRIMARY KEY, hub_name TEXT, region TEXT, capacity_tons INTEGER);
CREATE TABLE carriers (carrier_id INTEGER PRIMARY KEY, carrier_name TEXT, on_time_pct REAL);
CREATE TABLE shipments (ship_id INTEGER PRIMARY KEY, wh_id INTEGER, carrier_id INTEGER, weight_tons REAL, cost_usd REAL, status TEXT);
""",
            "seed": """
INSERT INTO warehouses VALUES (1,'NorthHub','North',500),(2,'SouthDock','South',300),(3,'EastGate','East',450),(4,'WestPort','West',600);
INSERT INTO carriers VALUES (10,'SwiftFreight',94.5),(20,'AeroCargo',98.0),(30,'RailLink',88.0);
INSERT INTO shipments VALUES
  (101,1,10,12.5,1400,'delivered'),(102,1,20,8.0,2200,'delivered'),(103,2,10,15.0,1650,'delayed'),
  (104,2,30,22.0,1100,'delivered'),(105,3,20,6.5,1900,'in_transit'),(106,3,30,18.0,950,'delivered'),
  (107,4,10,25.0,2800,'delivered'),(108,4,20,10.0,2600,'delayed'),(109,4,30,30.0,1500,'delivered'),
  (110,1,30,14.0,800,'in_transit');
""",
            "queries": [
                ("List the hub names of warehouses with capacity greater than 400 tons, ordered alphabetically.",
                 "SELECT hub_name FROM warehouses WHERE capacity_tons > 400 ORDER BY hub_name;"),
                ("How many total shipments have status 'delivered'?",
                 "SELECT COUNT(*) FROM shipments WHERE status = 'delivered';"),
                ("What is the total cost_usd of all shipments handled by carrier 'AeroCargo'?",
                 "SELECT SUM(s.cost_usd) FROM shipments s JOIN carriers c ON s.carrier_id = c.carrier_id WHERE c.carrier_name = 'AeroCargo';"),
                ("Which warehouse hub_name has the highest total shipment weight_tons?",
                 "SELECT w.hub_name FROM warehouses w JOIN shipments s ON w.wh_id = s.wh_id GROUP BY w.wh_id, w.hub_name ORDER BY SUM(s.weight_tons) DESC LIMIT 1;"),
                ("Return each status and the number of shipments with that status, ordered by status alphabetically.",
                 "SELECT status, COUNT(*) FROM shipments GROUP BY status ORDER BY status;"),
                ("List carrier names that have at least one 'delayed' shipment, ordered alphabetically.",
                 "SELECT DISTINCT c.carrier_name FROM carriers c JOIN shipments s ON c.carrier_id = s.carrier_id WHERE s.status = 'delayed' ORDER BY c.carrier_name;"),
                ("What is the average cost_usd of 'delivered' shipments from the 'North' region?",
                 "SELECT AVG(s.cost_usd) FROM shipments s JOIN warehouses w ON s.wh_id = w.wh_id WHERE s.status = 'delivered' AND w.region = 'North';"),
                ("List each carrier_name and its total delivered weight_tons, ordered by total delivered weight descending.",
                 "SELECT c.carrier_name, SUM(s.weight_tons) AS tot_w FROM carriers c JOIN shipments s ON c.carrier_id = s.carrier_id WHERE s.status = 'delivered' GROUP BY c.carrier_id, c.carrier_name ORDER BY tot_w DESC;"),
                ("Find the ship_id of the most expensive shipment with weight_tons under 15.",
                 "SELECT ship_id FROM shipments WHERE weight_tons < 15 ORDER BY cost_usd DESC LIMIT 1;"),
                ("Which regions have total shipment cost_usd exceeding 3500? Return region ordered alphabetically.",
                 "SELECT w.region FROM warehouses w JOIN shipments s ON w.wh_id = s.wh_id GROUP BY w.region HAVING SUM(s.cost_usd) > 3500 ORDER BY w.region;"),
            ],
        },
        # 2. Hospital Appointments
        {
            "schema_id": "hospital_appointments",
            "ddl": """
CREATE TABLE departments (dept_id INTEGER PRIMARY KEY, dept_name TEXT, floor_num INTEGER);
CREATE TABLE doctors (doc_id INTEGER PRIMARY KEY, doc_name TEXT, dept_id INTEGER, fee_usd INTEGER);
CREATE TABLE patients (pat_id INTEGER PRIMARY KEY, pat_name TEXT, age INTEGER);
CREATE TABLE appointments (appt_id INTEGER PRIMARY KEY, doc_id INTEGER, pat_id INTEGER, visit_month TEXT, attended INTEGER);
""",
            "seed": """
INSERT INTO departments VALUES (1,'Cardiology',3),(2,'Neurology',4),(3,'Orthopedics',2);
INSERT INTO doctors VALUES (11,'Dr. Lin',1,250),(12,'Dr. Patel',1,220),(13,'Dr. Gomez',2,280),(14,'Dr. Novak',3,200);
INSERT INTO patients VALUES (101,'Ava',62),(102,'Ben',34),(103,'Cara',49),(104,'Dan',71),(105,'Eli',28);
INSERT INTO appointments VALUES
  (1,11,101,'2025-01',1),(2,11,104,'2025-01',1),(3,12,102,'2025-01',0),
  (4,13,103,'2025-02',1),(5,13,101,'2025-02',1),(6,14,105,'2025-02',1),
  (7,14,102,'2025-02',1),(8,11,103,'2025-03',0),(9,12,104,'2025-03',1),(10,14,101,'2025-03',1);
""",
            "queries": [
                ("List all doctor names in the 'Cardiology' department, ordered alphabetically.",
                 "SELECT d.doc_name FROM doctors d JOIN departments dp ON d.dept_id = dp.dept_id WHERE dp.dept_name = 'Cardiology' ORDER BY d.doc_name;"),
                ("How many appointments were actually attended (attended = 1)?",
                 "SELECT COUNT(*) FROM appointments WHERE attended = 1;"),
                ("What is the total fee_usd earned from attended appointments across all doctors?",
                 "SELECT SUM(d.fee_usd) FROM appointments a JOIN doctors d ON a.doc_id = d.doc_id WHERE a.attended = 1;"),
                ("Which patient name has the highest number of attended appointments?",
                 "SELECT p.pat_name FROM patients p JOIN appointments a ON p.pat_id = a.pat_id WHERE a.attended = 1 GROUP BY p.pat_id, p.pat_name ORDER BY COUNT(*) DESC LIMIT 1;"),
                ("Return each dept_name and the number of attended appointments in that department, ordered by dept_name.",
                 "SELECT dp.dept_name, COUNT(*) FROM appointments a JOIN doctors d ON a.doc_id = d.doc_id JOIN departments dp ON d.dept_id = dp.dept_id WHERE a.attended = 1 GROUP BY dp.dept_id, dp.dept_name ORDER BY dp.dept_name;"),
                ("What is the average age of patients who had an appointment with 'Dr. Lin'?",
                 "SELECT AVG(p.age) FROM patients p JOIN appointments a ON p.pat_id = a.pat_id JOIN doctors d ON a.doc_id = d.doc_id WHERE d.doc_name = 'Dr. Lin';"),
                ("List the visit_month and total appointments per month, ordered by visit_month ascending.",
                 "SELECT visit_month, COUNT(*) FROM appointments GROUP BY visit_month ORDER BY visit_month;"),
                ("Which doctor name had the highest total fee revenue from attended appointments?",
                 "SELECT d.doc_name FROM doctors d JOIN appointments a ON d.doc_id = a.doc_id WHERE a.attended = 1 GROUP BY d.doc_id, d.doc_name ORDER BY SUM(d.fee_usd) DESC LIMIT 1;"),
                ("List names of patients older than 50 who missed (attended = 0) zero appointments, ordered alphabetically.",
                 "SELECT pat_name FROM patients WHERE age > 50 AND pat_id NOT IN (SELECT pat_id FROM appointments WHERE attended = 0) ORDER BY pat_name;"),
                ("Which floor_num has doctors with an average fee_usd greater than 230?",
                 "SELECT dp.floor_num FROM departments dp JOIN doctors d ON dp.dept_id = d.dept_id GROUP BY dp.dept_id, dp.floor_num HAVING AVG(d.fee_usd) > 230 ORDER BY dp.floor_num;"),
            ],
        },
        # 3. Library Loans
        {
            "schema_id": "library_loans",
            "ddl": """
CREATE TABLE members (mem_id INTEGER PRIMARY KEY, mem_name TEXT, tier TEXT);
CREATE TABLE books (book_id INTEGER PRIMARY KEY, title TEXT, genre TEXT, pages INTEGER);
CREATE TABLE loans (loan_id INTEGER PRIMARY KEY, mem_id INTEGER, book_id INTEGER, days_held INTEGER, returned INTEGER);
""",
            "seed": """
INSERT INTO members VALUES (1,'Maya','gold'),(2,'Noah','silver'),(3,'Iris','gold'),(4,'Leo','bronze');
INSERT INTO books VALUES
  (10,'Orbital Mechanics','Science',420),(11,'Ancient Bridges','History',310),
  (12,'Silent Harbor','Fiction',260),(13,'Quantum Optics','Science',510),(14,'Silk Routes','History',380);
INSERT INTO loans VALUES
  (1,1,10,14,1),(2,1,11,9,1),(3,2,12,21,0),(4,2,10,12,1),
  (5,3,13,30,1),(6,3,14,18,0),(7,4,12,7,1),(8,1,13,16,1),(9,3,11,11,1);
""",
            "queries": [
                ("List book titles with more than 350 pages, ordered by pages descending.",
                 "SELECT title FROM books WHERE pages > 350 ORDER BY pages DESC;"),
                ("How many loans have not yet been returned (returned = 0)?",
                 "SELECT COUNT(*) FROM loans WHERE returned = 0;"),
                ("What is the average days_held for returned loans of 'Science' books?",
                 "SELECT AVG(l.days_held) FROM loans l JOIN books b ON l.book_id = b.book_id WHERE l.returned = 1 AND b.genre = 'Science';"),
                ("Which member name borrowed the highest total number of books?",
                 "SELECT m.mem_name FROM members m JOIN loans l ON m.mem_id = l.mem_id GROUP BY m.mem_id, m.mem_name ORDER BY COUNT(*) DESC, m.mem_name ASC LIMIT 1;"),
                ("Return each genre and the total number of loans for that genre, ordered by genre alphabetically.",
                 "SELECT b.genre, COUNT(*) FROM loans l JOIN books b ON l.book_id = b.book_id GROUP BY b.genre ORDER BY b.genre;"),
                ("List the names of 'gold' tier members who currently have an unreturned book, ordered alphabetically.",
                 "SELECT DISTINCT m.mem_name FROM members m JOIN loans l ON m.mem_id = l.mem_id WHERE m.tier = 'gold' AND l.returned = 0 ORDER BY m.mem_name;"),
                ("What is the total number of pages across all books ever borrowed by 'Maya'?",
                 "SELECT SUM(b.pages) FROM loans l JOIN members m ON l.mem_id = m.mem_id JOIN books b ON l.book_id = b.book_id WHERE m.mem_name = 'Maya';"),
                ("Which book title has the highest average days_held across its loans?",
                 "SELECT b.title FROM books b JOIN loans l ON b.book_id = l.book_id GROUP BY b.book_id, b.title ORDER BY AVG(l.days_held) DESC LIMIT 1;"),
                ("Return each member tier and total days_held across all loans by that tier, ordered by total days descending.",
                 "SELECT m.tier, SUM(l.days_held) AS tot_days FROM members m JOIN loans l ON m.mem_id = l.mem_id GROUP BY m.tier ORDER BY tot_days DESC;"),
                ("List titles of books that were borrowed at least twice, ordered alphabetically.",
                 "SELECT b.title FROM books b JOIN loans l ON b.book_id = l.book_id GROUP BY b.book_id, b.title HAVING COUNT(*) >= 2 ORDER BY b.title;"),
            ],
        },
        # 4. Solar Power Grid
        {
            "schema_id": "solar_power_grid",
            "ddl": """
CREATE TABLE substations (sub_id INTEGER PRIMARY KEY, sub_name TEXT, zone TEXT, max_mw REAL);
CREATE TABLE panels (panel_id INTEGER PRIMARY KEY, sub_id INTEGER, model TEXT, install_year INTEGER);
CREATE TABLE readings (read_id INTEGER PRIMARY KEY, panel_id INTEGER, day_label TEXT, output_kwh REAL, fault_flag INTEGER);
""",
            "seed": """
INSERT INTO substations VALUES (1,'Helios-A','Desert',40.0),(2,'SunCrest','Ridge',25.0),(3,'MesaVolt','Valley',35.0);
INSERT INTO panels VALUES (11,1,'PV-X1',2021),(12,1,'PV-X2',2023),(13,2,'PV-X1',2020),(14,3,'PV-X2',2022),(15,3,'PV-X3',2024);
INSERT INTO readings VALUES
  (1,11,'Mon',120.0,0),(2,11,'Tue',115.0,0),(3,12,'Mon',140.0,0),(4,12,'Tue',90.0,1),
  (5,13,'Mon',85.0,1),(6,13,'Tue',95.0,0),(7,14,'Mon',130.0,0),(8,14,'Tue',135.0,0),
  (9,15,'Mon',150.0,0),(10,15,'Tue',145.0,0);
""",
            "queries": [
                ("List substation names with max_mw >= 35.0, ordered alphabetically.",
                 "SELECT sub_name FROM substations WHERE max_mw >= 35.0 ORDER BY sub_name;"),
                ("What is the total output_kwh recorded across all readings with fault_flag = 0?",
                 "SELECT SUM(output_kwh) FROM readings WHERE fault_flag = 0;"),
                ("Which substation name produced the highest total output_kwh?",
                 "SELECT s.sub_name FROM substations s JOIN panels p ON s.sub_id = p.sub_id JOIN readings r ON p.panel_id = r.panel_id GROUP BY s.sub_id, s.sub_name ORDER BY SUM(r.output_kwh) DESC LIMIT 1;"),
                ("Return each panel model and its average output_kwh, ordered by model alphabetically.",
                 "SELECT p.model, AVG(r.output_kwh) FROM panels p JOIN readings r ON p.panel_id = r.panel_id GROUP BY p.model ORDER BY p.model;"),
                ("How many distinct panels experienced at least one fault (fault_flag = 1)?",
                 "SELECT COUNT(DISTINCT panel_id) FROM readings WHERE fault_flag = 1;"),
                ("What is the total output_kwh on day_label 'Mon' for panels installed after 2021?",
                 "SELECT SUM(r.output_kwh) FROM readings r JOIN panels p ON r.panel_id = p.panel_id WHERE r.day_label = 'Mon' AND p.install_year > 2021;"),
                ("List each zone and the number of fault readings in that zone, ordered by zone alphabetically.",
                 "SELECT s.zone, SUM(r.fault_flag) FROM substations s JOIN panels p ON s.sub_id = p.sub_id JOIN readings r ON p.panel_id = r.panel_id GROUP BY s.zone ORDER BY s.zone;"),
                ("Find the panel_id with the lowest single-reading output_kwh.",
                 "SELECT panel_id FROM readings ORDER BY output_kwh ASC LIMIT 1;"),
                ("Return each day_label and total output_kwh across all panels, ordered by day_label alphabetically.",
                 "SELECT day_label, SUM(output_kwh) FROM readings GROUP BY day_label ORDER BY day_label;"),
                ("List models whose panels have average output_kwh strictly above 125.0, ordered alphabetically.",
                 "SELECT p.model FROM panels p JOIN readings r ON p.panel_id = r.panel_id GROUP BY p.model HAVING AVG(r.output_kwh) > 125.0 ORDER BY p.model;"),
            ],
        },
        # 5. Retail Inventory
        {
            "schema_id": "retail_inventory",
            "ddl": """
CREATE TABLE suppliers (sup_id INTEGER PRIMARY KEY, sup_name TEXT, country TEXT);
CREATE TABLE products (prod_id INTEGER PRIMARY KEY, prod_name TEXT, sup_id INTEGER, unit_price REAL, stock_qty INTEGER);
CREATE TABLE sales (sale_id INTEGER PRIMARY KEY, prod_id INTEGER, qty_sold INTEGER, quarter TEXT);
""",
            "seed": """
INSERT INTO suppliers VALUES (1,'NordCraft','Norway'),(2,'AlpsGear','Austria'),(3,'SolWorks','Chile');
INSERT INTO products VALUES
  (10,'ThermalFlask',1,30.0,120),(11,'CampLantern',1,45.0,60),
  (12,'TrailPack',2,80.0,40),(13,'SolarCharger',3,55.0,90),(14,'DrySack',2,25.0,150);
INSERT INTO sales VALUES
  (1,10,20,'Q1'),(2,10,15,'Q2'),(3,11,10,'Q1'),(4,12,8,'Q1'),
  (5,12,12,'Q2'),(6,13,14,'Q2'),(7,14,30,'Q1'),(8,14,25,'Q2'),(9,13,6,'Q1');
""",
            "queries": [
                ("List product names with stock_qty below 100, ordered by unit_price descending.",
                 "SELECT prod_name FROM products WHERE stock_qty < 100 ORDER BY unit_price DESC;"),
                ("What is the total revenue (qty_sold * unit_price) across all sales in 'Q1'?",
                 "SELECT SUM(s.qty_sold * p.unit_price) FROM sales s JOIN products p ON s.prod_id = p.prod_id WHERE s.quarter = 'Q1';"),
                ("Which product name generated the highest total revenue across all quarters?",
                 "SELECT p.prod_name FROM products p JOIN sales s ON p.prod_id = s.prod_id GROUP BY p.prod_id, p.prod_name ORDER BY SUM(s.qty_sold * p.unit_price) DESC LIMIT 1;"),
                ("Return each supplier name and total units sold across its products, ordered by total units descending.",
                 "SELECT sp.sup_name, SUM(s.qty_sold) AS tot_qty FROM suppliers sp JOIN products p ON sp.sup_id = p.sup_id JOIN sales s ON p.prod_id = s.prod_id GROUP BY sp.sup_id, sp.sup_name ORDER BY tot_qty DESC;"),
                ("What is the average unit_price of products supplied from 'Austria'?",
                 "SELECT AVG(p.unit_price) FROM products p JOIN suppliers sp ON p.sup_id = sp.sup_id WHERE sp.country = 'Austria';"),
                ("Return each quarter and total revenue in that quarter, ordered by quarter alphabetically.",
                 "SELECT s.quarter, SUM(s.qty_sold * p.unit_price) FROM sales s JOIN products p ON s.prod_id = p.prod_id GROUP BY s.quarter ORDER BY s.quarter;"),
                ("Which supplier country has the highest total stock_qty in inventory?",
                 "SELECT sp.country FROM suppliers sp JOIN products p ON sp.sup_id = p.sup_id GROUP BY sp.country ORDER BY SUM(p.stock_qty) DESC LIMIT 1;"),
                ("List product names that sold at least 30 total units across all quarters, ordered alphabetically.",
                 "SELECT p.prod_name FROM products p JOIN sales s ON p.prod_id = s.prod_id GROUP BY p.prod_id, p.prod_name HAVING SUM(s.qty_sold) >= 30 ORDER BY p.prod_name;"),
                ("How many total units were sold by supplier 'NordCraft' in 'Q1'?",
                 "SELECT SUM(s.qty_sold) FROM sales s JOIN products p ON s.prod_id = p.prod_id JOIN suppliers sp ON p.sup_id = sp.sup_id WHERE sp.sup_name = 'NordCraft' AND s.quarter = 'Q1';"),
                ("Find the prod_name with the highest total inventory value (unit_price * stock_qty).",
                 "SELECT prod_name FROM products ORDER BY (unit_price * stock_qty) DESC LIMIT 1;"),
            ],
        },
        # 6. University Courses
        {
            "schema_id": "university_courses",
            "ddl": """
CREATE TABLE instructors (inst_id INTEGER PRIMARY KEY, inst_name TEXT, faculty TEXT);
CREATE TABLE courses (course_id INTEGER PRIMARY KEY, course_code TEXT, inst_id INTEGER, credits INTEGER);
CREATE TABLE enrollments (enr_id INTEGER PRIMARY KEY, course_id INTEGER, student_name TEXT, score INTEGER);
""",
            "seed": """
INSERT INTO instructors VALUES (1,'Prof. Chen','Engineering'),(2,'Prof. Okafor','Math'),(3,'Prof. Lind','Engineering');
INSERT INTO courses VALUES (101,'ENG201',1,4),(102,'ENG305',3,3),(103,'MTH210',2,4),(104,'MTH401',2,3);
INSERT INTO enrollments VALUES
  (1,101,'Sam',88),(2,101,'Tia',94),(3,101,'Uma',76),
  (4,102,'Sam',82),(5,102,'Vic',91),
  (6,103,'Tia',97),(7,103,'Uma',85),(8,103,'Vic',79),
  (9,104,'Sam',90),(10,104,'Tia',92);
""",
            "queries": [
                ("List all course_code values worth 4 credits, ordered alphabetically.",
                 "SELECT course_code FROM courses WHERE credits = 4 ORDER BY course_code;"),
                ("What is the overall average score in course 'ENG201'?",
                 "SELECT AVG(e.score) FROM enrollments e JOIN courses c ON e.course_id = c.course_id WHERE c.course_code = 'ENG201';"),
                ("Which student_name has the highest average score across all enrolled courses?",
                 "SELECT student_name FROM enrollments GROUP BY student_name ORDER BY AVG(score) DESC LIMIT 1;"),
                ("Return each faculty and total number of enrollments in that faculty, ordered by faculty alphabetically.",
                 "SELECT i.faculty, COUNT(*) FROM enrollments e JOIN courses c ON e.course_id = c.course_id JOIN instructors i ON c.inst_id = i.inst_id GROUP BY i.faculty ORDER BY i.faculty;"),
                ("How many total credits is student 'Sam' enrolled in?",
                 "SELECT SUM(c.credits) FROM enrollments e JOIN courses c ON e.course_id = c.course_id WHERE e.student_name = 'Sam';"),
                ("Which instructor name teaches courses with the highest total number of enrollments?",
                 "SELECT i.inst_name FROM instructors i JOIN courses c ON i.inst_id = c.inst_id JOIN enrollments e ON c.course_id = e.course_id GROUP BY i.inst_id, i.inst_name ORDER BY COUNT(*) DESC LIMIT 1;"),
                ("List course_code values whose average enrollment score is strictly above 88, ordered alphabetically.",
                 "SELECT c.course_code FROM courses c JOIN enrollments e ON c.course_id = e.course_id GROUP BY c.course_id, c.course_code HAVING AVG(e.score) > 88 ORDER BY c.course_code;"),
                ("Return each student_name and their highest individual score, ordered by student_name alphabetically.",
                 "SELECT student_name, MAX(score) FROM enrollments GROUP BY student_name ORDER BY student_name;"),
                ("How many enrollments in the 'Math' faculty achieved a score of 90 or higher?",
                 "SELECT COUNT(*) FROM enrollments e JOIN courses c ON e.course_id = c.course_id JOIN instructors i ON c.inst_id = i.inst_id WHERE i.faculty = 'Math' AND e.score >= 90;"),
                ("List student names enrolled in BOTH 'ENG201' and 'MTH210', ordered alphabetically.",
                 "SELECT e1.student_name FROM enrollments e1 JOIN courses c1 ON e1.course_id = c1.course_id JOIN enrollments e2 ON e1.student_name = e2.student_name JOIN courses c2 ON e2.course_id = c2.course_id WHERE c1.course_code = 'ENG201' AND c2.course_code = 'MTH210' ORDER BY e1.student_name;"),
            ],
        },
        # 7. Flight Manifests
        {
            "schema_id": "flight_manifests",
            "ddl": """
CREATE TABLE aircraft (ac_id INTEGER PRIMARY KEY, tail_code TEXT, seat_cap INTEGER, range_km INTEGER);
CREATE TABLE flights (flt_id INTEGER PRIMARY KEY, ac_id INTEGER, origin_hub TEXT, dist_km INTEGER);
CREATE TABLE passengers (pax_id INTEGER PRIMARY KEY, flt_id INTEGER, cabin TEXT, bag_kg REAL);
""",
            "seed": """
INSERT INTO aircraft VALUES (1,'N101X',180,5500),(2,'N202Y',120,3200),(3,'N303Z',240,8000);
INSERT INTO flights VALUES (10,1,'Hub-A',2400),(11,1,'Hub-B',1800),(12,2,'Hub-A',1200),(13,3,'Hub-C',6200);
INSERT INTO passengers VALUES
  (1,10,'economy',18.0),(2,10,'business',26.0),(3,10,'economy',15.0),
  (4,11,'economy',20.0),(5,11,'business',24.0),
  (6,12,'economy',14.0),(7,12,'economy',16.0),
  (8,13,'business',30.0),(9,13,'first',32.0),(10,13,'economy',22.0);
""",
            "queries": [
                ("List aircraft tail_code values with range_km above 4000, ordered by seat_cap descending.",
                 "SELECT tail_code FROM aircraft WHERE range_km > 4000 ORDER BY seat_cap DESC;"),
                ("What is the total bag_kg across all passengers on flights departing from 'Hub-A'?",
                 "SELECT SUM(p.bag_kg) FROM passengers p JOIN flights f ON p.flt_id = f.flt_id WHERE f.origin_hub = 'Hub-A';"),
                ("Return each cabin class and its average bag_kg, ordered by cabin alphabetically.",
                 "SELECT cabin, AVG(bag_kg) FROM passengers GROUP BY cabin ORDER BY cabin;"),
                ("Which aircraft tail_code has flown the highest total dist_km?",
                 "SELECT a.tail_code FROM aircraft a JOIN flights f ON a.ac_id = f.ac_id GROUP BY a.ac_id, a.tail_code ORDER BY SUM(f.dist_km) DESC LIMIT 1;"),
                ("How many passengers flew on aircraft 'N101X'?",
                 "SELECT COUNT(*) FROM passengers p JOIN flights f ON p.flt_id = f.flt_id JOIN aircraft a ON f.ac_id = a.ac_id WHERE a.tail_code = 'N101X';"),
                ("Which origin_hub has the highest average passenger bag_kg?",
                 "SELECT f.origin_hub FROM flights f JOIN passengers p ON f.flt_id = p.flt_id GROUP BY f.origin_hub ORDER BY AVG(p.bag_kg) DESC LIMIT 1;"),
                ("Return each flt_id and the number of passengers on that flight, ordered by flt_id ascending.",
                 "SELECT flt_id, COUNT(*) FROM passengers GROUP BY flt_id ORDER BY flt_id;"),
                ("List tail_code values of aircraft that carried at least one 'business' cabin passenger, ordered alphabetically.",
                 "SELECT DISTINCT a.tail_code FROM aircraft a JOIN flights f ON a.ac_id = f.ac_id JOIN passengers p ON f.flt_id = p.flt_id WHERE p.cabin = 'business' ORDER BY a.tail_code;"),
                ("What is the maximum bag_kg among 'economy' passengers?",
                 "SELECT MAX(bag_kg) FROM passengers WHERE cabin = 'economy';"),
                ("Which flights (flt_id) have total passenger bag_kg strictly above 50 kg? Order by flt_id.",
                 "SELECT flt_id FROM passengers GROUP BY flt_id HAVING SUM(bag_kg) > 50 ORDER BY flt_id;"),
            ],
        },
        # 8. Sensor Telemetry
        {
            "schema_id": "sensor_telemetry",
            "ddl": """
CREATE TABLE zones (zone_id INTEGER PRIMARY KEY, zone_name TEXT, building TEXT);
CREATE TABLE sensors (sensor_id INTEGER PRIMARY KEY, zone_id INTEGER, kind TEXT, battery_pct INTEGER);
CREATE TABLE alerts (alert_id INTEGER PRIMARY KEY, sensor_id INTEGER, severity TEXT, latency_ms INTEGER);
""",
            "seed": """
INSERT INTO zones VALUES (1,'ServerVault','Alpha'),(2,'CryoLab','Alpha'),(3,'RoofDeck','Beta');
INSERT INTO sensors VALUES (10,1,'temp',92),(11,1,'smoke',45),(12,2,'temp',80),(13,2,'pressure',30),(14,3,'wind',65);
INSERT INTO alerts VALUES
  (1,10,'high',120),(2,10,'medium',90),(3,11,'critical',55),(4,12,'high',140),
  (5,13,'critical',60),(6,13,'high',110),(7,14,'low',200),(8,11,'high',80),(9,14,'medium',160);
""",
            "queries": [
                ("List sensor_id values with battery_pct strictly below 50, ordered by sensor_id ascending.",
                 "SELECT sensor_id FROM sensors WHERE battery_pct < 50 ORDER BY sensor_id;"),
                ("How many alerts have severity 'critical' or 'high'?",
                 "SELECT COUNT(*) FROM alerts WHERE severity IN ('critical', 'high');"),
                ("What is the average latency_ms of alerts originating in building 'Alpha'?",
                 "SELECT AVG(a.latency_ms) FROM alerts a JOIN sensors s ON a.sensor_id = s.sensor_id JOIN zones z ON s.zone_id = z.zone_id WHERE z.building = 'Alpha';"),
                ("Which zone_name triggered the highest total number of alerts?",
                 "SELECT z.zone_name FROM zones z JOIN sensors s ON z.zone_id = s.zone_id JOIN alerts a ON s.sensor_id = a.sensor_id GROUP BY z.zone_id, z.zone_name ORDER BY COUNT(*) DESC, z.zone_name ASC LIMIT 1;"),
                ("Return each sensor kind and the number of alerts it generated, ordered by kind alphabetically.",
                 "SELECT s.kind, COUNT(*) FROM sensors s JOIN alerts a ON s.sensor_id = a.sensor_id GROUP BY s.kind ORDER BY s.kind;"),
                ("Which sensor kind had the lowest minimum alert latency_ms?",
                 "SELECT s.kind FROM sensors s JOIN alerts a ON s.sensor_id = a.sensor_id GROUP BY s.kind ORDER BY MIN(a.latency_ms) ASC LIMIT 1;"),
                ("Return each severity level and its average latency_ms, ordered by average latency ascending.",
                 "SELECT severity, AVG(latency_ms) AS avg_lat FROM alerts GROUP BY severity ORDER BY avg_lat ASC;"),
                ("List zone names that had at least one 'critical' alert, ordered alphabetically.",
                 "SELECT DISTINCT z.zone_name FROM zones z JOIN sensors s ON z.zone_id = s.zone_id JOIN alerts a ON s.sensor_id = a.sensor_id WHERE a.severity = 'critical' ORDER BY z.zone_name;"),
                ("What is the average battery_pct of sensors located in building 'Alpha'?",
                 "SELECT AVG(s.battery_pct) FROM sensors s JOIN zones z ON s.zone_id = z.zone_id WHERE z.building = 'Alpha';"),
                ("List sensor_id values that generated at least 2 alerts with latency_ms under 120, ordered by sensor_id.",
                 "SELECT sensor_id FROM alerts WHERE latency_ms < 120 GROUP BY sensor_id HAVING COUNT(*) >= 2 ORDER BY sensor_id;"),
            ],
        },
    ]

    items: List[Dict[str, Any]] = []
    for sch in schemas:
        for qi, (question, gold_sql) in enumerate(sch["queries"]):
            items.append({
                "id": f"{sch['schema_id']}_q{qi:02d}",
                "schema_id": sch["schema_id"],
                "ddl": sch["ddl"].strip(),
                "seed": sch["seed"].strip(),
                "question": question,
                "gold_sql": gold_sql,
                "order_sensitive": ("order by" in gold_sql.lower()),
            })
    assert len(items) == 80, f"Expected 80 SQL queries, got {len(items)}"
    return items


def execute_sql_on_schema(ddl: str, seed: str, sql: str) -> Tuple[bool, Optional[List[Tuple[Any, ...]]], Optional[str]]:
    """Execute read-only SQL query in fresh in-memory sqlite3 DB; return (ok, rows, error_msg)."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(ddl + "\n" + seed)
        cur = conn.cursor()
        # Reject non-SELECT statements for safety
        stripped = sql.strip().rstrip(";").strip()
        if not re.match(r"^(SELECT|WITH)\b", stripped, flags=re.IGNORECASE):
            return False, None, "Non-SELECT query rejected"
        cur.execute(stripped)
        rows = cur.fetchall()
        return True, rows, None
    except Exception as exc:
        return False, None, str(exc)
    finally:
        conn.close()


def _norm_cell(val: Any) -> Any:
    if isinstance(val, float):
        return round(val, 4)
    if isinstance(val, str):
        return val.strip()
    return val


def sql_results_match(
    gold_rows: List[Tuple[Any, ...]],
    pred_rows: Optional[List[Tuple[Any, ...]]],
    order_sensitive: bool = False,
) -> bool:
    if pred_rows is None or len(pred_rows) != len(gold_rows):
        return False
    g_norm = [tuple(_norm_cell(c) for c in r) for r in gold_rows]
    p_norm = [tuple(_norm_cell(c) for c in r) for r in pred_rows]
    if order_sensitive:
        return g_norm == p_norm
    return sorted(g_norm, key=repr) == sorted(p_norm, key=repr)


def extract_sql_from_generation(text: str) -> str:
    """Extract clean SQL SELECT/WITH statement from model generation (handling markdown fences)."""
    t = text.strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", t, flags=re.IGNORECASE | re.DOTALL)
    if m:
        t = m.group(1).strip()
    m2 = re.search(r"\b(SELECT|WITH)\b.*", t, flags=re.IGNORECASE | re.DOTALL)
    if m2:
        t = m2.group(0).strip()
    # Keep first statement up to first semicolon
    if ";" in t:
        t = t.split(";", 1)[0].strip() + ";"
    return t


# ======================================================================================
# 4. GPU Runners: Part A (2-Hop Routing Policy Swap) & Part B (Text-to-SQLite Headroom)
# ======================================================================================

def _tokenize_batch(tok: Any, prompts: Sequence[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    old_pad = tok.padding_side
    tok.padding_side = "left"
    enc = tok(list(prompts), return_tensors="pt", padding=True, add_special_tokens=True)
    tok.padding_side = old_pad
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def _extract_last_tok_logits_and_layers(
    model: Any,
    tok: Any,
    prompts: Sequence[str],
    layer_indices: Sequence[int],
    device: torch.device,
    batch_size: int = 8,
    capture_mlp_mid: bool = False,
) -> Dict[str, Any]:
    """Extract last-token logits, residual stream states at `layer_indices`, and optional SwiGLU mid acts."""
    layers = get_decoder_layers(model)
    buf: Dict[int, Dict[str, torch.Tensor]] = {li: {} for li in layer_indices}
    hooks = []
    for li in layer_indices:
        def _mk_h(idx: int):
            def _fn(_m, _i, out):
                t = out[0] if isinstance(out, tuple) else out
                buf[idx]["h"] = t[:, -1, :].detach().float().cpu()
            return _fn
        def _mk_post_ln(idx: int):
            def _fn(_m, _i, out):
                buf[idx]["x_mlp"] = out[:, -1, :].detach().float().cpu()
            return _fn
        hooks.append(layers[li].register_forward_hook(_mk_h(li)))
        if capture_mlp_mid:
            hooks.append(layers[li].post_attention_layernorm.register_forward_hook(_mk_post_ln(li)))
            def _mk_down(idx: int):
                def _pre(_m, args):
                    buf[idx]["z_down"] = args[0][:, -1, :].detach().float().cpu()
                return _pre
            hooks.append(layers[li].mlp.down_proj.register_forward_pre_hook(_mk_down(li)))

    logits_l = []
    h_store: Dict[int, List[torch.Tensor]] = {li: [] for li in layer_indices}
    x_store: Dict[int, List[torch.Tensor]] = {li: [] for li in layer_indices}
    z_store: Dict[int, List[torch.Tensor]] = {li: [] for li in layer_indices}
    try:
        with torch.no_grad():
            for i0 in range(0, len(prompts), batch_size):
                ids, mask = _tokenize_batch(tok, prompts[i0 : i0 + batch_size], device)
                out = model(input_ids=ids, attention_mask=mask, use_cache=False)
                logits_l.append(out.logits[:, -1, :].detach().float().cpu())
                for li in layer_indices:
                    h_store[li].append(buf[li]["h"])
                    if capture_mlp_mid:
                        x_store[li].append(buf[li]["x_mlp"])
                        z_store[li].append(buf[li]["z_down"])
    finally:
        for h in hooks:
            h.remove()

    return {
        "logits": torch.cat(logits_l, dim=0),
        "h": {li: torch.cat(h_store[li], dim=0).numpy().astype(np.float64) for li in layer_indices},
        "x_mlp": {li: torch.cat(x_store[li], dim=0).numpy().astype(np.float64) for li in layer_indices} if capture_mlp_mid else {},
        "z_down": {li: torch.cat(z_store[li], dim=0).numpy().astype(np.float64) for li in layer_indices} if capture_mlp_mid else {},
    }


def _eval_2hop_policy_logits(
    logits: torch.Tensor,
    items: Sequence[Dict[str, Any]],
    tok: Any,
    policy: str,  # "L" or "R"
) -> Dict[str, Any]:
    """Score exact next-token greedy EM and forced-choice (val_L vs val_R) for Policy L or R."""
    em_vec, fc_vec = [], []
    for i, it in enumerate(items):
        tid_L = tok.encode(" " + it["val_L"], add_special_tokens=False)[0]
        tid_R = tok.encode(" " + it["val_R"], add_special_tokens=False)[0]
        row = logits[i]
        pred_id = int(row.argmax().item())
        tgt_id = tid_L if policy == "L" else tid_R
        oth_id = tid_R if policy == "L" else tid_L
        em_vec.append(1 if pred_id == tgt_id else 0)
        fc_vec.append(1 if float(row[tgt_id].item()) > float(row[oth_id].item()) else 0)
    return {
        "em": round(float(np.mean(em_vec)), 4),
        "fc": round(float(np.mean(fc_vec)), 4),
        "_em_vec": em_vec,
        "_fc_vec": fc_vec,
    }


def _joint_swap_summary(res_L: Dict[str, Any], res_R: Dict[str, Any]) -> Dict[str, float]:
    eL, eR = np.asarray(res_L["_em_vec"]), np.asarray(res_R["_em_vec"])
    fL, fR = np.asarray(res_L["_fc_vec"]), np.asarray(res_R["_fc_vec"])
    return {
        "left_em": res_L["em"],
        "right_em": res_R["em"],
        "joint_swap_em": round(float(np.mean((eL == 1) & (eR == 1))), 4),
        "left_fc": res_L["fc"],
        "right_fc": res_R["fc"],
        "joint_swap_fc": round(float(np.mean((fL == 1) & (fR == 1))), 4),
    }


def _map_donor_topk_to_recip_probs(
    donor_logits: torch.Tensor,
    tok_A: Any,
    tok_B: Any,
    top_k: int = 16,
    temp: float = 1.0,
) -> List[Tuple[List[int], List[float]]]:
    """
    Zero-Ground-Truth-Label Cross-Tokenizer Teacher Distribution Transfer:
    Decodes Donor 7B's top-K next tokens and maps 1-token matches into Recipient B's vocab IDs!
    """
    mapped: List[Tuple[List[int], List[float]]] = []
    probs = F.softmax(donor_logits.float() / float(temp), dim=-1)
    topv, topi = torch.topk(probs, k=top_k, dim=-1)
    for row_v, row_i in zip(topv.tolist(), topi.tolist()):
        acc: Dict[int, float] = {}
        for p, tid_a in zip(row_v, row_i):
            s = tok_A.decode([tid_a])
            if not s:
                continue
            b_ids = tok_B.encode(s, add_special_tokens=False)
            if len(b_ids) == 1:
                acc[b_ids[0]] = acc.get(b_ids[0], 0.0) + float(p)
        if not acc:
            # Fallback to argmax decoded first token
            s0 = tok_A.decode([row_i[0]])
            b0 = tok_B.encode(s0, add_special_tokens=False)
            if b0:
                acc[b0[0]] = 1.0
        tot = sum(acc.values()) + 1e-12
        tids = list(acc.keys())
        ps = [v / tot for v in acc.values()]
        mapped.append((tids, ps))
    return mapped


def _eval_100fact_ood_locality(
    model: Any,
    tok: Any,
    device: torch.device,
    facts: Sequence[Dict[str, Any]],
    base_correct: Optional[List[int]] = None,
    organ: Optional[Removable2BlockSidecarOrgan] = None,
) -> Tuple[Dict[str, Any], List[int]]:
    """Fast 100-fact OOD locality check on `dataset.json` test split."""
    from extract_and_baseline import check_answer_match

    prompts = [f.get("direct_prompt", f.get("prompt", "")) for f in facts]
    ok_vec: List[int] = []
    old_pad = tok.padding_side
    tok.padding_side = "left"
    try:
        with torch.no_grad():
            for i0 in range(0, len(prompts), 16):
                enc = tok(prompts[i0 : i0 + 16], return_tensors="pt", padding=True, add_special_tokens=True)
                mask = enc["attention_mask"].to(device)
                if organ is not None:
                    organ.set_attention_mask(mask)
                gen = model.generate(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=mask,
                    max_new_tokens=8,
                    do_sample=False,
                    use_cache=False,
                    pad_token_id=tok.pad_token_id,
                )
                in_len = enc["input_ids"].shape[1]
                for bi, seq in enumerate(gen):
                    txt = tok.decode(seq[in_len:], skip_special_tokens=True)
                    f = facts[i0 + bi]
                    ok_vec.append(1 if check_answer_match(txt, f["answer"], f.get("aliases", [])) else 0)
    finally:
        tok.padding_side = old_pad

    n_ok = int(sum(ok_vec))
    res = {"n_facts": len(facts), "n_correct": n_ok, "acc": round(n_ok / max(len(facts), 1), 4)}
    if base_correct is not None:
        base_n = int(sum(base_correct))
        broken = int(sum(1 for b, c in zip(base_correct, ok_vec) if b == 1 and c == 0))
        res["base_correct"] = base_n
        res["newly_broken"] = broken
        res["retention"] = round((base_n - broken) / max(base_n, 1), 4)
    return res, ok_vec


def run_part_a_2hop(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute Part A: Counterfactual 2-Hop Routing Policy Swap (Tier 1 vs Tier 2A/B/C vs Tier 3)."""
    t0 = time.time()
    from transformers import AutoTokenizer

    donor_key = args.donor_model
    recip_key = args.recipient_model
    print(f"\n================ PART A: COUNTERFACTUAL 2-HOP POLICY SWAP ({donor_key} -> {recip_key}) ================")

    # Load fast tokenizers to construct strictly 1-token disjoint 2-hop tables
    src_A = str(HF_CACHE_DIR / donor_key) if (HF_CACHE_DIR / donor_key).is_dir() else LARGE_TO_SMALL_MODELS.get(donor_key, donor_key)
    src_B = str(HF_CACHE_DIR / recip_key) if (HF_CACHE_DIR / recip_key).is_dir() else LARGE_TO_SMALL_MODELS.get(recip_key, recip_key)
    tok_A_pre = AutoTokenizer.from_pretrained(src_A, local_files_only=(HF_CACHE_DIR / donor_key).is_dir())
    tok_B_pre = AutoTokenizer.from_pretrained(src_B, local_files_only=(HF_CACHE_DIR / recip_key).is_dir())
    ds = build_2hop_routing_dataset([tok_A_pre, tok_B_pre], n_train=args.n_train, n_test=args.n_test, seed=SEED)
    train_items, test_items = ds["train"], ds["test"]
    print(f"  Dataset: {len(train_items)} train / {len(test_items)} sealed test | disjoint keys: {len(ds['train_key_vocab'])}/{len(ds['test_key_vocab'])} | vals: {len(ds['val_vocab'])}")

    # 1. Load Donor 7B (BF16) -> Verify Policy L & R -> Extract Soft Teacher Logits + Mid-Layer States -> UNLOAD!
    model_A, tok_A, dev_A, _ = load_model_bf16(donor_key, allow_download=args.allow_download)
    L_A = len(get_decoder_layers(model_A))
    lA1, lA2 = round(L_A * 0.50), round(L_A * 0.75)  # 14, 21 on 28-layer Qwen2.5-7B-Instruct!
    print(f"  Loaded Donor {donor_key} (L={L_A}, hint layers=({lA1},{lA2}))")

    d_tr_L = _extract_last_tok_logits_and_layers(model_A, tok_A, [x["donor_L_prompt"] for x in train_items], [lA1, lA2], dev_A, batch_size=args.batch_size)
    d_tr_R = _extract_last_tok_logits_and_layers(model_A, tok_A, [x["donor_R_prompt"] for x in train_items], [lA1, lA2], dev_A, batch_size=args.batch_size)
    d_te_L = _extract_last_tok_logits_and_layers(model_A, tok_A, [x["donor_L_prompt"] for x in test_items], [lA1], dev_A, batch_size=args.batch_size)
    d_te_R = _extract_last_tok_logits_and_layers(model_A, tok_A, [x["donor_R_prompt"] for x in test_items], [lA1], dev_A, batch_size=args.batch_size)

    donor_test_L = _eval_2hop_policy_logits(d_te_L["logits"], test_items, tok_A, "L")
    donor_test_R = _eval_2hop_policy_logits(d_te_R["logits"], test_items, tok_A, "R")
    donor_summary = _joint_swap_summary(donor_test_L, donor_test_R)
    s0_L = tok_A.decode([int(d_te_L["logits"][0].argmax().item())])
    s0_R = tok_A.decode([int(d_te_R["logits"][0].argmax().item())])
    print(
        f"  [DONOR 7B SEALED TEST] L_em={donor_summary['left_em']:.3f}, R_em={donor_summary['right_em']:.3f}, "
        f"joint_swap_em={donor_summary['joint_swap_em']:.3f} | L_fc={donor_summary['left_fc']:.3f}, "
        f"R_fc={donor_summary['right_fc']:.3f}, joint_fc={donor_summary['joint_swap_fc']:.3f} | "
        f"sample0 pred=({s0_L!r}, {s0_R!r}) gold=({test_items[0]['val_L']!r}, {test_items[0]['val_R']!r})"
    )

    # Cross-tokenizer soft teacher distributions (ZERO ground-truth labels!)
    teacher_probs_L = _map_donor_topk_to_recip_probs(d_tr_L["logits"], tok_A, tok_B_pre, top_k=16)
    teacher_probs_R = _map_donor_topk_to_recip_probs(d_tr_R["logits"], tok_A, tok_B_pre, top_k=16)

    # Save donor mid-layer states & strictly UNLOAD Donor 7B!
    hA_tr_L, hA_tr_R = d_tr_L["h"], d_tr_R["h"]
    unload_model_strict(model_A)
    print(f"  Donor {donor_key} 100% UNLOADED!")

    # 2. Load SOLO Recipient SmolLM2-1.7B-Instruct (BF16)
    model_B, tok_B, dev_B, _ = load_model_bf16(recip_key, allow_download=args.allow_download)
    for p in model_B.parameters():
        p.requires_grad_(False)
    layers_B = get_decoder_layers(model_B)
    L_B = len(layers_B)
    D_B = int(model_B.config.hidden_size)
    head_dim_B = int(getattr(model_B.config, "head_dim", D_B // model_B.config.num_attention_heads))
    rope_theta_B = float(getattr(model_B.config, "rope_theta", 130_000.0))
    lB1, lB2 = round(L_B * 0.50), round(L_B * 0.75)  # 12, 18 on 24-layer SmolLM2-1.7B!
    print(f"  Loaded SOLO Recipient {recip_key} (L={L_B}, D={D_B}, target layers=({lB1},{lB2}))")

    # Load 100 OOD locality facts from dataset.json
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        ood_facts = [x for x in json.load(f)["facts"] if x.get("split") == "test"][:100]
    base_loc, base_loc_ok = _eval_100fact_ood_locality(model_B, tok_B, dev_B, ood_facts)
    print(f"  [BASE OOD 100-fact] {base_loc['n_correct']}/{base_loc['n_facts']} ({base_loc['acc']:.3f})")

    # Matched-batch fresh-cache probe for exact bit-identical rollback verification
    probe_ids, probe_mask = _tokenize_batch(tok_B, [x["recipient_prompt"] for x in test_items[:4]], dev_B)
    with torch.no_grad():
        base_probe_logits = model_B(input_ids=probe_ids, attention_mask=probe_mask, use_cache=False).logits.detach().cpu()

    def _verify_rollback() -> Dict[str, Any]:
        with torch.no_grad():
            cur = model_B(input_ids=probe_ids, attention_mask=probe_mask, use_cache=False).logits.detach().cpu()
        return {
            "logits_bit_equal": bool(torch.equal(cur, base_probe_logits)),
            "max_abs_diff": float((cur.float() - base_probe_logits.float()).abs().max().item()),
        }

    # Baseline Recipient on identical instruction-free prompts
    r_tr_base = _extract_last_tok_logits_and_layers(
        model_B, tok_B, [x["recipient_prompt"] for x in train_items], [lB1, lB2], dev_B, batch_size=args.batch_size, capture_mlp_mid=True
    )
    r_te_base = _extract_last_tok_logits_and_layers(
        model_B, tok_B, [x["recipient_prompt"] for x in test_items], [lB1], dev_B, batch_size=args.batch_size
    )
    base_L = _eval_2hop_policy_logits(r_te_base["logits"], test_items, tok_B, "L")
    base_R = _eval_2hop_policy_logits(r_te_base["logits"], test_items, tok_B, "R")
    print(f"  [BASE RECIPIENT] L_em={base_L['em']:.3f}, R_em={base_R['em']:.3f}, L_fc={base_L['fc']:.3f}, R_fc={base_R['fc']:.3f}")

    # Contrast bridges mapping Donor 7B (L14, L21) -> Recipient 1.7B (L12, L18)
    hA1_mid = 0.5 * (hA_tr_L[lA1] + hA_tr_R[lA1])
    hA2_mid = 0.5 * (hA_tr_L[lA2] + hA_tr_R[lA2])
    M_1, m1_info = fit_contrast_bridge(hA1_mid - hA1_mid.mean(0), r_tr_base["h"][lB1] - r_tr_base["h"][lB1].mean(0), mode="ridge")
    M_2, m2_info = fit_contrast_bridge(hA2_mid - hA2_mid.mean(0), r_tr_base["h"][lB2] - r_tr_base["h"][lB2].mean(0), mode="ridge")

    dH_target = {
        "L": {lB1: (hA_tr_L[lA1] - hA1_mid) @ M_1.T, lB2: (hA_tr_L[lA2] - hA2_mid) @ M_2.T},
        "R": {lB1: (hA_tr_R[lA1] - hA1_mid) @ M_1.T, lB2: (hA_tr_R[lA2] - hA2_mid) @ M_2.T},
    }
    # OOD background MLP keys for Tier 1 locality preservation
    ood_st = _extract_last_tok_logits_and_layers(
        model_B, tok_B, [f.get("direct_prompt", f.get("prompt", "")) for f in ood_facts[:32]], [lB1, lB2], dev_B, batch_size=8, capture_mlp_mid=True
    )
    C_bg_1 = (ood_st["z_down"][lB1].T @ ood_st["z_down"][lB1]) / float(len(ood_st["z_down"][lB1]))
    C_bg_2 = (ood_st["z_down"][lB2].T @ ood_st["z_down"][lB2]) / float(len(ood_st["z_down"][lB2]))

    tiers_out: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Tier 1: CLOSED_FORM_SEQ_DELTA_W (ZERO Backprop! Sequential L12 -> L18 Reversible ΔW!)
    # ------------------------------------------------------------------
    t1_0 = time.time()
    t1_packs: Dict[str, ReversibleDeltaPack] = {}
    t1_eval: Dict[str, Any] = {}
    for pol in ("L", "R"):
        dW1, _ = solve_closed_form_layer_delta_w(r_tr_base["z_down"][lB1], dH_target[pol][lB1], Z_bg_cov=C_bg_1, lam_bg=5.0, lam_eye=0.05)
        p1_name = f"model.layers.{lB1}.mlp.down_proj.weight"
        p2_name = f"model.layers.{lB2}.mlp.down_proj.weight"
        pack1 = ReversibleDeltaPack({p1_name: torch.from_numpy(dW1.astype(np.float32))})
        pack1.apply(model_B)
        # Sequential depth-cocycle refit at L18 on L12-edited recipient!
        r_tr_ed = _extract_last_tok_logits_and_layers(
            model_B, tok_B, [x["recipient_prompt"] for x in train_items], [lB2], dev_B, batch_size=args.batch_size, capture_mlp_mid=True
        )
        # RMSNorm-aligned cocycle diagnostic between post-LN L18 states
        gB2 = layers_B[lB2].post_attention_layernorm.weight.detach().float().cpu().numpy().astype(np.float64)
        U_B_norm = apply_rmsnorm_np(dH_target[pol][lB1][:8], gB2).T
        U_A_norm = (hA_tr_L[lA1][:8] - hA1_mid[:8]).T
        _, rms_eq_err = fit_rmsnorm_aligned_cocycle_map(hA2_mid, r_tr_ed["x_mlp"][lB2], U_A_norm, U_B_norm)
        dW2, _ = solve_closed_form_layer_delta_w(r_tr_ed["z_down"][lB2], dH_target[pol][lB2], Z_bg_cov=C_bg_2, lam_bg=5.0, lam_eye=0.05)
        pack1.revert(model_B)

        joint_pack = ReversibleDeltaPack(
            {p1_name: torch.from_numpy(dW1.astype(np.float32)), p2_name: torch.from_numpy(dW2.astype(np.float32))},
            metadata={"rms_cocycle_eq_err": rms_eq_err},
        )
        t1_packs[pol] = joint_pack
        joint_pack.apply(model_B)
        te_out = _extract_last_tok_logits_and_layers(model_B, tok_B, [x["recipient_prompt"] for x in test_items], [lB1], dev_B, batch_size=args.batch_size)
        t1_eval[pol] = _eval_2hop_policy_logits(te_out["logits"], test_items, tok_B, pol)
        if pol == "L":
            t1_loc, _ = _eval_100fact_ood_locality(model_B, tok_B, dev_B, ood_facts, base_correct=base_loc_ok)
        joint_pack.revert(model_B)
    t1_rb = _verify_rollback()
    t1_sum = _joint_swap_summary(t1_eval["L"], t1_eval["R"])
    tiers_out["Tier1_CLOSED_FORM_SEQ_DELTA_W"] = {
        **t1_sum, "n_bytes_bf16": t1_packs["L"].n_bytes_bf16(), "train_sec": round(time.time() - t1_0, 1),
        "locality_100": t1_loc, "rollback": t1_rb,
    }
    print(f"  [Tier 1 CLOSED_FORM_SEQ_DELTA_W] joint_em={t1_sum['joint_swap_em']:.3f} (L={t1_sum['left_em']:.3f}, R={t1_sum['right_em']:.3f}) | loc_broken={t1_loc['newly_broken']} | rb={t1_rb['logits_bit_equal']}")

    # ------------------------------------------------------------------
    # Tier 2A / 2B / 2C: Removable 2-Block Sidecar Organ Distillation (ZERO Ground-Truth Labels!)
    # ------------------------------------------------------------------
    def _teacher_kl_loss(logits_last: torch.Tensor, batch_teacher: Sequence[Tuple[List[int], List[float]]]) -> torch.Tensor:
        lps = F.log_softmax(logits_last.float(), dim=-1)
        loss_sum = torch.tensor(0.0, device=logits_last.device)
        for bi, (tids, ps) in enumerate(batch_teacher):
            idx_t = torch.tensor(tids, device=logits_last.device, dtype=torch.long)
            p_t = torch.tensor(ps, device=logits_last.device, dtype=torch.float32)
            loss_sum = loss_sum - torch.sum(p_t * lps[bi, idx_t])
        return loss_sum / float(len(batch_teacher))

    def _train_and_eval_sidecar(arm_name: str, hint_weight: float, shuffle_supervision: bool) -> Dict[str, Any]:
        ts0 = time.time()
        organs: Dict[str, Removable2BlockSidecarOrgan] = {}
        eval_pol: Dict[str, Any] = {}
        ood_prompts_reg = [f.get("direct_prompt", f.get("prompt", "")) for f in ood_facts[:16]]
        ood_ids_reg, ood_mask_reg = _tokenize_batch(tok_B, ood_prompts_reg, dev_B)

        for pol in ("L", "R"):
            torch.manual_seed(SEED + (101 if pol == "L" else 202))
            organ = Removable2BlockSidecarOrgan(D_B, layer_indices=(lB1, lB2), head_dim=head_dim_B, mlp_width=64, rope_theta=rope_theta_B).to(dev_B)
            organ.attach(model_B)
            opt = torch.optim.AdamW(organ.parameters(), lr=args.sidecar_lr, weight_decay=0.01)

            t_probs = list(teacher_probs_L if pol == "L" else teacher_probs_R)
            if shuffle_supervision:
                rng_sh = np.random.default_rng(SEED + (555 if pol == "L" else 666))
                perm = rng_sh.permutation(len(t_probs)).tolist()
                t_probs = [t_probs[i] for i in perm]

            tr_prompts = [x["recipient_prompt"] for x in train_items]
            tgt_d1 = torch.from_numpy(dH_target[pol][lB1].astype(np.float32)).to(dev_B)
            tgt_d2 = torch.from_numpy(dH_target[pol][lB2].astype(np.float32)).to(dev_B)
            rng_ep = np.random.default_rng(SEED + 33)

            for ep in range(args.sidecar_epochs):
                order = rng_ep.permutation(len(tr_prompts)).tolist()
                for i0 in range(0, len(order), args.train_batch_size):
                    b_idx = order[i0 : i0 + args.train_batch_size]
                    ids, mask = _tokenize_batch(tok_B, [tr_prompts[i] for i in b_idx], dev_B)
                    organ.set_attention_mask(mask)
                    out = model_B(input_ids=ids, attention_mask=mask, use_cache=False)
                    loss = _teacher_kl_loss(out.logits[:, -1, :], [t_probs[i] for i in b_idx])
                    if hint_weight > 0.0:
                        idx_t = torch.tensor(b_idx, device=dev_B, dtype=torch.long)
                        d1_pred = organ.last_deltas[lB1][:, -1, :].float()
                        d2_pred = organ.last_deltas[lB2][:, -1, :].float()
                        loss = loss + hint_weight * (
                            F.mse_loss(d1_pred, tgt_d1[idx_t]) + F.mse_loss(d2_pred, tgt_d2[idx_t])
                        )
                    # Light OOD silence anchor every 4 steps
                    if (i0 // args.train_batch_size) % 4 == 0:
                        oi = (ep * 4 + (i0 // args.train_batch_size)) % len(ood_prompts_reg)
                        organ.set_attention_mask(ood_mask_reg[oi : oi + 2])
                        _ = model_B(input_ids=ood_ids_reg[oi : oi + 2], attention_mask=ood_mask_reg[oi : oi + 2], use_cache=False)
                        loss = loss + 0.05 * (organ.last_deltas[lB1].float().pow(2).mean() + organ.last_deltas[lB2].float().pow(2).mean())

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(organ.parameters(), 1.0)
                    opt.step()

            # Eval on 256 sealed test tables
            organ.eval()
            te_logits_l = []
            with torch.no_grad():
                for i0 in range(0, len(test_items), args.batch_size):
                    ids, mask = _tokenize_batch(tok_B, [x["recipient_prompt"] for x in test_items[i0 : i0 + args.batch_size]], dev_B)
                    organ.set_attention_mask(mask)
                    te_logits_l.append(model_B(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1, :].float().cpu())
            eval_pol[pol] = _eval_2hop_policy_logits(torch.cat(te_logits_l, dim=0), test_items, tok_B, pol)
            if pol == "L":
                arm_loc, _ = _eval_100fact_ood_locality(model_B, tok_B, dev_B, ood_facts, base_correct=base_loc_ok, organ=organ)
                organ.set_attention_mask(None)
            organ.detach()
            organs[pol] = organ

        # Verify modular hot-swap (L -> R -> bypass) & exact torch.equal rollback!
        organs["L"].attach(model_B)
        organs["L"].set_attention_mask(probe_mask)
        with torch.no_grad():
            pL_lg = model_B(input_ids=probe_ids, attention_mask=probe_mask, use_cache=False).logits[:, -1, :].float().cpu()
        organs["L"].detach()
        organs["R"].attach(model_B)
        organs["R"].set_attention_mask(probe_mask)
        with torch.no_grad():
            pR_lg = model_B(input_ids=probe_ids, attention_mask=probe_mask, use_cache=False).logits[:, -1, :].float().cpu()
        organs["R"].enabled = False
        with torch.no_grad():
            pByp_lg = model_B(input_ids=probe_ids, attention_mask=probe_mask, use_cache=False).logits.detach().cpu()
        organs["R"].detach()
        rb = _verify_rollback()
        bypass_eq = bool(torch.equal(pByp_lg, base_probe_logits))
        swap_diff = float((pL_lg - pR_lg).abs().max().item())

        sm = _joint_swap_summary(eval_pol["L"], eval_pol["R"])
        print(
            f"  [{arm_name:34s}] joint_em={sm['joint_swap_em']:.3f} (L={sm['left_em']:.3f}, R={sm['right_em']:.3f}) "
            f"joint_fc={sm['joint_swap_fc']:.3f} | bytes={organs['L'].n_bytes_bf16()/1e6:.2f}MB | "
            f"loc_broken={arm_loc['newly_broken']} | bypass_eq={bypass_eq} rb={rb['logits_bit_equal']} ({time.time()-ts0:.1f}s)"
        )
        return {
            **sm,
            "n_params": organs["L"].n_params(),
            "n_bytes_bf16": organs["L"].n_bytes_bf16(),
            "train_sec": round(time.time() - ts0, 1),
            "locality_100": arm_loc,
            "hot_swap_max_logit_diff": round(swap_diff, 4),
            "hard_bypass_bit_equal": bypass_eq,
            "rollback": rb,
        }

    tiers_out["Tier2A_SIDECAR_2BLOCK_WITH_HINT"] = _train_and_eval_sidecar("Tier2A_SIDECAR_2BLOCK_WITH_HINT", hint_weight=0.25, shuffle_supervision=False)
    tiers_out["Tier2B_SIDECAR_OUTPUT_ONLY_CTRL"] = _train_and_eval_sidecar("Tier2B_SIDECAR_OUTPUT_ONLY_CTRL", hint_weight=0.0, shuffle_supervision=False)
    tiers_out["Tier2C_SIDECAR_SHUFFLED_CTRL"] = _train_and_eval_sidecar("Tier2C_SIDECAR_SHUFFLED_CTRL", hint_weight=0.0, shuffle_supervision=True)

    # ------------------------------------------------------------------
    # Tier 3: TEACHER_LORA_BASELINE (Fair Rank-16 LoRA on Exact Same 192 Donor 7B Logit Pairs!)
    # ------------------------------------------------------------------
    t3_0 = time.time()
    lora_eval: Dict[str, Any] = {}
    lora_targets = {}
    for li in range(lB1, min(L_B, lB2 + 3)):
        lyr = layers_B[li]
        for sub in ("self_attn.q_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.down_proj"):
            mod = lyr
            for part in sub.split("."):
                mod = getattr(mod, part)
            lora_targets[f"L{li}.{sub}"] = mod

    lora_bytes = 0
    for pol in ("L", "R"):
        torch.manual_seed(SEED + (301 if pol == "L" else 402))
        lora = ReversibleLoRALinearHook(lora_targets, rank=16, alpha=32.0, device=dev_B).attach()
        lora_bytes = lora.n_bytes_bf16()
        opt = torch.optim.AdamW(lora.parameters(), lr=1e-3, weight_decay=0.01)
        t_probs = teacher_probs_L if pol == "L" else teacher_probs_R
        tr_prompts = [x["recipient_prompt"] for x in train_items]
        rng_ep = np.random.default_rng(SEED + 77)
        for _ in range(args.lora_epochs):
            order = rng_ep.permutation(len(tr_prompts)).tolist()
            for i0 in range(0, len(order), args.train_batch_size):
                b_idx = order[i0 : i0 + args.train_batch_size]
                ids, mask = _tokenize_batch(tok_B, [tr_prompts[i] for i in b_idx], dev_B)
                out = model_B(input_ids=ids, attention_mask=mask, use_cache=False)
                loss = _teacher_kl_loss(out.logits[:, -1, :], [t_probs[i] for i in b_idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora.parameters(), 1.0)
                opt.step()

        te_lg = []
        with torch.no_grad():
            for i0 in range(0, len(test_items), args.batch_size):
                ids, mask = _tokenize_batch(tok_B, [x["recipient_prompt"] for x in test_items[i0 : i0 + args.batch_size]], dev_B)
                te_lg.append(model_B(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1, :].float().cpu())
        lora_eval[pol] = _eval_2hop_policy_logits(torch.cat(te_lg, dim=0), test_items, tok_B, pol)
        if pol == "L":
            lora_loc, _ = _eval_100fact_ood_locality(model_B, tok_B, dev_B, ood_facts, base_correct=base_loc_ok)
        lora.detach()

    t3_rb = _verify_rollback()
    t3_sum = _joint_swap_summary(lora_eval["L"], lora_eval["R"])
    tiers_out["Tier3_TEACHER_LORA_R16"] = {
        **t3_sum, "n_bytes_bf16": lora_bytes, "train_sec": round(time.time() - t3_0, 1),
        "locality_100": lora_loc, "rollback": t3_rb,
    }
    print(f"  [Tier 3 TEACHER_LORA_R16           ] joint_em={t3_sum['joint_swap_em']:.3f} (L={t3_sum['left_em']:.3f}, R={t3_sum['right_em']:.3f}) | bytes={lora_bytes/1e6:.2f}MB | loc_broken={lora_loc['newly_broken']} | rb={t3_rb['logits_bit_equal']}")

    unload_model_strict(model_B)

    t2a_em = tiers_out["Tier2A_SIDECAR_2BLOCK_WITH_HINT"]["joint_swap_em"]
    t2c_em = tiers_out["Tier2C_SIDECAR_SHUFFLED_CTRL"]["joint_swap_em"]
    gate2a_pass = bool(t2a_em >= 0.70 and (t2a_em - t2c_em) >= 0.30)
    return {
        "donor_model": donor_key,
        "recipient_model": recip_key,
        "n_train": len(train_items),
        "n_test_sealed": len(test_items),
        "donor_7b_test": donor_summary,
        "base_recipient": {"left_em": base_L["em"], "right_em": base_R["em"], "left_fc": base_L["fc"], "right_fc": base_R["fc"], "locality_100": base_loc},
        "contrast_bridges": {"l1": m1_info, "l2": m2_info},
        "tiers": tiers_out,
        "gate2a_pass": gate2a_pass,
        "margin_over_shuffled_pp": round(100.0 * (t2a_em - t2c_em), 2),
        "elapsed_sec": round(time.time() - t0, 1),
    }


def run_part_b_sql_headroom(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute Part B: 80-Query Text-to-SQLite Headroom Check (`Qwen2.5-7B-Instruct` vs `SmolLM2-1.7B-Instruct`)."""
    t0 = time.time()
    bench = build_text_to_sqlite_benchmark()
    # Pre-compute gold execution results once
    gold_cache: Dict[str, List[Tuple[Any, ...]]] = {}
    for it in bench:
        ok, rows, err = execute_sql_on_schema(it["ddl"], it["seed"], it["gold_sql"])
        assert ok and rows is not None and len(rows) > 0, f"Gold SQL failed for {it['id']}: {err}"
        gold_cache[it["id"]] = rows

    print(f"\n================ PART B: TEXT-TO-SQLITE HEADROOM CHECK (80 Queries × 8 Schemas) ================")
    models_out: Dict[str, Any] = {}
    for mkey in (args.donor_model, args.recipient_model):
        tm0 = time.time()
        model, tok, dev, _ = load_model_bf16(mkey, allow_download=args.allow_download)
        exec_hits, valid_hits = 0, 0
        by_schema: Dict[str, Dict[str, int]] = {}
        samples: List[Dict[str, Any]] = []

        for it in bench:
            prompt = (
                "You are an expert SQLite assistant. Given the SQLite schema below, write ONLY a single valid "
                "SQLite SELECT query answering the question. Do not include any explanation.\n\n"
                f"Schema:\n{it['ddl']}\n\nQuestion: {it['question']}\nSQL:\n"
            )
            if hasattr(tok, "apply_chat_template") and tok.chat_template:
                msgs = [{"role": "user", "content": prompt}]
                text_in = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            else:
                text_in = prompt
            enc = tok(text_in, return_tensors="pt", add_special_tokens=True)
            with torch.no_grad():
                gen = model.generate(
                    input_ids=enc["input_ids"].to(dev),
                    attention_mask=enc["attention_mask"].to(dev),
                    max_new_tokens=96,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            gen_txt = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            pred_sql = extract_sql_from_generation(gen_txt)
            ok, rows, err = execute_sql_on_schema(it["ddl"], it["seed"], pred_sql)
            matched = bool(ok and sql_results_match(gold_cache[it["id"]], rows, order_sensitive=it["order_sensitive"]))
            valid_hits += int(ok)
            exec_hits += int(matched)
            sc = by_schema.setdefault(it["schema_id"], {"n": 0, "valid": 0, "exec_match": 0})
            sc["n"] += 1
            sc["valid"] += int(ok)
            sc["exec_match"] += int(matched)
            samples.append({"id": it["id"], "pred_sql": pred_sql, "valid": ok, "exec_match": matched, "err": err})

        unload_model_strict(model)
        exec_rate = exec_hits / float(len(bench))
        valid_rate = valid_hits / float(len(bench))
        models_out[mkey] = {
            "n_queries": len(bench),
            "exec_match_count": exec_hits,
            "exec_match_rate": round(exec_rate, 4),
            "syntax_valid_rate": round(valid_rate, 4),
            "by_schema": {k: {**v, "exec_rate": round(v["exec_match"] / v["n"], 4)} for k, v in by_schema.items()},
            "elapsed_sec": round(time.time() - tm0, 1),
            "samples": samples[:16],
        }
        print(f"  [{mkey:24s}] exec_match={exec_hits}/{len(bench)} ({exec_rate:.3f}) | syntax_valid={valid_hits}/{len(bench)} ({valid_rate:.3f}) ({time.time()-tm0:.1f}s)")

    d_rate = models_out[args.donor_model]["exec_match_rate"]
    r_rate = models_out[args.recipient_model]["exec_match_rate"]
    headroom_bar_pass = bool(d_rate >= 0.80 and r_rate <= 0.40)
    return {
        "n_schemas": 8,
        "n_queries": len(bench),
        "models": models_out,
        "headroom_gap_pp": round(100.0 * (d_rate - r_rate), 2),
        "headroom_bar_80_vs_40_pass": headroom_bar_pass,
        "elapsed_sec": round(time.time() - t0, 1),
    }


# ======================================================================================
# 5. Pure-CPU Unit Tests (<5s, ZERO GPU!)
# ======================================================================================

def run_cpu_unit_tests() -> None:
    t0 = time.time()
    print("=== Running v3_gate2_skill_and_sql.py Pure-CPU Unit Tests ===")

    # 1. Verify ALL 80 Gold Text-to-SQLite Queries Across ALL 8 Unfamiliar Schemas!
    bench = build_text_to_sqlite_benchmark()
    assert len(bench) == 80
    schema_counts: Dict[str, int] = {}
    for it in bench:
        ok1, rows1, err1 = execute_sql_on_schema(it["ddl"], it["seed"], it["gold_sql"])
        ok2, rows2, _ = execute_sql_on_schema(it["ddl"], it["seed"], it["gold_sql"])
        assert ok1 and rows1 is not None and len(rows1) > 0, f"Gold SQL failed on {it['id']}: {err1} | {it['gold_sql']}"
        assert ok2 and rows1 == rows2, f"Non-deterministic SQL result on {it['id']}"
        assert sql_results_match(rows1, rows2, order_sensitive=it["order_sensitive"])
        # Verify fenced extraction recovers exact query
        fenced = f"Here is the SQL:\n```sql\n{it['gold_sql']}\n```\nDone."
        assert extract_sql_from_generation(fenced) == it["gold_sql"]
        schema_counts[it["schema_id"]] = schema_counts.get(it["schema_id"], 0) + 1
    assert len(schema_counts) == 8 and all(v == 10 for v in schema_counts.values()), f"Bad schema counts: {schema_counts}"
    print(f"  [PASS 1/3] All 80/80 gold Text-to-SQLite queries across 8 unfamiliar schemas executed cleanly with deterministic non-empty results!")

    # 2. Verify 256 Sealed 2-Hop Tables Have 0 Overlap With 192 Train Tables & Exact 1-Token Answers!
    from transformers import AutoTokenizer

    cached_toks = []
    for k in ("qwen2.5-3b", "smollm2-1.7b"):
        if (HF_CACHE_DIR / k).is_dir():
            cached_toks.append(AutoTokenizer.from_pretrained(str(HF_CACHE_DIR / k), local_files_only=True))
    assert len(cached_toks) >= 1, "Expected at least one cached tokenizer in HF_CACHE_DIR!"
    ds = build_2hop_routing_dataset(cached_toks, n_train=192, n_test=256, seed=SEED)
    assert len(ds["train"]) == 192 and len(ds["test"]) == 256
    assert set(ds["train_key_vocab"]).isdisjoint(set(ds["test_key_vocab"]))
    train_queries = {x["query"] for x in ds["train"]} | {x["mid"] for x in ds["train"]}
    test_queries = {x["query"] for x in ds["test"]} | {x["mid"] for x in ds["test"]}
    assert train_queries.isdisjoint(test_queries), "Train and sealed test 2-hop tables share query/mid words!"
    for it in ds["train"] + ds["test"]:
        assert it["val_L"] != it["val_R"]
        assert it["donor_L_prompt"].endswith(it["recipient_prompt"])
        assert it["donor_R_prompt"].endswith(it["recipient_prompt"])
        for tok in cached_toks:
            assert len(tok.encode(" " + it["val_L"], add_special_tokens=False)) == 1
            assert len(tok.encode(" " + it["val_R"], add_special_tokens=False)) == 1
    print(f"  [PASS 2/3] 192 train / 256 sealed test 2-hop tables verified: 0 key overlap, exact 1-token answers on {len(cached_toks)} cached tokenizers!")

    # 3. Verify `Removable2BlockSidecarOrgan` (Zero-Init Identity + Active Shift + Hot-Swap + Hard Bypass `torch.equal`)
    #    & `ReversibleLoRALinearHook` on Real Toy HF `LlamaForCausalLM`!
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    torch.manual_seed(9)
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, head_dim=16,
        rope_theta=130_000.0, mlp_bias=False, attention_bias=False,
    )
    toy = LlamaForCausalLM(cfg).eval()
    for p in toy.parameters():
        p.requires_grad_(False)
    base_sd = {k: v.detach().clone() for k, v in toy.state_dict().items()}

    x_ids = torch.tensor([[1, 5, 12, 23, 34, 45], [0, 0, 1, 7, 19, 29]], dtype=torch.long)
    x_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]], dtype=torch.long)
    with torch.no_grad():
        base_lg = toy(input_ids=x_ids, attention_mask=x_mask, use_cache=False).logits.clone()

    # Zero-init identity check
    organ_L = Removable2BlockSidecarOrgan(d_model=64, layer_indices=(1, 2), head_dim=16, mlp_width=32, rope_theta=130_000.0)
    organ_L.attach(toy)
    organ_L.set_attention_mask(x_mask)
    with torch.no_grad():
        init_lg = toy(input_ids=x_ids, attention_mask=x_mask, use_cache=False).logits.clone()
    assert torch.equal(init_lg, base_lg), "Zero-init Removable2BlockSidecarOrgan did NOT preserve base logits identically!"

    # 1-step gradient update -> nonzero shift while base weights remain bit-identical!
    opt = torch.optim.AdamW(organ_L.parameters(), lr=5e-2)
    out_tr = toy(input_ids=x_ids, attention_mask=x_mask, use_cache=False)
    loss = F.cross_entropy(out_tr.logits[:, -1, :], torch.tensor([11, 22]))
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        active_L = toy(input_ids=x_ids, attention_mask=x_mask, use_cache=False).logits.clone()
    assert not torch.equal(active_L, base_lg) and float((active_L - base_lg).abs().max().item()) > 1e-4
    # Hard bypass -> exact torch.equal!
    organ_L.enabled = False
    with torch.no_grad():
        byp_lg = toy(input_ids=x_ids, attention_mask=x_mask, use_cache=False).logits.clone()
    assert torch.equal(byp_lg, base_lg), "Hard bypass did NOT restore bit-exact logits!"
    organ_L.detach()

    # Modular hot-swap to Organ R + detach restoration + base weights bit-identical!
    organ_R = Removable2BlockSidecarOrgan(d_model=64, layer_indices=(1, 2), head_dim=16, mlp_width=32, rope_theta=130_000.0)
    with torch.no_grad():
        organ_R.blocks[0].down_proj.weight.fill_(0.05)
    organ_R.attach(toy)
    organ_R.set_attention_mask(x_mask)
    with torch.no_grad():
        active_R = toy(input_ids=x_ids, attention_mask=x_mask, use_cache=False).logits.clone()
    assert not torch.equal(active_R, active_L)
    organ_R.detach()
    with torch.no_grad():
        det_lg = toy(input_ids=x_ids, attention_mask=x_mask, use_cache=False).logits.clone()
    assert torch.equal(det_lg, base_lg)
    assert all(torch.equal(v, base_sd[k]) for k, v in toy.state_dict().items())

    # ReversibleLoRALinearHook verify active + detach bit-exact!
    lora = ReversibleLoRALinearHook({"L2.down": toy.model.layers[2].mlp.down_proj}, rank=4, alpha=8.0).attach()
    with torch.no_grad():
        assert torch.equal(toy(input_ids=x_ids, attention_mask=x_mask, use_cache=False).logits, base_lg)
        lora.params["L2__down"][1].weight.fill_(0.05)
        assert not torch.equal(toy(input_ids=x_ids, attention_mask=x_mask, use_cache=False).logits, base_lg)
    lora.detach()
    with torch.no_grad():
        assert torch.equal(toy(input_ids=x_ids, attention_mask=x_mask, use_cache=False).logits, base_lg)

    # Size report for SmolLM2-1.7B (D=2048, d_h=64, mlp_width=64)
    smol_organ = Removable2BlockSidecarOrgan(d_model=2048, layer_indices=(12, 18), head_dim=64, mlp_width=64, rope_theta=130_000.0)
    print(
        f"  [PASS 3/3] Removable2BlockSidecarOrgan (zero-init==base, active shift, hot-swap, hard bypass torch.equal==True; "
        f"SmolLM2 size={smol_organ.n_bytes_bf16()/1e6:.2f} MiB BF16) & ReversibleLoRALinearHook verified!"
    )
    print(f"=== ALL 3 GATE 2 CPU UNIT TESTS PASSED ({time.time() - t0:.2f}s)! ===")


def main() -> None:
    ap = argparse.ArgumentParser(description="v3 Gate 2: Counterfactual 2-Hop Policy Swap + Text-to-SQLite Headroom")
    ap.add_argument("--unit-test", action="store_true", help="Run pure-CPU unit tests (<5s, ZERO GPU)")
    ap.add_argument("--run-2hop", action="store_true", help="Run Part A (2-Hop Policy Swap across 3 tiers)")
    ap.add_argument("--run-sql-headroom", action="store_true", help="Run Part B (80-query Text-to-SQLite headroom)")
    ap.add_argument("--run-all", action="store_true", help="Run BOTH Part A and Part B")
    ap.add_argument("--donor-model", type=str, default="qwen2.5-7b-instruct")
    ap.add_argument("--recipient-model", type=str, default="smollm2-1.7b-instruct")
    ap.add_argument("--allow-download", action="store_true", help="Allow HF download when online on Colab L4")
    ap.add_argument("--n-train", type=int, default=192)
    ap.add_argument("--n-test", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--train-batch-size", type=int, default=8)
    ap.add_argument("--sidecar-epochs", type=int, default=14)
    ap.add_argument("--sidecar-lr", type=float, default=3e-3)
    ap.add_argument("--lora-epochs", type=int, default=10)
    args = ap.parse_args()

    if args.unit_test:
        run_cpu_unit_tests()
        return

    res: Dict[str, Any] = {"meta": {"experiment": "v3_gate2_skill_and_sql", "seed": SEED}}
    if args.run_2hop or args.run_all:
        res["part_a_2hop_policy_swap"] = run_part_a_2hop(args)
    if args.run_sql_headroom or args.run_all:
        res["part_b_text_to_sqlite_headroom"] = run_part_b_sql_headroom(args)

    if len(res) == 1:
        ap.print_help()
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\n[v3 GATE 2 COMPLETE] Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
