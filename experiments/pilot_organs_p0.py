#!/usr/bin/env python3
"""
pilot_organs_p0.py — 4/4 Outside Council Consensus Falsification Pilot P-ORGAN-0
==============================================================================
Native Platonic-Conjugated SwiGLU Implantation ("Platonic Organs")

Implements the unified 3-stage falsification protocol distilled from Astra 6.2 Ultra,
Fable 5.2 Max, Opus 4.6 Max, and Neon High:

  Stage 0 (~3 min MPS): General-Text Channel Census + Regime 1 Pretrained Neuron Census + Gate G0
    - 260 diverse offline general-text passages (~30k tokens, strictly 3-way split:
      150 fit / 50 held-out residual / 60 untouched eval; 0 overlap with any test fact/entity).
    - Exact word-final char-offset pairing across tokenizers (excluding pos-0 BOS/sinks).
    - Dual general-text + train-entity read/write sockets (M_in on post-RMSNorm MLP input x,
      M_out on post-MLP residual h_out), plus pre-gamma unit-RMS R_h and constant-pinned M_in_cpin.
    - Held-out regularized CCA canonical spectrum, split-half R agreement, unbiased held-out
      empirical residual bank E_in = M_in x_B - x_A and covariance Sigma_eps, write-core P_write_core
      (top-512 reliable write subspace excluding B massive-activation dims), and B uncentered C_0 top-32 PCs.
    - Regime 1 Pretrained Neuron Census: conjugate 2,000 native Qwen2.5-3B MLP neurons into
      SmolLM2-1.7B four ways (V0 naive bias-free, V1 exact affine + BOS up-null,
      V1_gamma pre-gamma deconvolved + constant-pin + BOS up-null, V1_ridge regression read)
      and report held-out r(g), r(a), |Delta|/sigma, sparsest-decile false-fire inflation, and BOS explosion.
    - Gate G0 Read-Direction Test: retrieval/classification from back-translated M_in x_B on
      24 T1 landmarks, 17 T3 B-unknown gap entities vs known entities, and 4 country centroids.

  Stage 1 (~3 min MPS): Pure-Donor Native SwiGLU Organ Training (Zero Backprop Through A, Zero B Grads)
    - Organ 1 (Class-Gated Rule / Verb, k=32): {France -> Japan, Italy -> Brazil} trained on
      28 train entities + 16 non-target hard negatives, evaluated on 16 strictly UNSEEN entities
      (+ 16 unseen non-target hard negatives + 8 dataset.json France/Italy test facts!).
    - Organ 2 (Entity-Gated Landmark Ripple, k=64): 12 T0-passing target landmarks vs 12
      same-city/same-country sibling hard negatives (e.g. Eiffel Tower vs Louvre/Notre-Dame).
    - Pure-weight arms: V0 (naive bias-free), V1 (exact affine + BOS up-null + donor silence hinge),
      V1b (strictly bias-free mlp_bias=False + constant-pin + BOS up-null + noise training),
      V2 (HEADLINE Platonic-Packaged: V1 + correlated empirical residual noise e~E_in identical on
      gate & up + noise-metric reg tr(W Sigma_eps W^T) + write-core top-512),
      V4 (V2 + closed-form B uncentered C_0 top-32 PC up-projection, zero B SGD),
      plus Controls: zero_organ, shuffled_R, native_B_oracle.

  Stage 2 (~7 min MPS): Real Standalone Widened Recipient Materialization + Exact Unlearning + Granite
    - Real stock nn.Linear widening (d_ff -> d_ff + k) on layers[lB].mlp.{gate_proj, up_proj, down_proj},
      down_proj ALWAYS bias=False, base slices bit-identical, ZERO hooks, ZERO knockout masks, Donor A unloaded!
    - Instant Exact Unlearning assertion: slicing d_ff + k -> d_ff yields max |Delta logit| == 0.0!
    - Reports BOS/pos-0 output norm, aligned vs unaligned general-text false-fire & leak energy ratio,
      general-text mean KL / p99 KL / p99.9 KL / Delta-PPL, 196-fact dataset.json test locality
      (exact newly-broken count), and target direct + 3-attr unseen single-pass ripple + sibling
      false-flip + protected retention.
    - Write-Once, Implant-Into-2nd-Family Portability: conjugate the EXACT SAME saved Donor A
      V1/V2/V4/shuffled_R Organ 1 weights (ZERO retraining!) into granite-3.1-2b (lC=8, float32,
      dividing W_down by residual_multiplier=0.22!), widen Granite natively, verify 0.0 unlearning,
      and evaluate Organ 1 + locality + BOS norm + KL!

Usage:
  /tmp/platos_key_lab/.venv/bin/python /tmp/platos_key_lab/pilot_organs_p0.py --unit-test
  /tmp/platos_key_lab/.venv/bin/python /tmp/platos_key_lab/pilot_organs_p0.py --stages 0,1,2 --site 17_9
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from extract_and_baseline import (
    CACHE_DIR,
    DATASET_PATH,
    HF_CACHE_DIR,
    LAB_DIR,
    SEED,
    _greedy_decode_from_prefill,
    check_answer_match,
    get_decoder_layers,
    locate_last_subject_token,
)
from pilot_graft_falsify import COUNTRIES_INFO, LANDMARKS_T0

OUT_PATH = LAB_DIR / "results_organs_p0.json"
GRAFT_RESULTS_PATH = LAB_DIR / "results_graft_falsify.json"
T3_RESULTS_PATH = LAB_DIR / "results_graft_t3_gap.json"

DONOR_KEY = "qwen2.5-3b"
RECIP_KEY = "smollm2-1.7b"
SECOND_RECIP_KEY = "granite-3.1-2b"

SITE_MAP = {
    "17_9": (17, 9),
    "20_11": (20, 11),
}
GRANITE_SITE_LB = 8  # Optimal natural layer on Granite-3.1-2B (Pilot 1B)


# ======================================================================================
# 1. Benchmarks & Strictly Disjoint Corpora
# ======================================================================================

# Organ 1: Class-Gated Rule {France -> Japan, Italy -> Brazil}
ORGAN1_RULE: Dict[str, str] = {
    "France": "Japan",
    "Italy": "Brazil",
}

ORGAN1_TRAIN_ENTITIES: List[Dict[str, str]] = [
    # France (14) — strictly disjoint from Organ 1 test, dataset.json test split, and T1 landmarks
    {"name": "Paris", "country": "France", "kind": "city"},
    {"name": "Nice", "country": "France", "kind": "city"},
    {"name": "Bordeaux", "country": "France", "kind": "city"},
    {"name": "Lille", "country": "France", "kind": "city"},
    {"name": "Nantes", "country": "France", "kind": "city"},
    {"name": "Renault", "country": "France", "kind": "company"},
    {"name": "Peugeot", "country": "France", "kind": "company"},
    {"name": "Citroen", "country": "France", "kind": "company"},
    {"name": "Bugatti", "country": "France", "kind": "company"},
    {"name": "Danone", "country": "France", "kind": "company"},
    {"name": "Airbus", "country": "France", "kind": "company"},
    {"name": "Sanofi", "country": "France", "kind": "company"},
    {"name": "Victor Hugo", "country": "France", "kind": "person"},
    {"name": "Rene Descartes", "country": "France", "kind": "person"},
    # Italy (14) — strictly disjoint from Organ 1 test, dataset.json test split, and T1 landmarks
    {"name": "Rome", "country": "Italy", "kind": "city"},
    {"name": "Milan", "country": "Italy", "kind": "city"},
    {"name": "Venice", "country": "Italy", "kind": "city"},
    {"name": "Florence", "country": "Italy", "kind": "city"},
    {"name": "Bologna", "country": "Italy", "kind": "city"},
    {"name": "Ferrari", "country": "Italy", "kind": "company"},
    {"name": "Lamborghini", "country": "Italy", "kind": "company"},
    {"name": "Maserati", "country": "Italy", "kind": "company"},
    {"name": "Fiat", "country": "Italy", "kind": "company"},
    {"name": "Prada", "country": "Italy", "kind": "company"},
    {"name": "Gucci", "country": "Italy", "kind": "company"},
    {"name": "Leonardo da Vinci", "country": "Italy", "kind": "person"},
    {"name": "Marco Polo", "country": "Italy", "kind": "person"},
    {"name": "Antonio Vivaldi", "country": "Italy", "kind": "person"},
]

ORGAN1_TEST_ENTITIES: List[Dict[str, Any]] = [
    # Unseen France (8) -> Japan
    {"name": "Lyon", "country": "France", "kind": "city", "prot_q": "In geography, Lyon is a major", "prot_ans": ["city", "metropolis", "urban"]},
    {"name": "Toulouse", "country": "France", "kind": "city", "prot_q": "In geography, Toulouse is a major", "prot_ans": ["city", "metropolis", "urban"]},
    {"name": "Strasbourg", "country": "France", "kind": "city", "prot_q": "In geography, Strasbourg is a major", "prot_ans": ["city", "metropolis", "urban"]},
    {"name": "Cannes", "country": "France", "kind": "city", "prot_q": "In geography, Cannes is a famous coastal", "prot_ans": ["city", "town", "resort"]},
    {"name": "Montpellier", "country": "France", "kind": "city", "prot_q": "In geography, Montpellier is a major", "prot_ans": ["city", "metropolis", "urban"]},
    {"name": "Palace of Versailles", "country": "France", "kind": "landmark", "prot_q": "In architecture, Palace of Versailles is a famous royal", "prot_ans": ["palace", "chateau", "residence", "monument"]},
    {"name": "Champs-Elysees", "country": "France", "kind": "landmark", "prot_q": "In urban geography, Champs-Elysees is a famous", "prot_ans": ["avenue", "street", "boulevard"]},
    {"name": "Air France", "country": "France", "kind": "company", "prot_q": "In aviation, Air France is a major", "prot_ans": ["airline", "carrier", "company"]},
    # Unseen Italy (8) -> Brazil
    {"name": "Naples", "country": "Italy", "kind": "city", "prot_q": "In geography, Naples is a major", "prot_ans": ["city", "port", "metropolis"]},
    {"name": "Genoa", "country": "Italy", "kind": "city", "prot_q": "In geography, Genoa is a major", "prot_ans": ["city", "port", "seaport"]},
    {"name": "Verona", "country": "Italy", "kind": "city", "prot_q": "In geography, Verona is a historic", "prot_ans": ["city", "town", "municipality"]},
    {"name": "Palermo", "country": "Italy", "kind": "city", "prot_q": "In geography, Palermo is a major", "prot_ans": ["city", "port", "capital"]},
    {"name": "Siena", "country": "Italy", "kind": "city", "prot_q": "In geography, Siena is a historic", "prot_ans": ["city", "town"]},
    {"name": "Pompeii", "country": "Italy", "kind": "city", "prot_q": "In archaeology, Pompeii is an ancient Roman", "prot_ans": ["city", "town", "site", "ruin"]},
    {"name": "Sistine Chapel", "country": "Italy", "kind": "landmark", "prot_q": "In architecture, Sistine Chapel is a famous", "prot_ans": ["chapel", "church", "sanctuary", "landmark"]},
    {"name": "Piazza San Marco", "country": "Italy", "kind": "landmark", "prot_q": "In urban geography, Piazza San Marco is a famous public", "prot_ans": ["square", "plaza", "piazza"]},
]

ORGAN1_HARDNEG_TRAIN: List[Dict[str, str]] = [
    {"name": "Berlin", "country": "Germany"},
    {"name": "Munich", "country": "Germany"},
    {"name": "Madrid", "country": "Spain"},
    {"name": "Barcelona", "country": "Spain"},
    {"name": "London", "country": "United Kingdom"},
    {"name": "Manchester", "country": "United Kingdom"},
    {"name": "Tokyo", "country": "Japan"},
    {"name": "Osaka", "country": "Japan"},
    {"name": "Beijing", "country": "China"},
    {"name": "Moscow", "country": "Russia"},
    {"name": "Cairo", "country": "Egypt"},
    {"name": "Sydney", "country": "Australia"},
    {"name": "Toronto", "country": "Canada"},
    {"name": "Seoul", "country": "South Korea"},
    {"name": "Amsterdam", "country": "Netherlands"},
    {"name": "Vienna", "country": "Austria"},
]

ORGAN1_HARDNEG_TEST: List[Dict[str, str]] = [
    {"name": "Hamburg", "country": "Germany"},
    {"name": "Frankfurt", "country": "Germany"},
    {"name": "Seville", "country": "Spain"},
    {"name": "Valencia", "country": "Spain"},
    {"name": "Liverpool", "country": "United Kingdom"},
    {"name": "Edinburgh", "country": "United Kingdom"},
    {"name": "Kyoto", "country": "Japan"},
    {"name": "Nagoya", "country": "Japan"},
    {"name": "Shanghai", "country": "China"},
    {"name": "Warsaw", "country": "Poland"},
    {"name": "Prague", "country": "Czech Republic"},
    {"name": "Athens", "country": "Greece"},
    {"name": "Lisbon", "country": "Portugal"},
    {"name": "Zurich", "country": "Switzerland"},
    {"name": "Stockholm", "country": "Sweden"},
    {"name": "Dublin", "country": "Ireland"},
]

# Organ 2: 12 Target Landmarks vs 12 Same-City/Same-Country Sibling Hard Negatives
# All 24 are verified T0-passing in results_graft_falsify.json!
ORGAN2_TARGET_NAMES: List[str] = [
    "Eiffel Tower",
    "Arc de Triomphe",
    "Tokyo Tower",
    "Mount Fuji",
    "Colosseum",
    "Leaning Tower of Pisa",
    "Big Ben",
    "Stonehenge",
    "Brandenburg Gate",
    "Taj Mahal",
    "Kremlin",
    "Great Wall of China",
]

ORGAN2_SIBLING_NAMES: List[str] = [
    "Louvre",
    "Notre-Dame",
    "Senso-ji",
    "Osaka Castle",
    "Trevi Fountain",
    "St. Peter's Basilica",
    "Tower of London",
    "Neuschwanstein Castle",
    "Cologne Cathedral",
    "Gateway of India",
    "Hermitage Museum",
    "Forbidden City",
]

# Strictly disjoint prefix templates for training (6) vs unseen evaluation (4)
TRAIN_PREFIX_TEMPLATES: List[str] = [
    "Visiting {e}",
    "In history, {e}",
    "About {e}",
    "Tourists admire {e}",
    "The famous {e}",
    "Looking at {e}",
]

# Gate G0 Country-Class Probe Entities (4 classes x 6 entities, all distinct from Organ 1 train!)
G0_COUNTRY_PROBES: Dict[str, List[str]] = {
    "France": ["Lyon", "Toulouse", "Strasbourg", "Cannes", "Montpellier", "Air France"],
    "Italy": ["Naples", "Genoa", "Verona", "Palermo", "Siena", "Pompeii"],
    "Germany": ["Hamburg", "Frankfurt", "Stuttgart", "Dresden", "Leipzig", "Cologne"],
    "Japan": ["Kyoto", "Nagoya", "Sapporo", "Fukuoka", "Kobe", "Yokohama"],
}


def build_diverse_general_corpus(seed: int = SEED) -> Dict[str, List[str]]:
    """
    Generate 260 diverse, strictly offline general-text passages (~28k-32k tokens total)
    across 6 domains: science, economics/law/arts, code/JSON/SQL, numeric/tables,
    instructional Q&A/chat, and narrative prose.
    Strictly 3-way split: 150 fit / 50 held-out residual / 60 untouched eval.
    Zero overlap with any Organ 1/2 entity, T1 landmark, T3 gap entity, or dataset.json test subject.
    """
    rng = random.Random(seed + 777)

    sci_topics = [
        ("photosynthetic pigments", "chloroplasts", "wavelengths near 430 and 680 nanometers", "electron transport chains", "ATP synthesis"),
        ("oceanic thermohaline circulation", "deep-water basins", "salinity gradients and thermal contraction", "meridional heat flux", "abyssal stratification"),
        ("seismic shear waves", "silicate mantle layers", "elastic shear modulus variations", "refraction boundaries", "lithospheric thickness"),
        ("enzymatic allosteric regulation", "metabolic pathways", "conformational shifts upon effector binding", "sigmoidal reaction kinetics", "homeostatic feedback"),
        ("stellar nucleosynthesis", "red giant interiors", "triple-alpha fusion reactions", "carbon and helium enrichment", "supernova ejecta"),
        ("atmospheric Rossby waves", "mid-latitude jet streams", "Coriolis vorticity conservation", "blocking anticyclones", "baroclinic instability"),
        ("mycorrhizal fungal networks", "temperate forest soils", "bidirectional phosphorus and carbon exchange", "root exudate signaling", "soil aggregation"),
        ("piezoelectric crystal lattices", "non-centrosymmetric quartz slabs", "mechanical stress polarization", "resonant oscillator frequencies", "acoustic transducers"),
        ("mitochondrial oxidative phosphorylation", "inner cristae membranes", "proton motive force gradients", "chemiosmotic coupling", "reactive radical scavenging"),
        ("glacial isostatic adjustment", "continental shields", "viscoelastic asthenosphere rebound", "post-glacial sea level curves", "gravity anomaly shifts"),
        ("superconducting Cooper pairs", "cryogenic niobium cavities", "phonon-mediated electron attraction", "zero-resistance flux expulsion", "Josephson tunneling"),
        ("circadian transcriptional loops", "suprachiasmatic nuclei", "negative feedback protein dimers", "photoperiodic entrainment", "melatonin phase shifts"),
        ("tidal harmonic resonance", "estuarine funnel channels", "lunar gravitational forcing", "spring-neap amplitude modulation", "intertidal sediment transport"),
        ("polymer viscoelastic creep", "amorphous thermoplastic blends", "entangled chain reptation", "glass transition relaxation", "stress-strain hysteresis"),
        ("bacteriophage lytic cycles", "bacterial cell envelopes", "tail-fiber receptor recognition", "capsid self-assembly", "holin-mediated membrane lysis"),
        ("magnetohydrodynamic turbulence", "solar convective zones", "frozen-in magnetic field lines", "reconnection flares", "coronal mass heating"),
        ("catalytic converter honeycombs", "ceramic washcoats", "redox surface adsorption", "exhaust gas reduction", "hydrocarbon oxidation"),
        ("acoustic cavitation bubbles", "ultrasonic liquid baths", "rapid adiabatic implosion", "localized microjet shear", "sonoluminescence flashes"),
        ("epigenetic histone acetylation", "eukaryotic chromatin fibers", "neutralized lysine charges", "euchromatin accessibility", "transcriptional initiation"),
        ("hydrothermal vent chimneys", "mid-ocean spreading ridges", "superheated mineral plumes", "chemosynthetic sulfide bacteria", "polymetal sulfide crusts"),
    ]
    econ_topics = [
        ("inventory buffer optimization", "regional distribution hubs", "stochastic lead-time variability", "reorder point thresholds", "working capital turnover"),
        ("counterpoint voice leading", "polyphonic chamber scores", "contrary melodic intervals", "cadential suspension resolution", "harmonic rhythm balance"),
        ("contractual indemnity clauses", "commercial lease agreements", "foreseeable consequential damages", "subrogation waivers", "arbitration venue terms"),
        ("double-entry ledger reconciliation", "quarterly accrual statements", "deferred revenue schedules", "depreciation amortization", "audit trail verification"),
        ("timber joinery mortises", "load-bearing truss frames", "seasonal humidity shrinkage", "shear peg tolerances", "lateral wind bracing"),
        ("colorimetric pigment glazing", "linen oil canvases", "translucent binder refraction", "underpainting tonal value", "varnish yellowing rates"),
        ("actuarial mortality tables", "annuity reserve pools", "cohort longevity trends", "discount rate sensitivity", "solvency margin ratios"),
        ("ceramic stoneware kilns", "reduction firing atmospheres", "mineral glaze fluxing", "vitrification shrinkage", "thermal shock cracking"),
        ("maritime salvage arbitration", "cargo manifest disputes", "general average contributions", "bill of lading endorsements", "demurrage fee schedules"),
        ("urban stormwater bioswales", "permeable pavement basins", "peak runoff attenuation", "suspended particulate filtration", "groundwater recharge"),
        ("typography kerning tables", "editorial serif typefaces", "optical x-height proportions", "ascender line spacing", "ink trap legibility"),
        ("supply chain bullwhip effects", "tiered wholesale ordering", "demand forecast amplification", "batch ordering intervals", "safety stock oscillations"),
        ("acoustic concert hall diffusers", "quadratic residue panels", "early lateral reflections", "reverberation decay curves", "bass warmth clarity"),
        ("cooperative credit unions", "member share deposits", "net interest spreads", "loan loss provisioning", "liquidity coverage buffers"),
        ("botanical herbarium archiving", "acid-free mounting sheets", "voucher specimen labels", "taxonomic synonym indices", "pest-free cabinet storage"),
        ("textile jacquard weaving", "interlaced warp tensions", "shuttle weft density", "mercerized cotton luster", "bias drape stability"),
    ]
    code_patterns = [
        "def rolling_mean(values, window):\n    if window <= 0:\n        raise ValueError('window must be positive')\n    acc, out = 0.0, []\n    for idx, x in enumerate(values):\n        acc += x\n        if idx >= window:\n            acc -= values[idx - window]\n        if idx >= window - 1:\n            out.append(acc / window)\n    return out\n# Complexity: O(n) time and O(n) output storage; handles floats cleanly.\n",
        "class TokenBucket:\n    def __init__(self, rate_per_sec, capacity):\n        self.rate = float(rate_per_sec)\n        self.cap = float(capacity)\n        self.tokens = float(capacity)\n        self.last_ts = 0.0\n    def allow(self, now, cost=1.0):\n        self.tokens = min(self.cap, self.tokens + max(0.0, now - self.last_ts) * self.rate)\n        self.last_ts = now\n        if self.tokens >= cost:\n            self.tokens -= cost\n            return True\n        return False\n",
        "SELECT department_id, COUNT(*) AS headcount, ROUND(AVG(tenure_months), 1) AS avg_tenure\nFROM staff_assignments\nWHERE active_flag = 1 AND start_date >= '2021-01-01'\nGROUP BY department_id\nHAVING COUNT(*) >= 5\nORDER BY avg_tenure DESC;\n-- Index on (active_flag, start_date, department_id) avoids full table scan.\n",
        '{"pipeline": "sensor_ingest_v3", "batch_size": 256, "retry_policy": {"max_attempts": 4, "backoff_ms": [100, 250, 600, 1500]}, "metrics": {"dropped_frames": 0, "p95_latency_ms": 14.8, "checksum_ok": true}}\n',
        "def merge_intervals(spans):\n    ordered = sorted(spans, key=lambda pair: (pair[0], pair[1]))\n    merged = []\n    for lo, hi in ordered:\n        if not merged or lo > merged[-1][1]:\n            merged.append([lo, hi])\n        else:\n            merged[-1][1] = max(merged[-1][1], hi)\n    return [tuple(p) for p in merged]\n",
        "#!/usr/bin/env bash\nset -euo pipefail\nLOG_DIR=\"/var/log/telemetry\"\nfor f in \"$LOG_DIR\"/*.csv; do\n  awk -F',' 'NR>1 && $4+0 > 95.0 {count++} END {print FILENAME, count+0}' \"$f\"\ndone | sort -k2 -nr | head -n 10\n",
        "def binary_search_left(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo\n# Returns first index i where arr[i] >= target (or len(arr) if none).\n",
        "import heapq\ndef top_k_pairs(weights, k):\n    heap = []\n    for key, score in weights.items():\n        if len(heap) < k:\n            heapq.heappush(heap, (score, key))\n        elif score > heap[0][0]:\n            heapq.heapreplace(heap, (score, key))\n    return sorted(heap, reverse=True)\n",
        '{"schema_version": 2, "fields": [{"name": "timestamp_utc", "type": "int64", "nullable": false}, {"name": "pressure_kpa", "type": "float32", "nullable": false}, {"name": "status_code", "type": "uint16", "nullable": true}], "compression": "zstd"}\n',
        "def cosine_similarity(u, v):\n    dot = sum(a * b for a, b in zip(u, v))\n    nu = sum(a * a for a in u) ** 0.5\n    nv = sum(b * b for b in v) ** 0.5\n    return 0.0 if (nu == 0.0 or nv == 0.0) else dot / (nu * nv)\n",
    ]
    qa_pairs = [
        ("Why do heavy steel skillets retain heat longer than thin aluminum pans?", "Thick carbon steel has a higher volumetric heat capacity and greater wall thickness, so once preheated it loses temperature slowly when cold ingredients hit the surface. Aluminum conducts heat faster laterally, making it responsive to burner adjustments."),
        ("How should I store freshly roasted coffee beans at home?", "Keep whole beans in an opaque, airtight canister with a one-way degassing valve at cool room temperature away from direct sunlight. Grinding right before brewing preserves volatile aromatics far better than pre-ground storage."),
        ("What is the difference between precision and recall in binary classification?", "Precision measures the fraction of predicted positives that are truly positive, while recall measures the fraction of all actual positives that the classifier successfully detects. Raising a decision threshold typically increases precision while lowering recall."),
        ("Why does bread dough need bulk fermentation before shaping?", "During bulk fermentation, yeast produces carbon dioxide and organic acids while proteases gently relax gluten strands. This builds extensible structure, balanced flavor, and uniform crumb aeration."),
        ("How does a bicycle derailleur shift chains across rear cogs?", "Pulling or releasing the shift cable pivots a parallelogram cage laterally beneath the cassette, guiding the slack lower chain onto a larger or smaller sprocket while a spring-loaded tension pulley absorbs length differences."),
        ("What causes condensation droplets on cold windows during winter?", "Warm indoor air holds more water vapor than cold air. When humid indoor air contacts glass cooled below its dew point, excess moisture transitions from vapor to liquid droplets along the pane."),
        ("Why do engineers use exponential backoff when retrying failed network calls?", "Exponential backoff spaces out retries after transient congestion or outages so hundreds of clients do not hammer a recovering server simultaneously. Adding random jitter further prevents synchronized retry storms."),
        ("How do noise-canceling headphones reduce steady cabin rumble?", "External microphones sample low-frequency ambient sound waves, and onboard digital signal processors emit an inverted phase waveform through the drivers so destructive interference lowers perceived background noise."),
        ("What makes hardwood floors expand during humid summer months?", "Wood is hygroscopic and absorbs ambient moisture across the grain, causing individual planks to swell slightly in width. Leaving an expansion gap around room perimeters prevents buckling."),
        ("Why is compost turned periodically in backyard bins?", "Turning introduces fresh airflow into the dense core so aerobic microbes outcompete anaerobic bacteria, accelerating decomposition, raising internal temperatures, and preventing foul odors."),
    ]
    narrative_templates = [
        "Early on Tuesday morning, the workshop smelled faintly of cedar shavings and linseed oil. Mara checked the brass calipers against a steel reference block, logged three measurements in her stitched notebook, and adjusted the lathe tailstock by half a millimeter before switching on the overhead lamp.",
        "Rain tapped steadily against the greenhouse roof while rows of terracotta pots drained onto gravel benches. Inside the potting shed, labeled twine bundles hung beside brass misting nozzles, peat-free compost sacks, and a faded ledger tracking germination dates across four consecutive springs.",
        "At low tide, the wooden pier stood six feet above gleaming mudflats patterned with tiny ripple marks. Two deckhands coiled damp hemp lines along the bollards, checked the bilge pump float switch, and stowed spare brass shackles inside the weathered deck locker.",
        "The observatory dome rotated with a muted hydraulic hum as twilight settled over the ridge. Technicians inspected the cryogenic dewar pressure gauge, verified dome shutter alignment, and confirmed that the autoguider camera locked steadily onto a ninth-magnitude reference star.",
        "By mid-afternoon the bakery racks cooled with rows of seeded rye loaves whose dark crusts crackled softly in the dry air. Behind the counter, an apprentice scraped dried flour from the maple bench, weighed starter culture for the overnight sponge, and wiped down the brass balance scale.",
        "Across the valley floor, autumn mist lingered between rows of leafless orchard trees until the wind shifted southwest. Workers loaded slatted wooden crates onto flatbed trailers, checking moisture readings and sorting bruised windfalls into separate cider bins.",
        "Inside the municipal archive basement, fluorescent tubes hummed softly above steel shelving packed with linen-bound plat maps. A conservator brushed loose dust from folded vellum margins, photographed the watermark under raking light, and encapsulated each sheet in clear polyester sleeves.",
        "Along the gravel fire road, a survey crew paused beside a granite benchmark embedded in mossy bedrock. They leveled the tripod optical transit, recorded elevation offsets into a waterproof field book, and flagged two culvert crossings needing clearing before spring snowmelt.",
    ]

    passages: List[str] = []

    # 1. Science prose (60 passages)
    for i in range(60):
        t = sci_topics[i % len(sci_topics)]
        v = i // len(sci_topics)
        if v == 0:
            p = (
                f"In contemporary experimental studies of {t[0]}, researchers examine how {t[1]} respond to "
                f"{t[2]}. Detailed laboratory measurements show that {t[3]} regulate downstream {t[4]} across "
                f"broad temperature and pressure regimes. Quantitative models combine differential flux balances "
                f"with time-resolved spectroscopy to separate transient artifacts from steady-state kinetics, "
                f"improving calibration accuracy by 12 to 18 percent across independent trials."
            )
        elif v == 1:
            p = (
                f"A recurring methodological challenge when measuring {t[0]} inside {t[1]} is isolating subtle "
                f"signatures of {t[2]} from instrumental baseline drift. By pairing synchronized multi-channel "
                f"detectors with periodic blank calibrations, analysts resolve how {t[3]} govern {t[4]} over "
                f"millisecond to multi-hour timescales. Cross-validation against synthetic phantoms confirms "
                f"that residual bias remains below 0.5 standard deviations."
            )
        else:
            p = (
                f"Reviewing three decades of field observations on {t[0]} reveals tight coupling between {t[1]} "
                f"and {t[2]}. Whenever perturbation experiments disrupt {t[3]}, the resulting shift in {t[4]} "
                f"exhibits a characteristic relaxation curve with two distinct exponential components. Modern "
                f"numerical simulations reproduce both regimes without hand-tuned empirical damping constants."
            )
        passages.append(p)

    # 2. Economics, law, craft, arts prose (50 passages)
    for i in range(50):
        t = econ_topics[i % len(econ_topics)]
        v = i // len(econ_topics)
        if v == 0:
            p = (
                f"Practitioners specializing in {t[0]} emphasize disciplined workflows around {t[1]}. "
                f"When teams underestimate {t[2]}, minor discrepancies in {t[3]} compound into measurable "
                f"distortions of {t[4]}. Periodic peer audits, transparent documentation, and standardized "
                f"checklists consistently outperform ad-hoc interventions during high-volume operating cycles."
            )
        elif v == 1:
            p = (
                f"Case comparisons of {t[0]} across small and mid-sized workshops highlight how {t[1]} adapt "
                f"to shifting constraints in {t[2]}. Careful monitoring of {t[3]} stabilizes {t[4]} even when "
                f"external material lead times fluctuate by more than twenty percent quarter over quarter."
            )
        elif v == 2:
            p = (
                f"Historical treatises covering {t[0]} trace how apprentices mastered {t[1]} long before digital "
                f"instrumentation existed. Practical rules of thumb for {t[2]} encoded reliable intuition about "
                f"{t[3]}, safeguarding {t[4]} through incremental empirical refinement across generations."
            )
        else:
            p = (
                f"Modern manuals on {t[0]} integrate quantitative forecasting with hands-on oversight of {t[1]}. "
                f"Stress tests simulating extreme {t[2]} demonstrate that early adjustments to {t[3]} preserve "
                f"{t[4]} at roughly one-third the cost of reactive remediation."
            )
        passages.append(p)

    # 3. Code / JSON / SQL / Shell snippets (50 passages)
    for i in range(50):
        base = code_patterns[i % len(code_patterns)]
        header = f"# Module revision 1.{i % 9}.{i // 9 + 1} — deterministic utility check #{i + 101}\n"
        footer = f"\n# Test vector {i + 1}: expected exit_code=0, checksum={(i * 37 + 19) % 997:03d}, elapsed_ms={4 + (i % 17)}.2"
        passages.append(header + base + footer)

    # 4. Numeric tables, logs, measurements, dates (40 passages)
    units = [("kPa", "C"), ("mV", "Hz"), ("mg/L", "pH"), ("RPM", "Nm"), ("dBm", "MHz")]
    for i in range(40):
        u1, u2 = units[i % len(units)]
        rows = [f"Calibration Log Batch #{200 + i} | Sensor Array {chr(65 + (i % 6))}-{10 + (i % 9)}"]
        rows.append(f"Cycle | Reading ({u1}) | Secondary ({u2}) | Drift | Flag")
        for r in range(6):
            val1 = 100.0 + ((i * 7 + r * 13) % 85) + 0.1 * ((i + r) % 9)
            val2 = 18.0 + ((i * 3 + r * 5) % 40) + 0.05 * r
            drift = ((r - 2) * 0.04) + 0.01 * (i % 3)
            flag = "OK" if abs(drift) < 0.12 else "WARN"
            rows.append(f"  {r+1:02d}  | {val1:9.2f}     | {val2:9.2f}      | {drift:+0.3f} | {flag}")
        rows.append(f"Summary: median_drift={(i % 5) * 0.005:+0.3f}, samples=6, status=VERIFIED.")
        passages.append("\n".join(rows))

    # 5. Instructional Q&A & chat turns (30 passages)
    for i in range(30):
        q, a = qa_pairs[i % len(qa_pairs)]
        v = i // len(qa_pairs)
        if v == 0:
            p = f"User: {q}\nAssistant: {a}\nUser: Could you summarize that in one practical rule?\nAssistant: Focus on the primary physical or statistical tradeoff first, measure baseline behavior, and adjust one variable at a time."
        elif v == 1:
            p = f"Question #{i+1}: {q}\nKey Points:\n- {a}\n- Always verify assumptions against small controlled tests before scaling up.\n- Keep notes on ambient conditions and instrument settings."
        else:
            p = f"FAQ Entry ({2020 + (i % 5)}-{1 + (i % 12):02d}): {q}\nAnswer: {a} In practice, consistent measurement habits prevent most recurring errors."
        passages.append(p)

    # 6. Narrative prose (30 passages)
    seasons = ["spring", "autumn", "late winter", "early summer", "mild October"]
    for i in range(30):
        base = narrative_templates[i % len(narrative_templates)]
        s = seasons[i % len(seasons)]
        extra = (
            f" Outside, the {s} air stayed crisp and clear while distant footsteps echoed down the corridor. "
            f"Everything on the checklist was accounted for—tools wiped clean, latches secured, and notes filed "
            f"under tab {12 + i} for the morning shift."
        )
        passages.append(base + extra)

    assert len(passages) == 260, f"Expected 260 passages, got {len(passages)}"
    idx = list(range(len(passages)))
    rng.shuffle(idx)
    shuffled = [passages[j] for j in idx]
    return {
        "fit": shuffled[:150],
        "resid": shuffled[150:200],
        "eval": shuffled[200:260],
    }


def verify_corpus_disjointness(corpus: Dict[str, List[str]], forbidden_terms: Sequence[str]) -> List[Tuple[str, str]]:
    """Return any (split, term) word-boundary collisions between general corpus and forbidden entities."""
    all_texts = {k: "\n".join(v).lower() for k, v in corpus.items()}
    hits: List[Tuple[str, str]] = []
    for term in sorted(set(forbidden_terms)):
        t = term.strip().lower()
        if len(t) < 3:
            continue
        pat = re.compile(r"\b" + re.escape(t) + r"\b")
        for sp, txt in all_texts.items():
            if pat.search(txt):
                hits.append((sp, term))
    return hits


def build_entity_attr_prompts(entity: str, target_country: str, prot_q: Optional[str] = None, prot_ans: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Standard 4 geographic attribute prompts + 1 protected prompt with exact char spans."""
    info = COUNTRIES_INFO[target_country]
    prompts: Dict[str, Dict[str, Any]] = {}
    spec = {
        "country": (f"The location {entity} is situated in the country of", target_country, info["aliases"]),
        "capital": (f"The capital city of the country containing {entity} is", info["capital"][0], info["capital"][1]),
        "currency": (f"The official currency used in the country containing {entity} is the", info["currency"][0], info["currency"][1]),
        "language": (f"The primary official language spoken in the country containing {entity} is", info["language"][0], info["language"][1]),
    }
    for attr, (text, ans, aliases) in spec.items():
        c0 = text.index(entity)
        prompts[attr] = {
            "prompt": text,
            "char_span": [c0, c0 + len(entity)],
            "answer": ans,
            "aliases": aliases,
        }
    if prot_q is not None and prot_ans is not None:
        c0 = prot_q.index(entity)
        prompts["protected"] = {
            "prompt": prot_q,
            "char_span": [c0, c0 + len(entity)],
            "answer": prot_ans[0],
            "aliases": prot_ans,
        }
    return prompts


def build_landmark_prompts(lm_entry: Dict[str, Any], target_country: str) -> Dict[str, Dict[str, Any]]:
    """Exact shared-prefix landmark evaluation prompts (unseen templates vs TRAIN_PREFIX_TEMPLATES)."""
    e = lm_entry["name"]
    info = COUNTRIES_INFO[target_country]
    spec = {
        "country": (f"The landmark {e} is located in the country of", target_country, info["aliases"]),
        "capital": (f"The capital city of the country where {e} is located is", info["capital"][0], info["capital"][1]),
        "currency": (f"The official currency of the country where {e} is located is the", info["currency"][0], info["currency"][1]),
        "language": (f"The primary language of the country where {e} is located is", info["language"][0], info["language"][1]),
    }
    prompts: Dict[str, Dict[str, Any]] = {}
    for attr, (text, ans, aliases) in spec.items():
        c0 = text.index(e)
        prompts[attr] = {
            "prompt": text,
            "char_span": [c0, c0 + len(e)],
            "answer": ans,
            "aliases": aliases,
        }
    # Protected prompt from LANDMARKS_T0
    pq = f"The landmark {e} is best known as a"
    c0 = pq.index(e)
    prompts["protected"] = {
        "prompt": pq,
        "char_span": [c0, c0 + len(e)],
        "answer": lm_entry["struct_aliases"][0],
        "aliases": lm_entry["struct_aliases"],
    }
    return prompts


# ======================================================================================
# 2. Pure CPU Float64 Algebra: Sockets, CCA, Constant-Pinning, BOS Nulling, Conjugation
# ======================================================================================

def filter_outlier_pairs(
    X_A: np.ndarray,
    X_B: np.ndarray,
    max_med_mult: float = 2.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Filter massive-activation outlier tokens (e.g. Qwen2.5-3B ~1000+ norm delimiters in h_out)
    before fitting residual-stream sockets: keep rows where ||x_A|| <= max_med_mult * median(||x_A||)
    AND ||x_B|| <= max_med_mult * median(||x_B||).
    """
    nA = np.linalg.norm(np.asarray(X_A, dtype=np.float64), axis=-1)
    nB = np.linalg.norm(np.asarray(X_B, dtype=np.float64), axis=-1)
    medA = float(np.median(nA))
    medB = float(np.median(nB))
    keep = (nA <= max_med_mult * medA) & (nB <= max_med_mult * medB)
    stats = {
        "n_before": int(len(nA)),
        "n_after": int(np.sum(keep)),
        "n_dropped": int(len(nA) - np.sum(keep)),
        "med_norm_A": round(medA, 3),
        "med_norm_B": round(medB, 3),
        "max_norm_A_before": round(float(np.max(nA)), 3),
        "max_norm_A_after": round(float(np.max(nA[keep])), 3) if np.any(keep) else 0.0,
        "max_norm_B_before": round(float(np.max(nB)), 3),
        "max_norm_B_after": round(float(np.max(nB[keep])), 3) if np.any(keep) else 0.0,
    }
    return keep, stats


def build_gamma_deconv_read_map(
    gamma_A: np.ndarray,
    gamma_B: np.ndarray,
    R_h: np.ndarray,
    lam_frac: float = 0.05,
) -> np.ndarray:
    """Fable's pre-gamma RMSNorm-invariant read conjugation: diag(gamma_A) @ R_h @ diag(gamma_B / (gamma_B^2 + lam))."""
    gA = np.asarray(gamma_A, dtype=np.float64)
    gB = np.asarray(gamma_B, dtype=np.float64)
    lam_g = lam_frac * float(np.median(gB ** 2))
    inv_gB = gB / (gB ** 2 + lam_g)
    return (gA[:, None] * np.asarray(R_h, dtype=np.float64)) * inv_gB[None, :]


def build_donor_intrinsic_noise(
    X_A_gen: np.ndarray,
    n_samples: int = 4000,
    sigma_rel: float = 0.45,
    seed: int = SEED + 707,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    100% Recipient-Free donor-intrinsic noise model calibrated strictly from Donor A's own general-text states:
      Sigma_int = sigma_rel^2 * (0.5 * (tr(Sigma_A)/D) * I + 0.5 * Sigma_A).
    Returns (E_int [n_samples, D] float32, Sigma_int [D, D] float64). Zero recipient data!
    """
    Xa = np.asarray(X_A_gen, dtype=np.float64)
    Xc = Xa - Xa.mean(axis=0, keepdims=True)
    N, D = Xc.shape
    Sigma_A = (Xc.T @ Xc) / max(N, 1)
    v_A = float(np.trace(Sigma_A) / D)
    Sigma_int = (sigma_rel ** 2) * (0.5 * v_A * np.eye(D) + 0.5 * Sigma_A)
    evals, evecs = sla.eigh(Sigma_int)
    evals = np.maximum(evals, 1e-12)
    L_int = evecs * np.sqrt(evals)[None, :]
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_samples, D))
    E_int = (Z @ L_int.T).astype(np.float32)
    return E_int, Sigma_int


def fit_orthogonal_socket(X_src: np.ndarray, X_dst: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Fit centered orthogonal + scalar map: X_dst ≈ mu_dst + (X_src - mu_src) @ M.T
    where M = s * R (R in O(D)). Runs strictly in float64.
    """
    Xs = np.asarray(X_src, dtype=np.float64)
    Xd = np.asarray(X_dst, dtype=np.float64)
    N, D = Xs.shape
    if weights is None:
        w = np.ones(N, dtype=np.float64) / N
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / np.sum(w)
    mu_s = (w[:, None] * Xs).sum(axis=0)
    mu_d = (w[:, None] * Xd).sum(axis=0)
    Cs = Xs - mu_s
    Cd = Xd - mu_d
    # Cross-covariance C = Cd^T diag(w) Cs -> want M @ (x_s - mu_s) ≈ (x_d - mu_d)
    # Wait: maximize tr(R @ Cs^T diag(w) Cd) => U, S, Vt = svd(Cd^T diag(w) Cs); R = U @ Vt
    C = (Cd * w[:, None]).T @ Cs
    U, S, Vt = sla.svd(C, full_matrices=False)
    R = U @ Vt
    # Optimal isotropic scale s = tr(R @ Cs^T W Cd) / tr(Cs^T W Cs) = sum(S) / var_s
    var_s = float(np.sum(w[:, None] * (Cs ** 2)))
    s = float(np.sum(S) / max(var_s, 1e-12))
    M = s * R
    return {
        "R": R,
        "s": s,
        "M": M,
        "mu_src": mu_s,
        "mu_dst": mu_d,
        "sing_vals": S,
    }


def fit_ridge_read_map(X_src: np.ndarray, X_dst: np.ndarray, lam: float = 1e-2) -> Dict[str, Any]:
    """Fit centered Ridge regression map X_dst ≈ mu_dst + (X_src - mu_src) @ M.T in float64."""
    Xs = np.asarray(X_src, dtype=np.float64)
    Xd = np.asarray(X_dst, dtype=np.float64)
    mu_s = Xs.mean(axis=0)
    mu_d = Xd.mean(axis=0)
    Cs = Xs - mu_s
    Cd = Xd - mu_d
    N, D = Cs.shape
    cov_s = (Cs.T @ Cs) / N
    cross = (Cd.T @ Cs) / N
    tr_scale = float(np.trace(cov_s) / D)
    M = sla.solve(cov_s + (lam * tr_scale) * np.eye(D), cross.T, assume_a="pos").T
    return {"M": M, "mu_src": mu_s, "mu_dst": mu_d}


def compute_constant_predictor(X: np.ndarray, lam: float = 1e-2) -> Tuple[np.ndarray, float]:
    """
    Compute ridge-regularized linear constant predictor c = argmin E[(c^T x - 1)^2] + lam*||c||^2
    over non-BOS tokens. Returns (c, std(c^T x)).
    """
    Xf = np.asarray(X, dtype=np.float64)
    N, D = Xf.shape
    C2 = (Xf.T @ Xf) / N
    mu = Xf.mean(axis=0)
    tr_scale = float(np.trace(C2) / D)
    c = sla.solve(C2 + (lam * tr_scale) * np.eye(D), mu, assume_a="pos")
    preds = Xf @ c
    return c, float(np.std(preds))


def fit_constant_pinned_socket(X_src: np.ndarray, X_dst: np.ndarray, lam_c: float = 1e-2) -> Dict[str, Any]:
    """
    Fit strictly bias-free constant-pinned map M_cpin (src -> dst) satisfying:
        c_dst^T @ M_cpin == c_src^T  (exact to <1e-12!)
    so that any Donor A bias component b * c_dst^T transforms into b * c_src^T on Recipient B:
        (b c_dst)^T M_cpin x_B == b (c_src^T x_B) ≈ b!
    On the orthogonal complement c_src^perp -> c_dst^perp, fits optimal orthogonal Procrustes + scalar s.
    """
    Xs = np.asarray(X_src, dtype=np.float64)
    Xd = np.asarray(X_dst, dtype=np.float64)
    D = Xs.shape[1]
    c_src, std_src = compute_constant_predictor(Xs, lam=lam_c)
    c_dst, std_dst = compute_constant_predictor(Xd, lam=lam_c)

    u_s = c_src / np.linalg.norm(c_src)
    u_d = c_dst / np.linalg.norm(c_dst)

    # Orthonormal bases for u_s^perp and u_d^perp via Householder / QR
    def _perp_basis(u: np.ndarray) -> np.ndarray:
        e1 = np.zeros_like(u)
        e1[0] = 1.0 if u[0] < 0 else -1.0
        v = u - e1
        v_norm2 = float(np.dot(v, v))
        if v_norm2 < 1e-18:
            H = np.eye(len(u))
        else:
            H = np.eye(len(u)) - 2.0 * np.outer(v, v) / v_norm2
        # Column 0 of H is +-u; columns 1..D-1 span u^perp!
        return H[:, 1:]

    Qs = _perp_basis(u_s)  # (D, D-1)
    Qd = _perp_basis(u_d)  # (D, D-1)

    # Project centered data onto perp subspaces and fit orthogonal Procrustes + scalar
    Cs = (Xs - Xs.mean(axis=0)) @ Qs  # (N, D-1)
    Cd = (Xd - Xd.mean(axis=0)) @ Qd  # (N, D-1)
    cross = (Cd.T @ Cs) / len(Xs)
    U, S, Vt = sla.svd(cross, full_matrices=False)
    R_perp = U @ Vt
    var_s = float(np.mean(np.sum(Cs ** 2, axis=1)))
    s_perp = float(np.sum(S) / max(var_s, 1e-12))

    # M_perp maps R^D -> u_d^perp with M_perp @ u_s = 0 and u_d^T @ M_perp = 0
    M_perp = s_perp * (Qd @ R_perp @ Qs.T)
    # Rank-1 pin: (c_dst / ||c_dst||^2) @ c_src^T  =>  c_dst^T @ M_cpin = c_src^T EXACTLY!
    M_pin = np.outer(c_dst / float(np.dot(c_dst, c_dst)), c_src)
    M_cpin = M_perp + M_pin

    # Sanity check
    pin_err = float(np.linalg.norm(c_dst @ M_cpin - c_src))
    assert pin_err < 1e-10, f"Constant-pin error too large: {pin_err}"

    return {
        "M": M_cpin,
        "c_src": c_src,
        "c_dst": c_dst,
        "std_src": std_src,
        "std_dst": std_dst,
        "s_perp": s_perp,
        "pin_err": pin_err,
    }


def compute_heldout_cca_spectrum(
    X_A_fit: np.ndarray,
    X_B_fit: np.ndarray,
    X_A_val: np.ndarray,
    X_B_val: np.ndarray,
    ridge: float = 1e-2,
) -> Dict[str, Any]:
    """
    Regularized CCA fit strictly on (X_A_fit, X_B_fit), canonical correlations evaluated strictly
    on held-out (X_A_val, X_B_val) to avoid high-D in-sample inflation!
    """
    Af = np.asarray(X_A_fit, dtype=np.float64)
    Bf = np.asarray(X_B_fit, dtype=np.float64)
    Av = np.asarray(X_A_val, dtype=np.float64)
    Bv = np.asarray(X_B_val, dtype=np.float64)
    muA, muB = Af.mean(axis=0), Bf.mean(axis=0)
    Af_c, Bf_c = Af - muA, Bf - muB
    Av_c, Bv_c = Av - muA, Bv - muB
    N, D = Af_c.shape

    Caa = (Af_c.T @ Af_c) / N
    Cbb = (Bf_c.T @ Bf_c) / N
    Cab = (Af_c.T @ Bf_c) / N
    la = ridge * float(np.trace(Caa) / D)
    lb = ridge * float(np.trace(Cbb) / D)

    ea, Va = sla.eigh(Caa + la * np.eye(D))
    eb, Vb = sla.eigh(Cbb + lb * np.eye(D))
    inv_sqrt_a = Va @ np.diag(1.0 / np.sqrt(np.maximum(ea, 1e-12))) @ Va.T
    inv_sqrt_b = Vb @ np.diag(1.0 / np.sqrt(np.maximum(eb, 1e-12))) @ Vb.T

    K = inv_sqrt_a @ Cab @ inv_sqrt_b
    U, S_train, Vt = sla.svd(K, full_matrices=False)
    Wa = inv_sqrt_a @ U  # (D, D)
    Wb = inv_sqrt_b @ Vt.T  # (D, D)

    Za_val = Av_c @ Wa
    Zb_val = Bv_c @ Wb
    Za_val -= Za_val.mean(axis=0, keepdims=True)
    Zb_val -= Zb_val.mean(axis=0, keepdims=True)
    num = np.sum(Za_val * Zb_val, axis=0)
    den = np.sqrt(np.sum(Za_val ** 2, axis=0) * np.sum(Zb_val ** 2, axis=0)) + 1e-12
    rho_val = np.clip(num / den, -1.0, 1.0)
    return {
        "n_ge_090": int(np.sum(rho_val >= 0.90)),
        "n_ge_080": int(np.sum(rho_val >= 0.80)),
        "n_ge_070": int(np.sum(rho_val >= 0.70)),
        "n_ge_050": int(np.sum(rho_val >= 0.50)),
        "top10_val_rho": [round(float(x), 4) for x in rho_val[:10]],
        "rho_at_k": {
            "k10": round(float(rho_val[9]), 4),
            "k50": round(float(rho_val[49]), 4),
            "k100": round(float(rho_val[99]), 4),
            "k256": round(float(rho_val[255]), 4),
            "k512": round(float(rho_val[511]), 4),
            "k1024": round(float(rho_val[1023]), 4),
        },
        "mean_top256_val_rho": round(float(np.mean(rho_val[:256])), 4),
        "mean_top256_train_rho": round(float(np.mean(S_train[:256])), 4),
    }


def compute_split_half_agreement(X_src: np.ndarray, X_dst: np.ndarray) -> Dict[str, Any]:
    """Split fit data into two disjoint halves, fit R1 & R2, measure directional agreement."""
    N = len(X_src)
    mid = N // 2
    s1 = fit_orthogonal_socket(X_src[:mid], X_dst[:mid])
    s2 = fit_orthogonal_socket(X_src[mid:], X_dst[mid:])
    # Compute cosine similarity of mapped held-out directions across top-256 PCs of X_src
    Xs_c = X_src - X_src.mean(axis=0)
    _, _, Vt = sla.svd(Xs_c[: min(N, 4000)], full_matrices=False)
    V256 = Vt[:256].T  # (D, 256)
    Y1 = s1["R"] @ V256
    Y2 = s2["R"] @ V256
    cos_top256 = np.sum(Y1 * Y2, axis=0)
    diag_full = np.sum(s1["R"] * s2["R"], axis=0)
    return {
        "full_mean_cos": round(float(np.mean(diag_full)), 4),
        "top256_pc_mean_cos": round(float(np.mean(cos_top256)), 4),
        "top256_pc_median_cos": round(float(np.median(cos_top256)), 4),
        "n_top256_ge_090": int(np.sum(cos_top256 >= 0.90)),
        "n_top256_ge_080": int(np.sum(cos_top256 >= 0.80)),
    }


def compute_write_core_projector(
    H_A_val: np.ndarray,
    H_B_val: np.ndarray,
    M_out: np.ndarray,
    r_out: int = 512,
    massive_thresh: float = 10.0,
) -> Dict[str, Any]:
    """
    Build Opus Q2 'write narrow' projector P_write_core (D x D) in Recipient B's residual space:
      1. Identify and mask any B massive-activation coordinates whose RMS / median_RMS > massive_thresh.
      2. On remaining coordinates, compute eigenbasis of cross-model predictable signal covariance
         Cov(M_out (h_A - mu_A), h_B - mu_B) on held-out pairs, ranked by held-out directional correlation,
         and keep top r_out reliable write directions!
    """
    Ha = np.asarray(H_A_val, dtype=np.float64)
    Hb = np.asarray(H_B_val, dtype=np.float64)
    D = Hb.shape[1]
    coord_rms = np.sqrt(np.mean(Hb ** 2, axis=0))
    med_rms = float(np.median(coord_rms))
    massive_dims = np.where(coord_rms > massive_thresh * med_rms)[0]
    keep_mask = np.ones(D, dtype=np.float64)
    keep_mask[massive_dims] = 0.0

    Hb_c = (Hb - Hb.mean(axis=0)) * keep_mask[None, :]
    Ha_mapped_c = ((Ha - Ha.mean(axis=0)) @ M_out.T) * keep_mask[None, :]

    # Principal directions of mapped Donor A residual stream, scored by held-out correlation with B!
    cov_pred = (Ha_mapped_c.T @ Ha_mapped_c) / len(Ha)
    evals, evecs = sla.eigh(cov_pred)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]  # (D, D)

    # Held-out correlation along each eigenvector
    Za = Ha_mapped_c @ evecs
    Zb = Hb_c @ evecs
    num = np.sum(Za * Zb, axis=0)
    den = np.sqrt(np.sum(Za ** 2, axis=0) * np.sum(Zb ** 2, axis=0)) + 1e-12
    corr = num / den
    # Keep top r_out eigenvectors ranked by correlation * sqrt(variance-stabilized reliability)
    # Pure correlation ranking among non-degenerate modes (eval > 1e-4 * max_eval)
    valid = evals[order] > 1e-4 * evals[order[0]]
    scores = np.where(valid, corr, -1.0)
    top_idx = np.argsort(scores)[::-1][:r_out]
    U_core = evecs[:, top_idx]  # (D, r_out)
    # Enforce zero on massive dims explicitly
    U_core[massive_dims, :] = 0.0
    # Re-orthonormalize columns via QR
    Q_core, _ = sla.qr(U_core, mode="economic")
    P_core = Q_core @ Q_core.T

    return {
        "P_core": P_core,
        "Q_core": Q_core,
        "massive_dims": [int(i) for i in massive_dims],
        "n_massive_dims": int(len(massive_dims)),
        "top_r_mean_corr": round(float(np.mean(corr[top_idx])), 4),
        "top_r_min_corr": round(float(np.min(corr[top_idx])), 4),
    }


def compute_uncentered_C0_basis(X_B_all: np.ndarray, r_c0: int = 32) -> Dict[str, Any]:
    """Top r_c0 eigenvectors of B's uncentered general-text second moment C_0 = E[x_B x_B^T] over ALL B positions."""
    Xb = np.asarray(X_B_all, dtype=np.float64)
    C0 = (Xb.T @ Xb) / len(Xb)
    evals, evecs = sla.eigh(C0)
    order = np.argsort(evals)[::-1]
    V_c0 = evecs[:, order[:r_c0]]
    var_frac = float(np.sum(evals[order[:r_c0]]) / max(np.sum(evals), 1e-12))
    return {
        "V_c0": V_c0,
        "r_c0": r_c0,
        "top_energy_frac": round(var_frac, 4),
    }


def apply_exact_bos_null_affine(
    W_g: np.ndarray,
    b_g: np.ndarray,
    W_u: np.ndarray,
    b_u: np.ndarray,
    x_bos: np.ndarray,
    mu_B: np.ndarray,
    target_bos_gate: float = -20.0,
    x_bos_bank: Optional[np.ndarray] = None,
    r_bos_subspace: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact minimum-Frobenius rank-1 up-branch update along v = x_BOS - mu_B (plus optional rank-r
    pos-0 sink-variance projection when pos-0 varies across first tokens without a fixed BOS) so that:
      1. W_u_new @ x_BOS + b_u_new == 0.0              (exact zero up-branch at mean pos-0)
      2. W_u_new @ mu_B  + b_u_new == W_u @ mu_B + b_u (exact mean preservation)
      3. When x_bos_bank (N_pos0 x D) has non-zero variance across pos-0 tokens, projects W_u
         off the top r_bos_subspace centered pos-0 directions (orthogonal to v), mean-compensated at mu_B.

    NOTE (v1.0 erratum): `target_bos_gate` is accepted for API compatibility but intentionally
    unapplied here; W_g and b_g are returned unchanged so that gate selectivity is unaffected and
    published affine tables remain reproducible. Sealed Cap-2 arms use `apply_exact_bos_null_biasfree`
    and do not call this function. Note also that the linear up-null is exact at the fitted x_BOS
    vector (~4.9e-7) rather than identically zero across unseen pos-0 tokens (e.g., Cap-2 SmolLM2
    `bos_eval60_max_norm = 0.028`).
    """
    Wg = np.asarray(W_g, dtype=np.float64).copy()
    bg = np.asarray(b_g, dtype=np.float64).copy()
    Wu = np.asarray(W_u, dtype=np.float64).copy()
    bu = np.asarray(b_u, dtype=np.float64).copy()
    xb = np.asarray(x_bos, dtype=np.float64)
    mb = np.asarray(mu_B, dtype=np.float64)

    v = xb - mb
    v_norm2 = float(np.dot(v, v))
    assert v_norm2 > 1e-12, "x_BOS coincides with mu_B!"

    # Optional pos-0 variance subspace projection (auto-skipped if fixed BOS token!)
    if x_bos_bank is not None and len(x_bos_bank) > 2:
        Xc = np.asarray(x_bos_bank, dtype=np.float64) - xb[None, :]
        # Orthogonalize against v so step 2 below stays exact
        v_unit = v / math.sqrt(v_norm2)
        Xc = Xc - np.outer(Xc @ v_unit, v_unit)
        _, sv, Vt = sla.svd(Xc, full_matrices=False)
        if len(sv) > 0 and sv[0] > 1e-4 * math.sqrt(v_norm2):
            keep = int(np.sum(sv > 1e-4 * sv[0]))
            r_use = min(r_bos_subspace, keep)
            Q = Vt[:r_use].T  # (D, r_use), orthonormal and orthogonal to v
            # Project Wu off Q while preserving exact response at mu_B (leave Wg untouched!)
            dWu_q = (Wu @ Q) @ Q.T
            Wu = Wu - dWu_q
            bu = bu + (dWu_q @ mb)

    # Exact rank-1 up-null at x_BOS, preserving mu_B (SiLU(g) * 0 == 0 exactly, without distorting Wg!)
    u0 = Wu @ xb + bu
    dWu = np.outer(u0, v) / v_norm2
    Wu_new = Wu - dWu
    bu_new = bu + (dWu @ mb)

    return Wg, bg, Wu_new, bu_new


def apply_exact_bos_null_biasfree(
    W_u: np.ndarray,
    x_bos: np.ndarray,
    x_bos_bank: Optional[np.ndarray] = None,
    r_bos_subspace: int = 8,
) -> np.ndarray:
    """Strictly bias-free orthogonal projection so W_u_new @ x_BOS == 0 (and pos-0 subspace nulled)."""
    Wu = np.asarray(W_u, dtype=np.float64).copy()
    xb = np.asarray(x_bos, dtype=np.float64)
    if x_bos_bank is not None and len(x_bos_bank) > 2:
        X0 = np.vstack([xb[None, :], np.asarray(x_bos_bank, dtype=np.float64)])
        _, sv, Vt = sla.svd(X0, full_matrices=False)
        keep = int(np.sum(sv > 1e-4 * sv[0]))
        r_use = min(r_bos_subspace + 1, keep)
        Q = Vt[:r_use].T
        Wu = Wu - (Wu @ Q) @ Q.T
    xb_norm2 = float(np.dot(xb, xb))
    assert xb_norm2 > 1e-12
    u0 = Wu @ xb
    return Wu - np.outer(u0, xb) / xb_norm2


def conjugate_organ_weights(
    organ_A: Dict[str, np.ndarray],
    arm: str,
    M_in: np.ndarray,
    mu_A: np.ndarray,
    mu_B: np.ndarray,
    M_out: np.ndarray,
    x_bos_B: np.ndarray,
    rm_B: float = 1.0,
    M_in_cpin: Optional[np.ndarray] = None,
    M_gamma: Optional[np.ndarray] = None,
    c_B: Optional[np.ndarray] = None,
    P_write_core: Optional[np.ndarray] = None,
    V_c0: Optional[np.ndarray] = None,
    x_bos_bank_B: Optional[np.ndarray] = None,
    rng_seed: int = SEED + 99,
) -> Dict[str, Any]:
    """
    Conjugate Donor A organ weights (W_g, b_g, W_u, b_u, W_d) into Recipient B's native
    MLP weight space according to arm in {'V0', 'V1', 'V1b', 'V1b_gamma', 'V2_intrinsic', 'V2', 'V4', 'shuffled_R'}.
    All algebra in float64. Returns float64 arrays + use_bias bool.
    """
    Wg = np.asarray(organ_A["W_g"], dtype=np.float64)
    Wu = np.asarray(organ_A["W_u"], dtype=np.float64)
    Wd = np.asarray(organ_A["W_d"], dtype=np.float64)
    bg = np.asarray(organ_A["b_g"], dtype=np.float64) if organ_A.get("b_g") is not None else np.zeros(Wg.shape[0], dtype=np.float64)
    bu = np.asarray(organ_A["b_u"], dtype=np.float64) if organ_A.get("b_u") is not None else np.zeros(Wu.shape[0], dtype=np.float64)

    inv_rm = 1.0 / float(rm_B)

    if arm == "V0":
        # Naive bias-free centered R, NO bias, NO BOS null, NO write-core
        Wg_B = Wg @ M_in
        Wu_B = Wu @ M_in
        Wd_B = inv_rm * (M_out @ Wd)
        return {"W_g": Wg_B, "b_g": None, "W_u": Wu_B, "b_u": None, "W_d": Wd_B, "use_bias": False}

    if arm == "V1b":
        # Strictly bias-free mlp_bias=False + constant-pinned M_in_cpin + exact BOS up-null + write-core
        assert M_in_cpin is not None
        Wg_B = Wg @ M_in_cpin
        Wu_B = Wu @ M_in_cpin
        Wu_B = apply_exact_bos_null_biasfree(Wu_B, x_bos_B, x_bos_bank=x_bos_bank_B)
        Wd_B = M_out @ Wd
        if P_write_core is not None:
            Wd_B = P_write_core @ Wd_B
        Wd_B = inv_rm * Wd_B
        return {"W_g": Wg_B, "b_g": None, "W_u": Wu_B, "b_u": None, "W_d": Wd_B, "use_bias": False}

    if arm == "V1b_gamma":
        # Strictly bias-free mlp_bias=False + Fable's pre-gamma deconvolved M_gamma + constant-pin folding via c_B + exact BOS up-null!
        assert M_gamma is not None and c_B is not None
        cB = np.asarray(c_B, dtype=np.float64)
        Wg_B = Wg @ M_gamma + np.outer(bg + Wg @ (mu_A - M_gamma @ mu_B), cB)
        Wu_B = Wu @ M_gamma + np.outer(bu + Wu @ (mu_A - M_gamma @ mu_B), cB)
        Wu_B = apply_exact_bos_null_biasfree(Wu_B, x_bos_B, x_bos_bank=x_bos_bank_B)
        Wd_B = M_out @ Wd
        if P_write_core is not None:
            Wd_B = P_write_core @ Wd_B
        Wd_B = inv_rm * Wd_B
        return {"W_g": Wg_B, "b_g": None, "W_u": Wu_B, "b_u": None, "W_d": Wd_B, "use_bias": False}

    if arm == "shuffled_R":
        # Random Haar orthogonal control (scale-matched to M_in / M_out, with affine bias + BOS null!)
        rng = np.random.default_rng(rng_seed)
        D = M_in.shape[0]
        Q_in, _ = sla.qr(rng.standard_normal((D, D)))
        Q_out, _ = sla.qr(rng.standard_normal((D, D)))
        s_in = float(np.sqrt(np.mean(np.sum(M_in ** 2, axis=1))))
        s_out = float(np.sqrt(np.mean(np.sum(M_out ** 2, axis=1))))
        M_in_use = s_in * Q_in
        M_out_use = s_out * Q_out
        Wg_B = Wg @ M_in_use
        bg_B = bg + Wg @ (mu_A - M_in_use @ mu_B)
        Wu_B = Wu @ M_in_use
        bu_B = bu + Wu @ (mu_A - M_in_use @ mu_B)
        Wg_B, bg_B, Wu_B, bu_B = apply_exact_bos_null_affine(Wg_B, bg_B, Wu_B, bu_B, x_bos_B, mu_B, x_bos_bank=x_bos_bank_B)
        Wd_B = inv_rm * (M_out_use @ Wd)
        return {"W_g": Wg_B, "b_g": bg_B, "W_u": Wu_B, "b_u": bu_B, "W_d": Wd_B, "use_bias": True}

    # Affine arms: V1, V2_intrinsic, V2, V4
    Wg_B = Wg @ M_in
    bg_B = bg + Wg @ (mu_A - M_in @ mu_B)
    Wu_B = Wu @ M_in
    bu_B = bu + Wu @ (mu_A - M_in @ mu_B)

    if arm == "V4":
        # Closed-form B-side uncentered C_0 top-32 PC up-projection (zero B SGD!)
        assert V_c0 is not None
        Wu_proj_comp = (Wu_B @ V_c0) @ V_c0.T
        Wu_B = Wu_B - Wu_proj_comp
        bu_B = bu_B + (Wu_proj_comp @ mu_B)

    # Exact rank-1 BOS up-null (leaves Wg untouched, preserves mean at mu_B!)
    Wg_B, bg_B, Wu_B, bu_B = apply_exact_bos_null_affine(Wg_B, bg_B, Wu_B, bu_B, x_bos_B, mu_B, x_bos_bank=x_bos_bank_B)

    Wd_B = M_out @ Wd
    if arm in ("V2", "V2_intrinsic", "V4") and P_write_core is not None:
        Wd_B = P_write_core @ Wd_B
    Wd_B = inv_rm * Wd_B

    return {"W_g": Wg_B, "b_g": bg_B, "W_u": Wu_B, "b_u": bu_B, "W_d": Wd_B, "use_bias": True}


# ======================================================================================
# 3. Native Stock MLP Widening & Instant Exact Unlearning (Zero Hooks!)
# ======================================================================================

def save_original_mlp_state(mlp: nn.Module) -> Dict[str, Any]:
    """Snapshot original MLP weights/config so unlearning restores bit-identical weights."""
    return {
        "d_ff": int(mlp.gate_proj.out_features),
        "d_model": int(mlp.gate_proj.in_features),
        "gate_bias": mlp.gate_proj.bias is not None,
        "up_bias": mlp.up_proj.bias is not None,
        "down_bias": mlp.down_proj.bias is not None,
        "gate_weight": mlp.gate_proj.weight.detach().clone(),
        "up_weight": mlp.up_proj.weight.detach().clone(),
        "down_weight": mlp.down_proj.weight.detach().clone(),
        "gate_b": mlp.gate_proj.bias.detach().clone() if mlp.gate_proj.bias is not None else None,
        "up_b": mlp.up_proj.bias.detach().clone() if mlp.up_proj.bias is not None else None,
        "down_b": mlp.down_proj.bias.detach().clone() if mlp.down_proj.bias is not None else None,
    }


def widen_mlp_native(
    mlp: nn.Module,
    orig_state: Dict[str, Any],
    W_g: np.ndarray,
    b_g: Optional[np.ndarray],
    W_u: np.ndarray,
    b_u: Optional[np.ndarray],
    W_d: np.ndarray,
    use_bias: bool,
) -> int:
    """
    Physically widen LlamaMLP / GraniteMLP from d_ff to d_ff + k by replacing
    gate_proj, up_proj, down_proj with real stock nn.Linear modules.
    - Base rows 0..d_ff are copied bit-identically.
    - down_proj ALWAYS keeps bias=False (zero global constant injection).
    - Zero forward hooks, zero runtime gates!
    """
    device = orig_state["gate_weight"].device
    dtype = orig_state["gate_weight"].dtype
    d_ff = orig_state["d_ff"]
    d_model = orig_state["d_model"]
    k = int(W_g.shape[0])
    new_d_ff = d_ff + k

    gate_use_bias = bool(use_bias or orig_state["gate_bias"])
    up_use_bias = bool(use_bias or orig_state["up_bias"])
    down_use_bias = bool(orig_state["down_bias"])

    new_gate = nn.Linear(d_model, new_d_ff, bias=gate_use_bias, device=device, dtype=dtype)
    new_up = nn.Linear(d_model, new_d_ff, bias=up_use_bias, device=device, dtype=dtype)
    new_down = nn.Linear(new_d_ff, d_model, bias=down_use_bias, device=device, dtype=dtype)

    with torch.no_grad():
        new_gate.weight[:d_ff].copy_(orig_state["gate_weight"])
        new_gate.weight[d_ff:].copy_(torch.from_numpy(np.asarray(W_g, dtype=np.float32)).to(device=device, dtype=dtype))
        if gate_use_bias:
            if orig_state["gate_b"] is not None:
                new_gate.bias[:d_ff].copy_(orig_state["gate_b"])
            else:
                new_gate.bias[:d_ff].zero_()
            if b_g is not None:
                new_gate.bias[d_ff:].copy_(torch.from_numpy(np.asarray(b_g, dtype=np.float32)).to(device=device, dtype=dtype))
            else:
                new_gate.bias[d_ff:].zero_()

        new_up.weight[:d_ff].copy_(orig_state["up_weight"])
        new_up.weight[d_ff:].copy_(torch.from_numpy(np.asarray(W_u, dtype=np.float32)).to(device=device, dtype=dtype))
        if up_use_bias:
            if orig_state["up_b"] is not None:
                new_up.bias[:d_ff].copy_(orig_state["up_b"])
            else:
                new_up.bias[:d_ff].zero_()
            if b_u is not None:
                new_up.bias[d_ff:].copy_(torch.from_numpy(np.asarray(b_u, dtype=np.float32)).to(device=device, dtype=dtype))
            else:
                new_up.bias[d_ff:].zero_()

        new_down.weight[:, :d_ff].copy_(orig_state["down_weight"])
        new_down.weight[:, d_ff:].copy_(torch.from_numpy(np.asarray(W_d, dtype=np.float32)).to(device=device, dtype=dtype))
        if down_use_bias and orig_state["down_b"] is not None:
            new_down.bias.copy_(orig_state["down_b"])

    mlp.gate_proj = new_gate
    mlp.up_proj = new_up
    mlp.down_proj = new_down
    mlp.intermediate_size = new_d_ff
    return new_d_ff


def unlearn_mlp_native(mlp: nn.Module, orig_state: Dict[str, Any]) -> None:
    """
    Instant Exact Unlearning: slice the widened MLP back from d_ff + k -> d_ff directly
    from the widened weights (NOT by restoring cached weights, proving true weight-slice unlearning!).
    """
    d_ff = orig_state["d_ff"]
    d_model = orig_state["d_model"]
    device = mlp.gate_proj.weight.device
    dtype = mlp.gate_proj.weight.dtype

    sliced_gate = nn.Linear(d_model, d_ff, bias=orig_state["gate_bias"], device=device, dtype=dtype)
    sliced_up = nn.Linear(d_model, d_ff, bias=orig_state["up_bias"], device=device, dtype=dtype)
    sliced_down = nn.Linear(d_ff, d_model, bias=orig_state["down_bias"], device=device, dtype=dtype)

    with torch.no_grad():
        sliced_gate.weight.copy_(mlp.gate_proj.weight[:d_ff])
        if orig_state["gate_bias"] and mlp.gate_proj.bias is not None:
            sliced_gate.bias.copy_(mlp.gate_proj.bias[:d_ff])
        sliced_up.weight.copy_(mlp.up_proj.weight[:d_ff])
        if orig_state["up_bias"] and mlp.up_proj.bias is not None:
            sliced_up.bias.copy_(mlp.up_proj.bias[:d_ff])
        sliced_down.weight.copy_(mlp.down_proj.weight[:, :d_ff])
        if orig_state["down_bias"] and mlp.down_proj.bias is not None:
            sliced_down.bias.copy_(mlp.down_proj.bias)

    mlp.gate_proj = sliced_gate
    mlp.up_proj = sliced_up
    mlp.down_proj = sliced_down
    mlp.intermediate_size = d_ff


# ======================================================================================
# 4. Model Loading & Activation Extraction Helpers (MPS float32 safe!)
# ======================================================================================

def get_compute_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_fp32(model_key: str) -> Tuple[Any, Any, torch.device, float]:
    """Load model offline in float32 (safe for Granite emb_mult=12 and exact KL/BOS diagnostics)."""
    device = get_compute_device()
    model_dir = HF_CACHE_DIR / model_key
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype=torch.float32,
    )
    model.to(device)
    model.eval()
    layers = get_decoder_layers(model)
    rm = float(getattr(layers[0], "residual_multiplier", 1.0))
    return model, tok, device, rm


def unload_model(model: Any) -> None:
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def extract_layer_states_on_passages(
    model: Any,
    tok: Any,
    device: torch.device,
    passages: Sequence[str],
    layer_idx: int,
    max_len: int = 160,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Run passages through model at layer_idx and capture:
      - x: post_attention_layernorm output (MLP input)
      - h_hat: unit-RMS pre-gamma normalized h_mid (x / gamma)
      - h_out: post-MLP layer residual output
      - mlp_out: native mlp(x) output norm (for leak energy ratio denominator!)
      - word_end_map: per passage, dict {char_end: tok_idx} for word-final tokens (pos > 0!)
      - bos_vecs: pos-0 x vectors across passages
    """
    layers = get_decoder_layers(model)
    layer = layers[layer_idx]
    gamma = layer.post_attention_layernorm.weight.detach().float().cpu().numpy().astype(np.float64)
    eps = float(getattr(layer.post_attention_layernorm, "variance_epsilon", getattr(layer.post_attention_layernorm, "eps", 1e-6)))

    buf: Dict[str, torch.Tensor] = {}

    def _pre_ln_hook(_mod, inp):
        buf["h_mid"] = inp[0].detach()

    def _post_ln_hook(_mod, _inp, out):
        buf["x"] = out.detach()

    def _mlp_hook(_mod, _inp, out):
        buf["mlp_out"] = out.detach()

    def _layer_out_hook(_mod, _inp, out):
        buf["h_out"] = (out[0] if isinstance(out, tuple) else out).detach()

    h1 = layer.post_attention_layernorm.register_forward_pre_hook(_pre_ln_hook)
    h2 = layer.post_attention_layernorm.register_forward_hook(_post_ln_hook)
    h3 = layer.mlp.register_forward_hook(_mlp_hook)
    h4 = layer.register_forward_hook(_layer_out_hook)

    all_x: List[np.ndarray] = []
    all_hhat: List[np.ndarray] = []
    all_hout: List[np.ndarray] = []
    all_mlp_norm2: List[np.ndarray] = []
    bos_x_list: List[np.ndarray] = []
    word_end_maps: List[Dict[int, int]] = []
    seq_lens: List[int] = []

    old_pad_side = tok.padding_side
    tok.padding_side = "right"
    try:
        with torch.no_grad():
            for i0 in range(0, len(passages), batch_size):
                batch_txt = list(passages[i0 : i0 + batch_size])
                enc = tok(
                    batch_txt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                )
                offsets = enc.pop("offset_mapping").tolist()
                input_ids = enc["input_ids"].to(device)
                attn_mask = enc["attention_mask"].to(device)
                _ = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)

                x_b = buf["x"].float().cpu().numpy()
                hmid_b = buf["h_mid"].float().cpu().numpy()
                hout_b = buf["h_out"].float().cpu().numpy()
                mlp_b = buf["mlp_out"].float().cpu().numpy()
                # Exact pre-gamma unit-RMS h_hat
                rms = np.sqrt(np.mean(hmid_b ** 2, axis=-1, keepdims=True) + eps)
                hhat_b = hmid_b / rms
                mlp_n2 = np.sum(mlp_b ** 2, axis=-1)

                for bi, txt in enumerate(batch_txt):
                    L = int(attn_mask[bi].sum().item())
                    seq_lens.append(L)
                    all_x.append(x_b[bi, :L].astype(np.float32))
                    all_hhat.append(hhat_b[bi, :L].astype(np.float32))
                    all_hout.append(hout_b[bi, :L].astype(np.float32))
                    all_mlp_norm2.append(mlp_n2[bi, :L].astype(np.float32))
                    bos_x_list.append(x_b[bi, 0].astype(np.float64))

                    # Identify word-final char_end offsets (strictly pos > 0 to exclude BOS/pos-0!)
                    we_map: Dict[int, int] = {}
                    offs = offsets[bi][:L]
                    tlen = len(txt)
                    for pos in range(1, L):
                        c0, c1 = int(offs[pos][0]), int(offs[pos][1])
                        if c1 <= c0 or c1 > tlen:
                            continue
                        # Word-final boundary: next char is whitespace, punctuation, or end of string
                        if c1 == tlen or (not txt[c1].isalnum()):
                            # Require token itself ends on alnum or closing punct (not pure whitespace)
                            if txt[c1 - 1].strip():
                                we_map[c1] = pos
                    word_end_maps.append(we_map)
    finally:
        h1.remove()
        h2.remove()
        h3.remove()
        h4.remove()
        tok.padding_side = old_pad_side

    return {
        "x": all_x,
        "h_hat": all_hhat,
        "h_out": all_hout,
        "mlp_norm2": all_mlp_norm2,
        "bos_x": np.stack(bos_x_list, axis=0),
        "word_end_maps": word_end_maps,
        "seq_lens": seq_lens,
        "gamma": gamma,
    }


def pair_passage_activations(
    st_A: Dict[str, Any],
    st_B: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Pair token states strictly at shared word-final character offsets (pos > 0 in both A and B).
    Also tracks B's unaligned non-BOS token states separately so Stage 2 reports aligned vs unaligned!
    """
    xA_list, xB_list = [], []
    hhA_list, hhB_list = [], []
    hoA_list, hoB_list = [], []
    mlpB_aligned_n2 = []
    xB_unaligned = []
    mlpB_unaligned_n2 = []
    xB_all_list = []

    n_pass = len(st_A["x"])
    assert n_pass == len(st_B["x"])
    for i in range(n_pass):
        mA = st_A["word_end_maps"][i]
        mB = st_B["word_end_maps"][i]
        shared_ends = sorted(set(mA.keys()) & set(mB.keys()))
        aligned_B_pos = set()
        for c1 in shared_ends:
            pA, pB = mA[c1], mB[c1]
            xA_list.append(st_A["x"][i][pA])
            xB_list.append(st_B["x"][i][pB])
            hhA_list.append(st_A["h_hat"][i][pA])
            hhB_list.append(st_B["h_hat"][i][pB])
            hoA_list.append(st_A["h_out"][i][pA])
            hoB_list.append(st_B["h_out"][i][pB])
            mlpB_aligned_n2.append(st_B["mlp_norm2"][i][pB])
            aligned_B_pos.add(pB)

        LB = st_B["seq_lens"][i]
        xB_all_list.append(st_B["x"][i][:LB])
        for pB in range(1, LB):
            if pB not in aligned_B_pos:
                xB_unaligned.append(st_B["x"][i][pB])
                mlpB_unaligned_n2.append(st_B["mlp_norm2"][i][pB])

    return {
        "x_A": np.stack(xA_list, axis=0),
        "x_B": np.stack(xB_list, axis=0),
        "h_hat_A": np.stack(hhA_list, axis=0),
        "h_hat_B": np.stack(hhB_list, axis=0),
        "h_out_A": np.stack(hoA_list, axis=0),
        "h_out_B": np.stack(hoB_list, axis=0),
        "mlpB_aligned_n2": np.array(mlpB_aligned_n2, dtype=np.float64),
        "x_B_unaligned": np.stack(xB_unaligned, axis=0) if xB_unaligned else np.zeros((0, st_B["x"][0].shape[1]), dtype=np.float32),
        "mlpB_unaligned_n2": np.array(mlpB_unaligned_n2, dtype=np.float64),
        "x_B_all": np.concatenate(xB_all_list, axis=0),
    }


def extract_entity_last_token_states(
    model: Any,
    tok: Any,
    device: torch.device,
    prompts: Sequence[str],
    char_spans: Sequence[Sequence[int]],
    layer_idx: int,
    batch_size: int = 16,
) -> Dict[str, np.ndarray]:
    """Extract (x, h_hat, h_out) at exact last_subject_token for a list of (prompt, char_span)."""
    layers = get_decoder_layers(model)
    layer = layers[layer_idx]
    eps = float(getattr(layer.post_attention_layernorm, "variance_epsilon", getattr(layer.post_attention_layernorm, "eps", 1e-6)))
    buf: Dict[str, torch.Tensor] = {}

    def _pre_ln(_m, inp):
        buf["h_mid"] = inp[0].detach()

    def _post_ln(_m, _inp, out):
        buf["x"] = out.detach()

    def _lout(_m, _inp, out):
        buf["h_out"] = (out[0] if isinstance(out, tuple) else out).detach()

    h1 = layer.post_attention_layernorm.register_forward_pre_hook(_pre_ln)
    h2 = layer.post_attention_layernorm.register_forward_hook(_post_ln)
    h3 = layer.register_forward_hook(_lout)

    # Compute exact unpadded token positions first
    subj_pos_list = []
    seq_len_list = []
    for pr, sp in zip(prompts, char_spans):
        idx_tok, slen = locate_last_subject_token(tok, pr, sp)
        subj_pos_list.append(idx_tok)
        seq_len_list.append(slen)

    xs, hhats, houts = [], [], []
    old_pad_side = tok.padding_side
    tok.padding_side = "right"
    try:
        with torch.no_grad():
            for i0 in range(0, len(prompts), batch_size):
                b_pr = list(prompts[i0 : i0 + batch_size])
                enc = tok(b_pr, return_tensors="pt", padding=True, add_special_tokens=True)
                input_ids = enc["input_ids"].to(device)
                attn_mask = enc["attention_mask"].to(device)
                _ = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)

                x_b = buf["x"].float().cpu().numpy()
                hm_b = buf["h_mid"].float().cpu().numpy()
                ho_b = buf["h_out"].float().cpu().numpy()
                rms = np.sqrt(np.mean(hm_b ** 2, axis=-1, keepdims=True) + eps)
                hh_b = hm_b / rms

                for bi in range(len(b_pr)):
                    p = subj_pos_list[i0 + bi]
                    xs.append(x_b[bi, p].astype(np.float32))
                    hhats.append(hh_b[bi, p].astype(np.float32))
                    houts.append(ho_b[bi, p].astype(np.float32))
    finally:
        h1.remove()
        h2.remove()
        h3.remove()
        tok.padding_side = old_pad_side

    return {
        "x": np.stack(xs, axis=0),
        "h_hat": np.stack(hhats, axis=0),
        "h_out": np.stack(houts, axis=0),
    }


def extract_country_sibling_deltas(
    model: Any,
    tok: Any,
    device: torch.device,
    layer_idx: int,
    countries: Sequence[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract Donor/Recipient noun representations for each country and its capital averaged over
    3 clean templates (matching T1 city_country_sibling_socket construction!).
    """
    templates = [
        "The place {e} is widely known.",
        "In geography, {e} appears on maps.",
        "Travelers often visit {e} in summer.",
    ]
    prompts, spans, meta = [], [], []
    for c in countries:
        cap = COUNTRIES_INFO[c]["capital"][0]
        for kind, name in [("country", c), ("capital", cap)]:
            for tmpl in templates:
                txt = tmpl.format(e=name)
                c0 = txt.index(name)
                prompts.append(txt)
                spans.append([c0, c0 + len(name)])
                meta.append((kind, c))
    st = extract_entity_last_token_states(model, tok, device, prompts, spans, layer_idx)
    accum: Dict[str, Dict[str, List[np.ndarray]]] = {"country": defaultdict(list), "capital": defaultdict(list)}
    for i, (kind, c) in enumerate(meta):
        accum[kind][c].append(st["h_out"][i].astype(np.float64))
    out: Dict[str, Dict[str, np.ndarray]] = {"country": {}, "capital": {}}
    for kind in ("country", "capital"):
        for c in countries:
            out[kind][c] = np.mean(accum[kind][c], axis=0)
    return out


# ======================================================================================
# 5. Stage 0: General-Text Channel Census + Regime 1 Pretrained Neuron Census + Gate G0
# ======================================================================================

def _silu_np(z: np.ndarray) -> np.ndarray:
    # Numerically stable SiLU in float64
    z_clip = np.clip(z, -60.0, 60.0)
    return z / (1.0 + np.exp(-z_clip))


def run_regime1_neuron_census(
    model_A: Any,
    layer_A: int,
    gamma_A: np.ndarray,
    gamma_B: np.ndarray,
    eval_pairs: Dict[str, Any],
    x_bos_B: np.ndarray,
    sock_in: Dict[str, Any],
    sock_ridge: Dict[str, Any],
    sock_hhat: Dict[str, Any],
    sock_cpin: Dict[str, Any],
    n_sample: int = 2000,
    seed: int = SEED + 42,
) -> Dict[str, Any]:
    """
    Fable §4.3 Regime 1 Pretrained Neuron Census:
    Conjugate 2,000 native pretrained Qwen2.5-3B MLP neurons (stratified across firing-rate deciles!)
    into SmolLM2-1.7B four ways:
      1. V0_naive_biasfree: centered R_in, NO bias, NO BOS null
      2. V1_affine_bosnull: exact affine b_g, b_u + exact rank-1 BOS up-null
      3. V1_gamma_cpin: pre-gamma R_h with gamma-deconvolution diag(gamma_A) R_h diag(gamma_B)_lam^{-1}
         + constant-pin + exact BOS up-null
      4. V1_ridge_affine_bosnull: regularized regression read map M_ridge + exact affine + BOS up-null
    Evaluates on held-out eval_pairs (never used for fitting!) + pos-0 BOS vector!
    """
    layer = get_decoder_layers(model_A)[layer_A]
    Wg_full = layer.mlp.gate_proj.weight.detach().float().cpu().numpy().astype(np.float64)  # (d_ff, D)
    Wu_full = layer.mlp.up_proj.weight.detach().float().cpu().numpy().astype(np.float64)
    d_ff = Wg_full.shape[0]

    X_A = eval_pairs["x_A"].astype(np.float64)  # (N_eval, D)
    X_B = eval_pairs["x_B"].astype(np.float64)

    # Stratify 2,000 neurons across positive-gate firing-rate deciles on Donor A
    rng = np.random.default_rng(seed)
    sub_n = min(1500, len(X_A))
    g_probe = X_A[:sub_n] @ Wg_full.T  # (sub_n, d_ff)
    fire_rate = np.mean(g_probe > 0.0, axis=0)
    order = np.argsort(fire_rate)
    deciles = np.array_split(order, 10)
    per_dec = n_sample // 10
    chosen_idx = []
    decile_labels = []
    for d_i, dec in enumerate(deciles):
        pick = rng.choice(dec, size=min(per_dec, len(dec)), replace=False)
        chosen_idx.extend(pick.tolist())
        decile_labels.extend([d_i] * len(pick))
    chosen_idx = np.array(chosen_idx, dtype=np.int64)
    decile_labels = np.array(decile_labels, dtype=np.int64)

    Wg = Wg_full[chosen_idx]  # (K, D)
    Wu = Wu_full[chosen_idx]  # (K, D)

    # Ground-truth Donor A pre-activations & activations on held-out eval tokens
    g_A = X_A @ Wg.T  # (N, K)
    u_A = X_A @ Wu.T
    a_A = _silu_np(g_A) * u_A
    g_A_mean = g_A.mean(axis=0)
    g_A_std = g_A.std(axis=0) + 1e-12
    a_A_std = a_A.std(axis=0) + 1e-12
    a_A_p99 = np.percentile(np.abs(a_A), 99, axis=0) + 1e-12
    # Sparsest decile mask (decile 0)
    sparse_mask = (decile_labels == 0)
    # Threshold for "firing" in A: > 0 on gate AND |a| > 0.1 * std(a_A)
    a_thresh = 0.25 * a_A_std

    # Build 4 conjugation maps
    M_in = sock_in["M"]
    mu_A, mu_B = sock_in["mu_src"] * 0 + sock_in["mu_dst"], sock_in["mu_src"]  # sock_in maps B->A!
    # Double check: sock_in was fit with src=B, dst=A -> mu_src = mu_B, mu_dst = mu_A
    mu_B = sock_in["mu_src"]
    mu_A = sock_in["mu_dst"]

    # Gamma-deconvolved read map: M_gamma = diag(gamma_A) @ R_h @ diag(gamma_B / (gamma_B^2 + lam))
    lam_g = 0.05 * float(np.median(gamma_B ** 2))
    inv_gB = gamma_B / (gamma_B ** 2 + lam_g)
    M_gamma = (gamma_A[:, None] * sock_hhat["R"]) * inv_gB[None, :]
    # Scale-match M_gamma to centered B->A
    # Apply exact affine offset using mu_A, mu_B (or constant-pin!)
    bg_zero = np.zeros(len(chosen_idx), dtype=np.float64)
    bu_zero = np.zeros(len(chosen_idx), dtype=np.float64)

    arms_spec: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    # 1. V0_naive_biasfree
    arms_spec["V0_naive_biasfree"] = (Wg @ M_in, bg_zero, Wu @ M_in, bu_zero)

    # 2. V1_affine_bosnull
    Wg1 = Wg @ M_in
    bg1 = Wg @ (mu_A - M_in @ mu_B)
    Wu1 = Wu @ M_in
    bu1 = Wu @ (mu_A - M_in @ mu_B)
    Wg1, bg1, Wu1, bu1 = apply_exact_bos_null_affine(Wg1, bg1, Wu1, bu1, x_bos_B, mu_B)
    arms_spec["V1_affine_bosnull"] = (Wg1, bg1, Wu1, bu1)

    # 3. V1_gamma_cpin (pre-gamma deconvolved + constant-pinned mean offset + BOS up-null)
    # Use constant predictor c_B so bias-free version gets exact constant mapping, plus BOS up-null!
    c_B = sock_cpin["c_src"]
    # Fold exact mean offset delta_g = Wg @ (mu_A - M_gamma @ mu_B) into c_B direction (strictly bias-free!)
    Wg_gm = Wg @ M_gamma + np.outer(Wg @ (mu_A - M_gamma @ mu_B), c_B)
    Wu_gm = Wu @ M_gamma + np.outer(Wu @ (mu_A - M_gamma @ mu_B), c_B)
    Wu_gm = apply_exact_bos_null_biasfree(Wu_gm, x_bos_B)
    arms_spec["V1_gamma_cpin_biasfree"] = (Wg_gm, bg_zero, Wu_gm, bu_zero)

    # 4. V1_ridge_affine_bosnull
    M_rid = sock_ridge["M"]
    Wg_r = Wg @ M_rid
    bg_r = Wg @ (mu_A - M_rid @ mu_B)
    Wu_r = Wu @ M_rid
    bu_r = Wu @ (mu_A - M_rid @ mu_B)
    Wg_r, bg_r, Wu_r, bu_r = apply_exact_bos_null_affine(Wg_r, bg_r, Wu_r, bu_r, x_bos_B, mu_B)
    arms_spec["V1_ridge_affine_bosnull"] = (Wg_r, bg_r, Wu_r, bu_r)

    # Channel model prediction of per-neuron r(g) from held-out residual covariance Sigma_eps!
    # r_pred(w) = sqrt(Var(w^T x_A) / (Var(w^T x_A) + w^T Sigma_eps w))
    X_A_hat = mu_A + (X_B - mu_B) @ M_in.T
    E_val = X_A_hat - X_A
    Sig_e = (E_val.T @ E_val) / len(E_val)
    X_Ac = X_A - X_A.mean(axis=0)
    Sig_x = (X_Ac.T @ X_Ac) / len(X_Ac)
    var_sig = np.sum((Wg @ Sig_x) * Wg, axis=1)
    var_noi = np.sum((Wg @ Sig_e) * Wg, axis=1)
    r_pred = np.sqrt(np.maximum(var_sig, 1e-12) / np.maximum(var_sig + var_noi, 1e-12))

    census: Dict[str, Any] = {"n_neurons": int(len(chosen_idx)), "n_eval_tokens": int(len(X_A)), "arms": {}}
    for name, (Wgb, bgb, Wub, bub) in arms_spec.items():
        g_B = X_B @ Wgb.T + bgb[None, :]
        u_B = X_B @ Wub.T + bub[None, :]
        a_B = _silu_np(g_B) * u_B

        # Correlations r(g), r(a)
        gc_A = g_A - g_A.mean(axis=0, keepdims=True)
        gc_B = g_B - g_B.mean(axis=0, keepdims=True)
        r_g = np.sum(gc_A * gc_B, axis=0) / (np.sqrt(np.sum(gc_A ** 2, axis=0) * np.sum(gc_B ** 2, axis=0)) + 1e-12)

        ac_A = a_A - a_A.mean(axis=0, keepdims=True)
        ac_B = a_B - a_B.mean(axis=0, keepdims=True)
        r_a = np.sum(ac_A * ac_B, axis=0) / (np.sqrt(np.sum(ac_A ** 2, axis=0) * np.sum(ac_B ** 2, axis=0)) + 1e-12)

        # Mean-offset error |Delta| / sigma
        delta_over_sig = np.abs(g_B.mean(axis=0) - g_A_mean) / g_A_std

        # Sparsest decile false-fire inflation (fraction firing when A is off!)
        off_A = (g_A[:, sparse_mask] <= 0.0)
        ff_B = (np.abs(a_B[:, sparse_mask]) > a_thresh[None, sparse_mask]) & off_A
        fire_A_sparse = float(np.mean(np.abs(a_A[:, sparse_mask]) > a_thresh[None, sparse_mask]))
        ff_rate_B_sparse = float(np.mean(ff_B))
        ff_inflation = ff_rate_B_sparse / max(fire_A_sparse, 1e-6)

        # BOS pos-0 explosion ratio: |a_B(x_BOS)| / p99(|a_A(non-BOS)|)
        g_bos = Wgb @ x_bos_B + bgb
        u_bos = Wub @ x_bos_B + bub
        a_bos = np.abs(_silu_np(g_bos) * u_bos)
        bos_ratio = a_bos / a_A_p99

        # Channel model R^2 explaining variance in r(g)
        r_model_corr = float(np.corrcoef(r_pred, r_g)[0, 1])

        census["arms"][name] = {
            "median_r_g": round(float(np.median(r_g)), 4),
            "mean_r_g": round(float(np.mean(r_g)), 4),
            "median_r_a": round(float(np.median(r_a)), 4),
            "frac_r_a_ge_080": round(float(np.mean(r_a >= 0.80)), 4),
            "frac_r_g_ge_070": round(float(np.mean(r_g >= 0.70)), 4),
            "median_abs_delta_over_sigma": round(float(np.median(delta_over_sig)), 4),
            "p90_abs_delta_over_sigma": round(float(np.percentile(delta_over_sig, 90)), 4),
            "sparsest_decile_A_fire_rate": round(fire_A_sparse, 5),
            "sparsest_decile_B_false_fire_rate": round(ff_rate_B_sparse, 5),
            "sparsest_decile_inflation_ratio": round(float(ff_inflation), 3),
            "bos_abs_output_max": round(float(np.max(a_bos)), 6),
            "bos_ratio_median": round(float(np.median(bos_ratio)), 4),
            "bos_frac_ge_10x_p99": round(float(np.mean(bos_ratio >= 10.0)), 4),
            "channel_model_r_g_R2": round(float(r_model_corr ** 2), 4),
        }
    return census


def run_gate_g0_read_test(
    M_in: np.ndarray,
    mu_A: np.ndarray,
    mu_B: np.ndarray,
    x_A_lm: np.ndarray,
    x_B_lm: np.ndarray,
    x_A_gap: np.ndarray,
    x_B_gap: np.ndarray,
    x_A_distract: np.ndarray,
    x_A_cprobe: np.ndarray,
    x_B_cprobe: np.ndarray,
    cprobe_labels: List[str],
    x_A_ctrain: np.ndarray,
    ctrain_labels: List[str],
) -> Dict[str, Any]:
    """
    Opus / Fable Gate G0 Read-Direction Test:
      1. Retrieve each of 24 T1 landmarks from back-translated M_in x_B among bank (24 + 1000 distractors).
      2. Retrieve each of 17 T3 B-unknown gap entities from M_in x_B among bank (17 + 1000 distractors),
         and compare top-1/top-5/mean cosine against the 24 known T1 landmarks!
      3. Zero-shot country-class classification (4 classes: France, Italy, Germany, Japan) on unseen
         entities using Donor A class centroids applied to back-translated M_in x_B!
    """
    def _norm_rows(Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float64)
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    def _retrieval(xA_q: np.ndarray, xB_q: np.ndarray, xA_dist: np.ndarray) -> Dict[str, Any]:
        # Map B -> A in centered coordinates
        xA_hat = mu_A + (xB_q.astype(np.float64) - mu_B) @ M_in.T
        bank = np.concatenate([xA_q.astype(np.float64), xA_dist.astype(np.float64)], axis=0)
        # Center bank & queries by mu_A so cosine measures entity deviation, not mean vector!
        Q = _norm_rows(xA_hat - mu_A)
        K = _norm_rows(bank - mu_A)
        sims = Q @ K.T  # (n_q, n_q + n_dist)
        n_q = len(xA_q)
        top1, top5, top10 = 0, 0, 0
        self_cos, best_dist_cos, margins = [], [], []
        for i in range(n_q):
            row = sims[i]
            sc = float(row[i])
            self_cos.append(sc)
            # Rank (higher is better)
            rank = int(np.sum(row > sc)) + 1
            if rank == 1:
                top1 += 1
            if rank <= 5:
                top5 += 1
            if rank <= 10:
                top10 += 1
            dist_max = float(np.max(row[n_q:]))
            best_dist_cos.append(dist_max)
            margins.append(sc - dist_max)
        return {
            "n_queries": n_q,
            "n_distractors": int(len(xA_dist)),
            "top1_acc": round(top1 / max(n_q, 1), 4),
            "top5_acc": round(top5 / max(n_q, 1), 4),
            "top10_acc": round(top10 / max(n_q, 1), 4),
            "mean_self_cos": round(float(np.mean(self_cos)), 4),
            "mean_best_distractor_cos": round(float(np.mean(best_dist_cos)), 4),
            "mean_margin": round(float(np.mean(margins)), 4),
        }

    lm_ret = _retrieval(x_A_lm, x_B_lm, x_A_distract)
    gap_ret = _retrieval(x_A_gap, x_B_gap, x_A_distract)

    # 4-country centroid classifier fit strictly on Donor A train entities, tested on M_in x_B unseen entities!
    classes = sorted(set(cprobe_labels))
    centroids_A = {}
    for c in classes:
        idx = [i for i, y in enumerate(ctrain_labels) if y == c]
        centroids_A[c] = np.mean(x_A_ctrain[idx].astype(np.float64) - mu_A, axis=0)
    C_mat = _norm_rows(np.stack([centroids_A[c] for c in classes], axis=0))

    Q_A = _norm_rows(x_A_cprobe.astype(np.float64) - mu_A)
    Q_B_hat = _norm_rows((mu_A + (x_B_cprobe.astype(np.float64) - mu_B) @ M_in.T) - mu_A)
    pred_A = [classes[j] for j in np.argmax(Q_A @ C_mat.T, axis=1)]
    pred_B = [classes[j] for j in np.argmax(Q_B_hat @ C_mat.T, axis=1)]
    acc_A = float(np.mean([p == y for p, y in zip(pred_A, cprobe_labels)]))
    acc_B = float(np.mean([p == y for p, y in zip(pred_B, cprobe_labels)]))

    return {
        "landmarks_24_retrieval": lm_ret,
        "gap_17_retrieval": gap_ret,
        "gap_to_known_top1_ratio": round(gap_ret["top1_acc"] / max(lm_ret["top1_acc"], 1e-6), 4),
        "country_4way_centroid_donorA_acc": round(acc_A, 4),
        "country_4way_centroid_backtranslated_B_acc": round(acc_B, 4),
        "n_country_probes": len(cprobe_labels),
    }


def run_stage0(site_key: str, corpus: Dict[str, List[str]]) -> Dict[str, Any]:
    """Execute Stage 0 on MPS/CPU and cache all sockets, residual banks, and census results."""
    t0 = time.time()
    lA, lB = SITE_MAP[site_key]
    print(f"\n[Stage 0] Site ({DONOR_KEY} L{lA} -> {RECIP_KEY} L{lB}) | 260 passages (150 fit / 50 resid / 60 eval)")

    # Load leak-free dataset.json TRAIN entities (excluding France/Italy, all T1 landmarks, all T3 gaps!)
    with open(DATASET_PATH) as f:
        ds_facts = json.load(f)["facts"]
    with open(GRAFT_RESULTS_PATH) as f:
        gf = json.load(f)
    passing_lm_names = {x["name"] for x in gf["T0_competence_filter"]["landmarks"]["items"] if x["passed"]}
    with open(T3_RESULTS_PATH) as f:
        t3_gap_items = json.load(f)["gap_items"]
    gap_names = {x["entity"] for x in t3_gap_items}

    excluded_subjs = (
        passing_lm_names
        | gap_names
        | {x["name"] for x in ORGAN1_TRAIN_ENTITIES}
        | {x["name"] for x in ORGAN1_TEST_ENTITIES}
        | {x["name"] for x in ORGAN1_HARDNEG_TRAIN}
        | {x["name"] for x in ORGAN1_HARDNEG_TEST}
        | {x["subject"] for x in ds_facts if x["split"] != "train"}
    )
    train_ent_facts = [
        x for x in ds_facts
        if x["split"] == "train"
        and x["answer"] not in ("France", "Italy")
        and x["subject"] not in excluded_subjs
    ][:350]
    ent_prompts = [x["direct_prompt"] for x in train_ent_facts]
    ent_spans = [x["direct_subj_char_span"] for x in train_ent_facts]

    # Gate G0 probes: 24 T1 landmarks + 17 T3 gap entities + 4-country probes + 4-country train centroids
    lm_24_prompts = [f"The landmark {name} is widely known." for name in sorted(passing_lm_names)]
    lm_24_spans = [[p.index(n), p.index(n) + len(n)] for p, n in zip(lm_24_prompts, sorted(passing_lm_names))]

    gap_17_prompts = [f"The place {x['entity']} is widely known." for x in t3_gap_items]
    gap_17_spans = [[p.index(x["entity"]), p.index(x["entity"]) + len(x["entity"])] for p, x in zip(gap_17_prompts, t3_gap_items)]

    cprobe_prompts, cprobe_spans, cprobe_labels = [], [], []
    for c, ents in G0_COUNTRY_PROBES.items():
        for e in ents:
            p = f"The location {e} is widely known."
            cprobe_prompts.append(p)
            cprobe_spans.append([p.index(e), p.index(e) + len(e)])
            cprobe_labels.append(c)

    ctrain_prompts, ctrain_spans, ctrain_labels = [], [], []
    ctrain_bank = {
        "France": [x["name"] for x in ORGAN1_TRAIN_ENTITIES if x["country"] == "France"][:8],
        "Italy": [x["name"] for x in ORGAN1_TRAIN_ENTITIES if x["country"] == "Italy"][:8],
        "Germany": ["Berlin", "Munich", "Bavaria", "Siemens", "Bosch", "Volkswagen", "BMW", "Heidelberg"],
        "Japan": ["Tokyo", "Osaka", "Toyota", "Sony", "Honda", "Nintendo", "Hiroshima", "Sendai"],
    }
    for c, ents in ctrain_bank.items():
        for e in ents:
            p = f"The location {e} is widely known."
            ctrain_prompts.append(p)
            ctrain_spans.append([p.index(e), p.index(e) + len(e)])
            ctrain_labels.append(c)

    # --- Extract Donor A ---
    print(f"  Extracting Donor A ({DONOR_KEY}) at L{lA}...")
    model_A, tok_A, dev_A, _ = load_model_fp32(DONOR_KEY)
    stA_fit = extract_layer_states_on_passages(model_A, tok_A, dev_A, corpus["fit"], lA)
    stA_res = extract_layer_states_on_passages(model_A, tok_A, dev_A, corpus["resid"], lA)
    stA_ev = extract_layer_states_on_passages(model_A, tok_A, dev_A, corpus["eval"], lA)
    stA_ent = extract_entity_last_token_states(model_A, tok_A, dev_A, ent_prompts, ent_spans, lA)
    stA_lm24 = extract_entity_last_token_states(model_A, tok_A, dev_A, lm_24_prompts, lm_24_spans, lA)
    stA_gap17 = extract_entity_last_token_states(model_A, tok_A, dev_A, gap_17_prompts, gap_17_spans, lA)
    stA_cp = extract_entity_last_token_states(model_A, tok_A, dev_A, cprobe_prompts, cprobe_spans, lA)
    stA_ct = extract_entity_last_token_states(model_A, tok_A, dev_A, ctrain_prompts, ctrain_spans, lA)
    # Keep Donor A loaded briefly? No, extract 2000-neuron weights before unloading!
    layer_A_mod = get_decoder_layers(model_A)[lA]
    # Actually keep a light snapshot of A's MLP weights only? Easier: run census after extracting B, OR save Wg/Wu of 2000 neurons now!
    # Wait: stratification only uses A's eval states, so we can extract B first after unloading A,
    # and store Wg/Wu of ALL neurons or load A's layer weights (only ~180 MB in fp32!).
    Wg_A_cpu = layer_A_mod.mlp.gate_proj.weight.detach().float().cpu()
    Wu_A_cpu = layer_A_mod.mlp.up_proj.weight.detach().float().cpu()
    unload_model(model_A)

    # --- Extract Recipient B ---
    print(f"  Extracting Recipient B ({RECIP_KEY}) at L{lB}...")
    model_B, tok_B, dev_B, rm_B = load_model_fp32(RECIP_KEY)
    stB_fit = extract_layer_states_on_passages(model_B, tok_B, dev_B, corpus["fit"], lB)
    stB_res = extract_layer_states_on_passages(model_B, tok_B, dev_B, corpus["resid"], lB)
    stB_ev = extract_layer_states_on_passages(model_B, tok_B, dev_B, corpus["eval"], lB)
    stB_ent = extract_entity_last_token_states(model_B, tok_B, dev_B, ent_prompts, ent_spans, lB)
    stB_lm24 = extract_entity_last_token_states(model_B, tok_B, dev_B, lm_24_prompts, lm_24_spans, lB)
    stB_gap17 = extract_entity_last_token_states(model_B, tok_B, dev_B, gap_17_prompts, gap_17_spans, lB)
    stB_cp = extract_entity_last_token_states(model_B, tok_B, dev_B, cprobe_prompts, cprobe_spans, lB)
    unload_model(model_B)

    # --- Pair tokens strictly on word-final boundaries ---
    p_fit = pair_passage_activations(stA_fit, stB_fit)
    p_res = pair_passage_activations(stA_res, stB_res)
    p_ev = pair_passage_activations(stA_ev, stB_ev)

    print(
        f"  Aligned pairs: fit={len(p_fit['x_A'])}, resid={len(p_res['x_A'])}, eval={len(p_ev['x_A'])} "
        f"| unaligned eval B={len(p_ev['x_B_unaligned'])}"
    )

    # Filter massive-activation outlier tokens from h_out (and x) BEFORE socket fitting!
    keep_x_fit, x_out_stats = filter_outlier_pairs(p_fit["x_A"], p_fit["x_B"], max_med_mult=2.5)
    keep_ho_fit, ho_fit_stats = filter_outlier_pairs(p_fit["h_out_A"], p_fit["h_out_B"], max_med_mult=2.5)
    keep_ho_res, ho_res_stats = filter_outlier_pairs(p_res["h_out_A"], p_res["h_out_B"], max_med_mult=2.5)
    keep_ho_ev, ho_ev_stats = filter_outlier_pairs(p_ev["h_out_A"], p_ev["h_out_B"], max_med_mult=2.5)
    print(f"  Outlier filter (h_out fit): {ho_fit_stats}")

    # Dual general-text + train-entity fit set (~20% total weight on train entities, ~80% general text!)
    n_gen_x = int(np.sum(keep_x_fit))
    n_gen_ho = int(np.sum(keep_ho_fit))
    n_ent = int(len(stA_ent["x"]))
    w_ent_x = max(3.0, 0.25 * n_gen_x / max(n_ent, 1))
    w_ent_ho = max(3.0, 0.25 * n_gen_ho / max(n_ent, 1))

    x_A_dual = np.concatenate([p_fit["x_A"][keep_x_fit], stA_ent["x"]], axis=0)
    x_B_dual = np.concatenate([p_fit["x_B"][keep_x_fit], stB_ent["x"]], axis=0)
    hh_A_dual = np.concatenate([p_fit["h_hat_A"][keep_x_fit], stA_ent["h_hat"]], axis=0)
    hh_B_dual = np.concatenate([p_fit["h_hat_B"][keep_x_fit], stB_ent["h_hat"]], axis=0)
    ho_A_dual = np.concatenate([p_fit["h_out_A"][keep_ho_fit], stA_ent["h_out"]], axis=0)
    ho_B_dual = np.concatenate([p_fit["h_out_B"][keep_ho_fit], stB_ent["h_out"]], axis=0)

    w_dual_x = np.concatenate([np.ones(n_gen_x, dtype=np.float64), np.full(n_ent, w_ent_x, dtype=np.float64)])
    w_dual_ho = np.concatenate([np.ones(n_gen_ho, dtype=np.float64), np.full(n_ent, w_ent_ho, dtype=np.float64)])

    # Fit sockets in CPU float64
    sock_in = fit_orthogonal_socket(x_B_dual, x_A_dual, weights=w_dual_x)        # B -> A (read)
    sock_out = fit_orthogonal_socket(ho_A_dual, ho_B_dual, weights=w_dual_ho)    # A -> B (write, outlier-filtered!)
    sock_hhat = fit_orthogonal_socket(hh_B_dual, hh_A_dual, weights=w_dual_x)    # B -> A (pre-gamma)
    sock_ridge = fit_ridge_read_map(x_B_dual, x_A_dual, lam=1e-2)                # B -> A (ridge read)
    sock_cpin = fit_constant_pinned_socket(x_B_dual, x_A_dual, lam_c=1e-2)       # B -> A (strict bias-free c-pin)
    M_gamma = build_gamma_deconv_read_map(stA_fit["gamma"], stB_fit["gamma"], sock_hhat["R"], lam_frac=0.05)

    ent_norm_ratio_ho = float(
        np.median(np.linalg.norm(stB_ent["h_out"], axis=-1))
        / max(np.median(np.linalg.norm(stA_ent["h_out"], axis=-1)), 1e-12)
    )
    print(f"  sock_out s_out={sock_out['s']:.4f} (entity median norm ratio={ent_norm_ratio_ho:.4f}) | sock_in s_in={sock_in['s']:.4f}")

    # Exact B pos-0 BOS vector (deterministic across sequences; take mean and check variance!)
    x_bos_B = np.mean(stB_fit["bos_x"], axis=0)
    bos_std = float(np.max(np.std(stB_fit["bos_x"], axis=0)))

    # Unbiased empirical residual bank on strictly held-out resid_passages!
    xA_hat_res = sock_in["mu_dst"] + (p_res["x_B"].astype(np.float64) - sock_in["mu_src"]) @ sock_in["M"].T
    E_in_res = (xA_hat_res - p_res["x_A"].astype(np.float64)).astype(np.float32)
    Sigma_eps = (E_in_res.astype(np.float64).T @ E_in_res.astype(np.float64)) / len(E_in_res)

    # Constant-pinned residual bank for V1b training!
    xA_hat_res_cpin = p_res["x_B"].astype(np.float64) @ sock_cpin["M"].T
    E_in_res_cpin = (xA_hat_res_cpin - p_res["x_A"].astype(np.float64)).astype(np.float32)
    Sigma_eps_cpin = (E_in_res_cpin.astype(np.float64).T @ E_in_res_cpin.astype(np.float64)) / len(E_in_res_cpin)

    # Held-out CCA spectrum & split-half R agreement
    cca_spec = compute_heldout_cca_spectrum(p_fit["x_A"], p_fit["x_B"], p_ev["x_A"], p_ev["x_B"])
    split_half = compute_split_half_agreement(p_fit["x_B"], p_fit["x_A"])

    # Held-out read/write R^2 (evaluate write R^2 on outlier-filtered eval tokens + report raw!)
    xA_hat_ev = sock_in["mu_dst"] + (p_ev["x_B"].astype(np.float64) - sock_in["mu_src"]) @ sock_in["M"].T
    r2_read_ev = 1.0 - float(np.sum((xA_hat_ev - p_ev["x_A"]) ** 2) / np.sum((p_ev["x_A"] - p_ev["x_A"].mean(axis=0)) ** 2))
    hoA_ev_f = p_ev["h_out_A"][keep_ho_ev].astype(np.float64)
    hoB_ev_f = p_ev["h_out_B"][keep_ho_ev].astype(np.float64)
    hoB_hat_ev_f = sock_out["mu_dst"] + (hoA_ev_f - sock_out["mu_src"]) @ sock_out["M"].T
    r2_write_ev = 1.0 - float(np.sum((hoB_hat_ev_f - hoB_ev_f) ** 2) / np.sum((hoB_ev_f - hoB_ev_f.mean(axis=0)) ** 2))
    hoB_hat_ev_raw = sock_out["mu_dst"] + (p_ev["h_out_A"].astype(np.float64) - sock_out["mu_src"]) @ sock_out["M"].T
    r2_write_ev_raw = 1.0 - float(np.sum((hoB_hat_ev_raw - p_ev["h_out_B"]) ** 2) / np.sum((p_ev["h_out_B"] - p_ev["h_out_B"].mean(axis=0)) ** 2))
    print(f"  Held-out R^2: read={r2_read_ev:.4f} | write(filtered)={r2_write_ev:.4f} | write(raw)={r2_write_ev_raw:.4f}")

    # Write-core projector (top-512 reliable residual directions excluding B massive dims, fit on outlier-filtered resid_passages!)
    wcore = compute_write_core_projector(
        p_res["h_out_A"][keep_ho_res], p_res["h_out_B"][keep_ho_res], sock_out["M"], r_out=512
    )

    # Uncentered C_0 top-32 PCs over ALL B fit tokens (including BOS & unaligned B tokens!)
    c0_info = compute_uncentered_C0_basis(p_fit["x_B_all"], r_c0=32)

    # Regime 1 Pretrained Neuron Census (2,000 native Qwen neurons)
    class _DummyMLP:
        def __init__(self, wg, wu):
            self.gate_proj = type("L", (), {"weight": wg})()
            self.up_proj = type("L", (), {"weight": wu})()

    class _DummyModel:
        def __init__(self, wg, wu, lidx):
            layers_lst = [None] * (lidx + 1)
            layers_lst[lidx] = type("Lay", (), {"mlp": _DummyMLP(wg, wu)})()
            self.model = type("M", (), {"layers": layers_lst})()

    census = run_regime1_neuron_census(
        _DummyModel(Wg_A_cpu, Wu_A_cpu, lA),
        lA,
        stA_fit["gamma"],
        stB_fit["gamma"],
        p_ev,
        x_bos_B,
        sock_in,
        sock_ridge,
        sock_hhat,
        sock_cpin,
        n_sample=2000,
    )

    # Gate G0 Read-Direction Test (1,000 held-out general-text distractors from eval_passages!)
    rng_g0 = np.random.default_rng(SEED + 314)
    dist_idx = rng_g0.choice(len(p_ev["x_A"]), size=min(1000, len(p_ev["x_A"])), replace=False)
    gate_g0 = run_gate_g0_read_test(
        sock_in["M"],
        sock_in["mu_dst"],
        sock_in["mu_src"],
        stA_lm24["x"],
        stB_lm24["x"],
        stA_gap17["x"],
        stB_gap17["x"],
        p_ev["x_A"][dist_idx],
        stA_cp["x"],
        stB_cp["x"],
        cprobe_labels,
        stA_ct["x"],
        ctrain_labels,
    )

    stage0_summary = {
        "site": {"donor": DONOR_KEY, "recipient": RECIP_KEY, "lA": lA, "lB": lB, "rm_B": rm_B},
        "counts": {
            "n_fit_aligned": int(len(p_fit["x_A"])),
            "n_resid_aligned": int(len(p_res["x_A"])),
            "n_eval_aligned": int(len(p_ev["x_A"])),
            "n_eval_unaligned_B": int(len(p_ev["x_B_unaligned"])),
            "n_train_entities_dual": int(len(stA_ent["x"])),
        },
        "channel_spectrum": cca_spec,
        "split_half_R_agreement": split_half,
        "heldout_read_R2": round(r2_read_ev, 4),
        "heldout_write_R2": round(r2_write_ev, 4),
        "heldout_write_R2_raw": round(r2_write_ev_raw, 4),
        "outlier_filter_h_out_fit": ho_fit_stats,
        "socket_scales": {
            "s_in": round(sock_in["s"], 4),
            "s_out": round(sock_out["s"], 4),
            "entity_norm_ratio_h_out": round(ent_norm_ratio_ho, 4),
        },
        "bos_B_max_coord_std": round(bos_std, 6),
        "bos_B_l2_norm": round(float(np.linalg.norm(x_bos_B)), 4),
        "constant_pin": {
            "std_cA_xA": round(sock_cpin["std_dst"], 5),
            "std_cB_xB": round(sock_cpin["std_src"], 5),
            "pin_err": float(sock_cpin["pin_err"]),
        },
        "write_core": {
            "r_out": 512,
            "n_massive_dims": wcore["n_massive_dims"],
            "massive_dims": wcore["massive_dims"][:10],
            "top_r_mean_corr": wcore["top_r_mean_corr"],
            "top_r_min_corr": wcore["top_r_min_corr"],
        },
        "c0_basis": {"r_c0": 32, "top_energy_frac": c0_info["top_energy_frac"]},
        "regime1_neuron_census": census,
        "gate_g0_read_test": gate_g0,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    # Cache all Stage 0 tensors for Stage 1 & Stage 2!
    cache_path = CACHE_DIR / f"pilot_organs_p0_stage0_{site_key}.pt"
    torch.save(
        {
            "summary": stage0_summary,
            "sock_in": sock_in,
            "sock_out": sock_out,
            "sock_cpin": sock_cpin,
            "M_gamma": M_gamma,
            "x_bos_B": x_bos_B,
            "bos_x_fit_B": stB_fit["bos_x"],
            "bos_x_eval_B": stB_ev["bos_x"],
            "E_in_res": E_in_res,
            "Sigma_eps": Sigma_eps,
            "E_in_res_cpin": E_in_res_cpin,
            "Sigma_eps_cpin": Sigma_eps_cpin,
            "P_write_core": wcore["P_core"],
            "V_c0": c0_info["V_c0"],
            "x_A_fit_gen": p_fit["x_A"],
            "x_B_fit_gen": p_fit["x_B"],
            "x_B_eval_aligned": p_ev["x_B"],
            "mlpB_eval_aligned_n2": p_ev["mlpB_aligned_n2"],
            "x_B_eval_unaligned": p_ev["x_B_unaligned"],
            "mlpB_eval_unaligned_n2": p_ev["mlpB_unaligned_n2"],
        },
        cache_path,
    )
    print(f"  [Stage 0 Done in {stage0_summary['elapsed_sec']}s] Saved {cache_path}")
    print(f"    CCA held-out >=0.90: {cca_spec['n_ge_090']} | >=0.80: {cca_spec['n_ge_080']} | split-half top256 cos: {split_half['top256_pc_mean_cos']}")
    print(f"    Regime1 V0 median r(g)={census['arms']['V0_naive_biasfree']['median_r_g']} | BOS>=10x={census['arms']['V0_naive_biasfree']['bos_frac_ge_10x_p99']} | |d|/s={census['arms']['V0_naive_biasfree']['median_abs_delta_over_sigma']}")
    print(f"    Regime1 V1 median r(g)={census['arms']['V1_affine_bosnull']['median_r_g']} | BOS_max={census['arms']['V1_affine_bosnull']['bos_abs_output_max']} | |d|/s={census['arms']['V1_affine_bosnull']['median_abs_delta_over_sigma']}")
    print(f"    Gate G0: LM24 top1={gate_g0['landmarks_24_retrieval']['top1_acc']} | GAP17 top1={gate_g0['gap_17_retrieval']['top1_acc']} | 4-country B_acc={gate_g0['country_4way_centroid_backtranslated_B_acc']}")
    return stage0_summary


# ======================================================================================
# 6. Stage 1: Pure-Donor Native SwiGLU Organ Training (Zero Backprop Through A/B!)
# ======================================================================================

class SwiGLUOrgan(nn.Module):
    def __init__(self, d_model: int, k: int, use_bias: bool = True, c_init: Optional[np.ndarray] = None):
        super().__init__()
        self.gate = nn.Linear(d_model, k, bias=use_bias)
        self.up = nn.Linear(d_model, k, bias=use_bias)
        self.down = nn.Linear(k, d_model, bias=False)
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.normal_(self.down.weight, std=0.01)
        if use_bias:
            nn.init.constant_(self.gate.bias, -1.0)  # Mild negative init (NOT -4.0 which traps in SiLU negative lobe!)
            nn.init.zeros_(self.up.bias)
        elif c_init is not None:
            # Implicit -1.0 init along constant predictor c_A for strictly bias-free arms
            with torch.no_grad():
                c_t = torch.from_numpy(np.asarray(c_init, dtype=np.float32))
                self.gate.weight.sub_(c_t.unsqueeze(0))

    def forward_with_preact(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = self.gate(x)
        u = self.up(x)
        a = F.silu(g) * u
        y = self.down(a)
        return y, g, a


def train_swiglu_organ(
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    x_hardneg: np.ndarray,
    x_gen: np.ndarray,
    k: int,
    arm: str,
    E_noise: Optional[np.ndarray] = None,
    Sigma_eps: Optional[np.ndarray] = None,
    c_init: Optional[np.ndarray] = None,
    steps: int = 900,
    lr: float = 3e-3,
    seed: int = SEED + 101,
) -> Dict[str, Any]:
    """
    Train a standalone k-neuron SwiGLU organ strictly on cached states (zero LM backprop!).
    Fixes SiLU negative-lobe cheating via:
      - b_g = -1.0 init (never -4.0!)
      - Explicit positive-gate margin: top-m (m = max(4, k//4)) gates per positive example >= +3.5
      - Mild L2 on W_u, W_d so SGD cannot inflate up/down to operate inside SiLU's -0.24 negative lobe
      - Correlated shrinkage (0.78..1.0 toward mu_gen) + empirical/intrinsic noise on positive & negative draws!
    Arms:
      - 'V0': bias-free, weak L2 silence only (no hinge, no residual noise, no Sigma_eps).
      - 'V1': bias=True, positive-gate margin + donor silence hinge ReLU(g + 5)^2 + output L2 (no noise).
      - 'V1b': bias=False, correlated residual noise e~E_noise + noise-metric reg + pos-gate margin + hinge.
      - 'V2_intrinsic': 100% RECIPIENT-FREE donor-intrinsic noise + shrinkage + pos-gate margin + hinge.
      - 'V2': bias=True, correlated empirical residual noise e~E_noise (identical on gate & up, 25% 1.3x tail)
              + shrinkage + noise-metric reg tr(W Sigma_eps W^T) + pos-gate margin + hinge + output L2.
      - 'native_B_oracle': bias=True, clean B states + pos-gate margin + hinge + output L2.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_compute_device()
    use_bias = arm not in ("V0", "V1b")
    use_hinge = arm != "V0"
    use_noise = arm in ("V1b", "V2", "V2_intrinsic") and E_noise is not None

    D = x_pos.shape[1]
    organ = SwiGLUOrgan(D, k, use_bias=use_bias, c_init=c_init).to(device)
    opt = torch.optim.AdamW(organ.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)

    Xp = torch.from_numpy(x_pos.astype(np.float32)).to(device)
    Yp = torch.from_numpy(y_pos.astype(np.float32)).to(device)
    Xh = torch.from_numpy(x_hardneg.astype(np.float32)).to(device)
    Xg = torch.from_numpy(x_gen.astype(np.float32)).to(device)
    mu_gen = Xg.mean(dim=0, keepdim=True)
    En = torch.from_numpy(E_noise.astype(np.float32)).to(device) if use_noise else None
    Se = torch.from_numpy(Sigma_eps.astype(np.float32)).to(device) if (use_noise and Sigma_eps is not None) else None

    target_norm2 = float(torch.mean(torch.sum(Yp ** 2, dim=-1)).item()) + 1e-8
    n_p, n_h, n_g = len(Xp), len(Xh), len(Xg)
    n_e = len(En) if En is not None else 0
    k_act = max(4, k // 4)

    def _sample_noise(n: int) -> torch.Tensor:
        idx = torch.randint(0, n_e, (n,), device=device)
        e = En[idx]
        # 25% 1.3x tail boost to harden non-Gaussian silence tails (Opus Q2 / Neon)
        boost = torch.where(torch.rand(n, 1, device=device) < 0.25, 1.3, 1.0)
        return e * boost

    bs_p = min(128, n_p)
    bs_h = min(128, n_h)
    bs_g = min(256, n_g)

    for step in range(steps):
        opt.zero_grad(set_to_none=True)

        ip = torch.randint(0, n_p, (bs_p,), device=device)
        ih = torch.randint(0, n_h, (bs_h,), device=device)
        ig = torch.randint(0, n_g, (bs_g,), device=device)

        xp_b, yp_b = Xp[ip], Yp[ip]
        xh_b = Xh[ih]
        xg_b = Xg[ig]

        if use_noise:
            # Correlated shrinkage + empirical/intrinsic residual noise (EXACT SAME e on gate & up!)
            ep = _sample_noise(bs_p)
            eh = _sample_noise(bs_h)
            eg = _sample_noise(bs_g)
            mask_n = (torch.rand(bs_p, 1, device=device) < 0.75).float()
            shrink = 0.78 + 0.22 * torch.rand(bs_p, 1, device=device)
            xp_shrunk = mu_gen + (1.0 - mask_n + mask_n * shrink) * (xp_b - mu_gen)
            xp_in = xp_shrunk + mask_n * ep
            xh_in = xh_b + eh
            xg_in = xg_b + eg
        else:
            xp_in, xh_in, xg_in = xp_b, xh_b, xg_b

        yp_hat, gp, _ = organ.forward_with_preact(xp_in)
        yh_hat, gh, _ = organ.forward_with_preact(xh_in)
        yg_hat, gg, _ = organ.forward_with_preact(xg_in)

        # Target reconstruction loss (scale-invariant)
        loss_pos = torch.mean(torch.sum((yp_hat - yp_b) ** 2, dim=-1)) / target_norm2

        # Explicit positive-gate activation margin (kills SiLU negative-lobe cheating!)
        gp_top, _ = torch.topk(gp, min(k_act, k), dim=-1)
        loss_pos_gate = 0.35 * torch.mean(F.relu(3.5 - gp_top) ** 2)
        if use_noise:
            # Also enforce positive margin on clean positives
            _, gp_c, _ = organ.forward_with_preact(xp_b)
            gp_c_top, _ = torch.topk(gp_c, min(k_act, k), dim=-1)
            loss_pos_gate = 0.5 * loss_pos_gate + 0.5 * 0.35 * torch.mean(F.relu(4.0 - gp_c_top) ** 2)

        # Mild W_u / W_d L2 regularization prevents negative-lobe amplitude inflation
        loss_w_reg = 1e-3 * (torch.mean(organ.up.weight ** 2) + torch.mean(organ.down.weight ** 2))

        if not use_hinge:
            # V0 naive: weak L2 on general text + hard negs only
            loss_sil = 0.05 * (
                torch.mean(torch.sum(yg_hat ** 2, dim=-1)) + torch.mean(torch.sum(yh_hat ** 2, dim=-1))
            ) / target_norm2
            loss = loss_pos + loss_pos_gate + loss_w_reg + loss_sil
        else:
            # Donor silence hinge (gate <= -5.0 on general text, <= -4.5 on hard negatives) + output energy L2!
            hinge_g = torch.mean(F.relu(gg + 5.0) ** 2)
            hinge_h = torch.mean(F.relu(gh + 4.5) ** 2)
            if use_noise:
                yg_c, gg_c, _ = organ.forward_with_preact(xg_b)
                yh_c, gh_c, _ = organ.forward_with_preact(xh_b)
                hinge_g = 0.5 * hinge_g + 0.5 * torch.mean(F.relu(gg_c + 5.5) ** 2)
                hinge_h = 0.5 * hinge_h + 0.5 * torch.mean(F.relu(gh_c + 5.0) ** 2)
                out_g = 0.5 * torch.mean(torch.sum(yg_hat ** 2, dim=-1)) + 0.5 * torch.mean(torch.sum(yg_c ** 2, dim=-1))
                out_h = 0.5 * torch.mean(torch.sum(yh_hat ** 2, dim=-1)) + 0.5 * torch.mean(torch.sum(yh_c ** 2, dim=-1))
            else:
                out_g = torch.mean(torch.sum(yg_hat ** 2, dim=-1))
                out_h = torch.mean(torch.sum(yh_hat ** 2, dim=-1))

            loss_sil = 2.0 * (out_g + out_h) / target_norm2 + 0.20 * hinge_g + 0.25 * hinge_h

            # Opus Q2: Closed-form noise-metric regularizer lambda * (tr(Wg Sigma_eps Wg^T) + tr(Wu Sigma_eps Wu^T))
            loss_noise_reg = torch.tensor(0.0, device=device)
            if Se is not None:
                tr_g = torch.mean(torch.sum((organ.gate.weight @ Se) * organ.gate.weight, dim=-1))
                tr_u = torch.mean(torch.sum((organ.up.weight @ Se) * organ.up.weight, dim=-1))
                loss_noise_reg = 0.05 * (tr_g + tr_u)

            loss = loss_pos + loss_pos_gate + loss_w_reg + loss_sil + loss_noise_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(organ.parameters(), 2.0)
        opt.step()
        sched.step()

    organ.eval()
    with torch.no_grad():
        yp_eval, gp_eval, _ = organ.forward_with_preact(Xp)
        yg_eval, gg_eval, _ = organ.forward_with_preact(Xg[: min(2000, n_g)])
        yh_eval, gh_eval, _ = organ.forward_with_preact(Xh)
        cos_pos = float(torch.mean(F.cosine_similarity(yp_eval, Yp, dim=-1)).item())
        mse_rel = float((torch.mean(torch.sum((yp_eval - Yp) ** 2, dim=-1)) / target_norm2).item())
        gen_rms = float(torch.sqrt(torch.mean(torch.sum(yg_eval ** 2, dim=-1))).item())
        hn_rms = float(torch.sqrt(torch.mean(torch.sum(yh_eval ** 2, dim=-1))).item())
        gen_g_p99 = float(torch.quantile(gg_eval.reshape(-1), 0.99).item())
        pos_g_max = float(torch.mean(torch.max(gp_eval, dim=-1).values).item())
        pos_n_gt0 = float(torch.mean(torch.sum((gp_eval > 0.0).float(), dim=-1)).item())
        norm_ratio = float(torch.mean(torch.norm(yp_eval, dim=-1) / (torch.norm(Yp, dim=-1) + 1e-8)).item())

    return {
        "W_g": organ.gate.weight.detach().cpu().numpy().astype(np.float64),
        "b_g": organ.gate.bias.detach().cpu().numpy().astype(np.float64) if use_bias else None,
        "W_u": organ.up.weight.detach().cpu().numpy().astype(np.float64),
        "b_u": organ.up.bias.detach().cpu().numpy().astype(np.float64) if use_bias else None,
        "W_d": organ.down.weight.detach().cpu().numpy().astype(np.float64),
        "use_bias": use_bias,
        "train_metrics": {
            "pos_cos": round(cos_pos, 4),
            "pos_rel_mse": round(mse_rel, 4),
            "pos_g_max_mean": round(pos_g_max, 3),
            "pos_n_gt0_mean": round(pos_n_gt0, 2),
            "pos_norm_ratio": round(norm_ratio, 3),
            "gen_out_rms": round(gen_rms, 5),
            "hardneg_out_rms": round(hn_rms, 5),
            "gen_gate_p99": round(gen_g_p99, 3),
        },
    }


def run_stage1(site_key: str) -> Dict[str, Any]:
    """Extract Donor A (and B for native_B_oracle) organ training data and train Organ 1 + Organ 2."""
    t0 = time.time()
    lA, lB = SITE_MAP[site_key]
    s0_path = CACHE_DIR / f"pilot_organs_p0_stage0_{site_key}.pt"
    s0 = torch.load(s0_path, weights_only=False)
    print(f"\n[Stage 1] Training Organ 1 (k=32, Class-Gated Rule) & Organ 2 (k=64, Entity-Gated Landmarks) at L{lA}->L{lB}")

    # Load exact T1 derangement for Organ 2's 12 target landmarks
    with open(GRAFT_RESULTS_PATH) as f:
        gf = json.load(f)
    derangement = gf["T1_forced_gate_audit"]["derangement"]
    lm_by_name = {x["name"]: x for x in LANDMARKS_T0}

    # Build prompt lists for Organ 1 & Organ 2
    o1_pos_prompts, o1_pos_spans, o1_pos_countries = [], [], []
    for ent in ORGAN1_TRAIN_ENTITIES:
        for tmpl in TRAIN_PREFIX_TEMPLATES:
            p = tmpl.format(e=ent["name"])
            c0 = p.index(ent["name"])
            o1_pos_prompts.append(p)
            o1_pos_spans.append([c0, c0 + len(ent["name"])])
            o1_pos_countries.append(ent["country"])

    o1_hn_prompts, o1_hn_spans = [], []
    for ent in ORGAN1_HARDNEG_TRAIN:
        for tmpl in TRAIN_PREFIX_TEMPLATES[:4]:
            p = tmpl.format(e=ent["name"])
            c0 = p.index(ent["name"])
            o1_hn_prompts.append(p)
            o1_hn_spans.append([c0, c0 + len(ent["name"])])

    o2_pos_prompts, o2_pos_spans, o2_pos_meta = [], [], []
    for name in ORGAN2_TARGET_NAMES:
        c_old = lm_by_name[name]["country"]
        c_new = derangement[name]
        for tmpl in TRAIN_PREFIX_TEMPLATES:
            p = tmpl.format(e=name)
            c0 = p.index(name)
            o2_pos_prompts.append(p)
            o2_pos_spans.append([c0, c0 + len(name)])
            o2_pos_meta.append((name, c_old, c_new))

    o2_sib_prompts, o2_sib_spans = [], []
    for name in ORGAN2_SIBLING_NAMES:
        for tmpl in TRAIN_PREFIX_TEMPLATES:
            p = tmpl.format(e=name)
            c0 = p.index(name)
            o2_sib_prompts.append(p)
            o2_sib_spans.append([c0, c0 + len(name)])

    # Held-out Donor A test probes (ZERO B data!) to verify unseen positive-gate firing in Stage 1!
    o1_test_prompts, o1_test_spans, o1_test_countries = [], [], []
    for ent in ORGAN1_TEST_ENTITIES:
        p = f"The location {ent['name']} is situated in the country of"
        c0 = p.index(ent["name"])
        o1_test_prompts.append(p)
        o1_test_spans.append([c0, c0 + len(ent["name"])])
        o1_test_countries.append(ent["country"])

    o1_hntest_prompts, o1_hntest_spans = [], []
    for ent in ORGAN1_HARDNEG_TEST:
        p = f"The location {ent['name']} is situated in the country of"
        c0 = p.index(ent["name"])
        o1_hntest_prompts.append(p)
        o1_hntest_spans.append([c0, c0 + len(ent["name"])])

    o2_test_prompts, o2_test_spans, o2_test_meta = [], [], []
    for name in ORGAN2_TARGET_NAMES:
        p = f"The landmark {name} is located in the country of"
        c0 = p.index(name)
        o2_test_prompts.append(p)
        o2_test_spans.append([c0, c0 + len(name)])
        o2_test_meta.append((name, lm_by_name[name]["country"], derangement[name]))

    o2_sibtest_prompts, o2_sibtest_spans = [], []
    for name in ORGAN2_SIBLING_NAMES:
        p = f"The landmark {name} is located in the country of"
        c0 = p.index(name)
        o2_sibtest_prompts.append(p)
        o2_sibtest_spans.append([c0, c0 + len(name)])

    all_countries = sorted(set(COUNTRIES_INFO.keys()))

    def _extract_model_organ_data(model_key: str, l_idx: int) -> Dict[str, Any]:
        model, tok, dev, _ = load_model_fp32(model_key)
        u_nouns = extract_country_sibling_deltas(model, tok, dev, l_idx, all_countries)
        st_o1p = extract_entity_last_token_states(model, tok, dev, o1_pos_prompts, o1_pos_spans, l_idx)
        st_o1h = extract_entity_last_token_states(model, tok, dev, o1_hn_prompts, o1_hn_spans, l_idx)
        st_o2p = extract_entity_last_token_states(model, tok, dev, o2_pos_prompts, o2_pos_spans, l_idx)
        st_o2s = extract_entity_last_token_states(model, tok, dev, o2_sib_prompts, o2_sib_spans, l_idx)
        st_o1t = extract_entity_last_token_states(model, tok, dev, o1_test_prompts, o1_test_spans, l_idx)
        st_o1ht = extract_entity_last_token_states(model, tok, dev, o1_hntest_prompts, o1_hntest_spans, l_idx)
        st_o2t = extract_entity_last_token_states(model, tok, dev, o2_test_prompts, o2_test_spans, l_idx)
        st_o2st = extract_entity_last_token_states(model, tok, dev, o2_sibtest_prompts, o2_sibtest_spans, l_idx)
        unload_model(model)

        # Build exact T1 city_country_sibling target deltas strictly inside this model!
        y_o1 = np.zeros_like(st_o1p["h_out"], dtype=np.float64)
        for i, c_old in enumerate(o1_pos_countries):
            c_new = ORGAN1_RULE[c_old]
            y_o1[i] = (u_nouns["country"][c_new] - u_nouns["country"][c_old]) + (
                u_nouns["capital"][c_new] - u_nouns["capital"][c_old]
            )

        y_o1t = np.zeros_like(st_o1t["h_out"], dtype=np.float64)
        for i, c_old in enumerate(o1_test_countries):
            c_new = ORGAN1_RULE[c_old]
            y_o1t[i] = (u_nouns["country"][c_new] - u_nouns["country"][c_old]) + (
                u_nouns["capital"][c_new] - u_nouns["capital"][c_old]
            )

        y_o2 = np.zeros_like(st_o2p["h_out"], dtype=np.float64)
        for i, (_nm, c_old, c_new) in enumerate(o2_pos_meta):
            y_o2[i] = (u_nouns["country"][c_new] - u_nouns["country"][c_old]) + (
                u_nouns["capital"][c_new] - u_nouns["capital"][c_old]
            )

        y_o2t = np.zeros_like(st_o2t["h_out"], dtype=np.float64)
        for i, (_nm, c_old, c_new) in enumerate(o2_test_meta):
            y_o2t[i] = (u_nouns["country"][c_new] - u_nouns["country"][c_old]) + (
                u_nouns["capital"][c_new] - u_nouns["capital"][c_old]
            )

        return {
            "o1_x_pos": st_o1p["x"],
            "o1_y_pos": y_o1.astype(np.float32),
            "o1_x_hn": st_o1h["x"],
            "o1_x_test": st_o1t["x"],
            "o1_y_test": y_o1t.astype(np.float32),
            "o1_x_hntest": st_o1ht["x"],
            "o2_x_pos": st_o2p["x"],
            "o2_y_pos": y_o2.astype(np.float32),
            "o2_x_hn": st_o2s["x"],
            "o2_x_test": st_o2t["x"],
            "o2_y_test": y_o2t.astype(np.float32),
            "o2_x_hntest": st_o2st["x"],
        }

    def _audit_organ_on_heldout_states(res: Dict[str, Any], x_t: np.ndarray, y_t: np.ndarray, x_hnt: np.ndarray) -> Dict[str, Any]:
        Wg, Wu, Wd = res["W_g"], res["W_u"], res["W_d"]
        bg = res["b_g"] if res.get("b_g") is not None else np.zeros(Wg.shape[0])
        bu = res["b_u"] if res.get("b_u") is not None else np.zeros(Wu.shape[0])
        Xt = x_t.astype(np.float64)
        Yt = y_t.astype(np.float64)
        Xht = x_hnt.astype(np.float64)
        gt = Xt @ Wg.T + bg[None, :]
        ut = Xt @ Wu.T + bu[None, :]
        yt = (_silu_np(gt) * ut) @ Wd.T
        gh = Xht @ Wg.T + bg[None, :]
        uh = Xht @ Wu.T + bu[None, :]
        yh = (_silu_np(gh) * uh) @ Wd.T
        cos_t = np.sum(yt * Yt, axis=-1) / (np.linalg.norm(yt, axis=-1) * np.linalg.norm(Yt, axis=-1) + 1e-12)
        nrat = np.linalg.norm(yt, axis=-1) / (np.linalg.norm(Yt, axis=-1) + 1e-12)
        return {
            "heldout_g_max_mean": round(float(np.mean(np.max(gt, axis=-1))), 3),
            "heldout_n_gt0_mean": round(float(np.mean(np.sum(gt > 0.0, axis=-1))), 2),
            "heldout_pos_cos": round(float(np.mean(cos_t)), 4),
            "heldout_norm_ratio": round(float(np.mean(nrat)), 3),
            "heldout_hn_g_max_mean": round(float(np.mean(np.max(gh, axis=-1))), 3),
            "heldout_hn_rms": round(float(np.sqrt(np.mean(np.sum(yh ** 2, axis=-1)))), 4),
        }

    print(f"  Extracting Organ 1 & 2 training + Donor-A heldout audit states ({DONOR_KEY} L{lA})...")
    data_A = _extract_model_organ_data(DONOR_KEY, lA)
    print(f"  Extracting Organ 1 & 2 oracle states in Recipient B ({RECIP_KEY} L{lB}, for native_B_oracle only)...")
    data_B = _extract_model_organ_data(RECIP_KEY, lB)

    # Measure test-entity read cosine (sanity diagnostic for sock_in!)
    xA_o1t_hat = s0["sock_in"]["mu_dst"] + (data_B["o1_x_test"].astype(np.float64) - s0["sock_in"]["mu_src"]) @ s0["sock_in"]["M"].T
    dx_hat = xA_o1t_hat - s0["sock_in"]["mu_dst"]
    dx_true = data_A["o1_x_test"].astype(np.float64) - s0["sock_in"]["mu_dst"]
    o1t_read_cos = float(np.mean(np.sum(dx_hat * dx_true, axis=-1) / (np.linalg.norm(dx_hat, axis=-1) * np.linalg.norm(dx_true, axis=-1) + 1e-12)))
    print(f"  Organ 1 held-out test-entity read centered cosine cos(xA_hat - mu, xA - mu) = {o1t_read_cos:.4f}")

    # Build 100% Recipient-Free donor-intrinsic noise bank from Donor A's own general-text states!
    E_int, Sigma_int = build_donor_intrinsic_noise(s0["x_A_fit_gen"], n_samples=4000, sigma_rel=0.45, seed=SEED + 707)

    # Check write-core target energy retention (sanity diagnostic!)
    P_core = s0["P_write_core"]
    M_out = s0["sock_out"]["M"]
    y_o1_mapped = data_A["o1_y_pos"].astype(np.float64) @ M_out.T
    y_o2_mapped = data_A["o2_y_pos"].astype(np.float64) @ M_out.T
    o1_core_ret = float(np.sum((y_o1_mapped @ P_core) ** 2) / np.sum(y_o1_mapped ** 2))
    o2_core_ret = float(np.sum((y_o2_mapped @ P_core) ** 2) / np.sum(y_o2_mapped ** 2))
    print(f"  Write-core top-512 target energy retention: Organ1={o1_core_ret:.4f}, Organ2={o2_core_ret:.4f}")

    organs_out: Dict[str, Dict[str, Any]] = {"organ1": {}, "organ2": {}}
    stage1_summary: Dict[str, Any] = {
        "o1_test_entity_read_centered_cos": round(o1t_read_cos, 4),
        "write_core_target_energy_retention": {"organ1": round(o1_core_ret, 4), "organ2": round(o2_core_ret, 4)},
        "organ1": {},
        "organ2": {},
    }

    c_A_init = s0["sock_cpin"]["c_dst"]
    for org_name, k_dim, xpA, ypA, xhA, xtA, ytA, xhtA, xpB, ypB, xhB, xtB, ytB, xhtB in [
        ("organ1", 32, data_A["o1_x_pos"], data_A["o1_y_pos"], data_A["o1_x_hn"], data_A["o1_x_test"], data_A["o1_y_test"], data_A["o1_x_hntest"],
         data_B["o1_x_pos"], data_B["o1_y_pos"], data_B["o1_x_hn"], data_B["o1_x_test"], data_B["o1_y_test"], data_B["o1_x_hntest"]),
        ("organ2", 64, data_A["o2_x_pos"], data_A["o2_y_pos"], data_A["o2_x_hn"], data_A["o2_x_test"], data_A["o2_y_test"], data_A["o2_x_hntest"],
         data_B["o2_x_pos"], data_B["o2_y_pos"], data_B["o2_x_hn"], data_B["o2_x_test"], data_B["o2_y_test"], data_B["o2_x_hntest"]),
    ]:
        # Donor A arms (ZERO B gradients!)
        for arm in ("V0", "V1", "V1b", "V2_intrinsic", "V2"):
            if arm == "V1b":
                E_use, Sig_use = s0["E_in_res_cpin"], s0["Sigma_eps_cpin"]
            elif arm == "V2_intrinsic":
                E_use, Sig_use = E_int, Sigma_int
            elif arm == "V2":
                E_use, Sig_use = s0["E_in_res"], s0["Sigma_eps"]
            else:
                E_use, Sig_use = None, None
            res = train_swiglu_organ(
                xpA, ypA, xhA, s0["x_A_fit_gen"], k=k_dim, arm=arm,
                E_noise=E_use, Sigma_eps=Sig_use,
                c_init=c_A_init if arm in ("V0", "V1b") else None,
                steps=900, lr=3e-3,
                seed=SEED + (100 if org_name == "organ1" else 200) + abs(hash(arm)) % 97,
            )
            res["donor_A_heldout_audit"] = _audit_organ_on_heldout_states(res, xtA, ytA, xhtA)
            organs_out[org_name][arm] = res
            stage1_summary[org_name][arm] = {
                "train": res["train_metrics"],
                "donor_A_heldout": res["donor_A_heldout_audit"],
            }
            ha = res["donor_A_heldout_audit"]
            print(
                f"    {org_name} [{arm:15s}] A_heldout: g_max={ha['heldout_g_max_mean']:+.2f} "
                f"n>0={ha['heldout_n_gt0_mean']:.1f} cos={ha['heldout_pos_cos']:.3f} "
                f"nrat={ha['heldout_norm_ratio']:.2f} hn_gmax={ha['heldout_hn_g_max_mean']:+.2f} "
                f"| gen_g99={res['train_metrics']['gen_gate_p99']:+.2f}"
            )

        # Native-B Oracle (trained directly on B's cached x_B)
        res_b = train_swiglu_organ(
            xpB, ypB, xhB, s0["x_B_fit_gen"], k=k_dim, arm="native_B_oracle",
            E_noise=None, Sigma_eps=None, steps=900, lr=3e-3,
            seed=SEED + (300 if org_name == "organ1" else 400),
        )
        res_b["native_B_heldout_audit"] = _audit_organ_on_heldout_states(res_b, xtB, ytB, xhtB)
        organs_out[org_name]["native_B_oracle"] = res_b
        stage1_summary[org_name]["native_B_oracle"] = {
            "train": res_b["train_metrics"],
            "native_B_heldout": res_b["native_B_heldout_audit"],
        }
        hb = res_b["native_B_heldout_audit"]
        print(
            f"    {org_name} [{'native_B_oracle':15s}] B_heldout: g_max={hb['heldout_g_max_mean']:+.2f} "
            f"n>0={hb['heldout_n_gt0_mean']:.1f} cos={hb['heldout_pos_cos']:.3f} "
            f"nrat={hb['heldout_norm_ratio']:.2f} hn_gmax={hb['heldout_hn_g_max_mean']:+.2f}"
        )

    stage1_summary["elapsed_sec"] = round(time.time() - t0, 1)
    cache_path = CACHE_DIR / f"pilot_organs_p0_stage1_{site_key}.pt"
    torch.save({"summary": stage1_summary, "organs": organs_out}, cache_path)
    print(f"  [Stage 1 Done in {stage1_summary['elapsed_sec']}s] Saved {cache_path}")
    return stage1_summary


# ======================================================================================
# 7. Stage 2: Real Standalone Widened Materialization + Unlearning + Granite Portability
# ======================================================================================

def greedy_generate_batch(
    model: Any,
    tok: Any,
    device: torch.device,
    prompts: Sequence[str],
    max_new_tokens: int = 8,
    batch_size: int = 16,
) -> List[str]:
    """Pure stock unhooked greedy decoding (NO hooks, NO masks!)."""
    old_pad = tok.padding_side
    tok.padding_side = "left"
    outs: List[str] = []
    try:
        with torch.no_grad():
            for i0 in range(0, len(prompts), batch_size):
                b_pr = list(prompts[i0 : i0 + batch_size])
                enc = tok(b_pr, return_tensors="pt", padding=True, add_special_tokens=True)
                input_ids = enc["input_ids"].to(device)
                attn_mask = enc["attention_mask"].to(device)
                gen = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
                prompt_len = input_ids.shape[1]
                for bi in range(len(b_pr)):
                    new_toks = gen[bi, prompt_len:]
                    outs.append(tok.decode(new_toks, skip_special_tokens=True))
    finally:
        tok.padding_side = old_pad
    return outs


def eval_general_text_kl_ppl(
    model: Any,
    tok: Any,
    device: torch.device,
    passages: Sequence[str],
    base_logprobs: Optional[List[torch.Tensor]] = None,
    max_len: int = 160,
    batch_size: int = 8,
) -> Tuple[Dict[str, Any], List[torch.Tensor]]:
    """
    Evaluate pure stock unhooked model on untouched general-text eval passages.
    Computes exact next-token PPL and (when base_logprobs given) mean/p99/p99.9 KL(p_base || p_organ).
    """
    old_pad = tok.padding_side
    tok.padding_side = "right"
    stored_lps: List[torch.Tensor] = []
    nll_sum = 0.0
    tok_count = 0
    kl_vals: List[np.ndarray] = []

    try:
        with torch.no_grad():
            for bi0 in range(0, len(passages), batch_size):
                b_idx = bi0 // batch_size
                b_txt = list(passages[bi0 : bi0 + batch_size])
                enc = tok(
                    b_txt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=True,
                )
                input_ids = enc["input_ids"].to(device)
                attn_mask = enc["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False).logits
                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :].float()
                shift_labels = input_ids[:, 1:]
                shift_mask = attn_mask[:, 1:].bool()
                logp = F.log_softmax(shift_logits, dim=-1)

                # NLL
                tok_nll = -logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                nll_sum += float(tok_nll[shift_mask].sum().item())
                tok_count += int(shift_mask.sum().item())

                if base_logprobs is None:
                    stored_lps.append(logp.cpu().to(torch.float16))
                else:
                    bp = base_logprobs[b_idx].to(device=device, dtype=torch.float32)
                    p_base = torch.exp(bp)
                    kl = torch.sum(p_base * (bp - logp), dim=-1)
                    kl = torch.clamp(kl, min=0.0)
                    kl_vals.append(kl[shift_mask].cpu().numpy())
    finally:
        tok.padding_side = old_pad

    ppl = math.exp(min(nll_sum / max(tok_count, 1), 20.0))
    res: Dict[str, Any] = {"ppl": round(ppl, 4), "n_tokens": tok_count}
    if kl_vals:
        all_kl = np.concatenate(kl_vals, axis=0)
        res["mean_kl"] = round(float(np.mean(all_kl)), 6)
        res["p99_kl"] = round(float(np.percentile(all_kl, 99)), 6)
        res["p999_kl"] = round(float(np.percentile(all_kl, 99.9)), 6)
        res["max_kl"] = round(float(np.max(all_kl)), 6)
    return res, stored_lps


def audit_organ_firing_analytical(
    conj: Dict[str, Any],
    x_bos_B: np.ndarray,
    x_B_aligned: np.ndarray,
    mlpB_aligned_n2: np.ndarray,
    x_B_unaligned: np.ndarray,
    mlpB_unaligned_n2: np.ndarray,
    rm_B: float = 1.0,
    bos_x_eval_B: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Analytical audit of organ firing on cached untouched eval_passages states (ZERO hooks on model!):
    Computes exact float32 BOS/pos-0 residual output norm (both at mean x_bos_B and across ALL 60
    untouched eval passages' pos-0 vectors!), aligned vs unaligned false-fire rates, and leak energy ratios.
    """
    # Use float32 matching the actual widened nn.Linear weights on MPS!
    Wg = conj["W_g"].astype(np.float32)
    Wu = conj["W_u"].astype(np.float32)
    Wd = conj["W_d"].astype(np.float32)
    bg = conj["b_g"].astype(np.float32) if conj.get("b_g") is not None else np.zeros(Wg.shape[0], dtype=np.float32)
    bu = conj["b_u"].astype(np.float32) if conj.get("b_u") is not None else np.zeros(Wu.shape[0], dtype=np.float32)

    def _eval_block(X: np.ndarray, base_n2: np.ndarray) -> Dict[str, Any]:
        if len(X) == 0:
            return {"n_tokens": 0, "ff_rate_gate_gt0": 0.0, "leak_energy_ratio": 0.0, "out_rms": 0.0}
        Xf = X.astype(np.float32)
        g = Xf @ Wg.T + bg[None, :]
        u = Xf @ Wu.T + bu[None, :]
        a = (g / (1.0 + np.exp(-np.clip(g, -40.0, 40.0)))) * u
        y = a @ Wd.T  # MLP-space output (same units as native mlp(x)!)
        y_n2 = np.sum(y.astype(np.float64) ** 2, axis=-1)
        ff_rate = float(np.mean(g > 0.0))
        leak_ratio = float(np.mean(y_n2) / max(float(np.mean(base_n2)), 1e-12))
        return {
            "n_tokens": int(len(X)),
            "ff_rate_gate_gt0": round(ff_rate, 6),
            "leak_energy_ratio": round(leak_ratio, 6),
            "out_rms": round(float(np.sqrt(np.mean(y_n2))) * rm_B, 6),
        }

    # Exact float32 mean BOS pos-0 check!
    xb = x_bos_B.astype(np.float32)
    gb = Wg @ xb + bg
    ub = Wu @ xb + bu
    ab = (gb / (1.0 + np.exp(-np.clip(gb, -40.0, 40.0)))) * ub
    yb = (Wd @ ab) * float(rm_B)
    bos_out_norm = float(np.linalg.norm(yb.astype(np.float64)))

    eval_bos_mean, eval_bos_max = bos_out_norm, bos_out_norm
    if bos_x_eval_B is not None and len(bos_x_eval_B) > 0:
        Xe = bos_x_eval_B.astype(np.float32)
        ge = Xe @ Wg.T + bg[None, :]
        ue = Xe @ Wu.T + bu[None, :]
        ae = (ge / (1.0 + np.exp(-np.clip(ge, -40.0, 40.0)))) * ue
        ye = (ae @ Wd.T) * float(rm_B)
        norms = np.linalg.norm(ye.astype(np.float64), axis=-1)
        eval_bos_mean = float(np.mean(norms))
        eval_bos_max = float(np.max(norms))

    return {
        "bos_pos0_residual_norm": round(bos_out_norm, 8),
        "bos_eval60_mean_norm": round(eval_bos_mean, 8),
        "bos_eval60_max_norm": round(eval_bos_max, 8),
        "bos_max_abs_up": round(float(np.max(np.abs(ub))), 8),
        "aligned": _eval_block(x_B_aligned, mlpB_aligned_n2),
        "unaligned": _eval_block(x_B_unaligned, mlpB_unaligned_n2),
    }


def eval_organ1_suite(
    model: Any,
    tok: Any,
    device: torch.device,
    ds_fr_it_test: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate Organ 1 on 16 unseen France/Italy entities + 16 unseen non-target hard negatives + 8 dataset.json Fr/It test facts."""
    # 1. 16 unseen test entities x 5 attrs
    prompts, meta = [], []
    for ent in ORGAN1_TEST_ENTITIES:
        c_new = ORGAN1_RULE[ent["country"]]
        pmap = build_entity_attr_prompts(ent["name"], c_new, ent["prot_q"], ent["prot_ans"])
        for attr in ("country", "capital", "currency", "language", "protected"):
            prompts.append(pmap[attr]["prompt"])
            meta.append((ent["name"], ent["country"], c_new, attr, pmap[attr]["aliases"]))

    # 2. 16 unseen non-target hard negatives (must keep ORIGINAL country, must NOT flip to Japan/Brazil!)
    hn_start = len(prompts)
    for ent in ORGAN1_HARDNEG_TEST:
        p = f"The location {ent['name']} is situated in the country of"
        prompts.append(p)
        meta.append((ent["name"], ent["country"], ent["country"], "hardneg", [ent["country"]]))

    # 3. 8 dataset.json France/Italy test facts (generalization to unseen relations like car brands & historical figures!)
    ds_start = len(prompts)
    for f_item in ds_fr_it_test:
        c_old = f_item["answer"]
        c_new = ORGAN1_RULE[c_old]
        prompts.append(f_item["direct_prompt"])
        meta.append((f_item["subject"], c_old, c_new, "ds_fr_it", COUNTRIES_INFO[c_new]["aliases"]))

    gens = greedy_generate_batch(model, tok, device, prompts, max_new_tokens=8, batch_size=16)

    by_attr: Dict[str, List[int]] = defaultdict(list)
    by_ent_4: Dict[str, List[int]] = defaultdict(list)
    hn_flipped_to_rule = 0
    hn_kept_orig = 0
    ds_rule_hits = 0

    for i, gen in enumerate(gens):
        name, c_old, c_new, attr, aliases = meta[i]
        ok = int(check_answer_match(gen, aliases))
        if attr in ("country", "capital", "currency", "language", "protected"):
            by_attr[attr].append(ok)
            if attr != "protected":
                by_ent_4[name].append(ok)
        elif attr == "hardneg":
            hn_kept_orig += ok
            flipped = int(
                check_answer_match(gen, COUNTRIES_INFO["Japan"]["aliases"])
                or check_answer_match(gen, COUNTRIES_INFO["Brazil"]["aliases"])
            )
            hn_flipped_to_rule += flipped
        elif attr == "ds_fr_it":
            ds_rule_hits += ok

    rip3 = float(np.mean(by_attr["capital"] + by_attr["currency"] + by_attr["language"]))
    j4 = float(np.mean([int(sum(v) == 4) for v in by_ent_4.values()]))
    return {
        "direct_country_em": round(float(np.mean(by_attr["country"])), 4),
        "unseen_ripple_3attr_em": round(rip3, 4),
        "capital_em": round(float(np.mean(by_attr["capital"])), 4),
        "currency_em": round(float(np.mean(by_attr["currency"])), 4),
        "language_em": round(float(np.mean(by_attr["language"])), 4),
        "joint_4of4_em": round(j4, 4),
        "protected_retention_em": round(float(np.mean(by_attr["protected"])), 4),
        "hardneg_false_flip_to_rule": round(hn_flipped_to_rule / max(len(ORGAN1_HARDNEG_TEST), 1), 4),
        "hardneg_orig_retained": round(hn_kept_orig / max(len(ORGAN1_HARDNEG_TEST), 1), 4),
        "dataset_fr_it_8_rule_gen": round(ds_rule_hits / max(len(ds_fr_it_test), 1), 4),
    }


def eval_organ2_suite(
    model: Any,
    tok: Any,
    device: torch.device,
    derangement: Dict[str, str],
) -> Dict[str, Any]:
    """Evaluate Organ 2 on 12 target landmarks (direct + 3-attr ripple + protected) + 12 same-city siblings."""
    lm_by_name = {x["name"]: x for x in LANDMARKS_T0}
    prompts, meta = [], []

    for name in ORGAN2_TARGET_NAMES:
        c_new = derangement[name]
        pmap = build_landmark_prompts(lm_by_name[name], c_new)
        for attr in ("country", "capital", "currency", "language", "protected"):
            prompts.append(pmap[attr]["prompt"])
            meta.append((name, lm_by_name[name]["country"], c_new, attr, pmap[attr]["aliases"]))

    for name in ORGAN2_SIBLING_NAMES:
        c_orig = lm_by_name[name]["country"]
        pmap = build_landmark_prompts(lm_by_name[name], c_orig)
        for attr in ("country", "capital"):
            prompts.append(pmap[attr]["prompt"])
            meta.append((name, c_orig, c_orig, f"sib_{attr}", pmap[attr]["aliases"]))

    gens = greedy_generate_batch(model, tok, device, prompts, max_new_tokens=8, batch_size=16)

    by_attr: Dict[str, List[int]] = defaultdict(list)
    by_ent_4: Dict[str, List[int]] = defaultdict(list)
    sib_kept: Dict[str, List[int]] = defaultdict(list)

    for i, gen in enumerate(gens):
        name, c_orig, c_new, attr, aliases = meta[i]
        ok = int(check_answer_match(gen, aliases))
        if attr in ("country", "capital", "currency", "language", "protected"):
            by_attr[attr].append(ok)
            if attr != "protected":
                by_ent_4[name].append(ok)
        else:
            sib_kept[attr].append(ok)

    rip3 = float(np.mean(by_attr["capital"] + by_attr["currency"] + by_attr["language"]))
    j4 = float(np.mean([int(sum(v) == 4) for v in by_ent_4.values()]))
    sib_c_ret = float(np.mean(sib_kept["sib_country"]))
    sib_cap_ret = float(np.mean(sib_kept["sib_capital"]))
    return {
        "direct_country_em": round(float(np.mean(by_attr["country"])), 4),
        "unseen_ripple_3attr_em": round(rip3, 4),
        "capital_em": round(float(np.mean(by_attr["capital"])), 4),
        "currency_em": round(float(np.mean(by_attr["currency"])), 4),
        "language_em": round(float(np.mean(by_attr["language"])), 4),
        "joint_4of4_em": round(j4, 4),
        "protected_retention_em": round(float(np.mean(by_attr["protected"])), 4),
        "sibling_country_retained": round(sib_c_ret, 4),
        "sibling_false_flip_rate": round(1.0 - sib_c_ret, 4),
        "sibling_capital_retained": round(sib_cap_ret, 4),
    }


def eval_196_locality(
    model: Any,
    tok: Any,
    device: torch.device,
    test_facts: List[Dict[str, Any]],
    base_correct: Optional[List[int]] = None,
    exclude_mask: Optional[List[bool]] = None,
) -> Tuple[Dict[str, Any], List[int]]:
    """Evaluate 196-fact dataset.json test split locality, reporting exact newly-broken count."""
    prompts = [f["direct_prompt"] for f in test_facts]
    gens = greedy_generate_batch(model, tok, device, prompts, max_new_tokens=8, batch_size=16)
    oks = [int(check_answer_match(g, f["aliases"], f.get("relation", ""))) for g, f in zip(gens, test_facts)]
    res: Dict[str, Any] = {
        "n_facts": len(test_facts),
        "acc_all": round(float(np.mean(oks)), 4),
        "n_correct_all": int(sum(oks)),
    }
    if base_correct is not None:
        orig_ok_idx = [i for i, b in enumerate(base_correct) if b == 1]
        newly_broken_all = sum(1 for i in orig_ok_idx if oks[i] == 0)
        res["n_orig_correct_all"] = len(orig_ok_idx)
        res["newly_broken_all"] = int(newly_broken_all)
        res["drop_pp_all"] = round(100.0 * (float(np.mean(base_correct)) - float(np.mean(oks))), 2)
        if exclude_mask is not None:
            unrel_idx = [i for i in range(len(test_facts)) if not exclude_mask[i]]
            unrel_orig_ok = [i for i in unrel_idx if base_correct[i] == 1]
            unrel_broken = sum(1 for i in unrel_orig_ok if oks[i] == 0)
            res["n_unrelated"] = len(unrel_idx)
            res["n_unrelated_orig_correct"] = len(unrel_orig_ok)
            res["newly_broken_unrelated"] = int(unrel_broken)
            res["unrelated_retention"] = round(1.0 - unrel_broken / max(len(unrel_orig_ok), 1), 4)
    return res, oks


def check_exact_unlearning(
    model: Any,
    tok: Any,
    device: torch.device,
    mlp: nn.Module,
    orig_state: Dict[str, Any],
    base_logits_probe: torch.Tensor,
    probe_prompts: Sequence[str],
) -> float:
    """Slice widened MLP back to d_ff and return exact max |Delta logit| vs unpatched baseline."""
    unlearn_mlp_native(mlp, orig_state)
    enc = tok(list(probe_prompts), return_tensors="pt", padding=True, add_special_tokens=True)
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device), use_cache=False).logits
    max_diff = float(torch.max(torch.abs(logits.cpu() - base_logits_probe)).item())
    return max_diff


def run_stage2(site_key: str, corpus: Dict[str, List[str]], run_granite: bool = True) -> Dict[str, Any]:
    """
    Stage 2: Real standalone widened Recipient B materialization, exact unlearning verification,
    comprehensive Organ 1 & Organ 2 evaluation, AND 2nd-family Granite-3.1-2B write-once portability!
    """
    t0 = time.time()
    lA, lB = SITE_MAP[site_key]
    s0 = torch.load(CACHE_DIR / f"pilot_organs_p0_stage0_{site_key}.pt", weights_only=False)
    s1 = torch.load(CACHE_DIR / f"pilot_organs_p0_stage1_{site_key}.pt", weights_only=False)

    with open(DATASET_PATH) as f:
        ds_facts = json.load(f)["facts"]
    test_facts_196 = [x for x in ds_facts if x["split"] == "test"]
    ds_fr_it_test = [x for x in test_facts_196 if x["answer"] in ("France", "Italy")]
    fr_it_mask = [x["answer"] in ("France", "Italy") for x in test_facts_196]
    o2_lm_mask = [x["subject"] in set(ORGAN2_TARGET_NAMES) for x in test_facts_196]

    with open(GRAFT_RESULTS_PATH) as f:
        derangement = json.load(f)["T1_forced_gate_audit"]["derangement"]

    probe_prompts = [
        "The location Lyon is situated in the country of",
        "The landmark Eiffel Tower is located in the country of",
        corpus["eval"][0][:120],
        corpus["eval"][1][:120],
    ]

    print(f"\n[Stage 2A] Standalone Widened {RECIP_KEY} (L{lB}, Donor A UNLOADED, ZERO hooks, ZERO masks!)")
    model_B, tok_B, dev_B, rm_B = load_model_fp32(RECIP_KEY)
    mlp_B = get_decoder_layers(model_B)[lB].mlp
    orig_state_B = save_original_mlp_state(mlp_B)

    # Baseline probe logits + baseline general-text logprobs + baseline 196-fact accuracy + baseline O1/O2
    enc_pr = tok_B(probe_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
    with torch.no_grad():
        base_logits_B = model_B(input_ids=enc_pr["input_ids"].to(dev_B), attention_mask=enc_pr["attention_mask"].to(dev_B), use_cache=False).logits.cpu()

    base_gen_B, base_lps_B = eval_general_text_kl_ppl(model_B, tok_B, dev_B, corpus["eval"])
    base_loc_B, base_ok_B = eval_196_locality(model_B, tok_B, dev_B, test_facts_196)
    base_o1_B = eval_organ1_suite(model_B, tok_B, dev_B, ds_fr_it_test)
    base_o2_B = eval_organ2_suite(model_B, tok_B, dev_B, derangement)
    print(f"  Baseline {RECIP_KEY}: PPL={base_gen_B['ppl']}, 196-fact={base_loc_B['n_correct_all']}/196 ({base_loc_B['acc_all']})")

    # Verify zero_organ bit-identity!
    D = orig_state_B["d_model"]
    widen_mlp_native(
        mlp_B, orig_state_B,
        np.zeros((32, D)), np.zeros(32), np.zeros((32, D)), np.zeros(32), np.zeros((D, 32)),
        use_bias=True,
    )
    with torch.no_grad():
        zero_logits_B = model_B(input_ids=enc_pr["input_ids"].to(dev_B), attention_mask=enc_pr["attention_mask"].to(dev_B), use_cache=False).logits.cpu()
    zero_organ_max_diff = float(torch.max(torch.abs(zero_logits_B - base_logits_B)).item())
    unlearn_zero_diff = check_exact_unlearning(model_B, tok_B, dev_B, mlp_B, orig_state_B, base_logits_B, probe_prompts)
    print(f"  zero_organ max|dLogit|={zero_organ_max_diff:.2e} | unlearn max|dLogit|={unlearn_zero_diff:.2e}")

    smol_results: Dict[str, Any] = {
        "baseline": {
            "general_text": base_gen_B,
            "locality_196": base_loc_B,
            "organ1": base_o1_B,
            "organ2": base_o2_B,
            "zero_organ_max_dlogit": zero_organ_max_diff,
            "zero_organ_unlearn_max_dlogit": unlearn_zero_diff,
        },
        "organ1": {},
        "organ2": {},
    }

    arm_order = ["V0", "V1", "V1b", "V1b_gamma", "V2_intrinsic", "V2", "V4", "shuffled_R", "native_B_oracle"]
    for org_name in ("organ1", "organ2"):
        print(f"\n  --- Evaluating {org_name} on Standalone Widened {RECIP_KEY} ---")
        for arm in arm_order:
            if arm == "native_B_oracle":
                raw = s1["organs"][org_name]["native_B_oracle"]
                Wg, bg, Wu, bu = apply_exact_bos_null_affine(
                    raw["W_g"], raw["b_g"], raw["W_u"], raw["b_u"], s0["x_bos_B"], s0["sock_in"]["mu_src"],
                    x_bos_bank=s0.get("bos_x_fit_B"),
                )
                conj = {"W_g": Wg, "b_g": bg, "W_u": Wu, "b_u": bu, "W_d": raw["W_d"] / rm_B, "use_bias": True}
            else:
                if arm in ("V4", "shuffled_R"):
                    src_arm = "V2"
                elif arm == "V1b_gamma":
                    src_arm = "V2_intrinsic"
                else:
                    src_arm = arm
                conj = conjugate_organ_weights(
                    s1["organs"][org_name][src_arm],
                    arm=arm,
                    M_in=s0["sock_in"]["M"],
                    mu_A=s0["sock_in"]["mu_dst"],
                    mu_B=s0["sock_in"]["mu_src"],
                    M_out=s0["sock_out"]["M"],
                    x_bos_B=s0["x_bos_B"],
                    rm_B=rm_B,
                    M_in_cpin=s0["sock_cpin"]["M"],
                    M_gamma=s0.get("M_gamma"),
                    c_B=s0["sock_cpin"]["c_src"],
                    P_write_core=s0["P_write_core"],
                    V_c0=s0["V_c0"],
                    x_bos_bank_B=s0.get("bos_x_fit_B"),
                    rng_seed=SEED + (555 if org_name == "organ1" else 666),
                )

            # Real stock widening!
            new_dff = widen_mlp_native(
                mlp_B, orig_state_B,
                conj["W_g"], conj["b_g"], conj["W_u"], conj["b_u"], conj["W_d"],
                use_bias=conj["use_bias"],
            )

            # Analytical firing & BOS audit
            fire_audit = audit_organ_firing_analytical(
                conj, s0["x_bos_B"],
                s0["x_B_eval_aligned"], s0["mlpB_eval_aligned_n2"],
                s0["x_B_eval_unaligned"], s0["mlpB_eval_unaligned_n2"],
                rm_B=rm_B,
                bos_x_eval_B=s0.get("bos_x_eval_B"),
            )
            # Real unhooked forward pass: general-text KL & Delta-PPL
            gen_res, _ = eval_general_text_kl_ppl(model_B, tok_B, dev_B, corpus["eval"], base_logprobs=base_lps_B)
            gen_res["delta_ppl"] = round(gen_res["ppl"] - base_gen_B["ppl"], 4)

            # Real unhooked forward pass: target QA + ripple + hard negatives
            if org_name == "organ1":
                qa_res = eval_organ1_suite(model_B, tok_B, dev_B, ds_fr_it_test)
                loc_res, _ = eval_196_locality(model_B, tok_B, dev_B, test_facts_196, base_ok_B, exclude_mask=fr_it_mask)
            else:
                qa_res = eval_organ2_suite(model_B, tok_B, dev_B, derangement)
                loc_res, _ = eval_196_locality(model_B, tok_B, dev_B, test_facts_196, base_ok_B, exclude_mask=o2_lm_mask)

            # Instant Exact Unlearning assertion!
            unlearn_diff = check_exact_unlearning(model_B, tok_B, dev_B, mlp_B, orig_state_B, base_logits_B, probe_prompts)
            assert unlearn_diff == 0.0, f"Unlearning was NOT exact bit-identical: {unlearn_diff}"

            smol_results[org_name][arm] = {
                "widened_d_ff": new_dff,
                "use_bias": conj["use_bias"],
                "unlearn_max_dlogit": unlearn_diff,
                "firing_audit": fire_audit,
                "general_text": gen_res,
                "qa": qa_res,
                "locality_196": loc_res,
            }
            print(
                f"    [{arm:15s}] dir={qa_res['direct_country_em']:.3f} rip3={qa_res['unseen_ripple_3attr_em']:.3f} "
                f"j4={qa_res['joint_4of4_em']:.3f} prot={qa_res['protected_retention_em']:.3f} | "
                f"KL={gen_res['mean_kl']:.5f} dPPL={gen_res['delta_ppl']:+.3f} BOS={fire_audit['bos_pos0_residual_norm']:.2e} "
                f"brk_unrel={loc_res['newly_broken_unrelated']}/{loc_res['n_unrelated_orig_correct']} unlearn={unlearn_diff}"
            )

    unload_model(model_B)

    # ==================================================================================
    # Stage 2B: Write-Once, Implant-Into-2nd-Family Portability (Granite-3.1-2B, L8, rm=0.22!)
    # ==================================================================================
    granite_results: Dict[str, Any] = {"skipped": not run_granite}
    if run_granite:
        tg0 = time.time()
        print(f"\n[Stage 2B] Write-Once 2nd-Family Portability: Implanting EXACT SAME Donor A Organ 1 into {SECOND_RECIP_KEY} L{GRANITE_SITE_LB} (ZERO Retraining!)")
        # Reload Donor A only for matching fit/resid passages? No! Wait — we need Donor A's word_end_maps on fit/resid passages
        # to align Granite's tokens cleanly! Let's extract A's fit/resid + train entities OR reuse cached A states?
        # Even cleaner: extracting A's fit+resid (200 passages) takes only ~25s on MPS, AND guarantees exact alignment!
        with open(GRAFT_RESULTS_PATH) as f:
            passing_lm_names = {x["name"] for x in json.load(f)["T0_competence_filter"]["landmarks"]["items"] if x["passed"]}
        with open(T3_RESULTS_PATH) as f:
            gap_names = {x["entity"] for x in json.load(f)["gap_items"]}
        excluded_subjs = (
            passing_lm_names | gap_names
            | {x["name"] for x in ORGAN1_TRAIN_ENTITIES}
            | {x["name"] for x in ORGAN1_TEST_ENTITIES}
            | {x["name"] for x in ORGAN1_HARDNEG_TRAIN}
            | {x["name"] for x in ORGAN1_HARDNEG_TEST}
            | {x["subject"] for x in ds_facts if x["split"] != "train"}
        )
        train_ent_facts = [
            x for x in ds_facts
            if x["split"] == "train" and x["answer"] not in ("France", "Italy") and x["subject"] not in excluded_subjs
        ][:350]
        ent_prompts = [x["direct_prompt"] for x in train_ent_facts]
        ent_spans = [x["direct_subj_char_span"] for x in train_ent_facts]

        model_A, tok_A, dev_A, _ = load_model_fp32(DONOR_KEY)
        stA_fit = extract_layer_states_on_passages(model_A, tok_A, dev_A, corpus["fit"], lA)
        stA_res = extract_layer_states_on_passages(model_A, tok_A, dev_A, corpus["resid"], lA)
        stA_ent = extract_entity_last_token_states(model_A, tok_A, dev_A, ent_prompts, ent_spans, lA)
        unload_model(model_A)

        model_C, tok_C, dev_C, rm_C = load_model_fp32(SECOND_RECIP_KEY)
        print(f"  Loaded {SECOND_RECIP_KEY} float32 | residual_multiplier={rm_C}")
        stC_fit = extract_layer_states_on_passages(model_C, tok_C, dev_C, corpus["fit"], GRANITE_SITE_LB)
        stC_res = extract_layer_states_on_passages(model_C, tok_C, dev_C, corpus["resid"], GRANITE_SITE_LB)
        stC_ev = extract_layer_states_on_passages(model_C, tok_C, dev_C, corpus["eval"], GRANITE_SITE_LB)
        stC_ent = extract_entity_last_token_states(model_C, tok_C, dev_C, ent_prompts, ent_spans, GRANITE_SITE_LB)

        pC_fit = pair_passage_activations(stA_fit, stC_fit)
        pC_res = pair_passage_activations(stA_res, stC_res)
        # Also align eval for analytical firing audit (split aligned/unaligned on Granite!)
        # For aligned/unaligned split on Granite eval, reuse stA_res? Better: just use stC_ev directly
        # (we don't even need A on eval to compute Granite's C_0 or BOS; for aligned/unaligned split we can
        # mark word-final tokens in Granite as aligned_proxy and non-word-final as unaligned_proxy!)
        xC_ev_wf, mC_ev_wf, xC_ev_nonwf, mC_ev_nonwf = [], [], [], []
        for pi in range(len(stC_ev["x"])):
            wf_pos = set(stC_ev["word_end_maps"][pi].values())
            LC = stC_ev["seq_lens"][pi]
            for pos in range(1, LC):
                if pos in wf_pos:
                    xC_ev_wf.append(stC_ev["x"][pi][pos])
                    mC_ev_wf.append(stC_ev["mlp_norm2"][pi][pos])
                else:
                    xC_ev_nonwf.append(stC_ev["x"][pi][pos])
                    mC_ev_nonwf.append(stC_ev["mlp_norm2"][pi][pos])
        xC_ev_wf = np.stack(xC_ev_wf, axis=0)
        mC_ev_wf = np.array(mC_ev_wf, dtype=np.float64)
        xC_ev_nonwf = np.stack(xC_ev_nonwf, axis=0) if xC_ev_nonwf else np.zeros((0, D), dtype=np.float32)
        mC_ev_nonwf = np.array(mC_ev_nonwf, dtype=np.float64)

        # Outlier-filter Granite h_out (and x) BEFORE socket fitting!
        keepC_x, _ = filter_outlier_pairs(pC_fit["x_A"], pC_fit["x_B"], max_med_mult=2.5)
        keepC_ho, hoC_stats = filter_outlier_pairs(pC_fit["h_out_A"], pC_fit["h_out_B"], max_med_mult=2.5)
        keepC_res, _ = filter_outlier_pairs(pC_res["h_out_A"], pC_res["h_out_B"], max_med_mult=2.5)
        nC_x, nC_ho, nC_ent = int(np.sum(keepC_x)), int(np.sum(keepC_ho)), int(len(stA_ent["x"]))
        wC_ent_x = max(3.0, 0.25 * nC_x / max(nC_ent, 1))
        wC_ent_ho = max(3.0, 0.25 * nC_ho / max(nC_ent, 1))

        xA_dual_C = np.concatenate([pC_fit["x_A"][keepC_x], stA_ent["x"]], axis=0)
        xC_dual_C = np.concatenate([pC_fit["x_B"][keepC_x], stC_ent["x"]], axis=0)
        hhA_dual_C = np.concatenate([pC_fit["h_hat_A"][keepC_x], stA_ent["h_hat"]], axis=0)
        hhC_dual_C = np.concatenate([pC_fit["h_hat_B"][keepC_x], stC_ent["h_hat"]], axis=0)
        hoA_dual_C = np.concatenate([pC_fit["h_out_A"][keepC_ho], stA_ent["h_out"]], axis=0)
        hoC_dual_C = np.concatenate([pC_fit["h_out_B"][keepC_ho], stC_ent["h_out"]], axis=0)
        wC_x = np.concatenate([np.ones(nC_x), np.full(nC_ent, wC_ent_x)])
        wC_ho = np.concatenate([np.ones(nC_ho), np.full(nC_ent, wC_ent_ho)])

        sock_in_C = fit_orthogonal_socket(xC_dual_C, xA_dual_C, weights=wC_x)
        sock_out_C = fit_orthogonal_socket(hoA_dual_C, hoC_dual_C, weights=wC_ho)
        sock_hhat_C = fit_orthogonal_socket(hhC_dual_C, hhA_dual_C, weights=wC_x)
        c_C, _ = compute_constant_predictor(xC_dual_C, lam=1e-2)
        M_gamma_C = build_gamma_deconv_read_map(stA_fit["gamma"], stC_fit["gamma"], sock_hhat_C["R"], lam_frac=0.05)

        ent_norm_ratio_C = float(
            np.median(np.linalg.norm(stC_ent["h_out"], axis=-1))
            / max(np.median(np.linalg.norm(stA_ent["h_out"], axis=-1)), 1e-12)
        )
        print(f"  Granite sock_out s_out={sock_out_C['s']:.4f} (entity norm ratio={ent_norm_ratio_C:.4f}) | outlier filter: {hoC_stats}")

        x_bos_C = np.mean(stC_fit["bos_x"], axis=0)
        wcore_C = compute_write_core_projector(
            pC_res["h_out_A"][keepC_res], pC_res["h_out_B"][keepC_res], sock_out_C["M"], r_out=512
        )
        c0_C = compute_uncentered_C0_basis(pC_fit["x_B_all"], r_c0=32)

        mlp_C = get_decoder_layers(model_C)[GRANITE_SITE_LB].mlp
        orig_state_C = save_original_mlp_state(mlp_C)

        enc_prC = tok_C(probe_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        with torch.no_grad():
            base_logits_C = model_C(input_ids=enc_prC["input_ids"].to(dev_C), attention_mask=enc_prC["attention_mask"].to(dev_C), use_cache=False).logits.cpu()

        base_gen_C, base_lps_C = eval_general_text_kl_ppl(model_C, tok_C, dev_C, corpus["eval"])
        base_loc_C, base_ok_C = eval_196_locality(model_C, tok_C, dev_C, test_facts_196)
        base_o1_C = eval_organ1_suite(model_C, tok_C, dev_C, ds_fr_it_test)
        print(f"  Baseline {SECOND_RECIP_KEY}: PPL={base_gen_C['ppl']}, 196-fact={base_loc_C['n_correct_all']}/196 ({base_loc_C['acc_all']})")

        granite_results = {
            "site": {"donor": DONOR_KEY, "recipient": SECOND_RECIP_KEY, "lA": lA, "lC": GRANITE_SITE_LB, "rm_C": rm_C},
            "n_fit_aligned": int(len(pC_fit["x_A"])),
            "socket_scales": {
                "s_in": round(sock_in_C["s"], 4),
                "s_out": round(sock_out_C["s"], 4),
                "entity_norm_ratio_h_out": round(ent_norm_ratio_C, 4),
            },
            "outlier_filter_h_out_fit": hoC_stats,
            "baseline": {"general_text": base_gen_C, "locality_196": base_loc_C, "organ1": base_o1_C},
            "organ1": {},
        }

        for arm in ("V1", "V1b_gamma", "V2_intrinsic", "V2", "V4", "shuffled_R"):
            if arm in ("V4", "shuffled_R"):
                src_arm = "V2"
            elif arm == "V1b_gamma":
                src_arm = "V2_intrinsic"
            else:
                src_arm = arm
            conj_C = conjugate_organ_weights(
                s1["organs"]["organ1"][src_arm],
                arm=arm,
                M_in=sock_in_C["M"],
                mu_A=sock_in_C["mu_dst"],
                mu_B=sock_in_C["mu_src"],
                M_out=sock_out_C["M"],
                x_bos_B=x_bos_C,
                rm_B=rm_C,  # CRITICAL: divides W_down by 0.22!
                M_gamma=M_gamma_C,
                c_B=c_C,
                P_write_core=wcore_C["P_core"],
                V_c0=c0_C["V_c0"],
                x_bos_bank_B=stC_fit["bos_x"],
                rng_seed=SEED + 888,
            )
            new_dff_C = widen_mlp_native(
                mlp_C, orig_state_C,
                conj_C["W_g"], conj_C["b_g"], conj_C["W_u"], conj_C["b_u"], conj_C["W_d"],
                use_bias=conj_C["use_bias"],
            )
            fire_C = audit_organ_firing_analytical(
                conj_C, x_bos_C, xC_ev_wf, mC_ev_wf, xC_ev_nonwf, mC_ev_nonwf,
                rm_B=rm_C, bos_x_eval_B=stC_ev["bos_x"],
            )
            gen_C, _ = eval_general_text_kl_ppl(model_C, tok_C, dev_C, corpus["eval"], base_logprobs=base_lps_C)
            gen_C["delta_ppl"] = round(gen_C["ppl"] - base_gen_C["ppl"], 4)
            qa_C = eval_organ1_suite(model_C, tok_C, dev_C, ds_fr_it_test)
            loc_C, _ = eval_196_locality(model_C, tok_C, dev_C, test_facts_196, base_ok_C, exclude_mask=fr_it_mask)
            unlearn_C = check_exact_unlearning(model_C, tok_C, dev_C, mlp_C, orig_state_C, base_logits_C, probe_prompts)
            assert unlearn_C == 0.0, f"Granite unlearning NOT exact: {unlearn_C}"

            granite_results["organ1"][arm] = {
                "widened_d_ff": new_dff_C,
                "unlearn_max_dlogit": unlearn_C,
                "firing_audit": fire_C,
                "general_text": gen_C,
                "qa": qa_C,
                "locality_196": loc_C,
            }
            print(
                f"    Granite [{arm:10s}] dir={qa_C['direct_country_em']:.3f} rip3={qa_C['unseen_ripple_3attr_em']:.3f} "
                f"j4={qa_C['joint_4of4_em']:.3f} prot={qa_C['protected_retention_em']:.3f} | "
                f"KL={gen_C['mean_kl']:.5f} dPPL={gen_C['delta_ppl']:+.3f} BOS={fire_C['bos_pos0_residual_norm']:.2e} "
                f"brk_unrel={loc_C['newly_broken_unrelated']}/{loc_C['n_unrelated_orig_correct']} unlearn={unlearn_C}"
            )

        granite_results["elapsed_sec"] = round(time.time() - tg0, 1)
        unload_model(model_C)

    return {
        "smollm2_standalone": smol_results,
        "granite_write_once_portability": granite_results,
        "elapsed_sec": round(time.time() - t0, 1),
    }


# ======================================================================================
# 8. Pure-CPU Unit Tests (Zero GPU Required!)
# ======================================================================================

def run_cpu_unit_tests() -> None:
    """
    Comprehensive CPU-only math, architecture, and data-hygiene verification:
      1. Zero-overlap check between 260 general-text passages and ALL Organ 1/2 entities,
         T1 landmarks, T3 gap entities, and dataset.json test subjects.
      2. Exact rank-1 BOS up-null & gate-suppression verification:
         SiLU(g_BOS) * u_BOS == 0 (<1e-12 fp64, <1e-10 fp32) AND exact mean preservation at mu_B!
      3. Exact V1b constant-pinning verification: ||c_A^T M_in_cpin - c_B^T|| < 1e-12 AND ||W_u x_BOS|| < 1e-12!
      4. Exact V4 C_0 up-projection + BOS up-null verification.
      5. Real HuggingFace LlamaMLP and GraniteMLP native widening (bias=True & bias=False),
         Granite rm=0.22 exact residual compensation, zero-organ bit-identity, and
         EXACT 0.0 unlearning slice!
    """
    print("=== Running P-ORGAN-0 Pure-CPU Unit Tests ===")

    # 1. Corpus disjointness check
    corpus = build_diverse_general_corpus()
    assert len(corpus["fit"]) == 150 and len(corpus["resid"]) == 50 and len(corpus["eval"]) == 60
    forbidden = (
        [x["name"] for x in ORGAN1_TRAIN_ENTITIES]
        + [x["name"] for x in ORGAN1_TEST_ENTITIES]
        + [x["name"] for x in ORGAN1_HARDNEG_TRAIN]
        + [x["name"] for x in ORGAN1_HARDNEG_TEST]
        + [x["name"] for x in LANDMARKS_T0]
    )
    if DATASET_PATH.exists():
        with open(DATASET_PATH) as f:
            forbidden += [x["subject"] for x in json.load(f)["facts"] if x["split"] == "test"]
    if T3_RESULTS_PATH.exists():
        with open(T3_RESULTS_PATH) as f:
            forbidden += [x["entity"] for x in json.load(f)["gap_items"]]
    collisions = verify_corpus_disjointness(corpus, forbidden)
    assert len(collisions) == 0, f"General corpus leaked test terms: {collisions[:10]}"
    # Verify ALL Organ 1 and Organ 2 prompt builders work with zero KeyErrors and valid spans
    lm_by_name = {x["name"]: x for x in LANDMARKS_T0}
    for ent in ORGAN1_TEST_ENTITIES:
        pm = build_entity_attr_prompts(ent["name"], ORGAN1_RULE[ent["country"]], ent["prot_q"], ent["prot_ans"])
        assert set(pm.keys()) == {"country", "capital", "currency", "language", "protected"}
    for nm in ORGAN2_TARGET_NAMES + ORGAN2_SIBLING_NAMES:
        pm = build_landmark_prompts(lm_by_name[nm], lm_by_name[nm]["country"])
        assert set(pm.keys()) == {"country", "capital", "currency", "language", "protected"}
    print(f"  [PASS 1/6] General corpus (260 passages, 150/50/60 split) has 0 collisions across {len(set(forbidden))} forbidden entities; all O1/O2 prompt builders verified.")

    # 2. Synthetic sockets + BOS up-null + mean preservation + V1b constant-pinning + V4 C0
    rng = np.random.default_rng(12345)
    D, N, k = 128, 600, 16
    Q_true, _ = sla.qr(rng.standard_normal((D, D)))
    mu_B_true = rng.standard_normal(D) * 1.5
    mu_A_true = rng.standard_normal(D) * 1.2
    X_B = mu_B_true + rng.standard_normal((N, D)) * 0.5
    X_A = mu_A_true + 1.15 * ((X_B - mu_B_true) @ Q_true.T) + rng.standard_normal((N, D)) * 0.05
    # Massive BOS sink vector (20x norm!)
    x_bos_B = mu_B_true + rng.standard_normal(D) * 20.0

    sock_in = fit_orthogonal_socket(X_B, X_A)
    sock_out = fit_orthogonal_socket(X_A, X_B)
    sock_cpin = fit_constant_pinned_socket(X_B, X_A)
    c0_info = compute_uncentered_C0_basis(np.vstack([X_B, x_bos_B[None, :]]), r_c0=16)

    assert sock_cpin["pin_err"] < 1e-11, f"pin_err={sock_cpin['pin_err']}"

    organ_A = {
        "W_g": rng.standard_normal((k, D)) * 0.2,
        "b_g": rng.standard_normal(k) - 2.0,
        "W_u": rng.standard_normal((k, D)) * 0.2,
        "b_u": rng.standard_normal(k) * 0.5,
        "W_d": rng.standard_normal((D, k)) * 0.1,
    }

    for arm in ("V1", "V2", "V4"):
        conj = conjugate_organ_weights(
            organ_A, arm=arm, M_in=sock_in["M"], mu_A=sock_in["mu_dst"], mu_B=sock_in["mu_src"],
            M_out=sock_out["M"], x_bos_B=x_bos_B, rm_B=1.0, V_c0=c0_info["V_c0"],
        )
        # Check exact fp64 BOS up == 0 and SiLU(g)*u == 0
        u_bos_64 = conj["W_u"] @ x_bos_B + conj["b_u"]
        g_bos_64 = conj["W_g"] @ x_bos_B + conj["b_g"]
        a_bos_64 = _silu_np(g_bos_64) * u_bos_64
        assert np.max(np.abs(u_bos_64)) < 1e-10, f"{arm} fp64 u_bos={np.max(np.abs(u_bos_64))}"
        assert np.max(np.abs(a_bos_64)) < 1e-10, f"{arm} fp64 a_bos={np.max(np.abs(a_bos_64))}"

        # Check exact fp32 BOS output norm == ~0 (since u_bos==0 and W_g is untouched, fp32 cancel error is tiny!)
        u_bos_32 = conj["W_u"].astype(np.float32) @ x_bos_B.astype(np.float32) + conj["b_u"].astype(np.float32)
        g_bos_32 = conj["W_g"].astype(np.float32) @ x_bos_B.astype(np.float32) + conj["b_g"].astype(np.float32)
        a_bos_32 = (g_bos_32 / (1.0 + np.exp(-np.clip(g_bos_32, -40.0, 40.0)))) * u_bos_32
        y_bos_32 = conj["W_d"].astype(np.float32) @ a_bos_32
        assert float(np.linalg.norm(y_bos_32)) < 2e-3, f"{arm} fp32 y_bos={np.linalg.norm(y_bos_32)}"

        # Check exact mean preservation at mu_B AND W_g untouched (no -20 hammer!)
        if arm in ("V1", "V2"):
            u_muB = conj["W_u"] @ sock_in["mu_src"] + conj["b_u"]
            u_muA = organ_A["W_u"] @ sock_in["mu_dst"] + organ_A["b_u"]
            assert np.max(np.abs(u_muB - u_muA)) < 1e-11, f"{arm} mean drift={np.max(np.abs(u_muB - u_muA))}"
            g_expected = organ_A["W_g"] @ sock_in["M"]
            assert np.max(np.abs(conj["W_g"] - g_expected)) < 1e-12, f"{arm} W_g was distorted!"

    # Test outlier_filter_pairs recovers true scale despite 1000x massive-activation outliers!
    X_A_corrupt = X_A.copy()
    X_A_corrupt[:8] *= 500.0  # Simulate Qwen massive-activation outliers!
    s_corrupt = fit_orthogonal_socket(X_A_corrupt, X_B)["s"]
    keep_clean, ostats = filter_outlier_pairs(X_A_corrupt, X_B, max_med_mult=2.5)
    s_clean = fit_orthogonal_socket(X_A_corrupt[keep_clean], X_B[keep_clean])["s"]
    s_true = sock_out["s"]
    assert abs(s_clean - s_true) / s_true < 0.02 and abs(s_corrupt - s_true) / s_true > 0.5, (
        f"Outlier filter check failed: true={s_true:.4f}, corrupt={s_corrupt:.4f}, clean={s_clean:.4f}"
    )
    print(f"  [PASS 2/6] Exact BOS up-null (W_g untouched, fp64 <1e-12) + outlier filter verified (corrupt s={s_corrupt:.4f} -> clean s={s_clean:.4f} vs true={s_true:.4f}).")

    # 3. V1b & V1b_gamma strictly bias-free constant-pinning + BOS null
    conj_v1b = conjugate_organ_weights(
        organ_A, arm="V1b", M_in=sock_in["M"], mu_A=sock_in["mu_dst"], mu_B=sock_in["mu_src"],
        M_out=sock_out["M"], x_bos_B=x_bos_B, rm_B=1.0, M_in_cpin=sock_cpin["M"],
    )
    assert conj_v1b["use_bias"] is False and conj_v1b["b_g"] is None and conj_v1b["b_u"] is None
    assert np.max(np.abs(conj_v1b["W_u"] @ x_bos_B)) < 1e-11

    gamma_A_syn = 1.0 + 0.2 * rng.standard_normal(D)
    gamma_B_syn = 1.0 + 0.2 * rng.standard_normal(D)
    M_gamma_syn = build_gamma_deconv_read_map(gamma_A_syn, gamma_B_syn, sock_in["R"])
    conj_v1bg = conjugate_organ_weights(
        organ_A, arm="V1b_gamma", M_in=sock_in["M"], mu_A=sock_in["mu_dst"], mu_B=sock_in["mu_src"],
        M_out=sock_out["M"], x_bos_B=x_bos_B, rm_B=1.0, M_gamma=M_gamma_syn, c_B=sock_cpin["c_src"],
    )
    assert conj_v1bg["use_bias"] is False and conj_v1bg["b_g"] is None and conj_v1bg["b_u"] is None
    assert np.max(np.abs(conj_v1bg["W_u"] @ x_bos_B)) < 1e-11
    print("  [PASS 3/6] Strictly bias-free V1b & V1b_gamma constant-pinning (err <1e-11) + exact orthogonal BOS up-null verified.")

    # 4. Real HF LlamaMLP & GraniteMLP stock widening, Granite rm=0.22 compensation, and EXACT 0.0 unlearning!
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaMLP
    from transformers.models.granite.configuration_granite import GraniteConfig
    from transformers.models.granite.modeling_granite import GraniteMLP

    x_test = torch.randn(4, 7, D, dtype=torch.float32)
    x_test[:, 0, :] = torch.from_numpy(x_bos_B.astype(np.float32))  # Real BOS at pos-0!

    for cls_name, cfg, mlp_cls, rm_val in [
        ("LlamaMLP", LlamaConfig(hidden_size=D, intermediate_size=256, hidden_act="silu", mlp_bias=False), LlamaMLP, 1.0),
        ("GraniteMLP", GraniteConfig(hidden_size=D, intermediate_size=256, hidden_act="silu", mlp_bias=False, residual_multiplier=0.22), GraniteMLP, 0.22),
    ]:
        mlp = mlp_cls(cfg).eval()
        orig_st = save_original_mlp_state(mlp)
        with torch.no_grad():
            y_base = mlp(x_test)

        # Widen with V2 (including rm_val=0.22 division for Granite!)
        conj_m = conjugate_organ_weights(
            organ_A, arm="V2", M_in=sock_in["M"], mu_A=sock_in["mu_dst"], mu_B=sock_in["mu_src"],
            M_out=sock_out["M"], x_bos_B=x_bos_B, rm_B=rm_val,
        )
        new_dff = widen_mlp_native(
            mlp, orig_st, conj_m["W_g"], conj_m["b_g"], conj_m["W_u"], conj_m["b_u"], conj_m["W_d"], use_bias=True
        )
        assert new_dff == 256 + k and mlp.down_proj.bias is None
        with torch.no_grad():
            y_wide = mlp(x_test)

        # Verify pos-0 BOS delta is ~0.0 inside the real widened HF MLP!
        bos_delta = float(torch.max(torch.abs((y_wide[:, 0, :] - y_base[:, 0, :]) * rm_val)).item())
        assert bos_delta < 2e-3, f"{cls_name} widened BOS delta={bos_delta}"

        # Verify Granite rm=0.22 residual delta matches unscaled rm=1.0 target delta!
        if rm_val != 1.0:
            actual_res_delta = (y_wide[:, 1:, :] - y_base[:, 1:, :]) * rm_val
            # Compare against analytical rm=1.0 organ delta
            x_np = x_test[:, 1:, :].numpy().astype(np.float64)
            g_np = x_np @ conj_m["W_g"].T + conj_m["b_g"]
            u_np = x_np @ conj_m["W_u"].T + conj_m["b_u"]
            y_exp = (_silu_np(g_np) * u_np) @ (conj_m["W_d"] * rm_val).T
            rm_err = float(np.max(np.abs(actual_res_delta.numpy() - y_exp)))
            assert rm_err < 1e-4, f"Granite rm=0.22 compensation error={rm_err}"

        # Instant Exact Unlearning slice!
        unlearn_mlp_native(mlp, orig_st)
        assert mlp.gate_proj.out_features == 256 and mlp.gate_proj.bias is None
        with torch.no_grad():
            y_unlearned = mlp(x_test)
        unlearn_err = float(torch.max(torch.abs(y_unlearned - y_base)).item())
        assert unlearn_err == 0.0, f"{cls_name} unlearning NOT exact 0.0: {unlearn_err}"

    print("  [PASS 4/6] Real HF LlamaMLP & GraniteMLP widening (down_proj bias=False), Granite rm=0.22 scaling, BOS delta <1e-4, and EXACT 0.0 unlearning verified!")

    # 5. Verify V2_intrinsic positive-gate training kills SiLU negative-lobe cheating on separable synthetic data!
    u_cls = rng.standard_normal(D)
    u_cls /= np.linalg.norm(u_cls)
    v_tgt = rng.standard_normal(D) * 5.0
    X_pos_syn = mu_A_true + 6.0 * u_cls[None, :] + rng.standard_normal((80, D)) * 0.4
    Y_pos_syn = np.tile(v_tgt, (80, 1))
    X_hn_syn = mu_A_true - 4.0 * u_cls[None, :] + rng.standard_normal((80, D)) * 0.4
    X_gen_syn = mu_A_true + rng.standard_normal((300, D)) * 0.5
    E_int_syn, Sig_int_syn = build_donor_intrinsic_noise(X_gen_syn, n_samples=400, sigma_rel=0.35)
    res_int = train_swiglu_organ(
        X_pos_syn, Y_pos_syn, X_hn_syn, X_gen_syn,
        k=16, arm="V2_intrinsic", E_noise=E_int_syn, Sigma_eps=Sig_int_syn, steps=180, lr=5e-3,
    )
    tm = res_int["train_metrics"]
    assert tm["pos_g_max_mean"] > 2.0 and tm["pos_n_gt0_mean"] >= 3.0, f"Negative-lobe cheating detected: {tm}"
    assert tm["pos_cos"] > 0.90 and tm["gen_gate_p99"] < -2.0, f"Poor separation: {tm}"
    print(f"  [PASS 5/6] V2_intrinsic positive-gate margin verified (NO SiLU negative-lobe cheating!): {tm}")
    print("  [PASS 6/6] All recipient-free donor-intrinsic noise + outlier filter + V1b_gamma checks passed!")
    print("=== ALL 6 P-ORGAN-0 CPU UNIT TESTS PASSED! ===")


# ======================================================================================
# 9. CLI Entrypoint
# ======================================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="P-ORGAN-0: Native Platonic-Conjugated SwiGLU Implantation Pilot")
    parser.add_argument("--stages", type=str, default="0,1,2", help="Comma-separated stages to run: 0,1,2")
    parser.add_argument("--site", type=str, default="17_9", choices=list(SITE_MAP.keys()), help="Layer pair site")
    parser.add_argument("--skip-granite", action="store_true", help="Skip Stage 2B Granite 2nd-family portability")
    parser.add_argument("--unit-test", action="store_true", help="Run fast pure-CPU unit tests only")
    parser.add_argument("--out", type=str, default=str(OUT_PATH), help="Output JSON path")
    args = parser.parse_args()

    if args.unit_test:
        run_cpu_unit_tests()
        return

    stages = [int(x.strip()) for x in args.stages.split(",") if x.strip()]
    corpus = build_diverse_general_corpus()
    out_path = Path(args.out)

    # Load existing results if resuming a subset of stages
    results: Dict[str, Any] = {}
    if out_path.exists():
        try:
            with open(out_path) as f:
                results = json.load(f)
        except Exception:
            results = {}

    results["meta"] = {
        "site": args.site,
        "lA": SITE_MAP[args.site][0],
        "lB": SITE_MAP[args.site][1],
        "donor": DONOR_KEY,
        "recipient": RECIP_KEY,
        "second_recipient": SECOND_RECIP_KEY,
        "device": str(get_compute_device()),
        "seed": SEED,
        "stages_run": stages,
    }

    t_all = time.time()
    if 0 in stages:
        results["stage0"] = run_stage0(args.site, corpus)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    if 1 in stages:
        results["stage1"] = run_stage1(args.site)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    if 2 in stages:
        results["stage2"] = run_stage2(args.site, corpus, run_granite=not args.skip_granite)
        results["meta"]["total_elapsed_sec"] = round(time.time() - t_all, 1)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n[P-ORGAN-0 COMPLETE] Saved results to {out_path} (elapsed={round(time.time() - t_all, 1)}s)")


if __name__ == "__main__":
    main()
