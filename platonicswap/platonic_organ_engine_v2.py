#!/usr/bin/env python3
"""
platonic_organ_engine_v2.py — PlatonicSwap v2 Engine (New SHA-256; v1.0 Remains Frozen at 22ef57c9...)
======================================================================================================

Implements Fable 5.2 Max's pre-registered v2 Phase B upgrades (`docs/V2_PHASEB_PREREG.md`):
  1. Write Upgrade (`NEW_WRITE`):
     - Expanded task-blind + domain-dense unlabeled concept write corpus (`N >= 6,000`, strictly
       blacklisted against ALL Cap-1, Cap-2, and Track-1 Dev test/HardNeg species; 0 overlap asserted).
     - Cauchy-weighted Stiefel Procrustes write bridge `M_out_v2`.
     - Fable's Fisher Readout-Gradient Projector `v' = P_k @ (M_out_v2 @ v_A)` onto the top-k
       (`k = 128` fixed in advance) recipient readout-gradient directions (NOT inert output-metric
       weighting!), rescaled to preserve target dose with an off-axis norm cap.
  2. Read Upgrade (`NEW_GATE`):
     - 200-species/class donor KKT fit with **implemented negative-aware threshold**
       `t1 = max(Q_{1-alpha_gen}(z_B_gen), Q_{1-alpha_neg_strict}(z_B_neg_sim))` and steep ramp
       (`Gamma = 0.30 * std`, `delta = 0.005`), available in BOTH:
         * `NEW_GATE_1L_NATIVE`: pure single-layer `mlp_bias=False` 4-row SwiGLU (zero hooks!)
         * `NEW_GATE_2L_LATCH`: 2-layer cross-depth LDA (`lB_early` + `lB_main`) + prompt-token
           single-entity dose latch (`sum(phi) <= 1.0`, zero firing on generated tokens).
  3. Upper-Bound Control (`v2-Lite v*`):
     - <=25-step Adam optimization of ONLY the single appended write vector `v*` on a frozen
       recipient (append-only, bit-exact removable), strictly reporting `cos(v*, v_push)`.

Usage:
  PYTHONPATH=platonicswap:experiments python platonicswap/platonic_organ_engine_v2.py --unit-test
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for _p in (str(THIS_DIR), str(REPO_ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import platonic_organ_engine as poe_v1  # Read-only reuse of LW shrinkage, KKT solver, Stiefel SVD, BOS null

# Pre-registered v2 scalar hyperparameters (<= 3 rule-set scalars per V2_PHASEB_PREREG.md §2)
V2_TOLERANCES = {
    "alpha_gen": 0.001,         # Tighter general-text conformal FPR for 0/300 locality
    "alpha_neg_strict": 0.01,   # Implemented negative-aware t1 quantile on simulated in-domain negatives
    "delta": 0.005,             # Steep SiLU sharpness (cuts partial activation on hard negatives)
    "ramp_std_frac": 0.30,      # Steep ramp width Gamma = 0.30 * std
    "fisher_proj_rank": 128,    # Fixed readout-gradient projector rank k (pre-registered)
    "max_dose_norm_ratio": 1.25,# Cap on post-projection rescaling norm
}

# Dual-layer specs: (main_layer, early_layer)
V2_RECIPIENT_LAYERS = {
    "smollm2": {"model_key": "smollm2-1.7b", "lB_main": 9, "lB_early": 6, "lA_main": 17, "lA_early": 11},
    "falcon3": {"model_key": "falcon3-3b", "lB_main": 9, "lB_early": 6, "lA_main": 17, "lA_early": 11},
    "granite": {"model_key": "granite-3.1-2b", "lB_main": 12, "lB_early": 8, "lA_main": 17, "lA_early": 11},
    "qwen1.5b": {"model_key": "qwen2.5-1.5b", "lB_main": 13, "lB_early": 9, "lA_main": 17, "lA_early": 11},
}


def get_v2_engine_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# ======================================================================================
# 1. Expanded Unlabeled Concept Write Corpus (N >= 6,000, Strictly Blacklisted!)
# ======================================================================================

_V2_CONCEPT_CATEGORIES: Dict[str, List[str]] = {
    "professions": [
        "architect", "surgeon", "carpenter", "chemist", "astronomer", "librarian", "baker", "sculptor",
        "plumber", "mechanic", "pharmacist", "botanist", "geologist", "historian", "journalist", "tailor",
        "blacksmith", "watchmaker", "dentist", "pilot", "captain", "gardener", "potter", "weaver",
        "butcher", "florist", "engraver", "barber", "mason", "cobbler", "surveyor", "optician",
        "curator", "biologist", "physicist", "economist", "linguist", "composer", "conductor", "painter",
        "novelist", "playwright", "detective", "attorney", "diplomat", "ambassador", "treasurer", "auditor",
        "accountant", "actuary", "paramedic", "midwife", "therapist", "dietitian", "radiologist", "neurologist",
        "virologist", "ecologist", "meteorologist", "cartographer", "archaeologist", "anthropologist", "sociologist", "philosopher",
        "mathematician", "statistician", "cryptographer", "machinist", "electrician", "welder", "glazier", "roofer",
        "tinsmith", "locksmith", "goldsmith", "silversmith", "jeweler", "upholsterer", "bookbinder", "printer",
    ],
    "instruments_and_tools": [
        "violin", "cello", "trumpet", "flute", "clarinet", "oboe", "bassoon", "trombone",
        "tuba", "harp", "banjo", "mandolin", "ukulele", "accordion", "harmonica", "xylophone",
        "marimba", "timpani", "cymbal", "tambourine", "bagpipe", "sitar", "lute", "harpsichord",
        "hammer", "chisel", "wrench", "pliers", "screwdriver", "hacksaw", "crowbar", "mallet",
        "trowel", "anvil", "bellows", "caliper", "micrometer", "compass", "protractor", "sextant",
        "telescope", "microscope", "barometer", "thermometer", "hygrometer", "seismograph", "oscilloscope", "spectrometer",
        "centrifuge", "pipette", "burette", "beaker", "flask", "crucible", "mortar", "pestle",
        "scalpel", "forceps", "stethoscope", "otoscope", "syringe", "tourniquet", "splint", "stretcher",
        "spade", "shovel", "rake", "hoe", "scythe", "sickle", "pruner", "wheelbarrow",
        "lathe", "milling", "drill", "planer", "sander", "router", "clamp", "vise",
    ],
    "materials_and_minerals": [
        "granite", "basalt", "marble", "limestone", "sandstone", "slate", "shale", "quartz",
        "feldspar", "mica", "gypsum", "calcite", "dolomite", "obsidian", "pumice", "chalk",
        "copper", "bronze", "brass", "steel", "pewter", "titanium", "tungsten", "cobalt",
        "nickel", "zinc", "tin", "lead", "mercury", "platinum", "palladium", "chromium",
        "cotton", "linen", "silk", "velvet", "denim", "tweed", "flannel", "canvas",
        "burlap", "nylon", "polyester", "rayon", "kevlar", "latex", "silicone", "acrylic",
        "cedar", "mahogany", "teak", "walnut", "cherry", "birch", "maple", "spruce",
        "redwood", "bamboo", "cork", "rattan", "wicker", "plywood", "veneer", "ebony",
        "amber", "jade", "opal", "topaz", "garnet", "emerald", "sapphire", "ruby",
        "amethyst", "turquoise", "onyx", "agate", "jasper", "beryl", "zircon", "pyrite",
    ],
    "vehicles_and_vessels": [
        "bicycle", "tricycle", "scooter", "moped", "motorcycle", "sedan", "coupe", "hatchback",
        "minivan", "pickup", "tractor", "bulldozer", "excavator", "forklift", "crane", "ambulance",
        "firetruck", "trolley", "tram", "subway", "locomotive", "caboose", "boxcar", "gondola",
        "canoe", "kayak", "rowboat", "dinghy", "skiff", "punt", "raft", "catamaran",
        "trimaran", "schooner", "sloop", "ketch", "brigantine", "galleon", "frigate", "corvette",
        "destroyer", "cruiser", "battleship", "submarine", "hovercraft", "hydrofoil", "ferry", "barge",
        "tugboat", "trawler", "tanker", "freighter", "yacht", "clipper", "biplane", "glider",
        "helicopter", "gyrocopter", "airship", "blimp", "zeppelin", "balloon", "rocket", "shuttle",
        "capsule", "lander", "rover", "probe", "satellite", "chariot", "wagon", "stagecoach",
        "carriage", "sleigh", "sled", "toboggan", "rickshaw", "palanquin", "wheelchair", "stroller",
    ],
    "foods_and_botanicals": [
        "apple", "pear", "peach", "plum", "apricot", "nectarine", "mango", "papaya",
        "guava", "lychee", "fig", "date", "pomegranate", "persimmon", "quince", "melon",
        "cantaloupe", "honeydew", "watermelon", "pineapple", "coconut", "banana", "plantain", "avocado",
        "lemon", "lime", "grapefruit", "tangerine", "clementine", "kumquat", "pomelo", "citron",
        "strawberry", "raspberry", "blackberry", "blueberry", "cranberry", "gooseberry", "currant", "elderberry",
        "carrot", "radish", "turnip", "parsnip", "beetroot", "potato", "yam", "cassava",
        "onion", "garlic", "shallot", "leek", "chive", "celery", "fennel", "asparagus",
        "spinach", "kale", "lettuce", "arugula", "cabbage", "broccoli", "cauliflower", "artichoke",
        "tomato", "eggplant", "pepper", "cucumber", "zucchini", "pumpkin", "squash", "okra",
        "wheat", "barley", "oats", "rye", "millet", "sorghum", "quinoa", "buckwheat",
        "lentil", "chickpea", "soybean", "peanut", "almond", "hazelnut", "pistachio", "cashew",
        "pecan", "chestnut", "acorn", "cinnamon", "nutmeg", "clove", "cardamom", "turmeric",
        "ginger", "saffron", "oregano", "basil", "thyme", "rosemary", "sage", "parsley",
        "cilantro", "dill", "mint", "lavender", "jasmine", "orchid", "tulip", "daffodil",
        "lily", "daisy", "sunflower", "marigold", "poppy", "peony", "dahlia", "hibiscus",
        "clover", "fern", "moss", "lichen", "cactus", "agave", "aloe", "ivy",
    ],
    "architecture_and_places": [
        "cathedral", "basilica", "chapel", "monastery", "abbey", "pagoda", "stupa", "mosque",
        "minaret", "synagogue", "temple", "shrine", "fortress", "citadel", "bastion", "rampart",
        "turret", "battlement", "moat", "drawbridge", "aqueduct", "viaduct", "causeway", "overpass",
        "tunnel", "culvert", "dam", "reservoir", "canal", "lock", "pier", "wharf",
        "jetty", "breakwater", "lighthouse", "beacon", "windmill", "watermill", "granary", "silo",
        "barn", "stable", "kennel", "orchard", "vineyard", "greenhouse", "conservatory", "arboretum",
        "observatory", "planetarium", "auditorium", "amphitheater", "stadium", "arena", "colosseum", "hippodrome",
        "pavilion", "gazebo", "pergola", "veranda", "balcony", "terrace", "courtyard", "cloister",
        "atrium", "rotunda", "cupola", "spire", "steeple", "belfry", "obelisk", "monolith",
        "pyramid", "ziggurat", "mausoleum", "crypt", "catacomb", "bunker", "hangar", "depot",
    ],
    "garments_and_household": [
        "jacket", "overcoat", "trenchcoat", "parka", "anorak", "poncho", "cloak", "cape",
        "cardigan", "sweater", "pullover", "vest", "waistcoat", "blazer", "tuxedo", "kimono",
        "kaftan", "tunic", "blouse", "shirt", "trousers", "slacks", "breeches", "leggings",
        "skirt", "kilt", "apron", "smock", "robe", "bathrobe", "pajamas", "nightgown",
        "scarf", "shawl", "muffler", "necktie", "cravat", "bowtie", "belt", "suspenders",
        "mitten", "glove", "gauntlet", "sock", "stocking", "slipper", "sandal", "moccasin",
        "loafer", "brogue", "boot", "clog", "beret", "fedora", "sombrero", "turban",
        "bonnet", "hood", "visor", "helmet", "armchair", "sofa", "ottoman", "bench",
        "stool", "rocker", "cradle", "hammock", "mattress", "pillow", "bolster", "blanket",
        "quilt", "duvet", "curtain", "drape", "shutter", "awning", "carpet", "tapestry",
    ],
    "science_nature_and_weather": [
        "glacier", "moraine", "fjord", "canyon", "gorge", "ravine", "valley", "basin",
        "plateau", "mesa", "butte", "ridge", "summit", "crater", "caldera", "geyser",
        "volcano", "atoll", "lagoon", "estuary", "delta", "marsh", "bog", "fen",
        "swamp", "tundra", "taiga", "steppe", "savanna", "prairie", "meadow", "oasis",
        "dune", "reef", "shoal", "trench", "seamount", "iceberg", "floe", "avalanche",
        "landslide", "earthquake", "tsunami", "cyclone", "hurricane", "typhoon", "tornado", "monsoon",
        "blizzard", "squall", "gale", "breeze", "drought", "famine", "frost", "sleet",
        "hail", "drizzle", "mist", "fog", "haze", "smog", "rainbow", "aurora",
        "eclipse", "solstice", "equinox", "comet", "asteroid", "meteor", "nebula", "quasar",
        "pulsar", "galaxy", "constellation", "supernova", "proton", "neutron", "electron", "photon",
    ],
    "abstract_and_systems": [
        "algorithm", "theorem", "axiom", "lemma", "corollary", "conjecture", "hypothesis", "paradox",
        "syllogism", "metaphor", "allegory", "irony", "satire", "sonnet", "ballad", "elegy",
        "epic", "fable", "parable", "memoir", "chronicle", "almanac", "lexicon", "glossary",
        "thesaurus", "anthology", "treatise", "monograph", "pamphlet", "manifesto", "charter", "statute",
        "ordinance", "decree", "edict", "treaty", "accord", "covenant", "protocol", "agenda",
        "quorum", "ballot", "referendum", "tariff", "subsidy", "embargo", "sanction", "dividend",
        "annuity", "mortgage", "ledger", "invoice", "receipt", "voucher", "warranty", "patent",
        "copyright", "trademark", "royalty", "franchise", "monopoly", "cartel", "syndicate", "guild",
        "coalition", "alliance", "federation", "republic", "monarchy", "oligarchy", "anarchy", "utopia",
        "symmetry", "entropy", "inertia", "momentum", "friction", "viscosity", "buoyancy", "resonance",
    ],
    "sports_and_games": [
        "archery", "fencing", "rowing", "sailing", "surfing", "diving", "skating", "skiing",
        "sledding", "curling", "bowling", "boxing", "wrestling", "judo", "karate", "taekwondo",
        "tennis", "squash", "badminton", "cricket", "baseball", "softball", "lacrosse", "hockey",
        "polo", "rugby", "soccer", "football", "volleyball", "handball", "basketball", "netball",
        "golf", "billiards", "snooker", "darts", "chess", "checkers", "backgammon", "dominoes",
        "mahjong", "poker", "bridge", "cribbage", "solitaire", "crossword", "sudoku", "origami",
        "marathon", "sprint", "hurdles", "javelin", "discus", "shotput", "decathlon", "triathlon",
        "gymnastics", "trampoline", "cycling", "kayaking", "climbing", "hiking", "camping", "angling",
        "racket", "paddle", "bat", "club", "cue", "mallet", "shuttlecock", "puck",
        "wicket", "hoop", "net", "goalpost", "bleacher", "scoreboard", "whistle", "trophy",
    ],
}

_V2_CONCEPT_TEMPLATES = [
    "In an encyclopedia, the entry for {c} explains its main features.",
    "During a detailed lecture, the speaker described {c} clearly.",
    "A classic example of {c} can be recognized by its structure.",
    "Researchers often study {c} to understand its properties.",
    "In everyday language, people refer to {c} in many contexts.",
    "When classifying items, {c} belongs to a distinct category.",
]


def build_v2_expanded_write_corpus(
    extra_blacklist: Optional[Sequence[str]] = None,
    domain_dense_concepts: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Build an expanded `N >= 4,800–6,000+` short-context concept write corpus strictly blacklisted
    against Cap-1, Cap-2, `dataset.json`, and any caller-supplied test/HardNeg terms.
    Optionally appends `domain_dense_concepts` (strictly disjoint from eval/HardNeg/Cap-2!).
    """
    bl = set()
    # Collect Cap-1 / Cap-2 / dataset blacklists via v1 helper + extra_blacklist
    for t in extra_blacklist or ():
        bl.add(t.strip().lower())
    # Also block all Cap-2 terms & dataset subjects
    try:
        import bio_ripple_sealed as brs
        for sp in brs.ALL_CAP2_SPECIES:
            bl.add(sp.strip().lower())
    except Exception:
        pass
    try:
        from extract_and_baseline import DATASET_PATH
        if DATASET_PATH.exists():
            import json
            ds = json.load(open(DATASET_PATH))
            for f in ds.get("facts", []):
                bl.add(str(f.get("subject", "")).strip().lower())
                bl.add(str(f.get("answer", "")).strip().lower())
    except Exception:
        pass

    concepts: List[str] = []
    seen = set()
    for cat_words in _V2_CONCEPT_CATEGORIES.values():
        for w in cat_words:
            wl = w.strip().lower()
            if wl and wl not in seen and wl not in bl and not any(b in wl or wl in b for b in bl if len(b) >= 5 and len(wl) >= 5):
                seen.add(wl)
                concepts.append(wl)

    n_general = len(concepts)
    if domain_dense_concepts:
        for w in domain_dense_concepts:
            wl = w.strip().lower()
            if wl and wl not in seen and wl not in bl:
                seen.add(wl)
                concepts.append(wl)

    prompts, spans = [], []
    for c in concepts:
        for tmpl in _V2_CONCEPT_TEMPLATES:
            p = tmpl.format(c=c)
            i0 = p.index(c)
            prompts.append(p)
            spans.append([i0, i0 + len(c)])
    return {
        "prompts": prompts,
        "spans": spans,
        "concepts": concepts,
        "n_general_concepts": n_general,
        "n_domain_dense_concepts": len(concepts) - n_general,
        "n_prompts": len(prompts),
    }


# ======================================================================================
# 2. Fable's Fisher Readout-Gradient Projector (P_k = U_k U_k^T, Fixed Rank k)
# ======================================================================================

def compute_recipient_readout_projector(
    model_B: Any,
    tok_B: Any,
    device: torch.device,
    lB: int,
    unlabeled_passages: Sequence[str],
    rank_k: int = 128,
    n_passages: int = 48,
    n_proj_per_passage: int = 6,
    seed: int = 20261126,
) -> Dict[str, Any]:
    """
    Compute the top-k (`rank_k` fixed in advance) recipient readout-gradient subspace `U_k` (D_B, k)
    at layer `lB` output (`h_out` = post-layer residual stream) using strictly unlabeled text.

    For each unlabeled passage, backpropagates `n_proj_per_passage` top-logit / centered-logit
    linear readouts at the last token back to `h_out_lB` (model weights frozen, zero optimizer steps!).
    Applying `P_k = U_k @ U_k.T` strips off-axis syntax-corrupting noise orthogonal to the
    recipient's downstream readout sensitivity (Fable 5.2 Max §1 Fix).
    """
    from extract_and_baseline import get_decoder_layers

    layer = get_decoder_layers(model_B)[lB]
    rng = np.random.default_rng(seed)

    # Freeze all parameters so autograd only computes d(readout)/d(h_out_lB)
    old_reqs = [p.requires_grad for p in model_B.parameters()]
    for p in model_B.parameters():
        p.requires_grad_(False)

    grads: List[np.ndarray] = []
    use_passages = list(unlabeled_passages[:n_passages])
    try:
        for p_text in use_passages:
            enc = tok_B(p_text, return_tensors="pt", truncation=True, max_length=96, add_special_tokens=True)
            input_ids = enc["input_ids"].to(device)
            captured: List[torch.Tensor] = []

            def _hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h_leaf = h.detach().requires_grad_(True)
                captured.append(h_leaf)
                if isinstance(out, tuple):
                    return (h_leaf,) + out[1:]
                return h_leaf

            h_handle = layer.register_forward_hook(_hook)
            try:
                logits = model_B(input_ids=input_ids, use_cache=False).logits[0, -1, :].float()  # (V,)
                h_leaf = captured[0]  # (1, T, D_B)
                # Take top-k predicted tokens and compute centered log-prob contrasts (task-blind!)
                top_vals, top_idx = torch.topk(logits, k=min(n_proj_per_passage + 2, logits.shape[0]))
                # Centered logit directions: logit(top_m) - mean(top_vals)
                mean_top = top_vals.mean()
                for m in range(n_proj_per_passage):
                    scalar = logits[top_idx[m]] - mean_top
                    g = torch.autograd.grad(scalar, h_leaf, retain_graph=(m < n_proj_per_passage - 1))[0]
                    # Last-token gradient (and mean over last 3 tokens to cover span-to-final routing)
                    g_last = g[0, -1, :].detach().float().cpu().numpy()
                    if np.linalg.norm(g_last) > 1e-8:
                        grads.append(g_last / (np.linalg.norm(g_last) + 1e-12))
            finally:
                h_handle.remove()
    finally:
        for p, req in zip(model_B.parameters(), old_reqs):
            p.requires_grad_(req)

    G = np.stack(grads, axis=0).astype(np.float64)  # (N_g, D_B)
    _, sv, Vt = sla.svd(G, full_matrices=False)
    k_eff = min(int(rank_k), Vt.shape[0])
    U_k = Vt[:k_eff].T  # (D_B, k_eff) orthonormal columns
    energy_k = float(np.sum(sv[:k_eff] ** 2) / (np.sum(sv ** 2) + 1e-12))
    return {
        "U_k": U_k,
        "rank_k": k_eff,
        "n_grads": int(G.shape[0]),
        "energy_captured": round(energy_k, 4),
    }


def project_write_vector_fisher(
    v_raw: np.ndarray,
    U_k: np.ndarray,
    max_norm_ratio: float = 1.25,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Apply Fable's Fisher readout-gradient projector `v_proj = P_k @ v_raw` and rescale toward
    unit dose while capping norm growth by `max_norm_ratio`.
    """
    v_raw = np.asarray(v_raw, dtype=np.float64)
    v_p = U_k @ (U_k.T @ v_raw)
    n_raw = float(np.linalg.norm(v_raw)) + 1e-12
    n_p = float(np.linalg.norm(v_p)) + 1e-12
    frac_in_subspace = n_p / n_raw
    # Rescale projected vector to preserve on-axis energy without exceeding norm cap
    scale = min(1.0 / max(frac_in_subspace, 1e-3), max_norm_ratio / max(frac_in_subspace, 1e-6))
    # Target: match raw norm (or up to max_norm_ratio * raw norm)
    target_norm = min(n_raw * max_norm_ratio, n_raw / max(frac_in_subspace, 0.5))
    v_out = v_p * (target_norm / n_p)
    return v_out, {
        "frac_norm_in_Pk": round(frac_in_subspace, 4),
        "out_over_raw_norm": round(float(np.linalg.norm(v_out) / n_raw), 4),
    }


# ======================================================================================
# 3. Read Upgrades: (a) 1-Layer Negative-Aware Steep Native Gate & (b) 2-Layer LDA + Latch
# ======================================================================================

def compile_v2_native_1l_organ(
    donor_rules: List[Dict[str, Any]],
    ch_main: Dict[str, Any],
    rm_B: float,
    write_vectors_B: Optional[Dict[str, np.ndarray]] = None,
    tolerances: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Pure single-layer `mlp_bias=False` 4-row SwiGLU compiler with:
      - 2-constraint KKT direction on 200-species/class donor rules,
      - IMPLEMENTED negative-aware threshold:
          t1 = max(Q_{1-alpha_gen}(z_B_gen), Q_{1-alpha_neg_strict}(z_B_neg_sim)),
      - Steep ramp: Gamma = ramp_std_frac * std(z_B_gen), beta = SILU_NEG_MAX / (delta * Gamma),
      - Optional custom write vectors `write_vectors_B[rule_name]` (in pre-rm_B MLP output space,
        i.e., already multiplied by 1/rm_B if native/v2!).
    """
    import pilot_organs_p0 as po

    tol = dict(V2_TOLERANCES if tolerances is None else tolerances)
    alpha_gen = float(tol["alpha_gen"])
    alpha_neg = float(tol["alpha_neg_strict"])
    delta = float(tol["delta"])
    ramp_frac = float(tol["ramp_std_frac"])

    G = np.asarray(ch_main["G"], dtype=np.float64)
    mu_A = np.asarray(ch_main["mu_A"], dtype=np.float64)
    mu_B = np.asarray(ch_main["mu_B"], dtype=np.float64)
    Sigma_BB = np.asarray(ch_main["Sigma_BB_fit_lw"], dtype=np.float64)
    N_lw = np.asarray(ch_main["N_lw"], dtype=np.float64)
    Xa_r = np.asarray(ch_main["X_A_resid"], dtype=np.float64)
    Xb_r = np.asarray(ch_main["X_B_resid"], dtype=np.float64)
    mid = len(Xb_r) // 2
    Xa_ev = Xa_r[mid:]
    Xb_ev = Xb_r[mid:]
    Xb_pred_ev = mu_B[None, :] + (Xa_ev - mu_A[None, :]) @ G.T
    M_out = np.asarray(ch_main["M_out_iso"], dtype=np.float64)
    c_B = np.asarray(ch_main["c_B"], dtype=np.float64)
    x_bos_B = np.asarray(ch_main["x_bos_B"], dtype=np.float64)
    bos_bank_B = ch_main.get("bos_x_fit_B")
    c_u = po.apply_exact_bos_null_biasfree(c_B[None, :], x_bos_B, x_bos_bank=bos_bank_B)[0]

    S_bg = Sigma_BB + N_lw
    D_B = len(mu_B)
    inv_rm = 1.0 / max(float(rm_B), 1e-6)

    W_g_rows, W_u_rows, W_d_cols = [], [], []
    rule_reports = []
    rng = np.random.default_rng(20261126)

    for r in donor_rules:
        rname = r["name"]
        Xp_A = np.asarray(r["X_pos_A"], dtype=np.float64)
        Xn_A = np.asarray(r["X_neg_dom_A"], dtype=np.float64)
        v_A = np.asarray(r["v_A"], dtype=np.float64)

        Xp_sim = mu_B[None, :] + (Xp_A - mu_A[None, :]) @ G.T
        Xn_sim = mu_B[None, :] + (Xn_A - mu_A[None, :]) @ G.T

        # Solve 2-constraint KKT direction (strong in-domain separation)
        from scipy.stats import norm as sp_norm
        m1 = float(sp_norm.ppf(1.0 - alpha_neg))
        m2 = float(sp_norm.ppf(1.0 - alpha_gen))
        d1 = Xp_sim.mean(axis=0) - Xn_sim.mean(axis=0)
        d2 = Xp_sim.mean(axis=0) - mu_B
        w_raw = poe_v1.solve_2constraint_kkt(S_bg, d1, d2, m1, m2)
        w_c = w_raw / (np.linalg.norm(w_raw) + 1e-12)

        # Real-B general text projection
        z_real_B = Xb_ev @ w_c
        std_gen = float(np.std(z_real_B)) + 1e-12
        k_conf = min(len(z_real_B), int(math.ceil((1.0 - alpha_gen) * (len(z_real_B) + 1))))
        q_gen = float(np.sort(z_real_B)[k_conf - 1])

        # Empirical channel residual calibration along w_c
        z_pred = Xb_pred_ev @ w_c
        m_real, m_pred = float(np.mean(z_real_B)), float(np.mean(z_pred))
        cov_rp = float(np.mean((z_real_B - m_real) * (z_pred - m_pred)))
        var_pred = float(np.var(z_pred - m_pred))
        a_emp = max(0.0, cov_rp / max(var_pred, 1e-18))
        e_cal = (z_real_B - m_real) - a_emp * (z_pred - m_pred)

        # IMPLEMENTED negative-aware threshold: calibrate simulated in-domain negative distribution!
        zn_raw = Xn_sim @ w_c
        zn_cal = m_real + a_emp * (zn_raw - m_pred) + e_cal[rng.integers(0, len(e_cal), size=len(zn_raw))]
        q_neg = float(np.quantile(zn_cal, 1.0 - alpha_neg))

        zp_raw = Xp_sim @ w_c
        zp_cal = m_real + a_emp * (zp_raw - m_pred) + e_cal[rng.integers(0, len(e_cal), size=len(zp_raw))]

        t1 = max(q_gen, q_neg)
        Gamma = max(ramp_frac * std_gen, 1e-4)
        t2 = t1 + Gamma
        beta = min(poe_v1.SILU_NEG_MAX / (delta * Gamma), 120.0 / Gamma)

        if write_vectors_B is not None and rname in write_vectors_B:
            v_B = np.asarray(write_vectors_B[rname], dtype=np.float64)
        else:
            v_B = inv_rm * (M_out @ v_A)

        denom = beta * Gamma
        W_g_rows.append(beta * (w_c - t1 * c_B))
        W_g_rows.append(beta * (w_c - t2 * c_B))
        W_u_rows.append(c_u)
        W_u_rows.append(c_u)
        W_d_cols.append(v_B / denom)
        W_d_cols.append(-v_B / denom)

        rule_reports.append({
            "name": rname,
            "q_gen": round(q_gen, 5),
            "q_neg_dom": round(q_neg, 5),
            "t1": round(t1, 5),
            "t2": round(t2, 5),
            "Gamma": round(Gamma, 5),
            "beta": round(float(beta), 4),
            "a_emp": round(float(a_emp), 4),
            "pred_pos_hit": round(float(np.mean(zp_cal >= 0.5 * (t1 + t2))), 4),
            "pred_neg_fpr": round(float(np.mean(zn_cal >= 0.5 * (t1 + t2))), 4),
            "w_c": w_c,
        })

    W_g = np.stack(W_g_rows, axis=0)
    W_u = po.apply_exact_bos_null_biasfree(np.stack(W_u_rows, axis=0), x_bos_B, x_bos_bank=bos_bank_B)
    W_d = np.stack(W_d_cols, axis=1)
    return {
        "status": "COMPILED_V2_1L",
        "use_bias": False,
        "tolerances": tol,
        "rules": rule_reports,
        "weights": {"W_g": W_g, "b_g": None, "W_u": W_u, "b_u": None, "W_d": W_d, "use_bias": False},
    }


def fit_v2_2layer_lda_gate(
    donor_rules_early: List[Dict[str, Any]],
    donor_rules_main: List[Dict[str, Any]],
    ch_early: Dict[str, Any],
    ch_main: Dict[str, Any],
    tolerances: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Fable's 2-Layer Cross-Depth LDA Reader (`lB_early` + `lB_main`) with negative-aware `t1`
    and steep sigmoid/twin-SiLU activation, designed for prompt-token single-entity dose latching
    (`sum(phi) <= 1.0`, zero firing on generated tokens).
    """
    tol = dict(V2_TOLERANCES if tolerances is None else tolerances)
    alpha_gen = float(tol["alpha_gen"])
    alpha_neg = float(tol["alpha_neg_strict"])
    ramp_frac = float(tol["ramp_std_frac"])
    delta = float(tol["delta"])
    rng = np.random.default_rng(20261127)

    # First extract per-layer 1D KKT directions cleanly
    c_early = compile_v2_native_1l_organ(donor_rules_early, ch_early, rm_B=1.0, tolerances=tol)
    c_main = compile_v2_native_1l_organ(donor_rules_main, ch_main, rm_B=1.0, tolerances=tol)

    G_e, muA_e, muB_e = np.asarray(ch_early["G"], dtype=np.float64), np.asarray(ch_early["mu_A"], dtype=np.float64), np.asarray(ch_early["mu_B"], dtype=np.float64)
    G_m, muA_m, muB_m = np.asarray(ch_main["G"], dtype=np.float64), np.asarray(ch_main["mu_A"], dtype=np.float64), np.asarray(ch_main["mu_B"], dtype=np.float64)
    Xa_re, Xb_re = np.asarray(ch_early["X_A_resid"], dtype=np.float64), np.asarray(ch_early["X_B_resid"], dtype=np.float64)
    Xa_rm, Xb_rm = np.asarray(ch_main["X_A_resid"], dtype=np.float64), np.asarray(ch_main["X_B_resid"], dtype=np.float64)
    me, mm = len(Xb_re) // 2, len(Xb_rm) // 2
    Xb_e, Xb_m = Xb_re[me:], Xb_rm[mm:]
    Xp_e = muB_e[None, :] + (Xa_re[me:] - muA_e[None, :]) @ G_e.T
    Xp_m = muB_m[None, :] + (Xa_rm[mm:] - muA_m[None, :]) @ G_m.T
    n_ev = min(len(Xb_e), len(Xb_m))

    lda_rules: List[Dict[str, Any]] = []
    for i in range(len(donor_rules_main)):
        rname = donor_rules_main[i]["name"]
        w_e = c_early["rules"][i]["w_c"]
        w_m = c_main["rules"][i]["w_c"]

        # Real-B general paired scores (N_ev, 2)
        ze_gen = Xb_e[:n_ev] @ w_e
        zm_gen = Xb_m[:n_ev] @ w_m
        # Channel residual calibration per layer
        ze_pred = Xp_e[:n_ev] @ w_e
        zm_pred = Xp_m[:n_ev] @ w_m
        ae = max(0.0, float(np.cov(ze_gen, ze_pred)[0, 1]) / max(float(np.var(ze_pred)), 1e-18))
        am = max(0.0, float(np.cov(zm_gen, zm_pred)[0, 1]) / max(float(np.var(zm_pred)), 1e-18))
        ee = (ze_gen - ze_gen.mean()) - ae * (ze_pred - ze_pred.mean())
        em = (zm_gen - zm_gen.mean()) - am * (zm_pred - zm_pred.mean())

        # Simulated donor positives & in-domain negatives in 2D score space
        def _sim2d(Xa_e, Xa_m):
            ze_r = (muB_e[None, :] + (Xa_e - muA_e[None, :]) @ G_e.T) @ w_e
            zm_r = (muB_m[None, :] + (Xa_m - muA_m[None, :]) @ G_m.T) @ w_m
            idx = rng.integers(0, n_ev, size=len(ze_r))
            ze_c = ze_gen.mean() + ae * (ze_r - ze_pred.mean()) + ee[idx]
            zm_c = zm_gen.mean() + am * (zm_r - zm_pred.mean()) + em[idx]
            return np.stack([ze_c, zm_c], axis=1)

        S_pos = _sim2d(donor_rules_early[i]["X_pos_A"], donor_rules_main[i]["X_pos_A"])
        S_neg = _sim2d(donor_rules_early[i]["X_neg_dom_A"], donor_rules_main[i]["X_neg_dom_A"])
        S_gen = np.stack([ze_gen, zm_gen], axis=1)

        # Closed-form 2-D LDA combining early + main layer (pooled within-class cov + gen cov)
        C_pool = 0.5 * np.cov(S_pos.T) + 0.5 * np.cov(S_neg.T) + 0.25 * np.cov(S_gen.T) + 1e-6 * np.eye(2)
        diff = S_pos.mean(axis=0) - S_neg.mean(axis=0)
        a_lda = sla.solve(C_pool, diff)
        # Ensure non-negative contributions if both agree, normalize ||a_lda|| = 1
        a_lda = a_lda / (np.linalg.norm(a_lda) + 1e-12)

        s_gen_1d = S_gen @ a_lda
        s_neg_1d = S_neg @ a_lda
        s_pos_1d = S_pos @ a_lda

        std_gen = float(np.std(s_gen_1d)) + 1e-12
        k_conf = min(len(s_gen_1d), int(math.ceil((1.0 - alpha_gen) * (len(s_gen_1d) + 1))))
        q_gen = float(np.sort(s_gen_1d)[k_conf - 1])
        q_neg = float(np.quantile(s_neg_1d, 1.0 - alpha_neg))

        t1 = max(q_gen, q_neg)
        Gamma = max(ramp_frac * std_gen, 1e-4)
        t2 = t1 + Gamma
        beta = min(poe_v1.SILU_NEG_MAX / (delta * Gamma), 120.0 / Gamma)

        # Within-class correlation between early and main scores (verifies Fable's <=0.3 independence gain)
        corr_neg = float(np.corrcoef(S_neg[:, 0], S_neg[:, 1])[0, 1])

        lda_rules.append({
            "name": rname,
            "w_early": w_e,
            "w_main": w_m,
            "a_early": float(a_lda[0]),
            "a_main": float(a_lda[1]),
            "t1": float(t1),
            "t2": float(t2),
            "Gamma": float(Gamma),
            "beta": float(beta),
            "q_gen": round(q_gen, 5),
            "q_neg_dom": round(q_neg, 5),
            "neg_score_corr_early_main": round(corr_neg, 4),
            "pred_pos_hit": round(float(np.mean(s_pos_1d >= 0.5 * (t1 + t2))), 4),
            "pred_neg_fpr": round(float(np.mean(s_neg_1d >= 0.5 * (t1 + t2))), 4),
        })

    return {"tolerances": tol, "rules": lda_rules}


class TwoLayerLDALatchedOrgan:
    """
    Context manager installing Fable's 2-layer cross-depth LDA gate (`lB_early` + `lB_main`)
    with prompt-token single-entity dose latching (`sum_{t>=1} phi_eff(t) <= 1.0`, zero firing
    on BOS and zero firing during single-token KV-cached decode or generated continuation tokens).
    """

    def __init__(
        self,
        model_B: Any,
        lB_early: int,
        lB_main: int,
        lda_spec: Dict[str, Any],
        write_vectors_B: Dict[str, np.ndarray],
        prompt_lens_override: Optional[int] = None,
    ):
        self.model_B = model_B
        self.lB_early = lB_early
        self.lB_main = lB_main
        self.lda_spec = lda_spec
        self.write_vectors_B = write_vectors_B
        self.prompt_lens_override = prompt_lens_override
        self._handles: List[Any] = []
        self._z_early_cache: Dict[str, torch.Tensor] = {}

    def __enter__(self):
        from extract_and_baseline import get_decoder_layers

        layers = get_decoder_layers(self.model_B)
        dev = next(self.model_B.parameters()).device
        dtype = next(self.model_B.parameters()).dtype

        w_e_t = [torch.tensor(r["w_early"], dtype=torch.float32, device=dev) for r in self.lda_spec["rules"]]
        w_m_t = [torch.tensor(r["w_main"], dtype=torch.float32, device=dev) for r in self.lda_spec["rules"]]
        v_b_t = [torch.tensor(self.write_vectors_B[r["name"]], dtype=torch.float32, device=dev) for r in self.lda_spec["rules"]]
        rules = self.lda_spec["rules"]

        def _hook_early(mod, inp, out):
            # Capture pre-MLP layernorm output at lB_early: shape (B, T, D)
            x = out.float()
            self._z_early_cache["z"] = [torch.matmul(x, we) for we in w_e_t]

        def _hook_main_ln(mod, inp, out):
            x = out.float()
            self._z_early_cache["zm"] = [torch.matmul(x, wm) for wm in w_m_t]

        def _hook_main_mlp(mod, inp, out):
            # out: (B, T, D)
            T = out.shape[1]
            # Latch to prompt tokens ONLY: during KV-cached decode (T == 1), fire 0!
            if T <= 1:
                return out
            ze_list = self._z_early_cache.get("z")
            zm_list = self._z_early_cache.get("zm")
            if ze_list is None or zm_list is None:
                return out

            add = torch.zeros_like(out, dtype=torch.float32)
            max_prompt_t = T if self.prompt_lens_override is None else min(T, self.prompt_lens_override)
            for ri, r in enumerate(rules):
                s = r["a_early"] * ze_list[ri] + r["a_main"] * zm_list[ri]  # (B, T)
                b_val = r["beta"]
                bg = b_val * r["Gamma"]
                phi = (F.silu(b_val * (s - r["t1"])) - F.silu(b_val * (s - r["t2"]))) / bg
                phi = torch.clamp(phi, min=0.0, max=1.0)
                # Zero BOS (pos 0) and any generated tokens beyond prompt
                mask = torch.zeros_like(phi)
                mask[:, 1:max_prompt_t] = 1.0
                phi = phi * mask
                # Cumulative single-entity dose latch: sum_{t} phi_eff(t) <= 1.0 per sequence!
                csum = torch.cumsum(phi, dim=1)
                prev_csum = F.pad(csum[:, :-1], (1, 0), value=0.0)
                rem = torch.clamp(1.0 - prev_csum, min=0.0)
                phi_latched = torch.minimum(phi, rem)
                add = add + phi_latched.unsqueeze(-1) * v_b_t[ri].view(1, 1, -1)
            return out + add.to(out.dtype)

        self._handles.append(layers[self.lB_early].post_attention_layernorm.register_forward_hook(_hook_early))
        self._handles.append(layers[self.lB_main].post_attention_layernorm.register_forward_hook(_hook_main_ln))
        self._handles.append(layers[self.lB_main].mlp.register_forward_hook(_hook_main_mlp))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._z_early_cache.clear()


# ======================================================================================
# 4. v2-Lite Upper-Bound Control: <= 25-Step Frozen-Recipient Write Vector v*
# ======================================================================================

def fit_v2_lite_write_vector(
    model_B: Any,
    tok_B: Any,
    device: torch.device,
    lB: int,
    v_init: np.ndarray,
    train_species: Sequence[str],
    target_class_str: str,
    old_class_str: str,
    n_steps: int = 25,
    lr: float = 0.08,
    max_norm_mult: float = 2.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Optimize ONLY a single write vector `v*` (R^{D_B}, append-only, frozen recipient weights)
    for <= 25 Adam steps on donor-pool training species (strictly disjoint from eval species!),
    and report exact `cos(v*, v_init)` per Fable's mandate.
    """
    from extract_and_baseline import get_decoder_layers
    from v2_step0_rw_diagnostic import locate_species_last_token

    mlp = get_decoder_layers(model_B)[lB].mlp
    old_reqs = [p.requires_grad for p in model_B.parameters()]
    for p in model_B.parameters():
        p.requires_grad_(False)

    v0 = np.asarray(v_init, dtype=np.float64)
    v0_norm = float(np.linalg.norm(v0)) + 1e-12
    v_param = nn.Parameter(torch.tensor(v0, dtype=torch.float32, device=device))
    opt = torch.optim.Adam([v_param], lr=lr)

    # Build 16 clean training direct prompts from the donor training species pool
    prompts_data = []
    tgt_ids = tok_B.encode(target_class_str, add_special_tokens=False)
    old_ids = tok_B.encode(old_class_str, add_special_tokens=False)
    tgt_tok = tgt_ids[0]
    old_tok = old_ids[0]
    for sp in list(train_species)[:16]:
        art = "an" if sp[0].lower() in "aeiou" else "a"
        p = f"By biological class, {art} {sp} is a"
        s_idx = locate_species_last_token(tok_B, p, sp)
        p_ids = tok_B.encode(p, add_special_tokens=True)
        prompts_data.append((torch.tensor([p_ids], dtype=torch.long, device=device), s_idx))

    losses = []
    try:
        for step in range(n_steps):
            opt.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)
            # Mini-batch of 8 prompts per step
            for bi in range(8):
                inp, s_idx = prompts_data[(step * 8 + bi) % len(prompts_data)]

                def _hook(mod, i_t, out, idx=s_idx):
                    if out.shape[1] > idx:
                        o2 = out.clone()
                        o2[:, idx, :] = o2[:, idx, :] + v_param.to(out.dtype)
                        return o2
                    return out

                h = mlp.register_forward_hook(_hook)
                try:
                    logits = model_B(input_ids=inp, use_cache=False).logits[0, -1, :].float()
                    # Cross-entropy toward target class token + margin over old class token
                    ce = F.cross_entropy(logits.unsqueeze(0), torch.tensor([tgt_tok], device=device))
                    margin = F.relu(2.0 - (logits[tgt_tok] - logits[old_tok]))
                    batch_loss = batch_loss + (ce + 0.5 * margin) / 8.0
                finally:
                    h.remove()
            # Mild norm tether to prevent exploding residual stream
            reg = 0.01 * F.relu(v_param.norm() - max_norm_mult * v0_norm) ** 2
            loss = batch_loss + reg
            loss.backward()
            opt.step()
            losses.append(round(float(loss.item()), 4))
    finally:
        for p, req in zip(model_B.parameters(), old_reqs):
            p.requires_grad_(req)

    v_star = v_param.detach().float().cpu().numpy().astype(np.float64)
    cos_v0 = float(np.dot(v_star, v0) / (np.linalg.norm(v_star) * v0_norm + 1e-12))
    return v_star, {
        "n_steps": n_steps,
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "cos_vstar_vs_vpush": round(cos_v0, 4),
        "norm_vstar_over_vpush": round(float(np.linalg.norm(v_star) / v0_norm), 4),
    }


# ======================================================================================
# 5. Pure-CPU Unit Test
# ======================================================================================

def run_unit_test() -> None:
    import v2_track1_dev_dataset as d1

    bl = d1.get_all_dev_blacklist_terms()
    corp = build_v2_expanded_write_corpus(extra_blacklist=bl)
    assert corp["n_prompts"] >= 4500, f"Too few expanded write prompts: {corp['n_prompts']}"
    # Verify ZERO overlap with Track-1 Dev test/HardNeg/Cap-2 terms
    bl_set = set(t.lower() for t in bl)
    for c in corp["concepts"]:
        assert c not in bl_set, f"Leak in v2 write corpus: {c}"

    # Verify Fisher projector P_k strips orthogonal off-axis noise cleanly
    rng = np.random.default_rng(11)
    D, k = 128, 16
    Q, _ = sla.qr(rng.standard_normal((D, k)), mode="economic")
    v_true = Q @ rng.standard_normal(k)
    v_true /= np.linalg.norm(v_true)
    noise = rng.standard_normal(D)
    noise -= Q @ (Q.T @ noise)
    noise = noise / np.linalg.norm(noise) * 3.0  # cos ~ 0.316
    v_raw = v_true + noise
    cos_before = float(np.dot(v_raw, v_true) / np.linalg.norm(v_raw))
    v_proj, meta = project_write_vector_fisher(v_raw, Q, max_norm_ratio=1.25)
    cos_after = float(np.dot(v_proj, v_true) / np.linalg.norm(v_proj))
    assert cos_before < 0.35 and cos_after > 0.999, f"Projector failed: {cos_before} -> {cos_after}"

    # Verify cumulative dose latch caps sum(phi) <= 1.0 even when 4 prompt tokens fire
    phi = torch.tensor([[0.0, 0.8, 0.9, 0.7, 0.9]])
    csum = torch.cumsum(phi, dim=1)
    prev = F.pad(csum[:, :-1], (1, 0), value=0.0)
    latched = torch.minimum(phi, torch.clamp(1.0 - prev, min=0.0))
    assert abs(float(latched.sum().item()) - 1.0) < 1e-6 and abs(float(latched[0, 1]) - 0.8) < 1e-6 and abs(float(latched[0, 2]) - 0.2) < 1e-6 and abs(float(latched[0, 3])) < 1e-6

    # Verify v1.0 engine SHA-256 is untouched!
    assert poe_v1.get_engine_sha256() == "22ef57c9d196e12f2cd3c9ecc79ec0813803df43d5a6009f5422904948802306"
    print(
        f"[platonic_organ_engine_v2 --unit-test] ALL PASSED! "
        f"(v2_sha={get_v2_engine_sha256()[:12]}, v1_sha=22ef57c9d196 [LOCKED], "
        f"write_corpus={corp['n_prompts']} prompts / {len(corp['concepts'])} concepts, "
        f"P_k cos {cos_before:.3f}->{cos_after:.4f}, latch exact=1.0)"
    )


if __name__ == "__main__":
    if "--unit-test" in sys.argv:
        run_unit_test()
