#!/usr/bin/env python3
"""
bio_ripple_sealed.py — Strictly Sealed Capability 2 Benchmark (`BIO-RIPPLE`)
=============================================================================

Counterfactual Biological Taxonomy Rewrite (`Bird -> Fish`, `Mammal -> Reptile`)
strictly sealed against hyperparameter tuning and 100% disjoint from Capability 1
(Geography `{France -> Japan, Italy -> Brazil}`) and `dataset.json`.

Structure (mirroring Cap-1's 2-rule, 4-native-SwiGLU-row footprint):
  - Rule 1 (`Bird -> Fish`):    24 train flying birds + 24 strictly unseen test flying birds
  - Rule 2 (`Mammal -> Reptile`): 24 train land mammals + 24 strictly unseen test land mammals
  - 24 train in-domain non-target animals (amphibians, insects, arachnids, mollusks, fish, reptiles)
  - 24 strictly unseen hard negatives (flying non-birds, aquatic mammals, non-target reptiles/fish/insects)

Zero-Leakage & Anti-Confound Guarantees:
  1. Donor-A construction uses ONLY 6 frame-counterbalanced class-membership templates
     (ZERO ripple attributes in construction; exact `frame_only_auroc == 0.5000`!).
  2. Donor-A write displacements `v_A` are extracted ONLY from generic class nouns
     (`"fish" - "bird"` and `"reptile" - "mammal"`) in neutral frames — ZERO test species!
  3. Sealed Evaluation Battery:
       - 48 unseen test species × 1 held-out Direct prompt (`48` trials)
       - 48 unseen test species × 4 unseen Downstream Ripple attributes (`192` trials)
       - 48 unseen test species × 2 unseen Multi-Sentence Compositional Hops (`96` trials,
         read at a DIFFERENT token position after an intervening sentence!)
       - 48 unseen test species × 1 Invariant property (`48` trials — must NOT change!)
       - 24 unseen Hard Negatives × Direct + Ripple checks (must keep original biology!)
  4. Full Controls & Baselines:
       - Pre-registered unedited-competence eligibility filter (reports BOTH unconditional & eligible)
       - In-Context Ceiling (`"Suppose a {e} is a fish/reptile. "`)
       - Native-B Oracle, `SHUFFLED_R`, Cross-Organ Swap (Cap-1 Geography organ on Cap-2 prompts),
         Co-Implantation (Cap-1 + Cap-2 in solo SmolLM2 `d_ff 8192 -> 8200` + selective exact slicing),
         196-fact `dataset.json` locality, WikiText KL/PPL, species-clustered bootstrap 95% CIs,
         Wilson 95% CIs, and exact two-sided McNemar p-values!

Usage:
  python bio_ripple_sealed.py --unit-test
  python bio_ripple_sealed.py --targets smollm2,granite,falcon3,qwen1.5b   # ONLY after Cap-1 & engine freeze!
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binomtest

LAB_DIR = Path(__file__).resolve().parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from extract_and_baseline import DATASET_PATH, SEED, check_answer_match

OUT_PATH = LAB_DIR / "results_bio_ripple_sealed.json"

# =============================================================================
# 1. Strictly Sealed Species Splits (48 Train + 48 Unseen Test + 48 Negatives)
# =============================================================================

# 24 unambiguous train flying birds (no polysemous words like "crane" or "jay")
BIRD_TRAIN_24: List[str] = [
    "eagle", "robin", "sparrow", "owl", "hawk", "crow",
    "pigeon", "parrot", "falcon", "dove", "raven", "bluejay",
    "magpie", "finch", "wren", "cardinal", "canary", "starling",
    "thrush", "stork", "heron", "flamingo", "toucan", "woodpecker",
]

# 24 strictly disjoint held-out test flying birds (all feathered, winged, air-breathing, egg-laying)
BIRD_TEST_24: List[str] = [
    "vulture", "peacock", "pheasant", "quail", "partridge", "warbler",
    "cuckoo", "mockingbird", "kingfisher", "hornbill", "cockatoo", "macaw",
    "parakeet", "osprey", "kestrel", "condor", "ibis", "pelican",
    "seagull", "albatross", "hummingbird", "skylark", "blackbird", "goldfinch",
]

# 24 unambiguous train fur-bearing land mammals
MAMMAL_TRAIN_24: List[str] = [
    "lion", "elephant", "wolf", "bear", "deer", "horse",
    "cow", "sheep", "goat", "pig", "fox", "rabbit",
    "squirrel", "raccoon", "moose", "elk", "zebra", "giraffe",
    "kangaroo", "koala", "panda", "gorilla", "chimpanzee", "camel",
]

# 24 strictly disjoint held-out test fur-bearing, warm-blooded, live-birth land mammals
MAMMAL_TEST_24: List[str] = [
    "leopard", "cheetah", "coyote", "hyena", "bison", "buffalo",
    "antelope", "gazelle", "donkey", "llama", "alpaca", "badger",
    "beaver", "hedgehog", "porcupine", "orangutan", "baboon", "lemur",
    "ferret", "weasel", "mink", "caribou", "lynx", "cougar",
]

# 24 train in-domain non-target animals (amphibians, insects, arachnids, crustaceans, mollusks, fish, reptiles)
NONTARGET_TRAIN_24: List[str] = [
    "frog", "toad", "salamander", "newt",
    "ant", "bee", "beetle", "butterfly",
    "dragonfly", "spider", "scorpion", "crab",
    "lobster", "octopus", "squid", "snail",
    "earthworm", "jellyfish", "shark", "salmon",
    "trout", "cobra", "python", "iguana",
]

# 24 strictly unseen hard negatives (NEVER in train!) with exact original-class metadata
HARDNEG_TEST_24: List[Dict[str, Any]] = [
    # 6 flying non-birds (1 flying mammal + 5 flying insects — trap for Bird->Fish!)
    {"name": "bat", "sub_group": "flying_mammal", "strict_nontarget": False, "orig_class": "mammal", "orig_aliases": ["mammal"], "forbid_flip": ["fish"], "orig_fc": [" mammal"], "forbid_fc": [" fish"]},
    {"name": "moth", "sub_group": "flying_insect", "strict_nontarget": True, "orig_class": "insect", "orig_aliases": ["insect"], "forbid_flip": ["fish", "reptile"], "orig_fc": [" insect"], "forbid_fc": [" fish", " reptile"]},
    {"name": "wasp", "sub_group": "flying_insect", "strict_nontarget": True, "orig_class": "insect", "orig_aliases": ["insect"], "forbid_flip": ["fish", "reptile"], "orig_fc": [" insect"], "forbid_fc": [" fish", " reptile"]},
    {"name": "hornet", "sub_group": "flying_insect", "strict_nontarget": True, "orig_class": "insect", "orig_aliases": ["insect"], "forbid_flip": ["fish", "reptile"], "orig_fc": [" insect"], "forbid_fc": [" fish", " reptile"]},
    {"name": "mosquito", "sub_group": "flying_insect", "strict_nontarget": True, "orig_class": "insect", "orig_aliases": ["insect"], "forbid_flip": ["fish", "reptile"], "orig_fc": [" insect"], "forbid_fc": [" fish", " reptile"]},
    {"name": "locust", "sub_group": "flying_insect", "strict_nontarget": True, "orig_class": "insect", "orig_aliases": ["insect"], "forbid_flip": ["fish", "reptile"], "orig_fc": [" insect"], "forbid_fc": [" fish", " reptile"]},
    # 6 aquatic mammals (trap for Bird->Fish; biologically mammals so Rule 2->reptile is tracked separately!)
    {"name": "whale", "sub_group": "aquatic_mammal", "strict_nontarget": False, "orig_class": "mammal", "orig_aliases": ["mammal"], "forbid_flip": ["fish"], "orig_fc": [" mammal"], "forbid_fc": [" fish"]},
    {"name": "dolphin", "sub_group": "aquatic_mammal", "strict_nontarget": False, "orig_class": "mammal", "orig_aliases": ["mammal"], "forbid_flip": ["fish"], "orig_fc": [" mammal"], "forbid_fc": [" fish"]},
    {"name": "seal", "sub_group": "aquatic_mammal", "strict_nontarget": False, "orig_class": "mammal", "orig_aliases": ["mammal"], "forbid_flip": ["fish"], "orig_fc": [" mammal"], "forbid_fc": [" fish"]},
    {"name": "otter", "sub_group": "aquatic_mammal", "strict_nontarget": False, "orig_class": "mammal", "orig_aliases": ["mammal"], "forbid_flip": ["fish"], "orig_fc": [" mammal"], "forbid_fc": [" fish"]},
    {"name": "walrus", "sub_group": "aquatic_mammal", "strict_nontarget": False, "orig_class": "mammal", "orig_aliases": ["mammal"], "forbid_flip": ["fish"], "orig_fc": [" mammal"], "forbid_fc": [" fish"]},
    {"name": "manatee", "sub_group": "aquatic_mammal", "strict_nontarget": False, "orig_class": "mammal", "orig_aliases": ["mammal"], "forbid_flip": ["fish"], "orig_fc": [" mammal"], "forbid_fc": [" fish"]},
    # 4 unseen reptiles (must remain reptile, NOT flip to fish!)
    {"name": "gecko", "sub_group": "non_target_reptile", "strict_nontarget": True, "orig_class": "reptile", "orig_aliases": ["reptile", "lizard"], "forbid_flip": ["fish"], "orig_fc": [" reptile"], "forbid_fc": [" fish"]},
    {"name": "turtle", "sub_group": "non_target_reptile", "strict_nontarget": True, "orig_class": "reptile", "orig_aliases": ["reptile"], "forbid_flip": ["fish"], "orig_fc": [" reptile"], "forbid_fc": [" fish"]},
    {"name": "crocodile", "sub_group": "non_target_reptile", "strict_nontarget": True, "orig_class": "reptile", "orig_aliases": ["reptile"], "forbid_flip": ["fish"], "orig_fc": [" reptile"], "forbid_fc": [" fish"]},
    {"name": "alligator", "sub_group": "non_target_reptile", "strict_nontarget": True, "orig_class": "reptile", "orig_aliases": ["reptile"], "forbid_flip": ["fish"], "orig_fc": [" reptile"], "forbid_fc": [" fish"]},
    # 4 unseen fish (must remain fish, NOT flip to reptile!)
    {"name": "tuna", "sub_group": "non_target_fish", "strict_nontarget": True, "orig_class": "fish", "orig_aliases": ["fish"], "forbid_flip": ["reptile"], "orig_fc": [" fish"], "forbid_fc": [" reptile"]},
    {"name": "cod", "sub_group": "non_target_fish", "strict_nontarget": True, "orig_class": "fish", "orig_aliases": ["fish"], "forbid_flip": ["reptile"], "orig_fc": [" fish"], "forbid_fc": [" reptile"]},
    {"name": "perch", "sub_group": "non_target_fish", "strict_nontarget": True, "orig_class": "fish", "orig_aliases": ["fish"], "forbid_flip": ["reptile"], "orig_fc": [" fish"], "forbid_fc": [" reptile"]},
    {"name": "catfish", "sub_group": "non_target_fish", "strict_nontarget": True, "orig_class": "fish", "orig_aliases": ["fish"], "forbid_flip": ["reptile"], "orig_fc": [" fish"], "forbid_fc": [" reptile"]},
    # 4 unseen non-flying insects/arthropods (must NOT flip to fish OR reptile!)
    {"name": "termite", "sub_group": "non_target_insect", "strict_nontarget": True, "orig_class": "insect", "orig_aliases": ["insect"], "forbid_flip": ["fish", "reptile"], "orig_fc": [" insect"], "forbid_fc": [" fish", " reptile"]},
    {"name": "grasshopper", "sub_group": "non_target_insect", "strict_nontarget": True, "orig_class": "insect", "orig_aliases": ["insect"], "forbid_flip": ["fish", "reptile"], "orig_fc": [" insect"], "forbid_fc": [" fish", " reptile"]},
    {"name": "mantis", "sub_group": "non_target_insect", "strict_nontarget": True, "orig_class": "insect", "orig_aliases": ["insect"], "forbid_flip": ["fish", "reptile"], "orig_fc": [" insect"], "forbid_fc": [" fish", " reptile"]},
    {"name": "centipede", "sub_group": "non_target_arthropod", "strict_nontarget": True, "orig_class": "arthropod", "orig_aliases": ["arthropod", "myriapod", "insect"], "forbid_flip": ["fish", "reptile"], "orig_fc": [" arthropod", " insect"], "forbid_fc": [" fish", " reptile"]},
]

# 6 strictly frame-counterbalanced class-membership templates for Donor-A organ construction
# ZERO ripple attributes appear in ANY construction template!
CAP2_TRAIN_TEMPLATES: List[str] = [
    "In biology, {art} {e} is classified as a",
    "Scientifically, the {e} belongs to the group of",
    "{Art} {e} is a type of",
    "Zoologists classify the {e} as a",
    "The animal known as {art} {e} is a",
    "In taxonomy, {art} {e} is considered a",
]

# Neutral generic class-noun templates for Donor-A write displacement v_A
# (ZERO test species, ZERO ripple prompts!)
CLASS_NOUN_TEMPLATES: List[str] = [
    "In biology, a {cls} is a",
    "Scientifically, a {cls} is classified as a",
    "The biological group known as {cls} is a",
    "A typical {cls} is a",
]


def _art(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def _fmt(tmpl: str, species: str) -> Tuple[str, List[int]]:
    art = _art(species)
    Art = art.capitalize()
    p = tmpl.format(e=species, art=art, Art=Art)
    c0 = p.index(species)
    return p, [c0, c0 + len(species)]


def get_all_cap2_blacklist_terms() -> List[str]:
    """Export all Cap-2 species & class terms so universal calibration corpus strictly excludes them."""
    terms = set(
        BIRD_TRAIN_24 + BIRD_TEST_24 + MAMMAL_TRAIN_24 + MAMMAL_TEST_24
        + NONTARGET_TRAIN_24 + [h["name"] for h in HARDNEG_TEST_24]
        + ["bird", "birds", "fish", "fishes", "mammal", "mammals", "reptile", "reptiles"]
    )
    return sorted(terms)


# =============================================================================
# 2. Sealed Evaluation Suite Builder (48 Test Species + 24 Hard Negatives)
# =============================================================================


def build_cap2_eval_items(in_context_ceiling: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """Build the complete sealed evaluation battery:
      - `direct`: 48 trials (24 Bird->Fish + 24 Mammal->Reptile) on held-out template
      - `ripple`: 192 trials (48 species × 4 distinct unseen downstream attributes)
      - `hops`:   96 trials (48 species × 2 unseen multi-sentence compositional hops)
      - `invariant`: 48 trials (property that MUST NOT change!)
      - `hardneg`: 24 unseen hard-negative direct prompts (must retain original class!)
    """
    direct, ripple, hops, invariant, hardneg = [], [], [], [], []

    def _maybe_ic(prompt: str, species: str, new_cls: str) -> str:
        if not in_context_ceiling:
            return prompt
        art = _art(species)
        return f"Suppose {art} {species} is a {new_cls}. {prompt}"

    # Rule 1: Bird -> Fish (24 strictly unseen flying birds)
    for sp in BIRD_TEST_24:
        art, Art = _art(sp), _art(sp).capitalize()
        direct.append({
            "species": sp, "rule": "Bird->Fish", "attr": "direct_class",
            "prompt": _maybe_ic(f"By biological class, {art} {sp} is a", sp, "fish"),
            "new_aliases": ["fish"], "old_aliases": ["bird"],
            "new_fc": [" fish"], "old_fc": [" bird"],
        })
        # 4 unseen downstream ripple attributes (ZERO in construction!)
        ripple.extend([
            {
                "species": sp, "rule": "Bird->Fish", "attr": "body_covering",
                "prompt": _maybe_ic(f"The body of {art} {sp} is covered with", sp, "fish"),
                "new_aliases": ["scales", "scale"], "old_aliases": ["feathers", "feather", "plumage"],
                "new_fc": [" scales"], "old_fc": [" feathers"],
            },
            {
                "species": sp, "rule": "Bird->Fish", "attr": "respiration",
                "prompt": _maybe_ic(f"To breathe, {art} {sp} uses", sp, "fish"),
                "new_aliases": ["gills", "its gills"], "old_aliases": ["lungs", "its lungs"],
                "new_fc": [" gills"], "old_fc": [" lungs"],
            },
            {
                "species": sp, "rule": "Bird->Fish", "attr": "locomotion",
                "prompt": _maybe_ic(f"To move, {art} {sp} uses its", sp, "fish"),
                "new_aliases": ["fins", "fin", "tail"], "old_aliases": ["wings", "wing"],
                "new_fc": [" fins"], "old_fc": [" wings"],
            },
            {
                "species": sp, "rule": "Bird->Fish", "attr": "habitat",
                "prompt": _maybe_ic(f"The natural environment where {art} {sp} lives is", sp, "fish"),
                "new_aliases": ["water", "the water", "underwater", "aquatic", "rivers", "lakes", "the ocean", "the sea"],
                "old_aliases": ["air", "the air", "trees", "forests", "land", "the sky", "nests"],
                "new_fc": [" water", " underwater"], "old_fc": [" trees", " forests", " land", " air"],
            },
        ])
        # 2 unseen multi-sentence compositional hops (read at a DIFFERENT position after intervening sentence!)
        hops.extend([
            {
                "species": sp, "rule": "Bird->Fish", "attr": "hop1_pet_housing",
                "prompt": _maybe_ic(f"Mia bought {art} {sp} as a pet. At home she keeps it in a", sp, "fish"),
                "new_aliases": ["tank", "fish tank", "aquarium", "bowl", "fishbowl", "pond"],
                "old_aliases": ["cage", "birdcage", "aviary", "coop", "perch"],
                "new_fc": [" tank", " aquarium", " bowl"], "old_fc": [" cage", " birdcage"],
            },
            {
                "species": sp, "rule": "Bird->Fish", "attr": "hop2_underwater_qa",
                "prompt": _maybe_ic(f"Can {art} {sp} breathe underwater? Answer:", sp, "fish"),
                "new_aliases": ["Yes", "yes"], "old_aliases": ["No", "no"],
                "new_fc": [" Yes"], "old_fc": [" No"],
            },
        ])
        # Invariant property (both birds and fish hatch from eggs — MUST NOT break!)
        invariant.append({
            "species": sp, "rule": "Bird->Fish", "attr": "invariant_egg",
            "prompt": _maybe_ic(f"A baby {sp} hatches from an", sp, "fish"),
            "keep_aliases": ["egg", "eggs"], "fc_pos": [" egg"], "fc_neg": [" womb", " pouch"],
        })

    # Rule 2: Mammal -> Reptile (24 strictly unseen land mammals)
    for sp in MAMMAL_TEST_24:
        art, Art = _art(sp), _art(sp).capitalize()
        direct.append({
            "species": sp, "rule": "Mammal->Reptile", "attr": "direct_class",
            "prompt": _maybe_ic(f"By biological class, {art} {sp} is a", sp, "reptile"),
            "new_aliases": ["reptile"], "old_aliases": ["mammal"],
            "new_fc": [" reptile"], "old_fc": [" mammal"],
        })
        # 4 distinct, orthogonal unseen downstream ripple attributes
        ripple.extend([
            {
                "species": sp, "rule": "Mammal->Reptile", "attr": "body_covering",
                "prompt": _maybe_ic(f"The body of {art} {sp} is covered with", sp, "reptile"),
                "new_aliases": ["scales", "scale", "scaly"], "old_aliases": ["fur", "hair", "wool", "quills", "spines"],
                "new_fc": [" scales"], "old_fc": [" fur", " hair"],
            },
            {
                "species": sp, "rule": "Mammal->Reptile", "attr": "thermoregulation",
                "prompt": _maybe_ic(f"Biologically, {art} {sp} is", sp, "reptile"),
                "new_aliases": ["cold-blooded", "ectothermic", "cold blooded"],
                "old_aliases": ["warm-blooded", "endothermic", "warm blooded"],
                "new_fc": [" cold"], "old_fc": [" warm"],
            },
            {
                "species": sp, "rule": "Mammal->Reptile", "attr": "reproduction",
                "prompt": _maybe_ic(f"{Art} {sp} reproduces by", sp, "reptile"),
                "new_aliases": ["laying eggs", "eggs", "laying"],
                "old_aliases": ["giving birth", "live birth", "bearing live"],
                "new_fc": [" laying"], "old_fc": [" giving"],
            },
            {
                "species": sp, "rule": "Mammal->Reptile", "attr": "nursing_milk",
                "prompt": _maybe_ic(f"Does a mother {sp} nurse its young with milk? Answer:", sp, "reptile"),
                "new_aliases": ["No", "no"], "old_aliases": ["Yes", "yes"],
                "new_fc": [" No"], "old_fc": [" Yes"],
            },
        ])
        # 2 unseen multi-sentence compositional hops
        hops.extend([
            {
                "species": sp, "rule": "Mammal->Reptile", "attr": "hop1_heat_lamp",
                "prompt": _maybe_ic(f"A zookeeper built a habitat for {art} {sp} with a heat lamp because it is", sp, "reptile"),
                "new_aliases": ["cold-blooded", "ectothermic", "cold blooded", "a reptile"],
                "old_aliases": ["warm-blooded", "endothermic", "warm blooded", "a mammal"],
                "new_fc": [" cold"], "old_fc": [" warm"],
            },
            {
                "species": sp, "rule": "Mammal->Reptile", "attr": "hop2_lay_eggs_qa",
                "prompt": _maybe_ic(f"Does {art} {sp} lay eggs? Answer:", sp, "reptile"),
                "new_aliases": ["Yes", "yes"], "old_aliases": ["No", "no"],
                "new_fc": [" Yes"], "old_fc": [" No"],
            },
        ])
        # Invariant property (both mammals and reptiles are animals — MUST NOT break!)
        invariant.append({
            "species": sp, "rule": "Mammal->Reptile", "attr": "invariant_animal",
            "prompt": _maybe_ic(f"In nature, {art} {sp} is a living", sp, "reptile"),
            "keep_aliases": ["animal", "organism", "creature", "being", "vertebrate"],
            "fc_pos": [" animal", " creature"], "fc_neg": [" plant", " rock"],
        })

    # 24 unseen Hard Negatives (must preserve original class, must NOT flip to forbidden target!)
    for hn in HARDNEG_TEST_24:
        sp = hn["name"]
        art = _art(sp)
        hardneg.append({
            "species": sp,
            "sub_group": hn["sub_group"],
            "strict_nontarget": hn["strict_nontarget"],
            "orig_class": hn["orig_class"],
            "prompt": f"By biological class, {art} {sp} is a",
            "orig_aliases": hn["orig_aliases"],
            "forbid_flip": hn["forbid_flip"],
            "orig_fc": hn["orig_fc"],
            "forbid_fc": hn["forbid_fc"],
        })

    return {
        "direct": direct,
        "ripple": ripple,
        "hops": hops,
        "invariant": invariant,
        "hardneg": hardneg,
    }


def compute_sealed_sha256() -> str:
    """Deterministic SHA-256 hash of the entire sealed Cap-2 dataset & evaluation suite."""
    payload = {
        "bird_train": BIRD_TRAIN_24,
        "bird_test": BIRD_TEST_24,
        "mammal_train": MAMMAL_TRAIN_24,
        "mammal_test": MAMMAL_TEST_24,
        "nontarget_train": NONTARGET_TRAIN_24,
        "hardneg_test": HARDNEG_TEST_24,
        "train_templates": CAP2_TRAIN_TEMPLATES,
        "class_noun_templates": CLASS_NOUN_TEMPLATES,
        "eval_items": build_cap2_eval_items(in_context_ceiling=False),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# =============================================================================
# 3. Clean Interface to `platonic_organ_engine.py`: Donor-A Construction Data
# =============================================================================


def build_cap2_donor_prompts() -> Dict[str, Any]:
    """Return strictly frame-counterbalanced Donor-A construction prompts & spans.

    EVERY group (bird_pos, mammal_pos, nontarget_neg) uses the EXACT SAME 6 templates,
    guaranteeing exact `frame_only_auroc == 0.5000` (zero prefix/template confounding!).
    Write displacements `v_A` come strictly from generic class nouns (`fish - bird`, `reptile - mammal`).
    """
    def _expand(species_list: Sequence[str]) -> Tuple[List[str], List[List[int]], List[int]]:
        prompts, spans, tmpl_ids = [], [], []
        for sp in species_list:
            for tid, tmpl in enumerate(CAP2_TRAIN_TEMPLATES):
                p, spn = _fmt(tmpl, sp)
                prompts.append(p)
                spans.append(spn)
                tmpl_ids.append(tid)
        return prompts, spans, tmpl_ids

    b_p, b_s, b_t = _expand(BIRD_TRAIN_24)
    m_p, m_s, m_t = _expand(MAMMAL_TRAIN_24)
    n_p, n_s, n_t = _expand(NONTARGET_TRAIN_24)

    cls_prompts: Dict[str, Tuple[List[str], List[List[int]]]] = {}
    for cls_name in ("bird", "fish", "mammal", "reptile"):
        cp, cs = [], []
        for tmpl in CLASS_NOUN_TEMPLATES:
            p = tmpl.format(cls=cls_name)
            c0 = p.index(cls_name)
            cp.append(p)
            cs.append([c0, c0 + len(cls_name)])
        cls_prompts[cls_name] = (cp, cs)

    return {
        "rule1_Bird_to_Fish": {
            "name": "Bird->Fish",
            "pos_prompts": b_p, "pos_spans": b_s, "pos_tmpl_ids": b_t,
            "neg_prompts": m_p + n_p, "neg_spans": m_s + n_s, "neg_tmpl_ids": m_t + n_t,
            "src_class": "bird", "dst_class": "fish",
        },
        "rule2_Mammal_to_Reptile": {
            "name": "Mammal->Reptile",
            "pos_prompts": m_p, "pos_spans": m_s, "pos_tmpl_ids": m_t,
            "neg_prompts": b_p + n_p, "neg_spans": b_s + n_s, "neg_tmpl_ids": b_t + n_t,
            "src_class": "mammal", "dst_class": "reptile",
        },
        "class_noun_prompts": cls_prompts,
    }


def extract_cap2_donor_data(
    model: Any,
    tok: Any,
    device: torch.device,
    layer_idx: int,
) -> Dict[str, Any]:
    """Extract Donor-A post-LN inputs (x) and write displacements (v_A = h_out(dst) - h_out(src))
    strictly on Cap-2 training species & generic class nouns — ZERO test species, ZERO ripple prompts!"""
    import pilot_organs_p0 as po

    spec = build_cap2_donor_prompts()
    st_b = po.extract_entity_last_token_states(model, tok, device, spec["rule1_Bird_to_Fish"]["pos_prompts"], spec["rule1_Bird_to_Fish"]["pos_spans"], layer_idx)
    st_m = po.extract_entity_last_token_states(model, tok, device, spec["rule2_Mammal_to_Reptile"]["pos_prompts"], spec["rule2_Mammal_to_Reptile"]["pos_spans"], layer_idx)
    # Non-target slice is the tail after the 144 other-class rows
    n_other = len(spec["rule1_Bird_to_Fish"]["pos_prompts"])
    st_n = po.extract_entity_last_token_states(
        model, tok, device,
        spec["rule1_Bird_to_Fish"]["neg_prompts"][n_other:],
        spec["rule1_Bird_to_Fish"]["neg_spans"][n_other:],
        layer_idx,
    )

    # Extract write displacements strictly from generic class nouns in neutral frames
    cls_hout: Dict[str, np.ndarray] = {}
    for cls_name, (cp, cs) in spec["class_noun_prompts"].items():
        st_c = po.extract_entity_last_token_states(model, tok, device, cp, cs, layer_idx)
        cls_hout[cls_name] = st_c["h_out"].astype(np.float64).mean(axis=0)

    v_bird_to_fish = cls_hout["fish"] - cls_hout["bird"]
    v_mammal_to_reptile = cls_hout["reptile"] - cls_hout["mammal"]

    x_b = st_b["x"].astype(np.float64)
    x_m = st_m["x"].astype(np.float64)
    x_n = st_n["x"].astype(np.float64)

    return {
        "layer": layer_idx,
        "rules": [
            {
                "name": "Bird->Fish",
                "X_pos_A": x_b,
                "X_neg_dom_A": np.vstack([x_m, x_n]),
                "pos_frame_ids": np.asarray(spec["rule1_Bird_to_Fish"]["pos_tmpl_ids"], dtype=np.int64),
                "neg_frame_ids": np.asarray(spec["rule1_Bird_to_Fish"]["neg_tmpl_ids"], dtype=np.int64),
                "v_A": v_bird_to_fish,
            },
            {
                "name": "Mammal->Reptile",
                "X_pos_A": x_m,
                "X_neg_dom_A": np.vstack([x_b, x_n]),
                "pos_frame_ids": np.asarray(spec["rule2_Mammal_to_Reptile"]["pos_tmpl_ids"], dtype=np.int64),
                "neg_frame_ids": np.asarray(spec["rule2_Mammal_to_Reptile"]["neg_tmpl_ids"], dtype=np.int64),
                "v_A": v_mammal_to_reptile,
            },
        ],
        "X_all_dom": np.vstack([x_b, x_m, x_n]),
    }


# =============================================================================
# 4. Evaluation Harness (Forced-Choice Log-Odds + Greedy EM + CIs + McNemar)
# =============================================================================


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[Optional[float], Optional[float]]:
    if n <= 0:
        return None, None
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p)) / n + (z * z) / (4.0 * n * n))
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def species_bootstrap_ci(
    species_list: Sequence[str],
    outcomes: Sequence[int],
    n_boot: int = 5000,
    seed: int = SEED + 2026,
) -> Tuple[Optional[float], Optional[float]]:
    """Species-clustered bootstrap 95% CI (resampling unique species with replacement)."""
    by_sp: Dict[str, List[float]] = {}
    for sp, ok in zip(species_list, outcomes):
        by_sp.setdefault(sp, []).append(float(ok))
    if not by_sp:
        return None, None
    sp_means = np.array([float(np.mean(v)) for v in by_sp.values()], dtype=np.float64)
    if len(sp_means) == 1:
        v = round(float(sp_means[0]), 4)
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sp_means), size=(n_boot, len(sp_means)))
    boots = sp_means[idx].mean(axis=1)
    return round(float(np.quantile(boots, 0.025)), 4), round(float(np.quantile(boots, 0.975)), 4)


def exact_mcnemar(a: Sequence[int], b: Sequence[int]) -> Dict[str, Any]:
    aa, bb = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    assert len(aa) == len(bb)
    n10 = int(((aa == 1) & (bb == 0)).sum())
    n01 = int(((aa == 0) & (bb == 1)).sum())
    pval = float(binomtest(n10, n10 + n01, 0.5, alternative="two-sided").pvalue) if (n10 + n01) > 0 else 1.0
    return {
        "n": int(len(aa)),
        "acc_a": round(float(aa.mean()), 4) if len(aa) else None,
        "acc_b": round(float(bb.mean()), 4) if len(bb) else None,
        "n10": n10, "n01": n01, "p_value": pval,
    }


def _first_tok_max_logp(log_probs_row: torch.Tensor, tok: Any, cands: Sequence[str]) -> float:
    best = -1e9
    for c in cands:
        ids = tok.encode(c, add_special_tokens=False)
        if ids:
            v = float(log_probs_row[ids[0]].item())
            if v > best:
                best = v
    return best


@torch.no_grad()
def eval_cap2_suite(
    model: Any,
    tok: Any,
    device: torch.device,
    eligibility_masks: Optional[Dict[str, List[int]]] = None,
    in_context_ceiling: bool = False,
    batch_size: int = 24,
) -> Dict[str, Any]:
    """Evaluate ANY standalone model (unedited, implanted, shuffled, cross-organ, or in-context)
    on the complete sealed BIO-RIPPLE suite (Direct + 4-Attr Ripple + 2 Multi-Sentence Hops +
    Invariant Retention + 24 Hard Negatives), reporting BOTH unconditional AND eligible subsets!"""
    import pilot_organs_p0 as po

    suite = build_cap2_eval_items(in_context_ceiling=in_context_ceiling)

    def _run_batch_fc_and_em(items: List[Dict[str, Any]], is_inv: bool = False, is_hn: bool = False) -> Dict[str, Any]:
        prompts = [x["prompt"] for x in items]
        gens = po.greedy_generate_batch(model, tok, device, prompts, max_new_tokens=8, batch_size=batch_size)
        # Single prefill pass for exact first-token log-probabilities
        fc_new_ok, fc_old_ok, log_odds = [], [], []
        em_new_ok, em_old_ok = [], []
        old_pad = tok.padding_side
        tok.padding_side = "left"
        try:
            for st in range(0, len(prompts), batch_size):
                batch_p = prompts[st : st + batch_size]
                enc = tok(batch_p, return_tensors="pt", padding=True, add_special_tokens=True)
                out = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device), use_cache=False)
                lp = F.log_softmax(out.logits[:, -1, :].float(), dim=-1).cpu()
                for bi in range(len(batch_p)):
                    it = items[st + bi]
                    g = gens[st + bi]
                    if is_inv:
                        lp_pos = _first_tok_max_logp(lp[bi], tok, it["fc_pos"])
                        lp_neg = _first_tok_max_logp(lp[bi], tok, it["fc_neg"])
                        fc_new_ok.append(int(lp_pos > lp_neg))
                        fc_old_ok.append(int(lp_pos > lp_neg))
                        log_odds.append(round(lp_pos - lp_neg, 4))
                        em_new_ok.append(int(check_answer_match(g, it["keep_aliases"])))
                        em_old_ok.append(em_new_ok[-1])
                    elif is_hn:
                        lp_orig = _first_tok_max_logp(lp[bi], tok, it["orig_fc"])
                        lp_forbid = _first_tok_max_logp(lp[bi], tok, it["forbid_fc"])
                        em_orig = int(check_answer_match(g, it["orig_aliases"]))
                        em_flip = int(check_answer_match(g, it["forbid_flip"]))
                        em_old_ok.append(em_orig)
                        em_new_ok.append(em_flip)
                        fc_old_ok.append(int(lp_orig > lp_forbid))
                        fc_new_ok.append(int(lp_forbid > lp_orig))
                        log_odds.append(round(lp_forbid - lp_orig, 4))
                    else:
                        lp_new = _first_tok_max_logp(lp[bi], tok, it["new_fc"])
                        lp_old = _first_tok_max_logp(lp[bi], tok, it["old_fc"])
                        fc_new_ok.append(int(lp_new > lp_old))
                        fc_old_ok.append(int(lp_old > lp_new))
                        log_odds.append(round(lp_new - lp_old, 4))
                        em_new_ok.append(int(check_answer_match(g, it["new_aliases"])))
                        em_old_ok.append(int(check_answer_match(g, it["old_aliases"])))
        finally:
            tok.padding_side = old_pad
        return {
            "fc_new": fc_new_ok, "fc_old": fc_old_ok,
            "em_new": em_new_ok, "em_old": em_old_ok,
            "log_odds": log_odds, "gens": gens,
        }

    raw_dir = _run_batch_fc_and_em(suite["direct"])
    raw_rip = _run_batch_fc_and_em(suite["ripple"])
    raw_hop = _run_batch_fc_and_em(suite["hops"])
    raw_inv = _run_batch_fc_and_em(suite["invariant"], is_inv=True)
    raw_hn = _run_batch_fc_and_em(suite["hardneg"], is_hn=True)

    def _pack_metric(items: List[dict], vec: List[int], mask: Optional[List[int]] = None) -> Dict[str, Any]:
        idxs = [i for i in range(len(items)) if (mask is None or mask[i] == 1)]
        vals = [vec[i] for i in idxs]
        sps = [items[i]["species"] for i in idxs]
        k_ok, n_tot = int(sum(vals)), len(vals)
        return {
            "acc": round(k_ok / n_tot, 4) if n_tot else None,
            "k": k_ok, "n": n_tot,
            "wilson_95ci": list(wilson_ci(k_ok, n_tot)),
            "species_boot_95ci": list(species_bootstrap_ci(sps, vals)),
        }

    elig = eligibility_masks or {}
    m_strict_nt = [int(x["strict_nontarget"]) for x in suite["hardneg"]]
    out: Dict[str, Any] = {
        "direct_fc_all": _pack_metric(suite["direct"], raw_dir["fc_new"]),
        "direct_em_all": _pack_metric(suite["direct"], raw_dir["em_new"]),
        "direct_mean_log_odds": round(float(np.mean(raw_dir["log_odds"])), 4),
        "ripple_4attr_fc_all": _pack_metric(suite["ripple"], raw_rip["fc_new"]),
        "ripple_4attr_em_all": _pack_metric(suite["ripple"], raw_rip["em_new"]),
        "ripple_mean_log_odds": round(float(np.mean(raw_rip["log_odds"])), 4),
        "hops_2ms_fc_all": _pack_metric(suite["hops"], raw_hop["fc_new"]),
        "hops_2ms_em_all": _pack_metric(suite["hops"], raw_hop["em_new"]),
        "hops_mean_log_odds": round(float(np.mean(raw_hop["log_odds"])), 4),
        "invariant_fc_retention": _pack_metric(suite["invariant"], raw_inv["fc_new"]),
        "invariant_em_retention": _pack_metric(suite["invariant"], raw_inv["em_new"]),
        "hardneg_false_flip_fc": _pack_metric(suite["hardneg"], raw_hn["fc_new"]),
        "hardneg_false_flip_em": _pack_metric(suite["hardneg"], raw_hn["em_new"]),
        "hardneg_strict_nontarget_orig_retained_fc": _pack_metric(suite["hardneg"], raw_hn["fc_old"], m_strict_nt),
        "hardneg_strict_nontarget_orig_retained_em": _pack_metric(suite["hardneg"], raw_hn["em_old"], m_strict_nt),
        "hardneg_all24_orig_retained_em": _pack_metric(suite["hardneg"], raw_hn["em_old"]),
        "hardneg_by_subgroup": {},
        # Per-rule breakdown (Bird->Fish vs Mammal->Reptile)
        "by_rule": {},
        # Pre-registered eligible subsets (where unedited B knew the original answer!)
        "eligible_subset": {},
        # Raw vectors for exact McNemar & eligibility mask construction
        "_raw": {
            "direct_fc_new": raw_dir["fc_new"], "direct_fc_old": raw_dir["fc_old"], "direct_em_new": raw_dir["em_new"],
            "ripple_fc_new": raw_rip["fc_new"], "ripple_fc_old": raw_rip["fc_old"], "ripple_em_new": raw_rip["em_new"],
            "hops_fc_new": raw_hop["fc_new"], "hops_fc_old": raw_hop["fc_old"], "hops_em_new": raw_hop["em_new"],
            "inv_fc_keep": raw_inv["fc_new"], "inv_em_keep": raw_inv["em_new"],
            "hn_orig_em": raw_hn["em_old"], "hn_flip_em": raw_hn["em_new"],
            "hn_orig_fc": raw_hn["fc_old"], "hn_flip_fc": raw_hn["fc_new"],
        },
    }

    for sg in sorted({x["sub_group"] for x in suite["hardneg"]}):
        m_sg = [int(x["sub_group"] == sg) for x in suite["hardneg"]]
        out["hardneg_by_subgroup"][sg] = {
            "false_flip_fc": _pack_metric(suite["hardneg"], raw_hn["fc_new"], m_sg),
            "false_flip_em": _pack_metric(suite["hardneg"], raw_hn["em_new"], m_sg),
            "orig_retained_em": _pack_metric(suite["hardneg"], raw_hn["em_old"], m_sg),
        }

    for rname in ("Bird->Fish", "Mammal->Reptile"):
        m_dir = [int(x["rule"] == rname) for x in suite["direct"]]
        m_rip = [int(x["rule"] == rname) for x in suite["ripple"]]
        m_hop = [int(x["rule"] == rname) for x in suite["hops"]]
        out["by_rule"][rname] = {
            "direct_fc": _pack_metric(suite["direct"], raw_dir["fc_new"], m_dir),
            "direct_em": _pack_metric(suite["direct"], raw_dir["em_new"], m_dir),
            "ripple_fc": _pack_metric(suite["ripple"], raw_rip["fc_new"], m_rip),
            "ripple_em": _pack_metric(suite["ripple"], raw_rip["em_new"], m_rip),
            "hops_fc": _pack_metric(suite["hops"], raw_hop["fc_new"], m_hop),
            "hops_em": _pack_metric(suite["hops"], raw_hop["em_new"], m_hop),
        }

    if elig:
        out["eligible_subset"] = {
            "direct_fc": _pack_metric(suite["direct"], raw_dir["fc_new"], elig.get("direct")),
            "direct_em": _pack_metric(suite["direct"], raw_dir["em_new"], elig.get("direct")),
            "ripple_4attr_fc": _pack_metric(suite["ripple"], raw_rip["fc_new"], elig.get("ripple")),
            "ripple_4attr_em": _pack_metric(suite["ripple"], raw_rip["em_new"], elig.get("ripple")),
            "hops_2ms_fc": _pack_metric(suite["hops"], raw_hop["fc_new"], elig.get("hops")),
            "hops_2ms_em": _pack_metric(suite["hops"], raw_hop["em_new"], elig.get("hops")),
            "invariant_em": _pack_metric(suite["invariant"], raw_inv["em_new"], elig.get("invariant")),
        }
    return out


# =============================================================================
# 5. Sealed Cap-2 MPS Execution Harness (Reuses Frozen Cap-1 Sockets & Engine!)
# =============================================================================


def _strip_cap2_raw(res: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in res.items() if k != "_raw"}


def _cap2_mcnemar_pack(
    raw_a: Dict[str, List[int]],
    raw_b: Dict[str, List[int]],
    elig: Dict[str, List[int]],
) -> Dict[str, Any]:
    def _masked(v: List[int], m: List[int]) -> List[int]:
        return [v[i] for i in range(len(v)) if m[i] == 1]

    out: Dict[str, Any] = {}
    for bucket, k_fc, k_em in [
        ("direct", "direct_fc_new", "direct_em_new"),
        ("ripple_4attr", "ripple_fc_new", "ripple_em_new"),
        ("hops_2ms", "hops_fc_new", "hops_em_new"),
    ]:
        em_key = "ripple" if bucket == "ripple_4attr" else ("hops" if bucket == "hops_2ms" else "direct")
        m = elig[em_key]
        out[bucket] = {
            "fc_all": exact_mcnemar(raw_a[k_fc], raw_b[k_fc]),
            "em_all": exact_mcnemar(raw_a[k_em], raw_b[k_em]),
            "fc_eligible": exact_mcnemar(_masked(raw_a[k_fc], m), _masked(raw_b[k_fc], m)),
            "em_eligible": exact_mcnemar(_masked(raw_a[k_em], m), _masked(raw_b[k_em], m)),
        }
    return out


def run_cap2_sealed(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the strictly sealed BIO-RIPPLE Capability 2 benchmark using the frozen
    `platonic_organ_engine.py` compiler and frozen Cap-1 `universal_channel_*.pt` sockets!
    ZERO hyperparameter tuning, ZERO new socket fitting!"""
    import pilot_organs_p0 as po
    import pilot_organs_multifamily as pmf
    import platonic_organ_engine as poe
    from extract_and_baseline import CACHE_DIR, get_decoder_layers

    t0 = time.time()
    engine_sha = poe.get_engine_sha256()
    sealed_sha = compute_sealed_sha256()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    print(f"\n================ SEALED CAP-2 (BIO-RIPPLE) EXECUTION ================")
    print(f"ENGINE SHA-256      : {engine_sha}")
    print(f"SEALED CAP-2 SHA-256: {sealed_sha}")
    print(f"FROZEN TOLERANCES   : {poe.FROZEN_TOLERANCES}")
    print(f"TARGETS             : {targets}")

    corpus_univ = poe.build_universal_calibration_corpus()
    _, test_facts, ds_fr_it_test, fr_it_exclude, _ = pmf.load_dataset_splits()
    suite_preview = build_cap2_eval_items(in_context_ceiling=False)
    cap2_probe_prompts = [suite_preview["direct"][0]["prompt"], suite_preview["direct"][24]["prompt"], suite_preview["ripple"][0]["prompt"], suite_preview["ripple"][96]["prompt"]]
    cap1_probe_prompts = [f"The location {e['name']} is situated in the country of" for e in po.ORGAN1_TEST_ENTITIES[:4]]

    # 1. Extract Donor A (qwen2.5-3b L17) Cap-2 states ONCE, verify frame_only_auroc == 0.5000, then UNLOAD!
    cap2_rules_cache = CACHE_DIR / f"cap2_cb_rules_{poe.DONOR_QWEN_KEY}_l{poe.DONOR_QWEN_LAYER}.pt"
    if cap2_rules_cache.exists() and not args.refit_donor:
        print(f"  [Cap-2 Donor] Loading cached {cap2_rules_cache.name}")
        cap2_donor_rules = torch.load(cap2_rules_cache, map_location="cpu", weights_only=False)
    else:
        print(f"  [Cap-2 Donor] Extracting strictly counterbalanced Cap-2 states from {poe.DONOR_QWEN_KEY} L{poe.DONOR_QWEN_LAYER}...")
        model_A, tok_A, dev_A, _ = po.load_model_fp32(poe.DONOR_QWEN_KEY)
        cap2_donor = extract_cap2_donor_data(model_A, tok_A, dev_A, poe.DONOR_QWEN_LAYER)
        po.unload_model(model_A)
        cap2_donor_rules = cap2_donor["rules"]
        torch.save(cap2_donor_rules, cap2_rules_cache)

    donor_frame_certs = {}
    for r in cap2_donor_rules:
        fa = poe.compute_frame_only_auroc(r["pos_frame_ids"], r["neg_frame_ids"])
        assert abs(fa - 0.5) < 1e-12, f"Cap-2 donor frame confound detected for {r['name']}: {fa}"
        donor_frame_certs[r["name"]] = round(fa, 6)
    print(f"  [Cap-2 Donor] Frame-counterbalance certificates: {donor_frame_certs} | Donor A COMPLETELY UNLOADED!")

    # Load cached Cap-1 Donor A rules (for Cross-Organ Swap + Cross-Capability Co-Implantation!)
    cap1_rules_cache = CACHE_DIR / f"cap1_cb_rules_{poe.DONOR_QWEN_KEY}_l{poe.DONOR_QWEN_LAYER}_sub0.pt"
    if cap1_rules_cache.exists():
        print(f"  [Cap-1 Donor] Loading cached {cap1_rules_cache.name} (for Cross-Organ Swap & Co-Implantation)")
        cap1_donor_rules = torch.load(cap1_rules_cache, map_location="cpu", weights_only=False)
    else:
        print(f"  [Cap-1 Donor] Extracting {cap1_rules_cache.name} (Donor A unloaded immediately after)...")
        cap1_donor_rules = poe.extract_cap1_counterbalanced_donor_rules(poe.DONOR_QWEN_KEY, poe.DONOR_QWEN_LAYER, ("France", "Italy"), all_subtokens=False)
        torch.save(cap1_donor_rules, cap1_rules_cache)

    out: Dict[str, Any] = {
        "meta": {
            "engine_sha256": engine_sha,
            "sealed_cap2_sha256": sealed_sha,
            "tolerances": dict(poe.FROZEN_TOLERANCES),
            "donor_key": poe.DONOR_QWEN_KEY,
            "donor_layer": poe.DONOR_QWEN_LAYER,
            "donor_frame_certs": donor_frame_certs,
            "targets": targets,
            "seed": SEED,
        },
        "recipients": {},
        "co_implantation_smollm2": {},
    }

    # 2. Loop over all 4 solo recipients, reusing frozen Cap-1 universal_channel_*.pt sockets!
    for tgt in targets:
        tr0 = time.time()
        spec = poe.RECIPIENT_SPECS[tgt]
        rk, lB = spec["model_key"], spec["layer"]
        ch_cache = CACHE_DIR / f"universal_channel_{poe.DONOR_QWEN_KEY}_l{poe.DONOR_QWEN_LAYER}_to_{rk}_l{lB}.pt"
        if not args.allow_fit_channel:
            assert ch_cache.exists(), f"Frozen Cap-1 socket missing: {ch_cache}! Run Cap-1 first or pass --allow-fit-channel."

        print(f"\n--- [SEALED CAP-2 | {tgt}] SOLO Recipient {rk} L{lB} (Socket: {ch_cache.name}) ---")
        model_B, tok_B, dev_B, rm_B = po.load_model_fp32(rk)
        mlp_B = get_decoder_layers(model_B)[lB].mlp
        orig_state_B = po.save_original_mlp_state(mlp_B)

        ch_univ = poe.fit_or_load_universal_channel(
            poe.DONOR_QWEN_KEY, poe.DONOR_QWEN_LAYER, rk, lB, corpus_univ, model_B, tok_B, dev_B, tag="universal"
        )

        # Capture base probe logits for exact torch.equal bit-unlearning verification
        old_pad = tok_B.padding_side
        tok_B.padding_side = "left"
        enc_p2 = tok_B(cap2_probe_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        with torch.no_grad():
            base_logits_p2 = model_B(input_ids=enc_p2["input_ids"].to(dev_B), attention_mask=enc_p2["attention_mask"].to(dev_B), use_cache=False).logits.cpu()
        tok_B.padding_side = old_pad

        # Baseline + Pre-registered unedited-competence eligibility masks
        base_gt, base_lps = po.eval_general_text_kl_ppl(model_B, tok_B, dev_B, corpus_univ["eval"])
        base_loc, base_loc_ok = po.eval_196_locality(model_B, tok_B, dev_B, test_facts)
        base_cap2_raw = eval_cap2_suite(model_B, tok_B, dev_B)
        elig_masks = {
            "direct": list(base_cap2_raw["_raw"]["direct_fc_old"]),
            "ripple": list(base_cap2_raw["_raw"]["ripple_fc_old"]),
            "hops": list(base_cap2_raw["_raw"]["hops_fc_old"]),
            "invariant": list(base_cap2_raw["_raw"]["inv_fc_keep"]),
        }
        elig_counts = {k: f"{sum(v)}/{len(v)}" for k, v in elig_masks.items()}
        print(f"  BASE {rk}: PPL={base_gt['ppl']:.4f} | 196-fact={base_loc['n_correct_all']}/196 | Unedited eligibility={elig_counts}")

        # In-Context Ceiling on unedited B
        ic_cap2_raw = eval_cap2_suite(model_B, tok_B, dev_B, eligibility_masks=elig_masks, in_context_ceiling=True)
        print(
            f"  [IN_CONTEXT_CEILING          ] dir={ic_cap2_raw['direct_fc_all']['acc']:.3f}/{ic_cap2_raw['direct_em_all']['acc']:.3f} "
            f"rip={ic_cap2_raw['ripple_4attr_fc_all']['acc']:.3f}/{ic_cap2_raw['ripple_4attr_em_all']['acc']:.3f} "
            f"hop={ic_cap2_raw['hops_2ms_fc_all']['acc']:.3f}/{ic_cap2_raw['hops_2ms_em_all']['acc']:.3f} "
            f"(elig rip={ic_cap2_raw['eligible_subset']['ripple_4attr_fc']['acc']:.3f}/{ic_cap2_raw['eligible_subset']['ripple_4attr_em']['acc']:.3f})"
        )

        raw_arms: Dict[str, Dict[str, List[int]]] = {
            "BASE": base_cap2_raw["_raw"],
            "IN_CONTEXT_CEILING": ic_cap2_raw["_raw"],
        }

        def _eval_cap2_arm(arm_name: str, compiled: Dict[str, Any]) -> Dict[str, Any]:
            rep = poe.strip_rule_vectors(compiled)
            if compiled["weights"] is None:
                print(f"  [{arm_name:28s}] STATUS=ABSTAIN | rules={[(r['name'], 'G=' + str(r['Gamma']), 'pred_hit=' + str(r['pred_hit_rate'])) for r in rep['rules']]}")
                return {"compiled": rep, "abstained": True}
            w = compiled["weights"]
            new_dff = poe.widen_mlp_native(mlp_B, orig_state_B, w)
            fire = po.audit_organ_firing_analytical(
                w, ch_univ["x_bos_B"],
                ch_univ["x_B_eval_aligned"], ch_univ["mlpB_eval_aligned_n2"],
                ch_univ["x_B_eval_unaligned"], ch_univ["mlpB_eval_unaligned_n2"],
                rm_B=rm_B, bos_x_eval_B=ch_univ.get("bos_x_eval_B"),
            )
            gt_r, _ = po.eval_general_text_kl_ppl(model_B, tok_B, dev_B, corpus_univ["eval"], base_logprobs=base_lps)
            gt_r["delta_ppl"] = round(gt_r["ppl"] - base_gt["ppl"], 4)
            loc_r, _ = po.eval_196_locality(model_B, tok_B, dev_B, test_facts, base_correct=base_loc_ok)
            c2_r = eval_cap2_suite(model_B, tok_B, dev_B, eligibility_masks=elig_masks)
            # Exact torch.equal bit-unlearning verification (left-padded matching base_logits_p2)
            tok_B.padding_side = "left"
            try:
                unl = poe.verify_exact_bit_unlearning(model_B, tok_B, dev_B, mlp_B, orig_state_B, base_logits_p2, cap2_probe_prompts)
            finally:
                tok_B.padding_side = old_pad
            raw_arms[arm_name] = c2_r["_raw"]
            g_str = ",".join(f"{r['Gamma']:.3f}" for r in rep["rules"])
            print(
                f"  [{arm_name:28s}] {compiled['status']:8s} G=({g_str}) d_ff={new_dff} | "
                f"dir={c2_r['direct_fc_all']['acc']:.3f}/{c2_r['direct_em_all']['acc']:.3f} "
                f"rip={c2_r['ripple_4attr_fc_all']['acc']:.3f}/{c2_r['ripple_4attr_em_all']['acc']:.3f} "
                f"hop={c2_r['hops_2ms_fc_all']['acc']:.3f}/{c2_r['hops_2ms_em_all']['acc']:.3f} "
                f"inv={c2_r['invariant_em_retention']['acc']:.3f} hn_ff={c2_r['hardneg_false_flip_em']['acc']:.3f} | "
                f"loc={loc_r.get('newly_broken_unrelated', loc_r['newly_broken_all'])}/{loc_r.get('n_unrelated_orig_correct', loc_r['n_orig_correct_all'])} KL={gt_r['mean_kl']:.6f} bit_ul={unl['logits_bit_equal']}"
            )
            return {
                "compiled": rep,
                "abstained": False,
                "widened_d_ff": new_dff,
                "firing_audit": fire,
                "general_text": gt_r,
                "locality_196": loc_r,
                "cap2": _strip_cap2_raw(c2_r),
                "unlearning": unl,
            }

        arms_out: Dict[str, Any] = {}

        # Primary: Channel-Pushforward Bias-Free (exact frozen engine!)
        c_primary = poe.compile_channel_pushforward_organ(cap2_donor_rules, ch_univ, rm_B=rm_B, mode="biasfree", force_abstained=False)
        arms_out["CHANNEL_PUSHFORWARD_BIASFREE"] = _eval_cap2_arm("CHANNEL_PUSHFORWARD_BIASFREE", c_primary)
        if c_primary["weights"] is None:
            c_forced = poe.compile_channel_pushforward_organ(cap2_donor_rules, ch_univ, rm_B=rm_B, mode="biasfree", force_abstained=True)
            arms_out["CHANNEL_PUSHFORWARD_FORCED_IF_ABSTAINED"] = _eval_cap2_arm("CHANNEL_PUSHFORWARD_FORCED_IF_ABSTAINED", c_forced)

        # Control 1: SHUFFLED_R (auto ABSTAIN + forced control)
        c_sr_auto = poe.compile_channel_pushforward_organ(cap2_donor_rules, ch_univ, rm_B=rm_B, mode="biasfree", shuffle_read_seed=SEED + 501, force_abstained=False)
        arms_out["SHUFFLED_R_AUTO"] = _eval_cap2_arm("SHUFFLED_R_AUTO", c_sr_auto)
        c_sr_forced = poe.compile_channel_pushforward_organ(cap2_donor_rules, ch_univ, rm_B=rm_B, mode="biasfree", shuffle_read_seed=SEED + 501, force_abstained=True)
        arms_out["SHUFFLED_R_FORCED"] = _eval_cap2_arm("SHUFFLED_R_FORCED", c_sr_forced)

        # Control 2: CROSS_ORGAN_SWAP (Cap-1 Geography organ evaluated on Cap-2 Biology prompts!)
        c_cap1_swap = poe.compile_channel_pushforward_organ(cap1_donor_rules, ch_univ, rm_B=rm_B, mode="biasfree", force_abstained=False)
        if c_cap1_swap["weights"] is None:
            c_cap1_swap = poe.compile_channel_pushforward_organ(cap1_donor_rules, ch_univ, rm_B=rm_B, mode="biasfree", force_abstained=True)
        arms_out["CROSS_ORGAN_SWAP_CAP1_ON_CAP2"] = _eval_cap2_arm("CROSS_ORGAN_SWAP_CAP1_ON_CAP2", c_cap1_swap)

        # Upper Bound: NATIVE_B_ORACLE (exact same compiler on Recipient B's native Cap-2 states!)
        b_cap2_native = extract_cap2_donor_data(model_B, tok_B, dev_B, lB)["rules"]
        c_native = poe.compile_channel_pushforward_organ(b_cap2_native, channel=None, rm_B=rm_B, mode="biasfree", native_B_stats=ch_univ, force_abstained=False)
        arms_out["NATIVE_B_ORACLE"] = _eval_cap2_arm("NATIVE_B_ORACLE", c_native)

        # Exact two-sided McNemar comparisons
        mcn: Dict[str, Any] = {}
        primary_key = "CHANNEL_PUSHFORWARD_BIASFREE" if "CHANNEL_PUSHFORWARD_BIASFREE" in raw_arms else "CHANNEL_PUSHFORWARD_FORCED_IF_ABSTAINED"
        if primary_key in raw_arms:
            for ctrl in ("BASE", "SHUFFLED_R_FORCED", "CROSS_ORGAN_SWAP_CAP1_ON_CAP2"):
                if ctrl in raw_arms:
                    mcn[f"{primary_key}_vs_{ctrl}"] = _cap2_mcnemar_pack(raw_arms[primary_key], raw_arms[ctrl], elig_masks)
        if "NATIVE_B_ORACLE" in raw_arms and "BASE" in raw_arms:
            mcn["NATIVE_B_ORACLE_vs_BASE"] = _cap2_mcnemar_pack(raw_arms["NATIVE_B_ORACLE"], raw_arms["BASE"], elig_masks)

        out["recipients"][tgt] = {
            "model_key": rk,
            "layer": lB,
            "rm": rm_B,
            "socket_cache": ch_cache.name,
            "channel_summary": ch_univ["summary"],
            "eligibility_counts": elig_counts,
            "baseline": {
                "general_text": base_gt,
                "locality_196": base_loc,
                "cap2": _strip_cap2_raw(base_cap2_raw),
            },
            "in_context_ceiling": _strip_cap2_raw(ic_cap2_raw),
            "arms": arms_out,
            "mcnemar": mcn,
            "elapsed_sec": round(time.time() - tr0, 1),
        }

        # 3. Cross-Capability Co-Implantation on Solo SmolLM2-1.7B (d_ff 8192 -> 8200 -> 8196 -> 8192!)
        if tgt == "smollm2" and not args.skip_co_implantation:
            tco = time.time()
            print(f"\n--- [CROSS-CAPABILITY CO-IMPLANTATION] Solo smollm2-1.7b L{lB}: Cap-1 Geo (4) + Cap-2 Bio (4) -> d_ff 8192->8200 ---")
            c1_w = poe.compile_channel_pushforward_organ(cap1_donor_rules, ch_univ, rm_B=rm_B, mode="biasfree", force_abstained=False)
            c2_w = c_primary
            co_res: Dict[str, Any] = {
                "cap1_status": c1_w["status"],
                "cap2_status": c2_w["status"],
                "states": {},
            }
            if c1_w["weights"] is not None and c2_w["weights"] is not None:
                # Capture solo Cap-1 and solo Cap-2 probe logits for exact torch.equal slicing certificates!
                tok_B.padding_side = "left"
                enc_p1 = tok_B(cap1_probe_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
                with torch.no_grad():
                    base_logits_p1 = model_B(input_ids=enc_p1["input_ids"].to(dev_B), attention_mask=enc_p1["attention_mask"].to(dev_B), use_cache=False).logits.cpu()
                poe.widen_mlp_native(mlp_B, orig_state_B, c1_w["weights"])
                with torch.no_grad():
                    solo_cap1_logits_p1 = model_B(input_ids=enc_p1["input_ids"].to(dev_B), attention_mask=enc_p1["attention_mask"].to(dev_B), use_cache=False).logits.cpu()
                po.unlearn_mlp_native(mlp_B, orig_state_B)

                poe.widen_mlp_native(mlp_B, orig_state_B, c2_w["weights"])
                with torch.no_grad():
                    solo_cap2_logits_p2 = model_B(input_ids=enc_p2["input_ids"].to(dev_B), attention_mask=enc_p2["attention_mask"].to(dev_B), use_cache=False).logits.cpu()
                po.unlearn_mlp_native(mlp_B, orig_state_B)
                tok_B.padding_side = old_pad

                w_joint = poe.concat_compiled_organs([c1_w["weights"], c2_w["weights"]])
                n1 = int(c1_w["weights"]["W_g"].shape[0])
                n2 = int(c2_w["weights"]["W_g"].shape[0])

                def _eval_co_state(st_name: str, cur_dff: int) -> Dict[str, Any]:
                    gt_r, _ = po.eval_general_text_kl_ppl(model_B, tok_B, dev_B, corpus_univ["eval"], base_logprobs=base_lps)
                    gt_r["delta_ppl"] = round(gt_r["ppl"] - base_gt["ppl"], 4)
                    loc_r, _ = po.eval_196_locality(model_B, tok_B, dev_B, test_facts, base_correct=base_loc_ok, exclude_mask=fr_it_exclude)
                    c1_qa = pmf.eval_organ1_detailed(model_B, tok_B, dev_B, ds_fr_it_test)
                    c2_qa = eval_cap2_suite(model_B, tok_B, dev_B, eligibility_masks=elig_masks)
                    print(
                        f"  [CoImplant {st_name:24s}] d_ff={cur_dff} | "
                        f"Cap1(dir={c1_qa['direct_country_em']:.3f} rip3={c1_qa['unseen_ripple_3attr_em']:.3f}) | "
                        f"Cap2(dir={c2_qa['direct_fc_all']['acc']:.3f}/{c2_qa['direct_em_all']['acc']:.3f} "
                        f"rip={c2_qa['ripple_4attr_fc_all']['acc']:.3f}/{c2_qa['ripple_4attr_em_all']['acc']:.3f}) | "
                        f"loc={loc_r.get('newly_broken_unrelated', loc_r['newly_broken_all'])}/{loc_r.get('n_unrelated_orig_correct', loc_r['n_orig_correct_all'])} KL={gt_r['mean_kl']:.6f}"
                    )
                    return {
                        "d_ff": cur_dff,
                        "general_text": gt_r,
                        "locality_196": loc_r,
                        "cap1": pmf.strip_raw_vecs(c1_qa),
                        "cap2": _strip_cap2_raw(c2_qa),
                    }

                # State 1: Joint Cap-1 + Cap-2 (8192 -> 8200)
                dff_j = poe.widen_mlp_native(mlp_B, orig_state_B, w_joint)
                co_res["states"]["JOINT_CAP1_PLUS_CAP2"] = _eval_co_state("JOINT_CAP1_PLUS_CAP2", dff_j)

                # State 2: Slice out Cap-1 rows [0..n1-1], keep Cap-2 rows [n1..n1+n2-1] (8200 -> 8196)
                dff_c2 = poe.slice_mlp_rows_native(mlp_B, orig_state_B, list(range(n1, n1 + n2)))
                tok_B.padding_side = "left"
                with torch.no_grad():
                    cur_p2 = model_B(input_ids=enc_p2["input_ids"].to(dev_B), attention_mask=enc_p2["attention_mask"].to(dev_B), use_cache=False).logits.cpu()
                tok_B.padding_side = old_pad
                cap2_bit_eq = bool(torch.equal(cur_p2, solo_cap2_logits_p2))
                cap2_max_diff = float(torch.max(torch.abs(cur_p2 - solo_cap2_logits_p2)).item())
                st_c2 = _eval_co_state("SLICE_OUT_CAP1_KEEP_CAP2", dff_c2)
                st_c2["matches_solo_cap2_logits_bit_equal"] = cap2_bit_eq
                st_c2["matches_solo_cap2_max_abs_dlogit"] = cap2_max_diff
                co_res["states"]["SLICE_OUT_CAP1_KEEP_CAP2"] = st_c2

                # State 3: Re-widen joint & slice out Cap-2 rows, keep Cap-1 rows [0..n1-1] (8200 -> 8196)
                poe.widen_mlp_native(mlp_B, orig_state_B, w_joint)
                dff_c1 = poe.slice_mlp_rows_native(mlp_B, orig_state_B, list(range(0, n1)))
                tok_B.padding_side = "left"
                with torch.no_grad():
                    cur_p1 = model_B(input_ids=enc_p1["input_ids"].to(dev_B), attention_mask=enc_p1["attention_mask"].to(dev_B), use_cache=False).logits.cpu()
                tok_B.padding_side = old_pad
                cap1_bit_eq = bool(torch.equal(cur_p1, solo_cap1_logits_p1))
                cap1_max_diff = float(torch.max(torch.abs(cur_p1 - solo_cap1_logits_p1)).item())
                st_c1 = _eval_co_state("SLICE_OUT_CAP2_KEEP_CAP1", dff_c1)
                st_c1["matches_solo_cap1_logits_bit_equal"] = cap1_bit_eq
                st_c1["matches_solo_cap1_max_abs_dlogit"] = cap1_max_diff
                co_res["states"]["SLICE_OUT_CAP2_KEEP_CAP1"] = st_c1

                # State 4: Slice all remaining rows back to 8192 & verify bit-exact restoration on BOTH Cap-1 & Cap-2!
                tok_B.padding_side = "left"
                try:
                    unl_p1 = poe.verify_exact_bit_unlearning(model_B, tok_B, dev_B, mlp_B, orig_state_B, base_logits_p1, cap1_probe_prompts)
                    unl_p2 = poe.verify_exact_bit_unlearning(model_B, tok_B, dev_B, mlp_B, orig_state_B, base_logits_p2, cap2_probe_prompts)
                finally:
                    tok_B.padding_side = old_pad
                co_res["states"]["SLICE_ALL_RESTORE_BASE"] = {"cap1_probe": unl_p1, "cap2_probe": unl_p2}
                print(
                    f"  [CoImplant BIT-EXACT CERTS] keep_cap2==solo_cap2: {cap2_bit_eq} (max={cap2_max_diff:.1e}) | "
                    f"keep_cap1==solo_cap1: {cap1_bit_eq} (max={cap1_max_diff:.1e}) | base_restore: {unl_p1['logits_bit_equal'] and unl_p2['logits_bit_equal']}"
                )
            else:
                print(f"  [CoImplant] Skipped — one or both organs abstained: cap1={c1_w['status']}, cap2={c2_w['status']}")
            co_res["elapsed_sec"] = round(time.time() - tco, 1)
            out["co_implantation_smollm2"] = co_res

        po.unload_model(model_B)

    out["meta"]["total_elapsed_sec"] = round(time.time() - t0, 1)
    out_p = Path(args.out)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SEALED CAP-2 COMPLETE] ENGINE={engine_sha[:12]} SEAL={sealed_sha[:12]} | Saved {out_p} ({out['meta']['total_elapsed_sec']}s)")
    return out


# =============================================================================
# 6. Strict Sealing Verification & CPU Unit Tests (ZERO GPU!)
# =============================================================================


def run_unit_test() -> None:
    print("[bio_ripple_sealed --unit-test] Verifying counts, zero overlaps, exact frame balance, and SHA-256 seal...")

    # 1. Exact counts
    assert len(BIRD_TRAIN_24) == 24 and len(BIRD_TEST_24) == 24
    assert len(MAMMAL_TRAIN_24) == 24 and len(MAMMAL_TEST_24) == 24
    assert len(NONTARGET_TRAIN_24) == 24 and len(HARDNEG_TEST_24) == 24
    assert len(CAP2_TRAIN_TEMPLATES) == 6

    # 2. Zero duplicates & zero overlap across ALL 6 species splits
    all_lists = [
        ("BIRD_TRAIN", BIRD_TRAIN_24),
        ("BIRD_TEST", BIRD_TEST_24),
        ("MAMMAL_TRAIN", MAMMAL_TRAIN_24),
        ("MAMMAL_TEST", MAMMAL_TEST_24),
        ("NONTARGET_TRAIN", NONTARGET_TRAIN_24),
        ("HARDNEG_TEST", [h["name"] for h in HARDNEG_TEST_24]),
    ]
    seen_global: Dict[str, str] = {}
    for lname, lst in all_lists:
        for sp in lst:
            k = sp.lower().strip()
            assert k not in seen_global, f"Duplicate/overlap species {sp!r} in {lname} and {seen_global[k]}"
            seen_global[k] = lname
    assert len(seen_global) == 144

    # 3. Zero word-level overlap with Cap-1 geography entities, Pillar 3c T3 entities, and ALL dataset.json fields!
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        ds_facts = json.load(f)["facts"]
    forbidden: set[str] = set()
    for x in ds_facts:
        for field in ("subject", "answer", "prompt"):
            forbidden.update(re.findall(r"[a-z]+", str(x.get(field, "")).lower()))
        for a in x.get("aliases", []):
            forbidden.update(re.findall(r"[a-z]+", str(a).lower()))

    import pilot_organs_p0 as po
    for attr in ("ORGAN1_TRAIN_ENTITIES", "ORGAN1_TEST_ENTITIES", "ORGAN1_HARDNEG_TRAIN", "ORGAN1_HARDNEG_TEST"):
        if hasattr(po, attr):
            for item in getattr(po, attr):
                forbidden.update(re.findall(r"[a-z]+", item["name"].lower()))

    try:
        import pilot_graft_t3_gap_scaled as pg
        for attr in dir(pg):
            val = getattr(pg, attr)
            if isinstance(val, dict):
                for lst in val.values():
                    if isinstance(lst, list):
                        for it in lst:
                            if isinstance(it, str):
                                forbidden.update(re.findall(r"[a-z]+", it.lower()))
    except Exception:
        pass

    ov = sorted(sp for sp in seen_global if sp in forbidden)
    assert not ov, f"Word-level overlap with dataset.json or Cap-1/T3 entities: {ov}"

    # 4. Exact frame-counterbalance check (exact empirical frame_only_auroc == 0.5000!)
    donor_prompts = build_cap2_donor_prompts()
    for rk in ("rule1_Bird_to_Fish", "rule2_Mammal_to_Reptile"):
        rd = donor_prompts[rk]
        assert len(rd["pos_prompts"]) == 24 * 6 == 144
        assert len(rd["neg_prompts"]) == 48 * 6 == 288
        pos_dist = Counter(rd["pos_tmpl_ids"])
        neg_dist = Counter(rd["neg_tmpl_ids"])
        # Compute exact empirical Bayes P(y=1 | template) & ROC-AUC
        scores_pos = [pos_dist[t] / (pos_dist[t] + neg_dist[t]) for t in rd["pos_tmpl_ids"]]
        scores_neg = [pos_dist[t] / (pos_dist[t] + neg_dist[t]) for t in rd["neg_tmpl_ids"]]
        wins = sum(1.0 if sp > sn else (0.5 if sp == sn else 0.0) for sp in scores_pos for sn in scores_neg)
        frame_only_auroc = wins / (len(scores_pos) * len(scores_neg))
        assert abs(frame_only_auroc - 0.5) < 1e-12, f"Frame confound detected: AUROC={frame_only_auroc}"
        for tid in range(6):
            assert abs(pos_dist[tid] / len(rd["pos_tmpl_ids"]) - 1.0 / 6.0) < 1e-12
            assert abs(neg_dist[tid] / len(rd["neg_tmpl_ids"]) - 1.0 / 6.0) < 1e-12
        # Verify spans slice back to exact species string & ZERO test species appear in train prompts
        test_species = set(BIRD_TEST_24 + MAMMAL_TEST_24 + [h["name"] for h in HARDNEG_TEST_24])
        for p, (c0, c1) in zip(rd["pos_prompts"] + rd["neg_prompts"], rd["pos_spans"] + rd["neg_spans"]):
            sp_sliced = p[c0:c1].lower()
            assert sp_sliced in seen_global and sp_sliced not in test_species
        # Verify ZERO ripple words leak into construction templates
        ripple_leak_words = ["scales", "feathers", "gills", "lungs", "fins", "wings", "cold-blooded", "warm-blooded", "milk", "aquarium", "heat lamp"]
        for p in rd["pos_prompts"] + rd["neg_prompts"]:
            pl = p.lower()
            for rw in ripple_leak_words:
                assert rw not in pl, f"Ripple word {rw!r} leaked into construction prompt: {p!r}"

    # 5. Sealed evaluation battery counts, structure & 5-tokenizer collision/space-split audit (CPU-only!)
    suite = build_cap2_eval_items(in_context_ceiling=False)
    assert len(suite["direct"]) == 48
    assert len(suite["ripple"]) == 192
    assert len(suite["hops"]) == 96
    assert len(suite["invariant"]) == 48
    assert len(suite["hardneg"]) == 24
    suite_ic = build_cap2_eval_items(in_context_ceiling=True)
    assert suite_ic["ripple"][0]["prompt"].startswith("Suppose ")

    hf_cache = LAB_DIR / "hf_cache"
    tok_checked = 0
    if hf_cache.exists():
        from transformers import AutoTokenizer
        for sub in ("qwen2.5-3b", "qwen2.5-1.5b", "smollm2-1.7b", "granite-3.1-2b", "falcon3-3b"):
            mdir = hf_cache / sub
            if not mdir.exists():
                continue
            tok = AutoTokenizer.from_pretrained(str(mdir), local_files_only=True)
            tok_checked += 1
            for bucket, k_a, k_b in [
                ("direct", "new_fc", "old_fc"),
                ("ripple", "new_fc", "old_fc"),
                ("hops", "new_fc", "old_fc"),
                ("invariant", "fc_pos", "fc_neg"),
                ("hardneg", "orig_fc", "forbid_fc"),
            ]:
                seen_pairs = set()
                for it in suite[bucket]:
                    pair = (tuple(it[k_a]), tuple(it[k_b]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    a_ids = [tok.encode(c, add_special_tokens=False) for c in it[k_a]]
                    b_ids = [tok.encode(c, add_special_tokens=False) for c in it[k_b]]
                    assert all(len(x) > 0 for x in a_ids + b_ids), f"Empty tok on {sub}: {pair}"
                    a0 = {x[0] for x in a_ids}
                    b0 = {x[0] for x in b_ids}
                    assert a0.isdisjoint(b0), f"First-token collision on {sub} {bucket}: {pair} -> {a_ids} vs {b_ids}"
                    for c, x in zip(list(it[k_a]) + list(it[k_b]), a_ids + b_ids):
                        assert not (len(x) > 1 and tok.decode([x[0]]).strip() == ""), f"Leading-space split on {sub}: {c!r} -> {x}"

    # 6. Exact statistical helpers unit tests
    lo, hi = wilson_ci(36, 48)
    assert lo is not None and hi is not None and 0.60 < lo < 0.62 and 0.85 < hi < 0.86
    blo, bhi = species_bootstrap_ci(BIRD_TEST_24 * 4, [1] * 72 + [0] * 24)
    assert blo is not None and bhi is not None and blo <= 0.75 <= bhi
    mc = exact_mcnemar([1] * 20 + [0] * 4, [0] * 24)
    assert mc["n10"] == 20 and mc["n01"] == 0 and mc["p_value"] < 1e-5

    # 7. CPU-only synthetic end-to-end integration test with `platonic_organ_engine.py`!
    import platonic_organ_engine as poe
    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaMLP
    import pilot_organs_p0 as po_ref

    rng = np.random.default_rng(SEED + 77)
    D_A, D_B = 32, 24
    # Synthetic generative channel A -> B
    R_true, _ = np.linalg.qr(rng.standard_normal((D_A, D_B)))
    R_true = R_true.T  # (D_B, D_A)
    Xa_fit = rng.standard_normal((400, D_A))
    Xb_fit = Xa_fit @ R_true.T + 0.05 * rng.standard_normal((400, D_B))
    Xa_res = rng.standard_normal((200, D_A))
    Xb_res = Xa_res @ R_true.T + 0.05 * rng.standard_normal((200, D_B))
    bos_B = rng.standard_normal((16, D_B))
    ch_syn = poe.fit_generative_channel(Xa_fit, Xb_fit, Xa_res, Xb_res, Xa_fit, Xb_fit, Xa_res, Xb_res, bos_B)

    # Synthetic Cap-2 donor rules matching exact extract_cap2_donor_data() schema!
    u1, u2 = rng.standard_normal(D_A), rng.standard_normal(D_A)
    u1 /= np.linalg.norm(u1); u2 /= np.linalg.norm(u2)
    f_pos = np.repeat(np.arange(6), 24)
    f_neg = np.repeat(np.arange(6), 48)
    syn_cap2_rules = [
        {
            "name": "Bird->Fish",
            "X_pos_A": 5.0 * u1 + 0.3 * rng.standard_normal((144, D_A)),
            "X_neg_dom_A": 0.3 * rng.standard_normal((288, D_A)),
            "pos_frame_ids": f_pos, "neg_frame_ids": f_neg,
            "v_A": rng.standard_normal(D_A),
        },
        {
            "name": "Mammal->Reptile",
            "X_pos_A": 5.0 * u2 + 0.3 * rng.standard_normal((144, D_A)),
            "X_neg_dom_A": 0.3 * rng.standard_normal((288, D_A)),
            "pos_frame_ids": f_pos, "neg_frame_ids": f_neg,
            "v_A": rng.standard_normal(D_A),
        },
    ]
    c_ok = poe.compile_channel_pushforward_organ(syn_cap2_rules, ch_syn, rm_B=1.0, mode="biasfree")
    assert c_ok["status"] == "COMPILED" and c_ok["weights"] is not None
    assert c_ok["weights"]["W_g"].shape == (4, D_B) and c_ok["weights"]["use_bias"] is False
    c_shuf = poe.compile_channel_pushforward_organ(syn_cap2_rules, ch_syn, rm_B=1.0, mode="biasfree", shuffle_read_seed=SEED + 501)
    assert c_shuf["status"] == "ABSTAIN" and c_shuf["weights"] is None

    # Synthetic Co-Implantation (4 + 4 = 8 rows) + selective slicing + bit-exact unlearning on real LlamaMLP!
    cfg = LlamaConfig(hidden_size=D_B, intermediate_size=64, num_attention_heads=4, num_key_value_heads=4, hidden_act="silu", mlp_bias=False)
    mlp = LlamaMLP(cfg)
    orig = po_ref.save_original_mlp_state(mlp)
    x_probe = torch.randn(4, 3, D_B)
    with torch.no_grad():
        y_base = mlp(x_probe)
    w_joint = poe.concat_compiled_organs([c_ok["weights"], c_ok["weights"]])
    assert poe.widen_mlp_native(mlp, orig, w_joint) == 72
    assert poe.slice_mlp_rows_native(mlp, orig, [4, 5, 6, 7]) == 68
    po_ref.unlearn_mlp_native(mlp, orig)
    with torch.no_grad():
        y_restored = mlp(x_probe)
    assert torch.equal(y_base, y_restored) and mlp.gate_proj.out_features == 64

    seal_hash = compute_sealed_sha256()
    assert seal_hash == "d321fd0255c47cd3ab0113bb966acdf248296ca12da0796c0e21dfdb160bdf6d", f"SEAL HASH DRIFTED: {seal_hash}"
    print(f"[PASS] 144 unique species (24+24 birds, 24+24 mammals, 24 train negs, 24 hard negs), 0 word-level overlaps with dataset.json/Cap-1/T3!")
    print(f"[PASS] Exact frame-counterbalance verified: 6 templates × 1/6 exact pos/neg balance -> empirical frame_only_auroc == 0.5000, 0 ripple words, 0 test species in construction!")
    print(f"[PASS] Sealed eval battery verified: 48 direct + 192 unseen ripple + 96 multi-sentence hops + 48 invariant + 24 hard negatives = 408 trials!")
    print(f"[PASS] Forced-choice tokenizer audit verified across {tok_checked}/5 cached tokenizers: 0 first-token collisions, 0 leading-space splits!")
    print(f"[PASS] Synthetic CPU end-to-end integration with platonic_organ_engine.py PASS (COMPILED + SHUFFLED_R ABSTAIN + 64->72->68->64 torch.equal)!")
    print(f"[ENGINE SHA-256]: {poe.get_engine_sha256()}")
    print(f"[SEALED SHA-256]: {seal_hash}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly Sealed Cap-2 Benchmark: BIO-RIPPLE (Bird->Fish, Mammal->Reptile)")
    parser.add_argument("--unit-test", action="store_true")
    parser.add_argument("--print-seal", action="store_true")
    parser.add_argument("--run-cap2", action="store_true", help="ONLY run AFTER Cap-1 & platonic_organ_engine.py freeze!")
    parser.add_argument("--targets", type=str, default="smollm2,granite,falcon3,qwen1.5b")
    parser.add_argument("--out", type=str, default=str(OUT_PATH))
    parser.add_argument("--refit-donor", action="store_true")
    parser.add_argument("--allow-fit-channel", action="store_true")
    parser.add_argument("--skip-co-implantation", action="store_true")
    args = parser.parse_args()
    if args.unit_test:
        run_unit_test()
        return
    if args.print_seal:
        print(compute_sealed_sha256())
        return
    if args.run_cap2:
        run_cap2_sealed(args)
        return
    print("BIO-RIPPLE is STRICTLY SEALED. Run `--unit-test` for CPU verification, or `--run-cap2` ONLY after Cap-1 & engine freeze!")


if __name__ == "__main__":
    main()
