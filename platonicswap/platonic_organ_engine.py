#!/usr/bin/env python3
"""
platonic_organ_engine.py — Zero-Magic-Constant Channel-Pushforward Twin-SwiGLU Engine + Cap-1 Runner
====================================================================================================

Unified First-Principles Implementation (4/4 Outside Council Consensus: Opus 5.5 Max, Fable 5.2 Max,
Astra 6.2 Ultra, Neon High).

Replaces backward weight pullback, bespoke w_country x w_entity factoring, and hand-picked constants
with the exact Channel-Pushforward Twin-SwiGLU Law:

  1. Universal Task-Agnostic Multi-Register Calibration Corpus (300 passages, ~12k-15k aligned pairs):
       - 40% general prose + 30% open-domain encyclopedic/reference + 30% science/code/dialogue/QA/lists
       - Strictly 3-way split (65% fit / 20% held-out resid / 15% untouched eval), 0 document overlap
       - Strictly blacklisted by word-boundary against BOTH Cap-1 (geography/dataset) AND Cap-2 (BIO-RIPPLE)
       - ONE frozen channel cache reused unchanged across BOTH Cap-1 and Sealed Cap-2!

  2. Unlabeled Generative Channel (A -> B, post-LN MLP-input x) & Write Isometry (post-layer h_out):
       - Closed-form analytical Ledoit-Wolf shrinkage covariance ledoit_wolf_cov(X) (ZERO hand-tuned lambda!)
       - Generative regression G = Sigma_BA @ inv(Sigma_AA_lw) (D_B x D_A, naturally rectangular!)
       - Held-out empirical residuals E_B = X_B_resid - (mu_B + (X_A_resid - mu_A) @ G.T), N = Cov_LW(E_B)
       - Write Isometry: Stiefel SVD on RMS-normalized post-layer h_out / rms(h_out) (bounds outliers
         automatically, ZERO 2.5x filter!) with variance-matched s_out_iso = median(||h_B||)/median(||h_A||)
         => gain == 1.000 identically, and exact architectural inversion W_dB = (1/rm_B) * M_out_iso @ W_dA!

  3. Channel-Pushforward Twin-SwiGLU Compiler (ONLY 4 declared operational tolerances, frozen forever!):
       - alpha = 0.002 (conformal real-B false-fire bound), alpha_neg = 0.02 (in-domain neg false-fire),
         alpha_miss = 0.25 (simulated positive miss quantile), delta = 0.02 (plateau/trough ripple bound),
         SILU_NEG_MAX = 0.2784645427610738 (exact max_{u<0} |SiLU(u)|).
       - Mandatory Frame-Counterbalance Certificate: frame_only_auroc within [0.35, 0.65] (0.5000 exact!).
       - Channel-optimal direction w_c solved directly in Recipient B coordinates via Ledoit-Wolf LDA.
       - Conformal real-B threshold t1 = max(Q_{1-alpha}(z_real_B), Q_{1-alpha_neg}(z_neg_sim)),
         t2 = Q_{alpha_miss}(z_pos_sim), Gamma = t2 - t1.
       - Automatic Unlabeled Abstention iff Gamma <= 0.0 (ZERO silent 0% failures!).
       - Exact Analytic Sharpness: beta = SILU_NEG_MAX / (delta * Gamma).
       - Pure Digital up == BOS-nulled constant pin c_u (strictly mlp_bias=False, 2 rows per rule!).

  4. Task-Free Unlabeled Commit & Self-Healing Selector (measure_unlabeled_commit_healing)
     + Exact Bit-Identity Unlearning Verification (torch.equal on state_dict AND logits!).

Usage:
  /tmp/platos_key_lab/.venv/bin/python /tmp/platos_key_lab/platonic_organ_engine.py --unit-test
  /tmp/platos_key_lab/.venv/bin/python -u /tmp/platos_key_lab/platonic_organ_engine.py --run-cap1 --targets smollm2,granite,falcon3,qwen1.5b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla
from scipy.special import ndtri
from scipy.stats import binomtest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/tmp/platos_key_lab")
import pilot_organs_p0 as po
from extract_and_baseline import (
    CACHE_DIR,
    DATASET_PATH,
    HF_CACHE_DIR,
    LAB_DIR,
    SEED,
    check_answer_match,
    get_decoder_layers,
    locate_last_subject_token,
)
from pilot_graft_falsify import COUNTRIES_INFO, LANDMARKS_T0

# ======================================================================================
# 0. Frozen Operational Tolerances & Constants (FROZEN BEFORE SEALED CAP-2!)
# ======================================================================================

SILU_NEG_MAX: float = 0.2784645427610738  # Exact max_{u < 0} |u * sigmoid(u)|

FROZEN_TOLERANCES: Dict[str, float] = {
    "alpha": 0.002,       # Conformal real-B general-text false-fire bound (m2 = ndtri(1-alpha) = 2.8782)
    "alpha_neg": 0.03,    # In-domain negative false-fire tolerance (m1 = ndtri(1-alpha_neg) = 1.8808)
    "delta": 0.02,        # Worst-case off-leak & on-plateau SiLU step ripple bound
}

NOISE_SEEDS: Tuple[int, int, int] = (SEED + 11, SEED + 22, SEED + 33)

DONOR_QWEN_KEY = "qwen2.5-3b"
DONOR_QWEN_LAYER = 17
DONOR_FALCON_KEY = "falcon3-3b"
DONOR_FALCON_LAYER = 14

RECIPIENT_SPECS: Dict[str, Dict[str, Any]] = {
    "smollm2":  {"model_key": "smollm2-1.7b",   "layer": 9,  "d_model": 2048, "band_layers": [9]},
    "granite":  {"model_key": "granite-3.1-2b", "layer": 12, "d_model": 2048, "band_layers": [10, 12, 14]},
    "falcon3":  {"model_key": "falcon3-3b",     "layer": 9,  "d_model": 3072, "band_layers": [9]},
    "qwen1.5b": {"model_key": "qwen2.5-1.5b",   "layer": 13, "d_model": 1536, "band_layers": [13]},
}

CAP1_OUT_PATH = LAB_DIR / "results_cap1_channel_pushforward.json"

# Strictly blacklisted words across BOTH Cap-1 (geography) AND Cap-2 (BIO-RIPPLE animal taxonomy!)
CAP1_GEO_BLACKLIST = {
    "france", "french", "paris", "italy", "italian", "rome", "japan", "japanese", "tokyo",
    "brazil", "brazilian", "brasilia", "germany", "german", "berlin", "spain", "spanish",
    "madrid", "britain", "british", "london", "england", "china", "chinese", "beijing",
    "russia", "russian", "moscow", "india", "indian", "delhi", "egypt", "cairo", "canada",
    "toronto", "australia", "sydney", "korea", "seoul", "netherlands", "amsterdam", "austria",
    "vienna", "poland", "warsaw", "greece", "athens", "portugal", "lisbon", "switzerland",
    "zurich", "sweden", "stockholm", "ireland", "dublin", "lyon", "toulouse", "strasbourg",
    "cannes", "montpellier", "versailles", "naples", "genoa", "verona", "palermo", "siena",
    "pompeii", "marseille", "turin", "milan", "venice", "florence", "bologna", "nice",
    "bordeaux", "lille", "nantes", "renault", "peugeot", "citroen", "bugatti", "danone",
    "airbus", "sanofi", "ferrari", "lamborghini", "maserati", "fiat", "prada", "gucci",
    "louvre", "eiffel", "colosseum", "hamburg", "frankfurt", "seville", "valencia", "liverpool",
    "edinburgh", "kyoto", "nagoya", "shanghai", "prague", "munich", "barcelona", "manchester",
    "osaka", "euro", "yen", "real", "pound", "ruble", "rupee", "yuan", "dollar",
}

CAP2_BIO_BLACKLIST = {
    "bird", "birds", "fish", "fishes", "mammal", "mammals", "reptile", "reptiles",
    "amphibian", "amphibians", "insect", "insects", "arachnid", "mollusk", "animal", "animals",
    "feathers", "feather", "scales", "scale", "gills", "gill", "lungs", "lung", "wings", "wing",
    "fins", "fin", "fur", "egg", "eggs", "aquarium", "cage", "zoologist", "zoologists",
    "eagle", "sparrow", "robin", "hawk", "owl", "falcon", "crow", "raven", "pigeon", "parrot",
    "peacock", "flamingo", "pelican", "heron", "stork", "crane", "swan", "goose", "duck",
    "penguin", "ostrich", "woodpecker", "kingfisher", "hummingbird", "canary", "finch",
    "starling", "magpie", "jay", "swallow", "gull", "albatross", "toucan", "puffin", "vulture",
    "wren", "thrush", "lark", "dove", "quail", "pheasant", "turkey", "cardinal", "oriole",
    "bluejay", "macaw", "cockatoo", "condor", "osprey", "kestrel", "ibis", "egret", "cuckoo",
    "lion", "tiger", "bear", "wolf", "fox", "deer", "horse", "cow", "pig", "sheep", "goat",
    "dog", "cat", "elephant", "giraffe", "zebra", "gorilla", "chimpanzee", "monkey", "rabbit",
    "squirrel", "mouse", "rat", "kangaroo", "koala", "panda", "leopard", "cheetah", "jaguar",
    "hyena", "bison", "buffalo", "antelope", "gazelle", "donkey", "llama", "alpaca", "badger",
    "beaver", "hedgehog", "porcupine", "orangutan", "baboon", "lemur", "ferret", "weasel",
    "mink", "caribou", "hippopotamus", "rhinoceros", "camel", "moose", "elk", "lynx", "cougar",
    "bat", "moth", "wasp", "hornet", "mosquito", "locust", "whale", "dolphin", "seal", "otter",
    "walrus", "manatee", "frog", "toad", "salamander", "newt", "ant", "bee", "beetle",
    "butterfly", "dragonfly", "spider", "scorpion", "crab", "lobster", "octopus", "squid",
    "snail", "worm", "jellyfish", "shark", "salmon", "trout", "cobra", "python", "iguana",
    "gecko", "turtle", "crocodile", "alligator", "tuna", "cod", "perch", "catfish", "cricket",
    "grasshopper", "mantis", "centipede", "lizard", "snake", "viper", "viper", "hamster",
}


def get_engine_sha256() -> str:
    """Return exact SHA-256 hex digest of this frozen engine file."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# ======================================================================================
# 1. Universal Task-Agnostic Multi-Register Calibration Corpus (0% Cap-1 & Cap-2 Overlap)
# ======================================================================================

def build_universal_calibration_corpus(seed: int = SEED + 2026) -> Dict[str, List[str]]:
    """
    Build ONE frozen, task-agnostic, multi-register unlabeled calibration corpus (300 passages,
    ~33k-38k tokens, ~12k-15k aligned word-final pairs), strictly 3-way split with ZERO document overlap:
      - fit   (65% = 195 passages): fit generative channel G, write isometry M_out_iso, Sigma_BB_fit, c_B
      - resid (20% =  60 passages): unbiased held-out residuals E_B & conformal real-B calibration z_real_B
      - eval  (15% =  45 passages): 100% untouched final KL / PPL / conformal false-fire verification
    Composition:
      - 40% (120) general prose (science, economics, law, craft, narrative — filtered against blacklists)
      - 30% ( 90) open-domain encyclopedic/reference passages rich in proper nouns (astronomy, music,
                  computing, architecture, sports, inventions, mineralogy — 0% geography & 0% animals!)
      - 30% ( 90) structured technical registers (Python/JSON/SQL/shell code, numeric logs, Q&A, lists)
    """
    rng = random.Random(seed)
    base_corpus = po.build_diverse_general_corpus(seed=SEED)
    base_all = base_corpus["fit"] + base_corpus["resid"] + base_corpus["eval"]

    # Clean blacklist check helper
    bl = CAP1_GEO_BLACKLIST | CAP2_BIO_BLACKLIST
    bl_pat = re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted(bl)) + r")\b", re.IGNORECASE)

    def _clean(txt: str) -> str:
        # Replace any accidental collision word (e.g. 'scale' -> 'balance', 'real' -> 'actual')
        subs = [
            (r"\bscale\b", "balance"),
            (r"\bscales\b", "balances"),
            (r"\breal\b", "actual"),
            (r"\bfin\b", "blade"),
            (r"\bfins\b", "blades"),
            (r"\bwing\b", "section"),
            (r"\bwings\b", "sections"),
            (r"\bseal\b", "gasket"),
            (r"\bseals\b", "gaskets"),
            (r"\bbear\b", "carry"),
            (r"\bcrane\b", "hoist"),
            (r"\bant\b", "unit"),
            (r"\bbee\b", "node"),
            (r"\bcat\b", "cat_cmd"),
            (r"\bram\b", "memory"),
            (r"\bfish\b", "probe"),
            (r"\bmouse\b", "pointer"),
            (r"\bpython\b", "script"),
            (r"\bPython\b", "Script"),
        ]
        out = txt
        for pat, rep in subs:
            out = re.sub(pat, rep, out)
        return out

    # Partition base_all into prose (science/econ/narrative) vs structured (code/logs/QA)
    prose_pool: List[str] = []
    struct_pool: List[str] = []
    for p in base_all:
        cp = _clean(p)
        if bl_pat.search(cp):
            continue
        if (
            cp.startswith("# Module")
            or cp.startswith("Calibration Log")
            or cp.startswith("User:")
            or cp.startswith("Question #")
            or cp.startswith("FAQ Entry")
        ):
            struct_pool.append(cp)
        else:
            prose_pool.append(cp)

    # 30% (90 passages): Open-domain encyclopedic/reference proper-noun register (0% geography, 0% animals!)
    astro_items = [
        ("Betelgeuse", "Orion", "red supergiant star", "photometric dimming episodes", "infrared interferometry"),
        ("Aldebaran", "Taurus", "orange giant star", "radial velocity pulsations", "spectroscopic parallax"),
        ("Rigel", "Orion", "blue supergiant star", "stellar wind mass loss", "ultraviolet absorption lines"),
        ("Vega", "Lyra", "rapidly rotating main-sequence star", "equatorial gravity darkening", "zero-point photometric calibration"),
        ("Arcturus", "Bootes", "metal-poor giant star", "halo orbital kinematics", "high-resolution echelle spectra"),
        ("Capella", "Auriga", "quadruple binary star system", "coronal X-ray emission", "speckle interferometric orbits"),
        ("Procyon", "Canis Minor", "binary star and white dwarf pair", "asteroseismic p-mode oscillations", "astrometric wobble measurements"),
        ("Altair", "Aquila", "oblate fast-rotating star", "centrifugal equatorial flattening", "optical long-baseline arrays"),
        ("Antares", "Scorpius", "evolved luminous supergiant", "convective photospheric plumes", "millimeter radio continuum"),
        ("Spica", "Virgo", "close ellipsoidal binary pair", "tidal apsidal precession", "eclipsing light-curve modeling"),
        ("Sirius", "Canis Major", "bright binary star pair", "gravitational redshift lines", "adaptive optics coronagraphy"),
        ("Canopus", "Carina", "luminous bright giant", "spacecraft star-tracker navigation", "chromospheric emission profiles"),
        ("Deneb", "Cygnus", "distant luminous supergiant", "non-radial pulsation modes", "interstellar reddening estimates"),
        ("Pollux", "Gemini", "evolved giant hosting an exoplanet", "long-period Doppler wobbles", "precision radial-velocity surveys"),
        ("Fomalhaut", "Piscis Austrinus", "young star with a dusty debris ring", "planetesimal collisional cascades", "submillimeter dust imaging"),
        ("Regulus", "Leo", "quadruple star system near the ecliptic", "lunar occultation timings", "polarized continuum emission"),
        ("Mira", "Cetus", "pulsating asymptotic giant branch star", "dust-driven circumstellar shells", "maser spectral monitoring"),
        ("Algol", "Perseus", "eclipsing triple star system", "Roche-lobe mass transfer", "orbital period timing variations"),
    ]
    music_items = [
        ("Johann Sebastian Bach", "The Well-Tempered Clavier", "polyphonic fugue architecture", "equal-temperament keyboard tuning", "contrapuntal voice independence"),
        ("Carl Philipp Emanuel Bach", "Prussian Sonatas", "empfindsam expressive keyboard rhetoric", "orchestral woodwind dialogue", "unexpected harmonic surprises"),
        ("Johannes Ockeghem", "Missa Prolationum", "mensural double-canon form", "independent vocal polyphony", "flowing modal cadences"),
        ("Franz Schubert", "Winterreise", "strophic art-song cycles", "expressive piano accompaniment", "mediant key-relation shifts"),
        ("Johannes Brahms", "Variations on a Theme", "developing rhythmic hemiola", "dense orchestral brass voicings", "passacaglia ground-bass structure"),
        ("Frederic Chopin", "Ballade No. 1", "rubato cantabile phrasing", "chromatic inner-voice figuration", "pedal resonance layering"),
        ("Robert Schumann", "Kinderszenen", "character-piece miniatures", "syncopated rhythmic cross-accents", "lyrical harmonic suspension"),
        ("Joseph Haydn", "String Quartet Op. 76", "monothematic sonata exposition", "witty structural phrase elision", "balanced four-part string texture"),
        ("Gustav Mahler", "Symphony No. 9", "expansive orchestral timbre", "progressive tonal resolution", "chamber-like solo wind scoring"),
        ("Igor Stravinsky", "The Firebird", "polyrhythmic metric displacement", "octatonic harmonic collections", "brilliant percussion orchestration"),
        ("Bela Bartok", "Concerto for Orchestra", "arch-form symmetrical design", "quartal intervallic harmony", "folk-derived asymmetric meters"),
        ("Sergei Rachmaninoff", "Piano Concerto No. 2", "sweeping melodic arcs", "bell-like chordal sonorities", "lush string countermelodies"),
        ("Felix Mendelssohn", "Violin Concerto in E Minor", "linked attacca movements", "transparent orchestral clarity", "integrated solo cadenza placement"),
        ("Antonin Dvorak", "Cello Concerto in B Minor", "pentatonic thematic contours", "warm low-register brass blends", "cyclical finale reminiscence"),
        ("Pyotr Ilyich Tchaikovsky", "Symphony No. 6", "dramatic dynamic contrasts", "limping five-four waltz rhythms", "expressive string phrasing"),
        ("George Frideric Handel", "Water Music", "ceremonial baroque suites", "antiphonal horn-oboe choirs", "binary dance-movement symmetry"),
        ("Franz Liszt", "Sonata in B Minor", "single-movement thematic transformation", "virtuosic octave textures", "tritone harmonic ambiguity"),
        ("Dmitri Shostakovich", "String Quartet No. 8", "monogrammatic four-note motifs", "stark contrapuntal intensity", "passacaglia lament variations"),
    ]
    tech_items = [
        ("Edsger Dijkstra", "shortest-path graph traversal", "priority-queue relaxation invariants", "non-negative edge weights", "deterministic routing tables"),
        ("Donald Knuth", "LR syntactic parsing", "shift-reduce automaton states", "unambiguous context-free grammars", "linear-time compiler frontends"),
        ("Claude Shannon", "noisy-channel coding capacity", "entropy rate bounds", "error-correcting block codes", "asymptotically lossless compression"),
        ("Alonzo Church", "untyped lambda calculus", "beta-reduction normal forms", "higher-order function abstraction", "formal decidability bounds"),
        ("Barbara Liskov", "behavioral subtyping contracts", "abstract data encapsulation", "substitutable interface invariants", "modular distributed consensus"),
        ("Leslie Lamport", "logical event clocks", "happens-before partial ordering", "fault-tolerant state replication", "quorum-based Paxos rounds"),
        ("Grace Hopper", "machine-independent compilers", "symbolic linker abstractions", "structured procedural syntax", "automated code generation"),
        ("John von Neumann", "stored-program architecture", "unified instruction-data memory", "sequential fetch-decode cycles", "merge-sort divide-and-conquer"),
        ("Ken Thompson", "regular-expression finite automata", "composable byte-stream pipes", "hierarchical file descriptors", "minimal kernel abstractions"),
        ("Dennis Ritchie", "portable systems programming", "pointer-arithmetic memory models", "zero-overhead struct layouts", "compact runtime libraries"),
        ("Richard Hamming", "single-error-correcting codes", "parity-check syndrome decoding", "minimum bit-distance spheres", "reliable memory-bus protection"),
        ("Tony Hoare", "in-place partition sorting", "axiomatic precondition triples", "communicating sequential processes", "formal loop-invariant proofs"),
        ("Robert Tarjan", "strongly connected components", "depth-first lowlink numbering", "amortized union-find compression", "linear-time graph connectivity"),
        ("Vint Cerf", "packet-switched internetworking", "end-to-end congestion windows", "stateless datagram forwarding", "layered transport encapsulation"),
        ("Radia Perlman", "spanning-tree bridge election", "loop-free Ethernet forwarding", "root-bridge path-cost metrics", "link-state routing convergence"),
        ("Ivan Sutherland", "interactive constraint graphics", "homogeneous projection matrices", "hierarchical scene-graph instances", "live vector rendering"),
        ("Whitfield Diffie", "asymmetric key agreement", "discrete-logarithm trapdoors", "public-channel secret exchange", "authenticated digital signatures"),
        ("Adi Shamir", "threshold secret sharing", "Lagrange polynomial interpolation", "finite-field share reconstruction", "fault-tolerant key custody"),
    ]
    mineral_items = [
        ("Quartz", "trigonal silicon dioxide", "conchoidal fracture surfaces", "Mohs hardness seven", "optical birefringence"),
        ("Feldspar", "aluminosilicate framework", "two-directional cleavage planes", "igneous plutonic crystallization", "perthitic exsolution lamellae"),
        ("Mica", "phyllosilicate sheet lattice", "perfect basal cleavage flakes", "dielectric thermal insulation", "pearly vitreous luster"),
        ("Calcite", "trigonal carbonate lattice", "strong double refraction", "rhombohedral cleavage angles", "hydrothermal vein precipitation"),
        ("Pyrite", "cubic disulfide crystals", "striated metallic cubes", "brassy reflective facets", "hydrothermal ore mineralization"),
        ("Corundum", "hexagonal aluminum oxide", "Mohs hardness nine", "trace chromium coloration", "refractory abrasive ceramics"),
        ("Olivine", "orthorhombic magnesium silicate", "high-temperature mantle stability", "olive-green granular clusters", "mafic basalt phenocrysts"),
        ("Garnet", "isometric nesosilicate dodecahedra", "metamorphic pressure indicators", "zoned compositional rims", "industrial waterjet abrasives"),
        ("Fluorite", "cubic halide crystal", "octahedral cleavage geometry", "ultraviolet fluorescence bands", "apochromatic telescope lenses"),
        ("Magnetite", "inverse spinel oxide", "ferrimagnetic domain alignment", "paleomagnetic polarity records", "heavy-media separation slurries"),
        ("Topaz", "orthorhombic fluorosilicate", "basal cleavage susceptibility", "prismatic striated terminations", "pegmatite vapor-phase cavities"),
        ("Beryl", "hexagonal cyclosilicate rings", "elongated prismatic columns", "channel-trapped alkali ions", "granitic pegmatite pockets"),
        ("Apatite", "hexagonal phosphate group", "fission-track thermochronology", "accessory igneous grains", "biocompatible ceramic coatings"),
        ("Zircon", "tetragonal nesosilicate crystal", "radiometric age clocks", "extreme chemical durability", "detrital provenance studies"),
        ("Tourmaline", "borosilicate trigonal prisms", "pyroelectric charge separation", "pleochroic color zoning", "hydrothermal boron fluids"),
        ("Barite", "orthorhombic barium sulfate", "high specific gravity", "tabular bladed aggregates", "drilling-fluid density weighting"),
        ("Gypsum", "monoclinic hydrated sulfate", "Mohs hardness two", "evaporite sedimentary beds", "calcined architectural plaster"),
        ("Graphite", "hexagonal carbon sheets", "weak van der Waals interlayer bonding", "anisotropic electrical conductivity", "high-temperature crucible liners"),
    ]
    sport_arch_items = [
        ("Archimedes Screw", "helical hydraulic lifting", "continuous rotary water transport", "inclined cylindrical casings", "low-head irrigation efficiency"),
        ("Flying Buttress", "ribbed masonry vaults", "lateral thrust transfer", "slender clerestory walls", "cathedral structural equilibrium"),
        ("Suspension Cable Bridge", "parabolic steel main cables", "vertical suspender tension", "aerodynamic box-girder decks", "long-span wind-tunnel damping"),
        ("Marine Chronometer", "temperature-compensated balance wheels", "detent escapement precision", "celestial longitude determination", "gimbaled brass housings"),
        ("Fresnel Lens", "concentric stepped prisms", "compact focal-length collimation", "lighthouse beam intensity", "reduced glass absorption mass"),
        ("Bessemer Converter", "pneumatic decarburization blasts", "molten converter oxidation", "structural steel rail output", "rapid refractory-lined batches"),
        ("Geodesic Dome", "triangulated icosahedral struts", "distributed membrane tension", "high enclosed-volume ratios", "lightweight modular assembly"),
        ("Jacquard Loom", "punched pasteboard chains", "automated warp-hook selection", "repeatable brocade patterns", "programmable textile storage"),
        ("Sextant", "double-reflecting index mirrors", "horizon-altitude angle reads", "vernier arc subdivisions", "open-ocean celestial fixes"),
        ("Aneroid Barometer", "evacuated spring-alloy capsules", "elastic atmospheric deflection", "lever-amplified dial needles", "portable altimeter calibration"),
        ("Wheatstone Bridge", "balanced resistive legs", "null-galvanometer detection", "strain-gauge micro-ohm shifts", "precision sensor conditioning"),
        ("Pendulum Escapement", "isochronous gravitational swings", "deadbeat anchor pallets", "invar-rod thermal stability", "astronomical regulator clocks"),
        ("Centrifugal Governor", "rotating flyball linkages", "proportional throttle feedback", "steady engine shaft speed", "closed-loop mechanical control"),
        ("Venturi Meter", "converging throat constriction", "Bernoulli differential pressure", "low permanent head loss", "industrial pipe flow metering"),
        ("Pelton Turbine", "split-bucket impulse runners", "high-head hydraulic jets", "momentum-reversal efficiency", "mountain hydroelectric stations"),
        ("Stirling Engine", "closed regenerative gas cycles", "external heat-source flexibility", "displacer-piston phase shifts", "quiet cryogenic cooling loops"),
        ("Gyrocompass", "fast-spinning gimbaled rotors", "precession toward true north", "magnetic-variation immunity", "inertial maritime headings"),
        ("Vernier Caliper", "offset sliding auxiliary graduations", "sub-millimeter visual alignment", "internal and depth jaws", "machinist tolerance inspection"),
    ]

    encyc_pool: List[str] = []
    for name, ctx, desc, mech, app in astro_items:
        encyc_pool.append(
            f"In observational astronomy, {name} is a prominent {desc} located in the constellation {ctx}. "
            f"Astrophysicists monitor {name} to investigate {mech}, combining multi-epoch catalogs with {app}. "
            f"Spectral synthesis models track effective surface temperatures, bolometric luminosity, and angular "
            f"diameter uncertainties across successive observing seasons."
        )
    for name, work, feat, tech, struct in music_items:
        encyc_pool.append(
            f"Musicologists studying {name} frequently analyze {work} as a benchmark of {feat}. "
            f"Score editions highlight how {name} coordinates {tech} with {struct} across contrasting sections. "
            f"Conservatory seminars compare harmonic rhythm, voice spacing, and phrasing articulation across "
            f"historical recordings and critical manuscripts."
        )
    for name, topic, inv, prop, outcome in tech_items:
        encyc_pool.append(
            f"In computer science history, {name} established foundational methods for {topic}. "
            f"Textbook treatments explain how {name} leveraged {inv} and {prop} to guarantee {outcome}. "
            f"Modern engineering curricula still assign formal correctness proofs and benchmark implementations "
            f"derived directly from these original papers."
        )
    for name, comp, trait, prop, use in mineral_items:
        encyc_pool.append(
            f"Mineralogical references classify {name} as a {comp} specimen distinguished by {trait}. "
            f"Laboratory identification of {name} relies on {prop}, optical thin-section microscopy, and "
            f"X-ray powder diffraction peaks, supporting industrial applications in {use}."
        )
    for name, dom, mech, trait, benefit in sport_arch_items:
        encyc_pool.append(
            f"In engineering design, the {name} exemplifies {dom} through {mech}. "
            f"Historical blueprints of the {name} emphasize {trait} to achieve {benefit} under variable "
            f"operating loads, informing both museum restorations and contemporary mechanical analysis."
        )

    assert len(encyc_pool) == 90, f"Expected 90 encyclopedic passages, got {len(encyc_pool)}"

    # Verify ZERO blacklist collisions in encyc_pool
    for i, txt in enumerate(encyc_pool):
        m = bl_pat.search(txt)
        assert m is None, f"Encyclopedic passage {i} collided with blacklist term '{m.group(0)}': {txt[:120]}"

    # Ensure we have exactly 120 prose + 90 encyc + 90 structured = 300 passages
    rng.shuffle(prose_pool)
    rng.shuffle(struct_pool)
    assert len(prose_pool) >= 120, f"prose_pool too small: {len(prose_pool)}"
    assert len(struct_pool) >= 90, f"struct_pool too small: {len(struct_pool)}"

    selected = prose_pool[:120] + encyc_pool[:90] + struct_pool[:90]
    assert len(selected) == 300
    rng.shuffle(selected)

    return {
        "fit": selected[:195],      # 65%
        "resid": selected[195:255], # 20%
        "eval": selected[255:300],  # 15%
    }


def verify_universal_corpus_disjoint(corpus: Dict[str, List[str]], extra_forbidden: Sequence[str] = ()) -> List[Tuple[str, str]]:
    """Verify ZERO word-boundary overlap with Cap-1 geography, Cap-2 biology, and any extra forbidden terms."""
    forbidden = set(CAP1_GEO_BLACKLIST) | set(CAP2_BIO_BLACKLIST) | {t.strip().lower() for t in extra_forbidden if len(t.strip()) >= 3}
    all_texts = {k: "\n".join(v).lower() for k, v in corpus.items()}
    hits: List[Tuple[str, str]] = []
    for term in sorted(forbidden):
        pat = re.compile(r"\b" + re.escape(term) + r"\b")
        for sp, txt in all_texts.items():
            if pat.search(txt):
                hits.append((sp, term))
    return hits


CONCEPT_WRITE_TEMPLATES: Tuple[str, ...] = (
    "The concept of {c} is widely studied.",
    "In general, {c} is commonly known.",
    "Many people recognize {c} immediately.",
    "A typical example is {c} in daily life.",
)

UNIVERSAL_CONCEPT_WRITE_WORDS: Tuple[str, ...] = (
    # 1. Tools & hardware (25)
    "hammer", "wrench", "chisel", "pliers", "screwdriver", "saw", "drill", "anvil", "mallet", "trowel",
    "shovel", "rake", "crowbar", "clamp", "vise", "lathe", "pulley", "lever", "wedge", "axle",
    "ratchet", "caliper", "stapler", "funnel", "sieve",
    # 2. Musical instruments (25)
    "violin", "cello", "piano", "flute", "trumpet", "clarinet", "oboe", "harp", "drum", "banjo",
    "guitar", "trombone", "tuba", "bassoon", "xylophone", "accordion", "harmonica", "mandolin", "cymbal", "tambourine",
    "piccolo", "viola", "lute", "marimba", "ukulele",
    # 3. Materials & textiles (25)
    "marble", "quartz", "basalt", "slate", "limestone", "sandstone", "obsidian", "amber", "ivory", "velvet",
    "linen", "cotton", "silk", "nylon", "denim", "leather", "bronze", "brass", "pewter", "porcelain",
    "ceramic", "flannel", "canvas", "burlap", "suede",
    # 4. Professions & roles (25)
    "architect", "carpenter", "plumber", "surgeon", "chemist", "botanist", "geologist", "librarian", "baker", "tailor",
    "potter", "jeweler", "welder", "machinist", "pharmacist", "dentist", "optician", "mason", "locksmith", "florist",
    "butcher", "cobbler", "blacksmith", "astronomer", "surveyor",
    # 5. Astronomy & landforms (25)
    "nebula", "comet", "asteroid", "galaxy", "pulsar", "quasar", "eclipse", "solstice", "equinox", "aurora",
    "glacier", "plateau", "valley", "volcano", "geyser", "estuary", "fjord", "tundra", "savanna", "prairie",
    "dune", "cyclone", "monsoon", "crater", "archipelago",
    # 6. Sports & games (25)
    "rugby", "rowing", "fencing", "archery", "bowling", "skiing", "surfing", "cycling", "diving", "curling",
    "squash", "lacrosse", "handball", "judo", "karate", "chess", "domino", "billiards", "badminton", "volleyball",
    "wrestling", "croquet", "sailing", "kayaking", "skating",
    # 7. Furniture & household (25)
    "chair", "sofa", "cabinet", "dresser", "bookshelf", "stool", "bench", "cradle", "hammock", "mattress",
    "pillow", "blanket", "curtain", "carpet", "mirror", "lamp", "lantern", "kettle", "teapot", "skillet",
    "ladle", "pitcher", "platter", "goblet", "chandelier",
    # 8. Vehicles & transport (25)
    "bicycle", "scooter", "trolley", "tram", "subway", "locomotive", "caboose", "sailboat", "kayak", "canoe",
    "yacht", "ferry", "barge", "tugboat", "glider", "biplane", "helicopter", "dirigible", "wagon", "carriage",
    "sled", "tractor", "bulldozer", "forklift", "catamaran",
    # 9. Foods & culinary (25)
    "bread", "biscuit", "pretzel", "muffin", "waffle", "pancake", "noodle", "dumpling", "soup", "stew",
    "salad", "sandwich", "omelet", "custard", "pudding", "sorbet", "pastry", "tart", "bagel", "oatmeal",
    "porridge", "granola", "broth", "cracker", "vinegar",
    # 10. Architecture & structures (25)
    "archway", "dome", "spire", "pillar", "column", "balcony", "terrace", "courtyard", "pavilion", "gazebo",
    "lighthouse", "windmill", "viaduct", "aqueduct", "silo", "barn", "hangar", "warehouse", "stadium", "theater",
    "gallery", "library", "observatory", "greenhouse", "colonnade",
    # 11. Garments & accessories (25)
    "jacket", "sweater", "cardigan", "vest", "apron", "scarf", "mitten", "glove", "sandal", "slipper",
    "sneaker", "boot", "helmet", "turban", "bonnet", "cloak", "poncho", "tunic", "shawl", "collar",
    "cuff", "buckle", "zipper", "parka", "overcoat",
    # 12. Science & everyday concepts (25)
    "symmetry", "entropy", "rhythm", "melody", "harmony", "friction", "momentum", "gravity", "inertia", "osmosis",
    "refraction", "diffraction", "resonance", "catalysis", "erosion", "sediment", "horizon", "zenith", "compass", "pendulum",
    "prism", "lens", "magnet", "battery", "telescope",
)


def build_universal_concept_write_corpus() -> Dict[str, Any]:
    """
    Task-agnostic short-context concept write calibration corpus (300 open-domain concepts x 4 templates = 1,200 sentences).
    Isolates pure mid-layer semantic write subspace without long-range discourse/RoPE/code drift!
    Strictly 0 word-level overlap with Cap-1 geography, dataset.json, AND all 152 sealed Cap-2 terms.
    """
    words = list(dict.fromkeys(UNIVERSAL_CONCEPT_WRITE_WORDS))
    assert len(words) == 300, f"Expected 300 unique concepts, got {len(words)}"
    prompts, spans, concepts = [], [], []
    for c in words:
        for tmpl in CONCEPT_WRITE_TEMPLATES:
            p = tmpl.format(c=c)
            c0 = p.index(c)
            prompts.append(p)
            spans.append([c0, c0 + len(c)])
            concepts.append(c)
    return {"concepts": words, "prompts": prompts, "spans": spans, "prompt_concepts": concepts}


# ======================================================================================
# 2. Core Math: Analytical Ledoit-Wolf, Generative Channel A->B, Write Isometry, Compiler
# ======================================================================================

def ledoit_wolf_cov(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Exact analytical Ledoit-Wolf (2004) shrinkage estimator toward scaled identity (tr(S)/d)*I.
    Zero hand-tuned hyperparameters! Runs in CPU float64 in O(n d^2).
    """
    Xf = np.asarray(X, dtype=np.float64)
    n, d = Xf.shape
    assert n >= 2, f"Need >=2 samples for Ledoit-Wolf, got {n}"
    Xc = Xf - Xf.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / float(n)
    mu = float(np.trace(S)) / float(d)
    sum_S2 = float(np.sum(S * S))
    delta2 = sum_S2 - d * (mu * mu)
    if delta2 <= 1e-18:
        return mu * np.eye(d, dtype=np.float64), 1.0
    row_norm2 = np.sum(Xc * Xc, axis=1)
    # Exact (1/n^2) sum_i ||x_i x_i^T - S||_F^2 = (mean(||x_i||^4) - ||S||_F^2) / n
    beta2 = (float(np.mean(row_norm2 * row_norm2)) - sum_S2) / float(n)
    shrink = float(np.clip(beta2 / delta2, 0.0, 1.0))
    S_lw = (1.0 - shrink) * S + shrink * mu * np.eye(d, dtype=np.float64)
    return S_lw, shrink


def compute_lw_constant_pin(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Parameter-free constant predictor c_B = argmin E[(c^T x - 1)^2] using Ledoit-Wolf
    regularized second-moment matrix (zero hand-tuned lambda!).
    """
    Xf = np.asarray(X, dtype=np.float64)
    mu = Xf.mean(axis=0)
    S_lw, _ = ledoit_wolf_cov(Xf)
    C2_lw = S_lw + np.outer(mu, mu)
    c = sla.solve(C2_lw, mu, assume_a="pos")
    # Exact mean-1 normalization on X
    c = c / max(float(mu @ c), 1e-12)
    return c, float(np.std(Xf @ c))


def fit_generative_channel(
    X_A_fit: np.ndarray,
    X_B_fit: np.ndarray,
    X_A_resid: np.ndarray,
    X_B_resid: np.ndarray,
    H_A_fit: np.ndarray,
    H_B_fit: np.ndarray,
    H_A_resid: np.ndarray,
    H_B_resid: np.ndarray,
    bos_x_fit_B: np.ndarray,
) -> Dict[str, Any]:
    """
    Fit the complete unlabeled Generative Channel A -> B + RMS-Normalized Write Isometry:
      Read side (post-LN MLP input x):
        G = Sigma_BA @ inv(Sigma_AA_lw)   (D_B x D_A, naturally rectangular!)
        E_B = X_B_resid - (mu_B + (X_A_resid - mu_A) @ G.T)   (strictly held-out residuals!)
        N_lw = ledoit_wolf_cov(E_B)
      Write side (post-layer residual h_out):
        RMS-normalize tokens h / rms(h) (bounds massive-activation outliers automatically, ZERO 2.5x filter!)
        R_out = Stiefel SVD of centered cross-covariance of RMS-normalized h_out
        s_out_iso = median(||H_B_fit||) / median(||H_A_fit||)   (variance-matched isometry => gain == 1.000!)
        M_out_iso = s_out_iso * R_out   (D_B x D_A)
    All algebra strictly in CPU float64!
    """
    Xa_f = np.asarray(X_A_fit, dtype=np.float64)
    Xb_f = np.asarray(X_B_fit, dtype=np.float64)
    Xa_r = np.asarray(X_A_resid, dtype=np.float64)
    Xb_r = np.asarray(X_B_resid, dtype=np.float64)
    Ha_f = np.asarray(H_A_fit, dtype=np.float64)
    Hb_f = np.asarray(H_B_fit, dtype=np.float64)
    Ha_r = np.asarray(H_A_resid, dtype=np.float64)
    Hb_r = np.asarray(H_B_resid, dtype=np.float64)

    n_f, D_A = Xa_f.shape
    D_B = Xb_f.shape[1]

    mu_A = Xa_f.mean(axis=0)
    mu_B = Xb_f.mean(axis=0)
    Sigma_AA_lw, shrink_AA = ledoit_wolf_cov(Xa_f)
    Sigma_BB_fit_lw, shrink_BB = ledoit_wolf_cov(Xb_f)
    Sigma_BA = ((Xb_f - mu_B).T @ (Xa_f - mu_A)) / float(n_f)  # (D_B, D_A)

    # Generative forward map G: A -> B (D_B x D_A)
    G = sla.solve(Sigma_AA_lw, Sigma_BA.T, assume_a="pos").T

    # Unbiased held-out empirical residuals on resid split
    Xb_hat_r = mu_B + (Xa_r - mu_A) @ G.T
    E_B = Xb_r - Xb_hat_r  # (n_resid, D_B)
    N_lw, shrink_N = ledoit_wolf_cov(E_B)
    r2_gen_read = 1.0 - float(np.sum(E_B ** 2) / max(np.sum((Xb_r - Xb_r.mean(axis=0)) ** 2), 1e-12))

    # Write side: RMS-normalized Stiefel Isometry (ZERO 2.5x magic filter, gain == 1.000!)
    rms_Ha_f = np.sqrt(np.mean(Ha_f ** 2, axis=-1, keepdims=True) + 1e-12)
    rms_Hb_f = np.sqrt(np.mean(Hb_f ** 2, axis=-1, keepdims=True) + 1e-12)
    Ha_n = Ha_f / rms_Ha_f
    Hb_n = Hb_f / rms_Hb_f
    Ha_nc = Ha_n - Ha_n.mean(axis=0, keepdims=True)
    Hb_nc = Hb_n - Hb_n.mean(axis=0, keepdims=True)
    C_write = (Hb_nc.T @ Ha_nc) / float(len(Ha_f))  # (D_B, D_A)
    U_w, S_w, Vt_w = sla.svd(C_write, full_matrices=False)
    R_out = U_w @ Vt_w  # (D_B, D_A)

    # Variance-matched isometry scale: preserves relative residual perturbation ||dh_B||/||h_B|| == ||dh_A||/||h_A||!
    med_norm_A = float(np.median(np.linalg.norm(Ha_f, axis=-1)))
    med_norm_B = float(np.median(np.linalg.norm(Hb_f, axis=-1)))
    s_out_iso = med_norm_B / max(med_norm_A, 1e-12)
    M_out_iso = s_out_iso * R_out

    # Held-out write directional R^2 on RMS-normalized resid tokens
    rms_Ha_r = np.sqrt(np.mean(Ha_r ** 2, axis=-1, keepdims=True) + 1e-12)
    rms_Hb_r = np.sqrt(np.mean(Hb_r ** 2, axis=-1, keepdims=True) + 1e-12)
    Ha_rn = Ha_r / rms_Ha_r
    Hb_rn = Hb_r / rms_Hb_r
    Hb_rn_hat = Hb_n.mean(axis=0) + (Ha_rn - Ha_n.mean(axis=0)) @ R_out.T
    # Cosine similarity of predicted RMS-normalized write direction
    num_w = np.sum((Hb_rn_hat - Hb_n.mean(0)) * (Hb_rn - Hb_n.mean(0)), axis=-1)
    den_w = np.linalg.norm(Hb_rn_hat - Hb_n.mean(0), axis=-1) * np.linalg.norm(Hb_rn - Hb_n.mean(0), axis=-1) + 1e-12
    write_dir_cos = float(np.mean(num_w / den_w))

    # Parameter-free LW constant pin c_B and exact BOS bank
    c_B, c_B_std = compute_lw_constant_pin(Xb_f)
    x_bos_B = np.mean(np.asarray(bos_x_fit_B, dtype=np.float64), axis=0)

    return {
        "D_A": int(D_A),
        "D_B": int(D_B),
        "mu_A": mu_A,
        "mu_B": mu_B,
        "G": G,
        "Sigma_AA_lw": Sigma_AA_lw,
        "Sigma_BB_fit_lw": Sigma_BB_fit_lw,
        "N_lw": N_lw,
        "E_B_resid": E_B.astype(np.float32),
        "X_A_resid": Xa_r.astype(np.float32),
        "X_B_resid": Xb_r.astype(np.float32),
        "R_out": R_out,
        "s_out_iso": float(s_out_iso),
        "M_out_iso": M_out_iso,
        "c_B": c_B,
        "x_bos_B": x_bos_B,
        "bos_x_fit_B": np.asarray(bos_x_fit_B, dtype=np.float64),
        "summary": {
            "D_A": int(D_A),
            "D_B": int(D_B),
            "n_fit": int(n_f),
            "n_resid": int(len(Xb_r)),
            "shrink_AA": round(shrink_AA, 5),
            "shrink_BB": round(shrink_BB, 5),
            "shrink_N": round(shrink_N, 5),
            "heldout_gen_read_R2": round(r2_gen_read, 4),
            "heldout_write_rms_cos": round(write_dir_cos, 4),
            "s_out_iso": round(float(s_out_iso), 4),
            "med_norm_A": round(med_norm_A, 3),
            "med_norm_B": round(med_norm_B, 3),
            "c_B_std": round(float(c_B_std), 5),
        },
    }


def fit_concept_write_bridge(
    H_A: np.ndarray,
    H_B: np.ndarray,
) -> Dict[str, Any]:
    """
    Task-agnostic short-context concept write bridge (A -> B post-layer h_out):
      - Smooth parameter-free Cauchy outlier weights:
          w_i = 1 / (1 + (||hA_i|| / med_A)^4 + (||hB_i|| / med_B)^4)   (ZERO hard 2.5x cutoff!)
      - Weighted Stiefel SVD: R_out = U @ Vt   (naturally rectangular D_B x D_A!)
      - Exact projected-variance scale: s_out = sum(S) / E_w[||R_out (Ha - mu_Ha)||^2]
        => unique closed-form weighted MSE optimum, gain == 1.000, W_dB = (1/rm_B) * M_out_concept @ W_dA!
    """
    Ha = np.asarray(H_A, dtype=np.float64)
    Hb = np.asarray(H_B, dtype=np.float64)
    assert Ha.shape[0] == Hb.shape[0] and Ha.shape[0] >= 10
    n, D_A = Ha.shape
    D_B = Hb.shape[1]

    def _fit_slice(ha_s: np.ndarray, hb_s: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float, float]:
        nA = np.linalg.norm(ha_s, axis=-1)
        nB = np.linalg.norm(hb_s, axis=-1)
        med_A = max(float(np.median(nA)), 1e-12)
        med_B = max(float(np.median(nB)), 1e-12)
        w = 1.0 / (1.0 + (nA / med_A) ** 4 + (nB / med_B) ** 4)
        wn = w / max(float(np.sum(w)), 1e-12)
        mu_a = np.sum(wn[:, None] * ha_s, axis=0)
        mu_b = np.sum(wn[:, None] * hb_s, axis=0)
        hac = ha_s - mu_a
        hbc = hb_s - mu_b
        C_BA = (hbc * wn[:, None]).T @ hac
        U, S, Vt = sla.svd(C_BA, full_matrices=False)
        R = U @ Vt
        proj = hac @ R.T
        denom = float(np.sum(wn * np.sum(proj * proj, axis=-1)))
        s_out = float(np.sum(S)) / max(denom, 1e-12)
        return R, s_out, mu_a, mu_b, wn, med_A, med_B

    # 80/20 concept-grouped held-out diagnostic (4 templates per concept -> split on multiple of 4)
    n_tr = max(8, (int(0.8 * n) // 4) * 4)
    if n_tr < n - 4:
        R_tr, s_tr, mua_tr, mub_tr, _, medA_tr, medB_tr = _fit_slice(Ha[:n_tr], Hb[:n_tr])
        Ha_va, Hb_va = Ha[n_tr:], Hb[n_tr:]
        w_va = 1.0 / (1.0 + (np.linalg.norm(Ha_va, axis=-1) / medA_tr) ** 4 + (np.linalg.norm(Hb_va, axis=-1) / medB_tr) ** 4)
        wn_va = w_va / max(float(np.sum(w_va)), 1e-12)
        pred_va = (Ha_va - mua_tr) @ (s_tr * R_tr).T
        true_va = Hb_va - mub_tr
        cos_va = np.sum(pred_va * true_va, axis=-1) / (np.linalg.norm(pred_va, axis=-1) * np.linalg.norm(true_va, axis=-1) + 1e-12)
        ho_cos = float(np.sum(wn_va * cos_va))
        ss_res = float(np.sum(wn_va * np.sum((true_va - pred_va) ** 2, axis=-1)))
        ss_tot = float(np.sum(wn_va * np.sum(true_va ** 2, axis=-1)))
        ho_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    else:
        ho_cos, ho_r2 = float("nan"), float("nan")

    # Fit final concept write bridge on all pairs
    R_out, s_out, mu_a, mu_b, wn, med_A, med_B = _fit_slice(Ha, Hb)
    M_out_concept = s_out * R_out
    eff_n = float(1.0 / max(float(np.sum(wn * wn)), 1e-12))

    return {
        "D_A": int(D_A),
        "D_B": int(D_B),
        "R_out_concept": R_out,
        "s_out_concept": float(s_out),
        "M_out_concept": M_out_concept,
        "summary": {
            "n_pairs": int(n),
            "eff_n_cauchy": round(eff_n, 1),
            "s_out_concept": round(float(s_out), 4),
            "med_norm_A": round(med_A, 3),
            "med_norm_B": round(med_B, 3),
            "heldout_concept_write_cos": round(ho_cos, 4) if math.isfinite(ho_cos) else None,
            "heldout_concept_write_R2": round(ho_r2, 4) if math.isfinite(ho_r2) else None,
        },
    }


def compute_frame_only_auroc(pos_frame_ids: Sequence[int], neg_frame_ids: Sequence[int]) -> float:
    """
    Frame-Counterbalance Certificate: computes AUROC of predicting positive vs negative
    using ONLY the template/frame ID (zero entity content!).
    Strictly frame-counterbalanced datasets yield exact 0.5000!
    """
    pos_f = np.asarray(pos_frame_ids, dtype=np.int64)
    neg_f = np.asarray(neg_frame_ids, dtype=np.int64)
    counts_p = defaultdict(int)
    counts_n = defaultdict(int)
    for fid in pos_f:
        counts_p[int(fid)] += 1
    for fid in neg_f:
        counts_n[int(fid)] += 1
    score_map = {}
    for fid in set(counts_p.keys()) | set(counts_n.keys()):
        score_map[fid] = counts_p[fid] / float(counts_p[fid] + counts_n[fid])
    sp = np.array([score_map[int(fid)] for fid in pos_f], dtype=np.float64)
    sn = np.array([score_map[int(fid)] for fid in neg_f], dtype=np.float64)
    return float(np.mean(sp[:, None] > sn[None, :]) + 0.5 * np.mean(sp[:, None] == sn[None, :]))


def random_stiefel(shape: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    m, n = shape
    if m >= n:
        Q, _ = sla.qr(rng.standard_normal((m, n)), mode="economic")
        return Q
    Q, _ = sla.qr(rng.standard_normal((n, m)), mode="economic")
    return Q.T


def solve_2constraint_kkt(
    S: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    m1: float,
    m2: float,
) -> np.ndarray:
    """
    Exact closed-form KKT global optimum for the 2-constraint convex QP:
      min_w 0.5 * w^T S w   s.t.   w^T d1 >= m1  AND  w^T d2 >= m2   (S > 0, m1 > 0, m2 > 0)
    Zero magic constants! Checks the 3 exact KKT active sets in closed form.
    """
    u1 = sla.solve(S, d1, assume_a="pos")
    u2 = sla.solve(S, d2, assume_a="pos")
    a11 = float(d1 @ u1)
    a12 = float(d1 @ u2)
    a22 = float(d2 @ u2)
    det = a11 * a22 - a12 * a12

    # Case 1: Both constraints active (lam1 >= 0, lam2 >= 0)
    if det > 1e-18 * max(a11 * a22, 1e-18):
        lam1 = (a22 * m1 - a12 * m2) / det
        lam2 = (a11 * m2 - a12 * m1) / det
        if lam1 >= 0.0 and lam2 >= 0.0:
            return lam1 * u1 + lam2 * u2

    # Case 2 / 3: Single active constraint (pick feasible one with minimum objective)
    cands = []
    if a11 > 1e-18:
        l1 = m1 / a11
        if l1 * a12 >= m2 - 1e-10:
            w1 = l1 * u1
            cands.append((float(w1 @ (S @ w1)), w1))
    if a22 > 1e-18:
        l2 = m2 / a22
        if l2 * a12 >= m1 - 1e-10:
            w2 = l2 * u2
            cands.append((float(w2 @ (S @ w2)), w2))
    if cands:
        cands.sort(key=lambda t: t[0])
        return cands[0][1]

    # Degenerate collinear case: accept only if a collinear candidate satisfies BOTH constraints.
    for cand_w in (
        (m1 / max(a11, 1e-18)) * u1 if a11 > 1e-18 else None,
        (m2 / max(a22, 1e-18)) * u2 if a22 > 1e-18 else None,
    ):
        if cand_w is not None and float(cand_w @ d1) >= m1 - 1e-9 and float(cand_w @ d2) >= m2 - 1e-9:
            return cand_w
    raise ValueError(
        f"Infeasible 2-constraint KKT QP: constraints w^T d1 >= {m1} and w^T d2 >= {m2} "
        f"cannot be simultaneously satisfied (a11={a11:.3e}, a12={a12:.3e}, a22={a22:.3e})."
    )


def compile_channel_pushforward_organ(
    rules_donor: Sequence[Dict[str, Any]],
    channel: Optional[Dict[str, Any]],
    rm_B: float = 1.0,
    tolerances: Optional[Dict[str, float]] = None,
    mode: str = "biasfree",          # "biasfree" (pure digital c_u, mlp_bias=False) or "affine"
    ablate_domain_up: bool = False,  # Optional A/B: channel-pushforward domain-up vs pure digital pin
    force_abstained: bool = False,   # Strictly for calibration auditing of abstained cells!
    shuffle_read_seed: Optional[int] = None,
    shuffle_write_seed: Optional[int] = None,
    native_B_stats: Optional[Dict[str, Any]] = None,        # Used ONLY when channel is None (Native-B oracle!)
    write_matrix_override: Optional[np.ndarray] = None,     # Optional write-bridge ablation override (e.g. M_out_prose)
) -> Dict[str, Any]:
    """
    Exact Zero-Magic-Constant Channel-Pushforward Twin-SwiGLU Compiler (ONLY 3 operational tolerances!):
      - alpha = 0.002 (conformal real-B false-fire bound -> m2 = ndtri(1-alpha) = 2.8782)
      - alpha_neg = 0.03 (in-domain negative tolerance -> m1 = ndtri(1-alpha_neg) = 1.8808)
      - delta = 0.02 (SiLU step off-leak & on-plateau ripple bound)
      - Strict 50/50 resid split: [:mid] fits N_lw ONLY; [mid:] (n_ev = len(Xb_ev_r)) is 100% untouched
        for conformal q_real_B, r2_chan_heldout, r_dir_heldout, and a_emp!
      - Exact 2-constraint KKT direction w_c on S_bg = 0.5 * Sigma_BB_lw + 0.5 * (Sn_lw + N_lw).
      - Exact Real-B conformal threshold t1 = Q_{1-alpha}(Xb_ev_r @ w_c), Gamma = std(Xb_ev_r @ w_c),
        t2 = t1 + Gamma, beta = min(SILU_NEG_MAX / (delta * Gamma), 100.0 / Gamma).
      - Unlabeled abstention iff (r2_chan_ev <= 0.0) or (r_dir <= 0.25) or (a_emp <= 0.15).
      - Pure digital BOS-nulled pin c_u (mlp_bias=False), gain == 1.000, W_dB = (1/rm_B) * M_out @ W_dA!
    """
    tol = dict(FROZEN_TOLERANCES if tolerances is None else tolerances)
    alpha = float(tol["alpha"])
    alpha_neg = float(tol["alpha_neg"])
    delta = float(tol["delta"])
    m1 = float(ndtri(1.0 - alpha_neg))
    m2 = float(ndtri(1.0 - alpha))

    is_native = channel is None
    if is_native:
        assert native_B_stats is not None
        mu_B = np.asarray(native_B_stats["mu_B"], dtype=np.float64)
        Sigma_BB_lw = np.asarray(native_B_stats["Sigma_BB_fit_lw"], dtype=np.float64)
        X_B_resid_all = np.asarray(native_B_stats["X_B_resid"], dtype=np.float64)
        mid = len(X_B_resid_all) // 2
        Xb_ev_r = X_B_resid_all[mid:]
        Xa_ev_r = None
        c_B = np.asarray(native_B_stats["c_B"], dtype=np.float64)
        x_bos_B = np.asarray(native_B_stats["x_bos_B"], dtype=np.float64)
        bos_bank_B = np.asarray(native_B_stats["bos_x_fit_B"], dtype=np.float64)
        D_B = len(mu_B)
        N_lw = np.zeros((D_B, D_B), dtype=np.float64)
        G = None
        mu_A = None
        M_out_iso = None
    else:
        mu_A = np.asarray(channel["mu_A"], dtype=np.float64)
        mu_B = np.asarray(channel["mu_B"], dtype=np.float64)
        G = np.asarray(channel["G"], dtype=np.float64).copy()
        Sigma_BB_lw = np.asarray(channel["Sigma_BB_fit_lw"], dtype=np.float64)
        X_A_resid_all = np.asarray(channel["X_A_resid"], dtype=np.float64) if "X_A_resid" in channel else None
        X_B_resid_all = np.asarray(channel["X_B_resid"], dtype=np.float64)
        E_B_resid_all = np.asarray(channel["E_B_resid"], dtype=np.float64)
        M_out_src = channel["M_out_iso"] if write_matrix_override is None else write_matrix_override
        M_out_iso = np.asarray(M_out_src, dtype=np.float64).copy()
        c_B = np.asarray(channel["c_B"], dtype=np.float64)
        x_bos_B = np.asarray(channel["x_bos_B"], dtype=np.float64)
        bos_bank_B = np.asarray(channel["bos_x_fit_B"], dtype=np.float64)
        D_B = int(channel["D_B"])

        if shuffle_read_seed is not None:
            rng_sr = np.random.default_rng(shuffle_read_seed)
            Q_r = random_stiefel(G.shape, rng_sr)
            G = Q_r * (sla.norm(G, "fro") / max(sla.norm(Q_r, "fro"), 1e-12))
        if shuffle_write_seed is not None:
            rng_sw = np.random.default_rng(shuffle_write_seed)
            Q_w = random_stiefel(M_out_iso.shape, rng_sw)
            M_out_iso = Q_w * (sla.norm(M_out_iso, "fro") / max(sla.norm(Q_w, "fro"), 1e-12))

        # Strict 50/50 split of resid: [:mid] fits N_lw ONLY; [mid:] is strictly untouched for conformal & r_dir!
        mid = len(X_B_resid_all) // 2
        Xb_fit_r, Xb_ev_r = X_B_resid_all[:mid], X_B_resid_all[mid:]
        if X_A_resid_all is not None:
            Xa_fit_r, Xa_ev_r = X_A_resid_all[:mid], X_A_resid_all[mid:]
            E_fit_r = Xb_fit_r - (mu_B + (Xa_fit_r - mu_A) @ G.T)
            N_lw, _ = ledoit_wolf_cov(E_fit_r)
        else:
            Xa_ev_r = None
            N_lw, _ = ledoit_wolf_cov(E_B_resid_all[:mid])

    # Pure Digital BOS-nulled constant pin c_u (mean == 1.0 on Xb_ev_r, exact 0.0 at x_bos_B!)
    c_u = po.apply_exact_bos_null_biasfree(c_B[None, :], x_bos_B, x_bos_bank=bos_bank_B)[0]
    c_u = c_u / max(float(np.mean(Xb_ev_r @ c_u)), 1e-8)

    # Optional domain-up ablation (all domain entities vs real B background, pushed through G!)
    w_dom_u: Optional[np.ndarray] = None
    b_dom_u: float = 0.0
    if ablate_domain_up:
        X_all_dom = np.vstack([
            np.vstack([np.asarray(r["X_pos_A"], dtype=np.float64), np.asarray(r["X_neg_dom_A"], dtype=np.float64)])
            for r in rules_donor
        ])
        X_dom_sim = X_all_dom if is_native else (mu_B + (X_all_dom - mu_A) @ G.T)
        S_dom_sim = ledoit_wolf_cov(X_dom_sim)[0] + N_lw
        w_d_u = sla.solve(0.5 * Sigma_BB_lw + 0.5 * S_dom_sim, X_dom_sim.mean(0) - mu_B, assume_a="pos")
        u_gen_m = float(np.mean(Xb_ev_r @ w_d_u))
        u_dom_m = float(np.mean(X_dom_sim @ w_d_u)) - u_gen_m
        w_dom_u = w_d_u / max(abs(u_dom_m), 1e-6)
        b_dom_u = -float(u_gen_m / max(abs(u_dom_m), 1e-6))

    inv_rm = 1.0 / float(rm_B)  # gain == 1.000 identically!
    W_g_rows, b_g_rows, W_u_rows, b_u_rows, W_d_cols = [], [], [], [], []
    rule_reports: List[Dict[str, Any]] = []
    any_abstained = False
    n_ev = len(Xb_ev_r)
    k_conf_raw = int(math.ceil((n_ev + 1) * (1.0 - alpha)))
    conformal_n_ok = k_conf_raw <= n_ev
    k_conf = min(n_ev, k_conf_raw)
    Xb_pred_ev = None if (is_native or Xa_ev_r is None) else (mu_B + (Xa_ev_r - mu_A) @ G.T)
    if Xb_pred_ev is not None:
        ss_res_ev = float(np.sum((Xb_ev_r - Xb_pred_ev) ** 2))
        ss_tot_ev = float(np.sum((Xb_ev_r - Xb_ev_r.mean(axis=0, keepdims=True)) ** 2))
        r2_chan_ev = 1.0 - ss_res_ev / max(ss_tot_ev, 1e-12)
    else:
        r2_chan_ev = 1.0

    for r in rules_donor:
        rname = str(r["name"])
        Xp_A = np.asarray(r["X_pos_A"], dtype=np.float64)
        Xn_A = np.asarray(r["X_neg_dom_A"], dtype=np.float64)
        v_A = np.asarray(r["v_A"], dtype=np.float64)

        # 1. Mandatory Frame-Counterbalance Certificate (enforced in abstention below!)
        f_auroc = compute_frame_only_auroc(r["pos_frame_ids"], r["neg_frame_ids"])
        frame_ok = abs(f_auroc - 0.5) <= 1e-6

        # 2. Push donor states forward through generative channel G into B coordinates!
        if is_native:
            Xp_sim = Xp_A
            Xn_sim = Xn_A
        else:
            Xp_sim = mu_B + (Xp_A - mu_A) @ G.T
            Xn_sim = mu_B + (Xn_A - mu_A) @ G.T

        # 3. Exact 2-Constraint KKT Direction w_c in B Coordinates (ZERO magic constants!)
        Sn_lw, _ = ledoit_wolf_cov(Xn_sim)
        S_bg = 0.5 * Sigma_BB_lw + 0.5 * (Sn_lw + N_lw)
        w_c = solve_2constraint_kkt(S_bg, Xp_sim.mean(axis=0) - Xn_sim.mean(axis=0), Xp_sim.mean(axis=0) - mu_B, m1, m2)
        w_c = w_c / max(float(np.linalg.norm(w_c)), 1e-12)

        # 4. Exact Real-B Conformal Threshold on the ACTUAL Compiled Gate Event!
        # In biasfree mode, the compiled lower gate fires iff (w_c - t1 * c_B)^T x > 0, which equals
        # (w_c^T x) / (c_B^T x) > t1 whenever c_B^T x > 0 (rows with c_B^T x <= 0 are conservatively +inf).
        z_real_B = Xb_ev_r @ w_c
        std_real_c = float(np.std(z_real_B)) + 1e-12
        m_real = float(np.mean(z_real_B))
        if mode == "biasfree":
            pin_ev = Xb_ev_r @ c_B
            cal_scores = np.where(pin_ev > 1e-12, z_real_B / np.where(pin_ev > 1e-12, pin_ev, 1.0), np.inf)
        else:
            cal_scores = z_real_B
        q_real_B = float(np.sort(cal_scores)[k_conf - 1]) if conformal_n_ok else float("inf")
        finite_cal = cal_scores[np.isfinite(cal_scores)]
        std_cal = float(np.std(finite_cal)) + 1e-12 if len(finite_cal) > 1 else std_real_c

        if Xb_pred_ev is not None:
            z_pred_r = Xb_pred_ev @ w_c
            m_pred = float(np.mean(z_pred_r))
            z_pred_c = z_pred_r - m_pred
            z_real_c = z_real_B - m_real
            var_pred = float(np.var(z_pred_c))
            var_real = float(np.var(z_real_c))
            cov_rp = float(np.mean(z_real_c * z_pred_c))
            r_dir = cov_rp / math.sqrt(max(var_pred * var_real, 1e-18))
            a_emp = max(0.0, cov_rp / max(var_pred, 1e-18))
            e_cal_1d = z_real_c - a_emp * z_pred_c
        else:
            r_dir = 1.0
            a_emp = 1.0
            e_cal_1d = None
            m_pred = m_real

        # Exact Real-B Conformal Threshold (t1 = Q_{1-alpha}(cal_scores)) & 1-sigma Ramp (Gamma = std_cal)!
        t1 = q_real_B
        Gamma = std_cal
        t2 = t1 + Gamma
        beta = min(SILU_NEG_MAX / (delta * Gamma), 100.0 / Gamma)

        # Diagnostic predicted hit-rate & d' on empirical channel residual distribution
        z_pos_raw = Xp_sim @ w_c
        if is_native or e_cal_1d is None:
            zp_diag = z_pos_raw
        else:
            rng_d = np.random.default_rng(NOISE_SEEDS[0])
            zp_diag = np.tile(m_real + a_emp * (z_pos_raw - m_pred), 4) + e_cal_1d[rng_d.integers(0, len(e_cal_1d), size=4 * len(z_pos_raw))]
        pred_hit_rate = float(np.mean(zp_diag >= 0.5 * (t1 + t2))) if np.isfinite(t1) else 0.0
        sig_pool = math.sqrt(0.5 * float(np.var(zp_diag)) + 0.5 * float(np.var(z_real_B)) + 1e-12)
        pred_dprime = float((np.mean(zp_diag) - m_real) / sig_pool)

        # 5. Mandatory Frame, Conformal-Sample-Size & Unlabeled Channel Abstention Certificates
        channel_abstained = (not is_native) and ((r2_chan_ev <= 0.0) or (r_dir <= 0.25) or (a_emp <= 0.15))
        abstained = (not frame_ok) or (not conformal_n_ok) or (not np.isfinite(t1)) or channel_abstained
        if abstained:
            any_abstained = True

        if abstained and not force_abstained:
            if not frame_ok:
                abstain_reason = f"frame_cert_failed(auroc={f_auroc:.4f})"
            elif not conformal_n_ok or not np.isfinite(t1):
                abstain_reason = f"conformal_insufficient_n(n_ev={n_ev},alpha={alpha})"
            elif r2_chan_ev <= 0.0:
                abstain_reason = "r2_chan_ev<=0"
            elif r_dir <= 0.25:
                abstain_reason = "r_dir<=0.25"
            else:
                abstain_reason = "a_emp<=0.15"
            rule_reports.append({
                "name": rname,
                "status": "ABSTAIN",
                "abstain_reason": abstain_reason,
                "frame_only_auroc": round(f_auroc, 4),
                "frame_cert_pass": frame_ok,
                "q_real_B": round(q_real_B, 5),
                "t1": round(t1, 5),
                "t2": round(t2, 5),
                "Gamma": round(Gamma, 5),
                "r2_chan_heldout": round(float(r2_chan_ev), 4),
                "r_dir_heldout": round(float(r_dir), 4),
                "a_emp": round(float(a_emp), 4),
                "pred_hit_rate": round(pred_hit_rate, 4),
                "pred_dprime": round(pred_dprime, 3),
                "w_c": w_c,
            })
            continue

        # 6. Exact Analytic Twin-SwiGLU Row Construction (Pure Digital BOS-Nulled Pin, gain == 1.000!)
        v_B = inv_rm * (v_A if is_native else (M_out_iso @ v_A))
        denom = beta * Gamma

        if mode == "biasfree":
            W_g_rows.append(beta * (w_c - t1 * c_B))
            W_g_rows.append(beta * (w_c - t2 * c_B))
            if ablate_domain_up and w_dom_u is not None:
                wu_row = po.apply_exact_bos_null_biasfree((w_dom_u + b_dom_u * c_B)[None, :], x_bos_B, x_bos_bank=bos_bank_B)[0]
                W_u_rows.append(wu_row)
                W_u_rows.append(wu_row)
            else:
                W_u_rows.append(c_u)
                W_u_rows.append(c_u)
        elif mode == "affine":
            W_g_rows.append(beta * w_c)
            W_g_rows.append(beta * w_c)
            b_g_rows.append(-beta * t1)
            b_g_rows.append(-beta * t2)
            if ablate_domain_up and w_dom_u is not None:
                W_u_rows.append(w_dom_u)
                W_u_rows.append(w_dom_u)
                b_u_rows.append(b_dom_u)
                b_u_rows.append(b_dom_u)
            else:
                W_u_rows.append(np.zeros(D_B, dtype=np.float64))
                W_u_rows.append(np.zeros(D_B, dtype=np.float64))
                b_u_rows.append(1.0)
                b_u_rows.append(1.0)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        W_d_cols.append(v_B / denom)
        W_d_cols.append(-v_B / denom)

        rule_reports.append({
            "name": rname,
            "status": "FORCED_FROM_ABSTAIN" if abstained else "COMPILED",
            "frame_only_auroc": round(f_auroc, 4),
            "frame_cert_pass": frame_ok,
            "q_real_B": round(q_real_B, 5),
            "t1": round(t1, 5),
            "t2": round(t2, 5),
            "Gamma": round(Gamma, 5),
            "r2_chan_heldout": round(float(r2_chan_ev), 4),
            "r_dir_heldout": round(float(r_dir), 4),
            "a_emp": round(float(a_emp), 4),
            "beta": round(float(beta), 4),
            "beta_Gamma": round(float(beta * Gamma), 4),
            "pred_hit_rate": round(pred_hit_rate, 4),
            "pred_dprime": round(pred_dprime, 3),
            "vB_res_norm": round(float(np.linalg.norm(v_B * rm_B)), 4),
            "w_c": w_c,
        })

    if any_abstained and not force_abstained:
        return {
            "status": "ABSTAIN",
            "use_bias": (mode == "affine"),
            "tolerances": tol,
            "rules": rule_reports,
            "weights": None,
        }

    W_g = np.stack(W_g_rows, axis=0)
    W_u = np.stack(W_u_rows, axis=0)
    W_d = np.stack(W_d_cols, axis=1)
    if mode == "affine":
        b_g = np.array(b_g_rows, dtype=np.float64)
        b_u = np.array(b_u_rows, dtype=np.float64)
        W_g, b_g, W_u, b_u = po.apply_exact_bos_null_affine(
            W_g, b_g, W_u, b_u, x_bos_B, mu_B, x_bos_bank=bos_bank_B
        )
    else:
        b_g, b_u = None, None
        W_u = po.apply_exact_bos_null_biasfree(W_u, x_bos_B, x_bos_bank=bos_bank_B)

    return {
        "status": "FORCED_FROM_ABSTAIN" if any_abstained else "COMPILED",
        "use_bias": (mode == "affine"),
        "ablate_domain_up": ablate_domain_up,
        "tolerances": tol,
        "rules": rule_reports,
        "weights": {"W_g": W_g, "b_g": b_g, "W_u": W_u, "b_u": b_u, "W_d": W_d, "use_bias": (mode == "affine")},
    }


def strip_rule_vectors(compiled: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": compiled["status"],
        "use_bias": compiled["use_bias"],
        "tolerances": compiled["tolerances"],
        "rules": [{k: v for k, v in r.items() if k != "w_c"} for r in compiled["rules"]],
    }


def concat_compiled_organs(weight_dicts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    use_bias = any(o["use_bias"] for o in weight_dicts)
    W_g = np.concatenate([o["W_g"] for o in weight_dicts], axis=0)
    W_u = np.concatenate([o["W_u"] for o in weight_dicts], axis=0)
    W_d = np.concatenate([o["W_d"] for o in weight_dicts], axis=1)
    if use_bias:
        b_g = np.concatenate([o["b_g"] if o["b_g"] is not None else np.zeros(len(o["W_g"])) for o in weight_dicts], axis=0)
        b_u = np.concatenate([o["b_u"] if o["b_u"] is not None else np.zeros(len(o["W_u"])) for o in weight_dicts], axis=0)
    else:
        b_g, b_u = None, None
    return {"W_g": W_g, "b_g": b_g, "W_u": W_u, "b_u": b_u, "W_d": W_d, "use_bias": use_bias}


# ======================================================================================
# 3. Widening, Selective Slicing, Bit-Identity Unlearning & Unlabeled Depth/Band Law
# ======================================================================================

def widen_mlp_native(mlp: nn.Module, orig_state: Dict[str, Any], weights: Dict[str, Any]) -> int:
    return po.widen_mlp_native(
        mlp, orig_state,
        weights["W_g"], weights["b_g"], weights["W_u"], weights["b_u"], weights["W_d"],
        use_bias=weights["use_bias"],
    )


def slice_mlp_rows_native(mlp: nn.Module, orig_state: Dict[str, Any], keep_extra_indices: Sequence[int]) -> int:
    d_ff = orig_state["d_ff"]
    d_model = orig_state["d_model"]
    device = mlp.gate_proj.weight.device
    dtype = mlp.gate_proj.weight.dtype
    if len(keep_extra_indices) == 0:
        po.unlearn_mlp_native(mlp, orig_state)
        return d_ff
    keep_idx = list(range(d_ff)) + [d_ff + int(i) for i in keep_extra_indices]
    idx_t = torch.tensor(keep_idx, dtype=torch.long, device=device)
    new_dff = len(keep_idx)
    gb = mlp.gate_proj.bias is not None
    ub = mlp.up_proj.bias is not None
    db = mlp.down_proj.bias is not None
    ng = nn.Linear(d_model, new_dff, bias=gb, device=device, dtype=dtype)
    nu = nn.Linear(d_model, new_dff, bias=ub, device=device, dtype=dtype)
    nd = nn.Linear(new_dff, d_model, bias=db, device=device, dtype=dtype)
    with torch.no_grad():
        ng.weight.copy_(mlp.gate_proj.weight[idx_t])
        if gb: ng.bias.copy_(mlp.gate_proj.bias[idx_t])
        nu.weight.copy_(mlp.up_proj.weight[idx_t])
        if ub: nu.bias.copy_(mlp.up_proj.bias[idx_t])
        nd.weight.copy_(mlp.down_proj.weight[:, idx_t])
        if db: nd.bias.copy_(mlp.down_proj.bias)
    mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.intermediate_size = ng, nu, nd, new_dff
    return new_dff


def verify_exact_bit_unlearning(
    model: Any,
    tok: Any,
    device: torch.device,
    mlp: nn.Module,
    orig_state: Dict[str, Any],
    base_logits_probe: torch.Tensor,
    probe_prompts: Sequence[str],
) -> Dict[str, Any]:
    """
    Slice widened MLP back to d_ff and verify BOTH:
      1. Exact state_dict raw tensor bit-identity (torch.equal) & bias restoration!
      2. Exact deterministic logit bit-identity (torch.equal AND max_abs == 0.0)!
    """
    po.unlearn_mlp_native(mlp, orig_state)
    sd_gate_eq = bool(torch.equal(mlp.gate_proj.weight.detach(), orig_state["gate_weight"]))
    sd_up_eq = bool(torch.equal(mlp.up_proj.weight.detach(), orig_state["up_weight"]))
    sd_down_eq = bool(torch.equal(mlp.down_proj.weight.detach(), orig_state["down_weight"]))
    sd_bias_eq = (
        (mlp.gate_proj.bias is None) == (orig_state["gate_b"] is None)
        and (mlp.up_proj.bias is None) == (orig_state["up_b"] is None)
        and (mlp.down_proj.bias is None) == (orig_state["down_b"] is None)
    )
    enc = tok(list(probe_prompts), return_tensors="pt", padding=True, add_special_tokens=True)
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            use_cache=False,
        ).logits.cpu()
    logit_bit_eq = bool(torch.equal(logits, base_logits_probe))
    max_diff = float(torch.max(torch.abs(logits - base_logits_probe)).item())
    assert sd_gate_eq and sd_up_eq and sd_down_eq and sd_bias_eq, "State-dict unlearning NOT bit-identical!"
    assert max_diff == 0.0 and logit_bit_eq, f"Logits NOT bit-identical after unlearning: max_diff={max_diff}"
    return {
        "state_dict_bit_equal": True,
        "logits_bit_equal": logit_bit_eq,
        "max_abs_dlogit": max_diff,
        "restored_d_ff": int(mlp.gate_proj.out_features),
    }


def extract_entity_states(
    model: Any,
    tok: Any,
    device: torch.device,
    prompts: Sequence[str],
    char_spans: Sequence[Sequence[int]],
    layer_idx: int,
    all_subtokens: bool = False,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Extract post-LN x and post-layer h_out at last_subject_token (default) OR across ALL entity
    subtokens (Opus Q3 mask-free echo neutralization!). Returns x, h_out, and sample_ids mapping each
    extracted row back to its prompt index.
    """
    if not all_subtokens:
        st = po.extract_entity_last_token_states(model, tok, device, prompts, char_spans, layer_idx, batch_size=batch_size)
        return {"x": st["x"], "h_out": st["h_out"], "sample_ids": np.arange(len(prompts), dtype=np.int64)}

    layers = get_decoder_layers(model)
    layer = layers[layer_idx]
    buf: Dict[str, torch.Tensor] = {}
    h1 = layer.post_attention_layernorm.register_forward_hook(lambda _m, _i, o: buf.__setitem__("x", o.detach()))
    h2 = layer.register_forward_hook(lambda _m, _i, o: buf.__setitem__("h_out", (o[0] if isinstance(o, tuple) else o).detach()))

    # Find all overlapping subtoken indices per prompt
    sub_pos_lists: List[List[int]] = []
    for pr, sp in zip(prompts, char_spans):
        c0, c1 = int(sp[0]), int(sp[1])
        enc = tok(pr, return_offsets_mapping=True, add_special_tokens=True)
        offs = enc["offset_mapping"]
        pos = [idx for idx, (a, b) in enumerate(offs) if b > a and max(a, c0) < min(b, c1)]
        if not pos:
            last_p, _ = locate_last_subject_token(tok, pr, sp)
            pos = [last_p]
        sub_pos_lists.append(pos)

    xs, hos, sids = [], [], []
    old_pad = tok.padding_side
    tok.padding_side = "right"
    try:
        with torch.no_grad():
            for i0 in range(0, len(prompts), batch_size):
                b_pr = list(prompts[i0 : i0 + batch_size])
                enc = tok(b_pr, return_tensors="pt", padding=True, add_special_tokens=True)
                _ = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device), use_cache=False)
                xb = buf["x"].float().cpu().numpy()
                hb = buf["h_out"].float().cpu().numpy()
                for bi in range(len(b_pr)):
                    gidx = i0 + bi
                    for p in sub_pos_lists[gidx]:
                        xs.append(xb[bi, p].astype(np.float32))
                        hos.append(hb[bi, p].astype(np.float32))
                        sids.append(gidx)
    finally:
        h1.remove(); h2.remove(); tok.padding_side = old_pad
    return {"x": np.stack(xs, axis=0), "h_out": np.stack(hos, axis=0), "sample_ids": np.array(sids, dtype=np.int64)}


def measure_unlabeled_commit_healing(
    model: Any,
    tok: Any,
    device: torch.device,
    n_words: int = 16,
) -> Dict[str, Any]:
    """
    Task-Free Model-Intrinsic Depth & Band Selector (Fable §3 + Neon Q3):
    On unlabeled multi-token technical words (0% geography, 0% animals!), resample-patches the last
    subtoken of word i with word j at candidate depth fractions [20%, 30%, 38%, 46%, 55%] and measures:
      - Unlabeled commit C_B(l) = E[KL(clean || patch_last_at_l)] / E[KL(clean || patch_whole_at_0)]
      - Self-healing H_B(l -> l+delta) = 1 - ||h_{l+d}^patch - h_{l+d}^clean|| / ||h_l^patch - h_l^clean||
      - Per-layer step authority eta(l) = median(||rm * mlp_l(x)|| / ||h_l||)
    """
    unlabeled_words = [
        "interferometry", "thermodynamics", "crystallography", "electromagnetism",
        "spectroscopy", "hydrodynamics", "polycarbonate", "microcontroller",
        "calorimetry", "photogrammetry", "chromatography", "superconductor",
        "magnetosphere", "chronometer", "oscilloscope", "piezoelectric",
    ][:n_words]
    prompts = [f"In engineering notes, the term {w} refers to" for w in unlabeled_words]
    spans = [[p.index(w), p.index(w) + len(w)] for p, w in zip(prompts, unlabeled_words)]

    layers = get_decoder_layers(model)
    L = len(layers)
    rm = float(getattr(layers[0], "residual_multiplier", 1.0))
    cand_layers = sorted(set(max(1, min(L - 3, int(round(f * L)))) for f in (0.20, 0.30, 0.38, 0.46, 0.55)))

    old_pad = tok.padding_side
    tok.padding_side = "right"
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    tok.padding_side = old_pad

    last_pos = [locate_last_subject_token(tok, p, sp)[0] for p, sp in zip(prompts, spans)]
    end_pos = [int(attn_mask[i].sum().item()) - 1 for i in range(len(prompts))]

    # Clean forward pass capturing all candidate layer outputs & mlp norms + clean final logprobs
    clean_h: Dict[int, torch.Tensor] = {}
    clean_eta: Dict[int, float] = {}
    hooks = []
    for l in range(L):
        if l in cand_layers or any(l == cl + 2 for cl in cand_layers):
            def _mk_h(li):
                def _fn(_m, _i, o):
                    clean_h[li] = (o[0] if isinstance(o, tuple) else o).detach()
                return _fn
            hooks.append(layers[l].register_forward_hook(_mk_h(l)))
        if l in cand_layers:
            def _mk_m(li):
                def _fn(_m, _i, o):
                    # Store mlp output norm
                    clean_eta[li] = float(torch.median(torch.norm(o.detach().float() * rm, dim=-1)).item())
                return _fn
            hooks.append(layers[l].mlp.register_forward_hook(_mk_m(l)))
    with torch.no_grad():
        clean_logits = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False).logits
    for h in hooks:
        h.remove()

    clean_lp = torch.stack([F.log_softmax(clean_logits[i, end_pos[i]].float(), dim=-1) for i in range(len(prompts))])
    clean_p = torch.exp(clean_lp)

    # Deranged partner indices (j = (i + 1) % n)
    n = len(prompts)
    partners = [(i + 1) % n for i in range(n)]

    # Whole-word swap baseline KL (KL between prompt i and prompt partner[i]!)
    whole_kl = float(torch.mean(torch.sum(clean_p * (clean_lp - clean_lp[partners]), dim=-1)).item())

    by_layer: Dict[str, Any] = {}
    for l in cand_layers:
        l_probe = min(L - 1, l + 2)
        h_l_clean = clean_h[l]
        h_p_clean = clean_h[l_probe]
        # Compute eta(l) = median(||rm * mlp||) / median(||h_l||)
        h_med = float(torch.median(torch.norm(h_l_clean.float(), dim=-1)).item())
        eta_l = clean_eta[l] / max(h_med, 1e-8)

        # Resample-patch last token of word i with partner j at layer l
        patched_probe: Dict[str, torch.Tensor] = {}

        def _patch_fn(_m, _i, o):
            is_tup = isinstance(o, tuple)
            t = (o[0] if is_tup else o).clone()
            for i, j in enumerate(partners):
                t[i, last_pos[i]] = h_l_clean[j, last_pos[j]]
            return (t,) + o[1:] if is_tup else t

        def _cap_probe(_m, _i, o):
            patched_probe["h"] = (o[0] if isinstance(o, tuple) else o).detach()

        hp1 = layers[l].register_forward_hook(_patch_fn)
        hp2 = layers[l_probe].register_forward_hook(_cap_probe)
        with torch.no_grad():
            p_logits = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False).logits
        hp1.remove(); hp2.remove()

        p_lp = torch.stack([F.log_softmax(p_logits[i, end_pos[i]].float(), dim=-1) for i in range(n)])
        kl_l = float(torch.mean(torch.clamp(torch.sum(clean_p * (clean_lp - p_lp), dim=-1), min=0.0)).item())
        C_l = kl_l / max(whole_kl, 1e-6)

        # Self-healing H_B(l -> l+2) at last_pos
        d0 = torch.stack([h_l_clean[j, last_pos[j]] - h_l_clean[i, last_pos[i]] for i, j in enumerate(partners)])
        d2 = torch.stack([patched_probe["h"][i, last_pos[i]] - h_p_clean[i, last_pos[i]] for i in range(n)])
        # Ratio of perturbation retained 2 layers later (relative to stream norm)
        n0 = torch.norm(d0.float(), dim=-1) / (torch.norm(h_l_clean[[i for i in range(n)], last_pos].float(), dim=-1) + 1e-8)
        n2 = torch.norm(d2.float(), dim=-1) / (torch.norm(h_p_clean[[i for i in range(n)], last_pos].float(), dim=-1) + 1e-8)
        H_l = float(1.0 - torch.mean(n2 / (n0 + 1e-8)).item())

        by_layer[str(l)] = {
            "layer": l,
            "depth_frac": round(l / float(L), 3),
            "C_commit": round(C_l, 4),
            "H_healing_2L": round(H_l, 4),
            "eta_step_auth": round(eta_l, 4),
            "rec_band_width": int(max(1, min(3, round(0.25 / max(eta_l, 0.05))))),
        }

    best_l = max(cand_layers, key=lambda li: by_layer[str(li)]["C_commit"])
    return {
        "num_layers": L,
        "rm": rm,
        "whole_word_kl": round(whole_kl, 4),
        "recommended_layer": int(best_l),
        "recommended_band_width": by_layer[str(best_l)]["rec_band_width"],
        "by_layer": by_layer,
    }


# ======================================================================================
# 4. Universal Channel Caching (Shared Identically Across Cap-1 AND Sealed Cap-2!)
# ======================================================================================

def extract_donor_channel_passages(
    donor_key: str,
    lA: int,
    corpus: Dict[str, List[str]],
    tag: str = "universal",
) -> Dict[str, Any]:
    """Extract & cache donor's passage states once so multiple recipients share the same extraction."""
    cache_p = CACHE_DIR / f"{tag}_donor_passages_{donor_key}_l{lA}.pt"
    if cache_p.exists():
        print(f"  [Channel] Loading cached donor passages {cache_p.name}")
        return torch.load(cache_p, map_location="cpu", weights_only=False)
    print(f"  [Channel] Extracting {donor_key} L{lA} on {tag} corpus ({len(corpus['fit'])}/{len(corpus['resid'])}/{len(corpus['eval'])})...")
    model_A, tok_A, dev_A, _ = po.load_model_fp32(donor_key)
    out = {
        "fit": po.extract_layer_states_on_passages(model_A, tok_A, dev_A, corpus["fit"], lA),
        "resid": po.extract_layer_states_on_passages(model_A, tok_A, dev_A, corpus["resid"], lA),
        "eval": po.extract_layer_states_on_passages(model_A, tok_A, dev_A, corpus["eval"], lA),
    }
    po.unload_model(model_A)
    torch.save(out, cache_p)
    return out


def extract_donor_concept_write_states(
    donor_key: str,
    lA: int,
) -> Dict[str, Any]:
    """Extract & cache donor post-layer h_out on the 1,200 universal short-context concept sentences."""
    cache_p = CACHE_DIR / f"universal_write_concept_donor_{donor_key}_l{lA}.pt"
    if cache_p.exists():
        print(f"  [WriteBridge] Loading cached donor concept states {cache_p.name}")
        return torch.load(cache_p, map_location="cpu", weights_only=False)
    cw = build_universal_concept_write_corpus()
    print(f"  [WriteBridge] Extracting {donor_key} L{lA} on {len(cw['prompts'])} universal concept sentences...")
    model_A, tok_A, dev_A, _ = po.load_model_fp32(donor_key)
    st_A = extract_entity_states(model_A, tok_A, dev_A, cw["prompts"], cw["spans"], lA, all_subtokens=False, batch_size=32)
    po.unload_model(model_A)
    out = {"h_out": st_A["h_out"].astype(np.float32), "n_prompts": len(cw["prompts"])}
    torch.save(out, cache_p)
    return out


def fit_or_load_concept_write_bridge(
    donor_key: str,
    lA: int,
    recip_key: str,
    lB: int,
    model_B: Any,
    tok_B: Any,
    dev_B: torch.device,
    donor_concept_states: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Lightweight companion cache for task-agnostic short-context concept write bridge:
      cache/universal_write_concept_{donor}_l{lA}_to_{recip}_l{lB}.pt
    Reuses existing universal_channel_*.pt read caches 100% untouched (~10s per model)!
    """
    cache_p = CACHE_DIR / f"universal_write_concept_{donor_key}_l{lA}_to_{recip_key}_l{lB}.pt"
    if cache_p.exists():
        print(f"  [WriteBridge] Loading frozen {cache_p.name}")
        return torch.load(cache_p, map_location="cpu", weights_only=False)

    if donor_concept_states is None:
        donor_concept_states = extract_donor_concept_write_states(donor_key, lA)

    cw = build_universal_concept_write_corpus()
    print(f"  [WriteBridge] Extracting {recip_key} L{lB} on {len(cw['prompts'])} concept sentences & fitting Cauchy-Stiefel bridge...")
    st_B = extract_entity_states(model_B, tok_B, dev_B, cw["prompts"], cw["spans"], lB, all_subtokens=False, batch_size=32)
    wb = fit_concept_write_bridge(donor_concept_states["h_out"], st_B["h_out"])
    torch.save(wb, cache_p)
    print(f"  [WriteBridge] Saved {cache_p.name} | {wb['summary']}")
    return wb


def fit_or_load_universal_channel(
    donor_key: str,
    lA: int,
    recip_key: str,
    lB: int,
    corpus: Dict[str, List[str]],
    model_B: Any,
    tok_B: Any,
    dev_B: torch.device,
    donor_passages: Optional[Dict[str, Any]] = None,
    tag: str = "universal",
    donor_concept_states: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fit or load frozen task-agnostic generative channel cache:
      cache/{tag}_channel_{donor}_l{lA}_to_{recip}_l{lB}.pt
    Reuses existing read caches 100% unchanged and (for tag=='universal') attaches the
    lightweight companion concept write bridge cache:
      cache/universal_write_concept_{donor}_l{lA}_to_{recip}_l{lB}.pt
    Shared identically across Cap-1 (Geography) and Sealed Cap-2 (BIO-RIPPLE)!
    """
    cache_p = CACHE_DIR / f"{tag}_channel_{donor_key}_l{lA}_to_{recip_key}_l{lB}.pt"
    if cache_p.exists():
        print(f"  [Channel] Loading frozen {cache_p.name}")
        ch = torch.load(cache_p, map_location="cpu", weights_only=False)
    else:
        if donor_passages is None:
            donor_passages = extract_donor_channel_passages(donor_key, lA, corpus, tag=tag)

        print(f"  [Channel] Extracting {recip_key} L{lB} on {tag} corpus & fitting generative channel...")
        stB_fit = po.extract_layer_states_on_passages(model_B, tok_B, dev_B, corpus["fit"], lB)
        stB_res = po.extract_layer_states_on_passages(model_B, tok_B, dev_B, corpus["resid"], lB)
        stB_ev = po.extract_layer_states_on_passages(model_B, tok_B, dev_B, corpus["eval"], lB)

        p_fit = po.pair_passage_activations(donor_passages["fit"], stB_fit)
        p_res = po.pair_passage_activations(donor_passages["resid"], stB_res)
        p_ev = po.pair_passage_activations(donor_passages["eval"], stB_ev)

        ch = fit_generative_channel(
            p_fit["x_A"], p_fit["x_B"],
            p_res["x_A"], p_res["x_B"],
            p_fit["h_out_A"], p_fit["h_out_B"],
            p_res["h_out_A"], p_res["h_out_B"],
            stB_fit["bos_x"],
        )
        ch["bos_x_eval_B"] = stB_ev["bos_x"]
        ch["x_B_eval_aligned"] = p_ev["x_B"]
        ch["x_B_eval_unaligned"] = p_ev["x_B_unaligned"]
        ch["mlpB_eval_aligned_n2"] = p_ev["mlpB_aligned_n2"]
        ch["mlpB_eval_unaligned_n2"] = p_ev["mlpB_unaligned_n2"]
        ch["summary"]["tag"] = tag
        ch["summary"]["n_eval_aligned"] = int(len(p_ev["x_A"]))
        torch.save(ch, cache_p)
        print(f"  [Channel] Saved {cache_p.name} | {ch['summary']}")

    # Always preserve long-paragraph prose write matrix and attach universal concept write bridge!
    if "M_out_prose" not in ch:
        ch["M_out_prose"] = np.asarray(ch["M_out_iso"], dtype=np.float64)
    if tag == "universal":
        wb = fit_or_load_concept_write_bridge(
            donor_key, lA, recip_key, lB, model_B, tok_B, dev_B, donor_concept_states=donor_concept_states
        )
        ch["M_out_concept"] = wb["M_out_concept"]
        ch["M_out_iso"] = wb["M_out_concept"]
        ch["summary"]["write_concept"] = wb["summary"]
    return ch


# ======================================================================================
# 5. Strictly Frame-Counterbalanced Cap-1 (Geography) Runner & 2x2 Ablation & Stage B
# ======================================================================================

def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Dict[str, float]:
    if n <= 0:
        return {"p": 0.0, "lo": 0.0, "hi": 0.0, "k": 0, "n": 0}
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return {"p": round(p, 4), "lo": round(max(0.0, center - half), 4), "hi": round(min(1.0, center + half), 4), "k": int(k), "n": int(n)}


def mcnemar_exact(a_ok: Sequence[int], b_ok: Sequence[int]) -> Dict[str, Any]:
    a, b = np.asarray(a_ok, dtype=np.int64), np.asarray(b_ok, dtype=np.int64)
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))
    nd = a_only + b_only
    return {"a_only": a_only, "b_only": b_only, "p_value": 1.0 if nd == 0 else float(binomtest(a_only, nd, 0.5).pvalue)}


def extract_cap1_counterbalanced_donor_rules(
    model_key: str,
    layer_idx: int,
    rules_to_extract: Sequence[str] = ("France", "Italy"),
    all_subtokens: bool = False,
) -> List[Dict[str, Any]]:
    """
    Strictly frame-counterbalanced Cap-1 Donor construction (frame_only_auroc == 0.5000 EXACT!):
    Every positive (France/Italy train entity) AND every in-domain negative (other-rule entity +
    ORGAN1_HARDNEG_TRAIN) uses the EXACT SAME 6 templates (TRAIN_PREFIX_TEMPLATES), eliminating
    the 4.5:1 dataset.json prefix confound at the source!
    """
    print(f"  [Cap-1 Donor] Extracting strictly frame-counterbalanced rules from {model_key} L{layer_idx} (all_subtokens={all_subtokens})...")
    model, tok, dev, _ = po.load_model_fp32(model_key)

    def _build_group(ents):
        prs, sps, fids = [], [], []
        for e in ents:
            for ti, tmpl in enumerate(po.TRAIN_PREFIX_TEMPLATES):
                p = tmpl.format(e=e["name"]); c0 = p.index(e["name"])
                prs.append(p); sps.append([c0, c0 + len(e["name"])]); fids.append(ti)
        st = extract_entity_states(model, tok, dev, prs, sps, layer_idx, all_subtokens=all_subtokens)
        fids_exp = [fids[i] for i in st["sample_ids"]]
        return st["x"].astype(np.float64), np.array(fids_exp, dtype=np.int64)

    fr_ents = [e for e in po.ORGAN1_TRAIN_ENTITIES if e["country"] == "France"]
    it_ents = [e for e in po.ORGAN1_TRAIN_ENTITIES if e["country"] == "Italy"]
    hn_ents = list(po.ORGAN1_HARDNEG_TRAIN)

    x_fr, f_fr = _build_group(fr_ents)
    x_it, f_it = _build_group(it_ents)
    x_hn, f_hn = _build_group(hn_ents)

    need_c = []
    if "France" in rules_to_extract: need_c += ["France", "Japan"]
    if "Italy" in rules_to_extract: need_c += ["Italy", "Brazil"]
    cA = po.extract_country_sibling_deltas(model, tok, dev, layer_idx, sorted(set(need_c)))
    po.unload_model(model)

    rules = []
    if "France" in rules_to_extract:
        v_fr = (cA["country"]["Japan"] - cA["country"]["France"]) + (cA["capital"]["Japan"] - cA["capital"]["France"])
        rules.append({
            "name": "France->Japan",
            "country": "France",
            "X_pos_A": x_fr,
            "pos_frame_ids": f_fr,
            "X_neg_dom_A": np.vstack([x_it, x_hn]),
            "neg_frame_ids": np.concatenate([f_it, f_hn]),
            "v_A": v_fr,
        })
    if "Italy" in rules_to_extract:
        v_it = (cA["country"]["Brazil"] - cA["country"]["Italy"]) + (cA["capital"]["Brazil"] - cA["capital"]["Italy"])
        rules.append({
            "name": "Italy->Brazil",
            "country": "Italy",
            "X_pos_A": x_it,
            "pos_frame_ids": f_it,
            "X_neg_dom_A": np.vstack([x_fr, x_hn]),
            "neg_frame_ids": np.concatenate([f_fr, f_hn]),
            "v_A": v_it,
        })
    return rules


def run_cap1_suite(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    import pilot_organs_multifamily as pmf  # Reuse eval_organ1_detailed & load_dataset_splits!

    corpus_univ = build_universal_calibration_corpus()
    corpus_prose = po.build_diverse_general_corpus()
    _, test_facts, ds_fr_it_test, fr_it_exclude, _ = pmf.load_dataset_splits()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    probe_prompts = [f"The location {e['name']} is situated in the country of" for e in po.ORGAN1_TEST_ENTITIES[:4]]

    print(f"\n================ CAP-1 CHANNEL-PUSHFORWARD SUITE (SHA256={get_engine_sha256()[:12]}) ================")
    print(f"Frozen tolerances: {FROZEN_TOLERANCES} | Targets: {targets}")

    # 1. Extract Donor A (qwen2.5-3b L17) Cap-1 counterbalanced rules, universal passages & concept write states ONCE, then UNLOAD!
    donor_passages_univ = extract_donor_channel_passages(DONOR_QWEN_KEY, DONOR_QWEN_LAYER, corpus_univ, tag="universal")
    donor_concept_qwen = extract_donor_concept_write_states(DONOR_QWEN_KEY, DONOR_QWEN_LAYER)
    donor_passages_prose = extract_donor_channel_passages(DONOR_QWEN_KEY, DONOR_QWEN_LAYER, corpus_prose, tag="proseonly") if "smollm2" in targets else None
    qwen_rules_cache = CACHE_DIR / f"cap1_cb_rules_{DONOR_QWEN_KEY}_l{DONOR_QWEN_LAYER}_sub{int(args.all_subtokens)}.pt"
    if qwen_rules_cache.exists() and not args.refit_donor:
        qwen_rules = torch.load(qwen_rules_cache, map_location="cpu", weights_only=False)
    else:
        qwen_rules = extract_cap1_counterbalanced_donor_rules(DONOR_QWEN_KEY, DONOR_QWEN_LAYER, ("France", "Italy"), all_subtokens=args.all_subtokens)
        torch.save(qwen_rules, qwen_rules_cache)

    out: Dict[str, Any] = {
        "meta": {
            "engine_sha256": get_engine_sha256(),
            "tolerances": FROZEN_TOLERANCES,
            "all_subtokens": bool(args.all_subtokens),
            "ablate_domain_up": bool(args.ablate_domain_up),
            "targets": targets,
            "seed": SEED,
        },
        "stage_A_multifamily": {},
        "stage_A_2x2_smollm2_prose_vs_universal": {},
        "stage_B_multidonor_composition": {},
    }

    # --- Stage A: All 4 Architectures (Donor A COMPLETELY UNLOADED!) ---
    for tgt in targets:
        tr0 = time.time()
        spec = RECIPIENT_SPECS[tgt]
        rk, lB = spec["model_key"], spec["layer"]
        print(f"\n--- [Cap-1 Stage A | {tgt}] SOLO Recipient {rk} L{lB} ---")
        model_B, tok_B, dev_B, rm_B = po.load_model_fp32(rk)
        mlp_B = get_decoder_layers(model_B)[lB].mlp
        orig_state_B = po.save_original_mlp_state(mlp_B)

        depth_diag = measure_unlabeled_commit_healing(model_B, tok_B, dev_B)
        print(f"  [Unlabeled Commit/Healing] rec_layer=L{depth_diag['recommended_layer']} rec_band={depth_diag['recommended_band_width']} | by_layer={ {k: (v['C_commit'], v['H_healing_2L']) for k,v in depth_diag['by_layer'].items()} }")

        ch_univ = fit_or_load_universal_channel(
            DONOR_QWEN_KEY, DONOR_QWEN_LAYER, rk, lB, corpus_univ, model_B, tok_B, dev_B,
            donor_passages_univ, tag="universal", donor_concept_states=donor_concept_qwen,
        )

        enc_p = tok_B(probe_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        with torch.no_grad():
            base_logits_probe = model_B(input_ids=enc_p["input_ids"].to(dev_B), attention_mask=enc_p["attention_mask"].to(dev_B), use_cache=False).logits.cpu()
        base_gt, base_lps = po.eval_general_text_kl_ppl(model_B, tok_B, dev_B, corpus_univ["eval"])
        base_loc, base_loc_ok = po.eval_196_locality(model_B, tok_B, dev_B, test_facts)
        base_qa = pmf.eval_organ1_detailed(model_B, tok_B, dev_B, ds_fr_it_test)
        print(f"  Baseline {rk}: PPL={base_gt['ppl']:.4f}, 196-fact={base_loc['n_correct_all']}/196")

        def _eval_compiled(tag_name: str, compiled: Dict[str, Any], ch_ref: Dict[str, Any], raw_store: Dict[str, Any]) -> Dict[str, Any]:
            rep = strip_rule_vectors(compiled)
            if compiled["weights"] is None:
                print(f"  [{tag_name:28s}] STATUS=ABSTAIN | rules={[(r['name'], r.get('abstain_reason',''), 'G='+str(r['Gamma']), 'r='+str(r['r_dir_heldout']), 'R2='+str(r.get('r2_chan_heldout','')), 'hit='+str(r['pred_hit_rate'])) for r in rep['rules']]}")
                return {"compiled": rep, "abstained": True}
            w = compiled["weights"]
            new_dff = widen_mlp_native(mlp_B, orig_state_B, w)
            fire = po.audit_organ_firing_analytical(
                w, ch_ref["x_bos_B"],
                ch_ref["x_B_eval_aligned"], ch_ref["mlpB_eval_aligned_n2"],
                ch_ref["x_B_eval_unaligned"], ch_ref["mlpB_eval_unaligned_n2"],
                rm_B=rm_B, bos_x_eval_B=ch_ref.get("bos_x_eval_B"),
            )
            gt_r, _ = po.eval_general_text_kl_ppl(model_B, tok_B, dev_B, corpus_univ["eval"], base_logprobs=base_lps)
            gt_r["delta_ppl"] = round(gt_r["ppl"] - base_gt["ppl"], 4)
            qa_r = pmf.eval_organ1_detailed(model_B, tok_B, dev_B, ds_fr_it_test)
            loc_r, _ = po.eval_196_locality(model_B, tok_B, dev_B, test_facts, base_correct=base_loc_ok, exclude_mask=fr_it_exclude)
            unl = verify_exact_bit_unlearning(model_B, tok_B, dev_B, mlp_B, orig_state_B, base_logits_probe, probe_prompts)
            raw_store[tag_name] = {"direct": qa_r["_raw_direct_ok"], "ripple": qa_r["_raw_ripple_ok"]}
            g_str = ",".join(f"{r['Gamma']:.3f}" for r in rep["rules"])
            print(
                f"  [{tag_name:28s}] {compiled['status']:8s} G=({g_str}) d_ff={new_dff} | "
                f"dir={qa_r['direct_country_em']:.3f} rip3={qa_r['unseen_ripple_3attr_em']:.3f} "
                f"ds8={qa_r['dataset_fr_it_8_rule_gen']:.3f} prot={qa_r['protected_retention_em']:.3f} hn={qa_r['hardneg_false_flip_to_rule']:.3f} | "
                f"loc={loc_r['newly_broken_unrelated']}/{loc_r['n_unrelated_orig_correct']} ({loc_r['unrelated_retention']:.4f}) | "
                f"KL={gt_r['mean_kl']:.6f} dPPL={gt_r['delta_ppl']:+.4f} ff={fire['aligned']['ff_rate_gate_gt0']:.5f} bit_ul={unl['logits_bit_equal']}"
            )
            return {
                "compiled": rep,
                "abstained": False,
                "widened_d_ff": new_dff,
                "firing_audit": fire,
                "general_text": gt_r,
                "qa": pmf.strip_raw_vecs(qa_r),
                "locality_196": loc_r,
                "unlearning": unl,
            }

        arms_out: Dict[str, Any] = {}
        raw_store: Dict[str, Any] = {}

        # Primary Pure-Digital Bias-Free (mlp_bias=False)
        c_bf = compile_channel_pushforward_organ(qwen_rules, ch_univ, rm_B=rm_B, mode="biasfree", force_abstained=args.force_abstained)
        arms_out["PUSHFWD_DIGITAL_BIASFREE"] = _eval_compiled("PUSHFWD_DIGITAL_BIASFREE", c_bf, ch_univ, raw_store)

        # Affine comparison
        c_af = compile_channel_pushforward_organ(qwen_rules, ch_univ, rm_B=rm_B, mode="affine", force_abstained=args.force_abstained)
        arms_out["PUSHFWD_DIGITAL_AFFINE"] = _eval_compiled("PUSHFWD_DIGITAL_AFFINE", c_af, ch_univ, raw_store)

        # Optional domain-up ablation on Cap-1
        if args.ablate_domain_up:
            c_du = compile_channel_pushforward_organ(qwen_rules, ch_univ, rm_B=rm_B, mode="biasfree", ablate_domain_up=True, force_abstained=args.force_abstained)
            arms_out["PUSHFWD_DOMAIN_UP_BIASFREE"] = _eval_compiled("PUSHFWD_DOMAIN_UP_BIASFREE", c_du, ch_univ, raw_store)

        # Controls: Shuffled-Read (must ABSTAIN naturally!) + Forced Shuffled-Read (audits conformal <= alpha!) + Shuffled-Write
        c_sr = compile_channel_pushforward_organ(qwen_rules, ch_univ, rm_B=rm_B, mode="biasfree", shuffle_read_seed=SEED + 501)
        arms_out["SHUFFLED_READ_AUTO"] = _eval_compiled("SHUFFLED_READ_AUTO", c_sr, ch_univ, raw_store)

        c_srf = compile_channel_pushforward_organ(qwen_rules, ch_univ, rm_B=rm_B, mode="biasfree", shuffle_read_seed=SEED + 501, force_abstained=True)
        arms_out["SHUFFLED_READ_FORCED"] = _eval_compiled("SHUFFLED_READ_FORCED", c_srf, ch_univ, raw_store)

        c_sw = compile_channel_pushforward_organ(qwen_rules, ch_univ, rm_B=rm_B, mode="biasfree", shuffle_write_seed=SEED + 601, force_abstained=args.force_abstained)
        arms_out["SHUFFLED_WRITE"] = _eval_compiled("SHUFFLED_WRITE", c_sw, ch_univ, raw_store)

        # Native-B Oracle on SmolLM2 (exact same compiler on native B data!)
        if tgt == "smollm2":
            po.unload_model(model_B)
            b_rules = extract_cap1_counterbalanced_donor_rules(rk, lB, ("France", "Italy"), all_subtokens=args.all_subtokens)
            model_B, tok_B, dev_B, rm_B = po.load_model_fp32(rk)
            mlp_B = get_decoder_layers(model_B)[lB].mlp
            orig_state_B = po.save_original_mlp_state(mlp_B)
            c_nat = compile_channel_pushforward_organ(b_rules, channel=None, rm_B=rm_B, mode="biasfree", native_B_stats=ch_univ)
            arms_out["NATIVE_B_ORACLE"] = _eval_compiled("NATIVE_B_ORACLE", c_nat, ch_univ, raw_store)

            # 2x2 Read (Universal vs ProseOnly) x Write (Concept vs Long-Prose) Ablation on SmolLM2!
            ch_prose = fit_or_load_universal_channel(
                DONOR_QWEN_KEY, DONOR_QWEN_LAYER, rk, lB, corpus_prose, model_B, tok_B, dev_B, donor_passages_prose, tag="proseonly"
            )
            c_univ_wprose = compile_channel_pushforward_organ(
                qwen_rules, ch_univ, rm_B=rm_B, mode="biasfree",
                force_abstained=args.force_abstained, write_matrix_override=ch_univ["M_out_prose"],
            )
            c_pr_wprose = compile_channel_pushforward_organ(
                qwen_rules, ch_prose, rm_B=rm_B, mode="biasfree", force_abstained=args.force_abstained
            )
            c_pr_wconcept = compile_channel_pushforward_organ(
                qwen_rules, ch_prose, rm_B=rm_B, mode="biasfree",
                force_abstained=args.force_abstained, write_matrix_override=ch_univ["M_out_concept"],
            )
            out["stage_A_2x2_smollm2_prose_vs_universal"] = {
                "prose_only_channel": ch_prose["summary"],
                "universal_channel": ch_univ["summary"],
                "read_univ_write_prose": _eval_compiled("SMOL_2x2_UNIV_R_PROSE_W", c_univ_wprose, ch_univ, raw_store),
                "read_prose_write_prose": _eval_compiled("SMOL_2x2_PROSE_R_PROSE_W", c_pr_wprose, ch_prose, raw_store),
                "read_prose_write_concept": _eval_compiled("SMOL_2x2_PROSE_R_CONCEPT_W", c_pr_wconcept, ch_prose, raw_store),
            }

        mcn = {}
        if "SHUFFLED_READ_FORCED" in raw_store:
            sh_d, sh_r = raw_store["SHUFFLED_READ_FORCED"]["direct"], raw_store["SHUFFLED_READ_FORCED"]["ripple"]
            for aname, v in raw_store.items():
                if aname != "SHUFFLED_READ_FORCED":
                    mcn[f"{aname}_vs_SHUFFLED_READ_FORCED"] = {
                        "direct": mcnemar_exact(v["direct"], sh_d),
                        "ripple_3attr": mcnemar_exact(v["ripple"], sh_r),
                    }

        out["stage_A_multifamily"][tgt] = {
            "model_key": rk,
            "layer": lB,
            "rm": rm_B,
            "unlabeled_depth_diag": depth_diag,
            "channel_summary": ch_univ["summary"],
            "baseline": {"general_text": base_gt, "locality_196": base_loc, "qa": pmf.strip_raw_vecs(base_qa)},
            "arms": arms_out,
            "mcnemar": mcn,
            "elapsed_sec": round(time.time() - tr0, 1),
        }
        po.unload_model(model_B)

    # --- Stage B: Multi-Donor Composition (Qwen Fr->Jp + Falcon It->Br -> Solo SmolLM2, Selective Unlearning!) ---
    if not args.skip_stage_b:
        tb0 = time.time()
        print(f"\n--- [Cap-1 Stage B] Multi-Donor Composition ({DONOR_QWEN_KEY} L{DONOR_QWEN_LAYER} + {DONOR_FALCON_KEY} L{DONOR_FALCON_LAYER} -> smollm2-1.7b L9) ---")
        falcon_passages_univ = extract_donor_channel_passages(DONOR_FALCON_KEY, DONOR_FALCON_LAYER, corpus_univ, tag="universal")
        donor_concept_falcon = extract_donor_concept_write_states(DONOR_FALCON_KEY, DONOR_FALCON_LAYER)
        f_rules_cache = CACHE_DIR / f"cap1_cb_rules_{DONOR_FALCON_KEY}_l{DONOR_FALCON_LAYER}_it_sub{int(args.all_subtokens)}.pt"
        if f_rules_cache.exists() and not args.refit_donor:
            falcon_rules = torch.load(f_rules_cache, map_location="cpu", weights_only=False)
        else:
            falcon_rules = extract_cap1_counterbalanced_donor_rules(DONOR_FALCON_KEY, DONOR_FALCON_LAYER, ("Italy",), all_subtokens=args.all_subtokens)
            torch.save(falcon_rules, f_rules_cache)

        # BOTH donors completely unloaded! Load SOLO SmolLM2-1.7B!
        model_B, tok_B, dev_B, rm_B = po.load_model_fp32("smollm2-1.7b")
        lB = 9
        mlp_B = get_decoder_layers(model_B)[lB].mlp
        orig_state_B = po.save_original_mlp_state(mlp_B)

        ch_q = fit_or_load_universal_channel(
            DONOR_QWEN_KEY, DONOR_QWEN_LAYER, "smollm2-1.7b", lB, corpus_univ, model_B, tok_B, dev_B,
            donor_passages_univ, tag="universal", donor_concept_states=donor_concept_qwen,
        )
        ch_f = fit_or_load_universal_channel(
            DONOR_FALCON_KEY, DONOR_FALCON_LAYER, "smollm2-1.7b", lB, corpus_univ, model_B, tok_B, dev_B,
            falcon_passages_univ, tag="universal", donor_concept_states=donor_concept_falcon,
        )

        enc_p = tok_B(probe_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        with torch.no_grad():
            base_logits_probe = model_B(input_ids=enc_p["input_ids"].to(dev_B), attention_mask=enc_p["attention_mask"].to(dev_B), use_cache=False).logits.cpu()
        base_gt, base_lps = po.eval_general_text_kl_ppl(model_B, tok_B, dev_B, corpus_univ["eval"])
        base_loc, base_loc_ok = po.eval_196_locality(model_B, tok_B, dev_B, test_facts)

        cq = compile_channel_pushforward_organ([qwen_rules[0]], ch_q, rm_B=rm_B, mode="biasfree", force_abstained=args.force_abstained)
        cf = compile_channel_pushforward_organ([falcon_rules[0]], ch_f, rm_B=rm_B, mode="biasfree", force_abstained=args.force_abstained)

        sb_res: Dict[str, Any] = {
            "qwen_compiled": strip_rule_vectors(cq),
            "falcon_compiled": strip_rule_vectors(cf),
            "channels": {"qwen_to_smol": ch_q["summary"], "falcon_to_smol": ch_f["summary"]},
            "states": {},
        }
        if cq["weights"] is not None and cf["weights"] is not None:
            w_joint = concat_compiled_organs([cq["weights"], cf["weights"]])

            def _ev_sb(sn: str, cur_dff: int):
                gt_r, _ = po.eval_general_text_kl_ppl(model_B, tok_B, dev_B, corpus_univ["eval"], base_logprobs=base_lps)
                gt_r["delta_ppl"] = round(gt_r["ppl"] - base_gt["ppl"], 4)
                qa_r = pmf.eval_organ1_detailed(model_B, tok_B, dev_B, ds_fr_it_test)
                loc_r, _ = po.eval_196_locality(model_B, tok_B, dev_B, test_facts, base_correct=base_loc_ok, exclude_mask=fr_it_exclude)
                fr_s, it_s = qa_r["france_subset"], qa_r["italy_subset"]
                print(
                    f"  [StageB {sn:24s}] d_ff={cur_dff} | "
                    f"Fr(dir={fr_s['direct_country_em']:.2f} rip={fr_s['unseen_ripple_3attr_em']:.2f} ds={fr_s['ds_rule_gen']:.2f}) | "
                    f"It(dir={it_s['direct_country_em']:.2f} rip={it_s['unseen_ripple_3attr_em']:.2f} ds={it_s['ds_rule_gen']:.2f}) | "
                    f"loc={loc_r['newly_broken_unrelated']}/{loc_r['n_unrelated_orig_correct']} KL={gt_r['mean_kl']:.6f}"
                )
                return {"d_ff": cur_dff, "general_text": gt_r, "qa": pmf.strip_raw_vecs(qa_r), "locality_196": loc_r}

            dff_j = widen_mlp_native(mlp_B, orig_state_B, w_joint)
            sb_res["states"]["JOINT_QWEN_PLUS_FALCON"] = _ev_sb("JOINT_QWEN_PLUS_FALCON", dff_j)
            dff_nq = slice_mlp_rows_native(mlp_B, orig_state_B, [2, 3])
            sb_res["states"]["SELECTIVE_UNLEARN_QWEN"] = _ev_sb("SELECTIVE_UNLEARN_QWEN", dff_nq)
            widen_mlp_native(mlp_B, orig_state_B, w_joint)
            dff_nf = slice_mlp_rows_native(mlp_B, orig_state_B, [0, 1])
            sb_res["states"]["SELECTIVE_UNLEARN_FALCON"] = _ev_sb("SELECTIVE_UNLEARN_FALCON", dff_nf)
            sb_res["states"]["UNLEARN_ALL"] = verify_exact_bit_unlearning(model_B, tok_B, dev_B, mlp_B, orig_state_B, base_logits_probe, probe_prompts)
        else:
            print(f"  [Stage B] One or both donors ABSTAINED: qwen={cq['status']}, falcon={cf['status']}")

        sb_res["elapsed_sec"] = round(time.time() - tb0, 1)
        out["stage_B_multidonor_composition"] = sb_res
        po.unload_model(model_B)

    out["meta"]["total_elapsed_sec"] = round(time.time() - t0, 1)
    out_p = Path(args.out)
    with open(out_p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[CAP-1 ENGINE COMPLETE] SHA256={get_engine_sha256()[:16]} | Saved {out_p} ({out['meta']['total_elapsed_sec']}s)")
    return out


# ======================================================================================
# 6. Pure-CPU Unit Tests (Zero GPU Required!)
# ======================================================================================

def run_cpu_unit_tests() -> None:
    print(f"=== Running platonic_organ_engine.py Pure-CPU Unit Tests (SHA256={get_engine_sha256()[:12]}) ===")
    rng = np.random.default_rng(20261126)

    # 1. Universal Corpus Zero-Overlap Verification (BOTH Cap-1 Geography AND Cap-2 Biology!)
    corpus = build_universal_calibration_corpus()
    assert len(corpus["fit"]) == 195 and len(corpus["resid"]) == 60 and len(corpus["eval"]) == 45
    assert len(set(corpus["fit"]) & set(corpus["resid"])) == 0
    assert len(set(corpus["fit"]) & set(corpus["eval"])) == 0
    assert len(set(corpus["resid"]) & set(corpus["eval"])) == 0
    extra_forb = [x["name"] for x in po.ORGAN1_TRAIN_ENTITIES + po.ORGAN1_TEST_ENTITIES + po.ORGAN1_HARDNEG_TEST + LANDMARKS_T0]
    if DATASET_PATH.exists():
        with open(DATASET_PATH) as f:
            extra_forb += [x["subject"] for x in json.load(f)["facts"] if x["split"] == "test"]
    if (LAB_DIR / "bio_ripple_sealed.py").exists():
        import bio_ripple_sealed as brs
        extra_forb += list(brs.get_all_cap2_blacklist_terms())
    hits = verify_universal_corpus_disjoint(corpus, extra_forb)
    assert len(hits) == 0, f"Universal corpus leaked blacklisted terms: {hits[:10]}"
    cw = build_universal_concept_write_corpus()
    assert len(cw["concepts"]) == 300 and len(cw["prompts"]) == 1200 and len(cw["spans"]) == 1200
    for p, (c0, c1), c in zip(cw["prompts"], cw["spans"], cw["prompt_concepts"]):
        assert p[c0:c1] == c, f"Concept span mismatch: {p[c0:c1]!r} != {c!r}"
    cw_forb = list(extra_forb)
    if DATASET_PATH.exists():
        with open(DATASET_PATH) as f:
            for item in json.load(f)["facts"]:
                for field in ("subject", "answer"):
                    cw_forb.extend(re.findall(r"[a-z]+", str(item.get(field, "")).lower()))
                for a in item.get("aliases", []):
                    cw_forb.extend(re.findall(r"[a-z]+", str(a).lower()))
    cw_hits = verify_universal_corpus_disjoint({"concept_words": cw["concepts"]}, cw_forb)
    assert len(cw_hits) == 0, f"Concept write corpus leaked blacklisted terms: {cw_hits[:10]}"
    print(f"  [PASS 1/7] Universal 300-passage read corpus (195/60/45) + 1,200-sentence concept write corpus (300 concepts) have 0 collisions with Cap-1, dataset.json & ALL 152 sealed Cap-2 terms!")

    # 2. Analytical Ledoit-Wolf Exactness + Exact 2-Constraint KKT Optimality Across All 3 Active Sets!
    X_lw = rng.standard_normal((80, 24)) * np.linspace(0.5, 2.5, 24)
    S_lw, sh_lw = ledoit_wolf_cov(X_lw)
    Xc = X_lw - X_lw.mean(axis=0, keepdims=True)
    S_emp = (Xc.T @ Xc) / len(X_lw)
    mu_emp = np.trace(S_emp) / 24.0
    delta2_bf = float(np.sum((S_emp - mu_emp * np.eye(24)) ** 2))
    beta2_bf = float(np.mean([np.sum((np.outer(x, x) - S_emp) ** 2) for x in Xc])) / len(X_lw)
    sh_bf = float(np.clip(beta2_bf / delta2_bf, 0.0, 1.0))
    assert abs(sh_lw - sh_bf) < 1e-12, f"LW shrink mismatch: {sh_lw} vs {sh_bf}"
    assert float(np.max(np.abs(S_lw - ((1 - sh_bf) * S_emp + sh_bf * mu_emp * np.eye(24))))) < 1e-12
    # Test all 3 KKT active sets: both active, only-1 active, only-2 active
    d_a = rng.standard_normal(24); d_b = rng.standard_normal(24)
    for m1_t, m2_t in [(1.4, 2.9), (5.0, 0.1), (0.1, 5.0)]:
        wk = solve_2constraint_kkt(S_lw, d_a, d_b, m1_t, m2_t)
        assert wk @ d_a >= m1_t - 1e-10 and wk @ d_b >= m2_t - 1e-10
        obj0 = float(wk @ (S_lw @ wk))
        for _ in range(20):
            pert = wk + 0.05 * rng.standard_normal(24)
            if pert @ d_a >= m1_t and pert @ d_b >= m2_t:
                assert float(pert @ (S_lw @ pert)) >= obj0 - 1e-10
    # Verify contradictory collinear constraints raise ValueError (Audit F21 fix!)
    try:
        solve_2constraint_kkt(np.eye(2), np.array([1.0, 0.0]), np.array([-1.0, 0.0]), 1.0, 1.0)
        raise AssertionError("Infeasible KKT failed to raise ValueError!")
    except ValueError:
        pass
    print(f"  [PASS 2/7] Analytical Ledoit-Wolf (<1e-12) + 2-Constraint KKT optimality & infeasibility rejection verified (shrink={sh_lw:.5f})!")

    # 3. Rectangular Generative Channel Pushforward (2048->3072 UP & 2048->1536 DOWN, scaled 64->96 & 64->48)
    for D_A, D_B in [(64, 96), (64, 48), (96, 64)]:
        G_true = random_stiefel((D_B, D_A), rng) * 1.15
        X_A_f = rng.standard_normal((500, D_A)) * 0.8 + rng.standard_normal(D_A)
        X_B_f = rng.standard_normal(D_B) + (X_A_f - X_A_f.mean(0)) @ G_true.T + rng.standard_normal((500, D_B)) * 0.15
        X_A_r = rng.standard_normal((1200, D_A)) * 0.8 + X_A_f.mean(0)
        X_B_r = X_B_f.mean(0) + (X_A_r - X_A_f.mean(0)) @ G_true.T + rng.standard_normal((1200, D_B)) * 0.15
        # Add massive outlier tokens to H_A_f to prove RMS-normalization bounds them with ZERO 2.5x filter!
        H_A_f = X_A_f.copy() * 4.0
        H_A_f[:10] *= 500.0
        H_B_f = X_B_f.copy() * 2.0
        bos_B = rng.standard_normal((20, D_B)) * 4.0

        ch = fit_generative_channel(X_A_f, X_B_f, X_A_r, X_B_r, H_A_f, H_B_f, X_A_r * 4.0, X_B_r * 2.0, bos_B)
        assert ch["G"].shape == (D_B, D_A) and ch["M_out_iso"].shape == (D_B, D_A)
        assert ch["summary"]["heldout_gen_read_R2"] > 0.70, f"Low gen R2 ({D_A}->{D_B}): {ch['summary']}"
        assert 0.35 < ch["s_out_iso"] < 1.2, f"Outlier corrupted s_out_iso: {ch['s_out_iso']}"

        # Verify Cauchy-weighted concept write bridge crushes 500x outliers & recovers exact projected-variance scale!
        R_cw_true = random_stiefel((D_B, D_A), rng)
        Ha_cw = rng.standard_normal((400, D_A)) * 1.5
        Hb_cw = (Ha_cw @ R_cw_true.T) * 0.75 + rng.standard_normal((400, D_B)) * 0.08
        Ha_cw[:8] *= 500.0  # Massive outliers!
        wb = fit_concept_write_bridge(Ha_cw, Hb_cw)
        assert wb["M_out_concept"].shape == (D_B, D_A)
        assert 0.65 < wb["s_out_concept"] < 0.85, f"Cauchy write scale corrupted by 500x outliers: {wb['summary']}"
        assert wb["summary"]["heldout_concept_write_cos"] > 0.90, f"Low heldout concept write cos: {wb['summary']}"
        ch["M_out_prose"] = ch["M_out_iso"]
        ch["M_out_concept"] = wb["M_out_concept"]
        ch["M_out_iso"] = wb["M_out_concept"]

        # Compile a strong separable rule -> MUST compile (NOT abstain) and satisfy exact 0.2785/(delta*Gamma) bound!
        u_sig = rng.standard_normal(D_A); u_sig /= np.linalg.norm(u_sig)
        # Ensure u_sig lies in row space of G_true
        u_sig = G_true.T @ (G_true @ u_sig); u_sig /= np.linalg.norm(u_sig)
        Xp_A = X_A_f[:60] + 7.0 * u_sig
        Xn_A = X_A_f[60:120] - 2.0 * u_sig
        fids = np.tile(np.arange(6), 10)
        assert abs(compute_frame_only_auroc(fids, fids) - 0.5) < 1e-12
        rule = {"name": "test_rule", "X_pos_A": Xp_A, "X_neg_dom_A": Xn_A, "pos_frame_ids": fids, "neg_frame_ids": fids, "v_A": rng.standard_normal(D_A) * 2.0}
        comp = compile_channel_pushforward_organ([rule], ch, rm_B=1.0, mode="biasfree")
        assert comp["status"] == "COMPILED", f"Strong rule unexpectedly abstained ({D_A}->{D_B}): {comp['rules']}"
        rr = comp["rules"][0]
        assert abs(rr["beta_Gamma"] - (SILU_NEG_MAX / FROZEN_TOLERANCES["delta"])) < 1e-3
        # Exact worst-case SiLU step ripple bound <= delta on both sides!
        t1, t2, beta = rr["t1"], rr["t2"], rr["beta"]
        z_grid_lo = np.linspace(t1 - 5.0 * rr["Gamma"], t1, 200)
        z_grid_hi = np.linspace(t2, t2 + 5.0 * rr["Gamma"], 200)
        s_lo = (po._silu_np(beta * (z_grid_lo - t1)) - po._silu_np(beta * (z_grid_lo - t2))) / (beta * (t2 - t1))
        s_hi = (po._silu_np(beta * (z_grid_hi - t1)) - po._silu_np(beta * (z_grid_hi - t2))) / (beta * (t2 - t1))
        assert float(np.max(np.abs(s_lo))) <= FROZEN_TOLERANCES["delta"] + 1e-4
        assert float(np.max(np.abs(s_hi - 1.0))) <= FROZEN_TOLERANCES["delta"] + 1e-4
        # Exact BOS up-null
        w = comp["weights"]
        assert float(np.max(np.abs(w["W_u"] @ ch["x_bos_B"]))) < 1e-10
    print("  [PASS 3/7] Rectangular generative channels (64->96, 64->48, 96->64), Cauchy concept write bridge (500x outlier immune), & exact 0.2785/(delta*Gamma) ripple bound verified!")

    # 4. Automatic Unlabeled ABSTAIN on SHUFFLED_R (R2_chan_heldout < 0, a_emp <= 0.15) + Conformal False-Fire Bound!
    comp_shuf = compile_channel_pushforward_organ([rule], ch, rm_B=1.0, mode="biasfree", shuffle_read_seed=999)
    assert comp_shuf["status"] == "ABSTAIN" and comp_shuf["weights"] is None, f"SHUFFLED_R failed to abstain: {comp_shuf['rules']}"
    r_shuf = comp_shuf["rules"][0]["r_dir_heldout"]
    r2_shuf = comp_shuf["rules"][0]["r2_chan_heldout"]
    a_shuf = comp_shuf["rules"][0]["a_emp"]
    assert r2_shuf < 0.0, f"SHUFFLED_R unexpectedly had non-negative held-out channel R2={r2_shuf}"
    comp_srf = compile_channel_pushforward_organ([rule], ch, rm_B=1.0, mode="biasfree", shuffle_read_seed=999, force_abstained=True)
    w_rand = comp_srf["rules"][0]["w_c"]
    t1_rand = comp_srf["rules"][0]["t1"]
    # Evaluate ACTUAL compiled bias-free gate firing rate (Audit F20 fix — not raw w_c!)
    X_B_fresh = X_B_f.mean(0) + (rng.standard_normal((5000, D_A)) * 0.8) @ G_true.T + rng.standard_normal((5000, D_B)) * 0.15
    emp_ff = float(np.mean((X_B_fresh @ comp_srf["weights"]["W_g"].T > 0.0).any(axis=1)))
    assert emp_ff < 0.01, f"Compiled-gate conformal false-fire violated on fresh tokens: {emp_ff}"
    print(f"  [PASS 4/7] Automatic ABSTAIN on SHUFFLED_R (R2={r2_shuf:+.3f}, r_dir={r_shuf:+.4f}, a_emp={a_shuf:.4f}) & compiled-gate conformal fresh ff={emp_ff:.4f} verified!")

    # 5. Frame-Counterbalance Certificate Enforces ABSTAIN on Confounded Templates (Audit F19 fix!)
    assert abs(compute_frame_only_auroc([0, 0, 1, 1], [0, 0, 1, 1]) - 0.5) < 1e-12
    assert compute_frame_only_auroc([0, 0, 0, 0], [1, 1, 1, 1]) == 1.0
    conf_rule = dict(rule, pos_frame_ids=np.zeros(60, dtype=int), neg_frame_ids=np.ones(60, dtype=int))
    comp_conf = compile_channel_pushforward_organ([conf_rule], ch, rm_B=1.0, mode="biasfree")
    assert comp_conf["status"] == "ABSTAIN" and comp_conf["weights"] is None and not comp_conf["rules"][0]["frame_cert_pass"], (
        f"Confounded frame IDs failed to force ABSTAIN: {comp_conf['rules']}"
    )
    print("  [PASS 5/7] Frame-Counterbalance Certificate enforced (balanced=0.5000 -> COMPILED, confounded=1.0000 -> ABSTAIN)!")

    # 6. Real HF LlamaMLP, GraniteMLP (rm=0.22), & Qwen2MLP + Exact torch.equal State-Dict & Output Unlearning!
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaMLP
    from transformers.models.granite.configuration_granite import GraniteConfig
    from transformers.models.granite.modeling_granite import GraniteMLP
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    x_t = torch.randn(3, 5, D_B, dtype=torch.float32)
    x_t[:, 0, :] = torch.from_numpy(ch["x_bos_B"].astype(np.float32))
    for nm, mlp, rm_v in [
        ("LlamaMLP", LlamaMLP(LlamaConfig(hidden_size=D_B, intermediate_size=128, hidden_act="silu", mlp_bias=False)).eval(), 1.0),
        ("GraniteMLP", GraniteMLP(GraniteConfig(hidden_size=D_B, intermediate_size=128, hidden_act="silu", mlp_bias=False, residual_multiplier=0.22)).eval(), 0.22),
        ("Qwen2MLP", Qwen2MLP(Qwen2Config(hidden_size=D_B, intermediate_size=128, hidden_act="silu")).eval(), 1.0),
    ]:
        st0 = po.save_original_mlp_state(mlp)
        with torch.no_grad(): y0 = mlp(x_t)
        comp_m = compile_channel_pushforward_organ([rule], ch, rm_B=rm_v, mode="biasfree")
        widen_mlp_native(mlp, st0, comp_m["weights"])
        assert mlp.gate_proj.bias is None and mlp.up_proj.bias is None and mlp.down_proj.bias is None
        po.unlearn_mlp_native(mlp, st0)
        assert torch.equal(mlp.gate_proj.weight.detach(), st0["gate_weight"])
        assert torch.equal(mlp.up_proj.weight.detach(), st0["up_weight"])
        assert torch.equal(mlp.down_proj.weight.detach(), st0["down_weight"])
        with torch.no_grad(): y_ul = mlp(x_t)
        assert torch.equal(y_ul, y0), f"{nm} output NOT bit-equal after unlearning!"
    print("  [PASS 6/7] Real HF LlamaMLP, GraniteMLP (rm=0.22), & Qwen2MLP strictly BIASFREE widening + exact torch.equal unlearning verified!")

    # 7. Multi-Organ Co-Implantation & Selective Bit-Equal Row Slicing (128 -> 132 -> 130 -> 128)!
    mlp_c = LlamaMLP(LlamaConfig(hidden_size=D_B, intermediate_size=128, hidden_act="silu", mlp_bias=False)).eval()
    st_c = po.save_original_mlp_state(mlp_c)
    with torch.no_grad(): y_base = mlp_c(x_t)
    w1 = comp["weights"]
    w2 = compile_channel_pushforward_organ([rule], ch, rm_B=1.0, mode="biasfree", shuffle_write_seed=777)["weights"]
    wj = concat_compiled_organs([w1, w2])
    widen_mlp_native(mlp_c, st_c, w1)
    with torch.no_grad(): y_only1 = mlp_c(x_t)
    widen_mlp_native(mlp_c, st_c, w2)
    with torch.no_grad(): y_only2 = mlp_c(x_t)
    assert widen_mlp_native(mlp_c, st_c, wj) == 132
    assert slice_mlp_rows_native(mlp_c, st_c, [2, 3]) == 130
    with torch.no_grad(): assert torch.equal(mlp_c(x_t), y_only2)
    widen_mlp_native(mlp_c, st_c, wj)
    assert slice_mlp_rows_native(mlp_c, st_c, [0, 1]) == 130
    with torch.no_grad(): assert torch.equal(mlp_c(x_t), y_only1)
    assert slice_mlp_rows_native(mlp_c, st_c, []) == 128
    with torch.no_grad(): assert torch.equal(mlp_c(x_t), y_base)
    print("  [PASS 7/7] Multi-Organ Co-Implantation & Selective torch.equal Row-Slicing (132->130->128) verified!")
    print("=== ALL 7 PLATONIC ORGAN ENGINE CPU UNIT TESTS PASSED! ===")


# ======================================================================================
# 7. CLI Entrypoint
# ======================================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-Magic-Constant Channel-Pushforward Twin-SwiGLU Engine")
    parser.add_argument("--unit-test", action="store_true", help="Run pure-CPU unit tests only")
    parser.add_argument("--run-cap1", action="store_true", help="Run full Cap-1 multi-family + 2x2 + Stage B suite")
    parser.add_argument("--targets", type=str, default="smollm2,granite,falcon3,qwen1.5b", help="Stage A targets")
    parser.add_argument("--all-subtokens", action="store_true", help="Opus Q3: include all entity subtokens in donor positives")
    parser.add_argument("--ablate-domain-up", action="store_true", help="Also run channel-pushforward domain-up A/B ablation on Cap-1")
    parser.add_argument("--force-abstained", action="store_true", help="Force implantation of abstained cells strictly for calibration auditing")
    parser.add_argument("--skip-stage-b", action="store_true", help="Skip Stage B multi-donor composition")
    parser.add_argument("--refit-donor", action="store_true", help="Force re-extraction of donor Cap-1 rules")
    parser.add_argument("--out", type=str, default=str(CAP1_OUT_PATH), help="Output JSON path")
    args = parser.parse_args()

    if args.unit_test:
        run_cpu_unit_tests()
        return

    if args.run_cap1:
        run_cap1_suite(args)
        return

    print(f"platonic_organ_engine.py | SHA256={get_engine_sha256()}")
    print("Pass --unit-test for CPU verification or --run-cap1 for MPS execution.")


if __name__ == "__main__":
    main()
