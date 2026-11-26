#!/usr/bin/env python3
"""
pilot_graft_t3_gap_scaled.py — Pillar 3c: Scaled Fable T3 Long-Tail Knowledge-Gap Transplant
=============================================================================================

Scales Fable T3 (`pilot_graft_t3_gap.py`, originally n=17) to n >= 60 (targeting 70–100+)
genuinely obscure real-world `A-right, B-wrong` entities across 20 diverse countries!

Design & Zero-Leakage Guarantees:
  1. Curates ~270 genuinely obscure real-world historic sites, archaeological ruins,
     monasteries/temples, castles/forts, small historic towns, and natural landmarks across
     20 countries — 100% disjoint from ALL `dataset.json` facts and ALL `ORGAN1_*` entities,
     with zero country-adjective giveaways in any entity name.
  2. Uses a bit-identical neutral shared prefix `"In geography, {ent}"` across ALL 6 prompts:
       - `country`, `capital`, `currency`, `language`, `continent` (5 geographic attributes)
       - `protected` (non-geographic entity-type retention: town/temple/castle/ruin/park)
  3. Reuses ALREADY-CACHED Zero-QA `N=10,000` full-rank `O(d)` maps (ZERO new map fitting!):
       - Default `--pair qwen2.5-3b->smollm2-1.7b`: `cache/pilot1_maps_qwen2.5-3b_to_smollm2-1.7b_l17_9.pt`
       - Optional `--pair qwen2.5-3b->granite-3.1-2b`: `cache/pilot1_maps_qwen2.5-3b_to_granite-3.1-2b_l18_8.pt` (float32!)
  4. Evaluates 5 conditions on every screened `A-right, B-wrong` gap entity:
       - `p1_10k_procrustes_matched` (Zero-QA N=10k full-rank O(d) Procrustes)
       - `p1_10k_fisher_orth_matched` (Zero-QA N=10k O(d) Fisher-Orthogonal Procrustes)
       - `ctrl_B_self_roundtrip` (B's own unrotated state -> must stay 0.0%!)
       - `ctrl_shuffled_R` (norm-matched random orthogonal rotation R_rand)
       - `ctrl_wrong_entity_e_prime` (A's vector for a different gap entity e' -> must track e'!)
  5. Reports 3 clean cohorts (separating any province-first outputs from true blind spots!):
       - `all_gap_entities`: all screened `A_country=1, B_country=0` entities
       - `strict_blindspot_gap`: `A_country=1, B_country=0` AND B's generation does NOT contain
         the true country name anywhere AND B also misses >= 1 of {capital, currency, language}
       - `strict_A_right_B_wrong_per_attr`: per-attribute strict gaps (`A_attr=1, B_attr=0`)
  6. Computes exact Wilson 95% CIs, entity-clustered bootstrap 95% CIs (n_boot=5000), and
     exact two-sided McNemar p-values (`10k_procrustes` & `10k_fisher_orth` vs `B_self_roundtrip`,
     vs `shuffled_R`, and vs `wrong_entity e'`).

Usage:
  python pilot_graft_t3_gap_scaled.py --unit-test
  python pilot_graft_t3_gap_scaled.py --pair qwen2.5-3b->smollm2-1.7b
  python pilot_graft_t3_gap_scaled.py --pair qwen2.5-3b->granite-3.1-2b
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import binomtest

LAB_DIR = Path(__file__).resolve().parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from extract_and_baseline import CACHE_DIR, DATASET_PATH, SEED
from pilot_graft_falsify import (
    _polar_f64,
    eval_prompts_patched,
    extract_multi_layer_states,
    load_model_f32,
)
from pilot_graft_t3_gap import ALL_COUNTRIES

PAIR_CONFIGS: Dict[str, Dict[str, Any]] = {
    "qwen2.5-3b->smollm2-1.7b": {
        "donor": "qwen2.5-3b",
        "recipient": "smollm2-1.7b",
        "lA": 17,
        "lB": 9,
        "map_path": CACHE_DIR / "pilot1_maps_qwen2.5-3b_to_smollm2-1.7b_l17_9.pt",
        "out_path": LAB_DIR / "results_graft_t3_gap_scaled.json",
    },
    "qwen2.5-3b->granite-3.1-2b": {
        "donor": "qwen2.5-3b",
        "recipient": "granite-3.1-2b",
        "lA": 18,
        "lB": 8,
        "map_path": CACHE_DIR / "pilot1_maps_qwen2.5-3b_to_granite-3.1-2b_l18_8.pt",
        "out_path": LAB_DIR / "results_graft_t3_gap_scaled_granite.json",
    },
}

# Protected entity-type aliases (for non-geographic identity/category retention)
PROTECTED_KIND_ALIASES: Dict[str, List[str]] = {
    "town": [
        "city", "town", "village", "commune", "municipality", "port", "settlement",
        "resort", "borough", "historic town", "medieval town", "coastal town", "spa town",
    ],
    "castle": [
        "castle", "fortress", "fort", "palace", "citadel", "residence", "chateau",
        "château", "alcazar", "alcázar", "prison", "royal palace", "medieval castle",
    ],
    "temple": [
        "temple", "shrine", "monastery", "abbey", "cathedral", "church", "basilica",
        "mosque", "stupa", "convent", "sanctuary", "pagoda", "caves", "cave", "chapel",
        "buddhist", "hindu", "shinto", "religious",
    ],
    "historic_site": [
        "ruin", "ruins", "archaeological", "ancient", "historic", "historical", "site",
        "monument", "aqueduct", "theatre", "theater", "arena", "tower", "mine", "village",
        "settlement", "city", "town", "palace", "museum", "complex", "fortress", "citadel",
        "temple", "pyramid", "necropolis", "tomb", "sanctuary", "oasis", "caves", "cave",
        "world heritage", "unesco", "landmark", "bridge", "stepwell", "district", "quarter",
    ],
    "park": [
        "national park", "park", "gorge", "canyon", "valley", "forest", "desert", "wetland",
        "reserve", "peninsula", "oasis", "reef", "pass", "plateau", "mountain", "mountains",
        "archipelago", "island", "islands", "fjord", "glacier", "lake", "river", "waterfall",
        "region", "nature", "wilderness", "biosphere", "volcano",
    ],
}

# ~265 genuinely obscure real-world historic sites, archaeological ruins, temples/monasteries,
# castles/palaces, small historic towns, and natural landmarks across 20 diverse countries!
# Strictly disjoint from dataset.json and ORGAN1_* entities; zero country-adjective giveaways.
SCALED_OBSCURE_ENTITIES: List[Tuple[str, str, str]] = [
    # 1. France (14)
    ("Fontenay Abbey", "temple", "France"),
    ("Rocamadour", "town", "France"),
    ("Pont du Gard", "historic_site", "France"),
    ("Albi Cathedral", "temple", "France"),
    ("Chateau de Chenonceau", "castle", "France"),
    ("Palais des Papes", "castle", "France"),
    ("Sarlat-la-Caneda", "town", "France"),
    ("Vezelay", "town", "France"),
    ("Beaune", "town", "France"),
    ("Cahors", "town", "France"),
    ("Uzes", "town", "France"),
    ("Concarneau", "town", "France"),
    ("Verdon Gorge", "park", "France"),
    ("Camargue", "park", "France"),

    # 2. Italy (14)
    ("Castel del Monte", "castle", "Italy"),
    ("Paestum", "historic_site", "Italy"),
    ("Sassi di Matera", "historic_site", "Italy"),
    ("Certosa di Pavia", "temple", "Italy"),
    ("Basilica of San Vitale", "temple", "Italy"),
    ("Palazzo Te", "castle", "Italy"),
    ("Villa d'Este", "castle", "Italy"),
    ("Urbino", "town", "Italy"),
    ("Orvieto", "town", "Italy"),
    ("Spoleto", "town", "Italy"),
    ("Alberobello", "town", "Italy"),
    ("Volterra", "town", "Italy"),
    ("Cividale del Friuli", "town", "Italy"),
    ("Gran Paradiso", "park", "Italy"),

    # 3. Spain (14)
    ("El Escorial", "castle", "Spain"),
    ("Segovia Aqueduct", "historic_site", "Spain"),
    ("Mezquita of Cordoba", "temple", "Spain"),
    ("Alcazar of Segovia", "castle", "Spain"),
    ("Burgos Cathedral", "temple", "Spain"),
    ("Tarraco", "historic_site", "Spain"),
    ("Ubeda", "town", "Spain"),
    ("Baeza", "town", "Spain"),
    ("Trujillo", "town", "Spain"),
    ("Albarracin", "town", "Spain"),
    ("Ronda", "town", "Spain"),
    ("Cadaques", "town", "Spain"),
    ("Bardenas Reales", "park", "Spain"),
    ("Ordesa Valley", "park", "Spain"),

    # 4. Germany (13)
    ("Wartburg Castle", "castle", "Germany"),
    ("Quedlinburg", "town", "Germany"),
    ("Sanssouci Palace", "castle", "Germany"),
    ("Hohenzollern Castle", "castle", "Germany"),
    ("Speyer Cathedral", "temple", "Germany"),
    ("Wurzburg Residence", "castle", "Germany"),
    ("Lorsch Abbey", "temple", "Germany"),
    ("Goslar", "town", "Germany"),
    ("Rothenburg ob der Tauber", "town", "Germany"),
    ("Meissen", "town", "Germany"),
    ("Stralsund", "town", "Germany"),
    ("Wismar", "town", "Germany"),
    ("Spreewald", "park", "Germany"),

    # 5. Japan (14)
    ("Matsumoto Castle", "castle", "Japan"),
    ("Nikko Toshogu", "temple", "Japan"),
    ("Itsukushima Shrine", "temple", "Japan"),
    ("Todai-ji", "temple", "Japan"),
    ("Byodo-in", "temple", "Japan"),
    ("Shirakawa-go", "historic_site", "Japan"),
    ("Nijo Castle", "castle", "Japan"),
    ("Takayama Jinya", "historic_site", "Japan"),
    ("Kurashiki", "town", "Japan"),
    ("Onomichi", "town", "Japan"),
    ("Kakunodate", "town", "Japan"),
    ("Tsumago", "town", "Japan"),
    ("Yakushima", "park", "Japan"),
    ("Daisetsuzan", "park", "Japan"),

    # 6. Brazil (13)
    ("Ouro Preto", "town", "Brazil"),
    ("Olinda", "town", "Brazil"),
    ("Paraty", "town", "Brazil"),
    ("Congonhas", "historic_site", "Brazil"),
    ("Sao Miguel das Missoes", "historic_site", "Brazil"),
    ("Inhotim", "historic_site", "Brazil"),
    ("Tiradentes", "town", "Brazil"),
    ("Diamantina", "town", "Brazil"),
    ("Pirenopolis", "town", "Brazil"),
    ("Morretes", "town", "Brazil"),
    ("Alcantara", "town", "Brazil"),
    ("Chapada dos Veadeiros", "park", "Brazil"),
    ("Lencois Maranhenses", "park", "Brazil"),

    # 7. India (14)
    ("Hampi", "historic_site", "India"),
    ("Khajuraho", "temple", "India"),
    ("Ellora Caves", "temple", "India"),
    ("Ajanta Caves", "temple", "India"),
    ("Konark Sun Temple", "temple", "India"),
    ("Fatehpur Sikri", "historic_site", "India"),
    ("Sanchi Stupa", "temple", "India"),
    ("Mahabalipuram", "historic_site", "India"),
    ("Rani ki Vav", "historic_site", "India"),
    ("Brihadisvara Temple", "temple", "India"),
    ("Golconda Fort", "castle", "India"),
    ("Orchha", "town", "India"),
    ("Pattadakal", "historic_site", "India"),
    ("Kaziranga", "park", "India"),

    # 8. Egypt (13)
    ("Abydos Temple", "temple", "Egypt"),
    ("Abu Simbel", "temple", "Egypt"),
    ("Philae Temple", "temple", "Egypt"),
    ("Edfu Temple", "temple", "Egypt"),
    ("Dendera Temple", "temple", "Egypt"),
    ("Kom Ombo", "temple", "Egypt"),
    ("Saqqara", "historic_site", "Egypt"),
    ("Siwa Oasis", "park", "Egypt"),
    ("Esna", "town", "Egypt"),
    ("Qena", "town", "Egypt"),
    ("Dakhla", "park", "Egypt"),
    ("El Alamein", "town", "Egypt"),
    ("Wadi El Rayan", "park", "Egypt"),

    # 9. Greece (13)
    ("Delos", "historic_site", "Greece"),
    ("Epidaurus", "historic_site", "Greece"),
    ("Mycenae", "historic_site", "Greece"),
    ("Knossos", "historic_site", "Greece"),
    ("Meteora", "temple", "Greece"),
    ("Mystras", "historic_site", "Greece"),
    ("Vergina", "historic_site", "Greece"),
    ("Lindos", "town", "Greece"),
    ("Nafplio", "town", "Greece"),
    ("Monemvasia", "town", "Greece"),
    ("Kastoria", "town", "Greece"),
    ("Samaria Gorge", "park", "Greece"),
    ("Vikos Gorge", "park", "Greece"),

    # 10. Mexico (13)
    ("Palenque", "historic_site", "Mexico"),
    ("Monte Alban", "historic_site", "Mexico"),
    ("Uxmal", "historic_site", "Mexico"),
    ("Calakmul", "historic_site", "Mexico"),
    ("El Tajin", "historic_site", "Mexico"),
    ("Mitla", "historic_site", "Mexico"),
    ("Cholula", "town", "Mexico"),
    ("Taxco", "town", "Mexico"),
    ("Patzcuaro", "town", "Mexico"),
    ("Izamal", "town", "Mexico"),
    ("Real de Catorce", "town", "Mexico"),
    ("Tepoztlan", "town", "Mexico"),
    ("Sumidero Canyon", "park", "Mexico"),

    # 11. Turkey (13)
    ("Gobekli Tepe", "historic_site", "Turkey"),
    ("Hattusa", "historic_site", "Turkey"),
    ("Nemrut Dagi", "historic_site", "Turkey"),
    ("Aphrodisias", "historic_site", "Turkey"),
    ("Pergamon", "historic_site", "Turkey"),
    ("Sumela Monastery", "temple", "Turkey"),
    ("Safranbolu", "town", "Turkey"),
    ("Amasya", "town", "Turkey"),
    ("Mardin", "town", "Turkey"),
    ("Midyat", "town", "Turkey"),
    ("Hasankeyf", "historic_site", "Turkey"),
    ("Cumalikizik", "town", "Turkey"),
    ("Ihlara Valley", "park", "Turkey"),

    # 12. Australia (13)
    ("Port Arthur", "historic_site", "Australia"),
    ("Fremantle Prison", "castle", "Australia"),
    ("Purnululu", "park", "Australia"),
    ("Karijini", "park", "Australia"),
    ("Freycinet", "park", "Australia"),
    ("Daintree", "park", "Australia"),
    ("Broome", "town", "Australia"),
    ("Kalgoorlie", "town", "Australia"),
    ("Echuca", "town", "Australia"),
    ("Beechworth", "town", "Australia"),
    ("Busselton", "town", "Australia"),
    ("Esperance", "town", "Australia"),
    ("Flinders Ranges", "park", "Australia"),

    # 13. Peru (13)
    ("Sacsayhuaman", "historic_site", "Peru"),
    ("Chan Chan", "historic_site", "Peru"),
    ("Ollantaytambo", "historic_site", "Peru"),
    ("Chavin de Huantar", "historic_site", "Peru"),
    ("Caral", "historic_site", "Peru"),
    ("Kuelap", "historic_site", "Peru"),
    ("Moray", "historic_site", "Peru"),
    ("Chachapoyas", "town", "Peru"),
    ("Huaraz", "town", "Peru"),
    ("Pisac", "town", "Peru"),
    ("Urubamba", "town", "Peru"),
    ("Colca Canyon", "park", "Peru"),
    ("Paracas", "park", "Peru"),

    # 14. Morocco (13)
    ("Volubilis", "historic_site", "Morocco"),
    ("Ait Benhaddou", "historic_site", "Morocco"),
    ("Chefchaouen", "town", "Morocco"),
    ("El Badi Palace", "castle", "Morocco"),
    ("Hassan Tower", "historic_site", "Morocco"),
    ("Taroudant", "town", "Morocco"),
    ("Tiznit", "town", "Morocco"),
    ("Asilah", "town", "Morocco"),
    ("Moulay Idriss", "town", "Morocco"),
    ("Azrou", "town", "Morocco"),
    ("Zagora", "town", "Morocco"),
    ("Todra Gorge", "park", "Morocco"),
    ("Dades Gorge", "park", "Morocco"),

    # 15. Thailand (13)
    ("Ayutthaya", "historic_site", "Thailand"),
    ("Sukhothai", "historic_site", "Thailand"),
    ("Phimai", "historic_site", "Thailand"),
    ("Phanom Rung", "temple", "Thailand"),
    ("Wat Rong Khun", "temple", "Thailand"),
    ("Lampang", "town", "Thailand"),
    ("Mae Hong Son", "town", "Thailand"),
    ("Lopburi", "town", "Thailand"),
    ("Phetchaburi", "town", "Thailand"),
    ("Chanthaburi", "town", "Thailand"),
    ("Khao Sok", "park", "Thailand"),
    ("Doi Inthanon", "park", "Thailand"),
    ("Khao Yai", "park", "Thailand"),

    # 16. Poland (13)
    ("Malbork Castle", "castle", "Poland"),
    ("Wieliczka Salt Mine", "historic_site", "Poland"),
    ("Wawel Castle", "castle", "Poland"),
    ("Bialowieza Forest", "park", "Poland"),
    ("Zamosc", "town", "Poland"),
    ("Sandomierz", "town", "Poland"),
    ("Kazimierz Dolny", "town", "Poland"),
    ("Frombork", "town", "Poland"),
    ("Chelmno", "town", "Poland"),
    ("Przemysl", "town", "Poland"),
    ("Sanok", "town", "Poland"),
    ("Biebrza", "park", "Poland"),
    ("Slowinski", "park", "Poland"),

    # 17. Portugal (13)
    ("Batalha Monastery", "temple", "Portugal"),
    ("Alcobaca Monastery", "temple", "Portugal"),
    ("Tomar Convent", "temple", "Portugal"),
    ("Pena Palace", "castle", "Portugal"),
    ("Guimaraes Castle", "castle", "Portugal"),
    ("Obidos", "town", "Portugal"),
    ("Marvao", "town", "Portugal"),
    ("Monsaraz", "town", "Portugal"),
    ("Tavira", "town", "Portugal"),
    ("Silves", "town", "Portugal"),
    ("Lamego", "town", "Portugal"),
    ("Elvas", "town", "Portugal"),
    ("Peneda-Geres", "park", "Portugal"),

    # 18. Norway (13)
    ("Bryggen", "historic_site", "Norway"),
    ("Urnes Stave Church", "temple", "Norway"),
    ("Borgund Stave Church", "temple", "Norway"),
    ("Nidaros Cathedral", "temple", "Norway"),
    ("Roros", "town", "Norway"),
    ("Lillehammer", "town", "Norway"),
    ("Kongsberg", "town", "Norway"),
    ("Svolvaer", "town", "Norway"),
    ("Henningsvaer", "town", "Norway"),
    ("Kirkenes", "town", "Norway"),
    ("Jotunheimen", "park", "Norway"),
    ("Rondane", "park", "Norway"),
    ("Hardangervidda", "park", "Norway"),

    # 19. South Korea (13)
    ("Gyeongbokgung", "castle", "South Korea"),
    ("Bulguksa", "temple", "South Korea"),
    ("Seokguram", "temple", "South Korea"),
    ("Hwaseong Fortress", "castle", "South Korea"),
    ("Changdeokgung", "castle", "South Korea"),
    ("Hahoe Folk Village", "historic_site", "South Korea"),
    ("Jongmyo", "temple", "South Korea"),
    ("Andong", "town", "South Korea"),
    ("Gyeongju", "town", "South Korea"),
    ("Tongyeong", "town", "South Korea"),
    ("Buyeo", "town", "South Korea"),
    ("Seoraksan", "park", "South Korea"),
    ("Jirisan", "park", "South Korea"),

    # 20. Argentina (13)
    ("Talampaya", "park", "Argentina"),
    ("Ischigualasto", "park", "Argentina"),
    ("Quebrada de Humahuaca", "park", "Argentina"),
    ("Cueva de las Manos", "historic_site", "Argentina"),
    ("Valdes Peninsula", "park", "Argentina"),
    ("Tilcara", "town", "Argentina"),
    ("Purmamarca", "town", "Argentina"),
    ("Cafayate", "town", "Argentina"),
    ("El Calafate", "town", "Argentina"),
    ("El Chalten", "town", "Argentina"),
    ("Puerto Madryn", "town", "Argentina"),
    ("Esquel", "town", "Argentina"),
    ("Nahuel Huapi", "park", "Argentina"),
]


def _make_prompts(ent: str, kind: str, country: str) -> Dict[str, dict]:
    """Build bit-identical shared-prefix prompts across all 5 geo attrs + protected entity-type attr."""
    c_info = ALL_COUNTRIES[country]
    pref = f"In geography, {ent}"
    cspan = (len("In geography, "), len(pref))
    return {
        "country": {"prompt": f"{pref} is located in the country of", "span": cspan, "aliases": c_info["aliases"]},
        "capital": {"prompt": f"{pref} is located in a country whose capital city is", "span": cspan, "aliases": c_info["capital"][1]},
        "currency": {"prompt": f"{pref} is located in a country whose official currency is the", "span": cspan, "aliases": c_info["currency"][1]},
        "language": {"prompt": f"{pref} is located in a country whose primary official language is", "span": cspan, "aliases": c_info["language"][1]},
        "continent": {"prompt": f"{pref} is located on the continent of", "span": cspan, "aliases": c_info["continent"][1]},
        "protected": {"prompt": f"{pref} is best known as a", "span": cspan, "aliases": PROTECTED_KIND_ALIASES[kind]},
    }


# =============================================================================
# Exact Statistical Rigor: Wilson 95% CI, Entity-Clustered Bootstrap, McNemar
# =============================================================================


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[Optional[float], Optional[float]]:
    """Exact Wilson score 95% confidence interval for binomial proportion k/n."""
    if n <= 0:
        return None, None
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p)) / n + (z * z) / (4.0 * n * n))
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def entity_bootstrap_ci(
    per_entity_values: Sequence[float],
    n_boot: int = 5000,
    seed: int = SEED + 404,
) -> Tuple[Optional[float], Optional[float]]:
    """Entity-clustered percentile bootstrap 95% CI (resampling entities with replacement)."""
    vals = np.asarray(per_entity_values, dtype=np.float64)
    n = len(vals)
    if n == 0:
        return None, None
    if n == 1:
        v = round(float(vals[0]), 4)
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = vals[idx].mean(axis=1)
    return round(float(np.quantile(boots, 0.025)), 4), round(float(np.quantile(boots, 0.975)), 4)


def exact_mcnemar(a: Sequence[int], b: Sequence[int]) -> Dict[str, Any]:
    """Exact two-sided McNemar test via exact binomial test on discordant pairs (n10 vs n01)."""
    aa = np.asarray(a, dtype=np.int64)
    bb = np.asarray(b, dtype=np.int64)
    assert len(aa) == len(bb)
    n10 = int(((aa == 1) & (bb == 0)).sum())
    n01 = int(((aa == 0) & (bb == 1)).sum())
    n_disc = n10 + n01
    pval = float(binomtest(n10, n_disc, 0.5, alternative="two-sided").pvalue) if n_disc > 0 else 1.0
    return {
        "n": int(len(aa)),
        "acc_a": round(float(aa.mean()), 4) if len(aa) else None,
        "acc_b": round(float(bb.mean()), 4) if len(bb) else None,
        "diff_pp": round(100.0 * float(aa.mean() - bb.mean()), 2) if len(aa) else None,
        "n10_a_only": n10,
        "n01_b_only": n01,
        "p_value": pval,
    }


def _gen_mentions_country(gen_text: str, country: str) -> bool:
    gl = gen_text.lower()
    for al in ALL_COUNTRIES[country]["aliases"]:
        if al.lower() in gl:
            return True
    return False


# =============================================================================
# Unit Tests (Strictly CPU, Zero GPU!)
# =============================================================================


def run_unit_test() -> None:
    print("[unit-test] Checking SCALED_OBSCURE_ENTITIES, zero-overlap, prompts, stats, and caches...")
    assert len(SCALED_OBSCURE_ENTITIES) >= 220, f"Expected >=220 entities, got {len(SCALED_OBSCURE_ENTITIES)}"

    # 1. Zero duplicates & >=16 distinct countries
    names_lower = [e.lower() for e, _, _ in SCALED_OBSCURE_ENTITIES]
    dups = [k for k, c in Counter(names_lower).items() if c > 1]
    assert not dups, f"Duplicate entities found: {dups}"
    countries = sorted({c for _, _, c in SCALED_OBSCURE_ENTITIES})
    assert len(countries) >= 16, f"Expected >=16 countries, got {len(countries)}"

    # 2. Zero overlap with dataset.json (ALL subjects & answers) and ORGAN1_* entities
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        ds_facts = json.load(f)["facts"]
    forbidden = {x["subject"].lower() for x in ds_facts} | {x["answer"].lower() for x in ds_facts}

    import pilot_organs_p0 as po
    for attr in ("ORGAN1_TRAIN_ENTITIES", "ORGAN1_TEST_ENTITIES", "ORGAN1_HARDNEG_TRAIN", "ORGAN1_HARDNEG_TEST"):
        if hasattr(po, attr):
            for item in getattr(po, attr):
                forbidden.add(item["name"].lower())

    overlaps = [e for e, _, _ in SCALED_OBSCURE_ENTITIES if e.lower() in forbidden]
    assert not overlaps, f"Overlap with dataset.json or ORGAN1_*: {overlaps}"

    # 3. Valid country metadata, zero country-name giveaways, valid protected aliases & spans
    for ent, kind, c in SCALED_OBSCURE_ENTITIES:
        assert c in ALL_COUNTRIES, f"Missing country metadata: {c}"
        assert kind in PROTECTED_KIND_ALIASES, f"Invalid kind: {kind}"
        assert c.lower() not in ent.lower(), f"Country giveaway in entity name: {ent} ({c})"
        pm = _make_prompts(ent, kind, c)
        assert set(pm.keys()) == {"country", "capital", "currency", "language", "continent", "protected"}
        for ak, ad in pm.items():
            s0, s1 = ad["span"]
            assert ad["prompt"][s0:s1] == ent, f"Span mismatch for {ent} ({ak}): {ad['prompt'][s0:s1]!r}"
            assert len(ad["aliases"]) >= 1

    # 4. Exact statistical helper unit tests
    lo, hi = wilson_ci(80, 100)
    assert lo is not None and hi is not None and 0.70 < lo < 0.72 and 0.86 < hi < 0.88
    blo, bhi = entity_bootstrap_ci([1.0] * 80 + [0.0] * 20)
    assert blo is not None and bhi is not None and 0.70 <= blo <= 0.74 and 0.86 <= bhi <= 0.90
    mc = exact_mcnemar([1] * 15 + [0] * 5, [0] * 20)
    assert mc["n10_a_only"] == 15 and mc["n01_b_only"] == 0 and mc["p_value"] < 1e-4

    # 5. Cached Zero-QA Pilot 1 maps exist for both pairs
    for pkey, cfg in PAIR_CONFIGS.items():
        assert cfg["map_path"].exists(), f"Missing cached map for {pkey}: {cfg['map_path']}"

    print(
        f"[unit-test] ALL PASSED! n_entities={len(SCALED_OBSCURE_ENTITIES)} across {len(countries)} countries, "
        f"0 duplicates, 0 overlaps with dataset.json/ORGAN1_*, exact Wilson/bootstrap/McNemar & map caches OK!"
    )


# =============================================================================
# Main Evaluation Runner
# =============================================================================


def run_scaled_t3(pair_key: str = "qwen2.5-3b->smollm2-1.7b", batch_size: int = 32) -> dict:
    assert pair_key in PAIR_CONFIGS, f"Unsupported pair: {pair_key}"
    cfg = PAIR_CONFIGS[pair_key]
    donor_key, recip_key = cfg["donor"], cfg["recipient"]
    lA, lB = int(cfg["lA"]), int(cfg["lB"])
    map_path: Path = cfg["map_path"]
    out_path: Path = cfg["out_path"]

    t0 = time.time()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    with open(CACHE_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    pool = list(SCALED_OBSCURE_ENTITIES)
    print(f"[T3-scaled] Pair={pair_key} (lA={lA}->lB={lB}), screening {len(pool)} obscure entities in float32 on {device}...")

    model_A, tok_A = load_model_f32(Path(manifest["models"][donor_key]["local_path"]), device)
    model_B, tok_B = load_model_f32(Path(manifest["models"][recip_key]["local_path"]), device)

    # Step 1: Screen country prompts on A and B
    scr_p, scr_sp, scr_al = [], [], []
    for ent, kind, c in pool:
        pm = _make_prompts(ent, kind, c)["country"]
        scr_p.append(pm["prompt"])
        scr_sp.append(pm["span"])
        scr_al.append(pm["aliases"])

    res_A = eval_prompts_patched(model_A, tok_A, scr_p, scr_sp, scr_al, device=device, batch_size=batch_size)
    res_B = eval_prompts_patched(model_B, tok_B, scr_p, scr_sp, scr_al, device=device, batch_size=batch_size)

    gap_items: List[Dict[str, Any]] = []
    for i, (ent, kind, c) in enumerate(pool):
        if res_A["em_new"][i] == 1 and res_B["em_new"][i] == 0:
            gap_items.append({
                "entity": ent,
                "kind": kind,
                "country": c,
                "A_gen": res_A["gens"][i],
                "B_gen": res_B["gens"][i],
                "B_mentions_country": _gen_mentions_country(res_B["gens"][i], c),
            })

    n_gap = len(gap_items)
    by_country = Counter(g["country"] for g in gap_items)
    by_kind = Counter(g["kind"] for g in gap_items)
    print(
        f"[T3-scaled] Screened {len(pool)}: A_right={sum(res_A['em_new'])}/{len(pool)}, "
        f"B_right={sum(res_B['em_new'])}/{len(pool)} | GENUINE GAP (A-right, B-wrong) = {n_gap} "
        f"(kinds={dict(by_kind)}, countries={len(by_country)})"
    )

    if n_gap < 4:
        out = {"meta": {"pair": pair_key, "n_screened": len(pool), "n_gap": n_gap}, "gap_items": gap_items}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        return out

    # Step 2: Assign different gap entity e' (distinct country, capital, currency, language) for Control
    rng = np.random.default_rng(SEED + 5150)
    alt_idx_map: List[int] = []
    for i, gi in enumerate(gap_items):
        ci = ALL_COUNTRIES[gi["country"]]
        cands = [
            j for j, gj in enumerate(gap_items)
            if gj["country"] != gi["country"]
            and ALL_COUNTRIES[gj["country"]]["capital"][0] != ci["capital"][0]
            and ALL_COUNTRIES[gj["country"]]["currency"][0] != ci["currency"][0]
            and ALL_COUNTRIES[gj["country"]]["language"][0] != ci["language"][0]
        ]
        if not cands:
            cands = [j for j, gj in enumerate(gap_items) if gj["country"] != gi["country"]]
        alt_idx_map.append(int(cands[int(rng.integers(0, len(cands)))]) if cands else (i + 1) % n_gap)

    # Step 3: Build all 6 prompts per gap entity (5 geo + 1 protected)
    attrs = ["country", "capital", "currency", "language", "continent", "protected"]
    g_prompts, g_spans, g_al_self, g_al_alt, g_meta = [], [], [], [], []
    ent_prompts, ent_spans = [], []
    for gi_pos, gi in enumerate(gap_items):
        pm = _make_prompts(gi["entity"], gi["kind"], gi["country"])
        ent_prompts.append(pm["country"]["prompt"])
        ent_spans.append(pm["country"]["span"])
        alt_gi = gap_items[alt_idx_map[gi_pos]]
        pm_alt = _make_prompts(gi["entity"], alt_gi["kind"], alt_gi["country"])
        for ak in attrs:
            g_prompts.append(pm[ak]["prompt"])
            g_spans.append(pm[ak]["span"])
            g_al_self.append(pm[ak]["aliases"])
            g_al_alt.append(pm_alt[ak]["aliases"])
            g_meta.append({
                "gi_pos": gi_pos,
                "attr": ak,
                "entity": gi["entity"],
                "kind": gi["kind"],
                "country": gi["country"],
                "alt_country": alt_gi["country"],
            })

    base_a_ev = eval_prompts_patched(model_A, tok_A, g_prompts, g_spans, g_al_self, device=device, batch_size=batch_size)
    base_b_ev = eval_prompts_patched(model_B, tok_B, g_prompts, g_spans, g_al_self, device=device, batch_size=batch_size)

    # Build strict masks:
    # 1. Per-prompt strict A-right, B-wrong mask
    strict_attr_mask = [int(base_a_ev["em_new"][i] == 1 and base_b_ev["em_new"][i] == 0) for i in range(len(g_meta))]
    # 2. Protected baseline-known mask (unpatched B knows entity category)
    prot_base_ok_mask = [int(g_meta[i]["attr"] == "protected" and base_b_ev["em_new"][i] == 1) for i in range(len(g_meta))]
    # 3. Strict blind-spot entity mask: B_gen does NOT mention true country AND B fails >= 1 of {capital, currency, language}
    blindspot_entities = set()
    for gi_pos, gi in enumerate(gap_items):
        if gi["B_mentions_country"]:
            continue
        bk = gi_pos * len(attrs)
        # indices 1,2,3 are capital, currency, language
        if any(base_b_ev["em_new"][bk + a_idx] == 0 for a_idx in (1, 2, 3)):
            blindspot_entities.add(gi_pos)
    blindspot_mask = [int(m["gi_pos"] in blindspot_entities) for m in g_meta]
    print(f"[T3-scaled] Strict zero-hint blind-spot entities (no country mention & fails >=1 dep attr): {len(blindspot_entities)}/{n_gap}")

    # Step 4: Extract Donor A (lA) & Recipient B (lB) states, UNLOAD Donor A!
    st_A = extract_multi_layer_states(model_A, tok_A, ent_prompts, ent_spans, [lA], device)["resid"][lA]
    st_B = extract_multi_layer_states(model_B, tok_B, ent_prompts, ent_spans, [lB], device)["resid"][lB]
    del model_A, tok_A
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Load ALREADY-CACHED Zero-QA N=10,000 full-rank O(d) maps!
    p1_maps = torch.load(map_path, map_location="cpu", weights_only=False)
    p1_proc = p1_maps["procrustes_global"]
    p1_fish = p1_maps["fisher_orth_Od_multi"]
    D_A = st_A.shape[1]
    D_B = st_B.shape[1]
    R_rand = _polar_f64(rng.standard_normal((D_B, D_A)))

    def _map_z(zA: np.ndarray, mp: dict) -> np.ndarray:
        return mp["mu_B"] + mp["s"] * ((zA.astype(np.float64) - mp["mu_A"]) @ mp["R"].T)

    conds = [
        ("p1_10k_procrustes_matched", _map_z(st_A, p1_proc), False),
        ("p1_10k_fisher_orth_matched", _map_z(st_A, p1_fish), False),
        ("ctrl_B_self_roundtrip", st_B.astype(np.float64), False),
        (
            "ctrl_shuffled_R",
            p1_proc["mu_B"] + p1_proc["s"] * ((st_A.astype(np.float64) - p1_proc["mu_A"]) @ R_rand.T),
            False,
        ),
        ("ctrl_wrong_entity_e_prime", _map_z(st_A[alt_idx_map], p1_proc), True),
    ]

    def _summarize(em_list: Sequence[int], subset_mask: Optional[Sequence[int]] = None) -> dict:
        out: Dict[str, Any] = {}
        for ak in attrs:
            idxs = [
                i for i, m in enumerate(g_meta)
                if m["attr"] == ak and (subset_mask is None or subset_mask[i] == 1)
            ]
            vals = [int(em_list[i]) for i in idxs]
            k_ok = int(sum(vals))
            n_tot = len(vals)
            wlo, whi = wilson_ci(k_ok, n_tot)
            blo, bhi = entity_bootstrap_ci(vals)
            out[ak] = {
                "acc": round(k_ok / n_tot, 4) if n_tot else None,
                "k": k_ok,
                "n": n_tot,
                "wilson_95ci": [wlo, whi],
                "boot_95ci": [blo, bhi],
            }
        # Entity-level pooled 4-attr (country, capital, currency, language) AND 3-attr ripple (capital, currency, language)
        for label, ak_set in [
            ("geo_4attr_ccl", ("country", "capital", "currency", "language")),
            ("unseen_3attr_ripple", ("capital", "currency", "language")),
            ("unseen_4attr_with_cont", ("capital", "currency", "language", "continent")),
        ]:
            ent_scores = []
            for gi_pos in range(n_gap):
                e_vals = [
                    int(em_list[i]) for i, m in enumerate(g_meta)
                    if m["gi_pos"] == gi_pos and m["attr"] in ak_set
                    and (subset_mask is None or subset_mask[i] == 1)
                ]
                if e_vals:
                    ent_scores.append(float(np.mean(e_vals)))
            blo, bhi = entity_bootstrap_ci(ent_scores)
            out[label] = {
                "mean": round(float(np.mean(ent_scores)), 4) if ent_scores else None,
                "n_entities": len(ent_scores),
                "entity_boot_95ci": [blo, bhi],
            }
        # Protected retention strictly on entities where unpatched B knew the protected category
        p_idxs = [
            i for i, m in enumerate(g_meta)
            if prot_base_ok_mask[i] == 1 and (subset_mask is None or subset_mask[i] == 1)
        ]
        p_vals = [int(em_list[i]) for i in p_idxs]
        pwlo, pwhi = wilson_ci(int(sum(p_vals)), len(p_vals))
        out["protected_retention_on_B_known"] = {
            "acc": round(sum(p_vals) / len(p_vals), 4) if p_vals else None,
            "k": int(sum(p_vals)),
            "n": len(p_vals),
            "wilson_95ci": [pwlo, pwhi],
        }
        return out

    raw_em_by_cond: Dict[str, List[int]] = {}
    results: Dict[str, Any] = {
        "meta": {
            "pair": pair_key,
            "donor": donor_key,
            "recipient": recip_key,
            "layers": [lA, lB],
            "map_path": str(map_path),
            "n_screened": len(pool),
            "n_A_right": int(sum(res_A["em_new"])),
            "n_B_right": int(sum(res_B["em_new"])),
            "n_gap_total": n_gap,
            "n_strict_blindspot_entities": len(blindspot_entities),
            "n_countries_in_gap": len(by_country),
            "gap_by_country": dict(by_country),
            "gap_by_kind": dict(by_kind),
        },
        "unpatched_A": _summarize(base_a_ev["em_new"]),
        "unpatched_B": _summarize(base_b_ev["em_new"]),
        "unpatched_B_strict_blindspot": _summarize(base_b_ev["em_new"], subset_mask=blindspot_mask),
        "conditions": {},
        "mcnemar_tests": {},
        "gap_items": gap_items,
    }

    print(
        f"  [unpatched] A ctry={results['unpatched_A']['country']['acc']} 3rip={results['unpatched_A']['unseen_3attr_ripple']['mean']} | "
        f"B ctry={results['unpatched_B']['country']['acc']} 3rip={results['unpatched_B']['unseen_3attr_ripple']['mean']} "
        f"(blindspot 3rip={results['unpatched_B_strict_blindspot']['unseen_3attr_ripple']['mean']}, prot_known={sum(prot_base_ok_mask)}/{n_gap})"
    )

    for cname, z_hat, score_alt in conds:
        z_exp = np.repeat(z_hat, len(attrs), axis=0)
        ev = eval_prompts_patched(
            model_B, tok_B, g_prompts, g_spans, g_al_self,
            old_aliases_list=g_al_alt, layer_idx=lB, replace_mat=z_exp, device=device, batch_size=batch_size,
        )
        raw_em_by_cond[cname] = list(ev["em_new"])
        entry: Dict[str, Any] = {
            "all_gap_entities": _summarize(ev["em_new"]),
            "strict_blindspot_gap": _summarize(ev["em_new"], subset_mask=blindspot_mask),
            "strict_A_right_B_wrong_per_attr": _summarize(ev["em_new"], subset_mask=strict_attr_mask),
        }
        if score_alt:
            entry["tracks_swapped_alt_entity"] = _summarize(ev["em_old"])
        if cname == "p1_10k_procrustes_matched":
            exs = []
            for gi_pos in range(min(30, n_gap)):
                bk = gi_pos * len(attrs)
                exs.append({
                    "entity": gap_items[gi_pos]["entity"],
                    "kind": gap_items[gi_pos]["kind"],
                    "country": gap_items[gi_pos]["country"],
                    "is_blindspot": gi_pos in blindspot_entities,
                    "B_unpatched_country": gap_items[gi_pos]["B_gen"],
                    "transplanted_ok": {attrs[a]: int(ev["em_new"][bk + a]) for a in range(len(attrs))},
                    "transplanted_gens": {attrs[a]: ev["gens"][bk + a] for a in range(len(attrs))},
                })
            entry["examples"] = exs

        results["conditions"][cname] = entry
        ag = entry["all_gap_entities"]
        bs = entry["strict_blindspot_gap"]
        st = entry["strict_A_right_B_wrong_per_attr"]
        alt_s = (
            f" | tracks_e': ctry={entry['tracks_swapped_alt_entity']['country']['acc']} "
            f"3rip={entry['tracks_swapped_alt_entity']['unseen_3attr_ripple']['mean']}"
            if score_alt else ""
        )
        print(
            f"  [{cname:28s}] ALL(n={n_gap}): ctry={ag['country']['acc']}{ag['country']['wilson_95ci']} "
            f"4attr={ag['geo_4attr_ccl']['mean']} 3rip={ag['unseen_3attr_ripple']['mean']} prot={ag['protected_retention_on_B_known']['acc']} | "
            f"BLINDSPOT(n={len(blindspot_entities)}): ctry={bs['country']['acc']} 3rip={bs['unseen_3attr_ripple']['mean']} | "
            f"STRICT_ATTR: cap={st['capital']['acc']}(n={st['capital']['n']}) cur={st['currency']['acc']}(n={st['currency']['n']}) lng={st['language']['acc']}(n={st['language']['n']}){alt_s}"
        )

    # Step 5: Exact McNemar p-values (10k_procrustes & 10k_fisher_orth vs B_self_roundtrip, shuffled_R, wrong_entity)
    for treat in ("p1_10k_procrustes_matched", "p1_10k_fisher_orth_matched"):
        for ctrl in ("ctrl_B_self_roundtrip", "ctrl_shuffled_R", "ctrl_wrong_entity_e_prime"):
            comp_key = f"{treat}_vs_{ctrl}"
            mc_dict: Dict[str, Any] = {}
            for ak in ("country", "capital", "currency", "language", "continent"):
                idxs_all = [i for i, m in enumerate(g_meta) if m["attr"] == ak]
                idxs_str = [i for i, m in enumerate(g_meta) if m["attr"] == ak and strict_attr_mask[i] == 1]
                idxs_bs = [i for i, m in enumerate(g_meta) if m["attr"] == ak and blindspot_mask[i] == 1]
                mc_dict[f"{ak}_all"] = exact_mcnemar([raw_em_by_cond[treat][i] for i in idxs_all], [raw_em_by_cond[ctrl][i] for i in idxs_all])
                if idxs_str:
                    mc_dict[f"{ak}_strict_attr_gap"] = exact_mcnemar([raw_em_by_cond[treat][i] for i in idxs_str], [raw_em_by_cond[ctrl][i] for i in idxs_str])
                if idxs_bs:
                    mc_dict[f"{ak}_blindspot"] = exact_mcnemar([raw_em_by_cond[treat][i] for i in idxs_bs], [raw_em_by_cond[ctrl][i] for i in idxs_bs])
            # Pooled strict unseen 3-attr ripple (capital + currency + language where A=1, B=0)
            idxs_3str = [i for i, m in enumerate(g_meta) if m["attr"] in ("capital", "currency", "language") and strict_attr_mask[i] == 1]
            if idxs_3str:
                mc_dict["pooled_strict_3attr_ripple"] = exact_mcnemar(
                    [raw_em_by_cond[treat][i] for i in idxs_3str],
                    [raw_em_by_cond[ctrl][i] for i in idxs_3str],
                )
            results["mcnemar_tests"][comp_key] = mc_dict

    print("\n=== Exact McNemar Summary (p1_10k_procrustes_matched) ===")
    for ctrl in ("ctrl_B_self_roundtrip", "ctrl_shuffled_R", "ctrl_wrong_entity_e_prime"):
        mc = results["mcnemar_tests"][f"p1_10k_procrustes_matched_vs_{ctrl}"]
        c_all = mc["country_all"]
        r_str = mc.get("pooled_strict_3attr_ripple", {})
        print(
            f"  vs {ctrl:25s}: country={c_all['acc_a']} vs {c_all['acc_b']} ({c_all['n10_a_only']}:{c_all['n01_b_only']}, p={c_all['p_value']:.2e}) | "
            f"strict_3attr={r_str.get('acc_a')} vs {r_str.get('acc_b')} ({r_str.get('n10_a_only')}:{r_str.get('n01_b_only')}, p={r_str.get('p_value', 1.0):.2e})"
        )

    del model_B, tok_B
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    results["meta"]["elapsed_sec"] = round(time.time() - t0, 2)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[T3-scaled] COMPLETE in {time.time() - t0:.1f}s -> {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaled Fable T3 Long-Tail Knowledge-Gap Transplant (n >= 60)")
    parser.add_argument("--pair", type=str, default="qwen2.5-3b->smollm2-1.7b", choices=list(PAIR_CONFIGS.keys()))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--unit-test", action="store_true")
    args = parser.parse_args()
    if args.unit_test:
        run_unit_test()
        return
    run_scaled_t3(pair_key=args.pair, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
