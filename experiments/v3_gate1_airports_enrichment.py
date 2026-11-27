#!/usr/bin/env python3
"""
v3_gate1_airports_enrichment.py — 48h Gate 1: Real Large -> Small Knowledge Enrichment (OurAirports IATA)
=========================================================================================================

Real Large -> Small (`Qwen2.5-7B-Instruct` -> `SmolLM2-1.7B-Instruct`, sequential BF16 on single L4 GPU)
head-to-head benchmark comparing:
  - Stage 1: Strict blindspot screening (`7B` gets `3/3` exact IATA, `1.7B` gets `0/3`) + strictly disjoint
             shared-known calibration set (`7B` `3/3` AND `1.7B` `3/3`) for zero-leakage contrast bridges.
  - Stage 2 (Gate 1a): Mapped donor-state recovery check on held-out `eval` (`n <= 96`), requiring
             `MAPPED_DONOR_CONTRAST - max(RECIPIENT_SELF_CTRL, PERMUTED_DONOR_CTRL) >= +20 pp`.
  - Stage 3 (Gate 1b): Standalone Donor-Offline 3-Tier Head-to-Head on unseen phrasings (`p1, p2`),
             unseen 2-endpoint routes, 380-fact OOD locality, parameter bytes, wall-clock time, and
             `torch.equal` bit-exact rollback across ALL tiers:
               * Tier 1A (`CLOSED_FORM_DELTA_W`, zero backprop, `ReversibleDeltaPack`)
               * Tier 1B (`ROUTED_FACT_BANK_ZERO_SGD`, zero backprop, `HierarchicalRoutedFactBank` + hard bypass)
               * Tier 2  (`ROUTED_FACT_BANK_TINY_POLISH`, <= 25 frozen-base steps on `bank.V`)
                         vs. `TIER2_RECIPIENT_ONLY_CTRL` (equal supervision, random-init `bank.V`)
               * Tier 3  (`TEACHER_LORA_BASELINE`, fair rank-16 LoRA on `q_proj, v_proj, down_proj`)

Usage:
  PYTHONPATH=platonicswap:experiments python experiments/v3_gate1_airports_enrichment.py --unit-test
  PYTHONPATH=platonicswap:experiments python -u experiments/v3_gate1_airports_enrichment.py --max-screen 1000
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import math
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest
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

from extract_and_baseline import CACHE_DIR, SEED, get_decoder_layers, locate_last_subject_token
import large_to_small as l2s

RESULTS_DIR = REPO_ROOT / "results" if (REPO_ROOT / "results").is_dir() else SCRIPT_DIR
OUT_PATH = RESULTS_DIR / "results_v3_gate1_airports.json"
OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"

DONOR_KEY = "qwen2.5-7b-instruct"
RECIPIENT_KEY = "smollm2-1.7b-instruct"
# Candidate layer pairs (lA in Qwen2.5-7B L=28, lB in SmolLM2-1.7B L=24)
CANDIDATE_LAYER_PAIRS: Tuple[Tuple[int, int], ...] = (
    (18, 15),
    (21, 18),
    (24, 20),
    (26, 22),
)

# 3 strictly answer-free phrasings per airport (p0 = install/train phrasing; p1, p2 = unseen eval phrasings!)
AIRPORT_PHRASINGS: Tuple[str, str, str] = (
    "The three-letter IATA airport code for {name} is",
    "In international aviation, {name} uses the IATA code",
    "Flights departing from {name} are tagged with the three-letter IATA code",
)
ROUTE_TEMPLATE = "A nonstop flight from {name_a} to {name_b} connects IATA codes"


# =============================================================================
# 1. Embedded Offline Fallback: 160 Real Worldwide Airports (0 IATA-in-Name / 0 Prefix Leaks!)
# =============================================================================

EMBEDDED_AIRPORTS_160: List[Tuple[str, str, str, str]] = [
    # (airport_name, iata_code, city, country) — verified real worldwide airports, 0 lexical leaks!
    # Europe (48)
    ("Heathrow Airport", "LHR", "London", "GB"),
    ("Gatwick Airport", "LGW", "London", "GB"),
    ("Stansted Airport", "STN", "London", "GB"),
    ("Luton Airport", "LTN", "London", "GB"),
    ("Charles de Gaulle Airport", "CDG", "Paris", "FR"),
    ("Orly Airport", "ORY", "Paris", "FR"),
    ("Schiphol Airport", "AMS", "Amsterdam", "NL"),
    ("Fiumicino Airport", "FCO", "Rome", "IT"),
    ("Malpensa Airport", "MXP", "Milan", "IT"),
    ("Linate Airport", "LIN", "Milan", "IT"),
    ("Marco Polo Airport", "VCE", "Venice", "IT"),
    ("Barajas Airport", "MAD", "Madrid", "ES"),
    ("El Prat Airport", "BCN", "Barcelona", "ES"),
    ("Son Sant Joan Airport", "PMI", "Palma de Mallorca", "ES"),
    ("Kastrup Airport", "CPH", "Copenhagen", "DK"),
    ("Arlanda Airport", "ARN", "Stockholm", "SE"),
    ("Landvetter Airport", "GOT", "Gothenburg", "SE"),
    ("Gardermoen Airport", "OSL", "Oslo", "NO"),
    ("Flesland Airport", "BGO", "Bergen", "NO"),
    ("Vantaa Airport", "HEL", "Helsinki", "FI"),
    ("Kloten Airport", "ZRH", "Zurich", "CH"),
    ("Cointrin Airport", "GVA", "Geneva", "CH"),
    ("Schwechat Airport", "VIE", "Vienna", "AT"),
    ("Zaventem Airport", "BRU", "Brussels", "BE"),
    ("Portela Airport", "LIS", "Lisbon", "PT"),
    ("Francisco Sa Carneiro Airport", "OPO", "Porto", "PT"),
    ("Eleftherios Venizelos Airport", "ATH", "Athens", "GR"),
    ("Nikos Kazantzakis Airport", "HER", "Heraklion", "GR"),
    ("Vaclav Havel Airport", "PRG", "Prague", "CZ"),
    ("Ferenc Liszt International Airport", "BUD", "Budapest", "HU"),
    ("Chopin Airport", "WAW", "Warsaw", "PL"),
    ("John Paul II International Airport", "KRK", "Krakow", "PL"),
    ("Henri Coanda International Airport", "OTP", "Bucharest", "RO"),
    ("Vrazhdebna Airport", "SOF", "Sofia", "BG"),
    ("Nikola Tesla Airport", "BEG", "Belgrade", "RS"),
    ("Franjo Tudman Airport", "ZAG", "Zagreb", "HR"),
    ("Pleso Airfield", "SPU", "Split", "HR"),
    ("Keflavik International Airport", "KEF", "Reykjavik", "IS"),
    ("Collinstown Airport", "DUB", "Dublin", "IE"),
    ("Rhein-Main Airport", "FRA", "Frankfurt", "DE"),
    ("Franz Josef Strauss Airport", "MUC", "Munich", "DE"),
    ("Fuhlsbuttel Airport", "HAM", "Hamburg", "DE"),
    ("Lohausen Airport", "DUS", "Dusseldorf", "DE"),
    ("Echterdingen Airport", "STR", "Stuttgart", "DE"),
    ("Brandenburg Airport", "BER", "Berlin", "DE"),
    ("Sabiha Gokcen International Airport", "SAW", "Istanbul", "TR"),
    ("Esenboga International Airport", "ESB", "Ankara", "TR"),
    ("Adnan Menderes Airport", "ADB", "Izmir", "TR"),

    # North America (48)
    ("O'Hare International Airport", "ORD", "Chicago", "US"),
    ("Midway International Airport", "MDW", "Chicago", "US"),
    ("John F. Kennedy International Airport", "JFK", "New York", "US"),
    ("LaGuardia Airport", "LGA", "New York", "US"),
    ("Newark Liberty International Airport", "EWR", "Newark", "US"),
    ("Logan International Airport", "BOS", "Boston", "US"),
    ("Hartsfield-Jackson International Airport", "ATL", "Atlanta", "US"),
    ("Dulles International Airport", "IAD", "Washington", "US"),
    ("Ronald Reagan National Airport", "DCA", "Washington", "US"),
    ("Thurgood Marshall Airport", "BWI", "Baltimore", "US"),
    ("Sky Harbor International Airport", "PHX", "Phoenix", "US"),
    ("Harry Reid International Airport", "LAS", "Las Vegas", "US"),
    ("George Bush Intercontinental Airport", "IAH", "Houston", "US"),
    ("William P. Hobby Airport", "HOU", "Houston", "US"),
    ("Love Field", "DAL", "Dallas", "US"),
    ("Fort Worth International Airport", "DFW", "Dallas", "US"),
    ("Louis Armstrong International Airport", "MSY", "New Orleans", "US"),
    ("Lambert International Airport", "STL", "St. Louis", "US"),
    ("Hopkins International Airport", "CLE", "Cleveland", "US"),
    ("John Glenn International Airport", "CMH", "Columbus", "US"),
    ("General Mitchell International Airport", "MKE", "Milwaukee", "US"),
    ("Gerald R. Ford International Airport", "GRR", "Grand Rapids", "US"),
    ("Will Rogers World Airport", "OKC", "Oklahoma City", "US"),
    ("Eppley Airfield", "OMA", "Omaha", "US"),
    ("Bradley International Airport", "BDL", "Hartford", "US"),
    ("Theodore Francis Green Airport", "PVD", "Providence", "US"),
    ("Ted Stevens International Airport", "ANC", "Anchorage", "US"),
    ("Daniel K. Inouye International Airport", "HNL", "Honolulu", "US"),
    ("Norman Y. Mineta International Airport", "SJC", "San Jose", "US"),
    ("Lindbergh Field", "SAN", "San Diego", "US"),
    ("John Wayne Airport", "SNA", "Santa Ana", "US"),
    ("Bob Hope Airport", "BUR", "Burbank", "US"),
    ("McGhee Tyson Airport", "TYS", "Knoxville", "US"),
    ("Yeager Airport", "CRW", "Charleston", "US"),
    ("Blue Grass Airport", "LEX", "Lexington", "US"),
    ("Standiford Field", "SDF", "Louisville", "US"),
    ("Pearson International Airport", "YYZ", "Toronto", "CA"),
    ("Pierre Elliott Trudeau International Airport", "YUL", "Montreal", "CA"),
    ("Sea Island Airport", "YVR", "Vancouver", "CA"),
    ("McCall Field", "YYC", "Calgary", "CA"),
    ("Stanfield International Airport", "YHZ", "Halifax", "CA"),
    ("Macdonald-Cartier International Airport", "YOW", "Ottawa", "CA"),
    ("James Armstrong Richardson International Airport", "YWG", "Winnipeg", "CA"),
    ("Jean Lesage International Airport", "YQB", "Quebec City", "CA"),
    ("Benito Juarez International Airport", "MEX", "Mexico City", "MX"),
    ("Miguel Hidalgo y Costilla Airport", "GDL", "Guadalajara", "MX"),
    ("General Mariano Escobedo Airport", "MTY", "Monterrey", "MX"),
    ("Licenciado Gustavo Diaz Ordaz Airport", "PVR", "Puerto Vallarta", "MX"),

    # Asia & Oceania (40)
    ("Narita International Airport", "NRT", "Tokyo", "JP"),
    ("Haneda Airport", "HND", "Tokyo", "JP"),
    ("Kansai International Airport", "KIX", "Osaka", "JP"),
    ("Itami Airport", "ITM", "Osaka", "JP"),
    ("Chubu Centrair International Airport", "NGO", "Nagoya", "JP"),
    ("New Chitose Airport", "CTS", "Sapporo", "JP"),
    ("Incheon International Airport", "ICN", "Seoul", "KR"),
    ("Gimpo International Airport", "GMP", "Seoul", "KR"),
    ("Gimhae International Airport", "PUS", "Busan", "KR"),
    ("Pudong International Airport", "PVG", "Shanghai", "CN"),
    ("Hongqiao International Airport", "SHA", "Shanghai", "CN"),
    ("Baiyun International Airport", "CAN", "Guangzhou", "CN"),
    ("Bao'an International Airport", "SZX", "Shenzhen", "CN"),
    ("Shuangliu International Airport", "CTU", "Chengdu", "CN"),
    ("Daxing International Airport", "PKX", "Beijing", "CN"),
    ("Chek Lap Kok International Airport", "HKG", "Hong Kong", "HK"),
    ("Taoyuan International Airport", "TPE", "Taipei", "TW"),
    ("Songshan Airport", "TSA", "Taipei", "TW"),
    ("Changi Airport", "SIN", "Singapore", "SG"),
    ("Suvarnabhumi Airport", "BKK", "Bangkok", "TH"),
    ("Don Mueang International Airport", "DMK", "Bangkok", "TH"),
    ("Sepang International Airport", "KUL", "Kuala Lumpur", "MY"),
    ("Soekarno-Hatta International Airport", "CGK", "Jakarta", "ID"),
    ("Ngurah Rai International Airport", "DPS", "Denpasar", "ID"),
    ("Ninoy Aquino International Airport", "MNL", "Manila", "PH"),
    ("Mactan International Airport", "CEB", "Cebu", "PH"),
    ("Tan Son Nhat International Airport", "SGN", "Ho Chi Minh City", "VN"),
    ("Noi Bai International Airport", "HAN", "Hanoi", "VN"),
    ("Indira Gandhi International Airport", "DEL", "New Delhi", "IN"),
    ("Chhatrapati Shivaji Maharaj International Airport", "BOM", "Mumbai", "IN"),
    ("Kempegowda International Airport", "BLR", "Bengaluru", "IN"),
    ("Rajiv Gandhi International Airport", "HYD", "Hyderabad", "IN"),
    ("Netaji Subhas Chandra Bose International Airport", "CCU", "Kolkata", "IN"),
    ("Meenambakkam Airport", "MAA", "Chennai", "IN"),
    ("Bandaranaike International Airport", "CMB", "Colombo", "LK"),
    ("Tribhuvan International Airport", "KTM", "Kathmandu", "NP"),
    ("Hazrat Shahjalal International Airport", "DAC", "Dhaka", "BD"),
    ("Jinnah International Airport", "KHI", "Karachi", "PK"),
    ("Kingsford Smith Airport", "SYD", "Sydney", "AU"),
    ("Tullamarine Airport", "MEL", "Melbourne", "AU"),

    # Middle East, Africa & South America (24)
    ("Ben Gurion Airport", "TLV", "Tel Aviv", "IL"),
    ("Queen Alia International Airport", "AMM", "Amman", "JO"),
    ("King Khalid International Airport", "RUH", "Riyadh", "SA"),
    ("King Abdulaziz International Airport", "JED", "Jeddah", "SA"),
    ("Hamad International Airport", "DOH", "Doha", "QA"),
    ("Zayed International Airport", "AUH", "Abu Dhabi", "AE"),
    ("Seeb International Airport", "MCT", "Muscat", "OM"),
    ("Imam Khomeini International Airport", "IKA", "Tehran", "IR"),
    ("O. R. Tambo International Airport", "JNB", "Johannesburg", "ZA"),
    ("D. F. Malan Airport", "CPT", "Cape Town", "ZA"),
    ("Jomo Kenyatta International Airport", "NBO", "Nairobi", "KE"),
    ("Bole International Airport", "ADD", "Addis Ababa", "ET"),
    ("Mohammed V International Airport", "CMN", "Casablanca", "MA"),
    ("Murtala Muhammed International Airport", "LOS", "Lagos", "NG"),
    ("Kotoka International Airport", "ACC", "Accra", "GH"),
    ("Blaise Diagne International Airport", "DSS", "Dakar", "SN"),
    ("Guarulhos International Airport", "GRU", "Sao Paulo", "BR"),
    ("Galeao International Airport", "GIG", "Rio de Janeiro", "BR"),
    ("Congonhas Airport", "CGH", "Sao Paulo", "BR"),
    ("Ministro Pistarini International Airport", "EZE", "Buenos Aires", "AR"),
    ("Arturo Merino Benitez International Airport", "SCL", "Santiago", "CL"),
    ("El Dorado International Airport", "BOG", "Bogota", "CO"),
    ("Jorge Chavez International Airport", "LIM", "Lima", "PE"),
    ("Tocumen International Airport", "PTY", "Panama City", "PA"),
]


def _alpha_only(s: str) -> str:
    return re.sub(r"[^A-Za-z]", "", s).upper()


def _is_clean_airport(name: str, iata: str, exclude_prefix_leak: bool = True) -> bool:
    if not name or len(name.strip()) < 6:
        return False
    if not re.fullmatch(r"[A-Z]{3}", iata):
        return False
    # Exclude entries where IATA code appears literally in the name (case-insensitive substring or word)
    if re.search(r"\b" + re.escape(iata) + r"\b", name, flags=re.IGNORECASE):
        return False
    if f"({iata})" in name.upper():
        return False
    if exclude_prefix_leak:
        al = _alpha_only(name)
        if len(al) >= 3 and al[:3] == iata:
            return False
    return True


def fetch_or_fallback_airports(
    max_airports: int = 1200,
    seed: int = SEED,
    offline: bool = False,
    exclude_prefix_leak: bool = True,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Load OurAirports dataset online (with local CSV cache) filtering `type in {'large_airport', 'medium_airport'}`
    with valid 3-letter IATA codes and zero lexical leaks; gracefully falls back to the embedded
    160 real worldwide airports when offline or in `--unit-test`.
    """
    rows: List[Dict[str, Any]] = []
    source = "embedded_fallback_160"
    csv_text: Optional[str] = None
    csv_cache = CACHE_DIR / "ourairports_airports.csv"

    if not offline:
        if csv_cache.is_file() and csv_cache.stat().st_size > 10000:
            csv_text = csv_cache.read_text(encoding="utf-8", errors="ignore")
            source = f"cached_csv:{csv_cache.name}"
        else:
            try:
                req = urllib.request.Request(OURAIRPORTS_URL, headers={"User-Agent": "PlatonicSwap/3.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    csv_text = resp.read().decode("utf-8", errors="ignore")
                try:
                    csv_cache.parent.mkdir(parents=True, exist_ok=True)
                    csv_cache.write_text(csv_text, encoding="utf-8")
                except Exception:
                    pass
                source = "ourairports_online_csv"
            except Exception:
                csv_text = None

    seen_iata: set[str] = set()
    seen_name: set[str] = set()

    if csv_text is not None:
        reader = csv.DictReader(io.StringIO(csv_text))
        # Prioritize large_airport first, then medium_airport, deterministically shuffled within tier
        large_pool, med_pool = [], []
        for r in reader:
            atype = (r.get("type") or "").strip()
            if atype not in ("large_airport", "medium_airport"):
                continue
            name = (r.get("name") or "").strip()
            iata = (r.get("iata_code") or "").strip().upper()
            if not _is_clean_airport(name, iata, exclude_prefix_leak=exclude_prefix_leak):
                continue
            nkey = name.lower()
            if iata in seen_iata or nkey in seen_name:
                continue
            seen_iata.add(iata)
            seen_name.add(nkey)
            rec = {
                "name": name,
                "iata": iata,
                "city": (r.get("municipality") or "").strip(),
                "country": (r.get("iso_country") or "").strip(),
                "airport_type": atype,
            }
            if atype == "large_airport":
                large_pool.append(rec)
            else:
                med_pool.append(rec)
        rng = random.Random(seed)
        rng.shuffle(large_pool)
        rng.shuffle(med_pool)
        rows = (large_pool + med_pool)[:max_airports]
    else:
        for name, iata, city, country in EMBEDDED_AIRPORTS_160:
            if not _is_clean_airport(name, iata, exclude_prefix_leak=exclude_prefix_leak):
                continue
            nkey = name.lower()
            if iata in seen_iata or nkey in seen_name:
                continue
            seen_iata.add(iata)
            seen_name.add(nkey)
            rows.append({
                "name": name,
                "iata": iata,
                "city": city,
                "country": country,
                "airport_type": "large_airport",
            })
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_airports]

    return rows, source


# =============================================================================
# 2. Prompt Builders, Exact Regex IATA Matchers & Split Helpers
# =============================================================================


def format_airport_prompt(name: str, phrasing_idx: int) -> Tuple[str, List[int]]:
    p = AIRPORT_PHRASINGS[phrasing_idx].format(name=name)
    c0 = p.index(name)
    return p, [c0, c0 + len(name)]


def build_route_items(airports: Sequence[Dict[str, Any]], max_routes: int = 48, seed: int = SEED + 17) -> List[Dict[str, Any]]:
    """Build unseen 2-endpoint route prompts pairing distinct qualified airports."""
    n = len(airports)
    if n < 2:
        return []
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    routes: List[Dict[str, Any]] = []
    # Pair (idxs[i], idxs[(i + 1) % n]) so every airport appears symmetrically
    for i in range(min(max_routes, n)):
        a = airports[idxs[i]]
        b = airports[idxs[(i + 1) % n]]
        if a["iata"] == b["iata"]:
            continue
        p = ROUTE_TEMPLATE.format(name_a=a["name"], name_b=b["name"])
        routes.append({
            "name_a": a["name"],
            "iata_a": a["iata"],
            "name_b": b["name"],
            "iata_b": b["iata"],
            "prompt": p,
        })
    return routes


def extract_iata_code(generated_text: str) -> Optional[str]:
    """Extract the first clean 3-letter uppercase IATA code from a model completion."""
    if not generated_text:
        return None
    # Strip leading whitespace/quotes/punctuation and search for standalone 3-letter uppercase code
    s = generated_text.strip()
    m0 = re.match(r'^[\'"(\[]*([A-Z]{3})(?:[^A-Za-z]|$)', s)
    if m0:
        return m0.group(1)
    m1 = re.search(r'\b([A-Z]{3})\b', s)
    if m1:
        return m1.group(1)
    return None


def check_iata_match(generated_text: str, gold_iata: str) -> bool:
    got = extract_iata_code(generated_text)
    return got is not None and got == gold_iata.upper()


def check_route_match(generated_text: str, iata_a: str, iata_b: str) -> Dict[str, bool]:
    codes = re.findall(r'\b([A-Z]{3})\b', generated_text or "")
    a_u, b_u = iata_a.upper(), iata_b.upper()
    has_a = a_u in codes
    has_b = b_u in codes
    ordered = len(codes) >= 2 and codes[0] == a_u and codes[1] == b_u
    return {"both": bool(has_a and has_b), "ordered": bool(ordered), "either": bool(has_a or has_b)}


def partition_screened_airports(
    airports: Sequence[Dict[str, Any]],
    donor_hits_3: Sequence[int],
    recip_hits_3: Sequence[int],
    n_dev: int = 24,
    n_eval_max: int = 96,
    n_shared_max: int = 200,
    seed: int = SEED + 33,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Strictly partition screened airports with ZERO overlap:
      - `shared_known`: `7B == 3/3` AND `1.7B == 3/3` (calibration bridge ONLY!)
      - `blindspot_dev`: `7B == 3/3` AND `1.7B == 0/3` (layer/hyperparam selection ONLY!)
      - `blindspot_eval`: `7B == 3/3` AND `1.7B == 0/3` (strictly held-out final evaluation!)
    """
    assert len(airports) == len(donor_hits_3) == len(recip_hits_3)
    shared = []
    blind = []
    for ap, dh, rh in zip(airports, donor_hits_3, recip_hits_3):
        if dh == 3 and rh == 3:
            shared.append(ap)
        elif dh == 3 and rh == 0:
            blind.append(ap)
    rng = random.Random(seed)
    rng.shuffle(shared)
    rng.shuffle(blind)
    shared = shared[:n_shared_max]
    n_d = min(n_dev, max(1, len(blind) // 3)) if len(blind) < (n_dev + 8) else n_dev
    dev = blind[:n_d]
    ev = blind[n_d : n_d + n_eval_max]
    return {"shared_known": shared, "blindspot_dev": dev, "blindspot_eval": ev}


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


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[Optional[float], Optional[float]]:
    if n <= 0:
        return None, None
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p)) / n + (z * z) / (4.0 * n * n))
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


# =============================================================================
# 3. Mask-Safe Routed Bank Hook & Self-Contained Reversible Rank-16 Teacher LoRA
# =============================================================================


def build_valid_token_mask_from_attention(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Build explicit `(B, T)` boolean mask that is `True` ONLY on non-pad, non-BOS tokens
    (handles both left-padding and right-padding cleanly — eliminates Seat 1's padding/BOS bug!).
    """
    am = attention_mask.bool()
    # Position of the FIRST active token in each row (the BOS / first real token)
    cumsum = am.long().cumsum(dim=-1)
    # Valid tokens have attention_mask == 1 AND cumsum >= 2 (i.e. strictly after BOS/first token!)
    return am & (cumsum >= 2)


class RoutedBankLayerHook:
    """
    Context-managed residual hook attaching Seat 3's `HierarchicalRoutedFactBank` to `layers[lB]`.
    Tracks current batch `valid_token_mask` (explicitly excluding BOS & left/right padding!)
    and respects `bank.enabled == False` as a strict no-op (`torch.equal == True` hard bypass!).
    """

    def __init__(self, model: nn.Module, layer_idx: int, bank: l2s.HierarchicalRoutedFactBank):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.bank = bank
        self.current_valid_mask: Optional[torch.Tensor] = None
        self._handle: Optional[Any] = None

    def set_attention_mask(self, attention_mask: Optional[torch.Tensor]) -> None:
        if attention_mask is None:
            self.current_valid_mask = None
        else:
            self.current_valid_mask = build_valid_token_mask_from_attention(attention_mask)

    def _hook_fn(self, _mod: nn.Module, _inp: Any, out: Any) -> Any:
        if not self.bank.enabled:
            return out
        is_tup = isinstance(out, tuple)
        h = out[0] if is_tup else out
        # During KV-cached decode steps (`use_cache=True`, T=1), never re-fire on generated tokens!
        vmask = self.current_valid_mask
        if vmask is not None and vmask.shape[:2] != h.shape[:2]:
            return out
        delta = self.bank(h, valid_token_mask=vmask)
        h_new = h + delta
        return (h_new,) + out[1:] if is_tup else h_new

    def attach(self) -> "RoutedBankLayerHook":
        layer = get_decoder_layers(self.model)[self.layer_idx]
        self._handle = layer.register_forward_hook(self._hook_fn)
        return self

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.current_valid_mask = None


class ReversibleLoRALinear(nn.Module):
    """Zero-dependency reversible LoRA wrapper (`W_base` frozen; `delta_W = (alpha/r) * B @ A`)."""

    def __init__(self, base_linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        in_f = base_linear.in_features
        out_f = base_linear.out_features
        dev = base_linear.weight.device
        self.lora_A = nn.Parameter(torch.empty((rank, in_f), device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros((out_f, rank), device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        delta = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scale
        return y + delta.to(y.dtype)

    def materialize_delta_weight(self) -> torch.Tensor:
        return (self.lora_B.detach() @ self.lora_A.detach()) * self.scale


def attach_lora_adapters(
    model: nn.Module,
    layer_indices: Sequence[int],
    target_modules: Sequence[str] = ("q_proj", "v_proj", "down_proj"),
    rank: int = 16,
    alpha: float = 32.0,
) -> Dict[str, ReversibleLoRALinear]:
    """Attach zero-dependency LoRA adapters on `(q_proj, v_proj, down_proj)` across `layer_indices`."""
    layers = get_decoder_layers(model)
    attached: Dict[str, ReversibleLoRALinear] = {}
    for li in layer_indices:
        layer = layers[li]
        for tname in target_modules:
            if tname in ("q_proj", "k_proj", "v_proj", "o_proj") and hasattr(layer.self_attn, tname):
                parent = layer.self_attn
                full_param = f"model.layers.{li}.self_attn.{tname}.weight"
            elif tname in ("gate_proj", "up_proj", "down_proj") and hasattr(layer.mlp, tname):
                parent = layer.mlp
                full_param = f"model.layers.{li}.mlp.{tname}.weight"
            else:
                continue
            orig_lin = getattr(parent, tname)
            assert isinstance(orig_lin, nn.Linear)
            wrapped = ReversibleLoRALinear(orig_lin, rank=rank, alpha=alpha)
            setattr(parent, tname, wrapped)
            attached[full_param] = wrapped
    return attached


def detach_lora_to_delta_pack(
    model: nn.Module,
    attached: Dict[str, ReversibleLoRALinear],
    layer_indices: Sequence[int],
    target_modules: Sequence[str] = ("q_proj", "v_proj", "down_proj"),
) -> Tuple[l2s.ReversibleDeltaPack, int]:
    """Restore original `nn.Linear` modules and return a `ReversibleDeltaPack` + raw LoRA BF16 bytes."""
    deltas: Dict[str, torch.Tensor] = {}
    lora_param_bytes = 0
    layers = get_decoder_layers(model)
    for li in layer_indices:
        layer = layers[li]
        for tname in target_modules:
            if tname in ("q_proj", "k_proj", "v_proj", "o_proj") and hasattr(layer.self_attn, tname):
                parent = layer.self_attn
                full_param = f"model.layers.{li}.self_attn.{tname}.weight"
            elif tname in ("gate_proj", "up_proj", "down_proj") and hasattr(layer.mlp, tname):
                parent = layer.mlp
                full_param = f"model.layers.{li}.mlp.{tname}.weight"
            else:
                continue
            wrapped = getattr(parent, tname)
            if isinstance(wrapped, ReversibleLoRALinear):
                deltas[full_param] = wrapped.materialize_delta_weight().cpu()
                lora_param_bytes += (wrapped.lora_A.numel() + wrapped.lora_B.numel()) * 2
                setattr(parent, tname, wrapped.base)
    return l2s.ReversibleDeltaPack(deltas, metadata={"type": "teacher_lora"}), int(lora_param_bytes)


# =============================================================================
# 4. Batched Generation, Multi-Layer State Extraction & Training Loops (GPU/L4)
# =============================================================================


@torch.no_grad()
def greedy_generate_prompts(
    model: Any,
    tok: Any,
    device: torch.device,
    prompts: Sequence[str],
    max_new_tokens: int = 6,
    batch_size: int = 32,
    bank_hook: Optional[RoutedBankLayerHook] = None,
) -> List[str]:
    old_pad = tok.padding_side
    tok.padding_side = "left"
    gens: List[str] = []
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    try:
        for st in range(0, len(prompts), batch_size):
            bp = list(prompts[st : st + batch_size])
            enc = tok(bp, return_tensors="pt", padding=True, add_special_tokens=True)
            inp_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            if bank_hook is not None:
                bank_hook.set_attention_mask(attn)
            out = model.generate(
                input_ids=inp_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
            plen = inp_ids.shape[1]
            for bi in range(len(bp)):
                gens.append(tok.decode(out[bi, plen:], skip_special_tokens=True))
    finally:
        if bank_hook is not None:
            bank_hook.set_attention_mask(None)
        tok.padding_side = old_pad
    return gens


@torch.no_grad()
def extract_multilayer_airport_states(
    model: Any,
    tok: Any,
    device: torch.device,
    airports: Sequence[Dict[str, Any]],
    phrasing_idx: int,
    layer_indices: Sequence[int],
    capture_mlp_in: bool = False,
    batch_size: int = 24,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Extract post-layer residual states `h_out` (and optionally `mlp.down_proj` input `z_mlp`)
    at BOTH `airport_last` token AND `prompt_last` token across `layer_indices` in ONE pass!
    """
    layers = get_decoder_layers(model)
    prompts, spans = [], []
    for ap in airports:
        p, sp = format_airport_prompt(ap["name"], phrasing_idx)
        prompts.append(p)
        spans.append(sp)

    subj_pos = [locate_last_subject_token(tok, p, sp)[0] for p, sp in zip(prompts, spans)]
    buf_h: Dict[int, torch.Tensor] = {}
    buf_z: Dict[int, torch.Tensor] = {}
    handles = []
    for li in layer_indices:
        def _mk_h(idx):
            def _fn(_m, _i, o):
                buf_h[idx] = (o[0] if isinstance(o, tuple) else o).detach()
            return _fn
        handles.append(layers[li].register_forward_hook(_mk_h(li)))
        if capture_mlp_in:
            def _mk_z(idx):
                def _fn(_m, inp):
                    buf_z[idx] = inp[0].detach()
                return _fn
            handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(_mk_z(li)))

    acc: Dict[int, Dict[str, List[np.ndarray]]] = {
        li: {"h_airport_last": [], "h_prompt_last": [], "z_airport_last": [], "z_prompt_last": []}
        for li in layer_indices
    }
    old_pad = tok.padding_side
    tok.padding_side = "right"
    try:
        for st in range(0, len(prompts), batch_size):
            bp = prompts[st : st + batch_size]
            enc = tok(bp, return_tensors="pt", padding=True, add_special_tokens=True)
            inp_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            _ = model(input_ids=inp_ids, attention_mask=attn, use_cache=False)
            seq_lens = attn.sum(dim=-1).cpu().tolist()
            for li in layer_indices:
                hb = buf_h[li].float().cpu().numpy()
                zb = buf_z[li].float().cpu().numpy() if capture_mlp_in else None
                for bi in range(len(bp)):
                    gidx = st + bi
                    p_subj = min(subj_pos[gidx], int(seq_lens[bi]) - 1)
                    p_last = int(seq_lens[bi]) - 1
                    acc[li]["h_airport_last"].append(hb[bi, p_subj].astype(np.float32))
                    acc[li]["h_prompt_last"].append(hb[bi, p_last].astype(np.float32))
                    if zb is not None:
                        acc[li]["z_airport_last"].append(zb[bi, p_subj].astype(np.float32))
                        acc[li]["z_prompt_last"].append(zb[bi, p_last].astype(np.float32))
    finally:
        for h in handles:
            h.remove()
        tok.padding_side = old_pad

    out: Dict[int, Dict[str, np.ndarray]] = {}
    for li in layer_indices:
        out[li] = {
            "h_airport_last": np.stack(acc[li]["h_airport_last"], axis=0),
            "h_prompt_last": np.stack(acc[li]["h_prompt_last"], axis=0),
        }
        if capture_mlp_in:
            out[li]["z_airport_last"] = np.stack(acc[li]["z_airport_last"], axis=0)
            out[li]["z_prompt_last"] = np.stack(acc[li]["z_prompt_last"], axis=0)
    return out


@torch.no_grad()
def eval_patched_airports_gate1a(
    model_B: Any,
    tok_B: Any,
    device_B: torch.device,
    airports: Sequence[Dict[str, Any]],
    phrasing_idx: int,
    lB: int,
    site: str,
    patched_h_B: np.ndarray,
    batch_size: int = 24,
) -> Dict[str, Any]:
    """Patch Recipient B's layer `lB` residual at `site in {'airport_last', 'prompt_last'}` and score exact IATA EM."""
    layer = get_decoder_layers(model_B)[lB]
    prompts, spans = [], []
    for ap in airports:
        p, sp = format_airport_prompt(ap["name"], phrasing_idx)
        prompts.append(p)
        spans.append(sp)

    subj_pos_unpadded = [locate_last_subject_token(tok_B, p, sp)[0] for p, sp in zip(prompts, spans)]
    old_pad = tok_B.padding_side
    tok_B.padding_side = "left"
    pad_id = tok_B.pad_token_id if tok_B.pad_token_id is not None else tok_B.eos_token_id
    hits: List[int] = []
    gens: List[str] = []

    try:
        for st in range(0, len(prompts), batch_size):
            bp = prompts[st : st + batch_size]
            enc = tok_B(bp, return_tensors="pt", padding=True, add_special_tokens=True)
            inp_ids = enc["input_ids"].to(device_B)
            attn = enc["attention_mask"].to(device_B)
            plen = inp_ids.shape[1]
            # Compute exact left-padded target position for each row
            real_lens = attn.sum(dim=-1).cpu().tolist()
            t_pos = []
            for bi in range(len(bp)):
                pad_off = plen - int(real_lens[bi])
                if site == "airport_last":
                    t_pos.append(pad_off + min(subj_pos_unpadded[st + bi], int(real_lens[bi]) - 1))
                else:
                    t_pos.append(plen - 1)
            patch_t = torch.tensor(patched_h_B[st : st + len(bp)], dtype=model_B.dtype, device=device_B)
            did_prefill = {"done": False}

            def _patch_hook(_mod, _inp, out):
                if did_prefill["done"]:
                    return out
                is_tup = isinstance(out, tuple)
                h = (out[0] if is_tup else out).clone()
                if h.shape[1] == plen:
                    did_prefill["done"] = True
                    for bi, pos in enumerate(t_pos):
                        h[bi, pos] = patch_t[bi]
                return (h,) + out[1:] if is_tup else h

            handle = layer.register_forward_hook(_patch_hook)
            try:
                out_ids = model_B.generate(
                    input_ids=inp_ids,
                    attention_mask=attn,
                    max_new_tokens=6,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
            finally:
                handle.remove()

            for bi in range(len(bp)):
                g = tok_B.decode(out_ids[bi, plen:], skip_special_tokens=True)
                gens.append(g)
                hits.append(int(check_iata_match(g, airports[st + bi]["iata"])))
    finally:
        tok_B.padding_side = old_pad

    k_ok, n = int(sum(hits)), len(hits)
    return {"acc": round(k_ok / n, 4) if n else None, "k": k_ok, "n": n, "wilson_95ci": list(wilson_ci(k_ok, n)), "_raw": hits}


def eval_standalone_airport_suite(
    model_B: Any,
    tok_B: Any,
    device_B: torch.device,
    eval_airports: Sequence[Dict[str, Any]],
    route_items: Sequence[Dict[str, Any]],
    loc_facts: Sequence[Dict[str, Any]],
    base_loc_ok: Sequence[int],
    bank_hook: Optional[RoutedBankLayerHook] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Evaluate any standalone Donor-Offline model on p0 (train phrasing), p1 & p2 (unseen phrasings),
    unseen 2-endpoint routes, and 380-fact OOD locality!"""
    from extract_and_baseline import check_answer_match

    by_p: Dict[str, Any] = {}
    unseen_hits: List[int] = []
    for pidx in (0, 1, 2):
        prompts = [format_airport_prompt(ap["name"], pidx)[0] for ap in eval_airports]
        gens = greedy_generate_prompts(model_B, tok_B, device_B, prompts, max_new_tokens=6, batch_size=batch_size, bank_hook=bank_hook)
        hits = [int(check_iata_match(g, ap["iata"])) for g, ap in zip(gens, eval_airports)]
        k_ok, n = int(sum(hits)), len(hits)
        by_p[f"p{pidx}"] = {"acc": round(k_ok / n, 4) if n else None, "k": k_ok, "n": n, "wilson_95ci": list(wilson_ci(k_ok, n)), "_raw": hits}
        if pidx in (1, 2):
            unseen_hits.extend(hits)

    # Unseen 2-endpoint routes
    r_prompts = [r["prompt"] for r in route_items]
    r_gens = greedy_generate_prompts(model_B, tok_B, device_B, r_prompts, max_new_tokens=12, batch_size=batch_size, bank_hook=bank_hook) if r_prompts else []
    r_both, r_ord, r_eith = [], [], []
    for g, r in zip(r_gens, route_items):
        m = check_route_match(g, r["iata_a"], r["iata_b"])
        r_both.append(int(m["both"]))
        r_ord.append(int(m["ordered"]))
        r_eith.append(int(m["either"]))

    # Certified 380-fact OOD locality
    loc_prompts = [f["direct_prompt"] for f in loc_facts]
    loc_gens = greedy_generate_prompts(model_B, tok_B, device_B, loc_prompts, max_new_tokens=8, batch_size=batch_size, bank_hook=bank_hook)
    loc_ok = [int(check_answer_match(g, f["aliases"])) for g, f in zip(loc_gens, loc_facts)]
    n_base_ok = int(sum(base_loc_ok))
    newly_broken = int(sum(1 for b, c in zip(base_loc_ok, loc_ok) if b == 1 and c == 0))
    ret = (n_base_ok - newly_broken) / n_base_ok if n_base_ok > 0 else 1.0

    ku, nu = int(sum(unseen_hits)), len(unseen_hits)
    nr = len(route_items)
    return {
        "train_phrasing_p0": {k: v for k, v in by_p["p0"].items() if k != "_raw"},
        "unseen_p1": {k: v for k, v in by_p["p1"].items() if k != "_raw"},
        "unseen_p2": {k: v for k, v in by_p["p2"].items() if k != "_raw"},
        "unseen_p1_p2_pooled": {"acc": round(ku / nu, 4) if nu else None, "k": ku, "n": nu, "wilson_95ci": list(wilson_ci(ku, nu))},
        "unseen_routes": {
            "n": nr,
            "both_acc": round(sum(r_both) / nr, 4) if nr else None,
            "ordered_acc": round(sum(r_ord) / nr, 4) if nr else None,
            "either_acc": round(sum(r_eith) / nr, 4) if nr else None,
        },
        "locality_380": {
            "n_total": len(loc_facts),
            "n_base_correct": n_base_ok,
            "n_cur_correct": int(sum(loc_ok)),
            "newly_broken": newly_broken,
            "retention": round(ret, 4),
        },
        "_raw": {
            "p0": by_p["p0"]["_raw"],
            "unseen_p1_p2": unseen_hits,
            "routes_both": r_both,
        },
    }


def _train_ce_on_p0_targets(
    model_B: Any,
    tok_B: Any,
    device_B: torch.device,
    airports: Sequence[Dict[str, Any]],
    trainable_params: Sequence[nn.Parameter],
    steps: int = 25,
    lr: float = 3e-3,
    batch_size: int = 16,
    bank_hook: Optional[RoutedBankLayerHook] = None,
    seed: int = SEED + 71,
) -> Dict[str, Any]:
    """Train ONLY `trainable_params` on phrasing `p0` + Teacher-7B IATA labels (base weights frozen!)."""
    t0 = time.time()
    rng = random.Random(seed)
    # Build exact `(input_ids, labels)` pairs where loss is computed ONLY on `" {IATA}"` target tokens!
    examples: List[Tuple[List[int], List[int]]] = []
    for ap in airports:
        p = format_airport_prompt(ap["name"], 0)[0]
        p_ids = tok_B.encode(p, add_special_tokens=True)
        full_ids = tok_B.encode(p + " " + ap["iata"], add_special_tokens=True)
        if full_ids[: len(p_ids)] == p_ids and len(full_ids) > len(p_ids):
            c_ids = full_ids[len(p_ids) :]
        else:
            c_ids = tok_B.encode(" " + ap["iata"], add_special_tokens=False)
            full_ids = p_ids + c_ids
        labels = [-100] * len(p_ids) + list(c_ids)
        examples.append((full_ids, labels))

    pad_id = tok_B.pad_token_id if tok_B.pad_token_id is not None else tok_B.eos_token_id
    opt = torch.optim.AdamW(list(trainable_params), lr=lr, weight_decay=0.0)
    losses: List[float] = []

    for step in range(steps):
        batch = [examples[rng.randrange(len(examples))] for _ in range(min(batch_size, len(examples)))]
        max_len = max(len(x[0]) for x in batch)
        inp = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device_B)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long, device=device_B)
        lbl = torch.full((len(batch), max_len), -100, dtype=torch.long, device=device_B)
        for bi, (seq, lb) in enumerate(batch):
            inp[bi, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device_B)
            attn[bi, : len(seq)] = 1
            lbl[bi, : len(lb)] = torch.tensor(lb, dtype=torch.long, device=device_B)

        if bank_hook is not None:
            bank_hook.set_attention_mask(attn)
        opt.zero_grad(set_to_none=True)
        logits = model_B(input_ids=inp, attention_mask=attn, use_cache=False).logits.float()
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), lbl[:, 1:].reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()
        losses.append(float(loss.item()))

    if bank_hook is not None:
        bank_hook.set_attention_mask(None)
    return {
        "steps": steps,
        "lr": lr,
        "loss_start": round(losses[0], 4) if losses else None,
        "loss_end": round(losses[-1], 4) if losses else None,
        "elapsed_sec": round(time.time() - t0, 2),
    }


# =============================================================================
# 5. Full Gate 1 Pipeline (Sequential BF16: 7B -> Unload -> 1.7B 3-Tier Suite)
# =============================================================================


def run_gate1_airports(args: argparse.Namespace) -> Dict[str, Any]:
    import platonic_organ_engine as poe
    import v2_track1_dev_dataset as t1d
    from extract_and_baseline import check_answer_match

    t_all = time.time()
    airports, ds_source = fetch_or_fallback_airports(max_airports=args.max_screen, seed=SEED, offline=args.offline)
    donor_layers = sorted({la for la, _ in CANDIDATE_LAYER_PAIRS})
    recip_layers = sorted({lb for _, lb in CANDIDATE_LAYER_PAIRS})
    loc_facts = t1d.build_certified_locality_facts()
    ood_passages = poe.build_universal_calibration_corpus()["fit"][:100]

    print(f"\n================ V3 GATE 1: REAL LARGE -> SMALL KNOWLEDGE ENRICHMENT (OURAIRPORTS) ================")
    print(f"Donor: {DONOR_KEY} -> Recipient: {RECIPIENT_KEY} | Airports loaded: {len(airports)} ({ds_source})")

    # -------------------------------------------------------------------------
    # STAGE 1A: Screen Donor 7B (BF16) + Cache Donor States on 7B-3/3 Airports, THEN UNLOAD 7B!
    # -------------------------------------------------------------------------
    t1a = time.time()
    model_A, tok_A, dev_A, dt_A = l2s.load_model_bf16(DONOR_KEY, allow_download=args.allow_download)
    donor_hits = [0] * len(airports)
    for pidx in (0, 1, 2):
        prs = [format_airport_prompt(ap["name"], pidx)[0] for ap in airports]
        gens = greedy_generate_prompts(model_A, tok_A, dev_A, prs, max_new_tokens=6, batch_size=args.batch_size)
        for i, (g, ap) in enumerate(zip(gens, airports)):
            donor_hits[i] += int(check_iata_match(g, ap["iata"]))
    n_7b_3of3 = sum(1 for h in donor_hits if h == 3)
    print(f"  [Stage 1A | {DONOR_KEY}] 3/3 exact IATA: {n_7b_3of3}/{len(airports)} ({100.0 * n_7b_3of3 / max(len(airports), 1):.1f}%)")

    # Extract Donor 7B states ONCE on all 7B-3/3 airports (phrasing p0) BEFORE unloading 7B!
    idx_7b_ok = [i for i, h in enumerate(donor_hits) if h == 3]
    airports_7b_ok = [airports[i] for i in idx_7b_ok]
    donor_states_p0 = extract_multilayer_airport_states(
        model_A, tok_A, dev_A, airports_7b_ok, phrasing_idx=0, layer_indices=donor_layers, capture_mlp_in=False, batch_size=args.batch_size
    )
    # Map airport iata -> index inside donor_states_p0
    iata_to_didx = {ap["iata"]: idx for idx, ap in enumerate(airports_7b_ok)}

    l2s.unload_model_strict(model_A)
    del model_A
    gc.collect()
    print(f"  [Stage 1A Complete in {time.time() - t1a:.1f}s] Donor {DONOR_KEY} COMPLETELY UNLOADED!")

    # -------------------------------------------------------------------------
    # STAGE 1B: Screen Solo Recipient 1.7B (BF16) -> Partition Shared-Known vs Strict Blindspots
    # -------------------------------------------------------------------------
    model_B, tok_B, dev_B, dt_B = l2s.load_model_bf16(RECIPIENT_KEY, allow_download=args.allow_download)
    recip_hits_7b_ok = [0] * len(airports_7b_ok)
    for pidx in (0, 1, 2):
        prs = [format_airport_prompt(ap["name"], pidx)[0] for ap in airports_7b_ok]
        gens = greedy_generate_prompts(model_B, tok_B, dev_B, prs, max_new_tokens=6, batch_size=args.batch_size)
        for i, (g, ap) in enumerate(zip(gens, airports_7b_ok)):
            recip_hits_7b_ok[i] += int(check_iata_match(g, ap["iata"]))

    splits = partition_screened_airports(
        airports_7b_ok, [3] * len(airports_7b_ok), recip_hits_7b_ok,
        n_dev=args.n_dev, n_eval_max=args.n_eval, n_shared_max=200,
    )
    shared_cal = splits["shared_known"]
    dev_blind = splits["blindspot_dev"]
    eval_blind = splits["blindspot_eval"]
    route_items = build_route_items(eval_blind, max_routes=48)
    print(
        f"  [Stage 1B | {RECIPIENT_KEY}] Shared-Known (3/3 & 3/3): {len(shared_cal)} | "
        f"Strict Blindspots (7B 3/3 & 1.7B 0/3): dev={len(dev_blind)}, eval={len(eval_blind)} | Routes={len(route_items)}"
    )
    assert len(shared_cal) >= 16 and len(dev_blind) >= 4 and len(eval_blind) >= 8, "Insufficient qualified airports!"

    # Base 380-fact locality & matched-batch probe logits for torch.equal rollback verification
    loc_prompts = [f["direct_prompt"] for f in loc_facts]
    base_loc_gens = greedy_generate_prompts(model_B, tok_B, dev_B, loc_prompts, max_new_tokens=8, batch_size=args.batch_size)
    base_loc_ok = [int(check_answer_match(g, f["aliases"])) for g, f in zip(base_loc_gens, loc_facts)]
    print(f"  [Base 380-Fact Locality] {sum(base_loc_ok)}/{len(loc_facts)} base-correct")

    probe_prompts = [format_airport_prompt(ap["name"], 1)[0] for ap in eval_blind[:4]]
    enc_probe = tok_B(probe_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
    probe_ids = enc_probe["input_ids"].to(dev_B)
    probe_attn = enc_probe["attention_mask"].to(dev_B)
    with torch.no_grad():
        base_probe_logits = model_B(input_ids=probe_ids, attention_mask=probe_attn, use_cache=False).logits.detach().cpu()

    # -------------------------------------------------------------------------
    # STAGE 2: Gate 1a — Contrast Bridge Fit on Shared-Known + Dev Layer Sweep + Eval Test
    # -------------------------------------------------------------------------
    recip_cal_states = extract_multilayer_airport_states(model_B, tok_B, dev_B, shared_cal, 0, recip_layers, capture_mlp_in=False, batch_size=args.batch_size)
    recip_dev_states = extract_multilayer_airport_states(model_B, tok_B, dev_B, dev_blind, 0, recip_layers, capture_mlp_in=True, batch_size=args.batch_size)
    recip_eval_states = extract_multilayer_airport_states(model_B, tok_B, dev_B, eval_blind, 0, recip_layers, capture_mlp_in=True, batch_size=args.batch_size)

    def _gather_donor(aps: Sequence[Dict[str, Any]], lA: int, site: str) -> np.ndarray:
        key = f"h_{site}"
        idxs = [iata_to_didx[ap["iata"]] for ap in aps]
        return donor_states_p0[lA][key][idxs].astype(np.float64)

    dev_sweep: List[Dict[str, Any]] = []
    best_cfg: Optional[Dict[str, Any]] = None
    best_dev_acc = -1.0

    for lA, lB in CANDIDATE_LAYER_PAIRS:
        for site in ("airport_last", "prompt_last"):
            hA_cal = _gather_donor(shared_cal, lA, site)
            hB_cal = recip_cal_states[lB][f"h_{site}"].astype(np.float64)
            mu_A = hA_cal.mean(axis=0)
            mu_B = hB_cal.mean(axis=0)
            M, b_info = l2s.fit_contrast_bridge(hA_cal - mu_A, hB_cal - mu_B, mode="ridge")
            hA_dev = _gather_donor(dev_blind, lA, site)
            patched_dev = mu_B[None, :] + (hA_dev - mu_A[None, :]) @ M.T
            res_d = eval_patched_airports_gate1a(model_B, tok_B, dev_B, dev_blind, 0, lB, site, patched_dev, batch_size=args.batch_size)
            row = {"lA": lA, "lB": lB, "site": site, "bridge": b_info, "dev_acc": res_d["acc"], "dev_k": res_d["k"], "dev_n": res_d["n"]}
            dev_sweep.append(row)
            print(f"  [Gate1a Dev Sweep] (L{lA}->L{lB}, {site:12s}) bridge_cos={b_info['train_mean_cos']:.3f} -> dev EM={res_d['acc']:.3f} ({res_d['k']}/{res_d['n']})")
            if res_d["acc"] > best_dev_acc:
                best_dev_acc = res_d["acc"]
                best_cfg = {"lA": lA, "lB": lB, "site": site, "M": M, "mu_A": mu_A, "mu_B": mu_B, "bridge": b_info}

    assert best_cfg is not None
    lA_star, lB_star, site_star = best_cfg["lA"], best_cfg["lB"], best_cfg["site"]
    M_star, mu_A_star, mu_B_star = best_cfg["M"], best_cfg["mu_A"], best_cfg["mu_B"]

    # Held-out Gate 1a evaluation on eval_blind
    hA_eval = _gather_donor(eval_blind, lA_star, site_star)
    hB_eval = recip_eval_states[lB_star][f"h_{site_star}"].astype(np.float64)
    mapped_eval_h = mu_B_star[None, :] + (hA_eval - mu_A_star[None, :]) @ M_star.T
    perm_eval_h = np.roll(mapped_eval_h, shift=1, axis=0)

    g1a_mapped = eval_patched_airports_gate1a(model_B, tok_B, dev_B, eval_blind, 0, lB_star, site_star, mapped_eval_h, batch_size=args.batch_size)
    g1a_self = eval_patched_airports_gate1a(model_B, tok_B, dev_B, eval_blind, 0, lB_star, site_star, hB_eval, batch_size=args.batch_size)
    g1a_perm = eval_patched_airports_gate1a(model_B, tok_B, dev_B, eval_blind, 0, lB_star, site_star, perm_eval_h, batch_size=args.batch_size)
    g1a_margin = round(g1a_mapped["acc"] - max(g1a_self["acc"], g1a_perm["acc"]), 4)
    g1a_pass = bool(g1a_margin >= 0.20)
    print(
        f"  [GATE 1A HELD-OUT EVAL (L{lA_star}->L{lB_star}, {site_star})] "
        f"MAPPED={g1a_mapped['acc']:.3f} vs SELF={g1a_self['acc']:.3f}, PERM={g1a_perm['acc']:.3f} | "
        f"margin={g1a_margin:+.3f} -> {'PASS (>=+0.20)' if g1a_pass else 'FAIL (<+0.20)'}"
    )

    # -------------------------------------------------------------------------
    # STAGE 3: Gate 1b — Standalone Donor-Offline 3-Tier Head-to-Head
    # -------------------------------------------------------------------------
    # Collect 100 OOD sequences' hidden states & mlp.down_proj input covariance at lB_star
    ood_h_seqs: List[torch.Tensor] = []
    ood_z_rows: List[np.ndarray] = []
    layer_B = get_decoder_layers(model_B)[lB_star]
    buf_ood: Dict[str, torch.Tensor] = {}
    h_o1 = layer_B.register_forward_hook(lambda _m, _i, o: buf_ood.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    h_o2 = layer_B.mlp.down_proj.register_forward_pre_hook(lambda _m, inp: buf_ood.__setitem__("z", inp[0].detach()))
    try:
        with torch.no_grad():
            for txt in ood_passages:
                enc = tok_B(txt, return_tensors="pt", truncation=True, max_length=96, add_special_tokens=True)
                _ = model_B(input_ids=enc["input_ids"].to(dev_B), attention_mask=enc["attention_mask"].to(dev_B), use_cache=False)
                ood_h_seqs.append(buf_ood["h"].float().cpu())
                if buf_ood["z"].shape[1] > 1:
                    ood_z_rows.append(buf_ood["z"][0, 1:, :].float().cpu().numpy())
    finally:
        h_o1.remove(); h_o2.remove()
    Z_bg_all = np.vstack(ood_z_rows).astype(np.float64)
    C_bg = (Z_bg_all.T @ Z_bg_all) / max(len(Z_bg_all), 1)

    base_suite = eval_standalone_airport_suite(model_B, tok_B, dev_B, eval_blind, route_items, loc_facts, base_loc_ok, batch_size=args.batch_size)
    tier_results: Dict[str, Any] = {"BASE": {k: v for k, v in base_suite.items() if k != "_raw"}}
    tier_raw: Dict[str, Any] = {"BASE": base_suite["_raw"]}

    delta_Y_eval = mapped_eval_h - hB_eval  # (N_eval, D_B)

    # --- Tier 1A: Pure Closed-Form Reversible ΔW (ZERO backprop!) ---
    t_1a = time.time()
    Z_eval = recip_eval_states[lB_star][f"z_{site_star}"].astype(np.float64)
    dW_1a, dw_info = l2s.solve_closed_form_layer_delta_w(Z_eval, delta_Y_eval, Z_bg_cov=C_bg, lam_bg=5.0, lam_eye=1e-2)
    pname_1a = f"model.layers.{lB_star}.mlp.down_proj.weight"
    pack_1a = l2s.ReversibleDeltaPack({pname_1a: torch.tensor(dW_1a, dtype=torch.float32)}, metadata={"tier": "1A"})
    pack_1a.apply(model_B)
    res_1a = eval_standalone_airport_suite(model_B, tok_B, dev_B, eval_blind, route_items, loc_facts, base_loc_ok, batch_size=args.batch_size)
    pack_1a.revert(model_B)
    rb_1a = pack_1a.verify_bit_exact_rollback(model_B, probe_ids, probe_attn, base_probe_logits)
    tier_raw["TIER1A_CLOSED_FORM_DELTA_W"] = res_1a["_raw"]
    tier_results["TIER1A_CLOSED_FORM_DELTA_W"] = {
        **{k: v for k, v in res_1a.items() if k != "_raw"},
        "fit_info": dw_info,
        "param_bytes_bf16": pack_1a.n_bytes_bf16(),
        "build_sec": round(time.time() - t_1a, 2),
        "rollback": rb_1a,
    }
    print(f"  [TIER 1A CLOSED_FORM_ΔW ] p0={res_1a['train_phrasing_p0']['acc']:.3f} unseen={res_1a['unseen_p1_p2_pooled']['acc']:.3f} routes={res_1a['unseen_routes']['both_acc']:.3f} loc={res_1a['locality_380']['newly_broken']}/{res_1a['locality_380']['n_base_correct']} rb={rb_1a['logits_bit_equal']}")

    # --- Tier 1B: Hierarchical Routed Fact Bank (ZERO backprop!) ---
    # Key on airport_last_token so it triggers on ALL phrasings (p0, p1, p2) AND 2-endpoint routes!
    t_1b = time.time()
    hB_eval_al = recip_eval_states[lB_star]["h_airport_last"].astype(np.float64)
    hA_eval_al = _gather_donor(eval_blind, lA_star, "airport_last")
    hA_cal_al = _gather_donor(shared_cal, lA_star, "airport_last")
    hB_cal_al = recip_cal_states[lB_star]["h_airport_last"].astype(np.float64)
    M_al, _ = l2s.fit_contrast_bridge(hA_cal_al - hA_cal_al.mean(0), hB_cal_al - hB_cal_al.mean(0), mode="ridge")
    V_init_1b = (hB_cal_al.mean(0)[None, :] + (hA_eval_al - hA_cal_al.mean(0)[None, :]) @ M_al.T) - hB_eval_al

    bank_1b = l2s.HierarchicalRoutedFactBank.from_keys_and_writes(hB_eval_al, V_init_1b, d_key=64, tau_abs=0.80, margin_min=0.02).to(dev_B)
    cal_1b = bank_1b.calibrate_joint_max_threshold(ood_h_seqs, alpha_seq=0.005, margin_safety=0.01)
    bank_1b.tau_abs = min(float(bank_1b.tau_abs), 0.995)
    cal_1b["calibrated_tau_abs"] = round(bank_1b.tau_abs, 4)
    hook_1b = RoutedBankLayerHook(model_B, lB_star, bank_1b).attach()
    res_1b = eval_standalone_airport_suite(model_B, tok_B, dev_B, eval_blind, route_items, loc_facts, base_loc_ok, bank_hook=hook_1b, batch_size=args.batch_size)
    # Verify hard bypass bit-exactness!
    bank_1b.enabled = False
    with torch.no_grad():
        bypass_logits_1b = model_B(input_ids=probe_ids, attention_mask=probe_attn, use_cache=False).logits.detach().cpu()
    rb_1b = {"logits_bit_equal": bool(torch.equal(bypass_logits_1b, base_probe_logits)), "max_abs_diff": float((bypass_logits_1b.float() - base_probe_logits.float()).abs().max().item())}
    hook_1b.detach()
    tier_raw["TIER1B_ROUTED_FACT_BANK_ZERO_SGD"] = res_1b["_raw"]
    tier_results["TIER1B_ROUTED_FACT_BANK_ZERO_SGD"] = {
        **{k: v for k, v in res_1b.items() if k != "_raw"},
        "calibration": cal_1b,
        "param_bytes_bf16": bank_1b.n_bytes_bf16(),
        "build_sec": round(time.time() - t_1b, 2),
        "rollback": rb_1b,
    }
    print(f"  [TIER 1B ROUTED_BANK_0S ] p0={res_1b['train_phrasing_p0']['acc']:.3f} unseen={res_1b['unseen_p1_p2_pooled']['acc']:.3f} routes={res_1b['unseen_routes']['both_acc']:.3f} loc={res_1b['locality_380']['newly_broken']}/{res_1b['locality_380']['n_base_correct']} tau={cal_1b['calibrated_tau_abs']} rb={rb_1b['logits_bit_equal']}")

    # --- Tier 2: Routed Fact Bank + Tiny Polish (<= 25 frozen-base steps on bank.V) + Recipient-Only Ctrl ---
    for p in model_B.parameters():
        p.requires_grad_(False)

    for t2_name, v_init_mat in [
        ("TIER2_ROUTED_BANK_TINY_POLISH", V_init_1b),
        ("TIER2_RECIPIENT_ONLY_CTRL", np.random.default_rng(SEED + 999).normal(0.0, float(np.std(V_init_1b)), size=V_init_1b.shape)),
    ]:
        bank_t2 = l2s.HierarchicalRoutedFactBank.from_keys_and_writes(hB_eval_al, v_init_mat, d_key=64, tau_abs=cal_1b["calibrated_tau_abs"], margin_min=0.02).to(dev_B)
        bank_t2.V.requires_grad_(True)
        hook_t2 = RoutedBankLayerHook(model_B, lB_star, bank_t2).attach()
        tr_info = _train_ce_on_p0_targets(model_B, tok_B, dev_B, eval_blind, [bank_t2.V], steps=args.tier2_steps, lr=5e-2, batch_size=16, bank_hook=hook_t2)
        bank_t2.V.requires_grad_(False)
        res_t2 = eval_standalone_airport_suite(model_B, tok_B, dev_B, eval_blind, route_items, loc_facts, base_loc_ok, bank_hook=hook_t2, batch_size=args.batch_size)
        bank_t2.enabled = False
        with torch.no_grad():
            bl_t2 = model_B(input_ids=probe_ids, attention_mask=probe_attn, use_cache=False).logits.detach().cpu()
        rb_t2 = {"logits_bit_equal": bool(torch.equal(bl_t2, base_probe_logits)), "max_abs_diff": float((bl_t2.float() - base_probe_logits.float()).abs().max().item())}
        hook_t2.detach()
        tier_raw[t2_name] = res_t2["_raw"]
        tier_results[t2_name] = {
            **{k: v for k, v in res_t2.items() if k != "_raw"},
            "train_info": tr_info,
            "param_bytes_bf16": bank_t2.n_bytes_bf16(),
            "rollback": rb_t2,
        }
        print(f"  [{t2_name:23s}] p0={res_t2['train_phrasing_p0']['acc']:.3f} unseen={res_t2['unseen_p1_p2_pooled']['acc']:.3f} routes={res_t2['unseen_routes']['both_acc']:.3f} loc={res_t2['locality_380']['newly_broken']}/{res_t2['locality_380']['n_base_correct']} sec={tr_info['elapsed_sec']}s rb={rb_t2['logits_bit_equal']}")

    # --- Tier 3: Fair Rank-16 Teacher-LoRA Baseline (`q_proj, v_proj, down_proj`) ---
    lora_layers = list(range(max(0, lB_star - 3), min(len(get_decoder_layers(model_B)), lB_star + 4)))
    attached_lora = attach_lora_adapters(model_B, lora_layers, target_modules=("q_proj", "v_proj", "down_proj"), rank=16, alpha=32.0)
    lora_params = [p for w in attached_lora.values() for p in (w.lora_A, w.lora_B)]
    tr_lora = _train_ce_on_p0_targets(model_B, tok_B, dev_B, eval_blind, lora_params, steps=args.lora_steps, lr=3e-4, batch_size=16)
    pack_lora, lora_bytes = detach_lora_to_delta_pack(model_B, attached_lora, lora_layers)
    pack_lora.apply(model_B)
    res_t3 = eval_standalone_airport_suite(model_B, tok_B, dev_B, eval_blind, route_items, loc_facts, base_loc_ok, batch_size=args.batch_size)
    pack_lora.revert(model_B)
    rb_t3 = pack_lora.verify_bit_exact_rollback(model_B, probe_ids, probe_attn, base_probe_logits)
    tier_raw["TIER3_TEACHER_LORA_R16"] = res_t3["_raw"]
    tier_results["TIER3_TEACHER_LORA_R16"] = {
        **{k: v for k, v in res_t3.items() if k != "_raw"},
        "lora_layers": lora_layers,
        "train_info": tr_lora,
        "param_bytes_bf16": lora_bytes,
        "rollback": rb_t3,
    }
    print(f"  [TIER 3 TEACHER_LORA_R16] p0={res_t3['train_phrasing_p0']['acc']:.3f} unseen={res_t3['unseen_p1_p2_pooled']['acc']:.3f} routes={res_t3['unseen_routes']['both_acc']:.3f} loc={res_t3['locality_380']['newly_broken']}/{res_t3['locality_380']['n_base_correct']} sec={tr_lora['elapsed_sec']}s rb={rb_t3['logits_bit_equal']}")

    # Exact McNemar comparisons
    mcn = {
        "gate1a_mapped_vs_self": exact_mcnemar(g1a_mapped["_raw"], g1a_self["_raw"]),
        "gate1a_mapped_vs_perm": exact_mcnemar(g1a_mapped["_raw"], g1a_perm["_raw"]),
        "tier2_vs_recipient_only_unseen": exact_mcnemar(tier_raw["TIER2_ROUTED_BANK_TINY_POLISH"]["unseen_p1_p2"], tier_raw["TIER2_RECIPIENT_ONLY_CTRL"]["unseen_p1_p2"]),
        "tier2_vs_lora_unseen": exact_mcnemar(tier_raw["TIER2_ROUTED_BANK_TINY_POLISH"]["unseen_p1_p2"], tier_raw["TIER3_TEACHER_LORA_R16"]["unseen_p1_p2"]),
        "tier1b_vs_base_unseen": exact_mcnemar(tier_raw["TIER1B_ROUTED_FACT_BANK_ZERO_SGD"]["unseen_p1_p2"], tier_raw["BASE"]["unseen_p1_p2"]),
    }

    l2s.unload_model_strict(model_B)
    out = {
        "meta": {
            "donor_key": DONOR_KEY,
            "recipient_key": RECIPIENT_KEY,
            "dataset_source": ds_source,
            "n_screened": len(airports),
            "n_7b_3of3": n_7b_3of3,
            "n_shared_known": len(shared_cal),
            "n_dev_blindspot": len(dev_blind),
            "n_eval_blindspot": len(eval_blind),
            "n_routes": len(route_items),
            "best_site": {"lA": lA_star, "lB": lB_star, "site": site_star, "bridge": best_cfg["bridge"]},
            "seed": SEED,
            "total_elapsed_sec": round(time.time() - t_all, 1),
        },
        "stage2_gate1a": {
            "dev_sweep": dev_sweep,
            "eval_mapped": {k: v for k, v in g1a_mapped.items() if k != "_raw"},
            "eval_self": {k: v for k, v in g1a_self.items() if k != "_raw"},
            "eval_perm": {k: v for k, v in g1a_perm.items() if k != "_raw"},
            "margin_vs_best_ctrl": g1a_margin,
            "gate1a_pass_ge_20pp": g1a_pass,
        },
        "stage3_gate1b_tiers": tier_results,
        "mcnemar": mcn,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[V3 GATE 1 COMPLETE] Saved {OUT_PATH} ({out['meta']['total_elapsed_sec']}s)")
    return out


# =============================================================================
# 6. Pure-CPU Unit Tests (<5s, ZERO GPU!)
# =============================================================================


def run_unit_test() -> None:
    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

    print("[v3_gate1_airports --unit-test] Running pure-CPU verification (160 airports, regex matcher, disjoint splits, Tier 1A/1B/2/3 on toy LlamaForCausalLM)...")

    # 1. Verify embedded offline fallback airports (>= 150, 0 duplicates, 0 IATA-in-name / prefix leaks!)
    airports, src = fetch_or_fallback_airports(max_airports=200, seed=SEED, offline=True, exclude_prefix_leak=True)
    assert src == "embedded_fallback_160"
    assert len(airports) >= 150, f"Expected >= 150 clean embedded airports, got {len(airports)} (raw list={len(EMBEDDED_AIRPORTS_160)})"
    iatas = [a["iata"] for a in airports]
    names = [a["name"].lower() for a in airports]
    assert len(set(iatas)) == len(iatas), "Duplicate IATA codes in fallback!"
    assert len(set(names)) == len(names), "Duplicate airport names in fallback!"
    for a in airports:
        assert _is_clean_airport(a["name"], a["iata"], exclude_prefix_leak=True)
        for pidx in (0, 1, 2):
            p, (c0, c1) = format_airport_prompt(a["name"], pidx)
            assert p[c0:c1] == a["name"] and a["iata"] not in p

    # 2. Verify exact regex IATA & 2-endpoint route matchers
    assert extract_iata_code(" LHR.") == "LHR"
    assert extract_iata_code(' "NRT" (Narita)') == "NRT"
    assert extract_iata_code("The code is ORD, located in Chicago.") == "ORD"
    assert extract_iata_code("no code here 123") is None
    assert check_iata_match(" YYZ\n", "YYZ") and not check_iata_match(" YUL\n", "YYZ")
    rm = check_route_match(" LHR and NRT.", "LHR", "NRT")
    assert rm["both"] and rm["ordered"] and rm["either"]
    rm_rev = check_route_match(" NRT to LHR.", "LHR", "NRT")
    assert rm_rev["both"] and not rm_rev["ordered"]

    # 3. Verify strict disjoint split logic & route builder
    dh = [3] * 140 + [1] * (len(airports) - 140)
    rh = [3] * 40 + [0] * 100 + [0] * (len(airports) - 140)
    sp = partition_screened_airports(airports, dh, rh, n_dev=24, n_eval_max=76, n_shared_max=40)
    s_set = {x["iata"] for x in sp["shared_known"]}
    d_set = {x["iata"] for x in sp["blindspot_dev"]}
    e_set = {x["iata"] for x in sp["blindspot_eval"]}
    assert len(s_set) == 40 and len(d_set) == 24 and len(e_set) == 76
    assert s_set.isdisjoint(d_set) and s_set.isdisjoint(e_set) and d_set.isdisjoint(e_set)
    routes = build_route_items(sp["blindspot_eval"], max_routes=48)
    assert len(routes) == 48 and all(r["iata_a"] != r["iata_b"] for r in routes)

    # 4. Synthetic end-to-end CPU test on tiny HF LlamaForCausalLM (Tier 1A + Tier 1B + Tier 2 + Tier 3 + torch.equal!)
    torch.manual_seed(SEED)
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
        hidden_act="silu", mlp_bias=False,
    )
    toy = LlamaForCausalLM(cfg).eval()
    probe_ids = torch.randint(5, 120, (3, 10))
    probe_attn = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    vmask = build_valid_token_mask_from_attention(probe_attn)
    # Verify left-padding AND BOS (first active token) are strictly masked False!
    assert not vmask[0, 0] and not vmask[0, 1] and not vmask[0, 2] and bool(vmask[0, 3])
    assert not vmask[2, 0] and bool(vmask[2, 1])

    with torch.no_grad():
        base_logits = toy(input_ids=probe_ids, attention_mask=probe_attn, use_cache=False).logits.detach().cpu()

    # Tier 1A: Closed-form ΔW + ReversibleDeltaPack bit-exact rollback
    rng = np.random.default_rng(SEED + 5)
    dW, info_dw = l2s.solve_closed_form_layer_delta_w(rng.standard_normal((16, 64)), rng.standard_normal((16, 32)), Z_bg_cov=np.eye(64), lam_bg=0.1)
    assert info_dw["fit_cos"] > 0.90
    p1a = l2s.ReversibleDeltaPack({"model.layers.2.mlp.down_proj.weight": torch.tensor(dW, dtype=torch.float32)})
    p1a.apply(toy)
    with torch.no_grad():
        ed_logits = toy(input_ids=probe_ids, attention_mask=probe_attn, use_cache=False).logits.detach().cpu()
    assert not torch.equal(ed_logits, base_logits)
    p1a.revert(toy)
    rb1a = p1a.verify_bit_exact_rollback(toy, probe_ids, probe_attn, base_logits)
    assert rb1a["logits_bit_equal"] and rb1a["max_abs_diff"] == 0.0

    # Tier 1B + Tier 2: HierarchicalRoutedFactBank hook + valid_token_mask + hard bypass torch.equal + tiny polish!
    K_syn = rng.standard_normal((24, 32))
    V_syn = rng.standard_normal((24, 32)) * 0.5
    bank = l2s.HierarchicalRoutedFactBank.from_keys_and_writes(K_syn, V_syn, d_key=16, n_groups=4, tau_abs=0.50, margin_min=0.01)
    cal = bank.calibrate_joint_max_threshold([torch.randn(1, 8, 32) for _ in range(15)], alpha_seq=0.0, margin_safety=0.01)
    assert cal["n_ood_seqs"] == 15
    hook = RoutedBankLayerHook(toy, 2, bank).attach()
    bank.enabled = False
    hook.set_attention_mask(probe_attn)
    with torch.no_grad():
        byp_logits = toy(input_ids=probe_ids, attention_mask=probe_attn, use_cache=False).logits.detach().cpu()
    assert torch.equal(byp_logits, base_logits), "Hard bypass NOT bit-identical!"
    # Tiny polish step on bank.V with base frozen
    for p in toy.parameters():
        p.requires_grad_(False)
    bank.enabled = True
    bank.tau_abs = -1.0  # force fire on synthetic probe to verify grad flow into bank.V
    bank.margin_min = -1.0
    bank.V.requires_grad_(True)
    opt = torch.optim.AdamW([bank.V], lr=1e-2)
    opt.zero_grad()
    out_l = toy(input_ids=probe_ids, attention_mask=probe_attn, use_cache=False).logits
    loss = F.cross_entropy(out_l[:, :-1, :].reshape(-1, 128), probe_ids[:, 1:].reshape(-1))
    loss.backward()
    assert bank.V.grad is not None and float(bank.V.grad.abs().sum().item()) > 0.0
    opt.step()
    hook.detach()

    # Tier 3: Zero-dependency Reversible Rank-4 LoRA (`q_proj, v_proj, down_proj`) + rollback torch.equal!
    att_l = attach_lora_adapters(toy, [1, 2], target_modules=("q_proj", "v_proj", "down_proj"), rank=4, alpha=8.0)
    assert len(att_l) == 6
    l_params = [p for w in att_l.values() for p in (w.lora_A, w.lora_B)]
    opt_l = torch.optim.AdamW(l_params, lr=1e-2)
    opt_l.zero_grad()
    loss_l = F.cross_entropy(toy(input_ids=probe_ids, attention_mask=probe_attn, use_cache=False).logits[:, :-1, :].reshape(-1, 128), probe_ids[:, 1:].reshape(-1))
    loss_l.backward()
    opt_l.step()
    pack_l, l_bytes = detach_lora_to_delta_pack(toy, att_l, [1, 2])
    assert l_bytes > 0 and len(pack_l.deltas) == 6
    pack_l.apply(toy)
    with torch.no_grad():
        lora_logits = toy(input_ids=probe_ids, attention_mask=probe_attn, use_cache=False).logits.detach().cpu()
    assert not torch.equal(lora_logits, base_logits)
    pack_l.revert(toy)
    rb_l = pack_l.verify_bit_exact_rollback(toy, probe_ids, probe_attn, base_logits)
    assert rb_l["logits_bit_equal"] and rb_l["max_abs_diff"] == 0.0

    print(f"[PASS 1/4] Offline fallback verified: {len(airports)} real worldwide airports, 0 duplicates, 0 IATA-in-name or prefix leaks, 3 clean phrasings!")
    print(f"[PASS 2/4] Exact regex IATA & 2-endpoint route matchers verified!")
    print(f"[PASS 3/4] Disjoint screening partition verified: 40 shared_known / 24 dev / 76 eval (0 overlap) + 48 unseen 2-endpoint routes!")
    print(f"[PASS 4/4] Synthetic CPU LlamaForCausalLM integration PASS: Tier 1A ΔW + Tier 1B/2 RoutedBank (mask-safe + grad flow) + Tier 3 LoRA all torch.equal rollback!")


def main() -> None:
    parser = argparse.ArgumentParser(description="v3 Gate 1: Real Large -> Small Knowledge Enrichment (OurAirports IATA)")
    parser.add_argument("--unit-test", action="store_true")
    parser.add_argument("--max-screen", type=int, default=1000)
    parser.add_argument("--n-dev", type=int, default=24)
    parser.add_argument("--n-eval", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tier2-steps", type=int, default=25)
    parser.add_argument("--lora-steps", type=int, default=45)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if args.unit_test:
        run_unit_test()
        return
    run_gate1_airports(args)


if __name__ == "__main__":
    main()
